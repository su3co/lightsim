//! Persistent S-matrix gradient session.
//!
//! `S[m, n, k]` is the complex amplitude of output channel `m` (a mode
//! leaving the device through a port) per unit excitation `n` at frequency
//! `k`. The session
//!
//! 1. is set up once: the scene is rasterised, the sources and monitors of
//!    the N forward and M adjoint runs are compiled and uploaded, and every
//!    device gets one persistent engine;
//! 2. [`set_design`](GradientSession::set_design) evaluates the index-grid
//!    samples on the Yee voxels of the design box only and uploads that
//!    sub-block of the update coefficients;
//! 3. [`forward`](GradientSession::forward) runs the excitations (one at a
//!    time per device) and returns S; the design-region DFT fields stay on
//!    the device;
//! 4. [`backward`](GradientSession::backward) runs the adjoint simulations
//!    that the upstream gradient needs, forms
//!    `Re[conj(g) (i w eps0 / 4) E_adj E_fwd] dV` on the device and applies
//!    the transpose of the rasterisation.
//!
//! # Gradient convention
//!
//! The upstream gradient of a real loss L with respect to a complex S entry
//! is `g = dL/dRe(S) + i dL/dIm(S)` (so that `dL = Re(conj(g) dS)`), the
//! convention of `lightsim.invdes` and of `torch.Tensor.grad`.
//!
//! The normalisation of S, the placement of the monitors and the adjoint
//! formula are those of `lightsim.invdes.SMatrixProblem`, so the two paths
//! agree to rounding.

use crate::Real;
use super::design::{DesignGrid, DesignMap, IndexBox, voxel_volume};
use super::engine::{Engine, EngineFactory, IoSet, RunInfo, run_to_end};
use crate::constants::{C0, EPS0, TAU};
use crate::engine::RawResults;
use crate::geometry::Structure;
use crate::grid::{Component, Grid};
use crate::monitors::{Apodization, Monitor};
use crate::plan::{CompileOptions, Plan, compile_with_grid};
use crate::results::{assemble, finalize};
use crate::sources::{Direction, ModeProfile, Source, TimeSignal};
use crate::spec::Simulation;
use anyhow::{Result, anyhow, bail};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

/// A waveguide port: a mode plane and the direction pointing into the device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortSpec {
    #[serde(default)]
    pub name: String,
    pub axis: usize,
    pub position: f64,
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub profiles: Vec<ModeProfile>,
    pub direction: Direction,
}

/// An excitation or an output channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Channel {
    /// Mode `mode` of port `port` (index into [`SessionSpec::ports`]).
    Port {
        port: usize,
        #[serde(default)]
        mode: usize,
    },
    /// Any lightsim source (excitations only). S is normalised by the
    /// spectrum of the source signal times `norm` (`[re, im]` per frequency).
    Source {
        #[serde(default)]
        name: String,
        source: Source,
        #[serde(default)]
        norm: Option<Vec<[f64; 2]>>,
    },
}

fn d_normalize() -> String {
    "monitor".into()
}
fn d_components() -> Vec<Component> {
    Component::E.to_vec()
}

/// Everything a session is built from (what `lightsim.autodiff` serialises).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionSpec {
    /// Base simulation: its media must contain the design media; its
    /// sources and monitors are ignored.
    pub sim: Simulation,
    pub grids: Vec<DesignGrid>,
    #[serde(default)]
    pub ports: Vec<PortSpec>,
    pub inputs: Vec<Channel>,
    pub outputs: Vec<Channel>,
    pub freqs: Vec<f64>,
    /// Signal of the mode sources (forward and adjoint).
    pub signal: TimeSignal,
    /// `"monitor"`: the launched mode amplitude is measured by two mode
    /// monitors just downstream of the source and referred to the source
    /// plane; `"source"`: divide by the source spectrum only.
    #[serde(default = "d_normalize")]
    pub normalize: String,
    #[serde(default = "d_components")]
    pub components: Vec<Component>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Forward,
    Adjoint,
}

struct Run {
    kind: Kind,
    index: usize,
    io: usize,
    /// (port, mode) of a mode source.
    src_port: Option<(usize, usize)>,
    ex_norm: Option<Vec<Complex64>>,
    signal: TimeSignal,
    /// Index of the first port monitor in the run's DFT monitor list, and
    /// their names in order.
    first_port_monitor: usize,
    names: Vec<String>,
    /// Output ports with their monitor name (forward runs).
    out_names: Vec<(usize, String)>,
    /// Light plan (no coefficient arrays) holding the port monitors, for
    /// the post-processing of the mode overlaps.
    meta: Plan,
}

#[derive(Clone, Debug)]
struct RunResult {
    norm: Vec<Complex64>,
    /// `[output][freq]` (forward runs).
    amps: Vec<Vec<Complex64>>,
    info: RunInfo,
    engine: usize,
    /// Design-region DFT fields are on the primary engine.
    on_primary: bool,
}

/// One design-region DFT box: grid `g`, component `comp`, global index box
/// and voxel volumes.
#[derive(Clone, Debug)]
pub struct DesignBox {
    pub comp: Component,
    pub bx: IndexBox,
    pub vol: Vec<f64>,
}

/// Per-run statistics of the last dispatch.
#[derive(Clone, Debug, Default)]
pub struct RunStat {
    pub kind: &'static str,
    pub index: usize,
    pub device: String,
    pub steps: usize,
    pub seconds: f64,
    pub wall: f64,
}

pub struct GradientSession {
    spec: SessionSpec,
    grid: Grid,
    periodic: [bool; 3],
    cells: usize,
    design: DesignMap,
    master: Plan,
    engines: Vec<Mutex<Box<dyn Engine>>>,
    engine_names: Vec<String>,
    runs: Vec<Run>,
    results: Vec<Option<RunResult>>,
    /// `boxes[g][c]` for grid g and component index c of `spec.components`.
    boxes: Vec<Vec<DesignBox>>,
    design_set: bool,
    /// Sampling stride of the frequency-domain monitors (the largest over the runs).
    dft_stride: usize,
    /// Statistics of the runs since the last `set_design`.
    pub stats: Vec<RunStat>,
    pub setup_seconds: f64,
}

/// `numpy.clip` followed by Python indexing (negative indices wrap), as in
/// `SMatrixProblem._monitor_box`.
fn py_index(nodes: &[f64], i: isize) -> f64 {
    let n = nodes.len() as isize;
    nodes[(if i < 0 { i + n } else { i }) as usize]
}

fn clip(x: isize, lo: isize, hi: isize) -> isize {
    x.max(lo).min(hi)
}

fn searchsorted(nodes: &[f64], x: f64) -> isize {
    nodes.partition_point(|v| *v < x) as isize
}

/// Design box grown by more than a cell so that every owned Yee position is
/// recorded whatever the monitor's inclusion rule.
fn monitor_box(grid: &Grid, g: &DesignGrid) -> ([f64; 3], [f64; 3]) {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for a in 0..3 {
        let n = grid.edges(a);
        let len = n.len() as isize;
        let i0 = clip(searchsorted(n, g.min[a]), 2, len - 1);
        let i1 = clip(searchsorted(n, g.max[a]), 0, len - 3);
        lo[a] = py_index(n, i0 - 2).max(n[0]);
        hi[a] = py_index(n, i1 + 2).min(n[n.len() - 1]);
    }
    (lo, hi)
}

/// Two planes just downstream of a mode source (2 and 4 cells).
pub(crate) fn norm_planes(grid: &Grid, port: &PortSpec) -> Result<Vec<(f64, [f64; 3], [f64; 3])>> {
    let n = grid.edges(port.axis);
    let mut i0 = 0;
    for (i, v) in n.iter().enumerate() {
        if (v - port.position).abs() < (n[i0] - port.position).abs() {
            i0 = i;
        }
    }
    let s: isize = if port.direction == Direction::Plus { 1 } else { -1 };
    let mut out = vec![];
    for d in [2isize, 4] {
        let i = i0 as isize + s * d;
        if i < 0 || i as usize >= n.len() {
            bail!("port '{}' is too close to the domain boundary for the normalisation monitors", port.name);
        }
        let x = n[i as usize];
        let (mut mn, mut mx) = (port.min, port.max);
        mn[port.axis] = x;
        mx[port.axis] = x;
        out.push((x, mn, mx));
    }
    Ok(out)
}

pub(crate) fn named(mut s: Source, new: &str) -> Source {
    match &mut s {
        Source::PointDipole { name, .. } | Source::PlaneWave { name, .. } | Source::GaussianBeam { name, .. } | Source::Mode { name, .. } | Source::Tfsf { name, .. } => {
            *name = new.to_string()
        }
    }
    s
}

/// How the plans of gradient runs are compiled: every monitor keeps its own
/// accumulators. The engines form the adjoint product from the design-region
/// fields of a monitor directly on the device, and accumulate all monitors of
/// a run with one phasor table, so a box may not be left to another monitor.
pub(crate) fn session_options() -> CompileOptions {
    CompileOptions { share_monitors: false, ..Default::default() }
}

/// The gradient engine a scene is checked for (see [`check_media`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GradientEngine {
    /// The S-matrix session (`GradientSession`; engines "native" and, on the
    /// Python side, "python" of `lightsim.invdes`).
    Session,
    /// Reverse mode through the time loop (`ReverseSession`).
    Reverse,
    /// The reverse-mode S-matrix engine (`ReverseSMatrix`, engine "reverse").
    ReverseSMatrix,
}

/// Refuse the media a gradient engine cannot differentiate through, with an
/// error that names the medium and, where there is one, the engine to use.
/// Checked on the scene with its design grids (every medium that fills a
/// cell: the background and those of the structures).
///
/// - Raman (delayed chi3) and four-level gain media: nonlinear with their own
///   time-dependent state; the S-matrix adjoint needs a linear medium and
///   reverse mode does not step these models backwards. No engine.
/// - Gyrotropic (magneto-optic) media: the permittivity tensor is not
///   symmetric, so the medium is not reciprocal and the adjoint run of the
///   S-matrix engines is not an ordinary run (it would need the bias
///   reversed); reverse mode does not step the tensor update backwards.
///   No engine.
/// - Director fields: the tensor is symmetric at every point, but the
///   discrete off-diagonal update takes it at each component's own position,
///   so where the director varies from cell to cell the discrete scheme is
///   not symmetric and the session's adjoint run is not its transpose
///   (measured next to a slab: 1.1 % against finite differences for a
///   uniform director, 1.7 and 1.9 % for 90 and 180 degree twists, a factor
///   of 33 for a full turn in three cells); reverse mode does not step the
///   tensor update backwards. Refused by all engines; a uniform director is
///   an off-diagonal medium (`eps_off`), which the session handles.
pub(crate) fn check_media(sim: &Simulation, engine: GradientEngine) -> Result<()> {
    let used = |m: &crate::materials::Medium| m.name == sim.background || sim.structures.iter().any(|st| st.medium == m.name);
    let label = match engine {
        GradientEngine::Session => "the S-matrix session (engine \"native\" / \"python\")",
        GradientEngine::Reverse => "reverse mode",
        GradientEngine::ReverseSMatrix => "engine \"reverse\"",
    };
    if sim.fixed_angle {
        bail!("this simulation is a fixed-angle run (fixed_angle = true: broadband oblique incidence), which no gradient engine models ({label} refuses it): its update solves for transformed fields with implicit line solves along the tilt axis, which the adjoint and reverse passes do not include. Use Bloch boundaries with a fixed transverse wavevector for gradients at oblique incidence");
    if let Some(sh) = sim.sheets.first() {
        bail!("sheet '{}': zero-thickness sheets are not modelled by any gradient engine ({label} refuses them): their surface polarization is state that neither the S-matrix adjoint nor reverse mode carries. Represent the sheet as a one-cell layer (lightsim.sheets with a thickness) next to a design region instead", sh.name);
    }
    for m in sim.media.iter().filter(|m| used(m)) {
        if let Some(pl) = &m.plugin {
            bail!("medium '{}' is a user-defined material plug-in ('{}'), which no gradient engine models ({label} refuses it): the engines cannot differentiate through code they do not know", m.name, pl.name);
        }
        if m.raman.is_some() {
            bail!("medium '{}' has a Raman (delayed chi3) response, which no gradient engine models ({label} refuses it): the S-matrix adjoint needs a linear medium and reverse mode does not step the Raman oscillator backwards. An instantaneous chi3 alone is supported by reverse mode", m.name);
        }
        if m.eps_complex.is_some() {
            bail!("medium '{}' has a complex (Hermitian) permittivity tensor, which no gradient engine models ({label} refuses it): it runs with complex fields and is not reciprocal, so the adjoint run would need the transposed tensor", m.name);
        }
        if m.gain2.is_some() {
            bail!("medium '{}' is a four-level two-electron gain medium, which no gradient engine models ({label} refuses it): it is nonlinear (saturable) with population state that the S-matrix adjoint and reverse mode do not differentiate", m.name);
        }
        if m.gain.is_some() {
            bail!("medium '{}' is a four-level gain medium, which no gradient engine models ({label} refuses it): it is nonlinear (saturable) with population state that the S-matrix adjoint and reverse mode do not differentiate", m.name);
        }
        if !m.gyro.is_empty() {
            bail!("medium '{}' is gyrotropic (magneto-optic), which no gradient engine models ({label} refuses it): its permittivity tensor is not symmetric, so the medium is not reciprocal and the adjoint run of the S-matrix engines would need the bias reversed; reverse mode does not step the tensor update backwards", m.name);
        }
        if m.director.is_some() {
            bail!("medium '{}' has a director field, which no gradient engine models ({label} refuses it): where the director varies from cell to cell the discrete off-diagonal update is not symmetric, so the S-matrix adjoint is not its transpose, and reverse mode does not step the tensor update backwards. A uniform director is an off-diagonal medium: give it as eps_off / sigma_off (Medium.rotated), which the S-matrix session (engine \"native\" or \"python\") handles", m.name);
        }
    }
    Ok(())
}

/// A copy of `p` without the per-cell arrays (enough for result assembly).
pub(crate) fn light_plan(p: &Plan) -> Plan {
    Plan {
        name: p.name.clone(),
        grid: p.grid.clone(),
        n: p.n,
        x0: p.x0,
        nxl: p.nxl,
        dt: p.dt,
        nsteps: p.nsteps,
        inv_p: p.inv_p.clone(),
        inv_d: p.inv_d.clone(),
        cb: [vec![], vec![], vec![]],
        ca: None,
        ch: None,
        ch_scalar: p.ch_scalar,
        hloss: None,
        bc_y: p.bc_y,
        bc_z: p.bc_z,
        ghost_e_hi: p.ghost_e_hi.clone(),
        ghost_h_lo: p.ghost_h_lo.clone(),
        bloch: p.bloch,
        bloch_phase: p.bloch_phase,
        periodic: p.periodic,
        pml: vec![],
        disp: None,
        nonlin: None,
        tensor: None,
        inj_h: vec![],
        inj_e: vec![],
        dft: vec![],
        time: vec![],
        energy: vec![],
        source_analytic: p.source_analytic.clone(),
        media: p.media.clone(),
        shutoff: p.shutoff,
        shutoff_interval: p.shutoff_interval,
        source_end_time: p.source_end_time,
        sources: p.sources.clone(),
        twod: p.twod,
        slice: p.slice,
        sheet: None,
        plugin: None,
        gain2: None,
        herm: None,
    }
}

/// The base simulation with the design grids appended (constant fill).
pub(crate) fn base_simulation(sim: &Simulation, grids: &[DesignGrid]) -> Result<Simulation> {
    let mut base = sim.clone();
    base.sources.clear();
    base.monitors.clear();
    for g in grids {
        let md = base
            .media
            .iter()
            .find(|m| m.name == g.medium)
            .ok_or_else(|| anyhow!("design grid '{}': medium '{}' is not in the simulation's media", g.name, g.medium))?;
        let fill = md.eps[0];
        base.structures.push(Structure { name: g.name.clone(), shape: g.shape(vec![fill; g.samples()]), medium: g.medium.clone() });
    }
    Ok(base)
}

/// Sources must not sit on design voxels: their injection coefficients
/// depend on the local permittivity, which the gradient does not follow.
pub(crate) fn check_sources_outside(io: &IoSet, design: &DesignMap, plan: &Plan) -> Result<()> {
    let (plane, nz) = (plan.plane(), plan.nz());
    for inj in io.inj_h.iter() {
        let bx = design.bbox_h[inj.comp.axis()];
        for &s in &inj.idx {
            let s = s as usize;
            if s / plane >= 1 && bx.contains(s / plane - 1, (s % plane) / nz, s % nz) {
                bail!("a source injects into the design region; move the port or the source off the design grids");
            }
        }
    }
    for inj in io.inj_e.iter() {
        let bx = design.bbox[inj.comp.axis()];
        for &s in &inj.idx {
            let s = s as usize;
            if s / plane >= 1 && bx.contains(s / plane - 1, (s % plane) / nz, s % nz) {
                bail!("a source injects into the design region; move the port or the source off the design grids");
            }
        }
    }
    Ok(())
}

/// Power `1/2 Re int (E_m x H_m*) . n dA` (W) of profile `k` of a port,
/// evaluated as the mode overlap evaluates it (`results::mode_amplitudes`):
/// on the E samples of the port plane, with the H phase shift and the
/// surface elements clipped to the port window.
pub fn port_power(grid: &Grid, periodic: [bool; 3], port: &PortSpec, k: usize) -> f64 {
    let a = port.axis;
    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
    let p = grid.nearest_node(a, port.position).max(1);
    let (origin, _) = crate::plan::transverse_box(grid, a, Some(&port.min), Some(&port.max));
    let (u, v) = crate::geometry::plane_axes(a);
    let nat = |ax: usize| if ax == u { 0usize } else { 1usize };
    let tol = 1e-6 * grid.min_cell_size().iter().cloned().fold(f64::INFINITY, f64::min);
    let mode = &port.profiles[k];
    let delta = grid.node(a, p) - grid.center(a, p - 1);
    let shift = Complex64::from_polar(1.0, TAU * mode.freq * mode.neff / C0 * mode.h_offset.unwrap_or(delta));
    let get = |re: &Vec<f64>, im: &Vec<f64>, g: [usize; 3]| -> Complex64 {
        let iu = g[u] as isize - origin[u] as isize;
        let iv = g[v] as isize - origin[v] as isize;
        if iu < 0 || iv < 0 || iu as usize >= mode.dims[0] || iv as usize >= mode.dims[1] {
            return Complex64::new(0.0, 0.0);
        }
        let o = iu as usize * mode.dims[1] + iv as usize;
        Complex64::new(re.get(o).copied().unwrap_or(0.0), im.get(o).copied().unwrap_or(0.0))
    };
    let mut pm = 0.0;
    for (sign, e_ax, h_ax) in [(1.0, b, c), (-1.0, c, b)] {
        let comp = Component::e(e_ax);
        let (ru, rv) = (grid.index_range(comp, u, port.min[u], port.max[u]), grid.index_range(comp, v, port.min[v], port.max[v]));
        for gu in ru.0..ru.1 {
            for gv in rv.0..rv.1 {
                let mut g = [0usize; 3];
                g[a] = p;
                g[u] = gu;
                g[v] = gv;
                let mut da = 1.0;
                for ax in [u, v] {
                    let n = grid.n()[ax];
                    da *= if periodic[ax] && !comp.is_half(ax) && g[ax] == 0 {
                        0.5 * (grid.cell_size(ax, 0) + grid.cell_size(ax, n - 1))
                    } else {
                        let (lo, hi) = grid.voxel_bounds(comp, ax, g[ax]);
                        (hi.min(port.max[ax] + tol) - lo.max(port.min[ax] - tol)).max(0.0)
                    };
                }
                let em = get(&mode.e_re[nat(e_ax)], &mode.e_im[nat(e_ax)], g);
                let hm = get(&mode.h_re[nat(h_ax)], &mode.h_im[nat(h_ax)], g) * shift;
                pm += 0.5 * sign * (em * hm.conj()).re * da;
            }
        }
    }
    pm
}

impl GradientSession {
    /// Build the session: one engine per entry of `devices`.
    pub fn new(spec: SessionSpec, devices: &[String], factory: &EngineFactory) -> Result<GradientSession> {
        let t0 = Instant::now();
        if devices.is_empty() {
            bail!("no device given");
        }
        if spec.normalize != "monitor" && spec.normalize != "source" {
            bail!("normalize must be 'monitor' or 'source'");
        }
        if spec.freqs.is_empty() || spec.inputs.is_empty() {
            bail!("a session needs at least one frequency and one input");
        }
        if spec.components.iter().any(|c| !c.is_e()) {
            bail!("design components must be E components");
        }
        for ch in spec.inputs.iter().chain(spec.outputs.iter()) {
            if let Channel::Port { port, mode } = ch {
                let p = spec.ports.get(*port).ok_or_else(|| anyhow!("channel refers to unknown port {port}"))?;
                if *mode >= p.profiles.len() {
                    bail!("port '{}' has {} mode profiles, mode {mode} requested", p.name, p.profiles.len());
                }
            }
        }
        if spec.outputs.iter().any(|c| matches!(c, Channel::Source { .. })) {
            bail!("outputs must be port modes");
        }
        let base = base_simulation(&spec.sim, &spec.grids)?;
        check_media(&base, GradientEngine::Session)?;
        let grid = base.build_grid()?;
        let b = &base.boundaries;
        let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
        // The adjoint formula takes the port modes to carry unit power: scale
        // every profile to 1 W, as measured by the overlap itself. S is then
        // that of unit-power modes and the gradient is that of S, whatever
        // the power of the profiles given (the mode solvers return unit
        // power; a profile of power P otherwise scales S by sqrt(P_in / P_out)
        // and the gradient by sqrt(P_in P_out)).
        let mut spec = spec;
        for port in spec.ports.iter_mut() {
            for k in 0..port.profiles.len() {
                let pw = port_power(&grid, periodic, port, k);
                if !pw.is_finite() || pw == 0.0 {
                    bail!("port '{}': mode profile {k} carries no power through the port window ({pw} W)", port.name);
                }
                let f = 1.0 / pw.abs().sqrt();
                let q = &mut port.profiles[k];
                for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                    arr.iter_mut().for_each(|x| *x *= f);
                }
            }
        }

        // ---- the N forward and M adjoint runs ----
        let ng = spec.grids.len();
        let design_monitors: Vec<Monitor> = spec
            .grids
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let (lo, hi) = monitor_box(&grid, g);
                // stride 0: the stride of the simulation (`dft_stride`, or the automatic one),
                // which the port monitors get too. An engine accumulates all monitors of a run
                // in one pass with one phasor table, so they must not differ.
                Monitor::DftField { name: format!("__des{gi}__"), min: lo, max: hi, components: spec.components.clone(), freqs: spec.freqs.clone(), apodization: Apodization::default(), stride: 0 }
            })
            .collect();
        let mode_monitor = |name: &str, p: &PortSpec, pos: f64, min: [f64; 3], max: [f64; 3]| Monitor::Mode {
            name: name.to_string(),
            axis: p.axis,
            position: pos,
            min,
            max,
            freqs: spec.freqs.clone(),
            modes: p.profiles.clone(),
            apodization: Apodization::default(),
        };
        let mut jobs: Vec<(Kind, usize, &Channel)> = spec.inputs.iter().enumerate().map(|(n, c)| (Kind::Forward, n, c)).collect();
        jobs.extend(spec.outputs.iter().enumerate().map(|(m, c)| (Kind::Adjoint, m, c)));
        let mut runs: Vec<Run> = vec![];
        let mut ios: Vec<IoSet> = vec![];
        let mut master: Option<Plan> = None;
        let mut dft_stride = 1;
        for (kind, index, ch) in jobs {
            let mut sim = base.clone();
            let mut monitors = design_monitors.clone();
            let mut names = vec![];
            let mut out_names = vec![];
            let (src, src_port, ex_norm, signal) = match ch {
                Channel::Port { port, mode } => {
                    let p = &spec.ports[*port];
                    let src = Source::Mode {
                        name: "__src__".into(),
                        axis: p.axis,
                        position: p.position,
                        direction: p.direction,
                        min: p.min,
                        max: p.max,
                        profile: p.profiles[*mode].clone(),
                        signal: spec.signal.clone(),
                        amplitude: 1.0,
                    };
                    for (t, (x, mn, mx)) in norm_planes(&grid, p)?.into_iter().enumerate() {
                        let name = format!("__norm{t}__");
                        monitors.push(mode_monitor(&name, p, x, mn, mx));
                        names.push(name);
                    }
                    (src, Some((*port, *mode)), None, spec.signal.clone())
                }
                Channel::Source { source, norm, .. } => {
                    if let Some(v) = norm {
                        if v.len() != spec.freqs.len() {
                            bail!("an excitation's norm needs one value per frequency");
                        }
                    }
                    let sig = source.signal().clone();
                    (named(source.clone(), "__src__"), None, norm.as_ref().map(|v| v.iter().map(|z| Complex64::new(z[0], z[1])).collect()), sig)
                }
            };
            if kind == Kind::Forward {
                for out in &spec.outputs {
                    let Channel::Port { port, .. } = out else { unreachable!() };
                    if Some(*port) != src_port.map(|p| p.0) && !out_names.iter().any(|(q, _)| q == port) {
                        let p = &spec.ports[*port];
                        let name = format!("__out_{}__", out_names.len());
                        monitors.push(mode_monitor(&name, p, p.position, p.min, p.max));
                        names.push(name.clone());
                        out_names.push((*port, name));
                    }
                }
            }
            sim.sources = vec![src];
            sim.monitors = monitors;
            let mut plan = compile_with_grid(&sim, grid.clone(), &session_options())?;
            // Off-diagonal (tensor) media and oriented poles are allowed in
            // the background: they are reciprocal (symmetric tensors), so the
            // adjoint run is an ordinary run in the same scene.
            if plan.bloch {
                bail!("gradient sessions do not support Bloch boundaries");
            }
            if let Some(d) = plan.dft.iter().find(|d| d.stride != plan.dft[0].stride) {
                bail!("monitor '{}' samples every {} steps, '{}' every {}: the monitors of a gradient run must share one stride", d.name, d.stride, plan.dft[0].name, plan.dft[0].stride);
            }
            dft_stride = dft_stride.max(plan.dft.first().map(|d| d.stride).unwrap_or(1));
            let mut meta = light_plan(&plan);
            let io = IoSet::take(&mut plan);
            meta.dft = io.dft[ng..].to_vec();
            runs.push(Run { kind, index, io: ios.len(), src_port, ex_norm, signal, first_port_monitor: ng, names, out_names, meta });
            ios.push(io);
            if master.is_none() {
                master = Some(plan);
            }
        }
        let master = master.unwrap();
        if spec.grids.iter().any(|g| g.permeability) {
            bail!("permeability design grids are supported by reverse mode only, not by the S-matrix session");
        }
        let design = DesignMap::new(&master, &spec.grids)?;
        for io in &ios {
            check_sources_outside(io, &design, &master)?;
        }

        // ---- design boxes: origin, shape and voxel volumes per grid and component ----
        let mut boxes = vec![];
        for gi in 0..ng {
            let mut v = vec![];
            for (ci, c) in spec.components.iter().enumerate() {
                let cb = &ios[0].dft[gi].boxes[ci];
                let s = cb.gsize();
                let bx = IndexBox { lo: cb.glo, size: s };
                let own = design.owned[gi][c.axis()];
                if !own.is_empty() && bx.union(&own) != bx {
                    bail!("design monitor does not cover the grid '{}' for {}", spec.grids[gi].name, c.name());
                }
                let mut vol = Vec::with_capacity(bx.cells());
                for i in 0..s[0] {
                    for j in 0..s[1] {
                        for k in 0..s[2] {
                            vol.push(voxel_volume(&grid, periodic, *c, [bx.lo[0] + i, bx.lo[1] + j, bx.lo[2] + k]));
                        }
                    }
                }
                v.push(DesignBox { comp: *c, bx, vol });
            }
            boxes.push(v);
        }

        // ---- engines ----
        let mut engines = vec![];
        let mut engine_names = vec![];
        for dev in devices {
            let mut eng = factory(dev, master.clone())?;
            for io in &ios {
                eng.add_io(io.clone())?;
            }
            engine_names.push(eng.name());
            engines.push(Mutex::new(eng));
        }
        let cells = grid.num_cells();
        let nruns = runs.len();
        Ok(GradientSession {
            spec,
            grid,
            periodic,
            cells,
            design,
            master,
            engines,
            engine_names,
            runs,
            results: vec![None; nruns],
            boxes,
            design_set: false,
            dft_stride,
            stats: vec![],
            setup_seconds: t0.elapsed().as_secs_f64(),
        })
    }

    /// Number of time steps between two samples of the frequency-domain
    /// monitors (design-region fields and port monitors alike): the
    /// simulation's `dft_stride`, or the automatic stride of its source
    /// (see `plan::auto_dft_stride`). Runs whose sources differ in bandwidth
    /// (an `Excitation` next to port modes) have their own; this is the
    /// largest.
    pub fn dft_stride(&self) -> usize {
        self.dft_stride
    }

    /// (M, N, K): outputs, inputs, frequencies.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.spec.outputs.len(), self.spec.inputs.len(), self.spec.freqs.len())
    }
    pub fn grids(&self) -> &[DesignGrid] {
        &self.spec.grids
    }
    pub fn freqs(&self) -> &[f64] {
        &self.spec.freqs
    }
    pub fn components(&self) -> &[Component] {
        &self.spec.components
    }
    pub fn cells(&self) -> usize {
        self.cells
    }
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    /// Periodic (or Bloch) axes.
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }
    pub fn devices(&self) -> &[String] {
        &self.engine_names
    }
    /// Design-region boxes per grid and component.
    pub fn design_boxes(&self) -> &[Vec<DesignBox>] {
        &self.boxes
    }
    pub fn design(&self) -> &DesignMap {
        &self.design
    }

    /// Set the index-grid samples (one x-major array per grid): only the
    /// coefficients of the design box are re-evaluated and uploaded.
    pub fn set_design(&mut self, eps: &[Vec<f64>]) -> Result<()> {
        self.design.set_eps(eps)?;
        let blocks = self.design.write_plan(&mut self.master);
        for eng in &self.engines {
            let mut e = eng.lock().unwrap();
            for b in &blocks {
                e.write_coefficients(b.axis, b.bx, &b.cb, b.ca.as_deref())?;
            }
            e.commit_coefficients()?;
        }
        self.results.iter_mut().for_each(|r| *r = None);
        self.stats.clear();
        self.design_set = true;
        Ok(())
    }

    /// Permittivity of the E_a voxels over the design bounding box, as the
    /// engine uses it (NaN where no grid owns the voxel), with the box.
    pub fn yee_permittivity(&self, a: usize) -> (IndexBox, Vec<Real>) {
        (self.design.bbox[a], self.design.yee_eps(a))
    }

    fn execute(&self, ei: usize, eng: &mut dyn Engine, r: usize) -> Result<(RunResult, RunStat)> {
        let t0 = Instant::now();
        let run = &self.runs[r];
        let info = run_to_end(eng, run.io, true)?;
        let nmon = run.first_port_monitor + run.meta.dft.len();
        let mut raw = RawResults::default();
        for mi in run.first_port_monitor..nmon {
            raw.dft.push(eng.read_dft(run.io, mi)?);
        }
        let mut res = assemble(&[&run.meta], &[raw]);
        res.stats.steps = info.steps;
        finalize(&mut res, &run.meta);
        let nf = self.spec.freqs.len();
        let amp = |name: &str, mode: usize, plus: bool| -> Result<Vec<Complex64>> {
            let m = res.modes.get(name).ok_or_else(|| anyhow!("mode monitor {name} missing"))?;
            let (re, im) = if plus { (&m.amp_plus_re, &m.amp_plus_im) } else { (&m.amp_minus_re, &m.amp_minus_im) };
            Ok((0..nf).map(|k| Complex64::new(re[mode][k], im[mode][k])).collect())
        };
        let spectrum = || -> Vec<Complex64> { run.signal.spectrum(&self.spec.freqs, run.meta.dt, info.steps.max(1), 0.5) };
        let mut q: Option<Vec<Complex64>> = None;
        let norm: Vec<Complex64> = match (run.src_port, self.spec.normalize.as_str()) {
            (Some((port, mode)), "monitor") => {
                let plus = self.spec.ports[port].direction == Direction::Plus;
                let a1 = amp("__norm0__", mode, plus)?;
                let a2 = amp("__norm1__", mode, plus)?;
                // one-step phase factor of the launched mode over 2 cells; the
                // amplitude is referred back to the source plane
                let qq: Vec<Complex64> = a1.iter().zip(&a2).map(|(x, y)| x / y).collect();
                let n = a1.iter().zip(&qq).map(|(x, y)| x * y).collect();
                q = Some(qq);
                n
            }
            _ => {
                let mut s = spectrum();
                if let Some(en) = &run.ex_norm {
                    s.iter_mut().zip(en).for_each(|(a, b)| *a *= b);
                }
                s
            }
        };
        let mut amps = vec![];
        if run.kind == Kind::Forward {
            for out in &self.spec.outputs {
                let Channel::Port { port, mode } = out else { unreachable!() };
                // amplitude leaving the device: against the port direction
                let back_plus = self.spec.ports[*port].direction == Direction::Minus;
                let a = if Some(*port) == run.src_port.map(|p| p.0) {
                    // reflection: measured 2 cells inside, referred back to the port plane
                    let a = amp("__norm0__", *mode, back_plus)?;
                    (0..nf).map(|k| a[k] / q.as_ref().map(|v| v[k]).unwrap_or(Complex64::new(1.0, 0.0)) / norm[k]).collect::<Vec<_>>()
                } else {
                    let name = &run.out_names.iter().find(|(p, _)| p == port).unwrap().1;
                    let a = amp(name, *mode, back_plus)?;
                    (0..nf).map(|k| a[k] / norm[k]).collect::<Vec<_>>()
                };
                amps.push(a);
            }
        }
        let _ = &run.names;
        let stat = RunStat {
            kind: if run.kind == Kind::Forward { "fwd" } else { "adj" },
            index: run.index,
            device: self.engine_names[ei].clone(),
            steps: info.steps,
            seconds: info.seconds,
            wall: t0.elapsed().as_secs_f64(),
        };
        Ok((RunResult { norm, amps, info, engine: ei, on_primary: ei == 0 }, stat))
    }

    /// Run the given runs, one at a time per engine.
    fn dispatch(&mut self, jobs: Vec<usize>) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let mut done: Vec<(usize, Result<(RunResult, RunStat)>)> = vec![];
        if self.engines.len() == 1 || jobs.len() == 1 {
            let mut eng = self.engines[0].lock().unwrap();
            for r in jobs {
                done.push((r, self.execute(0, &mut **eng, r)));
            }
        } else {
            let queue = Mutex::new(VecDeque::from(jobs));
            let out = Mutex::new(vec![]);
            let this = &*self;
            std::thread::scope(|s| {
                for ei in 0..this.engines.len() {
                    let (queue, out) = (&queue, &out);
                    s.spawn(move || {
                        let mut eng = this.engines[ei].lock().unwrap();
                        loop {
                            let job = queue.lock().unwrap().pop_front();
                            let Some(r) = job else { break };
                            let res = this.execute(ei, &mut **eng, r);
                            out.lock().unwrap().push((r, res));
                        }
                    });
                }
            });
            done = out.into_inner().unwrap();
            done.sort_by_key(|d| d.0);
        }
        for (r, res) in done {
            let (rr, st) = res?;
            self.results[r] = Some(rr);
            self.stats.push(st);
        }
        Ok(())
    }

    fn run_index(&self, kind: Kind, index: usize) -> usize {
        match kind {
            Kind::Forward => index,
            Kind::Adjoint => self.spec.inputs.len() + index,
        }
    }

    /// Run the N excitations (and, with `adjoint`, all M adjoint
    /// simulations in the same dispatch) and return `S[m][n][k]` flattened
    /// in that order.
    pub fn forward(&mut self, adjoint: bool) -> Result<Vec<Complex64>> {
        if !self.design_set {
            bail!("set_design must be called before forward");
        }
        let (m, n, k) = self.shape();
        let mut jobs: Vec<usize> = (0..n).filter(|r| self.results[*r].is_none()).collect();
        if adjoint {
            jobs.extend((n..n + m).filter(|r| self.results[*r].is_none()));
        }
        self.dispatch(jobs)?;
        let mut s = vec![Complex64::new(0.0, 0.0); m * n * k];
        for ni in 0..n {
            let r = self.results[ni].as_ref().unwrap();
            for mi in 0..m {
                for ki in 0..k {
                    s[(mi * n + ni) * k + ki] = r.amps[mi][ki];
                }
            }
        }
        Ok(s)
    }

    /// Steps of every run done since the last `set_design` (forward runs
    /// first), `None` for runs not done.
    pub fn steps(&self) -> Vec<Option<usize>> {
        self.results.iter().map(|r| r.as_ref().map(|x| x.info.steps)).collect()
    }

    /// dL/d(eps_r) on the Yee voxels of every design box for the upstream
    /// gradient `g[m][n][k]` (flattened like S): `out[grid][component]`,
    /// arrays over [`design_boxes`](Self::design_boxes).
    pub fn backward_yee(&mut self, g: &[Complex64]) -> Result<Vec<Vec<Vec<f64>>>> {
        let (m, n, k) = self.shape();
        if g.len() != m * n * k {
            bail!("upstream gradient has {} entries, S has {}", g.len(), m * n * k);
        }
        if (0..n).any(|r| self.results[r].is_none()) {
            bail!("forward must be called before backward");
        }
        let at = |mi: usize, ni: usize, ki: usize| g[(mi * n + ni) * k + ki];
        let zero = Complex64::new(0.0, 0.0);
        let need_m: Vec<usize> = (0..m).filter(|&mi| (0..n).any(|ni| (0..k).any(|ki| at(mi, ni, ki) != zero))).collect();
        let jobs: Vec<usize> = need_m.iter().map(|&mi| self.run_index(Kind::Adjoint, mi)).filter(|r| self.results[*r].is_none()).collect();
        self.dispatch(jobs)?;
        let need_n: Vec<usize> = (0..n).filter(|&ni| need_m.iter().any(|&mi| (0..k).any(|ki| at(mi, ni, ki) != zero))).collect();
        // bring the design-region fields of runs done elsewhere to the primary engine
        let ng = self.spec.grids.len();
        let mut needed: Vec<usize> = need_n.clone();
        needed.extend(need_m.iter().map(|&mi| n + mi));
        for r in needed {
            let (src, on) = {
                let x = self.results[r].as_ref().unwrap();
                (x.engine, x.on_primary)
            };
            if on {
                continue;
            }
            let io = self.runs[r].io;
            for gi in 0..ng {
                let data = self.engines[src].lock().unwrap().read_dft(io, gi)?;
                self.engines[0].lock().unwrap().write_dft(io, gi, &data)?;
            }
            self.results[r].as_mut().unwrap().on_primary = true;
        }
        let mut eng = self.engines[0].lock().unwrap();
        let mut out = vec![];
        for gi in 0..ng {
            eng.product_begin((self.runs[0].io, gi))?;
            for &mi in &need_m {
                let ra = n + mi;
                let na = &self.results[ra].as_ref().unwrap().norm;
                for &ni in &need_n {
                    let nf = &self.results[ni].as_ref().unwrap().norm;
                    let coef: Vec<Complex64> = (0..k)
                        .map(|ki| {
                            let gv = at(mi, ni, ki);
                            if gv == zero { zero } else { Complex64::new(0.0, TAU * self.spec.freqs[ki] * EPS0 / 4.0) * gv.conj() / (na[ki] * nf[ki]) }
                        })
                        .collect();
                    if coef.iter().all(|c| *c == zero) {
                        continue;
                    }
                    eng.product_add((self.runs[ra].io, gi), (self.runs[ni].io, gi), &coef)?;
                }
            }
            let acc = eng.product_end()?;
            let mut per_comp = vec![];
            for (ci, b) in self.boxes[gi].iter().enumerate() {
                per_comp.push(acc[ci].iter().zip(&b.vol).map(|(a, v)| *a as f64 * v).collect::<Vec<f64>>());
            }
            out.push(per_comp);
        }
        Ok(out)
    }

    /// dL/d(eps) on the samples of every design grid for the upstream
    /// gradient `g[m][n][k]` (flattened like S).
    pub fn backward(&mut self, g: &[Complex64]) -> Result<Vec<Vec<f64>>> {
        let yee = self.backward_yee(g)?;
        Ok(self.pullback(&yee))
    }

    /// Transpose of the rasterisation applied to per-voxel gradients shaped
    /// like [`backward_yee`](Self::backward_yee)'s result.
    pub fn pullback(&self, yee: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = self.spec.grids.iter().map(|g| vec![0.0; g.samples()]).collect();
        for (gi, per_comp) in yee.iter().enumerate() {
            for (ci, data) in per_comp.iter().enumerate() {
                let b = &self.boxes[gi][ci];
                self.design.pullback(b.comp.axis(), &b.bx, data, Some(gi), &mut out);
            }
        }
        out
    }
}
