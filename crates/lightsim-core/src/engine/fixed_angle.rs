//! Broadband oblique incidence at a fixed angle in one run.
//!
//! An oblique plane wave in a structure that is periodic along the tilt axis
//! `u` has the physical fields `E(r, t) = P(r, t - s (u - u0) / c)`, where
//! `s = n sin(theta)` (`n` the index at the source plane) and `P` is periodic
//! along `u` for every frequency at once. A run with `Simulation::fixed_angle`
//! solves for `P` and `Q` (the transformed `H~ = eta0 H`). With the time
//! unit `c dt` and `u x X = X_v w - X_w v` for the cyclic axes `(u, v, w)`,
//! Maxwell's equations become
//!
//! ```text
//! eps dP/dt + J = curl Q - s u x dQ/dt,      mu dQ/dt = -curl P + s u x dP/dt.
//! ```
//!
//! The components along `u` are updated as usual. The pairs `(P_v, Q_w)` and
//! `(P_w, Q_v)` are coupled by the time derivatives; eliminating the partner's
//! derivative gives, for a pair `(P_p, Q_q)` (`sigma = +1` for `(v, w)`,
//! `-1` for `(w, v)` when `(u, v, w)` is a cyclic permutation of `(x, y, z)`,
//! the opposite signs otherwise; `A` the two-point average along `u` between
//! the E and H positions of the Yee cell),
//!
//! ```text
//! (eps - s^2 A mu^-1 A) dP_p/dt = c_p - J_p - s A mu^-1 (d_u P_p - d_p P_u)
//! (mu - s^2 A eps^-1 A) dQ_q/dt = -e_q + s A eps^-1 (d_q Q_u - d_u Q_q - sigma J_p)
//! ```
//!
//! (`c = curl Q`, `e = curl P`, `eps` the instantaneous permittivity). The
//! leapfrog keeps P at integer and Q at half-integer steps. The terms with
//! `d_u` of the field being updated and the mass terms are taken at the
//! centre of the step (Crank-Nicolson), which makes each update a cyclic
//! tridiagonal solve along the lines of `u` (factorised once); all other
//! terms are explicit. Dispersive poles are integrated with the trapezoidal
//! rule as currents `J` held at integer steps, so their state is known when
//! the Q update needs `J_p^n`; this keeps the scheme stable for any pole
//! strength (an explicit pole current there is not). CPML along `w` enters
//! through the usual stretched derivatives (the CPML auxiliaries of the
//! coupling terms are carried here).
//!
//! The scheme is stable for `c dt <= sqrt(kappa - s^2) / sqrt(sum_{a != u}
//! 1/d_a^2)` and the usual Courant limit (`kappa = min(1, eps_inf mu)` over
//! the media), so the time step only shrinks above about 35 degrees (3-D)
//! or 45 degrees (2-D) in vacuum, as `cos(theta)`. Frequency-domain data are
//! returned as physical fields: each DFT sample is multiplied by
//! `exp(i 2 pi f s (u - u0) / c)`, `u0` the centre of the source plane.
//!
//! The run uses the CPU, one GPU, or several GPUs (or CPU slabs) split along
//! x when x is the propagation axis ([`run_slabs`]: the tilt axis is never
//! split). It refuses scenes it cannot run correctly (see [`prepare`]).

use crate::Real;
use crate::constants::C0;
use crate::engine::cpu::CpuBackend;
use crate::engine::{Backend, RawResults, Side};
use crate::grid::Component;
use crate::materials::Pole;
use crate::monitors::Monitor;
use crate::plan::{CompileOptions, GhostRule, Plan, compile_with_grid};
use crate::results::Results;
use crate::run::{RunOptions, run_backend};
use crate::sources::Source;
use crate::spec::{Boundary, Simulation};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// Error for code paths that cannot run a fixed-angle simulation.
pub const ENGINE_ONLY: &str = "this simulation has fixed_angle = true (broadband oblique incidence at a fixed angle), \
     which runs on the fixed-angle engine only: on the CPU (device \"cpu\") or on the GPUs of one machine (\"cuda:N\", \"cuda:0,1\"). \
     It does not run over several processes or nodes, and the gradient engines do not support it";

/// Largest polar angle (degrees) of a fixed-angle run, in the incidence
/// medium and in the medium of lowest index.
pub const MAX_ANGLE_DEG: f64 = 80.0;

/// Geometry of a fixed-angle run.
#[derive(Clone, Debug)]
pub struct Setup {
    /// Tilt axis, the other transverse axis, the propagation axis.
    pub u: usize,
    pub v: usize,
    pub w: usize,
    /// Polar angle (radians) in the medium at the source plane.
    pub angle: f64,
    pub n_inc: f64,
    /// `n_inc sin(angle)`.
    pub s: f64,
    pub p_pol: bool,
    /// Phase reference along `u` (centre of the source plane).
    pub u_ref: f64,
}

fn medium_at<'a>(sim: &'a Simulation, p: [f64; 3]) -> &'a str {
    let mut m = sim.background.as_str();
    for st in &sim.structures {
        if st.shape.contains(p) {
            m = st.medium.as_str();
        }
    }
    m
}

/// Check that a fixed-angle simulation can be run, and return the
/// simulation the engine compiles (angle zero on the plane wave, `courant`
/// reduced by the fixed-angle stability limit, `fixed_angle` cleared) with
/// the geometry of the run.
pub fn prepare(sim: &Simulation) -> Result<(Simulation, Setup)> {
    if !sim.fixed_angle {
        bail!("not a fixed-angle simulation");
    }
    let mut probe = sim.clone();
    probe.fixed_angle = false;
    probe.validate()?;
    let src = match sim.sources.as_slice() {
        [s @ Source::PlaneWave { .. }] => s,
        _ => bail!(
            "a fixed-angle run (fixed_angle = true) needs exactly one source, a plane wave; this simulation has {} source(s){}",
            sim.sources.len(),
            if sim.sources.iter().any(|s| !matches!(s, Source::PlaneWave { .. })) { " of other kinds" } else { "" }
        ),
    };
    let Source::PlaneWave { axis, position, polarization, angle, min, max, .. } = src else { unreachable!() };
    let w = *axis;
    if min.is_some() || max.is_some() {
        bail!("fixed-angle run: the plane wave must fill its plane (give no min/max)");
    }
    let pa = polarization.axis();
    if pa == w {
        bail!("plane wave polarization must be a component tangential to the plane");
    }
    let other = 3 - w - pa;
    let inv = sim.invariant();
    let (u, p_pol) = if !polarization.is_e() {
        (other, true)
    } else if inv == Some(other) {
        (pa, true)
    } else {
        (other, false)
    };
    let v = 3 - u - w;
    if let Some(i) = inv {
        if i != v {
            bail!("fixed-angle run in 2-D: the plane of incidence must be the plane of the simulation; with the polarization {} the tilt would be along the invariant axis", polarization.name());
        }
    }
    let ax = ["x", "y", "z"];
    if !angle.is_finite() || angle.abs() > MAX_ANGLE_DEG.to_radians() + 1e-12 {
        bail!("fixed-angle run: angle {:.2} degrees is outside the supported range of +-{MAX_ANGLE_DEG} degrees", angle.to_degrees());
    }
    let bu = sim.boundaries.axis(u);
    if !matches!(bu, [Boundary::Periodic, Boundary::Periodic]) {
        if bu.iter().any(|b| matches!(b, Boundary::Bloch { .. })) {
            bail!("fixed-angle run: use periodic (not Bloch) boundaries along the tilt axis {}; the fixed-angle mode imposes the phase of every frequency itself", ax[u]);
        }
        bail!("fixed-angle run: the boundaries along the tilt axis {} must be periodic", ax[u]);
    }
    if inv != Some(v) && !matches!(sim.boundaries.axis(v), [Boundary::Periodic, Boundary::Periodic]) {
        bail!("fixed-angle run: the boundaries along {} (transverse, normal to the plane of incidence) must be periodic", ax[v]);
    }
    if sim.boundaries.axis(w).iter().any(|b| b.is_periodic()) {
        bail!("fixed-angle run: the propagation axis {} needs absorbing (PML) or wall boundaries, not periodic ones", ax[w]);
    }
    for m in &sim.monitors {
        match m {
            Monitor::Probe { name, .. } | Monitor::FieldTime { name, .. } => bail!(
                "fixed-angle run: monitor '{name}' records the field in time, where the run holds the transformed fields E(r, t + s u / c); use frequency-domain monitors (DFT, flux, mode)"
            ),
            Monitor::Energy { name, .. } => bail!("fixed-angle run: monitor '{name}' (energy in time) is not available; the run holds transformed fields"),
            Monitor::FarField { name, .. } => bail!("fixed-angle run: monitor '{name}' (far-field projection) is not available"),
            _ => {}
        }
    }
    // media
    let media = sim.media_table()?;
    let used = |name: &str| name == sim.background || sim.structures.iter().any(|s| s.medium == name);
    let mut kappa = 1.0f64;
    for m in media.iter().filter(|m| used(&m.name)) {
        let why = if m.is_nonlinear() || m.raman.is_some() {
            Some("nonlinear (chi2/chi3)")
        } else if m.gain.is_some() {
            Some("a gain medium")
        } else if !m.gyro.is_empty() {
            Some("gyrotropic")
        } else if m.director.is_some() {
            Some("a director-field medium")
        } else if !m.oriented.is_empty() || m.is_offdiag() {
            Some("an off-diagonal tensor medium")
        } else if m.sigma_m.iter().any(|&x| x != 0.0) {
            Some("magnetically lossy (sigma_m)")
        } else {
            None
        };
        if let Some(w) = why {
            bail!(
                "fixed-angle run: medium '{}' is {w}, which the fixed-angle mode does not support (supported: diagonal permittivity and permeability, conductivity, Lorentz/Drude/Debye poles); use Bloch boundaries and lightsim.sweep_angle",
                m.name
            );
        }
        if m.pec {
            continue;
        }
        for a in 0..3 {
            kappa = kappa.min(m.eps[a] * m.mu[a]);
        }
    }
    // medium at the source plane
    let mut pc = [0.0; 3];
    for a in 0..3 {
        pc[a] = 0.5 * (sim.region.min[a] + sim.region.max[a]);
    }
    pc[w] = *position;
    let mname = medium_at(sim, pc);
    let md = media.iter().find(|m| m.name == mname).ok_or_else(|| anyhow::anyhow!("unknown medium '{mname}'"))?;
    if md.pec || md.is_dispersive() || md.sigma.iter().any(|&x| x != 0.0) {
        bail!("fixed-angle run: the plane wave must start in a lossless, non-dispersive medium (medium '{}' at the source plane)", md.name);
    }
    if md.eps.iter().any(|&e| e != md.eps[0]) || md.mu.iter().any(|&e| e != md.mu[0]) {
        bail!("fixed-angle run: the medium at the source plane ('{}') must be isotropic", md.name);
    }
    let n_inc = (md.eps[0] * md.mu[0]).sqrt();
    let s = n_inc * angle.sin();
    let smax = kappa.sqrt() * MAX_ANGLE_DEG.to_radians().sin();
    if s.abs() > smax + 1e-12 {
        bail!(
            "fixed-angle run: n sin(angle) = {:.4} at the source; in the medium of lowest index (eps mu = {kappa:.4}) the wave would travel at more than {MAX_ANGLE_DEG} degrees (or be evanescent), which the fixed-angle mode does not support",
            s
        );
    }
    let grid = probe.build_grid()?;
    if grid.n()[u] < 3 {
        bail!("fixed-angle run: the grid needs at least 3 cells along the tilt axis {}", ax[u]);
    }
    // time step: the usual Courant limit and the fixed-angle limit (no
    // spatial step along u in it: the u derivatives are implicit)
    let d = grid.min_cell_size();
    let active = |a: usize| Some(a) != inv;
    let sum_all: f64 = (0..3).filter(|&a| active(a)).map(|a| 1.0 / (d[a] * d[a])).sum();
    let sum_t: f64 = (0..3).filter(|&a| active(a) && a != u).map(|a| 1.0 / (d[a] * d[a])).sum();
    let ratio = ((kappa - s * s).max(0.0).sqrt() * (sum_all / sum_t).sqrt()).min(1.0);
    let mut c = probe;
    c.courant *= ratio;
    if let Source::PlaneWave { angle, .. } = &mut c.sources[0] {
        *angle = 0.0;
    }
    let u_ref = grid.position(Component::e(u), u, grid.n()[u] / 2);
    Ok((c, Setup { u, v, w, angle: *angle, n_inc, s, p_pol, u_ref }))
}

/// Time-step factor of a fixed-angle run relative to the usual Courant
/// limit of the same mesh (1 below about 35 degrees in 3-D, 45 in 2-D).
pub fn time_step_factor(sim: &Simulation) -> Result<f64> {
    let (c, _) = prepare(sim)?;
    Ok(c.courant / sim.courant)
}

/// Compile a fixed-angle simulation: the plan of the zero-angle simulation
/// with the source amplitudes and delays of the fixed-angle wave.
pub fn compile(sim: &Simulation) -> Result<(Plan, Setup)> {
    compile_slab(sim, &CompileOptions::default())
}

/// As [`compile`], for one x-slab of a decomposed run (`opts.slab`).
pub fn compile_slab(sim: &Simulation, opts: &CompileOptions) -> Result<(Plan, Setup)> {
    let (c, st) = prepare(sim)?;
    let grid = c.build_grid()?;
    let mut plan = compile_with_grid(&c, grid, &CompileOptions { allow_2d: false, ..opts.clone() })?;
    if opts.nslabs > 1 && (st.u == 0 || plan.periodic[0]) {
        bail!("fixed-angle run: a run split into x-slabs needs x to be the propagation axis (not the tilt axis, not periodic)");
    }
    if plan.nonlin.is_some() || plan.tensor.is_some() || plan.hloss.is_some() || plan.bloch || plan.twod.is_some() {
        bail!("fixed-angle run: the compiled scene uses a feature the fixed-angle mode does not support");
    }
    if plan.pml.iter().any(|p| p.axis != st.w) {
        bail!("fixed-angle run: PML layers are supported along the propagation axis only");
    }
    // the plane wave in the transformed fields: uniform along u, travelling
    // along w at c / (n cos(theta)), tangential E (s) or H (p) of the
    // oblique wave
    let cos = st.angle.cos();
    let Source::PlaneWave { position, direction, signal, amplitude, .. } = &c.sources[0] else { unreachable!() };
    let scaled = if st.p_pol { &mut plan.inj_h } else { &mut plan.inj_e };
    for inj in scaled.iter_mut() {
        for v in inj.coef.iter_mut() {
            *v *= cos as Real;
        }
    }
    let g = &plan.grid;
    let p = g.nearest_node(st.w, *position).max(1);
    let delta = g.node(st.w, p) - g.center(st.w, p - 1);
    let sgn = direction.sign();
    let dt = plan.dt;
    let amps: Vec<Real> = (0..plan.nsteps).map(|n| (amplitude * signal.value((n as f64 + 0.5) * dt + sgn * delta * st.n_inc * cos / C0)) as Real).collect();
    for inj in plan.inj_e.iter_mut() {
        if inj.is_complex() || inj.ngroups != 1 {
            bail!("fixed-angle run: unexpected source injection");
        }
        inj.amps = amps.clone();
    }
    Ok((plan, st))
}

/// Cyclic tridiagonal factors of the update of one coupled component along
/// the lines of `u` (storage layout), and the Sherman-Morrison ratio per line.
#[derive(Clone, Debug, Default)]
pub struct LineFactors {
    pub lo: Vec<Real>,
    pub cp: Vec<Real>,
    pub inv_den: Vec<Real>,
    pub zs: Vec<Real>,
    pub r: Vec<Real>,
}

/// Dispersive cells of one E component (trapezoidal pole currents).
#[derive(Clone, Debug)]
pub struct PoleComp {
    pub comp: Component,
    pub idx: Vec<u64>,
    pub matid: Vec<u32>,
    /// `scale dt` (weight of the pole current in the E update).
    pub w: Vec<Real>,
    /// `scale dt / cb` (the pole current in curl units).
    pub jw: Vec<Real>,
}

/// Source-plane corrections of a derivative in the coupling terms: at `idx`
/// add `fac * amps[step]` (see `inject_*` of the run).
#[derive(Clone, Debug)]
pub struct Corr {
    pub idx: Vec<u64>,
    pub fac: Vec<Real>,
    pub amps: Vec<Real>,
}

/// Storage geometry of a single-slab plan.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub n: [usize; 3],
    pub stride: [usize; 3],
    pub off: usize,
}

impl Geo {
    pub fn new(p: &Plan) -> Geo {
        let (ny, nz) = (p.n[1], p.n[2]);
        Geo { n: [p.nxl, ny, nz], stride: [ny * nz, nz, 1], off: ny * nz }
    }
    #[inline(always)]
    pub fn sidx(&self, i: [usize; 3]) -> usize {
        self.off + i[0] * self.stride[0] + i[1] * self.stride[1] + i[2]
    }
    #[inline(always)]
    pub fn coords(&self, s: usize) -> [usize; 3] {
        let r = s - self.off;
        [r / self.stride[0], (r % self.stride[0]) / self.stride[1], r % self.stride[1]]
    }
    /// The two axes other than `u` (increasing) and the number of lines along `u`.
    pub fn lines(&self, u: usize) -> (usize, usize, usize) {
        let (a1, a2) = match u {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        (a1, a2, self.n[a1] * self.n[a2])
    }
    pub fn line_base(&self, u: usize, l: usize) -> usize {
        let (a1, a2, _) = self.lines(u);
        let mut i = [0usize; 3];
        i[a1] = l / self.n[a2];
        i[a2] = l % self.n[a2];
        self.sidx(i)
    }
    pub fn cells(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }
    /// Storage index of interior cell number `c` (x slowest).
    #[inline(always)]
    pub fn cell(&self, c: usize) -> usize {
        self.off + c
    }
}

/// Boundary value beyond the last (high side) or before the first (low side)
/// cell along a non-periodic axis: 0 (zero), or `factor *` the cell itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge {
    Zero,
    Mirror(Real),
    Periodic,
}

/// Host tables of a fixed-angle run, shared by the CPU and GPU engines.
#[derive(Clone, Debug)]
pub struct Tables {
    pub st: Setup,
    pub geo: Geo,
    pub s: Real,
    /// c dt.
    pub t: Real,
    /// +1 if (u, v, w) is a cyclic permutation of (x, y, z), else -1: the
    /// sign of the coupling of the pair (P_v, Q_w) (the other pair has the
    /// opposite sign).
    pub hand: Real,
    /// Cyclic factors of E_v, E_w, H_v, H_w.
    pub lines: [LineFactors; 4],
    pub npole: usize,
    /// `[material][pole][9]`: M (2x2, row major), v (2), c (2), d.
    pub pole_tab: Vec<Real>,
    pub gsum: Vec<Real>,
    pub poles: Vec<PoleComp>,
    /// E_v injection: correction of d_w H_u at E_v (Q_w update).
    pub corr_f: Vec<Corr>,
    /// Q_v injection: correction of d_w E_u at Q_v (P_w update).
    pub corr_k: Vec<Corr>,
    /// Boundary rule along w: E beyond the high side, H before the low side.
    pub e_hi: Edge,
    pub h_lo: Edge,
}

fn edge_rules(plan: &Plan, a: usize) -> (Edge, Edge) {
    if plan.periodic[a] {
        return (Edge::Periodic, Edge::Periodic);
    }
    if a == 0 {
        let e = match plan.ghost_e_hi {
            GhostRule::Mirror { factor } => Edge::Mirror(factor),
            _ => Edge::Zero,
        };
        let h = match plan.ghost_h_lo {
            GhostRule::Mirror { factor } => Edge::Mirror(factor),
            _ => Edge::Zero,
        };
        (e, h)
    } else {
        let bc = if a == 1 { plan.bc_y } else { plan.bc_z };
        let e = if bc[1] == crate::plan::BC_MIRROR { Edge::Mirror(1.0) } else { Edge::Zero };
        let h = if bc[0] == crate::plan::BC_MIRROR { Edge::Mirror(-1.0) } else { Edge::Zero };
        (e, h)
    }
}

/// Trapezoidal-rule pole: `y' = A y + b E`, `J = c . y + d E` (seconds).
fn pole_ode(p: &Pole) -> ([[f64; 2]; 2], [f64; 2], [f64; 2], f64) {
    match *p {
        Pole::Lorentz { delta_eps, omega0, gamma } => ([[0.0, 1.0], [-omega0 * omega0, -gamma]], [0.0, delta_eps * omega0 * omega0], [0.0, 1.0], 0.0),
        Pole::Drude { omega_p, gamma } => ([[-gamma, 0.0], [0.0, 0.0]], [omega_p * omega_p, 0.0], [1.0, 0.0], 0.0),
        Pole::Debye { delta_eps, tau } => ([[-1.0 / tau, 0.0], [0.0, 0.0]], [delta_eps / tau, 0.0], [-1.0 / tau, 0.0], delta_eps / tau),
    }
}

/// `[M (4), v (2), c (2), d]` of the trapezoidal update
/// `y^{n+1} = M y^n + v (E^n + E^{n+1})`.
pub fn pole_trapezoid(p: &Pole, dt: f64) -> [f64; 9] {
    let (a, b, c, d) = pole_ode(p);
    let h = 0.5 * dt;
    // (I - h A)^-1
    let m = [[1.0 - h * a[0][0], -h * a[0][1]], [-h * a[1][0], 1.0 - h * a[1][1]]];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let mi = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
    let pl = [[1.0 + h * a[0][0], h * a[0][1]], [h * a[1][0], 1.0 + h * a[1][1]]];
    let mm = |r: usize, k: usize| mi[r][0] * pl[0][k] + mi[r][1] * pl[1][k];
    let v = [mi[0][0] * h * b[0] + mi[0][1] * h * b[1], mi[1][0] * h * b[0] + mi[1][1] * h * b[1]];
    [mm(0, 0), mm(0, 1), mm(1, 0), mm(1, 1), v[0], v[1], c[0], c[1], d]
}

/// Susceptibility of a pole as seen by the trapezoidal scheme at angular
/// frequency `omega`: the continuous one at `2/dt tan(omega dt / 2)`.
pub fn chi_trapezoid(p: &Pole, omega: f64, dt: f64) -> num_complex::Complex64 {
    p.chi(2.0 / dt * (0.5 * omega * dt).tan())
}

impl Tables {
    /// Build the tables; takes the dispersive tables out of `plan` (the
    /// zero-angle engine must not apply its explicit poles).
    pub fn build(plan: &mut Plan, st: &Setup) -> Result<Tables> {
        if (plan.x0 != 0 || plan.nxl != plan.n[0]) && (st.u == 0 || plan.periodic[0]) {
            bail!("{ENGINE_ONLY}");
        }
        let geo = Geo::new(plan);
        let t = C0 * plan.dt;
        let s = st.s;
        let (u, v, w) = (st.u, st.v, st.w);
        let hand: Real = if v == (u + 1) % 3 { 1.0 } else { -1.0 };
        let disp = plan.disp.take();
        // poles
        let disp_media: Vec<usize> = (0..plan.media.len()).filter(|&m| plan.media[m].is_dispersive()).collect();
        let npole = disp_media.iter().map(|&m| plan.media[m].poles.len()).max().unwrap_or(0);
        let mut pole_tab = vec![0 as Real; disp_media.len() * npole * 9];
        let mut gsum = vec![0 as Real; disp_media.len()];
        for (mi, &m) in disp_media.iter().enumerate() {
            let mut g = 0.0;
            for (q, p) in plan.media[m].poles.iter().enumerate() {
                let c = pole_trapezoid(p, plan.dt);
                g += c[6] * c[4] + c[7] * c[5] + c[8];
                for (k, x) in c.iter().enumerate() {
                    pole_tab[(mi * npole + q) * 9 + k] = *x as Real;
                }
            }
            gsum[mi] = g as Real;
        }
        let ca = |a: usize, s: usize| plan.ca.as_ref().map(|c| c[a][s]).unwrap_or(1.0) as f64;
        let ch = |a: usize, s: usize| plan.ch.as_ref().map(|c| c[a][s]).unwrap_or(plan.ch_scalar) as f64;
        let mut poles = Vec::new();
        let mut wg: [Vec<f64>; 3] = [vec![], vec![], vec![]];
        if let Some(d) = &disp {
            for dc in &d.comps {
                let a = dc.comp.axis();
                let mut pc = PoleComp { comp: dc.comp, idx: dc.idx.clone(), matid: dc.matid.clone(), w: vec![], jw: vec![] };
                if wg[a].is_empty() {
                    wg[a] = vec![0.0; plan.storage_len()];
                }
                for (q, &sx) in dc.idx.iter().enumerate() {
                    let sx = sx as usize;
                    let wv = dc.scale[q] as f64 * plan.dt;
                    pc.w.push(wv as Real);
                    pc.jw.push((wv / plan.cb[a][sx] as f64) as Real);
                    wg[a][sx] = wv * gsum[dc.matid[q] as usize] as f64;
                }
                poles.push(pc);
            }
        }
        let wg_at = |a: usize, s: usize| if wg[a].is_empty() { 0.0 } else { wg[a][s] };
        // instantaneous inverse permittivity at E cells (0 in PEC)
        let ie = |a: usize, s: usize| {
            let cb = plan.cb[a][s] as f64;
            if cb == 0.0 { 0.0 } else { 2.0 * cb / (t * (1.0 + ca(a, s))) }
        };
        let gx = |i: usize| i; // single slab: local x index = global
        let inv_p = |i: usize| plan.inv_p[u][gx(i)] as f64;
        let inv_d = |i: usize| plan.inv_d[u][gx(i)] as f64;
        let (_, _, nlines) = geo.lines(u);
        let n = geo.n[u];
        let su = geo.stride[u];
        let mut lines: [LineFactors; 4] = Default::default();
        let len = plan.storage_len();
        for (ci, lf) in lines.iter_mut().enumerate() {
            let (is_e, ax) = match ci {
                0 => (true, v),
                1 => (true, w),
                2 => (false, v),
                _ => (false, w),
            };
            // partner component axis
            let pair = if is_e { if ax == v { w } else { v } } else if ax == v { w } else { v };
            lf.lo = vec![0 as Real; len];
            lf.cp = vec![0 as Real; len];
            lf.inv_den = vec![0 as Real; len];
            lf.zs = vec![0 as Real; len];
            lf.r = vec![0 as Real; nlines];
            let mut lo = vec![0f64; n];
            let mut dg = vec![0f64; n];
            let mut up = vec![0f64; n];
            for l in 0..nlines {
                let base = geo.line_base(u, l);
                let at = |i: usize| base + (i % n) * su;
                for i in 0..n {
                    let (si, sm, sp) = (at(i), at(i + n - 1), at(i + 1));
                    let (im, ip) = ((i + n - 1) % n, (i + 1) % n);
                    if is_e {
                        let r_hi = ch(pair, si) / t;
                        let r_lo = ch(pair, sm) / t;
                        let cbp = plan.cb[ax][si] as f64;
                        let k = cbp * s * s / t;
                        let hc = 0.5 * cbp * s;
                        dg[i] = 1.0 + 0.5 * wg_at(ax, si) - 0.25 * k * (r_hi + r_lo) + 0.5 * hc * (-r_hi * inv_p(i) + r_lo * inv_p(im));
                        up[i] = -0.25 * k * r_hi + 0.5 * hc * r_hi * inv_p(i);
                        lo[i] = -0.25 * k * r_lo - 0.5 * hc * r_lo * inv_p(im);
                    } else {
                        let (e0, e1) = (ie(pair, si), ie(pair, sp));
                        let chq = ch(ax, si);
                        let k = chq * s * s / t;
                        let hc = 0.5 * chq * s;
                        dg[i] = 1.0 - 0.25 * k * (e0 + e1) + 0.5 * hc * (e0 * inv_d(i) - e1 * inv_d(ip));
                        up[i] = -0.25 * k * e1 + 0.5 * hc * e1 * inv_d(ip);
                        lo[i] = -0.25 * k * e0 - 0.5 * hc * e0 * inv_d(i);
                    }
                }
                // Sherman-Morrison for the corners A[0][n-1] = lo0, A[n-1][0] = up_{n-1}
                let beta = lo[0];
                let alpha = up[n - 1];
                let gamma = -dg[0];
                let mut b = dg.clone();
                b[0] -= gamma;
                b[n - 1] -= alpha * beta / gamma;
                let mut cp = vec![0f64; n];
                let mut idn = vec![0f64; n];
                idn[0] = 1.0 / b[0];
                cp[0] = up[0] * idn[0];
                for i in 1..n {
                    let den = b[i] - lo[i] * cp[i - 1];
                    idn[i] = 1.0 / den;
                    cp[i] = if i + 1 < n { up[i] * idn[i] } else { 0.0 };
                }
                // A' z = (gamma, 0, ..., 0, alpha)
                let mut z = vec![0f64; n];
                let rhs = |i: usize| if i == 0 { gamma } else if i == n - 1 { alpha } else { 0.0 };
                z[0] = rhs(0) * idn[0];
                for i in 1..n {
                    z[i] = (rhs(i) - lo[i] * z[i - 1]) * idn[i];
                }
                for i in (0..n - 1).rev() {
                    z[i] -= cp[i] * z[i + 1];
                }
                let ratio = beta / gamma;
                let vz = z[0] + ratio * z[n - 1];
                for i in 0..n {
                    let si = at(i);
                    lf.lo[si] = if i == 0 { 0.0 } else { lo[i] as Real };
                    lf.cp[si] = cp[i] as Real;
                    lf.inv_den[si] = idn[i] as Real;
                    lf.zs[si] = (z[i] / (1.0 + vz)) as Real;
                }
                lf.r[l] = ratio as Real;
            }
        }
        // source-plane corrections of the coupling derivatives: an injection
        // into E_v changes c_v = hand (d_w H_u - d_u H_w), one into Q_v
        // changes e_v = hand (d_w E_u - d_u E_w)
        let mut corr_f = Vec::new();
        for inj in plan.inj_e.iter().filter(|j| j.comp == Component::e(v)) {
            let fac = inj.idx.iter().zip(&inj.coef).map(|(&i, &c)| {
                let cb = plan.cb[v][i as usize];
                if cb != 0.0 { hand * c / cb } else { 0.0 }
            }).collect();
            corr_f.push(Corr { idx: inj.idx.clone(), fac, amps: inj.amps.clone() });
        }
        let mut corr_k = Vec::new();
        for inj in plan.inj_h.iter().filter(|j| j.comp == Component::h(v)) {
            let fac = inj.idx.iter().zip(&inj.coef).map(|(&i, &c)| -hand * c / ch(v, i as usize) as Real).collect();
            corr_k.push(Corr { idx: inj.idx.clone(), fac, amps: inj.amps.clone() });
        }
        let (e_hi, h_lo) = edge_rules(plan, w);
        Ok(Tables { st: st.clone(), geo, s: s as Real, t: t as Real, hand, lines, npole, pole_tab, gsum, poles, corr_f, corr_k, e_hi, h_lo })
    }

    /// Multiply the DFT accumulators by `exp(i 2 pi f s (u - u0) / c)`: the
    /// physical fields.
    pub fn physical_dft(&self, plan: &Plan, raw: &mut RawResults) {
        let u = self.st.u;
        let g = &plan.grid;
        for (d, rd) in plan.dft.iter().zip(raw.dft.iter_mut()) {
            for (bx, (re, im)) in d.boxes.iter().zip(rd.boxes.iter_mut()) {
                if !bx.is_local() || re.is_empty() {
                    continue;
                }
                let n = bx.lcells();
                let ls = bx.lsize;
                for (fi, &f) in d.freqs.iter().enumerate() {
                    let k = std::f64::consts::TAU * f * self.st.s / C0;
                    for ii in 0..ls[0] {
                        for jj in 0..ls[1] {
                            for kk in 0..ls[2] {
                                let gi = [plan.x0 + bx.lstart[0] + ii, bx.lstart[1] + jj, bx.lstart[2] + kk];
                                let x = g.position(bx.comp, u, gi[u]);
                                let (sn, cs) = (k * (x - self.st.u_ref)).sin_cos();
                                let o = fi * n + (ii * ls[1] + jj) * ls[2] + kk;
                                let (a, b) = (re[o] as f64, im[o] as f64);
                                re[o] = (a * cs - b * sn) as Real;
                                im[o] = (a * sn + b * cs) as Real;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bytes of the fixed-angle state beyond the zero-angle engine's.
    pub fn memory_bytes(&self) -> usize {
        let r = std::mem::size_of::<Real>();
        let len = self.lines[0].lo.len();
        // factors (16), saved fields (6), two work arrays
        (16 + 6 + 2) * len * r + self.poles.iter().map(|p| p.idx.len() * (2 * self.npole + 3) * r).sum::<usize>()
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut Real);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    #[inline(always)]
    unsafe fn get(self, i: usize) -> Real {
        unsafe { *self.0.add(i) }
    }
    #[inline(always)]
    unsafe fn set(self, i: usize, v: Real) {
        unsafe { *self.0.add(i) = v }
    }
}

/// CPU engine of a fixed-angle run: the zero-angle CPU engine for the curl
/// updates, CPML and sources, and the coupling, line solves and poles here.
pub struct FixedAngleCpu {
    inner: CpuBackend,
    tb: Tables,
    hu_old: Vec<Real>,
    hq_old: [Vec<Real>; 3],
    e_old: [Vec<Real>; 3],
    /// Work arrays: the derivative terms at E (F) and H (K) positions.
    fbuf: Vec<Real>,
    kbuf: Vec<Real>,
    /// CPML auxiliaries of d_w H_u at E_v and d_w E_u at H_v, per PML slab.
    psi_e: Vec<Vec<Real>>,
    psi_h: Vec<Vec<Real>>,
    /// Pole states per component: `[pole][2][cell]`.
    ystate: Vec<Vec<Real>>,
    /// J^n + J0 per dispersive cell (between the two pole passes).
    jtmp: Vec<Vec<Real>>,
}

impl FixedAngleCpu {
    pub fn new(sim: &Simulation) -> Result<FixedAngleCpu> {
        Self::new_slab(sim, &CompileOptions::default())
    }

    /// One x-slab of a decomposed run (see [`run_slabs`]).
    pub fn new_slab(sim: &Simulation, opts: &CompileOptions) -> Result<FixedAngleCpu> {
        let (mut plan, st) = compile_slab(sim, opts)?;
        let tb = Tables::build(&mut plan, &st)?;
        let len = plan.storage_len();
        let psi_e = plan.pml.iter().map(|p| vec![0 as Real; p.cells()]).collect();
        let psi_h = plan.pml.iter().map(|p| vec![0 as Real; p.cells()]).collect();
        let ystate = tb.poles.iter().map(|p| vec![0 as Real; p.idx.len() * 2 * tb.npole]).collect();
        let jtmp = tb.poles.iter().map(|p| vec![0 as Real; p.idx.len()]).collect();
        let inner = CpuBackend::new(plan)?;
        Ok(FixedAngleCpu {
            inner,
            tb,
            hu_old: vec![0 as Real; len],
            hq_old: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
            e_old: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
            fbuf: vec![0 as Real; len],
            kbuf: vec![0 as Real; len],
            psi_e,
            psi_h,
            ystate,
            jtmp,
        })
    }

    pub fn tables(&self) -> &Tables {
        &self.tb
    }

    /// F at E_p cells (pair 0: p = v, d_w H_u; pair 1: p = w, d_v H_u) at
    /// time n, minus sigma J_p^n.
    fn pass_f(&mut self, pair: usize, step: usize) {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        // coupling sign of the pair (P_p, Q_q)
        let (p, q, sigma) = if pair == 0 { (v, w, tb.hand) } else { (w, v, -tb.hand) };
        let (plan, e, h) = self.inner.plan_and_fields();
        let geo = tb.geo;
        let hu = Ptr(h[u].as_ptr() as *mut Real);
        let ho = Ptr(self.hu_old.as_ptr() as *mut Real);
        let ep = Ptr(e[p].as_ptr() as *mut Real);
        let f = Ptr(self.fbuf.as_mut_ptr());
        let inv = &plan.inv_d[q];
        let nq = geo.n[q];
        let sq = geo.stride[q];
        let periodic = plan.periodic[q];
        let h_lo = tb.h_lo;
        let cb = &plan.cb[p];
        let ca = plan.ca.as_ref().map(|c| &c[p]);
        (0..geo.n[0]).into_par_iter().for_each(|i| unsafe {
            for j in 0..geo.n[1] {
                for k in 0..geo.n[2] {
                    let c = [i, j, k];
                    let s = geo.sidx(c);
                    let jq = c[q];
                    // (along x the ghost plane holds the neighbour: the run
                    // refreshes it before this pass)
                    let (a_new, a_old) = if jq > 0 || q == 0 {
                        (hu.get(s - sq), ho.get(s - sq))
                    } else if periodic {
                        (hu.get(s + (nq - 1) * sq), ho.get(s + (nq - 1) * sq))
                    } else {
                        match h_lo {
                            Edge::Mirror(fc) => (fc * hu.get(s), fc * ho.get(s)),
                            _ => (0.0, 0.0),
                        }
                    };
                    let d = 0.5 * ((hu.get(s) - a_new) + (ho.get(s) - a_old)) * inv[jq];
                    // conduction current (curl units) at time n
                    let mut jc = 0.0;
                    if let Some(ca) = ca {
                        let cbv = cb[s];
                        if cbv != 0.0 {
                            jc = (1.0 - ca[s]) / cbv * ep.get(s);
                        }
                    }
                    f.set(s, d - sigma * jc);
                }
            }
        });
        // CPML: stretched d_w H_u = kinv d + psi (pair 0 only: q = w)
        if q == w {
            for (si, slab) in plan.pml.iter().enumerate() {
                let psi = &mut self.psi_e[si];
                let (lx, ly, lz) = (slab.size[0], slab.size[1], slab.size[2]);
                for ii in 0..lx {
                    for jj in 0..ly {
                        for kk in 0..lz {
                            let t = (ii * ly + jj) * lz + kk;
                            let c = [slab.start[0] + ii, slab.start[1] + jj, slab.start[2] + kk];
                            let s = geo.sidx(c);
                            let along = c[w];
                            let qq = along - slab.start[w];
                            let (b, a, kinv) = (slab.e_b[qq], slab.e_a[qq], slab.e_kinv[qq]);
                            unsafe {
                                let (lo_new, lo_old) = if along > 0 || w == 0 {
                                    (hu.get(s - sq), ho.get(s - sq))
                                } else {
                                    match h_lo {
                                        Edge::Mirror(fc) => (fc * hu.get(s), fc * ho.get(s)),
                                        _ => (0.0, 0.0),
                                    }
                                };
                                let d_new = (hu.get(s) - lo_new) * inv[along];
                                let d_old = (ho.get(s) - lo_old) * inv[along];
                                let pn = b * psi[t] + a * d_new;
                                let extra = (kinv - 1.0) * 0.5 * (d_new + d_old) + 0.5 * (psi[t] + pn);
                                psi[t] = pn;
                                f.set(s, f.get(s) + extra);
                            }
                        }
                    }
                }
            }
            // source plane: the E_v injection is part of curl H there
            for c in &tb.corr_f {
                let a = 0.5 * (c.amps[step] + if step > 0 { c.amps[step - 1] } else { 0.0 });
                for (&ix, &fc) in c.idx.iter().zip(&c.fac) {
                    unsafe { f.set(ix as usize, f.get(ix as usize) + fc * a) };
                }
            }
        }
        // pole currents at time n
        let np = tb.npole;
        for (pc, ys) in tb.poles.iter().zip(self.ystate.iter()) {
            if pc.comp.axis() != p {
                continue;
            }
            let n = pc.idx.len();
            for t in 0..n {
                let s = pc.idx[t] as usize;
                let m = pc.matid[t] as usize;
                let en = unsafe { ep.get(s) };
                let mut jn = 0.0;
                for qp in 0..np {
                    let c = &tb.pole_tab[(m * np + qp) * 9..(m * np + qp) * 9 + 9];
                    jn += c[6] * ys[(qp * 2) * n + t] + c[7] * ys[(qp * 2 + 1) * n + t] + c[8] * en;
                }
                unsafe { f.set(s, f.get(s) - sigma * pc.jw[t] * jn) };
            }
        }
        let _ = u;
    }

    /// Solve the Q_q update along the lines of u (pair 0: q = w, 1: q = v).
    fn solve_h(&mut self, pair: usize) {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        let (p, q, ci) = if pair == 0 { (v, w, 3) } else { (w, v, 2) };
        let (plan, _e, h) = self.inner.plan_and_fields();
        let geo = tb.geo;
        let lf = &tb.lines[ci];
        let hq = Ptr(h[q].as_mut_ptr());
        let wo = Ptr(self.hq_old[q].as_ptr() as *mut Real);
        let f = Ptr(self.fbuf.as_ptr() as *mut Real);
        let cb = &plan.cb[p];
        let ca = plan.ca.as_ref().map(|c| &c[p]);
        let chv = plan.ch.as_ref().map(|c| &c[q]);
        let chs = plan.ch_scalar;
        let (s_, t) = (tb.s, tb.t);
        let id = &plan.inv_d[u];
        let n = geo.n[u];
        let su = geo.stride[u];
        let (_, _, nlines) = geo.lines(u);
        let ie = move |sx: usize| -> Real {
            let c = cb[sx];
            if c == 0.0 { 0.0 } else { 2.0 * c / (t * (1.0 + ca.map(|a| a[sx]).unwrap_or(1.0))) }
        };
        (0..nlines).into_par_iter().for_each(|l| unsafe {
            let base = geo.line_base(u, l);
            let at = |i: usize| base + (i % n) * su;
            // forward sweep with the right-hand side on the fly
            let mut prev = 0.0 as Real;
            for i in 0..n {
                let (si, sm, sp) = (at(i), at(i + n - 1), at(i + 1));
                let (e0, e1) = (ie(si), ie(sp));
                let chq = chv.map(|c| c[si]).unwrap_or(chs);
                let k = chq * s_ * s_ / t;
                let hc = 0.5 * chq * s_;
                let (w0, wm, wp) = (wo.get(si), wo.get(sm), wo.get(sp));
                let rhs = hq.get(si) - 0.25 * k * (e0 * (wm + w0) + e1 * (w0 + wp)) - 0.5 * hc * (e0 * (w0 - wm) * id[i] + e1 * (wp - w0) * id[(i + 1) % n])
                    + chq * s_ * 0.5 * (e0 * f.get(si) + e1 * f.get(sp));
                let d = (rhs - lf.lo[si] * prev) * lf.inv_den[si];
                hq.set(si, d);
                prev = d;
            }
            line_back(hq, lf, base, su, n, l);
        });
    }

    /// K at H_q cells: d_p E_u at time n plus at time n + 1 (pair 0: p = v
    /// at H_w; pair 1: p = w at H_v, with CPML and source corrections).
    fn pass_k(&mut self, pair: usize, step: usize) {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        let p = if pair == 0 { v } else { w };
        let (plan, e, _h) = self.inner.plan_and_fields();
        let geo = tb.geo;
        let eu = Ptr(e[u].as_ptr() as *mut Real);
        let eo = Ptr(self.e_old[u].as_ptr() as *mut Real);
        let kb = Ptr(self.kbuf.as_mut_ptr());
        let inv = &plan.inv_p[p];
        let np_ = geo.n[p];
        let sp = geo.stride[p];
        let periodic = plan.periodic[p];
        let e_hi = tb.e_hi;
        (0..geo.n[0]).into_par_iter().for_each(|i| unsafe {
            for j in 0..geo.n[1] {
                for k in 0..geo.n[2] {
                    let c = [i, j, k];
                    let s = geo.sidx(c);
                    let jp = c[p];
                    let (a_new, a_old) = if jp + 1 < np_ || p == 0 {
                        (eu.get(s + sp), eo.get(s + sp))
                    } else if periodic {
                        (eu.get(s - (np_ - 1) * sp), eo.get(s - (np_ - 1) * sp))
                    } else {
                        match e_hi {
                            Edge::Mirror(fc) => (fc * eu.get(s), fc * eo.get(s)),
                            _ => (0.0, 0.0),
                        }
                    };
                    kb.set(s, ((a_new - eu.get(s)) + (a_old - eo.get(s))) * inv[jp]);
                }
            }
        });
        if p == w {
            for (si, slab) in plan.pml.iter().enumerate() {
                let psi = &mut self.psi_h[si];
                let (lx, ly, lz) = (slab.size[0], slab.size[1], slab.size[2]);
                for ii in 0..lx {
                    for jj in 0..ly {
                        for kk in 0..lz {
                            let t = (ii * ly + jj) * lz + kk;
                            let c = [slab.start[0] + ii, slab.start[1] + jj, slab.start[2] + kk];
                            let s = geo.sidx(c);
                            let along = c[w];
                            let qq = along - slab.start[w];
                            let (b, a, kinv) = (slab.h_b[qq], slab.h_a[qq], slab.h_kinv[qq]);
                            unsafe {
                                let (hi_new, hi_old) = if along + 1 < np_ || w == 0 {
                                    (eu.get(s + sp), eo.get(s + sp))
                                } else {
                                    match e_hi {
                                        Edge::Mirror(fc) => (fc * eu.get(s), fc * eo.get(s)),
                                        _ => (0.0, 0.0),
                                    }
                                };
                                let d_new = (hi_new - eu.get(s)) * inv[along];
                                let d_old = (hi_old - eo.get(s)) * inv[along];
                                let pn = b * psi[t] + a * d_new;
                                let extra = (kinv - 1.0) * (d_new + d_old) + psi[t] + pn;
                                psi[t] = pn;
                                kb.set(s, kb.get(s) + extra);
                            }
                        }
                    }
                }
            }
            for c in &tb.corr_k {
                let last = c.amps.len() - 1;
                let a = c.amps[step] + c.amps[(step + 1).min(last)];
                for (&ix, &fc) in c.idx.iter().zip(&c.fac) {
                    unsafe { kb.set(ix as usize, kb.get(ix as usize) + fc * a) };
                }
            }
        }
    }

    /// Solve the P_p update along the lines of u (pair 0: p = v, 1: p = w).
    fn solve_e(&mut self, pair: usize) {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        let (p, q, ci) = if pair == 0 { (v, w, 0) } else { (w, v, 1) };
        let (plan, e, _h) = self.inner.plan_and_fields();
        let geo = tb.geo;
        let lf = &tb.lines[ci];
        let ep = Ptr(e[p].as_mut_ptr());
        let yo = Ptr(self.e_old[p].as_ptr() as *mut Real);
        let kb = Ptr(self.kbuf.as_ptr() as *mut Real);
        let cb = &plan.cb[p];
        let chv = plan.ch.as_ref().map(|c| &c[q]);
        let chs = plan.ch_scalar;
        let (s_, t) = (tb.s, tb.t);
        let ipv = &plan.inv_p[u];
        let n = geo.n[u];
        let su = geo.stride[u];
        let (_, _, nlines) = geo.lines(u);
        (0..nlines).into_par_iter().for_each(|l| unsafe {
            let base = geo.line_base(u, l);
            let at = |i: usize| base + (i % n) * su;
            let mut prev = 0.0 as Real;
            for i in 0..n {
                let (si, sm, sp) = (at(i), at(i + n - 1), at(i + 1));
                let r_hi = chv.map(|c| c[si]).unwrap_or(chs) / t;
                let r_lo = chv.map(|c| c[sm]).unwrap_or(chs) / t;
                let cbp = cb[si];
                let k = cbp * s_ * s_ / t;
                let hc = 0.5 * cbp * s_;
                let (y0, ym, yp) = (yo.get(si), yo.get(sm), yo.get(sp));
                let g_hi = (yp - y0) * ipv[i] - kb.get(si);
                let g_lo = (y0 - ym) * ipv[(i + n - 1) % n] - kb.get(sm);
                let rhs = ep.get(si) - 0.25 * k * (r_hi * (y0 + yp) + r_lo * (ym + y0)) - 0.5 * hc * (r_hi * g_hi + r_lo * g_lo);
                let d = (rhs - lf.lo[si] * prev) * lf.inv_den[si];
                ep.set(si, d);
                prev = d;
            }
            line_back(ep, lf, base, su, n, l);
        });
    }

    /// Before the P_p solves: E*_a -= w (J^n + J0) / 2 at the dispersive
    /// cells; the u component is final after division by 1 + w g / 2.
    fn poles_pre(&mut self) {
        let tb = &self.tb;
        let u = tb.st.u;
        let np = tb.npole;
        let (_, e, _) = self.inner.plan_and_fields();
        for ((pc, ys), jt) in tb.poles.iter().zip(self.ystate.iter()).zip(self.jtmp.iter_mut()) {
            let a = pc.comp.axis();
            let n = pc.idx.len();
            for t in 0..n {
                let s = pc.idx[t] as usize;
                let m = pc.matid[t] as usize;
                let y = self.e_old[a][s];
                let mut j = 0.0;
                for qp in 0..np {
                    let c = &tb.pole_tab[(m * np + qp) * 9..(m * np + qp) * 9 + 9];
                    let (y0, y1) = (ys[(qp * 2) * n + t], ys[(qp * 2 + 1) * n + t]);
                    // J^n + c . (M y + v Y)
                    j += c[6] * y0 + c[7] * y1 + c[8] * y;
                    j += c[6] * (c[0] * y0 + c[1] * y1 + c[4] * y) + c[7] * (c[2] * y0 + c[3] * y1 + c[5] * y);
                }
                jt[t] = j;
                let mut x = e[a][s] - 0.5 * pc.w[t] * j;
                if a == u {
                    x /= 1.0 + 0.5 * pc.w[t] * tb.gsum[m];
                }
                e[a][s] = x;
            }
        }
    }

    /// After the solves: y^{n+1} = M y^n + v (E^n + E^{n+1}).
    fn poles_post(&mut self) {
        let tb = &self.tb;
        let np = tb.npole;
        let (_, e, _) = self.inner.plan_and_fields();
        for (pc, ys) in tb.poles.iter().zip(self.ystate.iter_mut()) {
            let a = pc.comp.axis();
            let n = pc.idx.len();
            for t in 0..n {
                let s = pc.idx[t] as usize;
                let m = pc.matid[t] as usize;
                let sum = self.e_old[a][s] + e[a][s];
                for qp in 0..np {
                    let c = &tb.pole_tab[(m * np + qp) * 9..(m * np + qp) * 9 + 9];
                    let (i0, i1) = ((qp * 2) * n + t, (qp * 2 + 1) * n + t);
                    let (y0, y1) = (ys[i0], ys[i1]);
                    ys[i0] = c[0] * y0 + c[1] * y1 + c[4] * sum;
                    ys[i1] = c[2] * y0 + c[3] * y1 + c[5] * sum;
                }
            }
        }
    }
}

/// Back substitution and Sherman-Morrison correction of one line (the
/// forward sweep has left d' in place).
#[inline(always)]
unsafe fn line_back(x: Ptr, lf: &LineFactors, base: usize, su: usize, n: usize, l: usize) {
    unsafe {
        let at = |i: usize| base + i * su;
        let mut next = x.get(at(n - 1));
        for i in (0..n - 1).rev() {
            let si = at(i);
            let y = x.get(si) - lf.cp[si] * next;
            x.set(si, y);
            next = y;
        }
        let fct = x.get(at(0)) + lf.r[l] * x.get(at(n - 1));
        for i in 0..n {
            let si = at(i);
            x.set(si, x.get(si) - fct * lf.zs[si]);
        }
    }
}

impl FaSlab for FixedAngleCpu {
    fn tilt(&self) -> usize {
        self.tb.st.u
    }
    fn h_explicit(&mut self, step: usize) -> Result<()> {
        let (u, v, w) = (self.tb.st.u, self.tb.st.v, self.tb.st.w);
        {
            let (_, _, h) = self.inner.plan_and_fields();
            self.hu_old.copy_from_slice(&h[u]);
            self.hq_old[v].copy_from_slice(&h[v]);
            self.hq_old[w].copy_from_slice(&h[w]);
        }
        self.inner.update_h()?;
        self.inner.inject_h(step)
    }
    fn h_coupled(&mut self, step: usize) -> Result<()> {
        if self.tb.s != 0.0 || !self.tb.poles.is_empty() {
            for pair in 0..2 {
                self.pass_f(pair, step);
                self.solve_h(pair);
            }
        }
        Ok(())
    }
    fn e_explicit(&mut self, step: usize) -> Result<()> {
        {
            let (_, e, _) = self.inner.plan_and_fields();
            for a in 0..3 {
                self.e_old[a].copy_from_slice(&e[a]);
            }
        }
        self.inner.update_e()?;
        self.inner.inject_e(step)?;
        self.poles_pre();
        Ok(())
    }
    fn e_coupled(&mut self, step: usize) -> Result<()> {
        for pair in 0..2 {
            self.pass_k(pair, step);
            self.solve_e(pair);
        }
        self.poles_post();
        Ok(())
    }
}

impl Backend for FixedAngleCpu {
    fn name(&self) -> String {
        format!("{}, fixed angle {:.1} deg", self.inner.name(), self.tb.st.angle.to_degrees())
    }
    fn plan(&self) -> &Plan {
        self.inner.plan()
    }
    fn on_thread_pool(&self) -> bool {
        true
    }
    fn fill_ghost_e(&mut self) -> Result<()> {
        self.inner.fill_ghost_e()
    }
    fn fill_ghost_h(&mut self) -> Result<()> {
        self.inner.fill_ghost_h()
    }
    // A single-slab run through the generic time loop: the stages, with the
    // x ghost planes of the new fields refreshed before the coupled updates.
    fn update_h(&mut self) -> Result<()> {
        Ok(())
    }
    fn inject_h(&mut self, step: usize) -> Result<()> {
        self.h_explicit(step)?;
        self.inner.fill_ghost_h()?;
        self.h_coupled(step)
    }
    fn update_e(&mut self) -> Result<()> {
        Ok(())
    }
    fn inject_e(&mut self, step: usize) -> Result<()> {
        self.e_explicit(step)?;
        self.inner.fill_ghost_e()?;
        self.e_coupled(step)
    }
    fn record(&mut self, step: usize) -> Result<()> {
        self.inner.record(step)
    }
    fn energy(&mut self) -> Result<f64> {
        self.inner.energy()
    }
    fn download(&mut self) -> Result<RawResults> {
        let mut raw = self.inner.download()?;
        self.tb.physical_dft(self.inner.plan(), &mut raw);
        Ok(raw)
    }
    fn read_plane(&mut self, comp: Component, side: Side, out: &mut [Real]) -> Result<()> {
        self.inner.read_plane(comp, side, out)
    }
    fn write_ghost(&mut self, comp: Component, side: Side, data: &[Real]) -> Result<()> {
        self.inner.write_ghost(comp, side, data)
    }
    fn field(&mut self, comp: Component) -> Result<Vec<Real>> {
        self.inner.field(comp)
    }
}

/// One x-slab of a fixed-angle run split along x (`x` the propagation axis):
/// the stages of a step, between which [`run_slabs`] exchanges ghost planes.
pub trait FaSlab: Backend {
    /// Save the fields, the zero-angle H update and the H sources.
    fn h_explicit(&mut self, step: usize) -> Result<()>;
    /// The coupled H components (needs the x ghost of H_u at the new time).
    fn h_coupled(&mut self, step: usize) -> Result<()>;
    /// Save the fields, the zero-angle E update, the E sources and the
    /// first pole pass (E_u final).
    fn e_explicit(&mut self, step: usize) -> Result<()>;
    /// The coupled E components (needs the x ghost of E_u at the new time)
    /// and the pole states.
    fn e_coupled(&mut self, step: usize) -> Result<()>;
    /// The tilt axis.
    fn tilt(&self) -> usize;
}

/// Run the x-slabs of a fixed-angle simulation (in x order, `x` the
/// propagation axis). Each step: E ghosts, H stage 1, H_u ghosts, H stage 2,
/// H ghosts, E stage 1, E_u ghosts, E stage 2, monitors; then the results
/// of all slabs are assembled as for any decomposed run.
pub fn run_slabs(slabs: &mut [Box<dyn FaSlab + '_>], opts: &RunOptions) -> Result<Results> {
    use std::time::Instant;
    let p0 = slabs[0].plan().clone();
    let ns = slabs.len();
    let plane = p0.plane();
    let mut buf = vec![0 as Real; plane];
    let exch = |slabs: &mut [Box<dyn FaSlab + '_>], comps: &[Component], e: bool, buf: &mut Vec<Real>| -> Result<()> {
        for &c in comps {
            if e {
                for r in 0..ns - 1 {
                    slabs[r + 1].read_plane(c, Side::Lo, buf)?;
                    slabs[r].write_ghost(c, Side::Hi, buf)?;
                }
            } else {
                for r in 1..ns {
                    slabs[r - 1].read_plane(c, Side::Hi, buf)?;
                    slabs[r].write_ghost(c, Side::Lo, buf)?;
                }
            }
        }
        Ok(())
    };
    // the tilt components (along y or z) are the ones whose x ghosts the
    // coupled stages need
    let tilt = slabs[0].tilt();
    let (eu, hu) = (Component::e(tilt), Component::h(tilt));
    let tang_e = [Component::Ey, Component::Ez];
    let tang_h = [Component::Hy, Component::Hz];
    let t0 = Instant::now();
    let (nsteps, shutoff, interval, t_src_end) = (p0.nsteps, p0.shutoff, p0.shutoff_interval, p0.source_end_time);
    let (mut max_energy, mut source_peak) = (0.0f64, 0.0f64);
    let (mut steps_done, mut stopped) = (nsteps, false);
    let (mut energy_steps, mut energy_trace) = (vec![], vec![]);
    for n in 0..nsteps {
        for s in slabs.iter_mut() {
            s.fill_ghost_e()?;
        }
        exch(slabs, &tang_e, true, &mut buf)?;
        for s in slabs.iter_mut() {
            s.h_explicit(n)?;
            s.fill_ghost_h()?;
        }
        exch(slabs, &[hu], false, &mut buf)?;
        for s in slabs.iter_mut() {
            s.h_coupled(n)?;
            s.fill_ghost_h()?;
        }
        exch(slabs, &tang_h, false, &mut buf)?;
        for s in slabs.iter_mut() {
            s.e_explicit(n)?;
            s.fill_ghost_e()?;
        }
        exch(slabs, &[eu], true, &mut buf)?;
        for s in slabs.iter_mut() {
            s.e_coupled(n)?;
            s.record(n)?;
        }
        if opts.progress_every > 0 && (n + 1) % opts.progress_every == 0 {
            eprintln!("  step {}/{} ({:.1} s)", n + 1, nsteps, t0.elapsed().as_secs_f64());
        }
        if (n + 1) % interval == 0 || n + 1 == nsteps {
            let mut en = 0.0;
            for s in slabs.iter_mut() {
                en += s.energy()?;
            }
            let t = (n + 1) as f64 * p0.dt;
            if !en.is_finite() || (t > t_src_end && source_peak > 0.0 && en > 1e3 * source_peak) {
                bail!("the simulation is unstable: the field energy is not finite or has grown after the sources stopped (step {} of {nsteps}). Results are not returned.", n + 1);
            }
            energy_steps.push(n + 1);
            energy_trace.push(en);
            if t <= t_src_end || source_peak == 0.0 {
                source_peak = source_peak.max(en);
            }
            max_energy = max_energy.max(en);
            if let Some(th) = shutoff {
                if (n + 1) % interval == 0 && t > t_src_end && en <= th * max_energy {
                    steps_done = n + 1;
                    stopped = true;
                    break;
                }
            }
        }
    }
    for s in slabs.iter_mut() {
        s.synchronize()?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let mut raws = Vec::new();
    for s in slabs.iter_mut() {
        raws.push(s.download()?);
    }
    let plans: Vec<Plan> = slabs.iter().map(|s| s.plan().clone()).collect();
    let prefs: Vec<&Plan> = plans.iter().collect();
    let mut res = crate::results::assemble(&prefs, &raws);
    let names: Vec<String> = slabs.iter().map(|s| s.name()).collect();
    res.stats = crate::results::RunStats {
        backend: names.join(" + "),
        cells: p0.grid.num_cells(),
        steps: steps_done,
        seconds: secs,
        mcells_per_s: p0.grid.num_cells() as f64 * steps_done as f64 / secs / 1e6,
        dt: p0.dt,
        stopped_early: stopped,
        energy_steps,
        energy: energy_trace,
    };
    crate::results::finalize(&mut res, &plans[0]);
    Ok(res)
}

/// The CPU backend of a fixed-angle simulation.
pub fn cpu_backend(sim: &Simulation) -> Result<Box<dyn Backend>> {
    Ok(Box::new(FixedAngleCpu::new(sim)?))
}

/// Run a fixed-angle simulation on the CPU.
pub fn run_cpu(sim: &Simulation) -> Result<Results> {
    run_backend(cpu_backend(sim)?, &RunOptions::default())
}

/// Run a fixed-angle simulation on the CPU split into `nslabs` x-slabs
/// (x must be the propagation axis); exercises the decomposed time loop of
/// [`run_slabs`] without GPUs.
pub fn run_cpu_slabs(sim: &Simulation, nslabs: usize) -> Result<Results> {
    let (c, _) = prepare(sim)?;
    let grid = c.build_grid()?;
    let parts = crate::plan::decompose_x(grid.n()[0], &vec![1.0; nslabs]);
    let mut slabs: Vec<Box<dyn FaSlab>> = Vec::new();
    for (r, &(x0, nxl)) in parts.iter().enumerate() {
        let opts = CompileOptions { slab: Some((x0, nxl)), nslabs, slab_index: r, ..Default::default() };
        slabs.push(Box::new(FixedAngleCpu::new_slab(sim, &opts)?));
    }
    if rayon::current_thread_index().is_none() {
        return rayon::scope(|_| run_slabs(&mut slabs, &RunOptions::default()));
    }
    run_slabs(&mut slabs, &RunOptions::default())
}
