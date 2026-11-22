//! Discrete reverse-mode differentiation through the time loop.
//!
//! For an objective L of the time-domain fields recorded by a monitor, the
//! reverse pass walks the steps backwards. At every step it
//!
//! * reconstructs the forward fields by exact time reversal (the leapfrog
//!   with inverted coefficients; the one-cell shell of PML cells next to
//!   the interior, recorded on the forward run, is replayed, so nothing of
//!   the volume is stored),
//! * advances the adjoint fields with the transposed update (the forward
//!   step on the scaled adjoint variables, with the transposed CPML in the
//!   absorbing layers; see the module documentation), driven by the
//!   cotangents dL/dE, dL/dH of the recorded samples,
//! * accumulates `dL/d cb = sum_n a(n+1) curl-term(n)` (and `dL/d ca` in
//!   lossy media) over the design box.
//!
//! The memory is that of two field sets plus the shell and monitor
//! recordings (surfaces, not volumes). The result is the gradient of the
//! *discrete* objective: it agrees with finite differences of the same
//! simulation to the accuracy single precision allows.

use super::design::{DesignGrid, DesignMap, IndexBox, update_volume};
use super::engine::{AdjointSource, Engine, EngineFactory, IoSet, ReverseConfig};
use super::session::{base_simulation, check_sources_outside, session_options};
use crate::constants::ETA0;
use crate::grid::{Component, Grid};
use crate::monitors::{Apodization, Monitor};
use crate::plan::{CompBox, Plan, TimePlan, compile_with_grid};
use crate::spec::Simulation;
use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A time-domain objective.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Objective {
    /// Energy (J) crossing the plane `axis = position` (restricted to
    /// `[min, max]`) towards `+axis`, weighted by a time window:
    /// `L = sum_n w(t_n+1/2) dt sum_cells dA (E x H)_axis`, with E averaged
    /// between the two time levels around the H sample and H between the
    /// two planes around the E plane. The window is 1 on `[t_start, t_end]`
    /// (defaults: the whole run) with raised-cosine edges of width `taper`.
    Flux {
        axis: usize,
        position: f64,
        min: [f64; 3],
        max: [f64; 3],
        #[serde(default)]
        t_start: Option<f64>,
        #[serde(default)]
        t_end: Option<f64>,
        #[serde(default)]
        taper: Option<f64>,
    },
    /// Any function of the samples of the given components in a box,
    /// evaluated by the caller: [`ReverseSession::series`] returns the
    /// samples and [`ReverseSession::backward_with`] takes the cotangents.
    Fields { min: [f64; 3], max: [f64; 3], components: Vec<Component> },
}

/// What a reverse-mode session is built from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReverseSpec {
    /// The simulation with its sources (monitors are ignored); its media
    /// must contain the design media.
    pub sim: Simulation,
    pub grids: Vec<DesignGrid>,
    pub objective: Objective,
}

/// Samples of one component of the objective monitor.
#[derive(Clone, Debug)]
pub struct Series {
    pub comp: Component,
    /// Global index of the first cell and the box size.
    pub origin: [usize; 3],
    pub size: [usize; 3],
    /// `[sample][i][j][k]`: E components at `(n + 1) dt`, H at `(n + 1/2) dt`.
    pub data: Vec<f32>,
}

/// Timings of the last forward / backward pass.
#[derive(Clone, Debug, Default)]
pub struct ReverseStats {
    pub forward_seconds: f64,
    pub reverse_seconds: f64,
    pub steps: usize,
    pub cells: usize,
    /// Floats recorded per step for the PML shell and the objective monitor.
    pub shell_floats_per_step: usize,
    pub monitor_floats_per_step: usize,
    /// RMS of the forward fields left after the last reverse pass, relative
    /// to the largest field energy of the forward run (0 for an exact reversal).
    pub reconstruction_residual: f64,
}

pub struct ReverseSession {
    objective: Objective,
    grid: Grid,
    periodic: [bool; 3],
    design: DesignMap,
    master: Plan,
    engine: Box<dyn Engine>,
    io: usize,
    obj_monitor: usize,
    shell: Option<usize>,
    obj_boxes: Vec<CompBox>,
    region: ([f64; 3], [f64; 3]),
    nsteps: usize,
    design_set: bool,
    ran: bool,
    series: Vec<Vec<f32>>,
    /// Cotangents per objective box, `[step][cell]` (set by the built-in
    /// objective or by the caller).
    cotangent: Option<Vec<Vec<f64>>>,
    value: f64,
    /// Largest field energy (engine measure) seen on the forward run.
    peak_energy: f64,
    pub stats: ReverseStats,
}

fn window(t: f64, t0: Option<f64>, t1: Option<f64>, taper: Option<f64>) -> f64 {
    let w = taper.unwrap_or(0.0);
    let mut v = 1.0;
    if let Some(a) = t0 {
        if t < a {
            return 0.0;
        }
        if w > 0.0 && t < a + w {
            v *= 0.5 * (1.0 - (std::f64::consts::PI * (t - a) / w).cos());
        }
    }
    if let Some(b) = t1 {
        if t > b {
            return 0.0;
        }
        if w > 0.0 && t > b - w {
            v *= 0.5 * (1.0 - (std::f64::consts::PI * (b - t) / w).cos());
        }
    }
    v
}

/// Transverse area element of component `c` at global index `g` on a plane
/// normal to `a`, clipped to the monitor region (as in the flux monitors).
fn face_area(grid: &Grid, periodic: [bool; 3], c: Component, a: usize, g: [usize; 3], region: &([f64; 3], [f64; 3])) -> f64 {
    let tol = 1e-6 * grid.min_cell_size().iter().cloned().fold(f64::INFINITY, f64::min);
    let mut s = 1.0;
    for ax in 0..3 {
        if ax == a {
            continue;
        }
        let n = grid.n()[ax];
        let w = if periodic[ax] && !c.is_half(ax) && g[ax] == 0 {
            0.5 * (grid.cell_size(ax, 0) + grid.cell_size(ax, n - 1))
        } else {
            let (lo, hi) = grid.voxel_bounds(c, ax, g[ax]);
            (hi.min(region.1[ax] + tol) - lo.max(region.0[ax] - tol)).max(0.0)
        };
        s *= w;
    }
    s
}

/// The one-cell shell of PML cells next to the interior: per PML slab, the
/// tangential H components on the last layer of a low-side slab (read by
/// the E reversal of the first interior cells) and the tangential E
/// components on the first layer of a high-side slab (read by the H
/// reversal of the last interior cells).
pub fn shell_boxes(plan: &Plan) -> Vec<CompBox> {
    let n = plan.n;
    let mut out = vec![];
    for p in &plan.pml {
        let a = p.axis;
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let (layer, comps) = if p.start[a] == 0 {
            (p.size[a] - 1, [Component::h(b), Component::h(c)])
        } else if p.start[a] + p.size[a] == n[a] {
            (p.start[a], [Component::e(b), Component::e(c)])
        } else {
            continue;
        };
        for comp in comps {
            let mut glo = [0usize; 3];
            let mut ghi = n;
            glo[a] = layer;
            ghi[a] = layer + 1;
            out.push(CompBox { comp, glo, ghi, lstart: glo, lsize: [ghi[0] - glo[0], ghi[1] - glo[1], ghi[2] - glo[2]], gx_off: 0 });
        }
    }
    out
}

impl ReverseSession {
    pub fn new(spec: ReverseSpec, device: &str, factory: &EngineFactory) -> Result<ReverseSession> {
        let mut sim = base_simulation(&spec.sim, &spec.grids)?;
        sim.sources = spec.sim.sources.clone();
        if sim.sources.is_empty() {
            bail!("the simulation has no source");
        }
        sim.monitors = vec![match &spec.objective {
            Objective::Flux { axis, position, min, max, .. } => Monitor::Flux { name: "__obj__".into(), axis: *axis, position: *position, min: *min, max: *max, freqs: vec![1.0], apodization: Apodization::default() },
            Objective::Fields { min, max, components } => {
                if components.is_empty() {
                    bail!("the objective needs at least one field component");
                }
                Monitor::FieldTime { name: "__obj__".into(), min: *min, max: *max, components: components.clone(), interval: 1, start: 0.0 }
            }
        }];
        let grid = sim.build_grid()?;
        let b = &sim.boundaries;
        let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
        let mut plan = compile_with_grid(&sim, grid.clone(), &session_options())?;
        if plan.bloch || plan.tensor.is_some() {
            bail!("reverse mode does not support Bloch boundaries or off-diagonal permittivity");
        }
        let mut io = IoSet::take(&mut plan);
        // the objective monitor as a time monitor sampled every step
        let (obj_boxes, region) = match &spec.objective {
            Objective::Flux { .. } => {
                let d = io.dft.remove(0);
                (d.boxes, d.region)
            }
            Objective::Fields { min, max, .. } => (io.time.remove(0).boxes, (*min, *max)),
        };
        io.dft.clear();
        io.time.clear();
        io.time.push(TimePlan { name: "__obj__".into(), boxes: obj_boxes.clone(), interval: 1, start_step: 0, is_probe: false });
        let shell_b = shell_boxes(&plan);
        let shell = if shell_b.is_empty() {
            None
        } else {
            io.time.push(TimePlan { name: "__shell__".into(), boxes: shell_b, interval: 1, start_step: 0, is_probe: false });
            Some(1)
        };
        let design = DesignMap::new(&plan, &spec.grids)?;
        check_sources_outside(&io, &design, &plan)?;
        let stats = ReverseStats {
            steps: plan.nsteps,
            cells: grid.num_cells(),
            shell_floats_per_step: shell.map(|s| io.time[s].boxes.iter().map(|b| b.lcells()).sum()).unwrap_or(0),
            monitor_floats_per_step: obj_boxes.iter().map(|b| b.lcells()).sum(),
            ..Default::default()
        };
        let nsteps = plan.nsteps;
        let mut engine = factory(device, plan.clone())?;
        engine.prepare_reverse()?;
        let io = engine.add_io(io)?;
        Ok(ReverseSession {
            objective: spec.objective,
            grid,
            periodic,
            design,
            master: plan,
            engine,
            io,
            obj_monitor: 0,
            shell,
            obj_boxes,
            region,
            nsteps,
            design_set: false,
            ran: false,
            series: vec![],
            cotangent: None,
            value: 0.0,
            peak_energy: 0.0,
            stats,
        })
    }

    pub fn grids(&self) -> &[DesignGrid] {
        &self.design.grids
    }
    pub fn device(&self) -> String {
        self.engine.name()
    }
    pub fn steps(&self) -> usize {
        self.nsteps
    }
    pub fn dt(&self) -> f64 {
        self.master.dt
    }
    pub fn design(&self) -> &DesignMap {
        &self.design
    }

    /// Set the index-grid samples (one x-major array per grid).
    pub fn set_design(&mut self, eps: &[Vec<f64>]) -> Result<()> {
        self.design.set_eps(eps)?;
        for b in self.design.write_plan(&mut self.master) {
            self.engine.write_coefficients(b.axis, b.bx, &b.cb, b.ca.as_deref())?;
        }
        self.engine.commit_coefficients()?;
        self.design_set = true;
        self.ran = false;
        self.cotangent = None;
        Ok(())
    }

    /// Run the forward simulation, recording the objective monitor (and the
    /// PML shell). Returns the objective for the built-in ones (0 for
    /// `Fields`, whose value the caller computes from [`series`](Self::series)).
    pub fn forward(&mut self) -> Result<f64> {
        if !self.design_set {
            bail!("set_design must be called before forward");
        }
        let t0 = Instant::now();
        self.engine.begin(self.io)?;
        // the field energy at 64 points of the run: the scale of the check
        // that the reverse pass returns to the initial state
        let every = (self.nsteps / 64).max(1);
        self.peak_energy = 0.0;
        for n in 0..self.nsteps {
            self.engine.step(n)?;
            if (n + 1) % every == 0 || n + 1 == self.nsteps {
                self.peak_energy = self.peak_energy.max(self.engine.energy()?);
            }
        }
        self.engine.synchronize()?;
        self.series = self.engine.read_time(self.io, self.obj_monitor)?.boxes;
        self.stats.forward_seconds = t0.elapsed().as_secs_f64();
        self.ran = true;
        self.cotangent = None;
        self.value = 0.0;
        if let Objective::Flux { axis, t_start, t_end, taper, .. } = self.objective.clone() {
            self.flux(axis, t_start, t_end, taper)?;
        }
        Ok(self.value)
    }

    /// The recorded samples of the objective monitor, one entry per box
    /// (for `Fields`: per component, in the order given).
    pub fn series(&self) -> Result<Vec<Series>> {
        if !self.ran {
            bail!("forward has not been run for the current design");
        }
        Ok(self.obj_boxes.iter().zip(&self.series).map(|(b, d)| Series { comp: b.comp, origin: b.glo, size: b.gsize(), data: d.clone() }).collect())
    }

    /// Value and cotangents of the windowed flux objective.
    fn flux(&mut self, a: usize, t0: Option<f64>, t1: Option<f64>, taper: Option<f64>) -> Result<()> {
        let n = self.nsteps;
        let dt = self.master.dt;
        let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
        let w: Vec<f64> = (0..n).map(|s| window((s as f64 + 0.5) * dt, t0, t1, taper)).collect();
        let mut cot: Vec<Vec<f64>> = self.obj_boxes.iter().map(|b| vec![0.0; b.lcells() * n]).collect();
        let mut value = 0.0;
        // boxes: [E_b(p), E_c(p), H_b(p-1), H_c(p-1), H_b(p), H_c(p)]
        for (ei, h0, h1, sign, comp) in [(0usize, 3usize, 5usize, 1.0, Component::e(bax)), (1, 2, 4, -1.0, Component::e(cax))] {
            let bx = &self.obj_boxes[ei];
            let m = bx.lcells();
            if self.obj_boxes[h0].lcells() != m || self.obj_boxes[h1].lcells() != m {
                bail!("flux objective: the E and H boxes of the plane differ in size");
            }
            let s3 = bx.gsize();
            let area: Vec<f64> = (0..m)
                .map(|t| {
                    let g = [bx.glo[0] + t / (s3[1] * s3[2]), bx.glo[1] + (t / s3[2]) % s3[1], bx.glo[2] + t % s3[2]];
                    sign * face_area(&self.grid, self.periodic, comp, a, g, &self.region) / ETA0 * dt
                })
                .collect();
            let (e, ha, hb) = (&self.series[ei], &self.series[h0], &self.series[h1]);
            // E averaged over the two time levels around the H sample, H over the two planes
            let ebar = |s: usize, t: usize| -> f64 { 0.5 * (e[s * m + t] as f64 + if s > 0 { e[(s - 1) * m + t] as f64 } else { 0.0 }) };
            let hbar = |s: usize, t: usize| -> f64 { 0.5 * (ha[s * m + t] as f64 + hb[s * m + t] as f64) };
            let mut ce = vec![0.0; m * n];
            let mut ch = vec![0.0; m * n];
            value += ce
                .par_chunks_mut(m)
                .zip(ch.par_chunks_mut(m))
                .enumerate()
                .map(|(s, (ce, ch))| {
                    let mut v = 0.0;
                    for t in 0..m {
                        let (eb, hv) = (ebar(s, t), hbar(s, t));
                        v += w[s] * area[t] * eb * hv;
                        // E(s+1) enters the averages of samples s and s + 1
                        ce[t] = 0.5 * area[t] * (w[s] * hv + if s + 1 < n { w[s + 1] * hbar(s + 1, t) } else { 0.0 });
                        ch[t] = 0.5 * area[t] * w[s] * eb;
                    }
                    v
                })
                // per-step values in step order, summed in order: the same
                // value on every run, whatever the thread scheduling
                .collect::<Vec<f64>>()
                .iter()
                .sum::<f64>();
            cot[ei] = ce;
            cot[h0] = ch.clone();
            cot[h1] = ch;
        }
        self.value = value;
        self.cotangent = Some(cot);
        Ok(())
    }

    /// Reverse pass with the cotangents of the built-in objective.
    pub fn backward(&mut self) -> Result<Vec<Vec<f64>>> {
        match self.cotangent.take() {
            Some(c) => {
                let r = self.backward_with(&c);
                self.cotangent = Some(c);
                r
            }
            None => bail!("no cotangents: run forward (built-in objectives) or use backward_with"),
        }
    }

    /// Reverse pass for the cotangents `dL/d(sample)` of the objective
    /// monitor (one `[step][cell]` array per box, shaped like
    /// [`series`](Self::series)). Returns dL/d(eps) on the samples of every
    /// design grid.
    pub fn backward_with(&mut self, cot: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.ran {
            bail!("forward has not been run for the current design");
        }
        let n = self.nsteps;
        if cot.len() != self.obj_boxes.len() {
            bail!("expected {} cotangent arrays, got {}", self.obj_boxes.len(), cot.len());
        }
        let t0 = Instant::now();
        let plan = &self.master;
        let (ny, nz) = (plan.ny(), plan.nz());
        // scaled cotangent sources: E: cb/V_E dL/dE, H: -ch/V_H dL/dH, in
        // double precision, normalised to a peak of 1 (the tiny raw values
        // would underflow single precision); the gradient is rescaled
        let mut tables: Vec<Vec<f64>> = vec![];
        let mut peak = 0.0f64;
        for (b, c) in self.obj_boxes.iter().zip(cot) {
            let m = b.lcells();
            if c.len() != m * n {
                bail!("cotangent of {} has {} entries, expected {} x {}", b.comp.name(), c.len(), n, m);
            }
            let bx = IndexBox { lo: b.lstart, size: b.lsize };
            let a = b.comp.axis();
            let coef: Vec<f64> = (0..m)
                .map(|t| {
                    let s = bx.storage(t, ny, nz);
                    let g = [bx.lo[0] + t / (bx.size[1] * bx.size[2]), bx.lo[1] + (t / bx.size[2]) % bx.size[1], bx.lo[2] + t % bx.size[2]];
                    let v = update_volume(plan, b.comp, g);
                    if b.comp.is_e() { plan.cb[a][s] as f64 / v } else { -(plan.ch.as_ref().map(|c| c[a][s]).unwrap_or(plan.ch_scalar) as f64) / v }
                })
                .collect();
            let tab: Vec<f64> = c.par_chunks(m).flat_map_iter(|row| row.iter().zip(&coef).map(|(v, k)| k * v)).collect();
            peak = peak.max(tab.par_iter().map(|v| v.abs()).reduce(|| 0.0, f64::max));
            tables.push(tab);
        }
        if peak == 0.0 {
            return Ok(self.design.grids.iter().map(|g| vec![0.0; g.samples()]).collect());
        }
        let sources: Vec<AdjointSource> = self
            .obj_boxes
            .iter()
            .zip(&tables)
            .map(|(b, tab)| {
                let bx = IndexBox { lo: b.lstart, size: b.lsize };
                AdjointSource {
                    comp: b.comp,
                    idx: (0..b.lcells()).map(|t| bx.storage(t, ny, nz) as u64).collect(),
                    coef: vec![1.0; b.lcells()],
                    table: tab.par_iter().map(|v| (v / peak) as f32).collect(),
                }
            })
            .collect();
        self.engine.reverse_begin(ReverseConfig { shell: self.shell, sources, design: self.design.bbox })?;
        for s in (0..n).rev() {
            self.engine.reverse_step(s)?;
        }
        let out = self.engine.reverse_end()?;
        self.check_reconstruction(&out)?;
        // dL/d eps = dL/d cb d cb/d eps + dL/d ca d ca/d eps on the owned voxels
        let plan = &self.master;
        let mut grads: Vec<Vec<f64>> = self.design.grids.iter().map(|g| vec![0.0; g.samples()]).collect();
        for a in 0..3 {
            let bx = self.design.bbox[a];
            if bx.is_empty() {
                continue;
            }
            let mut yee = vec![0.0; bx.cells()];
            self.design.for_each_owned(a, |g, i, j, k, _| {
                let t = bx.offset(i, j, k);
                let s = ((i + 1) * ny + j) * nz + k;
                let cb = plan.cb[a][s] as f64;
                if cb == 0.0 {
                    return;
                }
                // the permittivity the engine uses is the single-precision one
                let eps = self.design.yee_eps_at(g, a, i, j, k) as f64;
                let v = update_volume(plan, Component::e(a), [i, j, k]);
                let (dca, dcb) = self.design.coefficient_derivatives(plan, g, a, eps);
                let mut d = v / (cb * cb) * out.p1[a][t] * dcb;
                if let Some(p2) = out.p2[a].get(t) {
                    d += v / cb * p2 * dca;
                }
                yee[t] = d * peak;
            });
            self.design.pullback(a, &bx, &yee, None, &mut grads);
        }
        self.stats.reverse_seconds = t0.elapsed().as_secs_f64();
        if grads.iter().flatten().any(|g| !g.is_finite()) {
            bail!("{}", Self::DIVERGED);
        }
        Ok(grads)
    }

    const DIVERGED: &'static str = "the reverse pass diverged: reconstructing the forward fields by running the scheme backwards amplifies round-off by exp(sigma t / eps) in absorbing media (strong conductors, damped or Debye poles) over long runs. Shorten the run or use the S-matrix session (lightsim.autodiff.GradientSession / invdes.SMatrixProblem), whose adjoint is a second forward run";

    /// After the reverse pass the forward fields must be back at the initial
    /// (zero) state outside the absorbing layers, which are not reconstructed.
    /// The residual (RMS over the volume, relative to the largest field of
    /// the forward run) is 3e-7 ... 9e-7 when the reversal is exact and 2e-3
    /// with a 1.5e4 S/m conductor whose gradient is still right to 1e-6; with
    /// 2e4 ... 3e4 S/m it is 1e3 ... 3e9 and the gradient is off by 2e-3 to
    /// 1e4 (relative). Above 0.1, or non-finite, the pass is refused.
    fn check_reconstruction(&mut self, out: &super::engine::ReverseOut) -> Result<()> {
        if out.p1.iter().chain(out.p2.iter()).flatten().any(|v| !v.is_finite()) {
            bail!("{}", Self::DIVERGED);
        }
        let plan = &self.master;
        let len = plan.storage_len();
        let (plane, nz) = (plan.plane(), plan.nz());
        let interior = |s: usize| -> bool {
            let p = s / plane;
            if p == 0 || p > plan.nxl {
                return false;
            }
            let g = [p - 1, (s % plane) / nz, s % nz];
            !plan.pml.iter().any(|q| (0..3).all(|a| g[a] >= q.start[a] && g[a] < q.start[a] + q.size[a]))
        };
        let mut residual = 0.0f64;
        for c in Component::ALL {
            let f = self.engine.reverse_field(c, false)?;
            if f.len() != len {
                return Ok(()); // a layout this check does not know (2-D storage)
            }
            residual += f.par_iter().enumerate().filter(|(s, _)| interior(*s)).map(|(_, v)| (*v as f64) * (*v as f64)).collect::<Vec<f64>>().iter().sum::<f64>();
        }
        let rel = (residual / self.peak_energy).sqrt();
        self.stats.reconstruction_residual = rel;
        if !rel.is_finite() || rel > 0.1 {
            bail!("{} (fields left after the reverse pass: {rel:.1e} of the forward peak)", Self::DIVERGED);
        }
        Ok(())
    }

    /// Forward and reverse pass for the built-in objective.
    pub fn value_and_grad(&mut self) -> Result<(f64, Vec<Vec<f64>>)> {
        let v = self.forward()?;
        Ok((v, self.backward()?))
    }
}
