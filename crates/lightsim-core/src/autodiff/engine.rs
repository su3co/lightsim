//! The engine interface the gradient drivers are written against.
//!
//! An [`Engine`] is a persistent backend for one device: it owns the fields,
//! the material coefficients and a set of *run configurations* ([`IoSet`]:
//! sources and monitors compiled against the same grid) that can be
//! attached in turn without re-allocating anything. On top of the plain time
//! stepping it offers what gradients need: partial coefficient updates,
//! DFT fields that stay on the device, the adjoint product, and the
//! primitives of the reverse pass.

use crate::Real;
use super::design::IndexBox;
use crate::engine::{Backend, RawDft, RawTime};
use crate::grid::Component;
use crate::plan::{DftPlan, Injection, Plan, TimePlan};
use crate::sources::Source;
use anyhow::{Result, bail};
use num_complex::Complex64;
use std::time::Instant;

/// Sources and monitors of one run, compiled against the session's grid.
#[derive(Clone, Debug, Default)]
pub struct IoSet {
    pub inj_h: Vec<Injection>,
    pub inj_e: Vec<Injection>,
    pub dft: Vec<DftPlan>,
    pub time: Vec<TimePlan>,
    pub sources: Vec<Source>,
    pub source_end_time: f64,
}

impl IoSet {
    /// Move the sources and monitors out of a compiled plan.
    pub fn take(plan: &mut Plan) -> IoSet {
        IoSet {
            inj_h: std::mem::take(&mut plan.inj_h),
            inj_e: std::mem::take(&mut plan.inj_e),
            dft: std::mem::take(&mut plan.dft),
            time: std::mem::take(&mut plan.time),
            sources: std::mem::take(&mut plan.sources),
            source_end_time: std::mem::replace(&mut plan.source_end_time, 0.0),
        }
    }
    /// Exchange the sources and monitors of `plan` with this set.
    pub fn swap(&mut self, plan: &mut Plan) {
        std::mem::swap(&mut self.inj_h, &mut plan.inj_h);
        std::mem::swap(&mut self.inj_e, &mut plan.inj_e);
        std::mem::swap(&mut self.dft, &mut plan.dft);
        std::mem::swap(&mut self.time, &mut plan.time);
        std::mem::swap(&mut self.sources, &mut plan.sources);
        std::mem::swap(&mut self.source_end_time, &mut plan.source_end_time);
    }
}

/// Outcome of a run.
#[derive(Clone, Debug, Default)]
pub struct RunInfo {
    pub steps: usize,
    pub seconds: f64,
    pub stopped_early: bool,
}

/// A cotangent source of the reverse pass: at reverse step `n`,
/// `field[idx[t]] += coef[t] * table[n * idx.len() + t]` on the scaled
/// adjoint field of component `comp`.
#[derive(Clone, Debug)]
pub struct AdjointSource {
    pub comp: Component,
    /// Storage indices.
    pub idx: Vec<u64>,
    pub coef: Vec<Real>,
    /// `[step][entry]`.
    pub table: Vec<Real>,
}

/// What the reverse pass needs besides the attached run configuration.
#[derive(Clone, Debug, Default)]
pub struct ReverseConfig {
    /// Time monitor (of the attached configuration) that recorded, every
    /// step, the fields of the PML cells next to the interior: H boxes are
    /// restored before the E reversal, E boxes before the H reversal.
    pub shell: Option<usize>,
    pub sources: Vec<AdjointSource>,
    /// Per E component: box over which `dL/d(coefficients)` is accumulated.
    pub design: [IndexBox; 3],
    /// Per H component: the same for permeability designs (`dL/d ch`).
    pub design_h: [IndexBox; 3],
    /// Accumulate the derivatives with respect to the ADE coefficients of the
    /// dispersive cells ([`ReverseOut::poles`]).
    pub poles: bool,
}

/// Accumulated sums of the reverse pass, per E component over the design
/// boxes of [`ReverseConfig`] (x-major): with `a^` the scaled adjoint field,
/// `p1 = sum_n a^(n+1) (E'(n+1) - ca E(n))` (E' before the source injection)
/// and `p2 = sum_n a^(n+1) E(n)` (lossy plans only, else empty).
#[derive(Clone, Debug, Default)]
pub struct ReverseOut {
    pub p1: [Vec<f64>; 3],
    pub p2: [Vec<f64>; 3],
    /// Per H component over `design_h`: `q1 = sum_n b^(n+1/2) (H'(n+1/2) -
    /// H(n-1/2))` with `b^` the scaled adjoint of H'(n+1/2) (H' before the
    /// source injection; the design cells have no magnetic loss).
    pub q1: [Vec<f64>; 3],
    /// With [`ReverseConfig::poles`]: per dispersive component of the plan
    /// (`plan.disp.comps`), `dL/d(c0, c1, c2)` of every pole of every cell,
    /// `[pole][coefficient][entry]`, for the cotangents as given (the caller
    /// rescales). The ADE step is `P(n+1) = c0 P(n) + c1 P(n-1) + c2 E(n)`.
    pub poles: Vec<Vec<f64>>,
}

/// Persistent single-device backend for gradient computations.
pub trait Engine: Send {
    fn name(&self) -> String;
    /// The plan (host copy, with the current material coefficients and the
    /// sources/monitors of the attached configuration).
    fn plan(&self) -> &Plan;
    /// Register a run configuration; returns its index.
    fn add_io(&mut self, io: IoSet) -> Result<usize>;

    /// Overwrite the E-update coefficients of component `a` over `bx`
    /// (box layout, x-major).
    fn write_coefficients(&mut self, a: usize, bx: IndexBox, cb: &[Real], ca: Option<&[Real]>) -> Result<()>;
    /// Overwrite the H-update coefficients `ch` of component `a` over `bx`
    /// (permeability designs).
    fn write_h_coefficients(&mut self, _a: usize, _bx: IndexBox, _ch: &[Real]) -> Result<()> {
        anyhow::bail!("this engine does not support permeability designs")
    }
    /// Called once after the `write_coefficients` of a design update.
    fn commit_coefficients(&mut self) -> Result<()> {
        Ok(())
    }

    /// Attach configuration `io` and zero the fields, the auxiliary state
    /// and that configuration's monitors.
    fn begin(&mut self, io: usize) -> Result<()>;
    /// One leapfrog step with sources and monitors.
    fn step(&mut self, n: usize) -> Result<()>;
    /// Sum of squared fields over the interior (shutoff criterion).
    fn energy(&mut self) -> Result<f64>;
    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
    /// True if the engine computes on the rayon thread pool (see
    /// [`Backend::on_thread_pool`]).
    fn on_thread_pool(&self) -> bool {
        false
    }

    /// DFT monitor `monitor` of configuration `io` after its last run.
    fn read_dft(&mut self, io: usize, monitor: usize) -> Result<RawDft>;
    /// Overwrite it (fields computed on another device).
    fn write_dft(&mut self, io: usize, monitor: usize, data: &RawDft) -> Result<()>;
    /// Time monitor `monitor` of configuration `io` after its last run.
    fn read_time(&mut self, io: usize, monitor: usize) -> Result<RawTime>;

    /// Start accumulating products shaped like the boxes of DFT monitor
    /// `like = (io, monitor)`.
    fn product_begin(&mut self, like: (usize, usize)) -> Result<()>;
    /// For every box b and frequency k:
    /// `acc_b[cell] += Re(coef[k] * A_k[cell] * B_k[cell])` with A, B the
    /// DFT fields of monitors `a` and `b` (same boxes and frequencies).
    fn product_add(&mut self, a: (usize, usize), b: (usize, usize), coef: &[Complex64]) -> Result<()>;
    /// The accumulated products, one array per box.
    fn product_end(&mut self) -> Result<Vec<Vec<Real>>>;

    /// Called once, before any run, by drivers that will use the reverse
    /// pass (lets an engine pick a stepping mode that can be reversed).
    fn prepare_reverse(&mut self) -> Result<()> {
        Ok(())
    }
    /// Start the reverse pass from the state left by the forward run of the
    /// attached configuration.
    fn reverse_begin(&mut self, _cfg: ReverseConfig) -> Result<()> {
        bail!("reverse mode is not supported by this engine")
    }
    /// Undo forward step `n` (E(n+1), H(n+1/2) -> E(n), H(n-1/2)), advance
    /// the adjoint fields across it and accumulate the coefficient sums.
    fn reverse_step(&mut self, _n: usize) -> Result<()> {
        bail!("reverse mode is not supported by this engine")
    }
    fn reverse_end(&mut self) -> Result<ReverseOut> {
        bail!("reverse mode is not supported by this engine")
    }
    /// Forward (`adjoint = false`) or scaled adjoint field (debug / tests).
    fn reverse_field(&mut self, _c: Component, _adjoint: bool) -> Result<Vec<Real>> {
        bail!("reverse mode is not supported by this engine")
    }
}

/// Creates an engine for a device string (`"cpu"`, `"cuda:0"`, ...) from a
/// compiled plan (without sources and monitors).
pub type EngineFactory<'a> = dyn Fn(&str, Plan) -> Result<Box<dyn Engine>> + Sync + 'a;

/// The CPU engine factory.
pub fn cpu_factory(device: &str, plan: Plan) -> Result<Box<dyn Engine>> {
    if device != "cpu" {
        bail!("device '{device}' is not available in this build (CPU only)");
    }
    Ok(Box::new(super::cpu::CpuEngine::new(plan)?))
}

/// One leapfrog step with sources and monitors on a single-slab backend
/// (the order of [`crate::run::Chain::run`]).
pub fn forward_step(be: &mut dyn Backend, n: usize) -> Result<()> {
    be.fill_ghost_e()?;
    be.update_h()?;
    be.inject_h(n)?;
    be.fill_ghost_h()?;
    be.update_e()?;
    be.finish_e()?;
    be.inject_e(n)?;
    be.record(n)
}

/// Run the attached configuration of `eng` to the end, with the plan's
/// energy-decay shutoff when `shutoff` is set (the criterion of
/// [`crate::run::Chain::run`]).
pub fn run_to_end(eng: &mut dyn Engine, io: usize, shutoff: bool) -> Result<RunInfo> {
    // CPU engines: step from a thread of the pool they compute on
    if rayon::current_thread_index().is_none() && eng.on_thread_pool() {
        return rayon::scope(|_| run_to_end(eng, io, shutoff));
    }
    eng.begin(io)?;
    let (nsteps, th, interval, t_src_end, dt) = {
        let p = eng.plan();
        (p.nsteps, p.shutoff, p.shutoff_interval, p.source_end_time, p.dt)
    };
    let th = if shutoff { th } else { None };
    let t0 = Instant::now();
    let mut max_energy = 0.0f64;
    let mut steps = nsteps;
    let mut stopped = false;
    for n in 0..nsteps {
        eng.step(n)?;
        if let Some(th) = th {
            if (n + 1) % interval == 0 {
                let en = eng.energy()?;
                max_energy = max_energy.max(en);
                if (n + 1) as f64 * dt > t_src_end && en <= th * max_energy {
                    steps = n + 1;
                    stopped = true;
                    break;
                }
            }
        }
    }
    eng.synchronize()?;
    Ok(RunInfo { steps, seconds: t0.elapsed().as_secs_f64(), stopped_early: stopped })
}
