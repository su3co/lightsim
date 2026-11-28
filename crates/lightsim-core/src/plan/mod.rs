//! Compilation of a [`Simulation`] into a backend-independent execution
//! [`Plan`]: coefficient arrays in storage layout, boundary codes, CPML slabs,
//! source injection tables, dispersive-pole tables and monitor tables.
//!
//! A plan can cover the whole domain or one x-slab `[x0, x0 + nxl)` of it
//! (multi-GPU / multi-node domain decomposition). All indices stored in the
//! plan are local storage indices of that slab.
//!
//! [`compile_with_grid`] runs the phases in order, one module each:
//!
//! * [`Setup`] — the slab, time step, media, raster options and grid spacings
//!   every phase reads;
//! * `layers` — stabilised absorbing layers (poles damped or frozen inside a
//!   PML), adiabatic absorbers and CPML slabs;
//! * `materials` — the update coefficients of E and H: rasterization with
//!   sub-pixel smoothing, the layers and absorbers, tensor and Hermitian
//!   media, wall conditions;
//! * `boundaries` — the kernels' boundary codes, Bloch phases and x ghost rules;
//! * `cell_lists` — the cells of dispersive, nonlinear, Raman, gain and
//!   off-diagonal media;
//! * `sources` and `monitors` — injection tables and monitor tables.
//!
//! The data types of the plan are in `types` and re-exported here.

use crate::Real;
use crate::constants::{C0, EPS0, ETA0, TAU};
use crate::grid::{Component, Grid};
use crate::materials::Medium;
use crate::monitors::{Apodization, Monitor};
use crate::raster::{RasterOptions, ResolvedStructure, rasterize_component, resolve};
use crate::sources::{Direction, Source, TimeSignal};
use crate::spec::{Boundary, Simulation};
use anyhow::{Result, bail};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::BTreeMap;

mod boundaries;
mod cell_lists;
mod coefficients;
mod layers;
mod materials;
mod monitors;
mod sources;
mod types;
#[cfg(test)]
mod tests;

pub use coefficients::*;
pub use layers::{ABSORBER_ATTENUATION, SourceBand, pml_pole_variant, source_band};
pub use monitors::auto_dft_stride;
pub use sources::{source_sets, transverse_box};
pub use types::*;

/// Global x-range decomposition into `parts` slabs with sizes proportional
/// to `weights` (e.g. GPU memory/bandwidth), each at least 4 planes.
pub fn decompose_x(nx: usize, weights: &[f64]) -> Vec<(usize, usize)> {
    let parts = weights.len().max(1);
    let total: f64 = weights.iter().sum::<f64>().max(1e-30);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0usize;
    let mut acc = 0.0;
    for (p, w) in weights.iter().enumerate() {
        acc += w;
        let end = if p + 1 == parts { nx } else { ((acc / total) * nx as f64).round() as usize };
        let end = end.max(start + 1).min(nx);
        out.push((start, end - start));
        start = end;
    }
    out
}

/// Compile options.
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// x-slab `(x0, nxl)`; `None` = whole domain.
    pub slab: Option<(usize, usize)>,
    /// Number of slabs in the decomposition (for ghost rules).
    pub nslabs: usize,
    pub slab_index: usize,
    pub raster: RasterOptions,
    /// Override the number of steps (benchmarks).
    pub nsteps: Option<usize>,
    /// Let frequency-domain monitors that record the same component box with
    /// the same frequencies, time window and stride share one accumulation
    /// (see [`DftPlan::shared`]). On by default. Off for engines that read
    /// the accumulators of a monitor directly instead of through
    /// [`crate::results::assemble`], as the gradient sessions do with the
    /// design-region fields.
    pub share_monitors: bool,
    /// Let one-cell periodic grids use the specialised 2-D update
    /// ([`Plan::twod`]); `false` keeps them on the general 3-D one (also
    /// forced by `LIGHTSIM_2D=0` and by `Simulation::force_3d_kernels`).
    pub allow_2d: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions { slab: None, nslabs: 1, slab_index: 0, raster: RasterOptions::default(), nsteps: None, share_monitors: true, allow_2d: true }
    }
}

/// Compile a simulation into a plan.
pub fn compile(sim: &Simulation, opts: &CompileOptions) -> Result<Plan> {
    sim.validate()?;
    let grid = sim.build_grid()?;
    compile_with_grid(sim, grid, opts)
}

/// Compile a simulation on a grid built beforehand (every slab of a
/// decomposed run shares one grid).
pub fn compile_with_grid(sim: &Simulation, grid: Grid, opts: &CompileOptions) -> Result<Plan> {
    // 2-D simulations: periodic along the invariant axis, sources and
    // monitors spanning the slab
    let canonical = sim.canonical();
    let sim = &*canonical;
    if sim.fixed_angle {
        bail!("{}", crate::engine::fixed_angle::ENGINE_ONLY);
    }
    let s = Setup::new(sim, &grid, opts)?;
    let layers = layers::StabilisedLayers::new(&s);
    let absorbers = layers::Absorbers::new(&s)?;
    let mut mat = materials::compile(&s, &layers, &absorbers)?;
    let bounds = boundaries::compile(&s)?;
    let pml = layers::cpml_slabs(&s);
    let disp = cell_lists::dispersive(&s, &mat, &layers, &absorbers);
    let nonlin = cell_lists::nonlinear(&s, &mat, bounds.complex_fields())?;
    let tensor = cell_lists::tensor(&s, &mat, &bounds)?;

    let (x0, nxl, n, dt) = (s.x0, s.nxl, s.n, s.dt);
    let mut plan = Plan {
        name: sim.name.clone(),
        grid: grid.clone(),
        n,
        x0,
        nxl,
        dt,
        nsteps: s.nsteps,
        inv_p: s.inv_p.clone(),
        inv_d: s.inv_d.clone(),
        // (the coefficient arrays are moved: they are as large as the grid)
        cb: std::mem::take(&mut mat.cb),
        ca: mat.ca.take(),
        ch: mat.ch.take(),
        ch_scalar: s.cdt(),
        hloss: mat.hloss.take(),
        bc_y: bounds.bc_y,
        bc_z: bounds.bc_z,
        ghost_e_hi: bounds.ghost_e_hi.clone(),
        ghost_h_lo: bounds.ghost_h_lo.clone(),
        bloch: bounds.complex_fields() || mat.herm.is_some(),
        bloch_phase: bounds.bloch_phase,
        periodic: s.periodic,
        pml,
        absorb: absorbers.slabs.clone(),
        disp,
        nonlin,
        tensor,
        inj_h: vec![],
        inj_e: vec![],
        dft: vec![],
        time: vec![],
        energy: vec![],
        media: s.media.clone(),
        shutoff: sim.shutoff,
        shutoff_interval: sim.shutoff_interval.max(1),
        source_end_time: sim.sources.iter().map(|s| s.signal().end_time()).fold(0.0, f64::max),
        sources: sim.sources.clone(),
        source_analytic: vec![],
        twod: None,
        slice: s.ropts.slice.map(|(a, w, _)| (a, w)),
        sheet: None,
        plugin: crate::plugin::compile_plugins(&s.media, &mat.med_local, &mat.cb_local, x0, nxl, n, dt, bounds.complex_fields())?,
        herm: mat.herm.take(),
        disp_subpixel: mat.disp_subpixel,
        gain2: crate::gain2::compile_gain2(&s.media, &mat.med_local, &mat.cb_local, &s.inv_p_full, &s.inv_d_full, x0, nxl, n, dt, bounds.complex_fields())?,
    };
    plan.sheet = crate::sheet::compile_sheets(
        &sim.sheets,
        &crate::sheet::SheetCtx { grid: &grid, x0, nxl, dt, periodic: s.periodic, invariant: s.inv, inv_d_full: &s.inv_d_full, cb_local: &mat.cb_local, med_local: &mat.med_local, media: &s.media },
    )?;
    plan.twod = two_d_update(&s, &plan, &bounds);
    sources::compile_sources(&mut plan, &s, &mat)?;
    monitors::compile_monitors(&mut plan, &s, &mat)?;
    Ok(plan)
}

/// What every phase of the compiler reads: the canonical simulation, its
/// grid, media and structures, the slab this plan covers and the time step.
pub(super) struct Setup<'a> {
    pub sim: &'a Simulation,
    pub grid: &'a Grid,
    pub opts: &'a CompileOptions,
    /// invariant axis of a 2-D simulation
    pub inv: Option<usize>,
    /// cells of the whole grid
    pub n: [usize; 3],
    /// this slab: first global x plane and number of planes
    pub x0: usize,
    pub nxl: usize,
    pub ny: usize,
    pub nz: usize,
    /// one y-z plane: ny * nz
    pub plane: usize,
    pub dt: f64,
    pub nsteps: usize,
    pub media: Vec<Medium>,
    pub background: usize,
    pub structures: Vec<ResolvedStructure<'a>>,
    pub ropts: RasterOptions,
    pub periodic: [bool; 3],
    /// inverse primal and dual (periodic-aware) spacings along each axis
    pub inv_p_full: [Vec<Real>; 3],
    pub inv_d_full: [Vec<Real>; 3],
    /// the same, with x restricted to this slab
    pub inv_p: [Vec<Real>; 3],
    pub inv_d: [Vec<Real>; 3],
}

impl<'a> Setup<'a> {
    fn new(sim: &'a Simulation, grid: &'a Grid, opts: &'a CompileOptions) -> Result<Setup<'a>> {
        let inv = sim.invariant();
        let n = grid.n();
        let (nx, ny, nz) = (n[0], n[1], n[2]);
        let (x0, nxl) = opts.slab.unwrap_or((0, nx));
        if x0 + nxl > nx || nxl == 0 {
            bail!("invalid slab {x0}+{nxl} for nx={nx}");
        }
        let plane = ny * nz;
        let dt = sim.time_step(grid)?;
        let nsteps = opts.nsteps.unwrap_or_else(|| (sim.run_time / dt).ceil() as usize);
        let media = sim.media_table()?;
        let background = media.iter().position(|m| m.name == sim.background).unwrap();
        let structures = resolve(&sim.structures, &media)?;
        let mut ropts = opts.raster.clone();
        ropts.smoothing = sim.subpixel && ropts.smoothing;
        ropts.slice = sim.raster_slice(grid);
        ropts.weighting = sim.subpixel_weighting;
        ropts.cuts = sim.raster_options(grid).cuts;
        // transparent dispersive dielectrics are smoothed at the sources' centre frequency
        ropts.disp_omega = if ropts.smoothing { source_center_omega(&sim.sources) } else { None };
        ropts.disp_dt = dt;
        if let Some(a) = inv {
            if n[a] != 1 {
                bail!("the grid of a 2-D simulation must have one cell along its invariant axis (got {})", n[a]);
            }
            if sim.polarization != crate::spec::Polarization::Both && media.iter().any(|m| m.is_tensor()) {
                bail!("off-diagonal permittivity couples the two field sets of a 2-D simulation: use polarization = \"both\"");
            }
        }
        let b = &sim.boundaries;
        let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
        let inv_p_full: [Vec<Real>; 3] = [grid.inv_primal(0), grid.inv_primal(1), grid.inv_primal(2)];
        let inv_d_full: [Vec<Real>; 3] =
            [grid.inv_dual(0, periodic[0]), grid.inv_dual(1, periodic[1]), grid.inv_dual(2, periodic[2])];
        let inv_p = [inv_p_full[0][x0..x0 + nxl].to_vec(), inv_p_full[1].clone(), inv_p_full[2].clone()];
        let inv_d = [inv_d_full[0][x0..x0 + nxl].to_vec(), inv_d_full[1].clone(), inv_d_full[2].clone()];
        Ok(Setup { sim, grid, opts, inv, n, x0, nxl, ny, nz, plane, dt, nsteps, media, background, structures, ropts, periodic, inv_p_full, inv_d_full, inv_p, inv_d })
    }

    /// c dt, the curl coefficient of the update in vacuum
    pub fn cdt(&self) -> Real {
        (C0 * self.dt) as Real
    }

    /// local (unpadded) index of cell (i, j, k) of the slab
    pub fn lidx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// storage index of local cell (i, j, k): one ghost plane before the slab
    pub fn sidx(&self, i: usize, j: usize, k: usize) -> u64 {
        ((i + 1) * self.plane + j * self.nz + k) as u64
    }
}

/// The specialised 2-D update (one periodic cell along an axis, diagonal
/// media), if the plan can use it; the field sets in use are set by the
/// sources.
fn two_d_update(s: &Setup, plan: &Plan, bounds: &boundaries::BoundaryCodes) -> Option<TwoD> {
    let (sim, opts, n) = (s.sim, s.opts, s.n);
    let b = &sim.boundaries;
    let allow_2d = opts.allow_2d && !sim.force_3d_kernels && std::env::var("LIGHTSIM_2D").as_deref() != Ok("0");
    // (the 2-D update has no magnetic loss)
    if !(allow_2d && plan.tensor.is_none() && plan.herm.is_none() && plan.hloss.is_none() && !s.media.iter().any(|m| m.is_tensor())) {
        return None;
    }
    let flat = |a: usize| -> bool { n[a] == 1 && matches!(b.axis(a)[0], Boundary::Periodic | Boundary::Bloch { .. }) && bounds.bloch_phase[a] == (1.0, 0.0) };
    // a slab decomposition (along x) of a grid that is one cell thick along x makes no sense
    let axis = match s.inv {
        Some(a) => Some(a).filter(|&a| flat(a)),
        None => [2usize, 1, 0].into_iter().find(|&a| flat(a)),
    }
    .filter(|&a| a != 0 || opts.nslabs <= 1);
    axis.map(|axis| TwoD { axis, allow: sim.polarization.sets(), active: [false, false] })
}
