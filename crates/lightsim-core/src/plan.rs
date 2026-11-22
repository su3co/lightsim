//! Compilation of a [`Simulation`] into a backend-independent execution
//! [`Plan`]: coefficient arrays in storage layout, boundary codes, CPML slabs,
//! source injection tables, dispersive-pole tables and monitor tables.
//!
//! A plan can cover the whole domain or one x-slab `[x0, x0 + nxl)` of it
//! (multi-GPU / multi-node domain decomposition). All indices stored in the
//! plan are local storage indices of that slab.

use crate::constants::{C0, EPS0, ETA0, TAU};
use crate::grid::{Component, Grid};
use crate::materials::Medium;
use crate::monitors::{Apodization, Monitor};
use crate::raster::{RasterOptions, rasterize_component, resolve};
use crate::sources::{Direction, Source, TimeSignal};
use crate::spec::{Boundary, Simulation};
use anyhow::{Result, bail};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Kernel boundary codes for the y/z sides.
pub const BC_ZERO: u32 = 0;
pub const BC_PERIODIC: u32 = 1;
pub const BC_MIRROR: u32 = 2;
/// Bloch-periodic (complex fields): wrap with phase exp(i k L).
pub const BC_BLOCH: u32 = 3;

/// How an x ghost plane is filled each step.
#[derive(Clone, Debug, PartialEq)]
pub enum GhostRule {
    /// Always zero (PEC wall, also behind a PML).
    Zero,
    /// Copy from the opposite end of the (global) domain, multiplied by
    /// `(cos, sin)` of the Bloch phase (`(1, 0)` for plain periodic).
    Periodic { cos: f64, sin: f64 },
    /// `factor *` the adjacent interior plane (PMC: +1 for E at the high side,
    /// -1 for H at the low side).
    Mirror { factor: f32 },
    /// Filled by the domain decomposition from the neighbouring slab.
    Neighbor,
}

/// One CPML slab normal to `axis`, restricted to the local x range.
#[derive(Clone, Debug)]
pub struct PmlSlab {
    pub axis: usize,
    /// Local box (interior indices): start and size per axis.
    pub start: [usize; 3],
    pub size: [usize; 3],
    /// Coefficients along `axis` for E positions and H positions, already
    /// sliced to the local box (index 0 = box start along axis).
    pub e_b: Vec<f32>,
    pub e_a: Vec<f32>,
    pub e_kinv: Vec<f32>,
    pub h_b: Vec<f32>,
    pub h_a: Vec<f32>,
    pub h_kinv: Vec<f32>,
}

impl PmlSlab {
    pub fn cells(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }
}

/// A list of field updates `field[idx] += coef * amp(step, group)`.
#[derive(Clone, Debug)]
pub struct Injection {
    pub comp: Component,
    pub idx: Vec<u64>,
    pub coef: Vec<f32>,
    /// Imaginary coefficients for complex injections (`amp` holds (re, im)).
    pub coef_im: Option<Vec<f32>>,
    pub group: Vec<u32>,
    pub ngroups: usize,
    /// Amplitude table `[step][group]` (real) or `[step][group][2]` (complex).
    pub amps: Vec<f32>,
}

impl Injection {
    pub fn is_complex(&self) -> bool {
        self.coef_im.is_some()
    }
    pub fn grouped(&self) -> bool {
        self.ngroups > 1
    }
    /// Offset (in floats) of the amplitude row of `step`.
    pub fn row(&self, step: usize) -> usize {
        step * self.ngroups * if self.is_complex() { 2 } else { 1 }
    }
}

/// Dispersive-media tables for one E component.
#[derive(Clone, Debug)]
pub struct DispComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    pub matid: Vec<u32>,
    pub scale: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DispPlan {
    pub npole: usize,
    /// `[material][pole][3]` ADE coefficients.
    pub coefs: Vec<f32>,
    pub comps: Vec<DispComponent>,
}

/// Cells of one E component inside chi2/chi3 media.
#[derive(Clone, Debug)]
pub struct NonlinComponent {
    pub comp: Component,
    /// Storage indices.
    pub idx: Vec<u64>,
    /// chi2 / eps and chi3 / eps of the cell (eps = linear relative permittivity).
    pub c2: Vec<f32>,
    pub c3: Vec<f32>,
}

/// Instantaneous nonlinear polarization P_i/eps0 = chi2 E_i^2 + chi3 E_i^3,
/// applied as an implicit per-cell correction after the linear E update.
#[derive(Clone, Debug)]
pub struct NonlinPlan {
    pub comps: Vec<NonlinComponent>,
}

/// Cells of one E component inside media with off-diagonal permittivity.
#[derive(Clone, Debug)]
pub struct TensorComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    /// K_{a,a+1} / (c dt) and K_{a,a+2} / (c dt) (lossless: kappa = eps^-1).
    pub c1: Vec<f32>,
    pub c2: Vec<f32>,
    /// M_{a,a+1}, M_{a,a+2} (zero for lossless media).
    pub m1: Vec<f32>,
    pub m2: Vec<f32>,
    /// Oriented poles: table row of the cell's medium (u32::MAX = none) and
    /// 1/eps_inf along the component.
    pub opm: Vec<u32>,
    pub oscale: Vec<f32>,
}

/// Off-diagonal permittivity: after the linear update (which applies
/// kappa_aa), E_a += kappa_ab <dD_b> + kappa_ac <dD_c>, where dD is recovered
/// from the linear update over storage planes [p0, p1).
#[derive(Clone, Debug)]
pub struct TensorPlan {
    /// Storage planes [p0, p1) held in the dD buffers (decomposed runs include
    /// the ghost planes 0 and nxl+1, filled by the chain's dD exchange).
    pub p0: usize,
    pub p1: usize,
    /// Interior planes [i0, i1) where dD is computed locally.
    pub i0: usize,
    pub i1: usize,
    /// Periodic wrap bits (x, y, z) for the neighbour averages.
    pub wrap: u32,
    pub cdt: f32,
    /// E^n of the neighbours is needed (lossy tensors or oriented poles).
    pub lossy: bool,
    /// Oriented poles: poles per medium row and the table
    /// `[row][pole][c0, c1, c2, u_x, u_y, u_z]` (rows padded with zero poles).
    pub op_np: usize,
    pub op_tab: Vec<f32>,
    pub comps: Vec<TensorComponent>,
}

/// A rectangular index box of one component, global and local.
#[derive(Clone, Debug)]
pub struct CompBox {
    pub comp: Component,
    /// Global index box [lo, hi) per axis.
    pub glo: [usize; 3],
    pub ghi: [usize; 3],
    /// Local (slab) interior box start and size; size[0] may be 0 if the box
    /// does not intersect this slab.
    pub lstart: [usize; 3],
    pub lsize: [usize; 3],
    /// x offset of the local part inside the global box.
    pub gx_off: usize,
}

impl CompBox {
    pub fn gsize(&self) -> [usize; 3] {
        [self.ghi[0] - self.glo[0], self.ghi[1] - self.glo[1], self.ghi[2] - self.glo[2]]
    }
    pub fn lcells(&self) -> usize {
        self.lsize[0] * self.lsize[1] * self.lsize[2]
    }
    pub fn gcells(&self) -> usize {
        let s = self.gsize();
        s[0] * s[1] * s[2]
    }
    pub fn is_local(&self) -> bool {
        self.lcells() > 0
    }
}

/// What a DFT monitor is used for (post-processing).
#[derive(Clone, Debug)]
pub enum DftKind {
    Field,
    /// Plane flux: axis, normal index p; boxes are [E_b, E_c, H_b(p-1), H_c(p-1), H_b(p), H_c(p)].
    Flux { axis: usize, p: usize },
    FluxBox,
    /// Far-field projection from a closed box (faces as in FluxBox).
    FarField { theta: Vec<f64>, phi: Vec<f64>, n_medium: f64 },
    /// Mode overlap; `origin` is the transverse node-box corner the mode
    /// profile arrays are indexed from.
    Mode { axis: usize, p: usize, origin: [usize; 3], modes: Vec<crate::sources::ModeProfile> },
}

#[derive(Clone, Debug)]
pub struct DftPlan {
    pub name: String,
    /// Position of this monitor in [`Plan::dft`] when the plan was compiled.
    /// `shared` refers to monitors by it, so that a subset of the list (the
    /// port monitors of a gradient session, say) still finds its data.
    pub id: usize,
    pub kind: DftKind,
    pub freqs: Vec<f64>,
    /// The boxes of this monitor. A box whose data is accumulated by an
    /// earlier monitor (see `shared`) is marked as not local, so that no
    /// backend allocates or accumulates it.
    pub boxes: Vec<CompBox>,
    /// Per box: `Some((id, box))` if the earlier monitor with that `id`
    /// records the same component box with the same frequencies, time window
    /// and stride. The data is taken from there when the results are
    /// assembled ([`DftPlan::source_of`]). Many mode monitors on one plane
    /// then cost one accumulation. All `None` when the plan was compiled
    /// with [`CompileOptions::share_monitors`] off.
    pub shared: Vec<Option<(usize, usize)>>,
    /// Time window of the accumulation.
    pub apodization: Apodization,
    pub stride: usize,
    /// Phasor tables `[step][freq][2]` for E-time and H-time samples.
    pub phasor_e: Vec<f32>,
    pub phasor_h: Vec<f32>,
    /// For FluxBox: sub-plane descriptors (axis, sign, first box index).
    pub faces: Vec<(usize, f64, usize)>,
    /// Physical monitor region; surface elements are clipped to it.
    pub region: ([f64; 3], [f64; 3]),
}

impl DftPlan {
    /// Where the data of box `bi` of the monitor at position `di` of `list`
    /// is recorded: `(position in list, box)`. That is the box itself, or
    /// for a shared box the box of the monitor that records it, looked up
    /// by its [`DftPlan::id`], so that `list` may be any part of a plan's
    /// monitors that contains the recording monitor. Panics if it does not:
    /// the data of the box is then not in the list at all.
    pub fn source_of(list: &[DftPlan], di: usize, bi: usize) -> (usize, usize) {
        match list[di].shared[bi] {
            None => (di, bi),
            Some((id, sb)) => {
                // the usual case first: the whole list, in compile order
                let at = if list.get(id).is_some_and(|d| d.id == id) { Some(id) } else { list.iter().position(|d| d.id == id) };
                match at {
                    Some(sd) => (sd, sb),
                    None => panic!(
                        "monitor '{}' takes the fields of its box {bi} from monitor {id} of the compiled plan, which is not in this list of monitors",
                        list[di].name
                    ),
                }
            }
        }
    }
}

/// An energy monitor: the time monitor that samples the six components in
/// the box, and the weight of every local sample in the energy sum.
#[derive(Clone, Debug)]
pub struct EnergyPlan {
    pub name: String,
    /// Index into [`Plan::time`].
    pub time: usize,
    /// Per box of that time monitor, per local cell (x-major, as sampled):
    /// `eps0/2 * (eps or mu) * volume` of the cell inside the monitor box, in
    /// J per (V/m)^2. H is stored as eta0 H, so the same factor serves both.
    pub weights: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct TimePlan {
    pub name: String,
    pub boxes: Vec<CompBox>,
    pub interval: usize,
    pub start_step: usize,
    pub is_probe: bool,
}

/// A plan whose fields do not depend on one coordinate: a single cell along
/// `axis` with periodic boundaries and media without off-diagonal
/// permittivity. Maxwell's equations then split into two independent sets,
/// and the backends run a specialised 2-D update on the sets in use instead
/// of the general 3-D one (same arithmetic, so the results agree to
/// rounding).
///
/// With `(u, v)` the in-plane axes (see [`crate::spec::plane_of`]) and `w`
/// the invariant one, set 0 ("TE") is (E_u, E_v, H_w) and set 1 ("TM") is
/// (E_w, H_u, H_v).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoD {
    /// The invariant axis.
    pub axis: usize,
    /// `[te, tm]`: sets the simulation admits (its `polarization`).
    pub allow: [bool; 2],
    /// `[te, tm]`: sets that are updated: admitted and driven by a source
    /// (a set nothing drives stays identically zero).
    pub active: [bool; 2],
}

impl TwoD {
    /// The in-plane axes `(u, v)` (v is contiguous in storage) and the
    /// handedness of `(u, v, w)`.
    pub fn plane(&self) -> (usize, usize, f32) {
        crate::spec::plane_of(self.axis)
    }
    /// Index (0 = TE, 1 = TM) of the set component `c` belongs to.
    pub fn set_of(&self, c: Component) -> usize {
        if crate::spec::is_te(c, self.axis) { 0 } else { 1 }
    }
    /// True if component `c` is updated.
    pub fn is_active(&self, c: Component) -> bool {
        self.active[self.set_of(c)]
    }
    /// Mark the set of component `c` as driven (if the simulation admits it).
    pub fn drive(&mut self, c: Component) {
        let k = self.set_of(c);
        self.active[k] |= self.allow[k];
    }
    /// "te", "tm", "te+tm" or "idle": the sets being updated.
    pub fn label(&self) -> &'static str {
        match self.active {
            [true, false] => "te",
            [false, true] => "tm",
            [true, true] => "te+tm",
            _ => "idle",
        }
    }
}

/// Scalar arguments of the 2-D step kernels (`step2d` in
/// `kernels/src/twod_step.rs`) for a plan: shared by the GPU host and by the
/// host-side test of the kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step2d {
    /// In-plane axes (v contiguous in storage) and the invariant axis.
    pub u: usize,
    pub v: usize,
    pub w: usize,
    pub nu: u32,
    pub nv: u32,
    /// Storage index of cell (u, v) = (0, 0).
    pub base: u32,
    /// Boundary codes u_lo | u_hi << 4 | v_lo << 8 | v_hi << 12 (0 zero,
    /// 1 periodic, 2 mirror, 3 Bloch, 4 ghost rows kept by the host: x).
    pub bc: u32,
    /// Columns of (u, v, w) in the material table: u | v << 4 | w << 8.
    pub cols: u32,
    /// +1 if (u, v, w) is a cyclic permutation of (x, y, z), else -1.
    pub sg: f32,
    /// The E update of row 0 is left to the host (periodic or neighbour low
    /// ghost along x).
    pub deferred: bool,
}

impl Step2d {
    /// Kernel `pol` argument for the sets in use.
    pub fn pol(td: &TwoD) -> u32 {
        td.active[0] as u32 | (td.active[1] as u32) << 1
    }
}

/// Magnetically lossy cells of one H component (see [`Plan::hloss`]).
#[derive(Clone, Debug, Default)]
pub struct HLoss {
    pub idx: Vec<u64>,
    pub da: Vec<f32>,
}

/// Compiled simulation for one slab.
#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    pub grid: Grid,
    pub n: [usize; 3],
    pub x0: usize,
    pub nxl: usize,
    pub dt: f64,
    pub nsteps: usize,
    pub inv_p: [Vec<f32>; 3],
    pub inv_d: [Vec<f32>; 3],
    pub cb: [Vec<f32>; 3],
    pub ca: Option<[Vec<f32>; 3]>,
    pub ch: Option<[Vec<f32>; 3]>,
    pub ch_scalar: f32,
    /// Magnetically lossy H cells (media with `sigma_m`): per H component,
    /// the storage indices and the factor `da` of `H~' = da H~ - ch curl E`
    /// (`ch` then holds the lossy coefficient). Applied as `H~ *= da` before
    /// the regular H update.
    pub hloss: Option<[HLoss; 3]>,
    pub bc_y: [u32; 2],
    pub bc_z: [u32; 2],
    /// Ghost rules for this slab: E high-side ghost and H low-side ghost.
    pub ghost_e_hi: GhostRule,
    pub ghost_h_lo: GhostRule,
    pub bloch: bool,
    /// Bloch phase (cos, sin) of k*L per axis ((1, 0) if not Bloch).
    pub bloch_phase: [(f64, f64); 3],
    /// Periodic (or Bloch) axes.
    pub periodic: [bool; 3],
    pub pml: Vec<PmlSlab>,
    pub disp: Option<DispPlan>,
    pub nonlin: Option<NonlinPlan>,
    pub tensor: Option<TensorPlan>,
    pub inj_h: Vec<Injection>,
    pub inj_e: Vec<Injection>,
    pub dft: Vec<DftPlan>,
    pub time: Vec<TimePlan>,
    pub energy: Vec<EnergyPlan>,
    pub media: Vec<Medium>,
    pub shutoff: Option<f64>,
    pub shutoff_interval: usize,
    pub source_end_time: f64,
    pub sources: Vec<Source>,
    /// Per source: it injects the complex (analytic) signal into complex
    /// fields, i.e. it has a complex field pattern and the run is a Bloch
    /// run. The recorded fields are then proportional to the spectrum of the
    /// analytic signal (see [`TimeSignal::spectrum_of`]).
    pub source_analytic: Vec<bool>,
    /// Set when the specialised 2-D update applies (see [`TwoD`]).
    pub twod: Option<TwoD>,
    /// 2-D simulations: (invariant axis, coordinate of the slice plane the
    /// structures were rasterised on).
    pub slice: Option<(usize, f64)>,
}

impl Plan {
    /// Mark as driven the 2-D field sets the current source lists touch
    /// (call after replacing `inj_h` / `inj_e`; sets are never unmarked).
    pub fn drive_2d(&mut self) {
        if let Some(td) = self.twod.as_mut() {
            for j in self.inj_h.iter().chain(self.inj_e.iter()) {
                if !j.idx.is_empty() {
                    td.drive(j.comp);
                }
            }
        }
    }
    /// The scalar arguments of the 2-D step kernels (`None`: not a 2-D plan).
    pub fn step2d(&self) -> Option<Step2d> {
        let td = self.twod?;
        let (u, v, sg) = td.plane();
        let w = td.axis;
        let bc_of = |a: usize| if a == 1 { self.bc_y } else { self.bc_z };
        // x keeps its ghost planes (slab neighbours, Bloch phases): code 4
        let (ulo, uhi, deferred) = if u == 0 {
            match self.ghost_h_lo {
                GhostRule::Zero => (BC_ZERO, 4, false),
                GhostRule::Mirror { .. } => (BC_MIRROR, 4, false),
                GhostRule::Periodic { .. } | GhostRule::Neighbor => (4, 4, true),
            }
        } else {
            let b = bc_of(u);
            (b[0], b[1], false)
        };
        let bv = bc_of(v);
        Some(Step2d {
            u,
            v,
            w,
            nu: if u == 0 { self.nxl as u32 } else { self.n[u] as u32 },
            nv: self.n[v] as u32,
            base: self.plane() as u32,
            bc: ulo | uhi << 4 | bv[0] << 8 | bv[1] << 12,
            cols: (u | v << 4 | w << 8) as u32,
            sg,
            deferred,
        })
    }
    /// Mark every admitted 2-D field set as driven (fields set from outside:
    /// time reversal, adjoint sources).
    pub fn drive_2d_all(&mut self) {
        if let Some(td) = self.twod.as_mut() {
            td.active = td.allow;
        }
    }

    pub fn ny(&self) -> usize {
        self.n[1]
    }
    pub fn nz(&self) -> usize {
        self.n[2]
    }
    pub fn plane(&self) -> usize {
        self.n[1] * self.n[2]
    }
    /// Number of floats in one storage-layout field array.
    pub fn storage_len(&self) -> usize {
        (self.nxl + 2) * self.plane()
    }
    /// Storage index of local interior cell (i, j, k).
    pub fn sidx(&self, i: usize, j: usize, k: usize) -> usize {
        ((i + 1) * self.n[1] + j) * self.n[2] + k
    }
    pub fn local_cells(&self) -> usize {
        self.nxl * self.plane()
    }
    pub fn lossy(&self) -> bool {
        self.ca.is_some()
    }

    /// Bytes of device memory needed (fields + coefficients + aux), approx.
    pub fn memory_bytes(&self) -> usize {
        let s = self.storage_len() * 4;
        let mut b = 6 * s + 3 * s;
        if self.ca.is_some() {
            b += 3 * s;
        }
        if self.ch.is_some() {
            b += 3 * s;
        }
        for p in &self.pml {
            b += 4 * p.cells() * 4;
        }
        for d in &self.dft {
            for bx in &d.boxes {
                b += 2 * bx.lcells() * d.freqs.len() * 4;
            }
        }
        b
    }
}

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

/// E-update coefficients `(ca, cb)` of a voxel of E component `a` whose
/// dominant medium is `md` and whose effective relative permittivity is
/// `eps`: `E <- ca E + cb curl H~`. Shared by the plan compiler and by
/// `crate::autodiff` (design updates re-evaluate exactly this).
pub fn e_update_coefficients(md: &Medium, a: usize, eps: f32, dt: f64) -> (f32, f32) {
    let cdt = (C0 * dt) as f32;
    if md.pec {
        (1.0, 0.0)
    } else if md.sigma[a] > 0.0 {
        let f = md.sigma[a] * dt / (2.0 * EPS0 * eps as f64);
        (((1.0 - f) / (1.0 + f)) as f32, (C0 * dt / eps as f64 / (1.0 + f)) as f32)
    } else {
        (1.0, cdt / eps)
    }
}

/// `(da, ch)` of the H update `H~' = da H~ - ch curl E` for medium `md`
/// along `a` at relative permeability `mu` (`H~ = eta0 H`): with magnetic
/// conductivity, `f = sigma_m dt / (2 mu0 mu)`, `da = (1 - f) / (1 + f)`,
/// `ch = c dt / mu / (1 + f)`; otherwise `(1, c dt / mu)`.
pub fn h_update_coefficients(md: &Medium, a: usize, mu: f32, dt: f64) -> (f32, f32) {
    if md.sigma_m[a] > 0.0 {
        let f = md.sigma_m[a] * dt / (2.0 * crate::constants::MU0 * mu as f64);
        (((1.0 - f) / (1.0 + f)) as f32, (C0 * dt / mu as f64 / (1.0 + f)) as f32)
    } else {
        (1.0, (C0 * dt) as f32 / mu)
    }
}

fn storage_from_local(local: &[f32], nxl: usize, plane: usize, fill: f32) -> Vec<f32> {
    let mut v = vec![fill; (nxl + 2) * plane];
    v[plane..(nxl + 1) * plane].copy_from_slice(local);
    v
}

/// Compile a simulation into a plan.
pub fn compile(sim: &Simulation, opts: &CompileOptions) -> Result<Plan> {
    sim.validate()?;
    let grid = sim.build_grid()?;
    compile_with_grid(sim, grid, opts)
}

pub fn compile_with_grid(sim: &Simulation, grid: Grid, opts: &CompileOptions) -> Result<Plan> {
    // 2-D simulations: periodic along the invariant axis, sources and
    // monitors spanning the slab
    let sim = &*sim.canonical();
    let inv = sim.invariant();
    let n = grid.n();
    let (nx, ny, nz) = (n[0], n[1], n[2]);
    let (x0, nxl) = opts.slab.unwrap_or((0, nx));
    if x0 + nxl > nx || nxl == 0 {
        bail!("invalid slab {x0}+{nxl} for nx={nx}");
    }
    let plane = ny * nz;
    let dt = sim.time_step(&grid)?;
    let nsteps = opts.nsteps.unwrap_or_else(|| (sim.run_time / dt).ceil() as usize);
    let media = sim.media_table()?;
    let background = media.iter().position(|m| m.name == sim.background).unwrap();
    let structures = resolve(&sim.structures, &media)?;
    let mut ropts = opts.raster.clone();
    ropts.smoothing = sim.subpixel && ropts.smoothing;
    ropts.slice = sim.raster_slice(&grid);
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

    // ---- spacing arrays ----
    let inv_p_full: [Vec<f32>; 3] = [grid.inv_primal(0), grid.inv_primal(1), grid.inv_primal(2)];
    let inv_d_full: [Vec<f32>; 3] =
        [grid.inv_dual(0, periodic[0]), grid.inv_dual(1, periodic[1]), grid.inv_dual(2, periodic[2])];
    let inv_p = [inv_p_full[0][x0..x0 + nxl].to_vec(), inv_p_full[1].clone(), inv_p_full[2].clone()];
    let inv_d = [inv_d_full[0][x0..x0 + nxl].to_vec(), inv_d_full[1].clone(), inv_d_full[2].clone()];

    // ---- materials ----
    let any_lossy = media.iter().any(|m| m.is_lossy());
    let any_magnetic = media.iter().any(|m| m.is_magnetic());
    let cdt = (C0 * dt) as f32;
    let mut cb_local: [Vec<f32>; 3] = [vec![], vec![], vec![]];
    let mut ca_local: [Vec<f32>; 3] = [vec![], vec![], vec![]];
    let mut eps_local: [Vec<f32>; 3] = [vec![], vec![], vec![]];
    let mut med_local: [Vec<u16>; 3] = [vec![], vec![], vec![]];
    for a in 0..3 {
        let comp = Component::e(a);
        let cm = rasterize_component(&grid, comp, x0, nxl, &structures, &media, background, &ropts);
        let mut cbv = vec![0f32; cm.value.len()];
        let mut cav = vec![1f32; cm.value.len()];
        cbv.par_iter_mut()
            .zip(cav.par_iter_mut())
            .zip(cm.value.par_iter().zip(cm.medium.par_iter()))
            .for_each(|((cb_o, ca_o), (&eps, &m))| {
                (*ca_o, *cb_o) = e_update_coefficients(&media[m as usize], a, eps, dt);
            });
        cb_local[a] = cbv;
        ca_local[a] = cav;
        eps_local[a] = cm.value;
        med_local[a] = cm.medium;
    }

    // Off-diagonal tensors: the linear update applies M_aa and K_aa of the
    // semi-implicit tensor update (staircased).
    for m in media.iter().filter(|m| m.is_tensor()) {
        if m.is_dispersive() || m.is_nonlinear() || m.pec {
            bail!("medium '{}': off-diagonal tensors cannot be combined with per-axis poles or nonlinearity yet", m.name);
        }
        if !m.oriented.is_empty() && (m.is_lossy() || m.eps_off.iter().any(|&v| v != 0.0)) {
            bail!("medium '{}': oriented poles need a lossless diagonal eps_inf", m.name);
        }
    }
    if media.iter().any(|m| m.is_tensor()) {
        for a in 0..3 {
            for o in 0..cb_local[a].len() {
                let md = &media[med_local[a][o] as usize];
                if md.is_tensor() {
                    let (mm, kk) = md.tensor_update(cdt as f64);
                    cb_local[a][o] = kk[a][a] as f32;
                    ca_local[a][o] = mm[a][a] as f32;
                }
            }
        }
    }

    // Wall conditions on cb.
    let lidx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    for a in 0..3 {
        let side = b.axis(a);
        let n_a = n[a];
        for (s, bd) in side.iter().enumerate() {
            let pec_like = matches!(bd, Boundary::Pec | Boundary::Pml { .. });
            let pmc = matches!(bd, Boundary::Pmc);
            // which (component, index along a) to zero
            let mut zero: Vec<(usize, usize)> = vec![];
            if s == 0 && pec_like {
                zero.push(((a + 1) % 3, 0));
                zero.push(((a + 2) % 3, 0));
            }
            if s == 1 && pmc {
                zero.push((a, n_a - 1));
            }
            for (c, ia) in zero {
                // visit only the plane index ia along axis a
                let (ir, jr, kr) = match a {
                    0 => {
                        if ia < x0 || ia >= x0 + nxl {
                            continue;
                        }
                        (ia - x0..ia - x0 + 1, 0..ny, 0..nz)
                    }
                    1 => (0..nxl, ia..ia + 1, 0..nz),
                    _ => (0..nxl, 0..ny, ia..ia + 1),
                };
                for i in ir.clone() {
                    for j in jr.clone() {
                        for k in kr.clone() {
                            cb_local[c][lidx(i, j, k)] = 0.0;
                            ca_local[c][lidx(i, j, k)] = 1.0;
                        }
                    }
                }
            }
        }
    }

    let cb = [
        storage_from_local(&cb_local[0], nxl, plane, 0.0),
        storage_from_local(&cb_local[1], nxl, plane, 0.0),
        storage_from_local(&cb_local[2], nxl, plane, 0.0),
    ];
    let ca = if any_lossy {
        Some([
            storage_from_local(&ca_local[0], nxl, plane, 1.0),
            storage_from_local(&ca_local[1], nxl, plane, 1.0),
            storage_from_local(&ca_local[2], nxl, plane, 1.0),
        ])
    } else {
        None
    };
    let mut mu_local: Option<[Vec<f32>; 3]> = None;
    let any_hloss = media.iter().any(|m| m.is_magnetic_lossy());
    let mut hloss: Option<[HLoss; 3]> = None;
    let ch = if any_magnetic {
        let mut arr: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        let mut mus: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        let mut hl: [HLoss; 3] = Default::default();
        for a in 0..3 {
            let cm = rasterize_component(&grid, Component::h(a), x0, nxl, &structures, &media, background, &ropts);
            let mut chv = vec![0f32; cm.value.len()];
            for (o, (&mu, &m)) in cm.value.iter().zip(cm.medium.iter()).enumerate() {
                let (da, c) = h_update_coefficients(&media[m as usize], a, mu, dt);
                chv[o] = c;
                if da != 1.0 {
                    hl[a].idx.push((o + plane) as u64);
                    hl[a].da.push(da);
                }
            }
            arr[a] = storage_from_local(&chv, nxl, plane, 0.0);
            mus[a] = cm.value;
        }
        mu_local = Some(mus);
        if any_hloss {
            hloss = Some(hl);
        }
        Some(arr)
    } else {
        None
    };

    // ---- boundary codes ----
    let code_lo = |bd: &Boundary| -> Result<u32> {
        Ok(match bd {
            Boundary::Periodic => BC_PERIODIC,
            Boundary::Bloch { k } => {
                if *k != 0.0 { BC_BLOCH } else { BC_PERIODIC }
            }
            Boundary::Pmc => BC_MIRROR,
            _ => BC_ZERO,
        })
    };
    let bc_y = [code_lo(&b.y[0])?, code_lo(&b.y[1])?];
    let bc_z = [code_lo(&b.z[0])?, code_lo(&b.z[1])?];
    let mut bloch_phase = [(1.0f64, 0.0f64); 3];
    for a in 0..3 {
        if let Boundary::Bloch { k } = &b.axis(a)[0] {
            let e = grid.edges(a);
            let ph = k * (e[e.len() - 1] - e[0]);
            bloch_phase[a] = (ph.cos(), ph.sin());
        }
    }
    let bloch_yz = bc_y[0] == BC_BLOCH || bc_z[0] == BC_BLOCH;
    // complex fields on every slab when any axis is Bloch with k != 0
    let bloch_x = matches!(b.x[0], Boundary::Bloch { k } if k != 0.0);

    // ---- x ghost rules ----
    let first = opts.slab_index == 0;
    let last = opts.slab_index + 1 == opts.nslabs.max(1);
    let lx = grid.x[nx] - grid.x[0];
    let (ghost_e_hi, ghost_h_lo, _) = {
        let (eh, bl) = if !last {
            (GhostRule::Neighbor, false)
        } else {
            match &b.x[1] {
                Boundary::Periodic => {
                    if opts.nslabs > 1 { (GhostRule::Neighbor, false) } else { (GhostRule::Periodic { cos: 1.0, sin: 0.0 }, false) }
                }
                Boundary::Bloch { k } => {
                    // several slabs: the chain's wrap exchange applies the phase
                    let ph = k * lx;
                    if opts.nslabs > 1 { (GhostRule::Neighbor, false) } else { (GhostRule::Periodic { cos: ph.cos(), sin: ph.sin() }, false) }
                }
                Boundary::Pmc => (GhostRule::Mirror { factor: 1.0 }, false),
                _ => (GhostRule::Zero, false),
            }
        };
        let hl = if !first {
            GhostRule::Neighbor
        } else {
            match &b.x[0] {
                Boundary::Periodic => {
                    if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: 1.0, sin: 0.0 } }
                }
                Boundary::Bloch { k } => {
                    let ph = -k * lx;
                    if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: ph.cos(), sin: ph.sin() } }
                }
                Boundary::Pmc => GhostRule::Mirror { factor: -1.0 },
                _ => GhostRule::Zero,
            }
        };
        (eh, hl, bl)
    };

    // ---- CPML slabs ----
    let mut pml = Vec::new();
    for a in 0..3 {
        for (s, bd) in b.axis(a).iter().enumerate() {
            let Boundary::Pml { layers, params, .. } = bd else { continue };
            let lyr = *layers;
            if lyr == 0 {
                continue;
            }
            let e = grid.edges(a);
            let n_a = n[a];
            let (g0, g1) = if s == 0 { (0, lyr) } else { (n_a - lyr, n_a) };
            let (inner, outer) = if s == 0 { (e[lyr], e[0]) } else { (e[n_a - lyr], e[n_a]) };
            let thick = (inner - outer).abs();
            let dcell = if s == 0 { e[1] - e[0] } else { e[n_a] - e[n_a - 1] };
            let m = params.order;
            let sigma_max = params.sigma_scale * 0.8 * (m + 1.0) / (ETA0 * dcell);
            let coef = |x: f64| -> (f32, f32, f32) {
                let d = ((x - inner).abs() / thick).clamp(0.0, 1.0);
                let sigma = sigma_max * d.powf(m);
                let kappa = 1.0 + (params.kappa_max - 1.0) * d.powf(m);
                let alpha = params.alpha_max * (1.0 - d);
                let bb = (-(sigma / kappa + alpha) * dt / EPS0).exp();
                let aa = if sigma > 0.0 { sigma / (sigma * kappa + kappa * kappa * alpha) * (bb - 1.0) } else { 0.0 };
                (bb as f32, aa as f32, (1.0 / kappa) as f32)
            };
            // local box
            let mut start = [0usize, 0, 0];
            let mut size = [nxl, ny, nz];
            let (l0, l1) = if a == 0 {
                let lo = g0.max(x0);
                let hi = g1.min(x0 + nxl);
                if lo >= hi {
                    continue;
                }
                (lo, hi)
            } else {
                (g0, g1)
            };
            if a == 0 {
                start[0] = l0 - x0;
                size[0] = l1 - l0;
            } else {
                start[a] = l0;
                size[a] = l1 - l0;
            }
            let mut eb_ = vec![];
            let mut ea_ = vec![];
            let mut ek_ = vec![];
            let mut hb_ = vec![];
            let mut ha_ = vec![];
            let mut hk_ = vec![];
            for gi in l0..l1 {
                let (b1, a1, k1) = coef(e[gi]);
                eb_.push(b1);
                ea_.push(a1);
                ek_.push(k1);
                let (b2, a2, k2) = coef(0.5 * (e[gi] + e[gi + 1]));
                hb_.push(b2);
                ha_.push(a2);
                hk_.push(k2);
            }
            pml.push(PmlSlab { axis: a, start, size, e_b: eb_, e_a: ea_, e_kinv: ek_, h_b: hb_, h_a: ha_, h_kinv: hk_ });
        }
    }

    // ---- dispersive tables ----
    let disp_media: Vec<usize> = (0..media.len()).filter(|&m| media[m].is_dispersive()).collect();
    let disp = if disp_media.is_empty() {
        None
    } else {
        let npole = disp_media.iter().map(|&m| media[m].poles.len()).max().unwrap();
        let mut coefs = Vec::new();
        for &m in &disp_media {
            for q in 0..npole {
                let c = media[m].poles.get(q).map(|p| p.ade_coefficients(dt)).unwrap_or([1.0, 0.0, 0.0]);
                coefs.extend(c.iter().map(|v| *v as f32));
            }
        }
        let mut comps = Vec::new();
        for a in 0..3 {
            let mut d = DispComponent { comp: Component::e(a), idx: vec![], matid: vec![], scale: vec![] };
            for i in 0..nxl {
                for j in 0..ny {
                    for k in 0..nz {
                        let o = lidx(i, j, k);
                        let m = med_local[a][o] as usize;
                        if let Some(mi) = disp_media.iter().position(|&x| x == m) {
                            if media[m].pole_axes[a] && cb_local[a][o] != 0.0 {
                                d.idx.push(((i + 1) * plane + j * nz + k) as u64);
                                d.matid.push(mi as u32);
                                // E^{n+1} (1 + f) = E^n (1 - f) + (c dt / eps) curl H - dP / eps,
                                // f = sigma dt / (2 eps0 eps): the semi-implicit conduction
                                // term divides the polarisation term as it does the curl
                                let f = media[m].sigma[a].max(0.0) * dt / (2.0 * EPS0 * media[m].eps[a]);
                                d.scale.push((1.0 / (media[m].eps[a] * (1.0 + f))) as f32);
                            }
                        }
                    }
                }
            }
            if !d.idx.is_empty() {
                comps.push(d);
            }
        }
        if comps.is_empty() { None } else { Some(DispPlan { npole, coefs, comps }) }
    };

    // ---- nonlinear (chi2 / chi3) cell lists ----
    let nonlin = if !media.iter().any(|m| m.is_nonlinear()) {
        None
    } else {
        if bloch_x || bloch_yz {
            bail!("nonlinear (chi2/chi3) media cannot be combined with Bloch boundaries (complex fields)");
        }
        let mut comps = Vec::new();
        for a in 0..3 {
            let mut c = NonlinComponent { comp: Component::e(a), idx: vec![], c2: vec![], c3: vec![] };
            for i in 0..nxl {
                for j in 0..ny {
                    for k in 0..nz {
                        let o = lidx(i, j, k);
                        let md = &media[med_local[a][o] as usize];
                        if (md.chi2[a] != 0.0 || md.chi3[a] != 0.0) && cb_local[a][o] != 0.0 {
                            let eps = eps_local[a][o] as f64;
                            c.idx.push(((i + 1) * plane + j * nz + k) as u64);
                            c.c2.push((md.chi2[a] / eps) as f32);
                            c.c3.push((md.chi3[a] / eps) as f32);
                        }
                    }
                }
            }
            if !c.idx.is_empty() {
                comps.push(c);
            }
        }
        if comps.is_empty() { None } else { Some(NonlinPlan { comps }) }
    };

    // ---- off-diagonal permittivity lists ----
    let tensor = if !media.iter().any(|m| m.is_tensor()) {
        None
    } else {
        if bloch_x || bloch_yz {
            bail!("off-diagonal permittivity cannot be combined with Bloch boundaries yet");
        }
        let (mut imin, mut imax) = (usize::MAX, 0usize);
        let mut lossy_t = false;
        // oriented-pole table rows: media with oriented poles
        let op_rows: Vec<usize> = (0..media.len()).filter(|&m| !media[m].oriented.is_empty()).collect();
        let op_np = op_rows.iter().map(|&m| media[m].oriented.len()).max().unwrap_or(0);
        let mut op_tab = Vec::new();
        for &m in &op_rows {
            for q in 0..op_np {
                match media[m].oriented.get(q) {
                    Some(op) => {
                        let c = op.pole.ade_coefficients(dt);
                        let u = op.unit();
                        op_tab.extend([c[0] as f32, c[1] as f32, c[2] as f32, u[0] as f32, u[1] as f32, u[2] as f32]);
                    }
                    None => op_tab.extend([1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]),
                }
            }
        }
        let mut comps = Vec::new();
        for a in 0..3 {
            let mut c = TensorComponent { comp: Component::e(a), idx: vec![], c1: vec![], c2: vec![], m1: vec![], m2: vec![], opm: vec![], oscale: vec![] };
            for i in 0..nxl {
                for j in 0..ny {
                    for k in 0..nz {
                        let o = lidx(i, j, k);
                        let md = &media[med_local[a][o] as usize];
                        if md.is_tensor() && cb_local[a][o] != 0.0 {
                            let (mm, kk) = md.tensor_update(cdt as f64);
                            let (b1, b2) = ((a + 1) % 3, (a + 2) % 3);
                            c.idx.push(((i + 1) * plane + j * nz + k) as u64);
                            c.c1.push((kk[a][b1] / cdt as f64) as f32);
                            c.c2.push((kk[a][b2] / cdt as f64) as f32);
                            c.m1.push(mm[a][b1] as f32);
                            c.m2.push(mm[a][b2] as f32);
                            let mi = med_local[a][o] as usize;
                            match op_rows.iter().position(|&r| r == mi) {
                                Some(row) => {
                                    c.opm.push(row as u32);
                                    c.oscale.push((1.0 / md.eps[a]) as f32);
                                }
                                None => {
                                    c.opm.push(u32::MAX);
                                    c.oscale.push(0.0);
                                }
                            }
                            lossy_t |= md.is_lossy() || !md.oriented.is_empty();
                            imin = imin.min(i);
                            imax = imax.max(i);
                        }
                    }
                }
            }
            if !c.idx.is_empty() {
                comps.push(c);
            }
        }
        let wrap_yz = ((bc_y[0] == BC_PERIODIC) as u32) << 1 | ((bc_z[0] == BC_PERIODIC) as u32) << 2;
        if opts.nslabs > 1 {
            // every slab exchanges dD boundary planes with its neighbours
            Some(TensorPlan { p0: 0, p1: nxl + 2, i0: 1, i1: nxl + 1, wrap: wrap_yz, cdt, lossy: media.iter().any(|m| m.is_tensor() && (m.is_lossy() || !m.oriented.is_empty())), op_np, op_tab, comps })
        } else if comps.is_empty() {
            None
        } else {
            let wrap_x = matches!(b.x[0], Boundary::Periodic);
            let (p0, p1) = if wrap_x { (1, nxl + 1) } else { (imin.max(1), (imax + 3).min(nxl + 1)) };
            Some(TensorPlan { p0, p1, i0: p0, i1: p1, wrap: wrap_x as u32 | wrap_yz, cdt, lossy: lossy_t, op_np, op_tab, comps })
        }
    };

    let mut plan = Plan {
        name: sim.name.clone(),
        grid: grid.clone(),
        n,
        x0,
        nxl,
        dt,
        nsteps,
        inv_p,
        inv_d,
        cb,
        ca,
        ch,
        ch_scalar: cdt,
        hloss,
        bc_y,
        bc_z,
        ghost_e_hi,
        ghost_h_lo,
        bloch: bloch_x || bloch_yz,
        bloch_phase,
        periodic,
        pml,
        disp,
        nonlin,
        tensor,
        inj_h: vec![],
        inj_e: vec![],
        dft: vec![],
        time: vec![],
        energy: vec![],
        media: media.clone(),
        shutoff: sim.shutoff,
        shutoff_interval: sim.shutoff_interval.max(1),
        source_end_time: sim.sources.iter().map(|s| s.signal().end_time()).fold(0.0, f64::max),
        sources: sim.sources.clone(),
        source_analytic: vec![],
        twod: None,
        slice: ropts.slice.map(|(a, w, _)| (a, w)),
    };

    // ---- 2-D update: one periodic cell along an axis, diagonal media ----
    let allow_2d = opts.allow_2d && !sim.force_3d_kernels && std::env::var("LIGHTSIM_2D").as_deref() != Ok("0");
    // (the 2-D update has no magnetic loss)
    if allow_2d && plan.tensor.is_none() && plan.hloss.is_none() && !media.iter().any(|m| m.is_tensor()) {
        let flat = |a: usize| -> bool {
            n[a] == 1 && matches!(b.axis(a)[0], Boundary::Periodic | Boundary::Bloch { .. }) && bloch_phase[a] == (1.0, 0.0)
        };
        // a slab decomposition (along x) of a grid that is one cell thick along x makes no sense
        let axis = match inv {
            Some(a) => Some(a).filter(|&a| flat(a)),
            None => [2usize, 1, 0].into_iter().find(|&a| flat(a)),
        }
        .filter(|&a| a != 0 || opts.nslabs <= 1);
        plan.twod = axis.map(|axis| TwoD { axis, allow: sim.polarization.sets(), active: [false, false] });
    }

    // ---- sources ----
    let ctx = SourceCtx {
        grid: &grid,
        plan_x0: x0,
        nxl,
        ny,
        nz,
        dt,
        nsteps,
        eps_local: &eps_local,
        mu_local: mu_local.as_ref(),
        cb_local: &cb_local,
        ch_scalar: cdt,
        inv_p_full: &inv_p_full,
        inv_d_full: &inv_d_full,
        pml_n: [
            [b.x[0].pml_layers(), b.x[1].pml_layers()],
            [b.y[0].pml_layers(), b.y[1].pml_layers()],
            [b.z[0].pml_layers(), b.z[1].pml_layers()],
        ],
        med_local: &med_local,
        media: &media,
        structs: &structures,
        background,
        periodic,
        slice: plan.slice,
    };
    for src in &sim.sources {
        let (mut h, mut e, complex) = compile_source(&ctx, src)?;
        // a 2-D simulation restricted to one field set has no place for the
        // other set's source terms
        if let Some(a) = inv {
            let sets = sim.polarization.sets();
            let keep = |j: &Injection| sets[if crate::spec::is_te(j.comp, a) { 0 } else { 1 }];
            h.retain(keep);
            e.retain(keep);
            let drives = source_sets(src, a);
            if !(drives[0] && sets[0] || drives[1] && sets[1]) {
                let w = ["x", "y", "z"][a];
                bail!(
                    "source '{}' drives only the {} field set, which this 2-D simulation (invariant along {w}, polarization = \"{}\") does not contain",
                    src.name(),
                    if sets[0] { "tm (E along the invariant axis)" } else { "te (E in the plane)" },
                    sim.polarization.name()
                );
            }
        }
        plan.inj_h.extend(h);
        plan.inj_e.extend(e);
        plan.source_analytic.push(complex && plan.bloch);
        // the sets in use follow from the source itself, not from its entries
        // in this slab: every slab of a decomposed run must update the same sets
        if let Some(td) = plan.twod.as_mut() {
            let drives = source_sets(src, td.axis);
            for k in 0..2 {
                td.active[k] |= drives[k] && td.allow[k];
            }
        }
    }

    // ---- monitors ----
    let nonlinear = plan.nonlin.is_some() || media.iter().any(|m| m.is_nonlinear());
    for mon in &sim.monitors {
        // stride of a frequency-domain monitor: its own, the simulation's, or automatic
        let stride_of = |own: usize, freqs: &[f64]| -> usize {
            if own > 0 {
                own
            } else {
                sim.dft_stride.filter(|s| *s > 0).unwrap_or_else(|| auto_dft_stride(&sim.sources, nonlinear, dt, freqs))
            }
        };
        compile_monitor(&mut plan, mon, &sim.boundaries, &eps_local, mu_local.as_ref(), &stride_of, inv)?;
    }
    for (i, d) in plan.dft.iter_mut().enumerate() {
        d.id = i;
    }
    if opts.share_monitors {
        share_dft_boxes(&mut plan);
    }
    Ok(plan)
}

struct SourceCtx<'a> {
    grid: &'a Grid,
    plan_x0: usize,
    nxl: usize,
    ny: usize,
    nz: usize,
    dt: f64,
    nsteps: usize,
    eps_local: &'a [Vec<f32>; 3],
    mu_local: Option<&'a [Vec<f32>; 3]>,
    cb_local: &'a [Vec<f32>; 3],
    ch_scalar: f32,
    inv_p_full: &'a [Vec<f32>; 3],
    inv_d_full: &'a [Vec<f32>; 3],
    /// PML thickness (cells) on the low/high side of each axis.
    pml_n: [[usize; 2]; 3],
    med_local: &'a [Vec<u16>; 3],
    media: &'a [Medium],
    structs: &'a [crate::raster::ResolvedStructure<'a>],
    background: usize,
    periodic: [bool; 3],
    /// 2-D simulations: (invariant axis, slice plane).
    slice: Option<(usize, f64)>,
}

impl SourceCtx<'_> {
    /// Medium at a point (last structure containing it, else the background):
    /// slab-independent, so every slab of a decomposed run agrees.
    fn medium_at(&self, mut x: [f64; 3]) -> &Medium {
        if let Some((a, w)) = self.slice {
            x[a] = w;
        }
        let mut m = self.background;
        for st in self.structs {
            if st.shape.contains(x) {
                m = st.medium;
            }
        }
        &self.media[m]
    }

    /// ADE coefficients of the dispersive poles acting on E_`axis` in the
    /// dominant medium of local cell (li, j, k) (empty if non-dispersive).
    fn poles_at(&self, axis: usize, li: usize, j: usize, k: usize) -> Vec<[f64; 3]> {
        let m = &self.media[self.med_local[axis][(li * self.ny + j) * self.nz + k] as usize];
        if !m.is_dispersive() || !m.pole_axes[axis] {
            return vec![];
        }
        m.poles.iter().map(|p| p.ade_coefficients(self.dt)).collect()
    }
    /// True if global cell index `g` lies inside a PML layer. Planar source
    /// entries there are dropped: injecting into an absorber is unphysical
    /// (Lumerical refuses it) and the guided-mode fields are negligible.
    fn in_pml(&self, g: [usize; 3]) -> bool {
        // one-cell margin so that no source entry (nor the discrete curl of
        // an H entry, used by the fused GPU step) reaches into the PML
        let n = self.grid.n();
        (0..3).any(|a| {
            (self.pml_n[a][0] > 0 && g[a] <= self.pml_n[a][0]) || (self.pml_n[a][1] > 0 && g[a] + self.pml_n[a][1] + 1 >= n[a])
        })
    }
    fn local(&self, gi: usize) -> Option<usize> {
        if gi >= self.plan_x0 && gi < self.plan_x0 + self.nxl { Some(gi - self.plan_x0) } else { None }
    }
    fn lidx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }
    fn sidx(&self, i: usize, j: usize, k: usize) -> u64 {
        (((i + 1) * self.ny + j) * self.nz + k) as u64
    }
    fn cb(&self, a: usize, i: usize, j: usize, k: usize) -> f32 {
        self.cb_local[a][self.lidx(i, j, k)]
    }
    fn ch(&self, a: usize, i: usize, j: usize, k: usize) -> f32 {
        match self.mu_local {
            Some(m) => self.ch_scalar / m[a][self.lidx(i, j, k)],
            None => self.ch_scalar,
        }
    }
    fn eps(&self, a: usize, i: usize, j: usize, k: usize) -> f32 {
        self.eps_local[a][self.lidx(i, j, k)]
    }
    fn mu(&self, a: usize, i: usize, j: usize, k: usize) -> f32 {
        match self.mu_local {
            Some(m) => m[a][self.lidx(i, j, k)],
            None => 1.0,
        }
    }
    fn table(&self, f: impl Fn(f64) -> f64, t_of_n: impl Fn(usize) -> f64) -> Vec<f32> {
        (0..self.nsteps).map(|n| f(t_of_n(n)) as f32).collect()
    }
}

/// Accumulates injection entries per component, merging duplicate indices.
#[derive(Default)]
struct InjBuilder {
    // comp -> (idx -> (coef_re, coef_im, group))
    map: BTreeMap<(Component, u32), BTreeMap<u64, (f64, f64)>>,
}

impl InjBuilder {
    fn add(&mut self, comp: Component, group: u32, idx: u64, re: f64, im: f64) {
        let e = self.map.entry((comp, group)).or_default().entry(idx).or_insert((0.0, 0.0));
        e.0 += re;
        e.1 += im;
    }
    /// Build injections. `amps(group)` returns the amplitude table for that group
    /// (real: nsteps; complex: 2*nsteps interleaved).
    fn build(self, complex: bool, ngroups: usize, amps: &dyn Fn(u32) -> Vec<f32>, nsteps: usize) -> Vec<Injection> {
        // merge groups per component into one injection with group ids
        let mut per_comp: BTreeMap<Component, Vec<(u64, f64, f64, u32)>> = BTreeMap::new();
        for ((c, g), m) in self.map {
            for (idx, (re, im)) in m {
                if re == 0.0 && im == 0.0 {
                    continue;
                }
                per_comp.entry(c).or_default().push((idx, re, im, g));
            }
        }
        let width = if complex { 2 } else { 1 };
        let tables: Vec<Vec<f32>> = (0..ngroups as u32).map(amps).collect();
        let mut table = vec![0f32; nsteps * ngroups * width];
        for n in 0..nsteps {
            for g in 0..ngroups {
                for w in 0..width {
                    table[(n * ngroups + g) * width + w] = tables[g][n * width + w];
                }
            }
        }
        per_comp
            .into_iter()
            .map(|(c, v)| Injection {
                comp: c,
                idx: v.iter().map(|e| e.0).collect(),
                coef: v.iter().map(|e| e.1 as f32).collect(),
                coef_im: if complex { Some(v.iter().map(|e| e.2 as f32).collect()) } else { None },
                group: v.iter().map(|e| e.3).collect(),
                ngroups,
                amps: table.clone(),
            })
            .collect()
    }
}

/// Transverse profile for a planar (TFSF-style) source.
struct PlanarProfile<'a> {
    /// E_b, E_c at plane p (complex) as function of (global j-ish indices).
    e: Box<dyn Fn(usize, [usize; 3]) -> Complex64 + 'a>,
    /// H~_b, H~_c (normalised eta0*H) at plane p-1/2.
    h: Box<dyn Fn(usize, [usize; 3]) -> Complex64 + 'a>,
    complex: bool,
    /// Effective index for the E/H half-cell delay.
    n_delay: f64,
    /// Transverse phase-delay function (s) for grouped/oblique delays: None.
    signal: TimeSignal,
    amplitude: f64,
    /// Incident-field amplitude tables from a 1-D auxiliary line (dispersive
    /// backgrounds): (E_inc at plane p at t_n, H~_inc / (n/mu) at p-1/2 at
    /// t_{n+1/2}). Replace the analytic signal when present.
    aux_amps: Option<(Vec<f32>, Vec<f32>)>,
}

/// Compile a one-sided TFSF plane at global node index `p` along `a`.
fn planar_injections(
    ctx: &SourceCtx,
    a: usize,
    p: usize,
    dir: Direction,
    tlo: [usize; 3],
    thi: [usize; 3],
    prof: &PlanarProfile,
) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    if p == 0 {
        bail!("planar source plane must not sit on the low boundary node");
    }
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let sgn = dir.sign();
    let mut hb = InjBuilder::default();
    let mut eb = InjBuilder::default();
    let grid = ctx.grid;
    // index ranges for each component in transverse axes
    let rng = |c: Component, ax: usize| -> (usize, usize) {
        let n_ax = grid.n()[ax];
        let lo = tlo[ax];
        let hi = thi[ax].min(n_ax);
        if c.is_half(ax) { (lo, hi.max(lo + 1).min(n_ax)) } else { (lo, (hi + 1).min(n_ax)) }
    };
    let id_d = ctx.inv_d_full[a][p] as f64;
    let id_p = ctx.inv_p_full[a][p - 1] as f64;
    // Iterate E_b and E_c at plane p, H_b and H_c at p-1.
    for (which, comp_axis) in [(0usize, bax), (1usize, cax)] {
        let ec = Component::e(comp_axis);
        let hc = Component::h(comp_axis);
        // E corrections: E_b += +s cb id H_c ; E_c += -s cb id H_b
        {
            let (r1lo, r1hi) = rng(ec, bax);
            let (r2lo, r2hi) = rng(ec, cax);
            for u in r1lo..r1hi {
                for v in r2lo..r2hi {
                    let mut g = [0usize; 3];
                    g[a] = p;
                    g[bax] = u;
                    g[cax] = v;
                    if ctx.in_pml(g) {
                        continue;
                    }
                    let Some(li) = ctx.local(g[0]) else { continue };
                    let other = if which == 0 { 1 } else { 0 }; // H_c for E_b, H_b for E_c
                    let hval = (prof.h)(other, g);
                    let sign_c = if which == 0 { 1.0 } else { -1.0 };
                    let cbv = ctx.cb(comp_axis, li, g[1], g[2]) as f64;
                    let coef = Complex64::new(sgn * sign_c * cbv * id_d, 0.0) * hval;
                    let idx = ctx.sidx(li, g[1], g[2]);
                    // amp is the analytic signal A; field += Re(coef * A) = re*Re A - im*Im A
                    eb.add(ec, 0, idx, coef.re, -coef.im);
                }
            }
        }
        // H corrections at p-1: H_b += -s ch id E_c ; H_c += +s ch id E_b
        {
            let (r1lo, r1hi) = rng(hc, bax);
            let (r2lo, r2hi) = rng(hc, cax);
            for u in r1lo..r1hi {
                for v in r2lo..r2hi {
                    let mut g = [0usize; 3];
                    g[a] = p - 1;
                    g[bax] = u;
                    g[cax] = v;
                    if ctx.in_pml(g) {
                        continue;
                    }
                    let Some(li) = ctx.local(g[0]) else { continue };
                    let other = if which == 0 { 1 } else { 0 };
                    let mut ge = g;
                    ge[a] = p;
                    let eval = (prof.e)(other, ge);
                    let sign_c = if which == 0 { -1.0 } else { 1.0 };
                    let chv = ctx.ch(comp_axis, li, g[1], g[2]) as f64;
                    let coef = Complex64::new(sgn * sign_c * chv * id_p, 0.0) * eval;
                    let idx = ctx.sidx(li, g[1], g[2]);
                    hb.add(hc, 0, idx, coef.re, -coef.im);
                }
            }
        }
    }
    // Delays: E_inc at plane p (t_n); H_inc at p-1/2 (t_{n+1/2}).
    let dt = ctx.dt;
    let delta = grid.node(a, p) - grid.center(a, p - 1);
    // The H plane sits half a cell below the E plane: upstream for +dir (the
    // wave arrives earlier), downstream for -dir.
    let tau_h = -sgn * delta * prof.n_delay / C0;
    let sig = prof.signal.clone();
    let amp = prof.amplitude;
    let complex = prof.complex;
    let nsteps = ctx.nsteps;
    let mk = |t_of_n: &dyn Fn(usize) -> f64| -> Vec<f32> {
        if complex {
            let mut v = Vec::with_capacity(2 * nsteps);
            for n in 0..nsteps {
                let z = sig.analytic(t_of_n(n)) * amp;
                v.push(z.re as f32);
                v.push(z.im as f32);
            }
            v
        } else {
            (0..nsteps).map(|n| (amp * sig.value(t_of_n(n))) as f32).collect()
        }
    };
    // For +dir, the H plane (upstream) sees the wave earlier: s(t + delta n/c).
    let (h_amps, e_amps) = match &prof.aux_amps {
        Some((ha, ea)) => (ha.clone(), ea.clone()),
        None => (
            mk(&|n| (n as f64) * dt),               // E_inc used by H corrections at t_n
            mk(&|n| (n as f64 + 0.5) * dt - tau_h), // H_inc used by E corrections at t_{n+1/2}
        ),
    };
    let _ = sgn;
    let inj_h = hb.build(complex, 1, &|_| h_amps.clone(), nsteps);
    let inj_e = eb.build(complex, 1, &|_| e_amps.clone(), nsteps);
    Ok((inj_h, inj_e, complex))
}

/// Injection lists of a source for this slab, and whether the source has a
/// complex field pattern (the same answer on every slab, also on slabs the
/// source does not touch).
fn compile_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let grid = ctx.grid;
    let n = grid.n();
    match src {
        Source::PointDipole { position, polarization, signal, amplitude, .. } => {
            let c = *polarization;
            let a = c.axis();
            let g = [
                grid.nearest_index(c, 0, position[0]),
                grid.nearest_index(c, 1, position[1]),
                grid.nearest_index(c, 2, position[2]),
            ];
            let Some(li) = ctx.local(g[0]) else { return Ok((vec![], vec![], false)) };
            // discrete-curl volume (dual/primal spacings as in the update), so a
            // source on a symmetry plane matches the full-domain injection
            let mut vol = 1.0;
            for ax in 0..3 {
                let inv = if c.is_half(ax) { ctx.inv_p_full[ax][g[ax]] } else { ctx.inv_d_full[ax][g[ax]] };
                vol /= inv as f64;
            }
            let idx = ctx.sidx(li, g[1], g[2]);
            let mut bld = InjBuilder::default();
            let sig = signal.clone();
            let amp = *amplitude;
            if c.is_e() {
                // dE = -(dt/(eps0 eps_r)) J,  J = I l / V
                let cbv = ctx.cb(a, li, g[1], g[2]) as f64;
                let coef = -(cbv / (C0 * EPS0)) / vol;
                bld.add(c, 0, idx, coef, 0.0);
                let dt = ctx.dt;
                let table = ctx.table(|t| amp * sig.value(t), |n| (n as f64 + 0.5) * dt);
                Ok((vec![], bld.build(false, 1, &|_| table.clone(), ctx.nsteps), false))
            } else {
                // dH~ = -(c dt / mu_r) M,  M = K / V
                let chv = ctx.ch(a, li, g[1], g[2]) as f64;
                let coef = -chv / vol;
                bld.add(c, 0, idx, coef, 0.0);
                let dt = ctx.dt;
                let table = ctx.table(|t| amp * sig.value(t), |n| n as f64 * dt);
                Ok((bld.build(false, 1, &|_| table.clone(), ctx.nsteps), vec![], false))
            }
        }
        Source::PlaneWave { axis, position, direction, polarization, signal, amplitude, angle, min, max, .. } => {
            let a = *axis;
            let bax = (a + 1) % 3;
            let cax = (a + 2) % 3;
            if polarization.axis() == a {
                bail!("plane wave polarization must be a component tangential to the plane");
            }
            // An H component asks for p-polarisation: H along that axis, E
            // (and the tilt of the wavevector) along the other in-plane axis.
            let h_pol = !polarization.is_e();
            let pol = if h_pol { Component::e(if polarization.axis() == bax { cax } else { bax }) } else { *polarization };
            let p = grid.nearest_node(a, *position).max(1);
            let (tlo, thi) = transverse_box(grid, a, min.as_ref(), max.as_ref());
            // local medium at the plane centre for impedance
            let (nmed, mur) = plane_medium(ctx, a, p, pol.axis());
            let s = direction.sign();
            let th = *angle;
            let k0 = TAU * signal.freq0().unwrap_or(0.0) / C0;
            let q = if pol.axis() == bax { cax } else { bax };
            // 2-D simulations: the k-vector stays in the plane. An E
            // polarisation in the plane is then tilted toward its own axis
            // (p-polarisation: tangential E = A cos(th), H~ = (n/mu) A along
            // the invariant axis) instead of out of the plane.
            // In 3-D an H polarisation does the same: H along the axis q.
            let in_plane_e = ctx.slice.map(|(w, _)| w == q).unwrap_or(false);
            if h_pol && th != 0.0 && ctx.slice.is_some() && !in_plane_e {
                bail!("2-D plane wave: an H polarisation at an angle must be along the invariant axis");
            }
            let p_pol = th != 0.0 && (h_pol || in_plane_e);
            let q = if p_pol { pol.axis() } else { q };
            let (e_amp, h_cos) = if p_pol { (th.cos(), 1.0) } else { (1.0, th.cos()) };
            let q0 = grid.position(Component::e(q), q, (tlo[q] + thi[q]) / 2);
            let complex = th != 0.0;
            let pol_is_b = pol.axis() == bax;
            let prof = PlanarProfile {
                e: Box::new(move |which, g| {
                    let on = (which == 0) == pol_is_b;
                    if !on {
                        return Complex64::new(0.0, 0.0);
                    }
                    if complex {
                        let r = grid.position(Component::e(if which == 0 { bax } else { cax }), q, g[q]);
                        Complex64::from_polar(e_amp, k0 * nmed * th.sin() * (r - q0))
                    } else {
                        Complex64::new(1.0, 0.0)
                    }
                }),
                h: Box::new(move |which, g| {
                    // H~ = (n/mu) k x E (tangential part): pol b -> H~_c = s n cos(th) A ; pol c -> H~_b = -s n cos A
                    let mag = s * nmed / mur * h_cos;
                    let val = if pol_is_b {
                        if which == 1 { mag } else { 0.0 }
                    } else if which == 0 {
                        -mag
                    } else {
                        0.0
                    };
                    if val == 0.0 {
                        return Complex64::new(0.0, 0.0);
                    }
                    if complex {
                        let comp = Component::h(if which == 0 { bax } else { cax });
                        let r = grid.position(comp, q, g[q]);
                        Complex64::from_polar(val, k0 * nmed * th.sin() * (r - q0))
                    } else {
                        Complex64::new(val, 0.0)
                    }
                }),
                complex,
                n_delay: nmed * th.cos(),
                signal: signal.clone(),
                amplitude: *amplitude,
                aux_amps: None,
            };
            let _ = n;
            // dispersive background at normal incidence: take the incident E
            // and H from a 1-D auxiliary line with the same ADE poles
            let mut prof = prof;
            if th == 0.0 {
                let ng = grid.n();
                let mut gc = [ng[0] / 2, ng[1] / 2, ng[2] / 2];
                gc[a] = p;
                let pc = [grid.node(0, gc[0]), grid.node(1, gc[1]), grid.node(2, gc[2])];
                let md = ctx.medium_at(pc);
                let pa = pol.axis();
                {
                    let poles: Vec<[f64; 3]> =
                        if md.is_dispersive() && md.pole_axes[pa] { md.poles.iter().map(|q| q.ade_coefficients(ctx.dt)).collect() } else { vec![] };
                    if !poles.is_empty() {
                        let r_ref = grid.node(a, p);
                        let aux = Aux1d::run(grid, a, s, p, p, nmed, mur, ctx.dt, ctx.nsteps, signal, *amplitude, r_ref, &poles);
                        let imp = nmed / mur;
                        let h_amps = aux.e_at_node(p);
                        let e_amps: Vec<f32> = aux.h_at_half(p - 1).iter().map(|v| v / imp as f32).collect();
                        prof.aux_amps = Some((h_amps, e_amps));
                    }
                }
            }
            planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
        }
        Source::GaussianBeam {
            axis,
            position,
            direction,
            polarization,
            center,
            waist,
            signal,
            amplitude,
            focus_distance,
            theta,
            phi,
            ..
        } => {
            let a = *axis;
            let pol = *polarization;
            if !pol.is_e() || pol.axis() == a {
                bail!("Gaussian beam polarization must be an E component tangential to the plane");
            }
            if !(theta.abs() < 0.5 * std::f64::consts::PI - 1e-6) {
                bail!("Gaussian beam: theta = {theta} rad; the beam axis must make less than 90 degrees with the plane normal");
            }
            let p = grid.nearest_node(a, *position).max(1);
            let (tlo, thi) = transverse_box(grid, a, None, None);
            let bax = (a + 1) % 3;
            let cax = (a + 2) % 3;
            let (nmed, mur) = plane_medium(ctx, a, p, pol.axis());
            let s = direction.sign();
            let f0 = signal.freq0().unwrap_or(0.0);
            let k = TAU * f0 * nmed / C0;
            let w0 = *waist;
            let zr = std::f64::consts::PI * w0 * w0 * nmed * f0 / C0;
            let fd = *focus_distance;
            // beam axis d, polarization e (the polarization axis projected
            // onto the plane normal to d) and h = d x e
            let (u_ax, v_ax) = crate::geometry::plane_axes(a);
            let mut d = [0.0f64; 3];
            d[a] = s * theta.cos();
            d[u_ax] = theta.sin() * phi.cos();
            d[v_ax] = theta.sin() * phi.sin();
            let mut e = [0.0f64; 3];
            e[pol.axis()] = 1.0;
            let ed = e[0] * d[0] + e[1] * d[1] + e[2] * d[2];
            for m in 0..3 {
                e[m] -= ed * d[m];
            }
            let en = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
            if en < 1e-6 {
                bail!("Gaussian beam: the beam axis is along the polarization axis");
            }
            for v in e.iter_mut() {
                *v /= en;
            }
            let h = [d[1] * e[2] - d[2] * e[1], d[2] * e[0] - d[0] * e[2], d[0] * e[1] - d[1] * e[0]];
            // a sheet beam along the invariant axis of a 2-D simulation, and along
            // periodic in-plane axes narrower than the beam
            let inv = ctx.slice.map(|(w, _)| w);
            let mut flat = [false; 3];
            for q in [bax, cax] {
                let edges = grid.edges(q);
                flat[q] = inv == Some(q) || (ctx.periodic[q] && (edges[edges.len() - 1] - edges[0]) < 2.0 * w0);
                if flat[q] && d[q].abs() > 1e-9 {
                    bail!("Gaussian beam: the simulation is uniform along axis {q} (periodic and narrower than the beam), so the beam axis must have no component along it");
                }
            }
            let sheet = flat[bax] || flat[cax];
            if flat[bax] && flat[cax] {
                bail!("Gaussian beam: the source plane is narrower than the beam along both in-plane axes; use a plane wave");
            }
            let complex = fd != 0.0 || *theta != 0.0;
            let mut ctr = *center;
            ctr[a] = grid.node(a, p);
            let field = move |comp: Component, g: [usize; 3]| -> Complex64 {
                // offset from the beam centre on the plane (zero along a uniform axis)
                let mut rho = [0.0f64; 3];
                for m in [bax, cax] {
                    if !flat[m] {
                        rho[m] = grid.position(comp, m, g[m]) - ctr[m];
                    }
                }
                let along = rho[0] * d[0] + rho[1] * d[1] + rho[2] * d[2];
                let r2 = (rho[0] * rho[0] + rho[1] * rho[1] + rho[2] * rho[2] - along * along).max(0.0);
                let z = along - fd; // position along the beam axis, from the focus
                let wz = w0 * (1.0 + (z / zr).powi(2)).sqrt();
                let inv_r = z / (z * z + zr * zr);
                let gouy = (z / zr).atan();
                // a sheet beam spreads in one direction only
                let (amp, gouy) = if sheet { ((w0 / wz).sqrt() * (-r2 / (wz * wz)).exp(), 0.5 * gouy) } else { (w0 / wz * (-r2 / (wz * wz)).exp(), gouy) };
                if !complex {
                    return Complex64::new(amp, 0.0);
                }
                // exp(-i w t) convention: forward wave exp(+i k z); the phase is zero at the beam centre on the plane
                Complex64::from_polar(amp, k * along + k * r2 * inv_r / 2.0 - gouy)
            };
            let prof = PlanarProfile {
                e: Box::new(move |which, g| {
                    let m = if which == 0 { bax } else { cax };
                    if e[m] == 0.0 {
                        return Complex64::new(0.0, 0.0);
                    }
                    field(Component::e(m), g) * e[m]
                }),
                h: Box::new(move |which, g| {
                    let m = if which == 0 { bax } else { cax };
                    if h[m] == 0.0 {
                        return Complex64::new(0.0, 0.0);
                    }
                    field(Component::h(m), g) * (h[m] * nmed / mur)
                }),
                complex,
                // the half cell between the E and H planes, along the normal
                n_delay: nmed * theta.cos(),
                signal: signal.clone(),
                amplitude: *amplitude,
                aux_amps: None,
            };
            planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
        }
        Source::Mode { axis, position, direction, min, max, profile, signal, amplitude, .. } => {
            let a = *axis;
            let p = grid.nearest_node(a, *position).max(1);
            let (tlo, thi) = transverse_box(grid, a, Some(min), Some(max));
            let bax = (a + 1) % 3;
            let cax = (a + 2) % 3;
            // profile arrays are over the plane's natural-order transverse axes
            let (u_ax, v_ax) = crate::geometry::plane_axes(a);
            let dims = profile.dims;
            let s = direction.sign();
            // map (which in cyclic (b,c)) -> index in natural (u,v) order
            let nat = |cyclic_axis: usize| -> usize { if cyclic_axis == u_ax { 0 } else { 1 } };
            let lo = tlo;
            let get = move |arr_re: &Vec<f64>, arr_im: &Vec<f64>, g: [usize; 3]| -> Complex64 {
                let iu = g[u_ax] as isize - lo[u_ax] as isize;
                let iv = g[v_ax] as isize - lo[v_ax] as isize;
                if iu < 0 || iv < 0 || iu as usize >= dims[0] || iv as usize >= dims[1] {
                    return Complex64::new(0.0, 0.0);
                }
                let o = iu as usize * dims[1] + iv as usize;
                Complex64::new(arr_re[o], arr_im.get(o).copied().unwrap_or(0.0))
            };
            // The profile stores H half a cell below the plane, for a mode
            // travelling along +axis: H_m(p) exp(-i beta d). The injection
            // delays the H signal in time by that half cell (`tau_h`, for
            // either direction and every frequency of the pulse), so the
            // profile's own shift is taken out here; otherwise the half cell
            // would be counted twice and part of the power would be launched
            // backwards. An absorbing mode keeps its amplitude step over the
            // half cell, exp(+-Im(beta) d).
            let d_h = profile.h_offset.unwrap_or(grid.node(a, p) - grid.center(a, p - 1));
            let beta = profile.beta();
            let unshift = (Complex64::i() * beta * d_h).exp() * (s * beta.im * d_h).exp();
            let fix = |re: &Vec<f64>, im: &Vec<f64>| -> (Vec<f64>, Vec<f64>) {
                let z: Vec<Complex64> = (0..re.len()).map(|o| Complex64::new(re[o], im.get(o).copied().unwrap_or(0.0)) * unshift).collect();
                (z.iter().map(|v| v.re).collect(), z.iter().map(|v| v.im).collect())
            };
            let h_fixed = [fix(&profile.h_re[0], &profile.h_im[0]), fix(&profile.h_re[1], &profile.h_im[1])];
            // complex injection only when the pattern has a phase of its own
            let peak = profile
                .e_re
                .iter()
                .chain(profile.e_im.iter())
                .flat_map(|v| v.iter())
                .fold(0f64, |m, x| m.max(x.abs()))
                .max(h_fixed.iter().flat_map(|(r, i)| r.iter().chain(i.iter())).fold(0f64, |m, x| m.max(x.abs() * ETA0)));
            let complex = profile.e_im.iter().any(|v| v.iter().any(|x| x.abs() > 1e-9 * peak))
                || h_fixed.iter().any(|(_, i)| i.iter().any(|x| x.abs() * ETA0 > 1e-9 * peak));
            let pr = profile.clone();
            let prof = PlanarProfile {
                e: Box::new(move |which, g| {
                    let ax = if which == 0 { bax } else { cax };
                    let t = nat(ax);
                    get(&pr.e_re[t], &pr.e_im[t], g)
                }),
                h: Box::new(move |which, g| {
                    let ax = if which == 0 { bax } else { cax };
                    let t = nat(ax);
                    // backward injection flips H; convert to H~ = eta0 H
                    get(&h_fixed[t].0, &h_fixed[t].1, g) * (s * ETA0)
                }),
                complex,
                n_delay: profile.neff,
                signal: signal.clone(),
                amplitude: *amplitude,
                aux_amps: None,
            };
            planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
        }
        Source::Tfsf { min, max, axis, direction, polarization, signal, amplitude, .. } => {
            let (h, e) = compile_tfsf(ctx, min, max, *axis, *direction, *polarization, signal, *amplitude)?;
            Ok((h, e, false))
        }
    }
}

/// `[te, tm]`: the field sets of a 2-D problem invariant along `w` that a
/// source drives (see [`TwoD`]).
pub fn source_sets(src: &Source, w: usize) -> [bool; 2] {
    let of = |c: Component| if crate::spec::is_te(c, w) { [true, false] } else { [false, true] };
    match src {
        Source::PointDipole { polarization, .. } | Source::PlaneWave { polarization, .. } | Source::GaussianBeam { polarization, .. } | Source::Tfsf { polarization, .. } => of(*polarization),
        Source::Mode { axis, profile, .. } => {
            // tangential E and H of the profile, over the plane's natural axes
            let (ua, va) = crate::geometry::plane_axes(*axis);
            let mut out = [false, false];
            let nz = |re: &Vec<f64>, im: &Vec<f64>| re.iter().chain(im.iter()).any(|x| *x != 0.0);
            for (t, ax) in [ua, va].into_iter().enumerate() {
                if nz(&profile.e_re[t], &profile.e_im[t]) {
                    let k = of(Component::e(ax));
                    out = [out[0] || k[0], out[1] || k[1]];
                }
                if nz(&profile.h_re[t], &profile.h_im[t]) {
                    let k = of(Component::h(ax));
                    out = [out[0] || k[0], out[1] || k[1]];
                }
            }
            out
        }
    }
}

/// Transverse index box for a plane normal to `a` (nodes); defaults to all.
pub fn transverse_box(grid: &Grid, a: usize, min: Option<&[f64; 3]>, max: Option<&[f64; 3]>) -> ([usize; 3], [usize; 3]) {
    let n = grid.n();
    let mut lo = [0usize; 3];
    let mut hi = n;
    for ax in 0..3 {
        if ax == a {
            continue;
        }
        if let (Some(mn), Some(mx)) = (min, max) {
            // A window edge that lies on a grid node (to rounding) is that
            // node: the window starts with the cell above it and ends with
            // the cell below it. Without the tolerance the last bit of the
            // coordinate decided whether a neighbouring cell was included.
            let e = grid.edges(ax);
            let tol = 1e-9 * (e[n[ax]] - e[0]).abs().max(1e-30);
            lo[ax] = grid.cell_index(ax, (mn[ax] + tol).max(e[0]));
            hi[ax] = (grid.cell_index(ax, (mx[ax] - tol).min(e[n[ax]])) + 1).min(n[ax]).max(lo[ax] + 1);
        }
    }
    (lo, hi)
}

/// Refractive index and mu_r at the centre of the injection plane.
fn plane_medium(ctx: &SourceCtx, a: usize, p: usize, pol_axis: usize) -> (f64, f64) {
    let n = ctx.grid.n();
    let mut g = [n[0] / 2, n[1] / 2, n[2] / 2];
    g[a] = p;
    // pick a slab-local x if needed
    let li = match ctx.local(g[0]) {
        Some(v) => v,
        None => {
            if a == 0 {
                return (1.0, 1.0);
            }
            ctx.nxl / 2
        }
    };
    let eps = ctx.eps(pol_axis, li, g[1], g[2]) as f64;
    let mu = ctx.mu(pol_axis, li, g[1], g[2]) as f64;
    ((eps * mu).sqrt(), mu)
}

#[allow(clippy::too_many_arguments)]
fn compile_tfsf(
    ctx: &SourceCtx,
    min: &[f64; 3],
    max: &[f64; 3],
    a: usize,
    dir: Direction,
    pol: Component,
    signal: &TimeSignal,
    amplitude: f64,
) -> Result<(Vec<Injection>, Vec<Injection>)> {
    if !pol.is_e() || pol.axis() == a {
        bail!("TFSF polarization must be an E component perpendicular to the propagation axis");
    }
    let grid = ctx.grid;
    let n = grid.n();
    // node index box of the TF region; a transverse axis that is periodic and
    // fully spanned by the box has no faces (a periodic "slab" TFSF, e.g. 2-D
    // scattering problems)
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    let mut span = [false; 3];
    for ax in 0..3 {
        let e = grid.edges(ax);
        span[ax] = ax != a && ctx.periodic[ax] && min[ax] <= e[0] + 1e-15 && max[ax] >= e[n[ax]] - 1e-15;
        if span[ax] {
            lo[ax] = 0;
            hi[ax] = n[ax];
            continue;
        }
        lo[ax] = grid.nearest_node(ax, min[ax]).max(1);
        hi[ax] = grid.nearest_node(ax, max[ax]).min(n[ax] - 1);
        if hi[ax] <= lo[ax] {
            bail!("TFSF box too small along axis {ax}");
        }
    }
    let p_ax = pol.axis();
    let s = dir.sign();
    // background medium at the box centre
    let mut gc = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    gc[a] = lo[a];
    // background medium at the box face centre, looked up geometrically so
    // that every slab of a decomposed run builds the same incident wave
    let pc = [grid.node(0, gc[0]), grid.node(1, gc[1]), grid.node(2, gc[2])];
    let md = ctx.medium_at(pc);
    let (nmed, mur) = ((md.eps[p_ax] * md.mu[p_ax]).sqrt(), md.mu[p_ax]);
    let poles: Vec<[f64; 3]> =
        if md.is_dispersive() && md.pole_axes[p_ax] { md.poles.iter().map(|q| q.ade_coefficients(ctx.dt)).collect() } else { vec![] };
    // incident E = A e_pol s(t - (r_a - r_ref) s n / c); H~ = s (n/mu) (a_hat x e_pol)
    let r_ref = if s > 0.0 { grid.node(a, lo[a]) } else { grid.node(a, hi[a]) };
    // H~ direction: a_hat x e_pol
    let mut hdir = [0.0f64; 3];
    {
        let mut av = [0.0; 3];
        av[a] = 1.0;
        let mut ev = [0.0; 3];
        ev[p_ax] = 1.0;
        hdir[0] = av[1] * ev[2] - av[2] * ev[1];
        hdir[1] = av[2] * ev[0] - av[0] * ev[2];
        hdir[2] = av[0] * ev[1] - av[1] * ev[0];
    }
    // ---- 1-D auxiliary FDTD line for the incident field ----
    // The incident wave is propagated on a 1-D Yee line with the same spacing
    // along `a` and the same time step, so its numerical dispersion matches
    // the 3-D grid exactly (no leakage into the scattered-field region).
    let aux = Aux1d::run(grid, a, s, lo[a], hi[a], nmed, mur, ctx.dt, ctx.nsteps, signal, amplitude, r_ref, &poles);
    let mut hb = InjBuilder::default();
    let mut eb = InjBuilder::default();
    let mut e_groups: Vec<usize> = vec![]; // half index j along a (H_inc position)
    let mut h_groups: Vec<usize> = vec![]; // node index i along a (E_inc position)
    let gid = |list: &mut Vec<usize>, key: usize| -> u32 {
        match list.iter().position(|&k| k == key) {
            Some(p) => p as u32,
            None => {
                list.push(key);
                (list.len() - 1) as u32
            }
        }
    };

    for f in 0..3 {
        if span[f] {
            continue;
        }
        let g_ax = (f + 1) % 3;
        let h_ax = (f + 2) % 3;
        for side in 0..2 {
            let q = if side == 0 { lo[f] } else { hi[f] };
            let sside = if side == 0 { 1.0 } else { -1.0 };
            // H plane index adjacent outside: lo -> q-1 (pos q-1/2), hi -> q (pos q+1/2)
            let qh = if side == 0 { q - 1 } else { q };
            let id_d = ctx.inv_d_full[f][q] as f64;
            let id_p = ctx.inv_p_full[f][qh] as f64;
            for (which, comp_ax) in [(0usize, g_ax), (1usize, h_ax)] {
                let other_ax = if which == 0 { h_ax } else { g_ax };
                let r = |ax: usize, c: Component| -> (usize, usize) {
                    if span[ax] {
                        (0, n[ax])
                    } else if c.is_half(ax) {
                        (lo[ax], hi[ax])
                    } else {
                        (lo[ax], hi[ax] + 1)
                    }
                };
                // ---- E corrections on the face (TF E nodes): need H~_inc(other) ----
                let hdir_v = hdir[other_ax] * s; // sign of the physical H~ component along other_ax
                if hdir_v != 0.0 {
                    let ec = Component::e(comp_ax);
                    let (u0, u1) = r(g_ax, ec);
                    let (v0, v1) = r(h_ax, ec);
                    for u in u0..u1 {
                        for v in v0..v1 {
                            let mut gg = [0usize; 3];
                            gg[f] = q;
                            gg[g_ax] = u;
                            gg[h_ax] = v;
                            let Some(li) = ctx.local(gg[0]) else { continue };
                            let sign_c = if which == 0 { 1.0 } else { -1.0 };
                            let cbv = ctx.cb(comp_ax, li, gg[1], gg[2]) as f64;
                            let coef = sside * sign_c * cbv * id_d * hdir_v;
                            let j = if f == a { qh } else { gg[a] };
                            let g = gid(&mut e_groups, j);
                            eb.add(ec, g, ctx.sidx(li, gg[1], gg[2]), coef, 0.0);
                        }
                    }
                }
                // ---- H corrections just outside (SF H nodes): need E_inc(other) ----
                if other_ax == p_ax {
                    let hc = Component::h(comp_ax);
                    let (u0, u1) = r(g_ax, hc);
                    let (v0, v1) = r(h_ax, hc);
                    for u in u0..u1 {
                        for v in v0..v1 {
                            let mut gg = [0usize; 3];
                            gg[f] = qh;
                            gg[g_ax] = u;
                            gg[h_ax] = v;
                            let Some(li) = ctx.local(gg[0]) else { continue };
                            let sign_c = if which == 0 { -1.0 } else { 1.0 };
                            let chv = ctx.ch(comp_ax, li, gg[1], gg[2]) as f64;
                            let coef = sside * sign_c * chv * id_p;
                            let i = if f == a { q } else { gg[a] };
                            let g = gid(&mut h_groups, i);
                            hb.add(hc, g, ctx.sidx(li, gg[1], gg[2]), coef, 0.0);
                        }
                    }
                }
            }
        }
    }
    let nsteps = ctx.nsteps;
    let amp_h = |g: u32| -> Vec<f32> { aux.e_at_node(h_groups[g as usize]) };
    let amp_e = |g: u32| -> Vec<f32> { aux.h_at_half(e_groups[g as usize]) };
    let inj_h = hb.build(false, h_groups.len().max(1), &amp_h, nsteps);
    let inj_e = eb.build(false, e_groups.len().max(1), &amp_e, nsteps);
    Ok((inj_h, inj_e))
}

/// 1-D auxiliary Yee line carrying the incident plane wave of a TFSF box.
///
/// Scalar fields: `e` (E along the polarization) at the 3-D grid's nodes
/// along the propagation axis and `h` (normalised H~ along a x e_pol, signed
/// so that a forward wave has h = (n/mu) e) at the half nodes. A hard source
/// drives the upstream end, a graded matched absorber terminates the
/// downstream end. Samples: e at t_n (used by the H corrections of step n),
/// h at t_{n+1/2} (used by the E corrections of step n).
struct Aux1d {
    dir: f64,
    g0: usize,
    g1: usize,
    nsteps: usize,
    /// e samples [k][n] for k = aux node index of the needed global nodes
    e_samp: Vec<Vec<f32>>,
    h_samp: Vec<Vec<f32>>,
}

impl Aux1d {
    #[allow(clippy::too_many_arguments)]
    fn run(
        grid: &Grid,
        a: usize,
        dir: f64,
        lo: usize,
        hi: usize,
        nmed: f64,
        mur: f64,
        dt: f64,
        nsteps: usize,
        signal: &TimeSignal,
        amplitude: f64,
        r_ref: f64,
        poles: &[[f64; 3]],
    ) -> Aux1d {
        let e = grid.edges(a);
        let na = e.len() - 1;
        let g0 = lo.saturating_sub(2);
        let g1 = (hi + 2).min(na);
        // aux node coordinates in the propagation frame
        let mut xs: Vec<f64> = if dir > 0.0 {
            (g0..=g1).map(|i| e[i]).collect()
        } else {
            (0..=(g1 - g0)).map(|k| -e[g1 - k]).collect()
        };
        let dlast = xs[xs.len() - 1] - xs[xs.len() - 2];
        let n_abs = 80usize;
        let n_buf = 20usize;
        let last = *xs.last().unwrap();
        xs.extend((1..=(n_buf + n_abs)).map(|t| last + t as f64 * dlast));
        let nn = xs.len(); // nodes
        let nh = nn - 1; // half nodes
        let eps_r = nmed * nmed / mur;
        let cdt = C0 * dt;
        // spacings
        let dxp: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let dxd: Vec<f64> = (0..nn).map(|k| if k == 0 { dxp[0] } else if k == nh { dxp[nh - 1] } else { 0.5 * (dxp[k - 1] + dxp[k]) }).collect();
        // graded loss in the last n_abs cells
        let abs_start = xs[nn - 1 - n_abs];
        let lab = xs[nn - 1] - abs_start;
        let alpha = |x: f64| -> f64 {
            if x <= abs_start { 0.0 } else { 0.35 * ((x - abs_start) / lab).powi(3) }
        };
        let ce: Vec<(f64, f64)> = (0..nn)
            .map(|k| {
                let al = alpha(xs[k]);
                ((1.0 - al) / (1.0 + al), cdt / eps_r / (1.0 + al))
            })
            .collect();
        let chh: Vec<(f64, f64)> = (0..nh)
            .map(|k| {
                let al = alpha(0.5 * (xs[k] + xs[k + 1]));
                ((1.0 - al) / (1.0 + al), cdt / mur / (1.0 + al))
            })
            .collect();
        // source delay so that e(r_ref, t) ~ s(t)
        let x_ref = if dir > 0.0 { r_ref } else { -r_ref };
        let tau0 = (x_ref - xs[0]) * nmed / C0;
        let src = |t: f64| amplitude * signal.value(t + tau0);
        let ncore = g1 - g0 + 1;
        let mut ev = vec![0.0f64; nn];
        let mut hv = vec![0.0f64; nh];
        // dispersive background: the same ADE as the 3-D grid (P^{n+1} from
        // E^n, E -= dP / eps_inf), so the incident wave obeys the identical
        // discrete dispersion relation
        let np = poles.len();
        let mut pp = vec![0.0f64; np * nn];
        let mut ppv = vec![0.0f64; np * nn];
        let mut dp = vec![0.0f64; nn];
        let mut e_samp = vec![Vec::with_capacity(nsteps); ncore];
        let mut h_samp = vec![Vec::with_capacity(nsteps); ncore];
        ev[0] = src(0.0);
        for n in 0..nsteps {
            for (k, s) in e_samp.iter_mut().enumerate() {
                s.push(ev[k] as f32);
            }
            for k in 0..nh {
                hv[k] = chh[k].0 * hv[k] - chh[k].1 * (ev[k + 1] - ev[k]) / dxp[k];
            }
            for (k, s) in h_samp.iter_mut().enumerate() {
                s.push(hv[k.min(nh - 1)] as f32);
            }
            if np > 0 {
                for k in 0..nn {
                    let mut acc = 0.0;
                    for (q, c) in poles.iter().enumerate() {
                        let o = q * nn + k;
                        let p0 = pp[o];
                        let pn = c[0] * p0 + c[1] * ppv[o] + c[2] * ev[k];
                        ppv[o] = p0;
                        pp[o] = pn;
                        acc += pn - p0;
                    }
                    dp[k] = acc;
                }
            }
            for k in 1..nn - 1 {
                ev[k] = ce[k].0 * ev[k] - ce[k].1 * (hv[k] - hv[k - 1]) / dxd[k];
                if np > 0 {
                    ev[k] -= dp[k] / eps_r / (1.0 + alpha(xs[k]));
                }
            }
            ev[nn - 1] = 0.0;
            ev[0] = src((n as f64 + 1.0) * dt);
        }
        Aux1d { dir, g0, g1, nsteps, e_samp, h_samp }
    }

    /// e(t_n) at global node index i.
    fn e_at_node(&self, i: usize) -> Vec<f32> {
        let k = if self.dir > 0.0 { i - self.g0 } else { self.g1 - i };
        self.e_samp.get(k).cloned().unwrap_or_else(|| vec![0.0; self.nsteps])
    }

    /// h(t_{n+1/2}) at global half index j (between nodes j and j+1),
    /// returned as the physical H~ component along `a x e_pol` (sign of
    /// the propagation direction included).
    fn h_at_half(&self, j: usize) -> Vec<f32> {
        let k = if self.dir > 0.0 { j - self.g0 } else { self.g1 - j - 1 };
        self.h_samp.get(k).cloned().unwrap_or_else(|| vec![0.0; self.nsteps])
    }
}

/// Component index box for positions inside [min, max].
fn comp_box(grid: &Grid, c: Component, min: &[f64; 3], max: &[f64; 3], x0: usize, nxl: usize) -> CompBox {
    let mut glo = [0usize; 3];
    let mut ghi = [0usize; 3];
    for a in 0..3 {
        let (lo, hi) = grid.index_range(c, a, min[a], max[a]);
        glo[a] = lo;
        ghi[a] = hi;
    }
    localize(c, glo, ghi, x0, nxl)
}

fn localize(c: Component, glo: [usize; 3], ghi: [usize; 3], x0: usize, nxl: usize) -> CompBox {
    let lx0 = glo[0].max(x0);
    let lx1 = ghi[0].min(x0 + nxl);
    let (lstart, lsize, gx_off) = if lx0 < lx1 {
        ([lx0 - x0, glo[1], glo[2]], [lx1 - lx0, ghi[1] - glo[1], ghi[2] - glo[2]], lx0 - glo[0])
    } else {
        ([0, glo[1], glo[2]], [0, ghi[1] - glo[1], ghi[2] - glo[2]], 0)
    };
    CompBox { comp: c, glo, ghi, lstart, lsize, gx_off }
}

fn phasor_tables(freqs: &[f64], nsteps: usize, dt: f64, apod: &Apodization, stride: usize) -> (Vec<f32>, Vec<f32>) {
    let nf = freqs.len();
    let mut pe = vec![0f32; nsteps * nf * 2];
    let mut ph = vec![0f32; nsteps * nf * 2];
    for n in 0..nsteps {
        if n % stride.max(1) != 0 {
            continue;
        }
        let w_dt = dt * stride.max(1) as f64;
        let te = (n as f64 + 1.0) * dt;
        let th = (n as f64 + 0.5) * dt;
        let we = apod.weight(te) * w_dt;
        let wh = apod.weight(th) * w_dt;
        for (m, f) in freqs.iter().enumerate() {
            let w = TAU * f;
            let o = (n * nf + m) * 2;
            pe[o] = ((w * te).cos() * we) as f32;
            pe[o + 1] = ((w * te).sin() * we) as f32;
            ph[o] = ((w * th).cos() * wh) as f32;
            ph[o + 1] = ((w * th).sin() * wh) as f32;
        }
    }
    (pe, ph)
}

/// Closed box snapped to grid nodes (faces of flux/far-field boxes sit on
/// nodes; snapping the transverse extents too keeps the box closed).
fn snapped_box(grid: &Grid, min: &[f64; 3], max: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    // no clamping: a face at node 0 is either on a PEC/PMC wall (skipped) or
    // moved to node 1 where the face plane is built
    for a in 0..3 {
        lo[a] = grid.node(a, grid.nearest_node(a, min[a]));
        hi[a] = grid.node(a, grid.nearest_node(a, max[a]).max(1));
    }
    (lo, hi)
}

/// Boxes for a flux plane normal to `a` at node `p` with transverse extent:
/// [E_b(p), E_c(p), H_b(p-1), H_c(p-1), H_b(p), H_c(p)].
fn flux_boxes(grid: &Grid, a: usize, p: usize, min: &[f64; 3], max: &[f64; 3], x0: usize, nxl: usize) -> Vec<CompBox> {
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let mut out = vec![];
    let mk = |c: Component, pa: usize| -> CompBox {
        let mut glo = [0usize; 3];
        let mut ghi = [0usize; 3];
        for ax in 0..3 {
            if ax == a {
                glo[ax] = pa;
                ghi[ax] = pa + 1;
            } else {
                let (lo, hi) = grid.index_range(c, ax, min[ax], max[ax]);
                glo[ax] = lo;
                ghi[ax] = hi;
            }
        }
        localize(c, glo, ghi, x0, nxl)
    };
    out.push(mk(Component::e(bax), p));
    out.push(mk(Component::e(cax), p));
    let pm = p.saturating_sub(1);
    out.push(mk(Component::h(bax), pm));
    out.push(mk(Component::h(cax), pm));
    out.push(mk(Component::h(bax), p.min(grid.n()[a] - 1)));
    out.push(mk(Component::h(cax), p.min(grid.n()[a] - 1)));
    out
}

/// Stride of the Fourier sums of a frequency-domain monitor that is safe
/// for the given sources: sampling every `s` steps folds the spectrum at
/// `1 / (s dt)`, so a component of the fields at frequency `f'` lands on the
/// monitor frequency `f` if `f' = 1/(s dt) - f`. The stride keeps
/// `1 / (s dt) >= f_monitor,max + f_source,max`, with `f_source,max` seven
/// widths above the centre of a Gaussian pulse (spectral amplitude 2e-11 of
/// the peak). Sources of other kinds (continuous wave, Ricker, sampled) have
/// no such edge, and chi2/chi3 media create harmonics: the stride is then 1.
pub fn auto_dft_stride(sources: &[Source], nonlinear: bool, dt: f64, freqs: &[f64]) -> usize {
    if nonlinear || sources.is_empty() || freqs.is_empty() {
        return 1;
    }
    let mut f_src = 0.0f64;
    for s in sources {
        match s.signal() {
            TimeSignal::GaussianPulse { freq0, fwidth, .. } => f_src = f_src.max(freq0.abs() + 7.0 * fwidth.abs()),
            _ => return 1,
        }
    }
    let f_mon = freqs.iter().cloned().fold(0.0, f64::max);
    let s = (1.0 / (dt * (f_mon + f_src))).floor();
    if s.is_finite() { (s as usize).clamp(1, 1000) } else { 1 }
}

/// Mark boxes that an earlier monitor already accumulates (same component
/// box, frequencies, time window and stride) as shared: see [`DftPlan::shared`].
fn share_dft_boxes(plan: &mut Plan) {
    use std::collections::HashMap;
    // key: (comp, glo, ghi, group of identical accumulation settings)
    let mut groups: Vec<(Vec<u64>, Apodization, usize)> = vec![];
    let mut seen: HashMap<(Component, [usize; 3], [usize; 3], usize), (usize, usize)> = HashMap::new();
    for di in 0..plan.dft.len() {
        let key = (plan.dft[di].freqs.iter().map(|f| f.to_bits()).collect::<Vec<u64>>(), plan.dft[di].apodization.clone(), plan.dft[di].stride);
        let gi = match groups.iter().position(|g| *g == key) {
            Some(g) => g,
            None => {
                groups.push(key);
                groups.len() - 1
            }
        };
        let mut all_shared = !plan.dft[di].boxes.is_empty();
        for bi in 0..plan.dft[di].boxes.len() {
            let b = &plan.dft[di].boxes[bi];
            match seen.get(&(b.comp, b.glo, b.ghi, gi)) {
                Some(&src) => {
                    plan.dft[di].shared[bi] = Some(src);
                    plan.dft[di].boxes[bi].lsize[0] = 0; // not local: nothing to allocate or accumulate
                }
                None => {
                    seen.insert((b.comp, b.glo, b.ghi, gi), (di, bi));
                    all_shared = false;
                }
            }
        }
        if all_shared {
            // no box of its own: the phasor tables are not needed either
            plan.dft[di].phasor_e = vec![];
            plan.dft[di].phasor_h = vec![];
        }
    }
}

fn compile_monitor(
    plan: &mut Plan,
    mon: &Monitor,
    sim_boundaries: &crate::spec::Boundaries,
    eps_local: &[Vec<f32>; 3],
    mu_local: Option<&[Vec<f32>; 3]>,
    stride_of: &dyn Fn(usize, &[f64]) -> usize,
    inv: Option<usize>,
) -> Result<()> {
    let grid = plan.grid.clone();
    let (x0, nxl) = (plan.x0, plan.nxl);
    let nsteps = plan.nsteps;
    let dt = plan.dt;
    match mon {
        Monitor::FieldTime { name, min, max, components, interval, start } => {
            let boxes = components.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            plan.time.push(TimePlan {
                name: name.clone(),
                boxes,
                interval: (*interval).max(1),
                start_step: (start / dt).floor().max(0.0) as usize,
                is_probe: false,
            });
        }
        Monitor::Probe { name, position, components } => {
            let boxes = components
                .iter()
                .map(|c| {
                    let g = [
                        grid.nearest_index(*c, 0, position[0]),
                        grid.nearest_index(*c, 1, position[1]),
                        grid.nearest_index(*c, 2, position[2]),
                    ];
                    localize(*c, g, [g[0] + 1, g[1] + 1, g[2] + 1], x0, nxl)
                })
                .collect();
            plan.time.push(TimePlan { name: name.clone(), boxes, interval: 1, start_step: 0, is_probe: true });
        }
        Monitor::DftField { name, min, max, components, freqs, apodization, stride } => {
            let boxes: Vec<CompBox> = components.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            let stride = stride_of(*stride, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Field,
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::Flux { name, axis, position, min, max, freqs, apodization } => {
            let p = grid.nearest_node(*axis, *position).max(1);
            let boxes = flux_boxes(&grid, *axis, p, min, max, x0, nxl);
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Flux { axis: *axis, p },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::FluxBox { name, min, max, freqs, apodization } => {
            let (smin, smax) = snapped_box(&grid, min, max);
            let mut boxes = vec![];
            let mut faces = vec![];
            for a in 0..3 {
                // 2-D: the faces normal to the invariant axis carry equal and
                // opposite fluxes (nothing varies along it)
                if inv == Some(a) {
                    continue;
                }
                for (side, coord) in [(-1.0, smin[a]), (1.0, smax[a])] {
                    let pn = grid.nearest_node(a, coord);
                    // a face on a PEC/PMC wall carries no power (symmetry planes)
                    let wall = |b: &Boundary| matches!(b, Boundary::Pec | Boundary::Pmc);
                    if (pn == 0 && wall(&sim_boundaries.axis(a)[0])) || (pn == grid.n()[a] && wall(&sim_boundaries.axis(a)[1])) {
                        continue;
                    }
                    let p = pn.max(1);
                    faces.push((a, side, boxes.len()));
                    boxes.extend(flux_boxes(&grid, a, p, &smin, &smax, x0, nxl));
                }
            }
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::FluxBox,
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces,
                region: (smin, smax),
            });
        }
        Monitor::Mode { name, axis, position, min, max, freqs, modes, apodization } => {
            let p = grid.nearest_node(*axis, *position).max(1);
            let boxes = flux_boxes(&grid, *axis, p, min, max, x0, nxl);
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            let (origin, _) = transverse_box(&grid, *axis, Some(min), Some(max));
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Mode { axis: *axis, p, origin, modes: modes.clone() },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::FarField { name, min, max, freqs, theta, phi, n_medium, apodization } => {
            let (smin, smax) = snapped_box(&grid, min, max);
            let mut boxes = vec![];
            let mut faces = vec![];
            for a in 0..3 {
                for (side, coord) in [(-1.0, smin[a]), (1.0, smax[a])] {
                    let p = grid.nearest_node(a, coord).max(1);
                    faces.push((a, side, boxes.len()));
                    boxes.extend(flux_boxes(&grid, a, p, &smin, &smax, x0, nxl));
                }
            }
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::FarField { theta: theta.clone(), phi: phi.clone(), n_medium: *n_medium },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces,
                region: (smin, smax),
            });
        }
        Monitor::Energy { name, min, max, interval } => {
            let boxes: Vec<CompBox> = Component::ALL.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            // weight of every local sample: eps0/2 * (eps | mu) * the part of
            // the cell's volume that lies inside the monitor box
            let (ny, nz) = (plan.n[1], plan.n[2]);
            let n = plan.n;
            let periodic = plan.periodic;
            let width = |c: Component, ax: usize, g: usize| -> f64 {
                let (lo, hi) = grid.voxel_bounds(c, ax, g);
                let mut w = (hi.min(max[ax]) - lo.max(min[ax])).max(0.0);
                if periodic[ax] && !c.is_half(ax) && g == 0 {
                    // the dual cell of the first node wraps around a periodic axis
                    let e = grid.edges(ax);
                    let (lo2, hi2) = (0.5 * (e[n[ax] - 1] + e[n[ax]]), e[n[ax]]);
                    w += (hi2.min(max[ax]) - lo2.max(min[ax])).max(0.0);
                }
                w
            };
            let weights: Vec<Vec<f32>> = boxes
                .iter()
                .map(|bx| {
                    let c = bx.comp;
                    let a = c.axis();
                    let mut w = Vec::with_capacity(bx.lcells());
                    for ii in 0..bx.lsize[0] {
                        let li = bx.lstart[0] + ii;
                        let wx = width(c, 0, x0 + li);
                        for jj in 0..bx.lsize[1] {
                            let j = bx.lstart[1] + jj;
                            let wy = width(c, 1, j);
                            for kk in 0..bx.lsize[2] {
                                let k = bx.lstart[2] + kk;
                                let o = (li * ny + j) * nz + k;
                                let m = if c.is_e() { eps_local[a][o] as f64 } else { mu_local.map(|m| m[a][o] as f64).unwrap_or(1.0) };
                                w.push((0.5 * EPS0 * m * wx * wy * width(c, 2, k)) as f32);
                            }
                        }
                    }
                    w
                })
                .collect();
            plan.energy.push(EnergyPlan { name: name.clone(), time: plan.time.len(), weights });
            plan.time.push(TimePlan {
                name: format!("__energy__{name}"),
                boxes,
                interval: (*interval).max(1),
                start_step: 0,
                is_probe: false,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_dft_stride() {
        use crate::grid::Component;
        let dipole = |signal: TimeSignal| Source::PointDipole { name: String::new(), position: [0.0; 3], polarization: Component::Ez, signal, amplitude: 1.0 };
        let f0 = 2e14;
        let dt = 4e-17;
        let pulse = dipole(TimeSignal::gaussian(f0, 0.1 * f0));
        // 1 / (s dt) >= f_monitor + f0 + 7 fwidth
        let s = auto_dft_stride(&[pulse.clone()], false, dt, &[0.9 * f0, 1.1 * f0]);
        assert_eq!(s, (1.0 / (dt * (1.1 * f0 + 1.7 * f0))).floor() as usize);
        assert!(s >= 20 && 1.0 / (s as f64 * dt) >= 2.8 * f0);
        // a broader pulse or a higher monitor frequency: a smaller stride
        assert!(auto_dft_stride(&[dipole(TimeSignal::gaussian(f0, 0.5 * f0))], false, dt, &[f0]) < s);
        assert!(auto_dft_stride(&[pulse.clone()], false, dt, &[3.0 * f0]) < s);
        // no known band edge: every step
        assert_eq!(auto_dft_stride(&[pulse.clone()], true, dt, &[f0]), 1, "nonlinear media make harmonics");
        assert_eq!(auto_dft_stride(&[pulse.clone(), dipole(TimeSignal::cw(f0, 0.1 * f0))], false, dt, &[f0]), 1);
        assert_eq!(auto_dft_stride(&[dipole(TimeSignal::Ricker { freq0: f0, offset: 1.5, amplitude: 1.0 })], false, dt, &[f0]), 1);
        assert_eq!(auto_dft_stride(&[], false, dt, &[f0]), 1);
        // a coarse time step: never below 1
        assert_eq!(auto_dft_stride(&[pulse], false, 1.0 / (2.0 * f0), &[f0]), 1);
    }

    #[test]
    fn monitors_on_one_plane_share_their_boxes() {
        use crate::grid::MeshSpec;
        use crate::monitors::Monitor;
        use crate::sources::ModeProfile;
        use crate::spec::Boundaries;
        let mut sim = Simulation::new([0.0; 3], [2e-6, 1e-6, 1e-6], MeshSpec::Uniform { dx: [0.1e-6; 3] }, Boundaries::all_pml(), 2e-14);
        let profile = ModeProfile { neff: 1.0, neff_imag: 0.0, dims: [1, 1], e_re: [vec![1.0], vec![0.0]], e_im: [vec![0.0], vec![0.0]], h_re: [vec![0.0], vec![1.0]], h_im: [vec![0.0], vec![0.0]], freq: 3e14, h_offset: None };
        let mode = |name: &str, x: f64, freqs: Vec<f64>| Monitor::Mode { name: name.into(), axis: 0, position: x, min: [x, 0.2e-6, 0.2e-6], max: [x, 0.8e-6, 0.8e-6], freqs, modes: vec![profile.clone()], apodization: Default::default() };
        for i in 0..40 {
            sim.monitors.push(mode(&format!("m{i}"), 1.0e-6, vec![3e14]));
        }
        sim.monitors.push(mode("other_plane", 1.5e-6, vec![3e14]));
        sim.monitors.push(mode("other_freqs", 1.0e-6, vec![3e14, 3.1e14]));
        sim.monitors.push(Monitor::Flux { name: "flux".into(), axis: 0, position: 1.0e-6, min: [1.0e-6, 0.2e-6, 0.2e-6], max: [1.0e-6, 0.8e-6, 0.8e-6], freqs: vec![3e14], apodization: Default::default() });
        let plan = compile(&sim, &CompileOptions::default()).unwrap();
        let own = |d: &DftPlan| d.boxes.iter().filter(|b| b.is_local()).count();
        assert_eq!(own(&plan.dft[0]), 6);
        for d in &plan.dft[1..40] {
            assert_eq!(own(d), 0, "{}", d.name);
            assert!(d.shared.iter().enumerate().all(|(bi, s)| *s == Some((0, bi))));
            assert!(d.phasor_e.is_empty() && d.phasor_h.is_empty());
        }
        assert_eq!(own(&plan.dft[40]), 6, "another plane records its own fields");
        assert_eq!(own(&plan.dft[41]), 6, "another frequency list records its own fields");
        assert_eq!(own(&plan.dft[42]), 0, "a flux monitor on the plane of a mode monitor shares its fields");
        let total: usize = plan.dft.iter().map(own).sum();
        assert_eq!(total, 18);
    }

    #[test]
    fn decomposition_covers_domain() {
        let d = decompose_x(100, &[1.0, 1.0, 2.0]);
        assert_eq!(d, vec![(0, 25), (25, 25), (50, 50)]);
    }
}
