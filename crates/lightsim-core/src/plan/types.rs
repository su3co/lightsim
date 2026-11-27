//! The data types of a plan: the [`Plan`] itself and the tables it holds
//! for the backends (CPML and absorber slabs, injections, dispersive,
//! nonlinear and tensor lists, DFT, time and energy monitors, the 2-D update).

use super::*;

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
    Mirror { factor: Real },
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
    pub e_b: Vec<Real>,
    pub e_a: Vec<Real>,
    pub e_kinv: Vec<Real>,
    pub h_b: Vec<Real>,
    pub h_a: Vec<Real>,
    pub h_kinv: Vec<Real>,
}

impl PmlSlab {
    pub fn cells(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }
}

/// The cells of one adiabatic absorber (`Boundary::Absorber`) normal to
/// `axis`, restricted to the local x range. The absorber itself lives in the
/// update coefficients (`ca`, `cb`, `ch`, `hloss`); this box is kept for the
/// checks that need to know where it is.
#[derive(Clone, Debug)]
pub struct AbsorbSlab {
    pub axis: usize,
    pub start: [usize; 3],
    pub size: [usize; 3],
}

/// A list of field updates `field[idx] += coef * amp(step, group)`.
#[derive(Clone, Debug)]
pub struct Injection {
    pub comp: Component,
    pub idx: Vec<u64>,
    pub coef: Vec<Real>,
    /// Imaginary coefficients for complex injections (`amp` holds (re, im)).
    pub coef_im: Option<Vec<Real>>,
    pub group: Vec<u32>,
    pub ngroups: usize,
    /// Amplitude table `[step][group]` (real) or `[step][group][2]` (complex).
    pub amps: Vec<Real>,
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
    /// Weight of the cell's polarisation in its E update (`E -= scale dP`):
    /// 1 / (eps_inf (1 + f)) of its medium, or w_A / eps_inf of a mixed voxel
    /// ([`crate::raster::MixedCell`]).
    pub scale: Vec<Real>,
    /// Positions (in `idx`) of the mixed voxels, whose `scale` differs from
    /// their medium's; empty without them. Layouts that store one scale per
    /// material (the dense GPU layout) treat these cells separately.
    pub mixed: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct DispPlan {
    pub npole: usize,
    /// `[material][pole][3]` ADE coefficients.
    pub coefs: Vec<Real>,
    pub comps: Vec<DispComponent>,
}

/// Cells of one E component inside chi2/chi3 media.
#[derive(Clone, Debug)]
pub struct NonlinComponent {
    pub comp: Component,
    /// Storage indices.
    pub idx: Vec<u64>,
    /// chi2 / eps and chi3 / eps of the cell (eps = linear relative permittivity).
    pub c2: Vec<Real>,
    pub c3: Vec<Real>,
}

/// Nonlinear media. `comps`: instantaneous nonlinear polarization
/// P_i/eps0 = chi2 E_i^2 + chi3 E_i^3, applied as an implicit per-cell
/// correction after the linear E update. `raman`: chi3 media with a delayed
/// (Raman) part, which have their own lists and state (the gradient engines
/// refuse such media, see `autodiff::session::check_media`).
#[derive(Clone, Debug)]
pub struct NonlinPlan {
    pub comps: Vec<NonlinComponent>,
    pub raman: Option<RamanPlan>,
    /// Four-level gain media (populations coupled to a Lorentz polarization).
    pub gain: Option<GainPlan>,
}

/// Cells of one E component inside gain media.
#[derive(Clone, Debug)]
pub struct GainComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    /// Row of the cell's medium in [`GainPlan::tab`].
    pub row: Vec<u32>,
    /// 1 / (eps (1 + f)) of the cell (f: conduction term).
    pub scale: Vec<Real>,
    /// Volume of the cell's E sample (m^3), for the population totals.
    pub vol: Vec<f64>,
}

/// Four-level gain media ([`crate::materials::FourLevel`]). Before the E
/// update: P^{n+1} = c0 P^n + c1 P^{n-1} + c2 (N1 - N2) E^n (P / eps0).
/// After it: E^{n+1} -= scale (P^{n+1} - P^n); the field does the work
/// w = eps0 (E^n + E^{n+1}) / 2 (P^{n+1} - P^n) per volume, which moves
/// w / (hbar omega) atoms per volume from level 1 to level 2; the decays
/// and the pump act through explicit fluxes from the populations at n.
#[derive(Clone, Debug)]
pub struct GainPlan {
    /// Per medium row: c0, c1, c2, eps0 / (hbar omega), dt / tau32,
    /// dt / tau21, dt / tau10, dt W_p.
    pub tab: Vec<f64>,
    /// Per medium row: initial densities N0..N3 (m^-3).
    pub init: Vec<f64>,
    pub comps: Vec<GainComponent>,
}

/// Width of a row of [`GainPlan::tab`].
pub const GAIN_TAB: usize = 8;

/// Cells of one E component inside chi3 media with a Raman (delayed) part.
#[derive(Clone, Debug)]
pub struct RamanComponent {
    pub comp: Component,
    /// Storage indices.
    pub idx: Vec<u64>,
    /// Row of the cell's medium in [`RamanPlan::coefs`].
    pub matid: Vec<u32>,
    /// chi2 / eps, (1 - f) chi3 / eps and f chi3 / eps of the cell.
    pub c2: Vec<Real>,
    pub c3: Vec<Real>,
    pub cr: Vec<Real>,
}

/// chi3 with a Raman response (see [`crate::materials::Raman`]): before the
/// E update, Q^{n+1} = c0 Q^n + c1 Q^{n-1} + c2 (E^n)^2; after the linear
/// update, an implicit per-cell solve for E^{n+1} of
/// eps E^{n+1} + P^{n+1}/eps0 = eps E_lin + P^n/eps0 with
/// P/eps0 = chi2 E^2 + chi3 E [(1 - f) E^2 + f Q].
#[derive(Clone, Debug)]
pub struct RamanPlan {
    /// `[medium row][c0, c1, c2]` of the Q update.
    pub coefs: Vec<Real>,
    pub comps: Vec<RamanComponent>,
}

/// Cells of one E component inside media with off-diagonal permittivity.
#[derive(Clone, Debug)]
pub struct TensorComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    /// K_{a,a+1} / (c dt) and K_{a,a+2} / (c dt) (lossless: kappa = eps^-1).
    pub c1: Vec<Real>,
    pub c2: Vec<Real>,
    /// M_{a,a+1}, M_{a,a+2} (zero for lossless media).
    pub m1: Vec<Real>,
    pub m2: Vec<Real>,
    /// Oriented poles: table row of the cell's medium (u32::MAX = none) and
    /// 1/eps_inf along the component.
    pub opm: Vec<u32>,
    pub oscale: Vec<Real>,
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
    pub cdt: Real,
    /// E^n of the neighbours is needed (lossy tensors or oriented poles).
    pub lossy: bool,
    /// Oriented poles: poles per medium row and the table
    /// `[row][pole][c0, c1, c2, u_x, u_y, u_z]` (rows padded with zero poles).
    pub op_np: usize,
    pub op_tab: Vec<Real>,
    pub comps: Vec<TensorComponent>,
    /// Gyrotropic poles: poles per medium row and the table
    /// `[row][pole][A (9), B (9), C (9)]` of the 3x3 update matrices (rows
    /// padded with zero poles), and the cells per E component.
    pub gyro_np: usize,
    pub gyro_tab: Vec<Real>,
    pub gyro: Vec<GyroComponent>,
}

/// Cells of one E component inside gyrotropic media. Each cell carries a
/// full polarization vector per pole, driven by E^n at the cell (its own
/// component and the four-point averages of the other two, as for oriented
/// poles); its own component feeds back: E_a -= dP_a / eps_inf.
#[derive(Clone, Debug)]
pub struct GyroComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    /// Row of the cell's medium in [`TensorPlan::gyro_tab`].
    pub row: Vec<u32>,
    /// 1 / eps_inf along the component.
    pub scale: Vec<Real>,
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
    pub phasor_e: Vec<Real>,
    pub phasor_h: Vec<Real>,
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
    pub weights: Vec<Vec<Real>>,
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
    pub fn plane(&self) -> (usize, usize, Real) {
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
    pub sg: Real,
    /// The E update of row 0 is left to the host (periodic or neighbour low
    /// ghost along x).
    pub deferred: bool,
}

impl Step2d {
    /// Kernel `pol` argument for the sets in use.
    pub fn pol(td: &TwoD) -> u32 {
        td.active[0] as u32 | (td.active[1] as u32) << 1
    }

    /// Tables that let the 2-D step kernel add the terms of injection lists
    /// to the cells it updates (its `irow`, `icell`, `ioff` and `ient`
    /// arguments). `lists` are (target component, storage indices,
    /// coefficients) in the order the lists are applied after the step; a
    /// cell receives the terms of all its entries in that order, so the
    /// fields are those of the separate launches. Entries of a field set
    /// that `pol` does not update are left out when their coefficient is
    /// zero (they add nothing: the E corrections of a magnetic source along
    /// the invariant axis, whose two halves cancel). `None` when another
    /// entry is not a cell the kernel writes: outside the plane, row 0 left
    /// to the host (`deferred`), or a nonzero term on a set not updated.
    pub fn inj_table(&self, pol: u32, lists: &[(Component, &[u64], &[Real])]) -> Option<Inj2d> {
        let cells = self.nu as u64 * self.nv as u64;
        let mut ent: Vec<(u32, u32, u32)> = vec![];
        for (li, (c, idx, coef)) in lists.iter().enumerate() {
            let a = c.axis();
            let slot = if a == self.u { 0 } else if a == self.v { 1 } else { 2 } + if c.is_e() { 3 } else { 0 };
            // TE = (E_u, E_v, H_w), TM = (H_u, H_v, E_w)
            let te = (2..=4).contains(&slot);
            let bit = if te { 1 } else { 2 };
            if li >= 1 << 24 || coef.len() != idx.len() {
                return None;
            }
            if (pol & bit) == 0 {
                if coef.iter().any(|&x| x != 0.0) {
                    return None;
                }
                continue;
            }
            for (e, &s) in idx.iter().enumerate() {
                let t = s.checked_sub(self.base as u64).filter(|&t| t < cells)?;
                if self.deferred && t < self.nv as u64 {
                    return None;
                }
                ent.push((t as u32, (li as u32) | (slot as u32) << 24, e as u32));
            }
        }
        // stable: entries of one cell keep the order of their lists
        ent.sort_by_key(|x| x.0);
        let mut out = Inj2d { rows: vec![0; self.nu as usize + 1], cells: vec![], off: vec![0], ent: Vec::with_capacity(2 * ent.len()) };
        for &(t, w, e) in &ent {
            if out.cells.last() != Some(&t) {
                if !out.cells.is_empty() {
                    out.off.push((out.ent.len() / 2) as u32);
                }
                out.cells.push(t);
                out.rows[(t / self.nv) as usize + 1] += 1;
            }
            out.ent.extend_from_slice(&[w, e]);
        }
        out.off.push((out.ent.len() / 2) as u32);
        for r in 1..out.rows.len() {
            out.rows[r] += out.rows[r - 1];
        }
        Some(out)
    }
}

/// Source tables of the 2-D step kernel: see [`Step2d::inj_table`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inj2d {
    /// Per row u, the range of `cells` in that row (`nu + 1` offsets).
    pub rows: Vec<u32>,
    /// Ascending in-plane indices t = u * nv + v of the cells with terms.
    pub cells: Vec<u32>,
    /// Per cell, the range of its entries (`cells.len() + 1` offsets).
    pub off: Vec<u32>,
    /// Two words per entry: list | slot << 24 (slot 0..2 = H (u, v, w),
    /// 3..5 = E (u, v, w)) and the entry's index in its list.
    pub ent: Vec<u32>,
}

/// Magnetically lossy cells of one H component (see [`Plan::hloss`]).
#[derive(Clone, Debug, Default)]
pub struct HLoss {
    pub idx: Vec<u64>,
    pub da: Vec<Real>,
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
    pub inv_p: [Vec<Real>; 3],
    pub inv_d: [Vec<Real>; 3],
    pub cb: [Vec<Real>; 3],
    pub ca: Option<[Vec<Real>; 3]>,
    pub ch: Option<[Vec<Real>; 3]>,
    pub ch_scalar: Real,
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
    /// Adiabatic absorbers (their boxes; see [`AbsorbSlab`]).
    pub absorb: Vec<AbsorbSlab>,
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
    /// Zero-thickness sheets ([`crate::sheet`]).
    pub sheet: Option<crate::sheet::SheetPlan>,
    /// User-defined material plug-ins ([`crate::plugin`]).
    pub plugin: Option<crate::plugin::PluginPlan>,
    /// Four-level two-electron gain media ([`crate::gain2`]).
    pub gain2: Option<crate::gain2::Gain2Plan>,
    /// Complex (Hermitian) permittivity tensors ([`crate::herm`]); the plan
    /// then runs with complex fields (`bloch`).
    pub herm: Option<crate::herm::HermPlan>,
    /// Sub-pixel smoothing of transparent dispersive dielectrics (see
    /// [`crate::raster::MixedCell`]): E samples of this slab that got a mixed
    /// voxel, and those that a guard sent back to the staircase.
    pub disp_subpixel: [usize; 2],
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
        let r = std::mem::size_of::<Real>();
        let s = self.storage_len() * r;
        let mut b = 6 * s + 3 * s;
        if self.ca.is_some() {
            b += 3 * s;
        }
        if self.ch.is_some() {
            b += 3 * s;
        }
        for p in &self.pml {
            b += 4 * p.cells() * r;
        }
        for d in &self.dft {
            for bx in &d.boxes {
                b += 2 * bx.lcells() * d.freqs.len() * r;
            }
        }
        b
    }
}
