//! Zero-thickness conducting sheets (graphene, 2-D conductors, fitted
//! surface conductivities).
//!
//! A [`Sheet`] is a plane normal to an axis, or a finite rectangle or
//! polygon on such a plane, that carries the surface current
//! `J_s = sigma_s * E_t` (tangential E only) with
//!
//! ```text
//! sigma_s(w) = sigma_0 - i w eps0 chi_0
//!            + sum Drude   i D / (w + i gamma)
//!            + sum Lorentz -i w s / (w0^2 - w^2 - i gamma w)
//! ```
//!
//! in siemens (fields `exp(-i w t)`). The surface polarization
//! `P_s = int J_s dt` of each pole obeys the volume ADE of a pole of the
//! same form (`P_s'' + gamma P_s' + w0^2 P_s = s E`, Drude: `w0 = 0`,
//! `s = D`), so the discrete update is the one of the dispersive volume
//! media with `P / eps0` in metres times V/m.
//!
//! **Discretisation.** The current enters Ampere's law on the planes of the
//! tangential E nodes, divided by the dual cell size `d` normal to the sheet
//! there (the plan computes it from the mesh, uniform or not): a node
//! carrying the sheet updates `E^{n+1} = E_lin - (P^{n+1} - P^n) / (eps d)`
//! (eps: the node's permittivity, as the linear update has it). Generated
//! meshes (uniform and automatic) put a node on every sheet plane
//! ([`align_nodes`]), so the user aligns nothing and the scheme is of
//! second order in the mesh step. On an explicit mesh (which is not moved)
//! a sheet between nodes is shared between the two tangential-E planes
//! around it by linear weights, the plane at distance `w * delta` from it
//! carrying the fraction `1 - w` of the conductance (`delta`: the distance
//! between the planes). That keeps the first moment of the current at the
//! sheet, but the two half-sheets interact across `delta`, which adds an
//! error of relative size `w (1 - w) k delta |eta sigma_s|` (`eta`: the wave
//! impedance) to the effective conductance: first order in `delta`. Edges of a finite patch are area-weighted: a node whose in-plane
//! cell the patch covers by a fraction `c` carries `c` times the current.
//!
//! The constant conductance `sigma_0` and susceptibility `chi_0` are applied
//! semi-implicitly (centred in time), the poles explicitly from `E^n` as the
//! volume ADE: `P^{n+1} = c0 P^n + c1 P^{n-1} + c2 E^n`. With
//! `k = w c / (eps (1 + f) d)` per node (`f`: the node's own conduction
//! term), `h = sigma_0 dt / (2 eps0)`:
//!
//! ```text
//! E^{n+1} (1 + k (h + chi_0)) = E_lin - k sum_q (P_q^{n+1} - P_q^n) - k (h - chi_0) E^n.
//! ```

use crate::Real;
use crate::constants::{C0, EPS0, ETA0};
use crate::grid::{Component, Grid};
use crate::materials::Pole;
use anyhow::{Result, bail};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A dispersive term of a surface conductivity (siemens).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SheetPole {
    /// `sigma(w) = i D / (w + i gamma)`: `drude_weight` D in S/s, `gamma`
    /// in rad/s (graphene's intraband term, a 2-D electron gas).
    Drude { drude_weight: f64, gamma: f64 },
    /// `sigma(w) = -i w s / (w0^2 - w^2 - i gamma w)`: `strength` s in
    /// S rad/s (the peak conductance s / gamma is reached at w0), `omega0`
    /// and `gamma` in rad/s.
    Lorentz { strength: f64, omega0: f64, gamma: f64 },
}

impl SheetPole {
    pub fn sigma(&self, omega: f64) -> Complex64 {
        let i = Complex64::i();
        match *self {
            SheetPole::Drude { drude_weight, gamma } => i * drude_weight / (omega + i * gamma),
            SheetPole::Lorentz { strength, omega0, gamma } => -i * omega * strength / (omega0 * omega0 - omega * omega - i * gamma * omega),
        }
    }
    /// The pole of the surface susceptibility `chi_s = i sigma / (eps0 w)`
    /// (in metres) as a volume [`Pole`]: the same ADE.
    pub fn surface_pole(&self) -> Pole {
        match *self {
            SheetPole::Drude { drude_weight, gamma } => Pole::Drude { omega_p: (drude_weight / EPS0).sqrt(), gamma },
            SheetPole::Lorentz { strength, omega0, gamma } => {
                if omega0 == 0.0 {
                    Pole::Drude { omega_p: (strength / EPS0).sqrt(), gamma }
                } else {
                    Pole::Lorentz { delta_eps: strength / (EPS0 * omega0 * omega0), omega0, gamma }
                }
            }
        }
    }
    /// The volume pole of a layer of thickness `d` carrying this term.
    pub fn volume_pole(&self, d: f64) -> Pole {
        match self.surface_pole() {
            Pole::Drude { omega_p, gamma } => Pole::Drude { omega_p: omega_p / d.sqrt(), gamma },
            Pole::Lorentz { delta_eps, omega0, gamma } => Pole::Lorentz { delta_eps: delta_eps / d, omega0, gamma },
            p => p,
        }
    }
    fn check(&self) -> bool {
        match *self {
            SheetPole::Drude { drude_weight, gamma } => drude_weight >= 0.0 && gamma >= 0.0 && drude_weight.is_finite() && gamma.is_finite(),
            SheetPole::Lorentz { strength, omega0, gamma } => strength >= 0.0 && omega0 >= 0.0 && gamma >= 0.0 && (strength + omega0 + gamma).is_finite(),
        }
    }
}

/// Surface conductivity of a sheet (see the module documentation).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct SheetConductivity {
    #[serde(default)]
    pub name: String,
    /// Constant (frequency-independent) conductance sigma_0 (S).
    #[serde(default)]
    pub conductance: f64,
    /// Constant surface susceptibility chi_0 (m): a frequency-independent
    /// surface polarization `P_s = eps0 chi_0 E_t` (the background term of a
    /// fitted conductivity).
    #[serde(default)]
    pub susceptibility: f64,
    #[serde(default)]
    pub poles: Vec<SheetPole>,
}

impl SheetConductivity {
    /// sigma_s(w) of the continuous model (S).
    pub fn sigma(&self, omega: f64) -> Complex64 {
        let mut s = Complex64::new(self.conductance, -omega * EPS0 * self.susceptibility);
        for p in &self.poles {
            s += p.sigma(omega);
        }
        s
    }
    /// sigma_s(w) seen by the discrete update with time step `dt`, referred
    /// to the field at the half step: the leapfrog current
    /// `(P^{n+1} - P^n) / dt` against `(E^n + E^{n+1}) / 2`.
    pub fn sigma_discrete(&self, omega: f64, dt: f64) -> Complex64 {
        let z = Complex64::new(0.0, -omega * dt).exp();
        // (P^{n+1} - P^n) / dt = eps0 (z - 1) / dt chi E^n, E^{n+1/2} = (1 + z) / 2 E^n
        let mut chi = Complex64::new(self.susceptibility, 0.0);
        for p in &self.poles {
            chi += p.surface_pole().chi_discrete(omega, dt);
        }
        let cur = EPS0 * (z - 1.0) / dt * chi * 2.0 / (1.0 + z);
        cur + self.conductance
    }
    pub fn check(&self, sheet: &str) -> Result<()> {
        if !(self.conductance >= 0.0 && self.susceptibility >= 0.0 && self.conductance.is_finite() && self.susceptibility.is_finite()) || !self.poles.iter().all(|p| p.check()) {
            bail!("sheet '{sheet}': the conductance, the susceptibility and every pole parameter must be finite and >= 0 (a passive sheet)");
        }
        Ok(())
    }
}

/// A zero-thickness conducting sheet: the plane `x_axis = position`, or a
/// part of it (`min` / `max`: a rectangle, `polygon`: a polygon).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Sheet {
    #[serde(default)]
    pub name: String,
    /// Axis normal to the sheet: 0 / 1 / 2 or "x" / "y" / "z".
    #[serde(deserialize_with = "crate::spec::de_axis")]
    pub axis: usize,
    /// Coordinate of the sheet along `axis` (m). It need not lie on a grid
    /// node.
    pub position: f64,
    pub conductivity: SheetConductivity,
    /// Corners of a rectangular patch (the components along `axis` are
    /// ignored). Without them (and without `polygon`) the sheet fills the
    /// whole plane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<[f64; 3]>,
    /// Vertices of a polygonal patch in the plane, as (u, v) pairs, u and v
    /// being the two other axes in increasing order (y, z for a sheet
    /// normal to x; x, z for y; x, y for z).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<[f64; 2]>>,
}

/// The two in-plane axes of a sheet normal to `a`, in increasing order.
pub fn in_plane(a: usize) -> (usize, usize) {
    match a {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

fn point_in_polygon(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Sheet {
    pub fn check(&self, invariant: Option<usize>) -> Result<()> {
        if self.axis > 2 {
            bail!("sheet '{}': axis must be 0, 1 or 2", self.name);
        }
        if invariant == Some(self.axis) {
            bail!("sheet '{}' is normal to the invariant axis of this 2-D simulation: in 2-D a sheet is a line, normal to one of the two in-plane axes", self.name);
        }
        if self.min.is_some() != self.max.is_some() {
            bail!("sheet '{}': give both min and max (a rectangle), or neither", self.name);
        }
        if self.min.is_some() && self.polygon.is_some() {
            bail!("sheet '{}': give a rectangle (min, max) or a polygon, not both", self.name);
        }
        if let Some(p) = &self.polygon {
            if p.len() < 3 {
                bail!("sheet '{}': a polygon needs at least three vertices", self.name);
            }
        }
        self.conductivity.check(&self.name)
    }

    /// Fraction of the in-plane rectangle `[lo, hi]` (two in-plane axes
    /// `(u, v)`; `skip`: an in-plane axis not to test, the invariant axis of a
    /// 2-D simulation) that the sheet covers.
    pub fn coverage(&self, lo: [f64; 2], hi: [f64; 2], skip: Option<usize>) -> f64 {
        let (u, v) = in_plane(self.axis);
        let ax = [u, v];
        if let (Some(mn), Some(mx)) = (self.min, self.max) {
            let mut c = 1.0;
            for q in 0..2 {
                if Some(ax[q]) == skip || hi[q] <= lo[q] {
                    continue;
                }
                let (a, b) = (mn[ax[q]].min(mx[ax[q]]), mn[ax[q]].max(mx[ax[q]]));
                c *= ((hi[q].min(b) - lo[q].max(a)) / (hi[q] - lo[q])).clamp(0.0, 1.0);
            }
            return c;
        }
        if let Some(poly) = &self.polygon {
            // supersampled, 16 x 16 points (one point along a skipped axis)
            let m = 16;
            let (mu, mv) = (if Some(u) == skip { 1 } else { m }, if Some(v) == skip { 1 } else { m });
            let mut hit = 0usize;
            for a in 0..mu {
                for b in 0..mv {
                    let pu = if mu == 1 { poly.iter().map(|p| p[0]).sum::<f64>() / poly.len() as f64 } else { lo[0] + (a as f64 + 0.5) / mu as f64 * (hi[0] - lo[0]) };
                    let pv = if mv == 1 { poly.iter().map(|p| p[1]).sum::<f64>() / poly.len() as f64 } else { lo[1] + (b as f64 + 0.5) / mv as f64 * (hi[1] - lo[1]) };
                    hit += point_in_polygon([pu, pv], poly) as usize;
                }
            }
            return hit as f64 / (mu * mv) as f64;
        }
        1.0
    }
}

/// Cells of one tangential E component on one sheet (one or two node
/// planes, which never repeat a cell, so the list updates in parallel).
#[derive(Clone, Debug)]
pub struct SheetList {
    pub comp: Component,
    /// Index of the sheet (row of [`SheetPlan::coefs`]).
    pub row: u32,
    pub idx: Vec<u64>,
    /// k = w c / (eps (1 + f) d) per cell (1/m).
    pub k: Vec<Real>,
    /// Sheet area that the cell carries (w c times its in-plane cell area,
    /// m^2), for the absorbed power.
    pub area: Vec<f64>,
}

/// Compiled sheets of one slab (see the module documentation).
#[derive(Clone, Debug)]
pub struct SheetPlan {
    /// Poles per row (rows padded with inert poles).
    pub np: usize,
    /// `[row][pole][c0, c1, c2]`.
    pub coefs: Vec<Real>,
    /// `[row][h, chi_0]`: h = sigma_0 dt / (2 eps0) and chi_0 (m).
    pub cond: Vec<Real>,
    pub lists: Vec<SheetList>,
}

/// Width of a row of [`SheetPlan::cond`].
pub const SHEET_COND: usize = 2;

/// Node planes `(index, weight)` along `axis` that carry a sheet at `pos`:
/// one plane if the sheet is on a node (to 1e-6 of the cell), else the two
/// planes around it with linear weights.
pub fn node_weights(grid: &Grid, axis: usize, pos: f64, periodic: bool, name: &str) -> Result<Vec<(usize, f64)>> {
    let e = grid.edges(axis);
    let n = e.len() - 1;
    let (lo, hi) = (e[0], e[n]);
    if !(pos >= lo && pos <= hi) {
        bail!("sheet '{name}' at {pos:e} lies outside the grid along axis {axis} ([{lo:e}, {hi:e}])");
    }
    let g = grid.cell_index(axis, pos);
    let w = ((pos - e[g]) / (e[g + 1] - e[g])).clamp(0.0, 1.0);
    let wrap = |i: usize| if i >= n && periodic { i - n } else { i };
    let out: Vec<(usize, f64)> = if w < 1e-6 {
        vec![(g, 1.0)]
    } else if w > 1.0 - 1e-6 {
        vec![(wrap(g + 1), 1.0)]
    } else {
        vec![(g, 1.0 - w), (wrap(g + 1), w)]
    };
    if out.iter().any(|&(i, _)| i >= n) {
        bail!("sheet '{name}' at {pos:e} lies on the upper face of the grid along axis {axis}, where there is no tangential E node; move it inside");
    }
    Ok(out)
}

/// Cells over which [`align_nodes`] tapers the move of a node.
pub const ALIGN_TAPER: usize = 8;

/// Put a grid node on every sheet plane normal to `axis` (generated meshes:
/// uniform and automatic). The node nearest to the sheet moves onto it and
/// the move is tapered linearly over [`ALIGN_TAPER`] cells on each side, so
/// neighbouring cells change size by at most 1 / (2 ALIGN_TAPER) of a cell
/// (6.25 %). The end nodes do not move; a sheet within half a cell of an
/// end, or one whose move would disturb a node already put on another
/// sheet beyond the taper, is left between nodes (it is then shared
/// between the two planes around it, see the module documentation).
pub fn align_nodes(nodes: &mut [f64], sheets: &[Sheet], axis: usize) {
    let n = nodes.len() - 1;
    if n < 2 {
        return;
    }
    let mut frozen: Vec<usize> = vec![];
    for s in sheets.iter().filter(|s| s.axis == axis) {
        let pos = s.position;
        if !(pos > nodes[0] && pos < nodes[n]) {
            continue;
        }
        let i = (0..=n).min_by(|&a, &b| (nodes[a] - pos).abs().partial_cmp(&(nodes[b] - pos).abs()).unwrap()).unwrap();
        let d = pos - nodes[i];
        let cell = if i < n { nodes[i + 1] - nodes[i] } else { nodes[i] - nodes[i - 1] };
        if d.abs() <= 1e-9 * cell {
            frozen.push(i);
            continue;
        }
        if i == 0 || i == n || frozen.contains(&i) {
            continue;
        }
        // taper: at most ALIGN_TAPER cells, never reaching an end or a frozen node
        let mut m = ALIGN_TAPER.min(i).min(n - i);
        for &f in &frozen {
            let dist = if f > i { f - i } else { i - f };
            m = m.min(dist);
        }
        if m == 0 {
            continue;
        }
        let old: Vec<f64> = nodes.to_vec();
        for j in (i + 1 - m)..(i + m) {
            let k = if j > i { j - i } else { i - j };
            nodes[j] += d * (1.0 - k as f64 / m as f64);
        }
        if nodes.windows(2).any(|w| w[1] <= w[0]) {
            nodes.copy_from_slice(&old);
            continue;
        }
        nodes[i] = pos;
        frozen.push(i);
    }
}

/// Inputs of [`compile_sheets`] from the plan compiler.
pub struct SheetCtx<'a> {
    pub grid: &'a Grid,
    pub x0: usize,
    pub nxl: usize,
    pub dt: f64,
    pub periodic: [bool; 3],
    pub invariant: Option<usize>,
    /// Inverse dual spacings over the whole grid (periodic-aware).
    pub inv_d_full: &'a [Vec<Real>; 3],
    /// E-update coefficients cb of the slab (local layout, before the
    /// storage padding) per E component.
    pub cb_local: &'a [Vec<Real>; 3],
    /// Medium index of each local E sample.
    pub med_local: &'a [Vec<u16>; 3],
    pub media: &'a [crate::materials::Medium],
}

/// Compile the sheets of a simulation for one slab (`None` without sheets
/// or when none of them has a node in the slab).
pub fn compile_sheets(sheets: &[Sheet], ctx: &SheetCtx) -> Result<Option<SheetPlan>> {
    if sheets.is_empty() {
        return Ok(None);
    }
    let grid = ctx.grid;
    let n = grid.n();
    let (ny, nz) = (n[1], n[2]);
    let plane = ny * nz;
    let cdt = C0 * ctx.dt;
    let np = sheets.iter().map(|s| s.conductivity.poles.len()).max().unwrap_or(0);
    let mut coefs = Vec::new();
    let mut cond = Vec::new();
    for s in sheets {
        s.check(ctx.invariant)?;
        for q in 0..np {
            let c = s.conductivity.poles.get(q).map(|p| p.surface_pole().ade_coefficients(ctx.dt)).unwrap_or([1.0, 0.0, 0.0]);
            coefs.extend(c.iter().map(|v| *v as Real));
        }
        cond.push((s.conductivity.conductance * ctx.dt / (2.0 * EPS0)) as Real);
        cond.push(s.conductivity.susceptibility as Real);
    }
    // in-plane cell of an E sample along axis q: primal along its own axis,
    // dual across (the material voxel)
    let span = |c: Component, q: usize, i: usize| -> (f64, f64) { grid.voxel_bounds(c, q, i) };
    let mut lists = Vec::new();
    for (si, s) in sheets.iter().enumerate() {
        let a = s.axis;
        let nodes = node_weights(grid, a, s.position, ctx.periodic[a], &s.name)?;
        let (u, v) = in_plane(a);
        for b in [u, v] {
            let comp = Component::e(b);
            let mut l = SheetList { comp, row: si as u32, idx: vec![], k: vec![], area: vec![] };
            for &(ga, w) in &nodes {
                // local index range of each axis: the node plane along a
                let rng = |q: usize| -> (usize, usize) {
                    if q == a {
                        if q == 0 {
                            if ga < ctx.x0 || ga >= ctx.x0 + ctx.nxl { (0, 0) } else { (ga - ctx.x0, ga - ctx.x0 + 1) }
                        } else {
                            (ga, ga + 1)
                        }
                    } else if q == 0 {
                        (0, ctx.nxl)
                    } else {
                        (0, n[q])
                    }
                };
                let (ri, rj, rk) = (rng(0), rng(1), rng(2));
                let d = 1.0 / ctx.inv_d_full[a][ga] as f64;
                for i in ri.0..ri.1 {
                    for j in rj.0..rj.1 {
                        for k in rk.0..rk.1 {
                            let o = (i * ny + j) * nz + k;
                            let cbv = ctx.cb_local[b][o] as f64;
                            if cbv == 0.0 {
                                continue;
                            }
                            let gidx = [ctx.x0 + i, j, k];
                            let (lu, hu) = span(comp, u, gidx[u]);
                            let (lv, hv) = span(comp, v, gidx[v]);
                            let cov = s.coverage([lu, lv], [hu, hv], ctx.invariant);
                            if cov <= 0.0 {
                                continue;
                            }
                            let md = &ctx.media[ctx.med_local[b][o] as usize];
                            if md.is_tensor() {
                                bail!("sheet '{}' touches medium '{}', which has off-diagonal permittivity; sheets on such media are not supported", s.name, md.name);
                            }
                            // cb / (c dt) = 1 / (eps (1 + f))
                            l.idx.push(((i + 1) * plane + j * nz + k) as u64);
                            l.k.push((w * cov * cbv / cdt / d) as Real);
                            let wid = |q: usize, lo: f64, hi: f64| if Some(q) == ctx.invariant { 1.0 } else { hi - lo };
                            l.area.push(w * cov * wid(u, lu, hu) * wid(v, lv, hv));
                        }
                    }
                }
            }
            if !l.idx.is_empty() {
                lists.push(l);
            }
        }
    }
    if lists.is_empty() {
        return Ok(None);
    }
    Ok(Some(SheetPlan { np, coefs, cond, lists }))
}

/// Volume poles that bound the stability of a sheet on a node of dual size
/// `d` (the strongest a node can carry: weight and coverage 1).
pub fn stability_poles(s: &Sheet, d: f64) -> Vec<Pole> {
    s.conductivity.poles.iter().map(|p| p.volume_pole(d)).collect()
}

/// Smallest dual cell normal to the sheet among the node planes that carry it.
pub fn min_dual(grid: &Grid, s: &Sheet, periodic: bool) -> Result<f64> {
    let inv = grid.inv_dual(s.axis, periodic);
    Ok(node_weights(grid, s.axis, s.position, periodic, &s.name)?.iter().map(|&(i, _)| 1.0 / inv[i] as f64).fold(f64::INFINITY, f64::min))
}

/// The time step reduced, if needed, so that every sheet's poles are
/// stable on the node planes that carry it (see
/// [`crate::materials::poles_stable`]): a sheet on a node of dual size `d`
/// acts as a layer of thickness `d` with the volume poles `chi_s / d`, on the
/// smallest permittivity of the media in use (conservative: weights and
/// coverage below 1 and a larger eps only make it more stable). The
/// conductance and susceptibility are implicit and do not limit the step.
pub fn limit_time_step(sim: &crate::spec::Simulation, grid: &Grid, media: &[crate::materials::Medium], dt_in: f64, dt0: f64, dt_courant: f64) -> Result<f64> {
    let mut dt = dt_in;
    if sim.sheets.is_empty() {
        return Ok(dt);
    }
    let used = |m: &crate::materials::Medium| m.name == sim.background || sim.structures.iter().any(|st| st.medium == m.name);
    let eps_min = media.iter().filter(|m| used(m) && !m.pec).flat_map(|m| m.eps.iter().cloned()).fold(f64::INFINITY, f64::min);
    let eps_min = if eps_min.is_finite() { eps_min } else { 1.0 };
    let mu_min = media.iter().filter(|m| used(m) && !m.pec).flat_map(|m| m.mu.iter().cloned()).fold(f64::INFINITY, f64::min);
    let mu_min = if mu_min.is_finite() { mu_min } else { 1.0 };
    let periodic = [sim.boundaries.x[0].is_periodic(), sim.boundaries.y[0].is_periodic(), sim.boundaries.z[0].is_periodic()];
    for s in &sim.sheets {
        if s.conductivity.poles.is_empty() {
            continue;
        }
        let d = min_dual(grid, s, periodic[s.axis] || sim.invariant() == Some(s.axis))?;
        let poles = stability_poles(s, d);
        let stable = |t: f64| {
            let a_max = 4.0 * (t / dt_courant).powi(2) / (eps_min * mu_min);
            crate::materials::poles_stable(eps_min, 0.0, &poles, t, a_max)
        };
        if stable(dt) {
            continue;
        }
        let (mut lo, mut hi) = (0.0, dt);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if stable(mid) { lo = mid } else { hi = mid }
        }
        dt = 0.98 * lo;
        if dt < 0.1 * dt0 {
            bail!(
                "sheet '{}': its poles on a node plane of dual size {d:.3e} m are stable only below {:.3} of the time step of this mesh ({dt0:.3e} s); \
                 refine the mesh along the sheet's in-plane axes or lower the pole strengths",
                s.name,
                lo / dt0
            );
        }
    }
    Ok(dt)
}

/// Update of one sheet cell before the linear E update (the `sheet_pre`
/// kernel): saves E^n and advances the poles, returning sum_q dP_q.
#[inline]
pub fn sheet_pre_cell(e: Real, coefs: &[Real], np: usize, p: &mut [Real], pp: &mut [Real], t: usize, n: usize) -> Real {
    let mut acc = 0 as Real;
    for q in 0..np {
        let c = &coefs[q * 3..q * 3 + 3];
        let o = q * n + t;
        let p0 = p[o];
        let pn = c[0] * p0 + c[1] * pp[o] + c[2] * e;
        acc += pn - p0;
        pp[o] = p0;
        p[o] = pn;
    }
    acc
}

/// E^{n+1} of one sheet cell after the linear E update (the `sheet_post`
/// kernel): `elin` the linear update, `en` = E^n, `dp` = sum_q dP_q,
/// `k` the cell factor, `h`, `chi0` the row's conductance and
/// susceptibility terms.
#[inline]
pub fn sheet_post_cell(elin: Real, en: Real, dp: Real, k: Real, h: Real, chi0: Real) -> Real {
    (elin - k * dp - k * (h - chi0) * en) / (1.0 + k * (h + chi0))
}

/// Analytic reflection and transmission (power) of a sheet of surface
/// conductivity `sigma` between media of indices `n1` (incidence side) and
/// `n2`, at angle `theta` (rad, in medium 1), s (`p = false`) or p
/// polarisation: with the wave admittances y = n cos(t) / eta0 (s) or
/// n / (eta0 cos(t)) (p) and continuity of E_t, H_t jumping by sigma E_t,
/// r = (y1 - y2 - sigma) / (y1 + y2 + sigma), t = 2 y1 / (y1 + y2 + sigma),
/// R = |r|^2, T = Re(y2) / y1 |t|^2.
pub fn sheet_rt(sigma: Complex64, n1: f64, n2: f64, theta: f64, p: bool) -> (f64, f64) {
    let c1 = theta.cos();
    let st2 = n1 * theta.sin() / n2;
    let c2 = Complex64::new(1.0 - st2 * st2, 0.0).sqrt();
    let (y1, y2) = if p { (Complex64::new(n1 / (ETA0 * c1), 0.0), n2 / (ETA0 * c2)) } else { (Complex64::new(n1 * c1 / ETA0, 0.0), n2 * c2 / ETA0) };
    let d = y1 + y2 + sigma;
    let r = (y1 - y2 - sigma) / d;
    let t = 2.0 * y1 / d;
    (r.norm_sqr(), y2.re / y1.re * t.norm_sqr())
}

/// Wavenumber q of the TM plasmon of a sheet between media of relative
/// permittivities `e1`, `e2` at angular frequency `omega` (retarded, fields
/// `exp(i q x - k_j |z| - i w t)`): `e1 / k1 + e2 / k2 = -i sigma / (omega eps0)`
/// with `k_j = sqrt(q^2 - e_j k0^2)` (Re k_j > 0), solved by Newton's method
/// from the quasi-static value `q = i eps0 (e1 + e2) omega / sigma`. Returns
/// (retarded, quasi-static).
pub fn sheet_plasmon_q(sigma: Complex64, e1: f64, e2: f64, omega: f64) -> (Complex64, Complex64) {
    let i = Complex64::i();
    let k0 = omega / C0;
    let qs = i * EPS0 * (e1 + e2) * omega / sigma;
    // in u = q / k0
    let rhs = -i * sigma / (omega * EPS0) * k0;
    let kz = |u: Complex64, e: f64| {
        let k = (u * u - e).sqrt();
        if k.re < 0.0 { -k } else { k }
    };
    let mut u = qs / k0;
    for _ in 0..100 {
        let (a, b) = (kz(u, e1), kz(u, e2));
        let f = e1 / a + e2 / b - rhs;
        let df = -e1 * u / (a * a * a) - e2 * u / (b * b * b);
        let du = f / df;
        u -= du;
        if du.norm() < 1e-15 * u.norm() {
            break;
        }
    }
    (u * k0, qs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TAU;

    #[test]
    fn discrete_conductivity_tends_to_the_model() {
        let c = SheetConductivity {
            name: "x".into(),
            conductance: 1e-4,
            susceptibility: 1e-9,
            poles: vec![SheetPole::Drude { drude_weight: 5e10, gamma: 1e13 }, SheetPole::Lorentz { strength: 2e9, omega0: 3e14, gamma: 1e13 }],
        };
        let w = TAU * 3e13;
        let (a, b) = (c.sigma(w), c.sigma_discrete(w, 1e-18));
        assert!((a - b).norm() < 1e-4 * a.norm(), "{a} vs {b}");
    }

    #[test]
    fn sheet_rt_limits() {
        // a free-standing sheet at normal incidence: T = |2 / (2 + eta0 sigma)|^2
        let s = Complex64::new(1e-3, 2e-3);
        let (r, t) = sheet_rt(s, 1.0, 1.0, 0.0, false);
        let tt = (2.0 / (2.0 + ETA0 * s)).norm_sqr();
        assert!((t - tt).abs() < 1e-12 && (r - (ETA0 * s / (2.0 + ETA0 * s)).norm_sqr()).abs() < 1e-12);
        // no sheet: Fresnel; s and p agree at normal incidence
        let (r0, t0) = sheet_rt(Complex64::new(0.0, 0.0), 1.0, 1.5, 0.0, true);
        assert!((r0 - 0.04).abs() < 1e-12 && (t0 - 0.96).abs() < 1e-12);
        let (rp, _) = sheet_rt(Complex64::new(0.0, 0.0), 1.0, 1.5, 1.5f64.atan(), true);
        assert!(rp < 1e-20, "Brewster {rp}");
        // energy: R + T = 1 for a lossless (reactive) sheet
        let (r1, t1) = sheet_rt(Complex64::new(0.0, 3e-3), 1.2, 1.7, 0.4, true);
        assert!((r1 + t1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn plasmon_dispersion_quasi_static_limit() {
        // a weak Drude sheet far in the non-retarded regime: q -> quasi-static
        let w = TAU * 1e13;
        let s = Complex64::new(0.0, 1.0) * 1e9 / w;
        let (q, qs) = sheet_plasmon_q(s, 1.0, 3.0, w);
        assert!((q - qs).norm() < 1e-3 * qs.norm(), "{q} {qs}");
    }

    #[test]
    fn polygon_coverage() {
        let s = Sheet {
            name: "p".into(),
            axis: 2,
            position: 0.0,
            conductivity: SheetConductivity::default(),
            min: None,
            max: None,
            polygon: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        };
        assert!((s.coverage([0.0, 0.0], [1.0, 1.0], None) - 0.5).abs() < 0.04);
        assert_eq!(s.coverage([2.0, 2.0], [3.0, 3.0], None), 0.0);
        let r = Sheet { polygon: None, min: Some([0.0, 0.0, 9.0]), max: Some([1.0, 0.5, 9.0]), ..s };
        assert!((r.coverage([0.5, 0.0], [1.5, 1.0], None) - 0.25).abs() < 1e-12);
    }
}
