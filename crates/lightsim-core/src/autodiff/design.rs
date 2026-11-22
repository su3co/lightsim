//! Design regions: index grids whose samples are the optimisation variables.
//!
//! [`DesignMap`] knows, for every E component, which Yee voxels each design
//! grid owns and how their permittivity follows from the grid samples. It
//! evaluates the samples through the rasteriser's own functions
//! ([`Shape::grid_eps`], [`center_inside`], [`e_update_coefficients`]), so a
//! design update reproduces a full re-compilation bit for bit, and it
//! scatters per-voxel gradients with the weights of [`Shape::grid_stencil`],
//! the stencil `grid_eps` is evaluated with, so the pullback is the exact
//! transpose of the rasterisation.

use crate::geometry::Shape;
use crate::grid::{Component, Grid};
use crate::plan::{BC_MIRROR, GhostRule, Plan, e_update_coefficients};
use crate::raster::center_inside;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// One index grid of the design: `dims` samples spanning `[min, max]`
/// (nodes on both faces), carried by the medium named `medium`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DesignGrid {
    #[serde(default)]
    pub name: String,
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub dims: [usize; 3],
    pub medium: String,
}

impl DesignGrid {
    pub fn samples(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    /// The structure shape for the given samples (x-major).
    pub fn shape(&self, eps: Vec<f64>) -> Shape {
        Shape::IndexGrid { min: self.min, max: self.max, dims: self.dims, eps, eps_axes: None }
    }
}

/// Index box of one field component: first index and size per axis
/// (interior indices of a single-slab plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct IndexBox {
    pub lo: [usize; 3],
    pub size: [usize; 3],
}

impl IndexBox {
    pub fn cells(&self) -> usize {
        self.size[0] * self.size[1] * self.size[2]
    }
    pub fn is_empty(&self) -> bool {
        self.cells() == 0
    }
    pub fn hi(&self) -> [usize; 3] {
        [self.lo[0] + self.size[0], self.lo[1] + self.size[1], self.lo[2] + self.size[2]]
    }
    /// Box-local offset (x-major) of global index (i, j, k); the index must be inside.
    pub fn offset(&self, i: usize, j: usize, k: usize) -> usize {
        ((i - self.lo[0]) * self.size[1] + (j - self.lo[1])) * self.size[2] + (k - self.lo[2])
    }
    pub fn contains(&self, i: usize, j: usize, k: usize) -> bool {
        let h = self.hi();
        i >= self.lo[0] && i < h[0] && j >= self.lo[1] && j < h[1] && k >= self.lo[2] && k < h[2]
    }
    /// Smallest box containing both (an empty box is the neutral element).
    pub fn union(&self, o: &IndexBox) -> IndexBox {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        let (a, b) = (self.hi(), o.hi());
        let lo = [self.lo[0].min(o.lo[0]), self.lo[1].min(o.lo[1]), self.lo[2].min(o.lo[2])];
        IndexBox { lo, size: [a[0].max(b[0]) - lo[0], a[1].max(b[1]) - lo[1], a[2].max(b[2]) - lo[2]] }
    }
    /// Storage index (one ghost plane on each side of x) of box-local offset `t`.
    pub fn storage(&self, t: usize, ny: usize, nz: usize) -> usize {
        let k = t % self.size[2];
        let j = (t / self.size[2]) % self.size[1];
        let i = t / (self.size[2] * self.size[1]);
        ((self.lo[0] + i + 1) * ny + self.lo[1] + j) * nz + self.lo[2] + k
    }
}

/// New coefficients of one E component over its design bounding box.
#[derive(Clone, Debug)]
pub struct CoefficientBlock {
    pub axis: usize,
    pub bx: IndexBox,
    pub cb: Vec<f32>,
    pub ca: Option<Vec<f32>>,
}

/// The design grids of a simulation and their relation to the Yee grid.
#[derive(Clone, Debug)]
pub struct DesignMap {
    pub grids: Vec<DesignGrid>,
    /// `Shape::IndexGrid` of every grid with its current samples.
    shapes: Vec<Shape>,
    /// Medium index (plan media table) of every grid.
    media: Vec<usize>,
    /// `owned[g][a]`: index box of the E_a positions inside grid `g`
    /// (half-open along every axis, like the rasteriser).
    pub owned: Vec<[IndexBox; 3]>,
    /// `bbox[a]`: union of `owned[g][a]` over the grids.
    pub bbox: [IndexBox; 3],
    /// Union of `bbox` over the three components.
    pub hull: IndexBox,
    /// Component positions per E axis and grid axis.
    pos: [[Vec<f64>; 3]; 3],
    /// `fixed[a][t]`: voxel of `bbox[a]` whose coefficients are imposed by a
    /// wall condition (cb = 0) and never change.
    fixed: [Vec<bool>; 3],
    dt: f64,
    n: [usize; 3],
    /// Ownership tolerance of the rasteriser (`raster::rasterize`).
    tol: f64,
}

impl DesignMap {
    /// Build the map for `grids` on a compiled single-slab plan. The plan
    /// must have been compiled with the grids as its last structures.
    pub fn new(plan: &Plan, grids: &[DesignGrid]) -> Result<DesignMap> {
        if plan.x0 != 0 || plan.nxl != plan.n[0] {
            bail!("design regions need a single-slab plan");
        }
        let grid: &Grid = &plan.grid;
        let n = plan.n;
        let tol = 1e-6 * grid.min_cell_size().iter().cloned().fold(f64::INFINITY, f64::min);
        let mut pos: [[Vec<f64>; 3]; 3] = Default::default();
        for a in 0..3 {
            for ax in 0..3 {
                pos[a][ax] = (0..n[ax]).map(|i| grid.position(Component::e(a), ax, i)).collect();
            }
            // 2-D simulations: the rasteriser evaluates every component on the slice plane
            if let Some((w, c)) = plan.slice {
                pos[a][w] = vec![c; n[w]];
            }
        }
        let mut media = vec![];
        let mut owned = vec![];
        let mut shapes = vec![];
        for g in grids {
            if g.samples() == 0 {
                bail!("design grid '{}': dims {:?} must all be >= 1", g.name, g.dims);
            }
            let m = plan
                .media
                .iter()
                .position(|m| m.name == g.medium)
                .ok_or_else(|| anyhow::anyhow!("design grid '{}' uses unknown medium '{}'", g.name, g.medium))?;
            let md = &plan.media[m];
            if md.pec || md.is_dispersive() || md.is_tensor() || md.is_nonlinear() {
                bail!("design grid '{}': the medium must be a plain or conductive dielectric (no PEC, poles, off-diagonal terms or chi2/chi3)", g.name);
            }
            let mut boxes = [IndexBox::default(); 3];
            for a in 0..3 {
                let mut bx = IndexBox::default();
                for ax in 0..3 {
                    // positions increase with the index: the owned ones form a range
                    let inside: Vec<usize> = (0..n[ax]).filter(|&i| pos[a][ax][i] >= g.min[ax] - tol && pos[a][ax][i] < g.max[ax] - tol).collect();
                    if let (Some(&lo), Some(&hi)) = (inside.first(), inside.last()) {
                        bx.lo[ax] = lo;
                        bx.size[ax] = hi + 1 - lo;
                    }
                }
                if bx.is_empty() {
                    bx = IndexBox::default();
                }
                boxes[a] = bx;
            }
            // The reverse pass does not reconstruct the fields inside the
            // absorbing layers, and the session's adjoint formula does not
            // hold in the stretched coordinates of a CPML: a design voxel
            // there would get a wrong gradient.
            for (a, bx) in boxes.iter().enumerate() {
                if bx.is_empty() {
                    continue;
                }
                if let Some(p) = plan.pml.iter().find(|p| (0..3).all(|ax| bx.lo[ax] < p.start[ax] + p.size[ax] && p.start[ax] < bx.lo[ax] + bx.size[ax])) {
                    bail!(
                        "design grid '{}' overlaps the absorbing layer (PML) along {} (E{} voxels); keep the design grids inside the simulation region, outside the PML",
                        g.name,
                        ["x", "y", "z"][p.axis],
                        ["x", "y", "z"][a]
                    );
                }
            }
            media.push(m);
            owned.push(boxes);
            shapes.push(g.shape(vec![md.eps[0]; g.samples()]));
        }
        let mut bbox = [IndexBox::default(); 3];
        for a in 0..3 {
            for o in &owned {
                bbox[a] = bbox[a].union(&o[a]);
            }
        }
        let hull = bbox[0].union(&bbox[1]).union(&bbox[2]);
        let (ny, nz) = (n[1], n[2]);
        let fixed = [0, 1, 2].map(|a| (0..bbox[a].cells()).map(|t| plan.cb[a][bbox[a].storage(t, ny, nz)] == 0.0).collect::<Vec<bool>>());
        Ok(DesignMap { grids: grids.to_vec(), shapes, media, owned, bbox, hull, pos, fixed, dt: plan.dt, n, tol })
    }

    /// Current samples of grid `g`.
    pub fn eps(&self, g: usize) -> &[f64] {
        match &self.shapes[g] {
            Shape::IndexGrid { eps, .. } => eps,
            _ => unreachable!(),
        }
    }

    /// Set the samples of every grid (x-major, `dims[0] * dims[1] * dims[2]` each).
    pub fn set_eps(&mut self, eps: &[Vec<f64>]) -> Result<()> {
        if eps.len() != self.grids.len() {
            bail!("expected {} permittivity arrays, got {}", self.grids.len(), eps.len());
        }
        for (g, e) in eps.iter().enumerate() {
            if e.len() != self.grids[g].samples() {
                bail!("design grid '{}' has {} samples, got {}", self.grids[g].name, self.grids[g].samples(), e.len());
            }
            if e.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                bail!("design grid '{}': permittivity samples must be finite and > 0", self.grids[g].name);
            }
        }
        for (g, e) in eps.iter().enumerate() {
            self.shapes[g] = self.grids[g].shape(e.clone());
        }
        Ok(())
    }

    fn position(&self, a: usize, i: usize, j: usize, k: usize) -> [f64; 3] {
        [self.pos[a][0][i], self.pos[a][1][j], self.pos[a][2][k]]
    }

    /// True if E_a voxel (i, j, k) takes its permittivity from grid `g`: it
    /// lies inside the grid and no later grid covers it (painter's rule).
    pub fn owner(&self, g: usize, a: usize, i: usize, j: usize, k: usize) -> bool {
        if !self.owned[g][a].contains(i, j, k) {
            return false;
        }
        let p = self.position(a, i, j, k);
        debug_assert!(center_inside(&self.shapes[g], p, self.tol));
        !(g + 1..self.grids.len()).any(|h| self.owned[h][a].contains(i, j, k) && center_inside(&self.shapes[h], p, self.tol))
    }

    /// Visit the voxels of E component `a` owned by grid `g`:
    /// `f(i, j, k, position)`.
    fn for_owned(&self, g: usize, a: usize, mut f: impl FnMut(usize, usize, usize, [f64; 3])) {
        let bx = self.owned[g][a];
        let hi = bx.hi();
        for i in bx.lo[0]..hi[0] {
            for j in bx.lo[1]..hi[1] {
                for k in bx.lo[2]..hi[2] {
                    if self.owner(g, a, i, j, k) {
                        f(i, j, k, self.position(a, i, j, k));
                    }
                }
            }
        }
    }

    /// Permittivity of the E_a voxels over `bbox[a]` as the rasteriser
    /// evaluates it (NaN where no grid owns the voxel).
    pub fn yee_eps(&self, a: usize) -> Vec<f32> {
        let bx = self.bbox[a];
        let mut out = vec![f32::NAN; bx.cells()];
        for g in 0..self.grids.len() {
            self.for_owned(g, a, |i, j, k, p| {
                out[bx.offset(i, j, k)] = self.shapes[g].grid_eps(a, p).unwrap() as f32;
            });
        }
        out
    }

    /// Single-precision permittivity of the E_a voxel (i, j, k) owned by grid `g`.
    pub fn yee_eps_at(&self, g: usize, a: usize, i: usize, j: usize, k: usize) -> f32 {
        self.shapes[g].grid_eps(a, self.position(a, i, j, k)).unwrap() as f32
    }

    /// Write the E-update coefficients of the owned voxels into the plan's
    /// arrays and return the changed blocks (one per component with a
    /// non-empty bounding box) for upload.
    pub fn write_plan(&self, plan: &mut Plan) -> Vec<CoefficientBlock> {
        let (ny, nz) = (self.n[1], self.n[2]);
        let mut out = vec![];
        for a in 0..3 {
            let bx = self.bbox[a];
            if bx.is_empty() {
                continue;
            }
            for g in 0..self.grids.len() {
                let md = plan.media[self.media[g]].clone();
                let mut upd: Vec<(usize, f32, f32)> = vec![];
                self.for_owned(g, a, |i, j, k, p| {
                    if self.fixed[a][bx.offset(i, j, k)] {
                        return;
                    }
                    let eps = self.shapes[g].grid_eps(a, p).unwrap() as f32;
                    let (ca, cb) = e_update_coefficients(&md, a, eps, self.dt);
                    upd.push((((i + 1) * ny + j) * nz + k, ca, cb));
                });
                for (s, ca, cb) in upd {
                    plan.cb[a][s] = cb;
                    if let Some(c) = plan.ca.as_mut() {
                        c[a][s] = ca;
                    }
                }
            }
            let cb: Vec<f32> = (0..bx.cells()).map(|t| plan.cb[a][bx.storage(t, ny, nz)]).collect();
            let ca = plan.ca.as_ref().map(|c| (0..bx.cells()).map(|t| c[a][bx.storage(t, ny, nz)]).collect::<Vec<f32>>());
            out.push(CoefficientBlock { axis: a, bx, cb, ca });
        }
        out
    }

    /// Transpose of the rasterisation: add to `out[g]` (one array per grid,
    /// x-major samples) the pullback of the per-voxel gradient `data` of E
    /// component `a` given over the index box `bx` (voxels outside `bx` or
    /// not owned by a grid contribute nothing). `only` restricts to one grid.
    pub fn pullback(&self, a: usize, bx: &IndexBox, data: &[f64], only: Option<usize>, out: &mut [Vec<f64>]) {
        for g in 0..self.grids.len() {
            if only.is_some_and(|o| o != g) {
                continue;
            }
            let acc = &mut out[g];
            self.for_owned(g, a, |i, j, k, p| {
                if !bx.contains(i, j, k) {
                    return;
                }
                let v = data[bx.offset(i, j, k)];
                if v == 0.0 {
                    return;
                }
                let (idx, wt, cnt) = self.shapes[g].grid_stencil(p).unwrap();
                for c in 0..cnt {
                    acc[idx[c]] += wt[c] * v;
                }
            });
        }
    }

    /// d(ca)/d(eps) and d(cb)/d(eps) of the E_a voxel owned by grid `g` with
    /// permittivity `eps` (matching [`e_update_coefficients`]).
    pub fn coefficient_derivatives(&self, plan: &Plan, g: usize, a: usize, eps: f64) -> (f64, f64) {
        let md = &plan.media[self.media[g]];
        let cdt = crate::constants::C0 * self.dt;
        if md.sigma[a] > 0.0 {
            let s = md.sigma[a] * self.dt / (2.0 * crate::constants::EPS0);
            (2.0 * s / ((eps + s) * (eps + s)), -cdt / ((eps + s) * (eps + s)))
        } else {
            (0.0, -cdt / (eps * eps))
        }
    }

    /// Visit the owned voxels of every grid for component `a`:
    /// `f(grid, i, j, k, eps)`.
    pub fn for_each_owned(&self, a: usize, mut f: impl FnMut(usize, usize, usize, usize, f64)) {
        for g in 0..self.grids.len() {
            self.for_owned(g, a, |i, j, k, p| f(g, i, j, k, self.shapes[g].grid_eps(a, p).unwrap()));
        }
    }
}

/// Volume of the Yee voxel of component `c` at global index `g`, as used by
/// the continuous adjoint formula: primal cell size along the half-integer
/// axes, dual cell size (clamped at the walls, wrapped along periodic axes)
/// along the others.
pub fn voxel_volume(grid: &Grid, periodic: [bool; 3], c: Component, g: [usize; 3]) -> f64 {
    let mut v = 1.0;
    for a in 0..3 {
        let e = grid.edges(a);
        let n = e.len() - 1;
        let i = g[a];
        let w = if c.is_half(a) {
            e[i + 1] - e[i]
        } else if periodic[a] && i == 0 {
            0.5 * ((e[1] - e[0]) + (e[n] - e[n - 1]))
        } else {
            let lo = if i > 0 { 0.5 * (e[i - 1] + e[i]) } else { e[i] };
            let hi = if i < n { 0.5 * (e[i] + e[i + 1]) } else { e[i] };
            hi - lo
        };
        v *= w;
    }
    v
}

/// "Update volume" of a field sample: the product of the spacings its
/// update divides by (the primal spacing along the half-integer axes, the
/// dual one along the others), with the dual cell halved at a mirror (PMC)
/// low wall, where the update doubles the one-sided difference. These are
/// the diagonal matrices `V_E`, `V_H` of the transposition identity
/// `Cd^T = V_H Cp V_E^-1` (see the module documentation).
pub fn update_volume(plan: &Plan, c: Component, g: [usize; 3]) -> f64 {
    let mut v = 1.0;
    for a in 0..3 {
        let i = g[a];
        // E samples divide by the dual spacing along their node axes and H
        // samples by the primal one along their half axes; the transposition
        // pairs every difference with the spacing of the *other* field
        let w = if c.is_half(a) {
            1.0 / plan.inv_p[a][i] as f64
        } else {
            let mirror = match a {
                0 => matches!(plan.ghost_h_lo, GhostRule::Mirror { .. }),
                1 => plan.bc_y[0] == BC_MIRROR,
                _ => plan.bc_z[0] == BC_MIRROR,
            };
            let d = 1.0 / plan.inv_d[a][i] as f64;
            if mirror && i == 0 { 0.5 * d } else { d }
        };
        v *= w;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Structure;
    use crate::grid::MeshSpec;
    use crate::materials::Medium;
    use crate::plan::{CompileOptions, compile};
    use crate::spec::{Boundaries, Boundary, Simulation};

    fn scene(eps: &[Vec<f64>], grids: &[DesignGrid], lossy: bool) -> Simulation {
        let mut sim = Simulation::new([0.0; 3], [2.0e-6, 1.6e-6, 0.8e-6], MeshSpec::Uniform { dx: [0.1e-6; 3] }, Boundaries::all(Boundary::Pec), 10e-15);
        sim.media.push(Medium::dielectric(2.1).with_name("ox"));
        sim.media.push(Medium::dielectric(12.0).with_name("si"));
        let mut d = Medium::dielectric(12.0).with_name("design");
        if lossy {
            d.sigma = [300.0; 3];
        }
        sim.media.push(d);
        sim.background = "ox".into();
        sim.structures.push(Structure { name: "wg".into(), shape: Shape::Box { min: [-1.0, 0.5e-6, 0.2e-6], max: [1.0, 1.1e-6, 0.6e-6] }, medium: "si".into() });
        for (g, e) in grids.iter().zip(eps) {
            sim.structures.push(Structure { name: g.name.clone(), shape: g.shape(e.clone()), medium: g.medium.clone() });
        }
        sim
    }

    fn grids() -> Vec<DesignGrid> {
        vec![
            DesignGrid { name: "a".into(), min: [0.43e-6, 0.31e-6, 0.2e-6], max: [1.52e-6, 1.27e-6, 0.4e-6], dims: [7, 5, 1], medium: "design".into() },
            // overlaps the first one along z: the later grid wins
            DesignGrid { name: "b".into(), min: [0.43e-6, 0.31e-6, 0.35e-6], max: [1.52e-6, 1.27e-6, 0.6e-6], dims: [4, 6, 2], medium: "design".into() },
        ]
    }

    fn samples(seed: u64, g: &DesignGrid) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..g.samples())
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                2.0 + 10.0 * ((s >> 33) as f64 / (1u64 << 31) as f64)
            })
            .collect()
    }

    /// A design update through the map equals a full re-compilation, bit for bit.
    #[test]
    fn design_update_matches_recompilation() {
        for lossy in [false, true] {
            let gs = grids();
            let e0: Vec<Vec<f64>> = gs.iter().map(|g| samples(1, g)).collect();
            let e1: Vec<Vec<f64>> = gs.iter().map(|g| samples(2, g)).collect();
            let mut plan = compile(&scene(&e0, &gs, lossy), &CompileOptions::default()).unwrap();
            let want = compile(&scene(&e1, &gs, lossy), &CompileOptions::default()).unwrap();
            let mut map = DesignMap::new(&plan, &gs).unwrap();
            map.set_eps(&e1).unwrap();
            let blocks = map.write_plan(&mut plan);
            assert_eq!(blocks.len(), 3);
            for a in 0..3 {
                assert_eq!(plan.cb[a], want.cb[a], "cb axis {a} lossy {lossy}");
                if lossy {
                    assert_eq!(plan.ca.as_ref().unwrap()[a], want.ca.as_ref().unwrap()[a]);
                }
                assert!(blocks[a].bx.cells() > 50);
            }
        }
    }

    /// <W eps, g> = <eps, W^T g> for the sampling W of every component.
    #[test]
    fn pullback_is_the_transpose_of_the_sampling() {
        let gs = grids();
        let e: Vec<Vec<f64>> = gs.iter().map(|g| samples(3, g)).collect();
        let plan = compile(&scene(&e, &gs, false), &CompileOptions::default()).unwrap();
        let mut map = DesignMap::new(&plan, &gs).unwrap();
        map.set_eps(&e).unwrap();
        for a in 0..3 {
            let bx = map.bbox[a];
            let yee = map.yee_eps(a);
            let g: Vec<f64> = (0..bx.cells()).map(|t| ((t * 2654435761) % 1000) as f64 / 500.0 - 1.0).collect();
            let lhs: f64 = yee.iter().zip(&g).filter(|(y, _)| !y.is_nan()).map(|(y, g)| *y as f64 * g).sum();
            let mut out: Vec<Vec<f64>> = gs.iter().map(|g| vec![0.0; g.samples()]).collect();
            map.pullback(a, &bx, &g, None, &mut out);
            let rhs: f64 = out.iter().zip(&e).map(|(o, e)| o.iter().zip(e).map(|(a, b)| a * b).sum::<f64>()).sum();
            assert!((lhs - rhs).abs() < 2e-6 * lhs.abs().max(1.0), "axis {a}: {lhs} vs {rhs}");
            // both grids receive gradient, and the overlap goes to the later one only
            assert!(out.iter().all(|o| o.iter().any(|v| *v != 0.0)));
        }
    }
}
