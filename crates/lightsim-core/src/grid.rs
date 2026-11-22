//! Rectilinear (possibly non-uniform) Yee grid.
//!
//! The grid is defined by its node coordinates (cell edges) along each axis.
//! Cell `i` along an axis spans `[x[i], x[i+1]]`. Yee component positions:
//!
//! | component | x        | y        | z        |
//! |-----------|----------|----------|----------|
//! | Ex        | i + 1/2  | j        | k        |
//! | Ey        | i        | j + 1/2  | k        |
//! | Ez        | i        | j        | k + 1/2  |
//! | Hx        | i        | j + 1/2  | k + 1/2  |
//! | Hy        | i + 1/2  | j        | k + 1/2  |
//! | Hz        | i + 1/2  | j + 1/2  | k        |
//!
//! "half" positions are cell centres, integer positions are nodes.

use crate::Real;
use serde::{Deserialize, Serialize};

/// One of the six Yee field components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Component {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
}

impl Component {
    pub const ALL: [Component; 6] = [
        Component::Ex,
        Component::Ey,
        Component::Ez,
        Component::Hx,
        Component::Hy,
        Component::Hz,
    ];
    pub const E: [Component; 3] = [Component::Ex, Component::Ey, Component::Ez];
    pub const H: [Component; 3] = [Component::Hx, Component::Hy, Component::Hz];

    /// Field axis (0 = x, 1 = y, 2 = z).
    pub fn axis(self) -> usize {
        match self {
            Component::Ex | Component::Hx => 0,
            Component::Ey | Component::Hy => 1,
            Component::Ez | Component::Hz => 2,
        }
    }
    pub fn is_e(self) -> bool {
        matches!(self, Component::Ex | Component::Ey | Component::Ez)
    }
    /// Index 0..6 (Ex, Ey, Ez, Hx, Hy, Hz).
    pub fn index(self) -> usize {
        match self {
            Component::Ex => 0,
            Component::Ey => 1,
            Component::Ez => 2,
            Component::Hx => 3,
            Component::Hy => 4,
            Component::Hz => 5,
        }
    }
    pub fn from_index(i: usize) -> Component {
        Component::ALL[i]
    }
    pub fn e(axis: usize) -> Component {
        Component::E[axis]
    }
    pub fn h(axis: usize) -> Component {
        Component::H[axis]
    }
    /// True if the component sits at a cell centre (half-integer index) along `axis`.
    pub fn is_half(self, axis: usize) -> bool {
        if self.is_e() {
            axis == self.axis()
        } else {
            axis != self.axis()
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Component::Ex => "Ex",
            Component::Ey => "Ey",
            Component::Ez => "Ez",
            Component::Hx => "Hx",
            Component::Hy => "Hy",
            Component::Hz => "Hz",
        }
    }
    pub fn parse(s: &str) -> Option<Component> {
        match s.to_ascii_lowercase().as_str() {
            "ex" => Some(Component::Ex),
            "ey" => Some(Component::Ey),
            "ez" => Some(Component::Ez),
            "hx" => Some(Component::Hx),
            "hy" => Some(Component::Hy),
            "hz" => Some(Component::Hz),
            _ => None,
        }
    }
}

/// Rectilinear grid given by node coordinates along x, y, z (metres).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Grid {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

impl Grid {
    /// Uniform grid covering `[min, min + n*d]` along each axis.
    pub fn uniform(min: [f64; 3], n: [usize; 3], d: [f64; 3]) -> Grid {
        let mk = |a: usize| (0..=n[a]).map(|i| min[a] + i as f64 * d[a]).collect::<Vec<_>>();
        Grid { x: mk(0), y: mk(1), z: mk(2) }
    }

    /// Uniform grid from a box and a target spacing (spacing is adjusted so an
    /// integer number of cells fits exactly).
    pub fn uniform_box(min: [f64; 3], max: [f64; 3], d: [f64; 3]) -> Grid {
        let mut n = [0usize; 3];
        let mut dd = [0f64; 3];
        for a in 0..3 {
            let len = max[a] - min[a];
            n[a] = ((len / d[a]).round() as usize).max(1);
            dd[a] = len / n[a] as f64;
        }
        Grid::uniform(min, n, dd)
    }

    pub fn from_edges(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Grid {
        Grid { x, y, z }
    }

    pub fn edges(&self, axis: usize) -> &[f64] {
        match axis {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }

    pub fn n(&self) -> [usize; 3] {
        [self.x.len() - 1, self.y.len() - 1, self.z.len() - 1]
    }

    pub fn num_cells(&self) -> usize {
        let n = self.n();
        n[0] * n[1] * n[2]
    }

    pub fn min(&self) -> [f64; 3] {
        [self.x[0], self.y[0], self.z[0]]
    }

    pub fn max(&self) -> [f64; 3] {
        [*self.x.last().unwrap(), *self.y.last().unwrap(), *self.z.last().unwrap()]
    }

    /// Width of cell `i` along `axis`.
    pub fn cell_size(&self, axis: usize, i: usize) -> f64 {
        let e = self.edges(axis);
        e[i + 1] - e[i]
    }

    /// Smallest cell size along each axis.
    pub fn min_cell_size(&self) -> [f64; 3] {
        let mut out = [f64::INFINITY; 3];
        for (a, o) in out.iter_mut().enumerate() {
            let e = self.edges(a);
            for w in e.windows(2) {
                *o = o.min(w[1] - w[0]);
            }
        }
        out
    }

    pub fn node(&self, axis: usize, i: usize) -> f64 {
        self.edges(axis)[i]
    }

    pub fn center(&self, axis: usize, i: usize) -> f64 {
        let e = self.edges(axis);
        0.5 * (e[i] + e[i + 1])
    }

    /// Coordinate of component `c` at index `i` along `axis`.
    pub fn position(&self, c: Component, axis: usize, i: usize) -> f64 {
        if c.is_half(axis) { self.center(axis, i) } else { self.node(axis, i) }
    }

    /// 3-D position of component `c` at index (i, j, k).
    pub fn position3(&self, c: Component, ijk: [usize; 3]) -> [f64; 3] {
        [self.position(c, 0, ijk[0]), self.position(c, 1, ijk[1]), self.position(c, 2, ijk[2])]
    }

    /// Bounds of the dual/primal "voxel" owned by component `c` at index `i` along
    /// `axis` (used for sub-pixel averaging).
    pub fn voxel_bounds(&self, c: Component, axis: usize, i: usize) -> (f64, f64) {
        let e = self.edges(axis);
        let n = e.len() - 1;
        if c.is_half(axis) {
            (e[i], e[i + 1])
        } else {
            let lo = if i == 0 { e[0] } else { 0.5 * (e[i - 1] + e[i]) };
            let hi = if i >= n { e[n] } else { 0.5 * (e[i] + e[i + 1]) };
            (lo, hi)
        }
    }

    /// Material-averaging voxel of a component: as [`voxel_bounds`] but the
    /// dual cells of the first and last node extend half a cell beyond the
    /// grid (mirror image of the neighbouring cell), so a node on a domain
    /// face - e.g. a symmetry plane - sees the same geometry as in the
    /// unreduced domain.
    pub fn material_voxel_bounds(&self, c: Component, axis: usize, i: usize) -> (f64, f64) {
        let e = self.edges(axis);
        let n = e.len() - 1;
        if c.is_half(axis) || n == 0 {
            return self.voxel_bounds(c, axis, i);
        }
        let lo = if i == 0 { e[0] - 0.5 * (e[1] - e[0]) } else { 0.5 * (e[i - 1] + e[i]) };
        let hi = if i >= n { e[n] + 0.5 * (e[n] - e[n - 1]) } else { 0.5 * (e[i] + e[i + 1]) };
        (lo, hi)
    }

    /// Inverse primal spacing `1/(x[i+1]-x[i])`, used by the H update.
    pub fn inv_primal(&self, axis: usize) -> Vec<Real> {
        let e = self.edges(axis);
        e.windows(2).map(|w| (1.0 / (w[1] - w[0])) as Real).collect()
    }

    /// Inverse dual spacing `2/(dx[i-1]+dx[i])`, used by the E update. At i = 0
    /// the missing cell is taken from the other end (periodic) or mirrored.
    pub fn inv_dual(&self, axis: usize, periodic: bool) -> Vec<Real> {
        let e = self.edges(axis);
        let n = e.len() - 1;
        let d: Vec<f64> = e.windows(2).map(|w| w[1] - w[0]).collect();
        (0..n)
            .map(|i| {
                let prev = if i == 0 {
                    if periodic { d[n - 1] } else { d[0] }
                } else {
                    d[i - 1]
                };
                (2.0 / (prev + d[i])) as Real
            })
            .collect()
    }

    /// Index of the cell containing `coord` along `axis` (clamped).
    pub fn cell_index(&self, axis: usize, coord: f64) -> usize {
        let e = self.edges(axis);
        let n = e.len() - 1;
        if coord <= e[0] {
            return 0;
        }
        if coord >= e[n] {
            return n - 1;
        }
        match e.binary_search_by(|v| v.partial_cmp(&coord).unwrap()) {
            Ok(i) => i.min(n - 1),
            Err(i) => i - 1,
        }
    }

    /// Index of the node closest to `coord` along `axis` (clamped to 0..n-1).
    pub fn nearest_node(&self, axis: usize, coord: f64) -> usize {
        let e = self.edges(axis);
        let n = e.len() - 1;
        let mut best = 0;
        let mut bd = f64::INFINITY;
        // binary search then check neighbours
        let i = self.cell_index(axis, coord);
        for cand in [i.saturating_sub(1), i, i + 1, i + 2] {
            if cand <= n {
                let dd = (e[cand] - coord).abs();
                if dd < bd {
                    bd = dd;
                    best = cand;
                }
            }
        }
        best.min(n - 1)
    }

    /// Index (along `axis`) of the position of component `c` closest to `coord`.
    pub fn nearest_index(&self, c: Component, axis: usize, coord: f64) -> usize {
        if c.is_half(axis) { self.cell_index(axis, coord) } else { self.nearest_node(axis, coord) }
    }

    /// Index range `[lo, hi)` of component positions inside `[a, b]` along `axis`
    /// (at least one index is returned).
    pub fn index_range(&self, c: Component, axis: usize, a: f64, b: f64) -> (usize, usize) {
        let n = self.n()[axis];
        let tol = 1e-9 * (self.edges(axis)[n] - self.edges(axis)[0]).abs().max(1e-30);
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for i in 0..n {
            let p = self.position(c, axis, i);
            if p >= a - tol && p <= b + tol {
                lo = lo.min(i);
                hi = hi.max(i + 1);
            }
        }
        if lo == usize::MAX {
            let i = self.nearest_index(c, axis, 0.5 * (a + b));
            (i, i + 1)
        } else {
            (lo, hi)
        }
    }

    /// Cell volume for component `c` voxel at (i, j, k).
    pub fn voxel_volume(&self, c: Component, ijk: [usize; 3]) -> f64 {
        let mut v = 1.0;
        for a in 0..3 {
            let (lo, hi) = self.voxel_bounds(c, a, ijk[a]);
            v *= hi - lo;
        }
        v
    }

    /// Sub-grid of the x range `[lo, hi)` (cells), keeping y and z.
    pub fn slab(&self, lo: usize, hi: usize) -> Grid {
        Grid { x: self.x[lo..=hi].to_vec(), y: self.y.clone(), z: self.z.clone() }
    }
}

/// Mesh specification used to generate a grid automatically.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MeshSpec {
    /// Uniform spacing along each axis.
    Uniform { dx: [f64; 3] },
    /// Automatic non-uniform mesh: `steps_per_wavelength` cells per wavelength in
    /// each local medium at `wavelength`, smoothly graded (max ratio `grading`),
    /// plus optional refinement boxes with explicit spacing.
    Auto {
        wavelength: f64,
        #[serde(default = "default_steps")]
        steps_per_wavelength: f64,
        #[serde(default = "default_grading")]
        grading: f64,
        #[serde(default)]
        min_step: Option<f64>,
        #[serde(default)]
        max_step: Option<f64>,
        #[serde(default)]
        overrides: Vec<MeshOverride>,
        /// Mirror planes of the mesh: `symmetry[a] = Some(c)` puts a node on
        /// the plane `axis a = c` and makes the cells on its two sides mirror
        /// images of each other. `None` (the default) builds the mesh
        /// outwards from the centre of the region, which is symmetric
        /// whenever the structures are symmetric about that centre.
        #[serde(default)]
        symmetry: [Option<f64>; 3],
    },
    /// Explicit node coordinates.
    Explicit { x: Vec<f64>, y: Vec<f64>, z: Vec<f64> },
}

fn default_steps() -> f64 {
    20.0
}
fn default_grading() -> f64 {
    1.3
}

/// Refinement box: inside `[min, max]`, spacing along each axis is at most `dx`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeshOverride {
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub dx: [f64; 3],
}

/// Build a 1-D graded mesh over `[a, b]` given a target local spacing function
/// sampled on intervals. `targets` is a list of `(x0, x1, step)` intervals; the
/// smallest applicable step wins and transitions are graded by `grading`.
///
/// The mesh is built outwards from the centre of `[a, b]`, so it is mirror
/// symmetric whenever the targets are (see [`graded_axis_about`]).
pub fn graded_axis(a: f64, b: f64, default_step: f64, targets: &[(f64, f64, f64)], grading: f64) -> Vec<f64> {
    graded_axis_about(a, b, None, default_step, targets, grading)
}

/// As [`graded_axis`], built outwards from a seed node at `plane` (default:
/// the centre of `[a, b]`).
///
/// The two halves are marched with the same rule from the same point, so the
/// cells on the two sides of the seed are mirror images of each other as far
/// as the spacing function is symmetric about it. Marching from one end of
/// the axis, as an earlier version did, gave the two arms of a symmetric
/// coupler different cells. With an explicit `plane` the targets are also
/// mirrored about it, which makes the spacing function symmetric by
/// construction. An axis shorter than 1.3 steps stays a single cell, and the
/// cells at the two ends of the axis are between 0.65 and 1.3 steps.
pub fn graded_axis_about(a: f64, b: f64, plane: Option<f64>, default_step: f64, targets: &[(f64, f64, f64)], grading: f64) -> Vec<f64> {
    let span = b - a;
    let seed = plane.unwrap_or(0.5 * (a + b)).clamp(a, b);
    let mut all: Vec<(f64, f64, f64)> = targets.to_vec();
    if plane.is_some() {
        all.extend(targets.iter().map(|&(x0, x1, st)| (2.0 * seed - x1, 2.0 * seed - x0, st)));
    }
    // Desired spacing at x: min over intervals containing x, else default; then
    // limit growth so that neighbouring cells differ by at most `grading`:
    // s(x) <= step + (grading - 1) * distance to the interval.
    let g = (grading - 1.0).max(1e-3);
    let eps = 1e-9 * span.abs().max(1e-30);
    let limited = |x: f64| -> f64 {
        let mut s = default_step;
        for &(x0, x1, st) in &all {
            let d = if x < x0 - eps {
                x0 - x
            } else if x > x1 + eps {
                x - x1
            } else {
                0.0
            };
            s = s.min(st + g * d);
        }
        s
    };
    // one cell if the axis is shorter than 1.3 steps (a 2-D simulation)
    if plane.is_none() && span < 1.3 * limited(a).min(limited(b)) {
        return vec![a, b];
    }
    // distances of the nodes from the seed, marching over `len` with spacing h(d)
    let march = |h: &dyn Fn(f64) -> f64, len: f64| -> Vec<f64> {
        let mut out = vec![];
        let mut d = 0.0;
        while d < len - 1e-12 * span.abs().max(1e-30) {
            let s = h(d).min(h((d + h(d)).min(len)));
            // the end of the axis: one cell of up to 1.3 steps, or two equal
            // cells of 0.65 to 1 step, never a sliver (the smallest cell of
            // the grid sets the time step)
            let rest = len - d;
            if rest <= 1.3 * s {
                out.push(len);
                break;
            }
            if rest < 2.0 * s {
                out.push(d + 0.5 * rest);
                out.push(len);
                break;
            }
            d += s;
            out.push(d);
            if out.len() > 50_000_000 {
                break;
            }
        }
        out
    };
    let up = march(&|d| limited(seed + d), b - seed);
    let mut down = march(&|d| limited(seed - d), seed - a);
    // exact mirror images where the two sides agree to rounding
    for (k, d) in down.iter_mut().enumerate() {
        match up.get(k) {
            Some(&u) if (u - *d).abs() <= 1e-9 * u.abs() => *d = u,
            _ => break,
        }
    }
    let mut nodes: Vec<f64> = down.iter().rev().map(|d| seed - d).collect();
    if seed - a > 0.0 || down.is_empty() {
        nodes.push(seed);
    }
    nodes.extend(up.iter().map(|d| seed + d));
    // the ends are the region's
    let n = nodes.len();
    nodes[0] = a;
    nodes[n - 1] = b;
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_basics() {
        let g = Grid::uniform([0.0; 3], [10, 5, 4], [0.1, 0.2, 0.25]);
        assert_eq!(g.n(), [10, 5, 4]);
        assert!((g.cell_size(0, 3) - 0.1).abs() < 1e-12);
        assert_eq!(g.cell_index(0, 0.35), 3);
        assert_eq!(g.nearest_node(0, 0.36), 4);
        let inv = g.inv_dual(1, false);
        assert!((inv[0] - 5.0).abs() < 1e-5);
        // Ex at half position along x
        assert!((g.position(Component::Ex, 0, 2) - 0.25).abs() < 1e-12);
        assert!((g.position(Component::Ex, 1, 2) - 0.4).abs() < 1e-12);
        assert!((g.position(Component::Hz, 1, 2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn graded_mesh_refines() {
        let nodes = graded_axis(0.0, 10.0, 1.0, &[(4.0, 6.0, 0.1)], 1.2);
        let steps: Vec<f64> = nodes.windows(2).map(|w| w[1] - w[0]).collect();
        let max_in = steps
            .iter()
            .zip(nodes.iter())
            .filter(|(_, x)| **x > 4.05 && **x < 5.9)
            .map(|(s, _)| *s)
            .fold(0.0, f64::max);
        assert!(max_in <= 0.1 + 1e-9, "max step inside refined region {max_in}");
        for w in steps.windows(2) {
            assert!(w[1] / w[0] < 1.6 && w[0] / w[1] < 1.6, "grading violated {:?}", w);
        }
        assert!((nodes.last().unwrap() - 10.0).abs() < 1e-12);
    }

    fn steps(nodes: &[f64]) -> Vec<f64> {
        nodes.windows(2).map(|w| w[1] - w[0]).collect()
    }

    #[test]
    fn graded_mesh_is_mirror_symmetric_for_symmetric_targets() {
        // two fine intervals placed symmetrically about the centre of [0, 10]
        let targets = [(2.0, 2.6, 0.07), (7.4, 8.0, 0.07)];
        let nodes = graded_axis(0.0, 10.0, 0.5, &targets, 1.3);
        let st = steps(&nodes);
        let n = st.len();
        for k in 0..n {
            assert!((st[k] - st[n - 1 - k]).abs() <= 1e-12, "cell {k}: {} vs {}", st[k], st[n - 1 - k]);
            // the nodes themselves are mirror images about x = 5
            assert!((nodes[k] + nodes[n - k] - 10.0).abs() <= 1e-12);
        }
        assert!(nodes.iter().any(|x| (x - 5.0).abs() < 1e-12), "a node sits on the mirror plane");
        // the same cells cover both fine intervals
        let cells_in = |lo: f64, hi: f64| nodes.windows(2).filter(|w| w[1] > lo && w[0] < hi).count();
        assert_eq!(cells_in(2.0, 2.6), cells_in(7.4, 8.0));
    }

    #[test]
    fn graded_mesh_about_an_off_centre_plane() {
        // one fine interval on one side only: the mirror plane at x = 3 makes
        // the mesh symmetric about it over the common extent
        let nodes = graded_axis_about(0.0, 10.0, Some(3.0), 0.5, &[(3.4, 4.0, 0.07)], 1.3);
        let i0 = nodes.iter().position(|x| (x - 3.0).abs() < 1e-12).expect("node on the plane");
        for k in 1..i0.min(nodes.len() - 1 - i0) {
            let (lo, hi) = (nodes[i0 - k], nodes[i0 + k]);
            if lo <= 1.1 {
                break; // the last two cells before the end of the region take up the remainder
            }
            assert!((3.0 - lo - (hi - 3.0)).abs() <= 1e-12, "node {k}: {lo} vs {hi}");
        }
        assert_eq!(nodes[0], 0.0);
        assert_eq!(*nodes.last().unwrap(), 10.0);
    }

    #[test]
    fn graded_mesh_keeps_a_thin_axis_as_one_cell() {
        assert_eq!(graded_axis(0.0, 0.04, 0.05, &[], 1.3), vec![0.0, 0.04]);
        assert_eq!(graded_axis(0.0, 0.06, 0.05, &[], 1.3), vec![0.0, 0.06]);
        assert_eq!(graded_axis(0.0, 0.1, 0.05, &[], 1.3).len(), 3);
    }

    #[test]
    fn graded_mesh_has_no_sliver_cells() {
        // whatever the length, no cell is below 0.65 or above 1.3 of the step
        for k in 0..200 {
            let len = 1.0 + 0.0137 * k as f64;
            let st = steps(&graded_axis(0.0, len, 0.1, &[], 1.3));
            let (mn, mx) = st.iter().fold((f64::INFINITY, 0.0f64), |(a, b), v| (a.min(*v), b.max(*v)));
            assert!(mn >= 0.065 - 1e-12 && mx <= 0.13 + 1e-12, "length {len}: cells from {mn} to {mx}");
        }
    }
}
