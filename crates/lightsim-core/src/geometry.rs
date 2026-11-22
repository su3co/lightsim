//! Geometric primitives and structures.

use serde::{Deserialize, Serialize};

/// A geometric shape. All coordinates in metres.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape {
    /// Axis-aligned box.
    Box { min: [f64; 3], max: [f64; 3] },
    Sphere { center: [f64; 3], radius: f64 },
    /// Circular cylinder along `axis` (0, 1, 2).
    Cylinder { center: [f64; 3], radius: f64, length: f64, axis: usize },
    /// Polygon in the plane normal to `axis`, extruded over `[slab_min, slab_max]`
    /// along `axis`. Vertices are given in the two remaining axes in natural
    /// order (axis 2 -> (x, y); axis 1 -> (x, z); axis 0 -> (y, z)).
    PolySlab {
        vertices: Vec<[f64; 2]>,
        #[serde(default = "default_axis")]
        axis: usize,
        slab_min: f64,
        slab_max: f64,
    },
    /// Imported / graded permittivity: a regular grid of relative-permittivity
    /// samples spanning the box `[min, max]` (nodes on both faces inclusive),
    /// x-major `eps[(i * ny + j) * nz + k]` with `dims = [nx, ny, nz]`. A
    /// dimension of 1 means "uniform along that axis". Inside the box the
    /// permittivity seen by a Yee E component is the trilinear interpolation
    /// of the samples at the component position; it *replaces* the `eps` of
    /// the structure's medium, which only supplies its name, loss
    /// (conductivity), chi2/chi3 and permeability. Dispersive (pole), PEC and
    /// off-diagonal tensor media are rejected for index grids.
    ///
    /// `eps_axes = [eps_x, eps_y, eps_z]` (each laid out like `eps`) gives a
    /// diagonal-anisotropic grid; when present it is used instead of `eps`.
    IndexGrid {
        min: [f64; 3],
        max: [f64; 3],
        dims: [usize; 3],
        #[serde(default)]
        eps: Vec<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        eps_axes: Option<[Vec<f64>; 3]>,
        /// Relative permeability samples (laid out like `eps`): a
        /// permeability grid. The H components it owns take the
        /// interpolated `mu`; its E components then take the medium's `eps`
        /// (when `eps` and `eps_axes` are empty).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        mu: Vec<f64>,
    },
}

fn default_axis() -> usize {
    2
}

/// In-plane axes for a slab normal to `axis` (natural order).
pub fn plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

impl Shape {
    pub fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        match self {
            Shape::Box { min, max } | Shape::IndexGrid { min, max, .. } => (*min, *max),
            Shape::Sphere { center, radius } => {
                let r = *radius;
                (
                    [center[0] - r, center[1] - r, center[2] - r],
                    [center[0] + r, center[1] + r, center[2] + r],
                )
            }
            Shape::Cylinder { center, radius, length, axis } => {
                let mut lo = [center[0] - radius, center[1] - radius, center[2] - radius];
                let mut hi = [center[0] + radius, center[1] + radius, center[2] + radius];
                lo[*axis] = center[*axis] - length / 2.0;
                hi[*axis] = center[*axis] + length / 2.0;
                (lo, hi)
            }
            Shape::PolySlab { vertices, axis, slab_min, slab_max } => {
                let (u, v) = plane_axes(*axis);
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for p in vertices {
                    lo[u] = lo[u].min(p[0]);
                    hi[u] = hi[u].max(p[0]);
                    lo[v] = lo[v].min(p[1]);
                    hi[v] = hi[v].max(p[1]);
                }
                lo[*axis] = *slab_min;
                hi[*axis] = *slab_max;
                (lo, hi)
            }
        }
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        match self {
            Shape::Box { min, max } | Shape::IndexGrid { min, max, .. } => {
                (0..3).all(|a| p[a] >= min[a] && p[a] <= max[a])
            }
            Shape::Sphere { center, radius } => {
                let d2: f64 = (0..3).map(|a| (p[a] - center[a]).powi(2)).sum();
                d2 <= radius * radius
            }
            Shape::Cylinder { center, radius, length, axis } => {
                let (u, v) = plane_axes(*axis);
                let r2 = (p[u] - center[u]).powi(2) + (p[v] - center[v]).powi(2);
                r2 <= radius * radius && (p[*axis] - center[*axis]).abs() <= length / 2.0
            }
            Shape::PolySlab { vertices, axis, slab_min, slab_max } => {
                if p[*axis] < *slab_min || p[*axis] > *slab_max {
                    return false;
                }
                let (u, v) = plane_axes(*axis);
                point_in_polygon(vertices, p[u], p[v])
            }
        }
    }

    /// Outward normal estimate at point `p` (used for anisotropic sub-pixel
    /// smoothing when the shape has an analytic normal).
    pub fn normal_hint(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        match self {
            Shape::Sphere { center, .. } => {
                let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
                normalize(d)
            }
            Shape::Cylinder { center, radius, length, axis } => {
                let (u, v) = plane_axes(*axis);
                let mut d = [0.0; 3];
                d[u] = p[u] - center[u];
                d[v] = p[v] - center[v];
                let rr = (d[u] * d[u] + d[v] * d[v]).sqrt();
                let dz = (p[*axis] - center[*axis]).abs();
                // closer to the cap or to the side?
                if (length / 2.0 - dz).abs() < (radius - rr).abs() {
                    let mut n = [0.0; 3];
                    n[*axis] = (p[*axis] - center[*axis]).signum();
                    Some(n)
                } else {
                    normalize(d)
                }
            }
            _ => None,
        }
    }
}

impl Shape {
    /// Check the sample arrays of an [`Shape::IndexGrid`] (no-op for other shapes).
    pub fn validate(&self) -> Result<(), String> {
        if let Shape::IndexGrid { min, max, dims, eps, eps_axes, mu } = self {
            let n = dims[0] * dims[1] * dims[2];
            if n == 0 {
                return Err(format!("index_grid dims {dims:?} must all be >= 1"));
            }
            if (0..3).any(|a| !(max[a] > min[a])) {
                return Err("index_grid needs max > min along every axis".into());
            }
            let mut arrays: Vec<&Vec<f64>> = match eps_axes {
                Some(ax) => ax.iter().collect(),
                None if eps.is_empty() && !mu.is_empty() => vec![],
                None => vec![eps],
            };
            if !mu.is_empty() {
                arrays.push(mu);
            }
            for v in arrays {
                if v.len() != n {
                    return Err(format!("index_grid has {} samples, dims {dims:?} need {n}", v.len()));
                }
                if v.iter().any(|e| !e.is_finite() || *e <= 0.0) {
                    return Err("index_grid permittivity / permeability samples must be finite and > 0".into());
                }
            }
        }
        Ok(())
    }

    /// Relative permittivity of an [`Shape::IndexGrid`] along `axis` at point
    /// `p` (trilinear interpolation, clamped to the box). `None` for other
    /// shapes. Does not test whether `p` is inside the box.
    pub fn grid_eps(&self, axis: usize, p: [f64; 3]) -> Option<f64> {
        let Shape::IndexGrid { eps, eps_axes, .. } = self else {
            return None;
        };
        let data: &[f64] = match eps_axes {
            Some(ax) => &ax[axis],
            None => eps,
        };
        let (idx, wt, n) = self.grid_stencil(p)?;
        let mut v = 0.0;
        for c in 0..n {
            v += wt[c] * data[idx[c]];
        }
        Some(v)
    }

    /// True if this is an [`Shape::IndexGrid`] carrying permittivity samples
    /// (`for_h = false`) or permeability samples (`for_h = true`).
    pub fn grid_carries(&self, for_h: bool) -> bool {
        match self {
            Shape::IndexGrid { eps, eps_axes, mu, .. } => {
                if for_h { !mu.is_empty() } else { !eps.is_empty() || eps_axes.is_some() }
            }
            _ => false,
        }
    }

    /// Relative permeability of a permeability grid at point `p` (trilinear
    /// interpolation through [`grid_stencil`](Self::grid_stencil)).
    pub fn grid_mu(&self, p: [f64; 3]) -> Option<f64> {
        let Shape::IndexGrid { mu, .. } = self else {
            return None;
        };
        if mu.is_empty() {
            return None;
        }
        let (idx, wt, n) = self.grid_stencil(p)?;
        let mut v = 0.0;
        for c in 0..n {
            v += wt[c] * mu[idx[c]];
        }
        Some(v)
    }

    /// Trilinear sampling stencil of an [`Shape::IndexGrid`] at point `p`:
    /// `(sample indices, weights, count)` with `grid_eps = sum_c w[c] *
    /// eps[idx[c]]` over the first `count` entries (corners of zero weight
    /// are left out). [`grid_eps`](Self::grid_eps) is evaluated through this
    /// stencil, so scattering a per-voxel gradient with the same weights is
    /// the exact transpose of the rasterisation (see `crate::autodiff`).
    pub fn grid_stencil(&self, p: [f64; 3]) -> Option<([usize; 8], [f64; 8], usize)> {
        let Shape::IndexGrid { min, max, dims, .. } = self else {
            return None;
        };
        // lower sample index and weight of the upper sample per axis
        let mut i0 = [0usize; 3];
        let mut w = [0f64; 3];
        for a in 0..3 {
            if dims[a] > 1 {
                let t = ((p[a] - min[a]) / (max[a] - min[a])).clamp(0.0, 1.0) * (dims[a] - 1) as f64;
                let f = (t.floor() as usize).min(dims[a] - 2);
                i0[a] = f;
                w[a] = t - f as f64;
            }
        }
        let mut out_i = [0usize; 8];
        let mut out_w = [0f64; 8];
        let mut n = 0;
        for c in 0..8usize {
            let mut wt = 1.0;
            let mut idx = [0usize; 3];
            for a in 0..3 {
                let up = (c >> a) & 1 == 1;
                wt *= if up { w[a] } else { 1.0 - w[a] };
                idx[a] = if up { (i0[a] + 1).min(dims[a] - 1) } else { i0[a] };
            }
            if wt != 0.0 {
                out_i[n] = (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2];
                out_w[n] = wt;
                n += 1;
            }
        }
        Some((out_i, out_w, n))
    }
}

fn normalize(d: [f64; 3]) -> Option<[f64; 3]> {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n > 0.0 { Some([d[0] / n, d[1] / n, d[2] / n]) } else { None }
}

/// Even-odd point-in-polygon test.
pub fn point_in_polygon(vs: &[[f64; 2]], x: f64, y: f64) -> bool {
    let n = vs.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (vs[i][0], vs[i][1]);
        let (xj, yj) = (vs[j][0], vs[j][1]);
        if (yi > y) != (yj > y) {
            let xc = xi + (y - yi) * (xj - xi) / (yj - yi);
            if x < xc {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Sorted x-crossings of the polygon boundary with the horizontal line `y`.
/// Inside intervals are `[c0, c1], [c2, c3], ...` (even-odd rule).
pub fn row_crossings(vs: &[[f64; 2]], y: f64, out: &mut Vec<f64>) {
    out.clear();
    let n = vs.len();
    if n < 3 {
        return;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (vs[i][0], vs[i][1]);
        let (xj, yj) = (vs[j][0], vs[j][1]);
        if (yi > y) != (yj > y) {
            out.push(xi + (y - yi) * (xj - xi) / (yj - yi));
        }
        j = i;
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
}

/// Length of `[a, b]` covered by the inside intervals of `crossings`.
pub fn covered_length(crossings: &[f64], a: f64, b: f64) -> f64 {
    let mut s = 0.0;
    let mut t = 0;
    while t + 1 < crossings.len() {
        let lo = crossings[t].max(a);
        let hi = crossings[t + 1].min(b);
        if hi > lo {
            s += hi - lo;
        }
        t += 2;
    }
    s
}

/// A shape filled with a medium (index into the simulation's media list).
/// Later structures take precedence over earlier ones.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Structure {
    #[serde(default)]
    pub name: String,
    pub shape: Shape,
    /// Medium name (resolved against the simulation media table or the library).
    pub medium: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_tests() {
        let sq = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon(&sq, 0.5, 0.5));
        assert!(!point_in_polygon(&sq, 1.5, 0.5));
        let mut c = vec![];
        row_crossings(&sq, 0.5, &mut c);
        assert_eq!(c.len(), 2);
        assert!((covered_length(&c, -1.0, 0.25) - 0.25).abs() < 1e-12);
        let s = Shape::PolySlab { vertices: sq, axis: 2, slab_min: 0.0, slab_max: 0.2 };
        assert!(s.contains([0.5, 0.5, 0.1]));
        assert!(!s.contains([0.5, 0.5, 0.3]));
    }

    #[test]
    fn index_grid_interpolation() {
        // eps = 1 + x + 10 y, uniform in z (dims 1)
        let dims = [3, 2, 1];
        let mut eps = vec![];
        for i in 0..3 {
            for j in 0..2 {
                eps.push(1.0 + i as f64 + 10.0 * j as f64);
            }
        }
        let s = Shape::IndexGrid { min: [0.0, 0.0, -1.0], max: [2.0, 1.0, 1.0], dims, eps, eps_axes: None, mu: vec![] };
        s.validate().unwrap();
        assert_eq!(s.bbox(), ([0.0, 0.0, -1.0], [2.0, 1.0, 1.0]));
        assert!(s.contains([1.0, 0.5, 0.0]) && !s.contains([2.1, 0.5, 0.0]));
        for p in [[0.0, 0.0, 0.0], [2.0, 1.0, 0.7], [0.3, 0.25, -0.4], [1.7, 0.9, 0.0]] {
            let want = 1.0 + p[0] + 10.0 * p[1];
            assert!((s.grid_eps(1, p).unwrap() - want).abs() < 1e-12);
        }
        // serde round trip, tagged "index_grid"
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains(r#""type":"index_grid""#));
        assert_eq!(serde_json::from_str::<Shape>(&j).unwrap(), s);
        // wrong sample count is rejected
        let bad = Shape::IndexGrid { min: [0.0; 3], max: [1.0; 3], dims: [2, 2, 2], eps: vec![1.0; 7], eps_axes: None, mu: vec![] };
        assert!(bad.validate().is_err());
    }
}
