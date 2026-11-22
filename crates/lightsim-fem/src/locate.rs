//! Point location in a triangle mesh (bucket grid) and evaluation of
//! Lagrange fields and their gradients at arbitrary points.

use crate::element::{Lagrange, barycentric, geometry, lagrange_ref_grads, lagrange_values};
use crate::mesh::Mesh;
use rayon::prelude::*;

/// Bucket grid over the triangles of a mesh.
pub struct Locator<'a> {
    mesh: &'a Mesh,
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
}

impl<'a> Locator<'a> {
    pub fn new(mesh: &'a Mesh) -> Locator<'a> {
        let mut bb = [f64::INFINITY, f64::INFINITY, -f64::INFINITY, -f64::INFINITY];
        for p in &mesh.nodes {
            bb = [
                bb[0].min(p[0]),
                bb[1].min(p[1]),
                bb[2].max(p[0]),
                bb[3].max(p[1]),
            ];
        }
        let nt = mesh.n_tris().max(1);
        let area = ((bb[2] - bb[0]) * (bb[3] - bb[1])).max(1e-300);
        let cell = (2.0 * area / nt as f64).sqrt().max(1e-300);
        let nx = (((bb[2] - bb[0]) / cell).ceil() as usize).clamp(1, 4096);
        let ny = (((bb[3] - bb[1]) / cell).ceil() as usize).clamp(1, 4096);
        let cell = ((bb[2] - bb[0]) / nx as f64)
            .max((bb[3] - bb[1]) / ny as f64)
            .max(1e-300);
        let mut buckets = vec![vec![]; nx * ny];
        for t in 0..mesh.n_tris() {
            let tri = mesh.tris[t];
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [-f64::INFINITY; 2]);
            for &v in &tri {
                for a in 0..2 {
                    lo[a] = lo[a].min(mesh.nodes[v][a]);
                    hi[a] = hi[a].max(mesh.nodes[v][a]);
                }
            }
            let ci = |x: f64, o: f64, n: usize| {
                (((x - o) / cell).floor() as isize).clamp(0, n as isize - 1) as usize
            };
            for j in ci(lo[1], bb[1], ny)..=ci(hi[1], bb[1], ny) {
                for i in ci(lo[0], bb[0], nx)..=ci(hi[0], bb[0], nx) {
                    buckets[j * nx + i].push(t);
                }
            }
        }
        Locator {
            mesh,
            x0: bb[0],
            y0: bb[1],
            cell,
            nx,
            ny,
            buckets,
        }
    }

    /// Triangle containing `p` and its barycentric coordinates (affine).
    pub fn locate(&self, p: [f64; 2]) -> Option<(usize, [f64; 3])> {
        let i = ((p[0] - self.x0) / self.cell).floor();
        let j = ((p[1] - self.y0) / self.cell).floor();
        if i < -1.0 || j < -1.0 || i > self.nx as f64 || j > self.ny as f64 {
            return None;
        }
        let i = (i as isize).clamp(0, self.nx as isize - 1) as usize;
        let j = (j as isize).clamp(0, self.ny as isize - 1) as usize;
        let mut best: Option<(usize, [f64; 3], f64)> = None;
        for &t in &self.buckets[j * self.nx + i] {
            let l = barycentric(self.mesh, t, p);
            let m = l[0].min(l[1]).min(l[2]);
            if m >= -1e-10 {
                return Some((t, l));
            }
            if best.is_none_or(|b| m > b.2) {
                best = Some((t, l, m));
            }
        }
        best.filter(|b| b.2 > -1e-6).map(|b| (b.0, b.1))
    }
}

/// Values and gradients of a Lagrange field at points; NaN outside the mesh.
pub fn eval_lagrange(
    mesh: &Mesh,
    space: &Lagrange,
    u: &[f64],
    points: &[[f64; 2]],
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let loc = Locator::new(mesh);
    let res: Vec<(f64, [f64; 2])> = points
        .par_iter()
        .map(|&p| match loc.locate(p) {
            None => (f64::NAN, [f64::NAN; 2]),
            Some((t, l)) => {
                let l = [l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)];
                let g = geometry(mesh, t, &l);
                let phi = lagrange_values(space.order, &l);
                let rg = lagrange_ref_grads(space.order, &l);
                let (mut v, mut gr) = (0.0, [0.0; 2]);
                for i in 0..space.nloc() {
                    let c = u[space.dofs[t][i]];
                    v += c * phi[i];
                    let gi = g.grad(&rg[i]);
                    gr[0] += c * gi[0];
                    gr[1] += c * gi[1];
                }
                (v, gr)
            }
        })
        .collect();
    (
        res.iter().map(|r| r.0).collect(),
        res.iter().map(|r| r.1).collect(),
    )
}
