//! Triangle meshes: nodes, counter-clockwise triangles with region tags, and
//! the derived edge structure (global edges, element-to-edge map, boundary
//! and interface edges, curved edge midpoints for isoparametric P2).

use anyhow::{Result, bail};
use std::collections::HashMap;

/// A conforming triangle mesh of a 2-D cross-section.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// Node coordinates (m or any consistent unit).
    pub nodes: Vec<[f64; 2]>,
    /// Triangles, counter-clockwise.
    pub tris: Vec<[usize; 3]>,
    /// Region tag of every triangle.
    pub tags: Vec<u32>,
    /// Global edges `[lo, hi]` with `lo < hi`.
    pub edges: Vec<[usize; 2]>,
    /// Local edge k of a triangle joins its local nodes k and (k+1)%3.
    pub tri_edges: Vec<[usize; 3]>,
    /// The (one or two) triangles on each edge.
    pub edge_tris: Vec<[usize; 2]>,
    /// Point on the true boundary where the midpoint of a curved edge lies
    /// (isoparametric P2 geometry); `None` for a straight edge.
    pub edge_mid: Vec<Option<[f64; 2]>>,
}

pub const NONE: usize = usize::MAX;

impl Mesh {
    /// Build a mesh, orient triangles counter-clockwise and derive the edges.
    pub fn new(nodes: Vec<[f64; 2]>, mut tris: Vec<[usize; 3]>, tags: Vec<u32>) -> Result<Mesh> {
        if tags.len() != tris.len() {
            bail!("{} triangles but {} tags", tris.len(), tags.len());
        }
        for t in tris.iter_mut() {
            for &v in t.iter() {
                if v >= nodes.len() {
                    bail!(
                        "triangle refers to node {v}, the mesh has {} nodes",
                        nodes.len()
                    );
                }
            }
            let a = signed_area(&nodes, t);
            if a < 0.0 {
                t.swap(1, 2);
            } else if a == 0.0 {
                bail!("degenerate triangle {:?}", t);
            }
        }
        let mut m = Mesh {
            nodes,
            tris,
            tags,
            ..Default::default()
        };
        m.build_edges();
        Ok(m)
    }

    fn build_edges(&mut self) {
        let mut map: HashMap<(usize, usize), usize> = HashMap::with_capacity(self.tris.len() * 2);
        self.edges.clear();
        self.edge_tris.clear();
        self.tri_edges = vec![[0; 3]; self.tris.len()];
        for (ti, t) in self.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                let e = *map.entry(key).or_insert_with(|| {
                    self.edges.push([key.0, key.1]);
                    self.edge_tris.push([NONE, NONE]);
                    self.edges.len() - 1
                });
                if self.edge_tris[e][0] == NONE {
                    self.edge_tris[e][0] = ti;
                } else {
                    self.edge_tris[e][1] = ti;
                }
                self.tri_edges[ti][k] = e;
            }
        }
        self.edge_mid = vec![None; self.edges.len()];
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn n_tris(&self) -> usize {
        self.tris.len()
    }
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// True for an edge with a single triangle.
    pub fn is_boundary_edge(&self, e: usize) -> bool {
        self.edge_tris[e][1] == NONE
    }

    /// Edges on the outer boundary.
    pub fn boundary_edges(&self) -> Vec<usize> {
        (0..self.n_edges())
            .filter(|&e| self.is_boundary_edge(e))
            .collect()
    }

    /// Edges between triangles of different tags.
    pub fn interface_edges(&self) -> Vec<usize> {
        (0..self.n_edges())
            .filter(|&e| {
                let [a, b] = self.edge_tris[e];
                b != NONE && self.tags[a] != self.tags[b]
            })
            .collect()
    }

    pub fn area(&self, t: usize) -> f64 {
        signed_area(&self.nodes, &self.tris[t])
    }

    /// Orientation sign of local edge k of triangle t relative to the global
    /// edge (global edges point from the lower to the higher node index).
    pub fn edge_sign(&self, t: usize, k: usize) -> f64 {
        let tri = self.tris[t];
        if tri[k] < tri[(k + 1) % 3] { 1.0 } else { -1.0 }
    }

    /// The six P2 geometry nodes of a triangle: vertices then midpoints of
    /// local edges 0, 1, 2 (curved midpoints where set).
    pub fn p2_geometry(&self, t: usize) -> [[f64; 2]; 6] {
        let tri = self.tris[t];
        let p = |i: usize| self.nodes[tri[i]];
        let mut g = [p(0), p(1), p(2), [0.0; 2], [0.0; 2], [0.0; 2]];
        for k in 0..3 {
            let e = self.tri_edges[t][k];
            g[3 + k] = match self.edge_mid[e] {
                Some(m) => m,
                None => {
                    let (a, b) = (p(k), p((k + 1) % 3));
                    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
                }
            };
        }
        g
    }

    /// True if any edge of the triangle is curved.
    pub fn is_curved(&self, t: usize) -> bool {
        self.tri_edges[t]
            .iter()
            .any(|&e| self.edge_mid[e].is_some())
    }

    /// Longest edge length of a triangle.
    pub fn diameter(&self, t: usize) -> f64 {
        let tri = self.tris[t];
        (0..3)
            .map(|k| {
                let (a, b) = (self.nodes[tri[k]], self.nodes[tri[(k + 1) % 3]]);
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// Smallest interior angle of a triangle (radians).
    pub fn min_angle(&self, t: usize) -> f64 {
        let tri = self.tris[t];
        let p: Vec<[f64; 2]> = tri.iter().map(|&i| self.nodes[i]).collect();
        let mut m = f64::INFINITY;
        for k in 0..3 {
            let (a, b, c) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
            let u = [b[0] - a[0], b[1] - a[1]];
            let v = [c[0] - a[0], c[1] - a[1]];
            let cos = (u[0] * v[0] + u[1] * v[1])
                / ((u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt());
            m = m.min(cos.clamp(-1.0, 1.0).acos());
        }
        m
    }

    /// Uniform red refinement: every triangle split into four. Curved edge
    /// midpoints become nodes; the new curved midpoints are not known, so
    /// curvature is dropped (re-mesh from the geometry for curved P2).
    pub fn refine_uniform(&self) -> Mesh {
        let mut nodes = self.nodes.clone();
        let mut mid = vec![0usize; self.n_edges()];
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            let p = self.edge_mid[e].unwrap_or([
                0.5 * (self.nodes[a][0] + self.nodes[b][0]),
                0.5 * (self.nodes[a][1] + self.nodes[b][1]),
            ]);
            mid[e] = nodes.len();
            nodes.push(p);
        }
        let mut tris = Vec::with_capacity(4 * self.n_tris());
        let mut tags = Vec::with_capacity(4 * self.n_tris());
        for (t, tri) in self.tris.iter().enumerate() {
            let m = [
                mid[self.tri_edges[t][0]],
                mid[self.tri_edges[t][1]],
                mid[self.tri_edges[t][2]],
            ];
            for nt in [
                [tri[0], m[0], m[2]],
                [m[0], tri[1], m[1]],
                [m[2], m[1], tri[2]],
                [m[0], m[1], m[2]],
            ] {
                tris.push(nt);
                tags.push(self.tags[t]);
            }
        }
        Mesh::new(nodes, tris, tags).expect("refinement of a valid mesh")
    }
}

pub fn signed_area(nodes: &[[f64; 2]], t: &[usize; 3]) -> f64 {
    let (a, b, c) = (nodes[t[0]], nodes[t[1]], nodes[t[2]]);
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
}

/// A structured mesh of the rectangle [x0, x1] x [y0, y1], nx by ny cells,
/// each split into two triangles (alternating diagonals), tag 0.
pub fn rectangle(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Mesh {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            nodes.push([
                x0 + (x1 - x0) * i as f64 / nx as f64,
                y0 + (y1 - y0) * j as f64 / ny as f64,
            ]);
        }
    }
    let id = |i: usize, j: usize| j * (nx + 1) + i;
    let mut tris = vec![];
    for j in 0..ny {
        for i in 0..nx {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            if (i + j) % 2 == 0 {
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            } else {
                tris.push([a, b, d]);
                tris.push([b, c, d]);
            }
        }
    }
    let n = tris.len();
    Mesh::new(nodes, tris, vec![0; n]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangle_edges() {
        let m = rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);
        assert_eq!(m.n_tris(), 12);
        // Euler: V - E + F = 1
        assert_eq!(
            m.n_nodes() as i64 - m.n_edges() as i64 + m.n_tris() as i64,
            1
        );
        assert_eq!(m.boundary_edges().len(), 10);
        let r = m.refine_uniform();
        assert_eq!(r.n_tris(), 48);
        let area: f64 = (0..r.n_tris()).map(|t| r.area(t)).sum();
        assert!((area - 1.0).abs() < 1e-14);
    }
}
