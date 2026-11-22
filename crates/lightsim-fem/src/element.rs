//! Finite elements on triangles: Lagrange P1 and P2 (isoparametric on curved
//! edges) and first-order Nedelec (Whitney) edge elements.
//!
//! Reference triangle (0,0), (1,0), (0,1) with barycentric coordinates
//! L0 = 1 - xi - eta, L1 = xi, L2 = eta. Local edge k joins local nodes k and
//! (k+1) % 3; P2 midside DOF k sits on local edge k.

use crate::mesh::Mesh;

/// Gradients of the barycentric coordinates in reference coordinates.
const DL: [[f64; 2]; 3] = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];

/// Lagrange shape values (P1: 3, P2: 6) at barycentric point `l`.
pub fn lagrange_values(order: usize, l: &[f64; 3]) -> [f64; 6] {
    let mut v = [0.0; 6];
    if order == 1 {
        v[..3].copy_from_slice(l);
    } else {
        for i in 0..3 {
            v[i] = l[i] * (2.0 * l[i] - 1.0);
            v[3 + i] = 4.0 * l[i] * l[(i + 1) % 3];
        }
    }
    v
}

/// Lagrange shape gradients in reference coordinates.
pub fn lagrange_ref_grads(order: usize, l: &[f64; 3]) -> [[f64; 2]; 6] {
    let mut g = [[0.0; 2]; 6];
    if order == 1 {
        g[..3].copy_from_slice(&DL);
    } else {
        for i in 0..3 {
            let s = 4.0 * l[i] - 1.0;
            g[i] = [s * DL[i][0], s * DL[i][1]];
            let j = (i + 1) % 3;
            g[3 + i] = [
                4.0 * (l[j] * DL[i][0] + l[i] * DL[j][0]),
                4.0 * (l[j] * DL[i][1] + l[i] * DL[j][1]),
            ];
        }
    }
    g
}

/// Geometry of one element at one reference point.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    /// Physical position.
    pub x: [f64; 2],
    /// |det J| (twice the area for an affine element).
    pub det: f64,
    /// J^{-T}: maps reference gradients to physical gradients.
    pub jit: [[f64; 2]; 2],
}

impl Geo {
    pub fn grad(&self, g: &[f64; 2]) -> [f64; 2] {
        [
            self.jit[0][0] * g[0] + self.jit[0][1] * g[1],
            self.jit[1][0] * g[0] + self.jit[1][1] * g[1],
        ]
    }
}

/// Geometry map of triangle `t` at barycentric point `l` (isoparametric P2
/// if the triangle has a curved edge, affine otherwise).
pub fn geometry(mesh: &Mesh, t: usize, l: &[f64; 3]) -> Geo {
    let (pts, grads, n): ([[f64; 2]; 6], [[f64; 2]; 6], usize) = if mesh.is_curved(t) {
        (mesh.p2_geometry(t), lagrange_ref_grads(2, l), 6)
    } else {
        let tri = mesh.tris[t];
        let mut p = [[0.0; 2]; 6];
        for i in 0..3 {
            p[i] = mesh.nodes[tri[i]];
        }
        (p, lagrange_ref_grads(1, l), 3)
    };
    let vals = lagrange_values(if n == 6 { 2 } else { 1 }, l);
    let mut x = [0.0; 2];
    let mut j = [[0.0; 2]; 2]; // j[a][b] = d x_a / d xi_b
    for i in 0..n {
        for a in 0..2 {
            x[a] += vals[i] * pts[i][a];
            for b in 0..2 {
                j[a][b] += pts[i][a] * grads[i][b];
            }
        }
    }
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    // J^{-1} = 1/det [[j11, -j01], [-j10, j00]]; J^{-T} = transpose
    let jit = [
        [j[1][1] / det, -j[1][0] / det],
        [-j[0][1] / det, j[0][0] / det],
    ];
    Geo {
        x,
        det: det.abs(),
        jit,
    }
}

/// A Lagrange space of order 1 or 2 on a mesh.
#[derive(Clone, Debug)]
pub struct Lagrange {
    pub order: usize,
    pub ndof: usize,
    /// Global DOFs of every triangle (first 3 or 6 entries used).
    pub dofs: Vec<[usize; 6]>,
}

impl Lagrange {
    pub fn new(mesh: &Mesh, order: usize) -> Lagrange {
        assert!(order == 1 || order == 2, "Lagrange order must be 1 or 2");
        let nn = mesh.n_nodes();
        let dofs = (0..mesh.n_tris())
            .map(|t| {
                let tri = mesh.tris[t];
                let mut d = [0usize; 6];
                d[..3].copy_from_slice(&tri);
                if order == 2 {
                    for k in 0..3 {
                        d[3 + k] = nn + mesh.tri_edges[t][k];
                    }
                }
                d
            })
            .collect();
        let ndof = if order == 1 { nn } else { nn + mesh.n_edges() };
        Lagrange { order, ndof, dofs }
    }
    pub fn nloc(&self) -> usize {
        if self.order == 1 { 3 } else { 6 }
    }
    /// Coordinates of every DOF.
    pub fn dof_coords(&self, mesh: &Mesh) -> Vec<[f64; 2]> {
        let mut c = mesh.nodes.clone();
        if self.order == 2 {
            for (e, &[a, b]) in mesh.edges.iter().enumerate() {
                c.push(mesh.edge_mid[e].unwrap_or([
                    0.5 * (mesh.nodes[a][0] + mesh.nodes[b][0]),
                    0.5 * (mesh.nodes[a][1] + mesh.nodes[b][1]),
                ]));
            }
        }
        c
    }
    /// DOFs on a set of edges (both end nodes, plus the midside DOF for P2).
    pub fn edge_dofs(&self, mesh: &Mesh, edges: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(edges.len() * 3);
        for &e in edges {
            out.extend_from_slice(&mesh.edges[e]);
            if self.order == 2 {
                out.push(mesh.n_nodes() + e);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    /// DOFs of all triangles with a tag in `tags` (closed: boundary DOFs included).
    pub fn region_dofs(&self, mesh: &Mesh, tags: &[u32]) -> Vec<usize> {
        let mut out = vec![];
        for t in 0..mesh.n_tris() {
            if tags.contains(&mesh.tags[t]) {
                out.extend_from_slice(&self.dofs[t][..self.nloc()]);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Values of the Whitney edge functions of local edges 0..3 at `l` (without
/// the global orientation sign), given the physical barycentric gradients.
pub fn whitney(l: &[f64; 3], gl: &[[f64; 2]; 3]) -> [[f64; 2]; 3] {
    let mut w = [[0.0; 2]; 3];
    for k in 0..3 {
        let (i, j) = (k, (k + 1) % 3);
        w[k] = [
            l[i] * gl[j][0] - l[j] * gl[i][0],
            l[i] * gl[j][1] - l[j] * gl[i][1],
        ];
    }
    w
}

/// Curls (z component) of the Whitney edge functions: 2 grad Li x grad Lj.
pub fn whitney_curl(gl: &[[f64; 2]; 3]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for k in 0..3 {
        let (i, j) = (k, (k + 1) % 3);
        c[k] = 2.0 * (gl[i][0] * gl[j][1] - gl[i][1] * gl[j][0]);
    }
    c
}

/// Physical gradients of the barycentric coordinates of an affine triangle.
pub fn bary_grads(mesh: &Mesh, t: usize) -> [[f64; 2]; 3] {
    let tri = mesh.tris[t];
    let (a, b, c) = (mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]);
    let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    [
        [(b[1] - c[1]) / det, (c[0] - b[0]) / det],
        [(c[1] - a[1]) / det, (a[0] - c[0]) / det],
        [(a[1] - b[1]) / det, (b[0] - a[0]) / det],
    ]
}

/// Barycentric coordinates of point `p` in the affine triangle `t`.
pub fn barycentric(mesh: &Mesh, t: usize, p: [f64; 2]) -> [f64; 3] {
    let tri = mesh.tris[t];
    let (a, b, c) = (mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]);
    let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det;
    let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det;
    [1.0 - l1 - l2, l1, l2]
}
