//! Full-vectorial waveguide modes with mixed Nedelec (edge) / Lagrange P1
//! elements.
//!
//! Fields `E(x, y) exp(i (beta z - omega t))`. With `E_t` on first-order
//! Nedelec elements and the scaled longitudinal unknown
//! `zeta = i beta E_z / k0^2` on P1, the transverse curl-curl equation and
//! Gauss's law give the generalised eigenproblem `A x = beta^2 B x`
//! (Vardapetyan & Demkowicz 2002, the formulation of femwell):
//!
//! ```text
//! A_tt = (1/k0^2) (1/mu_zz) curl e . curl v - (eps_t e) . v
//! A_tz = (mut^-1 grad zeta) . v          mut = diag(mu_yy, mu_xx)
//! A_zt = (eps_t e) . grad g
//! A_zz = -k0^2 eps_zz zeta g
//! B_tt = -(1/k0^2) (mut^-1 e) . v
//! ```
//!
//! `eps_t` is a full in-plane tensor (anisotropic, lossy, metals with
//! complex permittivity). Perfectly matched layers are complex coordinate
//! stretches `s = 1 + i S (d / L)^2` applied as the tensors
//! `eps Lambda`, `mu Lambda` with `Lambda = diag(s_y/s_x, s_x/s_y, s_x s_y)`;
//! a bend of radius R is the conformal map `Lambda_bend = diag(r, r, 1/r)`,
//! `r = 1 + (x - x_axis) / R` (the bend centre lies at `x_axis - R`).
//! Boundaries are perfect electric conductors on the listed edges and
//! perfect magnetic conductors (natural) elsewhere.

use crate::eigen::{EigOptions, eigs_shift_invert};
use crate::element::{bary_grads, whitney, whitney_curl};
use crate::mesh::Mesh;
use crate::quadrature;
use crate::sparse::{Csr, Triplets};
use anyhow::{Result, bail};
use num_complex::Complex64 as C;
use rayon::prelude::*;

/// Perfectly matched layers on the sides of a rectangular inner region.
#[derive(Clone, Debug)]
pub struct Pml {
    /// Inner (non-absorbing) box [x0, y0, x1, y1].
    pub inner: [f64; 4],
    /// Layer thicknesses [left, bottom, right, top] (0: no layer on that side).
    pub thickness: [f64; 4],
    /// Imaginary part of the stretch at the outer edge.
    pub strength: f64,
}

impl Pml {
    /// Stretch factors (s_x, s_y) at a point.
    pub fn stretch(&self, p: [f64; 2]) -> (C, C) {
        let s = |d: f64, l: f64| {
            if d > 0.0 && l > 0.0 {
                C::new(1.0, self.strength * (d / l).min(1.0).powi(2))
            } else {
                C::new(1.0, 0.0)
            }
        };
        let [x0, y0, x1, y1] = self.inner;
        let [tl, tb, tr, tt] = self.thickness;
        let sx = if p[0] < x0 {
            s(x0 - p[0], tl)
        } else if p[0] > x1 {
            s(p[0] - x1, tr)
        } else {
            C::new(1.0, 0.0)
        };
        let sy = if p[1] < y0 {
            s(y0 - p[1], tb)
        } else if p[1] > y1 {
            s(p[1] - y1, tt)
        } else {
            C::new(1.0, 0.0)
        };
        (sx, sy)
    }
    pub fn inside(&self, p: [f64; 2]) -> bool {
        let [x0, y0, x1, y1] = self.inner;
        p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1
    }
}

/// A waveguide mode problem.
#[derive(Clone, Debug)]
pub struct ModeProblem {
    /// Vacuum wavenumber (1/m).
    pub k0: f64,
    /// Relative permittivity per triangle: (eps_xx, eps_yy, eps_xy, eps_zz).
    pub eps: Vec<[C; 4]>,
    pub pml: Option<Pml>,
    /// Bend: (radius, x of the waveguide axis).
    pub bend: Option<(f64, f64)>,
    /// Boundary edges held as perfect electric conductors.
    pub pec_edges: Vec<usize>,
}

/// Material tensors at a point: eps_t (xx, yy, xy), eps_zz, mut^-1 (diag), 1/mu_zz.
struct Mat {
    et: [C; 3],
    ezz: C,
    mut_inv: [C; 2],
    muzz_inv: C,
}

impl ModeProblem {
    fn material(&self, t: usize, p: [f64; 2]) -> Mat {
        let [exx, eyy, exy, ezz] = self.eps[t];
        let (sx, sy) = self
            .pml
            .as_ref()
            .map_or((C::new(1.0, 0.0), C::new(1.0, 0.0)), |m| m.stretch(p));
        let r = self.bend.map_or(1.0, |(rad, xa)| 1.0 + (p[0] - xa) / rad);
        // Lambda = diag(lx, ly, lz)
        let lx = sy / sx * r;
        let ly = sx / sy * r;
        let lz = sx * sy / r;
        // eps Lambda for a diagonal stretch: eps_ij * sqrt(l_i l_j) keeps the tensor symmetric
        let exy_s = exy * (lx * ly).sqrt();
        Mat {
            et: [exx * lx, eyy * ly, exy_s],
            ezz: ezz * lz,
            // mu = Lambda (mu_r = 1); mut = diag(mu_yy, mu_xx) -> inverse diag(1/ly, 1/lx)
            mut_inv: [C::new(1.0, 0.0) / ly, C::new(1.0, 0.0) / lx],
            muzz_inv: C::new(1.0, 0.0) / lz,
        }
    }

    pub fn n_dofs(&self, mesh: &Mesh) -> usize {
        mesh.n_edges() + mesh.n_nodes()
    }

    /// Assemble A and B over all DOFs (edges first, then nodes).
    pub fn assemble(&self, mesh: &Mesh) -> Result<(Csr<C>, Csr<C>)> {
        if self.eps.len() != mesh.n_tris() {
            bail!(
                "eps has {} rows, the mesh {} triangles",
                self.eps.len(),
                mesh.n_tris()
            );
        }
        let ne = mesh.n_edges();
        let n = self.n_dofs(mesh);
        let k0 = self.k0;
        let k02 = k0 * k0;
        let deg = if self.pml.is_some() || self.bend.is_some() {
            4
        } else {
            2
        };
        let rule = quadrature::triangle(deg);
        let parts: Vec<(Triplets<C>, Triplets<C>)> = (0..mesh.n_tris())
            .into_par_iter()
            .fold(
                || (Triplets::with_capacity(0), Triplets::with_capacity(0)),
                |(mut ta, mut tb), t| {
                    let gl = bary_grads(mesh, t);
                    let area = mesh.area(t);
                    let tri = mesh.tris[t];
                    let sg: [f64; 3] = [
                        mesh.edge_sign(t, 0),
                        mesh.edge_sign(t, 1),
                        mesh.edge_sign(t, 2),
                    ];
                    let curl = whitney_curl(&gl);
                    let mut a = [[C::new(0.0, 0.0); 6]; 6]; // local: 3 edges, 3 nodes
                    let mut b = [[C::new(0.0, 0.0); 3]; 3];
                    for (l, w) in rule.points.iter().zip(&rule.weights) {
                        let p = [
                            l[0] * mesh.nodes[tri[0]][0]
                                + l[1] * mesh.nodes[tri[1]][0]
                                + l[2] * mesh.nodes[tri[2]][0],
                            l[0] * mesh.nodes[tri[0]][1]
                                + l[1] * mesh.nodes[tri[1]][1]
                                + l[2] * mesh.nodes[tri[2]][1],
                        ];
                        let m = self.material(t, p);
                        let dv = w * area;
                        let wf = whitney(l, &gl);
                        let ws: Vec<[f64; 2]> = (0..3)
                            .map(|k| [sg[k] * wf[k][0], sg[k] * wf[k][1]])
                            .collect();
                        let cs: Vec<f64> = (0..3).map(|k| sg[k] * curl[k]).collect();
                        let et = |v: [f64; 2]| -> [C; 2] {
                            [
                                m.et[0] * v[0] + m.et[2] * v[1],
                                m.et[2] * v[0] + m.et[1] * v[1],
                            ]
                        };
                        for k in 0..3 {
                            for q in 0..3 {
                                let ew = et(ws[q]);
                                a[k][q] += (m.muzz_inv * (cs[k] * cs[q] / k02)
                                    - (ew[0] * ws[k][0] + ew[1] * ws[k][1]))
                                    * dv;
                                b[k][q] += -(m.mut_inv[0] * ws[q][0] * ws[k][0]
                                    + m.mut_inv[1] * ws[q][1] * ws[k][1])
                                    * (dv / k02);
                            }
                            for j in 0..3 {
                                // A_tz: (mut^-1 grad phi_j) . w_k
                                a[k][3 + j] += (m.mut_inv[0] * gl[j][0] * ws[k][0]
                                    + m.mut_inv[1] * gl[j][1] * ws[k][1])
                                    * dv;
                                // A_zt: (eps_t w_k) . grad phi_j  (row j, column k)
                                let ew = et(ws[k]);
                                a[3 + j][k] += (ew[0] * gl[j][0] + ew[1] * gl[j][1]) * dv;
                            }
                        }
                        for i in 0..3 {
                            for j in 0..3 {
                                a[3 + i][3 + j] += -(m.ezz * (k02 * l[i] * l[j])) * dv;
                            }
                        }
                    }
                    let mut dofs = [0usize; 6];
                    for k in 0..3 {
                        dofs[k] = mesh.tri_edges[t][k];
                        dofs[3 + k] = ne + tri[k];
                    }
                    for i in 0..6 {
                        for j in 0..6 {
                            if a[i][j] != C::new(0.0, 0.0) {
                                ta.push(dofs[i], dofs[j], a[i][j]);
                            }
                        }
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            tb.push(dofs[i], dofs[j], b[i][j]);
                        }
                    }
                    (ta, tb)
                },
            )
            .collect();
        let mut ta = Triplets::with_capacity(36 * mesh.n_tris());
        let mut tb = Triplets::with_capacity(9 * mesh.n_tris());
        for (a, b) in parts {
            ta.extend(a);
            tb.extend(b);
        }
        Ok((ta.to_csr(n, n), tb.to_csr(n, n)))
    }

    /// DOFs fixed to zero by the PEC edges (their edge DOFs and node DOFs).
    pub fn fixed_dofs(&self, mesh: &Mesh) -> Vec<bool> {
        let ne = mesh.n_edges();
        let mut fixed = vec![false; self.n_dofs(mesh)];
        for &e in &self.pec_edges {
            fixed[e] = true;
            for &v in &mesh.edges[e] {
                fixed[ne + v] = true;
            }
        }
        fixed
    }

    /// Modes with beta^2 closest to (k0 n_guess)^2: (beta, full DOF vector).
    pub fn solve(
        &self,
        mesh: &Mesh,
        nev: usize,
        n_guess: f64,
        tol: f64,
    ) -> Result<Vec<(C, Vec<C>)>> {
        let (a, b) = self.assemble(mesh)?;
        let fixed = self.fixed_dofs(mesh);
        let n = fixed.len();
        let mut map = vec![None; n];
        let mut free = vec![];
        for i in 0..n {
            if !fixed[i] {
                map[i] = Some(free.len());
                free.push(i);
            }
        }
        let nf = free.len();
        let ar = a.submatrix(&map, nf, &map, nf);
        let br = b.submatrix(&map, nf, &map, nf);
        let sigma = C::new((self.k0 * n_guess).powi(2), 0.0);
        let opts = EigOptions {
            nev: nev.min(nf.saturating_sub(2)).max(1),
            tol,
            ..Default::default()
        };
        let res = eigs_shift_invert(&ar, &br, sigma, &opts)?;
        let mut out: Vec<(C, Vec<C>)> = res
            .values
            .iter()
            .zip(res.vectors)
            .map(|(lam, v)| {
                let mut beta = lam.sqrt();
                if beta.re < 0.0 {
                    beta = -beta;
                }
                let mut x = vec![C::new(0.0, 0.0); n];
                for (k, &i) in free.iter().enumerate() {
                    x[i] = v[k];
                }
                (beta, x)
            })
            .collect();
        out.sort_by(|a, b| b.0.re.partial_cmp(&a.0.re).unwrap());
        Ok(out)
    }

    /// E and H (A/m, with eta0 not applied: H here is the field for the
    /// given E amplitudes) at barycentric point `l` of triangle `t`.
    pub fn fields(
        &self,
        mesh: &Mesh,
        t: usize,
        l: &[f64; 3],
        beta: C,
        x: &[C],
    ) -> ([C; 3], [C; 3]) {
        let ne = mesh.n_edges();
        let gl = bary_grads(mesh, t);
        let tri = mesh.tris[t];
        let wf = whitney(l, &gl);
        let curl = whitney_curl(&gl);
        let k0 = self.k0;
        let (mut ex, mut ey, mut cz) = (C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(0.0, 0.0));
        let (mut z, mut zx, mut zy) = (C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(0.0, 0.0));
        for k in 0..3 {
            let c = x[mesh.tri_edges[t][k]] * mesh.edge_sign(t, k);
            ex += c * wf[k][0];
            ey += c * wf[k][1];
            cz += c * curl[k];
            let zn = x[ne + tri[k]];
            z += zn * l[k];
            zx += zn * gl[k][0];
            zy += zn * gl[k][1];
        }
        // E_z = -i k0^2 zeta / beta
        let f = C::new(0.0, -k0 * k0) / beta;
        let ez = f * z;
        let (ezx, ezy) = (f * zx, f * zy);
        let p = [
            l[0] * mesh.nodes[tri[0]][0]
                + l[1] * mesh.nodes[tri[1]][0]
                + l[2] * mesh.nodes[tri[2]][0],
            l[0] * mesh.nodes[tri[0]][1]
                + l[1] * mesh.nodes[tri[1]][1]
                + l[2] * mesh.nodes[tri[2]][1],
        ];
        let m = self.material(t, p);
        // i omega mu0 mu H = curl E; omega mu0 = k0 eta0 -> eta0 H = curl E / (i k0 mu)
        let i = C::new(0.0, 1.0);
        let ik0 = i * k0;
        let cx = ezy - i * beta * ey;
        let cy = i * beta * ex - ezx;
        // mu = Lambda: mu_xx = lx = 1 / mut_inv[1], mu_yy = ly = 1 / mut_inv[0]
        let hx = cx * m.mut_inv[1] / ik0;
        let hy = cy * m.mut_inv[0] / ik0;
        let hz = cz * m.muzz_inv / ik0;
        ([ex, ey, ez], [hx, hy, hz])
    }

    /// Integrals over the non-PML region with quadrature of degree 4:
    /// (P = 1/2 Re int (E x H*)_z with H = eta0 H, int |Ex|^2, int |Ey|^2,
    /// int |E|^2, int |E|^4, int (E* . Re(eps) E + |eta0 H|^2) / 4).
    pub fn integrals(&self, mesh: &Mesh, beta: C, x: &[C]) -> [f64; 6] {
        let rule = quadrature::triangle(4);
        (0..mesh.n_tris())
            .into_par_iter()
            .map(|t| {
                let mut acc = [0.0; 6];
                let tri = mesh.tris[t];
                let area = mesh.area(t);
                for (l, w) in rule.points.iter().zip(&rule.weights) {
                    let p = [
                        l[0] * mesh.nodes[tri[0]][0]
                            + l[1] * mesh.nodes[tri[1]][0]
                            + l[2] * mesh.nodes[tri[2]][0],
                        l[0] * mesh.nodes[tri[0]][1]
                            + l[1] * mesh.nodes[tri[1]][1]
                            + l[2] * mesh.nodes[tri[2]][1],
                    ];
                    if let Some(pml) = &self.pml {
                        if !pml.inside(p) {
                            continue;
                        }
                    }
                    let (e, h) = self.fields(mesh, t, l, beta, x);
                    let dv = w * area;
                    acc[0] += 0.5 * (e[0] * h[1].conj() - e[1] * h[0].conj()).re * dv;
                    acc[1] += e[0].norm_sqr() * dv;
                    acc[2] += e[1].norm_sqr() * dv;
                    let e2 = e[0].norm_sqr() + e[1].norm_sqr() + e[2].norm_sqr();
                    acc[3] += e2 * dv;
                    acc[4] += e2 * e2 * dv;
                    let [exx, eyy, exy, ezz] = self.eps[t];
                    let ee = (e[0].conj() * (exx * e[0] + exy * e[1])
                        + e[1].conj() * (exy * e[0] + eyy * e[1])
                        + e[2].conj() * ezz * e[2])
                        .re;
                    let hh = h[0].norm_sqr() + h[1].norm_sqr() + h[2].norm_sqr();
                    acc[5] += 0.25 * (ee + hh) * dv;
                }
                acc
            })
            .reduce(
                || [0.0; 6],
                |a, b| {
                    [
                        a[0] + b[0],
                        a[1] + b[1],
                        a[2] + b[2],
                        a[3] + b[3],
                        a[4] + b[4],
                        a[5] + b[5],
                    ]
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::{MeshOptions, mesh_regions, rect_region};

    /// TE0 of a symmetric slab between PEC side walls (x-uniform) against the
    /// exact dispersion relation.
    #[test]
    fn slab_te0() {
        let (n1, n2, d, lam) = (3.0f64, 1.5f64, 0.4e-6, 1.55e-6);
        let k0 = std::f64::consts::TAU / lam;
        let w = 0.2e-6;
        let regions = vec![
            rect_region(0.0, -1.5e-6, w, 1.5e-6, 0, 10, 0.0),
            rect_region(0.0, -d / 2.0, w, d / 2.0, 1, 1, 0.0),
        ];
        let (mesh, _) = mesh_regions(
            &regions,
            &MeshOptions {
                default_size: 0.02e-6,
                ..Default::default()
            },
        )
        .unwrap();
        let eps: Vec<[C; 4]> = mesh
            .tags
            .iter()
            .map(|&t| {
                let e = if t == 1 { n1 * n1 } else { n2 * n2 };
                [
                    C::new(e, 0.0),
                    C::new(e, 0.0),
                    C::new(0.0, 0.0),
                    C::new(e, 0.0),
                ]
            })
            .collect();
        // PEC on the side walls (x = 0, w) only; top and bottom far away (PMC)
        let pec: Vec<usize> = mesh
            .boundary_edges()
            .into_iter()
            .filter(|&e| {
                let [a, b] = mesh.edges[e];
                let xa = mesh.nodes[a][0];
                let xb = mesh.nodes[b][0];
                (xa.abs() < 1e-12 && xb.abs() < 1e-12)
                    || ((xa - w).abs() < 1e-12 && (xb - w).abs() < 1e-12)
            })
            .collect();
        let p = ModeProblem {
            k0,
            eps,
            pml: None,
            bend: None,
            pec_edges: pec,
        };
        let modes = p.solve(&mesh, 2, n1, 1e-10).unwrap();
        // exact TE0: tan(kx d/2) = gamma / kx
        let f = |ne: f64| {
            let kx = k0 * (n1 * n1 - ne * ne).sqrt();
            let g = k0 * (ne * ne - n2 * n2).sqrt();
            (kx * d / 2.0).tan() - g / kx
        };
        let (mut lo, mut hi) = (n2 + 1e-9, n1 - 1e-9);
        // bracket the first root
        let mut x = lo;
        let mut prev = f(x);
        let step = (hi - lo) / 2000.0;
        while x < hi {
            let y = f(x + step);
            if prev.signum() != y.signum() && (y - prev).abs() < 10.0 {
                lo = x;
                hi = x + step;
            }
            prev = y;
            x += step;
        }
        for _ in 0..100 {
            let m = 0.5 * (lo + hi);
            if f(lo).signum() == f(m).signum() {
                lo = m
            } else {
                hi = m
            }
        }
        let exact = 0.5 * (lo + hi);
        let ne = modes[0].0.re / k0;
        assert!((ne - exact).abs() < 2e-4, "neff {ne} vs exact {exact}");
        let ints = p.integrals(&mesh, modes[0].0, &modes[0].1);
        assert!(
            ints[1] / (ints[1] + ints[2]) > 0.999,
            "TE fraction {}",
            ints[1] / (ints[1] + ints[2])
        );
    }
}
