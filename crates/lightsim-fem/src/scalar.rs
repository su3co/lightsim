//! Second-order scalar problems on a triangle mesh with Lagrange P1 / P2:
//!
//! ```text
//! c du/dt - div(K grad u) = f          in the domain
//! u = g                                on Dirichlet edges / nodes / regions
//! n . K grad u = q                     on Neumann edges (q: inflow per length)
//! -n . K grad u = h (u - u_inf)        on convective (Robin) edges
//! ```
//!
//! `K` is a symmetric 2x2 tensor per triangle (anisotropic conductivity or
//! permittivity), `c` and `f` constants per triangle. Steady problems are
//! solved with a sparse Cholesky factorisation (or preconditioned CG);
//! transient ones with the theta scheme (backward Euler or Crank-Nicolson)
//! on one factorisation.

use crate::element::{Lagrange, geometry, lagrange_ref_grads, lagrange_values};
use crate::linsolve::{Direct, Precond, cg};
use crate::mesh::Mesh;
use crate::quadrature;
use crate::sparse::{Csr, Triplets};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// Coefficients and boundary conditions of a scalar problem.
#[derive(Clone, Debug, Default)]
pub struct ScalarProblem {
    /// Lagrange order (1 or 2).
    pub order: usize,
    /// Per triangle: (Kxx, Kyy, Kxy).
    pub k: Vec<[f64; 3]>,
    /// Per triangle: capacity c (transient) or reaction coefficient (steady).
    pub c: Vec<f64>,
    /// Per triangle: volumetric source f.
    pub f: Vec<f64>,
    /// Dirichlet DOFs and values (from [`Lagrange::edge_dofs`] or [`Lagrange::region_dofs`]).
    pub dirichlet: Vec<(usize, f64)>,
    /// Neumann edges and inflow q.
    pub neumann: Vec<(usize, f64)>,
    /// Robin edges, heat transfer coefficient h and ambient value u_inf.
    pub robin: Vec<(usize, f64, f64)>,
}

/// Assembled operators: stiffness (with Robin terms), capacity matrix, load.
pub struct Assembled {
    pub space: Lagrange,
    pub stiffness: Csr<f64>,
    pub capacity: Csr<f64>,
    pub load: Vec<f64>,
}

fn quad_degree(order: usize, curved: bool) -> usize {
    match (order, curved) {
        (1, false) => 2,
        (1, true) => 4,
        (_, false) => 4,
        (_, true) => 6,
    }
}

/// Integrate the stiffness K, the capacity c and the load f over every triangle.
pub fn assemble(mesh: &Mesh, p: &ScalarProblem) -> Result<Assembled> {
    let nt = mesh.n_tris();
    if p.k.len() != nt || p.c.len() != nt || p.f.len() != nt {
        bail!("coefficient arrays must have one entry per triangle ({nt})");
    }
    let space = Lagrange::new(mesh, p.order);
    let nl = space.nloc();
    let chunks: Vec<(Triplets<f64>, Triplets<f64>, Vec<(usize, f64)>)> = (0..nt)
        .into_par_iter()
        .fold(
            || {
                (
                    Triplets::with_capacity(0),
                    Triplets::with_capacity(0),
                    vec![],
                )
            },
            |(mut ks, mut cs, mut fs), t| {
                let curved = mesh.is_curved(t);
                let rule = quadrature::triangle(quad_degree(p.order, curved));
                let mut ke = [[0.0; 6]; 6];
                let mut ce = [[0.0; 6]; 6];
                let mut fe = [0.0; 6];
                let [kxx, kyy, kxy] = p.k[t];
                for (l, w) in rule.points.iter().zip(&rule.weights) {
                    let g = geometry(mesh, t, l);
                    let dv = w * 0.5 * g.det;
                    let phi = lagrange_values(p.order, l);
                    let rg = lagrange_ref_grads(p.order, l);
                    let mut gr = [[0.0; 2]; 6];
                    for i in 0..nl {
                        gr[i] = g.grad(&rg[i]);
                    }
                    for i in 0..nl {
                        let kg = [
                            kxx * gr[i][0] + kxy * gr[i][1],
                            kxy * gr[i][0] + kyy * gr[i][1],
                        ];
                        for j in 0..nl {
                            ke[i][j] += dv * (kg[0] * gr[j][0] + kg[1] * gr[j][1]);
                            ce[i][j] += dv * p.c[t] * phi[i] * phi[j];
                        }
                        fe[i] += dv * p.f[t] * phi[i];
                    }
                }
                let d = &space.dofs[t];
                for i in 0..nl {
                    for j in 0..nl {
                        ks.push(d[i], d[j], ke[i][j]);
                        if p.c[t] != 0.0 {
                            cs.push(d[i], d[j], ce[i][j]);
                        }
                    }
                    if fe[i] != 0.0 {
                        fs.push((d[i], fe[i]));
                    }
                }
                (ks, cs, fs)
            },
        )
        .collect();
    let mut ks = Triplets::with_capacity(nt * nl * nl);
    let mut cs = Triplets::with_capacity(0);
    let mut load = vec![0.0; space.ndof];
    for (k, c, f) in chunks {
        ks.extend(k);
        cs.extend(c);
        for (i, v) in f {
            load[i] += v;
        }
    }
    // boundary terms
    for &(e, h, uinf) in &p.robin {
        edge_terms(mesh, &space, e, |dofs, m, v| {
            for i in 0..dofs.len() {
                for j in 0..dofs.len() {
                    ks.push(dofs[i], dofs[j], h * m[i][j]);
                }
                load[dofs[i]] += h * uinf * v[i];
            }
        });
    }
    for &(e, q) in &p.neumann {
        edge_terms(mesh, &space, e, |dofs, _m, v| {
            for i in 0..dofs.len() {
                load[dofs[i]] += q * v[i];
            }
        });
    }
    let n = space.ndof;
    Ok(Assembled {
        stiffness: ks.to_csr(n, n),
        capacity: cs.to_csr(n, n),
        load,
        space,
    })
}

/// Edge mass matrix and edge integrals of the shape functions of edge `e`.
fn edge_terms(
    mesh: &Mesh,
    space: &Lagrange,
    e: usize,
    mut f: impl FnMut(&[usize], &[[f64; 3]; 3], &[f64; 3]),
) {
    let [a, b] = mesh.edges[e];
    let pa = mesh.nodes[a];
    let pb = mesh.nodes[b];
    let pm = mesh.edge_mid[e].unwrap_or([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])]);
    let (xs, ws) = quadrature::line(4);
    let p2 = space.order == 2;
    let dofs: Vec<usize> = if p2 {
        vec![a, b, mesh.n_nodes() + e]
    } else {
        vec![a, b]
    };
    let mut m = [[0.0; 3]; 3];
    let mut v = [0.0; 3];
    for (t, w) in xs.iter().zip(&ws) {
        // P2 geometry along the edge: x(t) = (1-t)(1-2t) pa + t(2t-1) pb + 4t(1-t) pm
        let d = [(4.0 * t - 3.0), (4.0 * t - 1.0), (4.0 - 8.0 * t)];
        let dx = [
            d[0] * pa[0] + d[1] * pb[0] + d[2] * pm[0],
            d[0] * pa[1] + d[1] * pb[1] + d[2] * pm[1],
        ];
        let js = (dx[0] * dx[0] + dx[1] * dx[1]).sqrt();
        let phi: [f64; 3] = if p2 {
            [
                (1.0 - t) * (1.0 - 2.0 * t),
                t * (2.0 * t - 1.0),
                4.0 * t * (1.0 - t),
            ]
        } else {
            [1.0 - t, *t, 0.0]
        };
        let nl = dofs.len();
        for i in 0..nl {
            v[i] += w * js * phi[i];
            for j in 0..nl {
                m[i][j] += w * js * phi[i] * phi[j];
            }
        }
    }
    f(&dofs, &m, &v);
}

/// Linear solver choice for [`solve_steady`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolver {
    /// Sparse Cholesky (symmetric positive definite systems).
    Cholesky,
    /// Sparse LU.
    Lu,
    /// Conjugate gradients with symmetric Gauss-Seidel.
    Cg,
}

/// Map from full DOFs to free (non-Dirichlet) DOFs.
pub fn free_map(
    n: usize,
    dirichlet: &[(usize, f64)],
) -> (Vec<Option<usize>>, usize, Vec<f64>, Vec<bool>) {
    let mut fixed = vec![false; n];
    let mut ud = vec![0.0; n];
    for &(i, v) in dirichlet {
        fixed[i] = true;
        ud[i] = v;
    }
    let mut map = vec![None; n];
    let mut k = 0;
    for i in 0..n {
        if !fixed[i] {
            map[i] = Some(k);
            k += 1;
        }
    }
    (map, k, ud, fixed)
}

/// Solve (K + c M) u = F with Dirichlet values; returns the full DOF vector.
pub fn solve_steady(
    a: &Assembled,
    dirichlet: &[(usize, f64)],
    reaction: bool,
    solver: LinearSolver,
) -> Result<Vec<f64>> {
    let n = a.space.ndof;
    let mat = if reaction && a.capacity.nnz() > 0 {
        Csr::axpby(1.0, &a.stiffness, 1.0, &a.capacity)
    } else {
        a.stiffness.clone()
    };
    solve_reduced(&mat, &a.load, dirichlet, n, solver)
}

/// Solve `mat u = rhs` with the given Dirichlet values eliminated.
pub fn solve_reduced(
    mat: &Csr<f64>,
    rhs: &[f64],
    dirichlet: &[(usize, f64)],
    n: usize,
    solver: LinearSolver,
) -> Result<Vec<f64>> {
    let dofs: Vec<usize> = dirichlet.iter().map(|d| d.0).collect();
    let vals: Vec<f64> = dirichlet.iter().map(|d| d.1).collect();
    Ok(solve_reduced_multi(mat, &[rhs.to_vec()], &dofs, &[vals], n, solver)?.remove(0))
}

/// Several solves with one factorisation: case `i` has load `rhs[i]` (or
/// `rhs[0]` if only one is given) and Dirichlet values `values[i]` on `dofs`.
pub fn solve_reduced_multi(
    mat: &Csr<f64>,
    rhs: &[Vec<f64>],
    dofs: &[usize],
    values: &[Vec<f64>],
    n: usize,
    solver: LinearSolver,
) -> Result<Vec<Vec<f64>>> {
    let ncase = values.len().max(rhs.len());
    let zero: Vec<(usize, f64)> = dofs.iter().map(|&d| (d, 0.0)).collect();
    let (map, nf, _, _) = free_map(n, &zero);
    let kff = if nf > 0 {
        Some(mat.submatrix(&map, nf, &map, nf))
    } else {
        None
    };
    enum S {
        D(Direct<f64>),
        Cg,
        None,
    }
    let fact = match (&kff, solver) {
        (None, _) => S::None,
        (Some(k), LinearSolver::Cholesky) => S::D(match Direct::cholesky(k) {
            Ok(f) => f,
            Err(_) => Direct::lu(k)?,
        }),
        (Some(k), LinearSolver::Lu) => S::D(Direct::lu(k)?),
        (Some(_), LinearSolver::Cg) => S::Cg,
    };
    let mut out = Vec::with_capacity(ncase);
    for c in 0..ncase {
        let mut u = vec![0.0; n];
        if !dofs.is_empty() {
            let v = &values[c.min(values.len() - 1)];
            for (k, &d) in dofs.iter().enumerate() {
                u[d] = v[k];
            }
        }
        if nf == 0 {
            out.push(u);
            continue;
        }
        let r = &rhs[c.min(rhs.len() - 1)];
        let aud = mat.matvec(&u);
        let mut b = vec![0.0; nf];
        for i in 0..n {
            if let Some(k) = map[i] {
                b[k] = r[i] - aud[i];
            }
        }
        let uf = match &fact {
            S::D(f) => f.solve(&b),
            S::Cg => {
                let mut x = vec![0.0; nf];
                let info = cg(
                    kff.as_ref().unwrap(),
                    &b,
                    &mut x,
                    Precond::Sgs,
                    1e-11,
                    20 * nf + 100,
                )?;
                if !info.converged {
                    bail!(
                        "conjugate gradients did not converge: residual {:.2e} after {} iterations",
                        info.residual,
                        info.iterations
                    );
                }
                x
            }
            S::None => unreachable!(),
        };
        for i in 0..n {
            if let Some(k) = map[i] {
                u[i] = uf[k];
            }
        }
        out.push(u);
    }
    Ok(out)
}

/// Transient solution by the theta scheme (1: backward Euler, 0.5:
/// Crank-Nicolson) from `u0`, with constant sources and boundary values.
/// Returns the DOF vectors at the steps listed in `save` (step indices,
/// 0 = initial) and the values of DOF combinations `probes` (weights over
/// DOFs) at every step.
pub fn solve_transient(
    a: &Assembled,
    dirichlet: &[(usize, f64)],
    u0: &[f64],
    dt: f64,
    nsteps: usize,
    theta: f64,
    save: &[usize],
    probes: &[Vec<(usize, f64)>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n = a.space.ndof;
    if a.capacity.nnz() == 0 {
        bail!("a transient problem needs a heat capacity (c > 0)");
    }
    let (map, nf, ud, _) = free_map(n, dirichlet);
    let lhs_full = Csr::axpby(1.0, &a.capacity, theta * dt, &a.stiffness);
    let rhs_full = Csr::axpby(1.0, &a.capacity, -(1.0 - theta) * dt, &a.stiffness);
    let lhs = lhs_full.submatrix(&map, nf, &map, nf);
    let fact = match Direct::cholesky(&lhs) {
        Ok(f) => f,
        Err(_) => Direct::lu(&lhs)?,
    };
    let mut u = u0.to_vec();
    for &(i, v) in dirichlet {
        u[i] = v;
    }
    let lud = lhs_full.matvec(&ud);
    let mut saved = vec![];
    let mut traces = vec![vec![0.0; nsteps + 1]; probes.len()];
    let record = |u: &[f64], step: usize, saved: &mut Vec<Vec<f64>>, traces: &mut Vec<Vec<f64>>| {
        if save.contains(&step) {
            saved.push(u.to_vec());
        }
        for (pi, pr) in probes.iter().enumerate() {
            traces[pi][step] = pr.iter().map(|&(i, w)| w * u[i]).sum();
        }
    };
    record(&u, 0, &mut saved, &mut traces);
    for step in 1..=nsteps {
        let r = rhs_full.matvec(&u);
        let mut b = vec![0.0; nf];
        for i in 0..n {
            if let Some(k) = map[i] {
                b[k] = r[i] + dt * a.load[i] - lud[i];
            }
        }
        fact.solve_in_place(&mut b);
        for i in 0..n {
            if let Some(k) = map[i] {
                u[i] = b[k];
            }
        }
        record(&u, step, &mut saved, &mut traces);
    }
    Ok((saved, traces))
}

/// Error norms against an exact solution: (L2 error, H1-seminorm error).
pub fn error_norms(
    mesh: &Mesh,
    space: &Lagrange,
    u: &[f64],
    exact: &(dyn Fn(f64, f64) -> (f64, f64, f64) + Sync),
) -> (f64, f64) {
    let (e0, e1): (f64, f64) = (0..mesh.n_tris())
        .into_par_iter()
        .map(|t| {
            let rule = quadrature::triangle(8);
            let nl = space.nloc();
            let (mut a, mut b) = (0.0, 0.0);
            for (l, w) in rule.points.iter().zip(&rule.weights) {
                let g = geometry(mesh, t, l);
                let phi = lagrange_values(space.order, l);
                let rg = lagrange_ref_grads(space.order, l);
                let (mut uh, mut gx, mut gy) = (0.0, 0.0, 0.0);
                for i in 0..nl {
                    let c = u[space.dofs[t][i]];
                    uh += c * phi[i];
                    let gr = g.grad(&rg[i]);
                    gx += c * gr[0];
                    gy += c * gr[1];
                }
                let (ue, ux, uy) = exact(g.x[0], g.x[1]);
                let dv = w * 0.5 * g.det;
                a += dv * (uh - ue).powi(2);
                b += dv * ((gx - ux).powi(2) + (gy - uy).powi(2));
            }
            (a, b)
        })
        .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
    (e0.sqrt(), e1.sqrt())
}

/// Load vector of a source function f(x, y) (quadrature of degree 8).
pub fn load_function(
    mesh: &Mesh,
    space: &Lagrange,
    f: &(dyn Fn(f64, f64) -> f64 + Sync),
) -> Vec<f64> {
    let parts: Vec<(usize, f64)> = (0..mesh.n_tris())
        .into_par_iter()
        .flat_map_iter(|t| {
            let rule = quadrature::triangle(8);
            let nl = space.nloc();
            let mut fe = [0.0; 6];
            for (l, w) in rule.points.iter().zip(&rule.weights) {
                let g = geometry(mesh, t, l);
                let phi = lagrange_values(space.order, l);
                let fv = f(g.x[0], g.x[1]) * w * 0.5 * g.det;
                for i in 0..nl {
                    fe[i] += fv * phi[i];
                }
            }
            (0..nl)
                .map(move |i| (space.dofs[t][i], fe[i]))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut b = vec![0.0; space.ndof];
    for (i, v) in parts {
        b[i] += v;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::rectangle;
    use std::f64::consts::PI;

    /// -lap u = f on the unit square, u = sin(pi x) sin(pi y) exp(x): orders 2/1 (P1), 3/2 (P2).
    #[test]
    fn manufactured_solution_convergence() {
        let exact = |x: f64, y: f64| {
            let (s, c) = ((PI * x).sin(), (PI * x).cos());
            let (sy, cy) = ((PI * y).sin(), (PI * y).cos());
            let e = x.exp();
            (s * sy * e, (PI * c + s) * sy * e, s * PI * cy * e)
        };
        // f = -lap u = (2 pi^2 s - 2 pi c - s) sy e
        let src = |x: f64, y: f64| {
            let (s, c) = ((PI * x).sin(), (PI * x).cos());
            (2.0 * PI * PI * s - 2.0 * PI * c - s) * (PI * y).sin() * x.exp()
        };
        for order in [1usize, 2] {
            let mut errs = vec![];
            for n in [8usize, 16, 32] {
                let mesh = rectangle(0.0, 1.0, 0.0, 1.0, n, n);
                // per-triangle constant source would limit the order: use a load vector from quadrature
                let space = Lagrange::new(&mesh, order);
                let p = ScalarProblem {
                    order,
                    k: vec![[1.0, 1.0, 0.0]; mesh.n_tris()],
                    c: vec![0.0; mesh.n_tris()],
                    f: vec![0.0; mesh.n_tris()],
                    ..Default::default()
                };
                let mut a = assemble(&mesh, &p).unwrap();
                a.load = load_function(&mesh, &space, &src);
                let bd = space.edge_dofs(&mesh, &mesh.boundary_edges());
                let dir: Vec<(usize, f64)> = bd.iter().map(|&i| (i, 0.0)).collect();
                let u = solve_steady(&a, &dir, false, LinearSolver::Cholesky).unwrap();
                errs.push(error_norms(&mesh, &space, &u, &exact));
            }
            let r0 = (errs[1].0 / errs[2].0).log2();
            let r1 = (errs[1].1 / errs[2].1).log2();
            let (w0, w1) = if order == 1 { (2.0, 1.0) } else { (3.0, 2.0) };
            assert!(
                (r0 - w0).abs() < 0.15 && (r1 - w1).abs() < 0.15,
                "P{order}: L2 order {r0}, H1 order {r1}, {errs:?}"
            );
        }
    }
}
