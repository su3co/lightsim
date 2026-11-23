//! Shift-invert eigen-solver for generalised sparse eigenproblems
//! `A x = lambda B x` (complex, non-Hermitian).
//!
//! The operator `(A - sigma B)^{-1} B` is factorised once with faer's sparse
//! LU; its largest eigenvalues `theta = 1 / (lambda - sigma)` (the
//! eigenvalues closest to `sigma`) are found by the implicitly restarted
//! Arnoldi method (Sorensen 1992, the algorithm of ARPACK) with exact shifts,
//! Givens-based shifted QR on the Hessenberg matrix and DGKS
//! re-orthogonalisation. The small dense eigenproblems use faer.

use crate::linsolve::Direct;
use crate::sparse::{Csr, dot, norm};
use anyhow::{Result, anyhow, bail};
use num_complex::Complex64 as C;

/// Options of [`eigs_shift_invert`].
#[derive(Clone, Debug)]
pub struct EigOptions {
    /// Number of eigenpairs.
    pub nev: usize,
    /// Krylov dimension (0: max(2 nev + 1, nev + 20)).
    pub ncv: usize,
    /// Relative tolerance on the Ritz residual of theta.
    pub tol: f64,
    /// Maximum number of restarts.
    pub max_restarts: usize,
}

impl Default for EigOptions {
    fn default() -> Self {
        EigOptions {
            nev: 4,
            ncv: 0,
            tol: 1e-10,
            max_restarts: 300,
        }
    }
}

/// Eigenpairs: eigenvalues closest to sigma first.
#[derive(Clone, Debug)]
pub struct EigResult {
    pub values: Vec<C>,
    /// Eigenvectors (unit 2-norm), one per value.
    pub vectors: Vec<Vec<C>>,
    pub restarts: usize,
    pub op_applications: usize,
}

fn hess_eig(h: &[Vec<C>], m: usize) -> Result<(Vec<C>, Vec<Vec<C>>)> {
    let a = faer::Mat::<C>::from_fn(m, m, |i, j| h[i][j]);
    let e = a
        .eigen()
        .map_err(|e| anyhow!("dense eigen-decomposition failed: {e:?}"))?;
    let s = e.S();
    let u = e.U();
    let vals: Vec<C> = (0..m).map(|i| s[i]).collect();
    let vecs: Vec<Vec<C>> = (0..m)
        .map(|j| {
            let v: Vec<C> = (0..m).map(|i| u[(i, j)]).collect();
            let n = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / n).collect()
        })
        .collect();
    Ok((vals, vecs))
}

const CHUNK: usize = 2048;

/// c_i = v_i^H w for all i, one parallel pass over row chunks.
fn block_dots(v: &[Vec<C>], w: &[C]) -> Vec<C> {
    use rayon::prelude::*;
    let k = v.len();
    let n = w.len();
    (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|b| {
            let (lo, hi) = (b * CHUNK, ((b + 1) * CHUNK).min(n));
            let mut c = vec![C::new(0.0, 0.0); k];
            for (i, vi) in v.iter().enumerate() {
                let mut s = C::new(0.0, 0.0);
                for q in lo..hi {
                    s += vi[q].conj() * w[q];
                }
                c[i] = s;
            }
            c
        })
        .reduce(
            || vec![C::new(0.0, 0.0); k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

/// w -= sum_i c_i v_i, parallel over row chunks.
fn block_update(v: &[Vec<C>], c: &[C], w: &mut [C]) {
    use rayon::prelude::*;
    w.par_chunks_mut(CHUNK).enumerate().for_each(|(b, wc)| {
        let lo = b * CHUNK;
        for (i, vi) in v.iter().enumerate() {
            let ci = c[i];
            for (q, wq) in wc.iter_mut().enumerate() {
                *wq -= ci * vi[lo + q];
            }
        }
    });
}

/// Implicitly restarted Arnoldi for the `nev` largest-magnitude eigenvalues
/// of the linear operator `op` of size `n`.
pub fn arnoldi(
    op: &mut dyn FnMut(&[C]) -> Vec<C>,
    n: usize,
    start: Vec<C>,
    opts: &EigOptions,
) -> Result<(Vec<C>, Vec<Vec<C>>, usize, usize)> {
    let k = opts.nev;
    if k == 0 || k >= n {
        bail!(
            "nev = {k} must be between 1 and n - 1 = {}",
            n.saturating_sub(1)
        );
    }
    let ncv_env = std::env::var("LIGHTSIM_FEM_NCV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let m = if opts.ncv > 0 {
        opts.ncv
    } else {
        ncv_env.unwrap_or((3 * k).max(k + 30))
    }
    .min(n);
    if m <= k {
        bail!("Krylov dimension {m} must exceed nev = {k}");
    }
    let mut v: Vec<Vec<C>> = Vec::with_capacity(m);
    let mut h = vec![vec![C::new(0.0, 0.0); m]; m];
    let mut napp = 0usize;
    // start vector in the range of the operator (purges the null space of B)
    let mut f = op(&start);
    napp += 1;
    let mut beta = norm(&f);
    if beta == 0.0 {
        bail!("the start vector is in the null space of the operator");
    }
    let mut j0 = 0usize;
    let mut zero_next = false;
    let mut beta_true = beta;
    for restart in 0..=opts.max_restarts {
        // extend the factorisation from j0 to m columns
        for j in j0..m {
            let vj: Vec<C> = f.iter().map(|x| x / beta).collect();
            if j > 0 {
                h[j][j - 1] = C::new(if zero_next { 0.0 } else { beta }, 0.0);
            }
            zero_next = false;
            v.truncate(j);
            v.push(vj);
            let mut w = op(&v[j]);
            napp += 1;
            let wn0 = norm(&w);
            let mut hcol = vec![C::new(0.0, 0.0); j + 1];
            for _pass in 0..2 {
                let c = block_dots(&v[..=j], &w);
                block_update(&v[..=j], &c, &mut w);
                for i in 0..=j {
                    hcol[i] += c[i];
                }
                if norm(&w) > 0.717 * wn0 {
                    break;
                }
            }
            for i in 0..=j {
                h[i][j] = hcol[i];
            }
            f = w;
            beta = norm(&f);
            beta_true = beta;
            if beta < 1e-14 * wn0.max(1e-300) && j + 1 < m {
                // invariant subspace: continue with a direction orthogonal to V, decoupled (h = 0)
                let mut r: Vec<C> = (0..n)
                    .map(|q| C::new(((q * 7919 + j * 104729) % 1000) as f64 / 1000.0 - 0.5, 0.0))
                    .collect();
                for _ in 0..2 {
                    for vi in v.iter() {
                        let c = dot(vi, &r);
                        for (rq, vq) in r.iter_mut().zip(vi) {
                            *rq -= c * vq;
                        }
                    }
                }
                f = r;
                beta = norm(&f);
                zero_next = true;
            }
        }
        // Ritz values of H
        let (vals, vecs) = hess_eig(&h, m)?;
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| vals[b].norm().partial_cmp(&vals[a].norm()).unwrap());
        let conv = order[..k]
            .iter()
            .all(|&i| beta_true * vecs[i][m - 1].norm() <= opts.tol * vals[i].norm().max(1e-300));
        if conv || restart == opts.max_restarts {
            if !conv {
                bail!(
                    "Arnoldi did not converge in {} restarts ({} operator applications); try a better target index or more modes",
                    opts.max_restarts,
                    napp
                );
            }
            let mut out_vals = vec![];
            let mut out_vecs = vec![];
            for &i in &order[..k] {
                let y = &vecs[i];
                let mut x = vec![C::new(0.0, 0.0); n];
                for (jj, vj) in v.iter().enumerate().take(m) {
                    let c = y[jj];
                    for (xq, vq) in x.iter_mut().zip(vj) {
                        *xq += c * vq;
                    }
                }
                let xn = norm(&x);
                x.iter_mut().for_each(|q| *q /= xn);
                out_vals.push(vals[i]);
                out_vecs.push(x);
            }
            return Ok((out_vals, out_vecs, restart, napp));
        }
        // implicit restart with the m - k unwanted Ritz values as shifts
        let mut q = vec![vec![C::new(0.0, 0.0); m]; m];
        for i in 0..m {
            q[i][i] = C::new(1.0, 0.0);
        }
        for &si in &order[k..] {
            let mu = vals[si];
            for i in 0..m {
                h[i][i] -= mu;
            }
            let mut rots = Vec::with_capacity(m - 1);
            for i in 0..m - 1 {
                let (a, b) = (h[i][i], h[i + 1][i]);
                let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
                let (c, s) = if r == 0.0 {
                    (1.0, C::new(0.0, 0.0))
                } else if a.norm() == 0.0 {
                    (0.0, C::new(1.0, 0.0))
                } else {
                    (a.norm() / r, (a / a.norm()) * b.conj() / r)
                };
                for col in i..m {
                    let (x, y) = (h[i][col], h[i + 1][col]);
                    h[i][col] = x * c + s * y;
                    h[i + 1][col] = -s.conj() * x + y * c;
                }
                rots.push((c, s));
            }
            for (i, &(c, s)) in rots.iter().enumerate() {
                let rmax = (i + 2).min(m);
                for row in 0..rmax {
                    let (x, y) = (h[row][i], h[row][i + 1]);
                    h[row][i] = x * c + s.conj() * y;
                    h[row][i + 1] = -s * x + y * c;
                }
                for row in 0..m {
                    let (x, y) = (q[row][i], q[row][i + 1]);
                    q[row][i] = x * c + s.conj() * y;
                    q[row][i + 1] = -s * x + y * c;
                }
            }
            for i in 0..m {
                h[i][i] += mu;
            }
        }
        // V <- V Q (first k + 1 columns are needed)
        let mut vnew: Vec<Vec<C>> = vec![vec![C::new(0.0, 0.0); n]; k + 1];
        for (jj, col) in vnew.iter_mut().enumerate() {
            for (i, vi) in v.iter().enumerate() {
                let c = q[i][jj];
                if c.norm() == 0.0 {
                    continue;
                }
                for (xq, vq) in col.iter_mut().zip(vi) {
                    *xq += c * vq;
                }
            }
        }
        let hk = h[k][k - 1];
        let sk = q[m - 1][k - 1];
        let fnew: Vec<C> = (0..n).map(|i| vnew[k][i] * hk + f[i] * sk).collect();
        vnew.truncate(k);
        v = vnew;
        for i in 0..m {
            for jj in 0..m {
                if i >= k || jj >= k {
                    h[i][jj] = C::new(0.0, 0.0);
                }
            }
        }
        f = fnew;
        beta = norm(&f);
        beta_true = beta;
        j0 = k;
    }
    unreachable!()
}

/// Eigenpairs of `A x = lambda B x` closest to `sigma` (complex).
pub fn eigs_shift_invert(a: &Csr<C>, b: &Csr<C>, sigma: C, opts: &EigOptions) -> Result<EigResult> {
    let n = a.nrows;
    let shifted = Csr::axpby(C::new(1.0, 0.0), a, -sigma, b);
    let t0 = std::time::Instant::now();
    let lu = Direct::lu(&shifted)?;
    if std::env::var_os("LIGHTSIM_FEM_DEBUG").is_some() {
        eprintln!(
            "shift-invert LU: n {n}, nnz {}, {:.3} s",
            shifted.nnz(),
            t0.elapsed().as_secs_f64()
        );
    }
    let solve_time = std::cell::Cell::new(0.0f64);
    let mut op = |x: &[C]| -> Vec<C> {
        let t = std::time::Instant::now();
        let mut y = b.matvec(x);
        lu.solve_in_place(&mut y);
        solve_time.set(solve_time.get() + t.elapsed().as_secs_f64());
        y
    };
    let start: Vec<C> = (0..n)
        .map(|i| C::new(1.0 + ((i * 2654435761usize) % 1000) as f64 * 1e-3, 0.0))
        .collect();
    let t0 = std::time::Instant::now();
    let (theta, vecs, restarts, napp) = arnoldi(&mut op, n, start, opts)?;
    if std::env::var_os("LIGHTSIM_FEM_DEBUG").is_some() {
        eprintln!(
            "arnoldi: n {n}, nev {}, restarts {restarts}, operator applications {napp}, {:.3} s ({:.3} s in the operator)",
            opts.nev,
            t0.elapsed().as_secs_f64(),
            solve_time.get()
        );
    }
    let values = theta.iter().map(|t| sigma + C::new(1.0, 0.0) / t).collect();
    Ok(EigResult {
        values,
        vectors: vecs,
        restarts,
        op_applications: napp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_eigenvalues() {
        // -u'' = lambda u on (0, 1), u(0) = u(1) = 0, FD with n interior points
        let n = 400;
        let hh = 1.0 / (n + 1) as f64;
        let (mut r, mut c, mut va) = (vec![], vec![], vec![]);
        for i in 0..n {
            r.push(i);
            c.push(i);
            va.push(C::new(2.0 / (hh * hh), 0.0));
            if i > 0 {
                r.push(i);
                c.push(i - 1);
                va.push(C::new(-1.0 / (hh * hh), 0.0));
            }
            if i + 1 < n {
                r.push(i);
                c.push(i + 1);
                va.push(C::new(-1.0 / (hh * hh), 0.0));
            }
        }
        let a = Csr::from_triplets(n, n, &r, &c, &va);
        let id: Vec<usize> = (0..n).collect();
        let b = Csr::from_triplets(n, n, &id, &id, &vec![C::new(1.0, 0.0); n]);
        let opts = EigOptions {
            nev: 5,
            ncv: 12,
            ..Default::default()
        };
        let res = eigs_shift_invert(&a, &b, C::new(150.0, 0.0), &opts).unwrap();
        let exact = |k: usize| {
            4.0 / (hh * hh) * ((k as f64) * std::f64::consts::PI * hh / 2.0).sin().powi(2)
        };
        let mut got: Vec<f64> = res.values.iter().map(|z| z.re).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // closest five to 150: k = 2, 3, 4, 1, 5 (pi^2 k^2 ~ 9.87, 39.5, 88.8, 157.9, 246.7, 355)
        let mut want: Vec<f64> = (1..=8).map(exact).collect();
        want.sort_by(|a, b| (a - 150.0).abs().partial_cmp(&(b - 150.0).abs()).unwrap());
        let mut want5 = want[..5].to_vec();
        want5.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (g, w) in got.iter().zip(&want5) {
            assert!((g - w).abs() < 1e-8 * w, "{g} vs {w}");
        }
        for (val, vec) in res.values.iter().zip(&res.vectors) {
            let av = a.matvec(vec);
            let res: f64 = av
                .iter()
                .zip(vec)
                .map(|(x, y)| (x - val * y).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(res < 1e-6 * val.norm());
        }
    }
}
