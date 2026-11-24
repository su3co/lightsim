//! Linear solvers: sparse direct (faer LU and Cholesky, multithreaded) and
//! preconditioned iterative solvers (conjugate gradients with Jacobi or
//! symmetric Gauss-Seidel, BiCGSTAB with ILU(0)).

use crate::sparse::{Csr, Scalar, dot, norm};
use anyhow::{Result, anyhow};
use faer::linalg::solvers::Solve;
use faer::sparse::SparseColMat;

/// Entry types faer can factorise (f64, Complex64).
pub trait FaerScalar: Scalar + faer::traits::ComplexField {}
impl<T: Scalar + faer::traits::ComplexField> FaerScalar for T {}

enum Fact<T: FaerScalar> {
    Lu(faer::sparse::linalg::solvers::Lu<usize, T>),
    Llt(faer::sparse::linalg::solvers::Llt<usize, T>),
}

/// A factorised sparse matrix.
pub struct Direct<T: FaerScalar> {
    fact: Fact<T>,
    n: usize,
}

/// Number of tasks faer splits its factorisations and dense kernels into:
/// fixed, so that the results have the same bits whatever the number of
/// threads that run them (faer's default follows the thread count).
/// Eight rather than four: measured on a GB10 with 10k to 200k triangles,
/// as fast with 1 to 4 threads and 2 to 4 % faster with 8 and 16.
pub const FAER_TASKS: usize = 8;

/// Fix faer's parallelism to `FAER_TASKS` tasks (once per process; the
/// `LIGHTSIM_FAER_TASKS` variable overrides it for measurements, 1 = serial).
pub(crate) fn fixed_parallelism() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let n = std::env::var("LIGHTSIM_FAER_TASKS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0).unwrap_or(FAER_TASKS);
        faer::set_global_parallelism(if n == 1 { faer::Par::Seq } else { faer::Par::rayon(n) });
    });
}

fn to_faer<T: FaerScalar>(a: &Csr<T>) -> Result<SparseColMat<usize, T>> {
    fixed_parallelism();
    SparseColMat::try_new_from_triplets(a.nrows, a.ncols, &a.faer_triplets())
        .map_err(|e| anyhow!("sparse matrix: {e:?}"))
}

impl<T: FaerScalar> Direct<T> {
    /// LU with partial pivoting (any square non-singular matrix).
    pub fn lu(a: &Csr<T>) -> Result<Self> {
        let m = to_faer(a)?;
        let lu = m.sp_lu().map_err(|e| anyhow!("sparse LU failed: {e:?}"))?;
        Ok(Direct {
            fact: Fact::Lu(lu),
            n: a.nrows,
        })
    }
    /// Cholesky LL^H (Hermitian positive definite).
    pub fn cholesky(a: &Csr<T>) -> Result<Self> {
        let m = to_faer(a)?;
        let l = m.sp_cholesky(faer::Side::Lower).map_err(|e| {
            anyhow!("sparse Cholesky failed (matrix not positive definite?): {e:?}")
        })?;
        Ok(Direct {
            fact: Fact::Llt(l),
            n: a.nrows,
        })
    }
    /// Solve in place.
    pub fn solve_in_place(&self, b: &mut [T]) {
        assert_eq!(b.len(), self.n);
        let v = faer::MatMut::from_column_major_slice_mut(b, self.n, 1);
        match &self.fact {
            Fact::Lu(f) => f.solve_in_place(v),
            Fact::Llt(f) => f.solve_in_place(v),
        }
    }
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let mut x = b.to_vec();
        self.solve_in_place(&mut x);
        x
    }
}

/// Preconditioner for the iterative solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precond {
    None,
    Jacobi,
    /// Symmetric Gauss-Seidel (SSOR with omega = 1); symmetric, for CG.
    Sgs,
    /// Incomplete LU without fill; for BiCGSTAB.
    Ilu0,
}

/// Result of an iterative solve.
#[derive(Clone, Debug)]
pub struct IterInfo {
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
}

struct Pre<'a, T: Scalar> {
    a: &'a Csr<T>,
    kind: Precond,
    dinv: Vec<T>,
    diag_pos: Vec<usize>,
    ilu: Vec<T>,
}

impl<'a, T: Scalar> Pre<'a, T> {
    fn new(a: &'a Csr<T>, kind: Precond) -> Result<Self> {
        let n = a.nrows;
        let mut diag_pos = vec![usize::MAX; n];
        for r in 0..n {
            for p in a.indptr[r]..a.indptr[r + 1] {
                if a.indices[p] == r {
                    diag_pos[r] = p;
                }
            }
            if diag_pos[r] == usize::MAX && kind != Precond::None {
                return Err(anyhow!("row {r} has no diagonal entry"));
            }
        }
        let dinv = if kind == Precond::None {
            vec![]
        } else {
            diag_pos
                .iter()
                .map(|&p| T::from_f64(1.0).div(a.values[p]))
                .collect()
        };
        let mut ilu = vec![];
        if kind == Precond::Ilu0 {
            ilu = a.values.clone();
            for i in 0..n {
                for p in a.indptr[i]..diag_pos[i] {
                    let k = a.indices[p];
                    let lik = ilu[p].div(ilu[diag_pos[k]]);
                    ilu[p] = lik;
                    // row i -= lik * row k (only existing entries, columns > k)
                    let (mut q, qe) = (diag_pos[k] + 1, a.indptr[k + 1]);
                    let mut s = p + 1;
                    let se = a.indptr[i + 1];
                    while q < qe && s < se {
                        let (cq, cs) = (a.indices[q], a.indices[s]);
                        if cq == cs {
                            let v = ilu[q];
                            ilu[s] = ilu[s] - lik * v;
                            q += 1;
                            s += 1;
                        } else if cq < cs {
                            q += 1;
                        } else {
                            s += 1;
                        }
                    }
                }
            }
        }
        Ok(Pre {
            a,
            kind,
            dinv,
            diag_pos,
            ilu,
        })
    }

    fn apply(&self, r: &[T]) -> Vec<T> {
        let a = self.a;
        let n = a.nrows;
        match self.kind {
            Precond::None => r.to_vec(),
            Precond::Jacobi => r.iter().zip(&self.dinv).map(|(x, d)| *x * *d).collect(),
            Precond::Sgs => {
                // (D + L) D^-1 (D + U) z = r
                let mut y = vec![T::zero(); n];
                for i in 0..n {
                    let mut s = r[i];
                    for p in a.indptr[i]..self.diag_pos[i] {
                        s = s - a.values[p] * y[a.indices[p]];
                    }
                    y[i] = s * self.dinv[i];
                }
                for i in 0..n {
                    y[i] = y[i] * a.values[self.diag_pos[i]];
                }
                let mut z = vec![T::zero(); n];
                for i in (0..n).rev() {
                    let mut s = y[i];
                    for p in self.diag_pos[i] + 1..a.indptr[i + 1] {
                        s = s - a.values[p] * z[a.indices[p]];
                    }
                    z[i] = s * self.dinv[i];
                }
                z
            }
            Precond::Ilu0 => {
                let mut y = vec![T::zero(); n];
                for i in 0..n {
                    let mut s = r[i];
                    for p in a.indptr[i]..self.diag_pos[i] {
                        s = s - self.ilu[p] * y[a.indices[p]];
                    }
                    y[i] = s;
                }
                let mut z = vec![T::zero(); n];
                for i in (0..n).rev() {
                    let mut s = y[i];
                    for p in self.diag_pos[i] + 1..a.indptr[i + 1] {
                        s = s - self.ilu[p] * z[a.indices[p]];
                    }
                    z[i] = s.div(self.ilu[self.diag_pos[i]]);
                }
                z
            }
        }
    }
}

/// Preconditioned conjugate gradients (Hermitian positive definite `a`).
/// `x` holds the initial guess and receives the solution.
pub fn cg<T: Scalar>(
    a: &Csr<T>,
    b: &[T],
    x: &mut [T],
    pc: Precond,
    rtol: f64,
    maxit: usize,
) -> Result<IterInfo> {
    let pre = Pre::new(a, pc)?;
    let bn = norm(b).max(1e-300);
    let ax = a.matvec(x);
    let mut r: Vec<T> = b.iter().zip(&ax).map(|(b, a)| *b - *a).collect();
    let mut z = pre.apply(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for it in 0..maxit {
        let res = norm(&r) / bn;
        if res <= rtol {
            return Ok(IterInfo {
                iterations: it,
                residual: res,
                converged: true,
            });
        }
        let ap = a.matvec(&p);
        let alpha = rz.div(dot(&p, &ap));
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
        }
        z = pre.apply(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new.div(rz);
        rz = rz_new;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
    }
    let res = norm(&r) / bn;
    Ok(IterInfo {
        iterations: maxit,
        residual: res,
        converged: res <= rtol,
    })
}

/// Right-preconditioned BiCGSTAB (general square `a`).
pub fn bicgstab<T: Scalar>(
    a: &Csr<T>,
    b: &[T],
    x: &mut [T],
    pc: Precond,
    rtol: f64,
    maxit: usize,
) -> Result<IterInfo> {
    let pre = Pre::new(a, pc)?;
    let n = b.len();
    let bn = norm(b).max(1e-300);
    let ax = a.matvec(x);
    let mut r: Vec<T> = b.iter().zip(&ax).map(|(b, a)| *b - *a).collect();
    let r0 = r.clone();
    let one = T::from_f64(1.0);
    let (mut rho, mut alpha, mut omega) = (one, one, one);
    let mut v = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];
    for it in 0..maxit {
        let res = norm(&r) / bn;
        if res <= rtol {
            return Ok(IterInfo {
                iterations: it,
                residual: res,
                converged: true,
            });
        }
        let rho_new = dot(&r0, &r);
        if rho_new.abs2() == 0.0 {
            break;
        }
        let beta = rho_new.div(rho) * alpha.div(omega);
        rho = rho_new;
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        let ph = pre.apply(&p);
        v = a.matvec(&ph);
        alpha = rho.div(dot(&r0, &v));
        let s: Vec<T> = (0..n).map(|i| r[i] - alpha * v[i]).collect();
        let sh = pre.apply(&s);
        let t = a.matvec(&sh);
        let tt = dot(&t, &t);
        omega = if tt.abs2() == 0.0 {
            T::zero()
        } else {
            dot(&t, &s).div(tt)
        };
        for i in 0..n {
            x[i] += alpha * ph[i] + omega * sh[i];
            r[i] = s[i] - omega * t[i];
        }
        if omega.abs2() == 0.0 {
            break;
        }
    }
    let ax = a.matvec(x);
    let res = norm(&b.iter().zip(&ax).map(|(b, a)| *b - *a).collect::<Vec<T>>()) / bn;
    Ok(IterInfo {
        iterations: maxit,
        residual: res,
        converged: res <= rtol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn lap1d(n: usize) -> Csr<f64> {
        let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(2.0);
            if i > 0 {
                r.push(i);
                c.push(i - 1);
                v.push(-1.0);
            }
            if i + 1 < n {
                r.push(i);
                c.push(i + 1);
                v.push(-1.0);
            }
        }
        Csr::from_triplets(n, n, &r, &c, &v)
    }

    #[test]
    fn direct_and_iterative_agree() {
        let n = 200;
        let a = lap1d(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x_lu = Direct::lu(&a).unwrap().solve(&b);
        let x_ch = Direct::cholesky(&a).unwrap().solve(&b);
        for pc in [Precond::Jacobi, Precond::Sgs] {
            let mut x = vec![0.0; n];
            let info = cg(&a, &b, &mut x, pc, 1e-12, 5000).unwrap();
            assert!(info.converged, "{pc:?} {info:?}");
            for i in 0..n {
                assert!((x[i] - x_lu[i]).abs() < 1e-7 * x_lu[i].abs().max(1.0));
            }
        }
        let mut x = vec![0.0; n];
        let info = bicgstab(&a, &b, &mut x, Precond::Ilu0, 1e-12, 500).unwrap();
        assert!(info.converged && info.iterations <= 2, "{info:?}"); // ILU(0) is exact for a tridiagonal matrix
        for i in 0..n {
            assert!((x_ch[i] - x_lu[i]).abs() < 1e-9 * x_lu[i].abs().max(1.0));
            assert!((x[i] - x_lu[i]).abs() < 1e-8 * x_lu[i].abs().max(1.0));
        }
    }

    #[test]
    fn complex_lu() {
        let n = 50;
        let a = lap1d(n);
        let i = Complex64::new(0.0, 1.0);
        let ac = Csr {
            nrows: n,
            ncols: n,
            indptr: a.indptr.clone(),
            indices: a.indices.clone(),
            values: a
                .values
                .iter()
                .map(|v| {
                    Complex64::new(*v, 0.0)
                        + if *v > 0.0 {
                            i * 0.5
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                })
                .collect(),
        };
        let b: Vec<Complex64> = (0..n).map(|k| Complex64::new(1.0, k as f64)).collect();
        let x = Direct::lu(&ac).unwrap().solve(&b);
        let r = ac.matvec(&x);
        for k in 0..n {
            assert!((r[k] - b[k]).norm() < 1e-10);
        }
    }
}
