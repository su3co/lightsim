//! Compressed sparse row matrices assembled from triplets, real or complex,
//! and their hand-off to faer's sparse direct solvers.

use num_complex::Complex64;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

/// Field of matrix entries: `f64` or `Complex64`.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + Default
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + std::fmt::Debug
    + 'static
{
    fn from_f64(x: f64) -> Self;
    fn conj(self) -> Self;
    fn abs2(self) -> f64;
    fn div(self, o: Self) -> Self;
    fn zero() -> Self {
        Self::default()
    }
}
impl Scalar for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn conj(self) -> Self {
        self
    }
    fn abs2(self) -> f64 {
        self * self
    }
    fn div(self, o: Self) -> Self {
        self / o
    }
}
impl Scalar for Complex64 {
    fn from_f64(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    fn abs2(self) -> f64 {
        self.norm_sqr()
    }
    fn div(self, o: Self) -> Self {
        self / o
    }
}

/// A square or rectangular CSR matrix.
#[derive(Clone, Debug)]
pub struct Csr<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Scalar> Csr<T> {
    /// Sum duplicate triplets into a CSR matrix (columns sorted in each row).
    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Csr<T> {
        let nnz = rows.len();
        let mut count = vec![0usize; nrows + 1];
        for &r in rows {
            count[r + 1] += 1;
        }
        for i in 0..nrows {
            count[i + 1] += count[i];
        }
        let mut next = count.clone();
        let mut ci = vec![0usize; nnz];
        let mut cv = vec![T::zero(); nnz];
        for k in 0..nnz {
            let r = rows[k];
            let p = next[r];
            ci[p] = cols[k];
            cv[p] = vals[k];
            next[r] += 1;
        }
        // sort and merge each row in parallel
        let rows_out: Vec<(Vec<usize>, Vec<T>)> = (0..nrows)
            .into_par_iter()
            .map(|r| {
                let (a, b) = (count[r], count[r + 1]);
                let mut pairs: Vec<(usize, T)> = (a..b).map(|p| (ci[p], cv[p])).collect();
                pairs.sort_unstable_by_key(|x| x.0);
                let mut idx = Vec::with_capacity(pairs.len());
                let mut val: Vec<T> = Vec::with_capacity(pairs.len());
                for (c, v) in pairs {
                    if idx.last() == Some(&c) {
                        *val.last_mut().unwrap() += v;
                    } else {
                        idx.push(c);
                        val.push(v);
                    }
                }
                (idx, val)
            })
            .collect();
        let mut indptr = Vec::with_capacity(nrows + 1);
        indptr.push(0);
        let total: usize = rows_out.iter().map(|r| r.0.len()).sum();
        let mut indices = Vec::with_capacity(total);
        let mut values = Vec::with_capacity(total);
        for (i, v) in rows_out {
            indices.extend(i);
            values.extend(v);
            indptr.push(indices.len());
        }
        Csr {
            nrows,
            ncols,
            indptr,
            indices,
            values,
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// y = A x
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        (0..self.nrows)
            .into_par_iter()
            .with_min_len(512)
            .map(|r| {
                let mut s = T::zero();
                for p in self.indptr[r]..self.indptr[r + 1] {
                    s += self.values[p] * x[self.indices[p]];
                }
                s
            })
            .collect()
    }

    pub fn diagonal(&self) -> Vec<T> {
        (0..self.nrows)
            .map(|r| {
                let mut d = T::zero();
                for p in self.indptr[r]..self.indptr[r + 1] {
                    if self.indices[p] == r {
                        d = self.values[p];
                    }
                }
                d
            })
            .collect()
    }

    /// Entry (r, c), zero if not stored.
    pub fn get(&self, r: usize, c: usize) -> T {
        let s = &self.indices[self.indptr[r]..self.indptr[r + 1]];
        match s.binary_search(&c) {
            Ok(k) => self.values[self.indptr[r] + k],
            Err(_) => T::zero(),
        }
    }

    /// a A + b B (same shape).
    pub fn axpby(a: T, x: &Csr<T>, b: T, y: &Csr<T>) -> Csr<T> {
        let mut rows = Vec::with_capacity(x.nnz() + y.nnz());
        let mut cols = Vec::with_capacity(x.nnz() + y.nnz());
        let mut vals = Vec::with_capacity(x.nnz() + y.nnz());
        for (m, s) in [(x, a), (y, b)] {
            for r in 0..m.nrows {
                for p in m.indptr[r]..m.indptr[r + 1] {
                    rows.push(r);
                    cols.push(m.indices[p]);
                    vals.push(s * m.values[p]);
                }
            }
        }
        Csr::from_triplets(x.nrows, x.ncols, &rows, &cols, &vals)
    }

    /// Keep rows and columns in `keep` (a map old -> Some(new)).
    pub fn submatrix(
        &self,
        rmap: &[Option<usize>],
        nr: usize,
        cmap: &[Option<usize>],
        nc: usize,
    ) -> Csr<T> {
        let (mut rows, mut cols, mut vals) = (vec![], vec![], vec![]);
        for r in 0..self.nrows {
            if let Some(nr_) = rmap[r] {
                for p in self.indptr[r]..self.indptr[r + 1] {
                    if let Some(nc_) = cmap[self.indices[p]] {
                        rows.push(nr_);
                        cols.push(nc_);
                        vals.push(self.values[p]);
                    }
                }
            }
        }
        Csr::from_triplets(nr, nc, &rows, &cols, &vals)
    }

    pub fn transpose(&self) -> Csr<T> {
        let (mut rows, mut cols, mut vals) = (vec![], vec![], vec![]);
        for r in 0..self.nrows {
            for p in self.indptr[r]..self.indptr[r + 1] {
                rows.push(self.indices[p]);
                cols.push(r);
                vals.push(self.values[p]);
            }
        }
        Csr::from_triplets(self.ncols, self.nrows, &rows, &cols, &vals)
    }

    /// Triplets in faer's format.
    pub(crate) fn faer_triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, T>> {
        let mut t = Vec::with_capacity(self.nnz());
        for r in 0..self.nrows {
            for p in self.indptr[r]..self.indptr[r + 1] {
                t.push(faer::sparse::Triplet::new(
                    r,
                    self.indices[p],
                    self.values[p],
                ));
            }
        }
        t
    }
}

/// Accumulates element contributions; each thread fills its own buffer.
#[derive(Default)]
pub struct Triplets<T> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<T>,
}

impl<T: Scalar> Triplets<T> {
    pub fn with_capacity(n: usize) -> Self {
        Triplets {
            rows: Vec::with_capacity(n),
            cols: Vec::with_capacity(n),
            vals: Vec::with_capacity(n),
        }
    }
    #[inline]
    pub fn push(&mut self, r: usize, c: usize, v: T) {
        self.rows.push(r);
        self.cols.push(c);
        self.vals.push(v);
    }
    pub fn extend(&mut self, o: Triplets<T>) {
        self.rows.extend(o.rows);
        self.cols.extend(o.cols);
        self.vals.extend(o.vals);
    }
    pub fn to_csr(&self, nrows: usize, ncols: usize) -> Csr<T> {
        Csr::from_triplets(nrows, ncols, &self.rows, &self.cols, &self.vals)
    }
}

/// Length of the fixed pieces of the reductions below: partial sums over
/// these pieces are added in order, so a result has the same bits whatever
/// the number of threads (and the pieces are shared out only when there are
/// enough of them to pay for waking other threads).
pub const PIECE: usize = 4096;
/// Fewest pieces worth sharing out between threads.
pub const PAR_PIECES: usize = 16;

/// `f(lo, hi)` over the pieces of `0..n` of length `piece`, in order (in
/// parallel when there are at least `PAR_PIECES` of them).
pub fn pieces<R: Send, F: Fn(usize, usize) -> R + Sync>(n: usize, piece: usize, f: F) -> Vec<R> {
    let np = n.div_ceil(piece);
    let one = |b: usize| f(b * piece, ((b + 1) * piece).min(n));
    if np < PAR_PIECES { (0..np).map(one).collect() } else { (0..np).into_par_iter().map(one).collect() }
}

pub fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    pieces(a.len(), PIECE, |lo, hi| {
        let mut s = T::zero();
        for q in lo..hi {
            s += a[q].conj() * b[q];
        }
        s
    })
    .into_iter()
    .fold(T::zero(), |x, y| x + y)
}
pub fn norm<T: Scalar>(a: &[T]) -> f64 {
    pieces(a.len(), PIECE, |lo, hi| a[lo..hi].iter().map(|x| x.abs2()).sum::<f64>())
        .into_iter()
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triplets_sum_duplicates() {
        let a = Csr::from_triplets(2, 2, &[0, 0, 1, 0], &[1, 0, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.get(0, 1), 5.0);
        assert_eq!(a.get(0, 0), 2.0);
        assert_eq!(a.get(1, 0), 0.0);
        assert_eq!(a.matvec(&[1.0, 1.0]), vec![7.0, 3.0]);
    }
}
