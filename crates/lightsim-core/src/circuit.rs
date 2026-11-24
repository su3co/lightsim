//! Network solve of a frequency-domain circuit by sub-network growth.
//!
//! A circuit is a set of blocks (the instances, each with a small dense
//! S-matrix `S[out, in]`) joined port to port. Instead of assembling one
//! matrix of all ports and solving `S_EE + S_EI C (1 - S_II C)^-1 S_IE`, the
//! blocks are joined one connection at a time (Filipsson and Gunnar's
//! sub-network growth): joining two different blocks at one port each costs
//! one scalar division, joining two ports of one block a 2x2 elimination.
//! The order of the joins is chosen by the caller (the Python package picks
//! the join that leaves the fewest ports), so a chain of two-ports never
//! holds more than a few ports at a time.
//!
//! The same [`Program`] runs at every point of a batch (frequencies, Monte
//! Carlo trials, ...), in parallel over the batch. Every value is read
//! through its own strides, so a value that does not depend on a batch axis
//! (stride 0) is never copied.

use num_complex::Complex64;
use rayon::prelude::*;

/// One join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Join port `ia` of block `a` to port `ib` of block `b` (`a != b`).
    /// The new block `out` has the ports of `a` without `ia`, in order,
    /// then those of `b` without `ib`.
    Merge { a: usize, ia: usize, b: usize, ib: usize, out: usize },
    /// Join ports `i` and `j` of block `a`; the new block `out` has the
    /// remaining ports of `a` in order.
    Join { a: usize, i: usize, j: usize, out: usize },
}

/// A network solve, independent of the values.
#[derive(Clone, Debug)]
pub struct Program {
    /// Port count of every block: the initial blocks `0..n_init`, then one
    /// per op, in order (`Op::*::out` must be `n_init + op index`).
    pub sizes: Vec<usize>,
    pub n_init: usize,
    pub ops: Vec<Op>,
    /// Initial entries: `(block, out, in, value)`; absent entries are zero.
    pub fill: Vec<(usize, usize, usize, usize)>,
    /// Result entries: `(slot, block, out, in)`; slots not listed are zero.
    pub outputs: Vec<(usize, usize, usize, usize)>,
    /// Number of result slots.
    pub n_out: usize,
}

/// Errors of [`Program::check`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    #[error("op {0}: output block must be {1}")]
    OutId(usize, usize),
    #[error("op {0}: block {1} is not available (undefined or already joined)")]
    Block(usize, usize),
    #[error("op {0}: port index out of range")]
    Port(usize),
    #[error("op {0}: block size {1} does not match the join")]
    Size(usize, usize),
    #[error("entry {0}: index out of range")]
    Entry(usize),
}

impl Program {
    /// Check the program is consistent (indices, sizes, every block used
    /// once). The sizes of the op outputs are checked against the joins.
    pub fn check(&self, n_values: usize) -> Result<(), ProgramError> {
        let nb = self.n_init + self.ops.len();
        if self.sizes.len() != nb {
            return Err(ProgramError::Size(usize::MAX, self.sizes.len()));
        }
        let mut alive = vec![false; nb];
        alive[..self.n_init].iter_mut().for_each(|a| *a = true);
        for (k, op) in self.ops.iter().enumerate() {
            let id = self.n_init + k;
            match *op {
                Op::Merge { a, ia, b, ib, out } => {
                    if out != id {
                        return Err(ProgramError::OutId(k, id));
                    }
                    for x in [a, b] {
                        if x >= id || !alive[x] {
                            return Err(ProgramError::Block(k, x));
                        }
                    }
                    if a == b {
                        return Err(ProgramError::Block(k, b));
                    }
                    if ia >= self.sizes[a] || ib >= self.sizes[b] {
                        return Err(ProgramError::Port(k));
                    }
                    if self.sizes[out] + 2 != self.sizes[a] + self.sizes[b] {
                        return Err(ProgramError::Size(k, self.sizes[out]));
                    }
                    alive[a] = false;
                    alive[b] = false;
                }
                Op::Join { a, i, j, out } => {
                    if out != id {
                        return Err(ProgramError::OutId(k, id));
                    }
                    if a >= id || !alive[a] {
                        return Err(ProgramError::Block(k, a));
                    }
                    if i >= self.sizes[a] || j >= self.sizes[a] || i == j {
                        return Err(ProgramError::Port(k));
                    }
                    if self.sizes[out] + 2 != self.sizes[a] {
                        return Err(ProgramError::Size(k, self.sizes[out]));
                    }
                    alive[a] = false;
                }
            }
            alive[id] = true;
        }
        for (e, &(b, o, i, v)) in self.fill.iter().enumerate() {
            if b >= self.n_init || o >= self.sizes[b] || i >= self.sizes[b] || v >= n_values {
                return Err(ProgramError::Entry(e));
            }
        }
        for (e, &(s, b, o, i)) in self.outputs.iter().enumerate() {
            if s >= self.n_out || b >= nb || !alive[b] || o >= self.sizes[b] || i >= self.sizes[b] {
                return Err(ProgramError::Entry(e));
            }
        }
        Ok(())
    }

    /// When each initial block is first needed: `before[k]` lists the
    /// initial blocks op `k` consumes for the first time, `end` those that
    /// no op consumes (read only by the outputs). The kernel copies a block
    /// in just before it is needed.
    pub fn schedule(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut seen = vec![false; self.n_init];
        let mut before = vec![vec![]; self.ops.len()];
        for (k, op) in self.ops.iter().enumerate() {
            let xs = match *op {
                Op::Merge { a, b, .. } => [Some(a), Some(b)],
                Op::Join { a, .. } => [Some(a), None],
            };
            for x in xs.into_iter().flatten() {
                if x < self.n_init && !seen[x] {
                    seen[x] = true;
                    before[k].push(x);
                }
            }
        }
        let end = (0..self.n_init).filter(|&b| !seen[b]).collect();
        (before, end)
    }

    /// Offsets of every block in a scratch buffer and the buffer length.
    /// Blocks are placed (first fit) where they overlap no live block: an
    /// initial block when it is first needed (see [`Program::schedule`]),
    /// the output of an op next to its live inputs, and the blocks an op
    /// consumes are freed, so the buffer holds the largest set of live
    /// blocks, not the whole circuit.
    pub fn layout(&self) -> (Vec<usize>, usize) {
        let area = |b: usize| self.sizes[b] * self.sizes[b];
        let mut offs = vec![0usize; self.sizes.len()];
        let mut total = 0usize;
        // free intervals [start, stop), sorted and coalesced
        let mut free: Vec<(usize, usize)> = vec![];
        let mut alloc = |len: usize, free: &mut Vec<(usize, usize)>| -> usize {
            match free.iter().position(|&(a, b)| b - a >= len) {
                Some(p) => {
                    let a = free[p].0;
                    if free[p].1 - a == len {
                        free.remove(p);
                    } else {
                        free[p].0 += len;
                    }
                    a
                }
                None => {
                    total += len;
                    total - len
                }
            }
        };
        let release = |s: usize, e: usize, free: &mut Vec<(usize, usize)>| {
            if s == e {
                return;
            }
            let pos = free.partition_point(|&(a, _)| a < s);
            free.insert(pos, (s, e));
            if pos + 1 < free.len() && free[pos].1 == free[pos + 1].0 {
                free[pos].1 = free[pos + 1].1;
                free.remove(pos + 1);
            }
            if pos > 0 && free[pos - 1].1 == free[pos].0 {
                free[pos - 1].1 = free[pos].1;
                free.remove(pos);
            }
        };
        let (before, end) = self.schedule();
        for (k, op) in self.ops.iter().enumerate() {
            for &x in &before[k] {
                offs[x] = alloc(area(x), &mut free);
            }
            let id = self.n_init + k;
            offs[id] = alloc(area(id), &mut free);
            let consumed = match *op {
                Op::Merge { a, b, .. } => [Some(a), Some(b)],
                Op::Join { a, .. } => [Some(a), None],
            };
            for x in consumed.into_iter().flatten() {
                release(offs[x], offs[x] + area(x), &mut free);
            }
        }
        for &x in &end {
            offs[x] = alloc(area(x), &mut free);
        }
        (offs, total)
    }

    /// Floating-point work per batch point (complex multiply-adds), roughly.
    pub fn cost(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match *op {
                Op::Merge { out, .. } | Op::Join { out, .. } => {
                    let n = self.sizes[out];
                    n * n + 4
                }
            })
            .sum()
    }
}

/// A value read through strides over the batch shape.
pub struct Value<'a> {
    pub data: &'a [Complex64],
    /// Offset of the first element in `data`.
    pub offset: usize,
    /// Stride (in elements) along every axis of the batch shape; 0 where
    /// the value is broadcast. Negative strides are not supported.
    pub strides: Vec<usize>,
}

/// Run `program` at every point of a batch of shape `shape` (row-major).
/// `out` is `(n_out, B)` row-major: slot `s` of point `b` is
/// `out[s * B + b]`. Runs on the current rayon pool.
///
/// The batch is processed in lanes of consecutive points: every block
/// entry holds the real and the imaginary parts of all lanes in two
/// contiguous runs (structure of arrays), so each join is a set of short
/// loops over the lanes that the compiler vectorises. The kernel is
/// compiled for the baseline instruction set and, on x86-64, also for AVX2
/// with FMA and for AVX-512, chosen at run time.
pub fn run(program: &Program, values: &[Value], shape: &[usize], out: &mut [Complex64]) {
    let nb: usize = shape.iter().product();
    assert_eq!(out.len(), program.n_out * nb);
    for v in values {
        assert_eq!(v.strides.len(), shape.len());
    }
    if nb == 0 {
        return;
    }
    out.iter_mut().for_each(|x| *x = Complex64::new(0.0, 0.0));
    let (offs, total) = program.layout();
    let (before, end) = program.schedule();
    let mut fills = vec![vec![]; program.n_init];
    for &(blk, o, i, v) in &program.fill {
        fills[blk].push((o, i, v));
    }
    let maxn = program.sizes.iter().copied().max().unwrap_or(0);
    // lanes per chunk (measured on a GB10 over chains, rings, lattices and
    // Clements meshes, benchmarks/circuit/lanes.py): narrow
    // programs gain from long contiguous reads of every input (up to 4096
    // lanes), wide blocks from keeping the scratch of a chunk (16 bytes per
    // entry and lane) within the cache a thread can count on, the larger of
    // 2 MB and a 32 MB share of the last-level cache; never fewer than 16
    // lanes (vector units), and never so many that a thread gets fewer than
    // two chunks
    let entries = total.max(1) + 2 * maxn + 8;
    let threads = rayon::current_num_threads().max(1);
    let budget = (2usize << 20).max((32usize << 20) / threads);
    let mut lanes = 4096usize;
    while lanes > 16 && entries * lanes * 16 > budget {
        lanes /= 2;
    }
    let balance = if threads > 1 { (nb / (2 * threads)).max(8) } else { nb };
    while lanes > 8 && lanes > balance {
        lanes /= 2;
    }
    let lanes = lanes.min(nb.next_power_of_two().max(8));
    // measurement override (benchmarks/circuit/lanes.py)
    let lanes = std::env::var("LIGHTSIM_CIRCUIT_LANES").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&l| l > 0).unwrap_or(lanes);
    let ex = Exec { program, values, shape, offs: &offs, total, maxn, lanes, nb, before: &before, end: &end, fills: &fills };
    let nchunks = nb.div_ceil(lanes);
    let outp = SendPtr(out.as_mut_ptr());
    let level = simd_level();
    // at most 8 tasks per thread, each with one scratch for a contiguous run
    // of chunks (rayon's per-split initialisers would allocate and clear a
    // scratch for nearly every chunk); 8 per thread still balances fast and
    // slow cores
    let ntasks = nchunks.min(8 * threads).max(1);
    (0..ntasks).into_par_iter().for_each(|t| {
        let mut sc = Scratch::new(&ex);
        for c in t * nchunks / ntasks..(t + 1) * nchunks / ntasks {
            let b0 = c * lanes;
            let len = lanes.min(nb - b0);
            ex.chunk(level, &mut sc, b0, len, &outp);
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut Complex64);
// disjoint ranges of the output are written by different chunks
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Simd {
    Base,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

fn simd_level() -> Simd {
    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var_os("LIGHTSIM_CIRCUIT_SIMD").is_some_and(|v| v == "base") {
            return Simd::Base;
        }
        if std::is_x86_feature_detected!("avx512f") && std::env::var_os("LIGHTSIM_CIRCUIT_SIMD").is_none_or(|v| v != "avx2") {
            return Simd::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Simd::Avx2;
        }
    }
    Simd::Base
}

struct Exec<'a> {
    program: &'a Program,
    values: &'a [Value<'a>],
    shape: &'a [usize],
    offs: &'a [usize],
    total: usize,
    maxn: usize,
    lanes: usize,
    nb: usize,
    before: &'a [Vec<usize>],
    end: &'a [usize],
    fills: &'a [Vec<(usize, usize, usize)>],
}

struct Scratch {
    re: Vec<f64>,
    im: Vec<f64>,
    // per-lane temporaries of the joins: up to 2 * maxn + 6 vectors
    tre: Vec<f64>,
    tim: Vec<f64>,
    coord: Vec<usize>,
    // row segments of the current chunk: (first lane, lanes, flat offset of
    // the segment's first point per value)
    segs: Vec<(usize, usize)>,
    seg_base: Vec<usize>,
}

impl Scratch {
    fn new(ex: &Exec) -> Scratch {
        let l = ex.lanes;
        let nt = 2 * ex.maxn + 8;
        Scratch { re: vec![0.0; ex.total * l], im: vec![0.0; ex.total * l], tre: vec![0.0; nt * l], tim: vec![0.0; nt * l], coord: vec![0; ex.shape.len()], segs: vec![], seg_base: vec![] }
    }
}

impl Exec<'_> {
    fn chunk(&self, level: Simd, sc: &mut Scratch, b0: usize, len: usize, out: &SendPtr) {
        match level {
            Simd::Base => self.chunk_impl(sc, b0, len, out),
            #[cfg(target_arch = "x86_64")]
            Simd::Avx2 => unsafe { self.chunk_avx2(sc, b0, len, out) },
            #[cfg(target_arch = "x86_64")]
            Simd::Avx512 => unsafe { self.chunk_avx512(sc, b0, len, out) },
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn chunk_avx2(&self, sc: &mut Scratch, b0: usize, len: usize, out: &SendPtr) {
        self.chunk_impl(sc, b0, len, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx2,fma")]
    unsafe fn chunk_avx512(&self, sc: &mut Scratch, b0: usize, len: usize, out: &SendPtr) {
        self.chunk_impl(sc, b0, len, out)
    }

    #[inline(always)]
    fn chunk_impl(&self, sc: &mut Scratch, b0: usize, len: usize, out: &SendPtr) {
        let p = self.program;
        let l = self.lanes;
        self.segments(sc, b0, len);
        for (k, op) in p.ops.iter().enumerate() {
            for &blk in &self.before[k] {
                self.fill_block(sc, blk);
            }
            match *op {
                Op::Merge { a, ia, b, ib, out } => merge(self, sc, a, ia, b, ib, out, len),
                Op::Join { a, i, j, out } => join(self, sc, a, i, j, out, len),
            }
        }
        for &blk in self.end {
            self.fill_block(sc, blk);
        }
        for &(s, blk, o, i) in &p.outputs {
            let e = (self.offs[blk] + o * p.sizes[blk] + i) * l;
            let (re, im) = (&sc.re[e..e + len], &sc.im[e..e + len]);
            // SAFETY: chunk c writes out[s * nb + b0 .. + len] only
            let dst = unsafe { std::slice::from_raw_parts_mut(out.0.add(s * self.nb + b0), len) };
            for k in 0..len {
                dst[k] = Complex64::new(re[k], im[k]);
            }
        }
    }

    /// Split the lanes `b0 .. b0 + len` into row segments of the batch
    /// (within a segment only the last index moves, so every value is read
    /// with one stride) and record each value's offset at every segment.
    #[inline(always)]
    fn segments(&self, sc: &mut Scratch, b0: usize, len: usize) {
        let shape = self.shape;
        let nd = shape.len();
        let last = if nd == 0 { 1 } else { shape[nd - 1] };
        let mut rem = b0;
        for d in (0..nd).rev() {
            sc.coord[d] = rem % shape[d];
            rem /= shape[d];
        }
        sc.segs.clear();
        sc.seg_base.clear();
        let mut k0 = 0;
        while k0 < len {
            let c_last = if nd == 0 { 0 } else { sc.coord[nd - 1] };
            let seg = (last - c_last).min(len - k0);
            sc.segs.push((k0, seg));
            for val in self.values {
                let mut base = val.offset;
                for d in 0..nd {
                    base += sc.coord[d] * val.strides[d];
                }
                sc.seg_base.push(base);
            }
            k0 += seg;
            if nd > 0 {
                sc.coord[nd - 1] += seg;
                let mut d = nd - 1;
                while d > 0 && sc.coord[d] >= shape[d] {
                    sc.coord[d] -= shape[d];
                    sc.coord[d - 1] += 1;
                    d -= 1;
                }
            }
        }
    }

    /// Copy initial block `blk` into its place in the scratch (absent
    /// entries are zero).
    #[inline(always)]
    fn fill_block(&self, sc: &mut Scratch, blk: usize) {
        let p = self.program;
        let l = self.lanes;
        let n = p.sizes[blk];
        let o0 = self.offs[blk] * l;
        sc.re[o0..o0 + n * n * l].iter_mut().for_each(|x| *x = 0.0);
        sc.im[o0..o0 + n * n * l].iter_mut().for_each(|x| *x = 0.0);
        let nd = self.shape.len();
        let nv = self.values.len();
        for &(o, i, v) in &self.fills[blk] {
            let val = &self.values[v];
            let st = if nd == 0 { 0 } else { val.strides[nd - 1] };
            let e0 = (self.offs[blk] + o * n + i) * l;
            for (q, &(k0, seg)) in sc.segs.iter().enumerate() {
                let base = sc.seg_base[q * nv + v];
                let e = e0 + k0;
                let (re, im) = (&mut sc.re[e..e + seg], &mut sc.im[e..e + seg]);
                if st == 0 {
                    let x = val.data[base];
                    re.iter_mut().for_each(|r| *r = x.re);
                    im.iter_mut().for_each(|r| *r = x.im);
                } else if st == 1 {
                    let src = &val.data[base..base + seg];
                    for k in 0..seg {
                        re[k] = src[k].re;
                        im[k] = src[k].im;
                    }
                } else {
                    for k in 0..seg {
                        let x = val.data[base + k * st];
                        re[k] = x.re;
                        im[k] = x.im;
                    }
                }
            }
        }
    }
}

// ---- lane-wise complex arithmetic on (re, im) slices ----------------------

/// Pointers to the lanes of an entry: (re, im).
#[derive(Clone, Copy)]
struct E(*mut f64, *mut f64);

impl E {
    #[inline(always)]
    fn at(sc: &mut Scratch, entry: usize, l: usize) -> E {
        unsafe { E(sc.re.as_mut_ptr().add(entry * l), sc.im.as_mut_ptr().add(entry * l)) }
    }
    #[inline(always)]
    fn tmp(sc: &mut Scratch, k: usize, l: usize) -> E {
        unsafe { E(sc.tre.as_mut_ptr().add(k * l), sc.tim.as_mut_ptr().add(k * l)) }
    }
    #[inline(always)]
    fn s<'a>(self, len: usize) -> (&'a [f64], &'a [f64]) {
        unsafe { (std::slice::from_raw_parts(self.0, len), std::slice::from_raw_parts(self.1, len)) }
    }
    #[inline(always)]
    fn m<'a>(self, len: usize) -> (&'a mut [f64], &'a mut [f64]) {
        unsafe { (std::slice::from_raw_parts_mut(self.0, len), std::slice::from_raw_parts_mut(self.1, len)) }
    }
}

/// r = x * y (r distinct from x and y)
#[inline(always)]
fn cmul(r: E, x: E, y: E, len: usize) {
    let (xr, xi) = x.s(len);
    let (yr, yi) = y.s(len);
    let (rr, ri) = r.m(len);
    for k in 0..len {
        let (a, b, c, d) = (xr[k], xi[k], yr[k], yi[k]);
        rr[k] = a * c - b * d;
        ri[k] = a * d + b * c;
    }
}

/// r = z + x * y (r distinct from z, x and y)
#[inline(always)]
fn cfma(r: E, z: E, x: E, y: E, len: usize) {
    let (zr, zi) = z.s(len);
    let (xr, xi) = x.s(len);
    let (yr, yi) = y.s(len);
    let (rr, ri) = r.m(len);
    for k in 0..len {
        let (a, b, c, d) = (xr[k], xi[k], yr[k], yi[k]);
        rr[k] = zr[k] + (a * c - b * d);
        ri[k] = zi[k] + (a * d + b * c);
    }
}

/// r += x * y (r distinct from x and y)
#[inline(always)]
fn cacc(r: E, x: E, y: E, len: usize) {
    let (xr, xi) = x.s(len);
    let (yr, yi) = y.s(len);
    let (rr, ri) = r.m(len);
    for k in 0..len {
        let (a, b, c, d) = (xr[k], xi[k], yr[k], yi[k]);
        rr[k] += a * c - b * d;
        ri[k] += a * d + b * c;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge(ex: &Exec, sc: &mut Scratch, a: usize, ia: usize, b: usize, ib: usize, out: usize, len: usize) {
    let l = ex.lanes;
    let sz = &ex.program.sizes;
    let (na, nb, no) = (sz[a], sz[b], sz[out]);
    let (oa, ob, oo) = (ex.offs[a], ex.offs[b], ex.offs[out]);
    let ea = |i: usize, j: usize| oa + i * na + j;
    let eb = |i: usize, j: usize| ob + i * nb + j;
    let eo = |i: usize, j: usize| oo + i * no + j;
    let aa = E::at(sc, ea(ia, ia), l);
    let bb = E::at(sc, eb(ib, ib), l);
    // temporaries: 0 = inv, 1 = scaled column, 2.. = u_j (A row times bb), then v_j
    let inv = E::tmp(sc, 0, l);
    let col = E::tmp(sc, 1, l);
    {
        // inv = 1 / (1 - aa bb)
        let (ar, ai) = aa.s(len);
        let (br, bi) = bb.s(len);
        let (rr, ri) = inv.m(len);
        for k in 0..len {
            let dr = 1.0 - (ar[k] * br[k] - ai[k] * bi[k]);
            let di = -(ar[k] * bi[k] + ai[k] * br[k]);
            let n = dr * dr + di * di;
            rr[k] = dr / n;
            ri[k] = -di / n;
        }
    }
    let ma = na - 1;
    // u_j = bb * A[ia, j] for j in A', v_j = aa * B[ib, j] for j in B'
    let u0 = 2;
    let v0 = u0 + ma;
    let mut k = 0;
    for j in 0..na {
        if j == ia {
            continue;
        }
        cmul(E::tmp(sc, u0 + k, l), bb, E::at(sc, ea(ia, j), l), len);
        k += 1;
    }
    let mut k = 0;
    for j in 0..nb {
        if j == ib {
            continue;
        }
        cmul(E::tmp(sc, v0 + k, l), aa, E::at(sc, eb(ib, j), l), len);
        k += 1;
    }
    let mut ri = 0;
    for i in 0..na {
        if i == ia {
            continue;
        }
        // col = A[i, ia] / D
        cmul(col, E::at(sc, ea(i, ia), l), inv, len);
        let mut rj = 0;
        for j in 0..na {
            if j == ia {
                continue;
            }
            cfma(E::at(sc, eo(ri, rj), l), E::at(sc, ea(i, j), l), col, E::tmp(sc, u0 + rj, l), len);
            rj += 1;
        }
        let mut rj = 0;
        for j in 0..nb {
            if j == ib {
                continue;
            }
            cmul(E::at(sc, eo(ri, ma + rj), l), col, E::at(sc, eb(ib, j), l), len);
            rj += 1;
        }
        ri += 1;
    }
    let mut ri = 0;
    for i in 0..nb {
        if i == ib {
            continue;
        }
        // col = B[i, ib] / D
        cmul(col, E::at(sc, eb(i, ib), l), inv, len);
        let mut rj = 0;
        for j in 0..na {
            if j == ia {
                continue;
            }
            cmul(E::at(sc, eo(ma + ri, rj), l), col, E::at(sc, ea(ia, j), l), len);
            rj += 1;
        }
        let mut rj = 0;
        for j in 0..nb {
            if j == ib {
                continue;
            }
            cfma(E::at(sc, eo(ma + ri, ma + rj), l), E::at(sc, eb(i, j), l), col, E::tmp(sc, v0 + rj, l), len);
            rj += 1;
        }
        ri += 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn join(ex: &Exec, sc: &mut Scratch, a: usize, k: usize, lp: usize, out: usize, len: usize) {
    let l = ex.lanes;
    let sz = &ex.program.sizes;
    let (n, no) = (sz[a], sz[out]);
    let (oa, oo) = (ex.offs[a], ex.offs[out]);
    let es = |i: usize, j: usize| oa + i * n + j;
    let eo = |i: usize, j: usize| oo + i * no + j;
    let (skl, slk, skk, sll) = (E::at(sc, es(k, lp), l), E::at(sc, es(lp, k), l), E::at(sc, es(k, k), l), E::at(sc, es(lp, lp), l));
    // temporaries: 0 inv, 1 ckl, 2 clk, 3 ckk, 4 cll, 5 pk, 6 pl
    let (inv, ckl, clk, ckk, cll, pk, pl) = (E::tmp(sc, 0, l), E::tmp(sc, 1, l), E::tmp(sc, 2, l), E::tmp(sc, 3, l), E::tmp(sc, 4, l), E::tmp(sc, 5, l), E::tmp(sc, 6, l));
    {
        // inv = 1 / ((1 - skl)(1 - slk) - skk sll)
        let (a1, a2) = skl.s(len);
        let (b1, b2) = slk.s(len);
        let (c1, c2) = skk.s(len);
        let (d1, d2) = sll.s(len);
        let (r1, r2) = inv.m(len);
        for q in 0..len {
            let (xr, xi) = (1.0 - a1[q], -a2[q]);
            let (yr, yi) = (1.0 - b1[q], -b2[q]);
            let pr = xr * yr - xi * yi - (c1[q] * d1[q] - c2[q] * d2[q]);
            let pi = xr * yi + xi * yr - (c1[q] * d2[q] + c2[q] * d1[q]);
            let nn = pr * pr + pi * pi;
            r1[q] = pr / nn;
            r2[q] = -pi / nn;
        }
    }
    {
        let (a1, a2) = skl.s(len);
        let (b1, b2) = slk.s(len);
        let (i1, i2) = inv.s(len);
        let (k1, k2) = ckl.m(len);
        let (l1, l2) = clk.m(len);
        for q in 0..len {
            let (xr, xi) = (1.0 - a1[q], -a2[q]);
            k1[q] = xr * i1[q] - xi * i2[q];
            k2[q] = xr * i2[q] + xi * i1[q];
            let (yr, yi) = (1.0 - b1[q], -b2[q]);
            l1[q] = yr * i1[q] - yi * i2[q];
            l2[q] = yr * i2[q] + yi * i1[q];
        }
    }
    cmul(ckk, skk, inv, len);
    cmul(cll, sll, inv, len);
    let mut ri = 0;
    for i in 0..n {
        if i == k || i == lp {
            continue;
        }
        let (sik, sil) = (E::at(sc, es(i, k), l), E::at(sc, es(i, lp), l));
        // pk = sik cll + sil clk (coefficient of S_kj), pl = sik ckl + sil ckk (of S_lj)
        cmul(pk, sik, cll, len);
        cacc(pk, sil, clk, len);
        cmul(pl, sik, ckl, len);
        cacc(pl, sil, ckk, len);
        let mut rj = 0;
        for j in 0..n {
            if j == k || j == lp {
                continue;
            }
            let r = E::at(sc, eo(ri, rj), l);
            cfma(r, E::at(sc, es(i, j), l), pk, E::at(sc, es(k, j), l), len);
            cacc(r, pl, E::at(sc, es(lp, j), l), len);
            rj += 1;
        }
        ri += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Dense reference: S_EE + S_EI C (1 - S_II C)^-1 S_IE by Gaussian
    /// elimination with partial pivoting.
    fn dense(s: &[Complex64], n: usize, ext: &[usize], pairs: &[(usize, usize)]) -> Vec<Complex64> {
        let int: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let ni = int.len();
        let pos = |g: usize| int.iter().position(|&x| x == g).unwrap();
        let partner = |p: usize| -> usize {
            for &(a, b) in pairs {
                if a == int[p] {
                    return pos(b);
                }
                if b == int[p] {
                    return pos(a);
                }
            }
            unreachable!()
        };
        // A = 1 - S_II C ; (S_II C)[r, q] = S_II[r, partner(q)]
        let ne = ext.len();
        let mut a = vec![c(0.0, 0.0); ni * ni];
        let mut rhs = vec![c(0.0, 0.0); ni * ne];
        for r in 0..ni {
            for q in 0..ni {
                a[r * ni + q] = if r == q { c(1.0, 0.0) } else { c(0.0, 0.0) } - s[int[r] * n + int[partner(q)]];
            }
            for (e, &x) in ext.iter().enumerate() {
                rhs[r * ne + e] = s[int[r] * n + x];
            }
        }
        // solve A X = rhs
        for col in 0..ni {
            let p = (col..ni).max_by(|&x, &y| a[x * ni + col].norm().partial_cmp(&a[y * ni + col].norm()).unwrap()).unwrap();
            for q in 0..ni {
                a.swap(col * ni + q, p * ni + q);
            }
            for e in 0..ne {
                rhs.swap(col * ne + e, p * ne + e);
            }
            let d = a[col * ni + col];
            for r in 0..ni {
                if r != col {
                    let f = a[r * ni + col] / d;
                    for q in 0..ni {
                        let t = a[col * ni + q];
                        a[r * ni + q] -= f * t;
                    }
                    for e in 0..ne {
                        let t = rhs[col * ne + e];
                        rhs[r * ne + e] -= f * t;
                    }
                }
            }
        }
        for r in 0..ni {
            let d = a[r * ni + r];
            for e in 0..ne {
                rhs[r * ne + e] /= d;
            }
        }
        // S_EE + S_EI C X
        let mut out = vec![c(0.0, 0.0); ne * ne];
        for (i, &xi) in ext.iter().enumerate() {
            for (j, &xj) in ext.iter().enumerate() {
                let mut v = s[xi * n + xj];
                for q in 0..ni {
                    v += s[xi * n + int[q]] * rhs[partner(q) * ne + j];
                }
                out[i * ne + j] = v;
            }
        }
        out
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
    }

    #[test]
    fn joins_match_dense_solve() {
        // three blocks: a 3-port, a 2-port and a 4-port, joined into a loop
        // with one self-join left at the end; random passive-ish entries
        let sizes0 = [3usize, 2, 4];
        let base = [0usize, 3, 5];
        let n = 9;
        let mut rng = Lcg(7);
        let mut s = vec![c(0.0, 0.0); n * n];
        let mut vals = vec![];
        let mut fill = vec![];
        for (blk, &m) in sizes0.iter().enumerate() {
            for o in 0..m {
                for i in 0..m {
                    let v = c(0.4 * rng.next(), 0.4 * rng.next());
                    s[(base[blk] + o) * n + base[blk] + i] = v;
                    fill.push((blk, o, i, vals.len()));
                    vals.push(v);
                }
            }
        }
        // global ports: block0 0..3, block1 3..5, block2 5..9
        // connections: (b0.1, b1.0), (b1.1, b2.0), (b2.1, b0.2); externals b0.0, b2.2, b2.3
        let pairs = [(1usize, 3usize), (4, 5), (6, 2)];
        let ext = [0usize, 7, 8];
        let reference = dense(&s, n, &ext, &pairs);
        // program: merge b0.1-b1.0 -> block 3 = [b0.0, b0.2, b1.1]
        // merge block3.2 (b1.1) - b2.0 -> block 4 = [b0.0, b0.2, b2.1, b2.2, b2.3]
        // join block4 ports 1 (b0.2) and 2 (b2.1) -> block 5 = [b0.0, b2.2, b2.3]
        let ops = vec![Op::Merge { a: 0, ia: 1, b: 1, ib: 0, out: 3 }, Op::Merge { a: 3, ia: 2, b: 2, ib: 0, out: 4 }, Op::Join { a: 4, i: 1, j: 2, out: 5 }];
        let mut outputs = vec![];
        for i in 0..3 {
            for j in 0..3 {
                outputs.push((i * 3 + j, 5, i, j));
            }
        }
        let p = Program { sizes: vec![3, 2, 4, 3, 5, 3], n_init: 3, ops, fill, outputs, n_out: 9 };
        p.check(vals.len()).unwrap();
        let values: Vec<Value> = vals.iter().map(|v| Value { data: std::slice::from_ref(v), offset: 0, strides: vec![0] }).collect();
        let mut out = vec![c(0.0, 0.0); 9 * 4];
        run(&p, &values, &[4], &mut out);
        for slot in 0..9 {
            for b in 0..4 {
                let d = (out[slot * 4 + b] - reference[slot]).norm();
                assert!(d < 1e-14, "slot {slot}: {d}");
            }
        }
    }

    #[test]
    fn strided_values_broadcast_over_the_batch() {
        // one two-port whose transmission varies along axis 1 and reflection along axis 0
        let t: Vec<Complex64> = (0..5).map(|k| c(0.1 * k as f64, 0.2)).collect();
        let r: Vec<Complex64> = (0..3).map(|k| c(0.0, 0.05 * k as f64)).collect();
        let values = vec![Value { data: &t, offset: 0, strides: vec![0, 1] }, Value { data: &r, offset: 0, strides: vec![1, 0] }];
        let fill = vec![(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)];
        let outputs = vec![(0, 0, 1, 0), (1, 0, 0, 0)];
        let p = Program { sizes: vec![2], n_init: 1, ops: vec![], fill, outputs, n_out: 3 };
        p.check(2).unwrap();
        let mut out = vec![c(9.0, 9.0); 3 * 15];
        run(&p, &values, &[3, 5], &mut out);
        for a in 0..3 {
            for k in 0..5 {
                assert_eq!(out[a * 5 + k], t[k]);
                assert_eq!(out[15 + a * 5 + k], r[a]);
                assert_eq!(out[30 + a * 5 + k], c(0.0, 0.0));
            }
        }
    }

    #[test]
    fn per_point_values_across_lanes_and_rows() {
        // the network of joins_match_dense_solve with every entry different
        // at every point of a (5, 37) batch; one entry broadcast along axis 0
        let sizes0 = [3usize, 2, 4];
        let base = [0usize, 3, 5];
        let n = 9;
        let (d0, d1) = (5usize, 37usize);
        let np_ = d0 * d1;
        let mut rng = Lcg(5);
        let mut data: Vec<Vec<Complex64>> = vec![];
        let mut fill = vec![];
        let mut strides = vec![];
        for (blk, &m) in sizes0.iter().enumerate() {
            for o in 0..m {
                for i in 0..m {
                    let bcast = blk == 2 && o == 0 && i == 1;
                    let len = if bcast { d1 } else { np_ };
                    data.push((0..len).map(|_| c(0.4 * rng.next(), 0.4 * rng.next())).collect());
                    strides.push(if bcast { vec![0, 1] } else { vec![d1, 1] });
                    fill.push((blk, o, i, data.len() - 1));
                }
            }
        }
        let ops = vec![Op::Merge { a: 0, ia: 1, b: 1, ib: 0, out: 3 }, Op::Merge { a: 3, ia: 2, b: 2, ib: 0, out: 4 }, Op::Join { a: 4, i: 1, j: 2, out: 5 }];
        let mut outputs = vec![];
        for i in 0..3 {
            for j in 0..3 {
                outputs.push((i * 3 + j, 5, i, j));
            }
        }
        let p = Program { sizes: vec![3, 2, 4, 3, 5, 3], n_init: 3, ops, fill: fill.clone(), outputs, n_out: 9 };
        p.check(data.len()).unwrap();
        let values: Vec<Value> = data.iter().zip(&strides).map(|(d, st)| Value { data: d, offset: 0, strides: st.clone() }).collect();
        let mut out = vec![c(0.0, 0.0); 9 * np_];
        run(&p, &values, &[d0, d1], &mut out);
        let pairs = [(1usize, 3usize), (4, 5), (6, 2)];
        let ext = [0usize, 7, 8];
        for a in 0..d0 {
            for b in 0..d1 {
                let mut s = vec![c(0.0, 0.0); n * n];
                for &(blk, o, i, v) in &fill {
                    let k = strides[v][0] * a + strides[v][1] * b;
                    s[(base[blk] + o) * n + base[blk] + i] = data[v][k];
                }
                let r = dense(&s, n, &ext, &pairs);
                for slot in 0..9 {
                    let d = (out[slot * np_ + a * d1 + b] - r[slot]).norm();
                    assert!(d < 1e-13, "point ({a}, {b}) slot {slot}: {d}");
                }
            }
        }
    }

    #[test]
    fn long_chain_reuses_the_scratch_buffer() {
        // 200 random two-ports in a chain, joined left to right; the external
        // ports are the two ends
        let m = 200;
        let n = 2 * m;
        let mut rng = Lcg(11);
        let mut s = vec![c(0.0, 0.0); n * n];
        let mut vals = vec![];
        let mut fill = vec![];
        for blk in 0..m {
            for o in 0..2 {
                for i in 0..2 {
                    let v = c(0.5 * rng.next(), 0.5 * rng.next());
                    s[(2 * blk + o) * n + 2 * blk + i] = v;
                    fill.push((blk, o, i, vals.len()));
                    vals.push(v);
                }
            }
        }
        let pairs: Vec<(usize, usize)> = (0..m - 1).map(|k| (2 * k + 1, 2 * k + 2)).collect();
        let reference = dense(&s, n, &[0, n - 1], &pairs);
        let mut ops = vec![];
        let mut sizes = vec![2usize; m];
        let mut cur = 0;
        for k in 1..m {
            let out = sizes.len();
            ops.push(Op::Merge { a: cur, ia: 1, b: k, ib: 0, out });
            sizes.push(2);
            cur = out;
        }
        let outputs = vec![(0, cur, 0, 0), (1, cur, 0, 1), (2, cur, 1, 0), (3, cur, 1, 1)];
        let p = Program { sizes, n_init: m, ops, fill, outputs, n_out: 4 };
        p.check(vals.len()).unwrap();
        let (_, total) = p.layout();
        assert!(total <= 16, "scratch {total}");
        let values: Vec<Value> = vals.iter().map(|v| Value { data: std::slice::from_ref(v), offset: 0, strides: vec![] }).collect();
        let mut out = vec![c(0.0, 0.0); 4];
        run(&p, &values, &[], &mut out);
        for k in 0..4 {
            let d = (out[k] - reference[k]).norm();
            assert!(d < 1e-12 * (1.0 + reference[k].norm()), "slot {k}: {d}");
        }
    }

    #[test]
    fn check_rejects_bad_programs() {
        let p = Program { sizes: vec![2, 2, 2], n_init: 2, ops: vec![Op::Merge { a: 0, ia: 0, b: 0, ib: 1, out: 2 }], fill: vec![], outputs: vec![], n_out: 0 };
        assert!(p.check(0).is_err());
        let p = Program { sizes: vec![2, 2, 3], n_init: 2, ops: vec![Op::Merge { a: 0, ia: 0, b: 1, ib: 1, out: 2 }], fill: vec![], outputs: vec![], n_out: 0 };
        assert!(p.check(0).is_err());
    }
}
