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

/// Elementwise functions of the model codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fun {
    Neg,
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Abs,
    Real,
    Imag,
    Conj,
    Arctan,
    /// Complete elliptic integral of the first kind K(m), parameter m (real).
    EllipK,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    /// `a > b` as 1.0 or 0.0 (real parts)
    Gt,
    /// `where(c, a, b)`: the third operand is the condition (real part != 0)
    Where,
    /// a complex value divided by a real one (the real part of `b`):
    /// `(re a / b, im a / b)`, two correctly rounded divisions, as numpy
    DivReal,
}

/// One instruction of a model code: `dst = f(a, b, c)` on registers, a
/// value (`Fun::Real` of nothing: see [`Instr::Param`]) or a constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Instr {
    /// `dst = value` (a strided value, as the fill entries read them)
    Param { dst: usize, value: usize },
    Const { dst: usize, re: f64, im: f64 },
    /// `dst = f(a, b, c)`; `real` when the result is real (numpy float64
    /// semantics: a real square root of a negative number is NaN)
    Op { dst: usize, f: Fun, a: usize, b: usize, c: usize, real: bool },
    /// `dst = exp(c0 + c1 t + c2 t^2)` where `t` advances by `dt` from one
    /// point to the next along the batch's last axis and the coefficients do
    /// not change along it (the tracer checks both): a true exponential at
    /// the start of every row segment and every [`EXP_RESEED`] points, one
    /// complex multiplication per point in between (two for `deg` 2).
    ExpPoly { dst: usize, t: usize, dt: usize, c0: usize, c1: usize, c2: usize, deg: u8 },
}

/// Points between two true exponentials of [`Instr::ExpPoly`]: the
/// recurrence adds about one rounding error per point to the relative error.
pub const EXP_RESEED: usize = 32;

/// Lanes per tile of a model code's evaluation: the registers of a tile
/// (16 bytes per register and lane) stay in the core's own cache.
pub const MODEL_TILE: usize = 128;

/// A model evaluated inside the solve: an elementwise code per batch point
/// that writes entries of an initial block (the block's `fill` entries, if
/// any, are written first).
#[derive(Clone, Debug)]
pub struct ModelBlock {
    pub block: usize,
    pub code: Vec<Instr>,
    pub n_regs: usize,
    /// `(out, in, register)`: block entries taken from registers
    pub outputs: Vec<(usize, usize, usize)>,
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
    /// Initial blocks computed by model codes, at most one per block.
    pub models: Vec<ModelBlock>,
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
        let mut seen = vec![false; self.n_init];
        for (e, m) in self.models.iter().enumerate() {
            if m.block >= self.n_init || seen[m.block] {
                return Err(ProgramError::Entry(e));
            }
            seen[m.block] = true;
            let n = self.sizes[m.block];
            for ins in &m.code {
                let ok = match *ins {
                    Instr::Param { dst, value } => dst < m.n_regs && value < n_values,
                    Instr::Const { dst, .. } => dst < m.n_regs,
                    Instr::Op { dst, a, b, c, .. } => dst < m.n_regs && a < m.n_regs && b < m.n_regs && c < m.n_regs,
                    Instr::ExpPoly { dst, t, dt, c0, c1, c2, deg } => [dst, t, dt, c0, c1, c2].iter().all(|&r| r < m.n_regs) && (1..=2).contains(&deg),
                };
                if !ok {
                    return Err(ProgramError::Entry(e));
                }
            }
            if m.outputs.iter().any(|&(o, i, r)| o >= n || i >= n || r >= m.n_regs) {
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

/// The numbers of a value: complex, or real (read as complex numbers with a
/// zero imaginary part, the same as converting them first, without the copy).
#[derive(Clone, Copy)]
pub enum Data<'a> {
    Complex(&'a [Complex64]),
    Real(&'a [f64]),
}

/// One number of a value's data, read as its real and imaginary parts.
trait Elem: Copy {
    fn parts(self) -> (f64, f64);
}

impl Elem for Complex64 {
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

impl Elem for f64 {
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

/// `re[k], im[k] = src[base + k * st]` for every `k`.
#[inline(always)]
fn read_strided<E: Elem>(src: &[E], base: usize, st: usize, re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if st == 0 {
        let (x, y) = src[base].parts();
        re.iter_mut().for_each(|r| *r = x);
        im[..n].iter_mut().for_each(|r| *r = y);
    } else if st == 1 {
        let src = &src[base..base + n];
        for k in 0..n {
            let (x, y) = src[k].parts();
            re[k] = x;
            im[k] = y;
        }
    } else {
        for k in 0..n {
            let (x, y) = src[base + k * st].parts();
            re[k] = x;
            im[k] = y;
        }
    }
}

impl Data<'_> {
    /// Element `i` as (real part, imaginary part).
    #[inline(always)]
    fn at(&self, i: usize) -> (f64, f64) {
        match self {
            Data::Complex(d) => d[i].parts(),
            Data::Real(d) => d[i].parts(),
        }
    }

    /// See [`read_strided`].
    #[inline(always)]
    fn read(&self, base: usize, st: usize, re: &mut [f64], im: &mut [f64]) {
        match self {
            Data::Complex(d) => read_strided(d, base, st, re, im),
            Data::Real(d) => read_strided(d, base, st, re, im),
        }
    }
}

/// A value read through strides over the batch shape.
pub struct Value<'a> {
    pub data: Data<'a>,
    /// Offset of the first element in `data`.
    pub offset: usize,
    /// Stride (in elements) along every axis of the batch shape; 0 where
    /// the value is broadcast. Negative strides are not supported.
    pub strides: Vec<usize>,
}

/// Run `program` at every point of a batch of shape `shape` (row-major).
/// `out` is `(n_out, B)` row-major: slot `s` of point `b` is
/// `out[s * B + b]`; slots that no output writes are set to zero. Runs on
/// the current rayon pool.
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
    // every chunk writes its points of every output slot; only the slots no
    // output names are cleared here (clearing all of `out` in this thread
    // first touched every page of a fresh output in one thread: 320 MB for
    // a 16-slot output of 1.25 million points, most of a call's time)
    let mut written = vec![false; program.n_out];
    for &(s, ..) in &program.outputs {
        written[s] = true;
    }
    for (s, w) in written.iter().enumerate() {
        if !w {
            out[s * nb..(s + 1) * nb].iter_mut().for_each(|x| *x = Complex64::new(0.0, 0.0));
        }
    }
    run_zeroed(program, values, shape, out);
}

/// [`run`] into an output that is already zero (a freshly allocated zeroed
/// buffer): the slots that no output names are left untouched. Clearing them
/// in the calling thread first touched every page of half a fresh output in
/// one thread (8 unwritten slots of 1.25 million points are 160 MB, 10 to
/// 12 ms per call on a 64-core x86 host, a third of a call on 32 threads);
/// left alone, the pages of a calloc'd buffer are never touched unless read.
/// The written slots are the same as [`run`]'s, bit for bit.
pub fn run_zeroed(program: &Program, values: &[Value], shape: &[usize], out: &mut [Complex64]) {
    let nb: usize = shape.iter().product();
    assert_eq!(out.len(), program.n_out * nb);
    for v in values {
        assert_eq!(v.strides.len(), shape.len());
    }
    if nb == 0 {
        return;
    }
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
    // the pool the chunks run on: the caller's when it is a worker, else
    // lightsim's (small work called from outside a pool runs in place, on
    // the chunks it would have on that pool)
    let threads = if rayon::current_thread_index().is_some() { rayon::current_num_threads() } else { crate::pool::pool().current_num_threads() }.max(1);
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
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok()).filter(|&v| v > 0);
    let reseed = env("LIGHTSIM_EXP_RESEED").unwrap_or(EXP_RESEED);
    // chunks start at multiples of the reseed interval: the true exponentials
    // of `Instr::ExpPoly` sit at row starts and at multiples of the interval
    // in the global point index, so the results do not depend on the chunking
    // (which follows the thread count)
    let lanes = lanes.div_ceil(reseed) * reseed;
    let mut model_of = vec![None; program.n_init];
    for (k, m) in program.models.iter().enumerate() {
        model_of[m.block] = Some(k);
    }
    // three scratch registers after every code's own
    let max_regs = program.models.iter().map(|m| m.n_regs + 3).max().unwrap_or(0);
    let plans: Vec<ModelPlan> = program.models.iter().map(|m| ModelPlan::new(m, values, shape.len())).collect();
    // measurement overrides (benchmarks/circuit/README.md)
    let tile = env("LIGHTSIM_MODEL_TILE").unwrap_or(MODEL_TILE).min(lanes);
    let ex = Exec { program, values, shape, offs: &offs, total, maxn, lanes, nb, before: &before, end: &end, fills: &fills, model_of: &model_of, plans: &plans, max_regs, tile, reseed };
    let nchunks = nb.div_ceil(lanes);
    let outp = SendPtr(out.as_mut_ptr());
    let level = simd_level();
    let one = |c: usize| {
        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            sc.fit(&ex);
            let b0 = c * lanes;
            let len = lanes.min(nb - b0);
            ex.chunk(level, &mut sc, b0, len, &outp);
        })
    };
    if is_small(program, nb) || nchunks == 1 {
        // too little work to wake other threads for
        (0..nchunks).for_each(one);
        return;
    }
    // one chunk per task, so that work stealing balances down to a chunk
    // (fast and slow cores, uneven chunks); every thread keeps its scratch
    // between chunks and calls (thread-local, grown when a call needs more)
    (0..nchunks).into_par_iter().for_each(|c| {
        SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            sc.fit(&ex);
            let b0 = c * lanes;
            let len = lanes.min(nb - b0);
            ex.chunk(level, &mut sc, b0, len, &outp);
        })
    });
}

/// Below this many complex multiply-adds (a few hundred microseconds on
/// one core) a call runs in the calling thread: handing it to a pool of
/// sleeping threads cost 0.1 to 0.5 ms per call on a GB10, more than the
/// work (single-frequency calls of small circuits were 2 to 4 times slower
/// than the dense solve on ten threads).
pub const SMALL_WORK: usize = 1 << 16;

/// Whether a call is small enough to run in the calling thread.
pub fn is_small(program: &Program, points: usize) -> bool {
    let work = program.cost() + 8 * program.models.iter().map(|m| m.code.len()).sum::<usize>();
    work.max(1).saturating_mul(points) < SMALL_WORK
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = const { std::cell::RefCell::new(Scratch { re: Vec::new(), im: Vec::new(), tre: Vec::new(), tim: Vec::new(), coord: Vec::new(), segs: Vec::new(), seg_base: Vec::new(), rre: Vec::new(), rim: Vec::new(), ure: Vec::new(), uim: Vec::new(), est: Vec::new(), pre: Vec::new(), pim: Vec::new(), sub: Vec::new(), b0: 0 }) };
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
    model_of: &'a [Option<usize>],
    plans: &'a [ModelPlan],
    max_regs: usize,
    tile: usize,
    reseed: usize,
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
    // registers of the model codes: a tile's lanes per register, one value
    // per segment for the uniform registers, the state of the exponential
    // recurrences per segment, the tile's pieces of segments
    rre: Vec<f64>,
    rim: Vec<f64>,
    ure: Vec<f64>,
    uim: Vec<f64>,
    est: Vec<Complex64>,
    pre: Vec<f64>,
    pim: Vec<f64>,
    sub: Vec<(usize, usize, usize, usize)>,
    // global index of the current chunk's first point
    b0: usize,
}

impl Scratch {
    /// Make the buffers large enough for `ex` (every entry is written before
    /// it is read, so old contents do not matter).
    fn fit(&mut self, ex: &Exec) {
        let l = ex.lanes;
        let nt = 2 * ex.maxn + 8;
        for (v, n) in [(&mut self.re, ex.total * l), (&mut self.im, ex.total * l), (&mut self.tre, nt * l), (&mut self.tim, nt * l), (&mut self.rre, ex.max_regs * ex.tile), (&mut self.rim, ex.max_regs * ex.tile)] {
            if v.len() < n {
                v.resize(n, 0.0);
            }
        }
        if self.coord.len() < ex.shape.len() {
            self.coord.resize(ex.shape.len(), 0);
        }
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
        sc.b0 = b0;
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
                val.data.read(base, st, &mut sc.re[e..e + seg], &mut sc.im[e..e + seg]);
            }
        }
        if let Some(m) = self.model_of[blk] {
            self.eval_model(sc, &p.models[m], &self.plans[m]);
        }
    }

    /// Evaluate a model code on the lanes of the current chunk and write its
    /// outputs into the block.
    ///
    /// Registers that do not change along a row segment (constants, values
    /// with a zero stride along the last axis, and functions of those only)
    /// are evaluated once per segment; the others in tiles of [`MODEL_TILE`]
    /// lanes, so that the registers of a tile stay in the core's cache. Every
    /// register gets the same arithmetic, in the same order, as one pass over
    /// the whole chunk would give it.
    #[inline(always)]
    fn eval_model(&self, sc: &mut Scratch, m: &ModelBlock, plan: &ModelPlan) {
        let l = self.lanes;
        let len = sc.segs.last().map(|&(k0, s)| k0 + s).unwrap_or(0);
        let nd = self.shape.len();
        let nv = self.values.len();
        let ns = sc.segs.len();
        let uni = &plan.uniform;
        let tl = self.tile;
        if sc.ure.len() < m.n_regs * ns {
            sc.ure.resize(m.n_regs * ns, 0.0);
            sc.uim.resize(m.n_regs * ns, 0.0);
        }
        if sc.est.len() < 3 * plan.n_exp * ns {
            sc.est.resize(3 * plan.n_exp * ns, Complex64::new(0.0, 0.0));
        }
        if sc.pre.len() < plan.n_exp * ns * self.reseed {
            sc.pre.resize(plan.n_exp * ns * self.reseed, 0.0);
            sc.pim.resize(plan.n_exp * ns * self.reseed, 0.0);
        }
        // the uniform registers, one value per segment
        for ins in &m.code {
            match *ins {
                Instr::Param { dst, value } if uni[dst] => {
                    for q in 0..ns {
                        let (x, y) = self.values[value].data.at(sc.seg_base[q * nv + value]);
                        sc.ure[dst * ns + q] = x;
                        sc.uim[dst * ns + q] = y;
                    }
                }
                Instr::Const { dst, re, im } => {
                    sc.ure[dst * ns..dst * ns + ns].iter_mut().for_each(|x| *x = re);
                    sc.uim[dst * ns..dst * ns + ns].iter_mut().for_each(|x| *x = im);
                }
                Instr::Op { dst, f, a, b, c, real } if uni[dst] => {
                    let (pr, pi) = (sc.ure.as_mut_ptr(), sc.uim.as_mut_ptr());
                    // SAFETY: registers are assigned once, so dst is none of a, b, c
                    unsafe { lane_op(f, real, sl(pr, a * ns, ns), sl(pi, a * ns, ns), sl(pr, b * ns, ns), sl(pi, b * ns, ns), sl(pr, c * ns, ns), sl(pi, c * ns, ns), slm(pr, dst * ns, ns), slm(pi, dst * ns, ns)) };
                }
                _ => {}
            }
        }
        let n = self.program.sizes[m.block];
        for &(o, i, r) in &m.outputs {
            if uni[r] {
                let e = (self.offs[m.block] + o * n + i) * l;
                for (q, &(k0, seg)) in sc.segs.iter().enumerate() {
                    sc.re[e + k0..e + k0 + seg].iter_mut().for_each(|x| *x = sc.ure[r * ns + q]);
                    sc.im[e + k0..e + k0 + seg].iter_mut().for_each(|x| *x = sc.uim[r * ns + q]);
                }
            }
        }
        if plan.all_uniform {
            return;
        }
        // the other registers, tile by tile
        let tmp = m.n_regs; // three scratch registers after the code's own
        let mut q0 = 0; // first segment that reaches into the tile
        let mut t0 = 0;
        while t0 < len {
            let tn = tl.min(len - t0);
            // the tile's pieces of segments: (first lane in the tile, lanes, segment, offset in the segment)
            sc.sub.clear();
            let mut q = q0;
            while q < ns {
                let (k0, seg) = sc.segs[q];
                if k0 >= t0 + tn {
                    break;
                }
                let a = k0.max(t0);
                let b = (k0 + seg).min(t0 + tn);
                if b > a {
                    sc.sub.push((a - t0, b - a, q, a - k0));
                }
                if k0 + seg <= t0 + tn {
                    q0 = q + 1;
                }
                q += 1;
            }
            let mut ei = 0; // index of the exponential recurrence
            for ins in &m.code {
                match *ins {
                    Instr::Param { dst, value } if !uni[dst] => {
                        let val = &self.values[value];
                        let st = if nd == 0 { 0 } else { val.strides[nd - 1] };
                        for &(k, cnt, q, j) in &sc.sub {
                            let base = sc.seg_base[q * nv + value] + j * st;
                            val.data.read(base, st, &mut sc.rre[dst * tl + k..dst * tl + k + cnt], &mut sc.rim[dst * tl + k..dst * tl + k + cnt]);
                        }
                    }
                    Instr::ExpPoly { dst, t, dt, c0, c1, c2, deg } if deg == 1 && uni[dt] && uni[c1] && uni[c2] => {
                        // the step factor r = exp(c1 dt) is one value per segment: the
                        // points after a true exponential e are e r^s with the powers
                        // r^s (s < reseed) tabulated once per segment, independent
                        // products instead of a chain of dependent ones
                        let kk = self.reseed;
                        for si in 0..sc.sub.len() {
                            let (k, cnt, q, j) = sc.sub[si];
                            let tb = (ei * ns + q) * kk;
                            if j == 0 {
                                let r = (Complex64::new(sc.ure[c1 * ns + q], sc.uim[c1 * ns + q]) * Complex64::new(sc.ure[dt * ns + q], sc.uim[dt * ns + q])).exp();
                                let mut pw = Complex64::new(1.0, 0.0);
                                for s in 0..kk {
                                    sc.pre[tb + s] = pw.re;
                                    sc.pim[tb + s] = pw.im;
                                    pw *= r;
                                }
                            }
                            let st = ei * ns + q;
                            let mut e = sc.est[3 * st];
                            let f0 = sc.b0 + sc.segs[q].0; // global index of the segment's first point
                            let mut s = 0;
                            while s < cnt {
                                let f = f0 + j + s;
                                let ph = f - f0.max(f - f % kk); // points since the last true exponential
                                if ph == 0 {
                                    let g = |r: usize| if uni[r] { Complex64::new(sc.ure[r * ns + q], sc.uim[r * ns + q]) } else { Complex64::new(sc.rre[r * tl + k + s], sc.rim[r * tl + k + s]) };
                                    let (a0, a1, tk) = (g(c0), g(c1), g(t));
                                    e = (a0 + a1 * tk + g(c2) * tk * tk).exp();
                                }
                                let m = (kk - f % kk).min(cnt - s);
                                let (dr, di) = (&mut sc.rre[dst * tl + k + s..dst * tl + k + s + m], &mut sc.rim[dst * tl + k + s..dst * tl + k + s + m]);
                                let (pr, pi) = (&sc.pre[tb + ph..tb + ph + m], &sc.pim[tb + ph..tb + ph + m]);
                                let (x, y) = (e.re, e.im);
                                for p in 0..m {
                                    let (u, v) = (pr[p], pi[p]);
                                    dr[p] = x * u - y * v;
                                    di[p] = x * v + y * u;
                                }
                                s += m;
                            }
                            sc.est[3 * st] = e;
                        }
                        ei += 1;
                    }
                    Instr::ExpPoly { dst, t, dt, c0, c1, c2, deg } => {
                        let reseed = self.reseed;
                        for si in 0..sc.sub.len() {
                            let (k, cnt, q, j) = sc.sub[si];
                            let st = 3 * (ei * ns + q);
                            let f0 = sc.b0 + sc.segs[q].0; // global index of the segment's first point
                            let fresh = |f: usize| f == f0 || f.is_multiple_of(reseed); // a true exponential here
                            let (mut e, mut rr, mut qq) = if fresh(f0 + j) { (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)) } else { (sc.est[st], sc.est[st + 1], sc.est[st + 2]) };
                            // runs of points between two true exponentials (no division per point)
                            let mut s = 0;
                            while s < cnt {
                                let f = f0 + j + s;
                                let mut stop = s + (reseed - f % reseed);
                                if fresh(f) {
                                    let g = |r: usize| if uni[r] { Complex64::new(sc.ure[r * ns + q], sc.uim[r * ns + q]) } else { Complex64::new(sc.rre[r * tl + k + s], sc.rim[r * tl + k + s]) };
                                    let (tk, h) = (g(t), g(dt));
                                    let (a0, a1, a2) = (g(c0), g(c1), g(c2));
                                    e = (a0 + a1 * tk + a2 * tk * tk).exp();
                                    if deg == 1 {
                                        rr = (a1 * h).exp();
                                    } else {
                                        rr = (a1 * h + a2 * h * (tk * 2.0 + h)).exp();
                                        qq = (a2 * h * h * 2.0).exp();
                                    }
                                    sc.rre[dst * tl + k + s] = e.re;
                                    sc.rim[dst * tl + k + s] = e.im;
                                    s += 1;
                                }
                                stop = stop.min(cnt);
                                let (dr, di) = (&mut sc.rre[dst * tl + k..dst * tl + k + cnt], &mut sc.rim[dst * tl + k..dst * tl + k + cnt]);
                                if deg == 1 {
                                    for p in s..stop {
                                        e *= rr;
                                        dr[p] = e.re;
                                        di[p] = e.im;
                                    }
                                } else {
                                    for p in s..stop {
                                        e *= rr;
                                        rr *= qq;
                                        dr[p] = e.re;
                                        di[p] = e.im;
                                    }
                                }
                                s = stop.max(s);
                            }
                            sc.est[st] = e;
                            sc.est[st + 1] = rr;
                            sc.est[st + 2] = qq;
                        }
                        ei += 1;
                    }
                    Instr::Op { dst, f, a, b, c, real } if !uni[dst] => {
                        let (pr, pi) = (sc.rre.as_mut_ptr(), sc.rim.as_mut_ptr());
                        let (ua, ub) = (uni[a] && f.arity() >= 1, uni[b] && f.arity() >= 2);
                        let (uc, va, vb) = (uni[c] && f.arity() >= 3, !uni[a], !uni[b]);
                        // SAFETY: registers are assigned once, so dst is none of a, b, c;
                        // the scratch registers lie after the code's own
                        unsafe {
                            if f.arity() == 2 && f.is_arith() && (ua != ub) {
                                // one operand constant along every segment
                                for &(k, cnt, q, _) in &sc.sub {
                                    let (dr, di) = (slm(pr, dst * tl + k, cnt), slm(pi, dst * tl + k, cnt));
                                    if ua {
                                        let (x, y) = (sc.ure[a * ns + q], sc.uim[a * ns + q]);
                                        scalar_left(f, real, x, y, sl(pr, b * tl + k, cnt), sl(pi, b * tl + k, cnt), dr, di);
                                    } else {
                                        let (u, v) = (sc.ure[b * ns + q], sc.uim[b * ns + q]);
                                        scalar_right(f, real, sl(pr, a * tl + k, cnt), sl(pi, a * tl + k, cnt), u, v, dr, di);
                                    }
                                }
                            } else {
                                // uniform operands spread over the tile first
                                let mut reg = [a, b, c];
                                for (w, used) in [ua, ub, uc].into_iter().enumerate() {
                                    if used {
                                        let r = reg[w];
                                        for &(k, cnt, q, _) in &sc.sub {
                                            slm(pr, (tmp + w) * tl + k, cnt).iter_mut().for_each(|x| *x = sc.ure[r * ns + q]);
                                            slm(pi, (tmp + w) * tl + k, cnt).iter_mut().for_each(|x| *x = sc.uim[r * ns + q]);
                                        }
                                        reg[w] = tmp + w;
                                    }
                                }
                                let _ = (va, vb);
                                let [a, b, c] = reg;
                                lane_op(f, real, sl(pr, a * tl, tn), sl(pi, a * tl, tn), sl(pr, b * tl, tn), sl(pi, b * tl, tn), sl(pr, c * tl, tn), sl(pi, c * tl, tn), slm(pr, dst * tl, tn), slm(pi, dst * tl, tn));
                            }
                        }
                    }
                    _ => {}
                }
            }
            for &(o, i, r) in &m.outputs {
                if !uni[r] {
                    let e = (self.offs[m.block] + o * n + i) * l + t0;
                    sc.re[e..e + tn].copy_from_slice(&sc.rre[r * tl..r * tl + tn]);
                    sc.im[e..e + tn].copy_from_slice(&sc.rim[r * tl..r * tl + tn]);
                }
            }
            t0 += tn;
        }
    }
}

/// A shared slice of `n` values at `p + o`.
///
/// # Safety
/// The range must lie inside the allocation and not be written while the slice lives.
#[inline(always)]
unsafe fn sl<'a>(p: *mut f64, o: usize, n: usize) -> &'a [f64] {
    unsafe { std::slice::from_raw_parts(p.add(o), n) }
}

/// A mutable slice of `n` values at `p + o`.
///
/// # Safety
/// The range must lie inside the allocation and overlap no other live slice.
#[inline(always)]
unsafe fn slm<'a>(p: *mut f64, o: usize, n: usize) -> &'a mut [f64] {
    unsafe { std::slice::from_raw_parts_mut(p.add(o), n) }
}

/// Per-call facts of a model code: which registers are constant along a row
/// segment (their inputs are constants or values with a zero stride along
/// the last axis), and the number of exponential recurrences.
struct ModelPlan {
    uniform: Vec<bool>,
    all_uniform: bool,
    n_exp: usize,
}

impl ModelPlan {
    fn new(m: &ModelBlock, values: &[Value], nd: usize) -> Self {
        let mut uniform = vec![false; m.n_regs];
        let mut n_exp = 0;
        for ins in &m.code {
            match *ins {
                Instr::Param { dst, value } => uniform[dst] = nd == 0 || values[value].strides[nd - 1] == 0,
                Instr::Const { dst, .. } => uniform[dst] = true,
                Instr::ExpPoly { .. } => n_exp += 1,
                Instr::Op { dst, f, a, b, c, .. } => {
                    let k = f.arity();
                    uniform[dst] = (k < 1 || uniform[a]) && (k < 2 || uniform[b]) && (k < 3 || uniform[c]);
                }
            }
        }
        let all_uniform = m.outputs.iter().all(|&(_, _, r)| uniform[r]);
        ModelPlan { uniform, all_uniform, n_exp }
    }
}

impl Fun {
    /// Number of operands.
    fn arity(self) -> usize {
        match self {
            Fun::Add | Fun::Sub | Fun::Mul | Fun::Div | Fun::DivReal | Fun::Pow | Fun::Max | Fun::Min | Fun::Gt => 2,
            Fun::Where => 3,
            _ => 1,
        }
    }

    fn is_arith(self) -> bool {
        matches!(self, Fun::Add | Fun::Sub | Fun::Mul | Fun::Div | Fun::DivReal)
    }
}

/// `d = (x + iy) f b` for the four arithmetic functions, the left operand
/// one value; the same arithmetic as [`lane_op`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scalar_left(f: Fun, real: bool, x: f64, y: f64, br: &[f64], bi: &[f64], dr: &mut [f64], di: &mut [f64]) {
    let n = dr.len();
    let (br, bi, di) = (&br[..n], &bi[..n], &mut di[..n]);
    if real {
        match f {
            Fun::Add => (0..n).for_each(|k| dr[k] = x + br[k]),
            Fun::Sub => (0..n).for_each(|k| dr[k] = x - br[k]),
            Fun::Mul => (0..n).for_each(|k| dr[k] = x * br[k]),
            _ => (0..n).for_each(|k| dr[k] = x / br[k]),
        }
        di.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    match f {
        Fun::Add => (0..n).for_each(|k| {
            dr[k] = x + br[k];
            di[k] = y + bi[k];
        }),
        Fun::Sub => (0..n).for_each(|k| {
            dr[k] = x - br[k];
            di[k] = y - bi[k];
        }),
        Fun::Mul => (0..n).for_each(|k| {
            let (u, v) = (br[k], bi[k]);
            dr[k] = x * u - y * v;
            di[k] = x * v + y * u;
        }),
        Fun::DivReal => (0..n).for_each(|k| {
            dr[k] = x / br[k];
            di[k] = y / br[k];
        }),
        _ => (0..n).for_each(|k| {
            let (u, v) = (br[k], bi[k]);
            let q = 1.0 / (u * u + v * v);
            dr[k] = (x * u + y * v) * q;
            di[k] = (y * u - x * v) * q;
        }),
    }
}

/// `d = a f (u + iv)` for the four arithmetic functions, the right operand
/// one value; the same arithmetic as [`lane_op`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scalar_right(f: Fun, real: bool, ar: &[f64], ai: &[f64], u: f64, v: f64, dr: &mut [f64], di: &mut [f64]) {
    let n = dr.len();
    let (ar, ai, di) = (&ar[..n], &ai[..n], &mut di[..n]);
    if real {
        match f {
            Fun::Add => (0..n).for_each(|k| dr[k] = ar[k] + u),
            Fun::Sub => (0..n).for_each(|k| dr[k] = ar[k] - u),
            Fun::Mul => (0..n).for_each(|k| dr[k] = ar[k] * u),
            _ => (0..n).for_each(|k| dr[k] = ar[k] / u),
        }
        di.iter_mut().for_each(|w| *w = 0.0);
        return;
    }
    match f {
        Fun::Add => (0..n).for_each(|k| {
            dr[k] = ar[k] + u;
            di[k] = ai[k] + v;
        }),
        Fun::Sub => (0..n).for_each(|k| {
            dr[k] = ar[k] - u;
            di[k] = ai[k] - v;
        }),
        Fun::Mul => (0..n).for_each(|k| {
            let (x, y) = (ar[k], ai[k]);
            dr[k] = x * u - y * v;
            di[k] = x * v + y * u;
        }),
        Fun::DivReal => (0..n).for_each(|k| {
            dr[k] = ar[k] / u;
            di[k] = ai[k] / u;
        }),
        _ => {
            let q = 1.0 / (u * u + v * v);
            (0..n).for_each(|k| {
                let (x, y) = (ar[k], ai[k]);
                dr[k] = (x * u + y * v) * q;
                di[k] = (y * u - x * v) * q;
            })
        }
    }
}

/// One instruction over the lanes: tight loops for the common functions,
/// the scalar functions below for the rest.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn lane_op(f: Fun, real: bool, ar: &[f64], ai: &[f64], br: &[f64], bi: &[f64], cr: &[f64], ci: &[f64], dr: &mut [f64], di: &mut [f64]) {
    let n = dr.len();
    // equal lengths, stated, so that the loops below vectorise without bounds checks
    let (ar, ai, br, bi, cr, ci, di) = (&ar[..n], &ai[..n], &br[..n], &bi[..n], &cr[..n], &ci[..n], &mut di[..n]);
    if real {
        match f {
            Fun::Add => (0..n).for_each(|k| dr[k] = ar[k] + br[k]),
            Fun::Sub => (0..n).for_each(|k| dr[k] = ar[k] - br[k]),
            Fun::Mul => (0..n).for_each(|k| dr[k] = ar[k] * br[k]),
            Fun::Div | Fun::DivReal => (0..n).for_each(|k| dr[k] = ar[k] / br[k]),
            Fun::Neg => (0..n).for_each(|k| dr[k] = -ar[k]),
            Fun::Exp if ar.iter().all(|x| x.abs() <= EXP_FAST_MAX) => (0..n).for_each(|k| dr[k] = exp_fast(ar[k])),
            _ => (0..n).for_each(|k| dr[k] = real_fun(f, ar[k], br[k], cr[k])),
        }
        di.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    match f {
        Fun::Add => (0..n).for_each(|k| {
            dr[k] = ar[k] + br[k];
            di[k] = ai[k] + bi[k];
        }),
        Fun::Sub => (0..n).for_each(|k| {
            dr[k] = ar[k] - br[k];
            di[k] = ai[k] - bi[k];
        }),
        Fun::Mul => (0..n).for_each(|k| {
            let (x, y, u, v) = (ar[k], ai[k], br[k], bi[k]);
            dr[k] = x * u - y * v;
            di[k] = x * v + y * u;
        }),
        Fun::Div => (0..n).for_each(|k| {
            // (x + iy) / (u + iv), the textbook formula with the reciprocal
            // of |z|^2 (vectorises; differs from Smith's algorithm, which
            // numpy uses, only in rounding for operands of ordinary size)
            let (x, y, u, v) = (ar[k], ai[k], br[k], bi[k]);
            let q = 1.0 / (u * u + v * v);
            dr[k] = (x * u + y * v) * q;
            di[k] = (y * u - x * v) * q;
        }),
        Fun::DivReal => (0..n).for_each(|k| {
            dr[k] = ar[k] / br[k];
            di[k] = ai[k] / br[k];
        }),
        Fun::Neg => (0..n).for_each(|k| {
            dr[k] = -ar[k];
            di[k] = -ai[k];
        }),
        Fun::Exp => cexp_lanes(ar, ai, dr, di),
        _ => (0..n).for_each(|k| {
            let z = complex_fun(f, Complex64::new(ar[k], ai[k]), Complex64::new(br[k], bi[k]), Complex64::new(cr[k], ci[k]));
            dr[k] = z.re;
            di[k] = z.im;
        }),
    }
}

/// Largest |x| of [`exp_fast`] (normal results only).
const EXP_FAST_MAX: f64 = 708.0;
/// Largest |x| of [`sincos_fast`] (exact Cody-Waite products up to 2^19 quadrants).
const SINCOS_FAST_MAX: f64 = 4.0e5;

/// `exp(x)` for |x| <= [`EXP_FAST_MAX`], without branches (vectorises):
/// x = n ln 2 + r with |r| <= ln 2 / 2, a degree-13 Taylor polynomial of
/// e^r (truncation below 1e-17 relative), scaled by 2^n. Within 2 ulp of
/// the correctly rounded result.
#[inline(always)]
fn exp_fast(x: f64) -> f64 {
    const MAGIC: f64 = f64::from_bits(0x4338000000000000); // 6755399441055744.0 // 1.5 * 2^52: rounds to an integer in the low bits
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = f64::from_bits(0x3fe62e42fee00000); // 6.93147180369123816490e-1
    const LN2_LO: f64 = f64::from_bits(0x3dea39ef35793c76); // 1.90821492927058770002e-10
    let t = x * LOG2E + MAGIC;
    let nf = t - MAGIC;
    let n = (t.to_bits() as i64).wrapping_sub(MAGIC.to_bits() as i64);
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    // Horner, 1/k! coefficients
    let mut p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f64::from_bits(((n + 1023) as u64) << 52)
}

/// `(sin y, cos y)` for |y| <= [`SINCOS_FAST_MAX`], without branches
/// (vectorises): y = k pi/2 + r by a three-part Cody-Waite reduction (the
/// products with the first two parts exact), |r| <= pi/4, and fdlibm's
/// kernel polynomials. Within 2 ulp of the correctly rounded results.
#[inline(always)]
fn sincos_fast(y: f64) -> (f64, f64) {
    const MAGIC: f64 = f64::from_bits(0x4338000000000000); // 6755399441055744.0
    const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
    const P1: f64 = f64::from_bits(0x3ff921fb54400000); // 1.57079632673412561417
    const P2: f64 = f64::from_bits(0x3dd0b4611a600000); // 6.07710050630396597660e-11
    const P3: f64 = f64::from_bits(0x3ba3198a2e037073); // 2.02226624879595063154e-21
    let t = y * TWO_OVER_PI + MAGIC;
    let kf = t - MAGIC;
    let q = t.to_bits() & 3;
    let r = ((y - kf * P1) - kf * P2) - kf * P3;
    let z = r * r;
    // fdlibm __kernel_sin / __kernel_cos
    const S1: f64 = f64::from_bits(0xbfc5555555555549); // -1.66666666666666324348e-01
    const S2: f64 = f64::from_bits(0x3f8111111110f8a6); // 8.33333333332248946183e-03
    const S3: f64 = f64::from_bits(0xbf2a01a019c161d5); // -1.98412698298579493134e-04
    const S4: f64 = f64::from_bits(0x3ec71de357b1fe7d); // 2.75573137070700676789e-06
    const S5: f64 = f64::from_bits(0xbe5ae5e68a2b9ceb); // -2.50507602534068634195e-08
    const S6: f64 = f64::from_bits(0x3de5d93a5acfd57d); // 1.58969099521155038407e-10
    const C1: f64 = f64::from_bits(0x3fa555555555554c); // 4.16666666666666019037e-02
    const C2: f64 = f64::from_bits(0xbf56c16c16c15177); // -1.38888888888741095749e-03
    const C3: f64 = f64::from_bits(0x3efa01a019cb1590); // 2.48015872894767294178e-05
    const C4: f64 = f64::from_bits(0xbe927e4f809c52ad); // -2.75573143513906633035e-07
    const C5: f64 = f64::from_bits(0x3e21ee9ebdb4b1c4); // 2.08757232129817482790e-09
    const C6: f64 = f64::from_bits(0xbda8fae9be8838d4); // -1.13596475577881948255e-11
    let sp = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    let sn = r + r * z * (S1 + z * sp);
    let cp = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    let cs = w + (((1.0 - w) - hz) + z * cp);
    let (s0, c0) = if q & 1 == 0 { (sn, cs) } else { (cs, -sn) };
    if q & 2 == 0 { (s0, c0) } else { (-s0, -c0) }
}

/// `exp(a)` over lanes of complex values: the branch-free functions above
/// when every argument is in their range, the library functions otherwise.
#[inline(always)]
fn cexp_lanes(ar: &[f64], ai: &[f64], dr: &mut [f64], di: &mut [f64]) {
    let n = dr.len();
    let (ar, ai, di) = (&ar[..n], &ai[..n], &mut di[..n]);
    if ar.iter().all(|x| x.abs() <= EXP_FAST_MAX) && ai.iter().all(|y| y.abs() <= SINCOS_FAST_MAX) {
        for k in 0..n {
            let e = exp_fast(ar[k]);
            let (s, c) = sincos_fast(ai[k]);
            dr[k] = e * c;
            di[k] = e * s;
        }
    } else {
        for k in 0..n {
            let e = ar[k].exp();
            let (s, c) = ai[k].sin_cos();
            dr[k] = e * c;
            di[k] = e * s;
        }
    }
}

/// K(m) by the arithmetic-geometric mean (scipy's parameter convention):
/// NaN above 1, infinity at 1.
fn ellipk(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return f64::INFINITY;
    }
    let (mut a, mut b) = (1.0f64, (1.0 - m).sqrt());
    for _ in 0..64 {
        if (a - b).abs() <= 1e-16 * a {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    std::f64::consts::PI / (a + b)
}

/// A function on real values (numpy float64 semantics).
#[inline(always)]
fn real_fun(f: Fun, a: f64, b: f64, c: f64) -> f64 {
    match f {
        Fun::Neg => -a,
        Fun::Exp => a.exp(),
        Fun::Log => a.ln(),
        Fun::Sqrt => a.sqrt(),
        Fun::Sin => a.sin(),
        Fun::Cos => a.cos(),
        Fun::Tan => a.tan(),
        Fun::Sinh => a.sinh(),
        Fun::Cosh => a.cosh(),
        Fun::Tanh => a.tanh(),
        Fun::Abs => a.abs(),
        Fun::Real | Fun::Conj => a,
        Fun::Imag => 0.0,
        Fun::Arctan => a.atan(),
        Fun::EllipK => ellipk(a),
        Fun::Add => a + b,
        Fun::Sub => a - b,
        Fun::Mul => a * b,
        Fun::Div | Fun::DivReal => a / b,
        Fun::Pow => {
            if b == 2.0 {
                a * a
            } else if b == 0.5 {
                a.sqrt()
            } else {
                a.powf(b)
            }
        }
        Fun::Max => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        Fun::Min => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
        Fun::Gt => {
            if a > b {
                1.0
            } else {
                0.0
            }
        }
        Fun::Where => {
            if c != 0.0 {
                a
            } else {
                b
            }
        }
    }
}

/// A function on complex values (results complex; `Abs`, `Real` and `Imag`
/// return real values with a zero imaginary part).
#[inline(always)]
fn complex_fun(f: Fun, a: Complex64, b: Complex64, c: Complex64) -> Complex64 {
    let r = |x: f64| Complex64::new(x, 0.0);
    match f {
        Fun::Neg => -a,
        Fun::Exp => a.exp(),
        Fun::Log => a.ln(),
        Fun::Sqrt => a.sqrt(),
        Fun::Sin => a.sin(),
        Fun::Cos => a.cos(),
        Fun::Tan => a.tan(),
        Fun::Sinh => a.sinh(),
        Fun::Cosh => a.cosh(),
        Fun::Tanh => a.tanh(),
        Fun::Abs => r(a.norm()),
        Fun::Real => r(a.re),
        Fun::Imag => r(a.im),
        Fun::Conj => a.conj(),
        Fun::Arctan => a.atan(),
        Fun::EllipK => r(ellipk(a.re)),
        Fun::Add => a + b,
        Fun::Sub => a - b,
        Fun::Mul => a * b,
        Fun::Div => a / b,
        Fun::DivReal => Complex64::new(a.re / b.re, a.im / b.re),
        Fun::Pow => {
            if b == r(2.0) {
                a * a
            } else {
                a.powc(b)
            }
        }
        Fun::Max | Fun::Min | Fun::Gt => r(real_fun(f, a.re, b.re, 0.0)),
        Fun::Where => {
            if c.re != 0.0 || c.im != 0.0 {
                a
            } else {
                b
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
        let p = Program { sizes: vec![3, 2, 4, 3, 5, 3], n_init: 3, ops, fill, outputs, n_out: 9, models: vec![] };
        p.check(vals.len()).unwrap();
        let values: Vec<Value> = vals.iter().map(|v| Value { data: Data::Complex(std::slice::from_ref(v)), offset: 0, strides: vec![0] }).collect();
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
        let values = vec![Value { data: Data::Complex(&t), offset: 0, strides: vec![0, 1] }, Value { data: Data::Complex(&r), offset: 0, strides: vec![1, 0] }];
        let fill = vec![(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)];
        let outputs = vec![(0, 0, 1, 0), (1, 0, 0, 0)];
        let p = Program { sizes: vec![2], n_init: 1, ops: vec![], fill, outputs, n_out: 3, models: vec![] };
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
        let p = Program { sizes: vec![3, 2, 4, 3, 5, 3], n_init: 3, ops, fill: fill.clone(), outputs, n_out: 9, models: vec![] };
        p.check(data.len()).unwrap();
        let values: Vec<Value> = data.iter().zip(&strides).map(|(d, st)| Value { data: Data::Complex(d), offset: 0, strides: st.clone() }).collect();
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
    fn real_values_and_zeroed_outputs_match() {
        // the network of joins_match_dense_solve on a (5, 37) batch: values with
        // zero imaginary parts given as real arrays give the same bits as given
        // as complex ones, and run_zeroed the same bits as run (the slots no
        // output names left zero); one entry strided, one broadcast
        let sizes0 = [3usize, 2, 4];
        let (d0, d1) = (5usize, 37usize);
        let np_ = d0 * d1;
        let mut rng = Lcg(11);
        let mut re: Vec<Vec<f64>> = vec![];
        let mut fill = vec![];
        let mut strides = vec![];
        for (blk, &m) in sizes0.iter().enumerate() {
            for o in 0..m {
                for i in 0..m {
                    let (len, st) = match (blk, o, i) {
                        (2, 0, 1) => (d1, vec![0, 1]),
                        (0, 1, 2) => (2 * np_, vec![2 * d1, 2]),
                        _ => (np_, vec![d1, 1]),
                    };
                    re.push((0..len).map(|_| 0.4 * rng.next()).collect());
                    strides.push(st);
                    fill.push((blk, o, i, re.len() - 1));
                }
            }
        }
        let cx: Vec<Vec<Complex64>> = re.iter().map(|v| v.iter().map(|&x| c(x, 0.0)).collect()).collect();
        let ops = vec![Op::Merge { a: 0, ia: 1, b: 1, ib: 0, out: 3 }, Op::Merge { a: 3, ia: 2, b: 2, ib: 0, out: 4 }, Op::Join { a: 4, i: 1, j: 2, out: 5 }];
        // slots 9 and 10 are named by no output
        let mut outputs = vec![];
        for i in 0..3 {
            for j in 0..3 {
                outputs.push((i * 3 + j, 5, i, j));
            }
        }
        let p = Program { sizes: vec![3, 2, 4, 3, 5, 3], n_init: 3, ops, fill, outputs, n_out: 11, models: vec![] };
        p.check(re.len()).unwrap();
        let vc: Vec<Value> = cx.iter().zip(&strides).map(|(d, st)| Value { data: Data::Complex(d), offset: 0, strides: st.clone() }).collect();
        let vr: Vec<Value> = re.iter().zip(&strides).map(|(d, st)| Value { data: Data::Real(d), offset: 0, strides: st.clone() }).collect();
        let mut a = vec![c(7.0, 7.0); 11 * np_];
        run(&p, &vc, &[d0, d1], &mut a);
        let mut b = vec![c(0.0, 0.0); 11 * np_];
        run_zeroed(&p, &vr, &[d0, d1], &mut b);
        assert!(a[9 * np_..].iter().all(|x| x.re == 0.0 && x.im == 0.0));
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits(), "element {k}: {x} != {y}");
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
        let p = Program { sizes, n_init: m, ops, fill, outputs, n_out: 4, models: vec![] };
        p.check(vals.len()).unwrap();
        let (_, total) = p.layout();
        assert!(total <= 16, "scratch {total}");
        let values: Vec<Value> = vals.iter().map(|v| Value { data: Data::Complex(std::slice::from_ref(v)), offset: 0, strides: vec![] }).collect();
        let mut out = vec![c(0.0, 0.0); 4];
        run(&p, &values, &[], &mut out);
        for k in 0..4 {
            let d = (out[k] - reference[k]).norm();
            assert!(d < 1e-12 * (1.0 + reference[k].norm()), "slot {k}: {d}");
        }
    }

    #[test]
    fn model_codes_match_direct_evaluation() {
        // a two-port whose transmission is t = a exp(i 2 pi n L / wl) with wl
        // varying along the batch and n per row; the reflection is
        // sqrt(x) (real, NaN where x < 0) and K(m) through `where`
        let (d0, d1) = (3usize, 50usize);
        let wl: Vec<Complex64> = (0..d1).map(|k| c(1.5 + 0.002 * k as f64, 0.0)).collect();
        let nidx: Vec<Complex64> = (0..d0).map(|r| c(2.4 + 0.1 * r as f64, 1e-4)).collect();
        let xs: Vec<Complex64> = (0..d1).map(|k| c(-0.5 + 0.02 * k as f64, 0.0)).collect();
        let values = vec![
            Value { data: Data::Complex(&wl), offset: 0, strides: vec![0, 1] },
            Value { data: Data::Complex(&nidx), offset: 0, strides: vec![1, 0] },
            Value { data: Data::Complex(&xs), offset: 0, strides: vec![0, 1] },
        ];
        let tau = 2.0 * std::f64::consts::PI;
        let code = vec![
            Instr::Param { dst: 0, value: 0 },
            Instr::Param { dst: 1, value: 1 },
            Instr::Const { dst: 2, re: 0.0, im: tau * 10.0 },
            Instr::Op { dst: 3, f: Fun::Mul, a: 2, b: 1, c: 0, real: false },
            Instr::Op { dst: 4, f: Fun::Div, a: 3, b: 0, c: 0, real: false },
            Instr::Op { dst: 5, f: Fun::Exp, a: 4, b: 0, c: 0, real: false },
            Instr::Const { dst: 6, re: 0.9, im: 0.0 },
            Instr::Op { dst: 7, f: Fun::Mul, a: 6, b: 5, c: 0, real: false },
            Instr::Param { dst: 8, value: 2 },
            Instr::Op { dst: 9, f: Fun::Sqrt, a: 8, b: 0, c: 0, real: true },
            Instr::Op { dst: 10, f: Fun::EllipK, a: 8, b: 0, c: 0, real: true },
            Instr::Op { dst: 11, f: Fun::Gt, a: 8, b: 6, c: 0, real: true },
            Instr::Op { dst: 12, f: Fun::Where, a: 10, b: 9, c: 11, real: true },
        ];
        let models = vec![ModelBlock { block: 0, code, n_regs: 13, outputs: vec![(1, 0, 7), (0, 1, 7), (0, 0, 12)] }];
        let outputs = vec![(0, 0, 1, 0), (1, 0, 0, 0)];
        let p = Program { sizes: vec![2], n_init: 1, ops: vec![], fill: vec![], outputs, n_out: 2, models };
        p.check(3).unwrap();
        let mut out = vec![c(0.0, 0.0); 2 * d0 * d1];
        run(&p, &values, &[d0, d1], &mut out);
        for r in 0..d0 {
            for k in 0..d1 {
                let t = c(0.9, 0.0) * (c(0.0, tau * 10.0) * nidx[r] / wl[k]).exp();
                let d = (out[r * d1 + k] - t).norm();
                assert!(d < 1e-13, "({r}, {k}): {d}");
                let x = xs[k].re;
                let refl = if x > 0.9 { ellipk(x) } else { x.sqrt() };
                let got = out[d0 * d1 + r * d1 + k];
                assert!(got.im == 0.0 && (got.re == refl || (got.re.is_nan() && refl.is_nan())), "({r}, {k}): {got} against {refl}");
            }
        }
        // K(m) against a reference value (m = 0.5: 1.8540746773013719)
        assert!((ellipk(0.5) - 1.854_074_677_301_371_9).abs() < 1e-15);
    }

    #[test]
    fn fast_exp_and_sincos_within_two_ulp() {
        let mut g = Lcg(7);
        let ulp = |x: f64| f64::from_bits(x.abs().to_bits() + 1) - x.abs();
        for k in 0..200_000 {
            // arguments spread over every scale of their ranges
            let u = g.next() * 2.0; // in [-1, 1)
            let x = u * EXP_FAST_MAX * if k % 3 == 0 { 1.0 } else { 1e-3 };
            let (e, r) = (exp_fast(x), x.exp());
            assert!((e - r).abs() <= 2.0 * ulp(r), "exp({x}): {e} against {r}");
            let y = u * SINCOS_FAST_MAX * [1.0, 1e-3, 1e-5, 1e-6][k % 4];
            let (s, c) = sincos_fast(y);
            let (sr, cr) = y.sin_cos();
            // 2 ulp of 1 (the size of sin and cos away from their zeros)
            assert!((s - sr).abs() <= 2.0 * ulp(1.0) && (c - cr).abs() <= 2.0 * ulp(1.0), "sincos({y}): {s} {c} against {sr} {cr}");
        }
    }

    #[test]
    fn check_rejects_bad_programs() {
        let p = Program { sizes: vec![2, 2, 2], n_init: 2, ops: vec![Op::Merge { a: 0, ia: 0, b: 0, ib: 1, out: 2 }], fill: vec![], outputs: vec![], n_out: 0, models: vec![] };
        assert!(p.check(0).is_err());
        let p = Program { sizes: vec![2, 2, 3], n_init: 2, ops: vec![Op::Merge { a: 0, ia: 0, b: 1, ib: 1, out: 2 }], fill: vec![], outputs: vec![], n_out: 0, models: vec![] };
        assert!(p.check(0).is_err());
    }
}
