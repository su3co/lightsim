//! Binding of the circuit network solve (`lightsim_core::circuit`), used by
//! `lightsim.circuit`. The program arrives as plain integer arrays and the
//! values as complex arrays already broadcast to the batch shape (views with
//! zero strides are read in place, not copied).

use lightsim_core::circuit::{Fun, Instr, ModelBlock, Op, Program, Value, run};
use num_complex::Complex64;
use numpy::{PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use crate::Compute;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// A pool of `threads` threads, kept for the life of the process (and built
/// again in a forked child, whose copies of the parent's pools have no threads).
fn pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    type Pools = (u32, HashMap<usize, Arc<rayon::ThreadPool>>);
    static POOLS: OnceLock<Mutex<Pools>> = OnceLock::new();
    let mut g = POOLS.get_or_init(Default::default).lock().unwrap();
    let pid = std::process::id();
    if g.0 != pid {
        // the parent's pools: never dropped here (their threads do not exist)
        std::mem::forget(std::mem::take(&mut g.1));
        g.0 = pid;
    }
    let m = &mut g.1;
    if let Some(p) = m.get(&threads) {
        return Ok(p.clone());
    }
    let p = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(|e| PyValueError::new_err(e.to_string()))?);
    m.insert(threads, p.clone());
    Ok(p)
}

struct Raw {
    ptr: *const Complex64,
    len: usize,
    strides: Vec<usize>,
}
// the arrays are borrowed (read-only) for the duration of the call
unsafe impl Send for Raw {}
unsafe impl Sync for Raw {}

fn rows(a: &PyReadonlyArray2<i64>, what: &str) -> PyResult<Vec<[usize; 4]>> {
    let v = a.as_array();
    if v.ncols() != 4 && v.nrows() > 0 {
        return Err(PyValueError::new_err(format!("{what} must have 4 columns")));
    }
    v.rows()
        .into_iter()
        .map(|r| {
            let mut x = [0usize; 4];
            for k in 0..4 {
                x[k] = usize::try_from(r[k]).map_err(|_| PyValueError::new_err(format!("{what}: negative index")))?;
            }
            Ok(x)
        })
        .collect()
}

/// The functions of the model codes, by the code `kind - 2` (the order of
/// `lightsim.circuit._FUNS`).
const FUNS: [Fun; 26] = [
    Fun::Neg, Fun::Exp, Fun::Log, Fun::Sqrt, Fun::Sin, Fun::Cos, Fun::Tan, Fun::Sinh, Fun::Cosh, Fun::Tanh, Fun::Abs, Fun::Real, Fun::Imag, Fun::Conj, Fun::Arctan, Fun::EllipK,
    Fun::Add, Fun::Sub, Fun::Mul, Fun::Div, Fun::Pow, Fun::Max, Fun::Min, Fun::Gt, Fun::Where, Fun::DivReal,
];

/// A model block from arrays: `code` is `(K, 9)` rows `(kind, dst, a, b, c,
/// real, d, e, f)` with kind 0 = parameter (value `a`), 1 = constant (row of
/// `consts`, `(K, 2)` real and imaginary parts), 2 + i = function `FUNS[i]`
/// on `a, b, c`, 100 = `exp(c0 + c1 t + c2 t^2)` on a uniform grid with
/// `t = a, dt = b, c0 = c, c1 = d, c2 = e` and degree `f`;
/// `outputs` is `(M, 3)` rows `(out, in, register)`.
fn model_block(block: usize, code: &PyReadonlyArray2<i64>, consts: &PyReadonlyArray2<f64>, outs: &PyReadonlyArray2<i64>, n_regs: usize) -> PyResult<ModelBlock> {
    let cv = code.as_array();
    let kv = consts.as_array();
    let bad = || PyValueError::new_err("model code: bad instruction");
    let u = |x: i64| usize::try_from(x).map_err(|_| bad());
    let mut instrs = Vec::with_capacity(cv.nrows());
    for (k, r) in cv.rows().into_iter().enumerate() {
        instrs.push(match r[0] {
            0 => Instr::Param { dst: u(r[1])?, value: u(r[2])? },
            1 => Instr::Const { dst: u(r[1])?, re: *kv.get((k, 0)).ok_or_else(bad)?, im: *kv.get((k, 1)).ok_or_else(bad)? },
            100 => Instr::ExpPoly { dst: u(r[1])?, t: u(r[2])?, dt: u(r[3])?, c0: u(r[4])?, c1: u(r[6])?, c2: u(r[7])?, deg: u8::try_from(r[8]).map_err(|_| bad())? },
            f => Instr::Op { dst: u(r[1])?, f: *FUNS.get(u(f - 2)?).ok_or_else(bad)?, a: u(r[2])?, b: u(r[3])?, c: u(r[4])?, real: r[5] != 0 },
        });
    }
    let outputs = outs.as_array().rows().into_iter().map(|r| Ok((u(r[0])?, u(r[1])?, u(r[2])?))).collect::<PyResult<Vec<_>>>()?;
    Ok(ModelBlock { block, code: instrs, n_regs, outputs })
}

/// A model block from Python: `(block, code, consts, outputs, registers)`.
type ModelArgs<'py> = (usize, PyReadonlyArray2<'py, i64>, PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, i64>, usize);

/// Run a network program over a batch. `ops` is `(K, 6)`: `(kind, a, i, b,
/// j, out)` with kind 0 a merge of port `i` of block `a` with port `j` of
/// block `b`, kind 1 a join of ports `i` and `j` of block `a` (`b` unused).
/// `fill` and `outputs` are `(F, 4)` and `(O, 4)` as in
/// `lightsim_core::circuit::Program`. Every value must have the batch
/// `shape` (broadcast views are fine). With `packed`, a `(V, prod(shape))`
/// array, the values are its rows instead (one array to hand over rather
/// than V, for small batches) and `values` must be empty. Returns
/// `(n_out, prod(shape))`.
#[pyfunction]
#[pyo3(signature = (sizes, n_init, ops, fill, outputs, n_out, values, shape, threads=None, packed=None, models=None))]
#[allow(clippy::too_many_arguments)]
fn circuit_network_run<'py>(
    py: Python<'py>,
    sizes: Vec<usize>,
    n_init: usize,
    ops: PyReadonlyArray2<i64>,
    fill: PyReadonlyArray2<i64>,
    outputs: PyReadonlyArray2<i64>,
    n_out: usize,
    values: Vec<PyReadonlyArrayDyn<'py, Complex64>>,
    shape: Vec<usize>,
    threads: Option<usize>,
    packed: Option<PyReadonlyArray2<'py, Complex64>>,
    models: Option<Vec<ModelArgs<'py>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let ov = ops.as_array();
    if ov.nrows() > 0 && ov.ncols() != 6 {
        return Err(PyValueError::new_err("ops must have 6 columns"));
    }
    let mut prog_ops = Vec::with_capacity(ov.nrows());
    for r in ov.rows() {
        let u = |k: usize| usize::try_from(r[k]).map_err(|_| PyValueError::new_err("ops: negative index"));
        prog_ops.push(match r[0] {
            0 => Op::Merge { a: u(1)?, ia: u(2)?, b: u(3)?, ib: u(4)?, out: u(5)? },
            1 => Op::Join { a: u(1)?, i: u(2)?, j: u(4)?, out: u(5)? },
            _ => return Err(PyValueError::new_err("ops: kind must be 0 or 1")),
        });
    }
    let program = Program {
        sizes,
        n_init,
        ops: prog_ops,
        fill: rows(&fill, "fill")?.into_iter().map(|x| (x[0], x[1], x[2], x[3])).collect(),
        outputs: rows(&outputs, "outputs")?.into_iter().map(|x| (x[0], x[1], x[2], x[3])).collect(),
        n_out,
        models: match models {
            Some(ms) => ms.iter().map(|(b, code, consts, outs, nr)| model_block(*b, code, consts, outs, *nr)).collect::<PyResult<Vec<_>>>()?,
            None => vec![],
        },
    };
    let nb: usize = shape.iter().product();
    let n_values = match &packed {
        Some(p) => {
            if !values.is_empty() {
                return Err(PyValueError::new_err("values must be empty with packed"));
            }
            let v = p.as_array();
            if v.ncols() != nb || !p.is_c_contiguous() {
                return Err(PyValueError::new_err(format!("packed must be a C-contiguous (V, {nb}) array")));
            }
            v.nrows()
        }
        None => values.len(),
    };
    program.check(n_values).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut raws = Vec::with_capacity(n_values);
    if let Some(p) = &packed {
        // row r is a value on the full batch, C order
        let mut strides = vec![0usize; shape.len()];
        let mut st = 1usize;
        for d in (0..shape.len()).rev() {
            strides[d] = st;
            st *= shape[d];
        }
        let base = p.as_array().as_ptr();
        for r in 0..n_values {
            raws.push(Raw { ptr: unsafe { base.add(r * nb) }, len: nb, strides: strides.clone() });
        }
    }
    for (k, v) in values.iter().enumerate() {
        if v.shape() != shape.as_slice() {
            return Err(PyValueError::new_err(format!("value {k}: shape {:?}, expected {:?}", v.shape(), shape)));
        }
        let view = v.as_array();
        let mut strides = Vec::with_capacity(shape.len());
        let mut len = 1usize;
        for (d, &s) in view.strides().iter().enumerate() {
            let s = usize::try_from(s).map_err(|_| PyValueError::new_err(format!("value {k}: negative strides")))?;
            if shape[d] > 0 {
                len += (shape[d] - 1) * s;
            }
            strides.push(s);
        }
        raws.push(Raw { ptr: view.as_ptr(), len, strides });
    }
    // the output is a numpy array (numpy marks large allocations for huge
    // pages: first touching a fresh 320 MB output in 4 KB pages cost 0.2 s
    // per call on a GB10, in 2 MB pages 0.02 s)
    let arr = numpy::PyArray2::<Complex64>::zeros(py, [program.n_out, nb], false);
    // SAFETY: the array was just created, C-contiguous, and is not shared yet
    let out: &mut [Complex64] = unsafe { arr.as_slice_mut() }.map_err(|e| PyValueError::new_err(e.to_string()))?;
    let work = |out: &mut [Complex64]| {
        let vals: Vec<Value> = raws
            .iter()
            .map(|r| Value { data: if nb == 0 { &[] } else { unsafe { std::slice::from_raw_parts(r.ptr, r.len) } }, offset: 0, strides: r.strides.clone() })
            .collect();
        if nb > 0 {
            run(&program, &vals, &shape, out);
        }
    };
    match threads {
        Some(t) if t > 0 && !lightsim_core::circuit::is_small(&program, nb) => {
            let p = pool(t)?;
            py.detach(|| p.install(|| work(out)));
        }
        // small work runs in place (no hand-over to a pool thread, which costs
        // about 10 us a call); `run` takes no thread then
        _ if lightsim_core::circuit::is_small(&program, nb) => py.detach(|| work(out)),
        _ => py.compute(|| work(out)),
    }
    Ok(arr.into_any())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_network_run, m)?)?;
    Ok(())
}
