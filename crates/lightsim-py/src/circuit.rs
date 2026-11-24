//! Binding of the circuit network solve (`lightsim_core::circuit`), used by
//! `lightsim.circuit`. The program arrives as plain integer arrays and the
//! values as complex arrays already broadcast to the batch shape (views with
//! zero strides are read in place, not copied).

use lightsim_core::circuit::{Op, Program, Value, run};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut m = POOLS.get_or_init(Default::default).lock().unwrap();
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
#[pyo3(signature = (sizes, n_init, ops, fill, outputs, n_out, values, shape, threads=None, packed=None))]
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
    let mut out = vec![Complex64::new(0.0, 0.0); program.n_out * nb];
    let work = |out: &mut Vec<Complex64>| {
        let vals: Vec<Value> = raws
            .iter()
            .map(|r| Value { data: if nb == 0 { &[] } else { unsafe { std::slice::from_raw_parts(r.ptr, r.len) } }, offset: 0, strides: r.strides.clone() })
            .collect();
        if nb > 0 {
            run(&program, &vals, &shape, out);
        }
    };
    match threads {
        Some(t) if t > 0 => {
            let p = pool(t)?;
            py.detach(|| p.install(|| work(&mut out)));
        }
        _ => py.detach(|| work(&mut out)),
    }
    Ok(out.into_pyarray(py).reshape(vec![program.n_out, nb])?.into_any())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(circuit_network_run, m)?)?;
    Ok(())
}
