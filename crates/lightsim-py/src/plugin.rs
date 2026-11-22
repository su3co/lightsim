//! Material plug-ins written in Python (`lightsim.plugins`): a registered
//! name maps to Python callables, called once per step and E component with
//! NumPy arrays over the medium's cells (see `lightsim_core::plugin`).

use lightsim_core::Real;
use lightsim_core::plugin::{MaterialPlugin, PluginCall};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// One instance (medium, E component, slab): the callables and NumPy
/// buffers that persist across calls.
struct PyPlugin {
    before: Option<Py<PyAny>>,
    after: Py<PyAny>,
    ncols: usize,
    params: Py<PyAny>,
    /// e, e_prev, inv_eps (n,), state (n, ncols)
    bufs: Option<(Py<PyArray1<Real>>, Py<PyArray1<Real>>, Py<PyArray1<Real>>, Py<PyArray2<Real>>)>,
    err: Option<String>,
}

impl PyPlugin {
    fn bufs<'py>(&mut self, py: Python<'py>, n: usize) -> (Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>, Bound<'py, PyArray2<Real>>) {
        if self.bufs.as_ref().map(|b| b.0.bind(py).len().ok() != Some(n)).unwrap_or(true) {
            self.bufs = Some((
                PyArray1::<Real>::zeros(py, n, false).unbind(),
                PyArray1::<Real>::zeros(py, n, false).unbind(),
                PyArray1::<Real>::zeros(py, n, false).unbind(),
                PyArray2::<Real>::zeros(py, [n, self.ncols], false).unbind(),
            ));
        }
        let b = self.bufs.as_ref().unwrap();
        (b.0.bind(py).clone(), b.1.bind(py).clone(), b.2.bind(py).clone(), b.3.bind(py).clone())
    }

    fn ctx<'py>(&self, py: Python<'py>, call: &PluginCall) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("axis", call.axis)?;
        d.set_item("step", call.step)?;
        d.set_item("dt", call.dt)?;
        d.set_item("params", self.params.bind(py))?;
        Ok(d)
    }

    fn fail(&mut self, py: Python<'_>, e: PyErr) {
        let tb = e.traceback(py).and_then(|t| t.format().ok()).unwrap_or_default();
        self.err = Some(format!("{tb}{e}"));
    }
}

impl MaterialPlugin for PyPlugin {
    fn before(&mut self, call: &PluginCall, e: &[Real], _state: &mut [Real]) {
        if self.err.is_some() {
            return;
        }
        Python::attach(|py| {
            let Some(f) = self.before.as_ref().map(|f| f.clone_ref(py)) else { return };
            let (ea, _, _, st) = self.bufs(py, e.len());
            unsafe { ea.as_slice_mut().unwrap() }.copy_from_slice(e);
            let r = self.ctx(py, call).and_then(|c| f.bind(py).call1((ea, st, c)));
            if let Err(err) = r {
                self.fail(py, err);
            }
        });
    }

    fn after(&mut self, call: &PluginCall, e: &mut [Real], e_prev: &[Real], inv_eps: &[Real], _state: &mut [Real]) {
        if self.err.is_some() {
            return;
        }
        Python::attach(|py| {
            let f = self.after.clone_ref(py);
            let (ea, pa, ia, st) = self.bufs(py, e.len());
            unsafe {
                ea.as_slice_mut().unwrap().copy_from_slice(e);
                pa.as_slice_mut().unwrap().copy_from_slice(e_prev);
                ia.as_slice_mut().unwrap().copy_from_slice(inv_eps);
            }
            let r = self.ctx(py, call).and_then(|c| f.bind(py).call1((ea.clone(), pa, ia, st, c)));
            match r {
                Err(err) => self.fail(py, err),
                Ok(ret) => {
                    // the callable updates `e` in place, or returns the new values
                    if ret.is_none() {
                        e.copy_from_slice(unsafe { ea.as_slice().unwrap() });
                    } else {
                        match ret.extract::<numpy::PyReadonlyArray1<Real>>() {
                            Ok(a) if a.len().ok() == Some(e.len()) => e.copy_from_slice(a.as_slice().unwrap()),
                            _ => self.err = Some(format!("the after() callable must update e in place (return None) or return a float array of {} values matching e", e.len())),
                        }
                    }
                }
            }
        });
    }

    fn error(&mut self) -> Option<String> {
        self.err.take()
    }
}

/// Register Python callables as the material plug-in `name`.
#[pyfunction]
#[pyo3(signature = (name, after, before=None, state=0))]
fn register_plugin(py: Python<'_>, name: &str, after: Py<PyAny>, before: Option<Py<PyAny>>, state: usize) -> PyResult<()> {
    let _ = py;
    let (after, before) = (Arc::new(after), Arc::new(before));
    lightsim_core::plugin::register(
        name,
        Arc::new(move |params: &serde_json::Value| {
            let text = params.to_string();
            Python::attach(|py| -> anyhow::Result<Box<dyn MaterialPlugin>> {
                let json = py.import("json").map_err(|e| anyhow::anyhow!("{e}"))?;
                let p = json.call_method1("loads", (text.as_str(),)).map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(Box::new(PyPlugin { before: before.as_ref().as_ref().map(|b| b.clone_ref(py)), after: after.clone_ref(py), ncols: state, params: p.unbind(), bufs: None, err: None }))
            })
        }),
    );
    Ok(())
}

/// Remove a registered plug-in.
#[pyfunction]
fn unregister_plugin(name: &str) {
    lightsim_core::plugin::unregister(name);
}

/// Register the Rust example plug-ins ("example_lorentz", "example_kerr",
/// "example_runaway").
#[pyfunction]
fn register_example_plugins() {
    lightsim_core::plugin::examples::register_examples();
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(register_example_plugins, m)?)?;
    Ok(())
}
