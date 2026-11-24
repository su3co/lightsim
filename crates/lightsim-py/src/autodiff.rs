//! Python classes of `lightsim.autodiff`: the persistent S-matrix gradient
//! session and the time-domain reverse-mode session (thin wrappers around
//! `lightsim_core::autodiff`; `python/lightsim/autodiff.py` builds the specs).

use lightsim_core::autodiff as ad;
use lightsim_core::grid::Component;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use crate::Compute;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Mutex;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

/// Engine for a device string: "cpu", "cuda:N", or "auto" (first GPU if any).
fn factory(device: &str, plan: lightsim_core::Plan) -> anyhow::Result<Box<dyn ad::Engine>> {
    #[cfg(feature = "cuda")]
    {
        let dev = if device == "auto" { if lightsim_cuda::device_count().unwrap_or(0) > 0 { "cuda:0" } else { "cpu" } } else { device };
        if dev.starts_with("cuda") {
            crate::gpu_count()?;
        }
        lightsim_cuda::autodiff::factory(dev, plan)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let dev = if device == "auto" { "cpu" } else { device };
        if dev != "cpu" {
            anyhow::bail!("{}", crate::no_cuda());
        }
        ad::engine::cpu_factory(dev, plan)
    }
}

fn arrays(v: Vec<PyReadonlyArrayDyn<f64>>) -> Vec<Vec<f64>> {
    v.iter().map(|a| a.as_array().iter().cloned().collect()).collect()
}

fn array<'py>(py: Python<'py>, v: Vec<f64>, dims: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    Ok(v.into_pyarray(py).reshape(dims.to_vec())?.into_any())
}

fn grid_arrays<'py>(py: Python<'py>, grids: &[ad::DesignGrid], v: Vec<Vec<f64>>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for (g, a) in grids.iter().zip(v) {
        out.append(array(py, a, &g.dims)?)?;
    }
    Ok(out)
}

/// Persistent S-matrix gradient session (see `lightsim.autodiff.GradientSession`).
#[pyclass(module = "lightsim._core")]
pub struct GradientSession {
    inner: Mutex<ad::GradientSession>,
}

impl GradientSession {
    fn cotangent(&self, s: &ad::GradientSession, re: PyReadonlyArrayDyn<f64>, im: PyReadonlyArrayDyn<f64>) -> PyResult<Vec<Complex64>> {
        let (m, n, k) = s.shape();
        let (re, im) = (re.as_array(), im.as_array());
        if re.shape() != [m, n, k] || im.shape() != [m, n, k] {
            return Err(PyValueError::new_err(format!("upstream gradient has shape {:?}, S has {:?}", re.shape(), (m, n, k))));
        }
        Ok(re.iter().zip(im.iter()).map(|(a, b)| Complex64::new(*a, *b)).collect())
    }
}

#[pymethods]
impl GradientSession {
    /// `spec`: JSON of `lightsim_core::autodiff::SessionSpec`; `devices`: one
    /// engine per entry ("cpu", "cuda:0", ...).
    #[new]
    fn new(py: Python<'_>, spec: &str, devices: Vec<String>) -> PyResult<Self> {
        let spec: ad::SessionSpec = serde_json::from_str(spec).map_err(|e| PyValueError::new_err(format!("invalid session spec: {e}")))?;
        let s = py.compute(|| ad::GradientSession::new(spec, &devices, &factory)).map_err(err)?;
        Ok(GradientSession { inner: Mutex::new(s) })
    }

    /// (M, N, K): outputs, inputs, frequencies.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.lock().unwrap().shape()
    }
    #[getter]
    fn devices(&self) -> Vec<String> {
        self.inner.lock().unwrap().devices().to_vec()
    }
    #[getter]
    fn cells(&self) -> usize {
        self.inner.lock().unwrap().cells()
    }
    #[getter]
    fn setup_seconds(&self) -> f64 {
        self.inner.lock().unwrap().setup_seconds
    }
    /// Time steps between two samples of the frequency-domain monitors.
    #[getter]
    fn dft_stride(&self) -> usize {
        self.inner.lock().unwrap().dft_stride()
    }

    /// Set the index-grid samples: one array per grid (any shape with
    /// `nx * ny * nz` elements, C order `[ix, iy, iz]`).
    fn set_design(&self, py: Python<'_>, eps: Vec<PyReadonlyArrayDyn<f64>>) -> PyResult<()> {
        let eps = arrays(eps);
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        py.compute(|| s.set_design(&eps)).map_err(err)
    }

    /// Run the excitations (with `adjoint`, also every adjoint simulation,
    /// dispatched together). Returns (Re S, Im S), each `[M, N, K]`.
    #[pyo3(signature = (adjoint=false))]
    fn forward<'py>(&self, py: Python<'py>, adjoint: bool) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let v = py.compute(|| s.forward(adjoint)).map_err(err)?;
        let (m, n, k) = s.shape();
        Ok((array(py, v.iter().map(|z| z.re).collect(), &[m, n, k])?, array(py, v.iter().map(|z| z.im).collect(), &[m, n, k])?))
    }

    /// dL/d(eps) on the samples of every design grid for the upstream
    /// gradient g = dL/dRe(S) + i dL/dIm(S), given as (Re g, Im g).
    fn backward<'py>(&self, py: Python<'py>, g_re: PyReadonlyArrayDyn<f64>, g_im: PyReadonlyArrayDyn<f64>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let g = self.cotangent(s, g_re, g_im)?;
        let v = py.compute(|| s.backward(&g)).map_err(err)?;
        grid_arrays(py, s.grids(), v)
    }

    /// dL/d(eps_r) on the Yee voxels of every design box: a list over grids
    /// of dicts component -> (origin index, array [i, j, k]).
    fn backward_yee<'py>(&self, py: Python<'py>, g_re: PyReadonlyArrayDyn<f64>, g_im: PyReadonlyArrayDyn<f64>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let g = self.cotangent(s, g_re, g_im)?;
        let v = py.compute(|| s.backward_yee(&g)).map_err(err)?;
        let out = PyList::empty(py);
        for (per_comp, boxes) in v.into_iter().zip(s.design_boxes()) {
            let d = PyDict::new(py);
            for (data, b) in per_comp.into_iter().zip(boxes) {
                d.set_item(b.comp.name(), (b.bx.lo.to_vec(), array(py, data, &b.bx.size)?))?;
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// Permittivity the engine uses on the E_x, E_y, E_z voxels of the
    /// design bounding box: component -> (origin, array, NaN where no grid
    /// owns the voxel).
    fn yee_permittivity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.lock().unwrap();
        let d = PyDict::new(py);
        let s = &*s;
        let parts: Vec<_> = py.compute(|| (0..3).map(|a| s.yee_permittivity(a)).collect());
        for (a, (bx, v)) in parts.into_iter().enumerate() {
            d.set_item(Component::e(a).name(), (bx.lo.to_vec(), array(py, v.iter().map(|x| *x as f64).collect(), &bx.size)?))?;
        }
        Ok(d)
    }

    /// Runs done since the last `set_design`: list of dicts
    /// (kind, index, device, steps, seconds, wall).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let s = self.inner.lock().unwrap();
        let out = PyList::empty(py);
        for r in &s.stats {
            let d = PyDict::new(py);
            d.set_item("kind", r.kind)?;
            d.set_item("index", r.index)?;
            d.set_item("device", &r.device)?;
            d.set_item("steps", r.steps)?;
            d.set_item("seconds", r.seconds)?;
            d.set_item("wall", r.wall)?;
            out.append(d)?;
        }
        Ok(out)
    }
}

/// Time-domain reverse-mode session (see `lightsim.autodiff.ReverseSession`).
#[pyclass(module = "lightsim._core")]
pub struct ReverseSession {
    inner: Mutex<ad::ReverseSession>,
}

#[pymethods]
impl ReverseSession {
    /// `spec`: JSON of `lightsim_core::autodiff::ReverseSpec`.
    #[new]
    fn new(py: Python<'_>, spec: &str, device: &str) -> PyResult<Self> {
        let spec: ad::ReverseSpec = serde_json::from_str(spec).map_err(|e| PyValueError::new_err(format!("invalid reverse-mode spec: {e}")))?;
        let s = py.compute(|| ad::ReverseSession::new(spec, device, &factory)).map_err(err)?;
        Ok(ReverseSession { inner: Mutex::new(s) })
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.lock().unwrap().device()
    }
    #[getter]
    fn steps(&self) -> usize {
        self.inner.lock().unwrap().steps()
    }
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.lock().unwrap().dt()
    }

    fn set_design(&self, py: Python<'_>, eps: Vec<PyReadonlyArrayDyn<f64>>) -> PyResult<()> {
        let eps = arrays(eps);
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        py.compute(|| s.set_design(&eps)).map_err(err)
    }

    /// Forward run; returns the built-in objective (0 for a `fields` objective).
    fn forward(&self, py: Python<'_>) -> PyResult<f64> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        py.compute(|| s.forward()).map_err(err)
    }

    /// Samples of the objective monitor: list of (component, origin, array
    /// `[step, i, j, k]`). E samples are at `(n + 1) dt`, H at `(n + 1/2) dt`.
    fn series<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let s = self.inner.lock().unwrap();
        let n = s.steps();
        let out = PyList::empty(py);
        for b in s.series().map_err(err)? {
            let a = b.data.into_pyarray(py).reshape(vec![n, b.size[0], b.size[1], b.size[2]])?;
            out.append((b.comp.name(), b.origin.to_vec(), a))?;
        }
        Ok(out)
    }

    /// Reverse pass for the built-in objective: dL/d(eps) per design grid.
    fn backward<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let v = py.compute(|| s.backward()).map_err(err)?;
        grid_arrays(py, s.grids(), v)
    }

    /// Reverse pass for caller-supplied cotangents dL/d(sample), one array
    /// per entry of `series()` with the same shape.
    fn backward_with<'py>(&self, py: Python<'py>, cotangents: Vec<PyReadonlyArrayDyn<f64>>) -> PyResult<Bound<'py, PyList>> {
        let c = arrays(cotangents);
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let v = py.compute(|| s.backward_with(&c)).map_err(err)?;
        grid_arrays(py, s.grids(), v)
    }

    /// Compute the gradients with respect to the dispersive poles on every
    /// reverse pass.
    fn set_pole_gradients(&self, on: bool) -> PyResult<()> {
        self.inner.lock().unwrap().set_pole_gradients(on).map_err(err)
    }

    /// Pole gradients of the last reverse pass, as a JSON list.
    fn pole_gradients(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.lock().unwrap().pole_gradients()).map_err(|e| err(e.into()))
    }

    /// Timings and sizes of the last passes.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.lock().unwrap();
        let d = PyDict::new(py);
        d.set_item("forward_seconds", s.stats.forward_seconds)?;
        d.set_item("reverse_seconds", s.stats.reverse_seconds)?;
        d.set_item("steps", s.stats.steps)?;
        d.set_item("cells", s.stats.cells)?;
        d.set_item("shell_floats_per_step", s.stats.shell_floats_per_step)?;
        d.set_item("monitor_floats_per_step", s.stats.monitor_floats_per_step)?;
        d.set_item("reconstruction_residual", s.stats.reconstruction_residual)?;
        d.set_item("device", s.device())?;
        Ok(d)
    }
}


/// S-matrix problem with reverse-mode gradients (see
/// `lightsim.autodiff.ReverseSMatrix`, engine "reverse" of `lightsim.invdes`).
#[pyclass(module = "lightsim._core")]
pub struct ReverseSMatrix {
    inner: Mutex<ad::ReverseSMatrix>,
}

#[pymethods]
impl ReverseSMatrix {
    /// `spec`: JSON of `lightsim_core::autodiff::SessionSpec`; `devices`: the
    /// inputs are spread over the entries ("cpu", "cuda:0", ...).
    #[new]
    fn new(py: Python<'_>, spec: &str, devices: Vec<String>) -> PyResult<Self> {
        let spec: ad::SessionSpec = serde_json::from_str(spec).map_err(|e| PyValueError::new_err(format!("invalid session spec: {e}")))?;
        let s = py.compute(|| ad::ReverseSMatrix::new(spec, &devices, &factory)).map_err(err)?;
        Ok(ReverseSMatrix { inner: Mutex::new(s) })
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.lock().unwrap().shape()
    }
    #[getter]
    fn devices(&self) -> Vec<String> {
        self.inner.lock().unwrap().devices()
    }

    fn set_design(&self, py: Python<'_>, eps: Vec<PyReadonlyArrayDyn<f64>>) -> PyResult<()> {
        let eps = arrays(eps);
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        py.compute(|| s.set_design(&eps)).map_err(err)
    }

    /// Run the forward simulations. Returns (Re S, Im S), each `[M, N, K]`.
    fn forward<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let v = py.compute(|| s.forward()).map_err(err)?;
        let (m, n, k) = s.shape();
        Ok((array(py, v.iter().map(|z| z.re).collect(), &[m, n, k])?, array(py, v.iter().map(|z| z.im).collect(), &[m, n, k])?))
    }

    /// dL/d(eps) on the samples of every design grid for g = dL/dRe(S) + i
    /// dL/dIm(S), given as (Re g, Im g): one reverse pass per input.
    fn backward<'py>(&self, py: Python<'py>, g_re: PyReadonlyArrayDyn<f64>, g_im: PyReadonlyArrayDyn<f64>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().unwrap();
        let s = &mut *guard;
        let (m, n, k) = s.shape();
        let (re, im) = (g_re.as_array(), g_im.as_array());
        if re.shape() != [m, n, k] || im.shape() != [m, n, k] {
            return Err(PyValueError::new_err(format!("upstream gradient has shape {:?}, S has {:?}", re.shape(), (m, n, k))));
        }
        let g: Vec<Complex64> = re.iter().zip(im.iter()).map(|(a, b)| Complex64::new(*a, *b)).collect();
        let v = py.compute(|| s.backward(&g)).map_err(err)?;
        grid_arrays(py, s.grids(), v)
    }

    /// One dict per input: input, device, steps, forward_seconds,
    /// reverse_seconds (of the last forward run and reverse pass).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let s = self.inner.lock().unwrap();
        let out = PyList::empty(py);
        for r in &s.stats {
            let d = PyDict::new(py);
            d.set_item("input", r.input)?;
            d.set_item("device", &r.device)?;
            d.set_item("steps", r.steps)?;
            d.set_item("forward_seconds", r.forward_seconds)?;
            d.set_item("reverse_seconds", r.reverse_seconds)?;
            out.append(d)?;
        }
        Ok(out)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GradientSession>()?;
    m.add_class::<ReverseSession>()?;
    m.add_class::<ReverseSMatrix>()?;
    Ok(())
}
