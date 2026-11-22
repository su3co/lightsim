//! Python extension module `lightsim._core`.

mod autodiff;
mod fem;
mod plugin;


#[cfg(all(feature = "cuda", feature = "f64"))]
compile_error!("the f64 (double-precision) build is CPU only: features \"cuda\" and \"f64\" cannot be combined");
use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::plan::CompileOptions;
use lightsim_core::results::Results;
use lightsim_core::run::{Chain, RunOptions, run_chain};
use lightsim_core::{Backend, Simulation};
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn parse(spec: &str) -> PyResult<Simulation> {
    Simulation::from_json(spec).map_err(err)
}

/// Device strings: "cpu", "cuda", "cuda:0", "cuda:0,1", "auto".
fn backends(sim: &Simulation, device: &str) -> anyhow::Result<Vec<Box<dyn Backend>>> {
    let dev = if device == "auto" {
        #[cfg(feature = "cuda")]
        {
            // (a small 2-D grid is faster on the CPU than on a GPU)
            if !sim.auto_prefers_cpu() && lightsim_cuda::device_count().unwrap_or(0) > 0 { "cuda:0".to_string() } else { "cpu".to_string() }
        }
        #[cfg(not(feature = "cuda"))]
        {
            "cpu".to_string()
        }
    } else {
        device.to_string()
    };
    if sim.fixed_angle && dev == "cpu" {
        return Ok(vec![lightsim_core::engine::fixed_angle::cpu_backend(sim)?]);
    }
    if dev == "cpu" {
        let plan = lightsim_core::compile(sim, &CompileOptions::default())?;
        return Ok(vec![Box::new(CpuBackend::new(plan)?)]);
    }
    #[cfg(feature = "cuda")]
    {
        let rest = dev.trim_start_matches("cuda").trim_start_matches(':');
        let n = gpu_count()?;
        let gpus: Vec<usize> = if rest.is_empty() {
            vec![0]
        } else if rest == "all" {
            (0..n).collect()
        } else {
            rest.split(',').map(|x| x.trim().parse().map_err(|_| anyhow::anyhow!("invalid device {device:?}: expected \"cpu\", \"auto\", \"cuda\", \"cuda:N\", \"cuda:N,M,...\" or \"cuda:all\""))).collect::<anyhow::Result<_>>()?
        };
        if let Some(g) = gpus.iter().find(|&&g| g >= n) {
            anyhow::bail!("cuda:{g} was requested, but this machine has {n} NVIDIA GPU(s) (cuda:0 to cuda:{})", n - 1);
        }
        if sim.fixed_angle {
            return Ok(vec![lightsim_cuda::fixed_angle::backend(sim, &gpus)?]);
        }
        let slabs = lightsim_cuda::multi::build_slabs(sim, &gpus, None)?;
        return Ok(slabs.into_iter().map(|b| Box::new(b) as Box<dyn Backend>).collect());
    }
    #[allow(unreachable_code)]
    Err(anyhow::anyhow!("{}", no_cuda()))
}

/// The number of usable NVIDIA GPUs, or a clear error when there is none: no
/// driver installed (the CUDA backend loads libcuda at run time, so a GPU
/// build also runs on machines without one), or a driver that finds no GPU.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_count() -> anyhow::Result<usize> {
    match lightsim_cuda::device_count() {
        Ok(0) => anyhow::bail!("no NVIDIA GPU is available: the NVIDIA driver found no GPU; use device='cpu'"),
        Ok(n) => Ok(n),
        Err(e) => anyhow::bail!(
            "no NVIDIA GPU is available: the NVIDIA driver (libcuda.so.1) could not be loaded or initialised. Running on a GPU needs an NVIDIA GPU and its driver, version 580 or later; use device='cpu' otherwise. (driver error: {e:#})"
        ),
    }
}

/// The error for a GPU request in a build without the CUDA backend.
pub(crate) fn no_cuda() -> &'static str {
    if lightsim_core::REAL_IS_F64 {
        "this lightsim build is double precision (f64), which runs on the CPU only; use device='cpu' (or a default f32 build with CUDA for GPUs)"
    } else {
        "this lightsim build has no CUDA support; use device='cpu'"
    }
}

fn arr<'py>(py: Python<'py>, v: &[lightsim_core::Real], dims: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let a = v.to_vec().into_pyarray(py);
    Ok(a.reshape(dims.to_vec())?.into_any())
}

fn results_to_py<'py>(py: Python<'py>, r: &Results) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("dt", r.dt)?;
    if !r.gain_levels.is_empty() {
        out.set_item("gain_levels", r.gain_levels.clone())?;
    }
    let stats = PyDict::new(py);
    stats.set_item("backend", &r.stats.backend)?;
    stats.set_item("cells", r.stats.cells)?;
    stats.set_item("steps", r.stats.steps)?;
    stats.set_item("seconds", r.stats.seconds)?;
    stats.set_item("mcells_per_s", r.stats.mcells_per_s)?;
    stats.set_item("stopped_early", r.stats.stopped_early)?;
    stats.set_item("energy_steps", r.stats.energy_steps.clone().into_pyarray(py))?;
    stats.set_item("energy", r.stats.energy.clone().into_pyarray(py))?;
    out.set_item("stats", stats)?;
    let dft = PyDict::new(py);
    for (name, d) in &r.dft {
        let dd = PyDict::new(py);
        dd.set_item("freqs", d.freqs.clone().into_pyarray(py))?;
        dd.set_item("stride", d.stride)?;
        let fields = PyDict::new(py);
        for f in &d.fields {
            let fd = PyDict::new(py);
            fd.set_item("origin", f.origin.to_vec())?;
            fd.set_item("re", arr(py, &f.re, &f.dims)?)?;
            fd.set_item("im", arr(py, &f.im, &f.dims)?)?;
            // flux/mode monitors hold several boxes of the same component
            let mut key = f.comp.clone();
            let mut t = 1;
            while fields.contains(&key)? {
                key = format!("{}_{}", f.comp, t);
                t += 1;
            }
            fields.set_item(key, fd)?;
        }
        dd.set_item("fields", fields)?;
        dft.set_item(name, dd)?;
    }
    out.set_item("dft", dft)?;
    let flux = PyDict::new(py);
    for (k, v) in &r.flux {
        flux.set_item(k, v.clone().into_pyarray(py))?;
    }
    out.set_item("flux", flux)?;
    let modes = PyDict::new(py);
    for (k, m) in &r.modes {
        let md = PyDict::new(py);
        md.set_item("freqs", m.freqs.clone().into_pyarray(py))?;
        let to2 = |re: &Vec<Vec<f64>>, im: &Vec<Vec<f64>>| -> Vec<Vec<(f64, f64)>> {
            re.iter().zip(im).map(|(a, b)| a.iter().zip(b).map(|(x, y)| (*x, *y)).collect()).collect()
        };
        let cplx = |v: Vec<Vec<(f64, f64)>>| -> PyResult<Bound<'py, PyList>> {
            let l = PyList::empty(py);
            for row in v {
                let r: Vec<f64> = row.iter().map(|z| z.0).collect();
                let i: Vec<f64> = row.iter().map(|z| z.1).collect();
                let d = PyDict::new(py);
                d.set_item("re", r.into_pyarray(py))?;
                d.set_item("im", i.into_pyarray(py))?;
                l.append(d)?;
            }
            Ok(l)
        };
        md.set_item("plus", cplx(to2(&m.amp_plus_re, &m.amp_plus_im))?)?;
        md.set_item("minus", cplx(to2(&m.amp_minus_re, &m.amp_minus_im))?)?;
        md.set_item("flux", m.flux.clone().into_pyarray(py))?;
        modes.set_item(k, md)?;
    }
    out.set_item("modes", modes)?;
    let time = PyDict::new(py);
    for (k, t) in &r.time {
        let td = PyDict::new(py);
        td.set_item("times", t.times.clone().into_pyarray(py))?;
        let fields = PyDict::new(py);
        for f in &t.fields {
            fields.set_item(&f.comp, arr(py, &f.data, &f.dims)?)?;
        }
        td.set_item("fields", fields)?;
        time.set_item(k, td)?;
    }
    out.set_item("time", time)?;
    let far = PyDict::new(py);
    for (k, f) in &r.far_field {
        let d = PyDict::new(py);
        let dims = vec![f.freqs.len(), f.theta.len(), f.phi.len()];
        let arr3 = |v: &Vec<f64>| -> PyResult<Bound<'py, PyAny>> { Ok(v.clone().into_pyarray(py).reshape(dims.clone())?.into_any()) };
        d.set_item("freqs", f.freqs.clone().into_pyarray(py))?;
        d.set_item("theta", f.theta.clone().into_pyarray(py))?;
        d.set_item("phi", f.phi.clone().into_pyarray(py))?;
        d.set_item("intensity", arr3(&f.intensity)?)?;
        d.set_item("flux", f.flux.clone().into_pyarray(py))?;
        d.set_item("e_theta_re", arr3(&f.e_theta_re)?)?;
        d.set_item("e_theta_im", arr3(&f.e_theta_im)?)?;
        d.set_item("e_phi_re", arr3(&f.e_phi_re)?)?;
        d.set_item("e_phi_im", arr3(&f.e_phi_im)?)?;
        far.set_item(k, d)?;
    }
    out.set_item("far_field", far)?;
    let energy = PyDict::new(py);
    for (k, e) in &r.energy {
        let d = PyDict::new(py);
        d.set_item("times", e.times.clone().into_pyarray(py))?;
        d.set_item("energy", e.energy.clone().into_pyarray(py))?;
        energy.set_item(k, d)?;
    }
    out.set_item("energy", energy)?;
    let ss = PyDict::new(py);
    for (k, (f, re, im)) in &r.source_spectra {
        let d = PyDict::new(py);
        d.set_item("freqs", f.clone().into_pyarray(py))?;
        d.set_item("re", re.clone().into_pyarray(py))?;
        d.set_item("im", im.clone().into_pyarray(py))?;
        ss.set_item(k, d)?;
    }
    out.set_item("source_spectra", ss)?;
    let sources = PyList::empty(py);
    for s in &r.sources {
        let d = PyDict::new(py);
        d.set_item("name", &s.name)?;
        d.set_item("signal", serde_json::to_string(&s.signal).map_err(|e| PyRuntimeError::new_err(e.to_string()))?)?;
        d.set_item("analytic", s.analytic)?;
        sources.append(d)?;
    }
    out.set_item("sources", sources)?;
    Ok(out)
}

/// Spectrum `sum_n s(t_n) exp(i w t_n) dt` of a time signal (a JSON object as
/// in the spec) at `freqs`, over the samples `t_n = (n + 1/2) dt` of a run
/// of `nsteps` steps. `analytic = true` gives the spectrum of the complex
/// (analytic) signal that sources with a complex pattern inject into
/// complex-field runs. Returns `(re, im)`.
#[pyfunction]
#[pyo3(signature = (signal, freqs, dt, nsteps, analytic=false))]
fn signal_spectrum<'py>(
    py: Python<'py>,
    signal: &str,
    freqs: Vec<f64>,
    dt: f64,
    nsteps: usize,
    analytic: bool,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let sig: lightsim_core::sources::TimeSignal = serde_json::from_str(signal).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let sp = sig.spectrum_of(&freqs, dt, nsteps.max(1), 0.5, analytic);
    let re: Vec<f64> = sp.iter().map(|z| z.re).collect();
    let im: Vec<f64> = sp.iter().map(|z| z.im).collect();
    Ok((re.into_pyarray(py).into_any(), im.into_pyarray(py).into_any()))
}

/// Run a simulation given as a JSON spec. Returns a dict of numpy arrays.
#[pyfunction]
#[pyo3(signature = (spec, device="auto", progress=0, hook=None, hook_every=0.5))]
fn run<'py>(py: Python<'py>, spec: &str, device: &str, progress: usize, hook: Option<Py<PyAny>>, hook_every: f64) -> PyResult<Bound<'py, PyDict>> {
    let sim = parse(spec)?;
    let hook = hook.map(|f| py_hook(f, hook_every));
    let res = py
        .detach(|| -> anyhow::Result<Results> {
            #[cfg(feature = "cuda")]
            if sim.fixed_angle && device.starts_with("cuda") {
                // (several GPUs: the fixed-angle engine's own decomposed loop)
                let rest = device.trim_start_matches("cuda").trim_start_matches(':');
                let gpus: Vec<usize> = if rest.is_empty() { vec![0] } else if rest == "all" { (0..lightsim_cuda::device_count()?).collect() } else { rest.split(',').map(|x| x.parse().unwrap()).collect() };
                return lightsim_cuda::fixed_angle::run(&sim, &gpus, &RunOptions { progress_every: progress });
            }
            let slabs = backends(&sim, device)?;
            let n = slabs.len();
            let mut chain = Chain::new(slabs);
            chain.local_wrap = n > 1 && sim.boundaries.x[0].is_periodic();
            chain.hook = hook;
            run_chain(&mut chain, &RunOptions { progress_every: progress })
        })
        .map_err(err)?;
    results_to_py(py, &res)
}

/// Grid node coordinates (x, y, z) including PML layers.
#[pyfunction]
fn build_grid<'py>(py: Python<'py>, spec: &str) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let sim = parse(spec)?;
    let g = sim.build_grid().map_err(err)?;
    Ok((g.x.into_pyarray(py).into_any(), g.y.into_pyarray(py).into_any(), g.z.into_pyarray(py).into_any()))
}

/// Time step and number of steps for a spec.
#[pyfunction]
fn time_info(spec: &str) -> PyResult<(f64, usize)> {
    let sim = parse(spec)?;
    let g = sim.build_grid().map_err(err)?;
    let dt = sim.time_step(&g).map_err(err)?;
    Ok((dt, (sim.run_time / dt).ceil() as usize))
}

/// Cross-section info for mode solving (see lightsim_core::plane::PlaneInfo).
/// With `freq`, `eps_*_im` hold the imaginary permittivity of lossy and
/// dispersive media (and `eps_*` their real part) at that frequency.
#[pyfunction]
#[pyo3(signature = (spec, axis, position, min, max, freq=None))]
fn plane_info<'py>(
    py: Python<'py>,
    spec: &str,
    axis: usize,
    position: f64,
    min: [f64; 3],
    max: [f64; 3],
    freq: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let sim = parse(spec)?;
    let pi = lightsim_core::plane::plane_info_at(&sim, axis, position, min, max, freq).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("axis", pi.axis)?;
    d.set_item("p", pi.p)?;
    d.set_item("axes", pi.axes.to_vec())?;
    d.set_item("origin", pi.origin.to_vec())?;
    d.set_item("dims", pi.dims.to_vec())?;
    d.set_item("u_nodes", pi.u_nodes.into_pyarray(py))?;
    d.set_item("v_nodes", pi.v_nodes.into_pyarray(py))?;
    let dims = [pi.dims[0], pi.dims[1]];
    d.set_item("eps_u", pi.eps_u.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("eps_v", pi.eps_v.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("eps_n", pi.eps_n.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("eps_u_im", pi.eps_u_im.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("eps_v_im", pi.eps_v_im.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("eps_n_im", pi.eps_n_im.into_pyarray(py).reshape(dims.to_vec())?)?;
    d.set_item("h_offset", pi.h_offset)?;
    d.set_item("coord", pi.coord)?;
    Ok(d)
}

/// Power (W) of every mode profile of a port, as the mode overlap measures
/// it (`lightsim_core::autodiff::session::port_power`): the measure both
/// S-matrix engines normalise the port profiles with.
#[pyfunction]
fn port_power(spec: &str, port: &str) -> PyResult<Vec<f64>> {
    let sim = parse(spec)?;
    let port: lightsim_core::autodiff::PortSpec = serde_json::from_str(port).map_err(|e| PyRuntimeError::new_err(format!("port: {e}")))?;
    let grid = sim.build_grid().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b = &sim.boundaries;
    let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
    Ok((0..port.profiles.len()).map(|k| lightsim_core::autodiff::session::port_power(&grid, periodic, &port, k)).collect())
}

/// Available devices.
#[pyfunction]
fn devices() -> Vec<String> {
    let mut v = vec!["cpu".to_string()];
    #[cfg(feature = "cuda")]
    {
        for i in 0..lightsim_cuda::device_count().unwrap_or(0) {
            let name = cuda_core::CudaContext::new(i).and_then(|c| c.name()).unwrap_or_else(|_| "GPU".into());
            v.push(format!("cuda:{i} {name}"));
        }
    }
    v
}

/// A Python callable as a run hook (see `lightsim_core::run::RunHook`): it is
/// called with a dict (`step`, `nsteps`, `dt`, `elapsed`, `decay`, `shutoff`
/// and, when one was asked for, `slice` = dict of `comp`, `axis`, `index`,
/// `data`) and returns `None` or a dict with `stop` (bool) and `slice`
/// (`(component, axis, index)` to hand over at the next call). A callable
/// that raises stops the run.
fn py_hook(f: Py<PyAny>, every: f64) -> lightsim_core::run::RunHook {
    use lightsim_core::run::{HookInfo, HookReply};
    let call = move |info: &HookInfo| -> HookReply {
        Python::attach(|py| -> PyResult<HookReply> {
            let d = PyDict::new(py);
            d.set_item("step", info.step)?;
            d.set_item("nsteps", info.nsteps)?;
            d.set_item("dt", info.dt)?;
            d.set_item("elapsed", info.elapsed)?;
            d.set_item("decay", info.decay)?;
            d.set_item("shutoff", info.shutoff)?;
            if let Some(s) = &info.slice {
                let sd = PyDict::new(py);
                sd.set_item("comp", s.comp.name())?;
                sd.set_item("axis", s.axis)?;
                sd.set_item("index", s.index)?;
                sd.set_item("data", arr(py, &s.data, &s.dims)?)?;
                d.set_item("slice", sd)?;
            }
            let r = f.call1(py, (d,))?;
            let r = r.bind(py);
            let mut reply = HookReply::default();
            if let Ok(rd) = r.cast::<PyDict>() {
                if let Some(v) = rd.get_item("stop")? {
                    reply.stop = v.is_truthy()?;
                }
                if let Some(v) = rd.get_item("slice")? {
                    if !v.is_none() {
                        let (c, a, i): (String, usize, usize) = v.extract()?;
                        let comp = lightsim_core::Component::parse(&c).ok_or_else(|| PyRuntimeError::new_err(format!("unknown component {c}")))?;
                        reply.slice = Some((comp, a, i));
                    }
                }
            }
            Ok(reply)
        })
        .unwrap_or_else(|e| {
            Python::attach(|py| e.print(py));
            HookReply { stop: true, slice: None }
        })
    };
    lightsim_core::run::RunHook { every_s: every, f: std::sync::Arc::new(call) }
}

/// One grid plane of the rasterised permittivity, as the engine uses it
/// (see `lightsim_core::inspect::cut`): normal `axis`, cell `index` along
/// it. Returns a dict with `axes`, `eps` (three arrays, E_x, E_y, E_z, each
/// `[u, v]`) and `medium` (index into `media_info`, same layout).
#[pyfunction]
fn material_cut<'py>(py: Python<'py>, spec: &str, axis: usize, index: usize) -> PyResult<Bound<'py, PyDict>> {
    let sim = parse(spec)?;
    let c = py.detach(|| lightsim_core::inspect::cut(&sim, axis, index)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("axis", c.axis)?;
    d.set_item("index", c.index)?;
    d.set_item("axes", c.axes.to_vec())?;
    let dims = c.dims.to_vec();
    let eps = PyList::empty(py);
    let med = PyList::empty(py);
    for a in 0..3 {
        eps.append(c.eps[a].clone().into_pyarray(py).reshape(dims.clone())?)?;
        med.append(c.medium[a].clone().into_pyarray(py).reshape(dims.clone())?)?;
    }
    d.set_item("eps", eps)?;
    d.set_item("medium", med)?;
    Ok(d)
}

/// The media table of a spec, in the order `material_cut` indexes it: per
/// medium a dict of `name`, `eps`, `sigma` (S/m, per axis), `pec`,
/// `dispersive`, `nonlinear`, `lossy`, `magnetic`, `anisotropic`.
#[pyfunction]
fn media_info<'py>(py: Python<'py>, spec: &str) -> PyResult<Bound<'py, PyList>> {
    let sim = parse(spec)?;
    let media = sim.canonical().media_table().map_err(err)?;
    let out = PyList::empty(py);
    for m in &media {
        let d = PyDict::new(py);
        d.set_item("name", &m.name)?;
        d.set_item("eps", m.eps.to_vec())?;
        d.set_item("sigma", m.sigma.to_vec())?;
        d.set_item("pec", m.pec)?;
        d.set_item("dispersive", m.is_dispersive())?;
        d.set_item("nonlinear", m.is_nonlinear())?;
        d.set_item("lossy", m.is_lossy())?;
        d.set_item("magnetic", m.is_magnetic())?;
        d.set_item("anisotropic", m.is_tensor() || m.eps[0] != m.eps[1] || m.eps[1] != m.eps[2])?;
        out.append(d)?;
    }
    Ok(out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // "f64" in a double-precision build (feature f64), else "f32"
    m.add("precision", if lightsim_core::REAL_IS_F64 { "f64" } else { "f32" })?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(build_grid, m)?)?;
    m.add_function(wrap_pyfunction!(time_info, m)?)?;
    m.add_function(wrap_pyfunction!(plane_info, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(port_power, m)?)?;
    m.add_function(wrap_pyfunction!(signal_spectrum, m)?)?;
    autodiff::register(m)?;
    fem::register(m)?;
    m.add_function(wrap_pyfunction!(material_cut, m)?)?;
    m.add_function(wrap_pyfunction!(media_info, m)?)?;
    plugin::register(m)?;
    Ok(())
}
