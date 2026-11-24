//! Bindings of the finite-element crate (`lightsim-fem`), used by the Python
//! package `lightsim.fem`. Meshes travel as arrays: `nodes` (N, 2),
//! `tris` (M, 3), `tags` (M,), `edge_mid` (E, 2) with NaN for straight edges;
//! the edge numbering is the one `fem_mesh_info` returns for the same
//! triangles.

use lightsim_fem::element::{Lagrange, geometry, lagrange_ref_grads, lagrange_values};
use lightsim_fem::mesh::{Mesh, NONE};
use lightsim_fem::mesher::{MeshOptions, Refinement, Region, mesh_regions};
use lightsim_fem::scalar::{self, LinearSolver, ScalarProblem};
use lightsim_fem::maxwell::{ModeProblem, Pml};
use lightsim_fem::{locate, quadrature};
use num_complex::Complex64;
use rayon::prelude::*;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use crate::Compute;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn f64s(a: &PyReadonlyArrayDyn<f64>) -> Vec<f64> {
    a.as_array().iter().copied().collect()
}
fn i64s(a: &PyReadonlyArrayDyn<i64>) -> Vec<i64> {
    a.as_array().iter().copied().collect()
}

fn arr2<'py>(py: Python<'py>, v: Vec<f64>, cols: usize) -> PyResult<Bound<'py, PyAny>> {
    let n = v.len() / cols.max(1);
    Ok(v.into_pyarray(py).reshape(vec![n, cols])?.into_any())
}
fn iarr2<'py>(py: Python<'py>, v: Vec<i64>, cols: usize) -> PyResult<Bound<'py, PyAny>> {
    let n = v.len() / cols.max(1);
    Ok(v.into_pyarray(py).reshape(vec![n, cols])?.into_any())
}

/// Rebuild a mesh from arrays.
pub(crate) fn mesh_from(
    nodes: &PyReadonlyArrayDyn<f64>,
    tris: &PyReadonlyArrayDyn<i64>,
    tags: &PyReadonlyArrayDyn<i64>,
    edge_mid: Option<&PyReadonlyArrayDyn<f64>>,
) -> PyResult<Mesh> {
    let n = f64s(nodes);
    let t = i64s(tris);
    let g = i64s(tags);
    if !n.len().is_multiple_of(2) || !t.len().is_multiple_of(3) {
        return Err(PyValueError::new_err(
            "nodes must be (N, 2) and tris (M, 3)",
        ));
    }
    let nodes: Vec<[f64; 2]> = n.chunks(2).map(|c| [c[0], c[1]]).collect();
    let mut tv = Vec::with_capacity(t.len() / 3);
    for c in t.chunks(3) {
        if c.iter().any(|&x| x < 0) {
            return Err(PyValueError::new_err("negative node index in tris"));
        }
        tv.push([c[0] as usize, c[1] as usize, c[2] as usize]);
    }
    let tags: Vec<u32> = g.iter().map(|&x| x as u32).collect();
    let mut mesh = Mesh::new(nodes, tv, tags).map_err(err)?;
    if let Some(em) = edge_mid {
        let em = f64s(em);
        if em.len() == 2 * mesh.n_edges() {
            for e in 0..mesh.n_edges() {
                if em[2 * e].is_finite() && em[2 * e + 1].is_finite() {
                    mesh.edge_mid[e] = Some([em[2 * e], em[2 * e + 1]]);
                }
            }
        } else if !em.is_empty() {
            return Err(PyValueError::new_err(format!(
                "edge_mid has {} rows, the mesh has {} edges",
                em.len() / 2,
                mesh.n_edges()
            )));
        }
    }
    Ok(mesh)
}

pub(crate) fn mesh_dict<'py>(py: Python<'py>, m: &Mesh) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "nodes",
        arr2(py, m.nodes.iter().flat_map(|p| [p[0], p[1]]).collect(), 2)?,
    )?;
    d.set_item(
        "tris",
        iarr2(
            py,
            m.tris.iter().flat_map(|t| t.map(|v| v as i64)).collect(),
            3,
        )?,
    )?;
    d.set_item(
        "tags",
        m.tags
            .iter()
            .map(|&t| t as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "edges",
        iarr2(
            py,
            m.edges.iter().flat_map(|e| e.map(|v| v as i64)).collect(),
            2,
        )?,
    )?;
    d.set_item(
        "tri_edges",
        iarr2(
            py,
            m.tri_edges
                .iter()
                .flat_map(|e| e.map(|v| v as i64))
                .collect(),
            3,
        )?,
    )?;
    d.set_item(
        "edge_tris",
        iarr2(
            py,
            m.edge_tris
                .iter()
                .flat_map(|e| e.map(|v| if v == NONE { -1 } else { v as i64 }))
                .collect(),
            2,
        )?,
    )?;
    d.set_item(
        "edge_mid",
        arr2(
            py,
            m.edge_mid
                .iter()
                .flat_map(|e| e.unwrap_or([f64::NAN; 2]))
                .collect(),
            2,
        )?,
    )?;
    Ok(d)
}

/// Mesh tagged polygon regions. Each region is a dict with `rings` (list of
/// (n, 2) arrays), `arcs` (list per ring of per-segment `None` or
/// `(cx, cy, r)`), `tag`, `priority` and `size`.
#[pyfunction]
#[pyo3(signature = (regions, default_size, grading=0.3, min_angle=25.0, refinements=vec![], max_vertices=2_000_000))]
fn fem_mesh<'py>(
    py: Python<'py>,
    regions: Vec<Bound<'py, PyDict>>,
    default_size: f64,
    grading: f64,
    min_angle: f64,
    refinements: Vec<(Vec<usize>, f64)>,
    max_vertices: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut regs = vec![];
    for r in &regions {
        let rings: Vec<Vec<[f64; 2]>> = r
            .get_item("rings")?
            .ok_or_else(|| PyValueError::new_err("region without rings"))?
            .extract::<Vec<Vec<[f64; 2]>>>()?;
        let arcs: Vec<Vec<Option<[f64; 3]>>> = match r.get_item("arcs")? {
            Some(a) if !a.is_none() => a.extract()?,
            _ => vec![],
        };
        let get = |k: &str| -> PyResult<Option<Bound<'py, PyAny>>> { r.get_item(k) };
        regs.push(Region {
            rings,
            arcs,
            tag: get("tag")?
                .map(|v| v.extract::<u32>())
                .transpose()?
                .unwrap_or(0),
            priority: get("priority")?
                .map(|v| v.extract::<i32>())
                .transpose()?
                .unwrap_or(0),
            size: get("size")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.0),
        });
    }
    let opts = MeshOptions {
        default_size,
        grading,
        min_angle,
        refinements: refinements
            .into_iter()
            .map(|(regions, size)| Refinement { regions, size })
            .collect(),
        max_vertices,
    };
    let (m, st) = py.compute(|| mesh_regions(&regs, &opts)).map_err(err)?;
    let d = mesh_dict(py, &m)?;
    d.set_item("min_angle", st.min_angle_deg)?;
    d.set_item("refinement_complete", st.refinement_complete)?;
    Ok(d)
}

/// Edge structure of a mesh given by nodes and triangles (triangles are
/// re-oriented counter-clockwise).
#[pyfunction]
fn fem_mesh_info<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = mesh_from(&nodes, &tris, &tags, None)?;
    mesh_dict(py, &m)
}

fn solver_of(s: &str) -> PyResult<LinearSolver> {
    Ok(match s {
        "cholesky" | "direct" => LinearSolver::Cholesky,
        "lu" => LinearSolver::Lu,
        "cg" | "iterative" => LinearSolver::Cg,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown solver '{s}' (direct, lu, cg)"
            )));
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn problem(
    order: usize,
    k: &PyReadonlyArrayDyn<f64>,
    c: &PyReadonlyArrayDyn<f64>,
    f: &PyReadonlyArrayDyn<f64>,
    neumann: (Vec<usize>, Vec<f64>),
    robin: (Vec<usize>, Vec<f64>, Vec<f64>),
) -> ScalarProblem {
    let kv = f64s(k);
    ScalarProblem {
        order,
        k: kv.chunks(3).map(|x| [x[0], x[1], x[2]]).collect(),
        c: f64s(c),
        f: f64s(f),
        dirichlet: vec![],
        neumann: neumann.0.into_iter().zip(neumann.1).collect(),
        robin: robin
            .0
            .into_iter()
            .zip(robin.1)
            .zip(robin.2)
            .map(|((e, h), u)| (e, h, u))
            .collect(),
    }
}

fn csr_dict<'py>(
    py: Python<'py>,
    a: &lightsim_fem::sparse::Csr<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(
        "indptr",
        a.indptr
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "indices",
        a.indices
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py),
    )?;
    d.set_item("data", a.values.clone().into_pyarray(py))?;
    d.set_item("shape", (a.nrows, a.ncols))?;
    Ok(d)
}

/// Steady scalar problem -div(K grad u) (+ c u) = f. `k` (M, 3) holds
/// (Kxx, Kyy, Kxy) per triangle; `dir_vals` is (ncase, len(dir_dofs)): one
/// solve per row with a single factorisation. `load` (optional, ndof)
/// replaces the assembled load. Returns `u` (ncase, ndof) and, with
/// `matrices=True`, the stiffness and capacity matrices and the load.
#[pyfunction]
#[pyo3(signature = (nodes, tris, tags, edge_mid, order, k, c, f, dir_dofs, dir_vals, neumann, robin, solver="direct", reaction=false, matrices=false, load=None))]
#[allow(clippy::too_many_arguments)]
fn fem_scalar_solve<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    edge_mid: Option<PyReadonlyArrayDyn<f64>>,
    order: usize,
    k: PyReadonlyArrayDyn<f64>,
    c: PyReadonlyArrayDyn<f64>,
    f: PyReadonlyArrayDyn<f64>,
    dir_dofs: Vec<usize>,
    dir_vals: Vec<Vec<f64>>,
    neumann: (Vec<usize>, Vec<f64>),
    robin: (Vec<usize>, Vec<f64>, Vec<f64>),
    solver: &str,
    reaction: bool,
    matrices: bool,
    load: Option<PyReadonlyArrayDyn<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mesh = mesh_from(&nodes, &tris, &tags, edge_mid.as_ref())?;
    let p = problem(order, &k, &c, &f, neumann, robin);
    let solver = solver_of(solver)?;
    let load = load.as_ref().map(f64s);
    let (a, us) = py
        .compute(|| -> anyhow::Result<_> {
            let mut a = scalar::assemble(&mesh, &p)?;
            if let Some(l) = load {
                if l.len() != a.space.ndof {
                    anyhow::bail!(
                        "load has {} entries, the space has {} DOFs",
                        l.len(),
                        a.space.ndof
                    );
                }
                a.load = l;
            }
            let mat = if reaction {
                lightsim_fem::sparse::Csr::axpby(1.0, &a.stiffness, 1.0, &a.capacity)
            } else {
                a.stiffness.clone()
            };
            let us = scalar::solve_reduced_multi(
                &mat,
                &[a.load.clone()],
                &dir_dofs,
                &dir_vals,
                a.space.ndof,
                solver,
            )?;
            Ok((a, us))
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    let ndof = a.space.ndof;
    d.set_item("u", arr2(py, us.into_iter().flatten().collect(), ndof)?)?;
    d.set_item("ndof", ndof)?;
    if matrices {
        d.set_item("stiffness", csr_dict(py, &a.stiffness)?)?;
        d.set_item("capacity", csr_dict(py, &a.capacity)?)?;
        d.set_item("load", a.load.clone().into_pyarray(py))?;
    }
    Ok(d)
}

/// Transient scalar problem c du/dt - div(K grad u) = f by the theta scheme.
/// `probes` is a list of (dofs, weights) combinations recorded every step.
#[pyfunction]
#[pyo3(signature = (nodes, tris, tags, edge_mid, order, k, c, f, dir_dofs, dir_vals, neumann, robin, u0, dt, nsteps, theta=1.0, save=vec![], probes=vec![]))]
#[allow(clippy::too_many_arguments)]
fn fem_scalar_transient<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    edge_mid: Option<PyReadonlyArrayDyn<f64>>,
    order: usize,
    k: PyReadonlyArrayDyn<f64>,
    c: PyReadonlyArrayDyn<f64>,
    f: PyReadonlyArrayDyn<f64>,
    dir_dofs: Vec<usize>,
    dir_vals: Vec<f64>,
    neumann: (Vec<usize>, Vec<f64>),
    robin: (Vec<usize>, Vec<f64>, Vec<f64>),
    u0: PyReadonlyArrayDyn<f64>,
    dt: f64,
    nsteps: usize,
    theta: f64,
    save: Vec<usize>,
    probes: Vec<(Vec<usize>, Vec<f64>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let mesh = mesh_from(&nodes, &tris, &tags, edge_mid.as_ref())?;
    let p = problem(order, &k, &c, &f, neumann, robin);
    let u0 = f64s(&u0);
    let dir: Vec<(usize, f64)> = dir_dofs.into_iter().zip(dir_vals).collect();
    let probes: Vec<Vec<(usize, f64)>> = probes
        .into_iter()
        .map(|(d, w)| d.into_iter().zip(w).collect())
        .collect();
    let (ndof, saved, traces) = py
        .compute(|| -> anyhow::Result<_> {
            let a = scalar::assemble(&mesh, &p)?;
            if u0.len() != a.space.ndof {
                anyhow::bail!(
                    "u0 has {} entries, the space has {} DOFs",
                    u0.len(),
                    a.space.ndof
                );
            }
            let (s, t) = scalar::solve_transient(&a, &dir, &u0, dt, nsteps, theta, &save, &probes)?;
            Ok((a.space.ndof, s, t))
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item(
        "saved",
        arr2(py, saved.into_iter().flatten().collect(), ndof)?,
    )?;
    d.set_item(
        "traces",
        arr2(py, traces.into_iter().flatten().collect(), nsteps + 1)?,
    )?;
    Ok(d)
}

/// Values and gradients of Lagrange fields at points (NaN outside). `u` is
/// (ncomp, ndof); returns values (ncomp, P) and gradients (ncomp, P, 2).
#[pyfunction]
#[allow(clippy::too_many_arguments)] // (the Python signature)
fn fem_eval<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    edge_mid: Option<PyReadonlyArrayDyn<f64>>,
    order: usize,
    u: PyReadonlyArrayDyn<f64>,
    points: PyReadonlyArrayDyn<f64>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let mesh = mesh_from(&nodes, &tris, &tags, edge_mid.as_ref())?;
    let space = Lagrange::new(&mesh, order);
    let uv = f64s(&u);
    if !uv.len().is_multiple_of(space.ndof) {
        return Err(PyValueError::new_err(format!(
            "u has {} entries, not a multiple of the {} DOFs",
            uv.len(),
            space.ndof
        )));
    }
    let pts: Vec<[f64; 2]> = f64s(&points).chunks(2).map(|c| [c[0], c[1]]).collect();
    let np = pts.len();
    let (vals, grads) = py.compute(|| {
        let mut vals = vec![];
        let mut grads = vec![];
        for comp in uv.chunks(space.ndof) {
            let (v, g) = locate::eval_lagrange(&mesh, &space, comp, &pts);
            vals.extend(v);
            grads.extend(g.into_iter().flatten());
        }
        (vals, grads)
    });
    let nc = uv.len() / space.ndof;
    Ok((
        vals.into_pyarray(py).reshape(vec![nc, np])?.into_any(),
        grads.into_pyarray(py).reshape(vec![nc, np, 2])?.into_any(),
    ))
}

/// Quadrature of degree `degree` over every triangle with a Lagrange field:
/// points (Q, 2), weights (Q,) including the area element, triangle index
/// (Q,), values (Q,) and gradients (Q, 2) of `u` (empty `u`: no values).
#[pyfunction]
#[pyo3(signature = (nodes, tris, tags, edge_mid, order, u, degree=6))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fem_quadrature<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    edge_mid: Option<PyReadonlyArrayDyn<f64>>,
    order: usize,
    u: PyReadonlyArrayDyn<f64>,
    degree: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mesh = mesh_from(&nodes, &tris, &tags, edge_mid.as_ref())?;
    let space = Lagrange::new(&mesh, order);
    let uv = f64s(&u);
    let has_u = uv.len() == space.ndof;
    let rule = quadrature::triangle(degree);
    let (mut pts, mut ws, mut tri, mut vals, mut grads) = (vec![], vec![], vec![], vec![], vec![]);
    for t in 0..mesh.n_tris() {
        for (l, w) in rule.points.iter().zip(&rule.weights) {
            let g = geometry(&mesh, t, l);
            pts.extend(g.x);
            ws.push(w * 0.5 * g.det);
            tri.push(t as i64);
            if has_u {
                let phi = lagrange_values(order, l);
                let rg = lagrange_ref_grads(order, l);
                let (mut v, mut gr) = (0.0, [0.0; 2]);
                for i in 0..space.nloc() {
                    let c = uv[space.dofs[t][i]];
                    v += c * phi[i];
                    let gi = g.grad(&rg[i]);
                    gr[0] += c * gi[0];
                    gr[1] += c * gi[1];
                }
                vals.push(v);
                grads.extend(gr);
            }
        }
    }
    let d = PyDict::new(py);
    d.set_item("points", arr2(py, pts, 2)?)?;
    d.set_item("weights", ws.into_pyarray(py))?;
    d.set_item("tri", tri.into_pyarray(py))?;
    if has_u {
        d.set_item("values", vals.into_pyarray(py))?;
        d.set_item("grads", arr2(py, grads, 2)?)?;
    }
    Ok(d)
}

/// Index of the triangle containing each point (-1 outside the mesh).
#[pyfunction]
fn fem_locate<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    points: PyReadonlyArrayDyn<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let mesh = mesh_from(&nodes, &tris, &tags, None)?;
    let pts: Vec<[f64; 2]> = f64s(&points).chunks(2).map(|c| [c[0], c[1]]).collect();
    let out: Vec<i64> = py.compute(|| {
        let loc = locate::Locator::new(&mesh);
        pts.iter()
            .map(|&p| loc.locate(p).map_or(-1, |(t, _)| t as i64))
            .collect()
    });
    Ok(out.into_pyarray(py).into_any())
}

fn mode_problem(mesh: &Mesh, eps: &PyReadonlyArrayDyn<Complex64>, k0: f64, pec_edges: Vec<usize>, pml: Option<([f64; 4], [f64; 4], f64)>, bend: Option<(f64, f64)>) -> PyResult<ModeProblem> {
    let ev: Vec<Complex64> = eps.as_array().iter().copied().collect();
    if ev.len() != 4 * mesh.n_tris() {
        return Err(PyValueError::new_err(format!("eps must be (M, 4) complex: (eps_xx, eps_yy, eps_xy, eps_zz) per triangle; got {} values for {} triangles", ev.len(), mesh.n_tris())));
    }
    Ok(ModeProblem {
        k0,
        eps: ev.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect(),
        pml: pml.map(|(inner, thickness, strength)| Pml { inner, thickness, strength }),
        bend,
        pec_edges,
    })
}

/// Full-vectorial modes (Nedelec / P1). `eps` (M, 4) complex. Returns
/// `beta` (nev,) complex, `x` (nev, ndof) complex DOF vectors (edges then
/// nodes) and `integrals` (nev, 6): power, int |Ex|^2, int |Ey|^2,
/// int |E|^2, int |E|^4, energy (see `ModeProblem::integrals`).
#[pyfunction]
#[pyo3(signature = (nodes, tris, tags, eps, k0, nev, n_guess, pec_edges, pml=None, bend=None, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn fem_modes<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    eps: PyReadonlyArrayDyn<Complex64>,
    k0: f64,
    nev: usize,
    n_guess: f64,
    pec_edges: Vec<usize>,
    pml: Option<([f64; 4], [f64; 4], f64)>,
    bend: Option<(f64, f64)>,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let mesh = mesh_from(&nodes, &tris, &tags, None)?;
    let p = mode_problem(&mesh, &eps, k0, pec_edges, pml, bend)?;
    let (modes, ints) = py
        .compute(|| -> anyhow::Result<_> {
            let modes = p.solve(&mesh, nev, n_guess, tol)?;
            let ints: Vec<[f64; 6]> = modes.iter().map(|(b, x)| p.integrals(&mesh, *b, x)).collect();
            Ok((modes, ints))
        })
        .map_err(err)?;
    let n = p.n_dofs(&mesh);
    let d = PyDict::new(py);
    d.set_item("beta", modes.iter().map(|m| m.0).collect::<Vec<Complex64>>().into_pyarray(py))?;
    let xs: Vec<Complex64> = modes.iter().flat_map(|m| m.1.iter().copied()).collect();
    d.set_item("x", xs.into_pyarray(py).reshape(vec![modes.len(), n])?)?;
    d.set_item("integrals", arr2(py, ints.into_iter().flatten().collect(), 6)?)?;
    Ok(d)
}

/// E and eta0 H (complex, (P, 3) each) of a mode at points (NaN outside),
/// or at the quadrature points of degree `degree` when `points` is None
/// (then also returns `points`, `weights`, `tri`).
#[pyfunction]
#[pyo3(signature = (nodes, tris, tags, eps, k0, beta, x, points=None, pml=None, bend=None, degree=4))]
#[allow(clippy::too_many_arguments)]
fn fem_mode_fields<'py>(
    py: Python<'py>,
    nodes: PyReadonlyArrayDyn<f64>,
    tris: PyReadonlyArrayDyn<i64>,
    tags: PyReadonlyArrayDyn<i64>,
    eps: PyReadonlyArrayDyn<Complex64>,
    k0: f64,
    beta: (f64, f64),
    x: PyReadonlyArrayDyn<Complex64>,
    points: Option<PyReadonlyArrayDyn<f64>>,
    pml: Option<([f64; 4], [f64; 4], f64)>,
    bend: Option<(f64, f64)>,
    degree: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mesh = mesh_from(&nodes, &tris, &tags, None)?;
    let p = mode_problem(&mesh, &eps, k0, vec![], pml, bend)?;
    let beta = Complex64::new(beta.0, beta.1);
    let xv: Vec<Complex64> = x.as_array().iter().copied().collect();
    if xv.len() != p.n_dofs(&mesh) {
        return Err(PyValueError::new_err("x has the wrong length for this mesh"));
    }
    let nan = Complex64::new(f64::NAN, f64::NAN);
    let d = PyDict::new(py);
    let (es, hs) = match points {
        Some(pts) => {
            let pts: Vec<[f64; 2]> = f64s(&pts).chunks(2).map(|c| [c[0], c[1]]).collect();
            py.compute(|| {
                let loc = locate::Locator::new(&mesh);
                let r: Vec<([Complex64; 3], [Complex64; 3])> = pts
                    .par_iter()
                    .map(|&q| match loc.locate(q) {
                        Some((t, l)) => p.fields(&mesh, t, &[l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)], beta, &xv),
                        None => ([nan; 3], [nan; 3]),
                    })
                    .collect();
                (r.iter().flat_map(|a| a.0).collect::<Vec<_>>(), r.iter().flat_map(|a| a.1).collect::<Vec<_>>())
            })
        }
        None => {
            let rule = quadrature::triangle(degree);
            let (mut pts, mut ws, mut tri, mut es, mut hs) = (vec![], vec![], vec![], vec![], vec![]);
            for t in 0..mesh.n_tris() {
                let area = mesh.area(t);
                let tv = mesh.tris[t];
                for (l, w) in rule.points.iter().zip(&rule.weights) {
                    for a in 0..2 {
                        pts.push(l[0] * mesh.nodes[tv[0]][a] + l[1] * mesh.nodes[tv[1]][a] + l[2] * mesh.nodes[tv[2]][a]);
                    }
                    ws.push(w * area);
                    tri.push(t as i64);
                    let (e, h) = p.fields(&mesh, t, l, beta, &xv);
                    es.extend(e);
                    hs.extend(h);
                }
            }
            d.set_item("points", arr2(py, pts, 2)?)?;
            d.set_item("weights", ws.into_pyarray(py))?;
            d.set_item("tri", tri.into_pyarray(py))?;
            (es, hs)
        }
    };
    let np_ = es.len() / 3;
    d.set_item("E", es.into_pyarray(py).reshape(vec![np_, 3])?)?;
    d.set_item("H", hs.into_pyarray(py).reshape(vec![np_, 3])?)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fem_modes, m)?)?;
    m.add_function(wrap_pyfunction!(fem_mode_fields, m)?)?;
    m.add_function(wrap_pyfunction!(fem_locate, m)?)?;
    m.add_function(wrap_pyfunction!(fem_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(fem_mesh_info, m)?)?;
    m.add_function(wrap_pyfunction!(fem_scalar_solve, m)?)?;
    m.add_function(wrap_pyfunction!(fem_scalar_transient, m)?)?;
    m.add_function(wrap_pyfunction!(fem_eval, m)?)?;
    m.add_function(wrap_pyfunction!(fem_quadrature, m)?)?;
    Ok(())
}
