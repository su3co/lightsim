//! `lightsim._core.read_gds`: the GDSII reader of lightsim-core.

use lightsim_core::gdsii::Library;
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use crate::Compute;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Read a GDSII file and flatten one cell.
///
/// Returns a dict: `library`, `database_unit_m`, `user_unit_m`, `cells`
/// (all names, in file order), `top_cells`, `cell` (the cell flattened:
/// `cell`, or the only top cell), `polygons` (`{(layer, datatype): [array
/// (N, 2) in metres, ...]}`) and `labels` (a list of `(text, layer,
/// texttype, x, y)` with x, y in metres) and `paths` (a list of `(layer,
/// datatype, width, pathtype, points (N, 2))` in metres: the centre lines
/// of the paths, whose outlines are also among the polygons).
#[pyfunction]
#[pyo3(signature = (path, cell=None))]
fn read_gds<'py>(py: Python<'py>, path: &str, cell: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
    let (lib, (name, flat)) = py
        .compute(|| -> Result<_, lightsim_core::gdsii::GdsError> {
            let lib = Library::read(path)?;
            let f = lib.flatten(cell)?;
            Ok((lib, f))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("library", &lib.name)?;
    d.set_item("database_unit_m", lib.db_in_m)?;
    d.set_item("user_unit_m", lib.db_in_m / lib.db_in_user)?;
    d.set_item("cells", lib.cell_names())?;
    d.set_item("top_cells", lib.top_cells())?;
    d.set_item("cell", name)?;
    let polys = PyDict::new(py);
    for ((l, dt), list) in &flat.polygons {
        let out = PyList::empty(py);
        for p in list {
            let v: Vec<f64> = p.iter().flat_map(|q| [q[0], q[1]]).collect();
            out.append(v.into_pyarray(py).reshape([p.len(), 2])?)?;
        }
        polys.set_item((*l, *dt), out)?;
    }
    d.set_item("polygons", polys)?;
    let paths = PyList::empty(py);
    for p in &flat.paths {
        let v: Vec<f64> = p.points.iter().flat_map(|q| [q[0], q[1]]).collect();
        paths.append((p.layer, p.datatype, p.width, p.pathtype, v.into_pyarray(py).reshape([p.points.len(), 2])?))?;
    }
    d.set_item("paths", paths)?;
    let labels = PyList::empty(py);
    for l in &flat.labels {
        labels.append((l.text.clone(), l.layer, l.texttype, l.position[0], l.position[1]))?;
    }
    d.set_item("labels", labels)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_gds, m)?)
}
