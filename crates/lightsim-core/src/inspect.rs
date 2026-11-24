//! Inspection of what the engine simulates, for the graphical interface:
//! cross-sections of the rasterised (sub-pixel-smoothed) permittivity of the
//! three E components and the dominant medium of every sample, on one grid
//! plane, without rasterising the whole volume.

use crate::geometry::plane_axes;
use crate::grid::{Component, Grid};
use crate::raster::{RasterOptions, rasterize_component, resolve};
use crate::spec::Simulation;
use anyhow::Result;

/// One grid plane of the rasterised material.
#[derive(Clone, Debug)]
pub struct Cut {
    /// Normal axis and cell index of the plane along it.
    pub axis: usize,
    pub index: usize,
    /// In-plane axes (u, v) in increasing order.
    pub axes: [usize; 2],
    /// Number of samples along (u, v): the cells of the grid.
    pub dims: [usize; 2],
    /// Relative permittivity of E_x, E_y and E_z (`[iu][iv]` row-major),
    /// as the engine uses it: smoothed at dielectric interfaces, eps_inf for
    /// dispersive media.
    pub eps: [Vec<f64>; 3],
    /// Index into the media table of the medium that owns each sample.
    pub medium: [Vec<u16>; 3],
}

/// The cut of the material normal to `axis` through the E samples of cell
/// index `index` along it (E components in the plane sit on the node plane
/// `index`, the normal component half a cell above). For a normal along y or
/// z only a window of three cells around the plane is rasterised; the voxel
/// bounds the smoothing integrates over are the same as in the full grid.
pub fn cut(sim: &Simulation, axis: usize, index: usize) -> Result<Cut> {
    let sim = &*sim.canonical();
    let grid = sim.build_grid()?;
    let n = grid.n();
    if axis > 2 {
        anyhow::bail!("axis must be 0, 1 or 2");
    }
    if index >= n[axis] {
        anyhow::bail!("index {index} is outside the grid ({} cells along axis {axis})", n[axis]);
    }
    let media = sim.media_table()?;
    let bg = media.iter().position(|m| m.name == sim.background).unwrap();
    let structs = resolve(&sim.structures, &media)?;
    let ropts = RasterOptions { smoothing: sim.subpixel, slice: sim.raster_slice(&grid), ..Default::default() };
    // window of cells along the normal that is rasterised
    let (g, x0, nxl, local) = if axis == 0 {
        (grid.clone(), index, 1, 0)
    } else {
        let lo = index.saturating_sub(1);
        let hi = (index + 2).min(n[axis]);
        let e = grid.edges(axis)[lo..=hi].to_vec();
        let sub = match axis {
            1 => Grid::from_edges(grid.x.clone(), e, grid.z.clone()),
            _ => Grid::from_edges(grid.x.clone(), grid.y.clone(), e),
        };
        (sub, 0, n[0], index - lo)
    };
    let (u, v) = plane_axes(axis);
    let (du, dv) = (n[u], n[v]);
    let mut eps: [Vec<f64>; 3] = Default::default();
    let mut medium: [Vec<u16>; 3] = Default::default();
    for a in 0..3 {
        let cm = rasterize_component(&g, Component::e(a), x0, nxl, &structs, &media, bg, &ropts);
        let mut out = vec![0.0; du * dv];
        let mut med = vec![0u16; du * dv];
        for iu in 0..du {
            for iv in 0..dv {
                let mut l = [0usize; 3];
                l[axis] = local;
                l[u] = iu;
                l[v] = iv;
                let ci = cm.idx(l[0], l[1], l[2]);
                out[iu * dv + iv] = cm.value[ci] as f64;
                med[iu * dv + iv] = cm.medium[ci];
            }
        }
        eps[a] = out;
        medium[a] = med;
    }
    Ok(Cut { axis, index, axes: [u, v], dims: [du, dv], eps, medium })
}
