//! Cross-section ("plane") information for mode solving: the transverse Yee
//! sub-grid of a source/monitor plane and the sub-pixel-smoothed permittivity
//! of the three E components on it, exactly as the FDTD engine sees them.

use crate::geometry::plane_axes;
use crate::grid::Component;
use crate::plan::transverse_box;
use crate::raster::{RasterOptions, rasterize_component, resolve};
use crate::spec::Simulation;
use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct PlaneInfo {
    /// Normal axis and node index of the plane.
    pub axis: usize,
    pub p: usize,
    /// Natural-order transverse axes (u, v).
    pub axes: [usize; 2],
    /// Global index of the box corner (node indices) along all axes.
    pub origin: [usize; 3],
    /// Number of cells of the box along (u, v).
    pub dims: [usize; 2],
    /// Node coordinates along u and v (dims + 1 entries each).
    pub u_nodes: Vec<f64>,
    pub v_nodes: Vec<f64>,
    /// E_u, E_v and E_normal permittivity on the box, `[iu][iv]` row-major,
    /// `dims[0] * dims[1]` entries each (E_u at (u half, v node), E_v at
    /// (u node, v half), E_normal at (u node, v node)).
    pub eps_u: Vec<f64>,
    pub eps_v: Vec<f64>,
    pub eps_n: Vec<f64>,
    /// Imaginary parts (and the real-part change of dispersive media) at the
    /// requested frequency, same layout; all zero without a frequency.
    pub eps_u_im: Vec<f64>,
    pub eps_v_im: Vec<f64>,
    pub eps_n_im: Vec<f64>,
    /// Distance from the H plane (half a cell below) to the E plane.
    pub h_offset: f64,
    /// Plane coordinate along the normal axis.
    pub coord: f64,
}

/// Compute the plane information for a plane normal to `axis` at `position`
/// with transverse extent `[min, max]`.
pub fn plane_info(sim: &Simulation, axis: usize, position: f64, min: [f64; 3], max: [f64; 3]) -> Result<PlaneInfo> {
    plane_info_at(sim, axis, position, min, max, None)
}

/// As [`plane_info`], with the complex permittivity of lossy and dispersive
/// media evaluated at `freq` (Hz): eps = smoothed eps_inf + (eps(w) - eps_inf)
/// of the dominant medium of each Yee cell (fields ~ exp(-i w t), so loss is
/// Im eps > 0).
pub fn plane_info_at(sim: &Simulation, axis: usize, position: f64, min: [f64; 3], max: [f64; 3], freq: Option<f64>) -> Result<PlaneInfo> {
    let sim = &*sim.canonical();
    if sim.invariant() == Some(axis) {
        anyhow::bail!("a port plane cannot be normal to the invariant axis of a 2-D simulation");
    }
    let grid = sim.build_grid()?;
    let n = grid.n();
    let p = grid.nearest_node(axis, position).max(1);
    let (lo, hi) = transverse_box(&grid, axis, Some(&min), Some(&max));
    let (u, v) = plane_axes(axis);
    let du = (hi[u] - lo[u]).max(1);
    let dv = (hi[v] - lo[v]).max(1);
    let media = sim.media_table()?;
    let bg = media.iter().position(|m| m.name == sim.background).unwrap();
    let structs = resolve(&sim.structures, &media)?;
    let mut ropts = RasterOptions { smoothing: sim.subpixel, slice: sim.raster_slice(&grid), ..Default::default() };
    // the smoothing of transparent dispersive dielectrics, as the run does it
    // (at the sources' centre frequency); a simulation without sources yet
    // (the mode is usually solved before its source is added) is smoothed at
    // the frequency of the solve, which is the centre of the run's source
    // when the mode is solved there
    let dt = sim.time_step(&grid)?;
    ropts.disp_omega = if ropts.smoothing { crate::plan::source_center_omega(&sim.sources).or(freq.map(|f| std::f64::consts::TAU * f)) } else { None };
    ropts.disp_dt = dt;
    let dt_courant = crate::plan::courant_step(&grid, sim.invariant());
    let mut memo = std::collections::HashMap::new();
    // x-slab to rasterize
    let (x0, nxl) = if axis == 0 { (p, 1) } else { (lo[0], (hi[0] - lo[0] + 1).min(n[0] - lo[0])) };
    let mut eps: Vec<Vec<f64>> = vec![];
    let mut eps_im: Vec<Vec<f64>> = vec![];
    for (which, comp_axis) in [(0usize, u), (1usize, v), (2usize, axis)] {
        let comp = Component::e(comp_axis);
        let mut cm = rasterize_component(&grid, comp, x0, nxl, &structs, &media, bg, &ropts);
        let (weight, _) = crate::plan::apply_mixed_cells(&mut cm, &media, comp_axis, dt, dt_courant, &mut memo);
        let mut out = vec![0.0; du * dv];
        let mut out_im = vec![0.0; du * dv];
        for iu in 0..du {
            for iv in 0..dv {
                let mut g = [0usize; 3];
                g[axis] = p;
                g[u] = (lo[u] + iu).min(n[u] - 1);
                g[v] = (lo[v] + iv).min(n[v] - 1);
                let li = g[0] - x0;
                let ci = cm.idx(li.min(nxl - 1), g[1], g[2]);
                out[iu * dv + iv] = cm.value[ci] as f64;
                if let Some(f) = freq {
                    let md = &media[cm.medium[ci] as usize];
                    if md.is_lossy() || md.is_dispersive() {
                        let ec = md.eps_complex(comp_axis, std::f64::consts::TAU * f);
                        // (a mixed voxel carries its medium's poles with a weight)
                        let w = weight.get(ci).filter(|w| !w.is_nan()).map(|&w| w as f64).unwrap_or(1.0);
                        out[iu * dv + iv] += w * (ec.re - md.eps[comp_axis]);
                        out_im[iu * dv + iv] = w * ec.im;
                    }
                }
            }
        }
        let _ = which;
        eps.push(out);
        eps_im.push(out_im);
    }
    let u_nodes = (0..=du).map(|t| grid.edges(u)[(lo[u] + t).min(n[u])]).collect();
    let v_nodes = (0..=dv).map(|t| grid.edges(v)[(lo[v] + t).min(n[v])]).collect();
    Ok(PlaneInfo {
        axis,
        p,
        axes: [u, v],
        origin: lo,
        dims: [du, dv],
        u_nodes,
        v_nodes,
        eps_u: eps[0].clone(),
        eps_v: eps[1].clone(),
        eps_n: eps[2].clone(),
        eps_u_im: eps_im[0].clone(),
        eps_v_im: eps_im[1].clone(),
        eps_n_im: eps_im[2].clone(),
        h_offset: grid.node(axis, p) - grid.center(axis, p - 1),
        coord: grid.node(axis, p),
    })
}
