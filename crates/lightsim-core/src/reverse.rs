//! Time reversal of the leapfrog (cf. FDTDX's time-reversal machinery).
//!
//! The Yee update is exactly invertible:
//!
//! ```text
//! E^n = (E^{n+1} - cb curl H^{n+1/2}) / ca,    H^{n-1/2} = H^{n+1/2} + ch curl E^n
//! ```
//!
//! Backends implement it as `Backend::reverse_e` / `reverse_h` (the forward
//! kernels with coefficients -cb/ca, 1/ca, -ch). Absorbing PML layers are not
//! reversible, so their fields are recorded on the forward run ([`record`])
//! and replayed between the two half steps ([`restore`]).

use crate::engine::Backend;
use crate::grid::Component;
use crate::plan::Plan;
use anyhow::Result;
use std::time::Instant;

const COMPS: [Component; 6] = [Component::Ex, Component::Ey, Component::Ez, Component::Hx, Component::Hy, Component::Hz];

/// Storage indices of the cells inside the PML layers.
pub fn pml_cells(plan: &Plan) -> Vec<usize> {
    let (ny, nz) = (plan.ny(), plan.nz());
    let mut mask = vec![false; plan.storage_len()];
    for sl in &plan.pml {
        for i in sl.start[0]..sl.start[0] + sl.size[0] {
            for j in sl.start[1]..sl.start[1] + sl.size[1] {
                for k in sl.start[2]..sl.start[2] + sl.size[2] {
                    mask[((i + 1) * ny + j) * nz + k] = true;
                }
            }
        }
    }
    mask.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i).collect()
}

fn sets(plan: &Plan) -> &'static [bool] {
    if plan.bloch { &[false, true] } else { &[false] }
}

/// Record the E and H fields inside the PML layers (both sets in Bloch mode).
pub fn record(be: &mut dyn Backend, cells: &[usize]) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    if cells.is_empty() {
        return Ok(out);
    }
    for &im in sets(be.plan()) {
        be.select_set(im)?;
        for c in COMPS {
            let f = be.field(c)?;
            out.extend(cells.iter().map(|&i| f[i]));
        }
    }
    be.select_set(false)?;
    Ok(out)
}

/// Overwrite the PML-layer E (`e`) or H fields from a recording.
pub fn restore(be: &mut dyn Backend, cells: &[usize], rec: &[f32], e: bool) -> Result<()> {
    if cells.is_empty() {
        return Ok(());
    }
    let n = cells.len();
    let ss = sets(be.plan());
    for (si, &im) in ss.iter().enumerate() {
        be.select_set(im)?;
        for (ci, c) in COMPS.iter().enumerate() {
            if c.is_e() != e {
                continue;
            }
            let mut f = be.field(*c)?;
            let base = (si * 6 + ci) * n;
            for (t, &i) in cells.iter().enumerate() {
                f[i] = rec[base + t];
            }
            be.set_field(*c, &f)?;
        }
    }
    be.select_set(false)?;
    Ok(())
}

/// One forward leapfrog step with sources (no monitors).
pub fn step(be: &mut dyn Backend, n: usize) -> Result<()> {
    be.fill_ghost_e()?;
    be.update_h()?;
    be.inject_h(n)?;
    be.fill_ghost_h()?;
    be.update_e()?;
    be.finish_e()?;
    be.inject_e(n)
}

/// Interior field values (both sets in Bloch mode).
pub fn interior(be: &mut dyn Backend) -> Result<Vec<f32>> {
    let p = be.plan().clone();
    let plane = p.plane();
    let mut out = vec![];
    for &im in sets(&p) {
        be.select_set(im)?;
        for c in COMPS {
            let f = be.field(c)?;
            out.extend_from_slice(&f[plane..(p.nxl + 1) * plane]);
        }
    }
    be.select_set(false)?;
    Ok(out)
}

/// Outcome of a forward-then-backward run.
#[derive(Clone, Debug)]
pub struct Reversal {
    /// max |reconstructed - original| / peak over the interior
    pub error: f64,
    /// max |final - original| / peak (how far the fields moved)
    pub moved: f64,
    /// seconds of the forward window and of the reversal (excluding the lead-in)
    pub forward_s: f64,
    pub reverse_s: f64,
}

/// Run `n1` forward steps (lead-in), snapshot, run to `n2` (recording the PML
/// layers), step back to `n1` and compare with the snapshot.
pub fn reversal(be: &mut dyn Backend, n1: usize, n2: usize) -> Result<Reversal> {
    let cells = pml_cells(be.plan());
    for n in 0..n1 {
        step(be, n)?;
    }
    let snap = interior(be)?;
    let mut rec = vec![record(be, &cells)?];
    be.synchronize()?;
    let t0 = Instant::now();
    for n in n1..n2 {
        step(be, n)?;
        rec.push(record(be, &cells)?);
    }
    be.synchronize()?;
    let forward_s = t0.elapsed().as_secs_f64();
    let fin = interior(be)?;
    let t1 = Instant::now();
    for t in (1..rec.len()).rev() {
        restore(be, &cells, &rec[t], true)?;
        restore(be, &cells, &rec[t], false)?;
        be.reverse_e()?;
        restore(be, &cells, &rec[t - 1], true)?;
        be.reverse_h()?;
        restore(be, &cells, &rec[t - 1], false)?;
    }
    be.synchronize()?;
    let reverse_s = t1.elapsed().as_secs_f64();
    let back = interior(be)?;
    let peak = snap.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let err = back.iter().zip(&snap).fold(0f32, |m, (a, b)| m.max((a - b).abs())) as f64;
    let moved = fin.iter().zip(&snap).fold(0f32, |m, (a, b)| m.max((a - b).abs())) as f64;
    Ok(Reversal { error: err / peak, moved: moved / peak, forward_s, reverse_s })
}
