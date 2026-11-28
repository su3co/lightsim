//! Monitor tables: DFT boxes (fields, flux planes and boxes, modes, far
//! field), time series and energy, with the phasor tables of each and the
//! sharing of identical DFT boxes between monitors.

use super::materials::Materials;
use super::*;

/// The tables of every monitor of the simulation.
pub(super) fn compile_monitors(plan: &mut Plan, s: &Setup, mat: &Materials) -> Result<()> {
    let (sim, dt, inv) = (s.sim, s.dt, s.inv);
    let nonlinear = plan.nonlin.is_some() || s.media.iter().any(|m| m.is_nonlinear());
    for mon in &sim.monitors {
        // stride of a frequency-domain monitor: its own, the simulation's, or automatic
        let stride_of = |own: usize, freqs: &[f64]| -> usize {
            if own > 0 {
                own
            } else {
                sim.dft_stride.filter(|s| *s > 0).unwrap_or_else(|| auto_dft_stride(&sim.sources, nonlinear, dt, freqs))
            }
        };
        compile_monitor(plan, mon, &sim.boundaries, &mat.eps_local, mat.mu_local.as_ref(), &stride_of, inv)?;
    }
    for (i, d) in plan.dft.iter_mut().enumerate() {
        d.id = i;
    }
    if s.opts.share_monitors {
        share_dft_boxes(plan);
    }
    Ok(())
}

/// Component index box for positions inside [min, max].
fn comp_box(grid: &Grid, c: Component, min: &[f64; 3], max: &[f64; 3], x0: usize, nxl: usize) -> CompBox {
    let mut glo = [0usize; 3];
    let mut ghi = [0usize; 3];
    for a in 0..3 {
        let (lo, hi) = grid.index_range(c, a, min[a], max[a]);
        glo[a] = lo;
        ghi[a] = hi;
    }
    localize(c, glo, ghi, x0, nxl)
}

fn localize(c: Component, glo: [usize; 3], ghi: [usize; 3], x0: usize, nxl: usize) -> CompBox {
    let lx0 = glo[0].max(x0);
    let lx1 = ghi[0].min(x0 + nxl);
    let (lstart, lsize, gx_off) = if lx0 < lx1 {
        ([lx0 - x0, glo[1], glo[2]], [lx1 - lx0, ghi[1] - glo[1], ghi[2] - glo[2]], lx0 - glo[0])
    } else {
        ([0, glo[1], glo[2]], [0, ghi[1] - glo[1], ghi[2] - glo[2]], 0)
    };
    CompBox { comp: c, glo, ghi, lstart, lsize, gx_off }
}

fn phasor_tables(freqs: &[f64], nsteps: usize, dt: f64, apod: &Apodization, stride: usize) -> (Vec<Real>, Vec<Real>) {
    let nf = freqs.len();
    let mut pe = vec![0 as Real; nsteps * nf * 2];
    let mut ph = vec![0 as Real; nsteps * nf * 2];
    for n in 0..nsteps {
        if n % stride.max(1) != 0 {
            continue;
        }
        let w_dt = dt * stride.max(1) as f64;
        let te = (n as f64 + 1.0) * dt;
        let th = (n as f64 + 0.5) * dt;
        let we = apod.weight(te) * w_dt;
        let wh = apod.weight(th) * w_dt;
        for (m, f) in freqs.iter().enumerate() {
            let w = TAU * f;
            let o = (n * nf + m) * 2;
            pe[o] = ((w * te).cos() * we) as Real;
            pe[o + 1] = ((w * te).sin() * we) as Real;
            ph[o] = ((w * th).cos() * wh) as Real;
            ph[o + 1] = ((w * th).sin() * wh) as Real;
        }
    }
    (pe, ph)
}

/// Closed box snapped to grid nodes (faces of flux/far-field boxes sit on
/// nodes; snapping the transverse extents too keeps the box closed).
fn snapped_box(grid: &Grid, min: &[f64; 3], max: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    // no clamping: a face at node 0 is either on a PEC/PMC wall (skipped) or
    // moved to node 1 where the face plane is built
    for a in 0..3 {
        lo[a] = grid.node(a, grid.nearest_node(a, min[a]));
        hi[a] = grid.node(a, grid.nearest_node(a, max[a]).max(1));
    }
    (lo, hi)
}

/// Boxes for a flux plane normal to `a` at node `p` with transverse extent:
/// [E_b(p), E_c(p), H_b(p-1), H_c(p-1), H_b(p), H_c(p)].
fn flux_boxes(grid: &Grid, a: usize, p: usize, min: &[f64; 3], max: &[f64; 3], x0: usize, nxl: usize) -> Vec<CompBox> {
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let mut out = vec![];
    let mk = |c: Component, pa: usize| -> CompBox {
        let mut glo = [0usize; 3];
        let mut ghi = [0usize; 3];
        for ax in 0..3 {
            if ax == a {
                glo[ax] = pa;
                ghi[ax] = pa + 1;
            } else {
                let (lo, hi) = grid.index_range(c, ax, min[ax], max[ax]);
                glo[ax] = lo;
                ghi[ax] = hi;
            }
        }
        localize(c, glo, ghi, x0, nxl)
    };
    out.push(mk(Component::e(bax), p));
    out.push(mk(Component::e(cax), p));
    let pm = p.saturating_sub(1);
    out.push(mk(Component::h(bax), pm));
    out.push(mk(Component::h(cax), pm));
    out.push(mk(Component::h(bax), p.min(grid.n()[a] - 1)));
    out.push(mk(Component::h(cax), p.min(grid.n()[a] - 1)));
    out
}

/// Stride of the Fourier sums of a frequency-domain monitor that is safe
/// for the given sources: sampling every `s` steps folds the spectrum at
/// `1 / (s dt)`, so a component of the fields at frequency `f'` lands on the
/// monitor frequency `f` if `f' = 1/(s dt) - f`. The stride keeps
/// `1 / (s dt) >= f_monitor,max + f_source,max`, with `f_source,max` seven
/// widths above the centre of a Gaussian pulse (spectral amplitude 2e-11 of
/// the peak). Sources of other kinds (continuous wave, Ricker, sampled) have
/// no such edge, and chi2/chi3 media create harmonics: the stride is then 1.
pub fn auto_dft_stride(sources: &[Source], nonlinear: bool, dt: f64, freqs: &[f64]) -> usize {
    if nonlinear || sources.is_empty() || freqs.is_empty() {
        return 1;
    }
    let mut f_src = 0.0f64;
    for s in sources {
        match s.signal() {
            TimeSignal::GaussianPulse { freq0, fwidth, .. } => f_src = f_src.max(freq0.abs() + 7.0 * fwidth.abs()),
            _ => return 1,
        }
    }
    let f_mon = freqs.iter().cloned().fold(0.0, f64::max);
    let s = (1.0 / (dt * (f_mon + f_src))).floor();
    if s.is_finite() { (s as usize).clamp(1, 1000) } else { 1 }
}

/// Mark boxes that an earlier monitor already accumulates (same component
/// box, frequencies, time window and stride) as shared: see [`DftPlan::shared`].
fn share_dft_boxes(plan: &mut Plan) {
    use std::collections::HashMap;
    // key: (comp, glo, ghi, group of identical accumulation settings)
    let mut groups: Vec<(Vec<u64>, Apodization, usize)> = vec![];
    let mut seen: HashMap<(Component, [usize; 3], [usize; 3], usize), (usize, usize)> = HashMap::new();
    for di in 0..plan.dft.len() {
        let key = (plan.dft[di].freqs.iter().map(|f| f.to_bits()).collect::<Vec<u64>>(), plan.dft[di].apodization.clone(), plan.dft[di].stride);
        let gi = match groups.iter().position(|g| *g == key) {
            Some(g) => g,
            None => {
                groups.push(key);
                groups.len() - 1
            }
        };
        let mut all_shared = !plan.dft[di].boxes.is_empty();
        for bi in 0..plan.dft[di].boxes.len() {
            let b = &plan.dft[di].boxes[bi];
            match seen.get(&(b.comp, b.glo, b.ghi, gi)) {
                Some(&src) => {
                    plan.dft[di].shared[bi] = Some(src);
                    plan.dft[di].boxes[bi].lsize[0] = 0; // not local: nothing to allocate or accumulate
                }
                None => {
                    seen.insert((b.comp, b.glo, b.ghi, gi), (di, bi));
                    all_shared = false;
                }
            }
        }
        if all_shared {
            // no box of its own: the phasor tables are not needed either
            plan.dft[di].phasor_e = vec![];
            plan.dft[di].phasor_h = vec![];
        }
    }
}

fn compile_monitor(
    plan: &mut Plan,
    mon: &Monitor,
    sim_boundaries: &crate::spec::Boundaries,
    eps_local: &[Vec<Real>; 3],
    mu_local: Option<&[Vec<Real>; 3]>,
    stride_of: &dyn Fn(usize, &[f64]) -> usize,
    inv: Option<usize>,
) -> Result<()> {
    let grid = plan.grid.clone();
    let (x0, nxl) = (plan.x0, plan.nxl);
    let nsteps = plan.nsteps;
    let dt = plan.dt;
    match mon {
        Monitor::FieldTime { name, min, max, components, interval, start } => {
            let boxes = components.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            plan.time.push(TimePlan {
                name: name.clone(),
                boxes,
                interval: (*interval).max(1),
                start_step: (start / dt).floor().max(0.0) as usize,
                is_probe: false,
            });
        }
        Monitor::Probe { name, position, components } => {
            let boxes = components
                .iter()
                .map(|c| {
                    let g = [
                        grid.nearest_index(*c, 0, position[0]),
                        grid.nearest_index(*c, 1, position[1]),
                        grid.nearest_index(*c, 2, position[2]),
                    ];
                    localize(*c, g, [g[0] + 1, g[1] + 1, g[2] + 1], x0, nxl)
                })
                .collect();
            plan.time.push(TimePlan { name: name.clone(), boxes, interval: 1, start_step: 0, is_probe: true });
        }
        Monitor::DftField { name, min, max, components, freqs, apodization, stride } => {
            let boxes: Vec<CompBox> = components.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            let stride = stride_of(*stride, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Field,
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::Flux { name, axis, position, min, max, freqs, apodization } => {
            let p = grid.nearest_node(*axis, *position).max(1);
            let boxes = flux_boxes(&grid, *axis, p, min, max, x0, nxl);
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Flux { axis: *axis, p },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::FluxBox { name, min, max, freqs, apodization } => {
            let (smin, smax) = snapped_box(&grid, min, max);
            let mut boxes = vec![];
            let mut faces = vec![];
            for a in 0..3 {
                // 2-D: the faces normal to the invariant axis carry equal and
                // opposite fluxes (nothing varies along it)
                if inv == Some(a) {
                    continue;
                }
                for (side, coord) in [(-1.0, smin[a]), (1.0, smax[a])] {
                    let pn = grid.nearest_node(a, coord);
                    // a face on a PEC/PMC wall carries no power (symmetry planes)
                    let wall = |b: &Boundary| matches!(b, Boundary::Pec | Boundary::Pmc);
                    if (pn == 0 && wall(&sim_boundaries.axis(a)[0])) || (pn == grid.n()[a] && wall(&sim_boundaries.axis(a)[1])) {
                        continue;
                    }
                    let p = pn.max(1);
                    faces.push((a, side, boxes.len()));
                    boxes.extend(flux_boxes(&grid, a, p, &smin, &smax, x0, nxl));
                }
            }
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::FluxBox,
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces,
                region: (smin, smax),
            });
        }
        Monitor::Mode { name, axis, position, min, max, freqs, modes, apodization, per_frequency } => {
            let per_frequency = crate::monitors::profiles_per_frequency(name, modes, freqs, *per_frequency)?;
            let p = grid.nearest_node(*axis, *position).max(1);
            let boxes = flux_boxes(&grid, *axis, p, min, max, x0, nxl);
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            let (origin, _) = transverse_box(&grid, *axis, Some(min), Some(max));
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::Mode { axis: *axis, p, origin, modes: modes.clone(), per_frequency },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces: vec![],
                region: (*min, *max),
            });
        }
        Monitor::FarField { name, min, max, freqs, theta, phi, n_medium, apodization } => {
            let (smin, smax) = snapped_box(&grid, min, max);
            let mut boxes = vec![];
            let mut faces = vec![];
            for a in 0..3 {
                for (side, coord) in [(-1.0, smin[a]), (1.0, smax[a])] {
                    let p = grid.nearest_node(a, coord).max(1);
                    faces.push((a, side, boxes.len()));
                    boxes.extend(flux_boxes(&grid, a, p, &smin, &smax, x0, nxl));
                }
            }
            let stride = stride_of(0, freqs);
            let (pe, ph) = phasor_tables(freqs, nsteps, dt, apodization, stride);
            plan.dft.push(DftPlan {
                name: name.clone(),
                id: 0, // set once all monitors are compiled
                kind: DftKind::FarField { theta: theta.clone(), phi: phi.clone(), n_medium: *n_medium },
                freqs: freqs.clone(),
                shared: vec![None; boxes.len()],
                boxes,
                apodization: apodization.clone(),
                stride,
                phasor_e: pe,
                phasor_h: ph,
                faces,
                region: (smin, smax),
            });
        }
        Monitor::Energy { name, min, max, interval } => {
            let boxes: Vec<CompBox> = Component::ALL.iter().map(|c| comp_box(&grid, *c, min, max, x0, nxl)).collect();
            // weight of every local sample: eps0/2 * (eps | mu) * the part of
            // the cell's volume that lies inside the monitor box
            let (ny, nz) = (plan.n[1], plan.n[2]);
            let n = plan.n;
            let periodic = plan.periodic;
            let width = |c: Component, ax: usize, g: usize| -> f64 {
                let (lo, hi) = grid.voxel_bounds(c, ax, g);
                let mut w = (hi.min(max[ax]) - lo.max(min[ax])).max(0.0);
                if periodic[ax] && !c.is_half(ax) && g == 0 {
                    // the dual cell of the first node wraps around a periodic axis
                    let e = grid.edges(ax);
                    let (lo2, hi2) = (0.5 * (e[n[ax] - 1] + e[n[ax]]), e[n[ax]]);
                    w += (hi2.min(max[ax]) - lo2.max(min[ax])).max(0.0);
                }
                w
            };
            let weights: Vec<Vec<Real>> = boxes
                .iter()
                .map(|bx| {
                    let c = bx.comp;
                    let a = c.axis();
                    let mut w = Vec::with_capacity(bx.lcells());
                    for ii in 0..bx.lsize[0] {
                        let li = bx.lstart[0] + ii;
                        let wx = width(c, 0, x0 + li);
                        for jj in 0..bx.lsize[1] {
                            let j = bx.lstart[1] + jj;
                            let wy = width(c, 1, j);
                            for kk in 0..bx.lsize[2] {
                                let k = bx.lstart[2] + kk;
                                let o = (li * ny + j) * nz + k;
                                let m = if c.is_e() { eps_local[a][o] as f64 } else { mu_local.map(|m| m[a][o] as f64).unwrap_or(1.0) };
                                w.push((0.5 * EPS0 * m * wx * wy * width(c, 2, k)) as Real);
                            }
                        }
                    }
                    w
                })
                .collect();
            plan.energy.push(EnergyPlan { name: name.clone(), time: plan.time.len(), weights });
            plan.time.push(TimePlan {
                name: format!("__energy__{name}"),
                boxes,
                interval: (*interval).max(1),
                start_step: 0,
                is_probe: false,
            });
        }
    }
    Ok(())
}
