//! Per-cell update coefficients and the sub-pixel smoothing of dispersive
//! voxels (helpers of the materials phase, also used by the gradient engines).

use super::*;

/// A medium whose electric update is the plain one, possibly with
/// conductivity and per-axis poles: no tensor, nonlinear, gain, plug-in or
/// complex terms.
pub fn is_plain_medium(m: &Medium) -> bool {
    !m.is_tensor() && !m.is_nonlinear() && m.oriented.is_empty() && m.gyro.is_empty() && m.raman.is_none() && m.director.is_none()
        && m.gain.is_none() && m.plugin.is_none() && m.gain2.is_none() && m.eps_complex.is_none()
}

/// E-update coefficients `(ca, cb)` of a voxel of E component `a` whose
/// dominant medium is `md` and whose effective relative permittivity is
/// `eps`: `E <- ca E + cb curl H~`. Shared by the plan compiler and by
/// `crate::autodiff` (design updates re-evaluate exactly this).
pub fn e_update_coefficients(md: &Medium, a: usize, eps: Real, dt: f64) -> (Real, Real) {
    let cdt = (C0 * dt) as Real;
    if md.pec {
        (1.0, 0.0)
    } else if md.sigma[a] > 0.0 {
        let f = md.sigma[a] * dt / (2.0 * EPS0 * eps as f64);
        (((1.0 - f) / (1.0 + f)) as Real, (C0 * dt / eps as f64 / (1.0 + f)) as Real)
    } else {
        (1.0, cdt / eps)
    }
}

/// `(da, ch)` of the H update `H~' = da H~ - ch curl E` for medium `md`
/// along `a` at relative permeability `mu` (`H~ = eta0 H`): with magnetic
/// conductivity, `f = sigma_m dt / (2 mu0 mu)`, `da = (1 - f) / (1 + f)`,
/// `ch = c dt / mu / (1 + f)`; otherwise `(1, c dt / mu)`.
pub fn h_update_coefficients(md: &Medium, a: usize, mu: Real, dt: f64) -> (Real, Real) {
    if md.sigma_m[a] > 0.0 {
        let f = md.sigma_m[a] * dt / (2.0 * crate::constants::MU0 * mu as f64);
        (((1.0 - f) / (1.0 + f)) as Real, (C0 * dt / mu as f64 / (1.0 + f)) as Real)
    } else {
        (1.0, (C0 * dt) as Real / mu)
    }
}

pub(super) fn storage_from_local(local: &[Real], nxl: usize, plane: usize, fill: Real) -> Vec<Real> {
    let mut v = vec![fill; (nxl + 2) * plane];
    v[plane..(nxl + 1) * plane].copy_from_slice(local);
    v
}

/// Reference angular frequency of the sources for the sub-pixel smoothing
/// of transparent dispersive dielectrics: the mean of their centre (peak)
/// frequencies, or `None` without sources or with a sampled signal.
pub fn source_center_omega(sources: &[Source]) -> Option<f64> {
    if sources.is_empty() {
        return None;
    }
    let mut acc = 0.0;
    for s in sources {
        acc += match s.signal() {
            TimeSignal::GaussianPulse { freq0, .. } | TimeSignal::ContinuousWave { freq0, .. } | TimeSignal::Ricker { freq0, .. } => *freq0,
            TimeSignal::Custom { .. } => return None,
        };
    }
    let f = acc / sources.len() as f64;
    (f > 0.0).then_some(TAU * f)
}

/// Apply the mixed voxels of a rasterised E component (those that pass
/// [`mixed_cell_ok`]): their `value` becomes eps_inf of the voxel and their
/// `medium` the dispersive medium A. Returns the pole weight per voxel (NaN
/// where not mixed; empty without mixed voxels) and the counts (applied,
/// sent back to the staircase).
pub fn apply_mixed_cells(
    cm: &mut crate::raster::ComponentMaterial,
    media: &[Medium],
    a: usize,
    dt: f64,
    dt_courant: f64,
    memo: &mut std::collections::HashMap<(u16, i64, i64), bool>,
) -> (Vec<Real>, [usize; 2]) {
    let mut counts = [0usize; 2];
    if cm.mixed.is_empty() {
        return (vec![], counts);
    }
    let mut wv = vec![Real::NAN; cm.value.len()];
    for mc in &cm.mixed {
        if mixed_cell_ok(&media[mc.medium as usize], a, mc, dt, dt_courant, memo) {
            cm.value[mc.o] = mc.eps_inf as Real;
            cm.medium[mc.o] = mc.medium;
            wv[mc.o] = mc.weight as Real;
            counts[0] += 1;
        } else {
            counts[1] += 1;
        }
    }
    if std::env::var("LIGHTSIM_DEBUG_SUBPIXEL").is_ok() {
        eprintln!("subpixel: E{} {} mixed dispersive voxels, {} sent back to the staircase", ["x", "y", "z"][a], counts[0], counts[1]);
    }
    (wv, counts)
}

/// Courant step of a grid (the 2-D one for a 2-D simulation).
pub fn courant_step(grid: &Grid, invariant: Option<usize>) -> f64 {
    let d = grid.min_cell_size();
    1.0 / (C0 * d.iter().enumerate().filter(|(q, _)| Some(*q) != invariant).map(|(_, v)| 1.0 / (v * v)).sum::<f64>().sqrt())
}

/// Guards of a mixed voxel ([`crate::raster::MixedCell`]) of medium `md`
/// along `a`: weight in [0, 1.05], eps_inf of the voxel >= 1, and the poles
/// with their weight stable on that eps_inf at the run's time step (the
/// check of [`crate::materials::poles_stable`], memoized on the medium and
/// the rounded weight and permittivity).
pub fn mixed_cell_ok(md: &Medium, a: usize, mc: &crate::raster::MixedCell, dt: f64, dt_courant: f64, memo: &mut std::collections::HashMap<(u16, i64, i64), bool>) -> bool {
    use crate::materials::Pole;
    if !(mc.weight >= 0.0 && mc.weight <= 1.05 && mc.eps_inf >= 1.0 && mc.eps_inf.is_finite()) {
        return false;
    }
    // (rounded up in weight, down in permittivity: the memoized case is never more stable than the voxel)
    let key = (mc.medium * 3 + a as u16, (mc.weight * 1e4).ceil() as i64, (mc.eps_inf * 1e4).floor() as i64);
    *memo.entry(key).or_insert_with(|| {
        let (w, e) = (key.1 as f64 * 1e-4, key.2 as f64 * 1e-4);
        let poles: Vec<Pole> = md
            .poles
            .iter()
            .map(|p| match *p {
                Pole::Lorentz { delta_eps, omega0, gamma } => Pole::Lorentz { delta_eps: w * delta_eps, omega0, gamma },
                Pole::Debye { delta_eps, tau } => Pole::Debye { delta_eps: w * delta_eps, tau },
                Pole::Drude { omega_p, gamma } => Pole::Drude { omega_p: w.sqrt() * omega_p, gamma },
            })
            .collect();
        let mu = md.mu.iter().cloned().fold(f64::INFINITY, f64::min);
        let a_max = 4.0 * (dt / dt_courant).powi(2) / (e * mu);
        e >= 1.0 && crate::materials::poles_stable(e, 0.0, &poles, dt, a_max)
    })
}
