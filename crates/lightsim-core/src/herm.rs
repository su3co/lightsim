//! Complex (Hermitian) permittivity tensors given directly.
//!
//! A medium with `eps_complex` has the non-dispersive relative permittivity
//! tensor eps (complex 3 x 3, Hermitian and positive definite, e.g.
//! eps_yz = -eps_zy^* = i g for a magneto-optic medium biased along x). Such a
//! tensor has no real time-domain counterpart: a real field would see eps
//! at positive and eps^* at negative frequencies. The run therefore uses
//! complex fields (the Bloch machinery: a real and an imaginary field set)
//! and applies eps to the complex field. The engine is linear and time
//! invariant, so each frequency evolves on its own; the frequency-domain
//! monitors (positive frequencies) see eps, whatever the time signal.
//! The model is exact only as a non-dispersive one: a physical
//! magneto-optic medium is dispersive, and eps is then its value at one
//! frequency (see the gyrotropic poles of [`crate::materials::GyroPole`]
//! for the dispersive model).
//!
//! With kappa = eps^-1 (Hermitian: kappa = kR + i kI, kR symmetric, kI
//! antisymmetric, real diagonal) the D-form update is that of the real
//! off-diagonal tensors ([`crate::plan::TensorPlan`]) applied to the complex
//! field: the linear update of each set applies kappa_aa (its `cb` is
//! kappa_aa c dt), dD = (E_lin - E^n) c dt / cb is taken per set, and then
//!
//! ```text
//! E_a,re += sum_b ( kR_ab <dD_b,re> - kI_ab <dD_b,im> )
//! E_a,im += sum_b ( kR_ab <dD_b,im> + kI_ab <dD_b,re> )
//! ```
//!
//! over the two other components b, with <.> the four-point average of
//! E_b positions around the E_a position (the scheme of the real tensors).
//! For a real symmetric tensor this is the real tensor update on each set.

use crate::Real;
use crate::grid::Component;
use crate::materials::Medium;
use anyhow::{Result, bail};
use num_complex::Complex64;

/// The complex tensor of a medium's `eps_complex`.
pub fn tensor(m: &Medium) -> Option<[[Complex64; 3]; 3]> {
    m.eps_complex.map(|t| t.map(|r| r.map(|c| Complex64::new(c[0], c[1]))))
}

/// Check that `e` is Hermitian (to 1e-12 of its largest entry) and
/// positive definite (Sylvester's criterion), and return kappa = e^-1.
pub fn check_and_invert(e: &[[Complex64; 3]; 3], medium: &str) -> Result<[[Complex64; 3]; 3]> {
    let scale = e.iter().flatten().fold(0.0f64, |m, v| m.max(v.norm()));
    for i in 0..3 {
        for j in 0..3 {
            if (e[i][j] - e[j][i].conj()).norm() > 1e-12 * scale {
                bail!("medium '{medium}': eps_complex must be Hermitian (eps_ij = conj(eps_ji)); entry ({i}, {j}) is {} and ({j}, {i}) is {}", e[i][j], e[j][i]);
            }
        }
    }
    let d1 = e[0][0].re;
    let d2 = (e[0][0] * e[1][1] - e[0][1] * e[1][0]).re;
    let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    if !(d1 > 0.0 && d2 > 0.0 && det.re > 0.0) {
        bail!("medium '{medium}': eps_complex must be positive definite (leading minors {d1:.4e}, {d2:.4e}, {:.4e}); a tensor that is not would make the update unstable", det.re);
    }
    let c = |a: usize, b: usize, cc: usize, d: usize| e[a][b] * e[cc][d] - e[a][d] * e[cc][b];
    let adj = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    Ok(adj.map(|r| r.map(|v| v / det)))
}

/// Cells of one E component inside complex-tensor media.
#[derive(Clone, Debug)]
pub struct HermComponent {
    pub comp: Component,
    pub idx: Vec<u64>,
    /// kR_{a,a+1}, kR_{a,a+2}, kI_{a,a+1}, kI_{a,a+2} per cell.
    pub kr1: Vec<Real>,
    pub kr2: Vec<Real>,
    pub ki1: Vec<Real>,
    pub ki2: Vec<Real>,
}

/// Complex-tensor media of one slab (see the module documentation).
#[derive(Clone, Debug)]
pub struct HermPlan {
    /// Storage planes [p0, p1) of the dD buffers.
    pub p0: usize,
    pub p1: usize,
    /// Periodic wrap bits (x, y, z) of the neighbour averages.
    pub wrap: u32,
    pub cdt: Real,
    pub comps: Vec<HermComponent>,
}

/// Compile the complex-tensor media of a slab and set `cb` of their cells
/// to kappa_aa c dt. `None` without such media.
#[allow(clippy::too_many_arguments)]
pub fn compile_herm(
    media: &[Medium],
    med_local: &[Vec<u16>; 3],
    cb_local: &mut [Vec<Real>; 3],
    nxl: usize,
    n: [usize; 3],
    cdt: Real,
    periodic: [bool; 3],
    nslabs: usize,
    invariant: Option<usize>,
) -> Result<Option<HermPlan>> {
    let hm: Vec<usize> = (0..media.len()).filter(|&m| media[m].eps_complex.is_some()).collect();
    if hm.is_empty() {
        return Ok(None);
    }
    if invariant.is_some() {
        bail!("complex permittivity tensors (eps_complex) are not available in 2-D simulations");
    }
    if let Some(m) = media.iter().find(|m| m.is_nonlinear() || m.raman.is_some() || m.gain.is_some() || m.gain2.is_some() || m.plugin.is_some() || m.is_tensor()) {
        bail!("medium '{}' cannot share a simulation with a complex permittivity tensor (eps_complex): that run uses complex fields, which nonlinear, gain, plug-in and real off-diagonal media do not support (give a real tensor as eps_complex with zero imaginary parts instead of eps_off)", m.name);
    }
    let mut kap = vec![];
    for &m in &hm {
        let md = &media[m];
        if md.is_dispersive() || md.is_lossy() || md.pec || md.is_magnetic() {
            bail!("medium '{}': a complex permittivity tensor is a non-dispersive, lossless, non-magnetic medium; it cannot also have poles, sigma, mu or be a perfect conductor", md.name);
        }
        let k = check_and_invert(&tensor(md).unwrap(), &md.name)?;
        if nslabs > 1 && [k[0][1], k[0][2]].iter().any(|v| v.norm() > 1e-12 * k[0][0].norm()) {
            bail!("medium '{}': on several slabs (several GPUs or nodes) a complex permittivity tensor may couple only y and z (its x-y and x-z terms need the neighbouring slab's fields, which are not exchanged); run it on one device", md.name);
        }
        kap.push(k);
    }
    let (ny, nz) = (n[1], n[2]);
    let plane = ny * nz;
    let mut comps = vec![];
    for a in 0..3 {
        let (b1, b2) = ((a + 1) % 3, (a + 2) % 3);
        let mut c = HermComponent { comp: Component::e(a), idx: vec![], kr1: vec![], kr2: vec![], ki1: vec![], ki2: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = (i * ny + j) * nz + k;
                    let Some(r) = hm.iter().position(|&m| m == med_local[a][o] as usize) else { continue };
                    if cb_local[a][o] == 0.0 {
                        continue;
                    }
                    let kk = &kap[r];
                    cb_local[a][o] = (kk[a][a].re * cdt as f64) as Real;
                    c.idx.push(((i + 1) * plane + j * nz + k) as u64);
                    c.kr1.push(kk[a][b1].re as Real);
                    c.kr2.push(kk[a][b2].re as Real);
                    c.ki1.push(kk[a][b1].im as Real);
                    c.ki2.push(kk[a][b2].im as Real);
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    if comps.is_empty() {
        return Ok(None);
    }
    let wrap_x = periodic[0] && nslabs <= 1;
    let wrap = wrap_x as u32 | (periodic[1] as u32) << 1 | (periodic[2] as u32) << 2;
    // storage planes the cells and their x neighbours occupy
    let (p0, p1) = if wrap_x {
        (1, nxl + 1)
    } else {
        let (mut lo, mut hi) = (usize::MAX, 0);
        for c in &comps {
            for &s in &c.idx {
                let pi = s as usize / plane;
                lo = lo.min(pi);
                hi = hi.max(pi);
            }
        }
        (lo.saturating_sub(1).max(1), (hi + 2).min(nxl + 1))
    };
    Ok(Some(HermPlan { p0, p1, wrap, cdt, comps }))
}

/// Four-point averages of `dd1` (component a+1) and `dd2` (a+2) around the
/// E_a cell at storage index `s` (the neighbour pattern of the real tensor
/// update), over storage planes [p0, p1).
#[inline]
pub fn averages(a: usize, s: usize, ny: usize, nz: usize, p0: usize, p1: usize, wrap: u32, dd1: &[Real], dd2: &[Real]) -> (Real, Real) {
    let plane = ny * nz;
    let (pi, rem) = ((s / plane) as i64, (s % plane) as i64);
    let (j, k) = (rem / nz as i64, rem % nz as i64);
    let (lo, hi, nyi, nzi) = (p0 as i64, p1 as i64, ny as i64, nz as i64);
    let mut acc = [0 as Real; 2];
    for w in 0..8 {
        let second = w >= 4;
        let b = if second { (a + 2) % 3 } else { (a + 1) % 3 };
        let (up, dn) = (w & 1 != 0, w & 2 != 0);
        let d = |ax: usize| (up && a == ax) as i64 - (dn && b == ax) as i64;
        let (mut x, mut y, mut z) = (pi + d(0), j + d(1), k + d(2));
        let mut ok = true;
        if x < lo || x >= hi {
            if wrap & 1 != 0 { x += if x < lo { hi - lo } else { lo - hi } } else { ok = false }
        }
        if y < 0 || y >= nyi {
            if wrap & 2 != 0 { y += if y < 0 { nyi } else { -nyi } } else { ok = false }
        }
        if z < 0 || z >= nzi {
            if wrap & 4 != 0 { z += if z < 0 { nzi } else { -nzi } } else { ok = false }
        }
        if ok {
            let o = ((x - lo) * nyi * nzi + y * nzi + z) as usize;
            acc[second as usize] += if second { dd2[o] } else { dd1[o] };
        }
    }
    (0.25 * acc[0], 0.25 * acc[1])
}
