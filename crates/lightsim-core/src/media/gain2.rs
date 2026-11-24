//! Four-level two-electron gain medium (Chang and Taflove, "Finite-difference
//! time-domain model of lasing action in a four-level two-electron atomic
//! system", Opt. Express 12, 3827 (2004)), the formulation of Lumerical's
//! "4-level 2-electron" material.
//!
//! Two polarizations, the lasing transition 2 -> 1 at `omega_a` and the
//! pump transition 3 -> 0 at `omega_b`, each a driven damped oscillator whose
//! strength follows its population difference, and four populations with
//! the Pauli-exclusion factors (two electrons per atom, at most one per
//! level, `density` atoms per volume):
//!
//! ```text
//! Pa'' + gamma_a Pa' + omega_a^2 Pa = kappa_a (N1 - N2) E
//! Pb'' + gamma_b Pb' + omega_b^2 Pb = kappa_b (N0 - N3) E
//! dN3/dt =  E.Pb' / (hbar omega_b) - N3 (1 - N2/Nd) / t32 - N3 (1 - N0/Nd) / t30
//! dN2/dt =  N3 (1 - N2/Nd) / t32 + E.Pa' / (hbar omega_a) - N2 (1 - N1/Nd) / t21
//! dN1/dt =  N2 (1 - N1/Nd) / t21 - E.Pa' / (hbar omega_a) - N1 (1 - N0/Nd) / t10
//! dN0/dt =  N3 (1 - N0/Nd) / t30 + N1 (1 - N0/Nd) / t10 - E.Pb' / (hbar omega_b)
//! ```
//!
//! N in m^-3 (they sum to 2 Nd), P in C/m^2, kappa = 6 pi eps0 c^3 /
//! (omega^2 t) (t = t21 for a, t30 for b) unless given. The medium is pumped
//! optically by a field at omega_b (a source in the simulation), not by a
//! rate: this is what distinguishes it from [`crate::materials::FourLevel`],
//! whose pump is a rate W_p and which has one electron per atom and no
//! pump polarization.
//!
//! Discretisation (as `FourLevel`): before the E update, both polarizations
//! advance with the populations at n (explicit ADE of a Lorentz pole);
//! after it, E^{n+1} -= (dPa + dPb) / (eps (1 + f)) and the populations
//! move by explicit fluxes from the populations at n, the field work
//! eps0 (E^n + E^{n+1}) / 2 dP / (hbar omega) moving electrons 1 -> 2 (a) and
//! 0 -> 3 (b). Every flux takes from one level what it gives another, so
//! the electron number is conserved to rounding.

use crate::Real;
use crate::constants::{C0, EPS0, HBAR, PI};
use crate::grid::Component;
use crate::materials::{Medium, Pole};
use anyhow::{Result, bail};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// The four-level two-electron medium (see the module documentation);
/// field names follow Lumerical's material (omega a / b, gamma a / b, t30,
/// t32, t21, t10, N density, initial populations).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TwoElectron {
    /// Atom density Nd (m^-3); the electrons number 2 Nd.
    pub density: f64,
    /// Angular frequency of the lasing transition 2 -> 1 (rad/s).
    pub omega_a: f64,
    /// Angular frequency of the pump transition 3 -> 0 (rad/s).
    pub omega_b: f64,
    /// Widths of the two transitions (rad/s, the gamma of the oscillator
    /// equations: full width at half maximum of the Lorentzian).
    pub gamma_a: f64,
    pub gamma_b: f64,
    /// Lifetimes (s).
    pub t30: f64,
    pub t32: f64,
    pub t21: f64,
    pub t10: f64,
    /// Initial population densities N0..N3 (m^-3); default [Nd, Nd, 0, 0]
    /// (the two electrons in levels 0 and 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<[f64; 4]>,
    /// Coupling constants kappa_a, kappa_b (C^2/kg); default
    /// 6 pi eps0 c^3 / (omega^2 t) with t21 and t30.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kappa_a: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kappa_b: Option<f64>,
}

impl TwoElectron {
    pub fn kappa_a(&self) -> f64 {
        self.kappa_a.unwrap_or(6.0 * PI * EPS0 * C0.powi(3) / (self.omega_a * self.omega_a * self.t21))
    }
    pub fn kappa_b(&self) -> f64 {
        self.kappa_b.unwrap_or(6.0 * PI * EPS0 * C0.powi(3) / (self.omega_b * self.omega_b * self.t30))
    }
    pub fn initial(&self) -> [f64; 4] {
        self.initial.unwrap_or([self.density, self.density, 0.0, 0.0])
    }
    /// The transition `a` (lasing) or `b` (pump) as a Lorentz pole per unit
    /// population difference (P / eps0 per (N_lower - N_upper)).
    pub fn pole(&self, b: bool) -> Pole {
        let (w, g, k) = if b { (self.omega_b, self.gamma_b, self.kappa_b()) } else { (self.omega_a, self.gamma_a, self.kappa_a()) };
        Pole::Lorentz { delta_eps: k / EPS0 / (w * w), omega0: w, gamma: g }
    }
    /// Susceptibility of the frozen populations `n` (continuous model):
    /// chi_a (N1 - N2) + chi_b (N0 - N3).
    pub fn chi(&self, n: [f64; 4], omega: f64) -> Complex64 {
        self.pole(false).chi(omega) * (n[1] - n[2]) + self.pole(true).chi(omega) * (n[0] - n[3])
    }
    /// The same for the discrete update.
    pub fn chi_discrete(&self, n: [f64; 4], omega: f64, dt: f64) -> Complex64 {
        self.pole(false).chi_discrete(omega, dt) * (n[1] - n[2]) + self.pole(true).chi_discrete(omega, dt) * (n[0] - n[3])
    }
    /// Right-hand side of the rate equations with the field work rates
    /// `wa` = E.Pa' and `wb` = E.Pb' (W/m^3).
    pub fn rates(&self, n: [f64; 4], wa: f64, wb: f64) -> [f64; 4] {
        let nd = self.density;
        let ua = wa / (HBAR * self.omega_a);
        let ub = wb / (HBAR * self.omega_b);
        let f32_ = n[3] * (1.0 - n[2] / nd) / self.t32;
        let f30 = n[3] * (1.0 - n[0] / nd) / self.t30;
        let f21 = n[2] * (1.0 - n[1] / nd) / self.t21;
        let f10 = n[1] * (1.0 - n[0] / nd) / self.t10;
        [f30 + f10 - ub, f21 - ua - f10, f32_ + ua - f21, ub - f32_ - f30]
    }
    pub fn check(&self, medium: &str, dt: f64) -> Result<()> {
        let n = self.initial();
        let ok = self.density > 0.0
            && self.omega_a > 0.0
            && self.omega_b > 0.0
            && self.gamma_a >= 0.0
            && self.gamma_b >= 0.0
            && [self.t30, self.t32, self.t21, self.t10].iter().all(|&t| t > 0.0)
            && self.kappa_a.map(|c| c > 0.0).unwrap_or(true)
            && self.kappa_b.map(|c| c > 0.0).unwrap_or(true)
            && n.iter().all(|&v| v >= 0.0 && v <= self.density * (1.0 + 1e-12))
            && ((n.iter().sum::<f64>() / (2.0 * self.density)) - 1.0).abs() < 1e-9;
        if !ok {
            bail!(
                "medium '{medium}': the four-level two-electron medium needs density, omega_a, omega_b, the lifetimes (and kappa_a, kappa_b if given) > 0, gamma_a, gamma_b >= 0, and initial populations between 0 and density that sum to 2 density (two electrons per atom)"
            );
        }
        let fastest = [1.0 / self.t30, 1.0 / self.t32, 1.0 / self.t21, 1.0 / self.t10].into_iter().fold(0.0, f64::max);
        if fastest * dt > 0.5 {
            bail!("medium '{medium}': the rate equations need dt / min(t30, t32, t21, t10) <= 0.5 (it is {:.3}); lengthen the lifetimes or refine the mesh", fastest * dt);
        }
        Ok(())
    }
}

/// Width of a row of [`Gain2Plan::tab`]: ca0, ca1, ca2, cb0, cb1, cb2,
/// eps0 / (hbar omega_a), eps0 / (hbar omega_b), dt / t30, dt / t32,
/// dt / t21, dt / t10, 1 / Nd.
pub const GAIN2_TAB: usize = 13;

/// Cells of one E component inside four-level two-electron media.
#[derive(Clone, Debug)]
pub struct Gain2Component {
    pub comp: Component,
    pub idx: Vec<u64>,
    pub row: Vec<u32>,
    /// 1 / (eps (1 + f)).
    pub scale: Vec<Real>,
    /// Volume of the E sample (m^3).
    pub vol: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Gain2Plan {
    pub tab: Vec<f64>,
    /// Per row: initial N0..N3 (m^-3).
    pub init: Vec<f64>,
    pub comps: Vec<Gain2Component>,
}

/// Compile the four-level two-electron media of a slab.
#[allow(clippy::too_many_arguments)]
pub fn compile_gain2(
    media: &[Medium],
    med_local: &[Vec<u16>; 3],
    cb_local: &[Vec<Real>; 3],
    inv_p_full: &[Vec<Real>; 3],
    inv_d_full: &[Vec<Real>; 3],
    x0: usize,
    nxl: usize,
    n: [usize; 3],
    dt: f64,
    complex_fields: bool,
) -> Result<Option<Gain2Plan>> {
    let gm: Vec<usize> = (0..media.len()).filter(|&m| media[m].gain2.is_some()).collect();
    if gm.is_empty() {
        return Ok(None);
    }
    if complex_fields {
        bail!("four-level two-electron gain media cannot be combined with Bloch boundaries (complex fields)");
    }
    if media.iter().any(|m| m.gain.is_some()) {
        bail!("a simulation cannot hold both four-level (rate-equation) and four-level two-electron gain media (their population totals share Results.gain_levels)");
    }
    let mut tab = vec![];
    let mut init = vec![];
    for &m in &gm {
        let md = &media[m];
        let g = md.gain2.as_ref().unwrap();
        g.check(&md.name, dt)?;
        if md.is_dispersive() || md.is_nonlinear() || md.raman.is_some() || md.gain.is_some() || md.is_tensor() || md.pec || md.plugin.is_some() {
            bail!("medium '{}': a four-level two-electron medium cannot also have poles, chi2/chi3, another gain model, off-diagonal terms or a plug-in, or be a perfect conductor", md.name);
        }
        let ca = g.pole(false).ade_coefficients(dt);
        let cb = g.pole(true).ade_coefficients(dt);
        tab.extend([ca[0], ca[1], ca[2], cb[0], cb[1], cb[2], EPS0 / (HBAR * g.omega_a), EPS0 / (HBAR * g.omega_b), dt / g.t30, dt / g.t32, dt / g.t21, dt / g.t10, 1.0 / g.density]);
        init.extend(g.initial());
    }
    let (ny, nz) = (n[1], n[2]);
    let plane = ny * nz;
    let mut comps = vec![];
    for a in 0..3 {
        let mut c = Gain2Component { comp: Component::e(a), idx: vec![], row: vec![], scale: vec![], vol: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = (i * ny + j) * nz + k;
                    let m = med_local[a][o] as usize;
                    if media[m].gain2.is_none() || cb_local[a][o] == 0.0 {
                        continue;
                    }
                    let f = media[m].sigma[a].max(0.0) * dt / (2.0 * EPS0 * media[m].eps[a]);
                    c.idx.push(((i + 1) * plane + j * nz + k) as u64);
                    c.row.push(gm.iter().position(|&x| x == m).unwrap() as u32);
                    c.scale.push((1.0 / (media[m].eps[a] * (1.0 + f))) as Real);
                    let g = [x0 + i, j, k];
                    let vol: f64 = (0..3).map(|q| 1.0 / if q == a { inv_p_full[q][g[q]] } else { inv_d_full[q][g[q]] } as f64).product();
                    c.vol.push(vol);
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    Ok(if comps.is_empty() { None } else { Some(Gain2Plan { tab, init, comps }) })
}

/// Step 1 of a cell (before the E update; the `gain2_pre` kernel): `st` =
/// (E^n, Pa, Pa_prev, Pb, Pb_prev, dPa, dPb) as columns `[k][n]`, `pop` =
/// `[level][n]`.
#[inline]
pub fn pre_cell(e: Real, k: &[f64], pop: &[f64], st: &mut [Vec<Real>; 7], t: usize, n: usize) {
    let da = (k[2] * (pop[n + t] - pop[2 * n + t])) as Real;
    let db = (k[5] * (pop[t] - pop[3 * n + t])) as Real;
    st[0][t] = e;
    let (pa, pb) = (st[1][t], st[3][t]);
    let pan = k[0] as Real * pa + k[1] as Real * st[2][t] + da * e;
    let pbn = k[3] as Real * pb + k[4] as Real * st[4][t] + db * e;
    st[2][t] = pa;
    st[1][t] = pan;
    st[4][t] = pb;
    st[3][t] = pbn;
    st[5][t] = pan - pa;
    st[6][t] = pbn - pb;
}

/// Population update of a cell after the E update (the `gain2_post`
/// kernel does the same arithmetic): `work_a`, `work_b` = (E^n + E^{n+1})/2
/// dP with P / eps0.
#[inline]
pub fn rates_cell(pop: &mut [f64], n: usize, t: usize, k: &[f64], work_a: f64, work_b: f64) {
    let (n0, n1, n2, n3) = (pop[t], pop[n + t], pop[2 * n + t], pop[3 * n + t]);
    let inv = k[12];
    let ua = k[6] * work_a;
    let ub = k[7] * work_b;
    let f30 = k[8] * n3 * (1.0 - n0 * inv);
    let f32_ = k[9] * n3 * (1.0 - n2 * inv);
    let f21 = k[10] * n2 * (1.0 - n1 * inv);
    let f10 = k[11] * n1 * (1.0 - n0 * inv);
    pop[t] = n0 + f30 + f10 - ub;
    pop[n + t] = n1 + f21 - ua - f10;
    pop[2 * n + t] = n2 + f32_ + ua - f21;
    pop[3 * n + t] = n3 + ub - f32_ - f30;
}
