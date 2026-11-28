//! Simulation results: assembly of raw backend data into global arrays and
//! post-processing (flux spectra, mode amplitudes).

use crate::Real;
use crate::constants::{C0, ETA0, TAU};
use crate::engine::RawResults;
use crate::grid::{Component, Grid};
use crate::plan::{CompBox, DftKind, DftPlan, Plan};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Complex array `[freq][i][j][k]` over a global component index box.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ComplexField {
    pub comp: String,
    /// Global index of the first cell.
    pub origin: [usize; 3],
    /// (nfreq, nx, ny, nz)
    pub dims: [usize; 4],
    pub re: Vec<Real>,
    pub im: Vec<Real>,
}

impl ComplexField {
    pub fn at(&self, f: usize, i: usize, j: usize, k: usize) -> Complex64 {
        let o = ((f * self.dims[1] + i) * self.dims[2] + j) * self.dims[3] + k;
        Complex64::new(self.re[o] as f64, self.im[o] as f64)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct DftResult {
    pub freqs: Vec<f64>,
    pub fields: Vec<ComplexField>,
    /// The Fourier sums were accumulated every `stride` steps.
    #[serde(default)]
    pub stride: usize,
}

impl DftResult {
    pub fn field(&self, c: Component) -> Option<&ComplexField> {
        self.fields.iter().find(|f| f.comp == c.name())
    }
}

/// Time samples `[sample][i][j][k]` of one component over a global box.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TimeField {
    pub comp: String,
    pub origin: [usize; 3],
    /// (nsamples, nx, ny, nz)
    pub dims: [usize; 4],
    pub data: Vec<Real>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TimeResult {
    pub steps: Vec<usize>,
    /// Sample times for E components; H samples are dt/2 earlier.
    pub times: Vec<f64>,
    pub fields: Vec<TimeField>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModeResult {
    pub freqs: Vec<f64>,
    /// `[mode][freq]` forward (+axis) amplitudes.
    pub amp_plus_re: Vec<Vec<f64>>,
    pub amp_plus_im: Vec<Vec<f64>>,
    pub amp_minus_re: Vec<Vec<f64>>,
    pub amp_minus_im: Vec<Vec<f64>>,
    /// Total flux through the plane (for reference).
    pub flux: Vec<f64>,
}

/// Far field of a far-field monitor. Arrays over directions are flat,
/// `[freq][theta][phi]`.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FarFieldResult {
    pub freqs: Vec<f64>,
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    /// Radiant intensity U (W/sr).
    pub intensity: Vec<f64>,
    /// Power radiated through the projection box (flux), for reference.
    pub flux: Vec<f64>,
    /// Far-field amplitudes (V): the electric field far away is
    /// `(e_theta theta^ + e_phi phi^) exp(i k r) / r`, with `r` measured from
    /// the coordinate origin, and `U = (|e_theta|^2 + |e_phi|^2) / (2 eta)`.
    #[serde(default)]
    pub e_theta_re: Vec<f64>,
    #[serde(default)]
    pub e_theta_im: Vec<f64>,
    #[serde(default)]
    pub e_phi_re: Vec<f64>,
    #[serde(default)]
    pub e_phi_im: Vec<f64>,
}

/// Electromagnetic energy in the box of an energy monitor versus time.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EnergyResult {
    pub steps: Vec<usize>,
    /// Sample times (s) of the electric field; the magnetic field of the
    /// same sample is half a time step earlier.
    pub times: Vec<f64>,
    /// Energy (J): `eps0/2 sum (eps |E|^2 + mu |eta0 H|^2) dV` over the box.
    pub energy: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RunStats {
    pub backend: String,
    pub cells: usize,
    pub steps: usize,
    pub seconds: f64,
    pub mcells_per_s: f64,
    pub dt: f64,
    pub stopped_early: bool,
    /// The quantity the shutoff test looks at: the sum of the squared field
    /// values over the grid, at the steps in `energy_steps` (every
    /// `shutoff_interval` steps and at the last step).
    #[serde(default)]
    pub energy_steps: Vec<usize>,
    #[serde(default)]
    pub energy: Vec<f64>,
}

/// A source as the results need it: its signal, to evaluate the spectrum at
/// any frequency, and whether it injected the analytic signal (a complex
/// field pattern in a run with complex fields).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceInfo {
    /// The source name, `source<i>` for an unnamed source.
    pub name: String,
    pub signal: crate::sources::TimeSignal,
    #[serde(default)]
    pub analytic: bool,
}

impl SourceInfo {
    /// Spectrum of the injected signal at `freqs`, sampled as in a run of
    /// `nsteps` steps of `dt` (the E-time samples `(n + 1/2) dt`).
    pub fn spectrum(&self, freqs: &[f64], dt: f64, nsteps: usize) -> Vec<Complex64> {
        self.signal.spectrum_of(freqs, dt, nsteps.max(1), 0.5, self.analytic)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Results {
    pub dt: f64,
    pub nsteps: usize,
    pub dft: BTreeMap<String, DftResult>,
    pub flux: BTreeMap<String, Vec<f64>>,
    pub modes: BTreeMap<String, ModeResult>,
    #[serde(default)]
    pub far_field: BTreeMap<String, FarFieldResult>,
    pub time: BTreeMap<String, TimeResult>,
    /// Energy monitors (their field samples are reduced to the energy and
    /// are not kept).
    #[serde(default)]
    pub energy: BTreeMap<String, EnergyResult>,
    pub stats: RunStats,
    /// Source spectra (numerical DFT of the injected signals, E-time samples)
    /// keyed by source name: `(freqs, re, im)`. The frequencies are the union
    /// of the frequencies of all frequency-domain monitors, **sorted in
    /// ascending order**; a monitor that lists its frequencies in another
    /// order or records only some of them must be matched by value (or use
    /// [`SourceInfo::spectrum`] at the monitor's own frequencies).
    #[serde(default)]
    pub source_spectra: BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// The sources, to evaluate their spectra at other frequencies.
    #[serde(default)]
    pub sources: Vec<SourceInfo>,
    /// Gain media: number of atoms in levels 0, 1, 2, 3 at the end of the
    /// run, summed over all gain cells (empty without gain media).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gain_levels: Vec<f64>,
}

fn copy_box(bx: &CompBox, nf: usize, local: &[Real], global: &mut [Real]) {
    let g = bx.gsize();
    let (ly, lz) = (bx.lsize[1], bx.lsize[2]);
    let lcells = bx.lcells();
    let gcells = bx.gcells();
    for f in 0..nf {
        for ii in 0..bx.lsize[0] {
            let gi = bx.gx_off + ii;
            for jj in 0..ly {
                let lo = f * lcells + (ii * ly + jj) * lz;
                let go = f * gcells + (gi * g[1] + jj) * g[2];
                global[go..go + lz].copy_from_slice(&local[lo..lo + lz]);
            }
        }
    }
}

/// Assemble raw data from one or more slabs (same simulation) into global
/// arrays. Disjoint slabs simply fill different parts of each array.
pub fn assemble(plans: &[&Plan], raws: &[RawResults]) -> Results {
    let p0 = plans[0];
    let mut res = Results { dt: p0.dt, nsteps: p0.nsteps, ..Default::default() };
    for (di, d) in p0.dft.iter().enumerate() {
        let nf = d.freqs.len();
        let mut fields: Vec<ComplexField> = d
            .boxes
            .iter()
            .map(|b| {
                let g = b.gsize();
                ComplexField {
                    comp: b.comp.name().into(),
                    origin: b.glo,
                    dims: [nf, g[0], g[1], g[2]],
                    re: vec![0.0; nf * b.gcells()],
                    im: vec![0.0; nf * b.gcells()],
                }
            })
            .collect();
        for (pl, raw) in plans.iter().zip(raws) {
            let dd = &pl.dft[di];
            for bi in 0..dd.boxes.len() {
                // a shared box takes its data from the monitor that records it
                let (sd, sb) = DftPlan::source_of(&pl.dft, di, bi);
                let bx = &pl.dft[sd].boxes[sb];
                if !bx.is_local() {
                    continue;
                }
                let (re, im) = &raw.dft[sd].boxes[sb];
                copy_box(bx, nf, re, &mut fields[bi].re);
                copy_box(bx, nf, im, &mut fields[bi].im);
            }
        }
        res.dft.insert(d.name.clone(), DftResult { freqs: d.freqs.clone(), fields, stride: d.stride });
    }
    for (ti, t) in p0.time.iter().enumerate() {
        // number of samples taken (identical on all slabs)
        let steps = raws.iter().map(|r| r.time[ti].steps.clone()).max_by_key(|s| s.len()).unwrap_or_default();
        let ns = steps.len();
        // an energy monitor: reduce the samples of every slab to the energy
        if let Some(ei) = p0.energy.iter().position(|e| e.time == ti) {
            let mut energy = vec![0.0f64; ns];
            for (pl, raw) in plans.iter().zip(raws) {
                let ep = &pl.energy[ei];
                for (bi, w) in ep.weights.iter().enumerate() {
                    let data = &raw.time[ti].boxes[bi];
                    let nc = w.len();
                    if nc == 0 {
                        continue;
                    }
                    for (s, en) in energy.iter_mut().enumerate().take(data.len() / nc) {
                        let row = &data[s * nc..(s + 1) * nc];
                        *en += row.iter().zip(w.iter()).map(|(v, wt)| (*v as f64) * (*v as f64) * (*wt as f64)).sum::<f64>();
                    }
                }
            }
            let times = steps.iter().map(|s| (*s as f64 + 1.0) * p0.dt).collect();
            res.energy.insert(p0.energy[ei].name.clone(), EnergyResult { steps, times, energy });
            continue;
        }
        let mut fields: Vec<TimeField> = t
            .boxes
            .iter()
            .map(|b| {
                let g = b.gsize();
                TimeField { comp: b.comp.name().into(), origin: b.glo, dims: [ns, g[0], g[1], g[2]], data: vec![0.0; ns * b.gcells()] }
            })
            .collect();
        for (pl, raw) in plans.iter().zip(raws) {
            let tt = &pl.time[ti];
            for (bi, bx) in tt.boxes.iter().enumerate() {
                if !bx.is_local() {
                    continue;
                }
                copy_box(bx, ns, &raw.time[ti].boxes[bi], &mut fields[bi].data);
            }
        }
        let times = steps.iter().map(|s| (*s as f64 + 1.0) * p0.dt).collect();
        res.time.insert(t.name.clone(), TimeResult { steps, times, fields });
    }
    res
}

thread_local! {
    static CLIP: std::cell::Cell<([f64; 3], [f64; 3])> =
        const { std::cell::Cell::new(([f64::NEG_INFINITY; 3], [f64::INFINITY; 3])) };
}

/// Set the clip rectangle for surface elements (monitor region).
fn set_clip(r: ([f64; 3], [f64; 3])) {
    CLIP.with(|c| c.set(r));
}

/// Transverse voxel area of component `c` at global (i, j, k) for a plane
/// normal to `a`.
fn area(grid: &Grid, periodic: [bool; 3], c: Component, a: usize, g: [usize; 3]) -> f64 {
    let (cmin, cmax) = CLIP.with(|c| c.get());
    let mut s = 1.0;
    for ax in 0..3 {
        if ax == a {
            continue;
        }
        let n = grid.n()[ax];
        let w = if periodic[ax] && !c.is_half(ax) && g[ax] == 0 {
            // dual cell wraps around the periodic boundary
            0.5 * (grid.cell_size(ax, 0) + grid.cell_size(ax, n - 1))
        } else {
            let (lo, hi) = grid.voxel_bounds(c, ax, g[ax]);
            // clip to the monitor rectangle so that box faces sharing an
            // edge do not double count the edge elements
            (hi.min(cmax[ax]) - lo.max(cmin[ax])).max(0.0)
        };
        s *= w;
    }
    s
}

/// Flux through a plane given boxes [E_b, E_c, H_b(p-1), H_c(p-1), H_b(p), H_c(p)].
fn plane_flux(grid: &Grid, periodic: [bool; 3], a: usize, f: &[ComplexField], nf: usize) -> Vec<f64> {
    let mut out = vec![0.0; nf];
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    // E_b with H_c (avg of planes p-1 and p)
    for (e_i, h0_i, h1_i, sign, comp) in [(0usize, 3usize, 5usize, 1.0, Component::e(bax)), (1, 2, 4, -1.0, Component::e(cax))] {
        let e = &f[e_i];
        let h0 = &f[h0_i];
        let h1 = &f[h1_i];
        let d = e.dims;
        for (fi, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for i in 0..d[1] {
                for j in 0..d[2] {
                    for k in 0..d[3] {
                        let ev = e.at(fi, i, j, k);
                        let hv = (h0.at(fi, i, j, k) + h1.at(fi, i, j, k)) * 0.5 / ETA0;
                        let g = [e.origin[0] + i, e.origin[1] + j, e.origin[2] + k];
                        acc += (ev * hv.conj()).re * area(grid, periodic, comp, a, g);
                    }
                }
            }
            *o += 0.5 * sign * acc;
        }
    }
    let _ = TAU;
    out
}

/// Mode overlap amplitudes for a mode monitor.
fn mode_amplitudes(
    grid: &Grid,
    periodic: [bool; 3],
    a: usize,
    p: usize,
    origin: [usize; 3],
    f: &[ComplexField],
    freqs: &[f64],
    modes: &[crate::sources::ModeProfile],
) -> ModeResult {
    let nf = freqs.len();
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let (u_ax, _v_ax) = crate::geometry::plane_axes(a);
    let nat = |ax: usize| if ax == u_ax { 0usize } else { 1usize };
    let delta = grid.node(a, p) - grid.center(a, p - 1);
    let mut res = ModeResult { freqs: freqs.to_vec(), flux: plane_flux(grid, periodic, a, f, nf), ..Default::default() };
    let nm = if modes.len() == nf && nf > 1 { 1 } else { modes.len() };
    for mi in 0..nm {
        let mut ap_re = vec![0.0; nf];
        let mut ap_im = vec![0.0; nf];
        let mut am_re = vec![0.0; nf];
        let mut am_im = vec![0.0; nf];
        for fi in 0..nf {
            // a profile that carries the mode at other frequencies is taken at this one
            let mode = if modes.len() == nf && nf > 1 { &modes[fi] } else { &modes[mi] }.at_frequency(freqs[fi]);
            let mode: &crate::sources::ModeProfile = &mode;
            // H_m(p) = H_m(p-1/2) e^{i beta d}, with d the half cell the profile was solved for
            let shift = Complex64::from_polar(1.0, TAU * mode.freq * mode.neff / C0 * mode.h_offset.unwrap_or(delta));
            let dims = mode.dims;
            let get = |re: &Vec<f64>, im: &Vec<f64>, g: [usize; 3], origin: [usize; 3]| -> Complex64 {
                let (u_ax, v_ax) = crate::geometry::plane_axes(a);
                let iu = g[u_ax] as isize - origin[u_ax] as isize;
                let iv = g[v_ax] as isize - origin[v_ax] as isize;
                if iu < 0 || iv < 0 || iu as usize >= dims[0] || iv as usize >= dims[1] {
                    return Complex64::new(0.0, 0.0);
                }
                let o = iu as usize * dims[1] + iv as usize;
                Complex64::new(re[o], im.get(o).copied().unwrap_or(0.0))
            };
            // integrals I1 = int E x Hm*, I2 = int Em* x H, Pm = 1/2 Re int Em x Hm*
            let mut i1 = Complex64::new(0.0, 0.0);
            let mut i2 = Complex64::new(0.0, 0.0);
            let mut pm = 0.0;
            for (e_i, h0_i, h1_i, sign, e_ax, h_ax) in [(0usize, 3usize, 5usize, 1.0, bax, cax), (1, 2, 4, -1.0, cax, bax)] {
                let e = &f[e_i];
                let h0 = &f[h0_i];
                let h1 = &f[h1_i];
                let d = e.dims;
                let comp = Component::e(e_ax);
                for i in 0..d[1] {
                    for j in 0..d[2] {
                        for k in 0..d[3] {
                            let g = [e.origin[0] + i, e.origin[1] + j, e.origin[2] + k];
                            let da = area(grid, periodic, comp, a, g);
                            let ev = e.at(fi, i, j, k);
                            let hv = (h0.at(fi, i, j, k) + h1.at(fi, i, j, k)) * 0.5 / ETA0;
                            let em = get(&mode.e_re[nat(e_ax)], &mode.e_im[nat(e_ax)], g, origin);
                            let hm = get(&mode.h_re[nat(h_ax)], &mode.h_im[nat(h_ax)], g, origin) * shift;
                            i1 += ev * hm.conj() * (sign * da);
                            i2 += em.conj() * hv * (sign * da);
                            pm += 0.5 * sign * (em * hm.conj()).re * da;
                        }
                    }
                }
            }
            let pm = if pm.abs() > 0.0 { pm } else { 1.0 };
            // The H of the simulation on the monitor plane is the mean of the two
            // half-cell planes around it. For the mode's own waves e^{+-i beta x}
            // that mean is H(p) times c+ = (e^{-i beta d0} + e^{i beta d1}) / 2
            // (forward) and c- = conj(c+) (backward): cos(beta dx / 2) on a uniform
            // mesh. With A = I1 / 2 Pm = a+ + a- and B = I2 / 2 Pm = c+ a+ - c- a-
            // the two waves separate exactly. (A +- B) / 2 instead gives
            // ((1 + c) a+- + (1 - c) a-+) / 2: each direction picks up (1 - c) /
            // (1 + c) of the other, 2 % of its amplitude at 11 cells per guided
            // wavelength, so a reflection changes the forward amplitude.
            // The amplitudes are scaled by sqrt(Re c+), which makes |a+|^2 -
            // |a-|^2 the flux through the plane as a flux monitor computes it
            // (with the same mean of H): Pm Re(c+) (|a+|^2 - |a-|^2).
            let beta = TAU * freqs[fi] * mode.neff / C0;
            let d1 = if p < grid.n()[a] { grid.center(a, p) - grid.node(a, p) } else { delta };
            let cp = (Complex64::from_polar(1.0, -beta * delta) + Complex64::from_polar(1.0, beta * d1)) * 0.5;
            let cm = cp.conj();
            let (aa, bb) = (i1 / (2.0 * pm), i2 / (2.0 * pm));
            let scale = cp.re.max(0.0).sqrt() / (cp + cm);
            let plus = (cm * aa + bb) * scale;
            let minus = (cp * aa - bb) * scale;
            ap_re[fi] = plus.re;
            ap_im[fi] = plus.im;
            am_re[fi] = minus.re;
            am_im[fi] = minus.im;
        }
        res.amp_plus_re.push(ap_re);
        res.amp_plus_im.push(ap_im);
        res.amp_minus_re.push(am_re);
        res.amp_minus_im.push(am_im);
    }
    res
}

/// Far-field projection of the DFT fields on the faces of a closed box.
fn far_field(
    grid: &Grid,
    periodic: [bool; 3],
    fields: &[ComplexField],
    faces: &[(usize, f64, usize)],
    freqs: &[f64],
    theta: &[f64],
    phi: &[f64],
    n_medium: f64,
) -> (Vec<f64>, Vec<Complex64>, Vec<Complex64>) {
    let nf = freqs.len();
    let (nt, np) = (theta.len(), phi.len());
    let eta = ETA0 / n_medium;
    // collect surface current samples: position, J (3), M (3), area
    struct Pt {
        r: [f64; 3],
        j: [Complex64; 3],
        m: [Complex64; 3],
    }
    let mut out = vec![0.0; nf * nt * np];
    let mut e_theta = vec![Complex64::new(0.0, 0.0); nf * nt * np];
    let mut e_phi = vec![Complex64::new(0.0, 0.0); nf * nt * np];
    for fi in 0..nf {
        let mut pts: Vec<Pt> = vec![];
        for &(a, sgn, first) in faces {
            let f = &fields[first..first + 6];
            let bax = (a + 1) % 3;
            let cax = (a + 2) % 3;
            // (E_b with H_c): J_b = -s H_c, M_c = -s E_b ; (E_c with H_b): J_c = s H_b, M_b = s E_c
            for (e_i, h0_i, h1_i, is_b) in [(0usize, 3usize, 5usize, true), (1, 2, 4, false)] {
                let e = &f[e_i];
                let (h0, h1) = (&f[h0_i], &f[h1_i]);
                let comp = if is_b { Component::e(bax) } else { Component::e(cax) };
                let d = e.dims;
                for i in 0..d[1] {
                    for j in 0..d[2] {
                        for k in 0..d[3] {
                            let g = [e.origin[0] + i, e.origin[1] + j, e.origin[2] + k];
                            let da = area(grid, periodic, comp, a, g);
                            let ev = e.at(fi, i, j, k);
                            let hv = (h0.at(fi, i, j, k) + h1.at(fi, i, j, k)) * 0.5 / ETA0;
                            let r = grid.position3(comp, g);
                            let mut jv = [Complex64::new(0.0, 0.0); 3];
                            let mut mv = [Complex64::new(0.0, 0.0); 3];
                            if is_b {
                                jv[bax] = -sgn * hv * da;
                                mv[cax] = -sgn * ev * da;
                            } else {
                                jv[cax] = sgn * hv * da;
                                mv[bax] = sgn * ev * da;
                            }
                            pts.push(Pt { r, j: jv, m: mv });
                        }
                    }
                }
            }
        }
        let k = TAU * freqs[fi] * n_medium / C0;
        let pref = k * k / (32.0 * std::f64::consts::PI * std::f64::consts::PI * eta);
        for (ti, &th) in theta.iter().enumerate() {
            for (pi, &ph) in phi.iter().enumerate() {
                let (st, ct, sp, cp) = (th.sin(), th.cos(), ph.sin(), ph.cos());
                let rhat = [st * cp, st * sp, ct];
                let that = [ct * cp, ct * sp, -st];
                let phat = [-sp, cp, 0.0];
                let mut n = [Complex64::new(0.0, 0.0); 3];
                let mut l = [Complex64::new(0.0, 0.0); 3];
                for p in &pts {
                    let ph_ = Complex64::from_polar(1.0, -k * (rhat[0] * p.r[0] + rhat[1] * p.r[1] + rhat[2] * p.r[2]));
                    for c in 0..3 {
                        n[c] += p.j[c] * ph_;
                        l[c] += p.m[c] * ph_;
                    }
                }
                let dot = |v: &[Complex64; 3], u: &[f64; 3]| v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
                let (nt_, np_) = (dot(&n, &that), dot(&n, &phat));
                let (lt_, lp_) = (dot(&l, &that), dot(&l, &phat));
                let u = pref * ((lp_ + eta * nt_).norm_sqr() + (lt_ - eta * np_).norm_sqr());
                let o = (fi * nt + ti) * np + pi;
                out[o] = u;
                // E = (i k / 4 pi) [ (L_phi + eta N_theta) theta^ - (L_theta - eta N_phi) phi^ ] exp(i k r) / r
                let c = Complex64::new(0.0, k / (4.0 * std::f64::consts::PI));
                e_theta[o] = c * (lp_ + eta * nt_);
                e_phi[o] = -c * (lt_ - eta * np_);
            }
        }
    }
    (out, e_theta, e_phi)
}

/// Post-process DFT monitors (flux, modes) in place.
pub fn finalize(res: &mut Results, plan: &Plan) {
    for d in &plan.dft {
        let Some(dr) = res.dft.get(&d.name) else { continue };
        let nf = d.freqs.len();
        // clip surface elements to the monitor region (tolerance: half the
        // smallest cell, so planes snapped to nodes keep their full extent)
        let tol = 1e-6 * plan.grid.min_cell_size().iter().cloned().fold(f64::INFINITY, f64::min);
        let (rmin, rmax) = d.region;
        set_clip(([rmin[0] - tol, rmin[1] - tol, rmin[2] - tol], [rmax[0] + tol, rmax[1] + tol, rmax[2] + tol]));
        match &d.kind {
            DftKind::Field => {}
            DftKind::Flux { axis, .. } => {
                let fl = plane_flux(&plan.grid, plan.periodic, *axis, &dr.fields, nf);
                res.flux.insert(d.name.clone(), fl);
            }
            DftKind::FluxBox => {
                let mut tot = vec![0.0; nf];
                for (a, sign, first) in &d.faces {
                    let fl = plane_flux(&plan.grid, plan.periodic, *a, &dr.fields[*first..*first + 6], nf);
                    for (t, v) in tot.iter_mut().zip(fl) {
                        *t += sign * v;
                    }
                }
                res.flux.insert(d.name.clone(), tot);
            }
            DftKind::FarField { theta, phi, n_medium } => {
                let mut tot = vec![0.0; nf];
                for (a, sign, first) in &d.faces {
                    let fl = plane_flux(&plan.grid, plan.periodic, *a, &dr.fields[*first..*first + 6], nf);
                    for (t, v) in tot.iter_mut().zip(fl) {
                        *t += sign * v;
                    }
                }
                let (u, et, ep) = far_field(&plan.grid, plan.periodic, &dr.fields, &d.faces, &d.freqs, theta, phi, *n_medium);
                res.flux.insert(d.name.clone(), tot.clone());
                res.far_field.insert(
                    d.name.clone(),
                    FarFieldResult {
                        freqs: d.freqs.clone(),
                        theta: theta.clone(),
                        phi: phi.clone(),
                        intensity: u,
                        flux: tot,
                        e_theta_re: et.iter().map(|z| z.re).collect(),
                        e_theta_im: et.iter().map(|z| z.im).collect(),
                        e_phi_re: ep.iter().map(|z| z.re).collect(),
                        e_phi_im: ep.iter().map(|z| z.im).collect(),
                    },
                );
            }
            DftKind::Mode { axis, p, origin, modes } => {
                let mr = mode_amplitudes(&plan.grid, plan.periodic, *axis, *p, *origin, &dr.fields, &d.freqs, modes);
                res.flux.insert(d.name.clone(), mr.flux.clone());
                res.modes.insert(d.name.clone(), mr);
            }
        }
    }
    set_clip(([f64::NEG_INFINITY; 3], [f64::INFINITY; 3]));
    // source spectra at the union of DFT frequencies (E-time samples)
    let mut freqs: Vec<f64> = plan.dft.iter().flat_map(|d| d.freqs.clone()).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    freqs.dedup();
    res.sources.clear();
    for (si, s) in plan.sources.iter().enumerate() {
        // a complex pattern in a complex-field (Bloch) run injects the analytic signal
        let analytic = plan.source_analytic.get(si).copied().unwrap_or(false);
        let key = if s.name().is_empty() { format!("source{si}") } else { s.name().to_string() };
        let info = SourceInfo { name: key.clone(), signal: s.signal().clone(), analytic };
        if !freqs.is_empty() {
            let sp = info.spectrum(&freqs, plan.dt, res.stats.steps);
            res.source_spectra.insert(
                key,
                (freqs.clone(), sp.iter().map(|z| z.re).collect(), sp.iter().map(|z| z.im).collect()),
            );
        }
        res.sources.push(info);
    }
}

impl Results {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
