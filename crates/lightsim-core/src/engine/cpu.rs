//! Multithreaded CPU reference backend.
//!
//! Mirrors the GPU kernels operation by operation (same storage layout, same
//! arithmetic order) so GPU results can be validated against it, and so the
//! physics test-suite runs on machines without a GPU.

use crate::Real;
use super::{Backend, RawDft, RawResults, RawTime, Side};
use crate::grid::Component;
use crate::plan::{GhostRule, Plan};
use anyhow::Result;

fn bail_unused() {}
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct Ptr(*mut Real);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    #[inline(always)]
    unsafe fn get(self, i: usize) -> Real {
        unsafe { *self.0.add(i) }
    }
    #[inline(always)]
    unsafe fn set(self, i: usize, v: Real) {
        unsafe { *self.0.add(i) = v }
    }
    #[inline(always)]
    unsafe fn add(self, i: usize, v: Real) {
        unsafe { *self.0.add(i) += v }
    }
}

pub struct CpuBackend {
    plan: Plan,
    e: [Vec<Real>; 3],
    h: [Vec<Real>; 3],
    psi: Vec<[Vec<Real>; 4]>,
    disp: Vec<(Vec<Real>, Vec<Real>, Vec<Real>)>,
    dft: Vec<RawDft>,
    time: Vec<RawTime>,
    /// Bloch mode: the imaginary field set (and its auxiliaries). The update
    /// routines always work on `e`/`h`; the sets are swapped to process the
    /// imaginary part, and `*_im` then holds the partner (real) set.
    cplx: Option<Box<ImagSet>>,
    /// Sign of the partner term for the set currently in `e`/`h`
    /// (-sin for the real part, +sin for the imaginary part), per axis.
    bsign: [Real; 3],
    imag_active: bool,
    /// E^n at the nonlinear cells, per nonlinear component.
    nl_prev: Vec<Vec<Real>>,
    /// chi3 Raman cells, per component: (E^n, Q^{n+1}, Q^n) (after `raman_save`).
    raman: Vec<(Vec<Real>, Vec<Real>, Vec<Real>)>,
    /// Gain cells, per component: (E^n, P, P_prev, dP) and the populations
    /// `[level][cell]` (f64: a step moves a tiny fraction of the atoms).
    gain: Vec<([Vec<Real>; 4], Vec<f64>)>,
    /// Off-diagonal tensors: dD of x/y/z over the plane range.
    t_prev: [Vec<Real>; 3],
    /// Off-diagonal tensors: E^n of x/y/z over the plane range.
    t_en: [Vec<Real>; 3],
    /// Oriented poles: (s, s_prev) per tensor component, `[pole][cell]`.
    t_os: Vec<(Vec<Real>, Vec<Real>)>,
    /// Gyrotropic poles: (P, P_prev) per gyrotropic component, `[pole][xyz][cell]`.
    t_gy: Vec<(Vec<Real>, Vec<Real>)>,
}

struct ImagSet {
    e: [Vec<Real>; 3],
    h: [Vec<Real>; 3],
    psi: Vec<[Vec<Real>; 4]>,
    disp: Vec<(Vec<Real>, Vec<Real>, Vec<Real>)>,
    dft: Vec<RawDft>,
}

impl CpuBackend {
    pub fn new(plan: Plan) -> Result<CpuBackend> {
        let len = plan.storage_len();
        let psi: Vec<[Vec<Real>; 4]> = plan.pml.iter().map(|p| {
            let c = p.cells();
            [vec![0 as Real; c], vec![0 as Real; c], vec![0 as Real; c], vec![0 as Real; c]]
        }).collect();
        let disp: Vec<(Vec<Real>, Vec<Real>, Vec<Real>)> = match &plan.disp {
            Some(d) => d
                .comps
                .iter()
                .map(|c| (vec![0 as Real; c.idx.len() * d.npole], vec![0 as Real; c.idx.len() * d.npole], vec![0 as Real; c.idx.len()]))
                .collect(),
            None => vec![],
        };
        let dft: Vec<RawDft> = plan
            .dft
            .iter()
            .map(|d| RawDft {
                boxes: d
                    .boxes
                    .iter()
                    .map(|b| (vec![0 as Real; b.lcells() * d.freqs.len()], vec![0 as Real; b.lcells() * d.freqs.len()]))
                    .collect(),
            })
            .collect();
        let time = plan.time.iter().map(|t| RawTime { boxes: vec![Vec::new(); t.boxes.len()], steps: vec![] }).collect();
        let cplx = if plan.bloch {
            Some(Box::new(ImagSet {
                e: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
                h: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
                psi: psi.clone(),
                disp: disp.clone(),
                dft: dft.clone(),
            }))
        } else {
            None
        };
        let bsign = [
            -(plan.bloch_phase[0].1 as Real),
            -(plan.bloch_phase[1].1 as Real),
            -(plan.bloch_phase[2].1 as Real),
        ];
        let _ = bail_unused;
        let t_prev = match &plan.tensor {
            Some(t) => {
                let n = (t.p1 - t.p0) * plan.plane();
                [vec![0 as Real; n], vec![0 as Real; n], vec![0 as Real; n]]
            }
            None => [vec![], vec![], vec![]],
        };
        let t_en = t_prev.clone();
        let t_os = plan.tensor.as_ref().map(|t| t.comps.iter().map(|c| (vec![0 as Real; c.idx.len() * t.op_np], vec![0 as Real; c.idx.len() * t.op_np])).collect()).unwrap_or_default();
        let t_gy = plan.tensor.as_ref().map(|t| t.gyro.iter().map(|c| (vec![0 as Real; c.idx.len() * 3 * t.gyro_np], vec![0 as Real; c.idx.len() * 3 * t.gyro_np])).collect()).unwrap_or_default();
        let nl_prev = plan.nonlin.as_ref().map(|n| n.comps.iter().map(|c| vec![0 as Real; c.idx.len()]).collect()).unwrap_or_default();
        let gain = plan
            .nonlin
            .as_ref()
            .and_then(|n| n.gain.as_ref())
            .map(|g| {
                g.comps
                    .iter()
                    .map(|c| {
                        let n = c.idx.len();
                        let mut pop = vec![0f64; 4 * n];
                        for (t, &r) in c.row.iter().enumerate() {
                            for l in 0..4 {
                                pop[l * n + t] = g.init[4 * r as usize + l];
                            }
                        }
                        ([vec![0 as Real; n], vec![0 as Real; n], vec![0 as Real; n], vec![0 as Real; n]], pop)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let raman = plan
            .nonlin
            .as_ref()
            .and_then(|n| n.raman.as_ref())
            .map(|r| r.comps.iter().map(|c| (vec![0 as Real; c.idx.len()], vec![0 as Real; c.idx.len()], vec![0 as Real; c.idx.len()])).collect())
            .unwrap_or_default();
        Ok(CpuBackend {
            e: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
            h: [vec![0 as Real; len], vec![0 as Real; len], vec![0 as Real; len]],
            psi,
            disp,
            dft,
            time,
            plan,
            cplx,
            bsign,
            imag_active: false,
            nl_prev,
            raman,
            gain,
            t_prev,
            t_en,
            t_os,
            t_gy,
        })
    }

    /// Off-diagonal tensors: save E^n over the interior plane range.
    fn tensor_save(&mut self) {
        if let Some(t) = &self.plan.tensor {
            let plane = self.plan.plane();
            let (s0, s1, o) = (t.i0 * plane, t.i1 * plane, (t.i0 - t.p0) * plane);
            for a in 0..3 {
                self.t_en[a][o..o + s1 - s0].copy_from_slice(&self.e[a][s0..s1]);
            }
        }
    }

    /// Off-diagonal tensors: dD = (E_lin - E^n) cdt / cb over the interior range.
    fn tensor_dd(&mut self) {
        let Some(t) = &self.plan.tensor else { return };
        let plane = self.plan.plane();
        let (s0, s1, o) = (t.i0 * plane, t.i1 * plane, (t.i0 - t.p0) * plane);
        for a in 0..3 {
            let (f, cb, prev, en) = (&self.e[a], &self.plan.cb[a], &mut self.t_prev[a], &self.t_en[a]);
            let ca = self.plan.ca.as_ref().map(|c| &c[a]);
            for (q, p) in prev[o..o + s1 - s0].iter_mut().enumerate() {
                let s = s0 + q;
                let cav = ca.map(|c| c[s]).unwrap_or(1.0);
                *p = if cb[s] != 0.0 { (f[s] - cav * en[o + q]) * (t.cdt / cb[s]) } else { 0.0 };
            }
        }
    }

    /// Off-diagonal tensors: E_a += kappa_ab <dD_b> + kappa_ac <dD_c> (see the
    /// `tensor_apply` kernel).
    fn tensor_apply(&mut self) {
        let Some(t) = &self.plan.tensor else { return };
        let plane = self.plan.plane();
        let (ny, nz) = (self.plan.ny() as i64, self.plan.nz() as i64);
        let (lo, hi) = (t.p0 as i64, t.p1 as i64);
        let np = t.op_np;
        for (ci, c) in t.comps.iter().enumerate() {
            let a = c.comp.axis();
            let dd = [&self.t_prev[(a + 1) % 3], &self.t_prev[(a + 2) % 3]];
            let en = [&self.t_en[(a + 1) % 3], &self.t_en[(a + 2) % 3]];
            let en_own = &self.t_en[a];
            let (ost, osp) = &mut self.t_os[ci];
            let ncell = c.idx.len();
            let f = &mut self.e[a];
            for (q, &s) in c.idx.iter().enumerate() {
                let s = s as usize;
                let pi = (s / plane) as i64;
                let rem = (s % plane) as i64;
                let (j, k) = (rem / nz, rem % nz);
                let mut acc = [0 as Real; 2];
                let mut acc_e = [0 as Real; 2];
                for w in 0..8 {
                    let second = w >= 4;
                    let b = if second { (a + 2) % 3 } else { (a + 1) % 3 };
                    let (up, dn) = (w & 1 != 0, w & 2 != 0);
                    let d = |ax: usize| (up && a == ax) as i64 - (dn && b == ax) as i64;
                    let (mut x, mut y, mut z) = (pi + d(0), j + d(1), k + d(2));
                    let mut ok = true;
                    if x < lo || x >= hi {
                        if t.wrap & 1 != 0 { x += if x < lo { hi - lo } else { lo - hi } } else { ok = false }
                    }
                    if y < 0 || y >= ny {
                        if t.wrap & 2 != 0 { y += if y < 0 { ny } else { -ny } } else { ok = false }
                    }
                    if z < 0 || z >= nz {
                        if t.wrap & 4 != 0 { z += if z < 0 { nz } else { -nz } } else { ok = false }
                    }
                    if ok {
                        let o = ((x - lo) * ny * nz + y * nz + z) as usize;
                        acc[second as usize] += dd[second as usize][o];
                        if t.lossy {
                            acc_e[second as usize] += en[second as usize][o];
                        }
                    }
                }
                f[s] += 0.25 * (c.c1[q] * acc[0] + c.c2[q] * acc[1] + c.m1[q] * acc_e[0] + c.m2[q] * acc_e[1]);
                // oriented poles: s^{n+1} = c0 s + c1 s_prev + c2 (u . E^n), E_a -= u_a ds / eps_inf
                if np > 0 && c.opm[q] != u32::MAX {
                    let row = c.opm[q] as usize;
                    let mut ev = [0 as Real; 3];
                    ev[a] = en_own[((pi - lo) * ny * nz + j * nz + k) as usize];
                    ev[(a + 1) % 3] = 0.25 * acc_e[0];
                    ev[(a + 2) % 3] = 0.25 * acc_e[1];
                    let mut ds = 0 as Real;
                    for pp in 0..np {
                        let tb = &t.op_tab[(row * np + pp) * 6..(row * np + pp) * 6 + 6];
                        let drive = tb[3] * ev[0] + tb[4] * ev[1] + tb[5] * ev[2];
                        let o = pp * ncell + q;
                        let s0 = ost[o];
                        let sn = tb[0] * s0 + tb[1] * osp[o] + tb[2] * drive;
                        osp[o] = s0;
                        ost[o] = sn;
                        ds += tb[3 + a] * (sn - s0);
                    }
                    f[s] -= c.oscale[q] * ds;
                }
            }
        }
    }

    /// Gyrotropic poles (see the `gyro_apply` kernel): at each listed E_a
    /// cell, E^n = (E_a, <E_b>, <E_c>) with the four-point averages of
    /// `tensor_apply`, P^{n+1} = A P^n + B P^{n-1} + C E^n per pole (3x3
    /// matrices), E_a -= (sum over poles of dP_a) / eps_inf.
    fn gyro_apply(&mut self) {
        let Some(t) = &self.plan.tensor else { return };
        if t.gyro.is_empty() {
            return;
        }
        let plane = self.plan.plane();
        let (ny, nz) = (self.plan.ny() as i64, self.plan.nz() as i64);
        let (lo, hi) = (t.p0 as i64, t.p1 as i64);
        let np = t.gyro_np;
        for (ci, c) in t.gyro.iter().enumerate() {
            let a = c.comp.axis();
            let en = [&self.t_en[(a + 1) % 3], &self.t_en[(a + 2) % 3]];
            let en_own = &self.t_en[a];
            let (pst, psp) = &mut self.t_gy[ci];
            let ncell = c.idx.len();
            let f = &mut self.e[a];
            for (q, &s) in c.idx.iter().enumerate() {
                let s = s as usize;
                let pi = (s / plane) as i64;
                let rem = (s % plane) as i64;
                let (j, k) = (rem / nz, rem % nz);
                let mut acc_e = [0 as Real; 2];
                for w in 0..8 {
                    let second = w >= 4;
                    let b = if second { (a + 2) % 3 } else { (a + 1) % 3 };
                    let (up, dn) = (w & 1 != 0, w & 2 != 0);
                    let d = |ax: usize| (up && a == ax) as i64 - (dn && b == ax) as i64;
                    let (mut x, mut y, mut z) = (pi + d(0), j + d(1), k + d(2));
                    let mut ok = true;
                    if x < lo || x >= hi {
                        if t.wrap & 1 != 0 { x += if x < lo { hi - lo } else { lo - hi } } else { ok = false }
                    }
                    if y < 0 || y >= ny {
                        if t.wrap & 2 != 0 { y += if y < 0 { ny } else { -ny } } else { ok = false }
                    }
                    if z < 0 || z >= nz {
                        if t.wrap & 4 != 0 { z += if z < 0 { nz } else { -nz } } else { ok = false }
                    }
                    if ok {
                        acc_e[second as usize] += en[second as usize][((x - lo) * ny * nz + y * nz + z) as usize];
                    }
                }
                let mut ev = [0 as Real; 3];
                ev[a] = en_own[((pi - lo) * ny * nz + j * nz + k) as usize];
                ev[(a + 1) % 3] = 0.25 * acc_e[0];
                ev[(a + 2) % 3] = 0.25 * acc_e[1];
                let row = c.row[q] as usize;
                let mut dp = 0 as Real;
                for pp in 0..np {
                    let tb = &t.gyro_tab[(row * np + pp) * 27..(row * np + pp) * 27 + 27];
                    let o = |r: usize| (pp * 3 + r) * ncell + q;
                    let p0 = [pst[o(0)], pst[o(1)], pst[o(2)]];
                    let pm = [psp[o(0)], psp[o(1)], psp[o(2)]];
                    for r in 0..3 {
                        let mut v = 0 as Real;
                        for cc in 0..3 {
                            v += tb[r * 3 + cc] * p0[cc] + tb[9 + r * 3 + cc] * pm[cc] + tb[18 + r * 3 + cc] * ev[cc];
                        }
                        psp[o(r)] = p0[r];
                        pst[o(r)] = v;
                        if r == a {
                            dp += v - p0[r];
                        }
                    }
                }
                f[s] -= c.scale[q] * dp;
            }
        }
    }

    // ---------------- time reversal ----------------
    //
    // The leapfrog update is exactly invertible:
    //   E^n = (E^{n+1} - cb curl H^{n+1/2}) / ca,   H^{n-1/2} = H^{n+1/2} + ch curl E^n.
    // `reverse_e` / `reverse_h` run the forward kernels with the inverted
    // coefficients (-cb/ca, 1/ca, -ch). Absorbing PML layers are not
    // reversible: their fields are recorded on the forward run
    // (`boundary_record`) and replayed (`boundary_restore_*`) between the
    // two half steps (cf. FDTDX's time-reversal with recording).

    fn reversible(&self) -> Result<()> {
        if self.plan.disp.is_some() || self.plan.nonlin.is_some() || self.plan.tensor.is_some() {
            anyhow::bail!("time reversal supports non-dispersive, linear, diagonal media");
        }
        Ok(())
    }

    /// Save E^n at the nonlinear cells (before the linear E update).
    fn nl_save(&mut self) {
        if let Some(nl) = &self.plan.nonlin {
            let twod = self.plan.twod;
            for (c, prev) in nl.comps.iter().zip(self.nl_prev.iter_mut()) {
                if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                    continue;
                }
                let f = &self.e[c.comp.axis()];
                for (t, &s) in c.idx.iter().enumerate() {
                    prev[t] = f[s as usize];
                }
            }
        }
    }

    /// Implicit chi2/chi3 correction of the linearly updated E (see the
    /// `nl_apply` kernel for the equation).
    fn nl_apply(&mut self) {
        if let Some(nl) = &self.plan.nonlin {
            let twod = self.plan.twod;
            for (c, prev) in nl.comps.iter().zip(self.nl_prev.iter()) {
                if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                    continue;
                }
                let f = &mut self.e[c.comp.axis()];
                for (t, &s) in c.idx.iter().enumerate() {
                    let s = s as usize;
                    f[s] = nl_solve(prev[t], f[s], c.c2[t], c.c3[t]);
                }
            }
        }
    }

    /// chi3 Raman cells, before the linear E update: save E^n and advance
    /// Q^{n+1} = c0 Q^n + c1 Q^{n-1} + c2 (E^n)^2 (see the `raman_save` kernel).
    fn raman_save(&mut self) {
        let Some(rp) = self.plan.nonlin.as_ref().and_then(|n| n.raman.as_ref()) else { return };
        let twod = self.plan.twod;
        for (c, (prev, q, qp)) in rp.comps.iter().zip(self.raman.iter_mut()) {
            if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                continue;
            }
            let f = &self.e[c.comp.axis()];
            for (t, &s) in c.idx.iter().enumerate() {
                let e = f[s as usize];
                let k = &rp.coefs[c.matid[t] as usize * 3..c.matid[t] as usize * 3 + 3];
                prev[t] = e;
                let qn = k[0] * q[t] + k[1] * qp[t] + k[2] * (e * e);
                qp[t] = q[t];
                q[t] = qn;
            }
        }
    }

    /// chi3 Raman cells, after the linear E update: implicit solve for E^{n+1}
    /// (see `raman_solve`).
    fn raman_apply(&mut self) {
        let Some(rp) = self.plan.nonlin.as_ref().and_then(|n| n.raman.as_ref()) else { return };
        let twod = self.plan.twod;
        for (c, (prev, q, qp)) in rp.comps.iter().zip(self.raman.iter()) {
            if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                continue;
            }
            let f = &mut self.e[c.comp.axis()];
            for (t, &s) in c.idx.iter().enumerate() {
                let s = s as usize;
                f[s] = raman_solve(prev[t], f[s], c.c2[t], c.c3[t], c.cr[t], q[t], qp[t]);
            }
        }
    }

    /// Gain cells, before the linear E update (see the `gain_pre` kernel).
    fn gain_pre(&mut self) {
        let Some(gp) = self.plan.nonlin.as_ref().and_then(|n| n.gain.as_ref()) else { return };
        let twod = self.plan.twod;
        for (c, (st, pop)) in gp.comps.iter().zip(self.gain.iter_mut()) {
            if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                continue;
            }
            let f = &self.e[c.comp.axis()];
            let n = c.idx.len();
            let [prev, p, pp, dp] = st;
            for (t, &s) in c.idx.iter().enumerate() {
                let e = f[s as usize];
                let k = &gp.tab[c.row[t] as usize * crate::plan::GAIN_TAB..];
                let drive = (k[2] * (pop[n + t] - pop[2 * n + t])) as Real;
                let p0 = p[t];
                let pn = k[0] as Real * p0 + k[1] as Real * pp[t] + drive * e;
                prev[t] = e;
                pp[t] = p0;
                p[t] = pn;
                dp[t] = pn - p0;
            }
        }
    }

    /// Gain cells, after the linear E update (see the `gain_post` kernel).
    fn gain_post(&mut self) {
        let Some(gp) = self.plan.nonlin.as_ref().and_then(|n| n.gain.as_ref()) else { return };
        let twod = self.plan.twod;
        for (c, (st, pop)) in gp.comps.iter().zip(self.gain.iter_mut()) {
            if twod.map(|t| !t.is_active(c.comp)).unwrap_or(false) {
                continue;
            }
            let f = &mut self.e[c.comp.axis()];
            let n = c.idx.len();
            let [prev, _, _, dp] = st;
            for (t, &s) in c.idx.iter().enumerate() {
                let s = s as usize;
                let e1 = f[s] - c.scale[t] * dp[t];
                f[s] = e1;
                let k = &gp.tab[c.row[t] as usize * crate::plan::GAIN_TAB..];
                gain_rates(pop, n, t, k, 0.5 * (prev[t] as f64 + e1 as f64) * dp[t] as f64);
            }
        }
    }

    /// Swap the real and imaginary field sets (Bloch mode).
    fn swap_sets(&mut self) {
        if let Some(c) = self.cplx.as_mut() {
            std::mem::swap(&mut self.e, &mut c.e);
            std::mem::swap(&mut self.h, &mut c.h);
            std::mem::swap(&mut self.psi, &mut c.psi);
            std::mem::swap(&mut self.disp, &mut c.disp);
            std::mem::swap(&mut self.dft, &mut c.dft);
            for b in self.bsign.iter_mut() {
                *b = -*b;
            }
            self.imag_active = !self.imag_active;
        }
    }

    /// Partner (other part) field pointers; dummies when not in Bloch mode.
    fn partner(&self) -> ([Ptr; 3], [Ptr; 3]) {
        match &self.cplx {
            Some(c) => (
                [Ptr(c.e[0].as_ptr() as *mut Real), Ptr(c.e[1].as_ptr() as *mut Real), Ptr(c.e[2].as_ptr() as *mut Real)],
                [Ptr(c.h[0].as_ptr() as *mut Real), Ptr(c.h[1].as_ptr() as *mut Real), Ptr(c.h[2].as_ptr() as *mut Real)],
            ),
            None => {
                let d = Ptr(self.e[0].as_ptr() as *mut Real);
                ([d, d, d], [d, d, d])
            }
        }
    }

    fn arr(&self, c: Component) -> &Vec<Real> {
        if c.is_e() { &self.e[c.axis()] } else { &self.h[c.axis()] }
    }
    fn arr_mut(&mut self, c: Component) -> &mut Vec<Real> {
        if c.is_e() { &mut self.e[c.axis()] } else { &mut self.h[c.axis()] }
    }

    fn update_h_core(&mut self) -> Result<()> {
        // magnetic loss: H~ *= da on the lossy cells, then the regular update
        if let Some(hl) = &self.plan.hloss {
            for (a, l) in hl.iter().enumerate() {
                let h = &mut self.h[a];
                for (&s, &d) in l.idx.iter().zip(&l.da) {
                    h[s as usize] *= d;
                }
            }
        }
        // one periodic cell along an axis: the 2-D update of the sets in use
        let twod = self.plan.twod;
        if let Some(td) = twod {
            let eo = self.cplx.as_deref().map(|c| &c.e);
            super::twod::update_h(&self.plan, td, &self.e, &mut self.h, eo, self.bsign);
            super::twod::pml(&self.plan, td, &mut self.e, &mut self.h, &mut self.psi, true);
            return Ok(());
        }
        let (eo, ho) = self.partner();
        let (bcy, bsy) = (self.plan.bloch_phase[1].0 as Real, self.bsign[1]);
        let (bcz, bsz) = (self.plan.bloch_phase[2].0 as Real, self.bsign[2]);
        let _ = (&ho, bcz, bsz);
        let p = &self.plan;
        let (ny, nz, nxl) = (p.ny(), p.nz(), p.nxl);
        let plane = ny * nz;
        let ex = Ptr(self.e[0].as_ptr() as *mut Real);
        let ey = Ptr(self.e[1].as_ptr() as *mut Real);
        let ez = Ptr(self.e[2].as_ptr() as *mut Real);
        let hx = Ptr(self.h[0].as_mut_ptr());
        let hy = Ptr(self.h[1].as_mut_ptr());
        let hz = Ptr(self.h[2].as_mut_ptr());
        let ch = p.ch.as_ref().map(|c| [Ptr(c[0].as_ptr() as *mut Real), Ptr(c[1].as_ptr() as *mut Real), Ptr(c[2].as_ptr() as *mut Real)]);
        let chs = p.ch_scalar;
        let (idx, idy, idz) = (&p.inv_p[0], &p.inv_p[1], &p.inv_p[2]);
        let (bc_yhi, bc_zhi) = (p.bc_y[1], p.bc_z[1]);
        (0..nxl).into_par_iter().for_each(|i| unsafe {
            let dxi = idx[i];
            for j in 0..ny {
                let dyi = idy[j];
                for k in 0..nz {
                    let s = (i + 1) * plane + j * nz + k;
                    let ex0 = ex.get(s);
                    let ey0 = ey.get(s);
                    let ez0 = ez.get(s);
                    let ey_xp = ey.get(s + plane);
                    let ez_xp = ez.get(s + plane);
                    let (ex_yp, ez_yp) = if j + 1 < ny {
                        (ex.get(s + nz), ez.get(s + nz))
                    } else if bc_yhi == 1 {
                        (ex.get(s - (ny - 1) * nz), ez.get(s - (ny - 1) * nz))
                    } else if bc_yhi == 2 {
                        (ex0, ez0)
                    } else if bc_yhi == 3 {
                        let w = s - (ny - 1) * nz;
                        (bcy * ex.get(w) + bsy * eo[0].get(w), bcy * ez.get(w) + bsy * eo[2].get(w))
                    } else {
                        (0.0, 0.0)
                    };
                    let (ex_zp, ey_zp) = if k + 1 < nz {
                        (ex.get(s + 1), ey.get(s + 1))
                    } else if bc_zhi == 1 {
                        (ex.get(s - (nz - 1)), ey.get(s - (nz - 1)))
                    } else if bc_zhi == 2 {
                        (ex0, ey0)
                    } else if bc_zhi == 3 {
                        let w = s - (nz - 1);
                        (bcz * ex.get(w) + bsz * eo[0].get(w), bcz * ey.get(w) + bsz * eo[1].get(w))
                    } else {
                        (0.0, 0.0)
                    };
                    let dzi = idz[k];
                    let cx = (ez_yp - ez0) * dyi - (ey_zp - ey0) * dzi;
                    let cy = (ex_zp - ex0) * dzi - (ez_xp - ez0) * dxi;
                    let cz = (ey_xp - ey0) * dxi - (ex_yp - ex0) * dyi;
                    match ch {
                        Some(c) => {
                            hx.set(s, hx.get(s) - c[0].get(s) * cx);
                            hy.set(s, hy.get(s) - c[1].get(s) * cy);
                            hz.set(s, hz.get(s) - c[2].get(s) * cz);
                        }
                        None => {
                            hx.set(s, hx.get(s) - chs * cx);
                            hy.set(s, hy.get(s) - chs * cy);
                            hz.set(s, hz.get(s) - chs * cz);
                        }
                    }
                }
            }
        });
        // CPML corrections
        for (si, slab) in p.pml.iter().enumerate() {
            let a = slab.axis;
            let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
            let hb = Ptr(self.h[bax].as_mut_ptr());
            let hc = Ptr(self.h[cax].as_mut_ptr());
            let eb = Ptr(self.e[bax].as_ptr() as *mut Real);
            let ec = Ptr(self.e[cax].as_ptr() as *mut Real);
            let chb = ch.map(|c| c[bax]);
            let chc = ch.map(|c| c[cax]);
            let psib = Ptr(self.psi[si][2].as_mut_ptr());
            let psic = Ptr(self.psi[si][3].as_mut_ptr());
            let inv = &p.inv_p[a];
            let n_along = p.n[a];
            let (lx, ly, lz) = (slab.size[0], slab.size[1], slab.size[2]);
            let stride = match a {
                0 => plane,
                1 => nz,
                _ => 1,
            };
            (0..lx).into_par_iter().for_each(|ii| unsafe {
                for jj in 0..ly {
                    for kk in 0..lz {
                        let t = (ii * ly + jj) * lz + kk;
                        let i = slab.start[0] + ii;
                        let j = slab.start[1] + jj;
                        let k = slab.start[2] + kk;
                        let s = ((i + 1) * ny + j) * nz + k;
                        let along = [i, j, k][a];
                        let q = along - slab.start[a];
                        let b = slab.h_b[q];
                        let aa = slab.h_a[q];
                        let km1 = slab.h_kinv[q] - 1.0;
                        let invv = inv[along];
                        let has_next = a == 0 || along + 1 < n_along;
                        let ec_n = if has_next { ec.get(s + stride) } else { 0.0 };
                        let eb_n = if has_next { eb.get(s + stride) } else { 0.0 };
                        let d_ec = (ec_n - ec.get(s)) * invv;
                        let d_eb = (eb_n - eb.get(s)) * invv;
                        let pb = b * psib.get(t) + aa * d_ec;
                        let pc = b * psic.get(t) + aa * d_eb;
                        psib.set(t, pb);
                        psic.set(t, pc);
                        let cb_ = chb.map(|c| c.get(s)).unwrap_or(chs);
                        let cc_ = chc.map(|c| c.get(s)).unwrap_or(chs);
                        hb.add(s, cb_ * (km1 * d_ec + pb));
                        hc.add(s, -cc_ * (km1 * d_eb + pc));
                    }
                }
            });
        }
        Ok(())
    }

    fn update_e_core(&mut self) -> Result<()> {
        let twod = self.plan.twod;
        let (eo, ho) = self.partner();
        let (bcy, bsy) = (self.plan.bloch_phase[1].0 as Real, self.bsign[1]);
        let (bcz, bsz) = (self.plan.bloch_phase[2].0 as Real, self.bsign[2]);
        let _ = (&eo, bcz, bsz);
        // dispersive step 1: P update from E^n
        if let Some(d) = &self.plan.disp {
            for (ci, dc) in d.comps.iter().enumerate() {
                if twod.map(|t| !t.is_active(dc.comp)).unwrap_or(false) {
                    continue;
                }
                let f = &self.e[dc.comp.axis()];
                let (p, pp, dp) = &mut self.disp[ci];
                let n = dc.idx.len();
                let np = d.npole;
                for t in 0..n {
                    let s = dc.idx[t] as usize;
                    let e = f[s];
                    let m = dc.matid[t] as usize;
                    let mut acc = 0 as Real;
                    for q in 0..np {
                        let c = &d.coefs[(m * np + q) * 3..(m * np + q) * 3 + 3];
                        let o = q * n + t;
                        let p0 = p[o];
                        let pn = c[0] * p0 + c[1] * pp[o] + c[2] * e;
                        acc += pn - p0;
                        pp[o] = p0;
                        p[o] = pn;
                    }
                    dp[t] = acc;
                }
            }
        }
        if let Some(td) = twod {
            let ho = self.cplx.as_deref().map(|c| &c.h);
            super::twod::update_e(&self.plan, td, &mut self.e, &self.h, ho, self.bsign);
            super::twod::pml(&self.plan, td, &mut self.e, &mut self.h, &mut self.psi, false);
        }
        let p = &self.plan;
        let (ny, nz, nxl) = (p.ny(), p.nz(), p.nxl);
        let plane = ny * nz;
        let hx = Ptr(self.h[0].as_ptr() as *mut Real);
        let hy = Ptr(self.h[1].as_ptr() as *mut Real);
        let hz = Ptr(self.h[2].as_ptr() as *mut Real);
        let ex = Ptr(self.e[0].as_mut_ptr());
        let ey = Ptr(self.e[1].as_mut_ptr());
        let ez = Ptr(self.e[2].as_mut_ptr());
        let cb = [Ptr(p.cb[0].as_ptr() as *mut Real), Ptr(p.cb[1].as_ptr() as *mut Real), Ptr(p.cb[2].as_ptr() as *mut Real)];
        let ca = p.ca.as_ref().map(|c| [Ptr(c[0].as_ptr() as *mut Real), Ptr(c[1].as_ptr() as *mut Real), Ptr(c[2].as_ptr() as *mut Real)]);
        let (idx, idy, idz) = (&p.inv_d[0], &p.inv_d[1], &p.inv_d[2]);
        let (bc_ylo, bc_zlo) = (p.bc_y[0], p.bc_z[0]);
        // (a 2-D plan has been updated above: the 3-D loops have nothing to do)
        (0..if twod.is_some() { 0 } else { nxl }).into_par_iter().for_each(|i| unsafe {
            let dxi = idx[i];
            for j in 0..ny {
                let dyi = idy[j];
                for k in 0..nz {
                    let s = (i + 1) * plane + j * nz + k;
                    let hx0 = hx.get(s);
                    let hy0 = hy.get(s);
                    let hz0 = hz.get(s);
                    let hy_xm = hy.get(s - plane);
                    let hz_xm = hz.get(s - plane);
                    let (hx_ym, hz_ym) = if j > 0 {
                        (hx.get(s - nz), hz.get(s - nz))
                    } else if bc_ylo == 1 {
                        (hx.get(s + (ny - 1) * nz), hz.get(s + (ny - 1) * nz))
                    } else if bc_ylo == 2 {
                        (-hx0, -hz0)
                    } else if bc_ylo == 3 {
                        let w = s + (ny - 1) * nz;
                        (bcy * hx.get(w) - bsy * ho[0].get(w), bcy * hz.get(w) - bsy * ho[2].get(w))
                    } else {
                        (0.0, 0.0)
                    };
                    let (hx_zm, hy_zm) = if k > 0 {
                        (hx.get(s - 1), hy.get(s - 1))
                    } else if bc_zlo == 1 {
                        (hx.get(s + (nz - 1)), hy.get(s + (nz - 1)))
                    } else if bc_zlo == 2 {
                        (-hx0, -hy0)
                    } else if bc_zlo == 3 {
                        let w = s + (nz - 1);
                        (bcz * hx.get(w) - bsz * ho[0].get(w), bcz * hy.get(w) - bsz * ho[1].get(w))
                    } else {
                        (0.0, 0.0)
                    };
                    let dzi = idz[k];
                    let cx = (hz0 - hz_ym) * dyi - (hy0 - hy_zm) * dzi;
                    let cy = (hx0 - hx_zm) * dzi - (hz0 - hz_xm) * dxi;
                    let cz = (hy0 - hy_xm) * dxi - (hx0 - hx_ym) * dyi;
                    match ca {
                        Some(a) => {
                            ex.set(s, a[0].get(s) * ex.get(s) + cb[0].get(s) * cx);
                            ey.set(s, a[1].get(s) * ey.get(s) + cb[1].get(s) * cy);
                            ez.set(s, a[2].get(s) * ez.get(s) + cb[2].get(s) * cz);
                        }
                        None => {
                            ex.add(s, cb[0].get(s) * cx);
                            ey.add(s, cb[1].get(s) * cy);
                            ez.add(s, cb[2].get(s) * cz);
                        }
                    }
                }
            }
        });
        for (si, slab) in p.pml.iter().enumerate().take(if twod.is_some() { 0 } else { usize::MAX }) {
            let a = slab.axis;
            let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
            let eb = Ptr(self.e[bax].as_mut_ptr());
            let ec = Ptr(self.e[cax].as_mut_ptr());
            let hb = Ptr(self.h[bax].as_ptr() as *mut Real);
            let hc = Ptr(self.h[cax].as_ptr() as *mut Real);
            let cbb = cb[bax];
            let cbc = cb[cax];
            let psib = Ptr(self.psi[si][0].as_mut_ptr());
            let psic = Ptr(self.psi[si][1].as_mut_ptr());
            let inv = &p.inv_d[a];
            let (lx, ly, lz) = (slab.size[0], slab.size[1], slab.size[2]);
            let stride = match a {
                0 => plane,
                1 => nz,
                _ => 1,
            };
            (0..lx).into_par_iter().for_each(|ii| unsafe {
                for jj in 0..ly {
                    for kk in 0..lz {
                        let t = (ii * ly + jj) * lz + kk;
                        let i = slab.start[0] + ii;
                        let j = slab.start[1] + jj;
                        let k = slab.start[2] + kk;
                        let s = ((i + 1) * ny + j) * nz + k;
                        let along = [i, j, k][a];
                        let q = along - slab.start[a];
                        let b = slab.e_b[q];
                        let aa = slab.e_a[q];
                        let km1 = slab.e_kinv[q] - 1.0;
                        let invv = inv[along];
                        let has_prev = a == 0 || along > 0;
                        let hc_p = if has_prev { hc.get(s - stride) } else { 0.0 };
                        let hb_p = if has_prev { hb.get(s - stride) } else { 0.0 };
                        let d_hc = (hc.get(s) - hc_p) * invv;
                        let d_hb = (hb.get(s) - hb_p) * invv;
                        let pb = b * psib.get(t) + aa * d_hc;
                        let pc = b * psic.get(t) + aa * d_hb;
                        psib.set(t, pb);
                        psic.set(t, pc);
                        eb.add(s, -cbb.get(s) * (km1 * d_hc + pb));
                        ec.add(s, cbc.get(s) * (km1 * d_hb + pc));
                    }
                }
            });
        }
        // dispersive step 2
        if let Some(d) = &self.plan.disp {
            for (ci, dc) in d.comps.iter().enumerate() {
                if twod.map(|t| !t.is_active(dc.comp)).unwrap_or(false) {
                    continue;
                }
                let f = &mut self.e[dc.comp.axis()];
                let dp = &self.disp[ci].2;
                for t in 0..dc.idx.len() {
                    f[dc.idx[t] as usize] -= dc.scale[t] * dp[t];
                }
            }
        }
        Ok(())
    }


    /// Running-DFT accumulation of the set currently in `e`/`h`.
    fn record_dft(&mut self, step: usize) {
        let (ny, nz) = (self.plan.ny(), self.plan.nz());
        // the samples of a box, gathered once per step and shared by its frequencies
        let mut vals: Vec<Real> = Vec::new();
        for (di, d) in self.plan.dft.iter().enumerate() {
            if step % d.stride != 0 {
                continue;
            }
            let nf = d.freqs.len();
            for (bi, bx) in d.boxes.iter().enumerate() {
                if !bx.is_local() || self.plan.twod.map(|t| !t.is_active(bx.comp)).unwrap_or(false) {
                    continue;
                }
                let f = if bx.comp.is_e() { &self.e[bx.comp.axis()] } else { &self.h[bx.comp.axis()] };
                let ph = if bx.comp.is_e() { &d.phasor_e } else { &d.phasor_h };
                let row = &ph[step * nf * 2..(step + 1) * nf * 2];
                let (re, im) = &mut self.dft[di].boxes[bi];
                let n = bx.lcells();
                let (lx, ly, lz) = (bx.lsize[0], bx.lsize[1], bx.lsize[2]);
                vals.clear();
                for ii in 0..lx {
                    for jj in 0..ly {
                        let s = ((bx.lstart[0] + ii + 1) * ny + bx.lstart[1] + jj) * nz + bx.lstart[2];
                        vals.extend_from_slice(&f[s..s + lz]);
                    }
                }
                let vals = &vals[..];
                let acc = |(fi, (rc, ic)): (usize, (&mut [Real], &mut [Real]))| {
                    let c = row[2 * fi];
                    let sn = row[2 * fi + 1];
                    for ((r, i_), v) in rc.iter_mut().zip(ic.iter_mut()).zip(vals.iter()) {
                        *r += *v * c;
                        *i_ += *v * sn;
                    }
                };
                if n * nf < 1 << 15 {
                    // a small box (a line or a small plane): handing it to the
                    // thread pool costs more than the sums themselves
                    re.chunks_mut(n).zip(im.chunks_mut(n)).enumerate().for_each(acc);
                } else {
                    re.par_chunks_mut(n).zip(im.par_chunks_mut(n)).enumerate().for_each(acc);
                }
            }
        }
    }
}

// ---- accessors used by `crate::autodiff` (gradient sessions, reverse mode) ----
impl CpuBackend {
    /// Mutable plan: sessions swap sources/monitors and rewrite the material
    /// coefficients of a design region between runs.
    pub fn plan_mut(&mut self) -> &mut Plan {
        &mut self.plan
    }
    /// A field component in storage layout (current set).
    pub fn field_ref(&self, c: Component) -> &[Real] {
        self.arr(c)
    }
    pub fn field_mut(&mut self, c: Component) -> &mut [Real] {
        self.arr_mut(c)
    }
    /// The plan with the E and H components (disjoint borrows).
    pub fn plan_and_fields(&mut self) -> (&Plan, &mut [Vec<Real>; 3], &mut [Vec<Real>; 3]) {
        (&self.plan, &mut self.e, &mut self.h)
    }
    /// Dispersive state per dispersive component: (P, P_prev, dP).
    pub fn disp_state(&self) -> &Vec<(Vec<Real>, Vec<Real>, Vec<Real>)> {
        &self.disp
    }
    pub fn disp_state_mut(&mut self) -> &mut Vec<(Vec<Real>, Vec<Real>, Vec<Real>)> {
        &mut self.disp
    }
    /// Raw monitor accumulators (same layout as [`Backend::download`]).
    pub fn dft_data(&self) -> &Vec<RawDft> {
        &self.dft
    }
    pub fn dft_data_mut(&mut self) -> &mut Vec<RawDft> {
        &mut self.dft
    }
    pub fn time_data(&self) -> &Vec<RawTime> {
        &self.time
    }
    /// Zero the fields, the auxiliary state and the monitors, re-sizing the
    /// monitor buffers for the plan's current monitor set (non-Bloch plans).
    pub fn reset(&mut self) -> Result<()> {
        if self.plan.bloch {
            anyhow::bail!("reset is not supported for Bloch (complex-field) plans");
        }
        for a in 0..3 {
            self.e[a].fill(0.0);
            self.h[a].fill(0.0);
            self.t_prev[a].fill(0.0);
            self.t_en[a].fill(0.0);
        }
        for p in self.psi.iter_mut() {
            for q in p.iter_mut() {
                q.fill(0.0);
            }
        }
        for d in self.disp.iter_mut() {
            d.0.fill(0.0);
            d.1.fill(0.0);
            d.2.fill(0.0);
        }
        for v in self.nl_prev.iter_mut() {
            v.fill(0.0);
        }
        if let Some(gp) = self.plan.nonlin.as_ref().and_then(|n| n.gain.as_ref()) {
            for (c, (st, pop)) in gp.comps.iter().zip(self.gain.iter_mut()) {
                for v in st.iter_mut() {
                    v.fill(0.0);
                }
                let n = c.idx.len();
                for (t, &r) in c.row.iter().enumerate() {
                    for l in 0..4 {
                        pop[l * n + t] = gp.init[4 * r as usize + l];
                    }
                }
            }
        }
        for (a, b, c) in self.raman.iter_mut() {
            a.fill(0.0);
            b.fill(0.0);
            c.fill(0.0);
        }
        for t in self.t_os.iter_mut().chain(self.t_gy.iter_mut()) {
            t.0.fill(0.0);
            t.1.fill(0.0);
        }
        self.dft = self
            .plan
            .dft
            .iter()
            .map(|d| RawDft {
                boxes: d.boxes.iter().map(|b| (vec![0 as Real; b.lcells() * d.freqs.len()], vec![0 as Real; b.lcells() * d.freqs.len()])).collect(),
            })
            .collect();
        self.time = self.plan.time.iter().map(|t| RawTime { boxes: vec![Vec::new(); t.boxes.len()], steps: vec![] }).collect();
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        match &self.plan.twod {
            Some(td) => format!("cpu ({} threads, 2-D {})", rayon::current_num_threads(), td.label()),
            None => format!("cpu ({} threads)", rayon::current_num_threads()),
        }
    }
    fn plan(&self) -> &Plan {
        &self.plan
    }
    fn on_thread_pool(&self) -> bool {
        true
    }

    fn fill_ghost_e(&mut self) -> Result<()> {
        let plane = self.plan.plane();
        let nxl = self.plan.nxl;
        let rule = self.plan.ghost_e_hi.clone();
        for a in [1usize, 2] {
            let f = &mut self.e[a];
            let ghost = (nxl + 1) * plane;
            match rule {
                GhostRule::Zero | GhostRule::Neighbor => {}
                GhostRule::Periodic { cos, sin } => {
                    if let Some(c) = self.cplx.as_mut() {
                        // Bloch: ghost = first plane * exp(i k L), mixing the parts
                        let fi = &mut c.e[a];
                        let (cs, sn) = (cos as Real, sin as Real);
                        for t in 0..plane {
                            let (r, i) = (f[plane + t], fi[plane + t]);
                            f[ghost + t] = cs * r - sn * i;
                            fi[ghost + t] = cs * i + sn * r;
                        }
                        continue;
                    }
                    let (lo, hi) = f.split_at_mut(ghost);
                    for (d, s) in hi[..plane].iter_mut().zip(lo[plane..2 * plane].iter()) {
                        *d = cos as Real * *s;
                    }
                }
                GhostRule::Mirror { factor } => {
                    let (lo, hi) = f.split_at_mut(ghost);
                    for (d, s) in hi[..plane].iter_mut().zip(lo[nxl * plane..(nxl + 1) * plane].iter()) {
                        *d = factor * *s;
                    }
                }
            }
        }
        Ok(())
    }

    fn fill_ghost_h(&mut self) -> Result<()> {
        let plane = self.plan.plane();
        let nxl = self.plan.nxl;
        let rule = self.plan.ghost_h_lo.clone();
        for a in [1usize, 2] {
            let f = &mut self.h[a];
            match rule {
                GhostRule::Zero | GhostRule::Neighbor => {}
                GhostRule::Periodic { cos, sin } => {
                    if let Some(c) = self.cplx.as_mut() {
                        // Bloch: ghost = last plane * exp(-i k L) ((cos, sin) already of -kL)
                        let fi = &mut c.h[a];
                        let (cs, sn) = (cos as Real, sin as Real);
                        let src = nxl * plane;
                        for t in 0..plane {
                            let (r, i) = (f[src + t], fi[src + t]);
                            f[t] = cs * r - sn * i;
                            fi[t] = cs * i + sn * r;
                        }
                        continue;
                    }
                    let (lo, hi) = f.split_at_mut(plane);
                    let src = &hi[(nxl - 1) * plane..nxl * plane];
                    for (d, s) in lo.iter_mut().zip(src.iter()) {
                        *d = cos as Real * *s;
                    }
                }
                GhostRule::Mirror { factor } => {
                    let (lo, hi) = f.split_at_mut(plane);
                    for (d, s) in lo.iter_mut().zip(hi[..plane].iter()) {
                        *d = factor * *s;
                    }
                }
            }
        }
        Ok(())
    }

    fn update_h(&mut self) -> Result<()> {
        self.update_h_core()?;
        if self.cplx.is_some() {
            self.swap_sets();
            self.update_h_core()?;
            self.swap_sets();
        }
        Ok(())
    }

    fn update_e(&mut self) -> Result<()> {
        self.nl_save();
        self.raman_save();
        self.gain_pre();
        self.tensor_save();
        self.update_e_core()?;
        self.nl_apply();
        self.raman_apply();
        self.gain_post();
        self.tensor_dd();
        if self.cplx.is_some() {
            self.swap_sets();
            self.update_e_core()?;
            self.swap_sets();
        }
        Ok(())
    }

    fn inject_h(&mut self, step: usize) -> Result<()> {
        let inj = std::mem::take(&mut self.plan.inj_h);
        for j in &inj {
            apply_injection(self.arr_mut(j.comp), j, step, false);
            if let Some(c) = self.cplx.as_mut() {
                let f = if j.comp.is_e() { &mut c.e[j.comp.axis()] } else { &mut c.h[j.comp.axis()] };
                apply_injection(f, j, step, true);
            }
        }
        self.plan.inj_h = inj;
        Ok(())
    }

    fn inject_e(&mut self, step: usize) -> Result<()> {
        let inj = std::mem::take(&mut self.plan.inj_e);
        for j in &inj {
            apply_injection(self.arr_mut(j.comp), j, step, false);
            if let Some(c) = self.cplx.as_mut() {
                let f = if j.comp.is_e() { &mut c.e[j.comp.axis()] } else { &mut c.h[j.comp.axis()] };
                apply_injection(f, j, step, true);
            }
        }
        self.plan.inj_e = inj;
        Ok(())
    }

    fn record(&mut self, step: usize) -> Result<()> {
        if self.cplx.is_some() {
            self.swap_sets();
            self.record_dft(step);
            self.swap_sets();
        }
        let (ny, nz) = (self.plan.ny(), self.plan.nz());
        self.record_dft(step);
        for (ti, tp) in self.plan.time.iter().enumerate() {
            if step < tp.start_step || (step - tp.start_step) % tp.interval != 0 {
                continue;
            }
            self.time[ti].steps.push(step);
            for (bi, bx) in tp.boxes.iter().enumerate() {
                if !bx.is_local() {
                    continue;
                }
                let f = if bx.comp.is_e() { &self.e[bx.comp.axis()] } else { &self.h[bx.comp.axis()] };
                let (ly, lz) = (bx.lsize[1], bx.lsize[2]);
                let out = &mut self.time[ti].boxes[bi];
                for t in 0..bx.lcells() {
                    let kk = t % lz;
                    let jj = (t / lz) % ly;
                    let ii = t / (lz * ly);
                    let s = ((bx.lstart[0] + ii + 1) * ny + bx.lstart[1] + jj) * nz + bx.lstart[2] + kk;
                    out.push(f[s]);
                }
            }
        }
        Ok(())
    }

    fn energy(&mut self) -> Result<f64> {
        let plane = self.plan.plane();
        let nxl = self.plan.nxl;
        let mut tot = 0f64;
        let mut sets: Vec<&Vec<Real>> = self.e.iter().chain(self.h.iter()).collect();
        if let Some(c) = &self.cplx {
            sets.extend(c.e.iter().chain(c.h.iter()));
        }
        for f in sets {
            tot += f[plane..(nxl + 1) * plane].par_iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
        }
        Ok(tot)
    }

    fn download(&mut self) -> Result<RawResults> {
        let mut dft = self.dft.clone();
        if let Some(c) = &self.cplx {
            // complex time-domain field: DFT(re + i im) = DFT(re) + i DFT(im)
            for (d, di) in dft.iter_mut().zip(c.dft.iter()) {
                for ((re, im), (ire, iim)) in d.boxes.iter_mut().zip(di.boxes.iter()) {
                    for t in 0..re.len() {
                        let (a, b) = (re[t], im[t]);
                        re[t] = a - iim[t];
                        im[t] = b + ire[t];
                    }
                }
            }
        }
        Ok(RawResults { dft, time: self.time.clone() })
    }

    fn read_plane(&mut self, comp: Component, side: Side, out: &mut [Real]) -> Result<()> {
        let plane = self.plan.plane();
        let nxl = self.plan.nxl;
        let pi = match side {
            Side::Lo => 1,
            Side::Hi => nxl,
        };
        out.copy_from_slice(&self.arr(comp)[pi * plane..(pi + 1) * plane]);
        Ok(())
    }

    fn write_ghost(&mut self, comp: Component, side: Side, data: &[Real]) -> Result<()> {
        let plane = self.plan.plane();
        let nxl = self.plan.nxl;
        let pi = match side {
            Side::Lo => 0,
            Side::Hi => nxl + 1,
        };
        self.arr_mut(comp)[pi * plane..(pi + 1) * plane].copy_from_slice(data);
        Ok(())
    }

    fn gain_levels(&mut self) -> Result<Option<[f64; 4]>> {
        let Some(gp) = self.plan.nonlin.as_ref().and_then(|n| n.gain.as_ref()) else { return Ok(None) };
        let mut out = [0f64; 4];
        for (c, (_, pop)) in gp.comps.iter().zip(self.gain.iter()) {
            let n = c.idx.len();
            for (l, o) in out.iter_mut().enumerate() {
                *o += (0..n).map(|t| pop[l * n + t] * c.vol[t]).sum::<f64>();
            }
        }
        Ok(Some(out))
    }
    fn finish_e(&mut self) -> Result<()> {
        self.tensor_apply();
        self.gyro_apply();
        Ok(())
    }

    /// E^{n+1} -> E^n (needs H^{n+1/2}; ghosts filled here).
    fn reverse_e(&mut self) -> Result<()> {
        self.reversible()?;
        let cb_f = self.plan.cb.clone();
        let ca_f = self.plan.ca.clone();
        for a in 0..3 {
            for (i, v) in self.plan.cb[a].iter_mut().enumerate() {
                let ca = ca_f.as_ref().map(|c| c[a][i]).unwrap_or(1.0);
                *v = -*v / ca;
            }
        }
        if let Some(ca) = self.plan.ca.as_mut() {
            for a in 0..3 {
                for v in ca[a].iter_mut() {
                    *v = 1.0 / *v;
                }
            }
        }
        let r = self.fill_ghost_h().and_then(|_| self.update_e());
        self.plan.cb = cb_f;
        self.plan.ca = ca_f;
        r
    }

    /// H^{n+1/2} -> H^{n-1/2} (needs E^n; ghosts filled here).
    fn reverse_h(&mut self) -> Result<()> {
        self.reversible()?;
        let ch_f = self.plan.ch.clone();
        let hl_f = self.plan.hloss.clone();
        let chs = self.plan.ch_scalar;
        if let Some(ch) = self.plan.ch.as_mut() {
            for a in 0..3 {
                for v in ch[a].iter_mut() {
                    *v = -*v;
                }
            }
        }
        // H(n-1/2) = (H(n+1/2) + ch curl E) / da
        if let (Some(ch), Some(hl)) = (self.plan.ch.as_mut(), self.plan.hloss.as_mut()) {
            reverse_hloss(ch, hl);
        }
        self.plan.ch_scalar = -chs;
        let r = self.fill_ghost_e().and_then(|_| self.update_h());
        self.plan.ch = ch_f;
        self.plan.hloss = hl_f;
        self.plan.ch_scalar = chs;
        r
    }

    fn set_field(&mut self, comp: Component, data: &[Real]) -> Result<()> {
        // 2-D: a field written from outside puts its set in use
        if let Some(td) = self.plan.twod.as_mut() {
            td.drive(comp);
        }
        self.arr_mut(comp).copy_from_slice(data);
        Ok(())
    }


    fn tensor_read(&mut self, comp: Component, side: Side, which: usize, out: &mut [Real]) -> Result<()> {
        if let Some(t) = &self.plan.tensor {
            let plane = self.plan.plane();
            let pi = if side == Side::Lo { 1 } else { self.plan.nxl };
            let o = (pi - t.p0) * plane;
            let buf = if which == 0 { &self.t_prev } else { &self.t_en };
            out.copy_from_slice(&buf[comp.axis()][o..o + plane]);
        }
        Ok(())
    }

    fn tensor_write_ghost(&mut self, comp: Component, side: Side, which: usize, data: &[Real]) -> Result<()> {
        if let Some(t) = &self.plan.tensor {
            let plane = self.plan.plane();
            let pi = if side == Side::Lo { 0 } else { self.plan.nxl + 1 };
            let o = (pi - t.p0) * plane;
            let buf = if which == 0 { &mut self.t_prev } else { &mut self.t_en };
            buf[comp.axis()][o..o + plane].copy_from_slice(data);
        }
        Ok(())
    }

    fn select_set(&mut self, imag: bool) -> Result<()> {
        if self.cplx.is_some() && imag != self.imag_active {
            self.swap_sets();
        }
        Ok(())
    }

    fn field(&mut self, comp: Component) -> Result<Vec<Real>> {
        Ok(self.arr(comp).clone())
    }
}

/// Apply an injection to the real part (`imag = false`) or, for complex
/// sources in Bloch mode, to the imaginary part of the complex field:
/// re += Re(c A) = c_re a_re + c_im a_im ; im += Im(c A) = c_re a_im - c_im a_re
/// (c_im stores -Im c). Real-valued sources only drive the real part.
/// E^{n+1} of a chi2/chi3 cell from E^n (`e`) and the linear update (`elin`):
/// solves u + c2 (2 e u + u^2) + c3 (3 e^2 u + 3 e u^2 + u^3) = elin - e.
pub fn nl_solve(e: Real, elin: Real, a: Real, b: Real) -> Real {
    let d = elin - e;
    let mut u = d / (1.0 + 2.0 * a * e + 3.0 * b * e * e);
    for _ in 0..4 {
        let f = u + a * (2.0 * e * u + u * u) + b * (3.0 * e * e * u + 3.0 * e * u * u + u * u * u) - d;
        let fp = 1.0 + a * (2.0 * e + 2.0 * u) + b * (3.0 * e * e + 6.0 * e * u + 3.0 * u * u);
        u -= f / fp;
    }
    e + u
}

/// E^{n+1} of a chi3 cell with a Raman part, from E^n (`e`), the linear
/// update (`elin`), c2 = chi2/eps, c3 = (1 - f) chi3/eps, cr = f chi3/eps and
/// Q^{n+1} (`q1`), Q^n (`q0`). With u = E^{n+1} - E^n and d = elin - e,
/// eps E^{n+1} + P^{n+1}/eps0 = eps elin + P^n/eps0 reads
///   u (1 + cr q1) + c2 (2 e u + u^2) + c3 (3 e^2 u + 3 e u^2 + u^3) = d - cr e (q1 - q0),
/// solved by Newton's method from the linearised guess (four iterations, as
/// the instantaneous step).
pub fn raman_solve(e: Real, elin: Real, a: Real, b: Real, cr: Real, q1: Real, q0: Real) -> Real {
    let d = elin - e - cr * e * (q1 - q0);
    let l = 1.0 + cr * q1;
    let mut u = d / (l + 2.0 * a * e + 3.0 * b * e * e);
    for _ in 0..4 {
        let f = u * l + a * (2.0 * e * u + u * u) + b * (3.0 * e * e * u + 3.0 * e * u * u + u * u * u) - d;
        let fp = l + a * (2.0 * e + 2.0 * u) + b * (3.0 * e * e + 6.0 * e * u + 3.0 * u * u);
        u -= f / fp;
    }
    e + u
}
/// Population update of one gain cell (`pop` = `[level][cell]` of `n`
/// cells, cell `t`, table row `k`): `work` = (E^n + E^{n+1})/2 (P^{n+1} - P^n)
/// with P / eps0, so k[3] work = eps0 work / (hbar omega) atoms per volume go
/// from level 1 to level 2; decays and pump as explicit fluxes. The `gain_post`
/// kernel does the same arithmetic.
pub fn gain_rates(pop: &mut [f64], n: usize, t: usize, k: &[f64], work: f64) {
    let (n0, n1, n2, n3) = (pop[t], pop[n + t], pop[2 * n + t], pop[3 * n + t]);
    let up = k[3] * work;
    let (f32_, f21, f10, fp) = (k[4] * n3, k[5] * n2, k[6] * n1, k[7] * n0);
    pop[t] = n0 + f10 - fp;
    pop[n + t] = n1 + f21 - f10 - up;
    pop[2 * n + t] = n2 + f32_ - f21 + up;
    pop[3 * n + t] = n3 + fp - f32_;
}
fn apply_injection(f: &mut [Real], j: &crate::plan::Injection, step: usize, imag: bool) {
    let row = j.row(step);
    match &j.coef_im {
        None => {
            if imag {
                return;
            }
            for e in 0..j.idx.len() {
                let g = if j.grouped() { j.group[e] as usize } else { 0 };
                f[j.idx[e] as usize] += j.coef[e] * j.amps[row + g];
            }
        }
        Some(ci) => {
            for e in 0..j.idx.len() {
                let g = if j.grouped() { j.group[e] as usize } else { 0 };
                let ar = j.amps[row + 2 * g];
                let ai = j.amps[row + 2 * g + 1];
                f[j.idx[e] as usize] += if imag { j.coef[e] * ai - ci[e] * ar } else { j.coef[e] * ar + ci[e] * ai };
            }
        }
    }
}

/// Turn the (already negated) H coefficients of the lossy cells into those
/// of the reversed update: `ch -> ch / da`, `da -> 1 / da`, so that the
/// update `H *= da; H -= ch curl E` computes `(H + ch curl E) / da`.
pub fn reverse_hloss(ch: &mut [Vec<Real>; 3], hl: &mut [crate::plan::HLoss; 3]) {
    for a in 0..3 {
        for (&s, d) in hl[a].idx.iter().zip(hl[a].da.iter_mut()) {
            ch[a][s as usize] /= *d;
            *d = 1.0 / *d;
        }
    }
}
