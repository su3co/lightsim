//! 2-D update of the CPU backend (plans with [`crate::plan::TwoD`]).
//!
//! A grid that is one periodic cell thick along an axis `w` carries fields
//! that do not depend on `w`, and Maxwell's equations split into
//!
//! * the TE set (E_u, E_v, H_w): E in the plane, H along the invariant axis,
//! * the TM set (E_w, H_u, H_v): E along the invariant axis, H in the plane,
//!
//! with `(u, v)` the in-plane axes of [`crate::spec::plane_of`]. The loops
//! here update only the sets in use, row by row along `v` (the contiguous
//! storage axis), without the terms of the general update that vanish. They
//! evaluate the remaining terms in the same order as the 3-D loops of
//! [`super::cpu`], so both give the same fields bit for bit (up to the sign
//! of zeros). CPML, dispersive poles and chi2/chi3 corrections stay in
//! `cpu.rs`; they skip the components of an unused set.
//!
//! Storage: `s = base + u * nv + v` with `base = ny * nz` (the low ghost
//! plane of x). When `u` is x its neighbours across the domain faces come
//! from the ghost planes, as in 3-D; the other axes apply their boundary
//! code (0 zero, 1 periodic, 2 mirror, 3 Bloch).

use crate::Real;
use crate::plan::{Plan, TwoD};
use rayon::prelude::*;

/// Rows per parallel task: enough cells that a task outweighs its scheduling.
fn rows_per_task(nv: usize) -> usize {
    (32_768 / nv.max(1)).max(1)
}

/// Loops over fewer cells than this run on the calling thread: handing a
/// loop to the thread pool costs more than a small grid's update
/// (`LIGHTSIM_2D_SERIAL_CELLS` overrides the default, for measurements).
fn serial_cells() -> usize {
    static CELLS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELLS.get_or_init(|| std::env::var("LIGHTSIM_2D_SERIAL_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(48_000))
}

/// Run `body(row index, row)` over the rows (`nv` cells each) of `a`, on the
/// pool when the loop is large enough.
fn rows1<F: Fn(usize, &mut [Real]) + Sync + Send>(a: &mut [Real], nv: usize, body: F) {
    if a.len() < serial_cells() {
        a.chunks_mut(nv).enumerate().for_each(|(u, r)| body(u, r));
    } else {
        a.par_chunks_mut(nv).with_min_len(rows_per_task(nv)).enumerate().for_each(|(u, r)| body(u, r));
    }
}

/// As [`rows1`] over the rows of two arrays.
fn rows2<F: Fn(usize, &mut [Real], &mut [Real]) + Sync + Send>(a: &mut [Real], b: &mut [Real], nv: usize, body: F) {
    if a.len() < serial_cells() {
        a.chunks_mut(nv).zip(b.chunks_mut(nv)).enumerate().for_each(|(u, (r, q))| body(u, r, q));
    } else {
        a.par_chunks_mut(nv).zip(b.par_chunks_mut(nv)).with_min_len(rows_per_task(nv)).enumerate().for_each(|(u, (r, q))| body(u, r, q));
    }
}

/// Layout of a 2-D plan.
#[derive(Clone, Copy)]
struct Lay {
    u: usize,
    v: usize,
    w: usize,
    nu: usize,
    nv: usize,
    base: usize,
    sg: Real,
    /// Boundary codes `[lo, hi]` along u (`None`: x, ghost planes) and v.
    bc_u: Option<[u32; 2]>,
    bc_v: [u32; 2],
}

impl Lay {
    fn new(plan: &Plan, td: TwoD) -> Lay {
        let (u, v, sg) = td.plane();
        let bc = |a: usize| if a == 1 { plan.bc_y } else { plan.bc_z };
        Lay {
            u,
            v,
            w: td.axis,
            nu: if u == 0 { plan.nxl } else { plan.n[u] },
            nv: plan.n[v],
            base: plan.plane(),
            sg,
            bc_u: if u == 0 { None } else { Some(bc(u)) },
            bc_v: bc(v),
        }
    }
    fn row(&self, u: usize) -> std::ops::Range<usize> {
        let s0 = self.base + u * self.nv;
        s0..s0 + self.nv
    }
}

/// Bloch phase of an axis for the field set being updated: (cos, signed sin).
fn phase(plan: &Plan, bsign: [Real; 3], a: usize) -> (Real, Real) {
    (plan.bloch_phase[a].0 as Real, bsign[a])
}

/// The row of `f` one step up along u from row `u` (E neighbours of the H
/// update): the next row, the high ghost plane, or the boundary rule.
fn row_up<'a>(l: &Lay, f: &'a [Real], fo: Option<&'a [Real]>, u: usize, ph: (Real, Real), tmp: &'a mut Vec<Real>) -> &'a [Real] {
    let nv = l.nv;
    if u + 1 < l.nu || l.bc_u.is_none() {
        let s = l.base + (u + 1) * nv;
        return &f[s..s + nv];
    }
    let first = &f[l.row(0)];
    match l.bc_u.unwrap()[1] {
        1 => first,
        2 => &f[l.row(u)],
        3 => {
            let o = &fo.expect("Bloch partner fields")[l.row(0)];
            tmp.clear();
            tmp.extend(first.iter().zip(o).map(|(a, b)| ph.0 * *a + ph.1 * *b));
            tmp
        }
        _ => {
            tmp.clear();
            tmp.resize(nv, 0.0);
            tmp
        }
    }
}

/// The row of `f` one step down along u from row `u` (H neighbours of the E
/// update): the previous row, the low ghost plane, or the boundary rule.
fn row_down<'a>(l: &Lay, f: &'a [Real], fo: Option<&'a [Real]>, u: usize, ph: (Real, Real), tmp: &'a mut Vec<Real>) -> &'a [Real] {
    let nv = l.nv;
    if u > 0 || l.bc_u.is_none() {
        let s = l.base + u * nv - nv;
        return &f[s..s + nv];
    }
    let last = &f[l.row(l.nu - 1)];
    match l.bc_u.unwrap()[0] {
        1 => last,
        2 => {
            tmp.clear();
            tmp.extend(f[l.row(0)].iter().map(|a| -*a));
            tmp
        }
        3 => {
            let o = &fo.expect("Bloch partner fields")[l.row(l.nu - 1)];
            tmp.clear();
            tmp.extend(last.iter().zip(o).map(|(a, b)| ph.0 * *a - ph.1 * *b));
            tmp
        }
        _ => {
            tmp.clear();
            tmp.resize(nv, 0.0);
            tmp
        }
    }
}

/// E value one step beyond the high end of a row along v (H update).
fn beyond_hi(code: u32, row: &[Real], other: Option<&[Real]>, ph: (Real, Real)) -> Real {
    let n = row.len();
    match code {
        1 => row[0],
        2 => row[n - 1],
        3 => ph.0 * row[0] + ph.1 * other.expect("Bloch partner fields")[0],
        _ => 0.0,
    }
}

/// H value one step below the low end of a row along v (E update).
fn beyond_lo(code: u32, row: &[Real], other: Option<&[Real]>, ph: (Real, Real)) -> Real {
    let n = row.len();
    match code {
        1 => row[n - 1],
        2 => -row[0],
        3 => ph.0 * row[n - 1] - ph.1 * other.expect("Bloch partner fields")[n - 1],
        _ => 0.0,
    }
}

/// H_w -= ch sg (d_u E_v - d_v E_u) over one row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn h_te_row_imp(hw: &mut [Real], eu: &[Real], ev: &[Real], ev_up: &[Real], eu_end: Real, ch: Option<&[Real]>, chs: Real, dui: Real, dvi: &[Real], sg: Real) {
    let n = hw.len();
    let m = n - 1;
    {
        let (hw, eu0, eu1, ev, evu, dvi) = (&mut hw[..m], &eu[..m], &eu[1..n], &ev[..m], &ev_up[..m], &dvi[..m]);
        match ch {
            Some(c) => {
                let c = &c[..m];
                for q in 0..m {
                    let cw = sg * ((evu[q] - ev[q]) * dui - (eu1[q] - eu0[q]) * dvi[q]);
                    hw[q] -= c[q] * cw;
                }
            }
            None => {
                for q in 0..m {
                    let cw = sg * ((evu[q] - ev[q]) * dui - (eu1[q] - eu0[q]) * dvi[q]);
                    hw[q] -= chs * cw;
                }
            }
        }
    }
    let cw = sg * ((ev_up[m] - ev[m]) * dui - (eu_end - eu[m]) * dvi[m]);
    hw[m] -= ch.map(|c| c[m]).unwrap_or(chs) * cw;
}

/// H_u -= ch sg d_v E_w, H_v += ch sg d_u E_w over one row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn h_tm_row_imp(hu: &mut [Real], hv: &mut [Real], ew: &[Real], ew_up: &[Real], ew_end: Real, chu: Option<&[Real]>, chv: Option<&[Real]>, chs: Real, dui: Real, dvi: &[Real], sg: Real) {
    let n = hu.len();
    let m = n - 1;
    let nsg = -sg;
    {
        let (hu, hv, ew0, ew1, ewu, dvi) = (&mut hu[..m], &mut hv[..m], &ew[..m], &ew[1..n], &ew_up[..m], &dvi[..m]);
        match (chu, chv) {
            (Some(cu), Some(cv)) => {
                let (cu, cv) = (&cu[..m], &cv[..m]);
                for q in 0..m {
                    let a = sg * ((ew1[q] - ew0[q]) * dvi[q]);
                    let b = nsg * ((ewu[q] - ew0[q]) * dui);
                    hu[q] -= cu[q] * a;
                    hv[q] -= cv[q] * b;
                }
            }
            _ => {
                for q in 0..m {
                    let a = sg * ((ew1[q] - ew0[q]) * dvi[q]);
                    let b = nsg * ((ewu[q] - ew0[q]) * dui);
                    hu[q] -= chs * a;
                    hv[q] -= chs * b;
                }
            }
        }
    }
    let a = sg * ((ew_end - ew[m]) * dvi[m]);
    let b = nsg * ((ew_up[m] - ew[m]) * dui);
    hu[m] -= chu.map(|c| c[m]).unwrap_or(chs) * a;
    hv[m] -= chv.map(|c| c[m]).unwrap_or(chs) * b;
}

/// E_u = ca E_u + cb sg d_v H_w, E_v = ca E_v - cb sg d_u H_w over one row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn e_te_row_imp(eu: &mut [Real], ev: &mut [Real], hw: &[Real], hw_dn: &[Real], hw_start: Real, cbu: &[Real], cbv: &[Real], cau: Option<&[Real]>, cav: Option<&[Real]>, dui: Real, dvi: &[Real], sg: Real) {
    let n = eu.len();
    let nsg = -sg;
    // first cell: the neighbour below along v comes from the boundary rule
    {
        let a = sg * ((hw[0] - hw_start) * dvi[0]);
        let b = nsg * ((hw[0] - hw_dn[0]) * dui);
        match (cau, cav) {
            (Some(au), Some(av)) => {
                eu[0] = au[0] * eu[0] + cbu[0] * a;
                ev[0] = av[0] * ev[0] + cbv[0] * b;
            }
            _ => {
                eu[0] += cbu[0] * a;
                ev[0] += cbv[0] * b;
            }
        }
    }
    let m = n - 1;
    let (eu, ev, h1, h0, hd, cbu, cbv, dvi) = (&mut eu[1..n], &mut ev[1..n], &hw[1..n], &hw[..m], &hw_dn[1..n], &cbu[1..n], &cbv[1..n], &dvi[1..n]);
    match (cau, cav) {
        (Some(au), Some(av)) => {
            let (au, av) = (&au[1..n], &av[1..n]);
            for q in 0..m {
                let a = sg * ((h1[q] - h0[q]) * dvi[q]);
                let b = nsg * ((h1[q] - hd[q]) * dui);
                eu[q] = au[q] * eu[q] + cbu[q] * a;
                ev[q] = av[q] * ev[q] + cbv[q] * b;
            }
        }
        _ => {
            for q in 0..m {
                let a = sg * ((h1[q] - h0[q]) * dvi[q]);
                let b = nsg * ((h1[q] - hd[q]) * dui);
                eu[q] += cbu[q] * a;
                ev[q] += cbv[q] * b;
            }
        }
    }
}

/// E_w = ca E_w + cb sg (d_u H_v - d_v H_u) over one row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn e_tm_row_imp(ew: &mut [Real], hu: &[Real], hv: &[Real], hv_dn: &[Real], hu_start: Real, cbw: &[Real], caw: Option<&[Real]>, dui: Real, dvi: &[Real], sg: Real) {
    let n = ew.len();
    {
        let c = sg * ((hv[0] - hv_dn[0]) * dui - (hu[0] - hu_start) * dvi[0]);
        match caw {
            Some(a) => ew[0] = a[0] * ew[0] + cbw[0] * c,
            None => ew[0] += cbw[0] * c,
        }
    }
    let m = n - 1;
    let (ew, v1, vd, u1, u0, cbw, dvi) = (&mut ew[1..n], &hv[1..n], &hv_dn[1..n], &hu[1..n], &hu[..m], &cbw[1..n], &dvi[1..n]);
    match caw {
        Some(a) => {
            let a = &a[1..n];
            for q in 0..m {
                let c = sg * ((v1[q] - vd[q]) * dui - (u1[q] - u0[q]) * dvi[q]);
                ew[q] = a[q] * ew[q] + cbw[q] * c;
            }
        }
        None => {
            for q in 0..m {
                let c = sg * ((v1[q] - vd[q]) * dui - (u1[q] - u0[q]) * dvi[q]);
                ew[q] += cbw[q] * c;
            }
        }
    }
}

/// Row kernels with run-time selection of the vector width. The row loops
/// are plain element-wise arithmetic, which the compiler vectorises for
/// whatever instruction set the function is compiled for: the baseline
/// (SSE2 on x86-64, NEON on AArch64), AVX2, or AVX-512 when the processor
/// has them. No fused multiply-add is enabled, so every variant gives the
/// same bits.
macro_rules! dispatch {
    ($name:ident, $imp:ident, ($($arg:ident: $ty:ty),* $(,)?)) => {
        #[allow(clippy::too_many_arguments)]
        #[inline(never)]
        fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx512f")]
                unsafe fn wide($($arg: $ty),*) {
                    $imp($($arg),*)
                }
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) {
                    $imp($($arg),*)
                }
                if std::arch::is_x86_feature_detected!("avx512f") {
                    return unsafe { wide($($arg),*) };
                }
                if std::arch::is_x86_feature_detected!("avx2") {
                    return unsafe { avx2($($arg),*) };
                }
            }
            $imp($($arg),*)
        }
    };
}

pub(crate) use dispatch;

dispatch!(h_te_row, h_te_row_imp, (hw: &mut [Real], eu: &[Real], ev: &[Real], ev_up: &[Real], eu_end: Real, ch: Option<&[Real]>, chs: Real, dui: Real, dvi: &[Real], sg: Real));
dispatch!(h_tm_row, h_tm_row_imp, (hu: &mut [Real], hv: &mut [Real], ew: &[Real], ew_up: &[Real], ew_end: Real, chu: Option<&[Real]>, chv: Option<&[Real]>, chs: Real, dui: Real, dvi: &[Real], sg: Real));
dispatch!(e_te_row, e_te_row_imp, (eu: &mut [Real], ev: &mut [Real], hw: &[Real], hw_dn: &[Real], hw_start: Real, cbu: &[Real], cbv: &[Real], cau: Option<&[Real]>, cav: Option<&[Real]>, dui: Real, dvi: &[Real], sg: Real));
dispatch!(e_tm_row, e_tm_row_imp, (ew: &mut [Real], hu: &[Real], hv: &[Real], hv_dn: &[Real], hu_start: Real, cbw: &[Real], caw: Option<&[Real]>, dui: Real, dvi: &[Real], sg: Real));

/// Two distinct elements of a 3-array, mutably.
fn two_mut<T>(v: &mut [T; 3], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b);
    if a < b {
        let (lo, hi) = v.split_at_mut(b);
        (&mut lo[a], &mut hi[0])
    } else {
        let (lo, hi) = v.split_at_mut(a);
        (&mut hi[0], &mut lo[b])
    }
}

/// H update (without CPML) of the sets in use. `eo` holds the other part of
/// complex fields (Bloch boundaries), `bsign` the sign of its sine term.
pub(crate) fn update_h(plan: &Plan, td: TwoD, e: &[Vec<Real>; 3], h: &mut [Vec<Real>; 3], eo: Option<&[Vec<Real>; 3]>, bsign: [Real; 3]) {
    let l = Lay::new(plan, td);
    let (nu, nv) = (l.nu, l.nv);
    if nu == 0 || nv == 0 {
        return;
    }
    let chs = plan.ch_scalar;
    let ch = plan.ch.as_ref();
    let (ipu, ipv) = (&plan.inv_p[l.u], &plan.inv_p[l.v]);
    let (pu, pv) = (phase(plan, bsign, l.u), phase(plan, bsign, l.v));
    let interior = l.base..l.base + nu * nv;
    if td.active[0] {
        let (eu, ev) = (&e[l.u], &e[l.v]);
        let (euo, evo) = (eo.map(|o| &o[l.u][..]), eo.map(|o| &o[l.v][..]));
        let chw = ch.map(|c| &c[l.w][..]);
        rows1(&mut h[l.w][interior.clone()], nv, |u, hw| {
            let r = l.row(u);
            let mut tmp = Vec::new();
            let ev_up = row_up(&l, ev, evo, u, pu, &mut tmp);
            let eu_row = &eu[r.clone()];
            let eu_end = beyond_hi(l.bc_v[1], eu_row, euo.map(|o| &o[r.clone()]), pv);
            h_te_row(hw, eu_row, &ev[r.clone()], ev_up, eu_end, chw.map(|c| &c[r.clone()]), chs, ipu[u], ipv, l.sg);
        });
    }
    if td.active[1] {
        let ew = &e[l.w];
        let ewo = eo.map(|o| &o[l.w][..]);
        let (chu, chv) = (ch.map(|c| &c[l.u][..]), ch.map(|c| &c[l.v][..]));
        let (hu, hv) = two_mut(h, l.u, l.v);
        rows2(&mut hu[interior.clone()], &mut hv[interior.clone()], nv, |u, hu, hv| {
            let r = l.row(u);
            let mut tmp = Vec::new();
            let ew_up = row_up(&l, ew, ewo, u, pu, &mut tmp);
            let ew_row = &ew[r.clone()];
            let ew_end = beyond_hi(l.bc_v[1], ew_row, ewo.map(|o| &o[r.clone()]), pv);
            h_tm_row(hu, hv, ew_row, ew_up, ew_end, chu.map(|c| &c[r.clone()]), chv.map(|c| &c[r.clone()]), chs, ipu[u], ipv, l.sg);
        });
    }
}

/// E update (without CPML and material corrections) of the sets in use.
/// `ho` holds the other part of complex fields (Bloch boundaries).
pub(crate) fn update_e(plan: &Plan, td: TwoD, e: &mut [Vec<Real>; 3], h: &[Vec<Real>; 3], ho: Option<&[Vec<Real>; 3]>, bsign: [Real; 3]) {
    let l = Lay::new(plan, td);
    let (nu, nv) = (l.nu, l.nv);
    if nu == 0 || nv == 0 {
        return;
    }
    let ca = plan.ca.as_ref();
    let (idu, idv) = (&plan.inv_d[l.u], &plan.inv_d[l.v]);
    let (pu, pv) = (phase(plan, bsign, l.u), phase(plan, bsign, l.v));
    let interior = l.base..l.base + nu * nv;
    if td.active[0] {
        let hw = &h[l.w];
        let hwo = ho.map(|o| &o[l.w][..]);
        let (cbu, cbv) = (&plan.cb[l.u], &plan.cb[l.v]);
        let (cau, cav) = (ca.map(|c| &c[l.u][..]), ca.map(|c| &c[l.v][..]));
        let (eu, ev) = two_mut(e, l.u, l.v);
        rows2(&mut eu[interior.clone()], &mut ev[interior.clone()], nv, |u, eu, ev| {
            let r = l.row(u);
            let mut tmp = Vec::new();
            let hw_dn = row_down(&l, hw, hwo, u, pu, &mut tmp);
            let hw_row = &hw[r.clone()];
            let hw_start = beyond_lo(l.bc_v[0], hw_row, hwo.map(|o| &o[r.clone()]), pv);
            e_te_row(eu, ev, hw_row, hw_dn, hw_start, &cbu[r.clone()], &cbv[r.clone()], cau.map(|c| &c[r.clone()]), cav.map(|c| &c[r.clone()]), idu[u], idv, l.sg);
        });
    }
    if td.active[1] {
        let (hu, hv) = (&h[l.u], &h[l.v]);
        let (huo, hvo) = (ho.map(|o| &o[l.u][..]), ho.map(|o| &o[l.v][..]));
        let cbw = &plan.cb[l.w];
        let caw = ca.map(|c| &c[l.w][..]);
        rows1(&mut e[l.w][interior.clone()], nv, |u, ew| {
            let r = l.row(u);
            let mut tmp = Vec::new();
            let hv_dn = row_down(&l, hv, hvo, u, pu, &mut tmp);
            let hu_row = &hu[r.clone()];
            let hu_start = beyond_lo(l.bc_v[0], hu_row, huo.map(|o| &o[r.clone()]), pv);
            e_tm_row(ew, hu_row, &hv[r.clone()], hv_dn, hu_start, &cbw[r.clone()], caw.map(|c| &c[r.clone()]), idu[u], idv, l.sg);
        });
    }
}

/// CPML cells of one row, layers along u: `d` is the difference with the
/// row one step along u (`nb`; none: zero beyond the wall) and the profile
/// is one value for the row. Same arithmetic per cell as the scalar loop it
/// replaces, with the choices that do not change along the row taken out of
/// the loop so that it vectorises.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_u_gen<const FWD: bool>(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: Option<&[Real]>, c: Option<&[Real]>, coef_s: Real, sign: Real, b: Real, a: Real, km1: Real, iv: Real) {
    let n = t.len();
    let (p, cur) = (&mut p[..n], &cur[..n]);
    macro_rules! cells {
        ($o:expr, $c:expr) => {
            for x in 0..n {
                let o: Real = $o(x);
                let d = if FWD { (o - cur[x]) * iv } else { (cur[x] - o) * iv };
                let q = b * p[x] + a * d;
                p[x] = q;
                let cc: Real = sign * $c(x);
                t[x] += cc * (km1 * d + q);
            }
        };
    }
    match (nb, c) {
        (Some(r), Some(c)) => {
            let (r, c) = (&r[..n], &c[..n]);
            cells!(|x: usize| r[x], |x: usize| c[x])
        }
        (Some(r), None) => {
            let r = &r[..n];
            cells!(|x: usize| r[x], |_: usize| coef_s)
        }
        (None, Some(c)) => {
            let c = &c[..n];
            cells!(|_: usize| 0.0, |x: usize| c[x])
        }
        (None, None) => cells!(|_: usize| 0.0, |_: usize| coef_s),
    }
}

/// CPML cells of one row, layers along v: `d` is the difference with the
/// neighbour along the row (`nb`, already shifted) and the profile runs
/// along the row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_v_gen<const FWD: bool>(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: &[Real], c: Option<&[Real]>, coef_s: Real, sign: Real, pb: &[Real], pa: &[Real], pk: &[Real], inv: &[Real]) {
    let n = t.len();
    let (p, cur, nb, pb, pa, pk, inv) = (&mut p[..n], &cur[..n], &nb[..n], &pb[..n], &pa[..n], &pk[..n], &inv[..n]);
    macro_rules! cells {
        ($c:expr) => {
            for x in 0..n {
                let d = if FWD { (nb[x] - cur[x]) * inv[x] } else { (cur[x] - nb[x]) * inv[x] };
                let q = pb[x] * p[x] + pa[x] * d;
                p[x] = q;
                let cc: Real = sign * $c(x);
                t[x] += cc * ((pk[x] - 1.0) * d + q);
            }
        };
    }
    match c {
        Some(c) => {
            let c = &c[..n];
            cells!(|x: usize| c[x])
        }
        None => cells!(|_: usize| coef_s),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_u_fwd_imp(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: Option<&[Real]>, c: Option<&[Real]>, coef_s: Real, sign: Real, b: Real, a: Real, km1: Real, iv: Real) {
    pml_row_u_gen::<true>(t, p, cur, nb, c, coef_s, sign, b, a, km1, iv)
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_u_bwd_imp(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: Option<&[Real]>, c: Option<&[Real]>, coef_s: Real, sign: Real, b: Real, a: Real, km1: Real, iv: Real) {
    pml_row_u_gen::<false>(t, p, cur, nb, c, coef_s, sign, b, a, km1, iv)
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_v_fwd_imp(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: &[Real], c: Option<&[Real]>, coef_s: Real, sign: Real, pb: &[Real], pa: &[Real], pk: &[Real], inv: &[Real]) {
    pml_row_v_gen::<true>(t, p, cur, nb, c, coef_s, sign, pb, pa, pk, inv)
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pml_row_v_bwd_imp(t: &mut [Real], p: &mut [Real], cur: &[Real], nb: &[Real], c: Option<&[Real]>, coef_s: Real, sign: Real, pb: &[Real], pa: &[Real], pk: &[Real], inv: &[Real]) {
    pml_row_v_gen::<false>(t, p, cur, nb, c, coef_s, sign, pb, pa, pk, inv)
}
dispatch!(pml_row_u_fwd, pml_row_u_fwd_imp, (t: &mut [Real], p: &mut [Real], cur: &[Real], nb: Option<&[Real]>, c: Option<&[Real]>, coef_s: Real, sign: Real, b: Real, a: Real, km1: Real, iv: Real));
dispatch!(pml_row_u_bwd, pml_row_u_bwd_imp, (t: &mut [Real], p: &mut [Real], cur: &[Real], nb: Option<&[Real]>, c: Option<&[Real]>, coef_s: Real, sign: Real, b: Real, a: Real, km1: Real, iv: Real));
dispatch!(pml_row_v_fwd, pml_row_v_fwd_imp, (t: &mut [Real], p: &mut [Real], cur: &[Real], nb: &[Real], c: Option<&[Real]>, coef_s: Real, sign: Real, pb: &[Real], pa: &[Real], pk: &[Real], inv: &[Real]));
dispatch!(pml_row_v_bwd, pml_row_v_bwd_imp, (t: &mut [Real], p: &mut [Real], cur: &[Real], nb: &[Real], c: Option<&[Real]>, coef_s: Real, sign: Real, pb: &[Real], pa: &[Real], pk: &[Real], inv: &[Real]));

/// One half of the CPML correction of a slab: `target += sign * coef *
/// ((1/kappa - 1) d + psi)` with `psi <- b psi + a d` and `d` the
/// difference of `src` along the slab's axis (forward for the H update,
/// backward for the E update), over the cells of the slab.
///
/// The slab covers rows `u0..u0 + lu` and columns `v0..v0 + lv`; `psi` is
/// slab-local, row-major. `prof = (b, a, kinv)` are indexed by the position
/// along the slab's axis from its start.
#[allow(clippy::too_many_arguments)]
fn pml_half(
    l: &Lay,
    target: &mut [Real],
    src: &[Real],
    psi: &mut [Real],
    coef: Option<&[Real]>,
    coef_s: Real,
    sign: Real,
    prof: (&[Real], &[Real], &[Real]),
    inv: &[Real],
    along_u: bool,
    forward: bool,
    (u0, lu, v0, lv): (usize, usize, usize, usize),
) {
    let nv = l.nv;
    let (pb, pa, pk) = prof;
    // neighbour across the end of an axis without ghost cells: zero (the
    // wall behind the layers)
    let ghost_u = l.bc_u.is_none();
    let body = |iu: usize, trow: &mut [Real], prow: &mut [Real]| {
        let u = u0 + iu;
        let s0 = l.base + u * nv + v0;
        let t = &mut trow[v0..v0 + lv];
        let cur = &src[s0..s0 + lv];
        let c = coef.map(|c| &c[s0..s0 + lv]);
        if along_u {
            let q = iu;
            let (b, a, km1, iv) = (pb[q], pa[q], pk[q] - 1.0, inv[u]);
            // the row one step along u (forward: next, backward: previous)
            let other: Option<&[Real]> = if forward {
                (ghost_u || u + 1 < l.nu).then(|| &src[s0 + nv..s0 + nv + lv])
            } else {
                (ghost_u || u > 0).then(|| &src[s0 - nv..s0 - nv + lv])
            };
            if forward {
                pml_row_u_fwd(t, prow, cur, other, c, coef_s, sign, b, a, km1, iv);
            } else {
                pml_row_u_bwd(t, prow, cur, other, c, coef_s, sign, b, a, km1, iv);
            }
        } else {
            // one cell whose neighbour lies beyond the end of the row: zero there
            let edge = |x: usize, t: &mut [Real], prow: &mut [Real]| {
                let d = if forward { (0.0 - cur[x]) * inv[v0 + x] } else { (cur[x] - 0.0) * inv[v0 + x] };
                let p = pb[x] * prow[x] + pa[x] * d;
                prow[x] = p;
                let cc = sign * c.map(|c| c[x]).unwrap_or(coef_s);
                t[x] += cc * ((pk[x] - 1.0) * d + p);
            };
            if forward {
                let m = if v0 + lv < nv { lv } else { lv - 1 };
                pml_row_v_fwd(&mut t[..m], &mut prow[..m], &cur[..m], &src[s0 + 1..s0 + 1 + m], c.map(|c| &c[..m]), coef_s, sign, &pb[..m], &pa[..m], &pk[..m], &inv[v0..v0 + m]);
                if m < lv {
                    edge(lv - 1, t, prow);
                }
            } else {
                let f = if v0 > 0 { 0 } else { 1 };
                if f == 1 {
                    edge(0, t, prow);
                }
                pml_row_v_bwd(&mut t[f..], &mut prow[f..lv], &cur[f..], &src[s0 + f - 1..s0 + lv - 1], c.map(|c| &c[f..]), coef_s, sign, &pb[f..lv], &pa[f..lv], &pk[f..lv], &inv[v0 + f..v0 + lv]);
            }
        }
    };
    let rows = &mut target[l.base + u0 * nv..l.base + (u0 + lu) * nv];
    if lu * lv < serial_cells() {
        rows.chunks_mut(nv).zip(psi.chunks_mut(lv)).enumerate().for_each(|(iu, (t, p))| body(iu, t, p));
    } else {
        rows.par_chunks_mut(nv).zip(psi.par_chunks_mut(lv)).with_min_len(rows_per_task(lv)).enumerate().for_each(|(iu, (t, p))| body(iu, t, p));
    }
}

/// CPML corrections of the H update (`h_side`) or of the E update, for the
/// sets in use: the slab passes of [`super::cpu`] restricted to the plane.
/// `psi[slab] = [E_b, E_c, H_b, H_c]` states, as in the 3-D passes.
pub(crate) fn pml(plan: &Plan, td: TwoD, e: &mut [Vec<Real>; 3], h: &mut [Vec<Real>; 3], psi: &mut [[Vec<Real>; 4]], h_side: bool) {
    use crate::grid::Component;
    let l = Lay::new(plan, td);
    if l.nu == 0 || l.nv == 0 {
        return;
    }
    for (slab, st) in plan.pml.iter().zip(psi.iter_mut()) {
        let a = slab.axis;
        if a == l.w {
            continue;
        }
        let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
        let along_u = a == l.u;
        let range = (slab.start[l.u], slab.size[l.u], slab.start[l.v], slab.size[l.v]);
        let [psi_eb, psi_ec, psi_hb, psi_hc] = st;
        if h_side {
            let prof = (&slab.h_b[..], &slab.h_a[..], &slab.h_kinv[..]);
            let inv = &plan.inv_p[a];
            let ch = plan.ch.as_ref();
            // H_b += ch ((1/kappa - 1) d_a E_c + psi), H_c -= ch (... d_a E_b ...)
            if td.is_active(Component::h(bax)) {
                pml_half(&l, &mut h[bax], &e[cax], psi_hb, ch.map(|c| &c[bax][..]), plan.ch_scalar, 1.0, prof, inv, along_u, true, range);
            }
            if td.is_active(Component::h(cax)) {
                pml_half(&l, &mut h[cax], &e[bax], psi_hc, ch.map(|c| &c[cax][..]), plan.ch_scalar, -1.0, prof, inv, along_u, true, range);
            }
        } else {
            let prof = (&slab.e_b[..], &slab.e_a[..], &slab.e_kinv[..]);
            let inv = &plan.inv_d[a];
            // E_b -= cb ((1/kappa - 1) d_a H_c + psi), E_c += cb (... d_a H_b ...)
            if td.is_active(Component::e(bax)) {
                pml_half(&l, &mut e[bax], &h[cax], psi_eb, Some(&plan.cb[bax][..]), 0.0, -1.0, prof, inv, along_u, false, range);
            }
            if td.is_active(Component::e(cax)) {
                pml_half(&l, &mut e[cax], &h[bax], psi_ec, Some(&plan.cb[cax][..]), 0.0, 1.0, prof, inv, along_u, false, range);
            }
        }
    }
}

#[cfg(test)]
mod pml_row_tests {
    use super::*;

    /// A deterministic sequence in (-1, 1).
    fn seq(n: usize, seed: u32) -> Vec<Real> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as Real / (1u32 << 23) as Real - 1.0
            })
            .collect()
    }

    /// The CPML row kernels give the bits of the cell formula they replace,
    /// for every choice taken out of the loop and on every vector width the
    /// dispatch selects (lengths around the vector widths, with remainders).
    #[test]
    fn pml_rows_match_the_cell_formula() {
        for n in [1usize, 3, 7, 8, 15, 16, 17, 33, 100] {
            let (cur, nb, c, pb, pa, pk, inv) = (seq(n, 1), seq(n, 2), seq(n, 3), seq(n, 4), seq(n, 5), seq(n, 6), seq(n, 7));
            let (coef_s, sign, b, a, km1, iv) = (0.37 as Real, -1.0 as Real, 0.91 as Real, -0.23 as Real, -0.4 as Real, 2.5e7 as Real);
            for forward in [true, false] {
                for (has_nb, has_c) in [(true, true), (true, false), (false, true), (false, false)] {
                    let (mut t, mut p) = (seq(n, 8), seq(n, 9));
                    let (mut t_ref, mut p_ref) = (t.clone(), p.clone());
                    for x in 0..n {
                        let o = if has_nb { nb[x] } else { 0.0 };
                        let d = if forward { (o - cur[x]) * iv } else { (cur[x] - o) * iv };
                        let q = b * p_ref[x] + a * d;
                        p_ref[x] = q;
                        let cc = sign * if has_c { c[x] } else { coef_s };
                        t_ref[x] += cc * (km1 * d + q);
                    }
                    let (nbo, co) = (has_nb.then_some(&nb[..]), has_c.then_some(&c[..]));
                    if forward {
                        pml_row_u_fwd(&mut t, &mut p, &cur, nbo, co, coef_s, sign, b, a, km1, iv);
                    } else {
                        pml_row_u_bwd(&mut t, &mut p, &cur, nbo, co, coef_s, sign, b, a, km1, iv);
                    }
                    assert!(t.iter().zip(&t_ref).all(|(x, y)| x.to_bits() == y.to_bits()) && p.iter().zip(&p_ref).all(|(x, y)| x.to_bits() == y.to_bits()), "layers along u: n {n} forward {forward} neighbour {has_nb} coefficients {has_c}");
                }
                for has_c in [true, false] {
                    let (mut t, mut p) = (seq(n, 10), seq(n, 11));
                    let (mut t_ref, mut p_ref) = (t.clone(), p.clone());
                    for x in 0..n {
                        let d = if forward { (nb[x] - cur[x]) * inv[x] } else { (cur[x] - nb[x]) * inv[x] };
                        let q = pb[x] * p_ref[x] + pa[x] * d;
                        p_ref[x] = q;
                        let cc = sign * if has_c { c[x] } else { coef_s };
                        t_ref[x] += cc * ((pk[x] - 1.0) * d + q);
                    }
                    let co = has_c.then_some(&c[..]);
                    if forward {
                        pml_row_v_fwd(&mut t, &mut p, &cur, &nb, co, coef_s, sign, &pb, &pa, &pk, &inv);
                    } else {
                        pml_row_v_bwd(&mut t, &mut p, &cur, &nb, co, coef_s, sign, &pb, &pa, &pk, &inv);
                    }
                    assert!(t.iter().zip(&t_ref).all(|(x, y)| x.to_bits() == y.to_bits()) && p.iter().zip(&p_ref).all(|(x, y)| x.to_bits() == y.to_bits()), "layers along v: n {n} forward {forward} coefficients {has_c}");
                }
            }
        }
    }
}
