//! Device kernels of the fixed-angle (broadband oblique incidence) engine,
//! `crates/lightsim-cuda/src/fixed_angle.rs`, mirroring the CPU engine in
//! `lightsim_core::engine::fixed_angle`. The curl updates, CPML and sources
//! of a fixed-angle run are those of the ordinary split kernels; the kernels
//! here add the coupling of the transformed fields:
//!
//! * `fa_pass_f`, `fa_pml_f`, `fa_pole_j`: the derivative term F at the E
//!   cells of a pair for the H update (time n, with its CPML auxiliary and
//!   the pole and conduction currents);
//! * `fa_pass_k`, `fa_pml_k`: the term K at the H cells for the E update;
//! * `fa_corr`: source-plane corrections of F and K;
//! * `fa_line_h`, `fa_line_e`: the cyclic tridiagonal solves along the tilt
//!   axis, one thread per line (factors precomputed on the host);
//! * `fa_poles_pre`, `fa_poles_post`: trapezoidal pole currents.
//!
//! Storage layout and units as in `main.rs`. Edge modes along the
//! propagation axis: 0 zero, 1 mirror with factor `edge_f`.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod fa_kernels {
    use super::*;

    /// Back substitution and Sherman-Morrison correction of one line (the
    /// forward sweep has left d' in place).
    macro_rules! line_back {
        ($x:expr, $cp:expr, $zs:expr, $r:expr, $base:expr, $su:expr, $n:expr) => {{
            let mut next = *$x.add($base + ($n - 1) * $su);
            let mut i = $n - 1;
            while i > 0 {
                i -= 1;
                let si = $base + i * $su;
                let y = *$x.add(si) - *$cp.add(si) * next;
                *$x.add(si) = y;
                next = y;
            }
            let fct = *$x.add($base) + $r * *$x.add($base + ($n - 1) * $su);
            let mut i = 0usize;
            while i < $n {
                let si = $base + i * $su;
                *$x.add(si) -= fct * *$zs.add(si);
                i += 1;
            }
        }};
    }

    /// F[s] = (d_q H_u at time n) - sigma (conduction current of E_p^n), over
    /// all interior cells (1-D launch).
    #[kernel]
    pub unsafe fn fa_pass_f(
        f: *mut f32,
        hu: *const f32,
        ho: *const f32,
        ep: *const f32,
        cb: *const f32,
        ca: *const f32,
        has_ca: u32,
        invq: *const f32,
        ncell: u32,
        ny: u32,
        nz: u32,
        q: u32,
        nq: u32,
        sq: u32,
        periodic: u32,
        edge_mode: u32,
        edge_f: f32,
        sigma: f32,
    ) {
        let c = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        if c >= ncell as usize {
            return;
        }
        let nzu = nz as usize;
        let plane = (ny as usize) * nzu;
        let s = plane + c;
        let jq = if q == 0 {
            c / plane
        } else if q == 1 {
            (c / nzu) % (ny as usize)
        } else {
            c % nzu
        };
        let sq = sq as usize;
        unsafe {
            let h0 = *hu.add(s);
            let o0 = *ho.add(s);
            let mut an = 0.0f32;
            let mut ao = 0.0f32;
            // (along x the ghost plane holds the neighbour, refreshed by the host)
            if jq > 0 || q == 0 {
                an = *hu.add(s - sq);
                ao = *ho.add(s - sq);
            } else if periodic != 0 {
                let w = s + (nq as usize - 1) * sq;
                an = *hu.add(w);
                ao = *ho.add(w);
            } else if edge_mode == 1 {
                an = edge_f * h0;
                ao = edge_f * o0;
            }
            let d = 0.5 * ((h0 - an) + (o0 - ao)) * *invq.add(jq);
            let mut jc = 0.0f32;
            if has_ca != 0 {
                let cbv = *cb.add(s);
                if cbv != 0.0 {
                    jc = (1.0 - *ca.add(s)) / cbv * *ep.add(s);
                }
            }
            *f.add(s) = d - sigma * jc;
        }
    }

    /// CPML part of d_w H_u at time n in F (one PML slab, 1-D launch over its
    /// cells): psi^{n+1/2} = b psi + a d(H_u^{n+1/2}); F += (kinv - 1) d^n +
    /// psi^n, with the time-n values as averages.
    #[kernel]
    pub unsafe fn fa_pml_f(
        f: *mut f32,
        hu: *const f32,
        ho: *const f32,
        psi: *mut f32,
        eb: *const f32,
        ea: *const f32,
        ek: *const f32,
        invw: *const f32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        w: u32,
        ny: u32,
        nz: u32,
        edge_mode: u32,
        edge_f: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let (lyu, lzu) = (ly as usize, lz as usize);
        if t >= (lx as usize) * lyu * lzu {
            return;
        }
        let ii = t / (lyu * lzu);
        let jj = (t / lzu) % lyu;
        let kk = t % lzu;
        let (i, j, k) = (x0 as usize + ii, y0 as usize + jj, z0 as usize + kk);
        let nzu = nz as usize;
        let plane = (ny as usize) * nzu;
        let s = (i + 1) * plane + j * nzu + k;
        let (along, start, sw) = if w == 0 {
            (i, x0 as usize, plane)
        } else if w == 1 {
            (j, y0 as usize, nzu)
        } else {
            (k, z0 as usize, 1usize)
        };
        let qq = along - start;
        unsafe {
            let h0 = *hu.add(s);
            let o0 = *ho.add(s);
            let mut ln = 0.0f32;
            let mut lo = 0.0f32;
            if along > 0 || w == 0 {
                ln = *hu.add(s - sw);
                lo = *ho.add(s - sw);
            } else if edge_mode == 1 {
                ln = edge_f * h0;
                lo = edge_f * o0;
            }
            let inv = *invw.add(along);
            let dn = (h0 - ln) * inv;
            let dold = (o0 - lo) * inv;
            let p0 = *psi.add(t);
            let pn = *eb.add(qq) * p0 + *ea.add(qq) * dn;
            *psi.add(t) = pn;
            *f.add(s) += (*ek.add(qq) - 1.0) * 0.5 * (dn + dold) + 0.5 * (p0 + pn);
        }
    }

    /// x[idx[e]] += fac[e] * a (source-plane corrections).
    #[kernel]
    pub unsafe fn fa_corr(x: *mut f32, idx: *const u64, fac: *const f32, n: u32, a: f32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            *x.add(s) += *fac.add(t) * a;
        }
    }

    /// F[s] -= sigma jw J^n at the dispersive cells of E_p.
    #[kernel]
    pub unsafe fn fa_pole_j(
        f: *mut f32,
        ep: *const f32,
        idx: *const u64,
        matid: *const u32,
        jw: *const f32,
        ys: *const f32,
        tab: *const f32,
        npole: u32,
        n: u32,
        sigma: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let nu = n as usize;
        if t >= nu {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let en = *ep.add(s);
            let mut jn = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = tab.add((m * np + q) * 9);
                jn += *c.add(6) * *ys.add((q * 2) * nu + t) + *c.add(7) * *ys.add((q * 2 + 1) * nu + t) + *c.add(8) * en;
                q += 1;
            }
            *f.add(s) -= sigma * *jw.add(t) * jn;
        }
    }

    /// Q_q update along one line of the tilt axis per thread: the right-hand
    /// side on the fly, forward sweep, back substitution, Sherman-Morrison.
    #[kernel]
    pub unsafe fn fa_line_h(
        hq: *mut f32,
        wo: *const f32,
        f: *const f32,
        cb: *const f32,
        ca: *const f32,
        has_ca: u32,
        ch: *const f32,
        has_ch: u32,
        chs: f32,
        sv: f32,
        tv: f32,
        idu: *const f32,
        n: u32,
        su: u32,
        nlines: u32,
        st1: u32,
        st2: u32,
        n2: u32,
        off: u32,
        lo: *const f32,
        cp: *const f32,
        idn: *const f32,
        zs: *const f32,
        rl: *const f32,
    ) {
        let l = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        if l >= nlines as usize {
            return;
        }
        let nn = n as usize;
        let su = su as usize;
        let base = off as usize + (l / n2 as usize) * st1 as usize + (l % n2 as usize) * st2 as usize;
        unsafe {
            let mut prev = 0.0f32;
            let mut i = 0usize;
            while i < nn {
                let si = base + i * su;
                let sm = base + ((i + nn - 1) % nn) * su;
                let ip1 = (i + 1) % nn;
                let sp = base + ip1 * su;
                let c0 = *cb.add(si);
                let c1 = *cb.add(sp);
                let mut e0 = 0.0f32;
                let mut e1 = 0.0f32;
                if c0 != 0.0 {
                    let a0 = if has_ca != 0 { *ca.add(si) } else { 1.0 };
                    e0 = 2.0 * c0 / (tv * (1.0 + a0));
                }
                if c1 != 0.0 {
                    let a1 = if has_ca != 0 { *ca.add(sp) } else { 1.0 };
                    e1 = 2.0 * c1 / (tv * (1.0 + a1));
                }
                let chq = if has_ch != 0 { *ch.add(si) } else { chs };
                let k = chq * sv * sv / tv;
                let hc = 0.5 * chq * sv;
                let w0 = *wo.add(si);
                let wm = *wo.add(sm);
                let wp = *wo.add(sp);
                let rhs = *hq.add(si) - 0.25 * k * (e0 * (wm + w0) + e1 * (w0 + wp)) - 0.5 * hc * (e0 * (w0 - wm) * *idu.add(i) + e1 * (wp - w0) * *idu.add(ip1))
                    + chq * sv * 0.5 * (e0 * *f.add(si) + e1 * *f.add(sp));
                let d = (rhs - *lo.add(si) * prev) * *idn.add(si);
                *hq.add(si) = d;
                prev = d;
                i += 1;
            }
            line_back!(hq, cp, zs, *rl.add(l), base, su, nn);
        }
    }

    /// P_p update along one line of the tilt axis per thread.
    #[kernel]
    pub unsafe fn fa_line_e(
        ep: *mut f32,
        yo: *const f32,
        kb: *const f32,
        cb: *const f32,
        ch: *const f32,
        has_ch: u32,
        chs: f32,
        sv: f32,
        tv: f32,
        ipu: *const f32,
        n: u32,
        su: u32,
        nlines: u32,
        st1: u32,
        st2: u32,
        n2: u32,
        off: u32,
        lo: *const f32,
        cp: *const f32,
        idn: *const f32,
        zs: *const f32,
        rl: *const f32,
    ) {
        let l = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        if l >= nlines as usize {
            return;
        }
        let nn = n as usize;
        let su = su as usize;
        let base = off as usize + (l / n2 as usize) * st1 as usize + (l % n2 as usize) * st2 as usize;
        unsafe {
            let mut prev = 0.0f32;
            let mut i = 0usize;
            while i < nn {
                let si = base + i * su;
                let im1 = (i + nn - 1) % nn;
                let sm = base + im1 * su;
                let sp = base + ((i + 1) % nn) * su;
                let r_hi = (if has_ch != 0 { *ch.add(si) } else { chs }) / tv;
                let r_lo = (if has_ch != 0 { *ch.add(sm) } else { chs }) / tv;
                let cbp = *cb.add(si);
                let k = cbp * sv * sv / tv;
                let hc = 0.5 * cbp * sv;
                let y0 = *yo.add(si);
                let ym = *yo.add(sm);
                let yp = *yo.add(sp);
                let g_hi = (yp - y0) * *ipu.add(i) - *kb.add(si);
                let g_lo = (y0 - ym) * *ipu.add(im1) - *kb.add(sm);
                let rhs = *ep.add(si) - 0.25 * k * (r_hi * (y0 + yp) + r_lo * (ym + y0)) - 0.5 * hc * (r_hi * g_hi + r_lo * g_lo);
                let d = (rhs - *lo.add(si) * prev) * *idn.add(si);
                *ep.add(si) = d;
                prev = d;
                i += 1;
            }
            line_back!(ep, cp, zs, *rl.add(l), base, su, nn);
        }
    }

    /// K[s] = d_p E_u at time n + d_p E_u at time n + 1 at the H cells, over
    /// all interior cells (1-D launch).
    #[kernel]
    pub unsafe fn fa_pass_k(
        kb: *mut f32,
        eu: *const f32,
        eo: *const f32,
        invp: *const f32,
        ncell: u32,
        ny: u32,
        nz: u32,
        p: u32,
        np: u32,
        sp: u32,
        periodic: u32,
        edge_mode: u32,
        edge_f: f32,
    ) {
        let c = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        if c >= ncell as usize {
            return;
        }
        let nzu = nz as usize;
        let plane = (ny as usize) * nzu;
        let s = plane + c;
        let jp = if p == 0 {
            c / plane
        } else if p == 1 {
            (c / nzu) % (ny as usize)
        } else {
            c % nzu
        };
        let spu = sp as usize;
        unsafe {
            let e0 = *eu.add(s);
            let o0 = *eo.add(s);
            let mut an = 0.0f32;
            let mut ao = 0.0f32;
            if jp + 1 < np as usize || p == 0 {
                an = *eu.add(s + spu);
                ao = *eo.add(s + spu);
            } else if periodic != 0 {
                let w = s - (np as usize - 1) * spu;
                an = *eu.add(w);
                ao = *eo.add(w);
            } else if edge_mode == 1 {
                an = edge_f * e0;
                ao = edge_f * o0;
            }
            *kb.add(s) = ((an - e0) + (ao - o0)) * *invp.add(jp);
        }
    }

    /// CPML part of d_w E_u (times n and n + 1) in K, one PML slab.
    #[kernel]
    pub unsafe fn fa_pml_k(
        kb: *mut f32,
        eu: *const f32,
        eo: *const f32,
        psi: *mut f32,
        hb: *const f32,
        ha: *const f32,
        hk: *const f32,
        invw: *const f32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        w: u32,
        ny: u32,
        nz: u32,
        nw: u32,
        edge_mode: u32,
        edge_f: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let (lyu, lzu) = (ly as usize, lz as usize);
        if t >= (lx as usize) * lyu * lzu {
            return;
        }
        let ii = t / (lyu * lzu);
        let jj = (t / lzu) % lyu;
        let kk = t % lzu;
        let (i, j, k) = (x0 as usize + ii, y0 as usize + jj, z0 as usize + kk);
        let nzu = nz as usize;
        let plane = (ny as usize) * nzu;
        let s = (i + 1) * plane + j * nzu + k;
        let (along, start, sw) = if w == 0 {
            (i, x0 as usize, plane)
        } else if w == 1 {
            (j, y0 as usize, nzu)
        } else {
            (k, z0 as usize, 1usize)
        };
        let qq = along - start;
        unsafe {
            let e0 = *eu.add(s);
            let o0 = *eo.add(s);
            let mut hn = 0.0f32;
            let mut ho = 0.0f32;
            if along + 1 < nw as usize || w == 0 {
                hn = *eu.add(s + sw);
                ho = *eo.add(s + sw);
            } else if edge_mode == 1 {
                hn = edge_f * e0;
                ho = edge_f * o0;
            }
            let inv = *invw.add(along);
            let dn = (hn - e0) * inv;
            let dold = (ho - o0) * inv;
            let p0 = *psi.add(t);
            let pn = *hb.add(qq) * p0 + *ha.add(qq) * dn;
            *psi.add(t) = pn;
            *kb.add(s) += (*hk.add(qq) - 1.0) * (dn + dold) + p0 + pn;
        }
    }

    /// Before the P_p solves: E*[s] -= w (J^n + J0) / 2 at the dispersive
    /// cells of one component (`is_u`: the tilt component, final after
    /// division by 1 + w g / 2).
    #[kernel]
    pub unsafe fn fa_poles_pre(
        e: *mut f32,
        eo: *const f32,
        idx: *const u64,
        matid: *const u32,
        wt: *const f32,
        ys: *const f32,
        tab: *const f32,
        gsum: *const f32,
        npole: u32,
        n: u32,
        is_u: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let nu = n as usize;
        if t >= nu {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let y = *eo.add(s);
            let mut j = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = tab.add((m * np + q) * 9);
                let y0 = *ys.add((q * 2) * nu + t);
                let y1 = *ys.add((q * 2 + 1) * nu + t);
                j += *c.add(6) * y0 + *c.add(7) * y1 + *c.add(8) * y;
                j += *c.add(6) * (*c * y0 + *c.add(1) * y1 + *c.add(4) * y) + *c.add(7) * (*c.add(2) * y0 + *c.add(3) * y1 + *c.add(5) * y);
                q += 1;
            }
            let wv = *wt.add(t);
            let mut x = *e.add(s) - 0.5 * wv * j;
            if is_u != 0 {
                x /= 1.0 + 0.5 * wv * *gsum.add(m);
            }
            *e.add(s) = x;
        }
    }

    /// After the solves: y^{n+1} = M y^n + v (E^n + E^{n+1}).
    #[kernel]
    pub unsafe fn fa_poles_post(e: *const f32, eo: *const f32, idx: *const u64, matid: *const u32, ys: *mut f32, tab: *const f32, npole: u32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let nu = n as usize;
        if t >= nu {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let sum = *eo.add(s) + *e.add(s);
            let mut q = 0usize;
            while q < np {
                let c = tab.add((m * np + q) * 9);
                let i0 = (q * 2) * nu + t;
                let i1 = (q * 2 + 1) * nu + t;
                let y0 = *ys.add(i0);
                let y1 = *ys.add(i1);
                *ys.add(i0) = *c * y0 + *c.add(1) * y1 + *c.add(4) * sum;
                *ys.add(i1) = *c.add(2) * y0 + *c.add(3) * y1 + *c.add(5) * sum;
                q += 1;
            }
        }
    }
}
