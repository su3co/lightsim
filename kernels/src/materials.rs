//! Device kernels of the material models that have their own per-cell state
//! beyond the ADE poles and the instantaneous chi2/chi3 step: the Raman
//! (delayed) chi3 response, gyrotropic (magneto-optic) poles and four-level
//! gain media. They are
//! launched only for plans that contain such media (`crates/lightsim-cuda/src/lib.rs`); the fused step and the
//! split update are not changed by them.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod material_kernels {
    use super::*;

    /// chi3 Raman cells, step 1 (before the E update): save E^n in `prev`
    /// and advance the Raman variable, Q^{n+1} = c0 Q^n + c1 Q^{n-1} +
    /// c2 (E^n)^2. On entry `q` = Q^n and `qp` = Q^{n-1}; on exit `q` =
    /// Q^{n+1} and `qp` = Q^n. `coefs` is `[medium row][c0, c1, c2]`.
    #[kernel]
    pub unsafe fn raman_save(
        field: *const f32,
        idx: *const u64,
        matid: *const u32,
        coefs: *const f32,
        prev: *mut f32,
        q: *mut f32,
        qp: *mut f32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let e = *field.add(*idx.add(t) as usize);
            let c = coefs.add((*matid.add(t) as usize) * 3);
            *prev.add(t) = e;
            let q0 = *q.add(t);
            let qn = *c * q0 + *c.add(1) * *qp.add(t) + *c.add(2) * (e * e);
            *qp.add(t) = q0;
            *q.add(t) = qn;
        }
    }

    /// chi3 Raman cells, step 2 (after the linear E update): with
    /// d = E_lin - E^n - cr E^n (Q^{n+1} - Q^n), l = 1 + cr Q^{n+1} and
    /// u = E^{n+1} - E^n, solve
    ///   u l + c2 (2 E u + u^2) + c3 (3 E^2 u + 3 E u^2 + u^3) = d
    /// (c2 = chi2/eps, c3 = (1 - f) chi3/eps, cr = f chi3/eps) by four Newton
    /// steps from the linearised guess, as `nl_apply`.
    #[kernel]
    pub unsafe fn raman_apply(
        field: *mut f32,
        idx: *const u64,
        prev: *const f32,
        c2: *const f32,
        c3: *const f32,
        cr: *const f32,
        q: *const f32,
        qp: *const f32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let e = *prev.add(t);
            let a = *c2.add(t);
            let b = *c3.add(t);
            let r = *cr.add(t);
            let q1 = *q.add(t);
            let d = *field.add(s) - e - r * e * (q1 - *qp.add(t));
            let l = 1.0 + r * q1;
            let mut u = d / (l + 2.0 * a * e + 3.0 * b * e * e);
            let mut it = 0;
            while it < 4 {
                let f = u * l + a * (2.0 * e * u + u * u) + b * (3.0 * e * e * u + 3.0 * e * u * u + u * u * u) - d;
                let fp = l + a * (2.0 * e + 2.0 * u) + b * (3.0 * e * e + 6.0 * e * u + 3.0 * u * u);
                u -= f / fp;
                it += 1;
            }
            *field.add(s) = e + u;
        }
    }
    /// Gyrotropic poles (after the linear E update and the dD/E^n exchange,
    /// with `tensor_apply`): at each listed E_a cell, the E^n vector
    /// (E_a, <E_b>, <E_c>) with the four-point averages of `tensor_apply`
    /// (`en_own`, `en1`, `en2` = E^n_a, E^n_{a+1}, E^n_{a+2} over storage
    /// planes [p0, p1)), then per pole P^{n+1} = A P^n + B P^{n-1} + C E^n
    /// with the 3x3 matrices of `tab` (`[row][pole][A, B, C]`, 27 floats), and
    /// E_a -= scale * sum_poles (P^{n+1}_a - P^n_a). `p` / `pp` = P^n / P^{n-1}
    /// as `[pole][xyz][cell]`.
    #[kernel]
    pub unsafe fn gyro_apply(
        field: *mut f32,
        idx: *const u64,
        en_own: *const f32,
        en1: *const f32,
        en2: *const f32,
        row: *const u32,
        scale: *const f32,
        tab: *const f32,
        np: u32,
        p: *mut f32,
        pp: *mut f32,
        n: u32,
        axis: u32,
        ny: u32,
        nz: u32,
        p0: u32,
        p1: u32,
        wrap: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let nyi = ny as i64;
            let nzi = nz as i64;
            let plane = (nyi * nzi) as usize;
            let pi = (s / plane) as i64;
            let rem = (s % plane) as i64;
            let j = rem / nzi;
            let k = rem % nzi;
            let lo = p0 as i64;
            let hi = p1 as i64;
            let a = axis;
            let mut ae1 = 0.0f32;
            let mut ae2 = 0.0f32;
            let mut q = 0u32;
            while q < 8 {
                let second = q >= 4;
                let b = if second { (a + 2) % 3 } else { (a + 1) % 3 };
                let up = (q & 1) != 0;
                let dn = (q & 2) != 0;
                let mut x = pi + (if up && a == 0 { 1 } else { 0 }) - (if dn && b == 0 { 1 } else { 0 });
                let mut y = j + (if up && a == 1 { 1 } else { 0 }) - (if dn && b == 1 { 1 } else { 0 });
                let mut z = k + (if up && a == 2 { 1 } else { 0 }) - (if dn && b == 2 { 1 } else { 0 });
                let mut ok = true;
                if x < lo || x >= hi {
                    if (wrap & 1) != 0 {
                        x = if x < lo { x + (hi - lo) } else { x - (hi - lo) };
                    } else {
                        ok = false;
                    }
                }
                if y < 0 || y >= nyi {
                    if (wrap & 2) != 0 {
                        y = if y < 0 { y + nyi } else { y - nyi };
                    } else {
                        ok = false;
                    }
                }
                if z < 0 || z >= nzi {
                    if (wrap & 4) != 0 {
                        z = if z < 0 { z + nzi } else { z - nzi };
                    } else {
                        ok = false;
                    }
                }
                if ok {
                    let o = ((x - lo) * nyi * nzi + y * nzi + z) as usize;
                    if second {
                        ae2 += *en2.add(o);
                    } else {
                        ae1 += *en1.add(o);
                    }
                }
                q += 1;
            }
            let own = *en_own.add(((pi - lo) * nyi * nzi + j * nzi + k) as usize);
            let eb1 = 0.25 * ae1;
            let eb2 = 0.25 * ae2;
            let ev = if a == 0 { [own, eb1, eb2] } else if a == 1 { [eb2, own, eb1] } else { [eb1, eb2, own] };
            let nn = n as usize;
            let npu = np as usize;
            let r = *row.add(t) as usize;
            let mut dp = 0.0f32;
            let mut pl = 0usize;
            while pl < npu {
                let tb = tab.add((r * npu + pl) * 27);
                let o0 = (pl * 3) * nn + t;
                let o1 = o0 + nn;
                let o2 = o1 + nn;
                let pc = [*p.add(o0), *p.add(o1), *p.add(o2)];
                let pm = [*pp.add(o0), *pp.add(o1), *pp.add(o2)];
                let mut rr = 0usize;
                while rr < 3 {
                    let mut v = 0.0f32;
                    let mut cc = 0usize;
                    while cc < 3 {
                        v += *tb.add(rr * 3 + cc) * pc[cc] + *tb.add(9 + rr * 3 + cc) * pm[cc] + *tb.add(18 + rr * 3 + cc) * ev[cc];
                        cc += 1;
                    }
                    let o = o0 + rr * nn;
                    *pp.add(o) = pc[rr];
                    *p.add(o) = v;
                    if rr == a as usize {
                        dp += v - pc[rr];
                    }
                    rr += 1;
                }
                pl += 1;
            }
            *field.add(s) -= *scale.add(t) * dp;
        }
    }
    /// Gain media, step 1 (before the E update): with the populations at n
    /// (`pop` = `[level][cell]`, f64), P^{n+1} = c0 P^n + c1 P^{n-1} +
    /// c2 (N1 - N2) E^n; saves E^n in `prev` and P^{n+1} - P^n in `dp`. `tab`
    /// is `[row][c0, c1, c2, eps0/(hbar w), dt/tau32, dt/tau21, dt/tau10,
    /// dt W_p]` (f64).
    #[kernel]
    pub unsafe fn gain_pre(
        field: *const f32,
        idx: *const u64,
        row: *const u32,
        tab: *const f64,
        pop: *const f64,
        prev: *mut f32,
        p: *mut f32,
        pp: *mut f32,
        dp: *mut f32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let nn = n as usize;
        if t >= nn {
            return;
        }
        unsafe {
            let e = *field.add(*idx.add(t) as usize);
            let k = tab.add((*row.add(t) as usize) * 8);
            let drive = (*k.add(2) * (*pop.add(nn + t) - *pop.add(2 * nn + t))) as f32;
            let p0 = *p.add(t);
            let pn = (*k) as f32 * p0 + (*k.add(1)) as f32 * *pp.add(t) + drive * e;
            *prev.add(t) = e;
            *pp.add(t) = p0;
            *p.add(t) = pn;
            *dp.add(t) = pn - p0;
        }
    }

    /// Gain media, step 2 (after the linear E update): E^{n+1} -= scale dP,
    /// then the populations: eps0/(hbar w) (E^n + E^{n+1})/2 dP atoms per
    /// volume go from level 1 to 2, and the decays and the pump move
    /// dt/tau and dt W_p of their source level (explicit), as `gain_rates`
    /// in the CPU engine.
    #[kernel]
    pub unsafe fn gain_post(
        field: *mut f32,
        idx: *const u64,
        row: *const u32,
        tab: *const f64,
        scale: *const f32,
        prev: *const f32,
        dp: *const f32,
        pop: *mut f64,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let nn = n as usize;
        if t >= nn {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let d = *dp.add(t);
            let e1 = *field.add(s) - *scale.add(t) * d;
            *field.add(s) = e1;
            let k = tab.add((*row.add(t) as usize) * 8);
            let work = 0.5 * (*prev.add(t) as f64 + e1 as f64) * d as f64;
            let n0 = *pop.add(t);
            let n1 = *pop.add(nn + t);
            let n2 = *pop.add(2 * nn + t);
            let n3 = *pop.add(3 * nn + t);
            let up = *k.add(3) * work;
            let f32_ = *k.add(4) * n3;
            let f21 = *k.add(5) * n2;
            let f10 = *k.add(6) * n1;
            let fp = *k.add(7) * n0;
            *pop.add(t) = n0 + f10 - fp;
            *pop.add(nn + t) = n1 + f21 - f10 - up;
            *pop.add(2 * nn + t) = n2 + f32_ - f21 + up;
            *pop.add(3 * nn + t) = n3 + fp - f32_;
        }
    }
}
