//! Device kernels of the material models that have their own per-cell state
//! beyond the ADE poles and the instantaneous chi2/chi3 step: the Raman
//! (delayed) chi3 response. They are launched only for plans that contain
//! such media (`crates/lightsim-cuda/src/lib.rs`); the fused step and the
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
}
