//! Device kernels of the zero-thickness sheets (`lightsim_core::sheet`).
//! Launched only for plans with sheets (`crates/lightsim-cuda/src/sheets.rs`);
//! the fused step and the split update are not changed by them.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod sheet_kernels {
    use super::*;

    /// Sheet cells, step 1 (before the E update): save E^n in `prev` and
    /// advance the poles of the sheet, P_q^{n+1} = c0 P_q^n + c1 P_q^{n-1} +
    /// c2 E^n (`coefs`: the sheet's row, `[pole][c0, c1, c2]`, `p` / `pp` =
    /// P^n / P^{n-1} as `[pole][cell]`); `dp` = sum_q (P_q^{n+1} - P_q^n).
    #[kernel]
    pub unsafe fn sheet_pre(
        field: *const f32,
        idx: *const u64,
        coefs: *const f32,
        np: u32,
        p: *mut f32,
        pp: *mut f32,
        prev: *mut f32,
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
            *prev.add(t) = e;
            let mut acc = 0.0f32;
            let mut q = 0usize;
            while q < np as usize {
                let c = coefs.add(q * 3);
                let o = q * nn + t;
                let p0 = *p.add(o);
                let pn = *c * p0 + *c.add(1) * *pp.add(o) + *c.add(2) * e;
                acc += pn - p0;
                *pp.add(o) = p0;
                *p.add(o) = pn;
                q += 1;
            }
            *dp.add(t) = acc;
        }
    }

    /// Sheet cells, step 2 (after the linear E update):
    /// E^{n+1} = (E_lin - k dp - k (h - chi0) E^n) / (1 + k (h + chi0)), with
    /// the cell factor `k`, h = sigma_0 dt / (2 eps0) and chi0 of the sheet
    /// (`sheet_post_cell` in the CPU engine).
    #[kernel]
    pub unsafe fn sheet_post(
        field: *mut f32,
        idx: *const u64,
        k: *const f32,
        prev: *const f32,
        dp: *const f32,
        h: f32,
        chi0: f32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let kk = *k.add(t);
            let en = *prev.add(t);
            *field.add(s) = (*field.add(s) - kk * *dp.add(t) - kk * (h - chi0) * en) / (1.0 + kk * (h + chi0));
        }
    }
}
