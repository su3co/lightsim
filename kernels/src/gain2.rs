//! Device kernels of the four-level two-electron gain medium
//! (`lightsim_core::gain2`). Launched only for plans with such media
//! (`crates/lightsim-cuda/src/gain2.rs`); no other kernel changes.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod gain2_kernels {
    use super::*;

    /// Step 1 (before the E update): with the populations at n (`pop` =
    /// `[level][cell]`, f64) advance both polarizations,
    /// Pa^{n+1} = ca0 Pa + ca1 Pa_prev + ca2 (N1 - N2) E^n and
    /// Pb^{n+1} = cb0 Pb + cb1 Pb_prev + cb2 (N0 - N3) E^n; `st` holds the
    /// columns E^n, Pa, Pa_prev, Pb, Pb_prev, dPa, dPb (`[7][n]`); `tab`
    /// is `[row][13]` (f64, `lightsim_core::gain2::GAIN2_TAB`).
    #[kernel]
    pub unsafe fn gain2_pre(field: *const f32, idx: *const u64, row: *const u32, tab: *const f64, pop: *const f64, st: *mut f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let nn = n as usize;
        if t >= nn {
            return;
        }
        unsafe {
            let e = *field.add(*idx.add(t) as usize);
            let k = tab.add((*row.add(t) as usize) * 13);
            let da = (*k.add(2) * (*pop.add(nn + t) - *pop.add(2 * nn + t))) as f32;
            let db = (*k.add(5) * (*pop.add(t) - *pop.add(3 * nn + t))) as f32;
            *st.add(t) = e;
            let pa = *st.add(nn + t);
            let pb = *st.add(3 * nn + t);
            let pan = (*k) as f32 * pa + (*k.add(1)) as f32 * *st.add(2 * nn + t) + da * e;
            let pbn = (*k.add(3)) as f32 * pb + (*k.add(4)) as f32 * *st.add(4 * nn + t) + db * e;
            *st.add(2 * nn + t) = pa;
            *st.add(nn + t) = pan;
            *st.add(4 * nn + t) = pb;
            *st.add(3 * nn + t) = pbn;
            *st.add(5 * nn + t) = pan - pa;
            *st.add(6 * nn + t) = pbn - pb;
        }
    }

    /// Step 2 (after the linear E update): E^{n+1} -= scale (dPa + dPb),
    /// then the populations move by explicit fluxes (Pauli factors) and the
    /// field work, as `rates_cell` in the CPU engine.
    #[kernel]
    pub unsafe fn gain2_post(field: *mut f32, idx: *const u64, row: *const u32, tab: *const f64, scale: *const f32, st: *const f32, pop: *mut f64, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let nn = n as usize;
        if t >= nn {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let dpa = *st.add(5 * nn + t);
            let dpb = *st.add(6 * nn + t);
            let e1 = *field.add(s) - *scale.add(t) * (dpa + dpb);
            *field.add(s) = e1;
            let k = tab.add((*row.add(t) as usize) * 13);
            let eavg = 0.5 * (*st.add(t) as f64 + e1 as f64);
            let n0 = *pop.add(t);
            let n1 = *pop.add(nn + t);
            let n2 = *pop.add(2 * nn + t);
            let n3 = *pop.add(3 * nn + t);
            let inv = *k.add(12);
            let ua = *k.add(6) * (eavg * dpa as f64);
            let ub = *k.add(7) * (eavg * dpb as f64);
            let f30 = *k.add(8) * n3 * (1.0 - n0 * inv);
            let f32_ = *k.add(9) * n3 * (1.0 - n2 * inv);
            let f21 = *k.add(10) * n2 * (1.0 - n1 * inv);
            let f10 = *k.add(11) * n1 * (1.0 - n0 * inv);
            *pop.add(t) = n0 + f30 + f10 - ub;
            *pop.add(nn + t) = n1 + f21 - ua - f10;
            *pop.add(2 * nn + t) = n2 + f32_ + ua - f21;
            *pop.add(3 * nn + t) = n3 + ub - f32_ - f30;
        }
    }
}
