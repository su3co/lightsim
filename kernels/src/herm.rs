//! Device kernel of the complex (Hermitian) permittivity tensors
//! (`lightsim_core::herm`): the off-diagonal coupling of the real and
//! imaginary field sets. Launched only for plans with such media
//! (`crates/lightsim-cuda/src/herm.rs`); dD is formed by `tensor_dd`.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod herm_kernels {
    use super::*;

    /// E_a,re += kR1 <dDr_b> + kR2 <dDr_c> - kI1 <dDi_b> - kI2 <dDi_c>,
    /// E_a,im += kR1 <dDi_b> + kR2 <dDi_c> + kI1 <dDr_b> + kI2 <dDr_c>, with
    /// b = a+1, c = a+2 and <.> the four-point averages of `tensor_apply`
    /// over storage planes [p0, p1) (`herm::averages` in the CPU engine).
    #[kernel]
    pub unsafe fn herm_apply(
        fr: *mut f32,
        fi: *mut f32,
        idx: *const u64,
        kr1: *const f32,
        kr2: *const f32,
        ki1: *const f32,
        ki2: *const f32,
        dr1: *const f32,
        dr2: *const f32,
        di1: *const f32,
        di2: *const f32,
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
            let mut r1 = 0.0f32;
            let mut r2 = 0.0f32;
            let mut i1 = 0.0f32;
            let mut i2 = 0.0f32;
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
                        r2 += *dr2.add(o);
                        i2 += *di2.add(o);
                    } else {
                        r1 += *dr1.add(o);
                        i1 += *di1.add(o);
                    }
                }
                q += 1;
            }
            let (r1, r2, i1, i2) = (0.25 * r1, 0.25 * r2, 0.25 * i1, 0.25 * i2);
            let a1 = *kr1.add(t);
            let a2 = *kr2.add(t);
            let b1 = *ki1.add(t);
            let b2 = *ki2.add(t);
            *fr.add(s) += a1 * r1 + a2 * r2 - b1 * i1 - b2 * i2;
            *fi.add(s) += a1 * i1 + a2 * i2 + b1 * r1 + b2 * r2;
        }
    }
}
