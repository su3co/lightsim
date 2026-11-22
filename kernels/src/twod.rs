//! Device kernels of the 2-D solver (`crates/lightsim-cuda/src/twod.rs`).
//!
//! A grid that is one periodic cell thick along an axis has fields that do
//! not depend on that coordinate, and Maxwell's equations split into two
//! independent sets of three components (TE: E in the plane, H along the
//! invariant axis; TM: the other three). `step_fused` treats such a grid as
//! a 3-D box one cell thick: one thread in 32 of each block does useful
//! work and all six components are updated.
//!
//! * `step2d`, `step2d_bloch` — a whole leapfrog step of the sets in use in
//!   one flat launch over the cells of the plane (see `twod_step.rs`, which
//!   holds the kernel body: it is shared with a host-side test);
//! * `dft_boxes` — running DFT of all the boxes of one or several monitors
//!   in one launch (`dft_accumulate` takes one launch per box, which is what
//!   bounds the step rate of a small 2-D problem with a few mode monitors);
//! * `inject_lists` — the source terms of several injection lists in one
//!   launch.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

include!("twod_step.rs");

#[cuda_module]
pub mod twod_kernels {
    use super::*;

    // Launch: 1-D, any block size, over the nu * nv cells.
    step2d_fn!($ [#[kernel]] step2d, false, [], thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x());
    step2d_fn!($ [#[kernel]] step2d_bloch, true, [], thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x());

    /// Running DFT of several monitor boxes in one launch (1-D over the
    /// cells of all the boxes together).
    ///
    /// `desc` holds 8 words per box: component (0..6 = Ex, Ey, Ez, Hx, Hy,
    /// Hz), x0, y0, z0, lx, ly, lz (interior indices, as `dft_accumulate`)
    /// and the index of the box's first cell in the launch (boxes in
    /// ascending order). `bufs` holds the device addresses of the real and
    /// imaginary accumulators of each box (two words per box), laid out
    /// `[freq][cell]`. `phasor_e` / `phasor_h` point at this step's rows of
    /// the E-time and H-time phasor tables.
    #[kernel]
    pub unsafe fn dft_boxes(
        ex: *const f32,
        ey: *const f32,
        ez: *const f32,
        hx: *const f32,
        hy: *const f32,
        hz: *const f32,
        bufs: *const u64,
        desc: *const u32,
        nbox: u32,
        phasor_e: *const f32,
        phasor_h: *const f32,
        nfreq: u32,
        total: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= total as usize {
            return;
        }
        unsafe {
            let mut b = 0usize;
            while b + 1 < nbox as usize && t >= *desc.add(8 * (b + 1) + 7) as usize {
                b += 1;
            }
            let d = desc.add(8 * b);
            let comp = *d;
            let lyu = *d.add(5) as usize;
            let lzu = *d.add(6) as usize;
            let ncell = (*d.add(4) as usize) * lyu * lzu;
            let local = t - *d.add(7) as usize;
            let kk = local % lzu;
            let jj = (local / lzu) % lyu;
            let ii = local / (lzu * lyu);
            let s = ((*d.add(1) as usize + ii + 1) * ny as usize + *d.add(2) as usize + jj)
                * nz as usize
                + *d.add(3) as usize
                + kk;
            let field = if comp == 0 {
                ex
            } else if comp == 1 {
                ey
            } else if comp == 2 {
                ez
            } else if comp == 3 {
                hx
            } else if comp == 4 {
                hy
            } else {
                hz
            };
            let phasor = if comp < 3 { phasor_e } else { phasor_h };
            let re = *bufs.add(2 * b) as *mut f32;
            let im = *bufs.add(2 * b + 1) as *mut f32;
            let v = *field.add(s);
            let mut f = 0usize;
            while f < nfreq as usize {
                let c = *phasor.add(2 * f);
                let sn = *phasor.add(2 * f + 1);
                let o = f * ncell + local;
                *re.add(o) += v * c;
                *im.add(o) += v * sn;
                f += 1;
            }
        }
    }

    /// Source terms of several injection lists in one launch (1-D over the
    /// entries of all the lists together): for entry `e` of a list,
    /// `field[idx[e]] += coef[e] * a[g(e)]` or, for complex coefficients,
    /// `coef[e] * a_re[g(e)] + coef_im[e] * a_im[g(e)]` (as `inject` /
    /// `inject_complex`).
    ///
    /// `lists` holds 8 words per list: the component of the target field
    /// (0..6), the device addresses of `idx`, `coef`, `coef_im` (0: real
    /// coefficients), `group` and of the amplitude table, the number of
    /// floats per step of that table with bit 32 set when the entries carry
    /// group indices, and the index of the list's first entry in the launch
    /// (lists in ascending order). The lists of one launch must target
    /// different fields or disjoint cells.
    #[kernel]
    pub unsafe fn inject_lists(
        ex: *mut f32,
        ey: *mut f32,
        ez: *mut f32,
        hx: *mut f32,
        hy: *mut f32,
        hz: *mut f32,
        lists: *const u64,
        nlist: u32,
        total: u32,
        step: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= total as usize {
            return;
        }
        unsafe {
            let mut b = 0usize;
            while b + 1 < nlist as usize && t >= *lists.add(8 * (b + 1) + 7) as usize {
                b += 1;
            }
            let l = lists.add(8 * b);
            let comp = *l;
            let e = t - *l.add(7) as usize;
            let field = if comp == 0 {
                ex
            } else if comp == 1 {
                ey
            } else if comp == 2 {
                ez
            } else if comp == 3 {
                hx
            } else if comp == 4 {
                hy
            } else {
                hz
            };
            let s = *(*l.add(1) as *const u64).add(e) as usize;
            let coef = *(*l.add(2) as *const f32).add(e);
            let rowlen = (*l.add(6) & 0xffff_ffff) as usize;
            let g = if (*l.add(6) >> 32) != 0 { *(*l.add(4) as *const u32).add(e) as usize } else { 0usize };
            let row = (*l.add(5) as *const f32).add(step as usize * rowlen);
            let cim = *l.add(3);
            if cim != 0 {
                *field.add(s) += coef * *row.add(2 * g) + *(cim as *const f32).add(e) * *row.add(2 * g + 1);
            } else {
                *field.add(s) += coef * *row.add(g);
            }
        }
    }
}
