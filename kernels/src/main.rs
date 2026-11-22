//! lightsim FDTD device kernels.
//!
//! Every kernel here is compiled to PTX by NVIDIA's cuda-oxide backend
//! (`cargo oxide build`). The host engine (`crates/lightsim-cuda`) loads the
//! generated PTX with `cuda-core` and launches kernels by name, so kernel
//! signatures are an ABI: raw device pointers are passed as `u64`, scalars as
//! their native 32-bit types. Keep the `Args` lists in
//! `crates/lightsim-cuda/src/lib.rs` in sync with any signature change here.
//!
//! # Storage layout
//!
//! All per-cell arrays (fields, material coefficients) are stored x-major with
//! one ghost plane on each side of the x axis:
//!
//! ```text
//! s(i, j, k) = ((i + 1) * ny + j) * nz + k,   i in -1..=nx,  j in 0..ny,  k in 0..nz
//! ```
//!
//! so a slab owned by one GPU is contiguous and halo planes are contiguous
//! `ny * nz` blocks. The y and z boundaries are handled inside the kernels with
//! a boundary-mode code per side:
//!
//! * `0` — zero ghost (PEC wall, also the outer wall behind a PML)
//! * `1` — periodic
//! * `2` — mirror (PMC on the low side: antisymmetric H; on the high side the
//!         E ghost equals the last interior plane)
//! * `3` — Bloch-periodic with a non-zero wave vector (complex fields as two
//!         real sets; the wrap mixes them with the phase of k L)
//!
//! Fields are in normalized units: `E` in V/m and `H~ = eta0 * H` (also V/m),
//! so the update coefficients are `c*dt/eps_r` and `c*dt/mu_r` (metres) and the
//! curl uses inverse cell spacings (1/m).

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{DynamicSharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Material index of storage cell `s`: `mode` 0 = uniform (always row
    /// 0, no per-cell array), 1 = u8, 2 = u16, 4 = u32 entries.
    macro_rules! mat_index {
        ($mat:expr, $mode:expr, $s:expr) => {
            if $mode == 0 {
                0usize
            } else if $mode == 1 {
                *($mat as *const u8).add($s) as usize
            } else if $mode == 2 {
                *($mat as *const u16).add($s) as usize
            } else {
                *($mat as *const u32).add($s) as usize
            }
        };
    }

    /// H~ <- H~ - ch * curl(E)
    ///
    /// Launch: block (bx, by, 1), grid (ceil(nz/bx), ceil(ny/by), nx).
    /// `ch_mode == 0` uses the scalar `ch_scalar`; `ch_mode == 1` reads the
    /// per-component arrays `chx/chy/chz` (magnetic materials).
    #[kernel]
    pub unsafe fn update_h(
        hx: *mut f32,
        hy: *mut f32,
        hz: *mut f32,
        ex: *const f32,
        ey: *const f32,
        ez: *const f32,
        chx: *const f32,
        chy: *const f32,
        chz: *const f32,
        ch_scalar: f32,
        ch_mode: u32,
        idx_p: *const f32,
        idy_p: *const f32,
        idz_p: *const f32,
        nx: u32,
        ny: u32,
        nz: u32,
        bc_yhi: u32,
        bc_zhi: u32,
        exo: *const f32,
        eyo: *const f32,
        ezo: *const f32,
        bcy: f32,
        bsy: f32,
        bcz: f32,
        bsz: f32,
    ) {
        let k = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let j = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as usize;
        let i = thread::blockIdx_z() as usize;
        let nyu = ny as usize;
        let nzu = nz as usize;
        if k >= nzu || j >= nyu || i >= nx as usize {
            return;
        }
        let plane = nyu * nzu;
        let s = (i + 1) * plane + j * nzu + k;
        unsafe {
            let ex0 = *ex.add(s);
            let ey0 = *ey.add(s);
            let ez0 = *ez.add(s);
            // +x neighbours: the ghost plane i = nx is maintained by the host.
            let ey_xp = *ey.add(s + plane);
            let ez_xp = *ez.add(s + plane);
            // +y neighbours
            let mut ex_yp = 0.0f32;
            let mut ez_yp = 0.0f32;
            if j + 1 < nyu {
                ex_yp = *ex.add(s + nzu);
                ez_yp = *ez.add(s + nzu);
            } else if bc_yhi == 1 {
                ex_yp = *ex.add(s - (nyu - 1) * nzu);
                ez_yp = *ez.add(s - (nyu - 1) * nzu);
            } else if bc_yhi == 2 {
                ex_yp = ex0;
                ez_yp = ez0;
            } else if bc_yhi == 3 {
                // Bloch: E(ny) = E(0) exp(i k L) (bsy carries the part's sign)
                let w = s - (nyu - 1) * nzu;
                ex_yp = bcy * *ex.add(w) + bsy * *exo.add(w);
                ez_yp = bcy * *ez.add(w) + bsy * *ezo.add(w);
            }
            // +z neighbours
            let mut ex_zp = 0.0f32;
            let mut ey_zp = 0.0f32;
            if k + 1 < nzu {
                ex_zp = *ex.add(s + 1);
                ey_zp = *ey.add(s + 1);
            } else if bc_zhi == 1 {
                ex_zp = *ex.add(s - (nzu - 1));
                ey_zp = *ey.add(s - (nzu - 1));
            } else if bc_zhi == 2 {
                ex_zp = ex0;
                ey_zp = ey0;
            } else if bc_zhi == 3 {
                let w = s - (nzu - 1);
                ex_zp = bcz * *ex.add(w) + bsz * *exo.add(w);
                ey_zp = bcz * *ey.add(w) + bsz * *eyo.add(w);
            }
            let dxi = *idx_p.add(i);
            let dyi = *idy_p.add(j);
            let dzi = *idz_p.add(k);
            let curl_x = (ez_yp - ez0) * dyi - (ey_zp - ey0) * dzi;
            let curl_y = (ex_zp - ex0) * dzi - (ez_xp - ez0) * dxi;
            let curl_z = (ey_xp - ey0) * dxi - (ex_yp - ex0) * dyi;
            if ch_mode == 1 {
                *hx.add(s) -= *chx.add(s) * curl_x;
                *hy.add(s) -= *chy.add(s) * curl_y;
                *hz.add(s) -= *chz.add(s) * curl_z;
            } else {
                *hx.add(s) -= ch_scalar * curl_x;
                *hy.add(s) -= ch_scalar * curl_y;
                *hz.add(s) -= ch_scalar * curl_z;
            }
        }
    }

    /// E <- ca * E + cb * curl(H~)   (ca == 1 unless `lossy != 0`)
    ///
    /// The coefficients come from the per-cell arrays (`mat_mode == 255`) or
    /// from the material table `tab` through the per-cell index `mat`
    /// (`mat_mode` = index width, see `mat_index!`), which reads 0-4 bytes
    /// per cell instead of 12-24.
    ///
    /// Launch: block (bx, by, 1), grid (ceil(nz/bx), ceil(ny/by), nx).
    #[kernel]
    pub unsafe fn update_e(
        ex: *mut f32,
        ey: *mut f32,
        ez: *mut f32,
        hx: *const f32,
        hy: *const f32,
        hz: *const f32,
        cbx: *const f32,
        cby: *const f32,
        cbz: *const f32,
        cax: *const f32,
        cay: *const f32,
        caz: *const f32,
        lossy: u32,
        idx_d: *const f32,
        idy_d: *const f32,
        idz_d: *const f32,
        nx: u32,
        ny: u32,
        nz: u32,
        bc_ylo: u32,
        bc_zlo: u32,
        hxo: *const f32,
        hyo: *const f32,
        hzo: *const f32,
        bcy: f32,
        bsy: f32,
        bcz: f32,
        bsz: f32,
        mat: *const u32,
        tab: *const f32,
        mat_mode: u32,
    ) {
        let k = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let j = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as usize;
        let i = thread::blockIdx_z() as usize;
        let nyu = ny as usize;
        let nzu = nz as usize;
        if k >= nzu || j >= nyu || i >= nx as usize {
            return;
        }
        let plane = nyu * nzu;
        let s = (i + 1) * plane + j * nzu + k;
        unsafe {
            let hx0 = *hx.add(s);
            let hy0 = *hy.add(s);
            let hz0 = *hz.add(s);
            // -x neighbours: the ghost plane i = -1 is maintained by the host.
            let hy_xm = *hy.add(s - plane);
            let hz_xm = *hz.add(s - plane);
            // -y neighbours
            let mut hx_ym = 0.0f32;
            let mut hz_ym = 0.0f32;
            if j > 0 {
                hx_ym = *hx.add(s - nzu);
                hz_ym = *hz.add(s - nzu);
            } else if bc_ylo == 1 {
                hx_ym = *hx.add(s + (nyu - 1) * nzu);
                hz_ym = *hz.add(s + (nyu - 1) * nzu);
            } else if bc_ylo == 2 {
                hx_ym = -hx0;
                hz_ym = -hz0;
            } else if bc_ylo == 3 {
                // Bloch: H(-1) = H(ny-1) exp(-i k L)
                let w = s + (nyu - 1) * nzu;
                hx_ym = bcy * *hx.add(w) - bsy * *hxo.add(w);
                hz_ym = bcy * *hz.add(w) - bsy * *hzo.add(w);
            }
            // -z neighbours
            let mut hx_zm = 0.0f32;
            let mut hy_zm = 0.0f32;
            if k > 0 {
                hx_zm = *hx.add(s - 1);
                hy_zm = *hy.add(s - 1);
            } else if bc_zlo == 1 {
                hx_zm = *hx.add(s + (nzu - 1));
                hy_zm = *hy.add(s + (nzu - 1));
            } else if bc_zlo == 2 {
                hx_zm = -hx0;
                hy_zm = -hy0;
            } else if bc_zlo == 3 {
                let w = s + (nzu - 1);
                hx_zm = bcz * *hx.add(w) - bsz * *hxo.add(w);
                hy_zm = bcz * *hy.add(w) - bsz * *hyo.add(w);
            }
            let dxi = *idx_d.add(i);
            let dyi = *idy_d.add(j);
            let dzi = *idz_d.add(k);
            let curl_x = (hz0 - hz_ym) * dyi - (hy0 - hy_zm) * dzi;
            let curl_y = (hx0 - hx_zm) * dzi - (hz0 - hz_xm) * dxi;
            let curl_z = (hy0 - hy_xm) * dxi - (hx0 - hx_ym) * dyi;
            if mat_mode != 255 {
                // coefficients from the material table (see `step_fused`)
                let mt = tab.add(6 * mat_index!(mat, mat_mode, s));
                if lossy != 0 {
                    *ex.add(s) = *mt.add(3) * *ex.add(s) + *mt * curl_x;
                    *ey.add(s) = *mt.add(4) * *ey.add(s) + *mt.add(1) * curl_y;
                    *ez.add(s) = *mt.add(5) * *ez.add(s) + *mt.add(2) * curl_z;
                } else {
                    *ex.add(s) += *mt * curl_x;
                    *ey.add(s) += *mt.add(1) * curl_y;
                    *ez.add(s) += *mt.add(2) * curl_z;
                }
            } else if lossy != 0 {
                *ex.add(s) = *cax.add(s) * *ex.add(s) + *cbx.add(s) * curl_x;
                *ey.add(s) = *cay.add(s) * *ey.add(s) + *cby.add(s) * curl_y;
                *ez.add(s) = *caz.add(s) * *ez.add(s) + *cbz.add(s) * curl_z;
            } else {
                *ex.add(s) += *cbx.add(s) * curl_x;
                *ey.add(s) += *cby.add(s) * curl_y;
                *ez.add(s) += *cbz.add(s) * curl_z;
            }
        }
    }

    /// CPML correction for the E update inside one PML slab normal to `axis`.
    ///
    /// For axis `a` with cyclic transverse components `(b, c)`:
    /// `E_b` contains `-d_a H_c`, `E_c` contains `+d_a H_b`. The main update
    /// used kappa = 1 and no convolution; this kernel adds
    /// `-cb_b * ((1/kappa - 1) d_a H_c + psi_b)` and
    /// `+cb_c * ((1/kappa - 1) d_a H_b + psi_c)`.
    ///
    /// The slab is the box `[x0, x0+lx) x [y0, y0+ly) x [z0, z0+lz)` (interior
    /// indices, no ghost offset); psi arrays are box-local, x-major. The
    /// per-position coefficient arrays (`pb`, `pa`, `pkinv`) are indexed by
    /// the position along `axis` relative to `a0`. Launch 1-D over lx*ly*lz.
    #[kernel]
    pub unsafe fn pml_e(
        eb: *mut f32,
        ec: *mut f32,
        hb: *const f32,
        hc: *const f32,
        cbb: *const f32,
        cbc: *const f32,
        psi_b: *mut f32,
        psi_c: *mut f32,
        pb: *const f32,
        pa: *const f32,
        pkinv: *const f32,
        id_a: *const f32,
        axis: u32,
        a0: u32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = lz as usize;
        let lyu = ly as usize;
        let total = (lx as usize) * lyu * lzu;
        if t >= total {
            return;
        }
        let kk = t % lzu;
        let jj = (t / lzu) % lyu;
        let ii = t / (lzu * lyu);
        let i = x0 as usize + ii;
        let j = y0 as usize + jj;
        let k = z0 as usize + kk;
        let nyu = ny as usize;
        let nzu = nz as usize;
        let s = ((i + 1) * nyu + j) * nzu + k;
        let (along, stride) = if axis == 0 {
            (i, nyu * nzu)
        } else if axis == 1 {
            (j, nzu)
        } else {
            (k, 1usize)
        };
        let q = along - a0 as usize;
        unsafe {
            let b = *pb.add(q);
            let a = *pa.add(q);
            let km1 = *pkinv.add(q) - 1.0;
            let inv = *id_a.add(along);
            // Low wall along y/z has no ghost; the PEC wall there has cb = 0.
            let has_prev = axis == 0 || along > 0;
            let hc_prev = if has_prev { *hc.add(s - stride) } else { 0.0 };
            let hb_prev = if has_prev { *hb.add(s - stride) } else { 0.0 };
            let d_hc = (*hc.add(s) - hc_prev) * inv;
            let d_hb = (*hb.add(s) - hb_prev) * inv;
            let pbv = b * *psi_b.add(t) + a * d_hc;
            let pcv = b * *psi_c.add(t) + a * d_hb;
            *psi_b.add(t) = pbv;
            *psi_c.add(t) = pcv;
            *eb.add(s) -= *cbb.add(s) * (km1 * d_hc + pbv);
            *ec.add(s) += *cbc.add(s) * (km1 * d_hb + pcv);
        }
    }

    /// CPML correction for the H update inside one PML slab normal to `axis`.
    ///
    /// `H_b` contains `+d_a E_c`, `H_c` contains `-d_a E_b`.
    #[kernel]
    pub unsafe fn pml_h(
        hb: *mut f32,
        hc: *mut f32,
        eb: *const f32,
        ec: *const f32,
        chb: *const f32,
        chc: *const f32,
        ch_scalar: f32,
        ch_mode: u32,
        psi_b: *mut f32,
        psi_c: *mut f32,
        pb: *const f32,
        pa: *const f32,
        pkinv: *const f32,
        id_a: *const f32,
        axis: u32,
        a0: u32,
        n_along: u32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = lz as usize;
        let lyu = ly as usize;
        let total = (lx as usize) * lyu * lzu;
        if t >= total {
            return;
        }
        let kk = t % lzu;
        let jj = (t / lzu) % lyu;
        let ii = t / (lzu * lyu);
        let i = x0 as usize + ii;
        let j = y0 as usize + jj;
        let k = z0 as usize + kk;
        let nyu = ny as usize;
        let nzu = nz as usize;
        let s = ((i + 1) * nyu + j) * nzu + k;
        let (along, stride) = if axis == 0 {
            (i, nyu * nzu)
        } else if axis == 1 {
            (j, nzu)
        } else {
            (k, 1usize)
        };
        let q = along - a0 as usize;
        unsafe {
            let b = *pb.add(q);
            let a = *pa.add(q);
            let km1 = *pkinv.add(q) - 1.0;
            let inv = *id_a.add(along);
            let has_next = axis == 0 || along + 1 < n_along as usize;
            let ec_next = if has_next { *ec.add(s + stride) } else { 0.0 };
            let eb_next = if has_next { *eb.add(s + stride) } else { 0.0 };
            let d_ec = (ec_next - *ec.add(s)) * inv;
            let d_eb = (eb_next - *eb.add(s)) * inv;
            let pbv = b * *psi_b.add(t) + a * d_ec;
            let pcv = b * *psi_c.add(t) + a * d_eb;
            *psi_b.add(t) = pbv;
            *psi_c.add(t) = pcv;
            let (cb_, cc_) = if ch_mode == 1 {
                (*chb.add(s), *chc.add(s))
            } else {
                (ch_scalar, ch_scalar)
            };
            *hb.add(s) += cb_ * (km1 * d_ec + pbv);
            *hc.add(s) -= cc_ * (km1 * d_eb + pcv);
        }
    }

    /// field[idx[e]] += coef[e] * amps[g(e)]      (1-D launch over n entries)
    ///
    /// `amps` points at this step's row of the source amplitude table;
    /// `g(e) = group[e]` when `grouped != 0` (sources whose entries carry
    /// different delays, e.g. TFSF faces parallel to the propagation axis),
    /// otherwise 0.
    #[kernel]
    pub unsafe fn inject(
        field: *mut f32,
        idx: *const u64,
        coef: *const f32,
        group: *const u32,
        n: u32,
        amps: *const f32,
        grouped: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let g = if grouped != 0 { *group.add(t) as usize } else { 0 };
            *field.add(s) += *coef.add(t) * *amps.add(g);
        }
    }

    /// field[idx[e]] += coef_re[e] * a_re[g] + coef_im[e] * a_im[g]
    ///
    /// Sources whose spatial profile carries a phase (oblique plane waves,
    /// complex mode profiles, focused beams): with a narrow-band carrier the
    /// delayed signal is `Re[c(r) A(t)]` with `A` the analytic signal.
    /// `amps` holds interleaved `(re, im)` pairs for this step.
    #[kernel]
    pub unsafe fn inject_complex(
        field: *mut f32,
        idx: *const u64,
        coef_re: *const f32,
        coef_im: *const f32,
        group: *const u32,
        n: u32,
        amps: *const f32,
        grouped: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let g = if grouped != 0 { *group.add(t) as usize } else { 0 };
            let ar = *amps.add(2 * g);
            let ai = *amps.add(2 * g + 1);
            *field.add(s) += *coef_re.add(t) * ar + *coef_im.add(t) * ai;
        }
    }

    /// Running DFT over a box: re/im[f][cell] += field * (cos, sin)[f].
    ///
    /// `phasor` points at this step's row of the phasor table (2*nfreq floats,
    /// already weighted by dt and the apodization window and carrying the
    /// sign convention). Box is interior-indexed; 1-D launch over the box.
    #[kernel]
    pub unsafe fn dft_accumulate(
        field: *const f32,
        re: *mut f32,
        im: *mut f32,
        phasor: *const f32,
        nfreq: u32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = lz as usize;
        let lyu = ly as usize;
        let ncell = (lx as usize) * lyu * lzu;
        if t >= ncell {
            return;
        }
        let kk = t % lzu;
        let jj = (t / lzu) % lyu;
        let ii = t / (lzu * lyu);
        let s = ((x0 as usize + ii + 1) * ny as usize + y0 as usize + jj) * nz as usize
            + z0 as usize
            + kk;
        unsafe {
            let v = *field.add(s);
            let mut f = 0usize;
            while f < nfreq as usize {
                let c = *phasor.add(2 * f);
                let sn = *phasor.add(2 * f + 1);
                let o = f * ncell + t;
                *re.add(o) += v * c;
                *im.add(o) += v * sn;
                f += 1;
            }
        }
    }

    /// out[t] = field[box(t)]   (1-D launch over the box)
    #[kernel]
    pub unsafe fn gather_box(
        field: *const f32,
        out: *mut f32,
        x0: u32,
        y0: u32,
        z0: u32,
        lx: u32,
        ly: u32,
        lz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = lz as usize;
        let lyu = ly as usize;
        let ncell = (lx as usize) * lyu * lzu;
        if t >= ncell {
            return;
        }
        let kk = t % lzu;
        let jj = (t / lzu) % lyu;
        let ii = t / (lzu * lyu);
        let s = ((x0 as usize + ii + 1) * ny as usize + y0 as usize + jj) * nz as usize
            + z0 as usize
            + kk;
        unsafe {
            *out.add(t) = *field.add(s);
        }
    }

    /// dst[t] = factor * src[t]   (ghost-plane fills: periodic, mirror, antisymmetric)
    #[kernel]
    pub unsafe fn scale_copy(dst: *mut f32, src: *const f32, n: u64, factor: f32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *dst.add(t) = factor * *src.add(t);
        }
    }

    /// Bloch ghost fill: dst = cr * src_self + ci * src_other
    /// (real part: cr = cos, ci = -sin ; imaginary part: cr = cos, ci = +sin)
    #[kernel]
    pub unsafe fn bloch_copy(
        dst: *mut f32,
        src_self: *const f32,
        src_other: *const f32,
        n: u64,
        cr: f32,
        ci: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *dst.add(t) = cr * *src_self.add(t) + ci * *src_other.add(t);
        }
    }

    /// Dispersive media, step 1 (before the E update): advance every pole's
    /// polarization `P^{n+1} = c0 P^n + c1 P^{n-1} + c2 E^n` and store
    /// `sum_poles(P^{n+1} - P^n)` in `dp`.
    ///
    /// `p`/`pprev` are `[pole][entry]`; `coefs` is `[material][pole][3]`.
    #[kernel]
    pub unsafe fn disp_update(
        field: *const f32,
        idx: *const u64,
        matid: *const u32,
        p: *mut f32,
        pprev: *mut f32,
        coefs: *const f32,
        dp: *mut f32,
        npole: u32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let nu = n as usize;
        if t >= nu {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let e = *field.add(s);
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let mut acc = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                let o = q * nu + t;
                let p0 = *p.add(o);
                let pn = *c * p0 + *c.add(1) * *pprev.add(o) + *c.add(2) * e;
                acc += pn - p0;
                *pprev.add(o) = p0;
                *p.add(o) = pn;
                q += 1;
            }
            *dp.add(t) = acc;
        }
    }

    /// Dispersive media, step 2 (after the E update): E -= scale * dP.
    #[kernel]
    pub unsafe fn disp_apply(field: *mut f32, idx: *const u64, dp: *const f32, scale: *const f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            *field.add(s) -= *scale.add(t) * *dp.add(t);
        }
    }

    /// Magnetic conductivity: H~ *= da on the lossy cells of one component,
    /// launched before `update_h` (which then subtracts ch curl E with the
    /// lossy `ch`). Only plans with magnetic conductivity launch it.
    #[kernel]
    pub unsafe fn hloss_scale(field: *mut f32, idx: *const u64, da: *const f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            *field.add(s) *= *da.add(t);
        }
    }

    /// Dispersive media, dense form, over the storage range [s0, s0 + n):
    /// the same update `step_fused` applies inline, for the cells whose E
    /// update is deferred (local x-plane 0 with a periodic or neighbour low
    /// ghost). `dm` is the per-cell dispersive material (0 = none, else
    /// id + 1; `dmu != 0` = that value for every cell, no array), `ea` holds
    /// E^n and `eb` the linear E^{n+1}; `p` = P^n and `pprev` = P^{n-1} are
    /// `[pole][storage cell]`, and P^{n+1} is written over P^{n-1} (the host
    /// swaps the two arrays every step).
    #[kernel]
    pub unsafe fn disp_dense(
        ea: *const f32,
        eb: *mut f32,
        dm: *const u8,
        p: *const f32,
        pprev: *mut f32,
        coefs: *const f32,
        scale: *const f32,
        npole: u32,
        s0: u64,
        n: u64,
        len: u64,
        dmu: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = s0 as usize + t;
            let mid = if dmu != 0 { dmu as usize } else { *dm.add(s) as usize };
            if mid == 0 {
                return;
            }
            let m = mid - 1;
            let e = *ea.add(s);
            let np = npole as usize;
            let mut acc = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                let o = q * (len as usize) + s;
                let p0 = *p.add(o);
                let pn = *c * p0 + *c.add(1) * *pprev.add(o) + *c.add(2) * e;
                acc += pn - p0;
                *pprev.add(o) = pn;
                q += 1;
            }
            *eb.add(s) -= *scale.add(m) * acc;
        }
    }

    /// Nonlinear media, step 1 (before the E update): save E^n at the listed cells.
    #[kernel]
    pub unsafe fn nl_save(field: *const f32, idx: *const u64, prev: *mut f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *prev.add(t) = *field.add(*idx.add(t) as usize);
        }
    }

    /// Nonlinear media, step 2 (after the linear E update): instantaneous
    /// chi2/chi3 polarization P/eps0 = chi2 E^2 + chi3 E^3 (per component),
    /// solved implicitly. With d = E_lin^{n+1} - E^n and u = E^{n+1} - E^n,
    ///   u + c2 (2 E u + u^2) + c3 (3 E^2 u + 3 E u^2 + u^3) = d,
    /// c2 = chi2/eps, c3 = chi3/eps; Newton from the linearised guess.
    #[kernel]
    pub unsafe fn nl_apply(field: *mut f32, idx: *const u64, prev: *const f32, c2: *const f32, c3: *const f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let e = *prev.add(t);
            let d = *field.add(s) - e;
            let a = *c2.add(t);
            let b = *c3.add(t);
            let mut u = d / (1.0 + 2.0 * a * e + 3.0 * b * e * e);
            let mut it = 0;
            while it < 4 {
                let f = u + a * (2.0 * e * u + u * u) + b * (3.0 * e * e * u + 3.0 * e * u * u + u * u * u) - d;
                let fp = 1.0 + a * (2.0 * e + 2.0 * u) + b * (3.0 * e * e + 6.0 * e * u + 3.0 * u * u);
                u -= f / fp;
                it += 1;
            }
            *field.add(s) = e + u;
        }
    }

    /// Off-diagonal tensors, step 1 (after the linear E update): over the
    /// storage range [s0, s0 + n), dD = (E_lin - ca E^n) * cdt / cb, i.e.
    /// c dt curl H~ recovered from the diagonal update (`en` holds E^n).
    #[kernel]
    pub unsafe fn tensor_dd(
        field: *const f32,
        dd: *mut f32,
        en: *const f32,
        cb: *const f32,
        ca: *const f32,
        has_ca: u32,
        cdt: f32,
        s0: u64,
        n: u64,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = s0 as usize + t;
            let b = *cb.add(s);
            let a = if has_ca != 0 { *ca.add(s) } else { 1.0 };
            *dd.add(t) = if b != 0.0 { (*field.add(s) - a * *en.add(t)) * (cdt / b) } else { 0.0 };
        }
    }

    /// Off-diagonal permittivity tensors, step 2: E_a += c1 <dD_b> + c2 <dD_c>
    /// with b = a+1, c = a+2 (mod 3), c1 = kappa_ab, c2 = kappa_ac (kappa =
    /// eps^-1), and <.> the average over the four E_b (E_c) Yee positions
    /// around the E_a position (index offsets {0, +e_a} x {0, -e_b}). `dd1`,
    /// `dd2` hold dD_b, dD_c over storage planes [p0, p1); `wrap` bit 0/1/2 =
    /// periodic x/y/z.
    #[kernel]
    pub unsafe fn tensor_apply(
        field: *mut f32,
        idx: *const u64,
        c1: *const f32,
        c2: *const f32,
        dd1: *const f32,
        dd2: *const f32,
        m1: *const f32,
        m2: *const f32,
        en1: *const f32,
        en2: *const f32,
        lossy: u32,
        en_own: *const f32,
        opm: *const u32,
        oscale: *const f32,
        optab: *const f32,
        op_np: u32,
        ost: *mut f32,
        osp: *mut f32,
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
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
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
                        acc2 += *dd2.add(o);
                        if lossy != 0 {
                            ae2 += *en2.add(o);
                        }
                    } else {
                        acc1 += *dd1.add(o);
                        if lossy != 0 {
                            ae1 += *en1.add(o);
                        }
                    }
                }
                q += 1;
            }
            let mut v = *c1.add(t) * acc1 + *c2.add(t) * acc2;
            if lossy != 0 {
                v += *m1.add(t) * ae1 + *m2.add(t) * ae2;
            }
            *field.add(s) += 0.25 * v;
            // oriented poles: s^{n+1} = c0 s + c1 s_prev + c2 (u . E^n),
            // E_a -= u_a ds / eps_inf
            if op_np != 0 {
                let row = *opm.add(t);
                if row != u32::MAX {
                    let own = *en_own.add(((pi - lo) * nyi * nzi + j * nzi + k) as usize);
                    let eb1 = 0.25 * ae1;
                    let eb2 = 0.25 * ae2;
                    let (ex, ey, ez) = if a == 0 { (own, eb1, eb2) } else if a == 1 { (eb2, own, eb1) } else { (eb1, eb2, own) };
                    let np = op_np as usize;
                    let nn = n as usize;
                    let mut ds = 0.0f32;
                    let mut pp = 0usize;
                    while pp < np {
                        let tb = optab.add(((row as usize) * np + pp) * 6);
                        let drive = *tb.add(3) * ex + *tb.add(4) * ey + *tb.add(5) * ez;
                        let o = pp * nn + t;
                        let s0 = *ost.add(o);
                        let sn = *tb * s0 + *tb.add(1) * *osp.add(o) + *tb.add(2) * drive;
                        *osp.add(o) = s0;
                        *ost.add(o) = sn;
                        ds += *tb.add(3 + a as usize) * (sn - s0);
                        pp += 1;
                    }
                    *field.add(s) -= *oscale.add(t) * ds;
                }
            }
        }
    }

    /// Per-(j,k) column sums of the squared fields over the interior x range:
    /// out[j*nz + k] = sum_i (E^2 + H~^2). Used for energy-decay shutoff.
    #[kernel]
    pub unsafe fn column_energy(
        ex: *const f32,
        ey: *const f32,
        ez: *const f32,
        hx: *const f32,
        hy: *const f32,
        hz: *const f32,
        out: *mut f32,
        nx: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let plane = (ny as usize) * (nz as usize);
        if t >= plane {
            return;
        }
        let mut acc = 0.0f32;
        let mut i = 0usize;
        unsafe {
            while i < nx as usize {
                let s = (i + 1) * plane + t;
                let a = *ex.add(s);
                let b = *ey.add(s);
                let c = *ez.add(s);
                let d = *hx.add(s);
                let e = *hy.add(s);
                let f = *hz.add(s);
                acc += a * a + b * b + c * c + d * d + e * e + f * f;
                i += 1;
            }
            *out.add(t) = acc;
        }
    }
    // ======================================================================
    // Fused leapfrog step (ping-pong A -> B)
    //
    // One kernel computes H^{n+1/2} and E^{n+1} for a whole step. Blocks of
    // bk (k) x bj (j) threads (the launch's block shape)
    // march along x over a chunk of planes; at each plane they compute the
    // new H for their tile plus a one-cell halo on the low j and k sides
    // (redundant recomputation instead of a grid-wide sync), stage it in
    // shared memory, then compute the new E. Fields are double-buffered (read A, write B), and so is the
    // H-side CPML state, so no block can observe another block's
    // half-updated data. E-side CPML state is updated in place (each E cell
    // is owned by exactly one thread).
    //
    // Materials are a per-cell index (u8/u16/u32, or none for a uniform
    // slab) into a coefficient table [cbx, cby, cbz, cax, cay, caz], which
    // cuts DRAM traffic from ~84 to 48-52 bytes per cell per step versus the
    // split kernels.
    //
    // What limits the throughput (measured on RTX PRO 6000 / sm_120 and
    // GB10 / sm_121; the kernel is not purely bandwidth-bound: dropping the
    // 4-byte material index alone gained only ~2 %), and the choices that
    // follow from it:
    // * Halo work per warp. The low-j halo row is computed by thread row 0
    //   and the low-k halo column by the first bj threads of the block, so
    //   both are whole-warp passes. (One halo cell per row, done by that
    //   row's first thread, costs every warp an extra pass for one lane.)
    // * Tile shape. Tiles long in z (the contiguous direction: 128-256
    //   cells) are faster than 32-wide ones; they must stay aligned to the
    //   warp size, since tiles of bk - 1 cells with a dedicated halo thread
    //   column were slower on GB10.
    // * Locality in x. Blocks that march far along x drift apart and the
    //   planes they touch stop sharing cache: chunks of 8-16 planes are
    //   faster than 64 although each chunk recomputes one warm-up plane.
    // * Registers. All cell indices are 32-bit, which takes the kernel from
    //   ~100 registers to 63, so 1024-thread blocks fit an SM. Forcing fewer
    //   registers with launch bounds (more blocks per SM) spills and is
    //   slower on both GPUs.
    // ======================================================================

    /// Fused step kernels (`step_fused`, and `step_fused_bloch` for complex
    /// fields with Bloch boundaries of non-zero wave vector). Launch: block
    /// (bk, bj, 1) = the (z, y) tile, grid
    /// (ceil(nz/bk), ceil(ny/bj), ceil(nx/chunk)), dynamic shared memory
    /// `24 * (bk + 1) * (bj + 1)` bytes (two staging buffers of three
    /// components, used on alternating planes so one barrier per plane is
    /// enough). With `npole != 0` the dense dispersive update runs inline
    /// (`dm*`, `p*a`/`p*b`, `dcoefs`, `sc*`, `dmu`: see `disp_dense`; bits
    /// 24..26 of `dmu` enable the x/y/z component, byte a is its uniform
    /// material byte or 0). `bc` packs the y/z boundary codes
    /// (ylo | yhi<<4 | zlo<<8 | zhi<<12); `ghost_lo` is the low-x ghost rule
    /// (0 zero, 1 deferred: skip E at local plane 0, 2 PMC mirror);
    /// `pml_*` pack (n_lo | n_hi << 16) per axis; `pr*` are per-position
    /// profiles [pos][6] = (be, ae, kinv_e, bh, ah, kinv_h). `mat_mode` is
    /// the material index width (see `mat_index!`).
    ///
    /// The Bloch variant runs once per part (real, imaginary) of the complex
    /// field: `e?o`/`h?o`/`so?_r` are the other part's current fields and
    /// H-side CPML state, and a y/z boundary code 3 wraps with the phase
    /// `(bc?, bs?)` = (cos, -/+ sin) of k L: E(n) = bc E(0) + bs E_other(0),
    /// H(-1) = bc H(n-1) - bs H_other(n-1). Both parts read only current
    /// (A) buffers, so the two launches are independent. The plain variant
    /// takes the same arguments and ignores these.
    macro_rules! fused_kernel {
        ($d:tt $name:ident, $bloch:tt) => {
            #[kernel]
            pub unsafe fn $name(
                exa: *const f32,
                eya: *const f32,
                eza: *const f32,
                hxa: *const f32,
                hya: *const f32,
                hza: *const f32,
                exb: *mut f32,
                eyb: *mut f32,
                ezb: *mut f32,
                hxb: *mut f32,
                hyb: *mut f32,
                hzb: *mut f32,
                mat: *const u32,
                tab: *const f32,
                lossy: u32,
                ch: f32,
                ipx: *const f32,
                ipy: *const f32,
                ipz: *const f32,
                idx_: *const f32,
                idy_: *const f32,
                idz_: *const f32,
                nx: u32,
                ny: u32,
                nz: u32,
                bc: u32,
                ghost_lo: u32,
                chunk: u32,
                pml_x: u32,
                pml_y: u32,
                pml_z: u32,
                prx: *const f32,
                pry: *const f32,
                prz: *const f32,
                sex: *mut f32,
                sey: *mut f32,
                sez: *mut f32,
                shx_r: *const f32,
                shy_r: *const f32,
                shz_r: *const f32,
                shx_w: *mut f32,
                shy_w: *mut f32,
                shz_w: *mut f32,
                mat_mode: u32,
                dmx: *const u8,
                dmy: *const u8,
                dmz: *const u8,
                pxa: *const f32,
                pya: *const f32,
                pza: *const f32,
                pxb: *mut f32,
                pyb: *mut f32,
                pzb: *mut f32,
                dcoefs: *const f32,
                scx: *const f32,
                scy: *const f32,
                scz: *const f32,
                npole: u32,
                dmu: u32,
                exo: *const f32,
                eyo: *const f32,
                ezo: *const f32,
                hxo: *const f32,
                hyo: *const f32,
                hzo: *const f32,
                sox_r: *const f32,
                soy_r: *const f32,
                soz_r: *const f32,
                bcy: f32,
                bsy: f32,
                bcz: f32,
                bsz: f32,
            ) {
                // Index arithmetic is 32-bit on purpose (the host guarantees fewer
                // than 2^32 storage cells): 64-bit indices double the register
                // footprint, which limits how many blocks fit on an SM.
                let bk = thread::blockDim_x();
                let bj = thread::blockDim_y();
                let sw = bk + 1;
                let sn = (sw * (bj + 1)) as usize;
                let smem: *mut f32 = DynamicSharedArray::<f32>::get();

                let tk = thread::threadIdx_x();
                let tj = thread::threadIdx_y();
                let k0 = thread::blockIdx_x() * bk;
                let j0 = thread::blockIdx_y() * bj;
                let k = k0 + tk;
                let j = j0 + tj;
                // the first bj threads of the block also compute the low-k
                // halo column, one cell each (see the task loop)
                let lin = tj * bk + tk;
                let plane = ny * nz;
                let i0 = thread::blockIdx_z() * chunk;
                let i1 = if i0 + chunk > nx { nx } else { i0 + chunk };
                let active = k < nz && j < ny;
                let bc_ylo = bc & 15;
                let bc_yhi = (bc >> 4) & 15;
                let bc_zlo = (bc >> 8) & 15;
                let bc_zhi = (bc >> 12) & 15;
                let nlx = pml_x & 0xffff;
                let nhx = pml_x >> 16;
                let nly = pml_y & 0xffff;
                let nhy = pml_y >> 16;
                let nlz = pml_z & 0xffff;
                let nhz = pml_z >> 16;
                let cnt_x = ((nlx + nhx) * plane) as usize;
                let npy = nly + nhy;
                let cnt_y = (nx * npy * nz) as usize;
                let npz = nlz + nhz;
                let cnt_z = (nx * ny * npz) as usize;
                let r = tj + 1;
                let c = tk + 1;

                // H^{n+1/2} at (i, jj, kk) from A buffers; writes H-side CPML state
                // when `wr`. `cy`/`cz` are Ey/Ez at the cell when `$d carry` (the own
                // cell: the previous plane already loaded them as its +x neighbours).
                macro_rules! hcell {
                    ($d oth:expr, $d i:expr, $d jj:expr, $d kk:expr, $d wr:expr, $d carry:expr, $d cy:expr, $d cz:expr) => {{
                        // field set of the cell (`oth`: the partner part, Bloch only)
                        // and the set its Bloch wraps mix in, with the sign of the
                        // sine term
                        let oth: bool = $bloch && $d oth;
                        let (exa, eya, eza, hxa, hya, hza, shx_r, shy_r, shz_r, exw, eyw, ezw, sg) = if oth {
                            (exo, eyo, ezo, hxo, hyo, hzo, sox_r, soy_r, soz_r, exa, eya, eza, -1.0f32)
                        } else {
                            (exa, eya, eza, hxa, hya, hza, shx_r, shy_r, shz_r, exo, eyo, ezo, 1.0f32)
                        };
                        let ii: u32 = $d i;
                        let jj: u32 = $d jj;
                        let kk: u32 = $d kk;
                        let s = ((ii + 1) * ny + jj) * nz + kk;
                        let su = s as usize;
                        let ex0 = *exa.add(su);
                        let (ey0, ez0) = if $d carry { ($d cy, $d cz) } else { (*eya.add(su), *eza.add(su)) };
                        let sxp = (s + plane) as usize;
                        let ey_xp = *eya.add(sxp);
                        let ez_xp = *eza.add(sxp);
                        let mut ex_yp = 0.0f32;
                        let mut ez_yp = 0.0f32;
                        if jj + 1 < ny {
                            ex_yp = *exa.add((s + nz) as usize);
                            ez_yp = *eza.add((s + nz) as usize);
                        } else if bc_yhi == 1 {
                            ex_yp = *exa.add((s - (ny - 1) * nz) as usize);
                            ez_yp = *eza.add((s - (ny - 1) * nz) as usize);
                        } else if bc_yhi == 2 {
                            ex_yp = ex0;
                            ez_yp = ez0;
                        } else if $bloch && bc_yhi == 3 {
                            // Bloch: E(ny) = E(0) exp(i k L)
                            let w = (s - (ny - 1) * nz) as usize;
                            ex_yp = bcy * *exa.add(w) + sg * bsy * *exw.add(w);
                            ez_yp = bcy * *eza.add(w) + sg * bsy * *ezw.add(w);
                        }
                        let mut ex_zp = 0.0f32;
                        let mut ey_zp = 0.0f32;
                        if kk + 1 < nz {
                            ex_zp = *exa.add((s + 1) as usize);
                            ey_zp = *eya.add((s + 1) as usize);
                        } else if bc_zhi == 1 {
                            ex_zp = *exa.add((s - (nz - 1)) as usize);
                            ey_zp = *eya.add((s - (nz - 1)) as usize);
                        } else if bc_zhi == 2 {
                            ex_zp = ex0;
                            ey_zp = ey0;
                        } else if $bloch && bc_zhi == 3 {
                            let w = (s - (nz - 1)) as usize;
                            ex_zp = bcz * *exa.add(w) + sg * bsz * *exw.add(w);
                            ey_zp = bcz * *eya.add(w) + sg * bsz * *eyw.add(w);
                        }
                        let dxi = *ipx.add(ii as usize);
                        let dyi = *ipy.add(jj as usize);
                        let dzi = *ipz.add(kk as usize);
                        let dey_x = (ey_xp - ey0) * dxi;
                        let dez_x = (ez_xp - ez0) * dxi;
                        let dex_y = (ex_yp - ex0) * dyi;
                        let dez_y = (ez_yp - ez0) * dyi;
                        let dex_z = (ex_zp - ex0) * dzi;
                        let dey_z = (ey_zp - ey0) * dzi;
                        let mut hx = *hxa.add(su) - ch * (dez_y - dey_z);
                        let mut hy = *hya.add(su) - ch * (dex_z - dez_x);
                        let mut hz = *hza.add(su) - ch * (dey_x - dex_y);
                        if ii < nlx || ii + nhx >= nx {
                            let ip = if ii < nlx { ii } else { nlx + ii + nhx - nx };
                            let o = ((ip * ny + jj) * nz + kk) as usize;
                            let pr = prx.add((6 * ii) as usize);
                            let b = *pr.add(3);
                            let a = *pr.add(4);
                            let km1 = *pr.add(5) - 1.0;
                            let pb = b * *shx_r.add(o) + a * dez_x;
                            let pc = b * *shx_r.add(o + cnt_x) + a * dey_x;
                            if $d wr {
                                *shx_w.add(o) = pb;
                                *shx_w.add(o + cnt_x) = pc;
                            }
                            hy += ch * (km1 * dez_x + pb);
                            hz -= ch * (km1 * dey_x + pc);
                        }
                        if jj < nly || jj + nhy >= ny {
                            let jp = if jj < nly { jj } else { nly + jj + nhy - ny };
                            let o = ((ii * npy + jp) * nz + kk) as usize;
                            let pr = pry.add((6 * jj) as usize);
                            let b = *pr.add(3);
                            let a = *pr.add(4);
                            let km1 = *pr.add(5) - 1.0;
                            let pb = b * *shy_r.add(o) + a * dex_y;
                            let pc = b * *shy_r.add(o + cnt_y) + a * dez_y;
                            if $d wr {
                                *shy_w.add(o) = pb;
                                *shy_w.add(o + cnt_y) = pc;
                            }
                            hz += ch * (km1 * dex_y + pb);
                            hx -= ch * (km1 * dez_y + pc);
                        }
                        if kk < nlz || kk + nhz >= nz {
                            let kp = if kk < nlz { kk } else { nlz + kk + nhz - nz };
                            let o = ((ii * ny + jj) * npz + kp) as usize;
                            let pr = prz.add((6 * kk) as usize);
                            let b = *pr.add(3);
                            let a = *pr.add(4);
                            let km1 = *pr.add(5) - 1.0;
                            let pb = b * *shz_r.add(o) + a * dey_z;
                            let pc = b * *shz_r.add(o + cnt_z) + a * dex_z;
                            if $d wr {
                                *shz_w.add(o) = pb;
                                *shz_w.add(o + cnt_z) = pc;
                            }
                            hx += ch * (km1 * dey_z + pb);
                            hy -= ch * (km1 * dex_z + pc);
                        }
                        (hx, hy, hz, ex0, ey0, ez0, su, ey_xp, ez_xp)
                    }};
                }

                // Dispersive poles of one E component at the own cell (dense form):
                // P^{n+1} = c0 P^n + c1 P^{n-1} + c2 E^n is written over P^{n-1}
                // (the host swaps the two arrays every step); returns
                // sum_poles(P^{n+1} - P^n) / eps_inf.
                macro_rules! dcomp {
                    ($d bit:expr, $d dm:expr, $d pa:expr, $d pb:expr, $d sc:expr, $d e:expr, $d s:expr) => {{
                        let mut dv = 0.0f32;
                        if (dmu >> (24 + $d bit)) & 1 != 0 {
                            let u = (dmu >> (8 * $d bit)) & 255;
                            let mid = if u != 0 { u } else { *$d dm.add($d s) as u32 };
                            if mid != 0 {
                                let m = mid - 1;
                                let dlen = (nx as usize + 2) * plane as usize;
                                let mut acc = 0.0f32;
                                let mut q = 0u32;
                                while q < npole {
                                    let cc = dcoefs.add(((m * npole + q) * 3) as usize);
                                    let o = q as usize * dlen + $d s;
                                    let p0 = *$d pa.add(o);
                                    let pn = *cc * p0 + *cc.add(1) * *$d pb.add(o) + *cc.add(2) * $d e;
                                    acc += pn - p0;
                                    *$d pb.add(o) = pn;
                                    q += 1;
                                }
                                dv = *$d sc.add(m as usize) * acc;
                            }
                        }
                        dv
                    }};
                }

                unsafe {
                    let mut hy_prev = 0.0f32;
                    let mut hz_prev = 0.0f32;
                    // Ey/Ez of the own cell at the next plane (valid when `have`)
                    let mut cy = 0.0f32;
                    let mut cz = 0.0f32;
                    let mut have = false;
                    // halo source rows/columns (u32::MAX = use the boundary rule)
                    let wrap_y = bc_ylo == 1 || ($bloch && bc_ylo == 3);
                    let wrap_z = bc_zlo == 1 || ($bloch && bc_zlo == 3);
                    let jh = if j0 > 0 { j0 - 1 } else if wrap_y { ny - 1 } else { u32::MAX };
                    let kh = if k0 > 0 { k0 - 1 } else if wrap_z { nz - 1 } else { u32::MAX };
                    // Bloch low wraps: H(-1) = H(n-1) exp(-i k L) needs the new H of
                    // both parts at the far row/column
                    let by = $bloch && bc_ylo == 3 && j0 == 0;
                    let bz = $bloch && bc_zlo == 3 && k0 == 0;
                    // halo duties: the low-j row goes to thread row 0, the
                    // low-k column to the first bj threads of the block, so
                    // each is computed by whole warps (one lane per warp, as
                    // with "first thread of each row", wastes a warp pass
                    // per row)
                    let halo_j = tj == 0 && k < nz && jh != u32::MAX;
                    let halo_k = lin < bj && j0 + lin < ny && kh != u32::MAX;
                    let ntask = if $bloch { 5u32 } else { 3u32 };
                    // start one plane early to obtain H^{n+1/2}(i0 - 1) for the x-derivative
                    let mut i = if i0 > 0 { i0 - 1 } else { i0 };
                    while i < i1 {
                        let warm = i + 1 == i0;
                        // two staging buffers on alternating planes: a thread can only
                        // write plane i + 2 into the buffer of plane i after the
                        // barrier of plane i + 1, i.e. after every read of plane i
                        let shx = smem.add(if (i & 1) != 0 { 3 * sn } else { 0 });
                        let shy = shx.add(sn);
                        let shz = shy.add(sn);
                        let mut ex0 = 0.0f32;
                        let mut ey0 = 0.0f32;
                        let mut ez0 = 0.0f32;
                        let mut s = 0usize;
                        let mut own = (0.0f32, 0.0f32, 0.0f32);
                        // tasks: 0 = own cell, 1 = low-j halo, 2 = low-k halo,
                        // 3 / 4 = the partner part's share of a Bloch low-j / low-k halo
                        let mut task = 0u32;
                        while task < ntask {
                            let (go, jj, kk, slot, wgt) = if task == 0 {
                                (active, j, k, r * sw + c, 1.0f32)
                            } else if task == 1 {
                                (!warm && halo_j, jh, k, c, if by { bcy } else { 1.0f32 })
                            } else if task == 2 {
                                (!warm && halo_k, j0 + lin, kh, (lin + 1) * sw, if bz { bcz } else { 1.0f32 })
                            } else if task == 3 {
                                (!warm && halo_j && by, jh, k, c, -bsy)
                            } else {
                                (!warm && halo_k && bz, j0 + lin, kh, (lin + 1) * sw, -bsz)
                            };
                            if go {
                                let h = hcell!(task >= 3, i, jj, kk, task == 0 && !warm, task == 0 && have, cy, cz);
                                if task == 0 {
                                    own = (h.0, h.1, h.2);
                                    ex0 = h.3;
                                    ey0 = h.4;
                                    ez0 = h.5;
                                    s = h.6;
                                    cy = h.7;
                                    cz = h.8;
                                    have = true;
                                }
                                if !warm {
                                    let q = slot as usize;
                                    if task >= 3 {
                                        *shx.add(q) += wgt * h.0;
                                        *shy.add(q) += wgt * h.1;
                                        *shz.add(q) += wgt * h.2;
                                    } else if $bloch {
                                        *shx.add(q) = wgt * h.0;
                                        *shy.add(q) = wgt * h.1;
                                        *shz.add(q) = wgt * h.2;
                                    } else {
                                        *shx.add(q) = h.0;
                                        *shy.add(q) = h.1;
                                        *shz.add(q) = h.2;
                                    }
                                }
                            }
                            task += 1;
                        }
                        if warm {
                            hy_prev = own.1;
                            hz_prev = own.2;
                            i += 1;
                            continue;
                        }
                        if active {
                            *hxb.add(s) = own.0;
                            *hyb.add(s) = own.1;
                            *hzb.add(s) = own.2;
                        }
                        thread::sync_threads();
                        if active && !(i == 0 && ghost_lo == 1) {
                            let hx0 = own.0;
                            let hy0 = own.1;
                            let hz0 = own.2;
                            let (hy_xm, hz_xm) = if i > 0 {
                                (hy_prev, hz_prev)
                            } else if ghost_lo == 2 {
                                (-hy0, -hz0)
                            } else {
                                (0.0f32, 0.0f32)
                            };
                            let (hx_ym, hz_ym) = if j > 0 || wrap_y {
                                let q = ((r - 1) * sw + c) as usize;
                                (*shx.add(q), *shz.add(q))
                            } else if bc_ylo == 2 {
                                (-hx0, -hz0)
                            } else {
                                (0.0f32, 0.0f32)
                            };
                            let (hx_zm, hy_zm) = if k > 0 || wrap_z {
                                let q = (r * sw + c - 1) as usize;
                                (*shx.add(q), *shy.add(q))
                            } else if bc_zlo == 2 {
                                (-hx0, -hy0)
                            } else {
                                (0.0f32, 0.0f32)
                            };
                            let dxi = *idx_.add(i as usize);
                            let dyi = *idy_.add(j as usize);
                            let dzi = *idz_.add(k as usize);
                            let dhz_x = (hz0 - hz_xm) * dxi;
                            let dhy_x = (hy0 - hy_xm) * dxi;
                            let dhx_y = (hx0 - hx_ym) * dyi;
                            let dhz_y = (hz0 - hz_ym) * dyi;
                            let dhy_z = (hy0 - hy_zm) * dzi;
                            let dhx_z = (hx0 - hx_zm) * dzi;
                            let mt = tab.add(6 * mat_index!(mat, mat_mode, s));
                            let cbx = *mt;
                            let cby = *mt.add(1);
                            let cbz = *mt.add(2);
                            let mut exn;
                            let mut eyn;
                            let mut ezn;
                            if lossy != 0 {
                                exn = *mt.add(3) * ex0 + cbx * (dhz_y - dhy_z);
                                eyn = *mt.add(4) * ey0 + cby * (dhx_z - dhz_x);
                                ezn = *mt.add(5) * ez0 + cbz * (dhy_x - dhx_y);
                            } else {
                                exn = ex0 + cbx * (dhz_y - dhy_z);
                                eyn = ey0 + cby * (dhx_z - dhz_x);
                                ezn = ez0 + cbz * (dhy_x - dhx_y);
                            }
                            if i < nlx || i + nhx >= nx {
                                let ip = if i < nlx { i } else { nlx + i + nhx - nx };
                                let o = ((ip * ny + j) * nz + k) as usize;
                                let pr = prx.add((6 * i) as usize);
                                let b = *pr;
                                let a = *pr.add(1);
                                let km1 = *pr.add(2) - 1.0;
                                let pb = b * *sex.add(o) + a * dhz_x;
                                let pc = b * *sex.add(o + cnt_x) + a * dhy_x;
                                *sex.add(o) = pb;
                                *sex.add(o + cnt_x) = pc;
                                eyn -= cby * (km1 * dhz_x + pb);
                                ezn += cbz * (km1 * dhy_x + pc);
                            }
                            if j < nly || j + nhy >= ny {
                                let jp = if j < nly { j } else { nly + j + nhy - ny };
                                let o = ((i * npy + jp) * nz + k) as usize;
                                let pr = pry.add((6 * j) as usize);
                                let b = *pr;
                                let a = *pr.add(1);
                                let km1 = *pr.add(2) - 1.0;
                                let pb = b * *sey.add(o) + a * dhx_y;
                                let pc = b * *sey.add(o + cnt_y) + a * dhz_y;
                                *sey.add(o) = pb;
                                *sey.add(o + cnt_y) = pc;
                                ezn -= cbz * (km1 * dhx_y + pb);
                                exn += cbx * (km1 * dhz_y + pc);
                            }
                            if k < nlz || k + nhz >= nz {
                                let kp = if k < nlz { k } else { nlz + k + nhz - nz };
                                let o = ((i * ny + j) * npz + kp) as usize;
                                let pr = prz.add((6 * k) as usize);
                                let b = *pr;
                                let a = *pr.add(1);
                                let km1 = *pr.add(2) - 1.0;
                                let pb = b * *sez.add(o) + a * dhy_z;
                                let pc = b * *sez.add(o + cnt_z) + a * dhx_z;
                                *sez.add(o) = pb;
                                *sez.add(o + cnt_z) = pc;
                                exn -= cbx * (km1 * dhy_z + pb);
                                eyn += cby * (km1 * dhx_z + pc);
                            }
                            if npole != 0 {
                                exn -= dcomp!(0, dmx, pxa, pxb, scx, ex0, s);
                                eyn -= dcomp!(1, dmy, pya, pyb, scy, ey0, s);
                                ezn -= dcomp!(2, dmz, pza, pzb, scz, ez0, s);
                            }
                            *exb.add(s) = exn;
                            *eyb.add(s) = eyn;
                            *ezb.add(s) = ezn;
                        }
                        hy_prev = own.1;
                        hz_prev = own.2;
                        i += 1;
                    }
                }
            }
        };
    }
    fused_kernel!($ step_fused, false);
    fused_kernel!($ step_fused_bloch, true);

    /// Deferred E update of local x-plane 0 (fused mode, periodic or
    /// neighbour low ghost): E_b(0) from E_a(0) and the new H in B, whose low
    /// ghost plane has been filled after the fused kernel. One thread per
    /// (j, k).
    #[kernel]
    pub unsafe fn e_plane_fused(
        exa: *const f32,
        eya: *const f32,
        eza: *const f32,
        hxb: *const f32,
        hyb: *const f32,
        hzb: *const f32,
        exb: *mut f32,
        eyb: *mut f32,
        ezb: *mut f32,
        mat: *const u32,
        tab: *const f32,
        lossy: u32,
        idx_: *const f32,
        idy_: *const f32,
        idz_: *const f32,
        nx: u32,
        ny: u32,
        nz: u32,
        bc: u32,
        pml_x: u32,
        pml_y: u32,
        pml_z: u32,
        prx: *const f32,
        pry: *const f32,
        prz: *const f32,
        sex: *mut f32,
        sey: *mut f32,
        sez: *mut f32,
        mat_mode: u32,
        hxo: *const f32,
        hyo: *const f32,
        hzo: *const f32,
        bcy: f32,
        bsy: f32,
        bcz: f32,
        bsz: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize) + thread::threadIdx_x() as usize;
        let nxu = nx as usize;
        let nyu = ny as usize;
        let nzu = nz as usize;
        let plane = nyu * nzu;
        if t >= plane {
            return;
        }
        let j = t / nzu;
        let k = t % nzu;
        let i = 0usize;
        let bc_ylo = bc & 15;
        let bc_zlo = (bc >> 8) & 15;
        let nlx = (pml_x & 0xffff) as usize;
        let nhx = (pml_x >> 16) as usize;
        let nly = (pml_y & 0xffff) as usize;
        let nhy = (pml_y >> 16) as usize;
        let nlz = (pml_z & 0xffff) as usize;
        let nhz = (pml_z >> 16) as usize;
        let cnt_x = (nlx + nhx) * plane;
        let npy = nly + nhy;
        let cnt_y = nxu * npy * nzu;
        let npz = nlz + nhz;
        let cnt_z = nxu * nyu * npz;
        let s = plane + j * nzu + k;
        unsafe {
            let hx0 = *hxb.add(s);
            let hy0 = *hyb.add(s);
            let hz0 = *hzb.add(s);
            let hy_xm = *hyb.add(s - plane);
            let hz_xm = *hzb.add(s - plane);
            let (hx_ym, hz_ym) = if j > 0 {
                (*hxb.add(s - nzu), *hzb.add(s - nzu))
            } else if bc_ylo == 1 {
                (*hxb.add(s + (nyu - 1) * nzu), *hzb.add(s + (nyu - 1) * nzu))
            } else if bc_ylo == 2 {
                (-hx0, -hz0)
            } else if bc_ylo == 3 {
                // Bloch: H(-1) = H(ny-1) exp(-i k L), `h?o` = the other part
                let w = s + (nyu - 1) * nzu;
                (bcy * *hxb.add(w) - bsy * *hxo.add(w), bcy * *hzb.add(w) - bsy * *hzo.add(w))
            } else {
                (0.0f32, 0.0f32)
            };
            let (hx_zm, hy_zm) = if k > 0 {
                (*hxb.add(s - 1), *hyb.add(s - 1))
            } else if bc_zlo == 1 {
                (*hxb.add(s + nzu - 1), *hyb.add(s + nzu - 1))
            } else if bc_zlo == 2 {
                (-hx0, -hy0)
            } else if bc_zlo == 3 {
                let w = s + nzu - 1;
                (bcz * *hxb.add(w) - bsz * *hxo.add(w), bcz * *hyb.add(w) - bsz * *hyo.add(w))
            } else {
                (0.0f32, 0.0f32)
            };
            let dxi = *idx_.add(i);
            let dyi = *idy_.add(j);
            let dzi = *idz_.add(k);
            let dhz_x = (hz0 - hz_xm) * dxi;
            let dhy_x = (hy0 - hy_xm) * dxi;
            let dhx_y = (hx0 - hx_ym) * dyi;
            let dhz_y = (hz0 - hz_ym) * dyi;
            let dhy_z = (hy0 - hy_zm) * dzi;
            let dhx_z = (hx0 - hx_zm) * dzi;
            let mt = tab.add(6 * mat_index!(mat, mat_mode, s));
            let cbx = *mt;
            let cby = *mt.add(1);
            let cbz = *mt.add(2);
            let ex0 = *exa.add(s);
            let ey0 = *eya.add(s);
            let ez0 = *eza.add(s);
            let mut exn;
            let mut eyn;
            let mut ezn;
            if lossy != 0 {
                exn = *mt.add(3) * ex0 + cbx * (dhz_y - dhy_z);
                eyn = *mt.add(4) * ey0 + cby * (dhx_z - dhz_x);
                ezn = *mt.add(5) * ez0 + cbz * (dhy_x - dhx_y);
            } else {
                exn = ex0 + cbx * (dhz_y - dhy_z);
                eyn = ey0 + cby * (dhx_z - dhz_x);
                ezn = ez0 + cbz * (dhy_x - dhx_y);
            }
            if i < nlx || i + nhx >= nxu {
                let ip = if i < nlx { i } else { nlx + i + nhx - nxu };
                let o = (ip * nyu + j) * nzu + k;
                let pr = prx.add(6 * i);
                let b = *pr;
                let a = *pr.add(1);
                let km1 = *pr.add(2) - 1.0;
                let pb = b * *sex.add(o) + a * dhz_x;
                let pc = b * *sex.add(o + cnt_x) + a * dhy_x;
                *sex.add(o) = pb;
                *sex.add(o + cnt_x) = pc;
                eyn -= cby * (km1 * dhz_x + pb);
                ezn += cbz * (km1 * dhy_x + pc);
            }
            if j < nly || j + nhy >= nyu {
                let jp = if j < nly { j } else { nly + j + nhy - nyu };
                let o = (i * npy + jp) * nzu + k;
                let pr = pry.add(6 * j);
                let b = *pr;
                let a = *pr.add(1);
                let km1 = *pr.add(2) - 1.0;
                let pb = b * *sey.add(o) + a * dhx_y;
                let pc = b * *sey.add(o + cnt_y) + a * dhz_y;
                *sey.add(o) = pb;
                *sey.add(o + cnt_y) = pc;
                ezn -= cbz * (km1 * dhx_y + pb);
                exn += cbx * (km1 * dhz_y + pc);
            }
            if k < nlz || k + nhz >= nzu {
                let kp = if k < nlz { k } else { nlz + k + nhz - nzu };
                let o = (i * nyu + j) * npz + kp;
                let pr = prz.add(6 * k);
                let b = *pr;
                let a = *pr.add(1);
                let km1 = *pr.add(2) - 1.0;
                let pb = b * *sez.add(o) + a * dhy_z;
                let pc = b * *sez.add(o + cnt_z) + a * dhx_z;
                *sez.add(o) = pb;
                *sez.add(o + cnt_z) = pc;
                exn -= cbx * (km1 * dhy_z + pb);
                eyn += cby * (km1 * dhx_z + pc);
            }
            *exb.add(s) = exn;
            *eyb.add(s) = eyn;
            *ezb.add(s) = ezn;
        }
    }

}

// Kernels of the gradient engine (design updates, adjoint product, reverse
// pass): a separate module, compiled into the same PTX.
mod autodiff;
// Kernels of the 2-D solver (one-cell grids): likewise.
mod twod;
// Kernels of the material models with their own per-cell state (Raman chi3).
mod materials;

fn main() {
    // The kernels crate is only built for its PTX artifact. Running it prints
    // the kernel inventory so `cargo oxide run` is a quick smoke test.
    println!("lightsim-kernels: update_h update_e pml_e pml_h inject inject_complex dft_accumulate gather_box scale_copy bloch_copy disp_update disp_apply disp_dense column_energy step_fused step_fused_bloch e_plane_fused");
}
