//! Device kernels of the gradient engine (`crates/lightsim-cuda/src/autodiff.rs`).
//!
//! Kept apart from the stepping kernels of `main.rs`: nothing here runs in a
//! plain simulation. Same conventions (storage layout, raw pointers as
//! `u64`, 1-D launches over the entries or over the cells of a box).
//!
//! * `ad_gather_multi`, `ad_scatter_multi` — read / write all the boxes of
//!   a time monitor in one launch (objective samples, record and replay of
//!   the PML shell)
//! * `ad_scatter_box` — write a box of a field from a packed buffer (design
//!   coefficient updates)
//! * `ad_inject_scaled`, `ad_inject_complex_scaled` — the source injections
//!   with a factor (-1 removes the sources of a step during time reversal)
//! * `ad_dft_multi` — running DFT of every monitor box of a run in one
//!   launch (the plain solver launches `dft_accumulate` once per box)
//! * `ad_product` — adjoint product of two DFT fields
//! * `ad_accumulate` — per-step coefficient sums of the reverse pass
//! * `ad_disp_pre`, `ad_disp_post` — time reversal of dispersive cells
//! * `ad_nl_post`, `ad_nl_adj_pre`, `ad_nl_adj_post` — chi2 / chi3 cells:
//!   time reversal of the implicit step and the two factors its adjoint needs
//! * `ad_pml_phi`, `ad_pml_h`, `ad_pml_e` — the transposed CPML of the
//!   adjoint step (the forward CPML is not its own transpose)
//!
//! Rust's Enzyme-based autodiff cannot generate these: cuda-oxide's codegen
//! backend rejects the `autodiff` intrinsic (see docs/modules/autodiff.md),
//! so they are written by hand; the scalar formulas are those of
//! `lightsim_core::autodiff::cell`, which the `enzyme` feature of
//! lightsim-core checks against Enzyme on the CPU.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod ad_kernels {
    use super::*;

    /// field[box(t)] = buf[t]   (1-D launch over the box; inverse of `gather_box`)
    #[kernel]
    pub unsafe fn ad_scatter_box(
        field: *mut f32,
        buf: *const f32,
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
            *field.add(s) = *buf.add(t);
        }
    }

    /// field[idx[e]] += scale * coef[e] * amps[g(e)]   (as `inject`, with a factor)
    #[kernel]
    pub unsafe fn ad_inject_scaled(
        field: *mut f32,
        idx: *const u64,
        coef: *const f32,
        group: *const u32,
        n: u32,
        amps: *const f32,
        grouped: u32,
        scale: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let g = if grouped != 0 { *group.add(t) as usize } else { 0 };
            *field.add(s) += scale * (*coef.add(t) * *amps.add(g));
        }
    }

    /// field[idx[e]] += scale * (coef_re[e] * a_re[g] + coef_im[e] * a_im[g])
    #[kernel]
    pub unsafe fn ad_inject_complex_scaled(
        field: *mut f32,
        idx: *const u64,
        coef_re: *const f32,
        coef_im: *const f32,
        group: *const u32,
        n: u32,
        amps: *const f32,
        grouped: u32,
        scale: f32,
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
            *field.add(s) += scale * (*coef_re.add(t) * ar + *coef_im.add(t) * ai);
        }
    }

    /// Running DFT of all the monitor boxes of a run in one launch (1-D over
    /// the cells of all boxes together).
    ///
    /// `desc` holds 8 words per box: component (0..6 = Ex, Ey, Ez, Hx, Hy,
    /// Hz), x0, y0, z0, lx, ly, lz and the global index of the box's first
    /// cell (boxes in ascending order). The accumulators of all boxes share
    /// two arrays: box b starts at `nfreq * first(b)` and is laid out
    /// `[freq][cell]` like the per-box monitors. `phasor_e` / `phasor_h`
    /// point at this step's rows of the E-time and H-time phasor tables.
    #[kernel]
    pub unsafe fn ad_dft_multi(
        ex: *const f32,
        ey: *const f32,
        ez: *const f32,
        hx: *const f32,
        hy: *const f32,
        hz: *const f32,
        re: *mut f32,
        im: *mut f32,
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
            let first = *d.add(7) as usize;
            let local = t - first;
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
            let v = *field.add(s);
            let base = (nfreq as usize) * first + local;
            let mut f = 0usize;
            while f < nfreq as usize {
                let c = *phasor.add(2 * f);
                let sn = *phasor.add(2 * f + 1);
                let o = base + f * ncell;
                *re.add(o) += v * c;
                *im.add(o) += v * sn;
                f += 1;
            }
        }
    }

    /// out[t] = field_of_box(t)[cell(t)] over all the boxes of a time monitor
    /// (1-D launch over their cells together). `desc` as in `ad_dft_multi`.
    #[kernel]
    pub unsafe fn ad_gather_multi(
        ex: *const f32,
        ey: *const f32,
        ez: *const f32,
        hx: *const f32,
        hy: *const f32,
        hz: *const f32,
        out: *mut f32,
        desc: *const u32,
        nbox: u32,
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
            *out.add(t) = *field.add(s);
        }
    }

    /// field_of_box(t)[cell(t)] = buf[t] over the boxes of a time monitor
    /// whose component is an E (`which & 1`) or an H (`which & 2`) component:
    /// the inverse of `ad_gather_multi`, used to replay the recorded PML
    /// shell during time reversal (E and H at different samples).
    #[kernel]
    pub unsafe fn ad_scatter_multi(
        ex: *mut f32,
        ey: *mut f32,
        ez: *mut f32,
        hx: *mut f32,
        hy: *mut f32,
        hz: *mut f32,
        buf: *const f32,
        desc: *const u32,
        nbox: u32,
        total: u32,
        ny: u32,
        nz: u32,
        which: u32,
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
            if (comp < 3 && (which & 1) == 0) || (comp >= 3 && (which & 2) == 0) {
                return;
            }
            let lyu = *d.add(5) as usize;
            let lzu = *d.add(6) as usize;
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
            *field.add(s) = *buf.add(t);
        }
    }

    /// acc[t] += Re((cr + i ci) (a_re + i a_im)[t] (b_re + i b_im)[t])
    ///
    /// The adjoint product of two DFT fields over a box at one frequency:
    /// with c = (i w eps0 / 4) conj(g) / (norm_a norm_b) this is the
    /// contribution of one (output, input, frequency) to dL/d(eps) per unit
    /// volume.
    #[kernel]
    pub unsafe fn ad_product(
        acc: *mut f32,
        a_re: *const f32,
        a_im: *const f32,
        b_re: *const f32,
        b_im: *const f32,
        n: u32,
        cr: f32,
        ci: f32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let ar = *a_re.add(t);
            let ai = *a_im.add(t);
            let br = *b_re.add(t);
            let bi = *b_im.add(t);
            let pr = ar * br - ai * bi;
            let pi = ar * bi + ai * br;
            *acc.add(t) += cr * pr - ci * pi;
        }
    }

    /// Coefficient sums of one reverse step over a box (1-D launch over it):
    ///
    /// p1[t] += lam[s] * (snap[t] - ca[s] * fwd[s]),   p2[t] += lam[s] * fwd[s]
    ///
    /// `lam` is the scaled adjoint E component, `fwd` the reconstructed
    /// E(n), `snap` the box of E(n+1) taken before the reversal. `p1` is
    /// summed with compensation (`comp`, Kahan): its terms change sign from
    /// step to step. `p2` is only touched when `has_ca != 0` (lossy plans).
    #[kernel]
    pub unsafe fn ad_accumulate(
        p1: *mut f32,
        comp: *mut f32,
        p2: *mut f32,
        lam: *const f32,
        fwd: *const f32,
        snap: *const f32,
        ca: *const f32,
        has_ca: u32,
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
            let l = *lam.add(s);
            let e = *fwd.add(s);
            let a = if has_ca != 0 { *ca.add(s) } else { 1.0 };
            let x = l * (*snap.add(t) - a * e);
            let y = x - *comp.add(t);
            let sum = *p1.add(t) + y;
            *comp.add(t) = (sum - *p1.add(t)) - y;
            *p1.add(t) = sum;
            if has_ca != 0 {
                *p2.add(t) += l * e;
            }
        }
    }

    /// Pole-coefficient gradients of the dispersive cells of one component
    /// (see `lightsim_core::autodiff::cell::pole_coef_step`), after the E
    /// reversal: `lam` is the scaled adjoint E (the cotangent of E(n+1) is
    /// `ascale * lam`), `field` holds E(n), `p` / `pprev` the poles P(n) /
    /// P(n-1); `gp` / `gpp` carry the pole cotangents `[pole][entry]`, `acc`
    /// accumulates dL/d(c0, c1, c2) as `[pole][3][entry]`.
    #[kernel]
    pub unsafe fn ad_pole_grad(
        lam: *const f32,
        field: *const f32,
        idx: *const u64,
        matid: *const u32,
        coefs: *const f32,
        scale: *const f32,
        ascale: *const f32,
        p: *const f32,
        pprev: *const f32,
        gp: *mut f32,
        gpp: *mut f32,
        acc: *mut f32,
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
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let a = *ascale.add(t) * *lam.add(s);
            let e = *field.add(s);
            let sc = *scale.add(t);
            let mut q = 0usize;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                let (c0, c1, c2) = (*c, *c.add(1), *c.add(2));
                if !(c0 == 1.0 && c1 == 0.0 && c2 == 0.0) {
                    let o = q * nu + t;
                    let w = *gp.add(o) - sc * a;
                    let g = c0 * w + sc * a + *gpp.add(o);
                    *gpp.add(o) = c1 * w;
                    *gp.add(o) = g;
                    *acc.add((q * 3) * nu + t) += w * *p.add(o);
                    *acc.add((q * 3 + 1) * nu + t) += w * *pprev.add(o);
                    *acc.add((q * 3 + 2) * nu + t) += w * e;
                }
                q += 1;
            }
        }
    }

    /// Time reversal of dispersive cells, before the linear E reversal:
    /// E <- E + scale * D with, per pole, D += P(n+1) - P(n) (second order)
    /// or P(n+1) - P(n+1) / c0 (first order, c1 = 0). `p` holds P(n+1) and
    /// `pprev` P(n), both `[pole][entry]`; `coefs` is `[material][pole][3]`.
    #[kernel]
    pub unsafe fn ad_disp_pre(
        field: *mut f32,
        idx: *const u64,
        matid: *const u32,
        p: *const f32,
        pprev: *const f32,
        coefs: *const f32,
        scale: *const f32,
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
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let mut d = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                let o = q * nu + t;
                let pn = *p.add(o);
                if *c.add(1) != 0.0 {
                    d += pn - *pprev.add(o);
                } else {
                    d += pn - pn / *c;
                }
                q += 1;
            }
            *field.add(s) += *scale.add(t) * d;
        }
    }

    /// Time reversal of dispersive cells, after the linear E reversal
    /// returned g: E(n) = g / (1 - scale / ca * sum_first c2 / c0), then the
    /// pole states before the step: second order (P(n), P(n-1) = (P(n+1) -
    /// c0 P(n) - c2 E(n)) / c1), first order (P(n) = (P(n+1) - c2 E(n)) / c0).
    #[kernel]
    pub unsafe fn ad_disp_post(
        field: *mut f32,
        idx: *const u64,
        matid: *const u32,
        p: *mut f32,
        pprev: *mut f32,
        coefs: *const f32,
        scale: *const f32,
        ca: *const f32,
        has_ca: u32,
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
            let m = *matid.add(t) as usize;
            let np = npole as usize;
            let mut kappa = 0.0f32;
            let mut q = 0usize;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                if *c.add(1) == 0.0 {
                    kappa += *c.add(2) / *c;
                }
                q += 1;
            }
            let a = if has_ca != 0 { *ca.add(s) } else { 1.0 };
            let e = *field.add(s) / (1.0 - *scale.add(t) / a * kappa);
            *field.add(s) = e;
            q = 0;
            while q < np {
                let c = coefs.add((m * np + q) * 3);
                let o = q * nu + t;
                let pn = *p.add(o);
                let pc = *pprev.add(o);
                if *c.add(1) != 0.0 {
                    *p.add(o) = pc;
                    *pprev.add(o) = (pn - *c * pc - *c.add(2) * e) / *c.add(1);
                } else {
                    *p.add(o) = (pn - *c.add(2) * e) / *c;
                    *pprev.add(o) = 0.0;
                }
                q += 1;
            }
        }
    }

    /// chi2 / chi3 cells after the linear E reversal. The forward step solved
    /// D(E(n+1)) = D(E(n)) + E_lin - E(n) with D(x) = x + a x^2 + b x^3 and
    /// E_lin = ca E(n) + cb curl. The reversal applied to E(n+1) (`e1`,
    /// saved before it) returned g = (E(n+1) - cb curl) / ca in `field`:
    /// solve D(e) + (ca - 1) e = D(e1) - e1 + ca g for e = E(n) (Newton from
    /// g) and store the partial derivatives of the step,
    /// fe = dE(n+1)/dE(n) = (D'(e) - 1) / D'(e1),  fl = dE(n+1)/dE_lin = 1 / D'(e1).
    /// Cells with `mask == 0` (inside a PML, where the reconstructed fields
    /// are not meaningful) are left as they are, with fe = 0, fl = 1.
    #[kernel]
    pub unsafe fn ad_nl_post(
        field: *mut f32,
        idx: *const u64,
        e1: *const f32,
        c2: *const f32,
        c3: *const f32,
        ca: *const f32,
        has_ca: u32,
        mask: *const u32,
        fe: *mut f32,
        fl: *mut f32,
        n: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            if *mask.add(t) == 0 {
                *fe.add(t) = 0.0;
                *fl.add(t) = 1.0;
                return;
            }
            let s = *idx.add(t) as usize;
            let a = *c2.add(t);
            let b = *c3.add(t);
            let cav = if has_ca != 0 { *ca.add(s) } else { 1.0 };
            let x1 = *e1.add(t);
            let g = *field.add(s);
            let rhs = (x1 + a * x1 * x1 + b * x1 * x1 * x1) - x1 + cav * g;
            let mut e = g;
            let mut it = 0;
            while it < 6 {
                let f = (e + a * e * e + b * e * e * e) + (cav - 1.0) * e - rhs;
                let fp = (1.0 + 2.0 * a * e + 3.0 * b * e * e) + cav - 1.0;
                e -= f / fp;
                it += 1;
            }
            *field.add(s) = e;
            let d1 = 1.0 + 2.0 * a * x1 + 3.0 * b * x1 * x1;
            *fe.add(t) = ((1.0 + 2.0 * a * e + 3.0 * b * e * e) - 1.0) / d1;
            *fl.add(t) = 1.0 / d1;
        }
    }

    /// Adjoint of a chi2 / chi3 step, before the linear transposed step:
    /// save[t] = fe[t] * lam (the part E(n) receives directly) and
    /// lam <- fl[t] * lam (the cotangent of E_lin, which drives the step).
    #[kernel]
    pub unsafe fn ad_nl_adj_pre(lam: *mut f32, idx: *const u64, fe: *const f32, fl: *const f32, save: *mut f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            let s = *idx.add(t) as usize;
            let l = *lam.add(s);
            *save.add(t) = *fe.add(t) * l;
            *lam.add(s) = *fl.add(t) * l;
        }
    }

    /// Adjoint of a chi2 / chi3 step, after the linear transposed step:
    /// lam += save.
    #[kernel]
    pub unsafe fn ad_nl_adj_post(lam: *mut f32, idx: *const u64, save: *const f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *lam.add(*idx.add(t) as usize) += *save.add(t);
        }
    }

    // ----------------------------------------------------------------------
    // Transposed CPML (adjoint step of the reverse pass).
    //
    // The forward CPML of a slab normal to `axis` convolves the *difference*
    // of the other field with the stretching coefficients of the cell being
    // updated (`pml_e`, `pml_h`). Its transpose, in the scaled adjoint
    // variables, convolves the *field* with the coefficients of the cell it
    // comes from and then differences: with (b, c) the cyclic components,
    //
    //   phi_b <- bE phi_b + e_b,       w_b = (1/kappaE - 1) e_b + aE phi_b   (E cells)
    //   h_c -= ch d(w_b),              h_b += ch d(w_c)
    //   theta_b <- bH theta_b + h_b,   v_b = (1/kappaH - 1) h_b + aH theta_b (H cells)
    //   e_c += cb d(v_b),              e_b -= cb d(v_c)
    //
    // with the forward / backward differences of the forward kernels and w,
    // v zero outside the slab. `ad_pml_phi` advances phi (or theta);
    // `ad_pml_h` / `ad_pml_e` apply the corrections over the slab extended
    // by the one layer whose difference reaches into it. Every thread writes
    // its own cell only.
    // ----------------------------------------------------------------------

    /// phi_b[t] = pb[q] phi_b[t] + fb[cell], phi_c likewise, over a slab
    /// (1-D launch over its cells; `q` = position along `axis` minus `a0`).
    #[kernel]
    pub unsafe fn ad_pml_phi(
        fb: *const f32,
        fc: *const f32,
        phi_b: *mut f32,
        phi_c: *mut f32,
        pb: *const f32,
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
        if t >= (lx as usize) * lyu * lzu {
            return;
        }
        let i = x0 as usize + t / (lzu * lyu);
        let j = y0 as usize + (t / lzu) % lyu;
        let k = z0 as usize + t % lzu;
        let s = ((i + 1) * ny as usize + j) * nz as usize + k;
        let along = if axis == 0 { i } else if axis == 1 { j } else { k };
        unsafe {
            let b = *pb.add(along - a0 as usize);
            *phi_b.add(t) = b * *phi_b.add(t) + *fb.add(s);
            *phi_c.add(t) = b * *phi_c.add(t) + *fc.add(s);
        }
    }

    /// Transposed E-side CPML, acting on the adjoint H:
    /// h_c -= ch (w_b(s + a) - w_b(s)) / dp,  h_b += ch (w_c(s + a) - w_c(s)) / dp,
    /// w = (kinv - 1) e + pa phi inside the slab `[a0, a0 + la)`, 0 outside.
    /// 1-D launch over the box `(bx0, by0, bz0) + (blx, bly, blz)`: the slab
    /// plus the layer below it. `(sx0, sy0, sz0)`, `sly`, `slz` locate a cell
    /// in the slab-local arrays `phi_b`, `phi_c`.
    #[kernel]
    pub unsafe fn ad_pml_h(
        hb: *mut f32,
        hc: *mut f32,
        eb: *const f32,
        ec: *const f32,
        phi_b: *const f32,
        phi_c: *const f32,
        chb: *const f32,
        chc: *const f32,
        ch_scalar: f32,
        ch_mode: u32,
        pa: *const f32,
        pkinv: *const f32,
        id_a: *const f32,
        axis: u32,
        a0: u32,
        la: u32,
        n_along: u32,
        bx0: u32,
        by0: u32,
        bz0: u32,
        blx: u32,
        bly: u32,
        blz: u32,
        sx0: u32,
        sy0: u32,
        sz0: u32,
        sly: u32,
        slz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = blz as usize;
        let lyu = bly as usize;
        if t >= (blx as usize) * lyu * lzu {
            return;
        }
        let i = bx0 as usize + t / (lzu * lyu);
        let j = by0 as usize + (t / lzu) % lyu;
        let k = bz0 as usize + t % lzu;
        let nyu = ny as usize;
        let nzu = nz as usize;
        let s = ((i + 1) * nyu + j) * nzu + k;
        let (along, stride, sstride) = if axis == 0 {
            (i, nyu * nzu, (sly as usize) * (slz as usize))
        } else if axis == 1 {
            (j, nzu, slz as usize)
        } else {
            (k, 1usize, 1usize)
        };
        let lo = a0 as usize;
        let hi = lo + la as usize;
        unsafe {
            // slab-local offset of this cell (meaningful when it is inside)
            let mut wb0 = 0.0f32;
            let mut wc0 = 0.0f32;
            let mut wb1 = 0.0f32;
            let mut wc1 = 0.0f32;
            if along >= lo && along < hi {
                let ts = ((i - sx0 as usize) * sly as usize + (j - sy0 as usize)) * slz as usize + (k - sz0 as usize);
                let q = along - lo;
                let km1 = *pkinv.add(q) - 1.0;
                let a = *pa.add(q);
                wb0 = km1 * *eb.add(s) + a * *phi_b.add(ts);
                wc0 = km1 * *ec.add(s) + a * *phi_c.add(ts);
            }
            if along + 1 >= lo && along + 1 < hi && along + 1 < n_along as usize {
                // the neighbour one layer up: same transverse position
                let (i1, j1, k1) = if axis == 0 { (i + 1, j, k) } else if axis == 1 { (i, j + 1, k) } else { (i, j, k + 1) };
                let ts = ((i1 - sx0 as usize) * sly as usize + (j1 - sy0 as usize)) * slz as usize + (k1 - sz0 as usize);
                let q = along + 1 - lo;
                let km1 = *pkinv.add(q) - 1.0;
                let a = *pa.add(q);
                wb1 = km1 * *eb.add(s + stride) + a * *phi_b.add(ts);
                wc1 = km1 * *ec.add(s + stride) + a * *phi_c.add(ts);
            }
            let _ = sstride;
            let inv = *id_a.add(along);
            let (cb_, cc_) = if ch_mode == 1 { (*chb.add(s), *chc.add(s)) } else { (ch_scalar, ch_scalar) };
            *hc.add(s) -= cc_ * (wb1 - wb0) * inv;
            *hb.add(s) += cb_ * (wc1 - wc0) * inv;
        }
    }

    /// Transposed H-side CPML, acting on the adjoint E:
    /// e_c += cb (v_b(s) - v_b(s - a)) / dd,  e_b -= cb (v_c(s) - v_c(s - a)) / dd,
    /// v = (kinv - 1) h + pa theta inside the slab, 0 outside. 1-D launch
    /// over the slab plus the layer above it (arguments as `ad_pml_h`).
    #[kernel]
    pub unsafe fn ad_pml_e(
        eb: *mut f32,
        ec: *mut f32,
        hb: *const f32,
        hc: *const f32,
        th_b: *const f32,
        th_c: *const f32,
        cbb: *const f32,
        cbc: *const f32,
        pa: *const f32,
        pkinv: *const f32,
        id_a: *const f32,
        axis: u32,
        a0: u32,
        la: u32,
        bx0: u32,
        by0: u32,
        bz0: u32,
        blx: u32,
        bly: u32,
        blz: u32,
        sx0: u32,
        sy0: u32,
        sz0: u32,
        sly: u32,
        slz: u32,
        ny: u32,
        nz: u32,
    ) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        let lzu = blz as usize;
        let lyu = bly as usize;
        if t >= (blx as usize) * lyu * lzu {
            return;
        }
        let i = bx0 as usize + t / (lzu * lyu);
        let j = by0 as usize + (t / lzu) % lyu;
        let k = bz0 as usize + t % lzu;
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
        let lo = a0 as usize;
        let hi = lo + la as usize;
        unsafe {
            let mut vb0 = 0.0f32;
            let mut vc0 = 0.0f32;
            let mut vbm = 0.0f32;
            let mut vcm = 0.0f32;
            if along >= lo && along < hi {
                let ts = ((i - sx0 as usize) * sly as usize + (j - sy0 as usize)) * slz as usize + (k - sz0 as usize);
                let q = along - lo;
                let km1 = *pkinv.add(q) - 1.0;
                let a = *pa.add(q);
                vb0 = km1 * *hb.add(s) + a * *th_b.add(ts);
                vc0 = km1 * *hc.add(s) + a * *th_c.add(ts);
            }
            if along >= 1 && along - 1 >= lo && along - 1 < hi {
                let (i1, j1, k1) = if axis == 0 { (i - 1, j, k) } else if axis == 1 { (i, j - 1, k) } else { (i, j, k - 1) };
                let ts = ((i1 - sx0 as usize) * sly as usize + (j1 - sy0 as usize)) * slz as usize + (k1 - sz0 as usize);
                let q = along - 1 - lo;
                let km1 = *pkinv.add(q) - 1.0;
                let a = *pa.add(q);
                vbm = km1 * *hb.add(s - stride) + a * *th_b.add(ts);
                vcm = km1 * *hc.add(s - stride) + a * *th_c.add(ts);
            }
            let inv = *id_a.add(along);
            *ec.add(s) += *cbc.add(s) * (vb0 - vbm) * inv;
            *eb.add(s) -= *cbb.add(s) * (vc0 - vcm) * inv;
        }
    }
}
