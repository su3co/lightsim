// The 2-D leapfrog step, as a macro that expands to one function.
//
// `kernels/src/twod.rs` expands it into the device kernels `step2d` and
// `step2d_bloch`; `crates/lightsim-core/tests/twod_kernel.rs` expands the
// same text into host functions and runs them cell by cell against the CPU
// backend, so the kernel logic is tested on machines without a GPU. Nothing
// here may depend on the device crates.
//
// # The step
//
// A grid that is one periodic cell thick along an axis `w` has fields that
// do not depend on `w`. With `(u, v)` the in-plane axes (`v` contiguous in
// storage: `s = base + u * nv + v`) Maxwell's equations split into
//
// * TE: (E_u, E_v, H_w)   H_w -= ch sg (d_u E_v - d_v E_u)
//                         E_u += cb sg d_v H_w,   E_v -= cb sg d_u H_w
// * TM: (E_w, H_u, H_v)   H_u -= ch sg d_v E_w,   H_v += ch sg d_u E_w
//                         E_w += cb sg (d_u H_v - d_v H_u)
//
// with `sg` = +1 if (u, v, w) is a cyclic permutation of (x, y, z), else -1.
// `pol` selects the sets to update (bit 0 TE, bit 1 TM).
//
// One launch does a whole step, ping-pong like `step_fused`: thread `t`
// owns cell `(u, v) = (t / nv, t % nv)`, computes its H^{n+1/2} from the
// current buffers (A), recomputes the H^{n+1/2} of the two neighbours its
// E update needs (cells u - 1 and v - 1: at most 3 components each, cheaper
// than staging them through shared memory and a barrier), and writes
// H^{n+1/2} and E^{n+1} of its own cell to the new buffers (B). Threads are
// independent: a flat 1-D launch over the nu * nv cells, any block size.
//
// Boundary codes (`bc` = u_lo | u_hi << 4 | v_lo << 8 | v_hi << 12):
// 0 zero, 1 periodic, 2 mirror (PMC), 3 Bloch-periodic (complex fields: the
// `*o` arguments are the other part, `bc?`/`bs?` the cosine and the signed
// sine of k L, as in `step_fused_bloch`), and along u only 4 = host-managed
// ghost rows: the +u neighbour of the last row is read from the row after
// it and the E update of row 0 is left to the host (`e_plane_fused`, after
// the ghost row of the new H has been filled). That is how the x axis is
// handled, exactly as by `step_fused`, so slabs and Bloch phases along x
// work unchanged.
//
// CPML: `pml_u` / `pml_v` pack (n_lo | n_hi << 16) layers; `pru` / `prv`
// are the per-position profiles [pos][6] = (be, ae, kinv_e, bh, ah, kinv_h)
// and `se?` / `sh?_r` / `sh?_w` the E-side state (in place) and the
// double-buffered H-side state, all with the layout of `step_fused`: along
// u, cell (up, v) of the layers is at `up * nv + v`, along v at
// `u * npv + vp`, and each array holds two slots of that size. Which slot
// a set uses follows from the cyclic order of the axes, so that
// `e_plane_fused` and this kernel address the same state.
//
// Materials: per-cell index `mat` (width `mat_mode`, see `mat_index!` in
// main.rs) into the table [cbx, cby, cbz, cax, cay, caz]; `cols` packs the
// columns of (u, v, w): u | v << 4 | w << 8.

/// Material index of storage cell `s` (as `mat_index!` of main.rs).
macro_rules! mat_index2 {
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

/// `step2d_fn!($ [attributes] name, bloch, [leading parameters], thread index)`
macro_rules! step2d_fn {
    ($d:tt [$($attr:tt)*] $name:ident, $bloch:tt, [$($lead:tt)*], $t:expr) => {
        $($attr)*
        pub unsafe fn $name(
            $($lead)*
            eua: *const f32,
            eva: *const f32,
            ewa: *const f32,
            hua: *const f32,
            hva: *const f32,
            hwa: *const f32,
            eub: *mut f32,
            evb: *mut f32,
            ewb: *mut f32,
            hub: *mut f32,
            hvb: *mut f32,
            hwb: *mut f32,
            mat: *const u32,
            tab: *const f32,
            mat_mode: u32,
            lossy: u32,
            cols: u32,
            ch: f32,
            ipu: *const f32,
            ipv: *const f32,
            idu: *const f32,
            idv: *const f32,
            nu: u32,
            nv: u32,
            base: u32,
            bc: u32,
            pml_u: u32,
            pml_v: u32,
            pru: *const f32,
            prv: *const f32,
            seu: *mut f32,
            sev: *mut f32,
            shu_r: *const f32,
            shv_r: *const f32,
            shu_w: *mut f32,
            shv_w: *mut f32,
            pol: u32,
            sg: f32,
            euo: *const f32,
            evo: *const f32,
            ewo: *const f32,
            huo: *const f32,
            hvo: *const f32,
            hwo: *const f32,
            sou_r: *const f32,
            sov_r: *const f32,
            bcu: f32,
            bsu: f32,
            bcv: f32,
            bsv: f32,
        ) {
            // 32-bit cell indices (the host guarantees fewer than 2^32 storage cells)
            let t: u32 = $t;
            if t >= nu * nv {
                return;
            }
            let u = t / nv;
            let v = t - u * nv;
            let s = (base + t) as usize;
            let bc_ulo = bc & 15;
            let bc_uhi = (bc >> 4) & 15;
            let bc_vlo = (bc >> 8) & 15;
            let bc_vhi = (bc >> 12) & 15;
            let nlu = pml_u & 0xffff;
            let nhu = pml_u >> 16;
            let nlv = pml_v & 0xffff;
            let nhv = pml_v >> 16;
            let npv = nlv + nhv;
            let cnt_u = ((nlu + nhu) * nv) as usize;
            let cnt_v = (nu * npv) as usize;
            let te = (pol & 1) != 0;
            let tm = (pol & 2) != 0;
            // Slot of each set in the CPML state arrays. For layers normal to
            // axis a with cyclic partners (b, c) = (a + 1, a + 2), `step_fused`
            // keeps the state of (E_b, H_c) in slot 0 of the E side and slot 1
            // of the H side, and that of (E_c, H_b) in the other ones. Along u
            // the TE pair is (E_v, H_w): slot 0 / 1 when v follows u cyclically
            // (sg > 0); along v it is (E_u, H_w), the other way round.
            let cyc = sg > 0.0;
            let (e_te_u, e_tm_u, h_te_u, h_tm_u) = if cyc { (0usize, cnt_u, cnt_u, 0usize) } else { (cnt_u, 0usize, 0usize, cnt_u) };
            let (e_te_v, e_tm_v, h_te_v, h_tm_v) = if cyc { (cnt_v, 0usize, 0usize, cnt_v) } else { (0usize, cnt_v, cnt_v, 0usize) };

            // H^{n+1/2} of cell (uu, vv), sets in use, as (H_u, H_v, H_w).
            // `oth`: of the other part of complex fields (Bloch only);
            // `wr`: store the H-side CPML state (own cell only).
            macro_rules! hcell {
                ($d oth:expr, $d uu:expr, $d vv:expr, $d wr:expr) => {{
                    let oth: bool = $bloch && $d oth;
                    // fields of the cell, the fields its Bloch wraps mix in and
                    // the sign of their sine term
                    let (eu_, ev_, ew_, hu_, hv_, hw_, su_, sv_, eux, evx, ewx, sgn) = if oth {
                        (euo, evo, ewo, huo, hvo, hwo, sou_r, sov_r, eua, eva, ewa, -1.0f32)
                    } else {
                        (eua, eva, ewa, hua, hva, hwa, shu_r, shv_r, euo, evo, ewo, 1.0f32)
                    };
                    let uu: u32 = $d uu;
                    let vv: u32 = $d vv;
                    let c = (base + uu * nv + vv) as usize;
                    // +u and +v neighbours of E: (how, where) with how = 0 zero,
                    // 1 the value there, 2 the cell's own value, 3 Bloch mix
                    let (mu_, ou_) = if uu + 1 < nu || bc_uhi == 4 {
                        (1u32, c + nv as usize)
                    } else if bc_uhi == 1 {
                        (1u32, c - ((nu - 1) * nv) as usize)
                    } else if bc_uhi == 2 {
                        (2u32, c)
                    } else if $bloch && bc_uhi == 3 {
                        (3u32, c - ((nu - 1) * nv) as usize)
                    } else {
                        (0u32, c)
                    };
                    let (mv_, ov_) = if vv + 1 < nv {
                        (1u32, c + 1)
                    } else if bc_vhi == 1 {
                        (1u32, c - (nv - 1) as usize)
                    } else if bc_vhi == 2 {
                        (2u32, c)
                    } else if $bloch && bc_vhi == 3 {
                        (3u32, c - (nv - 1) as usize)
                    } else {
                        (0u32, c)
                    };
                    let dui = *ipu.add(uu as usize);
                    let dvi = *ipv.add(vv as usize);
                    let in_u = uu < nlu || uu + nhu >= nu;
                    let in_v = vv < nlv || vv + nhv >= nv;
                    let pu_ = if in_u { ((if uu < nlu { uu } else { nlu + uu + nhu - nu }) * nv + vv) as usize } else { 0usize };
                    let pv_ = if in_v { (uu * npv + (if vv < nlv { vv } else { nlv + vv + nhv - nv })) as usize } else { 0usize };
                    let mut hu_n = 0.0f32;
                    let mut hv_n = 0.0f32;
                    let mut hw_n = 0.0f32;
                    if te {
                        let eu0 = *eu_.add(c);
                        let ev0 = *ev_.add(c);
                        let ev_up = if mu_ == 1 {
                            *ev_.add(ou_)
                        } else if mu_ == 2 {
                            ev0
                        } else if mu_ == 3 {
                            bcu * *ev_.add(ou_) + sgn * bsu * *evx.add(ou_)
                        } else {
                            0.0f32
                        };
                        let eu_vp = if mv_ == 1 {
                            *eu_.add(ov_)
                        } else if mv_ == 2 {
                            eu0
                        } else if mv_ == 3 {
                            bcv * *eu_.add(ov_) + sgn * bsv * *eux.add(ov_)
                        } else {
                            0.0f32
                        };
                        let d_u = (ev_up - ev0) * dui;
                        let d_v = (eu_vp - eu0) * dvi;
                        let mut hw = *hw_.add(c) - ch * (sg * (d_u - d_v));
                        if in_u {
                            let pr = pru.add((6 * uu) as usize);
                            let o = pu_ + h_te_u;
                            let psi = *pr.add(3) * *su_.add(o) + *pr.add(4) * d_u;
                            if $d wr {
                                *shu_w.add(o) = psi;
                            }
                            hw -= sg * ch * ((*pr.add(5) - 1.0) * d_u + psi);
                        }
                        if in_v {
                            let pr = prv.add((6 * vv) as usize);
                            let o = pv_ + h_te_v;
                            let psi = *pr.add(3) * *sv_.add(o) + *pr.add(4) * d_v;
                            if $d wr {
                                *shv_w.add(o) = psi;
                            }
                            hw += sg * ch * ((*pr.add(5) - 1.0) * d_v + psi);
                        }
                        hw_n = hw;
                    }
                    if tm {
                        let ew0 = *ew_.add(c);
                        let ew_up = if mu_ == 1 {
                            *ew_.add(ou_)
                        } else if mu_ == 2 {
                            ew0
                        } else if mu_ == 3 {
                            bcu * *ew_.add(ou_) + sgn * bsu * *ewx.add(ou_)
                        } else {
                            0.0f32
                        };
                        let ew_vp = if mv_ == 1 {
                            *ew_.add(ov_)
                        } else if mv_ == 2 {
                            ew0
                        } else if mv_ == 3 {
                            bcv * *ew_.add(ov_) + sgn * bsv * *ewx.add(ov_)
                        } else {
                            0.0f32
                        };
                        let d_u = (ew_up - ew0) * dui;
                        let d_v = (ew_vp - ew0) * dvi;
                        let mut hu = *hu_.add(c) - ch * (sg * d_v);
                        let mut hv = *hv_.add(c) + ch * (sg * d_u);
                        if in_u {
                            let pr = pru.add((6 * uu) as usize);
                            let o = pu_ + h_tm_u;
                            let psi = *pr.add(3) * *su_.add(o) + *pr.add(4) * d_u;
                            if $d wr {
                                *shu_w.add(o) = psi;
                            }
                            hv += sg * ch * ((*pr.add(5) - 1.0) * d_u + psi);
                        }
                        if in_v {
                            let pr = prv.add((6 * vv) as usize);
                            let o = pv_ + h_tm_v;
                            let psi = *pr.add(3) * *sv_.add(o) + *pr.add(4) * d_v;
                            if $d wr {
                                *shv_w.add(o) = psi;
                            }
                            hu -= sg * ch * ((*pr.add(5) - 1.0) * d_v + psi);
                        }
                        hu_n = hu;
                        hv_n = hv;
                    }
                    (hu_n, hv_n, hw_n)
                }};
            }

            unsafe {
                let own = hcell!(false, u, v, true);
                if te {
                    *hwb.add(s) = own.2;
                }
                if tm {
                    *hub.add(s) = own.0;
                    *hvb.add(s) = own.1;
                }
                // row 0 with host-managed ghost rows: E is computed by the host
                // once the ghost row of the new H is known
                if u == 0 && bc_ulo == 4 {
                    return;
                }
                // new H of the low-u neighbour: 0 zero, 1 the cell `lu`, 2 minus
                // the own value (mirror), 3 Bloch (cell `lu`, both parts)
                let (mlu, lu) = if u > 0 {
                    (1u32, u - 1)
                } else if bc_ulo == 1 {
                    (1u32, nu - 1)
                } else if bc_ulo == 2 {
                    (2u32, 0u32)
                } else if $bloch && bc_ulo == 3 {
                    (3u32, nu - 1)
                } else {
                    (0u32, 0u32)
                };
                let mut hlu = (0.0f32, 0.0f32, 0.0f32);
                if mlu == 1 || mlu == 3 {
                    hlu = hcell!(false, lu, v, false);
                    if $bloch && mlu == 3 {
                        // H(-1) = H(n-1) exp(-i k L)
                        let o = hcell!(true, lu, v, false);
                        hlu = (bcu * hlu.0 - bsu * o.0, bcu * hlu.1 - bsu * o.1, bcu * hlu.2 - bsu * o.2);
                    }
                } else if mlu == 2 {
                    hlu = (-own.0, -own.1, -own.2);
                }
                let (mlv, lv) = if v > 0 {
                    (1u32, v - 1)
                } else if bc_vlo == 1 {
                    (1u32, nv - 1)
                } else if bc_vlo == 2 {
                    (2u32, 0u32)
                } else if $bloch && bc_vlo == 3 {
                    (3u32, nv - 1)
                } else {
                    (0u32, 0u32)
                };
                let mut hlv = (0.0f32, 0.0f32, 0.0f32);
                if mlv == 1 || mlv == 3 {
                    hlv = hcell!(false, u, lv, false);
                    if $bloch && mlv == 3 {
                        let o = hcell!(true, u, lv, false);
                        hlv = (bcv * hlv.0 - bsv * o.0, bcv * hlv.1 - bsv * o.1, bcv * hlv.2 - bsv * o.2);
                    }
                } else if mlv == 2 {
                    hlv = (-own.0, -own.1, -own.2);
                }

                let mt = tab.add(6 * mat_index2!(mat, mat_mode, s));
                let du = *idu.add(u as usize);
                let dv = *idv.add(v as usize);
                let in_u = u < nlu || u + nhu >= nu;
                let in_v = v < nlv || v + nhv >= nv;
                let pu_ = if in_u { ((if u < nlu { u } else { nlu + u + nhu - nu }) * nv + v) as usize } else { 0usize };
                let pv_ = if in_v { (u * npv + (if v < nlv { v } else { nlv + v + nhv - nv })) as usize } else { 0usize };
                if te {
                    let cu = (cols & 15) as usize;
                    let cv = ((cols >> 4) & 15) as usize;
                    let cbu = *mt.add(cu);
                    let cbv = *mt.add(cv);
                    let d_u = (own.2 - hlu.2) * du;
                    let d_v = (own.2 - hlv.2) * dv;
                    let eu0 = *eua.add(s);
                    let ev0 = *eva.add(s);
                    let mut eun;
                    let mut evn;
                    if lossy != 0 {
                        eun = *mt.add(cu + 3) * eu0 + cbu * (sg * d_v);
                        evn = *mt.add(cv + 3) * ev0 - cbv * (sg * d_u);
                    } else {
                        eun = eu0 + cbu * (sg * d_v);
                        evn = ev0 - cbv * (sg * d_u);
                    }
                    if in_u {
                        let pr = pru.add((6 * u) as usize);
                        let o = pu_ + e_te_u;
                        let psi = *pr * *seu.add(o) + *pr.add(1) * d_u;
                        *seu.add(o) = psi;
                        evn -= sg * cbv * ((*pr.add(2) - 1.0) * d_u + psi);
                    }
                    if in_v {
                        let pr = prv.add((6 * v) as usize);
                        let o = pv_ + e_te_v;
                        let psi = *pr * *sev.add(o) + *pr.add(1) * d_v;
                        *sev.add(o) = psi;
                        eun += sg * cbu * ((*pr.add(2) - 1.0) * d_v + psi);
                    }
                    *eub.add(s) = eun;
                    *evb.add(s) = evn;
                }
                if tm {
                    let cw = ((cols >> 8) & 15) as usize;
                    let cbw = *mt.add(cw);
                    let d_u = (own.1 - hlu.1) * du;
                    let d_v = (own.0 - hlv.0) * dv;
                    let ew0 = *ewa.add(s);
                    let mut ewn;
                    if lossy != 0 {
                        ewn = *mt.add(cw + 3) * ew0 + cbw * (sg * (d_u - d_v));
                    } else {
                        ewn = ew0 + cbw * (sg * (d_u - d_v));
                    }
                    if in_u {
                        let pr = pru.add((6 * u) as usize);
                        let o = pu_ + e_tm_u;
                        let psi = *pr * *seu.add(o) + *pr.add(1) * d_u;
                        *seu.add(o) = psi;
                        ewn += sg * cbw * ((*pr.add(2) - 1.0) * d_u + psi);
                    }
                    if in_v {
                        let pr = prv.add((6 * v) as usize);
                        let o = pv_ + e_tm_v;
                        let psi = *pr * *sev.add(o) + *pr.add(1) * d_v;
                        *sev.add(o) = psi;
                        ewn -= sg * cbw * ((*pr.add(2) - 1.0) * d_v + psi);
                    }
                    *ewb.add(s) = ewn;
                }
            }
        }
    };
}
