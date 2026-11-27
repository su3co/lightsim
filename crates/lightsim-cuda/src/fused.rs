//! Fused-step execution state (see `step_fused` in `kernels/src/main.rs`).
//!
//! In fused mode a time step is one kernel that writes H^{n+1/2} and E^{n+1}
//! into a second set of buffers (B) while reading the current set (A); the
//! sets swap after the monitors have read B. H-side CPML state is likewise
//! double-buffered. Magnetic-current injections (TFSF/mode/plane-wave H
//! corrections, magnetic dipoles) are applied to H_B after the kernel, and
//! their effect on E^{n+1} — which the kernel computed from the uncorrected
//! H — is added through an equivalent list of E corrections (the discrete
//! curl of the H correction), so the result equals the split update.

use crate::{InjDev, dbuf, zbuf};
use anyhow::{Result, bail};
use cuda_core::{CudaStream, DeviceBuffer};
use lightsim_core::Component;
use lightsim_core::plan::{BC_BLOCH, BC_MIRROR, BC_PERIODIC, GhostRule, Injection, Plan};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) struct FusedState {
    pub eb: [DeviceBuffer<f32>; 3],
    pub hb: [DeviceBuffer<f32>; 3],
    /// Per-cell material index, stored as raw bytes of width `mat_mode`
    /// (0 = uniform slab: no per-cell array, the kernels use table row 0)
    pub mat: DeviceBuffer<u8>,
    pub mat_mode: u32,
    /// (z, y) tile of the fused kernel = its block shape
    pub tile: (u32, u32),
    pub tab: DeviceBuffer<f32>,
    pub lossy: u32,
    pub prof: [DeviceBuffer<f32>; 3],
    pub psi_e: [DeviceBuffer<f32>; 3],
    pub psi_h: [[DeviceBuffer<f32>; 3]; 2],
    pub pml_n: [u32; 3],
    /// 0 zero, 1 deferred (periodic / neighbour), 2 PMC mirror
    pub ghost_lo: u32,
    pub chunk: u32,
    /// E corrections equivalent to the H injections
    pub inj_hd: Vec<InjDev>,
    pub n_materials: usize,
    /// Second buffer set of the imaginary part (complex fields, Bloch)
    pub imag: Option<FusedImag>,
    /// 2-D plans: the step is `step2d` instead of `step_fused` (see `crate::twod`)
    pub twod: Option<crate::twod::TwoDState>,
}

/// Fused-step buffers of the imaginary part of complex fields. They swap
/// with the real ones in `FusedState` together with the field sets, so the
/// launch code always works on "the selected part".
pub(crate) struct FusedImag {
    pub eb: [DeviceBuffer<f32>; 3],
    pub hb: [DeviceBuffer<f32>; 3],
    pub psi_e: [DeviceBuffer<f32>; 3],
    pub psi_h: [[DeviceBuffer<f32>; 3]; 2],
    /// per derived E injection: amplitude table (a_im, -a_re) of complex sources
    pub amps_hd: Vec<Option<DeviceBuffer<f32>>>,
}

/// Why fused mode is not used for a plan (None = usable).
pub(crate) fn unsupported(plan: &Plan) -> Option<String> {
    if std::env::var("LIGHTSIM_FUSED").map(|v| v == "0").unwrap_or(false) {
        return Some("disabled by LIGHTSIM_FUSED=0".into());
    }
    if plan.ch.is_some() {
        return Some("magnetic materials".into());
    }
    // (the 2-D step takes dispersive cells from the lists, for both parts
    // of complex fields)
    if plan.bloch && (plan.nonlin.is_some() || (plan.disp.is_some() && plan.twod.is_none())) {
        return Some("Bloch boundaries with dispersive or nonlinear media".into());
    }
    if plan.tensor.is_some() && (plan.bloch || std::env::var("LIGHTSIM_TENSOR_FUSED").as_deref() == Ok("0")) {
        return Some("off-diagonal permittivity".into());
    }
    // the fused kernel indexes cells with 32 bits
    if plan.storage_len() as u64 >= 1u64 << 32 {
        return Some("more than 2^32 cells in one slab".into());
    }
    // x-PML must be fully contained in this slab
    for p in &plan.pml {
        if p.axis == 0 {
            let lo_ok = p.start[0] == 0 || p.start[0] + p.size[0] == plan.nxl;
            if !lo_ok {
                return Some("x-PML split across slabs".into());
            }
        }
    }
    None
}

/// Build the per-cell material index and coefficient table
/// [cbx, cby, cbz, cax, cay, caz] from the plan's storage-layout arrays.
pub(crate) fn material_table(plan: &Plan) -> (Vec<u32>, Vec<f32>) {
    let len = plan.storage_len();
    let plane = plan.plane();
    let ca = plan.ca.as_ref();
    let key = |s: usize| -> [u32; 6] {
        let c = |a: usize| ca.map(|v| v[a][s]).unwrap_or(1.0);
        [plan.cb[0][s].to_bits(), plan.cb[1][s].to_bits(), plan.cb[2][s].to_bits(), c(0).to_bits(), c(1).to_bits(), c(2).to_bits()]
    };
    // per plane: local unique keys (run-length cached), then merge
    let nplanes = len / plane;
    let locals: Vec<(Vec<[u32; 6]>, Vec<u32>)> = (0..nplanes)
        .into_par_iter()
        .map(|p| {
            let mut keys: Vec<[u32; 6]> = vec![];
            let mut map: HashMap<[u32; 6], u32> = HashMap::new();
            let mut idx = vec![0u32; plane];
            let mut last_k = [u32::MAX; 6];
            let mut last_i = 0u32;
            for (o, slot) in idx.iter_mut().enumerate() {
                let k = key(p * plane + o);
                if k == last_k {
                    *slot = last_i;
                    continue;
                }
                let id = *map.entry(k).or_insert_with(|| {
                    keys.push(k);
                    (keys.len() - 1) as u32
                });
                last_k = k;
                last_i = id;
                *slot = id;
            }
            (keys, idx)
        })
        .collect();
    let mut gmap: HashMap<[u32; 6], u32> = HashMap::new();
    let mut gkeys: Vec<[u32; 6]> = vec![];
    let remaps: Vec<Vec<u32>> = locals
        .iter()
        .map(|(keys, _)| {
            keys.iter()
                .map(|k| {
                    *gmap.entry(*k).or_insert_with(|| {
                        gkeys.push(*k);
                        (gkeys.len() - 1) as u32
                    })
                })
                .collect()
        })
        .collect();
    let mut mat = vec![0u32; len];
    mat.par_chunks_mut(plane).zip(locals.par_iter()).zip(remaps.par_iter()).for_each(|((out, (_, idx)), rm)| {
        for (o, v) in out.iter_mut().zip(idx.iter()) {
            *o = rm[*v as usize];
        }
    });
    let tab: Vec<f32> = gkeys.iter().flat_map(|k| k.iter().map(|b| f32::from_bits(*b))).collect();
    (mat, tab)
}

/// Pack the per-cell material index into the narrowest width that holds the
/// table: returns the raw bytes and the kernels' `mat_mode` (0 = uniform
/// slab, no per-cell array; 1 = u8, 2 = u16, 4 = u32). `LIGHTSIM_MAT_BITS`
/// forces a wider index (for measurements).
pub(crate) fn pack_index(mat: Vec<u32>, n_materials: usize) -> (Vec<u8>, u32) {
    let mut mode: u32 = if n_materials <= 1 { 0 } else if n_materials <= 256 { 1 } else if n_materials <= 65536 { 2 } else { 4 };
    if let Some(bits) = std::env::var("LIGHTSIM_MAT_BITS").ok().and_then(|v| v.parse::<usize>().ok()) {
        mode = mode.max(match bits { 8 => 1, 16 => 2, 32 => 4, _ => 0 });
    }
    let bytes: Vec<u8> = match mode {
        0 => vec![0u8; 4],
        1 => mat.par_iter().map(|&v| v as u8).collect(),
        2 => mat.par_iter().flat_map_iter(|&v| (v as u16).to_ne_bytes()).collect(),
        _ => mat.par_iter().flat_map_iter(|&v| v.to_ne_bytes()).collect(),
    };
    (bytes, mode)
}

/// E corrections equivalent to the H injections (discrete curl), or an error
/// string if an injection touches a slab boundary or a PML region.
pub(crate) fn derived_e_injections(plan: &Plan) -> Result<Vec<Injection>, String> {
    let (nxl, ny, nz) = (plan.nxl, plan.ny(), plan.nz());
    let plane = ny * nz;
    let n = [nxl, ny, nz];
    let lo_rule = |a: usize| -> u32 {
        match a {
            0 => match plan.ghost_h_lo {
                GhostRule::Periodic { .. } => BC_PERIODIC,
                GhostRule::Mirror { .. } => BC_MIRROR,
                GhostRule::Neighbor => 99,
                GhostRule::Zero => 0,
            },
            1 => plan.bc_y[0],
            _ => plan.bc_z[0],
        }
    };
    let in_pml = |i: usize, j: usize, k: usize| -> bool {
        plan.pml.iter().any(|p| {
            let g = [i, j, k];
            (0..3).all(|a| g[a] >= p.start[a] && g[a] < p.start[a] + p.size[a])
        })
    };
    // With a deferred plane-0 E update (periodic or neighbour low ghost) the
    // plane-0 kernel runs after the H injections and already sees them. A
    // 2-D plan whose invariant axis is x defers nothing: its single x-plane
    // is the whole grid, updated by the 2-D kernel.
    let deferred = matches!(plan.ghost_h_lo, GhostRule::Periodic { .. } | GhostRule::Neighbor) && plan.step2d().map(|k| k.deferred).unwrap_or(true);
    let mut out = vec![];
    for inj in &plan.inj_h {
        // (E comp, sidx) -> (re, im, group)
        let mut acc: HashMap<(usize, u64, u32), (f64, f64)> = HashMap::new();
        let ha = inj.comp.axis();
        for (e, &s) in inj.idx.iter().enumerate() {
            let s = s as usize;
            let i = s / plane - 1;
            let j = (s % plane) / nz;
            let k = s % nz;
            if in_pml(i, j, k) {
                return Err("H injection inside a PML region".into());
            }
            let cre = inj.coef[e] as f64;
            let cim = inj.coef_im.as_ref().map(|v| v[e] as f64).unwrap_or(0.0);
            let g = if inj.ngroups > 1 { inj.group[e] } else { 0 };
            // the two E components whose curl contains H_ha, with the axis of
            // the derivative and the sign of the "self" term
            let terms: [(usize, usize, f64); 2] = match ha {
                0 => [(1, 2, 1.0), (2, 1, -1.0)],  // Hx: Ey (+d_z), Ez (-d_y)
                1 => [(0, 2, -1.0), (2, 0, 1.0)],  // Hy: Ex (-d_z), Ez (+d_x)
                _ => [(0, 1, 1.0), (1, 0, -1.0)],  // Hz: Ex (+d_y), Ey (-d_x)
            };
            for (ec, dax, sgn) in terms {
                let g3 = [i, j, k];
                let q = g3[dax];
                // (position, weight, crosses a Bloch boundary)
                let mut targets: Vec<(usize, f64, bool)> = vec![(q, sgn, false)];
                if q + 1 < n[dax] {
                    targets.push((q + 1, -sgn, false));
                } else {
                    match lo_rule(dax) {
                        // x: with a Bloch phase the low ghost is deferred
                        // (below), so the phase never reaches this list
                        BC_PERIODIC => targets.push((0, -sgn, false)),
                        BC_BLOCH => targets.push((0, -sgn, true)),
                        99 => return Err("H injection on the last plane of a slab".into()),
                        _ => {}
                    }
                }
                if q == 0 && lo_rule(dax) == BC_MIRROR {
                    targets.push((0, sgn, false));
                }
                for (qq, w, wrap) in targets {
                    let mut t3 = g3;
                    t3[dax] = qq;
                    if deferred && t3[0] == 0 {
                        continue;
                    }
                    let st = ((t3[0] + 1) * ny + t3[1]) * nz + t3[2];
                    if in_pml(t3[0], t3[1], t3[2]) {
                        return Err("H injection adjacent to a PML region".into());
                    }
                    let inv = plan.inv_d[dax][qq] as f64;
                    let cb = plan.cb[ec][st] as f64;
                    let wgt = w * cb * inv;
                    // H(-1) = exp(-i k L) H(n-1): the complex coefficient
                    // (cre - i cim) picks up the phase (cos - i sin)
                    let (cre, cim) = if wrap {
                        if inj.coef_im.is_none() {
                            return Err("real-coefficient H injection across a Bloch boundary".into());
                        }
                        let (c, sn) = plan.bloch_phase[dax];
                        (c * cre - sn * cim, c * cim + sn * cre)
                    } else {
                        (cre, cim)
                    };
                    let ent = acc.entry((ec, st as u64, g)).or_insert((0.0, 0.0));
                    ent.0 += cre * wgt;
                    ent.1 += cim * wgt;
                }
            }
        }
        // one injection per E component, sharing the H injection's amplitudes
        for ec in 0..3 {
            let mut v: Vec<(u64, f64, f64, u32)> =
                acc.iter().filter(|((c, _, _), _)| *c == ec).map(|((_, s, g), (r, im))| (*s, *r, *im, *g)).collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by_key(|e| (e.3, e.0));
            out.push(Injection {
                comp: Component::e(ec),
                idx: v.iter().map(|e| e.0).collect(),
                coef: v.iter().map(|e| e.1 as f32).collect(),
                coef_im: inj.coef_im.as_ref().map(|_| v.iter().map(|e| e.2 as f32).collect()),
                group: v.iter().map(|e| e.3).collect(),
                ngroups: inj.ngroups,
                amps: inj.amps.clone(),
            });
        }
    }
    Ok(out)
}

pub(crate) fn build(
    plan: &Plan,
    s: &Arc<CudaStream>,
    mk_inj: &dyn Fn(&Vec<Injection>) -> Result<Vec<InjDev>>,
    swap_amps: &dyn Fn(&Vec<Injection>) -> Result<Vec<Option<DeviceBuffer<f32>>>>,
    sm_count: u32,
) -> Result<Option<FusedState>> {
    let verbose = std::env::var("LIGHTSIM_VERBOSE").is_ok();
    if let Some(why) = unsupported(plan) {
        if verbose {
            eprintln!("lightsim: fused mode off: {why}");
        }
        return Ok(None);
    }
    let derived = match derived_e_injections(plan) {
        Ok(d) => d,
        Err(why) => {
            if verbose {
                eprintln!("lightsim: fused mode off: {why}");
            }
            return Ok(None);
        }
    };
    let len = plan.storage_len();
    let (mat, tab) = material_table(plan);
    let n_materials = tab.len() / 6;
    let n = [plan.nxl, plan.ny(), plan.nz()];
    let mut prof: Vec<Vec<f32>> = (0..3).map(|a| {
        let mut v = vec![0f32; 6 * n[a]];
        for p in 0..n[a] {
            v[6 * p..6 * p + 6].copy_from_slice(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        }
        v
    }).collect();
    let mut nlo = [0usize; 3];
    let mut nhi = [0usize; 3];
    for p in &plan.pml {
        let a = p.axis;
        for q in 0..p.size[a] {
            let pos = p.start[a] + q;
            prof[a][6 * pos..6 * pos + 6].copy_from_slice(&[p.e_b[q], p.e_a[q], p.e_kinv[q], p.h_b[q], p.h_a[q], p.h_kinv[q]]);
        }
        if p.start[a] == 0 {
            nlo[a] = p.size[a];
        } else if p.start[a] + p.size[a] == n[a] {
            nhi[a] = p.size[a];
        } else {
            bail!("PML slab not attached to a boundary");
        }
    }
    let cnt = [
        (nlo[0] + nhi[0]) * n[1] * n[2],
        n[0] * (nlo[1] + nhi[1]) * n[2],
        n[0] * n[1] * (nlo[2] + nhi[2]),
    ];
    let psi = |a: usize| zbuf(s, 2 * cnt[a]);
    let twod = crate::twod::TwoDState::new(plan);
    let ghost_lo = match plan.ghost_h_lo {
        // 2-D plan invariant along x: the 2-D kernel updates every cell
        _ if twod.as_ref().map(|t| !t.k.deferred).unwrap_or(false) && matches!(plan.ghost_h_lo, GhostRule::Periodic { .. }) => 0,
        GhostRule::Zero => 0,
        GhostRule::Mirror { .. } => 2,
        _ => 1,
    };
    let env_u = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let (mat_bytes, mat_mode) = pack_index(mat, n_materials);
    // Tile and x-chunk (LIGHTSIM_TILE=KxJ and LIGHTSIM_CHUNK override them,
    // for measurements). On a large discrete GPU (RTX PRO 6000, 188 SMs) a
    // block holds 1024 threads, 64 along z (32 when z is an odd number of
    // warps, so that no block row is mostly padding), and takes 4-plane
    // x-chunks: enough blocks to fill the SMs evenly on small grids. A grid
    // that 128 x 8 tiles of 8-plane chunks cover in one block per SM keeps
    // those (one wave, no tail). A y-z plane of more cells than the SMs
    // hold in one wave of 1024-thread blocks (all the blocks in flight then
    // work on the same x-chunk) takes 32 x 32 tiles and 2-plane chunks.
    // A small unified-memory GPU (GB10, 48 SMs)
    // is limited by memory bandwidth and takes long z-tiles, and 16-plane
    // chunks once a plane no longer fits its cache.
    let big = sm_count >= 100;
    let nz32 = (n[2].div_ceil(32) * 32) as u32;
    let one_wave = (n[2] as u32).div_ceil(nz32.min(128)) * (n[1] as u32).div_ceil(8) * (n[0] as u32).div_ceil(8) <= sm_count;
    let wide = big && !one_wave && n[1] * n[2] > 1024 * sm_count as usize;
    let default_tile = if big && !one_wave {
        let tz = if nz32.is_multiple_of(64) && !wide { 64 } else { 32 };
        (tz, (1024 / tz).min((n[1].max(1).next_power_of_two() as u32).max(4)))
    } else if big {
        (nz32.min(128), 8)
    } else {
        (nz32.min(256), 4)
    };
    let default_chunk = if wide {
        2
    } else if big && !one_wave {
        4
    } else if big || n[1] * n[2] < 100_000 {
        8
    } else {
        16
    };
    let tile: (u32, u32) = std::env::var("LIGHTSIM_TILE")
        .ok()
        .and_then(|v| v.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))))
        .filter(|&(a, b): &(u32, u32)| a >= 1 && b >= 1 && a * b <= 1024)
        .unwrap_or(default_tile);
    // step_fused_bloch needs more registers than the plain kernel (78 vs
    // 63): at most 512 threads per block, or the launch does not fit an SM
    let mut tile = tile;
    while plan.bloch && tile.0 * tile.1 > 512 && tile.1 > 1 {
        tile.1 /= 2;
    }
    let chunk = env_u("LIGHTSIM_CHUNK").unwrap_or(default_chunk).clamp(1, 4096) as u32;
    Ok(Some(FusedState {
        eb: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
        hb: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
        mat: dbuf(s, &mat_bytes)?,
        mat_mode,
        tile,
        tab: dbuf(s, &tab)?,
        lossy: plan.ca.is_some() as u32,
        prof: [dbuf(s, &prof[0])?, dbuf(s, &prof[1])?, dbuf(s, &prof[2])?],
        psi_e: [psi(0)?, psi(1)?, psi(2)?],
        psi_h: [[psi(0)?, psi(1)?, psi(2)?], [psi(0)?, psi(1)?, psi(2)?]],
        pml_n: [
            (nlo[0] | (nhi[0] << 16)) as u32,
            (nlo[1] | (nhi[1] << 16)) as u32,
            (nlo[2] | (nhi[2] << 16)) as u32,
        ],
        ghost_lo,
        chunk,
        inj_hd: mk_inj(&derived)?,
        n_materials,
        imag: if plan.bloch {
            Some(FusedImag {
                eb: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
                hb: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
                psi_e: [psi(0)?, psi(1)?, psi(2)?],
                psi_h: [[psi(0)?, psi(1)?, psi(2)?], [psi(0)?, psi(1)?, psi(2)?]],
                amps_hd: swap_amps(&derived)?,
            })
        } else {
            None
        },
        twod,
    }))
}
