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
    // Tile and x-chunk: launch_shape below; LIGHTSIM_TILE=KxJ and
    // LIGHTSIM_CHUNK override them, for measurements.
    let scene = shape_scene(plan, &mat, nlo[2].max(nhi[2]));
    let (mat_bytes, mat_mode) = pack_index(mat, n_materials);
    let (default_tile, default_chunk) = launch_shape(n, sm_count, scene);
    let tile: (u32, u32) = std::env::var("LIGHTSIM_TILE")
        .ok()
        .and_then(|v| v.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))))
        .filter(|&(a, b): &(u32, u32)| a >= 1 && b >= 1 && a * b <= 1024)
        .unwrap_or(default_tile);
    let tile = fit_bloch(tile, plan.bloch);
    let chunk = env_u("LIGHTSIM_CHUNK").unwrap_or(default_chunk as usize).clamp(1, 4096) as u32;
    if verbose && twod.is_none() {
        eprintln!(
            "lightsim: fused launch shape {}x{}:{chunk} ({}x{}x{} cells, {}{}, z-PML {})",
            tile.0,
            tile.1,
            n[0],
            n[1],
            n[2],
            if scene.uniform { "one material" } else { "several materials" },
            if scene.bloch { ", Bloch" } else { "" },
            scene.zpml
        );
    }
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

/// step_fused_bloch needs more registers than the plain kernel (78 vs 63):
/// at most 512 threads per block, or the launch does not fit an SM
fn fit_bloch(mut tile: (u32, u32), bloch: bool) -> (u32, u32) {
    while bloch && tile.0 * tile.1 > 512 && tile.1 > 1 {
        tile.1 /= 2;
    }
    tile
}

/// What of a scene, besides its grid, the launch shape's speed depends on:
/// Bloch runs (complex fields, 512-thread blocks), the depth of the z-PML
/// (its cells take the CPML branch inside a z-row of a tile) and whether the
/// medium is uniform: one material over the interior (the benchmark boxes;
/// see [`uniform_interior`]). Built from the plan by [`shape_scene`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ShapeScene {
    pub bloch: bool,
    pub zpml: usize,
    pub uniform: bool,
}

/// The [`ShapeScene`] of a plan whose per-cell material index (from
/// [`material_table`]) is `mat` and whose z-PML is `zpml` layers deep (the
/// deeper side).
pub(crate) fn shape_scene(plan: &Plan, mat: &[u32], zpml: usize) -> ShapeScene {
    ShapeScene { bloch: plan.bloch, zpml, uniform: uniform_interior(plan, mat) }
}

/// Whether every interior cell of the slab has the same entry of the
/// material table: one medium, as in the benchmark boxes the cost model's
/// uniform tables were measured on. Not counted, since the update
/// coefficients there are not those of the medium: the two ghost x-planes
/// of the storage layout (cb = 0), and along each axis that is not periodic
/// the first and the last cell plane of the domain, where the wall
/// conditions zero the tangential E coefficients of a PEC, PML or absorber
/// face (low side) or the normal one of a PMC face (high side). (The
/// kernels' `mat_mode` is not this: those cells always make the table hold
/// more than one entry.)
pub(crate) fn uniform_interior(plan: &Plan, mat: &[u32]) -> bool {
    let (nxl, ny, nz) = (plan.nxl, plan.ny(), plan.nz());
    let plane = ny * nz;
    let inner = |i: usize, n: usize, periodic: bool| periodic || (i > 0 && i + 1 < n);
    let js: Vec<usize> = (0..ny).filter(|&j| inner(j, ny, plan.periodic[1])).collect();
    let (k0, k1) = if plan.periodic[2] { (0, nz) } else { (1, nz.saturating_sub(1).max(1)) };
    let span = (0..nxl)
        .into_par_iter()
        .filter(|&i| inner(plan.x0 + i, plan.n[0], plan.periodic[0]))
        .filter_map(|i| {
            let base = (i + 1) * plane;
            let mut r: Option<(u32, u32)> = None;
            for &j in &js {
                for &v in &mat[base + j * nz + k0..base + j * nz + k1] {
                    r = Some(r.map_or((v, v), |(a, b)| (a.min(v), b.max(v))));
                }
            }
            r
        })
        .reduce_with(|a, b| (a.0.min(b.0), a.1.max(b.1)));
    span.is_none_or(|(a, b)| a == b)
}

/// Default launch shape of the fused 3-D step for a slab of `n` = [nx, ny,
/// nz] cells (z the contiguous axis) on a GPU of `sm_count` SMs: the (z, y)
/// tile (= block shape) and the number of x-planes a block steps through.
/// The update of a cell does not depend on it: results are the same bit for
/// bit whatever the shape; only the speed changes.
///
/// Large discrete GPU (RTX PRO 6000, 188 SMs; a 1024-thread block of the
/// kernel takes a whole SM):
/// - a grid that 128 x 8 tiles of 8-plane chunks cover in one wave of blocks
///   keeps them;
/// - a y-z plane of more cells than one wave of 1024-thread blocks holds
///   (ny * nz > 1024 * SMs; all the blocks in flight then work on the same
///   x-chunk) takes 32 x 32 tiles and 2-plane chunks (Bloch runs excepted:
///   their 512-thread blocks lose 9 % with 2-plane chunks);
/// - otherwise 1024-thread blocks and 4-plane chunks, 64-wide tiles where the
///   z-row splits into whole 64-wide tiles or ends in 32 lanes of one, else
///   32-wide; where a z-row of 97 to 128 cells would be a whole and a partial
///   64-wide tile, the cost model below ([`shape_cost`]) may pick one 128-wide
///   tile instead, with 8- or 4-plane chunks. It does so only for the scene
///   classes it was measured on: uniform media (one material over the
///   interior, see [`uniform_interior`]), uniform Bloch boxes, and scenes of
///   several materials without Bloch phases. A Bloch run of several materials
///   (photonic-crystal slabs, metasurfaces) is not measured and keeps main's
///   shape: for ordinary scenes several materials
///   with a z-PML of 12 layers or more turns the 128-wide tile from a win into
///   a 1-5 % loss, so the uniform Bloch table cannot stand in for it.
///
/// Small unified-memory GPU (GB10, 48 SMs; bandwidth-bound): long z-tiles,
/// and 16-plane chunks once a plane no longer fits its cache.
pub(crate) fn launch_shape(n: [usize; 3], sm_count: u32, scene: ShapeScene) -> ((u32, u32), u32) {
    let bloch = scene.bloch;
    // the scene classes the cost model has measured tables for
    let modelled = scene.uniform || !bloch;
    let big = sm_count >= 100;
    let nz32 = (n[2].max(1).div_ceil(32) * 32) as u32;
    let one_wave = (n[2].max(1) as u32).div_ceil(nz32.min(128)) * (n[1].max(1) as u32).div_ceil(8) * (n[0].max(1) as u32).div_ceil(8) <= sm_count;
    let wide = big && !one_wave && !bloch && n[1] * n[2] > 1024 * sm_count as usize;
    let (tile, chunk) = if big && !one_wave {
        let tz = if wide { 32 } else if nz32.div_ceil(32).is_multiple_of(2) { 64 } else { 32 };
        let ty = (1024 / tz).min((n[1].max(1).next_power_of_two() as u32).max(4));
        let base = ((tz, ty), if wide { 2 } else { 4 });
        if modelled && !wide && nz32 == 128 && n[2] > 96 {
            let cost = |(t, c): ((u32, u32), u32)| shape_cost(n, fit_bloch(t, bloch), c, sm_count, scene);
            let limit = cost(base) * (1.0 - SHAPE_MARGIN);
            let ty8 = 8.min((n[1].max(1).next_power_of_two() as u32).max(4));
            [((128, ty8), 8), ((128, ty8), 4)]
                .into_iter()
                .map(|s| (cost(s), s))
                .filter(|&(c, _)| c < limit - 1e-9)
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, s)| s)
                .unwrap_or(base)
        } else {
            base
        }
    } else if big {
        ((nz32.min(128), 8), 8)
    } else {
        ((nz32.min(256), 4), if n[1] * n[2] < 100_000 { 8 } else { 16 })
    };
    (fit_bloch(tile, bloch), chunk)
}

/// A shape replaces main's 64-wide one only if the model makes it faster by
/// more than this fraction (CONSTANTS: see shape_cost)
const SHAPE_MARGIN: f64 = 0.01;

/// Cost model of a launch shape on a large GPU: the relative time of one step
/// with (tz x ty) tiles of `chunk` x-planes = k * pad * tail, where
/// - tail: the blocks run in waves of one block per SM; at the end of a step
///   the SMs fall idle one by one as their last blocks finish, which costs
///   about DRAIN of one block's time whatever the number of waves: 1 +
///   DRAIN / waves (fewer, longer blocks drain longer);
/// - pad: lanes of partial y-tiles (z-tiles never pad beyond a warp here);
/// - k: time per cell of a 128-wide tile against a whole and a partial 64-wide
///   one (1 for 64- and 32-wide tiles), by z-row length and chunk depth,
///   measured on grids where the two shapes have the same padding and the tail
///   is taken out: for a uniform medium (benchmark boxes; Bloch boxes apart)
///   by z-row length; with several materials (real devices) by z-row
///   length and z-PML depth, which turns the 128-wide tile from 1-8 % faster
///   (10 layers) to 2-5 % slower (12 or more) for z-rows of 104 cells and more.
///
/// Measured: uniform media with the CLI's 12-layer PML (other z-PML depths
/// use the same table; benchmark boxes only), uniform Bloch boxes, and scenes
/// of several materials at z-rows of 100..124 cells and z-PML of 10..24
/// layers (flat outside: below 10 layers the 10-layer column, as fewer PML
/// cells favour the 128-wide tile in every measured row). Not measured, and
/// never passed here by [`launch_shape`]: Bloch runs of several materials
/// (they keep main's shape).
pub(crate) fn shape_cost(n: [usize; 3], tile: (u32, u32), chunk: u32, sm_count: u32, scene: ShapeScene) -> f64 {
    let (tz, ty, c) = (tile.0 as usize, tile.1 as usize, chunk as usize);
    let (nx, ny, nz) = (n[0].max(1), n[1].max(1), n[2].max(1));
    let blocks = (nz.div_ceil(tz) * ny.div_ceil(ty) * nx.div_ceil(c)) as f64;
    let waves = blocks / sm_count as f64;
    let drain = if scene.uniform || scene.bloch { DRAIN } else { DRAIN_REAL };
    let tail = 1.0 + drain / waves;
    let pad = (ny.div_ceil(ty) * ty) as f64 / ny as f64;
    let k = if tz != 128 {
        1.0
    } else if scene.uniform || scene.bloch {
        k128(nz, chunk, scene.bloch)
    } else {
        k128_real(nz, scene.zpml, chunk)
    };
    k * pad * tail
}

/// idle time at the end of a step, in block times: fitted on calibration grids
/// whose 128-wide shapes with 8- and 4-plane chunks differ only in it (vacuum
/// 400 x 400 x 100 and 640 x 320 x 100..124: 0.27..0.30)
const DRAIN: f64 = 0.3;
/// the same with several materials (blocks of unequal cost drain
/// longer): from the 32 M cell scene (800 x 400 x 100, 8- against 4-plane
/// chunks 1.2 % faster) together with the wire scene on 400 x 300 x 100 (1 %
/// slower), whose waves differ 2.6-fold
const DRAIN_REAL: f64 = 0.7;

/// k of a 128-wide tile (see shape_cost) in a uniform medium at z-rows of
/// K_NZ cells, linear in between, flat outside: speed ratios on the benchmark
/// boxes of 800 x 400 x nz and 1200 x 600 x nz cells (best of 3; Bloch: of 2,
/// nz 97 not measured: the value of 100), whose tails are equal; with 4-plane
/// chunks the Bloch boxes were 6 % slower than with 8 at nz 100 and 120
const K_NZ: [f64; 8] = [97.0, 100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 124.0];
const K_VAC8: [f64; 8] = [0.937, 0.929, 0.954, 0.971, 0.973, 0.981, 1.004, 1.007];
const K_VAC4: [f64; 8] = [0.943, 0.936, 0.957, 0.977, 0.979, 0.986, 1.007, 1.011];
const K_BLOCH8: [f64; 8] = [0.894, 0.894, 0.939, 0.964, 0.965, 0.970, 1.007, 1.013];
const K_BLOCH4: [f64; 8] = [0.948, 0.948, 0.995, 1.022, 1.023, 1.028, 1.067, 1.074];

fn k128(nz: usize, chunk: u32, bloch: bool) -> f64 {
    let t = match (bloch, chunk >= 8) {
        (false, true) => &K_VAC8,
        (false, false) => &K_VAC4,
        (true, true) => &K_BLOCH8,
        (true, false) => &K_BLOCH4,
    };
    let z = nz as f64;
    if z <= K_NZ[0] {
        return t[0];
    }
    for i in 1..K_NZ.len() {
        if z <= K_NZ[i] {
            return t[i - 1] + (t[i] - t[i - 1]) * (z - K_NZ[i - 1]) / (K_NZ[i] - K_NZ[i - 1]);
        }
    }
    t[K_NZ.len() - 1]
}

/// k of a 128-wide tile in a scene of several materials, at z-rows
/// of KR_NZ cells and z-PML depths of KR_PZ layers (bilinear in between, flat
/// outside): forward runs of the wire scene (benchmarks/invdes wg3d) on 400 x
/// 300 x nz cells, x/y PML 10 (x/y PML 24 changed nothing), best of 3
const K_REAL8: [[f64; 6]; 7] = [
    [0.902, 0.921, 0.900, 0.929, 0.931, 0.933],
    [0.955, 0.972, 0.973, 1.031, 1.013, 1.024],
    [0.960, 1.006, 0.978, 1.033, 1.019, 1.031],
    [0.965, 1.010, 1.017, 1.038, 0.975, 1.038],
    [0.961, 1.009, 1.012, 1.032, 1.015, 1.038],
    [0.967, 1.012, 1.017, 1.039, 1.024, 1.032],
    [0.971, 1.015, 1.020, 1.040, 1.031, 1.035],
];
const K_REAL4: [[f64; 6]; 7] = [
    [0.923, 0.939, 0.921, 0.945, 0.949, 0.950],
    [0.978, 0.989, 0.993, 1.044, 1.028, 1.041],
    [0.978, 1.020, 0.993, 1.048, 1.034, 1.048],
    [0.984, 1.025, 1.033, 1.052, 0.993, 1.053],
    [0.984, 1.028, 1.031, 1.053, 1.036, 1.055],
    [0.986, 1.028, 1.033, 1.054, 1.041, 1.047],
    [0.990, 1.034, 1.037, 1.055, 1.043, 1.051],
];
const KR_NZ: [f64; 7] = [100.0, 104.0, 108.0, 112.0, 116.0, 120.0, 124.0];
const KR_PZ: [f64; 6] = [10.0, 12.0, 14.0, 16.0, 20.0, 24.0];

fn k128_real(nz: usize, zpml: usize, chunk: u32) -> f64 {
    let t = if chunk >= 8 { &K_REAL8 } else { &K_REAL4 };
    let at = |v: f64, g: &[f64]| -> (usize, f64) {
        if v <= g[0] {
            return (0, 0.0);
        }
        for i in 1..g.len() {
            if v <= g[i] {
                return (i - 1, (v - g[i - 1]) / (g[i] - g[i - 1]));
            }
        }
        (g.len() - 2, 1.0)
    };
    let (i, fi) = at(nz as f64, &KR_NZ);
    let (j, fj) = at(zpml as f64, &KR_PZ);
    let row = |i: usize| t[i][j] + (t[i][j + 1] - t[i][j]) * fj;
    row(i) + (row(i + 1) - row(i)) * fi
}

#[cfg(test)]
mod tests {
    use super::{launch_shape, material_table, shape_scene, ShapeScene};
    use lightsim_core::geometry::{Shape, Structure};
    use lightsim_core::grid::Component;
    use lightsim_core::sources::{Source, TimeSignal};
    use lightsim_core::{compile, Boundaries, Boundary, CompileOptions, Medium, MeshSpec, Plan, Simulation};
    use std::collections::HashMap;

    const RTX: u32 = 188;
    const GB10: u32 = 48;

    /// A box like the CLI's benchmark scenes (25 nm cells, a z-dipole in the
    /// middle) of `n` cells in all, PML included: `pml` layers on the x and y
    /// faces (none with `bloch`: Bloch walls there instead) and `pmlz` on the
    /// z faces; `medium` fills it (None: vacuum); `block`: a silicon block in
    /// the middle third.
    fn box_plan(n: [usize; 3], pml: usize, pmlz: usize, bloch: bool, medium: Option<Medium>, block: bool, slab: Option<(usize, usize)>) -> Plan {
        let dx = 25e-9;
        let f0 = 2e14;
        let layers = |a: usize| if a == 2 { pmlz } else if bloch { 0 } else { pml };
        let size: Vec<f64> = (0..3).map(|a| (n[a] - 2 * layers(a)) as f64 * dx).collect();
        let face = |l: usize| Boundary::Pml { layers: l, params: Default::default(), inside: false };
        let mut b = Boundaries::all_pml();
        b.x = [face(pml), face(pml)];
        b.y = [face(pml), face(pml)];
        b.z = [face(pmlz), face(pmlz)];
        if bloch {
            let k0 = 2.0 * std::f64::consts::PI * f0 / lightsim_core::constants::C0;
            b.x = [Boundary::Bloch { k: 0.3 * k0 }, Boundary::Bloch { k: 0.3 * k0 }];
            b.y = [Boundary::Bloch { k: 0.2 * k0 }, Boundary::Bloch { k: 0.2 * k0 }];
        }
        let mut sim = Simulation::new([0.0; 3], [size[0], size[1], size[2]], MeshSpec::Uniform { dx: [dx; 3] }, b, 1e-13);
        if let Some(m) = medium {
            sim.media.push(m.with_name("bg"));
            sim.background = "bg".into();
        }
        if block {
            sim.media.push(Medium::dielectric(12.0).with_name("si"));
            let third = |a: usize| size[a] / 3.0;
            sim.structures.push(Structure {
                name: "block".into(),
                shape: Shape::Box { min: [third(0), third(1), third(2)], max: [2.0 * third(0), 2.0 * third(1), 2.0 * third(2)] },
                medium: "si".into(),
            });
        }
        sim.sources.push(Source::PointDipole {
            name: "d".into(),
            position: [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
            polarization: Component::Ez,
            signal: TimeSignal::gaussian(f0, 5e13),
            amplitude: 1e-12,
        });
        let opts = match slab {
            Some((x0, nxl)) => CompileOptions { slab: Some((x0, nxl)), nslabs: 2, slab_index: (x0 > 0) as usize, nsteps: Some(10), ..Default::default() },
            None => CompileOptions { nsteps: Some(10), ..Default::default() },
        };
        let plan = compile(&sim, &opts).unwrap();
        assert_eq!(plan.n, n, "box of {n:?} cells");
        plan
    }

    /// The scene of a plan as `build` derives it
    fn scene_of(plan: &Plan) -> ShapeScene {
        let (mat, _) = material_table(plan);
        let zpml = plan.pml.iter().filter(|p| p.axis == 2).map(|p| p.size[2]).max().unwrap_or(0);
        shape_scene(plan, &mat, zpml)
    }

    /// The scene classes of the tests, each from a real (small) plan: the
    /// CLI's vacuum box (12-layer PML), its Bloch version, and boxes with a
    /// block and a z-PML of `z` layers (x/y PML 10), with or without Bloch
    /// walls
    fn vac() -> ShapeScene {
        scene_of(&box_plan([40, 40, 40], 12, 12, false, None, false, None))
    }
    fn bl() -> ShapeScene {
        scene_of(&box_plan([40, 40, 40], 12, 12, true, None, false, None))
    }
    fn devs(bloch: bool) -> HashMap<usize, ShapeScene> {
        [0, 1, 5, 8, 10, 11, 12, 14, 16, 20, 24, 30, 40]
            .into_iter()
            .map(|z| (z, scene_of(&box_plan([40, 40, 2 * z + 24], 10, z, bloch, None, true, None))))
            .collect()
    }
    fn scenes() -> Vec<ShapeScene> {
        let (d, bd) = (devs(false), devs(true));
        vec![vac(), bl(), d[&10], d[&24], bd[&10], bd[&24]]
    }

    /// The flag the cost model's tables are chosen by, and the shape, from
    /// real plans: one material over the interior (the CLI boxes: vacuum,
    /// Bloch, a uniform dispersive medium; also each slab of a two-slab
    /// split) is "uniform", a block in the box is not. On 240 x 160 x 108
    /// cells the four classes get four different shapes.
    #[test]
    fn scene_class_from_real_plans() {
        let n = [240, 160, 108];
        let lorentz = Medium::dielectric(2.25).with_poles(vec![lightsim_core::Pole::Lorentz { delta_eps: 1.0, omega0: 2.5e15, gamma: 1e13 }]);
        // (what, plan, Bloch, uniform, launch shape)
        type Case = (&'static str, Plan, bool, bool, ((u32, u32), u32));
        let cases: [Case; 4] = [
            ("vacuum box", box_plan(n, 12, 12, false, None, false, None), false, true, ((128, 8), 4)),
            ("Bloch box", box_plan(n, 12, 12, true, None, false, None), true, true, ((128, 4), 8)),
            ("box with a block", box_plan(n, 12, 12, false, None, true, None), false, false, ((64, 16), 4)),
            ("block and Bloch walls", box_plan(n, 12, 12, true, None, true, None), true, false, ((64, 8), 4)),
        ];
        for (what, plan, bloch, uniform, shape) in cases {
            let sc = scene_of(&plan);
            assert_eq!(sc, ShapeScene { bloch, zpml: 12, uniform }, "{what}");
            // the kernels' index always holds more than one entry (ghost planes, walls)
            assert!(material_table(&plan).1.len() / 6 > 1, "{what}");
            assert_eq!(launch_shape(plan.n, RTX, sc), shape, "{what}");
        }
        assert!(scene_of(&box_plan([40, 40, 40], 12, 12, false, Some(lorentz), false, None)).uniform, "uniform Lorentz medium");
        assert!(scene_of(&box_plan([40, 40, 40], 12, 12, false, Some(Medium::dielectric(4.0)), false, None)).uniform, "uniform dielectric");
        for slab in [(0, 20), (20, 20)] {
            assert!(scene_of(&box_plan([40, 40, 40], 12, 12, false, None, false, Some(slab))).uniform, "vacuum slab {slab:?}");
            assert!(!scene_of(&box_plan([40, 40, 40], 12, 12, false, None, true, Some(slab))).uniform, "slab {slab:?} with a block");
        }
        // the test classes: built from plans, never set by hand
        let (d, bd) = (devs(false), devs(true));
        assert!(vac().uniform && !vac().bloch && vac().zpml == 12);
        assert!(bl().uniform && bl().bloch && bl().zpml == 12);
        for (z, sc) in d.iter().chain(bd.iter()) {
            assert!(!sc.uniform && sc.zpml == *z, "block, z-PML {z}");
        }
    }

    /// grids thin, medium and thick along z and y, cubic ones and the
    /// benchmark scenes (x, y, z; z contiguous)
    fn grids() -> Vec<[usize; 3]> {
        let mut g = vec![
            [800, 400, 100], [1200, 600, 120], [400, 200, 76], [640, 320, 120], [400, 400, 100], [1000, 1000, 80],
            [2000, 500, 70], [600, 300, 90], [1000, 1000, 40], [1600, 1600, 24], [512, 512, 250], [600, 600, 200],
            [800, 400, 150], [700, 700, 180], [1000, 500, 140], [300, 300, 1000], [800, 100, 400], [1000, 40, 500],
            [1000, 20, 800], [1600, 64, 400], [2000, 100, 100], [600, 30, 1000], [96, 96, 500], [200, 120, 80],
            [300, 300, 40], [1024, 640, 640], [640, 640, 512], [1600, 800, 800], [800, 800, 800], [392, 288, 108],
        ];
        for c in [64, 100, 128, 144, 160, 192, 224, 256, 320, 400, 512, 640] {
            g.push([c, c, c]);
        }
        for nz in 1..=1100 {
            g.push([600, 300, nz]);
            g.push([300, 200, nz]);
        }
        for ny in 1..=300 {
            g.push([600, ny, 300]);
            g.push([600, ny, 108]);
        }
        g
    }

    fn one_wave(n: [usize; 3]) -> bool {
        let nz32 = n[2].div_ceil(32) * 32;
        n[2].div_ceil(nz32.min(128)) * n[1].div_ceil(8) * n[0].div_ceil(8) <= RTX as usize
    }

    /// main's launch shape (a504a78): the one the cost model starts from
    fn main_shape(n: [usize; 3], bloch: bool) -> ((u32, u32), u32) {
        let nz32 = (n[2].div_ceil(32) * 32) as u32;
        let wide = !bloch && n[1] * n[2] > 1024 * RTX as usize;
        let tz = if wide { 32 } else if nz32.div_ceil(32).is_multiple_of(2) { 64 } else { 32 };
        let ty = (1024 / tz).min((n[1].next_power_of_two() as u32).max(4));
        let ty = if bloch && tz * ty > 512 { ty / 2 } else { ty };
        ((tz, ty), if wide { 2 } else { 4 })
    }

    #[test]
    fn blocks_fit_and_chunks_are_sane() {
        let scenes = scenes();
        for n in grids() {
            for sm in [RTX, GB10] {
                for &sc in &scenes {
                    let ((tz, ty), c) = launch_shape(n, sm, sc);
                    assert!(tz >= 32 && tz.is_multiple_of(32) && ty >= 1, "{n:?} {sm} {sc:?}: {tz}x{ty}");
                    assert!(tz * ty <= if sc.bloch { 512 } else { 1024 }, "{n:?} {sm} {sc:?}: {tz}x{ty}");
                    assert!((1..=16).contains(&c), "{n:?}: chunk {c}");
                    if sc.bloch {
                        assert!(c >= 4, "{n:?}: Bloch runs lose 9 % with 2-plane chunks");
                    }
                }
            }
        }
    }

    /// On a large GPU no z-row of tiles carries more padding than the warp
    /// granularity forces, no y-tile is longer than the next power of two of
    /// the grid's y extent, and the cost model only ever replaces main's shape
    /// by a 128-wide tile on z-rows of 97 to 128 cells (one tile, no padding
    /// beyond main's)
    #[test]
    fn model_only_changes_rows_of_97_to_128_cells() {
        let scenes = scenes();
        for n in grids().into_iter().filter(|&n| !one_wave(n)) {
            for &sc in &scenes {
                let s = launch_shape(n, RTX, sc);
                let ((tz, ty), _) = s;
                let (nz, nz32) = (n[2] as u32, n[2].div_ceil(32) as u32 * 32);
                assert_eq!(nz.div_ceil(tz) * tz, nz32, "{n:?}: z-tiles of {tz} pad {nz} beyond {nz32}");
                assert!(ty <= (n[1].next_power_of_two() as u32).max(4), "{n:?}: y-tile {ty}");
                if s != main_shape(n, sc.bloch) {
                    assert!((97..=128).contains(&nz) && tz == 128, "{n:?} {sc:?}: {s:?}");
                }
            }
        }
    }

    /// A Bloch run of several materials is not a measured class:
    /// it gets exactly main's shape on every grid of the test set, at any
    /// z-PML depth (thin, the measured range and thick)
    #[test]
    fn bloch_with_materials_keeps_main_shape() {
        let bd = devs(true);
        for n in grids() {
            for (&zpml, &sc) in &bd {
                let want = if one_wave(n) {
                    let nz32 = (n[2].div_ceil(32) * 32) as u32;
                    let t = (nz32.min(128), 8);
                    (if t.0 * t.1 > 512 { (t.0, t.1 / 2) } else { t }, 8)
                } else {
                    main_shape(n, true)
                };
                assert_eq!(launch_shape(n, RTX, sc), want, "{n:?} z-PML {zpml}");
            }
        }
    }

    /// The shape the model picks on every grid measured on one RTX PRO 6000
    /// (s24 sweeps): a change of a constant that flips one of them fails here
    #[test]
    fn shapes_of_measured_grids() {
        let s = |n: [usize; 3], sc: ShapeScene| launch_shape(n, RTX, sc);
        let (vac, bl, d) = (vac(), bl(), devs(false));
        let dev = |z: usize| d[&z];
        // gradient scenes: 32 M (800 x 400 x 100) and 86 M (1200 x 600 x 120)
        // cells, z-PML 10: 128 x 8 tiles of 8 planes, 9.6 and 3.6 % faster
        // than 64 x 16 : 4 (0.1.0's shape); the 22 nm multi-port scene (392 x
        // 288 x 108, z-PML 24): main's 64 x 16 : 4, 4 % faster than 128 x 8
        assert_eq!(s([800, 400, 100], dev(10)), ((128, 8), 8));
        assert_eq!(s([1200, 600, 120], dev(10)), ((128, 8), 8));
        assert_eq!(s([392, 288, 108], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 200, 76], dev(10)), ((32, 32), 4));
        assert_eq!(s([1000, 40, 500], bl), ((64, 8), 4));
        assert_eq!(s([1200, 600, 100], bl), ((128, 4), 8));
        assert_eq!(s([1200, 600, 104], bl), ((128, 4), 8));
        assert_eq!(s([1200, 600, 108], bl), ((128, 4), 8));
        assert_eq!(s([1200, 600, 112], bl), ((128, 4), 8));
        assert_eq!(s([1200, 600, 116], bl), ((128, 4), 8));
        assert_eq!(s([1200, 600, 120], bl), ((64, 8), 4));
        assert_eq!(s([1200, 600, 124], bl), ((64, 8), 4));
        assert_eq!(s([128, 128, 128], bl), ((64, 8), 4));
        assert_eq!(s([400, 400, 400], bl), ((32, 16), 4));
        assert_eq!(s([512, 512, 250], bl), ((64, 8), 4));
        assert_eq!(s([800, 400, 100], bl), ((128, 4), 8));
        assert_eq!(s([800, 400, 104], bl), ((128, 4), 8));
        assert_eq!(s([800, 400, 108], bl), ((128, 4), 8));
        assert_eq!(s([800, 400, 112], bl), ((128, 4), 8));
        assert_eq!(s([800, 400, 116], bl), ((128, 4), 8));
        assert_eq!(s([800, 400, 120], bl), ((64, 8), 4));
        assert_eq!(s([800, 400, 124], bl), ((64, 8), 4));
        assert_eq!(s([1000, 40, 500], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 120], vac), ((64, 16), 4));
        assert_eq!(s([128, 128, 128], vac), ((64, 16), 4));
        assert_eq!(s([400, 400, 400], vac), ((32, 32), 4));
        assert_eq!(s([512, 512, 250], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 100], vac), ((128, 8), 8));
        assert_eq!(s([392, 288, 108], dev(24)), ((64, 16), 4));
        assert_eq!(s([800, 400, 100], dev(10)), ((128, 8), 8));
        assert_eq!(s([1200, 600, 120], dev(10)), ((128, 8), 8));
        assert_eq!(s([1000, 1000, 40], vac), ((64, 16), 4));
        assert_eq!(s([1000, 1000, 80], vac), ((32, 32), 4));
        assert_eq!(s([1000, 20, 800], vac), ((32, 32), 4));
        assert_eq!(s([1000, 500, 140], vac), ((32, 32), 4));
        assert_eq!(s([1200, 600, 100], vac), ((128, 8), 8));
        assert_eq!(s([1200, 600, 104], vac), ((128, 8), 8));
        assert_eq!(s([1200, 600, 108], vac), ((128, 8), 8));
        assert_eq!(s([1200, 600, 112], vac), ((128, 8), 8));
        assert_eq!(s([1200, 600, 116], vac), ((128, 8), 8));
        assert_eq!(s([1200, 600, 124], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 225], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 230], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 236], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 240], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 246], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 250], vac), ((64, 16), 4));
        assert_eq!(s([1200, 600, 97], vac), ((128, 8), 8));
        assert_eq!(s([160, 160, 160], vac), ((32, 32), 4));
        assert_eq!(s([1600, 1600, 24], vac), ((32, 32), 4));
        assert_eq!(s([1600, 64, 400], vac), ((32, 32), 4));
        assert_eq!(s([192, 192, 192], vac), ((64, 16), 4));
        assert_eq!(s([2000, 100, 100], vac), ((128, 8), 4));
        assert_eq!(s([2000, 200, 100], vac), ((128, 8), 8));
        assert_eq!(s([2000, 500, 70], vac), ((32, 32), 4));
        assert_eq!(s([256, 256, 256], vac), ((64, 16), 4));
        assert_eq!(s([300, 300, 1000], vac), ((32, 32), 2));
        assert_eq!(s([400, 400, 100], vac), ((128, 8), 4));
        assert_eq!(s([512, 512, 512], vac), ((32, 32), 2));
        assert_eq!(s([600, 30, 1000], vac), ((64, 16), 4));
        assert_eq!(s([600, 300, 90], vac), ((32, 32), 4));
        assert_eq!(s([600, 600, 200], vac), ((32, 32), 4));
        assert_eq!(s([640, 320, 100], vac), ((128, 8), 4));
        assert_eq!(s([640, 320, 104], vac), ((128, 8), 4));
        assert_eq!(s([640, 320, 108], vac), ((128, 8), 4));
        assert_eq!(s([640, 320, 112], vac), ((128, 8), 4));
        assert_eq!(s([640, 320, 116], vac), ((128, 8), 4));
        assert_eq!(s([640, 320, 120], vac), ((64, 16), 4));
        assert_eq!(s([640, 320, 124], vac), ((64, 16), 4));
        assert_eq!(s([640, 640, 640], vac), ((32, 32), 2));
        assert_eq!(s([700, 700, 180], vac), ((64, 16), 4));
        assert_eq!(s([800, 100, 400], vac), ((32, 32), 4));
        assert_eq!(s([800, 400, 104], vac), ((128, 8), 4));
        assert_eq!(s([800, 400, 108], vac), ((128, 8), 8));
        assert_eq!(s([800, 400, 112], vac), ((128, 8), 8));
        assert_eq!(s([800, 400, 116], vac), ((128, 8), 4));
        assert_eq!(s([800, 400, 120], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 124], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 150], vac), ((32, 32), 4));
        assert_eq!(s([800, 400, 225], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 230], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 236], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 240], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 246], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 250], vac), ((64, 16), 4));
        assert_eq!(s([800, 400, 97], vac), ((128, 8), 8));
        assert_eq!(s([400, 300, 100], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 100], dev(12)), ((128, 8), 4));
        assert_eq!(s([400, 300, 100], dev(14)), ((128, 8), 4));
        assert_eq!(s([400, 300, 100], dev(16)), ((128, 8), 4));
        assert_eq!(s([400, 300, 100], dev(20)), ((128, 8), 4));
        assert_eq!(s([400, 300, 100], dev(24)), ((128, 8), 4));
        assert_eq!(s([400, 300, 104], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 104], dev(12)), ((128, 8), 4));
        assert_eq!(s([400, 300, 104], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 104], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 104], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 104], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 300, 108], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 108], dev(12)), ((64, 16), 4));
        assert_eq!(s([400, 300, 108], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 108], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 108], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 108], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 300, 112], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 112], dev(12)), ((64, 16), 4));
        assert_eq!(s([400, 300, 112], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 112], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 112], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 112], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 300, 116], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 116], dev(12)), ((64, 16), 4));
        assert_eq!(s([400, 300, 116], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 116], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 116], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 116], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 300, 120], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 120], dev(12)), ((64, 16), 4));
        assert_eq!(s([400, 300, 120], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 120], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 120], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 120], dev(24)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(10)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(12)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(14)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(16)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(20)), ((64, 16), 4));
        assert_eq!(s([400, 300, 124], dev(24)), ((64, 16), 4));
        assert_eq!(s([1600, 400, 104], bl), ((128, 4), 8));
        assert_eq!(s([256, 256, 100], bl), ((128, 4), 8));
        assert_eq!(s([392, 288, 108], bl), ((128, 4), 8));
        assert_eq!(s([1600, 400, 104], vac), ((128, 8), 8));
        assert_eq!(s([256, 256, 100], vac), ((128, 8), 4));
        assert_eq!(s([392, 288, 108], vac), ((128, 8), 4));
        assert_eq!(s([392, 288, 108], dev(10)), ((128, 8), 4));
        assert_eq!(s([392, 288, 108], dev(16)), ((64, 16), 4));
        assert_eq!(s([1200, 600, 120], dev(24)), ((64, 16), 4));
        assert_eq!(s([1600, 400, 104], dev(10)), ((128, 8), 8));
        assert_eq!(s([256, 256, 100], dev(10)), ((128, 8), 4));
        assert_eq!(s([300, 600, 112], dev(10)), ((128, 8), 4));
        assert_eq!(s([400, 300, 110], dev(10)), ((128, 8), 4));
        assert_eq!(s([500, 200, 104], dev(10)), ((128, 8), 4));
        assert_eq!(s([800, 400, 100], dev(24)), ((128, 8), 8));
        assert_eq!(s([1000, 200, 100], vac), ((128, 8), 4));
        assert_eq!(s([1000, 200, 60], vac), ((64, 16), 4));
        assert_eq!(s([200, 1000, 116], vac), ((128, 8), 4));
        assert_eq!(s([300, 300, 200], vac), ((32, 32), 4));
        assert_eq!(s([300, 600, 112], vac), ((128, 8), 4));
        assert_eq!(s([320, 320, 97], vac), ((128, 8), 4));
        assert_eq!(s([392, 288, 124], vac), ((64, 16), 4));
        assert_eq!(s([392, 288, 90], vac), ((32, 32), 4));
        assert_eq!(s([400, 300, 110], vac), ((128, 8), 4));
        assert_eq!(s([400, 300, 140], vac), ((32, 32), 4));
        assert_eq!(s([500, 200, 104], vac), ((128, 8), 4));
        assert_eq!(s([600, 600, 120], vac), ((64, 16), 4));
    }

    /// The GB10 path is that of 0.1.0
    #[test]
    fn small_gpu_unchanged() {
        let scenes = scenes();
        for n in grids() {
            for &sc in &scenes {
                let nz32 = (n[2].div_ceil(32) * 32) as u32;
                let c = if n[1] * n[2] < 100_000 { 8 } else { 16 };
                let t = (nz32.min(256), 4);
                let t = if sc.bloch && t.0 * t.1 > 512 { (t.0, t.1 / 2) } else { t };
                assert_eq!(launch_shape(n, GB10, sc), (t, c), "{n:?}");
            }
        }
    }
}
