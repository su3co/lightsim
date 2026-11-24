//! Rasterization of structures onto the Yee grid with sub-pixel smoothing.
//!
//! For every E component we compute, per voxel (the dual cell owned by that
//! component), the arithmetic mean `<eps>` and the harmonic mean `<1/eps>` of
//! the permittivity, plus an interface normal. The effective permittivity
//! along the component axis `a` is the anisotropic average of Farjadpour et
//! al. (Opt. Lett. 31, 2972, 2006), restricted to the diagonal:
//!
//! ```text
//! 1/eps_eff,a = n_a^2 <1/eps> + (1 - n_a^2) / <eps>
//! ```
//!
//! Fill fractions are exact for boxes and extruded polygons (polygon area by
//! sub-row integration, slab overlap analytically) and sampled for curved
//! shapes. Overlapping structures are composited with a painter's rule: later
//! structures cover earlier ones in proportion to their fill fraction.
//! Conductive and PEC media, and dispersive media that are not transparent
//! dielectrics, are not smoothed: the voxel takes the dominant medium
//! ("staircase"), as in most FDTD codes.
//!
//! Transparent dispersive dielectrics (fitted glasses, nitride, silicon:
//! Lorentz / Debye poles, no Drude pole, no conductivity, Re eps > 0 and
//! negligible loss at the reference frequency `w_c` of the sources) are
//! smoothed with their permittivity at `w_c` when `disp_omega` is set
//! ([`MixedCell`]): the voxel gets the effective permittivity eps_eff(w_c)
//! of the average above, keeps the poles of one dispersive medium A with the
//! weight w_A = d eps_eff / d eps_A, and the rest goes into its
//! high-frequency permittivity eps_eff(w_c) - w_A chi_A(w_c).
//!
//! [`Shape::IndexGrid`] (imported / graded permittivity) is a special kind:
//! for the E components, every voxel whose component position lies inside the
//! box (half-open, like a staircased `Box`) takes the trilinearly interpolated
//! grid permittivity and the structure's medium index. The grid is already a
//! sampled field, so it is not smoothed at its own box faces; structures
//! painted later are composited on top of it as usual. For the H components
//! it behaves as a `Box` of its medium, unless it carries permeability
//! samples (`mu`): then its H components take the interpolated permeability
//! in the same way, and a grid with `mu` only is a `Box` of its medium for
//! the E components.

use crate::Real;
use crate::geometry::{Shape, Structure, covered_length, plane_axes, row_crossings};
use crate::grid::{Component, Grid};
use crate::materials::Medium;
use rayon::prelude::*;

/// Rasterization settings.
#[derive(Clone, Debug)]
pub struct RasterOptions {
    /// Sub-samples per axis used for curved shapes and polygon rows.
    pub subsamples: usize,
    /// Enable anisotropic sub-pixel smoothing of dielectric interfaces.
    pub smoothing: bool,
    /// 2-D simulations: (axis, coordinate, half thickness) of the sheet the
    /// structures are averaged over instead of the cell along the invariant
    /// axis (see `Simulation::raster_slice`).
    pub slice: Option<(usize, f64, f64)>,
    /// Reference angular frequency w_c at which transparent dispersive
    /// dielectrics are smoothed ([`MixedCell`]); `None` keeps them
    /// staircased.
    pub disp_omega: Option<f64>,
    /// Time step of the run: the smoothing uses the susceptibility the
    /// discrete pole update has at w_c (`Pole::chi_discrete`), so that a
    /// mixed voxel is exact at w_c in the scheme. 0 = the continuous one.
    pub disp_dt: f64,
}

impl Default for RasterOptions {
    fn default() -> Self {
        RasterOptions { subsamples: 6, smoothing: true, slice: None, disp_omega: None, disp_dt: 0.0 }
    }
}

/// A voxel that mixes media of which at least one is a transparent
/// dispersive dielectric (see the module documentation). `value` and
/// `medium` of [`ComponentMaterial`] hold the staircase of that voxel (the
/// fallback); the plan uses this instead where its guards allow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedCell {
    /// Local voxel index.
    pub o: usize,
    /// The dispersive medium whose poles the voxel keeps.
    pub medium: u16,
    /// Weight w_A of its poles.
    pub weight: f64,
    /// High-frequency permittivity of the voxel, eps_eff(w_c) - w_A chi_A(w_c).
    pub eps_inf: f64,
    /// eps_eff(w_c).
    pub eps_eff: f64,
}

/// True if `md` is a transparent dispersive dielectric along `axis` at the
/// angular frequency `w`: poles on every axis, none of them Drude, no
/// conductivity, PEC, tensor, nonlinear, gain or plug-in terms, Re eps(w) > 0
/// and Im eps(w) <= 1e-2 Re eps(w).
pub fn transparent_dispersive(md: &Medium, axis: usize, w: f64) -> bool {
    use crate::materials::Pole;
    if !md.is_dispersive() || md.pole_axes.iter().any(|&p| !p) || md.pec || md.is_lossy() || md.is_tensor() || md.is_nonlinear() {
        return false;
    }
    if md.raman.is_some() || md.gain.is_some() || md.gain2.is_some() || md.plugin.is_some() || md.eps_complex.is_some() || md.director.is_some() {
        return false;
    }
    if md.poles.iter().any(|p| matches!(p, Pole::Drude { .. })) {
        return false;
    }
    let e = md.eps_complex(axis, w);
    e.re > 0.0 && e.im.abs() <= 1e-2 * e.re
}

/// Per-voxel material data for one field component over a local box.
#[derive(Clone, Debug)]
pub struct ComponentMaterial {
    pub comp: Component,
    /// Local box dims (nx, ny, nz) and global x offset of the box.
    pub dims: [usize; 3],
    /// Effective relative permittivity (E) or permeability (H) along the
    /// component axis, x-major `[i][j][k]`.
    pub value: Vec<Real>,
    /// Dominant medium index per voxel.
    pub medium: Vec<u16>,
    /// Voxels that mix transparent dispersive dielectrics (empty unless
    /// [`RasterOptions::disp_omega`] is set), see [`MixedCell`].
    pub mixed: Vec<MixedCell>,
}

impl ComponentMaterial {
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.dims[1] + j) * self.dims[2] + k
    }
}

/// Resolved structure: shape plus medium index.
pub struct ResolvedStructure<'a> {
    pub shape: &'a Shape,
    pub medium: usize,
}

/// Resolve structure media names against a media table.
pub fn resolve<'a>(structures: &'a [Structure], media: &[Medium]) -> anyhow::Result<Vec<ResolvedStructure<'a>>> {
    structures
        .iter()
        .map(|s| {
            let m = media
                .iter()
                .position(|m| m.name == s.medium)
                .ok_or_else(|| anyhow::anyhow!("structure '{}' uses unknown medium '{}'", s.name, s.medium))?;
            s.shape.validate().map_err(|e| anyhow::anyhow!("structure '{}': {e}", s.name))?;
            // The plan compiler takes eps_inf of dispersive media and the full
            // tensor of off-diagonal media from the medium itself, so a grid
            // permittivity would be applied inconsistently there.
            if matches!(s.shape, Shape::IndexGrid { .. }) && (media[m].pec || media[m].is_dispersive() || media[m].is_tensor()) {
                anyhow::bail!(
                    "structure '{}': an index_grid medium must be a plain or conductive (sigma) medium, not PEC, dispersive or off-diagonal",
                    s.name
                );
            }
            Ok(ResolvedStructure { shape: &s.shape, medium: m })
        })
        .collect()
}

/// Rasterize component `comp` for x indices `[x0, x0 + nx_local)` of `grid`.
pub fn rasterize_component(
    grid: &Grid,
    comp: Component,
    x0: usize,
    nx_local: usize,
    structures: &[ResolvedStructure],
    media: &[Medium],
    background: usize,
    opts: &RasterOptions,
) -> ComponentMaterial {
    let n = grid.n();
    let (ny, nz) = (n[1], n[2]);
    let dims = [nx_local, ny, nz];
    let ncell = nx_local * ny * nz;
    let axis = comp.axis();
    let is_e = comp.is_e();
    let val_of = |m: usize| -> f64 {
        let md = &media[m];
        if is_e { md.eps[axis] } else { md.mu[axis] }
    };
    // A medium can be blended only if it is a plain (lossless, non-dispersive,
    // non-PEC) dielectric.
    let blendable = |m: usize| -> bool {
        let md = &media[m];
        // (plug-in media are staircased as dispersive media: the plug-in acts on the cells the medium owns)
        !(md.pec || md.is_dispersive() || md.is_lossy() || (!is_e && md.is_magnetic_lossy()) || (is_e && (md.plugin.is_some() || md.eps_complex.is_some())))
    };

    // Accumulators.
    let bg = val_of(background);
    let mut arith = vec![bg as Real; ncell];
    let mut harm = vec![(1.0 / bg) as Real; ncell];
    let mut nsq = vec![-1.0 as Real; ncell]; // n_a^2 of the last partial interface (-1 = none)
    let mut dom = vec![background as u16; ncell];
    let mut dom_frac = vec![1.0 as Real; ncell];
    // Permittivity imposed by an IndexGrid on the voxels it owns (NaN = none).
    // Needed where the blended accumulators are bypassed (lossy / dispersive
    // medium, or smoothing disabled).
    // (E components: permittivity grids; H components: permeability grids)
    let has_grid = structures.iter().any(|s| s.shape.grid_carries(!is_e));
    let mut gridv = vec![Real::NAN; if has_grid { ncell } else { 0 }];
    // Second accumulators: transparent dispersive dielectrics blended with
    // their permittivity at w_c (E components, smoothing on, w_c known).
    let w_c = if opts.smoothing && is_e { opts.disp_omega } else { None };
    let tdisp: Vec<bool> = (0..media.len()).map(|m| w_c.map(|w| transparent_dispersive(&media[m], axis, w)).unwrap_or(false)).collect();
    let mix = tdisp.iter().any(|&t| t);
    // Re eps(w_c) of a transparent dispersive medium, with the discrete-time
    // susceptibility of the run's pole update
    let val2 = |m: usize| -> f64 {
        if !tdisp[m] {
            return val_of(m);
        }
        let (w, md) = (w_c.unwrap(), &media[m]);
        md.eps[axis] + md.poles.iter().map(|p| if opts.disp_dt > 0.0 { p.chi_discrete(w, opts.disp_dt).re } else { p.chi(w).re }).sum::<f64>()
    };
    let blend2 = |m: usize| blendable(m) || tdisp[m];
    let mix0 = {
        let v = val2(background) as Real;
        let t = tdisp[background];
        Mix { a: v, h: 1.0 / v, n2: -1.0, blocked: !blend2(background), m: [if t { background as u16 } else { NO_MEDIUM }, NO_MEDIUM], f: [if t { 1.0 } else { 0.0 }, 0.0] }
    };
    let mut mixv = vec![mix0; if mix { ncell } else { 0 }];

    // Voxel bounds along each axis for this component.
    let mut vb: [Vec<(f64, f64)>; 3] = [
        (x0..x0 + nx_local).map(|i| grid.material_voxel_bounds(comp, 0, i)).collect(),
        (0..ny).map(|j| grid.material_voxel_bounds(comp, 1, j)).collect(),
        (0..nz).map(|k| grid.material_voxel_bounds(comp, 2, k)).collect(),
    ];
    let mut pos: [Vec<f64>; 3] = [
        (x0..x0 + nx_local).map(|i| grid.position(comp, 0, i)).collect(),
        (0..ny).map(|j| grid.position(comp, 1, j)).collect(),
        (0..nz).map(|k| grid.position(comp, 2, k)).collect(),
    ];
    // 2-D simulations: every component sees the structures on the slice plane
    if let Some((a, w, half)) = opts.slice {
        for b in vb[a].iter_mut() {
            *b = (w - half, w + half);
        }
        for q in pos[a].iter_mut() {
            *q = w;
        }
    }
    let ns = opts.subsamples.max(1);
    // ownership tolerance: a millionth of the smallest cell
    let own_tol = 1e-6 * grid.min_cell_size().iter().cloned().fold(f64::INFINITY, f64::min);

    // Group consecutive extruded polygons of the same medium and slab range:
    // their coverage is computed as a true union (abutting or overlapping
    // polygons of one GDS layer must not be composited as separate layers).
    let mut groups: Vec<(usize, usize)> = vec![]; // [start, end) into `structures`
    {
        let mut t = 0;
        while t < structures.len() {
            let mut e = t + 1;
            if let Shape::PolySlab { axis, slab_min, slab_max, .. } = structures[t].shape {
                while e < structures.len() {
                    match structures[e].shape {
                        Shape::PolySlab { axis: a2, slab_min: s2, slab_max: m2, .. }
                            if a2 == axis && s2 == slab_min && m2 == slab_max && structures[e].medium == structures[t].medium =>
                        {
                            e += 1
                        }
                        _ => break,
                    }
                }
            }
            groups.push((t, e));
            t = e;
        }
    }

    for &(g0, g1) in &groups {
        let st = &structures[g0];
        let members = &structures[g0..g1];
        let (mut lo, mut hi) = st.shape.bbox();
        for m in members.iter().skip(1) {
            let (l2, h2) = m.shape.bbox();
            for a in 0..3 {
                lo[a] = lo[a].min(l2[a]);
                hi[a] = hi[a].max(h2[a]);
            }
        }
        // index ranges of voxels overlapping the bbox
        let range = |a: usize| -> (usize, usize) {
            let b = &vb[a];
            let mut r0 = b.len();
            let mut r1 = 0;
            for (t, &(v0, v1)) in b.iter().enumerate() {
                if v1 > lo[a] && v0 < hi[a] {
                    r0 = r0.min(t);
                    r1 = r1.max(t + 1);
                }
            }
            (r0, r1)
        };
        let (i0, i1) = range(0);
        let (j0, j1) = range(1);
        let (k0, k1) = range(2);
        if i0 >= i1 || j0 >= j1 || k0 >= k1 {
            continue;
        }
        let m = st.medium;
        let mv = val_of(m);
        let can_blend = blendable(m);

        // Imported permittivity grid (E components): pure ownership by the
        // component position, value from the grid.
        if st.shape.grid_carries(!is_e) {
            let plane = ny * nz;
            arith[i0 * plane..i1 * plane]
                .par_chunks_mut(plane)
                .zip(harm[i0 * plane..i1 * plane].par_chunks_mut(plane))
                .zip(nsq[i0 * plane..i1 * plane].par_chunks_mut(plane))
                .zip(dom[i0 * plane..i1 * plane].par_chunks_mut(plane))
                .zip(dom_frac[i0 * plane..i1 * plane].par_chunks_mut(plane))
                .zip(gridv[i0 * plane..i1 * plane].par_chunks_mut(plane))
                .enumerate()
                .for_each(|(di, (((((pa, ph), pn), pd), pf), pg))| {
                    let i = i0 + di;
                    for j in j0..j1 {
                        for k in k0..k1 {
                            let p = [pos[0][i], pos[1][j], pos[2][k]];
                            if !center_inside(st.shape, p, own_tol) {
                                continue;
                            }
                            let e = if is_e { st.shape.grid_eps(axis, p) } else { st.shape.grid_mu(p) }.unwrap() as Real;
                            let o = j * nz + k;
                            pa[o] = e;
                            ph[o] = 1.0 / e;
                            pn[o] = -1.0;
                            pd[o] = m as u16;
                            pf[o] = 1.0;
                            pg[o] = e;
                        }
                    }
                });
            // (a voxel an imported grid owns is not mixed)
            if mix {
                for i in i0..i1 {
                    for j in j0..j1 {
                        for k in k0..k1 {
                            if center_inside(st.shape, [pos[0][i], pos[1][j], pos[2][k]], own_tol) {
                                mixv[(i * ny + j) * nz + k].blocked = true;
                            }
                        }
                    }
                }
            }
            continue;
        }

        // Per-voxel fill fraction and normal-squared along `axis`.
        let frac_and_normal = |i: usize, j: usize, k: usize, cache: &PolyCache| -> (f64, f64) {
            match st.shape {
                Shape::Box { min, max } | Shape::IndexGrid { min, max, .. } => {
                    let mut f = 1.0;
                    let mut partial_axis = None;
                    for (a, (bl, bh)) in [vb[0][i], vb[1][j], vb[2][k]].iter().enumerate() {
                        let ov = (bh.min(max[a]) - bl.max(min[a])).max(0.0) / (bh - bl);
                        if ov < 1.0 - 1e-12 {
                            partial_axis = Some(a);
                        }
                        f *= ov;
                    }
                    let n2 = match partial_axis {
                        Some(a) if a == axis => 1.0,
                        Some(_) => 0.0,
                        None => -1.0,
                    };
                    (f, n2)
                }
                Shape::PolySlab { axis: sa, slab_min, slab_max, .. } => {
                    let (u, v) = plane_axes(*sa);
                    let b3 = [vb[0][i], vb[1][j], vb[2][k]];
                    let (zl, zh) = b3[*sa];
                    let lz = (zh.min(*slab_max) - zl.max(*slab_min)).max(0.0) / (zh - zl);
                    if lz <= 0.0 {
                        return (0.0, -1.0);
                    }
                    let ijk = [i, j, k];
                    let area = cache.area(ijk[u], ijk[v]);
                    let f = area * lz;
                    let n2 = if area > 1e-12 && area < 1.0 - 1e-12 {
                        // in-plane interface: normal from the area gradient
                        let (gu, gv) = cache.gradient(ijk[u], ijk[v]);
                        let gn = (gu * gu + gv * gv).max(1e-30);
                        if axis == u {
                            gu * gu / gn
                        } else if axis == v {
                            gv * gv / gn
                        } else {
                            0.0
                        }
                    } else if lz < 1.0 - 1e-12 {
                        if axis == *sa { 1.0 } else { 0.0 }
                    } else {
                        -1.0
                    };
                    (f, n2)
                }
                _ => {
                    // sampled shapes
                    let (bx, by, bz) = (vb[0][i], vb[1][j], vb[2][k]);
                    // quick reject/accept using corners
                    let mut inside = 0usize;
                    let tot = ns * ns * ns;
                    for a in 0..ns {
                        let x = bx.0 + (a as f64 + 0.5) / ns as f64 * (bx.1 - bx.0);
                        for b in 0..ns {
                            let y = by.0 + (b as f64 + 0.5) / ns as f64 * (by.1 - by.0);
                            for c in 0..ns {
                                let z = bz.0 + (c as f64 + 0.5) / ns as f64 * (bz.1 - bz.0);
                                if st.shape.contains([x, y, z]) {
                                    inside += 1;
                                }
                            }
                        }
                    }
                    let f = inside as f64 / tot as f64;
                    let n2 = if inside > 0 && inside < tot {
                        let p = [pos[0][i], pos[1][j], pos[2][k]];
                        match st.shape.normal_hint(p) {
                            Some(nv) => nv[axis] * nv[axis],
                            None => 1.0 / 3.0,
                        }
                    } else {
                        -1.0
                    };
                    (f, n2)
                }
            }
        };

        // Polygon area cache over the in-plane voxel ranges.
        let cache = match st.shape {
            Shape::PolySlab { axis: sa, .. } => {
                let (u, v) = plane_axes(*sa);
                let ur = [(i0, i1), (j0, j1), (k0, k1)][u];
                let vr = [(i0, i1), (j0, j1), (k0, k1)][v];
                let polys: Vec<&Vec<[f64; 2]>> = members
                    .iter()
                    .filter_map(|m| match m.shape {
                        Shape::PolySlab { vertices, .. } => Some(vertices),
                        _ => None,
                    })
                    .collect();
                PolyCache::build(&polys, &vb[u], &vb[v], ur, vr, ns)
            }
            _ => PolyCache::empty(),
        };

        // Paint, parallel over x planes.
        let plane = ny * nz;
        let chunks_a = arith[i0 * plane..i1 * plane].par_chunks_mut(plane);
        let chunks_h = harm[i0 * plane..i1 * plane].par_chunks_mut(plane);
        let chunks_n = nsq[i0 * plane..i1 * plane].par_chunks_mut(plane);
        let chunks_d = dom[i0 * plane..i1 * plane].par_chunks_mut(plane);
        let chunks_f = dom_frac[i0 * plane..i1 * plane].par_chunks_mut(plane);
        // IndexGrid override planes (empty slices when no IndexGrid is present)
        let mut no_grid = vec![[0 as Real; 0]; if has_grid { 0 } else { i1 - i0 }];
        let chunks_g: Vec<&mut [Real]> = if has_grid {
            gridv[i0 * plane..i1 * plane].chunks_mut(plane).collect()
        } else {
            no_grid.iter_mut().map(|c| &mut c[..]).collect()
        };
        let mut no_mix: Vec<[Mix; 0]> = vec![[]; if mix { 0 } else { i1 - i0 }];
        let chunks_m: Vec<&mut [Mix]> = if mix {
            mixv[i0 * plane..i1 * plane].chunks_mut(plane).collect()
        } else {
            no_mix.iter_mut().map(|c| &mut c[..]).collect()
        };
        let (mv2, blend_m, tdisp_m) = (val2(m) as Real, blend2(m), tdisp[m]);
        chunks_a
            .zip(chunks_h)
            .zip(chunks_n)
            .zip(chunks_d)
            .zip(chunks_f)
            .zip(chunks_g.into_par_iter().zip(chunks_m))
            .enumerate()
            .for_each(|(di, (((((pa, ph), pn), pd), pf), (pg, pm)))| {
                let i = i0 + di;
                for j in j0..j1 {
                    for k in k0..k1 {
                        let (f, n2) = frac_and_normal(i, j, k, &cache);
                        if f <= 0.0 {
                            continue;
                        }
                        let o = j * nz + k;
                        if mix {
                            pm[o].paint(f as Real, n2 as Real, mv2, blend_m, if tdisp_m { m as u16 } else { NO_MEDIUM });
                        }
                        // Staircase ownership: the component position lies inside
                        // the shape (half-open along box/slab extents).
                        let p = [pos[0][i], pos[1][j], pos[2][k]];
                        let center_in = f >= 1.0 - 1e-9 || members.iter().any(|m| center_inside(m.shape, p, own_tol));
                        if !can_blend {
                            // dispersive / lossy / PEC: no blending, pure ownership
                            if center_in {
                                pa[o] = mv as Real;
                                ph[o] = 1.0 / mv as Real;
                                pn[o] = -1.0;
                                pd[o] = m as u16;
                                pf[o] = 1.0;
                                if has_grid {
                                    pg[o] = Real::NAN;
                                }
                            }
                            continue;
                        }
                        let f32f = f as Real;
                        pa[o] = f32f * mv as Real + (1.0 - f32f) * pa[o];
                        ph[o] = f32f / mv as Real + (1.0 - f32f) * ph[o];
                        if f < 1.0 - 1e-9 && n2 >= 0.0 {
                            pn[o] = n2 as Real;
                        } else if f >= 1.0 - 1e-9 {
                            pn[o] = -1.0;
                        }
                        if center_in {
                            pd[o] = m as u16;
                            pf[o] = f32f;
                            if has_grid {
                                pg[o] = Real::NAN;
                            }
                        }
                    }
                }
            });
    }

    // Final effective value per voxel.
    let mut value = vec![0 as Real; ncell];
    value.par_iter_mut().enumerate().for_each(|(o, out)| {
        let d = dom[o] as usize;
        if !blendable(d) || !opts.smoothing {
            *out = match gridv.get(o) {
                Some(g) if !g.is_nan() => *g,
                _ => val_of(d) as Real,
            };
            return;
        }
        let a = arith[o];
        let h = harm[o];
        let n2 = nsq[o];
        if n2 < 0.0 || (a * h - 1.0).abs() < 1e-6 {
            // homogeneous voxel (a*h == 1 iff single value)
            *out = a;
        } else {
            let inv = n2 * h + (1.0 - n2) / a;
            *out = 1.0 / inv;
        }
    });
    // If a non-blendable medium touches a voxel only partially, the dominant
    // rule above already selected it or the dielectric; nothing else to do.
    let medium = dom;
    // Voxels that mix transparent dispersive dielectrics.
    let mixed: Vec<MixedCell> = if !mix {
        vec![]
    } else {
        mixv.par_iter()
            .enumerate()
            .filter_map(|(o, mx)| {
                if mx.blocked || (has_grid && !gridv[o].is_nan()) {
                    return None;
                }
                let live: Vec<usize> = (0..2).filter(|&q| mx.m[q] != NO_MEDIUM && mx.f[q] > 1e-6).collect();
                if live.is_empty() || live.iter().any(|&q| mx.f[q] >= 1.0 - 1e-6) {
                    return None; // no dispersive medium, or one filling the voxel
                }
                let (a, h) = (mx.a as f64, mx.h as f64);
                let n2 = mx.n2 as f64;
                let parallel = n2 < 0.0;
                let eps_eff = if parallel { a } else { 1.0 / (n2 * h + (1.0 - n2) / a) };
                // d eps_eff / d eps_A for each candidate A, and A's susceptibility at w_c
                let cand = live.iter().map(|&q| {
                    let md = mx.m[q] as usize;
                    let ea = val2(md);
                    let fa = mx.f[q] as f64;
                    let wt = if parallel { fa } else { fa * eps_eff * eps_eff * (n2 / (ea * ea) + (1.0 - n2) / (a * a)) };
                    let chi = ea - media[md].eps[axis];
                    (md, wt, chi)
                });
                let (md, wt, chi) = cand.max_by(|x, y| (x.1 * x.2.abs()).partial_cmp(&(y.1 * y.2.abs())).unwrap())?;
                Some(MixedCell { o, medium: md as u16, weight: wt, eps_inf: eps_eff - wt * chi, eps_eff })
            })
            .collect()
    };
    ComponentMaterial { comp, dims, value, medium, mixed }
}

/// No medium (empty slot of [`Mix`]).
const NO_MEDIUM: u16 = u16::MAX;

/// Accumulators of one voxel with transparent dispersive dielectrics blended
/// at w_c: arithmetic and harmonic means, normal, a flag for media that are
/// never blended (the voxel then stays staircased) and the fill fractions of
/// up to two dispersive media (painter's rule, as the means).
#[derive(Clone, Copy, Debug)]
struct Mix {
    a: Real,
    h: Real,
    n2: Real,
    blocked: bool,
    m: [u16; 2],
    f: [Real; 2],
}

impl Mix {
    /// Paint a structure of value `v` covering the fraction `f` with
    /// normal-squared `n2`; `disp` = the medium if it is a transparent
    /// dispersive dielectric, else [`NO_MEDIUM`].
    fn paint(&mut self, f: Real, n2: Real, v: Real, blend: bool, disp: u16) {
        if !blend {
            self.blocked = true;
        } else {
            self.a = f * v + (1.0 - f) * self.a;
            self.h = f / v + (1.0 - f) * self.h;
            if f < 1.0 - 1e-9 && n2 >= 0.0 {
                self.n2 = n2;
            } else if f >= 1.0 - 1e-9 {
                self.n2 = -1.0;
            }
        }
        for q in 0..2 {
            self.f[q] *= 1.0 - f;
            if self.f[q] <= 0.0 {
                self.m[q] = NO_MEDIUM;
                self.f[q] = 0.0;
            }
        }
        if disp != NO_MEDIUM {
            if let Some(q) = (0..2).find(|&q| self.m[q] == disp) {
                self.f[q] += f;
            } else {
                let q = if self.f[0] <= self.f[1] { 0 } else { 1 };
                if f > self.f[q] {
                    self.m[q] = disp;
                    self.f[q] = f;
                }
            }
        }
    }
}

/// Point-in-shape test used for staircase ownership: half-open `[min, max)`
/// along box and slab extents so that abutting shapes never both own a node.
///
/// A face that passes through a component position (to within `tol`, a
/// small fraction of a cell) is treated as lying exactly on it: the lower
/// face owns the position, the upper face does not. Without the tolerance
/// the last bit of the coordinates decided, and a staircased film whose
/// faces lie on grid nodes came out one cell thinner or thicker than drawn
/// in a good part of all placements.
pub fn center_inside(shape: &Shape, p: [f64; 3], tol: f64) -> bool {
    match shape {
        Shape::Box { min, max } | Shape::IndexGrid { min, max, .. } => {
            (0..3).all(|a| p[a] >= min[a] - tol && p[a] < max[a] - tol)
        }
        Shape::PolySlab { vertices, axis, slab_min, slab_max } => {
            if p[*axis] < *slab_min - tol || p[*axis] >= *slab_max - tol {
                return false;
            }
            // the same convention in the plane: a point on a lower / left
            // edge is inside, one on an upper / right edge is outside
            let (u, v) = plane_axes(*axis);
            crate::geometry::point_in_polygon(vertices, p[u] + tol, p[v] + tol)
        }
        other => other.contains(p),
    }
}

/// Cache of polygon area fractions over an in-plane voxel range.
struct PolyCache {
    u0: usize,
    v0: usize,
    nu: usize,
    nv: usize,
    area: Vec<f64>,
}

impl PolyCache {
    fn empty() -> PolyCache {
        PolyCache { u0: 0, v0: 0, nu: 0, nv: 0, area: vec![] }
    }

    fn build(
        polys: &[&Vec<[f64; 2]>],
        vb_u: &[(f64, f64)],
        vb_v: &[(f64, f64)],
        ur: (usize, usize),
        vr: (usize, usize),
        ns: usize,
    ) -> PolyCache {
        // one extra voxel on each side for gradients
        let u0 = ur.0.saturating_sub(1);
        let u1 = (ur.1 + 1).min(vb_u.len());
        let v0 = vr.0.saturating_sub(1);
        let v1 = (vr.1 + 1).min(vb_v.len());
        let nu = u1 - u0;
        let nv = v1 - v0;
        let mut area = vec![0.0; nu * nv];
        area.par_chunks_mut(nu).enumerate().for_each(|(dv, row)| {
            let (yl, yh) = vb_v[v0 + dv];
            let mut cr = Vec::new();
            let mut iv: Vec<(f64, f64)> = Vec::new();
            let mut merged: Vec<f64> = Vec::new();
            for s in 0..ns {
                let y = yl + (s as f64 + 0.5) / ns as f64 * (yh - yl);
                // union of the inside intervals of all polygons on this row
                iv.clear();
                for poly in polys {
                    row_crossings(poly, y, &mut cr);
                    let mut t = 0;
                    while t + 1 < cr.len() {
                        iv.push((cr[t], cr[t + 1]));
                        t += 2;
                    }
                }
                if iv.is_empty() {
                    continue;
                }
                iv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                merged.clear();
                let (mut a, mut b) = iv[0];
                for &(c, d) in iv.iter().skip(1) {
                    if c <= b {
                        b = b.max(d);
                    } else {
                        merged.push(a);
                        merged.push(b);
                        a = c;
                        b = d;
                    }
                }
                merged.push(a);
                merged.push(b);
                for (du, ar) in row.iter_mut().enumerate() {
                    let (xl, xh) = vb_u[u0 + du];
                    *ar += covered_length(&merged, xl, xh) / (xh - xl) / ns as f64;
                }
            }
        });
        PolyCache { u0, v0, nu, nv, area }
    }

    fn at(&self, u: isize, v: isize) -> f64 {
        let du = u - self.u0 as isize;
        let dv = v - self.v0 as isize;
        if du < 0 || dv < 0 || du >= self.nu as isize || dv >= self.nv as isize {
            return 0.0;
        }
        self.area[dv as usize * self.nu + du as usize]
    }

    fn area(&self, u: usize, v: usize) -> f64 {
        self.at(u as isize, v as isize)
    }

    fn gradient(&self, u: usize, v: usize) -> (f64, f64) {
        let (u, v) = (u as isize, v as isize);
        (self.at(u + 1, v) - self.at(u - 1, v), self.at(u, v + 1) - self.at(u, v - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Structure;

    #[test]
    fn box_fill_fraction_and_smoothing() {
        let g = Grid::uniform([0.0; 3], [10, 10, 10], [1.0, 1.0, 1.0]);
        let media = vec![Medium::vacuum(), Medium::dielectric(4.0).with_name("d")];
        let st = vec![Structure {
            name: "b".into(),
            shape: Shape::Box { min: [-1.0, -1.0, -1.0], max: [5.25, 20.0, 20.0] },
            medium: "d".into(),
        }];
        let rs = resolve(&st, &media).unwrap();
        let ex = rasterize_component(&g, Component::Ex, 0, 10, &rs, &media, 0, &RasterOptions::default());
        // Ex at x = 5.5 voxel [5,6] is 25% filled, normal along x -> harmonic mean
        let v = ex.value[ex.idx(5, 4, 4)];
        let harm = 1.0 / (0.25 / 4.0 + 0.75 / 1.0);
        assert!((v as f64 - harm).abs() < 1e-4, "got {v} want {harm}");
        // Ey at x = 5 voxel [4.5, 5.5] is 75% filled, field parallel -> arithmetic
        let ey = rasterize_component(&g, Component::Ey, 0, 10, &rs, &media, 0, &RasterOptions::default());
        let v = ey.value[ey.idx(5, 4, 4)];
        let ar = 0.75 * 4.0 + 0.25;
        assert!((v as f64 - ar).abs() < 1e-4, "got {v} want {ar}");
        assert_eq!(ey.value[ey.idx(2, 4, 4)], 4.0);
        assert_eq!(ey.value[ey.idx(8, 4, 4)], 1.0);
    }

    #[test]
    fn staircased_film_on_grid_nodes_has_its_thickness() {
        // A conductive (not smoothed) film of 6 cells whose faces lie on grid
        // nodes, at 600 placements with coordinates computed in several ways:
        // always 6 cells for the tangential and for the normal component.
        let dx = 5e-9;
        let g = Grid::uniform([0.0; 3], [240, 2, 2], [dx; 3]);
        let media = vec![Medium::vacuum(), Medium::conductor(4.0, 1.0).with_name("film")];
        let mut seen = std::collections::BTreeMap::new();
        for k in 40..190usize {
            for x0 in [k as f64 * dx, (k * 5) as f64 * 1e-9, 0.2e-6 + (k - 40) as f64 * dx, k as f64 * 0.005 * 1e-6] {
                for slab in [false, true] {
                    let shape = if slab {
                        Shape::PolySlab { vertices: vec![[x0, -1.0], [x0 + 30e-9, -1.0], [x0 + 30e-9, 1.0], [x0, 1.0]], axis: 2, slab_min: -1.0, slab_max: 1.0 }
                    } else {
                        Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + 30e-9, 1.0, 1.0] }
                    };
                    let st = vec![Structure { name: "f".into(), shape, medium: "film".into() }];
                    let rs = resolve(&st, &media).unwrap();
                    let count = |c: Component| {
                        let m = rasterize_component(&g, c, 0, 240, &rs, &media, 0, &RasterOptions::default());
                        (0..240).filter(|&i| m.medium[m.idx(i, 0, 0)] == 1).count()
                    };
                    *seen.entry((count(Component::Ey), count(Component::Ex))).or_insert(0usize) += 1;
                }
            }
        }
        assert_eq!(seen.keys().cloned().collect::<Vec<_>>(), vec![(6, 6)], "cells (tangential, normal) -> placements: {seen:?}");
    }

    #[test]
    fn abutting_polygons_are_unioned() {
        let g = Grid::uniform([0.0; 3], [8, 8, 4], [1.0, 1.0, 1.0]);
        let media = vec![Medium::vacuum(), Medium::dielectric(9.0).with_name("si")];
        // two rectangles sharing the edge x = 4.0 (which cuts voxels in half)
        let left = vec![[0.0, 2.0], [4.0, 2.0], [4.0, 6.0], [0.0, 6.0]];
        let right = vec![[4.0, 2.0], [8.0, 2.0], [8.0, 6.0], [4.0, 6.0]];
        let st = vec![
            Structure { name: "a".into(), shape: Shape::PolySlab { vertices: left, axis: 2, slab_min: 0.0, slab_max: 4.0 }, medium: "si".into() },
            Structure { name: "b".into(), shape: Shape::PolySlab { vertices: right, axis: 2, slab_min: 0.0, slab_max: 4.0 }, medium: "si".into() },
        ];
        let rs = resolve(&st, &media).unwrap();
        let ey = rasterize_component(&g, Component::Ey, 0, 8, &rs, &media, 0, &RasterOptions::default());
        // Ey at x node 4 (voxel [3.5, 4.5]) is fully covered by the union
        assert!((ey.value[ey.idx(4, 4, 1)] - 9.0).abs() < 1e-5, "{}", ey.value[ey.idx(4, 4, 1)]);
    }

    #[test]
    fn index_grid_matches_staircased_box_and_interpolates() {
        let g = Grid::uniform([0.0; 3], [10, 8, 6], [1.0, 1.0, 1.0]);
        let lossy = Medium::conductor(2.0, 10.0).with_name("lossy");
        let media = vec![Medium::vacuum(), Medium::dielectric(4.0).with_name("d"), lossy];
        let (min, max) = ([2.3, 1.5, 0.9], [7.0, 6.2, 4.5]);
        let boxed = vec![Structure { name: "b".into(), shape: Shape::Box { min, max }, medium: "d".into() }];
        let constant = Shape::IndexGrid { min, max, dims: [2, 1, 3], eps: vec![4.0; 6], eps_axes: None, mu: vec![] };
        let grid = vec![Structure { name: "g".into(), shape: constant, medium: "d".into() }];
        let stair = RasterOptions { smoothing: false, ..Default::default() };
        for c in [Component::Ex, Component::Ey, Component::Ez] {
            let a = rasterize_component(&g, c, 0, 10, &resolve(&boxed, &media).unwrap(), &media, 0, &stair);
            // the grid is staircased whether or not smoothing is on
            for opts in [&stair, &RasterOptions::default()] {
                let b = rasterize_component(&g, c, 0, 10, &resolve(&grid, &media).unwrap(), &media, 0, opts);
                // equal (f32), or to round-off of the interpolation (f64)
                assert!(a.value.iter().zip(&b.value).all(|(x, y)| (x - y).abs() <= 1e-12 * x.abs()), "{:?} vs {:?}", a.value, b.value);
                assert_eq!(a.medium, b.medium);
            }
        }
        // linear ramp eps = 1 + x along x, in a lossy medium (the value must
        // come from the grid, not from the medium), partly covered by a later box
        let ramp = Shape::IndexGrid { min: [0.0, -1.0, -1.0], max: [8.0, 20.0, 20.0], dims: [2, 1, 1], eps: vec![1.0, 9.0], eps_axes: None, mu: vec![] };
        let st = vec![
            Structure { name: "g".into(), shape: ramp, medium: "lossy".into() },
            Structure { name: "b".into(), shape: Shape::Box { min: [-1.0, -1.0, 3.0], max: [20.0, 20.0, 20.0] }, medium: "d".into() },
        ];
        let rs = resolve(&st, &media).unwrap();
        let ey = rasterize_component(&g, Component::Ey, 0, 10, &rs, &media, 0, &RasterOptions::default());
        let ex = rasterize_component(&g, Component::Ex, 0, 10, &rs, &media, 0, &RasterOptions::default());
        for i in 0..8 {
            assert!((ey.value[ey.idx(i, 4, 1)] - (1.0 + i as Real)).abs() < 1e-6);
            assert_eq!(ey.medium[ey.idx(i, 4, 1)], 2);
            assert!((ex.value[ex.idx(i, 4, 1)] - (1.5 + i as Real)).abs() < 1e-6);
            // covered by the later box
            assert_eq!(ey.value[ey.idx(i, 4, 4)], 4.0);
            assert_eq!(ey.medium[ey.idx(i, 4, 4)], 1);
        }
        // x = 8 is outside (half-open): background
        assert_eq!(ey.value[ey.idx(8, 4, 1)], 1.0);
        assert_eq!(ey.medium[ey.idx(8, 4, 1)], 0);
        // H components see the medium's permeability only
        let hz = rasterize_component(&g, Component::Hz, 0, 10, &rs, &media, 0, &RasterOptions::default());
        assert!(hz.value.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn polyslab_area() {
        let g = Grid::uniform([0.0; 3], [8, 8, 4], [1.0, 1.0, 1.0]);
        let media = vec![Medium::vacuum(), Medium::dielectric(9.0).with_name("si")];
        let tri = vec![[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let st = vec![Structure {
            name: "t".into(),
            shape: Shape::PolySlab { vertices: tri, axis: 2, slab_min: 0.0, slab_max: 4.0 },
            medium: "si".into(),
        }];
        let rs = resolve(&st, &media).unwrap();
        let ez = rasterize_component(&g, Component::Ez, 0, 8, &rs, &media, 0, &RasterOptions::default());
        // deep inside the triangle
        assert!((ez.value[ez.idx(1, 1, 1)] - 9.0).abs() < 1e-5);
        // outside
        assert!((ez.value[ez.idx(7, 7, 1)] - 1.0).abs() < 1e-5);
        // on the hypotenuse: something in between
        let v = ez.value[ez.idx(4, 4, 1)];
        assert!(v > 1.0 && v < 9.0, "{v}");
    }
}
