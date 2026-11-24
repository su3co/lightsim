//! The update coefficients of E and H.
//!
//! E: each component rasterized with sub-pixel smoothing (mixed voxels of
//! transparent dispersive dielectrics keep a weight for their poles), then
//! the frozen poles of stabilised layers, the conductivity of adiabatic
//! absorbers, the diagonal of off-diagonal tensors, complex (Hermitian)
//! tensors and the wall conditions. H: rasterized, with magnetic loss and
//! the magnetic conductivity of absorbers.

use super::layers::{Absorbers, StabilisedLayers};
use super::*;

/// The coefficient arrays of a slab, in local (unpadded) layout while the
/// compiler works and in storage layout for the plan.
pub(super) struct Materials {
    /// E update: curl coefficient, decay factor, eps_inf and medium per local sample
    pub cb_local: [Vec<Real>; 3],
    pub ca_local: [Vec<Real>; 3],
    pub eps_local: [Vec<Real>; 3],
    pub med_local: [Vec<u16>; 3],
    /// mixed voxels of transparent dispersive dielectrics: weight of the
    /// poles per local E sample (NaN = not mixed)
    pub disp_weight: [Vec<Real>; 3],
    /// smoothed dispersive voxels: [kept, fell back to the dominant medium]
    pub disp_subpixel: [usize; 2],
    /// storage layout, for the plan
    pub cb: [Vec<Real>; 3],
    pub ca: Option<[Vec<Real>; 3]>,
    pub ch: Option<[Vec<Real>; 3]>,
    /// mu per local H sample (when any medium is magnetic or an absorber is used)
    pub mu_local: Option<[Vec<Real>; 3]>,
    pub hloss: Option<[HLoss; 3]>,
    pub herm: Option<crate::herm::HermPlan>,
}

pub(super) fn compile(s: &Setup, layers: &StabilisedLayers, absorbers: &Absorbers) -> Result<Materials> {
    let media = &s.media;
    let any_lossy = media.iter().any(|m| m.is_lossy()) || absorbers.any;
    let mut m = e_coefficients(s);
    freeze_layer_poles(s, layers, &mut m);
    if absorbers.any {
        absorber_conductivity(s, layers, absorbers, &mut m)?;
    }
    check_tensor_media(media)?;
    if media.iter().any(|m| m.is_tensor()) {
        tensor_diagonal(s, &mut m);
    }
    // complex permittivity tensors: cb = kappa_aa c dt on their cells
    m.herm = crate::herm::compile_herm(media, &m.med_local, &mut m.cb_local, s.nxl, s.n, s.cdt(), s.periodic, s.opts.nslabs, s.inv)?;
    wall_conditions(s, &mut m);
    let (nxl, plane) = (s.nxl, s.plane);
    m.cb = [
        storage_from_local(&m.cb_local[0], nxl, plane, 0.0),
        storage_from_local(&m.cb_local[1], nxl, plane, 0.0),
        storage_from_local(&m.cb_local[2], nxl, plane, 0.0),
    ];
    m.ca = if any_lossy {
        Some([
            storage_from_local(&m.ca_local[0], nxl, plane, 1.0),
            storage_from_local(&m.ca_local[1], nxl, plane, 1.0),
            storage_from_local(&m.ca_local[2], nxl, plane, 1.0),
        ])
    } else {
        None
    };
    h_coefficients(s, absorbers, &mut m);
    Ok(m)
}

/// E: rasterize each component (sub-pixel smoothing, mixed dispersive
/// voxels) and turn eps into the update coefficients.
fn e_coefficients(s: &Setup) -> Materials {
    let (media, dt) = (&s.media, s.dt);
    let mut cb_local: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut ca_local: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut eps_local: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut med_local: [Vec<u16>; 3] = [vec![], vec![], vec![]];
    let mut disp_weight: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut disp_subpixel = [0usize; 2];
    let dt_courant = courant_step(s.grid, s.inv);
    let mut stable_memo: std::collections::HashMap<(u16, i64, i64), bool> = std::collections::HashMap::new();
    for a in 0..3 {
        let comp = Component::e(a);
        let mut cm = rasterize_component(s.grid, comp, s.x0, s.nxl, &s.structures, media, s.background, &s.ropts);
        let (wv, counts) = apply_mixed_cells(&mut cm, media, a, dt, dt_courant, &mut stable_memo);
        disp_weight[a] = wv;
        disp_subpixel[0] += counts[0];
        disp_subpixel[1] += counts[1];
        let mut cbv = vec![0 as Real; cm.value.len()];
        let mut cav = vec![1 as Real; cm.value.len()];
        cbv.par_iter_mut()
            .zip(cav.par_iter_mut())
            .zip(cm.value.par_iter().zip(cm.medium.par_iter()))
            .for_each(|((cb_o, ca_o), (&eps, &m))| {
                (*ca_o, *cb_o) = e_update_coefficients(&media[m as usize], a, eps, dt);
            });
        cb_local[a] = cbv;
        ca_local[a] = cav;
        eps_local[a] = cm.value;
        med_local[a] = cm.medium;
    }
    Materials {
        cb_local,
        ca_local,
        eps_local,
        med_local,
        disp_weight,
        disp_subpixel,
        cb: [vec![], vec![], vec![]],
        ca: None,
        ch: None,
        mu_local: None,
        hloss: None,
        herm: None,
    }
}

/// Inside the stabilised layers the frozen poles of a medium are part of its
/// high-frequency permittivity.
fn freeze_layer_poles(s: &Setup, layers: &StabilisedLayers, m: &mut Materials) {
    if !layers.variant.iter().flatten().any(|(_, extra)| *extra != 0.0) {
        return;
    }
    let (media, dt, x0, nxl, ny, nz) = (&s.media, s.dt, s.x0, s.nxl, s.ny, s.nz);
    for a in 0..3 {
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = (i * ny + j) * nz + k;
                    let mi = m.med_local[a][o] as usize;
                    let Some((_, extra)) = &layers.variant[mi] else { continue };
                    if *extra == 0.0 || !media[mi].pole_axes[a] || !layers.contains([x0 + i, j, k]) {
                        continue;
                    }
                    // (a mixed voxel holds the medium's poles with the weight w)
                    let w = m.disp_weight[a].get(o).filter(|w| !w.is_nan()).copied().unwrap_or(1.0);
                    let eps = m.eps_local[a][o] + w * *extra as Real;
                    (m.ca_local[a][o], m.cb_local[a][o]) = e_update_coefficients(&media[mi], a, eps, dt);
                    m.eps_local[a][o] = eps;
                }
            }
        }
    }
}

/// The electric conductivity of adiabatic absorbers, in the semi-implicit
/// form of a lossy medium.
fn absorber_conductivity(s: &Setup, layers: &StabilisedLayers, absorbers: &Absorbers, m: &mut Materials) -> Result<()> {
    let (media, dt, x0, nxl, ny, nz) = (&s.media, s.dt, s.x0, s.nxl, s.ny, s.nz);
    for a in 0..3 {
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let sr = absorbers.e_rate(a, [x0 + i, j, k]);
                    if sr == 0.0 {
                        continue;
                    }
                    let o = (i * ny + j) * nz + k;
                    let mi = m.med_local[a][o] as usize;
                    let md = &media[mi];
                    if md.pec {
                        continue;
                    }
                    if !is_plain_medium(md) {
                        bail!(
                            "medium '{}' reaches into an Absorber boundary: only media with a diagonal permittivity, conductivity and per-axis poles can run through an absorber (no tensor, nonlinear, gain or plug-in media); use a PML on that face or end the medium before it",
                            md.name
                        );
                    }
                    let eps = m.eps_local[a][o] as f64;
                    let f = md.sigma[a].max(0.0) * dt / (2.0 * EPS0 * eps) + absorbers.e_loss(s, layers, &m.disp_weight, a, [x0 + i, j, k], o, mi, eps);
                    m.ca_local[a][o] = ((1.0 - f) / (1.0 + f)) as Real;
                    m.cb_local[a][o] = (C0 * dt / eps / (1.0 + f)) as Real;
                }
            }
        }
    }
    Ok(())
}

/// What off-diagonal tensor media cannot be combined with.
fn check_tensor_media(media: &[Medium]) -> Result<()> {
    for m in media.iter().filter(|m| m.is_tensor()) {
        if m.is_dispersive() || m.is_nonlinear() || m.pec {
            bail!("medium '{}': off-diagonal tensors cannot be combined with per-axis poles or nonlinearity yet", m.name);
        }
        if !m.oriented.is_empty() && (m.is_lossy() || m.eps_off.iter().any(|&v| v != 0.0)) {
            bail!("medium '{}': oriented poles need a lossless diagonal eps_inf", m.name);
        }
        if !m.gyro.is_empty() && (m.is_lossy() || m.eps_off.iter().any(|&v| v != 0.0)) {
            bail!("medium '{}': gyrotropic poles need a lossless diagonal eps_inf (no sigma, no eps_off)", m.name);
        }
        for gp in &m.gyro {
            if let Err(e) = gp.params() {
                bail!("medium '{}': {e}", m.name);
            }
        }
        if let Some(d) = &m.director {
            d.check(&m.name)?;
            if m.is_lossy() || m.is_offdiag() || !m.gyro.is_empty() {
                bail!("medium '{}': a director field cannot be combined with sigma, eps_off, oriented or gyrotropic poles", m.name);
            }
        }
    }
    Ok(())
}

/// Per-cell update matrices (M, K) of a tensor medium at local cell
/// (a, i, j, k): a director field is evaluated at the position of the E_a
/// sample.
pub(super) fn tensor_cell_update(s: &Setup, md: &Medium, a: usize, i: usize, j: usize, k: usize) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let cdt = s.cdt();
    match &md.director {
        Some(d) => crate::materials::tensor_update_of(d.eps_at(s.grid.position3(Component::e(a), [s.x0 + i, j, k])), [[0.0; 3]; 3], cdt as f64),
        None => md.tensor_update(cdt as f64),
    }
}

/// Off-diagonal tensors: the linear update applies M_aa and K_aa of the
/// semi-implicit tensor update (staircased).
fn tensor_diagonal(s: &Setup, m: &mut Materials) {
    let (ny, nz) = (s.ny, s.nz);
    for a in 0..3 {
        for o in 0..m.cb_local[a].len() {
            let md = &s.media[m.med_local[a][o] as usize];
            if md.is_offdiag() || md.director.is_some() {
                let (i, j, k) = (o / (ny * nz), (o / nz) % ny, o % nz);
                let (mm, kk) = tensor_cell_update(s, md, a, i, j, k);
                m.cb_local[a][o] = kk[a][a] as Real;
                m.ca_local[a][o] = mm[a][a] as Real;
            }
        }
    }
}

/// Wall conditions on cb: tangential E vanishes on the low side of a PEC
/// (and PML / absorber) face, normal E on the high side of a PMC face.
fn wall_conditions(s: &Setup, m: &mut Materials) {
    let (b, n, x0, nxl, ny, nz) = (&s.sim.boundaries, s.n, s.x0, s.nxl, s.ny, s.nz);
    for a in 0..3 {
        let side = b.axis(a);
        let n_a = n[a];
        for (sd, bd) in side.iter().enumerate() {
            let pec_like = matches!(bd, Boundary::Pec | Boundary::Pml { .. } | Boundary::Absorber { .. });
            let pmc = matches!(bd, Boundary::Pmc);
            // which (component, index along a) to zero
            let mut zero: Vec<(usize, usize)> = vec![];
            if sd == 0 && pec_like {
                zero.push(((a + 1) % 3, 0));
                zero.push(((a + 2) % 3, 0));
            }
            if sd == 1 && pmc {
                zero.push((a, n_a - 1));
            }
            for (c, ia) in zero {
                // visit only the plane index ia along axis a
                let (ir, jr, kr) = match a {
                    0 => {
                        if ia < x0 || ia >= x0 + nxl {
                            continue;
                        }
                        (ia - x0..ia - x0 + 1, 0..ny, 0..nz)
                    }
                    1 => (0..nxl, ia..ia + 1, 0..nz),
                    _ => (0..nxl, 0..ny, ia..ia + 1),
                };
                for i in ir.clone() {
                    for j in jr.clone() {
                        for k in kr.clone() {
                            m.cb_local[c][s.lidx(i, j, k)] = 0.0;
                            m.ca_local[c][s.lidx(i, j, k)] = 1.0;
                        }
                    }
                }
            }
        }
    }
}

/// H: rasterize mu (only when a medium is magnetic or an absorber needs the
/// magnetic conductivity) and record the cells whose H decays.
fn h_coefficients(s: &Setup, absorbers: &Absorbers, m: &mut Materials) {
    let (media, dt, x0, nxl, ny, nz, plane) = (&s.media, s.dt, s.x0, s.nxl, s.ny, s.nz, s.plane);
    let any_abs = absorbers.any;
    let any_magnetic = media.iter().any(|m| m.is_magnetic());
    if !(any_magnetic || any_abs) {
        return;
    }
    let any_hloss = media.iter().any(|m| m.is_magnetic_lossy()) || any_abs;
    let mut arr: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut mus: [Vec<Real>; 3] = [vec![], vec![], vec![]];
    let mut hl: [HLoss; 3] = Default::default();
    for a in 0..3 {
        let cm = rasterize_component(s.grid, Component::h(a), x0, nxl, &s.structures, media, s.background, &s.ropts);
        let mut chv = vec![0 as Real; cm.value.len()];
        for (o, (&mu, &mi)) in cm.value.iter().zip(cm.medium.iter()).enumerate() {
            let (mut da, mut c) = h_update_coefficients(&media[mi as usize], a, mu, dt);
            if any_abs {
                // the magnetic conductivity of an adiabatic absorber
                let sr = absorbers.h_rate(a, [x0 + o / (ny * nz), (o / nz) % ny, o % nz]);
                if sr > 0.0 {
                    let f = media[mi as usize].sigma_m[a].max(0.0) * dt / (2.0 * crate::constants::MU0 * mu as f64) + 0.5 * sr * dt;
                    da = ((1.0 - f) / (1.0 + f)) as Real;
                    c = (C0 * dt / mu as f64 / (1.0 + f)) as Real;
                }
            }
            chv[o] = c;
            if da != 1.0 {
                hl[a].idx.push((o + plane) as u64);
                hl[a].da.push(da);
            }
        }
        arr[a] = storage_from_local(&chv, nxl, plane, 0.0);
        mus[a] = cm.value;
    }
    m.mu_local = Some(mus);
    if any_hloss {
        m.hloss = Some(hl);
    }
    m.ch = Some(arr);
}
