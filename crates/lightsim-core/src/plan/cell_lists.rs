//! The cells of the media that the linear update does not cover: dispersive
//! poles, chi2 / chi3 nonlinearity, chi3 with a Raman response, four-level
//! gain and off-diagonal (tensor, oriented, gyrotropic, director) media.
//! Each list holds storage indices and per-cell coefficients.

use super::layers::{Absorbers, StabilisedLayers};
use super::materials::{Materials, tensor_cell_update};
use super::*;

/// Dispersive tables: the ADE coefficients of every dispersive medium (a
/// second row for a medium whose poles change inside stabilised layers) and
/// the cells of each E component.
pub(super) fn dispersive(s: &Setup, mat: &Materials, layers: &StabilisedLayers, absorbers: &Absorbers) -> Option<DispPlan> {
    let (media, dt, x0, nxl, ny, nz) = (&s.media, s.dt, s.x0, s.nxl, s.ny, s.nz);
    let (cb_local, eps_local, med_local, disp_weight) = (&mat.cb_local, &mat.eps_local, &mat.med_local, &mat.disp_weight);
    let any_abs = absorbers.any;
    let disp_media: Vec<usize> = (0..media.len()).filter(|&m| media[m].is_dispersive()).collect();
    if disp_media.is_empty() {
        return None;
    }
    let npole = disp_media.iter().map(|&m| media[m].poles.len()).max().unwrap();
    let mut coefs = Vec::new();
    for &m in &disp_media {
        for q in 0..npole {
            let c = media[m].poles.get(q).map(|p| p.ade_coefficients(dt)).unwrap_or([1.0, 0.0, 0.0]);
            coefs.extend(c.iter().map(|v| *v as Real));
        }
    }
    // Stabilised absorbing layers: a medium whose poles change inside the
    // layers gets a second row of coefficients, used by its cells there.
    let mut variant_row: Vec<Option<(usize, f64)>> = vec![None; disp_media.len()];
    for (mi, &m) in disp_media.iter().enumerate() {
        let Some((poles, extra)) = &layers.variant[m] else { continue };
        variant_row[mi] = Some((coefs.len() / (3 * npole), *extra));
        for q in 0..npole {
            let c = poles.get(q).map(|p| p.ade_coefficients(dt)).unwrap_or([1.0, 0.0, 0.0]);
            coefs.extend(c.iter().map(|v| *v as Real));
        }
    }
    let mut comps = Vec::new();
    for a in 0..3 {
        let mut d = DispComponent { comp: Component::e(a), idx: vec![], matid: vec![], scale: vec![], mixed: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = s.lidx(i, j, k);
                    let m = med_local[a][o] as usize;
                    if let Some(mi) = disp_media.iter().position(|&x| x == m) && media[m].pole_axes[a] && cb_local[a][o] != 0.0 {
                        d.idx.push(s.sidx(i, j, k));
                        let (row, eps) = match variant_row[mi] {
                            Some((v, extra)) if layers.contains([x0 + i, j, k]) => (v, media[m].eps[a] + extra),
                            _ => (mi, media[m].eps[a]),
                        };
                        d.matid.push(row as u32);
                        // E^{n+1} (1 + f) = E^n (1 - f) + (c dt / eps) curl H - dP / eps,
                        // f = sigma dt / (2 eps0 eps): the semi-implicit conduction
                        // term divides the polarisation term as it does the curl
                        let mut f = media[m].sigma[a].max(0.0) * dt / (2.0 * EPS0 * eps);
                        if any_abs {
                            f += absorbers.e_loss(s, layers, disp_weight, a, [x0 + i, j, k], o, m, eps_local[a][o] as f64);
                        }
                        match disp_weight[a].get(o).filter(|w| !w.is_nan()) {
                            // a mixed voxel: the poles with weight w on eps_inf of the voxel
                            // (no conduction of its own; an absorber's is in f)
                            Some(&w) => {
                                let fa = if any_abs { absorbers.e_loss(s, layers, disp_weight, a, [x0 + i, j, k], o, m, eps_local[a][o] as f64) } else { 0.0 };
                                d.mixed.push(d.scale.len() as u32);
                                d.scale.push((w as f64 / (eps_local[a][o] as f64 * (1.0 + fa))) as Real);
                            }
                            None => d.scale.push((1.0 / (eps * (1.0 + f))) as Real),
                        }
                    }
                }
            }
        }
        if !d.idx.is_empty() {
            comps.push(d);
        }
    }
    if comps.is_empty() { None } else { Some(DispPlan { npole, coefs, comps }) }
}

/// Nonlinear (chi2 / chi3), Raman and four-level gain lists; none of them
/// runs with complex (Bloch) fields.
pub(super) fn nonlinear(s: &Setup, mat: &Materials, complex_fields: bool) -> Result<Option<NonlinPlan>> {
    let comps = instantaneous(s, mat, complex_fields)?;
    let raman = raman(s, mat, complex_fields)?;
    let gain = gain(s, mat, complex_fields)?;
    Ok(if comps.is_empty() && raman.is_none() && gain.is_none() { None } else { Some(NonlinPlan { comps, raman, gain }) })
}

/// chi2 / chi3 without a Raman response.
fn instantaneous(s: &Setup, mat: &Materials, complex_fields: bool) -> Result<Vec<NonlinComponent>> {
    let (media, nxl, ny, nz) = (&s.media, s.nxl, s.ny, s.nz);
    if !media.iter().any(|m| m.is_nonlinear()) {
        return Ok(vec![]);
    }
    if complex_fields {
        bail!("nonlinear (chi2/chi3) media cannot be combined with Bloch boundaries (complex fields)");
    }
    let mut comps = Vec::new();
    for a in 0..3 {
        let mut c = NonlinComponent { comp: Component::e(a), idx: vec![], c2: vec![], c3: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = s.lidx(i, j, k);
                    let md = &media[mat.med_local[a][o] as usize];
                    if (md.chi2[a] != 0.0 || md.chi3[a] != 0.0) && mat.cb_local[a][o] != 0.0 && md.raman.is_none() {
                        let eps = mat.eps_local[a][o] as f64;
                        c.idx.push(s.sidx(i, j, k));
                        c.c2.push((md.chi2[a] / eps) as Real);
                        c.c3.push((md.chi3[a] / eps) as Real);
                    }
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    Ok(comps)
}

/// chi3 with a Raman response.
fn raman(s: &Setup, mat: &Materials, complex_fields: bool) -> Result<Option<RamanPlan>> {
    let (media, dt, nxl, ny, nz) = (&s.media, s.dt, s.nxl, s.ny, s.nz);
    let raman_media: Vec<usize> = (0..media.len()).filter(|&m| media[m].raman.is_some()).collect();
    for &m in &raman_media {
        media[m].raman.as_ref().unwrap().check(&media[m].name)?;
        if media[m].chi3.iter().all(|&c| c == 0.0) {
            bail!("medium '{}': a Raman response needs a non-zero chi3", media[m].name);
        }
    }
    if raman_media.is_empty() {
        return Ok(None);
    }
    if complex_fields {
        bail!("nonlinear (chi3 Raman) media cannot be combined with Bloch boundaries (complex fields)");
    }
    let mut coefs = Vec::new();
    for &m in &raman_media {
        coefs.extend(media[m].raman.as_ref().unwrap().ade_coefficients(dt).iter().map(|v| *v as Real));
    }
    let mut comps = Vec::new();
    for a in 0..3 {
        let mut c = RamanComponent { comp: Component::e(a), idx: vec![], matid: vec![], c2: vec![], c3: vec![], cr: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = s.lidx(i, j, k);
                    let m = mat.med_local[a][o] as usize;
                    let md = &media[m];
                    let Some(r) = &md.raman else { continue };
                    if (md.chi2[a] != 0.0 || md.chi3[a] != 0.0) && mat.cb_local[a][o] != 0.0 {
                        let eps = mat.eps_local[a][o] as f64;
                        c.idx.push(s.sidx(i, j, k));
                        c.matid.push(raman_media.iter().position(|&x| x == m).unwrap() as u32);
                        c.c2.push((md.chi2[a] / eps) as Real);
                        c.c3.push(((1.0 - r.fraction) * md.chi3[a] / eps) as Real);
                        c.cr.push((r.fraction * md.chi3[a] / eps) as Real);
                    }
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    Ok(if comps.is_empty() { None } else { Some(RamanPlan { coefs, comps }) })
}

/// Four-level gain media (rate equations).
fn gain(s: &Setup, mat: &Materials, complex_fields: bool) -> Result<Option<GainPlan>> {
    let (media, dt, x0, nxl, ny, nz) = (&s.media, s.dt, s.x0, s.nxl, s.ny, s.nz);
    let gain_media: Vec<usize> = (0..media.len()).filter(|&m| media[m].gain.is_some()).collect();
    if gain_media.is_empty() {
        return Ok(None);
    }
    if complex_fields {
        bail!("gain media cannot be combined with Bloch boundaries (complex fields)");
    }
    let mut tab = Vec::new();
    let mut init = Vec::new();
    for &m in &gain_media {
        let md = &media[m];
        let g = md.gain.as_ref().unwrap();
        g.check(&md.name, dt)?;
        if md.is_nonlinear() || md.raman.is_some() || md.is_tensor() || md.pec {
            bail!("medium '{}': a gain medium cannot also be nonlinear (chi2, chi3), off-diagonal or gyrotropic, or a perfect conductor", md.name);
        }
        let c = g.ade_coefficients(dt);
        tab.extend([c[0], c[1], c[2], EPS0 / (crate::constants::HBAR * g.omega), dt / g.tau32, dt / g.tau21, dt / g.tau10, dt * g.pump_rate]);
        init.extend(g.initial.iter().map(|f| f * g.density));
    }
    let mut comps = Vec::new();
    for a in 0..3 {
        let mut c = GainComponent { comp: Component::e(a), idx: vec![], row: vec![], scale: vec![], vol: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = s.lidx(i, j, k);
                    let m = mat.med_local[a][o] as usize;
                    if media[m].gain.is_none() || mat.cb_local[a][o] == 0.0 {
                        continue;
                    }
                    let f = media[m].sigma[a].max(0.0) * dt / (2.0 * EPS0 * media[m].eps[a]);
                    c.idx.push(s.sidx(i, j, k));
                    c.row.push(gain_media.iter().position(|&x| x == m).unwrap() as u32);
                    c.scale.push((1.0 / (media[m].eps[a] * (1.0 + f))) as Real);
                    // volume of the E_a sample: primal spacing along a, dual
                    // (periodic-aware) spacings across, as the update uses them
                    let g = [x0 + i, j, k];
                    let vol: f64 = (0..3).map(|q| 1.0 / if q == a { s.inv_p_full[q][g[q]] } else { s.inv_d_full[q][g[q]] } as f64).product();
                    c.vol.push(vol);
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    Ok(if comps.is_empty() { None } else { Some(GainPlan { tab, init, comps }) })
}

/// Off-diagonal permittivity: the cells of tensor and director media (with
/// their oriented poles) and of gyrotropic poles, and the x range the tensor
/// update must cover.
pub(super) fn tensor(s: &Setup, mat: &Materials, bounds: &boundaries::BoundaryCodes) -> Result<Option<TensorPlan>> {
    let (media, dt, nxl, ny, nz, opts) = (&s.media, s.dt, s.nxl, s.ny, s.nz, s.opts);
    let (cb_local, med_local) = (&mat.cb_local, &mat.med_local);
    let (bc_y, bc_z) = (bounds.bc_y, bounds.bc_z);
    let cdt = s.cdt();
    if !media.iter().any(|m| m.is_tensor()) {
        return Ok(None);
    }
    if bounds.complex_fields() {
        bail!("off-diagonal permittivity cannot be combined with Bloch boundaries yet");
    }
    let (mut imin, mut imax) = (usize::MAX, 0usize);
    let mut lossy_t = false;
    // oriented-pole table rows: media with oriented poles
    let op_rows: Vec<usize> = (0..media.len()).filter(|&m| !media[m].oriented.is_empty()).collect();
    let op_np = op_rows.iter().map(|&m| media[m].oriented.len()).max().unwrap_or(0);
    let mut op_tab = Vec::new();
    for &m in &op_rows {
        for q in 0..op_np {
            match media[m].oriented.get(q) {
                Some(op) => {
                    let c = op.pole.ade_coefficients(dt);
                    let u = op.unit();
                    op_tab.extend([c[0] as Real, c[1] as Real, c[2] as Real, u[0] as Real, u[1] as Real, u[2] as Real]);
                }
                None => op_tab.extend([1.0 as Real, 0.0, 0.0, 0.0, 0.0, 0.0]),
            }
        }
    }
    // gyrotropic-pole table rows: media with gyrotropic poles
    let gyro_rows: Vec<usize> = (0..media.len()).filter(|&m| !media[m].gyro.is_empty()).collect();
    let gyro_np = gyro_rows.iter().map(|&m| media[m].gyro.len()).max().unwrap_or(0);
    let mut gyro_tab = Vec::new();
    for &m in &gyro_rows {
        for q in 0..gyro_np {
            match media[m].gyro.get(q) {
                Some(gp) => {
                    for mat in gp.matrices(dt) {
                        for r in mat {
                            gyro_tab.extend(r.iter().map(|v| *v as Real));
                        }
                    }
                }
                None => gyro_tab.extend([0.0 as Real; 27]),
            }
        }
    }
    let mut gyro: Vec<GyroComponent> = (0..3).map(|a| GyroComponent { comp: Component::e(a), idx: vec![], row: vec![], scale: vec![] }).collect();
    let mut comps = Vec::new();
    for a in 0..3 {
        let mut c = TensorComponent { comp: Component::e(a), idx: vec![], c1: vec![], c2: vec![], m1: vec![], m2: vec![], opm: vec![], oscale: vec![] };
        for i in 0..nxl {
            for j in 0..ny {
                for k in 0..nz {
                    let o = s.lidx(i, j, k);
                    let md = &media[med_local[a][o] as usize];
                    if md.is_tensor() && cb_local[a][o] != 0.0 && !md.gyro.is_empty() {
                        let row = gyro_rows.iter().position(|&r| r == med_local[a][o] as usize).unwrap();
                        gyro[a].idx.push(s.sidx(i, j, k));
                        gyro[a].row.push(row as u32);
                        gyro[a].scale.push((1.0 / md.eps[a]) as Real);
                        imin = imin.min(i);
                        imax = imax.max(i);
                    }
                    if (md.is_offdiag() || md.director.is_some()) && cb_local[a][o] != 0.0 {
                        let (mm, kk) = tensor_cell_update(s, md, a, i, j, k);
                        let (b1, b2) = ((a + 1) % 3, (a + 2) % 3);
                        c.idx.push(s.sidx(i, j, k));
                        c.c1.push((kk[a][b1] / cdt as f64) as Real);
                        c.c2.push((kk[a][b2] / cdt as f64) as Real);
                        c.m1.push(mm[a][b1] as Real);
                        c.m2.push(mm[a][b2] as Real);
                        let mi = med_local[a][o] as usize;
                        match op_rows.iter().position(|&r| r == mi) {
                            Some(row) => {
                                c.opm.push(row as u32);
                                c.oscale.push((1.0 / md.eps[a]) as Real);
                            }
                            None => {
                                c.opm.push(u32::MAX);
                                c.oscale.push(0.0);
                            }
                        }
                        lossy_t |= md.is_lossy() || !md.oriented.is_empty();
                        imin = imin.min(i);
                        imax = imax.max(i);
                    }
                }
            }
        }
        if !c.idx.is_empty() {
            comps.push(c);
        }
    }
    let wrap_yz = ((bc_y[0] == BC_PERIODIC) as u32) << 1 | ((bc_z[0] == BC_PERIODIC) as u32) << 2;
    gyro.retain(|g| !g.idx.is_empty());
    // gyrotropic poles are driven by E^n of the neighbours, as oriented poles
    lossy_t |= !gyro.is_empty();
    Ok(if opts.nslabs > 1 {
        // every slab exchanges dD boundary planes with its neighbours
        let lossy = media.iter().any(|m| m.is_tensor() && (m.is_lossy() || !m.oriented.is_empty() || !m.gyro.is_empty()));
        Some(TensorPlan { p0: 0, p1: nxl + 2, i0: 1, i1: nxl + 1, wrap: wrap_yz, cdt, lossy, op_np, op_tab, comps, gyro_np, gyro_tab, gyro })
    } else if comps.is_empty() && gyro.is_empty() {
        None
    } else {
        let wrap_x = matches!(s.sim.boundaries.x[0], Boundary::Periodic);
        let (p0, p1) = if wrap_x { (1, nxl + 1) } else { (imin.max(1), (imax + 3).min(nxl + 1)) };
        Some(TensorPlan { p0, p1, i0: p0, i1: p1, wrap: wrap_x as u32 | wrap_yz, cdt, lossy: lossy_t, op_np, op_tab, comps, gyro_np, gyro_tab, gyro })
    })
}
