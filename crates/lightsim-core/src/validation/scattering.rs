//! Scatterers and beams: TFSF, Mie, Gaussian beams, energy balance.

use super::*;

/// TFSF box in vacuum: the plane wave is confined to the total-field region.
pub fn tfsf_leakage(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 25.0;
    let l = 1.2e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    let a = 0.3e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    let freqs = band(3);
    sim.monitors.push(Monitor::DftField { name: "in".into(), min: [0.0; 3], max: [0.0; 3], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::DftField { name: "out".into(), min: [0.45e-6, 0.0, 0.0], max: [0.45e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::FluxBox { name: "net".into(), min: [-0.42e-6; 3], max: [0.42e-6; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("tfsf_leakage");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let inside = res.dft["in"].fields[0].at(i, 0, 0, 0).norm();
        let outside = res.dft["out"].fields[0].at(i, 0, 0, 0).norm();
        rep.check(&format!("|E_in|/|S| @ {:.3} f0", f / f0()), inside / sp[i].norm(), 1.0, 0.02);
        rep.check_abs(&format!("|E_out|/|E_in| @ {:.3} f0", f / f0()), outside / inside, 0.0, 5e-3);
        let inc_face = 0.5 / ETA0 * sp[i].norm_sqr() * (2.0 * a).powi(2);
        rep.check_abs(&format!("|net flux|/P_face @ {:.3} f0", f / f0()), (res.flux["net"][i] / inc_face).abs(), 0.0, 5e-3);
    }
    Ok(rep)
}

/// Mie efficiencies (Bohren & Huffman BHMIE) for relative index m and size
/// parameter x. Returns (Q_ext, Q_sca).
pub fn mie_q(m: Complex64, x: f64) -> (f64, f64) {
    let nstop = (x + 4.0 * x.powf(1.0 / 3.0) + 2.0) as usize;
    let y = m * x;
    let nmx = (nstop as f64).max(y.norm()) as usize + 16;
    let mut d = vec![Complex64::new(0.0, 0.0); nmx + 1];
    for n in (1..=nmx).rev() {
        let en = Complex64::new(n as f64, 0.0) / y;
        d[n - 1] = en - 1.0 / (d[n] + en);
    }
    let (mut psi0, mut psi1) = (x.cos(), x.sin());
    let (mut chi0, mut chi1) = (-x.sin(), x.cos());
    let mut xi1 = Complex64::new(psi1, -chi1);
    let (mut qsca, mut qext) = (0.0, 0.0);
    for n in 1..=nstop {
        let nf = n as f64;
        let psi = (2.0 * nf - 1.0) / x * psi1 - psi0;
        let chi = (2.0 * nf - 1.0) / x * chi1 - chi0;
        let xi = Complex64::new(psi, -chi);
        let da = d[n] / m + nf / x;
        let db = d[n] * m + nf / x;
        let an = (da * psi - psi1) / (da * xi - xi1);
        let bn = (db * psi - psi1) / (db * xi - xi1);
        qsca += (2.0 * nf + 1.0) * (an.norm_sqr() + bn.norm_sqr());
        qext += (2.0 * nf + 1.0) * (an + bn).re;
        psi0 = psi1;
        psi1 = psi;
        chi0 = chi1;
        chi1 = chi;
        xi1 = Complex64::new(psi1, -chi1);
    }
    (2.0 * qext / (x * x), 2.0 * qsca / (x * x))
}

/// Mie scattering of a dielectric sphere under TFSF illumination: scattered
/// power (flux box in the scattered-field region) versus Mie theory.
pub fn mie_scattering(run: Runner) -> Result<CaseReport> {
    let n_s = 2.0;
    let radius = 0.25e-6;
    let dx = LAMBDA / (25.0 * n_s);
    let l = 1.5e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 45e-15);
    sim.media.push(Medium::from_index(n_s).with_name("bead"));
    sim.structures.push(Structure { name: "sphere".into(), shape: Shape::Sphere { center: [0.0; 3], radius }, medium: "bead".into() });
    let a = 0.4e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    let freqs = band(5);
    let b = 0.52e-6;
    sim.monitors.push(Monitor::FluxBox { name: "sca".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("mie_scattering");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let x = TAU * f / C0 * radius;
        let (_qe, qs) = mie_q(Complex64::new(n_s, 0.0), x);
        let p_exp = qs * PI * radius * radius * 0.5 / ETA0 * sp[i].norm_sqr();
        rep.check(&format!("P_sca/Mie @ {:.3} f0 (Qsca={:.3})", f / f0(), qs), res.flux["sca"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Gaussian beam focused at the source plane: transmitted power equals the
/// analytic 0.5 n/eta0 |A S|^2 pi w0^2 / 2.
pub fn gaussian_beam(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let w0 = 1.0e-6;
    let mut sim = Simulation::new([0.0, -3e-6, -3e-6], [3e-6, 3e-6, 3e-6], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.sources.push(Source::GaussianBeam {
        name: "gb".into(),
        axis: 0,
        position: 0.5e-6,
        direction: Direction::Plus,
        polarization: Component::Ey,
        center: [0.0, 0.0, 0.0],
        waist: w0,
        signal: pulse(),
        amplitude: 1.0,
        focus_distance: 0.0,
        theta: 0.0,
        phi: 0.0,
    });
    let freqs = band(3);
    sim.monitors.push(flux_x("T", 2.0e-6, 0.0, &freqs));
    if let Some(Monitor::Flux { min, max, .. }) = sim.monitors.last_mut() {
        *min = [2.0e-6, -2.9e-6, -2.9e-6];
        *max = [2.0e-6, 2.9e-6, 2.9e-6];
    }
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("gaussian_beam");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let p_exp = 0.5 / ETA0 * sp[i].norm_sqr() * PI * w0 * w0 / 2.0;
        rep.check(&format!("P/P_analytic @ {:.3} f0", f / f0()), res.flux["T"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Tilted Gaussian beam (polar angle 20 deg from +x, azimuth 30 deg, focus
/// 1 um beyond the source plane): the intensity centroid on two planes moves
/// along the stated direction, the power is that of the beam,
/// 0.5/eta0 |A S|^2 pi w0^2 / 2, and nothing is emitted behind the source.
pub fn gaussian_beam_tilted(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 16.0;
    let w0 = 1.5e-6;
    let (theta, phi) = (20f64.to_radians(), 30f64.to_radians());
    let half = 4.0e-6;
    let (xs, x1, x2) = (0.6e-6, 1.2e-6, 2.8e-6);
    let centre = [xs, -0.5e-6, -0.3e-6];
    let sig = TimeSignal::gaussian(f0(), 0.1 * f0());
    let mut sim = Simulation::new([0.0, -half, -half], [3.4e-6, half, half], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 50e-15);
    sim.sources.push(Source::GaussianBeam {
        name: "gb".into(),
        axis: 0,
        position: xs,
        direction: Direction::Plus,
        polarization: Component::Ey,
        center: centre,
        waist: w0,
        signal: sig.clone(),
        amplitude: 1.0,
        focus_distance: 1.0e-6,
        theta,
        phi,
    });
    let f = vec![f0()];
    for (name, x) in [("T", x2), ("R", 0.3e-6)] {
        sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, -1.0], max: [x, 1.0, 1.0], freqs: f.clone(), apodization: Apodization::default() });
    }
    for (name, x) in [("a", x1), ("b", x2)] {
        sim.monitors.push(Monitor::DftField { name: name.into(), min: [x, -half, -half], max: [x, half, half], components: vec![Component::Ey], freqs: f.clone(), apodization: Apodization::default(), stride: 0 });
    }
    let res = run(&sim)?;
    let g = sim.build_grid()?;
    // centroid of |Ey|^2 on a plane: (x of the plane, mean y, mean z)
    let centroid = |name: &str| -> [f64; 3] {
        let fld = &res.dft[name].fields[0];
        let (mut w, mut y, mut z) = (0.0, 0.0, 0.0);
        for j in 0..fld.dims[2] {
            for kk in 0..fld.dims[3] {
                let p2 = fld.at(0, 0, j, kk).norm_sqr();
                let pos = g.position3(Component::Ey, [fld.origin[0], fld.origin[1] + j, fld.origin[2] + kk]);
                w += p2;
                y += p2 * pos[1];
                z += p2 * pos[2];
            }
        }
        [g.node(0, fld.origin[0]), y / w, z / w]
    };
    let (ca, cb) = (centroid("a"), centroid("b"));
    let (ty, tz) = ((cb[1] - ca[1]) / (cb[0] - ca[0]), (cb[2] - ca[2]) / (cb[0] - ca[0]));
    let theta_m = (ty * ty + tz * tz).sqrt().atan().to_degrees();
    let phi_m = tz.atan2(ty).to_degrees();
    // the beam axis passes through `centre` on the source plane
    let xs_node = g.node(0, g.nearest_node(0, xs));
    let y_expected = centre[1] + (ca[0] - xs_node) * theta.tan() * phi.cos();
    let sp = sig.spectrum(&f, res.dt, res.stats.steps, 0.0);
    let p_exp = 0.5 / ETA0 * sp[0].norm_sqr() * PI * w0 * w0 / 2.0;
    let mut rep = CaseReport::new("gaussian_beam_tilted");
    rep.mcells_per_s = res.stats.mcells_per_s;
    // The centroid of |E|^2 on planes that are not normal to the beam moves
    // slightly steeper than the beam axis, because the beam diverges (12 deg
    // for this waist): about 0.5 deg here.
    rep.check_abs("polar angle of the beam (deg)", theta_m, 20.0, 1.0);
    rep.check_abs("azimuth of the beam (deg)", phi_m, 30.0, 1.5);
    rep.check_abs("beam centre on the first plane, y (um)", ca[1] * 1e6, y_expected * 1e6, 0.03);
    rep.check("P/P_analytic", res.flux["T"][0] / p_exp, 1.0, 0.03);
    rep.check_abs("power behind the source / P", (res.flux["R"][0] / p_exp).abs(), 0.0, 0.01);
    Ok(rep)
}

/// Energy conservation: zero net flux through a closed box around a lossless
/// scatterer (relative to the incident power through one face).
pub fn energy_conservation(run: Runner) -> Result<CaseReport> {
    let n_s = 1.8;
    let dx = LAMBDA / (20.0 * n_s);
    let l = 1.4e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.media.push(Medium::from_index(n_s).with_name("cube"));
    sim.structures.push(Structure { name: "c".into(), shape: Shape::Box { min: [-0.15e-6; 3], max: [0.15e-6; 3] }, medium: "cube".into() });
    let a = 0.45e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ez, signal: pulse(), amplitude: 1.0 });
    let freqs = band(3);
    let b = 0.3e-6;
    sim.monitors.push(Monitor::FluxBox { name: "net".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("energy_conservation");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let p_face = 0.5 / ETA0 * sp[i].norm_sqr() * (2.0 * b).powi(2);
        rep.check_abs(&format!("|net flux|/P_face @ {:.3} f0", f / f0()), (res.flux["net"][i] / p_face).abs(), 0.0, 1e-3);
    }
    Ok(rep)
}

/// Energy monitor: a plane-wave pulse of unit amplitude inside a box holds
/// the energy U = eps0 n A c int s(t)^2 dt (half electric, half magnetic), in
/// vacuum and in a dielectric, and nothing is left once it has gone.
pub fn stored_energy(run: Runner) -> Result<CaseReport> {
    let n_cross = 2usize;
    let mut rep = CaseReport::new("stored_energy");
    for n in [1.0f64, 2.0] {
        let dx = LAMBDA / (30.0 * n); // 30 cells per wavelength in the medium
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 8e-6, n_cross);
        sim.media.push(Medium::from_index(n).with_name("bg"));
        sim.background = "bg".into();
        sim.run_time = 40e-15 + 30e-15 * n;
        sim.sources.push(plane_source(1.0e-6, n));
        sim.monitors.push(Monitor::Energy { name: "u".into(), min: [1.5e-6, -1.0, -1.0], max: [7.0e-6, 1.0, 1.0], interval: 10 });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let u = &res.energy["u"].energy;
        // int s^2 dt of the Gaussian pulse: tau sqrt(pi) / 2 (1 + exp(-(w0 tau)^2))
        let tau = 1.0 / (TAU * 0.3 * f0());
        let int_s2 = tau * PI.sqrt() / 2.0 * (1.0 + (-(TAU * f0() * tau).powi(2)).exp());
        let exact = crate::constants::EPS0 * n * w * w * C0 * int_s2;
        let peak = u.iter().cloned().fold(0.0, f64::max);
        // the plateau while the whole pulse is inside the box
        let inside: Vec<f64> = u.iter().cloned().filter(|v| *v > 0.999 * peak).collect();
        rep.check(&format!("U / (eps0 n A c int s^2), n = {n}"), peak / exact, 1.0, 0.01);
        rep.check_abs(&format!("samples on the plateau, n = {n}"), (inside.len() >= 5) as u8 as f64, 1.0, 0.0);
        rep.check_abs(&format!("U after the pulse has left / peak, n = {n}"), u[u.len() - 1] / peak, 0.0, 1e-4);
    }
    Ok(rep)
}
