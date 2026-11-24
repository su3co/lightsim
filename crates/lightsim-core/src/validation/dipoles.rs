//! Point dipoles: radiated power, Green's function, far field, tilted
//! dipoles, a dipole in a dielectric.

use super::*;

/// Electric dipole in vacuum: radiated power (closed flux box) versus the
/// analytic P = eta0 k^2 |I l S(w)|^2 / (12 pi).
pub fn dipole_radiation(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 24.0;
    let l = 1.6e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    let sig = pulse();
    sim.sources.push(Source::PointDipole {
        name: "d".into(),
        position: [0.0; 3],
        polarization: Component::Ez,
        signal: sig.clone(),
        amplitude: 1e-12,
    });
    let freqs = band(5);
    let b = 0.45e-6;
    sim.monitors.push(Monitor::FluxBox {
        name: "P".into(),
        min: [-b; 3],
        max: [b; 3],
        freqs: freqs.clone(),
        apodization: Apodization::default(),
    });
    let res = run(&sim)?;
    // the dipole current uses half-step samples
    let spc = sig.spectrum(&freqs, res.dt, res.stats.steps, 0.5);
    let mut rep = CaseReport::new("dipole_radiation");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let k = TAU * f / C0;
        let p_exp = ETA0 * k * k * (1e-12f64).powi(2) * spc[i].norm_sqr() / (12.0 * PI);
        rep.check(&format!("P/P_analytic @ {:.3} f0", f / f0()), res.flux["P"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Magnetic dipole (Hz current element K): radiated power versus the analytic
/// P = k^2 |K S(w)|^2 / (12 pi eta0).
pub fn magnetic_dipole(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 24.0;
    let l = 1.6e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    let sig = pulse();
    let kmom = 1e-12 * ETA0;
    sim.sources.push(Source::PointDipole { name: "m".into(), position: [0.0; 3], polarization: Component::Hz, signal: sig.clone(), amplitude: kmom });
    let freqs = band(5);
    let b = 0.45e-6;
    sim.monitors.push(Monitor::FluxBox { name: "P".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let spc = sig.spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("magnetic_dipole");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let k = TAU * f / C0;
        let p_exp = k * k * kmom * kmom * spc[i].norm_sqr() / (12.0 * PI * ETA0);
        rep.check(&format!("P/P_analytic @ {:.3} f0", f / f0()), res.flux["P"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Dipole Green's function: ratio and phase difference of E_z at two points
/// in the equatorial plane versus the analytic dipole field
/// E_z ~ exp(ikr)/r^3 [ (kr)^2 - (1 - ikr) ].
pub fn dipole_green_function(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let l = 3.0e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 45e-15);
    sim.sources.push(Source::PointDipole { name: "d".into(), position: [0.0; 3], polarization: Component::Ez, signal: pulse(), amplitude: 1e-12 });
    let g = sim.build_grid()?;
    let z0 = g.position(Component::Ez, 2, g.nearest_index(Component::Ez, 2, 0.0));
    let pts = [0.5e-6, 1.0e-6];
    for (i, x) in pts.iter().enumerate() {
        sim.monitors.push(Monitor::DftField {
            name: format!("p{i}"),
            min: [*x, 0.0, z0],
            max: [*x, 0.0, z0],
            components: vec![Component::Ez],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
    }
    let res = run(&sim)?;
    let e: Vec<Complex64> = (0..2).map(|i| res.dft[&format!("p{i}")].fields[0].at(0, 0, 0, 0)).collect();
    let xs: Vec<f64> = (0..2).map(|i| g.node(0, res.dft[&format!("p{i}")].fields[0].origin[0])).collect();
    let k = TAU * f0() / C0;
    let green = |r: f64| -> Complex64 {
        let kr = k * r;
        Complex64::from_polar(1.0 / (r * r * r), kr) * (Complex64::new(kr * kr - 1.0, kr))
    };
    let ratio = (e[1] / e[0]).norm();
    let ratio_exp = (green(xs[1]) / green(xs[0])).norm();
    let dphi = (e[1] / e[0]).arg();
    let dphi_exp = (green(xs[1]) / green(xs[0])).arg();
    let mut rep = CaseReport::new("dipole_green_function");
    rep.mcells_per_s = res.stats.mcells_per_s;
    rep.check("|E(r2)/E(r1)|", ratio, ratio_exp, 0.02);
    rep.check_abs("arg(E(r2)/E(r1)) [rad]", dphi, dphi_exp, 0.05);
    Ok(rep)
}

/// Far-field projection of a z-dipole: U(theta) ~ sin^2(theta), no radiation
/// along the axis, and the integrated far-field power equals the near-field
/// flux through the projection box.
pub fn dipole_far_field(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 24.0;
    let l = 1.6e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.sources.push(Source::PointDipole { name: "d".into(), position: [0.0; 3], polarization: Component::Ez, signal: pulse(), amplitude: 1e-12 });
    let nt = 37usize;
    let npp = 24usize;
    let theta: Vec<f64> = (0..nt).map(|i| PI * i as f64 / (nt - 1) as f64).collect();
    let phi: Vec<f64> = (0..npp).map(|i| TAU * i as f64 / npp as f64).collect();
    let b = 0.45e-6;
    sim.monitors.push(Monitor::FarField { name: "ff".into(), min: [-b; 3], max: [b; 3], freqs: vec![f0()], theta: theta.clone(), phi: phi.clone(), n_medium: 1.0, apodization: Apodization::default() });
    let res = run(&sim)?;
    let ff = &res.far_field["ff"];
    let u = |ti: usize, pi: usize| ff.intensity[ti * npp + pi];
    let u90 = (0..npp).map(|p| u(18, p)).sum::<f64>() / npp as f64;
    let u45 = (0..npp).map(|p| u(9, p)).sum::<f64>() / npp as f64;
    let u0 = (0..npp).map(|p| u(0, p)).sum::<f64>() / npp as f64;
    // integrate U over the sphere (trapezoid in theta, uniform in phi)
    let mut ptot = 0.0;
    for ti in 0..nt {
        let w = if ti == 0 || ti == nt - 1 { 0.5 } else { 1.0 };
        let avg = (0..npp).map(|p| u(ti, p)).sum::<f64>() / npp as f64;
        ptot += w * avg * theta[ti].sin() * (PI / (nt - 1) as f64) * TAU;
    }
    let phi_var = (0..npp).map(|p| (u(18, p) / u90 - 1.0).abs()).fold(0.0, f64::max);
    let mut rep = CaseReport::new("dipole_far_field");
    rep.mcells_per_s = res.stats.mcells_per_s;
    rep.check("U(45)/U(90) = sin^2(45)", u45 / u90, 0.5, 0.02);
    rep.check_abs("U(0)/U(90)", u0 / u90, 0.0, 0.01);
    rep.check_abs("azimuthal variation of U(90)", phi_var, 0.0, 0.02);
    rep.check("integral U dOmega / box flux", ptot / res.flux["ff"][0], 1.0, 0.02);
    Ok(rep)
}

/// Fraction of the power of a sin^2(theta) pattern (dipole along `p`, unit
/// vector) that leaves through the two faces of a centred cube normal to
/// `axis`. The time-averaged Poynting vector of a Hertzian dipole is radial
/// with this pattern at every distance, so the shares hold on any box.
pub fn cube_face_share(p: [f64; 3], axis: usize) -> f64 {
    // a = integral of cos^2 over the face pair normal to the axis (solid
    // angle 4 pi / 3); b = integral of one transverse direction cosine^2
    let n = 400usize;
    let h = 2.0 / n as f64;
    let mut a = 0.0;
    for i in 0..n {
        for j in 0..n {
            let (u, v) = (-1.0 + (i as f64 + 0.5) * h, -1.0 + (j as f64 + 0.5) * h);
            a += (1.0 + u * u + v * v).powf(-2.5) * h * h;
        }
    }
    a *= 2.0;
    let pair = 4.0 * PI / 3.0;
    let b = (pair - a) / 2.0;
    let pa2 = p[axis] * p[axis] / (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    (pair - (pa2 * a + (1.0 - pa2) * b)) / (8.0 * PI / 3.0)
}

/// Mutual radiated power of two unit dipoles p1, p2 (unit vectors) a vector s
/// apart, relative to the power of one: (6 pi / k) p1 . Im G(s) . p2.
pub fn dipole_mutual(p1: [f64; 3], p2: [f64; 3], s: [f64; 3], k: f64) -> f64 {
    let d = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let pp = dot(p1, p2);
    if d == 0.0 {
        return pp;
    }
    let u = [s[0] / d, s[1] / d, s[2] / d];
    let q = dot(p1, u) * dot(p2, u);
    let x = k * d;
    1.5 * ((pp - q) * x.sin() / x + (pp - 3.0 * q) * (x.cos() / (x * x) - x.sin() / (x * x * x)))
}

/// FDTDX test_dipole_tilted_radiation_pattern and test_dipole_polarization_axes:
/// power through the six faces of a cube around (1) x-, y- and z-polarised
/// dipoles and (2) a dipole tilted 45 degrees in the xz plane (an Ex and an Ez
/// dipole with the same waveform and amplitude I l / sqrt2). Total power
/// against the Hertzian formula and against the z dipole; the face-pair shares
/// against the analytic shares of a sin^2(theta) pattern through a cube.
pub fn dipole_tilted(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 24.0;
    let half = 19.0 * dx;
    let b = 10.0 * dx;
    let amp = 1e-12;
    let sig = pulse();
    let freqs = band(3);
    let ax = ["x", "y", "z"];
    // outward power through each face pair, per frequency, and the run's dt/steps
    let domain = || Simulation::new([-half; 3], [half; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    let face_pairs = |dipoles: &[(Component, f64)]| -> Result<([Vec<f64>; 3], f64, usize, f64)> {
        let mut sim = domain();
        for (k, (c, a)) in dipoles.iter().enumerate() {
            sim.sources.push(Source::PointDipole { name: format!("d{k}"), position: [0.0; 3], polarization: *c, signal: sig.clone(), amplitude: *a });
        }
        for a in 0..3 {
            for (s, pos) in [("-", -b), ("+", b)] {
                sim.monitors.push(Monitor::Flux { name: format!("{}{s}", ax[a]), axis: a, position: pos, min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
            }
        }
        let res = run(&sim)?;
        let pairs = [0, 1, 2].map(|a| (0..freqs.len()).map(|i| res.flux[&format!("{}+", ax[a])][i] - res.flux[&format!("{}-", ax[a])][i]).collect::<Vec<f64>>());
        Ok((pairs, res.dt, res.stats.steps, res.stats.mcells_per_s))
    };
    let mut rep = CaseReport::new("dipole_tilted");
    let i0 = 1; // f0 for the shares (frequency independent)
    let mut p_axis = vec![];
    for (a, c) in [Component::Ex, Component::Ey, Component::Ez].into_iter().enumerate() {
        let (pairs, dt, steps, mc) = face_pairs(&[(c, amp)])?;
        rep.mcells_per_s = mc;
        let spc = sig.spectrum(&freqs, dt, steps, 0.5);
        let tot: Vec<f64> = (0..freqs.len()).map(|i| pairs[0][i] + pairs[1][i] + pairs[2][i]).collect();
        for (i, f) in freqs.iter().enumerate() {
            let k = TAU * f / C0;
            let p_exp = ETA0 * k * k * amp * amp * spc[i].norm_sqr() / (12.0 * PI);
            rep.check(&format!("{} dipole: P/P_analytic @ {:.3} f0", ax[a], f / f0()), tot[i] / p_exp, 1.0, 0.02);
        }
        let mut p = [0.0; 3];
        p[a] = 1.0;
        let mut argmin = 0;
        for f in 0..3 {
            if pairs[f][i0] < pairs[argmin][i0] {
                argmin = f;
            }
            rep.check(&format!("{} dipole: {} faces share", ax[a], ax[f]), pairs[f][i0] / tot[i0], cube_face_share(p, f), 0.01);
        }
        rep.check_abs(&format!("{} dipole: axis of least-flux face pair", ax[a]), argmin as f64, a as f64, 0.0);
        p_axis.push(tot);
    }
    for a in 0..2 {
        for (i, f) in freqs.iter().enumerate() {
            rep.check(&format!("P_{} / P_z @ {:.3} f0", ax[a], f / f0()), p_axis[a][i] / p_axis[2][i], 1.0, 1e-4);
        }
    }
    // tilted 45 degrees in the xz plane. On the Yee grid the Ex and Ez dipoles
    // sit half a cell apart (vector s), which adds the mutual radiation term of
    // two orthogonal dipoles: P / P_single = 1 + m(ks), m ~ -(ks)^2 / 20 here.
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let g = domain().build_grid()?;
    let at = |c: Component| [0, 1, 2].map(|a| g.position(c, a, g.nearest_index(c, a, 0.0)));
    let (px, pz) = (at(Component::Ex), at(Component::Ez));
    let sep = [px[0] - pz[0], px[1] - pz[1], px[2] - pz[2]];
    let (pairs, dt, steps, _) = face_pairs(&[(Component::Ex, amp * s2), (Component::Ez, amp * s2)])?;
    let spc = sig.spectrum(&freqs, dt, steps, 0.5);
    let tot: Vec<f64> = (0..freqs.len()).map(|i| pairs[0][i] + pairs[1][i] + pairs[2][i]).collect();
    for (i, f) in freqs.iter().enumerate() {
        let k = TAU * f / C0;
        let p_exp = ETA0 * k * k * amp * amp * spc[i].norm_sqr() / (12.0 * PI);
        let mutual = 1.0 + dipole_mutual([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], sep, k);
        rep.check(&format!("tilted: P/P_analytic @ {:.3} f0", f / f0()), tot[i] / p_exp, mutual, 0.02);
        rep.check(&format!("tilted: P / P_z @ {:.3} f0", f / f0()), tot[i] / p_axis[2][i], mutual, 2e-4);
        rep.check(&format!("tilted: x faces / z faces @ {:.3} f0", f / f0()), pairs[0][i] / pairs[2][i], 1.0, 1e-4);
    }
    let p = [s2, 0.0, s2];
    for f in 0..3 {
        rep.check(&format!("tilted: {} faces share", ax[f]), pairs[f][i0] / tot[i0], cube_face_share(p, f), 0.01);
    }
    Ok(rep)
}

/// FDTDX test_dipole_green_function (power scaling): a point current in a
/// dielectric of index n radiates n times its vacuum power.
pub fn dipole_in_dielectric(run: Runner) -> Result<CaseReport> {
    let n = 1.5f64;
    let dx = LAMBDA / (24.0 * n);
    let l = 1.2e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.media.push(Medium::from_index(n).with_name("bg"));
    sim.background = "bg".into();
    let sig = pulse();
    sim.sources.push(Source::PointDipole { name: "d".into(), position: [0.0; 3], polarization: Component::Ez, signal: sig.clone(), amplitude: 1e-12 });
    let freqs = band(3);
    let b = 0.33e-6;
    sim.monitors.push(Monitor::FluxBox { name: "P".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let spc = sig.spectrum(&freqs, res.dt, res.stats.steps, 0.5);
    let mut rep = CaseReport::new("dipole_in_dielectric");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let k = TAU * f / C0;
        let p_vac = ETA0 * k * k * (1e-12f64).powi(2) * spc[i].norm_sqr() / (12.0 * PI);
        rep.check(&format!("P / (n P_vac) @ {:.3} f0", f / f0()), res.flux["P"][i] / (n * p_vac), 1.0, 0.02);
    }
    Ok(rep)
}
