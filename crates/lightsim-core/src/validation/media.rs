//! Lossy, magnetic and dispersive media: conductors, magnetic loss, Lorentz
//! and Drude poles, hyperbolic media, sources inside a dispersive background.

use super::*;

/// Lossy conductor: field amplitude decays as exp(-z/delta) with the
/// analytic decay length of a conducting medium.
pub fn skin_depth(run: Runner) -> Result<CaseReport> {
    let sigma = 5e5; // S/m
    let f = f0();
    let w = TAU * f;
    // complex index of the conductor
    let eps = Complex64::new(1.0, sigma / (crate::constants::EPS0 * w));
    let nc = eps.sqrt();
    let delta = C0 / (w * nc.im);
    let dx = delta / 6.0;
    let n_cross = 2usize;
    let wd = n_cross as f64 * dx;
    let mut sim = column(dx, 3e-6, n_cross);
    sim.media.push(Medium::conductor(1.0, sigma).with_name("metal"));
    sim.structures.push(Structure {
        name: "m".into(),
        shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] },
        medium: "metal".into(),
    });
    sim.sources.push(Source::PlaneWave {
        name: "pw".into(),
        axis: 0,
        position: 1e-6,
        direction: Direction::Plus,
        polarization: Component::Ey,
        signal: TimeSignal::gaussian(f, 0.2 * f),
        amplitude: 1.0,
        angle: 0.0,
        min: None,
        max: None,
    });
    sim.monitors.push(Monitor::DftField {
        name: "e".into(),
        min: [2.0e-6, 0.0, 0.0],
        max: [2.0e-6 + 4.0 * delta, wd, wd],
        components: vec![Component::Ey],
        freqs: vec![f],
        apodization: Apodization::default(),
        stride: 1,
    });
    sim.run_time = 80e-15;
    let res = run(&sim)?;
    let ey = res.dft["e"].field(Component::Ey).unwrap();
    // fit log|E| vs x over the first ~3 skin depths
    let nx = ey.dims[1];
    let mut xs = vec![];
    let mut ys = vec![];
    let g = sim.build_grid()?;
    for i in 1..nx.saturating_sub(2) {
        let v = ey.at(0, i, 0, 0).norm();
        if v > 0.0 {
            xs.push(g.node(0, ey.origin[0] + i));
            ys.push(v.ln());
        }
    }
    let slope = linfit(&xs, &ys).0;
    let mut rep = CaseReport::new("skin_depth");
    rep.mcells_per_s = res.stats.mcells_per_s;
    rep.check("decay length / analytic", -1.0 / slope / delta, 1.0, 0.03);
    Ok(rep)
}

/// Transmission through a dispersive (Lorentz) slab versus the analytic
/// transfer-matrix result evaluated with the discrete-time susceptibility.
pub fn lorentz_slab(run: Runner) -> Result<CaseReport> {
    let pole = Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.3 * f0(), gamma: TAU * 0.1 * f0() };
    let eps_inf = 1.5;
    let dx = LAMBDA / 60.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    let thick = 0.5e-6;
    sim.media.push(Medium::dielectric(eps_inf).with_name("lor").with_poles(vec![pole.clone()]));
    sim.structures.push(Structure {
        name: "slab".into(),
        shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.0e-6 + thick, 1.0, 1.0] },
        medium: "lor".into(),
    });
    let freqs = band(5);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(flux_x("T", 3.2e-6, w, &freqs));
    sim.run_time = 150e-15;
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    let mut rep = CaseReport::new("lorentz_slab");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let wv = TAU * f;
        let eps = Complex64::new(eps_inf, 0.0) + pole.chi_discrete(wv, res.dt);
        let t = slab_transmission(eps.sqrt(), wv / C0, thick);
        let inc = 0.5 / ETA0 * sp[i] * w * w;
        rep.check_abs(&format!("T @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, t, 0.01);
    }
    Ok(rep)
}

/// Power transmission of a slab of complex index n and thickness d in vacuum.
pub fn slab_transmission(n: Complex64, k0: f64, d: f64) -> f64 {
    let one = Complex64::new(1.0, 0.0);
    let r12 = (one - n) / (one + n);
    let t12 = 2.0 * one / (one + n);
    let t21 = 2.0 * n / (one + n);
    let r21 = -r12;
    let ph = (Complex64::i() * n * k0 * d).exp();
    let t = t12 * t21 * ph / (one - r21 * r21 * ph * ph);
    t.norm_sqr()
}

/// Propagation inside a dispersive (Lorentz) background: phase velocity matches
/// Re(n) of the discrete-time permittivity.
pub fn dispersive_propagation(run: Runner) -> Result<CaseReport> {
    let pole = Pole::Lorentz { delta_eps: 0.8, omega0: TAU * 2.0 * f0(), gamma: TAU * 0.02 * f0() };
    let dx = LAMBDA / 50.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    sim.background = "lor".into();
    sim.media.push(Medium::dielectric(1.2).with_name("lor").with_poles(vec![pole.clone()]));
    sim.sources.push(plane_source(0.8e-6, 1.0));
    let freqs = band(3);
    for (nm, x) in [("a", 1.5e-6), ("b", 2.5e-6)] {
        sim.monitors.push(Monitor::DftField { name: nm.into(), min: [x, 0.0, 0.0], max: [x, w, w], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    }
    sim.run_time = 100e-15;
    let res = run(&sim)?;
    let g = sim.build_grid()?;
    let xa = g.node(0, res.dft["a"].fields[0].origin[0]);
    let xb = g.node(0, res.dft["b"].fields[0].origin[0]);
    let mut rep = CaseReport::new("dispersive_propagation");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let wv = TAU * f;
        let k0 = wv / C0;
        let n_exp = (Complex64::new(1.2, 0.0) + pole.chi_discrete(wv, res.dt)).sqrt().re;
        let mut dphi = (res.dft["b"].fields[0].at(i, 0, 0, 0) / res.dft["a"].fields[0].at(i, 0, 0, 0)).arg();
        let expected = k0 * n_exp * (xb - xa);
        while dphi < expected - PI {
            dphi += TAU;
        }
        while dphi > expected + PI {
            dphi -= TAU;
        }
        rep.check(&format!("n_eff @ {:.3} f0", f / f0()), dphi / (k0 * (xb - xa)), n_exp, 0.003);
    }
    Ok(rep)
}

/// Anisotropic dispersion: a Drude pole on the y axis only; Ey sees the
/// dispersive index, Ez the plain eps_inf.
pub fn anisotropic_dispersion(run: Runner) -> Result<CaseReport> {
    let pole = Pole::Lorentz { delta_eps: 1.0, omega0: TAU * 2.5 * f0(), gamma: TAU * 0.01 * f0() };
    let mut rep = CaseReport::new("anisotropic_dispersion");
    for pol in [Component::Ey, Component::Ez] {
        let dx = LAMBDA / 50.0;
        let n_cross = 2usize;
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 4e-6, n_cross);
        sim.background = "ani".into();
        let mut m = Medium::dielectric(1.5).with_name("ani").with_poles(vec![pole.clone()]);
        m.pole_axes = [false, true, false];
        sim.media.push(m);
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 0.8e-6, direction: Direction::Plus, polarization: pol, signal: pulse(), amplitude: 1.0, angle: 0.0, min: None, max: None });
        for (nm, x) in [("a", 1.5e-6), ("b", 2.5e-6)] {
            sim.monitors.push(Monitor::DftField { name: nm.into(), min: [x, 0.0, 0.0], max: [x, w, w], components: vec![pol], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
        }
        sim.run_time = 90e-15;
        let res = run(&sim)?;
        let g = sim.build_grid()?;
        let xa = g.node(0, res.dft["a"].fields[0].origin[0]);
        let xb = g.node(0, res.dft["b"].fields[0].origin[0]);
        let k0 = TAU * f0() / C0;
        let eps = if pol == Component::Ey { Complex64::new(1.5, 0.0) + pole.chi_discrete(TAU * f0(), res.dt) } else { Complex64::new(1.5, 0.0) };
        let n_exp = eps.sqrt().re;
        let mut dphi = (res.dft["b"].fields[0].at(0, 0, 0, 0) / res.dft["a"].fields[0].at(0, 0, 0, 0)).arg();
        let expected = k0 * n_exp * (xb - xa);
        while dphi < expected - PI {
            dphi += TAU;
        }
        while dphi > expected + PI {
            dphi -= TAU;
        }
        rep.check(&format!("n_eff ({})", pol.name()), dphi / (k0 * (xb - xa)), n_exp, 0.003);
        rep.mcells_per_s = res.stats.mcells_per_s;
    }
    Ok(rep)
}

/// FDTDX test_dispersion (Drude metal): a Drude half-space with f_p >> f
/// reflects as predicted by Fresnel with the discrete Drude permittivity.
pub fn drude_reflection(run: Runner) -> Result<CaseReport> {
    let (fp, fc) = (4.0 * f0(), 0.05 * f0());
    let dx = LAMBDA / 100.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 3e-6, n_cross);
    sim.media.push(crate::materials::library::drude(fp, fc).with_name("metal"));
    sim.structures.push(Structure { name: "m".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "metal".into() });
    let freqs = band(3);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
    sim.run_time = 70e-15;
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    let mut rep = CaseReport::new("drude_reflection");
    rep.mcells_per_s = res.stats.mcells_per_s;
    let pole = Pole::Drude { omega_p: TAU * fp, gamma: TAU * fc };
    for (i, f) in freqs.iter().enumerate() {
        let wv = TAU * f;
        let eps = Complex64::new(1.0, 0.0) + pole.chi_discrete(wv, res.dt);
        let nn = eps.sqrt();
        let r_exp = ((Complex64::new(1.0, 0.0) - nn) / (Complex64::new(1.0, 0.0) + nn)).norm_sqr();
        let inc = 0.5 / ETA0 * sp[i] * w * w;
        rep.check(&format!("R / R_Fresnel @ {:.3} f0", f / f0()), -res.flux["R"][i] / inc, r_exp, 0.01);
    }
    Ok(rep)
}

/// Power reflection and transmission of a slab of complex index n and
/// thickness d in vacuum at normal incidence (Airy sum of the two Fresnel
/// interfaces).
pub fn slab_rt(n: Complex64, k0: f64, d: f64) -> (f64, f64) {
    let one = Complex64::new(1.0, 0.0);
    let r12 = (one - n) / (one + n);
    let t12 = 2.0 * one / (one + n);
    let t21 = 2.0 * n / (one + n);
    let r21 = -r12;
    let ph = (Complex64::i() * n * k0 * d).exp();
    let den = one - r21 * r21 * ph * ph;
    let r = r12 + t12 * t21 * r21 * ph * ph / den;
    let t = t12 * t21 * ph / den;
    (r.norm_sqr(), t.norm_sqr())
}

/// FDTDX TestHyperbolicDrude::test_hyperbolic_medium_is_polarization_selective:
/// a Drude pole on the x axis only (f_p = 4 f0, eps_inf = 2.25), so that
/// Re eps_x < 0 < eps_y = eps_z at f0 (an indefinite, hyperbolic tensor). A
/// 60 nm slab, plane waves along z: x polarization reflects like a metal, R and
/// T against the slab Fresnel (Airy) result with the discrete eps_x(w); y
/// polarization is transmitted as by a plain eps_inf slab.
pub fn hyperbolic_drude(run: Runner) -> Result<CaseReport> {
    let (fp, fc) = (4.0 * f0(), 0.05 * f0());
    let eps_inf = 2.25;
    let pole = Pole::Drude { omega_p: TAU * fp, gamma: TAU * fc };
    let dx = LAMBDA / 200.0;
    let w = 2.0 * dx;
    let thick = 12.0 * dx;
    let z0 = 1.5e-6;
    let freqs = band(3);
    let mut rep = CaseReport::new("hyperbolic_drude");
    let mut t_meas = [0.0; 2];
    for (k, pol) in [Component::Ex, Component::Ey].into_iter().enumerate() {
        let mut bd = Boundaries::all(Boundary::Periodic);
        bd.z = [Boundary::pml(), Boundary::pml()];
        let mut sim = Simulation::new([0.0; 3], [w, w, 3.0e-6], MeshSpec::Uniform { dx: [dx; 3] }, bd, 70e-15);
        let mut m = Medium::dielectric(eps_inf).with_name("hyp").with_poles(vec![pole.clone()]);
        m.pole_axes = [true, false, false];
        sim.media.push(m);
        sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [-1.0, -1.0, z0], max: [1.0, 1.0, z0 + thick] }, medium: "hyp".into() });
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 0.8e-6, direction: Direction::Plus, polarization: pol, signal: pulse(), amplitude: 1.0, angle: 0.0, min: None, max: None });
        for (nm, z) in [("R", 0.4e-6), ("T", 2.2e-6)] {
            sim.monitors.push(Monitor::Flux { name: nm.into(), axis: 2, position: z, min: [-1.0, -1.0, z], max: [w + 1.0, w + 1.0, z], freqs: freqs.clone(), apodization: Apodization::default() });
        }
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
        for (i, f) in freqs.iter().enumerate() {
            let wv = TAU * f;
            let eps = if pol == Component::Ex { Complex64::new(eps_inf, 0.0) + pole.chi_discrete(wv, res.dt) } else { Complex64::new(eps_inf, 0.0) };
            let (r_exp, t_exp) = slab_rt(eps.sqrt(), wv / C0, thick);
            let inc = 0.5 / ETA0 * sp[i] * w * w;
            let (r, t) = (-res.flux["R"][i] / inc, res.flux["T"][i] / inc);
            if i == 1 {
                t_meas[k] = t;
                if pol == Component::Ex {
                    rep.check_abs("premise: Re eps_x(f0) < 0 (1 = holds)", (eps.re < 0.0) as u8 as f64, 1.0, 0.0);
                    rep.check_abs("premise: eps_y(f0) = eps_inf > 0 (1 = holds)", (eps_inf > 0.0) as u8 as f64, 1.0, 0.0);
                }
            }
            rep.check(&format!("{}: R / R_Fresnel @ {:.3} f0", pol.name(), f / f0()), r, r_exp, 0.01);
            rep.check(&format!("{}: T / T_Fresnel @ {:.3} f0", pol.name(), f / f0()), t, t_exp, 0.01);
        }
    }
    rep.check_abs("selectivity T_y - T_x > 0.8 (1 = holds)", (t_meas[1] - t_meas[0] > 0.8) as u8 as f64, 1.0, 0.0);
    Ok(rep)
}

/// FDTDX test_tfsf_region (dispersive background), test_dispersive_source and
/// test_dispersion (source inside a Lorentz medium): in a strongly dispersive
/// Lorentz background (1) a TFSF box keeps the scattered-field region dark
/// across the pulse band; (2) a plane-wave source inside the medium launches
/// only a forward wave whose impedance is |E| / (eta0 |H|) = 1 / |n(w)|.
pub fn dispersive_background_sources(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("dispersive_background_sources");
    let w0 = TAU * f0();
    let pole = Pole::Lorentz { delta_eps: 1.0, omega0: 1.5 * w0, gamma: 2e13 };
    let lor = Medium::dielectric(2.25).with_name("lor").with_poles(vec![pole.clone()]);
    let freqs = band(3);
    // (1) TFSF box in the Lorentz background
    let dx = LAMBDA / (25.0 * 2.0);
    let l = 1.0e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 60e-15);
    sim.media.push(lor.clone());
    sim.background = "lor".into();
    let a = 0.25e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    sim.monitors.push(Monitor::DftField { name: "in".into(), min: [0.0; 3], max: [0.0; 3], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::DftField { name: "out".into(), min: [0.38e-6, 0.0, 0.0], max: [0.38e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::DftField { name: "back".into(), min: [-0.38e-6, 0.0, 0.0], max: [-0.38e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    let r = run(&sim)?;
    for (i, f) in freqs.iter().enumerate() {
        let inside = r.dft["in"].fields[0].at(i, 0, 0, 0).norm();
        let out = r.dft["out"].fields[0].at(i, 0, 0, 0).norm().max(r.dft["back"].fields[0].at(i, 0, 0, 0).norm());
        rep.check_abs(&format!("TFSF in Lorentz background: |E_scat| / |E_inc| @ {:.3} f0", f / f0()), out / inside, 0.0, 1e-3);
    }
    rep.mcells_per_s = r.stats.mcells_per_s;
    // (2) plane-wave source inside the Lorentz medium
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    sim.media.push(lor);
    sim.background = "lor".into();
    sim.sources.push(plane_source(1.5e-6, 1.0));
    sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
    sim.monitors.push(flux_x("R", 0.6e-6, w, &freqs));
    sim.monitors.push(dft_point("z", 2.5e-6, vec![Component::Ey, Component::Hz], &freqs, 1));
    sim.run_time = 120e-15;
    let r = run(&sim)?;
    for (i, f) in freqs.iter().enumerate() {
        rep.check_abs(&format!("source in Lorentz medium: |P_back| / P_fwd @ {:.3} f0", f / f0()), (r.flux["R"][i] / r.flux["T"][i]).abs(), 0.0, 1e-3);
        let n = (Complex64::new(2.25, 0.0) + pole.chi_discrete(TAU * f, r.dt)).sqrt();
        let e = r.dft["z"].field(Component::Ey).unwrap().at(i, 0, 0, 0);
        let h = r.dft["z"].field(Component::Hz).unwrap().at(i, 0, 0, 0);
        rep.check(&format!("source in Lorentz medium: |E|/(eta0|H|) vs 1/|n(w)| @ {:.3} f0", f / f0()), e.norm() / h.norm(), 1.0 / n.norm(), 0.01);
    }
    Ok(rep)
}

/// Magnetic conductivity. (1) A half-space with eps = mu = 1 and matched
/// electric and magnetic loss, sigma_m / (mu0 mu) = sigma / (eps0 eps)
/// (field decay length 0.25 um): its impedance is that of vacuum at every
/// frequency, so it does not reflect (measured: 1e-3, against 2e-2 for the
/// same sigma without sigma_m), the field decays as exp(-sigma eta0 x)
/// and |E| / (eta0 |H|) = 1 inside. (2) A half-space with magnetic loss only
/// (eps = mu = 1, sigma_m = omega0 mu0 / 2): reflectance
/// |(Z - 1) / (Z + 1)|^2 with Z = sqrt(1 + i sigma_m / (omega mu0)) over the
/// band, decay rate k0 Im Z and impedance |Z| at f0.
pub fn magnetic_loss(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("magnetic_loss");
    let w0 = TAU * f0();
    let k0 = w0 / C0;
    for (matched, label) in [(true, "matched"), (false, "magnetic only")] {
        let (er, sigma, sigma_m, x1) = if matched {
            let sigma = 1.0 / (ETA0 * 0.25e-6);
            (1.0, sigma, sigma * crate::constants::MU0 / crate::constants::EPS0, 3.0e-6)
        } else {
            (1.0, 0.0, 0.5 * w0 * crate::constants::MU0, 4.4e-6)
        };
        let dx = LAMBDA / (30.0 * er);
        let n_cross = 2usize;
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 6e-6, n_cross);
        let mut m = Medium::dielectric(er).with_name("lossy");
        m.mu = [er; 3];
        m.sigma = [sigma; 3];
        m.sigma_m = [sigma_m; 3];
        sim.media.push(m);
        sim.structures.push(Structure { name: "half".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "lossy".into() });
        let freqs = band(5);
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim.monitors.push(Monitor::DftField {
            name: "line".into(),
            min: [2.2e-6, 0.0, 0.0],
            max: [x1, 0.0, 0.0],
            components: vec![Component::Ey, Component::Hz],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
        sim.run_time = 90e-15;
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
        for (i, f) in freqs.iter().enumerate() {
            let inc = 0.5 / ETA0 * sp[i] * w * w;
            let r = -res.flux["R"][i] / inc;
            let wf = TAU * f;
            if matched {
                // the reflectance the same half-space has without sigma_m
                let z_e = (Complex64::new(er, 0.0) / Complex64::new(er, sigma / (wf * crate::constants::EPS0))).sqrt();
                let r_e = ((z_e - 1.0) / (z_e + 1.0)).norm_sqr();
                rep.check(&format!("{label}: R / R(sigma only, {r_e:.1e}) @ {:.3} f0", f / f0()), r.abs() / r_e, 0.0, 0.1);
            } else {
                let z = Complex64::new(1.0, sigma_m / (wf * crate::constants::MU0)).sqrt();
                rep.check(&format!("{label}: R @ {:.3} f0", f / f0()), r, ((z - 1.0) / (z + 1.0)).norm_sqr(), 0.03);
            }
        }
        let ey = res.dft["line"].field(Component::Ey).unwrap();
        let hz = res.dft["line"].field(Component::Hz).unwrap();
        let g = sim.build_grid()?;
        let (mut xs, mut ys) = (vec![], vec![]);
        for i in 0..ey.dims[1] {
            xs.push(g.node(0, ey.origin[0] + i));
            ys.push(ey.at(0, i, 0, 0).norm().ln());
        }
        let alpha = -linfit(&xs, &ys).0;
        let z = Complex64::new(1.0, sigma_m / (w0 * crate::constants::MU0)).sqrt();
        let (alpha_exp, z_exp) = if matched { (sigma * ETA0, 1.0) } else { (k0 * z.im, z.norm()) };
        rep.check(&format!("{label}: decay rate / analytic"), alpha / alpha_exp, 1.0, 0.02);
        // |E| at the two nodes around an H_z sample (half a cell apart)
        let j = hz.dims[1] / 3;
        let gx = hz.origin[0] + j;
        let ei = |gi: usize| ey.at(0, gi - ey.origin[0], 0, 0).norm();
        let e = (ei(gx) * ei(gx + 1)).sqrt();
        rep.check(&format!("{label}: |E| / (eta0 |H|) / analytic"), e / hz.at(0, j, 0, 0).norm() / z_exp, 1.0, 0.01);
    }
    Ok(rep)
}
