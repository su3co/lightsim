//! Plane waves in uniform media and at interfaces: vacuum, Fresnel, walls,
//! impedance, anisotropic cells, CW against pulsed, DFT recording.

use super::*;

/// Plane wave in vacuum: transmitted flux equals the analytic incident flux
/// 0.5 n/eta0 |S(w)|^2 A, and the scattered-field side stays dark.
pub fn plane_wave_vacuum(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    let freqs = band(5);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
    sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    let mut rep = CaseReport::new("plane_wave_vacuum");
    rep.mcells_per_s = res.stats.mcells_per_s;
    let area = w * w;
    for (i, f) in freqs.iter().enumerate() {
        let inc = 0.5 / ETA0 * sp[i] * area;
        rep.check(&format!("T/inc @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0, 0.01);
        rep.check(&format!("|R|/inc @ {:.3} f0", f / f0()), (res.flux["R"][i] / inc).abs(), 0.0, 2e-3);
    }
    Ok(rep)
}

/// Normal-incidence Fresnel reflection/transmission at a dielectric interface.
pub fn fresnel_normal(run: Runner) -> Result<CaseReport> {
    let n2: f64 = 2.0;
    let dx = LAMBDA / (30.0 * n2);
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    sim.media.push(Medium::from_index(n2).with_name("glass"));
    sim.structures.push(Structure {
        name: "half".into(),
        shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] },
        medium: "glass".into(),
    });
    let freqs = band(5);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
    sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
    sim.run_time = 80e-15;
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    let mut rep = CaseReport::new("fresnel_normal_n2");
    rep.mcells_per_s = res.stats.mcells_per_s;
    let r_exp = ((1.0 - n2) / (1.0 + n2)).powi(2);
    for (i, f) in freqs.iter().enumerate() {
        let inc = 0.5 / ETA0 * sp[i] * w * w;
        rep.check(&format!("R @ {:.3} f0", f / f0()), -res.flux["R"][i] / inc, r_exp, 0.02);
        rep.check(&format!("T @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0 - r_exp, 0.01);
    }
    Ok(rep)
}

/// Perfect reflectors: PEC and PMC walls reflect all power at normal incidence.
pub fn pec_pmc_reflection(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("pec_pmc_reflection");
    for (label, wall) in [("PEC", Boundary::Pec), ("PMC", Boundary::Pmc)] {
        let dx = LAMBDA / 30.0;
        let n_cross = 2usize;
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 3e-6, n_cross);
        sim.boundaries.x[1] = wall.clone();
        let freqs = band(3);
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim.run_time = 70e-15;
        let res = run(&sim)?;
        let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
        for (i, f) in freqs.iter().enumerate() {
            let inc = 0.5 / ETA0 * sp[i] * w * w;
            rep.check(&format!("{label} R @ {:.3} f0", f / f0()), -res.flux["R"][i] / inc, 1.0, 0.01);
        }
        rep.mcells_per_s = res.stats.mcells_per_s;
    }
    Ok(rep)
}

/// Reciprocity: transmission through an asymmetric multilayer is the same in
/// both directions.
pub fn reciprocity(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 60.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mk = |dir: Direction| {
        let mut sim = column(dx, 5e-6, n_cross);
        sim.media.push(Medium::from_index(2.2).with_name("h"));
        sim.media.push(Medium::from_index(1.5).with_name("l"));
        sim.structures.push(Structure { name: "a".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.3e-6, 1.0, 1.0] }, medium: "h".into() });
        sim.structures.push(Structure { name: "b".into(), shape: Shape::Box { min: [2.3e-6, -1.0, -1.0], max: [2.9e-6, 1.0, 1.0] }, medium: "l".into() });
        let (src, mon) = if dir == Direction::Plus { (1.0e-6, 4.0e-6) } else { (4.0e-6, 1.0e-6) };
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: src, direction: dir, polarization: Component::Ey, signal: pulse(), amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.monitors.push(flux_x("T", mon, w, &band(3)));
        sim.run_time = 90e-15;
        sim
    };
    let fwd = run(&mk(Direction::Plus))?;
    let bwd = run(&mk(Direction::Minus))?;
    let mut rep = CaseReport::new("reciprocity");
    rep.mcells_per_s = fwd.stats.mcells_per_s;
    for (i, f) in band(3).iter().enumerate() {
        rep.check(&format!("T_fwd/T_bwd @ {:.3} f0", f / f0()), fwd.flux["T"][i] / -bwd.flux["T"][i], 1.0, 2e-3);
    }
    Ok(rep)
}

/// Pulsed versus continuous-wave excitation: the steady-state CW amplitude
/// transmitted through a dielectric slab equals |t(w0)| from the pulsed,
/// spectrally normalised flux measurement and from the analytic slab formula.
pub fn pulsed_vs_cw(run: Runner) -> Result<CaseReport> {
    let n2 = 2.0;
    let thick = 0.4e-6;
    let dx = LAMBDA / (40.0 * n2);
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mk = |sig: TimeSignal, t: f64| {
        let mut sim = column(dx, 4e-6, n_cross);
        sim.media.push(Medium::from_index(n2).with_name("slab"));
        sim.structures.push(Structure { name: "s".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.0e-6 + thick, 1.0, 1.0] }, medium: "slab".into() });
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: sig, amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.monitors.push(flux_x("T", 3.0e-6, w, &[f0()]));
        sim.monitors.push(Monitor::Probe { name: "p".into(), position: [3.0e-6, 0.0, 0.0], components: vec![Component::Ey] });
        sim.run_time = t;
        sim
    };
    let pulsed = run(&mk(pulse(), 80e-15))?;
    let sp = spectrum_power(&pulse(), &[f0()], pulsed.dt, pulsed.stats.steps);
    let t_pulsed = (pulsed.flux["T"][0] / (0.5 / ETA0 * sp[0] * w * w)).sqrt();
    let cw = run(&mk(TimeSignal::cw(f0(), 0.05 * f0()), 400e-15))?;
    let data = &cw.time["p"].fields[0].data;
    let period_steps = (1.0 / f0() / cw.dt).ceil() as usize;
    let tail = &data[data.len() - 3 * period_steps..];
    let amp = tail.iter().fold(0 as Real, |m, v| m.max(v.abs())) as f64;
    let t_exact = slab_transmission(Complex64::new(n2, 0.0), TAU * f0() / C0, thick).sqrt();
    let mut rep = CaseReport::new("pulsed_vs_cw");
    rep.mcells_per_s = cw.stats.mcells_per_s;
    rep.check("|t| CW steady state vs analytic", amp, t_exact, 0.01);
    rep.check("|t| pulsed vs analytic", t_pulsed, t_exact, 0.01);
    rep.check("|t| CW vs pulsed", amp, t_pulsed, 0.01);
    Ok(rep)
}

/// Time-window apodization suppresses spectral leakage of a truncated CW
/// signal in the running DFT.
pub fn apodization_leakage(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let t_end = 150e-15;
    // ~4.5 frequency bins (1/T) away from f0, between sinc nulls
    let f_off = f0() * 1.137;
    let mk = |apod: Apodization| {
        let mut sim = column(dx, 3e-6, n_cross);
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::cw(f0(), 0.1 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.monitors.push(Monitor::DftField { name: "d".into(), min: [2.0e-6, 0.0, 0.0], max: [2.0e-6, w, w], components: vec![Component::Ey], freqs: vec![f0(), f_off], apodization: apod, stride: 1 });
        sim.run_time = t_end;
        sim
    };
    let raw = run(&mk(Apodization { start: Some(40e-15), end: None, width: None }))?;
    let apo = run(&mk(Apodization { start: Some(40e-15), end: Some(t_end), width: Some(40e-15) }))?;
    let ratio = |r: &Results| -> f64 {
        let f = &r.dft["d"].fields[0];
        f.at(1, 0, 0, 0).norm() / f.at(0, 0, 0, 0).norm()
    };
    let (lr, la) = (ratio(&raw), ratio(&apo));
    let mut rep = CaseReport::new("apodization_leakage");
    rep.mcells_per_s = raw.stats.mcells_per_s;
    rep.check_abs("leakage(apodized)/leakage(raw) < 0.2", la / lr, 0.0, 0.2);
    Ok(rep)
}

/// FDTDX test_plane_wave: phase velocity and wave impedance |E|/|H| of a
/// plane wave in vacuum and in an n = 2 dielectric.
pub fn plane_wave_impedance(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("plane_wave_impedance");
    for n in [1.0f64, 2.0] {
        let dx = LAMBDA / (30.0 * n);
        let mut sim = column(dx, 4e-6, 2);
        sim.media.push(Medium::from_index(n).with_name("bg"));
        sim.background = "bg".into();
        sim.sources.push(plane_source(1.0e-6, n));
        let f = [f0()];
        sim.monitors.push(dft_point("a", 2.0e-6, vec![Component::Ey, Component::Hz], &f, 1));
        sim.monitors.push(dft_point("b", 2.6e-6, vec![Component::Ey, Component::Hz], &f, 1));
        sim.run_time = 70e-15;
        let res = run(&sim)?;
        let g = sim.build_grid()?;
        let ey = |nm: &str| res.dft[nm].field(Component::Ey).unwrap().at(0, 0, 0, 0);
        let hz = |nm: &str| res.dft[nm].field(Component::Hz).unwrap().at(0, 0, 0, 0);
        let xa = g.node(0, res.dft["a"].field(Component::Ey).unwrap().origin[0]);
        let xb = g.node(0, res.dft["b"].field(Component::Ey).unwrap().origin[0]);
        let k0 = TAU * f0() / C0;
        rep.check(&format!("n={n}: k / (n k0)"), wavenumber(ey("b") / ey("a"), xb - xa, n * k0) / (n * k0), 1.0, 5e-3);
        rep.check(&format!("n={n}: |E| / (eta0 |H|) vs 1/n"), ey("a").norm() / hz("a").norm(), 1.0 / n, 0.01);
        rep.mcells_per_s = res.stats.mcells_per_s;
    }
    Ok(rep)
}

/// FDTDX test_tfsf_grid_normalization: injected plane-wave power and TFSF
/// amplitude do not depend on the cell aspect ratio (dz = 2 dx, dy = 1.5 dx).
pub fn anisotropic_cells(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("anisotropic_cells");
    let dx = LAMBDA / 30.0;
    let d = [dx, 1.5 * dx, 2.0 * dx];
    let (wy, wz) = (2.0 * d[1], 2.0 * d[2]);
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [4e-6, wy, wz], MeshSpec::Uniform { dx: d }, b, 60e-15);
    let freqs = band(3);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(Monitor::Flux { name: "T".into(), axis: 0, position: 3.0e-6, min: [3.0e-6, -1.0, -1.0], max: [3.0e-6, 1.0, 1.0], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    for (i, f) in freqs.iter().enumerate() {
        let inc = 0.5 / ETA0 * sp[i] * wy * wz;
        rep.check(&format!("plane wave T/inc @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0, 0.01);
    }
    let l = 1.2e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [LAMBDA / 25.0, LAMBDA / 20.0, LAMBDA / 15.0] }, Boundaries::all_pml(), 40e-15);
    let a = 0.3e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    sim.monitors.push(Monitor::DftField { name: "in".into(), min: [0.0; 3], max: [0.0; 3], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::DftField { name: "out".into(), min: [0.45e-6, 0.0, 0.0], max: [0.45e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    let r = run(&sim)?;
    let spc = pulse().spectrum(&freqs, r.dt, r.stats.steps, 0.0);
    for (i, f) in freqs.iter().enumerate() {
        let inside = r.dft["in"].fields[0].at(i, 0, 0, 0).norm();
        let outside = r.dft["out"].fields[0].at(i, 0, 0, 0).norm();
        rep.check(&format!("TFSF |E_in|/|S| @ {:.3} f0", f / f0()), inside / spc[i].norm(), 1.0, 0.02);
        rep.check_abs(&format!("TFSF |E_out|/|E_in| @ {:.3} f0", f / f0()), outside / inside, 0.0, 5e-3);
    }
    rep.mcells_per_s = res.stats.mcells_per_s;
    Ok(rep)
}

/// FDTDX test_detector_region / test_phasor_poynting_flux: (1) a subsampled
/// running DFT (stride 4) matches every-step recording; (2) the running DFT
/// matches an offline DFT of the probe time series; (3) the phasor flux of a
/// plane wave equals 0.5 |E||H| / eta0 times the area.
pub fn phasor_recording(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let mut sim = column(dx, 4e-6, n_cross);
    let freqs = band(3);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(dft_point("s1", 2.5e-6, vec![Component::Ey, Component::Hz], &freqs, 1));
    sim.monitors.push(dft_point("s4", 2.5e-6, vec![Component::Ey], &freqs, 4));
    sim.monitors.push(Monitor::Probe { name: "p".into(), position: [2.5e-6, 0.0, 0.0], components: vec![Component::Ey] });
    sim.monitors.push(flux_x("T", 2.5e-6, w, &freqs));
    let res = run(&sim)?;
    let mut rep = CaseReport::new("phasor_recording");
    rep.mcells_per_s = res.stats.mcells_per_s;
    let tr = &res.time["p"];
    let data = &tr.fields[0].data;
    for (i, f) in freqs.iter().enumerate() {
        let e1 = res.dft["s1"].field(Component::Ey).unwrap().at(i, 0, 0, 0);
        let e4 = res.dft["s4"].field(Component::Ey).unwrap().at(i, 0, 0, 0);
        let h1 = res.dft["s1"].field(Component::Hz).unwrap().at(i, 0, 0, 0);
        rep.check(&format!("|DFT stride 4 - stride 1| / |DFT| @ {:.3} f0", f / f0()), (e4 - e1).norm() / e1.norm(), 0.0, 2e-3);
        let wv = TAU * f;
        let off: Complex64 = tr.steps.iter().zip(data.iter()).fold(Complex64::new(0.0, 0.0), |acc, (&n, &v)| {
            acc + Complex64::from_polar(v as f64 * res.dt, wv * (n as f64 + 1.0) * res.dt)
        });
        rep.check(&format!("|DFT - offline DFT of probe| / |DFT| @ {:.3} f0", f / f0()), (off - e1).norm() / e1.norm(), 0.0, 1e-3);
        rep.check(&format!("phasor flux / (0.5 |E||H| A / eta0) @ {:.3} f0", f / f0()), res.flux["T"][i] / (0.5 * e1.norm() * h1.norm() * w * w / ETA0), 1.0, 0.01);
    }
    Ok(rep)
}
