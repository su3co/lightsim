//! Absorbing, symmetric, periodic and Bloch boundaries.

use super::*;

/// CPML reflection: probe field near the PML versus a reference run in a much
/// larger domain (difference over the time window before the reference
/// boundary echo arrives).
pub fn pml_reflection(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let mk = |half: f64| {
        let mut sim =
            Simulation::new([-half; 3], [half; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 30e-15);
        sim.sources.push(Source::PointDipole {
            name: "d".into(),
            position: [0.0; 3],
            polarization: Component::Ez,
            signal: TimeSignal::gaussian(f0(), 0.5 * f0()),
            amplitude: 1e-12,
        });
        sim.monitors.push(Monitor::Probe { name: "p".into(), position: [0.4e-6, 0.0, 0.0], components: vec![Component::Ez] });
        sim
    };
    let small = run(&mk(0.5e-6))?;
    let big = run(&mk(2.5e-6))?;
    let a = &small.time["p"].fields[0].data;
    let b = &big.time["p"].fields[0].data;
    let n = a.len().min(b.len());
    let peak = b[..n].iter().fold(0 as Real, |m, v| m.max(v.abs())) as f64;
    let err = a[..n].iter().zip(&b[..n]).fold(0 as Real, |m, (x, y)| m.max((x - y).abs())) as f64;
    let mut rep = CaseReport::new("pml_reflection");
    rep.mcells_per_s = small.stats.mcells_per_s;
    rep.check("max |small - big| / peak", err / peak, 0.0, 1e-2);
    Ok(rep)
}

/// Adiabatic absorber (`Boundary::Absorber`): probe field in a small domain
/// closed by absorbers on all faces versus a reference run in a much larger
/// domain with PML, in vacuum and with a dispersive half space (a lossless
/// pole above the band, as in a fitted dielectric) running through the
/// absorbers. The absorber lives in the per-cell update coefficients of both
/// fields, so this also checks those paths of every engine.
pub fn absorber_reflection(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let mk = |half: f64, absorber: bool, slab: bool| {
        let b = if absorber { Boundaries::all(Boundary::Absorber { layers: 40, params: Default::default() }) } else { Boundaries::all_pml() };
        let mut sim = Simulation::new([-half; 3], [half; 3], MeshSpec::Uniform { dx: [dx; 3] }, b, 30e-15);
        if slab {
            let w0 = 5.0 * TAU * f0();
            sim.media.push(Medium::dielectric(1.5).with_poles(vec![Pole::Lorentz { delta_eps: 1.5, omega0: w0, gamma: 0.0 }]).with_name("glass"));
            sim.structures.push(Structure { name: "half".into(), shape: Shape::Box { min: [-1.0, -1.0, -1.0], max: [1.0, 1.0, -0.15e-6] }, medium: "glass".into() });
        }
        sim.sources.push(Source::PointDipole {
            name: "d".into(),
            position: [0.0; 3],
            polarization: Component::Ez,
            signal: TimeSignal::gaussian(f0(), 0.5 * f0()),
            amplitude: 1e-12,
        });
        sim.monitors.push(Monitor::Probe { name: "p".into(), position: [0.4e-6, 0.0, 0.0], components: vec![Component::Ez] });
        sim
    };
    let mut rep = CaseReport::new("absorber_reflection");
    for (label, slab) in [("vacuum", false), ("dispersive half space", true)] {
        let small = run(&mk(0.5e-6, true, slab))?;
        let big = run(&mk(2.5e-6, false, slab))?;
        let a = &small.time["p"].fields[0].data;
        let b = &big.time["p"].fields[0].data;
        let n = a.len().min(b.len());
        let peak = b[..n].iter().fold(0 as Real, |m, v| m.max(v.abs())) as f64;
        let err = a[..n].iter().zip(&b[..n]).fold(0 as Real, |m, (x, y)| m.max((x - y).abs())) as f64;
        rep.mcells_per_s = small.stats.mcells_per_s;
        rep.check(&format!("{label}: max |absorber - reference| / peak"), err / peak, 0.0, 1e-2);
    }
    Ok(rep)
}

/// Quarter-domain symmetry: a z-dipole on the x=0 and y=0 planes simulated in
/// the full domain and in one quadrant with PMC symmetry planes gives the
/// same fields.
pub fn symmetry_quarter_domain(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let l = 0.8e-6;
    let mk = |quarter: bool| {
        let lo = if quarter { [0.0, 0.0, -l] } else { [-l, -l, -l] };
        let mut b = Boundaries::all_pml();
        if quarter {
            b.x[0] = Boundary::Pmc;
            b.y[0] = Boundary::Pmc;
        }
        let mut sim = Simulation::new(lo, [l, l, l], MeshSpec::Uniform { dx: [dx; 3] }, b, 30e-15);
        sim.sources.push(Source::PointDipole { name: "d".into(), position: [0.0, 0.0, 0.0], polarization: Component::Ez, signal: pulse(), amplitude: 1e-12 });
        sim.monitors.push(Monitor::DftField {
            name: "f".into(),
            min: [0.0, 0.0, -0.3e-6],
            max: [0.5e-6, 0.5e-6, 0.3e-6],
            components: vec![Component::Ez, Component::Ex],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
        sim
    };
    let full = run(&mk(false))?;
    let quarter = run(&mk(true))?;
    let mut rep = CaseReport::new("symmetry_quarter_domain");
    rep.mcells_per_s = quarter.stats.mcells_per_s;
    for c in [Component::Ez, Component::Ex] {
        let a = full.dft["f"].field(c).unwrap();
        let b = quarter.dft["f"].field(c).unwrap();
        let (mut num, mut den) = (0.0, 0.0);
        let n = a.re.len().min(b.re.len());
        for t in 0..n {
            num += ((a.re[t] - b.re[t]) as f64).powi(2) + ((a.im[t] - b.im[t]) as f64).powi(2);
            den += (a.re[t] as f64).powi(2) + (a.im[t] as f64).powi(2);
        }
        rep.check(&format!("rel L2 diff {}", c.name()), (num / den.max(1e-300)).sqrt(), 0.0, 1e-4);
    }
    Ok(rep)
}

/// Bloch boundaries with a zero wave vector are plain periodic boundaries.
pub fn bloch_zero_vector(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let n_cross = 2usize;
    let w = n_cross as f64 * dx;
    let freqs = band(3);
    let mk = |bloch: bool| {
        let mut sim = column(dx, 4e-6, n_cross);
        if bloch {
            sim.boundaries.y = [Boundary::Bloch { k: 0.0 }, Boundary::Bloch { k: 0.0 }];
            sim.boundaries.z = [Boundary::Bloch { k: 0.0 }, Boundary::Bloch { k: 0.0 }];
        }
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
        sim
    };
    let a = run(&mk(false))?;
    let b = run(&mk(true))?;
    let mut rep = CaseReport::new("bloch_zero_vector");
    rep.mcells_per_s = a.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        rep.check(&format!("T_bloch/T_periodic @ {:.3} f0", f / f0()), b.flux["T"][i] / a.flux["T"][i], 1.0, 1e-9);
    }
    Ok(rep)
}

/// Oblique incidence with Bloch-periodic boundaries (complex fields): an
/// s-polarized plane wave at 30 deg propagates with k_x = k0 cos(theta) and a
/// glass half-space reflects/transmits the oblique Fresnel fractions.
pub fn bloch_oblique(run: Runner) -> Result<CaseReport> {
    let theta: f64 = 30f64.to_radians();
    let n2 = 1.5;
    let dx = LAMBDA / 40.0;
    let ly = 8.0 * dx;
    let lz = 2.0 * dx;
    let k0 = TAU * f0() / C0;
    let ky = k0 * theta.sin();
    let sig = TimeSignal::gaussian(f0(), 0.06 * f0());
    let mk = |slab: bool| {
        let mut b = Boundaries::all(Boundary::Periodic);
        b.x = [Boundary::pml(), Boundary::pml()];
        b.y = [Boundary::Bloch { k: ky }, Boundary::Bloch { k: ky }];
        let mut sim = Simulation::new([0.0; 3], [4e-6, ly, lz], MeshSpec::Uniform { dx: [dx; 3] }, b, 300e-15);
        if slab {
            sim.media.push(Medium::from_index(n2).with_name("glass"));
            sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [2.5e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
        }
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ez, signal: sig.clone(), amplitude: 1.0, angle: theta, min: None, max: None });
        let w = [-1.0, ly + 1.0];
        let _ = w;
        sim.monitors.push(flux_x("T", 3.2e-6, ly, &[f0()]));
        sim.monitors.push(flux_x("R", 0.5e-6, ly, &[f0()]));
        for (nm, x) in [("a", 1.6e-6), ("b", 2.2e-6)] {
            sim.monitors.push(Monitor::DftField { name: nm.into(), min: [x, 0.0, 0.0], max: [x, 0.0, 0.0], components: vec![Component::Ez], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
        }
        sim
    };
    let vac = run(&mk(false))?;
    let gl = run(&mk(true))?;
    let g = mk(false).build_grid()?;
    let xa = g.node(0, vac.dft["a"].fields[0].origin[0]);
    let xb = g.node(0, vac.dft["b"].fields[0].origin[0]);
    let mut dphi = (vac.dft["b"].fields[0].at(0, 0, 0, 0) / vac.dft["a"].fields[0].at(0, 0, 0, 0)).arg();
    let expected = k0 * theta.cos() * (xb - xa);
    while dphi < expected - PI {
        dphi += TAU;
    }
    while dphi > expected + PI {
        dphi -= TAU;
    }
    let ct = theta.cos();
    let tt = (theta.sin() / n2).asin();
    let r_exp = ((ct - n2 * tt.cos()) / (ct + n2 * tt.cos())).powi(2);
    let p_inc = vac.flux["T"][0];
    let mut rep = CaseReport::new("bloch_oblique");
    rep.mcells_per_s = gl.stats.mcells_per_s;
    rep.check("k_x / (k0 cos theta)", dphi / (k0 * (xb - xa)) / ct, 1.0, 0.01);
    // normalisation: the reported source spectrum S is that of the injected
    // (analytic) signal, so the power is |S|^2 A cos(theta) / (2 eta0) and the
    // field per unit S is 1, as at normal incidence
    let (_, sre, sim_) = &vac.source_spectra["pw"];
    let s2 = sre[0] * sre[0] + sim_[0] * sim_[0];
    rep.check("T vacuum / (|S|^2 A cos / 2 eta0)", p_inc / (0.5 / ETA0 * s2 * ly * lz * ct), 1.0, 0.01);
    rep.check("|E| / |S| in vacuum", vac.dft["a"].fields[0].at(0, 0, 0, 0).norm() / s2.sqrt(), 1.0, 0.01);
    rep.check_abs("|R| vacuum (no reflection)", (vac.flux["R"][0] / p_inc).abs(), 0.0, 2e-3);
    rep.check("R_s oblique Fresnel", -gl.flux["R"][0] / p_inc, r_exp, 0.05);
    rep.check("T_s oblique Fresnel", gl.flux["T"][0] / p_inc, 1.0 - r_exp, 0.01);
    Ok(rep)
}

/// p-polarised oblique incidence in 3-D (polarization `Hz`: H normal to the
/// plane of incidence, k tilted toward y, Bloch k_y = k0 sin(theta)): k_x,
/// the two E components of the incident wave, and the Fresnel R_p, T_p of an
/// n = 2 half-space at 30 deg and at Brewster's angle.
pub fn bloch_oblique_p(run: Runner) -> Result<CaseReport> {
    let n2: f64 = 2.0;
    let dx = LAMBDA / 40.0;
    let (ly, lz) = (8.0 * dx, 2.0 * dx);
    let k0 = TAU * f0() / C0;
    let sig = TimeSignal::gaussian(f0(), 0.06 * f0());
    let mut rep = CaseReport::new("bloch_oblique_p");
    for theta in [30f64.to_radians(), n2.atan()] {
        let mk = |glass: bool| {
            let mut b = Boundaries::all(Boundary::Periodic);
            b.x = [Boundary::pml(), Boundary::pml()];
            b.y = [Boundary::Bloch { k: k0 * theta.sin() }, Boundary::Bloch { k: k0 * theta.sin() }];
            let mut sim = Simulation::new([0.0; 3], [4e-6, ly, lz], MeshSpec::Uniform { dx: [dx; 3] }, b, 300e-15);
            if glass {
                sim.media.push(Medium::from_index(n2).with_name("glass"));
                sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [2.5e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
            }
            sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Hz, signal: sig.clone(), amplitude: 1.0, angle: theta, min: None, max: None });
            sim.monitors.push(flux_x("T", 3.2e-6, ly, &[f0()]));
            sim.monitors.push(flux_x("R", 0.5e-6, ly, &[f0()]));
            for (nm, x) in [("a", 1.6e-6), ("b", 2.2e-6)] {
                sim.monitors.push(dft_point(nm, x, vec![Component::Ey, Component::Ex], &[f0()], 1));
            }
            sim
        };
        let vac = run(&mk(false))?;
        let gl = run(&mk(true))?;
        let (ct, tt) = (theta.cos(), (theta.sin() / n2).asin());
        let r_exp = ((n2 * ct - tt.cos()) / (n2 * ct + tt.cos())).powi(2);
        let p_inc = vac.flux["T"][0];
        let deg = theta.to_degrees();
        rep.mcells_per_s = gl.stats.mcells_per_s;
        if deg < 45.0 {
            let g = mk(false).build_grid()?;
            let ey = |m: &str| vac.dft[m].fields[0].at(0, 0, 0, 0);
            let (xa, xb) = (g.node(0, vac.dft["a"].fields[0].origin[0]), g.node(0, vac.dft["b"].fields[0].origin[0]));
            let expected = k0 * ct * (xb - xa);
            let mut dphi = (ey("b") / ey("a")).arg();
            while dphi < expected - PI {
                dphi += TAU;
            }
            while dphi > expected + PI {
                dphi -= TAU;
            }
            rep.check(&format!("{deg:.1} deg: k_x / (k0 cos theta)"), dphi / expected, 1.0, 0.01);
            let (_, sre, sim_) = &vac.source_spectra["pw"];
            let s2 = sre[0] * sre[0] + sim_[0] * sim_[0];
            rep.check(&format!("{deg:.1} deg: T vacuum / (|S|^2 A cos / 2 eta0)"), p_inc / (0.5 / ETA0 * s2 * ly * lz * ct), 1.0, 0.01);
            rep.check(&format!("{deg:.1} deg: |E_y| / (|S| cos theta)"), ey("a").norm() / (s2.sqrt() * ct), 1.0, 0.01);
            let ex = vac.dft["a"].fields[1].at(0, 0, 0, 0);
            rep.check(&format!("{deg:.1} deg: |E_x| / (|S| sin theta)"), ex.norm() / (s2.sqrt() * theta.sin()), 1.0, 0.02);
        }
        rep.check_abs(&format!("{deg:.1} deg: |R| without the glass"), (vac.flux["R"][0] / p_inc).abs(), 0.0, 2e-3);
        if r_exp > 1e-6 {
            rep.check(&format!("{deg:.1} deg: R_p Fresnel"), -gl.flux["R"][0] / p_inc, r_exp, 0.05);
        } else {
            rep.check_abs(&format!("{deg:.1} deg: R_p (Brewster)"), -gl.flux["R"][0] / p_inc, 0.0, 2e-3);
        }
        rep.check(&format!("{deg:.1} deg: T_p Fresnel"), gl.flux["T"][0] / p_inc, 1.0 - r_exp, 0.01);
    }
    Ok(rep)
}

/// FDTDX test_mixed_boundaries / test_pec_pmc_symmetry: (1) a TEM wave in a
/// channel with PEC walls in y and PMC walls in z keeps k = k0, |E|/|H| = 1
/// and the full incident power; (2) a wave hitting a PEC wall has an E node at
/// the wall, a PMC wall an E antinode.
pub fn mixed_boundaries(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("mixed_boundaries");
    let dx = LAMBDA / 30.0;
    let (ny, nz) = (6usize, 4usize);
    let (wy, wz) = (ny as f64 * dx, nz as f64 * dx);
    let mut b = Boundaries::all(Boundary::Pec);
    b.x = [Boundary::pml(), Boundary::pml()];
    b.z = [Boundary::Pmc, Boundary::Pmc];
    let mut sim = Simulation::new([0.0; 3], [4e-6, wy, wz], MeshSpec::Uniform { dx: [dx; 3] }, b, 70e-15);
    let freqs = band(3);
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(Monitor::Flux { name: "T".into(), axis: 0, position: 3.0e-6, min: [3.0e-6, -1.0, -1.0], max: [3.0e-6, 1.0, 1.0], freqs: freqs.clone(), apodization: Apodization::default() });
    let pa = [2.0e-6, wy / 2.0, wz / 2.0];
    let pb = [2.6e-6, wy / 2.0, wz / 2.0];
    for (nm, p) in [("a", pa), ("b", pb)] {
        sim.monitors.push(Monitor::DftField { name: nm.into(), min: p, max: p, components: vec![Component::Ey, Component::Hz], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
    }
    let res = run(&sim)?;
    let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
    // PEC walls sit on the domain faces; the high-side PMC wall comes from the
    // mirrored E ghost and sits half a cell inside the face, so the channel
    // is wz - dz/2 wide.
    for (i, f) in freqs.iter().enumerate() {
        let inc = 0.5 / ETA0 * sp[i] * wy * (wz - 0.5 * dx);
        rep.check(&format!("PEC-y/PMC-z channel: T/inc @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0, 0.01);
    }
    let g = sim.build_grid()?;
    let ey = |nm: &str| res.dft[nm].field(Component::Ey).unwrap().at(0, 0, 0, 0);
    let hz = |nm: &str| res.dft[nm].field(Component::Hz).unwrap().at(0, 0, 0, 0);
    let (xa, xb) = (g.node(0, res.dft["a"].fields[0].origin[0]), g.node(0, res.dft["b"].fields[0].origin[0]));
    let k0 = TAU * f0() / C0;
    rep.check("PEC-y/PMC-z channel: k / k0", wavenumber(ey("b") / ey("a"), xb - xa, k0) / k0, 1.0, 5e-3);
    rep.check("PEC-y/PMC-z channel: |E| / (eta0 |H|)", ey("a").norm() / hz("a").norm(), 1.0, 0.01);
    rep.mcells_per_s = res.stats.mcells_per_s;
    // standing waves in front of a PEC / PMC wall (x high side)
    for (label, wall) in [("PEC", Boundary::Pec), ("PMC", Boundary::Pmc)] {
        let dx = LAMBDA / 60.0;
        let mut sim = column(dx, 3e-6, 2);
        sim.boundaries.x[1] = wall;
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(f0(), 0.1 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.monitors.push(Monitor::DftField { name: "line".into(), min: [3e-6 - 0.6 * LAMBDA, 0.0, 0.0], max: [3e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
        sim.run_time = 120e-15;
        let r = run(&sim)?;
        let fl = r.dft["line"].field(Component::Ey).unwrap();
        let n = fl.dims[1];
        let mags: Vec<f64> = (0..n).map(|i| fl.at(0, i, 0, 0).norm()).collect();
        let peak = mags.iter().cloned().fold(0.0, f64::max);
        let near = mags[n - 1] / peak;
        if label == "PEC" {
            rep.check_abs("PEC wall: |E| one cell from the wall / max (node)", near, 0.0, 0.15);
        } else {
            rep.check_abs("PMC wall: |E| one cell from the wall / max (antinode)", near, 1.0, 0.05);
        }
    }
    Ok(rep)
}

/// Dipole in a box that is Bloch-periodic along x and y (k != 0) with PML in
/// z: used to check that x-decomposed runs (CPU slabs, several GPUs, ranks)
/// reproduce the single-slab complex fields, including the phased x wrap.
pub fn bloch_xy_box() -> Simulation {
    let k0 = TAU * f0() / C0;
    let mut b = Boundaries::all_pml();
    b.x = [Boundary::Bloch { k: 0.3 * k0 }, Boundary::Bloch { k: 0.3 * k0 }];
    b.y = [Boundary::Bloch { k: 0.2 * k0 }, Boundary::Bloch { k: 0.2 * k0 }];
    let dx = 50e-9;
    let mut sim = Simulation::new([0.0; 3], [1.2e-6, 0.8e-6, 1.2e-6], MeshSpec::Uniform { dx: [dx; 3] }, b, 40e-15);
    sim.sources.push(Source::PointDipole {
        name: "d".into(),
        position: [0.31e-6, 0.42e-6, 0.6e-6],
        polarization: Component::Ez,
        signal: pulse(),
        amplitude: 1e-12,
    });
    sim.monitors.push(Monitor::DftField {
        name: "f".into(),
        min: [0.0, 0.0, 0.6e-6],
        max: [1.2e-6, 0.8e-6, 0.6e-6],
        components: vec![Component::Ez, Component::Hx],
        freqs: vec![f0()],
        apodization: Apodization::default(),
        stride: 1,
    });
    sim
}
