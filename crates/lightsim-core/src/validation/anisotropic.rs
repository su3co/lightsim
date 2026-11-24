//! Anisotropic media: uniaxial crystals, rotated and lossy tensors, oriented
//! poles, director fields, gyrotropic (magneto-optic) poles.

use super::*;

/// Uniaxial crystal: the two polarizations see different indices (phase of
/// the transmitted DFT field versus the analytic propagation phase).
pub fn birefringence(run: Runner) -> Result<CaseReport> {
    let (ny_, nz_) = (1.5f64, 2.0f64);
    let mut rep = CaseReport::new("birefringence");
    for (pol, n_exp) in [(Component::Ey, ny_), (Component::Ez, nz_)] {
        let dx = LAMBDA / (40.0 * nz_);
        let n_cross = 2usize;
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 4e-6, n_cross);
        sim.background = "crystal".into();
        sim.media.push(Medium::anisotropic([1.0, ny_ * ny_, nz_ * nz_]).with_name("crystal"));
        sim.sources.push(Source::PlaneWave {
            name: "pw".into(),
            axis: 0,
            position: 0.8e-6,
            direction: Direction::Plus,
            polarization: pol,
            signal: pulse(),
            amplitude: 1.0,
            angle: 0.0,
            min: None,
            max: None,
        });
        let x1 = 1.5e-6;
        let x2 = 2.5e-6;
        sim.monitors.push(Monitor::DftField {
            name: "a".into(),
            min: [x1, 0.0, 0.0],
            max: [x1, w, w],
            components: vec![pol],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
        sim.monitors.push(Monitor::DftField {
            name: "b".into(),
            min: [x2, 0.0, 0.0],
            max: [x2, w, w],
            components: vec![pol],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
        sim.run_time = 80e-15;
        let res = run(&sim)?;
        let a = res.dft["a"].fields[0].at(0, 0, 0, 0);
        let b = res.dft["b"].fields[0].at(0, 0, 0, 0);
        let g = sim.build_grid()?;
        let xa = g.node(0, res.dft["a"].fields[0].origin[0]);
        let xb = g.node(0, res.dft["b"].fields[0].origin[0]);
        let mut dphi = (b / a).arg();
        // unwrap with the expected phase
        let k0 = TAU * f0() / C0;
        let expected = k0 * n_exp * (xb - xa);
        while dphi < expected - PI {
            dphi += TAU;
        }
        while dphi > expected + PI {
            dphi -= TAU;
        }
        rep.check(&format!("n_eff ({})", pol.name()), dphi / (k0 * (xb - xa)), n_exp, 0.005);
        rep.check(&format!("|E| ratio ({})", pol.name()), b.norm() / a.norm(), 1.0, 0.01);
        rep.mcells_per_s = res.stats.mcells_per_s;
    }
    Ok(rep)
}

/// Uniaxial crystal (n_o, n_e) rotated 45 degrees about the propagation axis,
/// i.e. an off-diagonal permittivity coupling the two transverse components.
/// The eigen-polarizations are the rotated crystal axes; projecting the fields
/// at two depths onto them must give k0 n_e and k0 n_o separately, with the
/// amplitude ratio set by the per-eigenmode Fresnel transmission. `axis` 0
/// couples (Ey, Ez); `axis` 2 couples (Ex, Ey), which exercises the x-neighbour
/// averages (and the dD exchange when the run is split into x-slabs).
fn rotated_crystal(run: Runner, axis: usize, rep: &mut CaseReport) -> Result<()> {
    rotated_crystal_lossy(run, axis, (0.0, 0.0), rep)
}

/// The scene of `rotated_crystal_lossy`. With `director`, the (lossless)
/// crystal is given as a uniform director field along its optic axis
/// instead of a rotated tensor.
pub(super) fn rotated_crystal_sim(axis: usize, sig: (f64, f64), director: bool) -> Simulation {
    let (no, ne) = (1.5f64, 1.8f64);
    let lossy = sig.0 != 0.0 || sig.1 != 0.0;
    let w0 = TAU * f0();
    let dx = LAMBDA / (40.0 * ne);
    let (t1, t2) = if axis == 0 { (1, 2) } else { (0, 1) };
    let w = 2.0 * dx;
    let len = 6.0e-6;
    let mut b = Boundaries::all(Boundary::Periodic);
    *b.axis_mut(axis) = [Boundary::pml(), Boundary::pml()];
    let mut size = [w; 3];
    size[axis] = len;
    let mut sim = Simulation::new([0.0; 3], size, MeshSpec::Uniform { dx: [dx; 3] }, b, 90e-15);
    // crystal axes: optic axis along (t1 + t2)/sqrt2
    let (c, sn) = (std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2);
    let mut r = [[0.0; 3]; 3];
    r[axis][axis] = 1.0;
    r[t1][t1] = c;
    r[t2][t1] = sn;
    r[t1][t2] = -sn;
    r[t2][t2] = c;
    let mut d = [no * no; 3];
    d[t1] = ne * ne;
    let mut med = Medium::anisotropic(d).with_name("xtal");
    let eps0 = crate::constants::EPS0;
    let mut sg = [sig.1 * w0 * eps0; 3];
    sg[t1] = sig.0 * w0 * eps0;
    if lossy {
        med.sigma = sg;
    }
    if director {
        let mut u = [0.0; 3];
        u[t1] = c;
        u[t2] = sn;
        let mut lc = Medium::dielectric(ne * ne).with_name("xtal");
        lc.director = Some(crate::materials::Director { eps_o: no * no, eps_e: ne * ne, min: [0.0; 3], max: [1.0; 3], dims: [1, 1, 1], director: u.to_vec() });
        sim.media.push(lc);
    } else {
        sim.media.push(med.rotated(r));
    }
    let (mut lo, mut hi) = ([-1.0; 3], [1.0; 3]);
    lo[axis] = 2.5e-6;
    hi[axis] = 20e-6;
    sim.structures.push(Structure { name: "x".into(), shape: Shape::Box { min: lo, max: hi }, medium: "xtal".into() });
    sim.sources.push(Source::PlaneWave {
        name: "pw".into(),
        axis,
        position: 1.0e-6,
        direction: Direction::Plus,
        polarization: Component::e(t1),
        signal: TimeSignal::gaussian(f0(), 0.15 * f0()),
        amplitude: 1.0,
        angle: 0.0,
        min: None,
        max: None,
    });
    let depths = [3.2e-6, 4.0e-6];
    for (nm, z) in [("a", depths[0]), ("b", depths[1])] {
        let mut p = [0.0; 3];
        p[axis] = z;
        sim.monitors.push(Monitor::DftField {
            name: nm.into(),
            min: p,
            max: p,
            components: vec![Component::e(t1), Component::e(t2)],
            freqs: vec![f0()],
            apodization: Apodization::default(),
            stride: 1,
        });
    }
    sim
}

/// As `rotated_crystal`, with conductivities `sig = (Im eps_e, Im eps_o)` at
/// f0 along the crystal axes: the eigen-polarizations then carry complex
/// indices sqrt(eps + i sigma / (w eps0)) (cf. FDTDX test_complex_tensor).
fn rotated_crystal_lossy(run: Runner, axis: usize, sig: (f64, f64), rep: &mut CaseReport) -> Result<()> {
    let (no, ne) = (1.5f64, 1.8f64);
    let lossy = sig.0 != 0.0 || sig.1 != 0.0;
    let n_e = Complex64::new(ne * ne, sig.0).sqrt();
    let n_o = Complex64::new(no * no, sig.1).sqrt();
    let (t1, t2) = if axis == 0 { (1, 2) } else { (0, 1) };
    let sim = rotated_crystal_sim(axis, sig, false);
    let res = run(&sim)?;
    let g = sim.build_grid()?;
    let get = |nm: &str, comp: usize| res.dft[nm].field(Component::e(comp)).unwrap().at(0, 0, 0, 0);
    let x_of = |nm: &str| g.node(axis, res.dft[nm].fields[0].origin[axis]);
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let plus = |nm: &str| (get(nm, t1) + get(nm, t2)) * s2;
    let minus = |nm: &str| (get(nm, t2) - get(nm, t1)) * s2;
    let k0 = TAU * f0() / C0;
    let dz = x_of("b") - x_of("a");
    let index = |ra: Complex64, rb: Complex64, guess: f64| {
        let mut ph = (rb / ra).arg();
        let e = k0 * guess * dz;
        while ph < e - PI {
            ph += TAU;
        }
        while ph > e + PI {
            ph -= TAU;
        }
        ph / (k0 * dz)
    };
    let ax = ["x", "y", "z"][axis];
    if lossy {
        // complex k from the projections: phase -> Re, decay -> Im (phasors ~ e^{+ikz})
        for (lbl, f, nn) in [("e", &plus as &dyn Fn(&str) -> Complex64, n_e), ("o", &minus as &dyn Fn(&str) -> Complex64, n_o)] {
            let re = index(f("a"), f("b"), nn.re);
            let im = -(f("b").norm() / f("a").norm()).ln() / (k0 * dz);
            rep.check(&format!("prop {ax}: Re n_{lbl} (lossy rotated)"), re, nn.re, 5e-3);
            rep.check(&format!("prop {ax}: Im n_{lbl} (lossy rotated)"), im, nn.im, 0.03);
        }
        let t_ratio = ((Complex64::new(1.0, 0.0) + n_e) / (Complex64::new(1.0, 0.0) + n_o)).norm();
        rep.check(&format!("prop {ax}: |E_o| / |E_e| vs complex Fresnel t_o / t_e (at a)"), minus("a").norm() / plus("a").norm(), t_ratio * (-(k0 * (n_o.im - n_e.im) * (x_of("a") - 2.5e-6))).exp(), 0.03);
        rep.mcells_per_s = res.stats.mcells_per_s;
        return Ok(());
    }
    rep.check(&format!("prop {ax}: n along rotated optic axis / n_e"), index(plus("a"), plus("b"), ne) / ne, 1.0, 5e-3);
    rep.check(&format!("prop {ax}: n along rotated ordinary axis / n_o"), index(minus("a"), minus("b"), no) / no, 1.0, 5e-3);
    rep.check(&format!("prop {ax}: |E_e(b)| / |E_e(a)| (no mode mixing)"), plus("b").norm() / plus("a").norm(), 1.0, 0.01);
    rep.check(&format!("prop {ax}: |E_o| / |E_e| vs Fresnel t_o / t_e"), minus("a").norm() / plus("a").norm(), (1.0 + ne) / (1.0 + no), 0.02);
    rep.mcells_per_s = res.stats.mcells_per_s;
    Ok(())
}

pub fn lossy_offdiagonal_tensor(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("lossy_offdiagonal_tensor");
    rotated_crystal_lossy(run, 0, (0.2, 0.05), &mut rep)?;
    rotated_crystal_lossy(run, 2, (0.2, 0.05), &mut rep)?;
    Ok(rep)
}

pub fn offdiagonal_tensor(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("offdiagonal_tensor");
    rotated_crystal(run, 0, &mut rep)?;
    rotated_crystal(run, 2, &mut rep)?;
    Ok(rep)
}

/// z-propagating plane-wave column (periodic x, y) in a half-space medium.
fn zcolumn(dx: f64, med: Medium, pol: Component, depths: &[f64], t_run: f64) -> Simulation {
    let w = 2.0 * dx;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.z = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [w, w, 6.0e-6], MeshSpec::Uniform { dx: [dx; 3] }, b, t_run);
    let name = med.name.clone();
    sim.media.push(med);
    sim.structures.push(Structure { name: "m".into(), shape: Shape::Box { min: [-1.0, -1.0, 2.5e-6], max: [1.0, 1.0, 20e-6] }, medium: name });
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f0(), 0.15 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
    for (i, &z) in depths.iter().enumerate() {
        sim.monitors.push(Monitor::DftField { name: format!("d{i}"), min: [0.0, 0.0, z], max: [0.0, 0.0, z], components: vec![Component::Ex, Component::Ey], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
    }
    sim
}

/// FDTDX test_oriented_dispersion: (1) a pole oriented along x reproduces the
/// per-axis x pole; (2) a dispersive uniaxial crystal (oscillator along
/// (x+y)/sqrt2, propagation along z) has the complex index sqrt(eps_inf +
/// chi(w)) along u and sqrt(eps_inf) across it; (3) a monoclinic pair of
/// oscillators 30 degrees apart stays stable.
pub fn oriented_dispersion(run: Runner) -> Result<CaseReport> {
    use crate::materials::OrientedPole;
    let mut rep = CaseReport::new("oriented_dispersion");
    let w0 = TAU * f0();
    let pole = Pole::Lorentz { delta_eps: 1.0, omega0: 2.0 * w0, gamma: 1e13 };
    let dx = LAMBDA / (40.0 * 1.9);
    let depths = [3.2e-6, 4.0e-6];
    // (1) oriented along x == per-axis x pole
    let mut per_axis = Medium::dielectric(2.25).with_name("pa").with_poles(vec![pole.clone()]);
    per_axis.pole_axes = [true, false, false];
    let mut oriented = Medium::dielectric(2.25).with_name("or");
    oriented.oriented = vec![OrientedPole { pole: pole.clone(), u: [1.0, 0.0, 0.0] }];
    let ra = run(&zcolumn(dx, per_axis, Component::Ex, &depths, 90e-15))?;
    let rb = run(&zcolumn(dx, oriented, Component::Ex, &depths, 90e-15))?;
    let ea = ra.dft["d1"].field(Component::Ex).unwrap().at(0, 0, 0, 0);
    let eb = rb.dft["d1"].field(Component::Ex).unwrap().at(0, 0, 0, 0);
    rep.check_abs("oriented along x vs per-axis pole: |dE|/|E|", (ea - eb).norm() / ea.norm(), 0.0, 1e-4);
    // (2) rotated dispersive crystal
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut xtal = Medium::dielectric(2.25).with_name("dx");
    xtal.oriented = vec![OrientedPole { pole: pole.clone(), u: [1.0, 1.0, 0.0] }];
    let sim = zcolumn(dx, xtal, Component::Ex, &depths, 90e-15);
    let res = run(&sim)?;
    let g = sim.build_grid()?;
    let get = |nm: &str, c: Component| res.dft[nm].field(c).unwrap().at(0, 0, 0, 0);
    let along = |nm: &str| (get(nm, Component::Ex) + get(nm, Component::Ey)) * s2;
    let across = |nm: &str| (get(nm, Component::Ey) - get(nm, Component::Ex)) * s2;
    let z_of = |nm: &str| g.node(2, res.dft[nm].fields[0].origin[2]);
    let dz = z_of("d1") - z_of("d0");
    let k0 = w0 / C0;
    let eps_u = Complex64::new(2.25, 0.0) + pole.chi_discrete(w0, res.dt);
    let n_u = eps_u.sqrt();
    let n_p = 1.5f64;
    let re_n = |r: Complex64, guess: f64| wavenumber(r, dz, k0 * guess) / k0;
    rep.check("rotated dispersive crystal: Re n along u", re_n(along("d1") / along("d0"), n_u.re), n_u.re, 5e-3);
    rep.check("rotated dispersive crystal: n across u", re_n(across("d1") / across("d0"), n_p), n_p, 5e-3);
    let t_ratio = ((Complex64::new(1.0, 0.0) + n_u) / (1.0 + n_p)).norm();
    rep.check("rotated dispersive crystal: |E_across| / |E_along| vs Fresnel", across("d0").norm() / along("d0").norm(), t_ratio, 0.03);
    // (3) monoclinic pair (30 degrees apart): long run stays bounded
    let mut mono = Medium::dielectric(2.25).with_name("mono");
    let a30 = 30f64.to_radians();
    mono.oriented = vec![
        OrientedPole { pole: Pole::Lorentz { delta_eps: 1.5, omega0: 1.2 * w0, gamma: 2e13 }, u: [1.0, 0.0, 0.0] },
        OrientedPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: 0.9 * w0, gamma: 2e13 }, u: [a30.cos(), a30.sin(), 0.0] },
    ];
    let mut sim = zcolumn(dx, mono, Component::Ex, &depths, 400e-15);
    sim.monitors.push(Monitor::Probe { name: "p".into(), position: [0.0, 0.0, 3.0e-6], components: vec![Component::Ex, Component::Ey] });
    let r = run(&sim)?;
    let tr = &r.time["p"];
    let mut peak = 0f64;
    let mut late = 0f64;
    let n = tr.steps.len();
    for fld in &tr.fields {
        for (i, v) in fld.data.iter().enumerate() {
            let v = (*v as f64).abs();
            if !v.is_finite() {
                late = f64::INFINITY;
            }
            peak = peak.max(v);
            if i >= n * 9 / 10 {
                late = late.max(v);
            }
        }
    }
    rep.check_abs("monoclinic pair: late-time max |E| / peak (stable)", late / peak, 0.0, 1e-2);
    rep.mcells_per_s = res.stats.mcells_per_s;
    Ok(rep)
}

/// Anisotropy that varies from cell to cell (`Medium::director`, a liquid
/// crystal).
///
/// (1) A uniform director along the optic axis of the `offdiagonal_tensor`
/// crystal (rotated 45 degrees about the propagation axis, x and z) gives
/// the same fields as the rotated tensor medium.
///
/// (2) A twisted nematic layer (n_o = 1.5, n_e = 1.6, 10 um thick, the
/// director in the transverse plane turning uniformly by 90 degrees) between
/// index-matched half-spaces (n = 1.55), lit by a pulse polarised along the
/// director at the entrance: the transmitted Jones vector, normalised by a
/// run without the layer, against the Jones-matrix product of 4000 thin
/// rotated retarders, at five frequencies from 0.8 to 1.2 f0 (Mauguin
/// parameter Gamma / 2 Phi = 1.6 to 2.4: the polarisation follows the
/// twist, with an ellipticity the Jones matrix predicts). Compared are
/// the magnitudes of the two components, their relative phase and the
/// azimuth of the polarisation ellipse (90 degrees in the Mauguin limit).
pub fn director_field(run: Runner) -> Result<CaseReport> {
    use crate::materials::Director;
    let mut rep = CaseReport::new("director_field");
    // (1) uniform director == rotated tensor
    for axis in [0usize, 2] {
        let (t1, t2) = if axis == 0 { (1, 2) } else { (0, 1) };
        let a = run(&rotated_crystal_sim(axis, (0.0, 0.0), false))?;
        let b = run(&rotated_crystal_sim(axis, (0.0, 0.0), true))?;
        let mut d = 0f64;
        for nm in ["a", "b"] {
            for c in [t1, t2] {
                d = d.max(dft_rel_diff(&a, &b, nm, Component::e(c)));
            }
        }
        rep.check_abs(&format!("uniform director vs rotated tensor (propagation {}): max |dE| / peak", ["x", "y", "z"][axis]), d, 0.0, 1e-6);
    }
    // (2) twisted nematic layer
    let (no, ne, ns) = (1.5f64, 1.6f64, 1.55f64);
    let (z0, thick, twist) = (2.0e-6, 10.0e-6, PI / 2.0);
    let freqs: Vec<f64> = (0..5).map(|i| f0() * (0.8 + 0.1 * i as f64)).collect();
    let dx = C0 / (1.2 * f0()) / (40.0 * ne);
    let zm = z0 + thick + 1.0e-6;
    let mk = |layer: bool, two_d: bool| {
        let w = 2.0 * dx;
        let mut b = Boundaries::all(Boundary::Periodic);
        b.z = [Boundary::pml(), Boundary::pml()];
        let mut sim = Simulation::new([0.0; 3], [w, if two_d { dx } else { w }, z0 + thick + 2.0e-6], MeshSpec::Uniform { dx: [dx; 3] }, b, 140e-15);
        if two_d {
            // y-invariant; the twist couples the two field sets
            sim = sim.two_d(1, crate::spec::Polarization::Both);
        }
        sim.media.push(Medium::from_index(ns).with_name("bg"));
        sim.background = "bg".into();
        if layer {
            let n = 1001;
            let mut dir = Vec::with_capacity(3 * n);
            for i in 0..n {
                let phi = twist * i as f64 / (n - 1) as f64;
                dir.extend([phi.cos(), phi.sin(), 0.0]);
            }
            let mut lc = Medium::from_index(ne).with_name("lc");
            lc.director = Some(Director { eps_o: no * no, eps_e: ne * ne, min: [0.0, 0.0, z0], max: [0.0, 0.0, z0 + thick], dims: [1, 1, n], director: dir });
            sim.media.push(lc);
            sim.structures.push(Structure { name: "tn".into(), shape: Shape::Box { min: [-1.0, -1.0, z0], max: [1.0, 1.0, z0 + thick] }, medium: "lc".into() });
        }
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ex, signal: TimeSignal::gaussian(f0(), 0.25 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.monitors.push(Monitor::DftField { name: "out".into(), min: [0.0, 0.0, zm], max: [0.0, 0.0, zm], components: vec![Component::Ex, Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
        sim
    };
    let azimuth = |x: Complex64, y: Complex64| 0.5 * (2.0 * (x * y.conj()).re).atan2(x.norm_sqr() - y.norm_sqr());
    let mut speed = 0.0;
    for two_d in [false, true] {
        let tag = if two_d { "2-D, " } else { "" };
        let tn = run(&mk(true, two_d))?;
        let rf = run(&mk(false, two_d))?;
        if !two_d {
            speed = tn.stats.mcells_per_s;
        }
        for (fi, &f) in freqs.iter().enumerate() {
            let k0 = TAU * f / C0;
            // Jones matrix: product of thin retarders, J_m = R(phi) diag(e^{i k0 ne dz}, e^{i k0 no dz}) R(-phi)
            let m = 4000;
            let dz = thick / m as f64;
            let mut u = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
            for q in 0..m {
                let phi = twist * (q as f64 + 0.5) / m as f64;
                let (c, sn) = (phi.cos(), phi.sin());
                let (pe, po) = (Complex64::from_polar(1.0, k0 * ne * dz), Complex64::from_polar(1.0, k0 * no * dz));
                // to the local (e, o) frame, retard, back
                let (le, lo) = (c * u[0] + sn * u[1], -sn * u[0] + c * u[1]);
                let (le, lo) = (le * pe, lo * po);
                u = [c * le - sn * lo, sn * le + c * lo];
            }
            let er = tn.dft["out"].field(Component::Ex).unwrap().at(fi, 0, 0, 0);
            let ey = tn.dft["out"].field(Component::Ey).unwrap().at(fi, 0, 0, 0);
            let e0 = rf.dft["out"].field(Component::Ex).unwrap().at(fi, 0, 0, 0);
            let (vx, vy) = (er / e0, ey / e0);
            let r = f / f0();
            rep.check_abs(&format!("{tag}TN 90 deg: |E_x| / |E_inc| @ {r:.1} f0"), vx.norm(), u[0].norm(), 5e-3);
            rep.check_abs(&format!("{tag}TN 90 deg: |E_y| / |E_inc| @ {r:.1} f0"), vy.norm(), u[1].norm(), 5e-3);
            let dph = (vy / vx / (u[1] / u[0])).arg();
            rep.check_abs(&format!("{tag}TN 90 deg: arg(E_y / E_x) - Jones (rad) @ {r:.1} f0"), dph, 0.0, 0.02);
            rep.check_abs(&format!("{tag}TN 90 deg: ellipse azimuth (deg) vs Jones {:.2} @ {r:.1} f0", azimuth(u[0], u[1]).to_degrees().rem_euclid(180.0)), azimuth(vx, vy).to_degrees().rem_euclid(180.0), azimuth(u[0], u[1]).to_degrees().rem_euclid(180.0), 0.5);
        }
    }
    rep.mcells_per_s = speed;
    Ok(rep)
}

/// Plane-wave column along `axis` (periodic across, PML along) with
/// `med` filling `lo..hi` along the axis; a pulse polarised along the first
/// transverse axis launched at `src` in `dir`; DFT monitors of the two
/// transverse E components at `depths`.
pub(super) fn gyro_column(axis: usize, dx: f64, med: Medium, (lo, hi): (f64, f64), src: f64, dir: Direction, depths: &[f64], freqs: &[f64], t_run: f64, two_d: Option<usize>) -> Simulation {
    let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
    let (t1, t2) = (t1.min(t2), t1.max(t2));
    let mut b = Boundaries::all(Boundary::Periodic);
    *b.axis_mut(axis) = [Boundary::pml(), Boundary::pml()];
    let mut size = [2.0 * dx; 3];
    size[axis] = 6.0e-6;
    if let Some(w) = two_d {
        size[w] = dx;
    }
    let mut sim = Simulation::new([0.0; 3], size, MeshSpec::Uniform { dx: [dx; 3] }, b, t_run);
    if let Some(w) = two_d {
        sim = sim.two_d(w, crate::spec::Polarization::Both);
    }
    let name = med.name.clone();
    sim.media.push(med);
    let (mut mn, mut mx) = ([-1.0; 3], [1.0; 3]);
    mn[axis] = lo;
    mx[axis] = hi;
    sim.structures.push(Structure { name: "m".into(), shape: Shape::Box { min: mn, max: mx }, medium: name });
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis, position: src, direction: dir, polarization: Component::e(t1), signal: TimeSignal::gaussian(f0(), 0.15 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
    for (i, &z) in depths.iter().enumerate() {
        let mut p = [0.0; 3];
        p[axis] = z;
        sim.monitors.push(Monitor::DftField { name: format!("d{i}"), min: p, max: p, components: vec![Component::e(t1), Component::e(t2)], freqs: freqs.to_vec(), apodization: Apodization::default(), stride: 1 });
    }
    sim
}

/// Magneto-optic (gyrotropic) media, `Medium::gyro`: Faraday geometry, the
/// bias along the propagation axis. Inside a half-space the two circular
/// polarizations (1, +-i) of the transverse pair (t1, t2) travel with
/// n_+- = sqrt(eps_inf + chi_xx +- i chi_xy) of the scheme's discrete
/// susceptibility tensor, measured from the phase of the circular
/// projections between two depths, and the axis of the polarization ellipse
/// turns by (k0/2) Re(n_- - n_+) per length (from t1 towards t2).
///
/// (1) Lorentz pole, bias along z, three frequencies; (2) the same medium
/// as a half-space on the other side, crossed backwards: the rotation per
/// distance travelled has the same sense in the lab frame (non-reciprocal;
/// a round trip doubles it); (3) Drude pole (magnetised plasma), bias along
/// x, propagation along x (the x-neighbour averages); (4) case (1) as a 2-D
/// simulation (invariant y); (5) a long run of (3) stays bounded.
pub fn gyrotropic_faraday(run: Runner) -> Result<CaseReport> {
    use crate::materials::GyroPole;
    let mut rep = CaseReport::new("gyrotropic_faraday");
    let w0 = TAU * f0();
    let freqs = vec![0.9 * f0(), f0(), 1.1 * f0()];
    let lorentz = GyroPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: 1.5 * w0, gamma: 1e13 }, bias: [0.0, 0.0, 0.2 * w0] };
    let drude = GyroPole { pole: Pole::Drude { omega_p: 0.5 * w0, gamma: 1e13 }, bias: [0.3 * w0, 0.0, 0.0] };
    // n_+- of the discrete model for the transverse pair (t1, t2)
    let n_pm = |gp: &GyroPole, eps_inf: f64, t1: usize, t2: usize, w: f64, dt: f64| {
        let chi = gp.chi_discrete(w, dt);
        let i = Complex64::i();
        let root = |e: Complex64| {
            let n = e.sqrt();
            if n.im < 0.0 { -n } else { n }
        };
        (root(eps_inf + chi[t1][t1] + i * chi[t1][t2]), root(eps_inf + chi[t1][t1] - i * chi[t1][t2]))
    };
    let angle = |e1: Complex64, e2: Complex64| 0.5 * (2.0 * (e1 * e2.conj()).re).atan2(e1.norm_sqr() - e2.norm_sqr());
    let wrap_pi = |mut d: f64, guess: f64| {
        while d < guess - PI / 2.0 {
            d += PI;
        }
        while d > guess + PI / 2.0 {
            d -= PI;
        }
        d
    };
    struct Meas {
        n_plus: Vec<f64>,
        n_minus: Vec<f64>,
        rot: Vec<f64>,
        expect: Vec<(Complex64, Complex64)>,
        dist: f64,
    }
    let measure = |sim: &Simulation, gp: &GyroPole, eps_inf: f64, axis: usize| -> Result<(Meas, Results)> {
        let res = run(sim)?;
        let g = sim.build_grid()?;
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
        let (t1, t2) = (t1.min(t2), t1.max(t2));
        let pos = |nm: &str| g.node(axis, res.dft[nm].fields[0].origin[axis]);
        let dist = (pos("d1") - pos("d0")).abs();
        let mut m = Meas { n_plus: vec![], n_minus: vec![], rot: vec![], expect: vec![], dist };
        for (fi, &f) in freqs.iter().enumerate() {
            let get = |nm: &str, c: usize| res.dft[nm].field(Component::e(c)).unwrap().at(fi, 0, 0, 0);
            let k0 = TAU * f / C0;
            let (np, nm_) = n_pm(gp, eps_inf, t1, t2, TAU * f, res.dt);
            // circular amplitudes: E = a+ (1, i) + a- (1, -i)
            let ap = |nm: &str| (get(nm, t1) - Complex64::i() * get(nm, t2)) * 0.5;
            let am = |nm: &str| (get(nm, t1) + Complex64::i() * get(nm, t2)) * 0.5;
            // d0 is met first: a(d1) / a(d0) = exp(i k dist) in both directions
            let ratio = |a: &dyn Fn(&str) -> Complex64| a("d1") / a("d0");
            m.n_plus.push(wavenumber(ratio(&ap), dist, k0 * np.re) / k0);
            m.n_minus.push(wavenumber(ratio(&am), dist, k0 * nm_.re) / k0);
            let guess = 0.5 * k0 * (nm_.re - np.re) * dist;
            m.rot.push(wrap_pi(angle(get("d1", t1), get("d1", t2)) - angle(get("d0", t1), get("d0", t2)), guess));
            m.expect.push((np, nm_));
        }
        Ok((m, res))
    };
    let check = |rep: &mut CaseReport, tag: &str, m: &Meas| {
        for (fi, &f) in freqs.iter().enumerate() {
            let k0 = TAU * f / C0;
            let (np, nm_) = m.expect[fi];
            let r = f / f0();
            rep.check(&format!("{tag}: Re n_+ @ {r:.1} f0"), m.n_plus[fi], np.re, 2e-3);
            rep.check(&format!("{tag}: Re n_- @ {r:.1} f0"), m.n_minus[fi], nm_.re, 2e-3);
            rep.check(&format!("{tag}: rotation over {:.2} um @ {r:.1} f0", m.dist * 1e6), m.rot[fi], 0.5 * k0 * (nm_.re - np.re) * m.dist, 1e-2);
        }
    };
    // (1) Lorentz, bias z, forward
    let eps_inf = 2.25;
    let mut med = Medium::dielectric(eps_inf).with_name("gyro_l");
    med.gyro = vec![lorentz.clone()];
    let dx = LAMBDA / (40.0 * 2.2);
    let depths = [3.2e-6, 4.0e-6];
    let (fwd, res) = measure(&gyro_column(2, dx, med.clone(), (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &freqs, 110e-15, None), &lorentz, eps_inf, 2)?;
    check(&mut rep, "Lorentz, bias z", &fwd);
    rep.mcells_per_s = res.stats.mcells_per_s;
    // (2) backwards through the mirrored half-space: d0 = 2.8 um is met first
    let depths_b = [2.8e-6, 2.0e-6];
    let (bwd, _) = measure(&gyro_column(2, dx, med.clone(), (-20e-6, 3.5e-6), 5.0e-6, Direction::Minus, &depths_b, &freqs, 110e-15, None), &lorentz, eps_inf, 2)?;
    check(&mut rep, "Lorentz, bias z, backwards", &bwd);
    for (fi, &f) in freqs.iter().enumerate() {
        let r = f / f0();
        rep.check(&format!("non-reciprocal: rotation backwards / forwards (lab frame) @ {r:.1} f0"), bwd.rot[fi] / bwd.dist / (fwd.rot[fi] / fwd.dist), 1.0, 1e-2);
    }
    // (3) Drude, bias x, propagation x
    let mut plasma = Medium::dielectric(1.0).with_name("gyro_d");
    plasma.gyro = vec![drude.clone()];
    let dxp = LAMBDA / 50.0;
    let (dm, _) = measure(&gyro_column(0, dxp, plasma.clone(), (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &freqs, 110e-15, None), &drude, 1.0, 0)?;
    check(&mut rep, "Drude, bias x", &dm);
    // (4) case (1) as a 2-D simulation, invariant y
    let (m2, _) = measure(&gyro_column(2, dx, med, (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &freqs, 110e-15, Some(1)), &lorentz, eps_inf, 2)?;
    check(&mut rep, "2-D, Lorentz, bias z", &m2);
    // (5) stability: a long run of the plasma column
    let mut sim = gyro_column(0, dxp, plasma, (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &[f0()], 600e-15, None);
    sim.monitors.push(Monitor::Probe { name: "p".into(), position: [3.0e-6, 0.0, 0.0], components: vec![Component::Ey, Component::Ez] });
    let r = run(&sim)?;
    let tr = &r.time["p"];
    let n = tr.steps.len();
    let (mut peak, mut late) = (0f64, 0f64);
    for fld in &tr.fields {
        for (i, v) in fld.data.iter().enumerate() {
            let v = (*v as f64).abs();
            if !v.is_finite() {
                late = f64::INFINITY;
            }
            peak = peak.max(v);
            if i >= n * 9 / 10 {
                late = late.max(v);
            }
        }
    }
    rep.check_abs("plasma column, 600 fs: late-time max |E| / peak (stable)", late / peak, 0.0, 1e-3);
    Ok(rep)
}
