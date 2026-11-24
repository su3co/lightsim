//! 2-D simulations: the 2-D update against the 3-D one, Fresnel, cylinders,
//! slab modes, CPML, gratings, Gaussian beams.

use super::*;

/// In-plane coordinates (a, b) and the coordinate `c` along the invariant
/// axis `w`, as a 3-vector.
fn lift2(w: usize, a: f64, b: f64, c: f64) -> [f64; 3] {
    match w {
        0 => [c, a, b],
        1 => [a, c, b],
        _ => [a, b, c],
    }
}

/// Small 2-D problems (`dimensions = 2`) that together exercise what the 2-D
/// update has to get right: the three invariant axes, both field sets,
/// CPML, PEC / PMC walls, periodic and Bloch boundaries along either
/// in-plane axis, dielectric, lossy, dispersive and nonlinear media,
/// electric and magnetic dipoles, plane waves and TFSF boxes. Every scene has a DFT
/// monitor `f` over all six components. Used to compare the 2-D kernels
/// with the general 3-D ones on the same grid ([`two_d_matches_one_cell`]
/// and the GPU tests).
pub fn two_d_scenes() -> Vec<(String, Simulation)> {
    let fc = 2.0e14;
    let um = 1e-6;
    let sig = TimeSignal::gaussian(fc, 0.4 * fc);
    let k0 = TAU * fc / C0;
    let pml = || [Boundary::Pml { layers: 8, params: Default::default(), inside: false }, Boundary::Pml { layers: 8, params: Default::default(), inside: false }];
    let per = || [Boundary::Periodic, Boundary::Periodic];
    let bloch = |k: f64| [Boundary::Bloch { k }, Boundary::Bloch { k }];
    let base = |w: usize, bu: [Boundary; 2], bv: [Boundary; 2]| -> Simulation {
        let (u, v, _) = crate::spec::plane_of(w);
        let mut b = Boundaries::all_pml();
        *b.axis_mut(u) = bu;
        *b.axis_mut(v) = bv;
        let mut sim = Simulation::new(lift2(w, -um, -0.8 * um, -0.5 * um), lift2(w, 1.0 * um, 0.8 * um, 0.5 * um), MeshSpec::Uniform { dx: [40e-9, 36e-9, 44e-9] }, b, 14e-15)
            .two_d(w, crate::spec::Polarization::Both);
        let om = TAU * fc;
        sim.media.push(Medium::from_index(2.0).with_name("glass"));
        sim.media.push(Medium::conductor(2.25, 3e4).with_name("lossy"));
        sim.media.push(Medium::dielectric(2.0).with_name("lor").with_poles(vec![Pole::Lorentz { delta_eps: 1.5, omega0: 1.8 * om, gamma: 0.05 * om }]));
        sim.structures.push(Structure { name: "rod".into(), shape: Shape::Cylinder { center: lift2(w, -0.2 * um, 0.1 * um, 0.0), radius: 0.25 * um, length: 10.0, axis: w }, medium: "glass".into() });
        sim.structures.push(Structure { name: "blk".into(), shape: Shape::Box { min: lift2(w, 0.25 * um, -0.5 * um, -1.0), max: lift2(w, 0.5 * um, -0.1 * um, 1.0) }, medium: "lossy".into() });
        sim.structures.push(Structure { name: "lor".into(), shape: Shape::Box { min: lift2(w, 0.2 * um, 0.2 * um, -1.0), max: lift2(w, 0.6 * um, 0.5 * um, 1.0) }, medium: "lor".into() });
        sim.monitors.push(Monitor::DftField {
            name: "f".into(),
            min: lift2(w, -0.7 * um, -0.6 * um, 0.0),
            max: lift2(w, 0.7 * um, 0.6 * um, 0.0),
            components: Component::ALL.to_vec(),
            freqs: vec![0.9 * fc, 1.1 * fc],
            apodization: Apodization::default(),
            stride: 1,
        });
        sim
    };
    let dip = |sim: &mut Simulation, w: usize, a: f64, b: f64, c: Component| {
        sim.sources.push(Source::PointDipole { name: format!("d{}", sim.sources.len()), position: lift2(w, a, b, 0.0), polarization: c, signal: sig.clone(), amplitude: 1e-12 })
    };
    let mut out = vec![];
    for w in 0..3 {
        let ax = ["x", "y", "z"][w];
        let (u, v, _) = crate::spec::plane_of(w);
        // both sets from an electric and a magnetic dipole along the invariant axis
        let mut sim = base(w, pml(), pml());
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, -0.55 * um, 0.35 * um, Component::h(w));
        out.push((format!("{ax}-invariant, CPML, both sets"), sim));
        // one set only
        let mut sim = base(w, pml(), pml());
        dip(&mut sim, w, -0.55 * um, 0.35 * um, Component::h(w));
        out.push((format!("{ax}-invariant, CPML, TE"), sim));
        let mut sim = base(w, pml(), pml());
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        out.push((format!("{ax}-invariant, CPML, TM"), sim));
        // walls
        let mut sim = base(w, [Boundary::Pec, Boundary::Pmc], [Boundary::Pmc, Boundary::Pml { layers: 8, params: Default::default(), inside: false }]);
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, 0.0, 0.4 * um, Component::e(u));
        out.push((format!("{ax}-invariant, PEC / PMC / CPML"), sim));
        // periodic along u, CPML along v, and the other way round
        let mut sim = base(w, per(), pml());
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, -0.55 * um, 0.35 * um, Component::h(w));
        out.push((format!("{ax}-invariant, periodic along the outer in-plane axis"), sim));
        let mut sim = base(w, pml(), per());
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, 0.1 * um, 0.3 * um, Component::e(v));
        out.push((format!("{ax}-invariant, periodic along the inner in-plane axis"), sim));
        // Bloch (complex fields) along one and along both in-plane axes
        let mut sim = base(w, pml(), bloch(0.3 * k0));
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, 0.1 * um, 0.3 * um, Component::e(u));
        out.push((format!("{ax}-invariant, Bloch along the inner in-plane axis"), sim));
        let mut sim = base(w, bloch(-0.2 * k0), bloch(0.3 * k0));
        dip(&mut sim, w, -0.6 * um, -0.3 * um, Component::e(w));
        dip(&mut sim, w, 0.1 * um, 0.3 * um, Component::e(u));
        out.push((format!("{ax}-invariant, Bloch along both in-plane axes"), sim));
        // plane waves of both polarisations along u, periodic along v
        for (name, pol) in [("TM", Component::e(w)), ("TE", Component::e(v))] {
            let mut sim = base(w, pml(), per());
            sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: u, position: -0.75 * um, direction: Direction::Plus, polarization: pol, signal: sig.clone(), amplitude: 1.0, angle: 0.0, min: None, max: None });
            out.push((format!("{ax}-invariant, plane wave ({name})"), sim));
        }
        // oblique plane waves (complex source profile) with the matching Bloch vector
        let th = 0.35f64;
        for (name, pol) in [("TM, s-polarised", Component::e(w)), ("TE, p-polarised", Component::e(v))] {
            let mut sim = base(w, pml(), bloch(k0 * th.sin()));
            sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: u, position: -0.75 * um, direction: Direction::Plus, polarization: pol, signal: sig.clone(), amplitude: 1.0, angle: th, min: None, max: None });
            out.push((format!("{ax}-invariant, oblique plane wave ({name})"), sim));
        }
        // TFSF boxes
        for (name, pol) in [("TM", Component::e(w)), ("TE", Component::e(v))] {
            let mut sim = base(w, pml(), pml());
            sim.sources.push(Source::Tfsf { name: "tf".into(), min: lift2(w, -0.6 * um, -0.5 * um, 0.0), max: lift2(w, 0.6 * um, 0.5 * um, 0.0), axis: u, direction: Direction::Plus, polarization: pol, signal: sig.clone(), amplitude: 1.0 });
            out.push((format!("{ax}-invariant, TFSF ({name})"), sim));
        }
        // a column one cell wide: a 1-D problem (plane wave through a lossy
        // and a dispersive slab), periodic along the second in-plane axis
        let mut sim = base(w, pml(), per());
        sim.region.min[v] = 0.0;
        sim.region.max[v] = 40e-9;
        sim.structures.retain(|st| st.name != "rod");
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: u, position: -0.75 * um, direction: Direction::Plus, polarization: Component::e(v), signal: sig.clone(), amplitude: 1.0, angle: 0.0, min: None, max: None });
        sim.sources.push(Source::PlaneWave { name: "pw2".into(), axis: u, position: -0.6 * um, direction: Direction::Plus, polarization: Component::e(w), signal: sig.clone(), amplitude: 0.7, angle: 0.0, min: None, max: None });
        out.push((format!("{ax}-invariant, column one cell wide"), sim));
        // instantaneous chi2 / chi3 medium driven hard enough to matter
        let mut sim = base(w, pml(), pml());
        let mut kerr = Medium::from_index(1.8).with_name("kerr");
        kerr.chi2 = [1e-10; 3];
        kerr.chi3 = [2e-18; 3];
        sim.media.push(kerr);
        sim.structures.push(Structure { name: "kerr".into(), shape: Shape::Box { min: lift2(w, -0.65 * um, -0.5 * um, -1.0), max: lift2(w, -0.45 * um, 0.5 * um, 1.0) }, medium: "kerr".into() });
        for (c, a, b) in [(Component::e(w), -0.3 * um, -0.3 * um), (Component::e(u), -0.3 * um, 0.35 * um)] {
            sim.sources.push(Source::PointDipole { name: format!("d{}", sim.sources.len()), position: lift2(w, a, b, 0.0), polarization: c, signal: sig.clone(), amplitude: 4e-7 });
        }
        out.push((format!("{ax}-invariant, chi2 / chi3 medium"), sim));
    }
    out
}

/// The 2-D update against the general 3-D update on the same one-cell grid,
/// for every scene of [`two_d_scenes`]: largest difference of the DFT
/// fields relative to the largest field. The two evaluate the same
/// expressions, so the difference is rounding (exactly zero on the CPU
/// backend, where even the order of the operations is the same).
pub fn two_d_matches_one_cell(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("two_d_matches_one_cell");
    // The GPU's 2-D step is part of its fused mode: with the split kernels
    // requested (LIGHTSIM_FUSED=0) or the 2-D update switched off
    // (LIGHTSIM_2D=0) every scene runs on the 3-D update, and the case only
    // checks that the two runs agree.
    let env_off = |k: &str| std::env::var(k).map(|v| v == "0").unwrap_or(false);
    for (name, sim) in two_d_scenes() {
        let a = run(&sim)?;
        let expect_2d = !(env_off("LIGHTSIM_2D") || (env_off("LIGHTSIM_FUSED") && a.stats.backend.starts_with("cuda")));
        if expect_2d && !a.stats.backend.contains("2-D") {
            anyhow::bail!("scene '{name}' did not run on the 2-D update ({})", a.stats.backend);
        }
        let mut forced = sim.clone();
        forced.force_3d_kernels = true;
        let b = run(&forced)?;
        if b.stats.backend.contains("2-D") {
            anyhow::bail!("scene '{name}': force_3d_kernels was ignored ({})", b.stats.backend);
        }
        // normalise by the largest component: an unused set is zero in both
        let peak = Component::ALL
            .iter()
            .map(|c| {
                let f = b.dft["f"].field(*c).unwrap();
                f.re.iter().chain(f.im.iter()).fold(0 as Real, |m, x| m.max(x.abs())) as f64
            })
            .fold(0.0, f64::max);
        if peak == 0.0 {
            anyhow::bail!("scene '{name}': fields are identically zero");
        }
        let mut worst = 0.0f64;
        for c in Component::ALL {
            let (fa, fb) = (a.dft["f"].field(c).unwrap(), b.dft["f"].field(c).unwrap());
            for (x, y) in fa.re.iter().chain(fa.im.iter()).zip(fb.re.iter().chain(fb.im.iter())) {
                worst = worst.max((x - y).abs() as f64 / peak);
            }
        }
        rep.check_abs(&name, worst, 0.0, 1e-4);
        rep.mcells_per_s = a.stats.mcells_per_s;
    }
    Ok(rep)
}

/// Bessel functions J_n(x), n = 0..=nmax, from Bessel's integral
/// (1/pi) int_0^pi cos(n t - x sin t) dt with the trapezoid rule, which
/// converges exponentially on this periodic integrand.
pub fn bessel_j(nmax: usize, x: f64) -> Vec<f64> {
    let m = 64 + 8 * (nmax + x.abs().ceil() as usize);
    (0..=nmax)
        .map(|n| {
            let f = |t: f64| (n as f64 * t - x * t.sin()).cos();
            let mut acc = 0.5 * (f(0.0) + f(PI));
            for k in 1..m {
                acc += f(PI * k as f64 / m as f64);
            }
            acc / m as f64
        })
        .collect()
}

/// Bessel functions of the second kind Y_n(x), n = 0..=nmax (x > 0): Y_0 and
/// Y_1 from the integral representation
/// (1/pi) int_0^pi sin(x sin t - n t) dt - (1/pi) int_0^inf (e^{n s} + (-1)^n e^{-n s}) e^{-x sinh s} ds
/// (Simpson), the rest by upward recurrence.
pub fn bessel_y(nmax: usize, x: f64) -> Vec<f64> {
    let simpson = |f: &dyn Fn(f64) -> f64, a: f64, b: f64, m: usize| -> f64 {
        let h = (b - a) / m as f64;
        let mut acc = f(a) + f(b);
        for k in 1..m {
            acc += f(a + h * k as f64) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        acc * h / 3.0
    };
    let y = |n: i32| -> f64 {
        let nf = n as f64;
        let a = simpson(&|t| (x * t.sin() - nf * t).sin(), 0.0, PI, 4000);
        // the integrand is below 1e-30 once x sinh s - n s > 70
        let mut top = 1.0;
        while x * f64::sinh(top) - nf * top < 70.0 {
            top += 0.5;
        }
        let sg = if n % 2 == 0 { 1.0 } else { -1.0 };
        let b = simpson(&|s| ((nf * s).exp() + sg * (-nf * s).exp()) * (-x * s.sinh()).exp(), 0.0, top, 20000);
        (a - b) / PI
    };
    let mut out = vec![y(0), y(1)];
    for n in 1..nmax {
        let next = 2.0 * n as f64 / x * out[n] - out[n - 1];
        out.push(next);
    }
    out.truncate(nmax + 1);
    out
}

/// Scattering efficiency Q_sca = C_sca / (2 a) of an infinite circular
/// cylinder at normal incidence (2-D Mie series, Bohren & Huffman ch. 8):
/// size parameter `x = k a`, relative index `m` (`None`: perfect
/// conductor), E along the cylinder axis (`e_parallel`, the TM field set of
/// a 2-D simulation) or perpendicular to it (the TE set).
pub fn cylinder_qsca(m: Option<f64>, x: f64, e_parallel: bool) -> f64 {
    let nmax = (x + 4.0 * x.powf(1.0 / 3.0) + 12.0) as usize;
    let d = |z: &[f64], arg: f64, n: usize| -> f64 {
        // Z_n' = Z_{n-1} - (n / x) Z_n ; Z_0' = -Z_1
        if n == 0 { -z[1] } else { z[n - 1] - n as f64 / arg * z[n] }
    };
    let (jx, yx) = (bessel_j(nmax + 1, x), bessel_y(nmax + 1, x));
    let jm = m.map(|m| bessel_j(nmax + 1, m * x));
    let mut q = 0.0;
    for n in 0..=nmax {
        let h = Complex64::new(jx[n], yx[n]);
        let hp = Complex64::new(d(&jx, x, n), d(&yx, x, n));
        let (j, jp) = (jx[n], d(&jx, x, n));
        let c = match (m, &jm) {
            (Some(m), Some(jm)) => {
                let (a, ap) = (jm[n], d(jm, m * x, n));
                if e_parallel {
                    Complex64::new(a * jp - m * ap * j, 0.0) / (hp * a - h * (m * ap))
                } else {
                    Complex64::new(m * a * jp - ap * j, 0.0) / (hp * (m * a) - h * ap)
                }
            }
            _ => {
                if e_parallel {
                    Complex64::new(j, 0.0) / h
                } else {
                    Complex64::new(jp, 0.0) / hp
                }
            }
        };
        q += c.norm_sqr() * if n == 0 { 1.0 } else { 2.0 };
    }
    2.0 / x * q
}

/// A 2-D simulation invariant along z over `[min, max]` of the xy plane,
/// one metre thick (fluxes are then per metre).
pub(super) fn sim_2d(min: [f64; 2], max: [f64; 2], dx: f64, bx: [Boundary; 2], by: [Boundary; 2], run_time: f64, pol: crate::spec::Polarization) -> Simulation {
    let mut b = Boundaries::all_pml();
    b.x = bx;
    b.y = by;
    Simulation::new([min[0], min[1], -0.5], [max[0], max[1], 0.5], MeshSpec::Uniform { dx: [dx; 3] }, b, run_time).two_d(2, pol)
}

pub(super) fn pml2() -> [Boundary; 2] {
    [Boundary::pml(), Boundary::pml()]
}

/// The two field sets of a 2-D simulation: (label, polarization, E
/// component of a wave travelling along x).
pub(super) fn sets_2d() -> [(&'static str, crate::spec::Polarization, Component); 2] {
    [("TE", crate::spec::Polarization::Te, Component::Ey), ("TM", crate::spec::Polarization::Tm, Component::Ez)]
}

/// 2-D: Fresnel reflection of a plane wave at oblique incidence on a glass
/// half-space (Bloch-periodic along y), both polarisations: s (E along the
/// invariant axis, the TM set) and p (E in the plane, the TE set), at 30
/// degrees and at Brewster's angle, where the p reflection vanishes.
pub fn two_d_fresnel_oblique(run: Runner) -> Result<CaseReport> {
    let n2: f64 = 1.5;
    let dx = LAMBDA / 40.0;
    let ly = 8.0 * dx;
    let k0 = TAU * f0() / C0;
    let sig = TimeSignal::gaussian(f0(), 0.06 * f0());
    let mut rep = CaseReport::new("two_d_fresnel_oblique");
    let brewster = n2.atan();
    for (label, pol, comp) in sets_2d() {
        for theta in [30f64.to_radians(), brewster] {
            let ky = k0 * theta.sin();
            let mk = |glass: bool| {
                let mut sim = sim_2d([0.0, 0.0], [4e-6, ly], dx, pml2(), [Boundary::Bloch { k: ky }, Boundary::Bloch { k: ky }], 300e-15, pol);
                if glass {
                    sim.media.push(Medium::from_index(n2).with_name("glass"));
                    sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [2.5e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
                }
                sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: comp, signal: sig.clone(), amplitude: 1.0, angle: theta, min: None, max: None });
                for (name, x) in [("T", 3.2e-6), ("R", 0.5e-6)] {
                    sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, 0.0], max: [x, ly + 1.0, 0.0], freqs: vec![f0()], apodization: Apodization::default() });
                }
                sim
            };
            let vac = run(&mk(false))?;
            let gl = run(&mk(true))?;
            let (ci, tt) = (theta.cos(), (theta.sin() / n2).asin());
            let r_exp = if comp == Component::Ez {
                ((ci - n2 * tt.cos()) / (ci + n2 * tt.cos())).powi(2)
            } else {
                ((n2 * ci - tt.cos()) / (n2 * ci + tt.cos())).powi(2)
            };
            let p_inc = vac.flux["T"][0];
            let deg = theta.to_degrees();
            rep.mcells_per_s = gl.stats.mcells_per_s;
            rep.check_abs(&format!("{label} {deg:.1} deg: |R| without the glass"), (vac.flux["R"][0] / p_inc).abs(), 0.0, 2e-3);
            if r_exp > 1e-6 {
                rep.check(&format!("{label} {deg:.1} deg: R"), -gl.flux["R"][0] / p_inc, r_exp, 0.05);
            } else {
                rep.check_abs(&format!("{label} {deg:.1} deg: R (Brewster)"), -gl.flux["R"][0] / p_inc, 0.0, 2e-3);
            }
            rep.check(&format!("{label} {deg:.1} deg: T"), gl.flux["T"][0] / p_inc, 1.0 - r_exp, 0.01);
        }
    }
    Ok(rep)
}

/// 2-D Mie: scattering cross-section per unit length of a circular cylinder
/// under TFSF illumination against the 2-D Mie series, both polarisations.
fn cylinder_case(run: Runner, name: &str, conductor: bool, tol: f64) -> Result<CaseReport> {
    let n_c: f64 = 2.0;
    let radius = 0.3e-6;
    let dx = if conductor { LAMBDA / 160.0 } else { LAMBDA / (30.0 * n_c) };
    let (l, a, b) = (1.1e-6, 0.55e-6, 0.75e-6);
    let freqs = band(5);
    let mut rep = CaseReport::new(name);
    for (label, pol, comp) in sets_2d() {
        let mut sim = sim_2d([-l, -l], [l, l], dx, pml2(), pml2(), 70e-15, pol);
        let medium = if conductor {
            sim.media.push(Medium::pec().with_name("metal"));
            "metal"
        } else {
            sim.media.push(Medium::from_index(n_c).with_name("rod"));
            "rod"
        };
        sim.structures.push(Structure { name: "cyl".into(), shape: Shape::Cylinder { center: [0.0; 3], radius, length: 10.0, axis: 2 }, medium: medium.into() });
        sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a, -a, 0.0], max: [a, a, 0.0], axis: 0, direction: Direction::Plus, polarization: comp, signal: pulse(), amplitude: 1.0 });
        sim.monitors.push(Monitor::FluxBox { name: "sca".into(), min: [-b, -b, 0.0], max: [b, b, 0.0], freqs: freqs.clone(), apodization: Apodization::default() });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
        for (i, f) in freqs.iter().enumerate() {
            let x = TAU * f / C0 * radius;
            let q = cylinder_qsca(if conductor { None } else { Some(n_c) }, x, comp == Component::Ez);
            // the slab is 1 m thick: the flux is the power per metre
            let q_meas = res.flux["sca"][i] / (0.5 / ETA0 * sp[i].norm_sqr()) / (2.0 * radius);
            rep.check(&format!("{label} Q_sca / Mie @ {:.3} f0 (Q = {q:.3})", f / f0()), q_meas / q, 1.0, tol);
        }
    }
    Ok(rep)
}

pub fn two_d_mie_cylinder(run: Runner) -> Result<CaseReport> {
    cylinder_case(run, "two_d_mie_cylinder", false, 0.02)
}

/// As [`two_d_mie_cylinder`] for a perfectly conducting cylinder. Its
/// surface is staircased, so the error is first order in the cell size
/// (measured: 12 %, 4.4 %, 2.2 %, 1.0 % for E in the plane and 4.0 %, 1.7 %,
/// 0.8 %, 0.4 % for E along the axis at 40, 80, 160, 320 cells per
/// wavelength); the case runs at 160.
pub fn two_d_metal_cylinder(run: Runner) -> Result<CaseReport> {
    cylinder_case(run, "two_d_metal_cylinder", true, 0.03)
}

/// Effective index of the fundamental mode of a symmetric slab waveguide
/// (core `n1` of thickness `d` in a cladding `n2`) at vacuum wavenumber
/// `k0`: E along the invariant axis (`e_parallel`, slab TE) or in the plane
/// (slab TM). Bisection on the even-mode dispersion relation.
pub fn slab_neff(n1: f64, n2: f64, d: f64, k0: f64, e_parallel: bool) -> f64 {
    let f = |neff: f64| -> f64 {
        let kap = k0 * (n1 * n1 - neff * neff).sqrt();
        let gam = k0 * (neff * neff - n2 * n2).sqrt();
        let r = if e_parallel { 1.0 } else { n1 * n1 / (n2 * n2) };
        // kappa tan(kappa d / 2) - r gamma = 0, written without the pole
        kap * (kap * d / 2.0).sin() - r * gam * (kap * d / 2.0).cos()
    };
    // the fundamental mode is the root closest to n1
    let (mut lo, mut hi) = (n2 + 1e-9, n1 - 1e-9);
    let steps = 4000;
    let mut prev = hi;
    for t in 1..=steps {
        let x = n1 - (n1 - n2) * t as f64 / steps as f64;
        if f(x) * f(prev) <= 0.0 {
            lo = x;
            hi = prev;
            break;
        }
        prev = x;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if f(mid) * f(hi) <= 0.0 { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// 2-D: propagation constant of the fundamental guided mode of a slab
/// waveguide, excited by a dipole in the core and read from the phase of the
/// DFT field along the core, against the analytic dispersion relation; both
/// polarisations.
pub fn two_d_slab_mode(run: Runner) -> Result<CaseReport> {
    let (n1, n2, d): (f64, f64, f64) = (2.0, 1.45, 0.3e-6);
    let dx = LAMBDA / (n1 * 32.0);
    let k0 = TAU * f0() / C0;
    let mut rep = CaseReport::new("two_d_slab_mode");
    for (label, pol, comp) in sets_2d() {
        let mut sim = sim_2d([0.0, -1.2e-6], [14e-6, 1.2e-6], dx, pml2(), pml2(), 240e-15, pol);
        sim.media.push(Medium::from_index(n1).with_name("core"));
        sim.media.push(Medium::from_index(n2).with_name("clad"));
        sim.background = "clad".into();
        sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [-5e-6, -d / 2.0, -1.0], max: [30e-6, d / 2.0, 1.0] }, medium: "core".into() });
        sim.sources.push(Source::PointDipole { name: "d".into(), position: [1.0e-6, 0.0, 0.0], polarization: comp, signal: TimeSignal::gaussian(f0(), 0.1 * f0()), amplitude: 1e-12 });
        sim.monitors.push(Monitor::DftField { name: "line".into(), min: [6e-6, 0.0, 0.0], max: [12e-6, 0.0, 0.0], components: vec![comp], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let g = sim.build_grid()?;
        let fld = res.dft["line"].field(comp).unwrap();
        let (mut xs, mut ph) = (vec![], vec![]);
        let mut last = 0.0;
        for i in 0..fld.dims[1] {
            let mut a = fld.at(0, i, 0, 0).arg();
            if i > 0 {
                while a < last - PI {
                    a += TAU;
                }
                while a > last + PI {
                    a -= TAU;
                }
            }
            last = a;
            xs.push(g.position(comp, 0, fld.origin[0] + i));
            ph.push(a);
        }
        let beta = linfit(&xs, &ph).0;
        let exact = slab_neff(n1, n2, d, k0, comp == Component::Ez);
        rep.check(&format!("{label} n_eff (analytic {exact:.4})"), beta / k0, exact, 5e-3);
    }
    Ok(rep)
}

/// 2-D CPML: field of a line source next to the absorber against a run in a
/// domain five times larger, both polarisations.
pub fn two_d_cpml(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let mut rep = CaseReport::new("two_d_cpml");
    for (label, pol, _) in sets_2d() {
        // a line current along the invariant axis: electric (TM) or magnetic (TE)
        let c = if pol == crate::spec::Polarization::Tm { Component::Ez } else { Component::Hz };
        let mk = |half: f64| {
            let mut sim = sim_2d([-half, -half], [half, half], dx, pml2(), pml2(), 30e-15, pol);
            sim.sources.push(Source::PointDipole { name: "d".into(), position: [0.0; 3], polarization: c, signal: TimeSignal::gaussian(f0(), 0.5 * f0()), amplitude: 1e-12 });
            sim.monitors.push(Monitor::Probe { name: "p".into(), position: [0.4e-6, 0.0, 0.0], components: vec![c] });
            sim
        };
        let small = run(&mk(0.5e-6))?;
        let big = run(&mk(2.5e-6))?;
        let (a, b) = (&small.time["p"].fields[0].data, &big.time["p"].fields[0].data);
        let n = a.len().min(b.len());
        let peak = b[..n].iter().fold(0 as Real, |m, v| m.max(v.abs())) as f64;
        let err = a[..n].iter().zip(&b[..n]).fold(0 as Real, |m, (x, y)| m.max((x - y).abs())) as f64;
        rep.mcells_per_s = small.stats.mcells_per_s;
        rep.check_abs(&format!("{label} max |small - big| / peak"), err / peak, 0.0, 1e-2);
    }
    Ok(rep)
}

/// 2-D: diffraction efficiencies of a lamellar grating (Bloch-periodic
/// along y, plane wave at 15 degrees), both polarisations, against rigorous
/// coupled-wave analysis.
///
/// Glass ridges (n = 1.5, half the 1.5 um period, 0.5 um deep) on a glass
/// substrate, wavelength 1 um: three reflected and four transmitted
/// propagating orders. The reference values are those of
/// `python/tests/rcwa_reference.py` (Moharam et al. 1995, Li's factorisation
/// for H along the grooves), converged to the digits given with 161
/// harmonics. The order powers are taken from the Fourier components of the
/// DFT fields of a flux monitor along the period.
pub fn two_d_grating(run: Runner) -> Result<CaseReport> {
    let (n2, period, depth, theta): (f64, f64, f64, f64) = (1.5, 1.5e-6, 0.5e-6, 15f64.to_radians());
    let dx = LAMBDA / 40.0;
    let k0 = TAU * f0() / C0;
    let ky = k0 * theta.sin();
    let sig = TimeSignal::gaussian(f0(), 0.06 * f0());
    // (order, R, T) for E in the plane (TE set) and E along the grooves (TM set)
    let reference: [(&str, [(i32, f64, f64); 4]); 2] = [
        ("TE", [(-2, 0.0, 0.01274), (-1, 0.00013, 0.15352), (0, 0.02574, 0.60902), (1, 0.00247, 0.19638)]),
        ("TM", [(-2, 0.0, 0.03293), (-1, 0.00542, 0.20983), (0, 0.02171, 0.49553), (1, 0.00498, 0.22960)]),
    ];
    let mut rep = CaseReport::new("two_d_grating");
    for ((label, pol, comp), (_, orders)) in sets_2d().into_iter().zip(reference) {
        let mk = |grating: bool| {
            let mut sim = sim_2d([0.0, -period / 2.0], [4.5e-6, period / 2.0], dx, pml2(), [Boundary::Bloch { k: ky }, Boundary::Bloch { k: ky }], 400e-15, pol);
            if grating {
                sim.media.push(Medium::from_index(n2).with_name("glass"));
                sim.structures.push(Structure { name: "substrate".into(), shape: Shape::Box { min: [2.5e-6 + depth, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
                sim.structures.push(Structure { name: "ridge".into(), shape: Shape::Box { min: [2.5e-6, -period / 4.0, -1.0], max: [2.5e-6 + depth, period / 4.0, 1.0] }, medium: "glass".into() });
            }
            sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: comp, signal: sig.clone(), amplitude: 1.0, angle: theta, min: None, max: None });
            for (name, x) in [("T", 3.8e-6), ("R", 0.5e-6)] {
                sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, 0.0], max: [x, 1.0, 0.0], freqs: vec![f0()], apodization: Apodization::default() });
            }
            sim
        };
        // power of order m through a flux plane normal to x, per unit area:
        // S_x = E_y H_z - E_z H_y (the engine's H is eta0 H)
        let order_power = |res: &Results, name: &str, m: i32| -> f64 {
            let f = &res.dft[name].fields;
            // boxes: [E_y, E_z, H_y(p-1), H_z(p-1), H_y(p), H_z(p)]
            let (e, h0, h1, sign) = if comp == Component::Ez { (&f[1], &f[2], &f[4], -1.0) } else { (&f[0], &f[3], &f[5], 1.0) };
            let n = e.dims[2];
            let km = ky + TAU * m as f64 / period;
            let comp_m = |fld: &crate::results::ComplexField| -> Complex64 {
                (0..n).map(|j| fld.at(0, 0, j, 0) * Complex64::from_polar(1.0, -km * j as f64 * dx)).sum::<Complex64>() / n as f64
            };
            let (em, hm) = (comp_m(e), 0.5 * (comp_m(h0) + comp_m(h1)));
            sign * 0.5 * (em * hm.conj()).re / ETA0
        };
        let vac = run(&mk(false))?;
        let gr = run(&mk(true))?;
        rep.mcells_per_s = gr.stats.mcells_per_s;
        let p_inc = order_power(&vac, "T", 0);
        rep.check(&format!("{label} incident power in order 0 / total flux"), p_inc * period / (vac.flux["T"][0]), 1.0, 1e-3);
        let (mut sum_r, mut sum_t) = (0.0, 0.0);
        for (m, r_ref, t_ref) in orders {
            let (r, t) = (-order_power(&gr, "R", m) / p_inc, order_power(&gr, "T", m) / p_inc);
            if r_ref > 0.0 {
                rep.check_abs(&format!("{label} R order {m:+}"), r, r_ref, 5e-3);
            }
            rep.check_abs(&format!("{label} T order {m:+}"), t, t_ref, 8e-3);
            sum_r += r;
            sum_t += t;
        }
        rep.check(&format!("{label} sum of the orders (lossless)"), sum_r + sum_t, 1.0, 0.01);
    }
    Ok(rep)
}

/// 2-D Gaussian (sheet) beam, both polarisations: the power of a beam with
/// its waist on the source plane, 0.5 / eta0 |A S|^2 w0 sqrt(pi / 2) per unit
/// thickness, and a beam focused 4 um behind the source plane: its 1/e
/// field radius at the focus, from the second moment of |E|^2, and its
/// power there.
pub fn two_d_gaussian_beam(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let mut rep = CaseReport::new("two_d_gaussian_beam");
    for (label, pol, comp) in sets_2d() {
        // (a) waist on the source plane, three frequencies
        let w0 = 1.0e-6;
        let mut sim = sim_2d([0.0, -3e-6], [3e-6, 3e-6], dx, pml2(), pml2(), 40e-15, pol);
        sim.sources.push(Source::GaussianBeam { name: "gb".into(), axis: 0, position: 0.5e-6, direction: Direction::Plus, polarization: comp, center: [0.0; 3], waist: w0, signal: pulse(), amplitude: 1.0, focus_distance: 0.0, theta: 0.0, phi: 0.0 });
        let freqs = band(3);
        sim.monitors.push(Monitor::Flux { name: "T".into(), axis: 0, position: 2.0e-6, min: [2.0e-6, -2.9e-6, 0.0], max: [2.0e-6, 2.9e-6, 0.0], freqs: freqs.clone(), apodization: Apodization::default() });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
        for (i, f) in freqs.iter().enumerate() {
            // per metre of the invariant axis: the slab is 1 m thick
            let p_exp = 0.5 / ETA0 * sp[i].norm_sqr() * w0 * (PI / 2.0).sqrt();
            rep.check(&format!("{label} P / P_analytic @ {:.3} f0", f / f0()), res.flux["T"][i] / p_exp, 1.0, 0.02);
        }
        // (b) focused 4 um behind the source plane
        let (w0, dist) = (1.2e-6, 4.0e-6);
        let sig = TimeSignal::gaussian(f0(), 0.1 * f0());
        let mut sim = sim_2d([0.0, -5e-6], [5.5e-6, 5e-6], dx, pml2(), pml2(), 90e-15, pol);
        sim.sources.push(Source::GaussianBeam { name: "gb".into(), axis: 0, position: 0.5e-6, direction: Direction::Plus, polarization: comp, center: [0.0; 3], waist: w0, signal: sig.clone(), amplitude: 1.0, focus_distance: dist, theta: 0.0, phi: 0.0 });
        let xf = 0.5e-6 + dist;
        sim.monitors.push(Monitor::DftField { name: "focus".into(), min: [xf, -4.9e-6, 0.0], max: [xf, 4.9e-6, 0.0], components: vec![comp], freqs: vec![f0()], apodization: Apodization::default(), stride: 1 });
        sim.monitors.push(Monitor::Flux { name: "T".into(), axis: 0, position: xf, min: [xf, -4.9e-6, 0.0], max: [xf, 4.9e-6, 0.0], freqs: vec![f0()], apodization: Apodization::default() });
        let res = run(&sim)?;
        let g = sim.build_grid()?;
        let fld = res.dft["focus"].field(comp).unwrap();
        let (mut m0, mut m2) = (0.0, 0.0);
        for j in 0..fld.dims[2] {
            let y = g.position(comp, 1, fld.origin[1] + j);
            let i2 = fld.at(0, 0, j, 0).norm_sqr();
            m0 += i2;
            m2 += i2 * y * y;
        }
        // |E|^2 ~ exp(-2 y^2 / w^2): <y^2> = w^2 / 4
        rep.check(&format!("{label} radius at the focus / waist"), 2.0 * (m2 / m0).sqrt() / w0, 1.0, 0.03);
        let sp = sig.spectrum(&[f0()], res.dt, res.stats.steps, 0.0);
        let p_exp = 0.5 / ETA0 * sp[0].norm_sqr() * w0 * (PI / 2.0).sqrt();
        rep.check(&format!("{label} focused beam power / analytic"), res.flux["T"][0] / p_exp, 1.0, 0.03);
    }
    Ok(rep)
}
