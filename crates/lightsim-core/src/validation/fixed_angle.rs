//! Broadband oblique incidence at a fixed angle (`Simulation::fixed_angle`).

use super::*;

/// Reflection and transmission of a slab of relative permittivity `eps`
/// (thickness `d`) between vacuum half-spaces for a plane wave at polar angle
/// `theta` (s or p polarization), from the Airy sum with
/// `k_z = k0 sqrt(eps - sin^2 theta)`.
pub fn slab_rt_oblique(eps: Complex64, k0: f64, d: f64, theta: f64, p_pol: bool) -> (f64, f64) {
    let one = Complex64::new(1.0, 0.0);
    let s2 = theta.sin().powi(2);
    let k1 = Complex64::new(theta.cos(), 0.0);
    let mut k2 = (eps - s2).sqrt();
    if k2.im < 0.0 {
        k2 = -k2;
    }
    // interface coefficients (s: E amplitudes, p: H amplitudes)
    let (r12, t12, t21) = if p_pol {
        ((eps * k1 - k2) / (eps * k1 + k2), 2.0 * eps * k1 / (eps * k1 + k2), 2.0 * k2 / (k2 + eps * k1))
    } else {
        ((k1 - k2) / (k1 + k2), 2.0 * k1 / (k1 + k2), 2.0 * k2 / (k2 + k1))
    };
    let r21 = -r12;
    let ph = (Complex64::i() * k2 * k0 * d).exp();
    let den = one - r21 * r21 * ph * ph;
    let r = r12 + t12 * t21 * r21 * ph * ph / den;
    let t = t12 * t21 * ph / den;
    (r.norm_sqr(), t.norm_sqr())
}

/// The 2-D scene of the fixed-angle cases: propagation along x (PML), tilt
/// along y (periodic, `ny` cells of `dx`), invariant z. `pol` is `Ez` (s) or
/// `Hz` (p); monitors R and T (flux) at `freqs`.
fn fa_scene(dx: f64, ny: usize, theta: f64, pol: Component, freqs: &[f64], fwidth: f64, run_time: f64) -> Simulation {
    let ly = ny as f64 * dx;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0, 0.0, 0.0], [4.0e-6, ly, dx], MeshSpec::Uniform { dx: [dx; 3] }, b, run_time).two_d(2, crate::spec::Polarization::Both);
    sim.fixed_angle = true;
    sim.shutoff = Some(1e-7);
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f0(), fwidth), amplitude: 1.0, angle: theta, min: None, max: None });
    for (nm, x) in [("R", 0.6e-6), ("T", 3.2e-6)] {
        sim.monitors.push(Monitor::Flux { name: nm.into(), axis: 0, position: x, min: [x, -1.0, -1.0], max: [x, 1.0, 1.0], freqs: freqs.to_vec(), apodization: Apodization::default() });
    }
    sim
}

/// Incident power of a fixed-angle plane wave of unit amplitude through the
/// area `area`: `|S|^2 cos(theta) area / (2 eta0)` per frequency.
fn fa_incident(res: &Results, freqs: &[f64], fwidth: f64, theta: f64, area: f64) -> Vec<f64> {
    spectrum_power(&TimeSignal::gaussian(f0(), fwidth), freqs, res.dt, res.stats.steps).iter().map(|p| 0.5 / ETA0 * p * theta.cos() * area).collect()
}

fn fa_band(n: usize, rel: f64) -> Vec<f64> {
    (0..n).map(|i| f0() * (1.0 - rel + 2.0 * rel * i as f64 / (n - 1) as f64)).collect()
}

/// A dielectric slab (n = 1.5, 0.6 um) at 30 and 60 degrees, s and p, one
/// fixed-angle run each: R and T at 13 frequencies over +-30 % against the
/// Airy formula at that angle; R + T = 1; the incident power of the vacuum
/// run against |S|^2 cos(theta) / (2 eta0).
pub fn fixed_angle_slab(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 40.0;
    let ny = 8;
    let (n2, d) = (1.5f64, 0.6e-6);
    let freqs = fa_band(13, 0.3);
    let fw = 0.25 * f0();
    let mut rep = CaseReport::new("fixed_angle_slab");
    for deg in [30.0f64, 60.0] {
        let th = deg.to_radians();
        for (pname, pol) in [("s", Component::Ez), ("p", Component::Hz)] {
            let mut sim = fa_scene(dx, ny, th, pol, &freqs, fw, 300e-15);
            let vac = run(&sim)?;
            sim.media.push(Medium::from_index(n2).with_name("glass"));
            sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.0e-6 + d, 1.0, 1.0] }, medium: "glass".into() });
            let res = run(&sim)?;
            rep.mcells_per_s = res.stats.mcells_per_s;
            let area = ny as f64 * dx * dx;
            let inc: Vec<f64> = vac.flux["T"].clone();
            let inc_v = fa_incident(&vac, &freqs, fw, th, area);
            let (mut er, mut et, mut ec, mut ev, mut erv) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (i, f) in freqs.iter().enumerate() {
                let (r_exp, t_exp) = slab_rt_oblique(Complex64::new(n2 * n2, 0.0), TAU * f / C0, d, th, pname == "p");
                let (r, t) = (-res.flux["R"][i] / inc[i], res.flux["T"][i] / inc[i]);
                er = er.max((r - r_exp).abs());
                et = et.max((t - t_exp).abs());
                ec = ec.max((r + t - 1.0).abs());
                ev = ev.max((vac.flux["T"][i] / inc_v[i] - 1.0).abs());
                erv = erv.max((vac.flux["R"][i] / inc_v[i]).abs());
            }
            let tol = if deg < 45.0 { 6e-3 } else { 1.5e-2 };
            rep.check_abs(&format!("{deg:.0} deg {pname}: max |R - R_Airy| over band"), er, 0.0, tol);
            rep.check_abs(&format!("{deg:.0} deg {pname}: max |T - T_Airy| over band"), et, 0.0, tol);
            rep.check_abs(&format!("{deg:.0} deg {pname}: max |R + T - 1|"), ec, 0.0, 1e-3);
            rep.check_abs(&format!("{deg:.0} deg {pname}: vacuum T / (|S|^2 cos / 2 eta0) - 1"), ev, 0.0, 6e-3);
            rep.check_abs(&format!("{deg:.0} deg {pname}: vacuum |R| (backward leakage)"), erv, 0.0, 1e-6);
            if pname == "s" {
                // the same slab from narrow-band Bloch runs (k_y = k0(f) sin theta) at three frequencies
                let mut eb = 0f64;
                for f in [0.7 * f0(), f0(), 1.3 * f0()] {
                    let mkb = |slab: bool| {
                        let mut b = Boundaries::all(Boundary::Periodic);
                        b.x = [Boundary::pml(), Boundary::pml()];
                        let k = TAU * f / C0 * th.sin();
                        b.y = [Boundary::Bloch { k }, Boundary::Bloch { k }];
                        let mut sb = Simulation::new([0.0, 0.0, 0.0], [4.0e-6, ny as f64 * dx, dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 600e-15).two_d(2, crate::spec::Polarization::Both);
                        sb.shutoff = Some(1e-7);
                        sb.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f, 0.04 * f), amplitude: 1.0, angle: th, min: None, max: None });
                        for (nm, x) in [("R", 0.6e-6), ("T", 3.2e-6)] {
                            sb.monitors.push(Monitor::Flux { name: nm.into(), axis: 0, position: x, min: [x, -1.0, -1.0], max: [x, 1.0, 1.0], freqs: vec![f], apodization: Apodization::default() });
                        }
                        if slab {
                            sb.media.push(Medium::from_index(n2).with_name("glass"));
                            sb.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.0e-6 + d, 1.0, 1.0] }, medium: "glass".into() });
                        }
                        sb
                    };
                    let (bv, bs) = (run(&mkb(false))?, run(&mkb(true))?);
                    let (r_exp, _) = slab_rt_oblique(Complex64::new(n2 * n2, 0.0), TAU * f / C0, d, th, false);
                    eb = eb.max((-bs.flux["R"][0] / bv.flux["T"][0] - r_exp).abs());
                }
                rep.check_abs(&format!("{deg:.0} deg s: Bloch runs at 0.7, 1, 1.3 f0: max |R - R_Airy|"), eb, 0.0, tol);
            }
        }
    }
    Ok(rep)
}

/// p polarization at Brewster's angle of an n = 2 half-space, one
/// fixed-angle run over +-30 %: R stays at the level of the discretisation
/// error at every frequency (a run with a fixed transverse wavevector is at
/// Brewster's angle at its centre frequency only).
pub fn fixed_angle_brewster(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 40.0;
    let n2 = 2.0f64;
    let th = n2.atan();
    let freqs = fa_band(13, 0.3);
    let fw = 0.25 * f0();
    let mut sim = fa_scene(dx, 8, th, Component::Hz, &freqs, fw, 300e-15);
    let vac = run(&sim)?;
    sim.media.push(Medium::from_index(n2).with_name("glass"));
    sim.structures.push(Structure { name: "half".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
    let res = run(&sim)?;
    let mut rep = CaseReport::new("fixed_angle_brewster");
    rep.mcells_per_s = res.stats.mcells_per_s;
    let (mut rmax, mut tdev) = (0f64, 0f64);
    for i in 0..freqs.len() {
        let inc = vac.flux["T"][i];
        rmax = rmax.max((res.flux["R"][i] / inc).abs());
        tdev = tdev.max((res.flux["T"][i] / inc - 1.0).abs());
    }
    rep.check_abs("max |R| over +-30 % at Brewster's angle (63.4 deg)", rmax, 0.0, 2e-4);
    rep.check_abs("max |T - 1| over +-30 %", tdev, 0.0, 2e-4);
    Ok(rep)
}

/// Dispersive slabs at 45 degrees, s and p, one fixed-angle run each: a
/// Lorentz slab (eps_inf 1.5, delta_eps 1.5, f_0 = 1.6 f0, gamma = 0.1 f0,
/// 0.4 um) and a Drude film (f_p = 4 f0, gamma = 0.05 f0, 30 nm, dx = 5 nm),
/// R and T over +-30 % against the Airy formula with the permittivity of the
/// scheme's (trapezoidal) pole update.
pub fn fixed_angle_dispersive(run: Runner) -> Result<CaseReport> {
    let th = 45f64.to_radians();
    let freqs = fa_band(7, 0.3);
    let fw = 0.25 * f0();
    let mut rep = CaseReport::new("fixed_angle_dispersive");
    let cases: [(&str, f64, f64, f64, Pole); 2] = [
        ("Lorentz", LAMBDA / 60.0, 0.4e-6, 1.5, Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.6 * f0(), gamma: TAU * 0.1 * f0() }),
        ("Drude", LAMBDA / 200.0, 30e-9, 1.0, Pole::Drude { omega_p: TAU * 4.0 * f0(), gamma: TAU * 0.05 * f0() }),
    ];
    for (label, dx, d, eps_inf, pole) in cases {
        for (pname, pol) in [("s", Component::Ez), ("p", Component::Hz)] {
            let mut sim = fa_scene(dx, 4, th, pol, &freqs, fw, 300e-15);
            let vac = run(&sim)?;
            sim.media.push(Medium::dielectric(eps_inf).with_name("m").with_poles(vec![pole.clone()]));
            sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [2.0e-6, -1.0, -1.0], max: [2.0e-6 + d, 1.0, 1.0] }, medium: "m".into() });
            let res = run(&sim)?;
            rep.mcells_per_s = res.stats.mcells_per_s;
            let (mut er, mut et) = (0f64, 0f64);
            for (i, f) in freqs.iter().enumerate() {
                let wv = TAU * f;
                let eps = Complex64::new(eps_inf, 0.0) + crate::engine::fixed_angle::chi_trapezoid(&pole, wv, res.dt);
                let (r_exp, t_exp) = slab_rt_oblique(eps, wv / C0, d, th, pname == "p");
                let inc = vac.flux["T"][i];
                er = er.max((-res.flux["R"][i] / inc - r_exp).abs());
                et = et.max((res.flux["T"][i] / inc - t_exp).abs());
            }
            let tol = if label == "Drude" { 3e-3 } else { 1.2e-2 };
            rep.check_abs(&format!("{label} {pname}: max |R - R_Airy| over band"), er, 0.0, tol);
            rep.check_abs(&format!("{label} {pname}: max |T - T_Airy| over band"), et, 0.0, tol);
        }
    }
    Ok(rep)
}

/// Power in diffraction order `m` through a flux plane normal to x of a 2-D
/// scene periodic along y with `ny` cells of `dx` (physical DFT fields),
/// with the transverse wavevector `k_inc` of the incident wave.
fn fa_order_power(res: &Results, name: &str, fi: usize, m: i32, k_inc: f64, period: f64, dx: f64, s_pol: bool) -> f64 {
    let f = &res.dft[name].fields;
    // boxes: [E_y, E_z, H_y(p-1), H_z(p-1), H_y(p), H_z(p)]
    let (e, h0, h1, sign) = if s_pol { (&f[1], &f[2], &f[4], -1.0) } else { (&f[0], &f[3], &f[5], 1.0) };
    let n = e.dims[2];
    let km = k_inc + TAU * m as f64 / period;
    // E_z sits on y nodes, E_y on y half nodes (and H the other way round)
    let off = |c: &crate::results::ComplexField| if c.comp == "Ey" || c.comp == "Hz" { 0.5 } else { 0.0 };
    let comp_m = |fld: &crate::results::ComplexField| -> Complex64 {
        let o = off(fld);
        (0..n).map(|j| fld.at(fi, 0, j, 0) * Complex64::from_polar(1.0, -km * (j as f64 + o) * dx)).sum::<Complex64>() / n as f64
    };
    let (em, hm) = (comp_m(e), 0.5 * (comp_m(h0) + comp_m(h1)));
    sign * 0.5 * (em * hm.conj()).re / ETA0
}

/// A lamellar grating (n = 1.5 ridges of half the 1.5 um period, 0.5 um
/// deep, on n = 1.5) at 15 degrees, both polarizations: diffraction
/// efficiencies over +-25 % from one fixed-angle run against narrow-band
/// Bloch runs (k_y = k0(f) sin 15 deg, one per frequency) at 0.8, 0.95,
/// 1.1 and 1.2 f0. Orders appear in air at 0.90 f0 (+1) and 1.06 f0 (-2).
pub fn fixed_angle_grating(run: Runner) -> Result<CaseReport> {
    let (n2, period, depth): (f64, f64, f64) = (1.5, 1.5e-6, 0.5e-6);
    let th = 15f64.to_radians();
    let dx = LAMBDA / 40.0;
    let check_f = [0.8, 0.95, 1.1, 1.2];
    let freqs: Vec<f64> = check_f.iter().map(|x| x * f0()).collect();
    let mut rep = CaseReport::new("fixed_angle_grating");
    let structures = |sim: &mut Simulation| {
        sim.media.push(Medium::from_index(n2).with_name("glass"));
        sim.structures.push(Structure { name: "substrate".into(), shape: Shape::Box { min: [2.5e-6 + depth, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
        sim.structures.push(Structure { name: "ridge".into(), shape: Shape::Box { min: [2.5e-6, -period / 4.0, -1.0], max: [2.5e-6 + depth, period / 4.0, 1.0] }, medium: "glass".into() });
    };
    let dft = |sim: &mut Simulation, fr: &[f64]| {
        for (name, x) in [("T", 3.8e-6), ("R", 0.5e-6)] {
            sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, 0.0], max: [x, 1.0, 0.0], freqs: fr.to_vec(), apodization: Apodization::default() });
        }
    };
    for (label, pol) in [("s", Component::Ez), ("p", Component::Hz)] {
        let s_pol = pol == Component::Ez;
        // one broadband fixed-angle run (and its empty reference)
        let mk = |grating: bool| {
            let mut b = Boundaries::all(Boundary::Periodic);
            b.x = [Boundary::pml(), Boundary::pml()];
            let mut sim = Simulation::new([0.0, -period / 2.0, 0.0], [4.5e-6, period / 2.0, dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 600e-15).two_d(2, crate::spec::Polarization::Both);
            sim.fixed_angle = true;
            sim.shutoff = Some(1e-7);
            if grating {
                structures(&mut sim);
            }
            sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f0(), 0.25 * f0()), amplitude: 1.0, angle: th, min: None, max: None });
            dft(&mut sim, &freqs);
            sim
        };
        let vac = run(&mk(false))?;
        let gr = run(&mk(true))?;
        rep.mcells_per_s = gr.stats.mcells_per_s;
        let mut worst = 0f64;
        for (fi, &f) in freqs.iter().enumerate() {
            let k_inc = TAU * f / C0 * th.sin();
            // the Bloch reference at this frequency
            let mkb = |grating: bool| {
                let mut b = Boundaries::all(Boundary::Periodic);
                b.x = [Boundary::pml(), Boundary::pml()];
                b.y = [Boundary::Bloch { k: k_inc }, Boundary::Bloch { k: k_inc }];
                let mut sim = Simulation::new([0.0, -period / 2.0, 0.0], [4.5e-6, period / 2.0, dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 800e-15).two_d(2, crate::spec::Polarization::Both);
                sim.shutoff = Some(1e-7);
                if grating {
                    structures(&mut sim);
                }
                sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f, 0.04 * f), amplitude: 1.0, angle: th, min: None, max: None });
                dft(&mut sim, &[f]);
                sim
            };
            let bv = run(&mkb(false))?;
            let bg = run(&mkb(true))?;
            let p_fa = fa_order_power(&vac, "T", fi, 0, k_inc, period, dx, s_pol);
            let p_bl = fa_order_power(&bv, "T", 0, 0, k_inc, period, dx, s_pol);
            let k = TAU * f / C0;
            let (mut sum_fa, mut sum_bl) = (0.0, 0.0);
            for m in -3i32..=3 {
                let kt = k_inc + TAU * m as f64 / period;
                for (mon, nn, sgn) in [("R", 1.0, -1.0), ("T", n2, 1.0)] {
                    if kt.abs() >= nn * k {
                        continue;
                    }
                    let e_fa = sgn * fa_order_power(&gr, mon, fi, m, k_inc, period, dx, s_pol) / p_fa;
                    let e_bl = sgn * fa_order_power(&bg, mon, 0, m, k_inc, period, dx, s_pol) / p_bl;
                    sum_fa += e_fa;
                    sum_bl += e_bl;
                    let grazing = (kt / (nn * k)).asin().to_degrees();
                    rep.check_abs(&format!("{label} {:.2} f0 {mon} order {m:+} ({grazing:+.0} deg): fixed angle vs Bloch", f / f0()), e_fa, e_bl, 1e-3);
                    worst = worst.max((e_fa - e_bl).abs());
                }
            }
            rep.check_abs(&format!("{label} {:.2} f0: sum of orders, fixed angle", f / f0()), sum_fa, 1.0, 1e-3);
            rep.check_abs(&format!("{label} {:.2} f0: sum of orders, Bloch", f / f0()), sum_bl, 1.0, 1e-3);
        }
        let _ = worst;
    }
    Ok(rep)
}

/// Late-time stability at the largest angle (80 deg, p and s) with a
/// dielectric slab, a Drude film and a Lorentz layer behind PML: a run ten
/// times longer than the pulse needs (no shutoff) stays bounded, and its
/// energy at the end is far below its peak.
pub fn fixed_angle_stability(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let th = 80f64.to_radians();
    let mut rep = CaseReport::new("fixed_angle_stability");
    for (pname, pol) in [("p", Component::Hz), ("s", Component::Ez)] {
        let mut sim = fa_scene(dx, 6, th, pol, &[f0()], 0.25 * f0(), 1000e-15);
        sim.shutoff = None;
        sim.shutoff_interval = 500;
        sim.media.push(Medium::from_index(2.0).with_name("glass"));
        sim.media.push(Medium::dielectric(1.0).with_name("drude").with_poles(vec![Pole::Drude { omega_p: TAU * 3.0 * f0(), gamma: TAU * 0.02 * f0() }]));
        sim.media.push(Medium::dielectric(2.0).with_name("lor").with_poles(vec![Pole::Lorentz { delta_eps: 2.0, omega0: TAU * 1.2 * f0(), gamma: TAU * 0.01 * f0() }]));
        sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [1.6e-6, -1.0, -1.0], max: [2.0e-6, 1.0, 1.0] }, medium: "glass".into() });
        sim.structures.push(Structure { name: "d".into(), shape: Shape::Box { min: [2.2e-6, -1.0, -1.0], max: [2.25e-6, 1.0, 1.0] }, medium: "drude".into() });
        sim.structures.push(Structure { name: "l".into(), shape: Shape::Box { min: [2.6e-6, -1.0, -1.0], max: [3.0e-6, 1.0, 1.0] }, medium: "lor".into() });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let e = &res.stats.energy;
        let peak = e.iter().cloned().fold(0.0, f64::max);
        let last = *e.last().unwrap();
        let tail_max = e[e.len() / 2..].iter().cloned().fold(0.0, f64::max);
        rep.check_abs(&format!("{pname}: steps run"), res.stats.steps as f64, res.nsteps as f64, 0.0);
        rep.check_abs(&format!("{pname}: energy at the end / peak"), last / peak, 0.0, 1e-6);
        rep.check_abs(&format!("{pname}: max energy over the second half / peak"), tail_max / peak, 0.0, 1e-6);
    }
    Ok(rep)
}

/// 3-D, propagation along z, 45 degrees, s and p: (a) tilt along x
/// ((u, v, w) = (x, y, z), cyclic) on an n = 1.5 slab: R, T over +-30 %
/// against Airy and the transmitted field (amplitude and phase) at f0
/// against a Bloch run at f0; (b) tilt along y ((y, x, z), the other
/// handedness) on a Lorentz slab: R, T against Airy with the scheme's
/// permittivity (identical to the tilt-x result: the sign of the coupling
/// follows the handedness), and the energy left in the empty run once the
/// pulse has passed (a wrong-signed coupling leaves a static field at the
/// source plane).
pub fn fixed_angle_3d(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 30.0;
    let th = 45f64.to_radians();
    let (n2, d) = (1.5f64, 0.5e-6);
    let freqs = fa_band(7, 0.3);
    let k0 = TAU * f0() / C0;
    let lor = Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.6 * f0(), gamma: TAU * 0.1 * f0() };
    let mut rep = CaseReport::new("fixed_angle_3d");
    // (tilt axis, s polarization, p polarization, dispersive slab)
    for (u, s_pol, p_pol, disp) in [(0usize, Component::Ey, Component::Hy, false), (1, Component::Ex, Component::Hx, true)] {
        let tag = if u == 0 { "tilt x" } else { "tilt y" };
        for (pname, pol) in [("s", s_pol), ("p", p_pol)] {
            let mk = |fixed: bool, slab: bool| {
                let mut b = Boundaries::all(Boundary::Periodic);
                b.z = [Boundary::pml(), Boundary::pml()];
                if !fixed {
                    *b.axis_mut(u) = [Boundary::Bloch { k: k0 * th.sin() }, Boundary::Bloch { k: k0 * th.sin() }];
                }
                let sig = if fixed { TimeSignal::gaussian(f0(), 0.25 * f0()) } else { TimeSignal::gaussian(f0(), 0.05 * f0()) };
                let mut ext = [2.0 * dx, 2.0 * dx, 4.0e-6];
                ext[u] = 6.0 * dx;
                let mut sim = Simulation::new([0.0; 3], ext, MeshSpec::Uniform { dx: [dx; 3] }, b, 500e-15);
                sim.fixed_angle = fixed;
                sim.shutoff = Some(1e-9);
                if slab {
                    let m = if disp { Medium::dielectric(1.5).with_name("m").with_poles(vec![lor.clone()]) } else { Medium::from_index(n2).with_name("m") };
                    sim.media.push(m);
                    sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [-1.0, -1.0, 2.0e-6], max: [1.0, 1.0, 2.0e-6 + d] }, medium: "m".into() });
                }
                sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 1.0e-6, direction: Direction::Plus, polarization: pol, signal: sig, amplitude: 1.0, angle: th, min: None, max: None });
                let fr = if fixed { freqs.clone() } else { vec![f0()] };
                for (nm, z) in [("R", 0.6e-6), ("T", 3.2e-6)] {
                    sim.monitors.push(Monitor::Flux { name: nm.into(), axis: 2, position: z, min: [-1.0, -1.0, z], max: [1.0, 1.0, z], freqs: fr.clone(), apodization: Apodization::default() });
                }
                sim.monitors.push(Monitor::DftField { name: "pt".into(), min: [dx, dx, 3.0e-6], max: [dx, dx, 3.0e-6], components: vec![Component::Ex, Component::Ey], freqs: fr, apodization: Apodization::default(), stride: 1 });
                sim
            };
            let vac = run(&mk(true, false))?;
            let res = run(&mk(true, true))?;
            rep.mcells_per_s = res.stats.mcells_per_s;
            let (mut er, mut et) = (0f64, 0f64);
            for (i, f) in freqs.iter().enumerate() {
                let wv = TAU * f;
                let eps = if disp { Complex64::new(1.5, 0.0) + crate::engine::fixed_angle::chi_trapezoid(&lor, wv, res.dt) } else { Complex64::new(n2 * n2, 0.0) };
                let (r_exp, t_exp) = slab_rt_oblique(eps, wv / C0, d, th, pname == "p");
                let inc = vac.flux["T"][i];
                er = er.max((-res.flux["R"][i] / inc - r_exp).abs());
                et = et.max((res.flux["T"][i] / inc - t_exp).abs());
            }
            // (the Lorentz slab has n up to 2.4 at 1.3 f0: 12 cells per wavelength
            // in it; the 2-D case fixed_angle_dispersive resolves it 5 times finer)
            let tol = if disp { 4e-2 } else if pname == "s" { 2.5e-2 } else { 1e-2 };
            rep.check_abs(&format!("{tag} {pname}: max |R - R_Airy| over band"), er, 0.0, tol);
            rep.check_abs(&format!("{tag} {pname}: max |T - T_Airy| over band"), et, 0.0, tol);
            if disp {
                let e = &vac.stats.energy;
                let peak = e.iter().cloned().fold(0.0, f64::max);
                rep.check_abs(&format!("{tag} {pname}: energy left in the empty run / peak"), e.last().unwrap() / peak, 0.0, 1e-8);
                continue;
            }
            // transmitted field at f0, normalised by the source spectrum, against Bloch
            let bl = run(&mk(false, true))?;
            let fi = 3; // f0
            let ci = if pname == "s" { 1 - u } else { u };
            let spec = |r: &Results| -> Complex64 {
                let (fr, re, im) = &r.source_spectra["pw"];
                let k = fr.iter().position(|&f| (f - f0()).abs() < 1e-6 * f0()).unwrap();
                Complex64::new(re[k], im[k])
            };
            let a = res.dft["pt"].fields[ci].at(fi, 0, 0, 0) / spec(&res);
            let b = bl.dft["pt"].fields[ci].at(0, 0, 0, 0) / spec(&bl);
            rep.check_abs(&format!("{tag} {pname}: |E_t| fixed angle / Bloch - 1 at f0"), (a.norm() / b.norm() - 1.0).abs(), 0.0, 1e-3);
            rep.check_abs(&format!("{tag} {pname}: phase of E_t, fixed angle - Bloch (rad)"), (a / b).arg().abs(), 0.0, 5e-3);
        }
    }
    Ok(rep)
}

/// A fixed-angle scene propagating along x (PML) with a glass slab and a
/// Drude film, 50 degrees, DFT fields over the whole cell: 2-D (tilt y, p
/// polarization) or 3-D (tilt z, s polarization, the other handedness).
/// Used to compare engines and decompositions.
pub fn fixed_angle_scene(three_d: bool) -> Simulation {
    let dx = 20e-9;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [2.4e-6, 6.0 * dx, 6.0 * dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 60e-15);
    if !three_d {
        sim = sim.two_d(2, crate::spec::Polarization::Both);
    }
    sim.fixed_angle = true;
    sim.media.push(Medium::from_index(1.5).with_name("glass"));
    sim.media.push(Medium::dielectric(1.0).with_name("drude").with_poles(vec![Pole::Drude { omega_p: TAU * 2.0 * f0(), gamma: 1e13 }]));
    sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [1.0e-6, -1.0, -1.0], max: [1.4e-6, 1.0, 1.0] }, medium: "glass".into() });
    sim.structures.push(Structure { name: "d".into(), shape: Shape::Box { min: [1.6e-6, -1.0, -1.0], max: [1.64e-6, 1.0, 1.0] }, medium: "drude".into() });
    let pol = if three_d { Component::Ey } else { Component::Hz };
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 0.5e-6, direction: Direction::Plus, polarization: pol, signal: TimeSignal::gaussian(f0(), 0.3 * f0()), amplitude: 1.0, angle: 50f64.to_radians(), min: None, max: None });
    let freqs = vec![0.8 * f0(), f0(), 1.2 * f0()];
    for (nm, x) in [("R", 0.3e-6), ("T", 2.0e-6)] {
        sim.monitors.push(Monitor::Flux { name: nm.into(), axis: 0, position: x, min: [x, -1.0, -1.0], max: [x, 1.0, 1.0], freqs: freqs.clone(), apodization: Apodization::default() });
    }
    sim.monitors.push(Monitor::DftField { name: "f".into(), min: [0.2e-6, -1.0, -1.0], max: [2.2e-6, 1.0, 1.0], components: Component::ALL.to_vec(), freqs, apodization: Apodization::default(), stride: 1 });
    sim
}
