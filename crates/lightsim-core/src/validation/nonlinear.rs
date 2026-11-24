//! Nonlinear media: chi2, chi3 (Kerr) and chi3 with a Raman response.

use super::*;

/// All fast cases (used by the test-suite and the `lightsim validate` CLI).
/// Instantaneous chi(m) slab (m = 2: second-harmonic generation, m = 3: Kerr
/// self-phase modulation + third harmonic). In a lossless, dispersionless,
/// index-matched slab the forward-wave equation gives, to first order in chi,
///   E_gen(t) = -(chi L / 2c) d/dt [E_lin(t)^m]
/// at the output (undepleted pump, phase matched). We compare the spectrum of
/// E_nl - E_lin at a probe behind the slab with the spectrum of this
/// prediction built from the linear run's own probe signal.
fn chi_slab(run: Runner, m: u32) -> Result<CaseReport> {
    let dx = LAMBDA / if m == 2 { 60.0 } else { 80.0 };
    let (x0, len) = (1.5e-6, 2.0e-6);
    let k_l = TAU / LAMBDA * len;
    // generated / fundamental ~ 1 %
    let chi = if m == 2 { 0.01 / k_l } else { 0.01 / (1.5 * k_l) };
    let mk = |nonlinear: bool| {
        let mut sim = column(dx, 5.0e-6, 2);
        let mut med = Medium::dielectric(1.0).with_name("nl");
        if nonlinear {
            if m == 2 {
                med.chi2 = [chi; 3];
            } else {
                med.chi3 = [chi; 3];
            }
        }
        sim.media.push(med);
        sim.structures.push(Structure {
            name: "slab".into(),
            shape: Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + len, 1.0, 1.0] },
            medium: "nl".into(),
        });
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(Monitor::Probe { name: "p".into(), position: [4.0e-6, 0.0, 0.0], components: vec![Component::Ey] });
        sim.run_time = 45e-15;
        sim
    };
    let lin = run(&mk(false))?;
    let nl = run(&mk(true))?;
    let a: Vec<f64> = lin.time["p"].fields[0].data.iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = nl.time["p"].fields[0].data.iter().map(|&v| v as f64).collect();
    let n = a.len().min(b.len());
    let dt = lin.dt;
    let egen: Vec<f64> = (0..n).map(|i| b[i] - a[i]).collect();
    let pw: Vec<f64> = a[..n].iter().map(|v| v.powi(m as i32)).collect();
    let mut pred = vec![0.0; n];
    for i in 1..n - 1 {
        pred[i] = -chi * len / (2.0 * C0) * (pw[i + 1] - pw[i - 1]) / (2.0 * dt);
    }
    let dft = |x: &[f64], f: f64| -> Complex64 {
        x.iter().enumerate().fold(Complex64::new(0.0, 0.0), |acc, (i, v)| {
            acc + Complex64::from_polar(*v, -TAU * f * i as f64 * dt)
        })
    };
    let peak_lin = a[..n].iter().fold(0f64, |mx, v| mx.max(v.abs()));
    let peak_gen = egen.iter().fold(0f64, |mx, v| mx.max(v.abs()));
    let name = if m == 2 { "chi2_second_harmonic" } else { "chi3_kerr" };
    let mut rep = CaseReport::new(name);
    rep.mcells_per_s = nl.stats.mcells_per_s;
    let fr: Vec<f64> = if m == 2 { vec![1.8, 2.0, 2.2] } else { vec![0.9, 1.0, 1.1, 2.8, 3.0, 3.2] };
    for r in fr {
        let f = r * f0();
        let g = dft(&egen, f).norm();
        let p = dft(&pred, f).norm();
        rep.check(&format!("|E_gen| / theory @ {r:.1} f0"), g / p, 1.0, 0.03);
    }
    rep.check_abs("generated / fundamental peak", peak_gen / peak_lin, 0.01, 0.01);
    Ok(rep)
}

pub fn chi2_second_harmonic(run: Runner) -> Result<CaseReport> {
    chi_slab(run, 2)
}

pub fn chi3_kerr(run: Runner) -> Result<CaseReport> {
    chi_slab(run, 3)
}

/// chi3 with a Raman response (`Medium::raman`).
///
/// (1) Response to a known pulse: a thin index-matched slab, as in
/// `chi3_kerr`, radiates to first order E_gen(t) = -(L / 2c) d/dt P_NL/eps0
/// with P_NL/eps0 = chi3 E [(1 - f) E^2 + f (g_R * E^2)]. The convolution is
/// evaluated from the linear run's probe signal with the analytic g_R(t)
/// (trapezoidal rule) and compared with E_nl - E_lin at three frequencies,
/// for a purely delayed response (f = 1) and a mixed one (f = 0.5, which also
/// has a third harmonic).
///
/// (2) Stimulated Raman gain: a strong CW pump at w_p and a weak CW Stokes
/// probe at w_s = w_p - Omega_R, co-polarised, in an index-matched slab of
/// length L. The probe amplitude grows by exp(g L / 2) with the coupled-wave
/// intensity gain coefficient g = (w_s / c) (f chi3 / 2) A_p^2 Im g_R(w_p - w_s)
/// (g_R of the discrete update). The probe is isolated by subtracting a
/// pump-only run and normalised by a probe-only run; g L = 0.02 keeps the
/// anti-Stokes feedback (phase-matched in a dispersionless medium, relative
/// size ~ g L / 4) at 0.5 %.
pub fn chi3_raman(run: Runner) -> Result<CaseReport> {
    use crate::materials::Raman;
    let mut rep = CaseReport::new("chi3_raman");
    let wr = TAU * 0.1 * f0();
    // (1) pulse response, in 3-D and (f = 1) as a 2-D simulation
    let dx = LAMBDA / 80.0;
    let (x0, len) = (1.5e-6, 2.0e-6);
    let raman = Raman { fraction: 1.0, omega: wr, delta: 0.25 * wr };
    let k_l = TAU / LAMBDA * len;
    for (two_d, fraction, fr) in [(false, 1.0, vec![0.9, 1.0, 1.1]), (false, 0.5, vec![0.9, 1.0, 1.1, 3.0]), (true, 1.0, vec![0.9, 1.0, 1.1])] {
        let mk = |chi3: f64| {
            let mut sim = column(dx, 5.0e-6, 2);
            if two_d {
                // one cell along z, the TE set (Ey, Ex, Hz) of a z-invariant simulation
                sim.region.max[2] = dx;
                sim = sim.two_d(2, crate::spec::Polarization::Te);
            }
            let mut med = Medium::dielectric(1.0).with_name("nl");
            if chi3 != 0.0 {
                med.chi3 = [chi3; 3];
                med.raman = Some(Raman { fraction, ..raman.clone() });
            }
            sim.media.push(med);
            sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + len, 1.0, 1.0] }, medium: "nl".into() });
            sim.sources.push(plane_source(1.0e-6, 1.0));
            sim.monitors.push(Monitor::Probe { name: "p".into(), position: [4.0e-6, 0.0, 0.0], components: vec![Component::Ey] });
            sim.run_time = 70e-15;
            sim
        };
        let lin = run(&mk(0.0))?;
        let a: Vec<f64> = lin.time["p"].fields[0].data.iter().map(|&v| v as f64).collect();
        let dt = lin.dt;
        // Q = g_R * a^2 (trapezoidal; g_R(0) = 0), truncated where g_R < 1e-12 of its peak
        let nk = ((28.0 / raman.delta) / dt).ceil() as usize;
        let gk: Vec<f64> = (0..nk).map(|m| raman.kernel(m as f64 * dt)).collect();
        let a2: Vec<f64> = a.iter().map(|v| v * v).collect();
        let q: Vec<f64> = (0..a.len())
            .map(|n| {
                let mut acc = 0.0;
                for m in 0..nk.min(n + 1) {
                    let w = if m == n { 0.5 } else { 1.0 };
                    acc += w * gk[m] * a2[n - m];
                }
                acc * dt
            })
            .collect();
        let dft = |x: &[f64], f: f64| -> Complex64 {
            x.iter().enumerate().fold(Complex64::new(0.0, 0.0), |acc, (i, v)| acc + Complex64::from_polar(*v, -TAU * f * i as f64 * dt))
        };
        // generated / fundamental ~ 1 % (the delayed part of E^2 peaks lower)
        let chi3 = 0.01 / (k_l * (1.5 * (1.0 - fraction) + 0.6 * fraction));
        let nl = run(&mk(chi3))?;
        if two_d && !nl.stats.backend.contains("2-D") && std::env::var("LIGHTSIM_2D").as_deref() != Ok("0") && std::env::var("LIGHTSIM_FUSED").as_deref() != Ok("0") {
            anyhow::bail!("the 2-D Raman run did not use the 2-D update ({})", nl.stats.backend);
        }
        let b: Vec<f64> = nl.time["p"].fields[0].data.iter().map(|&v| v as f64).collect();
        let n = a.len().min(b.len());
        let egen: Vec<f64> = (0..n).map(|i| b[i] - a[i]).collect();
        let pnl: Vec<f64> = (0..n).map(|i| chi3 * a[i] * ((1.0 - fraction) * a2[i] + fraction * q[i])).collect();
        let mut pred = vec![0.0; n];
        for i in 1..n - 1 {
            pred[i] = -len / (2.0 * C0) * (pnl[i + 1] - pnl[i - 1]) / (2.0 * dt);
        }
        let tag = if two_d { "2-D, " } else { "" };
        for r in fr {
            let f = r * f0();
            // (the third harmonic carries the larger dispersion error of the grid, as in chi3_kerr)
            let tol = if r > 2.0 { 0.02 } else { 5e-3 };
            rep.check(&format!("{tag}f_R = {fraction}: |E_gen| / theory @ {r:.1} f0"), dft(&egen, f).norm() / dft(&pred, f).norm(), 1.0, tol);
        }
        if fraction == 1.0 && !two_d {
            // the delayed response radiates differently from an instantaneous one
            let inst: Vec<f64> = (0..n).map(|i| chi3 * a[i] * a2[i]).collect();
            let mut pi = vec![0.0; n];
            for i in 1..n - 1 {
                pi[i] = -len / (2.0 * C0) * (inst[i + 1] - inst[i - 1]) / (2.0 * dt);
            }
            let ratio = dft(&pred, f0()).norm() / dft(&pi, f0()).norm();
            rep.check_abs("f_R = 1: delayed / instantaneous theory @ f0 (must be < 0.5)", ratio, 0.0, 0.5);
        }
        rep.mcells_per_s = nl.stats.mcells_per_s;
    }
    // (2) stimulated Raman gain of a Stokes probe
    let dx = LAMBDA / 40.0;
    let (fp, fs) = (f0(), f0() - wr / TAU);
    let (x0, len) = (1.5e-6, 2.0e-6);
    let ap = 1.0;
    let gl = 0.02;
    let mk = |pump: bool, probe: bool, chi3: f64| {
        let mut sim = column(dx, 5.0e-6, 2);
        let mut med = Medium::dielectric(1.0).with_name("srs");
        med.chi3 = [chi3; 3];
        med.raman = Some(raman.clone());
        sim.media.push(med);
        sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + len, 1.0, 1.0] }, medium: "srs".into() });
        let cw = |f: f64, a: f64, nm: &str| Source::PlaneWave {
            name: nm.into(),
            axis: 0,
            position: 1.0e-6,
            direction: Direction::Plus,
            polarization: Component::Ey,
            signal: TimeSignal::cw(f, 0.03 * f0()),
            amplitude: a,
            angle: 0.0,
            min: None,
            max: None,
        };
        if pump {
            sim.sources.push(cw(fp, ap, "pump"));
        }
        if probe {
            sim.sources.push(cw(fs, 0.02 * ap, "probe"));
        }
        let apod = Apodization { start: Some(250e-15), end: None, width: Some(60e-15) };
        sim.monitors.push(Monitor::DftField { name: "out".into(), min: [4.0e-6, 0.0, 0.0], max: [4.0e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: vec![fs], apodization: apod, stride: 1 });
        sim.run_time = 700e-15;
        sim
    };
    // g L = (w_s / c) (chi3 / 2) A_p^2 Im g_R L with g_R of the scheme
    let dt = sim_dt(&mk(true, true, 1.0))?;
    let im_g = raman.response_discrete(TAU * (fp - fs), dt).im;
    let chi3 = gl / ((TAU * fs / C0) * 0.5 * ap * ap * im_g * len);
    let both = run(&mk(true, true, chi3))?;
    let pump = run(&mk(true, false, chi3))?;
    let probe = run(&mk(false, true, chi3))?;
    let at = |r: &Results| r.dft["out"].fields[0].at(0, 0, 0, 0);
    let gain = ((at(&both) - at(&pump)).norm() / at(&probe).norm()).ln();
    rep.check("SRS: ln(|E_s| with pump / without) vs g L / 2", gain, gl / 2.0, 0.02);
    Ok(rep)
}
