//! Validation of the four-level two-electron gain medium ([`crate::gain2`]).

use super::*;
use crate::constants::EPS0;
use crate::gain2::TwoElectron;

/// Steady state of the rate equations with time-averaged field work
/// `w(n) = (wa, wb)` on the two transitions, by integrating them to
/// convergence.
fn steady_state(g: &TwoElectron, w: &dyn Fn([f64; 4]) -> (f64, f64)) -> [f64; 4] {
    let mut n = g.initial();
    let h = 0.02 * [g.t30, g.t32, g.t21, g.t10].into_iter().fold(f64::INFINITY, f64::min);
    for _ in 0..2_000_000 {
        let (wa, wb) = w(n);
        let r = g.rates(n, wa, wb);
        let mut d = 0.0f64;
        for l in 0..4 {
            n[l] += h * r[l];
            d = d.max((h * r[l]).abs());
        }
        if d < 1e-13 * g.density {
            break;
        }
    }
    n
}

/// Populations of one cell driven by the field `e(t)` (samples `es` at
/// times `ts`), from the continuous model integrated by RK4 at the
/// sample spacing with cubic (Catmull-Rom) interpolation between samples.
fn integrate_cell(g: &TwoElectron, ts: &[f64], es: &[f64]) -> [f64; 4] {
    let (ka, kb) = (g.kappa_a(), g.kappa_b());
    let ef = |t: f64| -> f64 {
        let h = ts[1] - ts[0];
        let x = (t - ts[0]) / h;
        if x <= 0.0 {
            return 0.0;
        }
        let i = (x.floor() as usize).min(es.len() - 2);
        let u = x - i as f64;
        let p = |j: isize| es[(i as isize + j).clamp(0, es.len() as isize - 1) as usize];
        let (p0, p1, p2, p3) = (p(-1), p(0), p(1), p(2));
        0.5 * (2.0 * p1 + (-p0 + p2) * u + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u)
    };
    // y = (Pa, Pa', Pb, Pb', N0..N3)
    let f = |t: f64, y: &[f64; 8]| -> [f64; 8] {
        let e = ef(t);
        let n = [y[4], y[5], y[6], y[7]];
        let r = g.rates(n, e * y[1], e * y[3]);
        [
            y[1],
            ka * (n[1] - n[2]) * e - g.gamma_a * y[1] - g.omega_a * g.omega_a * y[0],
            y[3],
            kb * (n[0] - n[3]) * e - g.gamma_b * y[3] - g.omega_b * g.omega_b * y[2],
            r[0],
            r[1],
            r[2],
            r[3],
        ]
    };
    let n0 = g.initial();
    let mut y = [0.0, 0.0, 0.0, 0.0, n0[0], n0[1], n0[2], n0[3]];
    // start where the field starts (before the first sample it is zero)
    let h = (ts[1] - ts[0]) / 2.0;
    let mut t = 0.0;
    let t_end = *ts.last().unwrap();
    while t < t_end - 1e-3 * h {
        let k1 = f(t, &y);
        let a = |k: &[f64; 8], s: f64| -> [f64; 8] { std::array::from_fn(|q| y[q] + s * k[q]) };
        let k2 = f(t + 0.5 * h, &a(&k1, 0.5 * h));
        let k3 = f(t + 0.5 * h, &a(&k2, 0.5 * h));
        let k4 = f(t + h, &a(&k3, h));
        for q in 0..8 {
            y[q] += h / 6.0 * (k1[q] + 2.0 * k2[q] + 2.0 * k3[q] + k4[q]);
        }
        t += h;
    }
    [y[4], y[5], y[6], y[7]]
}

/// A transition coupling for |Im chi| = `im` at resonance per population
/// difference `nd`: kappa = im gamma omega eps0 / nd.
fn kappa_for(im: f64, gamma: f64, omega: f64, nd: f64) -> f64 {
    im * gamma * omega * EPS0 / nd
}

/// Four-level two-electron gain medium (Chang and Taflove's model,
/// [`crate::gain2::TwoElectron`]):
///
/// 1. Population dynamics under a CW pump at omega_b: a single plane of
///    atoms (one plane of E_y samples) driven by a pump wave for 1.5 ps;
///    the populations at the end against the continuous model (both
///    oscillators and the four rate equations with their Pauli factors)
///    integrated by RK4 from the field recorded at the atoms.
/// 2. Small-signal gain at omega_a for the steady-state inversion: a thin
///    slab pumped by a CW wave at omega_b to its steady state, with a weak
///    CW probe at omega_a; the probe's single-pass gain ln |E_out / E_in|
///    against -(k0 / 2) Im chi_a(omega_a) (N1 - N2) L with the steady-state
///    populations of the rate equations at the pump's work rate on both
///    transitions (the pump also drives the lasing transition off
///    resonance), with the discrete Lorentzians of the transitions,
///    integrated through the slab; and the attenuation of the pump.
/// 3. Electron number: sum of the levels / (2 Nd) after each run.
pub fn gain_two_electron(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("gain_two_electron");
    let w0 = TAU * f0();
    let nd = 1e24;
    let (wa, wb) = (w0, 1.3 * w0);
    let (ga, gb) = (0.1 * w0, 0.1 * w0);
    let base = TwoElectron {
        density: nd,
        omega_a: wa,
        omega_b: wb,
        gamma_a: ga,
        gamma_b: gb,
        t30: 300e-15,
        t32: 20e-15,
        t21: 500e-15,
        t10: 20e-15,
        initial: None,
        kappa_a: Some(kappa_for(0.05, ga, wa, nd)),
        kappa_b: Some(kappa_for(0.02, gb, wb, nd)),
    };
    let dx = LAMBDA / 60.0;
    let w = 2.0 * dx;

    // (1) population dynamics, one plane of atoms at x = 2 um
    let xs = 2.0e-6;
    let pump_amp = 7e7;
    let mut med = Medium::dielectric(1.0).with_name("g2");
    med.gain2 = Some(base.clone());
    let mut sim = column(dx, 4e-6, 2);
    sim.media.push(med);
    sim.structures.push(Structure { name: "atoms".into(), shape: Shape::Box { min: [xs - 0.5 * dx, -1.0, -1.0], max: [xs + 0.5 * dx, 1.0, 1.0] }, medium: "g2".into() });
    sim.sources.push(Source::PlaneWave { name: "pump".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::cw(wb / TAU, 0.1 * wb / TAU), amplitude: pump_amp, angle: 0.0, min: None, max: None });
    sim.monitors.push(Monitor::Probe { name: "e".into(), position: [xs, 0.0, 0.0], components: vec![Component::Ey] });
    sim.run_time = 1.5e-12;
    let r = run(&sim)?;
    rep.mcells_per_s = r.stats.mcells_per_s;
    // one plane of 2 x 2 samples per E component (volume dx^3 each): the E_y
    // atoms are driven, the E_x and E_z ones stay in their initial state
    let vc = dx * dx * dx;
    let n_init = base.initial();
    let sim_n: Vec<f64> = (0..4).map(|l| (r.gain_levels[l] - 8.0 * vc * n_init[l]) / (4.0 * vc)).collect();
    let tr = &r.time["e"];
    let es: Vec<f64> = tr.fields[0].data.iter().map(|&v| v as f64).collect();
    let ref_n = integrate_cell(&base, &tr.times, &es);
    for l in 0..4 {
        rep.check_abs(&format!("CW pump, 1.5 ps: N{l} / Nd vs the integrated model ({:.4})", ref_n[l] / nd), sim_n[l] / nd, ref_n[l] / nd, 2e-3);
    }
    rep.check_abs("CW pump: electrons / (2 Nd) - 1", r.gain_levels.iter().sum::<f64>() / (2.0 * nd * 12.0 * vc) - 1.0, 0.0, 1e-9);

    // (2) small-signal gain of the steady-state inversion
    let (x0, thick) = (2.0e-6, 0.3e-6);
    let x_out = x0 + thick + 0.4e-6;
    let probe_amp = 1e5;
    let mk = |gain: bool, pump: bool, probe: bool| {
        let mut sim = column(dx, x_out + 1.0e-6, 2);
        if gain {
            let mut med = Medium::dielectric(1.0).with_name("g2");
            med.gain2 = Some(base.clone());
            sim.media.push(med);
            sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + thick, 1.0, 1.0] }, medium: "g2".into() });
        }
        if pump {
            sim.sources.push(Source::PlaneWave { name: "pump".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::cw(wb / TAU, 0.1 * wb / TAU), amplitude: pump_amp, angle: 0.0, min: None, max: None });
        }
        if probe {
            sim.sources.push(Source::PlaneWave { name: "probe".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::cw(wa / TAU, 0.1 * wa / TAU), amplitude: probe_amp, angle: 0.0, min: None, max: None });
        }
        let apod = Apodization { start: Some(3.0e-12), end: None, width: Some(100e-15) };
        sim.monitors.push(Monitor::DftField { name: "o".into(), min: [x_out, 0.0, 0.0], max: [x_out, 0.0, 0.0], components: vec![Component::Ey], freqs: vec![wa / TAU, wb / TAU], apodization: apod, stride: 1 });
        sim.run_time = 3.5e-12;
        sim
    };
    // the probe field: the run with pump and probe minus the run with the pump
    // alone (removes what the strong pump leaks into the probe's frequency bin)
    let full = run(&mk(true, true, true))?;
    let pump_only = run(&mk(true, true, false))?;
    let empty = run(&mk(false, true, true))?;
    let probe_free = run(&mk(false, false, true))?;
    let dt = full.dt;
    let at = |r: &Results, f: usize| r.dft["o"].fields[0].at(f, 0, 0, 0);
    let gain = ((at(&full, 0) - at(&pump_only, 0)).norm() / at(&probe_free, 0).norm()).ln();
    let pump_t = (at(&full, 1).norm() / at(&empty, 1).norm()).ln();
    // steady state along the slab: the pump decays as -(k_b / 2) Im chi_b (N0 - N3)
    let (pa, pb) = (base.pole(false), base.pole(true));
    let (ima, imb) = (pa.chi_discrete(wa, dt).im, pb.chi_discrete(wb, dt).im);
    // the pump also drives the lasing transition off resonance (stimulated emission at omega_b)
    let ima_b = pa.chi_discrete(wb, dt).im;
    let steps = 300;
    let hx = thick / steps as f64;
    let (mut la, mut lb) = (0.0f64, 0.0f64);
    let mut n_mid = [0.0; 4];
    for s in 0..steps {
        let a2 = pump_amp * pump_amp * (2.0 * lb).exp();
        // time-averaged work on the pump transition: (eps0 w / 2) Im chi_b (N0 - N3) |A|^2
        let n = steady_state(&base, &|n: [f64; 4]| (0.5 * EPS0 * wb * ima_b * (n[1] - n[2]) * a2, 0.5 * EPS0 * wb * imb * (n[0] - n[3]) * a2));
        if s == steps / 2 {
            n_mid = n;
        }
        la += -0.5 * (wa / C0) * ima * (n[1] - n[2]) * hx;
        lb += -0.5 * (wb / C0) * (imb * (n[0] - n[3]) + ima_b * (n[1] - n[2])) * hx;
    }
    rep.check(&format!("steady state: probe gain ln|E_out/E_in| at omega_a (N2 - N1 = {:.3} Nd mid-slab)", (n_mid[2] - n_mid[1]) / nd), gain, la, 0.01);
    rep.check("steady state: pump attenuation ln|E_out/E_in| at omega_b", pump_t, lb, 0.01);
    rep.check_abs("steady state: electrons / (2 Nd) - 1", full.gain_levels.iter().sum::<f64>() / (2.0 * nd * thick * w * w * 3.0) - 1.0, 0.0, 1e-9);
    laser(run, &base, &mut rep)?;
    Ok(rep)
}

/// The threshold condition of a slab laser (gain medium of background
/// eps_b and length l in air, both facets bare): r^2 exp(2 i n k0 l) = 1 with
/// n^2 = eps_b + chi_discrete(populations), in the form
/// 2 ln r + 2 i n k0 l = 2 pi i m for the longitudinal order m, solved by
/// Newton's method for (omega, N1 - N2) from a start, with N0 - N3 held at
/// `dnb`.
fn threshold(g: &TwoElectron, eps_b: f64, l: f64, dt: f64, dnb: f64, m: f64, start: (f64, f64)) -> (f64, f64) {
    let f = |w: f64, dna: f64| -> Complex64 {
        let chi = g.pole(false).chi_discrete(w, dt) * dna + g.pole(true).chi_discrete(w, dt) * dnb;
        let n = (Complex64::new(eps_b, 0.0) + chi).sqrt();
        let r = (n - 1.0) / (n + 1.0);
        2.0 * r.ln() + Complex64::i() * (2.0 * n * (w / C0) * l - TAU * m)
    };
    let (mut w, mut d) = start;
    for _ in 0..60 {
        let f0v = f(w, d);
        let (hw, hd) = (1e-7 * w, 1e-7 * g.density);
        let fw = (f(w + hw, d) - f0v) / hw;
        let fd = (f(w, d + hd) - f0v) / hd;
        // solve [fw fd] [dw dd]^T = -f0v (real 2 x 2)
        let det = fw.re * fd.im - fd.re * fw.im;
        let dw = (-f0v.re * fd.im + fd.re * f0v.im) / det;
        let dd = (-fw.re * f0v.im + fw.im * f0v.re) / det;
        w += dw;
        d += dd;
        if dw.abs() < 1e-12 * w {
            break;
        }
    }
    (w, d)
}

/// (3) A one-dimensional laser: a slab of the medium (eps_b = 2.25,
/// 4 um, bare facets in air, R = 4 %, a cold mode at omega_a) pumped from one
/// side by a CW wave at omega_b and seeded by a weak pulse at omega_a, at
/// three pump intensities above threshold. Checks: the output is a single
/// steady line (power in the last two picoseconds equal) at the frequency
/// of the threshold condition r^2 exp(2 i n k0 L) = 1 (solved for frequency
/// and inversion with the discrete susceptibilities); the pump threshold
/// from a linear fit of the output power against the pump intensity against
/// the threshold the rate equations predict (the pump intensity whose
/// steady-state inversion, averaged over the slab with the pump's
/// attenuation, reaches the threshold inversion); the mean inversion
/// nearly clamped above threshold (it rises slowly: spatial hole burning in
/// the standing wave leaves the field nodes unsaturated).
fn laser(run: Runner, g0: &TwoElectron, rep: &mut CaseReport) -> Result<()> {
    let nd = g0.density;
    let w0 = TAU * f0();
    let (ga, wb) = (0.05 * w0, 1.5 * w0);
    let g = TwoElectron { gamma_a: ga, omega_b: wb, kappa_a: Some(kappa_for(0.6, ga, g0.omega_a, nd)), kappa_b: Some(kappa_for(0.02, g0.gamma_b, wb, nd)), ..g0.clone() };
    let dx = LAMBDA / 60.0;
    let eps_b: f64 = 2.25;
    let m = 12.0;
    let (x0, l) = (2.0e-6, m * LAMBDA / (2.0 * eps_b.sqrt()));
    let x_out = x0 + l + 0.8e-6;
    let t_run = 8e-12;
    let freqs: Vec<f64> = (0..301).map(|i| f0() * (0.9 + 0.2 * i as f64 / 300.0)).collect();
    let win = |a: f64, b: f64| Apodization { start: Some(a), end: Some(b), width: Some(100e-15) };
    let mk = |pump: f64| {
        let mut sim = column(dx, x_out + 1.0e-6, 2);
        let mut med = Medium::dielectric(eps_b).with_name("laser");
        med.gain2 = Some(g.clone());
        sim.media.push(med);
        sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [x0, -1.0, -1.0], max: [x0 + l, 1.0, 1.0] }, medium: "laser".into() });
        sim.sources.push(Source::PlaneWave { name: "pump".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::cw(wb / TAU, 0.1 * wb / TAU), amplitude: pump, angle: 0.0, min: None, max: None });
        sim.sources.push(Source::PlaneWave { name: "seed".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(f0(), 0.2 * f0()), amplitude: 1e4, angle: 0.0, min: None, max: None });
        for (name, a, b) in [("late", t_run - 1e-12, t_run), ("earlier", t_run - 2e-12, t_run - 1e-12)] {
            sim.monitors.push(Monitor::DftField { name: name.into(), min: [x_out, 0.0, 0.0], max: [x_out, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: win(a, b), stride: 1 });
        }
        sim.run_time = t_run;
        sim
    };
    let dt = sim_dt(&mk(1.0))?;
    // threshold inversion from the cold mode at omega_a
    let fsr = C0 / (2.0 * eps_b.sqrt() * l);
    let dnb_guess = 0.4 * nd;
    let (wth, dth) = threshold(&g, eps_b, l, dt, dnb_guess, m, (TAU * m * fsr, -0.3 * nd));
    // pump threshold: steady-state inversion averaged over the slab (pump attenuated along it)
    let (pa, pb) = (g.pole(false), g.pole(true));
    let (imb, ima_b) = (pb.chi_discrete(wb, dt).im, pa.chi_discrete(wb, dt).im);
    let nb = eps_b.sqrt();
    let mean_inv = |a2: f64| -> f64 {
        let steps = 40;
        let hx = l / steps as f64;
        let (mut lb, mut acc) = (0.0f64, 0.0f64);
        for _ in 0..steps {
            // intensity in the medium: |E|^2 = |A|^2 t^2 (transmission into the slab), t = 2 / (1 + n)
            let e2 = a2 * (2.0 / (1.0 + nb)).powi(2) * (2.0 * lb).exp();
            let n = steady_state(&g, &|n: [f64; 4]| (0.5 * EPS0 * wb * ima_b * (n[1] - n[2]) * e2, 0.5 * EPS0 * wb * imb * (n[0] - n[3]) * e2));
            acc += (n[2] - n[1]) * hx;
            lb += -0.5 * (wb / C0) / nb * (imb * (n[0] - n[3]) + ima_b * (n[1] - n[2])) * hx;
        }
        acc / l
    };
    let (mut lo, mut hi) = (1e14f64, 1e18f64);
    for _ in 0..40 {
        let mid = (lo * hi).sqrt();
        if mean_inv(mid) < -dth { lo = mid } else { hi = mid }
    }
    let i_th = (lo * hi).sqrt();
    let vc = dx * dx * dx;
    let ncell = (l / dx).round() * 4.0;
    let ini = g.initial();
    let mut pts = vec![];
    for frac in [1.4, 1.7, 2.0] {
        let r = run(&mk((frac * i_th).sqrt()))?;
        let n: Vec<f64> = (0..4).map(|q| (r.gain_levels[q] - 2.0 * ncell * vc * ini[q]) / (ncell * vc)).collect();
        let spec = |name: &str| -> Vec<f64> { (0..freqs.len()).map(|i| r.dft[name].fields[0].at(i, 0, 0, 0).norm_sqr()).collect() };
        let (s1, s0) = (spec("late"), spec("earlier"));
        let i = (1..freqs.len() - 1).max_by(|&a, &b| s1[a].partial_cmp(&s1[b]).unwrap()).unwrap();
        let (y0, y1, y2) = (s1[i - 1].ln(), s1[i].ln(), s1[i + 1].ln());
        let fl = freqs[i] + 0.5 * (freqs[1] - freqs[0]) * (y0 - y2) / (y0 - 2.0 * y1 + y2);
        let (p1, p0): (f64, f64) = (s1.iter().sum(), s0.iter().sum());
        pts.push((frac, n, fl, p1, p0));
    }
    for (frac, n, fl, p1, p0) in &pts {
        // the threshold condition with this run's mean pump-transition difference N0 - N3
        let (wl, _) = threshold(&g, eps_b, l, dt, n[0] - n[3], m, (wth, dth));
        rep.check(&format!("laser, pump {frac:.1} x threshold: lasing frequency / threshold condition ({:.4} f0)", wl / TAU / f0()), *fl / (wl / TAU), 1.0, 2e-3);
        rep.check(&format!("laser, pump {frac:.1} x threshold: steady output (power in the last ps / the ps before)"), p1 / p0, 1.0, 0.02);
        rep.check_abs(&format!("laser, pump {frac:.1} x threshold: electrons / (2 Nd) - 1"), n.iter().sum::<f64>() / (2.0 * nd) - 1.0, 0.0, 1e-9);
    }
    // threshold from a linear fit of the output power against the pump intensity
    let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pts.iter().map(|p| p.3).collect();
    let (slope, icpt) = linfit(&xs, &ys);
    rep.check(&format!("laser: pump threshold from the output powers / rate equations (inversion at threshold {:.3} Nd)", -dth / nd), -icpt / slope, 1.0, 0.1);
    let inv: Vec<f64> = pts.iter().map(|p| p.1[2] - p.1[1]).collect();
    // (the mean, not the field-weighted, inversion: spatial hole burning lets it rise slowly)
    rep.check_abs(&format!("laser: inversion clamped, pump 1.4 -> 2.0 x threshold (max / min - 1; mean inversion {:.3} Nd)", inv[0] / nd), inv.iter().cloned().fold(f64::MIN, f64::max) / inv.iter().cloned().fold(f64::MAX, f64::min) - 1.0, 0.0, 0.1);
    Ok(())
}
