//! Validation of complex (Hermitian) permittivity tensors ([`crate::herm`]).

use super::*;
use crate::materials::GyroPole;

/// A medium with the complex tensor `e` (real diagonal in `eps`).
fn herm_medium(name: &str, e: [[Complex64; 3]; 3]) -> Medium {
    let mut m = Medium::anisotropic([e[0][0].re, e[1][1].re, e[2][2].re]).with_name(name);
    m.eps_complex = Some(e.map(|r| r.map(|c| [c.re, c.im])));
    m
}

/// Index of the Yee scheme for a wave along an axis in a medium of index
/// `n`: sin(k dx / 2) / dx = n sin(w dt / 2) / (c dt).
fn n_yee(n: f64, w: f64, dx: f64, dt: f64) -> f64 {
    2.0 / (w / C0 * dx) * (n * dx / (C0 * dt) * (0.5 * w * dt).sin()).asin()
}

/// Complex (Hermitian) permittivity tensor given directly, run with complex
/// fields, wave along x:
///
/// 1. Faraday medium eps = [[2.25, 0, 0], [0, 2.25, i g], [0, -i g, 2.25]],
///    g = 0.3, in a half-space: the indices of the circular polarizations
///    (1, +-i) of (E_y, E_z), n = sqrt(2.25 -+ g), from their phase between
///    two depths, and the rotation of the polarization ellipse per length
///    (k0 / 2)(n_- - n_+), against the analytic values (Yee dispersion
///    included), at 0.9, 1.0, 1.1 f0 (the tensor is the same at every
///    frequency);
/// 2. the half-space crossed backwards: the rotation has the same sense in
///    the lab frame (non-reciprocal: a round trip doubles it);
/// 3. a lossless gyrotropic Lorentz pole (bias along x) and the Hermitian
///    tensor eps_inf + chi(f0) of its discrete susceptibility at f0: n_+, n_-
///    and the rotation agree at f0;
/// 4. a real symmetric tensor (a uniaxial crystal rotated 45 degrees about x,
///    the scene of `offdiagonal_tensor`) given as eps_complex reproduces the
///    off-diagonal tensor medium (eps_off).
pub fn hermitian_tensor(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("hermitian_tensor");
    let i = Complex64::i();
    let freqs = vec![0.9 * f0(), f0(), 1.1 * f0()];
    let (eps, g) = (2.25, 0.3);
    let z = Complex64::new(0.0, 0.0);
    let r = |v: f64| Complex64::new(v, 0.0);
    let fara = [[r(eps), z, z], [z, r(eps), i * g], [z, -i * g, r(eps)]];
    let dx = LAMBDA / (40.0 * 1.6);
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
    // (n_+, n_-, rotation, distance) per frequency of a column run
    let measure = |sim: &Simulation, fr: &[f64], guess: &dyn Fn(f64) -> (f64, f64)| -> Result<(Vec<(f64, f64, f64)>, f64, Results)> {
        let res = run(sim)?;
        let grid = sim.build_grid()?;
        let pos = |nm: &str| grid.node(0, res.dft[nm].fields[0].origin[0]);
        let dist = (pos("d1") - pos("d0")).abs();
        let mut out = vec![];
        for (fi, &f) in fr.iter().enumerate() {
            let get = |nm: &str, c: usize| res.dft[nm].field(Component::e(c)).unwrap().at(fi, 0, 0, 0);
            let k0 = TAU * f / C0;
            let (gp, gm) = guess(f);
            let ap = |nm: &str| (get(nm, 1) - i * get(nm, 2)) * 0.5;
            let am = |nm: &str| (get(nm, 1) + i * get(nm, 2)) * 0.5;
            let np = wavenumber(ap("d1") / ap("d0"), dist, k0 * gp) / k0;
            let nm_ = wavenumber(am("d1") / am("d0"), dist, k0 * gm) / k0;
            let rot = wrap_pi(angle(get("d1", 1), get("d1", 2)) - angle(get("d0", 1), get("d0", 2)), 0.5 * k0 * (gm - gp) * dist);
            out.push((np, nm_, rot));
        }
        Ok((out, dist, res))
    };
    let depths = [3.2e-6, 4.0e-6];
    let fsim = gyro_column(0, dx, herm_medium("faraday", fara), (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &freqs, 110e-15, None);
    let dt = sim_dt(&fsim)?;
    let exact = |f: f64| (n_yee((eps - g).sqrt(), TAU * f, dx, dt), n_yee((eps + g).sqrt(), TAU * f, dx, dt));
    let (fwd, dist, res) = measure(&fsim, &freqs, &exact)?;
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (fi, &f) in freqs.iter().enumerate() {
        let (np, nm_) = exact(f);
        let k0 = TAU * f / C0;
        let t = f / f0();
        rep.check(&format!("Faraday tensor: n_+ @ {t:.1} f0 (sqrt(eps - g) = {:.5})", (eps - g).sqrt()), fwd[fi].0, np, 2e-4);
        rep.check(&format!("Faraday tensor: n_- @ {t:.1} f0 (sqrt(eps + g) = {:.5})", (eps + g).sqrt()), fwd[fi].1, nm_, 2e-4);
        rep.check(&format!("Faraday tensor: rotation over {:.2} um @ {t:.1} f0", dist * 1e6), fwd[fi].2, 0.5 * k0 * (nm_ - np) * dist, 2e-3);
    }
    // (2) backwards
    let bsim = gyro_column(0, dx, herm_medium("faraday", fara), (-20e-6, 3.5e-6), 5.0e-6, Direction::Minus, &[2.8e-6, 2.0e-6], &freqs, 110e-15, None);
    let (bwd, bdist, _) = measure(&bsim, &freqs, &exact)?;
    for (fi, &f) in freqs.iter().enumerate() {
        rep.check(&format!("non-reciprocal: rotation per length backwards / forwards (lab frame) @ {:.1} f0", f / f0()), bwd[fi].2 / bdist / (fwd[fi].2 / dist), 1.0, 2e-3);
    }
    // (3) the gyrotropic pole and its tensor at f0
    let w0 = TAU * f0();
    let gp = GyroPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: 1.5 * w0, gamma: 0.0 }, bias: [0.2 * w0, 0.0, 0.0] };
    let mut gm = Medium::dielectric(eps).with_name("gyro");
    gm.gyro = vec![gp.clone()];
    let gsim = gyro_column(0, dx, gm, (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &[f0()], 110e-15, None);
    let gdt = sim_dt(&gsim)?;
    let chi = gp.chi_discrete(w0, gdt);
    let mut ht = [[z; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            ht[a][b] = chi[a][b] + if a == b { r(eps) } else { z };
        }
    }
    // (the discrete chi of a lossless pole is Hermitian up to rounding)
    for a in 0..3 {
        for b in 0..3 {
            let m = 0.5 * (ht[a][b] + ht[b][a].conj());
            ht[a][b] = m;
        }
    }
    let guess = |_f: f64| ((ht[1][1] + i * ht[1][2]).re.sqrt(), (ht[1][1] - i * ht[1][2]).re.sqrt());
    let mut hsim = gyro_column(0, dx, herm_medium("gyro", ht), (2.5e-6, 20e-6), 1.0e-6, Direction::Plus, &depths, &[f0()], 110e-15, None);
    hsim.courant = gsim.courant * gdt / sim_dt(&hsim)?; // the same time step as the pole run
    let (a, _, _) = measure(&gsim, &[f0()], &guess)?;
    let (b, _, _) = measure(&hsim, &[f0()], &guess)?;
    rep.check(&format!("vs gyrotropic pole at f0: n_+ (pole {:.5})", a[0].0), b[0].0, a[0].0, 1e-4);
    rep.check(&format!("vs gyrotropic pole at f0: n_- (pole {:.5})", a[0].1), b[0].1, a[0].1, 1e-4);
    rep.check("vs gyrotropic pole at f0: rotation", b[0].2, a[0].2, 2e-3);
    // (4) a real symmetric tensor through eps_complex
    let base = rotated_crystal_sim(0, (0.0, 0.0), false);
    let ra = run(&base)?;
    let mut hb = base.clone();
    let et = hb.media[0].eps_tensor();
    let name = hb.media[0].name.clone();
    hb.media[0] = herm_medium(&name, et.map(|row| row.map(r)));
    let rb = run(&hb)?;
    for nm in ["a", "b"] {
        for c in [Component::Ey, Component::Ez] {
            rep.check_abs(&format!("real symmetric eps_complex vs eps_off: {} at {nm} (max |diff| / peak)", c.name()), dft_rel_diff(&ra, &rb, nm, c), 0.0, 1e-5);
        }
    }
    Ok(rep)
}
