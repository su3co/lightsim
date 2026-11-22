//! Physics validation cases with analytic references.
//!
//! These mirror the physics test-suite of FDTDX (`tests/simulation/physics`)
//! and the classic Meep/Lumerical validation examples. Each case builds a
//! [`Simulation`], runs it through a caller-supplied runner (CPU reference or
//! GPU backend), compares against an analytic result and returns a
//! [`CaseReport`]. The Rust integration tests (`tests/physics.rs`) and the GPU
//! tests in `lightsim-cuda` call the same functions.

use crate::constants::{C0, ETA0, PI, TAU};
use crate::geometry::{Shape, Structure};
use crate::grid::{Component, MeshSpec};
use crate::materials::{Medium, Pole};
use crate::monitors::{Apodization, Monitor};
use crate::results::Results;
use crate::sources::{Direction, Source, TimeSignal};
use crate::spec::{Boundaries, Boundary, Simulation};
use anyhow::Result;
use num_complex::Complex64;

pub type Runner<'a> = &'a dyn Fn(&Simulation) -> Result<Results>;

/// Outcome of one validation case.
#[derive(Clone, Debug)]
pub struct CaseReport {
    pub name: String,
    /// (quantity, measured, expected, tolerance, absolute?)
    pub checks: Vec<(String, f64, f64, f64, bool)>,
    pub mcells_per_s: f64,
}

impl CaseReport {
    pub fn new(name: &str) -> CaseReport {
        CaseReport { name: name.into(), checks: vec![], mcells_per_s: 0.0 }
    }
    /// Relative check |m - e| <= tol * |e| (absolute if e == 0).
    pub fn check(&mut self, what: &str, measured: f64, expected: f64, tol: f64) {
        self.checks.push((what.into(), measured, expected, tol, expected == 0.0));
    }
    /// Absolute check |m - e| <= tol.
    pub fn check_abs(&mut self, what: &str, measured: f64, expected: f64, tol: f64) {
        self.checks.push((what.into(), measured, expected, tol, true));
    }
    fn err(m: f64, e: f64, abs: bool) -> f64 {
        if abs { (m - e).abs() } else { (m - e).abs() / e.abs() }
    }
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|(_, m, e, t, a)| Self::err(*m, *e, *a) <= *t)
    }
    pub fn summary(&self) -> String {
        let mut s = format!("{} [{}]\n", self.name, if self.passed() { "PASS" } else { "FAIL" });
        for (w, m, e, t, a) in &self.checks {
            let err = Self::err(*m, *e, *a);
            let kind = if *a { "abs" } else { "rel" };
            s += &format!("    {w:40} measured {m:12.6e}  expected {e:12.6e}  {kind} err {err:9.3e} (tol {t:.1e})\n");
        }
        s
    }
}

const LAMBDA: f64 = 1.0e-6;
fn f0() -> f64 {
    C0 / LAMBDA
}

fn pulse() -> TimeSignal {
    TimeSignal::gaussian(f0(), 0.3 * f0())
}

fn spectrum_power(sig: &TimeSignal, freqs: &[f64], dt: f64, nsteps: usize) -> Vec<f64> {
    sig.spectrum(freqs, dt, nsteps, 0.0).iter().map(|z| z.norm_sqr()).collect()
}

fn band(n: usize) -> Vec<f64> {
    (0..n).map(|i| f0() * (0.85 + 0.3 * i as f64 / (n - 1).max(1) as f64)).collect()
}

/// 1-D plane-wave column: periodic in y and z, PML along x.
fn column(dx: f64, length: f64, ncross: usize) -> Simulation {
    let w = ncross as f64 * dx;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    Simulation::new([0.0, 0.0, 0.0], [length, w, w], MeshSpec::Uniform { dx: [dx; 3] }, b, 60e-15)
}

fn plane_source(x: f64, n: f64) -> Source {
    let _ = n;
    Source::PlaneWave {
        name: "pw".into(),
        axis: 0,
        position: x,
        direction: Direction::Plus,
        polarization: Component::Ey,
        signal: pulse(),
        amplitude: 1.0,
        angle: 0.0,
        min: None,
        max: None,
    }
}

fn flux_x(name: &str, x: f64, w: f64, freqs: &[f64]) -> Monitor {
    Monitor::Flux {
        name: name.into(),
        axis: 0,
        position: x,
        min: [x, -1.0, -1.0],
        max: [x, w + 1.0, w + 1.0],
        freqs: freqs.to_vec(),
        apodization: Apodization::default(),
    }
}

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
    let peak = b[..n].iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let err = a[..n].iter().zip(&b[..n]).fold(0f32, |m, (x, y)| m.max((x - y).abs())) as f64;
    let mut rep = CaseReport::new("pml_reflection");
    rep.mcells_per_s = small.stats.mcells_per_s;
    rep.check("max |small - big| / peak", err / peak, 0.0, 1e-2);
    Ok(rep)
}

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

fn linfit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let s = sxy / sxx;
    (s, my - s * mx)
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

/// Non-uniform (graded) mesh: a plane wave crossing a refined region (4x
/// finer cells) behaves like on the uniform coarse grid — same transmitted
/// flux, no spurious reflection from the grading.
pub fn nonuniform_grid(run: Runner) -> Result<CaseReport> {
    let n_cross = 2usize;
    let dxc = LAMBDA / 25.0;
    let w = n_cross as f64 * dxc;
    let mk = |refine: bool| -> Simulation {
        let mut b = Boundaries::all(Boundary::Periodic);
        b.x = [Boundary::pml(), Boundary::pml()];
        let overrides = if refine {
            vec![crate::grid::MeshOverride { min: [1.8e-6, -1.0, -1.0], max: [2.2e-6, 1.0, 1.0], dx: [dxc / 4.0, dxc, dxc] }]
        } else {
            vec![]
        };
        let mesh = MeshSpec::Auto {
            wavelength: LAMBDA,
            steps_per_wavelength: 25.0,
            grading: 1.2,
            min_step: None,
            max_step: None,
            overrides,
            symmetry: [None; 3],
        };
        let mut sim = Simulation::new([0.0; 3], [4e-6, w, w], mesh, b, 60e-15);
        sim.sources.push(plane_source(1.0e-6, 1.0));
        let freqs = band(3);
        sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim
    };
    let fine = run(&mk(true))?;
    let coarse = run(&mk(false))?;
    let freqs = band(3);
    let mut rep = CaseReport::new("nonuniform_grid");
    rep.mcells_per_s = fine.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let tc = coarse.flux["T"][i];
        rep.check(&format!("T_graded/T_uniform @ {:.3} f0", f / f0()), fine.flux["T"][i] / tc, 1.0, 5e-3);
        rep.check(&format!("|R|/T_uniform @ {:.3} f0", f / f0()), (fine.flux["R"][i] / tc).abs(), 0.0, 1e-3);
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

/// TFSF box in vacuum: the plane wave is confined to the total-field region.
pub fn tfsf_leakage(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 25.0;
    let l = 1.2e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    let a = 0.3e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    let freqs = band(3);
    sim.monitors.push(Monitor::DftField { name: "in".into(), min: [0.0; 3], max: [0.0; 3], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::DftField { name: "out".into(), min: [0.45e-6, 0.0, 0.0], max: [0.45e-6, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
    sim.monitors.push(Monitor::FluxBox { name: "net".into(), min: [-0.42e-6; 3], max: [0.42e-6; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("tfsf_leakage");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let inside = res.dft["in"].fields[0].at(i, 0, 0, 0).norm();
        let outside = res.dft["out"].fields[0].at(i, 0, 0, 0).norm();
        rep.check(&format!("|E_in|/|S| @ {:.3} f0", f / f0()), inside / sp[i].norm(), 1.0, 0.02);
        rep.check_abs(&format!("|E_out|/|E_in| @ {:.3} f0", f / f0()), outside / inside, 0.0, 5e-3);
        let inc_face = 0.5 / ETA0 * sp[i].norm_sqr() * (2.0 * a).powi(2);
        rep.check_abs(&format!("|net flux|/P_face @ {:.3} f0", f / f0()), (res.flux["net"][i] / inc_face).abs(), 0.0, 5e-3);
    }
    Ok(rep)
}

/// Mie efficiencies (Bohren & Huffman BHMIE) for relative index m and size
/// parameter x. Returns (Q_ext, Q_sca).
pub fn mie_q(m: Complex64, x: f64) -> (f64, f64) {
    let nstop = (x + 4.0 * x.powf(1.0 / 3.0) + 2.0) as usize;
    let y = m * x;
    let nmx = (nstop as f64).max(y.norm()) as usize + 16;
    let mut d = vec![Complex64::new(0.0, 0.0); nmx + 1];
    for n in (1..=nmx).rev() {
        let en = Complex64::new(n as f64, 0.0) / y;
        d[n - 1] = en - 1.0 / (d[n] + en);
    }
    let (mut psi0, mut psi1) = (x.cos(), x.sin());
    let (mut chi0, mut chi1) = (-x.sin(), x.cos());
    let mut xi1 = Complex64::new(psi1, -chi1);
    let (mut qsca, mut qext) = (0.0, 0.0);
    for n in 1..=nstop {
        let nf = n as f64;
        let psi = (2.0 * nf - 1.0) / x * psi1 - psi0;
        let chi = (2.0 * nf - 1.0) / x * chi1 - chi0;
        let xi = Complex64::new(psi, -chi);
        let da = d[n] / m + nf / x;
        let db = d[n] * m + nf / x;
        let an = (da * psi - psi1) / (da * xi - xi1);
        let bn = (db * psi - psi1) / (db * xi - xi1);
        qsca += (2.0 * nf + 1.0) * (an.norm_sqr() + bn.norm_sqr());
        qext += (2.0 * nf + 1.0) * (an + bn).re;
        psi0 = psi1;
        psi1 = psi;
        chi0 = chi1;
        chi1 = chi;
        xi1 = Complex64::new(psi1, -chi1);
    }
    (2.0 * qext / (x * x), 2.0 * qsca / (x * x))
}

/// Mie scattering of a dielectric sphere under TFSF illumination: scattered
/// power (flux box in the scattered-field region) versus Mie theory.
pub fn mie_scattering(run: Runner) -> Result<CaseReport> {
    let n_s = 2.0;
    let radius = 0.25e-6;
    let dx = LAMBDA / (25.0 * n_s);
    let l = 1.5e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 45e-15);
    sim.media.push(Medium::from_index(n_s).with_name("bead"));
    sim.structures.push(Structure { name: "sphere".into(), shape: Shape::Sphere { center: [0.0; 3], radius }, medium: "bead".into() });
    let a = 0.4e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: pulse(), amplitude: 1.0 });
    let freqs = band(5);
    let b = 0.52e-6;
    sim.monitors.push(Monitor::FluxBox { name: "sca".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("mie_scattering");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let x = TAU * f / C0 * radius;
        let (_qe, qs) = mie_q(Complex64::new(n_s, 0.0), x);
        let p_exp = qs * PI * radius * radius * 0.5 / ETA0 * sp[i].norm_sqr();
        rep.check(&format!("P_sca/Mie @ {:.3} f0 (Qsca={:.3})", f / f0(), qs), res.flux["sca"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Gaussian beam focused at the source plane: transmitted power equals the
/// analytic 0.5 n/eta0 |A S|^2 pi w0^2 / 2.
pub fn gaussian_beam(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 20.0;
    let w0 = 1.0e-6;
    let mut sim = Simulation::new([0.0, -3e-6, -3e-6], [3e-6, 3e-6, 3e-6], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.sources.push(Source::GaussianBeam {
        name: "gb".into(),
        axis: 0,
        position: 0.5e-6,
        direction: Direction::Plus,
        polarization: Component::Ey,
        center: [0.0, 0.0, 0.0],
        waist: w0,
        signal: pulse(),
        amplitude: 1.0,
        focus_distance: 0.0,
        theta: 0.0,
        phi: 0.0,
    });
    let freqs = band(3);
    sim.monitors.push(flux_x("T", 2.0e-6, 0.0, &freqs));
    if let Some(Monitor::Flux { min, max, .. }) = sim.monitors.last_mut() {
        *min = [2.0e-6, -2.9e-6, -2.9e-6];
        *max = [2.0e-6, 2.9e-6, 2.9e-6];
    }
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("gaussian_beam");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let p_exp = 0.5 / ETA0 * sp[i].norm_sqr() * PI * w0 * w0 / 2.0;
        rep.check(&format!("P/P_analytic @ {:.3} f0", f / f0()), res.flux["T"][i] / p_exp, 1.0, 0.02);
    }
    Ok(rep)
}

/// Tilted Gaussian beam (polar angle 20 deg from +x, azimuth 30 deg, focus
/// 1 um beyond the source plane): the intensity centroid on two planes moves
/// along the stated direction, the power is that of the beam,
/// 0.5/eta0 |A S|^2 pi w0^2 / 2, and nothing is emitted behind the source.
pub fn gaussian_beam_tilted(run: Runner) -> Result<CaseReport> {
    let dx = LAMBDA / 16.0;
    let w0 = 1.5e-6;
    let (theta, phi) = (20f64.to_radians(), 30f64.to_radians());
    let half = 4.0e-6;
    let (xs, x1, x2) = (0.6e-6, 1.2e-6, 2.8e-6);
    let centre = [xs, -0.5e-6, -0.3e-6];
    let sig = TimeSignal::gaussian(f0(), 0.1 * f0());
    let mut sim = Simulation::new([0.0, -half, -half], [3.4e-6, half, half], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 50e-15);
    sim.sources.push(Source::GaussianBeam {
        name: "gb".into(),
        axis: 0,
        position: xs,
        direction: Direction::Plus,
        polarization: Component::Ey,
        center: centre,
        waist: w0,
        signal: sig.clone(),
        amplitude: 1.0,
        focus_distance: 1.0e-6,
        theta,
        phi,
    });
    let f = vec![f0()];
    for (name, x) in [("T", x2), ("R", 0.3e-6)] {
        sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, -1.0], max: [x, 1.0, 1.0], freqs: f.clone(), apodization: Apodization::default() });
    }
    for (name, x) in [("a", x1), ("b", x2)] {
        sim.monitors.push(Monitor::DftField { name: name.into(), min: [x, -half, -half], max: [x, half, half], components: vec![Component::Ey], freqs: f.clone(), apodization: Apodization::default(), stride: 0 });
    }
    let res = run(&sim)?;
    let g = sim.build_grid()?;
    // centroid of |Ey|^2 on a plane: (x of the plane, mean y, mean z)
    let centroid = |name: &str| -> [f64; 3] {
        let fld = &res.dft[name].fields[0];
        let (mut w, mut y, mut z) = (0.0, 0.0, 0.0);
        for j in 0..fld.dims[2] {
            for kk in 0..fld.dims[3] {
                let p2 = fld.at(0, 0, j, kk).norm_sqr();
                let pos = g.position3(Component::Ey, [fld.origin[0], fld.origin[1] + j, fld.origin[2] + kk]);
                w += p2;
                y += p2 * pos[1];
                z += p2 * pos[2];
            }
        }
        [g.node(0, fld.origin[0]), y / w, z / w]
    };
    let (ca, cb) = (centroid("a"), centroid("b"));
    let (ty, tz) = ((cb[1] - ca[1]) / (cb[0] - ca[0]), (cb[2] - ca[2]) / (cb[0] - ca[0]));
    let theta_m = (ty * ty + tz * tz).sqrt().atan().to_degrees();
    let phi_m = tz.atan2(ty).to_degrees();
    // the beam axis passes through `centre` on the source plane
    let xs_node = g.node(0, g.nearest_node(0, xs));
    let y_expected = centre[1] + (ca[0] - xs_node) * theta.tan() * phi.cos();
    let sp = sig.spectrum(&f, res.dt, res.stats.steps, 0.0);
    let p_exp = 0.5 / ETA0 * sp[0].norm_sqr() * PI * w0 * w0 / 2.0;
    let mut rep = CaseReport::new("gaussian_beam_tilted");
    rep.mcells_per_s = res.stats.mcells_per_s;
    // The centroid of |E|^2 on planes that are not normal to the beam moves
    // slightly steeper than the beam axis, because the beam diverges (12 deg
    // for this waist): about 0.5 deg here.
    rep.check_abs("polar angle of the beam (deg)", theta_m, 20.0, 1.0);
    rep.check_abs("azimuth of the beam (deg)", phi_m, 30.0, 1.5);
    rep.check_abs("beam centre on the first plane, y (um)", ca[1] * 1e6, y_expected * 1e6, 0.03);
    rep.check("P/P_analytic", res.flux["T"][0] / p_exp, 1.0, 0.03);
    rep.check_abs("power behind the source / P", (res.flux["R"][0] / p_exp).abs(), 0.0, 0.01);
    Ok(rep)
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

/// Energy conservation: zero net flux through a closed box around a lossless
/// scatterer (relative to the incident power through one face).
pub fn energy_conservation(run: Runner) -> Result<CaseReport> {
    let n_s = 1.8;
    let dx = LAMBDA / (20.0 * n_s);
    let l = 1.4e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 40e-15);
    sim.media.push(Medium::from_index(n_s).with_name("cube"));
    sim.structures.push(Structure { name: "c".into(), shape: Shape::Box { min: [-0.15e-6; 3], max: [0.15e-6; 3] }, medium: "cube".into() });
    let a = 0.45e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ez, signal: pulse(), amplitude: 1.0 });
    let freqs = band(3);
    let b = 0.3e-6;
    sim.monitors.push(Monitor::FluxBox { name: "net".into(), min: [-b; 3], max: [b; 3], freqs: freqs.clone(), apodization: Apodization::default() });
    let res = run(&sim)?;
    let sp = pulse().spectrum(&freqs, res.dt, res.stats.steps, 0.0);
    let mut rep = CaseReport::new("energy_conservation");
    rep.mcells_per_s = res.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let p_face = 0.5 / ETA0 * sp[i].norm_sqr() * (2.0 * b).powi(2);
        rep.check_abs(&format!("|net flux|/P_face @ {:.3} f0", f / f0()), (res.flux["net"][i] / p_face).abs(), 0.0, 1e-3);
    }
    Ok(rep)
}

/// Energy monitor: a plane-wave pulse of unit amplitude inside a box holds
/// the energy U = eps0 n A c int s(t)^2 dt (half electric, half magnetic), in
/// vacuum and in a dielectric, and nothing is left once it has gone.
pub fn stored_energy(run: Runner) -> Result<CaseReport> {
    let n_cross = 2usize;
    let mut rep = CaseReport::new("stored_energy");
    for n in [1.0f64, 2.0] {
        let dx = LAMBDA / (30.0 * n); // 30 cells per wavelength in the medium
        let w = n_cross as f64 * dx;
        let mut sim = column(dx, 8e-6, n_cross);
        sim.media.push(Medium::from_index(n).with_name("bg"));
        sim.background = "bg".into();
        sim.run_time = 40e-15 + 30e-15 * n;
        sim.sources.push(plane_source(1.0e-6, n));
        sim.monitors.push(Monitor::Energy { name: "u".into(), min: [1.5e-6, -1.0, -1.0], max: [7.0e-6, 1.0, 1.0], interval: 10 });
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let u = &res.energy["u"].energy;
        // int s^2 dt of the Gaussian pulse: tau sqrt(pi) / 2 (1 + exp(-(w0 tau)^2))
        let tau = 1.0 / (TAU * 0.3 * f0());
        let int_s2 = tau * PI.sqrt() / 2.0 * (1.0 + (-(TAU * f0() * tau).powi(2)).exp());
        let exact = crate::constants::EPS0 * n * w * w * C0 * int_s2;
        let peak = u.iter().cloned().fold(0.0, f64::max);
        // the plateau while the whole pulse is inside the box
        let inside: Vec<f64> = u.iter().cloned().filter(|v| *v > 0.999 * peak).collect();
        rep.check(&format!("U / (eps0 n A c int s^2), n = {n}"), peak / exact, 1.0, 0.01);
        rep.check_abs(&format!("samples on the plateau, n = {n}"), (inside.len() >= 5) as u8 as f64, 1.0, 0.0);
        rep.check_abs(&format!("U after the pulse has left / peak, n = {n}"), u[u.len() - 1] / peak, 0.0, 1e-4);
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
    let amp = tail.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
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

fn dft_point(name: &str, x: f64, comps: Vec<Component>, freqs: &[f64], stride: usize) -> Monitor {
    Monitor::DftField {
        name: name.into(),
        min: [x, 0.0, 0.0],
        max: [x, 0.0, 0.0],
        components: comps,
        freqs: freqs.to_vec(),
        apodization: Apodization::default(),
        stride,
    }
}

/// Wrap the phase of `ratio` around the expected value and return k.
fn wavenumber(ratio: Complex64, dist: f64, k_guess: f64) -> f64 {
    let mut ph = ratio.arg();
    let e = k_guess * dist;
    while ph < e - PI {
        ph += TAU;
    }
    while ph > e + PI {
        ph -= TAU;
    }
    ph / dist
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

/// As `rotated_crystal`, with conductivities `sig = (Im eps_e, Im eps_o)` at
/// f0 along the crystal axes: the eigen-polarizations then carry complex
/// indices sqrt(eps + i sigma / (w eps0)) (cf. FDTDX test_complex_tensor).
fn rotated_crystal_lossy(run: Runner, axis: usize, sig: (f64, f64), rep: &mut CaseReport) -> Result<()> {
    let (no, ne) = (1.5f64, 1.8f64);
    let lossy = sig.0 != 0.0 || sig.1 != 0.0;
    let w0 = TAU * f0();
    let n_e = Complex64::new(ne * ne, sig.0).sqrt();
    let n_o = Complex64::new(no * no, sig.1).sqrt();
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
    sim.media.push(med.rotated(r));
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

/// max |a - b| / max |a| over the DFT monitor `name`, component `c`.
pub fn dft_rel_diff(a: &Results, b: &Results, name: &str, c: Component) -> f64 {
    let (fa, fb) = (a.dft[name].field(c).unwrap(), b.dft[name].field(c).unwrap());
    assert_eq!(fa.dims, fb.dims);
    let (mut peak, mut err) = (0f64, 0f64);
    for f in 0..fa.dims[0] {
        for i in 0..fa.dims[1] {
            for j in 0..fa.dims[2] {
                for k in 0..fa.dims[3] {
                    let (x, y) = (fa.at(f, i, j, k), fb.at(f, i, j, k));
                    peak = peak.max(x.norm());
                    err = err.max((x - y).norm());
                }
            }
        }
    }
    err / peak
}

/// Power reflectance of a lossless multilayer at normal incidence
/// (characteristic-matrix method): incidence medium `n_in`, `layers` =
/// (index, thickness) in order, exit medium `n_out`, vacuum wavenumber `k0`.
pub fn multilayer_reflectance(n_in: f64, layers: &[(f64, f64)], n_out: f64, k0: f64) -> f64 {
    let i = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let mut m = [[one, 0.0 * one], [0.0 * one, one]];
    for &(n, d) in layers {
        let (sn, cs) = (k0 * n * d).sin_cos();
        let l = [[one * cs, i * sn / n], [i * sn * n, one * cs]];
        m = [
            [m[0][0] * l[0][0] + m[0][1] * l[1][0], m[0][0] * l[0][1] + m[0][1] * l[1][1]],
            [m[1][0] * l[0][0] + m[1][1] * l[1][0], m[1][0] * l[0][1] + m[1][1] * l[1][1]],
        ];
    }
    let b = m[0][0] + m[0][1] * n_out;
    let c = m[1][0] + m[1][1] * n_out;
    ((b * n_in - c) / (b * n_in + c)).norm_sqr()
}

/// Imported / graded permittivity (`Shape::IndexGrid`).
///
/// (a) A constant grid is the same structure as a staircased `Box` of that
///     permittivity: identical DFT fields and flux.
/// (b) A linear permittivity ramp 1 -> 4 over 1 um, as an anti-reflection
///     transition into an eps = 4 half-space and as a free-standing graded
///     layer (abrupt 4 -> 1 back face), against a fine transfer-matrix
///     calculation of the same continuous profile.
pub fn index_grid(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("index_grid");
    let n_cross = 2usize;

    // ---- (a) constant grid == box (faces deliberately off the Yee nodes) ----
    {
        let dx = LAMBDA / 40.0;
        let w = n_cross as f64 * dx;
        let freqs = band(3);
        let (lo, hi) = ([2.013e-6, -1.0, -1.0], [2.487e-6, 1.0, 1.0]);
        let build = |shape: Shape| {
            let mut sim = column(dx, 4e-6, n_cross);
            sim.subpixel = false;
            sim.media.push(Medium::dielectric(4.0).with_name("d"));
            sim.structures.push(Structure { name: "s".into(), shape, medium: "d".into() });
            sim.sources.push(plane_source(1.0e-6, 1.0));
            sim.monitors.push(flux_x("T", 3.2e-6, w, &freqs));
            sim.monitors.push(Monitor::DftField {
                name: "e".into(),
                min: [0.5e-6, 0.0, 0.0],
                max: [3.5e-6, 0.0, 0.0],
                components: vec![Component::Ey],
                freqs: freqs.clone(),
                apodization: Apodization::default(),
                stride: 1,
            });
            sim.run_time = 90e-15;
            sim
        };
        let a = run(&build(Shape::Box { min: lo, max: hi }))?;
        // 3 x 1 x 2 samples, all 4.0
        let grid = Shape::IndexGrid { min: lo, max: hi, dims: [3, 1, 2], eps: vec![4.0; 6], eps_axes: None };
        let b = run(&build(grid))?;
        rep.check_abs("const grid vs box: max |dE| / peak", dft_rel_diff(&a, &b, "e", Component::Ey), 0.0, 1e-6);
        let dt_max = (0..freqs.len()).map(|i| (b.flux["T"][i] / a.flux["T"][i] - 1.0).abs()).fold(0.0, f64::max);
        rep.check_abs("const grid vs box: max |dT| / T", dt_max, 0.0, 1e-6);
    }

    // ---- (b) linear eps ramp vs transfer matrix ----
    let dx = LAMBDA / 60.0;
    let w = n_cross as f64 * dx;
    let freqs = band(5);
    let (x0, len) = (2.0e-6, 1.0e-6);
    let (e0, e1) = (1.0, 4.0);
    for substrate in [true, false] {
        // Free-standing layer: end the grid half a cell past the last node it
        // owns, so the (staircased) back face sits exactly at the box face.
        let length = if substrate { len } else { len + 0.5 * dx };
        let slope = (e1 - e0) / len;
        let ns = 41usize;
        let eps: Vec<f64> = (0..ns).map(|i| e0 + slope * length * i as f64 / (ns - 1) as f64).collect();
        let mut sim = column(dx, 5e-6, n_cross);
        sim.media.push(Medium::dielectric(e1).with_name("sub"));
        // The medium's own eps is irrelevant (the grid replaces it).
        sim.media.push(Medium::dielectric(2.0).with_name("graded"));
        if substrate {
            // painted first, underneath the grid: the ramp ends on eps = 4
            sim.structures.push(Structure {
                name: "sub".into(),
                shape: Shape::Box { min: [x0 + 0.5 * len, -1.0, -1.0], max: [10e-6, 1.0, 1.0] },
                medium: "sub".into(),
            });
        }
        sim.structures.push(Structure {
            name: "ramp".into(),
            shape: Shape::IndexGrid { min: [x0, -1.0, -1.0], max: [x0 + length, 1.0, 1.0], dims: [ns, 1, 1], eps, eps_axes: None },
            medium: "graded".into(),
        });
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(flux_x("T", 4.0e-6, w, &freqs));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim.run_time = 120e-15;
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
        // transfer matrix of the continuous profile, 4000 sub-layers
        let nl = 4000usize;
        let d = length / nl as f64;
        let layers: Vec<(f64, f64)> = (0..nl).map(|i| ((e0 + slope * (i as f64 + 0.5) * d).sqrt(), d)).collect();
        let n_out = if substrate { e1.sqrt() } else { 1.0 };
        let label = if substrate { "ramp->eps4" } else { "free ramp" };
        for (i, f) in freqs.iter().enumerate() {
            let inc = 0.5 / ETA0 * sp[i] * w * w;
            let r_exp = multilayer_reflectance(1.0, &layers, n_out, TAU * f / C0);
            rep.check(&format!("{label} T @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0 - r_exp, 0.01);
            rep.check_abs(&format!("{label} R @ {:.3} f0", f / f0()), -res.flux["R"][i] / inc, r_exp, if substrate { 2e-4 } else { 3e-3 });
        }
    }
    Ok(rep)
}

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

// ----------------------------------------------------------------------
// 2-D simulations
// ----------------------------------------------------------------------

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
        let mut sim = Simulation::new(lift2(w, -1.0 * um, -0.8 * um, -0.5 * um), lift2(w, 1.0 * um, 0.8 * um, 0.5 * um), MeshSpec::Uniform { dx: [40e-9, 36e-9, 44e-9] }, b, 14e-15)
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
                f.re.iter().chain(f.im.iter()).fold(0f32, |m, x| m.max(x.abs())) as f64
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
fn sim_2d(min: [f64; 2], max: [f64; 2], dx: f64, bx: [Boundary; 2], by: [Boundary; 2], run_time: f64, pol: crate::spec::Polarization) -> Simulation {
    let mut b = Boundaries::all_pml();
    b.x = bx;
    b.y = by;
    Simulation::new([min[0], min[1], -0.5], [max[0], max[1], 0.5], MeshSpec::Uniform { dx: [dx; 3] }, b, run_time).two_d(2, pol)
}

fn pml2() -> [Boundary; 2] {
    [Boundary::pml(), Boundary::pml()]
}

/// The two field sets of a 2-D simulation: (label, polarization, E
/// component of a wave travelling along x).
fn sets_2d() -> [(&'static str, crate::spec::Polarization, Component); 2] {
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
        let peak = b[..n].iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
        let err = a[..n].iter().zip(&b[..n]).fold(0f32, |m, (x, y)| m.max((x - y).abs())) as f64;
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

pub fn all_cases() -> Vec<(&'static str, fn(Runner) -> Result<CaseReport>)> {
    vec![
        ("plane_wave_vacuum", plane_wave_vacuum),
        ("fresnel_normal", fresnel_normal),
        ("pec_pmc_reflection", pec_pmc_reflection),
        ("dipole_radiation", dipole_radiation),
        ("pml_reflection", pml_reflection),
        ("skin_depth", skin_depth),
        ("lorentz_slab", lorentz_slab),
        ("birefringence", birefringence),
        ("nonuniform_grid", nonuniform_grid),
        ("magnetic_dipole", magnetic_dipole),
        ("dipole_green_function", dipole_green_function),
        ("symmetry_quarter_domain", symmetry_quarter_domain),
        ("tfsf_leakage", tfsf_leakage),
        ("mie_scattering", mie_scattering),
        ("gaussian_beam", gaussian_beam),
        ("gaussian_beam_tilted", gaussian_beam_tilted),
        ("dispersive_propagation", dispersive_propagation),
        ("anisotropic_dispersion", anisotropic_dispersion),
        ("energy_conservation", energy_conservation),
        ("stored_energy", stored_energy),
        ("reciprocity", reciprocity),
        ("bloch_zero_vector", bloch_zero_vector),
        ("pulsed_vs_cw", pulsed_vs_cw),
        ("apodization_leakage", apodization_leakage),
        ("dipole_far_field", dipole_far_field),
        ("dipole_tilted", dipole_tilted),
        ("bloch_oblique", bloch_oblique),
        ("bloch_oblique_p", bloch_oblique_p),
        ("chi2_second_harmonic", chi2_second_harmonic),
        ("chi3_kerr", chi3_kerr),
        ("offdiagonal_tensor", offdiagonal_tensor),
        ("lossy_offdiagonal_tensor", lossy_offdiagonal_tensor),
        ("oriented_dispersion", oriented_dispersion),
        ("dispersive_background_sources", dispersive_background_sources),
        ("plane_wave_impedance", plane_wave_impedance),
        ("mixed_boundaries", mixed_boundaries),
        ("dipole_in_dielectric", dipole_in_dielectric),
        ("drude_reflection", drude_reflection),
        ("hyperbolic_drude", hyperbolic_drude),
        ("anisotropic_cells", anisotropic_cells),
        ("phasor_recording", phasor_recording),
        ("index_grid", index_grid),
        ("two_d_matches_one_cell", two_d_matches_one_cell),
        ("two_d_fresnel_oblique", two_d_fresnel_oblique),
        ("two_d_mie_cylinder", two_d_mie_cylinder),
        ("two_d_metal_cylinder", two_d_metal_cylinder),
        ("two_d_slab_mode", two_d_slab_mode),
        ("two_d_cpml", two_d_cpml),
        ("two_d_grating", two_d_grating),
        ("two_d_gaussian_beam", two_d_gaussian_beam),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mie_small_sphere_rayleigh_limit() {
        // Rayleigh: Qsca = 8/3 x^4 |(m^2-1)/(m^2+2)|^2
        let m = Complex64::new(1.5, 0.0);
        let x = 0.05;
        let (_, qs) = mie_q(m, x);
        let r = ((m * m - 1.0) / (m * m + 2.0)).norm_sqr();
        let ray = 8.0 / 3.0 * x.powi(4) * r;
        assert!((qs - ray).abs() / ray < 1e-3, "{qs} vs {ray}");
    }

    #[test]
    fn bessel_functions() {
        let (j, y) = (bessel_j(3, 1.0), bessel_y(3, 1.0));
        for (got, want) in [(j[0], 0.765_197_686_557_966_6), (j[1], 0.440_050_585_744_933_5), (j[2], 0.114_903_484_931_900_5), (y[0], 0.088_256_964_215_676_96), (y[1], -0.781_212_821_300_288_7), (y[2], -1.650_682_606_816_254_4)] {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        let (j, y) = (bessel_j(1, 2.5), bessel_y(1, 2.5));
        assert!((j[0] + 0.048_383_776_468_197_996).abs() < 1e-9 && (y[0] - 0.498_070_359_615_232_2).abs() < 1e-9, "{} {}", j[0], y[0]);
    }

    #[test]
    fn cylinder_series_limits() {
        // thin dielectric cylinder, E along the axis: Q = pi^2 x^3 (m^2 - 1)^2 / 8
        let (m, x) = (1.5f64, 0.02f64);
        let q = cylinder_qsca(Some(m), x, true);
        let ray = PI * PI * x.powi(3) * (m * m - 1.0).powi(2) / 8.0;
        assert!((q / ray - 1.0).abs() < 2e-3, "{q} vs {ray}");
        // E across it: Q = pi^2 x^3 ((m^2 - 1) / (m^2 + 1))^2 / 4
        let q = cylinder_qsca(Some(m), x, false);
        let ray = PI * PI * x.powi(3) * ((m * m - 1.0) / (m * m + 1.0)).powi(2) / 4.0;
        assert!((q / ray - 1.0).abs() < 2e-3, "{q} vs {ray}");
        // large conducting cylinder: Q -> 2 (geometric optics plus diffraction)
        let q = cylinder_qsca(None, 60.0, true);
        assert!((q - 2.0).abs() < 0.1, "{q}");
        // slab waveguide: a known single-mode slab, n_eff between the indices
        let ne = slab_neff(2.0, 1.45, 0.3e-6, TAU / 1e-6, true);
        let nm = slab_neff(2.0, 1.45, 0.3e-6, TAU / 1e-6, false);
        assert!(ne > nm && nm > 1.45 && ne < 2.0, "{ne} {nm}");
    }

    #[test]
    fn mie_known_value() {
        // m = 1.5, x = 1: Qsca ~ 0.2153 (Bohren & Huffman tables)
        let (_, qs) = mie_q(Complex64::new(1.5, 0.0), 1.0);
        assert!((qs - 0.2153).abs() < 2e-3, "{qs}");
    }
}
