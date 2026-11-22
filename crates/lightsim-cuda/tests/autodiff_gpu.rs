//! Gradient engines on the GPU (skipped without one): reverse mode against
//! finite differences and against the CPU engine, the persistent S-matrix
//! session against the CPU session, fused and split stepping.
//!
//! `LIGHTSIM_GPU` selects the device ordinal (default 0).

use lightsim_core::autodiff::engine::cpu_factory;
use lightsim_core::autodiff::testing::{Case, F0, Rng, plane_wave_session_spec, pml, reverse_scene, reverse_scene_tight, samples, session_spec};
use lightsim_core::autodiff::{Channel, GradientSession, ReverseSession, ReverseSpec};
use lightsim_core::materials::{Medium, Pole};
use lightsim_core::spec::{Boundaries, Boundary};
use lightsim_cuda::autodiff::factory;
use num_complex::Complex64;

fn device() -> Option<String> {
    if lightsim_cuda::device_count().unwrap_or(0) == 0 {
        eprintln!("skipped: no GPU");
        return None;
    }
    Some(format!("cuda:{}", std::env::var("LIGHTSIM_GPU").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0)))
}

fn reverse_check(c: &Case, dev: &str) {
    reverse_check_scene(c, dev, reverse_scene(c))
}

fn reverse_check_scene(c: &Case, dev: &str, (sim, grids, objective): (lightsim_core::Simulation, Vec<lightsim_core::autodiff::DesignGrid>, lightsim_core::autodiff::Objective)) {
    let spec = ReverseSpec { sim, grids: grids.clone(), objective };
    let mut s = ReverseSession::new(spec.clone(), dev, &factory).unwrap();
    let mut rng = Rng(11);
    let eps = vec![samples(&mut rng, grids[0].samples(), 2.5, 6.0)];
    s.set_design(&eps).unwrap();
    let (val, grad) = s.value_and_grad().unwrap();
    // a second pass on the same buffers gives the same numbers
    let (val2, grad2) = s.value_and_grad().unwrap();
    assert_eq!(val, val2);
    assert_eq!(grad, grad2);
    let d: Vec<f64> = (0..eps[0].len()).map(|_| 2.0 * rng.next() - 1.0).collect();
    let h = 0.04;
    let shifted = |sg: f64| vec![eps[0].iter().zip(&d).map(|(e, d)| e + sg * h * d).collect::<Vec<f64>>()];
    s.set_design(&shifted(1.0)).unwrap();
    let lp = s.forward().unwrap();
    s.set_design(&shifted(-1.0)).unwrap();
    let lm = s.forward().unwrap();
    let fd = (lp - lm) / (2.0 * h);
    let ad: f64 = grad[0].iter().zip(&d).map(|(g, d)| g * d).sum();
    // the same on the CPU engine
    let mut sc = ReverseSession::new(spec, "cpu", &cpu_factory).unwrap();
    sc.set_design(&eps).unwrap();
    let (vc, gc) = sc.value_and_grad().unwrap();
    let gmax = gc[0].iter().fold(0f64, |m, v| m.max(v.abs()));
    let gdiff = grad[0].iter().zip(&gc[0]).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / gmax;
    let rel = (ad - fd).abs() / fd.abs();
    println!(
        "{:22} [{}] L = {val:.4e} (CPU {:.1e})  dL adjoint {ad:.5e}  FD {fd:.5e}  rel {rel:.2e}  max |grad - CPU| / max {gdiff:.1e}",
        c.name,
        s.device(),
        (val - vc).abs() / vc.abs()
    );
    assert!(rel < c.tol, "{}: {rel}", c.name);
    assert!((val - vc).abs() < 1e-4 * vc.abs() && gdiff < 2e-3, "{}: GPU vs CPU {gdiff}", c.name);
}

#[test]
fn gpu_reverse_mode_matches_finite_differences_and_cpu() {
    let Some(dev) = device() else { return };
    let all_pml = Boundaries::all(pml(8));
    let walls = Boundaries { x: [Boundary::Pec, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Pmc, Boundary::Pec] };
    let periodic_x = Boundaries { x: [Boundary::Periodic, Boundary::Periodic], y: [Boundary::Pmc, Boundary::Pmc], z: [pml(8), pml(8)] };
    let mut lorentz = Medium::dielectric(2.5).with_name("lorentz");
    lorentz.poles = vec![Pole::Lorentz { delta_eps: 1.5, omega0: 2.0 * std::f64::consts::PI * 1.4 * F0, gamma: 2.0e14 }, Pole::Drude { omega_p: 2.0e15, gamma: 1.0e14 }];
    let mut debye = Medium::dielectric(2.0).with_name("debye");
    debye.poles = vec![Pole::Debye { delta_eps: 1.0, tau: 8.0e-15 }];
    let cases = [
        Case { name: "walls", boundaries: walls.clone(), design_sigma: 0.0, extra: None, tol: 3e-3 },
        Case { name: "PML", boundaries: all_pml.clone(), design_sigma: 0.0, extra: None, tol: 3e-3 },
        Case { name: "periodic x, PMC y, PML z", boundaries: periodic_x, design_sigma: 0.0, extra: None, tol: 3e-3 },
        Case { name: "PML, lossy design", boundaries: all_pml.clone(), design_sigma: 3000.0, extra: None, tol: 3e-3 },
        Case { name: "PML, Lorentz + Drude", boundaries: all_pml.clone(), design_sigma: 0.0, extra: Some(lorentz), tol: 3e-3 },
        Case { name: "walls, Debye", boundaries: walls, design_sigma: 0.0, extra: Some(debye), tol: 3e-3 },
    ];
    for c in &cases {
        reverse_check(c, &dev);
    }
    // design voxels right up to the absorbing layers (transposed CPML)
    let tight = Case { name: "CPML next to design", boundaries: all_pml.clone(), design_sigma: 0.0, extra: None, tol: 5e-4 };
    reverse_check_scene(&tight, &dev, reverse_scene_tight(&tight));
    // chi2 + chi3 medium behind the design: implicit step, linearised adjoint
    let mut kerr = Medium::dielectric(4.0).with_name("kerr");
    kerr.chi2 = [1.5; 3];
    kerr.chi3 = [30.0; 3];
    let walls = Boundaries { x: [Boundary::Pec, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Pmc, Boundary::Pec] };
    reverse_check(&Case { name: "walls, chi2 + chi3", boundaries: walls, design_sigma: 0.0, extra: Some(kerr.clone()), tol: 3e-3 }, &dev);
    reverse_check(&Case { name: "PML, chi2 + chi3", boundaries: Boundaries::all(pml(8)), design_sigma: 0.0, extra: Some(kerr), tol: 3e-3 }, &dev);
    // the split kernels give the same gradient as the fused step
    // (LIGHTSIM_FUSED is read when the backend is built)
    unsafe { std::env::set_var("LIGHTSIM_FUSED", "0") };
    reverse_check(&Case { name: "PML (split kernels)", boundaries: all_pml, design_sigma: 0.0, extra: None, tol: 3e-3 }, &dev);
    unsafe { std::env::remove_var("LIGHTSIM_FUSED") };
}

#[test]
fn gpu_session_matches_cpu_session() {
    let Some(dev) = device() else { return };
    let spec = session_spec();
    let mut rng = Rng(5);
    let e1: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let e2: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let g: Vec<Complex64> = (0..6).map(|i| Complex64::new(0.3 + 0.1 * i as f64, -0.2 + 0.05 * i as f64)).collect();
    let mut cpu = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
    cpu.set_design(&e1).unwrap();
    let sc = cpu.forward(true).unwrap();
    let gc = cpu.backward(&g).unwrap();
    let compare = |name: &str, s: &[Complex64], gr: &[Vec<f64>]| {
        let ds = s.iter().zip(&sc).fold(0f64, |m, (a, b)| m.max((a - b).norm() / b.norm().max(1e-3)));
        let gmax = gc.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
        let dg = gr.iter().flatten().zip(gc.iter().flatten()).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / gmax;
        println!("{name}: max |S - S_cpu| / |S_cpu| = {ds:.1e}, max |grad - grad_cpu| / max = {dg:.1e}");
        assert!(ds < 2e-4 && dg < 2e-3, "{name}: {ds} {dg}");
    };
    let mut gpu = GradientSession::new(spec.clone(), &[dev.clone()], &factory).unwrap();
    println!("session on {:?}", gpu.devices());
    gpu.set_design(&e1).unwrap();
    let s1 = gpu.forward(false).unwrap();
    let g1 = gpu.backward(&g).unwrap();
    compare("one GPU", &s1, &g1);
    // another design and back: the persistent buffers carry nothing over
    gpu.set_design(&e2).unwrap();
    let s2 = gpu.forward(true).unwrap();
    assert!(s1.iter().zip(&s2).any(|(a, b)| (a - b).norm() > 1e-3));
    gpu.set_design(&e1).unwrap();
    let s1b = gpu.forward(true).unwrap();
    let g1b = gpu.backward(&g).unwrap();
    assert_eq!(s1, s1b);
    assert_eq!(g1, g1b);
    // the design update is what a fresh session computes
    let mut fresh = GradientSession::new(spec.clone(), &[dev.clone()], &factory).unwrap();
    fresh.set_design(&e2).unwrap();
    assert_eq!(fresh.forward(false).unwrap(), s2);
    // mixed device list: runs dispatched over a GPU and the CPU, fields
    // gathered on the first engine for the product
    let mut mixed = GradientSession::new(spec.clone(), &[dev.clone(), "cpu".to_string()], &factory).unwrap();
    mixed.set_design(&e1).unwrap();
    let sm = mixed.forward(true).unwrap();
    let gm = mixed.backward(&g).unwrap();
    compare("GPU + CPU", &sm, &gm);
    // two engines on the same GPU, driven from two threads: the dispatch
    // and gather path of a device list, on one device
    let mut twin = GradientSession::new(spec.clone(), &[dev.clone(), dev.clone()], &factory).unwrap();
    twin.set_design(&e1).unwrap();
    let sw = twin.forward(true).unwrap();
    let gw = twin.backward(&g).unwrap();
    assert_eq!(twin.stats.iter().map(|r| r.device.clone()).collect::<std::collections::BTreeSet<_>>().len(), 1);
    compare("two engines, one GPU", &sw, &gw);
    assert_eq!(sw, s1);
    if lightsim_cuda::device_count().unwrap_or(0) >= 2 && std::env::var("LIGHTSIM_GPU").is_err() {
        let mut two = GradientSession::new(spec.clone(), &["cuda:0".to_string(), "cuda:1".to_string()], &factory).unwrap();
        two.set_design(&e1).unwrap();
        let st = two.forward(true).unwrap();
        let gt = two.backward(&g).unwrap();
        compare("two GPUs", &st, &gt);
    }
    // split kernels
    unsafe { std::env::set_var("LIGHTSIM_FUSED", "0") };
    let mut split = GradientSession::new(spec, &[dev], &factory).unwrap();
    unsafe { std::env::remove_var("LIGHTSIM_FUSED") };
    split.set_design(&e1).unwrap();
    let ss = split.forward(false).unwrap();
    let gs = split.backward(&g).unwrap();
    compare("split kernels", &ss, &gs);
}

#[test]
fn gpu_session_monitors_share_one_stride() {
    // The engine accumulates the design-region fields and the port monitors
    // of a run in one launch with one phasor table. They were compiled with
    // different strides (design fields at every step, port monitors at the
    // automatic stride), and the session was refused. With one stride for
    // both: the automatic stride gives the numbers of sampling every step.
    let Some(dev) = device() else { return };
    let mut spec = session_spec();
    spec.sim.run_time = 400e-15; // until the fields have decayed
    spec.sim.shutoff = Some(1e-8);
    let mut rng = Rng(11);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let g: Vec<Complex64> = (0..6).map(|i| Complex64::new(0.3 + 0.1 * i as f64, -0.2 + 0.05 * i as f64)).collect();
    let run = |spec: &lightsim_core::autodiff::SessionSpec| {
        let mut s = GradientSession::new(spec.clone(), &[dev.clone()], &factory).unwrap();
        s.set_design(&eps).unwrap();
        let sm = s.forward(true).unwrap();
        let gr = s.backward(&g).unwrap();
        (s.dft_stride(), sm, gr)
    };
    let (stride, s_auto, g_auto) = run(&spec);
    assert!(stride > 1, "a Gaussian pulse allows a stride above 1, got {stride}");
    let mut every = spec.clone();
    every.sim.dft_stride = Some(1);
    let (one, s_one, g_one) = run(&every);
    assert_eq!(one, 1);
    let scale = s_one.iter().fold(0f64, |m, z| m.max(z.norm()));
    let ds = s_auto.iter().zip(&s_one).fold(0f64, |m, (a, b)| m.max((a - b).norm())) / scale;
    let gmax = g_one.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let dg = g_auto.iter().flatten().zip(g_one.iter().flatten()).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / gmax;
    println!("GPU session, stride {stride} against every step: S {ds:.1e}, gradient {dg:.1e}");
    assert!(ds < 2e-4 && dg < 5e-4, "{ds} {dg}");
}

#[test]
fn gpu_session_with_two_ports_on_one_plane() {
    // Two ports with the same plane and window have identical monitor boxes;
    // each keeps its own accumulators in a session (see the CPU test of the
    // same name). The second carries the profile of the first with the
    // opposite sign: minus its amplitudes and gradients.
    let Some(dev) = device() else { return };
    let mut spec = session_spec();
    let mut flipped = spec.ports[1].clone();
    flipped.name = "out_flipped".into();
    for q in flipped.profiles.iter_mut() {
        for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
            arr.iter_mut().for_each(|v| *v = -*v);
        }
    }
    spec.ports.push(flipped);
    spec.outputs = vec![Channel::Port { port: 1, mode: 0 }, Channel::Port { port: 2, mode: 0 }];
    let mut rng = Rng(12);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let mut s = GradientSession::new(spec, &[dev], &factory).unwrap();
    s.set_design(&eps).unwrap();
    let sm = s.forward(true).unwrap();
    for k in 0..3 {
        let (a, b) = (sm[k], sm[3 + k]);
        assert!(a.norm() > 1e-3, "{a}");
        assert!((b + a).norm() < 1e-5 * a.norm(), "frequency {k}: {b} against minus {a}");
    }
    let unit = |m: usize| (0..6).map(|i| if i / 3 == m { Complex64::new(1.0, 0.5) } else { Complex64::new(0.0, 0.0) }).collect::<Vec<_>>();
    let ga = s.backward(&unit(0)).unwrap();
    let gb = s.backward(&unit(1)).unwrap();
    let gmax = ga.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    assert!(gmax > 0.0);
    let d = gb.iter().flatten().zip(ga.iter().flatten()).fold(0f64, |m, (b, a)| m.max((b + a).abs())) / gmax;
    assert!(d < 1e-4, "gradient of the second port against minus the first: {d}");
}

// ----------------------------------------------------------------------------
// FDTDX gradient tests, mirrored (see the CPU tests of the same names in
// lightsim-core/tests/autodiff.rs): reverse mode on the GPU against central
// differences and against the CPU engine, and the two gradient routes on one
// objective.
// ----------------------------------------------------------------------------

use lightsim_core::autodiff::reverse::Series;
use lightsim_core::autodiff::{DesignGrid, Objective, PortSpec};
use lightsim_core::engine::{RawDft, RawResults};
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::OrientedPole;
use lightsim_core::monitors::Monitor;
use lightsim_core::plan::{CompileOptions, compile};
use lightsim_core::results::{assemble, finalize};
use lightsim_core::sources::{Direction, Source, TimeSignal};
use lightsim_core::Simulation;
use std::f64::consts::TAU;

const UM: f64 = 1e-6;
type Loss = Box<dyn Fn(&[Series], f64) -> (f64, Vec<Vec<f64>>)>;

fn evaluate(s: &mut ReverseSession, eps: &[Vec<f64>], loss: Option<&Loss>, grad: bool) -> (f64, Option<Vec<Vec<f64>>>) {
    s.set_design(eps).unwrap();
    let v = s.forward().unwrap();
    match loss {
        None => (v, if grad { Some(s.backward().unwrap()) } else { None }),
        Some(f) => {
            let (l, cot) = f(&s.series().unwrap(), s.dt());
            (l, if grad { Some(s.backward_with(&cot).unwrap()) } else { None })
        }
    }
}

fn axpy(eps: &[Vec<f64>], h: f64, d: &[Vec<f64>]) -> Vec<Vec<f64>> {
    eps.iter().zip(d).map(|(e, d)| e.iter().zip(d).map(|(e, d)| e + h * d).collect()).collect()
}

fn dot(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| x * y).sum()
}

fn max_rel(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let scale = b.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    a.iter().flatten().zip(b.iter().flatten()).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

/// GPU reverse mode against central differences (GPU forward runs) along a
/// random direction of all design samples, and against the CPU engine.
/// Returns the GPU gradient.
fn gpu_check(name: &str, dev: &str, sim: Simulation, grids: Vec<DesignGrid>, objective: Objective, loss: Option<Loss>, tol: f64, seed: u64) -> Vec<Vec<f64>> {
    let spec = ReverseSpec { sim, grids: grids.clone(), objective };
    let mut s = ReverseSession::new(spec.clone(), dev, &factory).unwrap();
    let mut rng = Rng(seed);
    let eps: Vec<Vec<f64>> = grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let (val, grad) = evaluate(&mut s, &eps, loss.as_ref(), true);
    let grad = grad.unwrap();
    assert!(val.is_finite() && val != 0.0 && grad.iter().flatten().all(|g| g.is_finite()), "{name}: L = {val}");
    let d: Vec<Vec<f64>> = eps.iter().map(|e| e.iter().map(|_| 2.0 * rng.next() - 1.0).collect()).collect();
    let h = 0.04;
    let fd = (evaluate(&mut s, &axpy(&eps, h, &d), loss.as_ref(), false).0 - evaluate(&mut s, &axpy(&eps, -h, &d), loss.as_ref(), false).0) / (2.0 * h);
    let ad = dot(&grad, &d);
    let mut sc = ReverseSession::new(spec, "cpu", &cpu_factory).unwrap();
    let (vc, gc) = evaluate(&mut sc, &eps, loss.as_ref(), true);
    let gdiff = max_rel(&grad, &gc.unwrap());
    let rel = (ad - fd).abs() / fd.abs();
    println!("{name:44} [{}] L = {val:.4e} (CPU {:.1e})  dL reverse mode {ad:.5e}  FD {fd:.5e}  rel {rel:.2e}  max |grad - CPU| / max {gdiff:.1e}", s.device(), (val - vc).abs() / vc.abs());
    assert!(rel < tol, "{name}: {rel}");
    assert!((val - vc).abs() < 1e-4 * vc.abs() && gdiff < 2e-3, "{name}: GPU vs CPU {gdiff}");
    grad
}

#[derive(Clone)]
struct Window {
    freqs: Vec<f64>,
    gate: Option<(f64, f64)>,
    stride: usize,
    tukey: Option<(f64, f64)>,
}

impl Window {
    fn all(freqs: Vec<f64>) -> Window {
        Window { freqs, gate: None, stride: 1, tukey: None }
    }
    fn weight(&self, n: usize, t: f64, dt: f64) -> f64 {
        if n % self.stride != 0 || self.gate.is_some_and(|(a, b)| t < a || t > b) {
            return 0.0;
        }
        let mut w = dt * self.stride as f64;
        if let Some((a, b)) = self.tukey {
            let (x, e) = ((t - a) / (b - a), 0.25);
            w *= if !(0.0..=1.0).contains(&x) {
                0.0
            } else if x < e {
                0.5 * (1.0 - (std::f64::consts::PI * x / e).cos())
            } else if x > 1.0 - e {
                0.5 * (1.0 - (std::f64::consts::PI * (1.0 - x) / e).cos())
            } else {
                1.0
            };
        }
        w
    }
}

fn phasors(win: &Window, comp: Component, ns: usize, dt: f64) -> Vec<Vec<Complex64>> {
    let off = if comp.is_e() { 1.0 } else { 0.5 };
    (0..ns).map(|n| {
        let t = (n as f64 + off) * dt;
        let w = win.weight(n, t, dt);
        win.freqs.iter().map(|f| Complex64::from_polar(w, TAU * f * t)).collect()
    }).collect()
}

fn dft(s: &Series, ph: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let m: usize = s.size.iter().product();
    let nf = ph.first().map(|p| p.len()).unwrap_or(0);
    let mut out = vec![vec![Complex64::new(0.0, 0.0); nf]; m];
    for (n, p) in ph.iter().enumerate() {
        for (t, o) in out.iter_mut().enumerate() {
            let v = s.data[n * m + t] as f64;
            for k in 0..nf {
                o[k] += v * p[k];
            }
        }
    }
    out
}

fn phasor_power(win: Window) -> Loss {
    Box::new(move |series: &[Series], dt: f64| {
        let (mut l, mut cot) = (0.0, vec![]);
        for s in series {
            let m: usize = s.size.iter().product();
            let ns = s.data.len() / m;
            let ph = phasors(&win, s.comp, ns, dt);
            let f = dft(s, &ph);
            l += f.iter().flatten().map(|z| z.norm_sqr()).sum::<f64>();
            let mut c = vec![0.0; ns * m];
            for (n, p) in ph.iter().enumerate() {
                for t in 0..m {
                    c[n * m + t] = f[t].iter().zip(p).map(|(z, q)| 2.0 * (z.conj() * q).re).sum();
                }
            }
            cot.push(c);
        }
        (l, cot)
    })
}

fn phasor_plane() -> Objective {
    Objective::Fields { min: [1.25 * UM, 0.3 * UM, 0.3 * UM], max: [1.25 * UM, 0.9 * UM, 0.7 * UM], components: vec![Component::Ez, Component::Ey, Component::Hy] }
}

fn base_case(boundaries: Boundaries) -> Case {
    Case { name: "", boundaries, design_sigma: 0.0, extra: None, tol: 3e-3 }
}

fn slab(sim: &mut Simulation, m: Medium) {
    let name = m.name.clone();
    sim.media.push(m);
    sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.05 * UM, -1.0, -1.0], max: [1.15 * UM, 1.0, 1.0] }, medium: name });
}

fn graded_nodes(l: f64, n: usize, phase: f64) -> Vec<f64> {
    let w: Vec<f64> = (0..n).map(|i| 1.0 + 0.25 * (0.7 * i as f64 + phase).sin()).collect();
    let s: f64 = w.iter().sum();
    let mut x = vec![0.0];
    for v in &w {
        x.push(x.last().unwrap() + v / s * l);
    }
    *x.last_mut().unwrap() = l;
    x
}

#[test]
fn gpu_reverse_mode_fdtdx_parity_scenes() {
    // phasor objectives and windows (FDTDX test_dipole_*, test_exact_*,
    // test_frequencies_sharing_one_adjoint_run, test_mixed_cw_and_pulsed_sources_raise),
    // graded mesh (test_nonuniform_grid), anisotropic and conductive media
    // (test_anisotropic_and_conductive_background, at conductivities reverse
    // mode can reverse), per-axis dispersion (TestPerAxisDispersiveGradientCheckpointed),
    // final fields in an all-periodic box (TestReversibleFdtdGradients),
    // two adjacent grids and a uniform block (test_several_devices_with_overlapping_halos,
    // test_gradient_regions_device_and_object)
    let Some(dev) = device() else { return };
    let (sim, grids, flux) = reverse_scene(&base_case(Boundaries::all(pml(8))));
    let t1 = sim.run_time;
    let windows: Vec<(&str, Window)> = vec![
        ("phasor power, three frequencies", Window::all(vec![0.9 * F0, F0, 1.1 * F0])),
        ("phasor power, twelve dense frequencies", Window::all((0..12).map(|i| F0 * (0.85 + 0.3 * i as f64 / 11.0)).collect())),
        ("phasor power, gated + every 3rd + Tukey", Window { freqs: vec![0.9 * F0, 1.1 * F0], gate: Some((0.15 * t1, 0.92 * t1)), stride: 3, tukey: Some((0.2 * t1, 0.85 * t1)) }),
    ];
    for (name, win) in windows {
        gpu_check(name, &dev, sim.clone(), grids.clone(), phasor_plane(), Some(phasor_power(win)), 3e-3, 21);
    }
    let mut mixed = sim.clone();
    let cw = TimeSignal::ContinuousWave { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    mixed.sources.push(Source::PointDipole { name: "cw".into(), position: [0.3 * UM, 0.55 * UM, 0.45 * UM], polarization: Component::Ey, signal: cw, amplitude: 1e-15 });
    let win = Window { freqs: vec![F0], gate: Some((0.5 * t1, t1)), stride: 1, tukey: None };
    gpu_check("phasor power, CW + pulsed sources", &dev, mixed, grids.clone(), phasor_plane(), Some(phasor_power(win)), 3e-3, 22);

    let mut graded = sim.clone();
    let m = graded.region.max;
    graded.mesh = MeshSpec::Explicit { x: graded_nodes(m[0], 32, 0.0), y: graded_nodes(m[1], 24, 1.0), z: graded_nodes(m[2], 20, 2.0) };
    gpu_check("graded mesh, flux energy", &dev, graded, grids.clone(), flux.clone(), None, 3e-3, 23);

    let mut aniso = sim.clone();
    aniso.media.iter_mut().find(|m| m.name == "ox").unwrap().eps = [2.0, 2.4, 2.2];
    let mut lossy = Medium::dielectric(2.0).with_name("lossy");
    lossy.eps = [2.0, 3.0, 2.5];
    lossy.sigma = [1e4, 6e3, 1.5e4];
    slab(&mut aniso, lossy);
    gpu_check("anisotropic + conductive, flux energy", &dev, aniso, grids.clone(), flux.clone(), None, 3e-3, 25);

    let mut disp_sim = sim.clone();
    let mut disp = Medium::dielectric(2.0).with_name("disp");
    disp.eps = [2.0, 2.5, 3.0];
    disp.sigma = [100.0; 3];
    disp.poles = vec![Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 2.0e14 }];
    disp.pole_axes = [true, false, true];
    slab(&mut disp_sim, disp);
    gpu_check("per-axis Lorentz slab, flux energy", &dev, disp_sim, grids.clone(), flux.clone(), None, 3e-3, 27);

    let (periodic, pgrids, _) = reverse_scene(&base_case(Boundaries::all(Boundary::Periodic)));
    let obj = Objective::Fields { min: [1.0 * UM, 0.2 * UM, 0.2 * UM], max: [1.4 * UM, 1.0 * UM, 0.8 * UM], components: Component::ALL.to_vec() };
    let last: Loss = Box::new(|series: &[Series], _dt: f64| {
        let (mut l, mut cot) = (0.0, vec![]);
        for s in series {
            let m: usize = s.size.iter().product();
            let ns = s.data.len() / m;
            let mut c = vec![0.0; ns * m];
            for t in 0..m {
                let v = s.data[(ns - 1) * m + t] as f64;
                l += v * v;
                c[(ns - 1) * m + t] = 2.0 * v;
            }
            cot.push(c);
        }
        (l, cot)
    });
    gpu_check("all periodic, sum E^2 + H^2 at the end", &dev, periodic, pgrids, obj, Some(last), 3e-3, 28);

    let grid = |name: &str, min: [f64; 3], max: [f64; 3], dims: [usize; 3]| DesignGrid { name: name.into(), min, max, dims, medium: "design".into(), permeability: false };
    let three = vec![
        grid("lower", [0.52 * UM, 0.33 * UM, 0.3 * UM], [0.93 * UM, 0.91 * UM, 0.5 * UM], [5, 6, 1]),
        grid("upper", [0.52 * UM, 0.33 * UM, 0.5 * UM], [0.93 * UM, 0.91 * UM, 0.7 * UM], [4, 5, 2]),
        grid("block", [1.0 * UM, 0.4 * UM, 0.3 * UM], [1.2 * UM, 0.8 * UM, 0.7 * UM], [1, 1, 1]),
    ];
    gpu_check("two adjacent grids + block", &dev, sim, three, flux, None, 3e-3, 29);
}

#[test]
fn gpu_reverse_mode_refuses_what_it_cannot_differentiate() {
    // FDTDX TestOrientedDispersionGradients (a gap: refused, not computed)
    // and test_gradient_region_overlapping_a_source_raises
    let Some(dev) = device() else { return };
    let (mut sim, grids, flux) = reverse_scene(&base_case(Boundaries::all(pml(8))));
    let mut src_on_design = sim.clone();
    if let Source::PointDipole { position, .. } = &mut src_on_design.sources[0] {
        *position = [0.7 * UM, 0.6 * UM, 0.525 * UM];
    }
    let err = ReverseSession::new(ReverseSpec { sim: src_on_design, grids: grids.clone(), objective: flux.clone() }, &dev, &factory).err().expect("a source on the design must be refused");
    assert!(err.to_string().contains("design"), "{err}");
    let mut m = Medium::dielectric(2.0).with_name("oriented");
    m.oriented = vec![OrientedPole { pole: Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 1e13 }, u: [1.0, 1.0, 0.0] }];
    slab(&mut sim, m);
    let err = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, &dev, &factory).err().expect("an oriented pole must be refused");
    assert!(err.to_string().contains("off-diagonal"), "{err}");
}

fn mirror_scene(signs: (f64, f64), half: Option<Boundary>) -> (Simulation, Vec<DesignGrid>, Objective) {
    let dx = 0.05 * UM;
    let (ylo, b) = match &half {
        Some(w) => (0.6 * UM, Boundaries { x: [pml(8), pml(8)], y: [w.clone(), pml(8)], z: [pml(8), pml(8)] }),
        None => (0.0, Boundaries::all(pml(8))),
    };
    let mut sim = Simulation::new([0.0, ylo, 0.0], [1.6 * UM, 1.2 * UM, 1.0 * UM], MeshSpec::Uniform { dx: [dx; 3] }, b, 28e-15);
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("core"));
    sim.media.push(Medium::dielectric(6.0).with_name("design"));
    sim.background = "ox".into();
    sim.structures.push(Structure { name: "wg".into(), shape: Shape::Box { min: [-1.0, 0.4 * UM, 0.3 * UM], max: [1.0, 0.8 * UM, 0.7 * UM] }, medium: "core".into() });
    let sig = TimeSignal::GaussianPulse { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    for (y, a) in [(0.5 * UM, signs.0), (0.7 * UM, signs.1)] {
        if y > ylo {
            sim.sources.push(Source::PointDipole { name: format!("s{y}"), position: [0.3 * UM, y, 0.525 * UM], polarization: Component::Ez, signal: sig.clone(), amplitude: 1e-15 * a });
        }
    }
    let grid = |name: &str, y0: f64, y1: f64| DesignGrid { name: name.into(), min: [0.52 * UM, y0, 0.3 * UM], max: [0.93 * UM, y1, 0.7 * UM], dims: [5, 4, 2], medium: "design".into(), permeability: false };
    let mut grids = vec![grid("upper", 0.63 * UM, 0.93 * UM)];
    if half.is_none() {
        grids.push(grid("lower", 0.27 * UM, 0.57 * UM));
    }
    let obj = Objective::Flux { axis: 0, position: 1.35 * UM, min: [1.35 * UM, ylo.max(0.2 * UM), 0.1 * UM], max: [1.35 * UM, 1.0 * UM, 0.9 * UM], t_start: Some(6e-15), t_end: Some(24e-15), taper: Some(3e-15) };
    (sim, grids, obj)
}

fn flip_y(v: &[f64]) -> Vec<f64> {
    let (ny, nz) = (4, 2);
    (0..v.len()).map(|t| v[((t / (ny * nz)) * ny + (ny - 1 - (t / nz) % ny)) * nz + t % nz]).collect()
}

#[test]
fn gpu_reverse_mode_with_mirror_symmetry() {
    // FDTDX test_mirror_symmetry: half domain with a PMC / PEC wall against
    // the full domain with two mirrored grids, on the GPU
    let Some(dev) = device() else { return };
    for (name, signs, wall) in [("PMC (even)", (1.0, 1.0), Boundary::Pmc), ("PEC (odd)", (-1.0, 1.0), Boundary::Pec)] {
        let mut rng = Rng(30);
        let (sim, grids, obj) = mirror_scene(signs, None);
        let upper = samples(&mut rng, grids[0].samples(), 2.5, 6.0);
        let mut full = ReverseSession::new(ReverseSpec { sim, grids, objective: obj }, &dev, &factory).unwrap();
        let (lf, gf) = evaluate(&mut full, &[upper.clone(), flip_y(&upper)], None, true);
        let gf = gf.unwrap();
        let (sim, grids, obj) = mirror_scene(signs, Some(wall.clone()));
        gpu_check(&format!("mirror {name}, half domain"), &dev, sim.clone(), grids.clone(), obj.clone(), None, 3e-3, 31);
        let mut half = ReverseSession::new(ReverseSpec { sim, grids, objective: obj }, &dev, &factory).unwrap();
        let (lh, gh) = evaluate(&mut half, &[upper.clone()], None, true);
        let d_half = max_rel(&gh.unwrap()[..1], &gf[..1]);
        let d_mirror = max_rel(&[flip_y(&gf[1])], &gf[..1]);
        let dl = (2.0 * lh - lf).abs() / lf.abs();
        println!("mirror {name} [{dev}]: L full {lf:.5e}, 2 L half {:.5e} (rel {dl:.1e}); max |grad half - grad full| / max {d_half:.1e}; mirrored grids {d_mirror:.1e}", 2.0 * lh);
        assert!(dl < 1e-4 && d_half < 2e-3 && d_mirror < 2e-3, "{name}: {dl} {d_half} {d_mirror}");
    }
}

/// Frequency-domain flux through a plane of uniform cells of area `da`
/// (boxes of a flux objective), summed over the frequencies.
fn phasor_flux(win: Window, da: f64) -> Loss {
    Box::new(move |series: &[Series], dt: f64| {
        let m: usize = series[0].size.iter().product();
        let ns = series[0].data.len() / m;
        let nf = win.freqs.len();
        let ph: Vec<Vec<Vec<Complex64>>> = series.iter().map(|s| phasors(&win, s.comp, ns, dt)).collect();
        let f: Vec<Vec<Vec<Complex64>>> = series.iter().zip(&ph).map(|(s, p)| dft(s, p)).collect();
        let mut cot = vec![vec![0.0; ns * m]; 6];
        let mut l = 0.0;
        let c = 0.5 * da / lightsim_core::constants::ETA0;
        for (e, h0, h1, sign) in [(0usize, 3usize, 5usize, 1.0), (1, 2, 4, -1.0)] {
            for t in 0..m {
                for k in 0..nf {
                    let hb = 0.5 * (f[h0][t][k] + f[h1][t][k]);
                    l += sign * c * (f[e][t][k] * hb.conj()).re;
                    for n in 0..ns {
                        cot[e][n * m + t] += sign * c * (ph[e][n][k] * hb.conj()).re;
                        let dh = sign * c * 0.5 * (f[e][t][k] * ph[h0][n][k].conj()).re;
                        cot[h0][n * m + t] += dh;
                        cot[h1][n * m + t] += dh;
                    }
                }
            }
        }
        (l, cot)
    })
}

#[test]
fn gpu_reverse_mode_periodic_cell_plane_wave_flux() {
    // FDTDX test_periodic_plane_source_and_flux
    let Some(dev) = device() else { return };
    let dx = 0.05 * UM;
    let b = Boundaries { x: [Boundary::Periodic, Boundary::Periodic], y: [Boundary::Periodic, Boundary::Periodic], z: [pml(8), pml(8)] };
    let mut sim = Simulation::new([0.0; 3], [0.4 * UM, 0.4 * UM, 1.2 * UM], MeshSpec::Uniform { dx: [dx; 3] }, b, 40e-15);
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("design"));
    sim.background = "ox".into();
    let sig = TimeSignal::GaussianPulse { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 0.15 * UM, direction: Direction::Plus, polarization: Component::Ex, signal: sig, amplitude: 1.0, angle: 0.0, min: None, max: None });
    let grids = vec![DesignGrid { name: "g".into(), min: [0.0, 0.0, 0.45 * UM], max: [0.4 * UM, 0.4 * UM, 0.6 * UM], dims: [4, 4, 2], medium: "design".into(), permeability: false }];
    let flux = Objective::Flux { axis: 2, position: 0.9 * UM, min: [0.0, 0.0, 0.9 * UM], max: [0.4 * UM, 0.4 * UM, 0.9 * UM], t_start: None, t_end: None, taper: None };
    gpu_check("periodic cell, plane wave, phasor flux", &dev, sim.clone(), grids.clone(), flux.clone(), Some(phasor_flux(Window::all(vec![0.9 * F0, 1.1 * F0]), dx * dx)), 3e-3, 32);
    gpu_check("periodic cell, plane wave, flux energy", &dev, sim, grids, flux, None, 3e-3, 33);
}

// --- the two gradient routes on one objective (see the CPU test) ---

/// `sum_k |a_k / N_k|^2` from the samples of the output plane, with the
/// core's mode overlap (linear in the DFT fields; weights found by probing).
fn port_loss(sim: &Simulation, port: &PortSpec, freqs: &[f64], norm: Vec<Complex64>, series: &[Series]) -> Loss {
    let mut s = sim.clone();
    s.sources.clear();
    s.monitors = vec![Monitor::Mode { name: "out".into(), axis: port.axis, position: port.position, min: port.min, max: port.max, freqs: freqs.to_vec(), modes: port.profiles.clone(), apodization: Default::default() }];
    let plan = compile(&s, &CompileOptions::default()).unwrap();
    let boxes = plan.dft[0].boxes.clone();
    for (b, se) in boxes.iter().zip(series) {
        assert!(b.comp == se.comp && b.glo == se.origin && b.gsize() == se.size);
    }
    let nf = freqs.len();
    let minus = port.direction == Direction::Minus;
    let amps = |raw: RawDft| -> Vec<Complex64> {
        let mut res = assemble(&[&plan], &[RawResults { dft: vec![raw], time: vec![] }]);
        finalize(&mut res, &plan);
        let m = &res.modes["out"];
        let (re, im) = if minus { (&m.amp_plus_re[0], &m.amp_plus_im[0]) } else { (&m.amp_minus_re[0], &m.amp_minus_im[0]) };
        (0..nf).map(|k| Complex64::new(re[k], im[k])).collect()
    };
    let zero: Vec<(Vec<f32>, Vec<f32>)> = boxes.iter().map(|b| (vec![0f32; nf * b.lcells()], vec![0f32; nf * b.lcells()])).collect();
    let w: Vec<Vec<Vec<Complex64>>> = boxes
        .iter()
        .enumerate()
        .map(|(bi, b)| {
            (0..b.lcells())
                .map(|t| {
                    let mut raw = zero.clone();
                    for k in 0..nf {
                        raw[bi].0[k * b.lcells() + t] = 1.0;
                    }
                    amps(RawDft { boxes: raw })
                })
                .collect()
        })
        .collect();
    let freqs = freqs.to_vec();
    Box::new(move |series: &[Series], dt: f64| {
        let win = Window::all(freqs.clone());
        let ph: Vec<Vec<Vec<Complex64>>> = series.iter().map(|s| phasors(&win, s.comp, s.data.len() / s.size.iter().product::<usize>(), dt)).collect();
        let f: Vec<Vec<Vec<Complex64>>> = series.iter().zip(&ph).map(|(s, p)| dft(s, p)).collect();
        let sv: Vec<Complex64> = (0..nf).map(|k| w.iter().zip(&f).map(|(wb, fb)| wb.iter().zip(fb).map(|(a, z)| a[k] * z[k]).sum::<Complex64>()).sum::<Complex64>() / norm[k]).collect();
        let cot = w
            .iter()
            .zip(&ph)
            .map(|(wb, pb)| {
                let m = wb.len();
                let mut c = vec![0.0; pb.len() * m];
                for (n, p) in pb.iter().enumerate() {
                    for (t, a) in wb.iter().enumerate() {
                        c[n * m + t] = (0..nf).map(|k| 2.0 * (sv[k].conj() * a[k] * p[k] / norm[k]).re).sum();
                    }
                }
                c
            })
            .collect();
        (sv.iter().map(|z| z.norm_sqr()).sum(), cot)
    })
}

#[test]
fn gpu_reverse_mode_and_session_agree_on_one_objective() {
    // FDTDX TestReversibleVsCheckpointedGradient (the two routes of lightsim
    // on one objective, see the CPU test) and test_sharded_over_two_devices:
    // the session on two GPUs gives the gradient of one GPU. (The session
    // dispatches its runs over the devices; no gradient engine decomposes
    // one simulation over several GPUs.)
    let Some(dev) = device() else { return };
    let spec = plane_wave_session_spec();
    let mut rng = Rng(34);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let d: Vec<Vec<f64>> = eps.iter().map(|e| e.iter().map(|_| 2.0 * rng.next() - 1.0).collect()).collect();
    let run_session = |devs: &[String]| {
        let mut s = GradientSession::new(spec.clone(), devs, &factory).unwrap();
        s.set_design(&eps).unwrap();
        let sm = s.forward(true).unwrap();
        let g = s.backward(&sm.iter().map(|z| 2.0 * z).collect::<Vec<_>>()).unwrap();
        (s, sm, g)
    };
    let (_, s_sess, g_sess) = run_session(&[dev.clone()]);
    let p = &spec.ports[0];
    let mut sim = spec.sim.clone();
    sim.sources = vec![Source::Mode { name: "__src__".into(), axis: p.axis, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[0].clone(), signal: spec.signal.clone(), amplitude: 1.0 }];
    let q = spec.ports[1].clone();
    let objective = Objective::Flux { axis: q.axis, position: q.position, min: q.min, max: q.max, t_start: None, t_end: None, taper: None };
    let mut rev = ReverseSession::new(ReverseSpec { sim: sim.clone(), grids: spec.grids.clone(), objective }, &dev, &factory).unwrap();
    rev.set_design(&eps).unwrap();
    rev.forward().unwrap();
    let norm = spec.signal.spectrum(&spec.freqs, rev.dt(), rev.steps(), 0.5);
    let loss = port_loss(&sim, &q, &spec.freqs, norm, &rev.series().unwrap());
    let (l_rev, g_rev) = evaluate(&mut rev, &eps, Some(&loss), true);
    let g_rev = g_rev.unwrap();
    let l0: f64 = s_sess.iter().map(|z| z.norm_sqr()).sum();
    let h = 0.04;
    let fd = (evaluate(&mut rev, &axpy(&eps, h, &d), Some(&loss), false).0 - evaluate(&mut rev, &axpy(&eps, -h, &d), Some(&loss), false).0) / (2.0 * h);
    let (a_rev, a_sess) = (dot(&g_rev, &d), dot(&g_sess, &d));
    let gdiff = max_rel(&g_sess, &g_rev);
    println!(
        "[{dev}] sum |S21|^2: session {l0:.6e}, from the samples {l_rev:.6e}; reverse mode {a_rev:.6e}, session {a_sess:.6e}, FD {fd:.6e}; reverse vs FD {:.1e}, session vs reverse {:.1e}, max |grad session - grad reverse| / max {gdiff:.1e}",
        (a_rev - fd).abs() / fd.abs(),
        (a_sess - a_rev).abs() / a_rev.abs()
    );
    assert!((l_rev - l0).abs() < 1e-5 * l0);
    assert!((a_rev - fd).abs() < 3e-3 * fd.abs());
    assert!((a_sess - a_rev).abs() < 3e-2 * a_rev.abs() && gdiff < 3e-2);
    if lightsim_cuda::device_count().unwrap_or(0) >= 2 && std::env::var("LIGHTSIM_GPU").is_err() {
        let (two, s2, g2) = run_session(&["cuda:0".to_string(), "cuda:1".to_string()]);
        let used: std::collections::BTreeSet<String> = two.stats.iter().map(|r| r.device.clone()).collect();
        let ds = s2.iter().zip(&s_sess).fold(0f64, |m, (a, b)| m.max((a - b).norm() / b.norm()));
        let dg = max_rel(&g2, &g_sess);
        println!("session on two GPUs ({used:?}): max |S - S_one| / |S| = {ds:.1e}, max |grad - grad_one| / max = {dg:.1e}");
        assert_eq!(used.len(), 2, "both GPUs ran a part of the session");
        assert!(ds < 1e-5 && dg < 1e-4, "{ds} {dg}");
    }
}

#[test]
fn gpu_design_grids_must_stay_out_of_the_pml() {
    // FDTDX test_device_in_pml_raises (see the CPU test of the same name)
    let Some(dev) = device() else { return };
    let (sim, mut grids, flux) = reverse_scene(&base_case(Boundaries::all(pml(8))));
    grids[0].min[1] = -0.2 * UM;
    grids[0].dims = [5, 9, 2];
    let err = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, &dev, &factory).err().expect("a design grid in the PML must be refused");
    assert!(err.to_string().contains("overlaps the absorbing layer"), "{err}");
    let mut spec = session_spec();
    spec.grids[0].min[1] = -0.1 * UM;
    let err = GradientSession::new(spec, &[dev], &factory).err().expect("a design grid in the PML must be refused");
    assert!(err.to_string().contains("overlaps the absorbing layer"), "{err}");
}

#[test]
fn gpu_session_gradient_size_for_any_profile_power() {
    // see the CPU test session_gradient_size_for_any_profile_power
    let Some(dev) = device() else { return };
    let unit = plane_wave_session_spec();
    let mut odd = unit.clone();
    for (port, f) in odd.ports.iter_mut().zip([1e-8, 3.0]) {
        for q in port.profiles.iter_mut() {
            for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                arr.iter_mut().for_each(|x| *x *= f);
            }
        }
    }
    let mut rng = Rng(36);
    let eps: Vec<Vec<f64>> = unit.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let run = |spec: &lightsim_core::autodiff::SessionSpec| {
        let mut s = GradientSession::new(spec.clone(), &[dev.clone()], &factory).unwrap();
        s.set_design(&eps).unwrap();
        let sm = s.forward(false).unwrap();
        let g = s.backward(&sm.iter().map(|z| 2.0 * z).collect::<Vec<_>>()).unwrap();
        (s, sm, g)
    };
    let (mut s1, sm1, g1) = run(&unit);
    let (_, sm2, g2) = run(&odd);
    let ds = sm2.iter().zip(&sm1).fold(0f64, |m, (a, b)| m.max((a - b).norm() / b.norm()));
    let dg = max_rel(&g2, &g1);
    let gmax = g1.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let u: Vec<Vec<f64>> = g1.iter().map(|g| g.iter().map(|v| v / gmax).collect()).collect();
    let h = 0.04;
    let mut l = |e: &[Vec<f64>]| -> f64 {
        s1.set_design(e).unwrap();
        s1.forward(false).unwrap().iter().map(|z| z.norm_sqr()).sum()
    };
    let fd = (l(&axpy(&eps, h, &u)) - l(&axpy(&eps, -h, &u))) / (2.0 * h);
    let ad = dot(&g1, &u);
    println!("[{dev}] profiles of 1e-16 W and 9 W against 1 W: S {ds:.1e}, gradient {dg:.1e}; along the gradient: session {ad:.6e}, FD {fd:.6e}, rel {:.1e}", (ad - fd).abs() / fd.abs());
    assert!(ds < 1e-6 && dg < 1e-6, "{ds} {dg}");
    assert!((ad - fd).abs() < 3e-2 * fd.abs());
}

#[test]
fn gpu_reverse_mode_refuses_a_diverged_reverse_pass() {
    // see the CPU test of the same name: the 400 fs plane-wave run with a
    // slab of 1e4 ... 3e4 S/m, whose reverse pass overflows
    let Some(dev) = device() else { return };
    let spec = plane_wave_session_spec();
    let mut sim = spec.sim.clone();
    let mut lossy = Medium::dielectric(2.0).with_name("lossy");
    lossy.eps = [2.0, 3.0, 2.5];
    lossy.sigma = [3e4, 2e4, 1e4];
    sim.media.push(lossy);
    sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.6 * UM, -1.0, -1.0], max: [1.7 * UM, 1.0, 1.0] }, medium: "lossy".into() });
    let p = &spec.ports[0];
    sim.sources = vec![Source::Mode { name: "src".into(), axis: p.axis, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[0].clone(), signal: spec.signal.clone(), amplitude: 1.0 }];
    let q = &spec.ports[1];
    let objective = Objective::Flux { axis: q.axis, position: q.position, min: q.min, max: q.max, t_start: None, t_end: None, taper: None };
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: spec.grids.clone(), objective }, &dev, &factory).unwrap();
    let mut rng = Rng(35);
    s.set_design(&spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect::<Vec<_>>()).unwrap();
    assert!(s.forward().unwrap().is_finite());
    let err = s.backward().err().expect("a diverged reverse pass must be refused");
    println!("[{dev}] {err}");
    assert!(err.to_string().contains("diverged") && err.to_string().contains("session"), "{err}");
}

#[test]
fn gpu_reverse_mode_permeability_gradient() {
    // FDTDX test_inverse_permeability_gradient_inside_devices and the
    // magnetic lossy case of TestReversibleVsCheckpointedGradient (see the
    // CPU test reverse_mode_permeability_gradient): a permeability design
    // grid behind a magnetic slab, without and with magnetic conductivity;
    // the magnetic loss runs the new H scaling and its transpose.
    let Some(dev) = device() else { return };
    for (name, sigma_m) in [("permeability design, magnetic slab", 0.0), ("permeability design, magnetic lossy slab", 3e8)] {
        let (mut sim, mut grids, flux) = reverse_scene(&base_case(Boundaries::all(pml(8))));
        let mut d = Medium::dielectric(2.1).with_name("mudesign");
        d.mu = [6.0; 3];
        sim.media.push(d);
        grids[0].medium = "mudesign".into();
        grids[0].permeability = true;
        let mut m = Medium::dielectric(2.1).with_name("magnetic");
        m.mu = [1.5; 3];
        m.sigma_m = [sigma_m; 3];
        slab(&mut sim, m);
        gpu_check(&format!("{name}, flux energy"), &dev, sim.clone(), grids.clone(), flux, None, 3e-3, 38);
        gpu_check(&format!("{name}, phasor power"), &dev, sim, grids, phasor_plane(), Some(phasor_power(Window::all(vec![0.9 * F0, 1.1 * F0]))), 3e-3, 39);
    }
    // the permittivity gradient through a magnetically lossy block
    let mut m = Medium::dielectric(1.5).with_name("maglossy");
    m.mu = [2.0; 3];
    m.sigma_m = [3e8; 3];
    let (sim, grids, flux) = reverse_scene(&Case { name: "", boundaries: Boundaries::all(pml(8)), design_sigma: 0.0, extra: Some(m), tol: 3e-3 });
    gpu_check("permittivity design, magnetic lossy block", &dev, sim, grids, flux, None, 3e-3, 41);
}

#[test]
fn gpu_reverse_mode_pole_coefficient_gradients() {
    // FDTDX TestPerAxisDispersiveGradientCheckpointed (see the CPU test
    // reverse_mode_pole_coefficient_gradients): pole gradients on the GPU
    // against the CPU engine, and dL/d delta_eps of the x-only pole against
    // central differences of GPU runs.
    let Some(dev) = device() else { return };
    let lx = Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 2.0e14 };
    let lz = Pole::Lorentz { delta_eps: 0.8, omega0: TAU * 1.2 * F0, gamma: 1.0e14 };
    let dz = Pole::Debye { delta_eps: 0.6, tau: 6.0e-15 };
    let scene = |px: &[Pole]| {
        let (mut sim, grids, flux) = reverse_scene(&base_case(Boundaries::all(pml(8))));
        for (name, poles, axes, x0) in [("px", px.to_vec(), [true, false, false], 1.0), ("pz", vec![lz.clone(), dz.clone()], [false, false, true], 1.15)] {
            let mut m = Medium::dielectric(2.0).with_name(name);
            m.eps = [2.0, 2.5, 3.0];
            m.sigma = [100.0; 3];
            m.poles = poles;
            m.pole_axes = axes;
            sim.media.push(m);
            sim.structures.push(Structure { name: name.into(), shape: Shape::Box { min: [x0 * UM, 0.0, 0.0], max: [(x0 + 0.1) * UM, 1.2 * UM, 1.0 * UM] }, medium: name.into() });
        }
        ReverseSpec { sim, grids, objective: flux }
    };
    let spec = scene(&[lx.clone()]);
    let mut rng = Rng(42);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let grads = |dev: &str| {
        let mut s = ReverseSession::new(spec.clone(), dev, &factory).unwrap();
        s.set_pole_gradients(true).unwrap();
        evaluate(&mut s, &eps, None, true);
        s.pole_gradients().to_vec()
    };
    let (g, gc) = (grads(&dev), grads("cpu"));
    assert_eq!(g.len(), gc.len());
    for (a, b) in g.iter().zip(&gc) {
        let scale = b.params.iter().fold(0f64, |m, v| m.max(v.abs()));
        let d = a.params.iter().zip(&b.params).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale;
        let dc = a.coefficients.iter().zip(&b.coefficients).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / b.coefficients.iter().fold(0f64, |m, v| m.max(v.abs()));
        println!("[{dev}] {} pole {} axis {}: dL/d params {:?}; max |GPU - CPU| / max: params {d:.1e}, coefficients {dc:.1e}", a.medium, a.pole, a.axis, a.params);
        assert!(d < 1e-3 && dc < 1e-3, "{d} {dc}");
    }
    let v = 1.5;
    let h = 0.015;
    let value = |de: f64| {
        let mut s = ReverseSession::new(scene(&[Pole::Lorentz { delta_eps: de, omega0: TAU * 1.4 * F0, gamma: 2.0e14 }]), &dev, &factory).unwrap();
        evaluate(&mut s, &eps, None, false).0
    };
    let fd = (value(v + h) - value(v - h)) / (2.0 * h);
    let ad = g.iter().find(|p| p.medium == "px").unwrap().params[0];
    let rel = (ad - fd).abs() / fd.abs();
    println!("[{dev}] px dL/d delta_eps: reverse mode {ad:.6e}, finite difference {fd:.6e}, rel {rel:.2e}");
    assert!(rel < 2e-3);
}

#[test]
fn gpu_session_through_an_oriented_dispersive_medium() {
    // FDTDX TestOrientedDispersionGradients on the S-matrix session (see
    // the CPU test): an oriented-pole slab next to the design; the GPU
    // session against the CPU session and against central differences.
    let Some(dev) = device() else { return };
    let mut spec = plane_wave_session_spec();
    let mut m = Medium::dielectric(2.0).with_name("oriented");
    m.oriented = vec![OrientedPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: TAU * 2.0 * F0, gamma: 1e14 }, u: [1.0, 1.0, 0.0] }];
    spec.sim.media.push(m);
    spec.sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.6 * UM, -1.0, -1.0], max: [1.75 * UM, 1.0, 1.0] }, medium: "oriented".into() });
    let mut rng = Rng(43);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let run = |dev: &str| {
        let mut s = GradientSession::new(spec.clone(), &[dev.to_string()], &factory).unwrap();
        s.set_design(&eps).unwrap();
        let sm = s.forward(false).unwrap();
        let g = s.backward(&sm.iter().map(|z| 2.0 * z).collect::<Vec<_>>()).unwrap();
        (s, sm, g)
    };
    let (mut sg, smg, gg) = run(&dev);
    let (_, smc, gc) = run("cpu");
    let ds = smg.iter().zip(&smc).fold(0f64, |m, (a, b)| m.max((a - b).norm() / b.norm()));
    let dg = max_rel(&gg, &gc);
    let gmax = gg.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let u: Vec<Vec<f64>> = gg.iter().map(|g| g.iter().map(|v| v / gmax).collect()).collect();
    let h = 0.04;
    let mut l = |e: &[Vec<f64>]| -> f64 {
        sg.set_design(e).unwrap();
        sg.forward(false).unwrap().iter().map(|z| z.norm_sqr()).sum()
    };
    let fd = (l(&axpy(&eps, h, &u)) - l(&axpy(&eps, -h, &u))) / (2.0 * h);
    let ad = dot(&gg, &u);
    println!("[{dev}] oriented pole: S vs CPU {ds:.1e}, gradient vs CPU {dg:.1e}; along the gradient session {ad:.6e}, FD {fd:.6e}, rel {:.1e}", (ad - fd).abs() / fd.abs());
    assert!(ds < 1e-4 && dg < 2e-3);
    assert!(ad.is_finite() && ad != 0.0 && (ad - fd).abs() < 3e-2 * fd.abs());
}
