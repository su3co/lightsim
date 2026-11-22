//! Gradient engines on the GPU (skipped without one): reverse mode against
//! finite differences and against the CPU engine, the persistent S-matrix
//! session against the CPU session, fused and split stepping.
//!
//! `LIGHTSIM_GPU` selects the device ordinal (default 0).

use lightsim_core::autodiff::engine::cpu_factory;
use lightsim_core::autodiff::testing::{Case, F0, Rng, pml, reverse_scene, reverse_scene_tight, samples, session_spec};
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
