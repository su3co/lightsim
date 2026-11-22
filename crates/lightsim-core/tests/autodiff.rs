//! Gradients inside the engine (CPU): the transposed step, time-domain
//! reverse mode against finite differences, and the persistent S-matrix
//! session against plain runs.

use lightsim_core::autodiff::cpu::{AdjointPml, adjoint_step};
use lightsim_core::autodiff::design::update_volume;
use lightsim_core::autodiff::engine::cpu_factory;
use lightsim_core::autodiff::testing::{Case, F0, Rng, UM, pml, reverse_scene, reverse_scene_tight, samples, session_spec};
use lightsim_core::autodiff::{Channel, DesignGrid, GradientSession, Objective, ReverseSession, ReverseSpec};
use lightsim_core::engine::{Backend, RawResults};
use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::{Medium, Pole};
use lightsim_core::monitors::Monitor;
use lightsim_core::plan::{CompileOptions, Plan, compile};
use lightsim_core::results::{Results, assemble, finalize};
use lightsim_core::run::{Chain, RunOptions};
use lightsim_core::sources::Source;
use lightsim_core::spec::{Boundaries, Boundary, Simulation};
use num_complex::Complex64;

// ----------------------------------------------------------------------------
// the transposed step
// ----------------------------------------------------------------------------

fn step_no_sources(be: &mut CpuBackend) {
    be.fill_ghost_e().unwrap();
    be.update_h().unwrap();
    be.fill_ghost_h().unwrap();
    be.update_e().unwrap();
    be.finish_e().unwrap();
}

fn ijk(plan: &Plan, s: usize) -> Option<[usize; 3]> {
    let plane = plan.plane();
    let p = s / plane;
    if p == 0 || p > plan.nxl {
        return None;
    }
    Some([p - 1, (s % plane) / plan.nz(), s % plan.nz()])
}

/// <y, T^k x> against <(T^k)^T y, x> with the transposed step executed on
/// the scaled adjoint variables: the forward step outside the CPML layers
/// and, with `exact_pml`, the transposed CPML inside them (otherwise the
/// forward CPML). Returns the relative mismatch.
fn transpose_mismatch(sim: &Simulation, steps: usize, seed: u64, exact_pml: bool) -> f64 {
    let plan = compile(sim, &CompileOptions::default()).unwrap();
    let len = plan.storage_len();
    let mut rng = Rng(seed);
    let ch = |a: usize, s: usize| plan.ch.as_ref().map(|c| c[a][s]).unwrap_or(plan.ch_scalar) as f64;
    let interior = |s: usize| ijk(&plan, s).is_some();
    // x: a state of the invariant subspace (one step from a random E)
    let mut fwd = CpuBackend::new(plan.clone()).unwrap();
    for a in 0..3 {
        let e: Vec<f32> = (0..len).map(|s| if interior(s) && plan.cb[a][s] != 0.0 { rng.sym() } else { 0.0 }).collect();
        fwd.set_field(Component::e(a), &e).unwrap();
    }
    step_no_sources(&mut fwd);
    let x: Vec<Vec<f32>> = Component::ALL.iter().map(|c| fwd.field(*c).unwrap()).collect();
    // x is the field state alone: the CPML memory starts from zero on both sides
    fwd.reset().unwrap();
    for (c, v) in x.iter().enumerate() {
        fwd.set_field(Component::from_index(c), v).unwrap();
    }
    for _ in 0..steps {
        step_no_sources(&mut fwd);
    }
    let tx: Vec<Vec<f32>> = Component::ALL.iter().map(|c| fwd.field(*c).unwrap()).collect();
    // y in scaled form, likewise a reachable state: the samples a wall
    // freezes (tangential E on PEC, tangential H half a cell inside a
    // high-side PMC) are zero, as they are in any adjoint run
    let mut adj = CpuBackend::new(plan.clone()).unwrap();
    for a in 0..3 {
        let e: Vec<f32> = (0..len).map(|s| if interior(s) && plan.cb[a][s] != 0.0 { rng.sym() } else { 0.0 }).collect();
        adj.set_field(Component::e(a), &e).unwrap();
    }
    step_no_sources(&mut adj);
    let ys: Vec<Vec<f32>> = Component::ALL.iter().map(|c| adj.field(*c).unwrap()).collect();
    // true cotangent from the scaled one: a = V_E a^/cb, b = -V_H b^/ch
    let unscale = |c: usize, s: usize, v: f32| -> f64 {
        let Some(g) = ijk(&plan, s) else { return 0.0 };
        let vol = update_volume(&plan, Component::from_index(c), g);
        if c < 3 {
            if plan.cb[c][s] == 0.0 { 0.0 } else { vol * v as f64 / plan.cb[c][s] as f64 }
        } else {
            -vol * v as f64 / ch(c - 3, s)
        }
    };
    let dot = |ys: &[Vec<f32>], xs: &[Vec<f32>]| -> f64 { (0..6).map(|c| (0..len).map(|s| unscale(c, s, ys[c][s]) * xs[c][s] as f64).sum::<f64>()).sum() };
    let lhs = dot(&ys, &tx);
    adj.reset().unwrap();
    for (c, v) in ys.iter().enumerate() {
        adj.set_field(Component::from_index(c), v).unwrap();
    }
    let mut st = AdjointPml::new(&plan);
    for _ in 0..steps {
        if exact_pml {
            adjoint_step(&mut adj, &mut st).unwrap();
        } else {
            step_no_sources(&mut adj);
        }
    }
    let ty: Vec<Vec<f32>> = Component::ALL.iter().map(|c| adj.field(*c).unwrap()).collect();
    let rhs = dot(&ty, &x);
    (lhs - rhs).abs() / lhs.abs().max(rhs.abs())
}

fn walls_scene(b: Boundaries) -> Simulation {
    // non-uniform mesh, a lossy block and a magnetic block
    let x: Vec<f64> = (0..=14).map(|i| 0.1 * UM * (i as f64 + 0.02 * (i * i) as f64)).collect();
    let y: Vec<f64> = (0..=11).map(|i| 0.08 * UM * (i as f64 + 0.3 * ((i as f64) * 0.9).sin())).collect();
    let z: Vec<f64> = (0..=9).map(|i| 0.12 * UM * i as f64).collect();
    let (xm, ym, zm) = (*x.last().unwrap(), *y.last().unwrap(), *z.last().unwrap());
    let mut sim = Simulation::new([0.0; 3], [xm, ym, zm], MeshSpec::Explicit { x, y, z }, b, 5e-15);
    sim.media.push(Medium::dielectric(6.0).with_name("d"));
    let mut lossy = Medium::dielectric(3.0).with_name("lossy");
    lossy.sigma = [2000.0, 500.0, 1000.0];
    sim.media.push(lossy);
    let mut mag = Medium::dielectric(2.0).with_name("mag");
    mag.mu = [1.5, 2.0, 1.2];
    sim.media.push(mag);
    sim.structures.push(Structure { name: "a".into(), shape: Shape::Box { min: [0.3 * UM, 0.2 * UM, 0.25 * UM], max: [0.9 * UM, 0.6 * UM, 0.8 * UM] }, medium: "d".into() });
    sim.structures.push(Structure { name: "b".into(), shape: Shape::Box { min: [1.0 * UM, 0.1 * UM, 0.1 * UM], max: [1.5 * UM, 0.5 * UM, 0.7 * UM] }, medium: "lossy".into() });
    sim.structures.push(Structure { name: "c".into(), shape: Shape::Sphere { center: [0.6 * UM, 0.6 * UM, 0.4 * UM], radius: 0.25 * UM }, medium: "mag".into() });
    sim
}

#[test]
fn transposed_step_is_the_forward_step_on_scaled_adjoints() {
    let cases: Vec<(&str, Boundaries)> = vec![
        ("PEC", Boundaries::all(Boundary::Pec)),
        ("periodic", Boundaries::all(Boundary::Periodic)),
        ("PMC", Boundaries::all(Boundary::Pmc)),
        ("mixed", Boundaries { x: [Boundary::Pmc, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Pec, Boundary::Pmc] }),
        ("mixed2", Boundaries { x: [Boundary::Periodic, Boundary::Periodic], y: [Boundary::Pmc, Boundary::Pmc], z: [Boundary::Pmc, Boundary::Pec] }),
    ];
    for (name, b) in cases {
        for steps in [1, 30] {
            let m = transpose_mismatch(&walls_scene(b.clone()), steps, 7, true);
            println!("{name:9} {steps:3} steps: |<y,Tx> - <T'y,x>| / |<y,Tx>| = {m:.2e}");
            assert!(m < 3e-5, "{name} {steps}: {m}");
        }
    }
    // CPML layers, with random fields inside them. The transposed CPML
    // (stretching coefficients applied before the difference) is exact; the
    // forward CPML in its place is not, by a wide margin.
    let cases = [
        ("PML x,y", Boundaries { x: [pml(8), pml(8)], y: [pml(8), pml(8)], z: [Boundary::Periodic, Boundary::Periodic] }),
        ("PML all", Boundaries::all(pml(6))),
        ("PML z + PMC x", Boundaries { x: [Boundary::Pmc, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [pml(8), pml(8)] }),
    ];
    for (name, b) in cases {
        for steps in [1, 30, 120] {
            let m = transpose_mismatch(&walls_scene(b.clone()), steps, 7, true);
            let f = transpose_mismatch(&walls_scene(b.clone()), steps, 7, false);
            println!("{name:13} {steps:3} steps: transposed CPML {m:.2e}, forward CPML in its place {f:.2e}");
            assert!(m < 3e-5, "{name} {steps}: {m}");
            assert!(f > 1e-4, "{name} {steps}: the forward CPML is not expected to be its own transpose ({f})");
        }
    }
}

// ----------------------------------------------------------------------------
// reverse mode against finite differences
// ----------------------------------------------------------------------------

fn reverse_check(c: &Case) -> (f64, f64, f64) {
    reverse_check_scene(c, reverse_scene(c))
}

fn reverse_check_scene(c: &Case, (sim, grids, objective): (Simulation, Vec<DesignGrid>, Objective)) -> (f64, f64, f64) {
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: grids.clone(), objective }, "cpu", &cpu_factory).unwrap();
    let mut rng = Rng(11);
    let eps = vec![samples(&mut rng, grids[0].samples(), 2.5, 6.0)];
    s.set_design(&eps).unwrap();
    let (val, grad) = s.value_and_grad().unwrap();
    assert!(val.is_finite() && val != 0.0);
    // directional derivative along a random direction, central differences
    let d: Vec<f64> = (0..eps[0].len()).map(|_| 2.0 * rng.next() - 1.0).collect();
    let h = 0.04;
    let shifted = |sg: f64| vec![eps[0].iter().zip(&d).map(|(e, d)| e + sg * h * d).collect::<Vec<f64>>()];
    s.set_design(&shifted(1.0)).unwrap();
    let lp = s.forward().unwrap();
    s.set_design(&shifted(-1.0)).unwrap();
    let lm = s.forward().unwrap();
    let fd = (lp - lm) / (2.0 * h);
    let ad: f64 = grad[0].iter().zip(&d).map(|(g, d)| g * d).sum();
    // per-sample check on the two largest entries
    let rel = (ad - fd).abs() / fd.abs();
    println!("{:22} L = {val:.4e}  dL adjoint {ad:.5e}  finite difference {fd:.5e}  rel {rel:.2e}  (fwd {:.2} s, rev {:.2} s)", c.name, s.stats.forward_seconds, s.stats.reverse_seconds);
    assert!(rel < c.tol, "{}: {rel}", c.name);
    (val, ad, fd)
}

#[test]
fn reverse_mode_matches_finite_differences() {
    let all_pml = Boundaries::all(pml(8));
    let walls = Boundaries { x: [Boundary::Pec, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Pmc, Boundary::Pec] };
    let mut lorentz = Medium::dielectric(2.5).with_name("lorentz");
    lorentz.poles = vec![Pole::Lorentz { delta_eps: 1.5, omega0: 2.0 * std::f64::consts::PI * 1.4 * F0, gamma: 2.0e14 }, Pole::Drude { omega_p: 2.0e15, gamma: 1.0e14 }];
    let mut debye = Medium::dielectric(2.0).with_name("debye");
    debye.poles = vec![Pole::Debye { delta_eps: 1.0, tau: 8.0e-15 }];
    let cases = [
        Case { name: "walls (exact transpose)", boundaries: walls.clone(), design_sigma: 0.0, extra: None, tol: 2e-3 },
        Case { name: "PML", boundaries: all_pml.clone(), design_sigma: 0.0, extra: None, tol: 3e-3 },
        Case { name: "PML, lossy design", boundaries: all_pml.clone(), design_sigma: 3000.0, extra: None, tol: 3e-3 },
        Case { name: "PML, Lorentz + Drude", boundaries: all_pml.clone(), design_sigma: 0.0, extra: Some(lorentz), tol: 3e-3 },
        Case { name: "walls, Debye", boundaries: walls, design_sigma: 0.0, extra: Some(debye), tol: 3e-3 },
    ];
    for c in &cases {
        reverse_check(c);
    }
    // design voxels right up to the absorbing layers: with the forward CPML
    // standing in for its transpose this case is off by several per cent
    let tight = Case { name: "CPML next to design", boundaries: all_pml, design_sigma: 0.0, extra: None, tol: 5e-4 };
    reverse_check_scene(&tight, reverse_scene_tight(&tight));
}

#[test]
fn reverse_mode_through_a_kerr_medium() {
    // chi3 strong enough to matter: the gradient differs from the linear one
    let mut kerr = Medium::dielectric(4.0).with_name("kerr");
    kerr.chi3 = [30.0; 3];
    kerr.chi2 = [1.5; 3];
    let walls = Boundaries { x: [Boundary::Pec, Boundary::Pec], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Pmc, Boundary::Pec] };
    let nl = Case { name: "walls, chi2 + chi3", boundaries: walls.clone(), design_sigma: 0.0, extra: Some(kerr.clone()), tol: 4e-3 };
    let (v_nl, ad_nl, _) = reverse_check(&nl);
    let mut lin = kerr;
    lin.chi2 = [0.0; 3];
    lin.chi3 = [0.0; 3];
    let (v_lin, ad_lin, _) = reverse_check(&Case { name: "same, linear", boundaries: walls, design_sigma: 0.0, extra: Some(lin), tol: 2e-3 });
    assert!((v_nl - v_lin).abs() > 1e-3 * v_lin.abs(), "the nonlinearity is too weak to test anything: {v_nl} vs {v_lin}");
    assert!((ad_nl - ad_lin).abs() > 5e-3 * ad_lin.abs(), "{ad_nl} vs {ad_lin}");
}

/// After the reverse pass the forward fields are back at the initial state.
#[test]
fn reverse_pass_reconstructs_the_forward_fields() {
    use lightsim_core::autodiff::engine::{IoSet, ReverseConfig};
    use lightsim_core::autodiff::reverse::shell_boxes;
    use lightsim_core::plan::TimePlan;
    let c = Case { name: "", boundaries: Boundaries::all(pml(8)), design_sigma: 0.0, extra: None, tol: 0.0 };
    let (sim, _, _) = reverse_scene(&c);
    let mut plan = compile(&sim, &CompileOptions::default()).unwrap();
    let mut io = IoSet::take(&mut plan);
    io.time.push(TimePlan { name: "shell".into(), boxes: shell_boxes(&plan), interval: 1, start_step: 0, is_probe: false });
    let n = plan.nsteps;
    let mut eng = cpu_factory("cpu", plan.clone()).unwrap();
    let io = eng.add_io(io).unwrap();
    eng.begin(io).unwrap();
    let mut peak = 0f32;
    let mut mid = vec![];
    for s in 0..n {
        eng.step(s).unwrap();
        if s == n / 2 {
            mid = eng.reverse_field(Component::Ez, false).unwrap();
            peak = mid.iter().fold(0f32, |m, v| m.max(v.abs()));
        }
    }
    eng.reverse_begin(ReverseConfig { shell: Some(0), sources: vec![], design: Default::default() }).unwrap();
    let interior = |s: usize| -> bool {
        let Some(g) = ijk(&plan, s) else { return false };
        !plan.pml.iter().any(|p| (0..3).all(|a| g[a] >= p.start[a] && g[a] < p.start[a] + p.size[a]))
    };
    for s in (0..n).rev() {
        eng.reverse_step(s).unwrap();
        if s == n / 2 + 1 {
            // state after forward step n/2
            let f = eng.reverse_field(Component::Ez, false).unwrap();
            let err = (0..f.len()).filter(|s| interior(*s)).fold(0f32, |m, s| m.max((f[s] - mid[s]).abs()));
            println!("mid-run field reconstructed to {:.1e} of the peak", err / peak);
            assert!(err < 1e-4 * peak, "{err} vs peak {peak}");
        }
    }
    let f = eng.reverse_field(Component::Ez, false).unwrap();
    let err = (0..f.len()).filter(|s| interior(*s)).fold(0f32, |m, s| m.max(f[s].abs()));
    println!("initial (zero) field reconstructed to {:.1e} of the peak", err / peak);
    assert!(err < 1e-4 * peak, "{err} vs peak {peak}");
}

// ----------------------------------------------------------------------------
// S-matrix session
// ----------------------------------------------------------------------------

#[test]
fn session_reuses_buffers_and_matches_plain_runs() {
    let spec = session_spec();
    let mut rng = Rng(5);
    let e1: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let e2: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let mut s = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
    assert_eq!(s.shape(), (2, 1, 3));
    s.set_design(&e1).unwrap();
    let s1 = s.forward(false).unwrap();
    assert!(s1.iter().all(|z| z.norm() > 1e-4 && z.norm() < 1.5), "{s1:?}");
    let g: Vec<Complex64> = (0..6).map(|i| Complex64::new(0.3 + 0.1 * i as f64, -0.2 + 0.05 * i as f64)).collect();
    let g1 = s.backward(&g).unwrap();
    assert_eq!(s.stats.len(), 3);
    // another design, then back: nothing of the previous runs survives
    s.set_design(&e2).unwrap();
    let s2 = s.forward(true).unwrap();
    assert!(s1.iter().zip(&s2).any(|(a, b)| (a - b).norm() > 1e-3));
    s.set_design(&e1).unwrap();
    let s1b = s.forward(false).unwrap();
    let g1b = s.backward(&g).unwrap();
    assert_eq!(s1, s1b);
    assert_eq!(g1, g1b);
    assert!(g1.iter().all(|v| v.iter().any(|x| *x != 0.0)));
    // only the adjoint runs the upstream gradient needs are done
    let mut gz = g.clone();
    gz[3..].iter_mut().for_each(|z| *z = Complex64::new(0.0, 0.0));
    s.set_design(&e2).unwrap();
    s.forward(false).unwrap();
    s.backward(&gz).unwrap();
    assert_eq!(s.steps().iter().filter(|x| x.is_some()).count(), 2);

    // two engines (runs dispatched concurrently) give the same numbers
    let mut s2e = GradientSession::new(spec.clone(), &["cpu".to_string(), "cpu".to_string()], &cpu_factory).unwrap();
    s2e.set_design(&e1).unwrap();
    let sb = s2e.forward(true).unwrap();
    let gb = s2e.backward(&g).unwrap();
    assert_eq!(s1, sb);
    assert_eq!(g1, gb);

    // the fields the session keeps are those of a freshly compiled plain run
    let mut sim = spec.sim.clone();
    for (gr, e) in spec.grids.iter().zip(&e1) {
        sim.structures.push(Structure { name: gr.name.clone(), shape: gr.shape(e.clone()), medium: gr.medium.clone() });
    }
    let p = &spec.ports[0];
    sim.sources.push(Source::Mode { name: "__src__".into(), axis: 0, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[0].clone(), signal: spec.signal.clone(), amplitude: 1.0 });
    let q = &spec.ports[1];
    sim.monitors.push(Monitor::Mode { name: "out".into(), axis: 0, position: q.position, min: q.min, max: q.max, freqs: spec.freqs.clone(), modes: q.profiles.clone(), apodization: Default::default() });
    let grid = sim.build_grid().unwrap();
    let nodes = &grid.x;
    let i0 = (0..nodes.len()).min_by(|a, b| (nodes[*a] - p.position).abs().partial_cmp(&(nodes[*b] - p.position).abs()).unwrap()).unwrap();
    for (t, d) in [2usize, 4].iter().enumerate() {
        let x = nodes[i0 + d];
        let (mut mn, mut mx) = (p.min, p.max);
        mn[0] = x;
        mx[0] = x;
        sim.monitors.push(Monitor::Mode { name: format!("n{t}"), axis: 0, position: x, min: mn, max: mx, freqs: spec.freqs.clone(), modes: p.profiles.clone(), apodization: Default::default() });
    }
    let res = lightsim_core::run_cpu(&sim).unwrap();
    let amp = |name: &str, plus: bool, k: usize| {
        let m = &res.modes[name];
        if plus { Complex64::new(m.amp_plus_re[0][k], m.amp_plus_im[0][k]) } else { Complex64::new(m.amp_minus_re[0][k], m.amp_minus_im[0][k]) }
    };
    for k in 0..3 {
        let (a1, a2) = (amp("n0", true, k), amp("n1", true, k));
        let qf = a1 / a2;
        let norm = a1 * qf;
        let want = amp("out", true, k) / norm;
        let refl = amp("n0", false, k) / qf / norm;
        assert!((s1[k] - want).norm() < 2e-5 * want.norm(), "S21 at f{k}: {} vs {want}", s1[k]);
        assert!((s1[3 + k] - refl).norm() < 2e-5 * refl.norm().max(1e-3), "S11 at f{k}: {} vs {refl}", s1[3 + k]);
    }
}

/// Largest difference between two sets of S-parameters or gradients,
/// relative to the largest entry.
fn rel_diff_c(a: &[Complex64], b: &[Complex64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, z| m.max(z.norm()));
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).norm())) / scale
}

fn rel_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let scale = b.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    a.iter().flatten().zip(b.iter().flatten()).fold(0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn session_samples_all_its_monitors_with_one_stride() {
    // The design-region fields and the port monitors of a run are accumulated
    // together. They used to be compiled with different strides (the design
    // monitors at every step, the port monitors at the automatic stride),
    // which the GPU engine refuses. Both now take the stride of the
    // simulation, and the result is that of sampling every step once the
    // fields have decayed (the scene of the other tests stops in the middle
    // of the pulse, where the two sums differ by half a sampling interval of
    // the last field: 4 % of S).
    let mut spec = session_spec();
    spec.sim.run_time = 400e-15;
    spec.sim.shutoff = Some(1e-8);
    let mut rng = Rng(11);
    let eps: Vec<Vec<f64>> = spec.grids.iter().map(|g| samples(&mut rng, g.samples(), 2.5, 6.0)).collect();
    let g: Vec<Complex64> = (0..6).map(|i| Complex64::new(0.3 + 0.1 * i as f64, -0.2 + 0.05 * i as f64)).collect();
    let run = |spec: &lightsim_core::autodiff::SessionSpec| {
        let mut s = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
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
    let (ds, dg) = (rel_diff_c(&s_auto, &s_one), rel_diff(&g_auto, &g_one));
    println!("stride {stride} against every step: S {ds:.1e}, gradient {dg:.1e}");
    // the rounding of the single-precision sums: 5e-5 in S, 1e-4 in the gradient
    assert!(ds < 2e-4 && dg < 5e-4, "{ds} {dg}");
    // a stride of the simulation's choosing is taken over as it is
    let mut three = spec.clone();
    three.sim.dft_stride = Some(3);
    assert_eq!(run(&three).0, 3);
}

#[test]
fn session_with_two_ports_on_one_plane() {
    // Two ports with the same plane and window (two mode sets of one
    // waveguide) have identical monitor boxes. In an ordinary plan the second
    // monitor would take its fields from the first; in a session, where the
    // port monitors are a part of the monitor list and the engines read the
    // accumulators directly, every monitor keeps its own. The second port
    // carries the profile of the first with the opposite sign (the same
    // power), so its amplitudes and its gradients are minus those of the
    // first; fields taken from another monitor, or none, would not be.
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
    let mut s = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
    assert_eq!(s.shape(), (2, 1, 3));
    s.set_design(&eps).unwrap();
    let sm = s.forward(true).unwrap();
    for k in 0..3 {
        let (a, b) = (sm[k], sm[3 + k]);
        assert!(a.norm() > 1e-3, "{a}");
        assert!((b + a).norm() < 1e-5 * a.norm(), "frequency {k}: {b} against minus {a}");
    }
    // the gradient of each output alone
    let unit = |m: usize| (0..6).map(|i| if i / 3 == m { Complex64::new(1.0, 0.5) } else { Complex64::new(0.0, 0.0) }).collect::<Vec<_>>();
    let ga = s.backward(&unit(0)).unwrap();
    let gb = s.backward(&unit(1)).unwrap();
    let minus: Vec<Vec<f64>> = ga.iter().map(|v| v.iter().map(|x| -x).collect()).collect();
    assert!(ga.iter().flatten().any(|x| *x != 0.0));
    let d = rel_diff(&gb, &minus);
    assert!(d < 1e-4, "gradient of the second port against minus the first: {d}");
}

#[test]
fn shared_boxes_are_found_in_a_part_of_the_monitor_list() {
    // A field box, then two mode monitors and a flux monitor on one plane:
    // the later three take their fields from the first of them. The result
    // assembly must find those fields also when it is given only a part of
    // the monitors (the references used to be positions in the full list).
    let spec = session_spec();
    let (p, q) = (&spec.ports[0], &spec.ports[1]);
    let mut sim = spec.sim.clone();
    sim.sources.push(Source::Mode { name: "src".into(), axis: 0, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[0].clone(), signal: spec.signal.clone(), amplitude: 1.0 });
    let mode = |name: &str| Monitor::Mode { name: name.into(), axis: 0, position: q.position, min: q.min, max: q.max, freqs: spec.freqs.clone(), modes: q.profiles.clone(), apodization: Default::default() };
    sim.monitors.push(Monitor::DftField { name: "box".into(), min: [0.9 * UM, 0.3 * UM, 0.1 * UM], max: [1.5 * UM, 0.9 * UM, 0.5 * UM], components: Component::E.to_vec(), freqs: spec.freqs.clone(), apodization: Default::default(), stride: 0 });
    sim.monitors.push(mode("a"));
    sim.monitors.push(mode("b"));
    sim.monitors.push(Monitor::Flux { name: "flux".into(), axis: 0, position: q.position, min: q.min, max: q.max, freqs: spec.freqs.clone(), apodization: Default::default() });
    let run = |opts: &CompileOptions| -> (Plan, RawResults) {
        let plan = compile(&sim, opts).unwrap();
        let mut chain = Chain::new(vec![Box::new(CpuBackend::new(plan.clone()).unwrap())]);
        let (mut raws, _) = chain.run(&RunOptions::default()).unwrap();
        (plan, raws.remove(0))
    };
    let results = |plan: &Plan, raw: &RawResults, from: usize| -> Results {
        let mut part = plan.clone();
        part.dft = plan.dft[from..].to_vec();
        let raw = RawResults { dft: raw.dft[from..].to_vec(), time: raw.time.clone() };
        let mut res = assemble(&[&part], &[raw]);
        finalize(&mut res, &part);
        res
    };
    let (plan, raw) = run(&CompileOptions::default());
    assert!(plan.dft[1].shared.iter().all(|s| s.is_none()), "the first monitor on the plane records");
    for d in &plan.dft[2..] {
        assert!(d.shared.iter().all(|s| matches!(s, Some((1, _)))), "'{}' shares the boxes of 'a'", d.name);
        assert!(d.boxes.iter().all(|b| !b.is_local()));
    }
    let full = results(&plan, &raw, 0);
    assert!(full.modes["b"].amp_plus_re[0].iter().all(|v| *v != 0.0), "{:?}", full.modes["b"].amp_plus_re);
    assert!(full.flux["flux"].iter().all(|v| *v > 0.0), "{:?}", full.flux["flux"]);
    // without the field box in front
    let part = results(&plan, &raw, 1);
    assert!(!part.dft.contains_key("box"));
    for name in ["a", "b"] {
        assert_eq!(part.modes[name].amp_plus_re, full.modes[name].amp_plus_re, "{name}");
        assert_eq!(part.modes[name].amp_plus_im, full.modes[name].amp_plus_im, "{name}");
        assert_eq!(part.modes[name].amp_minus_re, full.modes[name].amp_minus_re, "{name}");
    }
    assert_eq!(part.flux["flux"], full.flux["flux"]);
    // a plan compiled without sharing gives the same numbers
    let (own_plan, own_raw) = run(&CompileOptions { share_monitors: false, ..Default::default() });
    assert!(own_plan.dft.iter().all(|d| d.shared.iter().all(|s| s.is_none()) && d.boxes.iter().all(|b| b.is_local())));
    let own = results(&own_plan, &own_raw, 1);
    assert_eq!(own.modes["b"].amp_plus_re, full.modes["b"].amp_plus_re);
    assert_eq!(own.flux["flux"], full.flux["flux"]);
    // a part that lacks the recording monitor has no data for the others:
    // that is refused, not filled with another monitor's fields or zeros
    let lost = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| results(&plan, &raw, 2)));
    assert!(lost.is_err());
}
