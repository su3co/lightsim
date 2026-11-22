//! Gradients inside the engine (CPU): the transposed step, time-domain
//! reverse mode against finite differences, and the persistent S-matrix
//! session against plain runs.

use lightsim_core::autodiff::cpu::{AdjointPml, adjoint_step};
use lightsim_core::autodiff::design::update_volume;
use lightsim_core::autodiff::engine::cpu_factory;
use lightsim_core::autodiff::testing::{Case, F0, Rng, UM, plane_wave_session_spec, pml, reverse_scene, reverse_scene_tight, samples, session_spec};
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
    // magnetic conductivity: the H update scales the lossy cells by da
    mag.sigma_m = [3e8, 1e8, 2e8];
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
    eng.reverse_begin(ReverseConfig { shell: Some(0), ..Default::default() }).unwrap();
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

    // the size of S and of the gradient does not depend on the power the
    // port profiles carry (the session scales them to 1 W): profiles of
    // other powers, different at the two ports, give the same numbers
    let mut scaled = spec.clone();
    for (port, f) in scaled.ports.iter_mut().zip([1e3, 1e-2]) {
        for q in port.profiles.iter_mut() {
            for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                arr.iter_mut().for_each(|x| *x *= f);
            }
        }
    }
    let mut ss = GradientSession::new(scaled, &["cpu".to_string()], &cpu_factory).unwrap();
    ss.set_design(&e1).unwrap();
    let (s1s, g1s) = (ss.forward(false).unwrap(), ss.backward(&g).unwrap());
    let (ds, dg) = (rel_diff_c(&s1s, &s1), rel_diff(&g1s, &g1));
    println!("profiles scaled by 1e3 and 1e-2: S {ds:.1e}, gradient {dg:.1e}");
    assert!(ds < 1e-6 && dg < 1e-6, "{ds} {dg}");

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

// ----------------------------------------------------------------------------
// FDTDX gradient tests, mirrored (test_adjoint_gradient.py, test_fdtd.py):
// reverse mode against central differences for the objectives, windows,
// media and boundaries those tests use, and the two gradient routes of
// lightsim (reverse mode, two-run adjoint session) on one objective.
// ----------------------------------------------------------------------------

use lightsim_core::autodiff::reverse::Series;
use lightsim_core::engine::RawDft;
use lightsim_core::materials::OrientedPole;
use lightsim_core::sources::{Direction, TimeSignal};
use std::f64::consts::TAU;

/// Value and cotangents of an objective evaluated on the recorded samples.
type Loss = Box<dyn Fn(&[Series], f64) -> (f64, Vec<Vec<f64>>)>;

struct Check {
    name: String,
    sim: Simulation,
    grids: Vec<DesignGrid>,
    objective: Objective,
    /// `None`: the built-in objective (flux energy).
    loss: Option<Loss>,
    tol: f64,
}

/// Objective value for the current design (and its gradient with `grad`).
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

fn random_design(rng: &mut Rng, grids: &[DesignGrid]) -> Vec<Vec<f64>> {
    grids.iter().map(|g| samples(rng, g.samples(), 2.5, 6.0)).collect()
}

fn random_direction(rng: &mut Rng, eps: &[Vec<f64>]) -> Vec<Vec<f64>> {
    eps.iter().map(|e| e.iter().map(|_| 2.0 * rng.next() - 1.0).collect()).collect()
}

fn axpy(eps: &[Vec<f64>], h: f64, d: &[Vec<f64>]) -> Vec<Vec<f64>> {
    eps.iter().zip(d).map(|(e, d)| e.iter().zip(d).map(|(e, d)| e + h * d).collect()).collect()
}

fn dot(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| x * y).sum()
}

/// Gradient against central differences along a random direction of all
/// design samples. Returns (L, gradient, relative difference).
fn fd_check(c: &Check, seed: u64) -> (f64, Vec<Vec<f64>>, f64) {
    let mut s = ReverseSession::new(ReverseSpec { sim: c.sim.clone(), grids: c.grids.clone(), objective: c.objective.clone() }, "cpu", &cpu_factory).unwrap();
    let mut rng = Rng(seed);
    let eps = random_design(&mut rng, &c.grids);
    let (val, grad) = evaluate(&mut s, &eps, c.loss.as_ref(), true);
    let grad = grad.unwrap();
    assert!(val.is_finite() && val != 0.0, "{}: L = {val}", c.name);
    assert!(grad.iter().flatten().all(|g| g.is_finite()) && grad.iter().flatten().any(|g| *g != 0.0), "{}: gradient not finite or zero", c.name);
    let d = random_direction(&mut rng, &eps);
    let h = 0.04;
    let lp = evaluate(&mut s, &axpy(&eps, h, &d), c.loss.as_ref(), false).0;
    let lm = evaluate(&mut s, &axpy(&eps, -h, &d), c.loss.as_ref(), false).0;
    let fd = (lp - lm) / (2.0 * h);
    let ad = dot(&grad, &d);
    let rel = (ad - fd).abs() / fd.abs();
    println!("{:44} L = {val:.4e}  dL reverse mode {ad:.5e}  finite difference {fd:.5e}  rel {rel:.2e}", c.name);
    assert!(rel < c.tol, "{}: {rel}", c.name);
    (val, grad, rel)
}

/// Time window of a frequency-domain (phasor) objective: every `stride`-th
/// sample inside the gate, weighted by a Tukey window (FDTDX: OnOffSwitch,
/// dft_subsample, TukeyWindow).
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
        if n % self.stride != 0 {
            return 0.0;
        }
        if let Some((a, b)) = self.gate {
            if t < a || t > b {
                return 0.0;
            }
        }
        let mut w = dt * self.stride as f64;
        if let Some((a, b)) = self.tukey {
            // Tukey window with alpha = 1/2 on [a, b]
            let x = (t - a) / (b - a);
            let e = 0.25;
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

/// Phasors `w(t_n) exp(i w t_n)` of one recorded component: E samples at
/// `(n + 1) dt`, H samples at `(n + 1/2) dt` (as the engine's DFT monitors).
fn phasors(win: &Window, comp: Component, ns: usize, dt: f64) -> Vec<Vec<Complex64>> {
    let off = if comp.is_e() { 1.0 } else { 0.5 };
    (0..ns)
        .map(|n| {
            let t = (n as f64 + off) * dt;
            let w = win.weight(n, t, dt);
            win.freqs.iter().map(|f| Complex64::from_polar(w, TAU * f * t)).collect()
        })
        .collect()
}

/// DFT of every cell of a series: `[cell][freq]`.
fn dft(s: &Series, ph: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let m: usize = s.size.iter().product();
    let nf = ph.first().map(|p| p.len()).unwrap_or(0);
    let mut out = vec![vec![Complex64::new(0.0, 0.0); nf]; m];
    for (n, p) in ph.iter().enumerate() {
        for (t, o) in out.iter_mut().enumerate() {
            let v = s.data[n * m + t] as f64;
            if v != 0.0 {
                for k in 0..nf {
                    o[k] += v * p[k];
                }
            }
        }
    }
    out
}

/// FDTDX's phasor-detector loss: `L = sum |phasor|^2` over the cells, the
/// components and the frequencies of the recorded box.
fn phasor_power(win: Window) -> Loss {
    Box::new(move |series: &[Series], dt: f64| {
        let mut l = 0.0;
        let mut cot = vec![];
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

fn base_case(name: &'static str, boundaries: Boundaries) -> Case {
    Case { name, boundaries, design_sigma: 0.0, extra: None, tol: 3e-3 }
}

#[test]
fn reverse_mode_phasor_power_objectives() {
    // FDTDX test_dipole_float32 / _float64, test_exact_dipole,
    // test_exact_dense_frequencies_in_one_adjoint_run,
    // test_frequencies_sharing_one_adjoint_run, test_exact_detector_windows,
    // test_mixed_cw_and_pulsed_sources_raise: a dipole, a design block and a
    // phasor detector (Ez, Ey, Hy on a plane) in a PML box. Reverse mode takes
    // any frequency set and any window in one reverse pass.
    let (sim, grids, _) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let t1 = sim.run_time;
    let dense: Vec<f64> = (0..12).map(|i| F0 * (0.85 + 0.3 * i as f64 / 11.0)).collect();
    let windows: Vec<(&str, Window)> = vec![
        ("phasor power, one frequency", Window::all(vec![F0])),
        ("phasor power, three frequencies", Window::all(vec![0.9 * F0, F0, 1.1 * F0])),
        ("phasor power, twelve dense frequencies", Window::all(dense)),
        ("phasor power, gated + every 3rd + Tukey", Window { freqs: vec![0.9 * F0, 1.1 * F0], gate: Some((0.15 * t1, 0.92 * t1)), stride: 3, tukey: Some((0.2 * t1, 0.85 * t1)) }),
    ];
    for (name, win) in windows {
        let c = Check { name: name.into(), sim: sim.clone(), grids: grids.clone(), objective: phasor_plane(), loss: Some(phasor_power(win)), tol: 3e-3 };
        fd_check(&c, 21);
    }
    // a continuous-wave source next to a pulsed one, phasor at the CW
    // frequency over the second half of the run
    let mut mixed = sim.clone();
    let cw = TimeSignal::ContinuousWave { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    mixed.sources.push(Source::PointDipole { name: "cw".into(), position: [0.3 * UM, 0.55 * UM, 0.45 * UM], polarization: Component::Ey, signal: cw, amplitude: 1e-15 });
    let win = Window { freqs: vec![F0], gate: Some((0.5 * t1, t1)), stride: 1, tukey: None };
    fd_check(&Check { name: "phasor power, CW + pulsed sources".into(), sim: mixed, grids, objective: phasor_plane(), loss: Some(phasor_power(win)), tol: 3e-3 }, 22);
}

/// Cell widths `1 + 0.25 sin(0.7 i + phase)` scaled to the length `l`.
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
fn reverse_mode_on_a_graded_mesh() {
    // FDTDX test_nonuniform_grid: graded cell sizes on all three axes
    // (rectilinear grid); flux and phasor objectives.
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let m = sim.region.max;
    sim.mesh = MeshSpec::Explicit { x: graded_nodes(m[0], 32, 0.0), y: graded_nodes(m[1], 24, 1.0), z: graded_nodes(m[2], 20, 2.0) };
    let g = sim.build_grid().unwrap();
    let (lo, hi) = (0..3).flat_map(|a| (0..g.n()[a]).map(move |i| (a, i))).fold((f64::INFINITY, 0f64), |(lo, hi), (a, i)| (lo.min(g.cell_size(a, i)), hi.max(g.cell_size(a, i))));
    println!("graded mesh: cell sizes {:.1} to {:.1} nm", lo * 1e9, hi * 1e9);
    assert!(hi / lo > 1.4);
    fd_check(&Check { name: "graded mesh, flux energy".into(), sim: sim.clone(), grids: grids.clone(), objective: flux, loss: None, tol: 3e-3 }, 23);
    let win = Window::all(vec![0.9 * F0, 1.1 * F0]);
    fd_check(&Check { name: "graded mesh, phasor power".into(), sim, grids, objective: phasor_plane(), loss: Some(phasor_power(win)), tol: 3e-3 }, 24);
}

/// A slab across the whole cross-section between the design and the
/// objective planes.
fn slab(sim: &mut Simulation, m: Medium) {
    let name = m.name.clone();
    sim.media.push(m);
    sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.05 * UM, -1.0, -1.0], max: [1.15 * UM, 1.0, 1.0] }, medium: name });
}

#[test]
fn reverse_mode_with_anisotropic_and_conductive_media() {
    // FDTDX test_anisotropic_and_conductive_background: a diagonally
    // anisotropic background and an anisotropic conductive slab.
    //
    // Reverse mode reconstructs the forward fields by running the scheme
    // backwards, which amplifies round-off in a conductor by
    // exp(sigma t / (eps0 eps_r)) over the time spent in it. FDTDX's
    // conductivity, 2e4 S/m (up to 3e4 S/m on one axis here), is a field
    // decay time of 0.7 fs, 40 times shorter than this run: the reverse-mode
    // gradient is then off by 2e-3 to 2e-2, and in a 400 fs run the
    // reversal overflows; reverse mode refuses such a pass
    // (reverse_mode_refuses_a_diverged_reverse_pass). The conductivities
    // below decay the fields in 1.5 to 4.4 fs and reverse exactly; the
    // strongly conductive case is checked on the session route, whose
    // adjoint is a second forward run.
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    sim.media.iter_mut().find(|m| m.name == "ox").unwrap().eps = [2.0, 2.4, 2.2];
    let mut lossy = Medium::dielectric(2.0).with_name("lossy");
    lossy.eps = [2.0, 3.0, 2.5];
    lossy.sigma = [1e4, 6e3, 1.5e4];
    slab(&mut sim, lossy);
    fd_check(&Check { name: "anisotropic + conductive, flux energy".into(), sim: sim.clone(), grids: grids.clone(), objective: flux, loss: None, tol: 3e-3 }, 25);
    let win = Window::all(vec![0.9 * F0, 1.1 * F0]);
    fd_check(&Check { name: "anisotropic + conductive, phasor power".into(), sim, grids, objective: phasor_plane(), loss: Some(phasor_power(win)), tol: 3e-3 }, 26);
}

#[test]
fn reverse_mode_next_to_per_axis_dispersion() {
    // FDTDX TestPerAxisDispersiveGradientCheckpointed: a diagonally
    // anisotropic, conductive medium whose Lorentz pole acts on x and z
    // only. (lightsim's gradient is with respect to the design permittivity;
    // the dispersive medium is part of the scene the gradient goes through.)
    let (sim0, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let mut disp = Medium::dielectric(2.0).with_name("disp");
    disp.eps = [2.0, 2.5, 3.0];
    disp.sigma = [100.0; 3];
    disp.poles = vec![Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 2.0e14 }];
    disp.pole_axes = [true, false, true];
    let mut sim = sim0.clone();
    slab(&mut sim, disp.clone());
    let (v, _, _) = fd_check(&Check { name: "per-axis Lorentz slab, flux energy".into(), sim, grids: grids.clone(), objective: flux.clone(), loss: None, tol: 3e-3 }, 27);
    // the pole mask matters: the same pole on all three axes gives another objective
    let mut all = disp;
    all.pole_axes = [true; 3];
    let mut sim = sim0;
    slab(&mut sim, all);
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: grids.clone(), objective: flux }, "cpu", &cpu_factory).unwrap();
    let eps = random_design(&mut Rng(27), &grids);
    let v_all = evaluate(&mut s, &eps, None, false).0;
    println!("pole on x, z: L = {v:.5e}; on x, y, z: L = {v_all:.5e}");
    assert!((v - v_all).abs() > 1e-3 * v.abs());
}

#[test]
fn reverse_mode_refuses_oriented_dispersion() {
    // FDTDX TestOrientedDispersionGradients: lightsim's gradient engines do
    // not support off-diagonal (tensor) media, oriented poles included. They
    // refuse such a scene instead of returning a wrong gradient.
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let mut m = Medium::dielectric(2.0).with_name("oriented");
    m.oriented = vec![OrientedPole { pole: Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 1e13 }, u: [1.0, 1.0, 0.0] }];
    slab(&mut sim, m);
    let err = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).err().expect("an oriented pole must be refused");
    println!("refused: {err}");
    assert!(err.to_string().contains("off-diagonal"), "{err}");
}

#[test]
fn reverse_mode_refuses_a_source_on_the_design() {
    // FDTDX test_gradient_region_overlapping_a_source_raises: the injection
    // coefficients of a source depend on the local permittivity, which the
    // gradient does not follow.
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    if let Source::PointDipole { position, .. } = &mut sim.sources[0] {
        *position = [0.7 * UM, 0.6 * UM, 0.525 * UM];
    }
    let err = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).err().expect("a source on the design must be refused");
    println!("refused: {err}");
    assert!(err.to_string().contains("design"), "{err}");
}

#[test]
fn reverse_mode_with_final_fields_in_a_periodic_box() {
    // FDTDX TestReversibleFdtdGradients (finite, non-trivial gradients of
    // sum E^2 + H^2 of the final fields, all-periodic box) and
    // test_final_field_in_loss_raises (which lightsim does not need: the
    // last samples are an objective like any other).
    let (sim, grids, _) = reverse_scene(&base_case("", Boundaries::all(Boundary::Periodic)));
    let obj = Objective::Fields { min: [1.0 * UM, 0.2 * UM, 0.2 * UM], max: [1.4 * UM, 1.0 * UM, 0.8 * UM], components: Component::ALL.to_vec() };
    let loss: Loss = Box::new(|series: &[Series], _dt: f64| {
        let mut l = 0.0;
        let mut cot = vec![];
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
    fd_check(&Check { name: "all periodic, sum E^2 + H^2 at the end".into(), sim, grids, objective: obj, loss: Some(loss), tol: 3e-3 }, 28);
}

#[test]
fn reverse_mode_with_two_adjacent_grids_and_a_uniform_block() {
    // FDTDX test_several_devices_with_overlapping_halos,
    // test_gradient_region_without_device, test_gradient_regions_device_and_object:
    // two design grids of different pixel sizes sharing a face (the Yee
    // samples on the face belong to one of them) and a uniform block, which
    // is a grid with a single sample. Each grid's gradient against central
    // differences along a direction confined to it.
    let (sim, _, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let grid = |name: &str, min: [f64; 3], max: [f64; 3], dims: [usize; 3]| DesignGrid { name: name.into(), min, max, dims, medium: "design".into(), permeability: false };
    let grids = vec![
        grid("lower", [0.52 * UM, 0.33 * UM, 0.3 * UM], [0.93 * UM, 0.91 * UM, 0.5 * UM], [5, 6, 1]),
        grid("upper", [0.52 * UM, 0.33 * UM, 0.5 * UM], [0.93 * UM, 0.91 * UM, 0.7 * UM], [4, 5, 2]),
        grid("block", [1.0 * UM, 0.4 * UM, 0.3 * UM], [1.2 * UM, 0.8 * UM, 0.7 * UM], [1, 1, 1]),
    ];
    let c = Check { name: "two adjacent grids + block, all".into(), sim: sim.clone(), grids: grids.clone(), objective: flux.clone(), loss: None, tol: 3e-3 };
    let (_, grad, _) = fd_check(&c, 29);
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: grids.clone(), objective: flux }, "cpu", &cpu_factory).unwrap();
    let mut rng = Rng(29);
    let eps = random_design(&mut rng, &grids);
    let h = 0.04;
    for gi in 0..grids.len() {
        let mut d = random_direction(&mut rng, &eps);
        for (j, v) in d.iter_mut().enumerate() {
            if j != gi {
                v.iter_mut().for_each(|x| *x = 0.0);
            }
        }
        let fd = (evaluate(&mut s, &axpy(&eps, h, &d), None, false).0 - evaluate(&mut s, &axpy(&eps, -h, &d), None, false).0) / (2.0 * h);
        let ad = dot(&grad, &d);
        let rel = (ad - fd).abs() / fd.abs();
        println!("  grid '{}' alone: reverse mode {ad:.5e}, finite difference {fd:.5e}, rel {rel:.2e}", grids[gi].name);
        assert!(rel < 3e-3, "{}: {rel}", grids[gi].name);
    }
}

/// The scene of the mirror test: a waveguide along x centred on y = 0.6 um,
/// two Ez dipoles at y = 0.6 -+ 0.1 um with the signs `(lower, upper)` and
/// two design grids mirrored about y = 0.6 um; with `half`, only the upper
/// half y > 0.6 um, closed by `wall`.
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

/// Samples of a [5, 4, 2] grid mirrored along y.
fn flip_y(v: &[f64]) -> Vec<f64> {
    let (ny, nz) = (4, 2);
    (0..v.len()).map(|t| {
        let (i, j, k) = (t / (ny * nz), (t / nz) % ny, t % nz);
        v[(i * ny + (ny - 1 - j)) * nz + k]
    }).collect()
}

#[test]
fn reverse_mode_with_mirror_symmetry() {
    // FDTDX test_mirror_symmetry: the symmetry-reduced domain (upper half,
    // closed by a PMC wall for E tangential even, a PEC wall for odd, as
    // lightsim.run_symmetric does) gives half the objective of the full
    // domain and the same gradient on its design grid; the full domain's two
    // mirrored grids have mirrored gradients. Also two separate grids.
    for (name, signs, wall) in [("PMC (even)", (1.0, 1.0), Boundary::Pmc), ("PEC (odd)", (-1.0, 1.0), Boundary::Pec)] {
        let mut rng = Rng(30);
        let (sim, grids, obj) = mirror_scene(signs, None);
        let upper = samples(&mut rng, grids[0].samples(), 2.5, 6.0);
        let mut full = ReverseSession::new(ReverseSpec { sim, grids, objective: obj }, "cpu", &cpu_factory).unwrap();
        let (lf, gf) = evaluate(&mut full, &[upper.clone(), flip_y(&upper)], None, true);
        let gf = gf.unwrap();
        let (sim, grids, obj) = mirror_scene(signs, Some(wall.clone()));
        let c = Check { name: format!("mirror {name}, half domain"), sim: sim.clone(), grids: grids.clone(), objective: obj.clone(), loss: None, tol: 3e-3 };
        fd_check(&c, 31);
        let mut half = ReverseSession::new(ReverseSpec { sim, grids, objective: obj }, "cpu", &cpu_factory).unwrap();
        let (lh, gh) = evaluate(&mut half, &[upper.clone()], None, true);
        let gh = gh.unwrap();
        let scale = gf[0].iter().fold(0f64, |m, v| m.max(v.abs()));
        let d_half = gh[0].iter().zip(&gf[0]).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
        let d_mirror = flip_y(&gf[1]).iter().zip(&gf[0]).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
        let dl = (2.0 * lh - lf).abs() / lf.abs();
        println!("mirror {name}: L full {lf:.5e}, 2 L half {:.5e} (rel {dl:.1e}); max |grad half - grad full| / max {d_half:.1e}; mirrored grids {d_mirror:.1e}", 2.0 * lh);
        assert!(dl < 1e-4 && d_half < 1e-3 && d_mirror < 1e-3, "{name}: {dl} {d_half} {d_mirror}");
    }
}

/// Frequency-domain flux through a flux plane (boxes `[E_b, E_c, H_b(p-1),
/// H_c(p-1), H_b(p), H_c(p)]`, as recorded by a flux objective) of a cross
/// section with uniform cells of area `da`: `P_k = 1/2 Re sum (E x H*)_a dA`,
/// H averaged over the two planes. `L = sum_k P_k`.
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
                        // d Re(E conj(H)) / dE(n) = Re(phi_E(n) conj(H)), / dH(n) = Re(E conj(phi_H(n)))
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
fn reverse_mode_periodic_cell_plane_wave_flux() {
    // FDTDX test_periodic_plane_source_and_flux: a plane wave in a cell
    // periodic in x and y (PML in z), a design grid spanning the whole
    // periodic cross-section (its outer samples touch the periodic edges)
    // and the flux through a plane spanning it too: in the frequency domain
    // at two frequencies (FDTDX's phasor Poynting flux) and as time-domain
    // energy (the built-in objective).
    let dx = 0.05 * UM;
    let b = Boundaries { x: [Boundary::Periodic, Boundary::Periodic], y: [Boundary::Periodic, Boundary::Periodic], z: [pml(8), pml(8)] };
    let mut sim = Simulation::new([0.0; 3], [0.4 * UM, 0.4 * UM, 1.2 * UM], MeshSpec::Uniform { dx: [dx; 3] }, b, 40e-15);
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("design"));
    sim.background = "ox".into();
    let sig = TimeSignal::GaussianPulse { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 2, position: 0.15 * UM, direction: Direction::Plus, polarization: Component::Ex, signal: sig, amplitude: 1.0, angle: 0.0, min: None, max: None });
    let grids = vec![DesignGrid { name: "g".into(), min: [0.0, 0.0, 0.45 * UM], max: [0.4 * UM, 0.4 * UM, 0.6 * UM], dims: [4, 4, 2], medium: "design".into(), permeability: false }];
    let (zp, mn, mx) = (0.9 * UM, [0.0, 0.0, 0.9 * UM], [0.4 * UM, 0.4 * UM, 0.9 * UM]);
    let flux = Objective::Flux { axis: 2, position: zp, min: mn, max: mx, t_start: None, t_end: None, taper: None };
    let freqs = vec![0.9 * F0, 1.1 * F0];
    let c = Check { name: "periodic cell, plane wave, phasor flux".into(), sim: sim.clone(), grids: grids.clone(), objective: flux.clone(), loss: Some(phasor_flux(Window::all(freqs.clone()), dx * dx)), tol: 3e-3 };
    fd_check(&c, 32);
    fd_check(&Check { name: "periodic cell, plane wave, flux energy".into(), sim: sim.clone(), grids: grids.clone(), objective: flux.clone(), loss: None, tol: 3e-3 }, 33);
    // the objective is the engine's own flux monitor
    let mut s = ReverseSession::new(ReverseSpec { sim: sim.clone(), grids: grids.clone(), objective: flux }, "cpu", &cpu_factory).unwrap();
    let eps = random_design(&mut Rng(32), &grids);
    s.set_design(&eps).unwrap();
    s.forward().unwrap();
    let series = s.series().unwrap();
    let per_freq: Vec<f64> = freqs.iter().map(|f| (phasor_flux(Window::all(vec![*f]), dx * dx))(&series, s.dt()).0).collect();
    let mut plain = sim;
    plain.structures.push(Structure { name: "g".into(), shape: grids[0].shape(eps[0].clone()), medium: "design".into() });
    plain.monitors.push(Monitor::Flux { name: "f".into(), axis: 2, position: zp, min: mn, max: mx, freqs, apodization: Default::default() });
    plain.dft_stride = Some(1);
    let res = lightsim_core::run_cpu(&plain).unwrap();
    for (k, (p, q)) in per_freq.iter().zip(&res.flux["f"]).enumerate() {
        println!("  frequency {k}: objective {p:.6e}, flux monitor {q:.6e}, rel {:.1e}", (p - q).abs() / q.abs());
        assert!((p - q).abs() < 1e-4 * q.abs(), "frequency {k}: {p} vs {q}");
    }
}

/// `|S|^2` of the session's output port, evaluated from the samples a flux
/// objective records on the port plane: the samples are transformed with the
/// engine's DFT convention, the mode amplitude is the core's own overlap
/// (`results::finalize` on a plan holding the port's mode monitor), which is
/// linear in the DFT fields; its weights are found by probing it with unit
/// fields. `S = a / N`, with `N` the source spectrum (normalize = "source").
struct PortObjective {
    freqs: Vec<f64>,
    norm: Vec<Complex64>,
    /// `[box][cell][freq]`
    w: Vec<Vec<Vec<Complex64>>>,
}

impl PortObjective {
    fn new(sim: &Simulation, port: &lightsim_core::autodiff::PortSpec, freqs: &[f64], norm: Vec<Complex64>, series: &[Series]) -> PortObjective {
        let mut s = sim.clone();
        s.sources.clear();
        s.monitors = vec![Monitor::Mode { name: "out".into(), axis: port.axis, position: port.position, min: port.min, max: port.max, freqs: freqs.to_vec(), modes: port.profiles.clone(), apodization: Default::default() }];
        let plan = compile(&s, &CompileOptions::default()).unwrap();
        let boxes = &plan.dft[0].boxes;
        assert_eq!(boxes.len(), series.len());
        for (b, se) in boxes.iter().zip(series) {
            assert!(b.comp == se.comp && b.glo == se.origin && b.gsize() == se.size, "mode monitor and flux objective record different boxes");
        }
        let nf = freqs.len();
        let amps = |raw: RawDft| -> Vec<Complex64> {
            let mut res = assemble(&[&plan], &[RawResults { dft: vec![raw], time: vec![] }]);
            finalize(&mut res, &plan);
            let m = &res.modes["out"];
            // the amplitude leaving the device: against the port direction
            let (re, im) = if port.direction == Direction::Minus { (&m.amp_plus_re[0], &m.amp_plus_im[0]) } else { (&m.amp_minus_re[0], &m.amp_minus_im[0]) };
            (0..nf).map(|k| Complex64::new(re[k], im[k])).collect()
        };
        let zero: Vec<(Vec<f32>, Vec<f32>)> = boxes.iter().map(|b| (vec![0f32; nf * b.lcells()], vec![0f32; nf * b.lcells()])).collect();
        assert!(amps(RawDft { boxes: zero.clone() }).iter().all(|a| a.norm() == 0.0));
        let w = boxes
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
        PortObjective { freqs: freqs.to_vec(), norm, w }
    }

    fn s(&self, f: &[Vec<Vec<Complex64>>]) -> Vec<Complex64> {
        (0..self.freqs.len()).map(|k| self.w.iter().zip(f).map(|(wb, fb)| wb.iter().zip(fb).map(|(w, z)| w[k] * z[k]).sum::<Complex64>()).sum::<Complex64>() / self.norm[k]).collect()
    }

    fn loss(self) -> Loss {
        Box::new(move |series: &[Series], dt: f64| {
            let win = Window::all(self.freqs.clone());
            let ph: Vec<Vec<Vec<Complex64>>> = series.iter().map(|s| phasors(&win, s.comp, s.data.len() / s.size.iter().product::<usize>(), dt)).collect();
            let f: Vec<Vec<Vec<Complex64>>> = series.iter().zip(&ph).map(|(s, p)| dft(s, p)).collect();
            let s = self.s(&f);
            let l = s.iter().map(|z| z.norm_sqr()).sum();
            // dL/dsample(n) = sum_k 2 Re(conj(S_k) w_k phi_k(n) / N_k)
            let cot = self
                .w
                .iter()
                .zip(&ph)
                .map(|(wb, pb)| {
                    let m = wb.len();
                    let mut c = vec![0.0; pb.len() * m];
                    for (n, p) in pb.iter().enumerate() {
                        for (t, w) in wb.iter().enumerate() {
                            c[n * m + t] = (0..self.freqs.len()).map(|k| 2.0 * (s[k].conj() * w[k] * p[k] / self.norm[k]).re).sum();
                        }
                    }
                    c
                })
                .collect();
            (l, cot)
        })
    }
}

/// Gradients of `sum_k |S_21(f_k)|^2` (normalize = "source": S is the
/// outgoing amplitude over the source spectrum, a constant) on both routes.
struct Routes {
    /// Directional derivatives along a random direction: reverse mode,
    /// session, central differences of the objective.
    reverse: f64,
    session: f64,
    fd: f64,
    /// max |grad session - grad reverse| / max |grad reverse|
    gdiff: f64,
    /// Along the session's gradient: session and central differences.
    steep: (f64, f64),
}

fn two_routes(spec: &lightsim_core::autodiff::SessionSpec, seed: u64, reverse: bool) -> Routes {
    let mut rng = Rng(seed);
    let eps: Vec<Vec<f64>> = random_design(&mut rng, &spec.grids);
    let d = random_direction(&mut rng, &eps);
    let h = 0.04;
    // route 1: the session
    let mut sp = spec.clone();
    sp.normalize = "source".into();
    let mut sess = GradientSession::new(sp, &["cpu".to_string()], &cpu_factory).unwrap();
    sess.set_design(&eps).unwrap();
    let s_sess = sess.forward(false).unwrap();
    let g_sess = sess.backward(&s_sess.iter().map(|z| 2.0 * z).collect::<Vec<_>>()).unwrap();
    let l0: f64 = s_sess.iter().map(|z| z.norm_sqr()).sum();
    let mut l_sess = |e: &[Vec<f64>]| -> f64 {
        sess.set_design(e).unwrap();
        sess.forward(false).unwrap().iter().map(|z| z.norm_sqr()).sum()
    };
    // along the session's gradient (no cancellation in the derivative)
    let gmax = g_sess.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let u: Vec<Vec<f64>> = g_sess.iter().map(|g| g.iter().map(|v| v / gmax).collect()).collect();
    let steep = (dot(&g_sess, &u), (l_sess(&axpy(&eps, h, &u)) - l_sess(&axpy(&eps, -h, &u))) / (2.0 * h));
    println!("  sum |S21|^2 = {l0:.6e}; along the session's gradient: session {:.6e}, finite difference {:.6e}, rel {:.1e}", steep.0, steep.1, (steep.0 - steep.1).abs() / steep.1.abs());
    let session = dot(&g_sess, &d);
    if !reverse {
        return Routes { reverse: f64::NAN, session, fd: f64::NAN, gdiff: f64::NAN, steep };
    }
    // route 2: reverse mode, same source, objective evaluated on the port plane
    let p = &spec.ports[0];
    let mut sim = spec.sim.clone();
    sim.sources = vec![Source::Mode { name: "__src__".into(), axis: p.axis, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[0].clone(), signal: spec.signal.clone(), amplitude: 1.0 }];
    let q = spec.ports[1].clone();
    let objective = Objective::Flux { axis: q.axis, position: q.position, min: q.min, max: q.max, t_start: None, t_end: None, taper: None };
    let mut rev = ReverseSession::new(ReverseSpec { sim: sim.clone(), grids: spec.grids.clone(), objective }, "cpu", &cpu_factory).unwrap();
    rev.set_design(&eps).unwrap();
    rev.forward().unwrap();
    assert_eq!(sess.steps()[0], Some(rev.steps()), "both routes must run the same steps (no shutoff)");
    let spectrum = spec.signal.spectrum(&spec.freqs, rev.dt(), rev.steps(), 0.5);
    let loss = PortObjective::new(&sim, &q, &spec.freqs, spectrum, &rev.series().unwrap()).loss();
    let (l_rev, g_rev) = evaluate(&mut rev, &eps, Some(&loss), true);
    let g_rev = g_rev.unwrap();
    println!("  objective from the samples of the output plane {l_rev:.6e} (session {l0:.6e}, rel {:.1e})", (l_rev - l0).abs() / l0);
    assert!((l_rev - l0).abs() < 1e-5 * l0, "the two routes do not evaluate the same objective");
    let fd = (evaluate(&mut rev, &axpy(&eps, h, &d), Some(&loss), false).0 - evaluate(&mut rev, &axpy(&eps, -h, &d), Some(&loss), false).0) / (2.0 * h);
    let scale = g_rev.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let gdiff = g_sess.iter().flatten().zip(g_rev.iter().flatten()).fold(0f64, |m, (a, b)| m.max((a - b).abs())) / scale;
    let r = Routes { reverse: dot(&g_rev, &d), session, fd, gdiff, steep };
    println!(
        "  random direction: reverse mode {:.6e}, session {:.6e}, finite difference {:.6e}; reverse vs FD {:.1e}, session vs FD {:.1e}, session vs reverse {:.1e}; max |grad session - grad reverse| / max {:.1e}",
        r.reverse,
        r.session,
        r.fd,
        (r.reverse - r.fd).abs() / r.fd.abs(),
        (r.session - r.fd).abs() / r.fd.abs(),
        (r.session - r.reverse).abs() / r.reverse.abs(),
        r.gdiff
    );
    r
}

#[test]
fn reverse_mode_and_session_agree_on_one_objective() {
    // FDTDX TestReversibleVsCheckpointedGradient: two independent gradient
    // routes on the same scene and objective. lightsim's are the discrete
    // reverse pass through the time loop and the two-run continuous adjoint
    // of the S-matrix session. Objective: sum_k |S_21(f_k)|^2 of a port,
    // three frequencies, two adjacent design grids, CPML, run until the
    // fields have decayed (the DFTs of both routes are complete).
    // The session's adjoint formula (i w eps0 / 4) E_adj E_fwd dV is the
    // continuous one: it differs from the derivative of the discrete scheme
    // by its discretisation error (0.14 % along a random direction, 0.8 %
    // of the largest entry, at 50 nm cells and eps = 2.5 ... 6 in the
    // design), which the reverse pass does not have.
    let spec = plane_wave_session_spec();
    {
        // the ports' "mode" is the plane wave: |S21| = 1 without a design
        let mut s = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
        s.set_design(&spec.grids.iter().map(|g| vec![2.1; g.samples()]).collect::<Vec<_>>()).unwrap();
        let t = s.forward(false).unwrap();
        println!("|S21| without a design: {:?}", t.iter().map(|z| z.norm()).collect::<Vec<_>>());
        assert!(t.iter().all(|z| (z.norm() - 1.0).abs() < 1e-2));
    }
    println!("plane-wave ports, two adjacent grids:");
    let r = two_routes(&spec, 34, true);
    assert!((r.reverse - r.fd).abs() < 3e-3 * r.fd.abs(), "reverse mode vs finite differences");
    assert!((r.session - r.reverse).abs() < 3e-2 * r.reverse.abs() && r.gdiff < 3e-2, "session vs reverse mode");
    assert!((r.steep.0 - r.steep.1).abs() < 3e-2 * r.steep.1.abs(), "session vs finite differences");
    // FDTDX test_anisotropic_and_conductive_background with its
    // conductivity: a slab of anisotropic medium conducting 1e4 ... 3e4 S/m
    // between the design and the output port. The session's adjoint is a
    // second forward run and is not affected by the absorption; reverse mode
    // would have to reverse it over this long run, which it cannot (see
    // reverse_mode_refuses_a_diverged_reverse_pass).
    let mut lossy = Medium::dielectric(2.0).with_name("lossy");
    lossy.eps = [2.0, 3.0, 2.5];
    lossy.sigma = [3e4, 2e4, 1e4];
    let mut spec = spec;
    spec.sim.media.push(lossy);
    spec.sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.6 * UM, -1.0, -1.0], max: [1.7 * UM, 1.0, 1.0] }, medium: "lossy".into() });
    println!("same, with a strongly conductive anisotropic slab (session only):");
    let r = two_routes(&spec, 35, false);
    assert!((r.steep.0 - r.steep.1).abs() < 3e-2 * r.steep.1.abs(), "session vs finite differences");
}

#[test]
fn design_grids_must_stay_out_of_the_pml() {
    // FDTDX test_device_in_pml_raises. The reverse pass does not reconstruct
    // the fields inside the absorbing layers (a grid reaching 0.2 um into
    // them got a gradient off by 2.5e-2) and the session's adjoint formula
    // does not hold in their stretched coordinates: both refuse such a grid.
    let (sim, mut grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    grids[0].min[1] = -0.2 * UM;
    grids[0].dims = [5, 9, 2];
    let err = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).err().expect("a design grid in the PML must be refused");
    println!("reverse mode: {err}");
    assert!(err.to_string().contains("overlaps the absorbing layer"), "{err}");
    let mut spec = session_spec();
    spec.grids[0].min[1] = -0.1 * UM;
    let err = GradientSession::new(spec, &["cpu".to_string()], &cpu_factory).err().expect("a design grid in the PML must be refused");
    println!("session: {err}");
    assert!(err.to_string().contains("overlaps the absorbing layer"), "{err}");
    // up to the layers is fine (reverse_scene_tight, checked against finite
    // differences in reverse_mode_matches_finite_differences)
    let c = base_case("", Boundaries::all(pml(8)));
    let (sim, grids, flux) = reverse_scene_tight(&c);
    assert!(ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).is_ok());
}

#[test]
fn session_gradient_size_for_any_profile_power() {
    // A port profile of power P used to scale S by sqrt(P_in / P_out) and
    // the gradient by sqrt(P_in P_out); the session now scales the profiles
    // to 1 W. Exact eigenmode ports (plane waves) with profiles of 1e-16 W
    // and 9 W give the S and gradient of unit-power profiles, and the
    // gradient has the size of the finite differences of S.
    let unit = plane_wave_session_spec();
    let grid = unit.sim.build_grid().unwrap();
    for port in &unit.ports {
        let p = lightsim_core::autodiff::session::port_power(&grid, [false, true, true], port, 0);
        assert!((p - 1.0).abs() < 1e-12, "{p}");
    }
    let mut odd = unit.clone();
    for (port, f) in odd.ports.iter_mut().zip([1e-8, 3.0]) {
        for q in port.profiles.iter_mut() {
            for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                arr.iter_mut().for_each(|x| *x *= f);
            }
        }
    }
    let mut rng = Rng(36);
    let eps = random_design(&mut rng, &unit.grids);
    let run = |spec: &lightsim_core::autodiff::SessionSpec| {
        let mut s = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).unwrap();
        s.set_design(&eps).unwrap();
        let sm = s.forward(false).unwrap();
        let g = s.backward(&sm.iter().map(|z| 2.0 * z).collect::<Vec<_>>()).unwrap();
        (s, sm, g)
    };
    let (mut s1, sm1, g1) = run(&unit);
    let (_, sm2, g2) = run(&odd);
    let (ds, dg) = (rel_diff_c(&sm2, &sm1), rel_diff(&g2, &g1));
    println!("profiles of 1e-16 W and 9 W against 1 W: S {ds:.1e}, gradient {dg:.1e}");
    assert!(ds < 1e-6 && dg < 1e-6, "{ds} {dg}");
    // the size: along the gradient, against central differences of sum |S|^2
    let gmax = g1.iter().flatten().fold(0f64, |m, v| m.max(v.abs()));
    let u: Vec<Vec<f64>> = g1.iter().map(|g| g.iter().map(|v| v / gmax).collect()).collect();
    let h = 0.04;
    let mut l = |e: &[Vec<f64>]| -> f64 {
        s1.set_design(e).unwrap();
        s1.forward(false).unwrap().iter().map(|z| z.norm_sqr()).sum()
    };
    let fd = (l(&axpy(&eps, h, &u)) - l(&axpy(&eps, -h, &u))) / (2.0 * h);
    let ad = dot(&g1, &u);
    println!("along the gradient: session {ad:.6e}, finite difference {fd:.6e}, rel {:.1e}", (ad - fd).abs() / fd.abs());
    assert!((ad - fd).abs() < 3e-2 * fd.abs());
}

#[test]
fn reverse_mode_refuses_a_diverged_reverse_pass() {
    // Reverse mode reconstructs the forward fields by running the scheme
    // backwards; in a strong conductor round-off grows by exp(sigma t / eps).
    // Two scenes where it did: a 3e4 S/m slab in a 28 fs run (gradient off
    // by 1.6e4 relative, finite) and the slab of 1e4 ... 3e4 S/m in the
    // 400 fs plane-wave run (NaN). Both are refused with an error that
    // points to the session.
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let mut lossy = Medium::dielectric(2.0).with_name("lossy");
    lossy.eps = [2.0, 3.0, 2.5];
    lossy.sigma = [3e4; 3];
    slab(&mut sim, lossy);
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: grids.clone(), objective: flux }, "cpu", &cpu_factory).unwrap();
    s.set_design(&random_design(&mut Rng(37), &grids)).unwrap();
    assert!(s.forward().unwrap().is_finite());
    let err = s.backward().err().expect("a diverged reverse pass must be refused");
    println!("28 fs, 3e4 S/m: {err}");
    assert!(err.to_string().contains("diverged") && err.to_string().contains("session"), "{err}");

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
    let mut s = ReverseSession::new(ReverseSpec { sim, grids: spec.grids.clone(), objective }, "cpu", &cpu_factory).unwrap();
    s.set_design(&random_design(&mut Rng(35), &spec.grids)).unwrap();
    assert!(s.forward().unwrap().is_finite());
    let err = s.backward().err().expect("a diverged reverse pass must be refused");
    println!("400 fs, 1e4 ... 3e4 S/m: {err}");
    assert!(err.to_string().contains("diverged") && err.to_string().contains("session"), "{err}");
}

/// [`reverse_scene`] with a permeability design grid in place of the
/// permittivity grid (medium "mudesign": eps of the background, mu up to 6)
/// and a magnetic slab (mu = 1.5, magnetic conductivity `sigma_m`) between
/// the design and the objective plane.
fn permeability_scene(sigma_m: f64) -> (Simulation, Vec<DesignGrid>, Objective) {
    let (mut sim, mut grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    let mut d = Medium::dielectric(2.1).with_name("mudesign");
    d.mu = [6.0; 3];
    sim.media.push(d);
    grids[0].medium = "mudesign".into();
    grids[0].permeability = true;
    let mut m = Medium::dielectric(2.1).with_name("magnetic");
    m.mu = [1.5; 3];
    m.sigma_m = [sigma_m; 3];
    slab(&mut sim, m);
    (sim, grids, flux)
}

#[test]
fn reverse_mode_permeability_gradient() {
    // FDTDX test_inverse_permeability_gradient_inside_devices (a magnetic
    // background, the gradient with respect to the permeability inside the
    // device) and TestReversibleVsCheckpointedGradient with magnetic, lossy
    // media (the permeability gradient with a magnetically conductive slab):
    // dL/d mu of a permeability design grid against central differences, on
    // the flux objective and on a phasor objective.
    for (name, sigma_m) in [("permeability design, magnetic slab", 0.0), ("permeability design, magnetic lossy slab", 3e8)] {
        let (sim, grids, flux) = permeability_scene(sigma_m);
        fd_check(&Check { name: format!("{name}, flux energy"), sim: sim.clone(), grids: grids.clone(), objective: flux, loss: None, tol: 3e-3 }, 38);
        let win = Window::all(vec![0.9 * F0, 1.1 * F0]);
        fd_check(&Check { name: format!("{name}, phasor power"), sim, grids, objective: phasor_plane(), loss: Some(phasor_power(win)), tol: 3e-3 }, 39);
    }
    // a permeability grid and a permittivity grid side by side
    let (mut sim, mut grids, flux) = permeability_scene(0.0);
    let mut eg = grids[0].clone();
    eg.name = "eps".into();
    eg.medium = "design".into();
    eg.permeability = false;
    eg.min[2] = 0.7 * UM;
    eg.max[2] = 0.9 * UM;
    eg.dims = [5, 6, 1];
    grids[0].max[2] = 0.65 * UM;
    grids.push(eg);
    sim.run_time = 28e-15;
    fd_check(&Check { name: "permeability + permittivity grids".into(), sim, grids, objective: flux, loss: None, tol: 3e-3 }, 40);
    // the permittivity gradient through a magnetically lossy block
    let mut m = Medium::dielectric(1.5).with_name("maglossy");
    m.mu = [2.0; 3];
    m.sigma_m = [3e8; 3];
    reverse_check(&Case { name: "permittivity design, magnetic lossy block", boundaries: Boundaries::all(pml(8)), design_sigma: 0.0, extra: Some(m), tol: 3e-3 });
}

/// Two dispersive slabs (across the region, not into the absorbing layers)
/// between the design and the objective plane: medium
/// "px" with a Lorentz pole on x only, medium "pz" with another Lorentz pole
/// and a Debye pole on z only (diagonally anisotropic eps_inf, conductive).
fn pole_scene(px: &[Pole], pz: &[Pole]) -> (Simulation, Vec<DesignGrid>, Objective) {
    let (mut sim, grids, flux) = reverse_scene(&base_case("", Boundaries::all(pml(8))));
    for (name, poles, axes, x0) in [("px", px, [true, false, false], 1.0), ("pz", pz, [false, false, true], 1.15)] {
        let mut m = Medium::dielectric(2.0).with_name(name);
        m.eps = [2.0, 2.5, 3.0];
        m.sigma = [100.0; 3];
        m.poles = poles.to_vec();
        m.pole_axes = axes;
        sim.media.push(m);
        // inside the region: the reverse pass does not reconstruct the
        // fields in the absorbing layers
        sim.structures.push(Structure { name: name.into(), shape: Shape::Box { min: [x0 * UM, 0.0, 0.0], max: [(x0 + 0.1) * UM, 1.2 * UM, 1.0 * UM] }, medium: name.into() });
    }
    (sim, grids, flux)
}

#[test]
fn reverse_mode_pole_coefficient_gradients() {
    // FDTDX TestPerAxisDispersiveGradientCheckpointed: the gradient with
    // respect to the coefficients of per-axis dispersive poles against
    // central differences. Here: dL/d(delta_eps, omega0, gamma) of a Lorentz
    // pole acting on x only, of another acting on z only, and dL/d(delta_eps,
    // tau) of a Debye pole on z, each from one reverse pass.
    use lightsim_core::autodiff::reverse::{pole_params, pole_with_params};
    let lx = Pole::Lorentz { delta_eps: 1.5, omega0: TAU * 1.4 * F0, gamma: 2.0e14 };
    let lz = Pole::Lorentz { delta_eps: 0.8, omega0: TAU * 1.2 * F0, gamma: 1.0e14 };
    let dz = Pole::Debye { delta_eps: 0.6, tau: 6.0e-15 };
    let eps = random_design(&mut Rng(42), &reverse_scene(&base_case("", Boundaries::all(pml(8)))).1);
    let value = |px: &[Pole], pz: &[Pole]| -> f64 {
        let (sim, grids, flux) = pole_scene(px, pz);
        let mut s = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).unwrap();
        evaluate(&mut s, &eps, None, false).0
    };
    let (sim, grids, flux) = pole_scene(&[lx.clone()], &[lz.clone(), dz.clone()]);
    let mut s = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).unwrap();
    s.set_pole_gradients(true).unwrap();
    let (l0, _) = evaluate(&mut s, &eps, None, true);
    let pg = s.pole_gradients().to_vec();
    // the x pole acts on Ex only, the z poles on Ez only
    assert!(pg.iter().all(|g| (g.medium == "px" && g.axis == 0) || (g.medium == "pz" && g.axis == 2)), "{pg:?}");
    let names = [["delta_eps", "omega0", "gamma"], ["delta_eps", "tau", ""]];
    for (medium, q) in [("px", 0usize), ("pz", 0), ("pz", 1)] {
        let g = pg.iter().find(|g| g.medium == medium && g.pole == q).unwrap();
        let pole = if medium == "px" { &lx } else if q == 0 { &lz } else { &dz };
        let v = pole_params(pole);
        for j in 0..v.len() {
            let set = |w: &[f64]| -> f64 {
                let p = pole_with_params(pole, w);
                match (medium, q) {
                    ("px", _) => value(&[p], &[lz.clone(), dz.clone()]),
                    (_, 0) => value(&[lx.clone()], &[p, dz.clone()]),
                    _ => value(&[lx.clone()], &[lz.clone(), p]),
                }
            };
            // central differences with Richardson extrapolation (steps 1 % and
            // 0.5 % of the parameter; 10 % and 5 % for the damping, whose
            // effect on the single-precision coefficients c0 ~ 2, c1 ~ -1 is
            // a few ulp at a 1 % step)
            let cd = |h: f64| -> f64 {
                let mut up = v.clone();
                let mut dn = v.clone();
                up[j] += h;
                dn[j] -= h;
                (set(&up) - set(&dn)) / (2.0 * h)
            };
            let damping = matches!(pole, Pole::Lorentz { .. }) && j == 2;
            let h = if damping { 0.1 } else { 1e-2 } * v[j];
            let fd = (4.0 * cd(0.5 * h) - cd(h)) / 3.0;
            let ad = g.params[j];
            let rel = (ad - fd).abs() / fd.abs();
            let name = names[if matches!(pole, Pole::Debye { .. }) { 1 } else { 0 }][j];
            println!("{medium} pole {q} (axis {}), dL/d {name:9}: reverse mode {ad:.6e}, finite difference {fd:.6e}, rel {rel:.2e}  (L = {l0:.4e})", g.axis);
            assert!(rel < 3e-3, "{medium} {q} {name}: {rel}");
        }
    }
    // a dispersive slab reaching into the absorbing layers is refused
    let (mut sim, grids, flux) = pole_scene(&[lx.clone()], &[lz]);
    if let Shape::Box { min, .. } = &mut sim.structures.last_mut().unwrap().shape {
        min[1] = -1.0;
    }
    let mut s = ReverseSession::new(ReverseSpec { sim, grids, objective: flux }, "cpu", &cpu_factory).unwrap();
    let err = s.set_pole_gradients(true).err().expect("dispersive cells in the PML must be refused");
    println!("refused: {err}");
    assert!(err.to_string().contains("absorbing layer"), "{err}");
}

#[test]
fn session_through_an_oriented_dispersive_medium() {
    // FDTDX TestOrientedDispersionGradients: the gradient through a medium
    // with an oriented dispersive pole (chi(w) u u^T, u along (1, 1, 0):
    // off-diagonal permittivity) next to the design is finite and non-trivial,
    // and agrees with central differences at the session's usual level. The
    // medium is reciprocal, so the S-matrix session's adjoint run is an
    // ordinary run; reverse mode refuses such a scene
    // (reverse_mode_refuses_oriented_dispersion). (With the pole at 1.4 f0
    // instead of 2 f0 the slab reflects more, the derivative is 5 times
    // smaller, and the same absolute error is 3.5 % of it.)
    let mut spec = plane_wave_session_spec();
    let mut m = Medium::dielectric(2.0).with_name("oriented");
    m.oriented = vec![OrientedPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: TAU * 2.0 * F0, gamma: 1e14 }, u: [1.0, 1.0, 0.0] }];
    spec.sim.media.push(m);
    spec.sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.6 * UM, -1.0, -1.0], max: [1.75 * UM, 1.0, 1.0] }, medium: "oriented".into() });
    let r = two_routes(&spec, 43, false);
    println!("oriented pole next to the design: along the gradient session {:.6e}, finite difference {:.6e}", r.steep.0, r.steep.1);
    assert!(r.steep.0.is_finite() && r.steep.0.abs() > 0.0);
    assert!((r.steep.0 - r.steep.1).abs() < 3e-2 * r.steep.1.abs(), "session vs finite differences");
    // the medium changes the result: the same scene without it
    let mut plain = plane_wave_session_spec();
    plain.sim.media.push(Medium::dielectric(2.0).with_name("iso"));
    plain.sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.6 * UM, -1.0, -1.0], max: [1.75 * UM, 1.0, 1.0] }, medium: "iso".into() });
    let q = two_routes(&plain, 43, false);
    println!("same slab without the pole: along the gradient {:.6e}", q.steep.0);
    assert!((q.steep.0 - r.steep.0).abs() > 1e-3 * r.steep.0.abs());
}
