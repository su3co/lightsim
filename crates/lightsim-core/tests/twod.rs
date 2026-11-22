//! The 2-D update against the general 3-D update on the same one-cell grids
//! (CPU backend): the two evaluate the same terms in the same order, so the
//! fields must be equal, not merely close.

use lightsim_core::autodiff::engine::forward_step;
use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::{Medium, Pole};
use lightsim_core::monitors::{Apodization, Monitor};
use lightsim_core::plan::{CompileOptions, compile};
use lightsim_core::sources::{Direction, Source, TimeSignal};
use lightsim_core::spec::{Boundaries, Boundary, Polarization, Simulation};
use lightsim_core::{Backend, Results};

const UM: f64 = 1e-6;
const F0: f64 = 2.0e14;

fn sig() -> TimeSignal {
    TimeSignal::gaussian(F0, 0.4 * F0)
}

fn dipole(pos: [f64; 3], c: Component) -> Source {
    Source::PointDipole { name: format!("d{}", c.name()), position: pos, polarization: c, signal: sig(), amplitude: 1e-12 }
}

/// Axis permutation putting the invariant axis at `w`: in-plane coordinates
/// (a, b) -> a 3-vector with `c` along `w`.
fn lift(w: usize, a: f64, b: f64, c: f64) -> [f64; 3] {
    match w {
        0 => [c, a, b],
        1 => [a, c, b],
        _ => [a, b, c],
    }
}

/// A one-cell grid along `w` with the given in-plane boundaries.
fn base(w: usize, bu: [Boundary; 2], bv: [Boundary; 2], mesh: Option<MeshSpec>) -> Simulation {
    let dx = 40e-9;
    let mut b = Boundaries::all(Boundary::Periodic);
    let (u, v) = match w {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    *b.axis_mut(u) = bu;
    *b.axis_mut(v) = bv;
    let mut d = [dx; 3];
    d[w] = 60e-9;
    let mesh = mesh.unwrap_or(MeshSpec::Uniform { dx: d });
    Simulation::new(lift(w, -1.0 * UM, -0.8 * UM, 0.0), lift(w, 1.0 * UM, 0.8 * UM, 60e-9), mesh, b, 12e-15)
}

fn monitors(sim: &mut Simulation, w: usize) {
    let (u, _) = match w {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    sim.monitors.push(Monitor::DftField {
        name: "f".into(),
        min: lift(w, -0.5 * UM, -0.5 * UM, -1.0),
        max: lift(w, 0.5 * UM, 0.5 * UM, 1.0),
        components: Component::ALL.to_vec(),
        freqs: vec![0.9 * F0, 1.1 * F0],
        apodization: Apodization::default(),
        stride: 1,
    });
    let mut lo = lift(w, 0.6 * UM, -0.6 * UM, -1.0);
    let mut hi = lift(w, 0.6 * UM, 0.6 * UM, 1.0);
    lo[u] = 0.6 * UM;
    hi[u] = 0.6 * UM;
    sim.monitors.push(Monitor::Flux { name: "flux".into(), axis: u, position: 0.6 * UM, min: lo, max: hi, freqs: vec![F0], apodization: Apodization::default() });
    sim.monitors.push(Monitor::Probe { name: "p".into(), position: lift(w, 0.3 * UM, 0.2 * UM, 0.0), components: Component::ALL.to_vec() });
}

/// Step both paths and compare every field component and the monitors.
fn compare(sim: &Simulation, label: &str) {
    let run = |allow_2d: bool| -> (Vec<Vec<f32>>, lightsim_core::engine::RawResults, bool) {
        let plan = compile(sim, &CompileOptions { allow_2d, ..Default::default() }).unwrap();
        let two = plan.twod.is_some();
        let n = plan.nsteps.min(260);
        let mut be = CpuBackend::new(plan).unwrap();
        for s in 0..n {
            forward_step(&mut be, s).unwrap();
        }
        let f = Component::ALL.iter().map(|c| be.field(*c).unwrap()).collect();
        (f, be.download().unwrap(), two)
    };
    let (fa, ra, two_a) = run(true);
    let (fb, rb, two_b) = run(false);
    assert!(two_a, "{label}: the plan was not recognised as 2-D");
    assert!(!two_b);
    let mut peak = 0f32;
    for (c, (a, b)) in Component::ALL.iter().zip(fa.iter().zip(fb.iter())) {
        for (s, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(x == y, "{label}: {} differs at storage index {s}: {x:e} (2-D) vs {y:e} (3-D)", c.name());
            peak = peak.max(x.abs());
        }
    }
    assert!(peak > 0.0, "{label}: the fields are identically zero");
    assert_eq!(ra.dft.len(), rb.dft.len());
    for (da, db) in ra.dft.iter().zip(rb.dft.iter()) {
        for ((ar, ai), (br, bi)) in da.boxes.iter().zip(db.boxes.iter()) {
            assert!(ar == br && ai == bi, "{label}: DFT monitors differ");
        }
    }
    for (ta, tb) in ra.time.iter().zip(rb.time.iter()) {
        assert!(ta.boxes == tb.boxes, "{label}: time monitors differ");
    }
}

fn pml() -> [Boundary; 2] {
    [Boundary::Pml { layers: 8, params: Default::default(), inside: false }, Boundary::Pml { layers: 8, params: Default::default(), inside: false }]
}

fn scene(sim: &mut Simulation, w: usize) {
    sim.media.push(Medium::from_index(2.0).with_name("glass"));
    sim.media.push(Medium::conductor(2.25, 3e4).with_name("lossy"));
    sim.structures.push(Structure { name: "rod".into(), shape: Shape::Cylinder { center: lift(w, -0.2 * UM, 0.1 * UM, 0.0), radius: 0.25 * UM, length: 10.0, axis: w }, medium: "glass".into() });
    sim.structures.push(Structure { name: "blk".into(), shape: Shape::Box { min: lift(w, 0.25 * UM, -0.5 * UM, -1.0), max: lift(w, 0.5 * UM, -0.1 * UM, 1.0) }, medium: "lossy".into() });
}

#[test]
fn both_sets_every_invariant_axis() {
    for w in 0..3 {
        let mut sim = base(w, pml(), pml(), None);
        scene(&mut sim, w);
        // one dipole per field set
        sim.sources.push(dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w)));
        sim.sources.push(dipole(lift(w, -0.6 * UM, 0.3 * UM, 0.0), Component::h(w)));
        monitors(&mut sim, w);
        let plan = compile(&sim, &CompileOptions::default()).unwrap();
        let td = plan.twod.expect("2-D plan");
        assert_eq!((td.axis, td.active), (w, [true, true]));
        compare(&sim, &format!("both sets, invariant axis {w}"));
    }
}

#[test]
fn single_set_is_detected_from_the_sources() {
    for w in 0..3 {
        for (k, c) in [(0usize, Component::h(w)), (1usize, Component::e(w))] {
            let mut sim = base(w, pml(), pml(), None);
            scene(&mut sim, w);
            sim.sources.push(dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), c));
            monitors(&mut sim, w);
            let plan = compile(&sim, &CompileOptions::default()).unwrap();
            let mut want = [false, false];
            want[k] = true;
            assert_eq!(plan.twod.unwrap().active, want);
            compare(&sim, &format!("set {k}, invariant axis {w}"));
        }
    }
}

#[test]
fn walls_and_periodic_boundaries() {
    let cases: Vec<([Boundary; 2], [Boundary; 2])> = vec![
        ([Boundary::Pec, Boundary::Pmc], [Boundary::Pmc, Boundary::Pec]),
        ([Boundary::Periodic, Boundary::Periodic], pml()),
        (pml(), [Boundary::Periodic, Boundary::Periodic]),
        ([Boundary::Periodic, Boundary::Periodic], [Boundary::Periodic, Boundary::Periodic]),
        ([Boundary::Pmc, Boundary::Pml { layers: 8, params: Default::default(), inside: false }], [Boundary::Pec, Boundary::Pec]),
    ];
    for w in 0..3 {
        for (t, (bu, bv)) in cases.iter().enumerate() {
            let mut sim = base(w, bu.clone(), bv.clone(), None);
            scene(&mut sim, w);
            sim.sources.push(dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w)));
            sim.sources.push(dipole(lift(w, 0.1 * UM, 0.45 * UM, 0.0), Component::h(w)));
            monitors(&mut sim, w);
            compare(&sim, &format!("boundaries {t}, invariant axis {w}"));
        }
    }
}

#[test]
fn bloch_boundaries_complex_fields() {
    let k = 0.3 * std::f64::consts::TAU * F0 / 299_792_458.0;
    for w in 0..3 {
        for both in [false, true] {
            let bl = [Boundary::Bloch { k }, Boundary::Bloch { k }];
            let bu = if both { [Boundary::Bloch { k: -0.5 * k }, Boundary::Bloch { k: -0.5 * k }] } else { pml() };
            let mut sim = base(w, bu, bl, None);
            scene(&mut sim, w);
            sim.sources.push(dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w)));
            sim.sources.push(dipole(lift(w, 0.1 * UM, 0.45 * UM, 0.0), Component::h(w)));
            monitors(&mut sim, w);
            let plan = compile(&sim, &CompileOptions::default()).unwrap();
            assert!(plan.bloch && plan.twod.is_some());
            compare(&sim, &format!("Bloch (both axes: {both}), invariant axis {w}"));
        }
    }
}

#[test]
fn dispersive_nonlinear_and_magnetic_media() {
    for w in 0..3 {
        let mut sim = base(w, pml(), pml(), None);
        let om = std::f64::consts::TAU * F0;
        sim.media.push(Medium::dielectric(2.0).with_name("lor").with_poles(vec![Pole::Lorentz { delta_eps: 1.5, omega0: 1.8 * om, gamma: 0.05 * om }, Pole::Drude { omega_p: 0.4 * om, gamma: 0.1 * om }]));
        let mut kerr = Medium::from_index(1.8).with_name("kerr");
        kerr.chi3 = [2e-18; 3];
        kerr.chi2 = [1e-10; 3];
        sim.media.push(kerr);
        let mut mag = Medium::from_index(1.3).with_name("mag");
        mag.mu = [1.4, 1.7, 2.0];
        sim.media.push(mag);
        sim.structures.push(Structure { name: "a".into(), shape: Shape::Cylinder { center: lift(w, -0.2 * UM, 0.1 * UM, 0.0), radius: 0.2 * UM, length: 10.0, axis: w }, medium: "lor".into() });
        sim.structures.push(Structure { name: "b".into(), shape: Shape::Box { min: lift(w, 0.2 * UM, -0.5 * UM, -1.0), max: lift(w, 0.5 * UM, -0.1 * UM, 1.0) }, medium: "kerr".into() });
        sim.structures.push(Structure { name: "c".into(), shape: Shape::Box { min: lift(w, 0.1 * UM, 0.2 * UM, -1.0), max: lift(w, 0.6 * UM, 0.5 * UM, 1.0) }, medium: "mag".into() });
        let mut d = dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w));
        if let Source::PointDipole { amplitude, .. } = &mut d {
            *amplitude = 3e-9; // strong enough for the chi3 term to matter
        }
        sim.sources.push(d);
        sim.sources.push(dipole(lift(w, 0.0, 0.45 * UM, 0.0), Component::h(w)));
        monitors(&mut sim, w);
        let plan = compile(&sim, &CompileOptions::default()).unwrap();
        assert!(plan.disp.is_some() && plan.nonlin.is_some() && plan.ch.is_some() && plan.twod.is_some());
        compare(&sim, &format!("dispersive / chi2 / chi3 / magnetic, invariant axis {w}"));
    }
}

#[test]
fn planar_sources_and_graded_mesh() {
    // plane wave, oblique plane wave (complex profile), Gaussian beam and a
    // TFSF box that spans the invariant axis, on a graded mesh
    let k0 = std::f64::consts::TAU * F0 / 299_792_458.0;
    for w in [2usize, 1] {
        let (u, v) = if w == 2 { (0usize, 1usize) } else { (0, 2) };
        let th = 0.3f64;
        for kind in 0..4 {
            let mesh = MeshSpec::Auto { wavelength: 1.5 * UM, steps_per_wavelength: 24.0, grading: 1.3, min_step: None, max_step: None, overrides: vec![], symmetry: [None; 3] };
            let bv = if kind == 1 { [Boundary::Bloch { k: k0 * th.sin() }, Boundary::Bloch { k: k0 * th.sin() }] } else if kind == 0 { [Boundary::Periodic, Boundary::Periodic] } else { pml() };
            let mut sim = base(w, pml(), bv, Some(mesh));
            sim.region.max[w] = 25e-9; // one cell also on the automatic mesh
            sim.media.push(Medium::from_index(2.0).with_name("glass"));
            sim.structures.push(Structure { name: "rod".into(), shape: Shape::Cylinder { center: lift(w, 0.1 * UM, 0.0, 0.0), radius: 0.2 * UM, length: 10.0, axis: w }, medium: "glass".into() });
            let pol_in = Component::e(v);
            let pol_out = Component::e(w);
            for pol in [pol_in, pol_out] {
                let src = match kind {
                    0 => Source::PlaneWave { name: "pw".into(), axis: u, position: -0.7 * UM, direction: Direction::Plus, polarization: pol, signal: sig(), amplitude: 1.0, angle: 0.0, min: None, max: None },
                    1 => {
                        // the legacy source tilts toward the other transverse axis; only the
                        // out-of-plane polarisation tilts in the plane
                        if pol == pol_in {
                            continue;
                        }
                        Source::PlaneWave { name: "pw".into(), axis: u, position: -0.7 * UM, direction: Direction::Plus, polarization: pol, signal: sig(), amplitude: 1.0, angle: th, min: None, max: None }
                    }
                    2 => Source::GaussianBeam { name: "gb".into(), axis: u, position: -0.7 * UM, direction: Direction::Plus, polarization: pol, center: lift(w, 0.0, 0.0, 30e-9), waist: 0.4 * UM, signal: sig(), amplitude: 1.0, focus_distance: 0.5 * UM, theta: 0.0, phi: 0.0 },
                    _ => Source::Tfsf { name: "tf".into(), min: lift(w, -0.6 * UM, -0.5 * UM, -1.0), max: lift(w, 0.6 * UM, 0.5 * UM, 1.0), axis: u, direction: Direction::Plus, polarization: pol, signal: sig(), amplitude: 1.0 },
                };
                let mut s = sim.clone();
                s.sources.push(src);
                monitors(&mut s, w);
                compare(&s, &format!("planar source {kind} {}, invariant axis {w}", pol.name()));
            }
        }
    }
}

/// Run on `nslabs` x-slabs of the CPU backend with or without the 2-D update.
fn run_slabs(sim: &Simulation, nslabs: usize, allow_2d: bool) -> Results {
    let grid = sim.build_grid().unwrap();
    let parts = lightsim_core::plan::decompose_x(grid.n()[0], &vec![1.0; nslabs]);
    let mut slabs: Vec<Box<dyn Backend>> = Vec::new();
    for (r, &(x0, nxl)) in parts.iter().enumerate() {
        let opts = CompileOptions { slab: Some((x0, nxl)), nslabs, slab_index: r, allow_2d, ..Default::default() };
        slabs.push(Box::new(CpuBackend::new(lightsim_core::plan::compile_with_grid(sim, grid.clone(), &opts).unwrap()).unwrap()));
    }
    let mut chain = lightsim_core::run::Chain::new(slabs);
    chain.local_wrap = nslabs > 1 && sim.boundaries.x[0].is_periodic();
    lightsim_core::run::run_chain(&mut chain, &Default::default()).unwrap()
}

#[test]
fn slab_decomposition() {
    // x-slabs with the 2-D update: equal to the 3-D update on the same
    // slabs, and to the single slab within the usual decomposition tolerance
    for (w, bx) in [(2usize, pml()), (1, pml()), (2, [Boundary::Periodic, Boundary::Periodic])] {
        let mut sim = base(w, bx, pml(), None);
        scene(&mut sim, w);
        sim.sources.push(dipole(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w)));
        sim.sources.push(dipole(lift(w, 0.3 * UM, 0.3 * UM, 0.0), Component::h(w)));
        monitors(&mut sim, w);
        let one = run_slabs(&sim, 1, true);
        let two = run_slabs(&sim, 3, true);
        let ref3 = run_slabs(&sim, 3, false);
        assert!(one.stats.backend.contains("2-D te+tm"), "{}", one.stats.backend);
        assert!(two.stats.backend.contains("2-D te+tm"), "{}", two.stats.backend);
        assert!(!ref3.stats.backend.contains("2-D"), "{}", ref3.stats.backend);
        for c in Component::ALL {
            let (a, b, r) = (one.dft["f"].field(c).unwrap(), two.dft["f"].field(c).unwrap(), ref3.dft["f"].field(c).unwrap());
            assert!(b.re == r.re && b.im == r.im, "{}: 2-D and 3-D updates differ on 3 slabs", c.name());
            let d = lightsim_core::validation::dft_rel_diff(&one, &two, "f", c);
            assert!(d < 1e-5, "{}: max |single - 3 slabs| / peak = {d:.3e}", c.name());
            let _ = a;
        }
        assert!(two.flux["flux"] == ref3.flux["flux"]);
    }
}

#[test]
fn explicit_two_d_spec() {
    // dimensions = 2: one cell whatever the mesh says, periodic whatever the
    // boundaries say, the 2-D Courant limit, structures cut by the mid-plane
    let dx = 40e-9;
    let mut sim = Simulation::new([-1.0 * UM, -0.8 * UM, -0.5], [1.0 * UM, 0.8 * UM, 0.5], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 12e-15).two_d(2, Polarization::Tm);
    sim.media.push(Medium::from_index(2.0).with_name("glass"));
    // a sphere centred 0.15 um above the slice plane: its cross-section is a disc of radius 0.2 um
    sim.structures.push(Structure { name: "ball".into(), shape: Shape::Sphere { center: [0.0, 0.0, 0.15 * UM], radius: 0.25 * UM }, medium: "glass".into() });
    // a box that the slice plane misses
    sim.structures.push(Structure { name: "off".into(), shape: Shape::Box { min: [0.4 * UM, -0.5 * UM, 0.1 * UM], max: [0.8 * UM, 0.5 * UM, 0.3 * UM] }, medium: "glass".into() });
    sim.sources.push(dipole([-0.5 * UM, 0.0, 7.0], Component::Ez));
    let g = sim.build_grid().unwrap();
    assert_eq!(g.n()[2], 1);
    assert_eq!((g.z[0], g.z[1]), (-0.5, 0.5));
    let dt = sim.time_step(&g).unwrap();
    let dt2 = 0.99 * dx / (299_792_458.0 * 2f64.sqrt());
    assert!((dt / dt2 - 1.0).abs() < 1e-12, "{dt} vs {dt2}");
    let plan = compile(&sim, &CompileOptions::default()).unwrap();
    let td = plan.twod.unwrap();
    assert_eq!((td.axis, td.allow, td.active), (2, [false, true], [false, true]));
    assert_eq!(plan.bc_z, [1, 1]);
    // eps of Ez from cb = c dt / eps at a few points
    let eps = |x: f64, y: f64| -> f64 {
        let (i, j) = (g.nearest_node(0, x), g.nearest_node(1, y));
        299_792_458.0 * plan.dt / plan.cb[2][plan.sidx(i, j, 0)] as f64
    };
    assert!((eps(0.0, 0.0) - 4.0).abs() < 1e-4);
    assert!((eps(0.12 * UM, 0.0) - 4.0).abs() < 1e-4, "inside the disc of radius 0.2 um");
    assert!((eps(0.24 * UM, 0.0) - 1.0).abs() < 1e-4, "outside the disc, inside the sphere's radius");
    assert!((eps(0.6 * UM, 0.0) - 1.0).abs() < 1e-4, "the box above the plane is not in the model");
    // a source of the other set is an error, not a silent no-op
    let mut bad = sim.clone();
    bad.sources = vec![dipole([0.0; 3], Component::Hz)];
    let e = compile(&bad, &CompileOptions::default()).unwrap_err().to_string();
    assert!(e.contains("drives only the te"), "{e}");
    // the same problem through the 3-D kernels gives the same fields
    let mut forced = sim.clone();
    forced.force_3d_kernels = true;
    assert!(compile(&forced, &CompileOptions::default()).unwrap().twod.is_none());
    monitors(&mut sim, 2);
    compare(&sim, "explicit 2-D spec");
    // device "auto": a small 2-D grid goes to the CPU, a large one and a 3-D one do not
    assert!(sim.auto_prefers_cpu());
    let mut large = sim.clone();
    large.region.max[0] = 40.0 * UM;
    large.region.max[1] = 40.0 * UM;
    assert!(!large.auto_prefers_cpu());
    let mut three = sim.clone();
    three.dimensions = 3;
    three.polarization = Polarization::Both;
    three.region.min[2] = -0.5 * UM;
    three.region.max[2] = 0.5 * UM;
    assert!(!three.auto_prefers_cpu());
    // JSON: polarization and axis by name
    let j = sim.to_json();
    assert!(j.contains("\"dimensions\": 2") && j.contains("\"polarization\": \"tm\""), "{j}");
    assert_eq!(Simulation::from_json(&j).unwrap(), sim);
    let y = Simulation::from_json(&j.replace("\"dimensions\": 2,", "\"dimensions\": 2, \"invariant_axis\": \"y\",")).unwrap();
    assert_eq!(y.invariant(), Some(1));
    let x = Simulation::from_json(&j.replace("\"dimensions\": 2,", "\"dimensions\": 2, \"invariant_axis\": 0,")).unwrap();
    assert_eq!(x.invariant(), Some(0));
}
