//! Fixed-angle runs (`Simulation::fixed_angle`): the time-step rule and the
//! scenes the mode refuses. The physics cases are in `physics.rs`.

use lightsim_core::autodiff::engine::cpu_factory;
use lightsim_core::autodiff::testing::{Case, plane_wave_session_spec, pml, reverse_scene};
use lightsim_core::autodiff::{GradientSession, ReverseSMatrix, ReverseSession, ReverseSpec};
use lightsim_core::engine::fixed_angle::{self, time_step_factor};
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::{GyroPole, Medium, Pole};
use lightsim_core::monitors::{Apodization, Monitor};
use lightsim_core::plan::{CompileOptions, compile};
use lightsim_core::sources::{Direction, Source, TimeSignal};
use lightsim_core::spec::{Boundaries, Boundary, Polarization, Simulation};

const F0: f64 = 3e14;

/// A small fixed-angle scene: 3-D or 2-D, tilt along y, propagation along x.
fn scene(two_d: bool, deg: f64) -> Simulation {
    let dx = 25e-9;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [2e-6, 8.0 * dx, 8.0 * dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 50e-15);
    if two_d {
        sim = sim.two_d(2, Polarization::Both);
    }
    sim.fixed_angle = true;
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 0.5e-6, direction: Direction::Plus, polarization: Component::Ez, signal: TimeSignal::gaussian(F0, 0.2 * F0), amplitude: 1.0, angle: deg.to_radians(), min: None, max: None });
    sim.monitors.push(Monitor::Flux { name: "T".into(), axis: 0, position: 1.5e-6, min: [1.5e-6, -1.0, -1.0], max: [1.5e-6, 1.0, 1.0], freqs: vec![F0], apodization: Apodization::default() });
    sim
}

fn refused(sim: &Simulation, what: &str) {
    let err = lightsim_core::run_cpu(sim).err().unwrap_or_else(|| panic!("{what}: the scene must be refused"));
    println!("{what}: {err}");
}

#[test]
fn time_step_rule() {
    // c dt <= min(Courant, sqrt(1 - s^2) / sqrt(sum over a != u of 1 / d_a^2))
    let cases = [(false, 30.0, 1.0), (false, 60.0, 0.5 * (1.5f64).sqrt()), (true, 30.0, 1.0), (true, 45.0, 1.0), (true, 60.0, 0.5 * 2f64.sqrt()), (true, 80.0, 80f64.to_radians().cos() * 2f64.sqrt())];
    for (two_d, deg, want) in cases {
        let got = time_step_factor(&scene(two_d, deg)).unwrap();
        println!("{} {deg} deg: dt factor {got:.4} (expected {want:.4})", if two_d { "2-D" } else { "3-D" });
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}

#[test]
fn refuses_unsupported_scenes() {
    // angle beyond the limit
    refused(&scene(true, 81.0), "angle above 80 degrees");
    // Bloch boundaries along the tilt axis
    let mut s = scene(true, 30.0);
    s.boundaries.y = [Boundary::Bloch { k: 1e6 }, Boundary::Bloch { k: 1e6 }];
    let e = lightsim_core::run_cpu(&s).err().unwrap();
    assert!(e.to_string().contains("Bloch"), "{e}");
    // a second source
    let mut s = scene(true, 30.0);
    s.sources.push(Source::PointDipole { name: "d".into(), position: [1e-6, 0.1e-6, 0.0], polarization: Component::Ez, signal: TimeSignal::gaussian(F0, 0.2 * F0), amplitude: 1.0 });
    refused(&s, "a second source");
    // time-domain monitors
    let mut s = scene(true, 30.0);
    s.monitors.push(Monitor::Probe { name: "p".into(), position: [1e-6, 0.1e-6, 0.0], components: vec![Component::Ez] });
    refused(&s, "a probe");
    // media the mode does not support
    let mut nl = Medium::dielectric(2.0).with_name("kerr");
    nl.chi3 = [1e-20; 3];
    let mut gyro = Medium::dielectric(2.0).with_name("gyro");
    gyro.gyro = vec![GyroPole { pole: Pole::Lorentz { delta_eps: 0.5, omega0: 6e15, gamma: 1e13 }, bias: [0.0, 0.0, 1e15] }];
    let mut off = Medium::dielectric(2.0).with_name("tensor");
    off.eps_off = [0.2, 0.0, 0.0];
    let mut gain = Medium::dielectric(2.0).with_name("gain");
    gain.gain = Some(lightsim_core::materials::FourLevel { density: 1e24, omega: 2e15, gamma: 1e13, tau32: 1e-13, tau21: 1e-12, tau10: 1e-13, pump_rate: 0.0, coupling: None, initial: [1.0, 0.0, 0.0, 0.0] });
    for m in [nl, gyro, off, gain] {
        let mut s = scene(true, 30.0);
        let name = m.name.clone();
        s.media.push(m);
        s.structures.push(Structure { name: "x".into(), shape: Shape::Box { min: [1.0e-6, -1.0, -1.0], max: [1.2e-6, 1.0, 1.0] }, medium: name.clone() });
        let e = lightsim_core::run_cpu(&s).err().unwrap_or_else(|| panic!("medium {name} must be refused"));
        assert!(e.to_string().contains("does not support"), "{e}");
    }
    // a dispersive medium at the source plane
    let mut s = scene(true, 30.0);
    s.media.push(Medium::dielectric(1.0).with_name("drude").with_poles(vec![Pole::Drude { omega_p: 1e16, gamma: 1e13 }]));
    s.background = "drude".into();
    refused(&s, "a dispersive medium at the source plane");
    // every path that compiles a plan other than the fixed-angle engine's
    let s = scene(false, 30.0);
    let e = compile(&s, &CompileOptions::default()).err().unwrap();
    assert!(e.to_string().contains(fixed_angle::ENGINE_ONLY), "{e}");
}

#[test]
fn gradient_engines_refuse_fixed_angle_runs() {
    let mut spec = plane_wave_session_spec();
    spec.sim.fixed_angle = true;
    let e = GradientSession::new(spec.clone(), &["cpu".to_string()], &cpu_factory).err().expect("the S-matrix session must refuse a fixed-angle run");
    assert!(e.to_string().contains("fixed-angle"), "{e}");
    let e = ReverseSMatrix::new(spec, &["cpu".to_string()], &cpu_factory).err().expect("engine reverse must refuse a fixed-angle run");
    assert!(e.to_string().contains("fixed-angle"), "{e}");
    let (mut sim, grids, objective) = reverse_scene(&Case { name: "", boundaries: Boundaries::all(pml(8)), design_sigma: 0.0, extra: None, tol: 0.0 });
    sim.fixed_angle = true;
    let e = ReverseSession::new(ReverseSpec { sim, grids, objective }, "cpu", &cpu_factory).err().expect("reverse mode must refuse a fixed-angle run");
    assert!(e.to_string().contains("fixed-angle"), "{e}");
}

#[test]
fn x_slabs_reproduce_the_single_slab_run() {
    for three_d in [false, true] {
        let sim = lightsim_core::validation::fixed_angle_scene(three_d);
        let one = lightsim_core::run_cpu(&sim).unwrap();
        let three = lightsim_core::run_cpu_slabs(&sim, 3).unwrap();
        let mut worst = 0f64;
        for c in Component::ALL {
            worst = worst.max(lightsim_core::validation::dft_rel_diff(&one, &three, "f", c));
        }
        for m in ["R", "T"] {
            for (a, b) in one.flux[m].iter().zip(&three.flux[m]) {
                worst = worst.max((a - b).abs() / one.flux["T"][1].abs());
            }
        }
        println!("{}: 1 vs 3 x-slabs, max relative difference {worst:.2e} ({})", if three_d { "3-D" } else { "2-D" }, three.stats.backend);
        assert!(worst < 1e-5, "{worst}");
    }
    // the tilt axis cannot be split
    let mut s = scene(false, 30.0);
    s.boundaries.x = [Boundary::Periodic, Boundary::Periodic];
    s.boundaries.z = [Boundary::pml(), Boundary::pml()];
    assert!(lightsim_core::run_cpu_slabs(&s, 2).is_err());
}
