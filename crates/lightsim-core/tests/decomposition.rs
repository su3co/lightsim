//! Domain decomposition on the CPU reference backend: x-slab runs must match
//! single-slab runs, including Bloch (complex-field) and nonlinear media.

use lightsim_core::grid::Component;
use lightsim_core::validation::{self, bloch_xy_box, dft_rel_diff};
use lightsim_core::{run_cpu, run_cpu_slabs};

#[test]
fn bloch_xy_three_slabs_match_single_slab() {
    let sim = bloch_xy_box();
    let a = run_cpu(&sim).unwrap();
    let b = run_cpu_slabs(&sim, 3).unwrap();
    for c in [Component::Ez, Component::Hx] {
        let d = dft_rel_diff(&a, &b, "f", c);
        assert!(d < 1e-5, "{c:?}: max |single - 3 slabs| / peak = {d:.3e}");
    }
}

#[test]
fn bloch_oblique_on_three_slabs() {
    let r = validation::bloch_oblique(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn bloch_oblique_p_on_two_slabs() {
    let r = validation::bloch_oblique_p(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn chi2_on_three_slabs() {
    let r = validation::chi2_second_harmonic(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn chi3_raman_on_three_slabs() {
    let r = validation::chi3_raman(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn gyrotropic_faraday_on_two_slabs() {
    let r = validation::gyrotropic_faraday(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn director_field_on_two_slabs() {
    let r = validation::director_field(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn gain_four_level_on_three_slabs() {
    // (the population totals are summed over the slabs)
    let r = validation::gain_four_level(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn offdiagonal_tensor_on_two_slabs() {
    let r = validation::offdiagonal_tensor(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn lossy_offdiagonal_tensor_on_two_slabs() {
    let r = validation::lossy_offdiagonal_tensor(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn oriented_dispersion_on_two_slabs() {
    let r = validation::oriented_dispersion(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn dispersive_background_sources_on_two_slabs() {
    let r = validation::dispersive_background_sources(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn stored_energy_on_three_slabs() {
    // the energy monitor sums the contributions of the slabs
    let r = validation::stored_energy(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn sheet_normal_incidence_on_three_slabs() {
    let r = validation::sheet_normal_incidence(&|s| run_cpu_slabs(s, 3)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn sheet_oblique_on_two_slabs() {
    let r = validation::sheet_oblique(&|s| run_cpu_slabs(s, 2)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn sheet_across_a_slab_boundary() {
    // a sheet normal to x between the last plane of one slab and the first
    // of the next (explicit mesh: shared by the two planes) and a sheet on
    // the first plane of a slab give the single-slab fields
    use lightsim_core::geometry::{Shape, Structure};
    use lightsim_core::monitors::{Apodization, Monitor};
    use lightsim_core::sheet::{Sheet, SheetConductivity, SheetPole};
    use lightsim_core::sources::{Direction, Source, TimeSignal};
    use lightsim_core::{Boundaries, Boundary, Medium, MeshSpec, Simulation};
    let dx = 25e-9;
    let nx = 160usize;
    let x: Vec<f64> = (0..=nx).map(|i| i as f64 * dx).collect();
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [nx as f64 * dx, 2.0 * dx, 2.0 * dx], MeshSpec::Explicit { x, y: vec![0.0, dx, 2.0 * dx], z: vec![0.0, dx, 2.0 * dx] }, b, 60e-15);
    // with the 12 PML cells, two slabs split at grid plane 92: the first
    // sheet sits between planes 91 and 92 (region planes 79 and 80)
    let c = SheetConductivity { poles: vec![SheetPole::Drude { drude_weight: 4e12, gamma: 3e14 }], conductance: 1e-3, ..Default::default() };
    let sheet = |x: f64| Sheet { name: format!("s{x}"), axis: 0, position: x, conductivity: c.clone(), min: None, max: None, polygon: None };
    sim.sheets.push(sheet(79.4 * dx));
    sim.sheets.push(sheet(120.0 * dx));
    sim.media.push(Medium::from_index(1.5).with_name("glass"));
    sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [100.0 * dx, -1.0, -1.0], max: [1.0, 1.0, 1.0] }, medium: "glass".into() });
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 30.0 * dx, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(3e14, 1e14), amplitude: 1.0, angle: 0.0, min: None, max: None });
    sim.monitors.push(Monitor::DftField { name: "f".into(), min: [0.0, 0.0, 0.0], max: [1.0, 1.0, 1.0], components: vec![Component::Ey], freqs: vec![2.5e14, 3e14, 3.5e14], apodization: Apodization::default(), stride: 1 });
    let a = run_cpu(&sim).unwrap();
    for n in [2, 3] {
        let b = run_cpu_slabs(&sim, n).unwrap();
        let d = dft_rel_diff(&a, &b, "f", Component::Ey);
        println!("{n} slabs: max |single - slabs| / peak = {d:.3e}");
        assert!(d < 1e-5, "{n} slabs: {d:.3e}");
    }
}
