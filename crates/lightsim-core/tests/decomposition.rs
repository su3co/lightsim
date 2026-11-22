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
