//! Physics validation on the CPU reference backend.
//!
//! Each test runs one case from `lightsim_core::validation` and asserts the
//! analytic checks. The same cases run on the GPU in `lightsim-cuda`.

use lightsim_core::validation::{self, CaseReport};
use lightsim_core::{Results, Simulation};

fn cpu(sim: &Simulation) -> anyhow::Result<Results> {
    lightsim_core::run_cpu(sim)
}

fn assert_case(r: anyhow::Result<CaseReport>) {
    let r = r.expect("case failed to run");
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn plane_wave_vacuum() {
    assert_case(validation::plane_wave_vacuum(&cpu));
}

#[test]
fn fresnel_normal() {
    assert_case(validation::fresnel_normal(&cpu));
}

#[test]
fn pec_pmc_reflection() {
    assert_case(validation::pec_pmc_reflection(&cpu));
}

#[test]
fn dipole_radiation() {
    assert_case(validation::dipole_radiation(&cpu));
}

#[test]
fn pml_reflection() {
    assert_case(validation::pml_reflection(&cpu));
}

#[test]
fn skin_depth() {
    assert_case(validation::skin_depth(&cpu));
}

#[test]
fn magnetic_loss() {
    assert_case(validation::magnetic_loss(&cpu));
}

#[test]
fn lorentz_slab() {
    assert_case(validation::lorentz_slab(&cpu));
}

#[test]
fn birefringence() {
    assert_case(validation::birefringence(&cpu));
}

#[test]
fn nonuniform_grid() {
    assert_case(validation::nonuniform_grid(&cpu));
}

#[test]
fn magnetic_dipole() {
    assert_case(validation::magnetic_dipole(&cpu));
}

#[test]
fn dipole_green_function() {
    assert_case(validation::dipole_green_function(&cpu));
}

#[test]
fn symmetry_quarter_domain() {
    assert_case(validation::symmetry_quarter_domain(&cpu));
}

#[test]
fn tfsf_leakage() {
    assert_case(validation::tfsf_leakage(&cpu));
}

#[test]
fn mie_scattering() {
    assert_case(validation::mie_scattering(&cpu));
}

#[test]
fn gaussian_beam() {
    assert_case(validation::gaussian_beam(&cpu));
}

#[test]
fn gaussian_beam_tilted() {
    assert_case(validation::gaussian_beam_tilted(&cpu));
}

#[test]
fn dispersive_propagation() {
    assert_case(validation::dispersive_propagation(&cpu));
}

#[test]
fn anisotropic_dispersion() {
    assert_case(validation::anisotropic_dispersion(&cpu));
}

#[test]
fn energy_conservation() {
    assert_case(validation::energy_conservation(&cpu));
}

#[test]
fn stored_energy() {
    assert_case(validation::stored_energy(&cpu));
}

#[test]
fn reciprocity() {
    assert_case(validation::reciprocity(&cpu));
}

#[test]
fn bloch_zero_vector() {
    assert_case(validation::bloch_zero_vector(&cpu));
}

#[test]
fn pulsed_vs_cw() {
    assert_case(validation::pulsed_vs_cw(&cpu));
}

#[test]
fn apodization_leakage() {
    assert_case(validation::apodization_leakage(&cpu));
}

#[test]
fn dipole_far_field() {
    assert_case(validation::dipole_far_field(&cpu));
}

#[test]
fn bloch_oblique() {
    assert_case(validation::bloch_oblique(&cpu));
}

#[test]
fn bloch_oblique_p() {
    assert_case(validation::bloch_oblique_p(&cpu));
}

#[test]
fn chi2_second_harmonic() {
    assert_case(validation::chi2_second_harmonic(&cpu));
}

#[test]
fn chi3_kerr() {
    assert_case(validation::chi3_kerr(&cpu));
}

#[test]
fn gyrotropic_faraday() {
    assert_case(validation::gyrotropic_faraday(&cpu));
}

#[test]
fn director_field() {
    assert_case(validation::director_field(&cpu));
}

#[test]
fn gain_four_level() {
    assert_case(validation::gain_four_level(&cpu));
}

#[test]
fn chi3_raman() {
    assert_case(validation::chi3_raman(&cpu));
}

#[test]
fn offdiagonal_tensor() {
    assert_case(validation::offdiagonal_tensor(&cpu));
}

#[test]
fn plane_wave_impedance() {
    assert_case(validation::plane_wave_impedance(&cpu));
}

#[test]
fn mixed_boundaries() {
    assert_case(validation::mixed_boundaries(&cpu));
}

#[test]
fn dipole_in_dielectric() {
    assert_case(validation::dipole_in_dielectric(&cpu));
}

#[test]
fn drude_reflection() {
    assert_case(validation::drude_reflection(&cpu));
}

#[test]
fn anisotropic_cells() {
    assert_case(validation::anisotropic_cells(&cpu));
}

#[test]
fn phasor_recording() {
    assert_case(validation::phasor_recording(&cpu));
}

#[test]
fn lossy_offdiagonal_tensor() {
    assert_case(validation::lossy_offdiagonal_tensor(&cpu));
}

#[test]
fn oriented_dispersion() {
    assert_case(validation::oriented_dispersion(&cpu));
}

#[test]
fn dispersive_background_sources() {
    assert_case(validation::dispersive_background_sources(&cpu));
}

#[test]
fn index_grid() {
    assert_case(validation::index_grid(&cpu));
}

#[test]
fn two_d_matches_one_cell() {
    let r = validation::two_d_matches_one_cell(&cpu).expect("case failed to run");
    println!("{}", r.summary());
    // same expressions in the same order on the CPU: not a rounding error apart
    assert!(r.checks.iter().all(|(_, m, _, _, _)| *m == 0.0), "{}", r.summary());
}

#[test]
fn two_d_fresnel_oblique() {
    assert_case(validation::two_d_fresnel_oblique(&cpu));
}

#[test]
fn two_d_mie_cylinder() {
    assert_case(validation::two_d_mie_cylinder(&cpu));
}

#[test]
fn two_d_metal_cylinder() {
    assert_case(validation::two_d_metal_cylinder(&cpu));
}

#[test]
fn two_d_slab_mode() {
    assert_case(validation::two_d_slab_mode(&cpu));
}

#[test]
fn two_d_cpml() {
    assert_case(validation::two_d_cpml(&cpu));
}

#[test]
fn two_d_grating() {
    assert_case(validation::two_d_grating(&cpu));
}

#[test]
fn two_d_gaussian_beam() {
    assert_case(validation::two_d_gaussian_beam(&cpu));
}

#[test]
fn dipole_tilted() {
    assert_case(validation::dipole_tilted(&cpu));
}

#[test]
fn hyperbolic_drude() {
    assert_case(validation::hyperbolic_drude(&cpu));
}

#[test]
fn fixed_angle_slab() {
    assert_case(validation::fixed_angle_slab(&cpu));
}

#[test]
fn fixed_angle_brewster() {
    assert_case(validation::fixed_angle_brewster(&cpu));
}

#[test]
fn fixed_angle_dispersive() {
    assert_case(validation::fixed_angle_dispersive(&cpu));
}

#[test]
fn fixed_angle_grating() {
    assert_case(validation::fixed_angle_grating(&cpu));
}

#[test]
fn fixed_angle_stability() {
    assert_case(validation::fixed_angle_stability(&cpu));
}

#[test]
fn fixed_angle_3d() {
    assert_case(validation::fixed_angle_3d(&cpu));
fn sheet_normal_incidence() {
    assert_case(validation::sheet_normal_incidence(&cpu));
}

#[test]
fn sheet_oblique() {
    assert_case(validation::sheet_oblique(&cpu));
}

#[test]
fn sheet_plasmon() {
    assert_case(validation::sheet_plasmon(&cpu));
}

#[test]
fn sheet_ribbon() {
    assert_case(validation::sheet_ribbon(&cpu));
}

#[test]
fn plugin_media() {
    assert_case(validation::plugin_media(&cpu));
}

#[test]
fn gain_two_electron() {
    assert_case(validation::gain_two_electron(&cpu));
}

#[test]
fn hermitian_tensor() {
    assert_case(validation::hermitian_tensor(&cpu));
}
