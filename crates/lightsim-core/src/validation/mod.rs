//! Physics validation cases with analytic references.
//!
//! These mirror the physics test-suite of FDTDX (`tests/simulation/physics`)
//! and the classic Meep/Lumerical validation examples. Each case builds a
//! [`Simulation`], runs it through a caller-supplied runner (CPU reference or
//! GPU backend), compares against an analytic result and returns a
//! [`CaseReport`]. The Rust integration tests (`tests/physics.rs`) and the GPU
//! tests in `lightsim-cuda` call the same functions.

use crate::Real;
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

// One module per family of cases. Every case and reference formula is
// re-exported here: `validation::fresnel_normal`, `validation::mie_q` and the
// others keep their paths.
mod plane_waves;
mod dipoles;
mod boundaries;
mod scattering;
mod media;
mod disp_subpixel;
mod anisotropic;
mod herm;
mod nonlinear;
mod gain;
mod grids;
mod two_d;
mod fixed_angle;
mod sheets;
mod plugins;

pub use plane_waves::*;
pub use dipoles::*;
pub use boundaries::*;
pub use scattering::*;
pub use media::*;
pub use disp_subpixel::*;
pub use anisotropic::*;
pub use herm::*;
pub use nonlinear::*;
pub use gain::*;
pub use grids::*;
pub use two_d::*;
pub use fixed_angle::*;
pub use sheets::*;
pub use plugins::*;

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

fn linfit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let s = sxy / sxx;
    (s, my - s * mx)
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

/// Time step of a simulation (as the engines will use it).
fn sim_dt(sim: &Simulation) -> Result<f64> {
    sim.time_step(&sim.build_grid()?)
}

pub fn all_cases() -> Vec<(&'static str, fn(Runner) -> Result<CaseReport>)> {
    vec![
        ("plane_wave_vacuum", plane_wave_vacuum),
        ("fresnel_normal", fresnel_normal),
        ("pec_pmc_reflection", pec_pmc_reflection),
        ("dipole_radiation", dipole_radiation),
        ("pml_reflection", pml_reflection),
        ("absorber_reflection", absorber_reflection),
        ("skin_depth", skin_depth),
        ("magnetic_loss", magnetic_loss),
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
        ("chi3_raman", chi3_raman),
        ("offdiagonal_tensor", offdiagonal_tensor),
        ("lossy_offdiagonal_tensor", lossy_offdiagonal_tensor),
        ("oriented_dispersion", oriented_dispersion),
        ("gyrotropic_faraday", gyrotropic_faraday),
        ("director_field", director_field),
        ("gain_four_level", gain_four_level),
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
        ("fixed_angle_slab", fixed_angle_slab),
        ("fixed_angle_brewster", fixed_angle_brewster),
        ("fixed_angle_dispersive", fixed_angle_dispersive),
        ("fixed_angle_grating", fixed_angle_grating),
        ("fixed_angle_stability", fixed_angle_stability),
        ("fixed_angle_3d", fixed_angle_3d),
        ("sheet_normal_incidence", sheet_normal_incidence),
        ("sheet_oblique", sheet_oblique),
        ("sheet_plasmon", sheet_plasmon),
        ("sheet_ribbon", sheet_ribbon),
        ("plugin_media", plugin_media),
        ("gain_two_electron", gain_two_electron),
        ("hermitian_tensor", hermitian_tensor),
        ("dispersive_subpixel", dispersive_subpixel),
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
        assert!((j[0] + 0.048_383_776_468_198).abs() < 1e-9 && (y[0] - 0.498_070_359_615_232_2).abs() < 1e-9, "{} {}", j[0], y[0]);
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
