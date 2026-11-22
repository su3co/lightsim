//! Time reversal (cf. FDTDX tests/simulation/fdtd/test_time_reversal.py):
//! after the source has died out, stepping the leapfrog backwards must
//! reconstruct an earlier field state. Periodic, PEC, PMC and Bloch domains
//! and lossy media are exactly reversible; absorbing PML layers are replayed
//! from a recording of their fields made on the forward run.

use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::sources::{Source, TimeSignal};
use lightsim_core::spec::{Boundaries, Boundary, Simulation};
use lightsim_core::{CompileOptions, Medium, compile};

const F0: f64 = 2.99792458e14;

fn scene(b: Boundaries, lossy: bool) -> Simulation {
    let l = 0.8e-6;
    let mut sim = Simulation::new([0.0; 3], [l; 3], MeshSpec::Uniform { dx: [50e-9; 3] }, b, 1.0);
    sim.sources.push(Source::PointDipole {
        name: "d".into(),
        position: [0.31e-6, 0.42e-6, 0.37e-6],
        polarization: Component::Ez,
        // zero-mean pulse: no static dipole field is left behind
        signal: TimeSignal::Ricker { freq0: F0, offset: 3.0, amplitude: 1.0 },
        amplitude: 1e-12,
    });
    if lossy {
        // ~e^-0.9 amplitude decay over the reversed window (reversal amplifies f32
        // rounding by the inverse of the physical decay)
        sim.media.push(Medium::conductor(2.0, 400.0).with_name("lossy"));
        sim.background = "lossy".into();
    }
    sim
}

fn reversal_error(sim: &Simulation, n1: usize, n2: usize) -> f64 {
    let plan = compile(sim, &CompileOptions { nsteps: Some(n2 + 1), ..Default::default() }).unwrap();
    let mut be = CpuBackend::new(plan).unwrap();
    let r = lightsim_core::reverse::reversal(&mut be, n1, n2).unwrap();
    println!("  moved {:.3e}, error {:.3e}", r.moved, r.error);
    assert!(r.moved > 0.1, "the fields must have evolved ({:.3e})", r.moved);
    r.error
}

fn check(name: &str, sim: Simulation, tol: f64) {
    let e = reversal_error(&sim, 200, 600);
    println!("{name}: max |reconstructed - original| / peak = {e:.3e}");
    assert!(e < tol, "{name}: {e:.3e}");
}

#[test]
fn periodic_fields_reconstructed() {
    check("periodic", scene(Boundaries::all(Boundary::Periodic), false), 1e-4);
}

#[test]
fn pec_fields_reconstructed() {
    check("PEC", scene(Boundaries::all(Boundary::Pec), false), 1e-4);
}

#[test]
fn pmc_fields_reconstructed() {
    check("PMC", scene(Boundaries::all(Boundary::Pmc), false), 1e-4);
}

#[test]
fn bloch_fields_reconstructed() {
    let k = 0.3 * 2.0 * std::f64::consts::PI / 1e-6;
    let mut b = Boundaries::all(Boundary::Bloch { k });
    b.z = [Boundary::Bloch { k: 0.5 * k }, Boundary::Bloch { k: 0.5 * k }];
    check("Bloch", scene(b, false), 1e-4);
}

#[test]
fn lossy_medium_reconstructed() {
    check("lossy (periodic)", scene(Boundaries::all(Boundary::Periodic), true), 1e-3);
}

#[test]
fn pml_fields_reconstructed_with_recording() {
    check("PML with recording", scene(Boundaries::all_pml(), false), 1e-4);
}

#[test]
fn mixed_pml_periodic_reconstructed_with_recording() {
    let mut b = Boundaries::all(Boundary::Periodic);
    b.z = [Boundary::pml(), Boundary::pml()];
    check("PML (z) + periodic (x, y)", scene(b, false), 1e-4);
}

#[test]
fn mixed_pml_bloch_reconstructed_with_recording() {
    let k = 0.3 * 2.0 * std::f64::consts::PI / 1e-6;
    let mut b = Boundaries::all(Boundary::Bloch { k });
    b.z = [Boundary::pml(), Boundary::pml()];
    check("PML (z) + Bloch (x, y)", scene(b, false), 1e-4);
}
