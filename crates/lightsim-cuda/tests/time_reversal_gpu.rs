//! Time reversal on the GPU (split kernels run backwards), mirroring the CPU
//! tests in lightsim-core/tests/time_reversal.rs and FDTDX's
//! test_time_reversal: periodic, PEC, PMC, Bloch, lossy, PML with recording,
//! mixed PML + periodic / Bloch.

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
        signal: TimeSignal::Ricker { freq0: F0, offset: 3.0, amplitude: 1.0 },
        amplitude: 1e-12,
    });
    if lossy {
        sim.media.push(Medium::conductor(2.0, 400.0).with_name("lossy"));
        sim.background = "lossy".into();
    }
    sim
}

fn check(name: &str, sim: Simulation, tol: f64) {
    if lightsim_cuda::device_count().unwrap_or(0) == 0 {
        eprintln!("skipped: no GPU");
        return;
    }
    let plan = compile(&sim, &CompileOptions { nsteps: Some(601), ..Default::default() }).unwrap();
    let mut be = lightsim_cuda::CudaBackend::new_with(plan, 0, false).unwrap();
    let r = lightsim_core::reverse::reversal(&mut be, 200, 600).unwrap();
    println!("{name}: moved {:.3e}, max |reconstructed - original| / peak = {:.3e}", r.moved, r.error);
    assert!(r.moved > 0.1 && r.error < tol, "{name}: {:?}", r);
}

#[test]
fn gpu_periodic() {
    check("periodic", scene(Boundaries::all(Boundary::Periodic), false), 1e-4);
}

#[test]
fn gpu_pec() {
    check("PEC", scene(Boundaries::all(Boundary::Pec), false), 1e-4);
}

#[test]
fn gpu_pmc() {
    check("PMC", scene(Boundaries::all(Boundary::Pmc), false), 1e-4);
}

#[test]
fn gpu_bloch() {
    let k = 0.3 * 2.0 * std::f64::consts::PI / 1e-6;
    let mut b = Boundaries::all(Boundary::Bloch { k });
    b.z = [Boundary::Bloch { k: 0.5 * k }, Boundary::Bloch { k: 0.5 * k }];
    check("Bloch", scene(b, false), 1e-4);
}

#[test]
fn gpu_lossy() {
    check("lossy", scene(Boundaries::all(Boundary::Periodic), true), 1e-3);
}

#[test]
fn gpu_pml_recorded() {
    check("PML with recording", scene(Boundaries::all_pml(), false), 1e-4);
}

#[test]
fn gpu_pml_periodic_recorded() {
    let mut b = Boundaries::all(Boundary::Periodic);
    b.z = [Boundary::pml(), Boundary::pml()];
    check("PML (z) + periodic", scene(b, false), 1e-4);
}

#[test]
fn gpu_pml_bloch_recorded() {
    let k = 0.3 * 2.0 * std::f64::consts::PI / 1e-6;
    let mut b = Boundaries::all(Boundary::Bloch { k });
    b.z = [Boundary::pml(), Boundary::pml()];
    check("PML (z) + Bloch", scene(b, false), 1e-4);
}

/// Throughput of time reversal on the GPU (run with --ignored).
#[test]
#[ignore]
fn gpu_reverse_throughput() {
    if lightsim_cuda::device_count().unwrap_or(0) == 0 {
        return;
    }
    let n = 256usize;
    let dx = 25e-9;
    let mut sim = Simulation::new([0.0; 3], [n as f64 * dx; 3], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all(Boundary::Periodic), 1.0);
    sim.sources.push(Source::PointDipole {
        name: "d".into(),
        position: [3.1e-6, 3.3e-6, 2.9e-6],
        polarization: Component::Ez,
        signal: TimeSignal::Ricker { freq0: 2e14, offset: 3.0, amplitude: 1.0 },
        amplitude: 1e-12,
    });
    let plan = compile(&sim, &CompileOptions { nsteps: Some(1200), ..Default::default() }).unwrap();
    let cells = plan.local_cells() as f64;
    let mut be = lightsim_cuda::CudaBackend::new_with(plan, 0, false).unwrap();
    // lead-in until the Ricker pulse (centred near 15 fs) has ended
    let r = lightsim_core::reverse::reversal(&mut be, 900, 1100).unwrap();
    let steps = 200.0;
    println!(
        "256^3 periodic, 200 steps: forward {:.2} Gcells/s, reverse {:.2} Gcells/s, reconstruction error {:.2e}",
        cells * steps / r.forward_s / 1e9,
        cells * steps / r.reverse_s / 1e9,
        r.error
    );
}
