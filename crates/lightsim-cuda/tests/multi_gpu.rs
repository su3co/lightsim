//! Multi-GPU checks (skipped when fewer than two GPUs are present).

use lightsim_core::grid::Component;
use lightsim_core::run::RunOptions;
use lightsim_core::validation::{bloch_xy_box, chi2_second_harmonic, dft_rel_diff};

fn gpus() -> usize {
    lightsim_cuda::device_count().unwrap_or(0)
}

#[test]
fn bloch_xy_two_gpus_match_one() {
    if gpus() < 2 {
        eprintln!("skipped: needs 2 GPUs");
        return;
    }
    let sim = bloch_xy_box();
    let a = lightsim_cuda::run_gpu(&sim, 0).unwrap();
    let b = lightsim_cuda::multi::run_multi_gpu(&sim, &[0, 1], &RunOptions::default()).unwrap();
    let c = lightsim_core::run_cpu(&sim).unwrap();
    for comp in [Component::Ez, Component::Hx] {
        let d = dft_rel_diff(&a, &b, "f", comp);
        let dc = dft_rel_diff(&c, &b, "f", comp);
        println!("{comp:?}: 1 vs 2 GPUs {d:.3e}, CPU vs 2 GPUs {dc:.3e}");
        assert!(d < 1e-5 && dc < 1e-4, "{comp:?}: {d:.3e} {dc:.3e}");
    }
}

#[test]
fn chi2_two_gpus() {
    if gpus() < 2 {
        eprintln!("skipped: needs 2 GPUs");
        return;
    }
    let run = |s: &lightsim_core::Simulation| lightsim_cuda::multi::run_multi_gpu(s, &[0, 1], &RunOptions::default());
    let r = chi2_second_harmonic(&run).unwrap();
    println!("{}", r.summary());
    assert!(r.passed());
}
