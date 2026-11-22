//! Fixed-angle engine on the GPU: one GPU against the CPU engine, and two
//! GPUs (x-slabs) against one (skipped with fewer than two GPUs).

use lightsim_core::grid::Component;
use lightsim_core::run::RunOptions;
use lightsim_core::validation::{dft_rel_diff, fixed_angle_scene};

fn worst(a: &lightsim_core::Results, b: &lightsim_core::Results) -> f64 {
    let mut w = 0f64;
    for c in Component::ALL {
        w = w.max(dft_rel_diff(a, b, "f", c));
    }
    for m in ["R", "T"] {
        for (x, y) in a.flux[m].iter().zip(&b.flux[m]) {
            w = w.max((x - y).abs() / a.flux["T"][1].abs());
        }
    }
    w
}

#[test]
fn one_gpu_matches_the_cpu_engine() {
    for three_d in [false, true] {
        let sim = fixed_angle_scene(three_d);
        let cpu = lightsim_core::run_cpu(&sim).unwrap();
        let gpu = lightsim_cuda::fixed_angle::run(&sim, &[0], &RunOptions::default()).unwrap();
        let d = worst(&cpu, &gpu);
        println!("{}: CPU vs GPU max relative difference {d:.2e} ({})", if three_d { "3-D" } else { "2-D" }, gpu.stats.backend);
        assert!(d < 1e-4, "{d}");
    }
}

#[test]
fn two_gpus_match_one() {
    if lightsim_cuda::device_count().unwrap_or(0) < 2 {
        eprintln!("skipped: needs 2 GPUs");
        return;
    }
    for three_d in [false, true] {
        let sim = fixed_angle_scene(three_d);
        let one = lightsim_cuda::fixed_angle::run(&sim, &[0], &RunOptions::default()).unwrap();
        let two = lightsim_cuda::fixed_angle::run(&sim, &[0, 1], &RunOptions::default()).unwrap();
        let d = worst(&one, &two);
        println!("{}: 1 vs 2 GPUs max relative difference {d:.2e} ({})", if three_d { "3-D" } else { "2-D" }, two.stats.backend);
        assert!(two.stats.backend.contains(" + "), "the run was not split: {}", two.stats.backend);
        assert!(d < 1e-5, "{d}");
    }
}
