//! The 2-D step on the GPU (`step2d`, grouped DFT and source launches)
//! against the general 3-D kernels and the CPU backend, on the scenes of
//! `lightsim_core::validation::two_d_scenes`.

use lightsim_core::grid::Component;
use lightsim_core::run::RunOptions;
use lightsim_core::validation::{two_d_matches_one_cell, two_d_scenes};
use lightsim_core::{Results, Simulation};

fn gpus() -> usize {
    lightsim_cuda::device_count().unwrap_or(0)
}

/// Largest difference of the DFT fields of monitor `f`, relative to the
/// largest field of any component.
fn diff(a: &Results, b: &Results) -> f64 {
    let peak = Component::ALL
        .iter()
        .map(|c| {
            let f = b.dft["f"].field(*c).unwrap();
            f.re.iter().chain(f.im.iter()).fold(0f32, |m, x| m.max(x.abs())) as f64
        })
        .fold(0.0, f64::max);
    assert!(peak > 0.0);
    let mut worst = 0.0f64;
    for c in Component::ALL {
        let (fa, fb) = (a.dft["f"].field(c).unwrap(), b.dft["f"].field(c).unwrap());
        for (x, y) in fa.re.iter().chain(fa.im.iter()).zip(fb.re.iter().chain(fb.im.iter())) {
            worst = worst.max((x - y).abs() as f64 / peak);
        }
    }
    worst
}

#[test]
fn two_d_kernels_match_three_d_kernels() {
    if gpus() == 0 {
        eprintln!("skipped: needs a GPU");
        return;
    }
    let r = two_d_matches_one_cell(&|s: &Simulation| lightsim_cuda::run_gpu(s, 0)).unwrap();
    println!("{}", r.summary());
    assert!(r.passed(), "{}", r.summary());
}

#[test]
fn two_d_gpu_matches_cpu() {
    if gpus() == 0 {
        eprintln!("skipped: needs a GPU");
        return;
    }
    for (name, sim) in two_d_scenes() {
        let g = lightsim_cuda::run_gpu(&sim, 0).unwrap();
        let c = lightsim_core::run_cpu(&sim).unwrap();
        assert!(g.stats.backend.contains("2-D"), "{name}: {}", g.stats.backend);
        let d = diff(&g, &c);
        println!("{name}: GPU vs CPU {d:.3e}  [{}]", g.stats.backend);
        assert!(d < 1e-4, "{name}: max |GPU - CPU| / peak = {d:.3e}");
    }
}

#[test]
fn two_d_split_kernels_still_match() {
    // LIGHTSIM_FUSED=0 is process-wide, so the split path is exercised
    // through a backend built without the fused step
    if gpus() == 0 {
        eprintln!("skipped: needs a GPU");
        return;
    }
    for (name, sim) in two_d_scenes().into_iter().step_by(5) {
        let plan = lightsim_core::compile(&sim, &Default::default()).unwrap();
        let be = lightsim_cuda::CudaBackend::new_with(plan, 0, false).unwrap();
        let s = lightsim_core::run::run_backend(Box::new(be), &RunOptions::default()).unwrap();
        assert!(!s.stats.backend.contains("2-D"), "{name}: {}", s.stats.backend);
        let c = lightsim_core::run_cpu(&sim).unwrap();
        let d = diff(&s, &c);
        println!("{name}: split kernels vs CPU {d:.3e}");
        assert!(d < 1e-4, "{name}: {d:.3e}");
    }
}

#[test]
fn two_d_on_two_gpus() {
    if gpus() < 2 {
        eprintln!("skipped: needs 2 GPUs");
        return;
    }
    for (name, sim) in two_d_scenes() {
        let one = lightsim_cuda::run_gpu(&sim, 0).unwrap();
        let two = lightsim_cuda::multi::run_multi_gpu(&sim, &[0, 1], &RunOptions::default()).unwrap();
        let d = diff(&two, &one);
        println!("{name}: 2 GPUs vs 1 {d:.3e}  [{}]", two.stats.backend);
        assert!(two.stats.backend.contains("2-D"), "{name}: {}", two.stats.backend);
        assert!(d < 1e-4, "{name}: max |2 GPUs - 1 GPU| / peak = {d:.3e}");
    }
}

#[test]
fn two_d_magnetic_medium_falls_back_to_the_general_kernels() {
    // a magnetic medium is outside the fused step, hence outside the 2-D
    // kernel: the same plan runs on the split 3-D kernels, with the same result
    if gpus() == 0 {
        eprintln!("skipped: needs a GPU");
        return;
    }
    for (name, mut sim) in two_d_scenes().into_iter().filter(|(n, _)| n.ends_with("CPML, both sets")) {
        let mut mag = lightsim_core::Medium::from_index(1.3).with_name("mag");
        mag.mu = [1.4, 1.7, 2.0];
        sim.media.push(mag);
        let um = 1e-6;
        let w = sim.invariant().unwrap();
        let lift = |a: f64, b: f64, c: f64| match w {
            0 => [c, a, b],
            1 => [a, c, b],
            _ => [a, b, c],
        };
        sim.structures.push(lightsim_core::geometry::Structure { name: "mag".into(), shape: lightsim_core::geometry::Shape::Box { min: lift(-0.1 * um, -0.6 * um, -1.0), max: lift(0.15 * um, -0.2 * um, 1.0) }, medium: "mag".into() });
        let g = lightsim_cuda::run_gpu(&sim, 0).unwrap();
        let c = lightsim_core::run_cpu(&sim).unwrap();
        assert!(!g.stats.backend.contains("2-D"), "{name}: {}", g.stats.backend);
        assert!(c.stats.backend.contains("2-D"), "{name}: {}", c.stats.backend);
        let d = diff(&g, &c);
        println!("{name} + magnetic block: GPU (general kernels) vs CPU (2-D) {d:.3e}");
        assert!(d < 1e-4, "{name}: {d:.3e}");
    }
}
