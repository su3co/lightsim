//! `lightsim` command-line interface.
//!
//! ```text
//! lightsim info
//! lightsim validate [--device cpu|cuda[:i[,j..]]] [--case NAME]
//! lightsim run SPEC.json [--device ...] [--out RESULTS.json] [--progress N]
//!              [--dimensions 2|3] [--invariant-axis x|y|z] [--polarization te|tm|both] [--kernels 3d]
//! lightsim bench [--device ...] [--n NX,NY,NZ | --n NU,NV] [--steps N] [--json OUT]
//!                [--invariant-axis x|y|z] [--polarization te|tm|both] [--kernels 3d]
//! lightsim rank SPEC.json --hosts h0,h1,.. --rank R [--port P] [--gpus i,j] [--out RESULTS.json]
//! lightsim launch SPEC.json --hosts h0,h1,.. [--addrs a0,a1,..] [--port P] [--out RESULTS.json]
//! ```
//!
//! `run`: the 2-D options override the spec's `dimensions`,
//! `invariant_axis` and `polarization` (a 2-D simulation has fields that do
//! not depend on the invariant axis; `te` = E in the plane, `tm` = E along
//! the invariant axis). `--kernels 3d` runs a 2-D problem through the
//! general 3-D kernels (cross-check, slower).
//!
//! `bench --n NU,NV` (two numbers) times a 2-D grid of NU x NV cells.


#[cfg(all(feature = "cuda", feature = "f64"))]
compile_error!("the f64 (double-precision) build is CPU only: features \"cuda\" and \"f64\" cannot be combined");
use anyhow::{Context, Result, anyhow, bail};
use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::plan::{CompileOptions, compile_with_grid, decompose_x};
use lightsim_core::run::{Chain, RunOptions, run_chain};
use lightsim_core::validation;
use lightsim_core::{Backend, Results, Simulation};
use std::collections::HashMap;

struct Args {
    pos: Vec<String>,
    kv: HashMap<String, String>,
}

fn parse_args() -> Args {
    let mut pos = vec![];
    let mut kv = HashMap::new();
    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        if let Some(k) = a.strip_prefix("--") {
            let v = match it.peek() {
                Some(n) if !n.starts_with("--") => it.next().unwrap(),
                _ => "true".into(),
            };
            kv.insert(k.to_string(), v);
        } else {
            pos.push(a);
        }
    }
    Args { pos, kv }
}

#[derive(Clone, Debug)]
enum Device {
    Cpu,
    Cuda(Vec<usize>),
}

fn parse_device(s: &str) -> Result<Device> {
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if let Some(rest) = s.strip_prefix("cuda") {
        let rest = rest.trim_start_matches(':');
        if rest.is_empty() {
            return Ok(Device::Cuda(vec![0]));
        }
        if rest == "all" {
            #[cfg(feature = "cuda")]
            {
                let n = lightsim_cuda::device_count()?;
                return Ok(Device::Cuda((0..n).collect()));
            }
        }
        let v: Result<Vec<usize>, _> = rest.split(',').map(|x| x.parse::<usize>()).collect();
        return Ok(Device::Cuda(v.map_err(|_| anyhow!("bad device list '{s}'"))?));
    }
    bail!("unknown device '{s}' (use cpu, cuda, cuda:0,1, cuda:all)")
}

fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if lightsim_cuda::device_count().unwrap_or(0) > 0 {
            return Device::Cuda(vec![0]);
        }
    }
    Device::Cpu
}

fn build_backends(sim: &Simulation, dev: &Device) -> Result<Vec<Box<dyn Backend>>> {
    if sim.fixed_angle {
        return match dev {
            Device::Cpu => Ok(vec![lightsim_core::engine::fixed_angle::cpu_backend(sim)?]),
            #[cfg(feature = "cuda")]
            Device::Cuda(gpus) => Ok(vec![lightsim_cuda::fixed_angle::backend(sim, gpus)?]),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => bail!("{}", no_cuda()),
        };
    }
    match dev {
        Device::Cpu => {
            let plan = lightsim_core::compile(sim, &CompileOptions::default())?;
            Ok(vec![Box::new(CpuBackend::new(plan)?)])
        }
        Device::Cuda(gpus) => {
            #[cfg(feature = "cuda")]
            {
                let slabs = lightsim_cuda::multi::build_slabs(sim, gpus, None)?;
                Ok(slabs.into_iter().map(|b| Box::new(b) as Box<dyn Backend>).collect())
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = gpus;
                bail!("{}", no_cuda())
            }
        }
    }
}

fn run_sim(sim: &Simulation, dev: &Device, opts: &RunOptions) -> Result<Results> {
    // (several GPUs: the fixed-angle engine's own decomposed loop)
    #[cfg(feature = "cuda")]
    if let (true, Device::Cuda(gpus)) = (sim.fixed_angle, dev) {
        return lightsim_cuda::fixed_angle::run(sim, gpus, opts);
    }
    let slabs = build_backends(sim, dev)?;
    let n = slabs.len();
    let mut chain = Chain::new(slabs);
    chain.local_wrap = n > 1 && sim.boundaries.x[0].is_periodic();
    run_chain(&mut chain, opts)
}

/// The error for a GPU request in a build without the CUDA backend.
#[cfg(not(feature = "cuda"))]
fn no_cuda() -> &'static str {
    if lightsim_core::REAL_IS_F64 {
        "this lightsim binary is a double-precision (f64) build, which runs on the CPU only: use --device cpu, or a default (f32) build with --features cuda for GPUs"
    } else {
        "this lightsim binary was built without CUDA support (build with --features cuda)"
    }
}

fn cmd_info() -> Result<()> {
    println!("lightsim {} ({} threads)", env!("CARGO_PKG_VERSION"), rayon_threads());
    #[cfg(feature = "cuda")]
    {
        let n = lightsim_cuda::device_count().unwrap_or(0);
        println!("CUDA devices: {n}");
        for i in 0..n {
            match cuda_name(i) {
                Ok(s) => println!("  cuda:{i}  {s}"),
                Err(e) => println!("  cuda:{i}  <error {e}>"),
            }
        }
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    println!("  P2P {i}->{j}: {}", lightsim_cuda::multi::peer_capable(i, j).unwrap_or(false));
                }
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("(built without CUDA)");
    println!("precision: {}", if lightsim_core::REAL_IS_F64 { "f64 (double, CPU only)" } else { "f32 (single)" });
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_name(i: usize) -> Result<String> {
    let ctx = cuda_core::CudaContext::new(i)?;
    Ok(ctx.name()?)
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn cmd_validate(a: &Args) -> Result<()> {
    let dev = a.kv.get("device").map(|s| parse_device(s)).transpose()?.unwrap_or_else(default_device);
    let only = a.kv.get("case").cloned();
    let runner = |sim: &Simulation| run_sim(sim, &dev, &RunOptions::default());
    let mut failed = 0;
    let mut total = 0;
    for (name, case) in validation::all_cases() {
        if let Some(o) = &only {
            if !name.contains(o.as_str()) {
                continue;
            }
        }
        total += 1;
        let t0 = std::time::Instant::now();
        match case(&runner) {
            Ok(r) => {
                print!("{}", r.summary());
                println!("    ({:.2} s, {:.0} Mcells/s)", t0.elapsed().as_secs_f64(), r.mcells_per_s);
                if !r.passed() {
                    failed += 1;
                }
            }
            Err(e) => {
                println!("{name} [ERROR] {e:#}");
                failed += 1;
            }
        }
    }
    println!("\n{} / {} cases passed on {:?}", total - failed, total, dev);
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Apply `--dimensions`, `--invariant-axis`, `--polarization` and
/// `--kernels 3d` to a simulation.
fn two_d_options(a: &Args, sim: &mut Simulation) -> Result<()> {
    use lightsim_core::spec::{Polarization, parse_axis};
    if let Some(d) = a.kv.get("dimensions") {
        sim.dimensions = match d.as_str() {
            "2" => 2,
            "3" => 3,
            _ => bail!("--dimensions must be 2 or 3"),
        };
    }
    if let Some(ax) = a.kv.get("invariant-axis") {
        sim.invariant_axis = parse_axis(ax).ok_or_else(|| anyhow!("--invariant-axis must be x, y or z"))?;
        sim.dimensions = 2;
    }
    if let Some(p) = a.kv.get("polarization") {
        sim.polarization = Polarization::parse(p).ok_or_else(|| anyhow!("--polarization must be te, tm or both"))?;
        if sim.polarization != Polarization::Both {
            sim.dimensions = 2;
        }
    }
    match a.kv.get("kernels").map(|s| s.as_str()) {
        None | Some("auto") => {}
        Some("3d") => sim.force_3d_kernels = true,
        Some(k) => bail!("--kernels must be auto or 3d, got '{k}'"),
    }
    Ok(())
}

fn cmd_run(a: &Args) -> Result<()> {
    let path = a.pos.get(1).ok_or_else(|| anyhow!("usage: lightsim run SPEC.json"))?;
    let mut sim = Simulation::from_json(&std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?)?;
    two_d_options(a, &mut sim)?;
    // without --device: the first GPU, or the CPU for a small 2-D grid
    let dev = a.kv.get("device").map(|s| parse_device(s)).transpose()?.unwrap_or_else(|| if sim.auto_prefers_cpu() { Device::Cpu } else { default_device() });
    let progress = a.kv.get("progress").map(|s| s.parse().unwrap_or(0)).unwrap_or(0);
    let res = run_sim(&sim, &dev, &RunOptions { progress_every: progress })?;
    eprintln!(
        "{}: {} cells x {} steps in {:.3} s = {:.1} Mcells/s",
        res.stats.backend, res.stats.cells, res.stats.steps, res.stats.seconds, res.stats.mcells_per_s
    );
    // a run that reached run_time with fields above the shutoff level
    if let (Some(th), false) = (sim.shutoff, res.stats.stopped_early) {
        let peak = res.stats.energy.iter().cloned().fold(0.0, f64::max);
        let pulsed = sim.sources.iter().all(|s| s.signal().end_time().is_finite());
        if let (Some(last), true) = (res.stats.energy.last(), peak > 0.0 && pulsed) {
            if last / peak > th {
                eprintln!(
                    "warning: the run reached run_time before the fields had decayed: their energy is {:.1e} of its peak, above shutoff = {:.1e}",
                    last / peak,
                    th
                );
            }
        }
    }
    if let Some(out) = a.kv.get("out") {
        std::fs::write(out, res.to_json())?;
    }
    Ok(())
}

/// Synthetic throughput benchmark: vacuum box with a dipole, CPML on all faces.
/// Benchmark scenes (all with a centred Ez dipole, 25 nm cells, 12-cell CPML):
/// `vacuum`; `lorentz` (whole domain a Lorentz medium); `bloch` (Bloch k != 0
/// along x and y, PML in z); `tensor` (whole domain a 45-degree rotated
/// uniaxial crystal, off-diagonal eps_xy); `--dispersive` adds a gold sphere.
///
/// `two_d = (invariant axis, polarization)`: a 2-D grid of `n[0] x n[1]`
/// cells (CPML included) in the plane normal to that axis, driven by an
/// electric dipole per field set in use (in the plane for `te`, along the
/// invariant axis for `tm`); scenes `vacuum`, `lorentz` and `bloch` (Bloch
/// along the second in-plane axis).
fn bench_sim(n: [usize; 3], scene: &str, dispersive: bool, two_d: Option<(usize, lightsim_core::spec::Polarization)>) -> Result<Simulation> {
    if let Some((w, pol)) = two_d {
        return bench_sim_2d([n[0], n[1]], scene, dispersive, w, pol);
    }
    use lightsim_core::grid::Component;
    use lightsim_core::sources::{Source, TimeSignal};
    let dx = 25e-9;
    let pml = 12usize;
    let f0 = 2e14;
    let bloch = scene == "bloch";
    let size: Vec<f64> = n
        .iter()
        .enumerate()
        .map(|(a, &c)| if bloch && a < 2 { c as f64 * dx } else { (c.saturating_sub(2 * pml)).max(4) as f64 * dx })
        .collect();
    let mut b = lightsim_core::Boundaries::all_pml();
    if bloch {
        let k0 = 2.0 * std::f64::consts::PI * f0 / lightsim_core::constants::C0;
        b.x = [lightsim_core::Boundary::Bloch { k: 0.3 * k0 }, lightsim_core::Boundary::Bloch { k: 0.3 * k0 }];
        b.y = [lightsim_core::Boundary::Bloch { k: 0.2 * k0 }, lightsim_core::Boundary::Bloch { k: 0.2 * k0 }];
    }
    let mut sim = Simulation::new([0.0; 3], [size[0], size[1], size[2]], lightsim_core::MeshSpec::Uniform { dx: [dx; 3] }, b, 1.0);
    match scene {
        "vacuum" | "bloch" => {}
        "lorentz" => {
            let w = 2.0 * std::f64::consts::PI * f0;
            sim.media.push(
                lightsim_core::Medium::dielectric(2.25)
                    .with_name("lorentz")
                    .with_poles(vec![lightsim_core::Pole::Lorentz { delta_eps: 1.0, omega0: 2.0 * w, gamma: 1e13 }]),
            );
            sim.background = "lorentz".into();
        }
        "tensor" => {
            let c = std::f64::consts::FRAC_1_SQRT_2;
            let r = [[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]];
            sim.media.push(lightsim_core::Medium::anisotropic([3.24, 2.25, 2.25]).with_name("xtal").rotated(r));
            sim.background = "xtal".into();
        }
        _ => bail!("unknown scene '{scene}' (vacuum, lorentz, bloch, tensor)"),
    }
    sim.sources.push(Source::PointDipole {
        name: "d".into(),
        position: [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
        polarization: Component::Ez,
        signal: TimeSignal::gaussian(f0, 5e13),
        amplitude: 1e-12,
    });
    if dispersive {
        sim.structures.push(lightsim_core::geometry::Structure {
            name: "au".into(),
            shape: lightsim_core::geometry::Shape::Sphere { center: [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0], radius: size[0] / 6.0 },
            medium: "gold".into(),
        });
    }
    Ok(sim)
}

fn bench_sim_2d(n: [usize; 2], scene: &str, dispersive: bool, w: usize, pol: lightsim_core::spec::Polarization) -> Result<Simulation> {
    use lightsim_core::grid::Component;
    use lightsim_core::sources::{Source, TimeSignal};
    let dx = 25e-9;
    let pml = 12usize;
    let f0 = 2e14;
    let (u, v, _) = lightsim_core::spec::plane_of(w);
    let bloch = scene == "bloch";
    let mut size = [dx; 3];
    size[u] = n[0].saturating_sub(2 * pml).max(4) as f64 * dx;
    size[v] = if bloch { n[1] as f64 * dx } else { n[1].saturating_sub(2 * pml).max(4) as f64 * dx };
    let mut b = lightsim_core::Boundaries::all_pml();
    if bloch {
        let k0 = 2.0 * std::f64::consts::PI * f0 / lightsim_core::constants::C0;
        *b.axis_mut(v) = [lightsim_core::Boundary::Bloch { k: 0.2 * k0 }, lightsim_core::Boundary::Bloch { k: 0.2 * k0 }];
    }
    let mut sim = Simulation::new([0.0; 3], size, lightsim_core::MeshSpec::Uniform { dx: [dx; 3] }, b, 1.0).two_d(w, pol);
    match scene {
        "vacuum" | "bloch" => {}
        "lorentz" => {
            let om = 2.0 * std::f64::consts::PI * f0;
            sim.media.push(
                lightsim_core::Medium::dielectric(2.25)
                    .with_name("lorentz")
                    .with_poles(vec![lightsim_core::Pole::Lorentz { delta_eps: 1.0, omega0: 2.0 * om, gamma: 1e13 }]),
            );
            sim.background = "lorentz".into();
        }
        _ => bail!("unknown 2-D scene '{scene}' (vacuum, lorentz, bloch)"),
    }
    let c = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0];
    let sets = pol.sets();
    let mut dip = |comp: Component| sim.sources.push(Source::PointDipole { name: comp.name().into(), position: c, polarization: comp, signal: TimeSignal::gaussian(f0, 5e13), amplitude: 1e-12 });
    if sets[0] {
        dip(Component::e(u));
    }
    if sets[1] {
        dip(Component::e(w));
    }
    if dispersive {
        sim.structures.push(lightsim_core::geometry::Structure {
            name: "au".into(),
            shape: lightsim_core::geometry::Shape::Cylinder { center: c, radius: size[u] / 6.0, length: 1.0, axis: w },
            medium: "gold".into(),
        });
    }
    Ok(sim)
}

fn cmd_bench(a: &Args) -> Result<()> {
    if a.kv.contains_key("membw") {
        #[cfg(feature = "cuda")]
        {
            let g: usize = a.kv.get("gpu").map(|s| s.parse().unwrap()).unwrap_or(0);
            println!("cuda:{g} copy bandwidth: {:.1} GB/s", lightsim_cuda::membw(g, 2048)?);
        }
        return Ok(());
    }
    let dev = a.kv.get("device").map(|s| parse_device(s)).transpose()?.unwrap_or_else(default_device);
    let n: Vec<usize> = a.kv.get("n").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![256, 256, 256]);
    let steps: usize = a.kv.get("steps").map(|s| s.parse().unwrap()).unwrap_or(200);
    let disp = a.kv.contains_key("dispersive");
    let scene = a.kv.get("scene").map(|s| s.as_str()).unwrap_or("vacuum");
    // two numbers: a 2-D grid
    let two_d = if n.len() == 2 {
        let w = match a.kv.get("invariant-axis") {
            Some(ax) => lightsim_core::spec::parse_axis(ax).ok_or_else(|| anyhow!("--invariant-axis must be x, y or z"))?,
            None => 2,
        };
        let pol = match a.kv.get("polarization") {
            Some(p) => lightsim_core::spec::Polarization::parse(p).ok_or_else(|| anyhow!("--polarization must be te, tm or both"))?,
            None => lightsim_core::spec::Polarization::Both,
        };
        Some((w, pol))
    } else if n.len() == 3 {
        None
    } else {
        bail!("--n takes NX,NY,NZ (3-D) or NU,NV (2-D)");
    };
    let mut sim = bench_sim([n[0], n[1], *n.get(2).unwrap_or(&1)], scene, disp, two_d)?;
    if a.kv.get("kernels").map(|s| s.as_str()) == Some("3d") {
        sim.force_3d_kernels = true;
    }
    let grid = sim.build_grid()?;
    let t0 = std::time::Instant::now();
    let slabs: Vec<Box<dyn Backend>> = match &dev {
        Device::Cpu => {
            let opts = CompileOptions { nsteps: Some(steps), ..Default::default() };
            vec![Box::new(CpuBackend::new(compile_with_grid(&sim, grid.clone(), &opts)?)?)]
        }
        Device::Cuda(gpus) => {
            #[cfg(feature = "cuda")]
            {
                let parts = decompose_x(grid.n()[0], &vec![1.0; gpus.len()]);
                let mut v: Vec<Box<dyn Backend>> = vec![];
                for (r, (&g, &(x0, nxl))) in gpus.iter().zip(parts.iter()).enumerate() {
                    let opts = CompileOptions { slab: Some((x0, nxl)), nslabs: gpus.len(), slab_index: r, nsteps: Some(steps), ..Default::default() };
                    v.push(Box::new(lightsim_cuda::CudaBackend::new(compile_with_grid(&sim, grid.clone(), &opts)?, g)?));
                }
                v
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = gpus;
                bail!("{}", no_cuda())
            }
        }
    };
    let setup = t0.elapsed().as_secs_f64();
    let mut chain = Chain::new(slabs);
    let res = run_chain(&mut chain, &RunOptions::default())?;
    // (the multi-slab bench of a grid that is one plane thick along x runs on the first device)
    let gb = res.stats.cells as f64 * 4.0 * 13.0 * 2.0 * res.stats.steps as f64 / res.stats.seconds / 1e9;
    println!(
        "bench {:?} grid {:?} = {:.1} Mcells, {} steps: {:.3} s ({:.2} s setup) -> {:.1} Mcells/s (~{:.0} GB/s effective) [{}]",
        dev,
        grid.n(),
        res.stats.cells as f64 / 1e6,
        res.stats.steps,
        res.stats.seconds,
        setup,
        res.stats.mcells_per_s,
        gb,
        res.stats.backend
    );
    if let Some(out) = a.kv.get("json") {
        let j = serde_json::json!({
            "device": format!("{:?}", dev), "grid": grid.n(), "cells": res.stats.cells, "steps": res.stats.steps,
            "seconds": res.stats.seconds, "setup_seconds": setup, "mcells_per_s": res.stats.mcells_per_s, "backend": res.stats.backend,
        });
        std::fs::write(out, serde_json::to_string_pretty(&j)?)?;
    }
    Ok(())
}

/// One rank of a multi-node run: compiles only its own slab(s), exchanges
/// halos with neighbouring ranks over TCP, and sends its partial results to
/// rank 0, which assembles and writes them.
fn cmd_rank(a: &Args) -> Result<()> {
    use lightsim_core::net;
    let path = a.pos.get(1).ok_or_else(|| anyhow!("usage: lightsim rank SPEC.json --hosts .. --rank R"))?;
    let sim = Simulation::from_json(&std::fs::read_to_string(path)?)?;
    let hosts: Vec<String> = a.kv.get("hosts").ok_or_else(|| anyhow!("--hosts required"))?.split(',').map(String::from).collect();
    let rank: usize = a.kv.get("rank").ok_or_else(|| anyhow!("--rank required"))?.parse()?;
    let port: u16 = a.kv.get("port").map(|s| s.parse().unwrap()).unwrap_or(47100);
    let weights: Vec<f64> = match a.kv.get("weights") {
        Some(s) => s.split(',').map(|x| x.parse().unwrap()).collect(),
        None => vec![1.0; hosts.len()],
    };
    let gpus: Vec<usize> = match a.kv.get("gpus") {
        Some(s) => s.split(',').map(|x| x.parse().unwrap()).collect(),
        None => vec![0],
    };
    let steps: Option<usize> = a.kv.get("steps").map(|s| s.parse().unwrap());
    let nr = hosts.len();
    let ring = sim.boundaries.x[0].is_periodic() && nr > 1;
    let grid = sim.build_grid()?;
    let parts = decompose_x(grid.n()[0], &weights);
    let (x0, nxl) = parts[rank];
    // split this rank's slab across its local GPUs
    let sub = decompose_x(nxl, &vec![1.0; gpus.len()]);
    let total_slabs = nr * gpus.len();
    eprintln!("[rank {rank}] slab x=[{x0},{}) over {} GPU(s) on {}", x0 + nxl, gpus.len(), hosts[rank]);
    let mut slabs: Vec<Box<dyn Backend>> = vec![];
    for (li, (&g, &(sx0, snx))) in gpus.iter().zip(sub.iter()).enumerate() {
        let opts = CompileOptions {
            slab: Some((x0 + sx0, snx)),
            nslabs: total_slabs,
            slab_index: rank * gpus.len() + li,
            nsteps: steps,
            ..Default::default()
        };
        let plan = compile_with_grid(&sim, grid.clone(), &opts)?;
        if a.kv.contains_key("cpu") {
            slabs.push(Box::new(CpuBackend::new(plan)?));
        } else {
            #[cfg(feature = "cuda")]
            slabs.push(Box::new(lightsim_cuda::CudaBackend::new(plan, g)?));
            #[cfg(not(feature = "cuda"))]
            {
                let _ = g;
                bail!("{}; pass --cpu", no_cuda());
            }
        }
    }
    let chain: Option<Vec<String>> = a.kv.get("chain-addrs").map(|s| s.split(',').map(String::from).collect());
    let mut links = net::connect_with(rank, &hosts, chain.as_deref(), port, ring)?;
    eprintln!("[rank {rank}] links up (lo={}, hi={})", links.lo.is_some(), links.hi.is_some());
    let mut chain = Chain::new(slabs);
    chain.lo_is_wrap = links.lo_is_wrap;
    chain.hi_is_wrap = links.hi_is_wrap;
    chain.lo_link = links.lo.take().map(|l| Box::new(l) as Box<dyn lightsim_core::run::Link>);
    chain.hi_link = links.hi.take().map(|l| Box::new(l) as Box<dyn lightsim_core::run::Link>);
    let (raws, stats) = chain.run(&RunOptions { progress_every: if rank == 0 { a.kv.get("progress").map(|s| s.parse().unwrap_or(0)).unwrap_or(0) } else { 0 } })?;
    let plans: Vec<_> = chain.slabs.iter().map(|s| s.plan().clone()).collect();
    let prefs: Vec<&lightsim_core::Plan> = plans.iter().collect();
    let mut partial = lightsim_core::results::assemble(&prefs, &raws);
    partial.stats = stats;
    if rank == 0 {
        // receive and merge partial results from all ranks
        for s in links.control.iter_mut() {
            let bytes = net::recv_bytes(s)?;
            let other: Results = serde_json::from_slice(&bytes)?;
            merge_results(&mut partial, &other);
        }
        let mut res = partial;
        let full_plan = compile_with_grid(&sim, grid.clone(), &CompileOptions { nsteps: steps, ..Default::default() })?;
        lightsim_core::results::finalize(&mut res, &full_plan);
        res.stats.backend = format!("{} ranks x {} GPU(s)", nr, gpus.len());
        res.stats.cells = grid.num_cells();
        res.stats.mcells_per_s = res.stats.cells as f64 * res.stats.steps as f64 / res.stats.seconds / 1e6;
        eprintln!(
            "[rank 0] {} cells x {} steps in {:.3} s = {:.1} Mcells/s ({})",
            res.stats.cells, res.stats.steps, res.stats.seconds, res.stats.mcells_per_s, res.stats.backend
        );
        if let Some(out) = a.kv.get("out") {
            std::fs::write(out, res.to_json())?;
        }
        if let Some(j) = a.kv.get("json") {
            let v = serde_json::json!({"cells": res.stats.cells, "steps": res.stats.steps, "seconds": res.stats.seconds,
                "mcells_per_s": res.stats.mcells_per_s, "backend": res.stats.backend, "grid": grid.n()});
            std::fs::write(j, serde_json::to_string_pretty(&v)?)?;
        }
    } else {
        let s = links.control.get_mut(0).ok_or_else(|| anyhow!("no control link"))?;
        net::send_bytes(s, &serde_json::to_vec(&partial)?)?;
    }
    Ok(())
}

fn merge_results(dst: &mut Results, src: &Results) {
    for (name, d) in src.dft.iter() {
        if let Some(t) = dst.dft.get_mut(name) {
            for (a, b) in t.fields.iter_mut().zip(d.fields.iter()) {
                for (x, y) in a.re.iter_mut().zip(b.re.iter()) {
                    *x += *y;
                }
                for (x, y) in a.im.iter_mut().zip(b.im.iter()) {
                    *x += *y;
                }
            }
        }
    }
    for (name, d) in src.time.iter() {
        if let Some(t) = dst.time.get_mut(name) {
            for (a, b) in t.fields.iter_mut().zip(d.fields.iter()) {
                for (x, y) in a.data.iter_mut().zip(b.data.iter()) {
                    *x += *y;
                }
            }
        }
    }
    for (name, d) in src.energy.iter() {
        if let Some(t) = dst.energy.get_mut(name) {
            for (x, y) in t.energy.iter_mut().zip(d.energy.iter()) {
                *x += *y;
            }
        }
    }
    dst.stats.seconds = dst.stats.seconds.max(src.stats.seconds);
}

/// Start one rank per host over ssh (the binary and spec must exist at the
/// same path on every host, e.g. after `scripts/sync.sh`).
fn cmd_launch(a: &Args) -> Result<()> {
    let spec = a.pos.get(1).ok_or_else(|| anyhow!("usage: lightsim launch SPEC.json --hosts ..."))?;
    let hosts: Vec<String> = a.kv.get("hosts").ok_or_else(|| anyhow!("--hosts required"))?.split(',').map(String::from).collect();
    let addrs: Vec<String> = match a.kv.get("addrs") {
        Some(s) => s.split(',').map(String::from).collect(),
        None => hosts.clone(),
    };
    let bin = a.kv.get("bin").cloned().unwrap_or_else(|| std::env::current_exe().unwrap().display().to_string());
    let extra: Vec<String> = ["port", "gpus", "steps", "weights", "out", "json", "progress", "chain-addrs", "cpu"]
        .iter()
        .filter_map(|k| a.kv.get(*k).map(|v| format!("--{k} {v}")))
        .collect();
    let mut children = vec![];
    for (r, h) in hosts.iter().enumerate() {
        let cmd = format!("{bin} rank {spec} --hosts {} --rank {r} {}", addrs.join(","), extra.join(" "));
        eprintln!("[launch] {h}: {cmd}");
        children.push(std::process::Command::new("ssh").arg(h).arg(&cmd).spawn()?);
    }
    let mut ok = true;
    for mut c in children {
        ok &= c.wait()?.success();
    }
    if !ok {
        bail!("one or more ranks failed");
    }
    Ok(())
}

/// Errors are printed as one line (the message and its causes), without a
/// backtrace, and the exit status is 1.
fn main() {
    if let Err(e) = run_cli() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run_cli() -> Result<()> {
    let a = parse_args();
    match a.pos.first().map(|s| s.as_str()) {
        Some("info") => cmd_info(),
        Some("validate") => cmd_validate(&a),
        Some("run") => cmd_run(&a),
        Some("bench") => cmd_bench(&a),
        Some("rank") => cmd_rank(&a),
        Some("launch") => cmd_launch(&a),
        _ => {
            eprintln!("usage: lightsim <info|validate|run|bench|rank|launch> ...");
            eprintln!("  run SPEC.json [--device cpu|cuda[:i,j]] [--out RESULTS.json] [--dimensions 2] [--invariant-axis x|y|z] [--polarization te|tm|both]");
            eprintln!("  bench [--n NX,NY,NZ | --n NU,NV] [--steps N] [--polarization te|tm|both] [--kernels 3d]");
            Ok(())
        }
    }
}
