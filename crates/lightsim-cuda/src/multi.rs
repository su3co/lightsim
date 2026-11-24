//! Multi-GPU execution inside one process.
//!
//! The domain is split into x-slabs, one per GPU, with sizes proportional to
//! per-GPU weights. Each slab is a [`CudaBackend`]; the in-process
//! [`Chain`](lightsim_core::run::Chain) exchanges halo planes device-to-device
//! (`cuMemcpyPeerAsync`, which uses NVLink/PCIe P2P when available and
//! host staging otherwise). Peer access is enabled between neighbours when
//! the topology supports it.

use crate::CudaBackend;
use anyhow::Result;
use cuda_core::CudaContext;
use lightsim_core::plan::{CompileOptions, compile_with_grid, decompose_x};
use lightsim_core::run::{Chain, RunOptions, run_chain};
use lightsim_core::{Results, Simulation};

/// Build the slab backends of a simulation for the given GPUs (in x order).
pub fn build_slabs(sim: &Simulation, gpus: &[usize], weights: Option<&[f64]>) -> Result<Vec<CudaBackend>> {
    let grid = sim.build_grid()?;
    let nx = grid.n()[0];
    // a grid with fewer x-planes than two per GPU (a 2-D simulation invariant
    // along x has one) runs on the first GPUs only
    let gpus = &gpus[..gpus.len().min((nx / 2).max(1))];
    let w: Vec<f64> = match weights {
        Some(w) => w[..gpus.len()].to_vec(),
        None => vec![1.0; gpus.len()],
    };
    let parts = decompose_x(nx, &w);
    let mut out = Vec::with_capacity(gpus.len());
    for (r, (&g, &(x0, nxl))) in gpus.iter().zip(parts.iter()).enumerate() {
        let opts = CompileOptions { slab: Some((x0, nxl)), nslabs: gpus.len(), slab_index: r, ..Default::default() };
        let plan = compile_with_grid(sim, grid.clone(), &opts)?;
        out.push(CudaBackend::new(plan, g)?);
    }
    // enable peer access between neighbours when possible
    for r in 0..out.len().saturating_sub(1) {
        enable_peer(out[r].context(), out[r + 1].context());
    }
    if out.len() > 1 && sim.boundaries.x[0].is_periodic() {
        let n = out.len();
        enable_peer(out[n - 1].context(), out[0].context());
    }
    Ok(out)
}

fn enable_peer(a: &std::sync::Arc<CudaContext>, b: &std::sync::Arc<CudaContext>) {
    if a.ordinal() == b.ordinal() {
        return;
    }
    if cuda_core::peer::can_access_peer(a, b).unwrap_or(false) {
        let _ = cuda_core::peer::enable_peer_access(a, b);
        let _ = cuda_core::peer::enable_peer_access(b, a);
    }
}

/// True if GPU `a` can directly access GPU `b`'s memory.
pub fn peer_capable(a: usize, b: usize) -> Result<bool> {
    let (ca, _) = crate::device_module(a)?;
    let (cb, _) = crate::device_module(b)?;
    Ok(cuda_core::peer::can_access_peer(&ca, &cb)?)
}

/// Run a simulation split over several GPUs of this machine.
pub fn run_multi_gpu(sim: &Simulation, gpus: &[usize], opts: &RunOptions) -> Result<Results> {
    let slabs = build_slabs(sim, gpus, None)?;
    let boxed: Vec<Box<dyn lightsim_core::Backend>> = slabs.into_iter().map(|b| Box::new(b) as Box<dyn lightsim_core::Backend>).collect();
    let mut chain = Chain::new(boxed);
    chain.local_wrap = gpus.len() > 1 && sim.boundaries.x[0].is_periodic();
    run_chain(&mut chain, opts)
}
