//! Device kernels of the user-defined material plug-ins
//! (`lightsim_core::plugin`): the GPU engine gathers the plug-in media's
//! cells to the host and scatters the plug-in's result back each step
//! (`crates/lightsim-cuda/src/plugins.rs`). Launched only for plans with
//! plug-in media.

#![allow(clippy::missing_safety_doc)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod plugin_kernels {
    use super::*;

    /// `out[t] = field[idx[t]]`.
    #[kernel]
    pub unsafe fn plugin_gather(field: *const f32, idx: *const u64, out: *mut f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *out.add(t) = *field.add(*idx.add(t) as usize);
        }
    }

    /// `field[idx[t]] = vals[t]`.
    #[kernel]
    pub unsafe fn plugin_scatter(field: *mut f32, idx: *const u64, vals: *const f32, n: u32) {
        let t = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + thread::threadIdx_x() as usize;
        if t >= n as usize {
            return;
        }
        unsafe {
            *field.add(*idx.add(t) as usize) = *vals.add(t);
        }
    }
}
