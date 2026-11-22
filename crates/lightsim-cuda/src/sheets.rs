//! Zero-thickness sheets on the GPU (`lightsim_core::sheet`): device lists,
//! per-set state and the launches of the `sheet_pre` / `sheet_post`
//! kernels (`kernels/src/sheets.rs`). Only plans with sheets build this.

use super::*;
use anyhow::Context as _;

/// Device lists of one sheet list (one sheet, one tangential component).
struct ListDev {
    comp: Component,
    n: usize,
    idx: DeviceBuffer<u64>,
    k: DeviceBuffer<f32>,
    /// Byte offset of the sheet's row in the coefficient table.
    row_off: u64,
    h: f32,
    chi0: f32,
}

/// Sheets of a backend: lists, coefficient table and the state of each
/// field set (real, imaginary): P, P_prev, E^n, dP per list.
pub(crate) struct SheetDev {
    pre: Function,
    post: Function,
    np: usize,
    coefs: DeviceBuffer<f32>,
    lists: Vec<ListDev>,
    state: [Vec<[DeviceBuffer<f32>; 4]>; 2],
}

pub(crate) fn build(plan: &Plan, module: &Arc<cuda_core::CudaModule>, s: &Arc<CudaStream>) -> Result<Option<SheetDev>> {
    let Some(sp) = &plan.sheet else { return Ok(None) };
    let f = |n: &str| module.load_function(n).with_context(|| format!("kernel {n}"));
    let mut lists = vec![];
    for l in &sp.lists {
        let r = l.row as usize;
        lists.push(ListDev {
            comp: l.comp,
            n: l.idx.len(),
            idx: dbuf(s, &l.idx)?,
            k: dbuf(s, &l.k)?,
            row_off: (r * sp.np * 3 * 4) as u64,
            h: sp.cond[r * lightsim_core::sheet::SHEET_COND],
            chi0: sp.cond[r * lightsim_core::sheet::SHEET_COND + 1],
        });
    }
    let mk = |s: &Arc<CudaStream>| -> Result<Vec<[DeviceBuffer<f32>; 4]>> {
        sp.lists.iter().map(|l| {
            let n = l.idx.len();
            Ok([zbuf(s, (n * sp.np).max(1))?, zbuf(s, (n * sp.np).max(1))?, zbuf(s, n)?, zbuf(s, n)?])
        }).collect()
    };
    let state = [mk(s)?, if plan.bloch { mk(s)? } else { vec![] }];
    let coefs = if sp.coefs.is_empty() { zbuf(s, 1)? } else { dbuf(s, &sp.coefs)? };
    Ok(Some(SheetDev { pre: f("sheet_pre")?, post: f("sheet_post")?, np: sp.np, coefs, lists, state }))
}

impl CudaBackend {
    /// Sheet poles of the selected field set, from E^n (before the E update).
    pub(crate) fn sheet_pre_launch(&self) -> Result<()> {
        let Some(sd) = &self.sheet else { return Ok(()) };
        let set = self.imag_sel as usize;
        for (l, st) in sd.lists.iter().zip(sd.state[set].iter()) {
            if self.idle_2d(l.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.field_ptr(l.comp))
                .p(l.idx.cu_deviceptr())
                .p(sd.coefs.cu_deviceptr() + l.row_off)
                .u(sd.np as u32)
                .p(st[0].cu_deviceptr())
                .p(st[1].cu_deviceptr())
                .p(st[2].cu_deviceptr())
                .p(st[3].cu_deviceptr())
                .u(l.n as u32);
            self.launch_1d(&sd.pre, l.n, args)?;
        }
        Ok(())
    }

    /// Sheet correction of the linearly updated E of the selected set.
    pub(crate) fn sheet_post_launch(&self) -> Result<()> {
        let Some(sd) = &self.sheet else { return Ok(()) };
        let set = self.imag_sel as usize;
        for (l, st) in sd.lists.iter().zip(sd.state[set].iter()) {
            if self.idle_2d(l.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.new_ptr(l.comp))
                .p(l.idx.cu_deviceptr())
                .p(l.k.cu_deviceptr())
                .p(st[2].cu_deviceptr())
                .p(st[3].cu_deviceptr())
                .f(l.h)
                .f(l.chi0)
                .u(l.n as u32);
            self.launch_1d(&sd.post, l.n, args)?;
        }
        Ok(())
    }

    /// `sheet_pre_launch` on every field set (fused step: before the step
    /// kernels of both parts).
    pub(crate) fn sheet_pre_all(&mut self) -> Result<()> {
        if self.sheet.is_none() {
            return Ok(());
        }
        self.sheet_pre_launch()?;
        if self.cplx.is_some() {
            self.swap_sets();
            self.sheet_pre_launch()?;
            self.swap_sets();
        }
        Ok(())
    }

    /// `sheet_post_launch` on every field set (fused step).
    pub(crate) fn sheet_post_all(&mut self) -> Result<()> {
        if self.sheet.is_none() {
            return Ok(());
        }
        self.sheet_post_launch()?;
        if self.cplx.is_some() {
            self.swap_sets();
            self.sheet_post_launch()?;
            self.swap_sets();
        }
        Ok(())
    }
}
