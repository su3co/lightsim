//! Complex (Hermitian) permittivity tensors on the GPU
//! (`lightsim_core::herm`). Plans with such media run the split kernels on
//! complex fields: per set, E^n is saved over the plane range, dD is formed
//! by `tensor_dd` after the linear update, and `herm_apply`
//! (`kernels/src/herm.rs`) couples the two sets.

use super::*;
use anyhow::Context as _;

struct CompDev {
    axis: usize,
    n: usize,
    idx: DeviceBuffer<u64>,
    k: [DeviceBuffer<f32>; 4],
}

pub(crate) struct HermDev {
    apply: Function,
    comps: Vec<CompDev>,
    /// per set (real, imaginary): E^n and dD of x, y, z over [p0, p1)
    en: [[DeviceBuffer<f32>; 3]; 2],
    dd: [[DeviceBuffer<f32>; 3]; 2],
}

pub(crate) fn build(plan: &Plan, module: &Arc<cuda_core::CudaModule>, s: &Arc<CudaStream>) -> Result<Option<HermDev>> {
    let Some(h) = &plan.herm else { return Ok(None) };
    let n = (h.p1 - h.p0) * plan.plane();
    let mut comps = vec![];
    for c in &h.comps {
        comps.push(CompDev { axis: c.comp.axis(), n: c.idx.len(), idx: dbuf(s, &c.idx)?, k: [dbuf(s, &c.kr1)?, dbuf(s, &c.kr2)?, dbuf(s, &c.ki1)?, dbuf(s, &c.ki2)?] });
    }
    let z3 = |s: &Arc<CudaStream>| -> Result<[DeviceBuffer<f32>; 3]> { Ok([zbuf(s, n)?, zbuf(s, n)?, zbuf(s, n)?]) };
    Ok(Some(HermDev { apply: module.load_function("herm_apply").context("kernel herm_apply")?, comps, en: [z3(s)?, z3(s)?], dd: [z3(s)?, z3(s)?] }))
}

impl CudaBackend {
    /// Save E^n of the selected set over the plane range.
    pub(crate) fn herm_save_launch(&self) -> Result<()> {
        let (Some(hd), Some(h)) = (&self.herm, &self.plan.herm) else { return Ok(()) };
        let plane = self.plan.plane();
        let set = self.imag_sel as usize;
        for a in 0..3 {
            let src = self.e[a].cu_deviceptr() + (h.p0 * plane * 4) as u64;
            unsafe { memcpy_dtod_async(hd.en[set][a].cu_deviceptr(), src, (h.p1 - h.p0) * plane * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        }
        Ok(())
    }

    /// dD = (E_lin - E^n) c dt / cb of the selected set.
    pub(crate) fn herm_dd_launch(&self) -> Result<()> {
        let (Some(hd), Some(h)) = (&self.herm, &self.plan.herm) else { return Ok(()) };
        let plane = self.plan.plane();
        let set = self.imag_sel as usize;
        let n = (h.p1 - h.p0) * plane;
        for a in 0..3 {
            let args = Args::new()
                .p(self.field_ptr(Component::e(a)))
                .p(hd.dd[set][a].cu_deviceptr())
                .p(hd.en[set][a].cu_deviceptr())
                .p(self.cb[a].cu_deviceptr())
                .p(self.dummy.cu_deviceptr())
                .u(0)
                .f(h.cdt)
                .u64((h.p0 * plane) as u64)
                .u64(n as u64);
            self.launch_1d(&self.k.tensor_dd, n, args)?;
        }
        Ok(())
    }

    /// The coupling of both sets (the real set selected).
    pub(crate) fn herm_apply_launch(&self) -> Result<()> {
        let (Some(hd), Some(h)) = (&self.herm, &self.plan.herm) else { return Ok(()) };
        for c in &hd.comps {
            let a = c.axis;
            let (b1, b2) = ((a + 1) % 3, (a + 2) % 3);
            let args = Args::new()
                .p(self.field_ptr(Component::e(a)))
                .p(self.new_ptr_imag(Component::e(a)))
                .p(c.idx.cu_deviceptr())
                .p(c.k[0].cu_deviceptr())
                .p(c.k[1].cu_deviceptr())
                .p(c.k[2].cu_deviceptr())
                .p(c.k[3].cu_deviceptr())
                .p(hd.dd[0][b1].cu_deviceptr())
                .p(hd.dd[0][b2].cu_deviceptr())
                .p(hd.dd[1][b1].cu_deviceptr())
                .p(hd.dd[1][b2].cu_deviceptr())
                .u(c.n as u32)
                .u(a as u32)
                .u(self.plan.ny() as u32)
                .u(self.plan.nz() as u32)
                .u(h.p0 as u32)
                .u(h.p1 as u32)
                .u(h.wrap);
            self.launch_1d(&hd.apply, c.n, args)?;
        }
        Ok(())
    }
}
