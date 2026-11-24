//! Launches of the per-medium kernels of the split step: dispersive poles
//! (sparse lists and the dense layout), chi2 / chi3, Raman, gain, tensor
//! and gyrotropic media.

use super::*;

impl CudaBackend {
    pub(crate) fn disp_update_launch(&self) -> Result<()> {
        if let Some(d) = &self.disp {
            for c in &d.comps {
                if self.idle_2d(c.comp) {
                    continue;
                }
                let args = Args::new()
                    .p(self.field_ptr(c.comp))
                    .p(c.idx.cu_deviceptr())
                    .p(c.matid.cu_deviceptr())
                    .p(c.p.cu_deviceptr())
                    .p(c.pprev.cu_deviceptr())
                    .p(d.coefs.cu_deviceptr())
                    .p(c.dp.cu_deviceptr())
                    .u(d.npole as u32)
                    .u(c.n as u32);
                self.launch_1d(&self.k.disp_update, c.n, args)?;
            }
        }
        Ok(())
    }

    pub(crate) fn disp_apply_launch(&self) -> Result<()> {
        if let Some(d) = &self.disp {
            for c in &d.comps {
                if self.idle_2d(c.comp) {
                    continue;
                }
                let args = Args::new()
                    .p(self.new_ptr(c.comp))
                    .p(c.idx.cu_deviceptr())
                    .p(c.dp.cu_deviceptr())
                    .p(c.scale.cu_deviceptr())
                    .u(c.n as u32);
                self.launch_1d(&self.k.disp_apply, c.n, args)?;
            }
        }
        Ok(())
    }

    /// Save E^n at the nonlinear cells (before the E update of the step).
    pub(crate) fn nl_save_launch(&self) -> Result<()> {
        for c in &self.nl {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new().p(self.field_ptr(c.comp)).p(c.idx.cu_deviceptr()).p(c.prev.cu_deviceptr()).u(c.n as u32);
            self.launch_1d(&self.k.nl_save, c.n, args)?;
        }
        Ok(())
    }

    /// Implicit chi2/chi3 correction of the linearly updated E.
    pub(crate) fn nl_apply_launch(&self) -> Result<()> {
        for c in &self.nl {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.new_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.prev.cu_deviceptr())
                .p(c.c2.cu_deviceptr())
                .p(c.c3.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&self.k.nl_apply, c.n, args)?;
        }
        Ok(())
    }

    /// chi3 Raman cells: save E^n and advance Q (before the E update).
    pub(crate) fn raman_save_launch(&self) -> Result<()> {
        let Some((coefs, comps)) = &self.raman else { return Ok(()) };
        for c in comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.field_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.matid.cu_deviceptr())
                .p(coefs.cu_deviceptr())
                .p(c.prev.cu_deviceptr())
                .p(c.q.cu_deviceptr())
                .p(c.qp.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&self.k.raman_save, c.n, args)?;
        }
        Ok(())
    }

    /// chi3 Raman cells: implicit solve for E^{n+1} (after the linear update).
    pub(crate) fn raman_apply_launch(&self) -> Result<()> {
        let Some((_, comps)) = &self.raman else { return Ok(()) };
        for c in comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.new_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.prev.cu_deviceptr())
                .p(c.c2.cu_deviceptr())
                .p(c.c3.cu_deviceptr())
                .p(c.cr.cu_deviceptr())
                .p(c.q.cu_deviceptr())
                .p(c.qp.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&self.k.raman_apply, c.n, args)?;
        }
        Ok(())
    }

    /// Gain cells: P update from E^n (before the E update).
    pub(crate) fn gain_pre_launch(&self) -> Result<()> {
        let Some((tab, comps)) = &self.gain else { return Ok(()) };
        for c in comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.field_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.row.cu_deviceptr())
                .p(tab.cu_deviceptr())
                .p(c.pop.cu_deviceptr())
                .p(c.prev.cu_deviceptr())
                .p(c.p.cu_deviceptr())
                .p(c.pp.cu_deviceptr())
                .p(c.dp.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&self.k.gain_pre, c.n, args)?;
        }
        Ok(())
    }

    /// Gain cells: E^{n+1} -= scale dP and the population update.
    pub(crate) fn gain_post_launch(&self) -> Result<()> {
        let Some((tab, comps)) = &self.gain else { return Ok(()) };
        for c in comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.new_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.row.cu_deviceptr())
                .p(tab.cu_deviceptr())
                .p(c.scale.cu_deviceptr())
                .p(c.prev.cu_deviceptr())
                .p(c.dp.cu_deviceptr())
                .p(c.pop.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&self.k.gain_post, c.n, args)?;
        }
        Ok(())
    }

    /// Off-diagonal tensors: save E^n over the interior plane range.
    pub(crate) fn tensor_save_launch(&self) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        let plane = self.plan.plane();
        let n = (t.i1 - t.i0) * plane;
        for a in 0..3 {
            let dst = td.en[a].cu_deviceptr() + ((t.i0 - t.p0) * plane * 4) as u64;
            let src = self.e[a].cu_deviceptr() + (t.i0 * plane * 4) as u64;
            unsafe { memcpy_dtod_async(dst, src, n * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        }
        Ok(())
    }

    /// Off-diagonal tensors: dD = (E_lin - E^n) cdt / cb over the interior range.
    pub(crate) fn tensor_dd_launch(&self) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        let plane = self.plan.plane();
        let n = (t.i1 - t.i0) * plane;
        for a in 0..3 {
            let off = ((t.i0 - t.p0) * plane * 4) as u64;
            let (ca_ptr, has_ca) = match &self.ca {
                Some(c) => (c[a].cu_deviceptr(), 1u32),
                None => (self.dummy.cu_deviceptr(), 0u32),
            };
            let args = Args::new()
                .p(self.new_ptr(Component::e(a)))
                .p(td.prev[a].cu_deviceptr() + off)
                .p(td.en[a].cu_deviceptr() + off)
                .p(self.cb[a].cu_deviceptr())
                .p(ca_ptr)
                .u(has_ca)
                .f(t.cdt)
                .u64((t.i0 * plane) as u64)
                .u64(n as u64);
            self.launch_1d(&self.k.tensor_dd, n, args)?;
        }
        Ok(())
    }

    /// Off-diagonal tensors: E_a += kappa_ab <dD_b> + kappa_ac <dD_c>.
    pub(crate) fn tensor_apply_launch(&self) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        for (ci, (comp, n, idx, c1, c2, m1, m2)) in td.comps.iter().enumerate() {
            let a = comp.axis();
            let (opm, oscale, ost, osp) = &td.ops[ci];
            let args = Args::new()
                .p(self.new_ptr(Component::e(a)))
                .p(idx.cu_deviceptr())
                .p(c1.cu_deviceptr())
                .p(c2.cu_deviceptr())
                .p(td.prev[(a + 1) % 3].cu_deviceptr())
                .p(td.prev[(a + 2) % 3].cu_deviceptr())
                .p(m1.cu_deviceptr())
                .p(m2.cu_deviceptr())
                .p(td.en[(a + 1) % 3].cu_deviceptr())
                .p(td.en[(a + 2) % 3].cu_deviceptr())
                .u(t.lossy as u32)
                .p(td.en[a].cu_deviceptr())
                .p(opm.cu_deviceptr())
                .p(oscale.cu_deviceptr())
                .p(td.optab.cu_deviceptr())
                .u(t.op_np as u32)
                .p(ost.cu_deviceptr())
                .p(osp.cu_deviceptr())
                .u(*n as u32)
                .u(a as u32)
                .u(self.plan.ny() as u32)
                .u(self.plan.nz() as u32)
                .u(t.p0 as u32)
                .u(t.p1 as u32)
                .u(t.wrap);
            self.launch_1d(&self.k.tensor_apply, *n, args)?;
        }
        Ok(())
    }

    /// Gyrotropic poles: P update from E^n and E_a -= dP_a / eps_inf.
    pub(crate) fn gyro_apply_launch(&self) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        for (a, n, idx, row, scale, p, pp) in &td.gyro {
            let a = *a;
            let args = Args::new()
                .p(self.new_ptr(Component::e(a)))
                .p(idx.cu_deviceptr())
                .p(td.en[a].cu_deviceptr())
                .p(td.en[(a + 1) % 3].cu_deviceptr())
                .p(td.en[(a + 2) % 3].cu_deviceptr())
                .p(row.cu_deviceptr())
                .p(scale.cu_deviceptr())
                .p(td.gyro_tab.cu_deviceptr())
                .u(t.gyro_np as u32)
                .p(p.cu_deviceptr())
                .p(pp.cu_deviceptr())
                .u(*n as u32)
                .u(a as u32)
                .u(self.plan.ny() as u32)
                .u(self.plan.nz() as u32)
                .u(t.p0 as u32)
                .u(t.p1 as u32)
                .u(t.wrap);
            self.launch_1d(&self.k.gyro_apply, *n, args)?;
        }
        Ok(())
    }

    /// Dense dispersive update of the deferred x-plane 0 (the fused kernel
    /// does every other plane inline; E^n in A, E^{n+1} in B).
    pub(crate) fn disp_dense_launch(&self) -> Result<()> {
        let Some(d) = &self.disp_dense else { return Ok(()) };
        if self.fused.as_ref().map(|f| f.ghost_lo) != Some(1) {
            return Ok(());
        }
        let plane = self.plan.plane();
        let n = plane;
        let len = self.plan.storage_len();
        for (a, dm, p, pp, scale, dmu) in &d.comps {
            let comp = Component::e(*a);
            let args = Args::new()
                .p(self.field_ptr(comp))
                .p(self.new_ptr(comp))
                .p(dm.cu_deviceptr())
                .p(p.cu_deviceptr())
                .p(pp.cu_deviceptr())
                .p(d.coefs.cu_deviceptr())
                .p(scale.cu_deviceptr())
                .u(d.npole as u32)
                .u64(plane as u64)
                .u64(n as u64)
                .u64(len as u64)
                .u(*dmu);
            self.launch_1d(&self.k.disp_dense, n, args)?;
        }
        Ok(())
    }
}
