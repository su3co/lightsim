//! [`CudaBackend`] as a [`Backend`]: the operations the time-stepping drivers
//! call (ghost planes, updates, injections, recording, halos, reversal).

use super::*;

/// Smallest and largest chunk of a time-monitor recording, in bytes.
const CHUNK_MIN_BYTES: usize = 1 << 20;
const CHUNK_MAX_BYTES: usize = 256 << 20;

/// The `[sample][cell]` recording of a time monitor (or of one of its
/// boxes), stored in chunks allocated as a run reaches them: the first
/// holds about 1 MiB, each next one twice the previous, up to 256 MiB, so
/// the memory follows the samples recorded (within a factor of two, at most
/// 256 MiB over) rather than the run-time cap. Chunks are kept for the
/// following runs of the same backend.
pub(crate) struct Recording {
    cells: usize,
    /// Samples a run can record (the run-time cap).
    count: usize,
    /// (first sample, samples, buffer).
    chunks: Vec<(usize, usize, DeviceBuffer<f32>)>,
}

impl Recording {
    pub(crate) fn new(cells: usize, count: usize) -> Recording {
        Recording { cells, count, chunks: vec![] }
    }

    fn end(&self) -> usize {
        self.chunks.last().map(|c| c.0 + c.1).unwrap_or(0)
    }

    /// Allocate chunks until sample `s` (< count) has room.
    pub(crate) fn reserve(&mut self, s: usize, stream: &Arc<CudaStream>) -> Result<()> {
        let row = 4 * self.cells.max(1);
        let (lo, hi) = ((CHUNK_MIN_BYTES / row).max(1), (CHUNK_MAX_BYTES / row).max(1));
        while self.end() <= s {
            let first = self.end();
            let k = self.chunks.len().min(40) as u32;
            let n = lo.saturating_mul(1usize << k).min(hi).min(self.count - first).max(1);
            let b = zbuf(stream, n * self.cells).map_err(|e| anyhow!("time monitor recording ({} samples of {} cells): {e}", s + 1, self.cells))?;
            self.chunks.push((first, n, b));
        }
        Ok(())
    }

    /// Device address of sample `s` (reserved).
    pub(crate) fn ptr(&self, s: usize) -> u64 {
        let c = self.chunks.iter().rev().find(|c| c.0 <= s).expect("reserved sample");
        c.2.cu_deviceptr() + ((s - c.0) * self.cells * 4) as u64
    }

    /// The first `ns` samples, `[sample][cell]`.
    pub(crate) fn download(&self, ns: usize, stream: &Arc<CudaStream>) -> Result<Vec<f32>> {
        let mut v = vec![0f32; ns * self.cells];
        if v.is_empty() {
            return Ok(v);
        }
        for (first, n, b) in &self.chunks {
            if *first >= ns {
                break;
            }
            let m = (*n).min(ns - first) * self.cells;
            let dst = &mut v[first * self.cells..first * self.cells + m];
            unsafe { memcpy_dtoh_async(dst.as_mut_ptr(), b.cu_deviceptr(), m * 4, stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        }
        stream.synchronize()?;
        Ok(v)
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> String {
        let mode = match (&self.fused, &self.plan.twod) {
            (Some(f), Some(td)) if f.twod.is_some() => format!(", 2-D {}", td.label()),
            (Some(_), _) => ", fused".to_string(),
            _ => String::new(),
        };
        format!("cuda:{} ({}{})", self.ordinal, self.device_name, mode)
    }
    fn plan(&self) -> &Plan {
        &self.plan
    }

    fn fill_ghost_e(&mut self) -> Result<()> {
        if self.no_x_ghosts() {
            return Ok(());
        }
        let nxl = self.plan.nxl;
        match self.plan.ghost_e_hi.clone() {
            GhostRule::Zero | GhostRule::Neighbor => Ok(()),
            GhostRule::Periodic { cos, sin } => {
                if self.cplx.is_some() {
                    return self.bloch_ghost(nxl + 1, 1, [Component::Ey, Component::Ez], cos as f32, sin as f32);
                }
                self.ghost_copy(nxl + 1, 1, [Component::Ey, Component::Ez], cos as f32)
            }
            GhostRule::Mirror { factor } => {
                self.ghost_copy(nxl + 1, nxl, [Component::Ey, Component::Ez], factor)?;
                if self.cplx.is_some() {
                    self.swap_sets();
                    self.ghost_copy(nxl + 1, nxl, [Component::Ey, Component::Ez], factor)?;
                    self.swap_sets();
                }
                Ok(())
            }
        }
    }

    fn fill_ghost_h(&mut self) -> Result<()> {
        if self.no_x_ghosts() {
            return Ok(());
        }
        let nxl = self.plan.nxl;
        if self.fused.is_some() && matches!(self.plan.ghost_h_lo, GhostRule::Mirror { .. }) {
            return Ok(()); // handled inside the fused kernel
        }
        match self.plan.ghost_h_lo.clone() {
            GhostRule::Zero | GhostRule::Neighbor => Ok(()),
            GhostRule::Periodic { cos, sin } => {
                if self.cplx.is_some() {
                    return self.bloch_ghost(0, nxl, [Component::Hy, Component::Hz], cos as f32, sin as f32);
                }
                self.ghost_copy(0, nxl, [Component::Hy, Component::Hz], cos as f32)
            }
            GhostRule::Mirror { factor } => {
                self.ghost_copy(0, 1, [Component::Hy, Component::Hz], factor)?;
                if self.cplx.is_some() {
                    self.swap_sets();
                    self.ghost_copy(0, 1, [Component::Hy, Component::Hz], factor)?;
                    self.swap_sets();
                }
                Ok(())
            }
        }
    }

    fn update_h(&mut self) -> Result<()> {
        if self.fused.is_some() {
            self.pre_step()?;
            self.fused_step()?;
            if self.cplx.is_some() {
                // both parts read only the current buffers: independent launches
                self.swap_sets();
                self.fused_step()?;
                self.swap_sets();
            }
            return Ok(());
        }
        self.update_h_split()?;
        if self.cplx.is_some() {
            self.swap_sets();
            self.update_h_split()?;
            self.swap_sets();
        }
        Ok(())
    }

    fn update_e(&mut self) -> Result<()> {
        if let Some(ghost_lo) = self.fused.as_ref().map(|f| f.ghost_lo) {
            if ghost_lo == 1 {
                self.fused_e_plane()?;
                if self.cplx.is_some() {
                    self.swap_sets();
                    self.fused_e_plane()?;
                    self.swap_sets();
                }
            }
            self.disp_apply_launch()?;
            if self.cplx.is_some() && self.disp.is_some() {
                // complex fields with dispersive cells (2-D step only): the
                // imaginary part has its own polarizations
                self.swap_sets();
                self.disp_apply_launch()?;
                self.swap_sets();
            }
            self.disp_dense_launch()?;
            self.nl_apply_launch()?;
            self.raman_apply_launch()?;
            self.gain_post_launch()?;
            self.gain2_post_launch()?;
            self.sheet_post_all()?;
            self.plugin_post()?;
            if self.tensor.is_some() {
                // E^n is still in the current buffers. The E corrections
                // that stand for the H injections belong to the linear
                // update, so they go in before dD is taken from it (as in
                // the split update, where the injected H is in the curl).
                self.tensor_save_launch()?;
                if let Some(f) = &self.fused {
                    self.run_injections(&f.inj_hd, self.inj_step)?;
                }
                self.tensor_dd_launch()?;
            }
            return Ok(());
        }
        self.nl_save_launch()?;
        self.raman_save_launch()?;
        self.gain_pre_launch()?;
        self.gain2_pre_launch()?;
        self.sheet_pre_launch()?;
        self.plugin_pre()?;
        self.tensor_save_launch()?;
        self.herm_save_launch()?;
        self.update_e_split()?;
        self.herm_dd_launch()?;
        self.nl_apply_launch()?;
        self.raman_apply_launch()?;
        self.gain_post_launch()?;
        self.gain2_post_launch()?;
        self.sheet_post_launch()?;
        self.plugin_post()?;
        self.tensor_dd_launch()?;
        if self.cplx.is_some() {
            self.swap_sets();
            self.sheet_pre_launch()?;
            self.herm_save_launch()?;
            self.update_e_split()?;
            self.herm_dd_launch()?;
            self.sheet_post_launch()?;
            self.swap_sets();
        }
        self.herm_apply_launch()?;
        Ok(())
    }

    fn update_h_inject(&mut self, step: usize) -> Result<()> {
        // 2-D: one launch for the step and its sources when it can
        if self.step2d_folded(step)? {
            return Ok(());
        }
        self.update_h()?;
        self.inject_h(step)
    }

    fn inject_h(&mut self, step: usize) -> Result<()> {
        self.inj_step = step;
        if self.inject_packed(true, step)? {
            return Ok(());
        }
        self.run_injections(&self.inj_h, step)?;
        if self.cplx.is_some() {
            self.run_injections_imag(true, step)?;
        }
        Ok(())
    }

    fn inject_e(&mut self, step: usize) -> Result<()> {
        // the step kernel has added this step's terms already
        if self.inj_folded.take() == Some(step) {
            return Ok(());
        }
        if self.inject_packed(false, step)? {
            return Ok(());
        }
        self.run_injections(&self.inj_e, step)?;
        if self.cplx.is_some() {
            self.run_injections_imag(false, step)?;
        }
        if let Some(f) = &self.fused {
            // with off-diagonal tensors these went in before dD (update_e)
            if self.tensor.is_none() {
                self.run_injections(&f.inj_hd, step)?;
            }
            if let Some(im) = &f.imag {
                self.inject_imag_step(&f.inj_hd, &im.amps_hd, step)?;
            }
        }
        Ok(())
    }

    fn record(&mut self, step: usize) -> Result<()> {
        if self.cplx.is_some() {
            self.swap_sets();
            self.record_dft(step)?;
            self.swap_sets();
        }
        self.record_dft(step)?;
        let (ny, nz) = (self.plan.ny() as u32, self.plan.nz() as u32);
        for t in self.time.iter_mut() {
            if step < t.start || (step - t.start) % t.interval != 0 {
                continue;
            }
            let sample = t.steps.len();
            if sample >= t.count {
                continue;
            }
            t.steps.push(step);
            for b in t.boxes.iter_mut() {
                if !b.local {
                    continue;
                }
                let n = b.lsize[0] * b.lsize[1] * b.lsize[2];
                b.rec.reserve(sample, &self.stream)?;
                let out = b.rec.ptr(sample);
                let field = match &self.fused {
                    Some(f) => {
                        if b.comp.is_e() { f.eb[b.comp.axis()].cu_deviceptr() } else { f.hb[b.comp.axis()].cu_deviceptr() }
                    }
                    None => {
                        if b.comp.is_e() { self.e[b.comp.axis()].cu_deviceptr() } else { self.h[b.comp.axis()].cu_deviceptr() }
                    }
                };
                let args = Args::new()
                    .p(field)
                    .p(out)
                    .u(b.lstart[0] as u32)
                    .u(b.lstart[1] as u32)
                    .u(b.lstart[2] as u32)
                    .u(b.lsize[0] as u32)
                    .u(b.lsize[1] as u32)
                    .u(b.lsize[2] as u32)
                    .u(ny)
                    .u(nz);
                let mut slots = args.0;
                let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
                let g = (n as u64).div_ceil(256) as u32;
                unsafe { launch_kernel_on_stream(&self.k.gather, (g.max(1), 1, 1), (256, 1, 1), 0, &self.stream, &mut ptrs) }
                    .map_err(|e| anyhow!("{e:?}"))?;
            }
        }
        if let Some(f) = self.fused.as_mut() {
            std::mem::swap(&mut self.e, &mut f.eb);
            std::mem::swap(&mut self.h, &mut f.hb);
            if let (Some(im), Some(c)) = (f.imag.as_mut(), self.cplx.as_mut()) {
                std::mem::swap(&mut c.e, &mut im.eb);
                std::mem::swap(&mut c.h, &mut im.hb);
            }
            self.cur_psi ^= 1;
            if let Some(d) = self.disp_dense.as_mut() {
                for c in d.comps.iter_mut() {
                    std::mem::swap(&mut c.2, &mut c.3);
                }
            }
        }
        Ok(())
    }

    fn energy(&mut self) -> Result<f64> {
        let mut tot = self.energy_set()?;
        if self.cplx.is_some() {
            self.swap_sets();
            tot += self.energy_set()?;
            self.swap_sets();
        }
        Ok(tot)
    }

    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }

    fn download(&mut self) -> Result<RawResults> {
        let mut raw = self.download_set()?;
        if let Some(c) = &self.cplx {
            // DFT of the complex field: DFT(re) + i DFT(im)
            for (d, (dd, cd)) in raw.dft.iter_mut().zip(self.dft.iter().zip(c.dft.iter())) {
                for ((re, im), (b, (ire, iim))) in d.boxes.iter_mut().zip(dd.boxes.iter().zip(cd.iter())) {
                    if !b.local {
                        continue;
                    }
                    let n = re.len();
                    let mut ar = ire.to_host_vec(&self.stream)?;
                    let mut ai = iim.to_host_vec(&self.stream)?;
                    ar.truncate(n);
                    ai.truncate(n);
                    for t in 0..n {
                        let (x, y) = (re[t], im[t]);
                        re[t] = x - ai[t];
                        im[t] = y + ar[t];
                    }
                }
            }
        }
        Ok(raw)
    }

    fn read_plane(&mut self, comp: Component, side: Side, out: &mut [f32]) -> Result<()> {
        let pi = match side {
            Side::Lo => 1,
            Side::Hi => self.plan.nxl,
        };
        let src = self.halo_ptr(comp) + self.plane_offset(pi);
        unsafe { memcpy_dtoh_async(out.as_mut_ptr(), src, out.len() * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn write_ghost(&mut self, comp: Component, side: Side, data: &[f32]) -> Result<()> {
        let pi = match side {
            Side::Lo => 0,
            Side::Hi => self.plan.nxl + 1,
        };
        let dst = self.halo_ptr(comp) + self.plane_offset(pi);
        unsafe { memcpy_htod_async(dst, data.as_ptr(), data.len() * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn gain_levels(&mut self) -> Result<Option<[f64; 4]>> {
        if self.gain2.is_some() {
            return self.gain2_levels();
        }
        let Some((_, comps)) = &self.gain else { return Ok(None) };
        let mut out = [0f64; 4];
        for c in comps {
            let pop = c.pop.to_host_vec(&self.stream)?;
            for (l, o) in out.iter_mut().enumerate() {
                *o += (0..c.n).map(|t| pop[l * c.n + t] * c.vol[t]).sum::<f64>();
            }
        }
        Ok(Some(out))
    }
    fn finish_e(&mut self) -> Result<()> {
        self.tensor_apply_launch()?;
        self.gyro_apply_launch()
    }

    fn reverse_e(&mut self) -> Result<()> {
        self.ensure_reverse()?;
        self.swap_reverse_e();
        let r = self.fill_ghost_h().and_then(|_| self.update_e());
        self.swap_reverse_e();
        r
    }

    fn reverse_h(&mut self) -> Result<()> {
        self.ensure_reverse()?;
        self.swap_reverse_h();
        let r = self.fill_ghost_e().and_then(|_| self.update_h());
        self.swap_reverse_h();
        r
    }

    fn set_field(&mut self, comp: Component, data: &[f32]) -> Result<()> {
        // 2-D: a field written from outside puts its set in use
        if let Some(td) = self.plan.twod.as_mut() {
            td.drive(comp);
        }
        let dst = self.field_ptr(comp);
        unsafe { memcpy_htod_async(dst, data.as_ptr(), data.len() * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn tensor_read(&mut self, comp: Component, side: Side, which: usize, out: &mut [f32]) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        let plane = self.plan.plane();
        let pi = if side == Side::Lo { 1 } else { self.plan.nxl };
        let buf = if which == 0 { &td.prev } else { &td.en };
        let src = buf[comp.axis()].cu_deviceptr() + ((pi - t.p0) * plane * 4) as u64;
        unsafe { memcpy_dtoh_async(out.as_mut_ptr(), src, plane * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn tensor_write_ghost(&mut self, comp: Component, side: Side, which: usize, data: &[f32]) -> Result<()> {
        let (Some(td), Some(t)) = (&self.tensor, &self.plan.tensor) else { return Ok(()) };
        let plane = self.plan.plane();
        let pi = if side == Side::Lo { 0 } else { self.plan.nxl + 1 };
        let buf = if which == 0 { &td.prev } else { &td.en };
        let dst = buf[comp.axis()].cu_deviceptr() + ((pi - t.p0) * plane * 4) as u64;
        unsafe { memcpy_htod_async(dst, data.as_ptr(), plane * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn select_set(&mut self, imag: bool) -> Result<()> {
        if self.cplx.is_some() && imag != self.imag_sel {
            self.swap_sets();
        }
        Ok(())
    }

    fn field(&mut self, comp: Component) -> Result<Vec<f32>> {
        let b = if comp.is_e() { &self.e[comp.axis()] } else { &self.h[comp.axis()] };
        Ok(b.to_host_vec(&self.stream)?)
    }

    fn device_plane(&self, comp: Component, storage_plane: usize) -> Option<(u64, usize)> {
        Some((self.halo_ptr(comp) + self.plane_offset(storage_plane), self.ctx.cu_ctx() as usize))
    }

    fn copy_plane_from_device(&mut self, comp: Component, dst_plane: usize, src: (u64, usize)) -> Result<()> {
        let dst = self.halo_ptr(comp) + self.plane_offset(dst_plane);
        let bytes = self.plan.plane() * 4;
        let dst_ctx = self.ctx.cu_ctx();
        self.ctx.bind_to_thread()?;
        let r = unsafe {
            cuda_core::sys::cuMemcpyPeerAsync(dst, dst_ctx, src.0, src.1 as cuda_core::sys::CUcontext, bytes, self.stream.cu_stream())
        };
        if r != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
            return Err(anyhow!("cuMemcpyPeerAsync failed: {r:?}"));
        }
        Ok(())
    }
}
