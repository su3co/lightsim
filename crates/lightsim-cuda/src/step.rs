//! The time step on the device: the fused single-kernel step, the split
//! H and E updates, x ghost planes (Bloch phases included), source
//! injection and the switch between the real and imaginary field sets.

use super::*;

impl CudaBackend {
    pub(crate) fn launch(&self, f: &Function, grid: (u32, u32, u32), block: (u32, u32, u32), args: Args) -> Result<()> {
        let mut slots = args.0;
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
        unsafe { launch_kernel_on_stream(f, grid, block, 0, &self.stream, &mut ptrs) }.map_err(|e| anyhow!("kernel launch failed: {e:?}"))
    }

    pub(crate) fn launch_1d(&self, f: &Function, n: usize, args: Args) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let b = 256u32;
        let g = (n as u64).div_ceil(b as u64) as u32;
        self.launch(f, (g, 1, 1), (b, 1, 1), args)
    }

    pub(crate) fn ghost_copy(&self, dst_plane: usize, src_plane: usize, comps: [Component; 2], factor: f32) -> Result<()> {
        let plane = self.plan.plane();
        for c in comps {
            let base = self.halo_ptr(c);
            let dst = base + self.plane_offset(dst_plane);
            let src = base + self.plane_offset(src_plane);
            if factor == 1.0 {
                unsafe { memcpy_dtod_async(dst, src, plane * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
            } else {
                self.launch_1d(&self.k.scale_copy, plane, Args::new().p(dst).p(src).u64(plane as u64).f(factor))?;
            }
        }
        Ok(())
    }

    pub(crate) fn run_injections(&self, inj: &[InjDev], step: usize) -> Result<()> {
        for j in inj {
            let field = self.new_ptr(j.comp);
            let width = if j.coef_im.is_some() { 2 } else { 1 };
            let row = j.amps.cu_deviceptr() + (step * j.ngroups * width * 4) as u64;
            let grouped = if j.ngroups > 1 { 1 } else { 0 };
            match &j.coef_im {
                None => self.launch_1d(
                    &self.k.inject,
                    j.n,
                    Args::new()
                        .p(field)
                        .p(j.idx.cu_deviceptr())
                        .p(j.coef.cu_deviceptr())
                        .p(j.group.cu_deviceptr())
                        .u(j.n as u32)
                        .p(row)
                        .u(grouped),
                )?,
                Some(ci) => self.launch_1d(
                    &self.k.inject_complex,
                    j.n,
                    Args::new()
                        .p(field)
                        .p(j.idx.cu_deviceptr())
                        .p(j.coef.cu_deviceptr())
                        .p(ci.cu_deviceptr())
                        .p(j.group.cu_deviceptr())
                        .u(j.n as u32)
                        .p(row)
                        .u(grouped),
                )?,
            }
        }
        Ok(())
    }

    pub(crate) fn bc_packed(&self) -> u32 {
        let p = &self.plan;
        p.bc_y[0] | (p.bc_y[1] << 4) | (p.bc_z[0] << 8) | (p.bc_z[1] << 12)
    }

    /// Work of the fused step that reads E^n before the step kernel.
    pub(crate) fn pre_step(&mut self) -> Result<()> {
        self.nl_save_launch()?;
        self.raman_save_launch()?;
        self.gain_pre_launch()?;
        self.gain2_pre_launch()?;
        self.sheet_pre_all()?;
        self.plugin_pre()
    }

    pub(crate) fn fused_step(&self) -> Result<()> {
        self.disp_update_launch()?;
        let f = self.fused.as_ref().unwrap();
        if f.twod.is_some() {
            return self.step2d_launch();
        }
        let p = &self.plan;
        let (nx, ny, nz) = (p.nxl as u32, p.ny() as u32, p.nz() as u32);
        let rd = &f.psi_h[self.cur_psi];
        let wr = &f.psi_h[1 - self.cur_psi];
        let args = Args::new()
            .p(self.e[0].cu_deviceptr())
            .p(self.e[1].cu_deviceptr())
            .p(self.e[2].cu_deviceptr())
            .p(self.h[0].cu_deviceptr())
            .p(self.h[1].cu_deviceptr())
            .p(self.h[2].cu_deviceptr())
            .p(f.eb[0].cu_deviceptr())
            .p(f.eb[1].cu_deviceptr())
            .p(f.eb[2].cu_deviceptr())
            .p(f.hb[0].cu_deviceptr())
            .p(f.hb[1].cu_deviceptr())
            .p(f.hb[2].cu_deviceptr())
            .p(f.mat.cu_deviceptr())
            .p(f.tab.cu_deviceptr())
            .u(f.lossy)
            .f(p.ch_scalar)
            .p(self.inv_p[0].cu_deviceptr())
            .p(self.inv_p[1].cu_deviceptr())
            .p(self.inv_p[2].cu_deviceptr())
            .p(self.inv_d[0].cu_deviceptr())
            .p(self.inv_d[1].cu_deviceptr())
            .p(self.inv_d[2].cu_deviceptr())
            .u(nx)
            .u(ny)
            .u(nz)
            .u(self.bc_packed())
            .u(f.ghost_lo)
            .u(f.chunk)
            .u(f.pml_n[0])
            .u(f.pml_n[1])
            .u(f.pml_n[2])
            .p(f.prof[0].cu_deviceptr())
            .p(f.prof[1].cu_deviceptr())
            .p(f.prof[2].cu_deviceptr())
            .p(f.psi_e[0].cu_deviceptr())
            .p(f.psi_e[1].cu_deviceptr())
            .p(f.psi_e[2].cu_deviceptr())
            .p(rd[0].cu_deviceptr())
            .p(rd[1].cu_deviceptr())
            .p(rd[2].cu_deviceptr())
            .p(wr[0].cu_deviceptr())
            .p(wr[1].cu_deviceptr())
            .p(wr[2].cu_deviceptr())
            .u(f.mat_mode);
        // dense dispersive state, per E component (dummy pointers when off)
        let dz = self.dummy.cu_deviceptr();
        let mut dd = [[dz; 4]; 3];
        let mut dmu = 0u32;
        let (mut npole, mut coefs) = (0u32, dz);
        if let Some(d) = &self.disp_dense {
            npole = d.npole as u32;
            coefs = d.coefs.cu_deviceptr();
            for (a, dm, p, pp, scale, u) in &d.comps {
                dd[*a] = [dm.cu_deviceptr(), p.cu_deviceptr(), pp.cu_deviceptr(), scale.cu_deviceptr()];
                dmu |= (1 << (24 + *a)) | (*u << (8 * *a));
            }
        }
        let mut args = args;
        for q in 0..3 {
            for a in 0..3 {
                args = args.p(dd[a][q]);
            }
        }
        let args = args.p(coefs).p(dd[0][3]).p(dd[1][3]).p(dd[2][3]).u(npole).u(dmu);
        // the other part of complex fields: current fields and H-side CPML
        // state (the plain kernel ignores these)
        let (eo, ho) = self.partner_ptrs();
        let so = match &f.imag {
            Some(im) => {
                let q = &im.psi_h[self.cur_psi];
                [q[0].cu_deviceptr(), q[1].cu_deviceptr(), q[2].cu_deviceptr()]
            }
            None => [dz; 3],
        };
        let args = args
            .p(eo[0])
            .p(eo[1])
            .p(eo[2])
            .p(ho[0])
            .p(ho[1])
            .p(ho[2])
            .p(so[0])
            .p(so[1])
            .p(so[2])
            .f(p.bloch_phase[1].0 as f32)
            .f(self.bsign[1])
            .f(p.bloch_phase[2].0 as f32)
            .f(self.bsign[2]);
        let kernel = if f.imag.is_some() { &self.k.step_fused_bloch } else { &self.k.step_fused };
        let (bk, bj) = f.tile;
        let grid = (nz.div_ceil(bk), ny.div_ceil(bj), nx.div_ceil(f.chunk));
        let smem = 24 * (bk + 1) * (bj + 1);
        let mut slots = args.0;
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
        unsafe { launch_kernel_on_stream(kernel, grid, (bk, bj, 1), smem, &self.stream, &mut ptrs) }
            .map_err(|e| anyhow!("kernel launch failed: {e:?}"))
    }

    pub(crate) fn fused_e_plane(&self) -> Result<()> {
        let f = self.fused.as_ref().unwrap();
        let p = &self.plan;
        let args = Args::new()
            .p(self.e[0].cu_deviceptr())
            .p(self.e[1].cu_deviceptr())
            .p(self.e[2].cu_deviceptr())
            .p(f.hb[0].cu_deviceptr())
            .p(f.hb[1].cu_deviceptr())
            .p(f.hb[2].cu_deviceptr())
            .p(f.eb[0].cu_deviceptr())
            .p(f.eb[1].cu_deviceptr())
            .p(f.eb[2].cu_deviceptr())
            .p(f.mat.cu_deviceptr())
            .p(f.tab.cu_deviceptr())
            .u(f.lossy)
            .p(self.inv_d[0].cu_deviceptr())
            .p(self.inv_d[1].cu_deviceptr())
            .p(self.inv_d[2].cu_deviceptr())
            .u(p.nxl as u32)
            .u(p.ny() as u32)
            .u(p.nz() as u32)
            .u(self.bc_packed())
            .u(f.pml_n[0])
            .u(f.pml_n[1])
            .u(f.pml_n[2])
            .p(f.prof[0].cu_deviceptr())
            .p(f.prof[1].cu_deviceptr())
            .p(f.prof[2].cu_deviceptr())
            .p(f.psi_e[0].cu_deviceptr())
            .p(f.psi_e[1].cu_deviceptr())
            .p(f.psi_e[2].cu_deviceptr())
            .u(f.mat_mode);
        // Bloch y/z wraps: the other part's new H
        let ho = match &f.imag {
            Some(im) => [im.hb[0].cu_deviceptr(), im.hb[1].cu_deviceptr(), im.hb[2].cu_deviceptr()],
            None => [self.dummy.cu_deviceptr(); 3],
        };
        let args = args
            .p(ho[0])
            .p(ho[1])
            .p(ho[2])
            .f(p.bloch_phase[1].0 as f32)
            .f(self.bsign[1])
            .f(p.bloch_phase[2].0 as f32)
            .f(self.bsign[2]);
        self.launch_1d(&self.k.e_plane, p.plane(), args)
    }

    /// Swap the real and imaginary field sets (Bloch mode).
    pub(crate) fn swap_sets(&mut self) {
        let Some(c) = self.cplx.as_mut() else { return };
        std::mem::swap(&mut self.e, &mut c.e);
        std::mem::swap(&mut self.h, &mut c.h);
        for (p, q) in self.pml.iter_mut().zip(c.psi.iter_mut()) {
            std::mem::swap(&mut p.psi, q);
        }
        if let Some(d) = self.disp.as_mut() {
            for (dc, q) in d.comps.iter_mut().zip(c.disp.iter_mut()) {
                std::mem::swap(&mut dc.p, &mut q[0]);
                std::mem::swap(&mut dc.pprev, &mut q[1]);
                std::mem::swap(&mut dc.dp, &mut q[2]);
            }
        }
        for (d, q) in self.dft.iter_mut().zip(c.dft.iter_mut()) {
            for (b, qb) in d.boxes.iter_mut().zip(q.iter_mut()) {
                std::mem::swap(&mut b.re, &mut qb.0);
                std::mem::swap(&mut b.im, &mut qb.1);
            }
        }
        if let Some(f) = self.fused.as_mut() {
            if let Some(im) = f.imag.as_mut() {
                std::mem::swap(&mut f.eb, &mut im.eb);
                std::mem::swap(&mut f.hb, &mut im.hb);
                std::mem::swap(&mut f.psi_e, &mut im.psi_e);
                std::mem::swap(&mut f.psi_h, &mut im.psi_h);
            }
        }
        for b in self.bsign.iter_mut() {
            *b = -*b;
        }
        self.imag_sel = !self.imag_sel;
    }

    /// Partner (other part) E and H pointers for the Bloch wrap.
    pub(crate) fn partner_ptrs(&self) -> ([u64; 3], [u64; 3]) {
        match &self.cplx {
            Some(c) => (
                [c.e[0].cu_deviceptr(), c.e[1].cu_deviceptr(), c.e[2].cu_deviceptr()],
                [c.h[0].cu_deviceptr(), c.h[1].cu_deviceptr(), c.h[2].cu_deviceptr()],
            ),
            None => {
                let d = self.dummy.cu_deviceptr();
                ([d; 3], [d; 3])
            }
        }
    }

    /// Bloch-mixing ghost copy: dst = cr * self + ci * other for both parts.
    pub(crate) fn bloch_ghost(&self, dst_plane: usize, src_plane: usize, comps: [Component; 2], cos: f32, sin: f32) -> Result<()> {
        let Some(c) = &self.cplx else { return Ok(()) };
        let plane = self.plan.plane();
        for comp in comps {
            // E ghosts are filled on the current buffers, H ghosts on the new
            // ones (the B buffers of a fused step)
            let fim = self.fused.as_ref().and_then(|f| f.imag.as_ref().map(|im| (f, im)));
            let (re, im) = if comp.is_e() {
                (self.e[comp.axis()].cu_deviceptr(), c.e[comp.axis()].cu_deviceptr())
            } else if let Some((f, im)) = fim {
                (f.hb[comp.axis()].cu_deviceptr(), im.hb[comp.axis()].cu_deviceptr())
            } else {
                (self.h[comp.axis()].cu_deviceptr(), c.h[comp.axis()].cu_deviceptr())
            };
            let d = self.plane_offset(dst_plane);
            let sp = self.plane_offset(src_plane);
            // re_dst = cos re - sin im ; im_dst = cos im + sin re
            self.launch_1d(&self.k_bloch()?, plane, Args::new().p(re + d).p(re + sp).p(im + sp).u64(plane as u64).f(cos).f(-sin))?;
            self.launch_1d(&self.k_bloch()?, plane, Args::new().p(im + d).p(im + sp).p(re + sp).u64(plane as u64).f(cos).f(sin))?;
        }
        Ok(())
    }

    pub(crate) fn k_bloch(&self) -> Result<Function> {
        Ok(self._module.load_function("bloch_copy")?)
    }

    pub(crate) fn update_h_split(&self) -> Result<()> {
        let (eo, _ho) = self.partner_ptrs();
        let p = &self.plan;
        // magnetic loss: H~ *= da on the lossy cells first
        if let Some(hl) = &self.hloss {
            for (a, l) in hl.iter().enumerate() {
                let args = Args::new().p(self.h[a].cu_deviceptr()).p(l.idx.cu_deviceptr()).p(l.da.cu_deviceptr()).u(l.n as u32);
                self.launch_1d(&self.k.hloss_scale, l.n, args)?;
            }
        }
        if p.nxl > 65535 {
            return Err(anyhow!("slab with more than 65535 x planes; use more slabs"));
        }
        let (grid, block) = self.main_grid();
        let (chx, chy, chz, mode) = match &self.ch {
            Some(c) => (c[0].cu_deviceptr(), c[1].cu_deviceptr(), c[2].cu_deviceptr(), 1u32),
            None => {
                let d = self.dummy.cu_deviceptr();
                (d, d, d, 0)
            }
        };
        let args = Args::new()
            .p(self.h[0].cu_deviceptr())
            .p(self.h[1].cu_deviceptr())
            .p(self.h[2].cu_deviceptr())
            .p(self.e[0].cu_deviceptr())
            .p(self.e[1].cu_deviceptr())
            .p(self.e[2].cu_deviceptr())
            .p(chx)
            .p(chy)
            .p(chz)
            .f(p.ch_scalar)
            .u(mode)
            .p(self.inv_p[0].cu_deviceptr())
            .p(self.inv_p[1].cu_deviceptr())
            .p(self.inv_p[2].cu_deviceptr())
            .u(p.nxl as u32)
            .u(p.ny() as u32)
            .u(p.nz() as u32)
            .u(p.bc_y[1])
            .u(p.bc_z[1])
            .p(eo[0])
            .p(eo[1])
            .p(eo[2])
            .f(p.bloch_phase[1].0 as f32)
            .f(self.bsign[1])
            .f(p.bloch_phase[2].0 as f32)
            .f(self.bsign[2]);
        self.launch(&self.k.update_h, grid, block, args)?;
        for slab in &self.pml {
            let a = slab.axis;
            let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
            let (chb, chc) = match &self.ch {
                Some(c) => (c[bax].cu_deviceptr(), c[cax].cu_deviceptr()),
                None => (self.dummy.cu_deviceptr(), self.dummy.cu_deviceptr()),
            };
            let n = slab.size[0] * slab.size[1] * slab.size[2];
            let args = Args::new()
                .p(self.h[bax].cu_deviceptr())
                .p(self.h[cax].cu_deviceptr())
                .p(self.e[bax].cu_deviceptr())
                .p(self.e[cax].cu_deviceptr())
                .p(chb)
                .p(chc)
                .f(p.ch_scalar)
                .u(mode)
                .p(slab.psi[2].cu_deviceptr())
                .p(slab.psi[3].cu_deviceptr())
                .p(slab.h_b.cu_deviceptr())
                .p(slab.h_a.cu_deviceptr())
                .p(slab.h_k.cu_deviceptr())
                .p(self.inv_p[a].cu_deviceptr())
                .u(a as u32)
                .u(slab.start[a] as u32)
                .u(if a == 0 { p.nxl as u32 } else { p.n[a] as u32 })
                .u(slab.start[0] as u32)
                .u(slab.start[1] as u32)
                .u(slab.start[2] as u32)
                .u(slab.size[0] as u32)
                .u(slab.size[1] as u32)
                .u(slab.size[2] as u32)
                .u(p.ny() as u32)
                .u(p.nz() as u32);
            self.launch_1d(&self.k.pml_h, n, args)?;
        }
        Ok(())
    }

    pub(crate) fn update_e_split(&self) -> Result<()> {
        let (_eo, ho) = self.partner_ptrs();
        self.disp_update_launch()?;
        let p = &self.plan;
        let (grid, block) = self.main_grid();
        let (cax, cay, caz, lossy) = match &self.ca {
            Some(c) => (c[0].cu_deviceptr(), c[1].cu_deviceptr(), c[2].cu_deviceptr(), 1u32),
            None => {
                let d = self.dummy.cu_deviceptr();
                (d, d, d, 0)
            }
        };
        let args = Args::new()
            .p(self.e[0].cu_deviceptr())
            .p(self.e[1].cu_deviceptr())
            .p(self.e[2].cu_deviceptr())
            .p(self.h[0].cu_deviceptr())
            .p(self.h[1].cu_deviceptr())
            .p(self.h[2].cu_deviceptr())
            .p(self.cb[0].cu_deviceptr())
            .p(self.cb[1].cu_deviceptr())
            .p(self.cb[2].cu_deviceptr())
            .p(cax)
            .p(cay)
            .p(caz)
            .u(lossy)
            .p(self.inv_d[0].cu_deviceptr())
            .p(self.inv_d[1].cu_deviceptr())
            .p(self.inv_d[2].cu_deviceptr())
            .u(p.nxl as u32)
            .u(p.ny() as u32)
            .u(p.nz() as u32)
            .u(p.bc_y[0])
            .u(p.bc_z[0])
            .p(ho[0])
            .p(ho[1])
            .p(ho[2])
            .f(p.bloch_phase[1].0 as f32)
            .f(self.bsign[1])
            .f(p.bloch_phase[2].0 as f32)
            .f(self.bsign[2]);
        // the table holds the forward coefficients only
        let args = match &self.split_tab {
            Some((m, t, mode)) if !self.reversing => args.p(m.cu_deviceptr()).p(t.cu_deviceptr()).u(*mode),
            _ => args.p(self.dummy.cu_deviceptr()).p(self.dummy.cu_deviceptr()).u(255),
        };
        self.launch(&self.k.update_e, grid, block, args)?;
        for slab in &self.pml {
            let a = slab.axis;
            let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
            let n = slab.size[0] * slab.size[1] * slab.size[2];
            let args = Args::new()
                .p(self.e[bax].cu_deviceptr())
                .p(self.e[cax].cu_deviceptr())
                .p(self.h[bax].cu_deviceptr())
                .p(self.h[cax].cu_deviceptr())
                .p(self.cb[bax].cu_deviceptr())
                .p(self.cb[cax].cu_deviceptr())
                .p(slab.psi[0].cu_deviceptr())
                .p(slab.psi[1].cu_deviceptr())
                .p(slab.e_b.cu_deviceptr())
                .p(slab.e_a.cu_deviceptr())
                .p(slab.e_k.cu_deviceptr())
                .p(self.inv_d[a].cu_deviceptr())
                .u(a as u32)
                .u(slab.start[a] as u32)
                .u(slab.start[0] as u32)
                .u(slab.start[1] as u32)
                .u(slab.start[2] as u32)
                .u(slab.size[0] as u32)
                .u(slab.size[1] as u32)
                .u(slab.size[2] as u32)
                .u(p.ny() as u32)
                .u(p.nz() as u32);
            self.launch_1d(&self.k.pml_e, n, args)?;
        }
        self.disp_apply_launch()?;
        Ok(())
    }

    /// Complex sources drive the imaginary part with Im(c A): same kernel with
    /// the amplitude table (a_im, -a_re).
    pub(crate) fn run_injections_imag(&self, h_side: bool, step: usize) -> Result<()> {
        let Some(c) = &self.cplx else { return Ok(()) };
        let (list, amps) = if h_side { (&self.inj_h, &c.amps_h) } else { (&self.inj_e, &c.amps_e) };
        self.inject_imag_step(list, amps, step)
    }

    /// Imaginary-part injections of the complex-coefficient entries of
    /// `list` (`amps` = their swapped amplitude tables), into the buffers
    /// that receive the new values of the step.
    pub(crate) fn inject_imag_step(&self, list: &[InjDev], amps: &[Option<DeviceBuffer<f32>>], step: usize) -> Result<()> {
        let Some(c) = &self.cplx else { return Ok(()) };
        let fim = self.fused.as_ref().and_then(|f| f.imag.as_ref());
        for (j, a) in list.iter().zip(amps.iter()) {
            let (Some(ci), Some(amp)) = (&j.coef_im, a) else { continue };
            let ax = j.comp.axis();
            let field = match (fim, j.comp.is_e()) {
                (Some(im), true) => im.eb[ax].cu_deviceptr(),
                (Some(im), false) => im.hb[ax].cu_deviceptr(),
                (None, true) => c.e[ax].cu_deviceptr(),
                (None, false) => c.h[ax].cu_deviceptr(),
            };
            let row = amp.cu_deviceptr() + (step * j.ngroups * 2 * 4) as u64;
            self.launch_1d(
                &self.k.inject_complex,
                j.n,
                Args::new()
                    .p(field)
                    .p(j.idx.cu_deviceptr())
                    .p(j.coef.cu_deviceptr())
                    .p(ci.cu_deviceptr())
                    .p(j.group.cu_deviceptr())
                    .u(j.n as u32)
                    .p(row)
                    .u(if j.ngroups > 1 { 1 } else { 0 }),
            )?;
        }
        Ok(())
    }

    pub(crate) fn main_grid(&self) -> ((u32, u32, u32), (u32, u32, u32)) {
        let (bx, by) = self.block;
        let g = ((self.plan.nz() as u32).div_ceil(bx), (self.plan.ny() as u32).div_ceil(by), self.plan.nxl as u32);
        (g, (bx, by, 1))
    }
}
