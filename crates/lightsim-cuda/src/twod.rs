//! 2-D execution on the GPU (plans with [`lightsim_core::plan::TwoD`]).
//!
//! A grid that is one periodic cell thick along an axis runs the fused step
//! with its own kernel (`step2d` / `step2d_bloch`, `kernels/src/twod.rs`)
//! in place of `step_fused`: only the three components of each field set in
//! use are updated, and every thread of the launch owns a cell (the 3-D
//! kernel's 32-wide tiles are 31/32 empty on such a grid). Everything else
//! is the fused machinery of [`crate::fused`] unchanged: ping-pong buffers,
//! material table, CPML profiles and state, the E corrections that stand
//! for magnetic-current injections, and along x the ghost planes with the
//! deferred E update of the first plane (so slabs on several GPUs and Bloch
//! phases along x work as in 3-D). When x itself is the invariant axis the
//! kernel applies the y boundary rule itself and nothing is deferred.
//!
//! Two more launches per step are merged here, because on small 2-D grids
//! the step rate is set by the number of launches, not by the cells:
//! [`DftGroup`] accumulates all the DFT boxes of the monitors that share
//! their frequencies in one launch (`dft_boxes`), and [`InjectRound`]
//! applies several source lists in one launch (`inject_lists`).
//!
//! What does not run here falls back to the general kernels, on the same
//! plan: see [`crate::fused::unsupported`] (magnetic media, sources next to
//! a CPML layer, ...); off-diagonal permittivity never yields a 2-D plan.

use crate::{Args, CudaBackend, InjDev, dbuf};
use anyhow::{Result, anyhow};
use cuda_core::{CudaStream, DeviceBuffer, launch_kernel_on_stream};
use lightsim_core::Component;
use lightsim_core::plan::{Plan, Step2d};
use std::ffi::c_void;
use std::sync::Arc;

/// Launch state of the 2-D step.
pub(crate) struct TwoDState {
    pub k: Step2d,
    /// Threads per block of the flat launch.
    pub block: u32,
}

impl TwoDState {
    pub(crate) fn new(plan: &Plan) -> Option<TwoDState> {
        let k = plan.step2d()?;
        // LIGHTSIM_2D_BLOCK overrides the block size (measurements)
        let block = std::env::var("LIGHTSIM_2D_BLOCK").ok().and_then(|v| v.parse::<u32>().ok()).filter(|b| (1..=1024).contains(b)).unwrap_or(256);
        Some(TwoDState { k, block })
    }
}

/// The DFT boxes of the monitors that share their phasor tables, accumulated
/// by one `dft_boxes` launch per step.
pub(crate) struct DftGroup {
    /// Index of the monitor whose phasor tables the group uses.
    pub monitor: usize,
    pub stride: usize,
    pub nf: usize,
    pub total: usize,
    pub nbox: usize,
    pub desc: DeviceBuffer<u32>,
    /// Accumulator addresses (re, im per box) of the real part and, for
    /// complex fields, of the imaginary part.
    pub bufs: DeviceBuffer<u64>,
    pub bufs_imag: Option<DeviceBuffer<u64>>,
}

/// Group the DFT monitors of `plan` (boxes on this slab only). `ptrs(m, b)`
/// gives the (re, im) accumulator addresses of box `b` of monitor `m` for
/// the real (`false`) or imaginary part.
pub(crate) fn dft_groups(plan: &Plan, s: &Arc<CudaStream>, complex: bool, ptrs: &dyn Fn(usize, usize, bool) -> (u64, u64)) -> Result<Vec<DftGroup>> {
    let mut groups: Vec<(usize, Vec<usize>)> = vec![];
    for (m, d) in plan.dft.iter().enumerate() {
        let pos = groups.iter().position(|(first, _)| {
            let f = &plan.dft[*first];
            f.stride == d.stride && f.freqs.len() == d.freqs.len() && f.phasor_e == d.phasor_e && f.phasor_h == d.phasor_h
        });
        match pos {
            Some(g) => groups[g].1.push(m),
            None => groups.push((m, vec![m])),
        }
    }
    let mut out = vec![];
    for (first, members) in groups {
        let mut desc: Vec<u32> = vec![];
        let mut bufs: Vec<u64> = vec![];
        let mut bufs_imag: Vec<u64> = vec![];
        let mut total = 0usize;
        for &m in &members {
            for (b, bx) in plan.dft[m].boxes.iter().enumerate() {
                // a component of a field set the simulation does not contain stays zero
                if !bx.is_local() || plan.twod.map(|t| !t.allow[t.set_of(bx.comp)]).unwrap_or(false) {
                    continue;
                }
                desc.extend_from_slice(&[bx.comp.index() as u32, bx.lstart[0] as u32, bx.lstart[1] as u32, bx.lstart[2] as u32, bx.lsize[0] as u32, bx.lsize[1] as u32, bx.lsize[2] as u32, total as u32]);
                let (re, im) = ptrs(m, b, false);
                bufs.extend_from_slice(&[re, im]);
                if complex {
                    let (re, im) = ptrs(m, b, true);
                    bufs_imag.extend_from_slice(&[re, im]);
                }
                total += bx.lcells();
            }
        }
        if total == 0 {
            continue;
        }
        if total >= u32::MAX as usize {
            return Err(anyhow!("DFT monitors too large for one launch"));
        }
        out.push(DftGroup {
            monitor: first,
            stride: plan.dft[first].stride.max(1),
            nf: plan.dft[first].freqs.len(),
            total,
            nbox: desc.len() / 8,
            desc: dbuf(s, &desc)?,
            bufs: dbuf(s, &bufs)?,
            bufs_imag: if complex { Some(dbuf(s, &bufs_imag)?) } else { None },
        });
    }
    Ok(out)
}

/// Source lists applied by one `inject_lists` launch: they target different
/// field components, so no two entries of a launch write the same cell.
pub(crate) struct InjectRound {
    pub nlist: usize,
    pub total: usize,
    pub table: DeviceBuffer<u64>,
}

/// Pack injection lists into launches. `lists` are (list, amplitude table
/// to use, complex) triples; lists that target the same component go to
/// different rounds.
pub(crate) fn inject_rounds(s: &Arc<CudaStream>, lists: &[(&InjDev, u64)]) -> Result<Vec<InjectRound>> {
    let mut rounds: Vec<Vec<usize>> = vec![];
    for (i, (j, _)) in lists.iter().enumerate() {
        let slot = rounds.iter().position(|r| r.iter().all(|&q| lists[q].0.comp != j.comp));
        match slot {
            Some(r) => rounds[r].push(i),
            None => rounds.push(vec![i]),
        }
    }
    let mut out = vec![];
    for r in rounds {
        let mut table: Vec<u64> = vec![];
        let mut total = 0usize;
        for &i in &r {
            let (j, amps) = lists[i];
            let width = if j.coef_im.is_some() { 2 } else { 1 };
            let rowlen = (j.ngroups * width) as u64;
            let grouped = (j.ngroups > 1) as u64;
            table.extend_from_slice(&[
                j.comp.index() as u64,
                j.idx.cu_deviceptr(),
                j.coef.cu_deviceptr(),
                j.coef_im.as_ref().map(|c| c.cu_deviceptr()).unwrap_or(0),
                j.group.cu_deviceptr(),
                amps,
                rowlen | grouped << 32,
                total as u64,
            ]);
            total += j.n;
        }
        if total >= u32::MAX as usize {
            return Err(anyhow!("source lists too large for one launch"));
        }
        out.push(InjectRound { nlist: r.len(), total, table: dbuf(s, &table)? });
    }
    Ok(out)
}

/// The source lists of a backend packed into launches: `h` after the H
/// update, `e` after the E update (with the E corrections that stand for
/// the H injections), and the same for the imaginary part of complex
/// fields (complex-coefficient lists only).
pub(crate) struct InjRounds {
    pub h: Vec<InjectRound>,
    pub e: Vec<InjectRound>,
    pub h_im: Vec<InjectRound>,
    pub e_im: Vec<InjectRound>,
}

impl CudaBackend {
    /// 2-D mode: apply the H-side (`h_side`) or E-side source lists of
    /// `step` with packed launches. Returns false when the lists must be
    /// applied one by one (not 2-D mode, or `LIGHTSIM_2D_BATCH=0`).
    pub(crate) fn inject_packed(&self, h_side: bool, step: usize) -> Result<bool> {
        if !self.is_2d() || std::env::var("LIGHTSIM_2D_BATCH").as_deref() == Ok("0") {
            return Ok(false);
        }
        if self.inj_rounds.borrow().is_none() {
            let r = self.build_inj_rounds()?;
            *self.inj_rounds.borrow_mut() = Some(r);
        }
        let guard = self.inj_rounds.borrow();
        let r = guard.as_ref().unwrap();
        if h_side {
            self.inject_rounds_launch(&r.h, step, false)?;
            self.inject_rounds_launch(&r.h_im, step, true)?;
        } else {
            self.inject_rounds_launch(&r.e, step, false)?;
            self.inject_rounds_launch(&r.e_im, step, true)?;
        }
        Ok(true)
    }

    fn build_inj_rounds(&self) -> Result<InjRounds> {
        let f = self.fused.as_ref().unwrap();
        let s = &self.stream;
        // Nothing reads the new H between the two injection points unless the
        // first x-plane is deferred or a neighbouring slab takes its halo
        // from it: otherwise the H lists go into the E-side launches.
        let deferred = f.twod.as_ref().map(|t| t.k.deferred).unwrap_or(true);
        let halo = matches!(self.plan.ghost_e_hi, lightsim_core::plan::GhostRule::Neighbor) || matches!(self.plan.ghost_h_lo, lightsim_core::plan::GhostRule::Neighbor);
        let merge = !deferred && !halo;
        fn own(v: &[InjDev]) -> Vec<(&InjDev, u64)> {
            v.iter().map(|j| (j, j.amps.cu_deviceptr())).collect()
        }
        let mut h = own(&self.inj_h);
        let mut e = own(&self.inj_e);
        e.extend(own(&f.inj_hd));
        // imaginary part: Im(c A) of the complex-coefficient lists, through
        // their amplitude tables (a_im, -a_re)
        fn swapped<'a>(v: &'a [InjDev], amps: &[Option<DeviceBuffer<f32>>]) -> Vec<(&'a InjDev, u64)> {
            v.iter()
                .zip(amps.iter())
                .filter_map(|(j, a)| match (&j.coef_im, a) {
                    (Some(_), Some(a)) => Some((j, a.cu_deviceptr())),
                    _ => None,
                })
                .collect()
        }
        let (mut h_im, mut e_im) = (vec![], vec![]);
        if let Some(c) = &self.cplx {
            h_im = swapped(&self.inj_h, &c.amps_h);
            e_im = swapped(&self.inj_e, &c.amps_e);
            if let Some(im) = &f.imag {
                e_im.extend(swapped(&f.inj_hd, &im.amps_hd));
            }
        }
        if merge {
            e.splice(0..0, h.drain(..));
            e_im.splice(0..0, h_im.drain(..));
        }
        Ok(InjRounds { h: inject_rounds(s, &h)?, e: inject_rounds(s, &e)?, h_im: inject_rounds(s, &h_im)?, e_im: inject_rounds(s, &e_im)? })
    }

    /// True when the 2-D step runs (fused mode on a 2-D plan).
    pub fn is_2d(&self) -> bool {
        self.fused.as_ref().map(|f| f.twod.is_some()).unwrap_or(false)
    }

    /// 2-D plans: a component of a field set that is not in use (its fields
    /// are identically zero and its lists need no launch).
    pub(crate) fn idle_2d(&self, c: Component) -> bool {
        self.is_2d() && self.plan.twod.map(|t| !t.is_active(c)).unwrap_or(false)
    }

    /// 2-D plans whose invariant axis is x: the kernel applies the in-plane
    /// boundary rules itself and the x ghost planes are not used.
    pub(crate) fn no_x_ghosts(&self) -> bool {
        self.fused.as_ref().and_then(|f| f.twod.as_ref()).map(|t| t.k.u != 0).unwrap_or(false)
    }

    /// One 2-D step of the selected part: H^{n+1/2} and E^{n+1} of the sets
    /// in use from the current buffers into the B buffers.
    pub(crate) fn step2d_launch(&self) -> Result<()> {
        let f = self.fused.as_ref().unwrap();
        let t = f.twod.as_ref().unwrap();
        let k = &t.k;
        let p = &self.plan;
        let td = p.twod.ok_or_else(|| anyhow!("2-D step on a plan that is not 2-D"))?;
        let pol = Step2d::pol(&td);
        if pol == 0 {
            // nothing drives the fields: they stay zero
            return Ok(());
        }
        let (u, v, w) = (k.u, k.v, k.w);
        let rd = &f.psi_h[self.cur_psi];
        let wr = &f.psi_h[1 - self.cur_psi];
        let (eo, ho) = self.partner_ptrs();
        let dz = self.dummy.cu_deviceptr();
        let so = match &f.imag {
            Some(im) => {
                let q = &im.psi_h[self.cur_psi];
                [q[0].cu_deviceptr(), q[1].cu_deviceptr(), q[2].cu_deviceptr()]
            }
            None => [dz; 3],
        };
        let args = Args::new()
            .p(self.e[u].cu_deviceptr())
            .p(self.e[v].cu_deviceptr())
            .p(self.e[w].cu_deviceptr())
            .p(self.h[u].cu_deviceptr())
            .p(self.h[v].cu_deviceptr())
            .p(self.h[w].cu_deviceptr())
            .p(f.eb[u].cu_deviceptr())
            .p(f.eb[v].cu_deviceptr())
            .p(f.eb[w].cu_deviceptr())
            .p(f.hb[u].cu_deviceptr())
            .p(f.hb[v].cu_deviceptr())
            .p(f.hb[w].cu_deviceptr())
            .p(f.mat.cu_deviceptr())
            .p(f.tab.cu_deviceptr())
            .u(f.mat_mode)
            .u(f.lossy)
            .u(k.cols)
            .f(p.ch_scalar)
            .p(self.inv_p[u].cu_deviceptr())
            .p(self.inv_p[v].cu_deviceptr())
            .p(self.inv_d[u].cu_deviceptr())
            .p(self.inv_d[v].cu_deviceptr())
            .u(k.nu)
            .u(k.nv)
            .u(k.base)
            .u(k.bc)
            .u(f.pml_n[u])
            .u(f.pml_n[v])
            .p(f.prof[u].cu_deviceptr())
            .p(f.prof[v].cu_deviceptr())
            .p(f.psi_e[u].cu_deviceptr())
            .p(f.psi_e[v].cu_deviceptr())
            .p(rd[u].cu_deviceptr())
            .p(rd[v].cu_deviceptr())
            .p(wr[u].cu_deviceptr())
            .p(wr[v].cu_deviceptr())
            .u(pol)
            .f(k.sg)
            .p(eo[u])
            .p(eo[v])
            .p(eo[w])
            .p(ho[u])
            .p(ho[v])
            .p(ho[w])
            .p(so[u])
            .p(so[v])
            .f(p.bloch_phase[u].0 as f32)
            .f(self.bsign[u])
            .f(p.bloch_phase[v].0 as f32)
            .f(self.bsign[v]);
        let kernel = if f.imag.is_some() { &self.k.step2d_bloch } else { &self.k.step2d };
        let cells = k.nu as u64 * k.nv as u64;
        let grid = cells.div_ceil(t.block as u64) as u32;
        let mut slots = args.0;
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
        unsafe { launch_kernel_on_stream(kernel, (grid, 1, 1), (t.block, 1, 1), 0, &self.stream, &mut ptrs) }.map_err(|e| anyhow!("kernel launch failed: {e:?}"))
    }

    /// Accumulate the grouped DFT monitors at `step` from the buffers the
    /// step has just written (the selected part).
    pub(crate) fn record_dft_groups(&self, step: usize) -> Result<()> {
        let (ny, nz) = (self.plan.ny() as u32, self.plan.nz() as u32);
        for g in &self.dft_groups {
            if step % g.stride != 0 {
                continue;
            }
            let d = &self.dft[g.monitor];
            let row = (step * g.nf * 2 * 4) as u64;
            let bufs = match (&g.bufs_imag, self.imag_sel) {
                (Some(b), true) => b.cu_deviceptr(),
                _ => g.bufs.cu_deviceptr(),
            };
            let mut args = Args::new();
            for c in Component::ALL {
                args = args.p(self.new_ptr(c));
            }
            let args = args
                .p(bufs)
                .p(g.desc.cu_deviceptr())
                .u(g.nbox as u32)
                .p(d.phasor_e.cu_deviceptr() + row)
                .p(d.phasor_h.cu_deviceptr() + row)
                .u(g.nf as u32)
                .u(g.total as u32)
                .u(ny)
                .u(nz);
            self.launch_1d(&self.k.dft_boxes, g.total, args)?;
        }
        Ok(())
    }

    /// Apply packed source lists at `step` to the buffers the step is
    /// writing (of the real part, or of the imaginary part when `imag`).
    pub(crate) fn inject_rounds_launch(&self, rounds: &[InjectRound], step: usize, imag: bool) -> Result<()> {
        if rounds.is_empty() {
            return Ok(());
        }
        let mut fields = [0u64; 6];
        for c in Component::ALL {
            fields[c.index()] = if imag { self.new_ptr_imag(c) } else { self.new_ptr(c) };
        }
        for r in rounds {
            let mut args = Args::new();
            for f in fields {
                args = args.p(f);
            }
            let args = args.p(r.table.cu_deviceptr()).u(r.nlist as u32).u(r.total as u32).u(step as u32);
            self.launch_1d(&self.k.inject_lists, r.total, args)?;
        }
        Ok(())
    }
}
