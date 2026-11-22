//! GPU implementation of [`lightsim_core::autodiff::Engine`].
//!
//! One persistent [`CudaBackend`] per device. Run configurations (sources
//! and monitors) are uploaded once and attached by swapping their device
//! lists into the backend, so a run allocates nothing: the fields, the CPML
//! state and the attached monitors are zeroed with `cuMemsetD8Async` and the
//! step kernels of the plain solver (fused when the plan allows it) do the
//! rest. On top of that:
//!
//! * a design update uploads the coefficient sub-blocks of the design box
//!   (`ad_scatter_box`); in fused mode the cells of the box get their own
//!   rows of the material table, so only those rows are rewritten;
//! * the DFT monitors of a run are accumulated by one kernel launch per step
//!   (`ad_dft_multi`) instead of one per box, which is what bounds the step
//!   rate of small problems; the design-region DFT fields of every run stay
//!   in that run's buffers and the adjoint product is formed on the device
//!   (`ad_product`); time monitors (objective samples, the PML shell) are
//!   likewise recorded and replayed by one launch each (`ad_gather_multi`,
//!   `ad_scatter_multi`);
//! * the reverse pass keeps a second field set for the adjoint, swapped into
//!   the backend for the transposed step: the forward update kernels, with
//!   the transposed CPML (`ad_pml_*`) in place of the forward one when the
//!   plan has absorbing layers (without layers the adjoint step is the plain
//!   forward step, fused included). The forward fields are reversed in place
//!   with the split kernels and the inverted coefficients, replaying the
//!   recorded PML shell (`ad_scatter_multi`), and `dL/d(cb, ca)` is
//!   accumulated over the design box (`ad_accumulate`).
//!
//! The kernels are in `kernels/src/autodiff.rs`.

use crate::{Args, CudaBackend, InjDev, dbuf, fused, zbuf};
use anyhow::{Result, anyhow, bail};
use cuda_core::memory::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async};
use cuda_core::{DeviceBuffer, Function};
use lightsim_core::Component;
use lightsim_core::autodiff::design::IndexBox;
use lightsim_core::autodiff::engine::{Engine, IoSet, ReverseConfig, ReverseOut};
use lightsim_core::engine::{Backend, RawDft, RawTime};
use lightsim_core::plan::{GhostRule, Injection, Plan};
use num_complex::Complex64;

struct AdKernels {
    dft_multi: Function,
    gather_multi: Function,
    scatter_multi: Function,
    scatter: Function,
    inject_scaled: Function,
    inject_complex_scaled: Function,
    product: Function,
    accumulate: Function,
    disp_pre: Function,
    disp_post: Function,
    nl_post: Function,
    nl_adj_pre: Function,
    nl_adj_post: Function,
    pml_phi: Function,
    pml_h: Function,
    pml_e: Function,
}

/// All the DFT monitors of a run configuration, accumulated by one kernel
/// launch per step: the boxes share one descriptor table and two arrays.
struct MultiDft {
    nf: usize,
    stride: usize,
    /// Cells over all boxes.
    total: usize,
    nbox: usize,
    desc: DeviceBuffer<u32>,
    re: DeviceBuffer<f32>,
    im: DeviceBuffer<f32>,
    phasor_e: DeviceBuffer<f32>,
    phasor_h: DeviceBuffer<f32>,
    /// Per monitor, per box: (first cell, cells); cells = 0 if not local.
    boxes: Vec<Vec<(usize, usize)>>,
}

impl MultiDft {
    /// Device pointers (re, im), cells and frequencies of every box of monitor `m`.
    fn views(&self, m: usize) -> Result<Vec<(u64, u64, usize, usize)>> {
        let b = self.boxes.get(m).ok_or_else(|| anyhow!("no DFT monitor {m}"))?;
        Ok(b.iter().map(|&(first, n)| (self.re.cu_deviceptr() + (self.nf * first * 4) as u64, self.im.cu_deviceptr() + (self.nf * first * 4) as u64, n, self.nf)).collect())
    }
}

/// One time monitor: all its boxes are sampled by one kernel launch into a
/// single `[sample][cell]` buffer.
struct MultiTime {
    interval: usize,
    start: usize,
    /// Samples the buffer can hold.
    count: usize,
    /// Cells over all (local) boxes.
    total: usize,
    nbox: usize,
    desc: DeviceBuffer<u32>,
    buf: DeviceBuffer<f32>,
    steps: Vec<usize>,
    /// Per box: (first cell, cells); cells = 0 if not local.
    boxes: Vec<(usize, usize)>,
}

/// Device lists of one run configuration.
struct IoDev {
    inj_h: Vec<InjDev>,
    inj_e: Vec<InjDev>,
    /// E corrections equivalent to the H injections (fused mode).
    inj_hd: Vec<InjDev>,
    /// Not swapped into the backend: the engine accumulates the DFT and
    /// samples the time monitors itself.
    dft: Option<MultiDft>,
    time: Vec<MultiTime>,
}

struct Io {
    host: IoSet,
    /// E corrections equivalent to the H injections (fused mode), computed
    /// when the configuration is registered.
    derived: Vec<Injection>,
    dev: Option<IoDev>,
}

/// A second set of fields and auxiliary state (the adjoint of the reverse
/// pass), exchanged with the backend's own around the transposed step.
struct StateSet {
    e: [DeviceBuffer<f32>; 3],
    h: [DeviceBuffer<f32>; 3],
    /// Plans without absorbing layers on a fused backend (the adjoint step
    /// is then the fused forward step): B buffers and CPML state.
    fused: Option<([DeviceBuffer<f32>; 3], [DeviceBuffer<f32>; 3], [DeviceBuffer<f32>; 3], [[DeviceBuffer<f32>; 3]; 2])>,
    cur_psi: usize,
    /// Plans with absorbing layers: state of the transposed CPML per slab,
    /// (phi_b, phi_c) on the E cells and (theta_b, theta_c) on the H cells.
    psi: Vec<[DeviceBuffer<f32>; 4]>,
    /// Dispersive state per component: (P, P_prev, dP).
    disp: Vec<[DeviceBuffer<f32>; 3]>,
}

struct SourceDev {
    comp: Component,
    n: usize,
    idx: DeviceBuffer<u64>,
    coef: DeviceBuffer<f32>,
    group: DeviceBuffer<u32>,
    table: DeviceBuffer<f32>,
}

struct Reverse {
    shell: Option<usize>,
    design: [IndexBox; 3],
    /// Permeability designs: H boxes, H'(n+1/2) and the sums q1 (compensated).
    design_h: [IndexBox; 3],
    snap_h: [DeviceBuffer<f32>; 3],
    q1: [DeviceBuffer<f32>; 3],
    compq: [DeviceBuffer<f32>; 3],
    sources: Vec<SourceDev>,
    snap: [DeviceBuffer<f32>; 3],
    p1: [DeviceBuffer<f32>; 3],
    comp: [DeviceBuffer<f32>; 3],
    p2: [DeviceBuffer<f32>; 3],
    zeros: DeviceBuffer<f32>,
    /// chi2 / chi3 components (as `be.nl`): interior mask, dE'/dE, dE'/dE_lin
    /// of the step being undone, and the direct part of the adjoint.
    nl: Vec<(DeviceBuffer<u32>, DeviceBuffer<f32>, DeviceBuffer<f32>, DeviceBuffer<f32>)>,
}

pub struct CudaEngine {
    be: CudaBackend,
    k: AdKernels,
    ios: Vec<Io>,
    cur: Option<usize>,
    /// Upload staging buffer and its capacity in floats.
    stage: Option<(DeviceBuffer<f32>, usize)>,
    /// Hull of the coefficient boxes written since the last commit.
    pending: IndexBox,
    /// Fused mode: the box whose cells own material-table rows, and the
    /// first of those rows.
    rows: Option<(IndexBox, usize)>,
    prod: Vec<(DeviceBuffer<f32>, usize)>,
    adj: Option<StateSet>,
    rev: Option<Reverse>,
}

fn ad_kernels(be: &CudaBackend) -> Result<AdKernels> {
    let f = |n: &str| be._module.load_function(n).map_err(|e| anyhow!("kernel {n}: {e:?} (rebuild the PTX with scripts/build_kernels.sh)"));
    Ok(AdKernels {
        dft_multi: f("ad_dft_multi")?,
        gather_multi: f("ad_gather_multi")?,
        scatter_multi: f("ad_scatter_multi")?,
        scatter: f("ad_scatter_box")?,
        inject_scaled: f("ad_inject_scaled")?,
        inject_complex_scaled: f("ad_inject_complex_scaled")?,
        product: f("ad_product")?,
        accumulate: f("ad_accumulate")?,
        disp_pre: f("ad_disp_pre")?,
        disp_post: f("ad_disp_post")?,
        nl_post: f("ad_nl_post")?,
        nl_adj_pre: f("ad_nl_adj_pre")?,
        nl_adj_post: f("ad_nl_adj_post")?,
        pml_phi: f("ad_pml_phi")?,
        pml_h: f("ad_pml_h")?,
        pml_e: f("ad_pml_e")?,
    })
}

fn inj_dev(be: &CudaBackend, v: &[Injection]) -> Result<Vec<InjDev>> {
    let s = &be.stream;
    v.iter()
        .filter(|j| !j.idx.is_empty())
        .map(|j| {
            Ok(InjDev {
                comp: j.comp,
                n: j.idx.len(),
                idx: dbuf(s, &j.idx)?,
                coef: dbuf(s, &j.coef)?,
                coef_im: match &j.coef_im {
                    Some(c) => Some(dbuf(s, c)?),
                    None => None,
                },
                group: dbuf(s, &j.group)?,
                amps: dbuf(s, &j.amps)?,
                ngroups: j.ngroups,
            })
        })
        .collect()
}

fn box_args(a: Args, lo: [usize; 3], size: [usize; 3], plan: &Plan) -> Args {
    a.u(lo[0] as u32).u(lo[1] as u32).u(lo[2] as u32).u(size[0] as u32).u(size[1] as u32).u(size[2] as u32).u(plan.ny() as u32).u(plan.nz() as u32)
}

impl CudaEngine {
    /// Engine for `plan` (without sources and monitors) on GPU `ordinal`.
    pub fn new(mut plan: Plan, ordinal: usize) -> Result<CudaEngine> {
        Self::build(&mut plan, ordinal, true)
    }

    fn build(plan: &mut Plan, ordinal: usize, allow_fused: bool) -> Result<CudaEngine> {
        if plan.bloch || plan.tensor.is_some() {
            bail!("gradient engines do not support Bloch boundaries or off-diagonal permittivity");
        }
        if matches!(plan.ghost_e_hi, GhostRule::Neighbor) || matches!(plan.ghost_h_lo, GhostRule::Neighbor) {
            bail!("gradient engines run on a single slab (one GPU per engine)");
        }
        let _ = IoSet::take(plan);
        let mut be = CudaBackend::new_with(plan.clone(), ordinal, allow_fused)?;
        // the split E update can read a material table instead of the
        // per-cell coefficient arrays; design updates rewrite the arrays
        // (which the PML kernels and the reversed update use anyway)
        be.split_tab = None;
        let k = ad_kernels(&be)?;
        Ok(CudaEngine { be, k, ios: vec![], cur: None, stage: None, pending: IndexBox::default(), rows: None, prod: vec![], adj: None, rev: None })
    }

    /// The underlying backend.
    pub fn backend(&self) -> &CudaBackend {
        &self.be
    }

    fn bind(&self) -> Result<()> {
        self.be.ctx.bind_to_thread().map_err(|e| anyhow!("{e:?}"))
    }

    fn zero<T>(&self, b: &DeviceBuffer<T>) -> Result<()> {
        unsafe { memset_d8_async(b.cu_deviceptr(), 0, b.num_bytes(), self.be.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))
    }

    /// Replace the fused backend by a split one (configurations whose
    /// sources the fused step cannot take, or the dense dispersive mode
    /// before a reverse pass). Called while configurations are registered,
    /// before any run: device-side results would not survive it.
    fn rebuild_split(&mut self) -> Result<()> {
        if let Some(c) = self.cur.take() {
            self.swap_io(c);
        }
        let mut plan = self.be.plan.clone();
        let fresh = Self::build(&mut plan, self.be.ordinal, false)?;
        self.be = fresh.be;
        self.k = fresh.k;
        for io in self.ios.iter_mut() {
            io.dev = None;
            io.derived.clear();
        }
        self.rows = None;
        self.adj = None;
        self.rev = None;
        self.prod.clear();
        Ok(())
    }

    /// Upload the device lists of configuration `io` (once).
    fn ensure_dev(&mut self, io: usize) -> Result<()> {
        if io >= self.ios.len() {
            bail!("unknown run configuration {io}");
        }
        if self.ios[io].dev.is_some() || self.cur == Some(io) {
            return Ok(());
        }
        let be = &self.be;
        let s = &be.stream;
        let host = &self.ios[io].host;
        let nsteps = be.plan.nsteps;
        let dft = match host.dft.first() {
            None => None,
            Some(d0) => {
                if host.dft.iter().any(|d| d.stride != d0.stride || d.phasor_e != d0.phasor_e || d.phasor_h != d0.phasor_h) {
                    bail!("the DFT monitors of a gradient run must share their frequencies, apodization and stride");
                }
                let nf = d0.freqs.len();
                let mut desc: Vec<u32> = vec![];
                let mut boxes = vec![];
                let mut total = 0usize;
                for d in &host.dft {
                    let mut per = vec![];
                    for b in &d.boxes {
                        let n = b.lcells();
                        per.push((total, n));
                        if n > 0 {
                            desc.extend_from_slice(&[b.comp.index() as u32, b.lstart[0] as u32, b.lstart[1] as u32, b.lstart[2] as u32, b.lsize[0] as u32, b.lsize[1] as u32, b.lsize[2] as u32, total as u32]);
                            total += n;
                        }
                    }
                    boxes.push(per);
                }
                if total >= u32::MAX as usize {
                    bail!("DFT monitors too large for one launch");
                }
                Some(MultiDft { nf, stride: d0.stride.max(1), total, nbox: desc.len() / 8, desc: dbuf(s, &desc)?, re: zbuf(s, nf * total)?, im: zbuf(s, nf * total)?, phasor_e: dbuf(s, &d0.phasor_e)?, phasor_h: dbuf(s, &d0.phasor_h)?, boxes })
            }
        };
        let mut time = vec![];
        for t in &host.time {
            let count = if nsteps > t.start_step { (nsteps - t.start_step).div_ceil(t.interval) } else { 0 };
            let mut desc: Vec<u32> = vec![];
            let mut boxes = vec![];
            let mut total = 0usize;
            for b in &t.boxes {
                let n = b.lcells();
                boxes.push((total, n));
                if n > 0 {
                    desc.extend_from_slice(&[b.comp.index() as u32, b.lstart[0] as u32, b.lstart[1] as u32, b.lstart[2] as u32, b.lsize[0] as u32, b.lsize[1] as u32, b.lsize[2] as u32, total as u32]);
                    total += n;
                }
            }
            if total >= u32::MAX as usize {
                bail!("time monitor too large for one launch");
            }
            time.push(MultiTime { interval: t.interval, start: t.start_step, count, total, nbox: desc.len() / 8, desc: dbuf(s, &desc)?, buf: zbuf(s, total * count)?, steps: vec![], boxes });
        }
        let dev = IoDev { inj_h: inj_dev(be, &host.inj_h)?, inj_e: inj_dev(be, &host.inj_e)?, inj_hd: inj_dev(be, &self.ios[io].derived)?, dft, time };
        self.ios[io].dev = Some(dev);
        Ok(())
    }

    /// Exchange the lists of configuration `io` with the backend's.
    fn swap_io(&mut self, io: usize) {
        let Io { host, dev, .. } = &mut self.ios[io];
        let d = dev.as_mut().expect("device lists");
        std::mem::swap(&mut self.be.inj_h, &mut d.inj_h);
        std::mem::swap(&mut self.be.inj_e, &mut d.inj_e);
        if let Some(f) = self.be.fused.as_mut() {
            std::mem::swap(&mut f.inj_hd, &mut d.inj_hd);
        }
        // 2-D mode packs the lists into launches: repack on the next step
        self.be.inj_rounds.replace(None);
        host.swap(&mut self.be.plan);
    }

    fn attach(&mut self, io: usize) -> Result<()> {
        if self.cur == Some(io) {
            return Ok(());
        }
        self.ensure_dev(io)?;
        if let Some(c) = self.cur.take() {
            self.swap_io(c);
        }
        self.swap_io(io);
        self.cur = Some(io);
        Ok(())
    }

    /// The DFT monitors of configuration `io`.
    fn dft_dev(&mut self, io: usize) -> Result<&MultiDft> {
        self.ensure_dev(io)?;
        self.ios[io].dev.as_ref().unwrap().dft.as_ref().ok_or_else(|| anyhow!("configuration {io} has no DFT monitor"))
    }

    /// Accumulate the DFT monitors of the attached configuration at step
    /// `n` from the buffers the step has just written.
    fn dft_step(&self, n: usize) -> Result<()> {
        let Some(io) = self.cur else { return Ok(()) };
        let Some(m) = self.ios[io].dev.as_ref().and_then(|d| d.dft.as_ref()) else { return Ok(()) };
        if m.total == 0 || n % m.stride != 0 {
            return Ok(());
        }
        let be = &self.be;
        let row = (n * m.nf * 2 * 4) as u64;
        let mut args = Args::new();
        for c in Component::ALL {
            args = args.p(be.new_ptr(c));
        }
        let args = args
            .p(m.re.cu_deviceptr())
            .p(m.im.cu_deviceptr())
            .p(m.desc.cu_deviceptr())
            .u(m.nbox as u32)
            .p(m.phasor_e.cu_deviceptr() + row)
            .p(m.phasor_h.cu_deviceptr() + row)
            .u(m.nf as u32)
            .u(m.total as u32)
            .u(be.plan.ny() as u32)
            .u(be.plan.nz() as u32);
        be.launch_1d(&self.k.dft_multi, m.total, args)
    }

    /// The six field pointers the step has just written (kernel argument order).
    fn new_fields(&self, mut args: Args) -> Args {
        for c in Component::ALL {
            args = args.p(self.be.new_ptr(c));
        }
        args
    }

    /// Sample the time monitors of the attached configuration at step `n`.
    fn time_step(&mut self, n: usize) -> Result<()> {
        let Some(io) = self.cur else { return Ok(()) };
        let (ny, nz) = (self.be.plan.ny() as u32, self.be.plan.nz() as u32);
        let mut launches = vec![];
        for t in self.ios[io].dev.as_mut().unwrap().time.iter_mut() {
            if n < t.start || (n - t.start) % t.interval != 0 || t.steps.len() >= t.count {
                continue;
            }
            let sample = t.steps.len();
            t.steps.push(n);
            if t.total > 0 {
                launches.push((t.buf.cu_deviceptr() + (sample * t.total * 4) as u64, t.desc.cu_deviceptr(), t.nbox, t.total));
            }
        }
        for (out, desc, nbox, total) in launches {
            let args = self.new_fields(Args::new()).p(out).p(desc).u(nbox as u32).u(total as u32).u(ny).u(nz);
            self.be.launch_1d(&self.k.gather_multi, total, args)?;
        }
        Ok(())
    }

    /// field[box] = data (staged through a device buffer).
    fn scatter(&mut self, dst: u64, bx: IndexBox, data: &[f32]) -> Result<()> {
        let n = bx.cells();
        if n == 0 {
            return Ok(());
        }
        if self.stage.as_ref().map(|s| s.1 < n).unwrap_or(true) {
            self.stage = Some((zbuf(&self.be.stream, n)?, n));
        }
        let st = self.stage.as_ref().unwrap().0.cu_deviceptr();
        unsafe { memcpy_htod_async(st, data.as_ptr(), n * 4, self.be.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        let args = box_args(Args::new().p(dst).p(st), bx.lo, bx.size, &self.be.plan);
        self.be.launch_1d(&self.k.scatter, n, args)?;
        self.be.stream.synchronize()?;
        Ok(())
    }

    /// dst[plane] = factor * src[plane] on the current buffers of `comps`.
    fn plane_copy(&self, dst_plane: usize, src_plane: usize, comps: [Component; 2], factor: f32) -> Result<()> {
        let plane = self.be.plan.plane();
        for c in comps {
            let base = self.be.field_ptr(c);
            let (dst, src) = (base + self.be.plane_offset(dst_plane), base + self.be.plane_offset(src_plane));
            if factor == 1.0 {
                unsafe { memcpy_dtod_async(dst, src, plane * 4, self.be.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
            } else {
                self.be.launch_1d(&self.be.k.scale_copy, plane, Args::new().p(dst).p(src).u64(plane as u64).f(factor))?;
            }
        }
        Ok(())
    }

    /// Low-x H ghost of the *current* buffers (the backend's own fill acts
    /// on the buffers being written, which in fused mode are the B set).
    fn ghost_h_current(&self) -> Result<()> {
        let nxl = self.be.plan.nxl;
        match self.be.plan.ghost_h_lo.clone() {
            GhostRule::Zero | GhostRule::Neighbor => Ok(()),
            GhostRule::Periodic { cos, .. } => self.plane_copy(0, nxl, [Component::Hy, Component::Hz], cos as f32),
            GhostRule::Mirror { factor } => self.plane_copy(0, 1, [Component::Hy, Component::Hz], factor),
        }
    }

    /// Inverted coefficients of the reversed updates: -cb/ca, 1/ca, -ch.
    fn ensure_rev_coefs(&mut self) -> Result<()> {
        if self.be.rev.is_some() {
            return Ok(());
        }
        self.be.rev = Some(Box::new(crate::reverse_coefs(&self.be.plan, &self.be.stream)?));
        Ok(())
    }

    fn new_state(&self) -> Result<StateSet> {
        let be = &self.be;
        let s = &be.stream;
        let len = be.plan.storage_len();
        let three = || -> Result<[DeviceBuffer<f32>; 3]> { Ok([zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?]) };
        let fused = match &be.fused {
            Some(f) if be.pml.is_empty() => {
                let like = |b: &[DeviceBuffer<f32>; 3]| -> Result<[DeviceBuffer<f32>; 3]> { Ok([zbuf(s, b[0].len())?, zbuf(s, b[1].len())?, zbuf(s, b[2].len())?]) };
                Some((three()?, three()?, like(&f.psi_e)?, [like(&f.psi_h[0])?, like(&f.psi_h[1])?]))
            }
            _ => None,
        };
        let mut psi = vec![];
        for p in &be.pml {
            psi.push([zbuf(s, p.psi[0].len())?, zbuf(s, p.psi[1].len())?, zbuf(s, p.psi[2].len())?, zbuf(s, p.psi[3].len())?]);
        }
        let mut disp = vec![];
        if let Some(d) = &be.disp {
            for c in &d.comps {
                disp.push([zbuf(s, c.p.len())?, zbuf(s, c.pprev.len())?, zbuf(s, c.dp.len())?]);
            }
        }
        Ok(StateSet { e: three()?, h: three()?, fused, cur_psi: 0, psi, disp })
    }

    fn zero_state(&self, st: &StateSet) -> Result<()> {
        for a in 0..3 {
            self.zero(&st.e[a])?;
            self.zero(&st.h[a])?;
        }
        if let Some((eb, hb, pe, ph)) = &st.fused {
            for a in 0..3 {
                self.zero(&eb[a])?;
                self.zero(&hb[a])?;
                self.zero(&pe[a])?;
                self.zero(&ph[0][a])?;
                self.zero(&ph[1][a])?;
            }
        }
        for p in &st.psi {
            for q in p {
                self.zero(q)?;
            }
        }
        for d in &st.disp {
            for q in d {
                self.zero(q)?;
            }
        }
        Ok(())
    }

    /// Exchange the backend's fields and auxiliary state with `st`.
    fn swap_state(be: &mut CudaBackend, st: &mut StateSet) {
        std::mem::swap(&mut be.e, &mut st.e);
        std::mem::swap(&mut be.h, &mut st.h);
        if let (Some(f), Some((eb, hb, pe, ph))) = (be.fused.as_mut(), st.fused.as_mut()) {
            std::mem::swap(&mut f.eb, eb);
            std::mem::swap(&mut f.hb, hb);
            std::mem::swap(&mut f.psi_e, pe);
            std::mem::swap(&mut f.psi_h, ph);
            std::mem::swap(&mut be.cur_psi, &mut st.cur_psi);
        }
        if let Some(d) = be.disp.as_mut() {
            for (c, q) in d.comps.iter_mut().zip(st.disp.iter_mut()) {
                std::mem::swap(&mut c.p, &mut q[0]);
                std::mem::swap(&mut c.pprev, &mut q[1]);
                std::mem::swap(&mut c.dp, &mut q[2]);
            }
        }
    }

    /// Transposed CPML of the adjoint step on the fields currently in the
    /// backend: `h_side` applies the transposed E-side CPML to H (after the
    /// plain H update), otherwise the transposed H-side CPML to E.
    fn pml_transposed(&self, slabs: &[crate::PmlDev], st: &[[DeviceBuffer<f32>; 4]], h_side: bool) -> Result<()> {
        let be = &self.be;
        let p = &be.plan;
        let n = [p.nxl, p.ny(), p.nz()];
        for (slab, phi) in slabs.iter().zip(st) {
            let a = slab.axis;
            let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
            let cells = slab.size[0] * slab.size[1] * slab.size[2];
            // phi <- bE phi + e on the E cells, or theta <- bH theta + h on the H cells
            let (fb, fc, sb, sc, pb) = if h_side {
                (be.e[bax].cu_deviceptr(), be.e[cax].cu_deviceptr(), &phi[0], &phi[1], &slab.e_b)
            } else {
                (be.h[bax].cu_deviceptr(), be.h[cax].cu_deviceptr(), &phi[2], &phi[3], &slab.h_b)
            };
            let args = Args::new().p(fb).p(fc).p(sb.cu_deviceptr()).p(sc.cu_deviceptr()).p(pb.cu_deviceptr()).u(a as u32).u(slab.start[a] as u32);
            be.launch_1d(&self.k.pml_phi, cells, box_args(args, slab.start, slab.size, p))?;
            // the slab plus the layer whose difference reaches into it
            let mut lo = slab.start;
            let mut size = slab.size;
            if h_side {
                if lo[a] > 0 {
                    lo[a] -= 1;
                    size[a] += 1;
                }
            } else if lo[a] + size[a] < n[a] {
                size[a] += 1;
            }
            let total = size[0] * size[1] * size[2];
            let tail = |args: Args| -> Args {
                args.u(lo[0] as u32)
                    .u(lo[1] as u32)
                    .u(lo[2] as u32)
                    .u(size[0] as u32)
                    .u(size[1] as u32)
                    .u(size[2] as u32)
                    .u(slab.start[0] as u32)
                    .u(slab.start[1] as u32)
                    .u(slab.start[2] as u32)
                    .u(slab.size[1] as u32)
                    .u(slab.size[2] as u32)
                    .u(p.ny() as u32)
                    .u(p.nz() as u32)
            };
            if h_side {
                let (chb, chc, mode) = match &be.ch {
                    Some(c) => (c[bax].cu_deviceptr(), c[cax].cu_deviceptr(), 1u32),
                    None => (be.dummy.cu_deviceptr(), be.dummy.cu_deviceptr(), 0u32),
                };
                let args = Args::new()
                    .p(be.h[bax].cu_deviceptr())
                    .p(be.h[cax].cu_deviceptr())
                    .p(be.e[bax].cu_deviceptr())
                    .p(be.e[cax].cu_deviceptr())
                    .p(phi[0].cu_deviceptr())
                    .p(phi[1].cu_deviceptr())
                    .p(chb)
                    .p(chc)
                    .f(p.ch_scalar)
                    .u(mode)
                    .p(slab.e_a.cu_deviceptr())
                    .p(slab.e_k.cu_deviceptr())
                    .p(be.inv_p[a].cu_deviceptr())
                    .u(a as u32)
                    .u(slab.start[a] as u32)
                    .u(slab.size[a] as u32)
                    .u(n[a] as u32);
                be.launch_1d(&self.k.pml_h, total, tail(args))?;
            } else {
                let args = Args::new()
                    .p(be.e[bax].cu_deviceptr())
                    .p(be.e[cax].cu_deviceptr())
                    .p(be.h[bax].cu_deviceptr())
                    .p(be.h[cax].cu_deviceptr())
                    .p(phi[2].cu_deviceptr())
                    .p(phi[3].cu_deviceptr())
                    .p(be.cb[bax].cu_deviceptr())
                    .p(be.cb[cax].cu_deviceptr())
                    .p(slab.h_a.cu_deviceptr())
                    .p(slab.h_k.cu_deviceptr())
                    .p(be.inv_d[a].cu_deviceptr())
                    .u(a as u32)
                    .u(slab.start[a] as u32)
                    .u(slab.size[a] as u32);
                be.launch_1d(&self.k.pml_e, total, tail(args))?;
            }
        }
        Ok(())
    }

    /// One transposed step on the adjoint fields `adj` (swapped into the
    /// backend for its duration). Without absorbing layers it is the forward
    /// step (fused if the backend is); with layers, the split update kernels
    /// with the transposed CPML in place of the forward one.
    fn adjoint_step(&mut self, adj: &mut StateSet) -> Result<()> {
        Self::swap_state(&mut self.be, adj);
        // magnetic loss: the transposed step scales H after the E update
        // (the forward step scales it before its curl update)
        let hloss = self.be.hloss.take();
        let res = if self.be.pml.is_empty() {
            (|| -> Result<()> {
                let be = &mut self.be;
                be.fill_ghost_e()?;
                be.update_h()?;
                be.fill_ghost_h()?;
                be.update_e()?;
                be.finish_e()?;
                // the buffer swap `record` does after a fused step
                if let Some(f) = be.fused.as_mut() {
                    std::mem::swap(&mut be.e, &mut f.eb);
                    std::mem::swap(&mut be.h, &mut f.hb);
                    be.cur_psi ^= 1;
                    if let Some(d) = be.disp_dense.as_mut() {
                        for c in d.comps.iter_mut() {
                            std::mem::swap(&mut c.2, &mut c.3);
                        }
                    }
                }
                Ok(())
            })()
        } else {
            let pml = std::mem::take(&mut self.be.pml);
            // the dispersive correction is launched here, on the current
            // buffers: the backend's own launch writes it to the buffers a
            // fused step would have filled
            let disp = self.be.disp.take();
            let res = (|| -> Result<()> {
                self.be.fill_ghost_e()?;
                self.be.update_h_split()?;
                self.pml_transposed(&pml, &adj.psi, true)?;
                self.ghost_h_current()?;
                if let Some(d) = &disp {
                    for c in &d.comps {
                        let args = Args::new()
                            .p(self.be.e[c.comp.axis()].cu_deviceptr())
                            .p(c.idx.cu_deviceptr())
                            .p(c.matid.cu_deviceptr())
                            .p(c.p.cu_deviceptr())
                            .p(c.pprev.cu_deviceptr())
                            .p(d.coefs.cu_deviceptr())
                            .p(c.dp.cu_deviceptr())
                            .u(d.npole as u32)
                            .u(c.n as u32);
                        self.be.launch_1d(&self.be.k.disp_update, c.n, args)?;
                    }
                }
                self.be.update_e_split()?;
                self.pml_transposed(&pml, &adj.psi, false)?;
                if let Some(d) = &disp {
                    for c in &d.comps {
                        let args = Args::new().p(self.be.e[c.comp.axis()].cu_deviceptr()).p(c.idx.cu_deviceptr()).p(c.dp.cu_deviceptr()).p(c.scale.cu_deviceptr()).u(c.n as u32);
                        self.be.launch_1d(&self.be.k.disp_apply, c.n, args)?;
                    }
                }
                Ok(())
            })();
            self.be.disp = disp;
            self.be.pml = pml;
            res
        };
        let res = res.and_then(|_| {
            if let Some(hl) = &hloss {
                for (a, l) in hl.iter().enumerate() {
                    let args = Args::new().p(self.be.h[a].cu_deviceptr()).p(l.idx.cu_deviceptr()).p(l.da.cu_deviceptr()).u(l.n as u32);
                    self.be.launch_1d(&self.be.k.hloss_scale, l.n, args)?;
                }
            }
            Ok(())
        });
        self.be.hloss = hloss;
        Self::swap_state(&mut self.be, adj);
        res
    }

    /// Remove the source injections of step `n` from the current buffers.
    fn uninject(&self, h_side: bool, n: usize) -> Result<()> {
        let list = if h_side { &self.be.inj_h } else { &self.be.inj_e };
        for j in list {
            let field = self.be.field_ptr(j.comp);
            let width = if j.coef_im.is_some() { 2 } else { 1 };
            let row = j.amps.cu_deviceptr() + (n * j.ngroups * width * 4) as u64;
            let grouped = if j.ngroups > 1 { 1 } else { 0 };
            match &j.coef_im {
                None => self.be.launch_1d(
                    &self.k.inject_scaled,
                    j.n,
                    Args::new().p(field).p(j.idx.cu_deviceptr()).p(j.coef.cu_deviceptr()).p(j.group.cu_deviceptr()).u(j.n as u32).p(row).u(grouped).f(-1.0),
                )?,
                Some(ci) => self.be.launch_1d(
                    &self.k.inject_complex_scaled,
                    j.n,
                    Args::new().p(field).p(j.idx.cu_deviceptr()).p(j.coef.cu_deviceptr()).p(ci.cu_deviceptr()).p(j.group.cu_deviceptr()).u(j.n as u32).p(row).u(grouped).f(-1.0),
                )?,
            }
        }
        Ok(())
    }

    /// Overwrite the shell boxes of E (`e`) or H from sample `sample` of the
    /// shell recording (`None`: zeros), on the current buffers.
    fn restore_shell(&self, e: bool, sample: Option<usize>) -> Result<()> {
        let r = self.rev.as_ref().unwrap();
        let (Some(ti), Some(io)) = (r.shell, self.cur) else { return Ok(()) };
        let t = &self.ios[io].dev.as_ref().unwrap().time[ti];
        if t.total == 0 {
            return Ok(());
        }
        let src = match sample {
            Some(s) => t.buf.cu_deviceptr() + (s * t.total * 4) as u64,
            None => r.zeros.cu_deviceptr(),
        };
        let mut args = Args::new();
        for c in Component::ALL {
            args = args.p(self.be.field_ptr(c));
        }
        let args = args.p(src).p(t.desc.cu_deviceptr()).u(t.nbox as u32).u(t.total as u32).u(self.be.plan.ny() as u32).u(self.be.plan.nz() as u32).u(if e { 1 } else { 2 });
        self.be.launch_1d(&self.k.scatter_multi, t.total, args)
    }

    /// Cotangent sources of step `n` into the adjoint fields.
    fn add_sources(&self, e: bool, n: usize) -> Result<()> {
        let r = self.rev.as_ref().unwrap();
        let adj = self.adj.as_ref().unwrap();
        for s in r.sources.iter().filter(|s| s.comp.is_e() == e) {
            let field = if e { adj.e[s.comp.axis()].cu_deviceptr() } else { adj.h[s.comp.axis()].cu_deviceptr() };
            let row = s.table.cu_deviceptr() + (n * s.n * 4) as u64;
            let args = Args::new().p(field).p(s.idx.cu_deviceptr()).p(s.coef.cu_deviceptr()).p(s.group.cu_deviceptr()).u(s.n as u32).p(row).u(if s.n > 1 { 1 } else { 0 });
            self.be.launch_1d(&self.be.k.inject, s.n, args)?;
        }
        Ok(())
    }

    fn disp_reverse(&self, post: bool) -> Result<()> {
        let Some(d) = &self.be.disp else { return Ok(()) };
        for c in &d.comps {
            let a = c.comp.axis();
            let base = Args::new()
                .p(self.be.field_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.matid.cu_deviceptr())
                .p(c.p.cu_deviceptr())
                .p(c.pprev.cu_deviceptr())
                .p(d.coefs.cu_deviceptr())
                .p(c.scale.cu_deviceptr());
            if post {
                let (ca, has) = match &self.be.ca {
                    Some(v) => (v[a].cu_deviceptr(), 1u32),
                    None => (self.be.dummy.cu_deviceptr(), 0u32),
                };
                self.be.launch_1d(&self.k.disp_post, c.n, base.p(ca).u(has).u(d.npole as u32).u(c.n as u32))?;
            } else {
                self.be.launch_1d(&self.k.disp_pre, c.n, base.u(d.npole as u32).u(c.n as u32))?;
            }
        }
        Ok(())
    }
}

impl Engine for CudaEngine {
    fn name(&self) -> String {
        self.be.name()
    }
    fn plan(&self) -> &Plan {
        &self.be.plan
    }
    fn add_io(&mut self, mut io: IoSet) -> Result<usize> {
        // fused mode applies the H injections through equivalent E
        // corrections; sources it cannot take (next to a PML or on the last
        // plane) put the whole engine on the split kernels, now, before any
        // run has left results on the device
        let mut derived = vec![];
        if self.be.fused.is_some() {
            if self.cur.is_some() {
                bail!("run configurations must be registered before the first run");
            }
            std::mem::swap(&mut self.be.plan.inj_h, &mut io.inj_h);
            let d = fused::derived_e_injections(&self.be.plan);
            std::mem::swap(&mut self.be.plan.inj_h, &mut io.inj_h);
            match d {
                Ok(v) => derived = v,
                Err(why) => {
                    if std::env::var("LIGHTSIM_VERBOSE").is_ok() {
                        eprintln!("lightsim: gradient engine uses the split kernels: {why}");
                    }
                    self.rebuild_split()?;
                }
            }
        }
        self.ios.push(Io { host: io, derived, dev: None });
        Ok(self.ios.len() - 1)
    }

    fn write_h_coefficients(&mut self, a: usize, bx: IndexBox, ch: &[f32]) -> Result<()> {
        self.bind()?;
        let (ny, nz) = (self.be.plan.ny(), self.be.plan.nz());
        let Some(pch) = self.be.plan.ch.as_mut() else { bail!("the plan has no per-cell H coefficients") };
        for t in 0..bx.cells() {
            pch[a][bx.storage(t, ny, nz)] = ch[t];
        }
        let dst = self.be.ch.as_ref().ok_or_else(|| anyhow!("no H coefficient arrays on the device"))?[a].cu_deviceptr();
        self.scatter(dst, bx, ch)?;
        // reversed coefficients (-ch; the design cells have no magnetic loss)
        if let Some(r) = self.be.rev.as_ref().and_then(|r| r.ch.as_ref()) {
            let neg: Vec<f32> = ch.iter().map(|v| -v).collect();
            let dst = r[a].cu_deviceptr();
            self.scatter(dst, bx, &neg)?;
        }
        Ok(())
    }

    fn write_coefficients(&mut self, a: usize, bx: IndexBox, cb: &[f32], ca: Option<&[f32]>) -> Result<()> {
        self.bind()?;
        let (ny, nz) = (self.be.plan.ny(), self.be.plan.nz());
        for t in 0..bx.cells() {
            let s = bx.storage(t, ny, nz);
            self.be.plan.cb[a][s] = cb[t];
            if let (Some(c), Some(v)) = (self.be.plan.ca.as_mut(), ca) {
                c[a][s] = v[t];
            }
        }
        let has_ca = self.be.ca.is_some();
        self.scatter(self.be.cb[a].cu_deviceptr(), bx, cb)?;
        if let (true, Some(v)) = (has_ca, ca) {
            self.scatter(self.be.ca.as_ref().unwrap()[a].cu_deviceptr(), bx, v)?;
        }
        if self.be.rev.is_some() {
            let rcb: Vec<f32> = (0..bx.cells()).map(|t| -cb[t] / ca.map(|v| v[t]).unwrap_or(1.0)).collect();
            let dst = self.be.rev.as_ref().unwrap().cb[a].cu_deviceptr();
            self.scatter(dst, bx, &rcb)?;
            if let (Some(dst), Some(v)) = (self.be.rev.as_ref().unwrap().ca.as_ref().map(|c| c[a].cu_deviceptr()), ca) {
                let rca: Vec<f32> = v.iter().map(|x| 1.0 / x).collect();
                self.scatter(dst, bx, &rca)?;
            }
        }
        self.pending = self.pending.union(&bx);
        Ok(())
    }

    fn commit_coefficients(&mut self) -> Result<()> {
        self.bind()?;
        let hull = std::mem::take(&mut self.pending);
        if hull.is_empty() || self.be.fused.is_none() {
            return Ok(());
        }
        let (ny, nz) = (self.be.plan.ny(), self.be.plan.nz());
        // the cells of the design hull get their own rows of the material
        // table (appended after the rows of the rest of the scene); the
        // per-cell index is rebuilt and packed once, here
        if self.rows.map(|(bx, _)| bx.union(&hull) != bx).unwrap_or(true) {
            let bx = self.rows.map(|(b, _)| b.union(&hull)).unwrap_or(hull);
            let (mut mat, mut tab) = fused::material_table(&self.be.plan);
            let n_old = tab.len() / 6;
            for t in 0..bx.cells() {
                mat[bx.storage(t, ny, nz)] = (n_old + t) as u32;
            }
            tab.resize(6 * (n_old + bx.cells()), 0.0);
            let n_materials = n_old + bx.cells();
            let (bytes, mode) = fused::pack_index(mat, n_materials);
            let f = self.be.fused.as_mut().unwrap();
            f.mat = dbuf(&self.be.stream, &bytes)?;
            f.mat_mode = mode;
            f.tab = dbuf(&self.be.stream, &tab)?;
            f.n_materials = n_materials;
            self.rows = Some((bx, n_old));
        }
        let (bx, row0) = self.rows.unwrap();
        let plan = &self.be.plan;
        let ca = |a: usize, s: usize| plan.ca.as_ref().map(|c| c[a][s]).unwrap_or(1.0);
        let mut rows = Vec::with_capacity(6 * bx.cells());
        for t in 0..bx.cells() {
            let s = bx.storage(t, ny, nz);
            rows.extend_from_slice(&[plan.cb[0][s], plan.cb[1][s], plan.cb[2][s], ca(0, s), ca(1, s), ca(2, s)]);
        }
        let dst = self.be.fused.as_ref().unwrap().tab.cu_deviceptr() + (6 * row0 * 4) as u64;
        unsafe { memcpy_htod_async(dst, rows.as_ptr(), rows.len() * 4, self.be.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.be.stream.synchronize()?;
        Ok(())
    }

    fn begin(&mut self, io: usize) -> Result<()> {
        self.bind()?;
        self.rev = None;
        self.attach(io)?;
        // 2-D plans: the field sets this configuration's sources drive
        if let Some(td) = self.be.plan.twod.as_mut() {
            td.active = [false, false];
        }
        self.be.plan.drive_2d();
        let be = &self.be;
        for a in 0..3 {
            self.zero(&be.e[a])?;
            self.zero(&be.h[a])?;
        }
        if let Some(f) = &be.fused {
            for a in 0..3 {
                self.zero(&f.eb[a])?;
                self.zero(&f.hb[a])?;
                self.zero(&f.psi_e[a])?;
                self.zero(&f.psi_h[0][a])?;
                self.zero(&f.psi_h[1][a])?;
            }
        }
        for p in &be.pml {
            for q in &p.psi {
                self.zero(q)?;
            }
        }
        if let Some(d) = &be.disp {
            for c in &d.comps {
                self.zero(&c.p)?;
                self.zero(&c.pprev)?;
                self.zero(&c.dp)?;
            }
        }
        if let Some(d) = &be.disp_dense {
            for c in &d.comps {
                self.zero(&c.2)?;
                self.zero(&c.3)?;
            }
        }
        for c in &be.nl {
            self.zero(&c.prev)?;
        }
        if let Some(m) = self.ios[io].dev.as_ref().and_then(|d| d.dft.as_ref()) {
            self.zero(&m.re)?;
            self.zero(&m.im)?;
        }
        self.be.cur_psi = 0;
        for t in self.ios[io].dev.as_mut().unwrap().time.iter_mut() {
            t.steps.clear();
        }
        Ok(())
    }

    fn step(&mut self, n: usize) -> Result<()> {
        // the order of `forward_step`, with the DFT monitors accumulated by
        // one launch before `record` (time monitors, buffer swap)
        let be = &mut self.be;
        be.fill_ghost_e()?;
        be.update_h()?;
        be.inject_h(n)?;
        be.fill_ghost_h()?;
        be.update_e()?;
        be.finish_e()?;
        be.inject_e(n)?;
        self.dft_step(n)?;
        self.time_step(n)?;
        self.be.record(n)
    }
    fn energy(&mut self) -> Result<f64> {
        self.be.energy()
    }
    fn synchronize(&mut self) -> Result<()> {
        self.be.stream.synchronize()?;
        Ok(())
    }

    fn read_dft(&mut self, io: usize, monitor: usize) -> Result<RawDft> {
        self.bind()?;
        self.be.stream.synchronize()?;
        let stream = self.be.stream.clone();
        let views = self.dft_dev(io)?.views(monitor)?;
        let mut boxes = vec![];
        for (re, im, n, nf) in views {
            let (mut r, mut i) = (vec![0f32; n * nf], vec![0f32; n * nf]);
            if n > 0 {
                unsafe {
                    memcpy_dtoh_async(r.as_mut_ptr(), re, n * nf * 4, stream.cu_stream()).map_err(|e| anyhow!("{e:?}"))?;
                    memcpy_dtoh_async(i.as_mut_ptr(), im, n * nf * 4, stream.cu_stream()).map_err(|e| anyhow!("{e:?}"))?;
                }
                stream.synchronize()?;
            }
            boxes.push((r, i));
        }
        Ok(RawDft { boxes })
    }

    fn write_dft(&mut self, io: usize, monitor: usize, data: &RawDft) -> Result<()> {
        self.bind()?;
        let stream = self.be.stream.clone();
        let views = self.dft_dev(io)?.views(monitor)?;
        if views.len() != data.boxes.len() {
            bail!("DFT data with {} boxes for a monitor with {}", data.boxes.len(), views.len());
        }
        for ((re, im, n, nf), (dr, di)) in views.into_iter().zip(&data.boxes) {
            if dr.is_empty() {
                continue;
            }
            if dr.len() != n * nf || di.len() != n * nf {
                bail!("DFT data does not fit the monitor box");
            }
            unsafe {
                memcpy_htod_async(re, dr.as_ptr(), dr.len() * 4, stream.cu_stream()).map_err(|e| anyhow!("{e:?}"))?;
                memcpy_htod_async(im, di.as_ptr(), di.len() * 4, stream.cu_stream()).map_err(|e| anyhow!("{e:?}"))?;
            }
            stream.synchronize()?;
        }
        Ok(())
    }

    fn read_time(&mut self, io: usize, monitor: usize) -> Result<RawTime> {
        self.bind()?;
        self.ensure_dev(io)?;
        self.be.stream.synchronize()?;
        let t = self.ios[io].dev.as_ref().unwrap().time.get(monitor).ok_or_else(|| anyhow!("configuration {io} has no time monitor {monitor}"))?;
        let ns = t.steps.len();
        let mut all = vec![0f32; ns * t.total];
        if !all.is_empty() {
            unsafe { memcpy_dtoh_async(all.as_mut_ptr(), t.buf.cu_deviceptr(), all.len() * 4, self.be.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
            self.be.stream.synchronize()?;
        }
        // per box: [sample][cell]
        let boxes = t
            .boxes
            .iter()
            .map(|&(first, n)| {
                let mut v = Vec::with_capacity(ns * n);
                for s in 0..ns {
                    v.extend_from_slice(&all[s * t.total + first..s * t.total + first + n]);
                }
                v
            })
            .collect();
        Ok(RawTime { boxes, steps: t.steps.clone() })
    }

    fn product_begin(&mut self, like: (usize, usize)) -> Result<()> {
        self.bind()?;
        let sizes: Vec<usize> = self.dft_dev(like.0)?.views(like.1)?.iter().map(|v| v.2).collect();
        if self.prod.len() != sizes.len() || self.prod.iter().zip(&sizes).any(|(p, n)| p.1 != *n) {
            self.prod = sizes.iter().map(|&n| Ok((zbuf(&self.be.stream, n)?, n))).collect::<Result<Vec<_>>>()?;
        } else {
            for p in &self.prod {
                self.zero(&p.0)?;
            }
        }
        Ok(())
    }

    fn product_add(&mut self, a: (usize, usize), b: (usize, usize), coef: &[Complex64]) -> Result<()> {
        self.bind()?;
        // device pointers of the two monitors, per box: (re, im, cells, nf)
        let pa = self.dft_dev(a.0)?.views(a.1)?;
        let pb = self.dft_dev(b.0)?.views(b.1)?;
        if pa.len() != self.prod.len() || pb.len() != self.prod.len() {
            bail!("product of DFT monitors with different shapes");
        }
        for (bi, (acc, n)) in self.prod.iter().enumerate() {
            let ((are, aim, na, nfa), (bre, bim, nb, nfb)) = (pa[bi], pb[bi]);
            if na != *n || nb != *n || nfa != coef.len() || nfb != coef.len() {
                bail!("product of DFT monitors with different shapes");
            }
            for (k, c) in coef.iter().enumerate() {
                if c.re == 0.0 && c.im == 0.0 {
                    continue;
                }
                let off = (k * n * 4) as u64;
                let args = Args::new().p(acc.cu_deviceptr()).p(are + off).p(aim + off).p(bre + off).p(bim + off).u(*n as u32).f(c.re as f32).f(c.im as f32);
                self.be.launch_1d(&self.k.product, *n, args)?;
            }
        }
        Ok(())
    }

    fn product_end(&mut self) -> Result<Vec<Vec<f32>>> {
        self.bind()?;
        self.be.stream.synchronize()?;
        let mut out = vec![];
        for (buf, n) in &self.prod {
            let mut v = buf.to_host_vec(&self.be.stream)?;
            v.truncate(*n);
            out.push(v);
        }
        Ok(out)
    }

    fn prepare_reverse(&mut self) -> Result<()> {
        // the dense dispersive form of the fused step keeps no per-cell lists
        if self.be.disp_dense.is_some() {
            self.rebuild_split()?;
        }
        Ok(())
    }

    fn reverse_begin(&mut self, cfg: ReverseConfig) -> Result<()> {
        self.bind()?;
        if self.be.disp_dense.is_some() {
            bail!("reverse mode needs the per-cell dispersive lists (prepare_reverse was not called)");
        }
        if self.cur.is_none() {
            bail!("no run configuration attached");
        }
        // 2-D plans: the cotangent sources may drive either field set
        self.be.plan.drive_2d_all();
        self.ensure_rev_coefs()?;
        if self.adj.is_none() {
            self.adj = Some(self.new_state()?);
        } else {
            self.zero_state(self.adj.as_ref().unwrap())?;
            self.adj.as_mut().unwrap().cur_psi = 0;
        }
        let s = &self.be.stream;
        let sized = |f: &dyn Fn(usize) -> usize| -> Result<[DeviceBuffer<f32>; 3]> { Ok([zbuf(s, f(0))?, zbuf(s, f(1))?, zbuf(s, f(2))?]) };
        let cells = |a: usize| cfg.design[a].cells();
        let lossy = self.be.ca.is_some();
        let mut sources = vec![];
        for src in &cfg.sources {
            let n = src.idx.len();
            if n == 0 {
                continue;
            }
            let group: Vec<u32> = (0..n as u32).collect();
            sources.push(SourceDev { comp: src.comp, n, idx: dbuf(s, &src.idx)?, coef: dbuf(s, &src.coef)?, group: dbuf(s, &group)?, table: dbuf(s, &src.table)? });
        }
        // chi2 / chi3 cells inside a PML are treated as linear in the reverse pass
        let mut nl = vec![];
        if let Some(np) = &self.be.plan.nonlin {
            let p = &self.be.plan;
            let (plane, nz) = (p.plane(), p.nz());
            for c in &np.comps {
                let mask: Vec<u32> = c
                    .idx
                    .iter()
                    .map(|&si| {
                        let si = si as usize;
                        let g = [si / plane - 1, (si % plane) / nz, si % nz];
                        !p.pml.iter().any(|q| (0..3).all(|a| g[a] >= q.start[a] && g[a] < q.start[a] + q.size[a])) as u32
                    })
                    .collect();
                let n = c.idx.len();
                nl.push((dbuf(s, &mask)?, zbuf(s, n)?, zbuf(s, n)?, zbuf(s, n)?));
            }
        }
        let max_shell = match (cfg.shell, self.cur) {
            (Some(ti), Some(io)) => self.ios[io].dev.as_ref().unwrap().time.get(ti).map(|t| t.total).ok_or_else(|| anyhow!("no time monitor {ti} for the shell"))?,
            _ => 1,
        };
        let cells_h = |a: usize| cfg.design_h[a].cells();
        self.rev = Some(Reverse {
            shell: cfg.shell,
            design: cfg.design,
            design_h: cfg.design_h,
            snap_h: sized(&cells_h)?,
            q1: sized(&cells_h)?,
            compq: sized(&cells_h)?,
            sources,
            snap: sized(&cells)?,
            p1: sized(&cells)?,
            comp: sized(&cells)?,
            p2: sized(&|a| if lossy { cells(a) } else { 1 })?,
            zeros: zbuf(s, max_shell)?,
            nl,
        });
        Ok(())
    }

    fn reverse_step(&mut self, n: usize) -> Result<()> {
        if self.rev.is_none() {
            bail!("reverse_begin has not been called");
        }
        // ---- forward fields: E(n+1), H(n+1/2) -> E(n) ----
        self.restore_shell(false, Some(n))?;
        self.uninject(false, n)?;
        {
            let r = self.rev.as_ref().unwrap();
            for a in 0..3 {
                let bx = r.design[a];
                if bx.is_empty() {
                    continue;
                }
                let args = box_args(Args::new().p(self.be.e[a].cu_deviceptr()).p(r.snap[a].cu_deviceptr()), bx.lo, bx.size, &self.be.plan);
                self.be.launch_1d(&self.be.k.gather, bx.cells(), args)?;
            }
        }
        // chi2 / chi3 cells: keep E(n+1) (the forward step's `prev` buffers are free)
        self.be.nl_save_launch()?;
        self.disp_reverse(false)?;
        {
            // the linear E update with -cb/ca, 1/ca, outside the PML
            self.ghost_h_current()?;
            let be = &mut self.be;
            let pml = std::mem::take(&mut be.pml);
            let disp = be.disp.take();
            be.swap_reverse_e();
            let res = be.update_e_split();
            be.swap_reverse_e();
            be.pml = pml;
            be.disp = disp;
            res?;
        }
        {
            let r = self.rev.as_ref().unwrap();
            for (c, (mask, fe, fl, _)) in self.be.nl.iter().zip(&r.nl) {
                let (ca, has) = match &self.be.ca {
                    Some(v) => (v[c.comp.axis()].cu_deviceptr(), 1u32),
                    None => (self.be.dummy.cu_deviceptr(), 0u32),
                };
                let args = Args::new()
                    .p(self.be.field_ptr(c.comp))
                    .p(c.idx.cu_deviceptr())
                    .p(c.prev.cu_deviceptr())
                    .p(c.c2.cu_deviceptr())
                    .p(c.c3.cu_deviceptr())
                    .p(ca)
                    .u(has)
                    .p(mask.cu_deviceptr())
                    .p(fe.cu_deviceptr())
                    .p(fl.cu_deviceptr())
                    .u(c.n as u32);
                self.be.launch_1d(&self.k.nl_post, c.n, args)?;
            }
        }
        self.disp_reverse(true)?;
        // ---- cotangent of E(n+1), coefficient sums ----
        self.add_sources(true, n)?;
        {
            let r = self.rev.as_ref().unwrap();
            let adj = self.adj.as_ref().unwrap();
            for a in 0..3 {
                let bx = r.design[a];
                if bx.is_empty() {
                    continue;
                }
                let (ca, has) = match &self.be.ca {
                    Some(v) => (v[a].cu_deviceptr(), 1u32),
                    None => (self.be.dummy.cu_deviceptr(), 0u32),
                };
                let args = Args::new()
                    .p(r.p1[a].cu_deviceptr())
                    .p(r.comp[a].cu_deviceptr())
                    .p(r.p2[a].cu_deviceptr())
                    .p(adj.e[a].cu_deviceptr())
                    .p(self.be.e[a].cu_deviceptr())
                    .p(r.snap[a].cu_deviceptr())
                    .p(ca)
                    .u(has);
                self.be.launch_1d(&self.k.accumulate, bx.cells(), box_args(args, bx.lo, bx.size, &self.be.plan))?;
            }
        }
        // ---- H(n+1/2) -> H(n-1/2) ----
        self.restore_shell(true, n.checked_sub(1))?;
        self.uninject(true, n)?;
        {
            // permeability designs: keep H'(n+1/2)
            let r = self.rev.as_ref().unwrap();
            for a in 0..3 {
                let bx = r.design_h[a];
                if bx.is_empty() {
                    continue;
                }
                let args = box_args(Args::new().p(self.be.h[a].cu_deviceptr()).p(r.snap_h[a].cu_deviceptr()), bx.lo, bx.size, &self.be.plan);
                self.be.launch_1d(&self.be.k.gather, bx.cells(), args)?;
            }
        }
        {
            let be = &mut self.be;
            let pml = std::mem::take(&mut be.pml);
            be.swap_reverse_h();
            let res = be.fill_ghost_e().and_then(|_| be.update_h_split());
            be.swap_reverse_h();
            be.pml = pml;
            res?;
        }
        // ---- adjoint: the forward step on the scaled adjoint fields ----
        self.add_sources(false, n)?;
        let mut adj = self.adj.take().unwrap();
        // chi2 / chi3 cells: the cotangent of E_lin drives the linear step,
        // E(n) receives dE'/dE times the cotangent directly (added after it)
        {
            let r = self.rev.as_ref().unwrap();
            for (c, (_, fe, fl, save)) in self.be.nl.iter().zip(&r.nl) {
                let args = Args::new().p(adj.e[c.comp.axis()].cu_deviceptr()).p(c.idx.cu_deviceptr()).p(fe.cu_deviceptr()).p(fl.cu_deviceptr()).p(save.cu_deviceptr()).u(c.n as u32);
                self.be.launch_1d(&self.k.nl_adj_pre, c.n, args)?;
            }
        }
        let nl = std::mem::take(&mut self.be.nl);
        let res = self.adjoint_step(&mut adj);
        self.be.nl = nl;
        if res.is_ok() {
            let r = self.rev.as_ref().unwrap();
            for (c, (_, _, _, save)) in self.be.nl.iter().zip(&r.nl) {
                let args = Args::new().p(adj.e[c.comp.axis()].cu_deviceptr()).p(c.idx.cu_deviceptr()).p(save.cu_deviceptr()).u(c.n as u32);
                self.be.launch_1d(&self.k.nl_adj_post, c.n, args)?;
            }
            // dL/d ch: q1 += b^ (H'(n+1/2) - H(n-1/2)), with b^ the adjoint of
            // H'(n+1/2) (the design cells have no magnetic loss)
            for a in 0..3 {
                let bx = r.design_h[a];
                if bx.is_empty() {
                    continue;
                }
                let d = self.be.dummy.cu_deviceptr();
                let args = Args::new()
                    .p(r.q1[a].cu_deviceptr())
                    .p(r.compq[a].cu_deviceptr())
                    .p(d)
                    .p(adj.h[a].cu_deviceptr())
                    .p(self.be.h[a].cu_deviceptr())
                    .p(r.snap_h[a].cu_deviceptr())
                    .p(d)
                    .u(0);
                self.be.launch_1d(&self.k.accumulate, bx.cells(), box_args(args, bx.lo, bx.size, &self.be.plan))?;
            }
        }
        self.adj = Some(adj);
        res
    }

    fn reverse_end(&mut self) -> Result<ReverseOut> {
        self.bind()?;
        let Some(r) = self.rev.take() else { bail!("reverse_begin has not been called") };
        self.be.stream.synchronize()?;
        let lossy = self.be.ca.is_some();
        let get = |b: &DeviceBuffer<f32>, n: usize| -> Result<Vec<f64>> {
            let v = b.to_host_vec(&self.be.stream)?;
            Ok(v[..n].iter().map(|x| *x as f64).collect())
        };
        let mut out = ReverseOut::default();
        for a in 0..3 {
            let n = r.design[a].cells();
            out.p1[a] = get(&r.p1[a], n)?;
            if lossy {
                out.p2[a] = get(&r.p2[a], n)?;
            }
            out.q1[a] = get(&r.q1[a], r.design_h[a].cells())?;
        }
        Ok(out)
    }

    fn reverse_field(&mut self, c: Component, adjoint: bool) -> Result<Vec<f32>> {
        self.bind()?;
        self.be.stream.synchronize()?;
        let b = match (&self.adj, adjoint && self.rev.is_some()) {
            (Some(a), true) => {
                if c.is_e() { &a.e[c.axis()] } else { &a.h[c.axis()] }
            }
            _ => {
                if c.is_e() { &self.be.e[c.axis()] } else { &self.be.h[c.axis()] }
            }
        };
        Ok(b.to_host_vec(&self.be.stream)?)
    }
}

/// Engine factory for device strings `"cpu"` and `"cuda:N"`.
pub fn factory(device: &str, plan: Plan) -> Result<Box<dyn Engine>> {
    if device == "cpu" {
        return lightsim_core::autodiff::engine::cpu_factory(device, plan);
    }
    let rest = device.strip_prefix("cuda").ok_or_else(|| anyhow!("unknown device '{device}' (use \"cpu\" or \"cuda:N\")"))?;
    let rest = rest.trim_start_matches(':');
    let ordinal: usize = if rest.is_empty() { 0 } else { rest.parse().map_err(|_| anyhow!("gradient engines take one GPU per device entry (\"cuda:N\"), got '{device}'"))? };
    Ok(Box::new(CudaEngine::new(plan, ordinal)?))
}
