//! Fixed-angle (broadband oblique incidence) engine on one GPU.
//!
//! The curl updates, CPML and sources run in the ordinary split kernels of a
//! [`CudaBackend`] (never the fused step); the coupling of the transformed
//! fields, the line solves along the tilt axis and the trapezoidal pole
//! currents run in the kernels of `kernels/src/fixed_angle.rs`, launched on
//! the same stream. The method, the tables and the CPU reference are in
//! `lightsim_core::engine::fixed_angle`.

use crate::{Args, CudaBackend, dbuf, zbuf};
use anyhow::{Context as _, Result, anyhow, bail};
use cuda_core::memory::memcpy_dtod_async;
use cuda_core::{DeviceBuffer, Function};
use lightsim_core::Component;
use lightsim_core::engine::fixed_angle::{ENGINE_ONLY, Edge, Tables, compile};
use lightsim_core::engine::{Backend, RawResults, Side};
use lightsim_core::plan::Plan;
use lightsim_core::Simulation;

struct FaKernels {
    pass_f: Function,
    pml_f: Function,
    corr: Function,
    pole_j: Function,
    line_h: Function,
    line_e: Function,
    pass_k: Function,
    pml_k: Function,
    poles_pre: Function,
    poles_post: Function,
}

struct LinesDev {
    lo: DeviceBuffer<f32>,
    cp: DeviceBuffer<f32>,
    idn: DeviceBuffer<f32>,
    zs: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
}

struct PoleDev {
    axis: usize,
    n: usize,
    idx: DeviceBuffer<u64>,
    matid: DeviceBuffer<u32>,
    w: DeviceBuffer<f32>,
    jw: DeviceBuffer<f32>,
    ys: DeviceBuffer<f32>,
}

struct CorrDev {
    n: usize,
    idx: DeviceBuffer<u64>,
    fac: DeviceBuffer<f32>,
    amps: Vec<f32>,
}

struct PmlFa {
    start: [usize; 3],
    size: [usize; 3],
    e_b: DeviceBuffer<f32>,
    e_a: DeviceBuffer<f32>,
    e_k: DeviceBuffer<f32>,
    h_b: DeviceBuffer<f32>,
    h_a: DeviceBuffer<f32>,
    h_k: DeviceBuffer<f32>,
    psi_e: DeviceBuffer<f32>,
    psi_h: DeviceBuffer<f32>,
}

/// One GPU running a fixed-angle simulation.
pub struct FixedAngleCuda {
    inner: CudaBackend,
    tb: Tables,
    k: FaKernels,
    hu_old: DeviceBuffer<f32>,
    hq_old: [DeviceBuffer<f32>; 3],
    e_old: [DeviceBuffer<f32>; 3],
    fbuf: DeviceBuffer<f32>,
    kbuf: DeviceBuffer<f32>,
    lines: Vec<LinesDev>,
    poles: Vec<PoleDev>,
    pole_tab: DeviceBuffer<f32>,
    gsum: DeviceBuffer<f32>,
    corr_f: Vec<CorrDev>,
    corr_k: Vec<CorrDev>,
    pml: Vec<PmlFa>,
}

fn edge(e: Edge) -> (u32, f32) {
    match e {
        Edge::Mirror(f) => (1, f),
        _ => (0, 0.0),
    }
}

impl FixedAngleCuda {
    pub fn new(sim: &Simulation, ordinal: usize) -> Result<FixedAngleCuda> {
        let (mut plan, st) = compile(sim)?;
        let tb = Tables::build(&mut plan, &st)?;
        let pml_host = plan.pml.clone();
        let len = plan.storage_len();
        let inner = CudaBackend::new_with(plan, ordinal, false)?;
        let s = inner.stream().clone();
        let f = |n: &str| inner._module.load_function(n).with_context(|| format!("kernel {n} (rebuild the PTX with scripts/build_kernels.sh)"));
        let k = FaKernels {
            pass_f: f("fa_pass_f")?,
            pml_f: f("fa_pml_f")?,
            corr: f("fa_corr")?,
            pole_j: f("fa_pole_j")?,
            line_h: f("fa_line_h")?,
            line_e: f("fa_line_e")?,
            pass_k: f("fa_pass_k")?,
            pml_k: f("fa_pml_k")?,
            poles_pre: f("fa_poles_pre")?,
            poles_post: f("fa_poles_post")?,
        };
        let mut lines = Vec::new();
        for lf in tb.lines.iter() {
            lines.push(LinesDev { lo: dbuf(&s, &lf.lo)?, cp: dbuf(&s, &lf.cp)?, idn: dbuf(&s, &lf.inv_den)?, zs: dbuf(&s, &lf.zs)?, r: dbuf(&s, &lf.r)? });
        }
        let mut poles = Vec::new();
        for p in &tb.poles {
            poles.push(PoleDev {
                axis: p.comp.axis(),
                n: p.idx.len(),
                idx: dbuf(&s, &p.idx)?,
                matid: dbuf(&s, &p.matid)?,
                w: dbuf(&s, &p.w)?,
                jw: dbuf(&s, &p.jw)?,
                ys: zbuf(&s, p.idx.len() * 2 * tb.npole.max(1))?,
            });
        }
        let pole_tab = dbuf(&s, &if tb.pole_tab.is_empty() { vec![0f32] } else { tb.pole_tab.clone() })?;
        let gsum = dbuf(&s, &if tb.gsum.is_empty() { vec![0f32] } else { tb.gsum.clone() })?;
        let corr = |c: &Vec<lightsim_core::engine::fixed_angle::Corr>| -> Result<Vec<CorrDev>> {
            c.iter().filter(|c| !c.idx.is_empty()).map(|c| Ok(CorrDev { n: c.idx.len(), idx: dbuf(&s, &c.idx)?, fac: dbuf(&s, &c.fac)?, amps: c.amps.clone() })).collect()
        };
        let corr_f = corr(&tb.corr_f)?;
        let corr_k = corr(&tb.corr_k)?;
        let mut pml = Vec::new();
        for p in &pml_host {
            pml.push(PmlFa {
                start: p.start,
                size: p.size,
                e_b: dbuf(&s, &p.e_b)?,
                e_a: dbuf(&s, &p.e_a)?,
                e_k: dbuf(&s, &p.e_kinv)?,
                h_b: dbuf(&s, &p.h_b)?,
                h_a: dbuf(&s, &p.h_a)?,
                h_k: dbuf(&s, &p.h_kinv)?,
                psi_e: zbuf(&s, p.cells())?,
                psi_h: zbuf(&s, p.cells())?,
            });
        }
        Ok(FixedAngleCuda {
            hu_old: zbuf(&s, len)?,
            hq_old: [zbuf(&s, len)?, zbuf(&s, len)?, zbuf(&s, len)?],
            e_old: [zbuf(&s, len)?, zbuf(&s, len)?, zbuf(&s, len)?],
            fbuf: zbuf(&s, len)?,
            kbuf: zbuf(&s, len)?,
            inner,
            tb,
            k,
            lines,
            poles,
            pole_tab,
            gsum,
            corr_f,
            corr_k,
            pml,
        })
    }

    fn copy(&self, dst: u64, src: u64) -> Result<()> {
        let n = self.inner.plan.storage_len();
        unsafe { memcpy_dtod_async(dst, src, n * 4, self.inner.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))
    }

    fn ca_ptr(&self, a: usize) -> (u64, u32) {
        match &self.inner.ca {
            Some(c) => (c[a].cu_deviceptr(), 1),
            None => (self.inner.dummy.cu_deviceptr(), 0),
        }
    }

    fn ch_ptr(&self, a: usize) -> (u64, u32) {
        match &self.inner.ch {
            Some(c) => (c[a].cu_deviceptr(), 1),
            None => (self.inner.dummy.cu_deviceptr(), 0),
        }
    }

    fn line_args(&self) -> (u32, u32, u32, u32, u32, u32, u32) {
        let g = self.tb.geo;
        let u = self.tb.st.u;
        let (a1, a2, nl) = g.lines(u);
        (g.n[u] as u32, g.stride[u] as u32, nl as u32, g.stride[a1] as u32, g.stride[a2] as u32, g.n[a2] as u32, g.off as u32)
    }

    fn launch_lines(&self, f: &Function, nlines: u32, args: Args) -> Result<()> {
        let b = 64u32;
        let g = nlines.div_ceil(b);
        self.inner.launch(f, (g, 1, 1), (b, 1, 1), args)
    }

    /// The Q_q update of pair 0 (q = w) or 1 (q = v), after the injections.
    fn solve_h(&self, pair: usize, step: usize) -> Result<()> {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        let (p, q, sigma, ci) = if pair == 0 { (v, w, 1.0f32, 3) } else { (w, v, -1.0f32, 2) };
        let g = tb.geo;
        let ncell = g.cells() as u32;
        let fb = self.fbuf.cu_deviceptr();
        let hu = self.inner.field_ptr(Component::h(u));
        let ho = self.hu_old.cu_deviceptr();
        let ep = self.inner.field_ptr(Component::e(p));
        let (ca, has_ca) = self.ca_ptr(p);
        let (em, ef) = edge(tb.h_lo);
        let per = self.inner.plan.periodic[q] as u32;
        self.inner.launch_1d(
            &self.k.pass_f,
            ncell as usize,
            Args::new()
                .p(fb)
                .p(hu)
                .p(ho)
                .p(ep)
                .p(self.inner.cb[p].cu_deviceptr())
                .p(ca)
                .u(has_ca)
                .p(self.inner.inv_d[q].cu_deviceptr())
                .u(ncell)
                .u(g.n[1] as u32)
                .u(g.n[2] as u32)
                .u(q as u32)
                .u(g.n[q] as u32)
                .u(g.stride[q] as u32)
                .u(per)
                .u(em)
                .f(ef)
                .f(sigma),
        )?;
        if q == w {
            for pm in &self.pml {
                let n = pm.size[0] * pm.size[1] * pm.size[2];
                self.inner.launch_1d(
                    &self.k.pml_f,
                    n,
                    Args::new()
                        .p(fb)
                        .p(hu)
                        .p(ho)
                        .p(pm.psi_e.cu_deviceptr())
                        .p(pm.e_b.cu_deviceptr())
                        .p(pm.e_a.cu_deviceptr())
                        .p(pm.e_k.cu_deviceptr())
                        .p(self.inner.inv_d[w].cu_deviceptr())
                        .u(pm.start[0] as u32)
                        .u(pm.start[1] as u32)
                        .u(pm.start[2] as u32)
                        .u(pm.size[0] as u32)
                        .u(pm.size[1] as u32)
                        .u(pm.size[2] as u32)
                        .u(w as u32)
                        .u(g.n[1] as u32)
                        .u(g.n[2] as u32)
                        .u(em)
                        .f(ef),
                )?;
            }
            for c in &self.corr_f {
                let a = 0.5 * (c.amps[step] + if step > 0 { c.amps[step - 1] } else { 0.0 });
                self.inner.launch_1d(&self.k.corr, c.n, Args::new().p(fb).p(c.idx.cu_deviceptr()).p(c.fac.cu_deviceptr()).u(c.n as u32).f(a))?;
            }
        }
        for pd in self.poles.iter().filter(|pd| pd.axis == p) {
            self.inner.launch_1d(
                &self.k.pole_j,
                pd.n,
                Args::new()
                    .p(fb)
                    .p(ep)
                    .p(pd.idx.cu_deviceptr())
                    .p(pd.matid.cu_deviceptr())
                    .p(pd.jw.cu_deviceptr())
                    .p(pd.ys.cu_deviceptr())
                    .p(self.pole_tab.cu_deviceptr())
                    .u(tb.npole as u32)
                    .u(pd.n as u32)
                    .f(sigma),
            )?;
        }
        let (n, su, nl, st1, st2, n2, off) = self.line_args();
        let (ch, has_ch) = self.ch_ptr(q);
        let lf = &self.lines[ci];
        self.launch_lines(
            &self.k.line_h,
            nl,
            Args::new()
                .p(self.inner.field_ptr(Component::h(q)))
                .p(self.hq_old[q].cu_deviceptr())
                .p(fb)
                .p(self.inner.cb[p].cu_deviceptr())
                .p(ca)
                .u(has_ca)
                .p(ch)
                .u(has_ch)
                .f(self.inner.plan.ch_scalar)
                .f(tb.s)
                .f(tb.t)
                .p(self.inner.inv_d[u].cu_deviceptr())
                .u(n)
                .u(su)
                .u(nl)
                .u(st1)
                .u(st2)
                .u(n2)
                .u(off)
                .p(lf.lo.cu_deviceptr())
                .p(lf.cp.cu_deviceptr())
                .p(lf.idn.cu_deviceptr())
                .p(lf.zs.cu_deviceptr())
                .p(lf.r.cu_deviceptr()),
        )
    }

    /// The P_p update of pair 0 (p = v) or 1 (p = w).
    fn solve_e(&self, pair: usize, step: usize) -> Result<()> {
        let tb = &self.tb;
        let (u, v, w) = (tb.st.u, tb.st.v, tb.st.w);
        let (p, q, ci) = if pair == 0 { (v, w, 0) } else { (w, v, 1) };
        let g = tb.geo;
        let ncell = g.cells() as u32;
        let kb = self.kbuf.cu_deviceptr();
        let eu = self.inner.field_ptr(Component::e(u));
        let eo = self.e_old[u].cu_deviceptr();
        let (em, ef) = edge(tb.e_hi);
        let per = self.inner.plan.periodic[p] as u32;
        self.inner.launch_1d(
            &self.k.pass_k,
            ncell as usize,
            Args::new()
                .p(kb)
                .p(eu)
                .p(eo)
                .p(self.inner.inv_p[p].cu_deviceptr())
                .u(ncell)
                .u(g.n[1] as u32)
                .u(g.n[2] as u32)
                .u(p as u32)
                .u(g.n[p] as u32)
                .u(g.stride[p] as u32)
                .u(per)
                .u(em)
                .f(ef),
        )?;
        if p == w {
            for pm in &self.pml {
                let n = pm.size[0] * pm.size[1] * pm.size[2];
                self.inner.launch_1d(
                    &self.k.pml_k,
                    n,
                    Args::new()
                        .p(kb)
                        .p(eu)
                        .p(eo)
                        .p(pm.psi_h.cu_deviceptr())
                        .p(pm.h_b.cu_deviceptr())
                        .p(pm.h_a.cu_deviceptr())
                        .p(pm.h_k.cu_deviceptr())
                        .p(self.inner.inv_p[w].cu_deviceptr())
                        .u(pm.start[0] as u32)
                        .u(pm.start[1] as u32)
                        .u(pm.start[2] as u32)
                        .u(pm.size[0] as u32)
                        .u(pm.size[1] as u32)
                        .u(pm.size[2] as u32)
                        .u(w as u32)
                        .u(g.n[1] as u32)
                        .u(g.n[2] as u32)
                        .u(g.n[w] as u32)
                        .u(em)
                        .f(ef),
                )?;
            }
            for c in &self.corr_k {
                let last = c.amps.len() - 1;
                let a = c.amps[step] + c.amps[(step + 1).min(last)];
                self.inner.launch_1d(&self.k.corr, c.n, Args::new().p(kb).p(c.idx.cu_deviceptr()).p(c.fac.cu_deviceptr()).u(c.n as u32).f(a))?;
            }
        }
        let (n, su, nl, st1, st2, n2, off) = self.line_args();
        let (ch, has_ch) = self.ch_ptr(q);
        let lf = &self.lines[ci];
        self.launch_lines(
            &self.k.line_e,
            nl,
            Args::new()
                .p(self.inner.field_ptr(Component::e(p)))
                .p(self.e_old[p].cu_deviceptr())
                .p(kb)
                .p(self.inner.cb[p].cu_deviceptr())
                .p(ch)
                .u(has_ch)
                .f(self.inner.plan.ch_scalar)
                .f(tb.s)
                .f(tb.t)
                .p(self.inner.inv_p[u].cu_deviceptr())
                .u(n)
                .u(su)
                .u(nl)
                .u(st1)
                .u(st2)
                .u(n2)
                .u(off)
                .p(lf.lo.cu_deviceptr())
                .p(lf.cp.cu_deviceptr())
                .p(lf.idn.cu_deviceptr())
                .p(lf.zs.cu_deviceptr())
                .p(lf.r.cu_deviceptr()),
        )
    }

    fn poles_pre(&self) -> Result<()> {
        let u = self.tb.st.u;
        for pd in &self.poles {
            self.inner.launch_1d(
                &self.k.poles_pre,
                pd.n,
                Args::new()
                    .p(self.inner.field_ptr(Component::e(pd.axis)))
                    .p(self.e_old[pd.axis].cu_deviceptr())
                    .p(pd.idx.cu_deviceptr())
                    .p(pd.matid.cu_deviceptr())
                    .p(pd.w.cu_deviceptr())
                    .p(pd.ys.cu_deviceptr())
                    .p(self.pole_tab.cu_deviceptr())
                    .p(self.gsum.cu_deviceptr())
                    .u(self.tb.npole as u32)
                    .u(pd.n as u32)
                    .u((pd.axis == u) as u32),
            )?;
        }
        Ok(())
    }

    fn poles_post(&self) -> Result<()> {
        for pd in &self.poles {
            self.inner.launch_1d(
                &self.k.poles_post,
                pd.n,
                Args::new()
                    .p(self.inner.field_ptr(Component::e(pd.axis)))
                    .p(self.e_old[pd.axis].cu_deviceptr())
                    .p(pd.idx.cu_deviceptr())
                    .p(pd.matid.cu_deviceptr())
                    .p(pd.ys.cu_deviceptr())
                    .p(self.pole_tab.cu_deviceptr())
                    .u(self.tb.npole as u32)
                    .u(pd.n as u32),
            )?;
        }
        Ok(())
    }
}

impl Backend for FixedAngleCuda {
    fn name(&self) -> String {
        format!("{}, fixed angle {:.1} deg", self.inner.name(), self.tb.st.angle.to_degrees())
    }
    fn plan(&self) -> &Plan {
        self.inner.plan()
    }
    fn fill_ghost_e(&mut self) -> Result<()> {
        self.inner.fill_ghost_e()
    }
    fn fill_ghost_h(&mut self) -> Result<()> {
        self.inner.fill_ghost_h()
    }
    fn update_h(&mut self) -> Result<()> {
        let (u, v, w) = (self.tb.st.u, self.tb.st.v, self.tb.st.w);
        self.copy(self.hu_old.cu_deviceptr(), self.inner.field_ptr(Component::h(u)))?;
        self.copy(self.hq_old[v].cu_deviceptr(), self.inner.field_ptr(Component::h(v)))?;
        self.copy(self.hq_old[w].cu_deviceptr(), self.inner.field_ptr(Component::h(w)))?;
        self.inner.update_h()
    }
    fn inject_h(&mut self, step: usize) -> Result<()> {
        self.inner.inject_h(step)?;
        if self.tb.s != 0.0 || !self.tb.poles.is_empty() {
            for pair in 0..2 {
                self.solve_h(pair, step)?;
            }
        }
        Ok(())
    }
    fn update_e(&mut self) -> Result<()> {
        for a in 0..3 {
            self.copy(self.e_old[a].cu_deviceptr(), self.inner.field_ptr(Component::e(a)))?;
        }
        self.inner.update_e()
    }
    fn inject_e(&mut self, step: usize) -> Result<()> {
        self.inner.inject_e(step)?;
        self.poles_pre()?;
        for pair in 0..2 {
            self.solve_e(pair, step)?;
        }
        self.poles_post()
    }
    fn record(&mut self, step: usize) -> Result<()> {
        self.inner.record(step)
    }
    fn energy(&mut self) -> Result<f64> {
        self.inner.energy()
    }
    fn synchronize(&mut self) -> Result<()> {
        self.inner.synchronize()
    }
    fn download(&mut self) -> Result<RawResults> {
        let mut raw = self.inner.download()?;
        self.tb.physical_dft(self.inner.plan(), &mut raw);
        Ok(raw)
    }
    fn read_plane(&mut self, _comp: Component, _side: Side, _out: &mut [f32]) -> Result<()> {
        bail!("{ENGINE_ONLY}")
    }
    fn write_ghost(&mut self, _comp: Component, _side: Side, _data: &[f32]) -> Result<()> {
        bail!("{ENGINE_ONLY}")
    }
    fn field(&mut self, comp: Component) -> Result<Vec<f32>> {
        self.inner.field(comp)
    }
}

/// The backend of a fixed-angle simulation on the GPUs `gpus` (one only).
pub fn backend(sim: &Simulation, gpus: &[usize]) -> Result<Box<dyn Backend>> {
    if gpus.len() != 1 {
        bail!("{ENGINE_ONLY} (requested {} GPUs)", gpus.len());
    }
    Ok(Box::new(FixedAngleCuda::new(sim, gpus[0])?))
}
