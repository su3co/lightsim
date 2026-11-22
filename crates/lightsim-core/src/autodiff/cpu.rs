//! CPU implementation of [`Engine`] on top of [`CpuBackend`]: the reference
//! for the GPU engine and what the tests run on machines without a GPU.

use super::cell;
use super::design::IndexBox;
use super::engine::{AdjointSource, Engine, IoSet, ReverseConfig, ReverseOut, forward_step};
use crate::engine::cpu::CpuBackend;
use crate::engine::{Backend, RawDft, RawTime};
use crate::grid::Component;
use crate::plan::{CompBox, GhostRule, Injection, Plan, PmlSlab};
use anyhow::{Result, bail};
use num_complex::Complex64;

/// Recursion state of the transposed CPML, per PML slab: `phi` on the E
/// cells (components b, c of the slab's cyclic axes) and `theta` on the H
/// cells (b, c).
///
/// The forward CPML convolves the *difference* of the other field with the
/// stretching coefficients of the cell being updated. Its transpose
/// convolves the *field* with the coefficients of the cell it comes from
/// and differences the result: in the scaled adjoint variables, for a slab
/// normal to `a` with cyclic components (b, c),
///
/// ```text
/// phi_b <- bE phi_b + e_b,        w_b = (1/kappaE - 1) e_b + aE phi_b     (E cells)
/// h_c -= ch d_a(w_b),             h_b += ch d_a(w_c)
/// theta_b <- bH theta_b + h_b,    v_b = (1/kappaH - 1) h_b + aH theta_b   (H cells)
/// e_c += cb d_a(v_b),             e_b -= cb d_a(v_c)
/// ```
///
/// with `d_a` the same forward / backward differences as in the forward
/// kernels and `w`, `v` zero outside the slab. Where the profile is
/// constant the two forms coincide.
pub struct AdjointPml {
    st: Vec<[Vec<f32>; 4]>,
}

impl AdjointPml {
    pub fn new(plan: &Plan) -> AdjointPml {
        AdjointPml { st: plan.pml.iter().map(|p| [0, 1, 2, 3].map(|_| vec![0f32; p.cells()])).collect() }
    }
}

/// Slab-local offset of interior cell `g` (must be inside the slab).
fn slab_offset(p: &PmlSlab, g: [usize; 3]) -> usize {
    ((g[0] - p.start[0]) * p.size[1] + (g[1] - p.start[1])) * p.size[2] + (g[2] - p.start[2])
}

/// Transposed E-side CPML: acts in the adjoint "H update" (see [`AdjointPml`]).
fn pml_h_transposed(adj: &mut CpuBackend, slabs: &[PmlSlab], st: &mut AdjointPml) {
    let (plan, e, h) = adj.plan_and_fields();
    let (ny, nz) = (plan.ny(), plan.nz());
    let n = [plan.nxl, ny, nz];
    let ch_arr = plan.ch.as_ref();
    for (si, p) in slabs.iter().enumerate() {
        let a = p.axis;
        let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
        let (phi_b, rest) = st.st[si].split_at_mut(1);
        let (phi_b, phi_c) = (&mut phi_b[0], &mut rest[0]);
        let sidx = |g: [usize; 3]| ((g[0] + 1) * ny + g[1]) * nz + g[2];
        // phi <- bE phi + e on the slab's E cells
        for i in p.start[0]..p.start[0] + p.size[0] {
            for j in p.start[1]..p.start[1] + p.size[1] {
                for k in p.start[2]..p.start[2] + p.size[2] {
                    let g = [i, j, k];
                    let (t, s, q) = (slab_offset(p, g), sidx(g), g[a] - p.start[a]);
                    phi_b[t] = p.e_b[q] * phi_b[t] + e[bax][s];
                    phi_c[t] = p.e_b[q] * phi_c[t] + e[cax][s];
                }
            }
        }
        // H cells of the slab and the layer below it
        let lo = p.start[a].saturating_sub(1);
        let hi = p.start[a] + p.size[a];
        let mut r = [(p.start[0], p.start[0] + p.size[0]), (p.start[1], p.start[1] + p.size[1]), (p.start[2], p.start[2] + p.size[2])];
        r[a] = (lo, hi);
        let w = |g: [usize; 3], comp: usize, phi: &Vec<f32>| -> f32 {
            if g[a] < p.start[a] || g[a] >= p.start[a] + p.size[a] {
                return 0.0;
            }
            let q = g[a] - p.start[a];
            (p.e_kinv[q] - 1.0) * e[comp][sidx(g)] + p.e_a[q] * phi[slab_offset(p, g)]
        };
        for i in r[0].0..r[0].1 {
            for j in r[1].0..r[1].1 {
                for k in r[2].0..r[2].1 {
                    let g = [i, j, k];
                    let mut gn = g;
                    gn[a] += 1;
                    let next = gn[a] < n[a];
                    let s = sidx(g);
                    let ip = plan.inv_p[a][g[a]];
                    let dwb = (if next { w(gn, bax, phi_b) } else { 0.0 }) - w(g, bax, phi_b);
                    let dwc = (if next { w(gn, cax, phi_c) } else { 0.0 }) - w(g, cax, phi_c);
                    let ch = |c: usize| ch_arr.map(|v| v[c][s]).unwrap_or(plan.ch_scalar);
                    h[cax][s] -= ch(cax) * dwb * ip;
                    h[bax][s] += ch(bax) * dwc * ip;
                }
            }
        }
    }
}

/// Transposed H-side CPML: acts in the adjoint "E update" (see [`AdjointPml`]).
fn pml_e_transposed(adj: &mut CpuBackend, slabs: &[PmlSlab], st: &mut AdjointPml) {
    let (plan, e, h) = adj.plan_and_fields();
    let (ny, nz) = (plan.ny(), plan.nz());
    let n = [plan.nxl, ny, nz];
    let cb = &plan.cb;
    for (si, p) in slabs.iter().enumerate() {
        let a = p.axis;
        let (bax, cax) = ((a + 1) % 3, (a + 2) % 3);
        let (_, rest) = st.st[si].split_at_mut(2);
        let (th_b, th_c) = rest.split_at_mut(1);
        let (th_b, th_c) = (&mut th_b[0], &mut th_c[0]);
        let sidx = |g: [usize; 3]| ((g[0] + 1) * ny + g[1]) * nz + g[2];
        // theta <- bH theta + h on the slab's H cells
        for i in p.start[0]..p.start[0] + p.size[0] {
            for j in p.start[1]..p.start[1] + p.size[1] {
                for k in p.start[2]..p.start[2] + p.size[2] {
                    let g = [i, j, k];
                    let (t, s, q) = (slab_offset(p, g), sidx(g), g[a] - p.start[a]);
                    th_b[t] = p.h_b[q] * th_b[t] + h[bax][s];
                    th_c[t] = p.h_b[q] * th_c[t] + h[cax][s];
                }
            }
        }
        // E cells of the slab and the layer above it
        let lo = p.start[a];
        let hi = (p.start[a] + p.size[a] + 1).min(n[a]);
        let mut r = [(p.start[0], p.start[0] + p.size[0]), (p.start[1], p.start[1] + p.size[1]), (p.start[2], p.start[2] + p.size[2])];
        r[a] = (lo, hi);
        let v = |g: [usize; 3], comp: usize, th: &Vec<f32>| -> f32 {
            if g[a] < p.start[a] || g[a] >= p.start[a] + p.size[a] {
                return 0.0;
            }
            let q = g[a] - p.start[a];
            (p.h_kinv[q] - 1.0) * h[comp][sidx(g)] + p.h_a[q] * th[slab_offset(p, g)]
        };
        for i in r[0].0..r[0].1 {
            for j in r[1].0..r[1].1 {
                for k in r[2].0..r[2].1 {
                    let g = [i, j, k];
                    let s = sidx(g);
                    let id = plan.inv_d[a][g[a]];
                    let (mut dvb, mut dvc) = (v(g, bax, th_b), v(g, cax, th_c));
                    if g[a] > 0 {
                        let mut gp = g;
                        gp[a] -= 1;
                        dvb -= v(gp, bax, th_b);
                        dvc -= v(gp, cax, th_c);
                    }
                    e[cax][s] += cb[cax][s] * dvb * id;
                    e[bax][s] -= cb[bax][s] * dvc * id;
                }
            }
        }
    }
}

/// One transposed leapfrog step on the scaled adjoint fields held by `adj`
/// (a backend on the forward materials, without sources and monitors): the
/// forward update outside the CPML layers, the transposed CPML inside.
///
/// Magnetic loss (`H' = da H - ch curl E`): the forward step scales H before
/// its curl update; the transposed step scales it after the E update
/// (`b~ = b - ch curl a`, `a' = ca a + cb curl b~`, `b' = da b~`).
pub fn adjoint_step(adj: &mut CpuBackend, st: &mut AdjointPml) -> Result<()> {
    adj.fill_ghost_e()?;
    let pml = std::mem::take(&mut adj.plan_mut().pml);
    let hloss = adj.plan_mut().hloss.take();
    let r = (|| -> Result<()> {
        adj.update_h()?;
        pml_h_transposed(adj, &pml, st);
        adj.fill_ghost_h()?;
        adj.update_e()?;
        pml_e_transposed(adj, &pml, st);
        adj.finish_e()
    })();
    if let Some(hl) = &hloss {
        for (a, l) in hl.iter().enumerate() {
            let h = adj.field_mut(Component::h(a));
            for (&s, &d) in l.idx.iter().zip(&l.da) {
                h[s as usize] *= d;
            }
        }
    }
    adj.plan_mut().pml = pml;
    adj.plan_mut().hloss = hloss;
    r
}

/// State of a reverse pass.
struct Reverse {
    /// The scaled adjoint fields: a second backend on the same materials,
    /// without sources and monitors.
    adj: CpuBackend,
    adj_pml: AdjointPml,
    cfg: ReverseConfig,
    /// E' (before the source injection) over the design boxes.
    snap: [Vec<f32>; 3],
    /// H'(n+1/2) - H(n-1/2) over the permeability design boxes.
    snap_h: [Vec<f32>; 3],
    out: ReverseOut,
    /// Coefficients of the reversed updates: -cb/ca, 1/ca, -ch.
    cb: [Vec<f32>; 3],
    ca: Option<[Vec<f32>; 3]>,
    ch: Option<[Vec<f32>; 3]>,
    /// Reversed magnetic-loss lists (1/da), with `ch` divided by da.
    hloss: Option<[crate::plan::HLoss; 3]>,
    /// chi2/chi3 cells (per nonlinear component): d E(n+1) / d E(n) and
    /// d E(n+1) / d E_lin of the step being undone, and the direct part of
    /// the adjoint, F_e a(n+1), added after the linear transposed update.
    nl_fe: Vec<Vec<f32>>,
    nl_fl: Vec<Vec<f32>>,
    nl_a: Vec<Vec<f32>>,
    /// Nonlinear cells outside the PML (inside, the reconstructed forward
    /// fields are not meaningful and the cell is treated as linear).
    nl_in: Vec<Vec<bool>>,
}

pub struct CpuEngine {
    be: CpuBackend,
    ios: Vec<IoSet>,
    cur: Option<usize>,
    dft: Vec<Vec<RawDft>>,
    time: Vec<Vec<RawTime>>,
    prod: Vec<Vec<f64>>,
    rev: Option<Reverse>,
}

/// `f[idx] += scale * coef * amp(step)` for a real or complex-coefficient
/// injection (real part: the engine has no Bloch support).
fn inject(f: &mut [f32], j: &Injection, step: usize, scale: f32) {
    let row = j.row(step);
    match &j.coef_im {
        None => {
            for e in 0..j.idx.len() {
                let g = if j.grouped() { j.group[e] as usize } else { 0 };
                f[j.idx[e] as usize] += scale * (j.coef[e] * j.amps[row + g]);
            }
        }
        Some(ci) => {
            for e in 0..j.idx.len() {
                let g = if j.grouped() { j.group[e] as usize } else { 0 };
                f[j.idx[e] as usize] += scale * (j.coef[e] * j.amps[row + 2 * g] + ci[e] * j.amps[row + 2 * g + 1]);
            }
        }
    }
}

fn box_of(b: &CompBox) -> IndexBox {
    IndexBox { lo: b.lstart, size: b.lsize }
}

impl CpuEngine {
    pub fn new(mut plan: Plan) -> Result<CpuEngine> {
        if plan.bloch {
            bail!("gradient engines do not support Bloch (complex-field) boundaries");
        }
        if matches!(plan.ghost_e_hi, GhostRule::Neighbor) || matches!(plan.ghost_h_lo, GhostRule::Neighbor) {
            bail!("gradient engines run on a single slab");
        }
        let _ = IoSet::take(&mut plan);
        Ok(CpuEngine { be: CpuBackend::new(plan)?, ios: vec![], cur: None, dft: vec![], time: vec![], prod: vec![], rev: None })
    }

    /// The underlying backend (tests).
    pub fn backend(&mut self) -> &mut CpuBackend {
        &mut self.be
    }

    fn attach(&mut self, io: usize) -> Result<()> {
        if io >= self.ios.len() {
            bail!("unknown run configuration {io}");
        }
        if self.cur == Some(io) {
            return Ok(());
        }
        if let Some(c) = self.cur {
            self.dft[c] = self.be.dft_data().clone();
            self.time[c] = self.be.time_data().clone();
            self.ios[c].swap(self.be.plan_mut());
        }
        self.ios[io].swap(self.be.plan_mut());
        self.cur = Some(io);
        Ok(())
    }

    /// Storage offsets of box `bx`.
    fn storage(&self, bx: &IndexBox) -> impl Iterator<Item = usize> + use<> {
        let (ny, nz) = (self.be.plan().ny(), self.be.plan().nz());
        let b = *bx;
        (0..b.cells()).map(move |t| b.storage(t, ny, nz))
    }

    /// E(n+1) -> E(n) outside the PML: the linear update with -cb/ca, 1/ca.
    fn reverse_e_main(&mut self) -> Result<()> {
        let r = self.rev.as_mut().unwrap();
        let plan = self.be.plan_mut();
        let pml = std::mem::take(&mut plan.pml);
        let disp = plan.disp.take();
        let nonlin = plan.nonlin.take();
        std::mem::swap(&mut plan.cb, &mut r.cb);
        std::mem::swap(&mut plan.ca, &mut r.ca);
        let res = self.be.fill_ghost_h().and_then(|_| self.be.update_e());
        let plan = self.be.plan_mut();
        std::mem::swap(&mut plan.cb, &mut r.cb);
        std::mem::swap(&mut plan.ca, &mut r.ca);
        plan.pml = pml;
        plan.disp = disp;
        plan.nonlin = nonlin;
        res
    }

    /// H(n+1/2) -> H(n-1/2) outside the PML: the update with -ch.
    fn reverse_h_main(&mut self) -> Result<()> {
        let r = self.rev.as_mut().unwrap();
        let plan = self.be.plan_mut();
        let pml = std::mem::take(&mut plan.pml);
        std::mem::swap(&mut plan.ch, &mut r.ch);
        std::mem::swap(&mut plan.hloss, &mut r.hloss);
        plan.ch_scalar = -plan.ch_scalar;
        let res = self.be.fill_ghost_e().and_then(|_| self.be.update_h());
        let plan = self.be.plan_mut();
        std::mem::swap(&mut plan.ch, &mut r.ch);
        std::mem::swap(&mut plan.hloss, &mut r.hloss);
        plan.ch_scalar = -plan.ch_scalar;
        plan.pml = pml;
        res
    }

    /// Overwrite the shell boxes of E (`e`) or H from sample `sample` of the
    /// shell recording (`None`: the initial zero state).
    fn restore_shell(&mut self, e: bool, sample: Option<usize>) {
        let Some(ti) = self.rev.as_ref().unwrap().cfg.shell else { return };
        let boxes: Vec<CompBox> = self.be.plan().time[ti].boxes.clone();
        for (bi, b) in boxes.iter().enumerate() {
            if b.comp.is_e() != e || !b.is_local() {
                continue;
            }
            let n = b.lcells();
            let row: Vec<f32> = match sample {
                Some(s) => self.be.time_data()[ti].boxes[bi][s * n..(s + 1) * n].to_vec(),
                None => vec![0.0; n],
            };
            let bx = box_of(b);
            let idx: Vec<usize> = self.storage(&bx).collect();
            let f = self.be.field_mut(b.comp);
            for (s, v) in idx.into_iter().zip(row) {
                f[s] = v;
            }
        }
    }
}

impl Engine for CpuEngine {
    fn name(&self) -> String {
        self.be.name()
    }
    fn plan(&self) -> &Plan {
        self.be.plan()
    }
    fn add_io(&mut self, io: IoSet) -> Result<usize> {
        self.dft.push(io.dft.iter().map(|d| RawDft { boxes: d.boxes.iter().map(|_| (vec![], vec![])).collect() }).collect());
        self.time.push(vec![]);
        self.ios.push(io);
        Ok(self.ios.len() - 1)
    }

    fn write_h_coefficients(&mut self, a: usize, bx: IndexBox, ch: &[f32]) -> Result<()> {
        let idx: Vec<usize> = self.storage(&bx).collect();
        let Some(c) = self.be.plan_mut().ch.as_mut() else { anyhow::bail!("the plan has no per-cell H coefficients") };
        for (t, &s) in idx.iter().enumerate() {
            c[a][s] = ch[t];
        }
        Ok(())
    }

    fn write_coefficients(&mut self, a: usize, bx: IndexBox, cb: &[f32], ca: Option<&[f32]>) -> Result<()> {
        let idx: Vec<usize> = self.storage(&bx).collect();
        let plan = self.be.plan_mut();
        for (t, &s) in idx.iter().enumerate() {
            plan.cb[a][s] = cb[t];
        }
        if let (Some(c), Some(v)) = (plan.ca.as_mut(), ca) {
            for (t, &s) in idx.iter().enumerate() {
                c[a][s] = v[t];
            }
        }
        Ok(())
    }

    fn on_thread_pool(&self) -> bool {
        true
    }
    fn begin(&mut self, io: usize) -> Result<()> {
        self.rev = None;
        self.attach(io)?;
        // 2-D plans: the field sets this configuration's sources drive
        let plan = self.be.plan_mut();
        if let Some(td) = plan.twod.as_mut() {
            td.active = [false, false];
        }
        plan.drive_2d();
        self.be.reset()
    }
    fn step(&mut self, n: usize) -> Result<()> {
        forward_step(&mut self.be, n)
    }
    fn energy(&mut self) -> Result<f64> {
        self.be.energy()
    }

    fn read_dft(&mut self, io: usize, monitor: usize) -> Result<RawDft> {
        Ok(if self.cur == Some(io) { self.be.dft_data()[monitor].clone() } else { self.dft[io][monitor].clone() })
    }
    fn write_dft(&mut self, io: usize, monitor: usize, data: &RawDft) -> Result<()> {
        if self.cur == Some(io) {
            self.be.dft_data_mut()[monitor] = data.clone();
        } else {
            self.dft[io][monitor] = data.clone();
        }
        Ok(())
    }
    fn read_time(&mut self, io: usize, monitor: usize) -> Result<RawTime> {
        Ok(if self.cur == Some(io) { self.be.time_data()[monitor].clone() } else { self.time[io][monitor].clone() })
    }

    fn product_begin(&mut self, like: (usize, usize)) -> Result<()> {
        let d = self.read_dft(like.0, like.1)?;
        let nf = if self.cur == Some(like.0) { self.be.plan().dft[like.1].freqs.len() } else { self.ios[like.0].dft[like.1].freqs.len() };
        self.prod = d.boxes.iter().map(|(re, _)| vec![0.0; re.len() / nf.max(1)]).collect();
        Ok(())
    }
    fn product_add(&mut self, a: (usize, usize), b: (usize, usize), coef: &[Complex64]) -> Result<()> {
        let (da, db) = (self.read_dft(a.0, a.1)?, self.read_dft(b.0, b.1)?);
        for (bi, acc) in self.prod.iter_mut().enumerate() {
            let n = acc.len();
            let ((are, aim), (bre, bim)) = (&da.boxes[bi], &db.boxes[bi]);
            if are.len() != n * coef.len() || bre.len() != n * coef.len() {
                bail!("product of DFT monitors with different shapes");
            }
            for (k, c) in coef.iter().enumerate() {
                if c.re == 0.0 && c.im == 0.0 {
                    continue;
                }
                let (cr, ci) = (c.re as f32, c.im as f32);
                for t in 0..n {
                    let o = k * n + t;
                    let pr = are[o] * bre[o] - aim[o] * bim[o];
                    let pi = are[o] * bim[o] + aim[o] * bre[o];
                    acc[t] += (cr * pr - ci * pi) as f64;
                }
            }
        }
        Ok(())
    }
    fn product_end(&mut self) -> Result<Vec<Vec<f32>>> {
        Ok(std::mem::take(&mut self.prod).into_iter().map(|v| v.into_iter().map(|x| x as f32).collect()).collect())
    }

    fn reverse_begin(&mut self, cfg: ReverseConfig) -> Result<()> {
        let plan = self.be.plan();
        if plan.tensor.is_some() {
            bail!("reverse mode does not support off-diagonal permittivity");
        }
        // the adjoint fields see the same materials, no sources, no monitors
        let mut ap = plan.clone();
        let _ = IoSet::take(&mut ap);
        // 2-D plans: the cotangent sources may drive either field set
        ap.drive_2d_all();
        // nonlinear cells: the adjoint is linear, with per-step factors from
        // the forward state applied around the linear update (below)
        ap.nonlin = None;
        let len = plan.storage_len();
        let cb = [0, 1, 2].map(|a| (0..len).map(|s| -plan.cb[a][s] / plan.ca.as_ref().map(|c| c[a][s]).unwrap_or(1.0)).collect::<Vec<f32>>());
        let ca = plan.ca.as_ref().map(|c| [0, 1, 2].map(|a| c[a].iter().map(|v| 1.0 / v).collect::<Vec<f32>>()));
        let mut ch = plan.ch.as_ref().map(|c| [0, 1, 2].map(|a| c[a].iter().map(|v| -v).collect::<Vec<f32>>()));
        let mut hloss = plan.hloss.clone();
        if let (Some(c), Some(h)) = (ch.as_mut(), hloss.as_mut()) {
            crate::engine::cpu::reverse_hloss(c, h);
        }
        let lossy = plan.ca.is_some();
        let snap = [0, 1, 2].map(|a| vec![0f32; cfg.design[a].cells()]);
        let snap_h = [0, 1, 2].map(|a| vec![0f32; cfg.design_h[a].cells()]);
        let out = ReverseOut {
            p1: [0, 1, 2].map(|a| vec![0f64; cfg.design[a].cells()]),
            p2: [0, 1, 2].map(|a| vec![0f64; if lossy { cfg.design[a].cells() } else { 0 }]),
            q1: [0, 1, 2].map(|a| vec![0f64; cfg.design_h[a].cells()]),
        };
        let nl: Vec<Vec<f32>> = plan.nonlin.as_ref().map(|n| n.comps.iter().map(|c| vec![0f32; c.idx.len()]).collect()).unwrap_or_default();
        let plane = plan.plane();
        let nzz = plan.nz();
        let in_pml = |s: u64| -> bool {
            let s = s as usize;
            let g = [s / plane - 1, (s % plane) / nzz, s % nzz];
            plan.pml.iter().any(|p| (0..3).all(|a| g[a] >= p.start[a] && g[a] < p.start[a] + p.size[a]))
        };
        let nl_in: Vec<Vec<bool>> = plan.nonlin.as_ref().map(|n| n.comps.iter().map(|c| c.idx.iter().map(|&s| !in_pml(s)).collect()).collect()).unwrap_or_default();
        let adj_pml = AdjointPml::new(&ap);
        self.rev = Some(Reverse { adj: CpuBackend::new(ap)?, adj_pml, cfg, snap, snap_h, out, cb, ca, ch, hloss, nl_fe: nl.clone(), nl_fl: nl.clone(), nl_a: nl, nl_in });
        self.be.plan_mut().drive_2d_all();
        Ok(())
    }

    fn reverse_step(&mut self, n: usize) -> Result<()> {
        if self.rev.is_none() {
            bail!("reverse_begin has not been called");
        }
        let (ny, nz) = (self.be.plan().ny(), self.be.plan().nz());
        // H(n+1/2) in the shell (needed by the E reversal next to the PML)
        self.restore_shell(false, Some(n));
        // E(n+1) minus the sources of step n
        let inj = std::mem::take(&mut self.be.plan_mut().inj_e);
        for j in &inj {
            inject(self.be.field_mut(j.comp), j, n, -1.0);
        }
        self.be.plan_mut().inj_e = inj;
        {
            let r = self.rev.as_mut().unwrap();
            for a in 0..3 {
                let bx = r.cfg.design[a];
                let f = self.be.field_ref(Component::e(a));
                for (t, v) in r.snap[a].iter_mut().enumerate() {
                    *v = f[bx.storage(t, ny, nz)];
                }
            }
        }
        // chi2/chi3 cells: keep E(n+1) (before the dispersive correction is undone)
        let nonlin = self.be.plan().nonlin.clone();
        let mut nl_e1: Vec<Vec<f32>> = vec![];
        if let Some(nl) = &nonlin {
            for c in &nl.comps {
                let f = self.be.field_ref(c.comp);
                nl_e1.push(c.idx.iter().map(|&s| f[s as usize]).collect());
            }
        }
        // dispersive cells: E(n+1) + scale D (see `cell::pole_reverse_d`)
        let disp = self.be.plan().disp.clone();
        if let Some(d) = &disp {
            for (ci, dc) in d.comps.iter().enumerate() {
                let nn = dc.idx.len();
                let add: Vec<f32> = {
                    let (p, pp, _) = &self.be.disp_state_mut()[ci];
                    (0..nn)
                        .map(|t| {
                            let m = dc.matid[t] as usize;
                            let mut acc = 0f32;
                            for q in 0..d.npole {
                                let c = &d.coefs[(m * d.npole + q) * 3..(m * d.npole + q) * 3 + 3];
                                acc += cell::pole_reverse_d(p[q * nn + t], pp[q * nn + t], [c[0], c[1], c[2]]);
                            }
                            dc.scale[t] * acc
                        })
                        .collect()
                };
                let f = self.be.field_mut(dc.comp);
                for t in 0..nn {
                    f[dc.idx[t] as usize] += add[t];
                }
            }
        }
        self.reverse_e_main()?;
        if let Some(nl) = &nonlin {
            // the linear reversal returned g = (E_lin' - cb curl) / ca with
            // E_lin' = E(n+1) in place of the linear update: solve the cell
            // equation for E(n) and keep the partial derivatives of the step
            let plan = self.be.plan();
            let cas: Vec<Vec<f32>> = nl.comps.iter().map(|c| c.idx.iter().map(|&s| plan.ca.as_ref().map(|v| v[c.comp.axis()][s as usize]).unwrap_or(1.0)).collect()).collect();
            let r = self.rev.as_mut().unwrap();
            for (ci, c) in nl.comps.iter().enumerate() {
                let f = self.be.field_mut(c.comp);
                for (t, &s) in c.idx.iter().enumerate() {
                    if !r.nl_in[ci][t] {
                        (r.nl_fe[ci][t], r.nl_fl[ci][t]) = (0.0, 1.0);
                        continue;
                    }
                    let e1 = nl_e1[ci][t];
                    let e = cell::nl_unsolve(e1, f[s as usize], cas[ci][t], c.c2[t], c.c3[t]);
                    f[s as usize] = e;
                    (r.nl_fe[ci][t], r.nl_fl[ci][t]) = cell::nl_factors(e, e1, c.c2[t], c.c3[t]);
                }
            }
        }
        // dispersive cells: E(n) from the reversal's output, then the poles
        if let Some(d) = &disp {
            for (ci, dc) in d.comps.iter().enumerate() {
                let nn = dc.idx.len();
                let ax = dc.comp.axis();
                let cas: Vec<f32> = {
                    let plan = self.be.plan();
                    dc.idx.iter().map(|&s| plan.ca.as_ref().map(|v| v[ax][s as usize]).unwrap_or(1.0)).collect()
                };
                let coef = |t: usize, q: usize| -> [f32; 3] {
                    let o = (dc.matid[t] as usize * d.npole + q) * 3;
                    [d.coefs[o], d.coefs[o + 1], d.coefs[o + 2]]
                };
                let e: Vec<f32> = {
                    let f = self.be.field_mut(dc.comp);
                    (0..nn)
                        .map(|t| {
                            let kappa: f32 = (0..d.npole).map(|q| cell::pole_reverse_kappa(coef(t, q))).sum();
                            let s = dc.idx[t] as usize;
                            f[s] /= 1.0 - dc.scale[t] / cas[t] * kappa;
                            f[s]
                        })
                        .collect()
                };
                let (p, pp, _) = &mut self.be.disp_state_mut()[ci];
                for t in 0..nn {
                    for q in 0..d.npole {
                        let o = q * nn + t;
                        (p[o], pp[o]) = cell::pole_reverse_state(p[o], pp[o], e[t], coef(t, q));
                    }
                }
            }
        }
        // cotangent of E(n+1), then the coefficient sums
        {
            let r = self.rev.as_mut().unwrap();
            for s in r.cfg.sources.iter().filter(|s| s.comp.is_e()) {
                add_source(r.adj.field_mut(s.comp), s, n);
            }
            let plan = self.be.plan();
            for a in 0..3 {
                let bx = r.cfg.design[a];
                let lam = r.adj.field_ref(Component::e(a));
                let f = self.be.field_ref(Component::e(a));
                let ca = plan.ca.as_ref().map(|c| &c[a]);
                for t in 0..bx.cells() {
                    let s = bx.storage(t, ny, nz);
                    let cav = ca.map(|c| c[s]).unwrap_or(1.0);
                    r.out.p1[a][t] += (lam[s] * (r.snap[a][t] - cav * f[s])) as f64;
                    if let Some(p2) = r.out.p2[a].get_mut(t) {
                        *p2 += (lam[s] * f[s]) as f64;
                    }
                }
            }
        }
        // E(n) in the shell, then H(n+1/2) -> H(n-1/2)
        self.restore_shell(true, n.checked_sub(1));
        let inj = std::mem::take(&mut self.be.plan_mut().inj_h);
        for j in &inj {
            inject(self.be.field_mut(j.comp), j, n, -1.0);
        }
        self.be.plan_mut().inj_h = inj;
        // permeability designs: H'(n+1/2) - H(n-1/2) over their boxes
        {
            let r = self.rev.as_mut().unwrap();
            for a in 0..3 {
                let bx = r.cfg.design_h[a];
                let f = self.be.field_ref(Component::h(a));
                for t in 0..bx.cells() {
                    r.snap_h[a][t] = f[bx.storage(t, ny, nz)];
                }
            }
        }
        self.reverse_h_main()?;
        {
            let r = self.rev.as_mut().unwrap();
            for a in 0..3 {
                let bx = r.cfg.design_h[a];
                let f = self.be.field_ref(Component::h(a));
                for t in 0..bx.cells() {
                    r.snap_h[a][t] -= f[bx.storage(t, ny, nz)];
                }
            }
        }
        // the adjoint step: the forward step on the scaled adjoint fields,
        // with the transposed CPML in the layers
        let r = self.rev.as_mut().unwrap();
        for s in r.cfg.sources.iter().filter(|s| !s.comp.is_e()) {
            add_source(r.adj.field_mut(s.comp), s, n);
        }
        if let Some(nl) = &nonlin {
            // E(n+1) = F(E(n), E_lin): the adjoint of E_lin, F_lin a(n+1),
            // drives the linear transposed update; E(n) receives F_e a(n+1)
            for (ci, c) in nl.comps.iter().enumerate() {
                let lam = r.adj.field_mut(c.comp);
                for (t, &s) in c.idx.iter().enumerate() {
                    let s = s as usize;
                    r.nl_a[ci][t] = r.nl_fe[ci][t] * lam[s];
                    lam[s] *= r.nl_fl[ci][t];
                }
            }
        }
        adjoint_step(&mut r.adj, &mut r.adj_pml)?;
        // dL/d ch: the adjoint of H'(n+1/2) (no magnetic loss on design cells)
        for a in 0..3 {
            let bx = r.cfg.design_h[a];
            let lam = r.adj.field_ref(Component::h(a));
            for t in 0..bx.cells() {
                r.out.q1[a][t] += (lam[bx.storage(t, ny, nz)] * r.snap_h[a][t]) as f64;
            }
        }
        if let Some(nl) = &nonlin {
            for (ci, c) in nl.comps.iter().enumerate() {
                let lam = r.adj.field_mut(c.comp);
                for (t, &s) in c.idx.iter().enumerate() {
                    lam[s as usize] += r.nl_a[ci][t];
                }
            }
        }
        Ok(())
    }

    fn reverse_end(&mut self) -> Result<ReverseOut> {
        match self.rev.take() {
            Some(r) => Ok(r.out),
            None => bail!("reverse_begin has not been called"),
        }
    }

    fn reverse_field(&mut self, c: Component, adjoint: bool) -> Result<Vec<f32>> {
        match (&self.rev, adjoint) {
            (Some(r), true) => Ok(r.adj.field_ref(c).to_vec()),
            _ => Ok(self.be.field_ref(c).to_vec()),
        }
    }
}

fn add_source(f: &mut [f32], s: &AdjointSource, n: usize) {
    let m = s.idx.len();
    let row = &s.table[n * m..(n + 1) * m];
    for t in 0..m {
        f[s.idx[t] as usize] += s.coef[t] * row[t];
    }
}
