//! S-matrix problems with the gradient taken by reverse mode through the
//! time loop: the exact derivative of the discrete scheme, with the S of the
//! S-matrix session.
//!
//! For every input (a port mode or another source) one [`ReverseSession`]
//! runs the forward simulation and records, every step, the fields on the
//! planes the S-parameters of that input are read from: the outputs' port
//! planes and, for a port input, the two normalisation planes the session
//! places behind the source (which also give the reflection). S is computed
//! from these samples exactly as the session computes it: the DFT with the
//! phasor tables of the session's mode monitors (same stride, same window),
//! the core's mode overlap (whose weights are found once by probing it, as it
//! is linear in the DFT fields), the session's normalisation and the
//! unit-power port profiles. For an upstream gradient `g = dL/dS` the
//! reverse pass of an input differentiates `Re sum conj(g) S` over all its
//! outputs and frequencies at once, including the dependence of the
//! normalisation (launched amplitude) on the design, which the continuous
//! formula of the session leaves out.

use crate::Real;
use super::design::DesignGrid;
use super::engine::EngineFactory;
use super::reverse::{Objective, PlaneSpec, ReverseSession, ReverseSpec, Series};
use super::session::{Channel, GradientEngine, SessionSpec, base_simulation, check_media, named, norm_planes, port_power, session_options};
use crate::engine::{RawDft, RawResults};
use crate::monitors::{Apodization, Monitor};
use crate::plan::{Plan, compile_with_grid};
use crate::results::{assemble, finalize};
use crate::sources::{Direction, Source, TimeSignal};
use anyhow::{Result, anyhow, bail};
use num_complex::Complex64;
use std::time::Instant;

/// What a plane is read for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    Norm0,
    Norm1,
    Out,
}

struct Plane {
    port: usize,
    role: Role,
    /// Mode-overlap weights: `w[box][cell][(mode * 2 + dir) * nf + k]`,
    /// `dir` 0 = towards +axis, 1 = towards -axis; the amplitude is
    /// `sum_box,cell w * F` with F the DFT of the samples.
    w: Vec<Vec<Vec<Complex64>>>,
    nmode: usize,
}

struct InputRun {
    sess: ReverseSession,
    device: usize,
    src_port: Option<(usize, usize)>,
    /// Source-spectrum normalisation factor (excitations: times `norm`).
    ex_norm: Option<Vec<Complex64>>,
    signal: TimeSignal,
    planes: Vec<Plane>,
    /// Phasor tables of the session's mode monitors, `[step][freq][2]`.
    pe: Vec<Real>,
    ph: Vec<Real>,
    /// Amplitudes of the last forward run, per plane `[(mode * 2 + dir) * nf + k]`.
    amps: Vec<Vec<Complex64>>,
    /// The DFT phasors of the boxes, `[box] -> E or H`.
    box_is_e: Vec<bool>,
    ran: bool,
}

/// One timing row per run.
#[derive(Clone, Debug)]
pub struct ReverseRunStat {
    pub input: usize,
    pub device: String,
    pub steps: usize,
    pub forward_seconds: f64,
    pub reverse_seconds: f64,
}

pub struct ReverseSMatrix {
    spec: SessionSpec,
    runs: Vec<InputRun>,
    devices: Vec<String>,
    pub stats: Vec<ReverseRunStat>,
}

impl ReverseSMatrix {
    /// Build one reverse-mode session per input (on `devices[n % len]`) and
    /// find the overlap weights of its planes.
    pub fn new(spec: SessionSpec, devices: &[String], factory: &EngineFactory) -> Result<ReverseSMatrix> {
        if devices.is_empty() {
            bail!("no device given");
        }
        if spec.normalize != "monitor" && spec.normalize != "source" {
            bail!("normalize must be 'monitor' or 'source'");
        }
        if spec.freqs.is_empty() || spec.inputs.is_empty() {
            bail!("an S-matrix problem needs at least one frequency and one input");
        }
        if spec.outputs.iter().any(|c| !matches!(c, Channel::Port { .. })) {
            bail!("outputs must be port modes");
        }
        for ch in spec.inputs.iter().chain(spec.outputs.iter()) {
            if let Channel::Port { port, mode } = ch {
                let p = spec.ports.get(*port).ok_or_else(|| anyhow!("channel refers to unknown port {port}"))?;
                if *mode >= p.profiles.len() {
                    bail!("port '{}' has {} mode profiles, mode {mode} requested", p.name, p.profiles.len());
                }
            }
        }
        let base = base_simulation(&spec.sim, &spec.grids)?;
        check_media(&base, GradientEngine::ReverseSMatrix)?;
        let grid = base.build_grid()?;
        let b = &base.boundaries;
        let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
        // unit-power profiles in the overlap's measure, as in the session
        let mut spec = spec;
        for port in spec.ports.iter_mut() {
            for k in 0..port.profiles.len() {
                let pw = port_power(&grid, periodic, port, k);
                if !pw.is_finite() || pw == 0.0 {
                    bail!("port '{}': mode profile {k} carries no power through the port window ({pw} W)", port.name);
                }
                let f = 1.0 / pw.abs().sqrt();
                let q = &mut port.profiles[k];
                for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                    arr.iter_mut().for_each(|x| *x *= f);
                }
            }
        }
        let nf = spec.freqs.len();
        let mut runs = vec![];
        for (n, ch) in spec.inputs.iter().enumerate() {
            let (src, src_port, ex_norm, signal) = match ch {
                Channel::Port { port, mode } => {
                    let p = &spec.ports[*port];
                    let src = Source::Mode { name: "__src__".into(), axis: p.axis, position: p.position, direction: p.direction, min: p.min, max: p.max, profile: p.profiles[*mode].clone(), signal: spec.signal.clone(), amplitude: 1.0 };
                    (src, Some((*port, *mode)), None, spec.signal.clone())
                }
                Channel::Source { source, norm, .. } => {
                    if let Some(v) = norm {
                        if v.len() != nf {
                            bail!("an excitation's norm needs one value per frequency");
                        }
                    }
                    let sig = source.signal().clone();
                    (named(source.clone(), "__src__"), None, norm.as_ref().map(|v| v.iter().map(|z| Complex64::new(z[0], z[1])).collect()), sig)
                }
            };
            // the planes of this input: normalisation planes, then the
            // planes of the other output ports (in order, once each)
            let mut planes_spec: Vec<(usize, Role, f64, [f64; 3], [f64; 3])> = vec![];
            if let Some((port, _)) = src_port {
                let p = &spec.ports[port];
                for (t, (x, mn, mx)) in norm_planes(&grid, p)?.into_iter().enumerate() {
                    planes_spec.push((port, if t == 0 { Role::Norm0 } else { Role::Norm1 }, x, mn, mx));
                }
            }
            for out in &spec.outputs {
                let Channel::Port { port, .. } = out else { unreachable!() };
                if Some(*port) != src_port.map(|p| p.0) && !planes_spec.iter().any(|q| q.0 == *port && q.1 == Role::Out) {
                    let p = &spec.ports[*port];
                    planes_spec.push((*port, Role::Out, p.position, p.min, p.max));
                }
            }
            // the mode monitors of these planes, compiled as in the session
            let mut msim = base.clone();
            msim.sources = vec![src.clone()];
            msim.monitors = planes_spec
                .iter()
                .enumerate()
                .map(|(i, (port, _, x, mn, mx))| {
                    let p = &spec.ports[*port];
                    Monitor::Mode { name: format!("__plane{i}__"), axis: p.axis, position: *x, min: *mn, max: *mx, freqs: spec.freqs.clone(), modes: p.profiles.clone(), apodization: Apodization::default() }
                })
                .collect();
            let mplan = compile_with_grid(&msim, grid.clone(), &session_options())?;
            if let Some(d) = mplan.dft.iter().find(|d| d.stride != mplan.dft[0].stride) {
                bail!("monitor '{}' has another stride", d.name);
            }
            let (pe, ph) = (mplan.dft[0].phasor_e.clone(), mplan.dft[0].phasor_h.clone());
            let mut planes = vec![];
            for (i, (port, role, ..)) in planes_spec.iter().enumerate() {
                let nmode = spec.ports[*port].profiles.len();
                planes.push(Plane { port: *port, role: *role, w: probe(&mplan, i, nf, nmode), nmode });
            }
            // the reverse-mode session recording these planes
            let mut sim = spec.sim.clone();
            sim.sources = vec![src];
            let objective = Objective::Planes { planes: planes_spec.iter().map(|(port, _, x, mn, mx)| PlaneSpec { axis: spec.ports[*port].axis, position: *x, min: *mn, max: *mx }).collect() };
            let device = n % devices.len();
            let mut sess = ReverseSession::new(ReverseSpec { sim, grids: spec.grids.clone(), objective }, &devices[device], factory)?;
            sess.set_shutoff(true);
            let box_is_e = (0..planes.len()).flat_map(|_| [true, true, false, false, false, false]).collect();
            runs.push(InputRun { sess, device, src_port, ex_norm, signal, planes, pe, ph, amps: vec![], box_is_e, ran: false });
        }
        Ok(ReverseSMatrix { spec, runs, devices: devices.to_vec(), stats: vec![] })
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.spec.outputs.len(), self.spec.inputs.len(), self.spec.freqs.len())
    }
    pub fn grids(&self) -> &[DesignGrid] {
        &self.spec.grids
    }
    pub fn devices(&self) -> Vec<String> {
        self.runs.iter().map(|r| r.sess.device()).collect()
    }

    /// Set the index-grid samples on every input's session.
    pub fn set_design(&mut self, eps: &[Vec<f64>]) -> Result<()> {
        for r in self.runs.iter_mut() {
            r.sess.set_design(eps)?;
            r.ran = false;
        }
        Ok(())
    }

    /// Run the forward simulations (inputs spread over the devices) and
    /// return `S[m][n][k]` flattened in that order.
    pub fn forward(&mut self) -> Result<Vec<Complex64>> {
        let (m, n, k) = self.shape();
        let spec = &self.spec;
        let nd = self.devices.len();
        let mut groups: Vec<Vec<&mut InputRun>> = (0..nd).map(|_| vec![]).collect();
        for r in self.runs.iter_mut() {
            groups[r.device].push(r);
        }
        let results: Vec<Result<()>> = std::thread::scope(|sc| {
            let hs: Vec<_> = groups
                .into_iter()
                .map(|g| {
                    sc.spawn(move || -> Result<()> {
                        for r in g {
                            if !r.ran {
                                r.forward(spec)?;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("a run panicked")))).collect()
        });
        for r in results {
            r?;
        }
        let mut s = vec![Complex64::new(0.0, 0.0); m * n * k];
        for (ni, r) in self.runs.iter().enumerate() {
            let col = r.s_column(&self.spec)?;
            for mi in 0..m {
                for ki in 0..k {
                    s[(mi * n + ni) * k + ki] = col[mi * k + ki];
                }
            }
        }
        self.stats = self.runs.iter().enumerate().map(|(i, r)| ReverseRunStat { input: i, device: r.sess.device(), steps: r.sess.steps(), forward_seconds: r.sess.stats.forward_seconds, reverse_seconds: 0.0 }).collect();
        Ok(s)
    }

    /// dL/d(eps) on the samples of every design grid for the upstream
    /// gradient `g[m][n][k]` (flattened like S): one reverse pass per input
    /// whose column of `g` is not zero.
    pub fn backward(&mut self, g: &[Complex64]) -> Result<Vec<Vec<f64>>> {
        let (m, n, k) = self.shape();
        if g.len() != m * n * k {
            bail!("upstream gradient has {} entries, S has {}", g.len(), m * n * k);
        }
        if self.runs.iter().any(|r| !r.ran) {
            bail!("forward must be called before backward");
        }
        let spec = &self.spec;
        let nd = self.devices.len();
        let mut groups: Vec<Vec<(usize, &mut InputRun)>> = (0..nd).map(|_| vec![]).collect();
        for (ni, r) in self.runs.iter_mut().enumerate() {
            groups[r.device].push((ni, r));
        }
        let results: Vec<Result<Vec<(usize, Vec<Vec<f64>>, f64)>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = groups
                .into_iter()
                .map(|grp| {
                    sc.spawn(move || -> Result<Vec<(usize, Vec<Vec<f64>>, f64)>> {
                        let mut out = vec![];
                        for (ni, r) in grp {
                            let col: Vec<Complex64> = (0..m * k).map(|t| g[((t / k) * n + ni) * k + t % k]).collect();
                            if col.iter().all(|z| z.norm() == 0.0) {
                                continue;
                            }
                            let t0 = Instant::now();
                            let gr = r.backward(spec, &col)?;
                            out.push((ni, gr, t0.elapsed().as_secs_f64()));
                        }
                        Ok(out)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow!("a run panicked")))).collect()
        });
        let mut total: Vec<Vec<f64>> = self.spec.grids.iter().map(|gr| vec![0.0; gr.samples()]).collect();
        for res in results {
            for (ni, gr, secs) in res? {
                for (t, v) in total.iter_mut().zip(&gr) {
                    t.iter_mut().zip(v).for_each(|(a, b)| *a += b);
                }
                if let Some(st) = self.stats.get_mut(ni) {
                    st.reverse_seconds = secs;
                }
            }
        }
        Ok(total)
    }
}

/// The overlap weights of plane `i` of `plan`: the amplitudes of every mode
/// and direction for a unit DFT value in one cell (all frequencies at once;
/// the overlap is linear in the fields and frequencies do not mix).
fn probe(plan: &Plan, i: usize, nf: usize, nmode: usize) -> Vec<Vec<Vec<Complex64>>> {
    let mut one = plan.clone();
    one.dft = vec![plan.dft[i].clone()];
    one.time = vec![];
    one.energy = vec![];
    let name = one.dft[0].name.clone();
    let boxes = one.dft[0].boxes.clone();
    let zero: Vec<(Vec<Real>, Vec<Real>)> = boxes.iter().map(|b| (vec![0 as Real; nf * b.lcells()], vec![0 as Real; nf * b.lcells()])).collect();
    boxes
        .iter()
        .enumerate()
        .map(|(bi, b)| {
            (0..b.lcells())
                .map(|t| {
                    let mut raw = zero.clone();
                    for k in 0..nf {
                        raw[bi].0[k * b.lcells() + t] = 1.0;
                    }
                    let mut res = assemble(&[&one], &[RawResults { dft: vec![RawDft { boxes: raw }], time: vec![] }]);
                    finalize(&mut res, &one);
                    let mr = &res.modes[&name];
                    let mut w = vec![Complex64::new(0.0, 0.0); nmode * 2 * nf];
                    for md in 0..nmode.min(mr.amp_plus_re.len()) {
                        for k in 0..nf {
                            w[(md * 2) * nf + k] = Complex64::new(mr.amp_plus_re[md][k], mr.amp_plus_im[md][k]);
                            w[(md * 2 + 1) * nf + k] = Complex64::new(mr.amp_minus_re[md][k], mr.amp_minus_im[md][k]);
                        }
                    }
                    w
                })
                .collect()
        })
        .collect()
}

impl InputRun {
    fn forward(&mut self, spec: &SessionSpec) -> Result<()> {
        self.sess.forward()?;
        let series = self.sess.series()?;
        let nf = spec.freqs.len();
        self.amps = self
            .planes
            .iter()
            .enumerate()
            .map(|(pi, pl)| {
                let mut a = vec![Complex64::new(0.0, 0.0); pl.nmode * 2 * nf];
                for bi in 0..6 {
                    let s = &series[pi * 6 + bi];
                    let f = self.dft(s, self.box_is_e[pi * 6 + bi], nf);
                    for (t, wt) in pl.w[bi].iter().enumerate() {
                        for x in 0..pl.nmode * 2 {
                            for k in 0..nf {
                                a[x * nf + k] += wt[x * nf + k] * f[t * nf + k];
                            }
                        }
                    }
                }
                a
            })
            .collect();
        self.ran = true;
        Ok(())
    }

    fn phasor(&self, is_e: bool, n: usize, k: usize, nf: usize) -> Complex64 {
        let tab = if is_e { &self.pe } else { &self.ph };
        let o = (n * nf + k) * 2;
        Complex64::new(tab[o] as f64, tab[o + 1] as f64)
    }

    /// DFT of the recorded samples of one box: `[cell][freq]`.
    fn dft(&self, s: &Series, is_e: bool, nf: usize) -> Vec<Complex64> {
        let m: usize = s.size.iter().product();
        let ns = s.data.len() / m.max(1);
        let mut f = vec![Complex64::new(0.0, 0.0); m * nf];
        for n in 0..ns {
            let ph: Vec<Complex64> = (0..nf).map(|k| self.phasor(is_e, n, k, nf)).collect();
            if ph.iter().all(|z| z.norm() == 0.0) {
                continue;
            }
            let row = &s.data[n * m..(n + 1) * m];
            for (t, v) in row.iter().enumerate() {
                if *v != 0.0 {
                    for k in 0..nf {
                        f[t * nf + k] += *v as f64 * ph[k];
                    }
                }
            }
        }
        f
    }

    fn amp(&self, role: Role, port: usize, mode: usize, dir: usize, k: usize, nf: usize) -> Option<(usize, usize, Complex64)> {
        let pi = self.planes.iter().position(|p| p.role == role && p.port == port)?;
        let x = (mode * 2 + dir) * nf + k;
        Some((pi, x, self.amps[pi][x]))
    }

    /// The column of S of this input, `[output][freq]`, and (with `grad`)
    /// the derivatives `dS[m][k] / d(amplitude)`, as `(plane, index, value)`.
    fn s_and_partials(&self, spec: &SessionSpec) -> Result<(Vec<Complex64>, Vec<Vec<Vec<(usize, usize, Complex64)>>>)> {
        let nf = spec.freqs.len();
        let steps = self.sess.steps();
        let one = Complex64::new(1.0, 0.0);
        let dir_of = |plus: bool| if plus { 0usize } else { 1 };
        // normalisation N_k and its partials
        let mut norm = vec![one; nf];
        let mut dnorm: Vec<Vec<(usize, usize, Complex64)>> = vec![vec![]; nf];
        let mut q: Option<Vec<(Complex64, (usize, usize), (usize, usize))>> = None;
        match (self.src_port, spec.normalize.as_str()) {
            (Some((port, mode)), "monitor") => {
                let plus = spec.ports[port].direction == Direction::Plus;
                let mut qs = vec![];
                for k in 0..nf {
                    let (p1, x1, a1) = self.amp(Role::Norm0, port, mode, dir_of(plus), k, nf).ok_or_else(|| anyhow!("normalisation plane missing"))?;
                    let (p2, x2, a2) = self.amp(Role::Norm1, port, mode, dir_of(plus), k, nf).ok_or_else(|| anyhow!("normalisation plane missing"))?;
                    // N = a1 q, q = a1 / a2: N = a1^2 / a2
                    norm[k] = a1 * a1 / a2;
                    dnorm[k] = vec![(p1, x1, 2.0 * a1 / a2), (p2, x2, -a1 * a1 / (a2 * a2))];
                    qs.push((a1 / a2, (p1, x1), (p2, x2)));
                }
                q = Some(qs);
            }
            _ => {
                let mut s = self.signal.spectrum(&spec.freqs, self.sess.dt(), steps.max(1), 0.5);
                if let Some(en) = &self.ex_norm {
                    s.iter_mut().zip(en).for_each(|(a, b)| *a *= b);
                }
                norm = s;
            }
        }
        let mut col = vec![];
        let mut parts = vec![];
        for out in &spec.outputs {
            let Channel::Port { port, mode } = out else { unreachable!() };
            let back_plus = spec.ports[*port].direction == Direction::Minus;
            let mut row = vec![];
            let mut prow = vec![];
            for k in 0..nf {
                let n_k = norm[k];
                let mut p: Vec<(usize, usize, Complex64)> = vec![];
                let s = if Some(*port) == self.src_port.map(|p| p.0) {
                    // reflection: measured 2 cells inside, referred back with q
                    let (pr, xr, ar) = self.amp(Role::Norm0, *port, *mode, dir_of(back_plus), k, nf).ok_or_else(|| anyhow!("normalisation plane missing"))?;
                    match &q {
                        Some(qs) => {
                            let (qk, (p1, x1), (p2, x2)) = qs[k];
                            let a1 = self.amps[p1][x1];
                            let a2 = self.amps[p2][x2];
                            // S = ar / (q N) = ar a2^2 / a1^3
                            let s = ar / (qk * n_k);
                            p.push((pr, xr, a2 * a2 / (a1 * a1 * a1)));
                            p.push((p2, x2, 2.0 * ar * a2 / (a1 * a1 * a1)));
                            p.push((p1, x1, -3.0 * ar * a2 * a2 / (a1 * a1 * a1 * a1)));
                            s
                        }
                        None => {
                            p.push((pr, xr, one / n_k));
                            ar / n_k
                        }
                    }
                } else {
                    let (po, xo, ao) = self.amp(Role::Out, *port, *mode, dir_of(back_plus), k, nf).ok_or_else(|| anyhow!("output plane missing"))?;
                    p.push((po, xo, one / n_k));
                    for (pn, xn, dn) in &dnorm[k] {
                        p.push((*pn, *xn, -ao / (n_k * n_k) * dn));
                    }
                    ao / n_k
                };
                row.push(s);
                prow.push(p);
            }
            col.extend(row);
            parts.push(prow);
        }
        Ok((col, parts))
    }

    fn s_column(&self, spec: &SessionSpec) -> Result<Vec<Complex64>> {
        Ok(self.s_and_partials(spec)?.0)
    }

    /// Reverse pass for `L = Re sum_m,k conj(g[m][k]) S[m][k]`.
    fn backward(&mut self, spec: &SessionSpec, g: &[Complex64]) -> Result<Vec<Vec<f64>>> {
        let nf = spec.freqs.len();
        let (_, parts) = self.s_and_partials(spec)?;
        // dL/d(amplitude) per plane: C[x * nf + k] = sum_m conj(g) dS/da
        let mut c: Vec<Vec<Complex64>> = self.planes.iter().map(|p| vec![Complex64::new(0.0, 0.0); p.nmode * 2 * nf]).collect();
        for (mi, prow) in parts.iter().enumerate() {
            for (k, p) in prow.iter().enumerate() {
                let gc = g[mi * nf + k].conj();
                for (pi, x, d) in p {
                    c[*pi][*x] += gc * d;
                }
            }
        }
        // cotangents of the samples: Re sum_k (sum_x C w) phasor(n, k)
        let series = self.sess.series()?;
        let mut cot = vec![];
        for (pi, pl) in self.planes.iter().enumerate() {
            for bi in 0..6 {
                let s = &series[pi * 6 + bi];
                let m: usize = s.size.iter().product();
                let ns = s.data.len() / m.max(1);
                let is_e = self.box_is_e[pi * 6 + bi];
                let ck: Vec<Complex64> = (0..m * nf)
                    .map(|j| {
                        let (t, k) = (j / nf, j % nf);
                        (0..pl.nmode * 2).map(|x| c[pi][x * nf + k] * pl.w[bi][t][x * nf + k]).sum()
                    })
                    .collect();
                let mut out = vec![0.0; ns * m];
                for n in 0..ns {
                    let ph: Vec<Complex64> = (0..nf).map(|k| self.phasor(is_e, n, k, nf)).collect();
                    if ph.iter().all(|z| z.norm() == 0.0) {
                        continue;
                    }
                    for t in 0..m {
                        out[n * m + t] = (0..nf).map(|k| (ck[t * nf + k] * ph[k]).re).sum();
                    }
                }
                cot.push(out);
            }
        }
        self.sess.backward_with(&cot)
    }
}
