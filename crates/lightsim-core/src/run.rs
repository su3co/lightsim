//! Time-stepping drivers: single slab, in-process multi-slab, and SPMD
//! multi-process (TCP halo exchange between nodes).

use crate::engine::{Backend, RawResults, Side};
use crate::grid::Component;
use crate::results::{Results, RunStats, assemble, finalize};
use anyhow::Result;
use std::time::Instant;

/// A bidirectional point-to-point link to a remote rank.
pub trait Link: Send {
    fn send(&mut self, data: &[f32]) -> Result<()>;
    fn recv(&mut self, out: &mut [f32]) -> Result<()>;
}

/// Progress callback: (step, nsteps).
pub type Progress<'a> = Option<&'a mut dyn FnMut(usize, usize)>;

/// Options for a run.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Print progress to stderr every `progress_every` steps (0 = never).
    pub progress_every: usize,
}

const E_GHOST: [Component; 2] = [Component::Ey, Component::Ez];

/// Multiply the complex plane (re = buf[..plane], im = buf[plane..]) by c + i s.
fn phase_mix(buf: &mut [f32], plane: usize, c: f32, s: f32) {
    let (re, im) = buf.split_at_mut(plane);
    for (a, b) in re.iter_mut().zip(im.iter_mut()) {
        let (x, y) = (*a, *b);
        *a = c * x - s * y;
        *b = s * x + c * y;
    }
}
const H_GHOST: [Component; 2] = [Component::Hy, Component::Hz];

/// In-process chain of slab backends (ordered along x) plus optional links to
/// remote neighbours. `periodic` wraps the chain (or the ring of ranks).
pub struct Chain<'a> {
    pub slabs: Vec<Box<dyn Backend + 'a>>,
    pub lo_link: Option<Box<dyn Link + 'a>>,
    pub hi_link: Option<Box<dyn Link + 'a>>,
    /// Wrap the local chain onto itself (periodic x, no remote ranks).
    pub local_wrap: bool,
    /// The low/high link is the periodic wrap-around link of a ring.
    pub lo_is_wrap: bool,
    pub hi_is_wrap: bool,
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
}

impl<'a> Chain<'a> {
    pub fn new(slabs: Vec<Box<dyn Backend + 'a>>) -> Chain<'a> {
        let plane = slabs[0].plan().plane();
        Chain {
            slabs,
            lo_link: None,
            hi_link: None,
            local_wrap: false,
            lo_is_wrap: false,
            hi_is_wrap: false,
            buf_a: vec![0.0; plane],
            buf_b: vec![0.0; plane],
        }
    }

    /// Copy plane `src_plane` of slab `src` into plane `dst_plane` of slab
    /// `dst`, device-to-device when both backends support it.
    fn local_copy(&mut self, c: Component, src: usize, src_side: Side, dst: usize, dst_side: Side) -> Result<()> {
        let src_plane = match src_side {
            Side::Lo => 1,
            Side::Hi => self.slabs[src].plan().nxl,
        };
        let dst_plane = match dst_side {
            Side::Lo => 0,
            Side::Hi => self.slabs[dst].plan().nxl + 1,
        };
        if let (Some(sp), Some(_)) = (self.slabs[src].device_plane(c, src_plane), self.slabs[dst].device_plane(c, dst_plane)) {
            return self.slabs[dst].copy_plane_from_device(c, dst_plane, sp);
        }
        let plane = self.slabs[src].plan().plane();
        self.slabs[src].read_plane(c, src_side, &mut self.buf_a[..plane])?;
        self.slabs[dst].write_ghost(c, dst_side, &self.buf_a[..plane])
    }

    fn sync_all(&mut self) -> Result<()> {
        if self.slabs.len() > 1 {
            for s in self.slabs.iter_mut() {
                s.synchronize()?;
            }
        }
        Ok(())
    }

    /// E ghosts: slab r's high ghost (Ey, Ez) <- slab r+1's first interior plane.
    fn exchange_e(&mut self) -> Result<()> {
        self.exchange(true)
    }

    /// H ghosts: slab r's low ghost (Hy, Hz) <- slab r-1's last interior plane.
    fn exchange_h(&mut self) -> Result<()> {
        self.exchange(false)
    }

    /// Halo exchange of the E (`e = true`) or H ghosts between in-process
    /// slabs and remote ranks. Bloch runs exchange the real and the imaginary
    /// field set; across a periodic wrap along a Bloch x axis the ghost picks
    /// up the phase e^{+i k L} (E, high side) or e^{-i k L} (H, low side).
    fn exchange(&mut self, e: bool) -> Result<()> {
        let ns = self.slabs.len();
        self.sync_all()?;
        let (cplx, (pc, ps), plane) = {
            let p = self.slabs[0].plan();
            (p.bloch, p.bloch_phase[0], p.plane())
        };
        let (wc, ws) = (pc as f32, if e { ps as f32 } else { -ps as f32 });
        let phased = cplx && (ws != 0.0 || wc != 1.0);
        let sets: &[bool] = if cplx { &[false, true] } else { &[false] };
        let comps = if e { E_GHOST } else { H_GHOST };
        // (source slab, side) -> (destination slab, side) of the wrap
        let (wsrc, wss, wdst, wds) = if e { (0, Side::Lo, ns - 1, Side::Hi) } else { (ns - 1, Side::Hi, 0, Side::Lo) };
        for c in comps {
            for &im in sets {
                for s in self.slabs.iter_mut() {
                    s.select_set(im)?;
                }
                if e {
                    for r in 0..ns.saturating_sub(1) {
                        self.local_copy(c, r + 1, Side::Lo, r, Side::Hi)?;
                    }
                } else {
                    for r in 1..ns {
                        self.local_copy(c, r - 1, Side::Hi, r, Side::Lo)?;
                    }
                }
                if self.local_wrap && ns > 1 && !phased {
                    self.local_copy(c, wsrc, wss, wdst, wds)?;
                }
            }
            for s in self.slabs.iter_mut() {
                s.select_set(false)?;
            }
            let nset = sets.len();
            if self.buf_a.len() < nset * plane {
                self.buf_a.resize(nset * plane, 0.0);
                self.buf_b.resize(nset * plane, 0.0);
            }
            if self.local_wrap && ns > 1 && phased {
                for (q, &im) in sets.iter().enumerate() {
                    self.slabs[wsrc].select_set(im)?;
                    self.slabs[wsrc].read_plane(c, wss, &mut self.buf_a[q * plane..(q + 1) * plane])?;
                }
                self.slabs[wsrc].select_set(false)?;
                phase_mix(&mut self.buf_a[..2 * plane], plane, wc, ws);
                for (q, &im) in sets.iter().enumerate() {
                    self.slabs[wdst].select_set(im)?;
                    self.slabs[wdst].write_ghost(c, wds, &self.buf_a[q * plane..(q + 1) * plane])?;
                }
                self.slabs[wdst].select_set(false)?;
            }
            // remote ranks: E sends the first plane down and receives the high
            // ghost from above; H sends the last plane up and receives the low
            // ghost from below. Bloch runs send (re, im) as one message.
            if self.lo_link.is_some() || self.hi_link.is_some() {
                let n = nset * plane;
                let (send_slab, send_side, recv_slab, recv_side) =
                    if e { (0, Side::Lo, ns - 1, Side::Hi) } else { (ns - 1, Side::Hi, 0, Side::Lo) };
                let (has_send, has_recv, recv_wrap) = if e {
                    (self.lo_link.is_some(), self.hi_link.is_some(), self.hi_is_wrap)
                } else {
                    (self.hi_link.is_some(), self.lo_link.is_some(), self.lo_is_wrap)
                };
                if has_send {
                    for (q, &im) in sets.iter().enumerate() {
                        self.slabs[send_slab].select_set(im)?;
                        self.slabs[send_slab].read_plane(c, send_side, &mut self.buf_a[q * plane..(q + 1) * plane])?;
                    }
                    self.slabs[send_slab].select_set(false)?;
                }
                let (lo, hi) = (&mut self.lo_link, &mut self.hi_link);
                let (send_l, recv_l) = if e { (lo, hi) } else { (hi, lo) };
                let buf_a = &self.buf_a[..n];
                let buf_b = &mut self.buf_b[..n];
                std::thread::scope(|s| -> Result<()> {
                    let h = send_l.as_mut().map(|l| s.spawn(move || l.send(buf_a)));
                    if let Some(rl) = recv_l.as_mut() {
                        rl.recv(buf_b)?;
                    }
                    if let Some(h) = h {
                        h.join().expect("send thread panicked")?;
                    }
                    Ok(())
                })?;
                if has_recv {
                    if phased && recv_wrap {
                        phase_mix(&mut self.buf_b[..n], plane, wc, ws);
                    }
                    for (q, &im) in sets.iter().enumerate() {
                        self.slabs[recv_slab].select_set(im)?;
                        self.slabs[recv_slab].write_ghost(c, recv_side, &self.buf_b[q * plane..(q + 1) * plane])?;
                    }
                    self.slabs[recv_slab].select_set(false)?;
                }
            }
        }
        Ok(())
    }

    /// Off-diagonal tensors: dD ghost planes between slabs and ranks (dD_x of
    /// the last plane goes up into the low ghost, dD_y/dD_z of the first plane
    /// go down into the high ghost). Host-staged.
    fn exchange_tensor(&mut self) -> Result<()> {
        let ns = self.slabs.len();
        self.sync_all()?;
        let plane = self.slabs[0].plan().plane();
        if self.buf_a.len() < plane {
            self.buf_a.resize(plane, 0.0);
            self.buf_b.resize(plane, 0.0);
        }
        let lossy = self.slabs[0].plan().tensor.as_ref().map(|t| t.lossy).unwrap_or(false);
        let which_list: &[usize] = if lossy { &[0, 1] } else { &[0] };
        for &which in which_list {
        for (c, up) in [(Component::Ex, true), (Component::Ey, false), (Component::Ez, false)] {
            // in-process neighbours (and the periodic wrap)
            let mut pairs: Vec<(usize, usize)> = if up { (1..ns).map(|r| (r - 1, r)).collect() } else { (1..ns).map(|r| (r, r - 1)).collect() };
            if self.local_wrap && ns > 1 {
                pairs.push(if up { (ns - 1, 0) } else { (0, ns - 1) });
            }
            let (src_side, dst_side) = if up { (Side::Hi, Side::Lo) } else { (Side::Lo, Side::Hi) };
            for (src, dst) in pairs {
                self.slabs[src].tensor_read(c, src_side, which, &mut self.buf_a[..plane])?;
                self.slabs[dst].tensor_write_ghost(c, dst_side, which, &self.buf_a[..plane])?;
            }
            // remote ranks
            if self.lo_link.is_some() || self.hi_link.is_some() {
                let (send_slab, recv_slab) = if up { (ns - 1, 0) } else { (0, ns - 1) };
                let has_send = if up { self.hi_link.is_some() } else { self.lo_link.is_some() };
                let has_recv = if up { self.lo_link.is_some() } else { self.hi_link.is_some() };
                if has_send {
                    self.slabs[send_slab].tensor_read(c, src_side, which, &mut self.buf_a[..plane])?;
                }
                let (lo, hi) = (&mut self.lo_link, &mut self.hi_link);
                let (send_l, recv_l) = if up { (hi, lo) } else { (lo, hi) };
                let buf_a = &self.buf_a[..plane];
                let buf_b = &mut self.buf_b[..plane];
                std::thread::scope(|sc| -> Result<()> {
                    let h = send_l.as_mut().map(|l| sc.spawn(move || l.send(buf_a)));
                    if let Some(rl) = recv_l.as_mut() {
                        rl.recv(buf_b)?;
                    }
                    if let Some(h) = h {
                        h.join().expect("send thread panicked")?;
                    }
                    Ok(())
                })?;
                if has_recv {
                    self.slabs[recv_slab].tensor_write_ghost(c, dst_side, which, &self.buf_b[..plane])?;
                }
            }
        }
        }
        Ok(())
    }

    fn has_exchange(&self) -> bool {
        self.slabs.len() > 1 || self.lo_link.is_some() || self.hi_link.is_some()
    }

    /// Run all steps. Returns raw results per slab and run statistics.
    pub fn run(&mut self, opts: &RunOptions) -> Result<(Vec<RawResults>, RunStats)> {
        // CPU slabs: step from a thread of the pool they compute on
        if rayon::current_thread_index().is_none() && self.slabs.iter().all(|s| s.on_thread_pool()) {
            return rayon::scope(|_| self.run_loop(opts));
        }
        self.run_loop(opts)
    }

    fn run_loop(&mut self, opts: &RunOptions) -> Result<(Vec<RawResults>, RunStats)> {
        let p0 = self.slabs[0].plan().clone();
        let nsteps = p0.nsteps;
        let shutoff = p0.shutoff;
        let interval = p0.shutoff_interval;
        let t_src_end = p0.source_end_time;
        let total_cells: usize = p0.grid.num_cells();
        let exch = self.has_exchange();
        let tensor = p0.tensor.is_some();
        let t0 = Instant::now();
        let mut max_energy = 0.0f64;
        // largest energy seen while the sources were on (or at the first check after)
        let mut source_peak = 0.0f64;
        let mut steps_done = nsteps;
        let mut stopped = false;
        let mut energy_steps: Vec<usize> = vec![];
        let mut energy_trace: Vec<f64> = vec![];
        for n in 0..nsteps {
            if exch {
                self.exchange_e()?;
            }
            for s in self.slabs.iter_mut() {
                s.fill_ghost_e()?;
                s.update_h()?;
                s.inject_h(n)?;
            }
            if exch {
                self.exchange_h()?;
            }
            for s in self.slabs.iter_mut() {
                s.fill_ghost_h()?;
                s.update_e()?;
            }
            if tensor && exch {
                self.exchange_tensor()?;
            }
            for s in self.slabs.iter_mut() {
                s.finish_e()?;
                s.inject_e(n)?;
                s.record(n)?;
            }
            if opts.progress_every > 0 && (n + 1) % opts.progress_every == 0 {
                eprintln!("  step {}/{} ({:.1} s)", n + 1, nsteps, t0.elapsed().as_secs_f64());
            }
            // Field energy every `interval` steps and at the end: a run whose
            // fields are no longer finite, or grow after the sources have
            // stopped, is unstable and must not return results.
            if (n + 1) % interval == 0 || n + 1 == nsteps {
                let mut en = 0.0;
                for s in self.slabs.iter_mut() {
                    en += s.energy()?;
                }
                // remote ranks: sum energies through the ring
                en = self.allreduce_sum(en)?;
                let t = (n + 1) as f64 * p0.dt;
                let runaway = t > t_src_end && source_peak > 0.0 && en > 1e3 * source_peak;
                if !en.is_finite() || runaway {
                    anyhow::bail!(
                        "the simulation is unstable: the field energy {} at step {} of {}. \
                         Results are not returned. Usual causes: a dispersive medium whose poles are too \
                         strong for the time step, a medium with eps < 1 or negative conductivity, or courant > 1.",
                        if en.is_finite() { format!("has grown to {:.1e} times its value when the sources stopped", en / source_peak) } else { "is not finite".to_string() },
                        n + 1,
                        nsteps
                    );
                }
                energy_steps.push(n + 1);
                energy_trace.push(en);
                if t <= t_src_end || source_peak == 0.0 {
                    source_peak = source_peak.max(en);
                }
                max_energy = max_energy.max(en);
                if let Some(th) = shutoff {
                    if (n + 1) % interval == 0 && t > t_src_end && en <= th * max_energy {
                        steps_done = n + 1;
                        stopped = true;
                        break;
                    }
                }
            }
        }
        for s in self.slabs.iter_mut() {
            s.synchronize()?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let mut raws = Vec::new();
        for s in self.slabs.iter_mut() {
            raws.push(s.download()?);
        }
        let names: Vec<String> = self.slabs.iter().map(|s| s.name()).collect();
        let stats = RunStats {
            backend: names.join(" + "),
            cells: total_cells,
            steps: steps_done,
            seconds: secs,
            mcells_per_s: total_cells as f64 * steps_done as f64 / secs / 1e6,
            dt: p0.dt,
            stopped_early: stopped,
            energy_steps,
            energy: energy_trace,
        };
        Ok((raws, stats))
    }

    /// Sum a scalar over all ranks of a linear chain of ranks: partial sums
    /// travel up to the last rank, the total travels back down. Links marked
    /// as periodic wrap links are not used.
    fn allreduce_sum(&mut self, local: f64) -> Result<f64> {
        fn split(v: f64) -> [f32; 2] {
            let hi = v as f32;
            [hi, (v - hi as f64) as f32]
        }
        fn join(a: [f32; 2]) -> f64 {
            a[0] as f64 + a[1] as f64
        }
        let use_lo = self.lo_link.is_some() && !self.lo_is_wrap;
        let use_hi = self.hi_link.is_some() && !self.hi_is_wrap;
        if !use_lo && !use_hi {
            return Ok(local);
        }
        let mut acc = [0f32; 2];
        let mut total = local;
        if use_lo {
            self.lo_link.as_mut().unwrap().recv(&mut acc)?;
            total += join(acc);
        }
        if use_hi {
            let h = self.hi_link.as_mut().unwrap();
            h.send(&split(total))?;
            h.recv(&mut acc)?;
            total = join(acc);
        }
        if use_lo {
            self.lo_link.as_mut().unwrap().send(&split(total))?;
        }
        Ok(total)
    }
}

/// Run a single backend (whole domain) and produce finalized results.
pub fn run_backend(backend: Box<dyn Backend + '_>, opts: &RunOptions) -> Result<Results> {
    let mut chain = Chain::new(vec![backend]);
    run_chain(&mut chain, opts)
}

/// Run an in-process chain and produce finalized results.
pub fn run_chain(chain: &mut Chain, opts: &RunOptions) -> Result<Results> {
    let (raws, stats) = chain.run(opts)?;
    let plans: Vec<_> = chain.slabs.iter().map(|s| s.plan().clone()).collect();
    let prefs: Vec<&crate::plan::Plan> = plans.iter().collect();
    let mut res = assemble(&prefs, &raws);
    res.stats = stats;
    // gain media: population totals of all slabs (and ranks)
    if plans[0].media.iter().any(|m| m.gain.is_some()) {
        let mut tot = [0f64; 4];
        for s in chain.slabs.iter_mut() {
            if let Some(l) = s.gain_levels()? {
                for (t, v) in tot.iter_mut().zip(l) {
                    *t += v;
                }
            }
        }
        if chain.lo_link.is_some() || chain.hi_link.is_some() {
            for t in tot.iter_mut() {
                *t = chain.allreduce_sum(*t)?;
            }
        }
        res.gain_levels = tot.to_vec();
    }
    finalize(&mut res, &plans[0]);
    Ok(res)
}
