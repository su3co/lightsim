//! The 2-D GPU step kernel, run on the host.
//!
//! `kernels/src/twod_step.rs` holds the body of the device kernels `step2d`
//! / `step2d_bloch` as a macro. Here the same text is expanded into ordinary
//! functions and called once per cell, with buffers laid out as the GPU
//! backend lays them out (ping-pong field sets, material table, CPML
//! profiles and state in the layout of the fused step). The fields must
//! match the CPU backend. This checks the kernel logic (signs, CPML slots,
//! boundary codes, all three invariant axes) without a GPU; the launch code
//! itself is covered by the GPU tests of `lightsim-cuda`.
//!
//! Not emulated here: magnetic-current injections (the GPU applies them
//! through equivalent E corrections) and the deferred E update of the first
//! x-plane (periodic x), which stay with `e_plane_fused`.

use lightsim_core::autodiff::engine::forward_step;
use lightsim_core::engine::cpu::CpuBackend;
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::{Medium, Pole};
use lightsim_core::plan::{CompileOptions, GhostRule, Plan, Step2d, compile};
use lightsim_core::sources::{Source, TimeSignal};
use lightsim_core::spec::{Boundaries, Boundary, Simulation};
use lightsim_core::Backend;

include!("../../../kernels/src/twod_step.rs");

step2d_fn!($ [#[allow(clippy::too_many_arguments, unused_variables, unused_assignments, unused_mut, clippy::missing_safety_doc)]] step2d_host, false, [t_in: u32,], t_in);
step2d_fn!($ [#[allow(clippy::too_many_arguments, unused_variables, unused_assignments, unused_mut, clippy::missing_safety_doc)]] step2d_bloch_host, true, [t_in: u32,], t_in);

/// One part (real or imaginary) of the fields and its CPML state.
struct Part {
    a: [Vec<f32>; 6],
    b: [Vec<f32>; 6],
    psi_e: [Vec<f32>; 3],
    psi_h: [[Vec<f32>; 3]; 2],
    /// dispersive state per component list: (P, P_prev, dP)
    disp: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

/// Host emulation of the GPU backend's 2-D step.
struct Emu {
    plan: Plan,
    k: Step2d,
    parts: Vec<Part>,
    mat: Vec<u32>,
    tab: Vec<f32>,
    prof: [Vec<f32>; 3],
    pml_n: [u32; 3],
    cur: usize,
}

impl Emu {
    fn new(plan: Plan) -> Emu {
        let k = plan.step2d().expect("2-D plan");
        assert!(!k.deferred, "the deferred first plane is not emulated");
        assert!(plan.inj_h.iter().all(|j| j.idx.is_empty()), "magnetic-current injections are not emulated");
        assert!(plan.ch.is_none() && plan.nonlin.is_none() && plan.tensor.is_none());
        let len = plan.storage_len();
        // one table row per storage cell
        let mat: Vec<u32> = (0..len as u32).collect();
        let mut tab = vec![0f32; 6 * len];
        for s in 0..len {
            for a in 0..3 {
                tab[6 * s + a] = plan.cb[a][s];
                tab[6 * s + 3 + a] = plan.ca.as_ref().map(|c| c[a][s]).unwrap_or(1.0);
            }
        }
        // CPML profiles and state sizes, as `fused::build` of lightsim-cuda
        let n = [plan.nxl, plan.ny(), plan.nz()];
        let mut prof: [Vec<f32>; 3] = [0, 1, 2].map(|a| (0..n[a]).flat_map(|_| [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0]).collect());
        let (mut nlo, mut nhi) = ([0usize; 3], [0usize; 3]);
        for p in &plan.pml {
            let a = p.axis;
            for q in 0..p.size[a] {
                let pos = p.start[a] + q;
                prof[a][6 * pos..6 * pos + 6].copy_from_slice(&[p.e_b[q], p.e_a[q], p.e_kinv[q], p.h_b[q], p.h_a[q], p.h_kinv[q]]);
            }
            if p.start[a] == 0 {
                nlo[a] = p.size[a];
            } else {
                nhi[a] = p.size[a];
            }
        }
        let cnt = [(nlo[0] + nhi[0]) * n[1] * n[2], n[0] * (nlo[1] + nhi[1]) * n[2], n[0] * n[1] * (nlo[2] + nhi[2])];
        let part = || Part {
            a: [0; 6].map(|_| vec![0f32; len]),
            b: [0; 6].map(|_| vec![0f32; len]),
            psi_e: [0, 1, 2].map(|a| vec![0f32; (2 * cnt[a]).max(1)]),
            psi_h: [0, 1].map(|_| [0, 1, 2].map(|a| vec![0f32; (2 * cnt[a]).max(1)])),
            disp: plan.disp.as_ref().map(|d| d.comps.iter().map(|c| (vec![0f32; c.idx.len() * d.npole], vec![0f32; c.idx.len() * d.npole], vec![0f32; c.idx.len()])).collect()).unwrap_or_default(),
        };
        let parts = if plan.bloch { vec![part(), part()] } else { vec![part()] };
        let pml_n = [0, 1, 2].map(|a| (nlo[a] | nhi[a] << 16) as u32);
        Emu { plan, k, parts, mat, tab, prof, pml_n, cur: 0 }
    }

    /// Copy x-plane `src` of the A buffers of `comps` to plane `dst`, times `f`.
    fn plane_copy(&mut self, dst: usize, src: usize, comps: [Component; 2], f: f32) {
        let plane = self.plan.plane();
        for p in self.parts.iter_mut() {
            for c in comps {
                let arr = &mut p.a[c.index()];
                for t in 0..plane {
                    arr[dst * plane + t] = f * arr[src * plane + t];
                }
            }
        }
    }

    fn step(&mut self, n: usize) {
        let plan = self.plan.clone();
        let k = self.k;
        let nxl = plan.nxl;
        // high-x ghost of E on the current buffers (as `fill_ghost_e`)
        match plan.ghost_e_hi {
            GhostRule::Periodic { cos, sin } => {
                assert!(sin == 0.0, "Bloch phase along x: not emulated");
                self.plane_copy(nxl + 1, 1, [Component::Ey, Component::Ez], cos as f32);
            }
            GhostRule::Mirror { factor } => self.plane_copy(nxl + 1, nxl, [Component::Ey, Component::Ez], factor),
            _ => {}
        }
        // dispersive poles, step 1: P from E^n (current buffers)
        if let Some(d) = &plan.disp {
            for p in self.parts.iter_mut() {
                for (ci, dc) in d.comps.iter().enumerate() {
                    let f = &p.a[dc.comp.index()];
                    let (pp, pprev, dp) = &mut p.disp[ci];
                    let ne = dc.idx.len();
                    for t in 0..ne {
                        let e = f[dc.idx[t] as usize];
                        let m = dc.matid[t] as usize;
                        let mut acc = 0f32;
                        for q in 0..d.npole {
                            let c = &d.coefs[(m * d.npole + q) * 3..(m * d.npole + q) * 3 + 3];
                            let o = q * ne + t;
                            let p0 = pp[o];
                            let pn = c[0] * p0 + c[1] * pprev[o] + c[2] * e;
                            acc += pn - p0;
                            pprev[o] = p0;
                            pp[o] = pn;
                        }
                        dp[t] = acc;
                    }
                }
            }
        }
        // the kernel, once per part, one call per cell
        let td = plan.twod.unwrap();
        let pol = Step2d::pol(&td);
        let (u, v, w) = (k.u, k.v, k.w);
        let nparts = self.parts.len();
        let rd = self.cur;
        let wr = 1 - self.cur;
        for part in 0..nparts {
            let other = if nparts == 2 { 1 - part } else { part };
            // -sin for the real part, +sin for the imaginary one
            let sign = if part == 0 { -1.0 } else { 1.0 };
            let bs = |a: usize| sign * plan.bloch_phase[a].1 as f32;
            let bcos = |a: usize| plan.bloch_phase[a].0 as f32;
            let me: *mut Part = &mut self.parts[part];
            let ot: *const Part = &self.parts[other];
            unsafe {
                let e = |c: usize| (*me).a[c].as_ptr();
                let h = |c: usize| (*me).a[3 + c].as_ptr();
                let eb = |c: usize| (*me).b[c].as_mut_ptr();
                let hb = |c: usize| (*me).b[3 + c].as_mut_ptr();
                let eo = |c: usize| (*ot).a[c].as_ptr();
                let ho = |c: usize| (*ot).a[3 + c].as_ptr();
                for t in 0..k.nu * k.nv {
                    let args = (
                        t,
                        e(u),
                        e(v),
                        e(w),
                        h(u),
                        h(v),
                        h(w),
                        eb(u),
                        eb(v),
                        eb(w),
                        hb(u),
                        hb(v),
                        hb(w),
                    );
                    macro_rules! call {
                        ($f:ident) => {
                            $f(
                                args.0,
                                args.1,
                                args.2,
                                args.3,
                                args.4,
                                args.5,
                                args.6,
                                args.7,
                                args.8,
                                args.9,
                                args.10,
                                args.11,
                                args.12,
                                self.mat.as_ptr(),
                                self.tab.as_ptr(),
                                4,
                                plan.ca.is_some() as u32,
                                k.cols,
                                plan.ch_scalar,
                                plan.inv_p[u].as_ptr(),
                                plan.inv_p[v].as_ptr(),
                                plan.inv_d[u].as_ptr(),
                                plan.inv_d[v].as_ptr(),
                                k.nu,
                                k.nv,
                                k.base,
                                k.bc,
                                self.pml_n[u],
                                self.pml_n[v],
                                self.prof[u].as_ptr(),
                                self.prof[v].as_ptr(),
                                (*me).psi_e[u].as_mut_ptr(),
                                (*me).psi_e[v].as_mut_ptr(),
                                (*me).psi_h[rd][u].as_ptr(),
                                (*me).psi_h[rd][v].as_ptr(),
                                (*me).psi_h[wr][u].as_mut_ptr(),
                                (*me).psi_h[wr][v].as_mut_ptr(),
                                pol,
                                k.sg,
                                eo(u),
                                eo(v),
                                eo(w),
                                ho(u),
                                ho(v),
                                ho(w),
                                (*ot).psi_h[rd][u].as_ptr(),
                                (*ot).psi_h[rd][v].as_ptr(),
                                bcos(u),
                                bs(u),
                                bcos(v),
                                bs(v),
                            )
                        };
                    }
                    if nparts == 2 {
                        call!(step2d_bloch_host)
                    } else {
                        call!(step2d_host)
                    }
                }
            }
        }
        // dispersive poles, step 2, and the electric-current sources, on the new E
        for (pi, p) in self.parts.iter_mut().enumerate() {
            if let Some(d) = &plan.disp {
                for (ci, dc) in d.comps.iter().enumerate() {
                    let f = &mut p.b[dc.comp.index()];
                    for t in 0..dc.idx.len() {
                        f[dc.idx[t] as usize] -= dc.scale[t] * p.disp[ci].2[t];
                    }
                }
            }
            if pi == 0 {
                for j in &plan.inj_e {
                    assert!(j.coef_im.is_none());
                    let row = j.row(n);
                    for e in 0..j.idx.len() {
                        let g = if j.grouped() { j.group[e] as usize } else { 0 };
                        p.b[j.comp.index()][j.idx[e] as usize] += j.coef[e] * j.amps[row + g];
                    }
                }
            }
            std::mem::swap(&mut p.a, &mut p.b);
        }
        self.cur ^= 1;
    }
}

const UM: f64 = 1e-6;
const F0: f64 = 2.0e14;

fn lift(w: usize, a: f64, b: f64, c: f64) -> [f64; 3] {
    match w {
        0 => [c, a, b],
        1 => [a, c, b],
        _ => [a, b, c],
    }
}

fn pml() -> [Boundary; 2] {
    [Boundary::Pml { layers: 8, params: Default::default(), inside: false }, Boundary::Pml { layers: 8, params: Default::default(), inside: false }]
}

/// One-cell grid along `w` with a dielectric rod, a lossy block and a
/// dispersive block, and electric dipoles of the requested sets.
fn scene(w: usize, bu: [Boundary; 2], bv: [Boundary; 2], sets: [bool; 2]) -> Simulation {
    let mut b = Boundaries::all(Boundary::Periodic);
    let (u, v) = match w {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    *b.axis_mut(u) = bu;
    *b.axis_mut(v) = bv;
    let mut d = [40e-9, 35e-9, 45e-9];
    d[w] = 60e-9;
    let mut sim = Simulation::new(lift(w, -1.0 * UM, -0.8 * UM, 0.0), lift(w, 1.0 * UM, 0.8 * UM, 60e-9), MeshSpec::Uniform { dx: d }, b, 12e-15);
    sim.media.push(Medium::from_index(2.0).with_name("glass"));
    sim.media.push(Medium::conductor(2.25, 3e4).with_name("lossy"));
    let om = std::f64::consts::TAU * F0;
    sim.media.push(Medium::dielectric(2.0).with_name("lor").with_poles(vec![Pole::Lorentz { delta_eps: 1.5, omega0: 1.8 * om, gamma: 0.05 * om }]));
    sim.structures.push(Structure { name: "rod".into(), shape: Shape::Cylinder { center: lift(w, -0.2 * UM, 0.1 * UM, 0.0), radius: 0.25 * UM, length: 10.0, axis: w }, medium: "glass".into() });
    sim.structures.push(Structure { name: "blk".into(), shape: Shape::Box { min: lift(w, 0.25 * UM, -0.5 * UM, -1.0), max: lift(w, 0.5 * UM, -0.1 * UM, 1.0) }, medium: "lossy".into() });
    sim.structures.push(Structure { name: "lor".into(), shape: Shape::Box { min: lift(w, 0.2 * UM, 0.2 * UM, -1.0), max: lift(w, 0.6 * UM, 0.5 * UM, 1.0) }, medium: "lor".into() });
    let sig = TimeSignal::gaussian(F0, 0.4 * F0);
    let mut dip = |pos: [f64; 3], c: Component| sim.sources.push(Source::PointDipole { name: c.name().into(), position: pos, polarization: c, signal: sig.clone(), amplitude: 1e-12 });
    if sets[1] {
        dip(lift(w, -0.6 * UM, -0.3 * UM, 0.0), Component::e(w));
    }
    if sets[0] {
        // in-plane electric dipoles: the TE set without magnetic currents
        dip(lift(w, -0.55 * UM, 0.35 * UM, 0.0), Component::e(u));
        dip(lift(w, 0.05 * UM, -0.6 * UM, 0.0), Component::e(v));
    }
    sim
}

/// Step the emulated kernel and the CPU backend; compare every component.
fn compare(sim: &Simulation, label: &str) {
    let plan = compile(sim, &CompileOptions::default()).unwrap();
    assert!(plan.twod.is_some(), "{label}: not a 2-D plan");
    let nsteps = plan.nsteps.min(300);
    let mut cpu = CpuBackend::new(plan.clone()).unwrap();
    let mut emu = Emu::new(plan.clone());
    for n in 0..nsteps {
        forward_step(&mut cpu, n).unwrap();
        emu.step(n);
    }
    let plane = plan.plane();
    let interior = plane..(plan.nxl + 1) * plane;
    let mut peak_all = 0f32;
    for part in 0..emu.parts.len() {
        cpu.select_set(part == 1).unwrap();
        for c in Component::ALL {
            let r = cpu.field(c).unwrap();
            let e = &emu.parts[part].a[c.index()];
            let peak = r[interior.clone()].iter().fold(0f32, |m, x| m.max(x.abs()));
            let diff = r[interior.clone()].iter().zip(&e[interior.clone()]).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let scale = Component::ALL.iter().map(|q| cpu.field(*q).unwrap()[interior.clone()].iter().fold(0f32, |m, x| m.max(x.abs()))).fold(0f32, f32::max);
            assert!(diff <= 2e-5 * scale, "{label}: part {part} {} differs: max |kernel - cpu| = {diff:e}, peak {peak:e} (all components {scale:e})", c.name());
            peak_all = peak_all.max(peak);
        }
    }
    cpu.select_set(false).unwrap();
    assert!(peak_all > 0.0, "{label}: fields are identically zero");
}

#[test]
fn kernel_matches_cpu_all_axes_and_sets() {
    for w in 0..3 {
        for sets in [[true, false], [false, true], [true, true]] {
            compare(&scene(w, pml(), pml(), sets), &format!("PML, invariant axis {w}, sets {sets:?}"));
        }
    }
}

#[test]
fn kernel_matches_cpu_walls() {
    let pm = |lo: Boundary, hi: Boundary| [lo, hi];
    let one = || Boundary::Pml { layers: 8, params: Default::default(), inside: false };
    for w in 0..3 {
        // PEC / PMC walls and mixed PML
        compare(&scene(w, pm(Boundary::Pec, Boundary::Pmc), pm(Boundary::Pmc, Boundary::Pec), [true, true]), &format!("PEC/PMC, invariant axis {w}"));
        compare(&scene(w, pm(Boundary::Pmc, one()), pm(Boundary::Pec, one()), [true, true]), &format!("PMC/PML + PEC/PML, invariant axis {w}"));
        compare(&scene(w, pm(one(), Boundary::Pec), pm(one(), Boundary::Pmc), [true, true]), &format!("PML/PEC + PML/PMC, invariant axis {w}"));
        // periodic along v (inline wrap)
        compare(&scene(w, pml(), pm(Boundary::Periodic, Boundary::Periodic), [true, true]), &format!("periodic v, invariant axis {w}"));
    }
    // periodic along u is inline only when u is not x (x keeps its ghost planes)
    compare(&scene(0, [Boundary::Periodic, Boundary::Periodic], pml(), [true, true]), "periodic u (y), invariant axis x");
    compare(&scene(0, [Boundary::Periodic, Boundary::Periodic], [Boundary::Periodic, Boundary::Periodic], [true, true]), "periodic u and v, invariant axis x");
}

#[test]
fn kernel_matches_cpu_bloch() {
    let k = 0.3 * std::f64::consts::TAU * F0 / 299_792_458.0;
    let bl = |f: f64| [Boundary::Bloch { k: f * k }, Boundary::Bloch { k: f * k }];
    for w in 0..3 {
        compare(&scene(w, pml(), bl(1.0), [true, true]), &format!("Bloch v, invariant axis {w}"));
    }
    // Bloch along u when u is y
    compare(&scene(0, bl(-0.7), pml(), [true, true]), "Bloch u (y), invariant axis x");
    compare(&scene(0, bl(-0.7), bl(1.0), [true, true]), "Bloch u and v, invariant axis x");
    compare(&scene(0, bl(0.5), [Boundary::Pec, Boundary::Pmc], [false, true]), "Bloch u, walls v, TM, invariant axis x");
}
