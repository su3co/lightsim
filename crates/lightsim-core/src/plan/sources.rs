//! Source injection tables: dipoles, planar sources (plane waves, Gaussian
//! beams, modes) as per-cell amplitude tables, TFSF boxes driven by an
//! auxiliary 1-D grid.

use super::materials::Materials;
use super::*;
use crate::sources::ModeProfile;

/// The injection tables of every source of the simulation, and the field
/// sets of a 2-D plan they drive.
pub(super) fn compile_sources(plan: &mut Plan, s: &Setup, mat: &Materials) -> Result<()> {
    let (sim, b, inv) = (s.sim, &s.sim.boundaries, s.inv);
    let ctx = SourceCtx {
        grid: s.grid,
        plan_x0: s.x0,
        nxl: s.nxl,
        ny: s.ny,
        nz: s.nz,
        dt: s.dt,
        nsteps: s.nsteps,
        eps_local: &mat.eps_local,
        mu_local: mat.mu_local.as_ref(),
        cb_local: &mat.cb_local,
        ch_scalar: s.cdt(),
        inv_p_full: &s.inv_p_full,
        inv_d_full: &s.inv_d_full,
        pml_n: [
            [b.x[0].pml_layers(), b.x[1].pml_layers()],
            [b.y[0].pml_layers(), b.y[1].pml_layers()],
            [b.z[0].pml_layers(), b.z[1].pml_layers()],
        ],
        media: &s.media,
        structs: &s.structures,
        background: s.background,
        periodic: s.periodic,
        slice: plan.slice,
    };
    for src in &sim.sources {
        let (mut h, mut e, complex) = compile_source(&ctx, src)?;
        // a 2-D simulation restricted to one field set has no place for the
        // other set's source terms
        if let Some(a) = inv {
            let sets = sim.polarization.sets();
            let keep = |j: &Injection| sets[if crate::spec::is_te(j.comp, a) { 0 } else { 1 }];
            h.retain(keep);
            e.retain(keep);
            let drives = source_sets(src, a);
            if !(drives[0] && sets[0] || drives[1] && sets[1]) {
                let w = ["x", "y", "z"][a];
                bail!(
                    "source '{}' drives only the {} field set, which this 2-D simulation (invariant along {w}, polarization = \"{}\") does not contain",
                    src.name(),
                    if sets[0] { "tm (E along the invariant axis)" } else { "te (E in the plane)" },
                    sim.polarization.name()
                );
            }
        }
        plan.inj_h.extend(h);
        plan.inj_e.extend(e);
        plan.source_analytic.push(complex && plan.bloch);
        // the sets in use follow from the source itself, not from its entries
        // in this slab: every slab of a decomposed run must update the same sets
        if let Some(td) = plan.twod.as_mut() {
            let drives = source_sets(src, td.axis);
            for k in 0..2 {
                td.active[k] |= drives[k] && td.allow[k];
            }
        }
    }
    Ok(())
}

struct SourceCtx<'a> {
    grid: &'a Grid,
    plan_x0: usize,
    nxl: usize,
    ny: usize,
    nz: usize,
    dt: f64,
    nsteps: usize,
    eps_local: &'a [Vec<Real>; 3],
    mu_local: Option<&'a [Vec<Real>; 3]>,
    cb_local: &'a [Vec<Real>; 3],
    ch_scalar: Real,
    inv_p_full: &'a [Vec<Real>; 3],
    inv_d_full: &'a [Vec<Real>; 3],
    /// PML thickness (cells) on the low/high side of each axis.
    pml_n: [[usize; 2]; 3],
    media: &'a [Medium],
    structs: &'a [crate::raster::ResolvedStructure<'a>],
    background: usize,
    periodic: [bool; 3],
    /// 2-D simulations: (invariant axis, slice plane).
    slice: Option<(usize, f64)>,
}

impl SourceCtx<'_> {
    /// Medium at a point (last structure containing it, else the background):
    /// slab-independent, so every slab of a decomposed run agrees.
    fn medium_at(&self, mut x: [f64; 3]) -> &Medium {
        if let Some((a, w)) = self.slice {
            x[a] = w;
        }
        let mut m = self.background;
        for st in self.structs {
            if st.shape.contains(x) {
                m = st.medium;
            }
        }
        &self.media[m]
    }

    /// True if global cell index `g` lies inside a PML layer. Planar source
    /// entries there are dropped: injecting into an absorber is unphysical
    /// (Lumerical refuses it) and the guided-mode fields are negligible.
    fn in_pml(&self, g: [usize; 3]) -> bool {
        // one-cell margin so that no source entry (nor the discrete curl of
        // an H entry, used by the fused GPU step) reaches into the PML
        let n = self.grid.n();
        (0..3).any(|a| {
            (self.pml_n[a][0] > 0 && g[a] <= self.pml_n[a][0]) || (self.pml_n[a][1] > 0 && g[a] + self.pml_n[a][1] + 1 >= n[a])
        })
    }
    fn local(&self, gi: usize) -> Option<usize> {
        if gi >= self.plan_x0 && gi < self.plan_x0 + self.nxl { Some(gi - self.plan_x0) } else { None }
    }
    fn lidx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }
    fn sidx(&self, i: usize, j: usize, k: usize) -> u64 {
        (((i + 1) * self.ny + j) * self.nz + k) as u64
    }
    fn cb(&self, a: usize, i: usize, j: usize, k: usize) -> Real {
        self.cb_local[a][self.lidx(i, j, k)]
    }
    fn ch(&self, a: usize, i: usize, j: usize, k: usize) -> Real {
        match self.mu_local {
            Some(m) => self.ch_scalar / m[a][self.lidx(i, j, k)],
            None => self.ch_scalar,
        }
    }
    fn eps(&self, a: usize, i: usize, j: usize, k: usize) -> Real {
        self.eps_local[a][self.lidx(i, j, k)]
    }
    fn mu(&self, a: usize, i: usize, j: usize, k: usize) -> Real {
        match self.mu_local {
            Some(m) => m[a][self.lidx(i, j, k)],
            None => 1.0,
        }
    }
    fn table(&self, f: impl Fn(f64) -> f64, t_of_n: impl Fn(usize) -> f64) -> Vec<Real> {
        (0..self.nsteps).map(|n| f(t_of_n(n)) as Real).collect()
    }
}

/// Accumulates injection entries per component, merging duplicate indices.
#[derive(Default)]
struct InjBuilder {
    // comp -> (idx -> (coef_re, coef_im, group))
    map: BTreeMap<(Component, u32), BTreeMap<u64, (f64, f64)>>,
}

impl InjBuilder {
    fn add(&mut self, comp: Component, group: u32, idx: u64, re: f64, im: f64) {
        let e = self.map.entry((comp, group)).or_default().entry(idx).or_insert((0.0, 0.0));
        e.0 += re;
        e.1 += im;
    }
    /// Build injections. `amps(group)` returns the amplitude table for that group
    /// (real: nsteps; complex: 2*nsteps interleaved).
    fn build(self, complex: bool, ngroups: usize, amps: &dyn Fn(u32) -> Vec<Real>, nsteps: usize) -> Vec<Injection> {
        // merge groups per component into one injection with group ids
        let mut per_comp: BTreeMap<Component, Vec<(u64, f64, f64, u32)>> = BTreeMap::new();
        for ((c, g), m) in self.map {
            for (idx, (re, im)) in m {
                if re == 0.0 && im == 0.0 {
                    continue;
                }
                per_comp.entry(c).or_default().push((idx, re, im, g));
            }
        }
        let width = if complex { 2 } else { 1 };
        let tables: Vec<Vec<Real>> = (0..ngroups as u32).map(amps).collect();
        let mut table = vec![0 as Real; nsteps * ngroups * width];
        for n in 0..nsteps {
            for g in 0..ngroups {
                for w in 0..width {
                    table[(n * ngroups + g) * width + w] = tables[g][n * width + w];
                }
            }
        }
        per_comp
            .into_iter()
            .map(|(c, v)| Injection {
                comp: c,
                idx: v.iter().map(|e| e.0).collect(),
                coef: v.iter().map(|e| e.1 as Real).collect(),
                coef_im: if complex { Some(v.iter().map(|e| e.2 as Real).collect()) } else { None },
                group: v.iter().map(|e| e.3).collect(),
                ngroups,
                amps: table.clone(),
            })
            .collect()
    }
}

/// Transverse profile for a planar (TFSF-style) source.
struct PlanarProfile<'a> {
    /// E_b, E_c at plane p (complex) as function of (global j-ish indices).
    e: Box<dyn Fn(usize, [usize; 3]) -> Complex64 + 'a>,
    /// H~_b, H~_c (normalised eta0*H) at plane p-1/2.
    h: Box<dyn Fn(usize, [usize; 3]) -> Complex64 + 'a>,
    complex: bool,
    /// Effective index for the E/H half-cell delay.
    n_delay: f64,
    /// Transverse phase-delay function (s) for grouped/oblique delays: None.
    signal: TimeSignal,
    amplitude: f64,
    /// Incident-field amplitude tables from a 1-D auxiliary line (dispersive
    /// backgrounds): (E_inc at plane p at t_n, H~_inc / (n/mu) at p-1/2 at
    /// t_{n+1/2}). Replace the analytic signal when present.
    aux_amps: Option<(Vec<Real>, Vec<Real>)>,
}

/// Compile a one-sided TFSF plane at global node index `p` along `a`.
fn planar_injections(
    ctx: &SourceCtx,
    a: usize,
    p: usize,
    dir: Direction,
    tlo: [usize; 3],
    thi: [usize; 3],
    prof: &PlanarProfile,
) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    if p == 0 {
        bail!("planar source plane must not sit on the low boundary node");
    }
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let sgn = dir.sign();
    let mut hb = InjBuilder::default();
    let mut eb = InjBuilder::default();
    let grid = ctx.grid;
    // index ranges for each component in transverse axes
    let rng = |c: Component, ax: usize| -> (usize, usize) {
        let n_ax = grid.n()[ax];
        let lo = tlo[ax];
        let hi = thi[ax].min(n_ax);
        if c.is_half(ax) { (lo, hi.max(lo + 1).min(n_ax)) } else { (lo, (hi + 1).min(n_ax)) }
    };
    let id_d = ctx.inv_d_full[a][p] as f64;
    let id_p = ctx.inv_p_full[a][p - 1] as f64;
    // Iterate E_b and E_c at plane p, H_b and H_c at p-1.
    for (which, comp_axis) in [(0usize, bax), (1usize, cax)] {
        let ec = Component::e(comp_axis);
        let hc = Component::h(comp_axis);
        // E corrections: E_b += +s cb id H_c ; E_c += -s cb id H_b
        {
            let (r1lo, r1hi) = rng(ec, bax);
            let (r2lo, r2hi) = rng(ec, cax);
            for u in r1lo..r1hi {
                for v in r2lo..r2hi {
                    let mut g = [0usize; 3];
                    g[a] = p;
                    g[bax] = u;
                    g[cax] = v;
                    if ctx.in_pml(g) {
                        continue;
                    }
                    let Some(li) = ctx.local(g[0]) else { continue };
                    let other = if which == 0 { 1 } else { 0 }; // H_c for E_b, H_b for E_c
                    let hval = (prof.h)(other, g);
                    let sign_c = if which == 0 { 1.0 } else { -1.0 };
                    let cbv = ctx.cb(comp_axis, li, g[1], g[2]) as f64;
                    let coef = Complex64::new(sgn * sign_c * cbv * id_d, 0.0) * hval;
                    let idx = ctx.sidx(li, g[1], g[2]);
                    // amp is the analytic signal A; field += Re(coef * A) = re*Re A - im*Im A
                    eb.add(ec, 0, idx, coef.re, -coef.im);
                }
            }
        }
        // H corrections at p-1: H_b += -s ch id E_c ; H_c += +s ch id E_b
        {
            let (r1lo, r1hi) = rng(hc, bax);
            let (r2lo, r2hi) = rng(hc, cax);
            for u in r1lo..r1hi {
                for v in r2lo..r2hi {
                    let mut g = [0usize; 3];
                    g[a] = p - 1;
                    g[bax] = u;
                    g[cax] = v;
                    if ctx.in_pml(g) {
                        continue;
                    }
                    let Some(li) = ctx.local(g[0]) else { continue };
                    let other = if which == 0 { 1 } else { 0 };
                    let mut ge = g;
                    ge[a] = p;
                    let eval = (prof.e)(other, ge);
                    let sign_c = if which == 0 { -1.0 } else { 1.0 };
                    let chv = ctx.ch(comp_axis, li, g[1], g[2]) as f64;
                    let coef = Complex64::new(sgn * sign_c * chv * id_p, 0.0) * eval;
                    let idx = ctx.sidx(li, g[1], g[2]);
                    hb.add(hc, 0, idx, coef.re, -coef.im);
                }
            }
        }
    }
    // Delays: E_inc at plane p (t_n); H_inc at p-1/2 (t_{n+1/2}).
    let dt = ctx.dt;
    let delta = grid.node(a, p) - grid.center(a, p - 1);
    // The H plane sits half a cell below the E plane: upstream for +dir (the
    // wave arrives earlier), downstream for -dir.
    let tau_h = -sgn * delta * prof.n_delay / C0;
    let sig = prof.signal.clone();
    let amp = prof.amplitude;
    let complex = prof.complex;
    let nsteps = ctx.nsteps;
    let mk = |t_of_n: &dyn Fn(usize) -> f64| -> Vec<Real> {
        if complex {
            let mut v = Vec::with_capacity(2 * nsteps);
            for n in 0..nsteps {
                let z = sig.analytic(t_of_n(n)) * amp;
                v.push(z.re as Real);
                v.push(z.im as Real);
            }
            v
        } else {
            (0..nsteps).map(|n| (amp * sig.value(t_of_n(n))) as Real).collect()
        }
    };
    // For +dir, the H plane (upstream) sees the wave earlier: s(t + delta n/c).
    let (h_amps, e_amps) = match &prof.aux_amps {
        Some((ha, ea)) => (ha.clone(), ea.clone()),
        None => (
            mk(&|n| (n as f64) * dt),               // E_inc used by H corrections at t_n
            mk(&|n| (n as f64 + 0.5) * dt - tau_h), // H_inc used by E corrections at t_{n+1/2}
        ),
    };
    let _ = sgn;
    let inj_h = hb.build(complex, 1, &|_| h_amps.clone(), nsteps);
    let inj_e = eb.build(complex, 1, &|_| e_amps.clone(), nsteps);
    Ok((inj_h, inj_e, complex))
}

/// Injection lists of a source for this slab, and whether the source has a
/// complex field pattern (the same answer on every slab, also on slabs the
/// source does not touch).
fn compile_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    match src {
        Source::PointDipole { .. } => dipole_source(ctx, src),
        Source::PlaneWave { .. } => plane_wave_source(ctx, src),
        Source::GaussianBeam { .. } => gaussian_beam_source(ctx, src),
        Source::Mode { .. } => mode_source(ctx, src),
        Source::Tfsf { min, max, axis, direction, polarization, signal, amplitude, .. } => {
            let (h, e) = compile_tfsf(ctx, min, max, *axis, *direction, *polarization, signal, *amplitude)?;
            Ok((h, e, false))
        }
    }
}

/// A point dipole: one entry in the E or H update.
fn dipole_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let grid = ctx.grid;
    let Source::PointDipole { position, polarization, signal, amplitude, .. } = src else { unreachable!() };
    let c = *polarization;
    let a = c.axis();
    let g = [
        grid.nearest_index(c, 0, position[0]),
        grid.nearest_index(c, 1, position[1]),
        grid.nearest_index(c, 2, position[2]),
    ];
    let Some(li) = ctx.local(g[0]) else { return Ok((vec![], vec![], false)) };
    // discrete-curl volume (dual/primal spacings as in the update), so a
    // source on a symmetry plane matches the full-domain injection
    let mut vol = 1.0;
    for ax in 0..3 {
        let inv = if c.is_half(ax) { ctx.inv_p_full[ax][g[ax]] } else { ctx.inv_d_full[ax][g[ax]] };
        vol /= inv as f64;
    }
    let idx = ctx.sidx(li, g[1], g[2]);
    let mut bld = InjBuilder::default();
    let sig = signal.clone();
    let amp = *amplitude;
    if c.is_e() {
        // dE = -(dt/(eps0 eps_r)) J,  J = I l / V
        let cbv = ctx.cb(a, li, g[1], g[2]) as f64;
        let coef = -(cbv / (C0 * EPS0)) / vol;
        bld.add(c, 0, idx, coef, 0.0);
        let dt = ctx.dt;
        let table = ctx.table(|t| amp * sig.value(t), |n| (n as f64 + 0.5) * dt);
        Ok((vec![], bld.build(false, 1, &|_| table.clone(), ctx.nsteps), false))
    } else {
        // dH~ = -(c dt / mu_r) M,  M = K / V
        let chv = ctx.ch(a, li, g[1], g[2]) as f64;
        let coef = -chv / vol;
        bld.add(c, 0, idx, coef, 0.0);
        let dt = ctx.dt;
        let table = ctx.table(|t| amp * sig.value(t), |n| n as f64 * dt);
        Ok((bld.build(false, 1, &|_| table.clone(), ctx.nsteps), vec![], false))
    }
}

/// A plane wave (normal or oblique, s or p) on a plane of the grid.
fn plane_wave_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let grid = ctx.grid;
    let Source::PlaneWave { axis, position, direction, polarization, signal, amplitude, angle, min, max, .. } = src else { unreachable!() };
    let a = *axis;
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    if polarization.axis() == a {
        bail!("plane wave polarization must be a component tangential to the plane");
    }
    // An H component asks for p-polarisation: H along that axis, E
    // (and the tilt of the wavevector) along the other in-plane axis.
    let h_pol = !polarization.is_e();
    let pol = if h_pol { Component::e(if polarization.axis() == bax { cax } else { bax }) } else { *polarization };
    let p = grid.nearest_node(a, *position).max(1);
    let (tlo, thi) = transverse_box(grid, a, min.as_ref(), max.as_ref());
    // local medium at the plane centre for impedance
    let (nmed, mur) = plane_medium(ctx, a, p, pol.axis());
    let s = direction.sign();
    let th = *angle;
    let k0 = TAU * signal.freq0().unwrap_or(0.0) / C0;
    let q = if pol.axis() == bax { cax } else { bax };
    // 2-D simulations: the k-vector stays in the plane. An E
    // polarisation in the plane is then tilted toward its own axis
    // (p-polarisation: tangential E = A cos(th), H~ = (n/mu) A along
    // the invariant axis) instead of out of the plane.
    // In 3-D an H polarisation does the same: H along the axis q.
    let in_plane_e = ctx.slice.map(|(w, _)| w == q).unwrap_or(false);
    if h_pol && th != 0.0 && ctx.slice.is_some() && !in_plane_e {
        bail!("2-D plane wave: an H polarisation at an angle must be along the invariant axis");
    }
    let p_pol = th != 0.0 && (h_pol || in_plane_e);
    let q = if p_pol { pol.axis() } else { q };
    let (e_amp, h_cos) = if p_pol { (th.cos(), 1.0) } else { (1.0, th.cos()) };
    let q0 = grid.position(Component::e(q), q, (tlo[q] + thi[q]) / 2);
    let complex = th != 0.0;
    let pol_is_b = pol.axis() == bax;
    let prof = PlanarProfile {
        e: Box::new(move |which, g| {
            let on = (which == 0) == pol_is_b;
            if !on {
                return Complex64::new(0.0, 0.0);
            }
            if complex {
                let r = grid.position(Component::e(if which == 0 { bax } else { cax }), q, g[q]);
                Complex64::from_polar(e_amp, k0 * nmed * th.sin() * (r - q0))
            } else {
                Complex64::new(1.0, 0.0)
            }
        }),
        h: Box::new(move |which, g| {
            // H~ = (n/mu) k x E (tangential part): pol b -> H~_c = s n cos(th) A ; pol c -> H~_b = -s n cos A
            let mag = s * nmed / mur * h_cos;
            let val = if pol_is_b {
                if which == 1 { mag } else { 0.0 }
            } else if which == 0 {
                -mag
            } else {
                0.0
            };
            if val == 0.0 {
                return Complex64::new(0.0, 0.0);
            }
            if complex {
                let comp = Component::h(if which == 0 { bax } else { cax });
                let r = grid.position(comp, q, g[q]);
                Complex64::from_polar(val, k0 * nmed * th.sin() * (r - q0))
            } else {
                Complex64::new(val, 0.0)
            }
        }),
        complex,
        n_delay: nmed * th.cos(),
        signal: signal.clone(),
        amplitude: *amplitude,
        aux_amps: None,
    };
    // dispersive background at normal incidence: take the incident E
    // and H from a 1-D auxiliary line with the same ADE poles
    let mut prof = prof;
    if th == 0.0 {
        let ng = grid.n();
        let mut gc = [ng[0] / 2, ng[1] / 2, ng[2] / 2];
        gc[a] = p;
        let pc = [grid.node(0, gc[0]), grid.node(1, gc[1]), grid.node(2, gc[2])];
        let md = ctx.medium_at(pc);
        let pa = pol.axis();
        {
            let poles: Vec<[f64; 3]> =
                if md.is_dispersive() && md.pole_axes[pa] { md.poles.iter().map(|q| q.ade_coefficients(ctx.dt)).collect() } else { vec![] };
            if !poles.is_empty() {
                let r_ref = grid.node(a, p);
                let aux = Aux1d::run(grid, a, s, p, p, nmed, mur, ctx.dt, ctx.nsteps, signal, *amplitude, r_ref, &poles);
                let imp = nmed / mur;
                let h_amps = aux.e_at_node(p);
                let e_amps: Vec<Real> = aux.h_at_half(p - 1).iter().map(|v| v / imp as Real).collect();
                prof.aux_amps = Some((h_amps, e_amps));
            }
        }
    }
    planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
}

/// A focused Gaussian beam crossing a plane of the grid.
fn gaussian_beam_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let grid = ctx.grid;
    let Source::GaussianBeam {
        axis,
        position,
        direction,
        polarization,
        center,
        waist,
        signal,
        amplitude,
        focus_distance,
        theta,
        phi,
        ..
    } = src else { unreachable!() };
    let a = *axis;
    let pol = *polarization;
    if !pol.is_e() || pol.axis() == a {
        bail!("Gaussian beam polarization must be an E component tangential to the plane");
    }
    if !(theta.abs() < 0.5 * std::f64::consts::PI - 1e-6) {
        bail!("Gaussian beam: theta = {theta} rad; the beam axis must make less than 90 degrees with the plane normal");
    }
    let p = grid.nearest_node(a, *position).max(1);
    let (tlo, thi) = transverse_box(grid, a, None, None);
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    let (nmed, mur) = plane_medium(ctx, a, p, pol.axis());
    let s = direction.sign();
    let f0 = signal.freq0().unwrap_or(0.0);
    let k = TAU * f0 * nmed / C0;
    let w0 = *waist;
    let zr = std::f64::consts::PI * w0 * w0 * nmed * f0 / C0;
    let fd = *focus_distance;
    // beam axis d, polarization e (the polarization axis projected
    // onto the plane normal to d) and h = d x e
    let (u_ax, v_ax) = crate::geometry::plane_axes(a);
    let mut d = [0.0f64; 3];
    d[a] = s * theta.cos();
    d[u_ax] = theta.sin() * phi.cos();
    d[v_ax] = theta.sin() * phi.sin();
    let mut e = [0.0f64; 3];
    e[pol.axis()] = 1.0;
    let ed = e[0] * d[0] + e[1] * d[1] + e[2] * d[2];
    for m in 0..3 {
        e[m] -= ed * d[m];
    }
    let en = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
    if en < 1e-6 {
        bail!("Gaussian beam: the beam axis is along the polarization axis");
    }
    for v in e.iter_mut() {
        *v /= en;
    }
    let h = [d[1] * e[2] - d[2] * e[1], d[2] * e[0] - d[0] * e[2], d[0] * e[1] - d[1] * e[0]];
    // a sheet beam along the invariant axis of a 2-D simulation, and along
    // periodic in-plane axes narrower than the beam
    let inv = ctx.slice.map(|(w, _)| w);
    let mut flat = [false; 3];
    for q in [bax, cax] {
        let edges = grid.edges(q);
        flat[q] = inv == Some(q) || (ctx.periodic[q] && (edges[edges.len() - 1] - edges[0]) < 2.0 * w0);
        if flat[q] && d[q].abs() > 1e-9 {
            bail!("Gaussian beam: the simulation is uniform along axis {q} (periodic and narrower than the beam), so the beam axis must have no component along it");
        }
    }
    let sheet = flat[bax] || flat[cax];
    if flat[bax] && flat[cax] {
        bail!("Gaussian beam: the source plane is narrower than the beam along both in-plane axes; use a plane wave");
    }
    let complex = fd != 0.0 || *theta != 0.0;
    let mut ctr = *center;
    ctr[a] = grid.node(a, p);
    let field = move |comp: Component, g: [usize; 3]| -> Complex64 {
        // offset from the beam centre on the plane (zero along a uniform axis)
        let mut rho = [0.0f64; 3];
        for m in [bax, cax] {
            if !flat[m] {
                rho[m] = grid.position(comp, m, g[m]) - ctr[m];
            }
        }
        let along = rho[0] * d[0] + rho[1] * d[1] + rho[2] * d[2];
        let r2 = (rho[0] * rho[0] + rho[1] * rho[1] + rho[2] * rho[2] - along * along).max(0.0);
        let z = along - fd; // position along the beam axis, from the focus
        let wz = w0 * (1.0 + (z / zr).powi(2)).sqrt();
        let inv_r = z / (z * z + zr * zr);
        let gouy = (z / zr).atan();
        // a sheet beam spreads in one direction only
        let (amp, gouy) = if sheet { ((w0 / wz).sqrt() * (-r2 / (wz * wz)).exp(), 0.5 * gouy) } else { (w0 / wz * (-r2 / (wz * wz)).exp(), gouy) };
        if !complex {
            return Complex64::new(amp, 0.0);
        }
        // exp(-i w t) convention: forward wave exp(+i k z); the phase is zero at the beam centre on the plane
        Complex64::from_polar(amp, k * along + k * r2 * inv_r / 2.0 - gouy)
    };
    let prof = PlanarProfile {
        e: Box::new(move |which, g| {
            let m = if which == 0 { bax } else { cax };
            if e[m] == 0.0 {
                return Complex64::new(0.0, 0.0);
            }
            field(Component::e(m), g) * e[m]
        }),
        h: Box::new(move |which, g| {
            let m = if which == 0 { bax } else { cax };
            if h[m] == 0.0 {
                return Complex64::new(0.0, 0.0);
            }
            field(Component::h(m), g) * (h[m] * nmed / mur)
        }),
        complex,
        // the half cell between the E and H planes, along the normal
        n_delay: nmed * theta.cos(),
        signal: signal.clone(),
        amplitude: *amplitude,
        aux_amps: None,
    };
    planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
}

/// A mode source follows the solves of its profile while the pulse is at most
/// this many times wider than their span about the carrier.
pub const MODE_SOURCE_BAND_WIDTHS: f64 = 4.0;

/// An eigenmode profile on a plane of the grid. A profile that carries its
/// mode at other frequencies (`others`) is followed over the band of the
/// pulse: the field launched at frequency `f` is the polynomial through up to
/// three solves, `sum_k w_k(f) P_k`, so each solve `P_k` is injected with the
/// signal whose spectrum is `w_k(f)` times the pulse's (a combination of the
/// envelope and its first two time derivatives on the same carrier).
fn mode_source(ctx: &SourceCtx, src: &Source) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let Source::Mode { profile, signal, .. } = src else { unreachable!() };
    let nodes = profile.source_nodes();
    // The weights of the solves are a parabola of the frequency, and the
    // pulse also holds frequencies outside the solves: there each solve is
    // injected with a multiple of the pulse (4.5 times at a pulse width three
    // times the span of the solves, 30 times at eight) and the sum is what
    // single precision leaves of their difference. Past a pulse four times
    // wider than the span the profile is injected alone: over so narrow a
    // band it is the mode to within the detuning (0.1 % at 1 %).
    let span = nodes.iter().map(|n| (n.freq - signal.freq0().unwrap_or(n.freq)).abs()).fold(0.0, f64::max);
    let f0 = match (signal.freq0(), signal.carrier_width()) {
        (Some(f), Some(width)) if nodes.len() > 1 && signal.carrier_envelope(0.0).is_some() && width <= MODE_SOURCE_BAND_WIDTHS * span => f,
        // one solve, a signal without a carrier, or a pulse far wider than the solves: the profile alone
        _ => return mode_injections(ctx, src, profile, None, None),
    };
    let complex = nodes.iter().any(|n| mode_patterns(ctx, src, n).1);
    let freqs: Vec<f64> = nodes.iter().map(|n| n.freq).collect();
    let (mut inj_h, mut inj_e) = (vec![], vec![]);
    for (k, node) in nodes.iter().enumerate() {
        let (h, e, _) = mode_injections(ctx, src, node, Some(ModeProfile::lagrange_weight(&freqs, k, f0)), Some(complex))?;
        inj_h.extend(h);
        inj_e.extend(e);
    }
    Ok((inj_h, inj_e, complex))
}

/// The H arrays of `profile` brought to the plane of a mode source (see
/// [`mode_injections`]) and whether the pattern has a phase of its own.
#[allow(clippy::type_complexity)]
fn mode_patterns(ctx: &SourceCtx, src: &Source, profile: &ModeProfile) -> ([(Vec<f64>, Vec<f64>); 2], bool) {
    let grid = ctx.grid;
    let Source::Mode { axis, position, direction, .. } = src else { unreachable!() };
    let a = *axis;
    let p = grid.nearest_node(a, *position).max(1);
    let s = direction.sign();
    // The profile stores H half a cell below the plane, for a mode
    // travelling along +axis: H_m(p) exp(-i beta d). The injection
    // delays the H signal in time by that half cell (`tau_h`, for
    // either direction and every frequency of the pulse), so the
    // profile's own shift is taken out here; otherwise the half cell
    // would be counted twice and part of the power would be launched
    // backwards. An absorbing mode keeps its amplitude step over the
    // half cell, exp(+-Im(beta) d).
    let d_h = profile.h_offset.unwrap_or(grid.node(a, p) - grid.center(a, p - 1));
    let beta = profile.beta();
    let unshift = (Complex64::i() * beta * d_h).exp() * (s * beta.im * d_h).exp();
    let fix = |re: &Vec<f64>, im: &Vec<f64>| -> (Vec<f64>, Vec<f64>) {
        let z: Vec<Complex64> = (0..re.len()).map(|o| Complex64::new(re[o], im.get(o).copied().unwrap_or(0.0)) * unshift).collect();
        (z.iter().map(|v| v.re).collect(), z.iter().map(|v| v.im).collect())
    };
    let h_fixed = [fix(&profile.h_re[0], &profile.h_im[0]), fix(&profile.h_re[1], &profile.h_im[1])];
    // complex injection only when the pattern has a phase of its own
    let peak = profile
        .e_re
        .iter()
        .chain(profile.e_im.iter())
        .flat_map(|v| v.iter())
        .fold(0f64, |m, x| m.max(x.abs()))
        .max(h_fixed.iter().flat_map(|(r, i)| r.iter().chain(i.iter())).fold(0f64, |m, x| m.max(x.abs() * ETA0)));
    let complex = profile.e_im.iter().any(|v| v.iter().any(|x| x.abs() > 1e-9 * peak))
        || h_fixed.iter().any(|(_, i)| i.iter().any(|x| x.abs() * ETA0 > 1e-9 * peak));
    (h_fixed, complex)
}

/// The injections of one solve `profile` of a mode source: with the source's
/// own signal (`weight` none), or with the signal whose spectrum is the
/// polynomial `weight` of `f - freq0` times the pulse's.
fn mode_injections(ctx: &SourceCtx, src: &Source, profile: &ModeProfile, weight: Option<[f64; 3]>, complex: Option<bool>) -> Result<(Vec<Injection>, Vec<Injection>, bool)> {
    let grid = ctx.grid;
    let Source::Mode { axis, position, direction, min, max, signal, amplitude, .. } = src else { unreachable!() };
    let a = *axis;
    let p = grid.nearest_node(a, *position).max(1);
    let (tlo, thi) = transverse_box(grid, a, Some(min), Some(max));
    let bax = (a + 1) % 3;
    let cax = (a + 2) % 3;
    // profile arrays are over the plane's natural-order transverse axes
    let (u_ax, v_ax) = crate::geometry::plane_axes(a);
    let dims = profile.dims;
    let s = direction.sign();
    // map (which in cyclic (b,c)) -> index in natural (u,v) order
    let nat = |cyclic_axis: usize| -> usize { if cyclic_axis == u_ax { 0 } else { 1 } };
    let lo = tlo;
    let get = move |arr_re: &Vec<f64>, arr_im: &Vec<f64>, g: [usize; 3]| -> Complex64 {
        let iu = g[u_ax] as isize - lo[u_ax] as isize;
        let iv = g[v_ax] as isize - lo[v_ax] as isize;
        if iu < 0 || iv < 0 || iu as usize >= dims[0] || iv as usize >= dims[1] {
            return Complex64::new(0.0, 0.0);
        }
        let o = iu as usize * dims[1] + iv as usize;
        Complex64::new(arr_re[o], arr_im.get(o).copied().unwrap_or(0.0))
    };
    let (h_fixed, own_phase) = mode_patterns(ctx, src, profile);
    let complex = complex.unwrap_or(own_phase);
    // the tables of a weighted solve: E_inc at t_n for the H corrections,
    // H_inc at t_{n+1/2}, delayed by the half cell, for the E corrections
    let aux_amps = weight.map(|c| {
        let tau_h = -s * (grid.node(a, p) - grid.center(a, p - 1)) * profile.neff / C0;
        let table = |t_of_n: &dyn Fn(usize) -> f64| -> Vec<Real> {
            let mut v = Vec::with_capacity(ctx.nsteps * if complex { 2 } else { 1 });
            for n in 0..ctx.nsteps {
                let z = signal.analytic_times_polynomial(t_of_n(n), c).expect("a signal with a carrier") * *amplitude;
                v.push(z.re as Real);
                if complex {
                    v.push(z.im as Real);
                }
            }
            v
        };
        (table(&|n| (n as f64) * ctx.dt), table(&|n| (n as f64 + 0.5) * ctx.dt - tau_h))
    });
    let pr = profile.clone();
    let prof = PlanarProfile {
        e: Box::new(move |which, g| {
            let ax = if which == 0 { bax } else { cax };
            let t = nat(ax);
            get(&pr.e_re[t], &pr.e_im[t], g)
        }),
        h: Box::new(move |which, g| {
            let ax = if which == 0 { bax } else { cax };
            let t = nat(ax);
            // backward injection flips H; convert to H~ = eta0 H
            get(&h_fixed[t].0, &h_fixed[t].1, g) * (s * ETA0)
        }),
        complex,
        n_delay: profile.neff,
        signal: signal.clone(),
        amplitude: *amplitude,
        aux_amps,
    };
    planar_injections(ctx, a, p, *direction, tlo, thi, &prof)
}

/// `[te, tm]`: the field sets of a 2-D problem invariant along `w` that a
/// source drives (see [`TwoD`]).
pub fn source_sets(src: &Source, w: usize) -> [bool; 2] {
    let of = |c: Component| if crate::spec::is_te(c, w) { [true, false] } else { [false, true] };
    match src {
        Source::PointDipole { polarization, .. } | Source::PlaneWave { polarization, .. } | Source::GaussianBeam { polarization, .. } | Source::Tfsf { polarization, .. } => of(*polarization),
        Source::Mode { axis, profile, .. } => {
            // tangential E and H of the profile, over the plane's natural axes
            let (ua, va) = crate::geometry::plane_axes(*axis);
            let mut out = [false, false];
            let nz = |re: &Vec<f64>, im: &Vec<f64>| re.iter().chain(im.iter()).any(|x| *x != 0.0);
            for (t, ax) in [ua, va].into_iter().enumerate() {
                if nz(&profile.e_re[t], &profile.e_im[t]) {
                    let k = of(Component::e(ax));
                    out = [out[0] || k[0], out[1] || k[1]];
                }
                if nz(&profile.h_re[t], &profile.h_im[t]) {
                    let k = of(Component::h(ax));
                    out = [out[0] || k[0], out[1] || k[1]];
                }
            }
            out
        }
    }
}

/// Transverse index box for a plane normal to `a` (nodes); defaults to all.
pub fn transverse_box(grid: &Grid, a: usize, min: Option<&[f64; 3]>, max: Option<&[f64; 3]>) -> ([usize; 3], [usize; 3]) {
    let n = grid.n();
    let mut lo = [0usize; 3];
    let mut hi = n;
    for ax in 0..3 {
        if ax == a {
            continue;
        }
        if let (Some(mn), Some(mx)) = (min, max) {
            // A window edge that lies on a grid node (to rounding) is that
            // node: the window starts with the cell above it and ends with
            // the cell below it. Without the tolerance the last bit of the
            // coordinate decided whether a neighbouring cell was included.
            let e = grid.edges(ax);
            let tol = 1e-9 * (e[n[ax]] - e[0]).abs().max(1e-30);
            lo[ax] = grid.cell_index(ax, (mn[ax] + tol).max(e[0]));
            hi[ax] = (grid.cell_index(ax, (mx[ax] - tol).min(e[n[ax]])) + 1).min(n[ax]).max(lo[ax] + 1);
        }
    }
    (lo, hi)
}

/// Refractive index and mu_r at the centre of the injection plane.
fn plane_medium(ctx: &SourceCtx, a: usize, p: usize, pol_axis: usize) -> (f64, f64) {
    let n = ctx.grid.n();
    let mut g = [n[0] / 2, n[1] / 2, n[2] / 2];
    g[a] = p;
    // pick a slab-local x if needed
    let li = match ctx.local(g[0]) {
        Some(v) => v,
        None => {
            if a == 0 {
                return (1.0, 1.0);
            }
            ctx.nxl / 2
        }
    };
    let eps = ctx.eps(pol_axis, li, g[1], g[2]) as f64;
    let mu = ctx.mu(pol_axis, li, g[1], g[2]) as f64;
    ((eps * mu).sqrt(), mu)
}

#[allow(clippy::too_many_arguments)]
fn compile_tfsf(
    ctx: &SourceCtx,
    min: &[f64; 3],
    max: &[f64; 3],
    a: usize,
    dir: Direction,
    pol: Component,
    signal: &TimeSignal,
    amplitude: f64,
) -> Result<(Vec<Injection>, Vec<Injection>)> {
    if !pol.is_e() || pol.axis() == a {
        bail!("TFSF polarization must be an E component perpendicular to the propagation axis");
    }
    let grid = ctx.grid;
    let n = grid.n();
    // node index box of the TF region; a transverse axis that is periodic and
    // fully spanned by the box has no faces (a periodic "slab" TFSF, e.g. 2-D
    // scattering problems)
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    let mut span = [false; 3];
    for ax in 0..3 {
        let e = grid.edges(ax);
        span[ax] = ax != a && ctx.periodic[ax] && min[ax] <= e[0] + 1e-15 && max[ax] >= e[n[ax]] - 1e-15;
        if span[ax] {
            lo[ax] = 0;
            hi[ax] = n[ax];
            continue;
        }
        lo[ax] = grid.nearest_node(ax, min[ax]).max(1);
        hi[ax] = grid.nearest_node(ax, max[ax]).min(n[ax] - 1);
        if hi[ax] <= lo[ax] {
            bail!("TFSF box too small along axis {ax}");
        }
    }
    let p_ax = pol.axis();
    let s = dir.sign();
    // background medium at the box centre
    let mut gc = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    gc[a] = lo[a];
    // background medium at the box face centre, looked up geometrically so
    // that every slab of a decomposed run builds the same incident wave
    let pc = [grid.node(0, gc[0]), grid.node(1, gc[1]), grid.node(2, gc[2])];
    let md = ctx.medium_at(pc);
    let (nmed, mur) = ((md.eps[p_ax] * md.mu[p_ax]).sqrt(), md.mu[p_ax]);
    let poles: Vec<[f64; 3]> =
        if md.is_dispersive() && md.pole_axes[p_ax] { md.poles.iter().map(|q| q.ade_coefficients(ctx.dt)).collect() } else { vec![] };
    // incident E = A e_pol s(t - (r_a - r_ref) s n / c); H~ = s (n/mu) (a_hat x e_pol)
    let r_ref = if s > 0.0 { grid.node(a, lo[a]) } else { grid.node(a, hi[a]) };
    // H~ direction: a_hat x e_pol
    let mut hdir = [0.0f64; 3];
    {
        let mut av = [0.0; 3];
        av[a] = 1.0;
        let mut ev = [0.0; 3];
        ev[p_ax] = 1.0;
        hdir[0] = av[1] * ev[2] - av[2] * ev[1];
        hdir[1] = av[2] * ev[0] - av[0] * ev[2];
        hdir[2] = av[0] * ev[1] - av[1] * ev[0];
    }
    // ---- 1-D auxiliary FDTD line for the incident field ----
    // The incident wave is propagated on a 1-D Yee line with the same spacing
    // along `a` and the same time step, so its numerical dispersion matches
    // the 3-D grid exactly (no leakage into the scattered-field region).
    let aux = Aux1d::run(grid, a, s, lo[a], hi[a], nmed, mur, ctx.dt, ctx.nsteps, signal, amplitude, r_ref, &poles);
    let mut hb = InjBuilder::default();
    let mut eb = InjBuilder::default();
    let mut e_groups: Vec<usize> = vec![]; // half index j along a (H_inc position)
    let mut h_groups: Vec<usize> = vec![]; // node index i along a (E_inc position)
    let gid = |list: &mut Vec<usize>, key: usize| -> u32 {
        match list.iter().position(|&k| k == key) {
            Some(p) => p as u32,
            None => {
                list.push(key);
                (list.len() - 1) as u32
            }
        }
    };

    for f in 0..3 {
        if span[f] {
            continue;
        }
        let g_ax = (f + 1) % 3;
        let h_ax = (f + 2) % 3;
        for side in 0..2 {
            let q = if side == 0 { lo[f] } else { hi[f] };
            let sside = if side == 0 { 1.0 } else { -1.0 };
            // H plane index adjacent outside: lo -> q-1 (pos q-1/2), hi -> q (pos q+1/2)
            let qh = if side == 0 { q - 1 } else { q };
            let id_d = ctx.inv_d_full[f][q] as f64;
            let id_p = ctx.inv_p_full[f][qh] as f64;
            for (which, comp_ax) in [(0usize, g_ax), (1usize, h_ax)] {
                let other_ax = if which == 0 { h_ax } else { g_ax };
                let r = |ax: usize, c: Component| -> (usize, usize) {
                    if span[ax] {
                        (0, n[ax])
                    } else if c.is_half(ax) {
                        (lo[ax], hi[ax])
                    } else {
                        (lo[ax], hi[ax] + 1)
                    }
                };
                // ---- E corrections on the face (TF E nodes): need H~_inc(other) ----
                let hdir_v = hdir[other_ax] * s; // sign of the physical H~ component along other_ax
                if hdir_v != 0.0 {
                    let ec = Component::e(comp_ax);
                    let (u0, u1) = r(g_ax, ec);
                    let (v0, v1) = r(h_ax, ec);
                    for u in u0..u1 {
                        for v in v0..v1 {
                            let mut gg = [0usize; 3];
                            gg[f] = q;
                            gg[g_ax] = u;
                            gg[h_ax] = v;
                            let Some(li) = ctx.local(gg[0]) else { continue };
                            let sign_c = if which == 0 { 1.0 } else { -1.0 };
                            let cbv = ctx.cb(comp_ax, li, gg[1], gg[2]) as f64;
                            let coef = sside * sign_c * cbv * id_d * hdir_v;
                            let j = if f == a { qh } else { gg[a] };
                            let g = gid(&mut e_groups, j);
                            eb.add(ec, g, ctx.sidx(li, gg[1], gg[2]), coef, 0.0);
                        }
                    }
                }
                // ---- H corrections just outside (SF H nodes): need E_inc(other) ----
                if other_ax == p_ax {
                    let hc = Component::h(comp_ax);
                    let (u0, u1) = r(g_ax, hc);
                    let (v0, v1) = r(h_ax, hc);
                    for u in u0..u1 {
                        for v in v0..v1 {
                            let mut gg = [0usize; 3];
                            gg[f] = qh;
                            gg[g_ax] = u;
                            gg[h_ax] = v;
                            let Some(li) = ctx.local(gg[0]) else { continue };
                            let sign_c = if which == 0 { -1.0 } else { 1.0 };
                            let chv = ctx.ch(comp_ax, li, gg[1], gg[2]) as f64;
                            let coef = sside * sign_c * chv * id_p;
                            let i = if f == a { q } else { gg[a] };
                            let g = gid(&mut h_groups, i);
                            hb.add(hc, g, ctx.sidx(li, gg[1], gg[2]), coef, 0.0);
                        }
                    }
                }
            }
        }
    }
    let nsteps = ctx.nsteps;
    let amp_h = |g: u32| -> Vec<Real> { aux.e_at_node(h_groups[g as usize]) };
    let amp_e = |g: u32| -> Vec<Real> { aux.h_at_half(e_groups[g as usize]) };
    let inj_h = hb.build(false, h_groups.len().max(1), &amp_h, nsteps);
    let inj_e = eb.build(false, e_groups.len().max(1), &amp_e, nsteps);
    Ok((inj_h, inj_e))
}

/// 1-D auxiliary Yee line carrying the incident plane wave of a TFSF box.
///
/// Scalar fields: `e` (E along the polarization) at the 3-D grid's nodes
/// along the propagation axis and `h` (normalised H~ along a x e_pol, signed
/// so that a forward wave has h = (n/mu) e) at the half nodes. A hard source
/// drives the upstream end, a graded matched absorber terminates the
/// downstream end. Samples: e at t_n (used by the H corrections of step n),
/// h at t_{n+1/2} (used by the E corrections of step n).
struct Aux1d {
    dir: f64,
    g0: usize,
    g1: usize,
    nsteps: usize,
    /// e samples [k][n] for k = aux node index of the needed global nodes
    e_samp: Vec<Vec<Real>>,
    h_samp: Vec<Vec<Real>>,
}

impl Aux1d {
    #[allow(clippy::too_many_arguments)]
    fn run(
        grid: &Grid,
        a: usize,
        dir: f64,
        lo: usize,
        hi: usize,
        nmed: f64,
        mur: f64,
        dt: f64,
        nsteps: usize,
        signal: &TimeSignal,
        amplitude: f64,
        r_ref: f64,
        poles: &[[f64; 3]],
    ) -> Aux1d {
        let e = grid.edges(a);
        let na = e.len() - 1;
        let g0 = lo.saturating_sub(2);
        let g1 = (hi + 2).min(na);
        // aux node coordinates in the propagation frame
        let mut xs: Vec<f64> = if dir > 0.0 {
            (g0..=g1).map(|i| e[i]).collect()
        } else {
            (0..=(g1 - g0)).map(|k| -e[g1 - k]).collect()
        };
        let dlast = xs[xs.len() - 1] - xs[xs.len() - 2];
        let n_abs = 80usize;
        let n_buf = 20usize;
        let last = *xs.last().unwrap();
        xs.extend((1..=(n_buf + n_abs)).map(|t| last + t as f64 * dlast));
        let nn = xs.len(); // nodes
        let nh = nn - 1; // half nodes
        let eps_r = nmed * nmed / mur;
        let cdt = C0 * dt;
        // spacings
        let dxp: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let dxd: Vec<f64> = (0..nn).map(|k| if k == 0 { dxp[0] } else if k == nh { dxp[nh - 1] } else { 0.5 * (dxp[k - 1] + dxp[k]) }).collect();
        // graded loss in the last n_abs cells
        let abs_start = xs[nn - 1 - n_abs];
        let lab = xs[nn - 1] - abs_start;
        let alpha = |x: f64| -> f64 {
            if x <= abs_start { 0.0 } else { 0.35 * ((x - abs_start) / lab).powi(3) }
        };
        let ce: Vec<(f64, f64)> = (0..nn)
            .map(|k| {
                let al = alpha(xs[k]);
                ((1.0 - al) / (1.0 + al), cdt / eps_r / (1.0 + al))
            })
            .collect();
        let chh: Vec<(f64, f64)> = (0..nh)
            .map(|k| {
                let al = alpha(0.5 * (xs[k] + xs[k + 1]));
                ((1.0 - al) / (1.0 + al), cdt / mur / (1.0 + al))
            })
            .collect();
        // source delay so that e(r_ref, t) ~ s(t)
        let x_ref = if dir > 0.0 { r_ref } else { -r_ref };
        let tau0 = (x_ref - xs[0]) * nmed / C0;
        let src = |t: f64| amplitude * signal.value(t + tau0);
        let ncore = g1 - g0 + 1;
        let mut ev = vec![0.0f64; nn];
        let mut hv = vec![0.0f64; nh];
        // dispersive background: the same ADE as the 3-D grid (P^{n+1} from
        // E^n, E -= dP / eps_inf), so the incident wave obeys the identical
        // discrete dispersion relation
        let np = poles.len();
        let mut pp = vec![0.0f64; np * nn];
        let mut ppv = vec![0.0f64; np * nn];
        let mut dp = vec![0.0f64; nn];
        let mut e_samp = vec![Vec::with_capacity(nsteps); ncore];
        let mut h_samp = vec![Vec::with_capacity(nsteps); ncore];
        ev[0] = src(0.0);
        for n in 0..nsteps {
            for (k, s) in e_samp.iter_mut().enumerate() {
                s.push(ev[k] as Real);
            }
            for k in 0..nh {
                hv[k] = chh[k].0 * hv[k] - chh[k].1 * (ev[k + 1] - ev[k]) / dxp[k];
            }
            for (k, s) in h_samp.iter_mut().enumerate() {
                s.push(hv[k.min(nh - 1)] as Real);
            }
            if np > 0 {
                for k in 0..nn {
                    let mut acc = 0.0;
                    for (q, c) in poles.iter().enumerate() {
                        let o = q * nn + k;
                        let p0 = pp[o];
                        let pn = c[0] * p0 + c[1] * ppv[o] + c[2] * ev[k];
                        ppv[o] = p0;
                        pp[o] = pn;
                        acc += pn - p0;
                    }
                    dp[k] = acc;
                }
            }
            for k in 1..nn - 1 {
                ev[k] = ce[k].0 * ev[k] - ce[k].1 * (hv[k] - hv[k - 1]) / dxd[k];
                if np > 0 {
                    ev[k] -= dp[k] / eps_r / (1.0 + alpha(xs[k]));
                }
            }
            ev[nn - 1] = 0.0;
            ev[0] = src((n as f64 + 1.0) * dt);
        }
        Aux1d { dir, g0, g1, nsteps, e_samp, h_samp }
    }

    /// e(t_n) at global node index i.
    fn e_at_node(&self, i: usize) -> Vec<Real> {
        let k = if self.dir > 0.0 { i - self.g0 } else { self.g1 - i };
        self.e_samp.get(k).cloned().unwrap_or_else(|| vec![0.0; self.nsteps])
    }

    /// h(t_{n+1/2}) at global half index j (between nodes j and j+1),
    /// returned as the physical H~ component along `a x e_pol` (sign of
    /// the propagation direction included).
    fn h_at_half(&self, j: usize) -> Vec<Real> {
        let k = if self.dir > 0.0 { j - self.g0 } else { self.g1 - j - 1 };
        self.h_samp.get(k).cloned().unwrap_or_else(|| vec![0.0; self.nsteps])
    }
}
