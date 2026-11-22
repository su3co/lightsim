//! Validation of zero-thickness sheets ([`crate::sheet`]).

use super::*;
use crate::sheet::{Sheet, SheetConductivity, SheetPole, sheet_plasmon_q, sheet_rt};

/// A Drude sheet with eta0 |sigma| ~ 1 near f0: `D = 0.8 w0 / eta0`,
/// gamma = 0.2 w0 (in the units of these cases, f0 = c / 1 um).
fn drude_sheet() -> SheetConductivity {
    let w0 = TAU * f0();
    SheetConductivity { name: "drude".into(), poles: vec![SheetPole::Drude { drude_weight: 0.8 * w0 / ETA0, gamma: 0.2 * w0 }], ..Default::default() }
}

/// A sheet with every term: conductance, susceptibility, a Drude and a
/// Lorentz pole (a stand-in for graphene's intra- and interband terms).
fn mixed_sheet() -> SheetConductivity {
    let w0 = TAU * f0();
    SheetConductivity {
        name: "mixed".into(),
        conductance: 0.3 / ETA0,
        susceptibility: 2e-9,
        poles: vec![SheetPole::Drude { drude_weight: 0.5 * w0 / ETA0, gamma: 0.1 * w0 }, SheetPole::Lorentz { strength: 0.15 * w0 / ETA0, omega0: 1.3 * w0, gamma: 0.3 * w0 }],
    }
}

fn full_sheet(axis: usize, position: f64, c: SheetConductivity) -> Sheet {
    Sheet { name: "sheet".into(), axis, position, conductivity: c, min: None, max: None, polygon: None }
}

/// R and T of a sheet at normal incidence in a 1-D column (periodic y, z),
/// incidence medium n1, substrate n2 from the sheet on. `mesh`: the x nodes
/// (None: uniform `dx`). Returns (max |R - R_a|, max |T - T_a|, max |R + T + A - 1|)
/// over the band, A from the sheet current (the DFT of E on the node
/// planes that carry the sheet, times Re sigma).
fn normal_incidence(run: Runner, c: &SheetConductivity, xs: f64, n1: f64, n2: f64, dx: f64, mesh: Option<Vec<f64>>) -> Result<(f64, f64, f64, f64)> {
    let ncross = 2usize;
    let w = ncross as f64 * dx;
    let length = 4e-6;
    let freqs = band(5);
    let mk = |sheet: bool, substrate: bool| {
        let mut sim = column(dx, length, ncross);
        if let Some(x) = &mesh {
            sim.mesh = MeshSpec::Explicit { x: x.clone(), y: (0..=ncross).map(|i| i as f64 * dx).collect(), z: (0..=ncross).map(|i| i as f64 * dx).collect() };
        }
        sim.media.push(Medium::from_index(n1).with_name("n1"));
        sim.background = "n1".into();
        if substrate {
            sim.media.push(Medium::from_index(n2).with_name("n2"));
            sim.structures.push(Structure { name: "sub".into(), shape: Shape::Box { min: [xs, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "n2".into() });
        }
        if sheet {
            sim.sheets.push(full_sheet(0, xs, c.clone()));
        }
        sim.sources.push(plane_source(1.0e-6, n1));
        sim.monitors.push(flux_x("T", 3.2e-6, w, &freqs));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        // E_y on the two node planes around the sheet
        sim.monitors.push(Monitor::DftField { name: "es".into(), min: [xs - 1.01 * dx, 0.0, 0.0], max: [xs + 1.01 * dx, 0.0, 0.0], components: vec![Component::Ey], freqs: freqs.clone(), apodization: Apodization::default(), stride: 1 });
        sim.run_time = 120e-15;
        sim.shutoff = Some(1e-8);
        sim
    };
    let inc = run(&mk(false, false))?;
    let res = run(&mk(true, true))?;
    let sim = mk(true, true);
    let grid = sim.build_grid()?;
    let dt = res.dt;
    let nodes = crate::sheet::node_weights(&grid, 0, xs, false, "sheet")?;
    let es = &res.dft["es"].fields[0];
    let (mut er, mut et, mut ea) = (0.0f64, 0.0f64, 0.0f64);
    for (i, f) in freqs.iter().enumerate() {
        let om = TAU * f;
        let p_inc = inc.flux["T"][i];
        let (ra, ta) = sheet_rt(c.sigma(om), n1, n2, 0.0, false);
        let (r, t) = (-res.flux["R"][i] / p_inc, res.flux["T"][i] / p_inc);
        // absorbed: each node plane carries w_j sigma (the discrete conductivity)
        let sd = c.sigma_discrete(om, dt);
        let mut a = 0.0;
        for &(g, wj) in &nodes {
            let e = es.at(i, g - es.origin[0], 0, 0);
            a += 0.5 * wj * sd.re * e.norm_sqr() * w * w;
        }
        let a = a / p_inc;
        er = er.max((r - ra).abs());
        et = et.max((t - ta).abs());
        ea = ea.max((r + t + a - 1.0).abs());
    }
    Ok((er, et, ea, res.stats.mcells_per_s))
}

/// Zero-thickness sheet at normal incidence, between two dielectrics
/// (n1 = 1.4 on the incidence side, n2 = 2 behind the sheet), against the
/// analytic sheet formulas, for a Drude sheet and for a sheet with every
/// term (conductance, susceptibility, Drude and Lorentz poles):
///
/// - three mesh steps (lambda / 20, 40, 80 in vacuum) with the sheet at a
///   position that is not a node of the uniform mesh (0.3, 0.6 and 0.2 of a
///   cell): the mesh puts a node on it ([`crate::sheet::align_nodes`]) and
///   the errors fall as the square of the step (the sheet has no thickness
///   to choose: the current goes on the node plane, divided by the dual
///   cell);
/// - the same positions on explicit uniform meshes, which are not moved: the
///   sheet is shared between the two planes around it (first-order error,
///   measured);
/// - energy balance R + T + A = 1, A from the sheet current;
/// - an explicit non-uniform mesh whose cells change size at the sheet
///   (lambda / 40 before it, lambda / 70 after).
pub fn sheet_normal_incidence(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("sheet_normal_incidence");
    let (n1, n2) = (1.4, 2.0);
    let x_node = 2.0e-6;
    let x_off = 2.0e-6 + 0.3 * LAMBDA / 20.0;
    let uniform = |dx: f64| -> Vec<f64> { (0..=((4e-6 / dx).round() as usize)).map(|i| i as f64 * dx).collect() };
    for (label, c) in [("Drude", drude_sheet()), ("mixed", mixed_sheet())] {
        let mut prev: Option<f64> = None;
        for (cells, tol, tol_off) in [(20.0, 0.012, 0.025), (40.0, 3e-3, 0.012), (80.0, 1e-3, 4e-3)] {
            let dx = LAMBDA / cells;
            let (er, et, ea, mc) = normal_incidence(run, &c, x_off, n1, n2, dx, None)?;
            rep.mcells_per_s = mc;
            rep.check_abs(&format!("{label}, lambda/{cells}: max |R - R_a|"), er, 0.0, tol);
            rep.check_abs(&format!("{label}, lambda/{cells}: max |T - T_a|"), et, 0.0, tol);
            rep.check_abs(&format!("{label}, lambda/{cells}: max |R + T + A - 1|"), ea, 0.0, 2e-3);
            if let Some(p) = prev {
                rep.check_abs(&format!("{label}, lambda/{cells}: error / error at twice the step (< 0.4)"), (er.max(et) / p).min(1.0), 0.0, 0.4);
            }
            prev = Some(er.max(et));
            let (er2, et2, ea2, _) = normal_incidence(run, &c, x_off, n1, n2, dx, Some(uniform(dx)))?;
            rep.check_abs(&format!("{label}, between nodes (explicit mesh), lambda/{cells}: max |R - R_a|"), er2, 0.0, tol_off);
            rep.check_abs(&format!("{label}, between nodes (explicit mesh), lambda/{cells}: max |T - T_a|"), et2, 0.0, tol_off);
            rep.check_abs(&format!("{label}, between nodes (explicit mesh), lambda/{cells}: max |R + T + A - 1|"), ea2, 0.0, 2e-3);
        }
        // non-uniform: lambda/40 before the sheet, lambda/70 after
        let (d1, d2) = (LAMBDA / 40.0, LAMBDA / 70.0);
        let mut x: Vec<f64> = (0..=((x_node / d1).round() as usize)).map(|i| i as f64 * d1).collect();
        let n_after = ((4e-6 - x_node) / d2).round() as usize;
        x.extend((1..=n_after).map(|i| x_node + i as f64 * (4e-6 - x_node) / n_after as f64));
        let (er, et, ea, _) = normal_incidence(run, &c, x_node, n1, n2, d2, Some(x))?;
        rep.check_abs(&format!("{label}, non-uniform mesh: max |R - R_a|"), er, 0.0, 4e-3);
        rep.check_abs(&format!("{label}, non-uniform mesh: max |T - T_a|"), et, 0.0, 4e-3);
        rep.check_abs(&format!("{label}, non-uniform mesh: max |R + T + A - 1|"), ea, 0.0, 2e-3);
    }
    Ok(rep)
}

/// Zero-thickness sheet at oblique incidence (2-D, Bloch-periodic along
/// the sheet, one frequency): s and p polarisation at 30 and 60 degrees,
/// sheet between vacuum and n = 1.5, against the analytic formulas.
pub fn sheet_oblique(run: Runner) -> Result<CaseReport> {
    let n2: f64 = 1.5;
    let dx = LAMBDA / 60.0;
    let ly = 8.0 * dx;
    let k0 = TAU * f0() / C0;
    let sig = TimeSignal::gaussian(f0(), 0.06 * f0());
    let c = mixed_sheet();
    let xs = 2.5e-6 + 0.4 * dx;
    let mut rep = CaseReport::new("sheet_oblique");
    for (label, pol, comp) in sets_2d() {
        let p = comp != Component::Ez;
        for theta in [30f64.to_radians(), 60f64.to_radians()] {
            let ky = k0 * theta.sin();
            let mk = |full: bool| {
                let mut sim = sim_2d([0.0, 0.0], [4e-6, ly], dx, pml2(), [Boundary::Bloch { k: ky }, Boundary::Bloch { k: ky }], 300e-15, pol);
                if full {
                    sim.media.push(Medium::from_index(n2).with_name("glass"));
                    sim.structures.push(Structure { name: "g".into(), shape: Shape::Box { min: [xs, -1.0, -1.0], max: [10e-6, 1.0, 1.0] }, medium: "glass".into() });
                    sim.sheets.push(full_sheet(0, xs, c.clone()));
                }
                sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 1.0e-6, direction: Direction::Plus, polarization: comp, signal: sig.clone(), amplitude: 1.0, angle: theta, min: None, max: None });
                for (name, x) in [("T", 3.2e-6), ("R", 0.5e-6)] {
                    sim.monitors.push(Monitor::Flux { name: name.into(), axis: 0, position: x, min: [x, -1.0, 0.0], max: [x, ly + 1.0, 0.0], freqs: vec![f0()], apodization: Apodization::default() });
                }
                sim
            };
            let vac = run(&mk(false))?;
            let res = run(&mk(true))?;
            let p_inc = vac.flux["T"][0];
            let (ra, ta) = sheet_rt(c.sigma(TAU * f0()), 1.0, n2, theta, p);
            let deg = theta.to_degrees();
            rep.mcells_per_s = res.stats.mcells_per_s;
            rep.check_abs(&format!("{label} ({}) {deg:.0} deg: R", if p { "p" } else { "s" }), -res.flux["R"][0] / p_inc, ra, 6e-3);
            rep.check_abs(&format!("{label} ({}) {deg:.0} deg: T", if p { "p" } else { "s" }), res.flux["T"][0] / p_inc, ta, 6e-3);
        }
    }
    Ok(rep)
}

/// Fitted complex wavenumbers of the plasmon at `freqs` (see
/// [`sheet_plasmon`]), `cells` cells per plasmon wavelength at f0.
fn plasmon_q(run: Runner, c: &SheetConductivity, e2: f64, freqs: &[f64], cells: f64) -> Result<(Vec<Complex64>, f64)> {
    let lp = LAMBDA / 8.0;
    let dx = lp / cells;
    let (lx, ly) = (14.0 * lp, 3.0 * lp);
    let mut b = Boundaries::all_pml();
    b.x = pml2();
    b.y = pml2();
    let mut sim = Simulation::new([0.0, -ly / 2.0, -0.5], [lx, ly / 2.0, 0.5], MeshSpec::Uniform { dx: [dx; 3] }, b, 400e-15).two_d(2, crate::spec::Polarization::Te);
    sim.media.push(Medium::dielectric(e2).with_name("sub"));
    sim.structures.push(Structure { name: "sub".into(), shape: Shape::Box { min: [-1.0, -1.0, -1.0], max: [1.0, 0.0, 1.0] }, medium: "sub".into() });
    sim.sheets.push(full_sheet(1, 0.0, c.clone()));
    let x_src = 1.5 * lp;
    sim.sources.push(Source::PointDipole { name: "d".into(), position: [x_src, 0.1 * lp, 0.0], polarization: Component::Ey, signal: TimeSignal::gaussian(f0(), 0.4 * f0()), amplitude: 1.0 });
    sim.monitors.push(Monitor::DftField { name: "line".into(), min: [0.0, 0.0, 0.0], max: [lx, 0.0, 0.0], components: vec![Component::Ex], freqs: freqs.to_vec(), apodization: Apodization::default(), stride: 0 });
    sim.shutoff = Some(1e-6);
    let res = run(&sim)?;
    let grid = sim.build_grid()?;
    let fl = &res.dft["line"].fields[0];
    let mut out = vec![];
    for fi in 0..freqs.len() {
        // fit window: from 3 plasmon wavelengths past the source to 3 before the PML
        let (x0, x1) = (x_src + 3.0 * lp, lx - 3.0 * lp);
        let (mut xs, mut lnamp, mut ph) = (vec![], vec![], vec![]);
        let mut last = 0.0f64;
        for i in 0..fl.dims[1] {
            let x = grid.center(0, fl.origin[0] + i);
            if x < x0 || x > x1 {
                continue;
            }
            let v = fl.at(fi, i, 0, 0);
            let mut a = v.arg();
            if !ph.is_empty() {
                while a < last - PI {
                    a += TAU;
                }
                while a > last + PI {
                    a -= TAU;
                }
            }
            last = a;
            xs.push(x);
            lnamp.push(v.norm().ln());
            ph.push(a);
        }
        out.push(Complex64::new(linfit(&xs, &ph).0, -linfit(&xs, &lnamp).0));
    }
    Ok((out, res.stats.mcells_per_s))
}

/// Sheet plasmon (2-D, the sheet a line normal to y between vacuum and
/// eps = 2.25): a dipole next to a Drude sheet launches the bound TM
/// plasmon; its complex wavenumber q along the sheet, fitted from the
/// field on the sheet at three frequencies, against the dispersion relation
/// `1 / k1 + e2 / k2 = -i sigma / (w eps0)` (retarded; the quasi-static
/// `q = i eps0 (1 + e2) w / sigma` is 1.5 % lower here, q ~ 8 k0), at 30 and
/// 60 cells per plasmon wavelength (second-order convergence).
pub fn sheet_plasmon(run: Runner) -> Result<CaseReport> {
    let e2 = 2.25;
    let w0 = TAU * f0();
    // q ~ 8 k0 at f0: D = eps0 (1 + e2) w c / 8
    let d = EPS0_ * (1.0 + e2) * w0 * C0 / 8.0;
    let c = SheetConductivity { name: "plasmon".into(), poles: vec![SheetPole::Drude { drude_weight: d, gamma: 0.01 * w0 }], ..Default::default() };
    let freqs: Vec<f64> = [0.85, 1.0, 1.15].iter().map(|r| r * f0()).collect();
    let mut rep = CaseReport::new("sheet_plasmon");
    let (q30, _) = plasmon_q(run, &c, e2, &freqs, 30.0)?;
    let (q60, mc) = plasmon_q(run, &c, e2, &freqs, 60.0)?;
    rep.mcells_per_s = mc;
    for (fi, f) in freqs.iter().enumerate() {
        let om = TAU * f;
        let (q, qs) = sheet_plasmon_q(c.sigma(om), 1.0, e2, om);
        let tag = format!("{:.2} f0 (q / k0 = {:.3}, quasi-static {:.3})", f / f0(), q.re / (om / C0), qs.re / (om / C0));
        rep.check(&format!("Re q, 30 cells: {tag}"), q30[fi].re, q.re, 0.02);
        rep.check(&format!("Re q, 60 cells: {tag}"), q60[fi].re, q.re, 0.005);
        rep.check_abs(&format!("Re q error ratio 60 / 30 cells @ {:.2} f0 (< 0.4)", f / f0()), ((q60[fi].re - q.re) / (q30[fi].re - q.re)).abs(), 0.0, 0.4);
        rep.check(&format!("Im q, 60 cells @ {:.2} f0", f / f0()), q60[fi].im, q.im, 0.05);
    }
    Ok(rep)
}

const EPS0_: f64 = crate::constants::EPS0;

/// Quasi-static plasmon resonances of a periodic array of strips (width
/// `w`, period `p`) of a sheet with sigma = i D / w (lossless Drude) in a
/// plane between media of mean relative permittivity `eps_mean`: the
/// smallest eigenvalue mu of K = B^T A B, omega_1^2 = D mu, where A is the
/// potential of cell charges on the strip (periodic line-charge Green's
/// function -ln|2 sin(pi x / p)| / (2 pi eps0 eps_mean)), B the divergence
/// of the edge currents (zero at the strip edges) over `m` cells.
pub fn strip_array_mu(w: f64, p: f64, eps_mean: f64, m: usize) -> f64 {
    let h = w / m as f64;
    let g = |x: f64| -(2.0 * (PI * x / p).sin().abs()).ln() / (TAU * EPS0_ * eps_mean);
    // A (m x m): midpoint rule, with the integrated log singularity on the diagonal
    let mut a = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            a[i * m + j] = if i == j {
                // int_{-h/2}^{h/2} -ln|2 sin(pi x / p)| dx ~ -h (ln(pi h / p) - 1) (+ O(h^3))
                -h * ((PI * h / p).ln() - 1.0) / (TAU * EPS0_ * eps_mean)
            } else {
                h * g((i as f64 - j as f64) * h)
            };
        }
    }
    // K = B^T A B over the m - 1 interior edges: edge k between cells k-1 and k
    let ne = m - 1;
    // A B: column k = (A[:, k] - A[:, k-1]) / h  ... B[j][k] = (+1 if k = j (right edge of cell j - 1)...)
    // B[j][k] = 1/h if edge k is the right edge of cell j (j = k - 1), -1/h if its left edge (j = k)
    let mut k = vec![0.0f64; ne * ne];
    for r in 0..ne {
        for c in 0..ne {
            // (B^T A B)[r][c] = sum_ij B[i][r] A[i][j] B[j][c]
            let (ir, jr) = (r, r + 1); // cell r has edge r+... edge index e = 0..ne-1 sits between cells e and e+1
            let (ic, jc) = (c, c + 1);
            let v = a[ir * m + ic] - a[ir * m + jc] - a[jr * m + ic] + a[jr * m + jc];
            k[r * ne + c] = v / (h * h);
        }
    }
    // smallest eigenvalue by inverse iteration (dense LU without pivoting: K is SPD)
    let n = ne;
    let mut lu = k.clone();
    for p_ in 0..n {
        for r in p_ + 1..n {
            let f = lu[r * n + p_] / lu[p_ * n + p_];
            lu[r * n + p_] = f;
            for c in p_ + 1..n {
                lu[r * n + c] -= f * lu[p_ * n + c];
            }
        }
    }
    let solve = |b: &mut Vec<f64>| {
        for r in 0..n {
            for c in 0..r {
                b[r] -= lu[r * n + c] * b[c];
            }
        }
        for r in (0..n).rev() {
            for c in r + 1..n {
                b[r] -= lu[r * n + c] * b[c];
            }
            b[r] /= lu[r * n + r];
        }
    };
    // start from a symmetric (dipolar) current
    let mut x: Vec<f64> = (0..n).map(|e| ((e as f64 + 1.0) / m as f64 * PI).sin()).collect();
    let mut mu = 0.0;
    for _ in 0..200 {
        let nrm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.iter_mut().for_each(|v| *v /= nrm);
        let mut y = x.clone();
        solve(&mut y);
        let ray = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>();
        let new = 1.0 / ray;
        x = y;
        if (new - mu).abs() < 1e-12 * new {
            mu = new;
            break;
        }
        mu = new;
    }
    mu
}

/// Absorption spectrum of a periodic array of sheet strips at normal
/// incidence (2-D, E across the strips), `cells` cells per strip width.
fn strip_absorption(run: Runner, c: &SheetConductivity, w: f64, p: f64, e2: f64, freqs: &[f64], cells: f64) -> Result<(Vec<f64>, f64)> {
    let h = w / cells;
    let nx = (p / h).round() as usize;
    let x: Vec<f64> = (0..=nx).map(|i| i as f64 * p / nx as f64).collect();
    // y: fine within 3 w of the sheet, then graded by 1.15 up to lambda / 30
    let mut ys = vec![0.0f64];
    let mut yv = 0.0;
    let mut d = h;
    let ymax = 2.0 * LAMBDA;
    while yv < ymax {
        if yv > 3.0 * w {
            d = (d * 1.15).min(LAMBDA / 30.0);
        }
        yv += d;
        ys.push(yv);
    }
    let mut y: Vec<f64> = ys.iter().rev().map(|v| -v).collect();
    y.extend(ys.iter().skip(1));
    let mut b = Boundaries::all(Boundary::Periodic);
    b.y = pml2();
    let mesh = MeshSpec::Explicit { x, y, z: vec![-0.5, 0.5] };
    let mk = |strip: bool| {
        let mut sim = Simulation::new([0.0, -ymax, -0.5], [p, ymax, 0.5], mesh.clone(), b.clone(), 2000e-15).two_d(2, crate::spec::Polarization::Te);
        sim.media.push(Medium::dielectric(e2).with_name("sub"));
        sim.structures.push(Structure { name: "sub".into(), shape: Shape::Box { min: [-1.0, -1.0, -1.0], max: [1.0, 0.0, 1.0] }, medium: "sub".into() });
        if strip {
            let x0 = 0.5 * (p - w);
            sim.sheets.push(Sheet { min: Some([x0, -1.0, -1.0]), max: Some([x0 + w, 1.0, 1.0]), ..full_sheet(1, 0.0, c.clone()) });
        }
        let ysrc = 0.8 * LAMBDA;
        sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 1, position: ysrc, direction: Direction::Minus, polarization: Component::Ex, signal: TimeSignal::gaussian(f0(), 0.3 * f0()), amplitude: 1.0, angle: 0.0, min: None, max: None });
        for (name, yy) in [("R", 1.2 * LAMBDA), ("T", -0.8 * LAMBDA)] {
            sim.monitors.push(Monitor::Flux { name: name.into(), axis: 1, position: yy, min: [-1.0, yy, 0.0], max: [1.0, yy, 0.0], freqs: freqs.to_vec(), apodization: Apodization::default() });
        }
        sim.shutoff = Some(1e-7);
        sim
    };
    let inc = run(&mk(false))?;
    let res = run(&mk(true))?;
    let a = (0..freqs.len())
        .map(|i| {
            // without the strip: the incident flux crosses R (behind the source: zero) and T (transmitted into the substrate)
            let r = res.flux["R"][i] / inc.flux["T"][i].abs();
            let t = res.flux["T"][i] / inc.flux["T"][i];
            1.0 - r.abs() - t
        })
        .collect();
    Ok((a, res.stats.mcells_per_s))
}

/// Resonance frequency of an absorption spectrum: for a sheet of
/// Lorentz-oscillator response between two media, A = 4 y1 Re(sigma) /
/// |y1 + y2 + sigma|^2 gives w^2 / A = alpha (w_r^2 - w^2)^2 + beta w^2 exactly,
/// a quadratic c0 + c1 x + c2 x^2 in x = w^2 with w_r^4 = c0 / c2. Fitted by
/// least squares over the samples above 0.3 of the peak.
fn resonance(freqs: &[f64], a: &[f64]) -> f64 {
    let amax = a.iter().cloned().fold(0.0, f64::max);
    let pts: Vec<(f64, f64)> = freqs.iter().zip(a).filter(|p| *p.1 > 0.3 * amax).map(|(&f, &v)| ((f / f0()).powi(2), (f / f0()).powi(2) / v)).collect();
    // normal equations of y = c0 + c1 x + c2 x^2
    let mut m = [[0.0f64; 3]; 3];
    let mut r = [0.0f64; 3];
    for &(x, y) in &pts {
        let b = [1.0, x, x * x];
        for p in 0..3 {
            for q in 0..3 {
                m[p][q] += b[p] * b[q];
            }
            r[p] += b[p] * y;
        }
    }
    let inv = crate::materials::inv3(m);
    let c: Vec<f64> = (0..3).map(|p| (0..3).map(|q| inv[p][q] * r[q]).sum()).collect();
    f0() * (c[0] / c[2]).powf(0.25)
}

/// Finite sheet patches: the plasmon resonance of a periodic array of
/// sheet strips (2-D, width w = 20 nm, period 3 w, between vacuum and
/// eps = 2.25, normal incidence with E across the strips) against the
/// quasi-static eigenvalue of the strip array ([`strip_array_mu`]): the
/// Drude weight is chosen so that the quasi-static dipole resonance is at
/// f0, and the absorption peak of the run is compared with f0, at 20 and
/// 40 cells across the strip.
pub fn sheet_ribbon(run: Runner) -> Result<CaseReport> {
    let (w, p, e2) = (20e-9, 60e-9, 2.25);
    let w0 = TAU * f0();
    let mu = strip_array_mu(w, p, 0.5 * (1.0 + e2), 400);
    let c = SheetConductivity { name: "ribbon".into(), poles: vec![SheetPole::Drude { drude_weight: w0 * w0 / mu, gamma: 0.05 * w0 }], ..Default::default() };
    let freqs: Vec<f64> = (0..61).map(|i| f0() * (0.8 + 0.4 * i as f64 / 60.0)).collect();
    let mut rep = CaseReport::new("sheet_ribbon");
    for (cells, tol) in [(20.0, 0.03), (40.0, 0.025)] {
        let (a, mc) = strip_absorption(run, &c, w, p, e2, &freqs, cells)?;
        rep.mcells_per_s = mc;
        let fpk = resonance(&freqs, &a);
        let amax = a.iter().cloned().fold(0.0, f64::max);
        rep.check(&format!("{cells} cells per width: absorption peak / quasi-static resonance (A_max = {amax:.3})"), fpk / f0(), 1.0, tol);
    }
    Ok(rep)
}
