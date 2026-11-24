//! Unit tests of the plan compiler.

use super::*;

#[test]
fn automatic_dft_stride() {
    use crate::grid::Component;
    let dipole = |signal: TimeSignal| Source::PointDipole { name: String::new(), position: [0.0; 3], polarization: Component::Ez, signal, amplitude: 1.0 };
    let f0 = 2e14;
    let dt = 4e-17;
    let pulse = dipole(TimeSignal::gaussian(f0, 0.1 * f0));
    // 1 / (s dt) >= f_monitor + f0 + 7 fwidth
    let s = auto_dft_stride(std::slice::from_ref(&pulse), false, dt, &[0.9 * f0, 1.1 * f0]);
    assert_eq!(s, (1.0 / (dt * (1.1 * f0 + 1.7 * f0))).floor() as usize);
    assert!(s >= 20 && 1.0 / (s as f64 * dt) >= 2.8 * f0);
    // a broader pulse or a higher monitor frequency: a smaller stride
    assert!(auto_dft_stride(&[dipole(TimeSignal::gaussian(f0, 0.5 * f0))], false, dt, &[f0]) < s);
    assert!(auto_dft_stride(std::slice::from_ref(&pulse), false, dt, &[3.0 * f0]) < s);
    // no known band edge: every step
    assert_eq!(auto_dft_stride(std::slice::from_ref(&pulse), true, dt, &[f0]), 1, "nonlinear media make harmonics");
    assert_eq!(auto_dft_stride(&[pulse.clone(), dipole(TimeSignal::cw(f0, 0.1 * f0))], false, dt, &[f0]), 1);
    assert_eq!(auto_dft_stride(&[dipole(TimeSignal::Ricker { freq0: f0, offset: 1.5, amplitude: 1.0 })], false, dt, &[f0]), 1);
    assert_eq!(auto_dft_stride(&[], false, dt, &[f0]), 1);
    // a coarse time step: never below 1
    assert_eq!(auto_dft_stride(&[pulse], false, 1.0 / (2.0 * f0), &[f0]), 1);
}

#[test]
fn monitors_on_one_plane_share_their_boxes() {
    use crate::grid::MeshSpec;
    use crate::monitors::Monitor;
    use crate::sources::ModeProfile;
    use crate::spec::Boundaries;
    let mut sim = Simulation::new([0.0; 3], [2e-6, 1e-6, 1e-6], MeshSpec::Uniform { dx: [0.1e-6; 3] }, Boundaries::all_pml(), 2e-14);
    let profile = ModeProfile { neff: 1.0, neff_imag: 0.0, dims: [1, 1], e_re: [vec![1.0], vec![0.0]], e_im: [vec![0.0], vec![0.0]], h_re: [vec![0.0], vec![1.0]], h_im: [vec![0.0], vec![0.0]], freq: 3e14, h_offset: None };
    let mode = |name: &str, x: f64, freqs: Vec<f64>| Monitor::Mode { name: name.into(), axis: 0, position: x, min: [x, 0.2e-6, 0.2e-6], max: [x, 0.8e-6, 0.8e-6], freqs, modes: vec![profile.clone()], apodization: Default::default() };
    for i in 0..40 {
        sim.monitors.push(mode(&format!("m{i}"), 1.0e-6, vec![3e14]));
    }
    sim.monitors.push(mode("other_plane", 1.5e-6, vec![3e14]));
    sim.monitors.push(mode("other_freqs", 1.0e-6, vec![3e14, 3.1e14]));
    sim.monitors.push(Monitor::Flux { name: "flux".into(), axis: 0, position: 1.0e-6, min: [1.0e-6, 0.2e-6, 0.2e-6], max: [1.0e-6, 0.8e-6, 0.8e-6], freqs: vec![3e14], apodization: Default::default() });
    let plan = compile(&sim, &CompileOptions::default()).unwrap();
    let own = |d: &DftPlan| d.boxes.iter().filter(|b| b.is_local()).count();
    assert_eq!(own(&plan.dft[0]), 6);
    for d in &plan.dft[1..40] {
        assert_eq!(own(d), 0, "{}", d.name);
        assert!(d.shared.iter().enumerate().all(|(bi, s)| *s == Some((0, bi))));
        assert!(d.phasor_e.is_empty() && d.phasor_h.is_empty());
    }
    assert_eq!(own(&plan.dft[40]), 6, "another plane records its own fields");
    assert_eq!(own(&plan.dft[41]), 6, "another frequency list records its own fields");
    assert_eq!(own(&plan.dft[42]), 0, "a flux monitor on the plane of a mode monitor shares its fields");
    let total: usize = plan.dft.iter().map(own).sum();
    assert_eq!(total, 18);
}

#[test]
fn pml_variant_damps_poles_below_the_band_and_freezes_poles_above_it() {
    use crate::materials::Pole;
    let w = |um: f64| TAU * C0 / (um * 1e-6);
    // a pulse centred at 1.55 um, 6 % wide: nothing outside 1.31 to 1.89 um, results from 1.38 to 1.76 um
    let band = SourceBand { edge_lo: w(1.89), used_lo: w(1.76), centre: w(1.55), used_hi: w(1.38), edge_hi: w(1.31) };
    // a Sellmeier-type fit of silica around 1.55 um: an infrared and a weak ultraviolet pole
    let ir = Pole::Lorentz { delta_eps: 4.7333, omega0: 8.626e13, gamma: 0.0 };
    let uv = Pole::Lorentz { delta_eps: 0.0127, omega0: 3.1e15, gamma: 0.0 };
    let silica = Medium::dielectric(2.103).with_poles(vec![ir.clone(), uv.clone()]);
    let eps = |e_inf: f64, poles: &[Pole], x: f64| e_inf + poles.iter().map(|p| p.chi(x)).sum::<Complex64>();
    let scan = |e_inf: f64, poles: &[Pole], from: f64| (0..6000).map(|i| eps(e_inf, poles, from * 1.002f64.powi(i)).re).fold(f64::INFINITY, f64::min);
    let (v, extra) = pml_pole_variant(&silica, &band).expect("both poles are treated");
    // the infrared pole is damped ...
    let Pole::Lorentz { delta_eps, omega0, gamma } = v[0] else { panic!() };
    assert_eq!((delta_eps, omega0), (4.7333, 8.626e13));
    assert!((gamma / omega0 - ((1.0f64 + 2.0 * 4.7333 / (2.103 + extra)).sqrt() - 1.0)).abs() < 1e-12);
    // ... the ultraviolet pole is frozen: switched off, its value at the centre of the band in eps_inf
    assert_eq!(v[1], Pole::Lorentz { delta_eps: 0.0, omega0: 3.1e15, gamma: 0.0 });
    assert!((extra - uv.chi(band.centre).re).abs() < 1e-15 && extra > 0.012 && extra < 0.02);
    // no band of negative permittivity is left anywhere (20 um to 10 nm), where the fit had two
    assert!(scan(2.103, &silica.poles, 2e13) < -10.0, "the lossless poles make the medium metal-like above their resonances");
    assert!(scan(2.103 + extra, &v, 2e13) >= 0.5 * 2.103, "{}", scan(2.103 + extra, &v, 2e13));
    // and the permittivity in the band moves by a few parts in a thousand
    for um in [1.76, 1.55, 1.38] {
        let d = (eps(2.103 + extra, &v, w(um)) - eps(2.103, &silica.poles, w(um))).norm();
        assert!(d < 6e-3 && d > 1e-4, "{um} um: {d}");
    }
    // silicon fitted with eps_inf = 1 and one strong ultraviolet pole: frozen, the change at the band ends about 0.05
    let si_pole = Pole::Lorentz { delta_eps: 10.67, omega0: w(0.299), gamma: 0.0 };
    let si = Medium::dielectric(1.0).with_poles(vec![si_pole.clone()]);
    let (v, extra) = pml_pole_variant(&si, &band).expect("frozen");
    assert_eq!(v[0], Pole::Lorentz { delta_eps: 0.0, omega0: w(0.299), gamma: 0.0 });
    assert!((1.0 + extra - 12.08).abs() < 0.05, "eps in the layers {}", 1.0 + extra);
    let ends = [(si_pole.chi(band.used_lo).re - extra).abs(), (si_pole.chi(band.used_hi).re - extra).abs()];
    assert!(ends[0] > 0.02 && ends[0] < 0.12 && ends[1] > 0.02 && ends[1] < 0.12, "{ends:?}");
    // a strong infrared term and a wide band: damped up to a change of 4 % at the low end of the band
    let nitride = Medium::dielectric(3.46).with_poles(vec![Pole::Lorentz { delta_eps: 1.92, omega0: 2.12e14, gamma: 0.0 }]);
    let wide = |x: f64| SourceBand { edge_lo: 0.5 * x, used_lo: x, centre: 1.5 * x, used_hi: 2.0 * x, edge_hi: 2.5 * x };
    let change = |m: &Medium, x: f64| pml_pole_variant(m, &wide(x)).map(|(v, _)| (v[0].chi(x) - m.poles[0].chi(x)).norm() / (m.eps[0] + m.poles[0].chi(x).re));
    let (c2, c3) = (change(&nitride, w(2.4)).unwrap(), change(&nitride, w(3.0)).unwrap());
    assert!(c2 > 0.005 && c2 < c3 && c3 < 0.04, "{c2} {c3}");
    assert!(change(&nitride, w(4.5)).is_none(), "too close to the band: left alone");
    // nothing to do: damped enough already, a pole just outside the band on either side, in the band, a metal, no poles
    let none = |m: Medium| pml_pole_variant(&m, &band).is_none();
    assert!(none(Medium::dielectric(2.103).with_poles(vec![Pole::Lorentz { delta_eps: 4.7333, omega0: 8.626e13, gamma: 3e14 }])));
    assert!(none(Medium::dielectric(2.0).with_poles(vec![Pole::Lorentz { delta_eps: 1.0, omega0: 0.7 * band.edge_lo, gamma: 0.0 }])), "just below the band");
    assert!(none(Medium::dielectric(2.0).with_poles(vec![Pole::Lorentz { delta_eps: 1.0, omega0: 1.2 * band.edge_hi, gamma: 0.0 }])), "just above the band");
    assert!(none(Medium::dielectric(2.0).with_poles(vec![Pole::Lorentz { delta_eps: 0.1, omega0: band.centre, gamma: 1e13 }])), "in the band");
    assert!(none(Medium::dielectric(1.0).with_poles(vec![Pole::Drude { omega_p: 1.3e16, gamma: 1e14 }])));
    assert!(none(Medium::dielectric(2.1)));
    // an absorbing ultraviolet pole that never makes eps negative is left alone; a medium that is not plain is not frozen
    assert!(none(Medium::dielectric(2.0).with_poles(vec![Pole::Lorentz { delta_eps: 0.5, omega0: 5e15, gamma: 5e15 }])));
    let mut kerr = Medium::dielectric(1.0).with_poles(vec![si_pole.clone()]);
    kerr.chi3 = [1e-18; 3];
    assert!(pml_pole_variant(&kerr, &band).is_none());
    // the band of the sources
    use crate::grid::Component;
    let dipole = |signal: TimeSignal| Source::PointDipole { name: String::new(), position: [0.0; 3], polarization: Component::Ez, signal, amplitude: 1.0 };
    let f0 = 2e14;
    let close = |a: f64, b: f64| (a / b - 1.0).abs() < 1e-12;
    let is = |b: SourceBand, v: [f64; 5]| close(b.edge_lo, TAU * v[0] * f0) && close(b.used_lo, TAU * v[1] * f0) && close(b.centre, TAU * v[2] * f0) && close(b.used_hi, TAU * v[3] * f0) && close(b.edge_hi, TAU * v[4] * f0);
    assert_eq!(source_band(&[]), None);
    assert!(is(source_band(&[dipole(TimeSignal::gaussian(f0, 0.1 * f0))]).unwrap(), [0.7, 0.8, 1.0, 1.2, 1.3]));
    assert!(is(source_band(&[dipole(TimeSignal::gaussian(f0, 0.1 * f0)), dipole(TimeSignal::cw(0.5 * f0, 0.01 * f0))]).unwrap(), [0.47, 0.48, 0.75, 1.2, 1.3]), "all the sources together");
    assert!(is(source_band(&[dipole(TimeSignal::Ricker { freq0: f0, offset: 1.5, amplitude: 1.0 })]).unwrap(), [0.25, 0.5, 1.0, 2.0, 3.0]));
    assert_eq!(source_band(&[dipole(TimeSignal::gaussian(f0, 0.5 * f0))]), None, "a pulse that reaches zero frequency has no lower edge");
    assert_eq!(source_band(&[dipole(TimeSignal::Custom { times: vec![0.0, 1.0], values: vec![0.0, 1.0] })]), None);
}

#[test]
fn an_absorber_lives_in_the_update_coefficients() {
    use crate::grid::{Component, MeshSpec};
    use crate::materials::Pole;
    use crate::spec::{AbsorberParams, Boundaries};
    let f0 = C0 / 1.55e-6;
    let layers = 8usize;
    let dx = 50e-9;
    let mut sim = Simulation::new([0.0; 3], [1e-6, 0.5e-6, 0.5e-6], MeshSpec::Uniform { dx: [dx; 3] }, Boundaries::all_pml(), 1e-14);
    sim.boundaries.x = [Boundary::Absorber { layers, params: AbsorberParams::default() }, Boundary::Absorber { layers, params: AbsorberParams::default() }];
    // a dielectric with a lossless pole above the band: eps = 2 + chi at the centre of the band
    let pole = Pole::Lorentz { delta_eps: 2.0, omega0: 6.0e15, gamma: 0.0 };
    sim.media.push(Medium::dielectric(2.0).with_poles(vec![pole.clone()]).with_name("glass"));
    sim.background = "glass".into();
    sim.sources.push(Source::PointDipole { name: String::new(), position: [0.5e-6, 0.25e-6, 0.25e-6], polarization: Component::Ez, signal: TimeSignal::gaussian(f0, 0.06 * f0), amplitude: 1.0 });
    let plan = compile(&sim, &CompileOptions::default()).unwrap();
    // no CPML slab along x: the absorber is not a PML
    assert!(plan.pml.iter().all(|p| p.axis != 0) && plan.pml.len() == 4);
    assert_eq!(plan.absorb.len(), 2);
    assert!(plan.absorb.iter().all(|a| a.axis == 0 && a.size[0] == layers));
    let n = plan.grid.n();
    assert_eq!(n[0], 20 + 2 * layers);
    let (ny, nz) = (plan.ny(), plan.nz());
    let plane = ny * nz;
    let ca = plan.ca.as_ref().expect("the electric conductivity needs ca");
    let hl = plan.hloss.as_ref().expect("the magnetic conductivity needs the H loss lists");
    let dt = plan.dt;
    let eps_c = 2.0 + pole.chi(TAU * f0).re;
    let thick = layers as f64 * dx;
    let smax = ABSORBER_ATTENUATION * 4.0 * C0 / (2.0 * thick * eps_c.sqrt());
    let (j, k) = (ny / 2, nz / 2);
    // E_y sits on the x nodes: rate smax (depth / thickness)^3, matched to eps at the band centre
    for i in 0..n[0] {
        let depth = if i < layers { (layers - i) as f64 } else if i > n[0] - layers { (i - (n[0] - layers)) as f64 } else { 0.0 };
        let f = 0.5 * smax * (depth / layers as f64).powi(3) * dt * eps_c / 2.0;
        let want = (1.0 - f) / (1.0 + f);
        let got = ca[1][(i + 1) * plane + j * nz + k] as f64;
        // (the first node is the outer wall, where the tangential field is held at zero)
        if i > 0 {
            assert!((got - want).abs() < 1e-6, "ca of Ey at node {i}: {got} against {want}");
        }
        assert_eq!(got == 1.0, depth == 0.0 || i == 0, "node {i}");
    }
    // H_z sits at the x cell centres: da = (1 - f) / (1 + f) with f = sigma^ dt / 2, the same rate
    let mut seen = 0;
    for (t, &o) in hl[2].idx.iter().enumerate() {
        let o = o as usize;
        let (i, jj, kk) = (o / plane - 1, (o / nz) % ny, o % nz);
        if jj != j || kk != k {
            continue;
        }
        let x = i as f64 + 0.5;
        let depth = if x < layers as f64 { layers as f64 - x } else { x - (n[0] - layers) as f64 };
        assert!(depth > 0.0, "an H loss entry outside the absorber, cell {i}");
        let f = 0.5 * smax * (depth / layers as f64).powi(3) * dt;
        assert!((hl[2].da[t] as f64 - (1.0 - f) / (1.0 + f)).abs() < 1e-6);
        seen += 1;
    }
    assert_eq!(seen, 2 * layers);
    // the polarisation term of a dispersive cell is divided by the same 1 + f as the curl
    let d = plan.disp.as_ref().unwrap();
    let c = d.comps.iter().find(|c| c.comp == Component::Ey).unwrap();
    let (mut inside, mut outside) = (0, 0);
    for (t, &si) in c.idx.iter().enumerate() {
        let si = si as usize;
        let i = si / plane - 1;
        let depth = if i < layers { (layers - i) as f64 } else if i > n[0] - layers { (i - (n[0] - layers)) as f64 } else { 0.0 };
        let f = 0.5 * smax * (depth / layers as f64).powi(3) * dt * eps_c / 2.0;
        assert!((c.scale[t] as f64 * 2.0 * (1.0 + f) - 1.0).abs() < 1e-6);
        if depth > 0.0 { inside += 1 } else { outside += 1 }
    }
    assert!(inside > 0 && outside > 0);
}

#[test]
fn an_absorber_is_the_same_in_every_slab_of_a_split_run() {
    use crate::geometry::{Shape, Structure};
    use crate::grid::{Component, MeshSpec};
    use crate::materials::Pole;
    use crate::spec::{AbsorberParams, Boundaries};
    use std::collections::HashMap;
    let f0 = C0 / 1.55e-6;
    let mut sim = Simulation::new([0.0; 3], [1e-6, 0.5e-6, 0.5e-6], MeshSpec::Uniform { dx: [50e-9; 3] }, Boundaries::all_pml(), 1e-14);
    let absorber = |layers| Boundary::Absorber { layers, params: AbsorberParams::default() };
    // absorbers on x (cut by the slabs) and on y (whole in every slab); corners where they overlap
    sim.boundaries.x = [absorber(8), absorber(6)];
    sim.boundaries.y = [absorber(5), absorber(5)];
    sim.media.push(Medium::dielectric(2.0).with_poles(vec![Pole::Lorentz { delta_eps: 2.0, omega0: 6.0e15, gamma: 0.0 }]).with_name("glass"));
    sim.media.push(Medium::dielectric(4.0).with_name("core"));
    sim.background = "glass".into();
    sim.structures.push(Structure { name: "core".into(), shape: Shape::Box { min: [-5e-6, 0.15e-6, 0.15e-6], max: [5e-6, 0.35e-6, 0.35e-6] }, medium: "core".into() });
    sim.sources.push(Source::PointDipole { name: String::new(), position: [0.5e-6, 0.25e-6, 0.25e-6], polarization: Component::Ez, signal: TimeSignal::gaussian(f0, 0.06 * f0), amplitude: 1.0 });
    let whole = compile(&sim, &CompileOptions::default()).unwrap();
    let nx = whole.grid.n()[0];
    let plane = whole.plane();
    // per global storage index: the magnetic loss factor, and (row, scale) of a dispersive sample
    let hloss_of = |p: &Plan, shift: usize| -> [HashMap<usize, Real>; 3] {
        let hl = p.hloss.as_ref().unwrap();
        [0, 1, 2].map(|a| hl[a].idx.iter().zip(&hl[a].da).map(|(&o, &d)| (o as usize + shift, d)).collect())
    };
    let disp_of = |p: &Plan, shift: usize| -> HashMap<(usize, usize), (u32, Real)> {
        let mut m = HashMap::new();
        for c in &p.disp.as_ref().unwrap().comps {
            for (t, &o) in c.idx.iter().enumerate() {
                m.insert((c.comp.axis(), o as usize + shift), (c.matid[t], c.scale[t]));
            }
        }
        m
    };
    let (whole_h, whole_d) = (hloss_of(&whole, 0), disp_of(&whole, 0));
    assert!(whole_h.iter().all(|m| !m.is_empty()) && !whole_d.is_empty());
    let (mut seen_h, mut seen_d) = (0, 0);
    let parts = decompose_x(nx, &[1.0, 2.0]);
    assert_eq!(parts.len(), 2);
    for (r, &(x0, nxl)) in parts.iter().enumerate() {
        let part = compile(&sim, &CompileOptions { slab: Some((x0, nxl)), nslabs: 2, slab_index: r, ..Default::default() }).unwrap();
        assert_eq!((part.x0, part.nxl), (x0, nxl));
        let (ca_w, ca_p) = (whole.ca.as_ref().unwrap(), part.ca.as_ref().unwrap());
        let (ch_w, ch_p) = (whole.ch.as_ref().unwrap(), part.ch.as_ref().unwrap());
        for a in 0..3 {
            // the interior planes of the slab are planes x0 .. x0 + nxl of the whole grid
            let (lo, hi) = (plane, (nxl + 1) * plane);
            let (wlo, whi) = ((x0 + 1) * plane, (x0 + nxl + 1) * plane);
            assert_eq!(part.cb[a][lo..hi], whole.cb[a][wlo..whi], "cb, component {a}, slab {r}");
            assert_eq!(ca_p[a][lo..hi], ca_w[a][wlo..whi], "ca, component {a}, slab {r}");
            assert_eq!(ch_p[a][lo..hi], ch_w[a][wlo..whi], "ch, component {a}, slab {r}");
        }
        let shift = x0 * plane;
        for (a, m) in hloss_of(&part, shift).iter().enumerate() {
            for (o, da) in m {
                assert_eq!(whole_h[a].get(o), Some(da), "H loss, component {a}, slab {r}");
                seen_h += 1;
            }
        }
        for (k, v) in disp_of(&part, shift) {
            assert_eq!(whole_d.get(&k), Some(&v), "dispersive sample {k:?}, slab {r}");
            seen_d += 1;
        }
        // the x absorbers are clipped to the slab; the y absorbers span it
        assert!(part.absorb.iter().all(|b| b.start[0] + b.size[0] <= nxl));
        assert_eq!(part.absorb.iter().filter(|b| b.axis == 1).count(), 2);
    }
    // every entry of the whole grid is in exactly one slab
    assert_eq!(seen_h, whole_h.iter().map(|m| m.len()).sum::<usize>());
    assert_eq!(seen_d, whole_d.len());
}

#[test]
fn stabilised_layers_use_the_damped_rows() {
    use crate::geometry::{Shape, Structure};
    use crate::grid::{Component, MeshSpec};
    use crate::materials::Pole;
    use crate::spec::Boundaries;
    let f0 = C0 / 1.55e-6;
    // fitted silica everywhere; optionally a core along x that runs through the x faces
    let build = |stabilize: bool, with_source: bool, core: bool| {
        let mut sim = Simulation::new([0.0; 3], [1e-6, 0.5e-6, 0.5e-6], MeshSpec::Uniform { dx: [50e-9; 3] }, Boundaries::all_pml(), 1e-14);
        for a in 0..3 {
            for bd in sim.boundaries.axis_mut(a).iter_mut() {
                if let Boundary::Pml { layers, params, .. } = bd {
                    *layers = 4;
                    params.stabilize = stabilize;
                }
            }
        }
        let mut silica = Medium::dielectric(2.103).with_poles(vec![
            Pole::Lorentz { delta_eps: 4.7333, omega0: 8.626e13, gamma: 0.0 },
            Pole::Lorentz { delta_eps: 0.0127, omega0: 3.1e15, gamma: 0.0 },
        ]);
        silica.name = "silica".into();
        sim.media.push(silica);
        sim.media.push(Medium::dielectric(4.0).with_name("core"));
        sim.background = "silica".into();
        if core {
            sim.structures.push(Structure { name: "core".into(), shape: Shape::Box { min: [-5e-6, 0.15e-6, 0.15e-6], max: [5e-6, 0.35e-6, 0.35e-6] }, medium: "core".into() });
        }
        if with_source {
            sim.sources.push(Source::PointDipole { name: String::new(), position: [0.5e-6, 0.25e-6, 0.25e-6], polarization: Component::Ez, signal: TimeSignal::gaussian(f0, 0.06 * f0), amplitude: 1.0 });
        }
        compile(&sim, &CompileOptions::default()).unwrap()
    };
    let plan = build(true, true, true);
    let d = plan.disp.as_ref().unwrap();
    assert_eq!(d.coefs.len(), 2 * d.npole * 3, "one row for the medium, one for its form inside the layers");
    let nx = plan.grid.n()[0];
    let (ny, nz) = (plan.ny(), plan.nz());
    let (mut inner, mut outer) = (0, 0);
    for c in &d.comps {
        for (t, &si) in c.idx.iter().enumerate() {
            let si = si as usize;
            let g = [si / (ny * nz) - 1, (si / nz) % ny, si % nz];
            // every face: the layers beside the core as well as those it runs into
            let layer = (0..3).any(|a| g[a] < 4 || g[a] >= [nx, ny, nz][a] - 4);
            assert_eq!(c.matid[t], layer as u32, "cell {g:?}");
            if layer { outer += 1 } else { inner += 1 }
        }
    }
    assert!(inner > 0 && outer > inner);
    // in the row of the layers the infrared pole is damped and the ultraviolet pole is switched off
    // (strength 0), its value at the centre of the band added to the high-frequency permittivity
    assert_ne!(d.coefs[0..3], d.coefs[d.npole * 3..d.npole * 3 + 3]);
    assert_ne!(d.coefs[5], 0.0);
    assert_eq!(d.coefs[d.npole * 3 + 5], 0.0);
    let extra = Pole::Lorentz { delta_eps: 0.0127, omega0: 3.1e15, gamma: 0.0 }.chi(TAU * f0).re;
    let cdt = C0 * plan.dt;
    for c in &d.comps {
        for (t, &si) in c.idx.iter().enumerate() {
            // (a voxel cut by the core's surface is mixed: its scale and eps_inf are its own)
            if c.mixed.binary_search(&(t as u32)).is_ok() {
                continue;
            }
            let e = if c.matid[t] == 1 { 2.103 + extra } else { 2.103 };
            assert!((c.scale[t] as f64 * e - 1.0).abs() < 1e-6);
            assert!((plan.cb[c.comp.axis()][si as usize] as f64 * e / cdt - 1.0).abs() < 1e-6);
        }
    }
    // switched off, without a band to go by, or with nothing but the uniform medium in the region: one row, as before
    for plan in [build(false, true, true), build(true, false, true), build(true, true, false)] {
        let d = plan.disp.as_ref().unwrap();
        assert!(d.comps.iter().all(|c| c.matid.iter().all(|&m| m == 0)));
    }
    assert_eq!(build(true, true, false).disp.as_ref().unwrap().coefs.len(), d.npole * 3, "no second row without a structure in the region");
}

#[test]
fn decomposition_covers_domain() {
    let d = decompose_x(100, &[1.0, 1.0, 2.0]);
    assert_eq!(d, vec![(0, 25), (25, 25), (50, 50)]);
}
