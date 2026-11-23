//! Sub-pixel smoothing of transparent dispersive dielectrics: the plan's
//! mixed voxels (eps_inf, pole weight) and the cases that stay staircased.

use lightsim_core::constants::{C0, TAU};
use lightsim_core::geometry::{Shape, Structure};
use lightsim_core::grid::{Component, MeshSpec};
use lightsim_core::materials::{Medium, Pole};
use lightsim_core::plan::{CompileOptions, compile};
use lightsim_core::sources::{Direction, Source, TimeSignal};
use lightsim_core::{Boundaries, Boundary, Simulation};

/// A glass-like medium: one UV Lorentz pole (n ~ sqrt(eps_inf + d_eps)).
fn glass(name: &str, eps_inf: f64, d_eps: f64) -> Medium {
    Medium::dielectric(eps_inf).with_name(name).with_poles(vec![Pole::Lorentz { delta_eps: d_eps, omega0: TAU * C0 / 0.1e-6, gamma: 0.0 }])
}

fn scene(core: Medium, clad: Medium, subpixel: bool) -> Simulation {
    let dx = 50e-9;
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([0.0; 3], [3e-6, 1e-6, 1e-6], MeshSpec::Uniform { dx: [dx; 3] }, b, 1e-15);
    sim.subpixel = subpixel;
    sim.background = clad.name.clone();
    sim.media.push(clad);
    let cn = core.name.clone();
    sim.media.push(core);
    sim.structures.push(Structure { name: "rod".into(), shape: Shape::Sphere { center: [1.5e-6, 0.5e-6, 0.5e-6], radius: 0.33e-6 }, medium: cn });
    let f0 = C0 / 1.55e-6;
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: 0.5e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(f0, 0.1 * f0), amplitude: 1.0, angle: 0.0, min: None, max: None });
    sim
}

#[test]
fn mixed_voxels_of_transparent_dispersive_dielectrics() {
    let sim = scene(glass("core", 1.5, 2.4), glass("clad", 1.2, 0.9), true);
    let p = compile(&sim, &CompileOptions::default()).unwrap();
    println!("mixed voxels {:?}", p.disp_subpixel);
    assert!(p.disp_subpixel[0] > 100, "{:?}", p.disp_subpixel);
    assert_eq!(p.disp_subpixel[1], 0);
    // every mixed voxel lies between the two media at w_c and keeps a weight in (0, 1]
    let d = p.disp.as_ref().unwrap();
    let w = TAU * C0 / 1.55e-6;
    let (lo, hi) = (sim.media[0].eps_complex(0, w).re, sim.media[1].eps_complex(0, w).re);
    let mut n = 0;
    for c in &d.comps {
        let a = c.comp.axis();
        for &t in &c.mixed {
            let s = c.idx[t as usize] as usize;
            // (in f64, so the test also builds with the double-precision feature)
            let eps_inf = C0 * p.dt / p.cb[a][s] as f64;
            let wt = c.scale[t as usize] as f64 * eps_inf;
            assert!(wt > 0.0 && wt <= 1.05, "weight {wt}");
            assert!(eps_inf >= 1.0, "eps_inf {eps_inf}");
            let row = c.matid[t as usize] as usize;
            let chi = sim.media[row].eps_complex(0, w).re - sim.media[row].eps[0];
            let eff = eps_inf as f64 + wt as f64 * chi;
            assert!(eff > lo.min(hi) - 1e-3 && eff < lo.max(hi) + 1e-3, "eps_eff {eff} outside [{lo}, {hi}]");
            n += 1;
        }
    }
    assert_eq!(n, p.disp_subpixel[0]);
    // the pure voxels are those of the staircase
    let st = compile(&scene(glass("core", 1.5, 2.4), glass("clad", 1.2, 0.9), false), &CompileOptions::default()).unwrap();
    for (c, cs) in d.comps.iter().zip(st.disp.as_ref().unwrap().comps.iter()) {
        assert_eq!(c.idx.len(), cs.idx.len());
        let mut k = 0;
        for t in 0..c.idx.len() {
            if k < c.mixed.len() && c.mixed[k] as usize == t {
                k += 1;
                continue;
            }
            assert_eq!((c.scale[t], c.matid[t]), (cs.scale[t], cs.matid[t]));
        }
    }
}

#[test]
fn staircase_without_subpixel_or_sources_or_for_metals() {
    let off = compile(&scene(glass("core", 1.5, 2.4), glass("clad", 1.2, 0.9), false), &CompileOptions::default()).unwrap();
    assert_eq!(off.disp_subpixel, [0, 0]);
    let mut nosrc = scene(glass("core", 1.5, 2.4), glass("clad", 1.2, 0.9), true);
    nosrc.sources.clear();
    assert_eq!(compile(&nosrc, &CompileOptions::default()).unwrap().disp_subpixel, [0, 0]);
    // a Drude metal in a glass: staircased
    let metal = Medium::vacuum().with_name("au").with_poles(vec![Pole::Drude { omega_p: 1.3e16, gamma: 1e14 }]);
    let m = compile(&scene(metal, glass("clad", 1.2, 0.9), true), &CompileOptions::default()).unwrap();
    assert_eq!(m.disp_subpixel, [0, 0]);
    // a lossy Lorentz medium (Im eps > 1e-2 Re eps at w_c): staircased
    let lossy = Medium::dielectric(1.5).with_name("lossy").with_poles(vec![Pole::Lorentz { delta_eps: 2.0, omega0: TAU * C0 / 1.6e-6, gamma: 3e14 }]);
    assert_eq!(compile(&scene(lossy, glass("clad", 1.2, 0.9), true), &CompileOptions::default()).unwrap().disp_subpixel, [0, 0]);
}


#[test]
fn mixed_voxels_are_exact_at_the_reference_frequency() {
    // eps_inf + w chi_A (discrete, at w_c) of every dispersive voxel equals the
    // permittivity plain-dielectric smoothing gives with the media's values at w_c
    let core = glass("core", 1.5, 2.4);
    let clad = glass("clad", 1.2, 0.9);
    let mut sim = scene(core.clone(), clad.clone(), true);
    sim.courant = 0.4;
    let p = compile(&sim, &CompileOptions::default()).unwrap();
    let w = TAU * C0 / 1.55e-6;
    let dt = p.dt;
    let fr = |m: &Medium| Medium::dielectric(m.eps[0] + m.poles.iter().map(|q| q.chi_discrete(w, dt).re).sum::<f64>()).with_name(&m.name);
    let mut plain = scene(fr(&core), fr(&clad), true);
    plain.courant = 0.4;
    let q = compile(&plain, &CompileOptions::default()).unwrap();
    assert!((q.dt / dt - 1.0).abs() < 1e-12);
    let cdt = C0 * dt;
    let media = [clad, core];
    let mut worst = 0f64;
    for c in &p.disp.as_ref().unwrap().comps {
        let a = c.comp.axis();
        for t in 0..c.idx.len() {
            // (rows past the media are their forms inside the stabilised layers)
            let Some(m) = media.get(c.matid[t] as usize) else { continue };
            let s = c.idx[t] as usize;
            let einf = cdt / p.cb[a][s] as f64;
            let chi: f64 = m.poles.iter().map(|x| x.chi_discrete(w, dt).re).sum();
            let eff = einf + c.scale[t] as f64 * einf * chi;
            let reference = cdt / q.cb[a][s] as f64;
            worst = worst.max((eff / reference - 1.0).abs());
        }
    }
    println!("largest relative difference {worst:.2e}");
    assert!(worst < 1e-6, "{worst}");
}

/// Two lossless Lorentz poles, one infrared and one ultraviolet: the form of
/// `fitted_medium` for a glass or nitride over a band around 1.55 um.
fn two_pole(name: &str, eps_inf: f64, d_ir: f64, w_ir: f64, d_uv: f64, w_uv: f64) -> Medium {
    Medium::dielectric(eps_inf)
        .with_name(name)
        .with_poles(vec![Pole::Lorentz { delta_eps: d_ir, omega0: w_ir, gamma: 0.0 }, Pole::Lorentz { delta_eps: d_uv, omega0: w_uv, gamma: 0.0 }])
}

/// A 90 degree bend (an arc polygon and two straight boxes that abut it,
/// guides running into the layers on two sides), z invariant over one cell.
fn bend(core: Medium, clad: Medium) -> Simulation {
    let dx = 22e-9;
    let (r, w) = (3e-6, 1e-6);
    let mut b = Boundaries::all(Boundary::Periodic);
    b.x = [Boundary::pml(), Boundary::pml()];
    b.y = [Boundary::pml(), Boundary::pml()];
    let mut sim = Simulation::new([-2e-6, -1.5e-6, 0.0], [r + 1.5e-6, r + 2e-6, dx], MeshSpec::Uniform { dx: [dx; 3] }, b, 1e-15);
    sim.subpixel = true;
    sim.courant = 0.5;
    sim.background = clad.name.clone();
    sim.media.push(clad);
    let cn = core.name.clone();
    sim.media.push(core);
    let n = 2000;
    let arc = |rr: f64, i: usize| {
        let t = -std::f64::consts::FRAC_PI_2 * (1.0 - i as f64 / (n - 1) as f64);
        [rr * t.cos(), r + rr * t.sin()]
    };
    let mut v: Vec<[f64; 2]> = (0..n).map(|i| arc(r + w / 2.0, i)).collect();
    v.extend((0..n).rev().map(|i| arc(r - w / 2.0, i)));
    sim.structures.push(Structure { name: "arc".into(), shape: Shape::PolySlab { vertices: v, axis: 2, slab_min: -1.0, slab_max: 1.0 }, medium: cn.clone() });
    sim.structures.push(Structure { name: "in".into(), shape: Shape::Box { min: [-10e-6, -w / 2.0, -1.0], max: [0.0, w / 2.0, 1.0] }, medium: cn.clone() });
    sim.structures.push(Structure { name: "out".into(), shape: Shape::Box { min: [r - w / 2.0, r, -1.0], max: [r + w / 2.0, 20e-6, 1.0] }, medium: cn });
    let f0 = C0 / 1.55e-6;
    sim.sources.push(Source::PlaneWave { name: "pw".into(), axis: 0, position: -1.0e-6, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(f0, 0.1 * f0), amplitude: 1.0, angle: 0.0, min: None, max: None });
    sim
}

#[test]
fn mixed_voxels_of_a_bend_with_abutting_structures_are_exact_at_the_reference_frequency() {
    // fitted Si3N4 in fitted SiO2 (1450-1650 nm); every voxel outside the
    // layers, including those on the faces where the boxes meet the arc,
    // has at w_c the permittivity of plain-dielectric smoothing
    let core = two_pole("core", 3.4600695223367763, 1.9194178842153427, 211666996591559.78, 0.5616296516408822, 6065966280349117.0);
    let clad = two_pole("clad", 2.0941363354068363, 4.7333334457815575, 86257481359460.22, 0.012724492163414746, 3100006070384575.0);
    let p = compile(&bend(core.clone(), clad.clone()), &CompileOptions::default()).unwrap();
    assert!(p.disp_subpixel[0] > 1000 && p.disp_subpixel[1] == 0, "{:?}", p.disp_subpixel);
    let w = TAU * C0 / 1.55e-6;
    let dt = p.dt;
    let fr = |m: &Medium| Medium::dielectric(m.eps[0] + m.poles.iter().map(|q| q.chi_discrete(w, dt).re).sum::<f64>()).with_name(&m.name);
    let q = compile(&bend(fr(&core), fr(&clad)), &CompileOptions::default()).unwrap();
    assert!((q.dt / dt - 1.0).abs() < 1e-12);
    let cdt = C0 * dt;
    let media = [clad, core];
    let (mut worst, mut n) = (0f64, 0);
    for c in &p.disp.as_ref().unwrap().comps {
        let a = c.comp.axis();
        for t in 0..c.idx.len() {
            // (rows past the media are their damped variants inside the layers)
            let Some(m) = media.get(c.matid[t] as usize) else { continue };
            let s = c.idx[t] as usize;
            let einf = cdt / p.cb[a][s] as f64;
            let chi: f64 = m.poles.iter().map(|x| x.chi_discrete(w, dt).re).sum();
            let eff = einf + c.scale[t] as f64 * einf * chi;
            worst = worst.max((eff / (cdt / q.cb[a][s] as f64) - 1.0).abs());
            n += 1;
        }
    }
    println!("{n} voxels, largest relative difference {worst:.2e}");
    assert!(worst < 1e-6, "{worst}");
}
