//! Scenes shared by the CPU and GPU tests of the gradient engines.
#![doc(hidden)]

use super::{Channel, DesignGrid, Objective, PortSpec, SessionSpec};
use crate::geometry::{Shape, Structure};
use crate::grid::{Component, MeshSpec};
use crate::materials::Medium;
use crate::sources::{Direction, ModeProfile, Source, TimeSignal};
use crate::spec::{Boundaries, Boundary, PmlParams, Simulation};

pub const UM: f64 = 1e-6;
pub const F0: f64 = 299_792_458.0 / 1.55e-6;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn sym(&mut self) -> f32 {
        (2.0 * self.next() - 1.0) as f32
    }
}

pub fn pml(layers: usize) -> Boundary {
    Boundary::Pml { layers, params: PmlParams::default(), inside: false }
}

pub struct Case {
    pub name: &'static str,
    pub boundaries: Boundaries,
    pub design_sigma: f64,
    pub extra: Option<Medium>,
    pub tol: f64,
}

pub fn reverse_scene(c: &Case) -> (Simulation, Vec<DesignGrid>, Objective) {
    let dx = 0.05 * UM;
    let mut sim = Simulation::new([0.0; 3], [1.6 * UM, 1.2 * UM, 1.0 * UM], MeshSpec::Uniform { dx: [dx; 3] }, c.boundaries.clone(), 28e-15);
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("core"));
    let mut d = Medium::dielectric(6.0).with_name("design");
    d.sigma = [c.design_sigma; 3];
    sim.media.push(d);
    sim.background = "ox".into();
    sim.structures.push(Structure { name: "wg".into(), shape: Shape::Box { min: [-1.0, 0.4 * UM, 0.3 * UM], max: [1.0, 0.8 * UM, 0.7 * UM] }, medium: "core".into() });
    if let Some(m) = &c.extra {
        sim.media.push(m.clone());
        sim.structures.push(Structure { name: "x".into(), shape: Shape::Box { min: [1.0 * UM, 0.4 * UM, 0.3 * UM], max: [1.2 * UM, 0.8 * UM, 0.7 * UM] }, medium: m.name.clone() });
    }
    let sig = TimeSignal::GaussianPulse { freq0: F0, fwidth: 0.3 * F0, offset: 4.0, phase: 0.0, amplitude: 1.0 };
    sim.sources.push(Source::PointDipole { name: "s".into(), position: [0.3 * UM, 0.6 * UM, 0.5 * UM], polarization: Component::Ez, signal: sig, amplitude: 1e-15 });
    let grids = vec![DesignGrid { name: "g".into(), min: [0.52 * UM, 0.33 * UM, 0.3 * UM], max: [0.93 * UM, 0.91 * UM, 0.7 * UM], dims: [5, 6, 2], medium: "design".into() }];
    let obj = Objective::Flux { axis: 0, position: 1.35 * UM, min: [1.35 * UM, 0.2 * UM, 0.1 * UM], max: [1.35 * UM, 1.0 * UM, 0.9 * UM], t_start: Some(6e-15), t_end: Some(24e-15), taper: Some(3e-15) };
    (sim, grids, obj)
}

/// [`reverse_scene`] with the design grid widened to the faces of the
/// region in y and z: its outermost voxels are the last cells before the
/// absorbing layers (which are added outside the region).
pub fn reverse_scene_tight(c: &Case) -> (Simulation, Vec<DesignGrid>, Objective) {
    let (sim, mut grids, obj) = reverse_scene(c);
    grids[0].min = [0.52 * UM, 0.0, 0.0];
    grids[0].max = [0.93 * UM, sim.region.max[1], sim.region.max[2]];
    grids[0].dims = [5, 7, 6];
    (sim, grids, obj)
}

pub fn samples(rng: &mut Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n).map(|_| lo + (hi - lo) * rng.next()).collect()
}

/// A Gaussian "mode" profile (TE-like: E_y, H_z) on a plane normal to x: not
/// an eigenmode, but enough to exercise sources, overlaps and normalisation.
pub fn fake_profile(sim: &Simulation, x: f64, min: [f64; 3], max: [f64; 3]) -> ModeProfile {
    let grid = sim.build_grid().unwrap();
    let (lo, hi) = crate::plan::transverse_box(&grid, 0, Some(&min), Some(&max));
    let dims = [hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let _ = x;
    let mut e = vec![0.0; dims[0] * dims[1]];
    let mut h = vec![0.0; dims[0] * dims[1]];
    let (yc, zc, w) = (0.5 * (min[1] + max[1]), 0.5 * (min[2] + max[2]), 0.25 * UM);
    for u in 0..dims[0] {
        for v in 0..dims[1] {
            let y = grid.center(1, (lo[1] + u).min(grid.n()[1] - 1));
            let z = grid.node(2, lo[2] + v);
            let g = (-((y - yc).powi(2) + (z - zc).powi(2)) / (w * w)).exp();
            e[u * dims[1] + v] = g;
            h[u * dims[1] + v] = g * 1.9 / 376.73;
        }
    }
    let zero = vec![0.0; dims[0] * dims[1]];
    ModeProfile { neff: 1.9, neff_imag: 0.0, dims, e_re: [e, zero.clone()], e_im: [zero.clone(), zero.clone()], h_re: [zero.clone(), h], h_im: [zero.clone(), zero], freq: F0, h_offset: Some(0.0) }
}

pub fn session_spec() -> SessionSpec {
    let dx = 0.05 * UM;
    let b = Boundaries { x: [pml(6), pml(6)], y: [pml(6), pml(6)], z: [Boundary::Pec, Boundary::Pec] };
    let mut sim = Simulation::new([0.0; 3], [2.4 * UM, 1.2 * UM, 0.6 * UM], MeshSpec::Uniform { dx: [dx; 3] }, b, 60e-15);
    sim.shutoff = Some(1e-5);
    sim.shutoff_interval = 50;
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("core"));
    sim.media.push(Medium::dielectric(6.0).with_name("design"));
    sim.background = "ox".into();
    sim.structures.push(Structure { name: "wg".into(), shape: Shape::Box { min: [-1.0, 0.4 * UM, 0.15 * UM], max: [1.0, 0.8 * UM, 0.45 * UM] }, medium: "core".into() });
    let port = |name: &str, x: f64, dir: Direction| {
        let (min, max) = ([x, 0.1 * UM, 0.05 * UM], [x, 1.1 * UM, 0.55 * UM]);
        PortSpec { name: name.into(), axis: 0, position: x, min, max, profiles: vec![fake_profile(&sim, x, min, max)], direction: dir }
    };
    let ports = vec![port("in", 0.4 * UM, Direction::Plus), port("out", 2.0 * UM, Direction::Minus)];
    let grids = vec![
        DesignGrid { name: "lower".into(), min: [0.9 * UM, 0.33 * UM, 0.15 * UM], max: [1.5 * UM, 0.87 * UM, 0.3 * UM], dims: [6, 5, 1], medium: "design".into() },
        DesignGrid { name: "upper".into(), min: [0.9 * UM, 0.33 * UM, 0.3 * UM], max: [1.5 * UM, 0.87 * UM, 0.45 * UM], dims: [4, 4, 1], medium: "design".into() },
    ];
    SessionSpec {
        sim,
        grids,
        ports,
        inputs: vec![Channel::Port { port: 0, mode: 0 }],
        outputs: vec![Channel::Port { port: 1, mode: 0 }, Channel::Port { port: 0, mode: 0 }],
        freqs: vec![0.97 * F0, F0, 1.03 * F0],
        signal: TimeSignal::gaussian(F0, 0.12 * F0),
        normalize: "monitor".into(),
        components: Component::E.to_vec(),
    }
}


/// Profiles of every port scaled to unit power, as the overlap measures it
/// ([`super::session::port_power`]).
pub fn unit_power_ports(spec: &mut SessionSpec) {
    let grid = spec.sim.build_grid().unwrap();
    let b = &spec.sim.boundaries;
    let periodic = [b.x[0].is_periodic(), b.y[0].is_periodic(), b.z[0].is_periodic()];
    for port in spec.ports.iter_mut() {
        for k in 0..port.profiles.len() {
            let f = 1.0 / super::session::port_power(&grid, periodic, port, k).abs().sqrt();
            let q = &mut port.profiles[k];
            for arr in q.e_re.iter_mut().chain(q.e_im.iter_mut()).chain(q.h_re.iter_mut()).chain(q.h_im.iter_mut()) {
                arr.iter_mut().for_each(|x| *x *= f);
            }
        }
    }
}

/// A session whose ports carry an exact eigenmode: a cell periodic in y and
/// z (CPML in x) filled with one medium, in which the "mode" of a port plane
/// is the plane wave (E_y, H_z), unit power, and two adjacent design grids
/// between the ports. S_21 is the transmission of the zeroth order (the
/// period is below the wavelength in the medium). Runs until the fields
/// have decayed, every step sampled, no shutoff.
pub fn plane_wave_session_spec() -> SessionSpec {
    let dx = 0.05 * UM;
    let b = Boundaries { x: [pml(8), pml(8)], y: [Boundary::Periodic, Boundary::Periodic], z: [Boundary::Periodic, Boundary::Periodic] };
    let mut sim = Simulation::new([0.0; 3], [2.4 * UM, 0.6 * UM, 0.4 * UM], MeshSpec::Uniform { dx: [dx; 3] }, b, 400e-15);
    let n = 2.1f64.sqrt();
    sim.media.push(Medium::dielectric(2.1).with_name("ox"));
    sim.media.push(Medium::dielectric(6.0).with_name("design"));
    sim.background = "ox".into();
    sim.dft_stride = Some(1);
    let grid = sim.build_grid().unwrap();
    let port = |name: &str, x: f64, dir: Direction| {
        let (min, max) = ([x, 0.0, 0.0], [x, 0.6 * UM, 0.4 * UM]);
        let (lo, hi) = crate::plan::transverse_box(&grid, 0, Some(&min), Some(&max));
        let dims = [hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        let m = dims[0] * dims[1];
        // H on the half cell below the plane, with its phase exp(-i beta dx / 2)
        let h = num_complex::Complex64::from_polar(n / crate::constants::ETA0, -2.0 * std::f64::consts::PI * F0 * n / crate::constants::C0 * 0.5 * dx);
        let zero = vec![0.0; m];
        let profile = ModeProfile { neff: n, neff_imag: 0.0, dims, e_re: [vec![1.0; m], zero.clone()], e_im: [zero.clone(), zero.clone()], h_re: [zero.clone(), vec![h.re; m]], h_im: [zero.clone(), vec![h.im; m]], freq: F0, h_offset: Some(0.5 * dx) };
        PortSpec { name: name.into(), axis: 0, position: x, min, max, profiles: vec![profile], direction: dir }
    };
    let ports = vec![port("in", 0.4 * UM, Direction::Plus), port("out", 2.0 * UM, Direction::Minus)];
    let grids = vec![
        DesignGrid { name: "lower".into(), min: [0.9 * UM, 0.12 * UM, 0.1 * UM], max: [1.5 * UM, 0.48 * UM, 0.2 * UM], dims: [6, 4, 1], medium: "design".into() },
        DesignGrid { name: "upper".into(), min: [0.9 * UM, 0.12 * UM, 0.2 * UM], max: [1.5 * UM, 0.48 * UM, 0.3 * UM], dims: [4, 3, 1], medium: "design".into() },
    ];
    let mut spec = SessionSpec {
        sim,
        grids,
        ports,
        inputs: vec![Channel::Port { port: 0, mode: 0 }],
        outputs: vec![Channel::Port { port: 1, mode: 0 }],
        freqs: vec![0.9 * F0, F0, 1.1 * F0],
        signal: TimeSignal::gaussian(F0, 0.2 * F0),
        normalize: "source".into(),
        components: Component::E.to_vec(),
    };
    unit_power_ports(&mut spec);
    spec
}
