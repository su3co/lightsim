//! Meshes and imported grids: graded meshes, index grids.

use super::*;

/// Non-uniform (graded) mesh: a plane wave crossing a refined region (4x
/// finer cells) behaves like on the uniform coarse grid — same transmitted
/// flux, no spurious reflection from the grading.
pub fn nonuniform_grid(run: Runner) -> Result<CaseReport> {
    let n_cross = 2usize;
    let dxc = LAMBDA / 25.0;
    let w = n_cross as f64 * dxc;
    let mk = |refine: bool| -> Simulation {
        let mut b = Boundaries::all(Boundary::Periodic);
        b.x = [Boundary::pml(), Boundary::pml()];
        let overrides = if refine {
            vec![crate::grid::MeshOverride { min: [1.8e-6, -1.0, -1.0], max: [2.2e-6, 1.0, 1.0], dx: [dxc / 4.0, dxc, dxc] }]
        } else {
            vec![]
        };
        let mesh = MeshSpec::Auto {
            wavelength: LAMBDA,
            steps_per_wavelength: 25.0,
            grading: 1.2,
            min_step: None,
            max_step: None,
            overrides,
            symmetry: [None; 3],
        };
        let mut sim = Simulation::new([0.0; 3], [4e-6, w, w], mesh, b, 60e-15);
        sim.sources.push(plane_source(1.0e-6, 1.0));
        let freqs = band(3);
        sim.monitors.push(flux_x("T", 3.0e-6, w, &freqs));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim
    };
    let fine = run(&mk(true))?;
    let coarse = run(&mk(false))?;
    let freqs = band(3);
    let mut rep = CaseReport::new("nonuniform_grid");
    rep.mcells_per_s = fine.stats.mcells_per_s;
    for (i, f) in freqs.iter().enumerate() {
        let tc = coarse.flux["T"][i];
        rep.check(&format!("T_graded/T_uniform @ {:.3} f0", f / f0()), fine.flux["T"][i] / tc, 1.0, 5e-3);
        rep.check(&format!("|R|/T_uniform @ {:.3} f0", f / f0()), (fine.flux["R"][i] / tc).abs(), 0.0, 1e-3);
    }
    Ok(rep)
}

/// max |a - b| / max |a| over the DFT monitor `name`, component `c`.
pub fn dft_rel_diff(a: &Results, b: &Results, name: &str, c: Component) -> f64 {
    let (fa, fb) = (a.dft[name].field(c).unwrap(), b.dft[name].field(c).unwrap());
    assert_eq!(fa.dims, fb.dims);
    let (mut peak, mut err) = (0f64, 0f64);
    for f in 0..fa.dims[0] {
        for i in 0..fa.dims[1] {
            for j in 0..fa.dims[2] {
                for k in 0..fa.dims[3] {
                    let (x, y) = (fa.at(f, i, j, k), fb.at(f, i, j, k));
                    peak = peak.max(x.norm());
                    err = err.max((x - y).norm());
                }
            }
        }
    }
    err / peak
}

/// Power reflectance of a lossless multilayer at normal incidence
/// (characteristic-matrix method): incidence medium `n_in`, `layers` =
/// (index, thickness) in order, exit medium `n_out`, vacuum wavenumber `k0`.
pub fn multilayer_reflectance(n_in: f64, layers: &[(f64, f64)], n_out: f64, k0: f64) -> f64 {
    let i = Complex64::new(0.0, 1.0);
    let one = Complex64::new(1.0, 0.0);
    let mut m = [[one, 0.0 * one], [0.0 * one, one]];
    for &(n, d) in layers {
        let (sn, cs) = (k0 * n * d).sin_cos();
        let l = [[one * cs, i * sn / n], [i * sn * n, one * cs]];
        m = [
            [m[0][0] * l[0][0] + m[0][1] * l[1][0], m[0][0] * l[0][1] + m[0][1] * l[1][1]],
            [m[1][0] * l[0][0] + m[1][1] * l[1][0], m[1][0] * l[0][1] + m[1][1] * l[1][1]],
        ];
    }
    let b = m[0][0] + m[0][1] * n_out;
    let c = m[1][0] + m[1][1] * n_out;
    ((b * n_in - c) / (b * n_in + c)).norm_sqr()
}

/// Imported / graded permittivity (`Shape::IndexGrid`).
///
/// (a) A constant grid is the same structure as a staircased `Box` of that
///     permittivity: identical DFT fields and flux.
/// (b) A linear permittivity ramp 1 -> 4 over 1 um, as an anti-reflection
///     transition into an eps = 4 half-space and as a free-standing graded
///     layer (abrupt 4 -> 1 back face), against a fine transfer-matrix
///     calculation of the same continuous profile.
pub fn index_grid(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("index_grid");
    let n_cross = 2usize;

    // ---- (a) constant grid == box (faces deliberately off the Yee nodes) ----
    {
        let dx = LAMBDA / 40.0;
        let w = n_cross as f64 * dx;
        let freqs = band(3);
        let (lo, hi) = ([2.013e-6, -1.0, -1.0], [2.487e-6, 1.0, 1.0]);
        let build = |shape: Shape| {
            let mut sim = column(dx, 4e-6, n_cross);
            sim.subpixel = false;
            sim.media.push(Medium::dielectric(4.0).with_name("d"));
            sim.structures.push(Structure { name: "s".into(), shape, medium: "d".into() });
            sim.sources.push(plane_source(1.0e-6, 1.0));
            sim.monitors.push(flux_x("T", 3.2e-6, w, &freqs));
            sim.monitors.push(Monitor::DftField {
                name: "e".into(),
                min: [0.5e-6, 0.0, 0.0],
                max: [3.5e-6, 0.0, 0.0],
                components: vec![Component::Ey],
                freqs: freqs.clone(),
                apodization: Apodization::default(),
                stride: 1,
            });
            sim.run_time = 90e-15;
            sim
        };
        let a = run(&build(Shape::Box { min: lo, max: hi }))?;
        // 3 x 1 x 2 samples, all 4.0
        let grid = Shape::IndexGrid { min: lo, max: hi, dims: [3, 1, 2], eps: vec![4.0; 6], eps_axes: None, mu: vec![] };
        let b = run(&build(grid))?;
        rep.check_abs("const grid vs box: max |dE| / peak", dft_rel_diff(&a, &b, "e", Component::Ey), 0.0, 1e-6);
        let dt_max = (0..freqs.len()).map(|i| (b.flux["T"][i] / a.flux["T"][i] - 1.0).abs()).fold(0.0, f64::max);
        rep.check_abs("const grid vs box: max |dT| / T", dt_max, 0.0, 1e-6);
    }

    // ---- (b) linear eps ramp vs transfer matrix ----
    let dx = LAMBDA / 60.0;
    let w = n_cross as f64 * dx;
    let freqs = band(5);
    let (x0, len) = (2.0e-6, 1.0e-6);
    let (e0, e1) = (1.0, 4.0);
    for substrate in [true, false] {
        // Free-standing layer: end the grid half a cell past the last node it
        // owns, so the (staircased) back face sits exactly at the box face.
        let length = if substrate { len } else { len + 0.5 * dx };
        let slope = (e1 - e0) / len;
        let ns = 41usize;
        let eps: Vec<f64> = (0..ns).map(|i| e0 + slope * length * i as f64 / (ns - 1) as f64).collect();
        let mut sim = column(dx, 5e-6, n_cross);
        sim.media.push(Medium::dielectric(e1).with_name("sub"));
        // The medium's own eps is irrelevant (the grid replaces it).
        sim.media.push(Medium::dielectric(2.0).with_name("graded"));
        if substrate {
            // painted first, underneath the grid: the ramp ends on eps = 4
            sim.structures.push(Structure {
                name: "sub".into(),
                shape: Shape::Box { min: [x0 + 0.5 * len, -1.0, -1.0], max: [10e-6, 1.0, 1.0] },
                medium: "sub".into(),
            });
        }
        sim.structures.push(Structure {
            name: "ramp".into(),
            shape: Shape::IndexGrid { min: [x0, -1.0, -1.0], max: [x0 + length, 1.0, 1.0], dims: [ns, 1, 1], eps, eps_axes: None, mu: vec![] },
            medium: "graded".into(),
        });
        sim.sources.push(plane_source(1.0e-6, 1.0));
        sim.monitors.push(flux_x("T", 4.0e-6, w, &freqs));
        sim.monitors.push(flux_x("R", 0.5e-6, w, &freqs));
        sim.run_time = 120e-15;
        let res = run(&sim)?;
        rep.mcells_per_s = res.stats.mcells_per_s;
        let sp = spectrum_power(&pulse(), &freqs, res.dt, res.stats.steps);
        // transfer matrix of the continuous profile, 4000 sub-layers
        let nl = 4000usize;
        let d = length / nl as f64;
        let layers: Vec<(f64, f64)> = (0..nl).map(|i| ((e0 + slope * (i as f64 + 0.5) * d).sqrt(), d)).collect();
        let n_out = if substrate { e1.sqrt() } else { 1.0 };
        let label = if substrate { "ramp->eps4" } else { "free ramp" };
        for (i, f) in freqs.iter().enumerate() {
            let inc = 0.5 / ETA0 * sp[i] * w * w;
            let r_exp = multilayer_reflectance(1.0, &layers, n_out, TAU * f / C0);
            rep.check(&format!("{label} T @ {:.3} f0", f / f0()), res.flux["T"][i] / inc, 1.0 - r_exp, 0.01);
            rep.check_abs(&format!("{label} R @ {:.3} f0", f / f0()), -res.flux["R"][i] / inc, r_exp, if substrate { 2e-4 } else { 3e-3 });
        }
    }
    Ok(rep)
}
