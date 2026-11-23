//! Validation of the sub-pixel smoothing of transparent dispersive
//! dielectrics ([`crate::raster::MixedCell`]).

use super::*;

/// A glass-like medium: eps_inf plus one ultraviolet Lorentz pole
/// (resonance at 0.1 um, lossless), the form of a Sellmeier fit.
fn uv_glass(name: &str, eps_inf: f64, d_eps: f64) -> Medium {
    Medium::dielectric(eps_inf).with_name(name).with_poles(vec![Pole::Lorentz { delta_eps: d_eps, omega0: TAU * C0 / 0.1e-6, gamma: 0.0 }])
}

/// A plain dielectric with the permittivity that the discrete pole update
/// of `m` has at angular frequency `w` with time step `dt`.
fn frozen(m: &Medium, w: f64, dt: f64) -> Medium {
    Medium::dielectric(m.eps[0] + m.poles.iter().map(|p| p.chi_discrete(w, dt).re).sum::<f64>()).with_name(&m.name)
}

const DX: f64 = 25e-9;
/// Every run uses this fraction of the Courant step, below the pole limit
/// of the media, so that all runs (dispersive and plain) share their time
/// step.
const COURANT: f64 = 0.9;

/// Scattering cross-section of an isolated sphere (radius 0.3 um, off the
/// grid nodes) of medium `core` in `clad` under a TFSF plane wave, at
/// `freqs`: the scattered power over the incident intensity, measured in a
/// run without the sphere (power through a plane inside the TFSF box), so
/// that the normalisation is the scheme's own in each medium.
fn sphere_scattering(run: Runner, core: &Medium, clad: &Medium, subpixel: bool, freqs: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
    let (sca, dt, mc) = sphere_run(run, core, clad, subpixel, freqs, true)?;
    let (inc, dt0, _) = sphere_run(run, core, clad, subpixel, freqs, false)?;
    anyhow::ensure!((dt0 / dt - 1.0).abs() < 1e-12, "the runs must share their time step");
    Ok((sca.iter().zip(&inc).map(|(s, i)| s / i).collect(), dt, mc))
}

fn sphere_run(run: Runner, core: &Medium, clad: &Medium, subpixel: bool, freqs: &[f64], sphere: bool) -> Result<(Vec<f64>, f64, f64)> {
    let l = 1.4e-6;
    let mut sim = Simulation::new([-l / 2.0; 3], [l / 2.0; 3], MeshSpec::Uniform { dx: [DX; 3] }, Boundaries::all_pml(), 150e-15);
    sim.subpixel = subpixel;
    sim.courant = COURANT;
    sim.media = vec![clad.clone(), core.clone()];
    sim.background = clad.name.clone();
    if sphere {
        sim.structures.push(Structure { name: "s".into(), shape: Shape::Sphere { center: [0.37 * DX, 0.21 * DX, -0.13 * DX], radius: 0.3e-6 }, medium: core.name.clone() });
    }
    let a = 0.45e-6;
    sim.sources.push(Source::Tfsf { name: "tf".into(), min: [-a; 3], max: [a; 3], axis: 0, direction: Direction::Plus, polarization: Component::Ey, signal: TimeSignal::gaussian(f0(), 0.3 * f0()), amplitude: 1.0 });
    let b = 0.56e-6;
    if sphere {
        sim.monitors.push(Monitor::FluxBox { name: "m".into(), min: [-b; 3], max: [b; 3], freqs: freqs.to_vec(), apodization: Apodization::default() });
    } else {
        // incident intensity: power through a 0.4 um square at the centre of the TFSF box
        let h = 0.2e-6;
        sim.monitors.push(Monitor::Flux { name: "m".into(), axis: 0, position: 0.0, min: [0.0, -h, -h], max: [0.0, h, h], freqs: freqs.to_vec(), apodization: Apodization::default() });
    }
    let res = run(&sim)?;
    Ok((res.flux["m"].clone(), res.dt, res.stats.mcells_per_s))
}

/// Sub-pixel smoothing of transparent dispersive dielectrics: an isolated
/// sphere of one glass-like pole medium in another (both dispersive, 3-D,
/// the curved interface at no particular grid position), scattered power at
/// f0 and at 0.9 and 1.1 f0.
///
/// The reference at each frequency f is the same sphere of plain dielectrics
/// with the media's permittivities at f (those of the discrete pole update,
/// same time step), sub-pixel smoothed as plain dielectrics are. At the
/// reference frequency w_c of the smoothing (f0, the sources' centre) the
/// smoothed dispersive run must agree with it far better than the
/// staircased one (a mixed voxel is exact at w_c); at 0.9 and 1.1 f0 the
/// difference also contains the dispersion that the mixed voxels freeze
/// (the cladding's), and must still beat the staircase.
pub fn dispersive_subpixel(run: Runner) -> Result<CaseReport> {
    let mut rep = CaseReport::new("dispersive_subpixel");
    let core = uv_glass("core", 1.5, 2.4);
    let clad = uv_glass("clad", 1.2, 0.9);
    let freqs = vec![0.9 * f0(), f0(), 1.1 * f0()];
    let (ps, dt, mc) = sphere_scattering(run, &core, &clad, true, &freqs)?;
    let (pst, _, _) = sphere_scattering(run, &core, &clad, false, &freqs)?;
    rep.mcells_per_s = mc;
    for (i, &f) in freqs.iter().enumerate() {
        let w = TAU * f;
        let (pr, dtr, _) = sphere_scattering(run, &frozen(&core, w, dt), &frozen(&clad, w, dt), true, &[f])?;
        anyhow::ensure!((dtr / dt - 1.0).abs() < 1e-12, "the runs must share their time step");
        let (es, est) = (ps[i] / pr[0] - 1.0, pst[i] / pr[0] - 1.0);
        let r = f / f0();
        rep.check_abs(&format!("{r:.1} f0: scattering cross-section, smoothed / plain-smoothed reference - 1 (staircase {est:+.2e})"), es, 0.0, if i == 1 { 5e-4 } else { 3e-3 });
        rep.check_abs(&format!("{r:.1} f0: |smoothed / staircase error| (< 0.3)"), (es / est).abs(), 0.0, 0.3);
    }
    Ok(rep)
}
