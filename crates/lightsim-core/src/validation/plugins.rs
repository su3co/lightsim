//! Validation of user-defined material plug-ins ([`crate::plugin`]).

use super::*;
use crate::plugin::PluginRef;

/// Largest |a - b| over the samples of probe `name` (first component),
/// relative to the peak of `a`.
fn probe_diff(a: &Results, b: &Results, name: &str) -> f64 {
    let (x, y) = (&a.time[name].fields[0].data, &b.time[name].fields[0].data);
    let peak = x.iter().fold(0f64, |m, v| m.max((*v as f64).abs()));
    x.iter().zip(y).fold(0f64, |m, (p, q)| m.max((*p as f64 - *q as f64).abs())) / peak
}

/// A 1-D column with a slab of `med` (2 um from x = 1.5 um) and a probe
/// behind it.
fn plugin_column(med: Medium, dx: f64, run_time: f64) -> Simulation {
    let mut sim = column(dx, 5.0e-6, 2);
    let name = med.name.clone();
    sim.media.push(med);
    sim.structures.push(Structure { name: "slab".into(), shape: Shape::Box { min: [1.5e-6, -1.0, -1.0], max: [3.5e-6, 1.0, 1.0] }, medium: name });
    sim.sources.push(plane_source(1.0e-6, 1.0));
    sim.monitors.push(Monitor::Probe { name: "p".into(), position: [4.0e-6, 0.0, 0.0], components: vec![Component::Ey] });
    sim.monitors.push(Monitor::Probe { name: "in".into(), position: [2.5e-6, 0.0, 0.0], components: vec![Component::Ey] });
    sim.run_time = run_time;
    sim
}

/// User-defined material plug-ins (the example plug-ins of
/// [`crate::plugin::examples`], registered by this case):
///
/// - one Lorentz pole written as a plug-in against the built-in pole in a
///   slab (eps_inf 1.5, delta_eps 1.5, w0 = 1.3 w_0, gamma = 0.1 w_0, conductive
///   background sigma = 2e4 S/m, and lossless with the slab faces between
///   nodes): the field transmitted through the slab and inside it (the CPU engine does the same arithmetic in the same order:
///   identical; the GPU computes the plug-in on the host);
/// - the instantaneous Kerr medium (chi3, solved implicitly for E^{n+1}) as a
///   plug-in against the built-in chi3 medium (third harmonic about 1 % of
///   the fundamental): the coefficient chi3 / eps is formed from 1 / eps in
///   single precision, which differs from the built-in chi3 / eps by up to
///   one unit in the last place;
/// - a plug-in that amplifies the field each step is stopped by the solver's
///   divergence check with an error instead of running on.
pub fn plugin_media(run: Runner) -> Result<CaseReport> {
    crate::plugin::examples::register_examples();
    let mut rep = CaseReport::new("plugin_media");
    let w0 = TAU * f0();
    let (de, wl, gl) = (1.5, 1.3 * w0, 0.1 * w0);
    let dx = LAMBDA / 60.0;
    let mut builtin = Medium::conductor(1.5, 2e4).with_name("lor");
    builtin.poles = vec![Pole::Lorentz { delta_eps: de, omega0: wl, gamma: gl }];
    let mut plug = Medium::conductor(1.5, 2e4).with_name("lor");
    plug.plugin = Some(PluginRef { name: "example_lorentz".into(), params: serde_json::json!({"delta_eps": de, "omega0": wl, "gamma": gl}) });
    let a = run(&plugin_column(builtin, dx, 150e-15))?;
    let b = run(&plugin_column(plug, dx, 150e-15))?;
    rep.mcells_per_s = b.stats.mcells_per_s;
    rep.check_abs(&format!("Lorentz plug-in vs built-in pole: transmitted field (time {:.2} vs {:.2} s)", b.stats.seconds, a.stats.seconds), probe_diff(&a, &b, "p"), 0.0, 2e-5);
    rep.check_abs("Lorentz plug-in vs built-in pole: field inside the slab", probe_diff(&a, &b, "in"), 0.0, 2e-5);
    // lossless background, slab faces off the nodes (plug-in media are staircased like dispersive media)
    let mut lb = Medium::dielectric(1.5).with_name("lor");
    lb.poles = vec![Pole::Lorentz { delta_eps: de, omega0: wl, gamma: gl }];
    let mut lp = Medium::dielectric(1.5).with_name("lor");
    lp.plugin = Some(PluginRef { name: "example_lorentz".into(), params: serde_json::json!({"delta_eps": de, "omega0": wl, "gamma": gl}) });
    let shift = |mut s: Simulation| {
        if let Shape::Box { min, max } = &mut s.structures[0].shape {
            min[0] += 0.37 * dx;
            max[0] += 0.61 * dx;
        }
        s
    };
    let a = run(&shift(plugin_column(lb, dx, 150e-15)))?;
    let b = run(&shift(plugin_column(lp, dx, 150e-15)))?;
    rep.check_abs("Lorentz plug-in vs built-in pole, lossless, faces between nodes: transmitted field", probe_diff(&a, &b, "p"), 0.0, 2e-5);

    let k_l = TAU / LAMBDA * 2e-6;
    let chi = 0.01 / (1.5 * k_l);
    let mut kb = Medium::dielectric(1.0).with_name("kerr");
    kb.chi3 = [chi; 3];
    let mut kp = Medium::dielectric(1.0).with_name("kerr");
    kp.plugin = Some(PluginRef { name: "example_kerr".into(), params: serde_json::json!({ "chi3": chi }) });
    let dxk = LAMBDA / 80.0;
    let a = run(&plugin_column(kb, dxk, 45e-15))?;
    let b = run(&plugin_column(kp, dxk, 45e-15))?;
    rep.check_abs(&format!("Kerr plug-in vs built-in chi3: transmitted field (time {:.2} vs {:.2} s)", b.stats.seconds, a.stats.seconds), probe_diff(&a, &b, "p"), 0.0, 2e-5);
    rep.check_abs("Kerr plug-in vs built-in chi3: field inside the slab", probe_diff(&a, &b, "in"), 0.0, 2e-5);

    let mut rp = Medium::dielectric(1.0).with_name("gain");
    rp.plugin = Some(PluginRef { name: "example_runaway".into(), params: serde_json::json!({ "gain": 1.002 }) });
    let mut sim = plugin_column(rp, dx, 2000e-15);
    sim.shutoff_interval = 100;
    let t0 = std::time::Instant::now();
    let r = run(&sim);
    let stopped = matches!(&r, Err(e) if e.to_string().contains("unstable"));
    rep.check_abs(&format!("energy-gaining plug-in stopped by the divergence check ({:.2} s)", t0.elapsed().as_secs_f64()), if stopped { 0.0 } else { 1.0 }, 0.0, 0.0);
    Ok(rep)
}
