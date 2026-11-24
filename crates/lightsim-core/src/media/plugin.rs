//! User-defined material models (plug-ins).
//!
//! A downstream crate implements [`MaterialPlugin`] and registers a factory
//! under a name with [`register`]; a [`crate::materials::Medium`] whose
//! `plugin` field names it (with optional JSON parameters) then runs the
//! plug-in on every E sample it fills. Per step and per E component the
//! engine calls
//!
//! 1. [`MaterialPlugin::before`] before the linear E update, with E^n of the
//!    medium's cells: an explicit model advances its polarization here;
//! 2. [`MaterialPlugin::after`] after the linear update, with `e` holding
//!    `E_lin` (the field the linear update gave, curl H term included) and
//!    `inv_eps = 1 / (eps (1 + f))` (eps the cell's permittivity, f its
//!    conduction term): the plug-in writes E^{n+1}. A polarization change
//!    `dP / eps0` enters as `E^{n+1} = E_lin - inv_eps dP / eps0` (Ampere's
//!    law with D = eps0 eps E + P); an instantaneous model solves for
//!    E^{n+1} there.
//!
//! The per-cell state (`state_len` numbers per cell, zero at the start) is
//! the plug-in's own. This is the same split as the built-in dispersive and
//! nonlinear updates, so a plug-in can reproduce them exactly (see
//! [`examples`]).
//!
//! The CPU engine runs plug-ins directly; the GPU engine gathers the
//! medium's cells to the host and back each step (see `lightsim-cuda`). The
//! gradient engines, time reversal and complex (Bloch) fields refuse
//! plug-in media.

use crate::Real;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// What the engine tells a plug-in about the cells of one call.
#[derive(Clone, Copy, Debug)]
pub struct PluginCall {
    /// E component of the cells (0, 1, 2).
    pub axis: usize,
    /// Time step (s).
    pub dt: f64,
    /// Step index n (the update goes from n to n + 1).
    pub step: usize,
}

/// A user-defined material model; see the module documentation.
pub trait MaterialPlugin: Send {
    /// Numbers of state per cell.
    fn state_len(&self) -> usize {
        0
    }
    /// Before the linear E update: `e` = E^n of the cells (read only),
    /// `state` = `[cell][state_len]`.
    fn before(&mut self, call: &PluginCall, e: &[Real], state: &mut [Real]) {
        let _ = (call, e, state);
    }
    /// After the linear E update: `e` holds E_lin on entry and must hold
    /// E^{n+1} on exit; `e_prev` = E^n, `inv_eps` = 1 / (eps (1 + f)).
    fn after(&mut self, call: &PluginCall, e: &mut [Real], e_prev: &[Real], inv_eps: &[Real], state: &mut [Real]);
    /// An error the plug-in hit in a call (checked after every call; the run
    /// stops with it). A plug-in written in Python reports its exceptions
    /// here.
    fn error(&mut self) -> Option<String> {
        None
    }
}

/// Reference from a medium to a registered plug-in.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PluginRef {
    /// The name the factory was registered under.
    pub name: String,
    /// Parameters handed to the factory.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Makes one plug-in instance per medium, E component and slab.
pub type PluginFactory = dyn Fn(&serde_json::Value) -> Result<Box<dyn MaterialPlugin>> + Send + Sync;

fn registry() -> &'static Mutex<HashMap<String, Arc<PluginFactory>>> {
    static R: OnceLock<Mutex<HashMap<String, Arc<PluginFactory>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register (or replace) a plug-in factory under `name`.
pub fn register(name: &str, factory: Arc<PluginFactory>) {
    registry().lock().unwrap().insert(name.to_string(), factory);
}

/// Remove a registered plug-in.
pub fn unregister(name: &str) {
    registry().lock().unwrap().remove(name);
}

/// True if a plug-in of that name is registered.
pub fn is_registered(name: &str) -> bool {
    registry().lock().unwrap().contains_key(name)
}

/// A new instance of a registered plug-in.
pub fn instantiate(r: &PluginRef) -> Result<Box<dyn MaterialPlugin>> {
    let f = registry().lock().unwrap().get(&r.name).cloned().ok_or_else(|| anyhow!("material plug-in '{}' is not registered (lightsim_core::plugin::register, or lightsim.plugins in Python)", r.name))?;
    f(&r.params)
}

/// Cells of one E component inside one plug-in medium.
#[derive(Clone, Debug)]
pub struct PluginList {
    pub comp: crate::grid::Component,
    pub plugin: PluginRef,
    pub medium: String,
    pub idx: Vec<u64>,
    /// 1 / (eps (1 + f)) per cell.
    pub inv_eps: Vec<Real>,
}

/// Plug-in media of one slab.
#[derive(Clone, Debug)]
pub struct PluginPlan {
    pub lists: Vec<PluginList>,
}

/// A running plug-in list (engine side): the instance and its buffers.
pub struct PluginRun {
    pub plugin: Box<dyn MaterialPlugin>,
    pub state: Vec<Real>,
    pub e: Vec<Real>,
    pub e_prev: Vec<Real>,
}

impl PluginRun {
    pub fn new(l: &PluginList) -> Result<PluginRun> {
        let plugin = instantiate(&l.plugin)?;
        let n = l.idx.len();
        let state = vec![0 as Real; n * plugin.state_len()];
        Ok(PluginRun { plugin, state, e: vec![0 as Real; n], e_prev: vec![0 as Real; n] })
    }
    pub fn reset(&mut self) {
        self.state.fill(0 as Real);
    }
}

/// Plug-ins that reproduce built-in models (validation, examples): one
/// Lorentz (or Drude) pole and the instantaneous chi3 (Kerr) medium.
pub mod examples {
    use super::*;
    use crate::materials::Pole;

    /// One dispersive pole as a plug-in: state (P, P_prev, dP) per cell;
    /// `before` advances P with the ADE coefficients of [`Pole`] for the
    /// step of the run, `after` subtracts inv_eps dP (the arithmetic of the
    /// built-in update).
    pub struct PolePlugin {
        pub pole: Pole,
        c: Option<[Real; 3]>,
    }

    impl PolePlugin {
        pub fn new(pole: Pole) -> PolePlugin {
            PolePlugin { pole, c: None }
        }
    }

    impl MaterialPlugin for PolePlugin {
        fn state_len(&self) -> usize {
            3
        }
        fn before(&mut self, call: &PluginCall, e: &[Real], state: &mut [Real]) {
            let c = *self.c.get_or_insert_with(|| self.pole.ade_coefficients(call.dt).map(|v| v as Real));
            for (t, &ev) in e.iter().enumerate() {
                let s = &mut state[3 * t..3 * t + 3];
                let p0 = s[0];
                let pn = c[0] * p0 + c[1] * s[1] + c[2] * ev;
                s[2] = pn - p0;
                s[1] = p0;
                s[0] = pn;
            }
        }
        fn after(&mut self, _call: &PluginCall, e: &mut [Real], _e_prev: &[Real], inv_eps: &[Real], state: &mut [Real]) {
            for (t, ev) in e.iter_mut().enumerate() {
                *ev -= inv_eps[t] * state[3 * t + 2];
            }
        }
    }

    /// Instantaneous Kerr medium P / eps0 = chi3 E^3, solved implicitly for
    /// E^{n+1} as the built-in update (`nl_solve`, four Newton steps).
    pub struct KerrPlugin {
        pub chi3: Real,
    }

    impl MaterialPlugin for KerrPlugin {
        fn after(&mut self, _call: &PluginCall, e: &mut [Real], e_prev: &[Real], inv_eps: &[Real], _state: &mut [Real]) {
            for (t, ev) in e.iter_mut().enumerate() {
                *ev = crate::engine::cpu::nl_solve(e_prev[t], *ev, 0.0, self.chi3 * inv_eps[t]);
            }
        }
    }

    /// A plug-in that amplifies the field every step (gain without bound):
    /// for tests of the divergence check.
    pub struct RunawayPlugin {
        pub gain: Real,
    }

    impl MaterialPlugin for RunawayPlugin {
        fn after(&mut self, _call: &PluginCall, e: &mut [Real], _e_prev: &[Real], _inv_eps: &[Real], _state: &mut [Real]) {
            for ev in e.iter_mut() {
                *ev *= self.gain;
            }
        }
    }

    /// Register the example plug-ins under the names "example_lorentz"
    /// (params `delta_eps`, `omega0`, `gamma`), "example_kerr" (`chi3`) and
    /// "example_runaway" (`gain`, default 1.01).
    pub fn register_examples() {
        register(
            "example_lorentz",
            Arc::new(|p: &serde_json::Value| {
                let g = |k: &str| p.get(k).and_then(|v| v.as_f64()).ok_or_else(|| anyhow!("plug-in 'example_lorentz' needs the parameter '{k}'"));
                let pole = Pole::Lorentz { delta_eps: g("delta_eps")?, omega0: g("omega0")?, gamma: g("gamma")? };
                Ok(Box::new(PolePlugin::new(pole)) as Box<dyn MaterialPlugin>)
            }),
        );
        register(
            "example_kerr",
            Arc::new(|p: &serde_json::Value| {
                let chi3 = p.get("chi3").and_then(|v| v.as_f64()).ok_or_else(|| anyhow!("plug-in 'example_kerr' needs 'chi3'"))?;
                Ok(Box::new(KerrPlugin { chi3: chi3 as Real }) as Box<dyn MaterialPlugin>)
            }),
        );
        register(
            "example_runaway",
            Arc::new(|p: &serde_json::Value| {
                let gain = p.get("gain").and_then(|v| v.as_f64()).unwrap_or(1.01);
                Ok(Box::new(RunawayPlugin { gain: gain as Real }) as Box<dyn MaterialPlugin>)
            }),
        );
    }
}

/// The plug-in lists of one slab (`None` without plug-in media): one list
/// per plug-in medium and E component, with the cells the medium fills.
#[allow(clippy::too_many_arguments)]
pub fn compile_plugins(
    media: &[crate::materials::Medium],
    med_local: &[Vec<u16>; 3],
    cb_local: &[Vec<Real>; 3],
    x0: usize,
    nxl: usize,
    n: [usize; 3],
    dt: f64,
    complex_fields: bool,
) -> Result<Option<PluginPlan>> {
    let pm: Vec<usize> = (0..media.len()).filter(|&m| media[m].plugin.is_some()).collect();
    if pm.is_empty() {
        return Ok(None);
    }
    let _ = x0;
    if complex_fields {
        anyhow::bail!("material plug-ins cannot be combined with Bloch boundaries (complex fields)");
    }
    for &m in &pm {
        let md = &media[m];
        let r = md.plugin.as_ref().unwrap();
        if md.is_dispersive() || md.is_nonlinear() || md.raman.is_some() || md.gain.is_some() || md.is_tensor() || md.pec || md.is_magnetic() {
            anyhow::bail!("medium '{}': a plug-in medium is a plain (eps, sigma) background plus the plug-in; it cannot also have poles, chi2/chi3, gain, off-diagonal terms, magnetic properties or be a perfect conductor", md.name);
        }
        if !is_registered(&r.name) {
            anyhow::bail!("medium '{}': material plug-in '{}' is not registered (lightsim_core::plugin::register, or lightsim.plugins.register in Python)", md.name, r.name);
        }
    }
    let (ny, nz) = (n[1], n[2]);
    let plane = ny * nz;
    let mut lists = vec![];
    for &m in &pm {
        let md = &media[m];
        for a in 0..3 {
            let mut l = PluginList { comp: crate::grid::Component::e(a), plugin: md.plugin.clone().unwrap(), medium: md.name.clone(), idx: vec![], inv_eps: vec![] };
            for i in 0..nxl {
                for j in 0..ny {
                    for k in 0..nz {
                        let o = (i * ny + j) * nz + k;
                        if med_local[a][o] as usize != m || cb_local[a][o] == 0.0 {
                            continue;
                        }
                        // the medium's own eps, as the dispersive update (staircased)
                        let f = md.sigma[a].max(0.0) * dt / (2.0 * crate::constants::EPS0 * md.eps[a]);
                        l.idx.push(((i + 1) * plane + j * nz + k) as u64);
                        l.inv_eps.push((1.0 / (md.eps[a] * (1.0 + f))) as Real);
                    }
                }
            }
            if !l.idx.is_empty() {
                lists.push(l);
            }
        }
    }
    Ok(if lists.is_empty() { None } else { Some(PluginPlan { lists }) })
}
