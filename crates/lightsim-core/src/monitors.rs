//! Monitor definitions.

use crate::grid::Component;
use crate::sources::ModeProfile;
use serde::{Deserialize, Serialize};

/// Spatial apodization of DFT accumulation in time (reduces truncation ringing).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Apodization {
    /// Start time (s): samples before `start` are ignored.
    #[serde(default)]
    pub start: Option<f64>,
    /// End time (s): samples after `end` are ignored.
    #[serde(default)]
    pub end: Option<f64>,
    /// Width (s) of the Gaussian taper applied at the start/end edges.
    #[serde(default)]
    pub width: Option<f64>,
}

impl Apodization {
    pub fn weight(&self, t: f64) -> f64 {
        let w = self.width.unwrap_or(0.0);
        if let Some(s) = self.start {
            if t < s {
                return 0.0;
            }
            if w > 0.0 && t < s + w {
                let x = (t - s - w) / (w / 3.0);
                return (-x * x / 2.0).exp();
            }
        }
        if let Some(e) = self.end {
            if t > e {
                return 0.0;
            }
            if w > 0.0 && t > e - w {
                let x = (t - e + w) / (w / 3.0);
                return (-x * x / 2.0).exp();
            }
        }
        1.0
    }
}

/// Monitor description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Monitor {
    /// Time-domain field samples inside a box every `interval` steps.
    FieldTime {
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        components: Vec<Component>,
        #[serde(default = "one_usize")]
        interval: usize,
        #[serde(default)]
        start: f64,
    },
    /// Time series of the fields at a point (every step).
    Probe { name: String, position: [f64; 3], components: Vec<Component> },
    /// Frequency-domain (running DFT) fields inside a box.
    DftField {
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        components: Vec<Component>,
        freqs: Vec<f64>,
        #[serde(default)]
        apodization: Apodization,
        /// Accumulate only every `stride` steps (must resolve the highest
        /// frequency in the fields). 0, the default, takes the simulation's
        /// `dft_stride`, which is automatic unless set.
        #[serde(default)]
        stride: usize,
    },
    /// Power flux through a plane normal to `axis` (Poynting vector of the DFT
    /// fields), positive along +axis.
    Flux {
        name: String,
        axis: usize,
        position: f64,
        min: [f64; 3],
        max: [f64; 3],
        freqs: Vec<f64>,
        #[serde(default)]
        apodization: Apodization,
    },
    /// Closed box flux (sum of outward fluxes through the 6 faces).
    FluxBox {
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        freqs: Vec<f64>,
        #[serde(default)]
        apodization: Apodization,
    },
    /// Mode-expansion monitor: overlap of the plane DFT fields with the given
    /// mode profiles (one per frequency or one for all).
    Mode {
        name: String,
        axis: usize,
        position: f64,
        min: [f64; 3],
        max: [f64; 3],
        freqs: Vec<f64>,
        modes: Vec<ModeProfile>,
        #[serde(default)]
        apodization: Apodization,
    },
    /// Far-field projection: DFT fields on a closed box are converted to
    /// equivalent surface currents (J = n x H, M = -n x E) and radiated into a
    /// homogeneous medium of index `n_medium`; returns the radiation intensity
    /// U(theta, phi) (W/sr) at each frequency. Angles in radians, theta from +z.
    FarField {
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        freqs: Vec<f64>,
        theta: Vec<f64>,
        phi: Vec<f64>,
        #[serde(default = "one_f64")]
        n_medium: f64,
        #[serde(default)]
        apodization: Apodization,
    },
    /// Electromagnetic energy in a box, every `interval` steps:
    /// `U = eps0/2 sum (eps |E|^2 + mu |eta0 H|^2) dV` in joules, with the
    /// cell permittivity the engine uses (eps_inf for dispersive media: the
    /// energy stored in the polarisation is not included). E is sampled at
    /// the end of the step, H half a step earlier.
    Energy {
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        #[serde(default = "one_usize")]
        interval: usize,
    },
}

fn one_usize() -> usize {
    1
}
fn one_f64() -> f64 {
    1.0
}

impl Monitor {
    pub fn name(&self) -> &str {
        match self {
            Monitor::FieldTime { name, .. }
            | Monitor::Probe { name, .. }
            | Monitor::DftField { name, .. }
            | Monitor::Flux { name, .. }
            | Monitor::FluxBox { name, .. }
            | Monitor::Mode { name, .. }
            | Monitor::FarField { name, .. }
            | Monitor::Energy { name, .. } => name,
        }
    }
}
