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
        /// What as many profiles as frequencies (more than one) mean: one
        /// mode with one profile per frequency (`true`) or separate modes
        /// (`false`). Not given: read from the profiles' own frequencies
        /// ([`profiles_per_frequency`]).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        per_frequency: Option<bool>,
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

/// Whether the profiles of a mode monitor are one mode with one profile per
/// frequency (`true`) or separate modes, each used at every frequency.
///
/// Only as many profiles as frequencies, and more than one, can be one mode
/// per frequency. They are when `explicit` says so, or when it says nothing
/// and each profile was solved at the monitor frequency of its position; they
/// are separate modes when they were all solved at one frequency. Anything
/// else (the same number, frequencies that are neither) has two readings and
/// is refused until `explicit` chooses.
pub fn profiles_per_frequency(name: &str, modes: &[ModeProfile], freqs: &[f64], explicit: Option<bool>) -> anyhow::Result<bool> {
    let nf = freqs.len();
    if modes.len() != nf || nf < 2 {
        if explicit == Some(true) {
            anyhow::bail!("mode monitor '{name}': one profile per frequency needs as many profiles as frequencies (and more than one), but it has {} profiles and {nf} frequencies", modes.len());
        }
        return Ok(false);
    }
    if let Some(v) = explicit {
        return Ok(v);
    }
    let same = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs();
    if modes.iter().zip(freqs).all(|(m, f)| same(m.freq, *f)) {
        return Ok(true);
    }
    if modes.iter().all(|m| same(m.freq, modes[0].freq)) {
        return Ok(false);
    }
    anyhow::bail!(
        "mode monitor '{name}' has {nf} profiles and {nf} frequencies, and the profiles' own frequencies are neither the monitor's (one mode, one profile per frequency) nor all the same (separate modes): set per_frequency to true or false"
    )
}

#[cfg(test)]
mod per_frequency_tests {
    use super::profiles_per_frequency;
    use crate::sources::ModeProfile;

    fn at(f: f64) -> ModeProfile {
        ModeProfile { neff: 2.0, neff_imag: 0.0, dims: [1, 1], e_re: [vec![1.0], vec![0.0]], e_im: [vec![], vec![]], h_re: [vec![0.0], vec![1.0]], h_im: [vec![], vec![]], freq: f, h_offset: None, others: vec![] }
    }

    #[test]
    fn as_many_profiles_as_frequencies_are_read_by_their_own_frequencies() {
        let freqs = [1.0, 2.0, 3.0];
        // three modes solved at one frequency, read at three frequencies: separate modes
        assert!(!profiles_per_frequency("m", &[at(2.0), at(2.0), at(2.0)], &freqs, None).unwrap());
        // one mode solved at each frequency
        assert!(profiles_per_frequency("m", &[at(1.0), at(2.0), at(3.0)], &freqs, None).unwrap());
        // neither: refused until said
        let mixed = [at(3.0), at(2.0), at(1.0)];
        assert!(profiles_per_frequency("m", &mixed, &freqs, None).unwrap_err().to_string().contains("per_frequency"));
        assert!(profiles_per_frequency("m", &mixed, &freqs, Some(true)).unwrap());
        assert!(!profiles_per_frequency("m", &mixed, &freqs, Some(false)).unwrap());
        assert!(!profiles_per_frequency("m", &[at(1.0), at(2.0), at(3.0)], &freqs, Some(false)).unwrap());
        // other counts are separate modes; one profile per frequency cannot be asked of them
        assert!(!profiles_per_frequency("m", &[at(2.0), at(2.0)], &freqs, None).unwrap());
        assert!(!profiles_per_frequency("m", &[at(2.0)], &[2.0], None).unwrap());
        assert!(profiles_per_frequency("m", &[at(2.0), at(2.0)], &freqs, Some(true)).is_err());
    }
}
