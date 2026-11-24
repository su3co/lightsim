//! Source definitions: temporal signals and spatial source types.

use crate::constants::TAU;
use crate::grid::Component;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Temporal source signal `s(t)`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimeSignal {
    /// `A cos(w0 (t - t0) + phase) exp(-(t - t0)^2 / (2 tau^2))`, with
    /// `tau = 1 / (2 pi fwidth)` and `t0 = offset * tau` (offset default 5).
    GaussianPulse {
        freq0: f64,
        fwidth: f64,
        #[serde(default = "default_offset")]
        offset: f64,
        #[serde(default)]
        phase: f64,
        #[serde(default = "one")]
        amplitude: f64,
    },
    /// `A cos(w0 t + phase)` with a smooth Gaussian turn-on of width
    /// `1/(2 pi fwidth)` centred at `offset * tau`.
    ContinuousWave {
        freq0: f64,
        fwidth: f64,
        #[serde(default = "default_offset")]
        offset: f64,
        #[serde(default)]
        phase: f64,
        #[serde(default = "one")]
        amplitude: f64,
    },
    /// Ricker wavelet (second derivative of a Gaussian) with peak frequency freq0.
    Ricker {
        freq0: f64,
        #[serde(default = "default_offset_ricker")]
        offset: f64,
        #[serde(default = "one")]
        amplitude: f64,
    },
    /// Sampled signal, linearly interpolated (zero outside the samples).
    Custom { times: Vec<f64>, values: Vec<f64> },
}

fn default_offset() -> f64 {
    5.0
}
fn default_offset_ricker() -> f64 {
    1.5
}
fn one() -> f64 {
    1.0
}

impl TimeSignal {
    pub fn gaussian(freq0: f64, fwidth: f64) -> TimeSignal {
        TimeSignal::GaussianPulse { freq0, fwidth, offset: 5.0, phase: 0.0, amplitude: 1.0 }
    }
    pub fn cw(freq0: f64, fwidth: f64) -> TimeSignal {
        TimeSignal::ContinuousWave { freq0, fwidth, offset: 5.0, phase: 0.0, amplitude: 1.0 }
    }

    /// Value at time `t`.
    pub fn value(&self, t: f64) -> f64 {
        match self {
            TimeSignal::GaussianPulse { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let t0 = offset * tau;
                let x = t - t0;
                amplitude * (TAU * freq0 * x + phase).cos() * (-x * x / (2.0 * tau * tau)).exp()
            }
            TimeSignal::ContinuousWave { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let t0 = offset * tau;
                let ramp = if t < t0 { (-(t - t0).powi(2) / (2.0 * tau * tau)).exp() } else { 1.0 };
                amplitude * ramp * (TAU * freq0 * t + phase).cos()
            }
            TimeSignal::Ricker { freq0, offset, amplitude } => {
                let t0 = offset / freq0;
                let a = (std::f64::consts::PI * freq0 * (t - t0)).powi(2);
                amplitude * (1.0 - 2.0 * a) * (-a).exp()
            }
            TimeSignal::Custom { times, values } => {
                if times.is_empty() || t < times[0] || t > *times.last().unwrap() {
                    return 0.0;
                }
                let i = match times.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
                    Ok(i) => return values[i],
                    Err(i) => i,
                };
                let (t0, t1) = (times[i - 1], times[i]);
                let w = (t - t0) / (t1 - t0);
                values[i - 1] * (1.0 - w) + values[i] * w
            }
        }
    }

    /// Analytic (quadrature) value `s(t) + i H[s](t)` for narrow-band signals,
    /// used for sources whose profile carries a phase. For Gaussian/CW pulses
    /// this is `A exp(-i(w0 (t-t0) + phase)) * envelope`, consistent with an
    /// `exp(-i w t)` time dependence.
    pub fn analytic(&self, t: f64) -> Complex64 {
        match self {
            TimeSignal::GaussianPulse { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let t0 = offset * tau;
                let x = t - t0;
                let env = amplitude * (-x * x / (2.0 * tau * tau)).exp();
                Complex64::from_polar(env, -(TAU * freq0 * x + phase))
            }
            TimeSignal::ContinuousWave { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let t0 = offset * tau;
                let ramp = if t < t0 { (-(t - t0).powi(2) / (2.0 * tau * tau)).exp() } else { 1.0 };
                Complex64::from_polar(amplitude * ramp, -(TAU * freq0 * t + phase))
            }
            _ => Complex64::new(self.value(t), 0.0),
        }
    }

    /// Central frequency (Hz), if defined.
    pub fn freq0(&self) -> Option<f64> {
        match self {
            TimeSignal::GaussianPulse { freq0, .. }
            | TimeSignal::ContinuousWave { freq0, .. }
            | TimeSignal::Ricker { freq0, .. } => Some(*freq0),
            TimeSignal::Custom { .. } => None,
        }
    }

    /// Time after which the signal is negligible (for run-time heuristics).
    pub fn end_time(&self) -> f64 {
        match self {
            TimeSignal::GaussianPulse { fwidth, offset, .. } => 2.0 * offset / (TAU * fwidth),
            TimeSignal::ContinuousWave { .. } => f64::INFINITY,
            TimeSignal::Ricker { freq0, offset, .. } => 2.0 * offset / freq0,
            TimeSignal::Custom { times, .. } => times.last().copied().unwrap_or(0.0),
        }
    }

    /// Numerical spectrum `sum_n s(t_n) exp(i w t_n) dt` at the given
    /// frequencies over `nsteps` samples `t_n = (n + time_offset) dt`.
    pub fn spectrum(&self, freqs: &[f64], dt: f64, nsteps: usize, time_offset: f64) -> Vec<Complex64> {
        self.spectrum_of(freqs, dt, nsteps, time_offset, false)
    }

    /// Spectrum of the real signal `s(t)` (`analytic = false`) or of the
    /// complex signal [`TimeSignal::analytic`] (`analytic = true`), which is
    /// what a source with a complex field pattern injects into a run with
    /// complex fields (Bloch boundaries). At positive frequencies the
    /// analytic signal has twice the spectrum of the real one.
    pub fn spectrum_of(&self, freqs: &[f64], dt: f64, nsteps: usize, time_offset: f64, analytic: bool) -> Vec<Complex64> {
        let samples: Vec<Complex64> = (0..nsteps)
            .map(|n| {
                let t = (n as f64 + time_offset) * dt;
                if analytic { self.analytic(t) } else { Complex64::new(self.value(t), 0.0) }
            })
            .collect();
        freqs
            .iter()
            .map(|&f| {
                let w = TAU * f;
                let mut acc = Complex64::new(0.0, 0.0);
                for (n, s) in samples.iter().enumerate() {
                    if s.re != 0.0 || s.im != 0.0 {
                        acc += *s * Complex64::from_polar(dt, w * (n as f64 + time_offset) * dt);
                    }
                }
                acc
            })
            .collect()
    }
}

/// Propagation direction along an axis.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Direction {
    #[serde(rename = "+")]
    Plus,
    #[serde(rename = "-")]
    Minus,
}

impl Direction {
    pub fn sign(self) -> f64 {
        match self {
            Direction::Plus => 1.0,
            Direction::Minus => -1.0,
        }
    }
}

/// Complex transverse mode profile sampled on the Yee positions of a source or
/// monitor plane. Arrays are `[nb][nc]` row-major over the plane's two
/// transverse axes `(b, c)` in natural order (the axes other than the normal),
/// restricted to the plane's index box. `e[0], e[1]` are the tangential E
/// components along `(b, c)` at the plane (integer index along the normal);
/// `h[0], h[1]` the tangential H components half a cell *below* the plane.
/// Fields are normalised to unit power flow along `+normal`; H is in A/m.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModeProfile {
    pub neff: f64,
    #[serde(default)]
    pub neff_imag: f64,
    pub dims: [usize; 2],
    pub e_re: [Vec<f64>; 2],
    pub e_im: [Vec<f64>; 2],
    pub h_re: [Vec<f64>; 2],
    pub h_im: [Vec<f64>; 2],
    /// Frequency (Hz) at which the profile was computed.
    pub freq: f64,
    /// Distance (m) between the E plane and the H plane the `h` arrays refer
    /// to, i.e. the half cell the solver used for their phase
    /// `exp(-i beta h_offset)`. `None`: the half cell below the plane the
    /// profile is used on.
    #[serde(default)]
    pub h_offset: Option<f64>,
}

impl ModeProfile {
    /// Complex propagation constant (rad/m) at the profile frequency.
    pub fn beta(&self) -> Complex64 {
        Complex64::new(self.neff, self.neff_imag) * (TAU * self.freq / crate::constants::C0)
    }
}

/// Spatial source description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
// (boxing the large variant would change the public type)
#[allow(clippy::large_enum_variant)]
pub enum Source {
    /// Point dipole. `polarization` Ex/Ey/Ez is an electric current element
    /// with moment `amplitude * s(t)` (A·m); Hx/Hy/Hz a magnetic current element
    /// (V·m).
    PointDipole {
        #[serde(default)]
        name: String,
        position: [f64; 3],
        polarization: Component,
        signal: TimeSignal,
        #[serde(default = "one")]
        amplitude: f64,
    },
    /// Uniform plane wave injected through a one-sided TFSF surface normal to
    /// `axis` at coordinate `position`, propagating along `direction`, with
    /// peak electric field `amplitude * s(t)` (V/m). `polarization` is a
    /// component tangential to the plane: an E component sets E along its
    /// axis, an H component sets H along its axis (E along the other
    /// in-plane axis). `angle` tilts the k-vector (radians) toward the
    /// in-plane axis perpendicular to `polarization`'s axis: s-polarisation
    /// for an E component, p-polarisation for an H component (oblique
    /// incidence uses a narrow-band phase profile). In a 2-D simulation an
    /// in-plane E component at an angle is p-polarised as well. Transverse
    /// extent defaults to the whole plane (use periodic boundaries for an
    /// infinite plane wave).
    PlaneWave {
        #[serde(default)]
        name: String,
        axis: usize,
        position: f64,
        direction: Direction,
        polarization: Component,
        signal: TimeSignal,
        #[serde(default = "one")]
        amplitude: f64,
        #[serde(default)]
        angle: f64,
        #[serde(default)]
        min: Option<[f64; 3]>,
        #[serde(default)]
        max: Option<[f64; 3]>,
    },
    /// Gaussian beam of waist `waist` (1/e field radius at the focus),
    /// injected on the plane normal to `axis` at `position`.
    ///
    /// The beam axis passes through `center` on the source plane (the two
    /// in-plane coordinates of `center` are used) and points along
    /// `d = s cos(theta) a + sin(theta) (cos(phi) u + sin(phi) v)`, with `a`
    /// the plane normal, `s` the sign of `direction` and `(u, v)` the two
    /// in-plane axes in increasing order ((y, z), (x, z), (x, y) for axis 0,
    /// 1, 2). The focus lies `focus_distance` from `center` along `d`, and
    /// the waist plane is the plane normal to `d` through the focus. E is
    /// polarized along the projection of the `polarization` axis onto the
    /// plane normal to `d`; `amplitude` is the peak field at the focus (V/m).
    /// If the simulation is periodic along an in-plane axis with a period
    /// below twice the waist (a 2-D simulation), the beam is uniform along
    /// that axis (a sheet beam), and `d` must lie in the other plane.
    GaussianBeam {
        #[serde(default)]
        name: String,
        axis: usize,
        position: f64,
        direction: Direction,
        polarization: Component,
        center: [f64; 3],
        waist: f64,
        signal: TimeSignal,
        #[serde(default = "one")]
        amplitude: f64,
        /// Distance from the source plane to the focus along the propagation
        /// direction (0 = focused at the plane).
        #[serde(default)]
        focus_distance: f64,
        /// Polar angle (rad) between the beam axis and the plane normal.
        #[serde(default)]
        theta: f64,
        /// Azimuth (rad) of the tilt, from the first in-plane axis towards
        /// the second.
        #[serde(default)]
        phi: f64,
    },
    /// Eigenmode source on a plane box normal to `axis`.
    Mode {
        #[serde(default)]
        name: String,
        axis: usize,
        position: f64,
        direction: Direction,
        min: [f64; 3],
        max: [f64; 3],
        profile: ModeProfile,
        signal: TimeSignal,
        #[serde(default = "one")]
        amplitude: f64,
    },
    /// Total-field/scattered-field box with a normally incident plane wave
    /// propagating along `axis`/`direction`, polarized along `polarization`.
    Tfsf {
        #[serde(default)]
        name: String,
        min: [f64; 3],
        max: [f64; 3],
        axis: usize,
        direction: Direction,
        polarization: Component,
        signal: TimeSignal,
        #[serde(default = "one")]
        amplitude: f64,
    },
}

impl Source {
    pub fn name(&self) -> &str {
        match self {
            Source::PointDipole { name, .. }
            | Source::PlaneWave { name, .. }
            | Source::GaussianBeam { name, .. }
            | Source::Mode { name, .. }
            | Source::Tfsf { name, .. } => name,
        }
    }
    pub fn signal(&self) -> &TimeSignal {
        match self {
            Source::PointDipole { signal, .. }
            | Source::PlaneWave { signal, .. }
            | Source::GaussianBeam { signal, .. }
            | Source::Mode { signal, .. }
            | Source::Tfsf { signal, .. } => signal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_spectrum_peaks_at_f0() {
        let s = TimeSignal::gaussian(2e14, 5e13);
        let dt = 1e-17;
        let n = 20000;
        let f = [1e14, 2e14, 3e14];
        let sp = s.spectrum(&f, dt, n, 0.0);
        assert!(sp[1].norm() > sp[0].norm() && sp[1].norm() > sp[2].norm());
        // analytic peak value: A tau sqrt(2 pi) / 2
        let tau = 1.0 / (TAU * 5e13);
        let peak = tau * (TAU).sqrt() / 2.0;
        assert!((sp[1].norm() - peak).abs() / peak < 1e-3, "{} vs {}", sp[1].norm(), peak);
    }
}
