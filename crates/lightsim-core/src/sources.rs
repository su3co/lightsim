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

    /// For a signal that is an envelope on a carrier, `a(t) exp(-i phi(t))`
    /// with `phi' = 2 pi freq0`: the carrier factor and the envelope with its
    /// first two time derivatives, `(exp(-i phi), [a, a', a''])`. `None` for
    /// signals without a carrier (a Ricker wavelet, a sampled signal).
    pub fn carrier_envelope(&self, t: f64) -> Option<(Complex64, [f64; 3])> {
        let gauss = |x: f64, tau: f64| {
            let g = (-x * x / (2.0 * tau * tau)).exp();
            [g, -x / (tau * tau) * g, (x * x / (tau * tau) - 1.0) / (tau * tau) * g]
        };
        match self {
            TimeSignal::GaussianPulse { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let x = t - offset * tau;
                Some((Complex64::from_polar(1.0, -(TAU * freq0 * x + phase)), gauss(x, tau).map(|v| amplitude * v)))
            }
            TimeSignal::ContinuousWave { freq0, fwidth, offset, phase, amplitude } => {
                let tau = 1.0 / (TAU * fwidth);
                let t0 = offset * tau;
                let env = if t < t0 { gauss(t - t0, tau) } else { [1.0, 0.0, 0.0] };
                Some((Complex64::from_polar(1.0, -(TAU * freq0 * t + phase)), env.map(|v| amplitude * v)))
            }
            _ => None,
        }
    }

    /// The analytic signal with its spectrum multiplied by the polynomial
    /// `c[0] + c[1] (f - freq0) + c[2] (f - freq0)^2` of the frequency `f`:
    /// with `exp(-i w t)`, a factor `f - freq0` is `(i / 2 pi) d/dt` on the
    /// envelope. `None` for signals without a carrier.
    pub fn analytic_times_polynomial(&self, t: f64, c: [f64; 3]) -> Option<Complex64> {
        let (carrier, a) = self.carrier_envelope(t)?;
        Some(carrier * Complex64::new(c[0] * a[0] - c[2] * a[2] / (TAU * TAU), c[1] * a[1] / TAU))
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
    /// The same mode solved at other frequencies, each with its own `freq`
    /// and `neff`, with the phase convention and the window of this one. A
    /// mode monitor then projects, at each of its frequencies, on the
    /// profile interpolated in frequency through this one and these
    /// ([`ModeProfile::at_frequency`]), and a mode source whose signal has a
    /// carrier launches, at each frequency of its pulse, the profile
    /// interpolated through up to three of them
    /// ([`ModeProfile::source_nodes`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub others: Vec<ModeProfile>,
}

impl ModeProfile {
    /// Complex propagation constant (rad/m) at the profile frequency.
    pub fn beta(&self) -> Complex64 {
        Complex64::new(self.neff, self.neff_imag) * (TAU * self.freq / crate::constants::C0)
    }

    /// The solves this profile carries: itself first, then those of `others`
    /// that have its window and a frequency of their own.
    pub fn band_nodes(&self) -> Vec<&ModeProfile> {
        let mut nodes: Vec<&ModeProfile> = vec![self];
        for o in &self.others {
            if o.dims == self.dims && nodes.iter().all(|n| (n.freq - o.freq).abs() > 1e-9 * self.freq.abs()) {
                nodes.push(o);
            }
        }
        nodes
    }

    /// The solves a source follows over its band: all of them up to three,
    /// which is the parabola a mode monitor projects on; of more, the
    /// profile itself and the solves at the lowest and the highest frequency
    /// (a polynomial of higher degree would be extrapolated over the tails
    /// of the pulse's spectrum).
    pub fn source_nodes(&self) -> Vec<&ModeProfile> {
        let nodes = self.band_nodes();
        if nodes.len() <= 3 {
            return nodes;
        }
        let rest = &nodes[1..];
        let lo = rest.iter().copied().min_by(|a, b| a.freq.total_cmp(&b.freq)).unwrap();
        let hi = rest.iter().copied().max_by(|a, b| a.freq.total_cmp(&b.freq)).unwrap();
        vec![self, lo, hi]
    }

    /// Coefficients `c` of the Lagrange weight of node `k` among the
    /// frequencies `freqs` (two or three), as a polynomial of `f - f0`:
    /// `w_k(f) = c[0] + c[1] (f - f0) + c[2] (f - f0)^2`.
    pub fn lagrange_weight(freqs: &[f64], k: usize, f0: f64) -> [f64; 3] {
        let u: Vec<f64> = freqs.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, f)| f - f0).collect();
        let uk = freqs[k] - f0;
        match u.as_slice() {
            [] => [1.0, 0.0, 0.0],
            [a] => [-a / (uk - a), 1.0 / (uk - a), 0.0],
            [a, b] => {
                let d = (uk - a) * (uk - b);
                [a * b / d, -(a + b) / d, 1.0 / d]
            }
            _ => panic!("a source follows at most three solves"),
        }
    }

    /// The profile at frequency `f` (Hz). Without `others` this is the
    /// profile itself, whatever `f`. With them it is the profile solved at
    /// `f` when there is one, and otherwise the polynomial through all of
    /// them in frequency (fields and effective index sample by sample: a
    /// parabola for three frequencies), which follows the mode over a band
    /// to the third order of the detuning.
    pub fn at_frequency(&self, f: f64) -> std::borrow::Cow<'_, ModeProfile> {
        use std::borrow::Cow;
        if self.others.is_empty() {
            return Cow::Borrowed(self);
        }
        let nodes = self.band_nodes();
        if let Some(n) = nodes.iter().find(|n| (n.freq - f).abs() <= 1e-12 * f.abs()) {
            return Cow::Borrowed(n);
        }
        let w: Vec<f64> = (0..nodes.len()).map(|k| (0..nodes.len()).filter(|&j| j != k).map(|j| (f - nodes[j].freq) / (nodes[k].freq - nodes[j].freq)).product()).collect();
        let mix = |get: &dyn Fn(&ModeProfile) -> &Vec<f64>| -> Vec<f64> {
            let n = nodes.iter().map(|m| get(m).len()).max().unwrap_or(0);
            let mut out = vec![0.0; n];
            for (m, wk) in nodes.iter().zip(&w) {
                for (o, x) in out.iter_mut().zip(get(m)) {
                    *o += wk * x;
                }
            }
            out
        };
        let scalar = |get: &dyn Fn(&ModeProfile) -> f64| -> f64 { nodes.iter().zip(&w).map(|(m, wk)| wk * get(m)).sum() };
        Cow::Owned(ModeProfile {
            neff: scalar(&|m| m.neff),
            neff_imag: scalar(&|m| m.neff_imag),
            dims: self.dims,
            e_re: [mix(&|m| &m.e_re[0]), mix(&|m| &m.e_re[1])],
            e_im: [mix(&|m| &m.e_im[0]), mix(&|m| &m.e_im[1])],
            h_re: [mix(&|m| &m.h_re[0]), mix(&|m| &m.h_re[1])],
            h_im: [mix(&|m| &m.h_im[0]), mix(&|m| &m.h_im[1])],
            freq: f,
            h_offset: self.h_offset,
            others: vec![],
        })
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

#[cfg(test)]
mod band_profile_tests {
    use super::ModeProfile;

    fn at(f: f64) -> ModeProfile {
        // fields and index that are parabolas of the frequency
        let v = |a: f64| vec![a + 0.5 * f + 0.25 * f * f, -a + f * f];
        ModeProfile { neff: 2.0 + 0.1 * f - 0.01 * f * f, neff_imag: 0.0, dims: [2, 1], e_re: [v(1.0), v(2.0)], e_im: [vec![], vec![]], h_re: [v(3.0), v(4.0)], h_im: [vec![], vec![]], freq: f, h_offset: Some(0.1), others: vec![] }
    }

    #[test]
    fn a_profile_alone_is_itself_at_every_frequency() {
        let p = at(2.0);
        assert_eq!(&*p.at_frequency(5.0), &p);
    }

    #[test]
    fn three_solves_give_the_parabola_and_the_solves_themselves() {
        let mut p = at(2.0);
        p.others = vec![at(1.0), at(3.0)];
        // at a solved frequency: that solve, bit for bit
        assert_eq!(&*p.at_frequency(3.0), &at(3.0));
        assert_eq!(p.at_frequency(2.0).e_re, at(2.0).e_re);
        let q = p.at_frequency(2.4);
        let r = at(2.4);
        assert!((q.neff - r.neff).abs() < 1e-12 && q.freq == 2.4 && q.others.is_empty());
        for (a, b) in q.e_re.iter().chain(q.h_re.iter()).zip(r.e_re.iter().chain(r.h_re.iter())) {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-12, "{x} {y}");
            }
        }
    }

    #[test]
    fn a_source_follows_at_most_three_solves() {
        let mut p = at(2.0);
        p.others = vec![at(1.0), at(3.0)];
        assert_eq!(p.source_nodes().iter().map(|n| n.freq).collect::<Vec<_>>(), [2.0, 1.0, 3.0]);
        p.others = vec![at(1.5), at(1.0), at(3.0), at(2.5)];
        assert_eq!(p.source_nodes().iter().map(|n| n.freq).collect::<Vec<_>>(), [2.0, 1.0, 3.0]);
        for freqs in [vec![2.0, 1.0, 3.0], vec![2.0, 2.5]] {
            for f in [1.2, 2.0, 2.9] {
                let w: Vec<f64> = (0..freqs.len())
                    .map(|k| {
                        let c = ModeProfile::lagrange_weight(&freqs, k, 1.9);
                        c[0] + c[1] * (f - 1.9) + c[2] * (f - 1.9) * (f - 1.9)
                    })
                    .collect();
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                // the weights reproduce a parabola (a line for two solves) of the frequency
                let g = |x: f64| if freqs.len() == 3 { 0.3 - x + 0.7 * x * x } else { 0.3 - x };
                assert!((w.iter().zip(&freqs).map(|(w, x)| w * g(*x)).sum::<f64>() - g(f)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_polynomial_of_the_frequency_is_derivatives_of_the_envelope() {
        use super::TimeSignal;
        use num_complex::Complex64;
        let (f0, fw) = (1.9e14, 1.5e13);
        for sig in [TimeSignal::GaussianPulse { freq0: f0, fwidth: fw, offset: 5.0, phase: 0.3, amplitude: 1.2 }, TimeSignal::ContinuousWave { freq0: f0, fwidth: fw, offset: 5.0, phase: 0.3, amplitude: 1.2 }] {
            let tau = 1.0 / (std::f64::consts::TAU * fw);
            for t in [2.1 * tau, 4.4 * tau, 4.99 * tau] {
                let z = |c| sig.analytic_times_polynomial(t, c).unwrap();
                assert!((z([1.0, 0.0, 0.0]) - sig.analytic(t)).norm() < 1e-14);
                // (f - f0) s is (i / 2 pi) ds/dt - f0 s: central differences of the analytic signal
                let h = 1e-4 / f0;
                let d1 = (sig.analytic(t + h) - sig.analytic(t - h)) / (2.0 * h);
                let d2 = (sig.analytic(t + h) - 2.0 * sig.analytic(t) + sig.analytic(t - h)) / (h * h);
                let i2pi = Complex64::i() / std::f64::consts::TAU;
                let one = i2pi * d1 - f0 * sig.analytic(t);
                let two = i2pi * i2pi * d2 - 2.0 * f0 * i2pi * d1 + f0 * f0 * sig.analytic(t);
                let scale = sig.analytic(t).norm();
                assert!((z([0.0, 1.0, 0.0]) - one).norm() < 1e-5 * fw * scale, "{:?} {:?}", z([0.0, 1.0, 0.0]), one);
                assert!((z([0.0, 0.0, 1.0]) - two).norm() < 1e-2 * fw * fw * scale, "{:?} {:?}", z([0.0, 0.0, 1.0]), two);
            }
        }
        assert!(TimeSignal::Custom { times: vec![0.0, 1.0], values: vec![0.0, 1.0] }.carrier_envelope(0.5).is_none());
    }
}
