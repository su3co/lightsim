//! Material models.
//!
//! A [`Medium`] has a diagonal (possibly anisotropic) high-frequency
//! permittivity `eps`, permeability `mu`, electric conductivity `sigma`
//! (S/m) and an optional list of dispersive poles applied to every diagonal
//! component (or only to the axes in `pole_axes`). Time convention: fields
//! vary as `exp(-i w t)`, so passive media have `Im(eps) > 0`.
//!
//! ```text
//! eps(w) = eps_inf + i sigma/(eps0 w)
//!        + sum Lorentz  d_eps w0^2 / (w0^2 - w^2 - i g w)
//!        + sum Drude    -wp^2 / (w^2 + i g w)
//!        + sum Debye    d_eps / (1 - i w tau)
//! ```

use crate::constants::EPS0;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A single dispersive pole. Angular frequencies in rad/s.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Pole {
    Lorentz { delta_eps: f64, omega0: f64, gamma: f64 },
    Drude { omega_p: f64, gamma: f64 },
    Debye { delta_eps: f64, tau: f64 },
}

impl Pole {
    /// Susceptibility contribution chi(w).
    pub fn chi(&self, omega: f64) -> Complex64 {
        let i = Complex64::i();
        match *self {
            Pole::Lorentz { delta_eps, omega0, gamma } => {
                delta_eps * omega0 * omega0 / (omega0 * omega0 - omega * omega - i * gamma * omega)
            }
            Pole::Drude { omega_p, gamma } => -omega_p * omega_p / (omega * omega + i * gamma * omega),
            Pole::Debye { delta_eps, tau } => delta_eps / (1.0 - i * omega * tau),
        }
    }

    /// ADE coefficients `(c0, c1, c2)` for `P^{n+1} = c0 P^n + c1 P^{n-1} + c2 E^n`
    /// (P normalised by eps0), central differences at time n (as in Meep).
    pub fn ade_coefficients(&self, dt: f64) -> [f64; 3] {
        match *self {
            Pole::Lorentz { delta_eps, omega0, gamma } => {
                let a = 1.0 / (dt * dt) + gamma / (2.0 * dt);
                [
                    (2.0 / (dt * dt) - omega0 * omega0) / a,
                    -(1.0 / (dt * dt) - gamma / (2.0 * dt)) / a,
                    delta_eps * omega0 * omega0 / a,
                ]
            }
            Pole::Drude { omega_p, gamma } => {
                let a = 1.0 / (dt * dt) + gamma / (2.0 * dt);
                [2.0 / (dt * dt) / a, -(1.0 / (dt * dt) - gamma / (2.0 * dt)) / a, omega_p * omega_p / a]
            }
            Pole::Debye { delta_eps, tau } => {
                let r = tau / dt;
                [(r - 0.5) / (r + 0.5), 0.0, delta_eps / (r + 0.5)]
            }
        }
    }

    /// Discrete-time susceptibility seen by the ADE scheme at angular frequency w
    /// (used to compare the simulated response with the continuous model).
    pub fn chi_discrete(&self, omega: f64, dt: f64) -> Complex64 {
        // P^{n+1} - c0 P^n - c1 P^{n-1} = c2 E^n with P,E ~ z^n, z = exp(-i w dt)
        let [c0, c1, c2] = self.ade_coefficients(dt);
        let z = Complex64::new(0.0, -omega * dt).exp();
        // P (z - c0 - c1/z) = c2 E   with P^{n+1} = z P^n
        c2 / (z - c0 - c1 / z)
    }
}

/// Stability of the explicit pole update coupled to the Yee step.
///
/// For one field component of a medium with high-frequency permittivity
/// `eps_inf`, conductivity factor `f = sigma dt / (2 eps0 eps_inf)` and the
/// given poles, a grid mode whose discrete curl-curl eigenvalue is `lambda`
/// obeys the linear recurrence
///
/// ```text
/// E^{n+1} = (1 + ca - a/(1+f)) E^n - ca E^{n-1} - (P^{n+1} - 2 P^n + P^{n-1}) / (eps_inf (1+f)),
/// P_q^{n+1} = c0 P_q^n + c1 P_q^{n-1} + c2 E^n,     a = (c dt)^2 lambda / (eps_inf mu),
/// ```
///
/// with `ca = (1-f)/(1+f)`. `a` reaches `4 (c dt)^2 sum 1/d_i^2 / (eps_inf mu)`
/// for the fastest mode of the grid. The scheme is stable when no power of
/// the one-step matrix of this recurrence grows. Without poles this is the
/// Courant condition `a <= 4`; a Lorentz pole of strength `d_eps` at `w0`
/// tightens it to `a + 4 (d_eps/eps_inf) b / (4 - b) <= 4` with
/// `b = (w0 dt)^2`, a Drude pole to `a + (wp dt)^2 / eps_inf <= 4`: a strong
/// pole on top of a small `eps_inf` needs a step below the Courant limit.
///
/// Returns true if the recurrence is stable for `a = a_max` and for smaller
/// values of `a`.
pub fn poles_stable(eps_inf: f64, sigma: f64, poles: &[Pole], dt: f64, a_max: f64) -> bool {
    let n = 2 + 2 * poles.len();
    let f = sigma * dt / (2.0 * EPS0 * eps_inf);
    let ca = (1.0 - f) / (1.0 + f);
    let k = 1.0 / (eps_inf * (1.0 + f));
    let cs: Vec<[f64; 3]> = poles.iter().map(|p| p.ade_coefficients(dt)).collect();
    for frac in [1.0, 0.75, 0.5, 0.25] {
        let a = a_max * frac;
        // state (E^n, E^{n-1}, P_1^n, P_1^{n-1}, ...)
        let mut m = vec![0.0f64; n * n];
        for (q, c) in cs.iter().enumerate() {
            let r = 2 + 2 * q;
            m[r * n + r] = c[0];
            m[r * n + r + 1] = c[1];
            m[r * n] = c[2];
            m[(r + 1) * n + r] = 1.0;
        }
        m[0] = (1.0 + ca) - a / (1.0 + f);
        m[1] = -ca;
        for q in 0..cs.len() {
            let r = 2 + 2 * q;
            for j in 0..n {
                let v = m[r * n + j];
                m[j] -= k * v;
            }
            m[r] += 2.0 * k;
            m[r + 1] -= k;
        }
        m[n] = 1.0;
        // norms of M^(2^j): bounded for a stable recurrence, exponentially
        // growing otherwise (2^21 steps cover any run; a factor 1000 between
        // 2^14 and 2^21 steps separates this from the slow growth of a
        // marginal mode)
        let mut early = 0.0;
        let mut late = 0.0;
        let mut tmp = vec![0.0f64; n * n];
        for j in 0..21 {
            for r in 0..n {
                for c in 0..n {
                    let mut acc = 0.0;
                    for t in 0..n {
                        acc += m[r * n + t] * m[t * n + c];
                    }
                    tmp[r * n + c] = acc;
                }
            }
            std::mem::swap(&mut m, &mut tmp);
            if m.iter().any(|v| !v.is_finite()) {
                return false;
            }
            let norm = m.iter().fold(0.0f64, |mx, v| mx.max(v.abs()));
            if j == 13 {
                early = norm;
            }
            late = norm;
        }
        if late > 1e3 * early.max(1.0) {
            return false;
        }
    }
    true
}

/// Homogeneous medium description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Medium {
    #[serde(default)]
    pub name: String,
    /// High-frequency relative permittivity (diagonal).
    #[serde(default = "ones")]
    pub eps: [f64; 3],
    /// Relative permeability (diagonal).
    #[serde(default = "ones")]
    pub mu: [f64; 3],
    /// Electric conductivity (S/m), diagonal.
    #[serde(default)]
    pub sigma: [f64; 3],
    /// Magnetic conductivity (Ohm/m), diagonal: the magnetic current density
    /// sigma_m H in Faraday's law, `-dB/dt = curl E + sigma_m H`. A medium
    /// with `sigma_m / (mu0 mu) = sigma / (eps0 eps)` has the impedance of
    /// the lossless medium at every frequency (no reflection from it).
    #[serde(default)]
    pub sigma_m: [f64; 3],
    /// Dispersive poles.
    #[serde(default)]
    pub poles: Vec<Pole>,
    /// Axes the poles apply to (default all three).
    #[serde(default = "all_axes")]
    pub pole_axes: [bool; 3],
    /// Perfect electric conductor.
    #[serde(default)]
    pub pec: bool,
    /// Off-diagonal relative permittivity [eps_xy, eps_xz, eps_yz] of a
    /// symmetric tensor (diagonal in `eps`), e.g. a rotated crystal.
    #[serde(default)]
    pub eps_off: [f64; 3],
    /// Off-diagonal conductivity [sigma_xy, sigma_xz, sigma_yz] (S/m) of a
    /// symmetric conductivity tensor (diagonal in `sigma`).
    #[serde(default)]
    pub sigma_off: [f64; 3],
    /// Oriented dispersive poles: chi(w) u u^T along a unit vector u (a 1-D
    /// oscillator along u; rotated dispersive crystals, monoclinic media).
    #[serde(default)]
    pub oriented: Vec<OrientedPole>,
    /// Instantaneous second-order susceptibility (m/V), per axis:
    /// P_i/eps0 += chi2_i E_i^2 (Lumerical "chi2" convention).
    #[serde(default)]
    pub chi2: [f64; 3],
    /// Instantaneous third-order (Kerr) susceptibility (m^2/V^2), per axis:
    /// P_i/eps0 += chi3_i E_i^3. With `raman`, a fraction of it is delayed.
    #[serde(default)]
    pub chi3: [f64; 3],
    /// Delayed (Raman) part of the chi3 response: P_i/eps0 += chi3_i E_i
    /// [(1 - f) E_i^2 + f (g_R * E_i^2)(t)] instead of chi3_i E_i^3 (see
    /// [`Raman`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raman: Option<Raman>,
    /// Gyrotropic poles: dispersive poles in a static magnetic bias, which
    /// add a permittivity tensor with antisymmetric off-diagonal terms (see
    /// [`GyroPole`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gyro: Vec<GyroPole>,
    /// Uniaxial medium whose optic axis (director) varies from cell to cell,
    /// e.g. a liquid crystal (see [`Director`]). Replaces `eps` inside the
    /// medium; `eps` is then only read by the automatic mesher.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub director: Option<Director>,
}

/// A director field: the permittivity at r is
/// `eps_o I + (eps_e - eps_o) Q(r)`, where Q is the director tensor d d^T
/// sampled on a regular grid of `dims` points spanning `min..max` (faces
/// included; a dimension of 1 is uniform along that axis), interpolated
/// trilinearly at each field position and normalised to unit trace.
/// Interpolating d d^T rather than d makes the field independent of the
/// sign of each sample (d and -d are the same director). Outside the box
/// the nearest face value is used.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Director {
    pub eps_o: f64,
    pub eps_e: f64,
    pub min: [f64; 3],
    pub max: [f64; 3],
    pub dims: [usize; 3],
    /// Director samples, three numbers per sample, sample (i, j, k) at
    /// position (i * ny + j) * nz + k (not necessarily normalised).
    pub director: Vec<f64>,
}

impl Director {
    pub fn check(&self, medium: &str) -> anyhow::Result<()> {
        let n: usize = self.dims.iter().product();
        if self.dims.iter().any(|&d| d == 0) || self.director.len() != 3 * n {
            anyhow::bail!("medium '{medium}': the director field needs 3 * nx * ny * nz = {} numbers, got {}", 3 * n, self.director.len());
        }
        if !(self.eps_o > 0.0 && self.eps_e > 0.0) {
            anyhow::bail!("medium '{medium}': the director field needs eps_o > 0 and eps_e > 0");
        }
        for a in 0..3 {
            if self.dims[a] > 1 && !(self.max[a] > self.min[a]) {
                anyhow::bail!("medium '{medium}': the director box needs max > min along axis {a}");
            }
        }
        if self.director.chunks(3).any(|d| !(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > 0.0) || d.iter().any(|v| !v.is_finite())) {
            anyhow::bail!("medium '{medium}': every director sample must be a finite non-zero vector");
        }
        Ok(())
    }
    /// Unit-trace director tensor at r.
    pub fn q_at(&self, r: [f64; 3]) -> [[f64; 3]; 3] {
        // per axis: (lower index, weight of the upper one)
        let mut ix = [(0usize, 0.0f64); 3];
        for a in 0..3 {
            let n = self.dims[a];
            if n > 1 {
                let t = ((r[a] - self.min[a]) / (self.max[a] - self.min[a])).clamp(0.0, 1.0) * (n - 1) as f64;
                let i = (t.floor() as usize).min(n - 2);
                ix[a] = (i, t - i as f64);
            }
        }
        let mut q = [[0.0; 3]; 3];
        for c in 0..8 {
            let mut w = 1.0;
            let mut id = [0usize; 3];
            for a in 0..3 {
                let up = c >> a & 1 == 1;
                if self.dims[a] == 1 {
                    if up {
                        w = 0.0;
                    }
                    continue;
                }
                id[a] = ix[a].0 + up as usize;
                w *= if up { ix[a].1 } else { 1.0 - ix[a].1 };
            }
            if w == 0.0 {
                continue;
            }
            let o = 3 * ((id[0] * self.dims[1] + id[1]) * self.dims[2] + id[2]);
            let d = &self.director[o..o + 3];
            let nn = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            for i in 0..3 {
                for j in 0..3 {
                    q[i][j] += w * d[i] * d[j] / nn;
                }
            }
        }
        let tr = q[0][0] + q[1][1] + q[2][2];
        q.map(|r| r.map(|v| v / tr))
    }
    /// Relative permittivity tensor at r.
    pub fn eps_at(&self, r: [f64; 3]) -> [[f64; 3]; 3] {
        let q = self.q_at(r);
        let mut e = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                e[i][j] = (self.eps_e - self.eps_o) * q[i][j] + if i == j { self.eps_o } else { 0.0 };
            }
        }
        e
    }
}

/// A Lorentz or Drude pole in a static magnetic bias (magneto-optic,
/// gyrotropic medium). The polarization (normalised by eps0) obeys
///
/// ```text
/// P'' + gamma P' + omega0^2 P - b x P' = s E,
/// ```
///
/// with `s = delta_eps omega0^2` (Lorentz) or `omega_p^2` (Drude,
/// `omega0 = 0`) and `b` the gyration vector `bias` in rad/s (for free
/// electrons the cyclotron frequency e B / m along B). With fields
/// `exp(-i w t)` and D = omega0^2 - w^2 - i gamma w, the susceptibility is the
/// tensor `chi = s (D I + i w [b x])^-1`; for a bias along z,
/// chi_xx = chi_yy = s D / (D^2 - w^2 b^2), chi_zz = s / D and
/// chi_xy = -chi_yx = i s w b / (D^2 - w^2 b^2) = i g. Circular polarizations
/// (1, +-i) propagating along the bias see chi_+- = s / (D +- w b).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GyroPole {
    pub pole: Pole,
    pub bias: [f64; 3],
}

/// Complex 3x3 matrix.
pub type C3 = [[Complex64; 3]; 3];

fn inv3c(m: C3) -> C3 {
    let c = |a: usize, b: usize, c: usize, d: usize| m[a][b] * m[c][d] - m[a][d] * m[c][b];
    let adj = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    let det = m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0];
    let mut k = [[Complex64::new(0.0, 0.0); 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            k[a][b] = adj[a][b] / det;
        }
    }
    k
}

impl GyroPole {
    /// (s, omega0^2, gamma) of the pole.
    pub fn params(&self) -> anyhow::Result<(f64, f64, f64)> {
        match self.pole {
            Pole::Lorentz { delta_eps, omega0, gamma } => Ok((delta_eps * omega0 * omega0, omega0 * omega0, gamma)),
            Pole::Drude { omega_p, gamma } => Ok((omega_p * omega_p, 0.0, gamma)),
            Pole::Debye { .. } => anyhow::bail!("a gyrotropic pole must be a Lorentz or a Drude pole"),
        }
    }
    /// The skew matrix W of `v -> b x v`.
    fn cross(&self) -> [[f64; 3]; 3] {
        let b = self.bias;
        [[0.0, -b[2], b[1]], [b[2], 0.0, -b[0]], [-b[1], b[0], 0.0]]
    }
    /// Continuous susceptibility tensor at angular frequency w.
    pub fn chi(&self, omega: f64) -> C3 {
        let (s, w0sq, g) = self.params().expect("gyrotropic pole");
        let d = Complex64::new(w0sq - omega * omega, -g * omega);
        let w = self.cross();
        let mut m = [[Complex64::new(0.0, 0.0); 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                m[a][b] = Complex64::new(0.0, omega * w[a][b]) + if a == b { d } else { Complex64::new(0.0, 0.0) };
            }
        }
        let k = inv3c(m);
        k.map(|r| r.map(|v| v * s))
    }
    /// `[A, B, C]` of the update P^{n+1} = A P^n + B P^{n-1} + C E^n
    /// (central differences at step n, the b x P' term included), each a
    /// 3x3 matrix: with a = 1/dt^2 + gamma/(2 dt), beta = 1/(2 dt) and
    /// M = a I - beta W, A = M^-1 (2/dt^2 - omega0^2),
    /// B = M^-1 (-(1/dt^2 - gamma/(2 dt)) I - beta W), C = M^-1 s.
    pub fn matrices(&self, dt: f64) -> [[[f64; 3]; 3]; 3] {
        let (s, w0sq, g) = self.params().expect("gyrotropic pole");
        let a = 1.0 / (dt * dt) + g / (2.0 * dt);
        let beta = 1.0 / (2.0 * dt);
        let w = self.cross();
        let b = self.bias;
        let bb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
        // M^-1 = (a^2 I + a beta W + beta^2 b b^T) / (a (a^2 + beta^2 |b|^2))
        let den = a * (a * a + beta * beta * bb);
        let mut mi = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                mi[i][j] = (if i == j { a * a } else { 0.0 } + a * beta * w[i][j] + beta * beta * b[i] * b[j]) / den;
            }
        }
        let mut out = [[[0.0; 3]; 3]; 3];
        let rb = -(1.0 / (dt * dt) - g / (2.0 * dt));
        for i in 0..3 {
            for j in 0..3 {
                out[0][i][j] = mi[i][j] * (2.0 / (dt * dt) - w0sq);
                out[1][i][j] = (0..3).map(|q| mi[i][q] * (if q == j { rb } else { 0.0 } - beta * w[q][j])).sum();
                out[2][i][j] = mi[i][j] * s;
            }
        }
        out
    }
    /// Susceptibility tensor of the discrete update at angular frequency w:
    /// (z I - A - B / z)^-1 C with z = exp(-i w dt).
    pub fn chi_discrete(&self, omega: f64, dt: f64) -> C3 {
        let [am, bm, cm] = self.matrices(dt);
        let z = Complex64::new(0.0, -omega * dt).exp();
        let mut m = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = -am[i][j] - bm[i][j] / z + if i == j { z } else { Complex64::new(0.0, 0.0) };
            }
        }
        let k = inv3c(m);
        let mut out = [[Complex64::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|q| k[i][q] * cm[q][j]).sum();
            }
        }
        out
    }
}

/// Delayed third-order response: a fraction `fraction` (f) of chi3 acts
/// through a damped-oscillator Raman response g_R,
///
/// ```text
/// P_i/eps0 = chi3_i E_i [(1 - f) E_i^2 + f Q_i],   Q_i = (g_R * E_i^2)(t),
/// Q'' + 2 delta Q' + omega^2 Q = omega^2 E_i^2,
/// g_R(t) = (omega^2 / nu) exp(-delta t) sin(nu t),  nu = sqrt(omega^2 - delta^2),
/// ```
///
/// normalised so that its integral is 1 (a static field sees the full chi3).
/// `omega` is the phonon (resonance) angular frequency and `delta` the damping
/// rate, both in rad/s; g_R(w) = omega^2 / (omega^2 - w^2 - 2 i delta w).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Raman {
    pub fraction: f64,
    pub omega: f64,
    pub delta: f64,
}

impl Raman {
    /// The response as a Lorentz pole of unit strength (same ODE).
    fn pole(&self) -> Pole {
        Pole::Lorentz { delta_eps: 1.0, omega0: self.omega, gamma: 2.0 * self.delta }
    }
    /// g_R(w) of the continuous model.
    pub fn response(&self, omega: f64) -> Complex64 {
        self.pole().chi(omega)
    }
    /// g_R(w) seen by the discrete update.
    pub fn response_discrete(&self, omega: f64, dt: f64) -> Complex64 {
        self.pole().chi_discrete(omega, dt)
    }
    /// `(c0, c1, c2)` of `Q^{n+1} = c0 Q^n + c1 Q^{n-1} + c2 (E^n)^2`.
    pub fn ade_coefficients(&self, dt: f64) -> [f64; 3] {
        self.pole().ade_coefficients(dt)
    }
    /// g_R(t) (underdamped, `delta < omega`).
    pub fn kernel(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        let nu = (self.omega * self.omega - self.delta * self.delta).sqrt();
        self.omega * self.omega / nu * (-self.delta * t).exp() * (nu * t).sin()
    }
    pub fn check(&self, medium: &str) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.fraction) || !(self.omega > 0.0) || !(self.delta >= 0.0) || self.delta >= self.omega {
            anyhow::bail!(
                "medium '{medium}': the Raman response needs 0 <= fraction <= 1, omega > 0 and 0 <= delta < omega (got {}, {}, {})",
                self.fraction,
                self.omega,
                self.delta
            );
        }
        Ok(())
    }
}

fn ones() -> [f64; 3] {
    [1.0; 3]
}
fn all_axes() -> [bool; 3] {
    [true; 3]
}

/// A dispersive pole acting along the unit vector `u` (normalized on use).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrientedPole {
    pub pole: Pole,
    pub u: [f64; 3],
}

impl OrientedPole {
    pub fn unit(&self) -> [f64; 3] {
        let n = (self.u[0] * self.u[0] + self.u[1] * self.u[1] + self.u[2] * self.u[2]).sqrt();
        [self.u[0] / n, self.u[1] / n, self.u[2] / n]
    }
}

/// Inverse of a 3x3 matrix.
pub fn inv3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |a: usize, b: usize, c: usize, d: usize| m[a][b] * m[c][d] - m[a][d] * m[c][b];
    let adj = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    let det = m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0];
    let mut k = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            k[a][b] = adj[a][b] / det;
        }
    }
    k
}

/// [`Medium::tensor_update`] for a given permittivity tensor `e` and
/// conductivity tensor `sg`.
pub fn tensor_update_of(e: [[f64; 3]; 3], sg: [[f64; 3]; 3], cdt: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let eta0 = crate::constants::ETA0;
    let mut a = [[0.0; 3]; 3];
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = e[i][j] / cdt + 0.5 * eta0 * sg[i][j];
            b[i][j] = e[i][j] / cdt - 0.5 * eta0 * sg[i][j];
        }
    }
    let k = inv3(a);
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|q| k[i][q] * b[q][j]).sum();
        }
    }
    (m, k)
}

impl Default for Medium {
    fn default() -> Self {
        Medium::vacuum()
    }
}

impl Medium {
    pub fn vacuum() -> Medium {
        Medium {
            name: "vacuum".into(),
            eps: [1.0; 3],
            mu: [1.0; 3],
            sigma: [0.0; 3],
            sigma_m: [0.0; 3],
            poles: vec![],
            pole_axes: [true; 3],
            pec: false,
            eps_off: [0.0; 3],
            sigma_off: [0.0; 3],
            oriented: vec![],
            chi2: [0.0; 3],
            chi3: [0.0; 3],
            raman: None,
            gyro: vec![],
            director: None,
        }
    }
    pub fn dielectric(eps: f64) -> Medium {
        Medium { name: format!("eps={eps}"), eps: [eps; 3], ..Medium::vacuum() }
    }
    pub fn from_index(n: f64) -> Medium {
        Medium { name: format!("n={n}"), eps: [n * n; 3], ..Medium::vacuum() }
    }
    pub fn anisotropic(eps: [f64; 3]) -> Medium {
        Medium { name: format!("eps={eps:?}"), eps, ..Medium::vacuum() }
    }
    pub fn conductor(eps: f64, sigma: f64) -> Medium {
        Medium { name: format!("eps={eps},sigma={sigma}"), eps: [eps; 3], sigma: [sigma; 3], ..Medium::vacuum() }
    }
    pub fn pec() -> Medium {
        Medium { name: "PEC".into(), pec: true, ..Medium::vacuum() }
    }
    pub fn with_name(mut self, name: &str) -> Medium {
        self.name = name.into();
        self
    }
    pub fn with_poles(mut self, poles: Vec<Pole>) -> Medium {
        self.poles = poles;
        self
    }

    pub fn is_dispersive(&self) -> bool {
        !self.poles.is_empty()
    }
    /// Has off-diagonal permittivity terms (symmetric tensors, oriented or
    /// gyrotropic poles): handled by the tensor update.
    pub fn is_tensor(&self) -> bool {
        self.is_offdiag() || !self.gyro.is_empty() || self.director.is_some()
    }
    /// Has symmetric off-diagonal terms or oriented poles (the
    /// `tensor_apply` lists).
    pub fn is_offdiag(&self) -> bool {
        self.eps_off.iter().chain(self.sigma_off.iter()).any(|&c| c != 0.0) || !self.oriented.is_empty()
    }
    /// Full relative permittivity tensor.
    pub fn eps_tensor(&self) -> [[f64; 3]; 3] {
        let (e, o) = (self.eps, self.eps_off);
        [[e[0], o[0], o[1]], [o[0], e[1], o[2]], [o[1], o[2], e[2]]]
    }
    /// Inverse permittivity tensor kappa = eps^-1 (symmetric).
    pub fn inv_eps_tensor(&self) -> [[f64; 3]; 3] {
        let m = self.eps_tensor();
        let c = |a: usize, b: usize, c: usize, d: usize| m[a][b] * m[c][d] - m[a][d] * m[c][b];
        let adj = [
            [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
            [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
            [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
        ];
        let det = m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0];
        let mut k = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                k[a][b] = adj[a][b] / det;
            }
        }
        k
    }
    /// Rotate the diagonal tensor `eps` by the rotation matrix `r` (columns =
    /// crystal axes in lab coordinates): eps_lab = R diag(eps) R^T.
    pub fn rotated(mut self, r: [[f64; 3]; 3]) -> Medium {
        let rot = |d: [f64; 3]| {
            let mut t = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    t[a][b] = (0..3).map(|q| r[a][q] * d[q] * r[b][q]).sum();
                }
            }
            ([t[0][0], t[1][1], t[2][2]], [t[0][1], t[0][2], t[1][2]])
        };
        let (e, eo) = rot(self.eps);
        let (sg, so) = rot(self.sigma);
        self.eps = e;
        self.eps_off = eo;
        self.sigma = sg;
        self.sigma_off = so;
        self
    }
    /// Full conductivity tensor (S/m).
    pub fn sigma_tensor(&self) -> [[f64; 3]; 3] {
        let (e, o) = (self.sigma, self.sigma_off);
        [[e[0], o[0], o[1]], [o[0], e[1], o[2]], [o[1], o[2], e[2]]]
    }
    /// Semi-implicit update matrices of a tensor medium in normalized units
    /// (E in V/m, H~ = eta0 H): E^{n+1} = M E^n + K curl H~ with
    /// A = eps/(c dt) + eta0 sigma / 2, B = eps/(c dt) - eta0 sigma / 2,
    /// M = A^-1 B, K = A^-1 (metres).
    pub fn tensor_update(&self, cdt: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        tensor_update_of(self.eps_tensor(), self.sigma_tensor(), cdt)
    }
    pub fn is_nonlinear(&self) -> bool {
        self.chi2.iter().chain(self.chi3.iter()).any(|&c| c != 0.0)
    }
    pub fn is_lossy(&self) -> bool {
        self.sigma.iter().chain(self.sigma_off.iter()).any(|&s| s != 0.0)
    }
    pub fn is_magnetic(&self) -> bool {
        self.mu.iter().any(|&m| (m - 1.0).abs() > 0.0) || self.is_magnetic_lossy()
    }
    /// Non-zero magnetic conductivity.
    pub fn is_magnetic_lossy(&self) -> bool {
        self.sigma_m.iter().any(|&s| s != 0.0)
    }

    /// Complex relative permittivity along `axis` at angular frequency w.
    pub fn eps_complex(&self, axis: usize, omega: f64) -> Complex64 {
        let mut e = Complex64::new(self.eps[axis], 0.0);
        if self.sigma[axis] != 0.0 && omega != 0.0 {
            e += Complex64::new(0.0, self.sigma[axis] / (EPS0 * omega));
        }
        if self.pole_axes[axis] {
            for p in &self.poles {
                e += p.chi(omega);
            }
        }
        // oriented poles: diagonal projection chi u_a^2 (exact for u along an axis)
        for op in &self.oriented {
            let u = op.unit();
            e += op.pole.chi(omega) * (u[axis] * u[axis]);
        }
        // gyrotropic poles: the diagonal element
        for gp in &self.gyro {
            if gp.params().is_ok() {
                e += gp.chi(omega)[axis][axis];
            }
        }
        e
    }

    /// Complex refractive index along `axis` (principal branch with Im >= 0).
    pub fn n_complex(&self, axis: usize, omega: f64) -> Complex64 {
        let e = self.eps_complex(axis, omega) * Complex64::new(self.mu[axis], 0.0);
        let n = e.sqrt();
        if n.im < 0.0 { -n } else { n }
    }
}

/// A small library of common photonic media (non-dispersive values are at
/// 1550 nm unless noted; metals use Drude-Lorentz fits).
pub mod library {
    use super::*;
    use crate::constants::TAU;

    pub fn air() -> Medium {
        Medium::vacuum().with_name("air")
    }
    /// Crystalline silicon at 1550 nm (n = 3.476).
    pub fn si() -> Medium {
        Medium::from_index(3.476).with_name("Si")
    }
    /// Thermal silica at 1550 nm (n = 1.444).
    pub fn sio2() -> Medium {
        Medium::from_index(1.444).with_name("SiO2")
    }
    /// LPCVD silicon nitride at 1550 nm (n = 1.996).
    pub fn si3n4() -> Medium {
        Medium::from_index(1.996).with_name("Si3N4")
    }
    /// Silicon nitride at 785 nm (n = 2.012).
    pub fn si3n4_785() -> Medium {
        Medium::from_index(2.012).with_name("Si3N4@785")
    }
    /// Water at visible/NIR (n = 1.33).
    pub fn water() -> Medium {
        Medium::from_index(1.33).with_name("H2O")
    }

    const EV: f64 = 1.519_267_447e15; // rad/s per eV

    /// Gold, Drude + 5 Lorentz terms (Rakic et al., Appl. Opt. 37, 5271 (1998)).
    pub fn gold() -> Medium {
        let wp = 9.03 * EV;
        let f0: f64 = 0.760;
        let g0 = 0.053 * EV;
        let lor = [
            (0.024, 0.415, 0.241),
            (0.010, 0.830, 0.345),
            (0.071, 2.969, 0.870),
            (0.601, 4.304, 2.494),
            (4.384, 13.32, 2.214),
        ];
        let mut poles = vec![Pole::Drude { omega_p: f0.sqrt() * wp, gamma: g0 }];
        for (f, w, g) in lor {
            let w0 = w * EV;
            poles.push(Pole::Lorentz { delta_eps: f * wp * wp / (w0 * w0), omega0: w0, gamma: g * EV });
        }
        Medium::vacuum().with_name("Au").with_poles(poles)
    }

    /// Silver, Drude + 5 Lorentz terms (Rakic et al. 1998).
    pub fn silver() -> Medium {
        let wp = 9.01 * EV;
        let f0: f64 = 0.845;
        let g0 = 0.048 * EV;
        let lor = [
            (0.065, 0.816, 3.886),
            (0.124, 4.481, 0.452),
            (0.011, 8.185, 0.065),
            (0.840, 9.083, 0.916),
            (5.646, 20.29, 2.419),
        ];
        let mut poles = vec![Pole::Drude { omega_p: f0.sqrt() * wp, gamma: g0 }];
        for (f, w, g) in lor {
            let w0 = w * EV;
            poles.push(Pole::Lorentz { delta_eps: f * wp * wp / (w0 * w0), omega0: w0, gamma: g * EV });
        }
        Medium::vacuum().with_name("Ag").with_poles(poles)
    }

    /// Simple single-pole Drude metal with plasma frequency `fp` (Hz) and
    /// collision frequency `fc` (Hz).
    pub fn drude(fp: f64, fc: f64) -> Medium {
        Medium::vacuum()
            .with_name("drude")
            .with_poles(vec![Pole::Drude { omega_p: TAU * fp, gamma: TAU * fc }])
    }

    /// Look up a medium by (case-insensitive) name.
    pub fn by_name(name: &str) -> Option<Medium> {
        match name.to_ascii_lowercase().as_str() {
            "air" | "vacuum" => Some(air()),
            "si" | "silicon" => Some(si()),
            "sio2" | "silica" | "oxide" => Some(sio2()),
            "si3n4" | "sin" | "nitride" => Some(si3n4()),
            "si3n4_785" | "sin_785" => Some(si3n4_785()),
            "water" | "h2o" => Some(water()),
            "au" | "gold" => Some(gold()),
            "ag" | "silver" => Some(silver()),
            "pec" => Some(Medium::pec()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TAU;

    #[test]
    fn drude_low_frequency_limit() {
        let m = library::drude(2e15, 1e13);
        let e = m.eps_complex(0, TAU * 1e14);
        assert!(e.re < 0.0 && e.im > 0.0);
    }

    #[test]
    fn ade_matches_continuous_for_small_dt() {
        let p = Pole::Lorentz { delta_eps: 2.0, omega0: 2e15, gamma: 1e14 };
        let dt = 1e-18;
        for w in [0.5e15, 1.5e15, 3e15] {
            let c = p.chi(w);
            let d = p.chi_discrete(w, dt);
            assert!((c - d).norm() / c.norm() < 1e-2, "w={w} cont={c} disc={d}");
        }
    }

    #[test]
    fn pole_stability_matches_the_lossless_criterion() {
        // one Lorentz pole: stable iff (4 - a)(4 - b) >= 4 b d_eps / eps_inf
        let (eps_inf, d_eps, w0) = (1.0, 10.7, 6.5e15);
        let dx = 22e-9;
        let dtc = dx / (crate::constants::C0 * 3f64.sqrt());
        let pole = [Pole::Lorentz { delta_eps: d_eps, omega0: w0, gamma: 0.0 }];
        let exact = |c: f64| {
            let (a, b) = (4.0 * c * c / eps_inf, (w0 * c * dtc).powi(2));
            b < 4.0 && (4.0 - a) * (4.0 - b) >= 4.0 * b * d_eps / eps_inf
        };
        for c in [0.5, 0.8, 0.9, 0.905, 0.915, 0.95, 0.99] {
            assert_eq!(poles_stable(eps_inf, 0.0, &pole, c * dtc, 4.0 * c * c / eps_inf), exact(c), "courant {c}");
        }
        // no poles: the Courant condition
        assert!(poles_stable(2.0, 0.0, &[], dtc, 3.99));
        assert!(!poles_stable(2.0, 0.0, &[], dtc, 4.01));
        // a Drude metal on a small eps_inf: a + (wp dt)^2 / eps_inf <= 4
        let wp = 1.2e16;
        let drude = [Pole::Drude { omega_p: wp, gamma: 0.0 }];
        for c in [0.9, 0.95, 0.97, 0.99] {
            let dt = c * dtc;
            assert_eq!(poles_stable(1.0, 0.0, &drude, dt, 4.0 * c * c), 4.0 * c * c + (wp * dt).powi(2) <= 4.0, "courant {c}");
        }
    }

    #[test]
    fn gyrotropic_pole_tensor() {
        let w = 2e15;
        let gp = GyroPole { pole: Pole::Lorentz { delta_eps: 1.0, omega0: 1.5 * w, gamma: 3e13 }, bias: [0.0, 0.0, 0.2 * w] };
        let chi = gp.chi(w);
        // antisymmetric off-diagonal pair, chi_zz of the plain pole
        assert!((chi[0][1] + chi[1][0]).norm() < 1e-12 * chi[0][1].norm());
        assert!((chi[0][0] - chi[1][1]).norm() < 1e-12 * chi[0][0].norm());
        assert!((chi[2][2] - gp.pole.chi(w)).norm() < 1e-12 * chi[2][2].norm());
        assert!(chi[0][2].norm() < 1e-12 && chi[2][1].norm() < 1e-12);
        // circular eigenvalues s / (D +- w b)
        let (s, w0sq, g) = gp.params().unwrap();
        let d = Complex64::new(w0sq - w * w, -g * w);
        let i = Complex64::i();
        assert!((chi[0][0] + i * chi[0][1] - s / (d + w * 0.2 * w)).norm() < 1e-10 * chi[0][0].norm());
        assert!((chi[0][0] - i * chi[0][1] - s / (d - w * 0.2 * w)).norm() < 1e-10 * chi[0][0].norm());
        // the discrete update tends to the continuous model; a bias along a
        // general direction rotates the tensor
        let dt = 1e-19;
        let cd = gp.chi_discrete(w, dt);
        for a in 0..3 {
            for b in 0..3 {
                assert!((cd[a][b] - chi[a][b]).norm() < 1e-3 * chi[0][0].norm(), "{a}{b}: {} vs {}", cd[a][b], chi[a][b]);
            }
        }
        let gx = GyroPole { bias: [0.2 * w, 0.0, 0.0], ..gp.clone() };
        let cx = gx.chi(w);
        assert!((cx[1][2] - chi[0][1]).norm() < 1e-12 * chi[0][1].norm() && (cx[0][0] - chi[2][2]).norm() < 1e-12 * chi[2][2].norm());
    }

    #[test]
    fn gold_is_metallic_at_1um() {
        let au = library::gold();
        let w = TAU * crate::constants::C0 / 1e-6;
        let e = au.eps_complex(0, w);
        assert!(e.re < -20.0 && e.re > -60.0, "Au eps at 1um = {e}");
    }
}
