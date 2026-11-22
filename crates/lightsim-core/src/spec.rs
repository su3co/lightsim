//! User-facing simulation specification (serializable; the Python API and the
//! CLI both produce this JSON).

use crate::Real;
use crate::constants::C0;
use crate::geometry::Structure;
use crate::grid::{Grid, MeshSpec, graded_axis_about};
use crate::materials::{Medium, library};
use crate::monitors::Monitor;
use crate::sources::Source;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// CPML grading parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PmlParams {
    /// Polynomial grading order for sigma and kappa.
    #[serde(default = "d_order")]
    pub order: f64,
    /// sigma_max as a multiple of the optimal 0.8 (m+1) / (eta0 dx).
    #[serde(default = "d_sigma")]
    pub sigma_scale: f64,
    #[serde(default = "d_kappa")]
    pub kappa_max: f64,
    /// CFS alpha_max in units of 2*pi*eps0*f_ref... given directly in S/m-like
    /// normalised units of eps0/s (alpha * dt / eps0 dimensionless).
    #[serde(default)]
    pub alpha_max: f64,
}

fn d_order() -> f64 {
    3.0
}
fn d_sigma() -> f64 {
    1.0
}
fn d_kappa() -> f64 {
    2.0
}

impl Default for PmlParams {
    fn default() -> Self {
        PmlParams { order: 3.0, sigma_scale: 1.0, kappa_max: 2.0, alpha_max: 0.0 }
    }
}

/// Boundary condition on one face.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Boundary {
    /// Convolutional PML of `layers` cells added *outside* the simulation region.
    Pml {
        #[serde(default = "d_layers")]
        layers: usize,
        #[serde(default)]
        params: PmlParams,
        /// Place the layers on the outermost cells *inside* the region (as
        /// FDTDX does) instead of adding them outside.
        #[serde(default)]
        inside: bool,
    },
    Periodic,
    /// Bloch-periodic with wavevector component `k` (rad/m) along the axis.
    Bloch { k: f64 },
    Pec,
    Pmc,
}

fn d_layers() -> usize {
    12
}

impl Boundary {
    pub fn pml() -> Boundary {
        Boundary::Pml { layers: 12, params: PmlParams::default(), inside: false }
    }
    pub fn pml_layers(&self) -> usize {
        match self {
            Boundary::Pml { layers, .. } => *layers,
            _ => 0,
        }
    }
    /// Number of cells added outside the region for this boundary.
    pub fn outside_layers(&self) -> usize {
        match self {
            Boundary::Pml { layers, inside: false, .. } => *layers,
            _ => 0,
        }
    }
    pub fn is_periodic(&self) -> bool {
        matches!(self, Boundary::Periodic | Boundary::Bloch { .. })
    }
}

/// Boundaries on the six faces, `[low, high]` per axis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Boundaries {
    pub x: [Boundary; 2],
    pub y: [Boundary; 2],
    pub z: [Boundary; 2],
}

impl Boundaries {
    pub fn all_pml() -> Boundaries {
        Boundaries {
            x: [Boundary::pml(), Boundary::pml()],
            y: [Boundary::pml(), Boundary::pml()],
            z: [Boundary::pml(), Boundary::pml()],
        }
    }
    pub fn all(b: Boundary) -> Boundaries {
        Boundaries { x: [b.clone(), b.clone()], y: [b.clone(), b.clone()], z: [b.clone(), b] }
    }
    pub fn axis(&self, a: usize) -> &[Boundary; 2] {
        match a {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
    pub fn axis_mut(&mut self, a: usize) -> &mut [Boundary; 2] {
        match a {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

/// Axis-aligned box.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Region {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

/// 2-D grids with fewer cells than this run on the CPU engine when the
/// device is left to `"auto"` (see [`Simulation::auto_prefers_cpu`]).
pub const AUTO_CPU_CELLS_2D: usize = 100_000;

/// Field set of a 2-D simulation (`dimensions = 2`), named with respect to
/// the invariant axis `w` and the two in-plane axes `u`, `v`:
///
/// * `te` - E in the plane, H along the invariant axis: (E_u, E_v, H_w);
/// * `tm` - E along the invariant axis, H in the plane: (E_w, H_u, H_v);
/// * `both` - the two sets, which are independent in media without
///   off-diagonal permittivity. Only the sets a source drives are updated.
///
/// For the invariant axis z this is (Ex, Ey, Hz) / (Ez, Hx, Hy), the TE / TM
/// naming of Lumerical's 2-D FDTD, of Meep and of Taflove's TE_z / TM_z.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Polarization {
    Te,
    Tm,
    #[default]
    Both,
}

impl Polarization {
    /// `[te, tm]`: the sets this choice admits.
    pub fn sets(self) -> [bool; 2] {
        match self {
            Polarization::Te => [true, false],
            Polarization::Tm => [false, true],
            Polarization::Both => [true, true],
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Polarization::Te => "te",
            Polarization::Tm => "tm",
            Polarization::Both => "both",
        }
    }
    pub fn parse(s: &str) -> Option<Polarization> {
        match s.to_ascii_lowercase().as_str() {
            "te" => Some(Polarization::Te),
            "tm" => Some(Polarization::Tm),
            "both" => Some(Polarization::Both),
            _ => None,
        }
    }
}

/// Axis index from a name or a number ("x" / "y" / "z" or 0 / 1 / 2).
pub fn parse_axis(s: &str) -> Option<usize> {
    match s.to_ascii_lowercase().as_str() {
        "x" | "0" => Some(0),
        "y" | "1" => Some(1),
        "z" | "2" => Some(2),
        _ => None,
    }
}

/// The two in-plane axes `(u, v)` of a 2-D simulation invariant along `w`
/// (v is the faster storage axis) and the handedness of `(u, v, w)`: +1 if
/// it is a cyclic permutation of (x, y, z), -1 otherwise.
pub fn plane_of(w: usize) -> (usize, usize, Real) {
    match w {
        0 => (1, 2, 1.0),
        1 => (0, 2, -1.0),
        _ => (0, 1, 1.0),
    }
}

/// True if component `c` belongs to the TE set of a 2-D simulation
/// invariant along `w` (E in the plane, H along `w`); the others form the
/// TM set.
pub fn is_te(c: crate::grid::Component, w: usize) -> bool {
    if c.is_e() { c.axis() != w } else { c.axis() == w }
}

pub(crate) fn de_axis<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<usize, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum A {
        N(usize),
        S(String),
    }
    match A::deserialize(d)? {
        A::N(n) if n < 3 => Ok(n),
        A::N(n) => Err(serde::de::Error::custom(format!("axis {n} out of range (0, 1, 2 or \"x\", \"y\", \"z\")"))),
        A::S(s) => parse_axis(&s).ok_or_else(|| serde::de::Error::custom(format!("unknown axis '{s}' (use \"x\", \"y\", \"z\" or 0, 1, 2)"))),
    }
}

/// Complete simulation specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Simulation {
    #[serde(default)]
    pub name: String,
    /// Simulation region (PML layers are added outside of it).
    pub region: Region,
    pub mesh: MeshSpec,
    #[serde(default)]
    pub media: Vec<Medium>,
    #[serde(default = "d_background")]
    pub background: String,
    #[serde(default)]
    pub structures: Vec<Structure>,
    #[serde(default)]
    pub sources: Vec<Source>,
    #[serde(default)]
    pub monitors: Vec<Monitor>,
    pub boundaries: Boundaries,
    /// Total simulated time (s).
    pub run_time: f64,
    /// Stop early once the field energy decays below `shutoff` times its
    /// maximum (after all sources have finished). `None` disables.
    #[serde(default)]
    pub shutoff: Option<f64>,
    /// Steps between shutoff checks.
    #[serde(default = "d_check")]
    pub shutoff_interval: usize,
    /// Fraction of the 3-D Courant limit.
    #[serde(default = "d_courant")]
    pub courant: f64,
    /// Sub-pixel smoothing of dielectric interfaces.
    #[serde(default = "d_true")]
    pub subpixel: bool,
    /// Frequency-domain monitors accumulate their Fourier sums every
    /// `dft_stride` steps. `None` (the default) chooses the stride per
    /// monitor from the bandwidth of the sources and the monitor frequencies
    /// (see [`crate::plan::auto_dft_stride`]); `Some(1)` samples every step. A
    /// `dft_field` monitor with its own `stride` keeps it.
    #[serde(default)]
    pub dft_stride: Option<usize>,
    /// 3 (default) or 2. A 2-D simulation has fields that do not depend on
    /// the coordinate along `invariant_axis`: the grid has a single cell
    /// there (the region's extent along that axis is the thickness of the
    /// modelled slab, which only scales integrated quantities), the
    /// boundaries along it are periodic whatever `boundaries` says, the
    /// structures are cut by the plane through the middle of that cell, and
    /// the time step follows the 2-D Courant limit.
    #[serde(default = "d_dims", skip_serializing_if = "is_three")]
    pub dimensions: u8,
    /// Invariant axis of a 2-D simulation: 0 / 1 / 2 or "x" / "y" / "z".
    #[serde(default = "d_inv_axis", deserialize_with = "de_axis", skip_serializing_if = "is_two_usize")]
    pub invariant_axis: usize,
    /// Field set of a 2-D simulation (see [`Polarization`]).
    #[serde(default, skip_serializing_if = "is_both")]
    pub polarization: Polarization,
    /// Run a one-cell (2-D) grid through the general 3-D update kernels
    /// instead of the specialised 2-D ones (cross-checks; same results to
    /// rounding, slower).
    #[serde(default, skip_serializing_if = "is_false")]
    pub force_3d_kernels: bool,
    /// Broadband oblique incidence at a fixed angle: the `angle` of the
    /// (single) plane-wave source holds at every frequency of its pulse
    /// instead of only at its centre frequency. The run solves for the
    /// transformed fields `E(r, t + s u / c)` (`s = n sin(angle)`, `u` the
    /// coordinate along the tilt axis), which are periodic along the tilt
    /// axis for all frequencies at once; see [`crate::engine::fixed_angle`].
    #[serde(default, skip_serializing_if = "is_false")]
    pub fixed_angle: bool,
    /// Zero-thickness conducting sheets (see [`crate::sheet`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sheets: Vec<crate::sheet::Sheet>,
}

fn d_background() -> String {
    "vacuum".into()
}
fn d_courant() -> f64 {
    0.99
}
fn d_true() -> bool {
    true
}
fn d_check() -> usize {
    200
}
fn d_dims() -> u8 {
    3
}
fn d_inv_axis() -> usize {
    2
}
fn is_three(v: &u8) -> bool {
    *v == 3
}
fn is_two_usize(v: &usize) -> bool {
    *v == 2
}
fn is_both(v: &Polarization) -> bool {
    *v == Polarization::Both
}
fn is_false(v: &bool) -> bool {
    !*v
}

impl Simulation {
    /// Empty simulation over `[min, max]` with the given mesh and boundaries.
    pub fn new(min: [f64; 3], max: [f64; 3], mesh: MeshSpec, boundaries: Boundaries, run_time: f64) -> Simulation {
        Simulation {
            name: String::new(),
            region: Region { min, max },
            mesh,
            media: vec![],
            background: d_background(),
            structures: vec![],
            sources: vec![],
            monitors: vec![],
            boundaries,
            run_time,
            shutoff: None,
            shutoff_interval: d_check(),
            courant: d_courant(),
            subpixel: true,
            dft_stride: None,
            dimensions: 3,
            invariant_axis: 2,
            polarization: Polarization::Both,
            force_3d_kernels: false,
            fixed_angle: false,
            sheets: vec![],
        }
    }

    /// Make this a 2-D simulation invariant along `axis` (see
    /// [`Simulation::dimensions`]).
    pub fn two_d(mut self, axis: usize, polarization: Polarization) -> Simulation {
        self.dimensions = 2;
        self.invariant_axis = axis;
        self.polarization = polarization;
        self
    }

    /// The invariant axis of a 2-D simulation (`None` for a 3-D one).
    pub fn invariant(&self) -> Option<usize> {
        (self.dimensions == 2).then_some(self.invariant_axis)
    }

    /// The slab of a 2-D simulation along its invariant axis:
    /// (axis, low face, high face).
    pub fn slab(&self) -> Option<(usize, f64, f64)> {
        self.invariant().map(|a| (a, self.region.min[a], self.region.max[a]))
    }

    /// The simulation the engine works with. A 3-D simulation is returned as
    /// it is. For a 2-D one the boundaries along the invariant axis become
    /// periodic and every source and monitor is made to span the slab (their
    /// extent and position along the invariant axis are ignored).
    /// Idempotent.
    pub fn canonical(&self) -> std::borrow::Cow<'_, Simulation> {
        let Some((a, lo, hi)) = self.slab() else { return std::borrow::Cow::Borrowed(self) };
        let mid = 0.5 * (lo + hi);
        let mut s = self.clone();
        *s.boundaries.axis_mut(a) = [Boundary::Periodic, Boundary::Periodic];
        let span = |mn: &mut [f64; 3], mx: &mut [f64; 3]| {
            mn[a] = lo;
            mx[a] = hi;
        };
        for src in s.sources.iter_mut() {
            match src {
                Source::PointDipole { position, .. } => position[a] = mid,
                Source::PlaneWave { min, max, .. } => {
                    if let (Some(mn), Some(mx)) = (min.as_mut(), max.as_mut()) {
                        span(mn, mx);
                    }
                }
                Source::GaussianBeam { center, .. } => center[a] = mid,
                Source::Mode { min, max, .. } | Source::Tfsf { min, max, .. } => span(min, max),
            }
        }
        for mon in s.monitors.iter_mut() {
            match mon {
                Monitor::Probe { position, .. } => position[a] = mid,
                Monitor::FieldTime { min, max, .. }
                | Monitor::DftField { min, max, .. }
                | Monitor::Flux { min, max, .. }
                | Monitor::FluxBox { min, max, .. }
                | Monitor::Mode { min, max, .. }
                | Monitor::FarField { min, max, .. }
                | Monitor::Energy { min, max, .. } => span(min, max),
            }
        }
        std::borrow::Cow::Owned(s)
    }

    pub fn from_json(s: &str) -> Result<Simulation> {
        Ok(serde_json::from_str(s)?)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    /// Media table: user media first, then library media referenced by name.
    pub fn media_table(&self) -> Result<Vec<Medium>> {
        let mut t: Vec<Medium> = self.media.clone();
        let mut need: Vec<String> = vec![self.background.clone()];
        need.extend(self.structures.iter().map(|s| s.medium.clone()));
        for name in need {
            if t.iter().any(|m| m.name == name) {
                continue;
            }
            match library::by_name(&name) {
                Some(mut m) => {
                    m.name = name.clone();
                    t.push(m)
                }
                None => bail!("unknown medium '{name}' (define it in `media`)"),
            }
        }
        Ok(t)
    }

    pub fn validate(&self) -> Result<()> {
        for a in 0..3 {
            let b = self.boundaries.axis(a);
            // (the boundaries along the invariant axis of a 2-D simulation are not used)
            if self.dimensions == 2 && self.invariant_axis == a {
                if self.region.max[a] <= self.region.min[a] {
                    bail!("region has non-positive extent along axis {a}");
                }
                continue;
            }
            if b[0].is_periodic() != b[1].is_periodic() {
                bail!("axis {a}: periodic/Bloch boundaries must be set on both sides");
            }
            if self.region.max[a] <= self.region.min[a] {
                bail!("region has non-positive extent along axis {a}");
            }
        }
        if self.run_time <= 0.0 {
            bail!("run_time must be positive");
        }
        // Results are keyed by name: two monitors (of any kind) or two
        // sources with one name would overwrite each other.
        let mut seen = std::collections::BTreeSet::new();
        for m in &self.monitors {
            if !seen.insert(m.name()) {
                bail!("duplicate monitor name '{}': every monitor needs its own name, whatever its type", m.name());
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for (i, s) in self.sources.iter().enumerate() {
            let key = if s.name().is_empty() { format!("source{i}") } else { s.name().to_string() };
            if !seen.insert(key.clone()) {
                bail!("duplicate source name '{key}' (an unnamed source is called source<index>)");
            }
        }
        if self.dimensions != 2 && self.dimensions != 3 {
            bail!("dimensions must be 2 or 3, got {}", self.dimensions);
        }
        if self.invariant_axis > 2 {
            bail!("invariant_axis must be 0, 1 or 2");
        }
        if self.dimensions == 3 && self.polarization != Polarization::Both {
            bail!("polarization = \"{}\" needs a 2-D simulation (dimensions = 2)", self.polarization.name());
        }
        for sh in &self.sheets {
            sh.check(self.invariant())?;
        }
        if let Some(a) = self.invariant() {
            let ax = ["x", "y", "z"][a];
            for src in &self.sources {
                let normal = match src {
                    Source::PlaneWave { axis, .. } | Source::GaussianBeam { axis, .. } | Source::Mode { axis, .. } | Source::Tfsf { axis, .. } => Some(*axis),
                    Source::PointDipole { .. } => None,
                };
                if normal == Some(a) {
                    bail!("source '{}' propagates along {ax}, the invariant axis of this 2-D simulation", src.name());
                }
            }
            for mon in &self.monitors {
                match mon {
                    Monitor::Flux { axis, name, .. } | Monitor::Mode { axis, name, .. } if *axis == a => {
                        bail!("monitor '{name}' is normal to {ax}, the invariant axis of this 2-D simulation")
                    }
                    Monitor::FarField { name, .. } => bail!("monitor '{name}': the far-field projection is three-dimensional and is not available in a 2-D simulation"),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Build the computational grid, including PML layers outside the region.
    pub fn build_grid(&self) -> Result<Grid> {
        self.validate()?;
        let media = self.media_table()?;
        let mut axes: Vec<Vec<f64>> = Vec::new();
        for a in 0..3 {
            let lo = self.region.min[a];
            let hi = self.region.max[a];
            if self.invariant() == Some(a) {
                // one cell, periodic: the slab of a 2-D simulation
                axes.push(vec![lo, hi]);
                continue;
            }
            let mut nodes: Vec<f64> = match &self.mesh {
                MeshSpec::Uniform { dx } => {
                    let n = ((hi - lo) / dx[a]).round().max(1.0) as usize;
                    let d = (hi - lo) / n as f64;
                    (0..=n).map(|i| lo + i as f64 * d).collect()
                }
                MeshSpec::Explicit { x, y, z } => {
                    let v = match a {
                        0 => x.clone(),
                        1 => y.clone(),
                        _ => z.clone(),
                    };
                    if v.len() < 2 {
                        bail!("explicit mesh needs at least 2 nodes per axis");
                    }
                    v
                }
                MeshSpec::Auto { wavelength, steps_per_wavelength, grading, min_step, max_step, overrides, symmetry } => {
                    let bg = media.iter().find(|m| m.name == self.background).unwrap();
                    let nbg = (bg.eps[a].max(bg.eps[(a + 1) % 3]).max(bg.eps[(a + 2) % 3]) * bg.mu[a]).sqrt().max(1.0);
                    let mut default_step = wavelength / (nbg * steps_per_wavelength);
                    if let Some(m) = max_step {
                        default_step = default_step.min(*m);
                    }
                    let mut targets: Vec<(f64, f64, f64)> = Vec::new();
                    for s in &self.structures {
                        if let Some((w, wl, wh)) = self.slab() {
                            // a structure the slice plane misses is not in the 2-D model
                            let (bl, bh) = s.shape.bbox();
                            let mid = 0.5 * (wl + wh);
                            if mid < bl[w] || mid > bh[w] {
                                continue;
                            }
                        }
                        let m = media.iter().find(|m| m.name == s.medium).unwrap();
                        let n = if m.pec || m.is_dispersive() {
                            // metals: resolve skin depth-ish with a finer step
                            4.0
                        } else {
                            (m.eps.iter().cloned().fold(1.0, f64::max) * m.mu.iter().cloned().fold(1.0, f64::max))
                                .sqrt()
                        };
                        let mut st = wavelength / (n * steps_per_wavelength);
                        if let Some(mn) = min_step {
                            st = st.max(*mn);
                        }
                        let (bl, bh) = s.shape.bbox();
                        if bh[a] > lo && bl[a] < hi {
                            targets.push((bl[a].max(lo), bh[a].min(hi), st));
                        }
                    }
                    for o in overrides {
                        targets.push((o.min[a].max(lo), o.max[a].min(hi), o.dx[a]));
                    }
                    if let Some(c) = symmetry[a] {
                        if !(c > lo && c < hi) {
                            bail!("mesh symmetry plane {c:e} along axis {a} is outside the region [{lo:e}, {hi:e}]");
                        }
                    }
                    graded_axis_about(lo, hi, symmetry[a], default_step, &targets, *grading)
                }
            };
            // zero-thickness sheets: a node on every sheet plane (generated meshes)
            if !matches!(self.mesh, MeshSpec::Explicit { .. }) {
                crate::sheet::align_nodes(&mut nodes, &self.sheets, a);
            }
            // PML layers outside, with the boundary cell size.
            let b = self.boundaries.axis(a);
            let n_lo = b[0].outside_layers();
            let n_hi = b[1].outside_layers();
            if n_lo > 0 {
                let d = nodes[1] - nodes[0];
                let mut pre: Vec<f64> = (1..=n_lo).rev().map(|i| nodes[0] - i as f64 * d).collect();
                pre.extend(nodes);
                nodes = pre;
            }
            if n_hi > 0 {
                let m = nodes.len();
                let d = nodes[m - 1] - nodes[m - 2];
                let last = nodes[m - 1];
                nodes.extend((1..=n_hi).map(|i| last + i as f64 * d));
            }
            axes.push(nodes);
        }
        let z = axes.pop().unwrap();
        let y = axes.pop().unwrap();
        let x = axes.pop().unwrap();
        Ok(Grid { x, y, z })
    }

    /// Time step on the given grid: `courant` times the Courant limit of the
    /// mesh, reduced further if a dispersive medium in use needs it (the
    /// explicit pole update is only stable below a limit that depends on the
    /// pole strengths and on `eps`, see [`crate::materials::poles_stable`]).
    ///
    /// A 2-D simulation uses the 2-D limit (the spacing along the invariant
    /// axis does not enter: nothing varies along it), and the pole limit is
    /// taken against that 2-D step.
    pub fn time_step(&self, grid: &Grid) -> Result<f64> {
        let d = grid.min_cell_size();
        let inv = self.invariant();
        let s: f64 = d.iter().enumerate().filter(|(a, _)| Some(*a) != inv).map(|(_, v)| 1.0 / (v * v)).sum();
        let dt_courant = 1.0 / (C0 * s.sqrt());
        let dt0 = self.courant / (C0 * s.sqrt());
        let media = self.media_table()?;
        let mut dt = dt0;
        for m in &media {
            let used = m.name == self.background || self.structures.iter().any(|st| st.medium == m.name);
            if !used || m.pec || (m.poles.is_empty() && m.oriented.is_empty() && m.gyro.is_empty() && m.gain.is_none()) {
                continue;
            }
            let mu = m.mu.iter().cloned().fold(f64::INFINITY, f64::min);
            for a in 0..3 {
                let mut poles: Vec<crate::materials::Pole> = if m.pole_axes[a] { m.poles.clone() } else { vec![] };
                poles.extend(m.oriented.iter().map(|o| o.pole.clone()));
                // (gyrotropic poles: the limit of the pole without its bias)
                poles.extend(m.gyro.iter().map(|g| g.pole.clone()));
                // (a gain medium: its transition at the largest absorbing strength, all atoms in level 1)
                if let Some(g) = &m.gain {
                    let w = g.omega.max(1.0);
                    poles.push(crate::materials::Pole::Lorentz { delta_eps: g.xi() * g.density / (crate::constants::EPS0 * w * w), omega0: g.omega, gamma: g.gamma });
                }
                if poles.is_empty() {
                    continue;
                }
                let stable = |t: f64| {
                    let a_max = 4.0 * (t / dt_courant).powi(2) / (m.eps[a] * mu);
                    crate::materials::poles_stable(m.eps[a], m.sigma[a].max(0.0), &poles, t, a_max)
                };
                if stable(dt) {
                    continue;
                }
                // largest stable step by bisection, with a 2 % margin
                let (mut lo, mut hi) = (0.0, dt);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if stable(mid) { lo = mid } else { hi = mid }
                }
                dt = 0.98 * lo;
                if dt < 0.1 * dt0 {
                    bail!(
                        "medium '{}': its dispersive poles are stable only below {:.3} of the time step of this mesh ({:.3e} s). \
                         Raise its eps (the permittivity at infinite frequency), lower the pole frequencies or strengths \
                         (for a fit, limit omega0_max), or refine the mesh.",
                        m.name,
                        lo / dt0,
                        dt0
                    );
                }
            }
        }
        let dt = crate::sheet::limit_time_step(self, grid, &media, dt, dt0, dt_courant)?;
        Ok(dt)
    }

    /// `device = "auto"`: true if this simulation should run on the CPU
    /// engine although a GPU is present. That is the case for a 2-D grid
    /// (one periodic cell along an axis) of fewer than
    /// [`AUTO_CPU_CELLS_2D`] cells: a GPU step costs a few kernel launches
    /// whatever the grid size, which is more than the CPU needs for the
    /// whole step of such a grid.
    pub fn auto_prefers_cpu(&self) -> bool {
        let Ok(grid) = self.build_grid() else { return false };
        let n = grid.n();
        let flat = self.invariant().is_some() || (0..3).any(|a| n[a] == 1 && self.boundaries.axis(a)[0].is_periodic());
        flat && grid.num_cells() < AUTO_CPU_CELLS_2D
    }

    /// Slice used to rasterise the structures of a 2-D simulation: (axis,
    /// plane coordinate, half thickness). The structures are averaged over a
    /// sheet much thinner than a cell around the mid-plane of the slab, so
    /// the 2-D model is their cross-section there, whatever their extent
    /// along the invariant axis.
    pub fn raster_slice(&self, grid: &Grid) -> Option<(usize, f64, f64)> {
        let (a, lo, hi) = self.slab()?;
        let d = grid.min_cell_size();
        let din = (0..3).filter(|&b| b != a).map(|b| d[b]).fold(f64::INFINITY, f64::min);
        Some((a, 0.5 * (lo + hi), (1e-3 * din).min(0.25 * (hi - lo))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_names_are_rejected() {
        use crate::monitors::Monitor;
        use crate::sources::{Source, TimeSignal};
        let mut sim = Simulation::new([0.0; 3], [1e-6; 3], MeshSpec::Uniform { dx: [0.1e-6; 3] }, Boundaries::all_pml(), 1e-15);
        let probe = |name: &str| Monitor::Probe { name: name.into(), position: [0.5e-6; 3], components: vec![crate::grid::Component::Ez] };
        sim.monitors = vec![probe("a"), probe("b")];
        assert!(sim.validate().is_ok());
        sim.monitors.push(Monitor::Flux { name: "a".into(), axis: 0, position: 0.5e-6, min: [0.0; 3], max: [1e-6; 3], freqs: vec![3e14], apodization: Default::default() });
        let err = sim.validate().unwrap_err().to_string();
        assert!(err.contains("duplicate monitor name 'a'"), "{err}");
        sim.monitors.pop();
        let dipole = |name: &str| Source::PointDipole { name: name.into(), position: [0.5e-6; 3], polarization: crate::grid::Component::Ez, signal: TimeSignal::gaussian(3e14, 3e13), amplitude: 1.0 };
        sim.sources = vec![dipole(""), dipole("")];
        assert!(sim.validate().is_ok(), "unnamed sources are source0, source1");
        sim.sources = vec![dipole("s"), dipole("s")];
        assert!(sim.validate().unwrap_err().to_string().contains("duplicate source name 's'"));
        sim.sources = vec![dipole("source1"), dipole("")];
        assert!(sim.validate().unwrap_err().to_string().contains("duplicate source name 'source1'"));
    }

    #[test]
    fn grid_with_pml() {
        let sim = Simulation {
            name: "t".into(),
            region: Region { min: [0.0; 3], max: [1e-6, 1e-6, 1e-6] },
            mesh: MeshSpec::Uniform { dx: [0.05e-6; 3] },
            media: vec![],
            background: "vacuum".into(),
            structures: vec![],
            sources: vec![],
            monitors: vec![],
            boundaries: Boundaries {
                x: [Boundary::pml(), Boundary::pml()],
                y: [Boundary::Periodic, Boundary::Periodic],
                z: [Boundary::Pec, Boundary::Pmc],
            },
            run_time: 1e-15,
            shutoff: None,
            shutoff_interval: 200,
            courant: 0.99,
            subpixel: true,
            dft_stride: None,
            dimensions: 3,
            invariant_axis: 2,
            polarization: Polarization::Both,
            force_3d_kernels: false,
            fixed_angle: false,
            sheets: vec![],
        };
        let g = sim.build_grid().unwrap();
        assert_eq!(g.n(), [20 + 24, 20, 20]);
        assert!((g.x[12] - 0.0).abs() < 1e-18);
    }
}
