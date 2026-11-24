//! Absorbing layers: stabilised PML layers (the poles of a medium damped or
//! frozen inside them, see [`pml_pole_variant`]), adiabatic absorbers (a
//! matched pair of conductivities) and the CPML slabs.

use super::*;
use crate::materials::Pole;

/// Default strength of an adiabatic absorber: the amplitude attenuation, in
/// nepers, of a plane wave in the background medium that crosses the layer,
/// reflects at the outer wall and crosses it again (`sigma_scale` multiplies
/// it). A wave of group index n_g is attenuated n_g / n_background times as
/// much. Stronger absorbers reflect more at their graded conductivity,
/// weaker ones let more through to the wall and back: with layers about a
/// wavelength thick the two meet near this value.
pub const ABSORBER_ATTENUATION: f64 = 6.0;

/// The band of the sources, as angular frequencies. `None` without sources,
/// with a pulse that reaches zero frequency, or with a sampled signal, whose
/// band is not known here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceBand {
    /// Below this the sources carry nothing: three widths under the centre of
    /// a pulse, a quarter of the peak frequency of a Ricker wavelet.
    pub edge_lo: f64,
    /// The lowest frequency that matters for results: two widths under the
    /// centre, half the peak frequency.
    pub used_lo: f64,
    /// The mean of the sources' centre frequencies.
    pub centre: f64,
    /// The highest frequency that matters: two widths over the centre, twice
    /// the peak frequency.
    pub used_hi: f64,
    /// Above this the sources carry nothing: three widths over the centre,
    /// three times the peak frequency.
    pub edge_hi: f64,
}

pub fn source_band(sources: &[Source]) -> Option<SourceBand> {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [0.0f64; 2]);
    let mut centre = 0.0;
    for s in sources {
        let (f0, e, u) = match s.signal() {
            TimeSignal::GaussianPulse { freq0, fwidth, .. } | TimeSignal::ContinuousWave { freq0, fwidth, .. } => (*freq0, [freq0 - 3.0 * fwidth, freq0 + 3.0 * fwidth], [freq0 - 2.0 * fwidth, freq0 + 2.0 * fwidth]),
            TimeSignal::Ricker { freq0, .. } => (*freq0, [0.25 * freq0, 3.0 * freq0], [0.5 * freq0, 2.0 * freq0]),
            TimeSignal::Custom { .. } => return None,
        };
        lo = [lo[0].min(e[0]), lo[1].min(u[0])];
        hi = [hi[0].max(e[1]), hi[1].max(u[1])];
        centre += f0 / sources.len() as f64;
    }
    (lo[0].is_finite() && lo[0] > 0.0).then_some(SourceBand { edge_lo: TAU * lo[0], used_lo: TAU * lo[1], centre: TAU * centre, used_hi: TAU * hi[1], edge_hi: TAU * hi[0] })
}

/// The poles of `medium` as they are used inside a stabilised absorbing layer
/// and the amount added there to its high-frequency permittivity, or `None`
/// when nothing changes.
///
/// A lossless Lorentz pole makes its medium metal-like between its resonance
/// w0 and w0 sqrt(1 + d_eps / eps_inf): Re eps < 0. A dielectric core clad
/// in such a medium (or, for a strong pole, made of it) carries backward
/// modes in that band (phase and group velocity opposed, as in a
/// metal-insulator-metal guide above the surface-plasmon frequency), and a
/// PML amplifies backward waves, so a guide that runs into the layer makes
/// the fields grow there, at a frequency far from the sources', until the
/// run diverges. Two kinds of pole do this in fits of transparent materials:
///
/// * the far-infrared pole of a glass (its Sellmeier infrared term), below
///   the band. Inside the layers of a region that holds a structure it gets
///   the damping at which the medium has no band of negative permittivity:
///   the susceptibility of an oscillator with g = gamma / w0 is never below
///   -d_eps / (g (g + 2)), so with n such poles
///
///     gamma = w0 (sqrt(1 + 2 n d_eps / eps_inf) - 1)
///
///   keeps Re eps >= eps_inf / 2 at every frequency. In the band this
///   changes the permittivity by about d_eps w0^2 gamma / w^3 (mostly
///   absorption).
/// * the ultraviolet pole, above the band (silicon fitted with eps_inf = 1
///   has one of strength 10.7). Damping it would change the band by order
///   one, so inside the layers it is frozen instead: its susceptibility at
///   the centre of the band goes into the high-frequency permittivity and
///   the pole is switched off (strength 0). The change in the band is the
///   pole's dispersion, zero at the centre.
///
/// Either change reflects (|d chi| / (4 eps))^2 of the power at the face of
/// the layer (1e-7 for fused silica around 1.55 um). A pole is treated only
/// if that stays below 1e-4 at the ends of the band that matters, i.e. the
/// change is under 4 % of the permittivity there; a pole is frozen only in a
/// plain medium (no tensor, nonlinearity, gain or plug-in) and if it is
/// lossless in the band. Poles in or near the band, Drude and Debye poles
/// are left as they are.
pub fn pml_pole_variant(medium: &Medium, band: &SourceBand) -> Option<(Vec<crate::materials::Pole>, f64)> {
    use crate::materials::Pole;
    if medium.poles.is_empty() {
        return None;
    }
    let eps_inf = (0..3).filter(|&a| medium.pole_axes[a]).map(|a| medium.eps[a]).fold(f64::INFINITY, f64::min);
    if !(eps_inf.is_finite() && eps_inf > 0.0) {
        return None;
    }
    // the permittivity at the two ends of the band (real part), from all the poles
    let eps_at = |w: f64| eps_inf + medium.poles.iter().map(|p| p.chi(w).re).sum::<f64>();
    let (eps_lo, eps_hi) = (eps_at(band.used_lo), eps_at(band.used_hi));
    if eps_lo <= 0.0 || eps_hi <= 0.0 {
        return None;
    }
    let mut poles = medium.poles.clone();
    let mut changed = false;
    // poles above the band: frozen at the centre of the band
    let plain = is_plain_medium(medium) && !medium.pec;
    let mut extra = 0.0;
    if plain {
        for p in poles.iter_mut() {
            let Pole::Lorentz { delta_eps, omega0, gamma } = *p else { continue };
            let g = gamma / omega0;
            // (a pole damped so much that it never makes eps negative needs nothing)
            if !(delta_eps > 0.0 && omega0 > band.edge_hi && delta_eps / (g * (g + 2.0)) > eps_inf) {
                continue;
            }
            let c = p.chi(band.centre);
            let small = |w: f64, eps: f64| (p.chi(w) - c.re).norm() <= 0.04 * eps;
            if small(band.used_lo, eps_lo) && small(band.used_hi, eps_hi) && c.im.abs() <= 1e-3 * eps_lo.min(eps_hi) {
                extra += c.re;
                *p = Pole::Lorentz { delta_eps: 0.0, omega0, gamma };
                changed = true;
            }
        }
    }
    // poles below the band: damped; each may lower Re eps by eps_inf / (2 n)
    let eps_inf = eps_inf + extra;
    let low = |p: &Pole| matches!(*p, Pole::Lorentz { delta_eps, omega0, .. } if delta_eps > 0.0 && omega0 > 0.0 && omega0 < band.edge_lo);
    let n = poles.iter().filter(|p| low(p)).count() as f64;
    for p in poles.iter_mut() {
        let Pole::Lorentz { delta_eps, omega0, gamma } = *p else { continue };
        if !low(p) {
            continue;
        }
        let g = omega0 * ((1.0 + 2.0 * n * delta_eps / eps_inf).sqrt() - 1.0);
        let q = Pole::Lorentz { delta_eps, omega0, gamma: g };
        if g > gamma && (q.chi(band.used_lo) - p.chi(band.used_lo)).norm() <= 0.04 * eps_lo {
            *p = q;
            changed = true;
        }
    }
    changed.then_some((poles, extra))
}

/// The stabilised absorbing layers (see [`pml_pole_variant`]).
///
/// Needed only when the region holds a structure of another medium than
/// the background: the backward modes live on such a structure (their
/// tails reach micrometres into the cladding, so the layers beside a guide
/// matter as much as those it runs into), and a uniform medium has none.
/// (Decided from the shapes, so every slab of a decomposed run makes the
/// same choice.)
pub(super) struct StabilisedLayers {
    /// (axis, first cell, end cell) of each stabilised layer, global indices
    cells: Vec<(usize, usize, usize)>,
    /// per medium: its poles inside the layers and what they add to eps_inf there
    pub variant: Vec<Option<(Vec<Pole>, f64)>>,
}

impl StabilisedLayers {
    pub fn new(s: &Setup) -> StabilisedLayers {
        let (grid, n, b, media) = (s.grid, s.n, &s.sim.boundaries, &s.media);
        let mixed = s.structures.iter().any(|st| {
            let (b0, b1) = st.shape.bbox();
            st.medium != s.background && (0..3).all(|t| b1[t] >= grid.edges(t)[0] && b0[t] <= grid.edges(t)[n[t]])
        });
        let cells: Vec<(usize, usize, usize)> = (0..3)
            .flat_map(|a| b.axis(a).iter().enumerate().map(move |(s, bd)| (a, s, bd)))
            .filter_map(|(a, s, bd)| match bd {
                Boundary::Pml { layers, params, .. } if mixed && *layers > 0 && params.stabilize => {
                    Some(if s == 0 { (a, 0, *layers) } else { (a, n[a] - *layers, n[a]) })
                }
                _ => None,
            })
            .collect();
        let variant: Vec<Option<(Vec<Pole>, f64)>> = match source_band(&s.sim.sources) {
            Some(band) if !cells.is_empty() => media.iter().map(|m| pml_pole_variant(m, &band)).collect(),
            _ => vec![None; media.len()],
        };
        StabilisedLayers { cells, variant }
    }

    /// Whether global cell `g` lies inside a stabilised layer.
    pub fn contains(&self, g: [usize; 3]) -> bool {
        self.cells.iter().any(|&(a, lo, hi)| g[a] >= lo && g[a] < hi)
    }
}

/// Adiabatic absorbers.
///
/// The rate sigma^ (1/s) of the matched pair of conductivities,
/// sigma_e = sigma^ eps0 eps and sigma_m = sigma^ mu0 mu, per axis at the
/// nodes and at the cell centres; where two absorbers overlap the rates add.
pub(super) struct Absorbers {
    node: [Vec<f64>; 3],
    mid: [Vec<f64>; 3],
    /// the cells of this slab inside each absorber
    pub slabs: Vec<AbsorbSlab>,
    /// any face closed by an absorber of at least one layer
    pub any: bool,
    /// The electric conductivity is matched to the permittivity at the centre
    /// of the source band: per medium, what its poles add to eps_inf there
    /// (0 without poles, or when the band is unknown).
    chi: Vec<f64>,
}

impl Absorbers {
    pub fn new(s: &Setup) -> Result<Absorbers> {
        let (grid, n, b, media, sim) = (s.grid, s.n, &s.sim.boundaries, &s.media, s.sim);
        let (nx, ny, nz, x0, nxl) = (n[0], n[1], n[2], s.x0, s.nxl);
        let mut abs_node: [Vec<f64>; 3] = [vec![0.0; nx + 1], vec![0.0; ny + 1], vec![0.0; nz + 1]];
        let mut abs_mid: [Vec<f64>; 3] = [vec![0.0; nx], vec![0.0; ny], vec![0.0; nz]];
        let mut absorb: Vec<AbsorbSlab> = Vec::new();
        // index of the background medium at the centre of the source band (at least 1)
        let abs_n_bg = {
            let m = &media[s.background];
            let chi: f64 = source_band(&sim.sources).map(|band| m.poles.iter().map(|p| p.chi(band.centre).re).sum()).unwrap_or(0.0);
            ((0..3).map(|a| (m.eps[a] + if m.pole_axes[a] { chi } else { 0.0 }).abs().sqrt()).sum::<f64>() / 3.0).max(1.0)
        };
        for a in 0..3 {
            let e = grid.edges(a);
            for (s, bd) in b.axis(a).iter().enumerate() {
                let Boundary::Absorber { layers, params } = bd else { continue };
                let lyr = (*layers).min(n[a]);
                if lyr == 0 {
                    continue;
                }
                let (g0, g1) = if s == 0 { (0, lyr) } else { (n[a] - lyr, n[a]) };
                let (inner, outer) = if s == 0 { (e[lyr], e[0]) } else { (e[n[a] - lyr], e[n[a]]) };
                let thick = (inner - outer).abs();
                let smax = params.sigma_scale * ABSORBER_ATTENUATION * (params.order + 1.0) * C0 / (2.0 * thick * abs_n_bg);
                let rate = |x: f64| {
                    let d = if s == 0 { inner - x } else { x - inner };
                    if d > 0.0 { smax * (d / thick).min(1.0).powf(params.order) } else { 0.0 }
                };
                for g in 0..=n[a] {
                    abs_node[a][g] += rate(e[g]);
                }
                for g in 0..n[a] {
                    abs_mid[a][g] += rate(0.5 * (e[g] + e[g + 1]));
                }
                let mut start = [0usize; 3];
                let mut size = [nxl, ny, nz];
                let (l0, l1) = if a == 0 { (g0.max(x0), g1.min(x0 + nxl)) } else { (g0, g1) };
                if l0 >= l1 {
                    continue;
                }
                start[a] = if a == 0 { l0 - x0 } else { l0 };
                size[a] = l1 - l0;
                absorb.push(AbsorbSlab { axis: a, start, size });
            }
        }
        let any_abs = b.x.iter().chain(b.y.iter()).chain(b.z.iter()).any(|bd| bd.is_absorber() && bd.pml_layers() > 0);
        let abs_chi: Vec<f64> = match source_band(&sim.sources) {
            Some(band) if any_abs => media.iter().map(|m| m.poles.iter().map(|p| p.chi(band.centre).re).sum()).collect(),
            _ => vec![0.0; media.len()],
        };
        if any_abs && !sim.sheets.is_empty() {
            bail!("zero-thickness sheets cannot be combined with an Absorber boundary yet; use a PML");
        }
        Ok(Absorbers { node: abs_node, mid: abs_mid, slabs: absorb, any: any_abs, chi: abs_chi })
    }

    /// Rate at E_c of global cell g: E_c sits at the cell centre along c and
    /// on the nodes along the other two axes.
    pub fn e_rate(&self, c: usize, g: [usize; 3]) -> f64 {
        (0..3).map(|t| if t == c { self.mid[t][g[t]] } else { self.node[t][g[t]] }).sum()
    }

    /// Rate at H_c of global cell g: on the nodes along c, at the cell
    /// centres along the other two axes.
    pub fn h_rate(&self, c: usize, g: [usize; 3]) -> f64 {
        (0..3).map(|t| if t == c { self.node[t][g[t]] } else { self.mid[t][g[t]] }).sum()
    }

    /// f = sigma_e dt / (2 eps0 eps_inf) of the E_a sample at global cell g
    /// (local index o, medium m) whose high-frequency permittivity is
    /// `eps_now`, with sigma_e matched to the sample's permittivity at the
    /// centre of the band (its poles with their weight `disp_weight`, the
    /// frozen ones of a stabilised layer included).
    pub fn e_loss(&self, s: &Setup, layers: &StabilisedLayers, disp_weight: &[Vec<Real>; 3], a: usize, g: [usize; 3], o: usize, m: usize, eps_now: f64) -> f64 {
        let sr = self.e_rate(a, g);
        if sr == 0.0 {
            return 0.0;
        }
        let md = &s.media[m];
        let mut eps_c = eps_now;
        if md.pole_axes[a] {
            let w = disp_weight[a].get(o).filter(|w| !w.is_nan()).map(|&w| w as f64).unwrap_or(1.0);
            let frozen = match &layers.variant[m] {
                Some((_, extra)) if layers.contains(g) => *extra,
                _ => 0.0,
            };
            eps_c += w * (self.chi[m] - frozen);
        }
        0.5 * sr * s.dt * (eps_c.abs() / eps_now).clamp(0.1, 1e3)
    }
}

/// The CPML slabs of this slab's part of the grid.
pub(super) fn cpml_slabs(s: &Setup) -> Vec<PmlSlab> {
    let (grid, n, b, dt, x0, nxl, ny, nz) = (s.grid, s.n, &s.sim.boundaries, s.dt, s.x0, s.nxl, s.ny, s.nz);
    let mut pml = Vec::new();
    for a in 0..3 {
        for (s, bd) in b.axis(a).iter().enumerate() {
            let Boundary::Pml { layers, params, .. } = bd else { continue };
            let lyr = *layers;
            if lyr == 0 {
                continue;
            }
            let e = grid.edges(a);
            let n_a = n[a];
            let (g0, g1) = if s == 0 { (0, lyr) } else { (n_a - lyr, n_a) };
            let (inner, outer) = if s == 0 { (e[lyr], e[0]) } else { (e[n_a - lyr], e[n_a]) };
            let thick = (inner - outer).abs();
            let dcell = if s == 0 { e[1] - e[0] } else { e[n_a] - e[n_a - 1] };
            let m = params.order;
            let sigma_max = params.sigma_scale * 0.8 * (m + 1.0) / (ETA0 * dcell);
            let coef = |x: f64| -> (Real, Real, Real) {
                let d = ((x - inner).abs() / thick).clamp(0.0, 1.0);
                let sigma = sigma_max * d.powf(m);
                let kappa = 1.0 + (params.kappa_max - 1.0) * d.powf(m);
                let alpha = params.alpha_max * (1.0 - d);
                let bb = (-(sigma / kappa + alpha) * dt / EPS0).exp();
                let aa = if sigma > 0.0 { sigma / (sigma * kappa + kappa * kappa * alpha) * (bb - 1.0) } else { 0.0 };
                (bb as Real, aa as Real, (1.0 / kappa) as Real)
            };
            // local box
            let mut start = [0usize, 0, 0];
            let mut size = [nxl, ny, nz];
            let (l0, l1) = if a == 0 {
                let lo = g0.max(x0);
                let hi = g1.min(x0 + nxl);
                if lo >= hi {
                    continue;
                }
                (lo, hi)
            } else {
                (g0, g1)
            };
            if a == 0 {
                start[0] = l0 - x0;
                size[0] = l1 - l0;
            } else {
                start[a] = l0;
                size[a] = l1 - l0;
            }
            let mut eb_ = vec![];
            let mut ea_ = vec![];
            let mut ek_ = vec![];
            let mut hb_ = vec![];
            let mut ha_ = vec![];
            let mut hk_ = vec![];
            for gi in l0..l1 {
                let (b1, a1, k1) = coef(e[gi]);
                eb_.push(b1);
                ea_.push(a1);
                ek_.push(k1);
                let (b2, a2, k2) = coef(0.5 * (e[gi] + e[gi + 1]));
                hb_.push(b2);
                ha_.push(a2);
                hk_.push(k2);
            }
            pml.push(PmlSlab { axis: a, start, size, e_b: eb_, e_a: ea_, e_kinv: ek_, h_b: hb_, h_a: ha_, h_kinv: hk_ });
        }
    }
    pml
}
