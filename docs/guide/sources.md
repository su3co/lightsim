# Sources

After this page you can choose the time signal and the kind of source for a
simulation, you know what each source assumes about its surroundings, and you
can normalise results by the source spectrum.

A source has two parts: a **signal**, which says how the excitation varies in
time, and a **spatial type**, which says where and with which field pattern it
is injected. Add a source with `sim.add_source(...)`. Give each source a
`name` if you want to look up its spectrum afterwards.

## Time signals

| Signal | Use |
|---|---|
| `GaussianPulse(freq0, fwidth)` | A pulse covering a band around `freq0`. The usual choice. |
| `ContinuousWave(freq0, fwidth)` | A single frequency, switched on smoothly. |
| Ricker wavelet | A broadband pulse without a carrier. |
| Custom samples | Any waveform. |

A Gaussian pulse is

$$
s(t) = A \cos\big(2\pi f_0 (t - t_0) + \varphi\big)\, e^{-(t - t_0)^2 / 2\tau^2},
\qquad \tau = \frac{1}{2\pi\, f_{\text{width}}}, \qquad t_0 = \text{offset} \cdot \tau .
$$

Its spectrum is a Gaussian centred on `freq0` with standard deviation
`fwidth`. Frequencies further than about two to three times `fwidth` from the
centre carry little energy, and results there are noisy. The pulse peaks at
$t_0$, five times $\tau$ after the start by default, so that it starts from a
negligible value.

| Parameter | Meaning | Default |
|---|---|---|
| `freq0` | Centre frequency in Hz | required |
| `fwidth` | Spectral width in Hz | required |
| `offset` | Peak time in units of $\tau$ | 5 |
| `phase` | Carrier phase $\varphi$ in radians | 0 |
| `amplitude` | Peak value $A$ | 1 |

`ContinuousWave` takes the same parameters. It is
$A\cos(2\pi f_0 t + \varphi)$, multiplied before $t_0$ by a Gaussian ramp of
width $\tau$. Use it to watch a steady state build up. For spectra, a pulse
gives every frequency in one run and lets the run stop early.

The Ricker wavelet and custom samples have no builder function; pass the
dictionary:

```python
ricker = {"type": "ricker", "freq0": 3e14}                  # peak frequency in Hz
samples = {"type": "custom", "times": [0, 5e-15, 10e-15], "values": [0, 1, 0]}
```

Custom samples are interpolated linearly and are zero outside the given
times.

> **Note:** Oblique plane waves, focused beams and modes of lossy waveguides
> have a complex field pattern. They need the signal in quadrature, which is
> available for `GaussianPulse` and `ContinuousWave` only. Use one of these
> two with such sources.

## Kinds of source

| Source | Builder | Typical use |
|---|---|---|
| Point dipole | `PointDipole` | Emitters, exciting cavities, radiated power |
| Plane wave | `PlaneWave` | Transmission and reflection of layers and periodic structures |
| Gaussian beam | `GaussianBeam` | Free-space beams, fibre and grating coupling |
| Waveguide mode | `ModeSource` | Integrated photonic components |
| Total-field/scattered-field box | `TFSF` | Scattering by an isolated object |

The plane wave, the Gaussian beam and the mode source are **one-sided**: they
launch light in the stated direction only. The region behind the source plane
contains no incident field, only what is reflected back, so a monitor placed
there measures reflection directly.

The part of such a source plane that falls inside an absorbing boundary
layer is not injected.

### Point dipole

`PointDipole(position, polarization, signal, amplitude=1.0, name="")`

`polarization` is a field component. `"Ex"`, `"Ey"` or `"Ez"` is an electric
dipole: a current element of moment `amplitude * s(t)` in ampere metres.
`"Hx"`, `"Hy"` or `"Hz"` is a magnetic dipole with moment in volt metres. The
dipole is placed on the grid position of that component nearest to
`position`.

### Plane wave

`PlaneWave(axis, position, direction, polarization, signal, amplitude=1.0, angle=0.0, min=None, max=None, name="", fixed_angle=False)`

| Parameter | Meaning |
|---|---|
| `axis`, `position` | The source plane is normal to `axis` at this coordinate |
| `direction` | `"+"` or `"-"` along `axis` |
| `polarization` | A component in the plane: electric (`"Ey"` for `axis=0`) sets E along that axis, magnetic (`"Hz"` for `axis=0`) sets H along that axis and E along the other in-plane axis |
| `amplitude` | Peak electric field in V/m, multiplied by the signal |
| `angle` | Tilt of the wavevector in radians, see below |
| `min`, `max` | Corners limiting the extent of the plane; the whole plane by default |
| `fixed_angle` | Hold `angle` at every frequency of the pulse (one broadband run at one angle), see [below](#broadband-oblique-incidence-at-a-fixed-angle) |

The wave fills the plane, so the side boundaries decide what it is. With
periodic boundaries it is an infinite plane wave. With absorbing side
boundaries the edges of the plane diffract.

The index of the medium is taken at the centre of the plane and applied over
the whole plane. Put the plane in a homogeneous region. A dispersive medium at
the plane is handled exactly at normal incidence.

With `angle` set, the wavevector is tilted by that angle towards the in-plane
axis **perpendicular to the polarisation axis**. The side boundaries along
that axis must be Bloch-periodic with $k = n\,k_0 \sin\theta$, where
$k_0 = 2\pi f_0 / c$ (see [Boundaries](boundaries.md)).

- With an electric `polarization` (for example `"Ez"` on a plane normal to
  x, tilted towards y) the wave is s-polarised: E is normal to the plane of
  incidence and has the amplitude `amplitude`.
- With a magnetic `polarization` (for example `"Hz"`, also tilted towards y)
  the wave is p-polarised: H is normal to the plane of incidence and E lies
  in it, with the component `amplitude * cos(angle)` along the tilt axis (on
  the plane) and `amplitude * sin(angle)` along the normal. The power is the
  same as for s-polarisation with the same `amplitude`.
- In a 2-D simulation an electric `polarization` in the plane is
  p-polarised as well (see [2-D, 2.5-D and 3-D](dimensions.md)); a magnetic
  one at an angle must be along the invariant axis.

Points to know:

- The tilt is imposed as a phase gradient computed at `freq0`. The transverse
  wavevector is the same at every frequency, so the angle is exact at `freq0`
  and changes with frequency as $\sin\theta(f) = \sin\theta\, f_0 / f$. Use a
  narrow pulse, the fixed-angle sweep described next, or a fixed-angle run
  (`fixed_angle=True`, described after it), which holds the angle at every
  frequency of a broadband pulse.
- Such a run uses complex fields. `Results.source_spectrum` accounts for
  that, so fields and powers are normalised as at normal incidence. See the
  normalisation section below.

The example below runs a p-polarised wave onto an n = 2 half-space at 30°
and at Brewster's angle, $\arctan 2 = 63.4°$, and compares the reflected and
transmitted fractions with the Fresnel formulas. The incident power is
taken from a run without the glass.

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam0 = 1.0 * um
f0 = ls.C0 / lam0
dx = lam0 / 40
ly, lz = 8 * dx, 2 * dx
n2 = 2.0

def fresnel_run(theta, glass):
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, ly, lz], mesh=ls.UniformMesh(dx), run_time=300e-15)
    ky = 2 * np.pi * f0 / ls.C0 * np.sin(theta)
    sim.set_boundaries(y=ls.Bloch(ky), z=ls.PERIODIC)
    if glass:
        sim.add_medium(ls.Medium("glass", n=n2))
        sim.add_structure(ls.Box([2.5 * um, -1, -1], [10 * um, 1, 1]), "glass")
    # H along z: E lies in the xy plane of incidence (p-polarisation)
    sim.add_source(ls.PlaneWave(0, 1.0 * um, "+", "Hz", ls.GaussianPulse(f0, 0.06 * f0), angle=theta))
    sim.add_monitor(ls.FluxMonitor("R", 0, 0.5 * um, [0.5 * um, -1, -1], [0.5 * um, 1, 1], [f0]))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.2 * um, [3.2 * um, -1, -1], [3.2 * um, 1, 1], [f0]))
    return sim.run(device="cpu")

for deg in (30.0, np.degrees(np.arctan(n2))):
    theta = np.radians(deg)
    incident = fresnel_run(theta, False).flux("T")[0]
    res = fresnel_run(theta, True)
    ct, tt = np.cos(theta), np.arcsin(np.sin(theta) / n2)
    r_p = ((n2 * ct - np.cos(tt)) / (n2 * ct + np.cos(tt))) ** 2
    print(f"{deg:.1f} deg: R = {-res.flux('R')[0] / incident:.2e} (Fresnel {r_p:.4f}), "
          f"T = {res.flux('T')[0] / incident:.4f} (Fresnel {1 - r_p:.4f})")
```

```text
30.0 deg: R = 7.76e-02 (Fresnel 0.0800), T = 0.9224 (Fresnel 0.9200)
63.4 deg: R = -7.51e-06 (Fresnel 0.0000), T = 0.9998 (Fresnel 1.0000)
```

At 20 cells per wavelength in the glass, R is 3 % below the Fresnel value at
30°, mostly from the grid dispersion of the discretised interface; T is
within 0.3 %. At Brewster's angle the reflected fraction is below 1e-5. The
validation case `bloch_oblique_p` checks the same quantities, the
wavevector and the two electric components of the incident wave.

### Spectra at a fixed angle

Because an oblique plane wave fixes the transverse wavevector, one broadband
run gives a spectrum whose angle changes with frequency. `lightsim.sweep_angle`
gives a spectrum at one angle instead: it runs one simulation per frequency
point, each with the transverse wavevector $k_t = n\,k_0(f) \sin\theta$ and a
pulse centred on that frequency, and assembles the results.

```text
sweep_angle(build, freqs, angle, *, n_medium=1.0, bandwidth=None, devices="cpu", progress=False) -> AngleSweep
```

For each frequency `f` the sweep calls `build(f, signal, k_t)`, which returns
the simulation for that point: put `signal` in the `PlaneWave` (with
`angle=angle`), `k_t` in the `Bloch` boundaries along the tilt axis, and
record `f` in the monitors. `n_medium` is the index at the source plane.
`bandwidth` is the relative width of the pulses; by default it keeps the
pulse spectrum above $f \sin\theta$, where the fixed $k_t$ turns grazing,
and at most 10 %. `devices` is a device string or a list of them
(`["cuda:0", "cuda:1"]`); the runs are independent and are dispatched
concurrently, one at a time per entry.

The returned `AngleSweep` has `freqs`, `k_t` and the list of `results`.
`flux(name)`, `field(name, component)` and `source_spectrum()` return each
run's value at its own frequency, and `incident_power(area)` the analytic
incident power $n\,|S|^2 A \cos\theta / (2\eta_0)$.

This is a different method from Lumerical's BFAST, which obtains a
fixed-angle spectrum from a single run. The sweep costs one run (two with a
reference run) per frequency point, and each run is longer than a broadband
one because its pulse is narrower. In return, each point is an ordinary
oblique run, exact at its frequency. A single-run mode of lightsim is described in the next section.

The example computes the reflection of a 2 um glass slab at 45° for
s-polarisation at five frequencies and compares it with the Airy formula at
45°. The incident power comes from a sweep without the slab.

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam0 = 1.0 * um
f0 = ls.C0 / lam0
dx = lam0 / 60
ly, lz = 8 * dx, 2 * dx
n_slab, d_slab = 1.5, 120 * dx                 # a 2 um glass slab
theta = np.radians(45.0)
freqs = np.linspace(0.85, 1.15, 5) * f0

def builder(slab):
    def build(f, signal, k_t):
        sim = ls.Simulation(min=[0, 0, 0], max=[7 * um, ly, lz], mesh=ls.UniformMesh(dx),
                            run_time=400e-15, shutoff=1e-6)
        sim.set_boundaries(y=ls.Bloch(k_t), z=ls.PERIODIC)
        if slab:
            sim.add_medium(ls.Medium("glass", n=n_slab))
            sim.add_structure(ls.Box([3 * um, -1, -1], [3 * um + d_slab, 1, 1]), "glass")
        sim.add_source(ls.PlaneWave(0, 1.5 * um, "+", "Ez", signal, angle=theta))
        for name, x in (("R", 1.0 * um), ("T", 6.0 * um)):
            sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], [f]))
        return sim
    return build

incident = ls.sweep_angle(builder(False), freqs, theta).flux("T")
sweep = ls.sweep_angle(builder(True), freqs, theta)
R, T = -sweep.flux("R") / incident, sweep.flux("T") / incident

# Airy formula for the slab at 45 degrees, s-polarisation
c1, c2 = np.cos(theta), np.sqrt(1 - (np.sin(theta) / n_slab) ** 2)
r12 = (c1 - n_slab * c2) / (c1 + n_slab * c2)
e = np.exp(2j * (2 * np.pi * freqs / ls.C0) * n_slab * d_slab * c2)
R_airy = np.abs(r12 * (1 - e) / (1 - r12**2 * e)) ** 2
for f, r, ra, t in zip(freqs, R, R_airy, T):
    print(f"{f / f0:.3f} f0  R = {r:.4f} (Airy {ra:.4f})  R + T = {r + t:.4f}")
```

```text
0.850 f0  R = 0.3079 (Airy 0.3086)  R + T = 1.0002
0.925 f0  R = 0.0426 (Airy 0.0450)  R + T = 1.0000
1.000 f0  R = 0.2210 (Airy 0.2192)  R + T = 0.9999
1.075 f0  R = 0.2304 (Airy 0.2352)  R + T = 1.0000
1.150 f0  R = 0.0337 (Airy 0.0303)  R + T = 1.0001
```

The test `python/tests/test_sweep_angle.py` does the same at nine
frequencies over ±15 %, for s and p polarisation: R and T are within 4.8e-3
of the Airy formula for s and 9.4e-4 for p (60 cells per vacuum wavelength;
the residual is the grid dispersion in the slab, which shifts the fringes).
A single broadband run with $k_t$ fixed at the centre frequency covers
angles from 38° to 56° over the same band; its transmission is within
4.8e-3 of the Airy formula at those angles and differs from the 45° curve by
up to 0.17 (s) and 0.04 (p). Its reflection is also off at the band edges,
because the source launches a backward wave of up to 1.5 % of the incident
power away from the centre frequency.

### Broadband oblique incidence at a fixed angle

`PlaneWave(..., angle=theta, fixed_angle=True)` gives the response of a
structure that is periodic along the tilt axis to a plane wave at the angle
`theta` at every frequency of a broadband pulse, from one run. Reflection
and transmission spectra, diffraction efficiencies and field maps at one
angle then cost one run (and one empty reference run) instead of one run
per frequency.

The example computes the reflection of a 0.6 um glass slab at 60° for
s-polarisation at seven frequencies over ±30 % and compares it with the
Airy formula at 60°:

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam0 = 1.0 * um
f0 = ls.C0 / lam0
dx = lam0 / 40
theta = np.radians(60.0)
n_slab, d_slab = 1.5, 0.6 * um
freqs = np.linspace(0.7, 1.3, 7) * f0


def run(slab):
    # 2-D: x is the propagation axis, y the tilt axis (periodic), z invariant
    sim = ls.Simulation(min=[0, 0], max=[4 * um, 8 * dx], mesh=ls.UniformMesh(dx), run_time=300e-15, shutoff=1e-7)
    sim.set_boundaries(x=ls.PML(), y=ls.PERIODIC)
    if slab:
        sim.add_structure(ls.Box([2 * um, -1], [2 * um + d_slab, 1]), ls.Medium("glass", n=n_slab))
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ez", ls.GaussianPulse(f0, 0.25 * f0), angle=theta, fixed_angle=True))
    for name, x in (("R", 0.6 * um), ("T", 3.2 * um)):
        sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1], [x, 1], freqs))
    return sim.run(device="cpu")


incident = run(False).flux("T")
res = run(True)
R, T = -res.flux("R") / incident, res.flux("T") / incident

# Airy formula at 60 degrees, s polarisation
c1, c2 = np.cos(theta), np.sqrt(1 - (np.sin(theta) / n_slab) ** 2)
r12 = (c1 - n_slab * c2) / (c1 + n_slab * c2)
e = np.exp(2j * (2 * np.pi * freqs / ls.C0) * n_slab * d_slab * c2)
R_airy = np.abs(r12 * (1 - e) / (1 - r12**2 * e)) ** 2
for f, r, ra, t in zip(freqs, R, R_airy, T):
    print(f"{f / f0:.1f} f0  R = {r:.4f} (Airy {ra:.4f})  R + T = {r + t:.4f}")
```

```text
0.7 f0  R = 0.0088 (Airy 0.0084)  R + T = 0.9998
0.8 f0  R = 0.2233 (Airy 0.2227)  R + T = 1.0000
0.9 f0  R = 0.4280 (Airy 0.4288)  R + T = 1.0001
1.0 f0  R = 0.5055 (Airy 0.5079)  R + T = 1.0000
1.1 f0  R = 0.4710 (Airy 0.4759)  R + T = 1.0000
1.2 f0  R = 0.3143 (Airy 0.3226)  R + T = 0.9999
1.3 f0  R = 0.0667 (Airy 0.0741)  R + T = 1.0000
```

At 40 cells per vacuum wavelength R is within 7.4e-3 of the Airy formula
over the band; the residual is the grid dispersion in the slab, which shifts
the fringes, and it is the same as that of a narrow-band Bloch run at each
frequency (see the measured accuracy below).

**What the run needs.**

- One source, the plane wave, filling its plane (no `min`/`max`), starting in
  a homogeneous, isotropic, lossless and non-dispersive medium. Any real
  signal works; a `GaussianPulse` whose band covers the frequencies of
  interest is the usual choice.
- `PERIODIC` boundaries along the tilt axis (not `Bloch`: the run imposes
  the phase of every frequency itself) and along the other transverse axis
  (or a 2-D simulation whose invariant axis is that axis), and PML or walls
  along the propagation axis.
- The tilt axis follows the polarisation as for any oblique plane wave: an
  E polarisation is s-polarised and tilted towards the in-plane axis normal
  to it, an H polarisation is p-polarised; in 2-D an E polarisation in the
  plane is p-polarised and an H polarisation must be along the invariant
  axis.
- Media: diagonal (isotropic or anisotropic) permittivity and permeability,
  conductivity, and Lorentz, Drude and Debye poles. PEC works.
- Monitors: flux, flux box, DFT field and mode monitors.

**What it returns.** The frequency-domain data are physical fields: each DFT
sample is multiplied by $e^{i k_u(f)(u - u_0)}$ with
$k_u(f) = 2\pi f\, n \sin\theta / c$ ($u_0$ is the centre of the source
plane, as for the phase of a `Bloch` run), so fluxes, field maps and mode
amplitudes are those of the oblique wave at each frequency.
`Results.source_spectrum` is the spectrum of the real signal, and the plane
wave carries $n\,|A\,S|^2 \cos\theta / (2\eta_0)$ watts per square metre of
the plane, as for an ordinary oblique plane wave. For diffraction orders
give `lightsim.gratings` one transverse wavevector per frequency:

```text
kx = 2 * np.pi * freqs / ls.C0 * n * np.sin(theta)
eff = gratings.grating_efficiencies(res, "t", "r", axis, period_axis, n_t=..., n_r=..., kx_inc=kx, sim=sim, incident_power=...)
```

**The equations solved.** With $s = n \sin\theta$ and $u$ the tilt axis,
the physical fields are $\mathbf E(\mathbf r, t) = \mathbf P(\mathbf r, t - s(u - u_0)/c)$
with $\mathbf P$ periodic along $u$ for every frequency at once, and the
same for $\tilde{\mathbf H} = \eta_0 \mathbf H$ and its transform
$\mathbf Q$. The run solves for $\mathbf P$ and $\mathbf Q$, which obey
(time in units of $c\,\Delta t$, $\mathbf J$ the conduction and polarisation
currents)

$$
\varepsilon\, \partial_t \mathbf P + \mathbf J = \nabla \times \mathbf Q - s\, \hat{\mathbf u} \times \partial_t \mathbf Q,
\qquad
\mu\, \partial_t \mathbf Q = -\nabla \times \mathbf P + s\, \hat{\mathbf u} \times \partial_t \mathbf P .
$$

The components along $u$ are updated as usual. The two pairs that
$\hat{\mathbf u}\times$ couples, $(P_v, Q_w)$ and $(P_w, Q_v)$, are updated
after eliminating the partner's time derivative; for a pair $(P_p, Q_q)$
with coupling sign $\sigma$ and $A$ the two-point average along $u$ between
the E and H samples of the Yee cell,

$$
\big(\varepsilon - s^2 A \mu^{-1} A\big)\, \partial_t P_p = (\nabla\times\mathbf Q)_p - J_p - s\, A \mu^{-1} \big(\partial_u P_p - \partial_p P_u\big),
$$

$$
\big(\mu - s^2 A \varepsilon_\infty^{-1} A\big)\, \partial_t Q_q = -(\nabla\times\mathbf P)_q + s\, A \varepsilon_\infty^{-1} \big(\partial_q Q_u - \partial_u Q_q - \sigma J_p\big).
$$

P stays at integer and Q at half-integer steps. The mass terms and the
derivative along $u$ of the field being updated are taken at the centre of
the step (Crank-Nicolson); every update of a coupled component is then a
cyclic tridiagonal solve along the lines of $u$, factorised once at the
start. All other terms are explicit and the curl updates, CPML and source
of the ordinary engine are used unchanged. Dispersive poles are integrated
with the trapezoidal rule as currents $J$ held at integer steps (so that
$J_p$ is known when the Q update needs it); the discrete permittivity is
the continuous one at $\Omega = (2/\Delta t)\tan(\omega \Delta t / 2)$. CPML
along the propagation axis enters the coupling terms through the same
stretched derivatives.

This is not Lumerical's BFAST. BFAST (after Liang et al., IEEE Trans.
Antennas Propag. 62, 354 (2014)) and the split-field method of Roden et al.
(IEEE Trans. MTT 46, 420 (1998)) update the same transformed fields
explicitly. An explicit update is limited by the fastest characteristic
speed along the tilt axis, $c/(n - s)$, so its time step shrinks about as
$1 - \sin\theta$ in vacuum (0.13 of the normal one at 60°). The implicit
treatment of the tilt axis here takes the tilt-axis spacing out of the
stability limit, which then shrinks as $\cos\theta$ and not at all at
moderate angles, at the price of the line solves. The transform and the
physics are the same, the discretisation is not, so results can differ
from BFAST by discretisation error.

**Time step.** The run chooses it; nothing needs tuning:

$$
c\,\Delta t = \texttt{courant} \times \min\!\Big( \frac{1}{\sqrt{\sum_a \Delta_a^{-2}}},\
\frac{\sqrt{\kappa - s^2}}{\sqrt{\sum_{a \ne u} \Delta_a^{-2}}} \Big),
\qquad \kappa = \min\big(1, \min \varepsilon_\infty \mu\big),
$$

the sums over the axes of the simulation (two in 2-D) with the smallest cell
sizes. The first term is the usual Courant limit. On a uniform mesh in
vacuum the second takes over above 35.3° in 3-D and 45° in 2-D:

| angle | 30° | 45° | 60° | 70° | 80° |
|---|---|---|---|---|---|
| time step, 3-D (of the normal one) | 1 | 0.87 | 0.61 | 0.42 | 0.21 |
| time step, 2-D | 1 | 1 | 0.71 | 0.48 | 0.25 |

**Largest angle.** 80°, both in the medium at the source and in the medium
of lowest index of the scene ($s \le \sqrt\kappa \sin 80°$). A larger angle
is refused with a message before the run starts. The validation case
`fixed_angle_stability` runs 80° (s and p, with a dielectric, a Drude and a
Lorentz layer) for ten times the duration of the pulse: the energy falls to
5e-10 of its peak and stays there.

**Measured accuracy.** From the validation cases (CPU, RTX PRO 6000 and GB10
agree to 1e-6), one fixed-angle run each:

| case | quantity | error |
|---|---|---|
| n = 1.5 slab, 2-D, 40 cells per wavelength, ±30 % | R and T vs Airy at 30° | 3.6e-3 (s), 2.2e-3 (p) |
| | at 60° | 9.3e-3 (s), 7.1e-4 (p) |
| | narrow-band Bloch runs at 0.7, 1, 1.3 f0 | 3.6e-3 (30°), 7.3e-3 (60°) |
| | R + T − 1 | ≤ 1.9e-4 |
| | incident power vs $\|S\|^2\cos\theta/2\eta_0$ | 3.9e-3 (30°), 1.3e-3 (60°) |
| | backward leakage of the source | 1e-9 |
| Brewster, n = 2 half-space, p, ±30 % | R | ≤ 6.3e-5 at every frequency |
| Lorentz slab, 45°, 60 cells | R, T vs Airy (discrete ε) | ≤ 7.2e-3 |
| Drude film (f_p = 4 f0), 30 nm, 45° | R, T vs Airy (discrete ε) | ≤ 1.0e-3 |
| lamellar grating, 15°, 0.8 to 1.2 f0 | order efficiencies vs narrow-band Bloch runs | ≤ 2.1e-4, orders up to 79° from the normal |
| 3-D slab, 45°, tilt along x | transmitted field at f0 vs a Bloch run | amplitude 1.3e-4, phase 9e-4 rad |

A narrow-band Bloch run at one frequency and the fixed-angle run share the
same spatial discretisation, and their spectra agree to the accuracy above;
the fixed-angle run has no grid dispersion along the tilt axis for the
specular order (its transformed fields are uniform along it), which
narrow-band runs do have.

FIXED_ANGLE_COST

**Limits.** The fixed-angle mode refuses, with a message, what it does not
model: more than one source, sources other than the plane wave, a plane
wave starting in a lossy or dispersive medium, Bloch boundaries along the
tilt axis, time-domain monitors (probes, field-time and energy monitors,
which would record the transformed fields), far-field monitors, nonlinear,
gain, gyrotropic, director-field and off-diagonal tensor media, magnetic
conductivity, angles above 80°, and the gradient engines. It tilts along one
axis only (no azimuth, no conical incidence). It runs on the CPU, on one GPU,
or split over several GPUs of one machine (`cuda:0,1`) when x is the
propagation axis: the slabs are cut along x, never along the tilt axis,
whose lines the coupled updates solve whole. Otherwise a request for
several GPUs runs on the first and says so. It does not run over several
processes or nodes (`lightsim launch` refuses it). A 2-D fixed-angle run
uses the general 3-D update with both polarisations, about ten times slower
per cell on the CPU than the specialised 2-D update of an ordinary run.

**When `sweep_angle` is the better choice:** a scene with something the
fixed-angle mode refuses (see the list above), a cell too large for the
GPUs of one machine (or for one GPU when x cannot be the propagation axis),
one or two frequencies only, or an angle above 80° in some medium.

### Gaussian beam

`GaussianBeam(axis, position, direction, polarization, center, waist, signal, amplitude=1.0, focus_distance=0.0, name="", theta=0.0, phi=0.0)`

| Parameter | Meaning |
|---|---|
| `direction` | `"+"` or `"-"`: the side of the plane the beam is launched to. Or a vector `(dx, dy, dz)` along the beam axis |
| `center` | A point on the beam axis; its two in-plane coordinates are where the axis crosses the source plane |
| `waist` | Radius at which the field falls to $1/e$ at the focus |
| `focus_distance` | Distance from `center` to the focus along the beam axis. Zero puts the focus on the plane; a negative value describes a beam that is already diverging |
| `theta` | Polar angle in radians between the beam axis and the normal of the plane, below $\pi/2$ |
| `phi` | Azimuth of the tilt in radians, from the first in-plane axis towards the second |
| `amplitude` | Peak electric field at the focus, in V/m |

The other parameters are as for the plane wave. The in-plane axes are (y, z)
for `axis=0`, (x, z) for `axis=1` and (x, y) for `axis=2`, so the beam axis is

$$
\hat{\mathbf d} = \pm\cos\theta\;\hat{\mathbf a}
 + \sin\theta\,\big(\cos\varphi\;\hat{\mathbf u} + \sin\varphi\;\hat{\mathbf v}\big) .
$$

The waist plane is normal to $\hat{\mathbf d}$ and passes through the focus.
The electric field points along the projection of the `polarization` axis
onto the plane normal to $\hat{\mathbf d}$. When the tilt is perpendicular to
that axis the field stays along it (s polarisation); when the tilt is towards
it, the field tilts with the beam (p polarisation).

The injected pattern is the scalar paraxial Gaussian beam at `freq0`. It is
accurate for waists of about a wavelength or more; a tightly focused beam is
not reproduced. The pattern spans the whole plane, so make the region several
waists wide where the beam crosses it. Points to know:

- The beam carries the power
  $P = n\,|A\,S|^2\,\pi w_0^2 / (4\eta_0)$ for an amplitude $A$, a waist
  $w_0$ and a medium of index $n$, whatever its angle.
- In a simulation that is periodic along an in-plane axis with a period
  smaller than twice the waist, which is how a two-dimensional problem is
  set up, the beam is uniform along that axis: a sheet beam, whose amplitude
  falls as $(w_0/w)^{1/2}$ away from the focus. It carries
  $P = n\,|A\,S|^2\,w_0\sqrt{\pi/2}\,L / (2\eta_0)$ over the period $L$, and
  its axis must lie in the plane of the simulation.
- A tilted or defocused beam has a complex pattern, computed at `freq0`: use
  a `GaussianPulse` or `ContinuousWave` signal, and a narrow pulse, because
  the transverse wavevector is the same at all frequencies and the angle
  therefore changes with frequency as $\sin\theta(f) = \sin\theta\, f_0/f$.

The example launches a sheet beam of 2.5 µm waist at 25° in the xy plane,
with the electric field along z, and checks its direction, its power and what
is emitted behind the source plane:

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam = 1.0 * um
f_c = ls.C0 / lam
h = lam / 20
w0 = 2.5 * um
theta = np.radians(25)
x_src, y_c = 2.0 * um, -1.0 * um

tilted = ls.Simulation(min=[0, -9 * um, 0], max=[12 * um, 9 * um, h],
                       mesh=ls.UniformMesh(h), run_time=160e-15, shutoff=1e-8)
tilted.set_boundaries(z=ls.PERIODIC)
tilted.add_source(ls.GaussianBeam(0, x_src, "+", "Ez", [x_src, y_c, 0], w0,
                                  ls.GaussianPulse(f_c, 0.08 * f_c),
                                  focus_distance=3 * um, theta=theta, name="beam"))
for name, x in (("near", 4 * um), ("far", 9 * um)):
    tilted.add_monitor(ls.DFTMonitor(name, [x, -9 * um, 0], [x, 9 * um, 0], [f_c], components=("Ez",)))
tilted.add_monitor(ls.FluxMonitor("forward", 0, 9 * um, [9 * um, -20 * um, -1], [9 * um, 20 * um, 1], [f_c]))
tilted.add_monitor(ls.FluxMonitor("behind", 0, 1 * um, [1 * um, -20 * um, -1], [1 * um, 20 * um, 1], [f_c]))
beam_run = tilted.run(device="cpu")
y_nodes = tilted.grid()[1]


def beam_centre(name):
    ez = beam_run.field(name, "Ez")[0, 0, :, 0]
    first = beam_run.raw["dft"][name]["fields"]["Ez"]["origin"][1]
    weight = np.abs(ez) ** 2
    return np.sum(weight * y_nodes[first:first + len(ez)]) / np.sum(weight)


angle = np.degrees(np.arctan((beam_centre("far") - beam_centre("near")) / (5 * um)))
_, S = beam_run.source_spectrum("beam", monitor="forward")
P_beam = abs(S[0]) ** 2 * w0 * np.sqrt(np.pi / 2) * h / (2 * 376.730313412)
print(f"direction {angle:.1f} degrees, power {beam_run.flux('forward')[0] / P_beam:.3f} of the formula, "
      f"behind the source {abs(beam_run.flux('behind')[0]) / P_beam:.0e}")
```

```text
direction 25.5 degrees, power 0.991 of the formula, behind the source 2e-04
```

The beam carries 99 % of the power of the formula at 20 cells per wavelength
and next to nothing is emitted backwards. The centre of $|E|^2$ moves at
25.5°: on planes that are not normal to a beam that converges and then
diverges, that centre runs slightly steeper than the beam axis.

`lightsim.gaussian_mode` takes the same `theta`, `phi`, `waist`, `center` and
`focus_distance` and returns the pattern of such a beam on a monitor plane,
for a `ModeMonitor` that measures how much of a field couples into a beam
arriving at an angle (see [Monitors](monitors.md)).

For an infinite plane wave at an angle, use `PlaneWave` with `angle` and
Bloch boundaries instead.

### Waveguide mode

`ModeSource(axis, position, direction, min, max, profile, signal, amplitude=1.0, name="")`

`profile` is the field pattern of a mode computed by `solve_modes` on the same
plane: pass `mode.profile`. `min` and `max` are the corners of the mode
window and must be those given to the mode solver. The profile carries unit
power, so the launched power spectrum is that of the signal.
[Modes and S-parameters](modes-sparams.md) has a complete example.

### Total-field/scattered-field box

`TFSF(min, max, axis, direction, polarization, signal, amplitude=1.0, name="")`

A plane wave travelling along `axis` exists only inside the box. Outside it
only the field scattered by objects in the box remains, which a flux monitor
around the box measures as scattered power. The wave is at normal incidence
to the box faces; there is no oblique version.

The incident wave is computed on a one-dimensional auxiliary grid with the
same spacing and time step, so its numerical dispersion matches the main grid
and very little leaks out of an empty box. Points to know:

- The medium around the box must be uniform. Its index, including dispersive
  poles, is read at one point on the box surface.
- If the box spans a periodic axis completely, the two faces normal to that
  axis are left out. This gives the incident wave for two-dimensional
  problems, such as scattering by an infinite cylinder.
- The file `examples/mie_sphere.py` in the repository computes the scattering
  cross-section of a sphere this way.

## Normalising by the source spectrum

Monitors record the response to the pulse. `Results.source_spectrum(name)`
returns the Fourier transform $S(\omega)$ of the source's `signal` (without
the source `amplitude`). Dividing a field by $S$, or a power by $|S|^2$,
gives the response per unit excitation.

The spectrum must be on the frequencies of the monitor it divides, in the
same order:

| Call | Frequencies |
|---|---|
| `source_spectrum(name, monitor="T")` | Those of the monitor `"T"`, in its order |
| `source_spectrum(name, freqs=f)` | The given ones, in the given order |
| `source_spectrum(name)` | Those of the monitors, if all monitors record the same list; otherwise an error |

`name` may be left out when the simulation has one source. A list of
wavelengths in ascending order is a list of frequencies in descending order;
the spectrum follows the order of the monitor, so the two always line up.

The worked example checks this for an electric dipole in vacuum. A dipole of
moment $p$ (in ampere metres) radiates the power

$$
P = \frac{\eta_0\, k^2\, |p|^2}{12\pi}, \qquad k = \frac{2\pi f}{c} .
$$

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam0 = 1.0 * um
f0 = ls.C0 / lam0
freqs = np.array([0.9, 1.0, 1.1]) * f0
half = 0.6 * um

sim = ls.Simulation(min=[-half] * 3, max=[half] * 3, mesh=ls.UniformMesh(lam0 / 20), run_time=40e-15)
moment = 1e-12                                          # dipole moment amplitude, A m
sim.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(f0, 0.3 * f0),
                              amplitude=moment, name="dipole"))
# power leaving a closed box around the dipole
sim.add_monitor(ls.FluxBoxMonitor("power", [-0.4 * um] * 3, [0.4 * um] * 3, freqs))
res = sim.run(device="cpu")

_, S = res.source_spectrum("dipole", monitor="power")
k = 2 * np.pi * freqs / ls.C0
P_exact = 376.730313412 * k**2 * np.abs(moment * S) ** 2 / (12 * np.pi)
print(np.round(np.asarray(res.flux("power")) / P_exact, 3))
```

```text
[1.009 1.01  1.012]
```

The simulated power is within about 1 % of the formula at 20 cells per
wavelength.

For a plane wave of unit amplitude the incident power through an area $A$
normal to the propagation direction, in a medium of index $n$, is
$P_{\text{inc}} = n\,|S|^2 A / (2\eta_0)$. For a mode source the incident power
is close to $|S|^2$ per unit `amplitude` squared, because the mode profile
carries unit power; S-parameter calculations measure it with a monitor
instead (see [Modes and S-parameters](modes-sparams.md)).

Points to know:

- **Oblique plane waves.** A run with Bloch boundaries uses complex fields,
  and a source with a complex field pattern injects the complex (analytic)
  signal into them, whose spectrum at positive frequencies is twice that of
  the real pulse. For such a source `source_spectrum` returns the spectrum of
  the analytic signal. The recorded field per unit amplitude is then $S$, as
  at normal incidence, and the power through a plane normal to the source
  axis is $P_{\text{inc}} = n\,|S|^2 A \cos\theta / (2\eta_0)$.
- **Coarse meshes.** The spectrum is that of the injected signal, not of the
  field on the grid. The two differ by the discretisation error, typically
  below 1 % at 20 cells per wavelength. A reference run removes this error
  as well, which is why it is the more accurate normalisation.
