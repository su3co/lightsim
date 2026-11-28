# Monitors

After this page you can record fields in the frequency and time domains,
measure power through planes and closed surfaces, project fields onto
waveguide modes and to the far field, and extract resonance frequencies and
quality factors from a ring-down.

A monitor is added with `sim.add_monitor(...)` and identified by its `name`
in the results. Every monitor of a simulation needs its own name, whatever
its type; a second monitor with a name that is taken is refused.

| Monitor | Records | Read with |
|---|---|---|
| `DFTMonitor` | Fields at chosen frequencies in a box, plane, line or point | `res.field(name, component)` |
| `FluxMonitor` | Power through a plane | `res.flux(name)` |
| `FluxBoxMonitor` | Net power out of a closed box | `res.flux(name)` |
| `ModeMonitor` | Amplitudes of waveguide modes on a plane | `res.mode_amplitudes(name, mode)` |
| `ProbeMonitor` | Field at a point at every time step | `res.time_series(name, component)` |
| `FieldTimeMonitor` | Fields in a box at regular time intervals | `res.time_series(name, component)` |
| `EnergyMonitor` | Electromagnetic energy in a box versus time | `res.energy(name)` |
| `FarFieldMonitor` | Radiated intensity and far-field amplitudes versus direction | `res.far_field(name)` |

Frequency-domain monitors accumulate a discrete Fourier transform (DFT)
while the simulation runs. Their results scale with the spectrum of the
source pulse and are normally divided by a reference (see
[Sources](sources.md)).

> **Note:** Field monitors, in the frequency and in the time domain, return
> the magnetic field multiplied by the impedance of free space,
> $\tilde{\mathbf{H}} = \eta_0 \mathbf{H}$ with $\eta_0 = 376.73\ \Omega$, in
> the unit of the electric field. Divide by $\eta_0$ before using it as a
> magnetic field. A Poynting vector or a mode overlap computed from
> $\mathbf{E}$ and $\tilde{\mathbf{H}}$ without this factor is wrong by a
> factor of about 377. The section on DFT monitors gives the correct
> formula.

## Worked example: a Fabry-Perot slab

The example on this page is a slab of index 3.5 and thickness 1 µm in vacuum,
illuminated at normal incidence. Light bounces between its two faces, so it
is a simple resonator: it transmits fully at the frequencies
$f_m = m\,c/(2nd)$ and stores light for a while after the pulse has passed.
Four monitors look at it in different ways.

```python
import numpy as np
import lightsim as ls
from lightsim.resonance import ring_down
from lightsim.spec import um

n, d = 3.5, 1.0 * um
fsr = ls.C0 / (2 * n * d)                   # spacing of the resonances, 42.8 THz
f0 = 4 * fsr                                # centre the pulse on the 4th resonance
dx = 12.5e-9
freqs = np.linspace(2.6, 5.4, 113) * fsr
x0 = 2.5 * um                               # front face of the slab


def column(with_slab):
    sim = ls.Simulation(min=[0, 0, 0], max=[6 * um, 2 * dx, 2 * dx],
                        mesh=ls.UniformMesh(dx), run_time=400e-15, shutoff=1e-8)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if with_slab:
        sim.add_structure(ls.Box([x0, -1, -1], [x0 + d, 1, 1]), ls.Medium("slab", n=n))
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
    # power through the plane x = 5 um, behind the slab
    sim.add_monitor(ls.FluxMonitor("T", 0, 5 * um, [5 * um, -1, -1], [5 * um, 1, 1], freqs))
    # field along a line through the slab, at the centre frequency
    sim.add_monitor(ls.DFTMonitor("inside", [x0, 0, 0], [x0 + d, 0, 0], [f0], components=("Ey",)))
    # fields on the plane of the flux monitor: Ey on the plane, Hz on the layers either side
    sim.add_monitor(ls.DFTMonitor("plane", [5 * um - 0.6 * dx, 0, 0], [5 * um + 0.6 * dx, 2 * dx, 2 * dx],
                                  freqs, components=("Ey", "Hz")))
    # field at one point inside the slab, at every time step
    sim.add_monitor(ls.ProbeMonitor("cavity", [x0 + 0.3 * d, 0, 0], components=("Ey",)))
    return sim


sim = column(True)
reference = column(False).run(device="cpu")
res = sim.run(device="cpu")
print(res.stats["steps"], res.stats["stopped_early"])
```

```text
9200 True
```

## Power: flux monitors

`FluxMonitor(name, axis, position, min, max, freqs)` integrates the
time-averaged Poynting vector over a plane normal to `axis`,

$$
P(\omega) = \frac{1}{2} \operatorname{Re} \int \big( \mathbf{E} \times \mathbf{H}^* \big) \cdot \hat{\mathbf{a}} \; dA .
$$

The result is positive for power flowing towards `+axis`. The plane is moved
to the nearest grid node; `min` and `max` limit its extent and may be larger
than the region.

In the example, dividing by the reference run gives the transmission. It
reaches 1 on the resonances and falls to $((1-R)/(1+R))^2 = 0.279$ between
them, where $R = ((n-1)/(n+1))^2$ is the reflectance of one face:

```python
T = np.asarray(res.flux("T")) / np.asarray(reference.flux("T"))
on_resonance = [int(np.argmin(np.abs(freqs - m * fsr))) for m in (3, 4, 5)]
print(np.round(T[on_resonance], 4), round(float(T.min()), 4))
```

```text
[0.9998 0.9994 0.9983] 0.2801
```

`FluxBoxMonitor(name, min, max, freqs)` sums the outward flux through the six
faces of a box. It gives the power radiated by a source inside it, the power
absorbed inside it (as a negative value) or, around a total-field/
scattered-field source, the scattered power. The box is snapped to grid
nodes. A face that lies on an electric or magnetic wall is skipped.

## Fields: DFT monitors

`DFTMonitor(name, min, max, freqs, components=("Ex", "Ey", "Ez"), stride=None, apodization=None)`
records the complex field amplitudes at the listed frequencies, for every
sample of each component inside the box from `min` to `max`. A box with zero
size along one or more axes records a plane, a line or a point. If no sample
lies exactly in a zero-size direction, the nearest layer is taken.

`res.field(name, component)` returns a complex array indexed
`[frequency, x, y, z]`. In the example the line through the slab holds 81
samples of `Ey`:

```python
ey = res.field("inside", "Ey")
print(ey.shape)
standing = np.abs(ey[0, :, 0, 0])
print(int(np.sum((standing[1:-1] < standing[:-2]) & (standing[1:-1] < standing[2:]))))
```

```text
(1, 81, 1, 1)
4
```

At the fourth resonance the slab is four half-wavelengths thick. The field
amplitude inside is modulated with that period and has four minima.

### Power from recorded fields

With the fields as returned, the power through a plane with normal
$\hat{\mathbf{a}}$ is

$$
P = \frac{1}{2\eta_0} \operatorname{Re} \int \big( \mathbf{E} \times \tilde{\mathbf{H}}^* \big) \cdot \hat{\mathbf{a}} \; dA .
$$

The factor $1/\eta_0$ converts $\tilde{\mathbf{H}}$ back to amperes per
metre. Two details of the grid matter for an accurate result. The tangential
electric field sits on the plane, but the tangential magnetic field sits half
a cell before and after it, so the two layers are averaged. And the arrays
are in single precision: convert them to double precision before multiplying
fields, because products of small numbers lose accuracy otherwise.

The monitor `"plane"` of the example records `Ey` on the plane at 5 µm and
`Hz` on the two layers around it. For a wave travelling along x the integrand
is $E_y \tilde{H}_z^* - E_z \tilde{H}_y^*$, and the second term is zero here:

```python
eta0 = 376.730313412
ey_plane = res.field("plane", "Ey").astype(complex)        # [frequency, 1, 2, 2]
hz_layers = res.field("plane", "Hz").astype(complex)       # [frequency, 2, 2, 2]
hz_plane = hz_layers.mean(axis=1, keepdims=True)           # average of the two layers
power = np.real(np.sum(ey_plane * np.conj(hz_plane), axis=(1, 2, 3))) * dx * dx / (2 * eta0)
print(ey_plane.shape, hz_layers.shape)
print(bool(np.allclose(power, res.flux("T"), rtol=1e-5)))
```

```text
(113, 1, 2, 2) (113, 2, 2, 2)
True
```

The result equals what the flux monitor on the same plane returns. Use a
`FluxMonitor` when the power is all you need; it does this bookkeeping,
including non-uniform cells.

| Option | Meaning |
|---|---|
| `components` | Any of the six field components |
| `stride` | Accumulate the Fourier sums every `stride`-th step. `None` chooses it automatically (see below); an integer forces it, and must then leave the sampling rate well above twice the highest frequency in the fields |
| `apodization` | A dictionary with `start`, `end` and `width` in seconds. Samples before `start` and after `end` are ignored, with a smooth taper of the given `width` at each edge. Use it to exclude the source transient, or to window a run that is cut before the fields have decayed |

A DFT monitor stores two 4-byte numbers per sample, per component and per
frequency. A volume monitor with many frequencies can need more memory than
the fields themselves.

### How often the fields are sampled

A frequency-domain monitor adds the field, times a phase factor per
frequency, to a running sum. Doing this at every time step is rarely
necessary: the time step is set by the cell size and is typically 50 to 200
times shorter than the optical period, while the sums only need two samples
per period of the fastest component of the fields. All frequency-domain
monitors (`DFTMonitor`, `FluxMonitor`, `FluxBoxMonitor`, `ModeMonitor`, far
field) therefore sample every `stride` steps, with the largest stride that
satisfies

$$
\frac{1}{\text{stride}\cdot\Delta t} \;\ge\; f_{\text{monitor, max}} + f_{\text{source, max}},
\qquad f_{\text{source, max}} = f_0 + 7 f_{\text{width}} ,
$$

so that nothing the source emits can fold onto a monitor frequency. Once the
fields have decayed, the results agree with sampling every step to the
rounding of single precision (a few $10^{-6}$ to $10^{-5}$), and the monitors
cost correspondingly less. A run that ends while fields are still passing the
monitor is different: the two sums then differ by about half a sampling
interval of the last field (4 % in a run cut off in the middle of its pulse),
which is small against the error of cutting the record there. Two flux
boxes with 61 frequencies around a scatterer in an 11-million-cell grid with
a plane-wave (TFSF) source run at 18.3 Gcells/s on an RTX PRO 6000, against
20.7 Gcells/s without the monitors; with the sums taken at every step, as
earlier versions did, the same scene ran at 2.7 Gcells/s.

The stride is 1 when the bandwidth of the fields is not known: with a source
that is not a `GaussianPulse` (a continuous wave, a Ricker wavelet, custom
samples) and with nonlinear media, which create harmonics.
`Simulation(dft_stride=n)` sets the stride for all monitors of a simulation,
and `dft_stride=1` restores sampling at every step. The stride used is
reported as `res.raw["dft"][name]["stride"]`.

> **Note:** A spectrum from a DFT or flux monitor is only meaningful if the
> fields have decayed by the end of the run. If the run is cut while a
> resonator still rings, the spectrum shows ripples and widened lines. Run
> longer, set `shutoff`, apodize, or use the ring-down analysis below.

## Time series: probes and field snapshots

`ProbeMonitor(name, position, components=("Ez",))` records the listed
components at every time step, at the grid position nearest to `position`.

`FieldTimeMonitor(name, min, max, components=("Ex", "Ey", "Ez"), interval=1, start=0.0)`
records all samples in a box every `interval` steps, beginning at time
`start`. Use it for movies of the field.

`res.time_series(name, component)` returns the sample times and an array
indexed `[sample, x, y, z]`. The times are those of the electric field;
magnetic samples are half a time step earlier, and they are multiplied by
$\eta_0$ as in the frequency domain. Everything a time monitor records is
kept in memory until the end of the run, so keep boxes small or intervals
long.

## Resonances and quality factors

After the pulse has passed, the field in a resonator is a sum of decaying
oscillations, one per mode. `lightsim.resonance.ring_down` fits that model to
a probe signal (harmonic inversion) and returns the frequency and the quality
factor $Q$ of each mode. Unlike a Fourier transform, it does not need the run
to last until the field has decayed.

```python
R = ((n - 1) / (n + 1)) ** 2
for mode in ring_down(res, "cavity", "Ey", 2.6 * fsr, 5.4 * fsr, sim=sim):
    m = round(mode.freq / fsr)
    print(f"m = {m}:  f = {mode.freq / 1e12:.2f} THz (exact {m * fsr / 1e12:.2f}),"
          f"  Q = {mode.Q:.2f} (exact {np.pi * m / np.log(1 / R):.2f})")
```

```text
m = 3:  f = 128.41 THz (exact 128.48),  Q = 8.01 (exact 8.02)
m = 4:  f = 171.14 THz (exact 171.31),  Q = 10.68 (exact 10.69)
m = 5:  f = 213.80 THz (exact 214.14),  Q = 13.34 (exact 13.36)
```

The exact values are those of an ideal slab, $f_m = m\,c/(2nd)$ and
$Q_m = \pi m / \ln(1/R)$. The small frequency offset is the numerical
dispersion of the grid.

`ring_down(results, probe, component, fmin, fmax, sim=sim)` analyses the part
of the record after the source has switched off, which it works out from
`sim`. Each returned `Resonance` has `freq`, `Q`, `decay` (amplitude decay
rate in 1/s), `amplitude` and `error` (an estimate of the relative frequency
error). Keep the window `fmin` to `fmax` reasonably tight around the modes of
interest.

The record after the pulse must be long enough for the lines to be told
apart: about two beat periods of neighbouring lines, which for the resonances
of a ring is two round trips. When the record is too short, or nothing stands
out of the noise, the function returns few modes or none and issues a warning
that says which of the two happened.

The module also provides `harminv` for any uniformly sampled signal and
`lorentzian_fit` for an isolated line in a spectrum. The method, the options
and the accuracy are in the [resonance notes](../modules/resonance.md).

## Mode amplitudes

`ModeMonitor(name, axis, position, min, max, freqs, modes)` projects the
field on a plane onto waveguide modes. `modes` is a list of mode profiles
from the mode solver, computed on the same plane and window.
`res.mode_amplitudes(name, mode)` returns two complex arrays over frequency:
the amplitude $a_+$ of the mode travelling towards `+axis` and $a_-$ of the
mode travelling towards `-axis`,

$$
a_\pm = \frac{1}{4 P_m} \int \big( \mathbf{E} \times \mathbf{H}_m^* \pm \mathbf{E}_m^* \times \mathbf{H} \big) \cdot \hat{\mathbf{a}} \; dA ,
$$

where $P_m$ is the power of the profile, which is 1 for profiles from the
mode solver. $\mathbf{H}$ and $\mathbf{H}_m$ are magnetic fields in amperes
per metre: the monitor divides the simulated $\tilde{\mathbf{H}}$ by $\eta_0$
itself, and mode profiles are already in these units. $|a_\pm|^2$ is then
the power carried by the mode in that direction, in the same units as a flux
monitor. Ratios of amplitudes at two ports are S-parameters;
[Modes and S-parameters](modes-sparams.md) has the complete workflow.

On the grid the magnetic field of the plane is the mean of the two layers half
a cell before and after it, which for a wave of propagation constant $\beta$
is the field on the plane times $c = \cos(\beta \Delta / 2)$, $\Delta$
being the cell size along the axis. The monitor takes this factor into account
when it separates the two directions (with the $\beta$ of the profile), so a
wave travelling one way does not show up in the amplitude of the other, and it
scales the amplitudes so that $|a_+|^2 - |a_-|^2$ is the flux a `FluxMonitor`
gives on the same plane. The plain formula above would add $(1-c)/(1+c)$ of
each wave to the other: 2 % of its amplitude at 11 cells per guided
wavelength, 0.4 % at 24. What remains is 0.05 % and 0.001 % at those
resolutions when the profile was solved at the monitor's frequency. A profile
used at other frequencies has the impedance of its own frequency, which
leaves a few tenths of a per cent of each wave in the other at 3 % detuning;
give the monitor a profile that follows the band where that matters:

```python
mode = ls.solve_modes(sim, 0, x, pmin, pmax, f0, num_modes=1)[0]
profile = ls.profile_over_band(sim, 0, x, pmin, pmax, mode.profile, freqs)
sim.add_monitor(ModeMonitor("out", 0, x, pmin, pmax, freqs, [profile]))
```

`profile_over_band` solves the same mode at two or three more frequencies
across `freqs` and the monitor projects, at each of its frequencies, on the
profile interpolated between them. On a silicon wire read from 1.50 to
1.60 µm with a profile of 1.55 µm, the share of a wave that shows up in the
other direction at the ends of the band falls from 0.13 to 0.16 % to
0.002 % at 20 nm cells (0.3 to 0.4 % to 0.002 % in 2-D), the same as with
one solve per frequency. The S-parameter functions and
`invdes.SMatrixProblem` take `port_profiles="band"` for the same thing on
every port (the default, `"center"`, uses the profiles of the centre
frequency).

Points to know:

- A profile is computed at one frequency and used at all monitor
  frequencies, unless it comes from `profile_over_band`. The projection is
  slightly less accurate away from that frequency.
- If the number of profiles equals the number of frequencies (and is larger
  than one), the profiles are taken as **one mode with one profile per
  frequency**, not as separate modes.
- `lightsim.gaussian_mode` builds the profile of a free-space Gaussian beam
  for the same monitor, which measures the coupling of a field into that
  beam. It takes the waist, the focus position and the direction (`theta`,
  `phi`) of the beam, with the conventions of the `GaussianBeam` source.

### Many profiles on one plane

Monitors that record the same plane at the same frequencies share one set of
Fourier sums. A scan of the waist, position or angle of a collecting beam can
therefore be written as one `ModeMonitor` per profile, or as one monitor with
all profiles in `modes=[p1, p2, ...]` (read with
`res.mode_amplitudes(name, i)`; mind the rule above about a number of
profiles equal to the number of frequencies): the run costs the same as with
a single monitor.

Monitors on **different** planes each record their own fields. Their cost is
proportional to the number of planes and of frequencies, and is kept small by
the sampling stride described under "Fields: DFT monitors".

The fields can also be recorded once with a `DFTMonitor` and projected
afterwards, which lets you try profiles that you did not think of before the
run.

The example does this for a Gaussian beam of 1 µm waist that travels
along z. The helper `overlap` evaluates the formula above on the grid, for a
plane normal to z on a uniform mesh. It divides the recorded magnetic field
by $\eta_0$ and averages its two layers, as in the power calculation
earlier.

```python
lam1 = 1.0 * um
f1 = ls.C0 / lam1
h = lam1 / 20                                   # cell size
z_plane, half = 1.0 * um, 2.0 * um
window_min, window_max = [-half, -half, z_plane], [half, half, z_plane]

beam = ls.Simulation(min=[-2.5 * um, -2.5 * um, 0], max=[2.5 * um, 2.5 * um, 2.0 * um],
                     mesh=ls.UniformMesh(h), run_time=60e-15)
beam.add_source(ls.GaussianBeam(2, 0.5 * um, "+", "Ex", [0, 0, 0.5 * um], 1.0 * um,
                                ls.GaussianPulse(f1, 0.1 * f1)))
# tangential E on the plane, tangential H on the layers half a cell either side of it
beam.add_monitor(ls.DFTMonitor("plane", [-half, -half, z_plane - 0.6 * h], [half, half, z_plane + 0.6 * h],
                               [f1], components=("Ex", "Ey", "Hx", "Hy")))


def collecting_beam(waist):
    # profile and power of a Gaussian beam focused 0.5 um before the plane
    return ls.gaussian_mode(beam, 2, z_plane, window_min, window_max, f1, waist,
                            center=[0, 0, z_plane], polarization="u", focus_distance=-0.5 * um)


matched, _ = collecting_beam(1.0 * um)
beam.add_monitor(ls.ModeMonitor("check", 2, z_plane, window_min, window_max, [f1], [matched]))
out = beam.run(device="cpu")

Ex, Ey = (out.field("plane", c).astype(complex)[0, :, :, 0] for c in ("Ex", "Ey"))
Hx, Hy = (out.field("plane", c).astype(complex)[0].mean(axis=2) / eta0 for c in ("Hx", "Hy"))
flux = 0.5 * np.real(np.sum(Ex * np.conj(Hy)) - np.sum(Ey * np.conj(Hx))) * h * h


def overlap(profile):
    # forward amplitude of a mode profile in the recorded field
    nu, nv = profile["dims"]

    def part(key, i, like):
        # the profile covers the cells of the window; the field array may hold one more node
        a = (np.array(profile[key + "_re"][i]) + 1j * np.array(profile[key + "_im"][i])).reshape(nu, nv)
        out = np.zeros(like.shape, dtype=complex)
        n0, n1 = min(nu, like.shape[0]), min(nv, like.shape[1])
        out[:n0, :n1] = a[:n0, :n1]
        return out

    # the profile holds H half a cell before the plane: advance its phase to the plane
    advance = np.exp(1j * 2 * np.pi * f1 * profile["neff"] / ls.C0 * h / 2)
    emx, hmy = part("e", 0, Ex), part("h", 1, Ex) * advance     # at the positions of Ex and Hy
    emy, hmx = part("e", 1, Ey), part("h", 0, Ey) * advance     # at the positions of Ey and Hx
    i1 = np.sum(Ex * np.conj(hmy)) - np.sum(Ey * np.conj(hmx))
    i2 = np.sum(np.conj(emx) * Hy) - np.sum(np.conj(emy) * Hx)
    pm = 0.5 * np.real(np.sum(emx * np.conj(hmy)) - np.sum(emy * np.conj(hmx)))
    return (i1 + i2) / (4 * pm)


print(abs(overlap(matched) / out.mode_amplitudes("check")[0][0] - 1) < 1e-5)
for waist in (0.6, 0.8, 1.0, 1.2, 1.4):
    profile, power = collecting_beam(waist * um)
    print(waist, round(abs(overlap(profile)) ** 2 * power / flux, 3))
```

```text
True
0.6 0.78
0.8 0.953
1.0 1.0
1.2 0.969
1.4 0.901
```

The first line shows that the projection computed from the recorded fields
equals the amplitude of the mode monitor. The other lines are the fraction
of the beam power that couples into a collecting beam of each waist. It
peaks at the waist of the launched beam and follows the textbook overlap of
two Gaussian beams of waists $w_1$ and $w_2$, $\big(2 w_1 w_2 / (w_1^2 + w_2^2)\big)^2$.

## Far field

`FarFieldMonitor(name, min, max, freqs, theta, phi, n_medium=1.0)` converts
the fields on a closed box into equivalent surface currents and radiates them
into a homogeneous medium. `theta` and `phi` are lists of angles in radians:
$\theta$ is measured from the z axis and $\varphi$ from the x axis in the xy
plane. `res.far_field(name)` returns a dictionary of arrays:

| Key | Content |
|---|---|
| `freqs`, `theta`, `phi` | The axes |
| `intensity` | Radiant intensity $U(\theta, \varphi)$, the power per unit solid angle in W/sr, indexed `[frequency, theta, phi]` |
| `e_theta`, `e_phi` | Complex far-field amplitudes in volts, same shape: far from the box the electric field is $(e_\theta\,\hat{\boldsymbol\theta} + e_\varphi\,\hat{\boldsymbol\varphi})\,e^{ikr}/r$, with $r$ measured from the coordinate origin |
| `flux` | Power through the box per frequency, for comparison with the integral of $U$ |

The intensity follows from the amplitudes as
$U = (|e_\theta|^2 + |e_\varphi|^2) / 2\eta$, with $\eta = \eta_0 / n$ the wave
impedance of the outer medium.

```python
lam0 = 1.0 * um
f1 = ls.C0 / lam0
theta = np.linspace(0, np.pi, 37)

dipole = ls.Simulation(min=[-0.6 * um] * 3, max=[0.6 * um] * 3, mesh=ls.UniformMesh(lam0 / 20), run_time=40e-15)
dipole.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(f1, 0.3 * f1)))
dipole.add_monitor(ls.FarFieldMonitor("pattern", [-0.4 * um] * 3, [0.4 * um] * 3, [f1], theta, [0.0]))
far = dipole.run(device="cpu").far_field("pattern")

U = far["intensity"][0, :, 0]
print(np.round(U[[0, 9, 18]] / U.max(), 3))                  # 0, 45 and 90 degrees from the dipole axis
total = 2 * np.pi * np.sum(0.5 * (U[1:] * np.sin(theta[1:]) + U[:-1] * np.sin(theta[:-1])) * np.diff(theta))
print(round(float(total / far["flux"][0]), 3))               # integral of U over the sphere / power through the box
print(bool(np.abs(far["e_phi"]).max() < 0.01 * np.abs(far["e_theta"]).max()))
```

```text
[0.    0.495 1.   ]
1.008
True
```

The pattern of a dipole is $\sin^2\theta$, which is 0.5 at 45°. Its integral
over all directions should equal the power through the box; here the two
agree within 1 % at 20 cells per wavelength. The field of a dipole along z
is polarised along $\hat{\boldsymbol\theta}$, so $e_\varphi$ vanishes.

Limits: the medium outside the box must be homogeneous, with index
`n_medium`. Substrates and other layered surroundings are not accounted for.
The box must be closed and lie inside the region.

## Diffraction orders

For periodic structures, `lightsim.gratings` decomposes the field on a plane
into diffraction orders. It takes a `DFTMonitor` that spans one unit cell and
records the four tangential field components, and returns the direction and
the power of each order, separately for the two directions of travel and for
the two polarisations. Transmission and reflection efficiencies follow from
two such planes and a reference run. The functions, a complete example and
the validation against rigorous coupled-wave analysis are in the
[grating notes](../modules/gratings.md).

## Stored energy

`EnergyMonitor(name, min, max, interval=1)` records the electromagnetic
energy in a box every `interval` steps,

$$
U(t) = \frac{\varepsilon_0}{2} \sum_{\text{cells}} \Big( \varepsilon\,|\mathbf{E}|^2 + \mu\,|\tilde{\mathbf{H}}|^2 \Big)\, \Delta V ,
$$

in joules, with the permittivity of each cell as the solver uses it and
$\tilde{\mathbf{H}} = \eta_0 \mathbf{H}$. A cell that the box cuts counts
with the part inside the box. `res.energy(name)` returns the sample times and
the energy.

The example sends a pulse of unit amplitude down a column of vacuum. While
the whole pulse is inside the box, half of its energy is electric and half
magnetic, and the total is $\varepsilon_0 A c \int s(t)^2\,dt$ for a
cross-section $A$ and a signal $s(t)$:

```python
lam2 = 1.0 * um
f2 = ls.C0 / lam2
cell = lam2 / 30
pulse = ls.Simulation(min=[0, 0, 0], max=[8 * um, 2 * cell, 2 * cell],
                      mesh=ls.UniformMesh(cell), run_time=60e-15)
pulse.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
pulse.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f2, 0.3 * f2)))
pulse.add_monitor(ls.EnergyMonitor("stored", [1.5 * um, -1, -1], [7.0 * um, 1, 1], interval=10))
t, U = pulse.run(device="cpu").energy("stored")

tau = 1 / (2 * np.pi * 0.3 * f2)                           # width of the Gaussian pulse
U_exact = 8.8541878128e-12 * (2 * cell) ** 2 * ls.C0 * tau * np.sqrt(np.pi) / 2
print(f"{U.max() / U_exact:.3f}", bool(U[-1] < 1e-9 * U.max()))
```

```text
0.996 True
```

The peak is within 0.4 % of the formula, and nothing is left after the pulse
has gone into the absorbing boundary.

Points to know:

- The electric field is sampled at the end of a step and the magnetic field
  half a step earlier. The sum therefore ripples at twice the optical
  frequency, by a fraction of about $(\omega\,\Delta t/2)^2$.
- For a dispersive medium $\varepsilon$ is the value at infinite frequency:
  the energy stored in the polarisation of the poles is not included.
- All six field components of the box are kept in memory for every sample
  until the end of the run, 24 bytes per cell and sample. Keep the box small
  or the interval long.
- For a decay rate or a quality factor, a probe and `ring_down` are simpler.
