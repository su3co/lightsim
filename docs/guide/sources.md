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

`PlaneWave(axis, position, direction, polarization, signal, amplitude=1.0, angle=0.0, min=None, max=None, name="")`

| Parameter | Meaning |
|---|---|
| `axis`, `position` | The source plane is normal to `axis` at this coordinate |
| `direction` | `"+"` or `"-"` along `axis` |
| `polarization` | Electric component in the plane, for example `"Ey"` for `axis=0` |
| `amplitude` | Peak electric field in V/m, multiplied by the signal |
| `angle` | Tilt of the wavevector in radians, see below |
| `min`, `max` | Corners limiting the extent of the plane; the whole plane by default |

The wave fills the plane, so the side boundaries decide what it is. With
periodic boundaries it is an infinite plane wave. With absorbing side
boundaries the edges of the plane diffract.

The index of the medium is taken at the centre of the plane and applied over
the whole plane. Put the plane in a homogeneous region. A dispersive medium at
the plane is handled exactly at normal incidence.

With `angle` set, the wavevector is tilted by that angle towards the in-plane
axis **perpendicular to the polarisation**, so the wave is s-polarised (the
electric field is normal to the plane of incidence). The side boundaries
along that axis must be Bloch-periodic with $k = n\,k_0 \sin\theta$, where
$k_0 = 2\pi f_0 / c$ (see [Boundaries](boundaries.md)). Points to know:

- There is no p-polarised oblique plane wave.
- The tilt is imposed as a phase gradient computed at `freq0`. The transverse
  wavevector is the same at every frequency, so the angle is exact at `freq0`
  and changes with frequency as $\sin\theta(f) = \sin\theta\, f_0 / f$. Use a
  narrow pulse.
- Such a run uses complex fields. `Results.source_spectrum` accounts for
  that, so fields and powers are normalised as at normal incidence. See the
  normalisation section below.

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
