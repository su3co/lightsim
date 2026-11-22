# Concepts

After this page you know how lightsim lays fields out on its grid, which units
and sign conventions it uses, what a simulation is made of, when a run stops
and how to read the results.

## The method

lightsim solves Maxwell's equations in the time domain with the
finite-difference time-domain (FDTD) method. Space is divided into box-shaped
cells, the fields are stored at fixed positions in each cell, and the solver
advances them in small time steps. A short pulse excites a whole band of
frequencies in one run. Monitors accumulate a Fourier transform of the fields
while the run proceeds, so the frequency response is available at the end
without storing the time history.

## The Yee grid

The grid is rectilinear: it is defined by the node coordinates along x, y and
z, and cell $i$ along an axis spans the interval between node $i$ and node
$i+1$. The cells need not have the same size (see
[Simulation setup](simulation.md)).

The six field components are not stored at the same point. Each sits at its
own position in the cell, the arrangement introduced by Yee, which makes the
discrete curl operators second-order accurate:

| Component | x | y | z |
|---|---|---|---|
| `Ex` | centre | node | node |
| `Ey` | node | centre | node |
| `Ez` | node | node | centre |
| `Hx` | node | centre | centre |
| `Hy` | centre | node | centre |
| `Hz` | centre | centre | node |

"Node" is a cell edge and "centre" is half way between two nodes. An electric
component is at the centre along its own axis and on nodes along the other
two; a magnetic component is the opposite.

Time is staggered as well. After step $n$ the electric field is known at time
$(n+1)\,\Delta t$ and the magnetic field half a step earlier.

Two consequences matter in practice:

- A plane given by a coordinate (a source plane, a flux monitor) is moved to
  the nearest grid node along its normal.
- A field monitor returns each component at that component's own positions.
  Arrays of different components can differ in size by one sample per axis.

Axes are numbered 0, 1, 2 for x, y, z. Field components are named `"Ex"`,
`"Ey"`, `"Ez"`, `"Hx"`, `"Hy"`, `"Hz"`. A direction along an axis is `"+"` or
`"-"`.

## Units and conventions

All inputs and outputs are in SI units.

| Quantity | Unit |
|---|---|
| Length, coordinates | metre |
| Time | second |
| Frequency | hertz |
| Angular frequency (pole parameters) | radian per second |
| Wavevector (Bloch boundaries) | radian per metre |
| Angle | radian |
| Conductivity | siemens per metre |
| Electric field | volt per metre |

`lightsim.spec` provides `um = 1e-6`, `nm = 1e-9` and the speed of light `C0`.
The permittivity $\varepsilon$ and permeability $\mu$ of media are relative
values.

Fields vary in time as $e^{-i\omega t}$. A wave travelling towards $+x$ is
therefore $e^{+ikx}$, and an absorbing medium has
$\operatorname{Im}\varepsilon > 0$.

> **Note:** Magnetic fields in result arrays are multiplied by the impedance
> of free space, $\tilde{H} = \eta_0 H$ with $\eta_0 = 376.73\ \Omega$, so
> that $E$ and $\tilde{H}$ have the same unit and equal magnitude in a vacuum
> plane wave. Divide by $\eta_0$ to obtain amperes per metre; a power or an
> overlap computed without this step is wrong by a factor of about 377 (see
> [Monitors](monitors.md) for the formula). Mode profiles are the exception:
> their `H` is in amperes per metre.

A frequency-domain monitor returns the Fourier transform of the field over the
run,

$$
F(\omega) = \sum_n F(t_n)\, e^{i \omega t_n}\, \Delta t ,
$$

where $t_n$ is the time at which that component is sampled. It is the
response to the pulse, so it scales with the pulse spectrum. Power fluxes are
computed from these transforms and scale with the squared pulse spectrum. To
obtain a response per unit excitation, divide by the source spectrum or by
the result of a reference run, as in the [Quick start](quickstart.md).

## What a simulation consists of

A `Simulation` holds everything the solver needs:

| Field | Meaning | Default |
|---|---|---|
| `min`, `max` | Corners of the simulation region | required |
| `mesh` | How the region is divided into cells | required |
| `run_time` | Simulated time | required |
| `boundaries` | Boundary condition on each of the six faces | absorbing (PML) on all faces |
| `background` | Name of the medium that fills the region | `"vacuum"` |
| `media` | Media defined by the user | none |
| `structures` | Shapes filled with a medium, later ones on top | none |
| `sources` | What excites the fields | none |
| `monitors` | What is recorded | none |
| `shutoff` | Stop early at this energy fraction | off |
| `shutoff_interval` | Steps between energy checks | 200 |
| `dft_stride` | Steps between samples of the frequency-domain monitors (see [Monitors](monitors.md)) | automatic |
| `courant` | Time step as a fraction of the stability limit | 0.99 |
| `subpixel` | Smooth dielectric interfaces inside cells | on |
| `dimensions` | 3, or 2 for fields that do not depend on one coordinate | 3 |
| `invariant_axis`, `polarization`, `thickness` | The axis, the field set and the slab thickness of a 2-D simulation | `"z"`, `"both"`, 1 m |

Two-component `min` and `max` make the simulation two-dimensional; the page
[2-D, 2.5-D and 3-D](dimensions.md) explains when to use one and what
changes.

Media, shapes, sources and monitors are created with builder functions such
as `Medium`, `Box`, `PlaneWave` and `FluxMonitor`. The builders return plain
dictionaries. `Simulation.to_json()` serialises the whole description; the
same JSON is what the command-line program reads (see
[Command line](cli.md)).

The example for the rest of this page is a dipole in an empty box, with one
frequency-domain monitor on a plane and one time-domain probe:

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

f0 = ls.C0 / (1.0 * um)
sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 1 * um, 1 * um],
                    mesh=ls.UniformMesh(50e-9), run_time=30e-15)
sim.add_source(ls.PointDipole([1 * um, 0.5 * um, 0.5 * um], "Ez",
                              ls.GaussianPulse(f0, 0.3 * f0), name="dipole"))
sim.add_monitor(ls.DFTMonitor("slice", [0, 0, 0.5 * um], [2 * um, 1 * um, 0.5 * um],
                              [f0], components=("Ez", "Hy")))
sim.add_monitor(ls.ProbeMonitor("probe", [1.5 * um, 0.5 * um, 0.5 * um], components=("Ez",)))

x, y, z = sim.grid()                       # node coordinates along each axis
print(len(x) - 1, len(y) - 1, len(z) - 1)  # number of cells
print(x[0], x[12], x[-1])
```

```text
64 44 44
-6e-07 0.0 2.5999999999999992e-06
```

The region is 40 by 20 by 20 cells. The default boundary is a perfectly
matched layer (PML), an absorbing layer 12 cells thick that is added
**outside** the region on every face. The grid therefore has 64 by 44 by 44
cells and starts at $x = -0.6\ \mu\text{m}$.

## Time step and how a run ends

The time step follows from the stability (Courant) limit of the scheme,

$$
\Delta t = \frac{S}{c \sqrt{\dfrac{1}{\Delta x_{\min}^2} + \dfrac{1}{\Delta y_{\min}^2} + \dfrac{1}{\Delta z_{\min}^2}}},
$$

where $S$ is `courant` (0.99 by default), $c$ is the speed of light in vacuum
and $\Delta x_{\min}$ is the smallest cell along x anywhere in the grid. One
small cell sets the time step of the whole run. In a 2-D simulation the term
of the invariant axis is absent. A dispersive medium with strong poles can
require a smaller step than this; the solver then reduces it (see
[Materials](materials.md)). The number of steps is `run_time` divided by
the time step, rounded up. `time_info()` returns both without running
anything:

```python
dt, steps = sim.time_info()
print(dt, steps)
```

```text
9.532874347655018e-17 315
```

A run ends in one of two ways:

- It reaches `run_time`.
- If `shutoff` is set, it ends earlier once the fields have decayed. Every
  `shutoff_interval` steps the solver sums the squared fields over the grid.
  When the sources have finished and that sum has fallen below `shutoff`
  times the largest value seen so far, the run stops.

A Gaussian pulse counts as finished at twice its peak time. A continuous-wave
source never finishes, so `shutoff` has no effect with one.

At the same intervals, and at the last step, the solver checks that the
fields are finite and that their energy does not grow after the sources have
finished. A run that fails this check is numerically unstable. It stops with
an error (a `RuntimeError` in Python) and returns no results. The usual
causes are `courant` above 1 and a medium that is not passive.

Choose `run_time` long enough for the light to cross the device several times
and let `shutoff` cut it short. Resonant devices need longer than the transit
time suggests, because they release their energy slowly.

The shutoff test looks at the energy that is left anywhere in the grid, as a
fraction of its peak. After the main pulse has left a device, what remains
are echoes: light reflected at a bend or a facet that crosses the device once
more, at a level of $10^{-4}$ to $10^{-6}$ of the energy. A run stops only
when the last echo above the threshold has left. Two similar devices can
therefore stop a whole transit time apart, when an echo is just above the
threshold in one and just below in the other, and a tail that stays slightly
above the threshold keeps a run going until `run_time`. Both runs are
converged to the level that was asked for or better.

`res.decay()` returns the energy fraction at the end of the run, and
`res.stats["energy_steps"]` with `res.stats["energy"]` hold the whole history,
which shows the echoes as plateaus. A run that reaches `run_time` with the
energy still above `shutoff` returns its results with a warning that states
the level reached. Then increase `run_time`, or accept that level by raising
`shutoff`.

## What comes back

`run` returns a `Results` object:

```python
res = sim.run(device="cpu")
print(res.stats["steps"], res.stats["stopped_early"], res.stats["backend"])
```

```text
315 False cpu (10 threads)
```

| Accessor | Returns |
|---|---|
| `stats` | `backend`, `cells`, `steps`, `seconds`, `mcells_per_s` (million cell updates per second), `stopped_early`, and `energy_steps` with `energy`, the decay history that `shutoff` looks at |
| `dt` | Time step |
| `decay()` | Field energy at the end of the run as a fraction of its peak, the number `shutoff` is compared with |
| `field(monitor, component)` | Complex array `[frequency, x, y, z]` of a `DFTMonitor`; magnetic components are multiplied by $\eta_0$ |
| `freqs(monitor)` | Frequencies of a `DFTMonitor` |
| `flux(monitor)` | Power through a `FluxMonitor` or out of a `FluxBoxMonitor`, one value per frequency |
| `mode_amplitudes(monitor, mode=0)` | Forward and backward amplitudes of a `ModeMonitor` |
| `time_series(monitor, component)` | Sample times and array `[sample, x, y, z]` of a `ProbeMonitor` or `FieldTimeMonitor`; magnetic components are multiplied by $\eta_0$ |
| `source_spectrum(name, freqs=None, monitor=None)` | Frequencies and Fourier transform of a source's pulse, at the frequencies of a monitor or at given frequencies |
| `raw` | The underlying dictionary, with everything above and the array origins |

Field arrays hold only the samples inside the monitor. The origin stored next
to each array is the grid index of its first sample, counted on the full grid
including the PML cells. Together with `grid()` and the table of positions
above it gives the coordinates:

```python
ez = res.field("slice", "Ez")
i, j, k = res.raw["dft"]["slice"]["fields"]["Ez"]["origin"]
print(ez.shape, (i, j, k))

x_ez = x[i:i + ez.shape[1]]                                  # Ez is on nodes along x
z_ez = 0.5 * (z[k:k + ez.shape[3]] + z[k + 1:k + 1 + ez.shape[3]])   # and on centres along z
print(x_ez[0], x_ez[-1], z_ez)
print(res.field("slice", "Hy").shape)
```

```text
(1, 41, 21, 1) (12, 12, 22)
0.0 2e-06 [5.25e-07]
(1, 40, 21, 1)
```

The monitor was requested on the plane $z = 0.5\ \mu\text{m}$. `Ez` has no
samples exactly there, because along z it sits at cell centres, so the nearest
layer, at $0.525\ \mu\text{m}$, is returned. `Hy` has one sample fewer along x
than `Ez`, because along x it sits at cell centres.

Time-domain monitors return the sample times with the data. The electric
field is sampled at the end of each step:

```python
t, ez_t = res.time_series("probe", "Ez")
print(t.shape, ez_t.shape, t[0] == res.dt)
```

```text
(315,) (315, 1, 1, 1) True
```

The spectrum of a source pulse is the usual normalisation for a field or a
flux when no reference run is made. `source_spectrum(name, monitor="slice")`
returns it at the frequencies of that monitor, in the monitor's order, and
`source_spectrum(name, freqs=[...])` at any frequencies. Without either
argument the frequencies are those of the monitors, provided all of them
record the same list; otherwise the call raises an error and asks which
monitor is meant. A source without a name is called `source0`, `source1` and
so on, in the order the sources were added.

```python
freqs, spectrum = res.source_spectrum("dipole", monitor="slice")
print(freqs, f"{abs(spectrum[0]):.3e}")
```

```text
[2.99792458e+14] 2.218e-15
```

[Monitors](monitors.md) describes each monitor and its result in detail.
