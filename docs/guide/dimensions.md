# 2-D, 2.5-D and 3-D

After this page you can decide whether a problem needs a three-dimensional
simulation, a two-dimensional one or the 2.5-D effective-index model, set up
a 2-D simulation with the right polarization, and estimate what each choice
costs.

## The three models

| | 3-D | 2-D | 2.5-D |
|---|---|---|---|
| What is solved | Maxwell's equations on a 3-D grid, six field components | Maxwell's equations for fields that do not depend on one coordinate: two independent sets of three components | A 2-D simulation of a planar circuit seen from above, each region of the layout replaced by the effective index of its vertical layer stack |
| Exact for | everything the grid resolves | structures that are uniform along one axis, illuminated in the plane normal to it | nothing; an approximation of a 3-D planar device |
| Typical use | final verification, anything with out-of-plane scattering | gratings and multilayers in cross-section, cylinders, slab waveguides, photonic-crystal slabs in the 2-D approximation, method studies | layout-level design of couplers, splitters, interferometers and resonators in a fully etched layer; inverse design |
| Cells at $N$ cells per wavelength over $L$ wavelengths | $(NL)^3$ | $(NL)^2$ | $(NL)^2$ |
| How to ask | `Simulation(min=[x, y, z], ...)` | `Simulation(min=[u, v], ...)` or `dimensions=2` | `lightsim.eim`, `lightsim.invdes.planar_simulation` |

A 2-D simulation and a 2.5-D simulation run on the same solver. The
difference is what the materials mean: in 2-D they are the materials of a
structure that really is uniform along the third axis; in 2.5-D they are
effective indices that stand for the vertical confinement of a slab.

Lumerical offers the same three: FDTD in 3-D, FDTD in 2-D with TE or TM
polarization, and varFDTD ("2.5D FDTD") in MODE. lightsim's 2.5-D model is
the plain effective-index method, not the variational reduction of varFDTD
(see [Effective index](../modules/eim.md)).

## A 2-D simulation

A simulation is two-dimensional when `min` and `max` have two components, or
when `dimensions=2` is passed with three-component corners. The fields then
do not depend on the coordinate along `invariant_axis` (`"z"` unless you
choose `"x"` or `"y"`), and the two components of every point are the
coordinates along the other two axes, in the order x, y, z.

The example is a gap between two slab waveguides, 400 nm wide with index 2 in
air, seen in the plane of the page. The mode of the left guide is launched
towards the gap, and the part that continues in the right guide is measured.

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

f0 = ls.C0 / (1.55 * um)


def gap(polarization, **options):
    sim = ls.Simulation(min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh(25e-9),
                        run_time=80e-15, polarization=polarization, **options)
    sim.add_medium(ls.Medium("core", n=2.0))
    sim.add_structure(ls.Box([-10 * um, -0.2 * um], [-0.3 * um, 0.2 * um]), "core")
    sim.add_structure(ls.Box([0.3 * um, -0.2 * um], [10 * um, 0.2 * um]), "core")
    line = lambda u: ([u, -0.9 * um], [u, 0.9 * um])          # a port: a line across the guide
    mode = ls.solve_modes(sim, 0, -1.5 * um, *line(-1.5 * um), f0, num_modes=1)[0]
    sim.add_source(ls.ModeSource(0, -1.5 * um, "+", *line(-1.5 * um), mode.profile,
                                 ls.GaussianPulse(f0, 0.1 * f0)))
    sim.add_monitor(ls.ModeMonitor("in", 0, -1.2 * um, *line(-1.2 * um), [f0], [mode.profile]))
    sim.add_monitor(ls.ModeMonitor("out", 0, 1.5 * um, *line(1.5 * um), [f0], [mode.profile]))
    return sim, mode


sim, mode = gap("te")
print([len(a) - 1 for a in sim.grid()])     # cells along x, y, z (12 PML cells per in-plane face)
print(sim.time_info())
```

```text
[184, 104, 1]
(5.837669483455412e-17, 1371)
```

What `dimensions=2` changes, compared with a 3-D simulation of the same
region:

- The grid has **one cell** along the invariant axis, whatever the mesh says,
  and the boundaries along it are periodic, whatever `boundaries` says.
- The time step follows the 2-D stability limit,
  $\Delta t = S / \bigl(c \sqrt{1/\Delta u_{\min}^2 + 1/\Delta v_{\min}^2}\bigr)$:
  22 % longer than in 3-D for cubic cells.
- Only the field components of the chosen polarization are advanced: three
  instead of six.
- Structures are **cut by the plane** through the middle of the region along
  the invariant axis. A sphere becomes a disc, a box that the plane misses is
  absent, and a 3-D layout can be reused as it is. Shapes given with two
  components span the invariant axis: `Box([u0, v0], [u1, v1])` is a
  rectangle, `Cylinder([u, v], radius)` a disc, `PolySlab(vertices)` a
  polygon.
- Sources and monitors span the invariant axis; their extent and position
  along it are ignored. A port plane is a line, and `solve_modes` returns the
  modes of that line (computed by `lightsim.eim.solve_port_mode`).

## Polarization

The six field components split into two sets that do not interact in a 2-D
problem, and `polarization` selects which exist:

| `polarization` | Fields | Invariant axis z | y | x |
|---|---|---|---|---|
| `"te"` | E in the plane, H along the invariant axis | `Ex`, `Ey`, `Hz` | `Ex`, `Ez`, `Hy` | `Ey`, `Ez`, `Hx` |
| `"tm"` | E along the invariant axis, H in the plane | `Ez`, `Hx`, `Hy` | `Ey`, `Hx`, `Hz` | `Ex`, `Hy`, `Hz` |
| `"both"` (default) | the two sets, independent | all six | all six | all six |

This is the convention of Lumerical's 2-D FDTD (TE: `Ex`, `Ey`, `Hz`; TM:
`Ez`, `Hx`, `Hy` in the xy plane), of Meep, and of the TE$_z$ / TM$_z$ of the
FDTD literature: "transverse" refers to the invariant axis.

> **Note:** The names say nothing about waveguide modes. Seen from above
> (invariant axis z, the wafer normal), the quasi-TE mode of a photonic
> waveguide has its electric field in the plane and belongs to `"te"`. Seen
> in cross-section (invariant axis along the grooves of a grating, say y),
> the TE mode of the slab has its electric field along y and belongs to
> `"tm"`.

With `"both"` the solver updates only the sets that a source drives, so a
simulation excited in one polarization costs the same as one restricted to
it; the restriction matters when a source would drive the other set. A
source that drives only a set the simulation does not contain is an error.
Components of an absent set read as zero in monitors.

```python
for polarization in ("te", "tm"):
    sim, mode = gap(polarization)
    res = sim.run(device="cpu")
    t = abs(res.mode_amplitudes("out")[0][0] / res.mode_amplitudes("in")[0][0]) ** 2
    print(polarization, round(mode.neff.real, 4), round(float(t), 3), res.stats["backend"])
```

```text
te 1.3595 0.666 cpu (10 threads, 2-D te)
tm 1.6736 0.451 cpu (10 threads, 2-D tm)
```

The two polarizations have different modes and cross the gap differently.
The backend name ends in `2-D` and the sets that were updated.

An oblique plane wave stays in the plane: `PlaneWave(..., angle=theta)` with
an in-plane electric polarization is p-polarized (the electric field keeps
the amplitude `amplitude` and tilts with the wavevector), and with the
electric field along the invariant axis it is s-polarized. Use a Bloch
boundary with $k = k_0 n \sin\theta$ along the transverse in-plane axis, as
in 3-D (see [Boundaries](boundaries.md)).

## Thickness and units

A 2-D simulation stands for a slab of the structure: the region's extent
along the invariant axis. With two-component corners that extent is
`thickness`, 1 m by default, centred on zero. The fields do not depend on
it. Quantities integrated over the slab scale with it:

- the power of `flux` is that through the slab, so with the default it is in
  watts per metre of the invariant direction;
- a `PointDipole` is a line source whose moment `amplitude` is spread over
  the thickness;
- mode profiles are normalised to unit power over the thickness.

Ratios such as transmissions, reflections and S-parameters do not depend on
the thickness. The example sends a plane wave of unit amplitude through a
line 1 µm long and compares the flux with the intensity of the wave:

```python
def plane_wave_flux(thickness):
    sim = ls.Simulation(min=[0, 0], max=[2 * um, 1 * um], mesh=ls.UniformMesh(25e-9), run_time=40e-15,
                        polarization="tm", thickness=thickness)
    sim.set_boundaries(y=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.5 * um, "+", "Ez", ls.GaussianPulse(f0, 0.3 * f0), name="wave"))
    sim.add_monitor(ls.FluxMonitor("flux", 0, 1.5 * um, [1.5 * um, 0], [1.5 * um, 1 * um], [f0]))
    res = sim.run(device="cpu")
    _, spectrum = res.source_spectrum("wave")
    intensity = abs(spectrum[0]) ** 2 / (2 * 376.73)         # power per area of the wave
    return res.flux("flux")[0] / (intensity * 1 * um)        # = the thickness of the slab


print(round(plane_wave_flux(1.0), 3), f"{plane_wave_flux(0.5e-6):.3e}")
```

```text
0.999 4.994e-07
```

The flux is the intensity times the length of the line times the thickness
of the slab, to the accuracy of the grid: per metre with the default.

## What runs on the 2-D path

Dielectric, conductive, dispersive and nonlinear ($\chi^{(2)}$, $\chi^{(3)}$)
media, sub-pixel smoothing, CPML, periodic and Bloch boundaries, PEC and PMC
walls, every source and every monitor except the far-field projection, which
is three-dimensional. The results equal those of the same grid run through
the general 3-D update to rounding, and exactly on the CPU; the validation
suite checks this on 54 scenes (see [Validation](../validation.md)).

A one-cell-thick 3-D simulation with periodic boundaries along the thin
axis, the way 2-D problems were set up before `dimensions=2`, is recognised
and runs on the same path with the same results as before.

Some features make a 2-D problem run through the general 3-D update instead,
on the same one-cell grid and with the same results. This happens without a
message; the backend name then lacks `2-D`:

| Feature | CPU | GPU |
|---|---|---|
| Media with off-diagonal permittivity (they couple the two sets, so they need `polarization="both"`) | 3-D update | 3-D update |
| Magnetic media (`mu` different from 1) | 2-D | 3-D split kernels |
| A source that injects magnetic field inside or next to an absorbing layer | 2-D | 3-D split kernels |
| Reverse-mode differentiation through the time loop (`lightsim.autodiff.ReverseSession`) | 2-D | forward steps 2-D; the time reversal and, with absorbing layers, the adjoint step use the 3-D split kernels |
| An absorbing layer along x cut by the boundary between two GPUs | | 3-D split kernels |
| Several GPUs when x is the invariant axis | | runs on the first GPU |
| Split GPU kernels requested (`LIGHTSIM_FUSED=0`): the 2-D step is part of the fused mode | | 3-D split kernels |

With `LIGHTSIM_VERBOSE=1` in the environment the GPU engine prints why it
left the 2-D path.

`Simulation(..., force_3d_kernels=True)` (or `LIGHTSIM_2D=0` in the
environment) sends a 2-D problem through the 3-D update on purpose, to
cross-check a result.

## 2.5-D: a planar circuit as a 2-D problem

A photonic circuit in a thin layer confines light vertically in one slab
mode almost everywhere. The effective-index method replaces every region of
the layout by a uniform medium whose index is the effective index of that
slab mode and solves the resulting 2-D problem, with the polarization of the
slab mode family: `"te"` for the modes with the electric field in the plane
of the chip.

`lightsim.eim.write_sparameters_2d` does this for a gdsfactory component, and
`lightsim.invdes.planar_simulation` builds the empty model for inverse
design; both return 2-D simulations as described above. What the model
leaves out is listed in the [effective-index notes](../modules/eim.md): light
scattered out of the plane, the wavelength dependence of the effective
index, and any change of the vertical mode shape between regions.

## One device, three models

The 1x2 multimode-interference splitter of gdsfactory's generic PDK, 220 nm
of silicon in oxide, simulated three ways at 18 cells per wavelength: in 3-D,
in 2.5-D, and as a plain 2-D simulation of the layout with the bulk indices
of silicon and oxide in place of the effective indices.

```python
import gdsfactory as gf
from lightsim import eim

gf.gpdk.PDK.activate()
mmi = gf.components.mmi1x2()
wavelengths = np.linspace(1.5e-6, 1.6e-6, 11)
common = dict(wavelengths=wavelengths, steps_per_wavelength=18, input_ports=["o1"], return_results=True)


def report(name, sp, runs):
    stats = runs["o1"].stats
    split = np.abs(sp["o2@0,o1@0"][[0, 5, 10]]) ** 2
    print(name, np.round(split, 3), round(float(np.real(sp["mode_neff"]["o1"])), 3), stats["cells"], stats["steps"])


report("2.5-D", *eim.write_sparameters_2d(mmi, device="cpu", **common))
report("2-D  ", *eim.write_sparameters_2d(mmi, device="cpu", level_indices={"core": 3.476},
                                          background_index=1.444, **common))
```

```text
2.5-D [0.493 0.496 0.493] 2.497 132628 7000
2-D   [0.385 0.409 0.433] 3.159 183708 12000
```

```python
from lightsim import sparams

report("3-D  ", *sparams.write_sparameters(mmi, device="cuda:0", **common))
```

```text
3-D   [0.493 0.495 0.484] 2.453 11389896 12200
```

| | 3-D | 2.5-D | 2-D with bulk indices |
|---|---|---|---|
| Power in one output at 1.50 / 1.55 / 1.60 µm | 0.493 / 0.495 / 0.484 | 0.493 / 0.496 / 0.493 | 0.385 / 0.409 / 0.433 |
| Effective index of the port mode | 2.453 | 2.497 | 3.159 |
| Cells | 11 389 896 | 132 628 | 183 708 |
| Time steps | 12 200 | 7 000 | 12 000 |
| Time per run, one RTX PRO 6000 | 8.5 s | 0.20 s | 0.25 s |
| Time per run, 64-core CPU (Threadripper 9980X) | | 3.0 s | 5.1 s |

The 2.5-D model reproduces the 3-D split within 0.2 % at the centre
wavelength and within 2 % over the band, at about one eighty-sixth of the
cells and in one fortieth of the time on the same GPU (times include the
mode solves and the set-up; one input port, timed after a warm-up run). The plain 2-D model is a different device: without the vertical
confinement the silicon guides far more strongly (port index 3.16 instead of
2.45), the self-imaging length of the multimode section changes, and a
splitter designed for the real stack no longer splits evenly. A 2-D
simulation with bulk indices is the right model for a structure that is
uniform along the third axis, not for a slab circuit.

## What each costs

The cost of a run is the number of cells times the number of steps divided
by the throughput.

- **Cells.** At equal resolution a 2-D grid has as many cells as one layer
  of the 3-D grid. The splitter above has 86 times as many cells in 3-D.
- **Steps.** The 2-D time step is longer by $\sqrt{3/2}$ for cubic cells,
  so the same simulated time takes 18 % fewer steps.
- **Work per cell.** One polarization updates three components instead of
  six.
- **Throughput.** Small grids do not load a processor fully; the
  [Benchmarks](../benchmarks.md) page gives measured figures for 2-D grids of
  $10^4$ to $10^8$ cells on a CPU and on a GPU.

Refining a 2-D mesh by a factor of two costs eight times more (four in cells,
two in steps); in 3-D it costs sixteen.

## Which to use

- The structure is uniform along one axis and the light travels in the plane
  normal to it: **2-D**, exact, with the polarization of the excitation.
- A planar circuit in a fully etched layer, at the design stage: **2.5-D**,
  then verify in 3-D.
- Anything that sends light out of the plane (grating couplers, vertical
  tapers, edge couplers, scattering loss), sidewall angles, polarization
  conversion, or a final number: **3-D**.
- Inverse design of planar devices: optimise in **2.5-D**, where an
  iteration takes seconds, and re-optimise or verify the result in 3-D (see
  [Inverse design](inverse-design.md)).
