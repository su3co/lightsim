# Modes and S-parameters

After this page you can solve the guided modes of a waveguide cross-section,
launch and detect a mode in a simulation, and compute the S-parameters of a
component, either by hand, from a gdsfactory layout in 3-D, or with the fast
2.5-D effective-index approximation.

## The mode solver

A waveguide mode is a field pattern that travels along the guide without
changing shape, as $e^{i\beta x}$. Its effective index is
$n_{\text{eff}} = \beta / k_0$, with $k_0 = 2\pi f / c$.

`solve_modes(sim, axis, position, min, max, freq, num_modes=2)` computes the
modes on a plane of a simulation. The solver is full-vectorial and works on
the same staggered grid as the simulation, with the same sub-pixel-smoothed
permittivity. A mode computed this way is an eigenmode of the discretised
waveguide, so it can be injected and detected without interpolation.

| Parameter | Meaning | Default |
|---|---|---|
| `sim` | The simulation that contains the waveguide | required |
| `axis`, `position` | The plane is normal to `axis` at this coordinate (moved to the nearest grid node) | required |
| `min`, `max` | Corners of the mode window on that plane | required |
| `freq` | Frequency in Hz | required |
| `num_modes` | Number of modes to compute | 2 |
| `target_neff` | Search near this effective index | the largest index in the window |
| `sort` | `"neff"` (highest effective index first) or `"te"` (TE-like modes first) | `"neff"` |

It returns a list of `Mode` objects:

| Attribute | Meaning |
|---|---|
| `neff` | Complex effective index. A positive imaginary part is loss |
| `te_fraction` | Share of the transverse electric field energy in the first in-plane axis |
| `profile` | The pattern in the form mode sources and mode monitors take |
| `E`, `H` | Tuples of the two transverse components, complex arrays over the window; `H` in A/m |
| `info` | The window: node coordinates and the permittivity maps the solver used |

The two in-plane axes are (y, z) for `axis=0`, (x, z) for `axis=1` and (x, y)
for `axis=2`. For a chip whose layers are stacked along z, a mode with
`te_fraction` near 1 is therefore TE-like (electric field mostly in the chip
plane) for waveguides along x or y, and a mode with `te_fraction` near 0 is
TM-like (electric field mostly perpendicular to the chip).

The example is a silicon strip waveguide, 500 by 220 nm, in silicon dioxide,
at a wavelength of 1.55 µm:

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

wavelength = 1.55 * um
f0 = ls.C0 / wavelength

sim = ls.Simulation(min=[0, -1.2 * um, -0.9 * um], max=[3 * um, 1.2 * um, 0.9 * um],
                    mesh=ls.AutoMesh(wavelength, steps_per_wavelength=14),
                    run_time=150e-15, background="sio2", shutoff=1e-5)
sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]), "si")

x_src = 0.5 * um
window_min, window_max = [x_src, -1.0 * um, -0.7 * um], [x_src, 1.0 * um, 0.7 * um]
modes = ls.solve_modes(sim, 0, x_src, window_min, window_max, f0, num_modes=2)
for mode in modes:
    print(round(mode.neff.real, 4), round(mode.te_fraction, 2))
print(modes[0].E[0].shape)
```

```text
2.4475 0.98
1.7736 0.04
(40, 26)
```

Each line gives the effective index and the TE fraction. The first mode is
the fundamental TE-like mode and the second the fundamental TM-like mode. The
field arrays cover the 40 by 26 cells of the window.

Points to know:

- The mode window has closed walls: the field is zero on its edge. Make the
  window large enough for the mode to decay before the edge, and keep it
  inside the simulation region.
- Profiles are normalised to unit power. Their overall phase is fixed by
  making the largest electric component real and positive.
- Lossy and dispersive media enter with their complex permittivity at `freq`,
  so absorbing waveguides give a complex effective index.
- On a simulation plane the solver sees the diagonal of the permittivity
  (the Rust core samples only the diagonal there) and no magnetic
  permeability. A [cross-section](../modules/mode-solver.md) built without
  a simulation takes full permittivity tensors.
- `pml=` adds perfectly matched layers to the window (leaky modes,
  substrate leakage), `bend_radius=` solves a bent guide, `group_index=True`
  adds the group index; see [Mode solver](../modules/mode-solver.md) for
  these options, the derived quantities of a `Mode` (effective area,
  confinement, overlaps, loss in dB/cm), sweeps and the gplugins-style
  functions.

## Launching and detecting a mode

A `ModeSource` launches a profile in one direction. A `ModeMonitor` projects
the field on a plane onto profiles and returns the amplitudes of the forward
and backward modes (see [Monitors](monitors.md)). Both take the window that
was given to the mode solver. A profile solved on one plane can be used on
any other plane where the waveguide cross-section and the transverse mesh are
the same.

A mode monitor takes a list of profiles. Monitors on the same plane with the
same frequencies share the fields they record, so several monitors on one
plane cost no more than one (see [Monitors](monitors.md)).

The example continues with a source near the left end and two monitors. The
first monitor sits two cells after the source and measures the amplitude that
was launched; the second sits 2 µm further down the guide.

```python
freqs = ls.C0 / np.linspace(1.5 * um, 1.6 * um, 5)
sim.add_source(ls.ModeSource(0, x_src, "+", window_min, window_max, modes[0].profile,
                             ls.GaussianPulse(f0, 0.1 * f0)))

x_nodes = sim.grid()[0]
x_in = x_nodes[np.argmin(np.abs(x_nodes - x_src)) + 2]         # two cells after the source
for name, x in (("in", x_in), ("out", 2.5 * um)):
    sim.add_monitor(ls.ModeMonitor(name, 0, x, [x, window_min[1], window_min[2]],
                                   [x, window_max[1], window_max[2]], freqs, [modes[0].profile]))

res = sim.run(device="cpu")
a_in, b_in = res.mode_amplitudes("in")          # forward and backward amplitude at the input
a_out, _ = res.mode_amplitudes("out")
print(np.round(np.abs(a_out / a_in) ** 2, 4))   # transmission
print(np.abs(b_in / a_in).max() ** 2 < 1e-3)    # reflection
```

```text
[0.9991 0.9997 1.     0.9996 0.9987]
True
```

The ratio `a_out / a_in` is the S-parameter $S_{21}$ of this length of
waveguide, a complex number per frequency. Its magnitude is 1 within 0.1 %.
Measuring the launched amplitude with a monitor, instead of taking it from
the source spectrum, makes the ratio independent of how exactly the source
couples to the grid.

The transmission is highest at the centre of the band because the profile
was solved at the centre frequency and is used at all five. Away from the
centre the true mode differs slightly from the profile.

## S-parameters of a gdsfactory component

`lightsim.sparams.write_sparameters(component, ...)` automates these steps
for a [gdsfactory](https://gdsfactory.github.io/gdsfactory/) component:

1. Every port is extended with a straight waveguide that runs into the
   absorbing boundary.
2. The polygons of the component are extruded according to the layer stack
   (see [gdsfactory](gdsfactory.md)).
3. The mode of every port is solved, at the centre frequency, on a plane just
   outside the device.
4. For each input port one simulation is run, with a mode source on that port
   and mode monitors on all ports.
5. $S_{ji} = a_{\text{out},j} / a_{\text{in},i}$.

The result is a dictionary. The key `"o2@0,o1@0"` holds $S_{21}$, the
amplitude leaving port `o2` per unit amplitude entering port `o1`, as a
complex array over `wavelengths`. `"wavelengths"` holds the wavelengths and
`"mode_neff"` the effective index of each port mode. The
[Quick start](quickstart.md) shows a complete call.

| Parameter | Meaning | Default |
|---|---|---|
| `wavelengths` | Wavelengths in metres | 21 points from 1.5 to 1.6 µm |
| `layer_stack` | gdsfactory layer stack | that of the active PDK |
| `material_index` | Refractive index per material name, overriding the built-in table | none |
| `steps_per_wavelength` | Mesh density | 20 |
| `input_ports` | Names of the ports to excite | all ports |
| `port_symmetry` | Use a symmetry of the device to run fewer simulations: `"auto"`, a port map such as `{"o2": "o3"}`, or a list of maps (see below) | none |
| `mode_index` | Which mode to use at every port, 0 being the highest effective index | 0 |
| `num_modes` | Modes solved per port | 2 |
| `port_offset` | Distance of the port planes outside the device | 0.5 µm |
| `port_lateral_margin` | Width of the mode window beyond the waveguide edge, on each side | 1 µm |
| `port_z_margin` | Height of the mode window below and above the patterned layers | 0.6 µm |
| `xy_margin` | Space between the component and the absorbing boundary | 1 µm |
| `z_margin` | Space below and above the patterned layers | 0.8 µm |
| `port_extension` | Extra waveguide length added at each port | 2 µm |
| `background` | Background medium | `"sio2"` |
| `wafer_levels` | Names of layer-stack levels that cover the whole wafer, in addition to those found automatically (see [gdsfactory](gdsfactory.md)) | none |
| `exclude` | Names of layer-stack levels to leave out | substrate, metals, vias, heater |
| `run_time_factor` | Run time in units of the transit time across the device | 6 |
| `shutoff` | Early-stop threshold | 1e-5 |
| `device`, `progress` | Passed to `run` | `"auto"`, 0 |
| `return_results` | Also return the `Results` of every run | off |

Points to know:

- One simulation is run per input port. For a reciprocal device, exciting a
  subset of the ports with `input_ports` is often enough.
- A device with a mirror symmetry needs fewer runs for its full S-matrix.
  With `port_symmetry="auto"` the layout is compared with its mirror images
  about the two centre lines of its bounding box (exactly, layer by layer);
  ports that are images of each other are excited once, and the columns of
  the others are copied, $S_{j,\sigma(i)} = S_{\sigma(j),i}$ for a symmetry
  $\sigma$ of the ports. A 1x2 splitter takes two runs instead of three, a
  symmetric directional coupler one instead of four. A dictionary such as
  `{"o2": "o3"}` states a symmetry by hand, and a list of dictionaries
  several. The copied entries equal those of their own runs to the accuracy
  of the port modes (2e-6 for a coupler, 5e-4 for the splitter, whose two
  output windows touch), because the automatic mesh is mirror symmetric for
  a mirror-symmetric layout. `lightsim.sparams.detect_port_symmetries`
  returns what `"auto"` finds.
- Ports must point along x or y. A port at another angle raises an error.
- One mode per port is used in a call. Keys end in `@0` whichever
  `mode_index` was chosen.
- The reference planes are the port planes, `port_offset` outside the device,
  and for the excited port the monitor two cells further in. Phases are not
  referred back to the port positions of the layout.
- The mode window of a port is narrowed, if necessary, so that it does not
  include a neighbouring port on the same side.
- Material indices are constants (see [gdsfactory](gdsfactory.md)); material
  dispersion is not included unless you pass your own media.

## The 2.5-D effective-index approximation

A planar circuit can be simulated much faster by collapsing the vertical
direction: every region of the layout is replaced by a uniform medium whose
index is the effective index of the slab mode of the layer stack there, and
the resulting problem is solved as a 2-D simulation (see
[2-D, 2.5-D and 3-D](dimensions.md)).
`lightsim.eim.write_sparameters_2d` does this with the same arguments and the
same result format as `write_sparameters`:

```python
import gdsfactory as gf
from lightsim import eim

gf.gpdk.PDK.activate()
mmi = gf.components.mmi1x2()                     # a 1x2 splitter of the generic PDK
sp = eim.write_sparameters_2d(mmi, wavelengths=np.linspace(1.5e-6, 1.6e-6, 5),
                              steps_per_wavelength=14, input_ports=["o1"], device="cpu")
print(np.round(np.abs(sp["o2@0,o1@0"]) ** 2, 3))
print(np.round(np.abs(sp["o3@0,o1@0"]) ** 2, 3))
print({name: round(value, 3) for name, value in sp["slab_neff"].items()})
```

```text
[0.494 0.496 0.497 0.496 0.492]
[0.494 0.496 0.497 0.496 0.492]
{'core': 2.848}
```

Each output arm receives close to half of the power. This run takes under
20 seconds on a laptop CPU; the same component in 3-D has about a hundred
times more cells and is a job for a GPU.

The approximation is good for devices whose vertical mode shape is the same
everywhere, such as couplers and interference devices in a fully etched
layer. It cannot describe anything that scatters light out of the plane, such
as grating couplers, and it uses one constant effective index per region, so
the wavelength dependence is approximate. The [effective-index
notes](../modules/eim.md) describe the method, compare it with 3-D for this
component and list the limits. Verify a final design in 3-D.
