# gdsfactory

After this page you know how lightsim turns a gdsfactory component and a
layer stack into a 3-D simulation, how to import a layout file from another
tool, how to supply your own layer stack and material indices, which parts
of a process design kit (PDK) are used and which are ignored, and how to
move a script written for gdsfactory's Meep, Tidy3D or Lumerical plugins to
lightsim ([Switching from gplugins](#switching-from-gplugins)).

[gdsfactory](https://gdsfactory.github.io/gdsfactory/) describes a photonic
layout as a **component**: polygons on layers, plus named **ports** where
waveguides connect. A **layer stack** says what each layout layer means in the
vertical direction: its height above the wafer, its thickness and its
material. lightsim extrudes the polygons accordingly. This requires
gdsfactory 8 or later.

## From layer stack to structures

`lightsim.gds.stack_layers(component, layer_stack=None)` lists the levels of
the stack that matter for a component. Without a `layer_stack` argument the
stack of the active PDK is used.

```python
import gdsfactory as gf
import numpy as np
from lightsim import gds

gf.gpdk.PDK.activate()                        # gdsfactory's generic PDK
mmi = gf.components.mmi1x2()
for level in gds.stack_layers(mmi):
    print(f"{level.name:5s} z = {level.zmin * 1e6:5.2f} to {level.zmax * 1e6:4.2f} um  "
          f"{level.material:5s} polygons: {len(level.polygons)}")
```

```text
box   z = -3.00 to 0.00 um  sio2  polygons: 0
core  z =  0.00 to 0.22 um  si    polygons: 1
clad  z =  0.00 to 3.00 um  sio2  polygons: 0
```

Two kinds of level appear:

- **Patterned levels** have polygons in the component. Each polygon becomes
  an extruded polygon (`PolySlab`) between `zmin` and `zmax`. Levels without
  polygons in this component are left out.
- **Wafer levels** cover the whole wafer and have no polygons: the buried
  oxide and the cladding. Each becomes a slab that covers the whole
  simulation region. A level is a wafer level if its layer is the PDK's
  wafer layer (a layer whose name contains `WAFER`, as in the generic PDK),
  if it is defined with `background=True`, or if you name it in the
  `wafer_levels` argument.

`component_to_structures` builds the structures and media in painting order:
wafer slabs first, then patterned levels. Where patterned levels overlap, the
one with the lower `mesh_order` in the stack is painted last and wins.

```python
structures, media, levels = gds.component_to_structures(mmi)
print([(s["name"], s["shape"]["type"], s["medium"]) for s in structures])
print([(m["name"], round(m["eps"][0] ** 0.5, 3)) for m in media])
```

```text
[('box', 'box', 'sio2'), ('clad', 'box', 'sio2'), ('core_0', 'poly_slab', 'si')]
[('sio2', 1.444), ('si', 3.476)]
```

`port_info(component)` returns the optical ports, in SI units:

```python
for port in gds.port_info(mmi):
    print(port["name"], np.round(np.array(port["center"]) * 1e6, 3), port["orientation"], port["width"])
```

```text
o1 [-10.   0.] 180.0 5e-07
o2 [15.5    0.625] 0.0 5e-07
o3 [15.5   -0.625] 0.0 5e-07
```

`lightsim.sparams.write_sparameters` and `build_simulation` call these
functions for you. `build_simulation(component, wavelengths, ...)` returns
the simulation without sources and monitors, the ports and the z range of the
patterned levels. Use it when you want the geometry of a component but your
own sources and monitors.

## Materials

The material of a level is a name. lightsim turns it to lower case and looks
up a refractive index in the table `lightsim.gds.MATERIAL_INDEX`:

| Name | Index | Name | Index |
|---|---|---|---|
| `si`, `silicon` | 3.476 | `ge` | 4.27 |
| `sio2`, `oxide` | 1.444 | `aln` | 2.12 |
| `sin`, `si3n4`, `nitride` | 1.996 | `ln` | 2.21 |
| `polysilicon` | 3.48 | `tio2` | 2.25 |
| `air`, `vacuum` | 1.0 | `ta2o5` | 2.07 |
| `water`, `h2o` | 1.33 | `al2o3` | 1.75 |
| `silica` | 1.444 | | |

These are values at 1550 nm. Pass `material_index={"si": 3.50, ...}` to
`write_sparameters`, `build_simulation` or `component_to_structures` to
override entries or add materials. The same table, with the same overrides,
gives the index of the `background` medium.

A material name that is in neither table stops the import with an error that
names the layer-stack level and the material, for example

```text
ValueError: layer stack level 'core': material 'sin_pecvd' has no refractive index. Pass material_index={'sin_pecvd': n} (index at your wavelength). Known materials: ...
```

The exception is a name that the solver defines itself and the table does
not, `gold` and `silver` (see [Materials](materials.md)): a level of such a
material is given the solver's dispersive medium. The 2.5-D functions need a
number for every material and refuse these as well.

The indices are constants. The refractive-index models and the dispersion
data of a PDK are not read. For a band away from 1550 nm, or for a broad
band, supply the indices at your centre wavelength; for material dispersion,
build the simulation with `build_simulation` and replace entries of
`sim.media` by fitted media (see [Materials](materials.md)).

## Importing a layout file

Any GDS or OASIS file that gdsfactory can read can be simulated. A layout
file from another tool has no ports and no layer stack, so both must be
supplied.

The worked example makes a layout file with one polygon, a taper from 0.5 to
1.2 µm width on layer (1, 0), to stand for a file received from elsewhere:

```python
import klayout.db as kdb                      # installed with gdsfactory

layout = kdb.Layout()
layout.dbu = 0.001                            # database unit in um
cell = layout.create_cell("taper")
outline = [kdb.DPoint(0, -0.25), kdb.DPoint(4, -0.6), kdb.DPoint(4, 0.6), kdb.DPoint(0, 0.25)]
cell.shapes(layout.layer(1, 0)).insert(kdb.DPolygon(outline))
layout.write("taper.gds")
```

Import it, add a port at each end, and describe the layer: 300 nm of silicon
starting at $z = 0$. gdsfactory lengths are in micrometres.

```python
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from lightsim.sparams import write_sparameters

taper = gf.import_gds("taper.gds")
taper.add_port("o1", center=(0, 0), width=0.5, orientation=180, layer=(1, 0))
taper.add_port("o2", center=(4, 0), width=1.2, orientation=0, layer=(1, 0))

stack = LayerStack(layers={
    "core": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.3, zmin=0.0,
                       material="si", mesh_order=1),
})

sp = write_sparameters(taper, wavelengths=np.linspace(1.5e-6, 1.6e-6, 3), layer_stack=stack,
                       background="sio2", steps_per_wavelength=10,
                       input_ports=["o1"], device="cpu")
print(np.round(np.abs(sp["o2@0,o1@0"]) ** 2, 3))
print({name: round(neff.real, 3) for name, neff in sp["mode_neff"].items()})
```

```text
[0.998 0.997 0.997]
{'o1': 2.673, 'o2': 2.982}
```

The orientation of a port is the direction in which light **leaves** the
component through it, in degrees from the x axis. This stack has no wafer
levels, so the `background` medium is the cladding on all sides.

## What is read and what is not

| From the component | Used |
|---|---|
| Polygons on the layers of the stack, including derived layers | yes |
| Polygons with holes | yes, cut into pieces without holes |
| Optical ports: name, centre, orientation, width | yes |
| Electrical and other non-optical ports | no |
| Port layer and cross-section | no |
| Ports that do not point along x or y | not supported |

| From the layer stack | Used |
|---|---|
| `zmin`, `thickness` (a negative thickness extends downwards) | yes |
| `material` name | yes, through the index table |
| `mesh_order` | yes, as the painting order |
| Wafer levels (buried oxide, cladding) | yes, as slabs over the whole region |
| `background=True` | yes, marks a wafer level |
| `background_exclude_layers` | no: the level covers the whole region, with a warning |
| `sidewall_angle` | no: all walls are vertical |
| Etch bias, thickness tolerances and other level metadata | no |
| The levels `substrate`, `undercut`, `heater`, `via_contact`, `via1`, `via2`, `metal1`, `metal2`, `metal3` | skipped by default |

Points to know:

- The skipped levels are skipped by name. `stack_layers`,
  `component_to_structures`, `build_simulation` and `write_sparameters` take
  an `exclude` argument to change the list. Metals would in any case need a
  material that the index table cannot express.
- The silicon substrate is among the skipped levels. Light that leaks through
  the buried oxide is absorbed by the boundary instead of being reflected by
  the substrate.
- A level on any other layer that has no polygons in the component looks
  like a patterned level the component does not use, and is left out. A
  cladding on a custom layer is such a level. Name it in
  `wafer_levels=["clad", ...]`, which `write_sparameters`, `build_simulation`,
  `component_to_structures` and `stack_layers` accept, or define it with
  `background=True` in the layer stack. When a level is left out and no wafer
  level covers its height, a warning names it, because the background medium
  then takes its place.
- The z range of the simulation follows the patterned levels: `z_margin`
  below the lowest and above the highest. Wafer levels thicker than that are
  cut by the region and continue into the absorbing boundary.
- Weakly confining platforms, such as thin nitride, need larger margins than
  the defaults, which suit silicon strip waveguides. Increase `xy_margin`,
  `z_margin`, `port_lateral_margin` and `port_z_margin` until the port modes
  no longer change (see [Boundaries](boundaries.md)).

## Switching from gplugins

gdsfactory's simulator integrations live in a separate package,
[gplugins](https://github.com/gdsfactory/gplugins), with one module per
solver: `gplugins.gmeep` (Meep), `gplugins.tidy3d`, `gplugins.lumerical` and
others. `lightsim.gplugins` gives lightsim the interface of `gplugins.gmeep`:
the same function names, arguments, units (micrometres), defaults, result
dictionary and cache files. A Meep script changes one line:

```text
import gplugins.gmeep as gm          # before: Meep
import lightsim.gplugins as gm       # after: lightsim, on a GPU if there is one
sp = gm.write_sparameters_meep(component, resolution=30, ymargin=3)
```

From Tidy3D or Lumerical the call changes as well, since those plugins have
their own arguments (see the table below):

```text
sp = gplugins.tidy3d.write_sparameters(c, wavelength=1.55, bandwidth=0.2, num_freqs=21)
sp = gplugins.lumerical.write_sparameters_lumerical(c)
sp = lightsim.gplugins.write_sparameters_lightsim(c, is_3d=True, wavelength_start=1.45, wavelength_stop=1.65, wavelength_points=21)
```

A complete run, in 2-D on the CPU:

```python
import lightsim.gplugins as gm

straight = gf.components.straight(length=2)
sp = gm.write_sparameters_lightsim(straight, ymargin=3, resolution=20, wavelength_points=5,
                                   effective_index=False, device="cpu", dirpath="sparameters")
print(sorted(sp))
print(sp["wavelengths"])
print(np.round(np.abs(sp["o2@0,o1@0"]), 3))
```

```text
['o1@0,o1@0', 'o1@0,o2@0', 'o2@0,o1@0', 'o2@0,o2@0', 'wavelengths']
[1.5   1.525 1.55  1.575 1.6  ]
[1. 1. 1. 1. 1.]
```

The result is a dictionary of numpy arrays: `wavelengths` in micrometres and
one complex array per S-parameter. `"o2@0,o1@0"` is the amplitude that
leaves through port `o2` in mode 0 when mode 0 is launched into port `o1`
(output first, as in `gplugins.gmeep`). The dictionary is also written to an
`.npz` file whose name is a hash of the layout and of the settings, next to
a `.yml` file of the settings and the run time. The same call returns the
file's contents instead of simulating again, until `overwrite=True`:

```python
path = gm.write_sparameters_lightsim(straight, ymargin=3, resolution=20, wavelength_points=5,
                                     effective_index=False, dirpath="sparameters",
                                     only_return_filepath_sim_settings=True)
print(path.suffix, path.with_suffix(".npz").exists())
```

```text
.yml True
```

The hash is computed by gplugins' own `get_sparameters_path` when gplugins
is installed, and by an identical function in `lightsim.gplugins`
otherwise; lightsim does not need gplugins. Neither `device` nor the
machine enters the hash.

### Arguments

`write_sparameters_lightsim` takes every argument of
`write_sparameters_meep`, in the same order and with the same defaults:

| Argument | Meaning in lightsim |
|---|---|
| `component` | as in gplugins |
| `port_source_names`, `port_source_modes`, `port_modes` | as in gplugins: the ports and modes excited, the modes measured |
| `port_symmetries` | gplugins' format `{"o2@0,o1@0": ["o1@0,o2@0"]}`; a source is skipped when its whole column follows from the runs before it |
| `resolution` | cells per micrometre of a uniform mesh (default 30) |
| `wavelength_start`, `wavelength_stop`, `wavelength_points` | the band, in micrometres (1.5, 1.6, 50) |
| `dirpath`, `filepath`, `overwrite`, `only_return_filepath_sim_settings` | as in gplugins |
| `layer_stack` | as in gplugins; default: the active PDK's |
| `port_margin` | the mode window reaches this far beside the port, at most half-way to a neighbouring port |
| `port_source_offset`, `port_monitor_offset` | as in gplugins (micrometres, outwards positive, default -0.1) |
| `distance_source_to_monitors` (keyword) | as in gplugins (0.2): the source-port monitor sits this far behind the source |
| `run` | `False` returns the lightsim `Simulation` of the first source |
| `dispersive` | 3-D: materials of `lightsim.materials` become pole fits over the band |
| `xmargin`, `ymargin`, `xmargin_left`, ... | distance from the component to the absorber, as in gplugins |
| `zmargin`, `zmargin_top`, `zmargin_bot` | 3-D: as in gplugins, but 0 (gplugins' default) becomes 0.8 µm |
| `decay_by` | the runs stop when the field energy has decayed by this factor (lightsim's `shutoff`) |
| `is_3d` | `False` (default): a 2-D model, see below |
| `verbosity` | above 0 prints progress |
| `material_name_to_meep` (keyword) | material name to an index or to a name of `lightsim.materials` |
| `extend_ports_length`, `tpml`, `clad_material`, `dfcen`, `exclude_layers` (keywords) | as in gplugins |
| `animate`, `animate_center`, `animate_size`, `lazy_parallelism`, `z`, `plot_args`, `padding_layers`, `meep.Simulation` keywords | accepted and ignored, with one `IgnoredArgumentWarning` that names them |
| `device` (lightsim only) | `"auto"`, `"cpu"`, `"cuda:0"`, `"cuda:0,1"` (each run split over two GPUs), or a list such as `["cuda:0", "cuda:1"]` over which the sources run concurrently |
| `effective_index` (lightsim only) | 2-D model: `True` the effective-index model, `False` the bulk-index cut |
| `polarization` (lightsim only) | 2-D: `"te"` or `"tm"` |
| `steps_per_wavelength` (lightsim only) | a graded mesh instead of `resolution` |

A keyword that neither gplugins nor lightsim knows raises a `ValueError`, as
in gplugins.

### The gplugins entry points and their lightsim equivalents

| gplugins | lightsim |
|---|---|
| `gmeep.write_sparameters_meep` | `lightsim.gplugins.write_sparameters_lightsim` (also as `write_sparameters_meep`) |
| `gmeep.write_sparameters_meep_1x1`, `_1x1_bend90` | `write_sparameters_lightsim_1x1`, `_1x1_bend90` |
| `gmeep.write_sparameters_meep_batch` (and `_1x1`, `_1x1_bend90`) | `write_sparameters_lightsim_batch`: jobs spread over the GPUs, one at a time on each; returns the `.npz` paths |
| `gmeep.write_sparameters_meep_mpi` (and `_1x1`, `_1x1_bend90`) | `write_sparameters_lightsim_mpi`: same arguments and return value, runs in the calling process (lightsim needs no MPI) |
| `gmeep.get_simulation` | `lightsim.gplugins.get_simulation`: a lightsim `Simulation` with its source and monitors |
| `gmeep.write_sparameters_grating`, `_grating_batch` | `write_sparameters_grating_lightsim`, `_grating_lightsim_batch` (same geometry arguments; the fibre mode is a Gaussian beam) |
| `gmeep.get_simulation_grating_fiber` | `write_sparameters_grating_lightsim(run=False)` |
| `gmeep.get_port_eigenmode` | `lightsim.solve_modes` |
| `modes.find_modes_waveguide`, `find_modes_coupler`, `find_neff_vs_width`, `find_mode_dispersion`, `find_neff_ng_dw_dh`, `find_coupling_vs_gap`, `coupling_length` (MPB) | the same names in `lightsim.gplugins` (from `lightsim.gplugins_modes`; see [Mode solver](../modules/mode-solver.md)) |
| `gmeep.get_meep_adjoint_optimizer`, `run_meep_adjoint_optimizer` | `lightsim.invdes` (its own interface; see [Inverse design](inverse-design.md)) |
| `gmeep.get_sparameters_data_meep`, `common.utils.get_sparameters_path_meep` | `lightsim.gplugins.get_sparameters_path` (tool `"lightsim"`), then `np.load` |
| `common.utils.port_symmetries` | `lightsim.gplugins.port_symmetries` (the same dictionaries) |
| `common.utils.plot.plot_sparameters` | works on lightsim's result unchanged |
| `tidy3d.write_sparameters` | `write_sparameters_lightsim(is_3d=True)`; see the differences below |
| `tidy3d.write_sparameters_batch` | `write_sparameters_lightsim_batch` |
| `tidy3d.write_sparameters_grating_coupler`, `_batch` | `write_sparameters_grating_lightsim` (2-D side view with Meep's arguments, not Tidy3D's) |
| `tidy3d.modes.Waveguide`, `WaveguideCoupler` | `lightsim.gplugins.Waveguide`, `WaveguideCoupler` (bends with `bend_radius`) |
| `tidy3d.modes.sweep_n_eff`, `sweep_n_group`, `sweep_fraction_te`, `sweep_mode_area`, `sweep_bend_mismatch`, `sweep_coupling_length` | the same names in `lightsim.gplugins` (arrays instead of xarrays) |
| `lumerical.write_sparameters_lumerical` | `write_sparameters_lightsim(is_3d=True)` |
| `lumerical.write_sparameters_lumerical_components` | `write_sparameters_lightsim_batch` |
| `sax.read.model_from_npz` | reads lightsim's `.npz` unchanged; `lightsim.gplugins.model_from_npz` calls it, or builds the same model without gplugins |

### What differs, and why

- **The 2-D model.** Meep's `is_3d=False` simulates the plain 2-D cut of
  the layout, with the bulk index of each material (silicon 3.48 in oxide),
  and launches the mode of highest index of either polarization, which in
  such a cut has E along z, normal to the chip. That is neither the
  polarization nor the effective index of the 3-D waveguide.
  lightsim's default 2-D model is the effective-index model (see
  [2-D, 2.5-D and 3-D](dimensions.md)): each level gets the effective index
  of its slab mode, and `polarization="te"` puts E in the chip plane, as for
  the quasi-TE mode of the 3-D guide. `effective_index=False` gives Meep's
  model instead: the bulk indices, and the mode of highest index unless
  `polarization` is given. The swap test below uses it.
- **Mode windows.** gplugins makes each port's mode window `port_margin`
  (2 µm) wider than the waveguide on each side. When two ports are closer
  than that, as the outputs of `mmi1x2` (1.25 µm apart), Meep's window holds
  both waveguides, solves for their joint mode and overstates the
  transmission: Meep's default run of `mmi1x2` gives 0.75 of the power in
  each arm. lightsim stops each window half-way to the neighbouring port.
- **The z margin.** gplugins passes `zmargin_top=zmargin_bot=0` to its 3-D
  simulation, which puts the absorber against the core. lightsim uses
  0.8 µm when the margin is 0.
- **Mode frequency.** Meep solves the port modes at every wavelength;
  lightsim at the centre of the band. Over the default 100 nm band the
  difference in |S| is below the discretisation error.
- **Phase reference.** The S-parameters are referred to the same planes
  (`port_monitor_offset` outside the ports; for the source port,
  `distance_source_to_monitors` behind the source) and use the same
  convention for reflections: the backward mode has the transverse E of the
  forward mode, so a perfect conductor reflects with -1. The phase of each
  port's mode profile is a convention of each mode solver: lightsim makes
  the largest transverse E component real and positive, Meep's MPB fixes it
  otherwise. Phases therefore agree up to a constant per pair of ports;
  their variation over the band (group delay) and the relative phases of
  ports with the same orientation agree.
- **Key order of other plugins.** `gplugins.tidy3d` writes
  `"o1@0,o2@0"` for the wave going from `o1` to `o2` (input first),
  `gplugins.gmeep` and lightsim for the wave going from `o2` to `o1`
  (output first). For a reciprocal component the two are equal.
  `gplugins.lumerical` keeps Lumerical's names (`S21`, ... by port index).
  All three give wavelengths in micrometres.
- **Tidy3D's arguments.** Tidy3D's plugin takes a centre `wavelength`,
  `bandwidth` and `num_freqs`, a `port_offset` measured inwards, and pads
  the region with `pad_xy_inner` / `pad_xy_outer`; translate them to
  `wavelength_start` / `wavelength_stop` / `wavelength_points`,
  `port_monitor_offset` (outwards positive) and the margins.
- **Port symmetries.** gplugins' Meep plugin runs every source port and uses
  `port_symmetries` only to copy entries; lightsim skips a source when the
  copies give its whole column. gplugins' own symmetry dictionaries leave
  out the reflections, so both run every source with them. lightsim's own
  `lightsim.sparams.write_sparameters(port_symmetry="auto")` finds the
  mirror symmetries of the layout and runs fewer sources.

### Batches, grating couplers, modes and circuits

`write_sparameters_lightsim_batch` takes a list of keyword dictionaries, as
`write_sparameters_meep_batch` does, and runs one at a time on each device
(all GPUs by default):

```python
jobs = [{"component": gf.components.straight(length=L), "ymargin": 3} for L in (1, 2)]
paths = gm.write_sparameters_lightsim_batch(jobs, devices=["cpu"], dirpath="sparameters",
                                            resolution=15, wavelength_points=3, effective_index=False)
print([p.suffix for p in paths])
```

```text
['.npz', '.npz']
```

`write_sparameters_grating_lightsim` builds gplugins' grating-coupler
geometry (substrate, buried oxide, etched grating, cladding, air gap and a
tilted fibre) as a 2-D side view and returns `"o2@0,o1@0"`, the coupling
from the waveguide mode into the fibre mode. The fibre mode is a Gaussian
beam of waist `fiber_core_diameter / 2`, tilted by `fiber_angle_deg`,
where Meep solves the mode of the tilted fibre core:

```python
gc = gm.write_sparameters_grating_lightsim(resolution=20, wavelength_points=5, device="cpu",
                                           dirpath="sparameters")
print(np.round(gc["wavelengths"], 3), np.round(np.abs(gc["o2@0,o1@0"]) ** 2, 2))
```

```text
[1.4   1.475 1.55  1.625 1.7  ] [0.02 0.26 0.05 0.   0.  ]
```

`Waveguide` and `WaveguideCoupler` have the interface of
`gplugins.tidy3d.modes` and solve the cross-section with lightsim's
full-vectorial mode solver (materials are indices, `(n, k)` pairs, names of
`lightsim.materials` or functions of the wavelength in µm; see
[Mode solver](../modules/mode-solver.md)):

```python
wg = gm.Waveguide(wavelength=1.55, core_width=0.5, core_thickness=0.22, core_material="si",
                  clad_material="sio2", num_modes=2, group_index_step=True)
print(np.round(wg.n_eff.real, 3), np.round(wg.n_group, 2), np.round(wg.fraction_te, 2))
```

```text
[2.443 1.778] [4.18 3.78] [0.98 0.04]
```

For circuits, `gplugins.sax.read.model_from_npz(path)` turns lightsim's
`.npz` file into a SAX model without changes;
`lightsim.gplugins.model_from_npz` calls it, or builds the same model when
gplugins is not installed (it needs `sax` and `jax`):

```text
from gplugins.sax.read import model_from_npz
model = model_from_npz(path)              # path: an .npz written by write_sparameters_lightsim
s = model(wl=np.linspace(1.5, 1.6, 11))   # {("o1", "o2"): ..., ...}
```

### Swap test against Meep

`benchmarks/gplugins_swap.py` runs the same call through
`gplugins.gmeep.write_sparameters_meep` (Meep 1.35.1, gplugins 2.1.5) and
`lightsim.gplugins.write_sparameters_lightsim` on five gdsfactory
components, in 2-D at `resolution=30`, 1.50 to 1.60 µm in 21 points,
`decay_by=1e-4`, silicon 3.476 in oxide 1.444 for both, and lightsim with
`effective_index=False` (Meep's 2-D model). The results are in
`benchmarks/results/gplugins_swap/`.

Transmission at 1.55 µm in dB, the largest difference in |S| over the band
for the main entries (|S| above 0.1) and for the small ones (reflections,
crosstalk), and how much the phase difference between the two solvers
varies over the band (a constant phase difference is a mode-phase
convention, see above):

| Component | Entry | Meep | lightsim | max Δ\|S\| main | max Δ\|S\| small | phase spread |
|---|---|---|---|---|---|---|
| straight (2 µm) | `o2@0,o1@0` | 0.001 | 0.000 | 0.000 | 0.003 | 0.1° |
| bend_euler (5 µm) | `o2@0,o1@0` | -0.007 | -0.009 | 0.000 | 0.016 | 0.3° |
| mmi1x2, `port_margin=0.5` | `o2@0,o1@0` | -4.25 | -4.36 | 0.056 | 0.032 | 20° |
| mmi1x2, default `port_margin=2` | `o2@0,o1@0` | -1.24 | -4.36 | 0.30 | 0.17 | |
| coupler | `o3@0,o1@0` | -6.95 | -5.73 | 0.072 | 0.017 | 4.2° |
| crossing | `o3@0,o1@0` | -8.25 | -7.32 | 0.060 | 0.072 | 8.7° |

- Port keys, their order and the wavelength axis (µm) are the same; the
  `.npz` files have the same layout.
- The straight and the bend agree to 0.003 in |S| and 0.3° in phase
  dispersion. The constant phase difference is -1.8° for the straight
  (both ports face along x) and -161° for the bend (one port faces along
  y): the mode-phase convention.
- With gplugins' default `port_margin=2` Meep's two output windows of
  `mmi1x2` overlap, and Meep reports 0.75 of the power in each arm
  (1.5 in total). With windows that do not overlap the two agree to 0.11 dB.
  lightsim gives the same answer with either margin.
- The coupler and the crossing scatter strongly, and at 30 cells/µm
  (13 cells per wavelength in silicon) both solvers are limited by the
  mesh. At 60 cells/µm Meep's crossing moves from -8.25 to -7.78 dB and
  lightsim's from -7.32 to -7.46 dB, so the gap closes from 0.93 to
  0.32 dB; lightsim's coupler moves from -5.73 to -5.97 dB.
- In 3-D (20 cells/µm, 1 µm z margin, run for the straight only, since a
  3-D Meep run on one process takes 41 minutes) the transmission agrees to
  0.001 in |S| and the phase difference varies by 1.9° over the band. Its
  constant part, 20°, is a difference of 0.05 in the effective index of
  the 220 nm core on this coarse mesh (4.4 cells), over the 1.6 µm between
  the reference planes.
- Reciprocity (largest |S_ij - S_ji|) is below 0.01 for both.
- Reflections use the same convention: a straight guide ending on a perfect
  conductor 1.7 µm behind the monitor gives S11 phases of -95.2° (Meep) and
  -96.0° (lightsim), |S11| = 1.000 in lightsim
  (`benchmarks/gplugins_reflection.py`).
- lightsim's GPU and CPU results agree to 1e-6.
- `gplugins.sax.read.model_from_npz` reads lightsim's `.npz` unchanged, and
  gplugins' `get_sparameters_path` and lightsim's built-in give the same
  file name.
- Grating coupler (`write_sparameters_grating`, 2-D, 30 cells/µm): Meep
  peaks at 1.52 µm with 0.24 coupling, lightsim at 1.49 µm with 0.29. The
  fibre mode differs (Meep's mode of the tilted fibre core, lightsim's
  Gaussian beam) and the 70 nm etch is two cells deep at this resolution;
  this comparison is looser than the others.

Run times in seconds. Meep ran as one process (gplugins'
`write_sparameters_meep`, no MPI); lightsim on one RTX PRO 6000, and on the
CPU of a shared 128-core machine that other jobs kept busy, so its CPU
times, and Meep's, are upper bounds:

| Component | Meep, 2-D | lightsim CPU, 2-D | lightsim GPU, 2-D | lightsim GPU, 3-D | lightsim CPU, 3-D |
|---|---|---|---|---|---|
| straight | 4.1 | 0.8 | 1.8 | 8.8 | 10.1 |
| bend_euler | 35.7 | 1.6 | 1.5 | 12.3 | 36.0 |
| mmi1x2 | 177 | 17.2 | 2.7 | 17.0 | 285 |
| coupler | 1223 | 36.8 | 6.3 | 54.4 | 2723 |
| crossing | 149 | 6.0 | 4.4 | 24.1 | 327 |

Meep's 3-D run of the straight took 2469 s. The 3-D runs use 20 cells/µm
and a 1 µm z margin. Small 2-D runs take a
second or two on the GPU whatever their size, the time of starting the
device. `write_sparameters_lightsim_batch` ran the six 3-D jobs of the test
in 142 s on one GPU and 84 s on two.
