# gdsfactory

After this page you know how lightsim turns a gdsfactory component and a
layer stack into a 3-D simulation, how to import a layout file from another
tool, how to supply your own layer stack and material indices, and which
parts of a process design kit (PDK) are used and which are ignored.

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
