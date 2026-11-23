# Using lightsim from any layout tool

After this page you can simulate a cell of a GDSII file made by any layout
tool, from a technology file and a port list, and hand the S-parameters to
a circuit simulator as a Touchstone file, as INTERCONNECT's text format, as
gplugins' `.npz` file or as a SAX model. Nothing here needs gdsfactory.
lightsim reads the GDSII file itself, with a reader in its Rust core.

| In | Out |
|---|---|
| GDSII file (`.gds`), one cell | Touchstone `.sNp` (1.1, or 2.0 keywords) |
| technology: TOML (gamma's `stack.toml`), YAML or JSON | INTERCONNECT text `.dat` |
| ports: JSON, YAML or CSV file; SiEPIC pins; pin + label layers | gplugins `.npz`, SAX model |
| | lightsim JSON specs, one per source, without running |

## A layout from another tool

The example writes a small layout with KLayout's Python module, as any tool
would: a 0.45 µm wide, 5 µm long waveguide on layer 1/0.

```python
import klayout.db as kdb

layout = kdb.Layout()
layout.dbu = 0.001
top = layout.create_cell("wire")
top.shapes(layout.layer(1, 0)).insert(kdb.DBox(0, -0.225, 5, 0.225))
layout.write("wire.gds")
```

`read_gds` flattens a cell (by default the only top cell) into polygons per
layer, in metres:

```python
import numpy as np
import lightsim.layout as ll

lay = ll.read_gds("wire.gds")
print(lay.cell, lay.layers, np.round(np.array(lay.bbox()) * 1e6, 3).tolist())
```

```text
wire [(1, 0)] [[0.0, -0.225], [5.0, 0.225]]
```

## The technology file

A technology file lists the levels of the stack and their materials, in
micrometres. lightsim reads gamma's TOML layer stack as it is:

```python
from pathlib import Path

Path("stack.toml").write_text('''
wavelength_um = 1.55

[[layer]]
name = "box"
zmin = -2.0
thickness = 2.0
material = "sio2"

[[layer]]
name = "core"
gds = [1, 0]
zmin = 0.0
thickness = 0.22
material = "si"
color = "#d9a441"        # display and design-rule keys are ignored

[[layer]]
name = "clad"
zmin = 0.0
thickness = 1.0
material = "sio2"

[[material]]
name = "si"
index = 3.476
lightsim = "Si"

[[material]]
name = "sio2"
index = 1.444
lightsim = "SiO2"
''')
tech = ll.read_technology("stack.toml")
print([(l.name, l.layers, l.zmin, l.thickness) for l in tech.levels])
```

```text
[('box', [], -2.0, 2.0), ('core', [(1, 0)], 0.0, 0.22), ('clad', [], 0.0, 1.0)]
```

The same model is written and read as YAML or JSON, with the level fields
of gdsfactory's `LayerLevel`:

```python
import json

tech.write("tech.json")
saved = json.loads(Path("tech.json").read_text())
print(saved["version"], saved["layers"]["core"], saved["materials"]["si"])
```

```text
1 {'layer': [1, 0], 'zmin': 0.0, 'thickness': 0.22, 'material': 'si', 'mesh_order': 3} {'index': 3.476, 'lightsim': 'Si'}
```

The format, version 1:

| Key | Meaning |
|---|---|
| `version` | `1` (YAML/JSON; gamma's TOML has none and is read as version 1) |
| `wavelength_um` | design wavelength; library materials take their index there |
| `background` | optional: the medium around the stack (default: the wafer level at the height of the patterned levels) |
| `layers` | a table of levels by name (or a list with `name`); TOML: `[[layer]]` |
| `layer` (TOML: `gds`) | `[layer, datatype]`, `"1/0"`, or a list of them; none: the level covers the whole wafer (buried oxide, cladding) |
| `zmin`, `thickness` | µm |
| `material` | a name from `materials` |
| `mesh_order` | where patterned levels overlap, the lower value is painted last and wins (default 3) |
| `subtract` | layers removed from the level (gdsfactory's derived `A - B`); not computed, see below |
| `sidewall_angle` | read; lightsim extrudes vertical walls and warns |
| `materials` | by name: `index` (a constant), or `lightsim` (a name from `lightsim.materials`) |

`ll.technology_from_gdsfactory()` converts a gdsfactory `LayerStack`, and
`ll.read_lyp("layers.lyp")` reads the layer names of a KLayout layer
properties file. lightsim does not compute layer booleans. A level that
subtracts other layers (the core of gdsfactory's generic PDK is `WG -
DEEP_ETCH - SHALLOW_ETCH`) is painted from its first layer, with a warning
when the cell has polygons on a subtracted layer. Draw rib and slab regions
on their own layers, as their own levels.

## The ports

A port file lists, per port, the centre, the width, the direction in which
light leaves the device and the waveguide layer, in micrometres. These are
gamma's two forms. JSON:

```python
Path("ports.json").write_text('''[
 {"name": "o1", "x": 0.0, "y": 0.0, "width": 0.45, "orientation": 180.0, "layer": [1, 0]},
 {"name": "o2", "x": 5.0, "y": 0.0, "width": 0.45, "orientation": 0.0, "layer": [1, 0]}
]''')
Path("ports.csv").write_text("name,x,y,width,orientation,layer,datatype\no1,0,0,0.45,180,1,0\no2,5,0,0.45,0,1,0\n")
ports = ll.read_ports("ports.json")
print(ports == ll.read_ports("ports.csv"), ports[1]["center"], ports[1]["orientation"])
```

```text
True (5e-06, 0.0) 0.0
```

The format, version 1:

| Key | Meaning |
|---|---|
| a list, or `{"version": 1, "units": "um", "ports": [...]}` | `units`: `um` (default), `nm`, `mm` or `m` |
| `name` | port name |
| `x`, `y` (or `center: [x, y]`) | the port's centre |
| `width` | waveguide width |
| `orientation` (or `direction`) | degrees, 0 = +x, 90 = +y, the direction out of the device; `direction` also takes `east`/`+x`, `north`/`+y`, `west`/`-x`, `south`/`-y` |
| `layer` | `[layer, datatype]`; the port waveguide is extended on that layer's level |
| `modes` | optional: mode numbers launched and measured at the port (default `[0]`) |
| `z` | optional: `[zmin, zmax]` of the mode window (default: the simulation height) |
| other keys | ignored |

CSV: the header `name,x,y,width,orientation,layer,datatype`, and an optional
`modes` column (`0;1`).

Without a port file, the ports can come from the layout:

- `ports="siepic"`: SiEPIC's pins. These are two-point paths on the PinRec
  layer 1/10, drawn from inside the device to outside, named by a text on
  1/10. The waveguide layer is Si 1/0. The layer numbers are those of the
  SiEPIC EBeam PDK's `klayout/EBeam/EBeam.lyp`.
- `ports={"convention": "pins", "pin_layer": "90/0", "label_layer": "91/0",
  "port_layer": "1/0"}`: the same convention on other layers. A pin may also
  be a box: the port is at its centre, as wide as its longer side, and faces
  away from the centre of the layout.
- `ports="gdsfactory"`: the port metadata gdsfactory writes into the GDS.
  It is stored in a form only KLayout reads, so this route needs the
  gdsfactory package.

## S-parameters

`lightsim.layout.sparameters` sets up the simulation as
`lightsim.gplugins` does (the same code): port waveguides extended through
the absorber, mode sources and monitors at the gplugins offsets, one run per
source port and mode. It returns an `SParameters` object. This is a 2-D run
with bulk indices, on the CPU:

```python
sp = ll.sparameters("wire.gds", "stack.toml", "ports.csv", is_3d=False, effective_index=False,
                    resolution=20, wavelengths=np.linspace(1.5e-6, 1.6e-6, 3), margin=2.0, device="cpu")
print(sp)
print(np.round(np.abs(sp["o2", "o1"]), 3))
```

```text
SParameters(2 ports ['o1@0', 'o2@0'], 3 frequencies 1.874e+14-1.999e+14 Hz)
[1. 1. 1.]
```

The defaults are a 3-D run with a graded mesh of 20 steps per wavelength,
1.50 to 1.60 µm in 21 points, 1 µm margins in x and y and 0.8 µm in z.
`device` takes `"cuda:0"`, `"cuda:0,1"` or a list of devices over which the
sources run concurrently.

`lightsim.layout.simulation` stops before running. It returns the
`Simulation` of each source, whose JSON spec holds the mode profiles of the
source and the monitors. `write_specs` writes the specs:

```python
sims = ll.simulation("wire.gds", "stack.toml", "ports.csv", is_3d=False, resolution=20)
print(list(sims), [p.name for p in ll.write_specs(sims, "wire.json")])
```

```text
['o1@0', 'o2@0'] ['wire_o1@0.json', 'wire_o2@0.json']
```

The command-line program does the same, through the Python package:

```text
lightsim sparams chip.gds --cell mmi --technology stack.toml --ports ports.csv --out mmi.s3p
lightsim layout-spec chip.gds --cell mmi --technology stack.toml --ports siepic --out spec.json
lightsim gds chip.gds                          # cells, top cells, units, layers (no Python needed)
lightsim gds chip.gds --cell mmi --out flat.json
```

`sparams` and `layout-spec` take `--2d`, `--bulk`, `--resolution`,
`--wavelengths 1.5:1.6:21` (µm), `--margin`, `--z-margin`, `--device` and,
instead of `--ports`, `--pins PIN,LABEL,PORT` layers. The output format
follows the suffix: `.sNp` (Touchstone), `.dat` (INTERCONNECT text) or
`.npz` (gplugins).

## Writing and reading S-parameters

```python
sp.to_touchstone("wire.s2p")                   # Touchstone 1.1, GHz, real/imaginary
sp.to_touchstone("wire_v2.s2p", version="2.0", freq_unit="Hz", fmt="MA")
sp.to_lumerical_dat("wire.dat")                # INTERCONNECT, SiEPIC, SAX, Simphony
sp.to_npz("wire.npz")                          # gplugins' layout
back = ll.SParameters.from_touchstone("wire.s2p")
print(back.allclose(sp), ll.SParameters.from_lumerical_dat("wire.dat").allclose(sp))
print(Path("wire.s2p").read_text().splitlines()[:5])
```

```text
True True
['! S-parameters written by lightsim', "! phase convention: exp(+j w t) (complex conjugate of lightsim's exp(-i w t) values)", '! port 1 = o1@0', '! port 2 = o2@0', '! model: 2-D bulk index']
```

Touchstone has no field for port names or modes. They go into the comment
lines and into a sidecar file (`wire.s2p.json`), which `from_touchstone`
reads. Entries that were not simulated are written as 0 in Touchstone,
listed in a comment, and left out of the INTERCONNECT file.

### Phase convention

lightsim's fields vary in time as exp(-iωt). A delay line of group delay τ
therefore has S21 ≈ exp(+iωτ): its phase grows with frequency. The formats
differ in their conventions:

| Format | Convention | What lightsim writes |
|---|---|---|
| INTERCONNECT text (`.dat`) | exp(-iωt). The through port of SiEPIC's ring-coupler data has a phase that grows with frequency, at the delay of its straight section | the values as they are |
| SAX models (`to_sax`), gplugins `.npz` | exp(-iωt) (`sax.models.straight` is exp(+2πi n L / λ)) | the values as they are |
| Touchstone (`.sNp`) | exp(+jωt), as in scikit-rf, ADS and Spectre | the complex conjugate, stated in the header; `from_touchstone` conjugates back (`conjugate=False` writes the values as they are) |

The tests check this with a delay line. scikit-rf reads lightsim's
Touchstone file and reports a positive group delay equal to τ, and SAX's
parser reads the `.dat` file with the phase unchanged. SAX's own
`parse_touchstone` does not conjugate. For SAX, use `to_sax()` or the
`.dat` file, not Touchstone.

## Per tool

Each subsection says what was tested with lightsim and what is taken from
the tool's public documentation without having been run.

### KLayout and SiEPIC

Tested: layouts written with KLayout's Python module (above) and with
gdstk; SiEPIC pins (`ports="siepic"`) and the SiEPIC layer numbers; layer
names from `.lyp` files; SiEPIC's INTERCONNECT data files
(`ebeam_dc_halfring_straight` from the EBeam PDK), parsed and written back
unchanged. OASIS files are not read: open them in KLayout and save them as
GDS2. Not run: loading lightsim's `.dat` into a SiEPIC circuit
([SiEPIC-Tools](https://github.com/SiEPIC/SiEPIC-Tools)).

### Nazca

Not run. Nazca writes GDSII files, which lightsim reads. Its pins are not
stored in the GDS, so write a port file from the pin positions your Nazca
script knows. See [nazca-design.org](https://nazca-design.org).

### IPKISS / Luceda

Not run; from Luceda's documentation. `i3.device_sim.SMatrix1DSweep.from_touchstone(filename, term_mode_map=..., unit=...)`
(also `ipkiss3.io.touchstone.import_touchstone_smatrix`) imports a Touchstone
file. `term_mode_map` maps `(name, mode)` pairs to Touchstone port numbers,
which lightsim's comments and sidecar list
([SMatrix1DSweep](https://docs.lucedaphotonics.com/reference/circuitmodels/ref/ipkiss3.all.circuit_sim.SMatrix1DSweep),
[import_touchstone_smatrix](https://docs.lucedaphotonics.com/reference/io/ref/ipkiss3.io.touchstone.import_touchstone_smatrix.html)).

### PhotonForge

Not run; from Flexcompute's documentation. PhotonForge's data model takes
S-parameter arrays such as those loaded from a Touchstone file, and its
documentation warns that its dictionary keys `(ports[i], ports[j])` are in
the reverse order of the usual s_ij
([Data Model](https://docs.flexcompute.com/projects/photonforge/en/latest/guides/Data_Model.html)).

### Lumerical INTERCONNECT

Not run (licensed software). The Optical N-Port S-Parameter element loads
the text format with "load from file", and recent versions also load
single-mode Touchstone files
([Optical N port S-parameter element](https://optics.ansys.com/hc/en-us/articles/360036107914)).
`to_lumerical_dat(path, sides={"o1": "LEFT", ...})` adds the port-position
lines the element uses to place the ports.

### SAX

Tested: `sax.parsers.parse_lumerical_dat` (SAX 0.18) reads lightsim's
`.dat` files with the values unchanged. `gplugins.sax.read.model_from_npz`
reads the `.npz` files. `sp.to_sax()` is a SAX model of the wavelength in
µm.

### Simphony

Not run; from Simphony's documentation. `simphony.plugins.lumerical.load_sparams`
loads a Lumerical `.dat` file
([load_sparams](https://simphonyphotonics.readthedocs.io/en/latest/api/simphony.plugins.lumerical.load_sparams.html)).
SAX's parser, which reads lightsim's files, was adapted from it.

## What the GDSII reader covers

Read: polygons (`BOUNDARY`), boxes (`BOX`), paths with widths and
`PATHTYPE` 0 (flush), 1 (round ends, 16 segments per half circle), 2
(extended by half the width) and 4 (`BGNEXTN`/`ENDEXTN`), with mitred
corners; cell references and arrays with rotation, magnification and
reflection; text labels; units. Flattened polygons match gdstk's to 1 nm in
the tests (references, arrays, rotations, reflections, magnifications,
paths of types 0, 2 and 4, and a gdsfactory layout).

Not read: OASIS; `NODE` elements; properties; text presentation and
transformations; the absolute-magnification and absolute-angle flags of
references (references always compose). KLayout's `$$$CONTEXT_INFO$$$`
cell is not taken as a top cell.
