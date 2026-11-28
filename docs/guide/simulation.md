# Simulation setup

After this page you can define the simulation region, choose a uniform,
automatic or hand-written mesh, place structures made of boxes, spheres,
cylinders and extruded polygons, and decide how material interfaces are
resolved inside a cell.

The running example is the cross-section of a silicon strip waveguide,
450 nm wide and 220 nm thick, buried in silicon dioxide.

## The region

`Simulation(min=..., max=..., ...)` takes the two corners of a box. This
region is what you model. Absorbing layers are added outside it, so sources,
monitors and structures can use the whole region (see
[Boundaries](boundaries.md)).

With two-component corners, `Simulation(min=[u0, v0], max=[u1, v1], ...)`,
the simulation is two-dimensional: nothing varies along the third axis and
the solver advances three field components instead of six. Everything on
this page applies to it, with one cell along that axis; see
[2-D, 2.5-D and 3-D](dimensions.md).

The region is filled with the `background` medium, `"vacuum"` unless you name
another. A medium is referred to by name. Common materials are built in
(`"si"`, `"sio2"`, `"si3n4"`, `"gold"` and others); your own are created with
`Medium` (see [Materials](materials.md)).

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

wavelength = 1.55 * um
f0 = ls.C0 / wavelength


def waveguide(mesh, subpixel=True):
    sim = ls.Simulation(min=[0, -1.5 * um, -1.0 * um], max=[2 * um, 1.5 * um, 1.0 * um],
                        mesh=mesh, run_time=100e-15, background="sio2", subpixel=subpixel)
    # the strip runs along x and is longer than the region
    sim.add_structure(ls.Box([-1, -0.225 * um, -0.11 * um], [1, 0.225 * um, 0.11 * um]), "si")
    return sim
```

## Mesh

The mesh decides the accuracy and the cost. The cost of a 3-D run grows with
the fourth power of the inverse cell size: three powers for the number of
cells and one for the number of time steps. A common starting point is 15 to
20 cells per wavelength in the densest material.

### Uniform

`UniformMesh(dx)` uses the same spacing everywhere. `dx` is one number or one
value per axis. The spacing is adjusted so that a whole number of cells fits
the region exactly: the number of cells along an axis is the length divided
by `dx`, rounded to the nearest integer.

```python
sim = waveguide(ls.UniformMesh(40e-9))
print([len(a) - 1 for a in sim.grid()])     # cells per axis, with 12 PML cells per face
```

```text
[74, 99, 74]
```

### Automatic

`AutoMesh(wavelength, steps_per_wavelength=20)` chooses the spacing from the
materials: `steps_per_wavelength` cells per wavelength **in the local medium**,
so high-index regions get small cells and the cladding gets larger ones. The
transition is gradual.

```python
sim = waveguide(ls.AutoMesh(wavelength, steps_per_wavelength=14))
for name, nodes in zip("xyz", sim.grid()):
    step = np.diff(nodes) * 1e9
    print(name, len(step), round(step.min(), 1), round(step.max(), 1))
```

```text
x 88 22.2 31.9
y 74 31.9 85.7
z 58 31.9 76.7
```

Across the strip the cells are 31.9 nm, which is the wavelength divided by
the index of silicon and by 14. In the oxide they grow to about 86 nm.

| Parameter | Meaning | Default |
|---|---|---|
| `wavelength` | Vacuum wavelength the mesh is designed for | required |
| `steps_per_wavelength` | Cells per wavelength in each medium | 20 |
| `grading` | Largest ratio between the sizes of neighbouring cells, approximately | 1.3 |
| `min_step` | Lower limit on the spacing chosen for structures | none |
| `max_step` | Upper limit on the spacing in the background | none |
| `overrides` | Boxes with an explicit spacing, `(min, max, [dx, dy, dz])` each | none |
| `symmetry` | Mirror planes of the mesh, `(x0, y0, z0)` with `None` for an axis without one | none |

How the spacing is chosen:

- The grid is rectilinear, so refinement cannot be local in all three
  directions. Along each axis, a structure refines the interval covered by
  its bounding box, across the whole region in the other two directions.
- The index used for a structure is the square root of its largest
  permittivity times its largest permeability. Dispersive media and perfect
  conductors are meshed as if their index were 4.
- An override box sets the largest spacing inside it, along each axis.
- Along each axis the mesh is built outwards from the centre of the region,
  with the same rule in both directions. A layout that is mirror symmetric
  about that centre therefore gets a mirror-symmetric mesh, with a grid node
  on the symmetry plane. This matters for devices that rely on symmetry: the
  two arms of a directional coupler must be discretised alike, or their modes
  differ slightly and the phase between the outputs drifts.
- If the symmetry plane of the device is not at the centre of the region,
  name it: `AutoMesh(wavelength, symmetry=(None, y0, None))` puts a node on
  the plane $y = y_0$ and mirrors the cells about it.
- The cells at the two ends of each axis take up the remainder. They are
  between 0.65 and 1.3 of the local spacing; in the example above these are
  the 22.2 nm cells along x. The smallest cell sets the time step.
- An axis shorter than 1.3 cells is a single cell, as in a two-dimensional
  simulation.

### Explicit

A mesh can be given as node coordinates, one list per axis, with
`ExplicitMesh(x, y, z)`. The first and last node of each list should coincide
with `min` and `max`.

```python
nodes_x = np.linspace(0, 2 * um, 51)
nodes_y = np.concatenate([np.linspace(-1.5, -0.3, 13)[:-1],     # 100 nm cells
                          np.linspace(-0.3, 0.3, 31)[:-1],      # 20 nm cells across the strip
                          np.linspace(0.3, 1.5, 13)]) * um
nodes_z = np.linspace(-1.0 * um, 1.0 * um, 81)
sim = waveguide(ls.ExplicitMesh(nodes_x, nodes_y, nodes_z))
print([len(a) - 1 for a in sim.grid()])
```

```text
[74, 78, 104]
```

> **Note:** Keep the ratio between neighbouring cells moderate. An abrupt
> change of cell size reflects a small part of a wave, as a material interface
> would.

### Freezing a mesh

An automatic mesh follows the structures, so a copy of a simulation with a
structure removed or added has a different grid. Mode profiles and recorded
fields are tied to a grid: a port mode solved on a copy that holds only one
waveguide of a device does not fit the device's grid.

`sim.frozen_mesh()` returns the grid a simulation has now as an explicit
mesh, which can be given to another simulation. `sim.freeze_mesh()` does the
same in place: it replaces the simulation's own mesh, after which changes to
the structures no longer move the grid.

```python
device = waveguide(ls.AutoMesh(wavelength, steps_per_wavelength=14))
empty = ls.Simulation(min=device.min, max=device.max, mesh=device.mesh,
                      run_time=device.run_time, background="sio2")
print([len(a) - 1 for a in empty.grid()])          # the automatic mesh of an empty region

empty.mesh = device.frozen_mesh()                  # the device's grid, without its structures
print([len(a) - 1 for a in empty.grid()])
print(all(np.array_equal(a, b) for a, b in zip(device.grid(), empty.grid())))
```

```text
[50, 64, 50]
[88, 74, 58]
True
```

The frozen mesh holds the nodes inside the region. Absorbing layers are added
outside it as usual, with the size of the outermost cell.

## Structures and shapes

`add_structure(shape, medium)` fills a shape with a medium. The medium is a
name or a `Medium`. Structures are painted in order: where two overlap, the
one added **later** wins. A shape may extend beyond the region; only the part
inside is used, which is a convenient way to make a waveguide run into the
absorbing boundary.

| Shape | Builder | Notes |
|---|---|---|
| Axis-aligned box | `Box(min, max)` | |
| Sphere | `Sphere(center, radius)` | |
| Cylinder | `Cylinder(center, radius, length, axis=2)` | `axis` is the cylinder axis; `length` is centred on `center` |
| Extruded polygon | `PolySlab(vertices, slab_min, slab_max, axis=2)` | see below |
| Sampled permittivity | `IndexGrid(min, max, eps)` | graded or imported permittivity, see [Materials](materials.md) |

A `PolySlab` is a polygon in the plane normal to `axis`, extruded between the
coordinates `slab_min` and `slab_max` along `axis`. The vertices are pairs of
coordinates on the two other axes: (x, y) for `axis=2`, (x, z) for `axis=1`
and (y, z) for `axis=0`. The polygon is closed automatically and may be
concave. It cannot have holes; cut such a shape into pieces, or paint the
hole afterwards with the surrounding medium. Consecutive polygons with the
same medium and the same extrusion range are treated as one union, so
abutting polygons from a layout leave no seam.

The example adds a taper that narrows the strip to a 100 nm tip:

```python
sim = waveguide(ls.AutoMesh(wavelength, steps_per_wavelength=14))
taper = [(1.0 * um, -0.225 * um), (1.8 * um, -0.05 * um), (1.8 * um, 0.05 * um), (1.0 * um, 0.225 * um)]
sim.add_structure(ls.Box([1.0 * um, -1, -1], [1, 1, 1]), "sio2")          # remove the strip beyond x = 1 um
sim.add_structure(ls.PolySlab(taper, -0.11 * um, 0.11 * um, axis=2), "si")  # then add the taper
print(len(sim.structures), [s["shape"]["type"] for s in sim.structures])
```

```text
3 ['box', 'box', 'poly_slab']
```

## Sub-pixel smoothing

A material interface rarely falls on a cell boundary. If each cell simply
took the material at its centre ("staircasing"), an interface would jump by a
whole cell when the geometry or the mesh changes slightly, and results would
converge slowly and erratically.

With `subpixel=True`, the default, a cell that is cut by an interface gets an
effective permittivity. The value depends on the orientation of the field
relative to the interface: a field component normal to the interface sees the
harmonic mean of the two permittivities, a parallel component the arithmetic
mean. With $n_a$ the component of the interface normal along the field
direction $a$,

$$
\frac{1}{\varepsilon_{\text{eff},a}} = n_a^2 \left\langle \frac{1}{\varepsilon} \right\rangle
 + \left(1 - n_a^2\right) \frac{1}{\langle \varepsilon \rangle} ,
$$

where the angle brackets are averages over the cell. The filled fraction is
computed exactly for boxes and polygons and by sampling for spheres and
cylinders.

The effect is visible in the effective index of the waveguide mode. The strip
is 450 nm wide, so its sidewalls fall inside a cell on a 40 nm and on a 20 nm
mesh:

```python
plane = (0, 1 * um, [1 * um, -1.2 * um, -0.8 * um], [1 * um, 1.2 * um, 0.8 * um])
for dx in (40e-9, 20e-9):
    for subpixel in (True, False):
        mode = ls.solve_modes(waveguide(ls.UniformMesh(dx), subpixel), *plane, f0, num_modes=1)[0]
        print(f"{dx * 1e9:.0f} nm  subpixel={subpixel!s:5}  n_eff = {mode.neff.real:.4f}")
```

```text
40 nm  subpixel=True   n_eff = 2.3545
40 nm  subpixel=False  n_eff = 2.3416
20 nm  subpixel=True   n_eff = 2.3576
20 nm  subpixel=False  n_eff = 2.3708
```

On a 10 nm mesh the same script gives 2.3529 with smoothing and 2.3591
without. With smoothing the three meshes agree within 0.005; without it they
spread over 0.03.

Limits of the smoothing:

- It applies to plain dielectrics and to dispersive dielectrics that are
  transparent in the band of the sources (smoothed with their permittivity at
  the sources' centre frequency, see [Materials](materials.md)). Metals and
  other lossy dispersive media, conductive and perfectly conducting media,
  and media with off-diagonal tensor terms are not blended: a field sample
  belongs to such a medium if its position lies inside the shape
  (staircasing). The surface of a metal therefore sits on the grid, and
  results converge with the first power of the cell size, not the second;
  [Materials](materials.md) gives numbers and advice.
- An `IndexGrid` is not smoothed at the faces of its own box.
- The effective permittivity is diagonal. The off-diagonal part of the exact
  smoothing tensor, which appears at interfaces oblique to the grid, is
  dropped.

Set `subpixel=False` to switch smoothing off, for example to compare with a
staircased reference.

### The weight of the average

`subpixel_weighting` chooses how the averages $\langle\varepsilon\rangle$ and
$\langle 1/\varepsilon\rangle$ of a field sample are weighted. The averaging
rule above is the same for both choices.

- `"cell"`, the default: the plain average over the sample's own cell.
- `"tent"`: a weight that is 1 at the sample and falls linearly to 0 at the
  neighbouring samples, along each axis. An interface is then spread over two
  cells instead of one, and the samples carry the volume and the first moment
  of every body exactly, wherever its faces fall inside a cell.

```python
sim = ls.Simulation(..., subpixel_weighting="tent")
```

On the command line, `lightsim run SPEC.json --subpixel-weighting tent`
overrides the spec, and `lightsim validate --subpixel-weighting tent` runs the
validation cases with it.

What the tent changes is how the error depends on where a body sits in the
grid. With the cell average the error of a flat face has a size and a sign
that change as the face moves through a cell; with the tent it is close to a
fixed multiple of the square of the cell size. Measured on waveguide
cross-sections and resonators against converged or analytic references, the
body moved through a cell in five steps
([`benchmarks/subpixel`](https://github.com/su3co/lightsim/tree/main/benchmarks/subpixel)):

| case, 20 nm cells | error with `"cell"` | error with `"tent"` |
|---|---|---|
| silicon slab 220 nm in oxide (2-D), index, E parallel to the faces | -2.5e-3 to +4.1e-3 | -2.5e-3 to -2.4e-3 |
| the same slab, E normal to the faces | +2.7e-4 to +7.7e-3 | -2.1e-3 to -9.7e-4 |
| silicon wire 480 x 220 nm in oxide, index of TE0 | -2.0e-3 to +5.1e-3 | -3.5e-3 to -3.0e-3 |
| the same wire, index of TM0 | -6.5e-4 to +8.0e-3 | -2.1e-3 to -7.3e-4 |
| nitride wire 1000 x 400 nm in oxide, index of TE0 | -1.0e-4 to +2.0e-4 | -2.0e-4 to -1.7e-4 |
| two silicon wires 200 nm apart, index splitting (0.0229) | -2.1e-4 to +3.4e-4 | +1.1e-4 to +4.1e-4 |
| silicon disc, resonance frequency (relative), E along its axis | -2.0e-3 to -1.5e-3 | -1.49e-3 to -1.49e-3 |
| silicon disc, resonance frequency (relative), E in its plane | -5.9e-4 to -3.2e-4 | +5.8e-4 to +6.0e-4 |

What the measurements say, at cells of 40 to 10 nm:

- **Field parallel to the faces.** The spread over placements falls by a
  factor of 5 to 100 (the slab: 4.6 at 40 nm, 53 at 20 nm, 100 at 10 nm; the
  disc with E along its axis: 16 to 100), and the error that remains falls
  close to fourfold when the cell is halved (the slab: -1.0e-2, -2.5e-3,
  -6.4e-4 at 40, 20 and 10 nm), so two meshes extrapolate cleanly and a sweep
  over a width or a thickness does not jitter when a face crosses a cell.
- **Field normal to the faces.** The spread falls by a factor of 1 to 13 (the
  slab: 2.0 at 40 nm, 6.7 at 20 nm, 1.1 at 15 nm, 13 at 10 nm); it is not
  removed. A wire's modes see both kinds of face: for the silicon wire's TE0
  the spread falls by 1.2 to 13, for its TM0 by 1 to 6.
- **The largest error is not smaller in general.** The tent's error has one
  sign; for the silicon wire's TE0 its largest value is 1.3e-2 against 2.0e-3
  at 40 nm cells, 3.5e-3 against 5.1e-3 at 20 nm and 9.7e-4 against 1.0e-3 at
  10 nm. In a 3-D run of a high-contrast guided mode it cancels most of the
  grid dispersion, which has the opposite sign; that is a property of such
  modes, not of the weight.
- **A gap of a few cells is described worse.** The index splitting of the two
  wires is off by up to 10 % at 40 nm cells (5 cells in the gap) and 6.5 % at
  30 nm, against 3.3 % and 1.2 % with the cell average; 1.8 % against 1.5 %
  at 20 nm, 1.9 % against 0.6 % at 15 nm, 0.5 % against 0.7 % at 10 nm.
- **Curved and oblique surfaces with a field component normal to them.**
  Neither weight is of second order there, because the off-diagonal part of
  the smoothing tensor is dropped (see the limits above), and the tent's
  first-order term is the larger one, since its interface is twice as wide:
  the disc resonance with E in its plane is off by 6.6e-4 at 10 nm cells
  against 2.8e-4, and in the validation cases the scattered power of a sphere
  by 1.8 % against 0.7 % and that of a cylinder with E in the plane by 1.6 %
  against 0.9 %.

With the tent, a cell that two or more bodies cover in part is composited
from the geometry, so boxes and stacked layers that share a face leave no seam
there. Metals and the other staircased media are staircased as before; media
with an off-diagonal permittivity are staircased on both sides of their
surface; the weight of the field component normal to a conducting sheet is
folded back at the sheet instead of reaching across it. Bodies are not
repeated across a periodic boundary: draw them across it, as for the cell
average. Setting up a grid of 13 million cells takes up to 0.2 s longer; the
time steps are the same.

## Many small runs at once

A small 2-D run does not fill a CPU, and on a GPU its time goes into starting
kernels rather than updating cells. `lightsim.run_many` runs a list of
simulations together and returns their results in the order given, each one
bit for bit what `sim.run(device)` returns:

```python
results = ls.run_many(sims)                       # CPU: one simulation per core
results = ls.run_many(sims, device="cuda:0", workers=4)   # four at a time on one GPU
results = ls.run_many(sims, device=["cuda:0", "cuda:1"])
```

On the CPU every simulation runs on one core and as many run at a time as
there are cores (`RAYON_NUM_THREADS` when set; `workers` to use fewer). On a
GPU `workers` simulations share the card at a time (default one). This is
for populations of designs, Monte Carlo over fabrication and data sets;
a simulation large enough to use every core by itself gains nothing.

A simulation that fails does not stop the others. The call then raises
`lightsim.RunManyError`, whose `results` holds the results of the others
(`None` where one failed) and whose `errors` maps each failed position to
its error; `run_many(sims, on_error="return")` returns the list with the
error in the place of each failed simulation instead.

## Other settings

| Field | Meaning |
|---|---|
| `courant` | Time step as a fraction of the stability limit. The default 0.99 rarely needs changing. |
| `shutoff`, `shutoff_interval` | Early stop when the fields have decayed, see [Concepts](concepts.md). |
| `name` | A label stored with the simulation. |

`Simulation` methods return the simulation, so calls can be chained:
`sim.add_medium(m).add_structure(shape, "name").add_source(s)`. Adding a
medium whose name is already defined is ignored.
