# Shape optimisation

`lightsim.invdes.shapes` optimises a few shape parameters, such as the
vertices of a polygon or the widths of a taper, instead of a free pixel
density. It is the same kind of capability as the parametrised-shape
optimisation of Lumerical's lumopt, reached by a different route: lumopt
moves the boundary and evaluates a boundary integral of the fields; lightsim
draws the shape onto the density grid of an [inverse design](invdes.md),
takes the permittivity gradient that the existing adjoint engines return,
and pulls it back to the shape parameters through the exact transpose of the
drawing step:

    parameters --shape--> density on the design grid --> permittivity --> S --> objective
    d objective / d parameters  <--  transpose of each step  <--  d objective / d eps

Everything after the density is the density chain of
[`lightsim.invdes`](invdes.md), unchanged: the `python` and `native`
engines, one-level (`Design.slab`, `Design.effective_index`) and two-level
(`Design.partial_etch`, `effective_index(n_partial=...)`) designs, the
objectives, `InverseDesign` and `optimize`. Nothing in the engine changed.

## What it does

### Polygons with movable vertices

`Polygon(vertices, free, mirror, axis, bounds, field)` is a closed polygon
whose free vertex coordinates are parameters. The density at a sample of
the design grid is the exact fraction of the sample's pixel cell (the pixel
rectangle centred on it) that the polygon covers: an anti-aliased
rasterisation. Per cell, the area is the sum over the edges of the signed
area between the edge and the bottom of the cell (Green's theorem column by
column), so it is exact for any simple polygon, including the parts that lie
outside the design rectangle (`polygon_density`).

The covered area is a continuous, piecewise-quadratic function of the
vertex coordinates, so the objective is smooth in them. Its derivative with
respect to a vertex is the boundary integral, over the two edges that meet
at the vertex, of the outward normal times the vertex's hat function, clipped
to each cell (`polygon_density_vjp`). This is the exact derivative of the
rasterised area: the transpose agrees with central differences of the
rasteriser to better than 1e-8 relative.

- `free`: `True`, a mask of free vertices, or a mask of free coordinates.
  Fixed vertices keep their position (the points where the outline meets
  the waveguides).
- `mirror="y"` with `axis=a`: the vertices are one half of the outline, an
  open chain from the line y = a back to it; the full polygon adds its mirror
  image. `mirror="x"` mirrors about x = a.
- `Polygon.from_function(fn, p0, bounds, jac=None)`: the vertices are
  `fn(p)` for a smaller parameter vector, for example a spline through a few
  control points. Without `jac` the Jacobian of `fn` is a central difference
  (exact to rounding for maps linear in `p`).
- Several polygons can be drawn into one density field if they do not
  overlap; `field=1` draws into the upper level of a two-level design.

### Level sets

`LevelSet(phi, p0, bounds, jac=None, width=None)` draws `rho = H(phi(p, X,
Y) / width)`, with `H` a C1 smoothed step from -1 to 1 (`smooth_heaviside`)
and `phi` a signed function, positive inside, that should be close to a
signed distance near the boundary. It suits shapes of fixed topology whose
boundary is described by a function. `LevelSet.width_profile(stations,
widths, bounds, free, center, kind)` is the common case: a strip whose width
w(x) interpolates the widths at a few stations (natural cubic spline or
linear), `phi = w(x) / 2 - |y - center|`. The default `width` is one pixel,
so the boundary is spread over two pixels.

### The design object

`ShapeDesign(design, shapes, projection=False)` puts the shapes on the grid
of a `Design` and has the interface that `InverseDesign` and `optimize`
expect from a design (`eps`, `vjp`, `grids`, `binarization`):

- The parameter vector `u` that the optimiser sees is normalised to `[0,
  1]` between per-parameter bounds, `p = lo + u (hi - lo)`. `optimize`
  already keeps its variable in `[0, 1]` (projected Adam, or L-BFGS-B with
  bounds), so the bounds are hard constraints. Polygon coordinates without
  bounds are bounded by the design rectangle.
- With `projection=False` (default) the density goes straight into the
  level fields, `eps = eps_base + sum_l coeff_l u_l` with `u_0 = rho_0`,
  `u_1 = rho_0 rho_1`, and `beta` and `eta` are ignored. With
  `projection=True` it goes through the design's own filter and projection.
- `density(u)`, `density_vjp(u, g)`, `params(u)`, `normalize(p)`,
  `initial()`, and `polygons(u)` (the outlines in metres, for a layout).

A minimum feature size is imposed through the bounds: the lower bound of a
width in `width_profile` is the narrowest the strip can become. For general
polygons there is no minimum-feature constraint beyond the vertex bounds.

## Example

A taper from a 0.5 um to a 2.5 um silicon waveguide over 2 um, in the 2.5-D
model on the CPU. Seven widths of a cubic width profile are free, bounded
below by 0.3 um; fifteen Adam iterations raise the transmission into the
fundamental mode of the wide guide from 0.76 (linear taper) to 0.94.

```python
import numpy as np
from lightsim import eim, invdes
from lightsim.spec import C0, Box, Medium, um

lam = 1.55 * um
n_bg, n_si = 1.444, eim.slab_neff(3.476, 220e-9, lam, 1.444)
w1, w2, x0, x1 = 0.5 * um, 2.5 * um, 1.5 * um, 3.5 * um
sim = invdes.planar_simulation([0, -2.2 * um], [5 * um, 2.2 * um], lam, n_bg, n_si,
                               steps_per_wavelength=14, run_time=600e-15, shutoff=1e-6)
sim.add_medium(Medium("core", n=n_si))
sim.add_structure(Box([-5 * um, -w1 / 2, -1], [x0 + 0.1 * um, w1 / 2, 1]), "core")
sim.add_structure(Box([x1 - 0.1 * um, -w2 / 2, -1], [20 * um, w2 / 2, 1]), "core")
p_in = invdes.port_mode(sim, "in", 0, 0.8 * um, [0.8 * um, -1.2 * um, 0], [0.8 * um, 1.2 * um, 1 * um], C0 / lam, "+", "te")
p_out = invdes.port_mode(sim, "out", 0, 4.3 * um, [4.3 * um, -2 * um, 0], [4.3 * um, 2 * um, 1 * um], C0 / lam, "-", "te")

design = invdes.Design.effective_index([x0, -1.6 * um], [x1, 1.6 * um], n_bg, n_si, pixel=40e-9)
stations = np.linspace(x0, x1, 9)
profile = invdes.LevelSet.width_profile(stations, np.linspace(w1, w2, 9), bounds=(0.3 * um, 3 * um),
                                        free=[0, 1, 1, 1, 1, 1, 1, 1, 0])
shapes = invdes.ShapeDesign(design, [profile])
problem = invdes.SMatrixProblem(sim, shapes.grids(), [p_in], [p_out], [C0 / lam], devices="cpu")
inv = invdes.InverseDesign(problem, shapes, objective=invdes.mean_coupling)
u, history = invdes.optimize(inv, shapes.initial(), 15, lr=0.02, verbose=False)
print(np.round(history["fom"][[0, 4, 9, 14]], 3))
print(np.round(profile.all_widths(shapes.params(u)) / um, 2))
```

```text
[0.756 0.89  0.922 0.942]
[0.5  1.26 1.1  1.13 1.47 1.7  1.58 1.58 2.5 ]
```

The same region with a polygon outline instead: half of it is given, from
the axis to the axis, mirrored about y = 0; the end vertices are fixed and
the three inner vertices are free, so there are six parameters. One forward
and one adjoint run give the derivative of |S21|^2 with respect to every
coordinate.

```python
half = np.array([[1.3, 0], [1.3, 0.25], [2.0, 0.6], [2.6, 0.9], [3.0, 1.1], [3.7, 1.25], [3.7, 0]]) * um
outline = invdes.Polygon(half, free=[0, 0, 1, 1, 1, 0, 0], mirror="y")
shapes2 = invdes.ShapeDesign(design, [outline])
problem2 = invdes.SMatrixProblem(sim, shapes2.grids(), [p_in], [p_out], [C0 / lam], devices="cpu")
inv2 = invdes.InverseDesign(problem2, shapes2, objective=invdes.mean_coupling)
value, grad, info = inv2.value_and_grad(shapes2.initial(), 1.0)
dp = grad / (shapes2.hi - shapes2.lo)          # d|S21|^2 / d(coordinate), per metre
print(round(value, 3), np.round(dp * um, 3), outline.vertices(shapes2.params(shapes2.initial())).shape)
```

```text
0.76 [ 0.096 -0.192  0.26  -0.52   0.095 -0.279] (12, 2)
```

## API

```text
from lightsim import invdes          # Polygon, LevelSet, ShapeDesign are also in lightsim.invdes.shapes

poly = invdes.Polygon(vertices, free=True, mirror=None, axis=0.0, bounds=None, field=0, check=True)
poly = invdes.Polygon.from_function(fn, p0, bounds, jac=None, mirror=None, axis=0.0, field=0)
lset = invdes.LevelSet(phi, p0, bounds, jac=None, width=None, field=0)
lset = invdes.LevelSet.width_profile(stations, widths, bounds, free=None, center=0.0, kind="cubic", axis="x")

shapes = invdes.ShapeDesign(design, [poly, lset], projection=False)
u0 = shapes.initial()                       # normalised parameters in [0, 1]
shapes.params(u), shapes.normalize(p)       # physical <-> normalised
shapes.density(u), shapes.density_vjp(u, g) # density on the design grid and its transpose
shapes.eps(u), shapes.vjp(u, beta, grads)   # the Design interface
shapes.polygons(u)                          # [(field, vertices)] for a layout

inv = invdes.InverseDesign(problem, shapes, objective=invdes.mean_coupling)
u, history = invdes.optimize(inv, u0, 30, lr=0.02)        # or method="lbfgs"

from lightsim.invdes import shapes as sh
sh.polygon_density(vertices, x, y), sh.polygon_density_vjp(vertices, x, y, g), sh.polygon_is_simple(vertices)
```

## Validation

`python/tests/test_shapes.py` (CPU by default, `LIGHTSIM_DEVICE=cuda:0` for
a GPU). Simulated derivatives are compared with central differences of the
simulated objective, one parameter at a time (two extra simulations per
parameter), with a step of 2 nm on a 40 nm cell and a 40 nm design pixel.
The error is stated as |adjoint - finite difference| divided by the largest
finite-difference derivative of the set, the measure the S-matrix tests use.

| check | result |
| --- | --- |
| area fractions against 1600-point supersampling per cell; total area against the shoelace area | within the sampling error; 1e-12 |
| `density_vjp` against central differences: free, fixed vertices, fixed coordinates, mirror x and y, spline map, width profiles, a disc level set (9 cases) | 1e-5 relative or better |
| shapes -> eps transpose: slab, partial etch, three-level 2.5-D, through the filter and projection | 1e-5 relative or better |
| 2.5-D taper outline, mirrored polygon, 6 vertex coordinates, `python` and `native` engines | 0.06 - 0.67 % |
| 2.5-D taper, cubic width profile, 3 widths | 0.16 - 0.25 % |
| 3-D bulge on a 0.8 x 0.4 um SiN strip, full etch, 3 vertex coordinates | 0.2, 1.1, 2.3 % |
| 3-D, partial etch, polygons on both levels, 3 vertex coordinates | 1.5, 1.2, 4.0 % |
| taper 0.5 -> 2.5 um over 2 um, 7 widths >= 0.3 um, 30 Adam iterations, 3 wavelengths | mean abs(S21)^2 0.756 -> 0.951 |

The numbers are the same on the CPU and on an RTX PRO 6000 to the digits
shown (the 2.5-D finite differences differ in the fourth digit). The whole
file takes 17 s on an RTX PRO 6000 and about 4 minutes on a CPU, most of it
the 3-D finite differences.

The 2.5-D derivatives agree to better than 1 %. The 3-D ones reach 4.0 %,
and the test allows 4.5 %. These 3-D differences are not a property of the
shape layer: its transposes are exact to 1e-5, and the derivative it passes
on is the engine's permittivity gradient along the direction d rho / d p.
They do not shrink with the finite-difference step (1, 2 and 5 nm give the
same values), with a longer run (500 fs, shutoff 1e-8), with a finer cell
(25 and 20 nm) or with a finer design pixel (20 nm), and they grow when the
index contrast falls (up to 6 % for a core of index 1.6).

They come from the adjoint formula of the S-matrix engines. Compared with
reverse mode, which differentiates the discrete scheme exactly, that formula
is a few per cent off the local value, smoothly across the design and with a
sign that alternates along the propagation axis
([measurements](autodiff.md)). A sum over a whole design averages this out
(0.13 % for the smooth directions of `test_invdes.py` in 3-D). A vertex
derivative is a sum along a boundary, where the values themselves partly
cancel and the errors do not. The `python` and `native` engines return the
same derivative (to 1e-7). The gradients are accurate enough to drive an
optimisation; a final design should be checked by simulation, as for
densities.

The optimisation history of the taper rises monotonically for 25
iterations to 0.952 and then oscillates within 0.0012 of it (Adam with a
fixed step); 30 iterations take 6 s on a CPU and 9 s on an RTX PRO 6000
(a 2.5-D scene this small does not load a GPU).

## Scope and limits

- Simple polygons only. When two edges cross (a vertex moved past another
  edge), `Polygon.density` raises `ValueError`, because the area formula
  then counts winding numbers. Keep vertices apart with bounds, a fixed
  vertex order along a coordinate, or a parametrisation such as a width
  profile that cannot fold. Overlapping shapes in one field raise
  `ValueError` as well.
- The area fraction is C1 in the vertices except where an edge is exactly
  parallel to a grid axis: moving such an edge across a cell boundary
  changes the slope of the cell's area. The objective stays continuous; a
  finite difference across that point averages the two slopes.
- The grid must resolve the shape: a feature narrower than about two pixels
  is drawn as grey levels, which the index grid interpolates linearly in
  permittivity, and the linear average across an edge is larger than the
  smoothed permittivity a field normal to the edge sees (see the limits of
  [`lightsim.invdes`](invdes.md)). Use a design pixel no larger than the
  cell and verify the final outline as polygons (`ShapeDesign.polygons` ->
  GDS -> `lightsim.sparams`), where sub-pixel smoothing applies.
- The density is drawn on the 2-D design grid and extruded through the
  layer: vertical sidewalls only, no sidewall angle.
- Topology is fixed. A polygon keeps its vertex count; a level set keeps
  the topology its parametrisation allows (a width profile cannot split).
  For topology changes use the density parametrisation.
- `phi` of a level set should be close to a signed distance near the
  boundary; `width_profile` uses `w(x)/2 - |y - c|`, which is one only for
  small slopes of w.
- With `projection=False`, `InverseDesign(robust=...)` gives three identical
  variants (the threshold is not used); eroded and dilated shapes are not
  generated.
- No minimum-radius-of-curvature or minimum-gap constraint for polygons;
  only the bounds of each coordinate.
