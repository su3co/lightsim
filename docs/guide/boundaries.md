# Boundaries

After this page you can choose the boundary condition on each face of the
simulation region: absorbing layers, periodic and Bloch-periodic walls,
perfect electric and magnetic walls. You can also use mirror symmetry to
simulate a fraction of the domain.

## Setting boundaries

Every face has its own condition. The default is an absorbing layer on all
six faces. `set_boundaries` changes one axis at a time; a single value applies
to both faces of the axis and a pair sets the low and the high face
separately.

| Condition | Value | Effect |
|---|---|---|
| Absorbing layer | `PML()` | Outgoing waves leave without reflection |
| Periodic | `PERIODIC` | The field repeats with the period of the region |
| Bloch-periodic | `Bloch(k)` | The field repeats up to a phase |
| Perfect electric conductor | `PEC_WALL` | Tangential electric field is zero on the face |
| Perfect magnetic conductor | `PMC_WALL` | Tangential magnetic field is zero on the face |

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=50e-15)
sim.set_boundaries(x=ls.PML(layers=16), y=ls.PERIODIC, z=[ls.PEC_WALL, ls.PML()])
print([len(a) - 1 for a in sim.grid()])
```

```text
[72, 20, 32]
```

The region is 40 by 20 by 20 cells. The absorbing layers add 16 cells on each
x face and 12 on the upper z face.

Periodic and Bloch conditions must be set on both faces of an axis. The other
conditions can be mixed freely.

## Absorbing layers (PML)

A perfectly matched layer (PML) is an artificial absorbing medium designed
not to reflect, whatever the angle and frequency of the incoming wave.
lightsim uses the convolutional formulation (CPML).

| Parameter | Meaning | Default |
|---|---|---|
| `layers` | Thickness in cells | 12 |
| `order` | Polynomial order $m$ of the absorption profile | 3 |
| `sigma_scale` | Peak absorption relative to the standard optimum | 1 |
| `kappa_max` | Peak coordinate stretching $\kappa$ | 2 |
| `alpha_max` | Peak of the frequency-shift term $\alpha$ | 0 |
| `inside` | Put the layers on the outermost cells of the region | off |
| `stabilize` | Damp, inside the layers, the poles of dispersive media that lie far below the band of the sources | on |

With $u$ the depth into the layer as a fraction of its thickness, $\Delta$
the cell size at the boundary and $s$ the value of `sigma_scale`, the
conductivity $\sigma$, the stretching $\kappa$ and the shift $\alpha$ (a
conductivity, in S/m) follow

$$
\sigma(u) = s\, \frac{0.8\,(m+1)}{\eta_0\,\Delta}\, u^m,
\qquad \kappa(u) = 1 + (\kappa_{\max} - 1)\, u^m,
\qquad \alpha(u) = \alpha_{\max}\,(1 - u) .
$$

Points to know:

- The layers are added **outside** the region, with the cell size found at
  the boundary. They enlarge the grid; they do not take space from the
  region. With `inside=True` they occupy the outermost cells of the region
  instead, and the grid has exactly the size of the region.
- A structure continues into the layer only if its shape does. A waveguide
  that should leave the domain must be drawn longer than the region, so that
  it runs through the layer. If it stops at the region boundary the layer
  sees an abrupt end and reflects.
- A PML absorbs propagating waves. Evanescent fields that reach it are also
  damped, which shows up as loss. Keep the layer away from structures that
  carry evanescent tails. As an example, a weakly guiding nitride waveguide
  (800 by 400 nm, effective index 1.60 at 1.55 µm) with the layers 0.6 µm
  from its faces lost 6 % of its power over 1.6 µm. With the layers 1.2 µm
  away the loss was 0.2 %.
- The default 12 layers reflect about $2 \times 10^{-5}$ of the power at
  normal incidence (see [Validation](../validation.md)). Increase `layers`
  for waves at grazing incidence.

### Dispersive media in the layers

A structure made of a dispersive medium keeps its poles inside the layer, so
the layer is matched to it at every frequency. One case needs care. A fit of
a glass over a telecom band (`fitted_medium("SiO2", ...)`, or any Sellmeier
formula) has a pole in the far infrared, with no damping. Between that pole
and a frequency somewhat above it the model has a negative permittivity: it
behaves like a metal there. A dielectric core clad in such a medium then
carries modes whose phase and energy travel in opposite directions (as in a
metal-insulator-metal guide), and a PML amplifies such backward waves instead
of absorbing them. Nothing drives these modes, but rounding noise is enough:
a waveguide that runs into the layer made a long run diverge after some tens
of thousands of steps, at a wavelength of 10 to 20 µm, far from the sources.

With `stabilize` on, which is the default, lightsim damps these poles
**inside the layers only**, and only when the region holds a structure of
another medium than the background (a uniform medium has no such modes, and
its layers are left as they are). A pole at $\omega_0$ of strength
$\Delta\varepsilon$ in a medium with high-frequency permittivity
$\varepsilon_\infty$ gets the damping rate

$$
\gamma = \omega_0 \left( \sqrt{1 + 2 N \Delta\varepsilon / \varepsilon_\infty} - 1 \right)
$$

($N$ is the number of such poles in the medium), at which the real part of
the permittivity never falls below $\varepsilon_\infty / 2$: the medium has
no metal-like band left, and the layer absorbs as it does for a plain
dielectric. Outside the layers nothing changes.

- Only Lorentz poles below the band of the sources are treated: below
  three widths under the centre frequency of every pulse. Poles in or near
  the band, ultraviolet poles, Drude poles (metals) and Debye poles stay as
  they are. With a sampled signal, or without sources, the band is not known
  and nothing is treated.
- A pole is treated only if the reflection this causes stays below
  $10^{-4}$ at the low end of the band (two widths under the centre
  frequency, where it is largest), that is, if the damping changes the
  permittivity there by less than 4 %. Fits of the library materials over
  bands up to an octave around 1.5 µm pass this test; a pole just below the
  band does not.
- The price is a small step in the permittivity at the face of the layer,
  mostly absorption, which falls as the cube of the frequency: for fused
  silica around 1.55 µm it is about 0.003, which reflects $10^{-7}$ of the
  power. In a test with a nitride-like core
  in fitted silica the reflection of the guided mode from the layers was the
  same with and without the damping to 0.3 % of its value
  ($7.7 \times 10^{-5}$ on that coarse mesh), and a run of 40 000 steps that
  diverged at step 13 000 without it stayed stable. In 3-D, a guide through
  the layers that diverged at step 26 000 ran 150 000 steps on an RTX PRO
  6000 with fused and with split kernels.
- Poles above the band are not treated yet. Glasses and nitride, whose
  ultraviolet poles are weak, ran 150 000 steps in 3-D and 300 000 in 2-D
  without growth, and so did a silicon strip fitted with one strong
  ultraviolet pole (`fitted_medium("Si")`, 100 000 steps). A low-index slot
  between walls of that fitted silicon, running into the layers, diverged
  after about 100 000 steps: for long runs of slot guides give silicon a
  constant index.
- `PML(stabilize=False)` gives the plain layers. A metal that runs into a
  layer is not covered by this: keep metals out of the layers, or end them
  before the boundary.

## Periodic and Bloch boundaries

`PERIODIC` makes the field at one face continue at the opposite face. It
models one unit cell of an infinite periodic structure, or, with a cell only
a few grid cells wide, a structure that is uniform in that direction.

`Bloch(k)` adds a phase: with $L$ the length of the region along the axis,

$$
F(\mathbf{r} + L\,\hat{\mathbf{a}}) = F(\mathbf{r})\, e^{i k L} .
$$

It is needed when a periodic structure is illuminated at an angle. For a
plane wave incident at angle $\theta$ in a medium of index $n$ the in-plane
wavevector is $k = n\,k_0 \sin\theta$ with $k_0 = 2\pi f_0/c$.

Points to know:

- A Bloch run with $k \neq 0$ stores complex fields. It needs twice the
  memory of a real run and is less than half as fast (see
  [Benchmarks](../benchmarks.md)). `Bloch(0)` is the same as `PERIODIC`.
- $k$ is fixed, so the angle of incidence depends on frequency:
  $\sin\theta(f) = k c / (2\pi f n)$. Use a narrow pulse, or run one
  simulation per angle and frequency.
- Bloch boundaries cannot be combined with nonlinear media or with media that
  have off-diagonal tensor terms.
- With an oblique plane-wave source, `Results.source_spectrum` returns the
  spectrum of the complex signal that such a run injects, so fields and
  powers are normalised by it as at normal incidence (see
  [Sources](sources.md)).

### Worked example: reflection at oblique incidence

A plane wave strikes a glass surface at 30°, with the electric field
perpendicular to the plane of incidence (s polarisation). The reflectance
should follow the Fresnel formula

$$
R_s = \left( \frac{\cos\theta - \sqrt{n^2 - \sin^2\theta}}{\cos\theta + \sqrt{n^2 - \sin^2\theta}} \right)^2 .
$$

The wave travels along x and is tilted towards y. The structure does not
vary along y or z, so a few cells suffice in those directions.

```python
lam0 = 1.0 * um
f0 = ls.C0 / lam0
k0 = 2 * np.pi / lam0
dx = lam0 / 40
theta = np.radians(30)
n_glass = 1.5


def cell(with_glass):
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 4 * dx, 2 * dx],
                        mesh=ls.UniformMesh(dx), run_time=300e-15, shutoff=1e-6)
    sim.set_boundaries(y=ls.Bloch(k0 * np.sin(theta)), z=ls.PERIODIC)     # x stays PML
    if with_glass:
        sim.add_structure(ls.Box([2 * um, -1, -1], [1, 1, 1]), ls.Medium("glass", n=n_glass))
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ez", ls.GaussianPulse(f0, 0.05 * f0), angle=theta))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.0 * um, [3.0 * um, -1, -1], [3.0 * um, 1, 1], [f0]))
    sim.add_monitor(ls.FluxMonitor("R", 0, 0.5 * um, [0.5 * um, -1, -1], [0.5 * um, 1, 1], [f0]))
    return sim


incident = cell(False).run(device="cpu").flux("T")[0]
res = cell(True).run(device="cpu")
T = res.flux("T")[0] / incident
R = -res.flux("R")[0] / incident          # behind the source: reflected light only, flowing towards -x

c1, c2 = np.cos(theta), np.sqrt(n_glass**2 - np.sin(theta) ** 2)
R_exact = ((c1 - c2) / (c1 + c2)) ** 2
print(f"R = {R:.4f} (Fresnel {R_exact:.4f}), T = {T:.4f} (Fresnel {1 - R_exact:.4f}), R + T = {R + T:.5f}")
```

```text
R = 0.0569 (Fresnel 0.0578), T = 0.9431 (Fresnel 0.9422), R + T = 1.00000
```

The glass box extends beyond the region on the far side, so it continues into
the absorbing layer and the transmitted wave leaves without reflection.

## Electric and magnetic walls

`PEC_WALL` forces the tangential electric field to zero on the face: a
perfect mirror. `PMC_WALL` forces the tangential magnetic field to zero.
Their main use is as symmetry planes: if a problem has a mirror plane, the
field on one side determines the other, and a wall on the plane lets you
simulate one half.

| On the mirror plane | Wall |
|---|---|
| Tangential electric field is even (mirror-symmetric) | `PMC_WALL` |
| Tangential electric field is odd (antisymmetric) | `PEC_WALL` |

## Automatic symmetry reduction

`run_symmetric(sim, (sx, sy, sz))` applies this for you. You describe the
full problem. It cuts the domain at the mid-plane of every axis with a
non-zero entry, keeps the upper half, puts the right wall on the cut, runs
the smaller simulation and unfolds the results to the full domain.

| Entry | Meaning |
|---|---|
| `+1` | Tangential electric field even about the mid-plane (magnetic wall) |
| `-1` | Tangential electric field odd about the mid-plane (electric wall) |
| `0` | No symmetry along this axis |

The example is a z-polarised dipole on the z axis with a dielectric sphere
above it. The arrangement is symmetric under reflection in the planes
$x = 0$ and $y = 0$, and the dipole field is tangential to both planes and
even:

```python
half = 0.8 * um
sim = ls.Simulation(min=[-half] * 3, max=[half] * 3, mesh=ls.UniformMesh(40e-9), run_time=30e-15)
sim.add_structure(ls.Sphere([0, 0, 0.2 * um], 0.25 * um), ls.Medium("bead", n=1.8))
sim.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(f0, 0.3 * f0), amplitude=1e-12))
sim.add_monitor(ls.FluxBoxMonitor("power", [-0.5 * um] * 3, [0.5 * um] * 3, [f0]))
sim.add_monitor(ls.DFTMonitor("plane", [-0.6 * um, -0.6 * um, 0.3 * um], [0.6 * um, 0.6 * um, 0.3 * um],
                              [f0], components=("Ez",)))

full = sim.run(device="cpu")
reduced = ls.run_symmetric(sim, (1, 1, 0), device="cpu")

print(full.stats["cells"], reduced.stats["cells"])
print(abs(reduced.flux("power")[0] / full.flux("power")[0] - 1) < 1e-4)
a, b = full.field("plane", "Ez"), reduced.field("plane", "Ez")
print(a.shape == b.shape, float(np.max(np.abs(a - b)) / np.max(np.abs(a))))
```

```text
262144 65536
True
True 0.0
```

The reduced run has a quarter of the cells and returns the same fields.

Requirements and limits:

- The region, the mesh, the structures and the sources must be
  mirror-symmetric, and a grid node must lie on each mid-plane. The function
  checks the mesh and the boundaries and raises an error if they are not
  symmetric. It does **not** check the structures or the sources.
- The two boundaries of a reduced axis must be equal.
- Unfolding covers `DFTMonitor` fields, `FluxMonitor` and `FluxBoxMonitor`
  values. A flux plane normal to a reduced axis must lie in the upper half.
- Point sources in the removed half are dropped; a source on the mid-plane is
  kept. Time-domain monitors are not unfolded, so place probes in the upper
  half. Mode sources and mode monitors are not supported, because their
  profiles are computed for the full plane.
