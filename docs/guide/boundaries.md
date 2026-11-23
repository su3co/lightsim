# Boundaries

After this page you can choose the boundary condition on each face of the
simulation region: absorbing layers (PML or adiabatic absorbers), periodic
and Bloch-periodic walls, perfect electric and magnetic walls. You can also use mirror symmetry to
simulate a fraction of the domain.

## Setting boundaries

Every face has its own condition. The default is an absorbing layer on all
six faces. `set_boundaries` changes one axis at a time; a single value applies
to both faces of the axis and a pair sets the low and the high face
separately.

| Condition | Value | Effect |
|---|---|---|
| Absorbing layer | `PML()` | Outgoing waves leave without reflection |
| Adiabatic absorber | `Absorber()` | Outgoing waves decay in a thicker, passive conducting layer |
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
conditions can be mixed freely, PML and absorbers included.

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
| `stabilize` | Inside the layers, damp the poles of dispersive media far below the band of the sources and freeze those far above it | on |

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
the layer is matched to it at every frequency. One case needs care. A
lossless pole makes its medium metal-like in a band just above its
resonance: the permittivity is negative there. A dielectric guide beside or
inside such a medium then carries modes whose phase and energy travel in
opposite directions (as in a metal-insulator-metal guide), and a PML
amplifies such backward waves instead of absorbing them. Nothing drives
these modes, but rounding noise is enough: a waveguide that runs into the
layer made a long run diverge after some tens of thousands of steps, at a
wavelength far from the sources'. Fits of transparent materials have two
kinds of pole that do this:

- the far-infrared pole of a glass (`fitted_medium("SiO2", ...)`, or any
  Sellmeier formula with an infrared term): the run diverged at a wavelength
  of 13 to 14 µm;
- a strong ultraviolet pole, as in silicon fitted with `eps_inf = 1`
  (`fitted_medium("Si")`, one pole of strength 10.7 at 0.30 µm): the run
  diverged at about 0.1 µm, sooner on finer cells.

With `stabilize` on, which is the default, lightsim changes these poles
**inside the layers only**, and only when the region holds a structure of
another medium than the background (a uniform medium has no such modes, and
its layers are left as they are). Outside the layers nothing changes.

- **Poles below the band are damped.** A pole at $\omega_0$ of strength
  $\Delta\varepsilon$ in a medium with high-frequency permittivity
  $\varepsilon_\infty$ gets the damping rate
  $\gamma = \omega_0 \left( \sqrt{1 + 2 N \Delta\varepsilon / \varepsilon_\infty} - 1 \right)$
  ($N$ is the number of such poles in the medium), at which the real part of
  the permittivity never falls below $\varepsilon_\infty / 2$. In the band
  this adds a little absorption, which falls as the cube of the frequency:
  for fused silica around 1.55 µm the permittivity changes by about 0.003.
- **Poles above the band are frozen.** The value of the pole at the centre
  of the band is added to the high-frequency permittivity and the pole is
  switched off, so the layer medium is non-dispersive there. The change in
  the band is the pole's own dispersion: zero at the centre, about 0.05 at
  the ends of a 12 % band for silicon.
- A pole is treated only if the reflection this causes at the face of the
  layer stays below $10^{-4}$ at both ends of the band (two widths either
  side of the centre frequency of the pulses), that is, if the permittivity
  changes by less than 4 % there. Fits of the library materials over bands
  up to an octave around 1.5 µm pass this test. Poles in or just outside
  the band, Drude poles (metals) and Debye poles stay as they are, and a
  pole is frozen only in a plain medium (no tensor, nonlinearity, gain or
  plug-in). The band is taken from the sources; with a sampled signal, or
  without sources, it is not known and nothing is treated.

Measured in 3-D on an RTX PRO 6000, with the guide through the layers at
both ends and runs of 300 000 steps (150 000 for the first row). The
surfaces are sub-pixel smoothed, as by default:

| Structure | Plain layers | Stabilised layers |
|---|---|---|
| nitride-like core (n = 1.996) in fitted silica, 25 nm cells | diverge at step 26 000 | flat |
| fitted-silicon strip in fitted silica, 25 nm cells | diverge at step 28 000 | flat |
| fitted-silicon strip in fitted silica, 18 nm cells | diverge at step 34 000 | flat |
| silica core in a fitted-silicon cladding, 25 nm cells | flat | flat |
| silica slot (100 by 220 nm) in a fitted-silicon cladding, 25 nm cells | diverge at step 111 000 | flat |
| the same slot on 18 nm cells | diverge at step 148 000 | flat |

The guided mode is absorbed as before. Reflected power at 1.60, 1.55 and
1.50 µm, in 2-D:

| Guide | Stabilised layers | Plain layers |
|---|---|---|
| nitride-like core in fitted silica, 50 nm cells | $6.67 / 7.86 / 9.03 \times 10^{-5}$ | $6.69 / 7.85 / 9.02 \times 10^{-5}$ |
| the same, 25 nm cells | $3.97 / 4.90 / 5.40 \times 10^{-6}$ | $4.02 / 4.88 / 5.39 \times 10^{-6}$ |
| fitted-silicon strip in silica, 25 nm cells | $4.15 / 4.23 / 4.42 \times 10^{-5}$ | $3.61 / 4.23 / 4.95 \times 10^{-5}$ |

The stabilised layers are not a guarantee. A frozen layer is a different
medium from the interior at ultraviolet frequencies: surface modes of the
metal-like band cannot enter it, they bounce between the layers, and their
tails in the first cells of a layer can still gain a little at each
reflection. Which runs grow depends on the structure and on how its
surface is discretised:

- With staircased surfaces (`subpixel=False`) the slot of the last row
  diverged at step 93 000 with stabilised layers (37 000 with plain ones,
  58 000 with 24 stabilised layers), and the silica core in a silicon
  cladding grew slowly.
- In 2-D with one axis periodic, a silicon strip and a silicon-clad guide
  that stayed flat for 450 000 steps with plain layers diverged near step
  334 000 with stabilised ones, and a slot with absorbing layers only at
  its sides grew with both.
- Thicker layers are not a remedy: they helped one case and hurt another.

`PML(stabilize=False)` gives the plain layers. A metal that runs into a
layer is not covered by any of this.

When a run still grows, or for a metal, use an
[adiabatic absorber](#adiabatic-absorbers) on the faces the structure runs
through. It cannot amplify.

## Adiabatic absorbers

`Absorber()` is a second kind of absorbing layer, for the cases where a PML
is unstable. Inside it an electric and a magnetic conductivity rise smoothly
from zero. With $u$ the depth into the layer as a fraction of its thickness
$L$,

$$
\sigma_e = \hat\sigma\, \varepsilon_0\, \varepsilon, \qquad
\sigma_m = \hat\sigma\, \mu_0\, \mu, \qquad
\hat\sigma(u) = s\, \frac{A\,(m+1)\, c}{2\, L\, n_b}\, u^m ,
$$

where $\varepsilon$ is the permittivity of each medium at the centre of the
source band, $n_b$ the index of the background medium there, $A = 6$ and $s$
the value of `sigma_scale`. The two conductivities are matched in every
medium, so every field decays at the rate $\hat\sigma$ in time: the wave
impedance does not change, and a guided mode keeps its profile while it
decays. A plane wave in the background medium is attenuated by $A = 6$
nepers in amplitude ($6 \times 10^{-6}$ in power) on its way to the outer
wall and back, a guided mode in proportion to its group index.

| Parameter | Meaning | Default |
|---|---|---|
| `layers` | Thickness in cells | 40 |
| `order` | Polynomial order $m$ of the conductivity profile | 3 |
| `sigma_scale` | Peak conductivity relative to the default | 1 |

```python
import lightsim as ls
from lightsim.spec import um

sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=50e-15)
sim.set_boundaries(x=ls.Absorber(layers=20), y=ls.PML(), z=ls.PML())
print([len(a) - 1 for a in sim.grid()])
```

```text
[80, 44, 44]
```

**An absorber is a passive medium.** Ohmic and magnetic loss can only
remove energy, whatever the wave. A PML is not passive: it is built to
damp waves whose phase travels outward, and it amplifies those whose phase
travels inward while their energy travels outward. Such backward waves
exist in guides with a metal-like medium (a metal with surface plasmons, a
fitted dielectric in the band above one of its poles) and in some photonic
crystal guides. Every case on this page that grew with a PML, plain or
stabilised, stayed flat with absorbers: 450 000 steps in 2-D, and 300 000
steps in 3-D for the silicon-cladding guides with staircased surfaces,
where the stabilised layers fail.

**It needs thickness.** A PML reflects at its discretisation; an absorber
reflects wherever its conductivity changes, so the change has to be slow
on the scale of the wavelength. One micrometre at 1.55 µm (about one
wavelength in a glass cladding) reflects $10^{-5}$ to $10^{-4}$ of a guided
mode, half of that thickness about $5 \times 10^{-4}$, and one and a half
micrometres as little as the PML. Choose `layers` from the thickness, not
the other way round. Reflected power of the guided mode at 1.60, 1.55 and
1.50 µm, in 2-D with the guide through the layer:

| Layer | Thickness | Nitride-like core (n = 1.996) in fitted silica | Fitted-silicon strip in silica |
|---|---|---|---|
| `PML()`, 12 layers of 25 nm | 0.3 µm | $4.0 / 4.9 / 5.4 \times 10^{-6}$ | $4.1 / 4.2 / 4.4 \times 10^{-5}$ |
| `Absorber(layers=20)`, 25 nm | 0.5 µm | $6.5 / 7.7 / 5.2 \times 10^{-4}$ | $3.9 / 2.0 / 1.5 \times 10^{-4}$ |
| `Absorber(layers=40)`, 25 nm | 1.0 µm | $4.5 / 1.5 / 0.8 \times 10^{-5}$ | $2.8 / 3.6 / 5.0 \times 10^{-5}$ |
| `Absorber(layers=60)`, 25 nm | 1.5 µm | $6.0 / 2.7 / 6.4 \times 10^{-6}$ | $3.5 / 4.2 / 4.9 \times 10^{-5}$ |
| `Absorber(layers=40)`, 12.5 nm | 0.5 µm | $8.0 / 6.6 / 5.9 \times 10^{-4}$ | $5.0 / 4.8 / 2.4 \times 10^{-4}$ |
| `Absorber(layers=80)`, 12.5 nm | 1.0 µm | $3.1 / 2.2 / 0.7 \times 10^{-5}$ | $2.3 / 3.3 / 2.9 \times 10^{-6}$ |
| `Absorber(layers=20)`, 50 nm | 1.0 µm | $1.4 / 0.9 / 0.5 \times 10^{-4}$ | |

(The silicon strip's values at 25 nm are at the floor of that
measurement, which the PML row shows.)

**Cost.** An absorber has more cells than a PML, and each costs more on a
GPU: a run with an absorber uses the split kernels instead of the fused
step. On an RTX PRO 6000, 600 steps of a glass box of 320 cells a side
took 1.0 s with a 12-layer PML on every face (24 Gcells/s) and 3.6 s with
40-layer absorbers (10.7 Gcells/s on a grid 1.6 times as large); at equal
thickness the absorber ran 1.8 times slower per cell. A small 3-D grid in
which every cell is dispersive ran about four times slower per cell,
because the dispersive update then works from cell lists. An absorber also
stores a 12-byte entry per magnetic field sample in its layers. Put
absorbers on the faces that need them and PML on the others.

**Strength.** A stronger absorber (`sigma_scale` above 1) reflects more at
its graded conductivity; a weaker one lets more through to the outer wall
and back. The default sits where the two meet for layers about a
wavelength thick. Thicker layers do not need a different value.

**Evanescent tails** that reach an absorber are damped, as in a PML: keep
it as far from a guide's side faces as you would a PML.

**What it takes.** Media with a diagonal permittivity, conductivity and
poles can run through an absorber; a tensor, nonlinear, gain or plug-in
medium that reaches into one is refused with an error, and so are
zero-thickness sheets. The fixed-angle engine and the `reverse` gradient
engine need PML, and design regions must stay outside an absorber. In 2-D
a run with an absorber uses the general kernels, about three times slower
per cell than the 2-D kernels.

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
