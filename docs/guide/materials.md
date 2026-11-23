# Materials

After this page you can define dielectric, absorbing, dispersive, anisotropic,
magnetic and nonlinear media, use the built-in materials and the dispersion
library, fit your own refractive-index data, and import a spatially varying
index.

## Defining a medium

`Medium(name, ...)` describes a homogeneous material. A structure or the
background refers to it by `name`.

| Parameter | Meaning | Default |
|---|---|---|
| `n` | Refractive index. One value, or three for the x, y, z axes | none |
| `eps` | Relative permittivity, as an alternative to `n` | 1 |
| `mu` | Relative permeability, one or three values | 1 |
| `sigma` | Electric conductivity in S/m, one or three values | 0 |
| `poles` | Dispersive poles: `Lorentz`, `Drude`, `Debye` | none |
| `pole_axes` | Axes the poles act on, three booleans | all |
| `oriented` | Dispersive poles acting along a given direction | none |
| `eps_off`, `sigma_off` | Off-diagonal terms (xy, xz, yz) of symmetric tensors | 0 |
| `rotation` | 3 by 3 rotation matrix applied to the diagonal `eps` and `sigma` | none |
| `chi2`, `chi3` | Second- and third-order susceptibility, one or three values | 0 |
| `raman` | Delayed (Raman) part of `chi3`: `Raman(fraction, omega, delta)` | none |
| `gyro` | Gyrotropic (magneto-optic) poles: `[(pole, bias), ...]` | none |
| `director` | Uniaxial medium with a director field (liquid crystal): `DirectorField(...)` | none |
| `gain` | Four-level atomic gain medium with pumping: `FourLevel(...)` | none |
| `pec` | Perfect electric conductor | off |

If both `n` and `eps` are given, `n` is used. When a medium has poles, `eps`
is the permittivity at infinite frequency, $\varepsilon_\infty$.

`add_structure(shape, medium)` accepts the medium itself and registers it. A
medium that is only used as the background must be added with `add_medium`.

## Built-in media

These names can be used without defining anything. They are matched without
regard to case. A medium you define with the same name takes precedence.

| Names | Medium |
|---|---|
| `vacuum`, `air` | $n = 1$ |
| `si`, `silicon` | $n = 3.476$ (crystalline silicon at 1550 nm) |
| `sio2`, `silica`, `oxide` | $n = 1.444$ (thermal oxide at 1550 nm) |
| `si3n4`, `sin`, `nitride` | $n = 1.996$ (LPCVD nitride at 1550 nm) |
| `si3n4_785`, `sin_785` | $n = 2.012$ (nitride at 785 nm) |
| `water`, `h2o` | $n = 1.33$ |
| `au`, `gold` | Drude plus five Lorentz poles (Rakić et al., Appl. Opt. 37, 5271, 1998) |
| `ag`, `silver` | Drude plus five Lorentz poles (same reference) |
| `pec` | Perfect electric conductor |

The dielectrics in this table are constants: they do not change with
wavelength. For broadband work use a fitted medium (below).

## Loss and dispersion

The permittivity of a medium along one axis is

$$
\varepsilon(\omega) = \varepsilon_\infty + \frac{i\sigma}{\varepsilon_0\,\omega}
 + \sum_{\text{Lorentz}} \frac{\Delta\varepsilon\,\omega_0^2}{\omega_0^2 - \omega^2 - i\gamma\omega}
 - \sum_{\text{Drude}} \frac{\omega_p^2}{\omega^2 + i\gamma\omega}
 + \sum_{\text{Debye}} \frac{\Delta\varepsilon}{1 - i\omega\tau} .
$$

| Builder | Parameters |
|---|---|
| `Lorentz(delta_eps, omega0, gamma)` | strength $\Delta\varepsilon$, resonance $\omega_0$ and damping $\gamma$ in rad/s |
| `Drude(omega_p, gamma)` | plasma frequency $\omega_p$ and collision rate $\gamma$ in rad/s |
| `Debye(delta_eps, tau)` | strength $\Delta\varepsilon$ and relaxation time $\tau$ in s |

Each pole adds a polarisation that the solver advances in time alongside the
fields (the auxiliary differential equation method). Points to know:

- The pole update is explicit, and it is stable only below a time step that
  depends on the poles and on $\varepsilon_\infty$. For one Lorentz pole the
  condition is $a + 4\,(\Delta\varepsilon/\varepsilon_\infty)\,b/(4 - b) \le 4$
  with $b = (\omega_0 \Delta t)^2$ and $a = 4S^2/\varepsilon_\infty$, where $S$
  is the fraction of the Courant limit of the mesh; for a Drude pole it is
  $a + (\omega_p \Delta t)^2/\varepsilon_\infty \le 4$. A strong pole on a small
  $\varepsilon_\infty$ therefore needs a step below the Courant limit. **The
  solver reduces the time step automatically** for the media that are in use,
  and `time_info()` shows the result. Built-in gold needs 0.82 of the mesh
  limit at 20 nm cells and 0.92 at 12.5 nm; a medium with all of silicon's
  permittivity in one ultraviolet pole ($\varepsilon_\infty = 1$) needs 0.90 at
  22 nm cells. If a medium would need less than a tenth of the mesh limit, the
  run is refused with an error that names the medium: raise its `eps`, lower
  the pole frequencies, or refine the mesh.
- `sigma` and poles can be combined in one medium: the conduction term adds
  to the pole terms, as in the formula above.
- `sigma` must be positive. A negative conductivity (gain) is ignored.
- Dispersive and conductive media are not sub-pixel smoothed (see
  [Simulation setup](simulation.md) and the section on metal surfaces
  below).

The worked example of this page measures the transmission of a slab for
several kinds of medium. It uses a narrow column with periodic side walls, as
in the [Quick start](quickstart.md):

```python
import numpy as np
import lightsim as ls
from lightsim import materials as mat
from lightsim.spec import um

lam0 = 1.0 * um
f0 = ls.C0 / lam0
w0 = 2 * np.pi * f0
dx = lam0 / 60
d = 0.5 * um                                   # slab thickness
freqs = np.linspace(0.8, 1.2, 9) * f0
slab = ls.Box([2 * um, -1, -1], [2 * um + d, 1, 1])


def column(structures):
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx],
                        mesh=ls.UniformMesh(dx), run_time=400e-15, shutoff=1e-7)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    for shape, medium in structures:
        sim.add_structure(shape, medium)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.5 * um, [3.5 * um, -1, -1], [3.5 * um, 1, 1], freqs))
    return sim


incident = np.asarray(column([]).run(device="cpu").flux("T"))


def transmission(structures):
    return np.asarray(column(structures).run(device="cpu").flux("T")) / incident


def slab_formula(eps):
    """Transmission of a slab of complex permittivity eps(w) in vacuum."""
    n = np.sqrt(np.asarray(eps, dtype=complex))
    r = (1 - n) / (1 + n)
    phi = 2 * np.pi * freqs / ls.C0 * n * d
    return np.abs((1 - r**2) * np.exp(1j * phi) / (1 - r**2 * np.exp(2j * phi))) ** 2
```

A medium with a Lorentz resonance in the middle of the band absorbs strongly
there. `materials.pole_permittivity` evaluates $\varepsilon(\omega)$ of a
medium, which gives the exact transmission to compare with:

```python
absorber = ls.Medium("absorber", eps=2.25, poles=[ls.Lorentz(0.5, w0, 0.05 * w0)])
T = transmission([(slab, absorber)])
T_exact = slab_formula(mat.pole_permittivity(absorber, 2 * np.pi * freqs))
print(np.round(T, 3))
print(np.round(T_exact, 3))
```

```text
[0.573 0.53  0.394 0.059 0.    0.    0.033 0.332 0.582]
[0.574 0.531 0.395 0.06  0.    0.    0.033 0.334 0.583]
```

A conductive medium is checked the same way:

```python
conductor = ls.Medium("lossy glass", eps=2.25, sigma=2000.0)
T = transmission([(slab, conductor)])
T_exact = slab_formula(mat.pole_permittivity(conductor, 2 * np.pi * freqs))
print(np.round(T, 3))
print(np.round(T_exact, 3))
```

```text
[0.722 0.702 0.685 0.675 0.673 0.68  0.694 0.714 0.735]
[0.723 0.703 0.686 0.676 0.674 0.681 0.695 0.714 0.735]
```

### Dispersive surfaces: smoothed for transparent dielectrics, staircased for metals

A cell cut by the surface of a plain dielectric gets an effective
permittivity, which makes results converge with the square of the cell size.
Dispersive dielectrics that are transparent in the band of the sources
(glasses, nitride, silicon from `lightsim.materials.fitted_medium`, any
medium of Lorentz or Debye poles) are smoothed the same way:

- The reference frequency w_c is the mean of the centre frequencies of the
  sources. In a cell cut by such a medium, the accumulators of the smoothing
  get each medium's permittivity at w_c (with the susceptibility of the
  discrete pole update), so the cell's effective permittivity at w_c is the
  one plain-dielectric smoothing would give.
- The cell keeps the poles of one dispersive medium A (the one whose
  weighted susceptibility is largest) with the weight
  $w_A = \partial \varepsilon_\text{eff} / \partial \varepsilon_A =
  f_A\, \varepsilon_\text{eff}^2 [n_a^2/\varepsilon_A^2 + (1 - n_a^2)/\langle\varepsilon\rangle^2]$
  ($f_A$ the filled fraction, $n_a$ the normal component along the field
  component), applied to the cell's polarisation term, and the rest goes
  into the cell's high-frequency permittivity
  $\varepsilon_{\infty,\text{cell}} = \varepsilon_\text{eff}(w_c) - w_A \chi_A(w_c)$.
  The cell is exact at w_c and has A's share of the dispersion, to first order;
  the dispersion of the other medium in that one layer of cells is frozen at
  its value at w_c. No pole and no memory per cell is added, and runs without
  such cells are unchanged.
- A cell falls back to the staircase if a guard fails: weight outside
  [0, 1.05], high-frequency permittivity below 1, or the weighted poles not
  stable at the run's time step on that permittivity. The time step is not
  changed.
- The mode solver (`solve_modes`) smooths these media the same way: at the
  centre frequency of the simulation's sources or, before a source has been
  added (the usual order), at the frequency of the solve. A mode solved at
  the centre frequency of the run's source is therefore the run's own.

Accuracy, against plain-dielectric smoothing of the same media with their
permittivities at the frequency of each result, and against analytic
results:

- The permittivity of every such cell at w_c is that of plain-dielectric
  smoothing to 3e-7, also where abutting structures meet (tests in
  `disp_subpixel.rs`).
- A glass sphere in a glass, both dispersive (validation case
  `dispersive_subpixel`): the scattering cross-section is within 2e-4 of the
  plain-dielectric result at w_c and within 1.1e-3 at 10 % from it, where the
  staircase is off by 4e-3.
- A fitted Si3N4 cylinder (2-D) or sphere (3-D) in fitted SiO2 (band
  1450 to 1650 nm, w_c at 1550 nm), 10 to 40 cells per wavelength in the
  core: the scattering efficiency at 1550 nm is within 2e-4 of
  plain-dielectric smoothing, and its error against the Mie series falls at
  about second order (sphere: 1.9e-2, 8.8e-3, 4.9e-3, 2.4e-3 at 10, 15, 20,
  30 cells, where the staircase gives 3.0e-2, 1.1e-2, 1.0e-2, 3.4e-3). At
  1450 and 1650 nm the error is of the same size and falls in the same way
  (sphere at 30 cells: 2.6e-3 and 2.2e-3): the dispersion frozen in the
  boundary cells does not show at these resolutions.
- A slab guide of the same media tilted by 15 degrees: the effective index
  at 1550 nm is within 5e-5 of plain-dielectric smoothing.
- A 90 degree bend of a 1 um Si3N4 guide in SiO2 (2-D, R = 30 um, 22 nm
  cells, both media fitted): the loss of the fundamental mode at 1550 nm is
  26.680 mdB smoothed and 26.672 mdB with plain dielectrics of the same
  permittivities, against 24.30 mdB staircased.

What is still staircased, and why: metals and any medium with a Drude pole,
conductivity, loss at w_c above 1 % of Re eps, Re eps(w_c) <= 0, poles on only
some axes, off-diagonal, gyrotropic, director, gain, nonlinear and plug-in
media, cells of an imported index grid, all dispersive surfaces when the
sources have no centre frequency (no sources, a sampled signal), and every
surface with `subpixel=False`. For metals the shortcut does not hold:

A correct treatment of a metal surface needs more than blending the two
media. For the field components parallel to a surface, the effective medium
is the average of the two permittivities, which for a dispersive medium
means scaling its poles by the filled fraction of the cell. For the
component normal to the surface it is the average of the inverse
permittivities, and the inverse of a dispersive permittivity is a different
set of poles for every filling fraction, coupled between components where
the surface is oblique to the grid. Linearising at one frequency, as above,
is accurate for a transparent dielectric whose permittivity changes little
over the band; a metal's permittivity changes sign across it and its
normal-component average passes through zero, where the normal field is
strongly and wrongly enhanced. Metal surfaces therefore stay staircased, with
two consequences:

- **A surface that lies on grid nodes is exact.** A film whose two faces
  coincide with grid nodes, in the direction normal to it, has exactly its
  thickness in cells. Make the mesh fit: with `UniformMesh`, choose a cell
  size that divides the film thickness and put the faces on multiples of it;
  with `AutoMesh`, add an override box around the film with such a cell size.
- **Any other surface is off by up to half a cell.** A film of 33 nm on 5 nm
  cells is simulated as 30 or 35 nm, and the surface of a sphere becomes a
  staircase. The error of a transmission, a resonance position or a
  cross-section then falls with the first power of the cell size. Halving
  the cells halves the error, where it would quarter it for a dielectric.
  For plasmonic particles and thin films, 1 to 2 nm cells at the metal are
  usual.

## The dispersion library and fitting

`lightsim.materials` holds published dispersion formulas for SiO2, Si3N4, Si,
LiNbO3, Al2O3, GaAs and Ge, and a fit that turns refractive-index data into
poles. `fitted_medium` does both steps for a library material and returns a
`Medium` valid over the band you ask for:

```python
print(mat.available_materials())
print(mat.refractive_index("Si3N4", 1.55e-6))

nitride = mat.fitted_medium("Si3N4", band=(1.2e-6, 1.7e-6), n_lorentz=2)
n_model = np.sqrt(mat.pole_permittivity(nitride, 2 * np.pi * ls.C0 / 1.55e-6))
print(round(nitride["eps"][0], 4), len(nitride["poles"]), round(n_model.real, 5))
```

```text
['SiO2', 'Si3N4', 'Si', 'LiNbO3_o', 'LiNbO3_e', 'Al2O3_o', 'Al2O3_e', 'GaAs', 'Ge']
(1.9962797317138814+0j)
3.2765 2 1.99628
```

The fitted medium is used like any other, for example
`sim.add_structure(shape, nitride)`.

The fit chooses $\varepsilon_\infty \ge 1$ together with the poles. The
library formulas are Sellmeier formulas, whose own $\varepsilon_\infty$ is 1,
so for some materials the best fit ends at that bound with one strong
ultraviolet pole: silicon gives `eps = 1` and a pole of strength 10.7. Such a
medium is accurate, but it costs a smaller time step (see above). Passing
`eps_inf` fixes it instead: `fitted_medium("Si", eps_inf=6.0)` needs no
reduction, and its largest index error in the band rises from 5e-5 to 1e-4.

A fit of a glass over a narrow band also has a pole in the far infrared (the
infrared term of the Sellmeier formula), and silicon fitted with
$\varepsilon_\infty = 1$ has a strong ultraviolet one; just above such a pole
the lossless model is metal-like. This matters only inside absorbing layers,
which damp the first kind and freeze the second by default: see
[Dispersive media in the layers](boundaries.md#dispersive-media-in-the-layers).

For your own data, `fit_poles(wavelengths, n, k)` fits Lorentz poles, and
optionally one Drude pole, to sampled values of $n$ and $k$. The fit is
constrained so that the result is passive and stable. The entries of the
library are transparent-window formulas with $k = 0$; absorbing materials must
be fitted from data. The functions, their options, the sources of the
formulas and the accuracy of the fits are in the
[material library notes](../modules/materials.md).

## Sheet materials

Two-dimensional conducting sheets, such as graphene, are given by a surface
conductivity $\sigma_s(\omega)$ in siemens. A zero-thickness `Sheet` carries
the surface current $\sigma_s \mathbf{E}_t$ on a plane, a rectangle or a
polygon, with a constant conductance, a constant surface susceptibility and
Drude and Lorentz terms; `lightsim.sheets` builds the conductivity from the
Kubo formula (`sheets.graphene`), a Drude weight or any fitted function:

```python
import lightsim as ls
from lightsim import sheets

g = sheets.graphene(fermi_level_eV=0.4, scattering_time=1e-13, temperature=300)
sim = ls.Simulation(min=[0, 0, 0], max=[300e-6, 7.5e-6, 7.5e-6], mesh=ls.UniformMesh(3.75e-6), run_time=1e-12)
sim.add_sheet(ls.Sheet("x", 151.3e-6, g))                         # the whole plane x = 151.3 um
sim.add_sheet(ls.Sheet("x", 200e-6, g, min=[0, 0, 0], max=[0, 3.75e-6, 7.5e-6]))   # a rectangle
print(sim.time_info())
```

The current enters Ampère's law on the plane of the tangential E nodes,
divided by the dual cell normal to the sheet; generated meshes put a node on
every sheet plane, so nothing has to be aligned and the scheme is of second
order (an explicit mesh is not moved: a sheet between its nodes is shared by
the two planes around it, with a first-order error). The sheets work on
every engine, with complex fields and in 2-D (as lines); the gradient
engines refuse them. With a `thickness`, the same functions return instead a
medium for a layer one cell thick. The equations, the measured accuracy
(second order, 2e-4 at 80 cells per wavelength; oblique incidence; sheet
plasmons and strip resonances) and the limits are in
[Sheet materials](../modules/sheets.md).

## Anisotropy

Three levels of anisotropy are supported.

**Diagonal.** Give `n`, `eps` or `sigma` as three values for the x, y and z
axes. Poles can be restricted to some axes with `pole_axes`.

**Full symmetric tensors.** A crystal whose axes are not aligned with the
grid has off-diagonal permittivity terms. Either give them directly with
`eps_off = (eps_xy, eps_xz, eps_yz)`, or give the principal values and a
`rotation` matrix whose columns are the crystal axes in grid coordinates. The
conductivity is treated the same way, so rotated absorbing crystals work.

**Oriented poles.** `oriented=[(pole, u), ...]` adds a dispersive pole that
responds only to the field component along the direction `u`. This describes
a dispersive crystal with a rotated optic axis.

```python
c = np.cos(np.pi / 4)
about_z = [[c, -c, 0], [c, c, 0], [0, 0, 1]]              # rotation by 45 degrees about z

uniaxial = ls.Medium("uniaxial", n=(1.8, 1.5, 1.5))                       # optic axis along x
rotated = ls.Medium("rotated", n=(1.8, 1.5, 1.5), rotation=about_z)       # optic axis along (x + y)
resonant = ls.Medium("resonant", eps=2.25,
                     oriented=[(ls.Lorentz(1.0, 2 * w0, 0.0), (1, 1, 0))])
print(np.round(rotated["eps"], 3), np.round(rotated["eps_off"], 3))
```

```text
[2.745 2.745 2.25 ] [0.495 0.    0.   ]
```

Limits:

- Media with off-diagonal terms, oriented or gyrotropic poles cannot also
  have ordinary `poles`, `chi2` or `chi3`, and cannot be perfect conductors.
- Oriented and gyrotropic poles require a diagonal `eps` without `sigma`.
- Off-diagonal media cannot be combined with Bloch boundaries.
- Off-diagonal media are not sub-pixel smoothed, and on a GPU they use the
  slower split kernels (see [GPUs and scaling](gpu.md)).

### Magneto-optic (gyrotropic) media

`gyro=[(pole, bias), ...]` adds a Lorentz or Drude pole in a static
magnetic bias. `bias` is the gyration vector $\mathbf b$ in rad/s, along the
bias field; for free carriers it is the cyclotron frequency $eB/m$. The
polarisation of each pole obeys

$$
\ddot{\mathbf P} + \gamma \dot{\mathbf P} + \omega_0^2 \mathbf P
 - \mathbf b \times \dot{\mathbf P} = \varepsilon_0\, s\, \mathbf E ,
$$

with $s = \Delta\varepsilon\,\omega_0^2$ for a Lorentz pole and
$s = \omega_p^2$, $\omega_0 = 0$, for a Drude pole. With
$D = \omega_0^2 - \omega^2 - i\gamma\omega$ the susceptibility is the tensor
$\chi = s\,(D\,I + i\omega[\mathbf b\times])^{-1}$. For a bias along z,

$$
\varepsilon = \begin{pmatrix} \varepsilon_\infty + \chi_{xx} & i g & 0 \\
 -i g & \varepsilon_\infty + \chi_{xx} & 0 \\ 0 & 0 & \varepsilon_\infty + s/D \end{pmatrix},
\qquad
\chi_{xx} = \frac{s D}{D^2 - \omega^2 b^2}, \quad
g = \frac{s\,\omega b}{D^2 - \omega^2 b^2},
$$

and the circular polarisations $(1, \pm i, 0)$ that travel along the bias
see $\varepsilon_\pm = \varepsilon_\infty + s/(D \pm \omega b)$. A linear
polarisation therefore turns by $\theta = (k_0/2)\,\mathrm{Re}(n_- - n_+)$
per unit length (Faraday rotation), in the same sense about $\mathbf b$
whichever way the wave travels, so a round trip doubles it.
`materials.gyrotropic_permittivity(medium, omega)` evaluates the tensor
(with `dt=` the tensor of the discrete update):

```python
from lightsim.materials import gyrotropic_permittivity

faraday = ls.Medium("faraday", eps=2.25,
                    gyro=[(ls.Lorentz(1.0, 1.5 * w0, 0.0), (0, 0, 0.2 * w0))])
print(np.round(gyrotropic_permittivity(faraday, w0), 4))
```

```text
[[4.0973+0.j     0.    +0.2956j 0.    +0.j    ]
 [0.    -0.2956j 4.0973+0.j     0.    +0.j    ]
 [0.    +0.j     0.    +0.j     4.05  +0.j    ]]
```

How it is solved: every E-component position of a gyrotropic medium
carries the whole polarisation vector of each pole, driven by $\mathbf E^n$
there (its own component and the averages of the other two over the four
nearest positions, as for oriented poles). The equation above, with central
differences for both derivatives, gives
$\mathbf P^{n+1} = A\mathbf P^n + B\mathbf P^{n-1} + C\mathbf E^n$ with
3 × 3 matrices $A, B, C$ computed once per pole; the component along the
position's own axis feeds back into the field,
$E_a^{n+1} \mathrel{-}= \Delta P_a/\varepsilon_\infty$. The bias can point in
any direction.

Accuracy (validation case `gyrotropic_faraday`): in a half-space the
indices $n_\pm$ of the circular polarisations, measured from the phase of
the circular components between two depths, match those of the scheme's
discrete $\varepsilon(\omega)$ within 0.02–0.1 % at 0.9, 1.0 and 1.1 $f_0$
(the residual is the grid's own dispersion at 40 cells per wavelength), and
the rotation of the polarisation within 0.1–0.5 %, for a Lorentz pole biased
along z, a Drude pole (magnetised plasma) biased along x, and in a 2-D
simulation; crossing the medium backwards gives the same rotation per length
in the lab frame within 3e-4 (non-reciprocity).

Limits:

- `eps` must be diagonal and lossless (`sigma` and `eps_off` are refused);
  loss comes from the poles' $\gamma$. Ordinary `poles`, `chi2`, `chi3` and
  `pec` are refused in the same medium; oriented poles may be combined.
- The time step is reduced, if needed, for the pole without its bias (see
  [Loss and dispersion](#loss-and-dispersion)); with the bias the update stays
  stable in the validation runs (a magnetised plasma with
  $\omega_p = 0.5\,\omega$, $b = 0.3\,\omega$ decays to 5e-7 of its peak
  over 600 fs), but no general bound with the bias is proven.
- A gyrotropic medium couples the two field sets of a 2-D simulation: use
  `polarization="both"`. Bloch boundaries are refused.
- The gyrotropy is electric (a permittivity tensor). Magnetic gyrotropy
  (ferrites, a permeability tensor) is not modelled.
- The gradient engines and `lightsim.adjoint` refuse scenes with gyrotropic
  media, and time reversal refuses them. The medium is not reciprocal (its
  tensor is antisymmetric off the diagonal), so the adjoint run of the
  S-matrix engines would have to use the medium with the bias reversed;
  this is not implemented.
- CPU, one or several GPUs and 2-D simulations are supported; on a GPU the
  update runs in its own kernel, after the off-diagonal tensor step.

### Complex (Hermitian) permittivity tensors given directly

`Medium(eps_complex=...)` takes a complex 3 × 3 relative permittivity tensor
directly: Hermitian ($\varepsilon_{ij} = \varepsilon_{ji}^*$) and positive
definite, for example the tensor of a magneto-optic medium biased along x,
$\varepsilon_{yz} = -\varepsilon_{zy} = ig$. The medium is non-dispersive and
lossless. Such a tensor has no real time-domain counterpart (a real field
would see $\varepsilon$ at positive frequencies and $\varepsilon^*$ at negative
ones), so a simulation with it runs with complex fields, the real and
imaginary field sets of Bloch runs, and applies $\varepsilon$ to the complex
field; every frequency evolves on its own, and the frequency-domain monitors
see $\varepsilon$ at all their (positive) frequencies. The model is exact only
as a non-dispersive one: a physical magneto-optic medium is dispersive, and
the tensor is then its value at one frequency (for the dispersive model use
gyrotropic poles, above).

```python
import lightsim as ls

g = 0.3
mo = ls.Medium("mo", eps_complex=[[2.25, 0, 0], [0, 2.25, 1j * g], [0, -1j * g, 2.25]])
print(mo["eps"], mo["eps_complex"][1][2])
```

The update is that of the real off-diagonal tensors on the complex field:
with $\kappa = \varepsilon^{-1} = \kappa_R + i\kappa_I$, the linear update of
each field set applies $\kappa_{aa}$, $\Delta D$ is taken per set, and

$$
E_{a,\mathrm{re}} \mathrel{+}= \sum_b \kappa_{R,ab}\langle\Delta D_{b,\mathrm{re}}\rangle - \kappa_{I,ab}\langle\Delta D_{b,\mathrm{im}}\rangle, \qquad
E_{a,\mathrm{im}} \mathrel{+}= \sum_b \kappa_{R,ab}\langle\Delta D_{b,\mathrm{im}}\rangle + \kappa_{I,ab}\langle\Delta D_{b,\mathrm{re}}\rangle
$$

over the two other components, with the four-point averages of the real
tensor update. Lumerical solves such a tensor through a unitary
transformation that diagonalises it; this is a different formulation (the
D-form tensor update on complex fields) with the same continuous model.

Accuracy (validation case `hermitian_tensor`, wave along x): with
$\varepsilon = 2.25$ and $g = 0.3$, the circular polarizations travel with
$n_\pm = \sqrt{\varepsilon \mp g}$ (Yee dispersion included) within 5e-6 and
the polarization turns by $(k_0/2)(n_- - n_+)$ per length within 4e-5, at
0.9, 1.0 and 1.1 f₀; crossed backwards, the rotation has the same sense in
the lab frame to 2e-6 (non-reciprocal); against a lossless gyrotropic
Lorentz pole biased along x, the tensor of its discrete susceptibility at f₀
gives the same $n_\pm$ within 3e-5 and the same rotation within 6e-4 at f₀; a
real symmetric tensor given as `eps_complex` reproduces the off-diagonal
tensor medium (`eps_off`) exactly.

Limits: no poles, `sigma`, `mu` or PEC in the same medium, and no nonlinear,
gain, plug-in or `eps_off` media in the same simulation (give a real tensor as
`eps_complex` instead); 3-D only (2-D simulations are refused); on several
slabs (two GPUs, several nodes) only tensors that couple y and z (the x–y and
x–z terms would need fields from the neighbouring slab); on a GPU such runs
use the split kernels (the fused step does not handle complex tensors);
refused by time reversal and the gradient engines.

### Anisotropy that varies from cell to cell (liquid crystals)

`director=DirectorField(min, max, director, n_o=..., n_e=...)` makes a
medium uniaxial with an optic axis (director) $\mathbf d(\mathbf r)$ given on
a regular grid:

$$
\varepsilon(\mathbf r) = \varepsilon_o I + (\varepsilon_e - \varepsilon_o)\, Q(\mathbf r),
\qquad Q = \frac{\langle \mathbf d \mathbf d^T \rangle}{\mathrm{tr} \langle \mathbf d \mathbf d^T \rangle} ,
$$

where $\langle\cdot\rangle$ is the trilinear interpolation of the samples at
each E-field position. Interpolating $\mathbf d \mathbf d^T$ instead of
$\mathbf d$ makes the result independent of the sign of each sample (a
director has no head or tail). The samples span the box from `min` to `max`,
faces included, and the array axes follow the rules of `IndexGrid`: the last
axis holds the three components, `axes=(2,)` makes a 1-D grid vary along z.
Outside the box the nearest face value is used, so a single sample is a
uniform director. The structure that uses the medium decides where the
liquid crystal is.

The example is a twisted nematic cell: the director turns by 90 degrees
about z across a 10 µm layer.

```python
z0, thick = 2 * um, 10 * um
phi = np.linspace(0, np.pi / 2, 201)
twist = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
tn = ls.Medium("tn", director=ls.DirectorField([0, 0, z0], [0, 0, z0 + thick], twist,
                                                n_o=1.5, n_e=1.6, axes=(2,)))
print(tn["director"]["dims"], round(tn["director"]["eps_e"], 4), tn["eps"][0])
```

```text
[1, 1, 201] 2.56 2.5600000000000005
```

How it is solved: each E-component position gets the permittivity tensor of
its own position, and the semi-implicit tensor update of the other
off-diagonal media uses it: the diagonal update applies
$\kappa_{aa} = (\varepsilon^{-1})_{aa}$ and the off-diagonal terms add
$\kappa_{ab}$ times the four-point average of the neighbouring $D_b$
increments. The kernels are those of the uniform tensors; only the
coefficients now differ from cell to cell.

Accuracy (validation case `director_field`): a uniform director along the
optic axis of the rotated crystal of `offdiagonal_tensor` reproduces that
case's fields exactly (difference 0). A 90-degree twisted nematic layer
($n_o = 1.5$, $n_e = 1.6$, 10 µm, between index-matched half-spaces) lit
with light polarised along the entrance director transmits, from 0.8 to
1.2 $f_0$, the Jones vector of the Jones-matrix product of 4000 thin rotated
retarders within 2e-3 in the magnitude of each component, 3–11 mrad in their
relative phase and 0.1° in the azimuth of the polarisation ellipse (which
follows the twist: 77° to 95° at this Mauguin parameter, 1.6 to 2.4), in
3-D and in a 2-D simulation. The phase residual grows with frequency as the
grid's dispersion does (40 cells per wavelength in the layer at 1.2 $f_0$).

Limits:

- The medium is lossless and non-dispersive: `sigma`, `eps_off`, oriented and
  gyrotropic poles are refused with a director field, as are ordinary
  `poles` and `chi2` / `chi3`.
- The tensor of a cell is taken at the position of each E sample; nothing
  is smoothed. A director that turns by a large angle between two cells
  (a disclination) is represented by the interpolated tensor, whose
  anisotropy is then smaller than $\varepsilon_e - \varepsilon_o$.
- The update is the one of the uniform off-diagonal tensors. With a
  spatially varying tensor its coupling terms are not exactly symmetric
  between neighbouring cells; smooth fields such as the validation's are
  stable, strongly varying ones have not been characterised.
- A 2-D simulation needs `polarization="both"`; Bloch boundaries are
  refused.
- The gradient engines refuse scenes with a director field (all three
  S-matrix engines, reverse mode and `lightsim.adjoint`). The tensor is
  symmetric at every point, but the off-diagonal update takes it at each
  component's own position, so where the director varies between cells the
  discrete scheme is not symmetric and the adjoint run is not its transpose:
  next to a slab the S-matrix session was 1.1 % off central differences for
  a uniform director, 1.7 and 1.9 % for 90 and 180 degree twists and a
  factor of 33 for a full turn within three cells. A uniform director is
  an off-diagonal medium (`Medium.rotated`, `eps_off`), which the S-matrix
  session handles.
- CPU, one or several GPUs and 2-D simulations are supported; no new kernel
  is involved.

## Magnetic media and perfect conductors

`mu` sets the relative permeability, one value or three. A simulation that
contains a magnetic medium uses the split GPU kernels.

`Medium(name, pec=True)` is a perfect electric conductor: the tangential
electric field is zero inside it. The constant `lightsim.PEC` is such a
medium, and the name `"pec"` is built in. A conductor is staircased to the
grid.

## Nonlinear media

`chi2` and `chi3` add an instantaneous nonlinear polarisation along each axis,

$$
\frac{P_i}{\varepsilon_0} = \chi^{(2)}_i E_i^2 + \chi^{(3)}_i E_i^3 ,
$$

with $\chi^{(2)}$ in $\text{m}/\text{V}$ and $\chi^{(3)}$ in $\text{m}^2/\text{V}^2$, and $E_i$ the real,
time-domain field. With this definition the Kerr coefficient of a medium of
index $n_0$ is $n_2 = 3\chi^{(3)} / (4 n_0^2 \varepsilon_0 c)$.

```python
kerr = ls.Medium("kerr", n=1.5, chi3=1e-18)
print(kerr["chi3"])
```

```text
[1e-18, 1e-18, 1e-18]
```

A nonlinear response depends on the field strength, so the source `amplitude`
(in V/m for a plane wave) matters, and results can no longer be normalised by
a reference run at a different power. New frequencies are generated; the mesh
must resolve the highest harmonic of interest.

Limits: there is no nonlinear coupling between field components, and
nonlinear media cannot be combined with Bloch boundaries or with off-diagonal
tensors. A delayed (Raman) part of $\chi^{(3)}$ is described next.

### Raman response of the χ³ term

`raman=Raman(fraction, omega, delta)` makes a fraction $f$ of $\chi^{(3)}$ a
delayed response. Along each axis

$$
\frac{P_i}{\varepsilon_0} = \chi^{(2)}_i E_i^2
 + \chi^{(3)}_i E_i \left[(1 - f)\, E_i^2 + f\, Q_i\right],
\qquad
Q_i(t) = \int_0^\infty g_R(\tau)\, E_i^2(t - \tau)\, d\tau ,
$$

with the damped-oscillator response

$$
g_R(t) = \frac{\Omega_R^2}{\nu}\, e^{-\delta_R t} \sin \nu t, \qquad
\nu = \sqrt{\Omega_R^2 - \delta_R^2}, \qquad
g_R(\omega) = \frac{\Omega_R^2}{\Omega_R^2 - \omega^2 - 2 i \delta_R \omega} .
$$

| Parameter | Meaning |
|---|---|
| `fraction` | $f$, the share of $\chi^{(3)}$ that is delayed, 0 to 1 |
| `omega` | $\Omega_R$, the phonon (resonance) angular frequency, rad/s |
| `delta` | $\delta_R$, the damping rate, rad/s, below `omega` |

$g_R$ integrates to 1, so a slowly varying field sees the whole
$\chi^{(3)}$. The common fibre form
$g_R = \frac{\tau_1^2 + \tau_2^2}{\tau_1 \tau_2^2} e^{-t/\tau_2} \sin(t/\tau_1)$
is `omega` $= \sqrt{1/\tau_1^2 + 1/\tau_2^2}$, `delta` $= 1/\tau_2$; for
silica ($f = 0.18$, $\tau_1 = 12.2$ fs, $\tau_2 = 32$ fs) that is
$\Omega_R = 8.77 \times 10^{13}$ rad/s and $\delta_R = 3.13 \times 10^{13}$ rad/s.

```python
silica_like = ls.Medium("silica_like", n=1.45, chi3=1.9e-22,
                        raman=ls.Raman(0.18, 8.77e13, 3.13e13))
print(silica_like["raman"])
```

```text
{'fraction': 0.18, 'omega': 87700000000000.0, 'delta': 31300000000000.0}
```

How it is solved: $Q$ obeys
$\ddot Q + 2\delta_R \dot Q + \Omega_R^2 Q = \Omega_R^2 E^2$, advanced with
central differences like a Lorentz pole, $Q^{n+1}$ from $E^n$, before the E
update. After the linear E update, each cell solves
$\varepsilon E^{n+1} + P^{n+1}/\varepsilon_0 = \varepsilon E_\text{lin} + P^{n}/\varepsilon_0$
for $E^{n+1}$ with four Newton steps, as the instantaneous terms are solved.

Accuracy (validation case `chi3_raman`): the field radiated by a thin
index-matched slab under a 1.8 fs pulse matches the analytic convolution with
$g_R(t)$ within 0.1–0.2 % for a purely delayed response and for $f = 0.5$
(third harmonic 0.9 %), in 3-D and in a 2-D simulation; the stimulated Raman
gain of a weak Stokes probe under a CW pump matches the coupled-wave gain
coefficient $g = (\omega_s / c)(f\chi^{(3)}/2) A_p^2\, \mathrm{Im}\, g_R(\omega_p - \omega_s)$
within 0.9 % at $gL = 0.02$. That residual grows with $gL$ (0.6 % at 0.01,
1.6 % at 0.04), as the coupling to the anti-Stokes wave, phase-matched in a
dispersionless medium, predicts.

Limits:

- The model follows lightsim's per-axis convention: $E_i$ drives only $Q_i$,
  and there is no coupling between components. It has the structure of
  Lumerical's "Chi3 Raman Kerr" material (a fraction of $\chi^{(3)}$ acting
  through a damped oscillator) but is not a re-implementation of it, and an
  isotropic $|\mathbf E|^2$ response for arbitrary polarisation is not
  modelled.
- The oscillator update is stable for $\Omega_R \Delta t < 2$, which any
  optical time step satisfies. The Newton step needs
  $1 + f\chi^{(3)} Q/\varepsilon$ to stay positive, as the instantaneous
  step needs a positive slope; very strong defocusing media fail there.
- As the other nonlinear media: no Bloch boundaries, no off-diagonal tensors.
  The gradient engines (`lightsim.autodiff`, the S-matrix problem,
  `lightsim.adjoint`) refuse scenes that use a Raman medium, and time
  reversal refuses them.
- CPU, one or several GPUs and 2-D simulations are supported.

## Gain media

`gain=FourLevel(density, omega, gamma, tau32, tau21, tau10, pump_rate=0,
coupling=None, initial=(1, 0, 0, 0))` adds a four-level atomic system whose
2 → 1 transition is coupled to the field through a Lorentz polarisation
whose strength follows the population difference:

$$
\ddot P + \gamma \dot P + \omega_a^2 P = \xi\,(N_1 - N_2)\, E ,
$$

$$
\dot N_3 = W_p N_0 - \frac{N_3}{\tau_{32}}, \quad
\dot N_2 = \frac{N_3}{\tau_{32}} - \frac{N_2}{\tau_{21}} + \frac{E \dot P}{\hbar\omega_a}, \quad
\dot N_1 = \frac{N_2}{\tau_{21}} - \frac{N_1}{\tau_{10}} - \frac{E \dot P}{\hbar\omega_a}, \quad
\dot N_0 = \frac{N_1}{\tau_{10}} - W_p N_0 .
$$

| Parameter | Meaning |
|---|---|
| `density` | total atom density $N_0 + N_1 + N_2 + N_3$, m⁻³ |
| `omega`, `gamma` | $\omega_a$ and the width $\gamma$ of the transition, rad/s |
| `tau32`, `tau21`, `tau10` | lifetimes, s |
| `pump_rate` | $W_p$, the pump rate from level 0 to level 3, 1/s |
| `coupling` | $\xi$ in C²/kg; default the radiative value $6\pi\varepsilon_0 c^3/(\omega_a^2 \tau_{21})$ |
| `initial` | starting populations as fractions of `density` |

At fixed populations the transition adds
$\chi(\omega) = \frac{\xi (N_1 - N_2)/\varepsilon_0}{\omega_a^2 - \omega^2 - i\gamma\omega}$
to the medium's `eps` (absorption while $N_1 > N_2$, gain once inverted).
`Results.gain_levels()` returns the number of atoms in each level at the end
of the run, summed over the medium.

```python
laser = ls.Medium("laser", eps=2.25,
                  gain=ls.FourLevel(1e24, w0, 0.1 * w0, 1e-13, 1e-9, 1e-13, pump_rate=1e8))
print(sorted(laser["gain"]))
```

```text
['density', 'gamma', 'initial', 'omega', 'pump_rate', 'tau10', 'tau21', 'tau32']
```

How it is solved: before the E update,
$P^{n+1} = c_0 P^n + c_1 P^{n-1} + c_2 (N_1 - N_2)^n E^n$ (the Lorentz-pole
update with a strength that changes every step); after it,
$E^{n+1} \mathrel{-}= \Delta P/\varepsilon$, and the field's work
$\tfrac12 (E^n + E^{n+1})\,\Delta P$ moves $\varepsilon_0 \cdot$ work$/\hbar\omega_a$
atoms per volume from level 1 to level 2. The decays and the pump are
explicit fluxes from the populations of step $n$, so the number of atoms is
conserved exactly; the populations are kept in double precision, since one
step moves only a tiny fraction of them.

Accuracy (validation case `gain_four_level`): an inverted 2 µm slab
transmits $T(\omega)$ within 0.05–0.2 % of the slab formula with the
frozen populations' discrete $\chi(\omega)$ (peak $T = 1.59$), in 3-D and in
a 2-D simulation; a pumped slab driven by a CW wave at 0.01, 1 and 3 times
the saturation intensity has the steady-state gain of the rate equations
(populations at the local intensity, integrated through the slab) within
0.1–0.17 %; the field energy an absorbing slab takes from a strong pulse,
from the Poynting flux in front of and behind it, equals $\hbar\omega_a$
times the atoms found in level 2 within 5e-7, and the atoms are conserved to
1e-13.

What it is and is not:

- This is a four-level rate-equation model pumped by a rate $W_p$. It is not
  Lumerical's "four-level two-electron" model, which pumps through a second
  polarisation driven by a pump field (Chang and Taflove); here the pump is
  a given rate, not an optical field in the simulation. That model is
  available as well, see below.
- Each E-component position carries its own atoms, coupled only to its own
  field component: an ensemble of dipoles aligned with x, y and z, each with
  the full `density`. A field polarised along one axis sees the model
  above; for fields of several components the populations at the
  different components do not share their saturation.
- The rate equations are explicit and need $\Delta t \le \tfrac12$ of every
  lifetime and of $1/W_p$; the solver refuses the medium otherwise. The time
  step is reduced, if needed, for the transition at its largest absorbing
  strength (all atoms in level 1).
- A gain medium cannot also have `chi2`, `chi3`, `raman`, off-diagonal or
  gyrotropic terms; ordinary `poles` and `sigma` may be added. Bloch
  boundaries are refused. The gradient engines refuse scenes with gain media,
  and time reversal refuses them.
- CPU, one or several GPUs (the population totals are summed over the slabs)
  and 2-D simulations are supported.

### Four-level two-electron medium (Chang and Taflove)

`Medium(gain2=FourLevelTwoElectron(...))` is the model of Chang and Taflove,
"Finite-difference time-domain model of lasing action in a four-level
two-electron atomic system", Opt. Express 12, 3827 (2004), the formulation
of Lumerical's "4-level 2-electron" material, with its parameter names. Two
polarizations, the lasing transition 2 → 1 at $\omega_a$ and the pump
transition 3 → 0 at $\omega_b$, follow their population differences; each
atom has two electrons and each level holds at most one, hence the Pauli
factors $(1 - N_j/N_d)$:

$$
\begin{aligned}
\ddot P_a + \gamma_a \dot P_a + \omega_a^2 P_a &= \kappa_a (N_1 - N_2) E, \qquad
\ddot P_b + \gamma_b \dot P_b + \omega_b^2 P_b = \kappa_b (N_0 - N_3) E, \\
\dot N_3 &= \frac{E \dot P_b}{\hbar\omega_b} - \frac{N_3 (1 - N_2/N_d)}{t_{32}} - \frac{N_3 (1 - N_0/N_d)}{t_{30}}, \\
\dot N_2 &= \frac{N_3 (1 - N_2/N_d)}{t_{32}} + \frac{E \dot P_a}{\hbar\omega_a} - \frac{N_2 (1 - N_1/N_d)}{t_{21}}, \\
\dot N_1 &= \frac{N_2 (1 - N_1/N_d)}{t_{21}} - \frac{E \dot P_a}{\hbar\omega_a} - \frac{N_1 (1 - N_0/N_d)}{t_{10}}, \\
\dot N_0 &= \frac{N_3 (1 - N_0/N_d)}{t_{30}} + \frac{N_1 (1 - N_0/N_d)}{t_{10}} - \frac{E \dot P_b}{\hbar\omega_b} .
\end{aligned}
$$

| Parameter | Meaning |
|---|---|
| `density` | $N_d$, atoms per m³ (the populations sum to $2N_d$) |
| `omega_a`, `omega_b` | $\omega_a$, $\omega_b$ (rad/s) |
| `gamma_a`, `gamma_b` | $\gamma_a$, $\gamma_b$ (rad/s) |
| `t30`, `t32`, `t21`, `t10` | lifetimes (s) |
| `initial` | $(N_0, N_1, N_2, N_3)$ in m⁻³, default $(N_d, N_d, 0, 0)$ |
| `kappa_a`, `kappa_b` | coupling (C²/kg), default $6\pi\varepsilon_0 c^3/(\omega^2 t)$ with $t_{21}$, $t_{30}$ |

The medium is pumped optically: a source at $\omega_b$ in the simulation.
The discretisation is that of the rate-equation model above (both
polarizations explicit from the populations of step $n$, populations moved
by explicit fluxes and by the field work $\varepsilon_0\tfrac12(E^n +
E^{n+1})\Delta P/\hbar\omega$), so the electron number is conserved to
rounding. `Results.gain_levels()` returns the electrons per level.

```python
import numpy as np
import lightsim as ls

w0 = 2 * np.pi * ls.C0 / 1e-6
g = ls.FourLevelTwoElectron(density=1e24, omega_a=w0, omega_b=1.3 * w0, gamma_a=0.1 * w0, gamma_b=0.1 * w0,
                            t30=300e-15, t32=20e-15, t21=500e-15, t10=20e-15)
laser = ls.Medium("dye", eps=2.25, gain2=g)
print(sorted(laser["gain2"]))
```

Accuracy (validation case `gain_two_electron`, CPU):

- a plane of atoms driven by a CW pump at $\omega_b$ for 1.5 ps: the four
  populations within 2e-5 $N_d$ of the continuous model (both oscillators
  and the rate equations) integrated by RK4 from the field recorded at the
  atoms;
- a thin pumped slab in its steady state with a weak probe at $\omega_a$:
  the probe's single-pass gain within 0.1 % and the pump's attenuation within
  0.2 % of the steady-state populations of the rate equations (with the
  pump's off-resonant drive of the lasing transition, which lowers the
  inversion by 6 % here) through the discrete Lorentzians;
- a 1-D laser (4 µm slab of $\varepsilon_b$ = 2.25, bare facets, pumped from
  one side at 1.4, 1.7 and 2 times threshold, seeded by a weak pulse): a
  single steady line (output power of the last two picoseconds equal
  within 0.13 %) at the frequency of the threshold condition
  $r^2 e^{2in k_0 L} = 1$ within 0.14–0.17 %; the pump threshold from a linear
  fit of the output power within 6 % of the rate-equation prediction; the
  mean inversion nearly clamped (it rises by 7.5 % from 1.4 to 2 times
  threshold: spatial hole burning in the standing wave, the clamped quantity
  being the field-weighted inversion);
- electrons conserved to 1e-13.

Differences from the rate-equation model above: two electrons per atom with
Pauli factors instead of one, optical pumping through a pump polarization
instead of a pump rate, and the pump field must be in the simulation (and is
absorbed by the medium). It is the same set of equations as Chang and
Taflove's paper and Lumerical's material; it is not Lumerical's code, and the
discretisation (explicit populations, explicit ADE polarizations) is
lightsim's. Limits as for the rate-equation model: explicit rate equations
($\Delta t \le \tfrac12$ of every lifetime), per-axis atoms, no Bloch
boundaries, no χ², χ³, poles, off-diagonal terms or plug-ins in the same
medium, not in the same simulation as rate-equation gain media, refused by
time reversal and the gradient engines. CPU and GPUs (2-D runs use the same per-cell update; not validated separately).

## User-defined material plug-ins

A medium can carry a material model written by the user: on top of its
background `eps` and `sigma`, a plug-in turns the field history into a
polarization (or solves for the new field) at every step. lightsim calls it
once per step and per E component, with the medium's cells:

1. `before(e, state, ctx)` before the linear E update, `e` = $E^n$ of the
   cells: an explicit model advances its polarization here;
2. `after(e, e_prev, inv_eps, state, ctx)` after it: `e` holds $E_\text{lin}$
   (the field the linear update gave) and is overwritten with $E^{n+1}$;
   `inv_eps` $= 1/(\varepsilon(1 + f))$, so a polarization change enters as
   $E^{n+1} = E_\text{lin} - \texttt{inv\_eps}\,\Delta P/\varepsilon_0$, and an
   instantaneous model solves for $E^{n+1}$ there.

`state` holds the plug-in's per-cell numbers (zero at the start) and `ctx`
the axis, the step index, $\Delta t$ and the medium's parameters. Plug-in
media are staircased, as metals are: the plug-in acts on the cells
the medium owns.

In Python the callables take NumPy arrays over the cells (vectorised, so
NumPy or numba code runs at array speed). One Lorentz pole as a plug-in:

```python
import numpy as np
import lightsim as ls
from lightsim import plugins

def coefficients(p, dt):
    a = 1 / dt**2 + p["gamma"] / (2 * dt)
    c = ((2 / dt**2 - p["omega0"] ** 2) / a, -(1 / dt**2 - p["gamma"] / (2 * dt)) / a, p["delta_eps"] * p["omega0"] ** 2 / a)
    return [np.float32(v) for v in c]

def before(e, state, ctx):                     # state columns: P, P_prev, dP
    c0, c1, c2 = coefficients(ctx["params"], ctx["dt"])
    p0 = state[:, 0].copy()
    pn = c0 * p0 + c1 * state[:, 1] + c2 * e
    state[:, 2], state[:, 1], state[:, 0] = pn - p0, p0, pn

def after(e, e_prev, inv_eps, state, ctx):
    e -= inv_eps * state[:, 2]

plugins.register("my_lorentz", after=after, before=before, state=3)
w0 = 2 * np.pi * 3e14
med = ls.Medium("lorentz", eps=1.5, plugin=plugins.use("my_lorentz", {"delta_eps": 1.5, "omega0": 1.3 * w0, "gamma": 0.1 * w0}))
sim = ls.Simulation(min=[0, 0, 0], max=[5e-6, 50e-9, 50e-9], mesh=ls.UniformMesh(25e-9), run_time=30e-15)
sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
sim.add_structure(ls.Box([1.5e-6, -1, -1], [3.5e-6, 1, 1]), med)
sim.add_source(ls.PlaneWave(0, 1e-6, "+", "Ey", ls.GaussianPulse(3e14, 1e14)))
print(sim.run(device="cpu").stats["steps"])
```

In Rust a downstream crate implements the trait `lightsim_core::plugin::MaterialPlugin`
(`state_len`, `before`, `after`, and `error` to stop the run) and registers a
factory under a name; the medium names it with JSON parameters:

```text
use lightsim_core::plugin::{register, MaterialPlugin, PluginCall, PluginRef};
struct Kerr { chi3: f32 }
impl MaterialPlugin for Kerr {
    fn after(&mut self, _c: &PluginCall, e: &mut [f32], e_prev: &[f32], inv_eps: &[f32], _s: &mut [f32]) {
        for t in 0..e.len() {
            e[t] = lightsim_core::engine::cpu::nl_solve(e_prev[t], e[t], 0.0, self.chi3 * inv_eps[t]);
        }
    }
}
register("kerr", std::sync::Arc::new(|p: &serde_json::Value| {
    Ok(Box::new(Kerr { chi3: p["chi3"].as_f64().unwrap() as f32 }) as Box<dyn MaterialPlugin>)
}));
medium.plugin = Some(PluginRef { name: "kerr".into(), params: serde_json::json!({"chi3": 1e-20}) });
```

`lightsim_core::plugin::examples` holds a Lorentz pole, the Kerr medium and an
energy-gaining test model (`register_examples`, also
`lightsim.plugins.register_examples()` in Python).

**Accuracy.** A Lorentz pole and the instantaneous Kerr medium written as
plug-ins (in Rust and in Python) reproduce the built-in media exactly on the
CPU: the same single-precision arithmetic in the same order (validation case
`plugin_media`, `python/tests/test_plugins.py`). A plug-in that adds energy
is stopped by the solver's divergence check with an error; an exception in a
Python callable stops the run with its text.

**Cost.** CPU, a slab of 96³ plug-in cells in a 288 × 96 × 96 grid, 200
steps: with 10 threads the built-in pole 6.0 s, the Rust plug-in 5.0 s, the
NumPy plug-in 9.7 s; with 128 threads 2.8, 2.6 and 4.3 s (the NumPy plug-in
1.5–1.6 times the built-in pole; the plug-in calls run in the engine's loop,
single-threaded like the built-in pole update).

**GPU.** The GPU engine runs the plug-in on the host: each step the cells of
every plug-in medium are gathered from the device (two kernels added for
it), copied to the host, handed to the plug-in, and the result is copied
back and scattered (two round trips per component and step); the rest of the
step stays on the device. On the RTX PRO 6000, for the slab above (96³
plug-in cells, 200 steps): the built-in pole 0.034 s, the Rust plug-in 1.09 s
(32 times), the NumPy plug-in 2.5 s (73 times); the plug-in runs reproduce
the built-in pole to 3e-6 of the peak field (the device's fused
multiply-adds against the host's separate ones). A GPU run with a large
plug-in medium is therefore limited by the host and the transfers, and is
worth it only when the plug-in medium is a small part of the grid.

**Limits.** A plug-in medium is a plain background plus the plug-in: it
cannot also have poles, χ², χ³, gain, off-diagonal terms or magnetic
properties. Complex (Bloch) fields, time reversal and every gradient engine
refuse plug-in media. The solver knows nothing of the model's stability: a
plug-in that needs a shorter time step must be run with a smaller `courant`.
Each E component's cells are handed separately (per-axis models; a plug-in
that couples the components would need the other components' fields, which
it does not get). Plug-ins are lightsim's own interface; they are not
Lumerical's plug-in API.

## Imported and graded index

`IndexGrid(min, max, eps)` is a shape whose permittivity is given by samples
on a regular grid instead of one value. The samples span the box from `min`
to `max`, both faces included, and the solver interpolates them linearly at
each field position. Use `n=` instead of `eps=` to give refractive indices.

| Array | Varies along |
|---|---|
| scalar | nothing (a uniform box) |
| 1-D | x, or the axis in `axes=(a,)` |
| 2-D | (x, y), or the two axes in `axes=(a, b)` |
| 3-D | x, y, z, indexed `[ix, iy, iz]` |

The medium of the structure supplies everything except the permittivity: its
name, `sigma`, `chi2`, `chi3` and `mu`. Set the medium's `eps` to the largest
value in the grid, because the automatic mesher reads it from there.

The example replaces the abrupt faces of a glass slab by gradual ones, which
removes most of the reflection:

```python
glass = ls.Medium("glass", eps=2.25)
print(np.round(transmission([(slab, glass)]), 3))

x = np.linspace(0, 1, 61)                                         # 61 samples across 1 um
profile = np.interp(x, [0, 0.3, 0.7, 1.0], [1.0, 2.25, 2.25, 1.0])  # ramp up, flat, ramp down
graded = ls.IndexGrid([1.8 * um, -1, -1], [2.8 * um, 1, 1], eps=profile)
print(np.round(transmission([(graded, ls.Medium("graded", eps=2.25))]), 3))
```

```text
[0.943 0.909 0.879 0.86  0.853 0.86  0.881 0.911 0.945]
[0.975 0.992 1.    0.998 0.992 0.988 0.989 0.993 0.998]
```

Limits:

- The samples are real. Loss is one uniform `sigma` for the whole grid.
- The medium of an index grid cannot be dispersive, a perfect conductor or an
  off-diagonal tensor. `eps_axes=(eps_x, eps_y, eps_z)` gives a diagonal
  anisotropic grid.
- The faces of the grid box are staircased. To embed a grid in another
  material, add that structure first and let the grid overlap it.
- Keep sources outside the grid. A source inside it may be normalised with
  the medium's `eps` instead of the local value.

`lightsim.eo` builds index grids for silicon modulators: it converts maps of
free-carrier density into the index change and absorption given by the
Soref-Bennett plasma-dispersion model. See the
[index grid notes](../modules/index_grid.md).
