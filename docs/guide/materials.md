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

### Metal surfaces are staircased

A cell cut by the surface of a plain dielectric gets an effective
permittivity, which makes results converge with the square of the cell size.
A dispersive or conductive medium is not treated this way: every field
sample is either inside the medium or outside. Two consequences:

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

A correct treatment needs more than blending the two media. For the field
components parallel to a surface, the effective medium is the average of the
two permittivities, which for a dispersive medium means scaling its poles by
the filled fraction of the cell. For the component normal to the surface it
is the average of the inverse permittivities, and the inverse of a
dispersive permittivity is a different set of poles for every filling
fraction, coupled between components where the surface is oblique to the
grid. Scaling the poles by the filled fraction for all components is not a
substitute: it is right for the parallel components only, and at a metal
surface it makes cells whose permittivity passes through zero, where the
normal field is strongly and wrongly enhanced. This version therefore
staircases all components.

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

For your own data, `fit_poles(wavelengths, n, k)` fits Lorentz poles, and
optionally one Drude pole, to sampled values of $n$ and $k$. The fit is
constrained so that the result is passive and stable. The entries of the
library are transparent-window formulas with $k = 0$; absorbing materials must
be fitted from data. The functions, their options, the sources of the
formulas and the accuracy of the fits are in the
[material library notes](../modules/materials.md).

## Sheet materials

Two-dimensional conducting sheets, such as graphene, are given by a surface
conductivity. `lightsim.sheets` turns one into a medium for a layer one cell
thick: `sheets.graphene(fermi_level_eV, scattering_time, thickness)` from the
Kubo formula, or `sheets.sheet_medium(name, thickness, ...)` from a Drude
weight or any conductivity function. How to place the layer, the accuracy
against the analytic sheet formulas and the limits are in
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
  media (they rely on reciprocity), and time reversal refuses them.
- CPU, one or several GPUs and 2-D simulations are supported; on a GPU the
  update runs in its own kernel, after the off-diagonal tensor step.

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
