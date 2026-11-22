# Material library and pole fitting

`lightsim.materials` is the equivalent of Lumerical's *material explorer /
multi-coefficient fit*:
published dispersion formulas for common photonic materials, and a constrained
fit that converts sampled `n`, `k` data into the Lorentz / Drude ADE poles the
solver time-steps.

```python
import lightsim as ls
from lightsim import materials as mat

mat.refractive_index("SiO2", 1.55e-6)            # (1.4440236+0j)

sin = mat.fitted_medium("Si3N4", band=(1.2e-6, 1.7e-6), n_lorentz=2)
sim = ls.Simulation(min=[0, 0, 0], max=[2e-6, 1e-6, 1e-6], mesh=ls.UniformMesh(50e-9),
                    run_time=10e-15, background="Si3N4")
sim.add_medium(sin)                              # a plain lightsim.spec.Medium dict
```

The module is not re-exported from the package root; import it as
`from lightsim import materials`. It needs `numpy`, and `scipy` for the fit.

Conventions are those of the core: SI units (wavelengths in metres, angular
frequencies in rad/s), fields ~ `exp(-i w t)`, so loss is `Im(eps) > 0`, `k > 0`.

## API

### Library

| function | returns |
| --- | --- |
| `refractive_index(name, wavelength_m, strict=False)` | `n + i k` as `complex` (scalar input) or complex `ndarray` |
| `permittivity(name, wavelength_m, strict=False)` | `eps = (n + i k)^2` |
| `available_materials()` | canonical names |
| `material_info(name)` | `{"name", "range": (wl_min_m, wl_max_m), "reference"}` |

Names are case-insensitive, with aliases (`silica`, `SiN`, `silicon`,
`sapphire`, `LiNbO3` = ordinary ray, ...). Outside a formula's published
range a `UserWarning` is issued; `strict=True` raises `ValueError` instead.

### Fitting

`fit_poles(wavelengths_m, n, k=None, n_lorentz=2, drude=False, eps_inf=None, *, name="fitted", weights=None, omega0_max=None, n_starts=8, seed=0) -> dict`

Fits

```text
eps(w) = eps_inf + sum_j delta_eps_j w0_j^2 / (w0_j^2 - w^2 - i gamma_j w)  [- wp^2 / (w^2 + i gamma_d w)]
```

to the samples by bounded nonlinear least squares (`scipy.optimize.least_squares`,
trust-region reflective) on the complex permittivity error relative to `|eps|`
(times optional per-sample `weights`), from `n_starts` deterministic starting
points, keeping the best.

- Constraints: `delta_eps >= 0`, `gamma >= 0`, `omega0 > 0`, `omega_p >= 0`,
  `eps_inf >= 1` (or fixed by the caller). Every term then has `Im(chi) >= 0`
  at all frequencies, so the medium is passive. The explicit ADE update is
  stable below a time step that depends on the pole strengths and on
  `eps_inf`; the solver reduces the time step to that limit when needed (see
  the Materials guide).
- `n` may be complex, or `k` given separately. With no absorption data at all
  the poles are fitted lossless (`gamma = 0` exactly).
- `omega0_max` bounds the resonance frequencies, default 6x the highest sampled
  frequency. A pole far above the band forces a small time step; the default
  keeps the reduction small down to roughly 10 cells per shortest vacuum
  wavelength of the band. Raise it for a closer fit on fine grids.
- For transparent-window data the least-squares optimum can lie at
  `eps_inf = 1` with one strong ultraviolet pole (Si, and Si3N4 in the
  visible). Fix `eps_inf` at a larger value, up to about half the in-band
  permittivity, to trade a little accuracy for the full time step.

Returned dict:

| key | content |
| --- | --- |
| `eps_inf` | high-frequency permittivity |
| `lorentz` | list of `{"delta_eps", "omega0", "gamma"}`, sorted by `omega0` |
| `drude` | `{"omega_p", "gamma"}` or `None` |
| `poles` | the same as spec pole dicts (`ls.Lorentz(...)`, `ls.Drude(...)`) |
| `medium` | ready `ls.Medium(name, eps=eps_inf, poles=poles)` |
| `max_n_error`, `max_k_error` | max `|n - n_fit|`, `|k - k_fit|` over the samples |
| `rms_eps_error` | rms relative permittivity error |
| `band` | `(wl_min, wl_max)` of the samples |

`fitted_medium(name, band=(1.2e-6, 1.7e-6), n_lorentz=2, *, drude=False, eps_inf=None, samples=101, medium_name=None, tol=1e-3, return_fit=False, **fit_kwargs)`

Library formula -> `samples` points uniform in frequency over `band` ->
`fit_poles` -> `Medium` (named `medium_name`, default the library name). Warns
if `max_n_error > tol`. `return_fit=True` returns the whole `fit_poles` dict.

`pole_permittivity(medium_or_fit, omega, dt=None, axis=0)`

Evaluates a pole model (a `Medium` dict or a `fit_poles` result). With `dt` it
returns the discrete-time permittivity seen by the ADE update at that time
step (the transfer function of the core's `Pole::chi_discrete`), for comparing
FDTD results with the model.

### Limitations

- The fit is only as good as its band: outside `band` the model is an
  extrapolation and is not checked.
- Uniaxial crystals are one library entry per ray (`LiNbO3_o`, `LiNbO3_e`,
  `Al2O3_o`, `Al2O3_e`); `fitted_medium` gives an isotropic medium for that
  ray. An anisotropic dispersive medium has to be assembled by hand
  (`Medium(..., pole_axes=...)` / `oriented=`).
- All library entries are transparent-window formulas (`k = 0`). Absorbing
  materials are fitted from user data through `fit_poles`.
- Debye poles are evaluated by `pole_permittivity` but not fitted.

## Sources

Coefficients were checked against the refractiveindex.info database entries
for the same papers. Wavelengths in the formulas are in micrometres.

| name | formula | range (um) | reference |
| --- | --- | --- | --- |
| `SiO2` | 3-term Sellmeier, fused silica, 20 C | 0.21 - 6.7 | I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965) |
| `Si3N4` | 2-term Sellmeier, LPCVD film | 0.31 - 5.504 | K. Luke et al., Opt. Lett. 40, 4823 (2015) |
| `Si` | `n^2 = eps(T) + A(T)/lam^2`, T = 293 K | 1.2 - 14 | H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980) |
| `LiNbO3_o`, `LiNbO3_e` | 3-term Sellmeier, congruent, 21 C | 0.4 - 5.0 | D. E. Zelmon, D. L. Small, D. Jundt, J. Opt. Soc. Am. B 14, 3319 (1997) |
| `Al2O3_o`, `Al2O3_e` | 3-term Sellmeier, synthetic sapphire, 20 C | 0.2 - 5.0 | I. H. Malitson, M. J. Dodge, J. Opt. Soc. Am. 62, 1405 (1972); M. J. Dodge, Handbook of Laser Science and Technology IV (1986) |
| `GaAs` | 3-term Sellmeier, 22 C | 0.97 - 17 | T. Skauli et al., J. Appl. Phys. 94, 6447 (2003) |
| `Ge` | 3-term Sellmeier, 22 C | 2.0 - 14 | J. H. Burnett et al., Proc. SPIE 9974, 99740X (2016) |

For silicon, Li's temperature-dependent formula is evaluated at 293 K only,
where its thermal-expansion factor is 1; it reproduces Li's recommended table
(3.5167, 3.5016, 3.4757, 3.4653 at 1.20, 1.30, 1.55, 1.70 um) to 1e-4. Ge has
no telecom-band entry because it absorbs below about 1.8 um.

## Validation

`python/tests/test_materials.py` (28 tests, CPU, about 17 s):

```bash
PYTHONPATH=python pytest -q python/tests/test_materials.py
```

### Library values

| material | wavelength (um) | n |
| --- | --- | --- |
| SiO2 | 1.55 | 1.44402 |
| SiO2 | 0.5876 | 1.45846 |
| Si3N4 | 1.55 | 1.99628 |
| Si | 1.55 | 3.47569 |
| LiNbO3 o / e | 1.55 | 2.21111 / 2.13756 |
| Al2O3 o / e | 1.55 | 1.74618 / 1.73838 |
| GaAs | 1.55 | 3.37017 |
| Ge | 4.0 | 4.02472 |

SiO2 and Si3N4 are also checked against the formulas written out longhand in
the test, and Si against four rows of Li's table.

### Fits (2 Lorentz poles, lossless, default settings)

| material | band (um) | max abs(n - n_fit) | eps_inf | poles (delta_eps @ omega0 rad/s) |
| --- | --- | --- | --- | --- |
| SiO2 | 1.2 - 1.7 | 8.0e-8 | 1.9042 | 1.0779 @ 1.749e14, 0.2004 @ 9.289e15 |
| Si3N4 | 1.2 - 1.7 | 5.7e-6 | 3.2765 | 5.1675 @ 1.330e14, 0.7490 @ 7.138e15 |
| Si | 1.2 - 1.7 | 5.4e-5 | 1.0000 | 50.955 @ 2.709e13, 10.7163 @ 6.488e15 |
| LiNbO3_o | 1.2 - 1.7 | 6.5e-6 | 3.6935 | 10.0244 @ 9.815e13, 1.2133 @ 6.223e15 |
| LiNbO3_e | 1.2 - 1.7 | 5.2e-6 | 3.4894 | 9.1415 @ 9.264e13, 1.0927 @ 6.432e15 |
| Al2O3_o | 1.2 - 1.7 | 1.3e-6 | 2.7310 | 3.5220 @ 1.283e14, 0.3510 @ 8.788e15 |
| Al2O3_e | 1.2 - 1.7 | 1.6e-6 | 2.7087 | 3.4004 @ 1.300e14, 0.3459 @ 8.808e15 |
| GaAs | 1.2 - 1.7 | 6.3e-6 | 6.5492 | 13.2835 @ 3.327e13, 4.3332 @ 3.829e15 |
| Ge | 3.0 - 5.0 | 8.5e-7 | 10.3175 | 1.9268 @ 8.428e12, 5.6870 @ 2.589e15 |

The tests require `max |n - n_fit| < 1e-3` over 1.2 - 1.7 um (re-evaluated on a
denser grid than the fit used) and passivity: parameter signs, and
`Im(eps) >= 0` on 4001 frequencies from 1e11 to 1e19 rad/s. A single Lorentz
pole already gives errors of 3e-5 to 1.2e-3 over the same band.

A lossy case fits synthetic data from a known Drude + Lorentz model
(0.8 - 1.6 um) and recovers `eps_inf`, `omega_p` and `omega0` to 1e-5 relative,
with `n`, `k` errors below 1e-6.

### FDTD

A plane wave in a column of fitted Si3N4 (`dx` = 12.5 nm, 4 um long, periodic
in y and z, Gaussian pulse at 1.55 um, CPU). The phase between two DFT planes
1 um apart gives `n_eff`, compared with the Luke formula and with the exact
discrete dispersion of the scheme (ADE in time through
`pole_permittivity(..., dt=dt)`, Yee in space:
`sin(k dx/2) = sqrt(eps_d) sin(w dt/2) dx / (c dt)`).

| wavelength (um) | n_eff FDTD | n library | rel. diff | n discrete | rel. diff |
| --- | --- | --- | --- | --- | --- |
| 1.65 | 1.99418 | 1.99349 | 3.5e-4 | 1.99417 | 4.0e-6 |
| 1.55 | 1.99707 | 1.99628 | 4.0e-4 | 1.99706 | 4.5e-6 |
| 1.45 | 1.99998 | 1.99909 | 4.5e-4 | 1.99998 | -4.2e-6 |

Test tolerances are 0.2 % against the library and 1e-4 against the discrete
dispersion. The 0.04 % offset from the library is the grid dispersion of the
12.5 nm mesh (about 62 cells per wavelength in the medium), not fit error.
