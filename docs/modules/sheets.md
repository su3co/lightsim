# Sheet materials

`lightsim.sheets` models two-dimensional conducting sheets, such as
graphene, given by a surface conductivity $\sigma_s(\omega)$ in siemens.
The solver has no zero-thickness material, so a sheet is a thin volumetric
layer of thickness $t$ whose permittivity carries the same sheet current:

$$
\varepsilon(\omega) = \varepsilon_b + \frac{i\,\sigma_s(\omega)}{\varepsilon_0\,\omega\,t} .
$$

This is a different method from Lumerical's 2-D sheet material, which
applies the conductivity as a boundary condition on a surface with no
thickness. Both give the transmission and reflection of a sheet; here the
sheet occupies one cell of the mesh.

The module is not re-exported from the package root; import it as
`from lightsim import sheets`. It needs `numpy`, and `scipy` for the
interband integral and for fitted conductivities. Conventions are those of
the core: SI units and fields $\propto e^{-i\omega t}$, so a lossy sheet has
$\mathrm{Re}\,\sigma_s > 0$ and an inductive one $\mathrm{Im}\,\sigma_s > 0$.

## API

| Function | Returns |
|---|---|
| `sheet_medium(name, thickness, drude_weight=, gamma=)` | A medium with one Drude pole, $\omega_p^2 = D / (\varepsilon_0 t)$: the exact layer of the Drude sheet $\sigma_s = i D / (\omega + i\gamma)$ |
| `sheet_medium(name, thickness, sigma=f, band=(f_min, f_max))` | A medium whose layer permittivity is fitted over the band by `lightsim.materials.fit_poles` (`n_lorentz` Lorentz poles and, with `drude=True`, one Drude pole). `sigma` is a function of frequency in Hz, or an array with `freqs=` |
| `graphene(fermi_level_eV, scattering_time, thickness, temperature=300, band=None)` | A graphene layer from the Kubo conductivity. The intraband term is one exact Drude pole at every frequency. With `band`, the interband term is added as `n_lorentz` (default 2) Lorentz poles fitted over that band |
| `graphene_conductivity(freq, fermi_level_eV, scattering_time, temperature=300, interband=True)` | The Kubo surface conductivity in S |
| `drude_sheet_conductivity(freq, drude_weight, gamma)` | $i D / (\omega + i\gamma)$ |
| `sheet_layer_conductivity(medium, freq, thickness, eps_background=1)` | The sheet conductivity a layer of the medium carries, to check a fit |
| `sheet_rt(freq, sigma_s, n1=1, n2=1)` | Analytic `(R, T)` at normal incidence of a sheet between indices `n1` and `n2` |

`sheet_medium` and `graphene` take `eps_background` (the permittivity of the
layer without the sheet, 1 by default) and `return_fit=True`, which returns
a dict with the medium, the pole fit and `max_sigma_error`, the largest
relative error of the layer's sheet conductivity against the target over
the fitted samples.

The graphene conductivity follows Falkovsky (J. Phys. Conf. Ser. 129,
012004, 2008) and Hanson (J. Appl. Phys. 103, 064302, 2008). The intraband
Drude weight is $D = (2 e^2 k_B T / \pi\hbar^2) \ln(2\cosh(E_F / 2 k_B T))$,
which is $e^2 E_F / \pi\hbar^2$ at $T = 0$, with $\gamma = 1/\tau$. The
interband term is the finite-temperature integral, evaluated numerically
without broadening; at $T = 0$ it reduces to
$(e^2/4\hbar)[\Theta(\hbar\omega - 2E_F) + (i/\pi)\ln|(\hbar\omega - 2E_F)/(\hbar\omega + 2E_F)|]$.

## Placing the layer

Dispersive media are staircased, not sub-pixel smoothed: each grid voxel
takes the medium that fills most of it. A layer thinner than its cell
either fills the cell or disappears, and a layer of $N$ cells carries $N$
times the conductivity it was built for. So:

- give `thickness` equal to the cell size normal to the sheet (or use a
  mesh override so that the cell there equals the thickness);
- draw the layer as a box exactly `thickness` thick, centred on a grid node
  along the normal, so that it owns one plane of the tangential electric
  field.

## Example: graphene at terahertz frequencies

A free-standing graphene sheet ($E_F$ = 0.4 eV, $\tau$ = 0.1 ps, 300 K) under
normal incidence between 1.4 and 2.6 THz, with 40 cells per wavelength (the
layer is 3.7 um thick), against the analytic sheet formulas
$T = |2/(2 + \eta_0\sigma_s)|^2$ and $R = |\eta_0\sigma_s/(2 + \eta_0\sigma_s)|^2$.

```python
import numpy as np
import lightsim as ls
from lightsim import sheets

f0 = 2e12                                    # 2 THz, wavelength 150 um
freqs = np.linspace(1.4e12, 2.6e12, 5)
dx = ls.C0 / f0 / 40                          # the cell, and the layer thickness
g = sheets.graphene(fermi_level_eV=0.4, scattering_time=1e-13, thickness=dx, temperature=300,
                    band=(1.4e12, 2.6e12))

nx = 120
length, xs = nx * dx, 66 * dx                 # xs is a grid node

def run(with_sheet):
    sim = ls.Simulation(min=[0, 0, 0], max=[length, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx),
                        run_time=30e-12, shutoff=1e-7)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if with_sheet:
        sim.add_medium(g)
        sim.add_structure(ls.Box([xs - dx / 2, -1, -1], [xs + dx / 2, 1, 1]), "graphene")
    sim.add_source(ls.PlaneWave(0, 0.35 * length, "+", "Ey", ls.GaussianPulse(f0, 0.5e12)))
    sim.add_monitor(ls.FluxMonitor("R", 0, 0.2 * length, [0.2 * length, -1, -1], [0.2 * length, 1, 1], freqs))
    sim.add_monitor(ls.FluxMonitor("T", 0, 0.8 * length, [0.8 * length, -1, -1], [0.8 * length, 1, 1], freqs))
    return sim.run(device="cpu")

incident = np.asarray(run(False).flux("T"))
res = run(True)
R, T = -np.asarray(res.flux("R")) / incident, np.asarray(res.flux("T")) / incident
R_sheet, T_sheet = sheets.sheet_rt(freqs, sheets.graphene_conductivity(freqs, 0.4, 1e-13, 300))
for f, t, ts, r, rs in zip(freqs, T, T_sheet, R, R_sheet):
    print(f"{f / 1e12:.1f} THz  T = {t:.4f} (sheet {ts:.4f})  R = {r:.4f} (sheet {rs:.4f})")
```

```text
1.4 THz  T = 0.4087 (sheet 0.4093)  R = 0.1819 (sheet 0.1815)
1.7 THz  T = 0.4546 (sheet 0.4554)  R = 0.1679 (sheet 0.1673)
2.0 THz  T = 0.5007 (sheet 0.5018)  R = 0.1538 (sheet 0.1530)
2.3 THz  T = 0.5453 (sheet 0.5467)  R = 0.1402 (sheet 0.1392)
2.6 THz  T = 0.5872 (sheet 0.5890)  R = 0.1274 (sheet 0.1262)
```

## Validation

`python/tests/test_sheets.py` runs free-standing sheets at normal incidence
(a layer one cell thick, periodic sides, Gaussian pulse) and compares R and
T with the analytic sheet formulas above. Measured on the CPU:

| Case | Band | Result |
|---|---|---|
| Drude sheet (graphene intraband term, $E_F$ = 0.2 eV, $\tau$ = 0.1 ps), 40 cells per wavelength | 2.25 - 3.75 THz, T 0.73 - 0.86 | max abs(T - T_sheet) 7.1e-4, max abs(R - R_sheet) 2.8e-4 |
| the same, 80 and 160 cells per wavelength (layer 1.25 and 0.62 um) | | 1.8e-4 and 4.7e-5 (T): second order in the cell size |
| graphene Kubo, $E_F$ = 0.4 eV, $\tau$ = 0.1 ps, 300 K, 40 cells per wavelength | 1.4 - 2.6 THz, T 0.41 - 0.59 | conductivity fit 2e-9; max abs(T - T_sheet) 1.8e-3, max abs(R - R_sheet) 1.2e-3 |
| graphene Kubo, $E_F$ = 0.1 eV, $\tau$ = 0.1 ps, 300 K, interband threshold inside the band, 3 Lorentz poles | 35 - 65 THz, absorbance 0.7 - 1.8 % | conductivity fit 4.5e-4; absorbance 1 - R - T within 0.6 % of the formula, max abs(T - T_sheet) 1.0e-4 |

The same file checks the conductivity formula: the Drude weight at
$T = 0$, the finite-temperature interband integral at 1 K against the closed
$T = 0$ form (to 1e-4 of $e^2/4\hbar$), and the universal value
$e^2/4\hbar$ far above the threshold.

## Limits

- The layer's medium is isotropic: an electric field normal to the sheet
  also sees its permittivity, which a real sheet does not have. At normal
  incidence this does not matter. Oblique incidence has not been validated.
- The layer must be one cell thick (see above). A sheet on a coarse cell
  needs a fine mesh override around it, as for any thin film.
- A fitted conductivity (a general `sigma`, or graphene's interband term)
  holds only inside the fitted band. Outside it the poles still give a
  passive, causal response, but not the target one.
- Graphene's interband term is computed without broadening, and there is
  no spatial dispersion, magnetic-field (Hall) response or nonlinearity.
- A strong sheet on a thin layer gives a large plasma frequency; the solver
  reduces the time step if the pole needs it, which makes the run slower.
