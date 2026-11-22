# Sheet materials

`lightsim.sheets` models two-dimensional conducting sheets, such as
graphene, given by a surface conductivity $\sigma_s(\omega)$ in siemens.
There are two routes:

- **Zero-thickness sheets** (the default): `sheets.graphene(...)`,
  `sheets.sheet_medium(...)` and `sheets.sheet_conductivity(...)` without a
  `thickness` return a surface conductivity, which `lightsim.Sheet` places
  on a plane, a rectangle or a polygon. The solver applies the surface
  current on a surface with no thickness; nothing has to be aligned with the
  mesh.
- **One-cell layers** (with `thickness`): a thin volumetric layer whose
  permittivity carries the same sheet current, one cell thick and centred on
  a grid node (the route lightsim had before zero-thickness sheets; it is
  kept, see [One-cell layers](#one-cell-layers)).

The module is not re-exported from the package root; import it as
`from lightsim import sheets`. It needs `numpy`, and `scipy` for the
interband integral and for fitted conductivities. Conventions are those of
the core: SI units and fields $\propto e^{-i\omega t}$, so a lossy sheet has
$\mathrm{Re}\,\sigma_s > 0$ and an inductive one $\mathrm{Im}\,\sigma_s > 0$.

## Zero-thickness sheets

A `Sheet` is the plane $x_a = p$ normal to an axis $a$, or a rectangle or a
polygon on that plane, that carries the surface current
$\mathbf{J}_s = \sigma_s(\omega)\,\mathbf{E}_t$ (tangential field only; the
normal field sees no sheet) with

$$
\sigma_s(\omega) = \sigma_0 - i\omega\varepsilon_0\chi_0
 + \sum \frac{iD}{\omega + i\gamma}
 + \sum \frac{-i\omega s}{\omega_0^2 - \omega^2 - i\gamma\omega} ,
$$

a constant conductance $\sigma_0$ (S), a constant surface susceptibility
$\chi_0$ (m), Drude terms (Drude weight $D$ in S/s) and Lorentz terms
(strength $s$ in S rad/s; the conductance at resonance is $s/\gamma$). Every
term must be passive (non-negative parameters).

| Function | Returns |
|---|---|
| `ls.Sheet(axis, position, conductivity, min=, max=, polygon=, name=)` | A sheet for `Simulation.add_sheet`. Without `min` / `max` / `polygon` it fills the plane; `min` / `max` (box corners, the components along `axis` ignored) make a rectangle; `polygon` takes `(u, v)` vertices in the two other axes in increasing order |
| `sheets.SheetConductivity(conductance=0, susceptibility=0, poles=())` | A surface conductivity from its terms; `sheets.DrudeSheetPole(D, gamma)` and `sheets.LorentzSheetPole(s, omega0, gamma)` make the poles |
| `sheets.sheet_conductivity(name, conductance=, drude_weight=, gamma=)` | A constant conductance and/or a Drude sheet (exact at every frequency) |
| `sheets.sheet_conductivity(name, sigma=f, band=(f_min, f_max))` | Any $\sigma_s(f)$ (a function of frequency in Hz, or an array with `freqs=`) fitted over the band with `n_lorentz` Lorentz terms, a Drude term (`drude=True`) and a constant susceptibility; `return_fit=True` adds `max_sigma_error` |
| `sheets.graphene(fermi_level_eV, scattering_time, temperature=300, band=None)` | Graphene from the Kubo formula: the intraband term is one exact Drude term; with `band`, the interband term is added as `n_lorentz` fitted Lorentz terms and a susceptibility |
| `sheets.sheet_medium(name, drude_weight=, gamma=)` | Without `thickness`, the same as `sheet_conductivity` |
| `sheets.sigma_of(conductivity, freq)` | $\sigma_s$ of a conductivity dict |
| `sheets.sheet_rt(freq, sigma_s, n1, n2, angle=0, polarization="s")` | Analytic `(R, T)` of a sheet between indices `n1` and `n2`, at normal or oblique incidence |

In JSON the sheets are the `sheets` array of the simulation (see
[JSON format](../reference/json-spec.md)).

### How the sheet enters the update

The surface polarization $P_s = \int J_s\,dt$ of each pole obeys the
equation of a volume pole of the same form,
$\ddot P_s + \gamma \dot P_s + \omega_0^2 P_s = s E_t$ (Drude: $\omega_0 = 0$,
$s = D$), stepped by the same explicit scheme as the dispersive volume
media, $P^{n+1} = c_0 P^n + c_1 P^{n-1} + c_2 E^n$. The current enters
Ampère's law on the plane of the tangential E nodes at the sheet, divided by
the dual cell size $d$ normal to the sheet there, which the plan takes from
the mesh (uniform or not). With $k = c/(\varepsilon (1 + f) d)$ per node
($\varepsilon$ the node's permittivity, $f$ its own conduction term, $c$ the
fraction of the node's in-plane cell the sheet covers) and
$h = \sigma_0\,\Delta t/2\varepsilon_0$, the node's field after the linear
update $E_\text{lin}$ becomes

$$
E^{n+1}\,\bigl(1 + k(h + \chi_0)\bigr) = E_\text{lin} - k\sum_q \bigl(P_q^{n+1} - P_q^n\bigr) - k\,(h - \chi_0)\,E^n ,
$$

so the constant conductance and susceptibility are centred in time
(implicit, unconditionally stable) and the poles are explicit. The result
does not depend on the mesh step normal to the sheet beyond the ordinary
discretisation error: there is no thickness.

**Position.** Uniform and automatic meshes put a grid node on every sheet
plane: the node nearest to the sheet moves onto it and the move is tapered
over 8 cells on each side, so no cell changes size by more than 6.25 % (the
time step follows the smallest cell). The user aligns nothing and the scheme
is of second order in the mesh step. Explicit meshes are not moved; a sheet
between two nodes is then shared between the two tangential-E planes around
it by linear weights (the plane at distance $w\delta$ carries the fraction
$1 - w$ of the conductance, $\delta$ the node spacing). That keeps the first
moment of the current at the sheet, but the two half-sheets interact across
$\delta$, which adds a relative error of about
$w(1 - w)\,k\delta\,|\eta\sigma_s|$ to the effective conductance: first order
in $\delta$ (measured below). A sheet within half a cell of the end of the
grid, or one whose node move would disturb a node already put on another
sheet, is also left between nodes.

**Patches.** A node whose in-plane cell a rectangle covers by a fraction
$c$ carries $c$ times the current (exact overlap); for a polygon the
fraction is sampled on 16 × 16 points per cell.

**Time step.** The poles of a sheet on a node of dual size $d$ act as volume
poles $\chi_s/d$; if their explicit update needs a shorter step than the
Courant step (a strong sheet on a fine normal mesh), the solver reduces the
time step, as it does for dispersive media.

### Example: graphene on a substrate

Graphene ($E_F$ = 0.4 eV, $\tau$ = 0.1 ps, 300 K) on a substrate of index
1.95, under normal incidence between 1.4 and 2.6 THz, 40 cells per
wavelength, the sheet at a position that is not a node of the uniform mesh,
against the analytic sheet formulas.

```python
import numpy as np
import lightsim as ls
from lightsim import sheets

f0 = 2e12                                   # 2 THz, wavelength 150 um
freqs = np.linspace(1.4e12, 2.6e12, 5)
g = sheets.graphene(fermi_level_eV=0.4, scattering_time=1e-13, temperature=300, band=(1.4e12, 2.6e12))
dx = ls.C0 / f0 / 40                        # 40 cells per wavelength
length = 120 * dx
xs = 0.537 * length                         # anywhere: the mesh puts a node on the sheet

def run(with_sheet):
    sim = ls.Simulation(min=[0, 0, 0], max=[length, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx),
                        run_time=30e-12, shutoff=1e-7)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if with_sheet:
        sim.add_structure(ls.Box([xs, -1, -1], [2 * length, 1, 1]), ls.Medium("SiO2_THz", n=1.95))
        sim.add_sheet(ls.Sheet("x", xs, g))
    sim.add_source(ls.PlaneWave(0, 0.35 * length, "+", "Ey", ls.GaussianPulse(f0, 0.5e12)))
    sim.add_monitor(ls.FluxMonitor("R", 0, 0.2 * length, [0.2 * length, -1, -1], [0.2 * length, 1, 1], freqs))
    sim.add_monitor(ls.FluxMonitor("T", 0, 0.8 * length, [0.8 * length, -1, -1], [0.8 * length, 1, 1], freqs))
    return sim.run(device="cpu")

incident = np.asarray(run(False).flux("T"))
res = run(True)
R, T = -np.asarray(res.flux("R")) / incident, np.asarray(res.flux("T")) / incident
R_a, T_a = sheets.sheet_rt(freqs, sheets.graphene_conductivity(freqs, 0.4, 1e-13, 300), n1=1.0, n2=1.95)
for f, t, ta, r, ra in zip(freqs, T, T_a, R, R_a):
    print(f"{f / 1e12:.1f} THz  T = {t:.4f} (sheet {ta:.4f})  R = {r:.4f} (sheet {ra:.4f})")
```

```text
1.4 THz  T = 0.4757 (sheet 0.4763)  R = 0.2793 (sheet 0.2794)
1.7 THz  T = 0.5170 (sheet 0.5179)  R = 0.2618 (sheet 0.2620)
2.0 THz  T = 0.5567 (sheet 0.5579)  R = 0.2448 (sheet 0.2453)
2.3 THz  T = 0.5937 (sheet 0.5950)  R = 0.2287 (sheet 0.2298)
2.6 THz  T = 0.6274 (sheet 0.6284)  R = 0.2139 (sheet 0.2158)
```

### Accuracy

Validation cases `sheet_normal_incidence`, `sheet_oblique`, `sheet_plasmon`
and `sheet_ribbon` (see [Validation](../validation.md)) and
`python/tests/test_sheets_zero.py`, measured on the CPU:

| Case | Reference | Result |
|---|---|---|
| Drude sheet ($\eta_0\lvert\sigma_s\rvert \approx 1$) and a sheet with every term (conductance, susceptibility, Drude and Lorentz terms) between n = 1.4 and n = 2, normal incidence, 0.85–1.15 f₀, sheet off the nodes of the uniform mesh | sheet formulas | max \|ΔR\|, \|ΔT\| at 20, 40, 80 cells per vacuum wavelength: 5.4e-3, 9.0e-4, 1.7e-4 (Drude) and 6.2e-3, 1.8e-3, 3.4e-4 (all terms): second order |
| the same on explicit uniform meshes (sheet shared by two planes, w = 0.3, 0.6, 0.2) | sheet formulas | 8.7e-3, 6.9e-3, 2.2e-3 (Drude) and 1.4e-2, 6.6e-3, 1.6e-3 (all terms): first order, as estimated above |
| energy balance, the same runs | R + T + A = 1, A from the sheet current (½ Re σ_s \|E_t\|² on the node planes) | 6e-4 (20 cells) to 2e-5 (80 cells) |
| explicit non-uniform mesh changing from λ/40 to λ/70 at the sheet | sheet formulas | 9e-4 (R), 1.5e-3 (T) |
| graphene (Kubo, interband fitted) and a Drude sheet between n = 1.45 and 1.95, 1.4–2.6 THz (Python) | sheet formulas | max \|ΔT\| 5.7e-3, 1.5e-3, 3.1e-4 at 30, 60, 120 cells per wavelength (graphene) |
| 2-D, Bloch-periodic, s and p at 30° and 60°, sheet with every term between vacuum and n = 1.5 | oblique sheet formulas | R within 4e-4, T within 3.5e-3 at 60 cells per wavelength |
| 2-D sheet plasmon (Drude sheet between vacuum and ε = 2.25, q ≈ 8 k₀), wavenumber fitted from the field on the sheet at 0.85, 1.0, 1.15 f₀ | retarded dispersion $1/\kappa_1 + \varepsilon_2/\kappa_2 = -i\sigma_s/(\omega\varepsilon_0)$ (the quasi-static $q = i\varepsilon_0(\varepsilon_1 + \varepsilon_2)\omega/\sigma_s$ is 1–2 % lower) | Re q within 0.4–1.3 % at 30 cells per plasmon wavelength, 0.005–0.26 % at 60 (second order); Im q within 0.4–1.5 % |
| periodic array of strips (2-D patches, width 20 nm, period 60 nm, on ε = 2.25), E across the strips | quasi-static eigenvalue of the strip array (periodic line-charge Green's function, 400 elements; the Drude weight is chosen to put it at f₀) | absorption resonance at 0.954, 0.979, 0.982, 0.994 f₀ with 10, 20, 40, 80 cells across the strip (first-order, not monotonic: the field is singular at the strip edges) |
| a rectangle covering the periodic cell | the full plane | identical |

### Engines and limits

- CPU and GPU engines (one GPU, several GPUs, several nodes), fused and split
  GPU kernels, complex fields (Bloch runs), 2-D simulations (a sheet is then
  a line, normal to an in-plane axis) and double precision. The update of a
  node depends on that node only, so a sheet whose two planes fall on two
  slabs of a domain decomposition needs no extra exchange (the slab tests
  reproduce single-slab runs to 1e-5).
- Refused: a sheet normal to the invariant axis of a 2-D simulation; a sheet
  on a cell of a medium with off-diagonal permittivity (rotated, gyrotropic or
  director media); an active term (negative conductance, susceptibility or
  pole parameter); polygons with fewer than three vertices; sheets with time
  reversal; and every gradient engine (S-matrix session, reverse mode,
  `invdes`, `mode_power_gradient`): the sheet's surface polarization is
  state that they do not carry. A one-cell layer can stand next to a design
  region instead.
- The sheet has no magnetic-field (Hall) response, no spatial dispersion and
  no nonlinearity. A fitted conductivity holds only inside its band.
- Mirror symmetry (`run_symmetric`) drops a sheet normal to a symmetry axis
  that lies in the removed half; a sheet on the symmetry plane is kept.
- This is lightsim's own formulation (surface polarization on the
  tangential-E node planes); it is not Lumerical's 2-D material code, which
  has its own discretisation.

## One-cell layers

With a `thickness`, a sheet is a thin volumetric layer of thickness $t$ whose
permittivity carries the same sheet current:

$$
\varepsilon(\omega) = \varepsilon_b + \frac{i\,\sigma_s(\omega)}{\varepsilon_0\,\omega\,t} .
$$

The layer occupies one cell of the mesh, and its result depends on placing
it there (see below). Gradient engines that support dispersive media accept
it.

### API

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

### Placing the layer

Dispersive media are staircased, not sub-pixel smoothed: each grid voxel
takes the medium that fills most of it. A layer thinner than its cell
either fills the cell or disappears, and a layer of $N$ cells carries $N$
times the conductivity it was built for. So:

- give `thickness` equal to the cell size normal to the sheet (or use a
  mesh override so that the cell there equals the thickness);
- draw the layer as a box exactly `thickness` thick, centred on a grid node
  along the normal, so that it owns one plane of the tangential electric
  field.

### Example: graphene at terahertz frequencies

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

### Validation

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

### Limits

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
