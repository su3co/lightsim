# Gratings: diffraction-order analysis

`lightsim.gratings` is the equivalent of Lumerical's *grating order
transmission* analysis. It takes the frequency-domain fields of a
`DFTMonitor` that spans one unit cell of a periodic (or Bloch-periodic)
simulation and decomposes them into Floquet (diffraction) orders: direction,
propagating/evanescent flag, s/p polarisation split and power of each order,
for waves travelling along `+axis` and `-axis` separately.

It is pure Python post-processing (numpy only); no solver changes are needed.

```python
import lightsim as ls
from lightsim import gratings as gr
```

The module is not re-exported from `lightsim/__init__.py`; import it as above.

## Setting up a grating simulation

* Periodic boundaries along the grating period(s); `ls.Bloch(k)` with
  `k = n_inc k0 sin(theta)` for oblique incidence, PML along the propagation
  axis. The cell must be exactly one period.
* A 1-D grating is simulated as a 2-D-equivalent cell: one or two cells along
  the invariant axis, periodic.
* One `DFTMonitor` per analysis plane, zero thickness along the propagation
  axis, covering the whole cell, recording the four tangential components.
  The helper below builds the unit cell of the lamellar grating of the
  validation section, with a transmission plane `"t"` and a reflection plane
  `"r"`:

```python
lam = 1e-6
f0 = ls.C0 / lam
dx = lam / 40
period, wz, x_t, x_r = 1.5 * lam, 2 * dx, 3.5e-6, 0.5e-6
freqs = [f0]


def unit_cell(grating):
    sim = ls.Simulation(min=[0, 0, 0], max=[4.5e-6, period, wz], mesh=ls.UniformMesh(dx), run_time=120e-15)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if grating:  # ridges of index 1.5, fill 0.5, depth lam / 2, on a substrate of index 1.5
        sim.add_structure(ls.Box([2e-6, 0.25 * period, -1], [2.5e-6, 0.75 * period, 1]), ls.Medium("ridge", n=1.5))
        sim.add_structure(ls.Box([2.5e-6, -1, -1], [1, 1, 1]), ls.Medium("sub", n=1.5))
    sim.add_source(ls.PlaneWave(0, 1e-6, "+", "Ez", ls.GaussianPulse(f0, 0.15 * f0)))
    sim.add_monitor(ls.DFTMonitor("t", [x_t, 0, 0], [x_t, period, wz], freqs, ("Ey", "Ez", "Hy", "Hz")))
    sim.add_monitor(ls.DFTMonitor("r", [x_r, 0, 0], [x_r, period, wz], freqs, ("Ey", "Ez", "Hy", "Hz")))
    return sim
```

* The medium at the monitor plane must be homogeneous, isotropic and lossless
  (index `n_medium`).
* The plane-wave source is one-sided, so a monitor behind the source sees only
  reflected light; because the analysis splits forward and backward waves, the
  reflection monitor may also sit between the source and the grating.

## API

### `diffraction_orders`

```text
gr.diffraction_orders(results, monitor, axis, period_axes, freqs=None, n_medium=1.0,
                      kx_inc=0.0, ky_inc=0.0, *, sim, incident_power=None,
                      direction="net", max_order=None) -> GratingOrders
```

| argument | meaning |
| --- | --- |
| `results`, `monitor` | `Results` and the name of the DFT monitor |
| `axis` | plane normal (0, 1, 2) |
| `period_axes` | one in-plane axis (1-D grating; fields are averaged along the other, invariant, in-plane axis) or both in-plane axes (2-D grating, orders `(m, n)`) |
| `freqs` | frequencies to analyse (must be monitor frequencies); `None` = all |
| `n_medium` | refractive index at the plane |
| `kx_inc`, `ky_inc` | in-plane wavevector of the incident wave along `period_axes[0]`, `period_axes[1]` (rad/m): the `k` of the `Bloch(k)` boundaries, 0 for periodic boundaries |
| `sim` (required keyword) | the `Simulation` (or the node arrays from `Simulation.grid()`); supplies sample coordinates and the period |
| `incident_power` | power to normalise `fraction` to (scalar or one per frequency), see below |
| `direction` | power that `fraction` refers to: `"net"`, `"+"` (along +axis) or `"-"` (along -axis, positive number) |
| `max_order` | default: only orders propagating at one of the analysed frequencies; an int (or one per period axis) returns all `|m| <= max_order`, evanescent included |

Order `(m, n)` has in-plane wavevector
`k_t = (kx_inc + 2 pi m / Lx, ky_inc + 2 pi n / Ly)` and normal wavevector
`k_n = sqrt((n_medium k0)^2 - |k_t|^2)`; it is propagating when `|k_t| < n_medium k0`.

**Normalisation of `fraction`.** With `incident_power=None`, `fraction` is
normalised to the *total power through the plane* in the chosen direction (the
sum over the propagating orders), so the fractions sum to 1. With
`incident_power` given (the *source power*, usually from
`gr.incident_power` on an empty reference run), `fraction` is the diffraction
efficiency of each order.

`GratingOrders` fields (per-order arrays have shape `(nf, n_orders)`):

| field | meaning |
| --- | --- |
| `freqs`, `orders` | frequencies; integer `(m, n)` per order (`n = 0` for a 1-D grating) |
| `k_t`, `k_n` | transverse wavevector along in-plane axes `(axis+1)%3`, `(axis+2)%3`; complex normal wavevector |
| `propagating` | bool |
| `theta`, `phi` | polar angle from the plane normal (nan if evanescent) and azimuth of `k_t` |
| `angles` | signed angles `asin(k_t / k)` along the two in-plane axes (grating-equation angle) |
| `power_forward`, `power_backward`, `power` | power along +axis, along -axis (both >= 0) and net; same units as `Results.flux` |
| `power_s_forward`, `power_p_forward`, `power_s_backward`, `power_p_backward` | s (TE, E normal to the plane of incidence) / p (TM) split |
| `amp_s_forward`, ..., `amp_p_backward` | complex amplitudes (E for s, eta0 H for p) at the E-sample plane, transverse origin `r_t = 0` |
| `fraction`, `norm` | power fraction per order and its normalisation |
| `total`, `total_forward`, `total_backward` | sums over orders, shape `(nf,)` |
| `index(m, n=0)`, `efficiency(m, n=0)`, `as_dict(fi=0)` | lookup helpers |

For an order at exactly normal incidence (`k_t = 0`) the s/p labels are a
convention: s = E along in-plane axis `(axis+2)%3`.

### Helpers

```text
gr.incident_power(reference, monitor, axis, period_axes, freqs=None, n_medium=1.0,
                  kx_inc=0.0, ky_inc=0.0, *, sim, direction="+")        # -> (nf,)
gr.grating_efficiencies(results, t_monitor, r_monitor, axis, period_axes, freqs=None,
                        n_t=1.0, n_r=1.0, kx_inc=0.0, ky_inc=0.0, *, sim,
                        incident_power, direction="+", max_order=None)   # -> dict
gr.grating_angles(freq, period, n_medium=1.0, k_inc=0.0, orders=None)   # -> (orders, angles)
```

* `incident_power`: source power per unit cell from a run of the same cell
  without the grating.
* `grating_efficiencies`: returns `{"T": GratingOrders, "R": GratingOrders,
  "T_total": (nf,), "R_total": (nf,)}`; `T.fraction` / `R.fraction` are the
  diffraction efficiencies (transmission = waves travelling along `direction`
  at `t_monitor`, reflection = waves travelling against it at `r_monitor`).
* `grating_angles`: the grating equation
  `sin(theta_m) = (k_inc + 2 pi m / period) / (n k0)` for a 1-D grating.

### Example

```python
empty_cell, cell = unit_cell(False), unit_cell(True)
ref = empty_cell.run()
p_inc = gr.incident_power(ref, "t", 0, 1, [f0], 1.0, sim=empty_cell)
res = cell.run()
eff = gr.grating_efficiencies(res, "t", "r", 0, 1, [f0], n_t=1.5, n_r=1.0, sim=cell, incident_power=p_inc)
eff["T"].as_dict()        # {(m, 0): efficiency} of the propagating transmitted orders
eff["T"].angles[0, :, 0]  # their angles in the substrate
```

## Method

Fields vary as `exp(-i w t)`, so a wave along +axis is `exp(+i k_n x_n)` and a
Bloch field obeys `F(r + L) = F(r) exp(i k L)`.

1. Each tangential component is Fourier transformed at its own Yee sample
   positions, `F_m = (1/L) sum_j w_j F(x_j) exp(-i k_m x_j)`, which removes the
   half-cell transverse stagger between components exactly.
2. For every order the four tangential amplitudes are projected on the unit
   vectors along `k_t` and `s = n x k_t` and solved for the forward/backward
   s and p plane-wave amplitudes. The tangential E samples lie on the monitor
   plane and the tangential H samples half a cell away; the offset `delta` is
   included analytically (`exp(+-i k_n delta)`), so the split is exact for
   standing waves too.
3. Order powers use the same discrete Poynting pairing as the core
   `FluxMonitor` (E on the plane times H half a cell away):
   `P = A |a|^2 Re(k_n) cos(k_n delta) / (2 eta0 k0)` for s (divided by `n^2`
   with `a = eta0 H` for p). The sum over orders therefore reproduces the flux
   monitor and energy is conserved on the grid; the difference from the
   continuum expression is `O((k_n dx)^2 / 8)` (0.3 % at 40 cells per
   wavelength at normal incidence, and it largely cancels in efficiencies).

Evanescent orders are reported with zero power.

## Validation

`python/tests/test_gratings.py` (CPU backend, about 85 s). lambda = 1 um,
dx = lambda/40 unless stated.

| test | check | result |
| --- | --- | --- |
| no grating, normal incidence, Ez and Ey | order-0 fraction; sum of orders vs `FluxMonitor`; backward power; s/p split | 1 to < 1e-6; agreement 3e-7 (tolerance 1e-3); backward < 1e-6; > 0.999 in the launched polarisation |
| no grating, 30 deg (Bloch), two frequencies | order-0 fraction and angle; sum vs flux; other orders vs grating equation | fraction 1 to < 1e-6, angle exact (30 deg at f0, asin(sin 30 deg f0/f) at 1.04 f0); sum/flux agreement 6e-8 |
| no grating, 2-D period axes (dx = lambda/20) | (0, 0) fraction, 9 propagating orders, theta/phi of (1, -1) | 1 to < 1e-6, sum/flux 9e-8 |
| lamellar grating, TE, vs RCWA | efficiencies of all propagating orders | max abs error 0.0043 (tolerance 0.01) |
| same | energy conservation, sum of orders vs flux monitors | T + R = 0.99993; < 1e-3 |
| lamellar grating, TM | energy conservation, p-only output, +-1 symmetry | T + R = 0.99995 |

Grating: period 1.5 lambda, fill 0.5, depth 0.5 lambda, n = 1.5 ridges on an
n = 1.5 substrate, incidence from air. The reference is a single-layer
Moharam/Gaylord RCWA for TE (41 harmonics, in the test file; checked there
against Fresnel and for convergence).

| order | FDTD | RCWA |
| --- | --- | --- |
| T0 | 0.5598 | 0.5555 |
| T+-1 | 0.1869 | 0.1878 |
| T+-2 | 0.0202 | 0.0209 |
| R0 | 0.0256 | 0.0264 |
| R+-1 | 0.0002 | 0.0003 |

TM efficiencies of the same grating (no RCWA reference, energy conservation
only): T0 0.6196, T+-1 0.1701, T+-2 0.0066, R0 0.0268, R+-1 0.0001.

### Oblique incidence on the grating (run by hand, not in the test suite)

Same grating, TE, 20 deg incidence (Bloch boundaries, 0.06 f0 pulse, 350 fs).
The orders are no longer symmetric, which checks the sign convention of the
order index against RCWA.

| order | FDTD dx = lambda/40 | FDTD dx = lambda/64 | RCWA |
| --- | --- | --- | --- |
| T-2 | 0.0061 | 0.0060 | 0.0060 |
| T-1 | 0.2187 | 0.2186 | 0.2185 |
| T0 | 0.4710 | 0.4660 | 0.4565 |
| T+1 | 0.2777 | 0.2824 | 0.2913 |
| R-2 | 0.0019 | 0.0018 | 0.0018 |
| R-1 | 0.0039 | 0.0041 | 0.0046 |
| R0 | 0.0209 | 0.0212 | 0.0214 |
| T + R | 1.0001 | 1.0002 | 1 |

The largest deviation (T0 / T+1, 1.4 % absolute at lambda/40, 0.9 % at
lambda/64) shrinks roughly linearly with the cell size. Power is conserved
and the no-grating decomposition is exact, so this is attributed to the
discretisation of the ridges in the solver rather than to the order
analysis; that attribution has not been verified further.

## Limitations

* Homogeneous, isotropic, lossless medium at the monitor plane; real `n_medium`.
* Rectangular unit cells only; `kx_inc`, `ky_inc` are frequency independent
  (as in a Bloch simulation), so the incidence angle varies with frequency.
* Orders exactly at a Rayleigh anomaly (`k_n = 0`) carry no power and are
  reported as non-propagating.
* The core plane-wave source tilts only in s polarisation (E normal to the
  plane of incidence); the analysis itself handles any polarisation.
