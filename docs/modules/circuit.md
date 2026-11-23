# Circuit simulation

`lightsim.circuit` computes the S-matrix of a photonic circuit from a
netlist (a gdsfactory component's `get_netlist()`) and one S-parameter
model per instance. It is the counterpart of SAX as gplugins uses it
(`gplugins.sax`), with the same model convention, so models move between
the two unchanged.

## Conventions

- A **model** is a function of keyword arguments with defaults. The
  wavelength `wl` is in **micrometres** (scalar or array), lengths in
  micrometres, as in gdsfactory and SAX.
- It returns an **SDict**: `{(port_in, port_out): amplitude}`, the complex
  amplitude leaving `port_out` per unit amplitude entering `port_in`
  (missing pairs are zero). Port names may carry a mode index, `"o1@0"`,
  `"o1@1"`, for several modes per port.
- In matrix form `S[out, in] = sdict[(in, out)]`; `circuit.sdense` returns
  this order, which is the transpose of SAX's `sdense`.
- lightsim's S-parameter files (`write_sparameters`, EME) key the same
  entry as `"out@m,in@n"` with wavelengths in metres; gplugins' Tidy3D
  writer uses `"in@n,out@m"` and micrometres. `model_from_npz` reads both
  (it decides from the wavelength unit, or `key_order=`); for reciprocal
  devices the two orders give the same matrix.

These are SAX's conventions (SAX 0.18: a model `{("in0", "out0"): t}`
transmits from `in0` to `out0`). The lightsim models use numpy, or
jax.numpy when they are called with jax arrays, so they also run inside
SAX's jitted circuits; SAX models (jax) run in `lightsim.circuit`.

## A Mach-Zehnder from its layout

```python
import numpy as np
import gdsfactory as gf
from lightsim import circuit

gf.gpdk.PDK.activate()
mzi = gf.components.mzi(delta_length=40)
wg = lambda wl=1.55, length=10.0: circuit.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=2.0)
models = {"straight": wg, "bend_euler": wg, "mmi1x2": circuit.mmi1x2}
fn, info = circuit.circuit(mzi.get_netlist(), models)
wl = np.linspace(1.5, 1.6, 2001)
S = fn(wl=wl)
T = np.abs(S[("o1", "o2")]) ** 2
print(info["instances"]["cp1"], f"max {T.max():.3f}  min {T.min():.2e}")
dips = wl[1:-1][(T[1:-1] < T[:-2]) & (T[1:-1] < T[2:])]
print(f"FSR {np.diff(dips).mean() * 1e3:.1f} nm")
```

```text
mmi1x2 max 0.867  min 1.56e-07
FSR 14.2 nm
```

Each instance's model receives the instance `settings` and the numbers of
its `info` that match its arguments (the Euler bends get their path length
from `info`, as in SAX). `fn(wl=..., cp1={"loss_dB": 1.0})` overrides an
instance's arguments; a keyword that is not an instance name is passed to
every model that accepts it.

## The network solve

With `S` the block-diagonal matrix of all instance ports and `C` the
permutation that joins connected ports, the external S-matrix is

`S_ext = S_EE + S_EI C (I - S_II C)^-1 S_IE`,

solved at every wavelength (batched dense solve; sparse LU above 400
internal ports). Loops (rings, feedback) need nothing special. A recursive
netlist (`get_netlist(recursive=True)`) becomes nested sub-circuits for
cells without a model of their own. Ports that are neither connected nor
external are treated as matched (nothing enters them).

## Models

| Model | Ports | Arguments (as SAX where SAX has the model) |
|---|---|---|
| `straight` | o1, o2 | `wl, wl0, neff, ng, length, loss_dB_cm` |
| `bend` | o1, o2 | as `straight`, plus a lumped `loss_dB` |
| `phase_shifter` | o1, o2 | `wl, neff, voltage, length, loss, v_pi` |
| `attenuator` | o1, o2 | `wl, loss` |
| `coupler_ideal` | o1, o2 / o3, o4 | `wl, coupling` (bar o1-o4, o2-o3; cross o1-o3, o2-o4, amplitude `i sqrt(coupling)`) |
| `coupler` | as above | `wl, wl0, length, coupling0, dk1, dk2, dn, dn1, dn2` (SAX's dispersive coupler) |
| `coupler_ring` | o1, o4 bus; o2, o3 ring | `wl, coupling, loss_dB` |
| `mmi1x2`, `mmi2x2` | gdsfactory ports | `wl, wl0, fwhm, loss_dB, ...` |
| `grating_coupler` | o1 waveguide, o2 fibre | `wl, wl0, loss, reflection, reflection_fiber, bandwidth` |
| `ring_single`, `ring_double` | o1..o4 | closed-form all-pass and add-drop rings |
| `model_from_npz`, `model_from_sparameters` | from the file | interpolation in wavelength (magnitude and unwrapped phase, or real and imaginary parts) |
| `straight_from_cross_section` | o1, o2 | complex effective index fitted over wavelength from the mode solver: phase with full dispersion, loss from the imaginary part |
| `bend_from_cross_section` | o1, o2 | bend mode (index and radiation loss) and the two junction overlaps |
| `coupler_from_cross_section` | as `coupler` | supermode index difference fitted over wavelength |

`DEFAULT_MODELS` maps the common gdsfactory component names to these
models; `circuit(..., models)` adds to it.

Models from lightsim's own solvers:

```python
from lightsim.xsection import CrossSection

xs = CrossSection.strip(core_width=0.5e-6, step=25e-9)
wg = circuit.straight_from_cross_section(xs, wavelengths=(1.5, 1.55, 1.6))
print(f"neff(1.55) = {wg.neff(1.55).real:.4f}")
S = wg(wl=np.array([1.55]), length=100.0)
print(np.round(abs(S[("o1", "o2")]), 6))
```

```text
neff(1.55) = 2.4425
[1.]
```

An S-parameter file from `lightsim.sparams.write_sparameters` or from
[EME](eme.md) becomes a model with `model_from_npz(path)`.

## Validation

`python/tests/test_circuit.py` compares the solver with closed forms, all
to better than 1e-12 over 801 wavelengths: a Mach-Zehnder built from
gdsfactory's `mzi` netlist (arm lengths read from the layout), an all-pass
ring and an add-drop ring with loss (against `ring_single` /
`ring_double`), and a four-coupler lattice filter (against the product of
2x2 transfer matrices). A two-mode circuit checks mode-by-mode
connections, and two files with the same device in lightsim's and
gplugins' key orders and units give the same model.

Against SAX 0.18 (gplugins' `test_mzi` models and gdsfactory netlists,
1000 wavelengths, laptop CPU):

| Circuit | Largest difference | SAX first call / repeat | lightsim first call / repeat |
|---|---|---|---|
| `mzi` | 1.2e-13 | 1842 ms / 13.7 ms | 42 ms / 27.9 ms |
| four `mzi` in series (recursive netlist) | 3.9e-13 | 901 ms / 56.8 ms | 115 ms / 112.6 ms |
| `ring_single` | 2.5e-12 | 694 ms / 9.5 ms | 6 ms / 5.2 ms |

SAX compiles the circuit on the first call (JAX) and is then about twice
as fast for the MZIs; lightsim has no compilation step. SAX also gives
gradients through JAX; lightsim's circuit functions do not.

## Limits

- Linear, steady-state (frequency-domain) only. Time-domain signals,
  modulator dynamics, and power-dependent effects (Kerr, two-photon and
  free-carrier absorption, self-heating) need a transient or nonlinear
  circuit solver, which lightsim does not have.
- No automatic differentiation (use SAX with the same models for
  gradients).
- Analytic models are as good as their parameters: take `neff`, `ng`,
  coupling and loss from the mode solver, EME or FDTD.
