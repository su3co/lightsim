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

`S_ext = S_EE + S_EI C (I - S_II C)^-1 S_IE`.

lightsim does not assemble that matrix. It joins the instances one
connection at a time (the sub-network growth of Filipsson and Gunnar):
joining two different blocks at one port each costs one complex division
per frequency, joining two ports of one block a 2x2 elimination. The order
is chosen once per circuit from the graph (at every step the join that
leaves the smallest block), disconnected parts stay separate, and ports
that are neither connected nor external are dropped (they are matched:
nothing enters them). The joins run in compiled code, in parallel over the
frequencies (and any other batch axes), on lanes of points that the
processor's vector units process together. Loops (rings, feedback) need
nothing special. The result equals the dense solve to rounding (better
than 3e-15 relative on the spread of topologies of
`benchmarks/circuit/generality.py`); `backend="dense"` (batched
dense solve) and `backend="sparse"` (sparse LU) remain, and `auto` falls
back to them when a port is connected twice or when the join order would
cost more operations than the dense solve. A recursive netlist
(`get_netlist(recursive=True)`) becomes nested sub-circuits for cells
without a model of their own.

### Many calls: Monte Carlo, sweeps, optimisation

Model arguments may be arrays: `fn(wl=wl[None, :], wg1={"width": w[:, None]})`
evaluates every trial of `w` at every wavelength in one call (the output has
the broadcast shape). Four options of `circuit` help when a circuit is
called many times:

- `threads=N`: threads of the network solve (default: all cores); for calls
  with many points (above about 33,000) the instances' models are also
  evaluated on `N` threads.
- `reuse_identical=True`: instances that call the same model with the same
  arguments share one evaluation, and an instance whose arguments equal
  those of the previous call is not evaluated again (numbers by value,
  numpy arrays by their contents), so a trial that perturbs a few instances
  re-evaluates only those. For models that are pure functions of their
  arguments.
- `stack_instances=True`: all instances of one model are evaluated in one
  call, the arguments that differ between them stacked along a new leading
  axis. For models written with numpy operations that broadcast; the first
  stacked call of each group is checked against separate calls, and a group
  that does not give the same result is evaluated instance by instance.
- `trace_models=True` (off by default): every model is called once per call
  with a traced wavelength (or frequency); what does not depend on it is
  computed by numpy as usual, and the operations on it are compiled into the
  network solve and evaluated per point there, without numpy temporaries.
  The first traced call of each model returns the direct result, after
  checking the trace against direct calls on a uniform subsample of the
  sweep (at most 256 points along each axis); a model that cannot be
  traced, or whose result differs, is called directly from then on. Calls
  of fewer than 8192 points are never traced, and models whose traced code
  is a few operations without the exponential recurrence below are called
  directly too. With `reuse_identical`, a model whose other arguments are plain
  numbers is traced once and its code reused with each new wavelength.
  On the resonator-chip Monte Carlo of the benchmarks it is 1.1 to 1.5 times
  faster than the untraced options on one core and 1.6 to 1.9 times on eight
  (GB10), at the call size that suits each, and no slower at any measured
  call size. On the photonic circuits, whose models are a single
  exponential, it changes nothing.

  Accuracy: an exponential of a phase that is linear (or quadratic) in the
  wavelength or in its inverse, on a grid uniform in that variable, is
  computed by a recurrence (a true exponential every 32 points, products in
  between), and a phase is only known to about eps times its size whichever
  way it is computed: the traced results differ from the direct numpy call
  by about 1.5 eps times the largest phase (measured: 3e-12 relative at
  1e4 rad, 3e-11 at 1e5 rad, about twice numpy's own rounding of that
  phase). Near a sharp resonance a model can be more sensitive to the order
  of its operations than that (the resonator chip of the benchmarks, 250 trials
  of 5001 points and 2000 refinement rows of 257: up to 1e-10 of the largest
  entry, 6e-10 of the entry itself at the bottom of a notch, where the
  direct call itself moves 8e-11 and 8e-10 for a 4-ulp change of the
  wavelength). The results do not depend on the number of threads.
  Leave the option off where results must match the direct calls to
  the last bits.

Microwave models take the frequency `f` in hertz instead of `wl`:
`fn(f=freq)` passes `f` to the models that take it and `wl = c / f` (µm) to
those that take only `wl`. With `wl` given, a model that takes only `f`
receives `f = c / wl` (earlier versions left it at its own default `f`); a
model that takes both `wl` and `f` receives each as given, as before.

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

Against SAX 0.18, on the same machine with the same models, netlists and
inputs: see [Speed against other packages](../comparison.md), which
`scripts/make_comparison.py` regenerates from `benchmarks/circuit/`. On the
quantum RF PDK's resonator-chip Monte Carlo (42 instances, 5001
frequencies, 25 trials with refinement) lightsim takes 0.19 s on one core
of a GB10 against 0.30 s for SAX's jitted Filipsson-Gunnar back end after
its 6 s compilation, and the results agree to 1.4e-11. On gdsfactory's MZI
netlists repeated calls are 1.1 to 4.9 times faster than jitted SAX with
the default options (1.5 to 13 times with `reuse_identical` and
`stack_instances`); the closest case is four MZIs in series at 100
wavelengths with the defaults (0.30 ms against 0.33 ms; 0.21 ms with both
options).
SAX gives gradients through JAX; lightsim's circuit functions do not.

## Limits

- Linear, steady-state (frequency-domain) only. Time-domain signals,
  modulator dynamics, and power-dependent effects (Kerr, two-photon and
  free-carrier absorption, self-heating) need a transient or nonlinear
  circuit solver, which lightsim does not have.
- No automatic differentiation (use SAX with the same models for
  gradients).
- Analytic models are as good as their parameters: take `neff`, `ng`,
  coupling and loss from the mode solver, EME or FDTD.
