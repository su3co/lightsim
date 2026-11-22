# Quick start

After this page you can compute the S-parameters of a gdsfactory component in
a few lines, and build, run and read a simulation by hand. Both examples run
on a laptop CPU in seconds. They assume the package is built as described in
[Install](install.md).

## S-parameters of a gdsfactory component

[gdsfactory](https://gdsfactory.github.io/gdsfactory/) describes photonic
layouts as components with named ports. `write_sparameters` takes a component,
builds the 3-D simulation from its polygons and the layer stack of the active
process design kit (PDK), solves the waveguide mode at every port (the field
pattern the waveguide guides) and returns the scattering matrix: the complex
amplitude that leaves through one port per unit amplitude sent into another.

```python
import gdsfactory as gf
import numpy as np
from lightsim.sparams import write_sparameters

gf.gpdk.PDK.activate()                               # gdsfactory's generic PDK
c = gf.components.straight(length=2)                 # 2 um of 500 x 220 nm silicon strip
sp = write_sparameters(c, wavelengths=np.linspace(1.5e-6, 1.6e-6, 5),
                       steps_per_wavelength=12, device="cpu")

print(sorted(k for k in sp if "@" in k))
print(np.round(np.abs(sp["o2@0,o1@0"]) ** 2, 3))     # power transmission o1 -> o2
print(round(sp["mode_neff"]["o1"].real, 3))          # effective index of the port mode
```

```text
['o1@0,o1@0', 'o1@0,o2@0', 'o2@0,o1@0', 'o2@0,o2@0']
[0.999 1.    1.    1.    0.999]
2.422
```

The key `"o2@0,o1@0"` reads "out of port `o2`, into port `o1`". Each entry is a
complex array over `wavelengths`. A straight waveguide transmits everything,
which is what the run shows. `steps_per_wavelength=12` is a coarse mesh chosen
so that the example is quick; the default is 20.

The same call handles any component with optical ports. Larger devices are
practical on a GPU:

```python
import gdsfactory as gf
import numpy as np
from lightsim.sparams import write_sparameters

gf.gpdk.PDK.activate()
sp = write_sparameters(gf.components.mmi1x2(), wavelengths=np.linspace(1.5e-6, 1.6e-6, 21),
                       device="cuda:0")
print(np.round(np.abs(sp["o2@0,o1@0"][10]) ** 2, 3)) # one output arm at 1.55 um
```

```text
0.494
```

The splitter sends 49.4 % of the power into this arm. The call runs three
simulations, one per port, of 14.6 million cells each; on one RTX PRO 6000
they take 42 seconds in total (13 to 14 seconds per port, about 15 000 time
steps each). Two options shorten this:

- `input_ports=["o1"]` excites only the ports named. The line above needs
  only light entering through `o1`: 14 seconds.
- `port_symmetry="auto"` detects that the splitter is mirror symmetric, runs
  `o1` and `o2`, and copies the column of `o3` from that of `o2`: the full
  matrix in 28 seconds, equal to the one from three runs within 5e-4.

[Modes and S-parameters](modes-sparams.md) explains the options, and
[gdsfactory](gdsfactory.md) explains what is read from the layout and the PDK.

## A simulation built by hand

The second example measures the transmission of a glass slab and compares it
with the textbook formula. It shows every step of a simulation: the domain and
mesh, a structure, a source, a monitor, the run and the result.

All quantities are in SI units: metres, seconds and hertz. `um` and `nm` are
the constants `1e-6` and `1e-9`.

```python
import numpy as np
import lightsim as ls
from lightsim.spec import um

lam0 = 1.0 * um                      # centre wavelength
f0 = ls.C0 / lam0                    # centre frequency (C0 is the speed of light)
dx = lam0 / 40                       # cell size
n_slab, d = 2.0, 0.5 * um            # slab index and thickness
freqs = np.linspace(0.8, 1.2, 9) * f0


def column(with_slab):
    # A column along x, two cells wide. Periodic walls in y and z make it an
    # infinite plane; x keeps the default absorbing boundary (PML).
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx],
                        mesh=ls.UniformMesh(dx), run_time=200e-15, shutoff=1e-6)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if with_slab:
        sim.add_structure(ls.Box([2 * um, -1, -1], [2 * um + d, 1, 1]),
                          ls.Medium("glass", n=n_slab))
    # A plane wave launched at x = 1 um towards +x, polarised along y,
    # with a short pulse that covers the band of interest.
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
    # The power through the plane x = 3.5 um at each frequency.
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.5 * um, [3.5 * um, -1, -1], [3.5 * um, 1, 1], freqs))
    return sim


reference = column(False).run(device="cpu")          # no slab: the incident power
result = column(True).run(device="cpu")
T = np.asarray(result.flux("T")) / np.asarray(reference.flux("T"))
print(result)
print(np.round(T, 3))
```

```text
<lightsim.Results 736 cells x 1000 steps on cpu (10 threads): 4 Mcells/s>
[0.669 0.74  0.847 0.956 1.    0.942 0.831 0.729 0.667]
```

The structure box extends from -1 m to 1 m in y and z. Shapes may be larger
than the domain; only the part inside is used.

A monitor records the response to the pulse, so the flux depends on the pulse
spectrum. Dividing by a run without the slab removes it. The ratio is the
transmission, which for a slab of index $n$ and thickness $d$ in vacuum is

$$
T(\omega) = \left| \frac{(1 - r^2)\, e^{i\phi}}{1 - r^2 e^{2 i \phi}} \right|^2,
\qquad r = \frac{1 - n}{1 + n}, \qquad \phi = \frac{\omega}{c}\, n\, d .
$$

```python
r = (1 - n_slab) / (1 + n_slab)
phi = 2 * np.pi * freqs / ls.C0 * n_slab * d
T_exact = np.abs((1 - r**2) * np.exp(1j * phi) / (1 - r**2 * np.exp(2j * phi))) ** 2
print(np.round(T_exact, 3))
print(round(float(np.max(np.abs(T - T_exact))), 4))
```

```text
[0.663 0.731 0.837 0.949 1.    0.949 0.837 0.731 0.663]
0.0098
```

The largest difference is 0.01. It comes from the finite cell size: light
travels slightly slower on the grid than in the continuum, which shifts the
interference fringes. The error falls with the square of the cell size; with
`dx = lam0 / 80` the same script gives 0.0024.

The run ended after 1000 of the 4197 steps that `run_time` allows, because
`shutoff=1e-6` stops a simulation once the field energy has decayed to that
fraction of its peak. [Concepts](concepts.md) describes the grid, the units,
how a run ends and what comes back.

## Where to go next

| To do this | Read |
|---|---|
| Understand the grid, units and results | [Concepts](concepts.md) |
| Set up the domain, mesh and structures | [Simulation setup](simulation.md) |
| Use dispersive, anisotropic or nonlinear media | [Materials](materials.md) |
| Choose a source | [Sources](sources.md) |
| Choose boundary conditions or use symmetry | [Boundaries](boundaries.md) |
| Record fields, power, modes or resonances | [Monitors](monitors.md) |
| Solve modes and compute S-parameters | [Modes and S-parameters](modes-sparams.md) |
| Simulate layouts and understand what is read from a PDK | [gdsfactory](gdsfactory.md) |
| Optimise a device with gradients | [Inverse design](inverse-design.md) |
| Run on one or several GPUs | [GPUs and scaling](gpu.md) |
| Run simulations without Python | [Command line](cli.md) |
