# Python API

Every public function, class and constant of the `lightsim` Python package, with its signature and description. This page is generated from the package by `scripts/gen_api.py` (package version 0.1.0). For explanations and worked examples see the guide pages.

Builders such as `Medium`, `Box` or `PlaneWave` return plain dictionaries in the [JSON format](json-spec.md). Lengths are in metres, times in seconds and frequencies in hertz.

## `lightsim`

`import lightsim as ls` gives access to everything in `lightsim.spec`, and to
`solve_modes`, `gaussian_mode` and `Mode` from `lightsim.modes`,
`run_symmetric` from `lightsim.symmetry`, and the two items below. The other
modules are imported explicitly, for example `from lightsim import materials`.

### `lightsim.devices`

```text
devices()
```

List of the devices this build can use: `"cpu"`, then one entry per GPU,
`"cuda:0 <model name>"`, `"cuda:1 <model name>"`, ...

### `lightsim.__version__`

The package version, a string.

## `lightsim.spec`

Builders for media, shapes, sources, monitors, boundaries and meshes, the `Simulation` class and the `Results` class. Everything here is also available as `lightsim.<name>`.

### `lightsim.spec.Medium`

```text
Medium(
    name,
    *,
    n=None,
    eps=None,
    mu=1.0,
    sigma=0.0,
    sigma_m=0.0,
    poles=(),
    pole_axes=(True, True, True),
    pec=False,
    chi2=0.0,
    chi3=0.0,
    eps_off=(0.0, 0.0, 0.0),
    sigma_off=(0.0, 0.0, 0.0),
    rotation=None,
    oriented=(),
    raman=None,
    gyro=(),
    director=None,
    gain=None,
    plugin=None,
    gain2=None,
    eps_complex=None,
)
```

A (possibly anisotropic, dispersive, conductive, nonlinear) medium.

`sigma` is the electric conductivity (S/m), `sigma_m` the magnetic conductivity (Ohm/m, the magnetic current sigma_m H in Faraday's law); a medium with `sigma_m / (mu0 mu) = sigma / (eps0 eps)` has the impedance of the lossless medium at every frequency.

`chi2` (m/V) and `chi3` (m^2/V^2) are instantaneous per-axis susceptibilities: P_i/eps0 = chi_lin E_i + chi2 E_i^2 + chi3 E_i^3 (Lumerical's chi2/chi3 material convention).

`eps_off` = (eps_xy, eps_xz, eps_yz) and `sigma_off` add off-diagonal terms to symmetric permittivity / conductivity tensors. `rotation` (3x3, columns = crystal axes in lab coordinates) rotates the diagonal `eps`/`n` and `sigma` into the lab frame (tilted uniaxial crystals, lossy or not).

`oriented` = [(pole, u), ...] adds dispersive poles acting along unit vectors `u` (chi(w) u u^T), e.g. `[(Lorentz(1.0, w0, g), (1, 1, 0))]` for a dispersive crystal whose optic axis is rotated in the xy plane.

`raman` = `Raman(fraction, omega, delta)` makes a fraction of `chi3` a delayed (Raman) response: P_i/eps0 = chi3 E_i [(1 - f) E_i^2 + f (g_R \* E_i^2)].

`gyro` = [(pole, bias), ...] adds gyrotropic (magneto-optic) poles: a `Lorentz` or `Drude` pole in a static bias, `bias` being the gyration vector b in rad/s (for free electrons the cyclotron frequency e B / m along B). The polarization obeys P'' + gamma P' + omega0^2 P - b x P' = s E, which gives a permittivity tensor with antisymmetric off-diagonal terms (eps_xy = -eps_yx = i g for a bias along z); see `lightsim.materials.gyrotropic_permittivity`. Needs a lossless diagonal `eps` and no other `poles`.

`director` = `DirectorField(...)` makes the medium uniaxial with an optic axis that varies from cell to cell (a liquid crystal); without `n` / `eps` the medium's `eps` (read only by the automatic mesher) is set to the larger of eps_o and eps_e.

`gain` = `FourLevel(...)` adds a four-level atomic gain medium with pumping (rate equations coupled to a Lorentz polarization of the lasing transition); `eps` is then the background permittivity.

`plugin` = `lightsim.plugins.use(name, params)` runs a user-defined material model registered under `name` (see `lightsim.plugins`) on top of the background `eps` and `sigma`.

`gain2` = `FourLevelTwoElectron(...)` adds Chang and Taflove's four-level two-electron gain medium (Lumerical's "4-level 2-electron" model), pumped optically by a field at its pump transition.

`eps_complex` = a complex 3x3 relative permittivity tensor given directly, Hermitian and positive definite (e.g. `[[e, 0, 0], [0, e, 1j * g], [0, -1j * g, e]]` for a magneto-optic medium biased along x). The medium is non-dispersive and lossless; the run then uses complex fields, and the tensor holds at every frequency (exact only as a non-dispersive model). It cannot be combined with `eps`/`n`, poles, `sigma` or other tensors.

### `lightsim.spec.FourLevel`

```text
FourLevel(
    density,
    omega,
    gamma,
    tau32,
    tau21,
    tau10,
    pump_rate=0.0,
    coupling=None,
    initial=(1.0, 0.0, 0.0, 0.0),
)
```

A four-level atomic gain medium with pumping, for `Medium(gain=...)`.

Rate equations coupled to a Lorentz polarization of the 2 -> 1 transition (fields exp(-i w t), N in m^-3, P in C/m^2)::

```text
P'' + gamma P' + omega^2 P = xi (N1 - N2) E
dN3/dt = W_p N0 - N3/tau32
dN2/dt = N3/tau32 - N2/tau21 + E.P' / (hbar omega)
dN1/dt = N2/tau21 - N1/tau10 - E.P' / (hbar omega)
dN0/dt = N1/tau10 - W_p N0
```

`density` is the total atom density (m^-3), `omega` and `gamma` the transition's angular frequency and width (rad/s), the `tau` the lifetimes (s), `pump_rate` W_p (1/s) the pump from level 0 to 3, and `coupling` xi (C^2/kg, default the radiative value 6 pi eps0 c^3 / (omega^2 tau21)). `initial` gives the starting populations as fractions of `density`. At fixed populations the transition adds chi(w) = (xi (N1 - N2) / eps0) / (omega^2 - w^2 - i gamma w): gain where N2 > N1. Each E-component position carries its own atoms, coupled only to its own field component. The rate equations are explicit: the time step must stay below half of every lifetime and of 1/W_p. `Results.gain_levels()` returns the atoms in each level at the end.

### `lightsim.spec.FourLevelTwoElectron`

```text
FourLevelTwoElectron(
    density,
    omega_a,
    omega_b,
    gamma_a,
    gamma_b,
    t30,
    t32,
    t21,
    t10,
    initial=None,
    kappa_a=None,
    kappa_b=None,
)
```

The four-level two-electron gain medium of Chang and Taflove (Opt. Express 12, 3827 (2004)), for `Medium(gain2=...)`; the parameters of Lumerical's "4-level 2-electron" material (N density, omega a / b, gamma a / b, t30, t32, t21, t10, initial populations)::

```text
Pa'' + gamma_a Pa' + omega_a^2 Pa = kappa_a (N1 - N2) E
Pb'' + gamma_b Pb' + omega_b^2 Pb = kappa_b (N0 - N3) E
dN3/dt =  E.Pb'/(hbar omega_b) - N3 (1 - N2/Nd)/t32 - N3 (1 - N0/Nd)/t30
dN2/dt =  N3 (1 - N2/Nd)/t32 + E.Pa'/(hbar omega_a) - N2 (1 - N1/Nd)/t21
dN1/dt =  N2 (1 - N1/Nd)/t21 - E.Pa'/(hbar omega_a) - N1 (1 - N0/Nd)/t10
dN0/dt =  N3 (1 - N0/Nd)/t30 + N1 (1 - N0/Nd)/t10 - E.Pb'/(hbar omega_b)
```

`density` Nd in m^-3 (two electrons per atom: the populations sum to 2 Nd, each at most Nd), angular frequencies and widths in rad/s, lifetimes in s; `initial` = (N0, N1, N2, N3) in m^-3, default (Nd, Nd, 0, 0); `kappa_a`, `kappa_b` (C^2/kg) default to 6 pi eps0 c^3 / (omega^2 t) with t21 and t30. Pump it with a source at `omega_b`; `Results.gain_levels()` returns the electrons in each level at the end. The rate equations are explicit: the time step must stay below half of every lifetime.

### `lightsim.spec.DirectorField`

```text
DirectorField(
    min,
    max,
    director,
    *,
    eps_o=None,
    eps_e=None,
    n_o=None,
    n_e=None,
    axes=None,
)
```

A director field for `Medium(director=...)`: a uniaxial medium (a liquid crystal) whose optic axis varies from cell to cell.

`director` is a numpy array of director vectors, its last axis of length 3 (x, y, z components; the length and the sign do not matter). The leading axes are the sample grid, whose nodes span `[min, max]` inclusive, as in `IndexGrid`: 3-D grids are indexed `[ix, iy, iz]`; a 1-D or 2-D grid varies along x or (x, y) unless `axes` says otherwise (`axes=(2,)` for a director that turns along z); a single vector is a uniform director. Give the ordinary and extraordinary permittivities (`eps_o`, `eps_e`) or indices (`n_o`, `n_e`). At each field position the solver interpolates the director tensor d d^T trilinearly, normalises it to unit trace and uses eps = eps_o I + (eps_e - eps_o) d d^T; outside the box the nearest face value is used.

### `lightsim.spec.Raman`

```text
Raman(fraction, omega, delta)
```

The delayed (Raman) part of a chi3 response, for `Medium(raman=...)`.

A fraction `fraction` (f, 0 to 1) of `chi3` acts through the damped-oscillator response g_R(t) = (omega^2 / nu) exp(-delta t) sin(nu t), nu = sqrt(omega^2 - delta^2), whose integral is 1:

```text
P_i/eps0 = chi3_i E_i [(1 - f) E_i^2 + f (g_R * E_i^2)(t)],
g_R(w) = omega^2 / (omega^2 - w^2 - 2j delta w).
```

`omega` is the phonon (resonance) angular frequency and `delta` the damping rate, in rad/s (0 <= delta < omega). The common fibre form g_R = (tau1^2 + tau2^2) / (tau1 tau2^2) exp(-t/tau2) sin(t/tau1) is omega = sqrt(1/tau1^2 + 1/tau2^2), delta = 1/tau2; for silica, f = 0.18, tau1 = 12.2 fs and tau2 = 32 fs give omega = 8.77e13 rad/s (14.0 THz) and delta = 3.13e13 rad/s.

### `lightsim.spec.Lorentz`

```text
Lorentz(delta_eps, omega0, gamma)
```

A Lorentz pole for `Medium(poles=...)`.

Adds the susceptibility `delta_eps * omega0**2 / (omega0**2 - w**2 - 1j * gamma * w)` to the permittivity.

**Parameters**

- `delta_eps` (float): Strength: the contribution of the pole to the static permittivity.
- `omega0` (float): Resonance angular frequency (rad/s).
- `gamma` (float): Damping rate (rad/s); the full width of the absorption line.

Returns the pole dict.

### `lightsim.spec.Drude`

```text
Drude(omega_p, gamma)
```

A Drude pole for `Medium(poles=...)`, the response of free carriers.

Adds the susceptibility `-omega_p**2 / (w**2 + 1j * gamma * w)`.

**Parameters**

- `omega_p` (float): Plasma angular frequency (rad/s).
- `gamma` (float): Collision rate (rad/s).

Returns the pole dict.

### `lightsim.spec.Debye`

```text
Debye(delta_eps, tau)
```

A Debye pole for `Medium(poles=...)`, a relaxation.

Adds the susceptibility `delta_eps / (1 - 1j * w * tau)`.

**Parameters**

- `delta_eps` (float): Strength: the contribution of the pole to the static permittivity.
- `tau` (float): Relaxation time (s). It should be resolved by the time step; a pole with `tau` far below the time step makes the run refuse to start.

Returns the pole dict.

### `lightsim.spec.Sheet`

```text
Sheet(axis, position, conductivity, *, min=None, max=None, polygon=None, name='')
```

A zero-thickness conducting sheet, for `Simulation.add_sheet`.

The plane `axis = position` (`axis` "x", "y", "z" or 0, 1, 2) carries the surface current `J_s = sigma_s(w) E_t` (tangential E only), with `conductivity` a surface conductivity in siemens from `lightsim.sheets.sheet_conductivity`, `lightsim.sheets.graphene` (without a thickness) or `lightsim.sheets.SheetConductivity`. Without `min` / `max` / `polygon` the sheet fills the plane; `min` / `max` (box corners, the components along `axis` ignored) make it a rectangle, `polygon` (vertices as `(u, v)` pairs in the two other axes in increasing order: y, z for a sheet normal to x; x, z for y; x, y for z) a polygon. The sheet need not lie on a grid node; see `docs/modules/sheets.md` for the discretisation and its accuracy. In a 2-D simulation a sheet is a line, normal to one of the in-plane axes.

### `lightsim.spec.Box`

```text
Box(min, max)
```

Axis-aligned box.

**Parameters**

- `min`, `max` ((x, y, z)): Opposite corners (m). The box may extend beyond the simulation region; only the part inside is used.

Returns the shape dict for `Simulation.add_structure`.

### `lightsim.spec.Sphere`

```text
Sphere(center, radius)
```

Sphere.

**Parameters**

- `center` ((x, y, z)): Centre (m).
- `radius` (float): Radius (m).

Returns the shape dict for `Simulation.add_structure`.

### `lightsim.spec.Cylinder`

```text
Cylinder(center, radius, length=None, axis=None)
```

Circular cylinder.

**Parameters**

- `center` ((x, y, z) or (a, b)): Centre of the cylinder (m), half-way along its length. In a 2-D simulation two in-plane coordinates are enough.
- `radius` (float): Radius (m).
- `length` (float, optional): Length (m) along `axis`, centred on `center`. Required in 3-D; in a 2-D simulation it defaults to more than the slab.
- `axis` (int or str, optional): Direction of the cylinder axis: 0, 1, 2 or "x", "y", "z". Default z in 3-D and the invariant axis in 2-D, where `Cylinder([a, b], radius)` is a disc.

Returns the shape dict for `Simulation.add_structure`.

### `lightsim.spec.PolySlab`

```text
PolySlab(vertices, slab_min=None, slab_max=None, axis=None)
```

Polygon extruded along an axis.

**Parameters**

- `vertices` (sequence of (a, b)): Corners of the polygon (m) in the plane normal to `axis`, as pairs of coordinates on the two other axes in increasing axis order: (x, y) for `axis=2`, (x, z) for `axis=1`, (y, z) for `axis=0`. The polygon is closed automatically, may be concave and is filled by the even-odd rule. It cannot have holes: cut such a shape into pieces.
- `slab_min`, `slab_max` (float, optional): Extent (m) of the extrusion along `axis`. Required in 3-D; in a 2-D simulation the extrusion covers the slab by default.
- `axis` (int or str, optional): Extrusion axis: 0, 1, 2 or "x", "y", "z". Default z in 3-D and the invariant axis in 2-D, where `PolySlab(vertices)` is the polygon itself.

Returns the shape dict for `Simulation.add_structure`. Consecutive polygons of one medium and one extent are painted as their union.

### `lightsim.spec.IndexGrid`

```text
IndexGrid(min, max, eps=None, *, n=None, axes=None, eps_axes=None, n_axes=None)
```

Imported / graded permittivity on a regular grid (Lumerical "import (n,k)").

`eps` (relative permittivity) or `n` (refractive index) is a real numpy array of samples whose nodes span the box `[min, max]` inclusive along every axis it varies on; inside the box the solver uses the trilinear interpolation of the samples at each Yee E-component position. 3-D arrays are indexed `[ix, iy, iz]`. 1-D and 2-D arrays are padded with length-1 (= uniform) axes: by default a 1-D array varies along x and a 2-D array along (x, y); pass `axes` to choose, e.g. `axes=(1, 2)` for a (y, z) cross-section map of a waveguide running along x. A scalar gives a constant box.

`eps_axes=(eps_x, eps_y, eps_z)` / `n_axes` (three arrays of the same shape) give a diagonal-anisotropic grid.

The grid *replaces* the permittivity of the structure's medium inside the box: the medium passed to `Simulation.add_structure` only supplies its name, conductivity (loss), chi2/chi3 and permeability, and its `eps` is what the automatic mesher uses to pick the cell size, so set it to the largest permittivity of the grid. Dispersive, PEC and off-diagonal media are not allowed. The box faces are staircased (no sub-pixel smoothing, the grid is already a sampled field); to embed a grid in another material, add that structure first and let the grid overlap it.

### `lightsim.spec.Structure`

```text
Structure(shape, medium, name='')
```

A shape filled with a medium, as stored in `Simulation.structures`.

**Parameters**

- `shape` (dict): A shape from `Box`, `Sphere`, `Cylinder`, `PolySlab` or `IndexGrid`.
- `medium` (str): Name of the medium, defined in the simulation or built in.
- `name` (str): A label, used in error messages.

Returns the structure dict. Normally created by `Simulation.add_structure`.

### `lightsim.spec.GaussianPulse`

```text
GaussianPulse(freq0, fwidth, offset=5.0, phase=0.0, amplitude=1.0)
```

Gaussian pulse, the usual signal for a spectrum in one run.

`amplitude * cos(2 pi freq0 (t - t0) + phase) * exp(-(t - t0)**2 / (2 tau**2))` with `tau = 1 / (2 pi fwidth)` and `t0 = offset * tau`. Its spectrum is a Gaussian of standard deviation `fwidth` around `freq0`.

**Parameters**

- `freq0` (float): Centre frequency (Hz).
- `fwidth` (float): Spectral width (Hz), the standard deviation of the spectrum.
- `offset` (float): Peak time in units of `tau`; 5 starts the pulse from a negligible value.
- `phase` (float): Carrier phase (rad).
- `amplitude` (float): Peak value, a factor on the source amplitude.

Returns the signal dict for a source.

### `lightsim.spec.ContinuousWave`

```text
ContinuousWave(freq0, fwidth, offset=5.0, phase=0.0, amplitude=1.0)
```

Continuous wave at one frequency, switched on smoothly.

`amplitude * cos(2 pi freq0 t + phase)`, multiplied before `t0 = offset * tau` by a Gaussian ramp of width `tau = 1 / (2 pi fwidth)`.

**Parameters**

- `freq0` (float): Frequency (Hz).
- `fwidth` (float): Sets the duration of the turn-on, `tau = 1 / (2 pi fwidth)` (Hz).
- `offset` (float): End of the ramp in units of `tau`.
- `phase` (float): Phase (rad).
- `amplitude` (float): Amplitude, a factor on the source amplitude.

Returns the signal dict for a source. The signal never ends, so `shutoff` has no effect and frequency-domain monitors sample every step.

### `lightsim.spec.PointDipole`

```text
PointDipole(position, polarization, signal, amplitude=1.0, name='')
```

Point dipole at a single field sample.

**Parameters**

- `position` ((x, y, z)): Position (m); the dipole is placed on the grid position of the component `polarization` nearest to it.
- `polarization` (str): `"Ex"`, `"Ey"` or `"Ez"`: an electric current element of moment `amplitude * signal(t)` in A m. `"Hx"`, `"Hy"` or `"Hz"`: a magnetic current element in V m.
- `signal` (dict): Time signal.
- `amplitude` (float): Dipole moment amplitude (A m, or V m for a magnetic dipole).
- `name` (str): Name, for `Results.source_spectrum`.

Returns the source dict. An electric dipole of moment `p` in a medium of index `n` radiates `n eta0 k0**2 abs(p)**2 / (12 pi)` watts, with `k0 = 2 pi f / c`.

### `lightsim.spec.PlaneWave`

```text
PlaneWave(
    axis,
    position,
    direction,
    polarization,
    signal,
    amplitude=1.0,
    angle=0.0,
    min=None,
    max=None,
    name='',
    fixed_angle=False,
)
```

One-sided plane-wave source on a plane.

The wave is launched towards `direction` only; behind the plane there is no incident field.

**Parameters**

- `axis`, `position` (int, float): The source plane is normal to `axis` (0, 1, 2) at this coordinate (m), moved to the nearest grid node.
- `direction` ("+" or "-"): Propagation direction along `axis`.
- `polarization` (str): A field component in the plane. An electric component (`"Ey"` for `axis=0`) gives E along that axis; with `angle` set the wave is s-polarised. A magnetic component (`"Hz"` for `axis=0`) gives H along that axis and E along the other in-plane axis; with `angle` set the wave is p-polarised.
- `signal` (dict): Time signal.
- `amplitude` (float): Peak electric field (V/m), times the signal.
- `angle` (float): Tilt of the wavevector (rad) towards the in-plane axis perpendicular to the polarisation axis: s-polarisation for an E component, p-polarisation for an H component (then E has the tangential part `amplitude * cos(angle)` and a normal part `amplitude * sin(angle)`). It needs Bloch boundaries along the tilt axis with `k = n k0 sin(angle)`, and a `GaussianPulse` or `ContinuousWave` signal. The angle is exact at the centre frequency of the signal.
- `min`, `max` ((x, y, z), optional): Corners limiting the extent of the plane (m); the whole plane by default.
- `name` (str): Name, for `Results.source_spectrum`.
- `fixed_angle` (bool): Hold `angle` at every frequency of the pulse instead of only at its centre (broadband oblique incidence in one run). The boundaries along the tilt axis must then be `PERIODIC` (the run imposes the phase of each frequency itself), the other transverse axis periodic, and the source must be the only one; any real signal works. The run solves for the transformed fields `E(r, t + n sin(angle) u / c)` and returns physical frequency-domain data. See "Broadband oblique incidence at a fixed angle" in the sources guide for what it supports and costs.

Returns the source dict. The index of the medium is taken at the centre of the plane; in a medium of index `n` the wave carries `n abs(amplitude * S)**2 cos(angle) / (2 eta0)` watts per square metre of the plane.

### `lightsim.spec.GaussianBeam`

```text
GaussianBeam(
    axis,
    position,
    direction,
    polarization,
    center,
    waist,
    signal,
    amplitude=1.0,
    focus_distance=0.0,
    name='',
    theta=0.0,
    phi=0.0,
)
```

One-sided Gaussian-beam source, at any angle to its plane.

The beam is injected on the plane normal to `axis` at `position` and travels away from it towards `+axis` or `-axis`.

**Parameters**

- `axis`, `position` (int, float): The source plane is normal to `axis` (0, 1, 2) at this coordinate (m), moved to the nearest grid node.
- `direction` ("+", "-" or a vector): Side of the plane the beam is launched to. A vector `(dx, dy, dz)` gives the direction of the beam axis itself; its component along `axis` sets the side, and `theta` and `phi` are computed from it (they must then be left at 0).
- `polarization` (str): An electric component in the plane, e.g. `"Ey"` for `axis=0`. The electric field of the beam points along the projection of this axis onto the plane normal to the beam axis: exactly along it when the tilt is perpendicular to it (s polarisation), tilted with the beam otherwise (p polarisation).
- `center` ((x, y, z)): A point of the beam axis; its two in-plane coordinates are where the axis crosses the source plane (m).
- `waist` (float): Radius (m) at which the field falls to 1/e at the focus.
- `signal` (dict): Time signal; a tilted or defocused beam needs `GaussianPulse` or `ContinuousWave`. The pattern is that of the centre frequency.
- `amplitude` (float): Peak electric field at the focus (V/m), times the signal.
- `focus_distance` (float): Distance (m) from `center` to the focus along the beam axis; negative for a beam that is already diverging at the plane. The waist plane is normal to the beam axis.
- `theta`, `phi` (float): Polar angle (rad) between the beam axis and the plane normal, below pi/2, and azimuth of the tilt, measured from the first in-plane axis towards the second. The in-plane axes are (y, z) for `axis=0`, (x, z) for `axis=1` and (x, y) for `axis=2`.

The pattern is the paraxial Gaussian beam: accurate for waists of about a wavelength or more. It spans the whole plane, so the region must be several waists wide where the beam crosses it. In a simulation that is periodic along an in-plane axis with a period below twice the waist (a two-dimensional simulation) the beam is uniform along that axis, a sheet beam, and must travel in the other plane. The beam carries the power `n |amplitude S|^2 pi waist^2 / (4 eta0)`, or `n |amplitude S|^2 waist sqrt(pi/2) / (2 eta0)` per unit length of a sheet beam, whatever its angle.

### `lightsim.spec.ModeSource`

```text
ModeSource(axis, position, direction, min, max, profile, signal, amplitude=1.0, name='')
```

One-sided waveguide-mode source on a plane window.

**Parameters**

- `axis`, `position` (int, float): The source plane is normal to `axis` at this coordinate (m), moved to the nearest grid node.
- `direction` ("+" or "-"): Direction in which the mode is launched.
- `min`, `max` ((x, y, z)): Corners of the mode window (m): those given to the mode solver.
- `profile` (dict): Field pattern: `Mode.profile` from `solve_modes` or `eim.solve_port_mode`, or a profile from `gaussian_mode`.
- `signal` (dict): Time signal; a complex profile (a lossy mode, a tilted beam) needs `GaussianPulse` or `ContinuousWave`.
- `amplitude` (float): Factor on the profile. A profile from the mode solver carries unit power, so the launched power spectrum is `abs(amplitude * S)**2` watts.
- `name` (str): Name, for `Results.source_spectrum`.

Returns the source dict. The part of the window inside an absorbing layer is not injected.

### `lightsim.spec.TFSF`

```text
TFSF(min, max, axis, direction, polarization, signal, amplitude=1.0, name='')
```

Total-field/scattered-field box with a plane wave inside.

The incident wave exists inside the box only; outside it only the field scattered by what the box contains remains.

**Parameters**

- `min`, `max` ((x, y, z)): Corners of the box (m), snapped to grid nodes.
- `axis` (int): Propagation axis of the incident wave (normal incidence on the faces).
- `direction` ("+" or "-"): Propagation direction along `axis`.
- `polarization` (str): Electric component of the wave, perpendicular to `axis`.
- `signal` (dict): Time signal.
- `amplitude` (float): Peak electric field (V/m), times the signal.
- `name` (str): Name, for `Results.source_spectrum`.

Returns the source dict. The medium around the box must be uniform (it may be dispersive). A box that spans a periodic axis completely has no faces normal to that axis, which gives the incident wave of a 2-D problem.

### `lightsim.spec.FieldTimeMonitor`

```text
FieldTimeMonitor(name, min, max, components=('Ex', 'Ey', 'Ez'), interval=1, start=0.0)
```

Fields in a box at regular time intervals.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.time_series(name, component)`.
- `min`, `max` ((x, y, z)): Corners of the box (m). A zero extent records a plane, a line or a point.
- `components` (sequence of str): Field components to record, from `"Ex"` ... `"Hz"`.
- `interval` (int): Record every `interval` steps.
- `start` (float): Time (s) of the first sample.

Returns the monitor dict. All samples stay in memory until the end of the run: 4 bytes per cell, component and sample.

### `lightsim.spec.ProbeMonitor`

```text
ProbeMonitor(name, position, components=('Ez',))
```

Fields at one point at every time step.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.time_series(name, component)`.
- `position` ((x, y, z)): Position (m); each component is taken at its nearest grid position.
- `components` (sequence of str): Field components to record.

Returns the monitor dict.

### `lightsim.spec.DFTMonitor`

```text
DFTMonitor(
    name,
    min,
    max,
    freqs,
    components=('Ex', 'Ey', 'Ez'),
    stride=None,
    apodization=None,
)
```

Frequency-domain fields in a box, a plane, a line or a point.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.field(name, component)`.
- `min`, `max` ((x, y, z)): Corners of the box (m). Along an axis of zero extent the nearest layer of each component is taken.
- `freqs` (sequence of float): Frequencies (Hz).
- `components` (sequence of str): Field components to record, from `"Ex"` ... `"Hz"`.
- `stride` (int, optional): Accumulate the Fourier sums every `stride` steps. `None` takes `Simulation.dft_stride`, which by default is chosen from the bandwidth of the sources so that the result equals that of every step.
- `apodization` (dict, optional): Time window of the sums: keys `start`, `end` (s) and `width` (s) of a Gaussian taper at each edge, all optional.

Returns the monitor dict. The monitor stores two 4-byte numbers per sample, component and frequency.

### `lightsim.spec.FluxMonitor`

```text
FluxMonitor(name, axis, position, min, max, freqs, apodization=None)
```

Power through a plane, per frequency.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.flux(name)`.
- `axis`, `position` (int, float): The plane is normal to `axis` at this coordinate (m), moved to the nearest grid node.
- `min`, `max` ((x, y, z)): Corners limiting the plane (m); they may lie outside the region.
- `freqs` (sequence of float): Frequencies (Hz).
- `apodization` (dict, optional): Time window of the Fourier sums, as for `DFTMonitor`.

Returns the monitor dict. The power is that of the time-averaged Poynting vector of the recorded spectra, in watts times the squared source spectrum, positive towards `+axis`.

### `lightsim.spec.FluxBoxMonitor`

```text
FluxBoxMonitor(name, min, max, freqs)
```

Net power leaving a closed box, per frequency.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.flux(name)`.
- `min`, `max` ((x, y, z)): Corners of the box (m), snapped to grid nodes.
- `freqs` (sequence of float): Frequencies (Hz).

Returns the monitor dict. The result is positive for a source inside the box, negative for absorption inside it; around a `TFSF` box it is the scattered power. A face on an electric or magnetic wall is skipped.

### `lightsim.spec.ModeMonitor`

```text
ModeMonitor(name, axis, position, min, max, freqs, modes)
```

Amplitudes of waveguide modes on a plane window.

**Parameters**

- `name` (str): Monitor name; read the result with `Results.mode_amplitudes(name, mode)`.
- `axis`, `position` (int, float): The plane is normal to `axis` at this coordinate (m), moved to the nearest grid node.
- `min`, `max` ((x, y, z)): Corners of the mode window (m): those given to the mode solver.
- `freqs` (sequence of float): Frequencies (Hz).
- `modes` (sequence of dict): Mode profiles (`Mode.profile`, or from `gaussian_mode`) solved on this window. If there are as many profiles as frequencies, and more than one, they are taken as one mode with one profile per frequency.

Returns the monitor dict. For unit-power profiles `abs(a)**2` is the power in the mode. Monitors on one plane with the same frequencies share the fields they record, so many of them cost what one costs.

### `lightsim.spec.FarFieldMonitor`

```text
FarFieldMonitor(name, min, max, freqs, theta, phi, n_medium=1.0, apodization=None)
```

Far field radiated by everything inside a box.

The fields on the six faces of the box `min`..`max` (m, snapped to grid nodes) are recorded at the frequencies `freqs` (Hz), turned into equivalent surface currents and radiated into a homogeneous medium of refractive index `n_medium`. Read the result with `Results.far_field(name)`.

**Parameters**

- `theta`, `phi` (sequences of float): Directions (radians): `theta` is the polar angle from the +z axis, `phi` the azimuth from the +x axis in the xy plane. The pattern is returned on the grid `theta` x `phi`.
- `n_medium` (float): Index of the medium outside the box, which must be homogeneous: a substrate or other layers outside the box are not accounted for.
- `apodization` (dict, optional): Time window of the Fourier sums, as for `DFTMonitor`.

The box must be closed, lie inside the simulation region and enclose all sources and scatterers whose radiation is wanted.

### `lightsim.spec.EnergyMonitor`

```text
EnergyMonitor(name, min, max, interval=1)
```

Electromagnetic energy stored in a box, versus time.

Every `interval` steps the monitor evaluates

```text
U = eps0/2 * sum( eps |E|^2 + mu |eta0 H|^2 ) dV      (joules)
```

over the cells of the box `min`..`max` (m), with the permittivity of each cell as the engine uses it. Cells cut by the box count with the part inside it. Read the result with `Results.energy(name)`.

The electric field is sampled at the end of a step and the magnetic field half a step earlier, so the sum ripples at twice the optical frequency by a fraction of about `(omega dt / 2)**2`. For a dispersive medium `eps` is its high-frequency value: the energy stored in the polarisation of the poles is not included. The six field components of the box are kept in memory for every sample until the end of the run (24 bytes per cell and sample), so keep the box small or the interval long.

### `lightsim.spec.PML`

```text
PML(
    layers=12,
    order=3.0,
    sigma_scale=1.0,
    kappa_max=2.0,
    alpha_max=0.0,
    inside=False,
    stabilize=True,
)
```

CPML boundary. `inside=True` puts the layers on the outermost cells of the region (FDTDX convention) instead of adding them outside.

`stabilize` (on by default) changes, inside the layers only and when the region holds a structure of another medium than the background, the lossless poles of transparent media that lie outside the band of the sources: the far-infrared pole of a fitted glass (`lightsim.materials.fitted_medium`) is damped, and a strong ultraviolet pole (silicon fitted with `eps_inf = 1`) is frozen at its value at the centre of the band. Without it a waveguide of such media running into the layer makes a long run diverge: the medium is metal-like in a band above each pole, the guide carries backward modes there, and a PML amplifies backward waves. The change reflects about 1e-7 of the power at the face of the layer for fused silica near 1.55 um, and at most 1e-4 at the ends of a wide band; poles in or near the band are left as they are. Damping the infrared pole has removed the growth in every case tried; freezing the ultraviolet pole removes it for silicon guides in a glass cladding and only delays it for some other structures (the boundaries guide has the measurements). `Absorber` is the boundary that cannot amplify.

### `lightsim.spec.Absorber`

```text
Absorber(layers=40, order=3.0, sigma_scale=1.0)
```

Adiabatic absorber: a boundary that can replace a PML where a PML is unstable. `layers` cells are added outside the region, in which an electric and a magnetic conductivity rise smoothly (as the power `order` of the depth) from zero. The two are matched to each other in every medium, at the centre of the source band, so a wave decays without a change of its impedance or of its mode profile.

An absorber is a passive medium: it cannot amplify any wave. A PML can: it amplifies backward waves, which fitted (dispersive) media with a band of negative permittivity, metals with surface plasmons, and some photonic crystal guides carry (`PML` describes the usual case and what `stabilize` does about it). Use an absorber on the faces that such a structure runs through when a run with PML diverges.

The price is thickness. An absorber reflects wherever its conductivity changes, so the change must be slow on the scale of the wavelength. At 1.55 um a layer 1 um thick (one wavelength in a glass cladding: 40 layers of 25 nm) reflects 1e-5 to 1e-4 of a guided mode, 0.5 um about 5e-4, and 1.5 um as little as a 12-layer PML (a few 1e-6): choose `layers` from the thickness. The boundaries guide gives the measured values. `sigma_scale` multiplies the peak conductivity: the default attenuates a plane wave in the background medium by 6 nepers in amplitude (1e-5 in power) on the way to the outer wall and back, and a guided mode in proportion to its group index.

Absorbers and PML can be mixed, face by face. Only media with a diagonal permittivity, conductivity and per-axis poles can run through an absorber; the fixed-angle engine and the `reverse` gradient engine do not take one, and design regions must stay outside it.

### `lightsim.spec.Bloch`

```text
Bloch(k)
```

Bloch-periodic boundary condition.

The field at the upper face of the axis is the field at the lower face times `exp(1j * k * L)`, with `L` the length of the region along the axis.

**Parameters**

- `k` (float): Wavevector component along the axis (rad/m); `n k0 sin(theta)` for a plane wave incident at `theta` in a medium of index `n`.

Returns the boundary dict, to be set on both faces of the axis with `Simulation.set_boundaries`. A run with `k != 0` uses complex fields.

### `lightsim.spec.UniformMesh`

```text
UniformMesh(dx)
```

Uniform mesh.

**Parameters**

- `dx` (float or (dx, dy, dz)): Cell size (m), one value or one per axis. It is adjusted so that a whole number of cells fits the region along each axis. In a 2-D simulation `dx` may hold the two in-plane spacings; the spacing along the invariant axis is not used.

Returns the mesh dict for `Simulation`.

### `lightsim.spec.AutoMesh`

```text
AutoMesh(
    wavelength,
    steps_per_wavelength=20.0,
    grading=1.3,
    min_step=None,
    max_step=None,
    overrides=(),
    symmetry=None,
)
```

Automatic graded mesh.

The cell size in each medium is `wavelength / (n * steps_per_wavelength)` with `n` the refractive index of the medium, and it grows away from a structure by at most the factor `grading` per cell.

**Parameters**

- `wavelength` (float): Vacuum wavelength (m) the resolution refers to; use the shortest wavelength of interest.
- `steps_per_wavelength` (float): Cells per wavelength in each medium.
- `grading` (float): Largest ratio of neighbouring cell sizes, about.
- `min_step`, `max_step` (float, optional): Lower bound (m) on the cell size chosen for structures, upper bound on the cell size of the background.
- `overrides` (sequence of (min, max, (dx, dy, dz))): Boxes with an explicit largest cell size (m) per axis.
- `symmetry` ((x0, y0, z0), optional): Mirror planes of the mesh, one coordinate (m) per axis or `None`. A plane gets a grid node and the cells on its two sides are mirror images of each other. Without it the mesh is built outwards from the centre of the region along each axis, which is already symmetric for structures that are symmetric about that centre; name a plane when the symmetry plane of the device is somewhere else.

Returns the mesh dict for `Simulation`.

### `lightsim.spec.ExplicitMesh`

```text
ExplicitMesh(x, y, z)
```

Mesh given by its node coordinates.

`x`, `y` and `z` are increasing lists of the cell edges (m) along each axis, at least two per axis. The first and the last node of each list should be the `min` and `max` of the simulation region; absorbing layers are added outside them with the size of the outermost cell. `Simulation.frozen_mesh` returns the grid of a simulation in this form.

### `lightsim.spec.Simulation`

```text
Simulation(
    min,
    max,
    mesh,
    run_time,
    boundaries=None,
    media=...,
    structures=...,
    sources=...,
    monitors=...,
    background='vacuum',
    shutoff=None,
    shutoff_interval=200,
    courant=0.99,
    subpixel=True,
    name='',
    dft_stride=None,
    dimensions=3,
    invariant_axis='z',
    polarization='both',
    thickness=1.0,
    force_3d_kernels=False,
    sheets=...,
)
```

A lightsim simulation. Build it, then call `run`.

**3-D** (default): `min` / `max` are the corners of the region.

**2-D** (`dimensions=2`, or `min` / `max` with two components): the fields do not depend on the coordinate along `invariant_axis` ("x", "y" or "z", default "z"). The grid has one cell along that axis, the boundaries there are periodic, the time step follows the 2-D Courant limit, and the solver updates three field components per polarization instead of six:

- `polarization="te"`: E in the plane, H along the invariant axis -- (Ex, Ey, Hz) for the invariant axis z;
- `polarization="tm"`: E along the invariant axis, H in the plane -- (Ez, Hx, Hy) for z;
- `polarization="both"` (default): the two sets, which are independent; only the sets that a source drives are updated.

(The naming of Lumerical's 2-D FDTD, Meep and Taflove's TE_z / TM_z. In a top view of a chip, z-invariant, "te" holds the quasi-TE waveguide modes; in a side view of a slab, e.g. y-invariant, a slab TE mode has E along y and belongs to "tm".)

The region's extent along the invariant axis is the thickness of the slab the 2-D model stands for: with two-component `min` / `max` it is `thickness` (default 1 m, centred on 0), so fluxes and dipole powers are per metre. Structures are cut by the mid-plane of the slab, whatever their extent along the invariant axis, and sources and monitors span it. Inside a 2-D simulation every point or box corner may be given by its two in-plane components.

Fields:

- `min`, `max`: corners of the simulation region (metres).
- `mesh`: a mesh from `UniformMesh` or `AutoMesh`, or an explicit-mesh dictionary `{"type": "explicit", "x": [...], "y": [...], "z": [...]}`.
- `run_time`: simulated time in seconds.
- `boundaries`: `{"x": [low, high], "y": [...], "z": [...]}`; `None` puts a `PML()` on all six faces.
- `media`, `structures`, `sources`, `monitors`: lists filled by the `add_*` methods.
- `background`: name of the background medium.
- `shutoff`: stop early when the field energy has fallen to this fraction of its peak; `None` disables it.
- `shutoff_interval`: steps between energy checks.
- `courant`: time step as a fraction of the stability limit.
- `subpixel`: sub-pixel smoothing of dielectric interfaces.
- `name`: a label.
- `dft_stride`: steps between samples of the frequency-domain monitors; `None` chooses it per monitor from the bandwidth of the sources, `1` samples every step.
- `dimensions`: 3, or 2 for a simulation whose fields do not depend on the coordinate along `invariant_axis` (set automatically when `min` / `max` have two components).
- `invariant_axis`: `"x"`, `"y"`, `"z"` or 0, 1, 2; stored as the number.
- `polarization`: `"te"`, `"tm"` or `"both"` (2-D only).
- `thickness`: extent of the region along the invariant axis when `min` / `max` have two components, in metres.
- `force_3d_kernels`: run a 2-D simulation through the general 3-D update (cross-checks).

### `lightsim.spec.Simulation.slab`

Property. 2-D simulations: (invariant axis, low face, high face) of the slab the model stands for; `None` for a 3-D simulation.

### `lightsim.spec.Simulation.add_medium`

```text
Simulation.add_medium(m)
```

Register a medium.

A medium whose name is already registered is ignored. `m` is a dict from `Medium`. Returns the simulation.

### `lightsim.spec.Simulation.add_structure`

```text
Simulation.add_structure(shape, medium, name='')
```

Add a shape filled with a medium.

`medium` is the name of a medium, or a `Medium` dict, which is registered as well. Structures are painted in the order they are added: where two overlap, the later one wins. Returns the simulation.

### `lightsim.spec.Simulation.add_source`

```text
Simulation.add_source(s)
```

Add a source (a dict from `PointDipole`, `PlaneWave`, `GaussianBeam`, `ModeSource` or `TFSF`). A source name may be used once. Returns the simulation.

### `lightsim.spec.Simulation.add_monitor`

```text
Simulation.add_monitor(m)
```

Add a monitor (a dict from one of the `...Monitor` builders). Every monitor needs its own name, whatever its type: results are looked up by name. Returns the simulation.

### `lightsim.spec.Simulation.add_sheet`

```text
Simulation.add_sheet(sheet)
```

Add a zero-thickness conducting sheet (a dict from `Sheet`). Returns the simulation.

### `lightsim.spec.Simulation.set_boundaries`

```text
Simulation.set_boundaries(x=None, y=None, z=None)
```

Set the boundary conditions of the given axes.

Each of `x`, `y`, `z` is one condition for both faces of that axis, or a pair `[low, high]`: `PML(...)`, `PERIODIC`, `Bloch(k)`, `PEC_WALL` or `PMC_WALL`. Periodic and Bloch conditions must be set on both faces. Axes left at `None` keep their conditions. Returns the simulation.

### `lightsim.spec.Simulation.to_dict`

```text
Simulation.to_dict()
```

The simulation as a plain dictionary in the JSON format of the core.

### `lightsim.spec.Simulation.to_json`

```text
Simulation.to_json()
```

The simulation as a JSON string, as read by `lightsim run`.

### `lightsim.spec.Simulation.geometry_json`

```text
Simulation.geometry_json()
```

The simulation as JSON without its sources and monitors: all that the grid and the permittivity of a plane depend on. The mode solvers send this to the core, so that their cost does not grow with the mode profiles already attached to monitors and sources.

### `lightsim.spec.Simulation.grid`

```text
Simulation.grid()
```

Node coordinates `(x, y, z)` of the grid as numpy arrays (m), absorbing layers included. Nothing is run.

### `lightsim.spec.Simulation.frozen_mesh`

```text
Simulation.frozen_mesh()
```

The grid of this simulation as an explicit mesh.

Returns the mesh dict (see `ExplicitMesh`) with the node coordinates the simulation has now, inside the region; absorbing layers added outside the region are not part of a mesh and come back when the mesh is used. An automatic mesh depends on the structures, so a copy of the simulation with structures removed or added gets another grid. Give the copy this mesh to keep the grid, for example to solve port modes on a single waveguide of a larger device:

```text
mesh = device.frozen_mesh()
single = Simulation(min=device.min, max=device.max, mesh=mesh, ...)
```

### `lightsim.spec.Simulation.freeze_mesh`

```text
Simulation.freeze_mesh()
```

Replace the mesh by the explicit mesh of the current grid (`frozen_mesh`), so that later changes of the structures no longer move the grid. Returns the simulation.

### `lightsim.spec.Simulation.time_info`

```text
Simulation.time_info()
```

`(dt, steps)`: the time step in seconds and the number of steps that `run_time` gives. The time step is the Courant step of the mesh times `courant`, lowered if a dispersive medium needs it. Nothing is run.

### `lightsim.spec.Simulation.run`

```text
Simulation.run(device='auto', progress=0)
```

Run the simulation.

**Parameters**

- `device` (str): `"auto"` (the first GPU if there is one, else the CPU), `"cpu"`, `"cuda"`, `"cuda:1"`, `"cuda:0,1"` (split over several GPUs) or `"cuda:all"`.
- `progress` (int): Print a line every `progress` steps; 0 prints nothing.

Returns a `Results`. A run whose fields grow without bound raises a `RuntimeError`; one that reaches `run_time` before the fields have decayed below `shutoff` returns with a warning.

### `lightsim.spec.Simulation.view`

```text
Simulation.view(results=None, **kwargs)
```

Open the graphical interface for this simulation (and optionally its `results`) and return its URL. Keyword arguments go to `lightsim.gui.view` (`host`, `port`, `open_browser`, `block`).

### `lightsim.spec.Results`

```text
Results(raw)
```

Thin wrapper around the result dict returned by the core.

The attribute `raw` is that dictionary, with the keys `dt`, `stats`, `dft`, `flux`, `modes`, `far_field`, `time`, `energy`, `sources` and `source_spectra`.

### `lightsim.spec.Results.stats`

Property. Run statistics: `backend`, `cells`, `steps` (executed), `seconds`, `mcells_per_s` (million cell updates per second), `stopped_early`, and the arrays `energy_steps` and `energy`, the field energy that `shutoff` looks at, every `shutoff_interval` steps.

### `lightsim.spec.Results.dt`

Property. Time step in seconds.

### `lightsim.spec.Results.decay`

```text
Results.decay()
```

Field energy at the end of the run as a fraction of its peak, the number that `shutoff` is compared with (`None` if the run was too short for a sample). `stats["energy_steps"]` and `stats["energy"]` hold the whole history.

### `lightsim.spec.Results.flux`

```text
Results.flux(name)
```

Power per frequency of a `FluxMonitor` or `FluxBoxMonitor` (also the total power through the plane of a `ModeMonitor` and through the box of a `FarFieldMonitor`), as an array in the order of the monitor frequencies.

### `lightsim.spec.Results.field`

```text
Results.field(monitor, comp)
```

One field component of a `DFTMonitor`: a complex array indexed `[frequency, x, y, z]`, in single precision. Magnetic components are multiplied by the impedance of free space (376.73 ohm), so they have the unit of the electric field; divide by it to obtain A/m.

### `lightsim.spec.Results.freqs`

```text
Results.freqs(monitor)
```

Frequencies (Hz) of a frequency-domain monitor, in the order given.

### `lightsim.spec.Results.mode_amplitudes`

```text
Results.mode_amplitudes(monitor, mode=0)
```

(a_plus, a_minus) complex arrays over frequency.

### `lightsim.spec.Results.time_series`

```text
Results.time_series(monitor, comp)
```

`(times, values)` of one component of a `ProbeMonitor` or `FieldTimeMonitor`: the sample times in seconds and an array indexed `[sample, x, y, z]`. The times are those of the electric field; magnetic samples are half a time step earlier and are multiplied by the impedance of free space.

### `lightsim.spec.Results.gain_levels`

```text
Results.gain_levels()
```

Gain media: the number of atoms in levels 0, 1, 2 and 3 at the end of the run, summed over all gain cells (density times the volume of each E-field sample), as a numpy array; `None` without gain media. For four-level two-electron media these are electrons (two per atom).

### `lightsim.spec.Results.far_field`

```text
Results.far_field(monitor)
```

Far field of a `FarFieldMonitor`, as a dict of numpy arrays:

- `freqs` (Hz), `theta`, `phi` (radians): the axes;
- `intensity`: radiant intensity `U` in W/sr, shape `[frequency, theta, phi]`; its integral over the sphere, `sum U sin(theta) dtheta dphi`, is the radiated power;
- `e_theta`, `e_phi`: complex far-field amplitudes in volts, same shape. Far from the box the electric field is `(e_theta theta^ + e_phi phi^) exp(i k r) / r` with `r` the distance from the coordinate origin, and `U = (|e_theta|^2 + |e_phi|^2) / (2 eta)` with `eta` the wave impedance of the outer medium, 376.73 ohm / `n_medium`;
- `flux`: power through the box per frequency (W), for comparison with the integral of `U`.

Like every frequency-domain result, the values scale with the source spectrum (fields with `S`, intensity and flux with `abs(S)**2`).

### `lightsim.spec.Results.energy`

```text
Results.energy(monitor)
```

`(times, energy)` of an `EnergyMonitor`: the sample times in seconds and the electromagnetic energy in the monitor box in joules, as numpy arrays.

### `lightsim.spec.Results.source_spectrum`

```text
Results.source_spectrum(name=None, freqs=None, monitor=None)
```

Spectrum of a source, for normalising monitor data.

Returns `(freqs, spectrum)`: the frequencies in Hz and the complex Fourier transform `S` of the source's signal at those frequencies, in the order given. Dividing a monitor field by `S`, or a power by `abs(S)**2`, gives the response per unit source `amplitude`.

**Parameters**

- `name` (str, optional): Name of the source; unnamed sources are `"source0"`, `"source1"`, ... May be omitted if there is one source.
- `freqs` (sequence of float, optional): Frequencies (Hz) to evaluate at, in any order.
- `monitor` (str, optional): Evaluate at the frequencies of this frequency-domain monitor, in the monitor's order, so that the result lines up with `flux(monitor)`, `field(monitor, ...)` or `mode_amplitudes(monitor)`.

With neither `freqs` nor `monitor`, the frequencies are those of the frequency-domain monitors if they all record the same list in the same order. If they differ, a `ValueError` asks for one of the two arguments, because a spectrum on some other grid would be divided into the wrong monitor samples without any sign of it.

`S` is computed from the time samples of the run that was made (same time step and number of steps), so it includes the truncation of a run that stopped early. For a source with a complex field pattern in a run with complex fields (an oblique plane wave with Bloch boundaries) it is the spectrum of the complex signal that was injected, which is twice that of the real pulse.

### `lightsim.spec.C0`

Speed of light in vacuum, 299 792 458 m/s.

### `lightsim.spec.um`

One micrometre in metres, `1e-6`.

### `lightsim.spec.nm`

One nanometre in metres, `1e-9`.

### `lightsim.spec.PEC`

A `Medium` named `"PEC"` that is a perfect electric conductor.

### `lightsim.spec.PERIODIC`

Periodic boundary condition, for `Simulation.set_boundaries`.

### `lightsim.spec.PEC_WALL`

Perfect-electric-conductor boundary condition: tangential E is zero on the face.

### `lightsim.spec.PMC_WALL`

Perfect-magnetic-conductor boundary condition: tangential H is zero on the face.

## `lightsim.modes`

Waveguide mode solver and Gaussian-beam profiles.

### `lightsim.modes.Mode`

```text
Mode(
    neff,
    freq,
    te_fraction,
    profile,
    E,
    H,
    info,
    E_n=None,
    H_n=None,
    n_group=None,
    bend_radius=None,
)
```

A waveguide mode returned by `solve_modes` (or by `lightsim.xsection.CrossSection.solve`).

**Attributes**

- `neff` (complex): Effective index; a positive imaginary part is loss. For a bent guide it refers to the bend radius: the field goes as `exp(i k0 neff R phi)`.
- `freq` (float): Frequency (Hz) the mode was solved at.
- `te_fraction` (float): Share of the transverse electric field energy in the first in-plane axis, between 0 and 1.
- `profile` (dict): The field pattern in the form `ModeSource` and `ModeMonitor` take, normalised to unit power.
- `E` (tuple): `(E_u, E_v)`: the two transverse electric components, complex arrays over the mode window (V/m for unit power).
- `H` (tuple): `(H_u, H_v)` in A/m, at the plane of `E`.
- `info` (dict): The mode window: `u_nodes`, `v_nodes` (m), `dims`, `axes`, `origin`, the permittivity maps `eps_u`, `eps_v`, `eps_n` (and `eps_*_im`), `coord` and `h_offset`. Modes from a `CrossSection` also carry `fill` (the fill fraction of every material at the three electric positions) and `eps_material`.
- `E_n`, `H_n` (ndarray or None): Longitudinal components (E at the nodes, H at the cell centres), or `None` if the solver that made the mode did not compute them.
- `n_group` (float or None): Group index, when it was asked for (`group_index=True`).
- `bend_radius` (float or None): Bend radius (m) the mode was solved for; `None` for a straight guide.

Fields:

- `neff`
- `freq`
- `te_fraction`
- `profile`: ModeProfile dict for lightsim sources/monitors
- `E`: (E_u, E_v) complex arrays on the plane (natural order)
- `H`: (H_u, H_v) complex arrays (A/m) at the E plane
- `info`
- `E_n`
- `H_n`
- `n_group`
- `bend_radius`

### `lightsim.modes.Mode.wavelength`

Property. Vacuum wavelength (m).

### `lightsim.modes.Mode.beta`

Property. Propagation constant `k0 * neff` (1/m).

### `lightsim.modes.Mode.fraction_te`

Property. Same as `te_fraction` (gplugins / Tidy3D name).

### `lightsim.modes.Mode.fraction_tm`

Property. `1 - te_fraction`.

### `lightsim.modes.Mode.alpha`

Property. Power attenuation coefficient `2 k0 Im(neff)` (1/m).

### `lightsim.modes.Mode.loss_db_per_cm`

Property. Propagation loss in dB/cm, `10 log10(e) * alpha / 100`. For a bend, per cm of path along the bend radius.

### `lightsim.modes.Mode.loss_db_per_90deg`

```text
Mode.loss_db_per_90deg()
```

Loss of a 90-degree turn (dB), for a mode solved with `bend_radius`.

### `lightsim.modes.Mode.effective_area`

```text
Mode.effective_area()
```

Effective mode area `(∫|E|² dA)² / ∫|E|⁴ dA` (m²), with all three electric components (the definition Tidy3D's `mode_area` uses).

### `lightsim.modes.Mode.energy_fraction`

```text
Mode.energy_fraction(material=None, region=None)
```

Share of the electric energy `∫ eps |E|² dA` inside a material (name, for cross-section modes) or a rectangular `region` `((u0, v0), (u1, v1))` in metres. See `mode_confinement`.

### `lightsim.modes.Mode.confinement`

```text
Mode.confinement(material=None, region=None, kind='energy')
```

Confinement factor in a material or region; `kind` is `"energy"` (electric energy) or `"power"` (longitudinal Poynting flux). See `mode_confinement`.

### `lightsim.modes.Mode.power_in`

```text
Mode.power_in(region)
```

Share of the mode power (Poynting flux through the plane) inside the rectangle `region = ((u0, v0), (u1, v1))` (m).

### `lightsim.modes.Mode.overlap`

```text
Mode.overlap(other, conjugate=True)
```

Overlap with another mode, see `mode_overlap`.

### `lightsim.modes.Mode.fields_at_centres`

```text
Mode.fields_at_centres()
```

All six components averaged to the cell centres: a dict with `u`, `v` (centre coordinates, m) and `Eu, Ev, En, Hu, Hv, Hn` (`En`/`Hn` zero if not computed).

### `lightsim.modes.pml_stretch`

```text
pml_stretch(nodes, cells_lo, cells_hi, freq, strength=4.0, kappa=1.0, order=2.0)
```

Complex stretch factors of a perfectly matched layer along one axis.

The layer covers the outer `cells_lo` / `cells_hi` cells of `nodes`. With the depth `rho` (0 at the inner edge of the layer, 1 at the wall) the factor is `s = 1 + (kappa - 1) rho^order + i strength rho^order`; a wave `exp(i k x)` is damped by `exp(-k Im ∫(s-1) dx)`, i.e. by `exp(-k * strength * thickness / (order + 1))` in one pass.

Returns `(s_at_centres, s_at_nodes)` for the cell centres and for the nodes `nodes[:-1]` (the last node is the closed wall).

### `lightsim.modes.bend_transform`

```text
bend_transform(eps, mu, eps_off, nodes_x, nodes_y, radius, bend_axis=0, center=0.0)
```

Material of the straight guide that is equivalent to a bend.

A bend of radius `radius` about an axis parallel to the solver's other transverse axis maps exactly onto a straight guide with `eps' = eps diag(r/R, r/R, R/r)` and `mu' = mu diag(r/R, r/R, R/r)`, `r = R + (x - center)` the distance from the centre of curvature, `x` the coordinate along `bend_axis` (0 or 1) of the solver lattice. A positive radius puts the centre of curvature on the low side. Off-diagonal terms scale with `(r/R) / (h_i h_j)`.

### `lightsim.modes.clear_cache`

```text
clear_cache()
```

Forget the cached mode solutions (see `cache=True` in the solvers).

### `lightsim.modes.solve_fields`

```text
solve_fields(
    eps,
    nodes_x,
    nodes_y,
    freq,
    num_modes=2,
    target_neff=None,
    *,
    mu=None,
    eps_off=None,
    stretch=None,
    periodic=(False, False),
    tol=1e-10,
    cache=False,
    timings=None,
)
```

Core eigen-solve on the solver lattice (right-handed x, y, propagation).

**Parameters**

- `eps` ((eps_x, eps_y, eps_z)): Complex (Nx, Ny) arrays at the E_x, E_y, E_z positions.
- `mu` ((mu_x, mu_y, mu_z), optional): Diagonal permeability at the H_x, H_y, H_z positions (default 1).
- `eps_off` (dict, optional): Off-diagonal permittivity: keys `"xy", "xz"` (arrays at the E_x positions), `"yx", "yz"` (at E_y), `"zx", "zy"` (at E_z). When given, the full four-component problem is solved.
- `stretch` (((sx_c, sx_n), (sy_c, sy_n)), optional): PML stretch factors from `pml_stretch` (`None` per axis for none).
- `periodic` ((bool, bool)): Periodic lattice along x / y instead of closed walls.
- `cache` (bool): Reuse the result of an identical earlier call (same arrays and settings) instead of solving again.
- `timings` (dict, optional): Filled with the seconds spent assembling, factorising and iterating.

Returns a list (highest real effective index first) of dicts with `neff` and the six field components `Ex, Ey, Ez, Hx, Hy, Hz` (H normalised as `eta0 H`), normalised to unit power, and the areas `a_ex`, `a_ey` used for the power.

### `lightsim.modes.solve_cross_section`

```text
solve_cross_section(
    eps_x,
    eps_y,
    eps_z,
    nodes_x,
    nodes_y,
    freq,
    num_modes=2,
    target_neff=None,
    **options,
)
```

Solve the transverse eigenproblem on a (right-handed x, y, propagation) cross-section. `eps_*` are (Nx, Ny) arrays at the Yee positions. Returns a list of `(neff, Ex, Ey, Hx, Hy, a_ex, a_ey)` with H normalized (H~ = eta0 H).

Keyword `options` are passed to `solve_fields` (`mu`, `eps_off`, `stretch`, `periodic`, `cache`, `timings`).

### `lightsim.modes.solve_plane`

```text
solve_plane(
    eps_u,
    eps_v,
    eps_n,
    u_nodes,
    v_nodes,
    freq,
    num_modes=2,
    target_neff=None,
    *,
    swap=False,
    info=None,
    pml=None,
    bend_radius=None,
    bend_axis='u',
    bend_center=None,
    eps_off=None,
    periodic=(False, False),
    sort='neff',
    cache=False,
    timings=None,
    profile=True,
)
```

Solve the modes of permittivity maps given in a plane's natural axes (u, v) and return `Mode` objects.

`eps_u`, `eps_v`, `eps_n` are complex (Nu, Nv) arrays at the E_u, E_v and E_n positions; `swap` is true when (u, v, normal) is left-handed (planes normal to y), in which case the solver works in (v, u). `eps_off` holds off-diagonal terms in natural axes, keys `"uv", "un", "vu", "vn", "nu", "nv"` at the positions of the row component. The other options are those of `solve_modes`.

### `lightsim.modes.solve_modes`

```text
solve_modes(
    sim,
    axis,
    position,
    min,
    max,
    freq,
    num_modes=2,
    target_neff=None,
    sort='neff',
    *,
    pml=None,
    bend_radius=None,
    bend_axis='u',
    bend_center=None,
    group_index=False,
    cache=False,
)
```

Solve waveguide modes on a plane of a `lightsim.Simulation`.

Returns a list of `Mode` sorted by effective index (or by TE fraction first when `sort="te"`).

In a 2-D simulation the plane is a line and the modes are those of the 1-D solver `lightsim.eim.solve_port_mode`, for the polarization(s) the simulation contains: `te_fraction` is 1 for the modes with E in the plane and 0 for those with E along the invariant axis.

Keyword-only options (3-D simulations):

pml : int, tuple or dict, optional

```text
    Perfectly matched layers in the outer cells of the window: a number
    of cells on all four sides, four numbers ``(u_lo, u_hi, v_lo, v_hi)``
    or a dict ``{"cells": ..., "strength": 4.0, "kappa": 1.0,
    "order": 2}`` (see :func:`pml_stretch`). Needed for leaky modes,
    substrate leakage and bends. Default: closed walls.
bend_radius : float, optional
    Solve the modes of a bend of this radius (m). The bend turns in the
    plane of ``bend_axis`` (``"u"``: the first in-plane axis is radial)
    about a centre at ``bend_center - bend_radius`` on that axis
    (``bend_center`` defaults to the middle of the window). ``neff``
    then refers to the radius ``bend_radius``. Use with ``pml``. The
    bend modes nearest the effective index of the straight guide (or
    ``target_neff``) are returned, least lossy first.
group_index : bool
    Also compute the group index ``n_group`` from the energy-flux
    relation, with the dispersion of the simulation's media.
cache : bool
    Reuse an identical earlier solve.
```

### `lightsim.modes.gaussian_mode`

```text
gaussian_mode(
    sim,
    axis,
    position,
    min,
    max,
    freq,
    waist,
    center=None,
    polarization='v',
    n=1.0,
    focus_distance=0.0,
    theta=0.0,
    phi=0.0,
)
```

Gaussian-beam reference profile for `ModeMonitor` / `ModeSource` (cf. FDTDX `GaussianModeOverlapDetector`).

The beam travels towards +`axis` in a medium of index `n`, with waist `waist` (1/e field radius at the focus, m), sampled on the Yee positions of the plane window `min`..`max` at `position`. It is the beam of `lightsim.GaussianBeam` with the same parameters:

**Parameters**

- `center` ((x, y, z), optional): Where the beam axis crosses the plane (its two in-plane coordinates are used); default: the centre of the window.
- `polarization` ("u" or "v"): The electric field points along the plane's first (`"u"`) or second (`"v"`) in-plane axis, projected onto the plane normal to the beam axis. The in-plane axes are (y, z), (x, z), (x, y) for axis 0, 1, 2.
- `focus_distance` (float): Distance (m) from the plane to the focus along the beam axis (negative: the focus lies before the plane).
- `theta`, `phi` (float): Polar angle (rad) between the beam axis and +`axis`, and azimuth of the tilt from the first in-plane axis towards the second.

In a simulation that is periodic along an in-plane axis with a period below twice the waist (a 2-D simulation), the beam is uniform along that axis (a sheet beam), as for the source.

Returns `(profile, power)`. `power` = 1/2 Re ∫ E × H\* dA of the profile, so the power a monitor couples into the beam is `abs(a_plus)**2 * power`.

### `lightsim.modes.mode_effective_area`

```text
mode_effective_area(mode)
```

Effective area `(∫|E|² dA)² / ∫|E|⁴ dA` (m²) of a `Mode`, with the three electric components averaged to the cell centres.

### `lightsim.modes.mode_confinement`

```text
mode_confinement(mode, material=None, region=None, kind='energy')
```

Confinement factor of a mode in a material or a region.

`kind="energy"`: share of the electric energy, `∫_A Re(eps) |E|² dA / ∫ Re(eps) |E|² dA`, with all three components at their own grid positions. In a cell that two materials share, the energy is split in proportion to `fill * eps` of each material.

`kind="power"`: share of the longitudinal Poynting flux, `∫_A Re(E × H*)·n dA / ∫ Re(E × H*)·n dA`.

`material` is a material name of a cross-section mode (one made by `lightsim.xsection.CrossSection`, which records the fill fraction of every material); `region` is a rectangle `((u0, v0), (u1, v1))` in metres. Give one of them, or both (their intersection).

### `lightsim.modes.group_index_of`

```text
group_index_of(mode, eps_group, mu=None)
```

Group index of a solved mode from the energy-flux relation `n_g = c W / P = (∫ d(w eps)/dw |E|² + mu |eta0 H|² dA) / (2 Re ∫ E × eta0 H* · n dA)`.

`eps_group = (eg_u, eg_v, eg_n)` holds `d(w eps)/dw` at the E_u, E_v, E_n positions (the permittivity itself for non-dispersive media).

### `lightsim.modes.mode_overlap`

```text
mode_overlap(mode1, mode2, conjugate=True)
```

Overlap integral of two modes on the plane, with fields as stored (unit power for solver modes).

`conjugate=True` (the power overlap, gplugins `innerprod_trapz` and Tidy3D `outer_dot(conjugate=True)`)::

```text
1/4 ∫ (E1* × H2 + E2 × H1*) · n dA
```

`|overlap|²` is the share of the power of `mode2` that couples into `mode1` for lossless, power-normalised modes, and the self-overlap of a solver mode is 1.

`conjugate=False` (the reciprocity form used for lossy and leaky modes, Tidy3D `outer_dot(conjugate=False)`)::

```text
1/4 ∫ (E1 × H2 + E2 × H1) · n dA
```

Modes on different grids are compared by interpolating `mode2` linearly onto `mode1`'s grid (zero outside `mode2`'s window); the two windows must share their coordinate frame.

### `lightsim.modes.overlap_matrix`

```text
overlap_matrix(modes1, modes2, conjugate=True)
```

`O[i, j] = mode_overlap(modes1[i], modes2[j], conjugate)`.

### `lightsim.modes.ETA0`

Impedance of free space, 376.730313412 ohm.

## `lightsim.xsection`

Waveguide cross-sections for the mode solver, from shapes or gdsfactory, without a 3-D simulation.

### `lightsim.xsection.uniaxial`

```text
uniaxial(n_o, n_e, optic_axis=(0.0, 0.0, 1.0))
```

Permittivity tensor of a uniaxial crystal as a function of wavelength.

`n_o`, `n_e`: ordinary and extraordinary indices (numbers, library names such as `"LiNbO3_o"` / `"LiNbO3_e"`, or functions of the wavelength in m). `optic_axis`: direction of the optic axis in the cross-section's (u, v, n) axes. Returns `f(wavelength) -> 3x3`.

Example: x-cut lithium niobate with the waveguide at an angle `a` to the crystal y axis in the chip plane (u horizontal, v vertical, n along the guide): `uniaxial("LiNbO3_o", "LiNbO3_e", (cos(a), 0, sin(a)))`.

### `lightsim.xsection.graded_nodes`

```text
graded_nodes(lo, hi, fine_lo, fine_hi, step, max_step=None, grading=1.2)
```

Nodes from `lo` to `hi`: uniform `step` in `[fine_lo, fine_hi]`, growing by `grading` per cell outside it up to `max_step`.

### `lightsim.xsection.CrossSection`

```text
CrossSection(
    background='sio2',
    u_nodes=None,
    v_nodes=None,
    material_index=None,
    subpixel=8,
)
```

A waveguide cross-section: shapes and materials on a (u, v) grid.

**Parameters**

- `background` (material): Material of everything not covered by a shape.
- `u_nodes`, `v_nodes` (array, optional): Grid nodes (m). If omitted, `mesh` builds a graded grid around the shapes when the cross-section is solved.
- `material_index` (dict, optional): Refractive index (or function of wavelength) per material name; it overrides the library for those names.
- `subpixel` (int): Sub-samples per cell and direction for the sub-pixel smoothing (even; 0 or 1: no smoothing, the material at the sample point).

Shapes added later are painted over earlier ones unless a `priority` says otherwise (higher wins).

### `lightsim.xsection.CrossSection.add_rect`

```text
CrossSection.add_rect(u0, v0, u1, v1, material, name=None, priority=None)
```

Rectangle `[u0, u1] x [v0, v1]` (m).

### `lightsim.xsection.CrossSection.add_layer`

```text
CrossSection.add_layer(v0, v1, material, name=None, priority=None)
```

Layer `v0 <= v < v1` across the whole width.

### `lightsim.xsection.CrossSection.add_polygon`

```text
CrossSection.add_polygon(vertices, material, name=None, priority=None)
```

Polygon with vertices `[(u, v), ...]` (m).

### `lightsim.xsection.CrossSection.add_trapezoid`

```text
CrossSection.add_trapezoid(
    u_center,
    width,
    v0,
    v1,
    material,
    sidewall_angle=0.0,
    name=None,
    priority=None,
)
```

Waveguide core with sloped sidewalls: `width` at `v0`, narrowing towards `v1` by `sidewall_angle` (degrees from the vertical, as in gdsfactory and Tidy3D).

### `lightsim.xsection.CrossSection.set_mesh`

```text
CrossSection.set_mesh(u_nodes, v_nodes)
```

Use this grid.

### `lightsim.xsection.CrossSection.mesh`

```text
CrossSection.mesh(
    window=None,
    step=2e-08,
    max_step=None,
    margin=3e-07,
    grading=1.2,
    conforming=True,
)
```

Build a graded grid over `window = ((u0, v0), (u1, v1))`: cells of at most `step` over the patterned shapes plus `margin`, growing by `grading` up to `max_step` (default `4 * step`) outside. With `conforming` (default) every rectangle and layer edge and every polygon vertex is a grid node (see `conforming_nodes`).

### `lightsim.xsection.CrossSection.strip`

```text
CrossSection.strip(
    core_width=5e-07,
    core_thickness=2.2e-07,
    core_material='si',
    clad_material='sio2',
    box_material=None,
    slab_thickness=0.0,
    slab_material=None,
    sidewall_angle=0.0,
    gap=None,
    core_width2=None,
    side_margin=1.5e-06,
    clad_thickness=1e-06,
    box_thickness=1e-06,
    step=2e-08,
    max_step=None,
    material_index=None,
)
```

Strip, rib or coupled waveguide, as gplugins' `Waveguide` and `WaveguideCoupler` (Tidy3D) and `get_mode_solver_rib` / `get_mode_solver_coupler` (MPB) describe them, in metres.

The core sits on the box at `v = 0` and is centred at `u = 0` (with `gap`, the two cores are at `u = -(gap + w1)/2` and `+(gap + w2)/2`). `slab_thickness` adds a slab of `slab_material` (default: the core material) under the core; the top cladding is `clad_material` and the bottom cladding `box_material` (default: the cladding).

### `lightsim.xsection.CrossSection.materials`

```text
CrossSection.materials()
```

Material name -> spec, background first.

### `lightsim.xsection.CrossSection.permittivity`

```text
CrossSection.permittivity(wavelength)
```

Smoothed permittivity at `wavelength` (m).

Returns `(eps_u, eps_v, eps_n, eps_off, fill, eps_material)`: the diagonal maps at the E_u, E_v, E_n positions, the off-diagonal terms (dict, empty for diagonal media), the fill fraction of every material at the three positions and the diagonal permittivity of every material. In a cell with an interface the diagonal term along the interface normal `n` is `n_i² <1/eps>^-1 + (1 - n_i²) <eps>` (the normal comes from the first moment of the permittivity).

### `lightsim.xsection.CrossSection.solve`

```text
CrossSection.solve(
    wavelength,
    num_modes=2,
    target_neff=None,
    *,
    pml=None,
    bend_radius=None,
    bend_center=0.0,
    group_index=False,
    sort='neff',
    cache=False,
    timings=None,
    profile=True,
    periodic=(False, False),
    extrapolate=False,
)
```

Modes of the cross-section at `wavelength` (m), highest effective index first (`sort="te"`: TE-like first).

`pml`, `bend_radius` as for `lightsim.solve_modes`; the bend turns about a centre at `u = bend_center - bend_radius` (default: the core centred on `u = 0`). `group_index=True` adds `n_group` from the energy-flux relation with the materials' dispersion (central difference of `w eps` over ±0.1 % in frequency).

`extrapolate=True` solves again on the grid with every cell split in two and returns the modes of the fine grid with the Richardson extrapolation `(4 n_fine - n_coarse) / 3` of `neff` (and of `n_group`); the two values are kept in `info["neff_grids"]`. It removes the leading (second-order) discretisation error when every interface lies on a grid node (the default `mesh`), and costs about five times one solve.

### `lightsim.xsection.CrossSection.group_permittivity`

```text
CrossSection.group_permittivity(wavelength, rel=0.001)
```

`d(w eps)/dw` at the E_u, E_v, E_n positions (central difference over `±rel` in frequency).

### `lightsim.xsection.CrossSection.add_group_index`

```text
CrossSection.add_group_index(modes, wavelength)
```

Set `n_group` of modes solved on this cross-section.

### `lightsim.xsection.CrossSection.plot`

```text
CrossSection.plot(wavelength=1.55e-06, ax=None, component='n')
```

Plot the real index map `sqrt(eps)` of one position set (needs matplotlib).

### `lightsim.xsection.from_component`

```text
from_component(
    component,
    x=None,
    y=None,
    layer_stack=None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    margin=1.5e-06,
    v_margin=1e-06,
    span=None,
    step=2e-08,
    max_step=None,
    background='sio2',
    sidewall=False,
)
```

Cross-section of a gdsfactory component on the plane `x = x` (light along x; u = y, v = z) or `y = y` (light along y; u = x, v = z), coordinates in metres.

The polygons of every patterned level of the layer stack are cut by the plane; wafer levels (buried oxide, cladding) become full-width layers. Levels are painted as in `lightsim.gds.component_to_structures` (lower `mesh_order` wins). `span = (u0, u1)` sets the lateral extent of the window, otherwise it is the extent of the cut plus `margin`; vertically the window covers the patterned levels plus `v_margin`. `sidewall=True` draws the levels' sidewall angles (the 3-D import draws vertical walls).

### `lightsim.xsection.from_gdsfactory`

```text
from_gdsfactory(
    cross_section='strip',
    layer_stack=None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    margin=1.5e-06,
    v_margin=1e-06,
    step=2e-08,
    max_step=None,
    background='sio2',
    sidewall=False,
    **cross_section_kwargs,
)
```

Cross-section of a gdsfactory cross-section (a name such as `"strip"` or `"rib"`, a `CrossSection`, or a function returning one) extruded with a `LayerStack` (default: the active PDK's).

The cross-section is drawn as a short straight and cut in the middle (see `from_component`), so derived layers and slab sections are handled as in the 3-D import. The guide runs along x: u = y, v = z.

## `lightsim.mode_sweeps`

Mode-solver sweeps (width, thickness, wavelength, temperature, gap, bend radius) as tidy tables.

### `lightsim.mode_sweeps.table`

```text
table(rows)
```

`pandas.DataFrame` of the rows, or a dict of lists without pandas.

### `lightsim.mode_sweeps.mode_row`

```text
mode_row(m, **extra)
```

The standard columns of one mode.

### `lightsim.mode_sweeps.sweep`

```text
sweep(
    make,
    values,
    wavelength=1.55e-06,
    num_modes=2,
    parameter='value',
    group_index=True,
    sort='neff',
    **solve_kwargs,
)
```

Solve `make(value)` (a function returning a `CrossSection`) for every value and tabulate the modes; the swept value is the column `parameter`.

### `lightsim.mode_sweeps.neff_vs_width`

```text
neff_vs_width(
    widths,
    wavelength=1.55e-06,
    num_modes=2,
    group_index=True,
    **strip_kwargs,
)
```

Effective and group index against core width (m), column `width`.

### `lightsim.mode_sweeps.neff_vs_thickness`

```text
neff_vs_thickness(
    thicknesses,
    wavelength=1.55e-06,
    num_modes=2,
    group_index=True,
    **strip_kwargs,
)
```

Effective and group index against core thickness (m), column `thickness`.

### `lightsim.mode_sweeps.neff_vs_wavelength`

```text
neff_vs_wavelength(wavelengths, num_modes=1, xs=None, **strip_kwargs)
```

Dispersion: effective index, group index and group-velocity dispersion against wavelength (m).

Extra columns: `wavelength`, `D_ps_nm_km` (`D = (1/c) d n_g / d lambda`, from central differences of `ng` between neighbouring wavelengths; NaN at the ends of the list) and `beta2_ps2_km` (`beta_2 = -lambda² D / (2 pi c)`). Modes are matched between wavelengths by their index in the list, so keep `num_modes` small where modes cross.

### `lightsim.mode_sweeps.with_temperature`

```text
with_temperature(material, delta_t, dn_dt=None)
```

Material spec at `delta_t` kelvin above the reference: `n + dn/dT delta_t` with `dn/dT` from `THERMO_OPTIC` (by name) unless given.

### `lightsim.mode_sweeps.neff_vs_temperature`

```text
neff_vs_temperature(
    temperatures,
    wavelength=1.55e-06,
    num_modes=1,
    t0=25.0,
    dn_dT=None,
    core_material='si',
    clad_material='sio2',
    **strip_kwargs,
)
```

Effective and group index against temperature (degrees C), with linear thermo-optic coefficients (`THERMO_OPTIC`, or `dn_dT` per material name). Extra column `dneff_dT` (1/K, central differences).

### `lightsim.mode_sweeps.supermodes`

```text
supermodes(gap, wavelength=1.55e-06, num_modes=4, xs=None, **strip_kwargs)
```

Supermodes of two parallel guides (`CrossSection.strip(gap=...)`).

### `lightsim.mode_sweeps.coupling_from_supermodes`

```text
coupling_from_supermodes(n_even, n_odd, wavelength, power_ratio=1.0)
```

Coupling of a symmetric directional coupler from its supermodes.

`kappa = pi (n_even - n_odd) / lambda` (1/m) is the coupling coefficient of coupled-mode theory, `Lc = pi / (2 kappa) = lambda / (2 dn)` the length for full transfer and `L(power_ratio) = arcsin( sqrt(power_ratio)) / kappa` the length for a fraction `power_ratio` of the power in the cross port.

### `lightsim.mode_sweeps.coupling_vs_gap`

```text
coupling_vs_gap(
    gaps,
    wavelength=1.55e-06,
    polarization='te',
    power_ratio=1.0,
    **strip_kwargs,
)
```

Supermodes and coupling against gap (m).

Columns: `gap`, `n_even`, `n_odd`, `dn`, `kappa` (1/m), `lc` (m, full transfer), `length` (m, for `power_ratio`). The even and odd supermodes are the two highest-index modes of the chosen polarization (`"te"`: TE fraction above 0.5).

### `lightsim.mode_sweeps.bend_sweep`

```text
bend_sweep(
    radii,
    wavelength=1.55e-06,
    num_modes=1,
    pml_cells=20,
    outer_margin=2e-06,
    vertical_pml=False,
    xs_factory=None,
    **strip_kwargs,
)
```

Bend modes against radius (m).

For each radius the cross-section is widened by `outer_margin` on the outer side and terminated there by `pml_cells` of PML (also below and above with `vertical_pml`, for leakage into a substrate; a PML that reaches the evanescent tail of the mode adds loss of its own). Columns: `radius`, `mode`, `neff` (at the radius), `loss_db_90` (loss of a 90-degree turn), `loss_db_cm`, `mismatch_db` (loss of one straight-to-bend junction, `-10 log10 |<bend|straight>|²` with the conjugated overlap of `lightsim.modes.mode_overlap`), `shift` (m, outward shift of the power centroid relative to the straight guide).

### `lightsim.mode_sweeps.overlap_table`

```text
overlap_table(modes1, modes2, conjugate=True)
```

Overlap matrix of two mode lists as a tidy table (columns `i`, `j`, `overlap`, `power` = `|overlap|²`); see `lightsim.modes.mode_overlap` for the definitions.

## `lightsim.gplugins_modes`

Mode-solver functions with the names, micrometre arguments and return types of gplugins.

### `lightsim.gplugins_modes.GMode`

A mode in the form gplugins' `gplugins.modes.types.Mode` has (micrometres): `mode_number` (1 = fundamental), `wavelength`, `neff`, `ng`, `fraction_te`, `fraction_tm`, `effective_area` (µm²), the fields on the window and `y`, `z` (µm). `mode` holds the lightsim `Mode`.

### `lightsim.gplugins_modes.find_modes_waveguide`

```text
find_modes_waveguide(
    tol=1e-06,
    wavelength=1.55,
    mode_number=1,
    parity=None,
    cache_path=None,
    overwrite=False,
    single_waveguide=True,
    nmodes=4,
    group_index=False,
    extrapolate=False,
    **kwargs,
)
```

Modes of a strip or rib waveguide (gplugins.modes.find_modes_waveguide).

Returns `{mode_number: GMode, ...}` for `nmodes` modes starting at `mode_number` (1 = highest effective index). Keyword arguments as gplugins: `core_width`, `core_thickness`, `slab_thickness`, `core_material`, `clad_material`, `nslab`, `sy`, `sz`, `resolution`, `sidewall_angle` (degrees), and for couplers `core_widths`, `gaps`, `gap`, `ymargin`, `sidewall_angles`. `tol`, `parity`, `cache_path` and `overwrite` are accepted for compatibility and not used.

### `lightsim.gplugins_modes.find_modes_coupler`

```text
find_modes_coupler(**kwargs)
```

Supermodes of coupled waveguides (gplugins.modes.find_modes_coupler): `core_widths`, `gaps` (µm) as in gplugins.

### `lightsim.gplugins_modes.find_neff_vs_width`

```text
find_neff_vs_width(
    width1=0.2,
    width2=1.0,
    steps=12,
    nmodes=4,
    wavelength=1.55,
    parity=None,
    filepath=None,
    overwrite=False,
    **kwargs,
)
```

Effective index against core width (gplugins.modes.find_neff_vs_width): a DataFrame with columns `1 .. nmodes` (effective index of each mode) and `width` (µm).

### `lightsim.gplugins_modes.find_mode_dispersion`

```text
find_mode_dispersion(
    wavelength=1.55,
    wavelength_step=0.01,
    core='Si',
    clad='SiO2',
    mode_number=1,
    **kwargs,
)
```

Effective and group index with material dispersion (gplugins.modes.find_mode_dispersion).

As in gplugins, the modes are solved at `wavelength` and `wavelength ± wavelength_step` with the core and cladding indices of those wavelengths, `ng = n_c - lambda (n_+ - n_-) / (2 step)` and the returned `neff` is the mean of the three. The materials are lightsim library names (`"Si"`: Li 1980, `"SiO2"`: Malitson 1965).

### `lightsim.gplugins_modes.find_neff_ng_dw_dh`

```text
find_neff_ng_dw_dh(
    width=0.465,
    thickness=0.215,
    delta_width=0.03,
    delta_thickness=0.02,
    wavelength=1.55,
    steps=11,
    mode_number=1,
    core='Si',
    clad='SiO2',
    **kwargs,
)
```

Effective and group index on a grid of width and thickness deviations (gplugins.modes.find_neff_ng_dw_dh): DataFrame with `dw`, `dh` (µm), `neff`, `ng`.

### `lightsim.gplugins_modes.coupling_length`

```text
coupling_length(neff1, neff2, power_ratio=1.0, wavelength=1.55)
```

Coupler length (µm) for `power_ratio` (1 = full transfer) from the even and odd supermode indices (gplugins.modes.find_coupling_vs_gap).

### `lightsim.gplugins_modes.find_coupling_vs_gap`

```text
find_coupling_vs_gap(
    gap1=0.2,
    gap2=0.4,
    steps=12,
    nmodes=4,
    wavelength=1.55,
    parity=None,
    filepath=None,
    overwrite=False,
    **kwargs,
)
```

Supermodes and full-transfer length against gap (gplugins.modes.find_coupling_vs_gap): DataFrame with `gap`, `ne`, `no`, `lc` (µm), `dn`. As in gplugins, `ne` and `no` are modes 1 and 2 (the two highest effective indices).

### `lightsim.gplugins_modes.innerprod_trapz`

```text
innerprod_trapz(mode1, mode2, **_ignored)
```

Conjugated overlap `1/4 ∫ (E1* × H2 + E2 × H1*) · x dA` of two modes (gplugins.modes.overlap.innerprod_trapz). Accepts `GMode` or `Mode`. lightsim modes carry unit power, so the self-overlap is 1 (gplugins' MPB fields are not power-normalised).

### `lightsim.gplugins_modes.Waveguide`

```text
Waveguide(
    wavelength,
    core_width,
    core_thickness,
    core_material,
    clad_material,
    box_material=None,
    slab_thickness=0.0,
    clad_thickness=None,
    box_thickness=None,
    side_margin=None,
    sidewall_angle=0.0,
    sidewall_thickness=0.0,
    sidewall_k=0.0,
    surface_thickness=0.0,
    surface_k=0.0,
    bend_radius=None,
    target_neff=None,
    num_modes=2,
    group_index_step=False,
    precision='double',
    grid_resolution=20,
    max_grid_scaling=1.2,
    cache_path=None,
    overwrite=False,
    gap=None,
)
```

Waveguide model with the interface of `gplugins.tidy3d.modes.Waveguide` (all lengths in µm), solved with lightsim.

The core (`core_width` x `core_thickness`) sits on the box at z = 0, centred laterally; `slab_thickness` adds a slab of the core material; `clad_material` above, `box_material` (default: the cladding) below. `side_margin`, `clad_thickness`, `box_thickness` default to the wavelength. `bend_radius` solves the bend, with 12 PML cells on the outer wall (gplugins asks Tidy3D for 12 on both transverse axes; here the vertical walls stay closed, so substrate leakage is not included). `grid_resolution` is the number of cells per wavelength in the core material, refined nowhere else; `max_grid_scaling` the growth of the cells into the claddings. `group_index_step` (True or a relative step) adds `n_group`. `sidewall_thickness`/`surface_thickness` layers are not modelled (an error is raised if they are set).

### `lightsim.gplugins_modes.Waveguide.settings`

```text
Waveguide.settings()
```

The constructor arguments, as a dict.

### `lightsim.gplugins_modes.Waveguide.cross_section`

Property. The lightsim `CrossSection` (metres).

### `lightsim.gplugins_modes.Waveguide.modes`

Property. The lightsim `Mode` objects, one list per wavelength.

### `lightsim.gplugins_modes.Waveguide.n_eff`

Property. Complex effective indices (wavelength x mode, squeezed as Tidy3D).

### `lightsim.gplugins_modes.Waveguide.n_complex`

Property. Same as `n_eff` (Tidy3D name).

### `lightsim.gplugins_modes.Waveguide.n_group`

Property. Group indices, present when `group_index_step` is set.

### `lightsim.gplugins_modes.Waveguide.fraction_te`

Property. TE fractions (wavelength x mode).

### `lightsim.gplugins_modes.Waveguide.fraction_tm`

Property. TM fractions, `1 - fraction_te`.

### `lightsim.gplugins_modes.Waveguide.mode_area`

Property. Effective mode areas (µm²).

### `lightsim.gplugins_modes.Waveguide.loss_dB_per_cm`

Property. Propagation loss (dB/cm) from the imaginary part of `n_eff`.

### `lightsim.gplugins_modes.Waveguide.overlap`

```text
Waveguide.overlap(waveguide, conjugate=True)
```

Overlap matrix between the modes of the two waveguides at the (first) wavelength, as Tidy3D's `outer_dot` (see `lightsim.modes.mode_overlap`).

### `lightsim.gplugins_modes.WaveguideCoupler`

```text
WaveguideCoupler(
    wavelength,
    core_width,
    core_thickness,
    core_material,
    clad_material,
    gap,
    **kwargs,
)
```

Two parallel cores (`core_width` a pair, `gap` in µm), as `gplugins.tidy3d.modes.WaveguideCoupler`.

### `lightsim.gplugins_modes.WaveguideCoupler.coupling_length`

```text
WaveguideCoupler.coupling_length(power_ratio=1.0)
```

Coupling length (µm) from consecutive mode pairs (even, odd), as gplugins computes it: modes 0/1, 2/3, ...

### `lightsim.gplugins_modes.sweep_n_eff`

```text
sweep_n_eff(waveguide, **sweep_kwargs)
```

Effective index for every combination of the swept arguments (gplugins.tidy3d.modes.sweep_n_eff); an array of shape (sweeps..., [wavelength], [mode]) instead of an xarray.

### `lightsim.gplugins_modes.sweep_n_group`

```text
sweep_n_group(waveguide, **sweep_kwargs)
```

Group index sweep (needs `group_index_step`).

### `lightsim.gplugins_modes.sweep_fraction_te`

```text
sweep_fraction_te(waveguide, **sweep_kwargs)
```

TE fraction sweep.

### `lightsim.gplugins_modes.sweep_mode_area`

```text
sweep_mode_area(waveguide, **sweep_kwargs)
```

Mode area sweep (µm²).

### `lightsim.gplugins_modes.sweep_bend_mismatch`

```text
sweep_bend_mismatch(waveguide, bend_radii, track_modes=False, modes_to_track=(0,))
```

Bend-mismatch loss (gplugins.tidy3d.modes.sweep_bend_mismatch): for each radius `|overlap(bend, straight)|⁴` (both junctions of a bend), the diagonal for several modes or, with `track_modes`, the best match of each tracked straight mode.

### `lightsim.gplugins_modes.sweep_coupling_length`

```text
sweep_coupling_length(coupler, gaps, power_ratio=1.0)
```

Coupling length against gap (µm).

## `lightsim.eme`

Eigenmode expansion of devices cut into sections.

### `lightsim.eme.SMatrix`

```text
SMatrix(s11, s12, s21, s22)
```

Scattering matrix of a two-sided section in the mode bases of its left and right ends.

`s11` (left -> left, reflection), `s21` (left -> right, transmission), `s12` (right -> left), `s22` (right -> right). The first index of every block is the outgoing mode, the second the incoming one.

### `lightsim.eme.SMatrix.identity`

```text
SMatrix.identity(n)
```

A section of zero length between identical mode sets.

### `lightsim.eme.SMatrix.propagation`

```text
SMatrix.propagation(beta, length)
```

Propagation over `length` (m): `exp(i beta length)` per mode.

### `lightsim.eme.SMatrix.full`

```text
SMatrix.full()
```

`[[s11, s12], [s21, s22]]` as one matrix (left modes first).

### `lightsim.eme.star`

```text
star(a, b)
```

Redheffer star product: `a` followed by `b` (`a`'s right basis is `b`'s left basis).

### `lightsim.eme.power`

```text
power(s, n)
```

`s` cascaded `n` times (`n >= 1`) by repeated squaring.

### `lightsim.eme.EME`

```text
EME(sections, num_modes=10, pml=None, target_neff=None)
```

Eigenmode expansion of a sequence of sections.

**Parameters**

- `sections` (list of (CrossSection, length)): The device from left to right; lengths in metres (zero-length sections are allowed, e.g. at the ports). All cross-sections must share the same grid. A section may also be `("period", sections, n)`: `n` repetitions of a sub-list, combined by repeated squaring.
- `num_modes` (int): Modes per section (the same in every section).
- `pml` (optional): PML of the mode solves (see `lightsim.solve_modes`).

After `solve`, `smatrix` is the `SMatrix` in the mode bases of the first and last sections and `timings` holds the seconds spent solving modes, computing overlaps and cascading.

### `lightsim.eme.EME.solve`

```text
EME.solve(wavelength)
```

S-matrix of the device at `wavelength` (m), in the mode bases of the first and last sections (reference planes at the two ends).

### `lightsim.eme.EME.port_smatrix`

```text
EME.port_smatrix(left_ports, right_ports)
```

S-matrix between port modes after `solve`.

`left_ports` / `right_ports` are lists of `(name, modes)` with modes (from `CrossSection.solve` on the grid of the end sections) that represent each port, e.g. the mode of one output guide of a splitter. Amplitudes are converted with the symmetric unconjugated projection `C[p, m] = (<E_m, H_p> + <E_p, H_m>) / 2` between the port modes (normalised to `<E_p, H_p> = 1`) and the modes of the end section. Returns `(S, names)`, names `"port@mode"` in the order of the matrix.

### `lightsim.eme.stack_transfer_matrix`

```text
stack_transfer_matrix(indices, thicknesses, wavelength)
```

Exact normal-incidence reflection and transmission amplitudes of a multilayer stack (indices of the input medium, the layers and the output medium; layer thicknesses in m). Returns `(r, t, r_back, t_back)` for fields normalised to unit power flux in each medium (`|r|² + |t|² = 1` without loss).

### `lightsim.eme.component_sections`

```text
component_sections(
    component,
    layer_stack=None,
    cell_length=5e-07,
    x_range=None,
    span=None,
    margin=1e-06,
    v_margin=8e-07,
    step=2.5e-08,
    max_step=None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    background='sio2',
    merge=True,
)
```

Sections of a gdsfactory component cut along x (light along +x).

The extent `x_range = (x0, x1)` (m; default the component's x extent) is cut at every polygon edge normal to x (abrupt changes such as grating teeth or the ends of an MMI body), and each interval into equal cells of at most `cell_length`; each cell takes the cross-section at its centre (a staircase approximation of tapers and bends). With `merge`, neighbouring cells with the same cut are joined into one section, so straight parts cost one mode solve. A zero-length section with the cross-section at each end gives the reference planes `x0` and `x1`. All sections share one grid: `step` over the component's y extent (`span` sets the lateral window, default the extent plus `margin`) and over the patterned levels in z (plus `v_margin`).

Returns `(sections, info)`; `info` has the cell boundaries and the number of distinct cross-sections.

### `lightsim.eme.sparameters`

```text
sparameters(
    component,
    layer_stack=None,
    wavelengths=1.55e-06,
    num_modes=10,
    port_modes=1,
    cell_length=5e-07,
    step=2.5e-08,
    margin=1e-06,
    v_margin=8e-07,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    background='sio2',
    pml=None,
    return_eme=False,
)
```

S-parameters of a gdsfactory component by EME, light along x.

Ports facing west (180 degrees) are on the left end, ports facing east (0 degrees) on the right end; other ports raise an error. Each port is represented by the `port_modes` highest-index modes of the end cut restricted to that port's guide (shapes within half the distance to the neighbouring port); the device is expanded in `num_modes` modes per section. Returns a dict as `lightsim.sparams.write_sparameters`: `"o2@0,o1@0"` (out, in) complex arrays over `wavelengths` (m), plus `"wavelengths"` and `"timings"` (seconds per stage, summed over wavelengths). Lengths and coordinates in metres.

### `lightsim.eme.EMEComponent`

```text
EMEComponent(
    component,
    layer_stack=None,
    wavelength=1.55,
    temperature=25.0,
    num_modes=4,
    cell_length=0.5,
    spacing_x=2.0,
    spacing_y=2.0,
    step=0.025,
    material_index=None,
    background='sio2',
    **_ignored,
)
```

EME of a gdsfactory component with the interface of `gplugins.meow.MEOW` (lengths in µm).

`compute_sparameters()` returns `{"o1@0,o2@0": ..., "wavelengths": ...}` with the modes of the two end cross-sections as port modes (`o1@i`, `o2@j` for `i, j < num_modes`), wavelengths in µm, as meow does. The component must have exactly two optical ports, `o1` facing west and `o2` facing east. `resolution_x` / `resolution_y` are replaced by the grid `step` (µm, default 0.025); `spacing_x` / `spacing_y` are the window margins.

### `lightsim.eme.EMEComponent.compute_sparameters`

```text
EMEComponent.compute_sparameters()
```

Solve and return the S-parameters (keys `"o1@0,o2@0"` ordered in, out as meow and SAX; wavelength in µm).

## `lightsim.circuit`

Frequency-domain circuit simulation from netlists, with SAX-compatible models.

### `lightsim.circuit.reciprocal`

```text
reciprocal(sd)
```

Add `(b, a)` for every `(a, b)` (passive reciprocal device).

### `lightsim.circuit.sdense`

```text
sdense(sd, ports=None)
```

SDict -> `(S, ports)` with `S[..., out, in]` (lightsim / usual order; SAX's `sdense` is the transpose).

### `lightsim.circuit.sdict`

```text
sdict(S, ports)
```

`(S[..., out, in], ports)` -> SDict.

### `lightsim.circuit.straight`

```text
straight(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.0)
```

Waveguide of `length` µm with first-order dispersion: `n_eff(wl) = neff - (wl - wl0) (ng - neff) / wl0`; transmission `10^(-loss_dB_cm length 1e-4 / 20) exp(2 pi i n_eff length / wl)` (as `sax.models. straight`). Ports o1, o2.

### `lightsim.circuit.bend`

```text
bend(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.1, loss_dB=0.0)
```

Bend as a straight of the bend's path `length` (gdsfactory reports it in the instance `info`), plus a lumped `loss_dB` (mode mismatch and radiation of the whole bend). Ports o1, o2.

### `lightsim.circuit.phase_shifter`

```text
phase_shifter(*, wl=1.55, neff=2.34, voltage=0.0, length=10.0, loss=0.0, v_pi=1.0)
```

Phase shifter: `exp(i (2 pi neff length / wl + pi voltage / v_pi))`, amplitude `10^(-loss length / 20)` (`loss` in dB/µm, as `sax.models.phase_shifter` with `v_pi = 1`). Ports o1, o2.

### `lightsim.circuit.attenuator`

```text
attenuator(*, wl=1.55, loss=0.0)
```

Flat attenuation of `loss` dB. Ports o1, o2.

### `lightsim.circuit.coupler_ideal`

```text
coupler_ideal(*, wl=1.55, coupling=0.5)
```

Lossless 2x2 coupler with power `coupling` to the cross port. gdsfactory ports: o1, o2 on the left (bottom, top), o3, o4 on the right (top, bottom); bar o1-o4 and o2-o3, cross o1-o3 and o2-o4 (cross amplitude `i sqrt(coupling)`).

### `lightsim.circuit.coupler`

```text
coupler(
    *,
    wl=1.55,
    wl0=1.55,
    length=0.0,
    coupling0=0.2,
    dk1=1.2435,
    dk2=5.3022,
    dn=0.02,
    dn1=0.1169,
    dn2=0.4821,
)
```

Dispersive directional coupler, the model of `sax.models.coupler`: `phi = kappa0(wl) + pi dn(wl) length / wl`, bar `cos(phi)`, cross `-i sin(phi)`, with `kappa0 = coupling0 + dk1 dwl + dk2 dwl²/2` (bends) and `dn = dn + dn1 dwl + dn2 dwl²/2` (supermode index difference), `dwl = wl - wl0`. Ports as `coupler_ideal`.

### `lightsim.circuit.coupler_ring`

```text
coupler_ring(*, wl=1.55, coupling=0.1, loss_dB=0.0)
```

Bus-ring coupler with gdsfactory's `coupler_ring` ports: o1 (bus, west), o4 (bus, east), o2 and o3 (ring, west and east). Bar o1-o4 and o2-o3, cross o1-o3 and o2-o4 with power `coupling`.

### `lightsim.circuit.mmi1x2`

```text
mmi1x2(*, wl=1.55, wl0=1.55, fwhm=0.2, loss_dB=0.3)
```

1x2 splitter (o1 -> o2, o3), Gaussian in frequency, as `sax.models.mmi1x2`.

### `lightsim.circuit.mmi2x2`

```text
mmi2x2(
    *,
    wl=1.55,
    wl0=1.55,
    fwhm=0.2,
    loss_dB=0.3,
    shift=0.0,
    splitting_ratio_cross=0.5,
    splitting_ratio_thru=0.5,
)
```

2x2 MMI, as `sax.models.mmi2x2`: thru o1-o3 and o2-o4, cross (`i`) o1-o4 and o2-o3 (gdsfactory `mmi2x2` ports).

### `lightsim.circuit.grating_coupler`

```text
grating_coupler(
    *,
    wl=1.55,
    wl0=1.55,
    loss=0.0,
    reflection=0.0,
    reflection_fiber=0.0,
    bandwidth=0.04,
)
```

Fibre grating coupler: Gaussian transmission of `bandwidth` (µm, full width at half maximum of the power... as in `sax.models. grating_coupler`: the amplitude is `exp(-(wl - wl0)² / (4 sigma²))`, `sigma = bandwidth / (2 sqrt(2 ln 2))`), peak loss `loss` dB, reflections at the waveguide (o1) and fibre (o2) sides.

### `lightsim.circuit.ring_single`

```text
ring_single(
    *,
    wl=1.55,
    wl0=1.55,
    neff=2.34,
    ng=3.4,
    length=50.0,
    loss_dB_cm=0.0,
    coupling=0.1,
)
```

All-pass ring in closed form: `t = (r - a e^{i phi}) / (1 - r a e^{i phi})` with `r = sqrt(1 - coupling)`, round trip `a e^{i phi}` of a ring of circumference `length` µm (`straight` dispersion). Ports o1 (in), o2 (through).

### `lightsim.circuit.ring_double`

```text
ring_double(
    *,
    wl=1.55,
    wl0=1.55,
    neff=2.34,
    ng=3.4,
    length=50.0,
    loss_dB_cm=0.0,
    coupling1=0.1,
    coupling2=0.1,
)
```

Add-drop ring in closed form (couplers `coupling1` at the input bus, `coupling2` at the drop bus, couplers at diametrically opposite points). Ports o1 (in), o2 (through), o3 (drop, on the input side), o4 (add).

### `lightsim.circuit.model_from_sparameters`

```text
model_from_sparameters(sp, key_order='auto', interpolation='polar', keep_modes=None)
```

Model that interpolates an S-parameter dictionary in wavelength.

`sp` has `"wavelengths"` and complex arrays under keys `"a@m,b@n"`. `key_order`: `"out,in"` (lightsim's `write_sparameters` and the gplugins Meep writer) or `"in,out"` (gplugins' Tidy3D writer and SAX); `"auto"` takes `"out,in"` when the wavelengths are in metres (a lightsim file) and `"in,out"` otherwise. For reciprocal devices the two agree. Wavelengths in metres (lightsim) or micrometres (gplugins) are both accepted; the model takes `wl` in µm. `interpolation`: `"polar"` (magnitude and unwrapped phase, linear) or `"linear"` (real and imaginary parts, as gplugins). Port names drop `@0` when every key is mode 0, unless `keep_modes`.

### `lightsim.circuit.model_from_npz`

```text
model_from_npz(filepath, **kwargs)
```

Model from an S-parameter `.npz` file (the layout lightsim and gplugins share: keys `"o1@0,o2@0"` and `wavelengths`); see `model_from_sparameters` for the options.

### `lightsim.circuit.straight_from_cross_section`

```text
straight_from_cross_section(
    xs,
    wavelengths=(1.5, 1.55, 1.6),
    mode=0,
    extra_loss_dB_cm=0.0,
    **solve_kwargs,
)
```

Straight-waveguide model from the mode solver: the complex effective index of `mode` of the `CrossSection` is solved at `wavelengths` (µm) and fitted with a polynomial (degree up to 3), so the phase carries the full dispersion (group index and GVD) and the imaginary part the propagation loss (e.g. substrate leakage with a PML). `extra_loss_dB_cm` adds loss the solver does not see (roughness). The model is `f(*, wl=1.55, length=10.0) -> SDict` with ports o1, o2; `f.neff(wl)` gives the fitted index.

### `lightsim.circuit.bend_from_cross_section`

```text
bend_from_cross_section(
    xs,
    radius,
    angle=90.0,
    wavelengths=(1.5, 1.55, 1.6),
    mode=0,
    pml_cells=20,
    **solve_kwargs,
)
```

Circular-bend model from the mode solver: effective index (at the radius) and radiation loss of the bend mode of radius `radius` µm over `angle` degrees, and the two straight-to-bend junction losses from the overlap of the bend and straight modes, at `wavelengths` (µm), fitted in wavelength. The cross-section needs room for the PML on its outer side (+u). Model `f(*, wl=1.55) -> SDict`, ports o1, o2.

### `lightsim.circuit.coupler_from_cross_section`

```text
coupler_from_cross_section(
    xs,
    wavelengths=(1.5, 1.55, 1.6),
    polarization='te',
    coupling0=0.0,
)
```

Directional-coupler model from the supermodes of the coupled section (`xs`, e.g. `CrossSection.strip(gap=...)`): `phi = coupling0 + pi dn(wl) length / wl`, bar `cos(phi)`, cross `-i sin(phi)`, with the supermode index difference `dn` solved at `wavelengths` (µm) and fitted. `coupling0` is the phase picked up in the bends (0: ignored). Model `f(*, wl=1.55, length=10.0) -> SDict` with the ports of `coupler`; `f.n_mean(wl)` is the mean supermode index, used for the common phase.

### `lightsim.circuit.circuit`

```text
circuit(netlist, models=None, backend='auto', recursive_netlists=None, use_info=True)
```

Build a circuit function from a netlist and models.

**Parameters**

- `netlist` (dict): `component.get_netlist()` (gdsfactory: `instances`, `nets`, `ports`), the dict of `component.get_netlist(recursive=True)` (the top cell is the one no other cell instantiates), or a SAX netlist (`instances`, `connections`, `ports`).
- `models` (dict): Component name -> model function. Missing names fall back to `DEFAULT_MODELS` and, in a recursive netlist, to the cell's own netlist (a sub-circuit).
- `backend`: `"dense"`, `"sparse"` or `"auto"` (sparse above 400 internal ports).
- `use_info` (bool): Also pass the numbers of each instance's gdsfactory `info` (the path length of a bend, for example) to the models, as SAX 0.18 does with gdsfactory 9 netlists. `False` passes `settings` only.

Returns `(circuit_fn, info)`. `circuit_fn(wl=1.55, **overrides)` returns the SDict of the external ports; `overrides` maps an instance name to a dict of model arguments (`circuit_fn(wl=wl, top={"length": 20})`) or sets a global argument passed to every model that takes it. `info` lists the instances, their models and the port count.

### `lightsim.circuit.simulate`

```text
simulate(component, models=None, wl=1.55, recursive=True, **overrides)
```

S-parameters of a gdsfactory component from its netlist: `circuit(component.get_netlist(recursive=...), models)(wl=wl, **overrides)`. `wl` in µm.

## `lightsim.sparams`

S-parameters of gdsfactory components.

### `lightsim.sparams.detect_port_symmetries`

```text
detect_port_symmetries(component, ports=None)
```

Mirror symmetries of a component that map its ports onto each other.

The layout is compared with its mirror image about the horizontal and the vertical centre line of its bounding box, layer by layer and exactly (integer database units). For each mirror that leaves the layout unchanged and maps every optical port onto a port of the same width facing the mirrored way, the port map `{name: image}` is returned. Maps that send every port to itself are left out. The result is the list of generators `write_sparameters` accepts as `port_symmetry`; it is empty for a component without such a symmetry.

### `lightsim.sparams.build_simulation`

```text
build_simulation(
    component,
    wavelengths,
    layer_stack=None,
    material_index=None,
    steps_per_wavelength=20.0,
    xy_margin=1e-06,
    z_margin=8e-07,
    port_extension=2e-06,
    background='sio2',
    run_time_factor=6.0,
    zmin=None,
    zmax=None,
    shutoff=1e-05,
    exclude=None,
    wafer_levels=None,
)
```

Build the base simulation (no sources/monitors) and port geometry.

### `lightsim.sparams.write_sparameters`

```text
write_sparameters(
    component,
    wavelengths=None,
    layer_stack=None,
    material_index=None,
    steps_per_wavelength=20.0,
    port_offset=5e-07,
    port_lateral_margin=1e-06,
    port_z_margin=6e-07,
    mode_index=0,
    num_modes=2,
    device='auto',
    input_ports=None,
    progress=0,
    return_results=False,
    port_symmetry=None,
    **kwargs,
)
```

Compute the S-matrix of a gdsfactory component.

Returns a dict with `wavelengths` (m) and complex arrays `"o2@0,o1@0"` = S(o2 <- o1), following the gdsfactory/gplugins naming convention, plus `"mode_neff"` per port.

One simulation is run per input port. `input_ports` limits the inputs to the ports named. `port_symmetry` uses a symmetry of the device to run fewer simulations for the same result: `"auto"` looks for mirror symmetries of the layout (`detect_port_symmetries`), a dict such as `{"o2": "o3"}` states that a symmetry of the device swaps those ports, and a list of dicts gives several symmetries. Ports that are images of each other are then excited once, and the columns of the others are copied: `S[j, image(i)] = S[image(j), i]`. With a mirror-symmetric mesh, which the automatic mesh is for a mirror-symmetric layout, the copied entries are what their own runs would give. `None` (the default) runs every input.

### `lightsim.sparams.sim_step`

```text
sim_step(sim, axis, position)
```

Local grid spacing at `position` along `axis`.

## `lightsim.gds`

gdsfactory components and layer stacks to structures.

### `lightsim.gds.material_table`

```text
material_table(material_index=None)
```

`MATERIAL_INDEX` with the caller's `material_index` on top (names in lower case).

### `lightsim.gds.material_index_of`

```text
material_index_of(material, idx, where)
```

Refractive index of `material` from the table `idx`.

Raises a `ValueError` that names `where` (the layer-stack level or the role of the material) and the material if the table has no entry. An unknown name used to become index 1 without a word.

### `lightsim.gds.StackLayer`

```text
StackLayer(name, zmin, zmax, material, polygons, mesh_order)
```

One level of a layer stack as lightsim uses it.

**Attributes**

- `name` (str): Name of the level in the layer stack.
- `zmin`, `zmax` (float): Height range of the level (m).
- `material` (str): Material name, in lower case.
- `polygons` (list): Polygons of the level in the component, each a list of (x, y) vertices in metres; empty for a wafer level, which covers the whole region.
- `mesh_order` (int): Painting priority from the layer stack: where patterned levels overlap, the lower value wins.

Fields:

- `name`
- `zmin`: metres
- `zmax`
- `material`
- `polygons`: list of (N, 2) vertex lists in metres; empty => full slab
- `mesh_order`

### `lightsim.gds.stack_layers`

```text
stack_layers(
    component,
    layer_stack=None,
    exclude=('substrate', 'via_contact', 'metal1', 'heater', 'via1', 'metal2', 'via2', 'metal3', 'undercut'),
    wafer_levels=None,
)
```

Collect the layer-stack levels with their polygons (in metres).

Two kinds of level are returned. A **patterned** level has polygons in `component`; it carries them. A **wafer** level covers the whole wafer (buried oxide, cladding); it carries no polygons and becomes a slab over the whole simulation region.

**Parameters**

- `component`: A gdsfactory component.
- `layer_stack`: A gdsfactory `LayerStack`; default: the stack of the active PDK.
- `exclude`: Names of levels to leave out, with or without polygons. Default: the substrate, the undercut, the heater, the vias and the metals.
- `wafer_levels`: Names of levels to treat as wafer levels. A level is also a wafer level if it has `background=True` or if the name of its layer contains `WAFER` (the generic PDK). A wafer level on any other layer must be named here, because it looks like a patterned level that this component does not use.

A level that is neither excluded nor a wafer level and has no polygons in the component is left out. If its height range is not covered by wafer levels, the background medium of the simulation will fill it; a `UserWarning` then names the level, because that is what a cladding on an unrecognised wafer layer looks like.

Returns the list of `StackLayer`, in the order of the stack.

### `lightsim.gds.component_to_structures`

```text
component_to_structures(
    component,
    layer_stack=None,
    material_index=None,
    region_min=None,
    region_max=None,
    exclude=None,
    wafer_levels=None,
)
```

Structures and media of a component for a lightsim `Simulation`.

Returns `(structures, media, layers)`: the structure dicts in painting order, the media they use, and the `StackLayer` list.

Wafer slabs come first, patterned levels after (higher priority), ordered by the stack's mesh_order (lower mesh_order wins, as in gdsfactory/gplugins). `region_min` / `region_max` (metres) bound the wafer slabs in x and y; `exclude` and `wafer_levels` are passed to `stack_layers`. `material_index` maps material names to refractive indices and overrides `MATERIAL_INDEX`.

### `lightsim.gds.port_info`

```text
port_info(component)
```

List of dicts (name, center[m], orientation, width[m]) for optical ports.

### `lightsim.gds.MATERIAL_INDEX`

Refractive index at 1550 nm by lower-case material name, used for the materials of a layer stack. Override entries with the `material_index` argument.

## `lightsim.sparameters`

S-parameters as one object: Touchstone, INTERCONNECT text, gplugins npz and SAX readers and writers, and the phase convention of each.

### `lightsim.sparameters.SParameters`

```text
SParameters(freqs, s, ports, computed=None, meta=None)
```

An S-matrix over frequency, with named ports.

**Parameters**

- `freqs` (array): Frequencies in Hz (any order; stored ascending).
- `s` (array): Complex array of shape `(len(freqs), N, N)`; `s[f, i, j]` is the wave leaving port `i` for a unit wave into port `j`, in lightsim's exp(-iωt) convention.
- `ports` (list): `N` port names, or `(name, mode)` pairs (mode 0 if a name only).
- `computed` (array of bool, optional): `(N, N)`: which entries were simulated (the others are zero, or copies from symmetries). Default: all.
- `meta` (dict, optional): Free-form information carried into the files' comments and sidecar.

### `lightsim.sparameters.SParameters.nports`

Property. Number of ports (port-mode pairs).

### `lightsim.sparameters.SParameters.wavelengths`

Property. Vacuum wavelengths in metres, in the order of `freqs`.

### `lightsim.sparameters.SParameters.port_labels`

Property. `"name@mode"` labels, as in gplugins' keys.

### `lightsim.sparameters.SParameters.index`

```text
SParameters.index(port)
```

Index of a port given as `"o1"`, `"o1@0"` or `("o1", 0)`.

### `lightsim.sparameters.SParameters.allclose`

```text
SParameters.allclose(other, rtol=1e-06, atol=1e-09)
```

Same ports and frequencies, and S-matrices equal within `rtol`/`atol`.

### `lightsim.sparameters.SParameters.from_dict`

```text
SParameters.from_dict(d, wavelength_unit=None, meta=None)
```

From a dict with `wavelengths` and keys `"o2@0,o1@0"` (output first).

`wavelength_unit`: `"m"` (lightsim.sparams), `"um"` (gplugins), or `None` to tell from the values (below 1e-3: metres). Entries the dict does not have are zero and marked not computed. Keys that are not S-parameters (`mode_neff`, ...) are ignored.

### `lightsim.sparameters.SParameters.to_dict`

```text
SParameters.to_dict(wavelength_unit='um')
```

The gplugins dict: `wavelengths` (µm by default, descending frequency order as gplugins writes them: ascending wavelength) and the computed entries.

### `lightsim.sparameters.SParameters.to_npz`

```text
SParameters.to_npz(path)
```

The gplugins `.npz` layout (µm, `"o2@0,o1@0"` keys), which `gplugins.sax.read.model_from_npz` reads.

### `lightsim.sparameters.SParameters.from_npz`

```text
SParameters.from_npz(path)
```

Read a gplugins-layout `.npz` file (wavelengths in µm).

### `lightsim.sparameters.SParameters.to_touchstone`

```text
SParameters.to_touchstone(
    path,
    version='1.1',
    freq_unit='GHz',
    fmt='RI',
    conjugate=True,
    z0=50.0,
    sidecar=True,
)
```

Write a Touchstone file (`.sNp`).

`version`: `"1.1"` (default) or `"2.0"` (adds the `[Version]`, `[Number of Ports]`, `[Number of Frequencies]` and `[Network Data]` keywords; 2.0 orders a 2-port's entries as all other sizes, row by row). `freq_unit`: `"Hz"`, `"kHz"`, `"MHz"`, `"GHz"` or `"THz"` (THz is not in the standard; most readers only take up to GHz). `fmt`: `"RI"`, `"MA"` or `"DB"`. `conjugate`: write the complex conjugate (the exp(+jωt) convention of circuit tools; see the module notes). Port names and modes go into comments and, with `sidecar`, into `<path>.json`. Entries that were not computed are written as 0.

### `lightsim.sparameters.SParameters.from_touchstone`

```text
SParameters.from_touchstone(path, conjugate=None)
```

Read a Touchstone 1.x or 2.0 file (S-parameters, any format and unit).

Port names and modes come from `<path>.json` or from the `! port k = name@mode` comments, else `port1`, `port2`, ... `conjugate`: undo the conjugation of `to_touchstone` (default: as the file's comment or sidecar says, else `True`, since Touchstone files are exp(+jωt)).

### `lightsim.sparameters.SParameters.to_lumerical_dat`

```text
SParameters.to_lumerical_dat(
    path,
    mode_label='TE',
    sides=None,
    include_uncomputed=False,
)
```

Write the INTERCONNECT "Optical N-port S-parameter" text file.

One block per entry, as INTERCONNECT, SiEPIC, SAX and Simphony read it::

```text
('o2','TE',1,'o1',1,'transmission')
(51,3)
<frequency Hz> <|S|> <phase rad>
...
```

Mode numbers are written from 1 (INTERCONNECT's), lightsim's mode 0 is mode 1. The values are lightsim's, not conjugated (INTERCONNECT uses the same exp(-iωt) convention). `sides` (`{"o1": "LEFT", ...}`) adds the port-position header lines INTERCONNECT uses to place the ports; `mode_label` is the polarization label of every block. Entries that were not computed are left out unless `include_uncomputed`.

### `lightsim.sparameters.SParameters.from_lumerical_dat`

```text
SParameters.from_lumerical_dat(path)
```

Read an INTERCONNECT text S-parameter file (`.dat`, `.sparam`).

Port names are kept as written (quotes removed); modes are converted to lightsim's numbering (INTERCONNECT's mode 1 is mode 0). A block with a group delay (a seventh header field) gets the linear phase INTERCONNECT adds. Blocks with sweep parameters are not supported.

### `lightsim.sparameters.SParameters.to_sax`

```text
SParameters.to_sax()
```

A SAX model: `model(wl=1.55)` returns `{("o2", "o1"): S, ...}` interpolated at the wavelengths `wl` (µm), in lightsim's (and SAX's) exp(-iωt) convention. Port keys carry the mode (`"o1@1"`) when the object has modes other than 0. Uses jax.numpy when it is installed, numpy otherwise.

## `lightsim.layout`

Any layout tool: GDSII + technology + ports in, S-parameters or simulation specs out.

### `lightsim.layout.Layout`

```text
Layout(
    cell,
    polygons,
    paths=...,
    labels=...,
    cells=...,
    top_cells=...,
    database_unit_m=1e-09,
    user_unit_m=1e-06,
    path='',
)
```

One flattened cell of a GDSII file (metres).

**Attributes**

- `cell` (str): The cell read.
- `polygons` (dict): `{(layer, datatype): [array (N, 2), ...]}`; paths and boxes are included as polygons.
- `paths` (list): `(layer, datatype, width, pathtype, points (N, 2))`: the centre lines of the paths (pins are paths in SiEPIC's convention).
- `labels` (list): `(text, layer, texttype, x, y)`.
- `cells`, `top_cells` (list of str): All cells of the file and those no other cell references.
- `database_unit_m`, `user_unit_m` (float): The file's units.

### `lightsim.layout.Layout.layer`

```text
Layout.layer(spec)
```

The polygons of one layer, `(layer, datatype)` or `"1/0"`.

### `lightsim.layout.Layout.layers`

Property. The `(layer, datatype)` pairs that have polygons, sorted.

### `lightsim.layout.Layout.bbox`

```text
Layout.bbox(layers=None)
```

`((xmin, ymin), (xmax, ymax))` of the polygons (of `layers` if given).

### `lightsim.layout.Level`

```text
Level(
    name,
    zmin,
    thickness,
    material,
    layers=...,
    sidewall_angle=0.0,
    mesh_order=3,
    subtract=...,
)
```

One level of the stack (µm).

Fields:

- `name`
- `zmin`
- `thickness`
- `material`
- `layers`: [(layer, datatype), ...]; empty: a wafer level
- `sidewall_angle`
- `mesh_order`
- `subtract`: layers a derived level removes (not computed: warned about)

### `lightsim.layout.Level.zmax`

Property. Top of the level (µm).

### `lightsim.layout.Material`

```text
Material(name, index=None, lightsim=None)
```

A material: a constant `index`, or a `lightsim.materials` name (`lightsim`).

### `lightsim.layout.Material.n`

```text
Material.n(wavelength_m)
```

Refractive index at `wavelength_m`: the constant, or the library value.

### `lightsim.layout.Technology`

```text
Technology(levels, materials=..., wavelength_um=1.55, background=None, version=1)
```

Levels, materials, the design wavelength and the background medium.

### `lightsim.layout.Technology.level`

```text
Technology.level(name)
```

The level named `name`.

### `lightsim.layout.Technology.background_material`

```text
Technology.background_material()
```

`background` if set, else the material of the wafer level that holds the middle of the patterned levels, else of the topmost wafer level, else `"air"`.

### `lightsim.layout.Technology.material_index`

```text
Technology.material_index(wavelength_m=None)
```

`{material name: index}` at `wavelength_m` (default the stack's wavelength).

### `lightsim.layout.Technology.stack_layers`

```text
Technology.stack_layers(layout)
```

`lightsim.gds.StackLayer` list for a `Layout` (metres): wafer levels, and the patterned levels that have polygons in it.

### `lightsim.layout.Technology.level_of_layer`

```text
Technology.level_of_layer(layer)
```

Name of the level drawn on `layer` (`None` if none is).

### `lightsim.layout.Technology.to_dict`

```text
Technology.to_dict()
```

The YAML/JSON form (version 1).

### `lightsim.layout.Technology.write`

```text
Technology.write(path)
```

Write the YAML (`.yaml`/`.yml`, needs PyYAML) or JSON form.

### `lightsim.layout.pins`

```text
pins(layout, pin_layer, label_layer=None, port_layer=None, tol=5e-07)
```

Ports from pins drawn on `pin_layer` and named by text labels.

A pin is a two-point path (SiEPIC's convention: from inside the device to outside; the port is at its midpoint, as wide as the path) or a box (the port is at its centre, as wide as its longer side, facing away from the centre of the layout along its shorter side). The name is the text on `label_layer` (default the pin layer, any texttype) nearest the port, within `tol` metres (default 0.5 µm). `port_layer` is the waveguide layer recorded with the port.

### `lightsim.layout.read_gds`

```text
read_gds(path, cell=None)
```

Read and flatten `cell` of a GDSII file (default: its only top cell).

No gdsfactory, gdstk or KLayout is needed. For an OASIS file, convert it first, for example with KLayout: `klayout -zz -rd input=chip.oas -rd output=chip.gds -r convert.py` or, interactively, File > Save As with the GDS2 format.

### `lightsim.layout.read_lyp`

```text
read_lyp(path)
```

Layer names of a KLayout layer-properties file: `{(layer, datatype): name}`.

Groups are walked; entries with wildcards (`*/*`) are skipped.

### `lightsim.layout.read_port_file`

```text
read_port_file(path)
```

Ports from a JSON, YAML or CSV port file (see the module notes).

### `lightsim.layout.read_ports`

```text
read_ports(ports, layout=None, gds_path=None, cell=None)
```

Ports from a port file, a list of dicts (port-file keys, µm), or a convention name (`"siepic"`, `"gdsfactory"`, or a `{"convention": "pins", ...}` dict) applied to `layout`.

### `lightsim.layout.read_technology`

```text
read_technology(path)
```

Read a technology file: `.toml` (gamma's stack), `.yaml`/`.yml` or `.json` (lightsim's form).

### `lightsim.layout.siepic`

```text
siepic(layout, tol=5e-07)
```

Ports from SiEPIC pins (PinRec 1/10 paths with labels; waveguide Si 1/0).

### `lightsim.layout.technology_from_gdsfactory`

```text
technology_from_gdsfactory(
    layer_stack=None,
    material_index=None,
    wavelength_um=1.55,
    exclude=None,
)
```

Convert a gdsfactory `LayerStack` (default: the active PDK's).

Levels on the wafer layer or with `background=True` become wafer levels; levels on a derived layer are left out (with a warning); the levels lightsim skips by default (substrate, metals, vias) are left out unless `exclude` says otherwise. Materials get the index from `material_index` or `lightsim.gds.MATERIAL_INDEX`.

### `lightsim.layout.write_port_file`

```text
write_port_file(ports, path)
```

Write ports (lightsim's dicts, metres) as a version-1 JSON port file (µm).

### `lightsim.layout.simulation`

```text
simulation(gds, technology, ports, sources=None, **kwargs)
```

The lightsim simulations of a layout, one per source port and mode.

Returns `{"o1@0": Simulation, ...}`, each with its mode source and the mode monitors of all ports (the port mode profiles are in the source and monitor entries of `sim.to_json()`). `sources` limits the sources to the ports (`"o1"`) or port modes (`"o1@1"`) listed. The other arguments are those of `sparameters`; nothing is run.

### `lightsim.layout.write_specs`

```text
write_specs(sims, out)
```

Write the specs of `simulation` as lightsim JSON: `out` itself for one simulation, else `<stem>_<port>@<mode>.json` next to it.

### `lightsim.layout.sparameters`

```text
sparameters(gds, technology, ports, device='auto', progress=0, **kwargs)
```

S-parameters of a cell of a GDSII file, as an `SParameters`.

**Parameters**

- `gds`: Path of the GDSII file (or a `Layout`).
- `technology`: Technology file (`.toml`, `.yaml`, `.json`) or `Technology`.
- `ports`: Port file (`.json`, `.yaml`, `.csv`), a list of port dicts (port-file keys, µm), `"siepic"`, `"gdsfactory"` or `{"convention": "pins", "pin_layer": ..., "label_layer": ..., "port_layer": ...}`.
- `cell`: Cell to simulate (default: the only top cell).
- `wavelengths`: Array of vacuum wavelengths in metres (default 1.50-1.60 µm, 21 points).
- `is_3d`: `True` (default) for 3-D; `False` for the 2-D model chosen by `effective_index` (effective-index levels, or bulk indices) and `polarization`.
- `resolution`, `steps_per_wavelength`: A uniform mesh of `resolution` cells per µm, or (default) a graded mesh of `steps_per_wavelength` steps per wavelength in each material.
- `margin`, `z_margin`: Distance from the layout to the absorber in x/y and in z (µm).
- `port_margin`, `port_source_offset`, `port_monitor_offset`, `distance_source_to_monitors`: Mode windows and the placement of sources and monitors, as in `lightsim.gplugins.write_sparameters_lightsim` (µm; offsets positive outwards).
- `decay_by`: The runs stop when the field energy has decayed by this factor.
- `background`: The medium around the stack (default: from the technology).
- `device`: `"auto"`, `"cpu"`, `"cuda:0"`, `"cuda:0,1"` or a list of devices over which the sources run concurrently.

## `lightsim.gplugins`

The gplugins interface: `write_sparameters_meep`-compatible S-parameters, batches, grating couplers and SAX models; it also re-exports the mode-solver front of `lightsim.gplugins_modes`.

### `lightsim.gplugins.IgnoredArgumentWarning`

```text
IgnoredArgumentWarning(...)
```

An argument of the gplugins.gmeep interface that lightsim does not use.

### `lightsim.gplugins.get_sparameters_path`

```text
get_sparameters_path(component, dirpath=None, **kwargs)
```

S-parameter `.npz` path for `component` and the simulation settings.

The file name is a hash of the component's layout and of the settings, as in gplugins (`get_sparameters_path_meep` and its siblings), with `tool="lightsim"`. gplugins' own function is used when gplugins is installed; otherwise an identical built-in gives the same path.

### `lightsim.gplugins.get_simulation`

```text
get_simulation(component, port_source_name='o1', port_source_mode=0, **kwargs)
```

The lightsim simulation of one source, with its sources and monitors.

The counterpart of `gplugins.gmeep.get_simulation`; the arguments are those of `write_sparameters_lightsim`. Look at it with `sim.view()` or run it with `sim.run()`.

### `lightsim.gplugins.write_sparameters_lightsim`

```text
write_sparameters_lightsim(
    component,
    port_source_names=None,
    port_source_modes=None,
    port_modes=None,
    port_symmetries=None,
    resolution=30,
    wavelength_start=1.5,
    wavelength_stop=1.6,
    wavelength_points=50,
    dirpath=None,
    layer_stack=None,
    port_margin=2,
    port_monitor_offset=-0.1,
    port_source_offset=-0.1,
    filepath=None,
    overwrite=False,
    animate=False,
    animate_center=None,
    animate_size=None,
    lazy_parallelism=False,
    run=True,
    dispersive=False,
    xmargin=0,
    ymargin=0,
    zmargin=0,
    xmargin_left=0,
    xmargin_right=0,
    ymargin_top=0,
    ymargin_bot=0,
    zmargin_top=0,
    zmargin_bot=0,
    decay_by=0.001,
    is_3d=False,
    z=0,
    plot_args=None,
    only_return_filepath_sim_settings=False,
    verbosity=0,
    padding_layers=('PADDING',),
    *,
    device='auto',
    effective_index=True,
    polarization=None,
    steps_per_wavelength=None,
    **settings,
)
```

S-parameters of a gdsfactory component with lightsim, the gplugins way.

Same arguments, units, defaults, result and cache files as `gplugins.gmeep.write_sparameters_meep`: one simulation per source port and mode, a mode source `port_source_offset` outside each source port (negative: inside the component), mode monitors `port_monitor_offset` outside every port and `distance_source_to_monitors` (0.2 um) behind the source on the source port, and `S = a_out / a_in` of the projections on the port modes. Lengths in micrometres.

**Parameters**

- `component`: gdsfactory component (or the name of one).
- `port_source_names`: Ports to excite (default: all).
- `port_source_modes`: `{port: [mode, ...]}` to excite (default: mode 0 of each).
- `port_modes`: Modes measured at every port (default `[0]`).
- `port_symmetries`: gplugins' format, `{"o2@0,o1@0": ["o1@0,o2@0"]}`: the entries listed copy the key. A source whose whole column follows from the runs before it by these copies is not run.
- `resolution`: Cells per micrometre of the uniform mesh.
- `wavelength_start`, `wavelength_stop`, `wavelength_points`: The band (um).
- `dirpath`, `filepath`, `overwrite`: Where the `.npz` goes (default: a name hashed from the component and the settings, in `dirpath`), and whether an existing file is computed again.
- `layer_stack`: Default: the active PDK's.
- `port_margin`: The mode windows extend this far beside the port, at most half the distance to a neighbouring port.
- `port_monitor_offset`, `port_source_offset`: um, positive outwards.
- `run`: `False` returns the simulation of the first source instead.
- `dispersive`: 3-D: the materials of `lightsim.materials` (`si`, `sio2`, `sin`, ...) become pole fits over the band instead of constant indices.
- `xmargin`, `ymargin`, `zmargin`, `xmargin_left`, `...`: Distance from the component to the absorber. A z margin of 0 is replaced by 0.8 um.
- `decay_by`: The runs stop when the field energy has decayed by this factor from its peak (lightsim's `shutoff`).
- `is_3d`: `False` runs the 2-D model chosen by `effective_index`.
- `only_return_filepath_sim_settings`: Return the path of the `.yml` settings file only.
- `verbosity`: Above 0 prints the progress of each run.
- `device`: lightsim only. `"auto"`, `"cpu"`, `"cuda:0"`, `"cuda:0,1"` (each run split over two GPUs) or a list of device strings, over which the sources run concurrently.
- `effective_index`: lightsim only, for `is_3d=False`. `True`: the 2.5-D effective-index model (each level gets the effective index of its slab mode). `False`: the plain 2-D cut with the bulk indices, which is what Meep's 2-D run simulates.
- `polarization`: lightsim only, 2-D. `"te"` (E in the chip plane, the default) or `"tm"`. With `effective_index=False` and no polarization the mode of highest effective index is used, as Meep's `eig_band=1` with no parity does.
- `steps_per_wavelength`: lightsim only. A graded mesh with this many steps per wavelength in each material instead of `resolution`.
- `settings`: Meep's `get_simulation` and `meep.Simulation` keywords. `extend_ports_length`, `tpml`, `clad_material`, `distance_source_to_monitors`, `dfcen`, `material_name_to_meep` and `exclude_layers` are used, the others are ignored with a warning, unknown names raise.

**Returns**

dict

```text
``{"wavelengths": um, "o1@0,o1@0": complex array, ...}``.
```

### `lightsim.gplugins.write_sparameters_lightsim_batch`

```text
write_sparameters_lightsim_batch(
    jobs,
    cores_per_run=None,
    total_cores=None,
    temp_dir=None,
    delete_temp_files=None,
    dirpath=None,
    layer_stack=None,
    *,
    devices=None,
    **kwargs,
)
```

S-parameters of several components or settings, spread over the GPUs.

The counterpart of `gplugins.gmeep.write_sparameters_meep_batch`: `jobs` is a list of `write_sparameters_lightsim` keyword dictionaries (each with a `component`), `kwargs` are common to all of them, and the `.npz` paths are returned in the order of `jobs`. Jobs whose file exists are not run again unless `overwrite=True`.

`devices` (lightsim only) lists the devices to use, one job at a time on each: by default every GPU (`["cuda:0", "cuda:1"]`), else the CPU. `cores_per_run`, `total_cores`, `temp_dir` and `delete_temp_files` (Meep's MPI pool) are ignored.

### `lightsim.gplugins.write_sparameters_lightsim_mpi`

```text
write_sparameters_lightsim_mpi(
    component,
    layer_stack=None,
    cores=None,
    filepath=None,
    dirpath=None,
    temp_dir=None,
    temp_file_str=None,
    live_output=False,
    overwrite=False,
    wait_to_finish=True,
    **kwargs,
)
```

The counterpart of `gplugins.gmeep.write_sparameters_meep_mpi`: simulates and returns the `.npz` path.

lightsim needs no MPI: the run happens in this process, on the device given by `device`. `cores`, `temp_dir`, `temp_file_str`, `live_output` and `wait_to_finish=False` are ignored.

### `lightsim.gplugins.write_sparameters_grating_lightsim`

```text
write_sparameters_grating_lightsim(
    plot=False,
    plot_contour=False,
    animate=False,
    overwrite=False,
    dirpath=None,
    decay_by=0.001,
    verbosity=0,
    *,
    device='auto',
    run=True,
    period=0.66,
    fill_factor=0.5,
    n_periods=30,
    widths=None,
    gaps=None,
    fiber_angle_deg=20.0,
    fiber_xposition=1.0,
    fiber_core_diameter=10.4,
    fiber_numerical_aperture=0.14,
    fiber_clad_material=1.44,
    nwg=3.48,
    nslab=None,
    clad_material=1.44,
    nbox=1.44,
    nsubstrate=3.48,
    pml_thickness=1.0,
    substrate_thickness=1.0,
    box_thickness=2.0,
    core_thickness=0.22,
    slab_thickness=0.15,
    top_clad_thickness=2.0,
    air_gap_thickness=1.0,
    fiber_thickness=2.0,
    resolution=64,
    wavelength_start=1.4,
    wavelength_stop=1.7,
    wavelength_points=150,
    eps_averaging=False,
    fiber_port_y_offset_from_air=1,
    waveguide_port_x_offset_from_grating_start=10,
    fiber_port_x_size=None,
    xmargin=10.0,
)
```

S-parameters of a 1-D grating coupler and a fibre, in a 2-D side view.

The counterpart of `gplugins.gmeep.write_sparameters_grating`, with the same geometry arguments (lengths in um, materials as refractive indices) and result: `o1` is the waveguide, `o2` the fibre; `"o2@0,o1@0"` is the coupling from the waveguide mode into the fibre mode. It is cached as `fiber_<hash>.npz` in `dirpath`.

The structure is the one gplugins builds (substrate, buried oxide, etched grating, cladding, air gap, a tilted fibre core and cladding), run as a lightsim 2-D simulation invariant along y with E along y (the slab TE mode). The waveguide mode is solved on the grid; the fibre mode is the Gaussian beam of waist `fiber_core_diameter / 2` tilted by `fiber_angle_deg` (a fit of the fibre's fundamental mode). Meep's `eps_averaging` and `animate` are ignored; `plot` returns the simulation without running it. With `run=False` likewise.

### `lightsim.gplugins.write_sparameters_grating_lightsim_batch`

```text
write_sparameters_grating_lightsim_batch(
    instances,
    cores_per_instance=None,
    total_cores=None,
    temp_dir=None,
    delete_temp_files=None,
    verbosity=False,
    *,
    devices=None,
    **kwargs,
)
```

Several `write_sparameters_grating_lightsim` runs spread over the GPUs (`gplugins.gmeep.write_sparameters_grating_batch`); returns the S-parameter dicts in the order of `instances`. Meep's process options are ignored.

### `lightsim.gplugins.model_from_npz`

```text
model_from_npz(filepath, xkey='wavelengths', xunits=1)
```

SAX model of a component from its `.npz` file.

`gplugins.sax.read.model_from_npz` reads lightsim's files as they are; this calls it when gplugins is installed. Without gplugins the same model is built here (it needs `sax` and `jax`): a function of the wavelength (um) that returns the S-parameters, interpolated, as a SAX dict `{("o1", "o2"): ...}`.

## `lightsim.eim`

2.5-D effective-index simulations.

### `lightsim.eim.slab_modes`

```text
slab_modes(indices, thicknesses, wavelength, polarization='te', samples=4000)
```

Guided-mode effective indices of a planar multilayer, highest first.

`indices` = [n_bottom_halfspace, n_1, ..., n_m, n_top_halfspace], `thicknesses` = [d_1, ..., d_m] (metres). Lossless, isotropic layers. `polarization`: `"te"` (E parallel to the layers) or `"tm"`.

### `lightsim.eim.slab_neff`

```text
slab_neff(
    n_core,
    thickness,
    wavelength,
    n_substrate,
    n_cladding=None,
    polarization='te',
    mode=0,
)
```

Effective index of mode `mode` of a three-layer slab waveguide (substrate / core of `thickness` / cladding). Raises if it is cut off.

### `lightsim.eim.effective_indices`

```text
effective_indices(
    component,
    wavelength,
    layer_stack=None,
    material_index=None,
    polarization='te',
    mode=0,
    background='sio2',
    exclude=None,
    wafer_levels=None,
)
```

Slab effective index of every patterned layer of `component`.

Returns `(neff, n_background, layers)`: `neff` maps the layer-stack level name to the effective index of slab mode `mode` of the vertical stack "wafer layers + that level", `n_background` is the index used where no patterned layer is present (the wafer material at the height of the patterned layers, i.e. the lateral cladding), `layers` the `lightsim.gds.StackLayer` list. Levels whose slab mode is cut off get the background index.

### `lightsim.eim.effective_index_simulation`

```text
effective_index_simulation(
    component,
    wavelengths,
    layer_stack=None,
    material_index=None,
    polarization='te',
    slab_mode=0,
    steps_per_wavelength=20.0,
    xy_margin=1e-06,
    port_extension=2e-06,
    background='sio2',
    run_time_factor=6.0,
    shutoff=1e-05,
    z_cell=1e-06,
    pml_layers=12,
    exclude=None,
    wafer_levels=None,
    level_indices=None,
    background_index=None,
)
```

Build the 2-D effective-index model of a gdsfactory component.

Every polygon of a patterned layer-stack level becomes a z-invariant region with the slab effective index of that level (see `effective_indices`); regions of higher effective index take precedence where levels overlap (e.g. a rib over its partially etched slab). The result is a 2-D simulation (`dimensions=2`, invariant along z, `polarization` = the slab mode family) with PML in x / y. The slab index is evaluated at the mean wavelength.

`z_cell` is the thickness of the slab the 2-D model stands for (`None`: one in-plane step). The fields do not depend on z, so it only scales fluxes and mode powers; S-parameters do not depend on it.

`level_indices` (level name -> index) and `background_index` replace the slab effective indices of the levels listed and the index between them. With the bulk indices of the materials the model is the plain 2-D cross-section of the layout, without the vertical confinement the effective index stands for.

Returns `(sim, ports, info)` with no sources / monitors; `info` holds `neff` (per level), `n_background`, `polarization` and `z` (the z range of the cell).

### `lightsim.eim.solve_port_mode`

```text
solve_port_mode(
    sim,
    axis,
    position,
    min,
    max,
    freq,
    polarization='te',
    mode_index=0,
    num_modes=None,
)
```

Modes of a 2-D simulation on the line normal to `axis` at `position` (a port "plane" of a 2-D simulation is a line: one in-plane axis and the slab along the invariant axis).

The 1-D eigenproblem is discretized on the Yee positions of the line with the permittivity the engine uses, like `lightsim.solve_modes` does in 3-D (whose 2-D operators assume closed walls and so cannot represent a periodic single cell):

- `"te"`: E along the line (in the plane), H along the invariant axis -- beta^2 e = (k0^2 + D_f eps_n^-1 D_b) eps_t e,
- `"tm"`: E along the invariant axis -- beta^2 e = (k0^2 eps_w + D_b D_f) e.

`sim` is a 2-D `lightsim.Simulation` (any invariant axis) or a hand-made one-cell grid with periodic z. `min` / `max` may have two (in-plane) components for a 2-D simulation.

Returns the `lightsim.Mode` number `mode_index` (by descending n_eff), or the list of the first `num_modes` modes if that is given. Profiles are normalized to unit power over the thickness of the slab.

### `lightsim.eim.write_sparameters_2d`

```text
write_sparameters_2d(
    component,
    wavelengths=None,
    layer_stack=None,
    material_index=None,
    polarization='te',
    steps_per_wavelength=20.0,
    port_offset=5e-07,
    port_lateral_margin=1e-06,
    mode_index=0,
    device='auto',
    input_ports=None,
    progress=0,
    return_results=False,
    port_symmetry=None,
    **kwargs,
)
```

S-matrix of a gdsfactory component from the 2.5-D effective-index model.

Same workflow and output as `lightsim.sparams.write_sparameters` (`"o2@0,o1@0"` = S(o2 <- o1) over `wavelengths`, `"mode_neff"` per port), plus `"slab_neff"` and `"n_background"`. `polarization` selects the slab mode family and the matching 2-D field set; extra keyword arguments go to `effective_index_simulation`. `port_symmetry` is as in `lightsim.sparams.write_sparameters`.

## `lightsim.adjoint`

Adjoint gradient of a mode power.

### `lightsim.adjoint.AdjointResult`

```text
AdjointResult(fom, a_plus, grad, volumes, da=None)
```

Result of `mode_power_gradient`.

**Attributes**

- `fom` (float): The figure of merit `abs(a_plus)**2`: power in the output mode per unit source spectrum.
- `a_plus` (complex): Amplitude of the output mode per unit source spectrum.
- `grad` (dict): Per electric component (`"Ex"`, `"Ey"`, `"Ez"`): `(origin, array)` with d(fom)/d(eps_r) of every cell of the design box; `origin` is the grid index of the first cell.
- `volumes` (dict): Per component, the cell volumes (m^3), same shapes.
- `da` (dict): Per component, `(origin, array)` with the complex d(a_plus)/d(eps_r) of every cell.

### `lightsim.adjoint.AdjointResult.amplitude_derivative_in_box`

```text
AdjointResult.amplitude_derivative_in_box(sim, bmin, bmax)
```

Complex d(a_+)/d(eps_r) for a uniform eps change of the block.

### `lightsim.adjoint.AdjointResult.total_gradient_in_box`

```text
AdjointResult.total_gradient_in_box(sim, bmin, bmax)
```

Derivative of F w.r.t. a uniform eps_r change of the block [bmin, bmax]: sum over E components of dF/d(eps_r) weighted by the voxel fill fraction (the sub-pixel-smoothed eps of a partially filled voxel changes by f \* d(eps), for both field orientations).

### `lightsim.adjoint.amplitude_sensitivity`

```text
amplitude_sensitivity(e_fwd, e_adj, volume, freq)
```

Complex d(a_+)/d(eps_r) of every Yee voxel: (i w eps0 / 4) E_adj E_fwd dV.

`e_fwd` is the forward field per unit incident amplitude, `e_adj` the field of a unit-amplitude mode launched backwards from the output plane, both at `freq` and at the positions of one E component. Shared by `mode_power_gradient` and `lightsim.invdes`.

### `lightsim.adjoint.check_gradient_media`

```text
check_gradient_media(sim)
```

Raise `ValueError` if `sim` uses a medium that no gradient engine models: a delayed (Raman) chi3 response, a four-level gain medium, a gyrotropic pole (not reciprocal) or a director field. Used by `mode_power_gradient` and `lightsim.invdes.SMatrixProblem` (all three engines); the Rust engines check the same media. A fixed-angle run (a plane wave with `fixed_angle=True`) is refused too: no gradient engine models its update.

### `lightsim.adjoint.mode_power_gradient`

```text
mode_power_gradient(sim, design_min, design_max, output, freq, device='auto')
```

Gradient of F = |a_+(freq)|^2 w.r.t. eps_r in the design box.

`sim` must already contain the forward source. `output` describes the output plane: dict(axis, position, min, max, profile, direction) where `direction` is the propagation direction of the outgoing mode ("+" or "-").

### `lightsim.adjoint.mode_power`

```text
mode_power(sim, output, freq, device='auto')
```

F = |a_+|^2 per unit source spectrum (for finite-difference checks).

### `lightsim.adjoint.EPS0`

Vacuum permittivity in F/m.

## `lightsim.invdes`

Inverse design: S-matrix gradients, parametrisation, optimiser.

### `lightsim.invdes.SMatrixProblem`

```text
SMatrixProblem(
    sim,
    grids,
    inputs,
    outputs,
    freqs,
    devices='cpu',
    fwidth=None,
    normalize='monitor',
    components=('Ex', 'Ey', 'Ez'),
    progress=0,
    monitors=(),
    engine=None,
)
```

S-matrix of a device whose design region is a set of index grids.

`S` has the axes (outputs, inputs, frequencies): `S[m, n, k]` is the amplitude in output `m` for a unit excitation of input `n` at `freqs[k]`, an array of shape `[M, N, K]`.

**Parameters**

- `sim`: the base `lightsim.Simulation` (fixed structures, mesh, boundaries, run time). Its sources and monitors are ignored.
- `grids`: list of `DesignGrid` (or one). They are appended after the structures of `sim`, so the design covers what lies under it; no structure of `sim` may be meant to cover the design.
- `inputs`: excitations: `Port` (mode 0), `(Port, mode_index)` or `Excitation`.
- `outputs`: output channels: `Port` or `(Port, mode_index)`.
- `freqs`: the K frequencies (Hz).
- `devices`: a device string or a list of them. The N + M runs of one `solve` are independent and are dispatched concurrently, one at a time per list entry (`["cuda:0", "cuda:1"]`; with `engine="python"` an entry may itself be a multi-GPU string such as `"cuda:0,1"`, which splits each run over those GPUs and gives the one-GPU S and gradient bit for bit; the native engine runs each simulation on one device).
- `fwidth`: bandwidth of the Gaussian pulses (default: the span of `freqs`, at least 10 % of the centre frequency).
- `monitors`: extra DFT field monitors (`lightsim.DFTMonitor` dicts) added to every forward run (recorded at `freqs`); their fields, per unit excitation, are returned in `SMatrixSolution.fields[n][name][component]` (field maps of the final design: by linearity any input vector is a weighted sum).
- `normalize`: `"monitor"` (default) measures the amplitude launched by every mode source with two mode monitors just downstream and refers it to the source plane, so S is exactly "per unit incident mode amplitude"; `"source"` divides by the source signal spectrum only.
- `engine`: `"reverse"` (one forward run and one reverse pass through the time loop per input: the exact derivative of the discrete scheme; see `lightsim.autodiff.ReverseSMatrix`), `"python"` (every run driven from Python, products in NumPy) or `"native"` (the persistent Rust session of `lightsim.autodiff`: buffers kept on the device across `solve` calls, adjoint product and transposed rasterisation in the engine). Default: `$LIGHTSIM_INVDES_ENGINE` or `"python"`.

### `lightsim.invdes.SMatrixProblem.shape`

Property. `(M, N, K)`: the numbers of outputs, inputs and frequencies, the shape of the S-matrix that `solve` returns.

### `lightsim.invdes.SMatrixProblem.simulation`

```text
SMatrixProblem.simulation(eps)
```

The simulation (no sources, no monitors) with the design grids holding the permittivity arrays `eps` (one per grid).

### `lightsim.invdes.SMatrixProblem.solve`

```text
SMatrixProblem.solve(eps, adjoint=True)
```

Run the N forward (and, with `adjoint`, the M adjoint) simulations for the grid permittivities `eps` (one array per grid).

### `lightsim.invdes.SMatrixProblem.solve_combined`

```text
SMatrixProblem.solve_combined(eps, weights)
```

P forward runs, run `p` exciting every input `n` at once with complex amplitude `weights[p, n]` (one weight per input for all frequencies). Returns a `CombinedSolution` with the launched input vectors `X`, the outputs `Y = S X` and `vjp`.

With random probe vectors (`E[x x^H] = I`) this estimates any objective of the form `sum_n |(S - T) e_n|^2 = E |(S - T) x|^2` and its gradient from `2 P` simulations per estimate, whatever the number of ports (`lightsim.invdes.ProbeDesign`). Python engine only; inputs and outputs must be distinct waveguide ports.

### `lightsim.invdes.SMatrixProblem.native_session`

```text
SMatrixProblem.native_session()
```

The `lightsim.autodiff.GradientSession` behind `engine="native"` (built on first use).

### `lightsim.invdes.SMatrixProblem.reverse_smatrix`

```text
SMatrixProblem.reverse_smatrix()
```

The `lightsim.autodiff.ReverseSMatrix` behind `engine="reverse"` (built on first use).

### `lightsim.invdes.SMatrixProblem.yee_permittivity`

```text
SMatrixProblem.yee_permittivity(eps)
```

What the solver uses: the permittivity at the Yee positions of every design monitor box interpolated from the grids (zero where the grid does not own the voxel). Needs one earlier `solve` with `engine="python"`; the native session reports the same through `native_session().yee_permittivity()`.

### `lightsim.invdes.CompositeProblem`

```text
CompositeProblem(blocks, network, inputs, outputs, devices='cpu', reflections=True)
```

S-matrix of a network of blocks, each solved by its own FDTD problem.

**Parameters**

- `blocks`: list of `Block`.
- `network`: the `lightsim.invdes.Network` that joins them (its `ports` must equal the blocks' port counts).
- `inputs`, `outputs`: indices into the network's external ports; `S` is `T[outputs][:, inputs]`.
- `devices`: a device string or a list of them; the blocks are solved concurrently, one block at a time per entry (`["cuda:0", "cuda:1"]`). Each block's own `devices` setting is replaced.
- `reflections`: keep the internal reflections of the network (the exact composition of the given block S-matrices). `False` drops the entries between ports on the same side (`network.sides`), which is the zeroth order of the multiple-reflection series.

### `lightsim.invdes.CompositeProblem.shape`

Property. `(outputs, inputs, frequencies)`: the shape of the composed S-matrix.

### `lightsim.invdes.CompositeProblem.block_eps`

```text
CompositeProblem.block_eps(eps)
```

Split the concatenated permittivity list into one list per block.

### `lightsim.invdes.CompositeProblem.solve`

```text
CompositeProblem.solve(eps, adjoint=True)
```

Solve every designed block (forward runs, and adjoint runs with `adjoint`), compose the network and return the solution.

### `lightsim.invdes.CompositeSolution`

```text
CompositeSolution(
    S,
    freqs,
    stats,
    T=None,
    block_S=None,
    _problem=None,
    _net=None,
    _sols=None,
)
```

Result of `CompositeProblem.solve`: `S[m, n, k]` between the chosen external outputs and inputs, and `vjp`.

Fields:

- `S`
- `freqs`
- `stats`
- `T`: all external ports, `[E, E, K]`
- `block_S`: per block `[P, P, K]` as used in the network

### `lightsim.invdes.CompositeSolution.vjp`

```text
CompositeSolution.vjp(g)
```

dL/d(eps) on the samples of every design grid (the blocks' grids, concatenated in block order) for the upstream gradient `g` of the shape of `S`.

### `lightsim.invdes.CompositeDesign`

```text
CompositeDesign(designs)
```

Several `lightsim.invdes.Design` objects seen as one: the density is the list (or, when all have the same shape, the stacked array `[B, nx, ny]`) of the blocks' densities, `eps` concatenates their permittivity arrays in the order of `CompositeProblem.grids`, and `vjp` splits the gradient back. It has the methods `lightsim.invdes.InverseDesign` and `lightsim.invdes.optimize` use.

### `lightsim.invdes.CompositeDesign.grids`

```text
CompositeDesign.grids()
```

The design grids of all blocks, block after block, in the order of `eps` and of the gradients `vjp` takes.

### `lightsim.invdes.CompositeDesign.random`

```text
CompositeDesign.random(seed=0, mean=0.5, amplitude=0.0)
```

A starting density for every block (see `lightsim.invdes.Design.random`); block `i` uses `seed + i`.

### `lightsim.invdes.CompositeDesign.eps`

```text
CompositeDesign.eps(rho, beta, eta=None)
```

The permittivity arrays of all blocks for the density `rho`, at projection strength `beta`, in the order of `grids`.

### `lightsim.invdes.CompositeDesign.vjp`

```text
CompositeDesign.vjp(rho, beta, grads, eta=None)
```

Gradient with respect to the density from the gradients with respect to the permittivity arrays (`grads`, in the order of `grids`): each block's part goes through its own filter and projection.

### `lightsim.invdes.CompositeDesign.projected`

```text
CompositeDesign.projected(rho, beta, eta=None)
```

The filtered and projected density of every block.

### `lightsim.invdes.CompositeDesign.binarization`

```text
CompositeDesign.binarization(rho, beta, eta=None)
```

Mean over the blocks of their binarisation (1 for a design of only the two materials).

### `lightsim.invdes.Block`

```text
Block(problem, in_ports=None, out_ports=None, ports=None, name='')
```

One block of a `CompositeProblem`.

`problem` is an `lightsim.invdes.SMatrixProblem` of the block alone, or a fixed S-matrix `[P, P, K]` (a block without design, such as a straight waveguide, computed once). `in_ports` / `out_ports` are the block port numbers (0 ... P-1, the numbering of the network) of the problem's inputs and outputs, in order; entries of the block S-matrix that no input-output pair measures are zero in the network (reflections left out, for instance). `ports` is P (default: the largest port number plus one).

### `lightsim.invdes.Block.fixed`

Property. True for a block given as a constant S-matrix (an array), which has no design and is not simulated.

### `lightsim.invdes.Block.unit`

Property. The object that runs the simulations: the problem itself, or the group that solves several blocks at once (see `_is_member`).

### `lightsim.invdes.Block.grids`

Property. The design grids of the block's problem (none for a fixed block).

### `lightsim.invdes.Network`

```text
Network(ports, connections, external, sides=None)
```

Topology of a scattering-matrix network.

**Parameters**

- `ports`: number of ports of every block, `[P_0, P_1, ...]`.
- `connections`: pairs `((b, p), (b2, p2))` of joined ports.
- `external`: the external ports `(b, p)` in the order of the rows and columns of the network S-matrix. Every port is either connected once or external.
- `sides`: optional, per block, a sequence of `P_b` labels (for example `"in"` / `"out"`); with `reflections=False` the S-matrix entries between two ports of the same label are dropped.

### `lightsim.invdes.Network.n_external`

Property. Number of external ports: the ports left unconnected, in the order they were declared.

### `lightsim.invdes.Network.dense`

```text
Network.dense(S_blocks, reflections=True)
```

All block ports in one array `S[k, out, in]` (block diagonal).

### `lightsim.invdes.Network.solve`

```text
Network.solve(S_blocks, reflections=True, store=True, backend='auto')
```

Network S-matrix `T[e, e2, k]` of the block S-matrices `S_blocks` (list of `[P_b, P_b, K]`), by `lightsim.circuit.solve_matrix` (the solver of `lightsim.circuit.circuit`). With `store` the factors needed by `NetworkSolution.vjp` are kept.

### `lightsim.invdes.Network.netlist`

```text
Network.netlist(S_blocks, names=None)
```

The same network as a `lightsim.circuit.solve_network` problem: `(sdicts, pairs, ports)` with instance `"b{i}"` (or `names[i]`), ports `"p0", "p1", ...` in block order and external ports `"e0", "e1", ...` in the order of `ext`. Block `S[p, q, k]` becomes the SDict entry `("p{q}", "p{p}")`.

### `lightsim.invdes.NetworkSolution`

```text
NetworkSolution(T, _net, _B, _R, _reflections)
```

Result of `Network.solve`: `T[e, e2, k]` and `vjp`.

### `lightsim.invdes.NetworkSolution.vjp`

```text
NetworkSolution.vjp(g)
```

Gradients with respect to the block S-matrices (list of `[P_b, P_b, K]`, convention of the module docstring) for the upstream gradient `g` with the shape of `T`.

### `lightsim.invdes.chain`

```text
chain(transmissions)
```

Feed-forward composition of a sequence of transmission matrices `[T_1, T_2, ...]` (each `[n_out, n_in, K]`, the outputs of one feeding the inputs of the next): `T = T_D ... T_2 T_1` and the pullback of an upstream gradient to every factor.

Returns `(T, pullback)`; `pullback(g)` gives the list of gradients in the convention of the module docstring. This is `Network.solve` with `reflections=False` for a cascade, without the sparse machinery: `O(D n^3)` per frequency for D dense factors.

### `lightsim.invdes.SMatrixSolution`

```text
SMatrixSolution(S, freqs, stats, _problem=None, _fwd=None, _adj=None, fields=None)
```

Result of `SMatrixProblem.solve`.

Fields:

- `S`: complex [M, N, K]
- `freqs`
- `stats`
- `fields`: per excitation: {monitor name: {component: complex array, "origin": {...}}}

### `lightsim.invdes.SMatrixSolution.yee_vjp`

```text
SMatrixSolution.yee_vjp(g)
```

dL/d(eps_r) on the Yee voxels of every design-grid monitor box for the upstream gradient `g` (see the module docstring): list over grids of `{comp: (origin, real array [i, j, k])}`.

### `lightsim.invdes.SMatrixSolution.vjp`

```text
SMatrixSolution.vjp(g)
```

dL/d(eps) on the samples of every design grid (list of arrays of shape `grid.dims`) for the upstream gradient `g`.

### `lightsim.invdes.SMatrixSolution.jacobian`

```text
SMatrixSolution.jacobian(m, n, k)
```

Complex dS[m, n, k]/d(eps) on the samples of every design grid.

### `lightsim.invdes.NativeSolution`

```text
NativeSolution(problem, S, stats, token, eps)
```

Result of `SMatrixProblem.solve` with `engine="native"`: the same interface as `SMatrixSolution`, backed by the fields the `lightsim.autodiff.GradientSession` keeps on the device.

The session holds one design at a time. A solution used after the problem has been solved for another design stays valid: its design is put back and its runs are repeated first (once), which costs one more set of simulations; the usual loop (solve, back-propagate, next design) never does.

### `lightsim.invdes.NativeSolution.yee_vjp`

```text
NativeSolution.yee_vjp(g)
```

dL/d(eps_r) on the Yee voxels of every design-grid monitor box: list over grids of `{comp: (origin, real array [i, j, k])}`.

### `lightsim.invdes.NativeSolution.vjp`

```text
NativeSolution.vjp(g)
```

dL/d(eps) on the samples of every design grid (list of arrays of shape `grid.dims`) for the upstream gradient `g`.

### `lightsim.invdes.NativeSolution.jacobian`

```text
NativeSolution.jacobian(m, n, k)
```

Complex dS[m, n, k]/d(eps) on the samples of every design grid.

### `lightsim.invdes.Port`

```text
Port(name, axis, position, min, max, profiles, direction)
```

A waveguide port: a mode plane and the direction pointing *into* the device (`"+"` or `"-"` along `axis`). `profiles` are the mode profile dicts (`Mode.profile`) of the modes used at this port.

### `lightsim.invdes.Excitation`

```text
Excitation(name, source, norm=None)
```

A forward excitation that is not a port mode (a beam, a plane wave, a dipole...). `source` is any lightsim source dict. S is normalised by the spectrum of the source signal times `norm` (one complex or real number per frequency; e.g. the square root of the incident power per unit spectrum, so that `|S|^2` is a coupling efficiency).

### `lightsim.invdes.DesignGrid`

```text
DesignGrid(min, max, dims, medium, name='design', permeability=False)
```

One index grid of the design: samples `dims = (nx, ny, nz)` spanning the box `[min, max]` (see `lightsim.IndexGrid`); a 2-tuple means one sample along z (a layer extruded through `[zmin, zmax)`). `medium` is the medium dict that carries the grid (its `eps` should be the largest permittivity the grid can take: the mesher uses it).

`permeability=True` makes the samples relative permeabilities: the grid then owns H voxels, its E voxels take the medium's `eps`, and the medium's `mu` must be the largest permeability the grid takes (above 1). Permeability grids are differentiated by reverse mode (`lightsim.autodiff`), not by the S-matrix paths.

### `lightsim.invdes.grid_weights`

```text
grid_weights(comp, sim_grid, origin, shape, grid)
```

Separable sampling matrices (Wx, Wy, Wz) of an index grid for the Yee positions of component `comp` in the index box `origin`/`shape`:

```text
eps_yee[i, j, k] = sum_abc Wx[i, a] Wy[j, b] Wz[k, c] eps_grid[a, b, c]
```

on the voxels the grid owns (rows of the others are zero). The transpose is the exact adjoint of the IndexGrid rasterisation.

### `lightsim.invdes.Design`

```text
Design(
    min,
    max,
    shape,
    layers,
    fields=1,
    filter_radius=0.0,
    filter='conic',
    eta=0.5,
    fixed_solid=None,
    fixed_void=None,
    symmetry=(),
    medium_sigma=0.0,
    name='design',
)
```

A planar density design.

**Parameters**

- `min`, `max`: (x, y) corners of the design rectangle (metres).
- `shape`: (nx, ny) density samples; the sample nodes span the rectangle inclusive, like `lightsim.IndexGrid`.
- `layers`: list of `(zmin, zmax, eps_base, coeffs)`: one index grid per entry with `eps = eps_base + sum_l coeffs[l] u_l`. Use the constructors `slab`, `partial_etch` and `effective_index`.
- `fields`: number of density fields (1, or 2 for a partial etch).
- `filter_radius`: radius of the conic filter (metres); the minimum feature size is about twice the radius. `filter="gaussian"` uses a Gaussian of that FWHM.
- `eta`: projection threshold.
- `fixed_solid / fixed_void`: boolean (nx, ny) masks of pixels held at 1 / 0 (waveguide stubs, cladding gaps). Applied before the filter (so the filter sees them) and again after the projection.
- `symmetry`: subset of `("x", "y")`: mirror-symmetrise the density about the centre line normal to that axis.

### `lightsim.invdes.Design.slab`

```text
Design.slab(min, max, z, eps_lo, eps_hi, shape=None, pixel=None, **kw)
```

Fully etched layer between `z = (zmin, zmax)` (3-D).

### `lightsim.invdes.Design.partial_etch`

```text
Design.partial_etch(min, max, z, eps_lo, eps_hi, shape=None, pixel=None, **kw)
```

Two etch depths (3-D): `z = (zmin, zmid, zmax)`. Field 0 is the lower part of the slab, field 1 the upper part (only where field 0 is).

### `lightsim.invdes.Design.effective_index`

```text
Design.effective_index(
    min,
    max,
    n_lo,
    n_hi,
    n_partial=None,
    shape=None,
    pixel=None,
    **kw,
)
```

2.5-D effective-index design: one z-invariant grid whose permittivity runs from `n_lo^2` (etched) to `n_hi^2` (slab); `n_partial` (slab index of the partially etched film) adds the second density field.

### `lightsim.invdes.Design.grids`

```text
Design.grids()
```

The `DesignGrid` list for `SMatrixProblem`.

### `lightsim.invdes.Design.coordinates`

```text
Design.coordinates()
```

`(x, y)`: the coordinates (metres) of the density samples along the two axes of the design rectangle, `shape[0]` and `shape[1]` values from `min` to `max` inclusive.

### `lightsim.invdes.Design.random`

```text
Design.random(seed=0, mean=0.5, amplitude=0.0)
```

A starting density: `mean` plus smooth noise of `amplitude`.

### `lightsim.invdes.Design.filtered`

```text
Design.filtered(rho)
```

rho~ : symmetrised, clamped at the fixed pixels, filtered.

### `lightsim.invdes.Design.projected`

```text
Design.projected(rho, beta, eta=None)
```

rho^ per density field (fixed pixels enforced), shape (fields, nx, ny).

### `lightsim.invdes.Design.levels`

```text
Design.levels(rho, beta, eta=None)
```

The level fields u_l (cumulative products of rho^).

### `lightsim.invdes.Design.eps`

```text
Design.eps(rho, beta, eta=None)
```

Permittivity arrays, one per grid (shape (nx, ny, 1)).

### `lightsim.invdes.Design.vjp`

```text
Design.vjp(rho, beta, grads, eta=None)
```

dL/d(rho) from dL/d(eps) on the grids (`grads`: one array per grid).

### `lightsim.invdes.Design.binarization`

```text
Design.binarization(rho, beta, eta=None)
```

Binarisation of the projected density over the free pixels.

### `lightsim.invdes.InverseDesign`

```text
InverseDesign(problem, design, objective=None, robust=None)
```

Ties an `SMatrixProblem` to a `Design`.

`S` has the axes (outputs, inputs, frequencies), as in `SMatrixProblem`: `S[m, n, k]`, shape `[M, N, K]`.

`forward(rho, beta)` returns `(S, pullback)` where `pullback(g)` is dL/d(rho) for the upstream gradient `g = dL/dRe S + i dL/dIm S`: the S-matrix as a differentiable layer in the density.

`objective(S) -> (value, g)` (see `lightsim.invdes.objectives`) is used by `value_and_grad`. `robust` = threshold offset d adds the eroded (eta + d) and dilated (eta - d) variants; the objective is then the mean over the three (three times the simulations).

### `lightsim.invdes.InverseDesign.s_matrix`

```text
InverseDesign.s_matrix(rho, beta, eta=None)
```

S only (N forward runs).

### `lightsim.invdes.InverseDesign.forward`

```text
InverseDesign.forward(rho, beta, eta=None)
```

S-matrix and its pullback for the density `rho` at projection strength `beta` (threshold `eta`, default that of the design).

Returns `(S, pullback)`: `S` is the complex array `[M, N, K]` (N forward and M adjoint runs are made), and `pullback(g)` maps the upstream gradient `g = dL/dRe S + i dL/dIm S` of a real loss, an array of the same shape, to `dL/d(rho)` with the shape of `rho`: through the S-matrix and through the filter and projection of the design. This is the pair an autodiff framework needs.

### `lightsim.invdes.InverseDesign.value_and_grad`

```text
InverseDesign.value_and_grad(rho, beta)
```

(L, dL/drho, info) for the stored objective.

### `lightsim.invdes.ProbeDesign`

```text
ProbeDesign(problem, design, target, probes=2, seed=0, weights=None)
```

Stochastic gradient of `matrix_distance` from random probe vectors, for devices with many ports.

Each call of `value_and_grad` draws `probes` random complex input vectors `x` (`E[x x^H] = I`), runs one forward simulation per vector with all inputs excited at once (`SMatrixProblem.solve_combined`) and one adjoint simulation per vector and frequency with the outputs driven by the residual `r = S x - e^{i phi} T x`. The mean of `|r|^2 / ||T||^2` is an unbiased estimate of the distance and its gradient an unbiased estimate of the distance's gradient (the global phase `phi` is estimated from the same probes). A gradient costs `probes * (1 + K)` simulations whatever the number of ports N, against `2 N` for the exact `InverseDesign`; the price is noise, which Adam averages over iterations. Use with `optimize(..., maximize=False)`.

### `lightsim.invdes.ProbeDesign.value_and_grad`

```text
ProbeDesign.value_and_grad(rho, beta)
```

(estimated distance, estimated gradient, info).

### `lightsim.invdes.CombinedSolution`

```text
CombinedSolution(problem, eps, X, Y, fwd, stats, fields)
```

Result of `SMatrixProblem.solve_combined`: P forward runs, each with *all* inputs excited at once with complex amplitudes.

`X` (complex `[P, N, K]`) holds the input vectors that were actually launched (measured at the source planes, so they differ from the requested weights by the small port-to-port differences of the mode sources) and `Y` (complex `[M, P, K]`) the output amplitudes, so that exactly `Y[:, p, k] = S[:, :, k] @ X[p, :, k]`. Both are expressed in one common unit per run (the mean launch factor of the run), which leaves that relation and the gradients exact.

`vjp` gives the gradient of a real loss of `Y` from one adjoint run per probe and frequency: the output modes launched backwards together, weighted by the upstream gradient. The cost of a gradient is then `P + P K` simulations whatever the number of ports, against `N + M` for `SMatrixProblem.solve`.

### `lightsim.invdes.CombinedSolution.vjp`

```text
CombinedSolution.vjp(g)
```

dL/d(eps) on the samples of every design grid for the upstream gradient `g = dL/dRe(Y) + i dL/dIm(Y)` (complex `[M, P, K]`).

Runs one adjoint simulation per probe `p` and frequency `k` with a non-zero `g[:, p, k]`: output mode `m` is launched with amplitude `conj(g[m, p, k])`. The amplitudes actually launched are measured; the gradient returned is exact for them, and `self.g_eff` holds the upstream gradient they correspond to (equal to `g` to the port-to-port spread of the launch factor, which is compensated from the previous call).

### `lightsim.invdes.optimize`

```text
optimize(
    fn,
    rho0,
    iterations,
    beta=None,
    method='adam',
    lr=0.02,
    checkpoint=None,
    callback=None,
    maximize=True,
    betas=(0.9, 0.999),
    verbose=True,
    save_every=1,
    reset_on_beta=True,
)
```

Optimise a density in [0, 1].

`fn` is an `InverseDesign` (its `value_and_grad` is used) or a callable `fn(rho, beta) -> (value, grad, info)`. `beta` is a number or a callable of the iteration (see `beta_schedule`).

`method="adam"` (projected Adam; the moments restart when beta changes unless `reset_on_beta=False`: a restart makes the next step `lr` on every pixel, which can throw a narrow-band design off its resonances) or `"lbfgs"` (scipy L-BFGS-B, restarted at every beta stage; an "iteration" is then one function evaluation).

`checkpoint` (npz path): the density, optimiser state and history are saved every `save_every` iterations and the run resumes from the file if it exists. History entries: `iteration, fom, beta, binarization, seconds` (wall time of the iteration) plus what `info` provides (`sim_seconds`: wall time of the forward + adjoint runs).

Returns `(rho, history)` with `history` a dict of arrays.

### `lightsim.invdes.beta_schedule`

```text
beta_schedule(start=1.0, factor=2.0, every=20, maximum=64.0)
```

beta(iteration) = min(maximum, start \* factor \*\* (iteration // every)).

### `lightsim.invdes.directional_derivative`

```text
directional_derivative(f, x, direction, h=0.01)
```

Finite-difference derivative of `f` at `x` along `direction`, for checking a gradient: `np.sum(gradient * direction)` should match it.

Central differences with steps `h` and `2 h` combined by Richardson extrapolation, `(4 D(h) - D(2 h)) / 3`, whose error is of order h\*\*4 instead of h\*\*2. A gradient of the whole chain (filter, projection, S-matrix, objective) needs it: the projection can make the objective strongly curved in the density, and a plain central difference with a step large enough to rise above the single-precision noise of the runs (two runs of one design differ in the last bits) is then off by its h\*\*2 term. It costs four evaluations of `f`. Keep `h` well above the noise: about 1e-2 of the density range in single precision, 1e-3 in the double-precision build.

### `lightsim.invdes.matrix_fidelity`

```text
matrix_fidelity(S, target, weights=None)
```

Matrix overlap with a target T[m, n], averaged over frequency:

```text
L = mean_k |<T, S_k>|^2 / ||T||^4,     <T, S> = sum_mn conj(T_mn) S_mn.
```

L = 1 exactly when `S_k = exp(i phi_k) T` (one free global phase per frequency): it is the product of the normalised fidelity `|<T,S>|^2 / (||T||^2 ||S||^2)` and the throughput `||S||^2 / ||T||^2`, so shape errors and loss are both penalised.

### `lightsim.invdes.matrix_distance`

```text
matrix_distance(S, target, weights=None)
```

Distance to a target up to one global phase per frequency, to be *minimised*:

```text
L = mean_k ||S_k - exp(i phi_k) T||_F^2 / ||T||_F^2,   phi_k = arg <T, S_k>,
```

which equals `mean_k (||S_k||^2 + ||T||^2 - 2 |<T, S_k>|) / ||T||^2`: zero only for `S_k = exp(i phi_k) T`, so it penalises both shape errors and loss. Being a sum over inputs of `|(S - e^{i phi} T) e_n|^2` it is also `E_x |(S - e^{i phi} T) x|^2` for random vectors with `E[x x^H] = I`, which is what `lightsim.invdes.ProbeDesign` estimates. The gradient treats `phi` as fixed (it is optimal, so its own derivative does not contribute).

### `lightsim.invdes.probe_vectors`

```text
probe_vectors(n, p, rng=None)
```

`p` random complex probe vectors of length `n` (rows), circular Gaussian with `E[x x^H] = I`.

### `lightsim.invdes.matrix_metrics`

```text
matrix_metrics(S, target)
```

Per-frequency diagnostics of S against T: normalised `fidelity` (shape only), `throughput` `||S||^2/||T||^2`, their product `overlap` and the best-fit global `phase`.

### `lightsim.invdes.band_transmission`

```text
band_transmission(S, want, crosstalk=1.0, alpha=None)
```

In-band transmission to the right port with a crosstalk penalty.

`want` is a boolean mask with the shape of S: True where power should arrive (pass bands), False where it should not (the other channels' bands); entries to ignore can be given as a masked array or as -1 in an integer array.

```text
L = T_pass - crosstalk * mean(|S|^2 over stop entries)
```

with `T_pass` the mean of `|S|^2` over the pass entries, or their smooth minimum when `alpha` is given (equalises the channels).

### `lightsim.invdes.mean_coupling`

```text
mean_coupling(S, weights=None)
```

Coupled power averaged over sources, output channels and frequencies (`weights`: optional array broadcastable to S).

### `lightsim.invdes.softmin`

```text
softmin(x, alpha)
```

Smooth minimum -(1/a) log(mean(exp(-a x))) and its gradient.

### `lightsim.invdes.density_polygons`

```text
density_polygons(density, min, max, level=0.5, upsample=4, pad=True)
```

Contours of a 2-D density at `level` as polygons.

`density[ix, iy]` has its nodes spanning `[min, max]` inclusive. It is resampled `upsample` times (bilinear, the same interpolation as the index grid) and contoured with marching squares (`contourpy`, shipped with matplotlib). With `pad` the outside of the rectangle counts as empty so every polygon is closed.

Returns a list of `(exterior, holes)`: `exterior` an (n, 2) array of xy points in metres, `holes` a list of such arrays.

### `lightsim.invdes.planar_simulation`

```text
planar_simulation(
    min,
    max,
    wavelength,
    n_background,
    n_max,
    steps_per_wavelength=20.0,
    z_cell=1e-06,
    pml_layers=12,
    run_time=None,
    shutoff=1e-05,
    background='eim_background',
    polarization='both',
)
```

Empty 2.5-D effective-index simulation: a 2-D simulation invariant along z (`dimensions=2`) on a uniform xy grid, standing for a slab of thickness `z_cell`, with PML in x and y (the model of `lightsim.eim`, built without a layout). `min` / `max` are the xy corners; `n_max` (largest effective index) sets the cell size `wavelength / (n_max * steps_per_wavelength)`. Add effective-index structures as z-spanning boxes / poly-slabs (z from -1 to 1) and the design from `lightsim.invdes.Design.effective_index`.

`polarization` = `"te"` / `"tm"` restricts the simulation to one field set; with `"both"` the engine updates the sets the port modes drive, which for one mode family is the same thing.

### `lightsim.invdes.port_mode`

```text
port_mode(
    sim,
    name,
    axis,
    position,
    min,
    max,
    freq,
    direction,
    polarization=None,
    num_modes=1,
)
```

Solve the port modes and return a `Port`.

`polarization` = `"te"` / `"tm"` selects the 1-D lateral solver of a 2-D / 2.5-D simulation (`lightsim.eim.solve_port_mode`); `None` uses `lightsim.solve_modes` (the full-vectorial solver in 3-D). Passing a polarization in a 3-D simulation is an error: the lateral solver would read one line of cells and return a mode of the cladding. `direction` points into the device.

### `lightsim.invdes.project`

```text
project(x, beta, eta=0.5)
```

Smoothed Heaviside: (tanh(b e) + tanh(b (x - e))) / (tanh(b e) + tanh(b (1 - e))).

### `lightsim.invdes.filter_density`

```text
filter_density(rho, kernel, transpose=False)
```

Normalised convolution `(K * rho) / (K * 1)` (zero padding, so the weights renormalise at the edges). `transpose=True` applies the adjoint linear map.

### `lightsim.invdes.conic_kernel`

```text
conic_kernel(radius_px)
```

Conic (hat) filter of radius `radius_px` pixels: w = max(0, 1 - d/R).

### `lightsim.invdes.gaussian_kernel`

```text
gaussian_kernel(radius_px)
```

Gaussian filter whose full width at half maximum is `radius_px`.

### `lightsim.invdes.binarization`

```text
binarization(u)
```

1 for a fully binary density, 0 for uniform grey: 1 - mean(4 u (1 - u)).

## `lightsim.invdes.shapes`

Shape parametrisations for inverse design: polygons with movable vertices and level sets.

### `lightsim.invdes.shapes.polygon_density`

```text
polygon_density(vertices, x, y)
```

Area fraction of every pixel cell covered by a simple polygon.

`vertices`: (n, 2) array of xy points (metres), either orientation, the last vertex joined to the first. `x`, `y`: the uniformly spaced sample coordinates of the design grid (`Design.coordinates()`); the cell of sample (i, j) is the pixel rectangle centred on it. Returns an `(len(x), len(y))` array in [0, 1].

The area in each cell is the sum over the edges of the signed area between the edge and the bottom of the cell (Green's theorem column by column), so it is exact for any simple polygon, including the parts of it that lie outside the grid. Edges that cross each other count with their winding number; check with `polygon_is_simple`.

### `lightsim.invdes.shapes.polygon_density_vjp`

```text
polygon_density_vjp(vertices, x, y, g)
```

Transpose of `polygon_density`: `dL/d vertices` (n, 2) from `g = dL/d density` on the grid.

For a vertex v between the edges (u, v) and (v, w), the derivative of the area of cell c is `sum over the two edges of n_e L_e int_{t in c} phi_v(t) dt`: the outward normal times the edge length times the integral of the vertex's hat function over the part of the edge inside the cell.

### `lightsim.invdes.shapes.polygon_is_simple`

```text
polygon_is_simple(vertices, tol=0.0)
```

True if no two non-adjacent edges of the closed polygon intersect (and no edge has zero length).

### `lightsim.invdes.shapes.smooth_heaviside`

```text
smooth_heaviside(s)
```

C1 smoothed step: 0 for s <= -1, 1 for s >= 1, `1/2 + s (3 - s^2) / 4` between. `H(s) - step(s)` is odd, so a flat boundary keeps its area.

### `lightsim.invdes.shapes.Polygon`

```text
Polygon(vertices, free=True, mirror=None, axis=0.0, bounds=None, field=0, check=True)
```

A polygon whose vertices are (functions of) shape parameters.

**Parameters**

- `vertices`: (n, 2) xy points in metres. Without `mirror` they are the closed polygon; with `mirror` they are one half of it, an open chain whose first and last points lie on the mirror line.
- `free`: which coordinates are parameters: `True` (all), `False` (none: a fixed polygon), a boolean (n,) mask of free vertices or a boolean (n, 2) mask of free coordinates. The parameter vector lists the free coordinates in row order (x0, y0, x1, y1, ...).
- `mirror`: `None`, `"x"` (mirror about the line x = `axis`) or `"y"` (about y = `axis`). The full polygon is the chain followed by its mirror image traversed backwards; the coordinate of the two end points normal to the line is held on it.
- `bounds`: `(lo, hi)`, scalars or arrays with one value per parameter. Unbounded entries take the extent of the design rectangle along the coordinate's axis (`ShapeDesign`).
- `field`: density field the polygon draws into (0, or 1 for the upper level of a two-level design).
- `check`: raise `ValueError` when the polygon is not simple (edges that cross); the density of a self-intersecting polygon counts winding numbers and is wrong.

Use `from_function` for vertices computed from fewer parameters (spline control points, a width profile).

### `lightsim.invdes.shapes.Polygon.from_function`

```text
Polygon.from_function(
    fn,
    p0,
    bounds,
    jac=None,
    mirror=None,
    axis=0.0,
    field=0,
    check=True,
)
```

Vertices `fn(p) -> (n, 2)` from a parameter vector `p` (`p0`: initial values, `bounds`: `(lo, hi)`, finite). `jac(p) -> (n, 2, P)` is the Jacobian; without it a central difference of `fn` (step 1e-6 of each parameter's range) is used, which is exact to rounding for maps linear in `p` (splines through control points). `mirror` applies to the vertices `fn` returns.

### `lightsim.invdes.shapes.Polygon.size`

Property. Number of parameters.

### `lightsim.invdes.shapes.Polygon.chain`

```text
Polygon.chain(p)
```

The vertices before mirroring.

### `lightsim.invdes.shapes.Polygon.vertices`

```text
Polygon.vertices(p)
```

The closed polygon (m, 2) for parameters `p` (physical units).

### `lightsim.invdes.shapes.Polygon.density`

```text
Polygon.density(p, x, y)
```

Area fractions on the samples `x`, `y` (`polygon_density`); `ValueError` if the polygon is not simple and `check` is set.

### `lightsim.invdes.shapes.Polygon.vjp`

```text
Polygon.vjp(p, x, y, g)
```

dL/dp from dL/d density.

### `lightsim.invdes.shapes.LevelSet`

```text
LevelSet(phi, p0, bounds, jac=None, width=None, field=0)
```

`rho = H(phi(p, X, Y) / width)`: a smoothed Heaviside (`smooth_heaviside`, a ramp from -width to +width) of a signed function that is positive inside the shape.

**Parameters**

- `phi`: `phi(p, X, Y) -> array` with the shape of `X`; `X`, `Y` are the sample coordinates (`meshgrid(x, y, indexing="ij")`). It should be close to a signed distance (|grad phi| ~ 1) near the boundary, so that `width` is a length.
- `p0`, `bounds`: initial parameters and `(lo, hi)` (finite).
- `jac`: `jac(p, X, Y) -> (P,) + X.shape`, d phi / d p. Default: central differences of `phi` (step 1e-6 of each parameter's range).
- `width`: half-width of the smoothed step (metres). Default: the mean pixel of the design.
- `field`: density field (0 or 1).

### `lightsim.invdes.shapes.LevelSet.width_profile`

```text
LevelSet.width_profile(
    x,
    widths,
    bounds,
    free=None,
    center=0.0,
    kind='cubic',
    axis='x',
    width=None,
    field=0,
)
```

A strip along `axis` whose full width w(s) interpolates `widths` at the stations `x` (`kind = "linear"` or `"cubic"`, a natural cubic spline), centred on the line `center`: `phi = w(s) / 2 - |t - center|` (s along `axis`, t across). `free`: boolean mask of the stations whose width is a parameter (default: all); the others keep their value in `widths` (the two ends, where the strip meets fixed waveguides). `bounds = (wmin, wmax)` on the free widths: `wmin` is a minimum feature size the optimisation cannot go below. Outside `[x[0], x[-1]]` the end widths continue.

### `lightsim.invdes.shapes.LevelSet.size`

Property. Number of parameters.

### `lightsim.invdes.shapes.LevelSet.density`

```text
LevelSet.density(p, x, y)
```

`H(phi / width)` on the samples `x`, `y`.

### `lightsim.invdes.shapes.LevelSet.vjp`

```text
LevelSet.vjp(p, x, y, g)
```

dL/dp from dL/d density.

### `lightsim.invdes.shapes.ShapeDesign`

```text
ShapeDesign(design, shapes, projection=False)
```

Shapes drawn on the grid of a `Design`.

`ShapeDesign(design, shapes)` has the interface `InverseDesign` and `optimize` expect from a design (`eps`, `vjp`, `grids`, `binarization`), with the normalised parameter vector `u` in `[0, 1]^P` in place of the density: `p = lo + u (hi - lo)`, the physical parameters of all shapes concatenated.

**Parameters**

- `design`: the `Design` that sets the grid (rectangle, samples), the layers (`slab`, `partial_etch`, `effective_index`) and the permittivities.
- `shapes`: `Polygon` and `LevelSet` objects. Each draws into its `field`; the densities of one field add up and must not overlap (a density above 1 raises `ValueError`).
- `projection`: `False` (default): the density goes straight into the level fields, `eps = eps_base + sum_l coeff_l u_l` with `u_0 = rho_0`, `u_1 = rho_0 rho_1`; the `beta` and `eta` arguments are ignored and the filter, projection, masks and symmetry of `design` are not used. `True`: the density goes through the design's own chain (filter, tanh projection at `beta`, masks), for example to smooth the boundary with a filter.

### `lightsim.invdes.shapes.ShapeDesign.size`

Property. Number of shape parameters.

### `lightsim.invdes.shapes.ShapeDesign.params`

```text
ShapeDesign.params(u)
```

Physical parameters `lo + u (hi - lo)`.

### `lightsim.invdes.shapes.ShapeDesign.normalize`

```text
ShapeDesign.normalize(p)
```

Normalised parameters `(p - lo) / (hi - lo)`.

### `lightsim.invdes.shapes.ShapeDesign.initial`

```text
ShapeDesign.initial()
```

The normalised initial parameters (from the shapes' `p0`).

### `lightsim.invdes.shapes.ShapeDesign.shape_params`

```text
ShapeDesign.shape_params(u)
```

List of the physical parameters of each shape.

### `lightsim.invdes.shapes.ShapeDesign.polygons`

```text
ShapeDesign.polygons(u)
```

`[(field, vertices)]` of the `Polygon` shapes (metres), for GDS export; level sets are traced with `density_polygons` on `density`.

### `lightsim.invdes.shapes.ShapeDesign.density`

```text
ShapeDesign.density(u)
```

The density on the design grid, `(nx, ny)` (or `(fields, nx, ny)`).

### `lightsim.invdes.shapes.ShapeDesign.density_vjp`

```text
ShapeDesign.density_vjp(u, g)
```

dL/du from dL/d density.

### `lightsim.invdes.shapes.ShapeDesign.grids`

```text
ShapeDesign.grids()
```

The `DesignGrid` list (that of `design`).

### `lightsim.invdes.shapes.ShapeDesign.eps`

```text
ShapeDesign.eps(u, beta=None, eta=None)
```

Permittivity arrays, one per grid (shape (nx, ny, 1)).

### `lightsim.invdes.shapes.ShapeDesign.vjp`

```text
ShapeDesign.vjp(u, beta, grads, eta=None)
```

dL/du from dL/d(eps) on the grids.

### `lightsim.invdes.shapes.ShapeDesign.binarization`

```text
ShapeDesign.binarization(u, beta=None, eta=None)
```

`1 - mean(4 rho (1 - rho))` of the density (the fraction of grey boundary pixels, not a convergence measure for shapes).

## `lightsim.invdes.torch`

The S-matrix as a PyTorch function.

### `lightsim.invdes.torch.SMatrixFunction`

```text
SMatrixFunction(*args, **kwargs)
```

The autograd function behind `s_matrix`: the simulations run in `forward` and the adjoint simulations in `backward`. Call `s_matrix` instead of using it directly.

### `lightsim.invdes.torch.SMatrixFunction.forward`

```text
SMatrixFunction.forward(ctx, rho, layer, beta, eta)
```

Solve for the density `rho` and return the complex S-matrix `[M, N, K]` on the device of `rho` (complex128 for a float64 density, complex64 otherwise).

### `lightsim.invdes.torch.SMatrixFunction.backward`

```text
SMatrixFunction.backward(ctx, grad_S)
```

Gradient with respect to the density from the gradient with respect to S (PyTorch's convention for complex tensors).

### `lightsim.invdes.torch.s_matrix`

```text
s_matrix(layer, rho, beta=8.0, eta=None)
```

Differentiable S-matrix `[M, N, K]` (complex) of the density tensor `rho`.

## `lightsim.invdes.jax`

The S-matrix as a JAX function.

### `lightsim.invdes.jax.s_matrix`

```text
s_matrix(layer, rho, beta=8.0, eta=None)
```

Differentiable S-matrix `[M, N, K]` (complex) of the density `rho`.

## `lightsim.autodiff`

Gradients computed inside the engine: persistent S-matrix session, reverse mode through the time loop.

### `lightsim.autodiff.GradientSession`

```text
GradientSession(
    sim,
    grids,
    inputs,
    outputs,
    freqs,
    devices='cpu',
    fwidth=None,
    normalize='monitor',
    components=('Ex', 'Ey', 'Ez'),
    signal=None,
)
```

Persistent native S-matrix session.

Parameters are those of `lightsim.invdes.SMatrixProblem`: `sim` (base simulation; its sources and monitors are ignored), `grids` (`lightsim.invdes.DesignGrid`), `inputs` (`Port`, `(Port, mode)` or `Excitation`), `outputs` (`Port` or `(Port, mode)`), `freqs` (Hz). `devices` is a device string or a list of them (`["cuda:0", "cuda:1"]`): the independent runs are dispatched over the list, one at a time per entry, and the design-region fields are gathered on the first device for the adjoint product.

```text
sess = ls.autodiff.GradientSession(sim, grids, [p_in], [p_out], freqs, devices="cuda:0")
sess.set_design(eps)                 # only the design box is re-rasterised and uploaded
S = sess.forward()                   # complex [M, N, K]
grads = sess.backward(dL_dS)         # list of dL/d(eps), one per grid
```

`forward(adjoint=True)` also runs every adjoint simulation in the same dispatch (useful with several devices); otherwise `backward` runs the adjoint simulations the upstream gradient needs (outputs whose rows of `g` are zero are skipped) and keeps them for further `backward` calls on the same design.

The design-region fields and the port monitors are sampled every `dft_stride` time steps (attribute of the session): the stride of `sim` (`Simulation.dft_stride`), by default the automatic one, which is as large as the bandwidth of the source allows.

### `lightsim.autodiff.GradientSession.set_design`

```text
GradientSession.set_design(eps)
```

Set the index-grid permittivities (one array per grid).

### `lightsim.autodiff.GradientSession.forward`

```text
GradientSession.forward(adjoint=False)
```

Run the N excitations; complex `S[M, N, K]`.

### `lightsim.autodiff.GradientSession.backward`

```text
GradientSession.backward(g)
```

dL/d(eps) on the samples of every design grid (list of arrays of shape `grid.dims`) for the upstream gradient `g` (see the module docstring for the convention).

### `lightsim.autodiff.GradientSession.backward_yee`

```text
GradientSession.backward_yee(g)
```

dL/d(eps_r) on the Yee voxels of every design box: list over grids of `{comp: (origin, real array [i, j, k])}`.

### `lightsim.autodiff.GradientSession.yee_permittivity`

```text
GradientSession.yee_permittivity()
```

What the engine uses: `{comp: (origin, array)}` over the design bounding box, NaN where no grid owns the voxel.

### `lightsim.autodiff.GradientSession.stats`

```text
GradientSession.stats()
```

Runs done since the last `set_design`.

### `lightsim.autodiff.ReverseSMatrix`

```text
ReverseSMatrix(
    sim,
    grids,
    inputs,
    outputs,
    freqs,
    devices='cpu',
    fwidth=None,
    normalize='monitor',
    components=('Ex', 'Ey', 'Ez'),
    signal=None,
)
```

S-matrix problem with the gradient taken by reverse mode through the time loop (engine `"reverse"` of `lightsim.invdes`).

Same arguments and conventions as `GradientSession`. `forward` returns the S of the session (same mode overlap, normalisation, unit-power ports and DFT); `backward(g)` runs one reverse pass per input whose column of `g` is non-zero and returns the exact derivative of the discrete scheme, including the design dependence of the launched amplitude (`normalize="monitor"`). The inputs are spread over `devices`; every input keeps one engine (forward fields, adjoint fields, the recorded PML shell and port planes).

Refused, with an error naming the other engines: media with off-diagonal permittivity or oriented poles, Bloch boundaries, design grids in a PML, and absorbers too strong to be stepped backwards (the reconstruction check of reverse mode).

### `lightsim.autodiff.ReverseSMatrix.set_design`

```text
ReverseSMatrix.set_design(eps)
```

Set the index-grid permittivities (one array per grid).

### `lightsim.autodiff.ReverseSMatrix.forward`

```text
ReverseSMatrix.forward()
```

Run the N forward simulations; complex `S[M, N, K]`.

### `lightsim.autodiff.ReverseSMatrix.backward`

```text
ReverseSMatrix.backward(g)
```

dL/d(eps) on the samples of every design grid for the upstream gradient `g` (convention of `GradientSession`).

### `lightsim.autodiff.ReverseSMatrix.stats`

```text
ReverseSMatrix.stats()
```

Per input: device, steps, forward_seconds, reverse_seconds.

### `lightsim.autodiff.flux_energy`

```text
flux_energy(axis, position, min, max, t_start=None, t_end=None, taper=None)
```

Objective: the energy (J) crossing the plane `axis = position` (restricted to `[min, max]`) towards `+axis` within a time window, 1 on `[t_start, t_end]` (defaults: the whole run) with raised-cosine edges of width `taper`. In a 2.5-D simulation it is the energy per z cell.

### `lightsim.autodiff.field_samples`

```text
field_samples(min, max, components=('Ex', 'Ey', 'Ez'))
```

Objective evaluated by the caller on the recorded samples of `components` in the box `[min, max]` (every step): pass a function `fn(series) -> (L, cotangents)` to `value_and_grad` or use `ReverseSession.series` / `ReverseSession.backward`.

### `lightsim.autodiff.ReverseSession`

```text
ReverseSession(sim, grids, objective, device='cpu')
```

Time-domain reverse mode for a simulation with a design region.

`sim` keeps its sources (its monitors are ignored); `objective` is `flux_energy` or `field_samples`. The session is persistent: `set_design` only uploads the design box.

```text
sess = ls.autodiff.ReverseSession(sim, grids, ls.autodiff.flux_energy(0, x, mn, mx), device="cuda:0")
sess.set_design(eps)
L = sess.forward()          # runs the simulation, records the monitor (and the PML shell)
grads = sess.backward()     # one reverse pass: list of dL/d(eps), one per grid
```

With `field_samples`, `series()` returns the recorded samples as `{comp: array [step, i, j, k]}` (E at `(n + 1) dt`, H at `(n + 1/2) dt`) and `backward(cotangents)` takes `dL/d(sample)` in the same layout.

### `lightsim.autodiff.ReverseSession.set_design`

```text
ReverseSession.set_design(eps)
```

Set the index-grid permittivities (one array per grid).

### `lightsim.autodiff.ReverseSession.forward`

```text
ReverseSession.forward()
```

Run the simulation, recording the objective monitor and the PML shell on the device; the value of a `flux_energy` objective (0 for `field_samples`, whose value the caller computes).

### `lightsim.autodiff.ReverseSession.series`

```text
ReverseSession.series()
```

Recorded samples of the last `forward`: `{comp: array [step, i, j, k]}`. A flux plane records several boxes of one component; their keys are `comp`, `comp_1`, ...

### `lightsim.autodiff.ReverseSession.times`

```text
ReverseSession.times(comp)
```

Sample times of component `comp` of `series`.

### `lightsim.autodiff.ReverseSession.backward`

```text
ReverseSession.backward(cotangents=None)
```

One reverse pass: `dL/d(eps)` on the samples of every design grid (list of arrays of shape `grid.dims`). `cotangents` holds `dL/d(sample)` in the layout of `series` (a dict with its keys, or a list in its order); omit it for `flux_energy`.

### `lightsim.autodiff.ReverseSession.set_pole_gradients`

```text
ReverseSession.set_pole_gradients(on=True)
```

Also compute, on every reverse pass, the gradient with respect to the dispersive poles of the scene's media (see `pole_gradients`). Dispersive structures must stay out of the absorbing layers (the reverse pass does not reconstruct the fields there).

### `lightsim.autodiff.ReverseSession.pole_gradients`

```text
ReverseSession.pole_gradients()
```

Gradients with respect to the dispersive poles from the last reverse pass: one dict per (medium, pole, axis) with `medium`, `pole` (index into the medium's `poles`), `axis`, `coefficients` (dL/d(c0, c1, c2) of the ADE step `P(n+1) = c0 P(n) + c1 P(n-1) + c2 E(n)`) and `params`: dL/d of the pole's parameters, Lorentz (delta_eps, omega0, gamma), Drude (omega_p, gamma), Debye (delta_eps, tau). The sum over the axes of a pole is the gradient with respect to that pole of the medium.

### `lightsim.autodiff.ReverseSession.stats`

```text
ReverseSession.stats()
```

Timings and sizes of the last forward run and reverse pass (`forward_seconds`, `reverse_seconds`, `steps`, `cells`, `shell_floats_per_step`, ...).

### `lightsim.autodiff.value_and_grad`

```text
value_and_grad(sim, grids, eps, objective, device='cpu', fn=None, session=None)
```

Objective value and its gradient with respect to the index-grid permittivities by reverse-mode differentiation through the time loop.

`objective` is `flux_energy` (built in) or `field_samples` with `fn(series) -> (L, cotangents)`, where `series` is the dict of `ReverseSession.series` and `cotangents` a dict with the same keys holding `dL/d(sample)`.

Returns `(L, grads, info)`: `grads` is a list with one array per grid (a single array when one grid was given as such), `info` holds the timings and the session (pass it back as `session=` to reuse the device buffers for the next design).

## `lightsim.symmetry`

Mirror-symmetry reduction.

### `lightsim.symmetry.reduce`

```text
reduce(sim, symmetry)
```

Return (reduced simulation, info) for mirror symmetry `(sx, sy, sz)`.

### `lightsim.symmetry.run_symmetric`

```text
run_symmetric(sim, symmetry, device='auto', progress=0)
```

Run the symmetry-reduced simulation and unfold its results.

## `lightsim.fem.geometry`

Finite-element solvers (`lightsim.fem`, every public name of the submodules below is also `lightsim.fem.<name>`): cross-section geometries and the gdsfactory cut.

### `lightsim.fem.geometry.Polygon`

```text
Polygon(points, holes=..., arcs=None)
```

A polygon: an outer ring and optional holes, each a list of (x, y) vertices in metres. `arcs` optionally marks segments as circular arcs (`arcs[ring][i] = (cx, cy, r)` for the segment starting at vertex i).

### `lightsim.fem.geometry.Polygon.rings`

```text
Polygon.rings()
```

The outer ring and the holes as (n, 2) arrays.

### `lightsim.fem.geometry.Polygon.contains`

```text
Polygon.contains(x, y)
```

Even-odd point-in-polygon test (vectorised over points).

### `lightsim.fem.geometry.Polygon.bounds`

Property. (x0, y0, x1, y1) of the outer ring.

### `lightsim.fem.geometry.Rectangle`

```text
Rectangle(x0, y0, x1, y1)
```

Axis-aligned rectangle with corners (x0, y0) and (x1, y1).

### `lightsim.fem.geometry.Trapezoid`

```text
Trapezoid(x_center, width_bottom, width_top, y0, y1)
```

Symmetric trapezoid (a waveguide with slanted sidewalls).

### `lightsim.fem.geometry.Circle`

```text
Circle(center, radius, n=None, size=None)
```

Circle as a polygon whose segments are marked as arcs: the mesher puts the subdivision points on the circle and P2 elements are curved. `n` (default 32) is the number of input vertices; the mesher subdivides them further to the local mesh size.

### `lightsim.fem.geometry.Region`

```text
Region(shape, name, material='', size=None, mesh_order=10)
```

A tagged region of a cross-section.

**Attributes**

- `shape` (Polygon): Its outline (see `Rectangle`, `Circle`, ...).
- `name` (str): Name used to address the region (sources, conductors, mesh sizes).
- `material` (str): Material name, looked up in the property tables of each solver.
- `size` (float or None): Target edge length inside the region (m); `None`: the default.
- `mesh_order` (int): Priority where regions overlap; the lowest value wins.

### `lightsim.fem.geometry.Geometry`

```text
Geometry(regions=None)
```

A list of regions to mesh.

>>> g = Geometry() >>> g.add(Rectangle(-2e-6, -1e-6, 2e-6, 0), "box", "SiO2")       # doctest: +SKIP

### `lightsim.fem.geometry.Geometry.add`

```text
Geometry.add(shape, name, material='', size=None, mesh_order=10)
```

Append a region; returns the geometry (calls can be chained).

### `lightsim.fem.geometry.Geometry.names`

```text
Geometry.names()
```

Region names in order of first appearance (the mesh tags).

### `lightsim.fem.geometry.Geometry.bounds`

Property. (x0, y0, x1, y1) of all regions.

### `lightsim.fem.geometry.Geometry.mesh`

```text
Geometry.mesh(
    default_size=None,
    sizes=None,
    refine=None,
    grading=0.3,
    min_angle=25.0,
    max_vertices=2000000,
)
```

Mesh the geometry.

**Parameters**

- `default_size` (float, optional): Largest edge length (m); default: a tenth of the diagonal.
- `sizes` (dict, optional): `{region name or material: edge length}`, overriding the regions' own `size`.
- `refine` (dict, optional): `{region name: edge length}`: refine along the boundary of these regions (interfaces), growing away from it by `grading`.
- `grading` (float): Size growth per unit distance away from small regions.
- `min_angle` (float): Minimum angle (degrees) of the quality refinement (at most ~30).

Returns a `lightsim.fem.Mesh`. Region names become the tags (`mesh.region_names`); regions sharing a name share a tag.

### `lightsim.fem.geometry.cut_component`

```text
cut_component(
    component,
    layer_stack=None,
    xsection_bounds=None,
    exclude=(),
    wafer_levels=None,
    u_offset=0.0,
)
```

Cut a gdsfactory component along a line through its layer stack.

`xsection_bounds = ((x0, y0), (x1, y1))` in **micrometres** (gdsfactory units, as in gplugins). Returns a list of dicts `{name, material, mesh_order, zmin, zmax, intervals, sidewall_angle, width_to_z}`: for every layer-stack level the intervals of the cut (metres, measured from the start of the line plus `u_offset`) where the level has material. Wafer levels (`background=True`, a `WAFER` layer or listed in `wafer_levels`) cover the whole line.

### `lightsim.fem.geometry.cross_section_geometry`

```text
cross_section_geometry(
    component,
    layer_stack=None,
    xsection_bounds=None,
    exclude=(),
    wafer_levels=None,
    sizes=None,
    background=None,
    u_offset=0.0,
)
```

The cross-section of a gdsfactory component along a line as a `Geometry`.

Every layer-stack level becomes regions named after the level (material from the stack, priority from its `mesh_order`), rectangles in the (u, z) plane, trapezoids where the level has a `sidewall_angle` (the drawn width applies at the fraction `width_to_z` of the thickness, and the sidewalls lean inwards for a positive angle, as in gdsfactory). `xsection_bounds` is in micrometres; the geometry is in metres with u measured from the start of the line plus `u_offset`. `background` (a material name) adds a region of lowest priority covering the bounding box, for stacks that leave gaps (air above the cladding).

## `lightsim.fem.mesh`

Triangle meshes and finite-element fields.

### `lightsim.fem.mesh.Mesh`

```text
Mesh(nodes, tris, tags, region_names=None, materials=None, edge_mid=None, _info=None)
```

A triangle mesh of a cross-section with named regions.

**Attributes**

- `nodes` (ndarray (N, 2)): Node coordinates (m).
- `tris` (ndarray (M, 3)): Counter-clockwise triangles.
- `tags` (ndarray (M,)): Region index of every triangle (into `region_names`).
- `region_names`, `materials` (list of str): Name and material of every region index.
- `edges` (ndarray (E, 2)): Global edges (lower node first); `edge_tris` (E, 2) the triangles on each (-1 for none), `tri_edges` (M, 3) the edges of each triangle (edge k joins local nodes k and k+1).
- `edge_mid` (ndarray (E, 2)): Midpoint of curved edges on the true boundary (NaN: straight).

### `lightsim.fem.mesh.Mesh.n_nodes`

Property. Number of nodes.

### `lightsim.fem.mesh.Mesh.n_tris`

Property. Number of triangles.

### `lightsim.fem.mesh.Mesh.tag`

```text
Mesh.tag(name)
```

Region index of a region name (or material name).

### `lightsim.fem.mesh.Mesh.tags_of`

```text
Mesh.tags_of(names)
```

Region indices of one name or several (names or materials).

### `lightsim.fem.mesh.Mesh.region_mask`

```text
Mesh.region_mask(names)
```

Boolean mask of the triangles of the named regions.

### `lightsim.fem.mesh.Mesh.areas`

```text
Mesh.areas(curved=False)
```

Triangle areas; with `curved=True` those of the isoparametric (curved-edge) triangles.

### `lightsim.fem.mesh.Mesh.centroids`

```text
Mesh.centroids()
```

Triangle centroids (M, 2).

### `lightsim.fem.mesh.Mesh.edge_lengths`

```text
Mesh.edge_lengths()
```

Length of every edge (E,).

### `lightsim.fem.mesh.Mesh.bounds`

Property. (x0, y0, x1, y1) of the nodes.

### `lightsim.fem.mesh.Mesh.boundary_edges`

```text
Mesh.boundary_edges(where=None)
```

Indices of outer-boundary edges.

`where`: `None` (all), `"top"`, `"bottom"`, `"left"`, `"right"` (the sides of the bounding box), a list of those, or a function `f(x, y) -> bool` of the edge midpoints.

### `lightsim.fem.mesh.Mesh.interface_edges`

```text
Mesh.interface_edges(a, b=None)
```

Edges between region(s) `a` and region(s) `b` (any other region, or the outer boundary, if `b` is None).

### `lightsim.fem.mesh.Mesh.edge_dofs`

```text
Mesh.edge_dofs(edges, order)
```

Lagrange DOFs on edges: their nodes, plus the midside DOF for P2.

### `lightsim.fem.mesh.Mesh.region_dofs`

```text
Mesh.region_dofs(names, order)
```

Lagrange DOFs of the closed triangles of named regions.

### `lightsim.fem.mesh.Mesh.dof_coords`

```text
Mesh.dof_coords(order)
```

Coordinates of the Lagrange DOFs: nodes, then edge midpoints for P2 (curved where set).

### `lightsim.fem.mesh.Mesh.ndof`

```text
Mesh.ndof(order)
```

Number of Lagrange DOFs of order 1 or 2.

### `lightsim.fem.mesh.Mesh.refine`

```text
Mesh.refine()
```

Uniformly refined mesh (each triangle split into four).

### `lightsim.fem.mesh.Mesh.submesh`

```text
Mesh.submesh(exclude=(), keep=None)
```

The mesh without the triangles of the regions in `exclude` (or with only those in `keep`); their outline becomes outer boundary (e.g. metal electrodes as electric walls of an optical mode solve). Curved edge midpoints are dropped.

### `lightsim.fem.mesh.Mesh.per_region`

```text
Mesh.per_region(values, default=None, what='value')
```

Per-triangle array from `{region name or material: value}` (region names take precedence over materials).

### `lightsim.fem.mesh.Mesh.plot`

```text
Mesh.plot(ax=None, show_regions=True, lw=0.3)
```

Plot the mesh with region colours (needs matplotlib).

### `lightsim.fem.mesh.Field`

```text
Field(mesh, order, values)
```

A scalar Lagrange (P1 or P2) field on a mesh.

Call it with points to evaluate: `f(x, y)` (arrays, metres) returns the values (NaN outside the mesh); `grad` the gradient.

### `lightsim.fem.mesh.Field.__call__`

```text
Field.__call__(x, y)
```

Values at points (arrays in m); NaN outside the mesh.

### `lightsim.fem.mesh.Field.grad`

```text
Field.grad(x, y)
```

Gradient (..., 2) at points.

### `lightsim.fem.mesh.Field.at_nodes`

```text
Field.at_nodes()
```

Values at the mesh nodes.

### `lightsim.fem.mesh.Field.per_triangle`

```text
Field.per_triangle()
```

Average over the vertices of every triangle.

### `lightsim.fem.mesh.Field.integrate`

```text
Field.integrate(region=None, degree=6)
```

Integral of the field over the mesh or over named regions.

### `lightsim.fem.mesh.Field.quadrature`

```text
Field.quadrature(degree=6)
```

Points, weights, triangle index, values and gradients at quadrature points.

### `lightsim.fem.mesh.Field.max`

```text
Field.max()
```

Largest DOF value.

### `lightsim.fem.mesh.Field.plot`

```text
Field.plot(ax=None, colorbar=True, **kw)
```

Plot the field (needs matplotlib); keyword arguments go to `tripcolor`.

### `lightsim.fem.mesh.rectangle_mesh`

```text
rectangle_mesh(x0, x1, y0, y1, nx, ny, name='domain', material='')
```

Structured triangle mesh of a rectangle (two triangles per cell).

## `lightsim.fem.heat`

Heat conduction and thermo-optic phase shifters.

### `lightsim.fem.heat.Current`

```text
Current(current, sigma=None)
```

A heater driven by a current `I` (A) along the waveguide axis through a region of electrical conductivity `sigma` (S/m, required: lightsim has no default film conductivities). The dissipated power per length is `I^2 / (sigma A)`.

### `lightsim.fem.heat.HeatResult`

```text
HeatResult(mesh, temperature, power_per_length, conductivity, order)
```

Steady temperature.

**Attributes**

- `temperature` (Field): Temperature (K), or the rise above the fixed boundary temperature when that is 0.
- `power_per_length` (dict): Dissipated power per unit length of each source region (W/m).
- `conductivity` (ndarray): Per-triangle conductivity used (W/(m K)).

### `lightsim.fem.heat.HeatResult.total_power`

Property. Total power per unit length (W/m).

### `lightsim.fem.heat.HeatResult.mean_temperature`

```text
HeatResult.mean_temperature(region)
```

Area average of the temperature over named regions.

### `lightsim.fem.heat.HeatResult.index_change`

```text
HeatResult.index_change(dndT=None, reference=0.0)
```

Per-triangle refractive-index change `dn/dT (T - reference)` (thermo-optic coefficients from the material library unless given; NaN in regions without a confirmed dn/dT).

### `lightsim.fem.heat.solve_heat`

```text
solve_heat(
    mesh,
    sources=None,
    conductivity=None,
    fixed=None,
    convection=None,
    heat_flux=None,
    order=2,
    solver='direct',
)
```

Steady heat conduction on a cross-section.

**Parameters**

- `mesh`: Mesh
- `sources` (dict): `{region: power per unit length (W/m)}`, or a `Current`, or `{"current_density": J (A/m^2), "sigma": S/m}` as in femwell.
- `conductivity` (dict, optional): `{region or material: k}` in W/(m K) (a scalar, or `(kxx, kyy)` / `(kxx, kyy, kxy)` for anisotropic media); others from the material tables (`lightsim.materials.THERMAL_PROPERTIES`, metals, air).
- `fixed` (dict, optional): `{boundary: T}`: `"bottom"`, `"top"`, `"left"`, `"right"`, a function `f(x, y)` of edge midpoints, or a region name (the whole region held at T). Default `{"bottom": 0.0}`: the result is the rise above the substrate temperature.
- `convection` (dict, optional): `{boundary: (h, T_inf)}`, h in W/(m^2 K).
- `heat_flux` (dict, optional): `{boundary: q}`, inflow in W/m^2.
- `order` (int): Lagrange order, 1 or 2.
- `solver` (str): `"direct"` (sparse Cholesky) or `"cg"` (preconditioned CG).

### `lightsim.fem.heat.TransientResult`

```text
TransientResult(times, probes, final, steady=None, snapshots=...)
```

Transient temperature: `times` (s), `probes` (`{region: mean temperature at every time}`), the final field and the steady state.

### `lightsim.fem.heat.TransientResult.time_constant`

```text
TransientResult.time_constant(region)
```

Time for the region's mean temperature to cover 1 - 1/e of the way from its initial to its steady value (s), interpolated.

### `lightsim.fem.heat.solve_heat_transient`

```text
solve_heat_transient(
    mesh,
    sources=None,
    t_end=0.0001,
    nsteps=400,
    conductivity=None,
    capacity=None,
    fixed=None,
    convection=None,
    initial=None,
    probes=(),
    order=2,
    theta=1.0,
    snapshots=(),
)
```

Transient heat conduction from `initial` (default 0, i.e. the rise after the sources switch on at t = 0) until `t_end` in `nsteps` equal steps (theta scheme: 1 = backward Euler, 0.5 = Crank-Nicolson).

`capacity`: `{region or material: rho * c_p}` in J/(m^3 K) (default from the material tables). `probes`: region names whose mean temperature is recorded every step. `initial`: a `Field`, an array of DOF values, or a function `f(x, y)`. `snapshots`: step indices to keep the whole field for.

### `lightsim.fem.heat.thermal_phase_shifter`

```text
thermal_phase_shifter(
    mesh,
    heater,
    waveguide,
    wavelength=1.55e-06,
    power_per_length=1000.0,
    neighbour=None,
    index=None,
    dndT=None,
    conductivity=None,
    fixed=None,
    convection=None,
    order=2,
    mode_window=1.5e-06,
    mode_resolution=2e-08,
    transient=False,
    t_end=0.0002,
    nsteps=400,
    capacity=None,
)
```

Thermo-optic phase shifter figures of merit.

Solves the heat equation with `power_per_length` (W/m) dissipated in the `heater` region, maps the temperature rise to an index change `dn/dT * dT` and re-solves the fundamental mode of `waveguide` with lightsim's finite-difference mode solver on a grid of `mode_resolution` around it (`mode_window` margin). The problem is linear in the power, so

`dphi/dP = 2 pi / lambda * dneff/dp` (rad/W, independent of the heater length L for a heater as long as the arm) and `P_pi = pi / (dphi/dP)`.

Returns a dict: `neff` (unheated), `dneff_dp` (per W/m), `dphi_dP` (rad/W), `P_pi` (W), `dT_waveguide_per_W_per_m`, `dT_heater_per_W_per_m`, `crosstalk` (`dneff` of the `neighbour` waveguide over that of `waveguide`), `heat` (the `HeatResult`) and, with `transient`, `time_constant` (s, the 1 - 1/e rise time of the waveguide temperature; `capacity` gives rho c_p of materials without confirmed table values, e.g. the heater).

## `lightsim.fem.electrostatics`

Electrostatics, capacitance matrices and Pockels modulators.

### `lightsim.fem.electrostatics.crystal_tensor`

```text
crystal_tensor(diag, axes=('x', 'y', 'z'))
```

(eps_uu, eps_vv, eps_ww) of a crystal-frame diagonal tensor for the crystal axes lying along (u, v, w).

### `lightsim.fem.electrostatics.rotate_tensor`

```text
rotate_tensor(eps_uu, eps_vv, angle_deg)
```

In-plane tensor (eps_uu, eps_vv, eps_uv) of a diagonal tensor rotated by `angle_deg`.

### `lightsim.fem.electrostatics.ElectrostaticResult`

```text
ElectrostaticResult(mesh, potential, permittivity, conductors)
```

Potential (V) of one solve, with the field and energy helpers.

Fields:

- `mesh`
- `potential`
- `permittivity`: per triangle (eps_uu, eps_vv, eps_uv), relative
- `conductors`

### `lightsim.fem.electrostatics.ElectrostaticResult.field`

```text
ElectrostaticResult.field(x, y)
```

Electric field E = -grad(phi) (V/m) at points, shape (..., 2).

### `lightsim.fem.electrostatics.ElectrostaticResult.field_per_triangle`

```text
ElectrostaticResult.field_per_triangle()
```

Field at the triangle centroids (M, 2).

### `lightsim.fem.electrostatics.ElectrostaticResult.energy`

```text
ElectrostaticResult.energy()
```

Electrostatic energy per unit length (J/m), from the field.

### `lightsim.fem.electrostatics.solve_electrostatics`

```text
solve_electrostatics(
    mesh,
    conductors,
    permittivity=None,
    boundary='neumann',
    charge=None,
    order=2,
)
```

Potential of conductors at fixed voltages.

**Parameters**

- `conductors` (dict): `{region name: potential (V)}`.
- `permittivity` (dict, optional): `{region or material: eps_r}`, a scalar or an in-plane tensor `(eps_uu, eps_vv)` / `(eps_uu, eps_vv, eps_uv)` (see `crystal_tensor`, `rotate_tensor`); others from the static-permittivity table.
- `boundary` (str or dict): `"neumann"` (insulating outer boundary, default), `"ground"` (outer boundary at 0 V) or `{side: V}`.
- `charge` (dict, optional): `{region: charge density (C/m^3)}`.

### `lightsim.fem.electrostatics.capacitance_matrix`

```text
capacitance_matrix(
    mesh,
    conductors,
    permittivity=None,
    ground=(),
    boundary='neumann',
    order=2,
)
```

Maxwell capacitance matrix per unit length (F/m).

`conductors`: names of the terminal regions (the matrix order); `ground`: regions always at 0 V (not terminals); `boundary` as in `solve_electrostatics`. Diagonal entries are self-capacitances, off-diagonal entries the negative mutual capacitances (the convention of gplugins' ElectrostaticResults). Returns `(C, names)`; one sparse factorisation serves all terminals.

### `lightsim.fem.electrostatics.electro_optic_vpil`

```text
electro_optic_vpil(
    mesh,
    signal,
    ground,
    crystal,
    waveguide,
    wavelength=1.55e-06,
    cut='x-cut',
    voltage=1.0,
    permittivity=None,
    index=None,
    order=2,
    mode_window=1.5e-06,
    mode_resolution=2e-08,
    mode_method='fd',
    polarization='te',
)
```

Half-wave voltage-length product of an electro-optic phase shifter.

`signal`: the electrode region at `voltage`; `ground`: region(s) at 0 V; `crystal`: the region(s) of Pockels material (LiNbO3) whose crystal axes lie along the cross-section axes as given by `cut` (a key of `CUTS` or a tuple of crystal axes along (u, v, w)). The static permittivity of the crystal follows the same axes unless given. `waveguide`: the region around which the optical mode is solved, in a window `mode_window` (m) larger on every side (or an explicit window `(x0, y0, x1, y1)`). `mode_method`: `"fd"` (lightsim's finite-difference solver on the perturbed, cell-averaged index) or `"fem"` (`lightsim.fem.solve_modes`); of the two modes of highest index the one with the larger (`"te"`) or smaller (`"tm"`) TE fraction is used. With `"fem"` the electrodes are removed from the optical problem and act as electric walls.

Returns a dict with `dneff_dV` (1/V), `VpiL` (V m), `neff`, the `ElectrostaticResult` `static` and the per-triangle crystal-frame field `E_crystal` (V/m at `voltage`).

## `lightsim.fem.modes`

Full-vectorial finite-element waveguide modes.

### `lightsim.fem.modes.eps_from`

```text
eps_from(mesh, wavelength, index=None, eps=None)
```

Per-triangle permittivity (M, 4) = (eps_xx, eps_yy, eps_xy, eps_zz), complex.

`eps`: a dict `{region or material: value}` or a per-triangle array; a value is a scalar, `(eps_xx, eps_yy, eps_zz)` (diagonal) or `(eps_xx, eps_yy, eps_xy, eps_zz)`. `index`: the same with refractive indices (scalars). Otherwise, and for regions missing from a dict, the index comes from lightsim's material library at `wavelength`.

### `lightsim.fem.modes.FemMode`

```text
FemMode(neff, freq, te_fraction, E, H, info, *, mesh, problem, x, beta, ints)
```

A mode of `solve_modes` (a `lightsim.modes.Mode`).

Extra attributes: `mesh`, `group_index` (from the field energy; material dispersion not included; also stored as `n_group`) and `power` (W, 1 after normalisation). `beta` (1/m) is the eigenvalue of the finite-element problem, `wavelength` and `effective_area()` (m^2, `(int |E|^2)^2 / int |E|^4`) are those of every `lightsim.modes.Mode`.

### `lightsim.fem.modes.FemMode.beta`

Property. Propagation constant (1/m): the eigenvalue of the finite-element problem.

### `lightsim.fem.modes.FemMode.effective_area`

```text
FemMode.effective_area()
```

Effective mode area `(int |E|^2 dA)^2 / int |E|^4 dA` (m^2), from the element integrals.

### `lightsim.fem.modes.FemMode.n_eff`

Property. Effective index (femwell's name for `neff`).

### `lightsim.fem.modes.FemMode.tm_fraction`

Property. 1 - `te_fraction`.

### `lightsim.fem.modes.FemMode.loss_db_per_m`

Property. Power loss (dB/m) from Im(neff): 20 log10(e) k0 Im(neff).

### `lightsim.fem.modes.FemMode.calculate_propagation_loss`

```text
FemMode.calculate_propagation_loss(distance)
```

Loss in dB over `distance` (m).

### `lightsim.fem.modes.FemMode.fields`

```text
FemMode.fields(x, y)
```

`(E, H)` at points (arrays in m): complex (..., 3) each, E in V/m and H in A/m for 1 W; NaN outside the mesh.

### `lightsim.fem.modes.FemMode.quadrature`

```text
FemMode.quadrature(degree=4)
```

Quadrature points, weights, triangle index, E and H (A/m) of the mode.

### `lightsim.fem.modes.FemMode.calculate_overlap`

```text
FemMode.calculate_overlap(other)
```

`1/2 int (E_self x H_other*) . z dA` over the mesh (both modes on the same mesh); 1 for a mode with itself (unit power).

### `lightsim.fem.modes.FemMode.calculate_confinement_factor`

```text
FemMode.calculate_confinement_factor(region)
```

Share of `int |E|^2` in the named region(s).

### `lightsim.fem.modes.FemMode.profile_for`

```text
FemMode.profile_for(sim, axis, position, min, max, offset=(0.0, 0.0))
```

The `profile` dict of lightsim's `ModeSource` / `ModeMonitor` for a window of a `lightsim.Simulation`, sampled from the FEM fields at the exact staggered positions the FDTD engine uses.

The cross-section coordinates (x, y) of the mesh map to the plane's in-plane axes (u, v) as `u = x + offset[0]`, `v = y + offset[1]`; the propagation direction is +`axis`. For `axis=1` the in-plane axes (x, z) are left-handed around +y, so the mode's x maps to z and its y to x as in `lightsim.solve_modes`.

### `lightsim.fem.modes.FemMode.plot`

```text
FemMode.plot(component='Ex', part='abs', ax=None, n=300)
```

Plot one field component (`Ex`, `Ey`, `Ez`, `Hx`, ..., `E` for |E|) on a regular grid (needs matplotlib).

### `lightsim.fem.modes.solve_modes`

```text
solve_modes(
    mesh,
    wavelength,
    num_modes=2,
    index=None,
    eps=None,
    n_guess=None,
    radius=None,
    bend_axis_x=0.0,
    pml=None,
    pml_strength=3.0,
    boundary='pec',
    window=None,
    grid=120,
    tol=1e-10,
    keep_pml_modes=False,
)
```

Waveguide modes of a cross-section mesh.

**Parameters**

- `mesh` (Mesh): Cross-section mesh (x horizontal, y vertical, propagation along z).
- `wavelength` (float): Vacuum wavelength (m).
- `num_modes` (int): Number of modes, highest effective index first (near `n_guess`).
- `index`, `eps`: Materials (see `eps_from`); default from the material library.
- `n_guess` (float, optional): Effective index the eigen-solver targets; default 1.05 times the largest index (the modes of highest index come first).
- `radius` (float, optional): Bend radius (m); the bend centre lies at `x = bend_axis_x - radius`, so positive x is the outside of the bend. Give the mesh a PML on the outer side to see the radiation (bend) loss.
- `pml` (float, sequence or dict, optional): Thickness of perfectly matched layers inside the mesh boundary: one value for all sides, `(left, bottom, right, top)`, or a dict. `pml_strength` is the imaginary stretch at the outer edge.
- `boundary` (str, sequence or callable): Outer boundary: `"pec"` (electric wall, default), `"pmc"` (magnetic wall, the natural condition), or the sides that are PEC (`["left", "right"]`, a function of edge midpoints).
- `window` ((x0, y0, x1, y1), optional): Region of the grid samples `Mode.E` / `Mode.H` (default: the mesh without its PML); `grid` points along the longer side.

- `keep_pml_modes` (bool): With a PML, modes with more than half of `int |E|^2` inside the layers (PML and radiation modes) are dropped unless this is True; a bend makes the outer cladding look like a high index, so give `n_guess` near the expected index (e.g. the straight guide's).

Returns a list of `FemMode`, sorted by decreasing Re(neff) (possibly fewer than `num_modes` when PML modes were dropped).

## `lightsim.fem.gplugins`

gplugins-compatible entry points of the finite-element solvers.

### `lightsim.fem.gplugins.solve_thermal`

```text
solve_thermal(
    mesh,
    thermal_conductivity=None,
    specific_conductivity=None,
    current_densities=None,
    fixed_boundaries=None,
    order=1,
    powers=None,
    convection=None,
)
```

Steady heat conduction with Joule heating, after `femwell.thermal.solve_thermal`.

`thermal_conductivity`: `{region or material: k}` in W/(m K) (others from the material tables). `specific_conductivity`: `{region: sigma}` in S/m. `current_densities`: `{region: J}` in A/m^2 (SI, unlike femwell's per-um^2 when meshing in um), heating `J^2/sigma`. `fixed_boundaries`: `{boundary: T}` (`"bottom"`, `"top"`, a function of the edge midpoints, or a region name). `powers`: `{region: W/m}` as an alternative to currents. Returns a `HeatResult` (`.temperature` is the field).

### `lightsim.fem.gplugins.compute_component_slice_thermal`

```text
compute_component_slice_thermal(
    component,
    xsection_bounds,
    layer_stack=None,
    heater='heater',
    power_per_length=None,
    current=None,
    sigma=None,
    thermal_conductivity=None,
    resolutions=None,
    default_resolution_max=0.5,
    order=2,
    fixed_boundaries=None,
    convection=None,
    exclude=('substrate',),
    background=None,
)
```

Temperature in a cross-section of a gdsfactory component.

The component is cut along `xsection_bounds` (um) through `layer_stack` (default: the active PDK's), meshed with `resolutions` (`{layer: um}`, also meshwell / femwell resolution specs), and the `heater` level dissipates `power_per_length` (W/m) or carries `current` (A) through a heater of electrical conductivity `sigma` (S/m, required with `current`). Heater alloys such as TiN have no default thermal conductivity: give it in `thermal_conductivity` (W/(m K)). The substrate level is excluded by default and the bottom of what remains is held at 0 (the result is the rise above the substrate temperature). Returns `(mesh, HeatResult)`.

### `lightsim.fem.gplugins.ElectrostaticResults`

```text
ElectrostaticResults(matrix, names, mesh, per_length)
```

Capacitance result with the fields of gplugins' `ElectrostaticResults`.

`capacitance_matrix`: `{(terminal_i, terminal_j): C}` (Maxwell convention: self-capacitances on the diagonal, negative mutual capacitances off it); `raw_capacitance_matrix`: the same as an array. Values are per unit length (F/m), or in F when a `length` (um) was given. `mesh` and `solutions` (the potential of each terminal at 1 V) replace gplugins' mesh and field file locations.

### `lightsim.fem.gplugins.run_capacitive_simulation`

```text
run_capacitive_simulation(
    component,
    xsection_bounds,
    layer_stack=None,
    material_spec=None,
    element_order=2,
    resolutions=None,
    default_resolution_max=1.0,
    conductor_refinement=None,
    terminals=None,
    ground=(),
    boundary='neumann',
    length=None,
    background='vacuum',
    exclude=(),
)
```

2-D capacitance matrix of a cross-section of a gdsfactory component, after `run_capacitive_simulation_palace` / `_elmer`.

`material_spec` follows gplugins (`{material: {"relative_permittivity": eps}}`); materials with an infinite permittivity are conductors, and every connected conductor piece of the cut is a terminal named `"<level>_<k>"` (k counts along the cut). `terminals` restricts and orders them; `ground` lists pieces held at 0 V that are not terminals. `length` (um, e.g. the component length along the cut normal) turns the per-length matrix into a total. A 2-D cut neglects the end effects a 3-D solve includes.

### `lightsim.fem.gplugins.Modes`

```text
Modes(iterable=())
```

List of `FemMode` with femwell's `Modes` conveniences (`n_effs`, `sorted`).

### `lightsim.fem.gplugins.Modes.n_effs`

Property. Effective indices of the modes.

### `lightsim.fem.gplugins.Modes.sorted`

```text
Modes.sorted(key)
```

The modes sorted by `key`.

### `lightsim.fem.gplugins.compute_component_slice_modes`

```text
compute_component_slice_modes(
    component,
    xsection_bounds,
    layer_stack=None,
    wavelength=1.55,
    num_modes=4,
    order=1,
    wafer_padding=2.0,
    radius=inf,
    metallic_boundaries=False,
    n_guess=None,
    solver='lightsim',
    material_name_to_index=None,
    resolution_specs=None,
    default_resolution_max=0.3,
    background_tag=None,
    pml=None,
    **kwargs,
)
```

Modes of a cut through a gdsfactory component, after `gplugins.femwell.compute_component_slice_modes`.

`xsection_bounds` and `wavelength` in um, `radius` in um (bend, centre on the negative-u side of the cut's midpoint). Materials: `material_name_to_index` (`{material: n}`), else lightsim's library at the wavelength. `resolution_specs`: `{layer: um}` or meshwell / femwell resolution specs. `metallic_boundaries`: electric walls on the outer boundary (default: magnetic walls, femwell's natural condition). `pml` (um): absorbing layers inside the outer boundary (leaky modes, bends). `order` 2 is not available: the elements are first-order Nedelec / P1 (femwell's `order=1`); finer meshes give the accuracy. `wafer_padding` is accepted for compatibility; the cut line sets the lateral extent. The fields of the returned modes are in SI units on a mesh in metres, with u measured from the start of the cut line.

### `lightsim.fem.gplugins.compute_cross_section_modes`

```text
compute_cross_section_modes(
    cross_section,
    layer_stack=None,
    wavelength=1.55,
    num_modes=4,
    order=1,
    radius=inf,
    wafer_padding=2.0,
    **kwargs,
)
```

Modes of a gdsfactory cross-section, after `gplugins.femwell.compute_cross_section_modes`: a straight of the cross-section is cut across its middle over its width plus `wafer_padding` (um) on each side; the other arguments go to `compute_component_slice_modes`.

## `lightsim.fem.validation`

Validation suite of the finite-element solvers.

### `lightsim.fem.validation.CaseReport`

```text
CaseReport(name, checks=..., notes=..., seconds=0.0)
```

Outcome of one validation case: checks, notes and run time.

### `lightsim.fem.validation.CaseReport.check`

```text
CaseReport.check(what, measured, expected, tol)
```

Relative check |m - e| <= tol |e| (absolute when e == 0).

### `lightsim.fem.validation.CaseReport.check_abs`

```text
CaseReport.check_abs(what, measured, expected, tol)
```

Absolute check |m - e| <= tol.

### `lightsim.fem.validation.CaseReport.passed`

Property. True if every check is within its tolerance.

### `lightsim.fem.validation.CaseReport.summary`

```text
CaseReport.summary()
```

Printable report.

### `lightsim.fem.validation.run`

```text
run(names=None, verbose=True)
```

Run the named cases (all by default) and print their summaries.

## `lightsim.sweep`

Spectra at a fixed angle of incidence from a sweep of oblique runs.

### `lightsim.sweep.AngleSweep`

```text
AngleSweep(freqs, angle, n_medium, k_t, results)
```

Result of `sweep_angle`: one run per frequency.

**Attributes**

- `freqs`: (nf,) frequencies (Hz), in the order given.
- `angle`: the angle of incidence (rad).
- `k_t`: (nf,) transverse wavevectors used (rad/m).
- `results`: list of `lightsim.spec.Results`, one per frequency.

### `lightsim.sweep.AngleSweep.flux`

```text
AngleSweep.flux(monitor)
```

Flux of `monitor` at each run's own frequency, shape (nf,).

### `lightsim.sweep.AngleSweep.source_spectrum`

```text
AngleSweep.source_spectrum(source=None)
```

Complex source spectrum S at each run's frequency, shape (nf,).

### `lightsim.sweep.AngleSweep.field`

```text
AngleSweep.field(monitor, component)
```

DFT field of `monitor` at each run's frequency, shape (nf, ...).

### `lightsim.sweep.AngleSweep.incident_power`

```text
AngleSweep.incident_power(area, source=None, amplitude=1.0)
```

Analytic power of the plane wave through `area` (m^2) of the source plane: `n |amplitude S|^2 area cos(angle) / (2 eta0)`.

### `lightsim.sweep.sweep_angle`

```text
sweep_angle(
    build,
    freqs,
    angle,
    *,
    n_medium=1.0,
    bandwidth=None,
    devices='cpu',
    progress=False,
)
```

Spectra at the fixed angle `angle` (rad) from one oblique run per frequency.

For every frequency `f` of `freqs` the sweep calls `build(f, signal, k_t)` and runs the simulation it returns, where

- `signal` is `GaussianPulse(f, bandwidth * f)`: give it to the `PlaneWave` (with `angle=angle`), whose phase gradient is then computed at `f`;
- `k_t = n_medium * 2 pi f / c * sin(angle)` is the transverse wavevector: give it to the `Bloch` boundaries along the tilt axis;
- the monitors must record `f` (they may record more frequencies; only `f` is read).

`n_medium` is the index at the source plane. `bandwidth` is the relative spectral width of the pulses; the default keeps the pulse well above `f sin(angle)`, below which the fixed `k_t` is evanescent, and is at most 0.1. `devices` is a device string or a list of them; the runs are dispatched concurrently, one at a time per entry (`["cuda:0", "cuda:1"]`; an entry may itself be `"cuda:0,1"`).

Returns an `AngleSweep`; `sweep.flux(name)` gives the flux of a monitor at each run's frequency, `sweep.incident_power(area)` the analytic incident power.

**Sweep or one fixed-angle run.** A plane wave with `fixed_angle=True` (periodic boundaries along the tilt axis) gives the whole spectrum at one angle from a single run, with the same accuracy as the sweep's runs at their frequencies; for more than a few frequency points it is the cheaper choice (see the sources guide for timings). Prefer the sweep when the scene uses something the fixed-angle mode refuses: nonlinear, gain, gyrotropic, director-field or off-diagonal tensor media, sources other than the one plane wave, time-domain or far-field monitors, conical incidence (a tilt along both transverse axes), a cell split over several processes or nodes (several GPUs of one machine work when x is the propagation axis), or gradients; or when only one or two frequencies are needed.

## `lightsim.materials`

Dispersion library and pole fitting.

### `lightsim.materials.available_materials`

```text
available_materials()
```

Canonical names of the library materials.

### `lightsim.materials.material_info`

```text
material_info(name)
```

`{"name", "range": (wl_min_m, wl_max_m), "reference"}` of a library entry.

### `lightsim.materials.permittivity`

```text
permittivity(name, wavelength_m, strict=False)
```

Relative permittivity `eps = (n + i k)^2` of a library material.

Same arguments as `refractive_index`.

### `lightsim.materials.refractive_index`

```text
refractive_index(name, wavelength_m, strict=False)
```

Complex refractive index `n + i k` of a library material.

`name` is case-insensitive (`"SiO2"`, `"Si3N4"`, `"Si"`, `"LiNbO3_o"`, `"LiNbO3_e"`, `"Al2O3_o"`, `"Al2O3_e"`, `"GaAs"`, `"Ge"`; see the module docstring for the sources). `wavelength_m` is the vacuum wavelength in metres, scalar (returns `complex`) or array (returns a complex `ndarray`). Outside the published validity range a warning is issued (`strict=True` raises instead).

### `lightsim.materials.pole_permittivity`

```text
pole_permittivity(medium_or_fit, omega, dt=None, axis=0)
```

Permittivity of a pole model at angular frequency `omega` (rad/s).

`medium_or_fit` is a `lightsim.spec.Medium` dict or the dict returned by `fit_poles`. With `dt` given, the *discrete-time* permittivity seen by the ADE update with time step `dt` is returned instead (same expression as `Pole::chi_discrete` in the core); use it to compare FDTD results with the model at coarse time steps.

### `lightsim.materials.gyrotropic_permittivity`

```text
gyrotropic_permittivity(medium, omega, dt=None)
```

3 x 3 complex permittivity tensor of a medium with gyrotropic poles (`Medium(gyro=...)`) at angular frequency `omega` (rad/s), fields as exp(-i w t).

Each pole adds chi = s (D I + i w [b x])^-1 with D = omega0^2 - w^2 - i gamma w, s = delta_eps omega0^2 (Lorentz) or omega_p^2 (Drude, omega0 = 0) and [b x] the matrix of v -> b x v. For a bias along z, chi_xy = -chi_yx = i s w b / (D^2 - w^2 b^2), and the circular polarizations (1, +-1j, 0) see eps_inf + s / (D +- w b). With `dt` given, the tensor of the discrete update with time step `dt` is returned instead (as `GyroPole::chi_discrete` in the core). Ordinary `poles` and `sigma` of the medium are added on the diagonal.

### `lightsim.materials.fit_poles`

```text
fit_poles(
    wavelengths_m,
    n,
    k=None,
    n_lorentz=2,
    drude=False,
    eps_inf=None,
    *,
    name='fitted',
    weights=None,
    omega0_max=None,
    n_starts=8,
    seed=0,
)
```

Fit `eps(w) = eps_inf + sum Lorentz (+ Drude)` to sampled `n` (and `k`).

The fit is a bounded nonlinear least squares on the complex permittivity error relative to `|eps|` (times optional per-sample `weights`), with several deterministic starting points. The bounds `delta_eps >= 0`, `gamma >= 0`, `omega0 > 0`, `omega_p >= 0` and `eps_inf >= 1` make every pole passive (`Im eps >= 0` at all frequencies). The explicit ADE update of strong poles on a small `eps_inf` is stable only below the Courant step of the mesh; the solver reduces the time step when a medium needs it.

**Parameters**

- `wavelengths_m`, `n`, `k`: Sample vacuum wavelengths (m), refractive index and extinction coefficient. `n` may be complex (then `k` must be `None`). With no absorption data at all (`k` is `None` and `n` real) the poles are fitted lossless (`gamma = 0` exactly).
- `n_lorentz`, `drude`: Number of Lorentz poles and whether to add one Drude pole.
- `eps_inf`: Fix the high-frequency permittivity; `None` fits it (>= 1). For transparent-window data the best fit can end at 1 with one strong ultraviolet pole (silicon); a fixed larger value, up to about half the in-band permittivity, avoids the time-step reduction that such a medium needs, for a slightly larger fit error.
- `name`: Name of the returned medium.
- `omega0_max`: Upper bound on the Lorentz resonance frequencies (rad/s); default 6x the highest sampled frequency. A pole far above the band forces a small time step, so the default keeps the fit usable down to about 10 cells per shortest vacuum wavelength of the band; raise it for a closer fit on fine grids, lower it for coarser ones.
- `n_starts`, `seed`: Number of starting points (the first is structured, the rest are drawn from a generator seeded with `seed`; the fit is deterministic).

**Returns**

dict with keys `eps_inf`, `lorentz` (list of `{"delta_eps", "omega0", "gamma"}`), `drude` (`{"omega_p", "gamma"}` or `None`), `poles` (spec pole dicts), `medium` (a ready `lightsim.spec.Medium`), `max_n_error` / `max_k_error` (max `|n - n_fit|`, `|k - k_fit|` over the samples), `rms_eps_error` (relative) and `band` (wl_min, wl_max).

### `lightsim.materials.fitted_medium`

```text
fitted_medium(
    name,
    band=(1.2e-06, 1.7e-06),
    n_lorentz=2,
    *,
    drude=False,
    eps_inf=None,
    samples=101,
    medium_name=None,
    tol=0.001,
    return_fit=False,
    **fit_kwargs,
)
```

Library data -> pole fit -> `Medium` for the wavelength `band` (m).

Samples the library formula for `name` at `samples` points uniform in frequency over `band = (wl_min, wl_max)`, runs `fit_poles` and returns the `lightsim.spec.Medium` dict (named `medium_name`, default the library name). A warning is issued if the fit misses the data by more than `tol` in `n` (`None` disables the check). With `return_fit=True` the full `fit_poles` dict is returned instead.

Uniaxial crystals are fitted one ray at a time (`"LiNbO3_o"`, `"LiNbO3_e"`, ...); the result is an isotropic medium for that ray.

### `lightsim.materials.thermal_properties`

```text
thermal_properties(name)
```

Thermal and thermo-optic properties of a library material (see `THERMAL_PROPERTIES`): `{"k", "cp", "rho", "dndT", "reference"}`; `None` where no source was confirmed.

### `lightsim.materials.electro_optic_coefficients`

```text
electro_optic_coefficients(name)
```

Pockels coefficients (m/V) and permittivities of a material (see `ELECTRO_OPTIC`; LiNbO3 only).

## `lightsim.sheets`

Sheet materials: surface conductivity and graphene as zero-thickness sheets or thin layers.

### `lightsim.sheets.drude_sheet_conductivity`

```text
drude_sheet_conductivity(freq, drude_weight, gamma)
```

Surface conductivity `1j * D / (w + 1j * gamma)` (S) of a Drude sheet.

`freq` in Hz, `drude_weight` `D` in S/s (siemens per second), `gamma` the collision rate in rad/s.

### `lightsim.sheets.graphene_conductivity`

```text
graphene_conductivity(
    freq,
    fermi_level_eV,
    scattering_time,
    temperature=300.0,
    interband=True,
)
```

Kubo surface conductivity of graphene (S), at frequencies `freq` (Hz).

`sigma = sigma_intra + sigma_inter` with (Falkovsky, J. Phys. Conf. Ser. 129, 012004, 2008; Hanson, J. Appl. Phys. 103, 064302, 2008, in the `exp(-i w t)` convention)

- `sigma_intra = 1j * D / (w + 1j / tau)`, `D = (2 e^2 kT / (pi hbar^2)) ln(2 cosh(Ef / 2kT))`, which is `e^2 Ef / (pi hbar^2)` at T = 0;
- `sigma_inter = (e^2 / 4 hbar) [H(w/2) + (4j w / pi) int_0^inf (H(x) - H(w/2)) / (w^2 - 4 x^2) dx]`, `H(x) = sinh(hbar x/kT) / (cosh(Ef/kT) + cosh(hbar x/kT))`; at T = 0 this is `(e^2/4 hbar) [step(hbar w - 2 Ef) + (1j/pi) ln|(hbar w - 2 Ef) / (hbar w + 2 Ef)|]`.

The interband term is computed without broadening. `scattering_time` `tau` is in s, `temperature` in K. Spatial dispersion and magnetic fields are not included.

### `lightsim.sheets.DrudeSheetPole`

```text
DrudeSheetPole(drude_weight, gamma)
```

A Drude term `sigma = 1j * D / (w + 1j * gamma)` of a surface conductivity: `drude_weight` D in S/s, `gamma` in rad/s.

### `lightsim.sheets.LorentzSheetPole`

```text
LorentzSheetPole(strength, omega0, gamma)
```

A Lorentz term `sigma = -1j * w * s / (w0**2 - w**2 - 1j * gamma * w)` of a surface conductivity: `strength` s in S rad/s (the conductance at resonance is s / gamma), `omega0` and `gamma` in rad/s.

### `lightsim.sheets.SheetConductivity`

```text
SheetConductivity(conductance=0.0, susceptibility=0.0, poles=(), name='')
```

A surface conductivity for a zero-thickness `lightsim.Sheet`:

`sigma_s(w) = conductance - 1j * w * eps0 * susceptibility + sum(poles)`

in siemens, with `conductance` (S) a constant conductance, `susceptibility` (m) a constant surface susceptibility and `poles` from `DrudeSheetPole` and `LorentzSheetPole`. All terms must be passive (non-negative).

### `lightsim.sheets.sigma_of`

```text
sigma_of(conductivity, freq)
```

`sigma_s` (S) of a `SheetConductivity` dict at `freq` (Hz).

### `lightsim.sheets.sheet_conductivity`

```text
sheet_conductivity(
    name='sheet',
    *,
    conductance=0.0,
    drude_weight=None,
    gamma=0.0,
    sigma=None,
    freqs=None,
    band=None,
    n_lorentz=2,
    drude=True,
    samples=81,
    return_fit=False,
    **fit_kwargs,
)
```

A surface conductivity for a zero-thickness `lightsim.Sheet`.

Give a constant `conductance` (S), and/or a Drude sheet (`drude_weight` D in S/s and `gamma` in rad/s, `sigma = 1j D / (w + 1j gamma)`, exact at every frequency), or `sigma`: a callable `sigma(freq_Hz)` with `band = (f_min, f_max)`, or an array on `freqs`, fitted with `n_lorentz` Lorentz terms, (`drude=True`) one Drude term and a constant susceptibility; the fit holds in the band only. Returns the `SheetConductivity` dict, or with `return_fit=True` a dict with `conductivity`, `fit` and `max_sigma_error` (the largest relative error over the samples, 0 for exact terms).

### `lightsim.sheets.sheet`

```text
sheet(axis, position, conductivity, **kwargs)
```

`lightsim.Sheet`: a zero-thickness sheet carrying `conductivity`.

### `lightsim.sheets.sheet_layer_conductivity`

```text
sheet_layer_conductivity(medium, freq, thickness, eps_background=1.0, axis=0)
```

Surface conductivity that a layer of `medium` and `thickness` carries.

`sigma_s = -1j * eps0 * w * t * (eps(w) - eps_background)`: the inverse of the mapping used by `sheet_medium`, to check a fit against the target conductivity.

### `lightsim.sheets.sheet_rt`

```text
sheet_rt(freq, sigma_s, n1=1.0, n2=1.0, angle=0.0, polarization='s')
```

Analytic power reflection and transmission of a sheet.

The sheet lies between media of indices `n1` (incidence side) and `n2`. `sigma_s` is a complex array over `freq`, a callable of `freq` or a `SheetConductivity` dict. At normal incidence `r = (n1 - n2 - eta0 sigma) / (n1 + n2 + eta0 sigma)`, `t = 2 n1 / (n1 + n2 + eta0 sigma)`, `T = (n2/n1)|t|^2`; for a free-standing sheet `T = |2 / (2 + eta0 sigma)|^2`. At `angle` (rad, in medium 1) the indices become the wave admittances `n cos(theta)` (`polarization="s"`) or `n / cos(theta)` (`"p"`). Returns `(R, T)`.

### `lightsim.sheets.sheet_medium`

```text
sheet_medium(
    name,
    thickness=None,
    *,
    drude_weight=None,
    gamma=0.0,
    sigma=None,
    freqs=None,
    band=None,
    eps_background=1.0,
    n_lorentz=2,
    drude=True,
    samples=81,
    return_fit=False,
    **fit_kwargs,
)
```

A sheet conductivity: zero-thickness (`thickness=None`, the default) or as a medium whose layer of `thickness` (m) carries it.

Without `thickness` this is `sheet_conductivity`: the returned surface conductivity goes into `lightsim.Sheet`. With a `thickness`, give either

- `drude_weight` (S/s) and `gamma` (rad/s): the Drude sheet `sigma_s = 1j * D / (w + 1j * gamma)`, mapped exactly onto one Drude pole with `omega_p**2 = D / (eps0 * thickness)`; or
- `sigma`: a callable `sigma(freq_Hz) -> complex` (S) with `band = (f_min, f_max)` in Hz, or an array of values on `freqs`. The layer permittivity `eps_b + 1j sigma / (eps0 w t)` is fitted with `n_lorentz` Lorentz poles and (`drude=True`) one Drude pole by `lightsim.materials.fit_poles`, so the result holds in that band only, with the fit error reported by `return_fit=True`.

`eps_background` is the permittivity of the layer without the sheet (1 for a free-standing sheet). Place the layer as a box exactly `thickness` thick, on a mesh whose cell normal to the sheet is `thickness`, centred on the grid nodes of the tangential E components (dispersive media are staircased, not smoothed). Returns the `lightsim.spec.Medium` dict; with `return_fit=True` a dict with the medium, the fit (or `None`) and the largest relative conductivity error over the fitted samples (0 for a Drude sheet).

### `lightsim.sheets.graphene`

```text
graphene(
    fermi_level_eV,
    scattering_time,
    thickness=None,
    temperature=300.0,
    *,
    band=None,
    name='graphene',
    eps_background=1.0,
    n_lorentz=2,
    samples=81,
    return_fit=False,
    **fit_kwargs,
)
```

Graphene from the Kubo conductivity: a zero-thickness surface conductivity (`thickness=None`, the default, for `lightsim.Sheet`) or a layer of `thickness` (m).

`fermi_level_eV` is the chemical potential (eV, its sign does not matter), `scattering_time` the intraband relaxation time (s), `temperature` in K. The intraband term is one exact Drude pole, valid at every frequency. With `band = (f_min, f_max)` (Hz) the interband term of `graphene_conductivity` is added as `n_lorentz` Lorentz poles fitted over that band (and only valid there); without `band` it is left out, which is accurate well below the interband threshold `2 Ef / h` (compare the two terms with `graphene_conductivity`).

Returns the `lightsim.spec.Medium` dict, or with `return_fit=True` a dict with the medium, the interband fit and the largest relative error of the layer's sheet conductivity against the Kubo formula over the band.

## `lightsim.plugins`

User-defined material models (plug-ins) written in Python.

### `lightsim.plugins.register`

```text
register(name, after, before=None, state=0)
```

Register Python callables as the material plug-in `name` (replacing one of that name). `state` is the number of state values per cell. Returns `name`.

### `lightsim.plugins.unregister`

```text
unregister(name)
```

Remove a registered plug-in.

### `lightsim.plugins.register_examples`

```text
register_examples()
```

Register the Rust example plug-ins of `lightsim_core::plugin::examples`: `"example_lorentz"` (params `delta_eps`, `omega0`, `gamma`), `"example_kerr"` (`chi3`) and `"example_runaway"` (`gain`).

### `lightsim.plugins.use`

```text
use(name, params=None)
```

The `plugin` entry of a `lightsim.Medium`: the registered plug-in `name` with `params` (JSON-serialisable, handed to the callables as `ctx["params"]`).

## `lightsim.eo`

Free-carrier plasma dispersion in silicon.

### `lightsim.eo.soref_bennett`

```text
soref_bennett(dN, dP, wavelength=1.55e-06, model='soref1987')
```

Carrier-induced index and absorption change of silicon.

`dN` / `dP`: free electron / hole densities in **cm^-3** (scalars or arrays, >= 0). `wavelength` in metres, within 30 nm of 1310 or 1550 nm. Returns `(dn, dalpha)`: the (negative) index change and the absorption change in **cm^-1** (power attenuation coefficient).

### `lightsim.eo.alpha_to_sigma`

```text
alpha_to_sigma(alpha_per_m, n)
```

Conductivity (S/m) giving a power absorption `alpha` (1/m) in a medium of index `n`: sigma = n \* alpha \* eps0 \* c (valid for kappa << n).

### `lightsim.eo.carrier_index_grid`

```text
carrier_index_grid(
    min,
    max,
    dN,
    dP,
    *,
    wavelength=1.55e-06,
    n_base=None,
    axes=None,
    model='soref1987',
    mask=None,
)
```

An `IndexGrid` holding `(n_base + dn)**2`.

`dN` / `dP` are 1-D, 2-D or 3-D carrier maps in cm^-3 on a regular grid whose nodes span `[min, max]` (see `IndexGrid` for `axes`; a (y, z) cross-section of a waveguide along x is `axes=(1, 2)`). `n_base` is the unperturbed index: a scalar (default: silicon at `wavelength`) or an array of the same shape, e.g. the silicon/oxide index map of the cross-section. `mask` (boolean, same shape) restricts the perturbation to the silicon; elsewhere the index stays `n_base`.

Only the real index change is carried by the grid. For the free-carrier loss give the structure a conductive medium, see `carrier_medium`.

### `lightsim.eo.carrier_medium`

```text
carrier_medium(
    name,
    dN,
    dP,
    *,
    wavelength=1.55e-06,
    n_base=None,
    model='soref1987',
    weights=None,
)
```

Medium to pair with `carrier_index_grid`.

An `IndexGrid` takes one (spatially uniform) conductivity from its medium, so the free-carrier absorption is represented by its average over the map (optionally weighted, e.g. by the optical mode intensity `weights`): sigma = n \* &lt;dalpha> \* eps0 \* c. The medium's `eps` is `n_base**2` (used by the auto-mesher only; the grid replaces it).

### `lightsim.eo.COEFFICIENTS`

Plasma-dispersion coefficients `(A_e, p_e, A_h, p_h, B_e, q_e, B_h, q_h)` by model name and wavelength in nm; densities in cm^-3, absorption in cm^-1.

### `lightsim.eo.N_SILICON`

Refractive index of undoped silicon by wavelength in nm: `{1550: 3.476, 1310: 3.503}`.

## `lightsim.gratings`

Diffraction-order analysis.

### `lightsim.gratings.GratingOrders`

Floquet-order decomposition on one plane.

All per-order arrays have shape `(nf, n_orders)`.

**Attributes**

- `freqs`: (nf,) analysed frequencies (Hz).
- `orders`: (n_orders, 2) integer order indices (m, n) along `period_axes[0]` and `period_axes[1]` (n = 0 for a 1-D grating).
- `k_t`: (nf, n_orders, 2) transverse wavevector along the two in-plane axes `(axis+1) % 3` and `(axis+2) % 3` (rad/m).
- `k_n`: complex normal wavevector magnitude sqrt(k^2 - |k_t|^2) (rad/m).
- `propagating`: True where the order is propagating in `n_medium`.
- `theta`: polar angle between the order's wavevector and the plane normal (radians, 0..pi/2; nan for evanescent orders). The same angle applies to the forward (+axis) and backward (-axis) wave of the order.
- `phi`: azimuth of `k_t` measured from in-plane axis `(axis+1) % 3` towards `(axis+2) % 3` (radians).
- `angles`: (nf, n_orders, 2) signed angles asin(k_t / k) along the two in-plane axes (the usual grating-equation angle of a 1-D grating).
- `power_forward`, `power_backward` (power carried along +axis / -axis through): the unit cell (both >= 0; 0 for evanescent orders).
- `power`: net power along +axis (forward - backward).
- `power_s_forward`, `power_p_forward`, `power_s_backward`, `power_p_backward`: split into s (E perpendicular to the plane of incidence, "TE") and p (H perpendicular, "TM"). At exactly normal incidence of an order the labels are a convention: s = E along `(axis+2) % 3`.
- `amp_s_forward`, `amp_s_backward` (complex s amplitudes (E, V/m) referred to): the plane of the tangential E samples and to the transverse origin r_t = 0.
- `amp_p_forward`, `amp_p_backward`: complex p amplitudes (eta0 \* H).
- `fraction`: power fraction per order, see `diffraction_orders`.
- `norm`: (nf,) the power `fraction` is normalised to.
- `area`: unit-cell area of the monitor plane (m^2).

### `lightsim.gratings.GratingOrders.power`

Property. Net power along +axis per order, `power_forward - power_backward`, in watts times the squared source spectrum; shape (nf, n_orders).

### `lightsim.gratings.GratingOrders.total_forward`

Property. Power travelling along +axis summed over the orders, shape (nf,).

### `lightsim.gratings.GratingOrders.total_backward`

Property. Power travelling along -axis summed over the orders (positive), shape (nf,).

### `lightsim.gratings.GratingOrders.total`

Property. Net power along +axis summed over the orders, shape (nf,).

### `lightsim.gratings.GratingOrders.index`

```text
GratingOrders.index(m, n=0)
```

Column index of order (m, n).

### `lightsim.gratings.GratingOrders.efficiency`

```text
GratingOrders.efficiency(m, n=0)
```

`fraction` of order (m, n) over frequency, shape (nf,).

### `lightsim.gratings.GratingOrders.as_dict`

```text
GratingOrders.as_dict(fi=0)
```

{(m, n): fraction} of the propagating orders at frequency index `fi`.

### `lightsim.gratings.grating_angles`

```text
grating_angles(freq, period, n_medium=1.0, k_inc=0.0, orders=None)
```

Grating equation for a 1-D grating.

Returns `(orders, angles)` with `sin(angle_m) = (k_inc + 2 pi m / period) / (n k0)` for the propagating orders (or the requested `orders`; nan when evanescent). `k_inc` is the in-plane (Bloch) wavevector of the incident wave, `n_inc * k0 * sin(theta_inc)`.

### `lightsim.gratings.diffraction_orders`

```text
diffraction_orders(
    results,
    monitor,
    axis,
    period_axes,
    freqs=None,
    n_medium=1.0,
    kx_inc=0.0,
    ky_inc=0.0,
    *,
    sim,
    incident_power=None,
    direction='net',
    max_order=None,
)
```

Decompose the fields on a DFT monitor plane into diffraction orders.

**Parameters**

- `results`: `lightsim.Results`.
- `monitor`: name of a `DFTMonitor` lying in a plane normal to `axis` (zero thickness along `axis`) and spanning exactly one period of the simulation along each of `period_axes`; it must record the four tangential components of E and H.
- `axis`: plane normal (0, 1, 2).
- `period_axes`: the periodic in-plane axis (1-D grating) or both in-plane axes (2-D grating / crossed grating). With a single axis the fields are averaged along the other in-plane axis (the invariant direction of a 2-D-equivalent simulation). The simulation must use periodic or Bloch boundaries on these axes and the cell must be one period.
- `freqs`: frequencies to analyse (must be monitor frequencies); `None` analyses all monitor frequencies.
- `n_medium`: real refractive index of the homogeneous medium at the plane.
- `kx_inc`, `ky_inc`: in-plane wavevector of the incident wave (rad/m) along `period_axes[0]` and `period_axes[1]`, i.e. the `k` of the `Bloch(k)` boundaries (0 for plain periodic boundaries). Order (m, n) has k_t = (kx_inc + 2 pi m / Lx, ky_inc + 2 pi n / Ly). A scalar is frequency independent, as in a Bloch simulation; an array gives one value per analysed frequency, as for a fixed-angle run (`PlaneWave(..., fixed_angle=True)`), where `kx_inc = 2 pi f n sin(angle) / C0`.
- `sim`: the `lightsim.Simulation` that produced `results` (or the node arrays returned by `Simulation.grid()`); provides the sample coordinates and the period.
- `incident_power`: optional (nf,) or scalar power to normalise `fraction` to -- typically the source power from `incident_power` on an empty reference run. If `None`, `fraction` is normalised to the total power through the plane in the chosen `direction` (sum over the propagating orders), so the fractions sum to 1.
- `direction`: which power `fraction` refers to: `"net"` (forward minus backward, default), `"+"` (waves travelling along +axis) or `"-"` (along -axis; reported as a positive number).
- `max_order`: by default only orders that propagate at one of the analysed frequencies are returned. Give an int (or one per period axis) to return all orders with |m| <= max_order instead, evanescent included.

**Returns**

`GratingOrders`

### `lightsim.gratings.incident_power`

```text
incident_power(
    reference,
    monitor,
    axis,
    period_axes,
    freqs=None,
    n_medium=1.0,
    kx_inc=0.0,
    ky_inc=0.0,
    *,
    sim,
    direction='+',
)
```

Source power per unit cell from an empty reference run.

`reference` is the result of the same simulation without the grating (homogeneous medium `n_medium` at the monitor); the power travelling along `direction` through `monitor` is returned, shape (nf,).

### `lightsim.gratings.grating_efficiencies`

```text
grating_efficiencies(
    results,
    t_monitor,
    r_monitor,
    axis,
    period_axes,
    freqs=None,
    n_t=1.0,
    n_r=1.0,
    kx_inc=0.0,
    ky_inc=0.0,
    *,
    sim,
    incident_power,
    direction='+',
    max_order=None,
)
```

Transmission and reflection diffraction efficiencies of a grating cell.

`t_monitor` lies behind the grating (medium `n_t`), `r_monitor` in front of it (medium `n_r`); the incident wave travels along `direction` of `axis`. `incident_power` is the source power per unit cell (see `incident_power`). Transmission counts the waves travelling along `direction` at `t_monitor`, reflection those travelling against it at `r_monitor` (the incident wave itself is excluded by the direction split, so `r_monitor` may sit on either side of the source plane).

Returns a dict with

- `"T"`, `"R"`: `GratingOrders` whose `fraction` is the diffraction efficiency of each order (power / incident power),
- `"T_total"`, `"R_total"`: (nf,) sums over the propagating orders.

## `lightsim.resonance`

Harmonic inversion and Lorentzian fits.

### `lightsim.resonance.Resonance`

```text
Resonance(freq, decay, Q, amplitude, error, t_ref=0.0)
```

One damped mode `amplitude * exp[(-2 pi i freq - decay) (t - t_ref)]`.

Fields:

- `freq`: frequency (Hz)
- `decay`: amplitude decay rate gamma (1/s); energy decays as exp(-2 gamma t)
- `Q`: quality factor pi \* freq / decay (inf for an undamped mode)
- `amplitude`: complex amplitude at `t_ref`
- `error`: relative error estimate of the complex frequency (see `harminv`)
- `t_ref`: time the amplitude refers to (first analysed sample)

### `lightsim.resonance.Resonance.omega`

Property. Complex angular frequency `2 pi freq - i decay` (`exp(-i omega t)` convention).

### `lightsim.resonance.harminv`

```text
harminv(
    signal,
    dt,
    fmin,
    fmax,
    nmax=100,
    *,
    t_start=0.0,
    t_end=None,
    t0=0.0,
    order=None,
    sv_tol=None,
    max_error=0.001,
    stability=0.1,
    noise_factor=1.5,
    min_amplitude=0.0001,
    min_Q=0.0,
    oversample=2.5,
    stopband_db=100.0,
    max_pencil=400,
    diagnostics=None,
)
```

Harmonic inversion of a uniformly sampled time series.

Finds the damped sinusoids `a exp[(-2 pi i f - gamma)(t - t_ref)]` with `fmin <= f <= fmax` contained in `signal` (real or complex; a real signal is the sum of each mode and its complex conjugate, so `abs(amplitude)` is half the peak amplitude of the real oscillation).

Method: the signal is mixed down to the band centre, low-pass filtered with a linear-phase Kaiser FIR and decimated to a sample rate `oversample * (fmax - fmin)` — a FIR filter maps a sum of damped exponentials to a sum of the same exponentials, so this loses no accuracy while averaging the noise and shrinking the problem — and the poles are then extracted with the SVD-truncated matrix-pencil method (Hua & Sarkar 1990), the least-squares relative of the filter-diagonalization method used by Harminv. Amplitudes come from a linear least-squares fit, corrected for the filter response at the complex pole.

**Parameters**

- `signal`: sequence of float or complex, sample `n` taken at `t0 + n*dt`.
- `dt`: sampling interval.
- `fmin`, `fmax`: frequency window (same units as `1/dt`). Keep it reasonably tight around the modes of interest; modes outside are ignored.
- `nmax`: maximum model order, i.e. number of exponentials fitted in the band-limited signal (the modes in the window plus a few around it).
- `t_start`, `t_end`: analyse only `t_start <= t <= t_end`. Set `t_start` after the excitation has died out (see `source_end_time`) — while the source is on the signal is not a sum of damped exponentials.
- `t0`: time of `signal[0]`.
- `order`: force the model order (number of exponentials in the decimated band, including out-of-window and noise poles). Default: automatic from the singular-value spectrum.
- `sv_tol`: keep singular values above `sv_tol * s_max` instead of the automatic noise-floor estimate.
- `noise_factor`: safety factor on the estimated noise singular values in the automatic order selection (larger: fewer, more certain modes).
- `max_error`: discard modes whose `error` exceeds this.
- `stability`: discard modes whose complex angular frequency differs between the two estimates by more than `stability * max(decay, 2 pi / T)` (`T` the analysed duration), i.e. that are not pinned down to within a fraction of their own linewidth. This is what rejects noise poles.
- `min_amplitude`: discard modes weaker than this fraction of the strongest. Modes outside `[fmin, fmax]` further than `(oversample - 1) * (fmax - fmin)` from its edges fold back into the window attenuated by `stopband_db`; anything weaker than that relative to the strongest out-of-window line cannot be told from such an image.
- `min_Q`: discard modes with `Q` below this (growing modes, `Q < 0`, are always discarded).
- `oversample`: decimated sample rate over `fmax - fmin` (> 1).
- `stopband_db`: attenuation of the anti-alias filter.
- `max_pencil`: cap on the pencil parameter (size of the SVD).
- `diagnostics`: optional dict that is filled with what was done: `samples`, `duration`, `decimation`, `decimated_samples`, `pencil`, `order`, `poles_in_window` (poles that carry signal), `unstable` (those that moved between the two pencil sizes), `rejected` and `closest_spacing`.

A `UserWarning` is issued when no mode is returned and when a quarter or more of the poles in the window are not reproducible, with the reason as far as it can be told. The usual one is a record too short for the line spacing: noisy lines are separated reliably only if the record covers two of their beat periods, `duration >= 2 / spacing` (two round trips, for the comb of a ring).

**Returns**

list of `Resonance`, sorted by frequency. `error` is the relative change `|d omega| / |omega|` of the complex frequency between two independent pencil sizes: a mode that is really in the data is stable to many digits, a spurious (noise) pole moves by a fraction of the bandwidth. It is an order-of-magnitude indicator of the frequency error, not a bound.

### `lightsim.resonance.source_end_time`

```text
source_end_time(sim_or_signal, cutoff=6.0)
```

Time after which a pulsed excitation is negligible.

Accepts a signal dict (`GaussianPulse(...)`), a source dict or a `Simulation` (the latest of its sources). For a Gaussian pulse `exp(-(t - t_off)^2 / 2 tau^2)` this is `t_off + cutoff * tau` (envelope down to `exp(-cutoff^2 / 2)`: 1.5e-8 at the default 6).

### `lightsim.resonance.ring_down`

```text
ring_down(
    results,
    probe,
    component='Ez',
    fmin=None,
    fmax=None,
    nmax=100,
    *,
    t_start=None,
    sim=None,
    point=None,
    **kwargs,
)
```

Harmonic inversion of a `ProbeMonitor` time series.

`results` is the `Results` of a run, `probe` and `component` select the recorded series. `t_start` is where the analysis begins; if omitted and `sim` (the `Simulation`) is given it defaults to `source_end_time` of its sources, otherwise to 0 (only correct if the record contains no excitation). `fmin` / `fmax` default to the band `freq0 -+ fwidth` of the first source of `sim`. A `FieldTimeMonitor` works too: pick one of its (flattened) cells with `point`. Remaining keyword arguments go to `harminv`.

### `lightsim.resonance.LorentzianFit`

```text
LorentzianFit(
    center,
    fwhm,
    Q,
    amplitude,
    background,
    slope,
    residual,
    center_err=nan,
    fwhm_err=nan,
)
```

`background + slope (f - center) + amplitude / (1 + ((f - center) / (fwhm/2))^2)`.

Fields:

- `center`: resonance frequency
- `fwhm`: full width at half maximum (of the peak or dip)
- `Q`: center / fwhm
- `amplitude`: peak height above the background (negative for a dip)
- `background`: baseline at `center`
- `slope`: linear baseline slope
- `residual`: rms residual over the rms of the data minus its mean
- `center_err`: 1-sigma uncertainty of `center` from the fit covariance
- `fwhm_err`: 1-sigma uncertainty of `fwhm`

### `lightsim.resonance.LorentzianFit.__call__`

```text
LorentzianFit.__call__(f)
```

Evaluate the fitted model, background plus slope plus Lorentzian, at the frequencies `f` (same unit as the fitted `freqs`). Returns an array of the shape of `f`.

### `lightsim.resonance.lorentzian_fit`

```text
lorentzian_fit(freqs, spectrum, dip=None, linear_background=True)
```

Fit one Lorentzian line to a power spectrum (transmission dip or peak).

`freqs` and `spectrum` are real arrays covering a single resonance — typically `|E(f)|^2` or a transmission `T(f)` from DFT / flux monitors, cropped around the line. `dip` forces a dip (True) or a peak (False); by default the sign is chosen from the data. The baseline is a constant plus, with `linear_background`, a slope.

The width resolved by a DFT of a record of length `T` cannot be narrower than `~1/T`: use this on spectra of runs that have decayed (or on normalised transmission), and `harminv` on truncated ring-downs.

## `lightsim.gui`

Graphical interface: a local web server to inspect a simulation, run it and read its results.

### `lightsim.gui.serve`

```text
serve(sim=None, results=None, *, host='127.0.0.1', port=0, ports=None, title=None)
```

Start the interface in a background thread and return the `Server` (`.url`, `.stop()`).

`sim` is a `Simulation`, a spec dict or a JSON string; `results` a `Results` (or its `raw` dict). A `Results` passed as `sim` is taken as the results. `ports` is an optional list of port dicts (as `lightsim.gds.port_info` returns) drawn on the cuts. `host` other than a loopback address makes the server reachable from other machines; the URL then carries a random access token.

### `lightsim.gui.view`

```text
view(
    sim=None,
    results=None,
    *,
    host='127.0.0.1',
    port=0,
    open_browser=None,
    block=False,
    ports=None,
    title=None,
)
```

Open the interface for a simulation and/or its results.

Starts a server (see `serve`), prints its URL and returns it. `open_browser` opens the URL in the default browser; by default it does so when a display is available and the session is not over SSH. `block=True` serves until interrupted (Ctrl-C), as a script that ends would otherwise stop the server with it.

### `lightsim.gui.wait`

```text
wait()
```

Serve until interrupted (Ctrl-C), then stop every server.

### `lightsim.gui.stop`

```text
stop()
```

Stop every server started by `view` or `serve`.
