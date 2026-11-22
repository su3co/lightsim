# Python API

Every public function, class and constant of the `lightsim` Python package, with its signature and description. This page is generated from the package by `docs/gen_api.py` (package version 0.1.0). For explanations and worked examples see the guide pages.

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
    poles=(),
    pole_axes=(True, True, True),
    pec=False,
    chi2=0.0,
    chi3=0.0,
    eps_off=(0.0, 0.0, 0.0),
    sigma_off=(0.0, 0.0, 0.0),
    rotation=None,
    oriented=(),
)
```

A (possibly anisotropic, dispersive, conductive, nonlinear) medium.

`chi2` (m/V) and `chi3` (m^2/V^2) are instantaneous per-axis susceptibilities: P_i/eps0 = chi_lin E_i + chi2 E_i^2 + chi3 E_i^3 (Lumerical's chi2/chi3 material convention).

`eps_off` = (eps_xy, eps_xz, eps_yz) and `sigma_off` add off-diagonal terms to symmetric permittivity / conductivity tensors. `rotation` (3x3, columns = crystal axes in lab coordinates) rotates the diagonal `eps`/`n` and `sigma` into the lab frame (tilted uniaxial crystals, lossy or not).

`oriented` = [(pole, u), ...] adds dispersive poles acting along unit vectors `u` (chi(w) u u^T), e.g. `[(Lorentz(1.0, w0, g), (1, 1, 0))]` for a dispersive crystal whose optic axis is rotated in the xy plane.

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
PML(layers=12, order=3.0, sigma_scale=1.0, kappa_max=2.0, alpha_max=0.0, inside=False)
```

CPML boundary. `inside=True` puts the layers on the outermost cells of the region (FDTDX convention) instead of adding them outside.

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
Mode(neff, freq, te_fraction, profile, E, H, info)
```

A waveguide mode returned by `solve_modes`.

**Attributes**

- `neff` (complex): Effective index; a positive imaginary part is loss.
- `freq` (float): Frequency (Hz) the mode was solved at.
- `te_fraction` (float): Share of the transverse electric field energy in the first in-plane axis, between 0 and 1.
- `profile` (dict): The field pattern in the form `ModeSource` and `ModeMonitor` take, normalised to unit power.
- `E` (tuple): `(E_u, E_v)`: the two transverse electric components, complex arrays over the mode window (V/m for unit power).
- `H` (tuple): `(H_u, H_v)` in A/m, at the plane of `E`.
- `info` (dict): The mode window: `u_nodes`, `v_nodes` (m), `dims`, `axes`, `origin`, the permittivity maps `eps_u`, `eps_v`, `eps_n` (and `eps_*_im`), `coord` and `h_offset`.

Fields:

- `neff`
- `freq`
- `te_fraction`
- `profile`: ModeProfile dict for lightsim sources/monitors
- `E`: (E_u, E_v) complex arrays on the plane (natural order)
- `H`: (H_u, H_v) complex arrays (A/m) at the E plane
- `info`

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
)
```

Solve the transverse eigenproblem on a (right-handed x, y, propagation) cross-section. `eps_*` are (Nx, Ny) arrays at the Yee positions. Returns (neffs, Ex, Ey, Hx, Hy) with H normalized (H~ = eta0 H).

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
)
```

Solve waveguide modes on a plane of a `lightsim.Simulation`.

Returns a list of `Mode` sorted by effective index (or by TE fraction first when `sort="te"`).

In a 2-D simulation the plane is a line and the modes are those of the 1-D solver `lightsim.eim.solve_port_mode`, for the polarization(s) the simulation contains: `te_fraction` is 1 for the modes with E in the plane and 0 for those with E along the invariant axis.

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

### `lightsim.modes.ETA0`

Impedance of free space, 376.730313412 ohm.

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

**Parameters**

- `sim`: the base `lightsim.Simulation` (fixed structures, mesh, boundaries, run time). Its sources and monitors are ignored.
- `grids`: list of `DesignGrid` (or one). They are appended after the structures of `sim`, so the design covers what lies under it; no structure of `sim` may be meant to cover the design.
- `inputs`: excitations: `Port` (mode 0), `(Port, mode_index)` or `Excitation`.
- `outputs`: output channels: `Port` or `(Port, mode_index)`.
- `freqs`: the K frequencies (Hz).
- `devices`: a device string or a list of them. The N + M runs of one `solve` are independent and are dispatched concurrently, one at a time per list entry (`["cuda:0", "cuda:1"]`; an entry may itself be a multi-GPU string such as `"cuda:0,1"`).
- `fwidth`: bandwidth of the Gaussian pulses (default: the span of `freqs`, at least 10 % of the centre frequency).
- `monitors`: extra DFT field monitors (`lightsim.DFTMonitor` dicts) added to every forward run (recorded at `freqs`); their fields, per unit excitation, are returned in `SMatrixSolution.fields[n][name][component]` (field maps of the final design: by linearity any input vector is a weighted sum).
- `normalize`: `"monitor"` (default) measures the amplitude launched by every mode source with two mode monitors just downstream and refers it to the source plane, so S is exactly "per unit incident mode amplitude"; `"source"` divides by the source signal spectrum only.
- `engine`: `"python"` (every run driven from Python, products in NumPy) or `"native"` (the persistent Rust session of `lightsim.autodiff`: buffers kept on the device across `solve` calls, adjoint product and transposed rasterisation in the engine). Default: `$LIGHTSIM_INVDES_ENGINE` or `"python"`.

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

### `lightsim.invdes.SMatrixProblem.native_session`

```text
SMatrixProblem.native_session()
```

The `lightsim.autodiff.GradientSession` behind `engine="native"` (built on first use).

### `lightsim.invdes.SMatrixProblem.yee_permittivity`

```text
SMatrixProblem.yee_permittivity(eps)
```

What the solver uses: the permittivity at the Yee positions of every design monitor box interpolated from the grids (zero where the grid does not own the voxel). Needs one earlier `solve` with `engine="python"`; the native session reports the same through `native_session().yee_permittivity()`.

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
DesignGrid(min, max, dims, medium, name='design')
```

One index grid of the design: samples `dims = (nx, ny, nz)` spanning the box `[min, max]` (see `lightsim.IndexGrid`); a 2-tuple means one sample along z (a layer extruded through `[zmin, zmax)`). `medium` is the medium dict that carries the grid (its `eps` should be the largest permittivity the grid can take: the mesher uses it).

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

### `lightsim.invdes.matrix_fidelity`

```text
matrix_fidelity(S, target, weights=None)
```

Matrix overlap with a target T[m, n], averaged over frequency:

```text
L = mean_k |<T, S_k>|^2 / ||T||^4,     <T, S> = sum_mn conj(T_mn) S_mn.
```

L = 1 exactly when `S_k = exp(i phi_k) T` (one free global phase per frequency): it is the product of the normalised fidelity `|<T,S>|^2 / (||T||^2 ||S||^2)` and the throughput `||S||^2 / ||T||^2`, so shape errors and loss are both penalised.

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

`polarization` = `"te"` / `"tm"` selects the 1-D lateral solver of a 2-D / 2.5-D simulation (`lightsim.eim.solve_port_mode`); `None` uses `lightsim.solve_modes` (the full-vectorial solver in 3-D). `direction` points into the device.

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

## `lightsim.invdes.torch`

The S-matrix as a PyTorch function.

### `lightsim.invdes.torch.SMatrixFunction`

```text
SMatrixFunction(*args, **kwargs)
```

### `lightsim.invdes.torch.SMatrixFunction.forward`

```text
SMatrixFunction.forward(ctx, rho, layer, beta, eta)
```

### `lightsim.invdes.torch.SMatrixFunction.backward`

```text
SMatrixFunction.backward(ctx, grad_S)
```

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
- `kx_inc`, `ky_inc`: in-plane wavevector of the incident wave (rad/m) along `period_axes[0]` and `period_axes[1]`, i.e. the `k` of the `Bloch(k)` boundaries (0 for plain periodic boundaries). Order (m, n) has k_t = (kx_inc + 2 pi m / Lx, ky_inc + 2 pi n / Ly). They are frequency independent, as in a Bloch simulation.
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
