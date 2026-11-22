# JSON format

This page lists every field of the JSON file that describes a simulation and
of the JSON file that holds its results. Use it to write simulation files by
hand or from another tool, and to read results without Python.

The simulation file is what `Simulation.to_json()` writes and what
`lightsim run` reads (see [Command line](../guide/cli.md)). The physical
meaning of each object is explained in the guide pages; this page gives the
exact names, types and defaults.

> **Note:** The format is the serialised form of the solver's own data
> structures at version 0.1.0. The file carries no format version, and the
> format may change between versions. Keys that are not recognised are
> **ignored**, so a misspelt optional key has no effect and produces no
> error. A missing required key, a wrong type or an unknown `type` value is an
> error.

Conventions: all quantities are in SI units. A "vector" is an array of three
numbers for x, y, z. Axes are the integers 0, 1, 2. Field components are the
strings `"Ex"`, `"Ey"`, `"Ez"`, `"Hx"`, `"Hy"`, `"Hz"`, with exactly this
capitalisation. A direction is `"+"` or `"-"`. Objects that come in several
kinds carry a `"type"` key that selects the kind.

## Simulation

| Key | Type | Default | Meaning |
|---|---|---|---|
| `region` | object | required | `{"min": vector, "max": vector}` |
| `mesh` | object | required | See Mesh |
| `boundaries` | object | required | `{"x": [low, high], "y": [low, high], "z": [low, high]}`, see Boundary |
| `run_time` | number | required | Simulated time in seconds |
| `name` | string | `""` | Label |
| `background` | string | `"vacuum"` | Name of the background medium |
| `media` | array | `[]` | Medium objects |
| `structures` | array | `[]` | Structure objects, painted in order |
| `sources` | array | `[]` | Source objects |
| `monitors` | array | `[]` | Monitor objects |
| `shutoff` | number or null | null | Early-stop threshold on the field energy |
| `shutoff_interval` | integer | 200 | Steps between energy checks |
| `courant` | number | 0.99 | Fraction of the stability limit |
| `subpixel` | boolean | true | Sub-pixel smoothing |
| `dft_stride` | integer or null | null | Steps between samples of the frequency-domain monitors; null chooses it per monitor from the bandwidth of the sources (see [Monitors](../guide/monitors.md)) |
| `dimensions` | integer | 3 | 3, or 2 for a simulation whose fields do not depend on the coordinate along `invariant_axis` |
| `invariant_axis` | integer or string | 2 | 0, 1, 2 or `"x"`, `"y"`, `"z"`; used when `dimensions` is 2 |
| `polarization` | string | `"both"` | `"te"` (E in the plane, H along the invariant axis), `"tm"` (E along the invariant axis, H in the plane) or `"both"`; other than `"both"` only when `dimensions` is 2 |
| `force_3d_kernels` | boolean | false | Run a 2-D simulation through the general 3-D update |
| `fixed_angle` | boolean | false | Broadband oblique incidence at a fixed angle: the `angle` of the single `plane_wave` source holds at every frequency of its signal; needs `periodic` boundaries along the tilt axis (see [Sources](../guide/sources.md#broadband-oblique-incidence-at-a-fixed-angle)) |
| `sheets` | array | `[]` | Zero-thickness conducting sheets, see Sheet |

In a 2-D file (`"dimensions": 2`) every vector still has three numbers. The
region's extent along the invariant axis is the thickness of the slab the
model stands for; the grid has one cell there whatever `mesh` says and is
periodic whatever `boundaries` says. Structures are cut by the plane through
the middle of that extent, and the coordinates of sources and monitors along
the invariant axis are ignored. See
[2-D, 2.5-D and 3-D](../guide/dimensions.md).

A complete minimal file:

```json
{
  "region": {"min": [0, 0, 0], "max": [1e-6, 1e-6, 1e-6]},
  "mesh": {"type": "uniform", "dx": [50e-9, 50e-9, 50e-9]},
  "boundaries": {
    "x": [{"type": "pml"}, {"type": "pml"}],
    "y": [{"type": "periodic"}, {"type": "periodic"}],
    "z": [{"type": "pec"}, {"type": "pmc"}]
  },
  "run_time": 1e-14
}
```

The same box as a 2-D file, invariant along z, one metre thick, with the
field set that has the electric field along z:

```json
{
  "region": {"min": [0, 0, -0.5], "max": [1e-6, 1e-6, 0.5]},
  "mesh": {"type": "uniform", "dx": [50e-9, 50e-9, 1]},
  "boundaries": {
    "x": [{"type": "pml"}, {"type": "pml"}],
    "y": [{"type": "periodic"}, {"type": "periodic"}],
    "z": [{"type": "periodic"}, {"type": "periodic"}]
  },
  "run_time": 1e-14,
  "dimensions": 2,
  "invariant_axis": "z",
  "polarization": "tm"
}
```

## Mesh

| `type` | Keys |
|---|---|
| `uniform` | `dx`: vector of target spacings |
| `auto` | `wavelength` (required); `steps_per_wavelength` (20); `grading` (1.3); `min_step`, `max_step` (null); `overrides` (`[]`), each `{"min": vector, "max": vector, "dx": vector}`; `symmetry` (`[null, null, null]`): per axis, the coordinate of a mirror plane of the mesh |
| `explicit` | `x`, `y`, `z`: arrays of node coordinates, at least two each |

## Boundary

| `type` | Keys |
|---|---|
| `pml` | `layers` (12); `params`: `{"order": 3, "sigma_scale": 1, "kappa_max": 2, "alpha_max": 0}`; `inside` (false) |
| `periodic` | none |
| `bloch` | `k`: wavevector component along the axis, rad/m |
| `pec` | none |
| `pmc` | none |

`periodic` and `bloch` must be used on both faces of an axis.

## Medium

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | `""` | Name used by structures and `background` |
| `eps` | vector | `[1, 1, 1]` | Relative permittivity along x, y, z ($\varepsilon_\infty$ if there are poles) |
| `mu` | vector | `[1, 1, 1]` | Relative permeability |
| `sigma` | vector | `[0, 0, 0]` | Conductivity, S/m |
| `eps_off` | vector | `[0, 0, 0]` | Off-diagonal permittivity (xy, xz, yz) |
| `sigma_off` | vector | `[0, 0, 0]` | Off-diagonal conductivity (xy, xz, yz) |
| `poles` | array | `[]` | Pole objects |
| `pole_axes` | array of 3 booleans | all true | Axes the poles act on |
| `oriented` | array | `[]` | `{"pole": Pole, "u": vector}`, a pole acting along `u` |
| `chi2`, `chi3` | vector | zeros | Nonlinear susceptibilities, in $\text{m}/\text{V}$ and $\text{m}^2/\text{V}^2$ |
| `raman` | object | absent | Delayed part of `chi3`: `{"fraction": f, "omega": rad/s, "delta": rad/s}` |
| `director` | object | absent | Director field of a uniaxial medium: `{"eps_o", "eps_e", "min", "max", "dims": [nx, ny, nz], "director": flat array of 3 * nx * ny * nz numbers}`, sample (i, j, k) at position (i * ny + j) * nz + k |
| `gain` | object | absent | Four-level gain medium: `{"density", "omega", "gamma", "tau32", "tau21", "tau10", "pump_rate" (0), "coupling" (optional), "initial" ([1, 0, 0, 0])}` |
| `gain2` | object | absent | Four-level two-electron gain medium (Chang and Taflove): `{"density", "omega_a", "omega_b", "gamma_a", "gamma_b", "t30", "t32", "t21", "t10", "initial" ([density, density, 0, 0], m^-3), "kappa_a", "kappa_b" (optional)}` |
| `eps_complex` | array | absent | Complex relative permittivity tensor given directly, `[row][column][re, im]` (3 × 3 × 2), Hermitian and positive definite; the run uses complex fields |
| `gyro` | array | `[]` | `{"pole": Pole, "bias": vector}`, a Lorentz or Drude pole in a static bias; `bias` is the gyration vector in rad/s |
| `plugin` | object | absent | User-defined material plug-in: `{"name": registered name, "params": any JSON (null)}` |
| `pec` | boolean | false | Perfect electric conductor |

In the file, `eps`, `mu`, `sigma`, `chi2` and `chi3` are always arrays of
three numbers. The single-number forms, the refractive index `n` and the
`rotation` matrix exist only in the Python `Medium` builder, which converts
them.

| Pole `type` | Keys |
|---|---|
| `lorentz` | `delta_eps`, `omega0`, `gamma` |
| `drude` | `omega_p`, `gamma` |
| `debye` | `delta_eps`, `tau` |

Media that are referred to but not defined are looked up among the built-in
names (see [Materials](../guide/materials.md)).

## Structure and shapes

A structure is `{"name": string, "shape": Shape, "medium": string}`. `name`
is optional.

| Shape `type` | Keys |
|---|---|
| `box` | `min`, `max`: vectors |
| `sphere` | `center`: vector; `radius` |
| `cylinder` | `center`: vector; `radius`; `length`; `axis` |
| `poly_slab` | `vertices`: array of `[u, v]` pairs; `slab_min`, `slab_max`; `axis` (2) |
| `index_grid` | `min`, `max`: vectors; `dims`: `[nx, ny, nz]`; `eps`: flat array of `nx*ny*nz` samples, x-major; optional `eps_axes`: three such arrays |

For `poly_slab` the vertex coordinates are on the two axes other than `axis`,
in increasing axis order. For `index_grid`, sample `(i, j, k)` is at position
`(i * ny + j) * nz + k` of the array, and a dimension of 1 means "uniform
along that axis".

## Sheet

A zero-thickness conducting sheet (see [Sheet materials](../modules/sheets.md)):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `axis` | integer or string | required | Axis normal to the sheet: 0, 1, 2 or `"x"`, `"y"`, `"z"` |
| `position` | number | required | Coordinate of the sheet along `axis` |
| `conductivity` | object | required | `{"conductance": S (0), "susceptibility": m (0), "poles": [...] ([]), "name": ""}` |
| `min`, `max` | vector | absent | Corners of a rectangular patch (the components along `axis` are ignored); without them and without `polygon` the sheet fills the plane |
| `polygon` | array | absent | Vertices `[u, v]` of a polygonal patch, on the two other axes in increasing order |
| `name` | string | `""` | Label used in error messages |

| Sheet pole `type` | Keys | $\sigma_s(\omega)$ |
|---|---|---|
| `drude` | `drude_weight` (S/s), `gamma` (rad/s) | $iD/(\omega + i\gamma)$ |
| `lorentz` | `strength` (S rad/s), `omega0`, `gamma` (rad/s) | $-i\omega s/(\omega_0^2 - \omega^2 - i\gamma\omega)$ |

## Source

All sources accept `name` (string, default `""`) and `amplitude` (number,
default 1), and require `signal`.

| `type` | Other keys |
|---|---|
| `point_dipole` | `position`: vector; `polarization`: component |
| `plane_wave` | `axis`; `position`: number; `direction`; `polarization`; `angle` (0); `min`, `max` (null or vectors) |
| `gaussian_beam` | `axis`; `position`; `direction`; `polarization`; `center`: vector; `waist`; `focus_distance` (0); `theta` (0), `phi` (0): polar angle from the plane normal and azimuth of the beam axis, in radians |
| `mode` | `axis`; `position`; `direction`; `min`, `max`: vectors; `profile`: Mode profile |
| `tfsf` | `min`, `max`: vectors; `axis`; `direction`; `polarization` |

| Signal `type` | Keys |
|---|---|
| `gaussian_pulse` | `freq0`, `fwidth` (required); `offset` (5); `phase` (0); `amplitude` (1) |
| `continuous_wave` | same as `gaussian_pulse` |
| `ricker` | `freq0` (required); `offset` (1.5, in periods of `freq0`); `amplitude` (1) |
| `custom` | `times`, `values`: arrays of equal length |

### Mode profile

A mode profile is normally produced by the Python mode solver
(`Mode.profile`). Its fields:

| Key | Type | Meaning |
|---|---|---|
| `neff` | number | Real part of the effective index |
| `neff_imag` | number | Imaginary part (default 0) |
| `dims` | `[nu, nv]` | Cells of the mode window along its two in-plane axes |
| `e_re`, `e_im` | two arrays each | Real and imaginary parts of the two transverse electric components, each a flat array of `nu*nv` values, row-major |
| `h_re`, `h_im` | two arrays each | The same for the magnetic components, in A/m, half a cell before the plane |
| `freq` | number | Frequency at which the profile was computed |
| `h_offset` | number or null | The half cell (m) between the plane and the magnetic components, as used by the solver. Default: that of the plane the profile is used on |

The in-plane axes are (y, z), (x, z) and (x, y) for a plane normal to x, y
and z. The profile is normalised to unit power.

## Monitor

All monitors require `name`. Names must be unique within a simulation,
across monitor types; a duplicate is an error. Source names must be unique
as well.

| `type` | Other keys |
|---|---|
| `dft_field` | `min`, `max`: vectors; `components`: array; `freqs`: array; `stride` (1); `apodization` |
| `flux` | `axis`; `position`; `min`, `max`; `freqs`; `apodization` |
| `flux_box` | `min`, `max`; `freqs`; `apodization` |
| `mode` | `axis`; `position`; `min`, `max`; `freqs`; `modes`: array of mode profiles; `apodization` |
| `far_field` | `min`, `max`; `freqs`; `theta`, `phi`: arrays of angles in radians; `n_medium` (1); `apodization` |
| `probe` | `position`: vector; `components`: array |
| `field_time` | `min`, `max`; `components`; `interval` (1); `start` (0) |
| `energy` | `min`, `max`; `interval` (1) |

`apodization` is optional: `{"start": seconds, "end": seconds, "width":
seconds}`, each key optional.

## Results

`lightsim run SPEC.json --out FILE` writes one JSON object.

| Key | Content |
|---|---|
| `dt` | Time step |
| `nsteps` | Number of steps planned from `run_time` |
| `stats` | `backend`, `cells`, `steps` (steps executed), `seconds`, `mcells_per_s`, `dt`, `stopped_early`, and `energy_steps` with `energy`: the sum of the squared field values over the grid that the shutoff test looks at, every `shutoff_interval` steps |
| `dft` | Per frequency-domain monitor: `{"freqs": [...], "fields": [...]}` |
| `flux` | Per `flux`, `flux_box`, `mode` and `far_field` monitor: array of power per frequency |
| `modes` | Per `mode` monitor: `freqs`, `amp_plus_re`, `amp_plus_im`, `amp_minus_re`, `amp_minus_im` (each `[mode][frequency]`) and `flux` |
| `far_field` | Per `far_field` monitor: `freqs`, `theta`, `phi`, `flux`, and the flat arrays `[frequency][theta][phi]` `intensity` (W/sr) and `e_theta_re`, `e_theta_im`, `e_phi_re`, `e_phi_im` (far-field amplitudes in volts) |
| `time` | Per `probe` and `field_time` monitor: `{"steps": [...], "times": [...], "fields": [...]}` |
| `energy` | Per `energy` monitor: `steps`, `times` (s) and `energy` (J) |
| `source_spectra` | Per source: `[freqs, real parts, imaginary parts]` of the signal spectrum, with `freqs` in ascending order |
| `gain_levels` | With gain media only: the number of atoms in levels 0, 1, 2, 3 at the end of the run, summed over all gain cells |
| `sources` | Per source: `name`, `signal` (as in the specification) and `analytic` (true if the complex signal was injected into complex fields) |

Each entry of a `fields` array describes one component:

| Key | Content |
|---|---|
| `comp` | Component name |
| `origin` | Grid index of the first sample, on the full grid including absorbing layers |
| `dims` | `[n0, nx, ny, nz]`, with `n0` the number of frequencies or of time samples |
| `re`, `im` | Flat arrays of the complex values (frequency-domain monitors) |
| `data` | Flat array of the values (time-domain monitors) |

The element `(f, i, j, k)` of a flat array is at position
`((f * nx + i) * ny + j) * nz + k`. Values are single precision. Magnetic
fields are multiplied by the impedance of free space (see
[Concepts](../guide/concepts.md)).

Points to know:

- The `fields` of flux, mode and far-field monitors are the fields on the
  monitor surfaces from which the power was computed. A component can occur
  more than once, for the two magnetic layers on either side of a plane.
- A source without a name is listed in `source_spectra` as `source0`,
  `source1` and so on. The spectra are given at the union of the frequencies
  of all frequency-domain monitors, **sorted in ascending order**, and are
  absent if there is no such monitor. This is not the order of a monitor
  that lists its frequencies differently or records only some of them: match
  the entries by frequency value before dividing monitor data by them. In
  Python, `Results.source_spectrum(name, monitor=...)` does this.
