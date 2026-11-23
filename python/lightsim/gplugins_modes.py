"""Mode-solver functions with the names, arguments (micrometres) and return
types of gplugins, computed with lightsim's finite-difference solver.

Two families, as in gplugins:

* ``gplugins.modes`` (MPB): :func:`find_modes_waveguide`,
  :func:`find_modes_coupler`, :func:`find_neff_vs_width`,
  :func:`find_mode_dispersion`, :func:`find_neff_ng_dw_dh`,
  :func:`find_coupling_vs_gap`, :func:`coupling_length`,
  :func:`innerprod_trapz`. The geometry is that of gplugins'
  ``get_mode_solver_rib`` / ``get_mode_solver_coupler``: a core of
  ``core_thickness`` centred on z = 0 in ``clad_material``, an optional
  slab below it, a window of ``sy`` by ``sz``, at least ``resolution``
  cells per micrometre (the grid is adjusted so that every interface lies
  on a node). MPB's window is periodic; here its walls are closed (the
  field is zero there), which matters only if the mode reaches the wall.
  ``extrapolate=True`` (lightsim only) applies the Richardson extrapolation
  of :meth:`lightsim.xsection.CrossSection.solve`.
* ``gplugins.tidy3d.modes`` (Tidy3D's mode solver): :class:`Waveguide`,
  :class:`WaveguideCoupler`, :func:`sweep_n_eff`, :func:`sweep_n_group`,
  :func:`sweep_fraction_te`, :func:`sweep_mode_area`,
  :func:`sweep_bend_mismatch`, :func:`sweep_coupling_length`.

``parity`` arguments (MPB symmetry selectors) are accepted and ignored: the
solver returns all modes, highest effective index first. Materials are
indices, (n, k) pairs, lightsim library names or functions of the
wavelength in micrometres (gplugins' convention).
"""

from __future__ import annotations

import itertools
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

from .modes import Mode, mode_overlap
from .xsection import CrossSection, graded_nodes

um = 1e-6
nm = 1e-3  # in micrometres, as gplugins.modes defines it


def _mat(spec):
    """gplugins material spec -> lightsim CrossSection material."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return complex(spec[0], spec[1])
    if callable(spec) and not isinstance(spec, str):
        f = lambda wl: spec(wl / um)  # noqa: E731  (gplugins functions take micrometres)
        f.__name__ = getattr(spec, "__name__", "material")
        return f
    return spec


@dataclass
class GMode:
    """A mode in the form gplugins' ``gplugins.modes.types.Mode`` has
    (micrometres): ``mode_number`` (1 = fundamental), ``wavelength``,
    ``neff``, ``ng``, ``fraction_te``, ``fraction_tm``, ``effective_area``
    (µm²), the fields on the window and ``y``, ``z`` (µm). ``mode`` holds the
    lightsim :class:`~lightsim.modes.Mode`."""

    mode_number: int
    wavelength: float
    neff: float
    ng: float | None = None
    fraction_te: float | None = None
    fraction_tm: float | None = None
    effective_area: float | None = None
    E: np.ndarray | None = None
    H: np.ndarray | None = None
    eps: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    mode: Mode | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"Mode{self.mode_number}"


def _gmode(m: Mode, number: int, wavelength: float) -> GMode:
    f = m.fields_at_centres()
    E = np.stack([f["En"], f["Eu"], f["Ev"]], axis=-1)[:, :, None, :]  # (y, z, 1, xyz): x = propagation
    H = np.stack([f["Hn"], f["Hu"], f["Hv"]], axis=-1)[:, :, None, :]
    eps = np.asarray(m.info["eps_n"])
    return GMode(
        mode_number=number,
        wavelength=wavelength,
        neff=float(np.real(m.neff)),
        ng=m.n_group,
        fraction_te=m.te_fraction,
        fraction_tm=1 - m.te_fraction,
        effective_area=m.effective_area() / um**2,
        E=E,
        H=H,
        eps=eps,
        y=f["u"] / um,
        z=f["v"] / um,
        mode=m,
    )


def _mpb_cross_section(
    core_width: float = 0.45,
    core_thickness: float = 0.22,
    slab_thickness: float = 0.0,
    core_material=3.47,
    clad_material=1.44,
    nslab=None,
    sy: float = 2.0,
    sz: float = 2.0,
    resolution: int = 32,
    sidewall_angle: float | None = None,
    gap: float | None = None,
    core_widths=None,
    gaps=None,
    ymargin: float = 2.0,
    sidewall_angles=None,
    single_waveguide: bool = True,
) -> CrossSection:
    """The geometry of gplugins' get_mode_solver_rib / _coupler (µm in, m out)."""
    core, clad = _mat(core_material), _mat(clad_material)
    slab = _mat(nslab) if nslab is not None else core
    xs = CrossSection(background=clad)
    if slab_thickness > 0:
        xs.add_layer(-slab_thickness * um, 0.0, slab, name="slab")
    if single_waveguide:
        cores = [(0.0, core_width)]
        half = sy / 2
    else:
        widths = list(core_widths) if core_widths is not None else [core_width, core_width]
        gl = list(gaps) if gaps is not None else [gap if gap is not None else 0.2]
        total = sum(widths) + sum(gl)
        x = -total / 2
        cores = []
        for i, w in enumerate(widths):
            cores.append((x + w / 2, w))
            x += w + (gl[i] if i < len(gl) else 0)
        half = total / 2 + ymargin
    angs = sidewall_angles if sidewall_angles is not None else sidewall_angle
    for i, (c, w) in enumerate(cores):
        a = angs[i] if isinstance(angs, (list, tuple)) else (angs or 0.0)
        xs.add_trapezoid(c * um, w * um, -core_thickness / 2 * um, core_thickness / 2 * um, core, sidewall_angle=a, name="core")
    step = 1.0 / resolution
    # cells of at most 1/resolution with every interface on a node
    xs.window = ((-half * um, -sz / 2 * um), (half * um, sz / 2 * um))
    xs.mesh(xs.window, step=step * um, max_step=step * um)
    return xs


def find_modes_waveguide(
    tol: float = 1e-6,
    wavelength: float = 1.55,
    mode_number: int = 1,
    parity=None,
    cache_path=None,
    overwrite: bool = False,
    single_waveguide: bool = True,
    nmodes: int = 4,
    group_index: bool = False,
    extrapolate: bool = False,
    **kwargs,
) -> dict[int, GMode]:
    """Modes of a strip or rib waveguide (gplugins.modes.find_modes_waveguide).

    Returns ``{mode_number: GMode, ...}`` for ``nmodes`` modes starting at
    ``mode_number`` (1 = highest effective index). Keyword arguments as
    gplugins: ``core_width``, ``core_thickness``, ``slab_thickness``,
    ``core_material``, ``clad_material``, ``nslab``, ``sy``, ``sz``,
    ``resolution``, ``sidewall_angle`` (degrees), and for couplers
    ``core_widths``, ``gaps``, ``gap``, ``ymargin``, ``sidewall_angles``.
    ``tol``, ``parity``, ``cache_path`` and ``overwrite`` are accepted for
    compatibility and not used.
    """
    xs = _mpb_cross_section(single_waveguide=single_waveguide, **kwargs)
    n_total = mode_number - 1 + nmodes
    ms = xs.solve(wavelength * um, n_total, group_index=group_index, extrapolate=extrapolate)
    return {mode_number + i: _gmode(m, mode_number + i, wavelength) for i, m in enumerate(ms[mode_number - 1 :])}


def find_modes_coupler(**kwargs) -> dict[int, GMode]:
    """Supermodes of coupled waveguides (gplugins.modes.find_modes_coupler):
    ``core_widths``, ``gaps`` (µm) as in gplugins."""
    return find_modes_waveguide(single_waveguide=False, **kwargs)


def _cached(filepath, overwrite, compute):
    if filepath and not overwrite and pathlib.Path(filepath).exists():
        import pandas as pd

        return pd.read_csv(filepath)
    df = compute()
    if filepath:
        p = pathlib.Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


def find_neff_vs_width(width1: float = 0.2, width2: float = 1.0, steps: int = 12, nmodes: int = 4, wavelength: float = 1.55, parity=None, filepath=None, overwrite: bool = False, **kwargs):
    """Effective index against core width (gplugins.modes.find_neff_vs_width):
    a DataFrame with columns ``1 .. nmodes`` (effective index of each mode)
    and ``width`` (µm)."""
    import pandas as pd

    def compute():
        width = np.linspace(width1, width2, steps)
        neff = {i: [] for i in range(1, nmodes + 1)}
        for w in width:
            modes = find_modes_waveguide(wavelength=wavelength, nmodes=nmodes, core_width=w, **kwargs)
            for i in range(1, nmodes + 1):
                neff[i].append(modes[i].neff)
        df = pd.DataFrame(neff)
        df["width"] = width
        return df

    return _cached(filepath, overwrite, compute)


def find_mode_dispersion(wavelength: float = 1.55, wavelength_step: float = 0.01, core: str = "Si", clad: str = "SiO2", mode_number: int = 1, **kwargs) -> GMode:
    """Effective and group index with material dispersion
    (gplugins.modes.find_mode_dispersion).

    As in gplugins, the modes are solved at ``wavelength`` and
    ``wavelength ± wavelength_step`` with the core and cladding indices of
    those wavelengths, ``ng = n_c - lambda (n_+ - n_-) / (2 step)`` and the
    returned ``neff`` is the mean of the three. The materials are lightsim
    library names (``"Si"``: Li 1980, ``"SiO2"``: Malitson 1965).
    """
    from .xsection import _index_of

    ns = []
    for wl in (wavelength - wavelength_step, wavelength, wavelength + wavelength_step):
        m = find_modes_waveguide(wavelength=wl, mode_number=mode_number, nmodes=1, core_material=_index_of(core, wl * um).real, clad_material=_index_of(clad, wl * um).real, **kwargs)[mode_number]
        ns.append(m.neff)
    ng = ns[1] - wavelength * (ns[2] - ns[0]) / (2 * wavelength_step)
    return GMode(mode_number=mode_number, ng=ng, neff=sum(ns) / 3, wavelength=wavelength)


def find_neff_ng_dw_dh(width: float = 0.465, thickness: float = 0.215, delta_width: float = 0.030, delta_thickness: float = 0.020, wavelength: float = 1.55, steps: int = 11, mode_number: int = 1, core: str = "Si", clad: str = "SiO2", **kwargs):
    """Effective and group index on a grid of width and thickness deviations
    (gplugins.modes.find_neff_ng_dw_dh): DataFrame with ``dw``, ``dh``
    (µm), ``neff``, ``ng``."""
    import pandas as pd

    kwargs.pop("cache_path", None)
    rows = {"dw": [], "dh": [], "neff": [], "ng": []}
    for dw in np.linspace(-delta_width, delta_width, steps):
        for dh in np.linspace(-delta_thickness, delta_thickness, steps):
            m = find_mode_dispersion(core=core, clad=clad, core_width=width + dw, core_thickness=thickness + dh, wavelength=wavelength, mode_number=mode_number, **kwargs)
            rows["dw"].append(dw)
            rows["dh"].append(dh)
            rows["neff"].append(m.neff)
            rows["ng"].append(m.ng)
    return pd.DataFrame(rows)


def coupling_length(neff1: float, neff2: float, power_ratio: float = 1.0, wavelength: float = 1.55) -> float:
    """Coupler length (µm) for ``power_ratio`` (1 = full transfer) from the
    even and odd supermode indices (gplugins.modes.find_coupling_vs_gap)."""
    dneff = float(np.real(neff1 - neff2))
    return wavelength / (np.pi * dneff) * np.arcsin(np.sqrt(power_ratio))


def find_coupling_vs_gap(gap1: float = 0.2, gap2: float = 0.4, steps: int = 12, nmodes: int = 4, wavelength: float = 1.55, parity=None, filepath=None, overwrite: bool = False, **kwargs):
    """Supermodes and full-transfer length against gap
    (gplugins.modes.find_coupling_vs_gap): DataFrame with ``gap``, ``ne``,
    ``no``, ``lc`` (µm), ``dn``. As in gplugins, ``ne`` and ``no`` are modes
    1 and 2 (the two highest effective indices)."""
    import pandas as pd

    def compute():
        gaps = np.linspace(gap1, gap2, steps)
        out = {"gap": gaps, "ne": [], "no": [], "lc": [], "dn": []}
        for g in gaps:
            modes = find_modes_coupler(gaps=(g,), wavelength=wavelength, nmodes=nmodes, **kwargs)
            n1, n2 = modes[1].neff, modes[2].neff
            out["ne"].append(n1)
            out["no"].append(n2)
            out["dn"].append(n1 - n2)
            out["lc"].append(coupling_length(n1, n2, wavelength=wavelength))
        return pd.DataFrame(out)

    return _cached(filepath, overwrite, compute)


def innerprod_trapz(mode1, mode2, **_ignored) -> complex:
    """Conjugated overlap ``1/4 ∫ (E1* × H2 + E2 × H1*) · x dA`` of two modes
    (gplugins.modes.overlap.innerprod_trapz). Accepts :class:`GMode` or
    :class:`~lightsim.modes.Mode`. lightsim modes carry unit power, so the
    self-overlap is 1 (gplugins' MPB fields are not power-normalised)."""
    a = mode1.mode if isinstance(mode1, GMode) else mode1
    b = mode2.mode if isinstance(mode2, GMode) else mode2
    return mode_overlap(a, b, conjugate=True)


# ----------------------------------------------------------------------------
# gplugins.tidy3d.modes
# ----------------------------------------------------------------------------


class Waveguide:
    """Waveguide model with the interface of ``gplugins.tidy3d.modes.Waveguide``
    (all lengths in µm), solved with lightsim.

    The core (``core_width`` x ``core_thickness``) sits on the box at z = 0,
    centred laterally; ``slab_thickness`` adds a slab of the core material;
    ``clad_material`` above, ``box_material`` (default: the cladding)
    below. ``side_margin``, ``clad_thickness``, ``box_thickness`` default to
    the wavelength. ``bend_radius`` solves the bend, with 12 PML cells on
    the outer wall (gplugins asks Tidy3D for 12 on both transverse axes;
    here the vertical walls stay closed, so substrate leakage is not
    included).
    ``grid_resolution`` is the number of cells per wavelength in the core
    material, refined nowhere else; ``max_grid_scaling`` the growth of the
    cells into the claddings. ``group_index_step`` (True or a relative
    step) adds ``n_group``. ``sidewall_thickness``/``surface_thickness``
    layers are not modelled (an error is raised if they are set).
    """

    def __init__(
        self,
        wavelength,
        core_width: float,
        core_thickness: float,
        core_material,
        clad_material,
        box_material=None,
        slab_thickness: float = 0.0,
        clad_thickness: float | None = None,
        box_thickness: float | None = None,
        side_margin: float | None = None,
        sidewall_angle: float = 0.0,
        sidewall_thickness: float = 0.0,
        sidewall_k: float = 0.0,
        surface_thickness: float = 0.0,
        surface_k: float = 0.0,
        bend_radius: float | None = None,
        target_neff: float | None = None,
        num_modes: int = 2,
        group_index_step=False,
        precision: str = "double",
        grid_resolution: int = 20,
        max_grid_scaling: float = 1.2,
        cache_path=None,
        overwrite: bool = False,
        gap: float | None = None,
    ):
        if sidewall_thickness or surface_thickness:
            raise NotImplementedError("sidewall/surface loss layers are not modelled; add them to a lightsim CrossSection")
        ignored = [k for k, v, d in (("precision", precision, "double"), ("cache_path", cache_path, None), ("overwrite", overwrite, False), ("sidewall_k", sidewall_k, 0.0), ("surface_k", surface_k, 0.0)) if v != d]
        if ignored:  # accepted for gplugins compatibility, without effect here: say so once
            import warnings

            from .gplugins import IgnoredArgumentWarning

            warnings.warn(f"Waveguide: lightsim ignores {', '.join(ignored)} (Tidy3D options without effect here)", IgnoredArgumentWarning, stacklevel=2)
        self.wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
        self.core_width = core_width
        self.core_thickness = core_thickness
        self.core_material = core_material
        self.clad_material = clad_material
        self.box_material = box_material
        self.slab_thickness = slab_thickness
        self.clad_thickness = clad_thickness
        self.box_thickness = box_thickness
        self.side_margin = side_margin
        self.sidewall_angle = sidewall_angle
        self.bend_radius = bend_radius
        self.target_neff = target_neff
        self.num_modes = num_modes
        self.group_index_step = group_index_step
        self.grid_resolution = grid_resolution
        self.max_grid_scaling = max_grid_scaling
        self.gap = gap
        self._modes = None

    _fields = ("wavelength", "core_width", "core_thickness", "core_material", "clad_material", "box_material", "slab_thickness", "clad_thickness", "box_thickness", "side_margin", "sidewall_angle", "bend_radius", "target_neff", "num_modes", "group_index_step", "grid_resolution", "max_grid_scaling", "gap")

    def settings(self) -> dict:
        """The constructor arguments, as a dict."""
        return {k: getattr(self, k) for k in self._fields}

    @property
    def cross_section(self) -> CrossSection:
        """The lightsim :class:`~lightsim.xsection.CrossSection` (metres)."""
        lam = float(np.mean(self.wavelength))
        side = self.side_margin if self.side_margin is not None else lam
        clad = self.clad_thickness if self.clad_thickness is not None else lam
        box = self.box_thickness if self.box_thickness is not None else lam
        if self.gap is not None:
            w1, w2 = (self.core_width, self.core_width) if np.ndim(self.core_width) == 0 else tuple(self.core_width)
            kw = dict(core_width=w1 * um, core_width2=w2 * um, gap=self.gap * um)
        else:
            kw = dict(core_width=self.core_width * um)
        from .xsection import material_tensor

        n_core = float(np.sqrt(np.real(material_tensor(_mat(self.core_material), lam * um)[0, 0])))
        step = lam * um / (self.grid_resolution * n_core)
        xs = CrossSection.strip(
            core_thickness=self.core_thickness * um,
            core_material=_mat(self.core_material),
            clad_material=_mat(self.clad_material),
            box_material=_mat(self.box_material) if self.box_material is not None else None,
            slab_thickness=self.slab_thickness * um,
            sidewall_angle=self.sidewall_angle,
            side_margin=side * um + (2 * lam * um if self.bend_radius else 0.0),  # outer side of a bend
            clad_thickness=clad * um,
            box_thickness=box * um,
            step=step,
            max_step=step * self.max_grid_scaling**6,
            **kw,
        )
        if self.bend_radius:  # widen the outer side only, keep the bend centre outside the window
            (u0, v0), (u1, v1) = xs.window
            u0 = max(u0, -0.8 * self.bend_radius * um)
            xs.window = ((u0, v0), (u1, v1))
        if self.max_grid_scaling != 1.2 or self.bend_radius:
            xs.mesh(xs.window, step=step, max_step=step * self.max_grid_scaling**6, grading=self.max_grid_scaling)
        return xs

    def _solve(self):
        if self._modes is None:
            xs = self.cross_section
            pml = {"cells": (0, 12, 0, 0)} if self.bend_radius else None
            out = []
            for lam in self.wavelength:
                ms = xs.solve(lam * um, self.num_modes, self.target_neff, pml=pml, bend_radius=self.bend_radius * um if self.bend_radius else None, sort="neff")
                if self.group_index_step:
                    xs.add_group_index(ms, lam * um)
                out.append(ms)
            self._modes = out
        return self._modes

    def _arr(self, f):
        a = np.array([[f(m) for m in ms] for ms in self._solve()])  # (wavelength, mode)
        a = a[0] if self.wavelength.size == 1 else a
        if self.num_modes == 1:
            a = a[..., 0]
        return a

    @property
    def modes(self) -> list:
        """The lightsim :class:`~lightsim.modes.Mode` objects, one list per wavelength."""
        return self._solve()

    @property
    def n_eff(self):
        """Complex effective indices (wavelength x mode, squeezed as Tidy3D)."""
        return self._arr(lambda m: m.neff)

    @property
    def n_complex(self):
        """Same as ``n_eff`` (Tidy3D name)."""
        return self.n_eff

    @property
    def n_group(self):
        """Group indices, present when ``group_index_step`` is set."""
        if not self.group_index_step:
            return None
        return self._arr(lambda m: m.n_group)

    @property
    def fraction_te(self):
        """TE fractions (wavelength x mode)."""
        return self._arr(lambda m: m.te_fraction)

    @property
    def fraction_tm(self):
        """TM fractions, ``1 - fraction_te``."""
        return self._arr(lambda m: 1 - m.te_fraction)

    @property
    def mode_area(self):
        """Effective mode areas (µm²)."""
        return self._arr(lambda m: m.effective_area() / um**2)

    @property
    def loss_dB_per_cm(self):
        """Propagation loss (dB/cm) from the imaginary part of ``n_eff``."""
        return self._arr(lambda m: m.loss_db_per_cm)

    def overlap(self, waveguide: "Waveguide", conjugate: bool = True):
        """Overlap matrix between the modes of the two waveguides at the
        (first) wavelength, as Tidy3D's ``outer_dot`` (see
        :func:`lightsim.modes.mode_overlap`)."""
        a, b = self._solve()[0], waveguide._solve()[0]
        o = np.array([[mode_overlap(x, y, conjugate) for y in b] for x in a])
        return o[0, 0] if o.size == 1 else o

    def __repr__(self) -> str:
        return f"{type(self).__name__}(" + ", ".join(f"{k}={getattr(self, k)!r}" for k in self._fields if getattr(self, k) is not None) + ")"


class WaveguideCoupler(Waveguide):
    """Two parallel cores (``core_width`` a pair, ``gap`` in µm), as
    ``gplugins.tidy3d.modes.WaveguideCoupler``."""

    def __init__(self, wavelength, core_width, core_thickness, core_material, clad_material, gap: float, **kwargs):
        kwargs.setdefault("num_modes", 4)
        super().__init__(wavelength, core_width, core_thickness, core_material, clad_material, gap=gap, **kwargs)

    def coupling_length(self, power_ratio: float = 1.0):
        """Coupling length (µm) from consecutive mode pairs (even, odd), as
        gplugins computes it: modes 0/1, 2/3, ..."""
        n = np.atleast_1d(np.real(self.n_eff))
        m = (n.shape[-1] // 2) * 2
        n_even, n_odd = n[..., :m:2], n[..., 1:m:2]
        lam = self.wavelength if self.wavelength.size > 1 else self.wavelength[0]
        return lam / (np.pi * (n_even - n_odd)) * np.arcsin(power_ratio**0.5)


def _sweep(waveguide: Waveguide, attribute: str, **sweep_kwargs):
    for prohibited in ("wavelength", "num_modes"):
        if prohibited in sweep_kwargs:
            raise ValueError(f"Parameter '{prohibited}' cannot be swept.")
    base = waveguide.settings()
    cls = type(waveguide)
    keys, values = tuple(sweep_kwargs), tuple(sweep_kwargs.values())
    shape = [len(v) for v in values]
    if waveguide.wavelength.size > 1:
        shape.append(waveguide.wavelength.size)
    if waveguide.num_modes > 1:
        shape.append(waveguide.num_modes)
    out = []
    for combo in itertools.product(*values):
        s = dict(base)
        s.update(zip(keys, combo))
        if cls is WaveguideCoupler:
            gap = s.pop("gap")
            w = cls(s.pop("wavelength"), s.pop("core_width"), s.pop("core_thickness"), s.pop("core_material"), s.pop("clad_material"), gap, **s)
        else:
            s.pop("gap", None)
            w = Waveguide(**s)
        out.append(getattr(w, attribute))
    return np.array(out).reshape(shape)


def sweep_n_eff(waveguide: Waveguide, **sweep_kwargs):
    """Effective index for every combination of the swept arguments
    (gplugins.tidy3d.modes.sweep_n_eff); an array of shape (sweeps...,
    [wavelength], [mode]) instead of an xarray."""
    return _sweep(waveguide, "n_eff", **sweep_kwargs)


def sweep_n_group(waveguide: Waveguide, **sweep_kwargs):
    """Group index sweep (needs ``group_index_step``)."""
    return _sweep(waveguide, "n_group", **sweep_kwargs)


def sweep_fraction_te(waveguide: Waveguide, **sweep_kwargs):
    """TE fraction sweep."""
    return _sweep(waveguide, "fraction_te", **sweep_kwargs)


def sweep_mode_area(waveguide: Waveguide, **sweep_kwargs):
    """Mode area sweep (µm²)."""
    return _sweep(waveguide, "mode_area", **sweep_kwargs)


def sweep_bend_mismatch(waveguide: Waveguide, bend_radii, track_modes: bool = False, modes_to_track=(0,)):
    """Bend-mismatch loss (gplugins.tidy3d.modes.sweep_bend_mismatch): for
    each radius ``|overlap(bend, straight)|⁴`` (both junctions of a bend),
    the diagonal for several modes or, with ``track_modes``, the best match
    of each tracked straight mode."""
    s = waveguide.settings()
    s.pop("gap", None)
    s["bend_radius"] = None
    straight = Waveguide(**s)
    res = []
    for r in bend_radii:
        s2 = dict(s, bend_radius=r)
        bend = Waveguide(**s2)
        o = np.atleast_2d(bend.overlap(straight))
        if track_modes:
            res.append([np.max(np.abs(o[:, m]) ** 2) for m in modes_to_track])
        else:
            res.append(np.diagonal(o) ** 2 if straight.num_modes > 1 else o[0, 0] ** 2)
    return np.abs(np.array(res)) ** 2


def sweep_coupling_length(coupler: WaveguideCoupler, gaps, power_ratio: float = 1.0):
    """Coupling length against gap (µm)."""
    out = []
    for g in gaps:
        s = coupler.settings()
        s.pop("gap")
        c = WaveguideCoupler(s.pop("wavelength"), s.pop("core_width"), s.pop("core_thickness"), s.pop("core_material"), s.pop("clad_material"), g, **s)
        out.append(c.coupling_length(power_ratio))
    return np.array(out)
