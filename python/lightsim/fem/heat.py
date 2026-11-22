"""Heat conduction on a cross-section, and thermo-optic phase shifters.

Steady state ``-div(k grad T) = q`` and transient ``rho c_p dT/dt -
div(k grad T) = q`` with Lagrange P1 or P2 elements (Rust core). Heat
sources sit in named regions (a heater): a power per unit length, or a
current along the waveguide axis through a conductor of known electrical
conductivity (Joule heating ``q = J^2 / sigma``). Boundaries: fixed
temperature (e.g. the substrate), convection ``-k dT/dn = h (T - T_inf)``
(e.g. the top surface), a heat flux; edges with no condition are insulated.

:func:`thermal_phase_shifter` turns the temperature into the change of the
effective index of a waveguide mode through the thermo-optic coefficients
(lightsim's mode solver on the perturbed index), and returns the phase shift
per power, P_pi, the crosstalk to a neighbouring waveguide and, with
``transient=True``, the thermal time constant. It is the counterpart of
``gplugins.femwell.solve_thermal`` plus the heater notebook of gplugins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _materials, _scalar
from .mesh import Field, Mesh

__all__ = ["HeatResult", "TransientResult", "Current", "solve_heat", "solve_heat_transient", "thermal_phase_shifter"]


@dataclass
class Current:
    """A heater driven by a current ``I`` (A) along the waveguide axis
    through a region of electrical conductivity ``sigma`` (S/m; default
    from the metal table). The dissipated power per length is
    ``I^2 / (sigma A)``."""

    current: float
    sigma: float | None = None


@dataclass
class HeatResult:
    """Steady temperature.

    Attributes
    ----------
    temperature : Field
        Temperature (K), or the rise above the fixed boundary temperature
        when that is 0.
    power_per_length : dict
        Dissipated power per unit length of each source region (W/m).
    conductivity : ndarray
        Per-triangle conductivity used (W/(m K)).
    """

    mesh: Mesh
    temperature: Field
    power_per_length: dict
    conductivity: np.ndarray
    order: int

    @property
    def total_power(self) -> float:
        """Total power per unit length (W/m)."""
        return float(sum(self.power_per_length.values()))

    def mean_temperature(self, region) -> float:
        """Area average of the temperature over named regions."""
        q = self.temperature.quadrature(6)
        w = q["weights"] * self.mesh.region_mask(region)[q["tri"]]
        return float(np.sum(w * q["values"]) / np.sum(w))

    def index_change(self, dndT: dict | None = None, reference: float = 0.0) -> np.ndarray:
        """Per-triangle refractive-index change ``dn/dT (T - reference)``
        (thermo-optic coefficients from the material library unless given)."""
        coef = np.array(self.mesh.per_region(dndT or {}, default=_dndt, what="dn/dT"), float)[self.mesh.tags]
        return coef * (self.temperature.per_triangle() - reference)


def _dndt(material: str) -> float:
    try:
        v = _materials.thermal(material)["dndT"]
    except KeyError:
        v = None
    return 0.0 if v is None else float(v)


def _conductivity(mesh: Mesh, conductivity: dict | None) -> np.ndarray:
    vals = mesh.per_region(conductivity or {}, default=lambda m: _materials.thermal(m)["k"], what="thermal conductivity")
    return np.array([np.atleast_1d(np.asarray(v, float)) for v in vals], dtype=object)


def _sources(mesh: Mesh, sources: dict | None) -> tuple[np.ndarray, dict]:
    f = np.zeros(mesh.n_tris)
    powers = {}
    areas = mesh.areas(curved=True)
    for name, s in (sources or {}).items():
        mask = mesh.region_mask(name)
        area = float(areas[mask].sum())
        if area == 0:
            raise ValueError(f"source region {name!r} is empty")
        if isinstance(s, Current):
            sigma = s.sigma
            if sigma is None:
                mats = {mesh.materials[t] for t in np.unique(mesh.tags[mask])}
                m = [_materials.metal(x) for x in mats]
                if not m or any(x is None for x in m):
                    raise ValueError(f"no electrical conductivity for {name!r} ({mats}); give Current(I, sigma=...)")
                sigma = m[0]["sigma"]
            p = s.current**2 / (sigma * area)
        elif isinstance(s, dict) and "current_density" in s:
            p = s["current_density"] ** 2 / s["sigma"] * area
        else:
            p = float(s)
        f[mask] += p / area
        powers[name] = p
    return f, powers


def _boundary_dofs(mesh: Mesh, where, order: int) -> np.ndarray:
    if isinstance(where, str) and where in mesh.region_names + mesh.materials and where not in ("top", "bottom", "left", "right"):
        return mesh.region_dofs(where, order)
    return mesh.edge_dofs(mesh.boundary_edges(where), order)


def _k_rows(k_obj: np.ndarray, tags: np.ndarray) -> np.ndarray:
    rows = _scalar.k_tensor(list(k_obj))
    return rows[tags]


def solve_heat(
    mesh: Mesh,
    sources: dict | None = None,
    conductivity: dict | None = None,
    fixed: dict | None = None,
    convection: dict | None = None,
    heat_flux: dict | None = None,
    order: int = 2,
    solver: str = "direct",
) -> HeatResult:
    """Steady heat conduction on a cross-section.

    Parameters
    ----------
    mesh : Mesh
    sources : dict
        ``{region: power per unit length (W/m)}``, or a :class:`Current`, or
        ``{"current_density": J (A/m^2), "sigma": S/m}`` as in femwell.
    conductivity : dict, optional
        ``{region or material: k}`` in W/(m K) (a scalar, or ``(kxx, kyy)``
        / ``(kxx, kyy, kxy)`` for anisotropic media); others from the
        material tables (:data:`lightsim.materials.THERMAL_PROPERTIES`,
        metals, air).
    fixed : dict, optional
        ``{boundary: T}``: ``"bottom"``, ``"top"``, ``"left"``, ``"right"``,
        a function ``f(x, y)`` of edge midpoints, or a region name (the
        whole region held at T). Default ``{"bottom": 0.0}``: the result is
        the rise above the substrate temperature.
    convection : dict, optional
        ``{boundary: (h, T_inf)}``, h in W/(m^2 K).
    heat_flux : dict, optional
        ``{boundary: q}``, inflow in W/m^2.
    order : int
        Lagrange order, 1 or 2.
    solver : str
        ``"direct"`` (sparse Cholesky) or ``"cg"`` (preconditioned CG).
    """
    fixed = {"bottom": 0.0} if fixed is None else fixed
    kobj = _conductivity(mesh, conductivity)
    k = _k_rows(kobj, mesh.tags)
    f, powers = _sources(mesh, sources)
    dirichlet = [(_boundary_dofs(mesh, w, order), float(t)) for w, t in fixed.items()]
    robin = [(mesh.boundary_edges(w), float(h), float(ti)) for w, (h, ti) in (convection or {}).items()]
    neumann = [(mesh.boundary_edges(w), float(q)) for w, q in (heat_flux or {}).items()]
    if not dirichlet and not robin:
        raise ValueError("the temperature is undetermined: give a fixed temperature or a convective boundary")
    r = _scalar.solve(mesh, order, k, None, f, dirichlet=dirichlet, neumann=neumann, robin=robin, solver=solver)
    kk = np.array([float(np.mean(v)) for v in kobj])[mesh.tags]
    return HeatResult(mesh, Field(mesh, order, r["u"][0]), powers, kk, order)


@dataclass
class TransientResult:
    """Transient temperature: ``times`` (s), ``probes`` (``{region: mean
    temperature at every time}``), the final field and the steady state."""

    times: np.ndarray
    probes: dict
    final: Field
    steady: HeatResult | None = None
    snapshots: list = field(default_factory=list)

    def time_constant(self, region: str) -> float:
        """Time for the region's mean temperature to cover 1 - 1/e of the way
        from its initial to its steady value (s), interpolated."""
        y = np.asarray(self.probes[region])
        y0 = y[0]
        yinf = self.steady.mean_temperature(region) if self.steady is not None else y[-1]
        target = y0 + (1 - math.exp(-1)) * (yinf - y0)
        s = np.sign(yinf - y0)
        idx = np.nonzero(s * (y - target) >= 0)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        if i == 0:
            return 0.0
        t0, t1, a, b = self.times[i - 1], self.times[i], y[i - 1], y[i]
        return float(t0 + (target - a) / (b - a) * (t1 - t0))


def solve_heat_transient(
    mesh: Mesh,
    sources: dict | None = None,
    t_end: float = 100e-6,
    nsteps: int = 400,
    conductivity: dict | None = None,
    capacity: dict | None = None,
    fixed: dict | None = None,
    convection: dict | None = None,
    initial=None,
    probes=(),
    order: int = 2,
    theta: float = 1.0,
    snapshots=(),
) -> TransientResult:
    """Transient heat conduction from ``initial`` (default 0, i.e. the rise
    after the sources switch on at t = 0) until ``t_end`` in ``nsteps``
    equal steps (theta scheme: 1 = backward Euler, 0.5 = Crank-Nicolson).

    ``capacity``: ``{region or material: rho * c_p}`` in J/(m^3 K) (default
    from the material tables). ``probes``: region names whose mean
    temperature is recorded every step. ``initial``: a :class:`Field`, an
    array of DOF values, or a function ``f(x, y)``. ``snapshots``: step
    indices to keep the whole field for.
    """
    fixed = {"bottom": 0.0} if fixed is None else fixed
    from .. import _core

    kobj = _conductivity(mesh, conductivity)
    k = _k_rows(kobj, mesh.tags)
    cvals = mesh.per_region(capacity or {}, default=lambda m: _materials.thermal(m)["rho"] * _materials.thermal(m)["cp"], what="heat capacity")
    c = np.asarray(cvals, float)[mesh.tags]
    f, _ = _sources(mesh, sources)
    ndof = mesh.ndof(order)
    if initial is None:
        u0 = np.zeros(ndof)
    elif isinstance(initial, Field):
        u0 = initial.values
    elif callable(initial):
        xy = mesh.dof_coords(order)
        u0 = np.asarray(initial(xy[:, 0], xy[:, 1]), float)
    else:
        u0 = np.asarray(initial, float)
    dd, dv = [], []
    for w, t in fixed.items():
        d = _boundary_dofs(mesh, w, order)
        dd.extend(d.tolist())
        dv.extend([float(t)] * len(d))
    rb = [(mesh.boundary_edges(w), float(h), float(ti)) for w, (h, ti) in (convection or {}).items()]
    rbt = (
        np.concatenate([e for e, _, _ in rb]).tolist() if rb else [],
        np.concatenate([np.full(len(e), h) for e, h, _ in rb]).tolist() if rb else [],
        np.concatenate([np.full(len(e), u) for e, _, u in rb]).tolist() if rb else [],
    )
    # probes: area averages of the vertex values over each region (weights per node)
    pr = []
    areas = mesh.areas()
    for name in probes:
        mask = mesh.region_mask(name)
        w = np.zeros(ndof)
        np.add.at(w, mesh.tris[mask].ravel(), np.repeat(areas[mask] / 3.0, 3))
        w /= float(areas[mask].sum())
        nz = np.nonzero(w)[0]
        pr.append((nz.tolist(), w[nz].tolist()))
    dt = t_end / nsteps
    r = _core.fem_scalar_transient(*mesh._args(), order, np.ascontiguousarray(k), np.ascontiguousarray(c), np.ascontiguousarray(f), dd, dv, ([], []), rbt, np.ascontiguousarray(u0), dt, int(nsteps), float(theta), sorted(set(list(snapshots) + [nsteps])), pr)
    times = dt * np.arange(nsteps + 1)
    probes_out = {name: np.asarray(r["traces"][i]) for i, name in enumerate(probes)}
    saved = [Field(mesh, order, v) for v in r["saved"]]
    steady = None
    if sources:
        steady = solve_heat(mesh, sources, conductivity, fixed, convection, order=order)
    return TransientResult(times, probes_out, saved[-1], steady, saved[:-1])


# ----------------------------------------------------------------------------
# thermo-optic phase shifter
# ----------------------------------------------------------------------------


def sample_per_triangle(mesh: Mesh, values: np.ndarray, x, y, outside=np.nan) -> np.ndarray:
    """Sample a per-triangle array at points (piecewise constant)."""
    from .. import _core

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    pts = np.ascontiguousarray(np.stack([x.ravel(), y.ravel()], 1))
    t = _core.fem_locate(mesh.nodes, mesh.tris, mesh.tags, pts)
    v = np.asarray(values)
    out = np.where(t >= 0, v[np.maximum(t, 0)], outside)
    return out.reshape(x.shape)


def neff_fd(mesh: Mesh, n_tri: np.ndarray, wavelength: float, window, resolution: float = 20e-9, num_modes: int = 1, target_neff=None, supersample: int = 4):
    """Modes of a cross-section with lightsim's finite-difference solver
    (:func:`lightsim.modes.solve_cross_section`), the per-triangle index
    ``n_tri`` averaged over each Yee cell (``supersample`` x ``supersample``
    sub-samples of the permittivity around every Yee position) on a uniform
    grid of ``window = (x0, y0, x1, y1)``. Returns the list of complex
    effective indices and the raw solver output."""
    from ..modes import C0, solve_cross_section

    x0, y0, x1, y1 = window
    nx = max(4, int(round((x1 - x0) / resolution)))
    ny = max(4, int(round((y1 - y0) / resolution)))
    xn = np.linspace(x0, x1, nx + 1)
    yn = np.linspace(y0, y1, ny + 1)
    xc = 0.5 * (xn[:-1] + xn[1:])
    yc = 0.5 * (yn[:-1] + yn[1:])
    eps = np.asarray(n_tri, complex) ** 2
    bg = complex(np.nanmedian(eps.real))
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5

    def avg(px, py):
        acc = 0.0
        for a in sub:
            for b in sub:
                acc = acc + sample_per_triangle(mesh, eps, px + a * dx, py + b * dy, outside=bg)
        return acc / supersample**2

    ex = avg(xc[:, None], yn[None, :ny])
    if not np.all(np.isfinite(ex)):
        bad = sorted({mesh.materials[t] for t in np.unique(mesh.tags[~np.isfinite(np.asarray(n_tri, complex))])})
        raise ValueError(f"materials without a refractive index inside the mode window: {bad}; pass index={{name: n}}")
    ey = avg(xn[:nx, None], yc[None, :])
    ez = avg(xn[:nx, None], yn[None, :ny])
    res = solve_cross_section(ex, ey, ez, xn, yn, C0 / wavelength, num_modes=num_modes, target_neff=target_neff)
    return [complex(r[0]) for r in res], res


def thermal_phase_shifter(
    mesh: Mesh,
    heater: str,
    waveguide: str,
    wavelength: float = 1.55e-6,
    power_per_length: float = 1e3,
    neighbour: str | None = None,
    index: dict | None = None,
    dndT: dict | None = None,
    conductivity: dict | None = None,
    fixed: dict | None = None,
    convection: dict | None = None,
    order: int = 2,
    mode_window: float = 1.5e-6,
    mode_resolution: float = 20e-9,
    transient: bool = False,
    t_end: float = 200e-6,
    nsteps: int = 400,
) -> dict:
    """Thermo-optic phase shifter figures of merit.

    Solves the heat equation with ``power_per_length`` (W/m) dissipated in
    the ``heater`` region, maps the temperature rise to an index change
    ``dn/dT * dT`` and re-solves the fundamental mode of ``waveguide`` with
    lightsim's finite-difference mode solver on a grid of
    ``mode_resolution`` around it (``mode_window`` margin). The problem is
    linear in the power, so

    ``dphi/dP = 2 pi / lambda * dneff/dp`` (rad/W, independent of the heater
    length L for a heater as long as the arm) and ``P_pi = pi / (dphi/dP)``.

    Returns a dict: ``neff`` (unheated), ``dneff_dp`` (per W/m),
    ``dphi_dP`` (rad/W), ``P_pi`` (W), ``dT_waveguide_per_W_per_m``,
    ``dT_heater_per_W_per_m``, ``crosstalk`` (``dneff`` of the
    ``neighbour`` waveguide over that of ``waveguide``), ``heat`` (the
    :class:`HeatResult`) and, with ``transient``, ``time_constant`` (s, the
    1 - 1/e rise time of the waveguide temperature).
    """
    heat = solve_heat(mesh, {heater: power_per_length}, conductivity, fixed, convection, order=order)
    def nmat(m):
        try:
            return _materials.refractive_index(m, wavelength, index).real
        except KeyError:
            return np.nan  # allowed outside the mode windows (heater metals)

    n0 = np.array([nmat(m) for m in mesh.materials])[mesh.tags]
    fixed_t = 0.0 if fixed is None else float(next(iter(fixed.values())))
    dn = heat.index_change(dndT, reference=fixed_t)
    out = {"heat": heat}

    def shift(region):
        lo = mesh.nodes[mesh.tris[mesh.region_mask(region)].ravel()].min(0)
        hi = mesh.nodes[mesh.tris[mesh.region_mask(region)].ravel()].max(0)
        win = (lo[0] - mode_window, lo[1] - mode_window, hi[0] + mode_window, hi[1] + mode_window)
        # only the guide in question: other high-index regions inside the window stay as they are
        a, _ = neff_fd(mesh, n0, wavelength, win, mode_resolution)
        b, _ = neff_fd(mesh, n0 + dn, wavelength, win, mode_resolution)
        return a[0], b[0] - a[0]

    neff, d = shift(waveguide)
    out["neff"] = neff
    out["dneff_dp"] = d.real / power_per_length
    out["dphi_dP"] = 2 * math.pi / wavelength * out["dneff_dp"]
    out["P_pi"] = math.pi / out["dphi_dP"] if out["dphi_dP"] != 0 else float("inf")
    out["dT_waveguide_per_W_per_m"] = heat.mean_temperature(waveguide) / power_per_length
    out["dT_heater_per_W_per_m"] = heat.mean_temperature(heater) / power_per_length
    if neighbour is not None:
        _, dn2 = shift(neighbour)
        out["dneff_dp_neighbour"] = dn2.real / power_per_length
        out["crosstalk"] = dn2.real / d.real
    if transient:
        tr = solve_heat_transient(mesh, {heater: power_per_length}, t_end, nsteps, conductivity, None, fixed, convection, probes=[waveguide], order=order)
        out["time_constant"] = tr.time_constant(waveguide)
        out["transient"] = tr
    return out
