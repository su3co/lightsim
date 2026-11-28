"""2.5-D effective-index simulations (cf. Lumerical varFDTD / MODE 2.5D FDTD).

A planar photonic circuit is collapsed along z: every lateral region of the
layout is replaced by a homogeneous medium whose index is the effective index
of the slab mode of the vertical layer stack found there, and the resulting
2-D problem is solved as a 2-D lightsim simulation (``dimensions=2``,
invariant along z), which updates only the three field components of the
chosen polarization:

    slab TE mode (E in the chip plane)  ->  polarization "te": (Ex, Ey, Hz)
    slab TM mode (E mostly along z)     ->  polarization "tm": (Ez, Hx, Hy)

Pieces:

* :func:`slab_modes` / :func:`slab_neff` -- exact multilayer slab solver
  (transfer-matrix dispersion relation, bracketed root finding),
* :func:`effective_indices` -- slab n_eff of every patterned layer of a
  gdsfactory component + layer stack,
* :func:`effective_index_simulation` -- the 2-D :class:`lightsim.Simulation`,
* :func:`solve_port_mode` -- 1-D lateral mode solver on the Yee positions of
  a port line of any 2-D simulation (what :func:`lightsim.solve_modes` calls
  for one),
* :func:`write_sparameters_2d` -- S-matrix of the 2.5-D model, same output
  convention as :func:`lightsim.sparams.write_sparameters`.

This is the plain effective-index method: one non-dispersive effective index
per region, evaluated at the centre wavelength. See ``docs/modules/eim.md``
for the limitations relative to a variational (varFDTD) reduction.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import gds as _gds
from .modes import ETA0, Mode
from .sparams import _copy_symmetric_columns, _half_width, _port_plane, _resolve_port_symmetry, _symmetry_plan, sim_step
from .spec import C0, AutoMesh, GaussianPulse, Medium, ModeMonitor, ModeSource, PML, PolySlab, Simulation, Structure, um

__all__ = [
    "slab_modes",
    "slab_neff",
    "effective_indices",
    "effective_index_simulation",
    "solve_port_mode",
    "write_sparameters_2d",
]


# ----------------------------------------------------------------------------
# slab solver
# ----------------------------------------------------------------------------


def _slab_characteristic(neff, n, d, k0, tm):
    """Dispersion function of a multilayer slab; zero at a guided mode.

    ``n[0]`` and ``n[-1]`` are the bottom / top half-spaces, ``n[1:-1]`` the
    layers of thickness ``d``. With F the tangential field (E_y for TE, H_y
    for TM) and G = F'/rho (rho = 1 for TE, n^2 for TM), both continuous at
    the interfaces, start from the solution decaying into the bottom
    half-space, propagate through the layers and require a decaying solution
    in the top half-space.
    """
    rho = (lambda ni: ni * ni) if tm else (lambda ni: 1.0)
    g0 = k0 * np.sqrt(neff * neff - n[0] ** 2)
    F, G = 1.0, g0 / rho(n[0])
    for ni, di in zip(n[1:-1], d):
        kap = k0 * np.sqrt(complex(ni * ni - neff * neff))
        r = rho(ni)
        c = np.cos(kap * di).real
        s_over = (di if abs(kap) < 1e-9 / max(di, 1e-30) else (np.sin(kap * di) / kap).real)
        s_times = (kap * np.sin(kap * di)).real
        F, G = F * c + r * G * s_over, -F * s_times / r + G * c
        scale = abs(F) + abs(G) / k0
        F, G = F / scale, G / scale
    gt = k0 * np.sqrt(neff * neff - n[-1] ** 2)
    return G / k0 + gt * F / (rho(n[-1]) * k0)


def slab_modes(indices, thicknesses, wavelength: float, polarization: str = "te", samples: int = 4000) -> list[float]:
    """Guided-mode effective indices of a planar multilayer, highest first.

    ``indices`` = [n_bottom_halfspace, n_1, ..., n_m, n_top_halfspace],
    ``thicknesses`` = [d_1, ..., d_m] (metres). Lossless, isotropic layers.
    ``polarization``: ``"te"`` (E parallel to the layers) or ``"tm"``.
    """
    n = [float(v) for v in indices]
    d = [float(v) for v in thicknesses]
    if len(n) != len(d) + 2:
        raise ValueError("need len(indices) == len(thicknesses) + 2")
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    lo = max(n[0], n[-1])
    hi = max(n)
    if not d or hi <= lo:
        return []
    k0 = 2 * np.pi / wavelength
    eps = 1e-10 * (hi - lo)
    # sample uniformly in neff^2 distance from the core line (roots crowd near cut-off evenly)
    xs = lo + eps + (hi - lo - 2 * eps) * np.linspace(0.0, 1.0, samples)
    fs = np.array([_slab_characteristic(x, n, d, k0, pol == "tm") for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            roots.append(float(xs[i]))
        elif fs[i] * fs[i + 1] < 0:
            roots.append(float(brentq(_slab_characteristic, xs[i], xs[i + 1], args=(n, d, k0, pol == "tm"), xtol=1e-14, rtol=1e-14)))
    return sorted(roots, reverse=True)


def slab_neff(
    n_core: float,
    thickness: float,
    wavelength: float,
    n_substrate: float,
    n_cladding: float | None = None,
    polarization: str = "te",
    mode: int = 0,
) -> float:
    """Effective index of mode ``mode`` of a three-layer slab waveguide
    (substrate / core of ``thickness`` / cladding). Raises if it is cut off."""
    nc = n_substrate if n_cladding is None else n_cladding
    modes = slab_modes([n_substrate, n_core, nc], [thickness], wavelength, polarization)
    if mode >= len(modes):
        raise ValueError(f"slab mode {mode} ({polarization}) is cut off: {len(modes)} guided mode(s)")
    return modes[mode]


# ----------------------------------------------------------------------------
# layer stack -> effective indices
# ----------------------------------------------------------------------------


def _material_table(material_index):
    return _gds.material_table(material_index)


def _n(material, idx, where):
    """Index of a material; a missing entry is an error that names ``where``."""
    return _gds.material_index_of(material, idx, where)


def _vertical_stack(layer, wafer, idx, background):
    """(indices, thicknesses) of the vertical stack under a patterned layer:
    wafer (unpatterned) slabs with ``layer`` on top of them; the outermost
    intervals become half-spaces."""
    cuts = sorted({layer.zmin, layer.zmax, *[z for w in wafer for z in (w.zmin, w.zmax)]})
    n_bg = _n(background, idx, "background")

    def index_at(z):
        if layer.zmin < z < layer.zmax:
            return _n(layer.material, idx, f"layer stack level '{layer.name}'")
        for w in reversed(wafer):
            if w.zmin < z < w.zmax:
                return _n(w.material, idx, f"layer stack level '{w.name}'")
        return n_bg

    ns, ds = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ni = index_at(0.5 * (a + b))
        if ns and abs(ns[-1] - ni) < 1e-12:
            ds[-1] += b - a
        else:
            ns.append(ni)
            ds.append(b - a)
    # the outermost intervals extend to infinity; if the patterned layer is
    # itself outermost, the background material closes the stack
    if layer.zmin <= cuts[0] + 1e-15:
        ns.insert(0, n_bg)
        ds.insert(0, 0.0)
    if layer.zmax >= cuts[-1] - 1e-15:
        ns.append(n_bg)
        ds.append(0.0)
    return ns, ds[1:-1]


def effective_indices(
    component,
    wavelength: float,
    layer_stack=None,
    material_index: dict | None = None,
    polarization: str = "te",
    mode: int = 0,
    background: str = "sio2",
    exclude=None,
    wafer_levels=None,
):
    """Slab effective index of every patterned layer of ``component``.

    Returns ``(neff, n_background, layers)``: ``neff`` maps the layer-stack
    level name to the effective index of slab mode ``mode`` of the vertical
    stack "wafer layers + that level", ``n_background`` is the index used
    where no patterned layer is present (the wafer material at the height of
    the patterned layers, i.e. the lateral cladding), ``layers`` the
    :class:`lightsim.gds.StackLayer` list. Levels whose slab mode is cut off
    get the background index.
    """
    idx = _material_table(material_index)
    layers = _gds.stack_layers(component, layer_stack, wafer_levels=wafer_levels, **({} if exclude is None else {"exclude": exclude}))
    wafer = [l for l in layers if not l.polygons]
    pattern = [l for l in layers if l.polygons]
    if not pattern:
        raise ValueError("component has no polygons on any layer of the layer stack")
    zc = np.mean([0.5 * (l.zmin + l.zmax) for l in pattern])
    n_bg = _n(background, idx, "background")
    for w in reversed(wafer):
        if w.zmin < zc < w.zmax:
            n_bg = _n(w.material, idx, f"layer stack level '{w.name}'")
            break
    neff = {}
    for l in pattern:
        ns, ds = _vertical_stack(l, wafer, idx, background)
        modes = slab_modes(ns, ds, wavelength, polarization)
        neff[l.name] = modes[mode] if mode < len(modes) else n_bg
    return neff, n_bg, layers


# ----------------------------------------------------------------------------
# 2-D simulation
# ----------------------------------------------------------------------------


def effective_index_simulation(
    component,
    wavelengths,
    layer_stack=None,
    material_index: dict | None = None,
    polarization: str = "te",
    slab_mode: int = 0,
    steps_per_wavelength: float = 20.0,
    xy_margin: float = 1.0 * um,
    port_extension: float = 2.0 * um,
    background: str = "sio2",
    run_time_factor: float = 6.0,
    shutoff: float = 1e-5,
    z_cell: float | None = 1.0 * um,
    pml_layers: int = 12,
    exclude=None,
    wafer_levels=None,
    level_indices: dict | None = None,
    background_index: float | None = None,
):
    """Build the 2-D effective-index model of a gdsfactory component.

    Every polygon of a patterned layer-stack level becomes a z-invariant
    region with the slab effective index of that level (see
    :func:`effective_indices`); regions of higher effective index take
    precedence where levels overlap (e.g. a rib over its partially etched
    slab). The result is a 2-D simulation (``dimensions=2``, invariant along
    z, ``polarization`` = the slab mode family) with PML in x / y. The slab
    index is evaluated at the mean wavelength.

    ``z_cell`` is the thickness of the slab the 2-D model stands for
    (``None``: one in-plane step). The fields do not depend on z, so it only
    scales fluxes and mode powers; S-parameters do not depend on it.

    ``level_indices`` (level name -> index) and ``background_index`` replace
    the slab effective indices of the levels listed and the index between
    them. With the bulk indices of the materials the model is the plain 2-D
    cross-section of the layout, without the vertical confinement the
    effective index stands for.

    Returns ``(sim, ports, info)`` with no sources / monitors; ``info`` holds
    ``neff`` (per level), ``n_background``, ``polarization`` and ``z`` (the
    z range of the cell).
    """
    import gdsfactory as gf

    pol = polarization.lower()
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    lam0 = float(np.mean(wl))
    neff, n_bg, _ = effective_indices(component, lam0, layer_stack, material_index, pol, slab_mode, background, exclude, wafer_levels)
    if level_indices:
        unknown = set(level_indices) - set(neff)
        if unknown:
            raise ValueError(f"level_indices names unknown levels {sorted(unknown)}; the patterned levels are {sorted(neff)}")
        neff.update({k: float(v) for k, v in level_indices.items()})
    if background_index is not None:
        n_bg = float(background_index)
    ports = _gds.port_info(component)
    ext = gf.components.extend_ports(component, length=(port_extension + xy_margin) / um) if ports else component
    import warnings

    with warnings.catch_warnings():  # already reported by effective_indices
        warnings.simplefilter("ignore", UserWarning)
        stack = _gds.stack_layers(ext, layer_stack, wafer_levels=wafer_levels, **({} if exclude is None else {"exclude": exclude}))
    layers = [l for l in stack if l.polygons]
    bb = component.dbbox()
    nmax = max([n_bg, *neff.values()])
    dz = lam0 / (nmax * steps_per_wavelength)
    rmin = [bb.left * um - xy_margin, bb.bottom * um - xy_margin, 0.0]
    rmax = [bb.right * um + xy_margin, bb.top * um + xy_margin, dz]
    length = max(rmax[0] - rmin[0], rmax[1] - rmin[1])
    pml = lambda: PML(layers=pml_layers)  # noqa: E731
    if z_cell is not None and z_cell > dz:
        rmax[2] = float(z_cell)
    sim = Simulation(
        min=rmin,
        max=rmax,
        mesh=AutoMesh(lam0, steps_per_wavelength=steps_per_wavelength, grading=1.3),
        run_time=run_time_factor * nmax * length / C0,
        boundaries={"x": [pml(), pml()], "y": [pml(), pml()], "z": [{"type": "periodic"}, {"type": "periodic"}]},
        background="eim_background",
        shutoff=shutoff,
        dimensions=2,
        invariant_axis="z",
        polarization=pol,
    )
    sim.add_medium(Medium("eim_background", n=n_bg))
    # lowest effective index first: later structures override earlier ones
    for l in sorted(layers, key=lambda l: neff.get(l.name, n_bg)):
        name = f"eim_{l.name}"
        sim.add_medium(Medium(name, n=neff.get(l.name, n_bg)))
        for i, poly in enumerate(l.polygons):
            sim.structures.append(Structure(PolySlab(poly, -1.0, 1.0, axis=2), name, name=f"{l.name}_{i}"))
    info = {"neff": neff, "n_background": n_bg, "polarization": pol, "z": (float(sim.min[2]), float(sim.max[2])), "wavelength": lam0}
    return sim, ports, info


# ----------------------------------------------------------------------------
# 1-D lateral port modes
# ----------------------------------------------------------------------------


def _invariant_axis(sim) -> int:
    """Invariant axis of a 2-D simulation (object or JSON spec); z for the
    one-cell, periodic-z grids that stood for 2-D before ``dimensions=2``."""
    if hasattr(sim, "to_json"):
        return int(sim.invariant_axis) if getattr(sim, "dimensions", 3) == 2 else 2
    import json

    d = json.loads(sim)
    w = d.get("invariant_axis", 2) if d.get("dimensions", 3) == 2 else 2
    return {"x": 0, "y": 1, "z": 2}.get(w, w)


def solve_port_mode(sim, axis: int, position: float, min, max, freq: float, polarization: str = "te", mode_index: int = 0, num_modes: int | None = None):
    """Modes of a 2-D simulation on the line normal to ``axis`` at
    ``position`` (a port "plane" of a 2-D simulation is a line: one in-plane
    axis and the slab along the invariant axis).

    The 1-D eigenproblem is discretized on the Yee positions of the line with
    the permittivity the engine uses, like :func:`lightsim.solve_modes` does
    in 3-D (whose 2-D operators assume closed walls and so cannot represent a
    periodic single cell):

    * ``"te"``: E along the line (in the plane), H along the invariant axis
      -- beta^2 e = (k0^2 + D_f eps_n^-1 D_b) eps_t e,
    * ``"tm"``: E along the invariant axis -- beta^2 e = (k0^2 eps_w + D_b D_f) e.

    ``sim`` is a 2-D :class:`lightsim.Simulation` (any invariant axis) or a
    hand-made one-cell grid with periodic z. ``min`` / ``max`` may have two
    (in-plane) components for a 2-D simulation.

    Returns the :class:`lightsim.Mode` number ``mode_index`` (by descending
    n_eff), or the list of the first ``num_modes`` modes if that is given.
    Profiles are normalized to unit power over the thickness of the slab.
    """
    from . import _core

    w = _invariant_axis(sim)
    if axis == w:
        raise ValueError("a port of a 2-D simulation cannot be normal to its invariant axis")
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    slab = getattr(sim, "slab", None)
    if slab is not None:  # two in-plane components -> span the slab
        from .spec import _lift

        min, max = _lift(min, w, slab[1]), _lift(max, w, slab[2])
    spec = sim.geometry_json() if hasattr(sim, "geometry_json") else sim
    info = _core.plane_info(spec, int(axis), float(position), [float(v) for v in min], [float(v) for v in max], float(freq))
    nu, nv = int(info["dims"][0]), int(info["dims"][1])
    un = np.asarray(info["u_nodes"], dtype=float)
    vn = np.asarray(info["v_nodes"], dtype=float)
    # the plane's natural axes (u, v): one is the line, the other the invariant axis
    along_u = int(info["axes"][1]) == w
    if not along_u and int(info["axes"][0]) != w:
        raise ValueError("the port plane does not contain the invariant axis")
    pick = (lambda a: np.asarray(a)[:, 0]) if along_u else (lambda a: np.asarray(a)[0, :])
    eps_t = pick(info["eps_u"] if along_u else info["eps_v"])  # E along the line, at half positions
    eps_w = pick(info["eps_v"] if along_u else info["eps_u"])  # E along the invariant axis, at the line nodes
    eps_n = pick(info["eps_n"])  # E along the normal, at the line nodes
    ln, wn = (un, vn) if along_u else (vn, un)
    if len(wn) - 1 > 1:  # the 1-D problem reads one line of cells: a 3-D cross-section is not one
        raise ValueError(
            f"solve_port_mode solves the lateral 1-D problem of a 2-D simulation, but this port spans {len(wn) - 1} cells "
            "along the invariant axis, so the simulation is 3-D. Use lightsim.solve_modes for the modes of a 3-D "
            "cross-section (invdes.port_mode with polarization=None)."
        )
    nl = len(ln) - 1
    dp = np.diff(ln)  # primal steps (half positions)
    dd = np.empty(nl)  # dual steps (nodes)
    dd[0] = dp[0]
    dd[1:] = 0.5 * (dp[:-1] + dp[1:])
    fwd = np.diag(-1.0 / dp) + np.diag(1.0 / dp[:-1], 1)  # nodes -> halves, f[N] = 0
    bwd = np.diag(1.0 / dd) + np.diag(-1.0 / dd[1:], -1)  # halves -> nodes, g[-1] = 0
    k0 = 2 * np.pi * freq / C0
    if pol == "te":
        A = (k0**2 * np.eye(nl) + fwd @ np.diag(1.0 / eps_n) @ bwd) @ np.diag(eps_t)
        w_lat = dp
    else:
        A = k0**2 * np.diag(eps_w) + bwd @ fwd
        w_lat = dd
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(-vals.real)
    thick = float(wn[-1] - wn[0])
    s = -1.0 if axis == 1 else 1.0  # (u, v, normal) is left-handed for axis 1
    want = (num_modes if num_modes is not None else mode_index + 1)
    modes = []
    zero = np.zeros(nl, dtype=complex)
    for j in order[:want]:
        beta = np.sqrt(complex(vals[j]))
        if beta.real < 0:
            beta = -beta
        neff = beta / k0
        e = vecs[:, j].astype(complex)
        e = e * np.exp(-1j * np.angle(e[np.argmax(np.abs(e))]))
        # H~ = eta0 H: (eps / neff) e for E along the line, neff e for E
        # along the invariant axis, directed along n x E (n x u = v,
        # n x v = -u for right-handed (u, v, n))
        if pol == "te":
            h = eps_t * e / neff
            (el, ew), (hl, hw) = (e, zero), (zero, h)
        else:
            h = neff * e
            (el, ew), (hl, hw) = (zero, e), (h, zero)
        p = 0.5 / ETA0 * np.real(np.sum(e * np.conj(h) * w_lat)) * thick
        if along_u:  # line = u, invariant = v
            eu, ev, hu, hv = el, ew, -s * hl, s * hw
        else:  # line = v, invariant = u
            eu, ev, hu, hv = ew, el, -s * hw, s * hl
        c = 1.0 / np.sqrt(abs(p)) if p != 0 else 1.0
        if along_u:
            tile = lambda a: np.repeat((a * c)[:, None], nv, axis=1)  # noqa: E731
        else:
            tile = lambda a: np.repeat((a * c)[None, :], nu, axis=0)  # noqa: E731
        Eu, Ev, Hu, Hv = tile(eu), tile(ev), tile(hu) / ETA0, tile(hv) / ETA0
        shift = np.exp(-1j * neff * k0 * info["h_offset"])  # H plane half a cell upstream
        hu_p, hv_p = Hu * shift, Hv * shift
        profile = {
            "neff": float(np.real(neff)),
            "neff_imag": float(np.imag(neff)),
            "dims": [nu, nv],
            "e_re": [np.real(Eu).ravel().tolist(), np.real(Ev).ravel().tolist()],
            "e_im": [np.imag(Eu).ravel().tolist(), np.imag(Ev).ravel().tolist()],
            "h_re": [np.real(hu_p).ravel().tolist(), np.real(hv_p).ravel().tolist()],
            "h_im": [np.imag(hu_p).ravel().tolist(), np.imag(hv_p).ravel().tolist()],
            "freq": float(freq),
            "h_offset": float(info["h_offset"]),
        }
        modes.append(Mode(neff=complex(neff), freq=freq, te_fraction=1.0 if pol == "te" else 0.0, profile=profile, E=(Eu, Ev), H=(Hu, Hv), info=info))
    return modes if num_modes is not None else modes[mode_index]


# ----------------------------------------------------------------------------
# S-parameters
# ----------------------------------------------------------------------------


def write_sparameters_2d(
    component,
    wavelengths=None,
    layer_stack=None,
    material_index: dict | None = None,
    polarization: str = "te",
    steps_per_wavelength: float = 20.0,
    port_offset: float = 0.5 * um,
    port_lateral_margin: float = 1.0 * um,
    mode_index: int = 0,
    device: str = "auto",
    input_ports=None,
    progress: int = 0,
    return_results: bool = False,
    port_symmetry=None,
    port_profiles: str = "center",
    **kwargs,
):
    """S-matrix of a gdsfactory component from the 2.5-D effective-index model.

    Same workflow and output as :func:`lightsim.sparams.write_sparameters`
    (``"o2@0,o1@0"`` = S(o2 <- o1) over ``wavelengths``, ``"mode_neff"`` per
    port), plus ``"slab_neff"`` and ``"n_background"``. ``polarization``
    selects the slab mode family and the matching 2-D field set; extra
    keyword arguments go to :func:`effective_index_simulation`.
    ``port_symmetry`` is as in :func:`lightsim.sparams.write_sparameters`.
    """
    if wavelengths is None:
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 21)
    wl = np.asarray(wavelengths, dtype=float).reshape(-1)
    freqs = C0 / wl
    f0 = float(np.mean(freqs))
    fw = float(np.ptp(freqs)) if len(freqs) > 1 else 0.2 * f0
    sim, ports, info = effective_index_simulation(component, wl, layer_stack, material_index, polarization, steps_per_wavelength=steps_per_wavelength, **kwargs)
    z0, z1 = info["z"]
    planes, profiles, neffs = {}, {}, {}
    for p in ports:
        axis, pos, direc, pmin, pmax = _port_plane(p, port_offset, z0, z1, _half_width(p, ports, port_lateral_margin))
        m = solve_port_mode(sim, axis, pos, pmin, pmax, f0, polarization, mode_index)
        planes[p["name"]] = (axis, pos, direc, pmin, pmax)
        profiles[p["name"]] = m.profile
        if port_profiles == "band":
            from .modes import profile_over_band

            profiles[p["name"]] = profile_over_band(sim, axis, pos, pmin, pmax, m.profile, freqs, polarization=polarization, num_modes=mode_index + 3)
        elif port_profiles != "center":
            raise ValueError("port_profiles must be 'center' or 'band'")
        neffs[p["name"]] = m.neff
    signal = GaussianPulse(f0, max(fw, 0.1 * f0))
    out = {"wavelengths": wl, "mode_neff": neffs, "slab_neff": info["neff"], "n_background": info["n_background"]}
    results = {}
    names = [p["name"] for p in ports]
    inputs = list(input_ports or names)
    run_ports, source_of = _symmetry_plan(names, _resolve_port_symmetry(component, ports, port_symmetry), inputs)
    for src in run_ports:
        s = Simulation(**{k: v for k, v in sim.__dict__.items()})
        s.sources = []
        s.monitors = []
        axis, pos, direc, pmin, pmax = planes[src]
        s.add_source(ModeSource(axis, pos, direc, pmin, pmax, profiles[src], signal, name=f"src_{src}"))
        for p in ports:
            a, ps, d, mn, mx = planes[p["name"]]
            if p["name"] == src:
                # just downstream of the source: incident and reflected modes
                shift = 2 * sim_step(sim, a, ps)
                ps = ps + (shift if d == "+" else -shift)
                mn, mx = list(mn), list(mx)
                mn[a] = mx[a] = ps
            s.add_monitor(ModeMonitor(f"m_{p['name']}", a, ps, mn, mx, freqs, [profiles[p["name"]]]))
        r = s.run(device=device, progress=progress)
        results[src] = r
        a_plus, a_minus = r.mode_amplitudes(f"m_{src}")
        a_in = a_plus if direc == "+" else a_minus
        for p in ports:
            ap, am = r.mode_amplitudes(f"m_{p['name']}")
            a_out = am if planes[p["name"]][2] == "+" else ap  # leaving the device
            out[f"{p['name']}@0,{src}@0"] = a_out / a_in
    _copy_symmetric_columns(out, names, source_of)
    if return_results:
        return out, results
    return out
