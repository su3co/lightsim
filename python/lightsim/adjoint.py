"""Adjoint-method gradients for inverse design (cf. Lumerical lumopt, FDTDX).

Figure of merit: the power coupled into an output waveguide mode,
F = |a_+|^2, with a_+ the forward mode amplitude at an output plane (per unit
source spectrum). By Lorentz reciprocity, a permittivity change Δε at r
radiates a_+ through the backward-launched output mode:

    da_+ = (i w eps0 / 4) E_adj(r) . E_fwd(r) Δε dV

where E_fwd is the forward field and E_adj the field excited in the device by
a unit-amplitude mode launched *backwards* from the output plane. Hence

    dF/dε(r) = 2 Re[ conj(a_+) (i w eps0 / 4) E_adj(r) . E_fwd(r) ] dV

Two simulations give the gradient with respect to the permittivity of every
Yee cell in the design region.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .spec import C0, DFTMonitor, GaussianPulse, ModeMonitor, ModeSource

EPS0 = 8.8541878128e-12


@dataclass
class AdjointResult:
    """Result of :func:`mode_power_gradient`.

    Attributes
    ----------
    fom : float
        The figure of merit ``abs(a_plus)**2``: power in the output mode per
        unit source spectrum.
    a_plus : complex
        Amplitude of the output mode per unit source spectrum.
    grad : dict
        Per electric component (``"Ex"``, ``"Ey"``, ``"Ez"``):
        ``(origin, array)`` with d(fom)/d(eps_r) of every cell of the design
        box; ``origin`` is the grid index of the first cell.
    volumes : dict
        Per component, the cell volumes (m^3), same shapes.
    da : dict
        Per component, ``(origin, array)`` with the complex
        d(a_plus)/d(eps_r) of every cell.
    """

    fom: float
    a_plus: complex
    #: dF/d(eps_r) per E component: dict comp -> (origin index, array [i, j, k])
    grad: dict
    #: voxel volumes per component (same shapes as grad)
    volumes: dict
    #: complex d(a_+)/d(eps_r) per voxel: dict comp -> (origin, array)
    da: dict = None

    def amplitude_derivative_in_box(self, sim, bmin, bmax) -> complex:
        """Complex d(a_+)/d(eps_r) for a uniform eps change of the block."""
        grid = sim.grid()
        per = _periodic_axes(sim)
        tot = 0j
        for comp, (origin, g) in self.da.items():
            tot += complex(np.sum(g * _fill_fraction(comp, grid, origin, g.shape, bmin, bmax, per)))
        return tot

    def total_gradient_in_box(self, sim, bmin, bmax) -> float:
        """Derivative of F w.r.t. a uniform eps_r change of the block
        [bmin, bmax]: sum over E components of dF/d(eps_r) weighted by the
        voxel fill fraction (the sub-pixel-smoothed eps of a partially filled
        voxel changes by f * d(eps), for both field orientations)."""
        grid = sim.grid()
        per = _periodic_axes(sim)
        tot = 0.0
        for comp, (origin, g) in self.grad.items():
            frac = _fill_fraction(comp, grid, origin, g.shape, bmin, bmax, per)
            tot += float(np.sum(g * frac))
        return tot


def _positions(comp, grid, origin, shape):
    """Coordinates (3 broadcastable arrays) of a component's Yee positions."""
    axis = "xyz".index(comp[1].lower())
    out = []
    for a in range(3):
        nodes = np.asarray(grid[a])
        idx = origin[a] + np.arange(shape[a])
        half = (a == axis) if comp[0] == "E" else (a != axis)
        p = 0.5 * (nodes[idx] + nodes[idx + 1]) if half else nodes[idx]
        sh = [1, 1, 1]
        sh[a] = shape[a]
        out.append(p.reshape(sh))
    return out


def _periodic_axes(sim) -> list[bool]:
    """Which axes of ``sim`` have periodic (or Bloch) boundaries. The
    invariant axis of a 2-D simulation is periodic whatever ``boundaries``
    says: the dual cell of its single node is the whole slab."""
    b = sim.boundaries or {}
    per = [str(b.get(k, [{}])[0].get("type", "")) in ("periodic", "bloch") for k in "xyz"]
    if getattr(sim, "dimensions", 3) == 2:
        per[_axis_index(getattr(sim, "invariant_axis", 2))] = True
    return per


def _axis_index(a) -> int:
    return "xyz".index(a) if isinstance(a, str) else int(a)


def _voxel_segments_1d(comp, grid, origin, shape, a, periodic=False):
    """Extent along axis ``a`` of the cells of a component, as a list of
    ``(lo, hi)`` array pairs whose lengths add up to the cell width.

    A component at half positions along ``a`` owns the primal cell. One at
    the nodes owns the dual cell, from the midpoint of the cell below to the
    midpoint of the cell above; at the first node this is half a cell next to
    a wall, but on a periodic axis the dual cell wraps around and its other
    half lies at the far end of the axis (the second segment).
    """
    axis = "xyz".index(comp[1].lower())
    nodes = np.asarray(grid[a])
    n = len(nodes) - 1
    idx = origin[a] + np.arange(shape[a])
    half = (a == axis) if comp[0] == "E" else (a != axis)
    if half:
        return [(nodes[idx], nodes[idx + 1])]
    lo = np.where(idx > 0, 0.5 * (nodes[np.maximum(idx - 1, 0)] + nodes[idx]), nodes[idx])
    hi = np.where(idx < n, 0.5 * (nodes[idx] + nodes[np.minimum(idx + 1, n)]), nodes[idx])
    if not periodic:
        return [(lo, hi)]
    first = idx == 0
    wrap_lo = np.where(first, 0.5 * (nodes[n - 1] + nodes[n]), nodes[n])
    wrap_hi = np.full(len(idx), nodes[n])
    return [(lo, hi), (wrap_lo, wrap_hi)]


def _fill_fraction(comp, grid, origin, shape, bmin, bmax, periodic=(False, False, False)):
    f = np.ones(shape)
    for a in range(3):
        segs = _voxel_segments_1d(comp, grid, origin, shape, a, periodic[a])
        width = sum(hi - lo for lo, hi in segs)
        ov = sum(np.clip(np.minimum(hi, bmax[a]) - np.maximum(lo, bmin[a]), 0, None) for lo, hi in segs) / width
        sh = [1, 1, 1]
        sh[a] = shape[a]
        f = f * ov.reshape(sh)
    return f


def _volumes(comp, grid, origin, shape, periodic=(False, False, False)):
    """Volume of the Yee voxel of every sample of a component box. Along a
    periodic axis the dual cell of node 0 wraps around (it is the full cell of
    a one-cell-thick 2-D simulation)."""
    axis = "xyz".index(comp[1].lower())
    vol = np.ones(shape)
    for a in range(3):
        nodes = np.asarray(grid[a])
        n = len(nodes) - 1
        idx = origin[a] + np.arange(shape[a])
        half = (a == axis) if comp[0] == "E" else (a != axis)
        if half:
            w = nodes[idx + 1] - nodes[idx]
        else:
            lo = np.where(idx > 0, 0.5 * (nodes[np.maximum(idx - 1, 0)] + nodes[idx]), nodes[idx])
            hi = np.where(idx < n, 0.5 * (nodes[idx] + nodes[np.minimum(idx + 1, n)]), nodes[idx])
            w = hi - lo
            if periodic[a]:
                w = np.where(idx == 0, 0.5 * ((nodes[1] - nodes[0]) + (nodes[n] - nodes[n - 1])), w)
        sh = [1, 1, 1]
        sh[a] = shape[a]
        vol = vol * w.reshape(sh)
    return vol


def amplitude_sensitivity(e_fwd, e_adj, volume, freq):
    """Complex d(a_+)/d(eps_r) of every Yee voxel: (i w eps0 / 4) E_adj E_fwd dV.

    ``e_fwd`` is the forward field per unit incident amplitude, ``e_adj`` the
    field of a unit-amplitude mode launched backwards from the output plane,
    both at ``freq`` and at the positions of one E component. Shared by
    :func:`mode_power_gradient` and :mod:`lightsim.invdes`.
    """
    return (1j * 2 * np.pi * np.asarray(freq) * EPS0 / 4.0) * e_adj * e_fwd * volume


def _only_source(results) -> str:
    """Name of the single source of a run (these functions take one source)."""
    names = [q["name"] for q in results.raw["sources"]]
    if len(names) != 1:
        raise ValueError(f"the simulation must contain exactly one source, it has {len(names)}: {names}")
    return names[0]


_GRADIENT_REFUSALS = (
    ("raman", "has a Raman (delayed chi3) response: the adjoint needs a linear medium, and reverse mode does not step the Raman oscillator backwards"),
    ("gain", "is a four-level gain medium: nonlinear (saturable), with population state the adjoint does not differentiate"),
    ("gyro", "is gyrotropic (magneto-optic): its permittivity tensor is not symmetric, the medium is not reciprocal, and the adjoint run would need the bias reversed"),
    ("director", "has a director field: where the director varies from cell to cell the discrete off-diagonal update is not symmetric, so the adjoint run is not its transpose (a uniform director is an off-diagonal medium: give it with eps_off / Medium.rotated, which engines 'native' and 'python' handle)"),
)


def check_gradient_media(sim) -> None:
    """Raise ``ValueError`` if ``sim`` uses a medium that no gradient engine
    models: a delayed (Raman) chi3 response, a four-level gain medium, a
    gyrotropic pole (not reciprocal) or a director field. Used by
    :func:`mode_power_gradient` and :class:`lightsim.invdes.SMatrixProblem`
    (all three engines); the Rust engines check the same media. A
    fixed-angle run (a plane wave with ``fixed_angle=True``) is refused too:
    no gradient engine models its update."""
    if any(s.get("fixed_angle") for s in sim.sources):
        raise ValueError(
            "the simulation is a fixed-angle run (PlaneWave(..., fixed_angle=True)), which no gradient engine models; "
            "use Bloch boundaries with a fixed transverse wavevector for gradients at oblique incidence"
        )
    for sh in getattr(sim, "sheets", None) or []:
        raise ValueError(f"sheet '{sh.get('name', '')}': zero-thickness sheets are not modelled by any gradient engine (their surface polarization is state that neither the adjoint nor reverse mode carries); use a one-cell layer (lightsim.sheets with a thickness) next to a design region instead")
    used = {s["medium"] for s in sim.structures} | {sim.background}
    for m in sim.media:
        if m["name"] not in used:
            continue
        for key, why in _GRADIENT_REFUSALS:
            if m.get(key):
                raise ValueError(f"medium '{m['name']}' {why}; no gradient engine supports it")


def mode_power_gradient(
    sim,
    design_min,
    design_max,
    output: dict,
    freq: float,
    device: str = "auto",
):
    """Gradient of F = |a_+(freq)|^2 w.r.t. eps_r in the design box.

    ``sim`` must already contain the forward source. ``output`` describes the
    output plane: dict(axis, position, min, max, profile, direction) where
    ``direction`` is the propagation direction of the outgoing mode ("+" or "-").
    """
    check_gradient_media(sim)
    comps = ("Ex", "Ey", "Ez")
    # ---------------- forward ----------------
    fwd = copy.deepcopy(sim)
    fwd.monitors = [m for m in fwd.monitors]
    fwd.add_monitor(DFTMonitor("__design__", design_min, design_max, [freq], components=comps))
    fwd.add_monitor(ModeMonitor("__out__", output["axis"], output["position"], output["min"], output["max"], [freq], [output["profile"]]))
    rf = fwd.run(device=device)
    src_name = _only_source(rf)
    _, sf = rf.source_spectrum(src_name, freqs=[freq])
    ap, am = rf.mode_amplitudes("__out__")
    a_out = (ap if output["direction"] == "+" else am)[0] / sf[0]
    # ---------------- adjoint ----------------
    adj = copy.deepcopy(sim)
    adj.sources = []
    adj.monitors = []
    back = "-" if output["direction"] == "+" else "+"
    sig = GaussianPulse(freq, 0.1 * freq)
    adj.add_source(ModeSource(output["axis"], output["position"], back, output["min"], output["max"], output["profile"], sig, name="__adj__"))
    adj.add_monitor(DFTMonitor("__design__", design_min, design_max, [freq], components=comps))
    ra = adj.run(device=device)
    _, sa = ra.source_spectrum("__adj__", freqs=[freq])
    grid = sim.grid()
    per = _periodic_axes(sim)
    grad = {}
    vols = {}
    das = {}
    for c in comps:
        ef = rf.field("__design__", c)[0] / sf[0]
        ea = ra.field("__design__", c)[0] / sa[0]
        origin = rf.raw["dft"]["__design__"]["fields"][c]["origin"]
        vol = _volumes(c, grid, origin, ef.shape, per)
        dav = amplitude_sensitivity(ef, ea, vol, freq)
        g = 2.0 * np.real(np.conj(a_out) * dav)
        grad[c] = (origin, g)
        vols[c] = vol
        das[c] = (origin, dav)
    return AdjointResult(fom=float(abs(a_out) ** 2), a_plus=complex(a_out), grad=grad, volumes=vols, da=das)


def mode_power(sim, output: dict, freq: float, device: str = "auto") -> float:
    """F = |a_+|^2 per unit source spectrum (for finite-difference checks)."""
    s = copy.deepcopy(sim)
    s.add_monitor(ModeMonitor("__out__", output["axis"], output["position"], output["min"], output["max"], [freq], [output["profile"]]))
    r = s.run(device=device)
    _, sf = r.source_spectrum(_only_source(r), freqs=[freq])
    ap, am = r.mode_amplitudes("__out__")
    a = (ap if output["direction"] == "+" else am)[0] / sf[0]
    return float(abs(a) ** 2)
