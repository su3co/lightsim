"""Grating / diffraction-order analysis (Floquet decomposition of DFT fields).

The equivalent of Lumerical's "grating order transmission" analysis: take the
frequency-domain fields recorded by a :func:`~lightsim.spec.DFTMonitor` on a
plane that spans exactly one unit cell of a periodic (or Bloch-periodic)
simulation, expand them in Floquet harmonics

    F(r_t) = sum_{m,n} F_mn exp(i (k_inc + m G_u + n G_v) . r_t),

and split every harmonic into s/p polarised plane waves travelling along +axis
and -axis. Each order's direction, propagating/evanescent character and power
are returned.

Conventions
-----------
* Fields vary as exp(-i w t) (the DFT convention of the core), so a wave
  travelling along +axis is exp(+i k_n x_n).
* ``Results.field`` returns E in V/m and H scaled by eta0 (so |E| = |H| for a
  vacuum plane wave); powers are returned in the same units as
  ``Results.flux`` (W per unit source spectral amplitude squared).
* The monitor must record the four field components tangential to the plane
  (e.g. ``("Ey", "Ez", "Hy", "Hz")`` for ``axis=0``) and the medium at the
  plane must be homogeneous, isotropic and lossless with index ``n_medium``.
* Yee staggering is handled exactly: every component is Fourier transformed at
  its own sample positions, and the half-cell offset between the E and H
  planes along the normal is removed analytically per order.
* Order powers use the same discrete Poynting pairing as the core's
  ``FluxMonitor`` (E on the plane times H half a cell away, i.e. a factor
  cos(k_n dx / 2) per order), so that the sum over propagating orders
  reproduces the flux monitor value and energy conservation holds on the
  grid. The difference to the continuum expression is O((k_n dx)^2 / 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .spec import C0

__all__ = [
    "GratingOrders",
    "diffraction_orders",
    "grating_angles",
    "incident_power",
    "grating_efficiencies",
]

_AX = "xyz"


@dataclass
class GratingOrders:
    """Floquet-order decomposition on one plane.

    All per-order arrays have shape ``(nf, n_orders)``.

    Attributes
    ----------
    freqs : (nf,) analysed frequencies (Hz).
    orders : (n_orders, 2) integer order indices (m, n) along
        ``period_axes[0]`` and ``period_axes[1]`` (n = 0 for a 1-D grating).
    k_t : (nf, n_orders, 2) transverse wavevector along the two in-plane axes
        ``(axis+1) % 3`` and ``(axis+2) % 3`` (rad/m).
    k_n : complex normal wavevector magnitude sqrt(k^2 - |k_t|^2) (rad/m).
    propagating : True where the order is propagating in ``n_medium``.
    theta : polar angle between the order's wavevector and the plane normal
        (radians, 0..pi/2; nan for evanescent orders). The same angle applies
        to the forward (+axis) and backward (-axis) wave of the order.
    phi : azimuth of ``k_t`` measured from in-plane axis ``(axis+1) % 3``
        towards ``(axis+2) % 3`` (radians).
    angles : (nf, n_orders, 2) signed angles asin(k_t / k) along the two
        in-plane axes (the usual grating-equation angle of a 1-D grating).
    power_forward, power_backward : power carried along +axis / -axis through
        the unit cell (both >= 0; 0 for evanescent orders).
    power : net power along +axis (forward - backward).
    power_s_forward, power_p_forward, power_s_backward, power_p_backward :
        split into s (E perpendicular to the plane of incidence, "TE") and p
        (H perpendicular, "TM"). At exactly normal incidence of an order the
        labels are a convention: s = E along ``(axis+2) % 3``.
    amp_s_forward, amp_s_backward : complex s amplitudes (E, V/m) referred to
        the plane of the tangential E samples and to the transverse origin
        r_t = 0.
    amp_p_forward, amp_p_backward : complex p amplitudes (eta0 * H).
    fraction : power fraction per order, see :func:`diffraction_orders`.
    norm : (nf,) the power ``fraction`` is normalised to.
    area : unit-cell area of the monitor plane (m^2).
    """

    freqs: np.ndarray
    orders: np.ndarray
    k_t: np.ndarray
    k_n: np.ndarray
    propagating: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    angles: np.ndarray
    power_forward: np.ndarray
    power_backward: np.ndarray
    power_s_forward: np.ndarray
    power_p_forward: np.ndarray
    power_s_backward: np.ndarray
    power_p_backward: np.ndarray
    amp_s_forward: np.ndarray
    amp_s_backward: np.ndarray
    amp_p_forward: np.ndarray
    amp_p_backward: np.ndarray
    fraction: np.ndarray
    norm: np.ndarray
    area: float
    axis: int
    period_axes: tuple

    @property
    def power(self) -> np.ndarray:
        """Net power along +axis per order, ``power_forward - power_backward``,
        in watts times the squared source spectrum; shape (nf, n_orders)."""
        return self.power_forward - self.power_backward

    @property
    def total_forward(self) -> np.ndarray:
        """Power travelling along +axis summed over the orders, shape (nf,)."""
        return self.power_forward.sum(axis=1)

    @property
    def total_backward(self) -> np.ndarray:
        """Power travelling along -axis summed over the orders (positive), shape (nf,)."""
        return self.power_backward.sum(axis=1)

    @property
    def total(self) -> np.ndarray:
        """Net power along +axis summed over the orders, shape (nf,)."""
        return self.power.sum(axis=1)

    def index(self, m: int, n: int = 0) -> int:
        """Column index of order (m, n)."""
        hit = np.nonzero((self.orders[:, 0] == m) & (self.orders[:, 1] == n))[0]
        if hit.size == 0:
            raise KeyError(f"order ({m}, {n}) was not analysed")
        return int(hit[0])

    def efficiency(self, m: int, n: int = 0) -> np.ndarray:
        """``fraction`` of order (m, n) over frequency, shape (nf,)."""
        return self.fraction[:, self.index(m, n)]

    def as_dict(self, fi: int = 0) -> dict:
        """{(m, n): fraction} of the propagating orders at frequency index ``fi``."""
        return {
            (int(m), int(n)): float(self.fraction[fi, j])
            for j, (m, n) in enumerate(self.orders)
            if self.propagating[fi, j]
        }


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _edges(sim):
    """Grid node coordinates (x, y, z) from a Simulation or a tuple of arrays."""
    if hasattr(sim, "grid"):
        sim = sim.grid()
    ex = [np.asarray(e, dtype=float) for e in sim]
    if len(ex) != 3:
        raise ValueError("sim must be a lightsim.Simulation or the (x, y, z) node arrays of Simulation.grid()")
    return ex


def _is_half(comp: str, ax: int) -> bool:
    """Whether component ``comp`` sits at cell centres along ``ax`` (Yee)."""
    ca = _AX.index(comp[1])
    return (ax == ca) if comp[0] == "E" else (ax != ca)


def _samples(edges, comp, ax, start, count, periodic):
    """Positions and quadrature weights of ``count`` samples of ``comp`` along ``ax``."""
    e = edges[ax]
    d = np.diff(e)
    idx = start + np.arange(count)
    if _is_half(comp, ax):
        pos = 0.5 * (e[idx] + e[idx + 1])
        w = d[idx]
    else:
        pos = e[idx]
        lo = d[idx - 1] if periodic else d[np.maximum(idx - 1, 0)]  # idx-1 = -1 wraps around
        w = 0.5 * (lo + d[np.minimum(idx, d.size - 1)])
    return pos, w


def _field(results, monitor, comp):
    try:
        fd = results.raw["dft"][monitor]["fields"][comp]
    except KeyError as exc:
        raise KeyError(
            f"monitor {monitor!r} has no component {comp}; a grating monitor needs the tangential E and H components"
        ) from exc
    return np.asarray(fd["re"]) + 1j * np.asarray(fd["im"]), [int(v) for v in fd["origin"]]


def _freq_index(results, monitor, freqs):
    mf = np.asarray(results.raw["dft"][monitor]["freqs"], dtype=float)
    if freqs is None:
        return np.arange(mf.size), mf
    want = np.atleast_1d(np.asarray(freqs, dtype=float))
    idx = np.array([int(np.argmin(np.abs(mf - f))) for f in want])
    bad = np.abs(mf[idx] - want) > 1e-6 * np.abs(want)
    if np.any(bad):
        raise ValueError(f"frequencies {want[bad]} were not recorded by monitor {monitor!r} (has {mf})")
    return idx, mf[idx]


def grating_angles(freq, period, n_medium=1.0, k_inc=0.0, orders=None):
    """Grating equation for a 1-D grating.

    Returns ``(orders, angles)`` with ``sin(angle_m) = (k_inc + 2 pi m / period) / (n k0)``
    for the propagating orders (or the requested ``orders``; nan when evanescent).
    ``k_inc`` is the in-plane (Bloch) wavevector of the incident wave,
    ``n_inc * k0 * sin(theta_inc)``.
    """
    k = 2 * np.pi * float(freq) * n_medium / C0
    g = 2 * np.pi / period
    if orders is None:
        orders = np.arange(int(np.ceil((-k - k_inc) / g - 1e-12)), int(np.floor((k - k_inc) / g + 1e-12)) + 1)
    orders = np.asarray(orders, dtype=int)
    s = (k_inc + orders * g) / k
    with np.errstate(invalid="ignore"):
        ang = np.where(np.abs(s) <= 1, np.arcsin(np.clip(s, -1, 1)), np.nan)
    return orders, ang


# ----------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------


def diffraction_orders(
    results,
    monitor: str,
    axis: int,
    period_axes: Sequence[int] | int,
    freqs=None,
    n_medium: float = 1.0,
    kx_inc: float = 0.0,
    ky_inc: float = 0.0,
    *,
    sim,
    incident_power=None,
    direction: str = "net",
    max_order: int | Sequence[int] | None = None,
) -> GratingOrders:
    """Decompose the fields on a DFT monitor plane into diffraction orders.

    Parameters
    ----------
    results : :class:`lightsim.Results`.
    monitor : name of a ``DFTMonitor`` lying in a plane normal to ``axis``
        (zero thickness along ``axis``) and spanning exactly one period of the
        simulation along each of ``period_axes``; it must record the four
        tangential components of E and H.
    axis : plane normal (0, 1, 2).
    period_axes : the periodic in-plane axis (1-D grating) or both in-plane
        axes (2-D grating / crossed grating). With a single axis the fields
        are averaged along the other in-plane axis (the invariant direction
        of a 2-D-equivalent simulation). The simulation must use periodic or
        Bloch boundaries on these axes and the cell must be one period.
    freqs : frequencies to analyse (must be monitor frequencies); ``None``
        analyses all monitor frequencies.
    n_medium : real refractive index of the homogeneous medium at the plane.
    kx_inc, ky_inc : in-plane wavevector of the incident wave (rad/m) along
        ``period_axes[0]`` and ``period_axes[1]``, i.e. the ``k`` of the
        ``Bloch(k)`` boundaries (0 for plain periodic boundaries). Order
        (m, n) has k_t = (kx_inc + 2 pi m / Lx, ky_inc + 2 pi n / Ly). A
        scalar is frequency independent, as in a Bloch simulation; an array
        gives one value per analysed frequency, as for a fixed-angle run
        (``PlaneWave(..., fixed_angle=True)``), where
        ``kx_inc = 2 pi f n sin(angle) / C0``.
    sim : the :class:`lightsim.Simulation` that produced ``results`` (or the
        node arrays returned by ``Simulation.grid()``); provides the sample
        coordinates and the period.
    incident_power : optional (nf,) or scalar power to normalise ``fraction``
        to -- typically the source power from :func:`incident_power` on an
        empty reference run. If ``None``, ``fraction`` is normalised to the
        total power through the plane in the chosen ``direction`` (sum over
        the propagating orders), so the fractions sum to 1.
    direction : which power ``fraction`` refers to: ``"net"`` (forward minus
        backward, default), ``"+"`` (waves travelling along +axis) or ``"-"``
        (along -axis; reported as a positive number).
    max_order : by default only orders that propagate at one of the analysed
        frequencies are returned. Give an int (or one per period axis) to
        return all orders with |m| <= max_order instead, evanescent included.

    Returns
    -------
    :class:`GratingOrders`
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    pax = (int(period_axes),) if np.isscalar(period_axes) else tuple(int(a) for a in period_axes)
    tan = ((axis + 1) % 3, (axis + 2) % 3)
    if not 1 <= len(pax) <= 2 or len(set(pax)) != len(pax) or any(a not in tan for a in pax):
        raise ValueError(f"period_axes must be one or two distinct axes out of {tan} for axis={axis}")
    if direction not in ("net", "+", "-"):
        raise ValueError("direction must be 'net', '+' or '-'")
    n_medium = float(n_medium)
    edges = _edges(sim)
    fidx, f = _freq_index(results, monitor, freqs)
    nf = f.size
    k0 = 2 * np.pi * f / C0
    k = k0 * n_medium
    def _kinc(v):
        a = np.asarray(v, dtype=float)
        if a.ndim == 0:
            return np.full(nf, float(a))
        if a.shape != (nf,):
            raise ValueError(f"kx_inc / ky_inc: give a scalar or one value per analysed frequency ({nf}), got shape {a.shape}")
        return a

    k_inc = {pax[0]: _kinc(kx_inc)}
    if len(pax) == 2:
        k_inc[pax[1]] = _kinc(ky_inc)
    period = {a: float(edges[a][-1] - edges[a][0]) for a in pax}
    ncell = {a: edges[a].size - 1 for a in pax}
    gvec = {a: 2 * np.pi / period[a] for a in pax}

    # order ranges
    if max_order is None:
        mo = [None] * len(pax)
    elif np.isscalar(max_order):
        mo = [int(max_order)] * len(pax)
    else:
        mo = [int(v) for v in max_order]
    rng = {}
    for a, m in zip(pax, mo):
        if m is None:
            lo = int(np.ceil(np.min((-k - k_inc[a]) / gvec[a]) - 1e-9))
            hi = int(np.floor(np.max((k - k_inc[a]) / gvec[a]) + 1e-9))
            rng[a] = np.arange(lo, hi + 1)
        else:
            rng[a] = np.arange(-m, m + 1)
        if rng[a].size == 0:
            rng[a] = np.array([0])
    ords = {a: rng.get(a, np.array([0])) for a in tan}  # per in-plane axis; non-periodic -> order 0

    # Fourier coefficients of the four tangential components, shape (nf, Mb, Mc)
    coef, z_norm = {}, {}
    area = None
    for comp in [f"{t}{_AX[a]}" for t in "EH" for a in tan]:
        arr, origin = _field(results, monitor, comp)
        arr = arr[fidx]
        if arr.shape[1 + axis] != 1:
            raise ValueError(f"monitor {monitor!r} must have zero thickness along axis {axis} (got {arr.shape[1 + axis]} samples of {comp})")
        zpos, _ = _samples(edges, comp, axis, origin[axis], 1, False)
        z_norm[comp] = float(zpos[0])
        a_area = 1.0
        for a in tan:
            cnt = arr.shape[1 + a]
            pos, w = _samples(edges, comp, a, origin[a], cnt, a in pax)
            if a in pax:
                if cnt != ncell[a]:
                    raise ValueError(
                        f"monitor {monitor!r} must span exactly one period along axis {a}: {comp} has {cnt} samples, the cell has {ncell[a]}"
                    )
                kt = k_inc[a][:, None] + ords[a][None, :] * gvec[a]  # (nf, M)
                basis = np.exp(-1j * kt[:, :, None] * pos[None, None, :]) * (w / period[a])  # (nf, M, cnt)
                a_area *= period[a]
            else:
                basis = np.broadcast_to((w / w.sum())[None, None, :], (nf, 1, w.size))  # average along the invariant axis
                a_area *= float(w.sum())
            arr = np.moveaxis(np.einsum("f...c,fmc->f...m", np.moveaxis(arr, 1 + a, -1), basis), -1, 1 + a)
        coef[comp] = np.take(arr, 0, axis=1 + axis)  # (nf, Mb, Mc): remaining axes are in cyclic order
        if axis == 1:  # remaining axes are (x, z) = (tan[1], tan[0])
            coef[comp] = np.swapaxes(coef[comp], 1, 2)
        if comp[0] == "E":
            area = a_area if area is None else min(area, a_area)
    b, c = tan
    eb, ec, hb, hc = (coef[f"{t}{_AX[a]}"].reshape(nf, -1) for t in "EH" for a in tan)
    ze = z_norm[f"E{_AX[b]}"]
    zh = z_norm[f"H{_AX[b]}"]
    if abs(z_norm[f"E{_AX[c]}"] - ze) > 1e-12 * max(abs(ze), 1e-30) + 1e-18 or abs(z_norm[f"H{_AX[c]}"] - zh) > 1e-12 * max(abs(zh), 1e-30) + 1e-18:
        raise RuntimeError("tangential components are not on common planes")
    delta = zh - ze

    mb, mc = np.meshgrid(ords[b], ords[c], indexing="ij")
    mb, mc = mb.ravel(), mc.ravel()
    kb = k_inc.get(b, np.zeros(nf))[:, None] + (mb * gvec.get(b, 0.0))[None, :]
    kc = k_inc.get(c, np.zeros(nf))[:, None] + (mc * gvec.get(c, 0.0))[None, :]
    orders = np.stack([mb if pax[0] == b else mc, (mc if pax[0] == b else mb) if len(pax) == 2 else 0 * mb], axis=1)

    ktm = np.hypot(kb, kc)
    kk = k[:, None]
    deg = ktm < 1e-9 * kk
    safe = np.where(deg, 1.0, ktm)
    ub = np.where(deg, 1.0, kb / safe)  # unit vector along k_t (b, c components)
    uc = np.where(deg, 0.0, kc / safe)
    sb, sc = -uc, ub  # s = n x k_t
    kn = np.sqrt((kk**2 - ktm**2).astype(complex))
    prop = ktm < kk * (1 - 1e-12)
    kn_safe = np.where(np.abs(kn) < 1e-12 * kk, 1e-12 * kk, kn)

    e_s = eb * sb + ec * sc
    e_k = eb * ub + ec * uc
    h_s = hb * sb + hc * sc
    h_k = hb * ub + hc * uc
    ph = kn_safe * delta
    den = 2 * np.cos(ph)
    k0c = k0[:, None]
    # s waves: E_s = a+ + a-,  eta0 H_k = -(kn/k0) (a+ e^{i ph} - a- e^{-i ph})
    q = -h_k * k0c / kn_safe
    a_f = (e_s * np.exp(-1j * ph) + q) / den
    a_b = (e_s * np.exp(1j * ph) - q) / den
    # p waves: eta0 H_s = b+ e^{i ph} + b- e^{-i ph},  E_k = kn/(n^2 k0) (b+ - b-)
    r = e_k * n_medium**2 * k0c / kn_safe
    b_f = (h_s + r * np.exp(-1j * ph)) / den
    b_b = (h_s - r * np.exp(1j * ph)) / den

    eta0 = 376.730313412
    # discrete Poynting weight (E plane x staggered H plane), cf. module docstring
    wgt = np.where(prop, kn.real / k0c * np.cos(ph.real), 0.0) * area / (2 * eta0)
    ps_f = np.abs(a_f) ** 2 * wgt
    ps_b = np.abs(a_b) ** 2 * wgt
    pp_f = np.abs(b_f) ** 2 * wgt / n_medium**2
    pp_b = np.abs(b_b) ** 2 * wgt / n_medium**2
    p_f, p_b = ps_f + pp_f, ps_b + pp_b

    with np.errstate(invalid="ignore"):
        theta = np.where(prop, np.arcsin(np.clip(ktm / kk, 0, 1)), np.nan)
        ang = np.stack([np.where(prop, np.arcsin(np.clip(kb / kk, -1, 1)), np.nan), np.where(prop, np.arcsin(np.clip(kc / kk, -1, 1)), np.nan)], axis=-1)
    phi = np.where(deg, 0.0, np.arctan2(kc, kb))

    chosen = {"net": p_f - p_b, "+": p_f, "-": p_b}[direction]
    if incident_power is None:
        norm = chosen.sum(axis=1)
    else:
        norm = np.broadcast_to(np.asarray(incident_power, dtype=float), (nf,)).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = chosen / norm[:, None]

    return GratingOrders(
        freqs=f,
        orders=orders.astype(int),
        k_t=np.stack([kb, kc], axis=-1),
        k_n=kn,
        propagating=prop,
        theta=theta,
        phi=phi,
        angles=ang,
        power_forward=p_f,
        power_backward=p_b,
        power_s_forward=ps_f,
        power_p_forward=pp_f,
        power_s_backward=ps_b,
        power_p_backward=pp_b,
        amp_s_forward=a_f,
        amp_s_backward=a_b,
        amp_p_forward=b_f,
        amp_p_backward=b_b,
        fraction=fraction,
        norm=norm,
        area=float(area),
        axis=axis,
        period_axes=pax,
    )


def incident_power(reference, monitor, axis, period_axes, freqs=None, n_medium=1.0, kx_inc=0.0, ky_inc=0.0, *, sim, direction="+"):
    """Source power per unit cell from an empty reference run.

    ``reference`` is the result of the same simulation without the grating
    (homogeneous medium ``n_medium`` at the monitor); the power travelling
    along ``direction`` through ``monitor`` is returned, shape (nf,).
    """
    o = diffraction_orders(reference, monitor, axis, period_axes, freqs, n_medium, kx_inc, ky_inc, sim=sim, direction=direction)
    return o.total_forward if direction == "+" else o.total_backward


def grating_efficiencies(
    results,
    t_monitor: str,
    r_monitor: str,
    axis: int,
    period_axes,
    freqs=None,
    n_t: float = 1.0,
    n_r: float = 1.0,
    kx_inc: float = 0.0,
    ky_inc: float = 0.0,
    *,
    sim,
    incident_power,
    direction: str = "+",
    max_order=None,
) -> dict:
    """Transmission and reflection diffraction efficiencies of a grating cell.

    ``t_monitor`` lies behind the grating (medium ``n_t``), ``r_monitor`` in
    front of it (medium ``n_r``); the incident wave travels along
    ``direction`` of ``axis``. ``incident_power`` is the source power per
    unit cell (see :func:`incident_power`). Transmission counts the waves
    travelling along ``direction`` at ``t_monitor``, reflection those
    travelling against it at ``r_monitor`` (the incident wave itself is
    excluded by the direction split, so ``r_monitor`` may sit on either side
    of the source plane).

    Returns a dict with

    * ``"T"``, ``"R"``: :class:`GratingOrders` whose ``fraction`` is the
      diffraction efficiency of each order (power / incident power),
    * ``"T_total"``, ``"R_total"``: (nf,) sums over the propagating orders.
    """
    if direction not in ("+", "-"):
        raise ValueError("direction must be '+' or '-'")
    back = "-" if direction == "+" else "+"
    kw = dict(sim=sim, incident_power=incident_power, max_order=max_order)
    t = diffraction_orders(results, t_monitor, axis, period_axes, freqs, n_t, kx_inc, ky_inc, direction=direction, **kw)
    r = diffraction_orders(results, r_monitor, axis, period_axes, freqs, n_r, kx_inc, ky_inc, direction=back, **kw)
    return {"T": t, "R": r, "T_total": np.nansum(t.fraction, axis=1), "R_total": np.nansum(r.fraction, axis=1)}
