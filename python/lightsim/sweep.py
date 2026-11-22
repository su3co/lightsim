"""Spectra at a fixed angle of incidence from a sweep of oblique runs.

An oblique :func:`~lightsim.spec.PlaneWave` with Bloch boundaries fixes the
transverse wavevector ``k_t``, so in one broadband run the angle of
incidence changes with frequency: ``sin(theta(f)) = k_t c / (2 pi n f)``.
:func:`sweep_angle` runs one simulation per frequency point instead, each
with ``k_t = n k0(f) sin(theta)``, so that every point of the assembled
spectrum is at the same angle. The runs are independent and can be spread
over several devices.

This is a different method from Lumerical's BFAST (broadband fixed-angle
source technique), which gets a fixed-angle spectrum from a single run with
a modified update; the sweep gets the same kind of result from several
ordinary runs, one per frequency.
"""

from __future__ import annotations

import math
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .spec import C0, GaussianPulse

__all__ = ["sweep_angle", "AngleSweep"]

ETA0 = 376.730313412


@dataclass
class AngleSweep:
    """Result of :func:`sweep_angle`: one run per frequency.

    Attributes
    ----------
    freqs : (nf,) frequencies (Hz), in the order given.
    angle : the angle of incidence (rad).
    k_t : (nf,) transverse wavevectors used (rad/m).
    results : list of :class:`lightsim.spec.Results`, one per frequency.
    """

    freqs: np.ndarray
    angle: float
    n_medium: float
    k_t: np.ndarray
    results: list = field(repr=False)

    def _index(self, res, monitor, f):
        fm = np.asarray(res.freqs(monitor), dtype=float)
        i = int(np.argmin(np.abs(fm - f)))
        if abs(fm[i] - f) > 1e-9 * f:
            raise ValueError(f"monitor {monitor!r} does not record {f:.6g} Hz; give it the frequency passed to build()")
        return i

    def flux(self, monitor: str) -> np.ndarray:
        """Flux of ``monitor`` at each run's own frequency, shape (nf,)."""
        out = np.empty(len(self.freqs))
        for j, (res, f) in enumerate(zip(self.results, self.freqs)):
            fl = np.asarray(res.flux(monitor))
            out[j] = fl[self._index(res, monitor, f)]
        return out

    def source_spectrum(self, source: str | None = None) -> np.ndarray:
        """Complex source spectrum S at each run's frequency, shape (nf,)."""
        out = np.empty(len(self.freqs), dtype=complex)
        for j, (res, f) in enumerate(zip(self.results, self.freqs)):
            _, s = res.source_spectrum(source, freqs=[f])
            out[j] = np.asarray(s)[0]
        return out

    def field(self, monitor: str, component: str) -> np.ndarray:
        """DFT field of ``monitor`` at each run's frequency, shape (nf, ...)."""
        out = []
        for res, f in zip(self.results, self.freqs):
            fl = np.asarray(res.field(monitor, component))
            out.append(fl[self._index(res, monitor, f)])
        return np.stack(out)

    def incident_power(self, area: float, source: str | None = None, amplitude: float = 1.0) -> np.ndarray:
        """Analytic power of the plane wave through ``area`` (m^2) of the
        source plane: ``n |amplitude S|^2 area cos(angle) / (2 eta0)``."""
        s = self.source_spectrum(source)
        return self.n_medium * np.abs(amplitude * s) ** 2 * area * math.cos(self.angle) / (2 * ETA0)


def _default_bandwidth(angle: float) -> float:
    # keep the pulse spectrum (4 standard deviations) above f sin(angle), the
    # frequency at which the fixed k_t turns grazing, and below 10 %
    return float(min(0.1, max(0.01, (1.0 - abs(math.sin(angle))) / 4.0)))


def sweep_angle(
    build: Callable,
    freqs: Sequence[float],
    angle: float,
    *,
    n_medium: float = 1.0,
    bandwidth: float | None = None,
    devices: str | Sequence[str] = "cpu",
    progress: bool = False,
) -> AngleSweep:
    """Spectra at the fixed angle ``angle`` (rad) from one oblique run per frequency.

    For every frequency ``f`` of ``freqs`` the sweep calls
    ``build(f, signal, k_t)`` and runs the simulation it returns, where

    * ``signal`` is ``GaussianPulse(f, bandwidth * f)``: give it to the
      ``PlaneWave`` (with ``angle=angle``), whose phase gradient is then
      computed at ``f``;
    * ``k_t = n_medium * 2 pi f / c * sin(angle)`` is the transverse
      wavevector: give it to the ``Bloch`` boundaries along the tilt axis;
    * the monitors must record ``f`` (they may record more frequencies; only
      ``f`` is read).

    ``n_medium`` is the index at the source plane. ``bandwidth`` is the
    relative spectral width of the pulses; the default keeps the pulse well
    above ``f sin(angle)``, below which the fixed ``k_t`` is evanescent, and
    is at most 0.1. ``devices`` is a device string or a list of them; the
    runs are dispatched concurrently, one at a time per entry (``["cuda:0",
    "cuda:1"]``; an entry may itself be ``"cuda:0,1"``).

    Returns an :class:`AngleSweep`; ``sweep.flux(name)`` gives the flux of a
    monitor at each run's frequency, ``sweep.incident_power(area)`` the
    analytic incident power.
    """
    f = np.asarray(freqs, dtype=float).ravel()
    if f.size == 0 or np.any(f <= 0):
        raise ValueError("freqs must be positive")
    if not abs(angle) < 0.5 * math.pi:
        raise ValueError("angle must be below 90 degrees")
    bw = _default_bandwidth(angle) if bandwidth is None else float(bandwidth)
    kt = n_medium * 2 * math.pi * f / C0 * math.sin(angle)
    devs = [devices] if isinstance(devices, str) else list(devices)
    if not devs:
        raise ValueError("devices is empty")
    pool: queue.Queue = queue.Queue()
    for d in devs:
        pool.put(d)

    def work(j):
        sim = build(float(f[j]), GaussianPulse(float(f[j]), bw * float(f[j])), float(kt[j]))
        dev = pool.get()
        try:
            res = sim.run(device=dev)
        finally:
            pool.put(dev)
        if progress:
            print(f"sweep_angle: {f[j]:.6g} Hz done on {dev}", flush=True)
        return res

    if len(devs) == 1:
        results = [work(j) for j in range(f.size)]
    else:
        with ThreadPoolExecutor(max_workers=len(devs)) as ex:
            results = list(ex.map(work, range(f.size)))
    return AngleSweep(freqs=f, angle=float(angle), n_medium=float(n_medium), k_t=kt, results=results)
