"""User-defined material plug-ins written in Python (``lightsim.plugins``):
one Lorentz pole and the instantaneous Kerr medium as NumPy callables
against the built-in media, the cost of the callables, the divergence
check on a plug-in that adds energy, and the refusals."""

import os
import time

import numpy as np
import pytest

import lightsim as ls
from lightsim import plugins
from lightsim.adjoint import check_gradient_media

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1e-6
F0 = ls.C0 / LAM
REAL = np.float64 if ls.precision == "f64" else np.float32


def lorentz_coefficients(delta_eps, w0, gamma, dt):
    """ADE coefficients (c0, c1, c2) of a Lorentz pole (as the core's
    Pole::ade_coefficients), rounded to the field precision."""
    a = 1 / dt**2 + gamma / (2 * dt)
    return [REAL(v) for v in ((2 / dt**2 - w0**2) / a, -(1 / dt**2 - gamma / (2 * dt)) / a, delta_eps * w0**2 / a)]


def lorentz_before(e, state, ctx):
    p = ctx["params"]
    c0, c1, c2 = lorentz_coefficients(p["delta_eps"], p["omega0"], p["gamma"], ctx["dt"])
    p0 = state[:, 0].copy()
    pn = c0 * p0 + c1 * state[:, 1] + c2 * e
    state[:, 2] = pn - p0
    state[:, 1] = p0
    state[:, 0] = pn


def lorentz_after(e, e_prev, inv_eps, state, ctx):
    e -= inv_eps * state[:, 2]


def kerr_after(e, e_prev, inv_eps, state, ctx):
    # the built-in implicit step: u + c3 (3 E^2 u + 3 E u^2 + u^3) = E_lin - E, four Newton steps
    b = REAL(ctx["params"]["chi3"]) * inv_eps
    d = e - e_prev
    u = d / (REAL(1) + REAL(3) * b * e_prev * e_prev)
    for _ in range(4):
        f = u + b * (REAL(3) * e_prev * e_prev * u + REAL(3) * e_prev * u * u + u * u * u) - d
        fp = REAL(1) + b * (REAL(3) * e_prev * e_prev + REAL(6) * e_prev * u + REAL(3) * u * u)
        u = u - f / fp
    e[:] = e_prev + u


def column(medium, dx, run_time, cross=2):
    sim = ls.Simulation(min=[0, 0, 0], max=[5e-6, cross * dx, cross * dx], mesh=ls.UniformMesh(dx), run_time=run_time)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_structure(ls.Box([1.5e-6, -1, -1], [3.5e-6, 1, 1]), medium)
    sim.add_source(ls.PlaneWave(0, 1e-6, "+", "Ey", ls.GaussianPulse(F0, 0.3 * F0)))
    sim.add_monitor(ls.ProbeMonitor("p", [4e-6, 0, 0], ["Ey"]))
    sim.add_monitor(ls.ProbeMonitor("in", [2.5e-6, 0, 0], ["Ey"]))
    return sim


def rel_diff(a, b, name):
    x, y = np.asarray(a.time_series(name, "Ey")[1]), np.asarray(b.time_series(name, "Ey")[1])
    return float(np.max(np.abs(x - y)) / np.max(np.abs(x)))


def test_lorentz_plugin_matches_builtin():
    """One Lorentz pole (eps_inf 1.5, conductive background) as Python
    callables against the built-in pole: identical on the CPU (the same
    single-precision arithmetic in the same order)."""
    w0 = 2 * np.pi * F0
    pars = {"delta_eps": 1.5, "omega0": 1.3 * w0, "gamma": 0.1 * w0}
    plugins.register("py_lorentz", after=lorentz_after, before=lorentz_before, state=3)
    builtin = ls.Medium("lor", eps=1.5, sigma=2e4, poles=[ls.Lorentz(pars["delta_eps"], pars["omega0"], pars["gamma"])])
    plug = ls.Medium("lor", eps=1.5, sigma=2e4, plugin=plugins.use("py_lorentz", pars))
    a = column(builtin, LAM / 60, 150e-15).run(device=DEVICE)
    b = column(plug, LAM / 60, 150e-15).run(device=DEVICE)
    d = max(rel_diff(a, b, "p"), rel_diff(a, b, "in"))
    print(f"Lorentz plug-in vs built-in on {DEVICE}: max |diff| / peak = {d:.2e}")
    assert d <= (0.0 if DEVICE == "cpu" else 2e-5)


def test_kerr_plugin_matches_builtin():
    """Instantaneous Kerr medium as a Python callable (the implicit step of
    the built-in update) against the built-in chi3 medium."""
    chi = 0.01 / (1.5 * 2 * np.pi * 2e-6 / LAM)
    plugins.register("py_kerr", after=kerr_after)
    a = column(ls.Medium("kerr", chi3=chi), LAM / 80, 45e-15).run(device=DEVICE)
    b = column(ls.Medium("kerr", plugin=plugins.use("py_kerr", {"chi3": chi})), LAM / 80, 45e-15).run(device=DEVICE)
    d = max(rel_diff(a, b, "p"), rel_diff(a, b, "in"))
    print(f"Kerr plug-in vs built-in chi3 on {DEVICE}: max |diff| / peak = {d:.2e}")
    assert d < 2e-5


def test_plugin_cost():
    """Cost of a Python plug-in against the built-in pole, a 3-D slab of
    40 x 40 x 40 cells in a 120 x 40 x 40 grid (printed, not asserted)."""
    w0 = 2 * np.pi * F0
    pars = {"delta_eps": 1.5, "omega0": 1.3 * w0, "gamma": 0.1 * w0}
    plugins.register("py_lorentz", after=lorentz_after, before=lorentz_before, state=3)
    dx = LAM / 40
    times = {}
    for kind in ("builtin", "plugin"):
        med = ls.Medium("lor", eps=1.5, poles=[ls.Lorentz(pars["delta_eps"], pars["omega0"], pars["gamma"])]) if kind == "builtin" else ls.Medium("lor", eps=1.5, plugin=plugins.use("py_lorentz", pars))
        sim = ls.Simulation(min=[0, 0, 0], max=[120 * dx, 40 * dx, 40 * dx], mesh=ls.UniformMesh(dx), run_time=300 * 0.99 * dx / (ls.C0 * 3**0.5))
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        sim.add_structure(ls.Box([40 * dx, -1, -1], [80 * dx, 1, 1]), med)
        sim.add_source(ls.PlaneWave(0, 20 * dx, "+", "Ey", ls.GaussianPulse(F0, 0.3 * F0)))
        t0 = time.perf_counter()
        r = sim.run(device=DEVICE)
        times[kind] = time.perf_counter() - t0
        print(f"{kind}: {r.stats['steps']} steps, {r.stats['seconds']:.2f} s, {r.stats['mcells_per_s']:.0f} Mcells/s")
    print(f"Python plug-in / built-in pole on {DEVICE}: {times['plugin'] / times['builtin']:.1f}x the run time")


def test_runaway_plugin_is_stopped():
    """A plug-in that amplifies the field every step raises the solver's
    divergence error instead of running on."""
    def amplify(e, e_prev, inv_eps, state, ctx):
        e *= REAL(1.002)

    plugins.register("py_runaway", after=amplify)
    sim = column(ls.Medium("gain", plugin=plugins.use("py_runaway")), LAM / 60, 2000e-15)
    sim.shutoff_interval = 100
    with pytest.raises(RuntimeError, match="unstable"):
        sim.run(device=DEVICE)


def test_plugin_errors_and_refusals():
    """An exception in a callable stops the run with its text; an
    unregistered name, Bloch boundaries and the gradient engines are refused."""
    def broken(e, e_prev, inv_eps, state, ctx):
        raise ValueError("model blew up")

    plugins.register("py_broken", after=broken)
    with pytest.raises(RuntimeError, match="model blew up"):
        column(ls.Medium("b", plugin=plugins.use("py_broken")), LAM / 40, 5e-15).run(device=DEVICE)
    with pytest.raises(RuntimeError, match="not registered"):
        column(ls.Medium("b", plugin=plugins.use("no_such_plugin")), LAM / 40, 5e-15).run(device=DEVICE)
    sim = column(ls.Medium("b", plugin=plugins.use("py_broken")), LAM / 40, 5e-15)
    sim.set_boundaries(y=ls.Bloch(1e5))
    with pytest.raises(RuntimeError, match="Bloch"):
        sim.run(device=DEVICE)
    with pytest.raises(ValueError, match="plug-in"):
        check_gradient_media(column(ls.Medium("b", plugin=plugins.use("py_broken")), LAM / 40, 5e-15))
    plugins.unregister("py_broken")
