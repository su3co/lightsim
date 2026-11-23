"""Sub-pixel smoothing of transparent dispersive dielectrics: 2-D Mie
scattering by a cylinder of fitted Si3N4 in fitted SiO2 (E in the plane, so
the normal-component rule is exercised), against the analytic series with
the pole model's eps(w), and against the same cylinder of plain dielectrics
with the fitted permittivities at the centre frequency."""

import os

import numpy as np
from scipy import special as sp

import lightsim as ls
from lightsim import materials as lm

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
ETA0 = 376.730313412
LAM0 = 1.55e-6
F0 = ls.C0 / LAM0
R = 0.45e-6


def q_sca_te(m, x, nmax=40):
    """Scattering efficiency of a cylinder, E perpendicular to the axis."""
    q = 0.0
    for n in range(nmax + 1):
        jm, jmp = sp.jv(n, m * x), sp.jvp(n, m * x)
        jx, jxp = sp.jv(n, x), sp.jvp(n, x)
        hx, hxp = sp.hankel1(n, x), sp.h1vp(n, x)
        a = (m * jm * jxp - jmp * jx) / (m * jm * hxp - jmp * hx)
        q += abs(a) ** 2 * (1 if n == 0 else 2)
    return 2 / x * q


def eps_of(med, f):
    if not med.get("poles"):
        return med["eps"][0]
    return float(lm.pole_permittivity(med, 2 * np.pi * f, axis=0).real)


def q_sca(core, clad, cells, subpixel=True):
    dx = LAM0 / (2.0 * cells)
    half = 1.6e-6
    sim = ls.Simulation(min=[-half, -half], max=[half, half], mesh=ls.UniformMesh(dx), run_time=200e-15,
                        subpixel=subpixel, polarization="te", shutoff=1e-7)
    sim.add_medium(clad)
    sim.background = clad["name"]
    sim.add_structure(ls.Cylinder([0.31 * dx, 0.11 * dx], R), core)
    a = 1.05e-6
    sim.add_source(ls.TFSF([-a, -a], [a, a], 0, "+", "Ey", ls.GaussianPulse(F0, 0.15 * F0), name="tf"))
    b = 1.3e-6
    for nm, ax, pos in [("xp", 0, b), ("xm", 0, -b), ("yp", 1, b), ("ym", 1, -b)]:
        lo, hi = [-b, -b], [b, b]
        lo[ax] = hi[ax] = pos
        sim.add_monitor(ls.FluxMonitor(nm, ax, pos, lo, hi, [F0]))
    res = sim.run(device=DEVICE)
    p = res.flux("xp")[0] - res.flux("xm")[0] + res.flux("yp")[0] - res.flux("ym")[0]
    _, s = res.source_spectrum("tf")
    inc = 0.5 * np.sqrt(eps_of(clad, F0)) / ETA0 * abs(s[0]) ** 2
    return p / inc / (2 * R)


def test_fitted_cylinder_smoothed_like_plain_dielectrics():
    band = (1.45e-6, 1.65e-6)
    core = lm.fitted_medium("Si3N4", band=band, medium_name="core")
    clad = lm.fitted_medium("SiO2", band=band, medium_name="clad")
    plain_core = ls.Medium("core", eps=eps_of(core, F0))
    plain_clad = ls.Medium("clad", eps=eps_of(clad, F0))
    nc, nl = np.sqrt(eps_of(core, F0)), np.sqrt(eps_of(clad, F0))
    exact = q_sca_te(nc / nl, 2 * np.pi * F0 / ls.C0 * nl * R)
    errs = {}
    for cells in (15, 30):
        smooth = q_sca(core, clad, cells)
        plain = q_sca(plain_core, plain_clad, cells)
        stair = q_sca(core, clad, cells, subpixel=False)
        errs[cells] = (smooth / exact - 1, stair / exact - 1)
        print(f"{cells} cells per wavelength in the core: Q smoothed {smooth:.5f} (error {smooth / exact - 1:+.2e}), "
              f"plain dielectrics {plain:.5f}, staircase {stair:.5f} (error {stair / exact - 1:+.2e}), Mie {exact:.5f}")
        # the smoothed dispersive cylinder is the plain-dielectric one at w_c
        assert abs(smooth / plain - 1) < 1e-3
    # second order: the smoothed error falls by more than 3 when the cell is halved
    assert abs(errs[30][0]) < abs(errs[15][0]) / 3
    assert abs(errs[30][0]) < 4e-3


def test_mode_solver_smooths_fitted_media_like_the_run():
    """A mode solved before any source is added (the usual order) sees the
    fitted guide smoothed at the frequency of the solve, as the run with a
    source at that frequency does: its effective index is that of the
    plain-dielectric guide, not that of the staircase."""
    band = (1.45e-6, 1.65e-6)
    core = lm.fitted_medium("Si3N4", band=band, medium_name="core")
    clad = lm.fitted_medium("SiO2", band=band, medium_name="clad")

    def neff(c, l, subpixel):
        sim = ls.Simulation(min=[-1e-6, -2e-6], max=[1e-6, 2e-6], mesh=ls.UniformMesh(40e-9), run_time=1e-13,
                            subpixel=subpixel, polarization="te")
        sim.add_medium(l)
        sim.background = l["name"]
        sim.add_structure(ls.Box([-3e-6, -0.513e-6], [3e-6, 0.487e-6]), c)
        return ls.solve_modes(sim, 0, 0.0, [0.0, -2e-6], [0.0, 2e-6], F0, num_modes=1)[0].neff.real

    plain = neff(ls.Medium("core", eps=eps_of(core, F0)), ls.Medium("clad", eps=eps_of(clad, F0)), True)
    smooth, stair = neff(core, clad, True), neff(core, clad, False)
    print(f"neff smoothed {smooth:.7f}, plain dielectrics {plain:.7f}, staircase {stair:.7f}")
    assert abs(smooth - plain) < 1e-5
    assert abs(stair - plain) > 10 * abs(smooth - plain)
