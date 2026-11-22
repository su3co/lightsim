"""Mode-solver features against exact references.

* slab waveguide TE and TM effective indices, and their second-order
  convergence on interface-conforming grids (Richardson extrapolation);
* a leaky slab (silicon slab on a buried oxide over a silicon substrate)
  with perfectly matched layers: the complex root of the exact dispersion
  relation;
* step-index fibre HE11 (exact vector eigenvalue equation);
* a 2-D slab bend against the exact Bessel / Hankel solution of complex
  order (needs mpmath, skipped without it) and the asymptotic formula;
* coupled slabs: even/odd supermodes and the coupling length;
* full anisotropy: a uniaxial slab with the optic axis in every plane,
  against the 4x4 (Berreman) transfer matrix;
* group index (energy-flux relation) against a finite difference in
  wavelength; overlaps, confinement, effective area;
* the cross-section rasterizer against the Rust core's sub-pixel smoothing;
* the gplugins-style functions.
"""

import math

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from lightsim.modes import C0, mode_overlap
from lightsim.xsection import CrossSection, uniaxial

LAM = 1.55e-6
K0 = 2 * np.pi / LAM
N_SI, N_OX = 3.476, 1.444


# ----------------------------------------------------------------------------
# exact references
# ----------------------------------------------------------------------------


def slab_det(neff, n_layers, t_layers, n_sub, n_clad, pol="te"):
    """Dispersion function of a multilayer slab: substrate wave outgoing
    (leaky) or decaying, cladding wave decaying."""

    def p(n):
        return 1.0 if pol == "te" else n**2

    ks = K0 * np.sqrt(complex(n_sub**2 - neff**2))
    if ks.real < 0:
        ks = -ks
    kc = 1j * K0 * np.sqrt(complex(neff**2 - n_clad**2))
    U = np.array([1.0 + 0j, -1j * ks / p(n_sub)])
    for n, t in zip(n_layers, t_layers):
        k = K0 * np.sqrt(complex(n**2 - neff**2))
        c, s = np.cos(k * t), np.sin(k * t)
        U = np.array([[c, p(n) * s / k], [-k * s / p(n), c]]) @ U
    return U[1] - 1j * kc / p(n_clad) * U[0]


def newton(f, x0, tol=1e-14):
    x = complex(x0)
    for _ in range(100):
        d = (f(x + 1e-7) - f(x - 1e-7)) / 2e-7
        dx = f(x) / d
        x -= dx
        if abs(dx) < tol:
            break
    return x


def slab_exact(n1, n2, t, tm):
    v2 = K0**2 * (n1**2 - n2**2)
    r = (n1 / n2) ** 2 if tm else 1.0
    f = lambda k: k * np.tan(k * t / 2) - r * np.sqrt(v2 - k * k)  # noqa: E731
    kap = brentq(f, 1e-3, min(np.pi / t, np.sqrt(v2)) * (1 - 1e-12), xtol=1e-12, rtol=1e-15)
    return np.sqrt(n1**2 - (kap / K0) ** 2)


def berreman_K(eps, beta):
    e, b = np.asarray(eps, dtype=complex), beta / K0
    cEy = np.array([-e[1, 0], -e[1, 2], -b, 0]) / e[1, 1]
    K = np.zeros((4, 4), complex)
    K[0] = [0, 0, 0, -1j * K0]
    K[1] = np.array([0, 0, 1j * K0, 0]) + 1j * beta * cEy
    K[2] = 1j * K0 * (np.array([e[2, 0], e[2, 2], 0, 0]) + e[2, 1] * cEy)
    K[3] = -1j * K0 * (np.array([e[0, 0], e[0, 2], 0, 0]) + e[0, 1] * cEy) + np.array([1j * beta * b, 0, 0, 0])
    return K


def berreman_det(neff, eps_core, t, n_out):
    beta = K0 * neff
    eo = n_out**2 * np.eye(3)

    def proj(sign):
        K = berreman_K(eo, beta)
        q = np.sqrt(complex(beta**2 - K0**2 * n_out**2))
        q = -q if q.real < 0 else q
        return ((K + sign * q * np.eye(4)) / (2 * q * sign))[:, [0, 2]]

    T = expm(berreman_K(eps_core, beta) * t)
    return np.linalg.det(np.c_[T @ proj(+1), proj(-1)])


def slab_xs(layers, step, half=1.5e-6, background=N_OX):
    xs = CrossSection(background=background)
    lo, hi = np.inf, -np.inf
    for v0, v1, mat in layers:
        xs.add_layer(v0, v1, mat)
        lo, hi = min(lo, v0), max(hi, v1)
    xs.window = ((0.0, lo - half), (step, hi + half))
    xs.mesh(xs.window, step=step, max_step=step)
    xs.set_mesh([0.0, step], xs.v_nodes)
    return xs


def pick(modes, te):
    return [m for m in modes if (m.te_fraction > 0.5) == te][0]


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_slab_second_order_and_extrapolation(pol):
    t = 0.22e-6
    ref = slab_exact(N_SI, N_OX, t, pol == "tm")
    errs = []
    for step in (22e-9, 11e-9):
        xs = slab_xs([(-t / 2, t / 2, N_SI)], step)
        errs.append(pick(xs.solve(LAM, 4, periodic=(True, False)), pol == "te").neff.real - ref)
    ratio = errs[0] / errs[1]
    xs = slab_xs([(-t / 2, t / 2, N_SI)], 22e-9)
    ext = pick(xs.solve(LAM, 4, periodic=(True, False), extrapolate=True), pol == "te").neff.real - ref
    print(f"{pol}: errors {errs[0]:+.2e} {errs[1]:+.2e} (ratio {ratio:.2f}), extrapolated {ext:+.1e}")
    assert abs(errs[1]) < 2.5e-3 and 3.6 < ratio < 4.4
    assert abs(ext) < 2e-5


@pytest.mark.parametrize("pol,loss_db_cm", [("te", 6.6022), ("tm", 242.23)])
def test_leaky_slab_with_pml(pol, loss_db_cm):
    t, box = 0.22e-6, 0.5e-6
    guess = newton(lambda n: slab_det(n, [N_OX, N_SI], [box, t], N_OX, N_OX, pol), 2.8 if pol == "te" else 2.0)
    ex = newton(lambda n: slab_det(n, [N_OX, N_SI], [box, t], N_SI, N_OX, pol), guess)
    exact_db = 20 * np.log10(np.e) * K0 * ex.imag / 100
    assert abs(exact_db / loss_db_cm - 1) < 1e-3  # the reference itself
    step = 10e-9
    xs = CrossSection(background=N_OX)
    xs.add_layer(-5e-6, -box, N_SI, name="substrate")
    xs.add_layer(0.0, t, N_SI, name="core")
    xs.set_mesh([0.0, step], np.arange(-box - 1.0e-6, t + 1.0e-6 + 1e-12, step))
    ms = xs.solve(LAM, 4, ex.real, periodic=(True, False), pml={"cells": (0, 0, 30, 0), "strength": 4.0})
    m = min([m for m in ms if (m.te_fraction > 0.5) == (pol == "te")], key=lambda m: abs(m.neff - ex))
    print(f"{pol}: neff {m.neff:.6f} exact {ex:.6f}; loss {m.loss_db_per_cm:.4f} dB/cm, exact {exact_db:.4f}")
    assert abs(m.loss_db_per_cm / exact_db - 1) < 0.02
    assert abs(m.neff.real - ex.real) < 2.5e-3


def test_fibre_he11():
    from scipy.special import jv, jvp, kv, kvp

    n1, n2, a = 2.0, 1.444, 0.5e-6

    def f(ne):
        b = K0 * ne
        u, w = a * np.sqrt(K0**2 * n1**2 - b * b), a * np.sqrt(b * b - K0**2 * n2**2)
        J, K = jvp(1, u) / (u * jv(1, u)), kvp(1, w) / (w * kv(1, w))
        return (J + K) * (J + (n2 / n1) ** 2 * K) - (ne / n1) ** 2 * (1 / u**2 + 1 / w**2) ** 2

    ns = np.linspace(n2 + 1e-6, n1 - 1e-6, 4000)
    fs = np.array([f(x) for x in ns])
    roots = [brentq(f, ns[i], ns[i + 1]) for i in range(len(ns) - 1) if np.sign(fs[i]) != np.sign(fs[i + 1]) and abs(fs[i]) < 10 and abs(fs[i + 1]) < 10]
    ref = max(roots)
    th = np.linspace(0, 2 * np.pi, 257)[:-1]
    xs = CrossSection(background=n2)
    xs.add_polygon(np.c_[a * np.cos(th), a * np.sin(th)], n1)
    nodes = np.arange(-1.6e-6, 1.6e-6 + 1e-12, 25e-9)
    xs.set_mesh(nodes, nodes)
    ms = xs.solve(LAM, 2)
    print(f"HE11 {ms[0].neff.real:.6f} {ms[1].neff.real:.6f}, exact {ref:.6f}")
    assert abs(ms[0].neff.real - ref) < 2e-4 and abs(ms[1].neff.real - ref) < 2e-4
    assert abs(ms[0].neff.real - ms[1].neff.real) < 1e-8  # degenerate pair


def test_slab_bend_against_exact_bessel_solution():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    n1, n2, w, step = 1.60, 1.444, 1.0e-6, 20e-9

    def exact(R, guess, pol):
        k = K0 * 1e-6
        R1, R2, Rm = (R - w / 2) * 1e6, (R + w / 2) * 1e6, R * 1e6
        p1, p2 = (1, 1) if pol == "te" else (n1**2, n2**2)

        def f(neff):
            nu = k * neff * Rm
            J = lambda n, r: mp.besselj(nu, k * n * r)  # noqa: E731
            Y = lambda n, r: mp.bessely(nu, k * n * r)  # noqa: E731
            H = lambda n, r: mp.hankel1(nu, k * n * r)  # noqa: E731
            d = lambda fn, n, r: k * n * mp.diff(lambda x: fn(nu, x), k * n * r)  # noqa: E731
            M = mp.matrix(
                [
                    [J(n2, R1), -J(n1, R1), -Y(n1, R1), 0],
                    [d(mp.besselj, n2, R1) / p2, -d(mp.besselj, n1, R1) / p1, -d(mp.bessely, n1, R1) / p1, 0],
                    [0, J(n1, R2), Y(n1, R2), -H(n2, R2)],
                    [0, d(mp.besselj, n1, R2) / p1, d(mp.bessely, n1, R2) / p1, -d(mp.hankel1, n2, R2) / p2],
                ]
            )
            return mp.det(M)

        return complex(mp.findroot(f, mp.mpc(guess), tol=1e-20))

    xs = CrossSection(background=n2)
    xs.add_rect(-w / 2, -1, w / 2, 1, n1)
    xs.set_mesh(np.arange(-3e-6, 6e-6 + 1e-12, step), [0, step])
    for R in (20e-6, 30e-6):
        ms = xs.solve(LAM, 4, periodic=(False, True), pml={"cells": (0, 40, 0, 0)}, bend_radius=R)
        for pol, te in (("te", False), ("tm", True)):  # "te": E along the bend axis
            m = pick(ms, te)
            ex = exact(R, m.neff, pol)
            print(f"R {R * 1e6:.0f} um {pol}: {m.neff:.6f} exact {ex:.6f}  loss {m.loss_db_per_90deg():.4f} dB/90deg")
            assert abs(m.neff.imag / ex.imag - 1) < 0.04
            assert abs(m.neff.real - ex.real) < 5e-5


def test_slab_bend_asymptotic_formula():
    """Marcuse-type asymptotic bend loss (WKB tunnelling through the mapped
    cladding): within a factor 1.5 of the solver over 20-40 um."""
    n1, n2, w, step = 1.60, 1.444, 1.0e-6, 20e-9
    xs = CrossSection(background=n2)
    xs.add_rect(-w / 2, -1, w / 2, 1, n1)
    xs.set_mesh(np.arange(-3e-6, 6e-6 + 1e-12, step), [0, step])
    ns = pick(xs.solve(LAM, 2, periodic=(False, True)), False).neff.real
    d, beta = w / 2, K0 * ns
    kap, gam = K0 * np.sqrt(n1**2 - ns**2), K0 * np.sqrt(ns**2 - n2**2)
    for R in (20e-6, 30e-6, 40e-6):
        m = pick(xs.solve(LAM, 4, periodic=(False, True), pml={"cells": (0, 40, 0, 0)}, bend_radius=R), False)
        asym = gam**2 * kap**2 * np.exp(2 * gam * d) / (beta * (1 + gam * d) * (n1**2 - n2**2) * K0**2) * np.exp(-2 * gam**3 * R / (3 * K0**2 * n2**2))
        print(f"R {R * 1e6:.0f}: alpha {m.alpha:.4e} /m, asymptotic {asym:.4e} /m, ratio {asym / m.alpha:.3f}")
        assert 1 / 1.5 < asym / m.alpha < 1.5


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_coupled_slab_supermodes(pol):
    t, gap, step = 0.22e-6, 0.2e-6, 5e-9
    ns = np.linspace(N_OX + 1e-4, N_SI - 1e-4, 20000)
    g = lambda n: slab_det(n, [N_SI, N_OX, N_SI], [t, gap, t], N_OX, N_OX, pol)  # noqa: E731
    vals = np.array([g(x) for x in ns])
    use_re = np.abs(vals.real).max() > np.abs(vals.imag).max()
    h = (lambda x: g(x).real) if use_re else (lambda x: g(x).imag)
    comp = vals.real if use_re else vals.imag
    roots = sorted([brentq(h, ns[i], ns[i + 1], xtol=1e-14) for i in range(len(ns) - 1) if np.sign(comp[i]) != np.sign(comp[i + 1])], reverse=True)[:2]
    xs = slab_xs([(-gap / 2 - t, -gap / 2, N_SI), (gap / 2, gap / 2 + t, N_SI)], step, half=1.2e-6)
    ms = sorted([m for m in xs.solve(LAM, 6, periodic=(True, False)) if (m.te_fraction > 0.5) == (pol == "te")], key=lambda m: -m.neff.real)[:2]
    lc, lc_ex = LAM / (2 * (ms[0].neff.real - ms[1].neff.real)), LAM / (2 * (roots[0] - roots[1]))
    print(f"{pol}: even {ms[0].neff.real:.6f}/{roots[0]:.6f} odd {ms[1].neff.real:.6f}/{roots[1]:.6f} Lc {lc * 1e6:.4f}/{lc_ex * 1e6:.4f} um")
    assert abs(ms[0].neff.real - roots[0]) < 1e-3 and abs(ms[1].neff.real - roots[1]) < 1e-3
    assert abs(lc / lc_ex - 1) < 2e-3


@pytest.mark.parametrize("axis", [(np.cos(0.8), np.sin(0.8), 0), (np.cos(0.7), 0, np.sin(0.7)), (0, np.cos(0.5), np.sin(0.5)), (0.5, 0.6, 0.62)])
def test_full_anisotropy_against_berreman(axis):
    no, ne, tc, step = 2.21, 2.14, 0.6e-6, 5e-9
    f = uniaxial(no, ne, axis)
    eps = f(LAM)
    xs = slab_xs([(-tc / 2, tc / 2, f)], step)
    for m in xs.solve(LAM, 2, periodic=(True, False)):
        ex = newton(lambda n: berreman_det(n, eps, tc, N_OX), m.neff.real)
        print(f"axis {np.round(axis, 2)}: {m.neff.real:.6f} exact {ex.real:.6f} TE {m.te_fraction:.2f}")
        assert abs(m.neff.real - ex.real) < 5e-5


def test_group_index_energy_relation_matches_finite_difference():
    xs = CrossSection.strip(core_width=0.5e-6, core_thickness=0.22e-6, step=20e-9)
    m = xs.solve(LAM, 1, group_index=True)[0]
    d = 1e-9
    n_p, n_m = xs.solve(LAM + d, 1)[0].neff.real, xs.solve(LAM - d, 1)[0].neff.real
    ng_fd = m.neff.real - LAM * (n_p - n_m) / (2 * d)
    print(f"ng {m.n_group:.7f} finite difference {ng_fd:.7f}")
    assert abs(m.n_group - ng_fd) < 1e-5
    assert 4.0 < m.n_group < 4.4


def test_derived_quantities_and_overlaps():
    xs = CrossSection.strip(core_width=0.5e-6, core_thickness=0.22e-6, step=20e-9)
    te, tm = xs.solve(LAM, 2)
    assert abs(mode_overlap(te, te) - 1) < 1e-12
    assert abs(mode_overlap(te, te, conjugate=False) - 1) < 1e-12
    assert abs(mode_overlap(te, tm)) < 1e-7
    g_si, g_ox = te.confinement("si"), te.confinement("sio2")
    assert abs(g_si + g_ox - 1) < 1e-12 and 0.85 < g_si < 0.95
    assert 0.6 < te.confinement("si", kind="power") < g_si
    assert 0.15e-12 < te.effective_area() < 0.25e-12
    assert te.te_fraction > 0.95 and tm.te_fraction < 0.1 and abs(te.fraction_tm + te.fraction_te - 1) < 1e-15
    # a mode resampled on a shifted, coarser grid still overlaps well with itself
    xs2 = CrossSection.strip(core_width=0.5e-6, core_thickness=0.22e-6, step=30e-9)
    te2 = xs2.solve(LAM, 1)[0]
    assert abs(mode_overlap(te, te2)) ** 2 > 0.99


def test_rasterizer_matches_rust_subpixel_smoothing():
    import lightsim as ls
    from lightsim.xsection import from_gdsfactory

    pytest.importorskip("gdsfactory")
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    xs = from_gdsfactory("strip", step=20e-9, material_index={"si": N_SI, "sio2": N_OX})
    m_xs = xs.solve(LAM, 1)[0]
    um = 1e-6
    sim = ls.Simulation(min=[0, xs.u_nodes[0], xs.v_nodes[0]], max=[1 * um, xs.u_nodes[-1], xs.v_nodes[-1]], mesh=ls.ExplicitMesh([0, 0.5 * um, 1 * um], list(xs.u_nodes), list(xs.v_nodes)), run_time=1e-15, background="ox_c")
    sim.add_medium(ls.Medium("si_c", n=N_SI))
    sim.add_medium(ls.Medium("ox_c", n=N_OX))
    sim.add_structure(ls.Box([-1, -0.25 * um, 0], [2, 0.25 * um, 0.22 * um]), "si_c")
    m3 = ls.solve_modes(sim, 0, 0.5 * um, [0.5 * um, xs.u_nodes[0], xs.v_nodes[0]], [0.5 * um, xs.u_nodes[-1], xs.v_nodes[-1]], C0 / LAM, 1)[0]
    print(f"cross-section {m_xs.neff.real:.9f}, simulation plane {m3.neff.real:.9f}")
    assert abs(m_xs.neff - m3.neff) < 1e-6


def test_solve_modes_pml_and_bend_options_on_a_simulation_plane():
    import lightsim as ls

    um = 1e-6
    sim = ls.Simulation(min=[0, -2 * um, -1 * um], max=[1 * um, 4 * um, 1 * um], mesh=ls.UniformMesh(25e-9), run_time=1e-15, background="ox")
    sim.add_medium(ls.Medium("ox", n=1.444))
    sim.add_medium(ls.Medium("core", n=1.60))
    sim.add_structure(ls.Box([-1, -0.5 * um, -2], [2, 0.5 * um, 2]), "core")
    lo, hi = [0.5 * um, -2 * um, -1 * um], [0.5 * um, 4 * um, 1 * um]
    straight = ls.solve_modes(sim, 0, 0.5 * um, lo, hi, C0 / LAM, 2)
    bend = ls.solve_modes(sim, 0, 0.5 * um, lo, hi, C0 / LAM, 2, pml={"cells": (0, 30, 0, 0)}, bend_radius=20e-6, bend_center=0.0)
    m = pick(bend, False)  # E along z, the bend axis
    print(straight, bend)
    assert m.neff.real > pick(straight, False).neff.real  # the field moves out, n_eff (at R) rises
    assert 1.0 < m.loss_db_per_90deg() < 5.0  # 2.55 dB from the cross-section solver


def test_mode_class_backward_compatible():
    from lightsim.modes import Mode

    m = Mode(neff=2.0 + 0j, freq=C0 / LAM, te_fraction=1.0, profile={}, E=(None, None), H=(None, None), info={})
    assert m.E_n is None and m.n_group is None and m.bend_radius is None
    assert abs(m.wavelength - LAM) < 1e-18 and m.loss_db_per_cm == 0.0


def test_gplugins_style_functions():
    pytest.importorskip("pandas")
    import lightsim.gplugins_modes as gm

    modes = gm.find_modes_waveguide(core_width=0.45, resolution=32)
    assert 2.30 < modes[1].neff < 2.40 and 1.65 < modes[2].neff < 1.80
    assert modes[1].fraction_te > 0.9 and modes[2].fraction_te < 0.1
    df = gm.find_neff_vs_width(width1=0.4, width2=0.6, steps=3, nmodes=2, resolution=32)
    assert list(df.columns) == [1, 2, "width"] and np.all(np.diff(df[1]) > 0)
    df = gm.find_coupling_vs_gap(gap1=0.2, gap2=0.3, steps=2, nmodes=4, resolution=32)
    assert set(df.columns) == {"gap", "ne", "no", "lc", "dn"} and df.lc[1] > df.lc[0]
    disp = gm.find_mode_dispersion(core_width=0.45, resolution=32)
    assert 4.1 < disp.ng < 4.4
    wg = gm.Waveguide(wavelength=1.55, core_width=0.5, core_thickness=0.22, core_material="si", clad_material="sio2", num_modes=2, group_index_step=True)
    assert wg.n_eff.shape == (2,) and 4.0 < wg.n_group[0] < 4.4
    c = gm.WaveguideCoupler(wavelength=1.55, core_width=(0.45, 0.45), core_thickness=0.22, core_material="si", clad_material="sio2", gap=0.1)
    assert 5 < c.coupling_length()[0] < 20
    ov = gm.sweep_bend_mismatch(gm.Waveguide(wavelength=1.55, core_width=0.5, core_thickness=0.22, core_material="si", clad_material="sio2", num_modes=1, bend_radius=5), [5, 10])
    assert 0.98 < ov[0] < ov[1] <= 1.0
