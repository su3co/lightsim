"""2-D simulations from Python: the API (two-component vectors, invariant
axis, polarization), the slice through 3-D structures, the 1-D port modes
for any invariant axis, and the grating efficiencies against RCWA."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import TFSF, Box, Cylinder, DFTMonitor, FluxMonitor, ModeMonitor, ModeSource, PolySlab, Sphere, um

from rcwa_reference import CASE, efficiencies

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = ls.C0 / (1.55 * um)


def test_two_component_api():
    sim = ls.Simulation(min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=1e-14, polarization="te")
    assert sim.dimensions == 2 and sim.invariant_axis == 2
    assert sim.min == [-2 * um, -1 * um, -0.5] and sim.max == [2 * um, 1 * um, 0.5]
    assert sim.slab == (2, -0.5, 0.5)
    sim.add_medium(ls.Medium("si", n=3.48))
    sim.add_structure(Box([-2 * um, -0.2 * um], [2 * um, 0.2 * um]), "si")
    sim.add_structure(Cylinder([0, 0.5 * um], 0.1 * um), "si")
    sim.add_structure(PolySlab([[0, 0], [1 * um, 0], [0, 0.3 * um]]), "si")
    sim.add_source(ls.PointDipole([0, 0], "Ey", ls.GaussianPulse(F0, 0.2 * F0)))
    sim.add_monitor(FluxMonitor("f", 0, 1 * um, [1 * um, -1 * um], [1 * um, 1 * um], [F0]))
    d = sim.to_dict()
    assert (d["dimensions"], d["invariant_axis"], d["polarization"]) == (2, 2, "te")
    assert d["boundaries"]["z"] == [{"type": "periodic"}, {"type": "periodic"}]
    box, cyl, poly = (s["shape"] for s in d["structures"])
    assert box["min"] == [-2 * um, -0.2 * um, -0.5] and box["max"] == [2 * um, 0.2 * um, 0.5]
    assert cyl["axis"] == 2 and cyl["center"] == [0.0, 0.5 * um, 0.0] and cyl["length"] > 1.0
    assert poly["axis"] == 2 and poly["slab_min"] < -0.5 and poly["slab_max"] > 0.5
    assert d["sources"][0]["position"] == [0.0, 0.0, 0.0]
    assert d["monitors"][0]["min"] == [1 * um, -1 * um, -0.5]
    nx, ny, nz = (len(g) - 1 for g in sim.grid())
    assert nz == 1 and nx == 80 + 24 and ny == 40 + 24
    # the 2-D Courant limit
    dt, _ = sim.time_info()
    assert abs(dt / (0.99 * 50e-9 / (ls.C0 * np.sqrt(2))) - 1) < 1e-9

    # other invariant axes: the two components are the in-plane ones, in order
    sy = ls.Simulation(min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh([50e-9, 40e-9]), run_time=1e-14, invariant_axis="y", thickness=2 * um)
    assert sy.min == [-2 * um, -1 * um, -1 * um] and sy.max == [2 * um, 1 * um, 1 * um]
    assert sy.to_dict()["mesh"]["dx"][::2] == [50e-9, 40e-9]
    assert [len(g) - 1 for g in sy.grid()] == [80 + 24, 1, 50 + 24]
    sx = ls.Simulation(min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=1e-14, invariant_axis=0)
    assert sx.min[1:] == [-2 * um, -1 * um] and [len(g) - 1 for g in sx.grid()] == [1, 80 + 24, 40 + 24]

    # mistakes are reported
    with pytest.raises(ValueError, match="2-D"):
        ls.Simulation(min=[0, 0, 0], max=[1e-6, 1e-6, 1e-6], mesh=ls.UniformMesh(50e-9), run_time=1e-14, polarization="te")
    s3 = ls.Simulation(min=[0, 0, 0], max=[1e-6, 1e-6, 1e-6], mesh=ls.UniformMesh(50e-9), run_time=1e-14)
    with pytest.raises(ValueError, match="2 components"):
        s3.add_monitor(FluxMonitor("f", 0, 0.5e-6, [0, 0], [1e-6, 1e-6], [F0]))
    with pytest.raises(ValueError, match="length"):
        s3.add_structure(Cylinder([0, 0, 0], 0.1 * um), "vacuum")
    # a source of the other polarization is an error, not a silent no-op
    sim.sources = [ls.PointDipole([0, 0], "Ez", ls.GaussianPulse(F0, 0.2 * F0), name="wrong")]
    with pytest.raises(RuntimeError, match="'wrong' drives only the tm"):
        sim.run(device="cpu")


def test_structures_are_cut_by_the_slice_plane():
    # a sphere centred 0.15 um above the plane leaves a disc of radius 0.2 um
    def model(center_z):
        sim = ls.Simulation(min=[-1 * um, -1 * um, -1 * um], max=[1 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(25e-9), run_time=1e-14, dimensions=2, polarization="tm")
        sim.add_medium(ls.Medium("glass", n=2.0))
        sim.add_structure(Sphere([0, 0, center_z], 0.25 * um), "glass")
        sim.add_structure(Box([0.5 * um, -1, 0.2 * um], [0.8 * um, 1, 0.6 * um]), "glass")  # above the plane: absent
        from lightsim import _core

        info = _core.plane_info(sim.to_json(), 0, 0.0, [0, -1e-6, 0], [0, 1e-6, 0], None)
        return np.asarray(info["u_nodes"]), np.asarray(info["eps_v"])[:, 0]  # E_z along the line x = 0

    y, eps = model(0.15 * um)
    inside = np.abs(y[:-1]) < 0.18 * um
    outside = np.abs(y[:-1]) > 0.22 * um
    assert np.allclose(eps[inside], 4.0) and np.allclose(eps[outside], 1.0)
    y, eps = model(0.3 * um)  # the plane misses the sphere
    assert np.allclose(eps, 1.0)


@pytest.mark.parametrize("axis,pol", [("z", "te"), ("z", "tm"), ("y", "te"), ("y", "tm"), ("x", "tm")])
def test_slab_mode_transmission_any_invariant_axis(axis, pol):
    """A slab waveguide along the first in-plane axis: the port mode from
    solve_modes is launched and arrives with unit transmission; nothing is
    reflected; only three components are updated."""
    n_core, n_clad, w = 2.0, 1.0, 0.4 * um
    sim = ls.Simulation(min=[-3 * um, -1.2 * um], max=[3 * um, 1.2 * um], mesh=ls.UniformMesh(25e-9), run_time=150e-15, invariant_axis=axis, polarization=pol, shutoff=1e-6)
    sim.add_medium(ls.Medium("core", n=n_core))
    sim.add_structure(Box([-10 * um, -w / 2], [10 * um, w / 2]), "core")
    a = [i for i in range(3) if i != sim.invariant_axis][0]  # the propagation axis
    lo, hi = [0.0, -1.0 * um], [0.0, 1.0 * um]
    plane = lambda x: ([x, lo[1]], [x, hi[1]])  # noqa: E731
    mode = ls.solve_modes(sim, a, -2 * um, *plane(-2 * um), F0, num_modes=1)[0]
    assert mode.te_fraction == (1.0 if pol == "te" else 0.0)
    assert n_clad < mode.neff.real < n_core
    sim.add_source(ModeSource(a, -2 * um, "+", *plane(-2 * um), mode.profile, ls.GaussianPulse(F0, 0.1 * F0)))
    for name, x in [("in", -1.5 * um), ("out", 2 * um)]:
        sim.add_monitor(ModeMonitor(name, a, x, *plane(x), [F0], [mode.profile]))
    res = sim.run(device=DEVICE)
    assert f"2-D {pol}" in res.stats["backend"], res.stats["backend"]
    (ain, bin_), (aout, _) = res.mode_amplitudes("in"), res.mode_amplitudes("out")
    t, r = abs(aout[0] / ain[0]) ** 2, abs(bin_[0] / ain[0]) ** 2
    print(axis, pol, "neff", mode.neff.real, "T", t, "R", r)
    assert abs(t - 1) < 5e-3 and r < 1e-4


def test_invariant_axes_are_equivalent():
    """The same problem with the invariant axis along z, y and x gives the
    same modes and the same transmission."""
    out = []
    for axis in "zyx":
        sim = ls.Simulation(min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh(25e-9), run_time=80e-15, invariant_axis=axis, polarization="te", background="clad")
        sim.add_medium(ls.Medium("clad", n=1.45)).add_medium(ls.Medium("core", n=2.0))
        sim.add_structure(Box([-10 * um, -0.2 * um], [-0.3 * um, 0.2 * um]), "core")
        sim.add_structure(Box([0.3 * um, -0.15 * um], [10 * um, 0.15 * um]), "core")  # a gap and a narrower guide
        a = [i for i in range(3) if i != sim.invariant_axis][0]
        m_in = ls.solve_modes(sim, a, -1.5 * um, [-1.5 * um, -0.9 * um], [-1.5 * um, 0.9 * um], F0, num_modes=1)[0]
        m_out = ls.solve_modes(sim, a, 1.5 * um, [1.5 * um, -0.9 * um], [1.5 * um, 0.9 * um], F0, num_modes=1)[0]
        sim.add_source(ModeSource(a, -1.5 * um, "+", [-1.5 * um, -0.9 * um], [-1.5 * um, 0.9 * um], m_in.profile, ls.GaussianPulse(F0, 0.1 * F0)))
        sim.add_monitor(ModeMonitor("in", a, -1.2 * um, [-1.2 * um, -0.9 * um], [-1.2 * um, 0.9 * um], [F0], [m_in.profile]))
        sim.add_monitor(ModeMonitor("out", a, 1.5 * um, [1.5 * um, -0.9 * um], [1.5 * um, 0.9 * um], [F0], [m_out.profile]))
        res = sim.run(device=DEVICE)
        s21 = res.mode_amplitudes("out")[0][0] / res.mode_amplitudes("in")[0][0]
        out.append((m_in.neff.real, m_out.neff.real, s21))
    for o in out[1:]:
        assert abs(o[0] - out[0][0]) < 1e-9 and abs(o[1] - out[0][1]) < 1e-9
        assert abs(o[2] - out[0][2]) < 2e-5 * abs(out[0][2]), (o[2], out[0][2])
    assert 0.2 < abs(out[0][2]) ** 2 < 0.95


def test_two_d_kernels_match_three_d_kernels_and_thickness_scales_flux():
    def run(thickness, force):
        sim = ls.Simulation(min=[-1.2 * um, -1.2 * um], max=[1.2 * um, 1.2 * um], mesh=ls.UniformMesh(30e-9), run_time=40e-15, thickness=thickness, force_3d_kernels=force)
        sim.add_medium(ls.Medium("rod", n=1.5))
        sim.add_structure(Cylinder([0, 0], 0.3 * um), "rod")
        sim.add_source(TFSF([-0.6 * um, -0.6 * um], [0.6 * um, 0.6 * um], 0, "+", "Ez", ls.GaussianPulse(F0, 0.3 * F0)))
        sim.add_monitor(FluxMonitor("f", 0, 0.9 * um, [0.9 * um, -0.9 * um], [0.9 * um, 0.9 * um], [F0]))
        sim.add_monitor(DFTMonitor("d", [-0.9 * um, -0.9 * um], [0.9 * um, 0.9 * um], [F0], components=("Ez", "Hx", "Hy")))
        return sim.run(device=DEVICE)

    a, b, c = run(1.0, False), run(1.0, True), run(2e-6, False)
    assert "2-D tm" in a.stats["backend"] and "2-D" not in b.stats["backend"]
    for comp in ("Ez", "Hx", "Hy"):
        fa, fb, fc = a.field("d", comp), b.field("d", comp), c.field("d", comp)
        assert np.max(np.abs(fa - fb)) <= 1e-4 * np.max(np.abs(fa))
        assert np.max(np.abs(fa - fc)) <= 1e-6 * np.max(np.abs(fa))  # the fields do not depend on the thickness
    assert a.flux("f")[0] > 0
    assert abs(c.flux("f")[0] / (a.flux("f")[0] * 2e-6) - 1) < 1e-5  # the flux is that of the slab


@pytest.mark.parametrize("pol,along", [("te", False), ("tm", True)])
def test_grating_orders_match_rcwa(pol, along):
    c = CASE
    lam, period, depth, theta = c["wavelength"], c["period"], c["depth"], c["theta"]
    f0 = ls.C0 / lam
    k0 = 2 * np.pi / lam
    ky = k0 * np.sin(theta)
    dx = lam / 40

    def run(grating):
        sim = ls.Simulation(min=[0, -period / 2], max=[4.5 * um, period / 2], mesh=ls.UniformMesh(dx), run_time=400e-15, polarization=pol)
        sim.set_boundaries(y=ls.Bloch(ky))
        if grating:
            sim.add_medium(ls.Medium("glass", n=c["n2"]))
            sim.add_structure(Box([2.5 * um + depth, -1], [10 * um, 1]), "glass")
            sim.add_structure(Box([2.5 * um, -c["fill"] * period / 2], [2.5 * um + depth, c["fill"] * period / 2]), "glass")
        sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ez" if along else "Ey", ls.GaussianPulse(f0, 0.06 * f0), angle=theta))
        for name, x in [("T", 3.8 * um), ("R", 0.5 * um)]:
            sim.add_monitor(FluxMonitor(name, 0, x, [x, -1], [x, 1], [f0]))
        return sim.run(device=DEVICE)

    def order_power(res, name, m):
        f = res.raw["dft"][name]["fields"]
        cplx = lambda k: (np.asarray(f[k]["re"]) + 1j * np.asarray(f[k]["im"]))[0, 0, :, 0]  # noqa: E731
        e, h, sign = (cplx("Ez"), 0.5 * (cplx("Hy") + cplx("Hy_1")), -1.0) if along else (cplx("Ey"), 0.5 * (cplx("Hz") + cplx("Hz_1")), 1.0)
        ph = np.exp(-1j * (ky + 2 * np.pi * m / period) * dx * np.arange(len(e)))
        return sign * 0.5 * np.real(np.mean(e * ph) * np.conj(np.mean(h * ph)))

    vac, gr = run(False), run(True)
    p_inc = order_power(vac, "T", 0)
    orders, r_ref, t_ref = efficiencies(**c, e_along_grooves=along, harmonics=81)
    total = 0.0
    for m, rr, tt in zip(orders, r_ref, t_ref):
        if rr == 0 and tt == 0:
            continue
        r, t = -order_power(gr, "R", m) / p_inc, order_power(gr, "T", m) / p_inc
        print(f"{pol} order {m:+d}: R {r:.4f} (RCWA {rr:.4f})  T {t:.4f} (RCWA {tt:.4f})")
        assert abs(r - rr) < 5e-3 and abs(t - tt) < 8e-3
        total += r + t
    assert abs(total - 1) < 0.01


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_mode_power_gradient_in_2d(pol):
    """The two-run adjoint gradient in a 2-D simulation against a central
    finite difference, for both polarizations (the in-plane components live
    on the faces of the single cell: their voxel is the whole slab)."""
    from lightsim.adjoint import mode_power_gradient

    block = ([-0.2 * um, 0.25 * um, -1.0], [0.2 * um, 0.45 * um, 1.0])
    line = lambda x: ([x, -0.9 * um], [x, 0.9 * um])  # noqa: E731

    def scene(delta):
        sim = ls.Simulation(min=[-1.5 * um, -1 * um], max=[1.5 * um, 1 * um], mesh=ls.UniformMesh(25e-9), run_time=80e-15, polarization=pol)
        sim.add_medium(ls.Medium("core", n=2.0))
        sim.add_medium(ls.Medium("patch", eps=2.5 + delta))
        sim.add_structure(Box([-10 * um, -0.2 * um], [10 * um, 0.2 * um]), "core")
        sim.add_structure(Box(block[0][:2], block[1][:2]), "patch")
        return sim

    def with_source(sim, mode):
        sim.add_source(ModeSource(0, -1.2 * um, "+", *line(-1.2 * um), mode.profile, ls.GaussianPulse(F0, 0.1 * F0), name="src"))
        return sim

    sim = scene(0.0)
    mode = ls.solve_modes(sim, 0, -1.2 * um, *line(-1.2 * um), F0, num_modes=1)[0]
    output = dict(axis=0, position=1.0 * um, min=[1.0 * um, -0.9 * um, -0.5], max=[1.0 * um, 0.9 * um, 0.5], profile=mode.profile, direction="+")
    result = mode_power_gradient(with_source(sim, mode), [-0.4 * um, 0.1 * um, -0.5], [0.4 * um, 0.6 * um, 0.5], output, F0, device=DEVICE)

    def amplitude(delta):
        s = with_source(scene(delta), mode)
        s.add_monitor(ModeMonitor("out", 0, 1.0 * um, *line(1.0 * um), [F0], [mode.profile]))
        r = s.run(device=DEVICE)
        return r.mode_amplitudes("out")[0][0] / r.source_spectrum("src")[1][0]

    da = result.amplitude_derivative_in_box(sim, *block)
    fd = (amplitude(0.1) - amplitude(-0.1)) / 0.2
    print(pol, "adjoint", da, "finite difference", fd, "ratio", abs(da) / abs(fd), "phase", np.angle(da / fd))
    assert abs(abs(da) / abs(fd) - 1) < 0.03
    assert abs(np.angle(da / fd)) < 0.15
