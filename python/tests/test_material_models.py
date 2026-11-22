"""Material models beyond the plain media: the Raman (delayed) chi3
response, gyrotropic (magneto-optic) poles, director fields and four-level
gain media.

The physics is validated by the Rust cases (``cargo test -p lightsim-core``,
``lightsim validate``); these tests check the Python interface: the JSON the
builders produce, a run through the API, and that the gradient paths refuse
the media they do not model.

Run on a GPU with LIGHTSIM_DEVICE=cuda:0.
"""

import json
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import autodiff, invdes
from lightsim.adjoint import check_gradient_media
from lightsim.spec import C0, Box, GaussianPulse, Medium, PointDipole, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = C0 / (1.0 * um)
W0 = 2 * np.pi * F0


def _column(medium, amplitude=1.0, run_time=40e-15):
    dx = 1.0 * um / 40
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=run_time)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if medium is not None:
        sim.add_structure(Box([1.5 * um, -1, -1], [2.5 * um, 1, 1]), medium)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", GaussianPulse(F0, 0.3 * F0), amplitude=amplitude))
    sim.add_monitor(ls.ProbeMonitor("p", [3.5 * um, 0, 0], components=["Ey"]))
    return sim


def _probe(sim):
    return np.asarray(sim.run(device=DEVICE).time_series("p", "Ey")[1]).ravel()


def test_raman_medium_json_and_run():
    r = ls.Raman(0.18, 0.1 * W0, 0.03 * W0)
    assert r == {"fraction": 0.18, "omega": 0.1 * W0, "delta": 0.03 * W0}
    m = Medium("silica_like", n=1.0, chi3=2e-3, raman=r)
    sim = _column(m)
    spec = json.loads(sim.to_json())
    assert spec["media"][0]["raman"] == r
    # no Raman key when the response is instantaneous
    assert "raman" not in Medium("kerr", chi3=1e-3)
    lin = _probe(_column(None))
    inst = _probe(_column(Medium("kerr", n=1.0, chi3=2e-3)))
    ram = _probe(sim)
    d_inst, d_ram = np.abs(inst - lin).max(), np.abs(ram - lin).max()
    print(f"generated field: instantaneous {d_inst:.3e}, with Raman {d_ram:.3e} (peak {np.abs(lin).max():.3e})")
    assert np.all(np.isfinite(ram)) and 0 < d_ram < d_inst


def test_raman_parameters_are_checked():
    with pytest.raises(Exception, match="Raman"):
        _column(Medium("bad", chi3=1e-3, raman=ls.Raman(1.5, W0, 0.1 * W0))).run(device="cpu")
    with pytest.raises(Exception, match="chi3"):
        _column(Medium("bad", raman=ls.Raman(0.2, W0, 0.1 * W0))).run(device="cpu")


def test_gradient_paths_refuse_raman_media():
    m = Medium("ram", n=1.5, chi3=1e-20, raman=ls.Raman(0.2, 0.1 * W0, 0.03 * W0))
    sim = _column(m)
    with pytest.raises(ValueError, match="no gradient engine supports it"):
        check_gradient_media(sim)
    # an unused medium does not matter
    sim2 = _column(None)
    sim2.add_medium(m)
    check_gradient_media(sim2)
    # reverse mode (Rust): the session builder refuses the scene
    dx = 50e-9
    sim = ls.Simulation(min=[0, 0, 0], max=[3.2 * um, 2.0 * um, dx], mesh=ls.UniformMesh(dx), run_time=30e-15, background="ox",
                        boundaries={"x": [ls.PML(), ls.PML()], "y": [ls.PML(), ls.PML()], "z": [ls.PERIODIC, ls.PERIODIC]})
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_structure(Box([-1, 0.8 * um, -1], [10 * um, 1.2 * um, 1]), m)
    sim.add_source(PointDipole([0.5 * um, 1.0 * um, 0.0], "Ez", GaussianPulse(F0, 0.25 * F0), amplitude=1e-15))
    grid = invdes.DesignGrid([1.2 * um, 0.6 * um, -1], [2.0 * um, 1.4 * um, 1], (7, 9), Medium("design", eps=2.6**2))
    obj = autodiff.flux_energy(0, 2.7 * um, [2.7 * um, 0.4 * um, 0.0], [2.7 * um, 1.6 * um, dx])
    with pytest.raises(Exception, match="no gradient engine models"):
        autodiff.value_and_grad(sim, grid, np.full(grid.dims, 4.0), obj, device=DEVICE)


def _gyro_medium(bz=0.2):
    return Medium("gyro", eps=2.25, gyro=[(ls.Lorentz(1.0, 1.5 * W0, 1e13), (0.0, 0.0, bz * W0))])


def test_gyrotropic_permittivity_and_json():
    from lightsim.materials import gyrotropic_permittivity

    m = _gyro_medium()
    assert m["gyro"] == [{"pole": ls.Lorentz(1.0, 1.5 * W0, 1e13), "bias": [0.0, 0.0, 0.2 * W0]}]
    assert "gyro" not in Medium("plain", eps=2.0)
    e = gyrotropic_permittivity(m, W0)
    # eps_xy = -eps_yx = i g, eps_xx = eps_yy, no coupling to z
    assert abs(e[0, 1] + e[1, 0]) < 1e-12 * abs(e[0, 1]) and abs(e[0, 0] - e[1, 1]) < 1e-12
    assert abs(e[0, 2]) == 0 and abs(e[2, 1]) == 0 and abs(e[0, 1].real) < 2e-2 * abs(e[0, 1])  # (loss makes g slightly complex)
    # circular polarizations: eps_inf + s / (D +- w b)
    s, d = 2.25 * W0**2, (1.5 * W0) ** 2 - W0**2 - 1j * 1e13 * W0
    for sign in (1, -1):
        v = np.array([1, sign * 1j, 0]) / np.sqrt(2)
        assert abs(np.conj(v) @ e @ v - (2.25 + s / (d + sign * W0 * 0.2 * W0))) < 1e-9
    # the discrete tensor tends to the continuous one
    ed = gyrotropic_permittivity(m, W0, dt=1e-19)
    assert np.abs(ed - e).max() < 1e-3 * abs(e[0, 0])


def test_gyrotropic_rotation_through_the_api():
    """Faraday rotation between two depths of a half-space against the
    permittivity helper (with the run's time step)."""
    from lightsim.materials import gyrotropic_permittivity

    dx = 1.0 * um / 88
    sim = ls.Simulation(min=[0, 0, 0], max=[2 * dx, 2 * dx, 6 * um], mesh=ls.UniformMesh(dx), run_time=110e-15)
    sim.set_boundaries(x=ls.PERIODIC, y=ls.PERIODIC)
    m = _gyro_medium()
    sim.add_structure(Box([-1, -1, 2.5 * um], [1, 1, 20 * um]), m)
    sim.add_source(ls.PlaneWave(2, 1 * um, "+", "Ex", GaussianPulse(F0, 0.15 * F0)))
    for name, z in (("a", 3.2 * um), ("b", 4.0 * um)):
        sim.add_monitor(ls.DFTMonitor(name, [0, 0, z], [0, 0, z], [F0], components=["Ex", "Ey"]))
    res = sim.run(device=DEVICE)

    def angle(name):
        ex, ey = (complex(np.asarray(res.field(name, c)).ravel()[0]) for c in ("Ex", "Ey"))
        return 0.5 * np.arctan2(2 * (ex * np.conj(ey)).real, abs(ex) ** 2 - abs(ey) ** 2)

    dt = res.raw["dt"]
    e = gyrotropic_permittivity(m, W0, dt=dt)
    n_p, n_m = np.sqrt(e[0, 0] + 1j * e[0, 1]), np.sqrt(e[0, 0] - 1j * e[0, 1])
    want = 0.5 * W0 / C0 * (n_m - n_p).real * 0.8 * um
    got = angle("b") - angle("a")
    print(f"rotation over 0.8 um: {got:.5f} rad, expected {want:.5f} rad")
    assert abs(got - want) < 1e-2 * abs(want)


def test_gyrotropic_media_refusals():
    m = _gyro_medium()
    sim = _column(m)
    with pytest.raises(ValueError, match="no gradient engine supports it"):
        check_gradient_media(sim)
    # a 2-D simulation must keep both field sets: the bias couples them
    dx = 50e-9
    sim2 = ls.Simulation(min=[0, 0, 0], max=[2 * um, 2 * um, dx], mesh=ls.UniformMesh(dx), run_time=10e-15, dimensions=2, polarization="te")
    sim2.add_structure(Box([0.5 * um, -1, -1], [1.5 * um, 3 * um, 1]), m)
    sim2.add_source(PointDipole([0.2 * um, 1 * um, 0], "Ex", GaussianPulse(F0, 0.3 * F0)))
    with pytest.raises(Exception, match="both"):
        sim2.run(device="cpu")
    # a lossy eps_inf is refused
    bad = Medium("bad", eps=2.25, sigma=10.0, gyro=[(ls.Drude(W0, 1e13), (0, 0, 1e14))])
    with pytest.raises(Exception, match="gyrotropic"):
        _column(bad).run(device="cpu")


def test_director_field_builder_and_uniform_equivalence():
    """A uniform director equals the rotated tensor medium exactly; the
    builder orders the grid axes like IndexGrid."""
    d = ls.DirectorField([0, 0, 0], [1, 1, 1], [1.0, 1.0, 0.0], n_o=1.5, n_e=1.8)
    assert d["dims"] == [1, 1, 1] and d["eps_o"] == 2.25 and abs(d["eps_e"] - 3.24) < 1e-12
    tw = np.stack([np.cos(np.linspace(0, 1, 5)), np.sin(np.linspace(0, 1, 5)), np.zeros(5)], axis=-1)
    dz = ls.DirectorField([0, 0, 0], [0, 0, 1e-6], tw, eps_o=2.25, eps_e=2.56, axes=(2,))
    assert dz["dims"] == [1, 1, 5] and len(dz["director"]) == 15
    m = Medium("lc", director=d)
    assert all(abs(v - 3.24) < 1e-12 for v in m["eps"])
    c = np.cos(np.pi / 4)
    rot = Medium("rot", n=(1.8, 1.5, 1.5), rotation=[[c, -c, 0], [c, c, 0], [0, 0, 1]])
    res = []
    for med in (rot, Medium("rot", director=d)):
        dx = 1.0 * um / 72
        sim = ls.Simulation(min=[0, 0, 0], max=[2 * dx, 2 * dx, 5 * um], mesh=ls.UniformMesh(dx), run_time=60e-15)
        sim.set_boundaries(x=ls.PERIODIC, y=ls.PERIODIC)
        sim.add_structure(Box([-1, -1, 2 * um], [1, 1, 20 * um]), med)
        sim.add_source(ls.PlaneWave(2, 1 * um, "+", "Ex", GaussianPulse(F0, 0.2 * F0)))
        sim.add_monitor(ls.DFTMonitor("m", [0, 0, 3 * um], [0, 0, 3 * um], [F0], components=["Ex", "Ey"]))
        r = sim.run(device=DEVICE)
        res.append(np.array([np.asarray(r.field("m", cc)).ravel()[0] for cc in ("Ex", "Ey")]))
    print("uniform director vs rotated tensor:", np.abs(res[0] - res[1]).max(), "of", np.abs(res[0]).max())
    assert np.abs(res[0] - res[1]).max() <= 1e-6 * np.abs(res[0]).max()
    with pytest.raises(ValueError, match="no gradient engine supports it"):
        sim2 = _column(Medium("lc", director=d))
        check_gradient_media(sim2)


def test_gain_medium_amplifies_and_conserves_atoms():
    """An inverted slab amplifies a weak pulse at the transition, against the
    slab formula with the frozen populations' susceptibility (discrete, as
    ``materials.pole_permittivity`` gives it), and the population totals
    come back through the results."""
    from lightsim.materials import pole_permittivity

    eps0 = 8.8541878128e-12
    gamma, density = 0.1 * W0, 1e24
    xi = 0.05 * gamma * W0 * eps0 / (0.5 * density)
    g = ls.FourLevel(density, W0, gamma, 1.0, 1.0, 1.0, coupling=xi, initial=(0.3, 0.1, 0.6, 0.0))
    assert g["initial"] == [0.3, 0.1, 0.6, 0.0] and g["coupling"] == xi
    dx, d = 1.0 * um / 60, 2 * um

    def run(medium):
        sim = ls.Simulation(min=[0, 0, 0], max=[5 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=200e-15)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        if medium is not None:
            sim.add_structure(Box([2 * um, -1, -1], [2 * um + d, 1, 1]), medium)
        sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", GaussianPulse(F0, 0.3 * F0)))
        sim.add_monitor(ls.FluxMonitor("T", 0, 4.6 * um, [4.6 * um, -1, -1], [4.6 * um, 1, 1], [F0]))
        return sim.run(device=DEVICE)

    ra, rb = run(Medium("gain", eps=2.25, gain=g)), run(None)
    T = np.asarray(ra.flux("T"))[0] / np.asarray(rb.flux("T"))[0]
    # frozen populations: a Lorentz pole of strength xi (N1 - N2) / eps0
    lor = Medium("lor", eps=2.25, poles=[ls.Lorentz(xi * (0.1 - 0.6) * density / (eps0 * W0**2), W0, gamma)])
    n = np.sqrt(pole_permittivity(lor, W0, dt=ra.raw["dt"]))
    r = (1 - n) / (1 + n)
    ph = np.exp(1j * n * W0 / C0 * d)
    want = abs((1 - r**2) * ph / (1 - r**2 * ph**2)) ** 2
    print(f"slab gain at f0: T = {T:.5f}, expected {want:.5f}")
    assert T > 1.4 and abs(T - want) < 5e-3 * want
    lv = ra.gain_levels()
    assert lv.shape == (4,) and abs(lv.sum() / (density * d * (2 * dx) ** 2 * 3) - 1) < 1e-6
    assert rb.gain_levels() is None
    with pytest.raises(ValueError, match="no gradient engine supports it"):
        check_gradient_media(_column(Medium("gain", eps=2.25, gain=g)))
    # the explicit rate equations need lifetimes well above the time step
    with pytest.raises(Exception, match="rate equations"):
        _column(Medium("fast", gain=ls.FourLevel(density, W0, gamma, 1e-18, 1.0, 1.0))).run(device="cpu")
