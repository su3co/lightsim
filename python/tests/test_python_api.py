"""Python-level tests: API, mode solver, gdsfactory import, S-parameters.

Run with `pytest python/tests` (uses the CPU backend unless LIGHTSIM_DEVICE is set).
"""

import math
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def slab_te_neff(n_core, n_clad, d, lam):
    """Fundamental TE mode of a symmetric slab (bisection on the dispersion relation)."""
    k0 = 2 * math.pi / lam

    def f(neff):
        kx = k0 * math.sqrt(max(n_core**2 - neff**2, 0))
        g = k0 * math.sqrt(max(neff**2 - n_clad**2, 0))
        return math.tan(kx * d / 2) - g / kx

    lo, hi = n_clad + 1e-9, n_core - 1e-9
    # restrict to the first branch kx d/2 < pi/2
    hi = min(hi, math.sqrt(n_core**2 - (math.pi / (k0 * d)) ** 2) - 1e-12) if (math.pi / (k0 * d)) < n_core else hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def strip_sim(dx=20e-9):
    sim = ls.Simulation(min=[0, -2 * um, -1.5 * um], max=[2 * um, 2 * um, 1.5 * um], mesh=ls.UniformMesh(dx), run_time=1e-15, background="sio2")
    sim.add_medium(ls.Medium("sio2", n=1.444))
    sim.add_medium(ls.Medium("si", n=3.476))
    return sim


def test_devices_and_version():
    assert "cpu" in ls.devices()
    assert ls.__version__


def test_release_gpus_then_run_again():
    """Letting the kept GPU contexts go and running again gives the same
    fields (on the CPU there is nothing to release)."""
    f0 = ls.C0 / (1.0 * um)
    sim = ls.Simulation(min=[0, 0, 0], max=[1 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=10e-15)
    sim.add_source(ls.PointDipole([0.5 * um] * 3, "Ez", ls.GaussianPulse(f0, 0.3 * f0)))
    sim.add_monitor(ls.DFTMonitor("plane", [0, 0, 0.5 * um], [1 * um, 1 * um, 0.5 * um], [f0], components=("Ez",)))
    a = np.asarray(sim.run(device=DEVICE).field("plane", "Ez"))
    n = ls.release_gpus()
    assert n >= 1 if DEVICE.startswith("cuda") else n >= 0
    b = np.asarray(sim.run(device=DEVICE).field("plane", "Ez"))
    assert np.array_equal(a, b)


def test_mode_solver_strip_waveguide():
    sim = strip_sim()
    sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]), "si")
    modes = ls.solve_modes(sim, 0, 1 * um, [1 * um, -1.5 * um, -1.2 * um], [1 * um, 1.5 * um, 1.2 * um], ls.C0 / (1.55 * um), num_modes=2)
    te = max(modes, key=lambda m: m.te_fraction)
    tm = min(modes, key=lambda m: m.te_fraction)
    # 500 x 220 nm Si strip at 1550 nm: TE0 ~ 2.44-2.45, TM0 ~ 1.77
    assert 2.43 < te.neff.real < 2.46
    assert 1.75 < tm.neff.real < 1.79
    assert te.te_fraction > 0.95


def test_mode_solver_slab_vs_analytic():
    # wide slab (5 um) approximates the 2-D slab; E along y (TE)
    dx = 10e-9
    sim = ls.Simulation(min=[0, -4 * um, -1.2 * um], max=[1 * um, 4 * um, 1.2 * um], mesh=ls.UniformMesh(dx), run_time=1e-15, background="clad")
    sim.add_medium(ls.Medium("clad", n=1.45))
    sim.add_medium(ls.Medium("core", n=2.0))
    sim.add_structure(ls.Box([-1, -2.5 * um, -0.15 * um], [2, 2.5 * um, 0.15 * um]), "core")
    modes = ls.solve_modes(sim, 0, 0.5 * um, [0.5 * um, -3.5 * um, -1.1 * um], [0.5 * um, 3.5 * um, 1.1 * um], ls.C0 / (1.55 * um), num_modes=1)
    slab = slab_te_neff(2.0, 1.45, 0.3 * um, 1.55 * um)
    # the 5 um finite width adds a lateral cos(pi y / W) variation:
    # neff^2 ~ neff_slab^2 - (pi / (k0 W))^2
    k0 = 2 * math.pi / (1.55 * um)
    expected = math.sqrt(slab**2 - (math.pi / (k0 * 5 * um)) ** 2)
    assert abs(modes[0].neff.real - expected) < 2e-3, (modes[0].neff, expected, slab)


def test_plane_wave_flux_through_api():
    lam = 1 * um
    f0 = ls.C0 / lam
    dx = lam / 30
    w = 2 * dx
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, w, w], mesh=ls.UniformMesh(dx), run_time=60e-15)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sig = ls.GaussianPulse(f0, 0.3 * f0)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", sig))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3 * um, [3 * um, -1, -1], [3 * um, 1, 1], [f0]))
    res = sim.run(device=DEVICE)
    freqs, spec = res.source_spectrum("pw") if "pw" in res.raw["source_spectra"] else res.source_spectrum("source0")
    # analytic incident flux 0.5/eta0 |S|^2 A with S over E-time samples ~ source spectrum magnitude
    eta0 = 376.730313412
    inc = 0.5 / eta0 * abs(spec[0]) ** 2 * w * w
    assert abs(res.flux("T")[0] / inc - 1) < 0.02


@pytest.mark.skipif(pytest.importorskip("gdsfactory", reason="gdsfactory not installed") is None, reason="no gdsfactory")
def test_gdsfactory_import_and_straight_sparams():
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    from lightsim.gds import component_to_structures
    from lightsim.sparams import write_sparameters

    c = gf.components.straight(length=2)
    structs, media, layers = component_to_structures(c)
    assert any(s["shape"]["type"] == "poly_slab" for s in structs)
    core = [l for l in layers if l.name == "core"][0]
    assert abs((core.zmax - core.zmin) - 0.22 * um) < 1e-12
    sp = write_sparameters(c, wavelengths=np.linspace(1.5e-6, 1.6e-6, 3), steps_per_wavelength=12, device=DEVICE, input_ports=["o1"])
    s21 = np.abs(sp["o2@0,o1@0"]) ** 2
    s11 = np.abs(sp["o1@0,o1@0"]) ** 2
    assert np.all(np.abs(s21 - 1) < 0.02), s21
    assert np.all(s11 < 0.01), s11


def test_mode_source_is_one_sided_in_both_directions():
    """A mode launched towards +x or -x carries the power of the signal
    spectrum and sends nothing the other way (the half-cell offset between
    the E and H planes of the source must be counted once)."""
    from lightsim import eim

    lam = 1.55 * um
    f0 = ls.C0 / lam
    dx = 40e-9
    base = ls.Simulation(min=[0, -1.5 * um, 0], max=[6 * um, 1.5 * um, dx], mesh=ls.UniformMesh(dx), run_time=300e-15, background="clad", shutoff=1e-7)
    base.set_boundaries(z=ls.PERIODIC)
    base.add_medium(ls.Medium("clad", n=1.444))
    base.add_structure(ls.Box([-1, -0.25 * um, -1], [9 * um, 0.25 * um, 1]), ls.Medium("core", n=2.85))
    xs = 3.0 * um
    wmin, wmax = [xs, -1.2 * um, 0], [xs, 1.2 * um, dx]
    for pol in ("te", "tm"):
        mode = eim.solve_port_mode(base, 0, xs, wmin, wmax, f0, pol)
        for direction, ahead, behind in (("+", 4.8 * um, 1.2 * um), ("-", 1.2 * um, 4.8 * um)):
            sim = ls.Simulation(**{k: (list(v) if isinstance(v, list) else v) for k, v in base.__dict__.items()})
            sim.add_source(ls.ModeSource(0, xs, direction, wmin, wmax, mode.profile, ls.GaussianPulse(f0, 0.1 * f0), name="src"))
            for name, x in (("ahead", ahead), ("behind", behind)):
                sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1.2 * um, 0], [x, 1.2 * um, dx], [f0]))
            sim.add_monitor(ls.ModeMonitor("mode", 0, ahead, [ahead, -1.2 * um, 0], [ahead, 1.2 * um, dx], [f0], [mode.profile]))
            r = sim.run(device=DEVICE)
            s2 = abs(r.source_spectrum("src")[1][0]) ** 2
            forward = abs(r.flux("ahead")[0]) / s2
            backward = abs(r.flux("behind")[0]) / s2
            a_plus, a_minus = r.mode_amplitudes("mode")
            a = abs((a_plus if direction == "+" else a_minus)[0]) ** 2 / s2
            print(f"{pol} {direction}: forward {forward:.4f} backward {backward:.2e} mode power {a:.4f}")
            assert abs(forward - 1) < 0.03 and abs(a - 1) < 0.03
            assert backward < 1e-4


def _two_strips(y_centre, mesh, ymin=-2.0 * um, ymax=2.0 * um):
    """Two parallel strips placed symmetrically about ``y_centre``."""
    sim = ls.Simulation(min=[0, ymin, 0], max=[3 * um, ymax, 0.3 * um], mesh=mesh, run_time=1e-14, background="sio2")
    for s in (-1, 1):
        y0 = y_centre + s * 0.37 * um
        sim.add_structure(ls.Box([-1, y0 - 0.25 * um, 0.04 * um], [1, y0 + 0.25 * um, 0.26 * um]), "si")
    return sim


def test_auto_mesh_is_mirror_symmetric_for_a_symmetric_layout():
    """The two arms of a symmetric coupler must get the same cells, or their
    modes differ and the bar/cross phase drifts."""
    lam = 1.55 * um
    y = _two_strips(0.0, ls.AutoMesh(lam, steps_per_wavelength=14)).grid()[1]
    steps = np.diff(y)
    assert np.allclose(y, -y[::-1], rtol=0, atol=1e-18) and np.array_equal(steps, steps[::-1])
    assert np.any(np.abs(y) < 1e-18)  # a node on the symmetry plane
    assert steps.min() < 0.5 * steps.max()  # it is a graded mesh

    # symmetry plane away from the centre of the region: not symmetric by itself ...
    y0 = 0.33 * um

    def mirror_error(nodes):  # distance from the mirror image of each node near the strips to the nearest node
        near = nodes[np.abs(nodes - y0) < 0.7 * um]
        return max(np.min(np.abs(nodes - (2 * y0 - v))) for v in near)

    off = _two_strips(y0, ls.AutoMesh(lam, steps_per_wavelength=14)).grid()[1]
    assert mirror_error(off) > 2e-9
    # ... until the plane is named
    sym = _two_strips(y0, ls.AutoMesh(lam, steps_per_wavelength=14, symmetry=(None, y0, None))).grid()[1]
    assert mirror_error(sym) < 1e-15
    assert np.min(np.abs(sym - y0)) < 1e-18  # a node on the plane
    with pytest.raises(RuntimeError, match="outside the region"):
        _two_strips(0.0, ls.AutoMesh(lam, symmetry=(None, 5 * um, None))).grid()


def test_frozen_mesh_survives_removing_structures():
    """Port modes solved on a single-waveguide copy of a device must land on
    the device's grid: the automatic mesh follows the structures, a frozen
    mesh does not."""
    lam = 1.55 * um
    device = _two_strips(0.0, ls.AutoMesh(lam, steps_per_wavelength=12))
    grid = [np.asarray(g) for g in device.grid()]

    def copy_with_one_strip(mesh):
        sim = ls.Simulation(min=device.min, max=device.max, mesh=mesh, run_time=device.run_time, background="sio2")
        sim.structures = device.structures[:1]
        return sim

    moved = [np.asarray(g) for g in copy_with_one_strip(device.mesh).grid()]
    assert any(len(a) != len(b) or not np.allclose(a, b, rtol=0, atol=1e-12) for a, b in zip(grid, moved))

    mesh = device.frozen_mesh()
    assert mesh["type"] == "explicit" and mesh["y"][0] == device.min[1] and mesh["y"][-1] == device.max[1]
    assert len(mesh["x"]) == len(grid[0]) - 24  # without the 12 + 12 absorbing cells
    same = [np.asarray(g) for g in copy_with_one_strip(mesh).grid()]
    assert all(np.array_equal(a, b) for a, b in zip(grid, same))
    assert mesh == ls.ExplicitMesh(mesh["x"], mesh["y"], mesh["z"])

    # in place, and with boundaries that add no cells
    device.set_boundaries(y=ls.PERIODIC)
    before = [np.asarray(g) for g in device.grid()]
    assert device.freeze_mesh() is device and device.mesh["type"] == "explicit"
    device.structures = []
    assert all(np.array_equal(a, b) for a, b in zip(before, device.grid()))
    with pytest.raises(ValueError):
        ls.ExplicitMesh([0, 1], [0, 0], [0, 1])


def test_duplicate_monitor_names_are_an_error():
    """Two monitors with one name overwrote each other in the results, and a
    flux monitor sharing its name with a field monitor crashed the core."""
    f0 = ls.C0 / (1.0 * um)

    def base():
        sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 0.1 * um, 0.1 * um], mesh=ls.UniformMesh(50e-9), run_time=20e-15)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        sim.add_source(ls.PlaneWave(0, 0.5 * um, "+", "Ey", ls.GaussianPulse(f0, 0.2 * f0), name="pw"))
        return sim

    flux = ls.FluxMonitor("T", 0, 1.5 * um, [1.5 * um, -1, -1], [1.5 * um, 1, 1], [f0])
    field = ls.DFTMonitor("T", [1 * um, 0, 0], [1 * um, 0, 0], [f0], components=("Ey",))
    sim = base().add_monitor(flux)
    with pytest.raises(ValueError, match="monitor named 'T' exists"):
        sim.add_monitor(field)
    with pytest.raises(ValueError, match="source named 'pw' exists"):
        sim.add_source(ls.PlaneWave(0, 0.6 * um, "+", "Ey", ls.GaussianPulse(f0, 0.2 * f0), name="pw"))
    # a list filled by hand reaches the core, which refuses it with an ordinary exception
    sim = base()
    sim.monitors = [flux, field]
    for call in (sim.run, sim.grid, sim.time_info):
        with pytest.raises(RuntimeError, match="duplicate monitor name 'T'"):
            call() if call is not sim.run else call(device=DEVICE)
    # distinct names are fine, also for monitors of the same kind
    sim = base().add_monitor(flux).add_monitor(dict(field, name="E"))
    assert np.isfinite(sim.run(device=DEVICE).flux("T")[0])


def test_energy_monitor_returns_joules():
    """The energy of a plane-wave pulse of unit amplitude that is wholly
    inside the monitor box is eps0 A c int s(t)^2 dt; when the box cuts the
    cells, only the part inside counts."""
    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 30
    area = (2 * dx) ** 2
    sim = ls.Simulation(min=[0, 0, 0], max=[8 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=60e-15)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
    sim.add_monitor(ls.EnergyMonitor("all", [1.5 * um, -1, -1], [7.0 * um, 1, 1], interval=10))
    sim.add_monitor(ls.EnergyMonitor("half", [1.5 * um, -1, -1], [7.0 * um, 1, dx], interval=10))  # half the z extent
    res = sim.run(device=DEVICE)
    t, u = res.energy("all")
    tau = 1 / (2 * np.pi * 0.3 * f0)
    exact = 8.8541878128e-12 * area * ls.C0 * tau * np.sqrt(np.pi) / 2
    print("U / exact", u.max() / exact, "samples", len(t))
    assert t.shape == u.shape and np.all(np.diff(t) > 0) and abs(t[1] - t[0] - 10 * res.dt) < 1e-22
    assert abs(u.max() / exact - 1) < 0.01
    assert u[-1] < 1e-6 * u.max() and np.all(u >= 0)
    assert abs(res.energy("half")[1].max() / u.max() - 0.5) < 1e-3
    # the field samples the monitor is computed from are not kept
    assert not any("energy" in k for k in res.raw["time"])


def test_run_that_ends_before_the_fields_decayed_says_so():
    """shutoff compares the field energy with its peak. A run that reaches
    run_time above that level returns results, with a warning; the decay
    history is in the statistics."""
    import warnings

    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 20

    def cavity(run_time, shutoff):
        # a slab between two mirrors rings for a long time
        sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=run_time, shutoff=shutoff, shutoff_interval=50)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        sim.add_structure(ls.Box([1.5 * um, -1, -1], [2.5 * um, 1, 1]), ls.Medium("high", n=6.0))
        sim.add_source(ls.PlaneWave(0, 0.5 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
        sim.add_monitor(ls.FluxMonitor("T", 0, 3.5 * um, [3.5 * um, -1, -1], [3.5 * um, 1, 1], [f0]))
        return sim

    with pytest.warns(UserWarning, match="reached run_time before the fields had decayed.*above shutoff = 1.0e-0?6"):
        res = cavity(60e-15, 1e-6).run(device=DEVICE)
    assert not res.stats["stopped_early"] and res.decay() > 1e-6
    steps, energy = res.stats["energy_steps"], res.stats["energy"]
    assert len(steps) == len(energy) > 5 and steps[-1] == res.stats["steps"] and np.all(np.diff(steps) > 0)
    assert res.decay() == pytest.approx(energy[-1] / energy.max())
    # long enough: it stops on shutoff, silently, below the threshold
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = cavity(3000e-15, 1e-6).run(device=DEVICE)
        assert res.stats["stopped_early"] and res.decay() <= 1e-6
        cavity(60e-15, None).run(device=DEVICE)  # no shutoff asked for: nothing to report


def test_large_field_values_are_not_mistaken_for_an_instability():
    """A dipole of unit moment in small cells makes fields of 1e20 V/m. Their
    squares overflow single precision, which the GPU energy sum uses; the
    run is stable and must return its results."""
    lam = 0.6 * um
    f0 = ls.C0 / lam
    sim = ls.Simulation(min=[-0.1 * um] * 3, max=[0.1 * um] * 3, mesh=ls.UniformMesh(5e-9), run_time=30e-15)
    sim.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(f0, 0.3 * f0), amplitude=1.0))
    sim.add_monitor(ls.FluxBoxMonitor("power", [-0.06 * um] * 3, [0.06 * um] * 3, [f0]))
    sim.add_monitor(ls.ProbeMonitor("near", [0, 0, 0], components=("Ez",)))
    res = sim.run(device=DEVICE)
    peak = np.abs(res.time_series("near", "Ez")[1]).max()
    print(f"peak field {peak:.2e} V/m, energy sum {res.stats['energy'].max():.2e}, steps {res.stats['steps']}")
    assert peak > 2e19 and np.isfinite(peak)  # the square is above the largest single-precision number
    assert np.all(np.isfinite(res.stats["energy"])) and res.stats["energy"].max() > 3.5e38
    assert np.isfinite(res.flux("power")[0]) and res.flux("power")[0] > 0
    # the energy history is the same decay curve on every backend
    e = res.stats["energy"] / res.stats["energy"].max()
    assert e[-1] < 0.1 and e.max() == 1.0
