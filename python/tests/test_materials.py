"""Material library, pole fitting and an FDTD check of a fitted medium.

Run with `pytest python/tests/test_materials.py` (CPU backend unless
LIGHTSIM_DEVICE is set).
"""

import math
import os
import warnings

import numpy as np
import pytest

import lightsim as ls
from lightsim import materials as mat
from lightsim.spec import C0, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
BAND = (1.2 * um, 1.7 * um)

# n at the given wavelength (um). Values are the published formulas evaluated
# independently (refractiveindex.info shows the same numbers to the digits
# quoted); Si rows are Li's recommended table at 293 K.
REFERENCE = [
    ("SiO2", 1.55, 1.44402, 2e-5),
    ("SiO2", 0.5876, 1.45846, 2e-5),  # fused silica n_d
    ("Si3N4", 1.55, 1.99628, 2e-5),
    ("Si", 1.20, 3.5167, 1e-4),
    ("Si", 1.30, 3.5016, 1e-4),
    ("Si", 1.55, 3.4757, 1e-4),
    ("Si", 1.70, 3.4653, 1e-4),
    ("LiNbO3_o", 1.55, 2.21111, 2e-5),
    ("LiNbO3_e", 1.55, 2.13756, 2e-5),
    ("Al2O3_o", 1.55, 1.74618, 2e-5),
    ("Al2O3_e", 1.55, 1.73838, 2e-5),
    ("GaAs", 1.55, 3.37017, 2e-5),
    ("Ge", 4.0, 4.02472, 2e-5),
]


@pytest.mark.parametrize("name,lam_um,n_ref,tol", REFERENCE)
def test_library_reference_values(name, lam_um, n_ref, tol):
    n = mat.refractive_index(name, lam_um * um, strict=True)
    assert isinstance(n, complex)
    assert n.imag == 0.0
    assert abs(n.real - n_ref) < tol, (name, lam_um, n.real, n_ref)


def test_library_independent_sellmeier():
    # the formulas written out longhand, independent of the module's helpers
    lam = 1.55
    sio2 = math.sqrt(1 + 0.6961663 * lam**2 / (lam**2 - 0.0684043**2) + 0.4079426 * lam**2 / (lam**2 - 0.1162414**2) + 0.8974794 * lam**2 / (lam**2 - 9.896161**2))
    sin = math.sqrt(1 + 3.0249 * lam**2 / (lam**2 - 0.1353406**2) + 40314 * lam**2 / (lam**2 - 1239.842**2))
    assert abs(mat.refractive_index("SiO2", lam * um).real - sio2) < 1e-12
    assert abs(mat.refractive_index("Si3N4", lam * um).real - sin) < 1e-12
    assert abs(sio2 - 1.4440) < 5e-5
    assert abs(sin - 1.996) < 5e-4


def test_library_api():
    names = mat.available_materials()
    assert {"SiO2", "Si3N4", "Si", "LiNbO3_o", "LiNbO3_e", "Al2O3_o", "Al2O3_e", "GaAs", "Ge"} <= set(names)
    # aliases, case-insensitivity, arrays
    assert mat.refractive_index("silica", 1.55 * um) == mat.refractive_index("SiO2", 1.55 * um)
    arr = mat.refractive_index("si3n4", np.array([1.3, 1.55]) * um)
    assert arr.shape == (2,) and arr.dtype == complex
    assert arr[0].real > arr[1].real  # normal dispersion
    eps = mat.permittivity("Si", 1.55 * um)
    assert abs(eps - mat.refractive_index("Si", 1.55 * um) ** 2) < 1e-12
    info = mat.material_info("Ge")
    assert info["range"] == pytest.approx((2e-6, 14e-6))
    # birefringence signs: LiNbO3 and sapphire are negative uniaxial
    assert mat.refractive_index("LiNbO3_e", 1.55 * um).real < mat.refractive_index("LiNbO3_o", 1.55 * um).real
    assert mat.refractive_index("Al2O3_e", 1.55 * um).real < mat.refractive_index("Al2O3_o", 1.55 * um).real
    with pytest.raises(KeyError):
        mat.refractive_index("unobtainium", 1.55 * um)
    with pytest.raises(ValueError):
        mat.refractive_index("Ge", 1.55 * um, strict=True)
    with pytest.warns(UserWarning):
        mat.refractive_index("Ge", 1.55 * um)


def _assert_passive(fit):
    assert fit["eps_inf"] >= 1.0
    for p in fit["lorentz"]:
        assert p["delta_eps"] >= 0 and p["gamma"] >= 0 and p["omega0"] > 0
    if fit["drude"]:
        assert fit["drude"]["omega_p"] >= 0 and fit["drude"]["gamma"] >= 0
    w = np.logspace(11, 19, 4001)
    eps = mat.pole_permittivity(fit["medium"], w)
    assert np.all(np.isfinite(eps))
    assert np.all(eps.imag >= 0.0)


@pytest.mark.parametrize("name", ["SiO2", "Si3N4", "Si", "LiNbO3_o", "LiNbO3_e", "Al2O3_o", "GaAs"])
def test_fit_telecom_band(name):
    fit = mat.fitted_medium(name, band=BAND, n_lorentz=2, return_fit=True)
    assert fit["max_n_error"] < 1e-3, (name, fit["max_n_error"])
    # the reported error is honest: re-evaluate on a denser grid than the fit used
    lam = np.linspace(BAND[0], BAND[1], 733)
    n_fit = np.sqrt(mat.pole_permittivity(fit["medium"], 2 * math.pi * C0 / lam))
    err = np.max(np.abs(n_fit - mat.refractive_index(name, lam)))
    assert err < 1e-3, (name, err)
    assert err < 2 * fit["max_n_error"] + 1e-9
    _assert_passive(fit)
    # lossless data -> exactly lossless poles
    assert all(p["gamma"] == 0.0 for p in fit["lorentz"])
    # the medium is a plain spec Medium usable by Simulation
    med = fit["medium"]
    assert med["name"] == name and med["eps"] == [fit["eps_inf"]] * 3
    assert all(p["type"] == "lorentz" for p in med["poles"])
    # resonances respect the default ADE-friendly bound (6x the top band frequency)
    assert all(p["omega0"] <= 6 * 2 * math.pi * C0 / BAND[0] * (1 + 1e-9) for p in fit["lorentz"])


def test_fit_midir_ge():
    fit = mat.fitted_medium("Ge", band=(3 * um, 5 * um), n_lorentz=2, return_fit=True)
    assert fit["max_n_error"] < 1e-3
    _assert_passive(fit)


def test_fitted_medium_returns_medium_and_warns_on_bad_fit():
    med = mat.fitted_medium("Si3N4", band=BAND, n_lorentz=2, medium_name="nitride")
    assert med["name"] == "nitride" and len(med["poles"]) >= 1
    with pytest.warns(UserWarning, match="pole fit error"):
        mat.fitted_medium("Si3N4", band=(0.4 * um, 5 * um), n_lorentz=1, eps_inf=1.0, tol=1e-9)


def test_fit_lossy_drude_lorentz_recovers_model():
    # synthetic absorbing data from a known Drude + Lorentz model
    true = ls.Medium("true", eps=2.0, poles=[ls.Lorentz(1.5, 2 * math.pi * C0 / (0.6 * um), 2e14), ls.Drude(1.2e16, 1e14)])
    lam = np.linspace(0.8 * um, 1.6 * um, 60)
    nk = np.sqrt(mat.pole_permittivity(true, 2 * math.pi * C0 / lam))
    assert np.all(nk.imag > 0)
    fit = mat.fit_poles(lam, nk.real, nk.imag, n_lorentz=1, drude=True)
    assert fit["max_n_error"] < 1e-6 and fit["max_k_error"] < 1e-6
    assert fit["eps_inf"] == pytest.approx(2.0, rel=1e-5)
    assert fit["drude"]["omega_p"] == pytest.approx(1.2e16, rel=1e-5)
    assert fit["lorentz"][0]["omega0"] == pytest.approx(2 * math.pi * C0 / (0.6 * um), rel=1e-5)
    _assert_passive(fit)
    # complex n is accepted too, and a fixed eps_inf is honoured
    fit2 = mat.fit_poles(lam, nk, n_lorentz=1, drude=True, eps_inf=2.0)
    assert fit2["eps_inf"] == 2.0 and fit2["max_n_error"] < 1e-6
    eps = mat.pole_permittivity(fit2["medium"], np.logspace(13, 17, 500))
    assert np.all(eps.imag > 0)


def test_fit_rejects_bad_input():
    lam = np.linspace(1.2, 1.7, 5) * um
    with pytest.raises(ValueError):
        mat.fit_poles(lam, [1.5] * 4)
    with pytest.raises(ValueError):
        mat.fit_poles(lam, [1.5] * 5, k=[-0.1] * 5)
    with pytest.raises(ValueError):
        mat.fit_poles(lam, [1.5] * 5, n_lorentz=0)
    with pytest.raises(ValueError):
        mat.fit_poles(lam, [1.5] * 5, eps_inf=0.5)


def test_discrete_time_permittivity_matches_core_form():
    # pole_permittivity(dt=...) must equal the ADE recursion's transfer function
    # P^{n+1} = c0 P^n + c1 P^{n-1} + c2 E^n (core `Pole::chi_discrete`)
    de, w0, g, dt = 0.8, 2.4e15, 5e13, 3e-17
    med = ls.Medium("m", eps=1.2, poles=[ls.Lorentz(de, w0, g)])
    a = 1 / dt**2 + g / (2 * dt)
    c0, c1, c2 = (2 / dt**2 - w0**2) / a, -(1 / dt**2 - g / (2 * dt)) / a, de * w0**2 / a
    for w in (1.0e15, 1.2e15, 1.5e15):
        z = np.exp(-1j * w * dt)
        assert abs(mat.pole_permittivity(med, w, dt=dt) - (1.2 + c2 / (z - c0 - c1 / z))) < 1e-9
        assert abs(mat.pole_permittivity(med, w, dt=dt) - mat.pole_permittivity(med, w)) < 1e-2


def test_fdtd_plane_wave_in_fitted_si3n4():
    """Phase velocity of a plane wave in the fitted Si3N4 medium vs the library formula."""
    name = "Si3N4"
    med = mat.fitted_medium(name, band=BAND, n_lorentz=2)
    dx = 12.5e-9
    w = 2 * dx
    f0 = C0 / (1.55 * um)
    freqs = [C0 / (1.65 * um), f0, C0 / (1.45 * um)]
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, w, w], mesh=ls.UniformMesh(dx), run_time=70e-15, background=name)
    sim.add_medium(med)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.8 * um, "+", "Ey", ls.GaussianPulse(f0, 0.2 * f0)))
    for mon, x in (("a", 1.5 * um), ("b", 2.5 * um)):
        sim.add_monitor(ls.DFTMonitor(mon, [x, 0, 0], [x, w, w], freqs, components=("Ey",)))
    res = sim.run(device=DEVICE)
    ia = res.raw["dft"]["a"]["fields"]["Ey"]["origin"][0]
    ib = res.raw["dft"]["b"]["fields"]["Ey"]["origin"][0]
    length = (ib - ia) * dx
    assert abs(length - 1 * um) < 2 * dx
    ea = res.field("a", "Ey").reshape(len(freqs), -1)[:, 0]
    eb = res.field("b", "Ey").reshape(len(freqs), -1)[:, 0]
    dt = res.dt
    for i, f in enumerate(freqs):
        omega = 2 * math.pi * f
        k0 = omega / C0
        n_lib = mat.refractive_index(name, C0 / f).real
        expected = k0 * n_lib * length
        dphi = np.angle(complex(eb[i]) / complex(ea[i]))
        dphi += 2 * math.pi * round((expected - dphi) / (2 * math.pi))
        n_eff = dphi / (k0 * length)
        # (1) against the published formula: 0.2 %
        assert abs(n_eff / n_lib - 1) < 2e-3, (f, n_eff, n_lib)
        # (2) against the exact discrete dispersion of the scheme (ADE in time +
        # Yee in space), which removes the O(dx^2) grid dispersion: 0.01 %
        eps_d = mat.pole_permittivity(med, omega, dt=dt)
        s = math.sqrt(eps_d.real) * math.sin(0.5 * omega * dt) * dx / (C0 * dt)
        n_yee = 2 * math.asin(s) / dx / k0
        assert abs(n_eff / n_yee - 1) < 1e-4, (f, n_eff, n_yee)
        # lossless fit -> no attenuation between the two planes
        assert abs(abs(eb[i] / ea[i]) - 1) < 1e-3


# ----------------------------------------------------------------------------
# stability of dispersive media and of the run
# ----------------------------------------------------------------------------


def _slab_column(medium, dx, run_time=300e-15, courant=0.99):
    lam = 1.55 * um
    f0 = C0 / lam
    freqs = np.linspace(0.9, 1.1, 5) * f0
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=run_time, shutoff=1e-7, courant=courant)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if medium is not None:
        sim.add_structure(ls.Box([2 * um, -1, -1], [2.5 * um, 1, 1]), medium)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.2 * f0), name="pw"))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.5 * um, [3.5 * um, -1, -1], [3.5 * um, 1, 1], freqs))
    return sim, freqs


def _slab_formula(eps, freqs, d=0.5 * um):
    n = np.sqrt(np.asarray(eps, dtype=complex))
    r = (1 - n) / (1 + n)
    phi = 2 * np.pi * freqs / C0 * n * d
    return np.abs((1 - r**2) * np.exp(1j * phi) / (1 - r**2 * np.exp(2j * phi))) ** 2


def test_time_step_respects_the_pole_stability_limit():
    """A strong Lorentz pole on eps_inf = 1 (all of silicon's permittivity in
    one ultraviolet pole) is unstable at the Courant step of the mesh: the
    time step is reduced, and the run is finite and correct."""
    dx = 20e-9
    strong = ls.Medium("strong", eps=1.0, poles=[ls.Lorentz(10.7, 6.5e15, 0.0)])
    sim, freqs = _slab_column(strong, dx)
    empty, _ = _slab_column(None, dx)
    dt_mesh, _ = empty.time_info()
    dt, _ = sim.time_info()
    # (4 - a)(4 - b) >= 4 b d_eps / eps_inf with a = 4 C^2, b = (w0 dt)^2
    dt_courant = dt_mesh / 0.99
    stable = lambda t: (4 - 4 * (t / dt_courant) ** 2) * (4 - (6.5e15 * t) ** 2) >= 4 * (6.5e15 * t) ** 2 * 10.7  # noqa: E731
    assert not stable(dt_mesh) and stable(dt) and not stable(dt / 0.97)
    res = sim.run(device=DEVICE)
    assert res.dt == pytest.approx(dt)
    T = np.asarray(res.flux("T")) / np.asarray(empty.run(device=DEVICE).flux("T"))
    assert np.all(np.isfinite(T))
    exact = _slab_formula(mat.pole_permittivity(strong, 2 * np.pi * freqs, dt=res.dt), freqs)
    print("T", np.round(T, 4), "exact", np.round(exact, 4), "dt / dt_mesh", dt / dt_mesh)
    assert np.max(np.abs(T - exact)) < 0.03


def test_a_medium_that_cannot_be_made_stable_is_rejected():
    hopeless = ls.Medium("hopeless", eps=1.0, poles=[ls.Lorentz(50.0, 1e18, 0.0)])
    sim, _ = _slab_column(hopeless, 20e-9)
    with pytest.raises(RuntimeError, match="hopeless.*stable only below"):
        sim.run(device=DEVICE)
    with pytest.raises(RuntimeError, match="hopeless"):
        sim.time_info()


def test_an_unstable_run_stops_with_an_error():
    """Above the Courant limit the fields blow up; the run must raise instead
    of returning NaN."""
    lam = 1.0 * um
    sim = ls.Simulation(min=[-0.4 * um] * 3, max=[0.4 * um] * 3, mesh=ls.UniformMesh(lam / 20), run_time=60e-15, courant=1.05)
    sim.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(C0 / lam, 0.3 * C0 / lam)))
    sim.add_monitor(ls.FluxBoxMonitor("power", [-0.25 * um] * 3, [0.25 * um] * 3, [C0 / lam]))
    with pytest.raises(RuntimeError, match="unstable"):
        sim.run(device=DEVICE)
    sim.courant = 0.99
    assert np.isfinite(sim.run(device=DEVICE).flux("power")[0])


@pytest.mark.parametrize("name", ["strong", "gold"])
def test_block_with_strong_poles_stays_finite(name):
    """A dipole next to a block of a medium with eps_inf = 1 and strong poles,
    in 3-D, where the fastest grid mode is excited: at the Courant step of
    the mesh these runs ended in NaN (library gold at 20 nm cells, a
    silicon-like single pole at 25 nm)."""
    lam = 1.55 * um
    f0 = C0 / lam
    medium, dx = {"strong": (ls.Medium("strong", eps=1.0, poles=[ls.Lorentz(10.7, 6.5e15, 0.0)]), 25e-9), "gold": ("gold", 20e-9)}[name]
    sim = ls.Simulation(min=[-0.4 * um] * 3, max=[0.4 * um] * 3, mesh=ls.UniformMesh(dx), run_time=150e-15)
    dt_mesh = sim.time_info()[0]
    sim.add_structure(ls.Box([-0.15 * um] * 3, [0.15 * um] * 3), medium)
    assert sim.time_info()[0] < 0.95 * dt_mesh
    sim.add_source(ls.PointDipole([0.3 * um, 0, 0], "Ez", ls.GaussianPulse(f0, 0.3 * f0)))
    sim.add_monitor(ls.FluxBoxMonitor("power", [-0.35 * um] * 3, [0.35 * um] * 3, [f0]))
    res = sim.run(device=DEVICE)
    power = res.flux("power")[0]
    print(name, "dt / dt_mesh", res.dt / dt_mesh, "power", power, "steps", res.stats["steps"])
    assert np.isfinite(power) and power > 0


@pytest.mark.parametrize("thickness", [0.5 * um, 3.2 * um])
def test_conductivity_together_with_poles(thickness):
    """sigma was ignored for a medium with poles. The transmission of a slab
    of a medium with a Lorentz pole and a conductivity follows the formula
    with both terms. (The thick slab fills most of the grid, which selects
    the dense dispersive path of the GPU engine.)"""
    lam = 1.0 * um
    f0 = C0 / lam
    w0 = 2 * np.pi * f0
    dx = lam / 60
    freqs = np.linspace(0.8, 1.2, 9) * f0
    both = ls.Medium("both", eps=2.25, sigma=2000.0, poles=[ls.Lorentz(0.5, 1.6 * w0, 0.05 * w0)])
    poles_only = dict(both, name="poles", sigma=[0.0, 0.0, 0.0])

    def transmission(medium):
        sim = ls.Simulation(min=[0, 0, 0], max=[4.2 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=600e-15, shutoff=1e-8)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        if medium is not None:
            sim.add_structure(ls.Box([0.6 * um, -1, -1], [0.6 * um + thickness, 1, 1]), medium)
        sim.add_source(ls.PlaneWave(0, 0.3 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
        sim.add_monitor(ls.FluxMonitor("T", 0, 4.0 * um, [4.0 * um, -1, -1], [4.0 * um, 1, 1], freqs))
        res = sim.run(device=DEVICE)
        return np.asarray(res.flux("T")), res.dt

    incident, _ = transmission(None)
    T, dt = transmission(both)
    T = T / incident
    exact = _slab_formula(mat.pole_permittivity(both, 2 * np.pi * freqs, dt=dt), freqs, d=thickness)
    without_sigma = _slab_formula(mat.pole_permittivity(poles_only, 2 * np.pi * freqs, dt=dt), freqs, d=thickness)
    print("T      ", np.round(T, 4))
    print("exact  ", np.round(exact, 4))
    print("sigma=0", np.round(without_sigma, 4))
    assert np.max(np.abs(without_sigma - exact)) > 0.1  # the conductivity matters here
    assert np.max(np.abs(T - exact)) < 0.01
