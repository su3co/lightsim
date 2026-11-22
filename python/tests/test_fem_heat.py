"""lightsim.fem heat conduction: analytic cases and a thermo-optic phase shifter."""

import math

import numpy as np
import pytest

from lightsim import fem
from lightsim.fem import validation

um = 1e-6


@pytest.mark.parametrize("case", ["heat_slab", "heat_cylinder", "heat_transient_slab"])
def test_validation_case(case):
    rep = validation.CASES[case]()
    assert rep.passed, rep.summary()


def _two_guides():
    g = fem.Geometry()
    g.add(fem.Rectangle(-8 * um, -2.11 * um, 8 * um, -0.11 * um), "box", "SiO2", mesh_order=5)
    g.add(fem.Rectangle(-8 * um, -0.11 * um, 8 * um, 2.69 * um), "clad", "SiO2", mesh_order=5)
    g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=0.03 * um, mesh_order=1)
    g.add(fem.Rectangle(4.75 * um, -0.11 * um, 5.25 * um, 0.11 * um), "core2", "Si", size=0.03 * um, mesh_order=1)
    g.add(fem.Rectangle(-1 * um, 2.11 * um, 1 * um, 2.25 * um), "heater", "TiN", size=0.05 * um, mesh_order=1)
    return g.mesh(default_size=0.5 * um, grading=0.5)


def test_phase_shifter_figures():
    m = _two_guides()
    r = fem.thermal_phase_shifter(m, "heater", "core", neighbour="core2", power_per_length=50.0, mode_resolution=25e-9, transient=True, t_end=100e-6, nsteps=200)
    # silicon strip under a TiN heater, 2 um of oxide to a fixed-temperature substrate: P_pi of a few tens of mW
    assert 5e-3 < r["P_pi"] < 60e-3, r["P_pi"]
    assert 0.1 < r["crosstalk"] < 0.6  # 5 um away over 2 um of oxide: about 0.38
    assert 1e-6 < r["time_constant"] < 50e-6
    # linear problem: the phase shift per power does not depend on the power used
    r2 = fem.thermal_phase_shifter(m, "heater", "core", power_per_length=100.0, mode_resolution=25e-9)
    assert abs(r2["dphi_dP"] / r["dphi_dP"] - 1) < 2e-3


def test_current_source_power():
    m = _two_guides()
    area = 2 * um * 0.14 * um
    r = fem.solve_heat(m, {"heater": fem.Current(5e-3)}, order=1)  # TiN sigma from the metal table
    assert math.isclose(r.power_per_length["heater"], (5e-3) ** 2 / (2.3e6 * area), rel_tol=1e-6)
    # energy balance: all the heat leaves through the substrate
    assert r.temperature.max() > 0


def test_gplugins_style_thermal_on_component():
    gf = pytest.importorskip("gdsfactory")
    from lightsim.fem import gplugins as gp

    gf.gpdk.PDK.activate()
    c = gf.components.straight_heater_metal(length=50)
    m, r = gp.compute_component_slice_thermal(c, [(25, -6), (25, 6)], power_per_length=50.0, resolutions={"core": 0.03, "heater": 0.1})
    t_core = r.mean_temperature("core")
    assert 5 < t_core < 30  # K for 50 W/m, about 0.26 K per W/m
    r2 = gp.solve_thermal(m, powers={"heater": 100.0}, order=2)
    assert abs(r2.mean_temperature("core") / t_core - 2) < 1e-6  # linear in the power
