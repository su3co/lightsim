"""lightsim.fem electrostatics: closed-form capacitances, EO modulator, gplugins-style API."""

import numpy as np
import pytest

from lightsim import fem
from lightsim.fem import validation

um = 1e-6


@pytest.mark.parametrize("case", ["es_coax", "es_anisotropic_ellipses", "es_parallel_plates", "es_microstrip"])
def test_validation_case(case):
    rep = validation.CASES[case]()
    assert rep.passed, rep.summary()


def test_two_wire_symmetry_and_energy():
    g = fem.Geometry().add(fem.Rectangle(-10 * um, -10 * um, 10 * um, 10 * um), "air", "air")
    g.add(fem.Circle((-2 * um, 0), 0.5 * um, size=0.1 * um), "a", "metal", mesh_order=1)
    g.add(fem.Circle((2 * um, 0), 0.5 * um, size=0.1 * um), "b", "metal", mesh_order=1)
    m = g.mesh(default_size=1 * um)
    C, names = fem.capacitance_matrix(m, ["a", "b"], boundary="ground")
    assert np.allclose(C, C.T) and abs(C[0, 0] / C[1, 1] - 1) < 1e-3 and C[0, 1] < 0
    r = fem.solve_electrostatics(m, {"a": 1.0, "b": 0.0}, boundary="ground")
    assert abs(2 * r.energy() / C[0, 0] - 1) < 1e-6  # W = C11 V^2 / 2


def test_lithium_niobate_modulator_vpil():
    # x-cut thin-film LiNbO3 ridge (600 nm film, 300 nm etch), gold electrodes 5 um apart
    g = fem.Geometry()
    g.add(fem.Rectangle(-10 * um, -2 * um, 10 * um, 0), "box", "SiO2", mesh_order=5)
    g.add(fem.Rectangle(-10 * um, 0, 10 * um, 0.3 * um), "slab", "LiNbO3", mesh_order=3)
    g.add(fem.Trapezoid(0, 1.6 * um, 1.2 * um, 0.3 * um, 0.6 * um), "ridge", "LiNbO3", size=0.05 * um, mesh_order=2)
    g.add(fem.Rectangle(-10 * um, 0.3 * um, 10 * um, 3 * um), "air", "air", mesh_order=10)
    g.add(fem.Rectangle(-6.5 * um, 0.3 * um, -2.5 * um, 0.8 * um), "gnd", "Au", mesh_order=1)
    g.add(fem.Rectangle(2.5 * um, 0.3 * um, 6.5 * um, 0.8 * um), "sig", "Au", mesh_order=1)
    m = g.mesh(default_size=0.5 * um, refine={"gnd": 0.05 * um, "sig": 0.05 * um})
    r = fem.electro_optic_vpil(m, "sig", "gnd", ["slab", "ridge"], "ridge", cut="x-cut", mode_window=1.0 * um, mode_resolution=25e-9)
    vpil_vcm = r["VpiL"] * 100
    # published devices of this kind: 2 to 3 V cm for one arm (plausibility range)
    assert 1.5 < vpil_vcm < 5.0, vpil_vcm
    assert 1.75 < r["neff"].real < 2.1


def test_gplugins_capacitance_on_component():
    gf = pytest.importorskip("gdsfactory")
    from gdsfactory.technology import LayerLevel, LayerStack

    from lightsim.fem import gplugins as gp

    gf.gpdk.PDK.activate()
    from gdsfactory.gpdk import LAYER

    stack = LayerStack(
        layers=dict(
            substrate=LayerLevel(layer=LAYER.WAFER, thickness=5, zmin=0, material="Si", mesh_order=99),
            metal=LayerLevel(layer=LAYER.M1, thickness=0.2, zmin=5, material="Nb", mesh_order=2),
            vacuum=LayerLevel(layer=LAYER.WAFER, thickness=5, zmin=5, material="vacuum", mesh_order=100),
        )
    )
    c = gf.Component()
    c.add_polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)], layer=LAYER.WAFER)
    for x in (-15.0, 15.0):
        c.add_polygon([(x - 10, -10), (x + 10, -10), (x + 10, 10), (x - 10, 10)], layer=LAYER.M1)
    spec = {"Si": {"relative_permittivity": 11.45}, "Nb": {"relative_permittivity": np.inf}, "vacuum": {"relative_permittivity": 1}}
    res = gp.run_capacitive_simulation(c, [(-50, 0), (50, 0)], stack, spec, resolutions={"metal": 0.5}, default_resolution_max=5.0, conductor_refinement=0.1)
    C = res.raw_capacitance_matrix
    assert res.terminals == ["metal_0", "metal_1"]
    assert np.allclose(C, C.T, rtol=1e-9) and C[0, 1] < 0 < C[0, 0]
    assert abs(C[0, 0] / C[1, 1] - 1) < 2e-3
    assert res.capacitance_matrix[("metal_0", "metal_1")] == C[0, 1]
