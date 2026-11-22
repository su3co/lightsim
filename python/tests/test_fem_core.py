"""lightsim.fem core: mesher, P1/P2 convergence, gdsfactory cross-section cut."""

import numpy as np
import pytest

from lightsim import fem
from lightsim.fem import validation

um = 1e-6


@pytest.mark.parametrize("case", ["poisson_convergence", "isoparametric_disk"])
def test_validation_case(case):
    rep = validation.CASES[case]()
    assert rep.passed, rep.summary()


def test_mesh_regions_sizes_and_tags():
    g = fem.Geometry()
    g.add(fem.Rectangle(-3 * um, -2 * um, 3 * um, 0), "box", "SiO2")
    g.add(fem.Rectangle(-3 * um, 0, 3 * um, 2 * um), "clad", "SiO2")
    g.add(fem.Rectangle(-0.25 * um, 0, 0.25 * um, 0.22 * um), "core", "Si", size=0.02 * um, mesh_order=1)
    m = g.mesh(default_size=0.3 * um, refine={"core": 0.01 * um})
    assert m.min_angle_deg > 20
    a = m.areas()
    assert abs(a[m.region_mask("core")].sum() - 0.5 * 0.22 * um**2) < 1e-6 * um**2
    assert abs(a.sum() - 24 * um**2) < 1e-6 * um**2
    # interface refinement: edges on the core boundary near 10 nm
    e = m.interface_edges("core")
    assert np.max(m.edge_lengths()[e]) < 0.012 * um
    assert len(m.boundary_edges("bottom")) > 0 and len(m.boundary_edges(["left", "right"])) > 0
    f = fem.Field(m, 1, m.nodes[:, 0])  # u = x is reproduced exactly
    x = np.array([0.1, -1.0]) * um
    y = np.array([0.1, 1.0]) * um
    assert np.allclose(f(x, y), x, rtol=0, atol=1e-18)


def test_gdsfactory_cut():
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    c = gf.components.straight_heater_metal(length=50)
    ls = gf.pdk.get_layer_stack()
    geo = fem.cross_section_geometry(c, ls, [(25, -6), (25, 6)], exclude=("substrate",))
    names = geo.names()
    for n in ("box", "core", "clad", "heater"):
        assert n in names
    m = geo.mesh(default_size=0.5 * um, sizes={"core": 0.03 * um, "heater": 0.1 * um})
    core = m.region_mask("core")
    # 10 degree sidewall, width 0.5 um at mid-height: trapezoid area = 0.5 * 0.22 um^2
    assert abs(m.areas()[core].sum() - 0.5 * 0.22 * um**2) < 1e-3 * um**2
    assert m.materials[m.tag("heater")] == "TiN"
