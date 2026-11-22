"""gdsfactory import: polygons with holes, wafer levels, unknown materials."""

import os

import numpy as np
import pytest

gf = pytest.importorskip("gdsfactory")

import lightsim as ls  # noqa: E402
from lightsim import _core, gds  # noqa: E402
from lightsim.spec import um  # noqa: E402

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


@pytest.fixture(autouse=True, scope="module")
def _pdk():
    gf.gpdk.PDK.activate()


def _polygon_area(points):
    x, y = np.asarray(points).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_ring_drawn_as_one_polygon_with_a_hole():
    """A ring made by subtracting two discs is one polygon with a hole. It is
    cut into hole-free pieces whose union is the ring: the area matches, the
    hole stays empty in the rasterized permittivity."""
    r_out, r_in = 3.0, 2.5
    ring = gf.boolean(A=gf.components.circle(radius=r_out, layer=(1, 0)), B=gf.components.circle(radius=r_in, layer=(1, 0)), operation="not", layer=(1, 0))
    level = gf.pdk.get_layer_stack().layers["core"]
    assert [p.holes() for p in level.layer.get_shapes(ring).each()] == [1]

    layers = {l.name: l for l in gds.stack_layers(ring)}
    pieces = layers["core"].polygons
    assert len(pieces) > 1 and all(len(p) >= 3 for p in pieces)
    area = sum(_polygon_area(p) for p in pieces)
    exact = np.pi * (r_out**2 - r_in**2) * um**2
    assert abs(area / exact - 1) < 0.02  # the discs are polygons

    # rasterized: silicon on the ring, oxide in the hole and outside
    structures, media, _ = gds.component_to_structures(ring)
    dx = 50e-9
    sim = ls.Simulation(min=[-3.5 * um, -3.5 * um, 0.0], max=[3.5 * um, 3.5 * um, 0.2 * um], mesh=ls.UniformMesh(dx), run_time=1e-15, background="sio2")
    for m in media:
        sim.add_medium(m)
    sim.structures.extend(structures)
    info = _core.plane_info(sim.to_json(), 2, 0.1 * um, [-3.4 * um, -3.4 * um, 0.1 * um], [3.4 * um, 3.4 * um, 0.1 * um])
    eps = np.asarray(info["eps_n"])  # Ez on the nodes of the plane
    x, y = np.asarray(info["u_nodes"])[: eps.shape[0], None], np.asarray(info["v_nodes"])[None, : eps.shape[1]]
    radius = np.hypot(x, y) / um
    si, ox = 3.476**2, 1.444**2
    assert np.allclose(eps[radius < r_in - 0.1], ox, rtol=1e-5)
    assert np.allclose(eps[radius > r_out + 0.1], ox, rtol=1e-5)
    assert np.allclose(eps[(radius > r_in + 0.1) & (radius < r_out - 0.1)], si, rtol=1e-5)
    # filled area from the smoothed permittivity: no seams between the pieces
    fill = np.sum((eps - ox) / (si - ox)) * dx * dx
    assert abs(fill / exact - 1) < 0.02


def _custom_stack(clad_background=False):
    from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

    return LayerStack(
        layers={
            "box": LayerLevel(layer=LogicalLayer(layer=(98, 0)), thickness=2.0, zmin=-2.0, material="sio2", mesh_order=9, background=True),
            "core": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.22, zmin=0.0, material="si", mesh_order=1),
            # a cladding on a layer of its own, which no component draws on
            "clad": LayerLevel(layer=LogicalLayer(layer=(99, 0)), thickness=2.0, zmin=0.0, material="sin", mesh_order=10, background=clad_background),
        }
    )


def test_wafer_levels_on_custom_layers():
    import warnings

    wg = gf.components.straight(length=2)

    # the level with background=True is a wafer level; the other one is left out, and said so
    with pytest.warns(UserWarning, match=r"left out: 'clad' \(sin, z = 0 to 2 um\)"):
        names = [l.name for l in gds.stack_layers(wg, _custom_stack())]
    assert names == ["box", "core"]

    # marked by the caller
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layers = gds.stack_layers(wg, _custom_stack(), wafer_levels=["clad"])
        structures, media, _ = gds.component_to_structures(wg, _custom_stack(), wafer_levels=["clad"])
    assert [(l.name, bool(l.polygons)) for l in layers] == [("box", False), ("core", True), ("clad", False)]
    assert [(s["name"], s["shape"]["type"], s["medium"]) for s in structures] == [("box", "box", "sio2"), ("clad", "box", "sin"), ("core_0", "poly_slab", "si")]
    assert {m["name"] for m in media} == {"sio2", "sin", "si"}

    # marked in the stack
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [l.name for l in gds.stack_layers(wg, _custom_stack(clad_background=True))] == ["box", "core", "clad"]

    with pytest.raises(ValueError, match="not levels of the layer stack"):
        gds.stack_layers(wg, _custom_stack(), wafer_levels=["cladding"])


def test_generic_pdk_levels_are_found_without_a_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layers = gds.stack_layers(gf.components.mmi1x2())
    assert [(l.name, bool(l.polygons)) for l in layers] == [("box", False), ("core", True), ("clad", False)]


def test_wafer_levels_reach_the_simulation():
    from lightsim.sparams import build_simulation

    wg = gf.components.straight(length=2)
    sim, _, _ = build_simulation(wg, [1.55e-6], layer_stack=_custom_stack(), wafer_levels=["clad"], steps_per_wavelength=8)
    assert [s["name"] for s in sim.structures][:2] == ["box", "clad"]
    with pytest.warns(UserWarning, match="left out: 'clad'"):
        sim, _, _ = build_simulation(wg, [1.55e-6], layer_stack=_custom_stack(), steps_per_wavelength=8)
    assert "clad" not in [s["name"] for s in sim.structures]


def _stack_with_material(material):
    from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

    return LayerStack(layers={"core": LayerLevel(layer=LogicalLayer(layer=(1, 0)), thickness=0.22, zmin=0.0, material=material, mesh_order=1)})


def test_unknown_layer_material_is_an_error():
    """A material name without a refractive index became n = 1 silently."""
    from lightsim import eim
    from lightsim.sparams import build_simulation

    wg = gf.components.straight(length=2)
    stack = _stack_with_material("SiN_PECVD")
    for call in (
        lambda: gds.component_to_structures(wg, stack),
        lambda: build_simulation(wg, [1.55e-6], layer_stack=stack),
        lambda: eim.effective_indices(wg, 1.55e-6, layer_stack=stack),
        lambda: eim.effective_index_simulation(wg, [1.55e-6], layer_stack=stack),
    ):
        with pytest.raises(ValueError, match=r"level 'core': material 'sin_pecvd' has no refractive index.*material_index=\{'sin_pecvd': n\}"):
            call()
    # with the index supplied (names are matched in lower case)
    _, media, _ = gds.component_to_structures(wg, stack, material_index={"SiN_PECVD": 1.9})
    assert [(m["name"], round(m["eps"][0] ** 0.5, 6)) for m in media] == [("sin_pecvd", 1.9)]
    neff, n_bg, _ = eim.effective_indices(wg, 1.55e-6, layer_stack=stack, material_index={"sin_pecvd": 1.9})
    assert 1.444 < neff["core"] < 1.9 and n_bg == 1.444

    # the background: unknown names are refused, names the core knows are left to it
    with pytest.raises(ValueError, match="background: material 'mystery' has no refractive index"):
        build_simulation(wg, [1.55e-6], background="mystery")
    with pytest.raises(ValueError, match="background: material 'mystery'"):
        eim.effective_indices(wg, 1.55e-6, background="mystery")
    sim, _, _ = build_simulation(wg, [1.55e-6], layer_stack=_stack_with_material("si"), background="silica", steps_per_wavelength=8)
    assert [m["eps"][0] for m in sim.media if m["name"] == "silica"] == [1.444**2]
    sim, _, _ = build_simulation(wg, [1.55e-6], layer_stack=_stack_with_material("si"), background="mystery", material_index={"mystery": 1.6}, steps_per_wavelength=8)
    assert [m["eps"][0] for m in sim.media if m["name"] == "mystery"] == [1.6**2]

    # a material that only the core defines (dispersive gold) is passed through by name
    structures, media, _ = gds.component_to_structures(wg, _stack_with_material("gold"))
    assert structures[0]["medium"] == "gold" and media == []
