"""Side-by-side runs of lightsim.fem and femwell (test-time comparison only).

femwell (https://github.com/HelgeGehring/femwell) is not a dependency of
lightsim; these functions are used by test_fem_compare_femwell.py and print
the numbers quoted in docs/guide/fem-heat.md and fem-modes.md when run as a
script: ``python python/tests/fem_femwell_reference.py``.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np

# TiN top heater over a silicon strip (femwell's heater example, the
# geometry of gplugins' femwell_02_heater notebook), lengths in um
W_SIM, H_CLAD, H_BOX, W_CORE, H_CORE, H_HEATER, W_HEATER, H_SI = 16.0, 2.8, 2.0, 0.5, 0.22, 0.14, 2.0, 0.5
OFFSET_HEATER = 2 + (H_CORE + H_HEATER) / 2
K = {"core": 90.0, "box": 1.38, "clad": 1.38, "heater": 28.0, "wafer": 148.0}
SIGMA_HEATER = 2.3e6
DNDT = {"core": 1.86e-4, "other": 1.0e-5}
N_SI, N_OX = 3.4777, 1.444
CURRENT = 7.4e-3  # A


def heater_femwell(core_res=0.02, default_max=0.4):
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from femwell.thermal import solve_thermal
    from shapely.geometry import LineString, Polygon
    from skfem import Basis, ElementTriP0
    from skfem.io import from_meshio

    hc = H_CORE / 2
    polygons = OrderedDict(
        bottom=LineString([(-W_SIM / 2, -hc - H_BOX - H_SI), (W_SIM / 2, -hc - H_BOX - H_SI)]),
        core=Polygon([(-W_CORE / 2, -hc), (-W_CORE / 2, hc), (W_CORE / 2, hc), (W_CORE / 2, -hc)]),
        heater=Polygon([(-W_HEATER / 2, -H_HEATER / 2 + OFFSET_HEATER), (-W_HEATER / 2, H_HEATER / 2 + OFFSET_HEATER), (W_HEATER / 2, H_HEATER / 2 + OFFSET_HEATER), (W_HEATER / 2, -H_HEATER / 2 + OFFSET_HEATER)]),
        clad=Polygon([(-W_SIM / 2, -hc), (-W_SIM / 2, -hc + H_CLAD), (W_SIM / 2, -hc + H_CLAD), (W_SIM / 2, -hc)]),
        box=Polygon([(-W_SIM / 2, -hc), (-W_SIM / 2, -hc - H_BOX), (W_SIM / 2, -hc - H_BOX), (W_SIM / 2, -hc)]),
        wafer=Polygon([(-W_SIM / 2, -hc - H_BOX - H_SI), (-W_SIM / 2, -hc - H_BOX), (W_SIM / 2, -hc - H_BOX), (W_SIM / 2, -hc - H_BOX - H_SI)]),
    )
    resolutions = dict(core={"resolution": core_res, "distance": 1}, clad={"resolution": 0.4, "distance": 1}, box={"resolution": 0.4, "distance": 1}, heater={"resolution": 0.05, "distance": 1})
    t0 = time.perf_counter()
    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=default_max))
    t_mesh = time.perf_counter() - t0
    t0 = time.perf_counter()
    basis0 = Basis(mesh, ElementTriP0(), intorder=4)
    kp0 = basis0.zeros()
    for d, v in K.items():
        kp0[basis0.get_dofs(elements=d)] = v
    kp0 *= 1e-12
    J = CURRENT / polygons["heater"].area
    basis, T = solve_thermal(basis0, kp0, specific_conductivity={"heater": SIGMA_HEATER}, current_densities={"heater": J}, fixed_boundaries={"bottom": 0})
    t_heat = time.perf_counter() - t0
    T0 = basis0.project(basis.interpolate(T))
    t0 = time.perf_counter()
    neffs = []
    for scale in (0.0, 1.0):
        eps = basis0.zeros() + (N_OX + DNDT["other"] * scale * T0) ** 2
        cd = basis0.get_dofs(elements="core")
        eps[cd] = (N_SI + DNDT["core"] * scale * T0[cd]) ** 2
        modes = compute_modes(basis0, eps, wavelength=1.55, num_modes=1, solver="scipy")
        neffs.append(np.real(modes[0].n_eff))
    t_modes = (time.perf_counter() - t0) / 2
    core_mask = basis0.get_dofs(elements="core")
    return {
        "T_core": float(np.mean(T0[core_mask])),
        "T_max": float(np.max(T)),
        "neff": float(neffs[0]),
        "dneff": float(neffs[1] - neffs[0]),
        "n_tris": int(mesh.t.shape[1]),
        "t_mesh": t_mesh,
        "t_heat": t_heat,
        "t_mode": t_modes,
    }


def heater_lightsim(core_res=0.02, default_max=0.4, order=1, mode_method="fem", size_scale=2.0):
    """The heater case with femwell's resolutions (``core_res``, ``default_max``
    and the cladding and heater values of heater_femwell) times ``size_scale``.

    lightsim's region size is a largest edge length: the mean edge comes out at
    about 0.4 times the size, against about 0.9 times gmsh's resolution, so
    ``size_scale=2`` gives femwell's triangle count (6 682 against 6 636 at
    ``core_res=0.02``). ``mode_method="fem"`` solves the two modes (unheated,
    heated) with the finite-element mode solver on the heat mesh, as femwell
    does; ``"fd"`` with the finite-difference solver on a 20 nm grid.
    """
    from lightsim import fem

    um = 1e-6 * size_scale  # region sizes in femwell's units times size_scale
    lu = 1e-6  # lengths
    hc = H_CORE / 2 * lu
    g = fem.Geometry()
    g.add(fem.Rectangle(-W_SIM / 2 * lu, -hc - (H_BOX + H_SI) * lu, W_SIM / 2 * lu, -hc - H_BOX * lu), "wafer", "Si", mesh_order=5)
    g.add(fem.Rectangle(-W_SIM / 2 * lu, -hc - H_BOX * lu, W_SIM / 2 * lu, -hc), "box", "SiO2", size=0.4 * um, mesh_order=5)
    g.add(fem.Rectangle(-W_SIM / 2 * lu, -hc, W_SIM / 2 * lu, -hc + H_CLAD * lu), "clad", "SiO2", size=0.4 * um, mesh_order=5)
    g.add(fem.Rectangle(-W_CORE / 2 * lu, -hc, W_CORE / 2 * lu, hc), "core", "Si", size=core_res * um, mesh_order=1)
    g.add(fem.Rectangle(-W_HEATER / 2 * lu, (OFFSET_HEATER - H_HEATER / 2) * lu, W_HEATER / 2 * lu, (OFFSET_HEATER + H_HEATER / 2) * lu), "heater", "TiN", size=0.05 * um, mesh_order=1)
    t0 = time.perf_counter()
    m = g.mesh(default_size=default_max * um, grading=0.5)
    t_mesh = time.perf_counter() - t0
    t0 = time.perf_counter()
    area = W_HEATER * H_HEATER * lu**2
    r = fem.solve_heat(m, {"heater": fem.Current(CURRENT, SIGMA_HEATER)}, conductivity=K, fixed={"bottom": 0.0}, order=order)
    t_heat = time.perf_counter() - t0
    out = {"T_core": r.mean_temperature("core"), "T_max": r.temperature.max(), "n_tris": m.n_tris, "t_mesh": t_mesh, "t_heat": t_heat, "power_per_length": CURRENT**2 / (SIGMA_HEATER * area)}
    n0 = np.where(m.region_mask("core"), N_SI, N_OX)
    dndt = np.where(m.region_mask("core"), DNDT["core"], DNDT["other"])
    dn = dndt * r.temperature.per_triangle()
    t0 = time.perf_counter()
    if mode_method == "fd":
        win = (-1.5 * lu - W_CORE / 2 * lu, -1.5 * lu - hc, 1.5 * lu + W_CORE / 2 * lu, 1.5 * lu + hc)
        a, _ = fem.heat.neff_fd(m, n0, 1.55e-6, win, 20e-9)
        b, _ = fem.heat.neff_fd(m, n0 + dn, 1.55e-6, win, 20e-9)
        out["neff"], out["dneff"] = a[0].real, (b[0] - a[0]).real
    else:
        ms0 = fem.solve_modes(m, 1.55e-6, index=n0, num_modes=1)
        ms1 = fem.solve_modes(m, 1.55e-6, index=n0 + dn, num_modes=1)
        out["neff"], out["dneff"] = ms0[0].neff.real, (ms1[0].neff - ms0[0].neff).real
    out["t_mode"] = (time.perf_counter() - t0) / 2
    return out


if __name__ == "__main__":
    for name, fn in (("femwell", heater_femwell), ("lightsim P1", lambda: heater_lightsim(order=1)), ("lightsim P2", lambda: heater_lightsim(order=2))):
        r = fn()
        print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


# rib waveguide of gplugins' femwell test (generic PDK, core 0.2 um thick)
RIB_INDEX = {"si": 3.48, "sio2": 1.44}


def _rib_cut():
    import gdsfactory as gf
    from gdsfactory.gpdk import LAYER_STACK
    from gdsfactory.technology import LayerStack

    from lightsim import fem

    gf.gpdk.PDK.activate()
    ls = LayerStack(layers={k: LAYER_STACK.layers[k].model_copy(deep=True) for k in ("core", "clad", "slab90", "box")})
    ls.layers["core"].thickness = 0.2
    c = gf.components.straight(length=10, cross_section="rib")
    xm = 0.5 * (c.xmin + c.xmax)
    bounds = [(xm, c.ymin - 2), (xm, c.ymax + 2)]
    return c, ls, bounds, fem.cross_section_geometry(c, ls, bounds)


def rib_femwell(core_res=0.02, slab_res=0.05, default_max=0.3, num_modes=2):
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from shapely.geometry import Polygon
    from skfem import Basis, ElementTriP0
    from skfem.io import from_meshio

    _, _, _, geo = _rib_cut()
    from shapely.ops import unary_union

    regions = sorted(geo.regions, key=lambda r: r.mesh_order)
    # disjoint pieces in priority order (what meshwell does for gplugins); the
    # rib (core and slab, both silicon) as one polygon: femwell's gmsh mesher
    # does not mesh the core and slab pieces conformingly otherwise
    by_name = {r.name: Polygon(np.asarray(r.shape.points) * 1e6) for r in regions}
    rib = unary_union([by_name["core"], by_name["slab90"]])
    polys = OrderedDict(core=rib, box=by_name["box"], clad=by_name["clad"].difference(rib))
    mats = {"core": "si", "box": "sio2", "clad": "sio2"}
    res = {"core": {"resolution": core_res, "distance": 0.5}}
    t0 = time.perf_counter()
    mesh = from_meshio(mesh_from_OrderedDict(polys, res, default_resolution_max=default_max))
    t_mesh = time.perf_counter() - t0
    basis0 = Basis(mesh, ElementTriP0())
    eps = basis0.zeros(dtype=complex)
    for name, mat in mats.items():
        eps[basis0.get_dofs(elements=name)] = RIB_INDEX[mat.lower()] ** 2
    t0 = time.perf_counter()
    modes = compute_modes(basis0, eps, wavelength=1.55, num_modes=num_modes, order=1)
    t_modes = time.perf_counter() - t0
    return {"neff": [complex(m.n_eff).real for m in modes], "te": [float(m.te_fraction) for m in modes], "n_tris": int(mesh.t.shape[1]), "t_mesh": t_mesh, "t_modes": t_modes}


def rib_lightsim(core_res=0.02, slab_res=0.05, default_max=0.3, num_modes=2):
    from lightsim.fem import gplugins as gp

    c, ls, bounds, _ = _rib_cut()
    t0 = time.perf_counter()
    modes = gp.compute_component_slice_modes(c, bounds, ls, 1.55, num_modes, material_name_to_index=RIB_INDEX, resolution_specs={"core": core_res, "slab90": slab_res}, default_resolution_max=default_max)
    t = time.perf_counter() - t0
    return {"neff": [m.neff.real for m in modes], "te": [m.te_fraction for m in modes], "n_tris": modes[0].mesh.n_tris, "t_total": t}


if __name__ == "__main__" and "rib" in __import__("sys").argv:
    for res in (0.04, 0.02, 0.01):
        print("core res", res, "femwell", rib_femwell(res), "lightsim", rib_lightsim(res))
