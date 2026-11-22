"""IndexGrid (imported / graded permittivity) builder and the silicon
plasma-dispersion helpers in `lightsim.eo`."""

import json
import math
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import _core, eo
from lightsim.spec import IndexGrid, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_builder_padding_and_layout():
    a = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    g = IndexGrid([0, 0, 0], [1, 2, 3], a)
    assert g["type"] == "index_grid" and g["dims"] == [2, 3, 4]
    # x-major: eps[(i * ny + j) * nz + k]
    assert g["eps"][(1 * 3 + 2) * 4 + 3] == a[1, 2, 3]
    # 1-D -> along x by default, or along the requested axis
    assert IndexGrid([0, 0, 0], [1, 1, 1], [1.0, 2.0, 3.0])["dims"] == [3, 1, 1]
    assert IndexGrid([0, 0, 0], [1, 1, 1], [1.0, 2.0, 3.0], axes=(2,))["dims"] == [1, 1, 3]
    # 2-D -> (x, y) by default; (y, z) cross-section; swapped axes are transposed
    b = np.arange(6, dtype=float).reshape(2, 3) + 1.0
    assert IndexGrid([0, 0, 0], [1, 1, 1], b)["dims"] == [2, 3, 1]
    yz = IndexGrid([0, 0, 0], [1, 1, 1], b, axes=(1, 2))
    assert yz["dims"] == [1, 2, 3] and yz["eps"] == b.ravel().tolist()
    zy = IndexGrid([0, 0, 0], [1, 1, 1], b, axes=(2, 1))
    assert zy["dims"] == [1, 3, 2] and zy["eps"] == b.T.ravel().tolist()
    # scalar -> constant box; refractive index is squared
    assert IndexGrid([0, 0, 0], [1, 1, 1], 4.0)["dims"] == [1, 1, 1]
    assert IndexGrid([0, 0, 0], [1, 1, 1], n=[1.0, 2.0])["eps"] == [1.0, 4.0]
    # anisotropic
    an = IndexGrid([0, 0, 0], [1, 1, 1], n_axes=([1.0, 2.0], [1.0, 3.0], [2.0, 2.0]))
    assert an["dims"] == [2, 1, 1] and an["eps_axes"] == [[1.0, 4.0], [1.0, 9.0], [4.0, 4.0]]
    json.dumps(g)  # plain JSON


def test_builder_rejects_bad_input():
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 1, 1], [1.0, 2.0], n=[1.0, 2.0])
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 1, 1], [1.0, -2.0])
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 1, 1], [1.0 + 0.1j, 2.0])
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 0, 1], [1.0, 2.0])
    with pytest.raises(ValueError):
        IndexGrid([0, 0, 0], [1, 1, 1], np.ones((2, 2)), axes=(0,))


# ---------------------------------------------------------------------------
# round trip through the Rust core
# ---------------------------------------------------------------------------


def cross_section_sim(dx=25e-9):
    sim = ls.Simulation(min=[0, -1 * um, -0.5 * um], max=[1 * um, 1 * um, 0.5 * um], mesh=ls.UniformMesh(dx), run_time=1e-15)
    return sim


def plane(sim, freq=None):
    args = (sim.to_json(), 0, 0.5 * um, [0.5 * um, -1 * um, -0.5 * um], [0.5 * um, 1 * um, 0.5 * um])
    return _core.plane_info(*args) if freq is None else _core.plane_info(*args, freq)


def test_core_interpolates_cross_section_map():
    # eps = 2 + a y + b z + c y z is reproduced exactly by (bi)linear interpolation
    lo, hi = [-1, -0.4 * um, -0.2 * um], [1, 0.4 * um, 0.2 * um]
    y = np.linspace(lo[1], hi[1], 2)
    z = np.linspace(lo[2], hi[2], 2)

    def f(yy, zz):
        return 2.0 + 1.5e6 * yy + 2.0e6 * zz + 1.0e12 * yy * zz

    sim = cross_section_sim()
    sim.add_structure(IndexGrid(lo, hi, f(y[:, None], z[None, :]), axes=(1, 2)), ls.Medium("graded", eps=3.0))
    info = plane(sim)
    # E_normal (Ex) sits on the (y node, z node) lattice of the plane
    u = info["u_nodes"][: info["dims"][0]]
    v = info["v_nodes"][: info["dims"][1]]
    U, V = np.meshgrid(u, v, indexing="ij")
    eps = info["eps_n"]
    inside = (U >= lo[1]) & (U < hi[1] - 1e-12) & (V >= lo[2]) & (V < hi[2] - 1e-12)
    outside = (U < lo[1] - 1e-12) | (U > hi[1] + 1e-12) | (V < lo[2] - 1e-12) | (V > hi[2] + 1e-12)
    assert inside.sum() > 400
    assert np.max(np.abs(eps[inside] - f(U[inside], V[inside]))) < 1e-5
    assert np.all(eps[outside] == 1.0)


def test_core_rejects_wrong_sample_count_and_dispersive_medium():
    sim = cross_section_sim()
    g = IndexGrid([-1, -1, -1], [1, 1, 1], [2.0, 3.0])
    g["eps"] = g["eps"] + [4.0]
    sim.add_structure(g, ls.Medium("m", eps=3.0))
    with pytest.raises(Exception, match="index_grid"):
        plane(sim)
    sim = cross_section_sim()
    disp = ls.Medium("disp", eps=3.0, poles=[ls.Lorentz(1.0, 1e16, 1e14)])
    sim.add_structure(IndexGrid([-1, -1, -1], [1, 1, 1], [2.0, 3.0]), disp)
    with pytest.raises(Exception, match="index_grid"):
        plane(sim)


def test_constant_grid_equals_box_in_fdtd():
    lam = 1 * um
    f0 = ls.C0 / lam
    dx = lam / 30
    w = 2 * dx
    lo, hi = [2.013 * um, -1, -1], [2.487 * um, 1, 1]
    flux = []
    for shape in (ls.Box(lo, hi), IndexGrid(lo, hi, np.full((3, 2), 4.0), axes=(0, 2))):
        sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, w, w], mesh=ls.UniformMesh(dx), run_time=80e-15, subpixel=False)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        sim.add_structure(shape, ls.Medium("d", eps=4.0))
        sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", ls.GaussianPulse(f0, 0.3 * f0)))
        sim.add_monitor(ls.FluxMonitor("T", 0, 3.2 * um, [3.2 * um, -1, -1], [3.2 * um, 1, 1], [0.9 * f0, f0, 1.1 * f0]))
        flux.append(np.asarray(sim.run(device=DEVICE).flux("T")))
    assert np.all(flux[0] > 0)
    assert np.max(np.abs(flux[1] / flux[0] - 1)) < 1e-6


# ---------------------------------------------------------------------------
# Soref-Bennett
# ---------------------------------------------------------------------------


def test_soref_bennett_published_values():
    # Soref & Bennett 1987 fit at 1550 nm, 1e18 cm^-3
    dn_e, da_e = eo.soref_bennett(1e18, 0.0, 1550e-9)
    dn_h, da_h = eo.soref_bennett(0.0, 1e18, 1550e-9)
    assert dn_e == pytest.approx(-8.8e-4, rel=1e-12)
    assert dn_h == pytest.approx(-8.5e-18 * 1e18**0.8, rel=1e-12)
    assert dn_h == pytest.approx(-2.1e-3, rel=0.02)
    assert da_e == pytest.approx(8.5, rel=1e-12) and da_h == pytest.approx(6.0, rel=1e-12)
    # both carriers: -3.0e-3 and 14.5 /cm
    dn, da = eo.soref_bennett(1e18, 1e18, 1550e-9)
    assert dn == pytest.approx(-3.015e-3, rel=1e-3) and da == pytest.approx(14.5)
    # 1310 nm: the abstract's "1e18 carriers/cm^3 -> 1.5e-3 at 1.3 um" (holes)
    dn_h13, da_h13 = eo.soref_bennett(0.0, 1e18, 1310e-9)
    assert dn_h13 == pytest.approx(-1.5e-3, rel=0.01) and da_h13 == pytest.approx(4.0)
    dn_e13, da_e13 = eo.soref_bennett(1e18, 0.0, 1310e-9)
    assert dn_e13 == pytest.approx(-6.2e-4) and da_e13 == pytest.approx(6.0)
    # the hole term is sub-linear (dP^0.8), the electron term linear
    assert eo.soref_bennett(0.0, 1e17, 1550e-9)[0] == pytest.approx(dn_h / 10**0.8)
    assert eo.soref_bennett(1e17, 0.0, 1550e-9)[0] == pytest.approx(dn_e / 10)
    # arrays, zero carriers, tolerance on the wavelength, errors
    dn, da = eo.soref_bennett(np.array([0.0, 1e18]), np.array([0.0, 1e18]), 1.56e-6)
    assert dn[0] == 0.0 and da[0] == 0.0 and dn[1] == pytest.approx(-3.015e-3, rel=1e-3)
    with pytest.raises(ValueError):
        eo.soref_bennett(1e18, 1e18, 1064e-9)
    with pytest.raises(ValueError):
        eo.soref_bennett(-1.0, 0.0)


def test_nedeljkovic_2011_fit():
    dn, da = eo.soref_bennett(1e18, 1e18, 1550e-9, model="nedeljkovic2011")
    want_dn = -(5.4e-22 * 1e18**1.011 + 1.53e-18 * 1e18**0.838)
    want_da = 8.88e-21 * 1e18**1.167 + 5.84e-20 * 1e18**1.109
    assert dn == pytest.approx(want_dn) and da == pytest.approx(want_da)
    # same physics as the 1987 fit: within a factor of ~2 at 1e18
    dn0, da0 = eo.soref_bennett(1e18, 1e18, 1550e-9)
    assert 0.5 < dn / dn0 < 1.5 and 0.5 < da / da0 < 2.5


def test_carrier_map_to_index_grid_and_loss():
    lam = 1550e-9
    n_si = eo.N_SILICON[1550]
    # pn-junction-like (y, z) map: electrons on one side, holes on the other
    ny, nz = 9, 5
    dN = np.zeros((ny, nz))
    dP = np.zeros((ny, nz))
    dN[: ny // 2] = 1e18
    dP[ny // 2 + 1 :] = 1e18
    lo, hi = [-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]
    g = eo.carrier_index_grid(lo, hi, dN, dP, wavelength=lam, axes=(1, 2))
    assert g["dims"] == [1, ny, nz]
    eps = np.array(g["eps"]).reshape(ny, nz)
    assert eps[0, 0] == pytest.approx((n_si - 8.8e-4) ** 2)
    assert eps[-1, 0] == pytest.approx((n_si - 8.5e-18 * 1e18**0.8) ** 2)
    assert eps[ny // 2, 0] == pytest.approx(n_si**2)
    # base index map + mask: no perturbation outside the mask
    base = np.full((ny, nz), 1.444)
    base[:, 1:4] = n_si
    gm = eo.carrier_index_grid(lo, hi, dN, dP, wavelength=lam, axes=(1, 2), n_base=base, mask=base > 2)
    em = np.array(gm["eps"]).reshape(ny, nz)
    assert em[0, 0] == pytest.approx(1.444**2) and em[0, 2] == pytest.approx((n_si - 8.8e-4) ** 2)

    # loss: uniform 1e18 / 1e18 -> 14.5 /cm, carried by the medium's conductivity
    med = eo.carrier_medium("si_doped", np.full((ny, nz), 1e18), np.full((ny, nz), 1e18), wavelength=lam)
    assert med["eps"][0] == pytest.approx(n_si**2)
    assert med["sigma"][0] == pytest.approx(n_si * 1450.0 * eo.EPS0 * ls.C0)
    # ... and seen by the core: Im(eps) at the carrier frequency gives back alpha
    sim = cross_section_sim()
    full = eo.carrier_index_grid(lo, hi, np.full((ny, nz), 1e18), np.full((ny, nz), 1e18), wavelength=lam, axes=(1, 2))
    sim.add_structure(full, med)
    info = plane(sim, ls.C0 / lam)
    iu = int(np.argmin(np.abs(info["u_nodes"])))
    iv = int(np.argmin(np.abs(info["v_nodes"])))
    eps_c = complex(info["eps_n"][iu, iv], info["eps_n_im"][iu, iv])
    assert eps_c.real == pytest.approx((n_si - 3.015e-3) ** 2, rel=1e-5)
    alpha = 2 * (2 * math.pi / lam) * np.sqrt(eps_c).imag  # 1/m
    assert alpha / 100 == pytest.approx(14.5, rel=1e-3)
