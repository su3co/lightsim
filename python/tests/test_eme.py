"""Eigenmode expansion: exact 1-D references, junction properties, a taper
from gdsfactory and the gplugins.meow-style interface.

The 3-D FDTD comparisons (taper, MMI, coupler, Bragg grating) need a GPU
and are recorded in docs/modules/eme.md, not run here.
"""

import numpy as np
import pytest

from lightsim.eme import EME, SMatrix, _ModeSet, interface, power, stack_transfer_matrix, star
from lightsim.xsection import CrossSection

LAM = 1.55e-6
K0 = 2 * np.pi / LAM


class Uniform(CrossSection):
    """A homogeneous cross-section, periodic in both directions: its two
    modes are the plane waves of the medium."""

    def __init__(self, n):
        super().__init__(background=n)
        self.set_mesh([0.0, 1e-7], [0.0, 1e-7])

    def solve(self, *a, **k):
        k["periodic"] = (True, True)
        return super().solve(*a, **k)


def test_multilayer_stack_against_transfer_matrix():
    ns, ds = [1.444, 3.476, 1.444, 2.0, 1.0], [0.3e-6, 0.15e-6, 0.4e-6]
    secs = [(Uniform(ns[0]), 0.0)] + [(Uniform(n), d) for n, d in zip(ns[1:-1], ds)] + [(Uniform(ns[-1]), 0.0)]
    S = EME(secs, num_modes=2).solve(LAM)
    r, t, rb, tb = stack_transfer_matrix(ns, ds, LAM)
    for got, ref in ((S.s11[0, 0], r), (S.s21[0, 0], t), (S.s22[0, 0], rb), (S.s12[0, 0], tb)):
        assert abs(got - ref) < 1e-13
    assert abs(abs(S.s11[0, 0]) ** 2 + abs(S.s21[0, 0]) ** 2 - 1) < 1e-13


def test_periodic_section_by_repeated_squaring():
    nh, nl = 2.0, 1.444
    dh, dl = LAM / (4 * nh), LAM / (4 * nl)
    N = 37
    secs = [(Uniform(1.444), 0.0), ("period", [(Uniform(nh), dh), (Uniform(nl), dl)], N), (Uniform(1.444), 0.0)]
    S = EME(secs, num_modes=2).solve(LAM)
    r, t, _, _ = stack_transfer_matrix([1.444] + [nh, nl] * N + [1.444], [dh, dl] * N, LAM)
    assert abs(S.s11[0, 0] - r) < 1e-12 and abs(S.s21[0, 0] - t) < 1e-10  # |t| ~ 1e-5: both sides lose digits
    # power() itself: s^5 by squaring equals five products
    a = SMatrix.propagation(np.array([1.0 + 0.1j, 2.0]), 0.3)
    b = interface(*[_ModeSet.from_modes(Uniform(n).solve(LAM, 2, profile=False), K0) for n in (1.5, 2.5)])
    u = star(a, b)
    five = star(star(star(star(u, u), u), u), u)
    assert np.allclose(power(u, 5).full(), five.full(), atol=1e-13)


def _strip(w):
    xs = CrossSection(background=1.444)
    xs.add_rect(-w / 2, 0, w / 2, 0.22e-6, 3.476)
    xs.set_mesh(np.arange(-1.5e-6, 1.5e-6 + 1e-12, 25e-9), np.arange(-0.6e-6, 0.82e-6 + 1e-12, 25e-9))
    return xs


def test_junction_conserves_power_and_is_reciprocal():
    a = _ModeSet.from_modes(_strip(0.5e-6).solve(LAM, 6, profile=False), K0)
    b = _ModeSet.from_modes(_strip(0.65e-6).solve(LAM, 6, profile=False), K0)
    J = interface(a, b)
    # all six modes propagate here: every column of the junction (light
    # from one side in one mode) carries unit power out
    S = J.full()
    assert np.allclose(np.sum(np.abs(S) ** 2, axis=0), 1.0, atol=1e-10)
    # reciprocity holds to the truncation of the mode sets: 1e-4 for the
    # fundamental mode, 1e-2 between the highest kept modes
    assert abs(J.s21[0, 0] - J.s12[0, 0]) < 1e-3 and np.abs(J.s21 - J.s12.T).max() < 2e-2
    assert 0.99 < abs(J.s21[0, 0]) ** 2 < 1.0 and abs(J.s11[0, 0]) ** 2 < 1e-3


def test_identical_sections_need_one_solve_and_no_junction():
    xs1, xs2 = _strip(0.5e-6), _strip(0.5e-6)
    e = EME([(xs1, 1e-6), (xs2, 2e-6)], num_modes=4)
    S = e.solve(LAM)
    assert e.unique_cross_sections == 1
    beta = e.left.beta
    assert np.allclose(np.diag(S.s21), np.exp(1j * beta * 3e-6)) and np.allclose(S.s11, 0)


def test_gdsfactory_taper_converges_in_cells_and_modes():
    pytest.importorskip("gdsfactory")
    import gdsfactory as gf

    from lightsim import eme

    gf.gpdk.PDK.activate()
    c = gf.components.taper(length=4, width1=0.5, width2=1.0)
    mi = {"si": 3.476, "sio2": 1.444}
    res = {}
    for cl, nm in ((0.5e-6, 6), (0.25e-6, 6), (0.25e-6, 12)):
        sp = eme.sparameters(c, cell_length=cl, num_modes=nm, material_index=mi, step=40e-9, margin=0.8e-6, v_margin=0.6e-6)
        res[(cl, nm)] = abs(sp["o2@0,o1@0"][0]) ** 2
        assert abs(sp["o1@0,o1@0"][0]) ** 2 < 1e-3
    print(res)
    assert all(0.98 < v <= 1.0 for v in res.values())
    assert abs(res[(0.25e-6, 6)] - res[(0.5e-6, 6)]) < 2e-3
    assert abs(res[(0.25e-6, 12)] - res[(0.25e-6, 6)]) < 2e-3


def test_meow_style_interface():
    pytest.importorskip("gdsfactory")
    import gdsfactory as gf

    from lightsim.eme import EMEComponent

    gf.gpdk.PDK.activate()
    c = gf.components.taper(length=3, width1=0.5, width2=0.8)
    sp = EMEComponent(c, num_modes=3, cell_length=0.5, step=0.04, spacing_x=1.6, spacing_y=1.2, material_index={"si": 3.476, "sio2": 1.444}).compute_sparameters()
    assert {"o1@0,o2@0", "o2@2,o1@1", "wavelengths"} <= set(sp)
    assert abs(sp["o1@0,o2@0"]) ** 2 > 0.95 and abs(sp["o1@0,o1@0"]) < 0.05
