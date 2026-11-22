"""2.5-D effective-index module: slab solver vs the analytic dispersion
relation, a straight waveguide, and a gdsfactory MMI 1x2 against the full
3-D result (examples/gds_mmi_sparams.py).

Set LIGHTSIM_EIM_3D=1 to also run the 3-D reference here (minutes on CPU)
instead of comparing with the recorded number.
"""

import os
import time

import numpy as np
import pytest
from scipy.optimize import brentq

from lightsim import eim

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1.55e-6
N_SI, N_OX = 3.476, 1.444
# |S21|^2 = |S31|^2 of gdsfactory mmi1x2 (generic PDK, 220 nm Si in SiO2) at
# 1.55 um from the full 3-D lightsim run (examples/gds_mmi_sparams.py).
MMI_3D_T = 0.49


def analytic_symmetric_slab(n1, n2, t, lam, tm=False):
    """Fundamental (even) mode: kappa tan(kappa t / 2) = r gamma, r = 1 (TE)
    or n1^2 / n2^2 (TM)."""
    k0 = 2 * np.pi / lam
    v2 = k0**2 * (n1**2 - n2**2)
    r = (n1 / n2) ** 2 if tm else 1.0
    f = lambda k: k * np.tan(k * t / 2) - r * np.sqrt(v2 - k * k)  # noqa: E731
    kmax = min(np.pi / t, np.sqrt(v2)) * (1 - 1e-12)
    kap = brentq(f, 1e-3, kmax, xtol=1e-6, rtol=1e-15)
    return np.sqrt(n1**2 - (kap / k0) ** 2)


@pytest.mark.parametrize("t", [90e-9, 150e-9, 220e-9, 400e-9])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_slab_neff_matches_analytic(t, pol):
    ref = analytic_symmetric_slab(N_SI, N_OX, t, LAM, tm=pol == "tm")
    got = eim.slab_neff(N_SI, t, LAM, N_OX, polarization=pol)
    print(f"{pol} t={t * 1e9:.0f} nm: n_eff {got:.6f}  analytic {ref:.6f}")
    assert abs(got - ref) < 1e-4


def test_slab_multilayer_consistency():
    # splitting the core into sub-layers, or adding a far, thick cladding
    # layer of the same material as the half-space, must not change n_eff
    a = eim.slab_modes([N_OX, N_SI, 1.0], [220e-9], LAM)
    b = eim.slab_modes([N_OX, N_OX, N_SI, N_SI, 1.0], [1e-6, 100e-9, 120e-9], LAM)
    assert len(a) == len(b) == 1 and abs(a[0] - b[0]) < 1e-9
    # asymmetric slab is below the symmetric oxide-clad one; a thick slab is multimode
    assert a[0] < eim.slab_neff(N_SI, 220e-9, LAM, N_OX)
    m = eim.slab_modes([N_OX, N_SI, N_OX], [600e-9], LAM)
    assert len(m) == 3 and m == sorted(m, reverse=True)
    with pytest.raises(ValueError):
        eim.slab_neff(N_SI, 220e-9, LAM, N_OX, mode=1)


gf = pytest.importorskip("gdsfactory")


@pytest.fixture(scope="module")
def pdk():
    gf.gpdk.PDK.activate()


def test_layer_stack_effective_indices(pdk):
    c = gf.components.straight(length=3, cross_section="rib")
    neff, n_bg, _ = eim.effective_indices(c, LAM)
    print("rib stack:", neff, n_bg)
    assert abs(neff["core"] - eim.slab_neff(N_SI, 220e-9, LAM, N_OX)) < 1e-9
    assert abs(neff["slab90"] - eim.slab_neff(N_SI, 90e-9, LAM, N_OX)) < 1e-9
    assert n_bg == N_OX
    sim, _, info = eim.effective_index_simulation(c, [LAM])
    # the rib (higher n_eff) is drawn after, i.e. on top of, its slab
    names = [s["medium"] for s in sim.structures]
    assert names.index("eim_slab90") < names.index("eim_core")
    assert len(sim.grid()[2]) == 2  # one periodic cell along z


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_straight_waveguide_transmission(pdk, pol):
    c = gf.components.straight(length=4)
    wl = np.array([1.5e-6, 1.55e-6, 1.6e-6])
    sp = eim.write_sparameters_2d(c, wavelengths=wl, polarization=pol, steps_per_wavelength=16, device=DEVICE, input_ports=["o1"])
    t = np.abs(sp["o2@0,o1@0"]) ** 2
    r = np.abs(sp["o1@0,o1@0"]) ** 2
    # port mode = slab mode of the lateral (n_bg / n_slab / n_bg) guide:
    # in-plane E is normal to the side walls -> the lateral problem is TM
    n_slab = sp["slab_neff"]["core"]
    ref = eim.slab_neff(n_slab, 0.5e-6, LAM, sp["n_background"], polarization="tm" if pol == "te" else "te")
    got = sp["mode_neff"]["o1"].real
    print(f"{pol}: |S21|^2 {t}, |S11|^2 {r}, port n_eff {got:.4f} (analytic lateral slab {ref:.4f})")
    assert np.all(np.abs(t - 1) < 0.01)
    assert np.all(r < 1e-3)
    assert abs(got - ref) < 0.02


def test_mmi1x2_matches_3d(pdk):
    c = gf.components.mmi1x2()
    wl = np.array([1.5e-6, 1.55e-6, 1.6e-6])
    spw = 14
    t0 = time.time()
    sp, res = eim.write_sparameters_2d(c, wavelengths=wl, steps_per_wavelength=spw, device=DEVICE, input_ports=["o1"], return_results=True)
    t_2d = time.time() - t0
    s21 = np.abs(sp["o2@0,o1@0"]) ** 2
    s31 = np.abs(sp["o3@0,o1@0"]) ** 2
    s11 = np.abs(sp["o1@0,o1@0"]) ** 2
    print(f"2.5-D MMI 1x2 ({res['o1']}, {t_2d:.1f} s): |S21|^2 {s21}, |S31|^2 {s31}, |S11|^2 {s11}")
    ref21 = ref31 = MMI_3D_T
    if os.environ.get("LIGHTSIM_EIM_3D"):
        from lightsim.sparams import write_sparameters

        t0 = time.time()
        sp3 = write_sparameters(c, wavelengths=wl, steps_per_wavelength=spw, device=DEVICE, input_ports=["o1"])
        t_3d = time.time() - t0
        ref21 = (np.abs(sp3["o2@0,o1@0"]) ** 2)[1]
        ref31 = (np.abs(sp3["o3@0,o1@0"]) ** 2)[1]
        print(f"3-D: |S21|^2 {ref21:.4f}, |S31|^2 {ref31:.4f} in {t_3d:.1f} s -> speed-up {t_3d / t_2d:.0f}x")
    assert abs(s21[1] / ref21 - 1) < 0.07
    assert abs(s31[1] / ref31 - 1) < 0.07
    assert abs(s21[1] / s31[1] - 1) < 0.02  # symmetric split
    assert np.all(s21 + s31 + s11 <= 1.005)
    assert np.all(s11 < 0.01)


def test_symmetric_coupler_bar_and_cross_are_in_quadrature():
    """A lossless symmetric coupler has arg(S_cross / S_bar) = +-90 degrees.
    With a mesh that discretised the two arms differently this was off by
    14 degrees for this coupler."""
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    c = gf.components.coupler(gap=0.3, length=12.0)
    sp = eim.write_sparameters_2d(c, wavelengths=[1.55e-6], steps_per_wavelength=14, input_ports=["o1"], device=DEVICE)
    centres = {p.name: (float(p.center[0]), float(p.center[1])) for p in c.ports}
    right = [n for n, (x, _) in centres.items() if x > 0]
    bar = next(n for n in right if np.sign(centres[n][1]) == np.sign(centres["o1"][1]))
    cross = next(n for n in right if n != bar)
    s_bar, s_cross = sp[f"{bar}@0,o1@0"][0], sp[f"{cross}@0,o1@0"][0]
    phase = np.degrees(np.angle(s_cross / s_bar))
    print(f"|bar|^2 {abs(s_bar) ** 2:.4f} |cross|^2 {abs(s_cross) ** 2:.4f} arg(cross/bar) {phase:+.2f} deg")
    assert abs(abs(phase) - 90.0) < 0.5
    assert abs(abs(s_bar) ** 2 + abs(s_cross) ** 2 - 1) < 0.01


def test_port_symmetry_reuses_runs_without_changing_the_s_matrix():
    """A mirror-symmetric device needs fewer runs: ports that are images of
    each other are excited once and the other columns are copied. The full
    S-matrix is the one that running every port gives."""
    gf = pytest.importorskip("gdsfactory")
    from lightsim.sparams import _symmetry_plan, detect_port_symmetries

    gf.gpdk.PDK.activate()
    wl = np.linspace(1.5e-6, 1.6e-6, 3)
    mmi = gf.components.mmi1x2()
    coupler = gf.components.coupler(gap=0.3, length=6.0)
    assert detect_port_symmetries(mmi) == [{"o1": "o1", "o2": "o3", "o3": "o2"}]
    assert len(detect_port_symmetries(coupler)) == 2  # two mirrors: all four ports are images of one
    assert detect_port_symmetries(gf.components.bend_euler(radius=5)) == []
    assert detect_port_symmetries(gf.components.taper(length=5, width1=0.5, width2=1.0)) == []

    for component, runs, tol in ((mmi, 2, 2e-3), (coupler, 1, 1e-5)):
        full, r_full = eim.write_sparameters_2d(component, wavelengths=wl, steps_per_wavelength=12, device=DEVICE, return_results=True)
        sym, r_sym = eim.write_sparameters_2d(component, wavelengths=wl, steps_per_wavelength=12, device=DEVICE, return_results=True, port_symmetry="auto")
        keys = sorted(k for k in full if "@" in k)
        assert sorted(k for k in sym if "@" in k) == keys and len(keys) == len(r_full) ** 2
        err = max(np.max(np.abs(full[k] - sym[k])) for k in keys)
        print(f"{component.name}: {len(r_full)} runs -> {len(r_sym)}, max |dS| = {err:.1e}")
        assert len(r_sym) == runs < len(r_full)
        assert err < tol

    # the same stated by hand, and for a subset of the inputs
    by_hand = eim.write_sparameters_2d(mmi, wavelengths=wl, steps_per_wavelength=12, device=DEVICE, port_symmetry={"o2": "o3"}, input_ports=["o2", "o3"])
    assert sorted(k for k in by_hand if "@" in k) == sorted(f"{j}@0,{i}@0" for i in ("o2", "o3") for j in ("o1", "o2", "o3"))
    assert np.allclose(by_hand["o1@0,o3@0"], by_hand["o1@0,o2@0"]) and np.allclose(by_hand["o3@0,o3@0"], by_hand["o2@0,o2@0"])
    assert np.allclose(by_hand["o2@0,o3@0"], by_hand["o3@0,o2@0"])
    # maps that are not a pairing of existing ports are refused
    with pytest.raises(ValueError, match="names port"):
        _symmetry_plan(["o1", "o2"], [{"o1": "o9"}], ["o1", "o2"])
    with pytest.raises(ValueError, match="not a pairing"):
        _symmetry_plan(["o1", "o2", "o3"], [{"o1": "o2", "o2": "o3", "o3": "o1"}], ["o1"])
    with pytest.raises(ValueError, match="port_symmetry must be"):
        eim.write_sparameters_2d(mmi, wavelengths=wl, port_symmetry="mirror")
