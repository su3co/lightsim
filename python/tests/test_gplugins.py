"""lightsim.gplugins: the gplugins.gmeep-compatible front.

Runs without Meep and without gplugins; the checks that compare with
gplugins' own helpers are skipped when gplugins is not installed.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

gf = pytest.importorskip("gdsfactory")

import lightsim.gplugins as lg  # noqa: E402
from lightsim import Simulation  # noqa: E402

gf.gpdk.PDK.activate()

FAST = dict(resolution=12, wavelength_points=3, device="cpu", effective_index=False)

# write_sparameters_meep's parameters (gplugins 2.1.5)
MEEP_PARAMETERS = [
    "component", "port_source_names", "port_source_modes", "port_modes", "port_symmetries", "resolution",
    "wavelength_start", "wavelength_stop", "wavelength_points", "dirpath", "layer_stack", "port_margin",
    "port_monitor_offset", "port_source_offset", "filepath", "overwrite", "animate", "animate_center", "animate_size",
    "lazy_parallelism", "run", "dispersive", "xmargin", "ymargin", "zmargin", "xmargin_left", "xmargin_right",
    "ymargin_top", "ymargin_bot", "zmargin_top", "zmargin_bot", "decay_by", "is_3d", "z", "plot_args",
    "only_return_filepath_sim_settings", "verbosity", "padding_layers", "settings",
]  # fmt: skip


@pytest.fixture(scope="module")
def straight():
    return gf.components.straight(length=2)


def test_same_parameters_and_defaults_as_meep():
    sig = inspect.signature(lg.write_sparameters_lightsim)
    params = list(sig.parameters)
    assert params[: len(MEEP_PARAMETERS) - 1] == MEEP_PARAMETERS[:-1]
    d = {k: v.default for k, v in sig.parameters.items()}
    assert (d["resolution"], d["wavelength_start"], d["wavelength_stop"], d["wavelength_points"]) == (30, 1.5, 1.6, 50)
    assert (d["port_margin"], d["port_monitor_offset"], d["port_source_offset"], d["decay_by"], d["is_3d"]) == (2, -0.1, -0.1, 1e-3, False)
    assert lg.write_sparameters_meep is lg.write_sparameters_lightsim
    assert lg.port_symmetries.port_symmetries_1x1 == {"o2@0,o1@0": ["o1@0,o2@0"]}


def test_result_format_and_cache(straight, tmp_path, monkeypatch):
    sp = lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, **FAST)
    assert sorted(sp) == ["o1@0,o1@0", "o1@0,o2@0", "o2@0,o1@0", "o2@0,o2@0", "wavelengths"]
    np.testing.assert_allclose(sp["wavelengths"], [1.5, 1.55, 1.6])  # micrometres
    assert all(np.iscomplexobj(v) and v.shape == (3,) for k, v in sp.items() if k != "wavelengths")
    assert np.all(np.abs(sp["o2@0,o1@0"]) > 0.95)
    np.testing.assert_allclose(sp["o2@0,o1@0"], sp["o1@0,o2@0"], atol=1e-2)  # reciprocity
    yml = lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, only_return_filepath_sim_settings=True, **FAST)
    npz = yml.with_suffix(".npz")
    assert npz.exists() and yml.exists()
    assert sorted(np.load(npz)) == sorted(sp)  # only S-parameters and wavelengths: what gplugins.sax reads

    def no_run(*a, **k):
        raise AssertionError("ran although the result is cached")

    monkeypatch.setattr(Simulation, "run", no_run)
    again = lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, **FAST)
    np.testing.assert_array_equal(again["o2@0,o1@0"], sp["o2@0,o1@0"])
    with pytest.raises(AssertionError):
        lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, overwrite=True, **FAST)


def test_path_depends_on_settings(straight, tmp_path):
    kw = dict(dirpath=tmp_path, only_return_filepath_sim_settings=True)
    a = lg.write_sparameters_lightsim(straight, **kw)
    assert a == lg.write_sparameters_lightsim(straight, device="cuda:1", **kw)  # where it runs does not matter
    assert a != lg.write_sparameters_lightsim(straight, resolution=20, **kw)
    assert a != lg.write_sparameters_lightsim(straight, is_3d=True, **kw)
    assert a != lg.write_sparameters_lightsim(gf.components.straight(length=3), **kw)


def test_builtin_path_matches_gplugins(straight, tmp_path):
    g = pytest.importorskip("gplugins.common.utils.get_sparameters_path")
    settings = dict(resolution=30, port_modes=[0], layer_stack=gf.get_active_pdk().layer_stack.to_dict())
    ours = lg._builtin_sparameters_path(straight, tmp_path, tool="lightsim", **settings)
    assert ours == g._get_sparameters_path(component=straight, dirpath=tmp_path, tool="lightsim", **settings)


def test_ignored_arguments_warn_once(straight, tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, animate=True, lazy_parallelism=True, eps_averaging=False, **FAST)
    msgs = [str(x.message) for x in w if issubclass(x.category, lg.IgnoredArgumentWarning)]
    assert len(msgs) == 1 and "animate" in msgs[0] and "eps_averaging" in msgs[0] and "lazy_parallelism" in msgs[0]
    with pytest.raises(ValueError, match="not in the settings"):
        lg.write_sparameters_lightsim(straight, no_such_setting=1)


def test_run_false_returns_simulation(straight):
    sim = lg.write_sparameters_lightsim(straight, ymargin=2, run=False, resolution=12)
    assert isinstance(sim, Simulation) and sim.dimensions == 2
    assert len(sim.sources) == 1 and len(sim.monitors) == 2
    sim3 = lg.get_simulation(straight, port_source_name="o2", ymargin=2, resolution=12, is_3d=True)
    assert sim3.dimensions == 3 and sim3.sources[0]["name"] == "src_o2"


def test_source_plan_uses_symmetries():
    names = ["o1", "o2"]
    sources = [("o1", 0), ("o2", 0)]
    # gplugins' 1x1 map gives o1@0,o2@0 but not the reflection o2@0,o2@0: both sources still run
    assert lg._source_plan(names, sources, lg.port_symmetries_1x1, [0]) == sources
    full = {"o2@0,o1@0": ["o1@0,o2@0"], "o1@0,o1@0": ["o2@0,o2@0"]}
    assert lg._source_plan(names, sources, full, [0]) == [("o1", 0)]
    assert lg._source_plan(names, sources, {}, [0]) == sources


def test_symmetric_copies(tmp_path):
    full = {"o2@0,o1@0": ["o1@0,o2@0"], "o1@0,o1@0": ["o2@0,o2@0"]}
    sp = lg.write_sparameters_lightsim(gf.components.straight(length=2), ymargin=2, dirpath=tmp_path, port_symmetries=full, **FAST)
    np.testing.assert_array_equal(sp["o1@0,o2@0"], sp["o2@0,o1@0"])
    np.testing.assert_array_equal(sp["o2@0,o2@0"], sp["o1@0,o1@0"])


def test_bend_and_modes(tmp_path):
    c = gf.components.bend_euler(radius=4)
    sp = lg.write_sparameters_lightsim_1x1_bend90(c, dirpath=tmp_path, **FAST)
    assert np.all(np.abs(sp["o2@0,o1@0"]) > 0.8)
    np.testing.assert_allclose(sp["o2@0,o1@0"], sp["o1@0,o2@0"], atol=2e-2)
    wide = gf.components.straight(length=2, width=2)
    sp = lg.write_sparameters_lightsim(
        wide, ymargin=2, dirpath=tmp_path, port_source_names=["o1"], port_source_modes={"o1": [0, 1]}, port_modes=[0, 1], **FAST
    )
    assert {"o2@1,o1@1", "o2@0,o1@1", "o2@1,o1@0"} <= set(sp)
    assert np.all(np.abs(sp["o2@1,o1@1"]) > 0.9) and np.all(np.abs(sp["o2@0,o1@1"]) < 0.1)


def test_effective_index_and_3d(straight, tmp_path):
    eim = lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, resolution=12, wavelength_points=3, device="cpu")
    assert np.all(np.abs(eim["o2@0,o1@0"]) > 0.95)
    sp3 = lg.write_sparameters_lightsim(straight, ymargin=1, dirpath=tmp_path, resolution=10, wavelength_points=3, device="cpu", is_3d=True, port_source_names=["o1"])
    assert np.all(np.abs(sp3["o2@0,o1@0"]) > 0.9)


def test_batch_and_mpi(tmp_path):
    jobs = [{"component": gf.components.straight(length=L), "ymargin": 2} for L in (1, 2)]
    paths = lg.write_sparameters_lightsim_batch(jobs, devices=["cpu", "cpu"], dirpath=tmp_path, **FAST)
    assert [p.exists() for p in paths] == [True, True] and paths[0] != paths[1]
    p = lg.write_sparameters_lightsim_mpi(gf.components.straight(length=1), dirpath=tmp_path, ymargin=2, **FAST)
    assert p == paths[0]  # same settings, same file


def test_waveguide_modes():
    wg = lg.Waveguide(wavelength=1.55, core_width=0.5, core_thickness=0.22, core_material=3.476, clad_material=1.444, num_modes=2, grid_resolution=15)
    n = wg.n_eff.real
    assert 2.3 < n[0] < 2.55 and 1.6 < n[1] < 1.9  # TE0, TM0 of a silicon strip
    assert wg.fraction_te[0] > 0.9 and wg.fraction_te[1] < 0.2
    assert 0.05 < wg.mode_area[0] < 0.4
    sweep = lg.sweep_n_eff(wg, core_width=[0.45, 0.55])
    assert sweep.shape == (2, 2) and sweep[1, 0].real > sweep[0, 0].real


def test_grating_coupler(tmp_path):
    sp = lg.write_sparameters_grating_lightsim(resolution=16, wavelength_points=5, dirpath=tmp_path, device="cpu")
    t = np.abs(sp["o2@0,o1@0"]) ** 2
    assert sorted(sp) == ["o1@0,o1@0", "o1@0,o2@0", "o2@0,o1@0", "o2@0,o2@0", "wavelengths"]
    assert np.all(np.isfinite(t)) and 0.05 < t.max() < 0.8


def test_sax_model(straight, tmp_path):
    pytest.importorskip("jax")
    lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, **FAST)
    npz = lg.write_sparameters_lightsim(straight, ymargin=2, dirpath=tmp_path, only_return_filepath_sim_settings=True, **FAST).with_suffix(".npz")
    s = lg.model_from_npz(npz)(wl=np.array([1.55]))
    assert ("o2", "o1") in s and float(np.abs(np.asarray(s[("o2", "o1")])[0])) > 0.95
