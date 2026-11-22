"""Material models with their own state: the Raman (delayed) chi3 response.

The physics is validated by the Rust cases (``cargo test -p lightsim-core``,
``lightsim validate``); these tests check the Python interface: the JSON the
builders produce, a run through the API, and that the gradient paths refuse
the media they do not model.

Run on a GPU with LIGHTSIM_DEVICE=cuda:0.
"""

import json
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import autodiff, invdes
from lightsim.adjoint import check_gradient_media
from lightsim.spec import C0, Box, GaussianPulse, Medium, PointDipole, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = C0 / (1.0 * um)
W0 = 2 * np.pi * F0


def _column(medium, amplitude=1.0, run_time=40e-15):
    dx = 1.0 * um / 40
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=run_time)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    if medium is not None:
        sim.add_structure(Box([1.5 * um, -1, -1], [2.5 * um, 1, 1]), medium)
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", GaussianPulse(F0, 0.3 * F0), amplitude=amplitude))
    sim.add_monitor(ls.ProbeMonitor("p", [3.5 * um, 0, 0], components=["Ey"]))
    return sim


def _probe(sim):
    return np.asarray(sim.run(device=DEVICE).time_series("p", "Ey")[1]).ravel()


def test_raman_medium_json_and_run():
    r = ls.Raman(0.18, 0.1 * W0, 0.03 * W0)
    assert r == {"fraction": 0.18, "omega": 0.1 * W0, "delta": 0.03 * W0}
    m = Medium("silica_like", n=1.0, chi3=2e-3, raman=r)
    sim = _column(m)
    spec = json.loads(sim.to_json())
    assert spec["media"][0]["raman"] == r
    # no Raman key when the response is instantaneous
    assert "raman" not in Medium("kerr", chi3=1e-3)
    lin = _probe(_column(None))
    inst = _probe(_column(Medium("kerr", n=1.0, chi3=2e-3)))
    ram = _probe(sim)
    d_inst, d_ram = np.abs(inst - lin).max(), np.abs(ram - lin).max()
    print(f"generated field: instantaneous {d_inst:.3e}, with Raman {d_ram:.3e} (peak {np.abs(lin).max():.3e})")
    assert np.all(np.isfinite(ram)) and 0 < d_ram < d_inst


def test_raman_parameters_are_checked():
    with pytest.raises(Exception, match="Raman"):
        _column(Medium("bad", chi3=1e-3, raman=ls.Raman(1.5, W0, 0.1 * W0))).run(device="cpu")
    with pytest.raises(Exception, match="chi3"):
        _column(Medium("bad", raman=ls.Raman(0.2, W0, 0.1 * W0))).run(device="cpu")


def test_gradient_paths_refuse_raman_media():
    m = Medium("ram", n=1.5, chi3=1e-20, raman=ls.Raman(0.2, 0.1 * W0, 0.03 * W0))
    sim = _column(m)
    with pytest.raises(ValueError, match="adjoint gradients do not support"):
        check_gradient_media(sim)
    # an unused medium does not matter
    sim2 = _column(None)
    sim2.add_medium(m)
    check_gradient_media(sim2)
    # reverse mode (native engine): the plan compiler refuses the scene
    dx = 50e-9
    sim = ls.Simulation(min=[0, 0, 0], max=[3.2 * um, 2.0 * um, dx], mesh=ls.UniformMesh(dx), run_time=30e-15, background="ox",
                        boundaries={"x": [ls.PML(), ls.PML()], "y": [ls.PML(), ls.PML()], "z": [ls.PERIODIC, ls.PERIODIC]})
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_structure(Box([-1, 0.8 * um, -1], [10 * um, 1.2 * um, 1]), m)
    sim.add_source(PointDipole([0.5 * um, 1.0 * um, 0.0], "Ez", GaussianPulse(F0, 0.25 * F0), amplitude=1e-15))
    grid = invdes.DesignGrid([1.2 * um, 0.6 * um, -1], [2.0 * um, 1.4 * um, 1], (7, 9), Medium("design", eps=2.6**2))
    obj = autodiff.flux_energy(0, 2.7 * um, [2.7 * um, 0.4 * um, 0.0], [2.7 * um, 1.6 * um, dx])
    with pytest.raises(Exception, match="gradient engines do not support"):
        autodiff.value_and_grad(sim, grid, np.full(grid.dims, 4.0), obj, device=DEVICE)
