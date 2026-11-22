"""Far-field monitor against the analytic pattern of a dipole."""

import os

import numpy as np

import lightsim as ls
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def test_far_field_of_a_dipole():
    """An electric dipole of current moment p along z radiates
    U = eta0 k^2 |p|^2 sin^2(theta) / (32 pi^2) and
    E_theta = -i k eta0 p sin(theta) exp(i k r) / (4 pi r), E_phi = 0."""
    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 24
    eta0 = 376.730313412
    moment = 1e-12  # A m
    theta = np.linspace(0, np.pi, 37)
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    sim = ls.Simulation(min=[-0.8 * um] * 3, max=[0.8 * um] * 3, mesh=ls.UniformMesh(dx), run_time=40e-15)
    sim.add_source(ls.PointDipole([0, 0, 0], "Ez", ls.GaussianPulse(f0, 0.3 * f0), amplitude=moment, name="dipole"))
    sim.add_monitor(ls.FarFieldMonitor("far", [-0.45 * um] * 3, [0.45 * um] * 3, [f0], theta, phi))
    res = sim.run(device=DEVICE)
    far = res.far_field("far")
    assert far["intensity"].shape == far["e_theta"].shape == far["e_phi"].shape == (1, 37, 8)
    assert np.allclose(far["theta"], theta) and np.allclose(far["phi"], phi) and np.allclose(far["freqs"], [f0])

    _, S = res.source_spectrum("dipole", monitor="far")
    k = 2 * np.pi * f0 / ls.C0
    p = moment * S[0]
    U = far["intensity"][0]
    U_exact = eta0 * k**2 * abs(p) ** 2 * np.sin(theta) ** 2 / (32 * np.pi**2)
    # pattern and absolute level, the same at every azimuth
    assert np.max(np.abs(U - U_exact[:, None])) < 0.02 * U_exact.max()
    # U from the complex amplitudes; the dipole has no E_phi
    assert np.allclose((np.abs(far["e_theta"]) ** 2 + np.abs(far["e_phi"]) ** 2)[0] / (2 * eta0), U, rtol=1e-9, atol=1e-12 * U.max())
    assert np.abs(far["e_phi"]).max() < 0.01 * np.abs(far["e_theta"]).max()
    # amplitude and phase of E_theta. The dipole sits on the Ez position of
    # the grid, half a cell above the origin, which shifts the far-field phase.
    z0 = 0.5 * dx
    e_exact = -1j * k * eta0 * p * np.sin(theta) / (4 * np.pi) * np.exp(-1j * k * z0 * np.cos(theta))
    mid = slice(6, 31)  # away from the poles, where the field vanishes
    ratio = far["e_theta"][0, mid, 0] / e_exact[mid]
    print("E_theta / exact: magnitude", abs(ratio).min(), abs(ratio).max(), "phase", np.angle(ratio).min(), np.angle(ratio).max())
    assert np.max(np.abs(ratio - 1)) < 0.03
    # total power: the integral of U over the sphere, the box flux and the Hertz formula
    ring = U.mean(axis=1) * np.sin(theta)
    total = 2 * np.pi * np.sum(0.5 * (ring[1:] + ring[:-1]) * np.diff(theta))
    assert abs(total / far["flux"][0] - 1) < 0.02
    assert abs(total / (eta0 * k**2 * abs(p) ** 2 / (12 * np.pi)) - 1) < 0.02
    assert np.allclose(res.flux("far"), far["flux"])
