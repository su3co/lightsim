"""Complex (Hermitian) permittivity tensors given directly
(``Medium(eps_complex=...)``): Faraday rotation per length in a half-space
against (k0 / 2)(n_- - n_+), n_+- = sqrt(eps -+ g) (continuous; the Rust case
`hermitian_tensor` includes the Yee dispersion); the refusals."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.adjoint import check_gradient_media

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1e-6
F0 = ls.C0 / LAM


def faraday(e, g):
    return [[e, 0, 0], [0, e, 1j * g], [0, -1j * g, e]]


def test_faraday_rotation():
    e, g = 2.25, 0.3
    dx = LAM / 64
    sim = ls.Simulation(min=[0, 0, 0], max=[6e-6, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=110e-15)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_structure(ls.Box([2.5e-6, -1, -1], [20e-6, 1, 1]), ls.Medium("mo", eps_complex=faraday(e, g)))
    sim.add_source(ls.PlaneWave(0, 1e-6, "+", "Ey", ls.GaussianPulse(F0, 0.15 * F0)))
    for name, x in (("a", 3.2e-6), ("b", 4.0e-6)):
        sim.add_monitor(ls.DFTMonitor(name, [x, 0, 0], [x, 0, 0], [F0], components=("Ey", "Ez")))
    r = sim.run(device=DEVICE)
    xs = ls.Simulation.grid(sim)[0]

    def ang(m):
        ey, ez = (np.asarray(r.field(m, c)).ravel()[0] for c in ("Ey", "Ez"))
        return 0.5 * np.arctan2(2 * (ey * np.conj(ez)).real, abs(ey) ** 2 - abs(ez) ** 2)

    ia, ib = (int(np.argmin(np.abs(xs - x))) for x in (3.2e-6, 4.0e-6))
    dist = xs[ib] - xs[ia]
    k0 = 2 * np.pi / LAM
    expect = 0.5 * k0 * (np.sqrt(e + g) - np.sqrt(e - g))
    rot = (ang("b") - ang("a")) / dist
    print(f"rotation per length {rot:.6e} rad/m, analytic {expect:.6e} (ratio {rot / expect:.5f})")
    # (the Yee dispersion at 40 cells per wavelength in the medium adds 0.26 %)
    assert abs(rot / expect - 1) < 4e-3


def test_refusals():
    with pytest.raises(ValueError, match="Hermitian"):
        ls.Medium("x", eps_complex=[[2, 1j, 0], [1j, 2, 0], [0, 0, 2]])
    with pytest.raises(ValueError, match="positive definite"):
        ls.Medium("x", eps_complex=[[1, 2j, 0], [-2j, 1, 0], [0, 0, 1]])
    sim = ls.Simulation(min=[0, 0, 0], max=[2e-6, 0.1e-6, 0.1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim.add_structure(ls.Box([0.5e-6, -1, -1], [1e-6, 1, 1]), ls.Medium("mo", eps_complex=faraday(2.25, 0.3)))
    with pytest.raises(ValueError, match="Hermitian"):
        check_gradient_media(sim)
    sim2 = ls.Simulation(min=[0, 0], max=[2e-6, 1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim2.add_structure(ls.Box([0.5e-6, -1], [1e-6, 1]), ls.Medium("mo", eps_complex=faraday(2.25, 0.3)))
    with pytest.raises(RuntimeError, match="2-D"):
        sim2.run(device="cpu")
