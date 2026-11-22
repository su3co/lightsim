"""Sources against the spectrum reported for them: a monitor divided by
``Results.source_spectrum`` gives the response per unit excitation."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
ETA0 = 376.730313412


@pytest.mark.parametrize("angle_deg", [0.0, 20.0, 40.0])
@pytest.mark.parametrize("n", [1.0, 1.5])
def test_plane_wave_transmission_through_empty_space_is_one(angle_deg, n):
    """Normal incidence runs with real fields, oblique incidence with complex
    fields (Bloch boundaries) and the analytic signal; in both the field per
    unit source spectrum is 1 and the power is n |S|^2 A cos(theta) / (2 eta0)."""
    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 40 / n
    theta = np.radians(angle_deg)
    ly, lz = 8 * dx, 2 * dx
    sim = ls.Simulation(min=[0, 0, 0], max=[3 * um, ly, lz], mesh=ls.UniformMesh(dx), run_time=200e-15, background="bg", shutoff=1e-8)
    sim.add_medium(ls.Medium("bg", n=n))
    ky = n * 2 * np.pi * f0 / ls.C0 * np.sin(theta)
    sim.set_boundaries(y=ls.Bloch(ky) if angle_deg else ls.PERIODIC, z=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.8 * um, "+", "Ez", ls.GaussianPulse(f0, 0.05 * f0), angle=theta, name="pw"))
    sim.add_monitor(ls.FluxMonitor("T", 0, 2.2 * um, [2.2 * um, -1, -1], [2.2 * um, 1, 1], [f0]))
    sim.add_monitor(ls.FluxMonitor("R", 0, 0.4 * um, [0.4 * um, -1, -1], [0.4 * um, 1, 1], [f0]))
    sim.add_monitor(ls.DFTMonitor("E", [2.2 * um, 0, 0], [2.2 * um, 0, 0], [f0], components=("Ez",)))
    res = sim.run(device=DEVICE)
    _, spectrum = res.source_spectrum("pw")
    incident = n * abs(spectrum[0]) ** 2 * ly * lz * np.cos(theta) / (2 * ETA0)
    transmission = res.flux("T")[0] / incident
    field = abs(res.field("E", "Ez")[0, 0, 0, 0] / spectrum[0])
    print(f"n = {n}, {angle_deg} deg: T = {transmission:.4f}, |E|/|S| = {field:.4f}, behind the source {res.flux('R')[0] / incident:.1e}")
    assert abs(transmission - 1) < 0.01
    assert abs(field - 1) < 0.01
    assert abs(res.flux("R")[0] / incident) < 2e-3


def test_source_spectrum_lines_up_with_the_monitor():
    """A monitor that lists its frequencies from short to long wavelength
    (descending frequency), next to a monitor with another list: the spectrum
    must be asked for the monitor it normalises."""
    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 40
    freqs = ls.C0 / (np.linspace(0.85, 1.2, 8) * um)  # descending
    area = (2 * dx) ** 2
    sim = ls.Simulation(min=[0, 0, 0], max=[3 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=200e-15, shutoff=1e-8)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.8 * um, "+", "Ez", ls.GaussianPulse(f0, 0.15 * f0), name="pw"))
    sim.add_monitor(ls.FluxMonitor("T", 0, 2.2 * um, [2.2 * um, -1, -1], [2.2 * um, 1, 1], freqs))
    sim.add_monitor(ls.DFTMonitor("probe", [2.2 * um, 0, 0], [2.2 * um, 0, 0], [f0, 1.1 * f0], components=("Ez",)))
    res = sim.run(device=DEVICE)

    # two different lists: no silent default
    with pytest.raises(ValueError, match="different frequency lists"):
        res.source_spectrum("pw")
    f, spectrum = res.source_spectrum("pw", monitor="T")
    assert np.allclose(f, freqs, rtol=1e-12, atol=0)
    T = np.asarray(res.flux("T")) / (np.abs(spectrum) ** 2 * area / (2 * ETA0))
    print("T per frequency", np.round(T, 4))
    assert np.max(np.abs(T - 1)) < 0.01
    # the same values at explicit frequencies, in the order asked for; the name may be left out with one source
    f_rev, s_rev = res.source_spectrum(freqs=freqs[::-1])
    assert np.array_equal(f_rev, freqs[::-1]) and np.allclose(s_rev, spectrum[::-1], rtol=1e-9)
    fp, sp = res.source_spectrum("pw", monitor="probe")
    assert np.allclose(np.abs(res.field("probe", "Ez")[:, 0, 0, 0] / sp), 1.0, atol=0.01)
    # the sorted union of all monitor frequencies (the results file keeps it) does not line up with "T"
    union = np.asarray(res.raw["source_spectra"]["pw"]["freqs"])
    assert len(union) == len(freqs) + 1 and np.all(np.diff(union) > 0)  # 1.1 f0 is new, f0 is in both lists
    misaligned = res.source_spectrum("pw", freqs=union[: len(freqs)])[1]
    wrong = np.asarray(res.flux("T")) / (np.abs(misaligned) ** 2 * area / (2 * ETA0))
    print("with the union grid", np.round(wrong, 3))
    assert np.max(np.abs(wrong - 1)) > 0.3
    with pytest.raises(KeyError):
        res.source_spectrum("nope")


def test_source_spectrum_default_when_all_monitors_agree():
    lam = 1.0 * um
    f0 = ls.C0 / lam
    dx = lam / 30
    freqs = [1.05 * f0, 0.95 * f0, f0]
    sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=100e-15)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.5 * um, "+", "Ez", ls.GaussianPulse(f0, 0.15 * f0)))
    for name, x in (("a", 1.0 * um), ("b", 1.5 * um)):
        sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], freqs))
    res = sim.run(device=DEVICE)
    f, spectrum = res.source_spectrum()  # one unnamed source, one frequency list
    assert np.allclose(f, freqs, rtol=1e-12, atol=0)
    T = np.asarray(res.flux("b")) / (np.abs(spectrum) ** 2 * (2 * dx) ** 2 / (2 * ETA0))
    assert np.max(np.abs(T - 1)) < 0.01


@pytest.mark.parametrize("polarization, direction, theta_deg", [("Ez", "+", 25.0), ("Ey", "+", 25.0), ("Ez", "-", -15.0), ("Ey", "+", 0.0)])
def test_tilted_gaussian_beam_in_two_dimensions(polarization, direction, theta_deg):
    """A sheet beam in the xy plane (one cell along z): it travels at the
    stated angle, carries n |S|^2 w0 sqrt(pi/2) Lz / (2 eta0), and nothing is
    launched behind the source, for E along z (s) and E in the plane (p)."""
    lam = 1.0 * um
    f0 = ls.C0 / lam
    h = lam / 20
    w0 = 2.5 * um
    theta = np.radians(theta_deg)
    s = 1.0 if direction == "+" else -1.0
    xs = 2.0 * um if s > 0 else 10.0 * um
    yc = -1.0 * um
    sim = ls.Simulation(min=[0, -9 * um, 0], max=[12 * um, 9 * um, h], mesh=ls.UniformMesh(h), run_time=160e-15, shutoff=1e-8)
    sim.set_boundaries(z=ls.PERIODIC)
    sim.add_source(ls.GaussianBeam(0, xs, direction, polarization, [xs, yc, 0], w0, ls.GaussianPulse(f0, 0.08 * f0), focus_distance=3 * um, theta=theta, name="beam"))
    comp = "Ez" if polarization == "Ez" else "Ey"
    planes = [xs + s * 2 * um, xs + s * 7 * um]
    for name, x in zip(("near", "far"), planes):
        sim.add_monitor(ls.DFTMonitor(name, [x, -9 * um, 0], [x, 9 * um, 0], [f0], components=(comp,)))
        sim.add_monitor(ls.FluxMonitor("P" + name, 0, x, [x, -20 * um, -1], [x, 20 * um, 1], [f0]))
    xb = xs - s * 1 * um
    sim.add_monitor(ls.FluxMonitor("behind", 0, xb, [xb, -20 * um, -1], [xb, 20 * um, 1], [f0]))
    res = sim.run(device=DEVICE)
    y_nodes = np.asarray(sim.grid()[1])

    def centre(name):
        f = res.raw["dft"][name]["fields"][comp]
        e = res.field(name, comp)[0, 0, :, 0]
        j0 = f["origin"][1]
        y = y_nodes[j0 : j0 + len(e)] if comp == "Ez" else 0.5 * (y_nodes[j0 : j0 + len(e)] + y_nodes[j0 + 1 : j0 + 1 + len(e)])
        w = np.abs(e) ** 2
        return np.sum(w * y) / np.sum(w)

    x_nodes = np.asarray(sim.grid()[0])
    xn = [x_nodes[np.argmin(np.abs(x_nodes - x))] for x in planes]
    angle = np.degrees(np.arctan((centre("far") - centre("near")) / abs(xn[1] - xn[0])))
    _, S = res.source_spectrum("beam", monitor="Pfar")
    exact = abs(S[0]) ** 2 * w0 * np.sqrt(np.pi / 2) * h / (2 * ETA0)
    forward = s * res.flux("Pfar")[0] / exact
    near = s * res.flux("Pnear")[0] / exact
    behind = abs(res.flux("behind")[0]) / exact
    print(f"{polarization} {direction} {theta_deg} deg: direction {angle:.2f} deg, power {forward:.4f} (near plane {near:.4f}), behind {behind:.1e}")
    # the |E|^2 centroid on planes that are not normal to a diverging beam runs slightly steeper than its axis
    assert abs(angle - theta_deg) < 0.8
    assert abs(forward - 1) < 0.02 and abs(near - 1) < 0.02
    assert behind < 5e-3


def test_gaussian_beam_direction_vector_and_errors():
    sig = ls.GaussianPulse(3e14, 3e13)
    b = ls.GaussianBeam(2, 0.0, (0.3, 0.4, -1.0), "Ex", [0, 0, 0], 2e-6, sig)
    assert b["direction"] == "-"
    assert b["theta"] == pytest.approx(np.arctan(0.5)) and b["phi"] == pytest.approx(np.arctan2(0.4, 0.3))
    assert ls.GaussianBeam(0, 0.0, "+", "Ey", [0, 0, 0], 2e-6, sig)["theta"] == 0.0
    with pytest.raises(ValueError, match="component along the source axis"):
        ls.GaussianBeam(2, 0.0, (1.0, 0.0, 0.0), "Ex", [0, 0, 0], 2e-6, sig)
    with pytest.raises(ValueError, match="not both"):
        ls.GaussianBeam(2, 0.0, (0.0, 0.0, 1.0), "Ex", [0, 0, 0], 2e-6, sig, theta=0.1)
    # a beam axis out of the plane of a two-dimensional simulation is refused
    h = 50e-9
    sim = ls.Simulation(min=[0, -4 * um, 0], max=[4 * um, 4 * um, h], mesh=ls.UniformMesh(h), run_time=1e-14)
    sim.set_boundaries(z=ls.PERIODIC)
    sim.add_source(ls.GaussianBeam(0, 1 * um, "+", "Ey", [0, 0, 0], 1.5 * um, ls.GaussianPulse(3e14, 3e13), theta=0.3, phi=np.pi / 2))
    with pytest.raises(RuntimeError, match="uniform along axis 2"):
        sim.run(device=DEVICE)
