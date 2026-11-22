"""Resonance analysis (lightsim.resonance): harmonic inversion on synthetic
ring-downs, Lorentzian fits, and the modes of a Fabry-Perot slab simulated with
FDTD against the analytic (Airy / transfer-matrix) poles."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.resonance import harminv, lorentzian_fit, ring_down, source_end_time
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def ringdown(t, modes, real=True):
    """sum a exp[(-2 pi i f - pi f / Q) t] (+ c.c.) for modes (f, Q, a)."""
    s = sum(a * np.exp((-2j * np.pi * f - np.pi * abs(f) / q) * t) for f, q, a in modes)
    return 2 * s.real if real else s


def nearest(found, f):
    return min(found, key=lambda m: abs(m.freq - f))


# two modes 0.5 % apart, Q = 1e3 and 1e4
TWO = [(1.0, 1e3, 1.0), (1.005, 1e4, 0.5 * np.exp(0.7j))]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_close_modes_with_noise(seed):
    dt = 0.05  # 20 samples per period
    t = np.arange(30000) * dt  # 1500 periods
    rng = np.random.default_rng(seed)
    x = ringdown(t, TWO) + 1e-2 * rng.standard_normal(len(t))  # 1 % noise
    found = harminv(x, dt, 0.98, 1.02)
    assert len(found) == 2, found
    for (f, q, a), m in zip(TWO, found):
        print(f"f={f}: df/f {m.freq / f - 1:+.2e}  dQ/Q {m.Q / q - 1:+.2e}  da/a {abs(m.amplitude / a - 1):.2e}  error {m.error:.1e}")
        assert abs(m.freq / f - 1) < 1e-6
        assert abs(m.Q / q - 1) < 0.02
        assert m.decay == pytest.approx(np.pi * m.freq / m.Q)
        assert abs(m.amplitude / a - 1) < 0.01
        assert m.error < 1e-5


def test_noise_free_is_exact_and_short_record_suffices():
    # 150 periods: far below the Q = 1e4 lifetime and only 0.75 beat periods of
    # the pair, where a DFT (resolution 1/T = 0.67 %) cannot even separate them
    dt = 0.05
    t = np.arange(3000) * dt
    found = harminv(ringdown(t, TWO), dt, 0.98, 1.02)
    assert len(found) == 2
    for (f, q, a), m in zip(TWO, found):
        assert abs(m.freq / f - 1) < 1e-9
        assert abs(m.Q / q - 1) < 1e-5
        assert abs(m.amplitude / a - 1) < 1e-6


def test_weak_mode_many_modes_and_window():
    dt = 0.05
    t = np.arange(40000) * dt
    rng = np.random.default_rng(3)
    # a mode 60 dB below its neighbour, 20 dB above the noise
    weak = [(1.0, 1000, 1.0), (1.01, 3000, 1e-3)]
    found = harminv(ringdown(t, weak) + 1e-4 * rng.standard_normal(len(t)), dt, 0.95, 1.05)
    assert len(found) == 2
    assert abs(found[1].freq / 1.01 - 1) < 2e-6 and abs(found[1].Q / 3000 - 1) < 0.02
    # six modes; analysing a sub-window returns only the modes inside it
    six = [(0.8 + 0.07 * k, 200.0 * (k + 1), np.exp(1j * k) / (k + 1)) for k in range(6)]
    x = ringdown(t, six) + 1e-3 * rng.standard_normal(len(t))
    found = harminv(x, dt, 0.7, 1.3)
    assert len(found) == 6
    for (f, q, a), m in zip(six, found):
        assert abs(m.freq / f - 1) < 1e-6 and abs(m.Q / q - 1) < 5e-3 and abs(m.amplitude / a - 1) < 5e-3
    found = harminv(x, dt, 0.9, 1.05)
    assert [round(m.freq, 3) for m in found] == [0.94, 1.01]


def test_complex_signal_and_undecimated_path():
    rng = np.random.default_rng(4)
    # broad window on a short record: no decimation, +f and -f modes of a complex signal
    dt = 0.25
    t = np.arange(600) * dt
    modes = [(-0.5, 40, 1.0), (0.6, 60, 0.5)]
    found = harminv(ringdown(t, modes, real=False), dt, -2, 2)
    assert len(found) == 2
    for (f, q, a), m in zip(modes, found):
        assert abs(m.freq / f - 1) < 1e-10 and abs(m.Q / q - 1) < 1e-9 and abs(m.amplitude / a - 1) < 1e-9
    # real signal, window from 0: only the positive-frequency member of each pair
    modes = [(0.3, 50, 1.0), (0.9, 30, 0.5j), (1.4, 80, 0.3)]
    found = harminv(ringdown(t, modes) + 1e-4 * rng.standard_normal(len(t)), dt, 0.0, 2.0)
    assert len(found) == 3
    for (f, q, a), m in zip(modes, found):
        assert abs(m.freq / f - 1) < 1e-5 and abs(m.Q / q - 1) < 1e-3


def test_source_tail_is_skipped_with_t_start():
    dt = 0.05
    t = np.arange(40000) * dt
    mode = (1.0, 500.0, 1.0)
    burst = 5 * np.exp(-((t - 30) ** 2) / 50) * np.cos(2 * np.pi * t)  # "source" on top of the ring-down
    x = ringdown(t, [mode]) + burst
    # with the excitation inside the record the decay is wrong ...
    bad = harminv(x, dt, 0.9, 1.1)
    assert not bad or abs(nearest(bad, 1.0).Q / 500 - 1) > 0.05
    # ... skipping it restores the mode; the amplitude refers to t_ref = t_start
    (m,) = harminv(x, dt, 0.9, 1.1, t_start=80.0)
    assert m.t_ref == pytest.approx(80.0)
    assert abs(m.freq - 1) < 1e-9 and abs(m.Q / 500 - 1) < 1e-6
    a0 = m.amplitude * np.exp((2j * np.pi * m.freq + m.decay) * m.t_ref)  # back to t = 0
    assert abs(a0 - 1) < 1e-6


def test_pure_noise_gives_no_modes():
    rng = np.random.default_rng(5)
    with pytest.warns(UserWarning, match="harminv: .*No mode is returned"):
        assert harminv(rng.standard_normal(40000), 0.05, 0.9, 1.1) == []
    with pytest.warns(UserWarning, match="the signal is zero"):
        assert harminv(np.zeros(1000), 0.05, 0.9, 1.1) == []


def _comb(duration, spacing, lines, noise, seed=1, dt=4.0e-17):
    """A ring-like comb: `lines` damped lines `spacing` apart around 190 THz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration / dt)) * dt
    f = 190e12 + spacing * (np.arange(lines) - lines / 2 + 0.37)
    q = rng.uniform(3e3, 2e4, lines)
    amp = rng.uniform(0.3, 1.0, lines) * np.exp(1j * rng.uniform(0, 2 * np.pi, lines))
    x = sum(np.real(a * np.exp((-2j * np.pi * fk - np.pi * fk / qk) * t)) for fk, qk, a in zip(f, q, amp))
    return x + noise * rng.standard_normal(len(t)), dt, f, q


def test_record_too_short_for_the_line_spacing_warns():
    """16 lines 1 THz apart in a 1.2 ps record: the record covers 1.2 beat
    periods of neighbouring lines (1.2 round trips of the ring), which is not
    enough to tell them apart. harminv returned a few wrong lines, or none,
    without a word. The same comb is found in a record twice as long."""
    import warnings

    spacing = 1.0e12
    x, dt, f, q = _comb(1.2e-12, spacing, 16, 1e-6)
    info = {}
    with pytest.warns(UserWarning, match=r"not reproducible between two pencil sizes.*as close as .* 2 / duration = 1\.667e\+12 apart") as rec:
        modes = harminv(x, dt, f.min() - spacing, f.max() + spacing, diagnostics=info)
    right = sum(any(abs(m.freq - fk) < 0.05 * spacing for m in modes) for fk in f)
    print(f"1.2 ps: {len(modes)} returned, {right} right; {info}")
    assert right < 16 and "may be wrong" in str(rec[0].message)
    assert info["unstable"] >= 4 and info["duration"] == pytest.approx(1.2e-12, rel=1e-3) and info["closest_spacing"] < 2 / 1.2e-12

    # twice the record: all 16, silently, to 1e-5 of the spacing
    x, dt, f, q = _comb(2.4e-12, spacing, 16, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        modes = harminv(x, dt, f.min() - spacing, f.max() + spacing)
    assert len(modes) == 16
    assert max(abs(m.freq - fk) for m, fk in zip(modes, f)) < 1e-5 * spacing
    assert max(abs(m.Q / qk - 1) for m, qk in zip(modes, q)) < 1e-2
    # the short record is enough when the lines are 2 THz apart
    x, dt, f, q = _comb(1.2e-12, 2 * spacing, 16, 1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(harminv(x, dt, f.min() - 2 * spacing, f.max() + 2 * spacing)) == 16


def test_empty_window_warns():
    x, dt, f, _ = _comb(2.4e-12, 1.0e12, 4, 1e-6)
    with pytest.warns(UserWarning, match="No mode is returned"):
        assert harminv(x, dt, 250e12, 260e12) == []


def test_lorentzian_fit_peak_and_dip():
    rng = np.random.default_rng(6)
    f0, q = 193.4e12, 12345.0
    w = f0 / q
    f = np.linspace(f0 - 8 * w, f0 + 8 * w, 401)
    lor = 1.0 / (1.0 + ((f - f0) / (w / 2)) ** 2)
    # transmission dip on a sloped baseline, 0.5 % noise
    y = 0.9 + 0.02 * (f - f0) / w - 0.7 * lor + 5e-3 * rng.standard_normal(len(f))
    fit = lorentzian_fit(f, y)
    print(f"dip: f0 {fit.center / f0 - 1:+.2e}  Q {fit.Q / q - 1:+.2e}  +- {fit.fwhm_err / fit.fwhm:.1e}")
    assert fit.amplitude < 0
    assert abs(fit.center - f0) < 0.01 * w
    assert abs(fit.Q / q - 1) < 0.02
    assert abs(fit.amplitude + 0.7) < 0.01 and abs(fit.background - 0.9) < 0.01
    assert abs(fit.center - f0) < 5 * fit.center_err and abs(fit.fwhm - w) < 5 * fit.fwhm_err
    assert np.max(np.abs(fit(f) - y)) < 0.03
    # exact peak (the |E|^2 spectrum of a ring-down), off-centre in the window
    f = np.linspace(f0 - 3 * w, f0 + 6 * w, 150)
    fit = lorentzian_fit(f, 3e-7 / (1.0 + ((f - f0) / (w / 2)) ** 2), linear_background=False)
    assert fit.amplitude > 0
    assert abs(fit.center / f0 - 1) < 1e-9 and abs(fit.Q / q - 1) < 1e-6 and fit.residual < 1e-6


def test_lorentzian_fit_matches_harminv_on_a_spectrum():
    # power spectrum of a fully decayed single-mode ring-down is a Lorentzian of FWHM f/Q
    f0, q, dt = 1.0, 300.0, 0.05
    t = np.arange(200000) * dt  # 33 lifetimes
    x = ringdown(t, [(f0, q, 1.0)], real=False)
    freqs = np.linspace(0.98, 1.02, 201)
    spec = np.abs(np.exp(2j * np.pi * freqs[:, None] * t[None, ::4]) @ x[::4]) ** 2
    fit = lorentzian_fit(freqs, spec)
    (m,) = harminv(x, dt, 0.95, 1.05)
    assert abs(fit.center / m.freq - 1) < 1e-5
    assert abs(fit.Q / m.Q - 1) < 5e-3


def test_fabry_perot_slab_ring_down():
    """Dielectric slab (n = 3.5, thickness d) in vacuum, normal incidence.

    Its quasi-normal modes are the poles of the Airy transmission
    t1 t2 e^{i phi} / (1 - r^2 e^{2 i phi}), phi = omega n d / c, with the
    Fresnel mirror reflectivity R = r^2 = ((n - 1) / (n + 1))^2:

        omega_m = (c / 2 n d) (2 pi m - i ln(1 / R))
        f_m = m c / (2 n d),     Q_m = omega' / (2 |omega''|) = pi m / ln(1 / R)
    """
    n, d = 3.5, 1.0 * um
    dx = 12.5e-9  # 80 cells across the slab, 40 per wavelength in it at m = 4
    R = ((n - 1) / (n + 1)) ** 2
    fsr = ls.C0 / (2 * n * d)
    f0 = 4 * fsr
    sim = ls.Simulation(min=[-1.0 * um, 0, 0], max=[1.0 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=220e-15)
    # column: PML along x, periodic transverse; a point dipole is then a current sheet
    sim.set_boundaries(x=ls.PML(), y={"type": "periodic"}, z={"type": "periodic"})
    sim.add_medium(ls.Medium("slab", n=n))
    sim.add_structure(ls.Box([-d / 2, -1, -1], [d / 2, 1, 1]), "slab")
    # source and probe off-centre and off the nodes so that modes of both parities show up
    sim.add_source(ls.PointDipole([0.2 * um, dx, dx], "Ez", ls.GaussianPulse(f0, 0.35 * f0)))
    sim.add_monitor(ls.ProbeMonitor("cav", [-0.13 * um, dx, dx], ["Ez"]))
    res = sim.run(device=DEVICE)

    t_start = source_end_time(sim)
    assert 25e-15 < t_start < 35e-15
    modes = ring_down(res, "cav", "Ez", 2.6 * fsr, 5.4 * fsr, sim=sim)
    assert [round(m.freq / fsr) for m in modes] == [3, 4, 5], modes
    for m in modes:
        k = round(m.freq / fsr)
        f_th = k * fsr
        q_th = np.pi * k / np.log(1 / R)
        print(f"m={k}: f {m.freq / 1e12:.3f} THz (Airy {f_th / 1e12:.3f}, {m.freq / f_th - 1:+.2e})  Q {m.Q:.4f} (Airy {q_th:.4f}, {m.Q / q_th - 1:+.2e})  error {m.error:.1e}")
        assert abs(m.freq / f_th - 1) < 2e-3
        assert abs(m.Q / q_th - 1) < 0.02
        assert m.error < 1e-4
    # the same decay rate for every mode: gamma = c ln(1/R) / (2 n d)
    gam = ls.C0 * np.log(1 / R) / (2 * n * d)
    assert all(abs(m.decay / gam - 1) < 0.02 for m in modes)
    # explicit window / t_start give the same answer as the defaults taken from sim
    again = ring_down(res, "cav", "Ez", 3.5 * fsr, 4.5 * fsr, t_start=t_start)
    assert len(again) == 1 and abs(again[0].freq / modes[1].freq - 1) < 1e-4 and abs(again[0].Q / modes[1].Q - 1) < 5e-3
