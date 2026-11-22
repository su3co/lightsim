"""Resonance analysis: harmonic inversion of ring-down signals and Lorentzian
fits of spectra (the equivalent of Harminv / Lumerical's high-Q analysis).

A cavity excited by a short pulse rings down as a sum of damped exponentials,

    s(t) = sum_k a_k exp[(-2 pi i f_k - gamma_k) (t - t_ref)]   (+ c.c. if s is real)

:func:`harminv` recovers ``f_k``, ``gamma_k`` and ``a_k`` directly from the time
series, without the Fourier limit ``df ~ 1/T`` of a DFT: a mode of quality
factor ``Q = pi f / gamma`` only needs a record of a few periods times the
inverse relative spacing to its neighbours, not ``Q`` periods.

    import lightsim as ls
    from lightsim.resonance import ring_down
    res = sim.run()
    for m in ring_down(res, "probe", "Ez", fmin, fmax, sim=sim):
        print(m.freq, m.Q, abs(m.amplitude), m.error)

Conventions (used everywhere in this module):

* ``freq``  — real part of the complex frequency, Hz (same units as ``1/dt``).
* ``decay`` — amplitude decay rate ``gamma`` in 1/s: the field envelope goes as
  ``exp(-gamma t)``, the stored energy as ``exp(-2 gamma t)``.
* ``Q = pi * freq / decay = omega / (2 gamma)`` — energy stored over energy lost
  per radian, equal to ``freq / FWHM`` of the Lorentzian power spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Resonance", "LorentzianFit", "harminv", "ring_down", "lorentzian_fit", "source_end_time"]


@dataclass
class Resonance:
    """One damped mode ``amplitude * exp[(-2 pi i freq - decay) (t - t_ref)]``."""

    freq: float  #: frequency (Hz)
    decay: float  #: amplitude decay rate gamma (1/s); energy decays as exp(-2 gamma t)
    Q: float  #: quality factor pi * freq / decay (inf for an undamped mode)
    amplitude: complex  #: complex amplitude at ``t_ref``
    error: float  #: relative error estimate of the complex frequency (see :func:`harminv`)
    t_ref: float = 0.0  #: time the amplitude refers to (first analysed sample)

    @property
    def omega(self) -> complex:
        """Complex angular frequency ``2 pi freq - i decay`` (``exp(-i omega t)`` convention)."""
        return 2 * np.pi * self.freq - 1j * self.decay

    def __repr__(self) -> str:
        return f"Resonance(freq={self.freq:.9g}, Q={self.Q:.6g}, decay={self.decay:.6g}, |amp|={abs(self.amplitude):.4g}, error={self.error:.2g})"


# ----------------------------------------------------------------------------
# harmonic inversion
# ----------------------------------------------------------------------------


def _pencil_poles(y: np.ndarray, L: int, order: int | None, sv_tol: float, nmax: int):
    """Matrix-pencil poles ``z`` of ``y[m] = sum a z^m`` and the singular values."""
    from scipy.linalg import svd

    n = len(y)
    idx = np.arange(n - L)[:, None] + np.arange(L + 1)[None, :]
    s, vh = svd(y[idx], full_matrices=False)[1:]
    if order is None:
        order = int(np.count_nonzero(s > sv_tol * s[0]))
    order = max(1, min(order, nmax, L - 1, len(s)))
    w = vh[:order].T  # columns span the vectors (z^0 .. z^L)
    phi = np.linalg.lstsq(w[:-1], w[1:], rcond=None)[0]
    return np.linalg.eigvals(phi), s


def _white_noise_sigma(x: np.ndarray, dt: float, fmin: float, fmax: float) -> float | None:
    """Standard deviation of the white noise in ``x``, read from the upper half
    of its (Hann-windowed) spectrum, away from the analysis window. The median
    is used so that a few out-of-band lines do not matter. None if too few
    bins are available."""
    n = len(x)
    w = np.hanning(n)
    p = np.abs(np.fft.fft(w * x)) ** 2
    f = np.abs(np.fft.fftfreq(n, dt))
    band = fmax - fmin
    sel = (f >= 0.25 / dt) & ((f < fmin - band) | (f > fmax + band))
    if np.count_nonzero(sel) < 16:
        return None
    # |X|^2 of white noise is exponentially distributed with mean sigma^2 sum(w^2), median ln 2 times that
    return float(np.sqrt(np.median(p[sel]) / (np.log(2.0) * np.sum(w**2))))


def _plateau_floor(s: np.ndarray) -> float | None:
    """Noise level of a singular-value spectrum from its median — valid when
    the matrix is much larger than the number of modes."""
    return float(np.median(s)) if len(s) >= 32 else None


def harminv(
    signal,
    dt: float,
    fmin: float,
    fmax: float,
    nmax: int = 100,
    *,
    t_start: float = 0.0,
    t_end: float | None = None,
    t0: float = 0.0,
    order: int | None = None,
    sv_tol: float | None = None,
    max_error: float = 1e-3,
    stability: float = 0.1,
    noise_factor: float = 1.5,
    min_amplitude: float = 1e-4,
    min_Q: float = 0.0,
    oversample: float = 2.5,
    stopband_db: float = 100.0,
    max_pencil: int = 400,
    diagnostics: dict | None = None,
) -> list[Resonance]:
    """Harmonic inversion of a uniformly sampled time series.

    Finds the damped sinusoids ``a exp[(-2 pi i f - gamma)(t - t_ref)]`` with
    ``fmin <= f <= fmax`` contained in ``signal`` (real or complex; a real signal
    is the sum of each mode and its complex conjugate, so ``abs(amplitude)`` is
    half the peak amplitude of the real oscillation).

    Method: the signal is mixed down to the band centre, low-pass filtered with
    a linear-phase Kaiser FIR and decimated to a sample rate ``oversample *
    (fmax - fmin)`` — a FIR filter maps a sum of damped exponentials to a sum of
    the same exponentials, so this loses no accuracy while averaging the noise
    and shrinking the problem — and the poles are then extracted with the
    SVD-truncated matrix-pencil method (Hua & Sarkar 1990), the least-squares
    relative of the filter-diagonalization method used by Harminv. Amplitudes
    come from a linear least-squares fit, corrected for the filter response at
    the complex pole.

    Parameters
    ----------
    signal : sequence of float or complex, sample ``n`` taken at ``t0 + n*dt``.
    dt : sampling interval.
    fmin, fmax : frequency window (same units as ``1/dt``). Keep it reasonably
        tight around the modes of interest; modes outside are ignored.
    nmax : maximum model order, i.e. number of exponentials fitted in the
        band-limited signal (the modes in the window plus a few around it).
    t_start, t_end : analyse only ``t_start <= t <= t_end``. Set ``t_start``
        after the excitation has died out (see :func:`source_end_time`) — while
        the source is on the signal is not a sum of damped exponentials.
    t0 : time of ``signal[0]``.
    order : force the model order (number of exponentials in the decimated
        band, including out-of-window and noise poles). Default: automatic from
        the singular-value spectrum.
    sv_tol : keep singular values above ``sv_tol * s_max`` instead of the
        automatic noise-floor estimate.
    noise_factor : safety factor on the estimated noise singular values in the
        automatic order selection (larger: fewer, more certain modes).
    max_error : discard modes whose ``error`` exceeds this.
    stability : discard modes whose complex angular frequency differs between
        the two estimates by more than ``stability * max(decay, 2 pi / T)``
        (``T`` the analysed duration), i.e. that are not pinned down to within
        a fraction of their own linewidth. This is what rejects noise poles.
    min_amplitude : discard modes weaker than this fraction of the strongest.
        Modes outside ``[fmin, fmax]`` further than ``(oversample - 1) * (fmax -
        fmin)`` from its edges fold back into the window attenuated by
        ``stopband_db``; anything weaker than that relative to the strongest
        out-of-window line cannot be told from such an image.
    min_Q : discard modes with ``Q`` below this (growing modes, ``Q < 0``, are
        always discarded).
    oversample : decimated sample rate over ``fmax - fmin`` (> 1).
    stopband_db : attenuation of the anti-alias filter.
    max_pencil : cap on the pencil parameter (size of the SVD).
    diagnostics : optional dict that is filled with what was done:
        ``samples``, ``duration``, ``decimation``, ``decimated_samples``,
        ``pencil``, ``order``, ``poles_in_window`` (poles that carry signal),
        ``unstable`` (those that moved between the two pencil sizes),
        ``rejected`` and ``closest_spacing``.

    A ``UserWarning`` is issued when no mode is returned and when a quarter
    or more of the poles in the window are not reproducible, with the reason
    as far as it can be told. The usual one is a record too short for the
    line spacing: noisy lines are separated reliably only if the record
    covers two of their beat periods, ``duration >= 2 / spacing`` (two round
    trips, for the comb of a ring).

    Returns
    -------
    list of :class:`Resonance`, sorted by frequency. ``error`` is the relative
    change ``|d omega| / |omega|`` of the complex frequency between two
    independent pencil sizes: a mode that is really in the data is stable to
    many digits, a spurious (noise) pole moves by a fraction of the bandwidth.
    It is an order-of-magnitude indicator of the frequency error, not a bound.
    """
    from scipy import signal as sps

    x = np.asarray(signal)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    is_real = not np.iscomplexobj(x)
    x = x.astype(float if is_real else complex)
    if not fmax > fmin:
        raise ValueError("need fmax > fmin")
    if oversample <= 1.0:
        raise ValueError("oversample must be > 1")
    n0 = max(0, int(np.ceil((t_start - t0) / dt - 1e-9)))
    n1 = len(x) if t_end is None else min(len(x), int(np.floor((t_end - t0) / dt + 1e-9)) + 1)
    x = x[n0:n1]
    if len(x) < 8:
        raise ValueError("fewer than 8 samples between t_start and t_end")
    t_ref = t0 + n0 * dt
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        import warnings

        warnings.warn("harminv: the signal is zero between t_start and t_end; no mode is returned", stacklevel=2)
        return []
    x = x / scale

    band = fmax - fmin
    fs = 1.0 / dt
    sigma = _white_noise_sigma(x, dt, fmin, fmax)
    dec = int(np.floor(fs / (oversample * band)))
    beta = sps.kaiser_beta(stopband_db)
    while dec >= 2:
        fs2 = fs / dec
        # pass |f| <= band/2, stop |f| >= fs2 - band/2 (what would alias into the window)
        ntaps = sps.kaiserord(stopband_db, (fs2 - band) / (0.5 * fs))[0] | 1
        # the filter's transient is lost from the record, and with it part of
        # the resolution (what can be told apart is set by duration * band):
        # keep the loss below a tenth. If the filter is longer, decimate less
        # (wider transition band, shorter filter), down to no filtering at all
        if ntaps <= len(x) // 10:
            break
        dec = int(dec / 1.5) if dec >= 3 else 1
    if dec >= 2:
        fc = 0.5 * (fmin + fmax)
        h = sps.firwin(ntaps, fs2 / fs, window=("kaiser", beta))
        n = np.arange(len(x))
        u = x * np.exp(2j * np.pi * fc * dt * n)
        # y[m] = sum_j h[j] u[m*dec + j]
        y = sps.fftconvolve(u, h[::-1], mode="valid")[::dec]
    else:
        dec, fc, h = 1, 0.0, np.ones(1)
        y = x.astype(complex)
    nd = len(y)
    if nd < 6:
        raise ValueError(f"only {nd} samples left after band-limiting; use a longer record or a wider [fmin, fmax]")

    L = int(min(max_pencil, max(2, nd // 3)))
    if order is None and sv_tol is None:
        from scipy.linalg import svdvals

        idx = np.arange(nd - L)[:, None] + np.arange(L + 1)[None, :]
        s = svdvals(y[idx])
        floor = 0.0
        if dec >= 2 and sigma is not None:
            # largest singular value of the filtered noise: sigma_y (sqrt(rows) + sqrt(cols)),
            # times at most sqrt(oversample) because that noise is band-limited
            sig_y = sigma * np.sqrt(np.sum(h**2))
            floor = noise_factor * np.sqrt(min(fs / (dec * band), oversample * 2)) * sig_y * (np.sqrt(nd - L) + np.sqrt(L + 1))
        elif _plateau_floor(s) is not None:
            floor = 2.0 * noise_factor * _plateau_floor(s)
        order = max(1, int(np.count_nonzero(s > max(floor, 1e-12 * s[0]))))
    z, _ = _pencil_poles(y, L, order, sv_tol or 0.0, nmax if dec >= 2 or not is_real else 2 * nmax)
    m_used = len(z)
    # independent estimate with a different pencil size, for the error
    L2 = int(round(0.6 * L))
    if L2 < m_used + 2:
        L2 = nd // 2
    z2, _ = _pencil_poles(y, L2, m_used, 0.0, m_used)

    ddt = dec * dt
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # s_k: complex rate of the demodulated mode, z = exp(s * ddt)
        sk = np.log(z) / ddt
        sk2 = np.log(z2) / ddt
        om = 1j * sk + 2 * np.pi * fc  # omega = 2 pi f - i gamma
        om2 = 1j * sk2 + 2 * np.pi * fc
        dom = np.min(np.abs(om[:, None] - om2[None, :]), axis=1)
        err = dom / np.abs(om)
        stable = dom <= stability * np.maximum(-om.imag, 2 * np.pi / (len(x) * dt))
        # amplitudes: least squares on the decimated signal
        vand = z[None, :] ** np.arange(nd)[:, None]
        ok = np.all(np.isfinite(vand), axis=0) & (np.abs(z) < 1.0 + 1e-3 + 50.0 / nd)
        amp = np.zeros(len(z), dtype=complex)
        if np.any(ok):
            amp[ok] = np.linalg.lstsq(vand[:, ok], y, rcond=None)[0]
        # undo the filter: y[m] = a H(z1) z^m with z1 = exp(s dt), H(z1) = sum h[j] z1^j
        z1 = np.exp(sk * dt)
        amp = amp / np.polyval(h[::-1], z1) * scale

    freq = om.real / (2 * np.pi)
    gamma = -om.imag
    out = []
    # poles in the window that carry signal (kept or not), for the diagnosis
    valid = np.isfinite(freq) & np.isfinite(gamma) & np.isfinite(amp) & ok & (freq >= fmin) & (freq <= fmax)
    strongest = float(np.max(np.abs(amp[valid]))) if np.any(valid) else 0.0
    window_freqs = []
    unstable = 0
    for k in range(len(z)):
        if not valid[k]:
            continue
        significant = abs(amp[k]) >= max(min_amplitude, 1e-3) * strongest
        if significant:
            window_freqs.append(float(freq[k]))
        if not stable[k]:
            unstable += significant
            continue
        q = np.pi * abs(freq[k]) / gamma[k] if gamma[k] != 0 else np.inf
        out.append(Resonance(float(freq[k]), float(gamma[k]), float(q), complex(amp[k]), float(err[k]), t_ref))
    in_window = len(window_freqs)
    candidates = len(out)
    if out:
        amax = max(abs(m.amplitude) for m in out)
        out = [m for m in out if m.error <= max_error and abs(m.amplitude) >= min_amplitude * amax and m.Q > 0 and m.Q >= min_Q]
        out.sort(key=lambda m: m.freq)

    # Say so when the answer is empty or cannot be trusted: a caller that gets
    # [] or a few wrong lines has no other way to know.
    duration = len(x) * dt
    closest = float(np.min(np.diff(np.sort(window_freqs)))) if in_window >= 2 else None
    if diagnostics is not None:
        diagnostics.update(
            samples=len(x), duration=duration, decimation=dec, decimated_samples=nd, pencil=L, order=int(m_used),
            poles_in_window=int(in_window), unstable=int(unstable), rejected=int(candidates - len(out)), closest_spacing=closest,
        )
    # Two signs of an answer that cannot be trusted: nothing came out, or a
    # good part of the poles that carry signal moved between the two pencil
    # sizes. The usual cause is lines that are too close for the record.
    problems = []
    if unstable >= max(2, (in_window + 3) // 4):
        problems.append(f"{unstable} of the {in_window} poles in the window are not reproducible between two pencil sizes")
    if not out:
        if not np.any(valid):
            problems.append(f"no pole lies in the window [{fmin:.6g}, {fmax:.6g}]")
        elif not problems:
            problems.append(f"none of the {in_window} pole(s) in the window passed the stability, max_error, min_amplitude and Q tests")
    if problems and closest is not None and duration * closest < 2.0:
        problems.append(
            f"poles in the window are as close as {closest:.4g}, and a record of duration {duration:.4g} separates noisy "
            f"lines reliably only if they are at least 2 / duration = {2.0 / duration:.4g} apart (two beat periods of neighbouring lines)"
        )
    if problems:
        import warnings

        warnings.warn(
            "harminv: "
            + "; ".join(problems)
            + (f". The {len(out)} mode(s) returned may be wrong" if out else ". No mode is returned")
            + ". A longer record (a longer run, or a later t_end) resolves closer lines; a narrower [fmin, fmax] helps when the window holds many.",
            stacklevel=2,
        )
    return out

def source_end_time(sim_or_signal, cutoff: float = 6.0) -> float:
    """Time after which a pulsed excitation is negligible.

    Accepts a signal dict (``GaussianPulse(...)``), a source dict or a
    :class:`~lightsim.Simulation` (the latest of its sources). For a Gaussian
    pulse ``exp(-(t - t_off)^2 / 2 tau^2)`` this is ``t_off + cutoff * tau``
    (envelope down to ``exp(-cutoff^2 / 2)``: 1.5e-8 at the default 6)."""
    if hasattr(sim_or_signal, "sources"):
        ends = [source_end_time(s, cutoff) for s in sim_or_signal.sources]
        return max(ends) if ends else 0.0
    sig = sim_or_signal.get("signal", sim_or_signal)
    kind = sig.get("type")
    if kind == "gaussian_pulse":
        tau = 1.0 / (2 * np.pi * sig["fwidth"])
        return (sig.get("offset", 5.0) + cutoff) * tau
    if kind == "ricker":
        return (sig.get("offset", 1.0) + cutoff / (np.pi * np.sqrt(2.0))) / sig["freq0"]
    raise ValueError(f"signal type {kind!r} never switches off; pass t_start explicitly")


def ring_down(
    results,
    probe: str,
    component: str = "Ez",
    fmin: float | None = None,
    fmax: float | None = None,
    nmax: int = 100,
    *,
    t_start: float | None = None,
    sim=None,
    point: int | None = None,
    **kwargs,
) -> list[Resonance]:
    """Harmonic inversion of a :func:`~lightsim.ProbeMonitor` time series.

    ``results`` is the :class:`~lightsim.Results` of a run, ``probe`` and
    ``component`` select the recorded series. ``t_start`` is where the analysis
    begins; if omitted and ``sim`` (the :class:`~lightsim.Simulation`) is given
    it defaults to :func:`source_end_time` of its sources, otherwise to 0 (only
    correct if the record contains no excitation). ``fmin`` / ``fmax`` default
    to the band ``freq0 -+ fwidth`` of the first source of ``sim``. A
    ``FieldTimeMonitor`` works too: pick one of its (flattened) cells with
    ``point``. Remaining keyword arguments go to :func:`harminv`.
    """
    times, series = results.time_series(probe, component)
    times = np.asarray(times, dtype=float)
    series = np.asarray(series, dtype=float).reshape(len(times), -1)
    if series.shape[1] > 1 and point is None:
        raise ValueError(f"monitor {probe!r} records {series.shape[1]} points; choose one with point=")
    series = series[:, point or 0]
    if len(times) < 2:
        raise ValueError(f"probe {probe!r} holds fewer than two samples")
    dt = float(times[1] - times[0])
    if not np.allclose(np.diff(times), dt, rtol=1e-6, atol=0.0):
        raise ValueError("probe samples are not uniformly spaced")
    if t_start is None:
        t_start = source_end_time(sim) if sim is not None else float(times[0])
    if fmin is None or fmax is None:
        if sim is None or not sim.sources:
            raise ValueError("give fmin and fmax (or sim= to take them from the source)")
        sig = sim.sources[0]["signal"]
        f0, fw = sig["freq0"], sig.get("fwidth", 0.5 * sig["freq0"])
        fmin = max(f0 - fw, 0.0) if fmin is None else fmin
        fmax = f0 + fw if fmax is None else fmax
    return harminv(series, dt, fmin, fmax, nmax, t_start=t_start, t0=float(times[0]), **kwargs)


# ----------------------------------------------------------------------------
# Lorentzian fit
# ----------------------------------------------------------------------------


@dataclass
class LorentzianFit:
    """``background + slope (f - center) + amplitude / (1 + ((f - center) / (fwhm/2))^2)``."""

    center: float  #: resonance frequency
    fwhm: float  #: full width at half maximum (of the peak or dip)
    Q: float  #: center / fwhm
    amplitude: float  #: peak height above the background (negative for a dip)
    background: float  #: baseline at ``center``
    slope: float  #: linear baseline slope
    residual: float  #: rms residual over the rms of the data minus its mean
    center_err: float = float("nan")  #: 1-sigma uncertainty of ``center`` from the fit covariance
    fwhm_err: float = float("nan")  #: 1-sigma uncertainty of ``fwhm``

    def __call__(self, f):
        """Evaluate the fitted model, background plus slope plus Lorentzian,
        at the frequencies ``f`` (same unit as the fitted ``freqs``). Returns
        an array of the shape of ``f``."""
        f = np.asarray(f, dtype=float)
        return self.background + self.slope * (f - self.center) + self.amplitude / (1.0 + ((f - self.center) / (0.5 * self.fwhm)) ** 2)


def lorentzian_fit(freqs, spectrum, dip: bool | None = None, linear_background: bool = True) -> LorentzianFit:
    """Fit one Lorentzian line to a power spectrum (transmission dip or peak).

    ``freqs`` and ``spectrum`` are real arrays covering a single resonance —
    typically ``|E(f)|^2`` or a transmission ``T(f)`` from DFT / flux monitors,
    cropped around the line. ``dip`` forces a dip (True) or a peak (False);
    by default the sign is chosen from the data. The baseline is a constant
    plus, with ``linear_background``, a slope.

    The width resolved by a DFT of a record of length ``T`` cannot be narrower
    than ``~1/T``: use this on spectra of runs that have decayed (or on
    normalised transmission), and :func:`harminv` on truncated ring-downs.
    """
    from scipy.optimize import least_squares

    f = np.asarray(freqs, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    if f.shape != y.shape or f.ndim != 1 or len(f) < 5:
        raise ValueError("freqs and spectrum must be 1-D arrays of equal length >= 5")
    o = np.argsort(f)
    f, y = f[o], y[o]
    span = f[-1] - f[0]
    fmid = 0.5 * (f[0] + f[-1])
    # baseline guess from the edges of the window
    ne = max(1, len(f) // 10)
    base = 0.5 * (np.mean(y[:ne]) + np.mean(y[-ne:]))
    if dip is None:
        dip = (base - y.min()) > (y.max() - base)
    i0 = int(np.argmin(y) if dip else np.argmax(y))
    a0 = y[i0] - base
    if a0 == 0.0:
        raise ValueError("spectrum is flat")
    # width guess: extent of the region beyond half depth
    beyond = (y - base) / a0 > 0.5
    w0 = max(float(np.sum(np.gradient(f)[beyond])), 2 * span / len(f))
    ys = np.max(np.abs(y - base))

    # unknowns in scaled units: centre, log width, amplitude, background, slope
    def model(p):
        xc = (f - fmid) / span - p[0]
        return p[3] + (p[4] * xc if linear_background else 0.0) + p[2] / (1.0 + (xc / (0.5 * np.exp(p[1]))) ** 2)

    p0 = np.array([(f[i0] - fmid) / span, np.log(w0 / span), a0 / ys, base / ys, 0.0])
    sol = least_squares(lambda p: model(p) - y / ys, p0, x_scale=[0.1, 1.0, 1.0, 1.0, 1.0], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    p = sol.x
    center = fmid + p[0] * span
    fwhm = float(np.exp(p[1]) * span)
    resid = model(p) - y / ys
    rms = float(np.sqrt(np.mean(resid**2)) / max(np.std(y / ys), 1e-300))
    cerr = werr = float("nan")
    dof = len(f) - (5 if linear_background else 4)
    if dof > 0:
        jac = sol.jac if linear_background else sol.jac[:, :4]
        try:
            cov = np.linalg.inv(jac.T @ jac) * float(np.sum(resid**2)) / dof
            cerr = float(np.sqrt(cov[0, 0]) * span)
            werr = float(np.sqrt(cov[1, 1]) * fwhm)
        except np.linalg.LinAlgError:
            pass
    return LorentzianFit(float(center), fwhm, float(center / fwhm), float(p[2] * ys), float(p[3] * ys), float(p[4] * ys / span), rms, cerr, werr)
