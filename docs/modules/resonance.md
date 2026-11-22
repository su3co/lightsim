# Resonance analysis

`lightsim.resonance`: frequencies and quality factors of cavity modes from a
time-domain run — the equivalent of Harminv (Meep) and of Lumerical's high-Q
analysis group — plus a Lorentzian fit for lines in a frequency-domain
spectrum.

```python
import lightsim as ls
from lightsim.resonance import harminv, ring_down, lorentzian_fit, source_end_time

# the Fabry-Perot slab of the validation below: n = 3.5, d = 1 um, in a
# column that is periodic in y and z
dx = 12.5e-9
fsr = ls.C0 / (2 * 3.5 * 1e-6)            # spacing of the slab modes
f0 = 4 * fsr
fmin, fmax = 2.6 * fsr, 5.4 * fsr
sim = ls.Simulation(min=[0, 0, 0], max=[3e-6, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=220e-15)
sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
sim.add_structure(ls.Box([1e-6, -1, -1], [2e-6, 1, 1]), ls.Medium("slab", n=3.5))
pos, probe_pos = [1.3e-6, 0, 0], [1.6e-6, 0, 0]

sim.add_source(ls.PointDipole(pos, "Ez", ls.GaussianPulse(f0, 0.3 * f0)))
sim.add_monitor(ls.ProbeMonitor("cav", probe_pos, ["Ez"]))
res = sim.run()
for m in ring_down(res, "cav", "Ez", fmin, fmax, sim=sim):
    print(m.freq, m.Q, m.decay, abs(m.amplitude), m.error)
```

The module is pure Python (numpy + scipy), needs no Rust or GPU support, and is
imported as `lightsim.resonance` (it is not re-exported from the top-level
package).

## Why not just a DFT

After the source has switched off, a probe inside a resonator records a sum of
damped exponentials,

```text
s(t) = sum_k a_k exp[(-2 pi i f_k - gamma_k)(t - t_ref)]     (+ c.c. for a real field)
```

A DFT monitor resolves `f_k` only to `~1/T` and gives a linewidth only if the run
lasts several lifetimes `Q / (pi f)`. Harmonic inversion fits the model above
directly, so the record only has to be long enough to tell the modes apart and
to beat the noise: a `Q = 1e4` mode sitting 0.5 % from a `Q = 1e3` neighbour is
recovered to nine digits from 150 periods of clean data (`test_noise_free_is_exact_and_short_record_suffices`).

## Conventions

| quantity | meaning |
|---|---|
| `freq` | real part of the complex frequency, in the units of `1/dt` (Hz for lightsim results) |
| `decay` | amplitude decay rate `gamma` (1/s): field `~ exp(-gamma t)`, stored energy `~ exp(-2 gamma t)` |
| `Q` | `pi * freq / decay = omega / (2 gamma)`; equals `freq / FWHM` of the Lorentzian power spectrum |
| `amplitude` | complex `a_k` at `t_ref`, the time of the first analysed sample. For a real signal the mode and its conjugate are both present, so the peak of the real oscillation is `2 |a_k|` |
| `omega` | property, `2 pi freq - i decay` (time dependence `exp(-i omega t)`) |
| `error` | relative change `|d omega| / |omega|` of the complex frequency between two independent pencil sizes (see below) |

## Method

`harminv` is a band-limited, SVD-truncated matrix-pencil method (Hua & Sarkar,
IEEE Trans. ASSP 38, 814 (1990)) — the least-squares relative of the
filter-diagonalization method (Mandelshtam & Taylor, J. Chem. Phys. 107, 6756
(1997)) that Harminv implements. Both diagonalise the one-step evolution
operator in a subspace spanned by the signal restricted to `[fmin, fmax]`.

1. **Crop** to `t_start <= t <= t_end`. While the source is on, the signal is
   not a sum of damped exponentials; `source_end_time` gives `t_off + 6 tau` for
   a `GaussianPulse` (envelope down to 1.5e-8).
2. **Band-limit**: mix down by the window centre, low-pass with a linear-phase
   Kaiser FIR (`stopband_db`, default 100 dB) and decimate to a sample rate
   `oversample * (fmax - fmin)` (default 2.5x). A FIR filter maps
   `a z^n` to `a H(z) z^n`, so the poles are untouched and the amplitudes are
   corrected exactly by dividing by the filter response `H` at the complex
   pole. This averages the noise over the decimation factor (hundreds for an
   FDTD time step) and makes the linear algebra small. The filter consumes
   `~4.3 / (fmax - fmin)` of the record; if that is more than a tenth of it the
   decimation is reduced automatically, down to none.
3. **Model order**: singular values of the Hankel matrix `Y[i, j] = y[i + j]`
   (pencil parameter `L = N/3`, capped by `max_pencil`) above the noise floor
   are kept. The white-noise level is read from the upper half of the spectrum
   of the raw signal (median of the periodogram, so stray lines do not matter)
   and propagated through the filter; without decimation the median singular
   value is used. `order=` or `sv_tol=` override this, `nmax` caps it.
4. **Poles**: eigenvalues of the shift operator restricted to the dominant right
   singular subspace; `z_k = exp[(-2 pi i (f_k - f_c) - gamma_k) D dt]`.
5. **Amplitudes**: linear least squares on the decimated signal.
6. **Error estimate and filtering**: the poles are recomputed with a different
   pencil size. A mode that is in the data is stable to many digits; a noise
   pole moves by a fraction of the bandwidth. `error` is that relative shift —
   an order-of-magnitude indicator of the frequency error, not a bound. Modes
   are dropped if `error > max_error` (1e-3), if the shift exceeds `stability`
   (0.1) times their own linewidth `max(gamma, 2 pi / T)`, if they are weaker
   than `min_amplitude` (1e-4) of the strongest, if they grow (`Q < 0`), or lie
   outside `[fmin, fmax]`.

Limits to know about:

* Out-of-window modes further than `(oversample - 1)(fmax - fmin)` from the
  window fold back into it attenuated by `stopband_db`. With the default 100 dB
  and `min_amplitude = 1e-4` they are not reported; lowering either can let
  such images through (seen at `stopband_db=60` in the slab case below).
* The automatic order assumes white noise. FDTD round-off noise is not quite
  white (it is shaped by the structure's response), which only leads to mild
  over-modelling; the stability filter then removes the extra poles.
* Keep `[fmin, fmax]` reasonably tight: at most `nmax` exponentials are fitted
  in the decimated band.
* **The record must cover about two beat periods of neighbouring lines**,
  `T >= 2 / df`. For the comb of a ring this is two round trips. Without noise
  closer lines are still found (see the validation below); with the noise of
  a simulation they are not. A comb of 16 lines 1 THz apart in a 1.2 ps
  record (1.2 beat periods) gives 4 lines, 3 of them right; the same comb in
  2.4 ps gives all 16 to 1e-6 of the spacing, and 16 lines 2 THz apart are
  found in 1.2 ps.

`harminv` warns when its answer is empty or doubtful: when no mode is
returned, and when a quarter or more of the poles that carry signal move
between the two pencil sizes. The message gives the numbers and, if the
closest poles are less than `2 / T` apart, says so. Pass a dictionary as
`diagnostics=` to receive them: `samples`, `duration`, `decimation`,
`decimated_samples`, `pencil`, `order`, `poles_in_window`, `unstable`,
`rejected` and `closest_spacing`.

`lorentzian_fit` is a non-linear least-squares fit (scipy `least_squares`) of
`background + slope (f - f0) + A / (1 + ((f - f0) / (FWHM / 2))^2)` with the
width in log scale and data-driven initial guesses; the sign of `A` (dip or
peak) is detected unless `dip=` is given. It returns 1-sigma uncertainties of
centre and width from the fit covariance. Use it on spectra that are actually
resolved (normalised transmission dips, spectra of fully decayed runs); the
width a DFT of a truncated ring-down shows is `~1/T`, not `f/Q`.

## API

```text
harminv(signal, dt, fmin, fmax, nmax=100, *, t_start=0.0, t_end=None, t0=0.0,
        order=None, sv_tol=None, max_error=1e-3, stability=0.1, noise_factor=1.5,
        min_amplitude=1e-4, min_Q=0.0, oversample=2.5, stopband_db=100.0,
        max_pencil=400, diagnostics=None) -> list[Resonance]
```
Real or complex uniformly sampled series, sample `n` at `t0 + n dt`. Returns the
modes in `[fmin, fmax]` sorted by frequency. If none stands out of the noise
the list is empty and a `UserWarning` says why.

```text
ring_down(results, probe, component="Ez", fmin=None, fmax=None, nmax=100, *,
          t_start=None, sim=None, point=None, **harminv_kwargs) -> list[Resonance]
```
`harminv` on a `ProbeMonitor` (or one cell, `point=`, of a `FieldTimeMonitor`)
of a `Results`. With `sim=` the start defaults to `source_end_time(sim)` and the
window to `freq0 -+ fwidth` of the first source; otherwise give `t_start`,
`fmin`, `fmax`.

```text
source_end_time(sim_or_source_or_signal, cutoff=6.0) -> float
```
Time after which a Gaussian (or Ricker) excitation is below `exp(-cutoff^2/2)`.

```text
lorentzian_fit(freqs, spectrum, dip=None, linear_background=True) -> LorentzianFit
```
`LorentzianFit` has `center`, `fwhm`, `Q = center / fwhm`, `amplitude` (negative
for a dip), `background`, `slope`, `residual` (relative rms), `center_err`,
`fwhm_err`, and is callable (`fit(f)` evaluates the model).

`Resonance` is a dataclass with `freq`, `decay`, `Q`, `amplitude`, `error`,
`t_ref` and the property `omega`.

## Validation (`python/tests/test_resonance.py`, CPU)

Synthetic signals (unit amplitude, 20 samples per period):

| case | result | asserted |
|---|---|---|
| two modes 0.5 % apart, Q = 1e3 and 1e4, amplitudes 1 and 0.5, 1 % white noise, 1500 periods, window 0.98–1.02 (3 seeds) | worst of 3 seeds: `df/f` 2.4e-7, `dQ/Q` 1.3e-3, amplitude 7e-4, `error` 1.5e-7 | `df/f` < 1e-6, `dQ/Q` < 2 %, amplitude < 1 % |
| same, no noise, only 150 periods | `df/f` < 1e-9, `dQ/Q` < 1e-5 | same |
| mode 60 dB below its neighbour (1 % away), 20 dB above the noise | found, `df/f` < 2e-6, `dQ/Q` < 2 % | same |
| six modes, Q 200–1200, 0.1 % noise; sub-window | all six, `df/f` < 1e-6, `dQ/Q` < 0.5 %; the sub-window returns exactly the two modes in it | same |
| complex signal with modes at -0.5 and +0.6, undecimated path | 1e-10 / 1e-9 | same |
| excitation burst left in the record vs `t_start` after it | wrong Q without `t_start`; exact (1e-9, 1e-6) with it, amplitude back-propagated to t = 0 within 1e-6 | same |
| pure white noise, all-zero signal | no modes | same |
| Lorentzian dip, Q = 12345, sloped baseline, 0.5 % noise, 401 points | centre off by 1.3e-7 relative (0.2 % of the linewidth), Q +0.7 % (fit's own 1-sigma: 0.4 %) | centre within 1 % of the linewidth, Q within 2 % |
| noise-free Lorentzian peak, off-centre window | centre 1e-9, Q 1e-6 | same |
| Lorentzian fit of the DFT power spectrum of a decayed ring-down vs `harminv` on the series | centre 1e-5, Q 0.5 % | same |

FDTD — Fabry-Perot slab (`test_fabry_perot_slab_ring_down`). A dielectric slab,
`n = 3.5`, `d = 1 um`, in vacuum; column of 2 x 2 cells periodic in y and z, PML
along x, `dx = 12.5 nm` (80 cells across the slab), a z-polarised dipole (a
current sheet in this column) with a Gaussian pulse centred on the 4th mode,
`run_time = 220 fs` (9232 steps, 736 cells, a few seconds on CPU). The
quasi-normal modes are the poles of the Airy transmission
`t1 t2 e^{i phi} / (1 - R e^{2 i phi})`, `phi = omega n d / c`, with the Fresnel
mirror reflectivity `R = ((n - 1) / (n + 1))^2 = 0.3086`:

```text
f_m = m c / (2 n d),     Q_m = pi m / ln(1 / R),     gamma = c ln(1 / R) / (2 n d)
```

`ring_down(res, "cav", "Ez", 2.6 fsr, 5.4 fsr, sim=sim)` returns exactly the
modes m = 3, 4, 5:

| m | f FDTD (THz) | f Airy (THz) | df/f | Q FDTD | Q Airy | dQ/Q | `error` |
|---|---|---|---|---|---|---|---|
| 3 | 128.409 | 128.482 | -0.057 % | 8.0129 | 8.0172 | -0.053 % | 2e-9 |
| 4 | 171.137 | 171.310 | -0.101 % | 10.6788 | 10.6896 | -0.100 % | 1e-7 |
| 5 | 213.801 | 214.137 | -0.157 % | 13.3412 | 13.3620 | -0.155 % | 2e-8 |

Asserted: frequency within 0.2 %, Q within 2 %, `error` < 1e-4, decay rate
within 2 % of the common analytic value. The residual frequency error grows as
`m^2`, as expected of the second-order numerical dispersion of the Yee grid (53,
40 and 32 cells per wavelength in the slab for m = 3, 4, 5; at `dx = 20 nm` the
same run gives -0.15 %, -0.26 %, -0.40 %), and Q follows it because the decay
rate itself is reproduced to better than 0.02 %. Note that these are low-Q modes (Q = 8–13) whose linewidths overlap
(linewidth 10 % of `f`, spacing 25 %), a case where reading peaks off a DFT is
unreliable.

Run:

```bash
PYTHONPATH=python pytest -q python/tests/test_resonance.py
```
