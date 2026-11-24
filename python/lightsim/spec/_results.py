"""The :class:`Results` of a run."""

from __future__ import annotations


class Results:
    """Thin wrapper around the result dict returned by the core."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def stats(self) -> dict:
        """Run statistics: ``backend``, ``cells``, ``steps`` (executed),
        ``seconds``, ``mcells_per_s`` (million cell updates per second),
        ``stopped_early``, and the arrays ``energy_steps`` and ``energy``, the
        field energy that ``shutoff`` looks at, every ``shutoff_interval`` steps.
        """
        return self.raw["stats"]

    @property
    def dt(self) -> float:
        """Time step in seconds.
        """
        return self.raw["dt"]

    def decay(self):
        """Field energy at the end of the run as a fraction of its peak, the
        number that ``shutoff`` is compared with (``None`` if the run was too
        short for a sample). ``stats["energy_steps"]`` and
        ``stats["energy"]`` hold the whole history."""
        import numpy as np

        e = np.asarray(self.stats.get("energy", []), dtype=float)
        if e.size == 0 or not np.max(e) > 0:
            return None
        return float(e[-1] / np.max(e))

    def flux(self, name: str):
        """Power per frequency of a ``FluxMonitor`` or ``FluxBoxMonitor`` (also the
        total power through the plane of a ``ModeMonitor`` and through the box of a
        ``FarFieldMonitor``), as an array in the order of the monitor frequencies.
        """
        return self.raw["flux"][name]

    def field(self, monitor: str, comp: str):
        """One field component of a ``DFTMonitor``: a complex array indexed
        ``[frequency, x, y, z]``, in single precision. Magnetic components are
        multiplied by the impedance of free space (376.73 ohm), so they have the
        unit of the electric field; divide by it to obtain A/m.
        """
        import numpy as np

        f = self.raw["dft"][monitor]["fields"][comp]
        return np.asarray(f["re"]) + 1j * np.asarray(f["im"])

    def freqs(self, monitor: str):
        """Frequencies (Hz) of a frequency-domain monitor, in the order given.
        """
        return self.raw["dft"][monitor]["freqs"]

    def mode_amplitudes(self, monitor: str, mode: int = 0):
        """(a_plus, a_minus) complex arrays over frequency."""
        import numpy as np

        m = self.raw["modes"][monitor]
        p = m["plus"][mode]
        q = m["minus"][mode]
        return np.asarray(p["re"]) + 1j * np.asarray(p["im"]), np.asarray(q["re"]) + 1j * np.asarray(q["im"])

    def time_series(self, monitor: str, comp: str):
        """``(times, values)`` of one component of a ``ProbeMonitor`` or
        ``FieldTimeMonitor``: the sample times in seconds and an array indexed
        ``[sample, x, y, z]``. The times are those of the electric field; magnetic
        samples are half a time step earlier and are multiplied by the impedance
        of free space.
        """
        t = self.raw["time"][monitor]
        return t["times"], t["fields"][comp]

    def gain_levels(self):
        """Gain media: the number of atoms in levels 0, 1, 2 and 3 at the end
        of the run, summed over all gain cells (density times the volume of
        each E-field sample), as a numpy array; ``None`` without gain media.
        For four-level two-electron media these are electrons (two per
        atom)."""
        import numpy as np

        v = self.raw.get("gain_levels")
        return np.asarray(v) if v else None

    def far_field(self, monitor: str) -> dict:
        """Far field of a :func:`FarFieldMonitor`, as a dict of numpy arrays:

        - ``freqs`` (Hz), ``theta``, ``phi`` (radians): the axes;
        - ``intensity``: radiant intensity ``U`` in W/sr, shape
          ``[frequency, theta, phi]``; its integral over the sphere,
          ``sum U sin(theta) dtheta dphi``, is the radiated power;
        - ``e_theta``, ``e_phi``: complex far-field amplitudes in volts, same
          shape. Far from the box the electric field is
          ``(e_theta theta^ + e_phi phi^) exp(i k r) / r`` with ``r`` the
          distance from the coordinate origin, and
          ``U = (|e_theta|^2 + |e_phi|^2) / (2 eta)`` with ``eta`` the wave
          impedance of the outer medium, 376.73 ohm / ``n_medium``;
        - ``flux``: power through the box per frequency (W), for comparison
          with the integral of ``U``.

        Like every frequency-domain result, the values scale with the source
        spectrum (fields with ``S``, intensity and flux with ``abs(S)**2``).
        """
        import numpy as np

        f = self.raw["far_field"][monitor]
        out = {k: np.asarray(f[k]) for k in ("freqs", "theta", "phi", "intensity", "flux")}
        out["e_theta"] = np.asarray(f["e_theta_re"]) + 1j * np.asarray(f["e_theta_im"])
        out["e_phi"] = np.asarray(f["e_phi_re"]) + 1j * np.asarray(f["e_phi_im"])
        return out

    def energy(self, monitor: str):
        """``(times, energy)`` of an :func:`EnergyMonitor`: the sample times
        in seconds and the electromagnetic energy in the monitor box in
        joules, as numpy arrays."""
        e = self.raw["energy"][monitor]
        return e["times"], e["energy"]

    def source_spectrum(self, name: str | None = None, freqs=None, monitor: str | None = None):
        """Spectrum of a source, for normalising monitor data.

        Returns ``(freqs, spectrum)``: the frequencies in Hz and the complex
        Fourier transform ``S`` of the source's signal at those frequencies,
        in the order given. Dividing a monitor field by ``S``, or a power by
        ``abs(S)**2``, gives the response per unit source ``amplitude``.

        Parameters
        ----------
        name : str, optional
            Name of the source; unnamed sources are ``"source0"``,
            ``"source1"``, ... May be omitted if there is one source.
        freqs : sequence of float, optional
            Frequencies (Hz) to evaluate at, in any order.
        monitor : str, optional
            Evaluate at the frequencies of this frequency-domain monitor, in
            the monitor's order, so that the result lines up with
            ``flux(monitor)``, ``field(monitor, ...)`` or
            ``mode_amplitudes(monitor)``.

        With neither ``freqs`` nor ``monitor``, the frequencies are those of
        the frequency-domain monitors if they all record the same list in the
        same order. If they differ, a ``ValueError`` asks for one of the two
        arguments, because a spectrum on some other grid would be divided
        into the wrong monitor samples without any sign of it.

        ``S`` is computed from the time samples of the run that was made
        (same time step and number of steps), so it includes the truncation
        of a run that stopped early. For a source with a complex field
        pattern in a run with complex fields (an oblique plane wave with
        Bloch boundaries) it is the spectrum of the complex signal that was
        injected, which is twice that of the real pulse.
        """
        import numpy as np

        from .. import _core

        sources = self.raw.get("sources", [])
        if name is None:
            if len(sources) != 1:
                raise ValueError(f"the run has {len(sources)} sources ({[q['name'] for q in sources]}); name one")
            src = sources[0]
        else:
            match = [q for q in sources if q["name"] == name]
            if not match:
                raise KeyError(f"no source named {name!r}; the sources are {[q['name'] for q in sources]}")
            src = match[0]
        if freqs is not None and monitor is not None:
            raise ValueError("give freqs or monitor, not both")
        if monitor is not None:
            freqs = self.freqs(monitor)
        elif freqs is None:
            lists = {m: np.asarray(d["freqs"], dtype=float) for m, d in self.raw["dft"].items()}
            if not lists:
                raise ValueError("the run has no frequency-domain monitor; pass freqs=")
            first = next(iter(lists.values()))
            if any(len(f) != len(first) or np.any(f != first) for f in lists.values()):
                raise ValueError(
                    "the monitors record different frequency lists ("
                    + ", ".join(f"{m}: {len(f)}" for m, f in lists.items())
                    + "); pass monitor=<name> or freqs=<frequencies> to say which the spectrum is for"
                )
            freqs = first
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        re, im = _core.signal_spectrum(src["signal"], f.tolist(), float(self.dt), int(self.stats["steps"]), bool(src["analytic"]))
        return f, np.asarray(re) + 1j * np.asarray(im)

    def __repr__(self) -> str:
        s = self.stats
        return f"<lightsim.Results {s['cells']} cells x {s['steps']} steps on {s['backend']}: {s['mcells_per_s']:.0f} Mcells/s>"
