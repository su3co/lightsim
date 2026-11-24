"""Monitors."""

from __future__ import annotations

from ._util import _v3


def FieldTimeMonitor(name, min, max, components=("Ex", "Ey", "Ez"), interval=1, start=0.0) -> dict:
    """Fields in a box at regular time intervals.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.time_series(name, component)``.
    min, max : (x, y, z)
        Corners of the box (m). A zero extent records a plane, a line or a point.
    components : sequence of str
        Field components to record, from ``"Ex"`` ... ``"Hz"``.
    interval : int
        Record every ``interval`` steps.
    start : float
        Time (s) of the first sample.

    Returns the monitor dict. All samples stay in memory until the end of the
    run: 4 bytes per cell, component and sample.
    """
    return {"type": "field_time", "name": name, "min": _v3(min), "max": _v3(max), "components": list(components), "interval": interval, "start": start}


def ProbeMonitor(name, position, components=("Ez",)) -> dict:
    """Fields at one point at every time step.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.time_series(name, component)``.
    position : (x, y, z)
        Position (m); each component is taken at its nearest grid position.
    components : sequence of str
        Field components to record.

    Returns the monitor dict.
    """
    return {"type": "probe", "name": name, "position": _v3(position), "components": list(components)}


def DFTMonitor(name, min, max, freqs, components=("Ex", "Ey", "Ez"), stride=None, apodization=None) -> dict:
    """Frequency-domain fields in a box, a plane, a line or a point.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.field(name, component)``.
    min, max : (x, y, z)
        Corners of the box (m). Along an axis of zero extent the nearest layer of
        each component is taken.
    freqs : sequence of float
        Frequencies (Hz).
    components : sequence of str
        Field components to record, from ``"Ex"`` ... ``"Hz"``.
    stride : int, optional
        Accumulate the Fourier sums every ``stride`` steps. ``None`` takes
        ``Simulation.dft_stride``, which by default is chosen from the bandwidth
        of the sources so that the result equals that of every step.
    apodization : dict, optional
        Time window of the sums: keys ``start``, ``end`` (s) and ``width`` (s)
        of a Gaussian taper at each edge, all optional.

    Returns the monitor dict. The monitor stores two 4-byte numbers per sample,
    component and frequency.
    """
    return {
        "type": "dft_field",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "components": list(components),
        "freqs": [float(f) for f in freqs],
        "stride": 0 if stride is None else int(stride),
        "apodization": apodization or {},
    }


def FluxMonitor(name, axis, position, min, max, freqs, apodization=None) -> dict:
    """Power through a plane, per frequency.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.flux(name)``.
    axis, position : int, float
        The plane is normal to ``axis`` at this coordinate (m), moved to the
        nearest grid node.
    min, max : (x, y, z)
        Corners limiting the plane (m); they may lie outside the region.
    freqs : sequence of float
        Frequencies (Hz).
    apodization : dict, optional
        Time window of the Fourier sums, as for ``DFTMonitor``.

    Returns the monitor dict. The power is that of the time-averaged Poynting
    vector of the recorded spectra, in watts times the squared source spectrum,
    positive towards ``+axis``.
    """
    return {
        "type": "flux",
        "name": name,
        "axis": axis,
        "position": position,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "apodization": apodization or {},
    }


def FluxBoxMonitor(name, min, max, freqs) -> dict:
    """Net power leaving a closed box, per frequency.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.flux(name)``.
    min, max : (x, y, z)
        Corners of the box (m), snapped to grid nodes.
    freqs : sequence of float
        Frequencies (Hz).

    Returns the monitor dict. The result is positive for a source inside the
    box, negative for absorption inside it; around a ``TFSF`` box it is the
    scattered power. A face on an electric or magnetic wall is skipped.
    """
    return {"type": "flux_box", "name": name, "min": _v3(min), "max": _v3(max), "freqs": [float(f) for f in freqs]}


def ModeMonitor(name, axis, position, min, max, freqs, modes) -> dict:
    """Amplitudes of waveguide modes on a plane window.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.mode_amplitudes(name, mode)``.
    axis, position : int, float
        The plane is normal to ``axis`` at this coordinate (m), moved to the
        nearest grid node.
    min, max : (x, y, z)
        Corners of the mode window (m): those given to the mode solver.
    freqs : sequence of float
        Frequencies (Hz).
    modes : sequence of dict
        Mode profiles (``Mode.profile``, or from ``gaussian_mode``) solved on
        this window. If there are as many profiles as frequencies, and more
        than one, they are taken as one mode with one profile per frequency.

    Returns the monitor dict. For unit-power profiles ``abs(a)**2`` is the power
    in the mode. Monitors on one plane with the same frequencies share the
    fields they record, so many of them cost what one costs.
    """
    return {
        "type": "mode",
        "name": name,
        "axis": axis,
        "position": position,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "modes": list(modes),
    }


def FarFieldMonitor(name, min, max, freqs, theta, phi, n_medium: float = 1.0, apodization=None) -> dict:
    """Far field radiated by everything inside a box.

    The fields on the six faces of the box ``min``..``max`` (m, snapped to
    grid nodes) are recorded at the frequencies ``freqs`` (Hz), turned into
    equivalent surface currents and radiated into a homogeneous medium of
    refractive index ``n_medium``. Read the result with
    ``Results.far_field(name)``.

    Parameters
    ----------
    theta, phi : sequences of float
        Directions (radians): ``theta`` is the polar angle from the +z axis,
        ``phi`` the azimuth from the +x axis in the xy plane. The pattern is
        returned on the grid ``theta`` x ``phi``.
    n_medium : float
        Index of the medium outside the box, which must be homogeneous: a
        substrate or other layers outside the box are not accounted for.
    apodization : dict, optional
        Time window of the Fourier sums, as for :func:`DFTMonitor`.

    The box must be closed, lie inside the simulation region and enclose
    all sources and scatterers whose radiation is wanted.
    """
    return {
        "type": "far_field",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "theta": [float(t) for t in theta],
        "phi": [float(p) for p in phi],
        "n_medium": float(n_medium),
        "apodization": apodization or {},
    }


def EnergyMonitor(name, min, max, interval=1) -> dict:
    """Electromagnetic energy stored in a box, versus time.

    Every ``interval`` steps the monitor evaluates

        U = eps0/2 * sum( eps |E|^2 + mu |eta0 H|^2 ) dV      (joules)

    over the cells of the box ``min``..``max`` (m), with the permittivity
    of each cell as the engine uses it. Cells cut by the box count with the
    part inside it. Read the result with ``Results.energy(name)``.

    The electric field is sampled at the end of a step and the magnetic field
    half a step earlier, so the sum ripples at twice the optical frequency by
    a fraction of about ``(omega dt / 2)**2``. For a dispersive medium ``eps``
    is its high-frequency value: the energy stored in the polarisation of the
    poles is not included. The six field components of the box are kept in
    memory for every sample until the end of the run (24 bytes per cell and
    sample), so keep the box small or the interval long.
    """
    return {"type": "energy", "name": name, "min": _v3(min), "max": _v3(max), "interval": int(interval)}
