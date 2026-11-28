"""Time signals and sources."""

from __future__ import annotations

from ._util import _v3, Vec3


def GaussianPulse(freq0: float, fwidth: float, offset: float = 5.0, phase: float = 0.0, amplitude: float = 1.0) -> dict:
    """Gaussian pulse, the usual signal for a spectrum in one run.

    ``amplitude * cos(2 pi freq0 (t - t0) + phase) * exp(-(t - t0)**2 / (2 tau**2))``
    with ``tau = 1 / (2 pi fwidth)`` and ``t0 = offset * tau``. Its spectrum is a
    Gaussian of standard deviation ``fwidth`` around ``freq0``.

    Parameters
    ----------
    freq0 : float
        Centre frequency (Hz).
    fwidth : float
        Spectral width (Hz), the standard deviation of the spectrum.
    offset : float
        Peak time in units of ``tau``; 5 starts the pulse from a negligible value.
    phase : float
        Carrier phase (rad).
    amplitude : float
        Peak value, a factor on the source amplitude.

    Returns the signal dict for a source.
    """
    return {"type": "gaussian_pulse", "freq0": freq0, "fwidth": fwidth, "offset": offset, "phase": phase, "amplitude": amplitude}


def ContinuousWave(freq0: float, fwidth: float, offset: float = 5.0, phase: float = 0.0, amplitude: float = 1.0) -> dict:
    """Continuous wave at one frequency, switched on smoothly.

    ``amplitude * cos(2 pi freq0 t + phase)``, multiplied before ``t0 = offset * tau``
    by a Gaussian ramp of width ``tau = 1 / (2 pi fwidth)``.

    Parameters
    ----------
    freq0 : float
        Frequency (Hz).
    fwidth : float
        Sets the duration of the turn-on, ``tau = 1 / (2 pi fwidth)`` (Hz).
    offset : float
        End of the ramp in units of ``tau``.
    phase : float
        Phase (rad).
    amplitude : float
        Amplitude, a factor on the source amplitude.

    Returns the signal dict for a source. The signal never ends, so ``shutoff``
    has no effect and frequency-domain monitors sample every step.
    """
    return {"type": "continuous_wave", "freq0": freq0, "fwidth": fwidth, "offset": offset, "phase": phase, "amplitude": amplitude}


def PointDipole(position: Vec3, polarization: str, signal: dict, amplitude: float = 1.0, name: str = "") -> dict:
    """Point dipole at a single field sample.

    Parameters
    ----------
    position : (x, y, z)
        Position (m); the dipole is placed on the grid position of the component
        ``polarization`` nearest to it.
    polarization : str
        ``"Ex"``, ``"Ey"`` or ``"Ez"``: an electric current element of moment
        ``amplitude * signal(t)`` in A m. ``"Hx"``, ``"Hy"`` or ``"Hz"``: a
        magnetic current element in V m.
    signal : dict
        Time signal.
    amplitude : float
        Dipole moment amplitude (A m, or V m for a magnetic dipole).
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. An electric dipole of moment ``p`` in a medium of
    index ``n`` radiates ``n eta0 k0**2 abs(p)**2 / (12 pi)`` watts, with
    ``k0 = 2 pi f / c``.
    """
    return {
        "type": "point_dipole",
        "name": name,
        "position": _v3(position),
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
    }


def PlaneWave(
    axis: int,
    position: float,
    direction: str,
    polarization: str,
    signal: dict,
    amplitude: float = 1.0,
    angle: float = 0.0,
    min: Vec3 | None = None,
    max: Vec3 | None = None,
    name: str = "",
    fixed_angle: bool = False,
) -> dict:
    """One-sided plane-wave source on a plane.

    The wave is launched towards ``direction`` only; behind the plane there is
    no incident field.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` (0, 1, 2) at this coordinate (m),
        moved to the nearest grid node.
    direction : "+" or "-"
        Propagation direction along ``axis``.
    polarization : str
        A field component in the plane. An electric component (``"Ey"`` for
        ``axis=0``) gives E along that axis; with ``angle`` set the wave is
        s-polarised. A magnetic component (``"Hz"`` for ``axis=0``) gives H
        along that axis and E along the other in-plane axis; with ``angle``
        set the wave is p-polarised.
    signal : dict
        Time signal.
    amplitude : float
        Peak electric field (V/m), times the signal.
    angle : float
        Tilt of the wavevector (rad) towards the in-plane axis perpendicular to
        the polarisation axis: s-polarisation for an E component,
        p-polarisation for an H component (then E has the tangential part
        ``amplitude * cos(angle)`` and a normal part ``amplitude * sin(angle)``).
        It needs Bloch boundaries along the tilt axis with
        ``k = n k0 sin(angle)``, and a ``GaussianPulse`` or ``ContinuousWave``
        signal. The angle is exact at the centre frequency of the signal.
    min, max : (x, y, z), optional
        Corners limiting the extent of the plane (m); the whole plane by default.
    name : str
        Name, for ``Results.source_spectrum``.
    fixed_angle : bool
        Hold ``angle`` at every frequency of the pulse instead of only at its
        centre (broadband oblique incidence in one run). The boundaries along
        the tilt axis must then be ``PERIODIC`` (the run imposes the phase of
        each frequency itself), the other transverse axis periodic, and the
        source must be the only one; any real signal works. The run solves
        for the transformed fields ``E(r, t + n sin(angle) u / c)`` and
        returns physical frequency-domain data. See "Broadband oblique
        incidence at a fixed angle" in the sources guide for what it
        supports and costs.

    Returns the source dict. The index of the medium is taken at the centre of
    the plane; in a medium of index ``n`` the wave carries
    ``n abs(amplitude * S)**2 cos(angle) / (2 eta0)`` watts per square metre of
    the plane.
    """
    return {
        "type": "plane_wave",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
        "angle": angle,
        "min": None if min is None else _v3(min),
        "max": None if max is None else _v3(max),
        **({"fixed_angle": True} if fixed_angle else {}),
    }


def GaussianBeam(
    axis: int,
    position: float,
    direction,
    polarization: str,
    center: Vec3,
    waist: float,
    signal: dict,
    amplitude: float = 1.0,
    focus_distance: float = 0.0,
    name: str = "",
    theta: float = 0.0,
    phi: float = 0.0,
) -> dict:
    """One-sided Gaussian-beam source, at any angle to its plane.

    The beam is injected on the plane normal to ``axis`` at ``position`` and
    travels away from it towards ``+axis`` or ``-axis``.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` (0, 1, 2) at this coordinate
        (m), moved to the nearest grid node.
    direction : "+", "-" or a vector
        Side of the plane the beam is launched to. A vector ``(dx, dy, dz)``
        gives the direction of the beam axis itself; its component along
        ``axis`` sets the side, and ``theta`` and ``phi`` are computed from
        it (they must then be left at 0).
    polarization : str
        An electric component in the plane, e.g. ``"Ey"`` for ``axis=0``.
        The electric field of the beam points along the projection of this
        axis onto the plane normal to the beam axis: exactly along it when
        the tilt is perpendicular to it (s polarisation), tilted with the
        beam otherwise (p polarisation).
    center : (x, y, z)
        A point of the beam axis; its two in-plane coordinates are where the
        axis crosses the source plane (m).
    waist : float
        Radius (m) at which the field falls to 1/e at the focus.
    signal : dict
        Time signal; a tilted or defocused beam needs ``GaussianPulse`` or
        ``ContinuousWave``. The pattern is that of the centre frequency.
    amplitude : float
        Peak electric field at the focus (V/m), times the signal.
    focus_distance : float
        Distance (m) from ``center`` to the focus along the beam axis;
        negative for a beam that is already diverging at the plane. The waist
        plane is normal to the beam axis.
    theta, phi : float
        Polar angle (rad) between the beam axis and the plane normal, below
        pi/2, and azimuth of the tilt, measured from the first in-plane axis
        towards the second. The in-plane axes are (y, z) for ``axis=0``,
        (x, z) for ``axis=1`` and (x, y) for ``axis=2``.

    The pattern is the paraxial Gaussian beam: accurate for waists of about a
    wavelength or more. It spans the whole plane, so the region must be
    several waists wide where the beam crosses it. In a simulation that is
    periodic along an in-plane axis with a period below twice the waist (a
    two-dimensional simulation) the beam is uniform along that axis, a sheet
    beam, and must travel in the other plane. The beam carries the power
    ``n |amplitude S|^2 pi waist^2 / (4 eta0)``, or
    ``n |amplitude S|^2 waist sqrt(pi/2) / (2 eta0)`` per unit length of a
    sheet beam, whatever its angle.
    """
    if not isinstance(direction, str):
        import math

        if theta or phi:
            raise ValueError("give the beam direction as a vector, or as '+'/'-' with theta and phi, not both")
        d = _v3(direction)
        norm = math.sqrt(sum(c * c for c in d))
        if norm == 0.0 or d[axis] == 0.0:
            raise ValueError("the direction vector of a Gaussian beam needs a component along the source axis")
        u, v = [i for i in range(3) if i != axis]
        theta = math.acos(min(1.0, abs(d[axis]) / norm))
        phi = math.atan2(d[v], d[u]) if theta > 0 else 0.0
        direction = "+" if d[axis] > 0 else "-"
    return {
        "type": "gaussian_beam",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "center": _v3(center),
        "waist": waist,
        "signal": signal,
        "amplitude": amplitude,
        "focus_distance": focus_distance,
        "theta": float(theta),
        "phi": float(phi),
    }


def ModeSource(axis, position, direction, min, max, profile, signal, amplitude=1.0, name="") -> dict:
    """One-sided waveguide-mode source on a plane window.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` at this coordinate (m), moved to
        the nearest grid node.
    direction : "+" or "-"
        Direction in which the mode is launched.
    min, max : (x, y, z)
        Corners of the mode window (m): those given to the mode solver.
    profile : dict
        Field pattern: ``Mode.profile`` from ``solve_modes`` or
        ``eim.solve_port_mode``, or a profile from ``gaussian_mode``. A
        profile from ``lightsim.profile_over_band`` is followed over the
        band of the pulse (``GaussianPulse`` or ``ContinuousWave``).
    signal : dict
        Time signal; a complex profile (a lossy mode, a tilted beam) needs
        ``GaussianPulse`` or ``ContinuousWave``.
    amplitude : float
        Factor on the profile. A profile from the mode solver carries unit
        power, so the launched power spectrum is ``abs(amplitude * S)**2`` watts.
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. The part of the window inside an absorbing layer
    is not injected.
    """
    return {
        "type": "mode",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "min": _v3(min),
        "max": _v3(max),
        "profile": profile,
        "signal": signal,
        "amplitude": amplitude,
    }


def TFSF(min: Vec3, max: Vec3, axis: int, direction: str, polarization: str, signal: dict, amplitude=1.0, name="") -> dict:
    """Total-field/scattered-field box with a plane wave inside.

    The incident wave exists inside the box only; outside it only the field
    scattered by what the box contains remains.

    Parameters
    ----------
    min, max : (x, y, z)
        Corners of the box (m), snapped to grid nodes.
    axis : int
        Propagation axis of the incident wave (normal incidence on the faces).
    direction : "+" or "-"
        Propagation direction along ``axis``.
    polarization : str
        Electric component of the wave, perpendicular to ``axis``.
    signal : dict
        Time signal.
    amplitude : float
        Peak electric field (V/m), times the signal.
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. The medium around the box must be uniform (it may
    be dispersive). A box that spans a periodic axis completely has no faces
    normal to that axis, which gives the incident wave of a 2-D problem.
    """
    return {
        "type": "tfsf",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "axis": axis,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
    }
