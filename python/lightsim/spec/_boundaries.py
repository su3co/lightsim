"""Boundary conditions and meshes."""

from __future__ import annotations

from typing import Sequence

from ._util import _v3


def PML(
    layers: int = 12,
    order: float = 3.0,
    sigma_scale: float = 1.0,
    kappa_max: float = 2.0,
    alpha_max: float = 0.0,
    inside: bool = False,
    stabilize: bool = True,
) -> dict:
    """CPML boundary. ``inside=True`` puts the layers on the outermost cells of
    the region (FDTDX convention) instead of adding them outside.

    ``stabilize`` (on by default) changes, inside the layers only and when the
    region holds a structure of another medium than the background, the
    lossless poles of transparent media that lie outside the band of the
    sources: the far-infrared pole of a fitted glass
    (:func:`lightsim.materials.fitted_medium`) is damped, and a strong
    ultraviolet pole (silicon fitted with ``eps_inf = 1``) is frozen at its
    value at the centre of the band. Without it a waveguide of such media
    running into the layer makes a long run diverge: the medium is metal-like
    in a band above each pole, the guide carries backward modes there, and a
    PML amplifies backward waves. The change reflects about 1e-7 of the power
    at the face of the layer for fused silica near 1.55 um, and at most 1e-4
    at the ends of a wide band; poles in or near the band are left as they
    are. Damping the infrared pole has removed the growth in every case
    tried; freezing the ultraviolet pole removes it for silicon guides in a
    glass cladding and only delays it for some other structures (the
    boundaries guide has the measurements). :func:`Absorber` is the boundary
    that cannot amplify."""
    return {
        "type": "pml",
        "layers": layers,
        "params": {"order": order, "sigma_scale": sigma_scale, "kappa_max": kappa_max, "alpha_max": alpha_max, "stabilize": stabilize},
        "inside": inside,
    }


def Absorber(layers: int = 40, order: float = 3.0, sigma_scale: float = 1.0) -> dict:
    """Adiabatic absorber: a boundary that can replace a PML where a PML is
    unstable. ``layers`` cells are added outside the region, in which an
    electric and a magnetic conductivity rise smoothly (as the power ``order``
    of the depth) from zero. The two are matched to each other in every
    medium, at the centre of the source band, so a wave decays without a
    change of its impedance or of its mode profile.

    An absorber is a passive medium: it cannot amplify any wave. A PML can:
    it amplifies backward waves, which fitted (dispersive) media with a band
    of negative permittivity, metals with surface plasmons, and some photonic
    crystal guides carry (:func:`PML` describes the usual case and what
    ``stabilize`` does about it). Use an absorber on the faces that such a
    structure runs through when a run with PML diverges.

    The price is thickness. An absorber reflects wherever its conductivity
    changes, so the change must be slow on the scale of the wavelength. At
    1.55 um a layer 1 um thick (one wavelength in a glass cladding: 40 layers
    of 25 nm) reflects 1e-5 to 1e-4 of a guided mode, 0.5 um about 5e-4, and
    1.5 um as little as a 12-layer PML (a few 1e-6): choose ``layers`` from
    the thickness. The boundaries guide gives the measured values. ``sigma_scale`` multiplies the peak conductivity: the default
    attenuates a plane wave in the background medium by 6 nepers in amplitude
    (1e-5 in power) on the way to the outer wall and back, and a guided mode
    in proportion to its group index.

    Absorbers and PML can be mixed, face by face. Only media with a diagonal
    permittivity, conductivity and per-axis poles can run through an
    absorber; the fixed-angle engine and the ``reverse`` gradient engine do
    not take one, and design regions must stay outside it."""
    return {"type": "absorber", "layers": layers, "params": {"order": order, "sigma_scale": sigma_scale}}


PERIODIC = {"type": "periodic"}
PEC_WALL = {"type": "pec"}
PMC_WALL = {"type": "pmc"}


def Bloch(k: float) -> dict:
    """Bloch-periodic boundary condition.

    The field at the upper face of the axis is the field at the lower face times
    ``exp(1j * k * L)``, with ``L`` the length of the region along the axis.

    Parameters
    ----------
    k : float
        Wavevector component along the axis (rad/m); ``n k0 sin(theta)`` for a
        plane wave incident at ``theta`` in a medium of index ``n``.

    Returns the boundary dict, to be set on both faces of the axis with
    ``Simulation.set_boundaries``. A run with ``k != 0`` uses complex fields.
    """
    return {"type": "bloch", "k": k}


def UniformMesh(dx: float | Sequence[float]) -> dict:
    """Uniform mesh.

    Parameters
    ----------
    dx : float or (dx, dy, dz)
        Cell size (m), one value or one per axis. It is adjusted so that a whole
        number of cells fits the region along each axis. In a 2-D simulation
        ``dx`` may hold the two in-plane spacings; the spacing along the
        invariant axis is not used.

    Returns the mesh dict for ``Simulation``.
    """
    d = [float(dx)] * 3 if isinstance(dx, (int, float)) else [float(v) for v in dx]
    return {"type": "uniform", "dx": d}


def AutoMesh(wavelength: float, steps_per_wavelength: float = 20.0, grading: float = 1.3, min_step=None, max_step=None, overrides=(), symmetry=None) -> dict:
    """Automatic graded mesh.

    The cell size in each medium is ``wavelength / (n * steps_per_wavelength)``
    with ``n`` the refractive index of the medium, and it grows away from a
    structure by at most the factor ``grading`` per cell.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength (m) the resolution refers to; use the shortest
        wavelength of interest.
    steps_per_wavelength : float
        Cells per wavelength in each medium.
    grading : float
        Largest ratio of neighbouring cell sizes, about.
    min_step, max_step : float, optional
        Lower bound (m) on the cell size chosen for structures, upper bound
        on the cell size of the background.
    overrides : sequence of (min, max, (dx, dy, dz))
        Boxes with an explicit largest cell size (m) per axis.
    symmetry : (x0, y0, z0), optional
        Mirror planes of the mesh, one coordinate (m) per axis or ``None``.
        A plane gets a grid node and the cells on its two sides are mirror
        images of each other. Without it the mesh is built outwards from the
        centre of the region along each axis, which is already symmetric for
        structures that are symmetric about that centre; name a plane when
        the symmetry plane of the device is somewhere else.

    Returns the mesh dict for :class:`Simulation`.
    """
    sym = [None, None, None] if symmetry is None else [None if v is None else float(v) for v in symmetry]
    if len(sym) != 3:
        raise ValueError("symmetry needs three entries (x0, y0, z0), each a coordinate or None")
    return {
        "type": "auto",
        "wavelength": wavelength,
        "steps_per_wavelength": steps_per_wavelength,
        "grading": grading,
        "min_step": min_step,
        "max_step": max_step,
        "overrides": [{"min": _v3(o[0]), "max": _v3(o[1]), "dx": _v3(o[2])} for o in overrides],
        "symmetry": sym,
    }


def ExplicitMesh(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> dict:
    """Mesh given by its node coordinates.

    ``x``, ``y`` and ``z`` are increasing lists of the cell edges (m) along
    each axis, at least two per axis. The first and the last node of each
    list should be the ``min`` and ``max`` of the simulation region;
    absorbing layers are added outside them with the size of the outermost
    cell. :meth:`Simulation.frozen_mesh` returns the grid of a simulation in
    this form.
    """
    out = {"type": "explicit"}
    for k, v in zip("xyz", (x, y, z)):
        nodes = [float(c) for c in v]
        if len(nodes) < 2 or any(b <= a for a, b in zip(nodes[:-1], nodes[1:])):
            raise ValueError(f"explicit mesh: the {k} nodes must be at least two increasing coordinates")
        out[k] = nodes
    return out
