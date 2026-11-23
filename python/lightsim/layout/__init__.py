"""Use lightsim from any layout tool: GDSII + technology + ports in,
S-parameters (Touchstone, INTERCONNECT text, npz, SAX) or simulation specs out.

    import lightsim.layout as ll
    sp = ll.sparameters("chip.gds", cell="mmi", technology="stack.toml", ports="ports.csv")
    sp.to_touchstone("mmi.s3p")

Nothing here needs gdsfactory: the GDSII file is read by lightsim's own
reader (:func:`read_gds`), the ports come from a port file or a pin
convention (:func:`read_ports`), and the layer stack from a technology file
(:func:`read_technology`). The simulation is set up as
:mod:`lightsim.gplugins` sets it up (same port planes, mode sources and
monitors), by the same code.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from ..sparameters import SParameters
from ..spec import C0, um
from .gds import Layout, read_gds
from .ports import read_port_file, read_ports, siepic, pins, write_port_file
from .technology import Level, Material, Technology, read_lyp, read_technology, technology_from_gdsfactory

__all__ = [
    "Layout",
    "Level",
    "Material",
    "SParameters",
    "Technology",
    "pins",
    "read_gds",
    "read_lyp",
    "read_port_file",
    "read_ports",
    "read_technology",
    "simulation",
    "siepic",
    "sparameters",
    "technology_from_gdsfactory",
    "write_port_file",
    "write_specs",
]


def _setup(gds, technology, ports, cell, wavelengths, is_3d, effective_index, polarization, resolution, steps_per_wavelength, margin, z_margin, port_margin, port_source_offset, port_monitor_offset, distance_source_to_monitors, decay_by, background, tpml):
    from ._build import PortRuns, build_from_levels, extend_ports, gaussian_signal, mesh_and_pml

    layout = gds if isinstance(gds, Layout) else read_gds(gds, cell)
    tech = technology if isinstance(technology, Technology) else read_technology(technology)
    plist = read_ports(ports, layout, gds_path=getattr(layout, "path", None), cell=cell)
    if any(l.sidewall_angle for l in tech.levels):
        warnings.warn("sidewall_angle is not modelled: lightsim extrudes vertical walls", stacklevel=3)
    wl = np.asarray(wavelengths if wavelengths is not None else np.linspace(1.5e-6, 1.6e-6, 21), dtype=float).reshape(-1)
    freqs = C0 / wl
    mesh, pml_layers, pml_len = mesh_and_pml(wl, resolution or 1.0, None if resolution else steps_per_wavelength, tpml)
    (x0, y0), (x1, y1) = layout.bbox()
    m = margin * um
    region = ((x0 - m, y0 - m), (x1 + m, y1 + m))
    levels = tech.stack_layers(layout)
    port_levels = {p["name"]: tech.level_of_layer(p["layer"]) for p in plist if p.get("layer") is not None}
    levels = extend_ports(levels, plist, m + pml_len + 1e-6, {k: v for k, v in port_levels.items() if v})
    sim, zr = build_from_levels(
        levels,
        region,
        wl,
        tech.material_index(float(np.mean(wl))),
        is_3d=is_3d,
        effective_index=effective_index,
        polarization=polarization,
        mesh=mesh,
        pml_layers=pml_layers,
        zmargin_top=z_margin,
        zmargin_bot=z_margin,
        background=(background or tech.background_material()).lower(),
        decay_by=decay_by,
    )
    modes_of = {p["name"]: p.get("modes") or [0] for p in plist}
    port_modes = sorted({m for ms in modes_of.values() for m in ms})
    for p in plist:  # the gplugins helpers expect only these keys; z is used by the planes below
        p.setdefault("z", None)
    pr = PortRuns(
        sim,
        zr,
        plist,
        port_margin=port_margin * um,
        port_monitor_offset=port_monitor_offset * um,
        port_source_offset=port_source_offset * um,
        gap=distance_source_to_monitors * um,
        port_modes=port_modes,
        port_source_modes=modes_of,
        freqs=freqs,
        signal=gaussian_signal(freqs),
    )
    return pr, plist, modes_of, wl


_DEFAULTS = dict(
    cell=None,
    wavelengths=None,
    is_3d=True,
    effective_index=True,
    polarization=None,
    resolution=None,
    steps_per_wavelength=20.0,
    margin=1.0,
    z_margin=0.8,
    port_margin=1.0,
    port_source_offset=-0.1,
    port_monitor_offset=-0.1,
    distance_source_to_monitors=0.2,
    decay_by=1e-5,
    background=None,
    tpml=None,
)


def simulation(gds, technology, ports, sources=None, **kwargs) -> dict:
    """The lightsim simulations of a layout, one per source port and mode.

    Returns ``{"o1@0": Simulation, ...}``, each with its mode source and the
    mode monitors of all ports (the port mode profiles are in the source and
    monitor entries of ``sim.to_json()``). ``sources`` limits the sources to
    the ports (``"o1"``) or port modes (``"o1@1"``) listed. The other
    arguments are those of :func:`sparameters`; nothing is run.
    """
    opts = {**_DEFAULTS, **kwargs}
    pr, plist, modes_of, _ = _setup(gds, technology, ports, **opts)
    out = {}
    for p in plist:
        for m in modes_of[p["name"]]:
            label = f"{p['name']}@{m}"
            if sources is not None and p["name"] not in sources and label not in sources:
                continue
            out[label] = pr.build(p["name"], m)[0]
    return out


def write_specs(sims: dict, out) -> list[Path]:
    """Write the specs of :func:`simulation` as lightsim JSON: ``out`` itself
    for one simulation, else ``<stem>_<port>@<mode>.json`` next to it."""
    out = Path(out)
    paths = []
    for label, sim in sims.items():
        p = out if len(sims) == 1 else out.with_name(f"{out.stem}_{label}{out.suffix or '.json'}")
        p.write_text(sim.to_json())
        paths.append(p)
    return paths


def sparameters(gds, technology, ports, device="auto", progress: int = 0, **kwargs) -> SParameters:
    """S-parameters of a cell of a GDSII file, as an :class:`SParameters`.

    Parameters
    ----------
    gds
        Path of the GDSII file (or a :class:`Layout`).
    technology
        Technology file (``.toml``, ``.yaml``, ``.json``) or :class:`Technology`.
    ports
        Port file (``.json``, ``.yaml``, ``.csv``), a list of port dicts
        (port-file keys, µm), ``"siepic"``, ``"gdsfactory"`` or
        ``{"convention": "pins", "pin_layer": ..., "label_layer": ...,
        "port_layer": ...}``.
    cell
        Cell to simulate (default: the only top cell).
    wavelengths
        Array of vacuum wavelengths in metres (default 1.50-1.60 µm, 21 points).
    is_3d
        ``True`` (default) for 3-D; ``False`` for the 2-D model chosen by
        ``effective_index`` (effective-index levels, or bulk indices) and
        ``polarization``.
    resolution, steps_per_wavelength
        A uniform mesh of ``resolution`` cells per µm, or (default) a graded
        mesh of ``steps_per_wavelength`` steps per wavelength in each material.
    margin, z_margin
        Distance from the layout to the absorber in x/y and in z (µm).
    port_margin, port_source_offset, port_monitor_offset, distance_source_to_monitors
        Mode windows and the placement of sources and monitors, as in
        :func:`lightsim.gplugins.write_sparameters_lightsim` (µm; offsets
        positive outwards).
    decay_by
        The runs stop when the field energy has decayed by this factor.
    background
        The medium around the stack (default: from the technology).
    device
        ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``"cuda:0,1"`` or a list of
        devices over which the sources run concurrently.
    """
    opts = {**_DEFAULTS, **kwargs}
    pr, plist, modes_of, wl = _setup(gds, technology, ports, **opts)
    runs = [(p["name"], m) for p in plist for m in modes_of[p["name"]]]
    parts = pr.run(runs, device=device, progress=progress)
    d = {"wavelengths": wl}
    for part in parts:
        d.update(part)
    meta = {"cell": opts["cell"] or "", "model": "3-D" if opts["is_3d"] else ("2-D effective index" if opts["effective_index"] else "2-D bulk index")}
    return SParameters.from_dict(d, "m", meta=meta)
