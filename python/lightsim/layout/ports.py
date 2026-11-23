"""Ports of a layout without gdsfactory objects.

:func:`read_ports` returns a list of port dicts (``name``, ``center`` (m),
``orientation`` (degrees, the direction in which light leaves the device),
``width`` (m), ``layer`` (``(layer, datatype)`` or ``None``), ``modes``
(list of mode numbers or ``None``), ``z`` (``(zmin, zmax)`` in m or
``None``)), from

* a port file (JSON, YAML or CSV; the format below, which is also gamma's),
* ``"siepic"``: SiEPIC's pins, paths on the PinRec layer 1/10 whose
  direction points out of the device, named by a text label on 1/10; the
  waveguide layer is Si 1/0 (layer numbers from SiEPIC EBeam PDK
  ``klayout/EBeam/EBeam.lyp``),
* ``{"convention": "pins", "pin_layer": ..., "label_layer": ...,
  "port_layer": ...}``: the same on other layers, where a pin may also be a
  box (its outward side is the one away from the centre of the layout),
* ``"gdsfactory"``: the port metadata gdsfactory writes into the GDS (needs
  the gdsfactory package, which reads it with KLayout).

Port file format, version 1 (lengths in micrometres unless ``units`` says
otherwise; keys not listed are ignored)::

    [{"name": "o1", "x": -10.0, "y": 0.0, "width": 0.5, "orientation": 180, "layer": [1, 0]}, ...]

or ``{"version": 1, "units": "um", "ports": [...]}``. ``orientation`` may be
given as ``direction`` instead, in degrees or as ``"east"``/``"+x"`` (0),
``"north"``/``"+y"`` (90), ``"west"``/``"-x"`` (180), ``"south"``/``"-y"``
(270). Optional: ``center`` (``[x, y]``, instead of ``x`` and ``y``),
``modes`` (mode numbers launched and measured at that port, default
``[0]``), ``z`` (``[zmin, zmax]`` of the mode window, default the
simulation height). CSV: a header line
``name,x,y,width,orientation,layer,datatype`` (plus optional ``modes``,
separated by ``;``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .gds import Layout, _layer

UNITS = {"m": 1.0, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "mm": 1e-3}
_PER_M = {"m": 1.0, "um": 1e6, "µm": 1e6, "nm": 1e9, "mm": 1e3}  # divisors: 5 / 1e6 is exactly 5e-06
DIRECTIONS = {"east": 0.0, "+x": 0.0, "right": 0.0, "north": 90.0, "+y": 90.0, "top": 90.0, "west": 180.0, "-x": 180.0, "left": 180.0, "south": 270.0, "-y": 270.0, "bottom": 270.0}

#: SiEPIC EBeam PDK layers (klayout/EBeam/EBeam.lyp): waveguide Si, PinRec, DevRec, Text.
SIEPIC = {"port_layer": (1, 0), "pin_layer": (1, 10), "label_layer": (1, 10), "devrec_layer": (68, 0), "text_layer": (10, 0)}


def _orientation(p: dict) -> float:
    v = p.get("orientation", p.get("direction"))
    if v is None:
        raise ValueError(f"port {p.get('name')!r} has neither orientation nor direction")
    if isinstance(v, str):
        key = v.strip().lower()
        if key in DIRECTIONS:
            return DIRECTIONS[key]
        v = float(key)
    return float(v) % 360.0


def normalize(p: dict, scale: float = 1e-6) -> dict:
    """One port dict of a port file -> lightsim's port dict (metres)."""
    per = {v: _PER_M[k] for k, v in UNITS.items()}.get(scale, 1.0 / scale)
    if "center" in p:
        x, y = p["center"]
    else:
        x, y = p["x"], p["y"]
    layer = p.get("layer")
    if layer is not None and not isinstance(layer, str) and np.ndim(layer) == 0:
        layer = (int(layer), int(p.get("datatype", 0)))
    out = {
        "name": str(p["name"]),
        "center": (float(x) / per, float(y) / per),
        "orientation": _orientation(p),
        "width": float(p["width"]) / per,
        "layer": _layer(layer) if layer is not None else None,
        "modes": [int(m) for m in p["modes"]] if p.get("modes") not in (None, "") else None,
        "z": (float(p["z"][0]) / per, float(p["z"][1]) / per) if p.get("z") is not None else None,
    }
    return out


def read_port_file(path) -> list[dict]:
    """Ports from a JSON, YAML or CSV port file (see the module notes)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        rows = []
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
                if not r.get("name"):
                    continue
                if r.get("layer") not in (None, ""):
                    r["layer"] = (int(r["layer"]), int(r.get("datatype") or 0))
                else:
                    r.pop("layer", None)
                if r.get("modes"):
                    r["modes"] = [int(m) for m in str(r["modes"]).replace(",", ";").split(";") if m.strip()]
                rows.append(normalize(r))
        return rows
    if suf in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("reading a YAML port file needs PyYAML (pip install pyyaml); JSON and CSV need nothing") from e
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if isinstance(data, dict):
        version = int(data.get("version", 1))
        if version != 1:
            raise ValueError(f"{path}: port file version {version}; this lightsim reads version 1")
        scale = UNITS[str(data.get("units", "um"))]
        items = data["ports"]
        if isinstance(items, dict):  # {"o1": {...}, ...}
            items = [{"name": k, **v} for k, v in items.items()]
    else:
        scale, items = 1e-6, data
    return [normalize(p, scale) for p in items]


def write_port_file(ports, path) -> Path:
    """Write ports (lightsim's dicts, metres) as a version-1 JSON port file (µm)."""
    out = []
    for p in ports:
        q = {"name": p["name"], "x": p["center"][0] / 1e-6, "y": p["center"][1] / 1e-6, "width": p["width"] / 1e-6, "orientation": p["orientation"]}
        if p.get("layer") is not None:
            q["layer"] = list(p["layer"])
        if p.get("modes"):
            q["modes"] = list(p["modes"])
        if p.get("z"):
            q["z"] = [p["z"][0] / 1e-6, p["z"][1] / 1e-6]
        out.append(q)
    path = Path(path)
    path.write_text(json.dumps({"version": 1, "units": "um", "ports": out}, indent=1))
    return path


def _label_near(layout: Layout, xy, layers, tol) -> str | None:
    best, bd = None, tol
    for text, l, tt, x, y in layout.labels:
        if layers is not None and l not in {a for a, _ in layers}:  # any texttype
            continue
        d = float(np.hypot(x - xy[0], y - xy[1]))
        if d <= bd:
            best, bd = text, d
    return best


def pins(layout: Layout, pin_layer, label_layer=None, port_layer=None, tol: float = 0.5e-6) -> list[dict]:
    """Ports from pins drawn on ``pin_layer`` and named by text labels.

    A pin is a two-point path (SiEPIC's convention: from inside the device
    to outside; the port is at its midpoint, as wide as the path) or a box
    (the port is at its centre, as wide as its longer side, facing away
    from the centre of the layout along its shorter side). The name is the
    text on ``label_layer`` (default the pin layer, any texttype) nearest
    the port, within ``tol`` metres (default 0.5 µm). ``port_layer`` is the
    waveguide layer recorded with the port.
    """
    pl = _layer(pin_layer)
    labels = [_layer(label_layer)] if label_layer is not None else [pl]
    found = []
    for l, dt, w, _pt, pts in layout.paths:
        if (l, dt) != pl or len(pts) < 2:
            continue
        a, b = np.asarray(pts[0]), np.asarray(pts[-1])
        d = b - a
        found.append(((a + b) / 2, float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0, w))
    if not found:  # boxes
        (x0, y0), (x1, y1) = layout.bbox()
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        for poly in layout.polygons.get(pl, []):
            q = np.asarray(poly)
            lo, hi = q.min(0), q.max(0)
            c = (lo + hi) / 2
            sx, sy = hi - lo
            if sx >= sy:  # wide in x: the light goes along y
                ang = 90.0 if c[1] >= centre[1] else 270.0
                width = sx
            else:
                ang = 0.0 if c[0] >= centre[0] else 180.0
                width = sy
            found.append((c, ang, float(width)))
    ports, unnamed = [], 0
    for c, ang, w in found:
        name = _label_near(layout, c, labels, tol)
        if name is None:
            unnamed += 1
            name = f"pin{unnamed}"
        ports.append({"name": name, "center": (float(c[0]), float(c[1])), "orientation": round(ang, 9) % 360.0, "width": float(w), "layer": _layer(port_layer) if port_layer is not None else None, "modes": None, "z": None})
    if not ports:
        raise ValueError(f"no pins on layer {pl[0]}/{pl[1]} in cell {layout.cell!r}")
    return ports


def siepic(layout: Layout, tol: float = 0.5e-6) -> list[dict]:
    """Ports from SiEPIC pins (PinRec 1/10 paths with labels; waveguide Si 1/0)."""
    return pins(layout, SIEPIC["pin_layer"], SIEPIC["label_layer"], SIEPIC["port_layer"], tol)


def from_gdsfactory(gds_path, cell=None) -> list[dict]:
    """Ports from the metadata gdsfactory writes into a GDS file.

    The metadata is kept in a form only KLayout reads; this route needs the
    gdsfactory package (which brings KLayout's Python module)."""
    try:
        import gdsfactory as gf
    except ImportError as e:
        raise ImportError("ports='gdsfactory' reads gdsfactory's port metadata, which needs gdsfactory (and KLayout); give a port file or a pin convention instead") from e
    from ..gds import port_info

    c = gf.import_gds(gds_path, cellname=cell)
    out = []
    for p in c.ports:
        if getattr(p, "port_type", "optical") != "optical":
            continue
        q = [x for x in port_info(c) if x["name"] == p.name][0]
        layer = gf.get_layer_tuple(p.layer) if hasattr(gf, "get_layer_tuple") else None
        out.append({**q, "layer": tuple(layer) if layer else None, "modes": None, "z": None})
    return out


def read_ports(ports, layout: Layout | None = None, gds_path=None, cell=None) -> list[dict]:
    """Ports from a port file, a list of dicts (port-file keys, µm), or a
    convention name (``"siepic"``, ``"gdsfactory"``, or a ``{"convention":
    "pins", ...}`` dict) applied to ``layout``."""
    if isinstance(ports, str | Path) and str(ports).lower() in ("siepic", "gdsfactory", "pins"):
        conv = str(ports).lower()
        if conv == "siepic":
            return siepic(layout)
        if conv == "gdsfactory":
            return from_gdsfactory(gds_path, cell)
        raise ValueError("ports='pins' needs the layers: pass {'convention': 'pins', 'pin_layer': [l, d], 'label_layer': [l, d], 'port_layer': [l, d]}")
    if isinstance(ports, dict):
        conv = ports.get("convention", "pins")
        if conv == "siepic":
            return siepic(layout, ports.get("tol", 0.5e-6))
        return pins(layout, ports["pin_layer"], ports.get("label_layer"), ports.get("port_layer"), ports.get("tol", 0.5e-6))
    if isinstance(ports, list | tuple):
        return [normalize(p) for p in ports]
    return read_port_file(ports)
