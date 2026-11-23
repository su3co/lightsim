"""``python -m lightsim``: the layout commands of lightsim.

    python -m lightsim sparams LAYOUT.gds --technology TECH --ports PORTS [--cell NAME] --out S.s3p
    python -m lightsim layout-spec LAYOUT.gds --technology TECH --ports PORTS [--cell NAME] --out spec.json

``lightsim sparams`` and ``lightsim layout-spec`` (the command-line program)
run these. ``--ports`` is a port file (.json, .yaml, .csv), ``siepic`` or
``gdsfactory``; ``--pins PIN_LAYER,LABEL_LAYER,PORT_LAYER`` (``1/10,10/0,1/0``)
uses the pin convention on those layers. ``--out``: ``.sNp`` (Touchstone),
``.dat`` (INTERCONNECT text) or ``.npz`` (gplugins layout).
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np


def _wavelengths(spec: str) -> np.ndarray:
    a, b, n = spec.split(":")
    return np.linspace(float(a), float(b), int(n)) * 1e-6


def _common(p):
    p.add_argument("gds", help="GDSII file")
    p.add_argument("--cell", help="cell to simulate (default: the only top cell)")
    p.add_argument("--technology", required=True, help="technology file: .toml (gamma), .yaml or .json")
    p.add_argument("--ports", help="port file (.json, .yaml, .csv), 'siepic' or 'gdsfactory'")
    p.add_argument("--pins", help="pin convention layers PIN,LABEL,PORT, e.g. 1/10,1/10,1/0")
    p.add_argument("--wavelengths", default="1.5:1.6:21", help="START:STOP:POINTS in um (default 1.5:1.6:21)")
    p.add_argument("--2d", dest="two_d", action="store_true", help="2-D model instead of 3-D")
    p.add_argument("--bulk", action="store_true", help="2-D: bulk indices instead of effective indices")
    p.add_argument("--polarization", choices=("te", "tm"))
    p.add_argument("--resolution", type=float, help="uniform mesh, cells per um (default: graded, 20 steps per wavelength)")
    p.add_argument("--steps-per-wavelength", type=float, default=20.0)
    p.add_argument("--margin", type=float, default=1.0, help="x/y distance to the absorber, um")
    p.add_argument("--z-margin", type=float, default=0.8, help="z distance to the absorber, um")
    p.add_argument("--out", required=True)


def _kwargs(a) -> dict:
    if a.pins:
        pl, ll, po = (s.strip() for s in a.pins.split(","))
        ports = {"convention": "pins", "pin_layer": pl, "label_layer": ll, "port_layer": po}
    elif a.ports:
        ports = a.ports
    else:
        raise SystemExit("give --ports or --pins")
    return dict(
        gds=a.gds,
        technology=a.technology,
        ports=ports,
        cell=a.cell,
        wavelengths=_wavelengths(a.wavelengths),
        is_3d=not a.two_d,
        effective_index=not a.bulk,
        polarization=a.polarization,
        resolution=a.resolution,
        steps_per_wavelength=a.steps_per_wavelength,
        margin=a.margin,
        z_margin=a.z_margin,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lightsim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sparams", help="simulate a cell and write its S-parameters")
    _common(s)
    s.add_argument("--device", default="auto")
    s.add_argument("--touchstone-version", default="1.1")
    s.add_argument("--freq-unit", default="GHz")
    s.add_argument("--progress", type=int, default=0)
    g = sub.add_parser("layout-spec", help="write the lightsim JSON spec of each source, without running")
    _common(g)
    g.add_argument("--sources", help="comma-separated ports or port@mode to build (default: all)")
    a = ap.parse_args(argv)
    from . import layout

    kw = _kwargs(a)
    if a.cmd == "layout-spec":
        sims = layout.simulation(sources=a.sources.split(",") if a.sources else None, **kw)
        for p in layout.write_specs(sims, a.out):
            print(p)
        return 0
    sp = layout.sparameters(device=a.device, progress=a.progress, **kw)
    out = a.out
    if re.search(r"\.s\d+p$", out, re.I):
        print(sp.to_touchstone(out, version=a.touchstone_version, freq_unit=a.freq_unit))
    elif out.lower().endswith((".dat", ".sparam", ".txt")):
        print(sp.to_lumerical_dat(out))
    elif out.lower().endswith(".npz"):
        print(sp.to_npz(out))
    else:
        raise SystemExit(f"--out {out}: use .sNp (Touchstone), .dat (INTERCONNECT text) or .npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
