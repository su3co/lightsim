#!/usr/bin/env python3
"""Reflection-phase convention of gplugins.gmeep and lightsim.gplugins.

A straight waveguide ends on a perfect electric conductor 1.7 um behind the
monitor of the source port. With the usual convention (the backward mode has
the same transverse E as the forward mode and the opposite H), the field
reflection at the wall is -1 and the monitor sees S11 = -exp(2 i beta d).
Both solvers are run through their gplugins front with the same settings,
and S11 is compared with that value at 1.55 um.

    python gplugins_reflection.py            # needs Meep, gplugins and lightsim
"""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "gplugins_swap"
gf.gpdk.PDK.activate()
X_WALL = 2.0  # um; the source-port monitor sits at x = 0.3 um
COMMON = dict(resolution=30, wavelength_start=1.5, wavelength_stop=1.6, wavelength_points=3, material_name_to_meep={"si": 3.476, "sio2": 1.444})


def component():
    return gf.add_padding_container(gf.components.straight(length=4), default=0, top=3, bottom=3)


def meep_s11():
    import meep as mp
    from gplugins.gmeep.get_simulation import get_simulation
    from gplugins.gmeep.write_sparameters_meep import parse_port_eigenmode_coeff

    c = component()
    d = get_simulation(c, port_source_name="o1", port_margin=2, port_source_offset=-0.1, port_monitor_offset=-0.1, is_3d=False, **COMMON)
    sim = d["sim"]
    sim.geometry.append(mp.Block(center=mp.Vector3(X_WALL + 10, 0), size=mp.Vector3(20, mp.inf, mp.inf), material=mp.metal))
    sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-6))
    cin, cout = parse_port_eigenmode_coeff("o1", c.ports, d)
    return cout / cin


def lightsim_s11():
    import lightsim as ls
    import lightsim.gplugins as lg

    c = component()
    sim = lg.write_sparameters_lightsim(c, run=False, effective_index=False, decay_by=1e-6, **COMMON)
    sim.add_medium(ls.Medium("wall", pec=True))
    sim.add_structure(ls.Box([X_WALL * 1e-6, -1, -1], [1, 1, 1]), "wall")
    r = sim.run(device="cpu")
    a, b = r.mode_amplitudes("m_o1")
    # the effective index of the mode the front injects
    p = sim.sources[0]
    m = ls.solve_modes(sim, 0, p["position"], p["min"], p["max"], ls.C0 / 1.55e-6, num_modes=1)[0]
    return b / a, m.neff.real


def main():
    s_ls, neff = lightsim_s11()
    d = X_WALL - 0.3
    theory = -np.exp(2j * 2 * np.pi * neff * d / 1.55)
    out = {"neff_lightsim": neff, "theory_phase_deg": float(np.degrees(np.angle(theory))), "lightsim_phase_deg": float(np.degrees(np.angle(s_ls[1]))), "lightsim_abs": float(abs(s_ls[1]))}
    try:
        s_mp = meep_s11()
        out.update(meep_phase_deg=float(np.degrees(np.angle(s_mp[1]))), meep_abs=float(abs(s_mp[1])))
    except ImportError as e:
        out["meep"] = f"not run: {e}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reflection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
