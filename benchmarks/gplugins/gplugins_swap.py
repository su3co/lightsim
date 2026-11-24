#!/usr/bin/env python3
"""Swap test: the same gplugins call through Meep and through lightsim.

Runs ``gplugins.gmeep.write_sparameters_meep`` and
``lightsim.gplugins.write_sparameters_lightsim`` with the same arguments on
five standard gdsfactory components (straight, Euler bend, mmi1x2,
directional coupler, crossing), saves the S-parameters and the run times,
and compares them: magnitudes, phases, reciprocity, port ordering, units.

    python gplugins_swap.py run --solver meep                     # Meep, CPU, 2-D
    python gplugins_swap.py run --solver lightsim --device cpu    # lightsim, CPU, 2-D
    python gplugins_swap.py run --solver lightsim --device cuda:0 # lightsim, GPU, 2-D
    python gplugins_swap.py run --solver lightsim --device cuda:0 --3d
    python gplugins_swap.py run --solver meep --3d --only straight
    python gplugins_swap.py batch --devices cuda:0,cuda:1 --3d    # write_sparameters_lightsim_batch
    python gplugins_swap.py checks                                # gplugins.sax, cache paths
    python gplugins_swap.py compare                               # table from the saved results

Results go to ``results/`` beside this script: one ``.npz`` of S-parameters per
solver, device, dimension and component, and a ``.json`` of run times.

Settings: resolution 30 cells/um (Meep's default), 1.50-1.60 um in 21
points, ``decay_by=1e-4``, the generic PDK, constant indices Si 3.476 and
SiO2 1.444 for both solvers (``material_name_to_meep``), margins as in the
gplugins notebooks. lightsim's 2-D runs use ``effective_index=False``: the
plain 2-D cut with the bulk indices, which is what Meep's ``is_3d=False``
simulates (Meep's default 2-D run is not an effective-index model).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

COMMON = dict(
    resolution=30,
    wavelength_start=1.5,
    wavelength_stop=1.6,
    wavelength_points=21,
    decay_by=1e-4,
    material_name_to_meep={"si": 3.476, "sio2": 1.444},
)
# 3-D: Meep's default z margin is 0 (the absorber then touches the core); both get 1 um
COMMON_3D = dict(resolution=20, zmargin=1.0)


def components():
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    return {
        "straight": (gf.components.straight(length=2), dict(ymargin=3)),
        "bend": (gf.components.bend_euler(radius=5), dict(ymargin=0, ymargin_bot=3, xmargin_right=3)),
        # the outputs are 1.25 um apart: gplugins' default port_margin=2 gives Meep mode windows that hold both
        "mmi1x2": (gf.components.mmi1x2(), dict(ymargin=2, port_margin=0.5)),
        "mmi1x2_pm2": (gf.components.mmi1x2(), dict(ymargin=2)),
        "coupler": (gf.components.coupler(), dict(ymargin=2)),
        "crossing": (gf.components.crossing(), dict(ymargin=0)),
    }


def tag(solver: str, device: str, is_3d: bool) -> str:
    return f"{solver}_{device.replace(':', '').replace(',', '-')}_{'3d' if is_3d else '2d'}"


def cmd_run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if args.solver == "meep":
        import gplugins.gmeep as gm

        write = gm.write_sparameters_meep
        extra = {}
    else:
        import lightsim.gplugins as lg

        write = lg.write_sparameters_lightsim
        extra = dict(device=args.device, effective_index=False)
    t = tag(args.solver, args.device if args.solver == "lightsim" else "cpu", args.three_d)
    if args.resolution:
        t += f"_res{args.resolution}"
    timings_path = OUT / f"{t}.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    for name, (c, margins) in components().items():
        if args.only and name not in args.only:
            continue
        kw = dict(COMMON, **margins, **extra)
        if args.three_d:
            kw.update(COMMON_3D, is_3d=True)
        if args.resolution:
            kw["resolution"] = args.resolution
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            sp = write(c, dirpath=tmp, overwrite=True, **kw)
            dt = time.perf_counter() - t0
        np.savez_compressed(OUT / f"{t}_{name}.npz", **sp)
        timings[name] = dt
        print(f"{t} {name}: {dt:.1f} s", flush=True)
        timings_path.write_text(json.dumps(timings, indent=1))


def cmd_batch(args) -> None:
    """write_sparameters_lightsim_batch over several GPUs: wall time of the batch."""
    import lightsim.gplugins as lg

    OUT.mkdir(parents=True, exist_ok=True)
    devs = args.devices.split(",")
    jobs = []
    for _, (c, margins) in components().items():
        kw = dict(COMMON, **margins, effective_index=False, component=c, overwrite=True)
        if args.three_d:
            kw.update(COMMON_3D, is_3d=True)
        jobs.append(kw)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        paths = lg.write_sparameters_lightsim_batch(jobs, dirpath=tmp, devices=devs)
        dt = time.perf_counter() - t0
        assert all(Path(p).exists() for p in paths)
    key = f"batch_{'-'.join(d.replace(':', '') for d in devs)}_{'3d' if args.three_d else '2d'}"
    path = OUT / "batch.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = dt
    path.write_text(json.dumps(data, indent=1))
    print(f"{key}: {dt:.1f} s for {len(jobs)} components")


def cmd_grating(args) -> None:
    """write_sparameters_grating through Meep and through lightsim (2-D, same geometry)."""
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(resolution=30, wavelength_points=31, overwrite=True)
    if args.solver == "meep":
        import gplugins.gmeep as gm

        write, extra, t = gm.write_sparameters_grating, {}, "meep_cpu"
    else:
        import lightsim.gplugins as lg

        write, extra, t = lg.write_sparameters_grating_lightsim, dict(device=args.device), tag("lightsim", args.device, False)[:-3]
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        sp = dict(write(dirpath=tmp, **kw, **extra))
        dt = time.perf_counter() - t0
    np.savez_compressed(OUT / f"{t}_grating.npz", **sp)
    path = OUT / "grating.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    k = np.argmax(np.abs(sp["o2@0,o1@0"]))
    data[t] = {"seconds": dt, "peak_wavelength": float(sp["wavelengths"][k]), "peak_coupling": float(np.abs(sp["o2@0,o1@0"][k]) ** 2)}
    path.write_text(json.dumps(data, indent=1))
    print(t, data[t])


def cmd_checks(args) -> None:
    """gplugins.sax reads lightsim's .npz; cache paths with and without gplugins agree."""
    import gdsfactory as gf
    import lightsim.gplugins as lg
    from gplugins.common.utils.get_sparameters_path import _get_sparameters_path
    from gplugins.sax.read import model_from_npz

    out = {}
    c, margins = components()["mmi1x2"]
    with tempfile.TemporaryDirectory() as tmp:
        sp = lg.write_sparameters_lightsim(c, dirpath=tmp, device=args.device, effective_index=False, **COMMON, **margins)
        npz = lg.write_sparameters_lightsim(c, dirpath=tmp, effective_index=False, only_return_filepath_sim_settings=True, **COMMON, **margins).with_suffix(".npz")
        model = model_from_npz(npz)
        s = model(wl=np.array([1.55]))
        out["sax_keys"] = sorted(f"{a},{b}" for a, b in s)
        out["sax_o2_o1_at_1550"] = float(np.abs(np.asarray(s[("o2", "o1")])[0]) ** 2)
        out["npz_o2_o1_at_1550"] = float(np.abs(np.interp(1.55, sp["wavelengths"], sp["o2@0,o1@0"])) ** 2)
        settings = dict(a=1, b=[1, 2], layer_stack=gf.get_active_pdk().layer_stack.to_dict())
        p1 = _get_sparameters_path(component=c, dirpath=tmp, tool="lightsim", **settings)
        p2 = lg._builtin_sparameters_path(c, tmp, tool="lightsim", **settings)
        out["paths_equal"] = p1 == p2
        out["gplugins_path_used"] = lg.get_sparameters_path(c, tmp, **settings) == p1
    (OUT / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------


def load(t: str, name: str):
    p = OUT / f"{t}_{name}.npz"
    return dict(np.load(p)) if p.exists() else None


def reciprocity(sp) -> float:
    worst = 0.0
    for k in sp:
        if k == "wavelengths":
            continue
        a, b = k.split(",")
        r = f"{b},{a}"
        if r in sp:
            worst = max(worst, float(np.max(np.abs(sp[k] - sp[r]))))
    return worst


def compare(ref, sp):
    """Per component: worst |S| difference over the main entries, phase agreement."""
    keys = [k for k in ref if k != "wavelengths"]
    assert sorted(keys) == sorted(k for k in sp if k != "wavelengths"), "different port keys"
    assert np.allclose(ref["wavelengths"], sp["wavelengths"]), "different wavelengths"
    main = [k for k in keys if np.mean(np.abs(ref[k])) > 0.1]  # transmissions, not the -30 dB entries
    dmag = max(float(np.max(np.abs(np.abs(sp[k]) - np.abs(ref[k])))) for k in main)
    small = [k for k in keys if k not in main]
    dmag_small = max([float(np.max(np.abs(np.abs(sp[k]) - np.abs(ref[k])))) for k in small] or [0.0])
    # phase: lightsim - Meep, unwrapped over the band; its mean is a convention offset, its spread is real disagreement
    offs, spread = {}, 0.0
    for k in main:
        d = np.unwrap(np.angle(sp[k] * np.conj(ref[k])))
        offs[k] = float(np.degrees(np.mean(d)))
        spread = max(spread, float(np.degrees(np.ptp(d))))
    return dict(dmag_main=dmag, dmag_small=dmag_small, phase_offset_deg=offs, phase_spread_deg=spread, main=main)


def cmd_compare(args) -> None:
    names = list(components())
    rows, rep = [], {}
    mid = 10  # 1.55 um
    entry = {"coupler": "o3@0,o1@0", "crossing": "o3@0,o1@0"}
    for dim in ("2d", "3d"):
        for name in names:
            ref = load(f"meep_cpu_{dim}", name)
            if ref is None:
                continue
            for dev in ("cuda0", "cpu"):
                sp = load(f"lightsim_{dev}_{dim}", name)
                if sp is None:
                    continue
                c = compare(ref, sp)
                rep[f"{dim}/{name}/{dev}"] = {**c, "reciprocity_lightsim": reciprocity(sp), "reciprocity_meep": reciprocity(ref)}
                k = entry.get(name, "o2@0,o1@0")
                rows.append((dim, name, dev, k, 10 * np.log10(abs(ref[k][mid]) ** 2), 10 * np.log10(abs(sp[k][mid]) ** 2), c["dmag_main"], c["dmag_small"], c["phase_spread_deg"], reciprocity(ref), reciprocity(sp)))
    tags = [t for t in ("meep_cpu_2d", "lightsim_cpu_2d", "lightsim_cuda0_2d", "meep_cpu_3d", "lightsim_cpu_3d", "lightsim_cuda0_3d") if (OUT / f"{t}.json").exists()]
    times = {t: json.loads((OUT / f"{t}.json").read_text()) for t in tags}
    lines = [
        "| Model | Component | lightsim on | Entry | Meep (dB) | lightsim (dB) | max Δ\\|S\\| main | max Δ\\|S\\| small | phase spread (deg) | reciprocity Meep | reciprocity lightsim |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | `{r[3]}` | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.1f} | {r[9]:.1e} | {r[10]:.1e} |")
    lines += ["", "| Component | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{times[t][name]:.1f}" if name in times[t] else "" for t in tags) + " |")
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text + "\n")
    (OUT / "comparison.json").write_text(json.dumps(rep, indent=1))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--solver", choices=("meep", "lightsim"), required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--3d", dest="three_d", action="store_true")
    r.add_argument("--only", nargs="*")
    r.add_argument("--resolution", type=int, help="override the resolution (convergence runs; tagged _res<N>)")
    b = sub.add_parser("batch")
    b.add_argument("--devices", default="cuda:0,cuda:1")
    b.add_argument("--3d", dest="three_d", action="store_true")
    g = sub.add_parser("grating")
    g.add_argument("--solver", choices=("meep", "lightsim"), required=True)
    g.add_argument("--device", default="cpu")
    k = sub.add_parser("checks")
    k.add_argument("--device", default="cpu")
    sub.add_parser("compare")
    args = ap.parse_args()
    {"run": cmd_run, "batch": cmd_batch, "grating": cmd_grating, "checks": cmd_checks, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
