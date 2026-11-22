"""Tables of the inverse-design benchmark (2-D lightsim rows from lightsim
0.1.0, commit 54e0973; 3-D lightsim rows from the build before the 2-D mode,
whose 3-D kernels are the same) from benchmarks/results/invdes_*.json.

    python summarize.py            # prints markdown, writes results/invdes_summary.md

Also compares the gradient maps of the tools on each scene (cosine similarity
of the interior design samples against lightsim's native engine).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import problems

RES = Path(__file__).resolve().parent.parent / "results"
CPU = "64-core CPU (Threadripper 9980X)"

FAMILIES = [
    ("wg2d", "2-D, one input and one output, one wavelength", ["wg2d_45nm", "wg2d_15nm", "wg2d_5nm"]),
    ("wg3d", "3-D, one input and one output, one wavelength", ["wg3d_6M", "wg3d_32M", "wg3d_86M"]),
    ("mm2d", "2-D, 4 inputs x 4 outputs x 3 wavelengths", ["mm2d_45nm", "mm2d_15nm", "mm2d_5nm"]),
    ("mm3d", "3-D, 4 inputs x 4 outputs x 3 wavelengths", ["mm3d_44nm", "mm3d_22nm"]),
]


def label(r):
    """(sort key, row label, hardware, precision) of a result."""
    t, v = r["tool"], r["variant"]
    if t == "lightsim":
        n = len(r.get("devices", [0]))
        v, split = (v[:-6], True) if v.endswith("_split") else (v, False)
        name = {"python": "lightsim, Python engine", "native": "lightsim, native engine", "reverse": "lightsim, reverse mode"}[v]
        order = {"native": 0, "python": 1, "reverse": 2}[v] + 10 * (n - 1) + (5 if split else 0)
        return order, name + (", split kernels" if split else ""), f"{n}x RTX PRO 6000", "single"
    if t == "fdtdx":
        rec = v.split("_")[1]
        name = {"example": "8-bit recorder of its examples", "fp16": "16-bit recorder", "fp16k10": "16-bit recorder, every 10th step", "exact": "exact recorder"}[rec]
        return 40 + {"example": 0, "fp16k10": 1, "fp16": 2, "exact": 3}[rec], f"FDTDX, reversible autodiff ({name})", "1x RTX PRO 6000", "single"
    if t == "meep":
        return 50 + ("stock" in v), "Meep adjoint" + (" (stock adjoint pulse)" if "stock" in v else ""), CPU + ", MPI", "double"
    if t == "ceviche":
        return 60 + (v == "lu"), "ceviche" + (" operators, one LU per wavelength" if v == "lu" else " (autograd, as shipped)"), CPU + ", MKL", "double"
    return 99, f"{t} {v}", r.get("hardware", ""), r.get("precision", "")


def load():
    out = []
    for p in sorted(RES.glob("invdes_*.json")):
        if "convergence" in p.name or "summary" in p.name:
            continue
        r = json.loads(p.read_text())
        if "seconds_per_gradient" not in r or r.get("timing_only"):
            continue
        # 2-D lightsim rows: only the current build (with the 2-D mode); the
        # earlier build's 2-D results stay in the result files
        if r["tool"] == "lightsim" and r["ndim"] == 2 and r.get("build", "").find("54e0973") < 0:
            continue
        r["_file"] = p.name
        out.append(r)
    return out


def note(r):
    """Process or thread count of a CPU result (the best one is shown)."""
    if r["tool"] == "meep" and r.get("extrapolated"):
        return f", {r['mpi_processes']} proc., extrapolated from 100- and 200-step runs ({r['cells_per_second'] / 1e6:.0f} M cells/s) and the full {r['calibration_scene']} gradient"
    if r["tool"] == "meep":
        return f", {r['mpi_processes']} proc."
    if r["tool"] == "ceviche":
        return f", {r['threads']} thr."
    return ""


def best(rows):
    """Fastest result per (scene, label, hardware)."""
    d = {}
    for r in rows:
        k = (r["scene"], label(r)[1], label(r)[2])
        if k in d and d[k].get("extrapolated") != r.get("extrapolated"):
            if r.get("extrapolated"):  # a measurement beats an extrapolation
                continue
            d[k] = r
        elif k not in d or r["seconds_per_gradient"] < d[k]["seconds_per_gradient"]:
            d[k] = r
    return d


def fmt_t(s):
    if s is None:
        return ""
    return f"{s:.3f} s" if s < 1 else f"{s:.2f} s" if s < 10 else f"{s:.1f} s" if s < 100 else f"{s:.0f} s"


def fmt_e(x):
    return "" if x is None else f"{100 * x:.2g} %" if x >= 1e-3 else f"{x:.0e}"


def fd_cell(r):
    fd = r.get("fd")
    if not fd:
        return ""
    a = fmt_e(fd["max_rel_to_largest"]) if "max_rel_to_largest" in fd else "-"
    b = fmt_e(fd["direction"]["rel"]) if "direction" in fd else "-"
    return f"{a} / {b}"


def cosine(a, b):
    a, b = a[1:-1, 1:-1].ravel(), b[1:-1, 1:-1].ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    rows = load()
    d = best(rows)
    lines = []
    for fam, title, scenes in FAMILIES:
        scs = [problems.make(s) for s in scenes]
        keys = sorted({(label(r)[0], label(r)[1], label(r)[2], label(r)[3]) for r in rows if r["scene"] in scenes})
        if not keys:
            continue
        lines += [f"### {title}", ""]
        head = [f"{s.n_cells / 1e6:.3g} M cells x {s.steps()} steps" if s.n_cells >= 1e6 else f"{s.n_cells / 1e3:.3g} k cells x {s.steps()} steps" for s in scs]
        lines.append("| tool | hardware | precision | " + " | ".join(head) + " |")
        lines.append("|---|---|---|" + "---|" * len(scenes))
        for _, name, hw, prec in keys:
            cells = []
            for s in scenes:
                r = d.get((s, name, hw))
                if r is None:
                    cells.append("")
                else:
                    n = len(r["seconds_all"])
                    few = "" if n >= 3 or r.get("extrapolated") else f" ({n} run{'s' if n > 1 else ''})"
                    cells.append(("~" if r.get("extrapolated") else "") + fmt_t(r["seconds_per_gradient"]) + note(r) + few)
            lines.append(f"| {name} | {hw} | {prec} | " + " | ".join(cells) + " |")
        lines += ["", "Gradient against central finite differences (worst patch, relative to the largest / derivative along a smooth direction), and against lightsim's gradient map (cosine similarity of the interior samples):", ""]
        lines.append("| tool | " + " | ".join(scenes) + " |")
        lines.append("|---|" + "---|" * len(scenes))
        ref = {}
        for s in scenes:
            r = d.get((s, "lightsim, native engine", "1x RTX PRO 6000"))
            if r is not None and (RES / r["_file"].replace(".json", ".npz")).exists():
                z = np.load(RES / r["_file"].replace(".json", ".npz"))
                if "grad" in z.files:
                    ref[s] = z["grad"]
        seen = set()
        for _, name, hw, _ in keys:
            if name in seen:
                continue
            seen.add(name)
            cells = []
            for s in scenes:
                # any run of this tool on this scene that carries the check
                cand = [r for r in rows if r["scene"] == s and label(r)[1] == name]
                cand = sorted(cand, key=lambda r: ("fd" not in r, "max_rel_to_largest" not in r.get("fd", {})))
                if not cand:
                    cells.append("")
                    continue
                r = cand[0]
                c = fd_cell(r)
                npz = RES / r["_file"].replace(".json", ".npz")
                if s in ref and npz.exists() and "reverse" not in name and name != "lightsim, native engine":
                    z = np.load(npz)
                    if "grad" in z.files and z["grad"].shape == ref[s].shape:
                        c += f"; cos {cosine(z['grad'], ref[s]):.4f}"
                cells.append(c)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "Objective value (same design in every tool); for the matrix scenes also the power reaching the four outputs per unit input power, averaged over inputs and wavelengths:", ""]
        lines.append("| tool | " + " | ".join(scenes) + " |")
        lines.append("|---|" + "---|" * len(scenes))
        seen = set()
        for _, name, hw, _ in keys:
            if name in seen or "reverse" in name:
                continue
            seen.add(name)
            cells = []
            for s in scenes:
                cand = [r for (sc_, nm, _), r in d.items() if sc_ == s and nm == name and "objective" in r]
                if not cand:
                    cells.append("")
                    continue
                c = f"{cand[0]['objective']:.4g}"
                p2 = np.asarray(cand[0].get("S_abs2", []), dtype=float)
                if p2.ndim == 3 and p2.shape[0] > 1:
                    c += f" (throughput {p2.sum(axis=0).mean():.3f})"
                cells.append(c)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    # cost in simulations, set-up, memory
    lines += ["### Cost per gradient in forward runs, one-time costs, memory", ""]
    lines.append("| tool | scene | hardware | runs per gradient | gradient / one forward run | first call | one-time set-up | peak memory |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (s, name, hw), r in sorted(d.items(), key=lambda kv: (kv[0][0], label(kv[1])[0])):
        sims = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in r.get("simulations", {}).items())
        fe = r.get("forward_equivalents")
        setup = [r.get(k) for k in ("session_setup_seconds", "compile_seconds", "setup_seconds", "scene_build_seconds")]
        setup = next((v for v in setup if v), None)
        extra = ""
        if r["tool"] == "fdtdx":
            setup = r["compile_seconds"] + r["place_seconds"] + r["apply_seconds"]
            extra = f" (compile {r['compile_seconds']:.1f} s)"
        mem = r.get("peak_device_bytes") or r.get("peak_host_bytes")
        where = "GPU" if r.get("peak_device_bytes") else "host"
        first = r.get("first_call_seconds")
        lines.append(f"| {name} | {s} | {hw} | {sims} | {'' if fe is None else f'{fe:.2f}'} | {fmt_t(first)} | {fmt_t(setup)}{extra} | {'' if not mem else f'{mem / 2**30:.1f} GB ({where})'} |")
    txt = "\n".join(lines) + "\n"
    (RES / "invdes_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
