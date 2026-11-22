"""Docs figure: time per gradient against problem size, lightsim and other tools.

    python benchmarks/plot_invdes.py   # reads benchmarks/results/invdes_*.json -> docs/img/invdes_speed.svg, .png and invdes_speed_dark.svg

One panel per problem family, one line per tool; lightsim solid with filled
markers, the other tools dashed with open markers; log axes; colours of
benchmarks/plot_speed.py. Each point is the fastest valid configuration of
that tool on that scene (for lightsim the faster of its native and Python
engines, which compute the same gradient; for FDTDX the most accurate recorder
that ran; FDTDX's unvalidated multi-port 3-D point is left out). The 2-D
lightsim points are from lightsim 0.1.0, the 3-D ones from the build before
its 2-D mode. Meep points that
are extrapolated from fixed-step runs are drawn dotted with a different
marker; FDTDX points whose gradient misses the finite-difference check by
more than 5 % are crossed. See benchmarks/invdes/README.md for how each
number is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# one palette per page theme: ink, grid, minor grid, paper, then lightsim on 1 GPU, on 2 GPUs, FDTDX, Meep
THEMES = {
    "light": dict(ink="#16191f", grid="#dfe2e7", minor="#eef0f3", paper="#ffffff", c1="#3b2a8c", c2="#c0461b", ca="#0f7b6c", cb="#6f8f2f"),
    "dark": dict(ink="#eceef2", grid="#232831", minor="#171b22", paper="#0c0e12", c1="#a99bff", c2="#ff9166", ca="#4fd1b5", cb="#b7d36b"),
}
PANELS = [
    ("wg2d", "2-D, one input, one output, one wavelength"),
    ("wg3d", "3-D, one input, one output, one wavelength"),
    ("mm2d", r"2-D, $4\times4$ ports, 3 wavelengths"),
    ("mm3d", r"3-D, $4\times4$ ports, 3 wavelengths"),
]
FDTDX_PREF = {"reversible_exact": 0, "reversible_fp16": 1, "reversible_fp16k10": 2, "reversible_example": 3}


def load():
    rows = []
    for p in sorted((ROOT / "benchmarks/results").glob("invdes_*.json")):
        r = json.loads(p.read_text())
        if "seconds_per_gradient" not in r or r.get("timing_only"):
            continue
        if r["tool"] == "lightsim" and r["ndim"] == 2 and "54e0973" not in r.get("build", ""):
            continue  # 2-D: only the build with the 2-D mode
        rows.append(r)
    return rows


def fd_error(r):
    """Worst relative error of the finite-difference checks of a result (None: no check)."""
    fd = r.get("fd") or {}
    errs = [fd["max_rel_to_largest"]] if "max_rel_to_largest" in fd else []
    if "direction" in fd:
        errs.append(fd["direction"]["rel"])
    return max(errs) if errs else None


def series(rows, family):
    """{label: [(cells, seconds, flag)]}; flag 'x' = fails the FD check, 'e' = extrapolated."""
    out = {}

    def put(lab, r, flag=""):
        pts = out.setdefault(lab, {})
        c = r["cells"]
        if c not in pts or r["seconds_per_gradient"] < pts[c][0]:
            pts[c] = (r["seconds_per_gradient"], flag)

    fam = [r for r in rows if r["scene"].startswith(family)]
    for r in fam:
        t, v = r["tool"], r["variant"]
        if t == "lightsim" and v in ("native", "python"):
            put("lightsim, 1 GPU" if len(r.get("devices", [0])) == 1 else "lightsim, 2 GPUs", r)
        elif t == "meep" and v == "adjoint":
            put("Meep adjoint, extrapolated" if r.get("extrapolated") else "Meep adjoint", r, "e" if r.get("extrapolated") else "")
        elif t == "ceviche":
            put("ceviche (as shipped)" if v == "stock" else "FDFD, one LU per wavelength", r)
    # FDTDX: per scene, the most accurate recorder that ran
    by_scene = {}
    for r in fam:
        # FDTDX's multi-port 3-D set-up is not validated (its S-matrix is not
        # physical, see docs/benchmarks.md): left out of the figure
        if r["tool"] == "fdtdx" and r["variant"] in FDTDX_PREF and family != "mm3d":
            s = r["scene"]
            if s not in by_scene or FDTDX_PREF[r["variant"]] < FDTDX_PREF[by_scene[s]["variant"]]:
                by_scene[s] = r
    for r in by_scene.values():
        e = fd_error(r)
        put("FDTDX, 1 GPU", r, "x" if e is not None and e > 0.05 else "")
    # an extrapolated Meep point where a measured one exists is dropped
    meas = out.get("Meep adjoint", {})
    if "Meep adjoint, extrapolated" in out:
        ex = {c: v for c, v in out["Meep adjoint, extrapolated"].items() if c not in meas}
        if ex and meas:  # join the dotted line to the last measured point
            c0 = max(meas)
            ex[c0] = meas[c0][0], ""
        out["Meep adjoint, extrapolated"] = ex
    return {k: sorted((c, s, f) for c, (s, f) in v.items()) for k, v in out.items() if v}


def styles(t):
    return {
        "lightsim, 1 GPU": dict(color=t["c1"], marker="o", ls="-", solid=True),
        "lightsim, 2 GPUs": dict(color=t["c2"], marker="s", ls="-", solid=True),
        "FDTDX, 1 GPU": dict(color=t["ca"], marker="o", ls="--", solid=False),
        "Meep adjoint": dict(color=t["cb"], marker="s", ls="--", solid=False),
        "Meep adjoint, extrapolated": dict(color=t["cb"], marker="^", ls=":", solid=False),
        "ceviche (as shipped)": dict(color=t["ink"], marker="D", ls="--", solid=False),
        "FDFD, one LU per wavelength": dict(color=t["ink"], marker="v", ls=":", solid=False),
    }


def render(theme, paths):
    t = THEMES[theme]
    INK, GRID, PAPER, STYLE = t["ink"], t["grid"], t["paper"], styles(t)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 10.5,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
                         "svg.fonttype": "path", "legend.frameon": False})
    rows = load()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    used = []
    crossed = False
    for ax, (family, title) in zip(axes.ravel(), PANELS):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.grid(True, which="minor", color=t["minor"], lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("cells")
        ax.set_ylabel("time per gradient (s)")
        for lab, pts in series(rows, family).items():
            st = STYLE[lab]
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            ax.plot(x, y, ls=st["ls"], color=st["color"], lw=1.9 if st["solid"] else 1.2, zorder=3 if st["solid"] else 2)
            for c, s, f in pts:
                if lab == "Meep adjoint, extrapolated" and f != "e":
                    continue  # the joining point is drawn by the measured series
                ax.plot([c], [s], ls="none", marker=st["marker"], color=st["color"], ms=5.5, mew=1.1,
                        mfc=st["color"] if st["solid"] else PAPER, zorder=4)
                if f == "x":
                    ax.plot([c], [s], ls="none", marker="x", color=st["color"], ms=9, mew=1.3, zorder=5)
                    crossed = True
            if lab not in used:
                used.append(lab)
        ax.margins(x=0.12, y=0.15)
    used = [l for l in STYLE if l in used]
    handles = [Line2D([], [], ls=STYLE[l]["ls"], color=STYLE[l]["color"], marker=STYLE[l]["marker"], ms=5.5, mew=1.1,
                      lw=1.9 if STYLE[l]["solid"] else 1.2, mfc=STYLE[l]["color"] if STYLE[l]["solid"] else PAPER) for l in used]
    labels = list(used)
    if crossed:
        handles.append(Line2D([], [], ls="none", marker="x", color=t["ca"], ms=9, mew=1.3))
        labels.append("FDTDX gradient off by $>5$ % (finite differences)")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    for path, kw in paths:
        fig.savefig(path, **kw)
        print("wrote", path)
    plt.close(fig)


def main():
    out = ROOT / "docs/img"
    out.mkdir(parents=True, exist_ok=True)
    # one rendering for a light page and one (invdes_speed_dark.svg) for a dark page
    render("light", [(out / "invdes_speed.svg", {}), (out / "invdes_speed.png", {"dpi": 180})])
    render("dark", [(out / "invdes_speed_dark.svg", {})])


if __name__ == "__main__":
    main()
