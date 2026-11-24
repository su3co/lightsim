"""README figure: throughput and speed-up vs problem size.

    python benchmarks/speed/plot_speed.py   # reads benchmarks/speed/results/sweep_*.json -> docs/img/speed.svg (+ .png), speed_throughput.svg, and a _dark.svg twin of each
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INK, GRID = "#16191f", "#dfe2e7"
# hardware is told apart by marker and ink weight, never by hue: on the site colour belongs to the example groups
STYLE = {"1x RTX PRO 6000": ("#16191f", "o"), "2x RTX PRO 6000": ("#16191f", "s"), "1x GB10": ("#5d6370", "^"), "2x GB10": ("#5d6370", "D")}
# published Lumerical FDTD GPU throughput (Ozen Engineering, metalens, A100 80 GB); not a measurement of ours
LUMERICAL_PUBLISHED = (0.85e9, 13.2)


LIGHT = dict(ink=INK, soft="#5d6370", paper="white", grid=GRID, minor="#eef0f3")
DARK = dict(ink="#eceef2", soft="#a9afbb", paper="#0c0e12", grid="#232831", minor="#171b22")


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 10.5,
                         "svg.fonttype": "path", "legend.frameon": False})
    sweeps = [json.loads(p.read_text()) for p in sorted((HERE / "results").glob("sweep_*.json"))]
    out = ROOT / "docs/img"
    out.mkdir(parents=True, exist_ok=True)
    # Every figure is written twice: for a light page and (<name>_dark.svg) for the dark documentation site.
    pair(sweeps, [out / "speed.svg", out / "speed.png"], **LIGHT)
    pair(sweeps, [out / "speed_dark.svg"], **DARK)
    # throughput panel alone, for narrow columns: larger text, so it still reads at phone width
    single(sweeps, out / "speed_throughput.svg", **LIGHT)
    single(sweeps, out / "speed_throughput_dark.svg", **DARK)


def pair(sweeps, paths, ink, soft, paper, grid, minor):
    """Throughput and speed-up side by side."""
    rc = {"axes.edgecolor": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink, "text.color": ink,
          "figure.facecolor": paper, "axes.facecolor": paper, "savefig.facecolor": paper}
    with plt.rc_context(rc):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
        for ax in (a1, a2):
            ax.set_xscale("log")
            ax.grid(True, which="major", color=grid, lw=0.8)
            ax.grid(True, which="minor", color=minor, lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel("cells")
        for sw in sweeps:
            mk = STYLE.get(sw["label"], (INK, "o"))[1]
            c = soft if "GB10" in sw["label"] else ink
            rows = sorted(sw["rows"], key=lambda r: r["cells"])
            x = [r["cells"] for r in rows if r.get("lightsim")]
            a1.plot(x, [r["lightsim"] for r in rows if r.get("lightsim")], "-", marker=mk, color=c, ms=4.5, lw=1.9, label=f"lightsim, {sw['label']}")
            fr = [r for r in rows if r.get("fdtdx") and r.get("lightsim")]
            if fr:
                a1.plot([r["cells"] for r in fr], [r["fdtdx"] for r in fr], "--", marker=mk, color=c, ms=4.5, mfc=paper, lw=1.1, label=f"FDTDX, {sw['label']}")
                a2.plot([r["cells"] for r in fr], [r["lightsim"] / r["fdtdx"] for r in fr], "-", marker=mk, color=c, ms=4.5, lw=1.9, label=sw["label"])
        a1.plot(*LUMERICAL_PUBLISHED, marker="*", color=ink, mfc=paper, ms=10, ls="none", label="Lumerical, published (A100)")
        a1.set_ylabel(r"throughput (Gcells/s)")
        a1.set_ylim(bottom=0)
        a2.axhline(1, color=ink, lw=0.8)
        a2.set_ylabel("speed-up over FDTDX")
        a2.set_ylim(bottom=0)
        # one legend under both panels, so it never sits on the curves
        handles, labels = a1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9, columnspacing=1.6, handlelength=2.4)
        a2.legend(fontsize=9, loc="upper right")
        for path in paths:
            fig.savefig(path, **({"dpi": 180} if path.suffix == ".png" else {}))
            print("wrote", path)
        plt.close(fig)


def single(sweeps, path, ink, soft, paper, grid, minor):
    rc = {"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.edgecolor": soft, "axes.labelcolor": ink,
          "xtick.color": ink, "ytick.color": ink, "text.color": ink, "figure.facecolor": paper, "axes.facecolor": paper, "savefig.facecolor": paper}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
        ax.set_xscale("log")
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.grid(True, which="minor", color=minor, lw=0.5)
        ax.set_axisbelow(True)
        for sw in sweeps:
            mk = STYLE.get(sw["label"], (INK, "o"))[1]
            c = soft if "GB10" in sw["label"] else ink
            rows = sorted(sw["rows"], key=lambda r: r["cells"])
            lr = [r for r in rows if r.get("lightsim")]
            ax.plot([r["cells"] for r in lr], [r["lightsim"] for r in lr], "-", marker=mk, color=c, ms=5, lw=2.0, label=f"lightsim, {sw['label']}")
            fr = [r for r in rows if r.get("fdtdx")]
            if fr:
                ax.plot([r["cells"] for r in fr], [r["fdtdx"] for r in fr], "--", marker=mk, color=c, ms=5, mfc=paper, lw=1.2, label=f"FDTDX, {sw['label']}")
        ax.plot(*LUMERICAL_PUBLISHED, marker="*", color=ink, mfc=paper, ms=11, ls="none", label="Lumerical, published (A100)")
        ax.set_xlabel("cells")
        ax.set_ylabel("throughput (Gcells/s)")
        ax.set_ylim(bottom=0)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=12.5, columnspacing=1.4, handlelength=2.2)
        fig.savefig(path)
        plt.close(fig)
    print("wrote", path)

if __name__ == "__main__":
    main()
