"""Units, the speed of light and the coordinate helpers shared by the builders."""

from __future__ import annotations

from typing import Sequence


C0 = 299_792_458.0
um = 1e-6
nm = 1e-9


Vec3 = Sequence[float]


def _v3(v: Vec3) -> list[float]:
    """A point or box corner: 3 components, or the 2 in-plane components of a
    2-D simulation (completed by :class:`Simulation` along its invariant
    axis)."""
    v = [float(x) for x in v]
    if len(v) not in (2, 3):
        raise ValueError(f"expected 3 components (or 2 in a 2-D simulation), got {v}")
    return v


_AXES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def _axis(a) -> int:
    try:
        return _AXES[a.lower() if isinstance(a, str) else int(a)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"unknown axis {a!r} (use 'x', 'y', 'z' or 0, 1, 2)") from None


def _lift(v, w: int, c: float) -> list[float]:
    """In-plane components (a, b) of a 2-D simulation invariant along axis
    ``w`` -> 3 components, with ``c`` along ``w``. 3-vectors pass through."""
    v = list(v)
    if len(v) == 3:
        return v
    return v[:w] + [float(c)] + v[w:]
