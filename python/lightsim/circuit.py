"""Frequency-domain circuit simulation: the S-matrix of a netlist of
components, each described by an S-parameter model.

Conventions (the same as SAX, so the models here drop into SAX unchanged
and SAX models drop in here):

* a **model** is a function of keyword arguments with defaults, among them
  ``wl``, the vacuum wavelength in **micrometres** (a scalar or an array);
  lengths are in micrometres, as in gdsfactory;
* it returns an **SDict**: ``{(port_in, port_out): value}``, the complex
  amplitude leaving ``port_out`` per unit amplitude entering ``port_in``.
  Pairs that are absent are zero. Port names may carry a mode index,
  ``"o1@0"``, ``"o1@1"`` (several modes per port);
* in matrix form this is ``S[out, in] = sdict[(in, out)]``, the transpose
  of SAX's ``sdense`` layout, and lightsim's own S-parameter files key the
  same entry as ``"out@m,in@n"`` (see :func:`model_from_npz`).

:func:`circuit` takes a netlist (gdsfactory's ``component.get_netlist()``,
flat or recursive, or SAX's ``instances``/``connections``/``ports`` form)
and a dict of models and returns a function with the same calling
convention: ``circuit_fn(wl=..., **instance_overrides) -> SDict`` over the
external ports. Every instance's model receives the instance's
``settings`` (and the numbers in its ``info``, e.g. the length of a
gdsfactory Euler bend) for the arguments the model accepts.

The network is solved exactly at every wavelength: with ``S`` the
block-diagonal matrix of all instance ports and ``C`` the connection
permutation, the external S-matrix is
``S_EE + S_EI C (I - S_II C)^-1 S_IE`` (linear solve, batched over
wavelengths; sparse LU for large circuits), so feedback loops such as rings
are handled. This is a linear, steady-state (frequency-domain) simulation:
time-domain signals, modulation dynamics and nonlinear effects (Kerr,
two-photon absorption, thermal or carrier effects that depend on the
power) need a transient circuit solver, which lightsim does not have.
"""

from __future__ import annotations

import inspect
import math
import operator
import os
import warnings
from typing import Callable

import numpy as np

__all__ = [
    "circuit",
    "simulate",
    "sdense",
    "sdict",
    "reciprocal",
    "straight",
    "bend",
    "phase_shifter",
    "attenuator",
    "coupler",
    "coupler_ideal",
    "coupler_ring",
    "mmi1x2",
    "mmi2x2",
    "grating_coupler",
    "ring_single",
    "ring_double",
    "model_from_npz",
    "model_from_sparameters",
    "straight_from_cross_section",
    "bend_from_cross_section",
    "coupler_from_cross_section",
    "DEFAULT_MODELS",
]


# ----------------------------------------------------------------------------
# array namespace: numpy, or jax.numpy when called with jax arrays (inside
# SAX's jitted circuits)
# ----------------------------------------------------------------------------


def _xp(*args):
    for a in args:
        if isinstance(a, _Sym):  # traced (see trace_models in circuit)
            return _SYM_NS
    for a in args:
        if type(a).__module__.startswith("jax"):
            import jax.numpy as jnp

            return jnp
    return np


def reciprocal(sd: dict) -> dict:
    """Add ``(b, a)`` for every ``(a, b)`` (passive reciprocal device)."""
    out = dict(sd)
    for (a, b), v in sd.items():
        out.setdefault((b, a), v)
    return out


def sdense(sd: dict, ports=None):
    """SDict -> ``(S, ports)`` with ``S[..., out, in]`` (lightsim / usual
    order; SAX's ``sdense`` is the transpose)."""
    names = list(ports) if ports is not None else sorted({p for k in sd for p in k}, key=_port_key)
    idx = {p: i for i, p in enumerate(names)}
    shape = np.broadcast_shapes(*[np.shape(v) for v in sd.values()]) if sd else ()
    S = np.zeros(shape + (len(names), len(names)), dtype=complex)
    for (a, b), v in sd.items():
        if a in idx and b in idx:
            S[..., idx[b], idx[a]] = v
    return S, names


def sdict(S, ports) -> dict:
    """``(S[..., out, in], ports)`` -> SDict."""
    return {(a, b): S[..., j, i] for i, a in enumerate(ports) for j, b in enumerate(ports)}


def _port_key(p: str):
    import re

    name, _, mode = p.partition("@")
    m = re.match(r"([A-Za-z_]*)(\d*)", name)
    return (m.group(1), int(m.group(2)) if m.group(2) else -1, name, int(mode) if mode else 0)


# ----------------------------------------------------------------------------
# traced models: the per-point work of a model compiled into the solve
# ----------------------------------------------------------------------------

#: functions of the model codes, in the order of the Rust binding's FUNS
_FUNS = ["neg", "exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "abs", "real", "imag", "conj", "arctan", "ellipk", "add", "sub", "mul", "div", "pow", "max", "min", "gt", "where", "divr"]
_FUN = {n: k + 2 for k, n in enumerate(_FUNS)}
_REAL_OUT = {"abs", "real", "imag", "gt", "ellipk"}


class _Untraceable(Exception):
    pass


class _Trace:
    """The code of one model call: instructions on registers, the concrete
    arrays it reads (``params``) and its constants."""

    def __init__(self):
        self.code = []  # (kind, dst, a, b, c, real, d, e, f): kind 0 param, 1 const, 100 exp recurrence, else function
        self.consts = []  # per instruction (re, im)
        self.params = []
        self._pid = {}
        self._cse = {}  # identical instructions share one register (same inputs, same arithmetic)
        self.n = 0
        self.var = None  # the traced variable's values (for the exp recurrence)
        self.derived = {}  # parameter index -> ("var",) or ("t" | "h", base): computed from the variable
        self.grids = []  # (base, uniform) of every grid test, in order
        self.recur = True

    def _emit(self, kind, a=0, b=0, c=0, real=False, const=(0.0, 0.0), d=0, e=0, f=0):
        key = (kind, a, b, c, int(real), d, e, f)
        r = self._cse.get(key)
        if r is not None:
            return r
        self._cse[key] = r = self.n
        self.n += 1
        self.code.append((kind, r, a, b, c, int(real), d, e, f))
        self.consts.append(const)
        return r

    def param(self, arr):
        k = self._pid.get(id(arr))
        if k is None:
            k = self._pid[id(arr)] = len(self.params)
            self.params.append(arr)
        return self._emit(0, a=k)

    def select(self, g, G, nd):
        """The trace of member ``g`` of a stacked call (``G`` members on the
        leading axis of an ``nd``-dimensional result): the same code, the
        parameters taken at ``g`` where they vary along that axis."""
        cache = self.__dict__.setdefault("_sel", {})
        sub = cache.get((g, G, nd))
        if sub is None:
            sub = _Trace()
            sub.code, sub.consts, sub.n, sub.var, sub.recur = self.code, self.consts, self.n, self.var, self.recur
            for p_ in self.params:
                k = p_.ndim - nd  # p_ is right-aligned with the result
                if p_.ndim == nd and p_.shape[0] == G and G > 1:
                    sub.params.append(p_[g])
                elif p_.ndim == nd and p_.shape[0] == 1:
                    sub.params.append(p_[0])
                elif k < 0:
                    sub.params.append(p_)
                else:
                    raise _Untraceable("parameter of an unexpected shape in a stacked trace")
            cache[(g, G, nd)] = sub
        return sub

    def const(self, z):
        z = complex(z)
        key = ("c", z.real, z.imag, math.copysign(1.0, z.real), math.copysign(1.0, z.imag))  # +0 and -0 apart
        r = self._cse.get(key)
        if r is not None and z == z:
            return r
        self._cse[key] = r = self.n
        self.n += 1
        self.code.append((1, r, 0, 0, 0, int(z.imag == 0), 0, 0, 0))
        self.consts.append((z.real, z.imag))
        return r


class _Sym:
    """A traced array: a register of a :class:`_Trace`, its shape and whether
    it is real. Arithmetic, numpy ufuncs and the functions of
    :data:`_SYM_NS` (the array namespace of the models) record instructions;
    anything else raises :class:`_Untraceable`."""

    __array_priority__ = 1000

    def __init__(self, tr, reg, real, shape, poly=None):
        self.tr, self.reg, self.isreal, self.shape = tr, reg, real, tuple(shape)
        # the value as a polynomial of degree <= 2 in the traced variable x
        # ("x") or in 1/x ("u"), with concrete coefficients: {base: [c0, c1, c2]}
        self.poly = poly

    ndim = property(lambda self: len(self.shape))
    size = property(lambda self: int(np.prod(self.shape)) if self.shape else 1)
    dtype = property(lambda self: np.dtype(float) if self.isreal else np.dtype(complex))
    real = property(lambda self: _apply("real", self))
    imag = property(lambda self: _apply("imag", self))

    def conj(self):
        return _apply("conj", self)

    conjugate = conj

    def __array_namespace__(self, api_version=None):
        return _SYM_NS

    def __array__(self, *a, **k):
        raise _Untraceable("a traced array cannot be converted to a numpy array")

    def __bool__(self):
        raise _Untraceable("control flow on a traced array")

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d traced array")
        return self.shape[0]

    def __getitem__(self, key):
        # one member of a stacked call: an integer on the leading axis
        if isinstance(key, (int, np.integer)) and self.shape and not isinstance(key, bool):
            return _Sym(self.tr.select(int(key), self.shape[0], len(self.shape)), self.reg, self.isreal, self.shape[1:], None)
        raise _Untraceable("indexing a traced array")

    def __add__(self, o):
        return _apply("add", self, o)

    def __radd__(self, o):
        return _apply("add", o, self)

    def __sub__(self, o):
        return _apply("sub", self, o)

    def __rsub__(self, o):
        return _apply("sub", o, self)

    def __mul__(self, o):
        return _apply("mul", self, o)

    def __rmul__(self, o):
        return _apply("mul", o, self)

    def __truediv__(self, o):
        return _apply("div", self, o)

    def __rtruediv__(self, o):
        return _apply("div", o, self)

    def __pow__(self, o):
        return _apply("pow", self, o)

    def __rpow__(self, o):
        return _apply("pow", o, self)

    def __neg__(self):
        return _apply("neg", self)

    def __pos__(self):
        return self

    def __abs__(self):
        return _apply("abs", self)

    def __gt__(self, o):
        return _apply("gt", self, o)

    def __lt__(self, o):
        return _apply("gt", o, self)

    _UFUNCS = {"add": "add", "subtract": "sub", "multiply": "mul", "true_divide": "div", "divide": "div", "power": "pow", "negative": "neg", "exp": "exp", "log": "log", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "absolute": "abs", "conjugate": "conj", "arctan": "arctan", "maximum": "max", "minimum": "min", "greater": "gt", "ellipk": "ellipk"}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == "less" and method == "__call__" and not kwargs:
            return _apply("gt", inputs[1], inputs[0])
        name = self._UFUNCS.get(ufunc.__name__)
        if method != "__call__" or name is None or kwargs:
            raise _Untraceable(f"ufunc {ufunc.__name__}")
        return _apply(name, *inputs)

    def __array_function__(self, func, types, args, kwargs):
        f = _SYM_FUNCS.get(func.__name__)
        if f is None:
            raise _Untraceable(f"numpy function {func.__name__}")
        return f(*args, **kwargs)


def _is_real(x) -> bool:
    if isinstance(x, _Sym):
        return x.isreal
    if isinstance(x, (bool, int, float)):
        return True
    if isinstance(x, complex):
        return False
    return np.asarray(x).dtype.kind in "biuf"


def _shape_of(x) -> tuple:
    if isinstance(x, _Sym):
        return x.shape
    if isinstance(x, (bool, int, float, complex)):
        return ()
    return np.shape(x)


def _bshape(shapes) -> tuple:
    """Broadcast shape (the common cases without numpy's machinery)."""
    out = ()
    for sh in shapes:
        if sh == out or not sh:
            continue
        if not out:
            out = sh
            continue
        out = np.broadcast_shapes(out, sh)
    return out


def _operand(tr, x):
    """Register of an operand: a symbol, a constant (0-d) or a parameter."""
    if isinstance(x, _Sym):
        if x.tr is not tr:
            raise _Untraceable("symbols of two traces")
        return x.reg
    if isinstance(x, (bool, int, float, complex)):
        return tr.const(x)
    a = np.asarray(x)
    if a.dtype.kind not in "biufc":
        raise _Untraceable(f"operand of type {a.dtype}")
    if a.ndim == 0:
        return tr.const(complex(a))
    return tr.param(a)


def _poly_of(x):
    """{base: coefficients} of an operand; a concrete operand is a constant
    in every base (key None)."""
    if isinstance(x, _Sym):
        return x.poly
    return {None: [x]}


def _padd(p, q, sign=1.0):
    if p is None or q is None:
        return None
    out = {}
    for base in {b for b in list(p) + list(q) if b is not None}:
        a = p.get(base) or p.get(None)
        b = q.get(base) or q.get(None)
        if a is None or b is None:
            continue
        n = max(len(a), len(b))
        a = list(a) + [0.0] * (n - len(a))
        b = list(b) + [0.0] * (n - len(b))
        out[base] = [x + sign * y for x, y in zip(a, b)]
    return out or None


def _pscale(p, c, inv=False):
    if p is None:
        return None
    return {b: [(x / c if inv else x * c) for x in v] for b, v in p.items() if b is not None} or None


def _pmul(p, q):
    if p is None or q is None:
        return None
    if None in p:
        return _pscale(q, p[None][0])
    if None in q:
        return _pscale(p, q[None][0])
    out = {}
    for base in set(p) & set(q):
        a, b = p[base], q[base]
        if len(a) + len(b) - 1 > 3:
            continue
        c = [0.0] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                c[i + j] = c[i + j] + x * y
        out[base] = c
    return out or None


def _monomial(v):
    """a for a coefficient list [0, a] (exactly zero constant term)."""
    if len(v) == 2 and np.all(np.asarray(v[0]) == 0):
        return v[1]
    return None


def _poly_result(name, xs):
    if name in ("add", "sub"):
        return _padd(_poly_of(xs[0]), _poly_of(xs[1]), 1.0 if name == "add" else -1.0)
    if name == "neg":
        return _pscale(_poly_of(xs[0]), -1.0)
    if name == "mul":
        return _pmul(_poly_of(xs[0]), _poly_of(xs[1]))
    if name == "div":
        a, b = xs
        if not isinstance(b, _Sym):
            return _pscale(_poly_of(a), b, inv=True)
        if b.poly:  # division by a monomial k x (or k u): (c0 + c1 x) / (k x) = c1 / k + (c0 / k) u
            pa = _poly_of(a)
            out = {}
            for base, other in (("x", "u"), ("u", "x")):
                m = _monomial(b.poly.get(base, []))
                if m is None or pa is None:
                    continue
                num = pa.get(base) or pa.get(None)
                if num is None or len(num) > 2:
                    continue
                c0 = num[0]
                c1 = num[1] if len(num) > 1 else 0.0
                out[other] = [c1 / m, c0 / m]
            return out or None
    return None


_GRIDS = {}  # (id of the traced variable, base) -> (variable, t, h): one test per call, not per model


def _uniform_grid(var, base):
    """``(t, h)``: ``t`` the variable (base "x") or its reciprocal (base "u")
    and ``h`` its mean spacing along the last axis when every point lies
    within 16 ulp of the straight line; ``(None, None)`` otherwise."""
    key = (id(var), base)
    hit = _GRIDS.get(key)
    if hit is not None and hit[0] is var:
        return hit[1], hit[2]
    grid = getattr(_core(), "circuit_uniform_grid", None)
    if grid is not None and isinstance(var, np.ndarray) and var.dtype == np.float64 and var.ndim >= 1 and var.shape[-1] >= 2 and var.flags.c_contiguous:
        # compiled and threaded: the same operations, the same t, h and outcome
        ok, t, h = grid(var, base != "x")
        if t is None:
            t = var
        if not ok:
            t = h = None
    else:
        t = var if base == "x" else 1.0 / var
        n = t.shape[-1]
        h = (t[..., -1:] - t[..., :1]) / (n - 1)  # mean spacing: its rounding error is n times smaller than a difference's
        dev = t - (t[..., :1] + h * np.arange(n))
        if not np.all(np.isfinite(t)) or np.any(h == 0) or np.max(np.abs(dev)) > 16 * np.finfo(float).eps * np.max(np.abs(t)):
            t = h = None
    if len(_GRIDS) > 16:
        _GRIDS.clear()
    _GRIDS[key] = (var, t, h)
    return t, h


def _exp_poly(tr, x):
    """``exp(x)`` as the recurrence instruction when ``x`` is a polynomial of
    degree <= 2 in x or 1/x whose coefficients do not change along the
    batch's last axis, on a grid uniform along that axis; else None."""
    if not x.poly or not x.shape or x.shape[-1] < 2 or tr.var is None:
        return None
    nd = len(x.shape)
    for base, coef in x.poly.items():
        if base is None or not 2 <= len(coef) <= 3:
            continue
        if any(np.ndim(c) == nd and np.shape(c)[-1] > 1 for c in coef):
            continue  # a coefficient varies along the sweep axis
        if np.ndim(tr.var) < 1 or np.shape(tr.var)[-1] != x.shape[-1]:
            continue
        t, h = _uniform_grid(tr.var, base)
        tr.grids.append((base, t is not None))
        if t is None:
            continue  # not uniform to rounding
        deg = len(coef) - 1
        regs = [_operand(tr, c) for c in coef] + ([tr.const(0.0)] if deg == 1 else [])
        rt, rh = tr.param(t), tr.param(h)
        tr.derived[tr.code[rt][2]] = ("t", base)
        tr.derived[tr.code[rh][2]] = ("h", base)
        real = all(_is_real(c) for c in coef)
        r = tr._emit(100, rt, rh, regs[0], real=real, d=regs[1], e=regs[2], f=deg)
        return _Sym(tr, r, real, x.shape)
    return None


def _apply(name, *xs):
    tr = next(x for x in xs if isinstance(x, _Sym)).tr
    if name == "exp" and tr.recur:
        s = _exp_poly(tr, xs[0])
        if s is not None:
            return s
    shape = _bshape([_shape_of(x) for x in xs])
    regs = [_operand(tr, x) for x in xs] + [0, 0]
    if name == "where":  # where(cond, a, b): the condition is the third operand of the code
        real = _is_real(xs[1]) and _is_real(xs[2])
        r = tr._emit(_FUN["where"], regs[1], regs[2], regs[0], real=real)
    else:
        real = name in _REAL_OUT or all(_is_real(x) for x in xs)
        # a complex value over a real one: two real divisions (as numpy), not the complex quotient
        kind = _FUN["divr"] if name == "div" and not real and _is_real(xs[1]) else _FUN[name]
        r = tr._emit(kind, regs[0], regs[1], 0, real=real)
    try:
        poly = _poly_result(name, xs) if name != "where" else None
    except Exception:  # coefficients that do not combine
        poly = None
    return _Sym(tr, r, real, shape, poly)


def _sym_asarray(x, dtype=None, **kw):
    if isinstance(x, _Sym):
        if dtype is not None and np.dtype(dtype).kind == "c" and x.isreal:
            return _apply("add", x, 0j)
        if dtype is not None and np.dtype(dtype).kind == "f" and not x.isreal:
            raise _Untraceable("complex traced array converted to real")
        return x
    return np.asarray(x, dtype=dtype, **kw)


def _sym_clip(x, lo, hi, **kw):
    return _apply("min", _apply("max", x, lo), hi)


def _sym_ones_like(x, dtype=None, **kw):
    return np.ones(np.shape(x), dtype=dtype or (float if _is_real(x) else complex))


def _sym_zeros_like(x, dtype=None, **kw):
    return np.zeros(np.shape(x), dtype=dtype or (float if _is_real(x) else complex))


def _sym_broadcast_to(x, shape, **kw):
    if isinstance(x, _Sym):
        return _Sym(x.tr, x.reg, x.isreal, np.broadcast_shapes(x.shape, tuple(shape)), x.poly)
    return np.broadcast_to(x, shape)


_SYM_FUNCS = {
    "where": lambda c, a, b: _apply("where", c, a, b),
    "real": lambda x: _apply("real", x) if isinstance(x, _Sym) else np.real(x),
    "imag": lambda x: _apply("imag", x) if isinstance(x, _Sym) else np.imag(x),
    "conj": lambda x: _apply("conj", x),
    "conjugate": lambda x: _apply("conj", x),
    "clip": _sym_clip,
    "ones_like": _sym_ones_like,
    "zeros_like": _sym_zeros_like,
    "broadcast_to": _sym_broadcast_to,
    "abs": lambda x: _apply("abs", x),
    "shape": lambda x: np.shape(x) if not isinstance(x, _Sym) else x.shape,
    "ndim": lambda x: np.ndim(x) if not isinstance(x, _Sym) else len(x.shape),
    "size": lambda x, axis=None: np.size(x, axis) if not isinstance(x, _Sym) else (x.size if axis is None else x.shape[axis]),
}


class _SymNamespace:
    """The array namespace models see for traced arrays (``x.__array_namespace__()``)."""

    pi, e, inf, nan, newaxis = np.pi, np.e, np.inf, np.nan, None
    float64, complex128 = np.float64, np.complex128
    asarray = staticmethod(_sym_asarray)
    clip = staticmethod(_sym_clip)
    ones_like = staticmethod(_sym_ones_like)
    zeros_like = staticmethod(_sym_zeros_like)
    broadcast_to = staticmethod(_sym_broadcast_to)

    def __getattr__(self, name):
        if name in _SYM_FUNCS:
            return _SYM_FUNCS[name]
        uf = {"exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "absolute", "arctan", "maximum", "minimum", "greater", "add", "subtract", "multiply", "divide", "true_divide", "power", "negative"}
        if name in uf:
            npf = getattr(np, name)
            return lambda *xs: _apply(_Sym._UFUNCS[name], *xs) if any(isinstance(x, _Sym) for x in xs) else npf(*xs)
        if name == "ellipk":
            return lambda m: _apply("ellipk", m)
        raise _Untraceable(f"array function {name}")


_SYM_NS = _SymNamespace()


def _eval_code(code, consts, values, shape):
    """Reference interpreter of a model code in numpy (the check of a trace)."""
    import warnings as _w

    regs = []
    with _w.catch_warnings(), np.errstate(all="ignore"):
        _w.simplefilter("ignore")
        for (kind, dst, a, b, c, real, d, e, f), cst in zip(code, consts):
            if kind == 0:
                v = np.broadcast_to(np.asarray(values[a]), shape)
            elif kind == 1:
                v = np.full(shape, complex(*cst)) if cst[1] else np.full(shape, cst[0])
            elif kind == 100:  # exp(c0 + c1 t + c2 t^2), evaluated directly
                t = regs[a]
                v = np.exp(regs[c] + regs[d] * t + regs[e] * t * t)
                v = np.broadcast_to(np.real(v) if real else v, shape)
            else:
                f = _FUNS[kind - 2]
                x, y, z = regs[a] if regs else None, regs[b] if b < len(regs) else None, regs[c] if c < len(regs) else None
                if real:
                    x, y = (np.real(x) if x is not None else None), (np.real(y) if y is not None else None)
                v = {"neg": lambda: -x, "exp": lambda: np.exp(x), "log": lambda: np.log(x), "sqrt": lambda: np.sqrt(x), "sin": lambda: np.sin(x), "cos": lambda: np.cos(x), "tan": lambda: np.tan(x), "sinh": lambda: np.sinh(x), "cosh": lambda: np.cosh(x), "tanh": lambda: np.tanh(x), "abs": lambda: np.abs(x), "real": lambda: np.real(x), "imag": lambda: np.imag(x), "conj": lambda: np.conj(x), "arctan": lambda: np.arctan(x), "ellipk": lambda: _ellipk_np(np.real(x)), "add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y, "div": lambda: x / y, "pow": lambda: x**y, "max": lambda: np.maximum(np.real(x), np.real(y)), "min": lambda: np.minimum(np.real(x), np.real(y)), "gt": lambda: (np.real(x) > np.real(y)).astype(float), "where": lambda: np.where(np.real(z) != 0, x, y), "divr": lambda: x / np.real(y)}[f]()
                v = np.broadcast_to(np.real(v) if real else v, shape)
            regs.append(v)
    return regs


def _largest_argument(tr, regs=None) -> float:
    """Largest |argument| of the exponentials of a trace (reference values;
    ``regs``: the trace's registers, if already evaluated)."""
    big = 0.0
    try:
        if regs is None:
            shapes = [np.shape(p_) for p_ in tr.params]
            shape = _bshape(shapes) if shapes else ()
            regs = _eval_code(tr.code, tr.consts, tr.params, shape)
        for kind, dst, a, b, c, real, d, e, f in tr.code:
            if kind == _FUN["exp"]:
                big = max(big, float(np.nanmax(np.abs(regs[a])) if np.size(regs[a]) else 0.0))
            elif kind == 100:
                t = regs[a]
                arg = regs[c] + regs[d] * t + regs[e] * t * t
                big = max(big, float(np.nanmax(np.abs(arg))))
    except Exception:
        return 0.0
    return big


def _ellipk_np(m):
    from scipy.special import ellipk

    return ellipk(m)


def _trace_call(model, kw: dict, var: str):
    """Call ``model(**kw)`` with ``kw[var]`` (the wavelength or the
    frequency) traced. Returns the SDict, with traced entries as
    :class:`_Sym` and the others concrete, or raises :class:`_Untraceable`."""
    tr = _Trace()
    x = np.asarray(kw[var], dtype=float)
    tr.var = x
    sym = _Sym(tr, tr.param(x), True, x.shape, {"x": [0.0, 1.0]})
    tr.derived[0] = ("var",)
    k2 = dict(kw)
    k2[var] = sym
    try:
        with np.errstate(all="ignore"):
            out = model(**k2)
    except _Untraceable:
        raise
    except Exception as e:  # anything the tracer cannot follow
        raise _Untraceable(str(e)) from e
    if not isinstance(out, dict):
        raise _Untraceable("the model did not return an SDict")
    for v in out.values():
        if isinstance(v, _Sym):
            continue
        a = np.asarray(v)
        if a.dtype.kind not in "biufc":
            raise _Untraceable("an SDict entry is not numeric")
    return out, tr


def _check_trace(out, tr, ref, tol=None, sens=None):
    """Compare a traced SDict, evaluated by the reference interpreter, with
    the model's direct (numpy) result ``ref``. The bound, relative to the
    largest |entry|, is the larger of 1e-12, the rounding of the largest
    exponent argument the trace evaluates (64 eps |arg|: a phase of 10^4 rad
    is only known to about 1e-12 whichever order computes it), and 16 times
    ``sens``, the change of the direct result when the swept variable moves
    by 4 ulp (near a resonance the model itself is that sensitive to the
    order of its operations)."""
    sens = sens or {}
    if out.keys() != ref.keys():
        return False
    syms = {k: v for k, v in out.items() if isinstance(v, _Sym)}
    regs = None
    if syms:
        shape = np.broadcast_shapes(*[s.shape for s in syms.values()])
        regs = _eval_code(tr.code, tr.consts, tr.params, shape)
    if tol is None:
        tol = max(1e-12, 64 * np.finfo(float).eps * _largest_argument(tr, regs))
    for k, v in out.items():
        y = np.asarray(ref[k])
        x = np.broadcast_to(regs[v.reg], v.shape) if isinstance(v, _Sym) else np.asarray(v)
        if np.broadcast_shapes(x.shape, y.shape) != np.broadcast_shapes(y.shape, y.shape) and x.shape != y.shape:
            if np.broadcast_shapes(x.shape, y.shape) != x.shape:
                return False
        scale = max(1.0, float(np.nanmax(np.abs(y), initial=0.0)))
        with np.errstate(invalid="ignore"):
            if not np.allclose(x, y, rtol=0.0, atol=max(tol * scale, 16.0 * sens.get(k, 0.0)), equal_nan=True):
                return False
    return True


def _sensitivity(m, kw: dict, var: str, ref: dict) -> dict:
    """Largest change of each entry of the direct result when the swept
    variable is multiplied by 1 + 4 eps (the model's own rounding
    sensitivity)."""
    k2 = dict(kw)
    k2[var] = np.asarray(kw[var], dtype=float) * (1 + 4 * np.finfo(float).eps)
    try:
        with np.errstate(all="ignore"):
            r2 = m(**k2)
        return {k: float(np.nanmax(np.abs(np.asarray(r2[k]) - np.asarray(ref[k])), initial=0.0)) for k in ref}
    except Exception:
        return {}


#: Most points along each axis of the swept variable on which the first
#: traced call of a model is checked (a uniform subsample of the call)
_CHECK_POINTS = 256
#: Most points along each other batch axis (trials, stacked instances) of
#: that check: the code is the same at every point of those axes
_CHECK_POINTS_OTHER = 16


def _check_sample(kw: dict, var: str):
    """``kw`` with every axis along which the swept variable has more than
    ``_CHECK_POINTS`` points, and every other batch axis of the arguments
    longer than ``_CHECK_POINTS_OTHER``, taken with a uniform stride (in
    every argument that spans that axis), or None when nothing is to be
    cut. The first-use
    check of a traced model runs on it: the trace is the same code applied
    point by point, so a subsample spanning the sweep checks that code."""
    x = kw[var]
    if not isinstance(x, np.ndarray) or x.ndim == 0:
        return None
    try:
        shape = np.broadcast_shapes(*[v.shape for v in kw.values() if isinstance(v, np.ndarray)])
    except ValueError:
        shape = x.shape
    cut = {}  # axis from the right -> (length, stride)
    for j in range(1, len(shape) + 1):
        n = shape[-j]
        lim = _CHECK_POINTS if j <= x.ndim and x.shape[-j] == n else _CHECK_POINTS_OTHER
        if n > lim:
            cut[j] = (n, -(-(n - 1) // (lim - 1)))
    if not cut:
        return None
    out = {}
    for a, v in kw.items():
        if isinstance(v, np.ndarray) and v.ndim:
            idx = [slice(None)] * v.ndim
            for j, (n, st) in cut.items():
                if j <= v.ndim and v.shape[-j] == n:
                    idx[v.ndim - j] = slice(None, None, st)
            v = v[tuple(idx)]
        out[a] = v
    return out


def _same_program(t1, o1: dict, t2, o2: dict) -> bool:
    """Two traces of a model compute the same thing point by point: the
    same instructions, constants, variable-derived parameters and grid
    tests, and the same entries traced, in the same registers."""
    if t1.code != t2.code or t1.consts != t2.consts or t1.derived != t2.derived or t1.recur != t2.recur:
        return False
    if [u for _, u in t1.grids] != [u for _, u in t2.grids] or o1.keys() != o2.keys():
        return False
    for k in o1:
        a, b = o1[k], o2[k]
        if isinstance(a, _Sym) != isinstance(b, _Sym) or (isinstance(a, _Sym) and a.reg != b.reg):
            return False
    return True


# ----------------------------------------------------------------------------
# analytic models (SAX-compatible signatures; gdsfactory port names)
# ----------------------------------------------------------------------------


def straight(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.0) -> dict:
    """Waveguide of ``length`` µm with first-order dispersion: ``n_eff(wl) =
    neff - (wl - wl0) (ng - neff) / wl0``; transmission ``10^(-loss_dB_cm
    length 1e-4 / 20) exp(2 pi i n_eff length / wl)`` (as ``sax.models.
    straight``). Ports o1, o2."""
    xp = _xp(wl, neff, ng, length)
    wl = xp.asarray(wl)
    n = neff - (wl - wl0) * (ng - neff) / wl0
    t = 10 ** (-1e-4 * loss_dB_cm * length / 20) * xp.exp(2j * xp.pi * n * length / wl)
    return reciprocal({("o1", "o2"): t})


def bend(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.1, loss_dB=0.0) -> dict:
    """Bend as a straight of the bend's path ``length`` (gdsfactory reports it
    in the instance ``info``), plus a lumped ``loss_dB`` (mode mismatch and
    radiation of the whole bend). Ports o1, o2."""
    s = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)
    a = 10 ** (-loss_dB / 20)
    return {k: v * a for k, v in s.items()}


def phase_shifter(*, wl=1.55, neff=2.34, voltage=0.0, length=10.0, loss=0.0, v_pi=1.0) -> dict:
    """Phase shifter: ``exp(i (2 pi neff length / wl + pi voltage / v_pi))``,
    amplitude ``10^(-loss length / 20)`` (``loss`` in dB/µm, as
    ``sax.models.phase_shifter`` with ``v_pi = 1``). Ports o1, o2."""
    xp = _xp(wl, voltage)
    wl = xp.asarray(wl)
    t = 10 ** (-loss * length / 20) * xp.exp(1j * (2 * xp.pi * neff * length / wl + xp.pi * voltage / v_pi))
    return reciprocal({("o1", "o2"): t})


def attenuator(*, wl=1.55, loss=0.0) -> dict:
    """Flat attenuation of ``loss`` dB. Ports o1, o2."""
    xp = _xp(wl)
    return reciprocal({("o1", "o2"): 10 ** (-loss / 20) * xp.ones_like(xp.asarray(wl, dtype=float)) + 0j})


def coupler_ideal(*, wl=1.55, coupling=0.5) -> dict:
    """Lossless 2x2 coupler with power ``coupling`` to the cross port.
    gdsfactory ports: o1, o2 on the left (bottom, top), o3, o4 on the right
    (top, bottom); bar o1-o4 and o2-o3, cross o1-o3 and o2-o4 (cross
    amplitude ``i sqrt(coupling)``)."""
    xp = _xp(wl, coupling)
    one = xp.ones_like(xp.asarray(wl, dtype=float))
    k = xp.sqrt(coupling) * one
    t = xp.sqrt(1 - coupling) * one
    return reciprocal({("o1", "o4"): t + 0j, ("o2", "o3"): t + 0j, ("o1", "o3"): 1j * k, ("o2", "o4"): 1j * k})


def coupler(*, wl=1.55, wl0=1.55, length=0.0, coupling0=0.2, dk1=1.2435, dk2=5.3022, dn=0.02, dn1=0.1169, dn2=0.4821) -> dict:
    """Dispersive directional coupler, the model of ``sax.models.coupler``:
    ``phi = kappa0(wl) + pi dn(wl) length / wl``, bar ``cos(phi)``, cross
    ``-i sin(phi)``, with ``kappa0 = coupling0 + dk1 dwl + dk2 dwl²/2``
    (bends) and ``dn = dn + dn1 dwl + dn2 dwl²/2`` (supermode index
    difference), ``dwl = wl - wl0``. Ports as :func:`coupler_ideal`."""
    xp = _xp(wl, length)
    wl = xp.asarray(wl)
    dwl = wl - wl0
    d = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    k0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    phi = k0 + xp.pi * d / wl * length
    tau, kap = xp.cos(phi) + 0j, -xp.sin(phi)
    return reciprocal({("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): 1j * kap, ("o2", "o4"): 1j * kap})


def coupler_ring(*, wl=1.55, coupling=0.1, loss_dB=0.0) -> dict:
    """Bus-ring coupler with gdsfactory's ``coupler_ring`` ports: o1 (bus,
    west), o4 (bus, east), o2 and o3 (ring, west and east). Bar o1-o4 and
    o2-o3, cross o1-o3 and o2-o4 with power ``coupling``."""
    xp = _xp(wl, coupling)
    one = xp.ones_like(xp.asarray(wl, dtype=float))
    a = 10 ** (-loss_dB / 20)
    t, k = a * xp.sqrt(1 - coupling) * one + 0j, a * 1j * xp.sqrt(coupling) * one
    return reciprocal({("o1", "o4"): t, ("o2", "o3"): t, ("o1", "o3"): k, ("o2", "o4"): k})


def _mmi_amp(wl, wl0, fwhm, loss_dB):
    xp = _xp(wl)
    f, f0 = 1 / xp.asarray(wl), 1 / wl0
    bw = 1 / (wl0 - fwhm / 2) - 1 / (wl0 + fwhm / 2)
    sigma = bw / (2 * math.sqrt(2 * math.log(2)))
    return 10 ** (-abs(loss_dB) / 20) * xp.exp(-((f - f0) ** 2) / (2 * sigma**2))


def mmi1x2(*, wl=1.55, wl0=1.55, fwhm=0.2, loss_dB=0.3) -> dict:
    """1x2 splitter (o1 -> o2, o3), Gaussian in frequency, as
    ``sax.models.mmi1x2``."""
    t = _mmi_amp(wl, wl0, fwhm, loss_dB) / math.sqrt(2) + 0j
    return reciprocal({("o1", "o2"): t, ("o1", "o3"): t})


def mmi2x2(*, wl=1.55, wl0=1.55, fwhm=0.2, loss_dB=0.3, shift=0.0, splitting_ratio_cross=0.5, splitting_ratio_thru=0.5) -> dict:
    """2x2 MMI, as ``sax.models.mmi2x2``: thru o1-o3 and o2-o4, cross
    (``i``) o1-o4 and o2-o3 (gdsfactory ``mmi2x2`` ports)."""
    thru = _mmi_amp(wl, wl0, fwhm, loss_dB) * math.sqrt(splitting_ratio_thru) + 0j
    cross = 1j * _mmi_amp(wl, wl0 + shift, fwhm, loss_dB) * math.sqrt(splitting_ratio_cross)
    return reciprocal({("o1", "o3"): thru, ("o2", "o4"): thru, ("o1", "o4"): cross, ("o2", "o3"): cross})


def grating_coupler(*, wl=1.55, wl0=1.55, loss=0.0, reflection=0.0, reflection_fiber=0.0, bandwidth=0.040) -> dict:
    """Fibre grating coupler: Gaussian transmission of ``bandwidth`` (µm,
    full width at half maximum of the power... as in ``sax.models.
    grating_coupler``: the amplitude is ``exp(-(wl - wl0)² / (4 sigma²))``,
    ``sigma = bandwidth / (2 sqrt(2 ln 2))``), peak loss ``loss`` dB,
    reflections at the waveguide (o1) and fibre (o2) sides."""
    xp = _xp(wl)
    wl = xp.asarray(wl, dtype=float)
    one = xp.ones_like(wl)
    sigma = bandwidth / (2 * math.sqrt(2 * math.log(2)))
    t = 10 ** (-loss / 20) * xp.exp(-((wl - wl0) ** 2) / (4 * sigma**2)) + 0j
    return {("o1", "o1"): reflection * one + 0j, ("o1", "o2"): t, ("o2", "o1"): t, ("o2", "o2"): reflection_fiber * one + 0j}


def ring_single(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=50.0, loss_dB_cm=0.0, coupling=0.1) -> dict:
    """All-pass ring in closed form: ``t = (r - a e^{i phi}) / (1 - r a
    e^{i phi})`` with ``r = sqrt(1 - coupling)``, round trip ``a e^{i phi}``
    of a ring of circumference ``length`` µm (:func:`straight` dispersion).
    Ports o1 (in), o2 (through)."""
    rt = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    r = math.sqrt(1 - coupling)
    t = (r - rt) / (1 - r * rt)
    return reciprocal({("o1", "o2"): t})


def ring_double(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=50.0, loss_dB_cm=0.0, coupling1=0.1, coupling2=0.1) -> dict:
    """Add-drop ring in closed form (couplers ``coupling1`` at the input bus,
    ``coupling2`` at the drop bus, couplers at diametrically opposite
    points). Ports o1 (in), o2 (through), o3 (drop, on the input side), o4
    (add)."""
    xp = _xp(wl)
    rt = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    half = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length / 2, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    r1, r2 = math.sqrt(1 - coupling1), math.sqrt(1 - coupling2)
    k1, k2 = math.sqrt(coupling1), math.sqrt(coupling2)
    den = 1 - r1 * r2 * rt
    thru = (r1 - r2 * rt) / den
    drop = -k1 * k2 * half / den
    return reciprocal({("o1", "o2"): thru, ("o1", "o3"): drop, ("o4", "o3"): (r2 - r1 * rt) / den, ("o4", "o2"): drop})


# ----------------------------------------------------------------------------
# models from data and from the lightsim solvers
# ----------------------------------------------------------------------------


def model_from_sparameters(sp: dict, key_order: str = "auto", interpolation: str = "polar", keep_modes: bool | None = None) -> Callable:
    """Model that interpolates an S-parameter dictionary in wavelength.

    ``sp`` has ``"wavelengths"`` and complex arrays under keys
    ``"a@m,b@n"``. ``key_order``: ``"out,in"`` (lightsim's
    :func:`~lightsim.sparams.write_sparameters` and the gplugins Meep
    writer) or ``"in,out"`` (gplugins' Tidy3D writer and SAX); ``"auto"``
    takes ``"out,in"`` when the wavelengths are in metres (a lightsim file)
    and ``"in,out"`` otherwise. For reciprocal devices the two agree.
    Wavelengths in metres (lightsim) or micrometres (gplugins) are both
    accepted; the model takes ``wl`` in µm. ``interpolation``: ``"polar"``
    (magnitude and unwrapped phase, linear) or ``"linear"`` (real and
    imaginary parts, as gplugins). Port names drop ``@0`` when every key is
    mode 0, unless ``keep_modes``.
    """
    x = np.asarray(sp["wavelengths"], dtype=float).ravel()
    metres = x.max() < 1e-3
    if metres:
        x = x * 1e6
    order = ("out,in" if metres else "in,out") if key_order == "auto" else key_order
    keys = [k for k in sp if "@" in k and "," in k]
    modes = {p.split("@")[1] for k in keys for p in k.split(",")}
    km = keep_modes if keep_modes is not None else modes != {"0"}
    idx = np.argsort(x)
    x = x[idx]
    data = {}
    for k in keys:
        a, b = k.split(",")
        if not km:
            a, b = a.split("@")[0], b.split("@")[0]
        pin, pout = (b, a) if order == "out,in" else (a, b)
        v = np.asarray(sp[k], dtype=complex).ravel()[idx]
        data[(pin, pout)] = v
    phase = {k: np.unwrap(np.angle(v)) for k, v in data.items()}
    mag = {k: np.abs(v) for k, v in data.items()}

    def model(*, wl=1.55) -> dict:
        w = np.asarray(wl, dtype=float)
        out = {}
        for k, v in data.items():
            if len(x) == 1:
                out[k] = np.broadcast_to(v[0], w.shape).astype(complex)
            elif interpolation == "linear":
                out[k] = np.interp(w, x, v.real) + 1j * np.interp(w, x, v.imag)
            else:
                out[k] = np.interp(w, x, mag[k]) * np.exp(1j * np.interp(w, x, phase[k]))
        return out

    model.ports = sorted({p for k in data for p in k}, key=_port_key)
    return model


def model_from_npz(filepath, **kwargs) -> Callable:
    """Model from an S-parameter ``.npz`` file (the layout lightsim and
    gplugins share: keys ``"o1@0,o2@0"`` and ``wavelengths``); see
    :func:`model_from_sparameters` for the options."""
    with np.load(filepath) as f:
        sp = {k: f[k] for k in f.files}
    return model_from_sparameters(sp, **kwargs)


def _poly_neff(xs, wavelengths_um, mode, solve_kwargs):
    lam = np.asarray(wavelengths_um, dtype=float)
    ns = []
    for w in lam:
        ms = xs.solve(w * 1e-6, mode + 1, **solve_kwargs)
        ns.append(ms[mode].neff)
    ns = np.array(ns)
    deg = min(3, len(lam) - 1)
    return np.polyfit(lam, ns.real, deg), np.polyfit(lam, ns.imag, deg), ns


def straight_from_cross_section(xs, wavelengths=(1.5, 1.55, 1.6), mode: int = 0, extra_loss_dB_cm: float = 0.0, **solve_kwargs) -> Callable:
    """Straight-waveguide model from the mode solver: the complex effective
    index of ``mode`` of the :class:`~lightsim.xsection.CrossSection` is
    solved at ``wavelengths`` (µm) and fitted with a polynomial (degree up
    to 3), so the phase carries the full dispersion (group index and GVD)
    and the imaginary part the propagation loss (e.g. substrate leakage
    with a PML). ``extra_loss_dB_cm`` adds loss the solver does not see
    (roughness). The model is ``f(*, wl=1.55, length=10.0) -> SDict`` with
    ports o1, o2; ``f.neff(wl)`` gives the fitted index."""
    pr, pi_, ns = _poly_neff(xs, wavelengths, mode, solve_kwargs)

    def neff(wl):
        return np.polyval(pr, wl) + 1j * np.polyval(pi_, wl)

    def model(*, wl=1.55, length=10.0) -> dict:
        w = np.asarray(wl, dtype=float)
        n = neff(w)
        t = np.exp(2j * np.pi * n * length / w) * 10 ** (-1e-4 * extra_loss_dB_cm * length / 20)
        return reciprocal({("o1", "o2"): t})

    model.neff = neff
    model.samples = (np.asarray(wavelengths, dtype=float), ns)
    return model


def bend_from_cross_section(xs, radius: float, angle: float = 90.0, wavelengths=(1.5, 1.55, 1.6), mode: int = 0, pml_cells: int = 20, **solve_kwargs) -> Callable:
    """Circular-bend model from the mode solver: effective index (at the
    radius) and radiation loss of the bend mode of radius ``radius`` µm
    over ``angle`` degrees, and the two straight-to-bend junction losses
    from the overlap of the bend and straight modes, at ``wavelengths``
    (µm), fitted in wavelength. The cross-section needs room for the PML
    on its outer side (+u). Model ``f(*, wl=1.55) -> SDict``, ports o1, o2.
    """
    from .modes import mode_overlap

    lam = np.asarray(wavelengths, dtype=float)
    ns, junction = [], []
    pml = {"cells": (0, pml_cells, 0, 0)}
    for w in lam:
        s = xs.solve(w * 1e-6, mode + 1, **solve_kwargs)[mode]
        b = xs.solve(w * 1e-6, mode + 1, pml=pml, bend_radius=radius * 1e-6, **solve_kwargs)[mode]
        ns.append(b.neff)
        junction.append(mode_overlap(s, b, conjugate=False))
    ns, junction = np.array(ns), np.array(junction)
    deg = min(3, len(lam) - 1)
    pr, pim = np.polyfit(lam, ns.real, deg), np.polyfit(lam, ns.imag, deg)
    jr, ji = np.polyfit(lam, junction.real, deg), np.polyfit(lam, junction.imag, deg)
    length = radius * np.deg2rad(angle)

    def model(*, wl=1.55) -> dict:
        w = np.asarray(wl, dtype=float)
        n = np.polyval(pr, w) + 1j * np.polyval(pim, w)
        j = np.polyval(jr, w) + 1j * np.polyval(ji, w)
        return reciprocal({("o1", "o2"): j**2 * np.exp(2j * np.pi * n * length / w)})

    model.samples = (lam, ns, junction)
    return model


def coupler_from_cross_section(xs, wavelengths=(1.5, 1.55, 1.6), polarization: str = "te", coupling0: float = 0.0) -> Callable:
    """Directional-coupler model from the supermodes of the coupled section
    (``xs``, e.g. ``CrossSection.strip(gap=...)``): ``phi = coupling0 + pi
    dn(wl) length / wl``, bar ``cos(phi)``, cross ``-i sin(phi)``, with the
    supermode index difference ``dn`` solved at ``wavelengths`` (µm) and
    fitted. ``coupling0`` is the phase picked up in the bends (0: ignored).
    Model ``f(*, wl=1.55, length=10.0) -> SDict`` with the ports of
    :func:`coupler`; ``f.n_mean(wl)`` is the mean supermode index, used for
    the common phase."""
    lam = np.asarray(wavelengths, dtype=float)
    dn, nm = [], []
    for w in lam:
        ms = [m for m in xs.solve(w * 1e-6, 6, sort="te") if (m.te_fraction > 0.5) == (polarization == "te")]
        ms.sort(key=lambda m: -m.neff.real)
        dn.append(ms[0].neff.real - ms[1].neff.real)
        nm.append(0.5 * (ms[0].neff.real + ms[1].neff.real))
    deg = min(3, len(lam) - 1)
    pd, pm = np.polyfit(lam, dn, deg), np.polyfit(lam, nm, deg)

    def model(*, wl=1.55, length=10.0) -> dict:
        w = np.asarray(wl, dtype=float)
        phi = coupling0 + np.pi * np.polyval(pd, w) * length / w
        common = np.exp(2j * np.pi * np.polyval(pm, w) * length / w)
        tau, kap = common * np.cos(phi), -common * np.sin(phi)
        return reciprocal({("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): 1j * kap, ("o2", "o4"): 1j * kap})

    model.n_mean = lambda wl: np.polyval(pm, wl)
    model.dn = lambda wl: np.polyval(pd, wl)
    return model


#: Models for the gdsfactory components of common circuits (SAX defaults).
DEFAULT_MODELS = {
    "straight": straight,
    "bend_euler": bend,
    "bend_circular": bend,
    "bend_s": bend,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "coupler": coupler,
    "coupler_ring": coupler_ring,
    "grating_coupler_elliptical": grating_coupler,
    "grating_coupler_elliptical_te": grating_coupler,
    "phase_shifter": phase_shifter,
}


# ----------------------------------------------------------------------------
# netlists
# ----------------------------------------------------------------------------


_WL_DEFAULT = object()  # circuit_fn called without wl


def _split(ref: str):
    inst, port = ref.split(",")
    return inst.strip(), port.strip()


def _normalise(netlist: dict) -> dict:
    """gdsfactory (``nets``) or SAX (``connections``) netlist -> instances,
    pairs, ports."""
    inst = {}
    for name, v in netlist["instances"].items():
        if isinstance(v, str):
            inst[name] = {"component": v, "settings": {}, "info": {}}
        else:
            inst[name] = {"component": v.get("component"), "settings": dict(v.get("settings") or {}), "info": dict(v.get("info") or {})}
    pairs = []
    if "nets" in netlist and netlist["nets"]:
        for n in netlist["nets"]:
            pairs.append((_split(n["p1"]), _split(n["p2"])))
    for a, b in (netlist.get("connections") or {}).items():
        pairs.append((_split(a), _split(b)))
    ports = {k: _split(v) for k, v in (netlist.get("ports") or {}).items()}
    return {"instances": inst, "pairs": pairs, "ports": ports}


def _call(model, settings: dict, info: dict, wl, overrides: dict):
    spec = _ModelCall(model, settings, info)
    return spec(wl, overrides)


#: speed of light in micrometres per second (``wl`` in µm from ``f`` in Hz)
_C_UM = 299_792_458.0e6


def _accepts(v) -> bool:
    return isinstance(v, (int, float, complex, np.ndarray)) or not isinstance(v, (dict, list, str, type(None)))


class _ModelCall:
    """An instance's model with the arguments that do not change between
    calls (its ``info`` and ``settings``) filtered once; ``__call__`` adds
    the per-call overrides and the wavelength with the rules of
    :func:`_call`."""

    __slots__ = ("model", "params", "takes_all", "static", "wants_wl", "wants_f")

    def __init__(self, model, settings: dict, info: dict):
        try:
            params = inspect.signature(model).parameters
        except (TypeError, ValueError):  # pragma: no cover
            params = {}
        self.model = model
        self.params = params
        self.takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        self.static = self._filter({}, (info, settings))
        self.wants_wl = "wl" in params or self.takes_all
        self.wants_f = "f" in params

    def _filter(self, kw: dict, sources) -> dict:
        params, takes_all = self.params, self.takes_all
        for src in sources:
            for k, v in src.items():
                if (k in params or takes_all) and k != "wl" and _accepts(v):
                    if isinstance(v, bool):
                        if k in params:
                            kw[k] = v
                        continue
                    kw[k] = v
        return kw

    def kwargs(self, wl, overrides: dict, wl_f=None, f_wl=None) -> dict:
        """The model's keyword arguments. ``wl_f``: the wavelength for
        models that do not take ``f`` (from a frequency sweep); ``f_wl``:
        the frequency for models that take ``f`` but not ``wl`` (from a
        wavelength given explicitly)."""
        kw = self._filter(dict(self.static), (overrides,)) if overrides else dict(self.static)
        if self.wants_wl:
            kw["wl"] = wl_f if (wl_f is not None and not self.wants_f) else wl
        elif f_wl is not None and self.wants_f and "f" not in kw:
            kw["f"] = f_wl
        return kw

    def __call__(self, wl, overrides: dict, wl_f=None, f_wl=None):
        return self.model(**self.kwargs(wl, overrides, wl_f, f_wl))


def _frozen(v, digests: dict):
    """A hashable stand-in for a model argument and whether it stays valid
    between calls: numbers, strings and tuples by value, numpy arrays by
    shape, type and a digest of their bytes (computed once per object and
    call, in ``digests``), anything else by identity (valid within one call
    only)."""
    if v is None or isinstance(v, (bool, int, float, complex, str)):
        return (type(v).__name__, v), True
    if isinstance(v, tuple):
        parts = [_frozen(x, digests) for x in v]
        return ("tuple",) + tuple(p for p, _ in parts), all(ok for _, ok in parts)
    if isinstance(v, (np.ndarray, np.generic)):
        d = digests.get(id(v))
        if d is None:
            a = np.ascontiguousarray(v)
            d = digests[id(v)] = (a.shape, a.dtype.str, _digest(a.view(np.uint8).reshape(-1) if a.size else b""))
        return ("array",) + d, True
    return ("id", id(v)), False


#: Arrays above this many bytes are hashed in pieces of this size on several
#: threads (hashlib releases the interpreter lock): a 10 MB argument took
#: 10 ms to hash in one thread, a third of a 32-thread call of 1.25 million
#: points
_DIGEST_PIECE = 1 << 20
_DIGEST_POOL = []


def _digest(b) -> bytes:
    """16-byte BLAKE2b digest of the bytes ``b``; large inputs are hashed in
    pieces of ``_DIGEST_PIECE`` bytes in parallel and the digest is that of
    the pieces' digests and the length (as fixed as the bytes)."""
    import hashlib

    n = len(b)
    if n <= 4 * _DIGEST_PIECE:
        return hashlib.blake2b(b, digest_size=16).digest()
    if not _DIGEST_POOL:
        from concurrent.futures import ThreadPoolExecutor

        _DIGEST_POOL.append(ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="lightsim-digest"))
    parts = _DIGEST_POOL[0].map(lambda a: hashlib.blake2b(b[a : a + _DIGEST_PIECE], digest_size=16).digest(), range(0, n, _DIGEST_PIECE))
    h = hashlib.blake2b(digest_size=16, person=b"lightsim-pieces")
    for p_ in parts:
        h.update(p_)
    h.update(n.to_bytes(8, "little"))
    return h.digest()


def _expand_ports(sd: dict) -> list:
    return sorted({p for k in sd for p in k}, key=_port_key)


def _modes_of(ports: list, port: str) -> list:
    """Port names of ``port`` in a model's port list: ``port`` itself, or
    ``port@0``, ``port@1``, ..."""
    if port in ports:
        return [port]
    return sorted([p for p in ports if p.split("@")[0] == port and "@" in p], key=_port_key)


def circuit(netlist: dict, models: dict | None = None, backend: str = "auto", recursive_netlists: dict | None = None, use_info: bool = True, threads: int | None = None, reuse_identical: bool = False, stack_instances: bool = False, trace_models: bool = False):
    """Build a circuit function from a netlist and models.

    Parameters
    ----------
    netlist : dict
        ``component.get_netlist()`` (gdsfactory: ``instances``, ``nets``,
        ``ports``), the dict of ``component.get_netlist(recursive=True)``
        (the top cell is the one no other cell instantiates), or a SAX
        netlist (``instances``, ``connections``, ``ports``).
    models : dict
        Component name -> model function. Missing names fall back to
        :data:`DEFAULT_MODELS` and, in a recursive netlist, to the cell's
        own netlist (a sub-circuit).
    backend : ``"auto"``, ``"joins"``, ``"dense"`` or ``"sparse"``.
        ``"joins"`` joins the instances one connection at a time (compiled,
        in parallel over the wavelengths); ``"dense"`` solves the matrix of
        all internal ports at every wavelength (batched), ``"sparse"`` with a
        sparse LU. ``"auto"`` takes the joins unless their operation count
        exceeds that of the dense solve, then dense (sparse above 400
        internal ports). All give the same S-matrix to rounding.
    use_info : bool
        Also pass the numbers of each instance's gdsfactory ``info`` (the
        path length of a bend, for example) to the models, as SAX 0.18 does
        with gdsfactory 9 netlists. ``False`` passes ``settings`` only.
    threads : int, optional
        Threads of the network solve (default: all cores, or
        ``RAYON_NUM_THREADS``). Given and above 1, the instances' models are
        also evaluated concurrently on that many threads when a call has
        enough points to gain from it (numpy releases the interpreter lock in
        its array loops); leave it unset for models that are not thread-safe.
    reuse_identical : bool
        Evaluate a model once for all instances that call it with the same
        arguments, and not at all for an instance whose model and arguments
        are those of the previous call (a Monte Carlo trial or an optimiser
        step re-evaluates only the instances it changes). Numbers, strings
        and tuples are compared by value, numpy arrays by their contents
        (a 128-bit digest), other objects by identity and within one call
        only. Only for models that are pure functions of their arguments (no
        random draws, no state); off by default.
    stack_instances : bool
        Evaluate all instances of one model in one call: arguments that
        differ between them (numbers, numpy arrays) are stacked along a new
        leading axis and the model's outputs split along it. For models
        written with array operations that broadcast (numpy semantics). The
        first stacked call of every group is checked against one call per
        instance; a group whose results differ, or whose model raises, is
        evaluated instance by instance from then on. Off by default.
    trace_models : bool
        Compile the per-wavelength work of the models into the network
        solve: every model is called once per call with a traced wavelength
        (or frequency); what does not depend on it is computed by numpy as
        usual, and the operations on it (arithmetic, exp, sqrt, the
        trigonometric and hyperbolic functions, abs, real, imag, where,
        clip, ...) are recorded and evaluated in compiled code at every
        point inside the solve, without numpy temporaries. For models
        written with array operations (numpy, or the namespace of their
        arguments, as lightsim's models and SAX's are). The first traced call
        of every model returns the direct result, after checking the trace
        against direct calls on a uniform subsample of the sweep (at most
        256 points along each axis); a model that cannot be traced, or whose
        result differs, is called directly from then on. Calls of fewer than 8192 points are not traced (a trace
        costs about 0.1 ms; small calls gain nothing), nor models whose code
        is a few operations without the recurrence below (it would save
        only numpy's temporaries). With
        ``reuse_identical``, a model whose other arguments are plain numbers
        is traced once and its code reused with each new wavelength. Off by
        default.

        Accuracy: an exponential of a phase that is linear (or quadratic) in
        the wavelength or in its inverse, on a grid uniform in that
        variable, is computed by a recurrence (a true exponential every 32
        points, products in between), and a phase is only known to about
        eps times its size whichever way it is computed: the traced results
        differ from the direct numpy call by about 1.5 eps times the largest
        phase (measured: 3e-12 relative at 1e4 rad, 3e-11 at 1e5 rad, about
        twice numpy's own rounding of that phase). Near a sharp resonance a
        model can be more sensitive to the order of its operations than
        that (the resonator chip of the benchmarks, 250 trials of 5001
        points and 2000 refinement rows of 257: up to 1e-10 of the largest
        entry, 6e-10 of the entry itself at the bottom of a notch, where the
        direct call itself moves 8e-11 and 8e-10 for a 4-ulp change of the
        wavelength). The results do not depend on the number of threads.
        Leave the option off where results must match the direct calls to
        the last bits.

    Returns ``(circuit_fn, info)``. ``circuit_fn(wl=1.55, **overrides)``
    returns the SDict of the external ports; ``overrides`` maps an instance
    name to a dict of model arguments (``circuit_fn(wl=wl, top={"length":
    20})``) or sets a global argument passed to every model that takes it.
    ``circuit_fn(f=...)`` sweeps the frequency in hertz instead: models that
    take ``f`` receive it, models that take only ``wl`` receive ``wl = c /
    f`` in µm (and with ``wl`` given, models that take only ``f`` receive
    ``f = c / wl``). ``info`` lists the instances, their models and the port count.
    """
    models = dict(DEFAULT_MODELS, **(models or {}))
    if "instances" not in netlist:  # recursive netlist dict {cell: netlist}
        recursive_netlists = netlist
        used = {v["component"] if isinstance(v, dict) else v for n in netlist.values() for v in n.get("instances", {}).values()}
        tops = [k for k in netlist if k not in used]
        if len(tops) != 1:
            raise ValueError(f"cannot find the top cell of the recursive netlist (candidates: {tops})")
        netlist = netlist[tops[0]]
    net = _normalise(netlist)
    sub = {}
    for name, v in net["instances"].items():
        comp = v["component"]
        if comp in models:
            continue
        if recursive_netlists and comp in recursive_netlists:
            if comp not in sub:
                sub[comp], _ = circuit(recursive_netlists[comp], models, backend, recursive_netlists, use_info, threads, reuse_identical, stack_instances, trace_models)
            continue
        raise KeyError(f"no model for component {comp!r} (instance {name!r}); pass models={{{comp!r}: model}}")

    inst_names = net["instances"]
    S0 = frozenset(inst_names)
    # sub-circuits are inlined into one network (one solve per call, and the
    # instances of all cells share the model reuse and stacking); a netlist
    # that cannot be inlined is solved cell by cell as before
    flat = None
    if sub and _FLATTEN:
        try:
            flat = _flatten(net, models, recursive_netlists, {})
        except (KeyError, ValueError):
            flat = None
    if flat is not None:
        leaves, f_pairs, f_ports = flat
        subinsts = {}
    else:
        leaves = {(n,): (v, ((n, S0),)) for n, v in inst_names.items() if v["component"] in models}
        f_pairs = [(((a[0],), a[1]), ((b[0],), b[1])) for a, b in net["pairs"]]
        f_ports = {k: ((t[0],), t[1]) for k, t in net["ports"].items()}
        subinsts = {(n,): v["component"] for n, v in inst_names.items() if v["component"] not in models}
    calls = {key: _ModelCall(models[v["component"]], v["settings"], v["info"] if use_info else {}) for key, (v, _) in leaves.items()}
    levels = {key: lv for key, (_, lv) in leaves.items()}
    plans = {}
    last = {}  # reuse_identical: model results of the previous call, by arguments
    stacking = {}  # stack_instances: group key -> True (checked), False (not stackable)
    tracing = {}  # trace_models: model key -> True (checked), False (call directly), _GridDecisions (direct for some grid outcomes)
    replay = {} if reuse_identical else None  # trace_models: traces of calls with plain arguments, reused
    plain_jobs = [(key, {}) for key in levels]

    def fn(wl=_WL_DEFAULT, **overrides):
        # the frequency sweep variable: ``f`` (Hz) gives the wavelength of the
        # models that take ``wl`` only, ``wl`` (µm) the frequency of those
        # that take ``f`` only
        f = overrides.get("f") if "f" not in S0 else None
        wl_f = f_wl = None
        sub_wl = {} if wl is _WL_DEFAULT else {"wl": wl}
        if wl is _WL_DEFAULT:
            wl = 1.55
            if f is not None:
                wl_f = _C_UM / np.asarray(f, dtype=float)
        elif f is None:
            f_wl = _C_UM / np.asarray(wl, dtype=float)
        # overrides as the cell-by-cell solve applies them: an instance
        # override at its own level, every other key passed down to the
        # sub-circuits (where a key naming one of their instances overrides it)
        gcache = {}

        def down(O, S):
            g = gcache.get((id(O), S))
            if g is None:
                g = gcache[(id(O), S)] = {a: b for a, b in O.items() if a not in S}
            return g

        sdicts = {}
        jobs = [] if overrides else plain_jobs  # without overrides every instance gets none
        for key, lv in levels.items() if overrides else ():
            O = overrides
            for _, S in lv[:-1]:
                O = down(O, S)
            n, S = lv[-1]
            g = down(O, S)
            ov = O.get(n)
            if isinstance(ov, dict) and ov:
                o = dict(g)
                o.update(ov)
            else:
                o = g
            jobs.append((key, o))
        for key, comp in subinsts.items():  # in this thread (their models may use the pool)
            sdicts[key] = sub[comp](**sub_wl, **down(overrides, S0))
        work = [(calls[name].model, calls[name].kwargs(wl, o, wl_f, f_wl)) for name, o in jobs]
        owner = list(range(len(jobs)))
        res = {}
        if reuse_identical:
            # one evaluation per distinct (model, arguments) in this call, and
            # none for those whose arguments equal the previous call's
            seen, digests, keep, keys = {}, {}, {}, [None] * len(jobs)
            for k, (m, kw) in enumerate(work):
                parts = [(a, _frozen(v, digests)) for a, v in kw.items()]
                key = (id(m), tuple(sorted((a, p) for a, (p, _) in parts)))
                owner[k] = seen.setdefault(key, k)
                if all(ok for _, (_, ok) in parts):
                    keys[k] = key
                    if owner[k] == k and key in last:
                        res[k] = last[key]
        todo = [k for k in range(len(jobs)) if owner[k] == k and k not in res]
        pooled = threads is not None and threads > 1 and _at_least(work, todo, _MODEL_POOL_MIN_POINTS)
        tasks = _stack_groups(work, todo, stacking) if stack_instances else [[k] for k in todo]
        if pooled and stack_instances and not trace_models:  # split the stacked groups over the threads
            # (not when tracing: a trace costs the same for any group size, and
            # its per-point work runs in the solve, on every thread)
            tasks = [t[i : i + max(2, -(-len(t) // threads))] for t in tasks for i in range(0, len(t), max(2, -(-len(t) // threads)))]

        # small calls are not traced (see _TRACE_MIN_POINTS): decided once per call
        tracing_on = trace_models and bool(todo) and _at_least(work, todo, _TRACE_MIN_POINTS)

        def run(t):
            if len(t) == 1:
                m, kw = work[t[0]]
                return [_call_traced(m, kw, tracing, replay) if tracing_on else m(**kw)]
            return _call_stacked(work, t, stacking, (lambda m, kw: _call_traced(m, kw, tracing, replay)) if tracing_on else None)

        if not (stack_instances or tracing_on or pooled):  # one direct call per instance
            for k in todo:
                m, kw = work[k]
                res[k] = m(**kw)
        else:
            if pooled and len(tasks) > 1:
                outs = list(_pool(threads).map(run, tasks))
            else:
                outs = [run(t) for t in tasks]
            for t, o in zip(tasks, outs):
                res.update(zip(t, o))
        if tracing_on and any(isinstance(v, _Sym) for sd in res.values() for v in sd.values()):
            # traced SDicts that would be evaluated more than needed inside the
            # solve become arrays first (compiled, over their own shape): those
            # reused from the previous call, those several instances share
            # unless their code is short (evaluating it again per instance
            # costs less than writing and reading the arrays), and those that
            # do not span the whole batch
            batch = _batch_shape(res.values())
            shared = {}
            for k in owner:
                shared[owner[k]] = shared.get(owner[k], 0) + 1
            for k, sd in list(res.items()):
                syms = [v for v in sd.values() if isinstance(v, _Sym)]
                if not syms:
                    continue
                own = np.broadcast_shapes(*[v.shape for v in syms])
                cached = reuse_identical and keys[k] is not None and keys[k] in last and last[keys[k]] is sd
                if cached or (shared[k] > 1 and _code_length(syms) * (shared[k] - 1) > _SHARED_CODE_MAX) or int(np.prod(own)) < int(np.prod(batch)):
                    res[k] = _concrete(sd, threads)
        for k, (name, _) in enumerate(jobs):
            sdicts[name] = res[owner[k]]
        if reuse_identical:
            keep = {keys[k]: res[k] for k in res if keys[k] is not None}
            last.clear()
            last.update(keep)
        return _solve(sdicts, f_pairs, f_ports, backend, threads, plans, fixed=True)

    info = {"instances": {n: v["component"] for n, v in net["instances"].items()}, "subcircuits": list(sub), "ports": list(net["ports"])}
    return fn, info


_FLATTEN = True  # inline sub-circuits (False: solve them cell by cell)


def _flatten(net: dict, models: dict, recnets: dict, memo: dict):
    """Inline the sub-circuits of a normalised netlist. Returns ``(leaves,
    pairs, ports)``: ``leaves`` maps an instance path (tuple of names from
    the top) to ``(instance, levels)``, ``levels`` being the ``(name, names
    of that level)`` of each level of the path; pairs and ports refer to
    ``(path, port)``. A connection to a port a sub-circuit does not export
    raises KeyError."""
    S = frozenset(net["instances"])
    leaves, sub_ports, pairs, ports = {}, {}, [], {}
    for name, v in net["instances"].items():
        comp = v["component"]
        if comp in models:
            leaves[(name,)] = (v, ((name, S),))
            continue
        if comp not in memo:
            memo[comp] = None  # recursion guard
            memo[comp] = _flatten(_normalise(recnets[comp]), models, recnets, memo)
        if memo[comp] is None:
            raise ValueError(f"recursive cell {comp!r}")
        sl, sp, sx = memo[comp]
        for k, (vv, lv) in sl.items():
            leaves[(name,) + k] = (vv, ((name, S),) + lv)
        pairs += [(((name,) + a[0], a[1]), ((name,) + b[0], b[1])) for a, b in sp]
        sub_ports[name] = {pn: ((name,) + t[0], t[1]) for pn, t in sx.items()}

    def at(inst, port):
        if inst not in sub_ports:
            return ((inst,), port)
        sp = sub_ports[inst]
        if port in sp:
            return sp[port]
        base, _, mode = port.partition("@")
        if mode and base in sp and "@" not in sp[base][1]:
            return (sp[base][0], sp[base][1] + "@" + mode)
        raise KeyError(f"{inst},{port}: not a port of the sub-circuit")

    for a, b in net["pairs"]:
        pairs.append((at(*a), at(*b)))
    for pn, t in net["ports"].items():
        ports[pn] = at(*t)
    return leaves, pairs, ports


def solve_matrix(S, E, I, Cm, backend: str = "auto", adjoint: bool = False):
    """The linear algebra of :func:`solve_network`, on index arrays.

    ``S[k, out, in]`` holds all instance ports (``k`` over wavelengths),
    ``E`` and ``I`` are the external and internal port indices and ``Cm`` the
    symmetric connection matrix on ``I``. Returns ``(Sx, CX, RI)``:
    ``Sx = S_EE + S_EI C (1 - S_II C)^-1 S_IE`` and, with ``adjoint``, the
    per-wavelength factors ``CX = C (1 - S_II C)^-1 S_IE`` and
    ``RI = S_EI C (1 - S_II C)^-1`` from which the gradient with respect to
    ``S`` follows (:mod:`lightsim.invdes.network`); else ``None, None``."""
    S_EE = S[:, E][:, :, E]
    nI = len(I)
    nw = S.shape[0]
    if nI == 0:
        return S_EE, ([None] * nw if adjoint else None), ([None] * nw if adjoint else None)
    S_EI, S_IE, S_II = S[:, E][:, :, I], S[:, I][:, :, E], S[:, I][:, :, I]
    use_sparse = backend == "sparse" or (backend == "auto" and nI > 400)
    CXs, RIs = ([], []) if adjoint else (None, None)
    if use_sparse:
        import scipy.sparse as sp_
        import scipy.sparse.linalg as spla

        Cs = sp_.csr_matrix(Cm)
        Sx = np.empty_like(S_EE)
        for k in range(nw):
            A = sp_.identity(nI, format="csc") - sp_.csr_matrix(S_II[k]) @ Cs
            lu = spla.splu(A.tocsc())
            CX = Cs @ lu.solve(S_IE[k])
            Sx[k] = S_EE[k] + S_EI[k] @ CX
            if adjoint:
                CXs.append(CX)
                RIs.append(lu.solve(np.ascontiguousarray((S_EI[k] @ Cs.toarray()).T), trans="T").T)
    else:
        A = np.eye(nI)[None] - S_II @ Cm[None]
        CX = Cm[None] @ np.linalg.solve(A, S_IE)
        Sx = S_EE + S_EI @ CX
        if adjoint:
            RI = np.swapaxes(np.linalg.solve(np.swapaxes(A, 1, 2), np.swapaxes(S_EI @ Cm[None], 1, 2)), 1, 2)
            CXs, RIs = list(CX), list(RI)
    return Sx, CXs, RIs


def solve_network(sdicts: dict, pairs: list, ports: dict, backend: str = "auto", threads: int | None = None) -> dict:
    """External SDict of instances ``sdicts`` (name -> SDict) connected by
    ``pairs`` (((inst, port), (inst, port)), ...) with external ``ports``
    (name -> (inst, port)). Several modes per port connect mode by mode.
    ``backend`` and ``threads`` as in :func:`circuit`."""
    return _solve(sdicts, pairs, ports, backend, threads, _PLANS)


_PLANS: dict = {}  # network plans of solve_network calls, by port lists and connections
_PORTS: dict = {}  # sorted port lists, by the keys of an SDict


def _ports_of(sd: dict) -> list:
    try:
        key = tuple(sd)
        ps = _PORTS.get(key)
    except TypeError:  # pragma: no cover  (unhashable port names)
        return _expand_ports(sd)
    if ps is None:
        if len(_PORTS) > 4096:
            _PORTS.clear()
        ps = _PORTS[key] = _expand_ports(sd)
    return ps


def _numeric(v) -> bool:
    return isinstance(v, (int, float, complex, np.number, np.ndarray)) and not isinstance(v, bool) and not (isinstance(v, np.ndarray) and v.dtype.kind not in "biufc")


def _stack_groups(work, todo, stacking) -> list:
    """Group the model calls ``todo`` (indices into ``work``) that can share
    one stacked call: same model, same argument names, equal non-numeric
    arguments; groups marked not stackable are split again."""
    groups = {}
    for k in todo:
        m, kw = work[k]
        key = (id(m), tuple(sorted((a, None if _numeric(v) else _frozen(v, {})[0]) for a, v in kw.items())))
        groups.setdefault(key, []).append(k)
    tasks = []
    for key, ks in groups.items():
        if len(ks) > 1 and stacking.get(key) is not False:
            tasks.append(ks)
        else:
            tasks += [[k] for k in ks]
    return tasks


def _call_stacked(work, ks, stacking, caller=None) -> list:
    """One call of the model for the instances ``ks``: the numeric arguments
    that differ are stacked along a new leading axis (in front of the batch
    axes of every argument), the outputs are split along it."""
    m, kw0 = work[ks[0]]
    key = (id(m), tuple(sorted((a, None if _numeric(v) else _frozen(v, {})[0]) for a, v in kw0.items())))
    G = len(ks)
    nd = max((np.ndim(v) for k in ks for v in work[k][1].values() if _numeric(v)), default=0)
    kw = {}
    for a, v0 in kw0.items():
        vs = [work[k][1][a] for k in ks]
        if not _numeric(v0) or all(v is v0 for v in vs) or all(np.ndim(v) == 0 and v == v0 for v in vs):
            kw[a] = v0
            continue
        shape = np.broadcast_shapes(*[np.shape(v) for v in vs])
        st = np.stack([np.broadcast_to(np.asarray(v), shape) for v in vs])
        kw[a] = st.reshape((G,) + (1,) * (nd - len(shape)) + shape)
    singles = None
    try:
        out = caller(m, kw) if (caller is not None and stacking.get(key)) else m(**kw)
        parts = []
        for g in range(G):
            parts.append({p: (v[g] if np.ndim(v) == nd + 1 and np.shape(v)[0] == G else v) for p, v in out.items()})
    except Exception:  # the model does not broadcast over these arguments
        stacking[key] = False
        return [work[k][0](**work[k][1]) for k in ks]
    if key not in stacking:  # first stacked call of this group: check it
        # on a subsample of the batch first (see _stack_sample): when it
        # agrees the group stacks; otherwise every instance is called in full
        sample = _stack_sample(work, ks, nd)
        if sample is not None:
            idx, kws = sample
            try:
                small = [m(**k2) for k2 in kws]
                if any(np.ndim(v) > nd for a in parts for v in a.values()):
                    raise ValueError("outputs with more axes than the batch")
                cut = [{p: (np.asarray(v)[idx[-np.ndim(v) :]] if np.ndim(v) else v) for p, v in a.items()} for a in parts]
                if _stack_agrees(cut, small):
                    stacking[key] = True
                    return parts
            except Exception:
                pass
        singles = [work[k][0](**work[k][1]) for k in ks]
        ok = _stack_agrees(parts, singles)
        stacking[key] = ok
        if not ok:
            return singles
    return parts


#: Most points along each batch axis on which the first stacked call of a
#: group is compared with the instances called one by one
_STACK_CHECK_POINTS = 16


def _stack_sample(work, ks, nd: int):
    """For the first-call check of a stacked group: the index (one uniform
    stride per batch axis longer than ``_STACK_CHECK_POINTS``, over the
    ``nd`` batch axes counted from the right) and each instance's arguments
    cut with it; None when nothing is to be cut or the arguments do not
    share the batch axes."""
    try:
        shape = np.broadcast_shapes(*[np.shape(v) for k in ks for v in work[k][1].values() if _numeric(v)])
    except ValueError:
        return None
    shape = (1,) * (nd - len(shape)) + tuple(shape)
    idx = tuple(slice(None, None, -(-(n - 1) // (_STACK_CHECK_POINTS - 1))) if n > _STACK_CHECK_POINTS else slice(None) for n in shape)
    if all(s.step is None for s in idx):
        return None
    kws = []
    for k in ks:
        k2 = {}
        for a, v in work[k][1].items():
            if _numeric(v) and np.ndim(v):
                v = np.asarray(v)
                v = v[tuple(idx[nd - v.ndim + j] if v.shape[j] == shape[nd - v.ndim + j] else slice(None) for j in range(v.ndim))]
            k2[a] = v
        kws.append(k2)
    return idx, kws


def _stack_agrees(parts, singles) -> bool:
    """The outputs of a stacked call (split per instance) equal those of the
    instances called one by one (same entries, shapes that broadcast to the
    single call's, values to 1e-12)."""
    for a, b in zip(parts, singles):
        if a.keys() != b.keys():
            return False
        for p in a:
            x, y = np.asarray(a[p]), np.asarray(b[p])
            if np.broadcast_shapes(x.shape, y.shape) != y.shape and x.shape != y.shape:
                return False
            if not np.allclose(x, y, rtol=1e-12, atol=1e-14 * max(1.0, float(np.max(np.abs(y), initial=0.0)))):
                return False
    return True


#: Smallest call (points of its broadcast arguments) whose model is traced:
#: below it the fixed cost of a trace exceeds the per-point work it saves
#: (measured: 8,000 to 10,000 points for one exponential per instance)
_TRACE_MIN_POINTS = 8192
#: Smallest code (instructions per point, transcendental functions 16) of a
#: traced model without an exponential recurrence: shorter ones are called
#: directly (measured: one complex exponential per point, as numpy's, was no faster traced)
_TRACE_MIN_CODE = 48


class _GridDecisions:
    """A model's "call directly" decisions per outcome of the grid tests of
    its trace (one bool per grid base: the sweep uniform in it or not), for
    models not worth tracing only because a sweep was not uniform."""

    __slots__ = ("bases", "direct")

    def __init__(self, bases):
        self.bases, self.direct = bases, set()

    def outcome(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim < 1 or x.shape[-1] < 2:
            return tuple(False for _ in self.bases)
        return tuple(_uniform_grid(x, b)[0] is not None for b in self.bases)


def _call_traced(m, kw: dict, tracing: dict, replay: dict | None = None):
    """``m(**kw)`` with the wavelength (or frequency) traced when the model
    allows it (see ``trace_models`` in :func:`circuit`). With ``replay`` (a
    dict kept between calls; models declared pure by ``reuse_identical``),
    a call whose other arguments are all plain values equal to an earlier
    call's reuses that call's trace with the new variable: the same code
    and the same parameters, as tracing again would give."""
    var = "wl" if "wl" in kw else ("f" if "f" in kw else None)
    if var is None or np.ndim(kw[var]) == 0 or isinstance(kw[var], _Sym):
        return m(**kw)
    try:
        points = int(np.prod(np.broadcast_shapes(*[np.shape(v) for v in kw.values() if isinstance(v, np.ndarray)])))
    except ValueError:
        points = 0
    if points < _TRACE_MIN_POINTS:  # a trace costs about 0.1 ms per call; small calls gain nothing
        return m(**kw)
    key = (id(m), tuple(sorted(kw)))
    state = tracing.get(key)
    if state is False:
        return m(**kw)
    grid_out = None
    if isinstance(state, _GridDecisions):
        # decided only for the grid outcomes seen so far: a call whose grid
        # tests come out differently is traced and checked as a first use
        grid_out = state.outcome(kw[var])
        if grid_out in state.direct:
            return m(**kw)
    rkey = _replay_key(m, kw, var) if replay is not None else None
    if rkey is not None and state is True:
        hit = replay.get(rkey)
        if hit is not None:
            out = _replayed(hit, np.asarray(kw[var], dtype=float))
            if out is not None:
                return out
    try:
        out, tr = _trace_call(m, kw, var)
    except _Untraceable:
        tracing[key] = False
        return m(**kw)
    if not any(isinstance(v, _Sym) for v in out.values()):
        return out
    if state is None or grid_out is not None:
        # first use (or first use with this grid outcome): the call returns the direct result, and the trace is
        # checked against direct calls on a subsample spanning the sweep (on
        # the whole call when the subsample's trace is not the same code)
        ref = m(**kw)
        syms = [v for v in out.values() if isinstance(v, _Sym)]
        # a short code without the exponential recurrence saves only numpy's
        # temporaries, less than the trace costs: such models are called directly
        worth = _code_length(syms) >= _TRACE_MIN_CODE or any(c[0] == 100 for c in _prune(tr.code, tr.consts, [v.reg for v in syms])[0])
        ok = False
        if worth:
            ck = (out, tr, kw, ref)
            sub = _check_sample(kw, var)
            if sub is not None:
                try:
                    o2, t2 = _trace_call(m, sub, var)
                    if _same_program(tr, out, t2, o2):
                        ck = (o2, t2, sub, m(**sub))
                except _Untraceable:
                    pass
            o_, t_, k_, r_ = ck
            ok = _check_trace(o_, t_, r_, sens=_sensitivity(m, k_, var, r_))
        if not worth and any(not g for _, g in tr.grids):
            # not worth it because a grid test failed (no recurrence on this
            # sweep): call directly for this grid outcome only, so that a
            # later call on a uniform grid is checked again
            if grid_out is None:
                state = _GridDecisions(tuple(dict.fromkeys(b for b, _ in tr.grids)))
                grid_out = state.outcome(kw[var])
            state.direct.add(grid_out)
            tracing[key] = state
        else:
            tracing[key] = ok
        if ok and rkey is not None:
            if len(replay) > 256:
                replay.clear()
            replay[rkey] = (out, tr)
        return ref
    if rkey is not None:
        if len(replay) > 256:
            replay.clear()
        replay[rkey] = (out, tr)
    return out


def _replay_key(m, kw: dict, var: str):
    """Key of a call for trace replay: the model, the variable's shape and
    the other arguments by value when all are plain numbers, strings or
    tuples of them; None otherwise (arrays are not compared)."""
    parts = []
    for a, v in kw.items():
        if a == var:
            continue
        if isinstance(v, np.ndarray) and v.ndim:  # arrays are not compared (and not hashed to find out)
            return None
        p, ok = _frozen(v, {})
        if not ok or p[0] in ("array", "id"):
            return None
        parts.append((a, p))
    return (id(m), var, np.shape(kw[var]), tuple(sorted(parts)))


def _replayed(hit, x):
    """The SDict of a stored trace with the traced variable ``x``, or None
    when ``x`` changes a grid test of the trace (a different code)."""
    out, tr = hit
    grids = {}
    for base, ok in tr.grids:
        if base not in grids:
            grids[base] = _uniform_grid(x, base)
        if (grids[base][0] is not None) != ok:
            return None
    new = _Trace()
    new.code, new.consts, new.n, new.var, new.recur, new.derived, new.grids = tr.code, tr.consts, tr.n, x, tr.recur, tr.derived, tr.grids
    for i, p_ in enumerate(tr.params):
        d = tr.derived.get(i)
        if d is None:
            new.params.append(p_)
        elif d[0] == "var":
            new.params.append(x)
        else:
            new.params.append(grids[d[1]][0] if d[0] == "t" else grids[d[1]][1])
    return {k: (_Sym(new, v.reg, v.isreal, v.shape) if isinstance(v, _Sym) else v) for k, v in out.items()}


#: A traced SDict several instances share is evaluated once into arrays when
#: its code, times the extra instances, exceeds this many instructions per
#: point; otherwise inside the solve, once per instance
_SHARED_CODE_MAX = 32


def _code_length(syms) -> int:
    """Instructions per point of the codes of traced entries (after
    dead-code elimination; constants not counted, transcendental functions
    counted as 16)."""
    n = 0
    for tr in {id(v.tr): v.tr for v in syms}.values():
        code = _prune(tr.code, tr.consts, [v.reg for v in syms if v.tr is tr])[0]
        n += sum(_COST.get(c[0], 1) for c in code if c[0] != 1)
    return n


#: relative cost per point of the functions that are not a few arithmetic operations
_COST = {_FUN[f]: 16 for f in ("exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arctan", "ellipk", "pow")}


def _batch_shape(sdicts) -> tuple:
    shapes = [np.shape(v) for sd in sdicts for v in sd.values()]
    try:
        return np.broadcast_shapes(*shapes) if shapes else ()
    except ValueError:
        return ()


_PRUNED = {}  # (id of a code, registers) -> (code, result): replayed traces share their code


def _prune(code, consts, regs):
    """Dead-code elimination: the instructions the registers ``regs`` depend
    on, with registers renumbered densely. Returns ``(code, consts,
    new numbers of regs, register count)``."""
    key = (id(code), id(consts), len(code), tuple(regs))
    hit = _PRUNED.get(key)
    if hit is not None and hit[0] is code and hit[1] is consts:
        return hit[2]
    out = _prune_code(code, consts, regs)
    if len(_PRUNED) > 1024:
        _PRUNED.clear()
    _PRUNED[key] = (code, consts, out)
    return out


def _prune_code(code, consts, regs):
    live = set(regs)
    keep = []
    for k in range(len(code) - 1, -1, -1):
        kind, dst, a, b, c, real, d, e, f = code[k]
        if dst not in live:
            continue
        keep.append(k)
        if kind == 0 or kind == 1:
            continue
        if kind == 100:
            live.update((a, b, c, d, e))
        else:
            live.update((a, b, c))
    keep.reverse()
    new = {}
    out_code, out_consts = [], []
    for k in keep:
        kind, dst, a, b, c, real, d, e, f = code[k]
        new[dst] = len(new)
        m = lambda r: new.get(r, 0)  # noqa: E731  (operands an instruction does not use map to 0)
        if kind == 0:
            row = (0, new[dst], a, 0, 0, real, 0, 0, 0)
        elif kind == 1:
            row = (1, new[dst], 0, 0, 0, real, 0, 0, 0)
        elif kind == 100:
            row = (100, new[dst], m(a), m(b), m(c), real, m(d), m(e), f)
        else:
            row = (kind, new[dst], m(a), m(b), m(c), real, 0, 0, 0)
        out_code.append(row)
        out_consts.append(consts[k])
    return out_code, out_consts, [new[r] for r in regs], max(1, len(new))


def _solve_value(v):
    """An argument of the compiled solve: a complex128 array, or a float64
    array as it is (read as complex numbers with a zero imaginary part, the
    same numbers as converting it, without the copy)."""
    if type(v) is np.ndarray:
        dt = v.dtype
        if dt.isnative and (dt.num == _COMPLEX128 or (dt.num == _FLOAT64 and _REAL_VALUES)):
            return v
    return np.asarray(v, dtype=complex)


_COMPLEX128, _FLOAT64 = np.dtype(np.complex128).num, np.dtype(np.float64).num
_SCALAR_TYPES = frozenset((float, complex, int))  # Python numbers: shape ()


#: Hand float64 arrays to the compiled solve as they are (False: convert
#: them to complex first, as before; for measurements)
_REAL_VALUES = True


def _eval_trace_native(tr, regs, shape, threads=None):
    """Evaluate the registers ``regs`` of a trace over ``shape`` in compiled
    code (one block, no joins); returns one array per register."""
    run = getattr(_core(), "circuit_network_run", None)
    if run is None:
        out = _eval_code(tr.code, tr.consts, tr.params, shape)
        return [np.asarray(out[r]) for r in regs]
    n = max(1, int(np.ceil(np.sqrt(len(regs)))))
    values, ids = [], {}
    pmap = []
    for i, p_ in enumerate(tr.params):
        if id(p_) not in ids:
            arr = np.broadcast_to(_solve_value(p_), shape)
            if any(st < 0 for st in arr.strides):
                arr = np.ascontiguousarray(arr)
            ids[id(p_)] = len(values)
            values.append(arr)
        pmap.append(ids[id(p_)])
    if not values:
        values.append(np.zeros(shape, dtype=complex))
    pcode, pconsts, regs, nregs = _prune(tr.code, tr.consts, regs)
    code = np.array(pcode, dtype=np.int64).reshape(-1, 9)
    pk = code[:, 0] == 0
    if pk.any():
        code[pk, 2] = np.asarray(pmap, dtype=np.int64)[code[pk, 2]]
    consts = np.array(pconsts, dtype=float).reshape(-1, 2)
    outs = np.array([(k // n, k % n, r) for k, r in enumerate(regs)], dtype=np.int64).reshape(-1, 3)
    slots = np.array([(k, 0, k // n, k % n) for k in range(len(regs))], dtype=np.int64).reshape(-1, 4)
    R = run([n], 1, np.zeros((0, 6), np.int64), np.zeros((0, 4), np.int64), slots, len(regs), values, list(shape), threads, None, [(0, code, consts, outs, nregs)])
    return [R[k].reshape(shape) if shape else R[k, 0] for k in range(len(regs))]


def _concrete(sd: dict, threads=None) -> dict:
    """An SDict with its traced entries evaluated (compiled, over their own
    shape)."""
    syms = [(k, v) for k, v in sd.items() if isinstance(v, _Sym)]
    if not syms:
        return sd
    out = dict(sd)
    by_tr = {}
    for k, v in syms:
        by_tr.setdefault(id(v.tr), (v.tr, []))[1].append((k, v))
    for tr, items in by_tr.values():
        shape = np.broadcast_shapes(*[v.shape for _, v in items])
        regs = list(dict.fromkeys(v.reg for _, v in items))  # entries in one register (S12 = S21) share one array
        vals = dict(zip(regs, _eval_trace_native(tr, regs, shape, threads)))
        for k, v in items:
            a = vals[v.reg]
            out[k] = np.broadcast_to(a, v.shape) if v.shape != np.shape(a) else a
    return out


def _materialize(sdicts: dict) -> dict:
    """The SDicts with traced entries evaluated by the reference interpreter
    (for the matrix solves)."""
    out = {}
    for inst, sd in sdicts.items():
        if not any(isinstance(v, _Sym) for v in sd.values()):
            out[inst] = sd
            continue
        cache = {}
        d = {}
        for k, v in sd.items():
            if isinstance(v, _Sym):
                if id(v.tr) not in cache:
                    shp = np.broadcast_shapes(*[s.shape for s in sd.values() if isinstance(s, _Sym) and s.tr is v.tr])
                    cache[id(v.tr)] = _eval_code(v.tr.code, v.tr.consts, v.tr.params, shp)
                d[k] = np.broadcast_to(cache[id(v.tr)][v.reg], v.shape)
            else:
                d[k] = v
        out[inst] = d
    return out


class _Plan:
    """Everything about a network solve that depends only on the port lists
    of the instances, the connections and the external ports: the global
    port numbering, the external names, the dense solve's index arrays and
    connection matrix, and the join program."""

    def __init__(self, plist: dict, pairs: list, ports: dict):
        names, index = [], {}
        for inst, ps in plist.items():
            for p in ps:
                index[(inst, p)] = len(names)
                names.append((inst, p))
        self.index, self.N = index, len(names)
        conn = []
        for (ia, pa), (ib, pb) in pairs:
            ma, mb = _modes_of(plist.get(ia, []), pa), _modes_of(plist.get(ib, []), pb)
            if not ma or not mb:
                raise KeyError(f"connection {_nm(ia)},{pa} - {_nm(ib)},{pb}: port not in the models' S-parameters")
            if len(ma) != len(mb):
                warnings.warn(f"connection {_nm(ia)},{pa} - {_nm(ib)},{pb}: {len(ma)} and {len(mb)} modes; connecting the first {min(len(ma), len(mb))}", stacklevel=4)
            for x, y in zip(ma, mb):
                conn.append((index[(ia, x)], index[(ib, y)]))
        ext_names, ext_idx = [], []
        for pname, (inst, p) in ports.items():
            ms = _modes_of(plist.get(inst, []), p)
            if not ms:
                raise KeyError(f"external port {pname} -> {_nm(inst)},{p}: port not in the model's S-parameters")
            for m in ms:
                ext_names.append(pname + (m[len(p):] if "@" in m else ""))
                ext_idx.append(index[(inst, m)])
        self.ext_names = ext_names
        internal = sorted({i for c in conn for i in c})
        self.E = np.array(ext_idx, dtype=int)
        self.I = np.array(internal, dtype=int)
        pos = {g: k for k, g in enumerate(internal)}
        nI = len(internal)
        self.Cm = np.zeros((nI, nI))
        for a, b in conn:
            self.Cm[pos[a], pos[b]] = self.Cm[pos[b], pos[a]] = 1.0
        self.inst_of = {inst: k for k, inst in enumerate(plist)}
        self._joins(plist, conn, ext_idx)

    def _joins(self, plist, conn, ext_idx):
        """The join program (:mod:`lightsim_core::circuit`): blocks are the
        instances' used ports (ports neither connected nor external are
        matched and dropped); connections are taken greedily, the one that
        leaves the smallest block first."""
        import heapq

        self.joins_ok = False
        ends = [i for c in conn for i in c]
        used = set(ends) | set(ext_idx)
        # a port joined twice, joined to itself, or both joined and external:
        # only the matrix solve defines the result
        if len(set(ends)) != len(ends) or set(ends) & set(ext_idx):
            return
        blocks, local, where = [], {}, {}
        self.fill_map = {}  # instance -> (block, {port name: local index})
        for inst, ps in plist.items():
            mine = [p for p in ps if self.index[(inst, p)] in used]
            if not mine:
                continue
            b = len(blocks)
            g = [self.index[(inst, p)] for p in mine]
            blocks.append(g)
            self.fill_map[inst] = (b, {p: k for k, p in enumerate(mine)})
            for k, x in enumerate(g):
                local[x] = k
                where[x] = b
        sizes = [len(g) for g in blocks]
        n_init = len(blocks)
        ports_of = dict(enumerate(blocks))

        def after(k):
            x, y = conn[k]
            bx, by = where[x], where[y]
            return len(ports_of[bx]) + (len(ports_of[by]) if bx != by else 0) - 2

        heap = [(after(k), k) for k in range(len(conn))]
        heapq.heapify(heap)
        done = [False] * len(conn)
        ops = []
        while heap:
            key, k = heapq.heappop(heap)
            if done[k]:
                continue
            cur = after(k)
            if cur > key:
                heapq.heappush(heap, (cur, k))
                continue
            done[k] = True
            x, y = conn[k]
            bx, by = where[x], where[y]
            new = len(sizes)
            if bx != by:
                px, py = ports_of.pop(bx), ports_of.pop(by)
                ops.append((0, bx, px.index(x), by, py.index(y), new))
                newp = [p for p in px if p != x] + [p for p in py if p != y]
            else:
                px = ports_of.pop(bx)
                ops.append((1, bx, px.index(x), 0, px.index(y), new))
                newp = [p for p in px if p != x and p != y]
            sizes.append(len(newp))
            ports_of[new] = newp
            for p in newp:
                where[p] = new
        self.sizes, self.n_init = sizes, n_init
        self.ops = np.array(ops, dtype=np.int64).reshape(-1, 6)
        nE = len(ext_idx)
        outs = []
        for i, gi in enumerate(ext_idx):
            for j, gj in enumerate(ext_idx):
                b = where[gi]
                if where[gj] == b:
                    pb = ports_of[b]
                    outs.append((i * nE + j, b, pb.index(gj), pb.index(gi)))
        self.outputs = np.array(outs, dtype=np.int64).reshape(-1, 4)
        self.n_out = nE * nE
        # operation counts per batch point: the joins against a dense LU of
        # the internal ports and its solves (complex multiply-adds)
        self.cost_joins = sum(n * n + 4 for n in sizes[n_init:]) + sum(n * n for n in sizes[:n_init])
        nI = len(self.I)
        self.cost_dense = nI**3 / 3 + nI * nI * nE + nE * nE * nI
        self.joins_ok = True

    def fill_order(self, keys):
        """For the S-parameter keys ``keys`` (per instance, in the order the
        SDicts hold them): which entries enter the join program, and their
        place ``(block, out, in)``, so that a call only looks up its values."""
        self.fill_mask, rows, take, pos = [], [], [], 0
        if not self.joins_ok:
            return
        for inst, ks in keys:
            fm = self.fill_map.get(inst)
            if fm is None:
                self.fill_mask.append(())
                pos += len(ks)
                continue
            b, loc = fm
            mask = []
            for j, (a, o) in enumerate(ks):
                ia, io = loc.get(a), loc.get(o)
                use = ia is not None and io is not None
                mask.append(use)
                if use:
                    rows.append((b, io, ia))
                    take.append(pos + j)
            self.fill_mask.append(tuple(mask))
            pos += len(ks)
        self.fill_static = np.array(rows, dtype=np.int64).reshape(-1, 3)
        # the entries of the program in the list of all values of a call
        # (every SDict's values in order)
        self.fill_take = operator.itemgetter(*take) if len(take) > 1 else (lambda vals: tuple(vals[i] for i in take))


def _nm(inst) -> str:
    """An instance name for messages (a path of an inlined sub-circuit as a.b)."""
    return ".".join(map(str, inst)) if isinstance(inst, tuple) else str(inst)


def _plan(sdicts: dict, pairs: list, ports: dict, plans: dict, fixed: bool = False):
    """The cached plan of a network. ``fixed``: ``pairs`` and ``ports`` are
    those of every call with this ``plans`` dict (a circuit function's), so
    the key is the instances' S-parameter keys alone."""
    try:
        keys = tuple(zip(sdicts, map(tuple, sdicts.values())))
        key = keys if fixed else (keys, tuple(pairs), tuple(ports.items()))
        plan = plans.get(key)
    except TypeError:  # pragma: no cover  (unhashable names)
        return _Plan({inst: _ports_of(sd) for inst, sd in sdicts.items()}, pairs, ports)
    if plan is None:
        if len(plans) > 256:
            plans.clear()
        plan = plans[key] = _Plan({inst: _ports_of(sd) for inst, sd in sdicts.items()}, pairs, ports)
        plan.fill_order(keys)
    return plan


def _solve(sdicts: dict, pairs: list, ports: dict, backend: str, threads, plans: dict, fixed: bool = False) -> dict:
    if backend not in ("auto", "joins", "dense", "sparse"):
        raise ValueError(f"backend must be 'auto', 'joins', 'dense' or 'sparse', not {backend!r}")
    plan = _plan(sdicts, pairs, ports, plans, fixed)
    # the batch shape: the broadcast of the values' distinct shapes (most
    # values of a call share one or two), and whether any value is traced
    vals = [v for sd in sdicts.values() for v in sd.values()]
    others = [v for v in vals if type(v) is not np.ndarray]
    shapes = {v.shape for v in vals if type(v) is np.ndarray}
    shapes.update([() if type(v) in _SCALAR_TYPES else np.shape(v) for v in others])
    traced = any(isinstance(v, _Sym) for v in others)
    shape = np.broadcast_shapes(*shapes) if shapes else ()
    run = getattr(_core(), "circuit_network_run", None)
    use_joins = run is not None and plan.joins_ok and (backend == "joins" or (backend == "auto" and (traced or plan.cost_joins <= plan.cost_dense)))
    if backend == "joins" and not plan.joins_ok:
        raise ValueError("backend='joins' cannot solve this network (a port is connected twice, to itself, or connected and external)")
    if use_joins:
        if not hasattr(plan, "fill_take"):  # a plan built outside _plan's cache
            plan.fill_order(tuple(zip(sdicts, map(tuple, sdicts.values()))))
        R = _solve_joins(run, plan, sdicts, shape, threads) if traced else _solve_joins_plain(run, plan, vals, shape, threads)
        out = {}
        nE = len(plan.ext_names)
        for i, a in enumerate(plan.ext_names):
            for j, b in enumerate(plan.ext_names):
                v = R[i * nE + j]
                out[(a, b)] = v.reshape(shape) if shape else v[0]
        return out
    sdicts = _materialize(sdicts)
    nw = int(np.prod(shape)) if shape else 1
    S = np.zeros((nw, plan.N, plan.N), dtype=complex)
    index = plan.index
    for inst, sd in sdicts.items():
        for (a, b), v in sd.items():
            S[:, index[(inst, b)], index[(inst, a)]] = np.broadcast_to(np.asarray(v, dtype=complex), shape).reshape(nw)
    Sx = solve_matrix(S, plan.E, plan.I, plan.Cm, "auto" if backend == "joins" else backend)[0]
    out = {}
    for i, a in enumerate(plan.ext_names):
        for j, b in enumerate(plan.ext_names):
            v = Sx[:, j, i]
            out[(a, b)] = v.reshape(shape) if shape else v[0]
    return out


_POOLS: dict = {}
# a forked child has none of the parent's pool threads: it starts its own
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOLS.clear)
    os.register_at_fork(after_in_child=_DIGEST_POOL.clear)

#: Smallest batch (points per call) for which the models are evaluated on
#: the thread pool. Below it the interpreter lock and the hand-off cost more
#: than the threads give: on a GB10 (8 threads) a 42-instance circuit took
#: 11.3 ms serial against 14.5 ms pooled at 5,001 points, 28 against 24 ms at
#: 20,004 and 47 against 39 ms at 40,008. With ``stack_instances`` the
#: stacked groups are split over the threads above the same size. The network
#: solve is parallel in every case.
_MODEL_POOL_MIN_POINTS = 32768


def _at_least(work, todo, n: int) -> bool:
    """Whether a call has at least ``n`` batch points, with the exact
    broadcast only when the array sizes leave it open."""
    sizes = {v.size for k in todo for v in work[k][1].values() if isinstance(v, np.ndarray)}
    if not sizes or max(sizes) >= n:
        return bool(sizes)
    if math.prod(sizes) < n:  # the broadcast size is at most the product of the distinct sizes
        return False
    return _points(work, todo) >= n


def _points(work, todo) -> int:
    """Batch points of a call: the broadcast size of the array arguments."""
    shapes = dict.fromkeys(v.shape for k in todo for v in work[k][1].values() if isinstance(v, np.ndarray))
    try:
        return int(np.prod(np.broadcast_shapes(*shapes))) if shapes else 1
    except ValueError:
        return max((int(np.prod(s)) for s in shapes), default=1)


def _pool(n: int):
    from concurrent.futures import ThreadPoolExecutor

    p = _POOLS.get(n)
    if p is None:
        p = _POOLS[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="lightsim-circuit")
    return p


def _core():
    from . import _core as core

    return core


def _solve_joins(run, plan: _Plan, sdicts: dict, shape: tuple, threads):
    if not hasattr(plan, "fill_static"):  # a plan built outside _plan
        plan.fill_order(tuple((i, tuple(sd)) for i, sd in sdicts.items()))
    values, ids, ks = [], {}, []
    models = {}  # (block, id(trace)) -> (trace, [(row of fill_static, register)])

    def value(v):
        k = ids.get(id(v))
        if k is None:
            arr = _solve_value(v)
            if arr.shape != shape:
                arr = np.broadcast_to(arr, shape)
            if any(st < 0 for st in arr.strides) or not arr.dtype.isnative:
                arr = np.ascontiguousarray(arr)
            k = ids[id(v)] = len(values)
            values.append(arr)
        return k

    row = 0
    for sd, mask in zip(sdicts.values(), plan.fill_mask):
        if not mask:
            continue
        for v, use in zip(sd.values(), mask):
            if not use:
                continue
            if isinstance(v, _Sym):
                b = int(plan.fill_static[row, 0])
                models.setdefault((b, id(v.tr)), (v.tr, []))[1].append((row, v.reg))
                ks.append(-1)
            else:
                ks.append(value(v))
            row += 1
    fill = np.empty((len(ks), 4), dtype=np.int64)
    fill[:, :3] = plan.fill_static
    fill[:, 3] = ks
    model_args = None
    if models:
        fill = fill[fill[:, 3] >= 0]
        model_args = []
        for (b, _), (tr, outs) in models.items():
            # the arrays of a code are kept between calls (replayed traces share
            # their code); only the value numbers of the parameters change
            key = (id(tr.code), id(tr.consts), b, tuple(outs))
            hit = _BLOCKS.get(key)
            if hit is None or hit[0] is not tr.code:
                pcode, pconsts, regs, nregs = _prune(tr.code, tr.consts, [reg for _, reg in outs])
                code = np.array(pcode, dtype=np.int64).reshape(-1, 9)
                pk = np.nonzero(code[:, 0] == 0)[0]
                consts = np.array(pconsts, dtype=float).reshape(-1, 2)
                o = np.array([(plan.fill_static[r, 1], plan.fill_static[r, 2], reg) for (r, _), reg in zip(outs, regs)], dtype=np.int64).reshape(-1, 3)
                hit = (tr.code, code, pk, code[pk, 2].copy(), consts, o, nregs)
                if len(_BLOCKS) > 1024:
                    _BLOCKS.clear()
                _BLOCKS[key] = hit
            _, code, pk, pidx, consts, o, nregs = hit
            code = code.copy()
            code[pk, 2] = [value(tr.params[int(i)]) for i in pidx]
            model_args.append((b, code, consts, o, nregs))
        if not values:  # the solve needs one value array for its shape
            value(np.zeros(shape))
    if model_args is not None:
        return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, values, list(shape), threads, None, model_args)
    nb = int(np.prod(shape)) if shape else 1
    if len(values) > 1 and len(values) * nb <= _PACK_MAX:
        # small batches: one array to hand over instead of one per value
        packed = np.array(values, dtype=complex).reshape(len(values), nb)
        return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, [], list(shape), threads, packed)
    return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, values, list(shape), threads)


def _solve_joins_plain(run, plan: _Plan, vals: list, shape: tuple, threads):
    """:func:`_solve_joins` for a call without traced values (``vals``: every
    SDict's values in order, as the plan's keys): the distinct values in fill
    order, then one packed array for small batches (each value written into
    its row: the same numbers as converting it to complex and broadcasting
    it) or the arrays themselves."""
    used = plan.fill_take(vals)
    # the program made once per plan (parsed, checked, its layout and
    # schedule computed), when the compiled module has it; it reads the
    # values (arrays in place, scalars) itself unless one is of another kind
    prog = plan.__dict__.get("program")
    if prog is None:
        cls = getattr(_core(), "CircuitProgram", None)
        prog = plan.program = cls(plan.sizes, plan.n_init, plan.ops, plan.fill_static, plan.outputs, plan.n_out) if cls is not None else False
    if prog:
        R = prog.run_objects(used, list(shape), threads)
        if R is not None:
            return R
    ids = {}
    ks = [ids.setdefault(id(v), len(ids)) for v in used]
    raw = list({id(v): v for v in used}.values())
    if prog:
        ix = np.array(ks, dtype=np.int64)

        def go(values, packed=None):
            return prog.run(ix, values, list(shape), threads, packed)
    else:
        fill = np.empty((len(ks), 4), dtype=np.int64)
        fill[:, :3] = plan.fill_static
        fill[:, 3] = ks

        def go(values, packed=None):
            return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, values, list(shape), threads, packed)
    nb = math.prod(shape)
    if len(raw) > 1 and len(raw) * nb <= _PACK_MAX:
        packed = np.empty((len(raw),) + tuple(shape), dtype=complex)
        for k, v in enumerate(raw):
            packed[k] = v
        return go([], packed.reshape(len(raw), nb))
    values = []
    for v in raw:
        arr = _solve_value(v)
        if arr.shape != shape:
            arr = np.broadcast_to(arr, shape)
        if any(st < 0 for st in arr.strides) or not arr.dtype.isnative:
            arr = np.ascontiguousarray(arr)
        values.append(arr)
    return go(values)


_BLOCKS = {}  # model blocks of the solve: (code, block, outputs) -> arrays, see _solve_joins


#: Largest number of complex values (values x batch points) copied into one
#: array for the network solve: handing an array to the compiled solve costs
#: about 1 us, the copy about 1 ns per value, so packing pays for small
#: batches; large ones are read in place through their strides.
_PACK_MAX = 1 << 16


def simulate(component, models: dict | None = None, wl=1.55, recursive: bool = True, **overrides) -> dict:
    """S-parameters of a gdsfactory component from its netlist:
    ``circuit(component.get_netlist(recursive=...), models)(wl=wl,
    **overrides)``. ``wl`` in µm."""
    nl = component.get_netlist(recursive=True) if recursive else component.get_netlist()
    fn, _ = circuit(nl, models)
    return fn(wl=wl, **overrides)
