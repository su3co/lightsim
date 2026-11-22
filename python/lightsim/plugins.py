"""User-defined material models (plug-ins) written in Python.

A plug-in medium is a plain background (``eps``, ``sigma``) plus a model
that turns the field history into a polarization. Register the model's
callables under a name, then give a medium ``plugin=use(name, params)``::

    import numpy as np
    import lightsim as ls
    from lightsim import plugins

    def after(e, e_prev, inv_eps, state, ctx):
        # instantaneous Kerr medium P / eps0 = chi3 E^3, one implicit step
        chi3 = ctx["params"]["chi3"]
        u = e - e_prev
        c3 = chi3 * inv_eps
        e[:] = e_prev + u / (1 + 3 * c3 * e_prev**2)

    plugins.register("my_kerr", after=after)
    kerr = ls.Medium("kerr", eps=1.0, plugin=plugins.use("my_kerr", {"chi3": 1e-20}))

Per step and per E component (x, y, z) of the medium's cells the engine
calls

* ``before(e, state, ctx)`` (optional) before the linear E update, with
  ``e`` = E^n of the cells (a copy; read it);
* ``after(e, e_prev, inv_eps, state, ctx)`` after the linear update: ``e``
  holds E_lin, the field the linear update gave (curl H term included), and
  must be overwritten in place with E^{n+1} (or the new values returned);
  ``e_prev`` = E^n, ``inv_eps`` = 1 / (eps (1 + f)) per cell (``f`` the
  background conduction term). A polarization change ``dP / eps0`` enters as
  ``E^{n+1} = E_lin - inv_eps * dP / eps0``.

``e``, ``e_prev``, ``inv_eps`` are 1-D NumPy arrays (float32, or float64 in
the double-precision build) over the medium's cells of that component;
``state`` is an array of shape ``(cells, state)`` that starts at zero and
persists between calls (one per medium, component and slab); ``ctx`` is a
dict with ``axis``, ``step``, ``dt`` and ``params``. The callables are
vectorised over the cells, so NumPy or numba code runs at array speed.

On the CPU engine the callables run in the engine's loop. On a GPU the
medium's cells are copied to the host and back every step (two round trips
per component and step), which is slow for large plug-in media (32 to 73
times the built-in pole for a 96^3 slab on an RTX PRO 6000; see
``docs/guide/materials.md``). Complex (Bloch) fields, time
reversal and the gradient engines refuse plug-in media. A plug-in that adds
energy without bound is stopped by the solver's divergence check; an
exception raised by a callable stops the run with the exception's text.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["register", "unregister", "use", "register_examples"]


def register(name: str, after: Callable, before: Callable | None = None, state: int = 0) -> str:
    """Register Python callables as the material plug-in ``name`` (replacing
    one of that name). ``state`` is the number of state values per cell.
    Returns ``name``."""
    from . import _core

    if not callable(after) or (before is not None and not callable(before)):
        raise TypeError("after (and before, if given) must be callable")
    _core.register_plugin(name, after, before, int(state))
    return name


def unregister(name: str) -> None:
    """Remove a registered plug-in."""
    from . import _core

    _core.unregister_plugin(name)


def register_examples() -> None:
    """Register the Rust example plug-ins of ``lightsim_core::plugin::examples``:
    ``"example_lorentz"`` (params ``delta_eps``, ``omega0``, ``gamma``),
    ``"example_kerr"`` (``chi3``) and ``"example_runaway"`` (``gain``)."""
    from . import _core

    _core.register_example_plugins()


def use(name: str, params: dict | None = None) -> dict:
    """The ``plugin`` entry of a :func:`lightsim.Medium`: the registered
    plug-in ``name`` with ``params`` (JSON-serialisable, handed to the
    callables as ``ctx["params"]``)."""
    return {"name": name, "params": dict(params or {})}
