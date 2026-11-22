"""JAX wrapper: the S-matrix as a differentiable function of the density.

    import jax, lightsim.invdes.jax as lj
    f = lambda rho: loss(lj.s_matrix(invdes, rho, beta=8.0))
    value, grad = jax.value_and_grad(f)(rho)

``invdes`` is an :class:`lightsim.invdes.InverseDesign` (or any object with
``forward(rho, beta) -> (S, pullback)``). JAX cotangents of a complex output
are ``dL/dRe S - i dL/dIm S``, the conjugate of the lightsim convention; the
wrapper conjugates. The simulations run in Python outside the trace, so the
function supports ``jax.grad`` / ``jax.vjp`` but not ``jax.jit`` or ``vmap``.
jax is imported lazily, only when this module is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["s_matrix"]


def s_matrix(layer, rho, beta=8.0, eta=None):
    """Differentiable S-matrix ``[M, N, K]`` (complex) of the density ``rho``."""

    state = {}  # the pullback is a Python closure, not a JAX type: keep it out of the residuals

    @jax.custom_vjp
    def f(r):
        return f_fwd(r)[0]

    def f_fwd(r):
        rn = np.asarray(r, dtype=float)
        S, pullback = layer.forward(rn, beta, eta) if eta is not None else layer.forward(rn, beta)
        state["pullback"] = pullback
        return jnp.asarray(S), jnp.zeros_like(r)

    def f_bwd(res, ct):
        g = np.conj(np.asarray(ct))
        return (jnp.asarray(np.asarray(state["pullback"](g)), dtype=res.dtype),)

    f.defvjp(f_fwd, f_bwd)
    return f(rho)
