"""PyTorch wrapper: the S-matrix as a differentiable layer in the density.

    import torch, lightsim.invdes.torch as lt
    S = lt.s_matrix(invdes, rho, beta=8.0)        # complex tensor [M, N, K]
    loss = some_real_function(S); loss.backward()  # rho.grad is filled

``invdes`` is an :class:`lightsim.invdes.InverseDesign` (or any object with
``forward(rho, beta) -> (S, pullback)``). For a real loss PyTorch hands the
backward pass ``dL/dRe S + i dL/dIm S`` for a complex tensor, which is the
lightsim convention, so the cotangent passes through unchanged. torch is
imported lazily, only when this module is.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["s_matrix", "SMatrixFunction"]


class SMatrixFunction(torch.autograd.Function):
    """The autograd function behind :func:`s_matrix`: the simulations run in
    ``forward`` and the adjoint simulations in ``backward``. Call
    :func:`s_matrix` instead of using it directly."""

    @staticmethod
    def forward(ctx, rho, layer, beta, eta):
        """Solve for the density ``rho`` and return the complex S-matrix
        ``[M, N, K]`` on the device of ``rho`` (complex128 for a float64
        density, complex64 otherwise)."""
        S, pullback = layer.forward(rho.detach().cpu().numpy().astype(float), beta, eta) if eta is not None else layer.forward(rho.detach().cpu().numpy().astype(float), beta)
        ctx.pullback = pullback
        ctx.rho_like = rho
        cd = torch.complex128 if rho.dtype == torch.float64 else torch.complex64
        return torch.as_tensor(np.asarray(S), dtype=cd, device=rho.device)

    @staticmethod
    def backward(ctx, grad_S):
        """Gradient with respect to the density from the gradient with
        respect to S (PyTorch's convention for complex tensors)."""
        g = grad_S.detach().resolve_conj().cpu().numpy()
        grad = np.asarray(ctx.pullback(g))
        return torch.as_tensor(grad, dtype=ctx.rho_like.dtype, device=ctx.rho_like.device), None, None, None


def s_matrix(layer, rho, beta=8.0, eta=None):
    """Differentiable S-matrix ``[M, N, K]`` (complex) of the density tensor ``rho``."""
    return SMatrixFunction.apply(rho, layer, beta, eta)
