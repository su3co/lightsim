"""Objective helpers on the S-matrix ``S[m, n, k]``.

Each returns ``(value, g)`` with ``g = dL/dRe(S) + i dL/dIm(S)`` (the
convention of :mod:`lightsim.invdes.smatrix`), ready for
``SMatrixSolution.vjp``. All are written to be *maximised*.
"""

from __future__ import annotations

import numpy as np

__all__ = ["matrix_fidelity", "matrix_metrics", "band_transmission", "mean_coupling", "softmin"]


def matrix_fidelity(S, target, weights=None):
    """Matrix overlap with a target T[m, n], averaged over frequency:

        L = mean_k |<T, S_k>|^2 / ||T||^4,     <T, S> = sum_mn conj(T_mn) S_mn.

    L = 1 exactly when ``S_k = exp(i phi_k) T`` (one free global phase per
    frequency): it is the product of the normalised fidelity
    ``|<T,S>|^2 / (||T||^2 ||S||^2)`` and the throughput ``||S||^2 / ||T||^2``,
    so shape errors and loss are both penalised."""
    S = np.asarray(S, dtype=complex)
    T = np.asarray(target, dtype=complex)
    K = S.shape[2]
    w = np.full(K, 1.0 / K) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    t2 = float(np.sum(np.abs(T) ** 2))
    c = np.einsum("mn,mnk->k", np.conj(T), S)
    val = float(np.sum(w * np.abs(c) ** 2) / t2**2)
    g = 2.0 * (w * c)[None, None, :] * T[:, :, None] / t2**2
    return val, g


def matrix_metrics(S, target):
    """Per-frequency diagnostics of S against T: normalised ``fidelity``
    (shape only), ``throughput`` ``||S||^2/||T||^2``, their product
    ``overlap`` and the best-fit global ``phase``."""
    S = np.asarray(S, dtype=complex)
    T = np.asarray(target, dtype=complex)
    t2 = np.sum(np.abs(T) ** 2)
    s2 = np.sum(np.abs(S) ** 2, axis=(0, 1))
    c = np.einsum("mn,mnk->k", np.conj(T), S)
    return {"fidelity": np.abs(c) ** 2 / (t2 * s2), "throughput": s2 / t2, "overlap": np.abs(c) ** 2 / t2**2, "phase": np.angle(c)}


def softmin(x, alpha):
    """Smooth minimum -(1/a) log(mean(exp(-a x))) and its gradient."""
    x = np.asarray(x, dtype=float)
    z = -alpha * (x - x.min())
    e = np.exp(z)
    val = float(x.min() - np.log(np.mean(e)) / alpha)
    return val, e / np.sum(e)


def band_transmission(S, want, crosstalk=1.0, alpha=None):
    """In-band transmission to the right port with a crosstalk penalty.

    ``want`` is a boolean mask with the shape of S: True where power should
    arrive (pass bands), False where it should not (the other channels'
    bands); entries to ignore can be given as a masked array or as -1 in an
    integer array.

        L = T_pass - crosstalk * mean(|S|^2 over stop entries)

    with ``T_pass`` the mean of ``|S|^2`` over the pass entries, or their
    smooth minimum when ``alpha`` is given (equalises the channels)."""
    S = np.asarray(S, dtype=complex)
    w = np.asarray(want)
    pas = w == 1
    stop = w == 0
    p = np.abs(S) ** 2
    dp = np.zeros(S.shape)
    if alpha is None:
        val = float(np.mean(p[pas]))
        dp[pas] = 1.0 / np.count_nonzero(pas)
    else:
        val, gs = softmin(p[pas], alpha)
        dp[pas] = gs
    if crosstalk and np.any(stop):
        val -= crosstalk * float(np.mean(p[stop]))
        dp[stop] = -crosstalk / np.count_nonzero(stop)
    return val, 2.0 * S * dp


def mean_coupling(S, weights=None):
    """Coupled power averaged over sources, output channels and frequencies
    (``weights``: optional array broadcastable to S)."""
    S = np.asarray(S, dtype=complex)
    w = np.ones(S.shape) if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), S.shape)
    w = w / np.sum(w)
    return float(np.sum(w * np.abs(S) ** 2)), 2.0 * S * w
