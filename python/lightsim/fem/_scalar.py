"""Shared driver of the scalar problems (heat conduction, electrostatics)."""

from __future__ import annotations

import numpy as np


def k_tensor(values) -> np.ndarray:
    """Per-triangle (Kxx, Kyy, Kxy) from scalars, (kxx, kyy) or (kxx, kyy, kxy)."""
    out = np.zeros((len(values), 3))
    for i, v in enumerate(values):
        v = np.atleast_1d(np.asarray(v, float))
        if v.size == 1:
            out[i] = (v[0], v[0], 0.0)
        elif v.size == 2:
            out[i] = (v[0], v[1], 0.0)
        elif v.size == 3:
            out[i] = v
        elif v.size == 4:  # 2x2 matrix
            out[i] = (v[0], v[3], 0.5 * (v[1] + v[2]))
        else:
            raise ValueError(f"a tensor coefficient needs 1, 2, 3 or 4 values, got {v.size}")
    return out


def solve(mesh, order, k_tri, c_tri=None, f_tri=None, dirichlet=(), cases=None, neumann=(), robin=(), solver="direct", reaction=False, matrices=False, load=None):
    """Solve -div(K grad u) (+ c u) = f.

    ``dirichlet``: list of (dofs, value). ``cases``: optional list of lists of
    values, one value per Dirichlet group per case (several right-hand sides
    with one factorisation). ``neumann``: list of (edges, q). ``robin``: list
    of (edges, h, u_inf). Returns the raw dict of ``_core.fem_scalar_solve``.
    """
    from .. import _core

    M = mesh.n_tris
    k = np.ascontiguousarray(k_tri, float).reshape(M, 3)
    c = np.ascontiguousarray(np.zeros(M) if c_tri is None else c_tri, float)
    f = np.ascontiguousarray(np.zeros(M) if f_tri is None else f_tri, float)
    dofs, groups = [], []
    for g, (d, _) in enumerate(dirichlet):
        d = np.asarray(d, np.int64)
        dofs.append(d)
        groups.append(np.full(len(d), g))
    dofs_all = np.concatenate(dofs) if dofs else np.zeros(0, np.int64)
    grp = np.concatenate(groups) if groups else np.zeros(0, np.int64)
    # later groups override earlier ones on shared DOFs
    uniq, last = {}, {}
    for i, d in enumerate(dofs_all):
        last[int(d)] = i
    sel = np.array(sorted(last.values()), np.int64)
    dofs_u = dofs_all[sel]
    grp_u = grp[sel]
    if cases is None:
        cases = [[v for _, v in dirichlet]]
    vals = [[float(case[g]) for g in grp_u] for case in cases]
    ne = [(np.asarray(e, np.int64), q) for e, q in neumann]
    nb = (np.concatenate([e for e, _ in ne]).tolist() if ne else [], np.concatenate([np.full(len(e), float(q)) for e, q in ne]).tolist() if ne else [])
    rb = [(np.asarray(e, np.int64), h, u) for e, h, u in robin]
    rbt = (
        np.concatenate([e for e, _, _ in rb]).tolist() if rb else [],
        np.concatenate([np.full(len(e), float(h)) for e, h, _ in rb]).tolist() if rb else [],
        np.concatenate([np.full(len(e), float(u)) for e, _, u in rb]).tolist() if rb else [],
    )
    return _core.fem_scalar_solve(
        *mesh._args(), int(order), k, c, f, dofs_u.tolist(), vals, nb, rbt, solver, bool(reaction), bool(matrices), None if load is None else np.ascontiguousarray(load, float)
    )


def csr(d):
    import scipy.sparse as sp

    return sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
