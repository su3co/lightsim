"""Scattering-matrix networks: the S-matrix of a circuit of blocks joined at
shared ports, and its exact vector-Jacobian product.

Every block ``b`` has ``P_b`` ports and an S-matrix ``S_b[p, q, k]``
(outgoing amplitude at port ``p`` per unit incoming amplitude at port ``q``,
frequency ``k``). A connection joins two ports that share their reference
plane and mode profile: what leaves one enters the other. The ports that are
not connected are the external ports of the network, in the order given.

With all block ports gathered into one block-diagonal matrix ``S``, the
internal ports ``I`` and the external ports ``E``, and the connection matrix
``C`` (a symmetric permutation on ``I``), the network S-matrix is

    T = S_EE + S_EI C (1 - S_II C)^-1 S_IE.

It is exact for the block S-matrices it is given: the approximation of a
network model lies only in what the block S-matrices leave out (coupling
through modes or paths that are not ports). Expanding ``(1 - S_II C)^-1`` as
a geometric series orders the result by the number of internal reflections;
:meth:`Network.solve` with ``reflections=False`` keeps only the zeroth term
for networks in which every block is passed in one direction (the reflection
blocks of the S-matrices are set to zero).

The gradient of a real loss ``L(T)`` is passed in the convention of
:mod:`lightsim.invdes.smatrix`, ``g = dL/dRe T + i dL/dIm T`` with
``dL = Re(sum conj(g) dT)``. Since ``dT = R dS B`` with
``R = [1, S_EI C (1 - S_II C)^-1]`` and ``B = [1; C (1 - S_II C)^-1 S_IE]``,
the gradient with respect to the block S-matrices is the block-diagonal part
of ``R^H g B^H``, in the same convention: it is what
:meth:`lightsim.invdes.SMatrixSolution.vjp` of each block expects.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Network", "NetworkSolution", "chain"]


class Network:
    """Topology of a scattering-matrix network.

    Parameters
    ----------
    ports : number of ports of every block, ``[P_0, P_1, ...]``.
    connections : pairs ``((b, p), (b2, p2))`` of joined ports.
    external : the external ports ``(b, p)`` in the order of the rows and
        columns of the network S-matrix. Every port is either connected once
        or external.
    sides : optional, per block, a sequence of ``P_b`` labels (for example
        ``"in"`` / ``"out"``); with ``reflections=False`` the S-matrix
        entries between two ports of the same label are dropped.
    """

    def __init__(self, ports, connections, external, sides=None):
        self.ports = [int(p) for p in ports]
        self.offset = np.concatenate([[0], np.cumsum(self.ports)]).astype(int)
        self.total = int(self.offset[-1])
        g = lambda bp: int(self.offset[bp[0]] + bp[1])  # noqa: E731
        for b, p in [x for c in connections for x in c] + list(external):
            if not (0 <= b < len(self.ports) and 0 <= p < self.ports[b]):
                raise ValueError(f"port {(b, p)} does not exist")
        self.ext = np.array([g(e) for e in external], dtype=int)
        partner = np.full(self.total, -1, dtype=int)
        for a, c in connections:
            ia, ic = g(a), g(c)
            if ia == ic or partner[ia] >= 0 or partner[ic] >= 0:
                raise ValueError(f"port {a} or {c} is connected twice")
            partner[ia], partner[ic] = ic, ia
        used = partner >= 0
        used[self.ext] = True
        if np.any(partner[self.ext] >= 0):
            raise ValueError("an external port is also connected")
        if not np.all(used) or len(set(self.ext.tolist())) != len(self.ext):
            free = [int(i) for i in np.nonzero(~used)[0]]
            raise ValueError(f"every port must be connected or external (free global ports: {free[:8]})")
        self.int = np.nonzero(partner >= 0)[0]
        pos = np.full(self.total, -1, dtype=int)
        pos[self.int] = np.arange(len(self.int))
        self.partner_local = pos[partner[self.int]]  # C[i, partner_local[i]] = 1 on the internal ports
        self.sides = None if sides is None else [list(s) for s in sides]

    @property
    def n_external(self):
        return len(self.ext)

    def _assemble(self, S_blocks, k, reflections):
        import scipy.sparse as sp

        rows, cols, vals = [], [], []
        for b, Sb in enumerate(S_blocks):
            P = self.ports[b]
            M = np.asarray(Sb[:, :, k], dtype=complex)
            if M.shape != (P, P):
                raise ValueError(f"block {b}: S has shape {M.shape[:2]}, expected {(P, P)}")
            if not reflections:
                if self.sides is None:
                    raise ValueError("reflections=False needs the port sides")
                lab = np.asarray(self.sides[b], dtype=object)
                M = np.where(lab[:, None] == lab[None, :], 0.0, M)
            r, c = np.nonzero(M)
            rows.append(r + self.offset[b])
            cols.append(c + self.offset[b])
            vals.append(M[r, c])
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.total, self.total))

    def solve(self, S_blocks, reflections=True, store=True):
        """Network S-matrix ``T[e, e2, k]`` of the block S-matrices
        ``S_blocks`` (list of ``[P_b, P_b, K]``). With ``store`` the
        factors needed by :meth:`NetworkSolution.vjp` are kept."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        K = int(np.shape(S_blocks[0])[2])
        nE, nI = len(self.ext), len(self.int)
        T = np.zeros((nE, nE, K), dtype=complex)
        Bs, Rs, mats = [], [], []
        Cperm = sp.csr_matrix((np.ones(nI), (np.arange(nI), self.partner_local)), shape=(nI, nI))
        for k in range(K):
            S = self._assemble(S_blocks, k, reflections)
            SEE = S[self.ext][:, self.ext].toarray()
            if nI == 0:
                T[:, :, k] = SEE
                Bs.append(None)
                Rs.append(None)
                mats.append(S)
                continue
            SEI = S[self.ext][:, self.int]
            SIE = S[self.int][:, self.ext].toarray()
            SII = S[self.int][:, self.int]
            A = (sp.identity(nI, dtype=complex, format="csc") - (SII @ Cperm).tocsc()).tocsc()
            lu = spl.splu(A)
            X = lu.solve(SIE)  # (1 - S_II C)^-1 S_IE
            CX = Cperm @ X
            T[:, :, k] = SEE + SEI @ CX
            if store:
                # R_I = S_EI C A^-1  <=>  A^T R_I^T = (S_EI C)^T
                RI = lu.solve(np.asarray((SEI @ Cperm).toarray()).T, trans="T").T
                Bs.append(CX)
                Rs.append(RI)
            mats.append(S)
        return NetworkSolution(T=T, _net=self, _B=Bs if store else None, _R=Rs if store else None, _reflections=reflections)


class NetworkSolution:
    """Result of :meth:`Network.solve`: ``T[e, e2, k]`` and :meth:`vjp`."""

    def __init__(self, T, _net, _B, _R, _reflections):
        self.T = T
        self._net = _net
        self._B = _B
        self._R = _R
        self._reflections = _reflections

    def vjp(self, g):
        """Gradients with respect to the block S-matrices (list of
        ``[P_b, P_b, K]``, convention of the module docstring) for the
        upstream gradient ``g`` with the shape of ``T``."""
        net = self._net
        if self._B is None:
            raise RuntimeError("solve(..., store=True) is needed for gradients")
        g = np.asarray(g, dtype=complex)
        if g.shape != self.T.shape:
            raise ValueError(f"upstream gradient has shape {g.shape}, T has {self.T.shape}")
        K = g.shape[2]
        nE = len(net.ext)
        out = [np.zeros((P, P, K), dtype=complex) for P in net.ports]
        # global port -> (block, local port)
        blk = np.repeat(np.arange(len(net.ports)), net.ports)
        loc = np.arange(net.total) - net.offset[blk]
        for k in range(K):
            # full R (nE x total) and B (total x nE)
            R = np.zeros((nE, net.total), dtype=complex)
            B = np.zeros((net.total, nE), dtype=complex)
            R[np.arange(nE), net.ext] = 1.0
            B[net.ext, np.arange(nE)] = 1.0
            if self._B[k] is not None:
                R[:, net.int] = self._R[k]
                B[net.int, :] = self._B[k]
            Rg = np.conj(R).T @ g[:, :, k]  # total x nE
            for b, P in enumerate(net.ports):
                idx = slice(net.offset[b], net.offset[b + 1])
                out[b][:, :, k] = Rg[idx] @ np.conj(B[idx]).T
            if not self._reflections and net.sides is not None:
                for b in range(len(net.ports)):
                    lab = np.asarray(net.sides[b], dtype=object)
                    out[b][lab[:, None] == lab[None, :], k] = 0.0
        del blk, loc
        return out


def chain(transmissions):
    """Feed-forward composition of a sequence of transmission matrices
    ``[T_1, T_2, ...]`` (each ``[n_out, n_in, K]``, the outputs of one
    feeding the inputs of the next): ``T = T_D ... T_2 T_1`` and the
    pullback of an upstream gradient to every factor.

    Returns ``(T, pullback)``; ``pullback(g)`` gives the list of gradients
    in the convention of the module docstring. This is
    :meth:`Network.solve` with ``reflections=False`` for a cascade, without
    the sparse machinery: ``O(D n^3)`` per frequency for D dense factors."""
    Ts = [np.asarray(t, dtype=complex) for t in transmissions]
    K = Ts[0].shape[2]
    fwd = [None] * (len(Ts) + 1)  # fwd[i] = T_i ... T_1 (fwd[0] = identity)
    n0 = Ts[0].shape[1]
    fwd[0] = np.repeat(np.eye(n0, dtype=complex)[:, :, None], K, axis=2)
    for i, t in enumerate(Ts):
        fwd[i + 1] = np.einsum("abk,bck->ack", t, fwd[i])
    T = fwd[-1]

    def pullback(g):
        g = np.asarray(g, dtype=complex)
        out = [None] * len(Ts)
        back = g  # dL/d(T_D ... T_{i+1} T_i ... ) w.r.t. the product after factor i
        for i in range(len(Ts) - 1, -1, -1):
            # T = A T_i F with A = T_D..T_{i+1}, F = fwd[i]; dT = A dT_i F
            out[i] = np.einsum("abk,cbk->ack", back, np.conj(fwd[i]))
            back = np.einsum("bak,bck->ack", np.conj(Ts[i]), back)
        return out

    return T, pullback
