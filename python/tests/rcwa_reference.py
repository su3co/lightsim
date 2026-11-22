"""Rigorous coupled-wave analysis of a lamellar (binary) grating: the
reference for the Bloch-periodic grating cases of the 2-D solver.

A layer of thickness ``depth`` with ridges of index ``n_ridge`` (fill factor
``fill`` of the period, centred on x = 0) in grooves of index ``n_groove``
separates the incidence medium ``n1`` from the substrate ``n2``. A plane
wave arrives at ``theta`` (radians, in the plane normal to the grooves).
Formulation of Moharam, Grann, Pommet and Gaylord, J. Opt. Soc. Am. A 12,
1068 (1995), with Li's Fourier factorisation for the polarisation with H
along the grooves. Dense solve of the boundary system: fine for the shallow
layers used here.

``efficiencies(...)`` returns ``(orders, R, T)``: the diffraction
efficiencies of the reflected and transmitted orders (fractions of the
incident power; zero for evanescent orders).

    python rcwa_reference.py     # prints the constants used by the Rust case
"""

import numpy as np


def efficiencies(wavelength, period, depth, fill, n1, n2, n_ridge, n_groove, theta, e_along_grooves, harmonics=41):
    m = np.arange(-(harmonics // 2), harmonics // 2 + 1)
    k0 = 2 * np.pi / wavelength
    kx = n1 * np.sin(theta) + m * wavelength / period  # in units of k0
    kz1 = np.conj(np.sqrt((n1**2 - kx**2).astype(complex)))  # exp(-i kz z) decays for Im < 0
    kz2 = np.conj(np.sqrt((n2**2 - kx**2).astype(complex)))

    def toeplitz(a, b):
        """Fourier coefficients of a ridge value ``a`` over a groove value ``b``."""
        h = m[:, None] - m[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = (a - b) * np.sin(np.pi * h * fill) / (np.pi * h)
        c[h == 0] = fill * a + (1 - fill) * b
        return c

    E = toeplitz(n_ridge**2, n_groove**2)
    Kx = np.diag(kx)
    n = len(m)
    eye = np.eye(n)
    if e_along_grooves:
        A = Kx @ Kx - E
        q2, W = np.linalg.eig(A)
        Q = np.sqrt(q2.astype(complex))
        V = W @ np.diag(Q)
        y1, y2 = np.diag(kz1), np.diag(kz2)
        inc_b = 1j * n1 * np.cos(theta)
    else:
        Einv = toeplitz(1 / n_ridge**2, 1 / n_groove**2)
        A = np.linalg.solve(Einv, Kx @ np.linalg.solve(E, Kx) - eye)
        q2, W = np.linalg.eig(A)
        Q = np.sqrt(q2.astype(complex))
        V = Einv @ W @ np.diag(Q)
        y1, y2 = np.diag(kz1 / n1**2), np.diag(kz2 / n2**2)
        inc_b = 1j * np.cos(theta) / n1
    Q = np.where(Q.real < 0, -Q, Q)
    V = (W @ np.diag(Q)) if e_along_grooves else (toeplitz(1 / n_ridge**2, 1 / n_groove**2) @ W @ np.diag(Q))
    X = np.diag(np.exp(-k0 * Q * depth))
    d0 = (m == 0).astype(complex)
    # unknowns: R (n), c+ (n), c- (n), T (n)
    Z = np.zeros((n, n), dtype=complex)
    M = np.block(
        [
            [-eye, W, W @ X, Z],
            [1j * y1, V, -V @ X, Z],
            [Z, W @ X, W, -eye],
            [Z, V @ X, -V, -1j * y2],
        ]
    )
    rhs = np.concatenate([d0, inc_b * d0, np.zeros(2 * n)])
    sol = np.linalg.solve(M, rhs)
    R, T = sol[:n], sol[3 * n :]
    if e_along_grooves:
        dr = np.abs(R) ** 2 * np.real(kz1 / (n1 * np.cos(theta)))
        dt = np.abs(T) ** 2 * np.real(kz2 / (n1 * np.cos(theta)))
    else:
        dr = np.abs(R) ** 2 * np.real(kz1 / (n1 * np.cos(theta)))
        dt = np.abs(T) ** 2 * np.real(kz2 / n2**2) / (np.cos(theta) / n1)
    return m, dr, dt


# the grating of `two_d_grating` (crates/lightsim-core/src/validation.rs) and of test_twod.py
CASE = dict(wavelength=1.0e-6, period=1.5e-6, depth=0.5e-6, fill=0.5, n1=1.0, n2=1.5, n_ridge=1.5, n_groove=1.0, theta=np.radians(15.0))

if __name__ == "__main__":
    for along, label in [(False, "TE set (E in the plane, H along the grooves)"), (True, "TM set (E along the grooves)")]:
        for nh in (21, 41, 81, 161):
            m, r, t = efficiencies(**CASE, e_along_grooves=along, harmonics=nh)
            keep = (m >= -2) & (m <= 2)
            print(label, "harmonics", nh, "sum", float(r.sum() + t.sum()))
            print("   m =", m[keep].tolist())
            print("   R =", np.round(r[keep], 5).tolist())
            print("   T =", np.round(t[keep], 5).tolist())
