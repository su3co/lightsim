"""Inverse-design module: S-matrix vector-Jacobian products versus finite
differences (real and imaginary parts), the parametrisation chain and its
transposes, the optimiser, and the torch / JAX wrappers.

Run on a GPU with LIGHTSIM_DEVICE=cuda:0.
"""

import copy
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import eim, invdes
from lightsim.adjoint import mode_power_gradient
from lightsim.invdes import objectives as obj
from lightsim.spec import C0, Box, Medium, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1.55 * um
F0 = C0 / LAM


# ----------------------------------------------------------------------------
# pure numpy pieces
# ----------------------------------------------------------------------------


def _fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g


@pytest.mark.parametrize("kind", ["slab", "partial", "eim", "eim_partial"])
def test_parametrisation_chain_transpose(kind):
    """filter -> projection -> level products -> eps: vjp equals finite differences."""
    rng = np.random.default_rng(3)
    mn, mx = [0.0, 0.0], [1.0 * um, 0.7 * um]
    shape = (11, 8)
    solid = np.zeros(shape, bool)
    solid[0, 3:5] = True
    void = np.zeros(shape, bool)
    void[-1, :2] = True
    kw = dict(shape=shape, filter_radius=0.22 * um, fixed_solid=solid, fixed_void=void, symmetry=("y",) if kind == "slab" else ())
    if kind == "slab":
        d = invdes.Design.slab(mn, mx, (0.0, 0.22 * um), 2.1, 12.1, **kw)
    elif kind == "partial":
        d = invdes.Design.partial_etch(mn, mx, (0.0, 0.07 * um, 0.22 * um), 2.1, 12.1, **kw)
    elif kind == "eim":
        d = invdes.Design.effective_index(mn, mx, 1.444, 2.85, **kw)
    else:
        d = invdes.Design.effective_index(mn, mx, 1.444, 2.85, n_partial=2.2, filter="gaussian", **kw)
    rho = rng.uniform(0.05, 0.95, (d.fields,) + shape)
    rho = rho[0] if d.fields == 1 else rho
    ws = [rng.standard_normal(shape + (1,)) for _ in d.layers]
    beta = 6.0
    f = lambda r: sum(float(np.sum(w * e)) for w, e in zip(ws, d.eps(r, beta)))  # noqa: E731
    g = d.vjp(rho, beta, ws)
    gfd = _fd_grad(f, rho)
    assert np.allclose(g, gfd, rtol=1e-5, atol=1e-7 * np.abs(gfd).max()), np.abs(g - gfd).max()
    # fixed pixels and bounds
    ph = d.projected(rho, beta)
    assert np.all(ph[:, solid] == 1.0) and np.all(ph[:, void] == 0.0)
    assert len(d.grids()) == len(d.layers)
    # eroded / dilated variants bracket the nominal design
    lo, hi = d.projected(rho, beta, 0.6).sum(), d.projected(rho, beta, 0.4).sum()
    assert lo < ph.sum() < hi
    assert 0.0 <= d.binarization(rho, beta) < d.binarization(rho, 200.0) <= 1.0


def test_objective_gradients():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 2, 4)) + 1j * rng.standard_normal((3, 2, 4))
    T = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
    want = rng.integers(-1, 2, S.shape)
    cases = [
        lambda s: obj.matrix_fidelity(s, T),
        lambda s: obj.band_transmission(s, want, crosstalk=0.7),
        lambda s: obj.band_transmission(s, want, crosstalk=0.7, alpha=3.0),
        lambda s: obj.mean_coupling(s),
    ]
    for f in cases:
        _, g = f(S)
        gr = _fd_grad(lambda x: f(x + 1j * S.imag)[0], S.real.copy())
        gi = _fd_grad(lambda x: f(S.real + 1j * x)[0], S.imag.copy())
        assert np.allclose(g, gr + 1j * gi, rtol=1e-5, atol=1e-7)
    # the overlap is 1 for S = exp(i phi) T and factorises into fidelity x throughput
    Sx = np.exp(0.7j) * T[:, :, None] * np.ones(4)
    assert abs(obj.matrix_fidelity(Sx, T)[0] - 1) < 1e-12
    mm = obj.matrix_metrics(0.5 * Sx, T)
    assert np.allclose(mm["fidelity"], 1) and np.allclose(mm["throughput"], 0.25) and np.allclose(mm["phase"], 0.7)


class _Mock:
    """Analytic stand-in for InverseDesign: S = (C rho)^2, complex."""

    def __init__(self, shape=(2, 2, 3), n=(4, 5), seed=0):
        rng = np.random.default_rng(seed)
        self.shape, self.n = shape, n
        m = int(np.prod(shape))
        self.C = rng.standard_normal((m, n[0] * n[1])) + 1j * rng.standard_normal((m, n[0] * n[1]))
        self.calls = 0

    def forward(self, rho, beta=1.0, eta=None):
        self.calls += 1
        z = self.C @ np.asarray(rho, dtype=float).ravel()
        S = (z * z * beta).reshape(self.shape)

        def pullback(g):
            return np.real(np.conj(np.asarray(g).ravel()) * 2 * z * beta @ self.C).reshape(self.n)

        return S, pullback

    def value_and_grad(self, rho, beta):
        S, pb = self.forward(rho, 1.0)
        target = np.arange(S.size).reshape(S.shape) / S.size
        val = -float(np.sum(np.abs(S - target) ** 2))
        return val, pb(-2 * (S - target)), {"wall": 0.0}


def _mock_loss_np(S):
    w = np.arange(S.size).reshape(S.shape) + 1.0
    return float(np.sum(w * np.abs(S) ** 2) + np.sum(np.real(S) * 0.3 - np.imag(S) * 0.7))


def test_optimizer_checkpoint_resume(tmp_path):
    mock = _Mock()
    rho0 = np.full(mock.n, 0.5)
    ck = str(tmp_path / "ck.npz")
    r1, h1 = invdes.optimize(mock, rho0, 12, beta=invdes.beta_schedule(1, 2, 5, 8), lr=0.05, checkpoint=ck, verbose=False)
    assert len(h1["fom"]) == 12 and h1["fom"][-1] > h1["fom"][0]
    assert list(h1["beta"][[0, 5, 10]]) == [1.0, 2.0, 4.0]
    # resume: nothing recomputed up to 12, continues to 20 exactly like an uninterrupted run
    calls = mock.calls
    r2, h2 = invdes.optimize(mock, rho0, 20, beta=invdes.beta_schedule(1, 2, 5, 8), lr=0.05, checkpoint=ck, verbose=False)
    assert mock.calls - calls == 8 and len(h2["fom"]) == 20
    r3, h3 = invdes.optimize(_Mock(), rho0, 20, beta=invdes.beta_schedule(1, 2, 5, 8), lr=0.05, verbose=False)
    assert np.allclose(r2, r3) and np.allclose(h2["fom"], h3["fom"])
    assert np.all((r2 >= 0) & (r2 <= 1))
    # without the Adam restart at beta steps the run differs but still improves
    r5, h5 = invdes.optimize(_Mock(), rho0, 20, beta=invdes.beta_schedule(1, 2, 5, 8), lr=0.05, verbose=False, reset_on_beta=False)
    assert not np.allclose(r5, r3) and h5["fom"][-1] > h5["fom"][0]
    # L-BFGS-B also improves
    _, h4 = invdes.optimize(_Mock(), rho0, 15, beta=4.0, method="lbfgs", verbose=False)
    assert max(h4["fom"]) > h4["fom"][0] and h4["best_fom"] == max(h4["fom"])


def test_robust_variants_gradient():
    """InverseDesign(robust=d): mean objective over nominal / eroded / dilated
    designs, gradient through the three projections (analytic stand-in solver)."""
    rng = np.random.default_rng(4)
    des = invdes.Design.effective_index([0, 0], [1 * um, 0.6 * um], 1.444, 2.85, shape=(9, 6), filter_radius=0.2 * um)
    w = rng.standard_normal((2, 1, 2) + des.shape) + 1j * rng.standard_normal((2, 1, 2) + des.shape)

    class Solver:
        def solve(self, eps, adjoint=True):
            e = np.asarray(eps[0]).reshape(des.shape)
            S = np.tensordot(w, e**2, axes=2)

            class Sol:
                pass

            sol = Sol()
            sol.S, sol.stats = S, {"wall": 0.0, "runs": [0, 0]}
            sol.vjp = lambda g: [np.real(np.tensordot(np.conj(g), w, axes=3)) * 2 * e]
            return sol

    inv = invdes.InverseDesign(Solver(), des, objective=obj.mean_coupling, robust=0.08)
    rho = rng.uniform(0.1, 0.9, des.shape)
    val, grad, info = inv.value_and_grad(rho, 7.0)
    assert info["simulations"] == 6 and len(info["S_variants"]) == 3
    one = invdes.InverseDesign(Solver(), des, objective=obj.mean_coupling)
    vals = [obj.mean_coupling(one.s_matrix(rho, 7.0, eta))[0] for eta in (0.5, 0.58, 0.42)]
    assert abs(val - np.mean(vals)) < 1e-12 * abs(val)
    gfd = _fd_grad(lambda r: inv.value_and_grad(r, 7.0)[0], rho.copy())
    assert np.allclose(grad, gfd, rtol=1e-5, atol=1e-8 * np.abs(gfd).max())


def test_torch_wrapper_convention():
    torch = pytest.importorskip("torch")
    import lightsim.invdes.torch as lt

    mock = _Mock()
    rho = torch.tensor(np.random.default_rng(1).uniform(0.2, 0.8, mock.n), dtype=torch.float64, requires_grad=True)
    S = lt.s_matrix(mock, rho, beta=1.3)
    w = torch.arange(S.numel(), dtype=torch.float64).reshape(S.shape) + 1.0
    loss = (w * S.abs() ** 2).sum() + (S.real * 0.3 - S.imag * 0.7).sum()
    loss.backward()
    gfd = _fd_grad(lambda r: _mock_loss_np(mock.forward(r, 1.3)[0]), rho.detach().numpy().copy())
    assert np.allclose(rho.grad.numpy(), gfd, rtol=1e-5)
    # the same loss written natively in torch gives the same gradient
    rho2 = rho.detach().clone().requires_grad_(True)
    z = torch.as_tensor(mock.C) @ rho2.to(torch.complex128).ravel()
    S2 = (z * z * 1.3).reshape(mock.shape)
    ((w * S2.abs() ** 2).sum() + (S2.real * 0.3 - S2.imag * 0.7).sum()).backward()
    assert np.allclose(rho.grad.numpy(), rho2.grad.numpy(), rtol=1e-9)


def test_jax_wrapper_convention():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import lightsim.invdes.jax as lj

    mock = _Mock()
    rho = jnp.asarray(np.random.default_rng(1).uniform(0.2, 0.8, mock.n))
    w = jnp.arange(int(np.prod(mock.shape)), dtype=jnp.float64).reshape(mock.shape) + 1.0

    def loss_of_S(S):
        return jnp.sum(w * jnp.abs(S) ** 2) + jnp.sum(jnp.real(S) * 0.3 - jnp.imag(S) * 0.7)

    val, g = jax.value_and_grad(lambda r: loss_of_S(lj.s_matrix(mock, r, beta=1.3)))(rho)
    gfd = _fd_grad(lambda r: _mock_loss_np(mock.forward(r, 1.3)[0]), np.array(rho))
    assert np.allclose(np.asarray(g), gfd, rtol=1e-5)
    # native JAX version of the same function
    C = jnp.asarray(mock.C)

    def native(r):
        z = C @ r.ravel().astype(jnp.complex128)
        return loss_of_S((z * z * 1.3).reshape(mock.shape))

    assert np.allclose(np.asarray(g), np.asarray(jax.grad(native)(rho)), rtol=1e-9)
    assert abs(float(val) - float(native(rho))) < 1e-9 * abs(float(val))


# ----------------------------------------------------------------------------
# 2.5-D: two inputs x two outputs x three frequencies, full chain
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def planar():
    n_si = eim.slab_neff(3.476, 220e-9, LAM, 1.444)
    n_bg = 1.444
    sim = invdes.planar_simulation([0, -1.9 * um], [6.4 * um, 1.9 * um], LAM, n_bg, n_si, steps_per_wavelength=14, run_time=500e-15, shutoff=1e-7)
    sim.add_medium(Medium("core", n=n_si))
    for y in (-0.8 * um, 0.8 * um):
        sim.add_structure(Box([-5 * um, y - 0.25 * um, -1], [12 * um, y + 0.25 * um, 1]), "core")
    freqs = C0 / (np.array([1.53, 1.55, 1.57]) * um)

    def port(name, x, y, d):
        return invdes.port_mode(sim, name, 0, x, [x, y - 0.75 * um, 0], [x, y + 0.75 * um, 1 * um], F0, d, "te")

    ports = [port("i1", 1 * um, -0.8 * um, "+"), port("i2", 1 * um, 0.8 * um, "+"), port("o1", 5.4 * um, -0.8 * um, "-"), port("o2", 5.4 * um, 0.8 * um, "-")]
    des = invdes.Design.effective_index([2 * um, -1.4 * um], [4.4 * um, 1.4 * um], n_bg, n_si, pixel=50e-9, filter_radius=120e-9)
    prob = invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=DEVICE)
    inv = invdes.InverseDesign(prob, des)
    rho = des.random(seed=1, mean=0.5, amplitude=0.4)
    beta = 4.0
    S, pullback = inv.forward(rho, beta)
    rng = np.random.default_rng(0)
    d = invdes.filter_density(rng.standard_normal(rho.shape), invdes.conic_kernel(4))
    d /= np.abs(d).max()
    h = 0.01
    assert rho.min() > h and rho.max() < 1 - h
    fd = (inv.s_matrix(rho + h * d, beta) - inv.s_matrix(rho - h * d, beta)) / (2 * h)
    return dict(inv=inv, rho=rho, beta=beta, S=S, pullback=pullback, d=d, fd=fd, sol=inv.last, ports=ports, sim=sim, des=des, freqs=freqs)


def test_complex_smatrix_vjp_2x2_multifrequency(planar):
    """dS[m, n, k] along a random density direction: real and imaginary parts
    of all 2 x 2 x 3 entries from 2 forward + 2 adjoint runs vs central
    differences (4 more runs)."""
    S, pb, d, fd = planar["S"], planar["pullback"], planar["d"], planar["fd"]
    assert S.shape == (2, 2, 3)
    adj = np.zeros_like(fd)
    for idx in np.ndindex(S.shape):
        g = np.zeros(S.shape, complex)
        g[idx] = 1.0  # L = Re S
        re = np.sum(pb(g) * d)
        g[idx] = 1.0j  # L = Im S
        adj[idx] = re + 1j * np.sum(pb(g) * d)
    err = np.abs(adj - fd)
    scale = np.abs(fd).max(axis=(0, 1))
    print("max |adj - fd| / max|fd| per frequency:", (err.max(axis=(0, 1)) / scale))
    print("relative error per entry:\n", err / np.abs(fd))
    assert np.all(np.abs(adj.real - fd.real) < 0.02 * scale)
    assert np.all(np.abs(adj.imag - fd.imag) < 0.02 * scale)
    assert np.all(err / np.abs(fd) < 0.08)
    assert np.median(err / np.abs(fd)) < 0.01
    # power is conserved to within the radiation loss of a random device
    assert np.all(np.sum(np.abs(S) ** 2, axis=0) < 1.01)


def test_real_objective_gradient(planar):
    """A real objective of the whole S (matrix fidelity) through the chain."""
    S, pb, d = planar["S"], planar["pullback"], planar["d"]
    T = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    val, g = obj.matrix_fidelity(S, T)
    dl_adj = float(np.sum(pb(g) * d))
    inv, rho, beta, h = planar["inv"], planar["rho"], planar["beta"], 0.01
    dl_fd = (obj.matrix_fidelity(inv.s_matrix(rho + h * d, beta), T)[0] - obj.matrix_fidelity(inv.s_matrix(rho - h * d, beta), T)[0]) / (2 * h)
    print(f"fidelity {val:.4f}: dL adjoint {dl_adj:.6f}, finite difference {dl_fd:.6f}")
    assert abs(dl_adj - dl_fd) < 0.03 * abs(dl_fd)


def test_concurrent_dispatch_and_reflection(planar):
    """Two workers give the same S; a port used as input and output gives a
    small reflection for a waveguide-matched device."""
    inv, des, ports, sim, freqs = planar["inv"], planar["des"], planar["ports"], planar["sim"], planar["freqs"]
    eps = des.eps(planar["rho"], planar["beta"])
    mon = ls.DFTMonitor("map", [2 * um, -1 * um, 0], [4 * um, 1 * um, sim.max[2]], freqs[:1], components=("Ey",))
    prob2 = invdes.SMatrixProblem(sim, des.grids(), ports[:2], [ports[2], ports[3], ports[0]], freqs, devices=[DEVICE, DEVICE], monitors=[mon])
    sol = prob2.solve(eps, adjoint=False)
    assert len(sol.fields) == 2 and sol.fields[0]["map"]["Ey"].shape[0] == 3 and np.abs(sol.fields[1]["map"]["Ey"]).max() > 0
    assert np.allclose(sol.S[:2], planar["S"], rtol=1e-4, atol=1e-5)
    assert len(sol.stats["runs"]) == 2 and sol.stats["adjoint_runs"] == 0
    assert np.all(np.abs(sol.S[2]) < 1.0)


def test_torch_layer_on_simulation(planar):
    torch = pytest.importorskip("torch")
    import lightsim.invdes.torch as lt

    rho = torch.tensor(planar["rho"], dtype=torch.float64, requires_grad=True)
    S = lt.s_matrix(planar["inv"], rho, beta=planar["beta"])
    T = torch.tensor(np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))
    c = (T.conj()[:, :, None] * S).sum(dim=(0, 1))
    loss = (c.abs() ** 2).mean() / 4.0
    loss.backward()
    val, g = obj.matrix_fidelity(planar["S"], T.numpy())
    assert abs(loss.item() - val) < 1e-4
    ref = planar["pullback"](g)
    assert np.allclose(rho.grad.numpy(), ref, rtol=1e-3, atol=1e-6 * np.abs(ref).max())


def test_jax_layer_on_simulation(planar):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import lightsim.invdes.jax as lj

    T = jnp.asarray(np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))

    def loss(r):
        S = lj.s_matrix(planar["inv"], r, beta=planar["beta"])
        c = jnp.sum(jnp.conj(T)[:, :, None] * S, axis=(0, 1))
        return jnp.mean(jnp.abs(c) ** 2) / 4.0

    val, grad = jax.value_and_grad(loss)(jnp.asarray(planar["rho"]))
    ref_val, g = obj.matrix_fidelity(planar["S"], np.asarray(T))
    assert abs(float(val) - ref_val) < 1e-4
    ref = planar["pullback"](g)
    assert np.allclose(np.asarray(grad), ref, rtol=1e-3, atol=1e-6 * np.abs(ref).max())


# ----------------------------------------------------------------------------
# 3-D
# ----------------------------------------------------------------------------


def _scene_3d():
    dx = 40e-9
    sim = ls.Simulation(min=[0, -1.2 * um, -0.9 * um], max=[4 * um, 1.2 * um, 0.9 * um], mesh=ls.UniformMesh(dx), run_time=220e-15, background="ox")
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_medium(Medium("sin", n=2.0))
    sim.add_structure(Box([-1, -0.4 * um, -0.2 * um], [5 * um, 0.4 * um, 0.2 * um]), "sin")
    pmin, pmax = [0.8 * um, -0.9 * um, -0.7 * um], [0.8 * um, 0.9 * um, 0.7 * um]
    mode = ls.solve_modes(sim, 0, 0.8 * um, pmin, pmax, F0, num_modes=1)[0]
    p_in = invdes.Port("in", 0, 0.8 * um, pmin, pmax, [mode.profile], "+")
    p_out = invdes.Port("out", 0, 3.2 * um, [3.2 * um, -0.9 * um, -0.7 * um], [3.2 * um, 0.9 * um, 0.7 * um], [mode.profile], "-")
    return sim, p_in, p_out, mode


def test_index_grid_sampling_matches_rasteriser():
    """grid_weights reproduces the core's IndexGrid rasterisation at the Yee
    positions of every E component, and its transpose is the exact adjoint."""
    from lightsim import _core

    sim, p_in, p_out, _ = _scene_3d()
    rng = np.random.default_rng(5)
    grid = invdes.DesignGrid([1.6 * um, 0.5 * um, -0.2 * um], [2.43 * um, 0.87 * um, 0.2 * um], (9, 6), Medium("g", eps=6.0))
    eps = rng.uniform(2.0, 6.0, grid.dims)
    prob = invdes.SMatrixProblem(sim, [grid], [p_in], [p_out], [F0], devices=DEVICE)
    s = prob.simulation([eps])
    sg = [np.asarray(a) for a in s.grid()]
    # a yz plane through the grid: eps_u = eps_y, eps_v = eps_z, eps_n = eps_x of the core
    x = 2.0 * um
    info = _core.plane_info(s.to_json(), 0, x, [x, 0.3 * um, -0.5 * um], [x, 1.05 * um, 0.5 * um], F0)
    o = list(info["origin"])
    o[0] = int(info["p"])  # x index of the plane
    nu, nv = info["dims"]
    checked = 0
    for comp, key in (("Ey", "eps_u"), ("Ez", "eps_v"), ("Ex", "eps_n")):
        w = invdes.grid_weights(comp, sg, o, (1, nu, nv), grid)
        mine = np.einsum("ia,jb,kc,abc->ijk", *w, eps)[0]
        core = np.asarray(info[key])
        own = mine != 0
        assert own.sum() > 20
        assert np.allclose(mine[own], core[own], rtol=2e-6), (comp, np.abs(mine - core)[own].max())
        # outside the owned voxels the core has the oxide background (or the smoothed core edge)
        assert np.all(core[~own] < 4.01)
        checked += own.sum()
        # dot-product test of the transpose
        g = rng.standard_normal(mine.shape)
        lhs = np.sum(g * mine)
        rhs = np.sum(np.einsum("ia,jb,kc,ijk->abc", *w, g[None]) * eps)
        assert abs(lhs - rhs) < 1e-10 * abs(lhs)
    assert checked > 100


def test_single_port_matches_mode_power_gradient():
    """One input, one output, one frequency: the Yee-voxel gradient of
    |S|^2 equals lightsim.adjoint.mode_power_gradient, and the complex
    derivative with respect to the grid samples matches finite differences."""
    sim, p_in, p_out, mode = _scene_3d()
    rng = np.random.default_rng(2)
    des = invdes.Design.slab([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.2 * um), 1.444**2, 4.0, pixel=40e-9, filter_radius=80e-9)
    rho = des.random(seed=4, mean=0.5, amplitude=0.3)
    beta = 3.0
    eps = des.eps(rho, beta)
    prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], [F0], devices=DEVICE, normalize="source")
    sol = prob.solve(eps)
    S = sol.S[0, 0, 0]
    yee = sol.yee_vjp(2 * sol.S)  # L = |S|^2
    # reference: same simulation through lightsim.adjoint
    ref_sim = prob.simulation(eps)
    ref_sim.add_source(ls.ModeSource(0, p_in.position, "+", p_in.min, p_in.max, mode.profile, prob.signal, name="src"))
    out = {"axis": 0, "position": p_out.position, "min": p_out.min, "max": p_out.max, "profile": mode.profile, "direction": "+"}
    dmin, dmax = prob._monitor_box(prob.grids[0])
    ref = mode_power_gradient(ref_sim, dmin, dmax, out, F0, device=DEVICE)
    assert abs(ref.a_plus - S) < 2e-3 * abs(S), (ref.a_plus, S)
    for c in ("Ex", "Ey", "Ez"):
        o, g = yee[0][c]
        assert list(o) == list(ref.grad[c][0])
        assert np.allclose(g, ref.grad[c][1], rtol=5e-2, atol=2e-3 * np.abs(g).max()), c
    # monitor-normalised S: unit incident amplitude (source-normalised is within a few %)
    prob_m = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], [F0], devices=DEVICE)
    inv = invdes.InverseDesign(prob_m, des)
    Sm, pb = inv.forward(rho, beta)
    assert abs(abs(Sm[0, 0, 0]) / abs(S) - 1) < 0.05
    # complex dS along a random density direction vs central differences
    d = rng.standard_normal(rho.shape)
    d /= np.abs(d).max()
    h = 0.02
    assert rho.min() > h and rho.max() < 1 - h
    fd = (inv.s_matrix(rho + h * d, beta) - inv.s_matrix(rho - h * d, beta))[0, 0, 0] / (2 * h)
    adj = np.sum(pb(np.ones((1, 1, 1), complex)) * d) + 1j * np.sum(pb(1j * np.ones((1, 1, 1), complex)) * d)
    print(f"3-D dS: adjoint {adj:.6f}, finite difference {fd:.6f}, rel {abs(adj - fd) / abs(fd):.4f}")
    assert abs(adj.real - fd.real) < 0.03 * abs(fd) and abs(adj.imag - fd.imag) < 0.03 * abs(fd)


def test_partial_etch_3d_gradient():
    """Two-level (partial etch) 3-D design: two stacked index grids, two
    frequencies; |S|^2 gradient along a density direction vs finite differences."""
    sim, p_in, p_out, _ = _scene_3d()
    freqs = C0 / (np.array([1.53, 1.57]) * um)
    des = invdes.Design.partial_etch([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.0, 0.2 * um), 1.444**2, 4.0, pixel=40e-9, filter_radius=80e-9)
    prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], freqs, devices=DEVICE)
    inv = invdes.InverseDesign(prob, des, objective=obj.mean_coupling)
    rho = des.random(seed=7, mean=0.5, amplitude=0.3)
    beta = 3.0
    val, grad, info = inv.value_and_grad(rho, beta)
    assert grad.shape == rho.shape == (2,) + des.shape and info["simulations"] == 2
    assert np.abs(grad[0]).max() > 0 and np.abs(grad[1]).max() > 0  # both levels carry gradient
    S, pb = inv.forward(rho, beta)
    rng = np.random.default_rng(1)
    d = rng.standard_normal(rho.shape)
    d /= np.abs(d).max()
    h = 0.02
    fd = (inv.s_matrix(rho + h * d, beta) - inv.s_matrix(rho - h * d, beta)) / (2 * h)
    for k in range(2):
        g = np.zeros(S.shape, complex)
        g[0, 0, k] = 1.0
        adj = np.sum(pb(g) * d) + 1j * np.sum(pb(1j * g) * d)
        print(f"partial etch, f{k}: dS adjoint {adj:.6f}, finite difference {fd[0, 0, k]:.6f}, rel {abs(adj - fd[0, 0, k]) / abs(fd[0, 0, k]):.4f}")
        assert abs(adj.real - fd[0, 0, k].real) < 0.03 * abs(fd[0, 0, k])
        assert abs(adj.imag - fd[0, 0, k].imag) < 0.03 * abs(fd[0, 0, k])


@pytest.mark.parametrize("engine", ["python", "native"])
def test_two_adjacent_design_grids_and_a_block(planar, engine):
    """FDTDX test_several_devices_with_overlapping_halos,
    test_gradient_region_without_device, test_gradient_regions_device_and_object:
    the pixel grid of the design and, sharing its face at x = 4.4 um, a
    uniform block between the two output waveguides (an index grid with a
    single sample). The gradient of sum |S|^2 on each grid against central
    differences along a direction confined to it.

    The continuous adjoint formula differs from the discrete derivative by
    its discretisation error, largest at material interfaces: 0.07 % for the
    smooth pixel direction, 2.1 % for the block, whose interface voxels all
    move together (the finite differences have converged: 2.07 ... 2.47 %
    for h = 0.01 ... 0.1)."""
    sim, ports, des, freqs = planar["sim"], planar["ports"], planar["des"], planar["freqs"]
    block = invdes.DesignGrid([4.4 * um, -0.4 * um, -1.0], [4.8 * um, 0.4 * um, 1.0], (1, 1), dict(des.grids()[0].medium), name="block")
    grids = [*des.grids(), block]
    prob = invdes.SMatrixProblem(sim, grids, ports[:2], ports[2:], freqs, devices=DEVICE, engine=engine)
    eps = [des.eps(planar["rho"], planar["beta"])[0], np.full(block.dims, 4.0)]
    sol = prob.solve(eps)
    grads = sol.vjp(2 * sol.S)  # L = sum |S|^2
    assert [g.shape for g in grads] == [g.dims for g in grids]
    rng = np.random.default_rng(1)
    d_pix = invdes.filter_density(rng.standard_normal(eps[0].shape[:2]), invdes.conic_kernel(4))[..., None]
    d_pix /= np.abs(d_pix).max()
    h = 0.05

    def loss(e):
        return float(np.sum(np.abs(prob.solve(e, adjoint=False).S) ** 2))

    for name, d, tol in (("pixel grid", [d_pix, np.zeros(block.dims)], 0.01), ("block", [np.zeros_like(eps[0]), np.ones(block.dims)], 0.04)):
        ad = sum(float(np.sum(g * di)) for g, di in zip(grads, d))
        fd = (loss([e + h * di for e, di in zip(eps, d)]) - loss([e - h * di for e, di in zip(eps, d)])) / (2 * h)
        print(f"{engine}, {name}: dL adjoint {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}")
        assert abs(ad - fd) < tol * abs(fd), (name, ad, fd)


def test_session_through_an_oriented_dispersive_medium():
    """FDTDX TestOrientedDispersionGradients: an oriented dispersive pole
    (off-diagonal permittivity, chi(w) u u^T with u along (1, 0, 1)) in a
    slab behind the design of the 3-D scene. Both engines accept it (the
    medium is reciprocal: the adjoint run is an ordinary run), return the
    same S and gradient, and the gradient is finite, non-trivial and agrees
    with central differences: 4.6 % along the gradient, where the same scene
    without the medium gives 3.5 % (the error of the continuous adjoint
    formula at the edges of this two-level design, see "Accuracy at
    interfaces" in docs/modules/autodiff.md)."""
    from lightsim.spec import Lorentz

    sim, p_in, p_out, _ = _scene_3d()
    sim.add_structure(Box([2.6 * um, -0.9 * um, -0.7 * um], [2.8 * um, 0.9 * um, 0.7 * um]), Medium("oriented", n=1.444, oriented=[(Lorentz(0.8, 2 * np.pi * 2.0 * F0, 1e14), (1.0, 0.0, 1.0))]))
    freqs = C0 / (np.array([1.53, 1.57]) * um)
    des = invdes.Design.partial_etch([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.0, 0.2 * um), 1.444**2, 4.0, pixel=40e-9, filter_radius=80e-9)
    rho = des.random(seed=7, mean=0.5, amplitude=0.3)
    beta = 3.0
    out = {}
    for engine in ("python", "native"):
        prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], freqs, devices=DEVICE, engine=engine)
        inv = invdes.InverseDesign(prob, des, objective=obj.mean_coupling)
        out[engine] = (inv, *inv.value_and_grad(rho, beta)[:2])
    (inv, vp, gp), (_, vn, gn) = out["python"], out["native"]
    print(f"oriented medium: objective python {vp:.6f}, native {vn:.6f}; max |grad_native - grad_python| / max {np.abs(gn - gp).max() / np.abs(gp).max():.1e}")
    assert np.all(np.isfinite(gp)) and np.abs(gp).max() > 0
    # one power measure for the port profiles in both engines: same gradient
    assert abs(vn - vp) < 1e-5 * abs(vp) and np.abs(gn - gp).max() < 1e-5 * np.abs(gp).max()
    d = gp / np.abs(gp).max()
    h = 0.02
    fd = (inv.value_and_grad(rho + h * d, beta)[0] - inv.value_and_grad(rho - h * d, beta)[0]) / (2 * h)
    ad = float(np.sum(gp * d))
    print(f"  along the gradient: adjoint {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}")
    assert abs(ad - fd) < 0.07 * abs(fd)


def _gpus():
    from lightsim import _core

    return sum(1 for d in _core.devices() if d.startswith("cuda"))


@pytest.mark.skipif(_gpus() < 2, reason="needs two GPUs")
def test_python_engine_runs_split_over_two_gpus():
    """FDTDX test_sharded_over_two_devices: with engine="python" every
    forward and adjoint run of the S-matrix problem is an ordinary run, and
    the device string "cuda:0,1" splits each of them over two GPUs (domain
    decomposition along x). S and the gradient equal the one-GPU result to
    round-off. (The native engine runs each simulation on one device.)"""
    sim, p_in, p_out, _ = _scene_3d()
    freqs = C0 / (np.array([1.53, 1.57]) * um)
    des = invdes.Design.partial_etch([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.0, 0.2 * um), 1.444**2, 4.0, pixel=40e-9, filter_radius=80e-9)
    eps = des.eps(des.random(seed=7, mean=0.5, amplitude=0.3), 3.0)
    out = {}
    for dev in ("cuda:0", "cuda:0,1"):
        prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], freqs, devices=dev, engine="python")
        sol = prob.solve(eps)
        out[dev] = (sol.S, sol.vjp(2 * sol.S), sol.stats)
    (s1, g1, st1), (s2, g2, st2) = out["cuda:0"], out["cuda:0,1"]
    ds = np.abs(s2 - s1).max() / np.abs(s1).max()
    dg = max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(g1, g2))
    print(f"split over two GPUs against one: S {ds:.1e}, gradient {dg:.1e}; backends {[r['backend'] for r in st2['runs']]}")
    assert all(" + " in r["backend"] for r in st2["runs"]) and all(" + " not in r["backend"] for r in st1["runs"])
    assert ds < 1e-5 and dg < 1e-4


def test_reverse_engine_3d_partial_etch():
    """engine="reverse": the S of the native engine; the gradient of the
    mean coupling of the two-level 3-D design against central differences,
    along the gradient and along a random direction, next to the native
    engine's."""
    sim, p_in, p_out, _ = _scene_3d()
    freqs = C0 / (np.array([1.53, 1.57]) * um)
    des = invdes.Design.partial_etch([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.0, 0.2 * um), 1.444**2, 4.0, pixel=40e-9, filter_radius=80e-9)
    rho = des.random(seed=7, mean=0.5, amplitude=0.3)
    beta = 3.0
    inv, res = {}, {}
    for engine in ("native", "reverse"):
        prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], freqs, devices=DEVICE, engine=engine)
        inv[engine] = invdes.InverseDesign(prob, des, objective=obj.mean_coupling)
        S, _ = inv[engine].forward(rho, beta)
        res[engine] = (S, *inv[engine].value_and_grad(rho, beta)[:2])
    (Sn, vn, gn), (Sr, vr, gr) = res["native"], res["reverse"]
    ds = np.abs(Sr - Sn).max() / np.abs(Sn).max()
    print(f"reverse engine: S against native {ds:.1e}")
    assert ds < 1e-5
    rng = np.random.default_rng(1)
    for name, d in (("gradient", gr / np.abs(gr).max()), ("random", rng.standard_normal(rho.shape))):
        d = d / np.abs(d).max()

        def f(r):
            return obj.mean_coupling(inv["reverse"].s_matrix(r, beta))[0]

        def cd(h):
            return (f(rho + h * d) - f(rho - h * d)) / (2 * h)

        # the random direction moves the objective 16 times less: Richardson
        fd = cd(0.01) if name == "gradient" else (4 * cd(0.01) - cd(0.02)) / 3
        ar, an = float(np.sum(gr * d)), float(np.sum(gn * d))
        print(f"  along the {name} direction: reverse {ar:.6e} (rel {abs(ar - fd) / abs(fd):.1e}), native {an:.6e} (rel {abs(an - fd) / abs(fd):.1e}), finite difference {fd:.6e}")
        # measured (CPU and RTX PRO 6000): 4e-5 ... 2e-4 along the gradient,
        # 2.5e-3 along the random direction (4.7e-7 absolute, 1.6e-4 of the
        # derivative along the gradient); the native engine: 4 % and 25 %
        assert abs(ar - fd) < 5e-3 * abs(fd)


def test_reverse_engine_refuses_oriented_media():
    """Reverse mode cannot step an oriented-pole (tensor) medium backwards:
    the reverse engine refuses it and names the engines that handle it."""
    from lightsim.spec import Lorentz

    sim, p_in, p_out, _ = _scene_3d()
    sim.add_structure(Box([2.6 * um, -0.9 * um, -0.7 * um], [2.8 * um, 0.9 * um, 0.7 * um]), Medium("oriented", n=1.444, oriented=[(Lorentz(0.8, 2 * np.pi * 2.0 * F0, 1e14), (1.0, 0.0, 1.0))]))
    des = invdes.Design.slab([1.6 * um, 0.48 * um], [2.4 * um, 0.8 * um], (-0.2 * um, 0.2 * um), 1.444**2, 4.0, pixel=40e-9)
    prob = invdes.SMatrixProblem(sim, des.grids(), [p_in], [p_out], [F0], devices=DEVICE, engine="reverse")
    with pytest.raises(RuntimeError, match="native"):
        prob.solve(des.eps(des.random(seed=1), 2.0))


def test_reverse_engine_with_an_excitation(planar):
    """engine="reverse" with an Excitation input (a mode source given as a
    plain source, normalised by its spectrum times ``norm``) next to a port
    input: the S of the native engine, and the gradient of a real objective
    against central differences (2.5-D)."""
    from lightsim.invdes import Excitation
    from lightsim.spec import GaussianPulse, ModeSource

    p, freqs, des, rho, beta = planar["ports"], planar["freqs"], planar["des"], planar["rho"], planar["beta"]
    f0 = 0.5 * (freqs.min() + freqs.max())
    sig = GaussianPulse(f0, max(float(np.ptp(freqs)), 0.1 * f0))
    src = ModeSource(p[0].axis, p[0].position, p[0].direction, p[0].min, p[0].max, p[0].profiles[0], sig)
    ex = Excitation("beam", src, norm=np.array([1.0, 0.8 + 0.3j, 1.2]))
    inv, S = {}, {}
    for engine in ("native", "reverse"):
        prob = invdes.SMatrixProblem(planar["sim"], des.grids(), [ex, p[1]], p[2:], freqs, devices=DEVICE, engine=engine, normalize="source")
        inv[engine] = invdes.InverseDesign(prob, des, objective=obj.mean_coupling)
        S[engine] = inv[engine].s_matrix(rho, beta)
    ds = np.abs(S["reverse"] - S["native"]).max() / np.abs(S["native"]).max()
    print(f"excitation input: S reverse against native {ds:.1e}")
    assert ds < 1e-5
    val, g = inv["reverse"].value_and_grad(rho, beta)[:2]
    d = planar["d"]

    def f(r):
        return obj.mean_coupling(inv["reverse"].s_matrix(r, beta))[0]

    def cd(h):
        return (f(rho + h * d) - f(rho - h * d)) / (2 * h)

    fd = (4 * cd(0.005) - cd(0.01)) / 3
    ad = float(np.sum(g * d))
    gn = inv["native"].value_and_grad(rho, beta)[1]
    an = float(np.sum(gn * d))
    print(f"  along a random direction: reverse {ad:.6e} (rel {abs(ad - fd) / abs(fd):.1e}), native {an:.6e} (rel {abs(an - fd) / abs(fd):.1e}), finite difference {fd:.6e}")
    assert abs(ad - fd) < 3e-3 * abs(fd)


def _newer_media():
    W0 = 2 * np.pi * F0
    return {
        "raman": Medium("raman", n=1.5, chi3=1e-20, raman=ls.Raman(0.2, 0.1 * W0, 0.03 * W0)),
        "gyro": Medium("gyro", eps=2.25, gyro=[(ls.Lorentz(1.0, 1.5 * W0, 1e13), (0.0, 0.0, 0.2 * W0))]),
        "director": Medium("director", director=ls.DirectorField([0, 0, 0], [1, 1, 1], [1.0, 1.0, 0.0], n_o=1.5, n_e=1.7)),
        "gain": Medium("gain", eps=2.25, gain=ls.FourLevel(1e24, W0, 1e13, 1e-13, 1e-12, 1e-13)),
    }


@pytest.mark.parametrize("kind", ["raman", "gyro", "director", "gain"])
def test_engines_refuse_the_newer_media(planar, kind):
    """A Raman, gyrotropic, director-field or gain medium next to the design
    is refused by every S-matrix engine (the Python check of SMatrixProblem)
    and by the Rust engines when they are built directly (GradientSession,
    ReverseSMatrix, reverse mode); each error names the medium."""
    from lightsim import autodiff
    from lightsim.spec import GaussianPulse, PointDipole

    sim = copy.deepcopy(planar["sim"])
    sim.add_structure(Box([4.6 * um, -1.9 * um, -1], [4.8 * um, 1.9 * um, 1]), _newer_media()[kind])
    p, freqs, des = planar["ports"], planar["freqs"], planar["des"]
    for engine in ("python", "native", "reverse"):
        with pytest.raises(ValueError, match=f"medium '{kind}'"):
            invdes.SMatrixProblem(sim, des.grids(), p[:1], p[2:3], freqs, devices=DEVICE, engine=engine)
    eps = des.eps(planar["rho"], planar["beta"])
    for cls in (autodiff.GradientSession, autodiff.ReverseSMatrix):
        with pytest.raises(RuntimeError, match=f"medium '{kind}'"):
            cls(sim, des.grids(), p[:1], p[2:3], freqs, devices=DEVICE).set_design(eps)
    sim.add_source(PointDipole([1.5 * um, -0.8 * um, 0.0], "Ez", GaussianPulse(F0, 0.25 * F0)))
    flux = autodiff.flux_energy(0, 5.4 * um, [5.4 * um, -1.4 * um, 0.0], [5.4 * um, 1.4 * um, 0.0])
    with pytest.raises(RuntimeError, match=f"medium '{kind}'"):
        autodiff.value_and_grad(sim, des.grids(), eps, flux, device=DEVICE)


@pytest.mark.skipif(ls.precision != "f64", reason="needs the double-precision build (maturin develop --features f64)")
def test_reverse_engine_double_precision(planar):
    """FDTDX's float64 gradient check through lightsim.invdes: engine
    "reverse" on the 2.5-D 2 x 2 x 3 problem, S against engine "python"
    and the gradient of a real objective against Richardson-extrapolated
    central differences, at double-precision tolerances."""
    p, freqs, des, rho, beta, d = planar["ports"], planar["freqs"], planar["des"], planar["rho"], planar["beta"], planar["d"]
    prob = invdes.SMatrixProblem(planar["sim"], des.grids(), p[:2], p[2:], freqs, devices=DEVICE, engine="reverse")
    inv = invdes.InverseDesign(prob, des, objective=obj.mean_coupling)
    S = inv.s_matrix(rho, beta)
    ds = np.abs(S - planar["S"]).max() / np.abs(planar["S"]).max()
    _, g = inv.value_and_grad(rho, beta)[:2]

    def f(r):
        return obj.mean_coupling(inv.s_matrix(r, beta))[0]

    def cd(h):
        return (f(rho + h * d) - f(rho - h * d)) / (2 * h)

    fd = (4 * cd(5e-4) - cd(1e-3)) / 3
    ad = float(np.sum(g * d))
    print(f"f64 reverse engine: S against the python engine {ds:.1e}; directional derivative {ad:.12e}, FD {fd:.12e}, rel {abs(ad - fd) / abs(fd):.1e}")
    assert ds < 1e-12
    assert abs(ad - fd) < 1e-8 * abs(fd)


# ----------------------------------------------------------------------------
# combined runs and random-probe gradients
# ----------------------------------------------------------------------------


def test_matrix_distance_gradient_and_probe_expectation():
    """matrix_distance: value identity, gradient against finite differences,
    and the random-probe estimator's mean equal to the exact gradient."""
    rng = np.random.default_rng(5)
    M, N, K = 3, 3, 2
    S = rng.standard_normal((M, N, K)) + 1j * rng.standard_normal((M, N, K))
    T = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))[0]
    val, g = obj.matrix_distance(S, T)
    t2 = np.sum(np.abs(T) ** 2)
    c = np.einsum("mn,mnk->k", np.conj(T), S)
    assert np.isclose(val, np.mean((np.sum(np.abs(S) ** 2, axis=(0, 1)) + t2 - 2 * np.abs(c)) / t2))
    assert obj.matrix_distance(np.exp(0.7j) * T[:, :, None] * np.ones(K), T)[0] < 1e-12
    for idx in [(0, 1, 0), (2, 2, 1)]:
        for unit in (1.0, 1.0j):
            h = 1e-6
            Sp, Sm = S.copy(), S.copy()
            Sp[idx] += h * unit
            Sm[idx] -= h * unit
            fd = (obj.matrix_distance(Sp, T)[0] - obj.matrix_distance(Sm, T)[0]) / (2 * h)
            assert abs(np.real(np.conj(g[idx]) * unit) - fd) < 1e-6 * max(1.0, abs(fd))
    # probe estimator (with the exact phase): g_S = sum_p g_Y[:, p] x_p^H, mean over probes -> g
    ph = np.exp(1j * np.angle(c))
    X = obj.probe_vectors(N, 200000, rng)  # [P, N]
    est = np.zeros_like(g)
    for k in range(K):
        R = (S[:, :, k] - ph[k] * T) @ X.T  # [M, P]
        gY = 2.0 / K * R / (X.shape[0] * t2)
        est[:, :, k] = gY @ np.conj(X)
    assert np.abs(est - g).max() < 0.02 * np.abs(g).max()


def test_combined_runs_match_the_exact_gradient(planar):
    """All inputs at once with complex weights: Y = S X with the launched X,
    X close to the requested weights (phase convention of the sources), and
    the gradient from P combined adjoint runs equal to the exact vjp of the
    equivalent upstream gradient on S."""
    inv, des, sol = planar["inv"], planar["des"], planar["sol"]
    prob = inv.problem
    eps = des.eps(planar["rho"], planar["beta"])
    rng = np.random.default_rng(11)
    W = obj.probe_vectors(2, 2, rng)
    prob.remeasure_launch = True  # exact check: launched amplitudes measured in every run
    cs = prob.solve_combined(eps, W)
    S = planar["S"]
    P, N, K = cs.X.shape
    assert (P, N, K) == (2, 2, 3) and cs.Y.shape == (2, 2, 3)
    pred = np.einsum("mnk,pnk->mpk", S, cs.X)
    assert np.abs(cs.Y - pred).max() < 2e-3 * np.abs(pred).max(), np.abs(cs.Y - pred).max() / np.abs(pred).max()
    werr = np.abs(cs.X - W[:, :, None]).max() / np.abs(W).max()
    print("launched vs requested input weights:", werr)
    assert werr < 0.05
    g = rng.standard_normal(cs.Y.shape) + 1j * rng.standard_normal(cs.Y.shape)
    g[:, :, 0] = 0.0  # one frequency without gradient: P * 2 adjoint runs
    grad = cs.vjp(g)
    assert cs.adjoint_stats["adjoint_runs"] == 4
    gerr = np.abs(cs.g_eff - g).max() / np.abs(g).max()
    print("launched vs requested adjoint weights:", gerr)
    assert gerr < 0.05
    gS = np.einsum("mpk,pnk->mnk", cs.g_eff, np.conj(cs.X))
    exact = sol.vjp(gS)
    for a, b in zip(grad, exact):
        rel = np.abs(a - b).max() / np.abs(b).max()
        print("combined vs exact gradient, max relative difference:", rel)
        assert rel < 5e-3
    # second call: the adjoint weights corrected with the launch factors measured by the first
    cs.vjp(g)
    assert np.abs(cs.g_eff - g).max() / np.abs(g).max() <= max(gerr, 5e-3)
    # default: launch factors measured once, reused on other designs (no norm monitors)
    prob.remeasure_launch = False
    cs.vjp(g)
    assert not any(r["launch_measured"] for r in cs.adjoint_stats["runs"])
    # a second combined forward solve on another design reuses the launch factors (no norm monitors)
    eps2 = des.eps(np.clip(planar["rho"] + 0.3 * planar["d"], 0, 1), planar["beta"])
    cs2 = prob.solve_combined(eps2, W)
    assert not any(r["launch_measured"] for r in cs2.stats["runs"])
    sol2 = prob.solve(eps2, adjoint=True)
    pred2 = np.einsum("mnk,pnk->mpk", sol2.S, cs2.X)
    assert np.abs(cs2.Y - pred2).max() < 1e-2 * np.abs(pred2).max(), np.abs(cs2.Y - pred2).max() / np.abs(pred2).max()
    grad2 = cs2.vjp(g)
    exact2 = sol2.vjp(np.einsum("mpk,pnk->mnk", cs2.g_eff, np.conj(cs2.X)))
    for a, b in zip(grad2, exact2):
        rel = np.abs(a - b).max() / np.abs(b).max()
        print("cached launch factors, combined vs exact gradient:", rel)
        assert rel < 2e-2


def test_probe_design_runs(planar):
    """ProbeDesign: estimated distance and gradient, 2 forward + 2 x 3 adjoint runs."""
    inv, des = planar["inv"], planar["des"]
    T = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    pd = invdes.ProbeDesign(inv.problem, des, T, probes=2, seed=3)
    val, grad, info = pd.value_and_grad(planar["rho"], planar["beta"])
    assert np.isfinite(val) and val > 0 and grad.shape == planar["rho"].shape and np.all(np.isfinite(grad))
    assert info["simulations"] == 2 + 2 * 3
    assert info["g_mismatch"] < 0.05
