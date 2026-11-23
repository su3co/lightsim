"""Composite S-matrix problems (lightsim.invdes.network / composite)."""

import numpy as np
import pytest

from lightsim import invdes
from lightsim.invdes.network import Network, chain


def _rand_s(rng, P, K, scale=0.4):
    return scale * (rng.standard_normal((P, P, K)) + 1j * rng.standard_normal((P, P, K))) / np.sqrt(P)


def _cascade():
    # three 4-port blocks (ports 0, 1 on the left, 2, 3 on the right) in a row
    conns = [((0, 2), (1, 0)), ((0, 3), (1, 1)), ((1, 2), (2, 0)), ((1, 3), (2, 1))]
    ext = [(0, 0), (0, 1), (2, 2), (2, 3)]
    return Network([4, 4, 4], conns, ext, sides=[["in", "in", "out", "out"]] * 3)


def _brute(S, k):
    """Solve the network by brute force: every incoming wave is an unknown."""
    Pt = 12
    Sg = np.zeros((Pt, Pt), complex)
    for b in range(3):
        Sg[4 * b : 4 * b + 4, 4 * b : 4 * b + 4] = S[b][:, :, k]
    C = np.zeros((Pt, Pt))
    for a, c in [(2, 4), (3, 5), (6, 8), (7, 9)]:
        C[a, c] = C[c, a] = 1
    ext = [0, 1, 10, 11]
    T = np.zeros((4, 4), complex)
    for j, e in enumerate(ext):
        x = np.zeros(Pt, complex)
        x[e] = 1
        b_in = np.linalg.solve(np.eye(Pt) - C @ Sg, x)
        T[:, j] = (Sg @ b_in)[ext]
    return T


def test_network_matches_brute_force():
    rng = np.random.default_rng(1)
    S = [_rand_s(rng, 4, 2) for _ in range(3)]
    sol = _cascade().solve(S)
    for k in range(2):
        assert np.allclose(sol.T[:, :, k], _brute(S, k), atol=1e-12)


def test_network_vjp_matches_finite_differences():
    rng = np.random.default_rng(2)
    net = _cascade()
    S = [_rand_s(rng, 4, 2) for _ in range(3)]
    W = _rand_s(rng, 4, 2)

    def loss(S):
        return float(np.sum(np.abs(net.solve(S, store=False).T - W) ** 2))

    sol = net.solve(S)
    G = sol.vjp(2 * (sol.T - W))
    for _ in range(3):
        D = [_rand_s(rng, 4, 2) for _ in range(3)]
        h = 1e-6
        fd = (loss([s + h * d for s, d in zip(S, D)]) - loss([s - h * d for s, d in zip(S, D)])) / (2 * h)
        an = sum(float(np.real(np.sum(np.conj(g) * d))) for g, d in zip(G, D))
        assert an == pytest.approx(fd, rel=1e-6, abs=1e-9)


def test_feed_forward_equals_chain():
    rng = np.random.default_rng(3)
    net = _cascade()
    S = [_rand_s(rng, 4, 1) for _ in range(3)]
    sol = net.solve(S, reflections=False)
    T, pb = chain([s[2:, :2] for s in S])
    assert np.allclose(sol.T[2:, :2], T, atol=1e-14)
    g = np.zeros_like(sol.T)
    g[2:, :2] = rng.standard_normal((2, 2, 1))
    Gn, Gc = sol.vjp(g), pb(g[2:, :2])
    for b in range(3):
        assert np.allclose(Gn[b][2:, :2], Gc[b], atol=1e-14)
        assert np.all(Gn[b][:2, :2] == 0) and np.all(Gn[b][2:, 2:] == 0)


def test_network_rejects_dangling_ports():
    with pytest.raises(ValueError):
        Network([2, 2], [((0, 1), (1, 0))], [(0, 0)])


# ---------------------------------------------------------------------------
# lightsim blocks: two waveguide segments with a design bump, composed, against
# the monolithic simulation of both (2.5-D, coarse, CPU)

um = 1e-6
DX = 0.05  # um


def _snap(v):
    return float(np.round(v / DX) * DX)


def _setup():
    import lightsim as ls  # noqa: F401
    from lightsim import eim
    from lightsim.spec import C0, Box, Medium

    n_slab = float(eim.slab_neff(3.476, 0.22 * um, 1.55 * um, 1.444))
    f0 = C0 / (1.55 * um)

    def sim(x0, x1):
        s = invdes.planar_simulation([_snap(x0) * um, -1.5 * um], [_snap(x1) * um, 1.5 * um], 1.55 * um, 1.444, n_slab, steps_per_wavelength=1.55 / (n_slab * DX), run_time=1.2e-12, shutoff=1e-5)
        s.add_medium(Medium("core", n=n_slab))
        s.add_structure(Box([-1.0, -0.25 * um, -1.0], [1.0, 0.25 * um, 1.0]), "core")
        return s

    ref = sim(-2.0, 2.0)
    p0 = invdes.port_mode(ref, "ref", 0, 0.0, [0.0, -0.75 * um, 0.0], [0.0, 0.75 * um, ref.max[2]], f0, "+", "te")
    prof = p0.profiles[0]

    def port(name, x, d):
        return invdes.Port(name, 0, _snap(x) * um, [_snap(x) * um, -0.75 * um, 0.0], [_snap(x) * um, 0.75 * um, 1 * um], [prof], d)

    def design(xc):
        return invdes.Design.effective_index([(xc - 0.5) * um, -0.5 * um], [(xc + 0.5) * um, 0.5 * um], 1.444, n_slab, shape=(11, 11))

    return sim, port, design, f0


def _two_blocks():
    sim, port, design, f0 = _setup()
    freqs = [f0]
    # block A: design centred at x = 0, ports at -1.5 and +1.5; block B: design at 3, ports at 1.5, 4.5
    blocks, designs = [], []
    for xc in (0.0, 3.0):
        s = sim(xc - 2.7, xc + 2.7)
        d = design(xc)
        ports = [port("L", xc - 1.5, "+"), port("R", xc + 1.5, "-")]
        prob = invdes.SMatrixProblem(s, d.grids(), ports, ports, freqs)
        blocks.append(invdes.Block(prob))
        designs.append(d)
    net = Network([2, 2], [((0, 1), (1, 0))], [(0, 0), (1, 1)], sides=[["in", "out"]] * 2)
    cp = invdes.CompositeProblem(blocks, net, inputs=[0], outputs=[1, 0])
    ms = sim(-2.7, 5.7)
    mono_ports = [port("L", -1.5, "+"), port("R", 4.5, "-")]
    grids = []
    for i, d in enumerate(designs):
        g = d.grids()[0]
        g.name, g.medium = f"d{i}", dict(g.medium, name=f"invdes_d{i}")
        grids.append(g)
    mp = invdes.SMatrixProblem(ms, grids, [mono_ports[0]], [mono_ports[1], mono_ports[0]], freqs)
    return cp, mp, invdes.CompositeDesign(designs)


@pytest.fixture(scope="module")
def two_blocks():
    return _two_blocks()


def test_composite_matches_monolithic(two_blocks):
    cp, mp, design = two_blocks
    # a waveguide through both blocks, cladding of a slightly raised index
    # around it and a bump on the core: scattering, reflection and some
    # radiation (which the composition leaves out)
    rho = np.full((2, 11, 11), 0.3)
    rho[:, :, 3:8] = 1.0
    rho[0, 4:7, 1:3] = 0.8
    rho[1, 2:5, 8:10] = 0.6
    eps = design.eps(rho, 2.0)
    Sc = cp.solve(eps, adjoint=False).S
    Sm = mp.solve(eps, adjoint=False).S
    # transmission and reflection of the cascade, including the multiple
    # reflections between the blocks; the difference is the radiation of one
    # block that reaches the other and the port planes 1 um from the designs
    # (1.5e-2 here), and for the reflection also the backward leakage of the
    # mode sources, about 1e-2 (a straight waveguide gives 5e-4 in
    # transmission)
    assert abs(Sc[0, 0, 0] - Sm[0, 0, 0]) < 2.5e-2, (Sc[..., 0], Sm[..., 0])
    assert abs(Sc[1, 0, 0] - Sm[1, 0, 0]) < 3e-2, (Sc[..., 0], Sm[..., 0])


def test_composite_gradient(two_blocks):
    cp, mp, design = two_blocks
    rng = np.random.default_rng(0)
    rho = 0.5 + 0.3 * rng.uniform(-1, 1, (2, 11, 11))
    obj = lambda S: (float(np.abs(S[0, 0, 0]) ** 2), np.array([[[2 * S[0, 0, 0]]], [[0.0]]]))  # noqa: E731
    inv = invdes.InverseDesign(cp, design, objective=obj)
    val, grad, _ = inv.value_and_grad(rho, 2.0)
    d = rng.standard_normal(rho.shape)
    h = 1e-2
    fd = (obj(inv.s_matrix(np.clip(rho + h * d, 0, 1), 2.0))[0] - obj(inv.s_matrix(np.clip(rho - h * d, 0, 1), 2.0))[0]) / (2 * h)
    assert float(np.sum(grad * d)) == pytest.approx(fd, rel=0.05)


def test_layer_problem_matches_separate_blocks():
    """Two blocks side by side (3 um apart) solved in one LayerProblem, with
    one colour (both driven together) and with two, against each block
    alone."""
    from lightsim import eim
    from lightsim.spec import C0, Box, Medium

    n_slab = float(eim.slab_neff(3.476, 0.22 * um, 1.55 * um, 1.444))
    f0 = C0 / (1.55 * um)

    def sim(ys, y0, y1):
        s = invdes.planar_simulation([-2.7 * um, y0 * um], [2.7 * um, y1 * um], 1.55 * um, 1.444, n_slab, steps_per_wavelength=1.55 / (n_slab * DX), run_time=1.2e-12, shutoff=1e-5)
        s.add_medium(Medium("core", n=n_slab))
        for y in ys:
            s.add_structure(Box([-1.0, (y - 0.25) * um, -1.0], [1.0, (y + 0.25) * um, 1.0]), "core")
        return s

    ref = sim([0.0], -1.5, 1.5)
    prof = invdes.port_mode(ref, "ref", 0, 0.0, [0.0, -0.75 * um, 0.0], [0.0, 0.75 * um, ref.max[2]], f0, "+", "te").profiles[0]

    def port(name, x, y, d):
        return invdes.Port(name, 0, x * um, [x * um, (y - 0.75) * um, 0.0], [x * um, (y + 0.75) * um, 1 * um], [prof], d)

    def design(yc, i):
        d = invdes.Design.effective_index([-0.5 * um, (yc - 0.5) * um], [0.5 * um, (yc + 0.5) * um], 1.444, n_slab, shape=(11, 11), name=f"blk{i}")
        return d

    ys = [-1.5, 1.5]
    designs = [design(y, i) for i, y in enumerate(ys)]
    rng = np.random.default_rng(4)
    rho = []
    for d in designs:
        r = np.full(d.shape, 0.3)
        r[:, 3:8] = 1.0
        r[3:8, 1:3] = rng.uniform(0.4, 0.9)
        rho.append(r)
    eps = [e for d, r in zip(designs, rho) for e in d.eps(r, 2.0)]
    alone = []
    for y, d, e in zip(ys, designs, eps):
        ports = [port("L", -1.5, y, "+"), port("R", 1.5, y, "-")]
        p = invdes.SMatrixProblem(sim([y], y - 1.5, y + 1.5), d.grids(), [ports[0]], [ports[1]], [f0])
        alone.append(p.solve([e], adjoint=False).S)
    for colours in (1, 2):
        members = []
        for i, (y, d) in enumerate(zip(ys, designs)):
            members.append({"grids": d.grids(), "ports": [port(f"L{i}", -1.5, y, "+"), port(f"R{i}", 1.5, y, "-")], "in": [0], "out": [1], "colour": i % colours})
        lp = invdes.LayerProblem(sim(ys, -3.0, 3.0), members, [f0])
        sol = lp.solve(eps, adjoint=True)
        for i in range(2):
            assert abs(sol.block(i).S[0, 0, 0] - alone[i][0, 0, 0]) < 1e-2
            g = sol.block(i).vjp(np.ones((1, 1, 1), complex))
            assert len(g) == 1 and g[0].shape == designs[i].grids()[0].dims
