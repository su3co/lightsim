"""Gradients inside the engine (lightsim.autodiff): the persistent native
S-matrix session against the Python path and against finite differences
(real and imaginary parts of S), and time-domain reverse mode against finite
differences.

Run on a GPU with LIGHTSIM_DEVICE=cuda:0.
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import autodiff, eim, invdes
from lightsim.spec import C0, Box, GaussianPulse, Medium, PointDipole, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1.55 * um
F0 = C0 / LAM


# ----------------------------------------------------------------------------
# 2.5-D: two inputs x two outputs x three frequencies
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
    rho = des.random(seed=1, mean=0.5, amplitude=0.4)
    eps = des.eps(rho, 4.0)
    return dict(sim=sim, ports=ports, des=des, freqs=freqs, rho=rho, eps=eps)


def test_native_session_matches_python_path(planar):
    """Same S, same per-voxel gradient, same grid gradient as the NumPy path,
    for an upstream gradient with independent real and imaginary parts."""
    sim, ports, des, freqs, eps = (planar[k] for k in ("sim", "ports", "des", "freqs", "eps"))
    py = invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=DEVICE, engine="python")
    nat = invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=DEVICE, engine="native")
    sp, sn = py.solve(eps), nat.solve(eps)
    assert sn.S.shape == (2, 2, 3) and isinstance(sn, invdes.NativeSolution)
    print("max |S_native - S_python| / |S|:", np.abs(sn.S - sp.S).max() / np.abs(sp.S).max())
    assert np.allclose(sn.S, sp.S, rtol=2e-5, atol=2e-6)
    rng = np.random.default_rng(3)
    g = rng.standard_normal(sp.S.shape) + 1j * rng.standard_normal(sp.S.shape)
    vp, vn = sp.vjp(g), sn.vjp(g)
    rel = np.abs(vn[0] - vp[0]).max() / np.abs(vp[0]).max()
    print("max |vjp_native - vjp_python| / max|vjp|:", rel)
    assert vn[0].shape == vp[0].shape == des.grids()[0].dims and rel < 2e-4
    yp, yn = sp.yee_vjp(g), sn.yee_vjp(g)
    for c in ("Ex", "Ey", "Ez"):
        assert list(yn[0][c][0]) == list(yp[0][c][0]) and yn[0][c][1].shape == yp[0][c][1].shape
        scale = max(np.abs(yp[0][c][1]).max(), 1e-300)
        assert np.abs(yn[0][c][1] - yp[0][c][1]).max() < 2e-4 * max(scale, np.abs(yp[0]["Ez"][1]).max())
    # complex Jacobian of one entry
    jp, jn = sp.jacobian(1, 0, 2), sn.jacobian(1, 0, 2)
    assert np.abs(jn[0] - jp[0]).max() < 2e-4 * np.abs(jp[0]).max()
    # a zero row of g needs no adjoint run for that output; stats are those of the python path
    assert sn.stats["forward_runs"] == 2 and sn.stats["adjoint_runs"] == 2 and len(sn.stats["runs"]) == 4
    assert sn.stats["cells"] == sp.stats["cells"] and sn.stats["steps"] == sp.stats["steps"]
    # the permittivity the engine uses is what the python path interpolates
    ye = nat.native_session().yee_permittivity()
    ref = py.yee_permittivity(eps)[0]
    for c in ("Ex", "Ey", "Ez"):
        o, a = ye[c]
        m = py._meta[0][c]
        oo = [o[i] - m["origin"][i] for i in range(3)]
        sub = ref[c][oo[0] : oo[0] + a.shape[0], oo[1] : oo[1] + a.shape[1], oo[2] : oo[2] + a.shape[2]]
        own = ~np.isnan(a)
        assert own.sum() > 100 and np.allclose(a[own], sub[own], rtol=2e-6)


def test_native_session_finite_differences(planar):
    """dS[m, n, k] along a random direction of the grid permittivity: real and
    imaginary parts of all 2 x 2 x 3 entries against central differences,
    using the session directly (set_design / forward / backward)."""
    sim, ports, des, freqs, eps = (planar[k] for k in ("sim", "ports", "des", "freqs", "eps"))
    sess = autodiff.GradientSession(sim, des.grids(), ports[:2], ports[2:], freqs, devices=DEVICE)
    sess.set_design(eps)
    S = sess.forward()
    assert [r["kind"] for r in sess.stats()] == ["fwd", "fwd"]
    rng = np.random.default_rng(0)
    d = invdes.filter_density(rng.standard_normal(eps[0].shape[:2]), invdes.conic_kernel(4))[..., None]
    d /= np.abs(d).max()
    adj = np.zeros(S.shape, complex)
    for idx in np.ndindex(S.shape):
        g = np.zeros(S.shape, complex)
        g[idx] = 1.0  # L = Re S
        re = np.sum(sess.backward(g)[0] * d)
        adj[idx] = re + 1j * np.sum(sess.backward(1j * g)[0] * d)  # L = Im S
    # only the two adjoint runs were added, whatever the number of backward calls
    assert [r["kind"] for r in sess.stats()] == ["fwd", "fwd", "adj", "adj"]
    h = 0.05
    sess.set_design([eps[0] + h * d])
    sp = sess.forward()
    sess.set_design([eps[0] - h * d])
    sm = sess.forward()
    fd = (sp - sm) / (2 * h)
    err = np.abs(adj - fd)
    scale = np.abs(fd).max(axis=(0, 1))
    print("native session, max |adj - fd| / max|fd| per frequency:", err.max(axis=(0, 1)) / scale)
    print("relative error per entry:\n", err / np.abs(fd))
    assert np.all(np.abs(adj.real - fd.real) < 0.02 * scale)
    assert np.all(np.abs(adj.imag - fd.imag) < 0.02 * scale)
    assert np.median(err / np.abs(fd)) < 0.01
    # back to the first design: the persistent buffers reproduce S exactly
    sess.set_design(eps)
    assert np.array_equal(sess.forward(), S)


def test_native_engine_through_the_inverse_design_layer(planar):
    """InverseDesign on engine='native': value and density gradient of a real
    objective equal those of the python engine; a pullback taken before
    other solves stays valid."""
    sim, ports, des, freqs, rho = (planar[k] for k in ("sim", "ports", "des", "freqs", "rho"))
    T = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    fom = lambda S: invdes.matrix_fidelity(S, T)  # noqa: E731
    out = {}
    for engine in ("python", "native"):
        prob = invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=DEVICE, engine=engine)
        inv = invdes.InverseDesign(prob, des, objective=fom)
        S, pb = inv.forward(rho, 4.0)
        inv.s_matrix(rho * 0.9, 4.0)  # another design in between
        val, g = fom(S)
        out[engine] = (val, pb(g), inv.value_and_grad(rho, 4.0))
    (vp, gp, (v2p, g2p, ip)), (vn, gn, (v2n, g2n, inn)) = out["python"], out["native"]
    assert abs(vn - vp) < 1e-5 and abs(v2n - v2p) < 1e-5
    assert np.abs(gn - gp).max() < 3e-4 * np.abs(gp).max() and np.abs(g2n - g2p).max() < 3e-4 * np.abs(g2p).max()
    assert inn["simulations"] == ip["simulations"] == 4


# ----------------------------------------------------------------------------
# time-domain reverse mode
# ----------------------------------------------------------------------------


def _reverse_scene(three_d):
    """A dipole next to a slab waveguide, a design block, a flux plane."""
    dx = 50e-9
    if three_d:
        sim = ls.Simulation(min=[0, 0, 0], max=[2.4 * um, 1.6 * um, 1.2 * um], mesh=ls.UniformMesh(dx), run_time=45e-15, background="ox")
        z0, z1, zc = 0.4 * um, 0.8 * um, 0.6 * um
        pol = "Ey"
    else:
        sim = ls.Simulation(
            min=[0, 0, 0],
            max=[3.2 * um, 2.0 * um, dx],
            mesh=ls.UniformMesh(dx),
            run_time=70e-15,
            background="ox",
            boundaries={"x": [ls.PML(), ls.PML()], "y": [ls.PML(), ls.PML()], "z": [ls.PERIODIC, ls.PERIODIC]},
        )
        z0, z1, zc = -1.0, 1.0, 0.0
        pol = "Ez"
    ly = sim.max[1]
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_medium(Medium("core", n=2.6))
    sim.add_structure(Box([-1, ly / 2 - 0.2 * um, z0], [10 * um, ly / 2 + 0.2 * um, z1]), "core")
    sim.add_source(PointDipole([0.5 * um, ly / 2, zc], pol, GaussianPulse(F0, 0.25 * F0), amplitude=1e-15))
    x0, x1 = (1.0 * um, 1.6 * um) if three_d else (1.2 * um, 2.0 * um)
    grid = invdes.DesignGrid([x0 + 0.01 * um, ly / 2 - 0.41 * um, z0], [x1 + 0.03 * um, ly / 2 + 0.43 * um, z1], (7, 9), Medium("design", eps=2.6**2))
    xp = sim.max[0] - 0.5 * um
    obj_box = ([xp, ly / 2 - 0.6 * um, max(z0, 0.1 * um) if three_d else 0.0], [xp, ly / 2 + 0.6 * um, 1.1 * um if three_d else dx])
    return sim, grid, xp, obj_box


@pytest.mark.parametrize("three_d", [False, True], ids=["2.5-D", "3-D"])
def test_reverse_mode_time_window_flux(three_d):
    """Energy through a plane in a time window: dL/d(eps) from one reverse
    pass against central differences along a random direction and for single
    samples."""
    sim, grid, xp, (mn, mx) = _reverse_scene(three_d)
    t1 = sim.run_time
    obj = autodiff.flux_energy(0, xp, mn, mx, t_start=0.25 * t1, t_end=0.9 * t1, taper=0.1 * t1)
    rng = np.random.default_rng(4)
    eps = rng.uniform(2.5, 6.5, grid.dims)
    L, g, info = autodiff.value_and_grad(sim, grid, eps, obj, device=DEVICE)
    sess = info["session"]
    assert g.shape == grid.dims and L > 0 and np.all(np.isfinite(g))
    d = rng.standard_normal(grid.dims)
    d /= np.abs(d).max()
    h = 0.03
    lp = autodiff.value_and_grad(sim, grid, eps + h * d, obj, session=sess)[0]
    lm = autodiff.value_and_grad(sim, grid, eps - h * d, obj, session=sess)[0]
    fd, ad = (lp - lm) / (2 * h), float(np.sum(g * d))
    print(f"{'3-D' if three_d else '2.5-D'} on {sess.device}: L = {L:.5e}, dL adjoint {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}")
    print(f"  forward {info['forward_seconds']:.2f} s, reverse {info['reverse_seconds']:.2f} s, {info['steps']} steps, {info['cells']} cells; shell {info['shell_floats_per_step']} floats/step")
    assert abs(ad - fd) < 3e-3 * abs(fd)
    # one sample: the largest entry of the gradient
    i = np.unravel_index(np.argmax(np.abs(g)), g.shape)
    e1 = np.zeros(grid.dims)
    e1[i] = 1.0
    h1 = 0.1
    f1 = (autodiff.value_and_grad(sim, grid, eps + h1 * e1, obj, session=sess)[0] - autodiff.value_and_grad(sim, grid, eps - h1 * e1, obj, session=sess)[0]) / (2 * h1)
    print(f"  largest entry: adjoint {g[i]:.6e}, finite difference {f1:.6e}")
    assert abs(g[i] - f1) < 5e-3 * abs(f1)


def test_reverse_mode_custom_objective():
    """A caller-defined objective on recorded samples (here the time-integrated
    |E|^2 behind the design) through the cotangent interface."""
    sim, grid, xp, (mn, mx) = _reverse_scene(False)
    obj = autodiff.field_samples(mn, mx, components=("Ez",))
    rng = np.random.default_rng(7)
    eps = rng.uniform(2.5, 6.5, grid.dims)

    def loss(series):
        e = series["Ez"].astype(np.float64)
        w = np.linspace(0.0, 1.0, e.shape[0])[:, None, None, None]  # weight late arrivals more
        return float(np.sum(w * e * e)), {"Ez": 2 * w * e}

    L, g, info = autodiff.value_and_grad(sim, grid, eps, obj, device=DEVICE, fn=loss)
    sess = info["session"]
    assert sess.series()["Ez"].shape[0] == sess.steps and sess.times("Ez")[0] == sess.dt
    d = rng.standard_normal(grid.dims)
    d /= np.abs(d).max()
    h = 0.03
    lp = autodiff.value_and_grad(sim, grid, eps + h * d, obj, fn=loss, session=sess)[0]
    lm = autodiff.value_and_grad(sim, grid, eps - h * d, obj, fn=loss, session=sess)[0]
    fd, ad = (lp - lm) / (2 * h), float(np.sum(g * d))
    print(f"custom objective on {sess.device}: L = {L:.5e}, dL adjoint {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}")
    assert abs(ad - fd) < 3e-3 * abs(fd)


# ----------------------------------------------------------------------------
# FDTDX gradient tests, mirrored (test_adjoint_gradient.py): phasor-detector
# objectives with their windows, and the symmetry-reduced domain
# ----------------------------------------------------------------------------


def _phasor_power(sess, freqs, window):
    """FDTDX's phasor-detector loss sum |phasor|^2 over the cells, the
    components and the frequencies of the recorded samples; ``window(n, t)``
    weights sample n at time t (gate, subsampling, apodization)."""

    def fn(series):
        L, cot = 0.0, {}
        for comp, s in series.items():
            s = s.astype(np.float64)
            t = sess.times(comp.split("_")[0])
            w = window(np.arange(len(t)), t) * sess.dt
            ph = w[:, None] * np.exp(2j * np.pi * np.outer(t, freqs))  # [step, freq]
            F = np.tensordot(ph, s, axes=(0, 0))  # [freq, i, j, k]
            L += float(np.sum(np.abs(F) ** 2))
            cot[comp] = 2.0 * np.real(np.tensordot(ph, np.conj(F), axes=(1, 0)))
        return L, cot

    return fn


def _tukey(t, a, b, alpha=0.5):
    x = (t - a) / (b - a)
    e = alpha / 2
    w = np.where((x < 0) | (x > 1), 0.0, 1.0)
    w = np.where((x >= 0) & (x < e), 0.5 * (1 - np.cos(np.pi * x / e)), w)
    return np.where((x <= 1) & (x > 1 - e), 0.5 * (1 - np.cos(np.pi * (1 - x) / e)), w)


@pytest.mark.parametrize("case", ["one frequency", "three frequencies", "gated, every 3rd, Tukey"])
def test_reverse_mode_phasor_power(case):
    """A dipole, a design block and a phasor detector (FDTDX test_dipole_*,
    test_exact_dipole, test_frequencies_sharing_one_adjoint_run,
    test_exact_detector_windows): reverse mode takes any frequency set and
    window in one reverse pass."""
    sim, grid, xp, (mn, mx) = _reverse_scene(False)
    t1 = sim.run_time
    freqs, window = {
        "one frequency": ([F0], lambda n, t: np.ones_like(t)),
        "three frequencies": ([0.9 * F0, F0, 1.1 * F0], lambda n, t: np.ones_like(t)),
        "gated, every 3rd, Tukey": ([0.9 * F0, 1.1 * F0], lambda n, t: 3.0 * (n % 3 == 0) * ((t >= 0.15 * t1) & (t <= 0.92 * t1)) * _tukey(t, 0.2 * t1, 0.85 * t1)),
    }[case]
    sess = autodiff.ReverseSession(sim, grid, autodiff.field_samples(mn, mx, components=("Ez", "Hx", "Hy")), device=DEVICE)
    fn = _phasor_power(sess, np.asarray(freqs), window)
    rng = np.random.default_rng(5)
    eps = rng.uniform(2.5, 6.5, grid.dims)
    L, g, _ = autodiff.value_and_grad(sim, grid, eps, None, fn=fn, session=sess)
    d = rng.standard_normal(grid.dims)
    d /= np.abs(d).max()
    h = 0.03
    lp = autodiff.value_and_grad(sim, grid, eps + h * d, None, fn=fn, session=sess)[0]
    lm = autodiff.value_and_grad(sim, grid, eps - h * d, None, fn=fn, session=sess)[0]
    fd, ad = (lp - lm) / (2 * h), float(np.sum(g * d))
    print(f"phasor power, {case}, on {sess.device}: L = {L:.5e}, dL reverse mode {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}")
    assert L > 0 and abs(ad - fd) < 3e-3 * abs(fd)


@pytest.mark.parametrize("parity", [1, -1], ids=["PMC", "PEC"])
def test_reverse_mode_symmetry_reduced_domain(parity):
    """FDTDX test_mirror_symmetry: the domain reduced by
    ``lightsim.symmetry.reduce`` (upper half in y, closed by a PMC wall for
    E tangential even, a PEC wall for odd) gives half the objective of the
    full domain and the same gradient on its design grid; the full domain's
    two mirrored grids have mirrored gradients."""
    from lightsim import symmetry

    sim, _, xp, (mn, mx) = _reverse_scene(False)
    c = sim.max[1] / 2
    if parity < 0:
        # E_z odd about y = c: two dipoles of opposite sign off the plane
        sim.sources = []
        for y, a in ((c - 0.1 * um, -1.0), (c + 0.1 * um, 1.0)):
            sim.add_source(PointDipole([0.5 * um, y, 0.0], "Ez", GaussianPulse(F0, 0.25 * F0), amplitude=a * 1e-15))
    med = Medium("design", eps=2.6**2)
    upper = invdes.DesignGrid([1.21 * um, c + 0.03 * um, -1.0], [2.03 * um, c + 0.43 * um, 1.0], (7, 4), med, name="upper")
    lower = invdes.DesignGrid([1.21 * um, c - 0.43 * um, -1.0], [2.03 * um, c - 0.03 * um, 1.0], (7, 4), med, name="lower")
    t1 = sim.run_time
    window = dict(t_start=0.25 * t1, t_end=0.9 * t1, taper=0.1 * t1)
    rng = np.random.default_rng(6)
    eps = rng.uniform(2.5, 6.5, upper.dims)
    full_obj = autodiff.flux_energy(0, xp, mn, mx, **window)
    Lf, (gu, gl), _ = autodiff.value_and_grad(sim, [upper, lower], [eps, eps[:, ::-1]], full_obj, device=DEVICE)
    red, info = symmetry.reduce(sim, (0, parity, 0))
    assert red.boundaries["y"][0]["type"] == ("pmc" if parity > 0 else "pec")
    half_obj = autodiff.flux_energy(0, xp, [mn[0], c, mn[2]], mx, **window)
    Lh, gh, info = autodiff.value_and_grad(red, [upper], [eps], half_obj, device=DEVICE)
    scale = np.abs(gu).max()
    d_half = np.abs(gh[0] - gu).max() / scale
    d_mirror = np.abs(gl[:, ::-1] - gu).max() / scale
    print(f"symmetry {'PMC' if parity > 0 else 'PEC'} on {info['session'].device}: L full {Lf:.6e}, 2 L half {2 * Lh:.6e}; grad half vs full {d_half:.1e}, mirrored grids {d_mirror:.1e}; cells {info['cells']}")
    assert abs(2 * Lh - Lf) < 1e-4 * Lf
    assert d_half < 2e-3 and d_mirror < 2e-3


def test_reverse_mode_permeability_gradient():
    """FDTDX test_inverse_permeability_gradient_inside_devices: a permeability
    design grid behind a magnetically lossy slab (sigma_m), dL/d(mu) by
    reverse mode against central differences; the S-matrix problem refuses
    permeability grids."""
    sim, grid, xp, (mn, mx) = _reverse_scene(False)
    ly = sim.max[1]
    sim.add_structure(Box([2.1 * um, -1, -1], [2.2 * um, 10 * um, 1]), Medium("magnetic", n=1.444, mu=1.5, sigma_m=3e8))
    mug = invdes.DesignGrid(grid.min, grid.max, grid.dims[:2], Medium("mudesign", n=1.444, mu=4.0), name="mu", permeability=True)
    t1 = sim.run_time
    obj = autodiff.flux_energy(0, xp, mn, mx, t_start=0.25 * t1, t_end=0.9 * t1, taper=0.1 * t1)
    rng = np.random.default_rng(8)
    mu = rng.uniform(1.0, 4.0, mug.dims)
    L, g, info = autodiff.value_and_grad(sim, mug, mu, obj, device=DEVICE)
    sess = info["session"]
    d = rng.standard_normal(mug.dims)
    d /= np.abs(d).max()
    h = 0.03
    lp = autodiff.value_and_grad(sim, mug, mu + h * d, obj, session=sess)[0]
    lm = autodiff.value_and_grad(sim, mug, mu - h * d, obj, session=sess)[0]
    fd, ad = (lp - lm) / (2 * h), float(np.sum(g * d))
    print(f"permeability grid on {sess.device}: L = {L:.5e}, dL reverse mode {ad:.6e}, finite difference {fd:.6e}, rel {abs(ad - fd) / abs(fd):.2e}, residual {info['reconstruction_residual']:.1e}")
    assert abs(ad - fd) < 3e-3 * abs(fd)
    assert 0 <= info["reconstruction_residual"] < 1e-3
    with pytest.raises(ValueError, match="reverse mode"):
        invdes.SMatrixProblem(sim, [mug], [], [], [F0])
    del ly
