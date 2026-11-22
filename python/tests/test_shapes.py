"""Shape parametrisations (lightsim.invdes.shapes): the area-fraction
rasteriser and its transpose, level sets, the shape -> permittivity chain,
d|S21|^2 / d(vertex) of real simulations against central finite differences
(2.5-D and 3-D, one and two levels, both engines), and a short taper
optimisation.

Run on a GPU with LIGHTSIM_DEVICE=cuda:0.
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import eim, invdes
from lightsim.invdes import shapes as sh
from lightsim.spec import C0, Box, Medium, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1.55 * um
F0 = C0 / LAM


def _fd(f, x, h):
    g = np.zeros_like(x)
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _star(rng, n, c, r0, r1):
    ang = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * np.pi / n
    r = rng.uniform(r0, r1, n)
    return np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]


# ----------------------------------------------------------------------------
# rasteriser
# ----------------------------------------------------------------------------


def test_polygon_area_fractions():
    """Exact area fractions: aligned squares, both orientations, a polygon
    reaching outside the grid (against 400-point supersampling per cell)."""
    from matplotlib.path import Path

    x, y = np.linspace(0, 1.0, 11), np.linspace(0, 0.7, 8)  # cells of 0.1 x 0.1 centred on the samples
    sq = np.array([[0.25, 0.15], [0.45, 0.15], [0.45, 0.35], [0.25, 0.35]])
    d = sh.polygon_density(sq, x, y)
    assert np.allclose(d[3:5, 2:4], 1.0) and np.isclose(d.sum(), 4.0) and np.isclose(d.max(), 1.0)
    rng = np.random.default_rng(0)
    for V in (_star(rng, 9, (0.5, 0.35), 0.2, 0.45), _star(rng, 12, (0.9, 0.1), 0.15, 0.5)):
        d = sh.polygon_density(V, x, y)
        assert np.allclose(d, sh.polygon_density(V[::-1], x, y), atol=1e-14)
        assert d.min() > -1e-12 and d.max() < 1 + 1e-12
        k = 40
        cx = (np.arange(11 * k) + 0.5) / k * 0.1 - 0.05
        cy = (np.arange(8 * k) + 0.5) / k * 0.1 - 0.05
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        ss = Path(V).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(11, k, 8, k).mean(axis=(1, 3))
        assert np.abs(ss - d).max() < 2.0 / k, np.abs(ss - d).max()
    V = _star(rng, 9, (0.5, 0.35), 0.2, 0.3)
    assert np.isclose(sh.polygon_density(V, x, y).sum() * 0.01, abs(sh._signed_area(V)), rtol=1e-12)


def _shape_cases():
    rng = np.random.default_rng(1)
    star = _star(rng, 8, (0.5e-6, 0.35e-6), 0.15e-6, 0.32e-6)
    half = np.array([[0.05, 0.35], [0.2, 0.55], [0.45, 0.62], [0.7, 0.5], [0.95, 0.35]]) * 1e-6
    halfx = np.array([[0.5, 0.02], [0.75, 0.2], [0.8, 0.45], [0.62, 0.68], [0.5, 0.69]]) * 1e-6

    def spline(c):  # 4 control heights -> 12-vertex upper outline of a strip
        from scipy.interpolate import CubicSpline

        xs = np.linspace(0.1e-6, 0.9e-6, 12)
        top = CubicSpline(np.linspace(0.1e-6, 0.9e-6, 4), c)(xs)
        return np.concatenate([[[0.1e-6, 0.35e-6]], np.c_[xs, top], [[0.9e-6, 0.35e-6]]])

    return {
        "all vertices": lambda: sh.Polygon(star),
        "fixed vertices": lambda: sh.Polygon(star, free=np.array([1, 0, 1, 1, 0, 1, 0, 1], bool)),
        "fixed coordinates": lambda: sh.Polygon(star, free=rng.uniform(size=star.shape) > 0.4),
        "mirror y": lambda: sh.Polygon(half, mirror="y", axis=0.35e-6),
        "mirror x": lambda: sh.Polygon(halfx, mirror="x", axis=0.5e-6),
        "spline, mirror y": lambda: sh.Polygon.from_function(spline, [0.5e-6, 0.6e-6, 0.45e-6, 0.55e-6], (0.4e-6, 0.68e-6), mirror="y", axis=0.35e-6),
        "width profile, linear": lambda: sh.LevelSet.width_profile(np.linspace(0, 1e-6, 6), [0.3e-6, 0.4e-6, 0.5e-6, 0.35e-6, 0.42e-6, 0.3e-6], (0.1e-6, 0.7e-6), center=0.35e-6, kind="linear"),
        "width profile, cubic": lambda: sh.LevelSet.width_profile(np.linspace(0, 1e-6, 5), [0.3e-6, 0.4e-6, 0.5e-6, 0.35e-6, 0.3e-6], (0.1e-6, 0.7e-6), free=[0, 1, 1, 1, 0], center=0.35e-6),
        "level set, disc, FD jacobian": lambda: sh.LevelSet(lambda p, X, Y: p[2] - np.hypot(X - p[0], Y - p[1]), [0.5e-6, 0.33e-6, 0.21e-6], ([0.3e-6, 0.2e-6, 0.1e-6], [0.7e-6, 0.5e-6, 0.3e-6])),
    }


@pytest.mark.parametrize("case", list(_shape_cases()))
def test_shape_density_transpose(case):
    """ShapeDesign.density_vjp (rasteriser transpose, mirror, masks, user
    map, normalisation) equals central differences of the density."""
    shape = _shape_cases()[case]()
    des = invdes.Design.effective_index([0, 0], [1 * um, 0.7 * um], 1.444, 2.85, shape=(26, 19))
    sd = sh.ShapeDesign(des, [shape])
    rng = np.random.default_rng(2)
    u = np.clip(sd.initial() + 0.01 * rng.standard_normal(sd.size), 0.01, 0.99)
    w = rng.standard_normal(des.shape)
    g = sd.density_vjp(u, w)
    gfd = _fd(lambda v: float(np.sum(w * sd.density(v))), u, 1e-6)
    assert np.abs(g).max() > 0
    assert np.allclose(g, gfd, rtol=1e-5, atol=1e-6 * np.abs(gfd).max()), np.abs(g - gfd).max() / np.abs(gfd).max()
    if isinstance(shape, sh.Polygon) and shape.mirror == "y":
        d = sd.density(u)
        assert np.allclose(d, d[:, ::-1], atol=1e-12)  # axis on the centre row


@pytest.mark.parametrize("kind", ["slab", "partial", "eim_partial", "projection"])
def test_shape_permittivity_transpose(kind):
    """Shapes -> density -> level fields -> eps: vjp equals central
    differences, one and two density fields, with and without the design's
    filter and projection."""
    mn, mx = [0.0, 0.0], [1 * um, 0.7 * um]
    kw = dict(shape=(21, 15))
    a = np.array([[0.05, 0.35], [0.2, 0.62], [0.5, 0.66], [0.8, 0.55], [0.95, 0.35]]) * 1e-6
    b = np.array([[0.15, 0.35], [0.3, 0.5], [0.6, 0.52], [0.85, 0.35]]) * 1e-6
    proj = kind == "projection"
    if kind == "slab":
        des = invdes.Design.slab(mn, mx, (0, 0.22 * um), 2.1, 12.1, **kw)
    elif kind == "partial":
        des = invdes.Design.partial_etch(mn, mx, (0, 0.07 * um, 0.22 * um), 2.1, 12.1, **kw)
    elif kind == "eim_partial":
        des = invdes.Design.effective_index(mn, mx, 1.444, 2.85, n_partial=2.2, **kw)
    else:
        des = invdes.Design.effective_index(mn, mx, 1.444, 2.85, filter_radius=0.12 * um, **kw)
    shapes = [sh.Polygon(a, mirror="y", axis=0.35e-6)]
    if des.fields == 2:
        shapes.append(sh.Polygon(b, mirror="y", axis=0.35e-6, field=1))
    sd = sh.ShapeDesign(des, shapes, projection=proj)
    rng = np.random.default_rng(3)
    u = np.clip(sd.initial() + 0.02 * rng.standard_normal(sd.size), 0.01, 0.99)
    ws = [rng.standard_normal(des.shape + (1,)) for _ in des.layers]
    beta = 3.0
    g = sd.vjp(u, beta, ws)
    gfd = _fd(lambda v: sum(float(np.sum(w * e)) for w, e in zip(ws, sd.eps(v, beta))), u, 1e-6)
    assert np.allclose(g, gfd, rtol=1e-5, atol=1e-6 * np.abs(gfd).max()), np.abs(g - gfd).max() / np.abs(gfd).max()
    if not proj:  # linear: eps runs from eps_lo to eps_hi with the density
        e = sd.eps(u)[0]
        assert e.min() >= des.layers[0].eps_base - 1e-12 and e.max() <= des.layers[0].eps_max + 1e-12


def test_shape_checks():
    des = invdes.Design.effective_index([0, 0], [1 * um, 0.7 * um], 1.444, 2.85, shape=(11, 8))
    bow = np.array([[0.2, 0.2], [0.8, 0.5], [0.8, 0.2], [0.2, 0.5]]) * 1e-6  # edges cross
    assert not sh.polygon_is_simple(bow)
    sd = sh.ShapeDesign(des, [sh.Polygon(bow)])
    with pytest.raises(ValueError, match="cross"):
        sd.density(sd.initial())
    sq = np.array([[0.2, 0.2], [0.6, 0.2], [0.6, 0.5], [0.2, 0.5]]) * 1e-6
    sd = sh.ShapeDesign(des, [sh.Polygon(sq), sh.Polygon(sq + 0.1e-6)])
    with pytest.raises(ValueError, match="overlap"):
        sd.density(sd.initial())
    with pytest.raises(ValueError, match="mirror line"):
        sh.Polygon(sq, mirror="y", axis=0.0)
    with pytest.raises(ValueError, match="field"):
        sh.ShapeDesign(des, [sh.Polygon(sq, field=1)])
    sd = sh.ShapeDesign(des, [sh.Polygon(sq, free=[1, 0, 0, 1])])
    assert sd.size == 4 and np.allclose(sd.params(sd.initial()), sq[[0, 3]].ravel())
    assert np.allclose(sd.lo, [0, 0, 0, 0]) and np.allclose(sd.hi, [1e-6, 0.7e-6, 1e-6, 0.7e-6])  # design rectangle
    assert np.allclose(sd.polygons(sd.initial())[0][1], sq)


# ----------------------------------------------------------------------------
# 2.5-D: a mirrored taper outline, d|S21|^2 / d(parameter) against FD
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def taper2d():
    n_si = eim.slab_neff(3.476, 220e-9, LAM, 1.444)
    n_bg = 1.444
    w1, w2 = 0.5 * um, 1.5 * um
    sim = invdes.planar_simulation([0, -1.6 * um], [4.4 * um, 1.6 * um], LAM, n_bg, n_si, steps_per_wavelength=14, run_time=400e-15, shutoff=1e-7)
    sim.add_medium(Medium("core", n=n_si))
    sim.add_structure(Box([-5 * um, -w1 / 2, -1], [1.4 * um, w1 / 2, 1]), "core")
    sim.add_structure(Box([3.0 * um, -w2 / 2, -1], [9 * um, w2 / 2, 1]), "core")

    def port(name, x, d):
        return invdes.port_mode(sim, name, 0, x, [x, -1.2 * um, 0], [x, 1.2 * um, 1 * um], F0, d, "te")

    des = invdes.Design.effective_index([1.2 * um, -1.0 * um], [3.2 * um, 1.0 * um], n_bg, n_si, pixel=40e-9)
    return dict(sim=sim, ports=(port("in", 0.8 * um, "+"), port("out", 3.6 * um, "-")), des=des)


def _check_fd(sd, inv, params, h, tol, label, extrapolate=False):
    """Adjoint d|S21|^2/dp (physical units) against central differences of
    the simulated objective, for the parameter indices ``params``. With
    ``extrapolate``, ``2 FD(h/2) - FD(h)``: a vertex on a pixel boundary has
    a jump in the second derivative of the area fraction there, and the
    central difference across it is off by O(h) instead of O(h^2)."""
    u = sd.initial()
    val, g, _ = inv.value_and_grad(u, 1.0)
    ad = (g / (sd.hi - sd.lo))[params]
    P = sd.params(u)

    def f(p):
        return invdes.mean_coupling(inv.s_matrix(sd.normalize(p), 1.0))[0]

    def cd(h):
        out = np.empty(len(params))
        for i, k in enumerate(params):
            e = np.zeros(sd.size)
            e[k] = h
            out[i] = (f(P + e) - f(P - e)) / (2 * h)
        return out

    fd = 2 * cd(h / 2) - cd(h) if extrapolate else cd(h)
    err = np.abs(ad - fd) / np.abs(fd).max()
    print(f"{label}: |S21|^2 = {val:.4f}; per um: adjoint {np.round(ad * 1e-6, 5)}, FD {np.round(fd * 1e-6, 5)}; |adj - fd| / max|fd| = {np.round(err, 4)}")
    assert np.all(err < tol), err
    return ad, fd


@pytest.mark.parametrize("engine", ["python", "native", "reverse"])
def test_planar_polygon_gradient(taper2d, engine):
    """2.5-D taper 0.5 -> 1.5 um whose outline is a polygon mirrored about
    y = 0 with three free vertices (six parameters, the end vertices fixed):
    d|S21|^2 / d(coordinate) against central differences (step 2 nm, pixel
    and cell 40 nm) for every parameter."""
    chain = np.array([[1.0, 0], [1.0, 0.25], [1.6, 0.33], [2.0, 0.62], [2.5, 0.55], [3.4, 0.75], [3.4, 0]]) * um
    poly = sh.Polygon(chain, free=np.array([0, 0, 1, 1, 1, 0, 0], bool), mirror="y")
    sd = sh.ShapeDesign(taper2d["des"], [poly])
    assert sd.size == 6
    prob = invdes.SMatrixProblem(taper2d["sim"], sd.grids(), [taper2d["ports"][0]], [taper2d["ports"][1]], [F0], devices=DEVICE, engine=engine)
    inv = invdes.InverseDesign(prob, sd, objective=invdes.mean_coupling)
    _check_fd(sd, inv, list(range(6)), 2e-9, 0.02, f"2.5-D polygon, {engine}")


def test_planar_level_set_gradient(taper2d):
    """The same taper as a cubic width profile through five stations (the
    three inner widths free): d|S21|^2 / d(width) against central differences."""
    xs = np.linspace(1.2 * um, 3.2 * um, 5)
    lset = sh.LevelSet.width_profile(xs, [0.5 * um, 0.75 * um, 1.25 * um, 1.1 * um, 1.5 * um], (0.3 * um, 1.9 * um), free=[0, 1, 1, 1, 0])
    sd = sh.ShapeDesign(taper2d["des"], [lset])
    prob = invdes.SMatrixProblem(taper2d["sim"], sd.grids(), [taper2d["ports"][0]], [taper2d["ports"][1]], [F0], devices=DEVICE)
    inv = invdes.InverseDesign(prob, sd, objective=invdes.mean_coupling)
    _check_fd(sd, inv, [0, 1, 2], 2e-9, 0.02, "2.5-D width profile")


# ----------------------------------------------------------------------------
# 3-D: a bulge on a silicon-nitride strip, one level and two levels
# ----------------------------------------------------------------------------


def _scene_3d():
    sim = ls.Simulation(min=[0, -1.2 * um, -0.9 * um], max=[4 * um, 1.2 * um, 0.9 * um], mesh=ls.UniformMesh(40e-9), run_time=220e-15, background="ox")
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_medium(Medium("sin", n=2.0))
    sim.add_structure(Box([-1, -0.4 * um, -0.2 * um], [5 * um, 0.4 * um, 0.2 * um]), "sin")
    pmin, pmax = [0.8 * um, -0.9 * um, -0.7 * um], [0.8 * um, 0.9 * um, 0.7 * um]
    mode = ls.solve_modes(sim, 0, 0.8 * um, pmin, pmax, F0, num_modes=1)[0]
    p_in = invdes.Port("in", 0, 0.8 * um, pmin, pmax, [mode.profile], "+")
    p_out = invdes.Port("out", 0, 3.2 * um, [3.2 * um, -0.9 * um, -0.7 * um], [3.2 * um, 0.9 * um, 0.7 * um], [mode.profile], "-")
    return sim, p_in, p_out


# The 3-D differences (up to 3.5 % of the largest derivative) do not shrink
# with the FD step (1 - 5 nm), the run time (220 - 500 fs, shutoff 1e-8),
# the cell (40 - 20 nm) or the design pixel (40 - 20 nm); they are the error
# of the continuous adjoint formula, summed along the boundary a vertex moves
# ("Accuracy at interfaces" in docs/modules/autodiff.md), so the 3-D
# tolerance was 4 %. Since both S-matrix
# engines scale the port profiles to unit power in the mode overlap's measure
# (a 3-D solver profile carries 0.9953 W there), the 3-D gradients are 0.47 %
# larger and the partial-etch case reaches 4.0 % (3.5 % before): 4.5 %.
THREE_D_TOL = 0.045
# engine="reverse" differentiates the discrete scheme itself. The second
# upper vertex of the partial etch (parameter 7) lies on a pixel boundary
# (y = 0.42 um): central differences there are off by O(h) (0.6, 1.2, 2.1,
# 3.3 % for h = 1, 2, 4, 8 nm, the same for both engines), so the reverse
# check extrapolates 2 FD(1 nm) - FD(2 nm). Measured (RTX PRO 6000): slab
# 0.03, 0.03, 0.00 %, partial etch 0.02, 0.10, 0.03 %.
REVERSE_TOL = 0.003


@pytest.mark.parametrize("engine", ["python", "reverse"])
@pytest.mark.parametrize("kind", ["slab", "partial etch"])
def test_3d_polygon_gradient(kind, engine):
    """3-D: a mirrored polygon over a 0.8 x 0.4 um SiN strip (full etch), or
    two polygons on the two levels of a partial etch (the lower level wider);
    d|S21|^2 / d(vertex y) against central differences (step 2 nm, cell
    and pixel 40 nm) for three vertex parameters. Measured with
    engine="python": slab 0.2, 1.1, 2.3 %, partial etch 1.5, 1.2, 4.0 % of
    the largest derivative (see THREE_D_TOL); with engine="reverse" and
    extrapolated differences: slab 0.03, 0.03, 0.00 %, partial etch 0.02,
    0.10, 0.03 % (see REVERSE_TOL)."""
    sim, p_in, p_out = _scene_3d()
    mn, mx = [1.4 * um, -0.8 * um], [2.6 * um, 0.8 * um]
    chain = np.array([[1.3, 0], [1.3, 0.4], [1.75, 0.5], [2.05, 0.68], [2.3, 0.47], [2.7, 0.4], [2.7, 0]]) * um
    free = np.array([0, 0, 1, 1, 1, 0, 0], bool)
    if kind == "slab":
        des = invdes.Design.slab(mn, mx, (-0.2 * um, 0.2 * um), 1.444**2, 4.0, pixel=40e-9)
        shapes = [sh.Polygon(chain, free=free, mirror="y")]
        params = [1, 3, 5]  # y of the three free vertices
    else:
        des = invdes.Design.partial_etch(mn, mx, (-0.2 * um, 0.0, 0.2 * um), 1.444**2, 4.0, pixel=40e-9)
        lower = chain.copy()
        lower[1:-1, 1] += 0.12 * um
        upper = chain.copy()
        upper[2:-2, 1] -= 0.08 * um
        shapes = [sh.Polygon(lower, free=free, mirror="y", field=0), sh.Polygon(upper, free=free, mirror="y", field=1)]
        params = [3, 7, 9]  # y of the middle lower vertex, y of two upper vertices
    sd = sh.ShapeDesign(des, shapes)
    prob = invdes.SMatrixProblem(sim, sd.grids(), [p_in], [p_out], [F0], devices=DEVICE, engine=engine)
    inv = invdes.InverseDesign(prob, sd, objective=invdes.mean_coupling)
    if engine == "python":
        _check_fd(sd, inv, params, 2e-9, THREE_D_TOL, f"3-D {kind}, {engine}")
    else:
        _check_fd(sd, inv, params, 2e-9, REVERSE_TOL, f"3-D {kind}, {engine}", extrapolate=True)


# ----------------------------------------------------------------------------
# optimisation: width profile of a short taper 0.5 -> 2.5 um
# ----------------------------------------------------------------------------


def test_taper_width_optimisation():
    """2.5-D taper from 0.5 to 2.5 um over 2 um: seven free widths of a
    cubic width profile, bounded below by 0.3 um (minimum feature) and above
    by 3 um, 30 Adam iterations on the mean |S21|^2 over three wavelengths.
    The linear taper transmits about 0.76; the optimised one about 0.95."""
    import time

    n_si = eim.slab_neff(3.476, 220e-9, LAM, 1.444)
    n_bg = 1.444
    w1, w2, x0, x1 = 0.5 * um, 2.5 * um, 1.5 * um, 3.5 * um
    sim = invdes.planar_simulation([0, -2.2 * um], [x1 + 1.5 * um, 2.2 * um], LAM, n_bg, n_si, steps_per_wavelength=14, run_time=600e-15, shutoff=1e-6)
    sim.add_medium(Medium("core", n=n_si))
    sim.add_structure(Box([-5 * um, -w1 / 2, -1], [x0 + 0.1 * um, w1 / 2, 1]), "core")
    sim.add_structure(Box([x1 - 0.1 * um, -w2 / 2, -1], [20 * um, w2 / 2, 1]), "core")
    freqs = C0 / (np.array([1.53, 1.55, 1.57]) * um)
    pin = invdes.port_mode(sim, "in", 0, 0.8 * um, [0.8 * um, -1.2 * um, 0], [0.8 * um, 1.2 * um, 1 * um], F0, "+", "te")
    pout = invdes.port_mode(sim, "out", 0, x1 + 0.8 * um, [x1 + 0.8 * um, -2.0 * um, 0], [x1 + 0.8 * um, 2.0 * um, 1 * um], F0, "-", "te")
    des = invdes.Design.effective_index([x0, -1.6 * um], [x1, 1.6 * um], n_bg, n_si, pixel=40e-9)
    xs = np.linspace(x0, x1, 9)
    wmin = 0.3 * um
    profile = sh.LevelSet.width_profile(xs, np.linspace(w1, w2, 9), bounds=(wmin, 3.0 * um), free=[0, 1, 1, 1, 1, 1, 1, 1, 0])
    sd = sh.ShapeDesign(des, [profile])
    prob = invdes.SMatrixProblem(sim, sd.grids(), [pin], [pout], freqs, devices=DEVICE)
    inv = invdes.InverseDesign(prob, sd, objective=invdes.mean_coupling)
    t0 = time.time()
    u, hist = invdes.optimize(inv, sd.initial(), 30, lr=0.02, verbose=False)
    fom = hist["fom"]
    widths = profile.all_widths(sd.params(u))
    print(f"taper: |S21|^2 {fom[0]:.4f} -> {fom[-1]:.4f} (best {hist['best_fom']:.4f}) in {len(fom)} iterations, {time.time() - t0:.1f} s; widths (um) {np.round(widths * 1e6, 3)}")
    print("history", np.round(fom, 4))
    assert fom[-1] > fom[0] + 0.12 and hist["best_fom"] > 0.9
    assert np.min(np.diff(fom)) > -0.005  # monotone up to small Adam overshoots
    assert widths.min() >= wmin - 1e-12
    # the shape is what the density says: a strip of the profile's width
    from scipy.interpolate import CubicSpline

    d = sd.density(u)
    w_grid = CubicSpline(xs, widths, bc_type="natural")(des.coordinates()[0])
    assert np.allclose(d.sum(axis=1) * des.pixel[1], w_grid, rtol=1e-3)
