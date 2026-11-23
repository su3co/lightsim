"""2.5-D test bed for the scaling study: meshes of free-form n-port blocks.

A mesh of N ports at pitch ``PITCH`` is built from D layers of n-port
free-form blocks in a brick-wall arrangement: even layers hold blocks on the
ports [0, n), [n, 2n), ...; odd layers are shifted by n/2 ports and the n/2
outermost ports on either side pass straight through. Consecutive layers are
joined by straight single-mode waveguides of length 2 * stub; the port
planes sit in the middle of these waveguides, so a block and its neighbour
share a reference plane.

Everything is laid out on a uniform grid of cell DX whose nodes are at
integer multiples of DX in every simulation, so a port window covers the
same nodes, with the same mode profile, in a block simulation and in the
monolithic simulation of the whole mesh.

Used by the measurements in this folder; the solver itself is
``lightsim.invdes.composite``.
"""

from __future__ import annotations

import os

import numpy as np

import lightsim as ls
from lightsim import eim, invdes
from lightsim.spec import C0, Box, Medium

um = 1e-6
N_SI, N_OX, T_SI = 3.476, 1.444, 0.22
LAM0 = 1.55
DX = 1.0 / 30.0  # um: 30 cells per port pitch, lambda/(n_slab * 16.3)
PITCH = float(os.environ.get("MESH_PITCH", 1.5))  # um
W_WG = 0.5  # um
GAP_Y = float(os.environ.get("MESH_GAP", 1.0))  # um of cladding between the design regions of neighbouring blocks
LB = 4.0  # um, design length along x
STUB = 1.2  # um, design edge to port plane
MARGIN = 1.0  # um, port plane to the domain edge (PML inside)
MARGIN_Y = 1.0
PIXEL = 2 * DX
FILTER_RADIUS = 0.08


def snap(v):
    return float(np.round(v / DX) * DX)


def slab_index(lam_um=LAM0):
    return float(eim.slab_neff(N_SI, T_SI * um, lam_um * um, N_OX))


N_SLAB = slab_index()


def port_y(N):
    return [snap((j - (N - 1) / 2) * PITCH) for j in range(N)]


def layer_blocks(N, n, layer):
    """Blocks of one layer: list of (first port, n_ports, designed)."""
    out = []
    if layer % 2 == 0:
        for b in range(N // n):
            out.append((b * n, n, True))
    else:
        h = n // 2
        out.append((0, h, False))
        for b in range(N // n - 1):
            out.append((h + b * n, n, True))
        out.append((N - h, h, False))
    return out


def layer_x(layer, stub=STUB):
    """x of the centre of the design region of a layer (um)."""
    return snap(layer * (LB + 2 * stub))


def block_design(n, stub=STUB):
    """The Design of an n-port block centred at the origin (um -> m inside)."""
    h = (n * PITCH - GAP_Y) / 2
    mn, mx = [-LB / 2 * um, -h * um], [LB / 2 * um, h * um]
    nx = int(round(LB / PIXEL)) + 1
    ny = int(round(2 * h / PIXEL)) + 1
    xs = np.linspace(-LB / 2, LB / 2, nx)
    ys = np.linspace(-h, h, ny)
    solid = np.zeros((nx, ny), bool)
    for y in [(j - (n - 1) / 2) * PITCH for j in range(n)]:
        rows = np.abs(ys - y) <= W_WG / 2 + 1e-9
        solid[:2, rows] = True
        solid[-2:, rows] = True
    del xs
    return invdes.Design.effective_index(mn, mx, N_OX, N_SLAB, shape=(nx, ny), filter_radius=FILTER_RADIUS * um, fixed_solid=solid)


def _sim(x0, x1, y0, y1, run_time):
    lo = [snap(x0) * um, snap(y0) * um]
    hi = [snap(x1) * um, snap(y1) * um]
    sim = invdes.planar_simulation(lo, hi, LAM0 * um, N_OX, N_SLAB, steps_per_wavelength=LAM0 / (N_SLAB * DX), run_time=run_time, shutoff=1e-6)
    sim.add_medium(Medium("eim_si", n=N_SLAB))
    return sim


_PROFILE = {}


def wg_profile(freq):
    """Mode profile of the single-mode waveguide on a port window of the
    standard size, solved once (in a reference simulation) and reused, so
    every port uses the same array with the same sign."""
    key = round(freq / 1e9)
    if key not in _PROFILE:
        sim = _sim(-2.0, 2.0, -2.0, 2.0, 1e-13)
        sim.add_structure(Box([-1.0, -W_WG / 2 * um, -1.0], [1.0, W_WG / 2 * um, 1.0]), "eim_si")
        hw = PITCH / 2
        p = invdes.port_mode(sim, "ref", 0, 0.0, [0.0, -hw * um, 0.0], [0.0, hw * um, sim.max[2]], freq, "+", "te")
        _PROFILE[key] = (p.profiles[0], p.neff[0])
    return _PROFILE[key]


def make_port(name, x, y, direction, freq):
    prof, neff = wg_profile(freq)
    hw = PITCH / 2
    p = invdes.Port(name, 0, snap(x) * um, [snap(x) * um, (snap(y) - hw) * um, 0.0], [snap(x) * um, (snap(y) + hw) * um, 1 * um], [prof], direction)
    p.neff = [neff]
    return p


def block_problem(n_ports, designed, freqs, devices="cpu", stub=STUB, run_time=None, engine=None, sides="both"):
    """S-matrix problem of one block centred at the origin, with all 2 n
    ports (left ports 0..n-1 pointing +x, right ports n..2n-1 pointing -x)
    as inputs and outputs. ``sides="ff"`` keeps only the left ports as
    inputs and the right ports as outputs (feed-forward model).

    Returns ``(problem, design, in_ports, out_ports)`` where the last two
    are the local port numbers of the problem's inputs and outputs."""
    f0 = float(np.mean(freqs))
    h = (n_ports * PITCH) / 2
    xp = LB / 2 + stub
    L = 2 * (xp + MARGIN)
    if run_time is None:
        run_time = 12 * N_SLAB * L * um / C0 + 0.6e-12
    sim = _sim(-xp - MARGIN, xp + MARGIN, -h - MARGIN_Y, h + MARGIN_Y, run_time)
    ys = [(j - (n_ports - 1) / 2) * PITCH for j in range(n_ports)]
    # strips run through the block (the design grid is painted over them), as
    # in the monolithic layout: no waveguide end faces at the design edge
    for y in ys:
        sim.add_structure(Box([-1.0, (snap(y) - W_WG / 2) * um, -1.0], [1.0, (snap(y) + W_WG / 2) * um, 1.0]), "eim_si")
    left = [make_port(f"L{j}", -xp, y, "+", f0) for j, y in enumerate(ys)]
    right = [make_port(f"R{j}", xp, y, "-", f0) for j, y in enumerate(ys)]
    ports = left + right
    des = block_design(n_ports, stub) if designed else None
    grids = des.grids() if designed else []
    if sides == "ff":
        ins, outs = list(range(n_ports)), list(range(n_ports, 2 * n_ports))
    else:
        ins = outs = list(range(2 * n_ports))
    prob = invdes.SMatrixProblem(sim, grids, [ports[i] for i in ins], [ports[i] for i in outs], freqs, devices=devices, engine=engine)
    return prob, des, ins, outs


def mesh_layout(N, n, D, stub=STUB):
    """Blocks of the mesh: list of dicts with layer, first port, n_ports,
    designed, centre (x, y) in um."""
    ys = port_y(N)
    out = []
    for layer in range(D):
        for first, k, designed in layer_blocks(N, n, layer):
            yc = 0.5 * (ys[first] + ys[first + k - 1])
            out.append(dict(layer=layer, first=first, n=k, designed=designed, x=layer_x(layer, stub), y=yc))
    return out


def mesh_network(N, n, D):
    """lightsim.invdes.Network of the mesh: external ports = the N left
    inputs of layer 0, then the N right outputs of layer D-1."""
    blocks = mesh_layout(N, n, D)
    owner = {}  # (layer, port index) -> (block, local index)
    for b, bl in enumerate(blocks):
        for j in range(bl["n"]):
            owner[(bl["layer"], bl["first"] + j)] = (b, j)
    conns = []
    for layer in range(D - 1):
        for q in range(N):
            b, j = owner[(layer, q)]
            b2, j2 = owner[(layer + 1, q)]
            conns.append(((b, blocks[b]["n"] + j), (b2, j2)))
    ext = [owner[(0, q)] for q in range(N)]
    ext = [(b, j) for b, j in ext] + [(owner[(D - 1, q)][0], blocks[owner[(D - 1, q)][0]]["n"] + owner[(D - 1, q)][1]) for q in range(N)]
    sides = [["in"] * bl["n"] + ["out"] * bl["n"] for bl in blocks]
    return invdes.Network([2 * bl["n"] for bl in blocks], conns, ext, sides=sides), blocks


def monolithic_problem(N, n, D, freqs, devices="cpu", stub=STUB, run_time=None, engine=None, reflections=False):
    """S-matrix problem of the whole mesh in one simulation. Inputs: the N
    left ports; outputs: the N right ports (and with ``reflections`` the N
    left ports as well). Its design grids are those of the designed blocks
    in the order of :func:`mesh_layout`."""
    f0 = float(np.mean(freqs))
    blocks = mesh_layout(N, n, D, stub)
    ys = port_y(N)
    x_in = layer_x(0, stub) - LB / 2 - stub
    x_out = layer_x(D - 1, stub) + LB / 2 + stub
    h = N * PITCH / 2
    L = (x_out - x_in) + 2 * MARGIN
    if run_time is None:
        run_time = 12 * N_SLAB * L * um / C0 + 0.6e-12 + 4 * D * 0.15e-12
    sim = _sim(x_in - MARGIN, x_out + MARGIN, -h - MARGIN_Y, h + MARGIN_Y, run_time)
    # waveguides everywhere outside the design regions: one long strip per port, designs painted over
    for y in ys:
        sim.add_structure(Box([-1.0, (snap(y) - W_WG / 2) * um, -1.0], [1.0, (snap(y) + W_WG / 2) * um, 1.0]), "eim_si")
    grids = []
    designs = []
    for b, bl in enumerate(blocks):
        if not bl["designed"]:
            continue
        d = block_design(bl["n"], stub)
        # blank the strips under the design (the design grid is painted last and covers them)
        for g in d.grids():
            g.min = [g.min[0] + bl["x"] * um, g.min[1] + bl["y"] * um, g.min[2]]
            g.max = [g.max[0] + bl["x"] * um, g.max[1] + bl["y"] * um, g.max[2]]
            g.name = f"blk{b}"
            g.medium = dict(g.medium)
            g.medium["name"] = f"invdes_blk{b}"
            grids.append(g)
        designs.append(d)
    ins = [make_port(f"in{q}", x_in, y, "+", f0) for q, y in enumerate(ys)]
    outs = [make_port(f"out{q}", x_out, y, "-", f0) for q, y in enumerate(ys)]
    prob = invdes.SMatrixProblem(sim, grids, ins, outs + (ins if reflections else []), freqs, devices=devices, engine=engine)
    return prob, designs


def dft_target(N):
    m, k = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.exp(-2j * np.pi * m * k / N) / np.sqrt(N)


def layer_problem(N, n, layer, freqs, devices="cpu", stub=STUB, colours=2, run_time=None):
    """All designed blocks of one mesh layer in one strip simulation
    (lightsim.invdes.LayerProblem), feed-forward ports (left in, right out).
    Returns (problem, member block indices into mesh_layout order)."""
    f0 = float(np.mean(freqs))
    blocks = mesh_layout(N, n, layer + 1, stub)
    ys = port_y(N)
    xc = layer_x(layer, stub)
    xp = LB / 2 + stub
    h = N * PITCH / 2
    if run_time is None:
        run_time = 12 * N_SLAB * 2 * (xp + MARGIN) * um / C0 + 0.6e-12
    sim = _sim(xc - xp - MARGIN, xc + xp + MARGIN, -h - MARGIN_Y, h + MARGIN_Y, run_time)
    for y in ys:
        sim.add_structure(Box([-1.0, (snap(y) - W_WG / 2) * um, -1.0], [1.0, (snap(y) + W_WG / 2) * um, 1.0]), "eim_si")
    members, idx = [], []
    k = 0
    for b, bl in enumerate(blocks):
        if bl["layer"] != layer or not bl["designed"]:
            continue
        d = block_design(bl["n"], stub)
        grids = []
        for g in d.grids():
            g.min = [g.min[0] + bl["x"] * um, g.min[1] + bl["y"] * um, g.min[2]]
            g.max = [g.max[0] + bl["x"] * um, g.max[1] + bl["y"] * um, g.max[2]]
            g.name = f"blk{b}"
            g.medium = dict(g.medium)
            g.medium["name"] = f"invdes_blk{b}"
            grids.append(g)
        py = [ys[bl["first"] + j] for j in range(bl["n"])]
        ports = [make_port(f"b{b}L{j}", xc - xp, y, "+", f0) for j, y in enumerate(py)] + [make_port(f"b{b}R{j}", xc + xp, y, "-", f0) for j, y in enumerate(py)]
        members.append({"grids": grids, "ports": ports, "in": list(range(bl["n"])), "out": list(range(bl["n"], 2 * bl["n"])), "colour": k % colours})
        idx.append(b)
        k += 1
    from lightsim import invdes as _inv

    return _inv.LayerProblem(sim, members, freqs, devices=devices), idx
