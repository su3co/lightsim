"""Tool-neutral input and output: SParameters writers/readers, the GDSII
reader, port and technology files, and the layout entry points.

The self round-trips and the GDSII reader on hand-written files always run;
the checks against independent readers and writers (scikit-rf, SAX, gdstk,
gdsfactory) run where those packages are installed.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from lightsim.sparameters import C0, SParameters

FIX = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------
# S-parameters
# ----------------------------------------------------------------------------


def delay_line(tau=1e-12, n=11):
    """Two ports, no reflection, S21 = S12 = exp(+i w tau): lightsim's convention."""
    f = np.linspace(185e12, 200e12, n)
    s = np.zeros((n, 2, 2), complex)
    s[:, 1, 0] = s[:, 0, 1] = 0.9 * np.exp(1j * 2 * np.pi * f * tau)
    s[:, 0, 0] = 0.05 * np.exp(0.3j)
    return SParameters(f, s, ["o1", "o2"])


def three_port(n=7):
    rng = np.random.default_rng(1)
    f = np.linspace(190e12, 195e12, n)
    s = rng.normal(size=(n, 3, 3)) + 1j * rng.normal(size=(n, 3, 3))
    return SParameters(f, 0.3 * s, [("o1", 0), ("o2", 0), ("o3", 1)])


@pytest.mark.parametrize("version", ["1.1", "2.0"])
@pytest.mark.parametrize("fmt", ["RI", "MA", "DB"])
@pytest.mark.parametrize("unit", ["Hz", "GHz"])
def test_touchstone_self_roundtrip(tmp_path, version, fmt, unit):
    for sp in (delay_line(), three_port()):
        p = sp.to_touchstone(tmp_path / "x", version=version, fmt=fmt, freq_unit=unit)
        assert p.suffix == f".s{sp.nports}p"
        back = SParameters.from_touchstone(p)
        assert back.ports == sp.ports and np.allclose(back.freqs, sp.freqs, rtol=1e-11)
        assert np.allclose(back.s, sp.s, atol=1e-9)
        # without the sidecar the names come from the comments
        Path(str(p) + ".json").unlink()
        assert SParameters.from_touchstone(p).allclose(sp, atol=1e-9)


def test_touchstone_is_conjugated(tmp_path):
    sp = delay_line()
    p = sp.to_touchstone(tmp_path / "d.s2p", fmt="RI", freq_unit="Hz")
    rows = [l.split() for l in p.read_text().splitlines() if l and l[0].isdigit()]
    s21_file = complex(float(rows[3][3]), float(rows[3][4]))  # Touchstone 1 order: S11 S21 S12 S22
    assert abs(s21_file - np.conj(sp["o2", "o1"][3])) < 1e-9
    raw = sp.to_touchstone(tmp_path / "raw.s2p", conjugate=False)
    assert SParameters.from_touchstone(raw).allclose(sp, atol=1e-9)


def test_touchstone_delay_with_scikit_rf(tmp_path):
    """scikit-rf (exp(+jwt)) sees a positive group delay for a delay line."""
    rf = pytest.importorskip("skrf")
    tau = 1.3e-12
    sp = delay_line(tau, n=41)
    net = rf.Network(str(sp.to_touchstone(tmp_path / "delay.s2p")))
    assert np.allclose(net.f, sp.freqs)
    gd = net.s21.group_delay.flatten()  # -d(phase)/d(omega)
    assert np.allclose(gd[2:-2], tau, rtol=1e-3)
    # and a file scikit-rf writes reads back into lightsim's convention
    line = rf.Network(frequency=rf.Frequency.from_f(sp.freqs, unit="hz"), s=np.conj(sp.s))
    line.write_touchstone(str(tmp_path / "skrf"))
    back = SParameters.from_touchstone(tmp_path / "skrf.s2p", conjugate=True)
    assert np.allclose(back.s, sp.s, atol=1e-8)


def test_lumerical_dat_roundtrip_and_sax(tmp_path):
    sp = three_port()
    p = sp.to_lumerical_dat(tmp_path / "x.dat")
    back = SParameters.from_lumerical_dat(p)
    assert back.ports == sp.ports and np.allclose(back.s, sp.s, atol=1e-9)
    assert "('o3','TE',2,'o1',1,'transmission')" in p.read_text()  # mode 1 written as INTERCONNECT's 2
    sax = pytest.importorskip("sax")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = sax.parsers.parse_lumerical_dat(str(p))
    row = df[(df.port_out == "o2") & (df.port_in == "o1")].sort_values("f")
    assert np.allclose(row.amp.to_numpy() * np.exp(1j * row.phi.to_numpy()), sp["o2@0", "o1@0"], atol=1e-9)


def test_siepic_dat_fixture(tmp_path):
    """INTERCONNECT text written by SiEPIC: parsed, round-tripped, same convention as lightsim."""
    sp = SParameters.from_lumerical_dat(FIX / "siepic" / "halfring_gap100nm_r5um.dat")
    assert sp.nports == 4 and sp.freqs.size == 101 and sp.computed.all()
    through = sp["port 3", "port 1"]
    assert np.mean(np.abs(through)) > 0.95
    # the through port's phase grows with frequency, at the delay of ~10 um of waveguide: exp(-i w t)
    slope = np.polyfit(2 * np.pi * sp.freqs, np.unwrap(np.angle(through)), 1)[0]
    assert 0.1e-12 < slope < 0.2e-12
    back = SParameters.from_lumerical_dat(sp.to_lumerical_dat(tmp_path / "again.dat"))
    assert back.allclose(sp, atol=1e-9)
    t = SParameters.from_touchstone(sp.to_touchstone(tmp_path / "ring"))
    assert t.allclose(sp, atol=1e-9)


def test_dict_npz_and_sax_model(tmp_path):
    sp = three_port()
    d = sp.to_dict()
    assert "o3@1,o1@0" in d and np.all(np.diff(d["wavelengths"]) > 0) and d["wavelengths"][0] > 1
    assert SParameters.from_dict(d).allclose(sp)
    assert SParameters.from_npz(sp.to_npz(tmp_path / "x.npz")).allclose(sp)
    model = sp.to_sax()
    wl = sp.wavelengths[3] * 1e6
    s = model(wl=wl)
    assert abs(complex(np.asarray(s[("o2@0", "o1@0")])) - sp["o2", "o1"][3]) < 1e-9
    # a lightsim.sparams-style dict (wavelengths in metres)
    m = SParameters.from_dict({"wavelengths": np.array([1.5e-6, 1.6e-6]), "o2@0,o1@0": np.array([1, 1j]), "mode_neff": {}})
    assert np.allclose(m.wavelengths, [1.6e-6, 1.5e-6]) and not m.computed[0, 0]


# ----------------------------------------------------------------------------
# GDSII
# ----------------------------------------------------------------------------


def _real8(x):
    if x == 0:
        return bytes(8)
    e, m = 64, abs(x)
    while m >= 1:
        m /= 16
        e += 1
    while m < 1 / 16:
        m *= 16
        e -= 1
    mant = round(m * 2**56)
    return bytes([e | (0x80 if x < 0 else 0)]) + mant.to_bytes(7, "big")


def _rec(rt, dt, payload=b""):
    return struct.pack(">HBB", len(payload) + 4, rt, dt) + payload


def _xy(pts):
    return b"".join(struct.pack(">ii", int(x), int(y)) for x, y in pts)


def write_small_gds(path):
    """A hand-written library: BOX, PATH type 4, TEXT, SREF with mag 2 and 45 deg, AREF with reflection."""
    d = _rec(0, 2, struct.pack(">h", 600)) + _rec(1, 2, bytes(24)) + _rec(2, 6, b"LIB\0") + _rec(3, 5, _real8(1e-3) + _real8(1e-9))
    d += _rec(5, 2, bytes(24)) + _rec(6, 6, b"leaf")
    d += _rec(0x2D, 0) + _rec(0x0D, 2, struct.pack(">h", 5)) + _rec(0x2E, 2, struct.pack(">h", 0)) + _rec(0x10, 3, _xy([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)])) + _rec(0x11, 0)
    d += _rec(0x09, 0) + _rec(0x0D, 2, struct.pack(">h", 6)) + _rec(0x0E, 2, struct.pack(">h", 0)) + _rec(0x21, 2, struct.pack(">h", 4)) + _rec(0x0F, 3, struct.pack(">i", 20))
    d += _rec(0x30, 3, struct.pack(">i", 5)) + _rec(0x31, 3, struct.pack(">i", 15)) + _rec(0x10, 3, _xy([(0, 0), (200, 0)])) + _rec(0x11, 0)
    d += _rec(0x0C, 0) + _rec(0x0D, 2, struct.pack(">h", 10)) + _rec(0x16, 2, struct.pack(">h", 0)) + _rec(0x10, 3, _xy([(50, 25)])) + _rec(0x19, 6, b"tag\0") + _rec(0x11, 0)
    d += _rec(7, 0)
    d += _rec(5, 2, bytes(24)) + _rec(6, 6, b"top\0")
    d += _rec(0x0A, 0) + _rec(0x12, 6, b"leaf") + _rec(0x1A, 1, struct.pack(">H", 0)) + _rec(0x1B, 5, _real8(2.0)) + _rec(0x1C, 5, _real8(45.0)) + _rec(0x10, 3, _xy([(1000, 0)])) + _rec(0x11, 0)
    d += _rec(0x0B, 0) + _rec(0x12, 6, b"leaf") + _rec(0x1A, 1, struct.pack(">H", 0x8000)) + _rec(0x13, 2, struct.pack(">hh", 3, 2)) + _rec(0x10, 3, _xy([(0, 5000), (900, 5000), (0, 5600)])) + _rec(0x11, 0)
    d += _rec(7, 0) + _rec(4, 0)
    Path(path).write_bytes(d)


def test_gds_reader_hand_written(tmp_path):
    from lightsim.layout import read_gds

    write_small_gds(tmp_path / "s.gds")
    lay = read_gds(tmp_path / "s.gds")
    assert lay.cell == "top" and lay.cells == ["leaf", "top"] and lay.top_cells == ["top"]
    assert lay.database_unit_m == pytest.approx(1e-9) and lay.user_unit_m == pytest.approx(1e-6)
    boxes = lay.layer((5, 0))
    assert len(boxes) == 1 + 3 * 2
    # SREF: x2 then 45 deg, at (1000, 0) dbu
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    expect = [(1000 + 2 * (c * x - s * y), 2 * (s * x + c * y)) for x, y in [(0, 0), (100, 0), (100, 50), (0, 50)]]
    assert np.allclose(boxes[0] / 1e-9, expect, atol=1e-6)
    # AREF reflected about x: the box spans y = -50..0 relative to each lattice point
    last = boxes[-1] / 1e-9  # column 2, row 1: (600, 5300)
    assert np.allclose(sorted(map(tuple, last)), sorted([(600, 5300), (700, 5300), (700, 5250), (600, 5250)]), atol=1e-6)
    path = lay.layer((6, 0))[0] / 1e-9  # pathtype 4: extended 5 at the start, 15 at the end; x2 and rotated in the SREF
    xs = [(p[0] - 1000) * c + p[1] * s for p in path]  # back along the rotated axis
    assert min(xs) / 2 == pytest.approx(-5, abs=1e-6) and max(xs) / 2 == pytest.approx(215, abs=1e-6)
    assert [l[0] for l in lay.labels] == ["tag"] * 7
    assert lay.paths[0][2] == pytest.approx(40e-9)  # the path's width scaled by the magnification


def _canon(poly, nm=1e-9):
    """A polygon as a hashable set of vertices rounded to 1 nm (start and orientation free)."""
    q = np.round(np.asarray(poly) / nm).astype(np.int64)
    return frozenset(map(tuple, q))


def _seg_dist(points, poly):
    """Distance of each point to the boundary of ``poly``."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    l2 = np.maximum((ab**2).sum(1), 1e-300)
    t = np.clip(((points[:, None, :] - a[None]) * ab[None]).sum(-1) / l2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - proj, axis=-1).min(1)


def _area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _match_layers(ours: dict, theirs: dict, tol_nm=1.0):
    """Same layers, same number of polygons, and each polygon's vertices
    within ``tol_nm`` of the other's outline (both ways), same area."""
    assert sorted(ours) == sorted(theirs), (sorted(ours), sorted(theirs))
    for k in ours:
        a = sorted((np.asarray(p) for p in ours[k]), key=lambda p: tuple(np.round((p.min(0) + p.max(0)) * 0.5e9, 0)))
        b = sorted((np.asarray(p) for p in theirs[k]), key=lambda p: tuple(np.round((p.min(0) + p.max(0)) * 0.5e9, 0)))
        assert len(a) == len(b), (k, len(a), len(b))
        for p, q in zip(a, b):
            assert _seg_dist(p, q).max() <= tol_nm * 1e-9 and _seg_dist(q, p).max() <= tol_nm * 1e-9, k
            assert abs(_area(p) - _area(q)) <= 1e-6 * _area(q) + 1e-21, k


def _gdstk_layers(cell):
    out = {}
    for p in cell.get_polygons():
        out.setdefault((p.layer, p.datatype), []).append(np.asarray(p.points) * 1e-6)
    return out


def test_gds_reader_against_gdstk(tmp_path):
    """References, arrays, rotations, reflection, magnification and paths, as gdstk flattens them."""
    gdstk = pytest.importorskip("gdstk")
    from lightsim.layout import read_gds

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    leaf = lib.new_cell("leaf")
    leaf.add(gdstk.Polygon([(0, 0), (1.2, 0), (1.2, 0.4), (0.3, 0.9)], layer=1, datatype=0))
    leaf.add(gdstk.rectangle((0, -1), (0.5, -0.5), layer=2, datatype=3))
    for ends, layer in (("flush", 3), ("extended", 4), ((0.05, 0.12), 5)):
        leaf.add(gdstk.FlexPath([(0, 2), (2, 2), (2, 4), (3.5, 5.5)], 0.25, ends=ends, simple_path=True, layer=layer))
    leaf.add(gdstk.Label("p1", (0.6, 0.2), layer=10))
    top = lib.new_cell("top")
    top.add(gdstk.Reference(leaf, (10, 0), rotation=math.pi / 2))
    top.add(gdstk.Reference(leaf, (0, 10), rotation=math.radians(30), magnification=1.5, x_reflection=True))
    top.add(gdstk.Reference(leaf, (20, 20), columns=3, rows=2, spacing=(5, 7), rotation=math.pi, x_reflection=True))
    lib.write_gds(tmp_path / "g.gds")
    ours = read_gds(tmp_path / "g.gds").polygons
    theirs_lib = gdstk.read_gds(tmp_path / "g.gds")
    t = [c for c in theirs_lib.top_level()][0]
    _match_layers(ours, _gdstk_layers(t))
    labels = read_gds(tmp_path / "g.gds").labels
    assert len(labels) == 1 + 1 + 6


def test_gds_reader_on_gdsfactory_and_gamma_files(tmp_path):
    from lightsim.layout import read_gds

    lay = read_gds(FIX / "gamma" / "mmi.gds")
    assert (1, 0) in lay.polygons
    (x0, y0), (x1, y1) = lay.bbox([(1, 0)])
    assert (x0, x1) == (pytest.approx(-10e-6, abs=1e-9), pytest.approx(15.5e-6, abs=1e-9))
    gf = pytest.importorskip("gdsfactory")
    gdstk = pytest.importorskip("gdstk")
    gf.gpdk.PDK.activate()
    c = gf.Component()
    r1 = c << gf.components.mmi2x2()
    r2 = c << gf.components.bend_euler(radius=5)
    r2.drotate(90)
    r2.dmove((30, 0))
    r3 = c << gf.components.straight(length=3)
    r3.dmirror_y()
    r3.dmove((0, -8))
    path = c.write_gds(tmp_path / "c.gds")
    ours = read_gds(path).polygons
    t = [c for c in gdstk.read_gds(path).top_level() if not c.name.startswith("$$$")][0]
    _match_layers(ours, _gdstk_layers(t))


def test_cli_gds(tmp_path):
    import shutil

    exe = shutil.which("lightsim") or (Path(__file__).resolve().parents[2] / "target" / "release" / "lightsim")
    if not Path(exe).exists():
        pytest.skip("the lightsim program is not built")
    write_small_gds(tmp_path / "s.gds")
    r = subprocess.run([str(exe), "gds", str(tmp_path / "s.gds"), "--out", str(tmp_path / "flat.json")], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    assert info["top_cells"] == ["top"]
    flat = json.loads((tmp_path / "flat.json").read_text())
    assert flat["cell"] == "top" and {(l["layer"], l["datatype"]) for l in flat["layers"]} == {(5, 0), (6, 0)}


# ----------------------------------------------------------------------------
# ports and technology
# ----------------------------------------------------------------------------


def test_gamma_port_files(tmp_path):
    from lightsim.layout import read_ports

    j = read_ports(FIX / "gamma" / "mmi.ports.json")
    assert [p["name"] for p in j] == ["o1", "o2", "o3"]
    assert j[1]["center"] == pytest.approx((15.5e-6, 0.625e-6)) and j[1]["orientation"] == 0 and j[0]["layer"] == (1, 0)
    csv = tmp_path / "ports.csv"
    raw = json.loads((FIX / "gamma" / "mmi.ports.json").read_text())
    csv.write_text("name,x,y,width,orientation,layer,datatype\n" + "".join(f"{p['name']},{p['x']},{p['y']},{p['width']},{p['orientation']},{p['layer'][0]},{p['layer'][1]}\n" for p in raw))
    assert read_ports(csv) == j
    ext = tmp_path / "ext.json"
    ext.write_text(json.dumps({"version": 1, "units": "nm", "ports": [{"name": "a", "center": [1000, 0], "width": 500, "direction": "west", "modes": [0, 1], "z": [-500, 720], "color": "red"}]}))
    (p,) = read_ports(ext)
    assert p["orientation"] == 180 and p["modes"] == [0, 1] and p["z"] == pytest.approx((-0.5e-6, 0.72e-6)) and p["width"] == pytest.approx(0.5e-6)


def test_technology_files(tmp_path):
    from lightsim.layout import read_technology

    t = read_technology(FIX / "gamma" / "stack.toml")
    assert [l.name for l in t.levels] == ["box", "core", "clad"]
    assert t.level("core").layers == [(1, 0)] and t.level("box").layers == []
    assert t.material_index()["si"] == pytest.approx(3.476) and t.background_material() == "sio2"
    t.write(tmp_path / "tech.json")
    t2 = read_technology(tmp_path / "tech.json")
    assert t2.to_dict() == t.to_dict()
    pytest.importorskip("yaml")
    t.write(tmp_path / "tech.yaml")
    assert read_technology(tmp_path / "tech.yaml").to_dict() == t.to_dict()


def test_lyp(tmp_path):
    from lightsim.layout import read_lyp

    (tmp_path / "x.lyp").write_text(
        "<?xml version='1.0' encoding='utf-8'?><layer-properties>"
        "<properties><name>SiEPIC</name><source>*/*@*</source>"
        "<group-members><name>PinRec</name><source>1/10@1</source></group-members>"
        "</properties><properties><name>Si</name><source>1/0@1</source></properties></layer-properties>"
    )
    assert read_lyp(tmp_path / "x.lyp") == {(1, 10): "PinRec", (1, 0): "Si"}  # groups are walked, wildcards skipped
    (tmp_path / "y.lyp").write_text("<layer-properties><properties><name>DevRec</name><source>68/0@1</source></properties><properties><source>1/10@1</source></properties></layer-properties>")
    assert read_lyp(tmp_path / "y.lyp") == {(68, 0): "DevRec", (1, 10): "1/10"}


def test_technology_from_gdsfactory():
    gf = pytest.importorskip("gdsfactory")
    from lightsim.layout import technology_from_gdsfactory

    gf.gpdk.PDK.activate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = technology_from_gdsfactory()
    core = t.level("core")
    assert core.layers == [(1, 0)] and core.subtract and core.thickness == pytest.approx(0.22)
    assert t.level("box").layers == [] and "metal1" not in [l.name for l in t.levels]


def _siepic_gds(path):
    """A waveguide with SiEPIC pins (PinRec 1/10 paths pointing outwards, labels on 1/10)."""
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library()
    c = lib.new_cell("wg")
    c.add(gdstk.rectangle((0, -0.25), (5, 0.25), layer=1))
    c.add(gdstk.FlexPath([(0.05, 0), (-0.05, 0)], 0.5, layer=1, datatype=10, simple_path=True))
    c.add(gdstk.FlexPath([(4.95, 0), (5.05, 0)], 0.5, layer=1, datatype=10, simple_path=True))
    c.add(gdstk.Label("opt1", (0, 0), layer=1, texttype=10))
    c.add(gdstk.Label("opt2", (5, 0), layer=1, texttype=10))
    c.add(gdstk.rectangle((0, -1), (5, 1), layer=68))
    lib.write_gds(path)


def test_siepic_and_pin_ports(tmp_path):
    from lightsim.layout import read_gds, read_ports

    _siepic_gds(tmp_path / "s.gds")
    lay = read_gds(tmp_path / "s.gds")
    ports = read_ports("siepic", lay)
    assert [(p["name"], p["orientation"], p["layer"]) for p in ports] == [("opt1", 180.0, (1, 0)), ("opt2", 0.0, (1, 0))]
    assert ports[1]["center"] == pytest.approx((5e-6, 0.0), abs=1e-12) and ports[1]["width"] == pytest.approx(0.5e-6)
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library()
    c = lib.new_cell("wg")
    c.add(gdstk.rectangle((0, -0.25), (5, 0.25), layer=1))
    c.add(gdstk.rectangle((-0.05, -0.25), (0.05, 0.25), layer=90))
    c.add(gdstk.rectangle((4.95, -0.25), (5.05, 0.25), layer=90))
    c.add(gdstk.Label("in", (0, 0.1), layer=91))
    c.add(gdstk.Label("out", (5, 0.1), layer=91))
    lib.write_gds(tmp_path / "p.gds")
    p2 = read_ports({"convention": "pins", "pin_layer": "90/0", "label_layer": "91/0", "port_layer": "1/0"}, read_gds(tmp_path / "p.gds"))
    assert [(p["name"], p["orientation"]) for p in p2] == [("in", 180.0), ("out", 0.0)]


# ----------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------

FAST = dict(is_3d=False, effective_index=False, resolution=20, wavelengths=np.linspace(1.5e-6, 1.6e-6, 3), margin=2.0, decay_by=1e-4)


def test_layout_sparameters_gamma_wire():
    import lightsim.layout as ll

    sp = ll.sparameters(FIX / "gamma" / "wire.gds", FIX / "gamma" / "stack.toml", FIX / "gamma" / "wire.ports.json", device="cpu", **FAST)
    assert sp.ports == [("o1", 0), ("o2", 0)]
    assert np.all(np.abs(sp["o2", "o1"]) > 0.98) and np.allclose(sp["o2", "o1"], sp["o1", "o2"], atol=1e-2)


def test_layout_matches_gplugins_front(tmp_path):
    """The neutral route and lightsim.gplugins build the same simulation for the same layout."""
    gf = pytest.importorskip("gdsfactory")
    import lightsim.gplugins as lg
    import lightsim.layout as ll

    gf.gpdk.PDK.activate()
    c = gf.components.straight(length=5, width=0.45)
    ref = lg.write_sparameters_lightsim(
        c, dirpath=tmp_path, device="cpu", resolution=20, wavelength_points=3, effective_index=False, xmargin=2, ymargin=2, decay_by=1e-4,
        port_margin=1.0, material_name_to_meep={"si": 3.476, "sio2": 1.444},
    )  # fmt: skip
    sp = ll.sparameters(FIX / "gamma" / "wire.gds", FIX / "gamma" / "stack.toml", FIX / "gamma" / "wire.ports.json", device="cpu", **FAST)
    assert np.allclose(sp.to_dict()["o2@0,o1@0"], ref["o2@0,o1@0"], atol=2e-3)


def test_layout_simulation_specs_and_cli(tmp_path):
    import lightsim.layout as ll
    from lightsim import Simulation

    sims = ll.simulation(FIX / "gamma" / "mmi.gds", FIX / "gamma" / "stack.toml", FIX / "gamma" / "mmi.ports.json", sources=["o1"], **FAST)
    assert list(sims) == ["o1@0"] and isinstance(sims["o1@0"], Simulation)
    (p,) = ll.write_specs(sims, tmp_path / "spec.json")
    spec = json.loads(p.read_text())
    assert len(spec["sources"]) == 1 and len(spec["monitors"]) == 3
    assert spec["sources"][0]["type"] == "mode" and "profile" in spec["sources"][0]
    r = subprocess.run(
        [sys.executable, "-m", "lightsim", "layout-spec", str(FIX / "gamma" / "mmi.gds"), "--technology", str(FIX / "gamma" / "stack.toml"), "--ports", str(FIX / "gamma" / "mmi.ports.json"), "--2d", "--bulk", "--resolution", "15", "--out", str(tmp_path / "mmi.json")],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert len(r.stdout.split()) == 3 and (tmp_path / "mmi_o2@0.json").exists()


def test_layout_cli_sparams(tmp_path):
    out = tmp_path / "wire.s2p"
    subprocess.run(
        [sys.executable, "-m", "lightsim", "sparams", str(FIX / "gamma" / "wire.gds"), "--technology", str(FIX / "gamma" / "stack.toml"), "--ports", str(FIX / "gamma" / "wire.ports.json"), "--2d", "--bulk", "--resolution", "15", "--wavelengths", "1.5:1.6:3", "--margin", "2", "--device", "cpu", "--out", str(out)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    sp = SParameters.from_touchstone(out)
    assert sp.ports == [("o1", 0), ("o2", 0)] and np.all(np.abs(sp["o2", "o1"]) > 0.95)
