"""The graphical interface: the server's endpoints against the Python API,
runs started from it (to completion and stopped), the access rules and a
clean shutdown. CPU only, no browser."""

import base64
import http.client
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import lightsim as ls
from lightsim import _core
from lightsim import gui
from lightsim.gui.app import Session
from lightsim.gui.server import Server, is_loopback, needs_token
from lightsim.spec import um

STATIC = Path(gui.__file__).resolve().parent / "static"


def small_sim(run_time=25e-15, shutoff=None):
    """A silicon strip in oxide with a dipole, a flux plane, a field plane, a
    probe and a mode monitor: about 70 000 cells."""
    f0 = ls.C0 / (1.55 * um)
    sim = ls.Simulation(min=[0, -0.6 * um, -0.5 * um], max=[2 * um, 0.6 * um, 0.5 * um], mesh=ls.UniformMesh(50e-9),
                        run_time=run_time, background="sio2", shutoff=shutoff, name="gui test",
                        boundaries={k: [ls.PML(layers=6), ls.PML(layers=6)] for k in "xyz"})
    sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [3 * um, 0.25 * um, 0.11 * um]), "si", name="strip")
    sim.add_source(ls.PointDipole([0.5 * um, 0, 0], "Ey", ls.GaussianPulse(f0, 0.2 * f0), name="dipole"))
    freqs = f0 * np.linspace(0.9, 1.1, 5)
    sim.add_monitor(ls.FluxMonitor("T", 0, 1.5 * um, [0, -0.5 * um, -0.4 * um], [0, 0.5 * um, 0.4 * um], freqs))
    sim.add_monitor(ls.DFTMonitor("plane", [0, -0.6 * um, 0], [2 * um, 0.6 * um, 0], freqs[::2], components=("Ey", "Hz")))
    sim.add_monitor(ls.ProbeMonitor("probe", [1.2 * um, 0, 0], components=("Ey",)))
    lo, hi = [1.6 * um, -0.5 * um, -0.4 * um], [1.6 * um, 0.5 * um, 0.4 * um]
    modes = ls.solve_modes(sim, 0, 1.6 * um, lo, hi, f0, num_modes=1)
    sim.add_monitor(ls.ModeMonitor("mode", 0, 1.6 * um, lo, hi, freqs, [modes[0].profile]))
    return sim


@pytest.fixture(scope="module")
def sim():
    return small_sim()


@pytest.fixture(scope="module")
def results(sim):
    return sim.run(device="cpu")


@pytest.fixture()
def server(sim):
    srv = Server(Session(sim)).start()
    yield srv
    srv.stop()


def get(srv, path, headers=None):
    req = urllib.request.Request(srv.url.split("?")[0].rstrip("/") + path, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def post(srv, path, body):
    req = urllib.request.Request(srv.url.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def f32(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def wait_for(srv, cond, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = get(srv, "/api/run/state")
        if cond(s):
            return s
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state {s}")


def test_info_matches_the_python_api(server, sim):
    info = get(server, "/api/info")
    nodes = sim.grid()
    assert info["n"] == [len(a) - 1 for a in nodes]
    for a in range(3):
        np.testing.assert_allclose(info["nodes"][a], np.asarray(nodes[a]) / um, atol=1e-6)
    dt, steps = sim.time_info()
    s = info["summary"]
    assert s["dt"] == pytest.approx(dt) and s["nsteps"] == steps
    assert s["cells"] == int(np.prod(info["n"]))
    assert s["band"]["lambda_min_um"] == pytest.approx(ls.C0 / (1.1 * ls.C0 / (1.55 * um)) / um)
    assert s["densest"]["medium"] == "si" and s["densest"]["n"] == pytest.approx(3.476, abs=1e-3)
    # 50 nm cells, 1.409 um / 3.476 in silicon: 8.1 cells per wavelength, flagged as coarse
    assert s["densest"]["cells_per_wavelength"] == pytest.approx(1.55 / 1.1 / 3.476 / 0.05, rel=1e-3)
    assert any("coarse mesh" in w for w in s["warnings"])
    assert s["memory"]["per_device"]["2"] < s["memory"]["per_device"]["1"]
    assert [x["name"] for x in info["monitors"]] == ["T", "plane", "probe", "mode"]
    assert info["sources"][0]["geometry"]["kind"] == "point"
    pml = info["boundaries"]["x"][0]
    assert pml["type"] == "pml" and pml["layers"] == 6
    assert pml["max"] - pml["min"] == pytest.approx(6 * 0.05, abs=1e-6)
    assert info["devices"][0]["id"] == "cpu"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cut_equals_the_plan_permittivity(server, sim, axis):
    """A cut is the permittivity the engine uses (sub-pixel smoothed): the
    same numbers as the mode solver's plane_info on a full window."""
    js = sim.to_json()
    nodes = sim.grid()
    n = len(nodes[axis]) - 1
    index = n // 2 + 1
    u, v = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]
    pi = _core.plane_info(js, axis, float(nodes[axis][index]), [-1.0] * 3, [1.0] * 3)
    assert pi["p"] == index and list(pi["dims"]) == [len(nodes[u]) - 1, len(nodes[v]) - 1]
    for q, ref in ((f"eps_{'xyz'[u]}", pi["eps_u"]), (f"eps_{'xyz'[v]}", pi["eps_v"]), (f"eps_{'xyz'[axis]}", pi["eps_n"])):
        c = get(server, f"/api/cut?axis={axis}&index={index}&q={q}")
        a = f32(c["data"]).reshape(c["nu"], c["nv"])
        assert c["stride"] == [1, 1]
        np.testing.assert_array_equal(a, np.asarray(ref, dtype=np.float32))
        np.testing.assert_allclose(c["u_edges"], np.asarray(nodes[u]) / um, atol=1e-6)
        np.testing.assert_allclose(c["v_edges"], np.asarray(nodes[v]) / um, atol=1e-6)
    # smoothing makes intermediate values at the strip edges
    c = get(server, f"/api/cut?axis={axis}&index={index}&q=n")
    a = f32(c["data"])
    assert c["max"] == pytest.approx(3.476, abs=1e-3) and c["min"] == pytest.approx(1.444, abs=1e-3)
    assert np.any((a > 1.5) & (a < 3.4))
    med = get(server, f"/api/cut?axis={axis}&index={index}&q=medium")
    assert med["categories"] == ["sio2", "si"]
    assert set(np.unique(f32(med["data"]))) == {0.0, 1.0}


def test_cut_is_downsampled_not_the_volume(server):
    full = get(server, "/api/cut?axis=2&index=10&q=eps_y")
    small = get(server, "/api/cut?axis=2&index=10&q=eps_y&side=12")
    assert small["nu"] <= 12 and small["nv"] <= 12
    ku, kv = small["stride"]
    a = f32(full["data"]).reshape(full["nu"], full["nv"])
    np.testing.assert_array_equal(f32(small["data"]).reshape(small["nu"], small["nv"]), a[::ku, ::kv])
    assert len(small["u_edges"]) == small["nu"] + 1 and small["u_edges"][-1] == full["u_edges"][-1]


def test_results_endpoints_equal_the_python_api(sim, results):
    srv = Server(Session(sim, results)).start()
    try:
        idx = get(srv, "/api/results")
        assert idx["available"]
        kinds = {m["name"]: m["type"] for m in idx["monitors"]}
        assert kinds == {"T": "flux", "plane": "dft_field", "probe": "probe", "mode": "mode"}
        f = get(srv, "/api/results/flux?name=T")
        np.testing.assert_array_equal(f["flux"], np.asarray(results.flux("T"), dtype=float))
        np.testing.assert_allclose(f["freqs"], results.freqs("T"))
        _, S = results.source_spectrum("dipole", monitor="T")
        np.testing.assert_allclose(f["source_power"], np.abs(S) ** 2, rtol=1e-12)
        m = get(srv, "/api/results/mode?name=mode")
        ap, am = results.mode_amplitudes("mode")
        np.testing.assert_allclose(np.asarray(m["modes"][0]["plus_re"]) + 1j * np.asarray(m["modes"][0]["plus_im"]), ap)
        np.testing.assert_allclose(np.asarray(m["modes"][0]["minus_re"]) + 1j * np.asarray(m["modes"][0]["minus_im"]), am)
        assert m["modes"][0]["neff"] > 2
        p = get(srv, "/api/results/profiles?name=mode")
        assert p["profiles"][0]["te_fraction"] > 0.8 and p["axis"] == 0
        for part, fn in (("abs", np.abs), ("re", np.real), ("phase", np.angle)):
            d = get(srv, f"/api/results/map?name=plane&comp=Ey&f=1&part={part}")
            field = results.field("plane", "Ey")[1][:, :, 0]
            np.testing.assert_allclose(f32(d["data"]).reshape(d["nu"], d["nv"]), fn(field).astype(np.float32), rtol=1e-6, atol=1e-30)
            assert d["axis"] == 2 and d["signed"] == (part != "abs")
            assert len(d["outline"]) > 0
        t = get(srv, "/api/results/time?name=probe&comp=Ey")
        times, vals = results.time_series("probe", "Ey")
        np.testing.assert_allclose(t["times"], times)
        np.testing.assert_allclose(t["values"], np.asarray(vals)[:, 0, 0, 0], rtol=1e-6)
        assert len(t["spec"]) == len(t["spec_freqs"]) > 10
        assert idx["stats"]["steps"] == results.stats["steps"]
    finally:
        srv.stop()


def test_run_to_completion_gives_the_same_flux(server, results):
    s = post(server, "/api/run/start", {"device": "cpu", "comp": "Ey", "axis": 2, "index": 10, "every": 0.02})
    assert s["status"] in ("compiling", "running", "done")
    s = wait_for(server, lambda s: s["status"] not in ("compiling", "running"))
    assert s["status"] == "done", s
    assert s["step"] == results.stats["steps"]
    f = get(server, "/api/results/flux?name=T")
    np.testing.assert_allclose(f["flux"], np.asarray(results.flux("T")), rtol=1e-6)
    # the live slice: the requested plane and component, symmetric limits
    sl = get(server, "/api/run/slice")
    assert s["slice_version"] > 0
    assert sl["comp"] == "Ey" and sl["axis"] == 2 and sl["index"] == 10
    assert len(f32(sl["data"])) == sl["nu"] * sl["nv"] and sl["min"] == -sl["max"]


def test_live_slice_and_stop():
    sim = small_sim(run_time=4e-12)
    srv = Server(Session(sim)).start()
    try:
        post(srv, "/api/run/start", {"device": "cpu", "comp": "Ez", "axis": 1, "index": 7, "every": 0.02})
        s = wait_for(srv, lambda s: s["slice_version"] >= 2)
        sl = get(srv, "/api/run/slice")
        n = sim.grid()
        assert sl["comp"] == "Ez" and sl["axis"] == 1 and sl["index"] == 7
        assert (sl["nu"], sl["nv"]) == (len(n[0]) - 1, len(n[2]) - 1)
        assert sl["step"] > 0 and sl["time"] == pytest.approx(sl["step"] * s["dt"])
        # a running run takes another plane
        post(srv, "/api/run/plane", {"comp": "Hx", "axis": 0, "index": 5})
        v = s["slice_version"]
        wait_for(srv, lambda s: s["slice_version"] >= v + 2)
        sl = get(srv, "/api/run/slice")
        assert sl["comp"] == "Hx" and sl["axis"] == 0 and (sl["nu"], sl["nv"]) == (len(n[1]) - 1, len(n[2]) - 1)
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, "/api/run/start", {"device": "cpu"})
        assert e.value.code == 400
        post(srv, "/api/run/stop", {})
        s = wait_for(srv, lambda s: s["status"] not in ("compiling", "running"))
        assert s["status"] == "stopped" and 0 < s["step"] < s["nsteps"]
        idx = get(srv, "/api/results")
        assert idx["available"] and idx["stats"]["steps"] == s["step"] and idx["stats"]["stopped_early"]
    finally:
        srv.stop()


def test_bad_requests(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, "/api/run/start", {"device": "cuda:7"})
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/api/cut?axis=2&index=0&q=nothing")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/api/results/flux?name=T")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/../spec.py")
    assert e.value.code == 404


def test_loopback_and_token_rules(sim):
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1") and is_loopback("[::1]")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.0.2.4") and not is_loopback("example.org")
    assert needs_token("0.0.0.0") and needs_token("192.0.2.4") and not needs_token("127.0.0.1")
    srv = Server(Session(sim)).start()
    try:
        assert srv.host == "127.0.0.1" and srv.token is None and srv.url.startswith("http://127.0.0.1:")
        # a request through another host name (DNS rebinding) is refused
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        c.request("GET", "/api/info", headers={"Host": f"attacker.example:{srv.port}"})
        assert c.getresponse().status == 403
        c.close()
        assert get(srv, "/api/info")["title"] == "gui test"
    finally:
        srv.stop()
    # with a token (as on a non-loopback address) every request needs it
    srv = Server(Session(sim), token="s3cret-token").start()
    try:
        assert srv.url.endswith("?token=s3cret-token")
        base = f"http://127.0.0.1:{srv.port}"
        for path in ("/", "/api/info", "/js/main.js"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + path, timeout=10)
            assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/info?token=wrong", timeout=10)
        r = urllib.request.urlopen(base + "/?token=s3cret-token", timeout=10)
        cookie = r.headers["Set-Cookie"]
        assert "lightsim_token=s3cret-token" in cookie and "HttpOnly" in cookie
        req = urllib.request.Request(base + "/api/info", headers={"Cookie": "lightsim_token=s3cret-token"})
        assert json.loads(urllib.request.urlopen(req, timeout=10).read())["has_spec"]
        req = urllib.request.Request(base + "/api/info", headers={"X-Lightsim-Token": "s3cret-token"})
        assert urllib.request.urlopen(req, timeout=10).status == 200
    finally:
        srv.stop()


def test_clean_shutdown_releases_the_port(sim):
    srv = Server(Session(sim)).start()
    port = srv.port
    assert get(srv, "/api/run/state")["status"] == "idle"
    srv.stop()
    assert srv.thread is None
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_view_returns_the_url_and_stops(sim, results, capsys):
    url = sim.view(results, open_browser=False)
    try:
        assert url.startswith("http://127.0.0.1:") and url.endswith("/")
        assert url in capsys.readouterr().out
        with urllib.request.urlopen(url + "api/results", timeout=10) as r:
            assert json.loads(r.read())["available"]
        url2 = gui.view(results, open_browser=False)
        with urllib.request.urlopen(url2 + "api/info", timeout=10) as r:
            info = json.loads(r.read())
        assert info["has_results"] and not info["has_spec"]
    finally:
        gui.stop()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(url + "api/info", timeout=2)


def test_two_dimensional_simulation():
    f0 = ls.C0 / (1.55 * um)
    sim = ls.Simulation(min=[0, -1 * um], max=[3 * um, 1 * um], mesh=ls.UniformMesh(40e-9), run_time=40e-15, dimensions=2, polarization="te", background="sio2")
    sim.add_structure(ls.Box([-1, -0.25 * um], [4 * um, 0.25 * um]), "si")
    sim.add_source(ls.PointDipole([0.5 * um, 0], "Ey", ls.GaussianPulse(f0, 0.2 * f0)))
    sim.add_monitor(ls.DFTMonitor("field", [0, -1 * um], [3 * um, 1 * um], [f0], components=("Ex", "Ey", "Hz")))
    s = Session(sim)
    info = s.info()
    assert info["dims"] == 2 and info["invariant"] == 2 and info["n"][2] == 1
    assert info["run_defaults"] == {"comp": "Hz", "axis": 2, "index": 0}
    c = s.cut(2, 0, "n")
    assert c["nu"] == info["n"][0] and c["nv"] == info["n"][1]
    srv = Server(s).start()
    try:
        post(srv, "/api/run/start", {"device": "cpu", "comp": "Hz", "axis": 2, "index": 0, "every": 0.02})
        st = wait_for(srv, lambda q: q["status"] not in ("compiling", "running"))
        assert st["status"] == "done"
        d = get(srv, "/api/results/map?name=field&comp=Hz&f=0&part=abs")
        assert d["axis"] == 2 and (d["nu"], d["nv"]) == (75, 50)
        assert d["u_edges"][0] == pytest.approx(0.0, abs=1e-9) and d["u_edges"][-1] == pytest.approx(3.0, abs=1e-6)
    finally:
        srv.stop()


def test_static_files_load_nothing_from_the_network():
    files = [p for p in STATIC.rglob("*") if p.is_file()]
    assert {p.name for p in files} >= {"index.html", "style.css", "main.js", "colormaps.js"}
    for p in files:
        text = p.read_text()
        for bad in ("http://", "https://", "//cdn", "@import", "fonts.googleapis"):
            hits = [line for line in text.splitlines() if bad in line and "www.w3.org/2000/svg" not in line]
            assert not hits, (p.name, hits[:2])
    total = sum(p.stat().st_size for p in files)
    assert total < 250_000


def test_module_command_line():
    r = subprocess.run([sys.executable, "-m", "lightsim.gui", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--results" in r.stdout and "--host" in r.stdout


def _cli():
    import shutil

    exe = Path(__file__).resolve().parents[2] / "target" / "release" / "lightsim"
    return str(exe) if exe.is_file() else shutil.which("lightsim")


@pytest.mark.skipif(_cli() is None, reason="the lightsim command-line program is not built")
def test_results_file_of_the_command_line(sim, results, tmp_path):
    spec = tmp_path / "spec.json"
    out = tmp_path / "results.json"
    spec.write_text(sim.to_json())
    subprocess.run([_cli(), "run", str(spec), "--device", "cpu", "--out", str(out)], check=True, capture_output=True, timeout=120)
    from lightsim.gui.app import results_from_cli_json

    raw = results_from_cli_json(json.loads(out.read_text()))
    s = Session(json.loads(spec.read_text()), raw)
    np.testing.assert_allclose(s.flux_data("T")["flux"], np.asarray(results.flux("T")), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(s.mode_data("mode")["modes"][0]["plus_re"]), results.mode_amplitudes("mode")[0].real, rtol=1e-4, atol=1e-30)
    d = s.dft_map("plane", "Ey", 1, "abs")
    np.testing.assert_allclose(f32(d["data"]).reshape(d["nu"], d["nv"]), np.abs(results.field("plane", "Ey")[1][:, :, 0]), rtol=1e-5, atol=1e-30)
    assert s.time_data("probe", "Ey")["values"][:5] == pytest.approx(np.asarray(results.time_series("probe", "Ey")[1])[:5, 0, 0, 0].tolist(), rel=1e-5, abs=1e-30)
    assert s.results_index()["available"]
