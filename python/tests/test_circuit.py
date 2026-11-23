"""Circuit solver against closed forms (Mach-Zehnder, all-pass and add-drop
rings, a lattice filter), several modes per port, S-parameter files in both
key orders, models from the mode solver, and SAX interoperability (skipped
without SAX)."""

import numpy as np
import pytest

from lightsim import circuit as C

WL = np.linspace(1.50, 1.60, 801)
NEFF, NG = 2.4, 4.2


def wg(length, loss=0.0):
    return C.straight(wl=WL, neff=NEFF, ng=NG, length=length, loss_dB_cm=loss)[("o1", "o2")]


def test_mzi_from_gdsfactory_netlist():
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    c = gf.components.mzi(delta_length=10)
    split = lambda wl=1.55: C.reciprocal({("o1", "o2"): np.sqrt(0.5) + 0 * wl, ("o1", "o3"): np.sqrt(0.5) + 0 * wl})  # noqa: E731
    line = lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=NEFF, ng=NG, length=length)  # noqa: E731
    S = C.simulate(c, {"straight": line, "bend_euler": line, "mmi1x2": split}, wl=WL)
    n = c.get_netlist()
    L = lambda ks: sum(n["instances"][k]["settings"].get("length") or n["instances"][k]["info"]["length"] for k in ks)  # noqa: E731
    top, bot = L(["b1", "sytl", "b2", "sxt", "b3", "sytr", "b4"]), L(["b5", "syl", "b6", "sxb", "b7", "sybr", "b8"])
    assert abs(bot - top - 10) < 1e-9
    ref = 0.5 * (wg(top) + wg(bot))
    assert np.abs(S[("o1", "o2")] - ref).max() < 1e-12
    assert np.abs(S[("o2", "o1")] - ref).max() < 1e-12


def _ring_netlist(double=False):
    inst = {"A": "dc", "L1": "wg"}
    conn = {"A,o3": "L1,o1", "L1,o2": "A,o2"}
    ports = {"o1": "A,o1", "o2": "A,o4"}
    if double:
        inst = {"A": "dc", "B": "dc2", "L1": "wg", "L2": "wg"}
        conn = {"A,o3": "L1,o1", "L1,o2": "B,o2", "B,o3": "L2,o1", "L2,o2": "A,o2"}
        ports = {"o1": "A,o1", "o2": "A,o4", "o3": "B,o4", "o4": "B,o1"}
    return {"instances": inst, "connections": conn, "ports": ports}


def test_all_pass_ring_closed_form():
    k, L, loss = 0.05, 60.0, 30.0  # 30 dB/cm makes the dips finite
    models = {"dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=k), "wg": lambda wl=1.55: C.straight(wl=wl, neff=NEFF, ng=NG, length=L, loss_dB_cm=loss)}
    fn, _ = C.circuit(_ring_netlist(), models)
    S = fn(wl=WL)
    ref = C.ring_single(wl=WL, neff=NEFF, ng=NG, length=L, loss_dB_cm=loss, coupling=k)[("o1", "o2")]
    assert np.abs(S[("o1", "o2")] - ref).max() < 1e-12
    assert (np.abs(ref) ** 2).min() < 0.5  # resonances inside the band


def test_add_drop_ring_closed_form():
    k1, k2, L, loss = 0.08, 0.05, 80.0, 20.0
    models = {
        "dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=k1),
        "dc2": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=k2),
        "wg": lambda wl=1.55: C.straight(wl=wl, neff=NEFF, ng=NG, length=L / 2, loss_dB_cm=loss),
    }
    S = C.circuit(_ring_netlist(True), models)[0](wl=WL)
    ref = C.ring_double(wl=WL, neff=NEFF, ng=NG, length=L, loss_dB_cm=loss, coupling1=k1, coupling2=k2)
    for key in [("o1", "o2"), ("o1", "o3"), ("o4", "o3"), ("o4", "o2")]:
        assert np.abs(S[key] - ref[key]).max() < 1e-12, key
    assert (np.abs(S[("o1", "o3")]) ** 2).max() > 0.5


def test_lattice_filter_against_transfer_matrices():
    ks = [0.5, 0.13, 0.4, 0.07]
    dL = [20.0, 40.0, 20.0]
    inst, conn = {}, {}
    for i, _ in enumerate(ks):
        inst[f"c{i}"] = f"dc{i}"
    for i, d in enumerate(dL):
        inst[f"u{i}"], inst[f"l{i}"] = f"long{i}", "short"
        conn[f"c{i},o3"] = f"u{i},o1"
        conn[f"u{i},o2"] = f"c{i + 1},o2"
        conn[f"c{i},o4"] = f"l{i},o1"
        conn[f"l{i},o2"] = f"c{i + 1},o1"
    n = len(ks) - 1
    net = {"instances": inst, "connections": conn, "ports": {"in1": "c0,o1", "in2": "c0,o2", "out_top": f"c{n},o3", "out_bot": f"c{n},o4"}}
    models = {f"dc{i}": (lambda k: lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=k))(k) for i, k in enumerate(ks)}
    models.update({f"long{i}": (lambda d: lambda wl=1.55: C.straight(wl=wl, neff=NEFF, ng=NG, length=10.0 + d))(d) for i, d in enumerate(dL)})
    models["short"] = lambda wl=1.55: C.straight(wl=wl, neff=NEFF, ng=NG, length=10.0)
    S = C.circuit(net, models)[0](wl=WL)
    # transfer matrices on the (bottom, top) waveguide pair
    T = np.broadcast_to(np.eye(2, dtype=complex), (len(WL), 2, 2))
    for i, k in enumerate(ks):
        t, x = np.sqrt(1 - k), 1j * np.sqrt(k)
        T = np.array([[t, x], [x, t]])[None] @ T
        if i < len(dL):
            D = np.zeros((len(WL), 2, 2), dtype=complex)
            D[:, 0, 0], D[:, 1, 1] = wg(10.0), wg(10.0 + dL[i])
            T = D @ T
    assert np.abs(S[("in1", "out_bot")] - T[:, 0, 0]).max() < 1e-12
    assert np.abs(S[("in1", "out_top")] - T[:, 1, 0]).max() < 1e-12
    assert np.abs(S[("in2", "out_top")] - T[:, 1, 1]).max() < 1e-12


def test_several_modes_per_port_and_dangling_ports():
    # a mode converter (TE0 -> TE1 at 30 %) between two two-mode straights
    def mc(wl=1.55):
        one = np.ones_like(np.asarray(wl, dtype=float))
        a, b = np.sqrt(0.7), np.sqrt(0.3)
        return C.reciprocal({("o1@0", "o2@0"): a * one, ("o1@0", "o2@1"): b * one, ("o1@1", "o2@1"): a * one, ("o1@1", "o2@0"): -b * one})

    def wg2(wl=1.55, length=5.0):
        w = np.asarray(wl, dtype=float)
        return C.reciprocal({("o1@0", "o2@0"): np.exp(2j * np.pi * 2.4 * length / w), ("o1@1", "o2@1"): np.exp(2j * np.pi * 1.9 * length / w)})

    net = {"instances": {"a": "wg2", "m": "mc", "b": "wg2", "t": "wg"}, "connections": {"a,o2": "m,o1", "m,o2": "b,o1"}, "ports": {"in": "a,o1", "out": "b,o2"}}
    S = C.circuit(net, {"wg2": wg2, "mc": mc, "wg": lambda wl=1.55: C.straight(wl=wl)})[0](wl=WL)
    p0, p1 = np.exp(2j * np.pi * 2.4 * 5 / WL), np.exp(2j * np.pi * 1.9 * 5 / WL)
    assert {"in@0", "in@1", "out@0", "out@1"} == {p for k in S for p in k}
    assert np.abs(S[("in@0", "out@1")] - p0 * np.sqrt(0.3) * p1).max() < 1e-12
    assert np.abs(S[("in@0", "out@0")] - p0 * np.sqrt(0.7) * p0).max() < 1e-12


def test_sparameter_files_in_both_key_orders(tmp_path):
    wl_m = np.linspace(1.5e-6, 1.6e-6, 11)
    s21 = 0.9 * np.exp(1j * np.linspace(0, 3, 11))  # o1 -> o2
    s12 = 0.1 + 0j * wl_m  # o2 -> o1 (a non-reciprocal test device)
    # lightsim layout: metres, keys "out,in"
    np.savez(tmp_path / "ls.npz", wavelengths=wl_m, **{"o2@0,o1@0": s21, "o1@0,o2@0": s12})
    # gplugins / Tidy3D layout: micrometres, keys "in,out"
    np.savez(tmp_path / "gp.npz", wavelengths=wl_m * 1e6, **{"o1@0,o2@0": s21, "o2@0,o1@0": s12})
    for f in ("ls.npz", "gp.npz"):
        m = C.model_from_npz(tmp_path / f)
        sd = m(wl=wl_m * 1e6)
        assert np.allclose(sd[("o1", "o2")], s21) and np.allclose(sd[("o2", "o1")], s12)
        mid = m(wl=1.555)[("o1", "o2")]
        assert abs(abs(mid) - 0.9) < 1e-12  # polar interpolation keeps the magnitude


def test_straight_model_from_mode_solver_carries_the_group_index():
    from lightsim.xsection import CrossSection

    xs = CrossSection.strip(core_width=0.5e-6, step=25e-9)
    m = C.straight_from_cross_section(xs, wavelengths=(1.53, 1.55, 1.57))
    ng_solver = xs.solve(1.55e-6, 1, group_index=True)[0].n_group
    w = np.array([1.5499, 1.5501])
    n = m.neff(w).real
    ng_fit = m.neff(1.55).real - 1.55 * (n[1] - n[0]) / (w[1] - w[0])
    print(ng_fit, ng_solver)
    assert abs(ng_fit - ng_solver) < 5e-3
    ph = np.unwrap(np.angle(m(wl=WL, length=100.0)[("o1", "o2")]))
    assert abs(abs(ph[-1] - ph[0]) / (2 * np.pi) - 100 * abs(1 / WL[-1] * m.neff(WL[-1]).real - 1 / WL[0] * m.neff(WL[0]).real)) < 1e-6


def test_lightsim_models_and_circuits_agree_with_sax():
    sax = pytest.importorskip("sax")
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    # the models of gplugins' sax tests
    import jax.numpy as jnp

    def straight(wl=1.5, length=10.0, neff=2.4):
        return sax.reciprocal({("o1", "o2"): jnp.exp(2j * jnp.pi * neff * length / wl)})

    def mmi1x2():
        return sax.reciprocal({("o1", "o2"): 0.5**0.5, ("o1", "o3"): 0.5**0.5})

    def bend_euler(wl=1.5, length=20.0):
        return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}

    models = {"bend_euler": bend_euler, "mmi1x2": mmi1x2, "straight": straight}
    c = gf.components.mzi(delta_length=10)
    wl = np.linspace(1.5, 1.6, 51)
    s_sax = sax.circuit(netlist=c.get_netlist(), models=models)[0](wl=wl)
    s_ls = C.circuit(c.get_netlist(), models)[0](wl=wl)
    assert np.abs(np.asarray(s_sax[("o1", "o2")]) - s_ls[("o1", "o2")]).max() < 1e-12
    # lightsim models inside a SAX circuit
    ls_models = {"straight": C.straight, "bend_euler": C.bend, "mmi1x2": C.mmi1x2}
    s_sax2 = sax.circuit(netlist=c.get_netlist(), models=ls_models)[0](wl=wl)
    s_ls2 = C.circuit(c.get_netlist(), ls_models)[0](wl=wl)
    assert np.abs(np.asarray(s_sax2[("o1", "o2")]) - s_ls2[("o1", "o2")]).max() < 1e-12
