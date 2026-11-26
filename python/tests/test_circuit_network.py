"""The network solve of lightsim.circuit: the join back end (compiled,
sub-network growth) against the dense matrix solve on a spread of
topologies and batch shapes, the fallback rules, threads, the frequency
sweep variable and the reuse of identical model calls."""

import numpy as np
import pytest

from lightsim import circuit as C


def rand_block(rng, n, shape, modes=1):
    ports = [f"o{k + 1}" for k in range(n)] if modes == 1 else [f"o{k + 1}@{m}" for k in range(n) for m in range(modes)]
    m = len(ports)
    A = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    A *= 0.95 / np.linalg.norm(A, 2)
    ph = rng.uniform(0, 30, size=(m, m))
    x = np.linspace(0, 1, int(np.prod(shape))).reshape(shape) if shape else np.float64(0.3)
    return {(a, b): A[j, i] * np.exp(1j * ph[j, i] * x) for i, a in enumerate(ports) for j, b in enumerate(ports)}


def chain(rng, n, shape):
    sd = {f"i{k}": rand_block(rng, 2, shape) for k in range(n)}
    pairs = [((f"i{k}", "o2"), (f"i{k + 1}", "o1")) for k in range(n - 1)]
    return sd, pairs, {"in": ("i0", "o1"), "out": (f"i{n - 1}", "o2")}


def rings(rng, n, shape):
    sd, pairs = {}, []
    for k in range(n):
        sd[f"c{k}"], sd[f"r{k}"] = rand_block(rng, 4, shape), rand_block(rng, 2, shape)
        pairs += [((f"c{k}", "o3"), (f"r{k}", "o1")), ((f"r{k}", "o2"), (f"c{k}", "o2"))]
        if k:
            pairs.append(((f"c{k - 1}", "o4"), (f"c{k}", "o1")))
    return sd, pairs, {"in": ("c0", "o1"), "out": (f"c{n - 1}", "o4")}


def lattice(rng, n, shape):
    sd, pairs, ports = {}, [], {}
    for i in range(n):
        for j in range(n):
            sd[f"g{i}_{j}"] = rand_block(rng, 4, shape)
            if j:
                pairs.append(((f"g{i}_{j - 1}", "o3"), (f"g{i}_{j}", "o1")))
            if i:
                pairs.append(((f"g{i - 1}_{j}", "o4"), (f"g{i}_{j}", "o2")))
    for i in range(n):
        ports[f"w{i}"], ports[f"e{i}"] = (f"g{i}_0", "o1"), (f"g{i}_{n - 1}", "o3")
    return sd, pairs, ports


def tree(rng, depth, shape):
    sd, pairs, ports = {"s0": rand_block(rng, 3, shape)}, [], {"in": ("s0", "o1")}
    level, k = ["s0"], 0
    for _ in range(depth - 1):
        nxt = []
        for s in level:
            for p in ("o2", "o3"):
                k += 1
                sd[f"s{k}"] = rand_block(rng, 3, shape)
                pairs.append(((s, p), (f"s{k}", "o1")))
                nxt.append(f"s{k}")
        level = nxt
    for s in level:
        ports[f"{s}a"], ports[f"{s}b"] = (s, "o2"), (s, "o3")
    return sd, pairs, ports


def random_graph(rng, n, shape):
    sizes = rng.integers(2, 6, size=n)
    sd = {f"d{k}": rand_block(rng, int(sizes[k]), shape) for k in range(n)}
    free = [(f"d{k}", f"o{p + 1}") for k in range(n) for p in range(sizes[k])]
    rng.shuffle(free)
    ports = {f"e{k}": free[k] for k in range(3)}
    rest = free[3:]
    return sd, [(rest[2 * k], rest[2 * k + 1]) for k in range(len(rest) // 2)], ports


def multimode(rng, n, shape):
    sd = {f"i{k}": rand_block(rng, 2, shape, modes=2) for k in range(n)}
    pairs = [((f"i{k}", "o2"), (f"i{k + 1}", "o1")) for k in range(n - 1)]
    return sd, pairs, {"in": ("i0", "o1"), "out": (f"i{n - 1}", "o2")}


CASES = [(chain, 12), (rings, 5), (lattice, 4), (tree, 4), (random_graph, 14), (multimode, 6), (chain, 1)]


@pytest.mark.parametrize("gen,size", CASES, ids=[f"{g.__name__}{s}" for g, s in CASES])
@pytest.mark.parametrize("shape", [(), (1,), (37,), (3, 70)], ids=["scalar", "one", "line", "grid"])
def test_joins_equal_dense(gen, size, shape):
    rng = np.random.default_rng(size * 31 + len(shape))
    sd, pairs, ports = gen(rng, size, shape)
    ref = C.solve_network(sd, pairs, ports, backend="dense")
    for be in ("joins", "auto"):
        S = C.solve_network(sd, pairs, ports, backend=be)
        assert S.keys() == ref.keys()
        for k in ref:
            assert np.shape(S[k]) == np.shape(ref[k]) == shape
            assert np.abs(S[k] - ref[k]).max() < 1e-12, (be, k)


def test_threads_and_broadcast_shapes():
    rng = np.random.default_rng(3)
    sd, pairs, ports = rings(rng, 4, (1, 50))
    sd["c1"] = rand_block(rng, 4, (6, 1))  # a different batch axis
    ref = C.solve_network(sd, pairs, ports, backend="dense")
    for t in (1, 3):
        S = C.solve_network(sd, pairs, ports, backend="joins", threads=t)
        for k in ref:
            assert S[k].shape == (6, 50)
            assert np.abs(S[k] - ref[k]).max() < 1e-12


def test_unconnected_ports_are_matched_and_disconnected_parts_zero():
    rng = np.random.default_rng(5)
    a, b = chain(rng, 3, (20,)), chain(rng, 4, (20,))
    sd = {**a[0], **{"x" + k: v for k, v in b[0].items()}}
    sd["i1"] = rand_block(rng, 3, (20,))  # o3 neither connected nor external
    pairs = a[1] + [(("x" + p[0], p[1]), ("x" + q[0], q[1])) for p, q in b[1]]
    ports = {**a[2], **{"x" + k: ("x" + v[0], v[1]) for k, v in b[2].items()}}
    ref = C.solve_network(sd, pairs, ports, backend="dense")
    S = C.solve_network(sd, pairs, ports, backend="joins")
    for k in ref:
        assert np.abs(S[k] - ref[k]).max() < 1e-12
    assert np.all(S[("in", "xout")] == 0)


def test_port_joined_twice_falls_back_to_the_matrix_solve():
    rng = np.random.default_rng(7)
    sd, pairs, ports = chain(rng, 3, (10,))
    pairs = pairs + [(("i0", "o2"), ("i2", "o1"))]  # i0,o2 now joined twice
    ref = C.solve_network(sd, pairs, ports, backend="dense")
    S = C.solve_network(sd, pairs, ports)  # auto
    for k in ref:
        assert np.abs(S[k] - ref[k]).max() < 1e-14
    with pytest.raises(ValueError):
        C.solve_network(sd, pairs, ports, backend="joins")
    with pytest.raises(ValueError):
        C.solve_network(sd, pairs, ports, backend="nope")


def test_circuit_backends_and_threads_agree():
    k, L = 0.05, 60.0
    models = {"dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=k), "wg": lambda wl=1.55, length=L: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=30.0)}
    net = {"instances": {"A": "dc", "L1": "wg"}, "connections": {"A,o3": "L1,o1", "L1,o2": "A,o2"}, "ports": {"o1": "A,o1", "o2": "A,o4"}}
    wl = np.linspace(1.5, 1.6, 301)
    ref = C.circuit(net, models, backend="dense")[0](wl=wl)
    for kw in ({}, {"backend": "joins"}, {"threads": 4}, {"threads": 4, "reuse_identical": True}):
        S = C.circuit(net, models, **kw)[0](wl=wl, L1={"length": np.array([[60.0], [61.0]])})
        R = C.circuit(net, models, backend="dense")[0](wl=wl, L1={"length": np.array([[60.0], [61.0]])})
        assert S[("o1", "o2")].shape == (2, 301)
        assert np.abs(S[("o1", "o2")] - R[("o1", "o2")]).max() < 1e-12
    assert np.abs(C.circuit(net, models)[0](wl=wl)[("o1", "o2")] - ref[("o1", "o2")]).max() < 1e-12


def test_frequency_sweep_variable():
    c = C._C_UM
    f = np.linspace(190e12, 200e12, 101)
    wl = c / f
    m_wl = {"wg": lambda wl=1.55, length=30.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length)}

    def wg_f(f=193e12, length=30.0):
        return C.straight(wl=c / np.asarray(f), neff=2.4, ng=4.2, length=length)

    net = {"instances": {"a": "wg", "b": "wgf"}, "connections": {"a,o2": "b,o1"}, "ports": {"in": "a,o1", "out": "b,o2"}}
    fn = C.circuit(net, {**m_wl, "wgf": wg_f})[0]
    ref = C.straight(wl=wl, neff=2.4, ng=4.2, length=60.0)[("o1", "o2")]
    for S in (fn(f=f), fn(wl=wl)):
        assert np.abs(S[("in", "out")] - ref).max() < 1e-9
    # a model that takes both keeps the old behaviour: f as given, wl at its value
    seen = {}

    def both(wl=1.55, f=1.0):
        seen["wl"], seen["f"] = wl, f
        return C.attenuator(wl=np.broadcast_to(np.asarray(f, float), np.shape(f)))

    C.circuit({"instances": {"x": "both"}, "ports": {"a": "x,o1", "b": "x,o2"}}, {"both": both})[0](f=f)
    assert seen["wl"] == 1.55 and seen["f"] is f


def test_reuse_identical_model_calls():
    calls = []

    def wg(wl=1.55, length=10.0):
        calls.append(length)
        return C.straight(wl=wl, neff=2.4, ng=4.2, length=length)

    inst = {f"w{k}": {"component": "wg", "settings": {"length": 10.0 if k % 2 else 20.0}} for k in range(6)}
    net = {"instances": inst, "connections": {f"w{k},o2": f"w{k + 1},o1" for k in range(5)}, "ports": {"in": "w0,o1", "out": "w5,o2"}}
    wl = np.linspace(1.5, 1.6, 51)
    ref = C.circuit(net, {"wg": wg})[0](wl=wl)
    assert len(calls) == 6
    fn = C.circuit(net, {"wg": wg}, reuse_identical=True)[0]
    calls.clear()
    S = fn(wl=wl)
    assert sorted(calls) == [10.0, 20.0]  # two distinct argument sets
    assert np.abs(S[("in", "out")] - ref[("in", "out")]).max() == 0
    calls.clear()
    fn(wl=wl.copy())  # same contents: nothing re-evaluated
    assert calls == []
    calls.clear()
    fn(wl=wl, w3={"length": 15.0})  # one instance changed
    assert calls == [15.0]
    calls.clear()
    w2 = wl.copy()
    fn(wl=w2)  # back to the first arguments: still cached
    assert calls == []
    w2[0] += 1e-3  # changed in place: contents differ, re-evaluated
    fn(wl=w2)
    assert sorted(calls) == [10.0, 20.0]


def test_stack_instances():
    calls = []

    def wg(wl=1.55, length=10.0, loss=0.0):
        calls.append(np.shape(length))
        return C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=loss)

    def scalar_only(wl=1.55, length=10.0):  # does not broadcast over length
        import math

        calls.append("s")
        return C.straight(wl=wl, neff=2.4, ng=4.2, length=math.fsum([length]))

    def summing(wl=1.55, length=10.0):  # broadcasts, but wrongly (reduces over all axes)
        calls.append("r")
        t = C.straight(wl=wl, neff=2.4, ng=4.2, length=10.0)[("o1", "o2")]
        return C.reciprocal({("o1", "o2"): t * np.exp(1j * np.mean(length))})

    def along_wl(wl=1.55, length=10.0):  # broadcasts correctly, but not point by point along wl:
        # the first-call check on a subsample of wl disagrees, the full check stacks it
        calls.append(np.shape(length))
        t = C.straight(wl=wl, neff=2.4, ng=4.2, length=length)[("o1", "o2")]
        return C.reciprocal({("o1", "o2"): t * np.mean(np.real(t), axis=-1, keepdims=True)})

    wl = np.linspace(1.5, 1.6, 41)
    for model, expect_stacked in ((wg, True), (scalar_only, False), (summing, False), (along_wl, True)):
        inst = {f"w{k}": {"component": "m", "settings": {"length": 10.0 + k}} for k in range(5)}
        net = {"instances": inst, "connections": {f"w{k},o2": f"w{k + 1},o1" for k in range(4)}, "ports": {"in": "w0,o1", "out": "w4,o2"}}
        ref = C.circuit(net, {"m": model})[0](wl=wl)
        fn = C.circuit(net, {"m": model}, stack_instances=True)[0]
        for rep in range(3):
            R = C.circuit(net, {"m": model})[0](wl=wl, w2={"length": 12.0 + 0.1 * rep}) if rep else ref
            calls.clear()
            S = fn(wl=wl, w2={"length": 12.0 + 0.1 * rep}) if rep else fn(wl=wl)
            assert np.abs(S[("in", "out")] - R[("in", "out")]).max() < 1e-12, model.__name__
            if rep and expect_stacked:
                assert len(calls) == 1 and calls[0] == (5, 1), calls  # one stacked call
            if rep and not expect_stacked:
                assert len(calls) == 5, (model.__name__, calls)  # one call per instance


def test_models_on_the_thread_pool(monkeypatch):
    monkeypatch.setattr(C, "_MODEL_POOL_MIN_POINTS", 0)  # use the pool whatever the size
    rng = np.random.default_rng(11)
    lengths = rng.uniform(5, 50, size=8)
    inst = {f"w{k}": {"component": "wg", "settings": {"length": float(lengths[k])}} for k in range(8)}
    net = {"instances": inst, "connections": {f"w{k},o2": f"w{k + 1},o1" for k in range(7)}, "ports": {"in": "w0,o1", "out": "w7,o2"}}
    models = {"wg": lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=3.0)}
    wl = np.linspace(1.5, 1.6, 201)
    ref = C.circuit(net, models, backend="dense")[0](wl=wl)
    for kw in ({"threads": 4}, {"threads": 4, "reuse_identical": True}):
        S = C.circuit(net, models, **kw)[0](wl=wl)
        assert np.abs(S[("in", "out")] - ref[("in", "out")]).max() < 1e-12


def _two_level_netlist():
    """top: two copies of cell 'stage' and one of 'pair' (which holds a
    'stage' itself), in series; every waveguide is model 'wg'."""
    stage = {"instances": {"a": {"component": "wg", "settings": {"length": 10.0}}, "b": {"component": "dc", "settings": {}}, "c": {"component": "wg", "settings": {"length": 25.0}}},
             "connections": {"a,o2": "b,o1", "b,o4": "c,o1"}, "ports": {"o1": "a,o1", "o2": "c,o2", "x": "b,o2"}}
    pair = {"instances": {"s": "stage", "w": {"component": "wg", "settings": {"length": 7.0}}}, "connections": {"s,o2": "w,o1"}, "ports": {"o1": "s,o1", "o2": "w,o2"}}
    top = {"instances": {"s1": "stage", "s2": "stage", "p": "pair", "a": {"component": "wg", "settings": {"length": 3.0}}},
           "connections": {"s1,o2": "s2,o1", "s2,o2": "p,o1", "p,o2": "a,o1"}, "ports": {"in": "s1,o1", "out": "a,o2", "tap": "s2,x"}}
    return {"top": top, "stage": stage, "pair": pair}


def test_flattened_hierarchy_equals_cell_by_cell(monkeypatch):
    def wg(wl=1.55, length=10.0, loss=0.0):
        return C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=loss)

    def dc(wl=1.55, coupling=0.3):
        return C.coupler_ideal(wl=wl, coupling=coupling)

    models = {"wg": wg, "dc": dc}
    wl = np.linspace(1.5, 1.6, 61)
    calls = [dict(), dict(loss=5.0), dict(b={"coupling": 0.1}), dict(a={"length": 4.0}, c={"length": 30.0}), dict(s1={"x": 1}, loss=2.0)]
    for kw in ({}, {"reuse_identical": True}, {"stack_instances": True, "reuse_identical": True}):
        for ov in calls:
            monkeypatch.setattr(C, "_FLATTEN", False)
            ref = C.circuit(_two_level_netlist(), models)[0](wl=wl, **ov)
            monkeypatch.setattr(C, "_FLATTEN", True)
            fn, info = C.circuit(_two_level_netlist(), models, **kw)
            S = fn(wl=wl, **ov)
            assert sorted(info["subcircuits"]) == ["pair", "stage"]
            assert S.keys() == ref.keys()
            for k in ref:
                assert np.abs(S[k] - ref[k]).max() < 1e-12, (kw, ov, k)


def test_importing_the_circuit_module_does_not_import_scipy_signal():
    import subprocess
    import sys

    code = "import sys, lightsim.circuit; print('scipy.signal' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def _ring_bus(n=6):
    inst, conn = {}, {}
    for k in range(n):
        inst[f"c{k}"] = {"component": "dc", "settings": {"coupling0": 0.1 + 0.02 * k}}
        inst[f"w{k}"] = {"component": "wg", "settings": {"length": 30.0 + 3 * k}}
        conn[f"c{k},o3"] = f"w{k},o1"
        conn[f"w{k},o2"] = f"c{k},o2"
        if k:
            conn[f"c{k - 1},o4"] = f"c{k},o1"
    return {"instances": inst, "connections": conn, "ports": {"in": "c0,o1", "out": f"c{n - 1},o4"}}


@pytest.fixture
def trace_all(monkeypatch):
    """Trace every model call, however small (the default traces calls of
    8192 points or more, and codes that are long or use the recurrence)."""
    monkeypatch.setattr(C, "_TRACE_MIN_POINTS", 0)
    monkeypatch.setattr(C, "_TRACE_MIN_CODE", 0)


def test_trace_models_equal_direct_models(monkeypatch, trace_all):
    traced = []
    orig = C._trace_call

    def counting(*a, **k):
        r = orig(*a, **k)
        traced.append(any(isinstance(v, C._Sym) for v in r[0].values()))
        return r

    monkeypatch.setattr(C, "_trace_call", counting)
    models = {"dc": lambda wl=1.55, coupling0=0.2: C.coupler(wl=wl, coupling0=coupling0, length=5.0), "wg": lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=20.0)}
    net = _ring_bus()
    wl = np.linspace(1.5, 1.6, 2001)
    ref = C.circuit(net, models)[0](wl=wl)
    for kw in ({}, {"reuse_identical": True}, {"threads": 3}):
        fn = C.circuit(net, models, trace_models=True, **kw)[0]
        for k in range(3):
            x = wl * (1 + 1e-13 * k)
            S, R = fn(wl=x), C.circuit(net, models)[0](wl=x)
            for key in R:
                assert np.abs(S[key] - R[key]).max() < 1e-10, (kw, key)
    # a two-dimensional batch (a parameter per row)
    L = np.array([[30.0], [31.0], [32.5]])
    S = C.circuit(net, models, trace_models=True)[0](wl=wl[None, :], w2={"length": L})
    R = C.circuit(net, models)[0](wl=wl[None, :], w2={"length": L})
    assert S[("in", "out")].shape == (3, 2001)
    assert np.abs(S[("in", "out")] - R[("in", "out")]).max() < 1e-10
    assert traced and all(traced)  # every model was traced, none fell back


def test_trace_models_single_model_to_rounding_and_numpy_semantics(trace_all):
    wl = np.linspace(1.4, 1.7, 501)

    def odd(wl=1.55, a=0.3):
        x = (wl - 1.55) * 10  # negative and positive
        return C.reciprocal({("o1", "o2"): np.exp(1j * np.pi * wl) * np.sqrt(np.abs(x)) + np.sqrt(x) * 0, ("o1", "o1"): np.where(x > 0, np.cos(x) + 0j, np.tanh(x) * a)})

    one = {"instances": {"x": "odd"}, "ports": {"a": "x,o1", "b": "x,o2"}}
    with np.errstate(invalid="ignore"):
        R = C.circuit(one, {"odd": odd}, backend="dense")[0](wl=wl)
    S = C.circuit(one, {"odd": odd}, trace_models=True)[0](wl=wl)
    for k in R:
        assert np.array_equal(np.isnan(S[k]), np.isnan(R[k])), k  # sqrt of a negative real is NaN, as in numpy
        good = ~np.isnan(R[k])
        assert np.abs(S[k][good] - R[k][good]).max() < 1e-14, k


def test_trace_models_fall_back_to_direct_calls(trace_all):
    calls = []

    def branchy(wl=1.55):  # control flow on the wavelength: not traceable
        calls.append(1)
        t = C.straight(wl=wl, neff=2.4, ng=4.2, length=20.0)[("o1", "o2")]
        if np.mean(wl) > 1.0:
            t = t * 0.5
        return C.reciprocal({("o1", "o2"): t})

    def indexing(wl=1.55):  # indexing a traced array: not traceable
        t = C.straight(wl=wl, neff=2.4, ng=4.2, length=20.0)[("o1", "o2")]
        return C.reciprocal({("o1", "o2"): t * t[0]})

    wl = np.linspace(1.5, 1.6, 101)
    for model in (branchy, indexing):
        one = {"instances": {"x": "m"}, "ports": {"a": "x,o1", "b": "x,o2"}}
        R = C.circuit(one, {"m": model})[0](wl=wl)
        fn = C.circuit(one, {"m": model}, trace_models=True)[0]
        for _ in range(2):
            S = fn(wl=wl)
            assert np.abs(S[("a", "b")] - R[("a", "b")]).max() < 1e-14


def test_trace_models_frequency_sweep(trace_all):
    c = C._C_UM

    def wg_f(f=193e12, length=30.0):
        xp = C._xp(f)
        return C.reciprocal({("o1", "o2"): xp.exp(2j * xp.pi * 2.4 * length * xp.asarray(f) / c)})

    net = {"instances": {"a": "wgf", "b": "wgf"}, "connections": {"a,o2": "b,o1"}, "ports": {"in": "a,o1", "out": "b,o2"}}
    f = np.linspace(190e12, 200e12, 301)
    R = C.circuit(net, {"wgf": wg_f})[0](f=f)
    S = C.circuit(net, {"wgf": wg_f}, trace_models=True)[0](f=f)
    assert np.abs(S[("in", "out")] - R[("in", "out")]).max() < 1e-12


def _traced_kinds(monkeypatch):
    """Record the instruction kinds of every trace."""
    kinds = []
    orig = C._trace_call

    def rec(*a, **k):
        out, tr = orig(*a, **k)
        kinds.append({c[0] for c in tr.code})
        return out, tr

    monkeypatch.setattr(C, "_trace_call", rec)
    return kinds


@pytest.mark.parametrize("phase", [1e2, 1e3, 1e4])
def test_trace_models_large_phase_bound(monkeypatch, trace_all, phase):
    """The documented accuracy of trace_models: about eps times the largest
    phase (the exponential recurrence on a uniform grid, in wavelength and
    in frequency), against the direct numpy call."""
    kinds = _traced_kinds(monkeypatch)
    wl = np.linspace(1.5, 1.6, 5001)
    eps = np.finfo(float).eps
    L = phase * 1.6 / (2 * np.pi * 2.4)  # largest phase 2 pi neff L / wl = phase

    def wg(wl=1.55, length=1.0):  # phase in 1/wl: the recurrence on a grid uniform in frequency
        return C.reciprocal({("o1", "o2"): 0.9 * np.exp(2j * np.pi * 2.4 * length / wl)})

    def wgx(wl=1.55, length=1.0):  # phase linear in wl: the recurrence on a grid uniform in wavelength
        return C.reciprocal({("o1", "o2"): 0.9 * np.exp(1j * length * 2.4 * wl)})

    one = {"instances": {"x": "m"}, "ports": {"a": "x,o1", "b": "x,o2"}}
    for model, x in ((wg, 1.0 / np.linspace(1 / 1.6, 1 / 1.5, 5001)), (wgx, wl)):
        R = C.circuit(one, {"m": model})[0](wl=x, x={"length": L})
        S = C.circuit(one, {"m": model}, trace_models=True)[0](wl=x, x={"length": L})
        err = np.abs(S[("a", "b")] - R[("a", "b")]).max() / 0.9
        assert err <= 16 * eps * phase, (model.__name__, phase, err)
    assert all(100 in k for k in kinds), kinds  # the recurrence was used


def test_trace_models_tiles_give_identical_results(monkeypatch, trace_all):
    """The model codes are evaluated in tiles of lanes: any tile size gives
    the same bits (rows of the batch cross tile boundaries)."""
    models = {"dc": lambda wl=1.55, coupling0=0.2: C.coupler(wl=wl, coupling0=coupling0, length=5.0), "wg": lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=20.0)}
    net = _ring_bus()
    wl = np.linspace(1.5, 1.6, 1001)[None, :]
    L = np.array([[30.0], [31.0], [32.5]])
    out = []
    for tile in ("7", "128", "100000"):
        monkeypatch.setenv("LIGHTSIM_MODEL_TILE", tile)
        out.append(C.circuit(net, models, trace_models=True)[0](wl=wl, w2={"length": L})[("in", "out")])
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[1], out[2])


_TRACE_THREADS_SCRIPT = """
import sys, numpy as np
from lightsim import circuit as C
models = {"dc": lambda wl=1.55, coupling0=0.2: C.coupler(wl=wl, coupling0=coupling0, length=5.0),
          "wg": lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=20.0)}
inst, conn = {}, {}
for k in range(6):
    inst[f"c{k}"] = {"component": "dc", "settings": {"coupling0": 0.1 + 0.02 * k}}
    inst[f"w{k}"] = {"component": "wg", "settings": {"length": 3000.0 + (300.0 if k % 2 else 0.0)}}
    conn[f"c{k},o3"] = f"w{k},o1"
    conn[f"w{k},o2"] = f"c{k},o2"
    if k:
        conn[f"c{k - 1},o4"] = f"c{k},o1"
net = {"instances": inst, "connections": conn, "ports": {"in": "c0,o1", "out": "c5,o4"}}
wl = 1 / np.linspace(1 / 1.6, 1 / 1.5, 2001)[None, :]  # uniform in 1/wl: the phases use the recurrence
L = 3000.0 + np.linspace(0.0, 10.0, 37)[:, None]  # 37 rows of 2001 points: rows cross chunk boundaries
res = []
for kw in ({}, {"reuse_identical": True}, {"stack_instances": True}, {"reuse_identical": True, "stack_instances": True}):
    fn = C.circuit(net, models, trace_models=True, **kw)[0]
    for k in range(3):  # the first call is direct and checks the trace
        S = fn(wl=wl * (1 + 1e-12 * k), w1={"length": L}, w4={"length": L + 1.0})
    res.append(np.asarray(S[("in", "out")]))
np.save(sys.argv[1], np.array(res))
"""


def test_trace_models_identical_for_any_thread_count(tmp_path):
    """The exponential recurrence of traced models restarts at row starts and
    at fixed multiples of the global point index, not at chunk boundaries
    (which follow the thread count): 1, 4, 16 and 64 threads give bitwise
    identical results, with and without stacking and identical-instance reuse."""
    import os
    import subprocess
    import sys

    res = []
    for t in (1, 4, 16, 64):
        o = tmp_path / f"t{t}.npy"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RAYON_NUM_THREADS=str(t))
        subprocess.run([sys.executable, "-c", _TRACE_THREADS_SCRIPT, str(o)], check=True, env=env, timeout=600)
        res.append(np.load(o))
    assert res[0].shape == (4, 37, 2001)
    for r in res[1:]:
        assert np.array_equal(r, res[0])


def test_trace_models_with_stacking_and_replay(monkeypatch, trace_all):
    """Stacked groups are traced as one call; with reuse_identical, a model
    whose other arguments are plain values is traced once and replayed with
    each new wavelength."""
    calls = []
    orig = C._trace_call

    def rec(m, kw, var):
        out, tr = orig(m, kw, var)
        if np.shape(kw[var])[-1] == wl.size:  # (not the subsample of the first-use check)
            calls.append(tuple(np.ndim(v) for k, v in sorted(kw.items()) if k != var))
        return out, tr

    monkeypatch.setattr(C, "_trace_call", rec)
    models = {"dc": lambda wl=1.55, coupling0=0.2: C.coupler(wl=wl, coupling0=coupling0, length=5.0), "wg": lambda wl=1.55, length=10.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=20.0)}
    net = _ring_bus()
    wl = np.linspace(1.5, 1.6, 2001)
    fn = C.circuit(net, models, trace_models=True, reuse_identical=True, stack_instances=True)[0]
    for k in range(4):
        x = wl * (1 + 1e-12 * k)
        S, R = fn(wl=x), C.circuit(net, models)[0](wl=x)
        assert np.abs(S[("in", "out")] - R[("in", "out")]).max() < 1e-10
    # from the second call on: one stacked trace per model (the stacked
    # argument has a leading axis), no instance traced alone
    later = calls[len(calls) // 2 :]
    assert later and all(any(d >= 2 for d in c) for c in later), calls
    # replay: a model with only plain arguments is traced once
    calls.clear()
    one = {"instances": {"x": "wg"}, "ports": {"a": "x,o1", "b": "x,o2"}}
    fn = C.circuit(one, models, trace_models=True, reuse_identical=True)[0]
    for k in range(3):
        x = wl * (1 + 1e-12 * k)
        S, R = fn(wl=x), C.circuit(one, models)[0](wl=x)
        assert np.abs(S[("a", "b")] - R[("a", "b")]).max() < 1e-12
    assert len(calls) == 1, calls


def test_trace_models_rechecked_on_a_uniform_grid_after_a_non_uniform_one(monkeypatch):
    """A model judged not worth tracing on a non-uniform sweep (no recurrence
    there, and a short code) is judged again on the first uniform sweep:
    a warm-up, then a refinement call on a non-uniform grid, then uniform
    calls, which used to stay on direct calls for good."""
    used = []
    orig = C._trace_call

    def counting(*a, **k):
        r = orig(*a, **k)
        used.append(any(isinstance(v, C._Sym) for v in r[0].values()))
        return r

    monkeypatch.setattr(C, "_trace_call", counting)
    models = {"wg": lambda wl=1.55, length=60.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=30.0)}
    net = {"instances": {"L1": "wg"}, "connections": {}, "ports": {"o1": "L1,o1", "o2": "L1,o2"}}
    fn = C.circuit(net, models, trace_models=True)[0]
    ref = C.circuit(net, models)[0]
    uniform = 1.0 / np.linspace(1 / 1.6, 1 / 1.5, 20001)  # uniform in frequency, as the recurrence needs
    refine = 1.0 / np.sort(np.concatenate([np.linspace(1 / 1.6, 1 / 1.5, 10001), np.linspace(1 / 1.559, 1 / 1.541, 9000)]))
    for x in (uniform[:101], refine):  # a short warm-up (not traced), then a non-uniform call
        assert np.array_equal(fn(wl=x)[("o1", "o2")], ref(wl=x)[("o1", "o2")])
    n = len(used)
    for k in range(3):  # uniform calls: traced and checked once, then the trace is used
        x = uniform * (1 + 1e-13 * k)
        assert np.abs(fn(wl=x)[("o1", "o2")] - ref(wl=x)[("o1", "o2")]).max() < 1e-10
    assert len(used) > n + 1 and all(used[n:])  # the uniform calls ran through the trace
    x = refine * (1 + 1e-13)
    assert np.abs(fn(wl=x)[("o1", "o2")] - ref(wl=x)[("o1", "o2")]).max() < 1e-10


def test_trace_models_first_call_is_direct_and_checked_on_a_subsample(monkeypatch, trace_all):
    """The first traced call of a model returns the direct result; its trace
    is checked on a uniform subsample of the sweep (a few hundred points),
    not on the whole call; later calls use the trace."""
    sizes = []
    orig = C._check_trace

    def rec(out, tr, ref, tol=None, sens=None):
        sizes.append(max(np.size(np.asarray(v)) for v in ref.values()))
        return orig(out, tr, ref, tol, sens)

    monkeypatch.setattr(C, "_check_trace", rec)
    models = {"dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=0.05), "wg": lambda wl=1.55, length=60.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=30.0)}
    net = {"instances": {"A": "dc", "L1": "wg"}, "connections": {"A,o3": "L1,o1", "L1,o2": "A,o2"}, "ports": {"o1": "A,o1", "o2": "A,o4"}}
    wl = np.linspace(1.5, 1.6, 20001)
    fn = C.circuit(net, models, trace_models=True)[0]
    ref = C.circuit(net, models)[0]
    assert np.array_equal(fn(wl=wl)[("o1", "o2")], ref(wl=wl)[("o1", "o2")])
    assert sizes and max(sizes) <= C._CHECK_POINTS
    x = wl * (1 + 1e-12)
    assert np.abs(fn(wl=x)[("o1", "o2")] - ref(wl=x)[("o1", "o2")]).max() < 1e-10


def _uniform_grid_numpy(var, base):
    t = var if base == "x" else 1.0 / var
    n = t.shape[-1]
    h = (t[..., -1:] - t[..., :1]) / (n - 1)
    dev = t - (t[..., :1] + h * np.arange(n))
    if not np.all(np.isfinite(t)) or np.any(h == 0) or np.max(np.abs(dev)) > 16 * np.finfo(float).eps * np.max(np.abs(t)):
        return None, None
    return t, h


def test_compiled_uniform_grid_test_equals_numpy():
    """The compiled grid test gives numpy's outcome, t and h bit for bit, on
    uniform, reciprocal-uniform, perturbed, non-finite and constant grids,
    small (in the calling thread) and large (threaded)."""
    if getattr(C._core(), "circuit_uniform_grid", None) is None:
        pytest.skip("extension without the compiled grid test")
    f = np.linspace(190.0, 200.0, 257)[None, :] + 0.37 * np.arange(300)[:, None]
    cases = [np.linspace(1.5, 1.6, 5001)[None, :], 299.792458 / f, np.linspace(1.5, 1.6, 300 * 257).reshape(300, 257)]
    bad = cases[2].copy()
    bad[17, 100] *= 1 + 1e-12
    nan = cases[1].copy()
    nan[3, 3] = np.nan
    cases += [bad, nan, np.full((4, 9), 2.0), np.linspace(1.0, 2.0, 3)]
    for var in cases:
        for base in ("x", "u"):
            C._GRIDS.clear()
            t, h = C._uniform_grid(var, base)
            t0, h0 = _uniform_grid_numpy(var, base)
            assert (t is None) == (t0 is None), (var.shape, base)
            if t0 is not None:
                assert t.tobytes() == np.ascontiguousarray(t0).tobytes() and h.tobytes() == np.ascontiguousarray(h0).tobytes()
    C._GRIDS.clear()


def test_real_values_give_the_same_bits_as_complex(monkeypatch, trace_all):
    """Float arrays reach the compiled solve as they are (no complex copy):
    the S-parameters are the same bits as with the conversion."""
    models = {"dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=0.05), "wg": lambda wl=1.55, length=60.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=30.0)}
    net = {"instances": {"A": "dc", "L1": "wg", "L2": "wg"}, "connections": {"A,o3": "L1,o1", "L1,o2": "L2,o1", "L2,o2": "A,o2"}, "ports": {"o1": "A,o1", "o2": "A,o4"}}
    wl = np.linspace(1.5, 1.6, 4001)[None, :] + 1e-3 * np.arange(12)[:, None]
    out = {}
    for real in (True, False):
        monkeypatch.setattr(C, "_REAL_VALUES", real)
        fn = C.circuit(net, models, trace_models=True, reuse_identical=True, stack_instances=True)[0]
        out[real] = [fn(wl=wl * (1 + 1e-13 * k)) for k in range(3)]
    for a, b in zip(out[True], out[False]):
        for key in a:
            assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes(), key
