"""The worked example of docs/guide/dimensions.md, timed: the gdsfactory 1x2 MMI
in 2.5-D and in plain 2-D on the CPU and on cuda:0, and in 3-D on cuda:0.
Each run is timed once after a warm-up run. Run from the repository root."""
import time, json, numpy as np, gdsfactory as gf
from lightsim import eim, sparams
gf.gpdk.PDK.activate()
mmi = gf.components.mmi1x2()
wl = np.linspace(1.5e-6, 1.6e-6, 11)
common = dict(wavelengths=wl, steps_per_wavelength=18, input_ports=["o1"], return_results=True)
out = {}
def rep(name, fn):
    fn()  # warm-up
    t0 = time.perf_counter(); sp, runs = fn(); t = time.perf_counter() - t0
    st = runs["o1"].stats
    out[name] = dict(seconds=t, split=[float(v) for v in np.abs(sp["o2@0,o1@0"][[0, 5, 10]]) ** 2], neff=float(np.real(sp["mode_neff"]["o1"])), cells=int(st["cells"]), steps=int(st["steps"]), backend=st["backend"], sim_seconds=float(st["seconds"]))
    print(name, out[name], flush=True)
for dev in ("cpu", "cuda:0"):
    rep(f"2.5-D {dev}", lambda: eim.write_sparameters_2d(mmi, device=dev, **common))
    rep(f"2-D {dev}", lambda: eim.write_sparameters_2d(mmi, device=dev, level_indices={"core": 3.476}, background_index=1.444, **common))
rep("3-D cuda:0", lambda: sparams.write_sparameters(mmi, device="cuda:0", **common))
json.dump(out, open("benchmarks/two_d/results/twod_apps/three_ways.json", "w"), indent=1)
