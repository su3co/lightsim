"""Table of wire_fdtd.py: index of the wire's TE0 from the phase of a 3-D FDTD run minus the reference 2.43750, per cell
size, weighting and z placement. usage: python benchmarks/subpixel/wire_fdtd_table.py results/wire_fdtd_gb10.jsonl"""
import json, sys, collections
REF = 2.43750
d = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    r = json.loads(l); d[(r["dx_nm"], r.get("weighting", "cell"))][r["zshift"]] = (r["neff_fdtd"] - REF, r["n_mode_port"] - REF, r["s_abs"])
print("dx    weighting  FDTD index error at z shift 0, 0.25, 0.5 cells      port-mode index error at the same shifts     |S21| (shift 0)")
for (dx, wt), v in sorted(d.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
    g = lambda i: "  ".join(f"{v[s][i]:+.2e}" if s in v else "    -    " for s in (0.0, 0.25, 0.5))
    print(f"{dx:4.0f}  {wt:5s}      {g(0)}      {g(1)}      {v[0.0][2]:.4f}")
