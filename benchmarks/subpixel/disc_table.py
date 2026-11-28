"""Table of the disc resonance study. usage: python benchmarks/subpixel/disc_table.py benchmarks/subpixel/results/disc.jsonl"""
import json, sys, collections
d = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    r = json.loads(l); d[(r["pol"], r["dx_nm"], r["weighting"])][r["shift"]] = r["rel_err"]
print("pol  dx   weighting  err(s=0)   err(0.25)  err(0.5)   max|err|   spread over the shifts   (relative error of the resonance frequency)")
for (pol, dx, wt), e in sorted(d.items(), key=lambda kv: (kv[0][0] != "tm", -kv[0][1], kv[0][2])):
    g = lambda s: f"{e[s]:+.2e}" if s in e else "    -    "
    print(f"{pol}  {dx:4.0f}  {wt:5s}     {g(0.0)}  {g(0.25)}  {g(0.5)}  {max(abs(x) for x in e.values()):.2e}   {max(e.values()) - min(e.values()):.2e}  ({len(e)} shifts)")
