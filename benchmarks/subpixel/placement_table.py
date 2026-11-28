"""Table of the placement study: per case, mode, cell size and weighting: error against the reference at shift 0, 0.25, 0.5
and the range over the five shifts. usage: python benchmarks/subpixel/placement_table.py benchmarks/subpixel/results/placement.jsonl"""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
ref, names = {}, {}
data = collections.defaultdict(dict)
for r in rows:
    if "reference" in r: ref[r["case"]] = r["reference"]; names[r["case"]] = r["names"]; continue
    data[(r["case"], r["dx_nm"], r["weighting"])][r["shift"]] = r["neff"]
cases = []
for k in data:
    if k[0] not in cases: cases.append(k[0])
for c in cases:
    nm = names.get(c, ["TE0", "TM0"]); rf = ref.get(c)
    qs = [(i, nm[i]) for i in range(len(nm))]
    if c == "coupled_wires": qs = [("split", "even - odd")]
    for qi, qn in qs:
        def val(v): return v[0] - v[1] if qi == "split" else v[qi]
        r0 = (rf[0] - rf[1] if qi == "split" else rf[qi]) if rf else None
        print(f"\n{c} {qn}: reference {r0 if r0 is None else round(r0, 6)}")
        print("  dx    weighting  err(s=0)   err(0.25)  err(0.5)   max|err|   spread(max-min over 5 shifts)")
        for (cc, dx, wt), d in sorted(data.items(), key=lambda kv: (-kv[0][1], kv[0][2])):
            if cc != c: continue
            v = {s: val(x) for s, x in d.items()}
            base = r0 if r0 is not None else None
            e = {s: (x - base) if base is not None else x for s, x in v.items()}
            sp = max(v.values()) - min(v.values())
            f = (lambda x: f"{x:+.2e}") if base is not None else (lambda x: f"{x:.6f}")
            print(f"  {dx:4.0f}  {wt:5s}     {f(e[0.0])}  {f(e[0.25])}  {f(e[0.5])}  {max(abs(x) for x in e.values()):.2e}   {sp:.2e}")
