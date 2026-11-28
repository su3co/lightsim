"""Compare two outputs of `lightsim validate`. usage: python benchmarks/subpixel/validate_compare.py A.txt B.txt [--moved] [--cases]
Prints: number of checks with identical measured values, cases that pass / fail in each; with --moved, for the checks
whose value differs: error against the check's reference in A and in B (in units of the tolerance)."""
import re, sys
pat = re.compile(r"^\s+(.*?)\s+measured\s+(\S+)\s+expected\s+(\S+)\s+(abs|rel) err\s+(\S+) \(tol (\S+)\)")
def load(p):
    cases, cur, status = {}, None, {}
    for l in open(p):
        m = re.match(r"^(\S+) \[(PASS|FAIL|ERROR)\]", l)
        if m: cur = m.group(1); status[cur] = m.group(2); cases[cur] = []; continue
        m = pat.match(l)
        if m and cur: cases[cur].append((m.group(1), m.group(2), m.group(3), float(m.group(5)), float(m.group(6))))
    return cases, status
a, sa = load(sys.argv[1]); b, sb = load(sys.argv[2])
tot = same = 0; moved = []
for c in a:
    for i, chk in enumerate(a[c]):
        tot += 1
        o = b.get(c, [None] * len(a[c])); o = o[i] if i < len(o) else None
        if o is not None and o[1] == chk[1]: same += 1
        elif o is not None: moved.append((c, chk, o))
print(f"checks {tot}: identical measured values {same}, differing {tot - same}")
print(f"cases A: {sum(v == 'PASS' for v in sa.values())}/{len(sa)} pass; B: {sum(v == 'PASS' for v in sb.values())}/{len(sb)} pass; failing in B: {[c for c, v in sb.items() if v != 'PASS']}")
mc = sorted({m[0] for m in moved}); print(f"cases with at least one moved value: {len(mc)} of {len(a)}; unchanged cases: {len(a) - len(mc)}")
if "--moved" in sys.argv:
    better = worse = 0; fails = []
    for c, x, y in moved:
        ra, rb = x[3] / x[4], y[3] / y[4]
        if rb < ra: better += 1
        elif rb > ra: worse += 1
        if rb > 1: fails.append((c, x[0], x[3], y[3], x[4]))
    print(f"moved checks: error against the reference smaller in B: {better}, larger in B: {worse}")
    big = sorted(moved, key=lambda m: -(m[2][3] / m[2][4] - m[1][3] / m[1][4]))[:12]
    print("largest increases (error / tolerance, A -> B):")
    for c, x, y in big: print(f"  {c}: {x[0][:70]}: {x[3] / x[4]:.3f} -> {y[3] / y[4]:.3f}")
    big = sorted(moved, key=lambda m: (m[2][3] / m[2][4] - m[1][3] / m[1][4]))[:8]
    print("largest decreases:")
    for c, x, y in big: print(f"  {c}: {x[0][:70]}: {x[3] / x[4]:.3f} -> {y[3] / y[4]:.3f}")
    print("checks beyond tolerance in B:"); [print("  ", f) for f in fails]
if "--cases" in sys.argv:
    import collections
    per = collections.defaultdict(list)
    for c, x, y in moved: per[c].append((x, y))
    print("\ncase: moved checks | error against the reference summed over the moved checks, in units of each check's tolerance: A -> B | worst moved check in B (error, tolerance)")
    for c, v in per.items():
        sa_, sb_ = sum(x[3] / x[4] for x, y in v), sum(y[3] / y[4] for x, y in v)
        w = max(v, key=lambda t: t[1][3] / t[1][4])
        print(f"  {c}: {len(v)} of {len(a[c])} | {sa_:.3f} -> {sb_:.3f} ({sb_ / sa_ if sa_ else float('nan'):.2f}x) | {w[0][0][:48]}: {w[0][3]:.2e} -> {w[1][3]:.2e} (tol {w[0][4]:.0e})")
