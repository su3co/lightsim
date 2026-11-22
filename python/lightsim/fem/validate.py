"""Command line of the FEM validation suite: ``python -m lightsim.fem.validate [CASE ...]``."""

import sys

from .validation import CASES, run


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-l", "--list"):
        for name, fn in CASES.items():
            print(f"{name:28} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    unknown = [a for a in argv if a not in CASES]
    if unknown:
        print(f"unknown case(s) {unknown}; known: {list(CASES)}")
        return 2
    reps = run(argv or None)
    failed = [r.name for r in reps if not r.passed]
    print(f"{len(reps) - len(failed)}/{len(reps)} cases passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
