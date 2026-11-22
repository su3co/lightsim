"""fdtd-z (spinsphotonics/fdtdz, with pjz): does the kernel run on this GPU?

    CUDA_VISIBLE_DEVICES=0 python fdtdz_probe.py

fdtd-z ships kernels precompiled for compute capabilities 3.7 to 8.0 and
launch-parameter presets for four GPU models (Quadro RTX 4000, Tesla T4,
V100, A100). For any other GPU the layout of its systolic update scheme,
``((blocku, blockv), (gridu, gridv), spacing, (cc_major, cc_minor))``, has
to be given by hand. This script first calls it with the default (None) and
then with the A100 preset, on a small uniform block with a dipole-like
source, and prints what happens.
"""

from __future__ import annotations

import json
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

import fdtdz_jax
import pjz


def attempt(launch, reduced):
    xx, yy, zz = 120, 120, (96 if reduced else 32)
    eps = jnp.full((3, xx, yy, zz), 2.25, dtype=jnp.float32)
    omega = jnp.array([2 * np.pi / 30.0], dtype=jnp.float32)
    src = jnp.zeros((2, 1, yy, zz), dtype=jnp.float32).at[0, 0, yy // 2, zz // 2].set(1.0)
    sp = pjz.SimParams(omega_range=(float(omega[0]), float(omega[0])), tt=3000, absorption_padding=40, pml_widths=(16, 16), use_reduced_precision=reduced, launch_params=launch)
    t0 = time.perf_counter()
    f = jax.block_until_ready(pjz.field(eps, src, omega, source_pos=30, sim_params=sp))
    t1 = time.perf_counter()
    f = jax.block_until_ready(pjz.field(eps, src, omega, source_pos=30, sim_params=sp))
    t2 = time.perf_counter()
    cells = (xx + 80) * (yy + 80) * (128 if reduced else 64)
    return {"shape": list(f.shape), "max_abs": float(jnp.abs(f).max()), "finite": bool(jnp.all(jnp.isfinite(f))), "first_seconds": t1 - t0, "second_seconds": t2 - t1, "cells_per_second": cells * 3000 / (t2 - t1)}


def main():
    dev = jax.devices()[0]
    print(json.dumps({"fdtdz_jax": getattr(fdtdz_jax, "__version__", "?"), "pjz": pjz.__version__, "jax": jax.__version__, "device": dev.device_kind}))
    for launch, name in ((None, "default (None)"), (((2, 4), (12, 9), 4, (8, 0)), "A100 preset ((2,4),(12,9),4,(8,0))")):
        for reduced in (False, True):
            try:
                out = attempt(launch, reduced)
                print(json.dumps({"launch_params": name, "reduced_precision": reduced, "ok": True, **out}), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"launch_params": name, "reduced_precision": reduced, "ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}", "trace_tail": traceback.format_exc().strip().splitlines()[-1][:300]}), flush=True)


if __name__ == "__main__":
    main()
