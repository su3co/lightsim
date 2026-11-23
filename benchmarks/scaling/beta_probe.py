"""Figure of merit of a checkpointed design at several projection strengths."""
import sys
import numpy as np
import mesh2d as M
from lightsim import invdes
from lightsim.spec import C0
from validate_mesh import composite

ck = np.load(sys.argv[1])
N, D = int(sys.argv[2]), int(sys.argv[3])
freqs = np.array([C0 / (M.LAM0 * 1e-6)])
_, blocks = M.mesh_network(N, 4, D)
design = invdes.CompositeDesign([M.block_design(bl["n"]) for bl in blocks if bl["designed"]])
cp = composite(N, 4, D, freqs, ["cpu"] * 4, M.STUB, False)
for key in ("rho", "best_rho"):
    rho = ck[key]
    for b in (8, 12, 16, 32):
        S = cp.solve(design.eps(rho, b), adjoint=False).S
        m = invdes.matrix_metrics(S, M.dft_target(N))
        print(key, b, round(float(invdes.matrix_fidelity(S, M.dft_target(N))[0]), 4), round(float(m["fidelity"][0]), 3), round(float(m["throughput"][0]), 3), "binarization", round(design.binarization(rho, b), 3), flush=True)
