"""Gradient against central finite differences on the 2-D inverse-design scene at 1.50 / 1.55 / 1.60 um, with the port
profiles of the centre frequency and with profiles that follow the band. Objective per wavelength: |S21|^2 + |S11|^2 / 2
of a grey design; derivative along a smooth direction over the whole design, and on single samples.
usage: PYTHONPATH=<lightsim>/python python3 band_gradient_check.py [engine,engine..] [device] [r]"""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import numpy as np
import two_normalisations_2d as m
engines = (sys.argv[1] if len(sys.argv) > 1 else "python,native,reverse").split(",")
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
r = int(sys.argv[3]) if len(sys.argv) > 3 else 1
for engine in engines:
    for pp in ("center", "band"):
        sc = m.Scene(straight=False, r=r, normalize="monitor", engine=engine, device=dev, port_profiles=pp)
        f = m.smooth_noise(np.random.default_rng(m.SEED), sc.n, 3.0)
        eps = sc.with_perimeter(m.EPS_OX + (m.EPS_SI - m.EPS_OX) / (1.0 + np.exp(-2.0 * f)))
        d = m.smooth_noise(np.random.default_rng(5), sc.n, 4.0); d[0, :] = d[-1, :] = 0.0; d[:, 0] = d[:, -1] = 0.0; d /= np.abs(d).max()
        def loss(e):
            S = np.asarray(sc.problem.solve([e], adjoint=False).S)
            return np.abs(S[0, 0]) ** 2 + 0.5 * np.abs(S[1, 0]) ** 2
        sol = sc.problem.solve([eps], adjoint=True); S = np.asarray(sol.S)
        h = 0.2
        # along the direction: Richardson from two steps, since the objective is not linear over 0.05 of permittivity
        f1 = (loss(eps + 0.05 * d) - loss(eps - 0.05 * d)) / 0.1; f2 = (loss(eps + 0.025 * d) - loss(eps - 0.025 * d)) / 0.05
        fd = (4 * f2 - f1) / 3
        pts = [(14, 20), (27, 27), (40, 33), (22, 41)]
        fdp = []
        for (i, j) in pts:
            e1 = eps.copy(); e1[i, j] += h; e2 = eps.copy(); e2[i, j] -= h
            fdp.append((loss(e1) - loss(e2)) / (2 * h))
        fdp = np.array(fdp)
        out = []
        for k in range(3):
            g = np.zeros_like(S); g[0, 0, k] = 2 * S[0, 0, k]; g[1, 0, k] = S[1, 0, k]
            G = np.asarray(sol.vjp(g)[0]).reshape(sc.n, sc.n)
            ad = float((G * d).sum()); adp = np.array([G[i, j] for (i, j) in pts])
            out.append(f"{100 * abs(ad - fd[k]) / abs(fd[k]):.3f} % / {100 * np.abs(adp - fdp[:, k]).max() / np.abs(G).max():.3f} %")
        print(f"{engine:8s} port_profiles={pp:6s} {45 / r:g} nm on {dev}: direction / worst single sample (of the largest derivative) at 1.50, 1.55, 1.60 um: " + "  |  ".join(out), flush=True)
