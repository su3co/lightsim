"""S-parameters of a gdsfactory MMI 1x2 with lightsim (GPU if available).

    python examples/gds_mmi_sparams.py --device cuda:0
"""

import argparse

import gdsfactory as gf
import numpy as np

from lightsim.sparams import write_sparameters

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--steps-per-wavelength", type=float, default=18)
a = ap.parse_args()

gf.gpdk.PDK.activate()
c = gf.components.mmi1x2()
wl = np.linspace(1.5e-6, 1.6e-6, 11)
sp = write_sparameters(c, wavelengths=wl, steps_per_wavelength=a.steps_per_wavelength, device=a.device, input_ports=["o1"])
print("port mode n_eff:", {k: round(v.real, 4) for k, v in sp["mode_neff"].items()})
for k in ("o1@0,o1@0", "o2@0,o1@0", "o3@0,o1@0"):
    print(k, np.round(np.abs(sp[k]) ** 2, 4))
