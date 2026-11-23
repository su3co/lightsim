"""Material property lookup for the finite-element solvers.

Every default here is checked against the source named next to it. A
quantity without a confirmed source is absent: the solvers then raise an
error that asks for it (pass ``conductivity=``, ``capacity=``, ``dndT=``,
``permittivity=``, ``Current(I, sigma=...)``).
"""

from __future__ import annotations

import warnings

#: Thermal conductivity (W/(m K)) of heater and contact metals at 300 K:
#: recommended values for well-annealed high-purity bulk metals of C. Y. Ho,
#: R. W. Powell, P. E. Liley, J. Phys. Chem. Ref. Data 1, 279 (1972)
#: (Al p. 305, Cu p. 332, Au p. 340, W p. 408). Thin films conduct less.
#: Electrical conductivities, heat capacities and densities are not given:
#: pass them (``Current(I, sigma=...)``, ``capacity=``). TiN and other
#: film alloys have no default.
METALS = {
    "Al": {"k": 237.0},
    "Cu": {"k": 401.0},
    "Au": {"k": 317.0},
    "W": {"k": 174.0},
}
_METAL_ALIASES = {
    "al": "Al", "aluminum": "Al", "aluminium": "Al", "cu": "Cu", "copper": "Cu",
    "w": "W", "tungsten": "W", "au": "Au", "gold": "Au",
}


def metal(name: str):
    key = _METAL_ALIASES.get(str(name).strip().lower().replace(" ", "_"))
    return METALS.get(key) if key else None


_ALIAS = {"oxide": "SiO2", "sio2": "SiO2", "silica": "SiO2", "sin": "Si3N4", "nitride": "Si3N4"}


def thermal(name: str) -> dict:
    """{"k", "cp", "rho", "dndT"} of a material name (library or metal);
    missing quantities are ``None``."""
    from ..materials import thermal_properties

    m = metal(name)
    if m is not None:
        return {"k": m["k"], "cp": None, "rho": None, "dndT": None}
    key = str(name).strip().lower()
    try:
        return thermal_properties(_ALIAS.get(key, name))
    except KeyError:
        pass
    raise KeyError(f"no thermal properties for material {name!r}; pass conductivity={{'{name}': k}} (and capacity= for transients)")


def need(name: str, quantity: str, what: str, argument: str) -> float:
    """A thermal quantity of a material, or an error naming the argument to pass."""
    v = thermal(name).get(quantity)
    if v is None:
        raise KeyError(f"no confirmed {what} for material {name!r} in lightsim's tables; pass {argument}={{'{name}': value}}")
    return float(v)


def refractive_index(name: str, wavelength: float, table: dict | None = None) -> complex:
    """Refractive index of a material at ``wavelength`` (m): the caller's
    table first (case-insensitive), then lightsim's dispersion library, then
    lightsim.gds.MATERIAL_INDEX. Air is taken as 1 (vacuum)."""
    from .. import materials
    from ..gds import MATERIAL_INDEX

    if table:
        low = {str(k).lower(): v for k, v in table.items()}
        if name in table:
            return complex(table[name])
        if str(name).lower() in low:
            return complex(low[str(name).lower()])
    key = str(name).strip().lower()
    if key in ("air", "vacuum"):
        return complex(1.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return complex(materials.refractive_index(name, wavelength))
    except KeyError:
        pass
    if key in MATERIAL_INDEX:
        return complex(MATERIAL_INDEX[key])
    raise KeyError(f"no refractive index for material {name!r}; pass it in the index table, e.g. {{'{name}': 1.5}}")


#: Relative permittivities for electrostatics (scalar, or the crystal-frame
#: diagonal of an anisotropic crystal). SiO2 3.9 and Si 11.7: D. Zhu et al.,
#: Adv. Opt. Photon. 13, 242 (2021), Table 1 (RF dielectric constant); Si
#: 11.7, GaAs 12.9 (static), Ge 16.2: Ioffe Institute database, 300 K;
#: LiNbO3 (44, 44, 27.9): clamped, high-frequency, Zhu et al. Table 1 (the
#: low-frequency unclamped values are higher, about 83 and 29). Vacuum is 1;
#: air is taken as 1 (its permittivity is 1.0005).
STATIC_PERMITTIVITY = {
    "SiO2": 3.9, "Si": 11.7, "GaAs": 12.9, "Ge": 16.2,
    "LiNbO3": (44.0, 44.0, 27.9), "air": 1.0, "vacuum": 1.0,
}


def static_permittivity(name: str):
    """Relative permittivity for electrostatics (scalar or crystal-frame diagonal)."""
    from ..materials import _resolve

    key = str(name).strip().lower()
    if key in ("air", "vacuum"):
        return 1.0
    try:
        canon = _ALIAS.get(key) or _resolve(name)
    except KeyError:
        canon = None
    canon = "LiNbO3" if canon and canon.startswith("LiNbO3") else canon
    if canon not in STATIC_PERMITTIVITY:
        raise KeyError(f"no confirmed static permittivity for {name!r}; pass permittivity={{'{name}': eps}}")
    return STATIC_PERMITTIVITY[canon]
