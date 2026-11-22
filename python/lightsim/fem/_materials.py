"""Material property lookup for the finite-element solvers."""

from __future__ import annotations

import warnings

#: Heater and contact metals (thin films differ from bulk; pass your own
#: values when known): thermal conductivity k (W/(m K)), electrical
#: conductivity sigma (S/m), specific heat cp (J/(kg K)), density rho
#: (kg/m^3). Bulk metals: CRC Handbook of Chemistry and Physics (300 K).
#: TiN: thin-film values used in the gplugins/femwell heater examples
#: (k = 28, sigma = 2.3e6); film conductivity depends on the process.
METALS = {
    "TiN": {"k": 28.0, "sigma": 2.3e6, "cp": 600.0, "rho": 5220.0},
    "Al": {"k": 237.0, "sigma": 3.77e7, "cp": 897.0, "rho": 2700.0},
    "Cu": {"k": 401.0, "sigma": 5.96e7, "cp": 385.0, "rho": 8960.0},
    "W": {"k": 173.0, "sigma": 1.79e7, "cp": 132.0, "rho": 19300.0},
    "Au": {"k": 318.0, "sigma": 4.10e7, "cp": 129.0, "rho": 19300.0},
    "NiCr": {"k": 11.3, "sigma": 9.1e5, "cp": 450.0, "rho": 8400.0},
}
_METAL_ALIASES = {
    "tin": "TiN", "titanium_nitride": "TiN", "al": "Al", "aluminum": "Al", "aluminium": "Al", "cu": "Cu", "copper": "Cu",
    "w": "W", "tungsten": "W", "au": "Au", "gold": "Au", "nicr": "NiCr", "nichrome": "NiCr",
}
#: Gases and vacuum.
GASES = {"air": {"k": 0.026, "cp": 1005.0, "rho": 1.18}, "vacuum": {"k": 0.0, "cp": 0.0, "rho": 0.0}}


def metal(name: str):
    key = _METAL_ALIASES.get(str(name).strip().lower().replace(" ", "_"))
    return METALS.get(key) if key else None


def thermal(name: str) -> dict:
    """{"k", "cp", "rho", "dndT"} of a material name (library, metals, air)."""
    from ..materials import thermal_properties

    m = metal(name)
    if m is not None:
        return {"k": m["k"], "cp": m["cp"], "rho": m["rho"], "dndT": None}
    key = str(name).strip().lower()
    if key in GASES:
        return dict(GASES[key], dndT=0.0)
    try:
        return thermal_properties(name)
    except KeyError:
        pass
    # gdsfactory generic names
    alias = {"oxide": "SiO2", "sio2": "SiO2", "silica": "SiO2", "polysilicon": "Si", "poly": "Si", "sin": "Si3N4", "nitride": "Si3N4", "ln": "LiNbO3"}.get(key)
    if alias:
        return thermal_properties(alias)
    raise KeyError(f"no thermal properties for material {name!r}; pass conductivity={{'{name}': k}} (and capacity for transients)")


#: Refractive indices near 1550 nm for names the dispersion library lacks.
_INDEX_FALLBACK = {"air": 1.0, "vacuum": 1.0, "polysilicon": 3.48, "poly": 3.48, "aln": 2.12, "tio2": 2.25, "ta2o5": 2.07, "water": 1.33, "h2o": 1.33}


def refractive_index(name: str, wavelength: float, table: dict | None = None) -> complex:
    """Refractive index of a material at ``wavelength`` (m): the caller's
    table first (case-insensitive), then lightsim's dispersion library,
    then gdsfactory's generic names at 1550 nm."""
    from .. import materials
    from ..gds import MATERIAL_INDEX

    if table:
        low = {str(k).lower(): v for k, v in table.items()}
        if name in table:
            return complex(table[name])
        if str(name).lower() in low:
            return complex(low[str(name).lower()])
    key = str(name).strip().lower()
    if key in _INDEX_FALLBACK:
        return complex(_INDEX_FALLBACK[key])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return complex(materials.refractive_index(name, wavelength))
    except KeyError:
        pass
    if key in MATERIAL_INDEX:
        return complex(MATERIAL_INDEX[key])
    raise KeyError(f"no refractive index for material {name!r}; pass it in the index table, e.g. {{'{name}': 1.5}}")


#: Low-frequency relative permittivities (diagonal (xx, yy, zz) in the
#: crystal frame for anisotropic crystals). SiO2: thermal oxide; Si, GaAs,
#: Ge: Ioffe Institute semiconductor database; Si3N4: LPCVD film (typical);
#: Al2O3: sapphire (perpendicular / parallel to c); LiNbO3: unclamped,
#: Weis & Gaylord, Appl. Phys. A 37, 191 (1985).
STATIC_PERMITTIVITY = {
    "SiO2": 3.9, "Si": 11.7, "Si3N4": 7.5, "GaAs": 12.9, "Ge": 16.2,
    "Al2O3": (9.4, 9.4, 11.6), "LiNbO3": (85.0, 85.0, 28.7), "air": 1.0, "vacuum": 1.0,
}


def static_permittivity(name: str):
    """Low-frequency relative permittivity (scalar or crystal-frame diagonal)."""
    from ..materials import _resolve

    key = str(name).strip().lower()
    if key in ("air", "vacuum"):
        return 1.0
    alias = {"oxide": "SiO2", "sin": "Si3N4", "nitride": "Si3N4", "polysilicon": "Si", "ln": "LiNbO3"}.get(key)
    try:
        canon = alias or _resolve(name)
    except KeyError:
        raise KeyError(f"no static permittivity for {name!r}; pass permittivity={{'{name}': eps}}") from None
    canon = "LiNbO3" if canon.startswith("LiNbO3") else "Al2O3" if canon.startswith("Al2O3") else canon
    return STATIC_PERMITTIVITY[canon]
