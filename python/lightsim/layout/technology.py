"""Technology (layer-stack) files without gdsfactory.

One model, :class:`Technology` (levels and materials), and several readers:

* lightsim's YAML or JSON form, whose level fields are those of
  gdsfactory's ``LayerLevel`` (``layer``, ``thickness``, ``zmin``,
  ``material``, ``sidewall_angle``, ``mesh_order``)::

      version: 1
      wavelength_um: 1.55
      background: sio2            # optional: the medium around everything
      layers:
        box:  {zmin: -2.0, thickness: 2.0, material: sio2}     # no layer: covers the wafer
        core: {layer: [1, 0], zmin: 0.0, thickness: 0.22, material: si, mesh_order: 1}
        clad: {zmin: 0.0, thickness: 1.0, material: sio2}
      materials:
        si:   {index: 3.476}          # a constant index ...
        sio2: {lightsim: SiO2}        # ... or a lightsim.materials name

  (``layers`` may also be a list of tables with a ``name`` each);
* gamma's TOML (``stack.toml``): ``wavelength_um``, ``[[layer]]`` tables
  (``name``, optional ``gds = [layer, datatype]``, ``zmin``, ``thickness``,
  ``material``; ``color``, ``min_width``, ``min_space`` are ignored) and
  ``[[material]]`` tables (``name``, ``index``, optional ``lightsim``);
* a gdsfactory ``LayerStack`` (:func:`technology_from_gdsfactory`).

A level may name layers it ``subtract``-s (a gdsfactory derived layer
``A - B``, such as the generic PDK's core, ``WG - DEEP_ETCH -
SHALLOW_ETCH``); lightsim does not compute layer booleans and warns when a
cell has polygons on a subtracted layer.

All lengths are micrometres. A level without a layer covers the whole wafer
(buried oxide, cladding). A level's ``layer`` may be a list of layers: their
polygons are painted together.
:func:`read_lyp` reads layer names from a KLayout ``.lyp`` file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .gds import _layer


@dataclass
class Level:
    """One level of the stack (µm)."""

    name: str
    zmin: float
    thickness: float
    material: str
    layers: list = field(default_factory=list)  # [(layer, datatype), ...]; empty: a wafer level
    sidewall_angle: float = 0.0
    mesh_order: int = 3
    subtract: list = field(default_factory=list)  # layers a derived level removes (not computed: warned about)

    @property
    def zmax(self) -> float:
        """Top of the level (µm)."""
        return self.zmin + self.thickness


@dataclass
class Material:
    """A material: a constant ``index``, or a :mod:`lightsim.materials` name (``lightsim``)."""

    name: str
    index: float | None = None
    lightsim: str | None = None

    def n(self, wavelength_m: float) -> float:
        """Refractive index at ``wavelength_m``: the constant, or the library value."""
        if self.index is not None:
            return float(self.index)
        if self.lightsim:
            from .. import materials

            return float(materials.refractive_index(self.lightsim, wavelength_m).real)
        from ..gds import MATERIAL_INDEX

        if self.name.lower() in MATERIAL_INDEX:
            return float(MATERIAL_INDEX[self.name.lower()])
        raise ValueError(f"material {self.name!r} has neither an index nor a lightsim name")


@dataclass
class Technology:
    """Levels, materials, the design wavelength and the background medium."""

    levels: list
    materials: dict = field(default_factory=dict)
    wavelength_um: float = 1.55
    background: str | None = None
    version: int = 1

    def level(self, name: str) -> Level:
        """The level named ``name``."""
        for l in self.levels:
            if l.name == name:
                return l
        raise KeyError(name)

    def background_material(self) -> str:
        """``background`` if set, else the material of the wafer level that
        holds the middle of the patterned levels, else of the topmost wafer
        level, else ``"air"``."""
        if self.background:
            return self.background.lower()
        wafer = [l for l in self.levels if not l.layers]
        pat = [l for l in self.levels if l.layers]
        if pat and wafer:
            zc = sum(0.5 * (l.zmin + l.zmax) for l in pat) / len(pat)
            for w in reversed(wafer):
                if w.zmin < zc < w.zmax:
                    return w.material.lower()
        if wafer:
            return max(wafer, key=lambda w: w.zmax).material.lower()
        return "air"

    def material_index(self, wavelength_m: float | None = None) -> dict:
        """``{material name: index}`` at ``wavelength_m`` (default the stack's wavelength)."""
        lam = wavelength_m if wavelength_m is not None else self.wavelength_um * 1e-6
        return {k.lower(): m.n(lam) for k, m in self.materials.items()}

    def stack_layers(self, layout) -> list:
        """:class:`lightsim.gds.StackLayer` list for a :class:`Layout` (metres):
        wafer levels, and the patterned levels that have polygons in it."""
        from ..gds import StackLayer

        out = []
        for l in self.levels:
            z0, z1 = sorted((l.zmin, l.zmax))
            if not l.layers:
                out.append(StackLayer(l.name, z0 * 1e-6, z1 * 1e-6, l.material.lower(), [], l.mesh_order))
                continue
            polys = [[(float(x), float(y)) for x, y in p] for spec in l.layers for p in layout.polygons.get(spec, [])]
            cut = [s for s in l.subtract if layout.polygons.get(s)]
            if polys and cut:
                warnings.warn(f"level {l.name!r} subtracts layers {cut} that have polygons in this cell; lightsim does not compute layer booleans and paints the level unsubtracted", stacklevel=2)
            if polys:
                out.append(StackLayer(l.name, z0 * 1e-6, z1 * 1e-6, l.material.lower(), polys, l.mesh_order))
        return out

    def level_of_layer(self, layer) -> str | None:
        """Name of the level drawn on ``layer`` (``None`` if none is)."""
        layer = _layer(layer)
        for l in self.levels:
            if layer in l.layers:
                return l.name
        return None

    def to_dict(self) -> dict:
        """The YAML/JSON form (version 1)."""
        return {
            "version": self.version,
            "wavelength_um": self.wavelength_um,
            **({"background": self.background} if self.background else {}),
            "layers": {
                l.name: {
                    **({"layer": list(l.layers[0]) if len(l.layers) == 1 else [list(x) for x in l.layers]} if l.layers else {}),
                    "zmin": l.zmin,
                    "thickness": l.thickness,
                    "material": l.material,
                    **({"sidewall_angle": l.sidewall_angle} if l.sidewall_angle else {}),
                    **({"subtract": [list(x) for x in l.subtract]} if l.subtract else {}),
                    "mesh_order": l.mesh_order,
                }
                for l in self.levels
            },
            "materials": {k: ({"index": m.index} if m.index is not None else {}) | ({"lightsim": m.lightsim} if m.lightsim else {}) for k, m in self.materials.items()},
        }

    def write(self, path) -> Path:
        """Write the YAML (``.yaml``/``.yml``, needs PyYAML) or JSON form."""
        path = Path(path)
        d = self.to_dict()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(d, sort_keys=False))
        else:
            path.write_text(json.dumps(d, indent=1))
        return path


def _layers_of(v) -> list:
    if v is None:
        return []
    if isinstance(v, str) or (len(v) == 2 and all(isinstance(x, int) for x in v)):
        return [_layer(v)]
    return [_layer(x) for x in v]


def _from_dict(d: dict, source: str = "") -> Technology:
    version = int(d.get("version", 1))
    if version != 1:
        raise ValueError(f"{source}: technology version {version}; this lightsim reads version 1")
    layers = d.get("layers", d.get("layer", []))
    if isinstance(layers, dict):
        layers = [{"name": k, **v} for k, v in layers.items()]
    levels = []
    for i, t in enumerate(layers):
        th = float(t["thickness"])
        levels.append(
            Level(
                name=str(t["name"]),
                zmin=float(t.get("zmin", 0.0)),
                thickness=th,
                material=str(t.get("material", "air")),
                layers=_layers_of(t.get("layer", t.get("gds"))),
                sidewall_angle=float(t.get("sidewall_angle", 0.0) or 0.0),
                mesh_order=int(t.get("mesh_order", 3) or 3),
                subtract=_layers_of(t.get("subtract")),
            )
        )
    mats = d.get("materials", d.get("material", []))
    if isinstance(mats, dict):
        mats = [{"name": k, **v} for k, v in mats.items()]
    materials = {str(m["name"]): Material(str(m["name"]), m.get("index"), m.get("lightsim")) for m in mats}
    return Technology(levels, materials, float(d.get("wavelength_um", 1.55)), d.get("background"), version)


def read_technology(path) -> Technology:
    """Read a technology file: ``.toml`` (gamma's stack), ``.yaml``/``.yml``
    or ``.json`` (lightsim's form)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10
            try:
                import tomli as tomllib
            except ModuleNotFoundError as e:
                raise ImportError("reading a TOML technology file on Python 3.10 needs tomli (pip install tomli); Python 3.11 and later read it without it") from e
        d = tomllib.loads(path.read_text())
    elif suf in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading a YAML technology file needs PyYAML (pip install pyyaml); TOML and JSON need nothing") from e
        d = yaml.safe_load(path.read_text())
    else:
        d = json.loads(path.read_text())
    return _from_dict(d, str(path))


def technology_from_gdsfactory(layer_stack=None, material_index: dict | None = None, wavelength_um: float = 1.55, exclude=None) -> Technology:
    """Convert a gdsfactory ``LayerStack`` (default: the active PDK's).

    Levels on the wafer layer or with ``background=True`` become wafer
    levels; levels on a derived layer are left out (with a warning); the
    levels lightsim skips by default (substrate, metals, vias) are left out
    unless ``exclude`` says otherwise. Materials get the index from
    ``material_index`` or :data:`lightsim.gds.MATERIAL_INDEX`.
    """
    import gdsfactory as gf

    from ..gds import DEFAULT_EXCLUDE, MATERIAL_INDEX, _is_wafer_level

    ls = layer_stack or gf.get_active_pdk().layer_stack
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    idx = {**MATERIAL_INDEX, **{k.lower(): v for k, v in (material_index or {}).items()}}
    levels, skipped = [], []
    for name, lev in ls.layers.items():
        if name in exclude:
            continue
        layers, subtract = [], []
        if not _is_wafer_level(name, lev, None):
            try:
                base, subtract = _derived(lev.layer, gf)
            except ValueError:
                skipped.append(name)
                continue
            layers = [base]
        levels.append(Level(name, float(lev.zmin), float(lev.thickness), str(lev.material).lower(), layers, float(getattr(lev, "sidewall_angle", 0.0) or 0.0), int(getattr(lev, "mesh_order", 3) or 3), subtract))
    if skipped:
        warnings.warn(f"levels on derived layers other than A - B - ... left out: {', '.join(skipped)}", stacklevel=2)
    mats = {l.material: Material(l.material, idx.get(l.material)) for l in levels}
    return Technology(levels, mats, wavelength_um)


def _derived(layer, gf):
    """``(base, [subtracted, ...])`` of a gdsfactory layer: a logical layer, or
    a derived layer of the form ``A - B - C``; anything else raises ValueError."""
    if type(layer).__name__ == "DerivedLayer":
        if layer.operation not in ("-", "not"):
            raise ValueError(layer.operation)
        base, sub = _derived(layer.layer1, gf)
        b2, s2 = _derived(layer.layer2, gf)
        if s2:
            raise ValueError("nested")
        return base, sub + [b2]
    inner = getattr(layer, "layer", layer)
    return tuple(int(x) for x in gf.get_layer_tuple(inner)), []


def read_lyp(path) -> dict:
    """Layer names of a KLayout layer-properties file: ``{(layer, datatype): name}``.

    Groups are walked; entries with wildcards (``*/*``) are skipped."""
    import re
    import xml.etree.ElementTree as ET

    out = {}
    root = ET.parse(path).getroot()
    for p in [e for e in root.iter() if e.tag in ("properties", "group-members")]:
        src = (p.findtext("source") or "").strip()
        m = re.match(r"^(\d+)/(\d+)", src)
        if m:
            name = (p.findtext("name") or "").strip()
            out[(int(m.group(1)), int(m.group(2)))] = name or f"{m.group(1)}/{m.group(2)}"
    return out
