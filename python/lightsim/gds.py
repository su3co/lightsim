"""gdsfactory integration: turn a gdsfactory Component + LayerStack into
lightsim structures (extruded polygons on the right z ranges and materials).

Layers on the wafer layer (substrate, box, cladding) have no polygons in the
component; they are represented as slabs covering the whole simulation
region, which is how a layer stack describes the background.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import Box, Medium, PolySlab, Structure, um

# Refractive indices at 1550 nm (override with `material_index`).
MATERIAL_INDEX = {
    "si": 3.476,
    "silicon": 3.476,
    "sio2": 1.444,
    "oxide": 1.444,
    "sin": 1.996,
    "si3n4": 1.996,
    "nitride": 1.996,
    "air": 1.0,
    "vacuum": 1.0,
    "silica": 1.444,
    "h2o": 1.33,
    "polysilicon": 3.48,
    "ge": 4.27,
    "aln": 2.12,
    "ln": 2.21,
    "tio2": 2.25,
    "ta2o5": 2.07,
    "al2o3": 1.75,
    "water": 1.33,
}


# Media the core defines itself (see the Materials guide). A layer material
# with one of these names and no entry in the index table is left to the
# core; gold and silver are dispersive there.
CORE_MEDIA = frozenset({"vacuum", "air", "si", "silicon", "sio2", "silica", "oxide", "si3n4", "sin", "nitride", "si3n4_785", "sin_785", "water", "h2o", "au", "gold", "ag", "silver", "pec"})


def material_table(material_index: dict | None = None) -> dict:
    """:data:`MATERIAL_INDEX` with the caller's ``material_index`` on top
    (names in lower case)."""
    idx = dict(MATERIAL_INDEX)
    if material_index:
        idx.update({str(k).lower(): v for k, v in material_index.items()})
    return idx


def material_index_of(material: str, idx: dict, where: str) -> float:
    """Refractive index of ``material`` from the table ``idx``.

    Raises a ``ValueError`` that names ``where`` (the layer-stack level or
    the role of the material) and the material if the table has no entry.
    An unknown name used to become index 1 without a word.
    """
    if material in idx:
        return idx[material]
    known = ", ".join(sorted(idx))
    raise ValueError(
        f"{where}: material '{material}' has no refractive index. "
        f"Pass material_index={{'{material}': n}} (index at your wavelength). Known materials: {known}."
    )


def _medium_for(material: str, idx: dict, where: str):
    """The ``Medium`` of a layer material, or ``None`` if the core defines a
    medium of that name itself and the table has no entry for it."""
    if material not in idx and material in CORE_MEDIA:
        return None
    return Medium(material, n=material_index_of(material, idx, where))


@dataclass
class StackLayer:
    """One level of a layer stack as lightsim uses it.

    Attributes
    ----------
    name : str
        Name of the level in the layer stack.
    zmin, zmax : float
        Height range of the level (m).
    material : str
        Material name, in lower case.
    polygons : list
        Polygons of the level in the component, each a list of (x, y)
        vertices in metres; empty for a wafer level, which covers the whole
        region.
    mesh_order : int
        Painting priority from the layer stack: where patterned levels
        overlap, the lower value wins.
    """

    name: str
    zmin: float  # metres
    zmax: float
    material: str
    polygons: list  # list of (N, 2) vertex lists in metres; empty => full slab
    mesh_order: int


def _level_polygons(component, level) -> list:
    layer = level.layer
    dbu = component.kcl.dbu
    try:
        region = layer.get_shapes(component)
    except Exception:
        return []
    polys = []
    for poly in region.each():
        if poly.holes() > 0:
            # A polygon with holes (a ring drawn as one polygon) is cut into
            # hole-free pieces. The pieces are klayout SimplePolygons, which
            # have points but no hull / hole distinction. The rasterizer
            # paints the pieces of one level as a union, so the cuts between
            # them leave no trace.
            for piece in poly.decompose_trapezoids():
                pts = [(p.x * dbu * um, p.y * dbu * um) for p in piece.each_point()]
                if len(pts) >= 3:
                    polys.append(pts)
        else:
            polys.append([(p.x * dbu * um, p.y * dbu * um) for p in poly.each_point_hull()])
    return polys


DEFAULT_EXCLUDE = ("substrate", "via_contact", "metal1", "heater", "via1", "metal2", "via2", "metal3", "undercut")


def _is_wafer_level(name, level, wafer_levels) -> bool:
    """A level that covers the whole wafer instead of following polygons."""
    if wafer_levels is not None and name in wafer_levels:
        return True
    if bool(getattr(level, "background", False)):  # gdsfactory's own flag
        return True
    return "WAFER" in str(level.layer).upper()  # the generic PDK's wafer layer


def stack_layers(component, layer_stack=None, exclude=DEFAULT_EXCLUDE, wafer_levels=None) -> list[StackLayer]:
    """Collect the layer-stack levels with their polygons (in metres).

    Two kinds of level are returned. A **patterned** level has polygons in
    ``component``; it carries them. A **wafer** level covers the whole
    wafer (buried oxide, cladding); it carries no polygons and becomes a slab
    over the whole simulation region.

    Parameters
    ----------
    component
        A gdsfactory component.
    layer_stack
        A gdsfactory ``LayerStack``; default: the stack of the active PDK.
    exclude
        Names of levels to leave out, with or without polygons. Default:
        the substrate, the undercut, the heater, the vias and the metals.
    wafer_levels
        Names of levels to treat as wafer levels. A level is also a wafer
        level if it has ``background=True`` or if the name of its layer
        contains ``WAFER`` (the generic PDK). A wafer level on any other
        layer must be named here, because it looks like a patterned level
        that this component does not use.

    A level that is neither excluded nor a wafer level and has no polygons in
    the component is left out. If its height range is not covered by wafer
    levels, the background medium of the simulation will fill it; a
    ``UserWarning`` then names the level, because that is what a cladding on
    an unrecognised wafer layer looks like.

    Returns the list of :class:`StackLayer`, in the order of the stack.
    """
    import gdsfactory as gf

    if layer_stack is None:
        layer_stack = gf.pdk.get_layer_stack()
    names = set(layer_stack.layers)
    unknown = [n for n in (wafer_levels or ()) if n not in names]
    if unknown:
        raise ValueError(f"wafer_levels names {unknown}, which are not levels of the layer stack ({sorted(names)})")
    out = []
    empty = []  # (name, zmin, zmax, material) of levels left out for lack of polygons
    for name, lev in layer_stack.layers.items():
        if name in exclude:
            continue
        mat = str(lev.material).lower() if lev.material is not None else "air"
        th = float(lev.thickness)
        z0 = float(lev.zmin)
        zmin, zmax = (z0, z0 + th) if th >= 0 else (z0 + th, z0)
        is_wafer = _is_wafer_level(name, lev, wafer_levels)
        polys = [] if is_wafer else _level_polygons(component, lev)
        if not polys and not is_wafer:
            empty.append((name, zmin * um, zmax * um, mat))
            continue
        if is_wafer and tuple(getattr(lev, "background_exclude_layers", ()) or ()):
            import warnings

            warnings.warn(f"layer stack level '{name}': background_exclude_layers is not supported; the level covers the whole region", stacklevel=2)
        out.append(
            StackLayer(
                name=name,
                zmin=zmin * um,
                zmax=zmax * um,
                material=mat,
                polygons=polys,
                mesh_order=int(getattr(lev, "mesh_order", 3) or 3),
            )
        )
    # an empty level outside the wafer levels would have been a cladding
    slabs = sorted((l.zmin, l.zmax) for l in out if not l.polygons)
    uncovered = [e for e in empty if not _covered(e[1], e[2], slabs)]
    if uncovered:
        import warnings

        listed = ", ".join(f"'{n}' ({m}, z = {a / um:g} to {b / um:g} um)" for n, a, b, m in uncovered)
        warnings.warn(
            f"layer stack levels without polygons in this component were left out: {listed}. "
            "The background medium fills their height. If one of them should cover the whole wafer "
            "(a cladding or buried oxide on a custom layer), pass wafer_levels=[...] or set background=True on the level.",
            stacklevel=2,
        )
    return out


def _covered(z0: float, z1: float, slabs) -> bool:
    """True if [z0, z1] lies inside the union of the sorted intervals ``slabs``."""
    tol = 1e-12
    z = z0
    for a, b in slabs:
        if a > z + tol:
            break
        z = max(z, b)
        if z >= z1 - tol:
            return True
    return z >= z1 - tol


def component_to_structures(
    component,
    layer_stack=None,
    material_index: dict | None = None,
    region_min=None,
    region_max=None,
    exclude=None,
    wafer_levels=None,
):
    """Structures and media of a component for a lightsim ``Simulation``.

    Returns ``(structures, media, layers)``: the structure dicts in painting
    order, the media they use, and the :class:`StackLayer` list.

    Wafer slabs come first, patterned levels after (higher priority),
    ordered by the stack's mesh_order (lower mesh_order wins, as in
    gdsfactory/gplugins). ``region_min`` / ``region_max`` (metres) bound the
    wafer slabs in x and y; ``exclude`` and ``wafer_levels`` are passed to
    :func:`stack_layers`. ``material_index`` maps material names to
    refractive indices and overrides :data:`MATERIAL_INDEX`.
    """
    idx = material_table(material_index)
    kw = {} if exclude is None else {"exclude": exclude}
    layers = stack_layers(component, layer_stack, wafer_levels=wafer_levels, **kw)
    media = {}
    structures = []
    wafer = [l for l in layers if not l.polygons]
    pattern = [l for l in layers if l.polygons]
    pattern.sort(key=lambda l: -l.mesh_order)
    big = 1.0  # 1 m: effectively infinite in xy
    def register(l):
        if l.material not in media:
            m = _medium_for(l.material, idx, f"layer stack level '{l.name}'")
            if m is not None:
                media[l.material] = m

    for l in wafer:
        register(l)
        x0 = region_min[0] - 1e-3 if region_min is not None else -big
        y0 = region_min[1] - 1e-3 if region_min is not None else -big
        x1 = region_max[0] + 1e-3 if region_max is not None else big
        y1 = region_max[1] + 1e-3 if region_max is not None else big
        structures.append(Structure(Box([x0, y0, l.zmin], [x1, y1, l.zmax]), l.material, name=l.name))
    for l in pattern:
        register(l)
        for i, poly in enumerate(l.polygons):
            structures.append(Structure(PolySlab(poly, l.zmin, l.zmax, axis=2), l.material, name=f"{l.name}_{i}"))
    return structures, list(media.values()), layers


def port_info(component):
    """List of dicts (name, center[m], orientation, width[m]) for optical ports."""
    out = []
    for p in component.ports:
        if getattr(p, "port_type", "optical") != "optical":
            continue
        out.append(
            {
                "name": p.name,
                "center": (float(p.center[0]) * um, float(p.center[1]) * um),
                "orientation": float(p.orientation) % 360.0,
                "width": float(p.width) * um,
            }
        )
    return out
