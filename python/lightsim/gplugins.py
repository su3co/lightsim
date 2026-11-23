"""gplugins-compatible front: run gdsfactory components through lightsim with
the calls gdsfactory users already write for Meep.

gdsfactory's simulator adapters live in the separate package gplugins
(``gplugins.gmeep``, ``gplugins.tidy3d``, ``gplugins.lumerical``, ...). This
module gives lightsim the same entry points, argument names, units, defaults,
return value and cache files as ``gplugins.gmeep``, so that a script switches
solver by changing an import::

    import gplugins.gmeep as gm                  # before
    import lightsim.gplugins as gm               # after
    sp = gm.write_sparameters_meep(component, resolution=30, is_3d=False)

Every Meep name is also available under a lightsim name
(``write_sparameters_meep`` = :func:`write_sparameters_lightsim`, ...).

What is the same as ``gplugins.gmeep``:

* the result is a ``dict`` of numpy arrays: ``"wavelengths"`` in micrometres
  and one complex array per S-parameter, keyed ``"o2@0,o1@0"`` = S(o2 mode 0
  <- o1 mode 0) (output port first, as in gplugins.gmeep);
* it is written to an ``.npz`` file whose name is a hash of the component and
  the settings (``gplugins.common.utils.get_sparameters_path`` when gplugins
  is installed, an identical built-in otherwise), with a ``.yml`` file of the
  settings next to it, and read back instead of simulated on the next call
  unless ``overwrite=True``;
* all lengths in micrometres; sources and monitors where gplugins.gmeep puts
  them (``port_source_offset``, ``port_monitor_offset``,
  ``distance_source_to_monitors``), so the phase reference planes match;
* ``port_symmetries`` in gplugins' format, ``port_source_names``,
  ``port_source_modes``, ``port_modes``, the margins, ``layer_stack``,
  ``material_name_to_meep``, ``dispersive``, ``decay_by``, ``run=False``.

What lightsim adds: ``device`` (``"auto"``, ``"cpu"``, ``"cuda:0"``,
``"cuda:0,1"`` to split each run over two GPUs, or a list such as
``["cuda:0", "cuda:1"]`` to run the sources concurrently, one per GPU),
``effective_index`` (the 2.5-D effective-index model for ``is_3d=False``),
``polarization`` and ``steps_per_wavelength`` (a graded mesh instead of the
uniform ``resolution``).

Arguments that only make sense for Meep (``animate``, ``lazy_parallelism``,
``plot_args``, ``eps_averaging``, ...) are accepted and ignored, with one
:class:`IgnoredArgumentWarning` that names them.
"""

from __future__ import annotations

import hashlib
import inspect
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from . import gds as _gds
from .modes import solve_modes
from .spec import C0, GaussianPulse, ModeMonitor, ModeSource, PML, Simulation, UniformMesh, um

__all__ = [
    "GMode",
    "IgnoredArgumentWarning",
    "Waveguide",
    "WaveguideCoupler",
    "coupling_length",
    "find_coupling_vs_gap",
    "find_mode_dispersion",
    "find_modes_coupler",
    "find_modes_waveguide",
    "find_neff_ng_dw_dh",
    "find_neff_vs_width",
    "get_simulation",
    "get_sparameters_path",
    "model_from_npz",
    "port_symmetries_1x1",
    "port_symmetries_crossing",
    "sweep_bend_mismatch",
    "sweep_coupling_length",
    "sweep_fraction_te",
    "sweep_mode_area",
    "sweep_n_eff",
    "sweep_n_group",
    "write_sparameters_grating_lightsim",
    "write_sparameters_grating_lightsim_batch",
    "write_sparameters_lightsim",
    "write_sparameters_lightsim_1x1",
    "write_sparameters_lightsim_1x1_bend90",
    "write_sparameters_lightsim_batch",
    "write_sparameters_lightsim_batch_1x1",
    "write_sparameters_lightsim_batch_1x1_bend90",
    "write_sparameters_lightsim_mpi",
    "write_sparameters_lightsim_mpi_1x1",
    "write_sparameters_lightsim_mpi_1x1_bend90",
]

TOOL = "lightsim"

#: gplugins' port symmetries (``gplugins.common.utils.port_symmetries``).
port_symmetries_1x1 = {"o2@0,o1@0": ["o1@0,o2@0"]}
port_symmetries_crossing = {
    "o2@0,o1@0": ["o1@0,o2@0", "o3@0,o4@0", "o4@0,o3@0"],
    "o3@0,o1@0": ["o1@0,o3@0", "o2@0,o4@0", "o4@0,o2@0"],
    "o4@0,o1@0": ["o1@0,o4@0", "o2@0,o3@0", "o3@0,o2@0"],
}


class IgnoredArgumentWarning(UserWarning):
    """An argument of the gplugins.gmeep interface that lightsim does not use."""


# Keyword arguments ``write_sparameters_meep`` passes on to Meep's
# ``get_simulation`` and ``meep.Simulation`` (Meep 1.35, gplugins 2.1).
_MAPPED_SETTINGS = {
    "extend_ports_length",  # -> length of the port extensions
    "tpml",  # -> PML thickness
    "clad_material",  # -> background medium
    "distance_source_to_monitors",  # -> source-port monitor position
    "dfcen",  # -> source bandwidth
    "material_name_to_meep",  # -> material table
    "material_name_to_lightsim",  # (same, lightsim's name)
    "exclude_layers",  # -> levels left out
    "zmargin_top",
    "zmargin_bot",
}
_IGNORED_SETTINGS = {
    "continuous_source", "normal_2d", "point_2d", "port_source_mode", "port_source_name",
    # meep.Simulation
    "Courant", "accurate_fields_near_cylorigin", "allow_3d_subpixel", "bfast_scaled_k", "boundary_layers", "cell_size",
    "chunk_layout", "collect_stats", "default_material", "dimensions", "ensure_periodicity", "eps_averaging",
    "epsilon_func", "epsilon_input_file", "extra_materials", "filename_prefix", "force_all_components",
    "force_complex_fields", "geometry", "geometry_center", "k_point", "kz_2d", "loop_tile_base_db", "loop_tile_base_eh",
    "m", "material_function", "num_chunks", "output_single_precision", "output_volume", "progress_interval",
    "sources", "split_chunks_evenly", "subpixel_maxeval", "subpixel_tol", "symmetries",
}  # fmt: skip
#: Named arguments of ``write_sparameters_meep`` that have no meaning here,
#: with the default that does not trigger the warning.
_IGNORED_ARGS = {
    "animate": False,
    "animate_center": None,
    "animate_size": None,
    "lazy_parallelism": False,
    "z": 0,
    "plot_args": None,
    "padding_layers": ("PADDING",),
}


def _warn_ignored(names, where: str) -> None:
    if names:
        warnings.warn(
            f"{where}: lightsim ignores {', '.join(sorted(names))} (Meep-only options; the results do not depend on them)",
            IgnoredArgumentWarning,
            stacklevel=3,
        )


# ----------------------------------------------------------------------------
# cache paths (gplugins.common.utils.get_sparameters_path)
# ----------------------------------------------------------------------------


def _kwargs_hash(**kwargs) -> str:
    from gdsfactory.name import clean_value

    s = "_".join(f"{k}={clean_value(kwargs[k])}" for k in sorted(kwargs))
    return hashlib.md5(s.encode()).hexdigest()


def _component_hash(component) -> str:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        gdspath = Path(component.write_gds(gdsdir=tmp, no_empty_cells=True, with_metadata=False))
        return hashlib.md5(gdspath.read_bytes()).hexdigest()


def _builtin_sparameters_path(component, dirpath=None, **kwargs) -> Path:
    """The path ``gplugins.common.utils.get_sparameters_path`` returns, without gplugins."""
    import gdsfactory as gf
    from gdsfactory.config import GDSDIR_TEMP

    dirpath = Path(dirpath or GDSDIR_TEMP / "sparameters")
    component = gf.get_component(component)
    if hasattr(component, "function_name"):
        dirpath = dirpath / component.name
    h = hashlib.md5((_component_hash(component) + _kwargs_hash(**kwargs)).encode()).hexdigest()
    dirpath.mkdir(exist_ok=True, parents=True)
    return dirpath / f"{component.name}_{h}.npz"


def get_sparameters_path(component, dirpath=None, **kwargs) -> Path:
    """S-parameter ``.npz`` path for ``component`` and the simulation settings.

    The file name is a hash of the component's layout and of the settings, as
    in gplugins (``get_sparameters_path_meep`` and its siblings), with
    ``tool="lightsim"``. gplugins' own function is used when gplugins is
    installed; otherwise an identical built-in gives the same path.
    """
    kwargs.setdefault("tool", TOOL)
    try:
        from gplugins.common.utils.get_sparameters_path import _get_sparameters_path
    except ImportError:
        return _builtin_sparameters_path(component, dirpath, **kwargs)
    return _get_sparameters_path(component=component, dirpath=dirpath, **kwargs)


# ----------------------------------------------------------------------------
# materials
# ----------------------------------------------------------------------------


def _material_index(mapping: dict | None, wavelength_m: float) -> tuple[dict, dict]:
    """``(material_index, library)`` from a gplugins material mapping.

    A number is a refractive index; a string names a material of
    :mod:`lightsim.materials` (Meep's names ``"Si"``, ``"SiO2"``,
    ``"Si3N4"`` are among them), whose index at ``wavelength_m`` is used and
    which ``dispersive=True`` replaces by its pole fit. ``library`` maps the
    layer-stack material name to that library name.
    """
    from . import materials

    index, library = {}, {}
    for name, value in (mapping or {}).items():
        key = str(name).lower()
        if isinstance(value, str):
            library[key] = value
            index[key] = float(np.real(materials.refractive_index(value, wavelength_m)))
        elif np.iscomplexobj(value):
            raise ValueError(f"material {name!r}: give a real index; lossy media need dispersive=True or a lightsim Medium")
        else:
            index[key] = float(value)
    return index, library


def _library_name(material: str, library: dict) -> str | None:
    from . import materials

    name = library.get(material, material)
    try:
        materials._resolve(name)
    except KeyError:
        return None
    return name


def _make_dispersive(sim, library: dict, wl_m) -> list[str]:
    """Replace the media that :mod:`lightsim.materials` knows by pole fits over the band."""
    from . import materials

    band = (0.95 * float(np.min(wl_m)), 1.05 * float(np.max(wl_m)))
    done, media = [], []
    for m in sim.media:
        lib = _library_name(m["name"], library)
        if lib is None or m["name"].startswith("eim_"):
            media.append(m)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            media.append(materials.fitted_medium(lib, band=band, medium_name=m["name"]))
        done.append(m["name"])
    sim.media = media
    return done


def _exclude_levels(layer_stack, exclude_layers) -> list[str] | None:
    """Layer-stack level names of the layers in ``exclude_layers`` (Meep's argument)."""
    if not exclude_layers:
        return None
    import gdsfactory as gf

    targets = {gf.get_layer(l) for l in exclude_layers}
    names = []
    for name, level in layer_stack.layers.items():
        layer = getattr(level.layer, "layer", level.layer)
        try:
            if gf.get_layer(layer) in targets:
                names.append(name)
        except Exception:  # derived layers have no single layer index
            continue
    return list(_gds.DEFAULT_EXCLUDE) + names


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------


def _region(component, margins):
    """Region corners (m) in x and y: the component's bounding box plus the margins."""
    left, right, top, bot = margins
    bb = component.dbbox()
    return [bb.left * um - left, bb.bottom * um - bot], [bb.right * um + right, bb.top * um + top]


def _build_simulation(
    component,
    wl_m,
    layer_stack,
    material_index,
    *,
    is_3d,
    effective_index,
    polarization,
    resolution,
    steps_per_wavelength,
    margins,
    zmargin_top,
    zmargin_bot,
    extend_ports_length,
    tpml,
    clad_material,
    exclude,
    decay_by,
):
    """The lightsim simulation without sources and monitors, and the port z range."""
    import gdsfactory as gf

    from .layout._build import build_from_levels, mesh_and_pml

    mesh, pml_layers, pml_len = mesh_and_pml(wl_m, resolution, steps_per_wavelength, tpml)
    region = _region(component, margins)
    ext = max(max(margins) + pml_len + 1e-6, (extend_ports_length or 0.0) * um)
    kw = {} if exclude is None else {"exclude": exclude}
    extended = gf.components.extend_ports(component, length=ext / um)
    if is_3d:
        levels = _gds.stack_layers(extended, layer_stack, **kw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            levels = _gds.stack_layers(extended, layer_stack, **kw)
    return build_from_levels(
        levels,
        region,
        wl_m,
        material_index,
        is_3d=is_3d,
        effective_index=effective_index,
        polarization=polarization,
        mesh=mesh,
        pml_layers=pml_layers,
        zmargin_top=zmargin_top,
        zmargin_bot=zmargin_bot,
        background=clad_material.lower(),
        decay_by=decay_by,
    )


def _source_plan(port_names, sources, port_symmetries, port_modes):
    """The (source, mode) runs needed, given gplugins-style ``port_symmetries``.

    A source is skipped when every entry of its column is a copy that the
    symmetries give from the runs before it.
    """
    runs, filled = [], set()
    for src, mode in sources:
        column = {f"{p}@{m},{src}@{mode}" for p in port_names for m in port_modes}
        if port_symmetries and column <= filled:
            continue
        runs.append((src, mode))
        computed = {f"{p}@{m},{src}@{mode}" for p in port_names for m in port_modes}
        filled |= computed
        for key, copies in (port_symmetries or {}).items():
            if key in filled:
                filled |= set(copies)
    return runs


def get_simulation(component, port_source_name: str = "o1", port_source_mode: int = 0, **kwargs) -> Simulation:
    """The lightsim simulation of one source, with its sources and monitors.

    The counterpart of ``gplugins.gmeep.get_simulation``; the arguments are
    those of :func:`write_sparameters_lightsim`. Look at it with
    ``sim.view()`` or run it with ``sim.run()``.
    """
    return write_sparameters_lightsim(
        component, port_source_names=[port_source_name], port_source_modes={port_source_name: [port_source_mode]}, run=False, **kwargs
    )


def write_sparameters_lightsim(
    component,
    port_source_names: list[str] | None = None,
    port_source_modes: dict[str, list] | None = None,
    port_modes: list[int] | None = None,
    port_symmetries: dict | None = None,
    resolution: int = 30,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    wavelength_points: int = 50,
    dirpath=None,
    layer_stack=None,
    port_margin: float = 2,
    port_monitor_offset: float = -0.1,
    port_source_offset: float = -0.1,
    filepath=None,
    overwrite: bool = False,
    animate: bool = False,
    animate_center=None,
    animate_size=None,
    lazy_parallelism: bool = False,
    run: bool = True,
    dispersive: bool = False,
    xmargin: float = 0,
    ymargin: float = 0,
    zmargin: float = 0,
    xmargin_left: float = 0,
    xmargin_right: float = 0,
    ymargin_top: float = 0,
    ymargin_bot: float = 0,
    zmargin_top: float = 0,
    zmargin_bot: float = 0,
    decay_by: float = 1e-3,
    is_3d: bool = False,
    z: float = 0,
    plot_args: dict | None = None,
    only_return_filepath_sim_settings: bool = False,
    verbosity: int = 0,
    padding_layers=("PADDING",),
    *,
    device="auto",
    effective_index: bool = True,
    polarization: str | None = None,
    steps_per_wavelength: float | None = None,
    **settings,
) -> dict[str, np.ndarray]:
    r"""S-parameters of a gdsfactory component with lightsim, the gplugins way.

    Same arguments, units, defaults, result and cache files as
    ``gplugins.gmeep.write_sparameters_meep``: one simulation per source port
    and mode, a mode source ``port_source_offset`` outside each source port
    (negative: inside the component), mode monitors ``port_monitor_offset``
    outside every port and ``distance_source_to_monitors`` (0.2 um) behind
    the source on the source port, and ``S = a_out / a_in`` of the
    projections on the port modes. Lengths in micrometres.

    Parameters
    ----------
    component
        gdsfactory component (or the name of one).
    port_source_names
        Ports to excite (default: all).
    port_source_modes
        ``{port: [mode, ...]}`` to excite (default: mode 0 of each).
    port_modes
        Modes measured at every port (default ``[0]``).
    port_symmetries
        gplugins' format, ``{"o2@0,o1@0": ["o1@0,o2@0"]}``: the entries
        listed copy the key. A source whose whole column follows from the
        runs before it by these copies is not run.
    resolution
        Cells per micrometre of the uniform mesh.
    wavelength_start, wavelength_stop, wavelength_points
        The band (um).
    dirpath, filepath, overwrite
        Where the ``.npz`` goes (default: a name hashed from the component
        and the settings, in ``dirpath``), and whether an existing file is
        computed again.
    layer_stack
        Default: the active PDK's.
    port_margin
        The mode windows extend this far beside the port, at most half the
        distance to a neighbouring port.
    port_monitor_offset, port_source_offset
        um, positive outwards.
    run
        ``False`` returns the simulation of the first source instead.
    dispersive
        3-D: the materials of :mod:`lightsim.materials` (``si``, ``sio2``,
        ``sin``, ...) become pole fits over the band instead of constant
        indices.
    xmargin, ymargin, zmargin, xmargin_left, ...
        Distance from the component to the absorber. A z margin of 0 is
        replaced by 0.8 um.
    decay_by
        The runs stop when the field energy has decayed by this factor from
        its peak (lightsim's ``shutoff``).
    is_3d
        ``False`` runs the 2-D model chosen by ``effective_index``.
    only_return_filepath_sim_settings
        Return the path of the ``.yml`` settings file only.
    verbosity
        Above 0 prints the progress of each run.
    device
        lightsim only. ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``"cuda:0,1"``
        (each run split over two GPUs) or a list of device strings, over
        which the sources run concurrently.
    effective_index
        lightsim only, for ``is_3d=False``. ``True``: the 2.5-D
        effective-index model (each level gets the effective index of its
        slab mode). ``False``: the plain 2-D cut with the bulk indices,
        which is what Meep's 2-D run simulates.
    polarization
        lightsim only, 2-D. ``"te"`` (E in the chip plane, the default) or
        ``"tm"``. With ``effective_index=False`` and no polarization the
        mode of highest effective index is used, as Meep's ``eig_band=1``
        with no parity does.
    steps_per_wavelength
        lightsim only. A graded mesh with this many steps per wavelength in
        each material instead of ``resolution``.
    settings
        Meep's ``get_simulation`` and ``meep.Simulation`` keywords.
        ``extend_ports_length``, ``tpml``, ``clad_material``,
        ``distance_source_to_monitors``, ``dfcen``,
        ``material_name_to_meep`` and ``exclude_layers`` are used, the
        others are ignored with a warning, unknown names raise.

    Returns
    -------
    dict
        ``{"wavelengths": um, "o1@0,o1@0": complex array, ...}``.
    """
    import gdsfactory as gf
    from gdsfactory.pdk import get_layer_stack

    component = component if isinstance(component, gf.Component) else gf.get_component(component)
    layer_stack = layer_stack or get_layer_stack()

    unknown = [k for k in settings if k not in _MAPPED_SETTINGS | _IGNORED_SETTINGS]
    if unknown:
        raise ValueError(f"{unknown} not in the settings write_sparameters_lightsim accepts: {sorted(_MAPPED_SETTINGS | _IGNORED_SETTINGS)}")
    local = locals()
    ignored = [k for k, d in _IGNORED_ARGS.items() if local[k] != d and not (isinstance(d, tuple) and tuple(local[k] or ()) == d)]
    ignored += [k for k in settings if k in _IGNORED_SETTINGS]
    if not is_3d and dispersive:
        ignored.append("dispersive (2-D runs use the indices at the centre wavelength)")
    _warn_ignored(ignored, "write_sparameters_lightsim")

    port_names = [p["name"] for p in _gds.port_info(component)]
    port_source_names = list(port_source_names or port_names)
    for p in port_source_names:
        if p not in port_names:
            raise ValueError(f"port = {p!r} not in {port_names}.")
    port_source_modes = port_source_modes or {k: [0] for k in port_source_names}
    port_modes = list(port_modes or [0])
    port_symmetries = port_symmetries or {}

    xmargin_left = xmargin_left or xmargin
    xmargin_right = xmargin_right or xmargin
    ymargin_top = ymargin_top or ymargin
    ymargin_bot = ymargin_bot or ymargin
    zmargin_top = settings.get("zmargin_top", zmargin_top) or zmargin
    zmargin_bot = settings.get("zmargin_bot", zmargin_bot) or zmargin

    sim_settings = dict(
        port_source_names=port_source_names,
        port_source_modes=port_source_modes,
        port_modes=port_modes,
        resolution=resolution,
        port_symmetries=port_symmetries,
        wavelength_start=wavelength_start,
        wavelength_stop=wavelength_stop,
        wavelength_points=wavelength_points,
        port_margin=port_margin,
        port_monitor_offset=port_monitor_offset,
        port_source_offset=port_source_offset,
        dispersive=dispersive,
        zmargin_top=zmargin_top,
        zmargin_bot=zmargin_bot,
        ymargin_top=ymargin_top,
        ymargin_bot=ymargin_bot,
        xmargin_left=xmargin_left,
        xmargin_right=xmargin_right,
        is_3d=is_3d,
        decay_by=decay_by,
        effective_index=effective_index,
        polarization=polarization,
        steps_per_wavelength=steps_per_wavelength,
        **{k: v for k, v in settings.items() if k in _MAPPED_SETTINGS - {"zmargin_top", "zmargin_bot"}},
    )
    filepath = Path(filepath or get_sparameters_path(component=component, dirpath=dirpath, layer_stack=layer_stack.to_dict(), **sim_settings))
    filepath_sim_settings = filepath.with_suffix(".yml")
    if only_return_filepath_sim_settings:
        return filepath_sim_settings
    if run and filepath.exists():
        if not overwrite:
            return dict(np.load(filepath))
        filepath.unlink()

    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    wl_m = wavelengths * um
    freqs = C0 / wl_m
    f0 = float(np.mean(freqs))
    fw = max(settings.get("dfcen", 0.2) * f0, float(np.ptp(freqs)))
    material_map = settings.get("material_name_to_lightsim", settings.get("material_name_to_meep"))
    material_index, library = _material_index(material_map, float(np.mean(wl_m)))
    exclude = _exclude_levels(layer_stack, settings.get("exclude_layers"))

    start = time.time()
    sim, (z0, z1) = _build_simulation(
        component,
        wl_m,
        layer_stack,
        material_index,
        is_3d=is_3d,
        effective_index=effective_index,
        polarization=polarization,
        resolution=resolution,
        steps_per_wavelength=steps_per_wavelength,
        margins=(xmargin_left * um, xmargin_right * um, ymargin_top * um, ymargin_bot * um),
        zmargin_top=zmargin_top,
        zmargin_bot=zmargin_bot,
        extend_ports_length=settings.get("extend_ports_length"),
        tpml=settings.get("tpml"),
        clad_material=settings.get("clad_material", "sio2"),
        exclude=exclude,
        decay_by=decay_by,
    )
    if dispersive and is_3d:
        _make_dispersive(sim, library, wl_m)

    from .layout._build import PortRuns

    ports = _gds.port_info(component)
    pr = PortRuns(
        sim,
        (z0, z1),
        ports,
        port_margin=port_margin * um,
        port_monitor_offset=port_monitor_offset * um,
        port_source_offset=port_source_offset * um,
        gap=settings.get("distance_source_to_monitors", 0.2) * um,
        port_modes=port_modes,
        port_source_modes=port_source_modes,
        freqs=freqs,
        signal=GaussianPulse(f0, fw),
    )
    sources = [(s, m) for s in port_source_names for m in port_source_modes.get(s, [0])]
    runs = _source_plan(port_names, sources, port_symmetries, port_modes) if run else sources[:1]
    if not run:
        return pr.build(*runs[0])[0]
    parts = pr.run(runs, device=device, progress=1000 if verbosity else 0)
    sp = {}
    for part in parts:
        sp.update(part)
        for key, copies in port_symmetries.items():
            for sym in copies:
                if key in sp and sym not in sp:
                    sp[sym] = sp[key]
    for part in parts:  # computed entries take precedence over copies
        sp.update(part)
    sp["wavelengths"] = wavelengths
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **sp)
    end = time.time()
    _write_settings(filepath_sim_settings, sim_settings, layer_stack, component, end - start, runs, device)
    return sp


def _write_settings(path: Path, sim_settings: dict, layer_stack, component, seconds: float, runs, device) -> None:
    import yaml
    from gdsfactory.serialization import clean_value_json

    s = dict(sim_settings)
    s["layer_stack"] = layer_stack.to_dict()
    s["component"] = component.to_dict()
    s["compute_time_seconds"] = seconds
    s["compute_time_minutes"] = seconds / 60
    s["simulations"] = [f"{a}@{b}" for a, b in runs]
    s["device"] = device if isinstance(device, str) else list(device)
    s["tool"] = TOOL
    try:
        from . import __version__

        s["lightsim_version"] = __version__
    except ImportError:
        pass
    path.write_text(yaml.dump(clean_value_json(s)))


write_sparameters_lightsim_1x1 = partial(write_sparameters_lightsim, port_symmetries=port_symmetries_1x1)
write_sparameters_lightsim_1x1_bend90 = partial(
    write_sparameters_lightsim, ymargin=0, ymargin_bot=3, xmargin_right=3, port_symmetries=port_symmetries_1x1
)


# ----------------------------------------------------------------------------
# batches (write_sparameters_meep_batch / _mpi)
# ----------------------------------------------------------------------------


def _default_devices() -> list[str]:
    from . import devices

    gpus = [str(d).split()[0] for d in devices() if str(d).startswith("cuda")]  # "cuda:0 <name>" -> "cuda:0"
    return gpus or ["cpu"]


def write_sparameters_lightsim_batch(
    jobs: list[dict],
    cores_per_run: int | None = None,
    total_cores: int | None = None,
    temp_dir=None,
    delete_temp_files: bool | None = None,
    dirpath=None,
    layer_stack=None,
    *,
    devices=None,
    **kwargs,
) -> list[Path]:
    """S-parameters of several components or settings, spread over the GPUs.

    The counterpart of ``gplugins.gmeep.write_sparameters_meep_batch``:
    ``jobs`` is a list of :func:`write_sparameters_lightsim` keyword
    dictionaries (each with a ``component``), ``kwargs`` are common to all
    of them, and the ``.npz`` paths are returned in the order of ``jobs``.
    Jobs whose file exists are not run again unless ``overwrite=True``.

    ``devices`` (lightsim only) lists the devices to use, one job at a time
    on each: by default every GPU (``["cuda:0", "cuda:1"]``), else the CPU.
    ``cores_per_run``, ``total_cores``, ``temp_dir`` and
    ``delete_temp_files`` (Meep's MPI pool) are ignored.
    """
    local = locals()
    _warn_ignored([k for k in ("cores_per_run", "total_cores", "temp_dir", "delete_temp_files") if local[k] is not None], "write_sparameters_lightsim_batch")
    devs = list(devices) if devices is not None else _default_devices()
    common = dict(kwargs)
    if dirpath is not None:
        common["dirpath"] = dirpath
    if layer_stack is not None:
        common["layer_stack"] = layer_stack
    merged = [{**common, **job} for job in jobs]
    paths = [Path(write_sparameters_lightsim(**{**job, "only_return_filepath_sim_settings": True})).with_suffix(".npz") for job in merged]
    pool: queue.Queue = queue.Queue()
    for d in devs:
        pool.put(d)

    def work(i):
        dev = pool.get()
        try:
            job = dict(merged[i])
            job.setdefault("filepath", paths[i])
            job["device"] = dev
            write_sparameters_lightsim(**job)
        finally:
            pool.put(dev)

    with ThreadPoolExecutor(max_workers=len(devs)) as ex:
        list(ex.map(work, range(len(merged))))
    return paths


def write_sparameters_lightsim_mpi(
    component,
    layer_stack=None,
    cores: int | None = None,
    filepath=None,
    dirpath=None,
    temp_dir=None,
    temp_file_str: str | None = None,
    live_output: bool = False,
    overwrite: bool = False,
    wait_to_finish: bool = True,
    **kwargs,
) -> Path:
    """The counterpart of ``gplugins.gmeep.write_sparameters_meep_mpi``:
    simulates and returns the ``.npz`` path.

    lightsim needs no MPI: the run happens in this process, on the device
    given by ``device``. ``cores``, ``temp_dir``, ``temp_file_str``,
    ``live_output`` and ``wait_to_finish=False`` are ignored.
    """
    _warn_ignored(
        [k for k, v in (("cores", cores), ("temp_dir", temp_dir), ("temp_file_str", temp_file_str)) if v is not None]
        + (["live_output"] if live_output else [])
        + (["wait_to_finish"] if not wait_to_finish else []),
        "write_sparameters_lightsim_mpi",
    )
    kw = dict(kwargs, layer_stack=layer_stack, dirpath=dirpath, overwrite=overwrite)
    path = Path(filepath) if filepath else Path(write_sparameters_lightsim(component, **kw, only_return_filepath_sim_settings=True)).with_suffix(".npz")
    write_sparameters_lightsim(component, filepath=path, **kw)
    return path


write_sparameters_lightsim_batch_1x1 = partial(write_sparameters_lightsim_batch, port_symmetries=port_symmetries_1x1)
write_sparameters_lightsim_batch_1x1_bend90 = partial(
    write_sparameters_lightsim_batch, ymargin=0, ymargin_bot=3, xmargin_right=3, port_symmetries=port_symmetries_1x1
)
write_sparameters_lightsim_mpi_1x1 = partial(write_sparameters_lightsim_mpi, port_symmetries=port_symmetries_1x1)
write_sparameters_lightsim_mpi_1x1_bend90 = partial(
    write_sparameters_lightsim_mpi, ymargin=0, ymargin_bot=3, xmargin_right=3, port_symmetries=port_symmetries_1x1
)


# ----------------------------------------------------------------------------
# grating coupler with fibre (write_sparameters_grating)
# ----------------------------------------------------------------------------


def write_sparameters_grating_lightsim(
    plot: bool = False,
    plot_contour: bool = False,
    animate: bool = False,
    overwrite: bool = False,
    dirpath=None,
    decay_by: float = 1e-3,
    verbosity: int = 0,
    *,
    device: str = "auto",
    run: bool = True,
    period: float = 0.66,
    fill_factor: float = 0.5,
    n_periods: int = 30,
    widths=None,
    gaps=None,
    fiber_angle_deg: float = 20.0,
    fiber_xposition: float = 1.0,
    fiber_core_diameter: float = 10.4,
    fiber_numerical_aperture: float = 0.14,
    fiber_clad_material: float = 1.44,
    nwg: float = 3.48,
    nslab: float | None = None,
    clad_material: float = 1.44,
    nbox: float = 1.44,
    nsubstrate: float = 3.48,
    pml_thickness: float = 1.0,
    substrate_thickness: float = 1.0,
    box_thickness: float = 2.0,
    core_thickness: float = 0.22,
    slab_thickness: float = 0.15,
    top_clad_thickness: float = 2.0,
    air_gap_thickness: float = 1.0,
    fiber_thickness: float = 2.0,
    resolution: int = 64,
    wavelength_start: float = 1.4,
    wavelength_stop: float = 1.7,
    wavelength_points: int = 150,
    eps_averaging: bool = False,
    fiber_port_y_offset_from_air: float = 1,
    waveguide_port_x_offset_from_grating_start: float = 10,
    fiber_port_x_size: float | None = None,
    xmargin: float = 10.0,
):
    """S-parameters of a 1-D grating coupler and a fibre, in a 2-D side view.

    The counterpart of ``gplugins.gmeep.write_sparameters_grating``, with the
    same geometry arguments (lengths in um, materials as refractive indices)
    and result: ``o1`` is the waveguide, ``o2`` the fibre; ``"o2@0,o1@0"``
    is the coupling from the waveguide mode into the fibre mode. It is cached
    as ``fiber_<hash>.npz`` in ``dirpath``.

    The structure is the one gplugins builds (substrate, buried oxide, etched
    grating, cladding, air gap, a tilted fibre core and cladding), run as a
    lightsim 2-D simulation invariant along y with E along y (the slab TE
    mode). The waveguide mode is solved on the grid; the fibre mode is the
    Gaussian beam of waist ``fiber_core_diameter / 2`` tilted by
    ``fiber_angle_deg`` (a fit of the fibre's fundamental mode). Meep's
    ``eps_averaging`` and ``animate`` are ignored; ``plot`` returns the
    simulation without running it. With ``run=False`` likewise.
    """
    import gdsfactory as gf  # noqa: F401  (gplugins' environment)
    from .modes import gaussian_mode
    from .spec import Box, Medium, PolySlab, Structure

    _warn_ignored([k for k, v in (("animate", animate), ("eps_averaging", eps_averaging), ("plot_contour", plot_contour)) if v], "write_sparameters_grating_lightsim")
    widths = list(widths or n_periods * [period * fill_factor])
    gaps = list(gaps or n_periods * [period * (1 - fill_factor)])
    nslab = nslab or nwg
    settings = dict(
        widths=widths, gaps=gaps, n_periods=n_periods, nslab=nslab, fiber_angle_deg=fiber_angle_deg, fiber_xposition=fiber_xposition,
        fiber_core_diameter=fiber_core_diameter, fiber_numerical_aperture=fiber_numerical_aperture, fiber_clad_material=fiber_clad_material,
        nwg=nwg, clad_material=clad_material, nbox=nbox, nsubstrate=nsubstrate, pml_thickness=pml_thickness,
        substrate_thickness=substrate_thickness, box_thickness=box_thickness, core_thickness=core_thickness,
        slab_thickness=slab_thickness, top_clad_thickness=top_clad_thickness, air_gap_thickness=air_gap_thickness,
        fiber_thickness=fiber_thickness, resolution=resolution, wavelength_start=wavelength_start, wavelength_stop=wavelength_stop,
        wavelength_points=wavelength_points, decay_by=decay_by, fiber_port_y_offset_from_air=fiber_port_y_offset_from_air,
        waveguide_port_x_offset_from_grating_start=waveguide_port_x_offset_from_grating_start, fiber_port_x_size=fiber_port_x_size,
        xmargin=xmargin, tool=TOOL,
    )  # fmt: skip
    from gdsfactory.config import PATH

    dirpath = Path(dirpath or PATH.sparameters)
    filepath = dirpath / f"fiber_{_kwargs_hash(**settings)[:8]}.npz"
    if run and not plot and filepath.exists() and not overwrite:
        return dict(np.load(filepath))

    # vertical coordinate: z, with the same layer order and origin as gplugins (cell centred on 0)
    sz = pml_thickness + substrate_thickness + box_thickness + core_thickness + top_clad_thickness + air_gap_thickness + fiber_thickness + pml_thickness
    zb = -sz / 2 + pml_thickness + substrate_thickness  # bottom of the buried oxide
    z_core0 = zb + box_thickness
    z_core1 = z_core0 + core_thickness
    z_clad1 = z_core1 + top_clad_thickness
    z_air1 = z_clad1 + air_gap_thickness
    angle = np.radians(fiber_angle_deg)
    fiber_port_z = z_air1 + fiber_port_y_offset_from_air
    fiber_port_x = abs(fiber_port_z * np.tan(angle))
    length_grating = float(np.sum(widths) + np.sum(gaps))
    sxy = 2 * xmargin + 2 * pml_thickness + 2 * fiber_port_x + length_grating
    grating_start = -fiber_xposition
    x0, x1 = -sxy / 2 + pml_thickness, sxy / 2 - pml_thickness
    zlo, zhi = -sz / 2 + pml_thickness, sz / 2 - pml_thickness
    n_core_fiber = (fiber_numerical_aperture**2 + fiber_clad_material**2) ** 0.5
    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    wl_m = wavelengths * um
    freqs = C0 / wl_m
    f0 = float(np.mean(freqs))
    dx = 1e-6 / resolution
    pml_layers = max(8, int(round(pml_thickness * resolution)))
    U = um
    sim = Simulation(
        min=[x0 * U, -0.5 * dx, zlo * U],
        max=[x1 * U, 0.5 * dx, zhi * U],
        mesh=UniformMesh(dx),
        run_time=1.0,
        boundaries={"x": [PML(layers=pml_layers)] * 2, "y": [{"type": "periodic"}] * 2, "z": [PML(layers=pml_layers)] * 2},
        background="gc_air",
        shutoff=decay_by,
        dimensions=2,
        invariant_axis="y",
        polarization="tm",  # y-invariant: E along y is the "tm" set (Ey, Hx, Hz)
    )
    media = {"gc_air": 1.0, "gc_fclad": fiber_clad_material, "gc_fcore": n_core_fiber, "gc_clad": clad_material, "gc_box": nbox, "gc_slab": nslab, "gc_wg": nwg, "gc_sub": nsubstrate}
    for name, n in media.items():
        sim.add_medium(Medium(name, n=n))
    big = 1e3

    def box(name, xa, xb, za, zb_, medium):
        sim.structures.append(Structure(Box([xa * U, -1.0, za * U], [xb * U, 1.0, zb_ * U]), medium, name=name))

    def tilted(name, width, medium):
        # a strip of width `width` centred on the fibre axis, through the origin, tilted by the fibre angle
        d = np.array([np.sin(angle), np.cos(angle)])  # axis direction (x, z): towards +z, leaning to +x
        nrm = np.array([d[1], -d[0]])
        L = 200.0
        pts = [(s * L * d + t * width / 2 * nrm) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        verts = [(float(p[0]) * U, float(p[1]) * U) for p in pts]  # (x, z) for axis 1
        sim.structures.append(Structure(PolySlab(verts, -1.0, 1.0, axis=1), medium, name=name))

    # painting order: later structures win (gplugins: Meep's later blocks win as well)
    zc_wg = zb + box_thickness / 2 + core_thickness / 2 + top_clad_thickness / 2
    tilted("fiber_clad", 120.0, "gc_fclad")
    tilted("fiber_core", fiber_core_diameter, "gc_fcore")
    box("air_gap", -big, big, z_clad1, z_air1, "gc_air")
    box("top_clad", -big, big, z_core0, z_clad1, "gc_clad")
    box("box", -big, big, zb, z_core0, "gc_box")
    box("slab", -big, big, z_core0, z_core0 + slab_thickness, "gc_slab")
    etch = core_thickness - slab_thickness
    x = grating_start
    for i, (w, g) in enumerate(zip(widths, gaps)):
        box(f"tooth_{i}", x + g / 2 - w / 2, x + g / 2 + w / 2, z_core1 - etch, z_core1, "gc_wg")  # as gplugins places them
        x += w + g
    box("wg", -big, grating_start, z_core1 - etch, z_core1, "gc_wg")
    box("substrate", -big, big, -big, zb, "gc_sub")
    dz_wg = box_thickness + core_thickness / 2 + top_clad_thickness
    wg_x = grating_start - waveguide_port_x_offset_from_grating_start
    wmin, wmax = [wg_x * U, 0.0, (zc_wg - dz_wg / 2) * U], [wg_x * U, 0.0, (zc_wg + dz_wg / 2) * U]
    fx = (fiber_port_x_size or 3.5 * fiber_core_diameter) / 2
    fmin, fmax = [(fiber_port_x - fx) * U, 0.0, fiber_port_z * U], [(fiber_port_x + fx) * U, 0.0, fiber_port_z * U]
    wg_mode = solve_modes(sim, 0, wg_x * U, wmin, wmax, f0, num_modes=1)[0]
    fiber_center = [0.0, 0.0, fiber_port_z * U]
    fiber_center[0] = fiber_port_z * np.tan(angle) * U  # where the fibre axis crosses the port plane
    # the beam travels down (-z) and towards -x: from +z by pi - angle, tilted to -x
    prof, power = gaussian_mode(sim, 2, fiber_port_z * U, fmin, fmax, f0, fiber_core_diameter / 2 * U, center=fiber_center, polarization="v", n=fiber_clad_material, theta=angle, phi=0.0)
    signal = GaussianPulse(f0, max(0.2 * f0, float(np.ptp(freqs))))
    sim.run_time = 30.0 * nwg * sxy * U / C0 + 2e-12
    gap = 0.2
    sim.add_source(ModeSource(0, wg_x * U, "+", wmin, wmax, wg_mode.profile, signal, name="src_wg"))
    wmin2, wmax2 = list(wmin), list(wmax)
    wmin2[0] = wmax2[0] = (wg_x + gap) * U
    sim.add_monitor(ModeMonitor("m_wg", 0, (wg_x + gap) * U, wmin2, wmax2, freqs, [wg_mode.profile]))
    fmin2, fmax2 = list(fmin), list(fmax)
    fmin2[2] = fmax2[2] = (fiber_port_z - gap) * U
    sim.add_monitor(ModeMonitor("m_fiber", 2, (fiber_port_z - gap) * U, fmin2, fmax2, freqs, [prof]))
    if plot or not run:
        return sim
    r = sim.run(device=device, progress=1000 if verbosity else 0)
    a1, b1 = r.mode_amplitudes("m_wg")
    # the Gaussian reference travels towards +z with the tilt; light going up into the fibre is its "+" amplitude
    a2, b2 = r.mode_amplitudes("m_fiber")
    s11 = np.asarray(b1 / a1)
    s21 = np.asarray(a2 / a1) * np.sqrt(power)
    s22 = np.asarray(b2 / a2)  # as gplugins.gmeep computes it (no source in the fibre)
    sp = {"o1@0,o1@0": s11, "o1@0,o2@0": s21, "o2@0,o1@0": s21.copy(), "o2@0,o2@0": s22, "wavelengths": wavelengths}
    dirpath.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **sp)
    return sp


def write_sparameters_grating_lightsim_batch(instances: list[dict], cores_per_instance: int | None = None, total_cores: int | None = None, temp_dir=None, delete_temp_files: bool | None = None, verbosity: bool = False, *, devices=None, **kwargs) -> list[dict]:
    """Several :func:`write_sparameters_grating_lightsim` runs spread over the
    GPUs (``gplugins.gmeep.write_sparameters_grating_batch``); returns the
    S-parameter dicts in the order of ``instances``. Meep's process options
    are ignored."""
    local = locals()
    _warn_ignored([k for k in ("cores_per_instance", "total_cores", "temp_dir", "delete_temp_files") if local[k] is not None], "write_sparameters_grating_lightsim_batch")
    devs = list(devices) if devices is not None else _default_devices()
    pool: queue.Queue = queue.Queue()
    for d in devs:
        pool.put(d)

    def work(inst):
        dev = pool.get()
        try:
            return write_sparameters_grating_lightsim(**{**kwargs, **inst, "device": dev})
        finally:
            pool.put(dev)

    with ThreadPoolExecutor(max_workers=len(devs)) as ex:
        return list(ex.map(work, instances))


# ----------------------------------------------------------------------------
# mode solvers: one implementation, in lightsim.gplugins_modes
# ----------------------------------------------------------------------------

# gplugins.tidy3d.modes (Waveguide, WaveguideCoupler, sweeps) and gplugins.modes (MPB-style functions)
from .gplugins_modes import (  # noqa: E402
    GMode,
    Waveguide,
    WaveguideCoupler,
    coupling_length,
    find_coupling_vs_gap,
    find_mode_dispersion,
    find_modes_coupler,
    find_modes_waveguide,
    find_neff_ng_dw_dh,
    find_neff_vs_width,
    sweep_bend_mismatch,
    sweep_coupling_length,
    sweep_fraction_te,
    sweep_mode_area,
    sweep_n_eff,
    sweep_n_group,
)


# ----------------------------------------------------------------------------
# SAX
# ----------------------------------------------------------------------------


def model_from_npz(filepath, xkey: str = "wavelengths", xunits: float = 1):
    """SAX model of a component from its ``.npz`` file.

    ``gplugins.sax.read.model_from_npz`` reads lightsim's files as they are;
    this calls it when gplugins is installed. Without gplugins the same
    model is built here (it needs ``sax`` and ``jax``): a function of the
    wavelength (um) that returns the S-parameters, interpolated, as a SAX
    dict ``{("o1", "o2"): ...}``.
    """
    try:
        from gplugins.sax.read import model_from_npz as gp_model
    except ImportError:
        gp_model = None
    if gp_model is not None:
        return gp_model(filepath, xkey=xkey, xunits=xunits)
    import jax.numpy as jnp

    sp = np.load(filepath) if isinstance(filepath, str | Path) else filepath
    x = np.asarray(sp[xkey]) * xunits
    order = np.argsort(x)
    x = x[order]
    data = {k: np.asarray(sp[k])[order] for k in sp.keys() if not k.startswith("wav")}

    def model(wl=1.55):
        out = {}
        for key, v in data.items():
            a, b = (s.split("@")[0] for s in key.split(","))
            out[(a, b)] = jnp.interp(wl, x, v)
        return out

    return model


# Meep's names, so that ``import lightsim.gplugins as gm`` runs gplugins.gmeep scripts unchanged
write_sparameters_meep = write_sparameters_lightsim
write_sparameters_meep_1x1 = write_sparameters_lightsim_1x1
write_sparameters_meep_1x1_bend90 = write_sparameters_lightsim_1x1_bend90
write_sparameters_meep_batch = write_sparameters_lightsim_batch
write_sparameters_meep_batch_1x1 = write_sparameters_lightsim_batch_1x1
write_sparameters_meep_batch_1x1_bend90 = write_sparameters_lightsim_batch_1x1_bend90
write_sparameters_meep_mpi = write_sparameters_lightsim_mpi
write_sparameters_meep_mpi_1x1 = write_sparameters_lightsim_mpi_1x1
write_sparameters_meep_mpi_1x1_bend90 = write_sparameters_lightsim_mpi_1x1_bend90
write_sparameters_grating = write_sparameters_grating_lightsim
write_sparameters_grating_batch = write_sparameters_grating_lightsim_batch


class _PortSymmetries:
    """``gm.port_symmetries.port_symmetries_1x1`` as in gplugins."""

    port_symmetries_1x1 = port_symmetries_1x1
    port_symmetries_crossing = port_symmetries_crossing


port_symmetries = _PortSymmetries()

_SIGNATURE = inspect.signature(write_sparameters_lightsim)
