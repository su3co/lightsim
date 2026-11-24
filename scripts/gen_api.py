#!/usr/bin/env python3
"""Generate docs/reference/python-api.md from the lightsim package.

    python scripts/gen_api.py              # rewrite the reference
    python scripts/gen_api.py --check      # exit 1 if the committed file is stale
    python scripts/gen_api.py --missing    # list public objects without a docstring

Every public function, class, method and constant of the documented modules
gets an entry: its signature (from ``inspect``) and its docstring, converted
from the reST-flavoured text used in the sources to markdown. A public object
is one defined in the module itself whose name does not start with an
underscore.

Objects that have no docstring take their text from ``HANDWRITTEN`` below.
When a docstring is added to the source, it replaces the hand-written entry
automatically; delete the entry here then. ``--missing`` lists what is still
hand-written and what has neither.

The package must be importable (``maturin develop --release``). A module
whose import needs an optional package that is not installed
(``lightsim.invdes.torch`` without PyTorch, ``lightsim.invdes.jax`` without
JAX) keeps the section the page already has; the script says so. Regenerate
with the package installed after changing such a module.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "reference" / "python-api.md"

# (module, one-line description). Order of the page.
MODULES = [
    ("lightsim.spec", "Builders for media, shapes, sources, monitors, boundaries and meshes, the `Simulation` class and the `Results` class. Everything here is also available as `lightsim.<name>`."),
    ("lightsim.modes", "Waveguide mode solver and Gaussian-beam profiles."),
    ("lightsim.xsection", "Waveguide cross-sections for the mode solver, from shapes or gdsfactory, without a 3-D simulation."),
    ("lightsim.mode_sweeps", "Mode-solver sweeps (width, thickness, wavelength, temperature, gap, bend radius) as tidy tables."),
    ("lightsim.gplugins_modes", "Mode-solver functions with the names, micrometre arguments and return types of gplugins."),
    ("lightsim.eme", "Eigenmode expansion of devices cut into sections."),
    ("lightsim.circuit", "Frequency-domain circuit simulation from netlists, with SAX-compatible models."),
    ("lightsim.sparams", "S-parameters of gdsfactory components."),
    ("lightsim.gds", "gdsfactory components and layer stacks to structures."),
    ("lightsim.sparameters", "S-parameters as one object: Touchstone, INTERCONNECT text, gplugins npz and SAX readers and writers, and the phase convention of each."),
    ("lightsim.layout", "Any layout tool: GDSII + technology + ports in, S-parameters or simulation specs out."),
    ("lightsim.gplugins", "The gplugins interface: `write_sparameters_meep`-compatible S-parameters, batches, grating couplers and SAX models; it also re-exports the mode-solver front of `lightsim.gplugins_modes`."),
    ("lightsim.eim", "2.5-D effective-index simulations."),
    ("lightsim.adjoint", "Adjoint gradient of a mode power."),
    ("lightsim.invdes", "Inverse design: S-matrix gradients, parametrisation, optimiser."),
    ("lightsim.invdes.shapes", "Shape parametrisations for inverse design: polygons with movable vertices and level sets."),
    ("lightsim.invdes.torch", "The S-matrix as a PyTorch function."),
    ("lightsim.invdes.jax", "The S-matrix as a JAX function."),
    ("lightsim.autodiff", "Gradients computed inside the engine: persistent S-matrix session, reverse mode through the time loop."),
    ("lightsim.symmetry", "Mirror-symmetry reduction."),
    ("lightsim.fem.geometry", "Finite-element solvers (`lightsim.fem`, every public name of the submodules below is also `lightsim.fem.<name>`): cross-section geometries and the gdsfactory cut."),
    ("lightsim.fem.mesh", "Triangle meshes and finite-element fields."),
    ("lightsim.fem.heat", "Heat conduction and thermo-optic phase shifters."),
    ("lightsim.fem.electrostatics", "Electrostatics, capacitance matrices and Pockels modulators."),
    ("lightsim.fem.modes", "Full-vectorial finite-element waveguide modes."),
    ("lightsim.fem.gplugins", "gplugins-compatible entry points of the finite-element solvers."),
    ("lightsim.fem.validation", "Validation suite of the finite-element solvers."),
    ("lightsim.sweep", "Spectra at a fixed angle of incidence from a sweep of oblique runs."),
    ("lightsim.materials", "Dispersion library and pole fitting."),
    ("lightsim.sheets", "Sheet materials: surface conductivity and graphene as zero-thickness sheets or thin layers."),
    ("lightsim.plugins", "User-defined material models (plug-ins) written in Python."),
    ("lightsim.eo", "Free-carrier plasma dispersion in silicon."),
    ("lightsim.gratings", "Diffraction-order analysis."),
    ("lightsim.resonance", "Harmonic inversion and Lorentzian fits."),
    ("lightsim.gui", "Graphical interface: a local web server to inspect a simulation, run it and read its results."),
]

# Modules whose import needs an optional package: documented only if it is there.
OPTIONAL = {"lightsim.invdes", "lightsim.invdes.shapes", "lightsim.invdes.torch", "lightsim.invdes.jax", "lightsim.autodiff"}

# Names that are imported into a module but belong to the typing machinery.
SKIP_NAMES = {"Any", "Iterable", "Sequence", "Callable", "Vec3", "annotations", "dataclass", "field"}

CONSTANTS = {
    "lightsim.spec.C0": "Speed of light in vacuum, 299 792 458 m/s.",
    "lightsim.spec.um": "One micrometre in metres, `1e-6`.",
    "lightsim.spec.nm": "One nanometre in metres, `1e-9`.",
    "lightsim.spec.PEC": "A `Medium` named `\"PEC\"` that is a perfect electric conductor.",
    "lightsim.spec.PERIODIC": "Periodic boundary condition, for `Simulation.set_boundaries`.",
    "lightsim.spec.PEC_WALL": "Perfect-electric-conductor boundary condition: tangential E is zero on the face.",
    "lightsim.spec.PMC_WALL": "Perfect-magnetic-conductor boundary condition: tangential H is zero on the face.",
    "lightsim.modes.C0": None,
    "lightsim.modes.ETA0": "Impedance of free space, 376.730313412 ohm.",
    "lightsim.gds.MATERIAL_INDEX": "Refractive index at 1550 nm by lower-case material name, used for the materials of a layer stack. Override entries with the `material_index` argument.",
    "lightsim.adjoint.EPS0": "Vacuum permittivity in F/m.",
    "lightsim.eo.EPS0": None,
    "lightsim.eo.COEFFICIENTS": "Plasma-dispersion coefficients `(A_e, p_e, A_h, p_h, B_e, q_e, B_h, q_h)` by model name and wavelength in nm; densities in cm^-3, absorption in cm^-1.",
    "lightsim.eo.N_SILICON": "Refractive index of undoped silicon by wavelength in nm: `{1550: 3.476, 1310: 3.503}`.",
}

# ----------------------------------------------------------------------------
# Entries for objects that have no docstring in the source. Markdown.
# ----------------------------------------------------------------------------

HANDWRITTEN = {
}

# Text appended to an entry after its docstring.
EXTRA = {
    "lightsim.spec.Simulation": "Fields:\n\n- `min`, `max`: corners of the simulation region (metres).\n- `mesh`: a mesh from `UniformMesh` or `AutoMesh`, or an explicit-mesh dictionary `{\"type\": \"explicit\", \"x\": [...], \"y\": [...], \"z\": [...]}`.\n- `run_time`: simulated time in seconds.\n- `boundaries`: `{\"x\": [low, high], \"y\": [...], \"z\": [...]}`; `None` puts a `PML()` on all six faces.\n- `media`, `structures`, `sources`, `monitors`: lists filled by the `add_*` methods.\n- `background`: name of the background medium.\n- `shutoff`: stop early when the field energy has fallen to this fraction of its peak; `None` disables it.\n- `shutoff_interval`: steps between energy checks.\n- `courant`: time step as a fraction of the stability limit.\n- `subpixel`: sub-pixel smoothing of dielectric interfaces.\n- `name`: a label.\n- `dft_stride`: steps between samples of the frequency-domain monitors; `None` chooses it per monitor from the bandwidth of the sources, `1` samples every step.\n- `dimensions`: 3, or 2 for a simulation whose fields do not depend on the coordinate along `invariant_axis` (set automatically when `min` / `max` have two components).\n- `invariant_axis`: `\"x\"`, `\"y\"`, `\"z\"` or 0, 1, 2; stored as the number.\n- `polarization`: `\"te\"`, `\"tm\"` or `\"both\"` (2-D only).\n- `thickness`: extent of the region along the invariant axis when `min` / `max` have two components, in metres.\n- `force_3d_kernels`: run a 2-D simulation through the general 3-D update (cross-checks).",
    "lightsim.spec.Results": "The attribute `raw` is that dictionary, with the keys `dt`, `stats`, `dft`, `flux`, `modes`, `far_field`, `time`, `energy`, `sources` and `source_spectra`.",
}

TOP_LEVEL = """\
`import lightsim as ls` gives access to everything in `lightsim.spec`, and to
`solve_modes`, `gaussian_mode` and `Mode` from `lightsim.modes`,
`run_symmetric` from `lightsim.symmetry`, and the two items below. The other
modules are imported explicitly, for example `from lightsim import materials`.

### `lightsim.devices`

```text
devices()
```

List of the devices this build can use: `"cpu"`, then one entry per GPU,
`"cuda:0 <model name>"`, `"cuda:1 <model name>"`, ...

### `lightsim.__version__`

The package version, a string.
"""

# ----------------------------------------------------------------------------
# docstring -> markdown
# ----------------------------------------------------------------------------

ROLE = re.compile(r":(?:func|class|meth|mod|attr|data|obj):`(~?)([^`]+)`")
SECTIONS = {"Parameters", "Attributes", "Returns", "Raises", "Notes", "Examples", "Yields"}
ITEM_SECTIONS = {"Parameters", "Attributes"}


def inline(text: str) -> str:
    """Inline reST to markdown: roles and double backticks become code."""
    text = ROLE.sub(lambda m: "`" + (m.group(2).rsplit(".", 1)[-1] if m.group(1) else m.group(2)) + "`", text)
    out = []
    for i, part in enumerate(re.split(r"(``.+?``|`[^`\n]+`)", text)):
        if i % 2:
            out.append("`" + part.strip("`") + "`")
        else:
            # keep **bold** and *emphasis*, escape other asterisks (they would start emphasis in markdown)
            part = re.sub(r"(?<![\w*\\])\*\*(?=\S)([^*\n]*?\S)\*\*(?![\w*])", "\x00\x00\\1\x00\x00", part)
            part = re.sub(r"(?<![\w*\\\x00])\*(?=\S)([^*\n]*?\S)\*(?![\w*])", "\x00\\1\x00", part)
            part = part.replace("*", "\\*").replace("\x00", "*")
            # "<word>" would be read as an HTML tag
            part = re.sub(r"<(?=[A-Za-z/])", "&lt;", part)
            out.append(part)
    return "".join(out)


def names(text: str) -> str:
    return ", ".join("`" + n.strip() + "`" for n in text.split(",") if n.strip())


TYPE_WORDS = {"int", "float", "str", "bool", "dict", "list", "tuple", "complex", "callable", "array", "ndarray", "sequence", "sequences", "of", "or", "a", "vector", "none", "optional"}


def is_type(kind: str, described: bool) -> bool:
    """Whether the text after ``name :`` is a type (``float``, ``sequence of
    str, optional``) and not the start of the description itself, the other
    style in use (``name : what it is, continued on indented lines``). A type
    is short and is not a sentence; without a description below it, it must
    also consist of type words, literals and tuples only."""
    if len(kind) > 45 or kind.endswith((".", ",", ";", ":")):
        return False
    if described:
        return True
    words = re.findall(r"[A-Za-z_]+", re.sub(r"\(.*?\)|\".*?\"|`.*?`", "", kind))
    return all(w.lower() in TYPE_WORDS for w in words)


def convert(doc: str) -> str:
    """A docstring as markdown."""
    lines = inspect.cleandoc(doc).split("\n")
    blocks, cur = [], []
    for line in lines + [""]:
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    out, mode = [], None
    for block in blocks:
        # numpy-style section header
        if len(block) >= 2 and block[0].strip() in SECTIONS and set(block[1].strip()) == {"-"}:
            mode = "items" if block[0].strip() in ITEM_SECTIONS else None
            out.append(f"**{block[0].strip()}**")
            block = block[2:]
            if not block:
                continue
        if mode == "items" and not block[0].startswith(" "):
            # a block of "name", "name : type" or "name : description" lines,
            # each possibly followed by indented lines; anything else ends
            # the item list
            heads = [line for line in block if not line.startswith(" ")]
            if not (any(line.startswith(" ") for line in block) or all(" : " in h for h in heads)):
                mode = None
        if mode == "items" and not block[0].startswith(" "):
            items = []
            for line in block:
                if line.startswith(" ") and items:
                    items[-1][1].append(line.strip())
                elif " : " in line:
                    head, kind = line.split(" : ", 1)
                    items.append([head, [], kind.strip()])
                else:
                    items.append([line.strip().rstrip(":").strip(), [], ""])
            for item in items:
                if item[2] and not is_type(item[2], bool(item[1])):
                    # "name : description": the text after the colon starts the description
                    item[1].insert(0, item[2])
                    item[2] = ""
            out.append(
                "\n".join(
                    f"- {names(h)}" + (f" ({inline(kind)})" if kind else "") + (f": {inline(' '.join(r))}" if r else "")
                    for h, r, kind in items
                )
            )
            continue
        if all(line.startswith("  ") for line in block):
            indent = min(len(line) - len(line.lstrip()) for line in block)
            out.append("```text\n" + "\n".join(line[indent:] for line in block) + "\n```")
            continue
        if re.match(r"([*-]|\d+\.) ", block[0]):
            items = []
            for line in block:
                m = re.match(r"([*-]|\d+\.) (.*)", line)
                if m:
                    items.append(("-" if m.group(1) in "*-" else m.group(1), [m.group(2)]))
                else:
                    items[-1][1].append(line.strip())
            out.append("\n".join(f"{mark} {inline(' '.join(text))}" for mark, text in items))
            continue
        # a paragraph, possibly followed in the same block by an indented literal
        para, literal = [], []
        for line in block:
            (literal if line.startswith("    ") and para and not literal or literal else para).append(line)
        out.append(inline(" ".join(line.strip() for line in para)))
        if literal:
            indent = min(len(line) - len(line.lstrip()) for line in literal)
            out.append("```text\n" + "\n".join(line[indent:] for line in literal) + "\n```")
    return "\n\n".join(out)


# ----------------------------------------------------------------------------
# signatures
# ----------------------------------------------------------------------------


def signature(name: str, obj, drop_self: bool = False) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{name}(...)"
    parts, star_done = [], False
    params = list(sig.parameters.values())
    if drop_self and params and params[0].name in ("self", "cls"):
        params = params[1:]
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            parts.append("*" + p.name)
            star_done = True
            continue
        if p.kind is p.VAR_KEYWORD:
            parts.append("**" + p.name)
            continue
        if p.kind is p.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        if p.default is p.empty:
            parts.append(p.name)
        elif dataclasses.is_dataclass(obj) and isinstance(p.default, dataclasses._MISSING_TYPE) or "factory" in repr(p.default):
            parts.append(f"{p.name}=...")
        else:
            parts.append(f"{p.name}={p.default!r}")
    one = f"{name}({', '.join(parts)})"
    if len(one) <= 88:
        return one
    return f"{name}(\n" + "".join(f"    {p},\n" for p in parts) + ")"


def field_comments(cls) -> dict:
    """``#:`` comments next to dataclass fields."""
    out = {}
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return out
    for m in re.finditer(r"^\s+(\w+)\s*:[^#\n]*#:?\s*(.+)$", src, flags=re.M):
        out[m.group(1)] = m.group(2).strip()
    return out


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------


class Page:
    def __init__(self):
        self.out = []
        self.handwritten = []
        self.undocumented = []
        self.kept = []  # optional modules that could not be imported: section taken from the page on disk

    def text(self, qual: str, obj) -> str:
        raw = (obj.fget if isinstance(obj, property) else obj).__doc__  # not inherited docstrings
        doc = inspect.cleandoc(raw) if raw else None
        if dataclasses.is_dataclass(obj) and doc and doc.startswith(obj.__name__ + "("):
            doc = None  # the automatic dataclass docstring is only the signature
        if doc:
            return convert(doc) + ("\n\n" + EXTRA[qual] if qual in EXTRA else "")
        if qual in HANDWRITTEN:
            self.handwritten.append(qual)
            return HANDWRITTEN[qual]
        self.undocumented.append(qual)
        return ""

    def entry(self, qual: str, sig: str | None, body: str, level: str = "###"):
        self.out.append(f"{level} `{qual}`")
        if sig:
            self.out.append("```text\n" + sig + "\n```")
        if body:
            self.out.append(body)

    def function(self, mod: str, name: str, obj):
        self.entry(f"{mod}.{name}", signature(name, obj), self.text(f"{mod}.{name}", obj))

    def klass(self, mod: str, name: str, cls):
        qual = f"{mod}.{name}"
        body = self.text(qual, cls)
        if dataclasses.is_dataclass(cls):
            notes = field_comments(cls)
            rows = []
            for f in dataclasses.fields(cls):
                if f.name.startswith("_"):
                    continue
                rows.append(f"- `{f.name}`" + (f": {inline(notes[f.name])}" if f.name in notes else ""))
            if rows and any(":" in r for r in rows):
                body += ("\n\n" if body else "") + "Fields:\n\n" + "\n".join(rows)
        many = dataclasses.is_dataclass(cls) and len(dataclasses.fields(cls)) > 12 and name != "Simulation"
        self.entry(qual, None if many else signature(name, cls), body)  # result records: no constructor
        for mname, member in cls.__dict__.items():
            if mname.startswith("_") and mname != "__call__":
                continue
            if isinstance(member, (staticmethod, classmethod)):
                fn = member.__func__
                self.entry(f"{qual}.{mname}", signature(f"{name}.{mname}", fn, drop_self=isinstance(member, classmethod)), self.text(f"{qual}.{mname}", fn), "###")
            elif inspect.isfunction(member):
                self.entry(f"{qual}.{mname}", signature(f"{name}.{mname}", member, drop_self=True), self.text(f"{qual}.{mname}", member), "###")
            elif isinstance(member, property):
                self.entry(f"{qual}.{mname}", None, "Property. " + self.text(f"{qual}.{mname}", member), "###")

    def module(self, modname: str, summary: str):
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            if modname in OPTIONAL:
                # the optional package is not installed here: keep the section
                # the page already has instead of dropping it
                kept = existing_section(modname)
                if kept:
                    self.out.append(kept)
                    self.kept.append(f"{modname} ({e})")
                return
            raise
        self.out.append(f"## `{modname}`")
        self.out.append(summary)
        public = getattr(mod, "__all__", None)

        # defined in the module itself, or in one of its private submodules
        # (a package such as lightsim.spec that keeps its code in `_name.py`)
        def own(obj) -> bool:
            return obj.__module__ == modname or obj.__module__.startswith(modname + "._")

        members = []
        for index, (name, obj) in enumerate(vars(mod).items()):
            if name.startswith("_") or name in SKIP_NAMES or inspect.ismodule(obj):
                continue
            if public is not None and name not in public and not (inspect.isclass(obj) and own(obj)):
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # a package re-exports the names in its __all__ from its submodules
                reexport = public is not None and name in public and obj.__module__.startswith(modname + ".")
                if not own(obj) and not reexport:
                    continue
                # (in order of definition, which for a package is the order of its imports)
                order = list(public).index(name) if reexport else index
                members.append((order, name, obj))
            elif f"{modname}.{name}" in CONSTANTS:
                if CONSTANTS[f"{modname}.{name}"]:
                    members.append((10**9, name, obj))
        for _, name, obj in sorted(members, key=lambda m: m[0]):
            if inspect.isclass(obj):
                self.klass(modname, name, obj)
            elif inspect.isfunction(obj):
                self.function(modname, name, obj)
            else:
                self.entry(f"{modname}.{name}", None, CONSTANTS[f"{modname}.{name}"])


def existing_section(modname: str) -> str:
    """The section of ``modname`` in the page as it is on disk ("" if none)."""
    if not OUT.is_file():
        return ""
    lines = OUT.read_text().split("\n")
    head = f"## `{modname}`"
    if head not in lines:
        return ""
    start = lines.index(head)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).rstrip("\n")


def build() -> Page:
    import lightsim

    page = Page()
    page.out.append("# Python API")
    page.out.append(
        "Every public function, class and constant of the `lightsim` Python package, with its signature and "
        "description. This page is generated from the package by `scripts/gen_api.py` "
        f"(package version {lightsim.__version__}). For explanations and worked examples see the guide pages."
    )
    page.out.append(
        "Builders such as `Medium`, `Box` or `PlaneWave` return plain dictionaries in the "
        "[JSON format](json-spec.md). Lengths are in metres, times in seconds and frequencies in hertz."
    )
    page.out.append("## `lightsim`")
    page.out.append(TOP_LEVEL.rstrip())
    for modname, summary in MODULES:
        page.module(modname, summary)
    return page


def main() -> int:
    page = build()
    text = "\n\n".join(page.out) + "\n"
    if "--missing" in sys.argv:
        print("hand-written in gen_api.py (no docstring in the source):")
        for q in page.handwritten:
            print("  " + q)
        print("no text at all:")
        for q in page.undocumented:
            print("  " + q)
        unused = sorted(set(HANDWRITTEN) - set(page.handwritten))
        if unused:
            print("hand-written entries no longer needed (the source has a docstring now, or the object is gone):")
            for q in unused:
                print("  " + q)
        return 0
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"{OUT.relative_to(OUT.parents[2])} is out of date: run python scripts/gen_api.py")
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines, {len(page.handwritten)} hand-written entries, {len(page.undocumented)} without text)")
    for q in page.kept:
        print(f"kept the existing section of {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
