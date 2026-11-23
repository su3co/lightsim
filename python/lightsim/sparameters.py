"""S-parameters as one object, with readers and writers for circuit tools.

:class:`SParameters` holds an S-matrix over frequency with named ports and
modes. It is built from what :func:`lightsim.sparams.write_sparameters`,
:func:`lightsim.eim.write_sparameters_2d`, :mod:`lightsim.gplugins` and
:func:`lightsim.layout.sparameters` return, and written to

* Touchstone (``.sNp``, version 1.1, or 2.0 keywords on request),
* the Lumerical INTERCONNECT text format (``.dat``; read by INTERCONNECT,
  SiEPIC, SAX and Simphony),
* the gplugins ``.npz`` layout (read by ``gplugins.sax``),
* a SAX model (a function of the wavelength in micrometres).

**Phase convention.** lightsim's fields vary in time as exp(-iωt), so a
delay line of group delay τ has S21 ≈ exp(+iωτ): its phase grows with
frequency. INTERCONNECT's text format uses the same convention (the
measured through port of SiEPIC's ring couplers has a phase that grows with
frequency, at the group delay of the straight), and so does SAX
(``sax.models.straight`` is exp(+2πi n L / λ)); those formats are written
as they are. Touchstone files are read by circuit and microwave tools
(scikit-rf, ADS, Spectre) that use exp(+jωt), where the same delay line has
S21 = exp(-jωτ): :meth:`SParameters.to_touchstone` writes the complex
conjugate and :meth:`SParameters.from_touchstone` conjugates back, and the
file says so in a comment. ``conjugate=False`` writes lightsim's values as
they are.

Ports are ``(name, mode)`` pairs; mode numbers start at 0 as in gplugins'
keys (``"o2@0,o1@0"``). Touchstone has no field for port names or modes:
they go into comment lines (``! port 1 = o1@0``) and into a sidecar JSON
file (``<file>.json``) that :meth:`from_touchstone` reads when it is there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

C0 = 299_792_458.0

__all__ = ["SParameters"]


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


class SParameters:
    """An S-matrix over frequency, with named ports.

    Parameters
    ----------
    freqs : array
        Frequencies in Hz (any order; stored ascending).
    s : array
        Complex array of shape ``(len(freqs), N, N)``; ``s[f, i, j]`` is the
        wave leaving port ``i`` for a unit wave into port ``j``, in
        lightsim's exp(-iωt) convention.
    ports : list
        ``N`` port names, or ``(name, mode)`` pairs (mode 0 if a name only).
    computed : array of bool, optional
        ``(N, N)``: which entries were simulated (the others are zero, or
        copies from symmetries). Default: all.
    meta : dict, optional
        Free-form information carried into the files' comments and sidecar.
    """

    def __init__(self, freqs, s, ports, computed=None, meta=None):
        f = np.asarray(freqs, dtype=float).reshape(-1)
        s = np.asarray(s, dtype=complex)
        if s.ndim != 3 or s.shape[0] != f.size or s.shape[1] != s.shape[2]:
            raise ValueError(f"s must have shape (nfreq, N, N); got {s.shape} for {f.size} frequencies")
        order = np.argsort(f)
        self.freqs = f[order]
        self.s = s[order]
        self.ports = [(p, 0) if isinstance(p, str) else (str(p[0]), int(p[1])) for p in ports]
        if len(self.ports) != s.shape[1]:
            raise ValueError(f"{len(self.ports)} ports for a {s.shape[1]} x {s.shape[1]} matrix")
        self.computed = np.ones(s.shape[1:], bool) if computed is None else np.asarray(computed, bool)
        self.meta = dict(meta or {})

    # -- basic views --
    @property
    def nports(self) -> int:
        """Number of ports (port-mode pairs)."""
        return len(self.ports)

    @property
    def wavelengths(self) -> np.ndarray:
        """Vacuum wavelengths in metres, in the order of :attr:`freqs`."""
        return C0 / self.freqs

    @property
    def port_labels(self) -> list[str]:
        """``"name@mode"`` labels, as in gplugins' keys."""
        return [f"{n}@{m}" for n, m in self.ports]

    def index(self, port) -> int:
        """Index of a port given as ``"o1"``, ``"o1@0"`` or ``("o1", 0)``."""
        if isinstance(port, str):
            name, _, mode = port.partition("@")
            port = (name, int(mode or 0))
        return self.ports.index((str(port[0]), int(port[1])))

    def __getitem__(self, key) -> np.ndarray:
        """``sp["o2", "o1"]`` or ``sp["o2@0,o1@0"]``: S(out <- in) over frequency."""
        if isinstance(key, str):
            key = tuple(key.split(","))
        out, inp = key
        return self.s[:, self.index(out), self.index(inp)]

    def __repr__(self) -> str:
        return f"SParameters({self.nports} ports {self.port_labels}, {self.freqs.size} frequencies {self.freqs[0]:.4g}-{self.freqs[-1]:.4g} Hz)"

    def allclose(self, other: "SParameters", rtol=1e-6, atol=1e-9) -> bool:
        """Same ports and frequencies, and S-matrices equal within ``rtol``/``atol``."""
        return self.ports == other.ports and np.allclose(self.freqs, other.freqs, rtol=1e-9) and np.allclose(self.s, other.s, rtol=rtol, atol=atol)

    # -- dictionaries (lightsim.sparams, lightsim.gplugins) --
    @classmethod
    def from_dict(cls, d: dict, wavelength_unit: str | None = None, meta=None) -> "SParameters":
        """From a dict with ``wavelengths`` and keys ``"o2@0,o1@0"`` (output first).

        ``wavelength_unit``: ``"m"`` (lightsim.sparams), ``"um"``
        (gplugins), or ``None`` to tell from the values (below 1e-3: metres).
        Entries the dict does not have are zero and marked not computed.
        Keys that are not S-parameters (``mode_neff``, ...) are ignored.
        """
        wl = np.asarray(d["wavelengths"], dtype=float).reshape(-1)
        unit = wavelength_unit or ("m" if np.max(wl) < 1e-3 else "um")
        wl_m = wl * (1e-6 if unit == "um" else 1.0)
        keys = [k for k in d if isinstance(k, str) and "," in k and "@" in k]
        labels = sorted({p for k in keys for p in k.split(",")}, key=lambda p: (_natural(p.split("@")[0]), int(p.split("@")[1])))
        ports = [(p.split("@")[0], int(p.split("@")[1])) for p in labels]
        pos = {p: i for i, p in enumerate(labels)}
        s = np.zeros((wl.size, len(ports), len(ports)), complex)
        computed = np.zeros((len(ports), len(ports)), bool)
        for k in keys:
            o, i = k.split(",")
            s[:, pos[o], pos[i]] = np.asarray(d[k])
            computed[pos[o], pos[i]] = True
        return cls(C0 / wl_m, s, ports, computed, meta)

    def to_dict(self, wavelength_unit: str = "um") -> dict:
        """The gplugins dict: ``wavelengths`` (µm by default, descending
        frequency order as gplugins writes them: ascending wavelength) and the
        computed entries."""
        order = np.argsort(self.wavelengths)
        wl = self.wavelengths[order] * (1e6 if wavelength_unit == "um" else 1.0)
        out = {}
        for i, a in enumerate(self.port_labels):
            for j, b in enumerate(self.port_labels):
                if self.computed[i, j]:
                    out[f"{a},{b}"] = self.s[order, i, j]
        out["wavelengths"] = wl
        return out

    # -- npz --
    def to_npz(self, path) -> Path:
        """The gplugins ``.npz`` layout (µm, ``"o2@0,o1@0"`` keys), which
        ``gplugins.sax.read.model_from_npz`` reads."""
        path = Path(path)
        np.savez_compressed(path, **self.to_dict("um"))
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def from_npz(cls, path) -> "SParameters":
        """Read a gplugins-layout ``.npz`` file (wavelengths in µm)."""
        with np.load(path) as f:
            return cls.from_dict({k: f[k] for k in f.files}, "um")

    # -- Touchstone --
    def to_touchstone(self, path, version: str = "1.1", freq_unit: str = "GHz", fmt: str = "RI", conjugate: bool = True, z0: float = 50.0, sidecar: bool = True) -> Path:
        """Write a Touchstone file (``.sNp``).

        ``version``: ``"1.1"`` (default) or ``"2.0"`` (adds the
        ``[Version]``, ``[Number of Ports]``, ``[Number of Frequencies]``
        and ``[Network Data]`` keywords; 2.0 orders a 2-port's entries as all
        other sizes, row by row). ``freq_unit``: ``"Hz"``, ``"kHz"``,
        ``"MHz"``, ``"GHz"`` or ``"THz"`` (THz is not in the standard; most
        readers only take up to GHz). ``fmt``: ``"RI"``, ``"MA"`` or
        ``"DB"``. ``conjugate``: write the complex conjugate (the exp(+jωt)
        convention of circuit tools; see the module notes). Port names and
        modes go into comments and, with ``sidecar``, into ``<path>.json``.
        Entries that were not computed are written as 0.
        """
        path = Path(path)
        n = self.nports
        if not re.search(r"\.s\d+p$", path.name, re.I):
            path = path.with_name(path.name + f".s{n}p")
        scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}[freq_unit.lower()]
        fmt = fmt.upper()
        if fmt not in ("RI", "MA", "DB"):
            raise ValueError("fmt must be 'RI', 'MA' or 'DB'")
        s = np.conj(self.s) if conjugate else self.s
        lines = ["! S-parameters written by lightsim"]
        lines.append("! phase convention: " + ("exp(+j w t) (complex conjugate of lightsim's exp(-i w t) values)" if conjugate else "exp(-i w t) (lightsim's values, not conjugated)"))
        for k, (name, mode) in enumerate(self.ports):
            lines.append(f"! port {k + 1} = {name}@{mode}")
        if not self.computed.all():
            miss = [f"S{i + 1},{j + 1}" for i in range(n) for j in range(n) if not self.computed[i, j]]
            lines.append("! not computed (written as 0): " + " ".join(miss))
        for k, v in self.meta.items():
            if v not in (None, ""):
                lines.append(f"! {k}: {v}")
        if version.startswith("2"):
            lines += ["[Version] 2.0", f"# {freq_unit.upper()} S {fmt} R {z0:g}", f"[Number of Ports] {n}"]
            if n == 2:
                lines.append("[Two-Port Data Order] 21_12")
            lines += [f"[Number of Frequencies] {self.freqs.size}", "[Network Data]"]
        else:
            lines.append(f"# {freq_unit.upper()} S {fmt} R {z0:g}")

        def pair(z):
            if fmt == "RI":
                return f"{z.real:.12e} {z.imag:.12e}"
            if fmt == "MA":
                return f"{abs(z):.12e} {np.degrees(np.angle(z)):.12e}"
            return f"{20 * np.log10(max(abs(z), 1e-300)):.12e} {np.degrees(np.angle(z)):.12e}"

        for fi, f in enumerate(self.freqs):
            m = s[fi]
            if n == 2 and not version.startswith("2"):
                vals = [m[0, 0], m[1, 0], m[0, 1], m[1, 1]]  # Touchstone 1: S11 S21 S12 S22
                lines.append(f"{f / scale:.12e} " + " ".join(pair(v) for v in vals))
            elif n <= 2:
                vals = [m[i, j] for i in range(n) for j in range(n)]
                lines.append(f"{f / scale:.12e} " + " ".join(pair(v) for v in vals))
            else:
                for i in range(n):  # one row of the matrix per group of lines, at most 4 entries per line
                    row = [pair(m[i, j]) for j in range(n)]
                    chunks = [row[k : k + 4] for k in range(0, n, 4)]
                    for c, chunk in enumerate(chunks):
                        lead = f"{f / scale:.12e} " if (i == 0 and c == 0) else "  "
                        lines.append(lead + " ".join(chunk))
        if version.startswith("2"):
            lines.append("[End]")
        path.write_text("\n".join(lines) + "\n")
        if sidecar:
            Path(str(path) + ".json").write_text(json.dumps({"format": "lightsim touchstone sidecar", "version": 1, "ports": [{"index": k + 1, "name": nme, "mode": md} for k, (nme, md) in enumerate(self.ports)], "conjugated": bool(conjugate), "computed": self.computed.tolist(), "meta": self.meta}, indent=1))
        return path

    @classmethod
    def from_touchstone(cls, path, conjugate: bool | None = None) -> "SParameters":
        """Read a Touchstone 1.x or 2.0 file (S-parameters, any format and unit).

        Port names and modes come from ``<path>.json`` or from the
        ``! port k = name@mode`` comments, else ``port1``, ``port2``, ...
        ``conjugate``: undo the conjugation of :meth:`to_touchstone`
        (default: as the file's comment or sidecar says, else ``True``,
        since Touchstone files are exp(+jωt)).
        """
        path = Path(path)
        text = path.read_text()
        n_ext = re.search(r"\.s(\d+)p$", path.name, re.I)
        names, conj_comment, nports_kw, order21_12 = {}, None, None, False
        unit, fmt = "GHZ", "MA"
        nums = []
        for raw in text.splitlines():
            line, _, comment = raw.partition("!")
            if comment:
                m = re.match(r"\s*port (\d+) = (.+)@(\d+)\s*$", comment)
                if m:
                    names[int(m.group(1))] = (m.group(2), int(m.group(3)))
                if "phase convention" in comment:
                    conj_comment = "conjugate" in comment
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                toks = line[1:].upper().split()
                for t in toks:
                    if t in ("HZ", "KHZ", "MHZ", "GHZ", "THZ"):
                        unit = t
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t in ("Y", "Z", "G", "H"):
                        raise ValueError(f"{path}: only S-parameter Touchstone files are supported (found {t})")
                continue
            if line.startswith("["):
                kw = line.lower()
                if kw.startswith("[number of ports]"):
                    nports_kw = int(line.split("]")[1])
                elif kw.startswith("[two-port data order]"):
                    order21_12 = "21_12" in line
                continue
            nums.extend(float(x) for x in line.split())
        n = nports_kw or (int(n_ext.group(1)) if n_ext else None)
        if n is None:
            raise ValueError(f"{path}: the number of ports is neither in the name (.sNp) nor in a [Number of Ports] keyword")
        per = 1 + 2 * n * n
        a = np.asarray(nums).reshape(-1, per)
        scale = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9, "THZ": 1e12}[unit]
        f = a[:, 0] * scale
        x, y = a[:, 1::2], a[:, 2::2]
        if fmt == "RI":
            z = x + 1j * y
        elif fmt == "MA":
            z = x * np.exp(1j * np.radians(y))
        else:
            z = 10 ** (x / 20) * np.exp(1j * np.radians(y))
        s = z.reshape(-1, n, n)
        if n == 2 and (not nports_kw or order21_12):  # Touchstone 1 (and 2.0 with 21_12): S11 S21 S12 S22
            s = s.transpose(0, 2, 1)
        side = Path(str(path) + ".json")
        computed, meta = None, {}
        if side.exists():
            info = json.loads(side.read_text())
            names = {p["index"]: (p["name"], int(p["mode"])) for p in info.get("ports", [])}
            conj_comment = info.get("conjugated", conj_comment)
            computed = info.get("computed")
            meta = info.get("meta", {})
        ports = [names.get(k + 1, (f"port{k + 1}", 0)) for k in range(n)]
        if conjugate is None:
            conjugate = True if conj_comment is None else bool(conj_comment)
        return cls(f, np.conj(s) if conjugate else s, ports, computed, meta)

    # -- Lumerical INTERCONNECT text (.dat) --
    def to_lumerical_dat(self, path, mode_label: str = "TE", sides: dict | None = None, include_uncomputed: bool = False) -> Path:
        """Write the INTERCONNECT "Optical N-port S-parameter" text file.

        One block per entry, as INTERCONNECT, SiEPIC, SAX and Simphony read
        it::

            ('o2','TE',1,'o1',1,'transmission')
            (51,3)
            <frequency Hz> <|S|> <phase rad>
            ...

        Mode numbers are written from 1 (INTERCONNECT's), lightsim's mode 0
        is mode 1. The values are lightsim's, not conjugated (INTERCONNECT
        uses the same exp(-iωt) convention). ``sides``
        (``{"o1": "LEFT", ...}``) adds the port-position header lines
        INTERCONNECT uses to place the ports; ``mode_label`` is the
        polarization label of every block. Entries that were not computed
        are left out unless ``include_uncomputed``.
        """
        path = Path(path)
        lines = []
        names = list(dict.fromkeys(n for n, _ in self.ports))
        if sides:
            for n in names:
                lines.append(f'["{n}","{sides.get(n, "LEFT").upper()}"]')
        for j, (ni, mi) in enumerate(self.ports):
            for i, (no, mo) in enumerate(self.ports):
                if not (self.computed[i, j] or include_uncomputed):
                    continue
                v = self.s[:, i, j]
                lines.append(f"('{no}','{mode_label}',{mo + 1},'{ni}',{mi + 1},'transmission')")
                lines.append(f"({self.freqs.size},3)")
                ph = np.unwrap(np.angle(v))
                for f, a, p in zip(self.freqs, np.abs(v), ph):
                    lines.append(f"{f:.12e}\t{a:.12e}\t{p:.12e}")
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def from_lumerical_dat(cls, path) -> "SParameters":
        """Read an INTERCONNECT text S-parameter file (``.dat``, ``.sparam``).

        Port names are kept as written (quotes removed); modes are converted
        to lightsim's numbering (INTERCONNECT's mode 1 is mode 0). A block
        with a group delay (a seventh header field) gets the linear phase
        INTERCONNECT adds. Blocks with sweep parameters are not supported.
        """
        lines = Path(path).read_text().splitlines()
        hdr = re.compile(r"^\s*\(\s*(['\"]?)(.*?)\1\s*,\s*(['\"]?)(.*?)\3\s*,\s*(\d+)\s*,\s*(['\"]?)(.*?)\6\s*,\s*(\d+)\s*,\s*(['\"]?)(.*?)\9\s*(?:,\s*([-+0-9.eE]+)\s*)?\)\s*$")
        blocks = []
        freqs = None
        i = 0
        while i < len(lines):
            m = hdr.match(lines[i])
            if not m:
                i += 1
                continue
            out, mode_out, inp, mode_in, kind = m.group(2), int(m.group(5)), m.group(7), int(m.group(8)), m.group(10)
            gd = m.group(11)
            shp = re.findall(r"\d+", lines[i + 1])
            rows, cols = int(shp[0]), int(shp[1])
            if cols != 3:
                raise ValueError(f"{path}: block ({out} <- {inp}) has {cols} columns; sweep parameters are not supported")
            data = np.array([[float(x) for x in lines[i + 2 + r].split()] for r in range(rows)])
            i += 2 + rows
            if kind.strip().lower() != "transmission":
                continue
            f, amp, ph = data[:, 0], data[:, 1], data[:, 2]
            if gd is not None:
                ph = ph + 2 * np.pi * float(gd) * (f - f[rows // 2])
            order = np.argsort(f)
            f, amp, ph = f[order], amp[order], ph[order]
            if freqs is None:
                freqs = f
            elif f.size != freqs.size or not np.allclose(f, freqs, rtol=1e-9):
                raise ValueError(f"{path}: the blocks have different frequencies")
            blocks.append(((out, mode_out - 1), (inp, mode_in - 1), amp * np.exp(1j * ph)))
        if not blocks:
            raise ValueError(f"{path}: no S-parameter blocks found")
        ports = sorted({b[0] for b in blocks} | {b[1] for b in blocks}, key=lambda p: (_natural(p[0]), p[1]))
        pos = {p: k for k, p in enumerate(ports)}
        s = np.zeros((freqs.size, len(ports), len(ports)), complex)
        computed = np.zeros((len(ports), len(ports)), bool)
        for o, inp, v in blocks:
            s[:, pos[o], pos[inp]] = v
            computed[pos[o], pos[inp]] = True
        return cls(freqs, s, ports, computed)

    # -- SAX --
    def to_sax(self):
        """A SAX model: ``model(wl=1.55)`` returns ``{("o2", "o1"): S, ...}``
        interpolated at the wavelengths ``wl`` (µm), in lightsim's (and
        SAX's) exp(-iωt) convention. Port keys carry the mode
        (``"o1@1"``) when the object has modes other than 0. Uses jax.numpy
        when it is installed, numpy otherwise."""
        try:
            import jax.numpy as xp
        except ImportError:  # pragma: no cover - depends on the environment
            xp = np
        multi = any(m for _, m in self.ports)
        labels = self.port_labels if multi else [n for n, _ in self.ports]
        grid = self.wavelengths * 1e6
        order = np.argsort(grid)
        grid = grid[order]
        s = self.s[order]
        pairs = [(i, j) for i in range(self.nports) for j in range(self.nports) if self.computed[i, j]]

        def model(wl=1.55):
            w = xp.asarray(wl)
            return {(labels[i], labels[j]): xp.interp(w, grid, s[:, i, j].real) + 1j * xp.interp(w, grid, s[:, i, j].imag) for i, j in pairs}

        return model
