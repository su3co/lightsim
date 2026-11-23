//! A small GDSII stream reader: cells, polygons, paths, boxes, references,
//! arrays and text labels, flattened into polygons per (layer, datatype).
//!
//! It depends on nothing but the standard library, so the engine and the
//! command-line program can read a layout without a Python geometry
//! package. What it reads:
//!
//! * `BOUNDARY` (polygons; the repeated closing point is dropped),
//! * `PATH` with `WIDTH` and `PATHTYPE` 0 (flush ends), 1 (round ends,
//!   approximated by [`ROUND_SEGMENTS`] segments per half circle), 2 (ends
//!   extended by half the width) and 4 (`BGNEXTN` / `ENDEXTN`); corners are
//!   mitred, as the GDSII specification describes the outline of a path; a
//!   negative width (absolute, unscaled by references) is used as positive,
//! * `BOX` (as a polygon),
//! * `SREF` and `AREF` with `STRANS` reflection, `MAG` and `ANGLE`,
//! * `TEXT` with its `LAYER`, `TEXTTYPE`, position and `STRING`,
//! * `UNITS`: the user unit and the size of a database unit in metres.
//!
//! `NODE` elements, properties, `PRESENTATION`, text transformations and
//! the absolute-magnification / absolute-angle flags of `STRANS` are read
//! and ignored (references then compose relatively).

use std::collections::{BTreeMap, HashMap};

/// Segments per half circle for the round ends of `PATHTYPE` 1.
pub const ROUND_SEGMENTS: usize = 16;

/// A 2-D affine transformation `p -> a * p + t` (a row-major 2x2 matrix).
#[derive(Clone, Copy, Debug)]
struct Affine {
    a: [f64; 4],
    t: [f64; 2],
}

impl Affine {
    const IDENTITY: Affine = Affine { a: [1.0, 0.0, 0.0, 1.0], t: [0.0, 0.0] };

    /// The transformation of a reference: reflect about x (if `reflect`),
    /// scale by `mag`, rotate by `angle` degrees, translate to `origin`.
    fn reference(reflect: bool, mag: f64, angle_deg: f64, origin: [f64; 2]) -> Affine {
        let (s, c) = exact_sin_cos(angle_deg);
        let r = if reflect { -1.0 } else { 1.0 };
        Affine { a: [mag * c, -mag * s * r, mag * s, mag * c * r], t: origin }
    }

    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        [self.a[0] * p[0] + self.a[1] * p[1] + self.t[0], self.a[2] * p[0] + self.a[3] * p[1] + self.t[1]]
    }

    /// `self` after `inner`: `p -> self(inner(p))`.
    fn compose(&self, inner: &Affine) -> Affine {
        let a = &self.a;
        let b = &inner.a;
        Affine {
            a: [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]],
            t: self.apply(inner.t),
        }
    }

    fn translated(&self, d: [f64; 2]) -> Affine {
        Affine { a: self.a, t: [self.t[0] + d[0], self.t[1] + d[1]] }
    }
}

/// sin and cos of an angle in degrees, exact for multiples of 90.
fn exact_sin_cos(deg: f64) -> (f64, f64) {
    let q = deg / 90.0;
    if (q - q.round()).abs() < 1e-12 {
        match (q.round() as i64).rem_euclid(4) {
            0 => (0.0, 1.0),
            1 => (1.0, 0.0),
            2 => (0.0, -1.0),
            _ => (-1.0, 0.0),
        }
    } else {
        let r = deg.to_radians();
        (r.sin(), r.cos())
    }
}

/// One element of a cell, in database units.
#[derive(Clone, Debug)]
enum Element {
    Polygon { layer: i16, datatype: i16, points: Vec<[f64; 2]> },
    Path { layer: i16, datatype: i16, width: f64, pathtype: i16, ext: [f64; 2], points: Vec<[f64; 2]> },
    Reference { cell: String, reflect: bool, mag: f64, angle: f64, origin: [f64; 2], array: Option<([f64; 2], [f64; 2], u16, u16)> },
    Text { layer: i16, texttype: i16, position: [f64; 2], text: String },
}

/// A text label after flattening (position in metres).
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub text: String,
    pub layer: i16,
    pub texttype: i16,
    pub position: [f64; 2],
}

/// A GDSII library as read from a stream.
#[derive(Clone, Debug, Default)]
pub struct Library {
    pub name: String,
    /// The first `UNITS` number: the size of a database unit in user units.
    pub db_in_user: f64,
    /// Size of a database unit in metres (the second `UNITS` number).
    pub db_in_m: f64,
    cells: BTreeMap<String, Vec<Element>>,
    order: Vec<String>,
}

/// A path after flattening: its centre line and width (metres). Its
/// outline is also among the polygons of its layer.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatPath {
    pub layer: i16,
    pub datatype: i16,
    pub width: f64,
    pub pathtype: i16,
    pub points: Vec<[f64; 2]>,
}

/// Polygons per (layer, datatype), paths and labels of a flattened cell, in metres.
#[derive(Clone, Debug, Default)]
pub struct Flat {
    pub polygons: BTreeMap<(i16, i16), Vec<Vec<[f64; 2]>>>,
    pub paths: Vec<FlatPath>,
    pub labels: Vec<Label>,
}

#[derive(Debug, thiserror::Error)]
pub enum GdsError {
    #[error("GDSII stream ends inside a record")]
    Truncated,
    #[error("GDSII record {0:#04x} has a malformed payload")]
    Malformed(u8),
    #[error("cell '{0}' is referenced but not defined in the library")]
    MissingCell(String),
    #[error("cell '{0}' does not exist; cells: {1}")]
    NoSuchCell(String, String),
    #[error("the reference hierarchy of cell '{0}' is cyclic")]
    Cycle(String),
    #[error("the library has {0} top cells ({1}); name the cell to read")]
    Ambiguous(usize, String),
    #[error("cannot read '{0}': {1}")]
    Io(String, std::io::Error),
}

/// GDSII 8-byte real: sign, excess-64 base-16 exponent, 56-bit mantissa.
fn real8(b: &[u8]) -> f64 {
    let sign = if b[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b[0] & 0x7f) as i32 - 64;
    let mut mant: u64 = 0;
    for &x in &b[1..8] {
        mant = (mant << 8) | x as u64;
    }
    sign * (mant as f64) / 2f64.powi(56) * 16f64.powi(exp)
}

fn i16s(p: &[u8]) -> Vec<i16> {
    p.chunks_exact(2).map(|c| i16::from_be_bytes([c[0], c[1]])).collect()
}

fn i32s(p: &[u8]) -> Vec<i32> {
    p.chunks_exact(4).map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn ascii(p: &[u8]) -> String {
    String::from_utf8_lossy(p).trim_end_matches('\0').to_string()
}

#[derive(Default)]
struct Pending {
    kind: u8,
    layer: i16,
    datatype: i16,
    width: f64,
    pathtype: i16,
    ext: [f64; 2],
    xy: Vec<[f64; 2]>,
    sname: String,
    reflect: bool,
    mag: f64,
    angle: f64,
    colrow: (u16, u16),
    text: String,
}

impl Library {
    /// Parse a GDSII stream.
    pub fn parse(data: &[u8]) -> Result<Library, GdsError> {
        let mut lib = Library { db_in_user: 1e-3, db_in_m: 1e-9, ..Default::default() };
        let mut pos = 0usize;
        let mut cell: Option<(String, Vec<Element>)> = None;
        let mut el: Option<Pending> = None;
        while pos + 4 <= data.len() {
            let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            let rt = data[pos + 2];
            if len < 4 {
                // zero padding after ENDLIB
                if len == 0 {
                    break;
                }
                return Err(GdsError::Malformed(rt));
            }
            if pos + len > data.len() {
                return Err(GdsError::Truncated);
            }
            let p = &data[pos + 4..pos + len];
            pos += len;
            match rt {
                0x02 => lib.name = ascii(p),
                0x03 => {
                    if p.len() < 16 {
                        return Err(GdsError::Malformed(rt));
                    }
                    lib.db_in_user = real8(&p[0..8]);
                    lib.db_in_m = real8(&p[8..16]);
                }
                0x05 => cell = Some((String::new(), vec![])),
                0x06 => {
                    if let Some(c) = cell.as_mut() {
                        c.0 = ascii(p);
                    }
                }
                0x07 => {
                    if let Some((name, els)) = cell.take() {
                        lib.order.push(name.clone());
                        lib.cells.insert(name, els);
                    }
                }
                0x08 | 0x09 | 0x0A | 0x0B | 0x0C | 0x2D | 0x15 => {
                    el = Some(Pending { kind: rt, mag: 1.0, ..Default::default() });
                }
                0x0D => with(&mut el, |e| e.layer = *i16s(p).first().unwrap_or(&0)),
                0x0E | 0x16 | 0x2E => with(&mut el, |e| e.datatype = *i16s(p).first().unwrap_or(&0)),
                0x0F => with(&mut el, |e| e.width = *i32s(p).first().unwrap_or(&0) as f64),
                0x21 => with(&mut el, |e| e.pathtype = *i16s(p).first().unwrap_or(&0)),
                0x30 => with(&mut el, |e| e.ext[0] = *i32s(p).first().unwrap_or(&0) as f64),
                0x31 => with(&mut el, |e| e.ext[1] = *i32s(p).first().unwrap_or(&0) as f64),
                0x10 => {
                    let v = i32s(p);
                    with(&mut el, |e| e.xy = v.chunks_exact(2).map(|c| [c[0] as f64, c[1] as f64]).collect());
                }
                0x12 => with(&mut el, |e| e.sname = ascii(p)),
                0x13 => {
                    let v = i16s(p);
                    if v.len() < 2 {
                        return Err(GdsError::Malformed(rt));
                    }
                    with(&mut el, |e| e.colrow = (v[0] as u16, v[1] as u16));
                }
                0x1A => {
                    let v = i16s(p);
                    with(&mut el, |e| e.reflect = v.first().map(|f| (*f as u16) & 0x8000 != 0).unwrap_or(false));
                }
                0x1B => {
                    if p.len() < 8 {
                        return Err(GdsError::Malformed(rt));
                    }
                    with(&mut el, |e| e.mag = real8(&p[0..8]));
                }
                0x1C => {
                    if p.len() < 8 {
                        return Err(GdsError::Malformed(rt));
                    }
                    with(&mut el, |e| e.angle = real8(&p[0..8]));
                }
                0x19 => with(&mut el, |e| e.text = ascii(p)),
                0x11 => {
                    if let (Some(e), Some(c)) = (el.take(), cell.as_mut()) {
                        if let Some(x) = finish(e)? {
                            c.1.push(x);
                        }
                    }
                }
                0x04 => break,
                _ => {} // HEADER, BGNLIB, PRESENTATION, PROPATTR/VALUE, ELFLAGS, PLEX, ...
            }
        }
        Ok(lib)
    }

    /// Read a GDSII file.
    pub fn read(path: &str) -> Result<Library, GdsError> {
        let data = std::fs::read(path).map_err(|e| GdsError::Io(path.to_string(), e))?;
        Library::parse(&data)
    }

    /// Cell names in the order of the file.
    pub fn cell_names(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Cells that no other cell references, in the order of the file.
    pub fn top_cells(&self) -> Vec<String> {
        let mut used = std::collections::HashSet::new();
        for (name, els) in &self.cells {
            if name.starts_with("$$$CONTEXT_INFO$$$") {
                continue; // KLayout's meta-data cell references the cells it describes
            }
            for e in els {
                if let Element::Reference { cell, .. } = e {
                    used.insert(cell.clone());
                }
            }
        }
        // KLayout's meta-data cell (gdsfactory writes one) is not a layout cell
        self.order.iter().filter(|n| !used.contains(*n) && !n.starts_with("$$$CONTEXT_INFO$$$")).cloned().collect()
    }

    /// Flatten `cell` (default: the only top cell) into polygons per
    /// (layer, datatype) and labels, in metres.
    pub fn flatten(&self, cell: Option<&str>) -> Result<(String, Flat), GdsError> {
        let name = match cell {
            Some(c) => {
                if !self.cells.contains_key(c) {
                    return Err(GdsError::NoSuchCell(c.to_string(), self.order.join(", ")));
                }
                c.to_string()
            }
            None => {
                let tops = self.top_cells();
                if tops.len() != 1 {
                    return Err(GdsError::Ambiguous(tops.len(), tops.join(", ")));
                }
                tops[0].clone()
            }
        };
        let mut flat = Flat::default();
        let scale = Affine { a: [self.db_in_m, 0.0, 0.0, self.db_in_m], t: [0.0, 0.0] };
        let mut stack = vec![];
        self.walk(&name, &scale, &mut flat, &mut stack)?;
        Ok((name, flat))
    }

    fn walk(&self, name: &str, tr: &Affine, out: &mut Flat, stack: &mut Vec<String>) -> Result<(), GdsError> {
        if stack.iter().any(|s| s == name) {
            return Err(GdsError::Cycle(name.to_string()));
        }
        let els = self.cells.get(name).ok_or_else(|| GdsError::MissingCell(name.to_string()))?;
        stack.push(name.to_string());
        for e in els {
            match e {
                Element::Polygon { layer, datatype, points } => {
                    out.polygons.entry((*layer, *datatype)).or_default().push(points.iter().map(|p| tr.apply(*p)).collect());
                }
                Element::Path { layer, datatype, width, pathtype, ext, points } => {
                    if let Some(poly) = path_outline(points, *width, *pathtype, *ext) {
                        out.polygons.entry((*layer, *datatype)).or_default().push(poly.iter().map(|p| tr.apply(*p)).collect());
                    }
                    let scale = (tr.a[0] * tr.a[3] - tr.a[1] * tr.a[2]).abs().sqrt();
                    out.paths.push(FlatPath { layer: *layer, datatype: *datatype, width: width * scale, pathtype: *pathtype, points: points.iter().map(|p| tr.apply(*p)).collect() });
                }
                Element::Text { layer, texttype, position, text } => {
                    out.labels.push(Label { text: text.clone(), layer: *layer, texttype: *texttype, position: tr.apply(*position) });
                }
                Element::Reference { cell, reflect, mag, angle, origin, array } => match array {
                    None => {
                        let t = tr.compose(&Affine::reference(*reflect, *mag, *angle, *origin));
                        self.walk(cell, &t, out, stack)?;
                    }
                    Some((col_end, row_end, cols, rows)) => {
                        let (nc, nr) = ((*cols).max(1) as f64, (*rows).max(1) as f64);
                        let dc = [(col_end[0] - origin[0]) / nc, (col_end[1] - origin[1]) / nc];
                        let dr = [(row_end[0] - origin[0]) / nr, (row_end[1] - origin[1]) / nr];
                        let base = Affine::reference(*reflect, *mag, *angle, *origin);
                        for j in 0..(*rows).max(1) {
                            for i in 0..(*cols).max(1) {
                                let d = [i as f64 * dc[0] + j as f64 * dr[0], i as f64 * dc[1] + j as f64 * dr[1]];
                                let t = tr.compose(&base.translated(d));
                                self.walk(cell, &t, out, stack)?;
                            }
                        }
                    }
                },
            }
        }
        stack.pop();
        Ok(())
    }
}

fn with(el: &mut Option<Pending>, f: impl FnOnce(&mut Pending)) {
    if let Some(e) = el.as_mut() {
        f(e)
    }
}

fn finish(e: Pending) -> Result<Option<Element>, GdsError> {
    Ok(match e.kind {
        0x08 | 0x2D => {
            let mut pts = e.xy;
            if pts.len() > 1 && pts.first() == pts.last() {
                pts.pop();
            }
            if pts.len() < 3 {
                return Ok(None);
            }
            Some(Element::Polygon { layer: e.layer, datatype: e.datatype, points: pts })
        }
        0x09 => Some(Element::Path { layer: e.layer, datatype: e.datatype, width: e.width.abs(), pathtype: e.pathtype, ext: e.ext, points: e.xy }),
        0x0A => {
            let o = *e.xy.first().ok_or(GdsError::Malformed(0x10))?;
            Some(Element::Reference { cell: e.sname, reflect: e.reflect, mag: e.mag, angle: e.angle, origin: o, array: None })
        }
        0x0B => {
            if e.xy.len() < 3 {
                return Err(GdsError::Malformed(0x10));
            }
            Some(Element::Reference { cell: e.sname, reflect: e.reflect, mag: e.mag, angle: e.angle, origin: e.xy[0], array: Some((e.xy[1], e.xy[2], e.colrow.0, e.colrow.1)) })
        }
        0x0C => {
            let o = *e.xy.first().ok_or(GdsError::Malformed(0x10))?;
            Some(Element::Text { layer: e.layer, texttype: e.datatype, position: o, text: e.text })
        }
        _ => None, // NODE
    })
}

/// Outline polygon of a GDSII path (database units).
fn path_outline(points: &[[f64; 2]], width: f64, pathtype: i16, ext: [f64; 2]) -> Option<Vec<[f64; 2]>> {
    // drop repeated points
    let mut pts: Vec<[f64; 2]> = vec![];
    for p in points {
        if pts.last().map(|q: &[f64; 2]| q[0] == p[0] && q[1] == p[1]).unwrap_or(false) {
            continue;
        }
        pts.push(*p);
    }
    if pts.len() < 2 || width <= 0.0 {
        return None;
    }
    let hw = 0.5 * width;
    let n = pts.len();
    let dirs: Vec<[f64; 2]> = (0..n - 1)
        .map(|i| {
            let d = [pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            [d[0] / l, d[1] / l]
        })
        .collect();
    let dir = |i: usize| dirs[i];
    // end extensions
    let (e0, e1) = match pathtype {
        2 => (hw, hw),
        4 => (ext[0], ext[1]),
        _ => (0.0, 0.0),
    };
    let d0 = dir(0);
    let dn = dir(n - 2);
    pts[0] = [pts[0][0] - d0[0] * e0, pts[0][1] - d0[1] * e0];
    pts[n - 1] = [pts[n - 1][0] + dn[0] * e1, pts[n - 1][1] + dn[1] * e1];
    // offset points on the left (+) and right (-) side, mitred at the corners
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = if i == 0 {
            (d0, d0)
        } else if i == n - 1 {
            (dn, dn)
        } else {
            (dir(i - 1), dir(i))
        };
        let na = [-a[1], a[0]];
        let nb = [-b[1], b[0]];
        let m = [na[0] + nb[0], na[1] + nb[1]];
        let dot = m[0] * na[0] + m[1] * na[1];
        let off = if dot.abs() < 1e-12 { na } else { [m[0] / dot, m[1] / dot] }; // miter: hw / cos(theta/2) along the bisector
        left.push([pts[i][0] + hw * off[0], pts[i][1] + hw * off[1]]);
        right.push([pts[i][0] - hw * off[0], pts[i][1] - hw * off[1]]);
    }
    let mut out = vec![];
    if pathtype == 1 {
        // round ends: half circles around the end points
        out.extend(right.iter().copied());
        arc(&mut out, pts[n - 1], hw, dn, false);
        out.extend(left.iter().rev().copied());
        arc(&mut out, pts[0], hw, [-d0[0], -d0[1]], false);
    } else {
        out.extend(right.iter().copied());
        out.extend(left.iter().rev().copied());
    }
    Some(out)
}

/// Interior points of the half circle of radius `r` around `c`, from the
/// right side of direction `d` to its left side, through `c + r d`.
fn arc(out: &mut Vec<[f64; 2]>, c: [f64; 2], r: f64, d: [f64; 2], _closed: bool) {
    let a0 = d[1].atan2(d[0]) - std::f64::consts::FRAC_PI_2;
    for k in 1..ROUND_SEGMENTS {
        let a = a0 + std::f64::consts::PI * k as f64 / ROUND_SEGMENTS as f64;
        out.push([c[0] + r * a.cos(), c[1] + r * a.sin()]);
    }
}

/// Summary of a library: cells, top cells, units, and per cell the layers it uses.
pub fn summary(lib: &Library) -> serde_json::Value {
    let mut layers: HashMap<String, Vec<(i16, i16)>> = HashMap::new();
    for (name, els) in &lib.cells {
        let mut set = std::collections::BTreeSet::new();
        for e in els {
            match e {
                Element::Polygon { layer, datatype, .. } | Element::Path { layer, datatype, .. } => {
                    set.insert((*layer, *datatype));
                }
                _ => {}
            }
        }
        layers.insert(name.clone(), set.into_iter().collect());
    }
    serde_json::json!({
        "library": lib.name,
        "database_unit_m": lib.db_in_m,
        "user_unit_m": lib.db_in_m / lib.db_in_user,
        "cells": lib.order,
        "top_cells": lib.top_cells(),
        "layers_by_cell": layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(rt: u8, dt: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 4) as u16).to_be_bytes().to_vec();
        v.push(rt);
        v.push(dt);
        v.extend_from_slice(payload);
        v
    }

    fn r8(x: f64) -> [u8; 8] {
        // encode a GDSII real (enough for the unit values of the tests)
        if x == 0.0 {
            return [0; 8];
        }
        let mut e = 64i32;
        let mut m = x.abs();
        while m >= 1.0 {
            m /= 16.0;
            e += 1;
        }
        while m < 1.0 / 16.0 {
            m *= 16.0;
            e -= 1;
        }
        let mant = (m * 2f64.powi(56)).round() as u64;
        let mut b = [0u8; 8];
        b[0] = (e as u8) | if x < 0.0 { 0x80 } else { 0 };
        for i in 0..7 {
            b[7 - i] = ((mant >> (8 * i)) & 0xff) as u8;
        }
        b
    }

    fn xy(p: &[(i32, i32)]) -> Vec<u8> {
        p.iter().flat_map(|(x, y)| x.to_be_bytes().into_iter().chain(y.to_be_bytes())).collect()
    }

    fn library() -> Vec<u8> {
        let mut d = vec![];
        d.extend(rec(0x00, 2, &600i16.to_be_bytes()));
        d.extend(rec(0x01, 2, &[0; 24]));
        d.extend(rec(0x02, 6, b"LIB\0"));
        d.extend(rec(0x03, 5, &[r8(1e-3), r8(1e-9)].concat()));
        // cell A: a 1000 x 500 rectangle on 1/0 and a path on 2/0
        d.extend(rec(0x05, 2, &[0; 24]));
        d.extend(rec(0x06, 6, b"A\0"));
        d.extend(rec(0x08, 0, &[]));
        d.extend(rec(0x0D, 2, &1i16.to_be_bytes()));
        d.extend(rec(0x0E, 2, &0i16.to_be_bytes()));
        d.extend(rec(0x10, 3, &xy(&[(0, 0), (1000, 0), (1000, 500), (0, 500), (0, 0)])));
        d.extend(rec(0x11, 0, &[]));
        d.extend(rec(0x09, 0, &[]));
        d.extend(rec(0x0D, 2, &2i16.to_be_bytes()));
        d.extend(rec(0x0E, 2, &0i16.to_be_bytes()));
        d.extend(rec(0x21, 2, &2i16.to_be_bytes()));
        d.extend(rec(0x0F, 3, &100i32.to_be_bytes()));
        d.extend(rec(0x10, 3, &xy(&[(0, 0), (1000, 0), (1000, 1000)])));
        d.extend(rec(0x11, 0, &[]));
        d.extend(rec(0x0C, 0, &[]));
        d.extend(rec(0x0D, 2, &10i16.to_be_bytes()));
        d.extend(rec(0x16, 2, &0i16.to_be_bytes()));
        d.extend(rec(0x10, 3, &xy(&[(10, 20)])));
        d.extend(rec(0x19, 6, b"pin\0"));
        d.extend(rec(0x11, 0, &[]));
        d.extend(rec(0x07, 0, &[]));
        // cell TOP: A rotated by 90 and reflected at (5000, 0); a 2 x 1 array of A
        d.extend(rec(0x05, 2, &[0; 24]));
        d.extend(rec(0x06, 6, b"TOP\0"));
        d.extend(rec(0x0A, 0, &[]));
        d.extend(rec(0x12, 6, b"A\0"));
        d.extend(rec(0x1A, 1, &0x8000u16.to_be_bytes()));
        d.extend(rec(0x1C, 5, &r8(90.0)));
        d.extend(rec(0x10, 3, &xy(&[(5000, 0)])));
        d.extend(rec(0x11, 0, &[]));
        d.extend(rec(0x0B, 0, &[]));
        d.extend(rec(0x12, 6, b"A\0"));
        d.extend(rec(0x13, 2, &[2i16.to_be_bytes(), 1i16.to_be_bytes()].concat()));
        d.extend(rec(0x10, 3, &xy(&[(0, 10000), (4000, 10000), (0, 13000)])));
        d.extend(rec(0x11, 0, &[]));
        d.extend(rec(0x07, 0, &[]));
        d.extend(rec(0x04, 0, &[]));
        d
    }

    #[test]
    fn real8_roundtrip() {
        for x in [1e-3, 1e-9, 90.0, 0.5, 2.0, -45.0] {
            assert!((real8(&r8(x)) - x).abs() <= 1e-15 * x.abs().max(1.0));
        }
    }

    #[test]
    fn flatten_references_and_arrays() {
        let lib = Library::parse(&library()).unwrap();
        assert_eq!(lib.cell_names(), vec!["A", "TOP"]);
        assert_eq!(lib.top_cells(), vec!["TOP"]);
        let (name, flat) = lib.flatten(None).unwrap();
        assert_eq!(name, "TOP");
        let rects = &flat.polygons[&(1, 0)];
        assert_eq!(rects.len(), 3);
        // reflected about x, rotated by 90 deg, moved to (5000, 0): (x, y) -> (5000 + y, x)
        let r = &rects[0];
        let expect = [[5000.0, 0.0], [5000.0, 1000.0], [5500.0, 1000.0], [5500.0, 0.0]];
        for (p, q) in r.iter().zip(expect) {
            assert!((p[0] - q[0] * 1e-9).abs() < 1e-15 && (p[1] - q[1] * 1e-9).abs() < 1e-15, "{r:?}");
        }
        // array: the second column is 2000 dbu to the right
        assert!((rects[2][0][0] - 2000e-9).abs() < 1e-15 && (rects[2][0][1] - 10000e-9).abs() < 1e-15);
        assert_eq!(flat.labels.len(), 3);
        assert_eq!(flat.labels[0].text, "pin");
        assert_eq!(flat.paths.len(), 3);
        assert!((flat.paths[0].width - 100e-9).abs() < 1e-18);
    }

    #[test]
    fn path_outline_extended_and_mitred() {
        let p = path_outline(&[[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0]], 100.0, 2, [0.0, 0.0]).unwrap();
        // right side, then left side reversed; extended by 50 at both ends, mitred corner
        let expect = [[-50.0, -50.0], [1050.0, -50.0], [1050.0, 1050.0], [950.0, 1050.0], [950.0, 50.0], [-50.0, 50.0]];
        assert_eq!(p.len(), expect.len());
        for (a, b) in p.iter().zip(expect) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9, "{p:?}");
        }
    }
}
