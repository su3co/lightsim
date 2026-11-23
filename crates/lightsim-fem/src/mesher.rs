//! Quality triangle meshes of 2-D cross-sections made of tagged polygons.
//!
//! Method:
//!
//! 1. A size field `h(x)` is the smallest of the default size, the target
//!    size of every region graded away from it (`size + grading * distance`)
//!    and the sizes of the chosen interface refinements, graded the same way.
//! 2. Every polygon segment is subdivided to the local size (points of a
//!    segment marked as an arc lie on the arc, so curved boundaries are
//!    followed at the mesh resolution) and inserted as a constraint of a
//!    constrained Delaunay triangulation (`spade`, MIT OR Apache-2.0);
//!    crossing and overlapping segments of different polygons are split
//!    at their intersections, so overlapping polygons need no Boolean
//!    operations.
//! 3. Interior points are seeded at the centres of a quadtree refined until
//!    each cell is smaller than `h`, away from the constraints; the
//!    triangulation is then refined to a minimum angle (Ruppert-style
//!    refinement of spade, encroached constraints split at their midpoints).
//! 4. Each triangle takes the tag of the highest-priority (lowest
//!    `priority`) polygon containing its centroid; triangles outside every
//!    polygon are dropped. Vertices on constraints that belong to an arc are
//!    projected onto it and the edges along an arc get curved midpoints for
//!    isoparametric P2.

use crate::mesh::Mesh;
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};
use std::collections::HashMap;

/// A polygon region: an outer ring and optional holes (even-odd rule).
#[derive(Clone, Debug, Default)]
pub struct Region {
    pub rings: Vec<Vec<[f64; 2]>>,
    /// Arc carried by the segment that starts at vertex i of ring r:
    /// `arcs[r][i] = Some([cx, cy, radius])`. Empty: all straight.
    pub arcs: Vec<Vec<Option<[f64; 3]>>>,
    pub tag: u32,
    /// Painting priority: where polygons overlap, the lowest value wins.
    pub priority: i32,
    /// Target edge length inside the region (0: default).
    pub size: f64,
}

/// Refinement near the boundary of some regions.
#[derive(Clone, Debug)]
pub struct Refinement {
    /// Indices of the regions whose boundaries are refined.
    pub regions: Vec<usize>,
    /// Edge length on the boundary.
    pub size: f64,
}

/// Mesher options.
#[derive(Clone, Debug)]
pub struct MeshOptions {
    /// Largest edge length anywhere.
    pub default_size: f64,
    /// Growth of the size per unit distance away from a refined region.
    pub grading: f64,
    /// Minimum angle (degrees) the quality refinement aims at (<= 30).
    pub min_angle: f64,
    pub refinements: Vec<Refinement>,
    /// Upper bound on the number of vertices.
    pub max_vertices: usize,
}

impl Default for MeshOptions {
    fn default() -> Self {
        MeshOptions {
            default_size: f64::INFINITY,
            grading: 0.3,
            min_angle: 25.0,
            refinements: vec![],
            max_vertices: 2_000_000,
        }
    }
}

fn seg_dist2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    qx * qx + qy * qy
}

impl Region {
    /// Even-odd point-in-polygon test over all rings.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        let mut inside = false;
        for ring in &self.rings {
            let n = ring.len();
            let mut j = n - 1;
            for i in 0..n {
                let (a, b) = (ring[i], ring[j]);
                if (a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
                {
                    inside = !inside;
                }
                j = i;
            }
        }
        inside
    }
    fn boundary_dist(&self, p: [f64; 2]) -> f64 {
        let mut d2 = f64::INFINITY;
        for ring in &self.rings {
            let n = ring.len();
            for i in 0..n {
                d2 = d2.min(seg_dist2(p, ring[i], ring[(i + 1) % n]));
            }
        }
        d2.sqrt()
    }
    fn dist(&self, p: [f64; 2]) -> f64 {
        if self.contains(p) {
            0.0
        } else {
            self.boundary_dist(p)
        }
    }
    fn arc(&self, r: usize, i: usize) -> Option<[f64; 3]> {
        self.arcs.get(r).and_then(|a| a.get(i)).copied().flatten()
    }
}

struct SizeField<'a> {
    regions: &'a [Region],
    opts: &'a MeshOptions,
}

impl SizeField<'_> {
    fn h(&self, p: [f64; 2]) -> f64 {
        let g = self.opts.grading;
        let mut h = self.opts.default_size;
        for r in self.regions {
            if r.size > 0.0 && r.size < h {
                h = h.min(r.size + g * r.dist(p));
            }
        }
        for rf in &self.opts.refinements {
            if rf.size < h {
                let d = rf
                    .regions
                    .iter()
                    .map(|&i| self.regions[i].boundary_dist(p))
                    .fold(f64::INFINITY, f64::min);
                h = h.min(rf.size + g * d);
            }
        }
        h
    }
}

/// Segment bucket grid for fast "distance to the nearest constraint" queries.
struct SegGrid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<usize>>,
    segs: Vec<([f64; 2], [f64; 2])>,
}

impl SegGrid {
    fn new(segs: Vec<([f64; 2], [f64; 2])>, bb: [f64; 4], cell: f64) -> SegGrid {
        let nx = (((bb[2] - bb[0]) / cell).ceil() as usize).clamp(1, 2000);
        let ny = (((bb[3] - bb[1]) / cell).ceil() as usize).clamp(1, 2000);
        let cell = ((bb[2] - bb[0]) / nx as f64)
            .max((bb[3] - bb[1]) / ny as f64)
            .max(1e-300);
        let mut buckets = vec![vec![]; nx * ny];
        for (k, (a, b)) in segs.iter().enumerate() {
            let i0 = (((a[0].min(b[0]) - bb[0]) / cell).floor() as isize).clamp(0, nx as isize - 1)
                as usize;
            let i1 = (((a[0].max(b[0]) - bb[0]) / cell).floor() as isize).clamp(0, nx as isize - 1)
                as usize;
            let j0 = (((a[1].min(b[1]) - bb[1]) / cell).floor() as isize).clamp(0, ny as isize - 1)
                as usize;
            let j1 = (((a[1].max(b[1]) - bb[1]) / cell).floor() as isize).clamp(0, ny as isize - 1)
                as usize;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    buckets[j * nx + i].push(k);
                }
            }
        }
        SegGrid {
            x0: bb[0],
            y0: bb[1],
            cell,
            nx,
            ny,
            buckets,
            segs,
        }
    }
    /// True if a constraint lies closer than `r` to `p`.
    fn near(&self, p: [f64; 2], r: f64) -> bool {
        let i0 = (((p[0] - r - self.x0) / self.cell).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let i1 = (((p[0] + r - self.x0) / self.cell).floor() as isize)
            .clamp(0, self.nx as isize - 1) as usize;
        let j0 = (((p[1] - r - self.y0) / self.cell).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        let j1 = (((p[1] + r - self.y0) / self.cell).floor() as isize)
            .clamp(0, self.ny as isize - 1) as usize;
        for j in j0..=j1 {
            for i in i0..=i1 {
                for &k in &self.buckets[j * self.nx + i] {
                    let (a, b) = self.segs[k];
                    if seg_dist2(p, a, b) < r * r {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Statistics of a generated mesh.
#[derive(Clone, Debug, Default)]
pub struct MeshStats {
    pub min_angle_deg: f64,
    pub refinement_complete: bool,
}

/// Mesh a set of tagged polygon regions.
pub fn mesh_regions(regions: &[Region], opts: &MeshOptions) -> Result<(Mesh, MeshStats)> {
    if regions.is_empty() {
        bail!("no regions to mesh");
    }
    let mut bb = [f64::INFINITY, f64::INFINITY, -f64::INFINITY, -f64::INFINITY];
    for r in regions {
        if r.rings.is_empty() || r.rings.iter().any(|ring| ring.len() < 3) {
            bail!(
                "region with tag {} has a ring with fewer than 3 vertices",
                r.tag
            );
        }
        for ring in &r.rings {
            for p in ring {
                if !p[0].is_finite() || !p[1].is_finite() {
                    bail!("region with tag {} has a non-finite vertex", r.tag);
                }
                bb = [
                    bb[0].min(p[0]),
                    bb[1].min(p[1]),
                    bb[2].max(p[0]),
                    bb[3].max(p[1]),
                ];
            }
        }
    }
    let diag = ((bb[2] - bb[0]).powi(2) + (bb[3] - bb[1]).powi(2)).sqrt();
    let mut o = opts.clone();
    if !o.default_size.is_finite() || o.default_size <= 0.0 {
        o.default_size = diag / 10.0;
    }
    let field = SizeField { regions, opts: &o };
    let hmin = regions
        .iter()
        .map(|r| if r.size > 0.0 { r.size } else { f64::INFINITY })
        .chain(o.refinements.iter().map(|r| r.size))
        .fold(o.default_size, f64::min);
    // vertex estimate from the size field sampled on a coarse grid
    let ns = 96usize;
    let (cw, ch) = ((bb[2] - bb[0]) / ns as f64, (bb[3] - bb[1]) / ns as f64);
    let est: f64 = (0..ns * ns)
        .into_par_iter()
        .map(|k| {
            let c = [
                bb[0] + (k % ns) as f64 * cw + 0.5 * cw,
                bb[1] + (k / ns) as f64 * ch + 0.5 * ch,
            ];
            cw * ch / field.h(c).powi(2)
        })
        .sum::<f64>()
        .max(1.0);
    if est > 4.0 * o.max_vertices as f64 {
        bail!(
            "the requested sizes need about {est:.0} vertices (smallest size {hmin:.3e} over a {:.3e} x {:.3e} box); raise the sizes or max_vertices",
            bb[2] - bb[0],
            bb[3] - bb[1]
        );
    }
    let snap = 1e-10 * diag;
    let key = |p: [f64; 2]| ((p[0] / snap).round() as i64, (p[1] / snap).round() as i64);

    // 1-2: constraint segments. The input segments are first split at their
    // mutual intersections and overlaps (so collinear edges of different
    // polygons share their points), then each atomic segment is subdivided
    // once to the local size.
    let mut arcs: Vec<[f64; 3]> = vec![];
    let mut raw: Vec<([f64; 2], [f64; 2], Option<[f64; 3]>)> = vec![];
    for r in regions {
        for (ri, ring) in r.rings.iter().enumerate() {
            let n = ring.len();
            for i in 0..n {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                if ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() < snap {
                    continue;
                }
                let arc = r.arc(ri, i);
                if let Some(c) = arc {
                    if !arcs.iter().any(|x| {
                        (x[0] - c[0]).abs() + (x[1] - c[1]).abs() + (x[2] - c[2]).abs() < snap
                    }) {
                        arcs.push(c);
                    }
                }
                raw.push((a, b, arc));
            }
        }
    }
    let splits: Vec<Vec<f64>> = raw
        .par_iter()
        .enumerate()
        .map(|(i, &(p, q, _))| {
            let d = [q[0] - p[0], q[1] - p[1]];
            let ld = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let mut ts = vec![0.0, 1.0];
            let (lo, hi) = (
                [p[0].min(q[0]) - snap, p[1].min(q[1]) - snap],
                [p[0].max(q[0]) + snap, p[1].max(q[1]) + snap],
            );
            for (j, &(r, s2, _)) in raw.iter().enumerate() {
                if i == j
                    || r[0].max(s2[0]) < lo[0]
                    || r[0].min(s2[0]) > hi[0]
                    || r[1].max(s2[1]) < lo[1]
                    || r[1].min(s2[1]) > hi[1]
                {
                    continue;
                }
                let e = [s2[0] - r[0], s2[1] - r[1]];
                let le = (e[0] * e[0] + e[1] * e[1]).sqrt();
                let den = d[0] * e[1] - d[1] * e[0];
                let w = [r[0] - p[0], r[1] - p[1]];
                if den.abs() <= 1e-12 * ld * le {
                    // parallel: collinear if r lies on the line of p q
                    if (w[0] * d[1] - w[1] * d[0]).abs() / ld <= snap {
                        for x in [r, s2] {
                            let t = ((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1]) / (ld * ld);
                            if t > 1e-12 && t < 1.0 - 1e-12 {
                                ts.push(t);
                            }
                        }
                    }
                    continue;
                }
                let t = (w[0] * e[1] - w[1] * e[0]) / den;
                let u = (w[0] * d[1] - w[1] * d[0]) / den;
                let (et, eu) = (snap / ld, snap / le);
                if t > et && t < 1.0 - et && u >= -eu && u <= 1.0 + eu {
                    ts.push(t);
                }
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ts.dedup_by(|a, b| (*a - *b).abs() * ld < snap);
            ts
        })
        .collect();
    let mut pts: Vec<[f64; 2]> = vec![];
    let mut index: HashMap<(i64, i64), usize> = HashMap::new();
    let mut add_pt = |p: [f64; 2], pts: &mut Vec<[f64; 2]>| -> usize {
        *index.entry(key(p)).or_insert_with(|| {
            pts.push(p);
            pts.len() - 1
        })
    };
    let wrap = |mut d: f64| {
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        d
    };
    let point_at = |p: [f64; 2], q: [f64; 2], arc: Option<[f64; 3]>, t: f64| -> [f64; 2] {
        match arc {
            Some(c) => {
                let ta = (p[1] - c[1]).atan2(p[0] - c[0]);
                let d = wrap((q[1] - c[1]).atan2(q[0] - c[0]) - ta);
                let th = ta + t * d;
                [c[0] + c[2] * th.cos(), c[1] + c[2] * th.sin()]
            }
            None => [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])],
        }
    };
    // atomic segments, deduplicated
    let mut atomic: Vec<([f64; 2], [f64; 2], Option<[f64; 3]>)> = vec![];
    let mut seen: std::collections::HashSet<((i64, i64), (i64, i64))> =
        std::collections::HashSet::new();
    for (k, &(p, q, arc)) in raw.iter().enumerate() {
        let ts = &splits[k];
        for w in ts.windows(2) {
            let (a, b) = (
                if w[0] == 0.0 {
                    p
                } else {
                    point_at(p, q, arc, w[0])
                },
                if w[1] == 1.0 {
                    q
                } else {
                    point_at(p, q, arc, w[1])
                },
            );
            let (ka, kb) = (key(a), key(b));
            if ka == kb {
                continue;
            }
            let kk = if ka < kb { (ka, kb) } else { (kb, ka) };
            if seen.insert(kk) {
                atomic.push((a, b, arc));
            }
        }
    }
    let mut segs: Vec<(usize, usize)> = vec![];
    for &(a, b, arc) in &atomic {
        let mid = point_at(a, b, arc, 0.5);
        let hloc = field.h(a).min(field.h(b)).min(field.h(mid));
        let len = match arc {
            Some(c) => (wrap((b[1] - c[1]).atan2(b[0] - c[0]) - (a[1] - c[1]).atan2(a[0] - c[0]))
                * c[2])
                .abs(),
            None => ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt(),
        };
        let k = ((len / hloc).ceil() as usize).max(1);
        let mut prev = add_pt(a, &mut pts);
        for s in 1..=k {
            let p = if s == k {
                b
            } else {
                point_at(a, b, arc, s as f64 / k as f64)
            };
            let cur = add_pt(p, &mut pts);
            if cur != prev {
                segs.push((prev, cur));
            }
            prev = cur;
        }
    }
    segs.sort_unstable_by_key(|&(a, b)| (a.min(b), a.max(b)));
    segs.dedup_by_key(|&mut (a, b)| (a.min(b), a.max(b)));

    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles = Vec::with_capacity(pts.len());
    for p in &pts {
        handles.push(
            cdt.insert(Point2::new(p[0], p[1]))
                .map_err(|e| anyhow!("vertex insertion failed: {e:?}"))?,
        );
    }
    for &(a, b) in &segs {
        if handles[a] != handles[b] {
            cdt.add_constraint_and_split(handles[a], handles[b], |p| p);
        }
    }

    // 3: interior seeds from a quadtree
    let seg_geo: Vec<([f64; 2], [f64; 2])> = segs.iter().map(|&(a, b)| (pts[a], pts[b])).collect();
    let grid = SegGrid::new(seg_geo, bb, (hmin * 2.0).max(diag / 1000.0));
    let side = (bb[2] - bb[0]).max(bb[3] - bb[1]);
    let mut stack = vec![(bb[0], bb[1], side, 0u32)];
    let mut leaves = vec![];
    while let Some((x, y, s, depth)) = stack.pop() {
        let c = [x + 0.5 * s, y + 0.5 * s];
        if c[0] - 0.5 * s > bb[2] || c[1] - 0.5 * s > bb[3] {
            continue;
        }
        // the size field changes by at most `grading` per unit length, so a
        // cell is final once it is below the smallest h over its extent
        let hc = field.h(c);
        if s > 0.9 * (hc - o.grading * 0.71 * s).max(hc / 2.0) && depth < 24 {
            let hs = 0.5 * s;
            stack.extend([
                (x, y, hs, depth + 1),
                (x + hs, y, hs, depth + 1),
                (x, y + hs, hs, depth + 1),
                (x + hs, y + hs, hs, depth + 1),
            ]);
        } else {
            leaves.push((c, s));
        }
        if leaves.len() > o.max_vertices {
            bail!(
                "more than {} interior points; raise the sizes or max_vertices",
                o.max_vertices
            );
        }
    }
    let seeds: Vec<[f64; 2]> = leaves
        .par_iter()
        .filter_map(|&(c, s)| {
            if c[0] <= bb[0] || c[0] >= bb[2] || c[1] <= bb[1] || c[1] >= bb[3] {
                return None;
            }
            if !regions.iter().any(|r| r.contains(c)) {
                return None;
            }
            if grid.near(c, 0.45 * s) {
                return None;
            }
            Some(c)
        })
        .collect();
    for p in seeds {
        cdt.insert(Point2::new(p[0], p[1]))
            .map_err(|e| anyhow!("vertex insertion failed: {e:?}"))?;
    }

    // size check: split triangles that are still too large (rare)
    for _ in 0..8 {
        let mut extra = vec![];
        for f in cdt.inner_faces() {
            let v = f.vertices();
            let p: Vec<[f64; 2]> = v.iter().map(|h| [h.position().x, h.position().y]).collect();
            let c = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            let lmax = (0..3)
                .map(|k| seg_dist2(p[k], p[(k + 1) % 3], p[(k + 1) % 3]).sqrt())
                .fold(0.0, f64::max);
            if lmax > 1.6 * field.h(c) && regions.iter().any(|r| r.contains(c)) {
                extra.push(c);
            }
        }
        if extra.is_empty() {
            break;
        }
        for p in extra {
            cdt.insert(Point2::new(p[0], p[1]))
                .map_err(|e| anyhow!("vertex insertion failed: {e:?}"))?;
        }
    }

    // quality refinement
    let params = RefinementParameters::<f64>::new()
        .with_angle_limit(AngleLimit::from_deg(o.min_angle.min(32.0)))
        .with_min_required_area(0.02 * hmin * hmin)
        .with_max_additional_vertices(o.max_vertices.saturating_sub(cdt.num_vertices()).max(1000));
    let result = cdt.refine(params);

    // 4: tags, extraction, arcs
    let verts: Vec<[f64; 2]> = cdt
        .vertices()
        .map(|v| [v.position().x, v.position().y])
        .collect();
    let mut on_constraint = vec![false; verts.len()];
    for e in cdt.undirected_edges() {
        if e.is_constraint_edge() {
            for v in e.vertices() {
                on_constraint[v.fix().index()] = true;
            }
        }
    }
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by_key(|&i| regions[i].priority);
    let faces: Vec<[usize; 3]> = cdt
        .inner_faces()
        .map(|f| f.vertices().map(|v| v.fix().index()))
        .collect();
    let tagged: Vec<Option<u32>> = faces
        .par_iter()
        .map(|f| {
            let c = [
                (verts[f[0]][0] + verts[f[1]][0] + verts[f[2]][0]) / 3.0,
                (verts[f[0]][1] + verts[f[1]][1] + verts[f[2]][1]) / 3.0,
            ];
            order
                .iter()
                .find(|&&i| regions[i].contains(c))
                .map(|&i| regions[i].tag)
        })
        .collect();
    let mut remap = vec![usize::MAX; verts.len()];
    let mut nodes = vec![];
    let mut tris = vec![];
    let mut tags = vec![];
    let mut node_on_c = vec![];
    for (f, t) in faces.iter().zip(&tagged) {
        if let Some(t) = t {
            let mut tri = [0usize; 3];
            for k in 0..3 {
                if remap[f[k]] == usize::MAX {
                    remap[f[k]] = nodes.len();
                    nodes.push(verts[f[k]]);
                    node_on_c.push(on_constraint[f[k]]);
                }
                tri[k] = remap[f[k]];
            }
            tris.push(tri);
            tags.push(*t);
        }
    }
    if tris.is_empty() {
        bail!("the mesh is empty: no triangle lies inside a region");
    }
    // Vertices on the chords of arc segments (inserted when the refinement
    // split a constraint) are projected onto their arc, unless that would
    // invert a triangle.
    if !arcs.is_empty() {
        let on_arc = |p: [f64; 2]| {
            arcs.iter().position(|c| {
                (((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt() - c[2]).abs() <= 1e-9 * c[2]
            })
        };
        let chords: Vec<([f64; 2], [f64; 2], usize)> = segs
            .iter()
            .filter_map(|&(i, j)| {
                let (pa, pb) = (pts[i], pts[j]);
                match (on_arc(pa), on_arc(pb)) {
                    (Some(x), Some(y)) if x == y => Some((pa, pb, x)),
                    _ => None,
                }
            })
            .collect();
        let mut vert_tris: Vec<Vec<usize>> = vec![vec![]; nodes.len()];
        for (t, tri) in tris.iter().enumerate() {
            for &v in tri {
                vert_tris[v].push(t);
            }
        }
        for i in 0..nodes.len() {
            if !node_on_c[i] || on_arc(nodes[i]).is_some() {
                continue;
            }
            let p = nodes[i];
            let hit = chords.iter().find(|(pa, pb, _)| {
                let l2 = (pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2);
                seg_dist2(p, *pa, *pb) <= 1e-18 * l2
            });
            if let Some(&(_, _, ai)) = hit {
                let c = arcs[ai];
                let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
                let r = (dx * dx + dy * dy).sqrt();
                if r == 0.0 {
                    continue;
                }
                nodes[i] = [c[0] + dx * c[2] / r, c[1] + dy * c[2] / r];
                let inverted = vert_tris[i]
                    .iter()
                    .any(|&t| crate::mesh::signed_area(&nodes, &tris[t]) <= 0.0);
                if inverted {
                    nodes[i] = p;
                }
            }
        }
    }
    let mut mesh = Mesh::new(nodes, tris, tags)?;
    if !arcs.is_empty() {
        for e in 0..mesh.n_edges() {
            let [a, b] = mesh.edge_tris[e];
            let constrained = b == crate::mesh::NONE || mesh.tags[a] != mesh.tags[b];
            if !constrained {
                continue;
            }
            let [i, j] = mesh.edges[e];
            let (pa, pb) = (mesh.nodes[i], mesh.nodes[j]);
            for c in &arcs {
                let ra = ((pa[0] - c[0]).powi(2) + (pa[1] - c[1]).powi(2)).sqrt();
                let rb = ((pb[0] - c[0]).powi(2) + (pb[1] - c[1]).powi(2)).sqrt();
                let len = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
                if (ra - c[2]).abs() < 1e-9 * c[2] && (rb - c[2]).abs() < 1e-9 * c[2] && len < c[2]
                {
                    let m = [0.5 * (pa[0] + pb[0]) - c[0], 0.5 * (pa[1] + pb[1]) - c[1]];
                    let mr = (m[0] * m[0] + m[1] * m[1]).sqrt();
                    mesh.edge_mid[e] = Some([c[0] + m[0] * c[2] / mr, c[1] + m[1] * c[2] / mr]);
                    break;
                }
            }
        }
    }
    let min_angle = (0..mesh.n_tris())
        .map(|t| mesh.min_angle(t))
        .fold(f64::INFINITY, f64::min)
        .to_degrees();
    Ok((
        mesh,
        MeshStats {
            min_angle_deg: min_angle,
            refinement_complete: result.refinement_complete,
        },
    ))
}

/// Discretise a circle as a region with arc-marked segments.
pub fn circle_region(c: [f64; 2], r: f64, n: usize, tag: u32, priority: i32, size: f64) -> Region {
    let ring: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            [c[0] + r * t.cos(), c[1] + r * t.sin()]
        })
        .collect();
    Region {
        rings: vec![ring],
        arcs: vec![vec![Some([c[0], c[1], r]); n]],
        tag,
        priority,
        size,
    }
}

/// An axis-aligned rectangle region.
pub fn rect_region(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    tag: u32,
    priority: i32,
    size: f64,
) -> Region {
    Region {
        rings: vec![vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
        arcs: vec![],
        tag,
        priority,
        size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_rectangles_and_circle() {
        let regions = vec![
            rect_region(-2.0, -2.0, 2.0, 0.0, 0, 10, 0.0),
            rect_region(-2.0, 0.0, 2.0, 2.0, 1, 10, 0.0),
            rect_region(-0.5, -0.1, 0.5, 0.1, 2, 1, 0.05), // straddles the interface
            circle_region([1.0, 1.0], 0.3, 24, 3, 1, 0.05),
        ];
        let opts = MeshOptions {
            default_size: 0.4,
            ..Default::default()
        };
        let (m, st) = mesh_regions(&regions, &opts).unwrap();
        let area = |tag: u32| {
            (0..m.n_tris())
                .filter(|&t| m.tags[t] == tag)
                .map(|t| m.area(t))
                .sum::<f64>()
        };
        assert!((area(2) - 0.2).abs() < 1e-12, "{}", area(2));
        assert!((area(0) - (8.0 - 0.1)).abs() < 1e-9);
        // straight-sided area: O(h^2) short of pi r^2; isoparametric P2 area: O(h^4)
        let exact = std::f64::consts::PI * 0.09;
        assert!(
            (area(3) - exact).abs() / exact < 5e-3,
            "polygonal circle area {} vs {exact}",
            area(3)
        );
        let rule = crate::quadrature::triangle(6);
        let curved: f64 = (0..m.n_tris())
            .filter(|&t| m.tags[t] == 3)
            .map(|t| {
                rule.points
                    .iter()
                    .zip(&rule.weights)
                    .map(|(l, w)| w * 0.5 * crate::element::geometry(&m, t, l).det)
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (curved - exact).abs() / exact < 2e-5,
            "curved circle area {curved} vs {exact}"
        );
        let total: f64 = (0..m.n_tris()).map(|t| m.area(t)).sum();
        assert!((total - 16.0).abs() < 1e-9);
        assert!(st.min_angle_deg > 20.0, "min angle {}", st.min_angle_deg);
        // the core is meshed at its size, the outside coarser
        let core_h = (0..m.n_tris())
            .filter(|&t| m.tags[t] == 2)
            .map(|t| m.diameter(t))
            .fold(0.0, f64::max);
        assert!(core_h < 0.05 * 1.7, "core h {core_h}");
        assert!(m.n_tris() < 20000, "{} triangles", m.n_tris());
        assert!(
            (0..m.n_edges())
                .filter(|&e| m.edge_mid[e].is_some())
                .count()
                > 20
        );
    }
}
