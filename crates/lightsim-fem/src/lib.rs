//! Finite-element solvers of lightsim on 2-D cross-sections.
//!
//! * [`mesher`]: constrained Delaunay meshes of polygons with region tags,
//!   per-region sizes, refinement near interfaces and quality refinement
//!   (on the `spade` triangulation crate).
//! * [`element`]: Lagrange P1 / P2 (isoparametric on curved edges) and
//!   first-order Nedelec edge elements.
//! * [`sparse`], [`linsolve`], [`eigen`]: sparse assembly, direct (faer) and
//!   preconditioned iterative solvers, a shift-invert Arnoldi eigen-solver.
//! * [`scalar`]: second-order scalar problems -div(K grad u) + c u = f with
//!   Dirichlet, Neumann and convective (Robin) conditions, steady and
//!   transient: heat conduction and electrostatics.
//! * [`maxwell`]: full-vectorial waveguide modes with mixed Nedelec / Lagrange
//!   elements, PML and bends.
//!
//! Everything is computed in `f64` whatever the precision of the FDTD core
//! (`lightsim_core::Real` is the type of the FDTD fields; geometry and
//! post-processing are `f64` in both builds, and so is this crate).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::type_complexity, clippy::large_enum_variant)]

pub mod eigen;
pub mod element;
pub mod linsolve;
pub mod locate;
pub mod mesh;
pub mod mesher;
pub mod quadrature;
pub mod scalar;
pub mod sparse;

pub use mesh::Mesh;
