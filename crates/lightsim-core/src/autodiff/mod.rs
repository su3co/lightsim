//! Gradients of simulation results with respect to the permittivity of a
//! design region, computed inside the engine.
//!
//! The design region is a set of [`DesignGrid`]s: index grids
//! ([`crate::geometry::Shape::IndexGrid`]) painted after every other
//! structure, whose samples are the optimisation variables. Two gradient
//! paths share the same design machinery ([`design`]):
//!
//! * [`session`] — a persistent multi-port, multi-frequency S-matrix session
//!   ([`GradientSession`]). The scene is rasterised and the sources/monitors
//!   of every excitation are compiled once; a design update rewrites only the
//!   coefficients of the design box; the forward and adjoint simulations keep
//!   their design-region DFT fields on the device, where the adjoint product
//!   `(i w eps0 / 4) E_adj E_fwd dV` is formed; the transpose of the
//!   rasterisation maps the per-voxel gradient onto the grid samples.
//! * [`reverse`] — discrete reverse-mode differentiation through the time
//!   loop ([`ReverseSession`]) for time-domain objectives: the adjoint
//!   (transposed) leapfrog is stepped backwards in time while the forward
//!   fields are reconstructed by exact time reversal, and `dL/d eps` is
//!   accumulated step by step, without storing the field history.
//!
//! Both are written against the [`Engine`] trait ([`engine`]); the CPU
//! implementation lives here ([`cpu`]), the GPU one in `lightsim-cuda`.
//!
//! # The discrete adjoint
//!
//! With `E' = ca E + cb Cd H'` and `H' = H - ch Cp E` (`Cp`, `Cd` the curls
//! with the primal / dual spacings), the Yee curls satisfy
//! `Cd^T = V_H Cp V_E^-1`, where `V_E`, `V_H` are the diagonal matrices of
//! the "update volumes" of the E and H samples (products of the spacings the
//! update divides by). The transposed step is therefore the forward step
//! itself, applied to the scaled adjoint variables
//!
//! ```text
//! a^ = V_E^-1 cb a,      b^ = -V_H^-1 ch b,
//! ```
//!
//! run backwards in time with the cotangents of the objective as sources.
//! The same holds for lossy media and for dispersive poles (the pole
//! recursion is diagonal in space). PEC, PMC and periodic boundaries are
//! transposed exactly by it. The CPML is the one piece that is not its own
//! transpose: it convolves the *difference* of the other field with the
//! stretching coefficients of the cell being updated, whereas its transpose
//! convolves the *field* with the coefficients of the cell it comes from and
//! differences the result. The adjoint step therefore runs the forward
//! update kernels with a transposed CPML in the layers
//! ([`cpu::AdjointPml`], the `ad_pml_*` GPU kernels). With it
//! `<y, T^k x> = <(T^k)^T y, x>` holds to rounding with fields inside the
//! layers; with the forward CPML in its place gradients are off by up to a
//! few per cent when scatterers sit within a fraction of a wavelength of
//! the layers.
//!
//! [`cell`] holds the per-cell material updates and their hand-written
//! adjoints; with the `enzyme` cargo feature (nightly toolchain with the
//! Enzyme component) [`enzyme`] generates the same adjoints with
//! `#[autodiff_reverse]` and the tests compare the two.

pub mod cell;
pub mod cpu;
pub mod design;
pub mod engine;
#[cfg(feature = "enzyme")]
pub mod enzyme;
pub mod reverse;
pub mod session;
pub mod testing;

pub use cpu::CpuEngine;
pub use design::{DesignGrid, DesignMap, IndexBox};
pub use engine::{AdjointSource, Engine, EngineFactory, IoSet, ReverseConfig, ReverseOut, RunInfo};
pub use reverse::{Objective, PoleGradient, ReverseSession, ReverseSpec};
pub use session::{Channel, GradientSession, PortSpec, SessionSpec};
