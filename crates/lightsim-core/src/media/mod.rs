//! Material models.
//!
//! * [`materials`] — the [`Medium`](materials::Medium): permittivity,
//!   permeability and conductivity (diagonal or full tensors), Lorentz–Drude–
//!   Debye poles (per axis or oriented), chi2 / chi3 with a Raman response,
//!   gyrotropic poles, director fields, four-level gain, the Sellmeier
//!   library and passive pole fitting;
//! * [`gain2`] — the four-level two-electron gain medium;
//! * [`herm`] — complex (Hermitian) permittivity tensors given directly;
//! * [`sheet`] — zero-thickness conducting sheets (graphene);
//! * [`plugin`] — user-defined material models.
//!
//! Each module is also reachable at the crate root (`lightsim_core::materials`
//! and so on).

pub mod gain2;
pub mod herm;
pub mod materials;
pub mod plugin;
pub mod sheet;
