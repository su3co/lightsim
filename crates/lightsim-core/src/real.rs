//! The floating-point type of the fields, coefficients and stored results.
//!
//! `f32` by default (the GPU engine and the benchmarks); `f64` with the cargo
//! feature `f64`, a CPU-only build in double precision: the CPU engine, the
//! plan, results and monitors, time reversal and both gradient engines. The
//! CUDA crate refuses to compile against an `f64` core. Parameters,
//! geometry, frequencies and post-processing are `f64` in both builds.

/// The real type of fields, update coefficients and recorded data.
#[cfg(not(feature = "f64"))]
pub type Real = f32;
/// The real type of fields, update coefficients and recorded data.
#[cfg(feature = "f64")]
pub type Real = f64;

/// `true` in a build with the `f64` feature.
pub const REAL_IS_F64: bool = cfg!(feature = "f64");
