//! Physical constants (SI, CODATA 2018).

/// Speed of light in vacuum (m/s).
pub const C0: f64 = 299_792_458.0;
/// Vacuum permeability (H/m).
pub const MU0: f64 = 1.256_637_062_12e-6;
/// Vacuum permittivity (F/m).
pub const EPS0: f64 = 1.0 / (MU0 * C0 * C0);
/// Impedance of free space (Ohm).
pub const ETA0: f64 = MU0 * C0;
/// 2*pi
pub const TAU: f64 = std::f64::consts::TAU;
/// pi
pub const PI: f64 = std::f64::consts::PI;
