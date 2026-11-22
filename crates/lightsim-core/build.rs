// Publishes the precision of `Real` to the build scripts of dependent
// crates (DEP_LIGHTSIM_CORE_PRECISION = "f32" or "f64"): lightsim-cuda
// refuses an f64 core before compiling anything.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let p = if std::env::var_os("CARGO_FEATURE_F64").is_some() { "f64" } else { "f32" };
    println!("cargo:precision={p}");
}
