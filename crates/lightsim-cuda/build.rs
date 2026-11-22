// The device kernels are single precision: refuse a lightsim-core built
// with the `f64` feature (a CPU-only build) with a clear message, before the
// type errors that compiling against it would give.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("DEP_LIGHTSIM_CORE_PRECISION").as_deref() == Ok("f64") {
        eprintln!("lightsim-cuda needs the default (f32) lightsim-core: the f64 (double-precision) build is CPU only. Build without --features cuda, or without f64.");
        std::process::exit(1);
    }
}
