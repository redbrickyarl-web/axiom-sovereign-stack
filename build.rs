// Turnkey Aethelarch native build via the `cc` crate.
// No AETHELARCH_LIB / AETHELARCH_INCLUDE env vars required.

fn main() {
    if std::env::var("CARGO_FEATURE_AETHELARCH").is_err() {
        return;
    }

    let mut build = cc::Build::new();
    build
        .include("native/aethelarch/include")
        .file("native/aethelarch/src/quantize.c")
        .file("native/aethelarch/src/kernel_scalar.c")
        .opt_level(3)
        .warnings(false);

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if arch == "aarch64" {
        build.file("native/aethelarch/src/kernel_neon.c");
        // Basic NEON is enough for vcnt; avoid over-specifying if toolchain lacks full 8.2
        build.flag_if_supported("-march=armv8-a");
    } else if arch == "x86_64" {
        build.file("native/aethelarch/src/kernel_avx512.c");
        // Enable AVX-512 + VPOPCNTDQ when the toolchain accepts the flags.
        // Scalar fallback remains if defines are not set.
        build.flag_if_supported("-mavx512f");
        build.flag_if_supported("-mavx512vpopcntdq");
        build.flag_if_supported("-mavx512bw");
        build.flag_if_supported("-mavx512vl");
    }

    build.compile("aethelarch");

    println!("cargo:rerun-if-changed=native/aethelarch");
    println!("cargo:rerun-if-changed=native/aethelarch/src/quantize.c");
    println!("cargo:rerun-if-changed=native/aethelarch/src/kernel_scalar.c");
    println!("cargo:rerun-if-changed=native/aethelarch/src/kernel_neon.c");
    println!("cargo:rerun-if-changed=native/aethelarch/src/kernel_avx512.c");
}
