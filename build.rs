// Optional Aethelarch native link hook.
// Set AETHELARCH_LIB to the directory containing libaethelarch.a
// and AETHELARCH_INCLUDE to the include path when building with --features aethelarch.

fn main() {
    if std::env::var("CARGO_FEATURE_AETHELARCH").is_ok() {
        if let Ok(lib) = std::env::var("AETHELARCH_LIB") {
            println!("cargo:rustc-link-search=native={}", lib);
            println!("cargo:rustc-link-lib=static=aethelarch");
        } else {
            println!("cargo:warning=AETHELARCH_LIB not set; link will fail if feature aethelarch is enabled");
        }
        if let Ok(inc) = std::env::var("AETHELARCH_INCLUDE") {
            println!("cargo:rerun-if-changed={}", inc);
        }
    }
}
