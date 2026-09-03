//! Foreign Function Interface to the Aethelarch C microkernel.
//!
//! Aethelarch provides dual-bitplane ternary GEMV (1.58-bit weights).
//! Link against `libaethelarch` built from the sibling/vendored C sources.
//!
//! Enable with feature flag `aethelarch` once the native library is linked.

pub mod aethelarch;

pub use aethelarch::{AethelarchMatrix, gemv, quantize_activation, act_bytes};
