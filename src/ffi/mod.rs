//! Foreign Function Interface to the Aethelarch C microkernel.
//!
//! Dual-bitplane ternary GEMV (1.58-bit weights).
//! Enable with `--features aethelarch` after linking `libaethelarch`.

pub mod aethelarch;

pub use aethelarch::{
    act_bytes, gemv, quantize_activation, AethelarchMatrix,
};
