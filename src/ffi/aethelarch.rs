//! Safe Rust wrapper around the Aethelarch C API.
//!
//! ## Build
//! 1. Build https://github.com/redbrickyarl-web/aethelarch as a static lib.
//! 2. Point `AETHELARCH_LIB` / `AETHELARCH_INCLUDE` or use the `build.rs` hook.
//! 3. Enable Cargo feature `aethelarch`.
//!
//! Until the native lib is linked, this module compiles as documentation +
//! stub types only when the feature is off.

#![allow(dead_code)]

use std::os::raw::{c_float, c_int};
use std::ptr;

/// Byte length required for an activation mask of `dim` bits (64-byte padded).
pub fn act_bytes(dim: usize) -> usize {
    let raw = (dim + 7) / 8;
    (raw + 63) & !63
}

/// Opaque handle matching `aethelarch_matrix_t` layout (for documentation).
/// Actual FFI uses raw pointers from the C library.
#[repr(C)]
pub struct AethelarchMatrixRaw {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub w_pos: *mut u8,
    pub w_neg: *mut u8,
    pub scale: c_float,
}

/// Safe owned matrix wrapper (requires linked libaethelarch).
pub struct AethelarchMatrix {
    raw: *mut AethelarchMatrixRaw,
}

// Safety: matrix is immutable after encode; C lib is thread-safe for concurrent reads.
unsafe impl Send for AethelarchMatrix {}
unsafe impl Sync for AethelarchMatrix {}

impl AethelarchMatrix {
    /// Allocate a new matrix. Returns None on OOM or invalid dims.
    /// Requires `aethelarch` feature + linked native library.
    #[cfg(feature = "aethelarch")]
    pub fn new(rows: usize, cols: usize) -> Option<Self> {
        unsafe {
            let p = aethelarch_matrix_create(rows, cols);
            if p.is_null() {
                None
            } else {
                Some(Self { raw: p })
            }
        }
    }

    #[cfg(not(feature = "aethelarch"))]
    pub fn new(_rows: usize, _cols: usize) -> Option<Self> {
        None
    }

    pub fn rows(&self) -> usize {
        if self.raw.is_null() {
            return 0;
        }
        unsafe { (*self.raw).rows }
    }

    pub fn cols(&self) -> usize {
        if self.raw.is_null() {
            return 0;
        }
        unsafe { (*self.raw).cols }
    }
}

impl Drop for AethelarchMatrix {
    fn drop(&mut self) {
        #[cfg(feature = "aethelarch")]
        if !self.raw.is_null() {
            unsafe { aethelarch_matrix_free(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// Quantize f32 activations to a padded 1-bit mask.
pub fn quantize_activation(activations: &[f32]) -> Vec<u8> {
    let nbytes = act_bytes(activations.len());
    let mut bits = vec![0u8; nbytes];
    #[cfg(feature = "aethelarch")]
    unsafe {
        aethelarch_quantize_activation(bits.as_mut_ptr(), activations.as_ptr(), activations.len());
    }
    #[cfg(not(feature = "aethelarch"))]
    {
        for (i, &v) in activations.iter().enumerate() {
            if v > 0.0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
    }
    bits
}

/// Run GEMV if matrix and native lib are available.
pub fn gemv(mat: &AethelarchMatrix, act_bits: &[u8], out: &mut [f32]) -> bool {
    if mat.raw.is_null() || out.len() < mat.rows() {
        return false;
    }
    #[cfg(feature = "aethelarch")]
    unsafe {
        return aethelarch_gemv(out.as_mut_ptr(), mat.raw, act_bits.as_ptr());
    }
    #[cfg(not(feature = "aethelarch"))]
    {
        let _ = act_bits;
        false
    }
}

// ---------------------------------------------------------------------------
// FFI declarations (only linked when feature = "aethelarch")
// ---------------------------------------------------------------------------
#[cfg(feature = "aethelarch")]
extern "C" {
    fn aethelarch_matrix_create(rows: usize, cols: usize) -> *mut AethelarchMatrixRaw;
    fn aethelarch_matrix_free(mat: *mut AethelarchMatrixRaw);
    fn aethelarch_quantize_activation(
        dst_bits: *mut u8,
        src_activations: *const c_float,
        dim: usize,
    );
    fn aethelarch_gemv(
        dst_out: *mut c_float,
        mat: *const AethelarchMatrixRaw,
        act_bits: *const u8,
    ) -> bool;
    fn aethelarch_encode_dense(
        dst: *mut AethelarchMatrixRaw,
        src_ternary: *const i8,
        rows: usize,
        cols: usize,
    ) -> bool;
    fn aethelarch_dot_product(
        w_pos: *const u8,
        w_neg: *const u8,
        act_bits: *const u8,
        len_bits: usize,
    ) -> c_int;
}
