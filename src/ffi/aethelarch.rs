//! Safe Rust wrapper around the Aethelarch C API.
//!
//! ## Build
//! 1. Build https://github.com/redbrickyarl-web/aethelarch as a static lib.
//! 2. Point `AETHELARCH_LIB` / `AETHELARCH_INCLUDE` or use the `build.rs` hook.
//! 3. Enable Cargo feature `aethelarch`.
//!
//! Without the feature, stubs compile so the rest of the crate still builds.

#![allow(dead_code)]

use std::os::raw::{c_float, c_int};
use std::ptr;

/// Byte length required for an activation mask of `dim` bits (64-byte padded).
pub fn act_bytes(dim: usize) -> usize {
    let raw = (dim + 7) / 8;
    (raw + 63) & !63
}

/// C layout matching `aethelarch_matrix_t`.
#[repr(C)]
pub struct AethelarchMatrixRaw {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub w_pos: *mut u8,
    pub w_neg: *mut u8,
    pub scale: c_float,
}

/// Safe owned matrix wrapper.
pub struct AethelarchMatrix {
    raw: *mut AethelarchMatrixRaw,
}

unsafe impl Send for AethelarchMatrix {}
unsafe impl Sync for AethelarchMatrix {}

impl AethelarchMatrix {
    /// Allocate an empty matrix (weights zeroed). Requires native lib.
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

    /// Allocate and encode ternary weights in one step.
    /// `ternary` must be row-major, length `rows * cols`, values in {{-1, 0, 1}}.
    pub fn from_ternary(rows: usize, cols: usize, ternary: &[i8]) -> Option<Self> {
        if ternary.len() != rows.checked_mul(cols)? {
            return None;
        }
        let mat = Self::new(rows, cols)?;
        if !mat.encode_dense(ternary) {
            return None;
        }
        Some(mat)
    }

    /// Encode ternary weights {{-1,0,+1}} into dual bitplanes.
    /// Returns false on size mismatch, invalid values, or disjointness violation.
    pub fn encode_dense(&self, ternary: &[i8]) -> bool {
        if self.raw.is_null() {
            return false;
        }
        let rows = self.rows();
        let cols = self.cols();
        if ternary.len() != rows * cols {
            return false;
        }
        #[cfg(feature = "aethelarch")]
        unsafe {
            return aethelarch_encode_dense(self.raw, ternary.as_ptr(), rows, cols);
        }
        #[cfg(not(feature = "aethelarch"))]
        {
            let _ = ternary;
            false
        }
    }

    /// Set uniform scale applied in GEMV.
    pub fn set_scale(&self, scale: f32) {
        if self.raw.is_null() {
            return;
        }
        unsafe {
            (*self.raw).scale = scale;
        }
    }

    pub fn scale(&self) -> f32 {
        if self.raw.is_null() {
            return 1.0;
        }
        unsafe { (*self.raw).scale }
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

    pub fn is_valid(&self) -> bool {
        !self.raw.is_null()
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

/// GEMV: `out = scale * W * act_bits`.
pub fn gemv(mat: &AethelarchMatrix, act_bits: &[u8], out: &mut [f32]) -> bool {
    if mat.raw.is_null() || out.len() < mat.rows() {
        return false;
    }
    if act_bits.len() < act_bytes(mat.cols()) {
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
