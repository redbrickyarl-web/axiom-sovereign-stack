//! Safe Rust wrapper around the Aethelarch C API.
//!
//! ## Build
//! 1. Build https://github.com/redbrickyarl-web/aethelarch as a static lib.
//! 2. Set `AETHELARCH_LIB` / `AETHELARCH_INCLUDE` (see `build.rs`).
//! 3. Enable Cargo feature `aethelarch`.
//!
//! Without the feature, methods return `AethelarchError::NativeUnavailable`.

#![allow(dead_code)]

use std::fmt;
use std::os::raw::{c_float, c_int};
use std::ptr;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Structured errors for the Aethelarch FFI layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AethelarchError {
    /// `rows` or `cols` was zero, or `rows * cols` overflowed.
    InvalidDimensions { rows: usize, cols: usize },

    /// Native allocation failed (OOM or invalid request).
    AllocationFailed { rows: usize, cols: usize },

    /// `ternary.len()` does not equal `rows * cols`.
    WeightLengthMismatch {
        expected: usize,
        actual: usize,
    },

    /// Encode rejected input (invalid ternary value or +1/−1 overlap).
    EncodeFailed {
        reason: EncodeFailure,
    },

    /// Matrix handle is null / already freed.
    InvalidMatrix,

    /// Activation bit buffer shorter than `act_bytes(cols)`.
    ActivationBufferTooShort {
        required: usize,
        actual: usize,
    },

    /// Output buffer shorter than `rows`.
    OutputBufferTooShort {
        required: usize,
        actual: usize,
    },

    /// Built without `--features aethelarch` or native lib not linked.
    NativeUnavailable,

    /// Native `gemv` returned failure (null inputs inside C).
    GemvFailed,
}

/// Why `encode_dense` / `from_ternary` failed after length checks passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeFailure {
    /// C layer returned false (invalid value outside {{-1,0,1}} or disjointness).
    NativeReject,
    /// Feature off or matrix invalid.
    Unavailable,
}

impl fmt::Display for AethelarchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { rows, cols } => {
                write!(f, "invalid matrix dimensions {rows}x{cols}")
            }
            Self::AllocationFailed { rows, cols } => {
                write!(f, "failed to allocate {rows}x{cols} Aethelarch matrix")
            }
            Self::WeightLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "ternary weight length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::EncodeFailed { reason } => write!(f, "encode failed: {reason:?}"),
            Self::InvalidMatrix => write!(f, "invalid or null matrix handle"),
            Self::ActivationBufferTooShort { required, actual } => {
                write!(
                    f,
                    "activation buffer too short: need {required} bytes, got {actual}"
                )
            }
            Self::OutputBufferTooShort { required, actual } => {
                write!(
                    f,
                    "output buffer too short: need {required} floats, got {actual}"
                )
            }
            Self::NativeUnavailable => {
                write!(
                    f,
                    "Aethelarch native library unavailable (build with --features aethelarch and link libaethelarch)"
                )
            }
            Self::GemvFailed => write!(f, "native gemv failed"),
        }
    }
}

impl std::error::Error for AethelarchError {}

pub type AethelarchResult<T> = Result<T, AethelarchError>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Byte length required for an activation mask of `dim` bits (64-byte padded).
pub fn act_bytes(dim: usize) -> usize {
    let raw = (dim + 7) / 8;
    (raw + 63) & !63
}

fn checked_elems(rows: usize, cols: usize) -> AethelarchResult<usize> {
    if rows == 0 || cols == 0 {
        return Err(AethelarchError::InvalidDimensions { rows, cols });
    }
    rows.checked_mul(cols)
        .ok_or(AethelarchError::InvalidDimensions { rows, cols })
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct AethelarchMatrixRaw {
    pub rows: usize,
    pub cols: usize,
    pub row_stride_bytes: usize,
    pub w_pos: *mut u8,
    pub w_neg: *mut u8,
    pub scale: c_float,
}

pub struct AethelarchMatrix {
    raw: *mut AethelarchMatrixRaw,
}

unsafe impl Send for AethelarchMatrix {}
unsafe impl Sync for AethelarchMatrix {}

impl AethelarchMatrix {
    /// Allocate an empty matrix (weights zeroed).
    pub fn new(rows: usize, cols: usize) -> AethelarchResult<Self> {
        checked_elems(rows, cols)?;

        #[cfg(feature = "aethelarch")]
        {
            unsafe {
                let p = aethelarch_matrix_create(rows, cols);
                if p.is_null() {
                    return Err(AethelarchError::AllocationFailed { rows, cols });
                }
                Ok(Self { raw: p })
            }
        }

        #[cfg(not(feature = "aethelarch"))]
        {
            let _ = (rows, cols);
            Err(AethelarchError::NativeUnavailable)
        }
    }

    /// Allocate and encode ternary weights (row-major, values in {{-1, 0, 1}}).
    pub fn from_ternary(rows: usize, cols: usize, ternary: &[i8]) -> AethelarchResult<Self> {
        let expected = checked_elems(rows, cols)?;
        if ternary.len() != expected {
            return Err(AethelarchError::WeightLengthMismatch {
                expected,
                actual: ternary.len(),
            });
        }
        let mat = Self::new(rows, cols)?;
        mat.encode_dense(ternary)?;
        Ok(mat)
    }

    /// Encode ternary weights into dual bitplanes.
    pub fn encode_dense(&self, ternary: &[i8]) -> AethelarchResult<()> {
        if self.raw.is_null() {
            return Err(AethelarchError::InvalidMatrix);
        }
        let rows = self.rows();
        let cols = self.cols();
        let expected = rows * cols;
        if ternary.len() != expected {
            return Err(AethelarchError::WeightLengthMismatch {
                expected,
                actual: ternary.len(),
            });
        }

        #[cfg(feature = "aethelarch")]
        {
            let ok = unsafe { aethelarch_encode_dense(self.raw, ternary.as_ptr(), rows, cols) };
            if ok {
                Ok(())
            } else {
                Err(AethelarchError::EncodeFailed {
                    reason: EncodeFailure::NativeReject,
                })
            }
        }

        #[cfg(not(feature = "aethelarch"))]
        {
            let _ = ternary;
            Err(AethelarchError::EncodeFailed {
                reason: EncodeFailure::Unavailable,
            })
        }
    }

    pub fn set_scale(&self, scale: f32) -> AethelarchResult<()> {
        if self.raw.is_null() {
            return Err(AethelarchError::InvalidMatrix);
        }
        unsafe {
            (*self.raw).scale = scale;
        }
        Ok(())
    }

    pub fn scale(&self) -> AethelarchResult<f32> {
        if self.raw.is_null() {
            return Err(AethelarchError::InvalidMatrix);
        }
        Ok(unsafe { (*self.raw).scale })
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
/// Always succeeds; uses pure-Rust path when native lib is absent.
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
pub fn gemv(
    mat: &AethelarchMatrix,
    act_bits: &[u8],
    out: &mut [f32],
) -> AethelarchResult<()> {
    if mat.raw.is_null() {
        return Err(AethelarchError::InvalidMatrix);
    }
    let rows = mat.rows();
    let cols = mat.cols();
    let need_act = act_bytes(cols);
    if act_bits.len() < need_act {
        return Err(AethelarchError::ActivationBufferTooShort {
            required: need_act,
            actual: act_bits.len(),
        });
    }
    if out.len() < rows {
        return Err(AethelarchError::OutputBufferTooShort {
            required: rows,
            actual: out.len(),
        });
    }

    #[cfg(feature = "aethelarch")]
    {
        let ok = unsafe { aethelarch_gemv(out.as_mut_ptr(), mat.raw, act_bits.as_ptr()) };
        if ok {
            Ok(())
        } else {
            Err(AethelarchError::GemvFailed)
        }
    }

    #[cfg(not(feature = "aethelarch"))]
    {
        let _ = act_bits;
        Err(AethelarchError::NativeUnavailable)
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
