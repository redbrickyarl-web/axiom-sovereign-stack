//! Edge AI Hardware Pipeline
//! INT4 quantization and de-quantization with SIMD-style acceleration.

pub struct EdgeAIPipeline;

impl EdgeAIPipeline {
    /// Quantize f32 activations to packed INT4 (two values per byte).
    /// Processes 8 elements at a time for better throughput.
    pub fn quantize_int4(activations: &[f32]) -> Vec<u8> {
        let len = activations.len();
        let mut quantized = Vec::with_capacity((len + 1) / 2);

        // Process 8 floats at a time (4 output bytes)
        let chunks = activations.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Unrolled 8-wide quantize
            let b0 = pack_two(chunk[0], chunk[1]);
            let b1 = pack_two(chunk[2], chunk[3]);
            let b2 = pack_two(chunk[4], chunk[5]);
            let b3 = pack_two(chunk[6], chunk[7]);
            quantized.extend_from_slice(&[b0, b1, b2, b3]);
        }

        // Handle remaining elements
        for pair in remainder.chunks(2) {
            let lo = pair[0];
            let hi = if pair.len() > 1 { pair[1] } else { 0.0 };
            quantized.push(pack_two(lo, hi));
        }

        quantized
    }

    /// Dequantize packed INT4 back to f32 with scale.
    /// Processes 4 bytes (8 values) at a time.
    pub fn dequantize_register_level(packed: &[u8], scale: f32) -> Vec<f32> {
        let mut output = Vec::with_capacity(packed.len() * 2);

        let chunks = packed.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            // Unrolled 4-byte (8-value) dequantize
            unpack_byte(chunk[0], scale, &mut output);
            unpack_byte(chunk[1], scale, &mut output);
            unpack_byte(chunk[2], scale, &mut output);
            unpack_byte(chunk[3], scale, &mut output);
        }

        for &byte in remainder {
            unpack_byte(byte, scale, &mut output);
        }

        output
    }
}

#[inline(always)]
fn pack_two(a: f32, b: f32) -> u8 {
    let lo = (a.clamp(-8.0, 7.0) + 8.0) as u8 & 0x0F;
    let hi = (b.clamp(-8.0, 7.0) + 8.0) as u8 & 0x0F;
    (hi << 4) | lo
}

#[inline(always)]
fn unpack_byte(byte: u8, scale: f32, out: &mut Vec<f32>) {
    let low = (byte & 0x0F) as f32 - 8.0;
    let high = ((byte >> 4) & 0x0F) as f32 - 8.0;
    out.push(low * scale);
    out.push(high * scale);
}
