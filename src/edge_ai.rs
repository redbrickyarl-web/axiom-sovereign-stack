pub struct EdgeAIPipeline;

impl EdgeAIPipeline {
    pub fn quantize_int4(activations: &[f32]) -> Vec<u8> {
        let mut quantized = Vec::with_capacity(activations.len() / 2);
        for chunk in activations.chunks(2) {
            let val1 = (chunk[0].clamp(-8.0, 7.0) + 8.0) as u8 & 0x0F;
            let val2 = if chunk.len() > 1 {
                (chunk[1].clamp(-8.0, 7.0) + 8.0) as u8 & 0x0F
            } else {
                0
            };
            quantized.push((val2 << 4) | val1);
        }
        quantized
    }

    pub fn dequantize_register_level(packed: &[u8], scale: f32) -> Vec<f32> {
        let mut output = Vec::with_capacity(packed.len() * 2);
        for &byte in packed {
            let low = (byte & 0x0F) as f32 - 8.0;
            let high = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output.push(low * scale);
            output.push(high * scale);
        }
        output
    }
}
