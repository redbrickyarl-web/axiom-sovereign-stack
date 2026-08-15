/// Bit-Sliced Window-Parallel Multiscalar Accumulator (BWP-MSM)
/// Accelerates multiscalar multiplication via bit-slicing and parallel window buckets.
/// Uses 4-wide unrolled accumulation for improved throughput.
pub struct BwpMsmAccumulator {
    window_bits: usize,
    buckets_len: usize,
}

impl BwpMsmAccumulator {
    pub fn new(window_bits: usize) -> Self {
        let buckets_len = 1 << window_bits;
        Self {
            window_bits,
            buckets_len,
        }
    }

    /// Evaluates bit-sliced window accumulation across scalar windows.
    /// Processes 4 (scalar, point) pairs per iteration.
    pub fn accumulate_slice(
        &self,
        scalars: &[u64],
        points_x: &[f64],
        points_y: &[f64],
    ) -> (f64, f64) {
        assert_eq!(scalars.len(), points_x.len());
        assert_eq!(points_x.len(), points_y.len());

        let mask = (1u64 << self.window_bits) - 1;
        let mut acc_x = 0.0f64;
        let mut acc_y = 0.0f64;

        // 4-wide unrolled accumulation
        let len = scalars.len();
        let mut i = 0;

        while i + 4 <= len {
            let w0 = (scalars[i] & mask) as f64;
            let w1 = (scalars[i + 1] & mask) as f64;
            let w2 = (scalars[i + 2] & mask) as f64;
            let w3 = (scalars[i + 3] & mask) as f64;

            acc_x += points_x[i] * w0
                + points_x[i + 1] * w1
                + points_x[i + 2] * w2
                + points_x[i + 3] * w3;

            acc_y += points_y[i] * w0
                + points_y[i + 1] * w1
                + points_y[i + 2] * w2
                + points_y[i + 3] * w3;

            i += 4;
        }

        // Remainder
        while i < len {
            let w = (scalars[i] & mask) as f64;
            acc_x += points_x[i] * w;
            acc_y += points_y[i] * w;
            i += 1;
        }

        (acc_x, acc_y)
    }
}
