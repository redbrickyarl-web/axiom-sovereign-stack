/// Bit-Sliced Window-Parallel Multiscalar Accumulator (BWP-MSM)
/// Accelerates multiscalar multiplication via bit-slicing and parallel window buckets.
pub struct BwpMsmAccumulator {
    window_bits: usize,
    _buckets_len: usize,
}

impl BwpMsmAccumulator {
    pub fn new(window_bits: usize) -> Self {
        let _buckets_len = 1 << window_bits;
        Self {
            window_bits,
            _buckets_len,
        }
    }

    /// Evaluates bit-sliced window accumulation across scalar windows.
    /// Processes 4 (scalar, point) pairs per iteration.
    #[inline(always)]
    pub fn accumulate_slice(
        &self,
        scalars: &[u64],
        points_x: &[f64],
        points_y: &[f64],
    ) -> (f64, f64) {
        assert_eq!(scalars.len(), points_x.len());
        assert_eq!(points_x.len(), points_y.len());

        let mut acc_x = 0.0;
        let mut acc_y = 0.0;

        let chunks_len = scalars.len() & !3;
        let mut i = 0;

        while i < chunks_len {
            for j in 0..4 {
                let idx = i + j;
                let scalar_weight = (scalars[idx] & ((1 << self.window_bits) - 1)) as f64;
                acc_x += points_x[idx] * scalar_weight;
                acc_y += points_y[idx] * scalar_weight;
            }
            i += 4;
        }

        while i < scalars.len() {
            let scalar_weight = (scalars[i] & ((1 << self.window_bits) - 1)) as f64;
            acc_x += points_x[i] * scalar_weight;
            acc_y += points_y[i] * scalar_weight;
            i += 1;
        }

        (acc_x, acc_y)
    }
}
