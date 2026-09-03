//! Criterion benchmarks for Aethelarch GEMV and activation quantize.
//!
//! Run:
//! ```bash
//! cargo bench --features aethelarch --bench aethelarch_gemv
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use axiom_sovereign_stack::{
    act_bytes, gemv, quantize_activation, AethelarchMatrix,
};

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_ternary(&mut self) -> i8 {
        match self.next_u64() % 3 {
            1 => 1,
            2 => -1,
            _ => 0,
        }
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 16_777_216.0 - 0.5
    }
}

fn bench_gemv_projections(c: &mut Criterion) {
    let mut group = c.benchmark_group("aethelarch_gemv");

    // LLaMA / BitNet-style projection shapes: (name, rows, cols)
    let shapes = [
        ("Attention_4096x4096", 4096usize, 4096usize),
        ("FFN_Up_4096x11008", 11008, 4096),
        ("FFN_Down_11008x4096", 4096, 11008),
    ];

    let mut rng = SimpleRng::new(0xCAFE_BEEF_1337);

    for (name, rows, cols) in shapes {
        let total_weights = rows * cols;
        group.throughput(Throughput::Elements(total_weights as u64));

        let weights: Vec<i8> = (0..total_weights).map(|_| rng.next_ternary()).collect();
        let matrix = match AethelarchMatrix::from_ternary(rows, cols, &weights) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip {name}: matrix construction failed: {e}");
                continue;
            }
        };

        let act_floats: Vec<f32> = (0..cols).map(|_| rng.next_f32()).collect();
        let act_bits = quantize_activation(&act_floats);
        assert!(act_bits.len() >= act_bytes(cols));

        let mut output = vec![0.0f32; rows];

        group.bench_with_input(BenchmarkId::new("projection", name), &matrix, |b, mat| {
            b.iter(|| {
                gemv(
                    black_box(mat),
                    black_box(&act_bits),
                    black_box(&mut output),
                )
                .expect("GEMV must succeed");
            });
        });
    }

    group.finish();
}

fn bench_activation_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("aethelarch_activation");
    let mut rng = SimpleRng::new(0x1234_5678_9ABC);

    for dim in [512usize, 1024, 4096, 11008] {
        group.throughput(Throughput::Elements(dim as u64));

        let act_floats: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();

        group.bench_with_input(BenchmarkId::new("quantize", dim), &dim, |b, _| {
            b.iter(|| {
                let bits = quantize_activation(black_box(&act_floats));
                black_box(bits);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gemv_projections, bench_activation_quantization);
criterion_main!(benches);
