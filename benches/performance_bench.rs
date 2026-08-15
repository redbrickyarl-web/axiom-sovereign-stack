use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use axiom_sovereign_stack::cosmic_queue::{CosmicQueue, PacketFrame};
use axiom_sovereign_stack::EdgeAIPipeline;
use axiom_sovereign_stack::BwpMsmAccumulator;

fn bench_cosmic_queue(c: &mut Criterion) {
    let queue = CosmicQueue::new(1024);
    c.bench_function("cosmic_queue push/pop", |b| {
        b.iter(|| {
            let frame = PacketFrame {
                timestamp: 123456,
                payload: vec![0u8; 32],
            };
            let _ = queue.push(frame);
            let _ = queue.pop();
        })
    });
}

fn bench_edge_ai_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_ai_quantize");

    for size in [64, 256, 1024, 4096].iter() {
        let activations: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.1) - 4.0).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| EdgeAIPipeline::quantize_int4(&activations))
        });
    }
    group.finish();
}

fn bench_edge_ai_dequantize(c: &mut Criterion) {
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1) - 4.0).collect();
    let packed = EdgeAIPipeline::quantize_int4(&activations);

    c.bench_function("edge_ai_dequantize_1024", |b| {
        b.iter(|| EdgeAIPipeline::dequantize_register_level(&packed, 1.0))
    });
}

fn bench_bwp_msm(c: &mut Criterion) {
    let msm = BwpMsmAccumulator::new(4);
    let scalars: Vec<u64> = (0..1024).map(|i| i as u64).collect();
    let xs: Vec<f64> = (0..1024).map(|i| i as f64 * 0.01).collect();
    let ys: Vec<f64> = (0..1024).map(|i| i as f64 * 0.02).collect();

    c.bench_function("bwp_msm_accumulate_1024", |b| {
        b.iter(|| msm.accumulate_slice(&scalars, &xs, &ys))
    });
}

criterion_group!(
    benches,
    bench_cosmic_queue,
    bench_edge_ai_quantize,
    bench_edge_ai_dequantize,
    bench_bwp_msm
);
criterion_main!(benches);
