use criterion::{criterion_group, criterion_main, Criterion};
use axiom_sovereign_stack::cosmic_queue::{CosmicQueue, PacketFrame};

fn bench_cosmic_queue(c: &mut Criterion) {
    let queue = CosmicQueue::new(1024);
    c.bench_function("cosmic_queue push/pop", |b| {
        b.iter(|| {
            let frame = PacketFrame { timestamp: 123456, payload: vec![0u8; 32] };
            let _ = queue.push(frame);
            let _ = queue.pop();
        })
    });
}

criterion_group!(benches, bench_cosmic_queue);
criterion_main!(benches);
