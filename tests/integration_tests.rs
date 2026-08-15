#[cfg(test)]
mod tests {
    use axiom_sovereign_stack::cosmic_queue::{CosmicQueue, PacketFrame};
    use axiom_sovereign_stack::primitives::{ZcssRing, BcpScheduler, TscBedr, ZahIndex};
    use axiom_sovereign_stack::EdgeAIPipeline;
    use axiom_sovereign_stack::PredatorGate;

    #[test]
    fn test_cosmic_queue_flow() {
        let queue = CosmicQueue::new(4);
        let frame = PacketFrame {
            timestamp: 100,
            payload: vec![1, 2, 3],
        };

        assert!(queue.push(frame.clone()).is_ok());
        let popped = queue.pop().expect("Should pop frame");
        assert_eq!(popped.timestamp, 100);
        assert_eq!(popped.payload, vec![1, 2, 3]);
    }

    #[test]
    fn test_zcss_ring_buffer() {
        let ring = ZcssRing::new(2);
        assert!(ring.push(42).is_ok());
        assert!(ring.push(84).is_ok());
        assert!(ring.push(126).is_err()); // Full

        assert_eq!(ring.pop(), Some(42));
        assert_eq!(ring.pop(), Some(84));
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn test_bcp_scheduler_priorities() {
        let mut scheduler = BcpScheduler::new();
        scheduler.set_priority(5);
        scheduler.set_priority(12);
        scheduler.set_priority(3);

        assert_eq!(scheduler.next_highest_priority(), Some(12));
        assert_eq!(scheduler.active_count(), 3);
        assert!(!scheduler.is_empty());

        scheduler.clear_priority(12);
        assert_eq!(scheduler.next_highest_priority(), Some(5));
    }

    #[test]
    fn test_edge_ai_quantization() {
        let activations = vec![1.5, -2.0, 3.2, -0.5];
        let quantized = EdgeAIPipeline::quantize_int4(&activations);
        assert_eq!(quantized.len(), 2);

        let dequantized = EdgeAIPipeline::dequantize_register_level(&quantized, 1.0);
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_predator_gate_validation() {
        let gate = PredatorGate::new(0xDEADBEEF);
        assert!(gate.validate_order_packet(0x12345678));
    }

    #[test]
    fn test_tsc_bedr_epoch() {
        let bedr = TscBedr::new();
        let e1 = bedr.current_epoch();
        let e2 = bedr.tick();
        assert!(e2 > e1);
        assert_eq!(bedr.pending(), 0);
    }

    #[test]
    fn test_zah_index_insert_contains() {
        let idx = ZahIndex::new();
        assert!(idx.insert(0xABCDu64));
        assert!(idx.contains(0xABCDu64));
        assert!(!idx.contains(0x1234u64));

        assert!(idx.remove(0xABCDu64));
        assert!(!idx.contains(0xABCDu64));
    }
}
