use crossbeam_queue::ArrayQueue;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct PacketFrame {
    pub timestamp: u64,
    pub payload: Vec<u8>,
}

pub struct CosmicQueue {
    queue: Arc<ArrayQueue<PacketFrame>>,
}

impl CosmicQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(capacity)),
        }
    }

    pub fn push(&self, frame: PacketFrame) -> Result<(), PacketFrame> {
        self.queue.push(frame)
    }

    pub fn pop(&self) -> Option<PacketFrame> {
        self.queue.pop()
    }
}
