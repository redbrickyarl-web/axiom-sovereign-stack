pub struct PredatorGate {
    node_id: u64,
    active: bool,
}

impl PredatorGate {
    pub fn new(node_id: u64) -> Self {
        Self { node_id, active: true }
    }

    pub fn validate_order_packet(&self, packet_sig: u64) -> bool {
        self.active && (packet_sig ^ self.node_id) != 0
    }
}
