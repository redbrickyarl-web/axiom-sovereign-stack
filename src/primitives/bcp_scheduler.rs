pub struct BcpScheduler {
    bitmask: u64,
}

impl BcpScheduler {
    pub fn new() -> Self {
        Self { bitmask: 0 }
    }

    pub fn set_priority(&mut self, level: u32) {
        if level < 64 {
            self.bitmask |= 1 << level;
        }
    }

    pub fn next_highest_priority(&self) -> Option<u32> {
        if self.bitmask == 0 {
            None
        } else {
            Some(63 - self.bitmask.leading_zeros())
        }
    }
}
