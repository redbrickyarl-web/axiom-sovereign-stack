//! BCP-Scheduler — Lock-free Cascading Bitmask Priority Scheduler
//!
//! Uses a 64-bit bitmask to track active priority levels.
//! Highest priority is found in O(1) via leading-zero count.

pub struct BcpScheduler {
    bitmask: u64,
}

impl BcpScheduler {
    pub fn new() -> Self {
        Self { bitmask: 0 }
    }

    /// Mark a priority level as active (0–63).
    pub fn set_priority(&mut self, level: u32) {
        if level < 64 {
            self.bitmask |= 1u64 << level;
        }
    }

    /// Clear a priority level.
    pub fn clear_priority(&mut self, level: u32) {
        if level < 64 {
            self.bitmask &= !(1u64 << level);
        }
    }

    /// Return the highest active priority level, or None if empty.
    pub fn next_highest_priority(&self) -> Option<u32> {
        if self.bitmask == 0 {
            None
        } else {
            Some(63 - self.bitmask.leading_zeros())
        }
    }

    /// Number of active priority levels.
    pub fn active_count(&self) -> u32 {
        self.bitmask.count_ones()
    }

    /// Returns true if no priorities are set.
    pub fn is_empty(&self) -> bool {
        self.bitmask == 0
    }
}

impl Default for BcpScheduler {
    fn default() -> Self {
        Self::new()
    }
}
