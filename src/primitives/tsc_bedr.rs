//! TSC-BEDR — Hardware Timestamp Counter Bounded Deferred Reclamation

use std::sync::atomic::{AtomicU64, Ordering};

struct Retired {
    _ptr: *mut u8,
    epoch: u64,
}

pub struct TscBedr {
    current_epoch: AtomicU64,
    retired_list: parking_lot::Mutex<Vec<Retired>>,
}

impl TscBedr {
    pub fn new() -> Self {
        Self {
            current_epoch: AtomicU64::new(1),
            retired_list: parking_lot::Mutex::new(Vec::new()),
        }
    }

    pub fn advance_epoch(&self) -> u64 {
        self.current_epoch.fetch_add(1, Ordering::Release)
    }

    /// Alias used by existing tests.
    pub fn tick(&self) -> u64 {
        self.advance_epoch() + 1
    }

    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    /// # Safety
    ///
    /// The caller must ensure that `ptr` points to a valid allocation that is no longer
    /// accessed concurrently and is ready for deferred reclamation.
    pub unsafe fn retire(&self, ptr: *mut u8) -> bool {
        let epoch = self.current_epoch();
        let mut list = self.retired_list.lock();
        list.push(Retired {
            _ptr: ptr,
            epoch,
        });
        true
    }

    /// # Safety
    ///
    /// The caller must ensure that `safe_epoch` is verified and that reclaiming underlying
    /// pointers will not cause use-after-free conditions for active reader threads.
    pub unsafe fn reclaim(&self, safe_epoch: u64) -> usize {
        let mut list = self.retired_list.lock();
        let initial_len = list.len();
        list.retain(|r| r.epoch > safe_epoch);
        initial_len - list.len()
    }

    pub fn pending(&self) -> usize {
        self.retired_list.lock().len()
    }
}

impl Default for TscBedr {
    fn default() -> Self {
        Self::new()
    }
}
