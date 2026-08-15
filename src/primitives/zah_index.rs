//! ZAH-Index — Zero-probe Atomic Hopscotch Bit-Matrix Index
//!
//! A simplified hopscotch-style open-addressing index that stores
//! presence bits and allows atomic insertion / lookup with bounded probes.

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of slots in the index (must be power of two).
const SLOTS: usize = 64;
/// Hopscotch neighborhood size.
const HOP: usize = 8;

/// A single hopscotch bucket holding a 64-bit presence bitmap
/// and a small neighborhood of keys.
pub struct ZahIndex {
    /// Presence bitmaps for each slot.
    bitmaps: [AtomicU64; SLOTS],
    /// Key storage (0 means empty).
    keys: [AtomicU64; SLOTS],
}

impl ZahIndex {
    pub fn new() -> Self {
        // AtomicU64::new is const, so we can initialize the arrays directly.
        const EMPTY: AtomicU64 = AtomicU64::new(0);
        Self {
            bitmaps: [EMPTY; SLOTS],
            keys: [EMPTY; SLOTS],
        }
    }

    /// Hash a key into a slot index.
    fn hash(key: u64) -> usize {
        // Simple multiplicative hash.
        let h = key.wrapping_mul(0x9E3779B97F4A7C15);
        (h as usize) & (SLOTS - 1)
    }

    /// Attempt to insert a key. Returns true on success, false if no space
    /// found within the hopscotch neighborhood.
    pub fn insert(&self, key: u64) -> bool {
        if key == 0 {
            return false; // 0 is reserved as empty marker
        }

        let start = Self::hash(key);

        // First try to find an empty slot in the neighborhood.
        for i in 0..HOP {
            let idx = (start + i) & (SLOTS - 1);

            // Try to claim an empty key slot.
            match self.keys[idx].compare_exchange(
                0,
                key,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Set the corresponding bit in the start bucket's bitmap.
                    let bit = 1u64 << i;
                    self.bitmaps[start].fetch_or(bit, Ordering::Release);
                    return true;
                }
                Err(_) => continue,
            }
        }

        false // Neighborhood full
    }

    /// Check whether a key is present.
    pub fn contains(&self, key: u64) -> bool {
        if key == 0 {
            return false;
        }

        let start = Self::hash(key);
        let bitmap = self.bitmaps[start].load(Ordering::Acquire);

        for i in 0..HOP {
            if (bitmap & (1u64 << i)) != 0 {
                let idx = (start + i) & (SLOTS - 1);
                if self.keys[idx].load(Ordering::Acquire) == key {
                    return true;
                }
            }
        }
        false
    }

    /// Remove a key if present. Returns true if the key was found and removed.
    pub fn remove(&self, key: u64) -> bool {
        if key == 0 {
            return false;
        }

        let start = Self::hash(key);
        let bitmap = self.bitmaps[start].load(Ordering::Acquire);

        for i in 0..HOP {
            if (bitmap & (1u64 << i)) != 0 {
                let idx = (start + i) & (SLOTS - 1);
                match self.keys[idx].compare_exchange(
                    key,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        let bit = 1u64 << i;
                        self.bitmaps[start].fetch_and(!bit, Ordering::Release);
                        return true;
                    }
                    Err(_) => continue,
                }
            }
        }
        false
    }
}

impl Default for ZahIndex {
    fn default() -> Self {
        Self::new()
    }
}
