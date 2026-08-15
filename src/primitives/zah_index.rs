//! ZAH-Index — Zero-probe Atomic Hopscotch Bit-Matrix Index
//!
//! A simplified hopscotch-style open-addressing index that stores
//! presence bits and allows atomic insertion / lookup with bounded probes.

use std::sync::atomic::{AtomicU64, Ordering};

/// Number of slots in the index (must be power of two).
const SLOTS: usize = 64;
/// Hopscotch neighborhood size.
const HOP: usize = 8;
/// Empty sentinel (primitive u64, not Atomic, to avoid interior-mutability lint).
const EMPTY: u64 = 0;

/// A single hopscotch bucket holding a 64-bit presence bitmap
/// and a small neighborhood of keys.
pub struct ZahIndex {
    /// Presence bitmaps for each slot.
    bitmaps: Vec<AtomicU64>,
    /// Key storage (0 means empty).
    keys: Vec<AtomicU64>,
}

impl ZahIndex {
    pub fn new() -> Self {
        let mut bitmaps = Vec::with_capacity(SLOTS);
        let mut keys = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            bitmaps.push(AtomicU64::new(EMPTY));
            keys.push(AtomicU64::new(EMPTY));
        }
        Self { bitmaps, keys }
    }

    /// Hash a key into a slot index.
    fn hash(key: u64) -> usize {
        let h = key.wrapping_mul(0x9E3779B97F4A7C15);
        (h as usize) & (SLOTS - 1)
    }

    /// Attempt to insert a key. Returns true on success, false if no space
    /// found within the hopscotch neighborhood.
    pub fn insert(&self, key: u64) -> bool {
        if key == EMPTY {
            return false; // 0 is reserved as empty marker
        }

        let start = Self::hash(key);

        for i in 0..HOP {
            let idx = (start + i) & (SLOTS - 1);

            match self.keys[idx].compare_exchange(
                EMPTY,
                key,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
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
        if key == EMPTY {
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
        if key == EMPTY {
            return false;
        }

        let start = Self::hash(key);
        let bitmap = self.bitmaps[start].load(Ordering::Acquire);

        for i in 0..HOP {
            if (bitmap & (1u64 << i)) != 0 {
                let idx = (start + i) & (SLOTS - 1);
                match self.keys[idx].compare_exchange(
                    key,
                    EMPTY,
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
