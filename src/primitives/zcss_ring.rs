use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub struct ZcssRing<T> {
    buffer: Vec<UnsafeCell<Option<T>>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Send> Sync for ZcssRing<T> {}
unsafe impl<T: Send> Send for ZcssRing<T> {}

impl<T> ZcssRing<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(None));
        }
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, value: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail.wrapping_sub(head) >= self.capacity {
            return Err(value);
        }

        let idx = tail % self.capacity;
        unsafe {
            *self.buffer[idx].get() = Some(value);
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        let idx = head % self.capacity;
        let val = unsafe { (*self.buffer[idx].get()).take() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        val
    }
}
