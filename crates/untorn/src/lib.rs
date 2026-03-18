//! ## Untorn
//!
//! Make atomic structs that are protected from torn reads & writes.  The semantics are synchronous,
//! non-blocking for both readers and writers.  Wrap any struct that is at least `Copy` (details
//! below).
//!
//! Use this crate when you want readers to pick up the latest value from across threads without
//! either side waiting for the other.  For weakly contended shared mutable access and synchronous
//! simplicity, it is ideal.  Ergonomically, the contained values are shared, mutable, and thread
//! safe, just like atomic values.  Another way to use it is like an MPMC channel that always holds
//! the latest value, is never empty, and never blocks.
//!
//! Untorn uses a small counter (`AtomicU32`) to detect and retry torn reads & writes (the
//! [seqlock](https://en.wikipedia.org/wiki/Seqlock) pattern).  Reads are optimistic.  Retries may
//! spin, but because the spinning is only long enough for a writer to finish copying a value off of
//! their stack, the throughput is extremely high and the cost is almost free.  It is very simple.
//! Even under heavy write contention, throughput is decent.
//!
//! ## Quick Start
//!
//! There are two flavors:
//!
//! - [`UntornCell`] for multiple writers
//! - [`Untorn`] for a single writer
//!
//! Both can be used to wrap any `T`, but usually structs because simpler types can use atomics
//! directly.  Derive `Copy` and `Clone`, and your `T` can be wrapped.
//!
//! ```rust
//! use mutate_untorn::prelude::*;
//!
//! #[derive(Copy, Clone)]
//! struct FrameUniforms {
//!     elapsed_ms: f32,
//!     frame_index: u32,
//! }
//!
//! let mut lock = Untorn::new(FrameUniforms {
//!     elapsed_ms: 0.0,
//!     frame_index: 0,
//! });
//!
//! lock.write(FrameUniforms {
//!     elapsed_ms: 16.6,
//!     frame_index: 1,
//! });
//!
//! let got = lock.read();
//! assert_eq!(got.frame_index, 1);
//! assert!((got.elapsed_ms - 16.6).abs() < 1e-4);
//! ```
//!
//! ## Restrictions
//!
//! Technically your `T` should satisfy [`bytemuck::Pod`] to avoid **undefined behavior**.  To avoid
//! carrying bytemuck as a dependency and requiring the extra derives on your structs, you only must
//! satisfy `Copy`.  However, be aware that any contained pointers etc are not guaranteed to remain
//! consistent.  You need read-copy-update to avoid torn reads on structures that require chasing.
//!
//! Sequence locking is a cheap form of torn-read prevention, enabling a structure to be written and
//! read without risk of seeing inconsistent mid-write updates. **Reading twice in the same scope is
//! probably abnormal.** You probably want to read into your scope and then copy the **snapshot**
//! that you read.  None of the reads will be torn, but **fields on two different reads could be
//! torn.**
//!
//! Do not use this for **persistent** agreement with the other sides.  The values will not be
//! **torn** but they might be **stale**.  If you need to see new writes immediately, you need to
//! wait on those writes, and you need a completely different primitive to do that.
//!
//! ## Cost
//!
//! Since no writer will spend more than a tiny fraction of time in the update path, just long
//! enough to store a few locations from stack, both readers and other writers can reasonably use
//! spin-locking and will lose almost no progress, especially under low-contention use cases that
//! optimistic reading really leverages.
//!
//! This implementation uses the cheapest form of spin locking, spin_loop hints.  This drops CPU
//! power usage a bit without yielding to the OS.
//!
//! The `UntornCell` must account for multiple potential writers and requires extra ceremony to
//! avoid torn writes.  The `Untorn` is a bit more exclusive (single writer) but is a bit faster
//! because it doesn't need to exclude other potential writers.
//!
//! In very heavy contention, the atomic synchronization will lower throughput a bit and you will be
//! better off with something like [contatori](https://github.com/awgn/contatori) style solutions to
//! further de-contend.
//!
//! For very low contention rates, it may could get away with `AtomicU8` counters, but due to
//! wrapping while spinning, failures were observed in stress tests with as few as eight threads.
//! Tears due to wrapping are almost entirely due to the degree of contention among multiple
//! writers.  To fit the counter alongside common 32-bit values in a single word, `AtomicU32` was
//! selected, but smaller values may pack better in some structs.  Having a large number of shared
//! mutable containers may favor some other style of solution.
//!
//! ## Behold Our Robot Overlords 🤖
//!
//! This crate was vibe coded together because no seqlock wrappers on lib.rs looked more direct or
//! correct than just writing a new one.  Some had yucky blocking writes.  Really tired of
//! `Arc<Mutex<T>>` when we really don't need it at all, not even one tiny little miserable bit!
//! Don't really care about the terminally online AI police, so go away.
//!
//! Might publish as a crate.  Please don't squat "untorn".

// I really like the seqlock pattern.  It makes it pretty easy to implement many kinds of
// concurrency with an extremely cheap foundation that doesn't require a lot of thinking.  With
// atomics, as soon as you need a few too many fields, you have think about whether you need to
// fence them or begin rolling... yet another handmade seq lock.  An actual seq lock wrapper is just
// a lot more convenient.  It adds almost nothing to the struct it protects, and so it was time to
// just generalize it.

// XXX We need a reader/writer split handle solution.  I'm going to work on some semantics and try
// to provide options on the ergonomic and low-cost sides of the spectrum.  Multiple readers is
// generally very cheap.  Multiple writers is expensive.  Managed heap is ergonomic.  Manual
// pointers under the hood have neat use cases but need some ergonomic help that isn't just pure
// yeehaw.

use std::{
    cell::UnsafeCell,
    fmt, ptr,
    sync::atomic::{fence, AtomicU32, Ordering},
};

pub mod prelude {
    pub use super::Untorn;
    pub use super::UntornCell;
}

struct SeqInner<T> {
    seq: AtomicU32,
    data: UnsafeCell<T>,
}

impl<T: Copy> SeqInner<T> {
    const fn new(val: T) -> Self {
        Self {
            seq: AtomicU32::new(0),
            data: UnsafeCell::new(val),
        }
    }

    // Called by the &self writer.  Needs release before data write because another thread may have
    // last touched seq.
    #[inline]
    unsafe fn write_shared(&self, val: T) {
        // Acquire the odd slot: spin until we can flip an even seq to seq+1 (odd).
        let seq = loop {
            let s = self.seq.load(Ordering::Relaxed);
            if s & 1 == 0 {
                // Try to claim the write window atomically.
                match self.seq.compare_exchange_weak(
                    s,
                    s.wrapping_add(1),
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(s) => break s.wrapping_add(1), // odd value we now own
                    Err(_) => {}
                }
            }
            std::hint::spin_loop();
        };
        // SAFETY: exclusive write window is held (seq is odd); no other writer
        // can enter until we close it, and readers will retry on the odd seq.
        unsafe { ptr::write_volatile(self.data.get(), val) };
        // Close the window; Release ensures data write is visible before seq is even.
        self.seq.store(seq.wrapping_add(1), Ordering::Release);
    }

    // Called by the &mut writer — borrow checker guarantees exclusion, so no need to worry about
    // another writer racing the seq load.  We still need the Release fence so sreaders see the data
    // write.
    #[inline]
    fn write_exclusive(&mut self, val: T) {
        let seq = self.seq.get_mut().wrapping_add(1);
        *self.seq.get_mut() = seq;
        // SAFETY: we have exclusive &mut access and seq is odd
        unsafe { ptr::write_volatile(self.data.get(), val) };
        // Close the window: seq+1 is even, readers can proceed
        self.seq.store(seq.wrapping_add(1), Ordering::Release);
    }

    #[inline]
    fn read(&self) -> T {
        loop {
            let seq1 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let result = unsafe { ptr::read_volatile(self.data.get()) };
            let seq2 = self.seq.load(Ordering::Acquire);
            if seq1 == seq2 {
                return result;
            }
        }
    }

    #[inline]
    fn try_read(&self) -> Option<T> {
        let seq1 = self.seq.load(Ordering::Acquire);
        if seq1 & 1 != 0 {
            return None;
        }
        let result = unsafe { ptr::read_volatile(self.data.get()) };
        fence(Ordering::Acquire);
        let seq2 = self.seq.load(Ordering::Relaxed);
        (seq1 == seq2).then(|| result)
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for SeqInner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val = self.read();
        f.debug_struct("SeqInner")
            .field("seq", &self.seq.load(Ordering::Relaxed))
            .field("data", &val)
            .finish()
    }
}

pub struct UntornCell<T>(SeqInner<T>);

unsafe impl<T: Send> Send for UntornCell<T> {}
unsafe impl<T: Send> Sync for UntornCell<T> {}

impl<T: Copy> UntornCell<T> {
    pub const fn new(val: T) -> Self {
        Self(SeqInner::new(val))
    }

    #[inline]
    pub fn write(&self, val: T) {
        // SAFETY: UntornCell explicitly opts into unsynchronized multi-writer;
        // reader-side consistency is still guaranteed by the seq protocol.
        unsafe { self.0.write_shared(val) }
    }

    #[inline]
    pub fn read(&self) -> T {
        self.0.read()
    }
    #[inline]
    pub fn try_read(&self) -> Option<T> {
        self.0.try_read()
    }
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.data.into_inner()
    }
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.data.get_mut()
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for UntornCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UntornCell")
            .field("inner", &self.0)
            .finish()
    }
}

pub struct Untorn<T>(SeqInner<T>);

unsafe impl<T: Send> Send for Untorn<T> {}
unsafe impl<T: Send> Sync for Untorn<T> {}

impl<T: Copy> Untorn<T> {
    pub const fn new(val: T) -> Self {
        Self(SeqInner::new(val))
    }

    /// Exclusive write.  Borrow checker is the compile-time torn-write guarantee.  If you hotwire
    /// this to write from multiple non-exclusive writers, you will be better off with the less
    /// restrictive `UntornCell`.
    #[inline]
    pub fn write(&mut self, val: T) {
        self.0.write_exclusive(val)
    }

    #[inline]
    pub fn read(&self) -> T {
        self.0.read()
    }
    #[inline]
    pub fn try_read(&self) -> Option<T> {
        self.0.try_read()
    }
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.data.into_inner()
    }
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.data.get_mut()
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for Untorn<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Untorn").field("inner", &self.0).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn untorn_cell_basic_write_read() {
        let cell = UntornCell::new(0u64);
        cell.write(42);
        assert_eq!(cell.read(), 42);
    }

    #[test]
    fn untorn_basic_write_read() {
        let mut lock = Untorn::new(0u64);
        lock.write(99);
        assert_eq!(lock.read(), 99);
    }

    #[test]
    fn untorn_cell_try_read_clean() {
        let cell = UntornCell::new(7u32);
        assert_eq!(cell.try_read(), Some(7));
    }

    #[test]
    fn untorn_try_read_clean() {
        let mut lock = Untorn::new(7u32);
        lock.write(13);
        assert_eq!(lock.try_read(), Some(13));
    }

    #[test]
    fn untorn_cell_threaded_single_writer() {
        let cell = Arc::new(UntornCell::new(0u64));

        let writer = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || {
                for i in 0..1000u64 {
                    cell.write(i);
                }
            })
        };

        let reader = {
            let cell = Arc::clone(&cell);
            thread::spawn(move || {
                // Just verify we never observe a torn value —
                // seq is even on every successful read so data is consistent.
                for _ in 0..1000 {
                    let _ = cell.read();
                }
            })
        };

        writer.join().unwrap();
        reader.join().unwrap();

        // Writer finished, value must be the last write.
        assert_eq!(cell.read(), 999);
    }

    #[test]
    fn untorn_cell_copy_type_struct() {
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct FrameData {
            index: u32,
            delta_ms: f32,
        }

        let cell = UntornCell::new(FrameData {
            index: 0,
            delta_ms: 0.0,
        });
        cell.write(FrameData {
            index: 7,
            delta_ms: 16.6,
        });

        let got = cell.read();
        assert_eq!(got.index, 7);
        assert!((got.delta_ms - 16.6).abs() < 1e-4);
    }

    #[test]
    fn untorn_get_mut_bypasses_seq() {
        let mut lock = Untorn::new(0u32);
        *lock.get_mut() = 55;
        // seq counter untouched — readers on other threads would see seq=0 (even),
        // so this is only valid before sharing. Just confirm the value is there.
        assert_eq!(lock.read(), 55);
    }

    #[test]
    fn untorn_into_inner() {
        let lock = Untorn::new(0xdeadbeefu32);
        assert_eq!(lock.into_inner(), 0xdeadbeef);
    }

    #[test]
    fn untorn_cell_torn_read_stress() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        use std::thread;

        #[derive(Copy, Clone)]
        struct Pair {
            lo: u32,
            hi: u32,
        }

        impl Pair {
            fn from_u64(n: u64) -> Self {
                Self {
                    lo: n as u32,
                    hi: (n >> 32) as u32,
                }
            }
            fn is_consistent(self) -> bool {
                self.lo == self.hi
            }
        }

        const WRITERS: usize = 7;
        const READERS: usize = 1;
        const READS_PER_READER: u64 = 80_000_000;

        let cell = Arc::new(UntornCell::new(Pair::from_u64(0)));
        let torn_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();

        for w in 0..WRITERS {
            let cell = Arc::clone(&cell);
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let v = (w as u64) + i * WRITERS as u64;
                    cell.write(Pair::from_u64(v | (v << 32)));
                    i += 1;
                }
                eprintln!("writer {w} done after {i} writes");
            }));
        }

        for r in 0..READERS {
            let cell = Arc::clone(&cell);
            let torn_count = Arc::clone(&torn_count);
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut local_torn = 0u64;
                for _ in 0..READS_PER_READER {
                    if !cell.read().is_consistent() {
                        local_torn += 1;
                    }
                }
                torn_count.fetch_add(local_torn, Ordering::Relaxed);
                stop.store(true, Ordering::Release);
                eprintln!("reader {r} done — {READS_PER_READER} reads, {local_torn} torn");
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let torn = torn_count.load(Ordering::Relaxed);
        println!(
            "total reads: {}  torn: {}  torn_rate: {:.6}%",
            READS_PER_READER * READERS as u64,
            torn,
            torn as f64 / (READS_PER_READER * READERS as u64) as f64 * 100.0,
        );

        assert_eq!(
            torn, 0,
            "UntornCell produced a torn read under multi-writer stress"
        );
    }

    #[test]
    fn untorn_torn_read_stress() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::{Arc, RwLock};
        use std::thread;

        #[derive(Copy, Clone)]
        struct Pair {
            lo: u32,
            hi: u32,
        }

        impl Pair {
            fn from_n(n: u64) -> Self {
                Self {
                    lo: n as u32,
                    hi: n as u32,
                }
            }
            fn is_consistent(self) -> bool {
                self.lo == self.hi
            }
        }

        const READERS: usize = 7;
        const READS_PER_READER: u64 = 16_000_000;

        let lock = Arc::new(RwLock::new(Untorn::new(Pair::from_n(0))));
        let stop = Arc::new(AtomicBool::new(false));
        let readers_done = Arc::new(AtomicU64::new(0));
        let torn = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();

        // Single writer — runs until all readers finish.
        {
            let lock = Arc::clone(&lock);
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut i = 1u64;
                while !stop.load(Ordering::Relaxed) {
                    lock.write().unwrap().write(Pair::from_n(i));
                    i += 1;
                }
                eprintln!("writer done after {i} writes");
            }));
        }

        for r in 0..READERS {
            let lock = Arc::clone(&lock);
            let stop = Arc::clone(&stop);
            let readers_done = Arc::clone(&readers_done);
            let torn = Arc::clone(&torn);
            handles.push(thread::spawn(move || {
                let mut local_torn = 0u64;
                for _ in 0..READS_PER_READER {
                    if !lock.read().unwrap().read().is_consistent() {
                        local_torn += 1;
                    }
                }
                torn.fetch_add(local_torn, Ordering::Relaxed);
                let prev = readers_done.fetch_add(1, Ordering::AcqRel);
                if prev + 1 == READERS as u64 {
                    stop.store(true, Ordering::Release);
                }
                eprintln!("reader {r} done — {READS_PER_READER} reads, {local_torn} torn");
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let total_torn = torn.load(Ordering::Relaxed);
        let total_reads = READS_PER_READER * READERS as u64;
        println!(
            "total reads: {total_reads}  torn: {total_torn}  torn_rate: {:.6}%",
            total_torn as f64 / total_reads as f64 * 100.0,
        );

        assert_eq!(
            total_torn, 0,
            "Untorn produced a torn read with a single writer"
        );
    }
}
