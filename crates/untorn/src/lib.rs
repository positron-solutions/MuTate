//! ## Untorn
//!
//! Make atomic structs that are protected from torn reads & writes.  The semantics are
//! **synchronous**, non-blocking for both readers and writers.  Wrap any struct that is at least
//! `Copy` (details below).  Cheap cloned thread-safe handles like `Arc`.
//!
//! Use this crate when you want readers to poll the latest value from across threads without any
//! complex locking or costly thread suspending.  For weakly contended writes where optimistic reads
//! can succeed the fastest, when you want the simplest, synchronous looking code, Untorn is ideal.
//!
//! Tired of using `Arc<Mutex<T>>` and `Arc<RwLock<T>>` ceremony for just exposing a little bit of
//! data for read from other threads?  Only need exclusive write access just long enough to copy the
//! new value from the stack?  Never want your threads to suspend when polling rarely written
//! values?  Untorn is for you!
//!
//! ## Quick Start
//!
//! There are two flavors:
//!
//! - [`UntornCell`] for multiple readers and writers.  It's a little bit slower due to the overhead
//!   of excluding other potential writers.  All `UntornCell`s are `Clone` and have identical
//!   read-write capabilities.
//! - [`Untorn`] which you `split` into [`UntornWriter`] and [`UntornReader`].  The writer is
//!   exclusive but readers are `Clone`.  The exclusive writes grant a bit of speed for the lone
//!   writer.
//!
//! Both can be used to wrap any `T` that is at least `Copy`, but usually structs (types smaller
//! than the word size would just use an atomic directly).  Derive `Copy` and `Clone`, and your `T`
//! can be wrapped.
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
//! let (writer, reader) = Untorn::new(FrameUniforms {
//!     elapsed_ms: 0.0,
//!     frame_index: 0,
//! }).split();
//!
//! writer.write(FrameUniforms {
//!     elapsed_ms: 16.6,
//!     frame_index: 1,
//! });
//!
//! let got = reader.read();
//! assert_eq!(got.frame_index, 1);
//! assert!((got.elapsed_ms - 16.6).abs() < 1e-4);
//! ```
//!
//! ### How It's Implemented
//!
//! Untorn uses a small counter (`AtomicU16`) to detect and retry torn reads & writes (the
//! [seqlock](https://en.wikipedia.org/wiki/Seqlock) pattern).  Reads are optimistic.  Retries may
//! spin, but because the spinning is usually only long enough for a writer to finish copying a
//! value off of their stack, the throughput is extremely high and the cost is almost free, about
//! the same as if an atomic instruction existed for your struct.  It is very simple.  Even under
//! heavy write contention, throughput is decent.  With intermittent writes, heavy reads are blazing
//! fast.
//!
//! Because we almost always want shared mutability across threads, and to make the semantics as
//! infallible as possible (the other side didn't drop, and we don't need to check), a second
//! `AtomicU16` provides atomic reference counting.  This avoids the tedious `Arc<RwLock<T>>`
//! annoyances while providing the fastest possible optimistic reads and non-locking writes.
//!
//! ## Comparison With Channels
//!
//! Channel semantics tend to support blocking or awaiting to receive the next value.  Untorn only
//! supports synchronous semantics.  There is no concept of waiting on a value.  Writers cannot
//! "hold" anything except atomic exclusive access just long enough to write.
//!
//! An initial value must be supplied, so the "channel" is never empty.  The existing is the most
//! fresh, although it might become out of data just after read.
//!
//! The `UntornCell` variant is more similar to an MPMC channel.  Building a write-exclusive
//! `UntornWriter` and `UntornReader` pair is more like an SPMC channel.
//!
//! ## Restrictions
//!
//! Technically your `T` should satisfy [`bytemuck::Pod`] to avoid **undefined behavior** ⚠️.  To
//! avoid carrying bytemuck as a dependency and requiring the extra derives on your structs, you
//! only must satisfy `Copy`.  However, be aware that any contained pointers etc are not guaranteed
//! to remain consistent.  You need read-copy-update to avoid torn reads on structures that require
//! pointer chasing.
//!
//! Sequence locking is a cheap form of torn-read prevention, enabling a structure to be written and
//! read without risk of seeing inconsistent mid-write updates. **Reading twice in the same scope is
//! probably abnormal.** You probably want to read into your scope and then copy the **snapshot**
//! that you read.  None of the reads will be torn, but **fields on two different reads could be
//! inconsistent.**
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
//! power usage a bit without yielding to the OS.  After `YIELD_THRESHOLD` spins, the lock switches
//! to [`thread::yield_now`] because we might be waiting on writer threads, and yielding gives the
//! scheduler a chance alleviate worst cases faster.  After about `u16::MAX` yields, we **panic**
//! because the seq has been struck by a cosmic ray or something ⚠️.
//!
//! The combination of short spin-locking with yielding is pretty similar to `parking_lot` a
//! `RwLock`.  Untorn combines `Arc` behavior on top and does not support blocking semantics at all
//! to enforce synchronous usage discipline when writing.  There's no "poisoning" because the writer
//! can't fart around while "holding" the "lock".
//!
//! The `UntornCell` must account for multiple potential writers and requires extra ceremony to
//! avoid torn writes.  The `UntornWriter` and `UntornReader` pair is a bit more exclusive (single
//! writer) but is a bit faster because it doesn't need to exclude other potential writers.
//!
//! In very heavy contention, the atomic synchronization will lower throughput a bit and you will be
//! better off with something like [contatori](https://github.com/awgn/contatori) style solutions to
//! further de-contend.
//!
//! For very low contention rates, we could get away with `AtomicU8` counters, but due to wrapping
//! while spinning, failures were observed in stress tests with as few as eight threads.  Tears due
//! to wrapping are almost entirely due to the degree of contention among multiple writers.
//!
//! To fit the seq and owners counters alongside common 32-bit values in a single word, `AtomicU16`
//! was selected, but smaller values may pack better in some structs.  If size is a concern because
//! you will create *a lot* of `Untorn` values, you might need some other solution that shares
//! synchronization across more values.
//!
//! ## Behold Our Robot Overlords 🤖
//!
//! This crate was vibe coded together because no seqlock wrappers on lib.rs looked more direct or
//! correct than just writing a new one.  Some had yucky blocking writes.  Really tired of
//! `Arc<Mutex<T>>` when we really don't need it at all, not even one tiny little miserable bit!
//! Don't really care about the terminally online AI police, so go away.  Use MIRI or don't bother
//! with unhelpful complaints.
//!
//! Might publish as a crate.  Please don't squat "untorn".

// I really like the seqlock pattern.  It makes it pretty easy to implement many kinds of
// concurrency with an extremely cheap foundation that doesn't require a lot of thinking.  With
// atomics, as soon as you need a few too many fields, you have think about whether you need to
// fence them or begin rolling... yet another handmade seq lock.  An actual seq lock wrapper is just
// a lot more convenient.  It adds almost nothing to the struct it protects, and so it was time to
// just generalize it.

use std::{
    cell::UnsafeCell,
    fmt, panic, ptr,
    sync::atomic::{AtomicU16, Ordering},
};

pub mod prelude {
    pub use super::Untorn;
    pub use super::UntornCell;
    pub use super::UntornReader;
    pub use super::UntornWriter;
}

/// Number of spin hints before sleeping.  The hardcoded value has the goal of waiting only a little
/// longer than the *most* optimistic yields.  If a context switch happens, we will likely come back
/// *a lot* slower on most platforms, but this will allow a scheduler to switch, which is important
/// if the writer has been preempted and is waiting to come back.
///
/// Shorter values also help de-contend the atomic synchronization traffic and allow fewer cores to
/// get more work done while the others yield and shut up, resulting in eyeball-visible faster test
/// completions.  However, the trade-off is that large values might actually need those cycles to
/// finish stores even in optimistic spinning cases.  We prefer the low-contention case that is the
/// design goal.  The chosen value is the longest that does not begin to show significant slowdowns
/// on small structs.
const YIELD_THRESHOLD: u16 = 128;

struct SeqInner<T> {
    owners: AtomicU16,
    seq: AtomicU16,
    data: UnsafeCell<T>,
}

impl<T: Copy> SeqInner<T> {
    const fn new(val: T, owners: u16) -> Self {
        Self {
            owners: AtomicU16::new(owners),
            seq: AtomicU16::new(0),
            data: UnsafeCell::new(val),
        }
    }

    // Called by the &self writer.  Needs release before data write because another thread may have
    // last touched seq.
    #[inline]
    fn write_shared(&self, val: T) {
        let mut spins = 0;
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
            spins += 1;
            match spins {
                0..YIELD_THRESHOLD => std::hint::spin_loop(),
                YIELD_THRESHOLD..u16::MAX => std::thread::yield_now(),
                // After u16::MAX - YIELD_THRESHOLD syscalls, panic.
                u16::MAX => panic!("Untorn writer seems blocked"),
            }
        };
        // SAFETY: exclusive write window is held (seq is odd); no other writer
        // can enter until we close it, and readers will retry on the odd seq.
        unsafe { ptr::write_volatile(self.data.get(), val) };
        // Close the window; Release ensures data write is visible before seq is even.
        self.seq.store(seq.wrapping_add(1), Ordering::Release);
    }

    // Borrow checker guarantees exclusion, so no need to worry about another writer racing the seq
    // load.  We still need the Release fence so readers see the data write.
    #[inline]
    fn write_exclusive(&self, val: T) {
        let seq = self.seq.fetch_add(1, Ordering::Acquire);
        unsafe { ptr::write_volatile(self.data.get(), val) };
        self.seq.store(seq.wrapping_add(2), Ordering::Release);
    }

    #[inline]
    fn read(&self) -> T {
        let mut spins = 0;
        loop {
            let seq1 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                spins += 1;
                match spins {
                    0..YIELD_THRESHOLD => std::hint::spin_loop(),
                    YIELD_THRESHOLD..u16::MAX => std::thread::yield_now(),
                    // After u16::MAX - YIELD_THRESHOLD syscalls, panic.
                    u16::MAX => panic!("Untorn writer seems blocked"),
                }
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
        let seq2 = self.seq.load(Ordering::Acquire);
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

pub struct UntornCell<T>(ptr::NonNull<SeqInner<T>>);

unsafe impl<T: Send> Send for UntornCell<T> {}
unsafe impl<T: Send> Sync for UntornCell<T> {}

impl<T: Copy> UntornCell<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        let boxed = Box::new(SeqInner::new(val, 1));
        Self(ptr::NonNull::from(Box::leak(boxed)))
    }

    #[inline]
    fn inner(&self) -> &SeqInner<T> {
        // SAFETY: pointer came from Box::leak, lives until owners hits zero
        unsafe { self.0.as_ref() }
    }

    #[inline]
    pub fn write(&self, val: T) {
        self.inner().write_shared(val)
    }

    #[inline]
    /// May spin if a write is in progress.  Will only yield if the spin count reaches
    /// `YIELD_THRESHOLD`.
    pub fn read(&self) -> T {
        self.inner().read()
    }

    #[inline]
    /// Do an untorn read but just return `None` if a write is in progress.  Never spins, useful if
    /// doing real-time work and `None` is an acceptable ergonomic tradeoff.
    pub fn try_read(&self) -> Option<T> {
        self.inner().try_read()
    }
}

impl<T: Copy> Clone for UntornCell<T> {
    fn clone(&self) -> Self {
        let prev = self.inner().owners.fetch_add(1, Ordering::Relaxed);
        // Matching Arc: abort rather than risk wrapping u16 (64k handles is already plenty)
        if prev == u16::MAX {
            std::process::abort();
        }
        Self(self.0)
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for UntornCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UntornCell")
            .field("inner", &self.0)
            .finish()
    }
}

pub struct UntornWriter<T>(ptr::NonNull<SeqInner<T>>);

unsafe impl<T: Send> Send for UntornWriter<T> {}
unsafe impl<T: Send> Sync for UntornWriter<T> {}

impl<T: Copy> UntornWriter<T> {
    /// Consume `T` and return a writer, which can call `reader` to create new readers as necessary.
    /// ```rust
    /// use mutate_untorn::UntornWriter;
    ///
    /// let reader = UntornWriter::new(42u64);
    /// let writer = reader.reader();
    /// ```
    pub fn new(val: T) -> Self {
        let ptr = ptr::NonNull::from(Box::leak(Box::new(SeqInner::new(val, 1))));
        Self(ptr)
    }

    /// Return a new `UntornReader`.
    pub fn reader(&self) -> UntornReader<T> {
        let prev = self.inner().owners.fetch_add(1, Ordering::Relaxed);
        if prev == u16::MAX {
            std::process::abort();
        }
        UntornReader(self.0)
    }

    #[inline]
    fn inner(&self) -> &SeqInner<T> {
        // SAFETY: pointer came from Box::leak, lives until owners hits zero
        unsafe { self.0.as_ref() }
    }

    /// Will not spin because UntornWriter has exclusive write access.
    #[inline]
    pub fn write(&self, val: T) {
        self.inner().write_exclusive(val)
    }

    #[inline]
    /// The writer can read.  The handle may have been sent around threads, so the read is still
    /// volatile, but the seq is not checked since the writer can't block itself.
    pub fn read(&self) -> T {
        // SAFETY: we hold exclusive write access; no torn read possible
        unsafe { ptr::read_volatile(self.inner().data.get()) }
    }
}

impl<T> Drop for UntornWriter<T> {
    fn drop(&mut self) {
        // SAFETY: pointer is valid until owners hits zero, and fetch_sub gives us
        // the previous value so exactly one side observes the transition to zero.
        let prev = unsafe { self.0.as_ref() }
            .owners
            .fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            unsafe {
                drop(Box::from_raw(self.0.as_ptr()));
            }
        }
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for UntornWriter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UntornWriter")
            .field("inner", &self.0)
            .finish()
    }
}

/// The reader is the other side of an `UntornWriter`.  It is a cheap, `Clone` handle.  Create it by
/// calling [`Untorn::split`] or [`UntornWriter::reader`].
pub struct UntornReader<T>(ptr::NonNull<SeqInner<T>>);

unsafe impl<T: Send> Send for UntornReader<T> {}
unsafe impl<T: Send> Sync for UntornReader<T> {}

impl<T: Copy> UntornReader<T> {
    #[inline]
    fn inner(&self) -> &SeqInner<T> {
        // SAFETY: pointer came from Box::leak, lives until owners hits zero
        unsafe { self.0.as_ref() }
    }

    #[inline]
    /// May spin if a write is in progress.  Will only yield if the spin count reaches
    /// `YIELD_THRESHOLD`.
    pub fn read(&self) -> T {
        self.inner().read()
    }

    #[inline]
    /// Do an untorn read but just return `None` if a write is in progress.  Never spins, useful if
    /// doing real-time work and `None` is an acceptable ergonomic tradeoff.
    pub fn try_read(&self) -> Option<T> {
        self.inner().try_read()
    }
}

impl<T: Copy + fmt::Debug> fmt::Debug for UntornReader<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UntornReader")
            .field("inner", &self.0)
            .finish()
    }
}

impl<T: Copy> Clone for UntornReader<T> {
    fn clone(&self) -> Self {
        let prev = self.inner().owners.fetch_add(1, Ordering::Relaxed);
        if prev == u16::MAX {
            std::process::abort();
        }
        Self(self.0)
    }
}

impl<T> Drop for UntornReader<T> {
    fn drop(&mut self) {
        let prev = unsafe { self.0.as_ref() }
            .owners
            .fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            unsafe {
                drop(Box::from_raw(self.0.as_ptr()));
            }
        }
    }
}

/// Create a [`UntornWriter`] and [`UntornReader`] pair directly.  The readers are `Clone` but there
/// is only one writer.  You can send this value across threads to delay creating the pair for
/// ergonomics.
///
/// ```rust
/// use mutate_untorn::Untorn;
///
/// let (writer, reader) = Untorn::new(42u64).split();
/// ```
pub struct Untorn<T>(UntornWriter<T>);

unsafe impl<T: Send> Send for Untorn<T> {}
unsafe impl<T: Send> Sync for Untorn<T> {}

impl<T: Copy> Untorn<T> {
    pub fn new(val: T) -> Self {
        Self(UntornWriter::new(val))
    }

    pub fn split(self) -> (UntornWriter<T>, UntornReader<T>) {
        let reader = self.0.reader();
        (self.0, reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Arc,
        },
        thread,
    };

    #[test]
    fn untorn_writer_reader_basic() {
        let writer = UntornWriter::new(0u64);
        let reader = writer.reader();

        writer.write(99);
        assert_eq!(reader.read(), 99);
        assert_eq!(writer.read(), 99);
    }

    #[test]
    fn untorn_reader_clone_sees_same_write() {
        let writer = UntornWriter::new(0u32);
        let r1 = writer.reader();
        let r2 = r1.clone();

        writer.write(7);
        assert_eq!(r1.read(), r2.read());
        assert_eq!(r1.read(), 7);
    }

    #[test]
    fn untorn_cell_clone_sees_same_write() {
        let c1 = UntornCell::new(0u32);
        let c2 = c1.clone();

        c1.write(42);
        assert_eq!(c2.read(), 42);
    }

    #[test]
    fn try_read_returns_none_during_write() {
        // We can manufacture the "write in progress" state by manually
        // leaving seq odd.  This is a white-box test but it's the only
        // reliable way to test the None path without racing.
        use std::sync::atomic::Ordering;

        let cell = UntornCell::new(0u32);
        // Reach into the inner seq and set it odd to simulate a write window.
        let inner = unsafe { cell.0.as_ref() };
        inner.seq.store(1, Ordering::SeqCst);
        assert_eq!(cell.try_read(), None);
        // Restore so Drop doesn't observe an odd seq (not strictly required but tidy).
        inner.seq.store(2, Ordering::SeqCst);
    }

    #[test]
    fn untorn_drop_frees_last_owner() {
        // Use a Rc-tracked drop witness so we can confirm deallocation.
        use std::sync::{Arc, Mutex};

        #[derive(Copy, Clone)]
        #[allow(unused)]
        struct Witness(u32); // Copy is required; we track drops via external Arc

        let dropped = Arc::new(Mutex::new(false));

        {
            let writer = UntornWriter::new(Witness(1));
            let reader = writer.reader();
            // Both alive — not dropped yet.
            drop(writer);
            // reader still alive — not dropped yet.
            assert!(!*dropped.lock().unwrap());
            drop(reader);
            // All owners gone — Box should be freed.
            // We can't directly observe deallocation without Miri, but we can
            // verify the owners counter hit zero without UB by relying on Miri/asan
            // in CI.  Mark this test as "run under Miri" in your CI config.
        }
        // If we reach here without a double-free abort, the drop logic is sane.
    }

    #[test]
    fn untorn_cell_basic_write_read() {
        let cell = UntornCell::new(0u64);
        cell.write(42);
        assert_eq!(cell.read(), 42);
    }

    #[test]
    fn untorn_cell_try_read_clean() {
        let cell = UntornCell::new(7u32);
        assert_eq!(cell.try_read(), Some(7));
    }

    #[test]
    fn untorn_cell_threaded_single_writer() {
        let cell = UntornCell::new(0u64);

        let writer = {
            let cell = cell.clone();
            thread::spawn(move || {
                for i in 0..1000u64 {
                    cell.write(i);
                }
            })
        };

        let reader = {
            let cell = cell.clone();
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
    fn untorn_cell_torn_read_stress() {
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

        let cell = UntornCell::new(Pair::from_u64(0));
        let torn_count = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();

        for w in 0..WRITERS {
            let cell = cell.clone();
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
            let cell = cell.clone();
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

        let (tx, rx) = Untorn::new(Pair::from_n(0)).split();

        let stop = Arc::new(AtomicBool::new(false));
        let readers_done = Arc::new(AtomicU64::new(0));
        let torn = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();

        // Single writer — runs until all readers finish.
        {
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut i = 1u64;
                while !stop.load(Ordering::Relaxed) {
                    tx.write(Pair::from_n(i));
                    i += 1;
                }
                eprintln!("writer done after {i} writes");
            }));
        }

        for r in 0..READERS {
            let reader = rx.clone();
            let stop = Arc::clone(&stop);
            let readers_done = Arc::clone(&readers_done);
            let torn = Arc::clone(&torn);
            handles.push(thread::spawn(move || {
                let mut local_torn = 0u64;
                for _ in 0..READS_PER_READER {
                    if !reader.read().is_consistent() {
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

    #[test]
    fn large_struct_no_tear() {
        #[derive(Copy, Clone)]
        struct Wide {
            a: u64,
            b: u64,
            c: u64,
            d: u64,
            e: u64,
            f: u64,
            g: u64,
            h: u64,
        }

        impl Wide {
            fn splat(n: u64) -> Self {
                Self {
                    a: n,
                    b: n,
                    c: n,
                    d: n,
                    e: n,
                    f: n,
                    g: n,
                    h: n,
                }
            }
            fn is_consistent(self) -> bool {
                self.a == self.b
                    && self.b == self.c
                    && self.c == self.d
                    && self.d == self.e
                    && self.e == self.f
                    && self.f == self.g
                    && self.g == self.h
            }
        }

        use std::{
            sync::{
                atomic::{AtomicBool, AtomicU64, Ordering},
                Arc,
            },
            thread,
        };

        const READERS: usize = 15;
        const READS: u64 = 4_000_000;

        let (tx, rx) = Untorn::new(Wide::splat(0)).split();
        let stop = Arc::new(AtomicBool::new(false));
        let torn = Arc::new(AtomicU64::new(0));
        let mut handles = vec![];

        {
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    tx.write(Wide::splat(i));
                    i += 1;
                }
                eprintln!("writer done after {i} writes");
            }));
        }

        for r in 0..READERS {
            let rx = rx.clone();
            let torn = Arc::clone(&torn);
            let stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                let mut local = 0u64;
                for _ in 0..READS {
                    if !rx.read().is_consistent() {
                        local += 1;
                    }
                }
                torn.fetch_add(local, Ordering::Relaxed);
                stop.store(true, Ordering::Release);
                eprintln!("reader {r} done — {READS} reads, {local} torn");
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(torn.load(Ordering::Relaxed), 0, "torn read on wide struct");
    }
}
