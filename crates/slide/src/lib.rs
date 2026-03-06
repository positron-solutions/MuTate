// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Sliding Window
//!
//! A sliding window is just a fast circular view over a logically infinite stream of input.
//!
//! Sliding window semantics are slightly different than ring buffers:
//!
//! - The window is **always full** (items are `Default`).
//! - Window readers always read large segments of the window, usually the entire window.
//! - The window never rejects writes and can only represents a window of the input stream.
//!
//! To enforce no-skips or to synchronize, use an intermediate writer than can back-pressure any
//! chatty upstream, such as one that might burst on connection.  **The window is not itself a
//! back-pressure or synchronization device!**
//!
//! ## Backing store initialization
//!
//! The window is always considered full: every slot is a live element.  When
//! constructing via [`SlidingWindow::new`] or [`SlidingWindow::new_heap`] this
//! is guaranteed automatically — slots are zeroed to [`Default`].  When
//! supplying your own buffer via [`SlidingWindow::from_storage`], **you are
//! responsible for ensuring the buffer is fully initialized**, or you must call
//! [`SlidingWindow::clear`] before the window is read.

// Even sliding_window crate uses semantics like "is_full".  This crate exists to allow flexible
// development while we nail down the semantics of upstream and downstream particular to our use
// cases.

use core::fmt::Debug;

/// Backing storage for a [`SlidingWindow`].
///
/// Implement this trait for any contiguous, fixed-length buffer you own.
pub trait Storage {
    type Item;

    fn as_slice(&self) -> &[Self::Item];
    fn as_mut_slice(&mut self) -> &mut [Self::Item];
}

impl<T, const N: usize> Storage for [T; N] {
    type Item = T;

    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

#[cfg(feature = "alloc")]
impl<T> Storage for Vec<T> {
    type Item = T;

    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// A circular sliding window backed by any [`Storage`] implementation.
///
/// The window is always considered full: slots that have never been written hold
/// `S::Item::default()`.  The logical order of elements is oldest-first; the newest
/// element is always at the *back*.
///
/// # Storage options
///
/// ## Stack-allocated array (no `alloc`)
///
/// The size is part of the type and requires no heap allocation, making this suitable
/// for `no_std`/`no_alloc` environments:
///
/// ```
/// use mutate_slide::SlidingWindow;
///
/// let mut w = SlidingWindow::<[u8; 32]>::new();
/// w.push(42);
/// ```
///
/// ## Caller-owned buffer
///
/// Hand an existing buffer to the window; the library performs zero allocation:
///
/// ```
/// use mutate_slide::SlidingWindow;
///
/// let backing = [0u8; 32];
/// let mut w = SlidingWindow::from_storage(backing);
/// w.push(1);
/// ```
///
/// ## Heap-allocated `Vec` (requires `alloc` feature)
///
/// ```
/// # #[cfg(feature = "alloc")]
/// # {
/// use mutate_slide::SlidingWindow;
///
/// let mut w = SlidingWindow::<Vec<u8>>::new_heap(1024);
/// w.push(99);
/// # }
/// ```
pub struct SlidingWindow<S: Storage> {
    buf: S,
    /// Index of the *next* write position (oldest element when the window is full).
    write: usize,
}

impl<S> SlidingWindow<S>
where
    S: Storage,
    S::Item: Copy + Default,
{
    /// Wrap an already-initialized backing store.
    ///
    /// # Storage initialization
    ///
    /// The window assumes every slot in `buf` contains a valid, meaningful element.
    /// Slots that have never been written will be returned by [`iter`](Self::iter) and
    /// [`as_slices`](Self::as_slices) as legitimate "old" data.  If `buf` is not fully
    /// initialized, call [`clear`](Self::clear) immediately after construction to reset
    /// every slot to [`Default`].
    ///
    /// # Panics
    ///
    /// Panics if `buf` is empty.
    pub fn from_storage(buf: S) -> Self {
        assert!(!buf.as_slice().is_empty(), "storage must be non-empty");
        Self { buf, write: 0 }
    }

    /// Length of backing store.  Since the backing store is **always full of elements**, there is
    /// no notion of a partially filled backing store.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.buf.as_slice().len()
    }

    /// Push a single element.
    pub fn push(&mut self, value: S::Item) {
        self.buf.as_mut_slice()[self.write] = value;
        self.write = (self.write + 1) % self.len();
    }

    /// Push a slice of elements.
    ///
    /// More optimal than `push`.  If `slice` is longer than `N`, only the last `N` elements are
    /// retained.  Index is updated once.
    pub fn push_slice(&mut self, slice: &[S::Item]) {
        let n = self.len();
        let slice = if slice.len() > n {
            &slice[slice.len() - n..]
        } else {
            slice
        };
        let len = slice.len();
        let head = n - self.write;

        if len <= head {
            self.buf.as_mut_slice()[self.write..self.write + len].copy_from_slice(slice);
        } else {
            self.buf.as_mut_slice()[self.write..].copy_from_slice(&slice[..head]);
            self.buf.as_mut_slice()[..len - head].copy_from_slice(&slice[head..]);
        }

        self.write = (self.write + len) % n;
    }

    /// Returns the window contents as two contiguous slices in logical (oldest-first) order.
    ///
    /// Concatenating `first` and `second` gives the full window from oldest to newest.
    /// Either slice may be empty (e.g. `second` is empty when `write == 0`).
    pub fn as_slices(&self) -> (&[S::Item], &[S::Item]) {
        let (left, right) = self.buf.as_slice().split_at(self.write);
        // `right` starts at `write` (the oldest element); `left` ends at `write` (the newest).
        (right, left)
    }

    /// Iterate over elements from oldest to newest
    pub fn iter(&self) -> impl Iterator<Item = &S::Item> {
        let (a, b) = self.as_slices();
        a.iter().chain(b.iter())
    }

    /// Reset all elements to default and zero the index.
    pub fn clear(&mut self) {
        self.buf.as_mut_slice().fill(S::Item::default());
        self.write = 0;
    }
}

#[cfg(feature = "alloc")]
impl<T: Copy + Default> SlidingWindow<Vec<T>> {
    /// Allocate a heap-backed window of the given `size`.
    pub fn new_heap(size: usize) -> Self {
        Self::from_storage(vec![T::default(); size])
    }
}

impl<T: Copy + Default, const N: usize> SlidingWindow<[T; N]> {
    /// Create a zero-initialized stack-backed window.
    ///
    /// The capacity is fixed at compile time by the array length `N`.
    pub fn new() -> Self {
        Self::from_storage([T::default(); N])
    }

    /// Borrow the underlying physical array.
    ///
    /// Note that the physical layout is *not* the logical order; use [`iter`](Self::iter)
    /// or [`as_slices`](Self::as_slices) when you need oldest-first ordering.
    pub fn as_array(&self) -> &[T; N] {
        &self.buf
    }
}

impl<S> Debug for SlidingWindow<S>
where
    S: Storage,
    S::Item: Copy + Default + Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<S> Copy for SlidingWindow<S>
where
    S: Storage + Copy,
    S::Item: Copy,
{
}

impl<S> Clone for SlidingWindow<S>
where
    S: Storage + Copy,
    S::Item: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Default for SlidingWindow<S>
where
    S: Storage + Default,
    S::Item: Default,
{
    fn default() -> Self {
        Self {
            buf: S::default(),
            write: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_push_single_elements() {
        let mut w = SlidingWindow::<[u8; 3]>::new();
        w.push(1);
        w.push(2);
        w.push(3);
        assert_eq!(w.iter().copied().collect::<Vec<_>>(), [1, 2, 3]);
    }

    #[test]
    fn sliding_window_push_overwrites_oldest() {
        let mut w = SlidingWindow::<[u8; 3]>::new();
        w.push(1);
        w.push(2);
        w.push(3);
        w.push(4); // 1 is evicted
        assert_eq!(w.iter().copied().collect::<Vec<_>>(), [2, 3, 4]);
    }

    #[test]
    fn sliding_window_as_slices_no_wrap() {
        let mut w = SlidingWindow::<[u8; 4]>::new();
        w.push_slice(&[1, 2]);
        let (a, b) = w.as_slices();
        let combined: Vec<u8> = a.iter().chain(b.iter()).copied().collect();
        assert_eq!(combined, [0, 0, 1, 2]);
    }

    #[test]
    fn sliding_window_as_slices_after_wrap() {
        let mut w = SlidingWindow::<[u8; 4]>::new();
        w.push_slice(&[1, 2, 3, 4, 5]); // write wraps to 1
        let (a, b) = w.as_slices();
        let combined: Vec<u8> = a.iter().chain(b.iter()).copied().collect();
        assert_eq!(combined, w.iter().copied().collect::<Vec<_>>());
    }

    #[test]
    fn sliding_window_clear_resets_data_and_index() {
        let mut w = SlidingWindow::<[u8; 4]>::new();
        w.push_slice(&[1, 2, 3, 4]);
        w.clear();
        assert_eq!(w.as_array(), &[0, 0, 0, 0]);
        // After clear the write head is back at 0, so iter yields all zeros oldest-first.
        assert_eq!(w.iter().copied().collect::<Vec<_>>(), [0, 0, 0, 0]);
    }

    #[test]
    fn sliding_window_len_reflects_storage_size() {
        let w = SlidingWindow::<[u8; 8]>::new();
        assert_eq!(w.len(), 8);
    }

    #[test]
    fn sliding_window_default_equals_new() {
        let a = SlidingWindow::<[u8; 4]>::new();
        let b = SlidingWindow::<[u8; 4]>::default();
        assert_eq!(a.as_array(), b.as_array());
    }

    #[test]
    fn sliding_window_output_is_oldest_first() {
        let mut w = SlidingWindow::<[u8; 3]>::new();
        w.push(7);
        w.push(8);
        w.push(9);
        assert_eq!(format!("{:?}", w), "[7, 8, 9]");
    }

    #[test]
    fn sliding_window_iter() {
        let mut w = SlidingWindow::<[u8; 3]>::new();
        w.push(7);
        w.push(8);
        w.push(9);
        let i: Vec<u8> = w.iter().copied().collect();
        assert_eq!(format!("{:?}", i), "[7, 8, 9]");
    }
}
