// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # The Graph.
//!
//! ⚠️ Most of this module documentation is not yet implemented.  Take inspiration from the problem
//! descriptions.
//!
//! Rendering audio to various frontends has many aspects of dependency:
//!
//! - Scheduling node evaluation with render phase timings.
//!
//! - Transmitting GPU synchronization information to dependent nodes.
//!
//! - Transferring resources to or from GPU memory.
//!
//! - Reactive updates from upstream node configuration changes.
//!
//! - Memory transitions and barrier insertions.
//!
//! - Feedback rendering.
//!
//! Many of these operations can be inferred from the natural topology of how nodes depend on each
//! other, a graph structure.  Operations such as feedback rendering should clearly imply that the
//! graph contains cycles.  By unrolling this dependency across frames, we obtain a DAG.
//!
//! A proper DAG can be used to solve many dependency problems.  The graph solves dependency
//! problems and hands off necessary information to nodes.    Different aspects of dependency
//! must still be handled by different interfaces on the nodes themselves.
//!
//! ## Driving Node Evaluation
//!
//! Nodes must rely on the graph to handle timing data.  They may be evaluated in a dedicated thread
//! or a worker, depending on their scheduling needs.  There are multiple timelines to drive,
//! including slower machine learning training, fast audio response, and GPU frames.
//!
//! Work such as deletion queue processing and transferring onto the GPU are represented with
//! mementos that are inserted into appropriate queues and evaluated by other nodes.  The Graph
//! facilitates this communication.

// Designing phases is not going to happen in one day.
//
// To begin with, MuTate aims for relatively low CPU and GPU stress.  Without some slack compute and
// memory, we cannot do online training.
//
// Compared to something like games, since we don't have player inputs in the middle of a frame,
// there is no potential to get a faster reaction by over-rendering.  The output is deterministic.
// We can use this to get away with a double buffer swapchain and no more.
//
// In order to present frames that include the latest possible audio, we need to delay draws that
// depend on that audio as near to the presentation as possible.  The delay phase is timed to
// provide us enough slack to wake up early enough to avoid being late due to timer jitter and to
// get our submissions done in time to be ready for presentation.
//
// Timing this with the display depends on whether the display has variable for fixed timing.
// Variable rate rendering (VRR) give us a lot more flexibility to display at the exact speed of
// audio presentation.  Fixed rate rendering (FRR) forces us to take a chance that we miss
// presentation deadline and wind up dropping a frame.  This risk is manageable by basic statistics.
// It is better to be smooth with slightly old audio than to be jittery, displaying even older
// audio.
//
// A potential use for triple buffering may exist, but to correct late audio in feedback rendering
// rather than to always run the GPU as fast as possible.

/// It is but an idea.  Read the module documents and comments.
pub struct Graph {}

/// Includes a VulkanContext
pub struct GraphContext {}

/// Nature of an individual `GraphEvent`.
#[derive(PartialEq, Eq)]
pub enum EventIntent {
    /// Entirely old data.  Upstream is underfed, but downstreams can interpret the duplicate as
    /// valid.
    Duplicate,
    /// Downstreams should reset.  Data is type-correct default.
    Invalid,
    /// Entirely fresh data.
    Full,
    /// Some new data and some old or derived data.  These can be emitted because the upstream is
    /// changing the alignment or because it is pacing itself.  If the semantics are important, the
    /// downstream should track new data since their last `produce` call and adjust their output
    /// accordingly.
    Partial,
    /// A partial read only intended to catch up to an upstream that excess buffer.
    Seek,
}

pub struct GraphEvent<'a, T> {
    /// The buffer, borrowed from a `Node`.
    pub buffer: &'a GraphBuffer<T>,
    /// How downstream should interpret this event.
    pub intent: EventIntent,
}

/// A generic type for graph inputs and outputs.  Very raw 🤠.
pub struct GraphBuffer<T> {
    /// Boxed array for lightweight copying.
    // NEXT Just used a flat buffer instead of a ring.  Ring slices or chunks style API will be a
    // teeny bit faster, but it's more complex for now.
    // NEXT efficient allocation, such as batching and recycling.
    pub data: Box<[T]>,
    /// index of first new datum.  If offset is equal to length, no new data was received.
    pub offset: usize,
}

impl<T: Copy + Default> GraphBuffer<T> {
    /// Return a default buffer of `size`.
    ///
    /// Note: There is no way to indicate actual old versus default data.  This is a deliberate
    /// choice.  If your downstream node cares, it should track the amount of new data it has seen.
    /// The graph should "just work" with all default data at all times.
    pub fn new(size: usize) -> Self {
        GraphBuffer {
            data: vec![T::default(); size].into_boxed_slice(),
            offset: 0,
        }
    }

    /// New data from the graph.
    pub fn fresh(&self) -> &[T] {
        &self.data[self.offset..] // stored in tail of buffer
    }

    /// Data that is being repeated to keep consistent data in the backing buffer.
    pub fn recycled(&self) -> &[T] {
        &self.data[0..self.offset] // stored in head of buffer
    }

    /// Write new data.  Makes any existing data into old data.   Panics if new data is too large.
    /// Nodes should agree on their buffer sizes early and remain coordinated through graph updates.
    pub fn write(&mut self, fresh: &[T]) {
        let fresh_len = fresh.len().min(self.data.len());
        let old = self.data.len() - fresh_len;
        if fresh_len > 0 {
            self.data.copy_within(fresh_len..self.data.len(), 0);
            self.data[old..].copy_from_slice(&fresh[0..fresh_len]);
        }
        self.offset = old;
    }

    /// For easier streaming, give us a slice so we can write element by element.  If `size` is
    /// larger than the internal buffer, the lesser size will be used.
    // XXX does not enforce that receiver actually writes size new elements.
    pub fn writeable_slice(&mut self, size: usize) -> &mut [T] {
        let size = size.min(self.data.len());
        let old = self.data.len() - size;
        if size > 0 {
            self.data.copy_within(size..self.data.len(), 0);
        }
        self.offset = size;
        return &mut self.data[old..];
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_graph_buffer() {
        let mut gb = GraphBuffer::<u8>::new(4);

        // partial write to fresh buffer
        gb.write(&[1u8; 2]);
        // new data in the head
        assert_eq!(gb.fresh(), &[1u8; 2]);
        // default data in the tail
        assert_eq!(gb.recycled(), &[0u8; 2]);
        // offset is first index of old data
        assert_eq!(gb.offset, 2);

        // fill up remaining
        gb.write(&[2u8; 2]);
        // offset points to beginning of old data
        assert_eq!(gb.offset, 2);
        // old head is now the tail
        assert_eq!(gb.recycled(), &[1u8, 1]);
        // head is new data
        assert_eq!(gb.fresh(), &[2u8, 2]);

        // full overwrite
        gb.write(&[3u8; 4]);
        assert_eq!(gb.offset, 0);
        // empty tail
        assert_eq!(gb.recycled(), &[]);
        // full head
        assert_eq!(gb.fresh(), &[3u8; 4]);

        // overwrite an empty byte
        gb.write(&[]);
        // empty head
        assert_eq!(gb.offset, 4);
        assert_eq!(gb.fresh(), &[]);
        assert_eq!(gb.recycled(), &[3u8; 4]);

        // write five bytes, larger than the gb can accept
        gb.write(&[0u8, 1u8, 2u8, 3u8, 4u8]);
        assert_eq!(gb.fresh(), &[0u8, 1u8, 2u8, 3u8]);

        // write to some slices
        let out = gb.writeable_slice(2);
        out[0] = 8;
        out[1] = 10;
        assert_eq!(gb.fresh(), &[8, 10]);
        assert_eq!(gb.offset, 2);
        assert_eq!(gb.recycled(), &[2, 3]);
    }
}
