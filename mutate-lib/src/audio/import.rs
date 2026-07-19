//! # Import to Device
//!
//! Publish audio server chunks to a Vulkan device.  Pick an `AudioChoice` and call
//! `AudioContext::import` with a [`Device`](crate::vulkan::device::Device) and `AudioChoice` to
//! obtain a `Consumer` handle.  The consumer serves three roles:
//!
//! - Own upstream audio connection that copies chunks to a persistently mapped device buffer.
//! - Call a user supplied callback with snapshots of the ring state and timing data.
//! - Forward reclaim notifications for retired audio data back to the producer.
//!
//! When chunks arrive from the upstream audio server, they are written to the ring.  A coherent
//! snapshot of the ring state and timing data is then given to the user-supplied [`ImportSink`].
//! The timing data enables the `ImportSink` to smoothly track the incoming stream.
//!
//! ## Reclaim
//!
//! The `ImportSink` **must** return the number of samples that have been retired.  This enables
//! reclaim by the producer.  Without reclaim, the producer will be forced to drop when the ring is
//! full.  A sufficiently large burst of data will always to discontinuity in the visible stream,
//! but aggressive consumption of small chunks to maintain more ring buffer slack can mitigate this
//! likelihood.
//!
//! To track reclaim, it is recommended to read back progress updates pushed from the device.
//! Tracking a timeline semaphore is also valid, but should use a zero timeout to avoid blocking
//! within the `ImportSink`.
//!
//! In the normal-functioning case, audio hits the physical sink in small, well-paced chunks.
//! Hitting the edge cases of the ring buffer means audible consequences for the user.  The fact
//! that audio plays smoothly is evidence that these edge cases are rare.  Correctness signals can
//! mainly be used for smooth recovery and prevention of discontinuity artifacts rather than
//! smoothing out sporadic delivery.
//!
//! The implementation goal is to support multiple consumers with the freshest data and at the
//! lowest latency possible.  The callback structure is deliberately agnostic so that downstreams
//! may be updated via some thread-safe mechanisms that this module need not know about.
//!
//! ## Usage
//!
//! ```ignore
//! // Create an audio context
//! let context = mutate_lib::audio::AudioContext::new()?;
//!
//! // Decide on a choice of audio source.
//! let mut choices = None;
//! context.with_choices_blocking(|choices| {
//!   choice = choice.pop();
//! })?;
//!
//! // A minimal callback that just returns the number of occupied samples, enabling the upstream to
//! // reclaim the entire ring after each callback.
//! let callback = |view| {return view.occupied_len()};
//!
//! // Initialize a stream onto the device (initialization not shown);
//! let stream = context.import_to_device::<2, _>(&device, &choice, 48_000, "mutate", callback)?;
//!
//! // XXX demonstrate a competent ImportSink.
//! ```
//!
//! ### Memory Layout
//!
//! A single device allocation is used for control data and all channels.  Each channel is laid out
//! in a single contiguous sub-allocation but with slight padding for the device flush atom size.
//! Call [`channel_offsets`] to obtain these sub-allocation offsets.
//!
//! ## Ownership
//!
//! The implementation creates a `Consumer` that owns a `AudioConsumer` via thread scope.
//! `AudioConsume` owns the upstream pipewire stream (`AudioConnection`, not the entire
//! `AudioContext`).

// DEBT We've hit a mildly intersection between manual drop of Vulkan resources (which require
// owning a device pointer in order to drop) and the tombstone style cooperative drop of either end
// of audio server connections.  If the Vulkan resources could be dropped into a deletion queue, we
// don't really need to change `Device` ownership physics because ownership of drop-queue capable
// references becomes decoupled from logical device.  Resource management can't come soon enough.
// DEBT Tbh, we don't need the intermediate consumer ring.  If we read into host-mapped memory,
// release the pipewire chunks, and then flush, we are just as fast as when using an intermediate
// ring.  Consistent visibility for consumers is the last concern that may motivate an intermediate
// ring buffer.  We have to update control data.  The initial pipewire read doesn't want to wait on
// that.  Currently we are also de-interleaving to planar for SoA on the device.  Pipewire can do
// the de-interleaving, but this will require work on pipewire side to support multiple rings on the
// process callback side.
// DEBT Sample formats.
// NOTE Channels are doing a fairly naive sub-allocation that could be re-derived on demand.
// Storing the full array of offsets was chosen to duplicate less logic on the device.
// NEXT sub-allocation alignments were not designed for wide loads.  Vectorized reading of the ring
// is a bit more complex for the consumer, more complexity than it's worth on this pass.
// NEXT consumer hazard tracking and slack rotation-reclaim support on producer so that
// discontinuities are swallowed faster and without being presented to the consumer.
// DEBT pipewire mis-indirection.  We may still need one set of hooks outside pipewire because our
// model is to put one callback in pipewire.  After the main connection takes ownership of the
// pipewire chunks, we want to run a callback to write to the GPU ring.  Multiple downstreams can
// dispatch on that as soon as it's written, either notifying a thread owned by the downstream or
// just doing the dispatch synchronously.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use ash::vk;

use crate::audio::{timing::AudioTiming, AudioChoice, AudioConsumer, AudioContext};
use crate::vulkan::prelude::*;
use crate::MutateError;

/// When dispatching a shader, provide the base address as a buffer and read `len` samples.
/// Physical index straddles will be returned as two spans, and dispatching twice is appropriate.
/// Barrier insertion is **not** needed because flushed ranges are guaranteed safe for read until
/// your [`ImportSink`] allows reclaim, which **must** only be called after the dispatch is retired.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSpan {
    pub base: vk::DeviceAddress,
    /// Length in *samples*, not bytes. Callers multiply by the sample stride.
    pub len: u32,
}

/// One channel's occupied region, expressed as device spans.
///
/// `Contiguous` means a single dispatch covers the channel. `Split` means the
/// occupied region wraps the ring end and needs two dispatches (or one dispatch
/// that handles two spans).
#[derive(Clone, Copy, Debug)]
pub enum ChannelRegion {
    Contiguous(DeviceSpan),
    Split { head: DeviceSpan, tail: DeviceSpan },
}

impl ChannelRegion {
    /// Total occupied samples across the region's spans.
    pub fn occupied_len(&self) -> u32 {
        match self {
            ChannelRegion::Contiguous(s) => s.len,
            ChannelRegion::Split { head, tail } => head.len + tail.len,
        }
    }

    /// True when the consumer must dispatch over two spans.
    pub fn is_split(&self) -> bool {
        matches!(self, ChannelRegion::Split { .. })
    }

    /// Spans in logical (oldest-first) order, for uniform iteration.
    pub fn spans(&self) -> impl Iterator<Item = DeviceSpan> {
        let (a, b) = match self {
            ChannelRegion::Contiguous(s) => (Some(*s), None),
            ChannelRegion::Split { head, tail } => (Some(*head), Some(*tail)),
        };
        a.into_iter().chain(b)
    }
}

// NOTE this struct is probably very close to dead.
struct Control {
    // MAYBE wrap up state into the untorn.
    /// Writer has completed writes up to this logical address
    write_head: AtomicU64,
    /// Reader has allowed reclaim up to this logical address
    read_head: AtomicU64,
    /// When closed is set, the thread's read-write loop breaks.
    closed: AtomicBool,
}

/// Immutable ring geometry.  Copy + Send, so both the host handle and the writer thread hold
/// their own.  Owns the wrap math so it exists in exactly one place.
#[derive(Clone, Copy, Debug)]
pub struct RingLayout<const CHANNELS: usize> {
    /// Backing memory buffer device address.
    pub base_address: vk::DeviceAddress,
    /// An array of buffer offsets for each ring's sub-allocation.  Base address + offset =
    /// on-device address.
    pub channel_offsets: [u32; CHANNELS],
    /// Length of each channel
    pub sample_count: u32,
}

impl<const CHANNELS: usize> RingLayout<CHANNELS> {
    /// Snapshot the occupied region for every channel from a single head read.
    ///
    /// All channels share the same logical [read, write) interval, so the wrap
    /// geometry is computed once and only the per-channel base address differs.
    pub fn regions(&self, read: u64, write: u64) -> [ChannelRegion; CHANNELS] {
        let cap = self.sample_count as u64;
        debug_assert!(read <= write, "read head passed write head");
        debug_assert!(write - read <= cap, "occupied exceeds capacity");
        let occupied = write - read;
        let first = read % cap;
        let to_end = cap - first;
        let split = occupied > to_end;

        std::array::from_fn(|c| {
            let base = self.base_address + self.channel_offsets[c] as u64;
            let phys_addr = |s: u64| base + s * 4;
            if split {
                ChannelRegion::Split {
                    head: DeviceSpan {
                        base: phys_addr(first),
                        len: to_end as u32,
                    },
                    tail: DeviceSpan {
                        base,
                        len: (occupied - to_end) as u32,
                    },
                }
            } else {
                ChannelRegion::Contiguous(DeviceSpan {
                    base: phys_addr(first),
                    len: occupied as u32,
                })
            }
        })
    }
}

/// A snapshot of ring state and consumer-owned data that will be valid for use in a dispatch.
/// Callback **must** advance the `read_head` by returning the number of reetired samples or else
/// the producer will just fill up.  Advancing the read head allows the producer to reclaim, meaning
/// the data could be overwritten and flushed.
///
/// Many methods accept a `local_read_head` which should be a value tracked by your downstream, such
/// the farthest advance you have dispatched so far.  The index after the last sample you read
/// should be your new `local_read_head`.  If you are smoothly tracking timing data and want to
/// allow some buffer to prevent underruns on audio server stutter, do not consume up until read,
/// but instead use the `AudioTiming` data.
#[derive(Clone, Debug)]
pub struct DeviceRingView<const CHANNELS: usize> {
    pub ring_layout: RingLayout<CHANNELS>, // add Copy to RingLayout
    pub timing: AudioTiming,
    /// Logical heads at snapshot time.
    pub write_head: u64,
    pub read_head: u64,
}

impl<const CHANNELS: usize> DeviceRingView<CHANNELS> {
    /// Total number of samples in the ring.  If you return a positive intent to reclaim, you must
    /// not use more than the difference between samples and reclaim or else undefined behavior may
    /// occur.
    pub fn occupied_len(&self) -> u32 {
        (self.write_head - self.read_head) as u32
    }

    /// Obtain the ring data starting at `local_read_head`.
    pub fn regions_since(&self, local_read_head: u64) -> [ChannelRegion; CHANNELS] {
        debug_assert!(
            local_read_head >= self.read_head,
            "local head behind reclaim head"
        );
        debug_assert!(
            local_read_head <= self.write_head,
            "local head ahead of write head"
        );
        self.ring_layout.regions(local_read_head, self.write_head)
    }

    /// Like `occupied_len` but only includes samples starting at `local_read_head`.
    pub fn occupied_since(&self, local_read_head: u64) -> u32 {
        debug_assert!(local_read_head <= self.write_head);
        (self.write_head - local_read_head) as u32
    }
}

/// Import data consumer.  Receives [`DeviceRingView`] snapshot when new data has been published to
/// the on-device rings.  Implemented for `Send + 'static` closures.
pub trait ImportSink<const CHANNELS: usize>: Send + 'static {
    /// Return samples the caller is done with; the writer folds that into the read head.
    fn process(&mut self, view: &DeviceRingView<CHANNELS>) -> u32;
}

impl<F, const CHANNELS: usize> ImportSink<CHANNELS> for F
where
    F: FnMut(&DeviceRingView<CHANNELS>) -> u32 + Send + 'static,
{
    fn process(&mut self, view: &DeviceRingView<CHANNELS>) -> u32 {
        self(view)
    }
}

/// The owned side a device import stream.  Data import to the GPU is handled by an owned a reader
/// thread.  This type gathers up ownership and provides an interface to the published control data
/// for host-side and setting up device-side reads.
pub struct Consumer<const CHANNELS: usize> {
    /// Just a persistent bag of bytes being used for ad-hoc sub-allocations. (DEBT).
    buffer: MappedAllocation<u8>,
    /// Reader thread.
    read_thread_handle: Option<JoinHandle<Result<(), MutateError>>>,
    /// Address, offsets of each channel, length in samples.
    ring_layout: RingLayout<CHANNELS>,
    /// Shared control data.
    control: Arc<Control>,
}

impl<const CHANNELS: usize> Consumer<CHANNELS> {
    /// `sample_count` is the length of each channel's ring buffer in samples.  More buffer means
    /// less potential for bursts leading to discontinuities.
    pub(crate) fn new<S: ImportSink<CHANNELS>>(
        device: &Device,
        mut rx: AudioConsumer,
        sample_count: u32,
        mut import_sink: S,
    ) -> Result<Consumer<CHANNELS>, MutateError> {
        let control = Arc::new(Control {
            write_head: AtomicU64::new(0),
            read_head: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });

        let non_coherent_atom_size = device.memory.non_coherent_atom_size;
        let channel_bytes = sample_count as u64 * 4;
        // rounded up for atom flush size
        let channel_stride = channel_bytes.next_multiple_of(non_coherent_atom_size as u64);
        let size = channel_stride as usize * CHANNELS;
        let channel_offsets: [u32; CHANNELS] =
            std::array::from_fn(|i| (channel_stride * i as u64) as u32);
        let mut buffer: MappedAllocation<u8> = MappedAllocation::new(device, size)?;
        let base_address = buffer.device_address(device)?;

        let ring_layout = RingLayout {
            base_address,
            channel_offsets,
            sample_count,
        };

        // f32 0.0 is all-zero bytes, so this write is safe.
        buffer.as_mut_slice().fill(0u8);
        buffer.flush(device)?;

        // Owned write view for the thread.
        // NOTE in a perfect world, we would figure out the mutability or mark unsafe.  Until that
        // world exists, this just works.
        let mut view = buffer.write_view(device);
        let writer_control = control.clone();
        let writer_layout = ring_layout.clone();
        let mut timing = rx.timing()?;

        let read_thread_handle = Some(std::thread::spawn(move || {
            let mut scratch = vec![0u8; 4 * CHANNELS * sample_count as usize];
            let mut write_head: u64 = 0;
            let mut read_head: u64 = 0;

            while !writer_control.closed.load(Ordering::Relaxed) {
                // Wait up to 16ms for a chunk and then warn that chunks are late.
                match rx.wait(std::time::Duration::from_micros(16_000)) {
                    Ok(got) => {
                        let read = rx.read(&mut scratch)?;
                        let frame_bytes = 4 * CHANNELS;
                        // NOTE not checking for partial samples as pipewire seems well-behaved so far.
                        let incoming = read / frame_bytes;
                        let occupied = write_head - read_head;
                        debug_assert!(occupied <= sample_count as u64);
                        let free = sample_count as u64 - occupied;
                        let to_write = incoming.min(free as usize);
                        if to_write < incoming {
                            println!(
                                "audio ring full: dropping {} of {} samples",
                                incoming - to_write,
                                incoming
                            );
                        }
                        let start = write_head;
                        let dst = unsafe { view.as_mut_slice() };

                        // Scatter interleaved input samples across channels
                        // DEBT the interleaved assumption here can break pretty spectacularly.
                        for c in 0..CHANNELS {
                            let ring_base = channel_offsets[c] as usize;
                            for s in 0..to_write {
                                let src = (s * CHANNELS + c) * 4;
                                let logical = (start + s as u64) % sample_count as u64;
                                let dst_byte = ring_base + logical as usize * 4;
                                dst[dst_byte..dst_byte + 4].copy_from_slice(&scratch[src..src + 4]);
                            }
                        }

                        // Per-channel flush.  One run if the written region is contiguous, two if it
                        // wraps the ring end. Ring slots are 4 bytes. ring_base is stride-aligned.
                        // NOTE over-flushing just means more coherence traffic and will not
                        // affect the contents of slots that were not modified.
                        let atom = non_coherent_atom_size as u64;
                        let cap = sample_count as u64;

                        // NOTE flush is no-op on coherent memory.  We are tending to put this in
                        // BAR memory, but our buffer type wrappers are super immature, so we don't
                        // know what kind of memory was actually given to us yet.
                        for c in 0..CHANNELS {
                            let ring_base = channel_offsets[c] as u64;
                            let first = start % cap;
                            let end = (start + to_write as u64) % cap;
                            let wrapped = to_write != 0 && end <= first;
                            let flush_run =
                                |view: &mut MappedWriteView<u8>,
                                 lo_sample: u64,
                                 hi_sample: u64|
                                 -> Result<(), MutateError> {
                                    // Byte range [lo, hi) within the channel, widened to atom alignment.
                                    let lo = ring_base + lo_sample * 4;
                                    let hi = ring_base + hi_sample * 4;
                                    // NOTE atom is PoT.  Equivalent expression is (lo / atom) * atom
                                    let lo_aligned = lo & !(atom - 1);
                                    let hi_aligned = hi
                                        .next_multiple_of(atom)
                                        .min(ring_base + channel_stride as u64);
                                    Ok(view.flush_range(lo_aligned, hi_aligned - lo_aligned)?)
                                };

                            if to_write == 0 {
                                // nothing to flush
                            } else if wrapped {
                                flush_run(&mut view, first, cap)?; // head: [first, cap)
                                flush_run(&mut view, 0, end)?; // tail: [0, end)
                            } else {
                                flush_run(&mut view, first, end)?; // contiguous: [first, end)
                            }
                        }

                        // Publish new write head
                        write_head += to_write as u64;
                        writer_control
                            .write_head
                            .store(write_head, Ordering::Release);

                        // Pick up new timing data, which is written by now.
                        match rx.timing() {
                            Ok(new_time) => {
                                timing = new_time;
                            }
                            Err(e) => {
                                // Realistically poisoning was the only error type.  Suddenly closed
                                // upstream still has a phase, just one that will never arrive again.
                                return Err(e);
                            }
                        };
                        // Ring update is flushed & coherent.  Host view of state is updated.  Call
                        // the callback with the updated view, including the latest timing data and
                        // a pointer to the read_head so that it can update it.
                        let device_ring_view = DeviceRingView {
                            ring_layout,
                            timing,
                            read_head,
                            write_head,
                        };
                        let retired = import_sink.process(&device_ring_view) as u64;
                        // A sink that retires more than it was shown is a contract violation.
                        // Clamping keeps the invariant rather than trusting the return blindly.
                        debug_assert!(retired <= write_head - read_head, "sink over-retired");
                        read_head = (read_head + retired).min(write_head);
                        writer_control.read_head.store(read_head, Ordering::Release);
                    }
                    Err(MutateError::Timeout(_)) => {
                        // DEBT logging and runtime toggles for scopes of messages 💀
                        println!("audio server chunk was late");
                    }
                    Err(e) => {
                        println!("error: audio consumer {:?}", e);
                        writer_control.closed.store(true, Ordering::Relaxed);
                        return Err(e);
                    }
                };
            }
            Ok(())
        }));
        Ok(Consumer {
            buffer,
            read_thread_handle,
            ring_layout,
            control,
        })
    }

    /// The size in elements that the physical rings can store when full.  This is also the repeat
    /// modulus for physical indexes.
    pub fn capacity(&self) -> u32 {
        self.ring_layout.sample_count
    }

    /// Number of samples currently occupied (written but not reclaimed).
    /// This is the same for every channel — they advance in lockstep.
    pub fn occupied_len(&self) -> Result<u32, MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        // XXX this could be a torn read.  Use untorn crate.
        let write = self.control.write_head.load(Ordering::Acquire);
        let read = self.control.read_head.load(Ordering::Relaxed);
        Ok((write - read) as u32)
    }

    /// Physical base address of each channel's ring sub-allocation.  **Bare use of these addresses
    /// is undefined behavior** that will read torn data.  Use for fun or ring diagnostics.  All bit
    /// patterns are valid float, but the data found may be nonsensical and pollute heuristics.
    /// That said, observing the data tearing in real time is **pretty rad** 😎!
    pub unsafe fn channels(&self) -> Result<[vk::DeviceAddress; CHANNELS], MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        Ok(std::array::from_fn(|c| {
            self.ring_layout.base_address + self.ring_layout.channel_offsets[c] as u64
        }))
    }

    /// Will set a flag for upstream and returns when that thread joins.  Therefore, blocking.
    pub fn destroy(&mut self, device: &Device) -> Result<(), MutateError> {
        // tombstone, join the reader thread, and destroy the allocation.
        self.control.closed.store(true, Ordering::Relaxed);
        let join_result = if let Some(handle) = self.read_thread_handle.take() {
            handle.join()
        } else {
            Ok(Ok(()))
        };
        self.buffer.destroy(device)?;
        join_result.map_err(|_| MutateError::AudioTerminate)?
    }
}
