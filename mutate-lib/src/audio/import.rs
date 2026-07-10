//! # Import to Device
//!
//! Publish audio server chunks to a Vulkan device.  Pick an `AudioChoice` and call
//! `AudioContext::import` with a [`Device`](crate::vulkan::device::Device) and `AudioChoice` to
//! obtain a `Consumer` handle.  The consumer serves three roles:
//!
//! - own audio consumption that copies upstream chunks to a persistently mapped device buffer.
//! - provide access to control data used to direct dispatches
//! - receive reclaim notifications for retired audio data
//!
//! Control data is published in host memory as atomic read and write heads.  If you do not tell the
//! consumer to reclaim data, the producer will be forced to drop when the ring is full. A
//! sufficiently large burst of data will always to discontinuity in the visible stream, but
//! aggressive consumption and larger buffer sizes can mitigate this likelihood.
//!
//! In the normal-functioning case, audio hits the physical sink in small, well-paced chunks.
//! Hitting the edge cases of the ring buffer means audible consequences for the user.  The fact
//! that audio plays smoothly is evidence that these edge cases are rare.  Correctness signals can
//! mainly be used for smooth recovery and prevention of discontinuity artifacts rather than
//! smoothing out sporadic delivery.
//!
//! The implementation goal is to support multiple consumers with the freshest data and at the
//! lowest latency possible.  Letting the consumer block the producer on-device would be very
//! tricky, and so the responsibility is placed on the consumer to catch a consistent segment of the
//! arrow in flight.  The producer makes control data visible to both host and device.  On-device
//! consumer logic can check this data to determine if the arrow in-flight was indeed caught or was
//! torn during the reads.  Deciding how to manage a torn processing result is beyond the scope of
//! this module.
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
//! // Initialize a stream onto the device (initialization not shown);
//! let stream = context.import_to_device(&device, &choice, "mutate")?;
//!
//! // Obtain occupied regions.  Each region is either `Contiguous` or `Split` when region straddles
//! // the ring's physical indexes.
//! let regions = consumer.regions()?;
//! let mut retired = 0u32;
//! for region in regions {
//!     for span in region.spans() {
//!         // Bind span.base as a buffer device address and read span.len samples.
//!         // No barrier needed: flushed ranges stay valid until advance_read.
//!         dispatch_over(span.base, span.len);
//!     }
//!     // All channels advance in lockstep, so one region's length is the
//!     // reclaim count for this snapshot.
//!     retired = region.occupied_len();
//! }
//!
//! // Advance read to allow reclaim by producer.  MUST be called after dispatches are retired or
//! // in-flight dispatches may observe torn data. Call as aggressively as possible to avoid stalling
//! // upstream production.
//! consumer.advance_read(retired)?;
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

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use ash::vk;

use crate::audio::{AudioChoice, AudioConsumer, AudioContext};
use crate::vulkan::prelude::*;
use crate::MutateError;

/// When dispatching a shader, provide the base address as a buffer and read `len` samples.
/// Physical index straddles will be returned as two spans, and dispatching twice is appropriate.
/// Barrier insertion is **not** needed because flushed ranges are guaranteed safe for read until
/// the consumer allows reclaim by calling [`advance_read`], which **must** only be called after the
/// dispatch is retired.
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

/// The owned side a device import stream.  Data import to the GPU is handled by an owned a reader
/// thread.  This type gathers up ownership and provides an interface to the published control data
/// for host-side and setting up device-side reads.
pub struct Consumer<const CHANNELS: usize> {
    /// Just a persistent bag of bytes being used for ad-hoc sub-allocations. (DEBT).
    buffer: MappedAllocation<u8>,
    /// Reader thread.
    read_thread_handle: Option<JoinHandle<Result<(), MutateError>>>,
    /// Backing memory buffer device address.
    pub base_address: vk::DeviceAddress,
    /// Length of each channel
    sample_count: u32,
    /// An array of buffer offsets for each ring's sub-allocation.  Base address + offset =
    /// on-device address.
    channel_offsets: [u32; CHANNELS],
    /// Shared control data.
    control: Arc<Control>,
}

struct Control {
    /// Writer has completed writes up to this logical address
    write_head: AtomicU64,
    /// Reader has allowed reclaim up to this logical address
    read_head: AtomicU64,
    /// When closed is set, the thread's read-write loop breaks.
    closed: AtomicBool,
}

impl<const CHANNELS: usize> Consumer<CHANNELS> {
    /// `sample_count` is the length of each channel in samples.
    pub(crate) fn new(
        context: &AudioContext,
        device: &Device,
        choice: &AudioChoice,
        sample_count: u32,
        name: &str,
    ) -> Result<Consumer<CHANNELS>, MutateError> {
        let control = Arc::new(Control {
            write_head: AtomicU64::new(0),
            read_head: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });

        let non_coherent_atom_size = device.non_coherent_atom_size();
        let channel_bytes = sample_count as u64 * 4;
        // rounded up for atom flush size
        let channel_stride = channel_bytes.next_multiple_of(non_coherent_atom_size as u64);
        let size = channel_stride as usize * CHANNELS;
        let channel_offsets: [u32; CHANNELS] =
            std::array::from_fn(|i| (channel_stride * i as u64) as u32);
        // FIXME reverse device-size argument order in buffer module
        let mut buffer: MappedAllocation<u8> = MappedAllocation::new(size, device)?;
        let base_address = buffer.device_address(device)?;

        // f32 0.0 is all-zero bytes, so this write is safe.
        buffer.as_mut_slice().fill(0u8);
        buffer.flush(device)?;

        let mut view = buffer.write_view(device);
        let mut rx = context.connect(choice, name)?;
        let writer_control = control.clone();

        let read_thread_handle = Some(std::thread::spawn(move || {
            let mut scratch = vec![0u8; 4 * CHANNELS * sample_count as usize];
            let mut write_head: u64 = 0;

            while !writer_control.closed.load(Ordering::Relaxed) {
                // Wait up to 16ms for a chunk and then warn that chunks are late.
                match rx.wait(std::time::Duration::from_micros(16_000)) {
                    Ok(got) => {
                        let got = rx.read(&mut scratch)?;
                        let frame_bytes = 4 * CHANNELS;
                        // NOTE ignoring incomplete samples as pipewire seems well-behaved so far.
                        let incoming = got / frame_bytes;
                        let read_head = writer_control.read_head.load(Ordering::Acquire);
                        let occupied = write_head.wrapping_sub(read_head);
                        let free = (sample_count as u64).saturating_sub(occupied);
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
                                let logical = start.wrapping_add(s as u64) % sample_count as u64;
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

                        for c in 0..CHANNELS {
                            let ring_base = channel_offsets[c] as u64;
                            let first = start % cap;
                            let end = start.wrapping_add(to_write as u64) % cap;
                            let wrapped = to_write as u64 != 0 && end <= first;

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
                        let new_head = write_head.wrapping_add(to_write as u64);
                        writer_control.write_head.store(new_head, Ordering::Release);
                        write_head = new_head;
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
            base_address,
            sample_count,
            channel_offsets: channel_offsets,
            control,
        })
    }

    /// The size in elements that the physical rings can store when full.  This is also the repeat
    /// modulus for physical indexes.
    pub fn capacity(&self) -> u32 {
        self.sample_count
    }

    /// Number of samples currently occupied (written but not reclaimed).
    /// This is the same for every channel — they advance in lockstep.
    pub fn occupied_len(&self) -> Result<u32, MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        let write = self.control.write_head.load(Ordering::Acquire);
        let read = self.control.read_head.load(Ordering::Relaxed);
        Ok(write.wrapping_sub(read) as u32)
    }

    /// Physical base address of each channel's ring sub-allocation.  **Bare use of these addresses
    /// is undefined behavior** that will read uninitialized or torn data.  Use for fun or ring
    /// diagnostics.  All bit patterns are valid float, but the data found may be nonsensical and
    /// pollute heuristics.  That said, observing the data tearing in real time is **pretty rad**
    /// 😎!
    pub unsafe fn channels(&self) -> Result<[vk::DeviceAddress; CHANNELS], MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        Ok(std::array::from_fn(|c| {
            self.base_address + self.channel_offsets[c] as u64
        }))
    }

    /// Snapshot the occupied region for every channel from a single head read.
    ///
    /// All channels share the same logical [read, write) interval, so the wrap
    /// geometry is computed once and only the per-channel base address differs.
    pub fn regions(&self) -> Result<[ChannelRegion; CHANNELS], MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        let write = self.control.write_head.load(Ordering::Acquire);
        let read = self.control.read_head.load(Ordering::Relaxed);

        let occupied = write.wrapping_sub(read);
        let cap = self.sample_count as u64;
        let first = read % cap; // physical start index
                                // Does [first, first + occupied) cross the ring end?
        let to_end = cap - first;
        let split = occupied > to_end;

        Ok(std::array::from_fn(|c| {
            let base = self.base_address + self.channel_offsets[c] as u64;
            let phys_addr = |sample_idx: u64| base + sample_idx * 4;
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
        }))
    }

    /// Inform the writer that `consumed` samples have been retired and their ring slots may be
    /// reclaimed.  Call as aggressively as possible so upstream production is not blocked.
    ///
    /// `consumed` is a count of samples, matching what the caller pulled from [`regions`].
    ///
    /// This **must** only be called after the samples have been retired from use.  Allowing reclaim
    /// early may result in device dispatches observing overwritten samples mid-dispatch.
    pub fn advance_read(&self, consumed: u32) -> Result<(), MutateError> {
        if self.control.closed.load(Ordering::Relaxed) {
            return Err(MutateError::Dropped);
        }
        let read = self.control.read_head.load(Ordering::Acquire);
        let write = self.control.write_head.load(Ordering::Acquire);
        let new_head = read.wrapping_add(consumed as u64);
        debug_assert!(
            new_head <= write,
            "advance_read({consumed}) overruns write head: read={read} write={write} new={new_head} \
             (occupied={})",
            write.wrapping_sub(read),
        );
        self.control.read_head.store(new_head, Ordering::Release);
        Ok(())
    }

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
