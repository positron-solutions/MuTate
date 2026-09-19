//! # Import to Device
//!
//! Publish audio server chunks to a Vulkan device.  Pick an `AudioChoice` and call
//! `AudioContext::import` with a [`Device`](crate::vulkan::device::Device) and `AudioChoice` to
//! obtain a `Consumer` handle.  The consumer serves three roles:
//!
//! - Own upstream audio connection that copies chunks to a persistently mapped device buffer.
//! - Call a user supplied callback with snapshots of the ring state and timing data.
//! - Forward retirement notifications for audio data back to the producer.
//!
//! When chunks arrive from the upstream audio server, they are written to the ring.  A coherent
//! snapshot of the ring state and timing data is then given to the user-supplied [`ImportSink`].
//! The timing data enables the `ImportSink` and other downstreams to smoothly track the incoming
//! raw audio stream.
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
//! let callback = |view| {return view.read_count() };
//!
//! // Initialize a stream onto the device (initialization not shown);
//! let stream = context.import_to_device::<2, _>(&device, &choice, 6_000, "µTate", callback)?;
//!
//! ```
//!
//! ### Memory Layout
//!
//! A single allocation is used for all channels and data is stored planar, not interleaved.
//! Channel zero is left by convention.
//!
//! ## Ownership
//!
//! The implementation creates an `AudioImport` that owns a `AudioConsumer` via thread scope.
//! `AudioConsumer` owns the upstream pipewire stream (`AudioConnection`, not the entire
//! `AudioContext`).

// DEBT Sample formats.
// NEXT sub-allocation alignments were not designed for wide loads.  Just support vectorized reading
// in case a consumer wants that.
// NEXT consumer hazard tracking and slack rotation-reclaim support on producer so that
// discontinuities are swallowed faster and without restart discontinuities being presented to the
// consumer.
// DEBT pipewire mis-indirection.  We don't need an intermediate ring in the AudioConsumer. A simple
// copy into mapped memory, with or without intermediate buffer to allow pipewire to reclaim as soon
// as possible, would do just fine.  Pipewire probably can de-interleave faster already, but we have
// to request it that way.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use ash::vk;

use crate::audio::{timing::AudioTiming, AudioChoice, AudioConsumer, AudioContext};
use crate::vulkan::prelude::*;
use crate::MutateError;

pub(crate) mod core {
    pub use super::{AudioImport, DeviceAudioView};
}

// NOTE this struct is probably very close to dead.
struct Control {
    // MAYBE wrap up state into the Untorn.
    /// Writer has completed writes up to this logical address
    write_head: AtomicU64,
    /// Reader has allowed reclaim up to this logical address
    read_head: AtomicU64,
    /// When closed is set, the thread's read-write loop breaks.
    closed: AtomicBool,
}

/// Downstream readers dispatch against the data using this output type.
#[derive(Clone, Copy, Debug)]
pub struct DeviceAudioView {
    /// device address.
    pub base_address: DeviceAddress,
    /// Length of each channel in samples.
    pub sample_count: u32,
    /// number of channels
    pub channel_count: u32,
    /// Output rate of data arriving on this ring.  Samples per second.
    pub sample_rate: u32,
    /// Logical write head at snapshot time.
    pub write_head: u64,
    /// Consumer owned logical read head.
    pub read_head: u64,
    /// Upstream filtered timing signal to enable smooth tracking.  Only available after the first
    /// chunk is published by the audio server.
    pub timing: Option<AudioTiming>,
}

impl DeviceAudioView {
    /// Convert logical read head to a pre-wrapped physical index.  Shaders usually start reading
    /// here.
    pub fn read_head_physical(&self) -> u32 {
        (self.read_head % self.sample_count as u64) as u32
    }

    /// Convert occupied length to a count for use with a physical start index and the wrap mask.
    pub fn read_count(&self) -> u32 {
        (self.write_head - self.read_head) as u32
    }

    /// Wrap modulus for physical indexing.
    pub fn mask(&self) -> u32 {
        self.sample_count - 1
    }

    /// Compute a device address of a specific channel.  Not expecting much use, but a single
    /// channel reader may prefer it.
    pub fn channel(&self, channel_index: usize) -> DeviceAddress {
        (self.base_address.raw() + channel_index as u64 * self.sample_count as u64 * 4).into()
    }
}

/// User callback type that receives published updates of the device audio rings.  Implemented for
/// `Send + 'static` closures.
pub trait ImportSink: Send + 'static {
    /// Callback must return retired sample count.
    fn process(&mut self, view: &DeviceAudioView) -> Result<u32, MutateError>;
}

impl<F> ImportSink for F
where
    F: FnMut(&DeviceAudioView) -> Result<u32, MutateError> + Send + 'static,
{
    fn process(&mut self, view: &DeviceAudioView) -> Result<u32, MutateError> {
        self(view)
    }
}

/// The owned side a device import stream.  Data import to the GPU is handled by an owned a reader
/// thread.  This type gathers up ownership and provides an interface to the published control data
/// for host-side and setting up device-side reads.
pub struct AudioImport {
    /// Just a persistent bag of bytes being used for ad-hoc sub-allocations. (DEBT).
    buffer: MappedAllocation<u8>,
    /// Reader thread.
    read_thread_handle: Option<JoinHandle<Result<(), MutateError>>>,
    /// Address, offsets of each channel, length in samples.
    ring_layout: DeviceAudioView,
    /// Shared control data.
    control: Arc<Control>,
}

impl AudioImport {
    /// Create an import from the audio server to the device, using `AudioConsumer` as an embodiment
    /// of `AudioChoice` until we cut out the middle man (see mis-indirection DEBT).
    ///
    /// `sample_count` is the length of each channel's ring buffer in samples.  More buffer means
    /// less potential for bursts leading to discontinuities.
    pub fn new<S: ImportSink>(
        device: &Device,
        mut rx: AudioConsumer,
        sample_count: u32,
        mut import_sink: S,
    ) -> Result<AudioImport, MutateError> {
        let control = Arc::new(Control {
            write_head: AtomicU64::new(0),
            read_head: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        });

        // DEBT the `rx` has no idea how many channels it has 🤡🤦
        let channel_count: u32 = 2;
        let (buffer, ring_layout) = {
            let size = sample_count * 4 * channel_count;
            let mut buffer: MappedAllocation<u8> = MappedAllocation::new(device, size as usize)?;

            // f32 0.0 is all-zero bytes, so this write is safe.
            buffer.as_mut_slice().fill(0u8);
            buffer.flush(device)?;

            // LIES Just made up the rate and channel count to work downstream
            // DEBT sample rate / channel count.  The fix can likely be implemented incidental to
            // getting rid of the intermediate consumer, so let's do that.
            let base_address = buffer.device_address(device)?;
            (
                buffer,
                DeviceAudioView {
                    base_address: base_address.into(),
                    channel_count: 2,
                    sample_count: sample_count as u32,
                    sample_rate: 48_000,
                    read_head: 0,
                    write_head: 0,
                    timing: None,
                },
            )
        };

        // Planar layout we will de-interleave into.
        let channel_bytes = sample_count as usize * 4;
        let total_bytes = channel_bytes * channel_count as usize;
        let channels = channel_count as usize;

        // Owned write view for the thread.
        // NOTE write view mutability story is not worked out at all, but not super critical here
        // either.  There's one canonical writer, the audio server.
        let mut view = buffer.write_view(device);

        let writer_control = control.clone();
        let mut timing = Some(rx.timing()?);

        let read_thread_handle = Some(std::thread::spawn(move || {
            let mut scratch = vec![0u8; (4 * channel_count * sample_count) as usize];
            let mut write_head: u64 = 0;
            let mut read_head: u64 = 0;

            while !writer_control.closed.load(Ordering::Relaxed) {
                // Wait up to 16ms for a chunk and then warn that chunks are late.
                match rx.wait(std::time::Duration::from_micros(16_000)) {
                    // LIES we don't use `got` because it's not guaranteed available until we copy
                    // it from the pipewire ingestion ring to our scratch buffer.
                    Ok(_got) => {
                        let read = rx.read(&mut scratch)?;
                        let frame_bytes = (4 * channel_count) as usize;

                        // NOTE not checking for partial frames as pipewire seems well-behaved so far.
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
                        for c in 0..channels {
                            let ring_base = c * channel_bytes;
                            for s in 0..to_write {
                                let src = (s * channels + c) * 4;
                                let logical = (start + s as u64) % sample_count as u64;
                                let dst_byte = ring_base + logical as usize * 4;
                                dst[dst_byte..dst_byte + 4].copy_from_slice(&scratch[src..src + 4]);
                            }
                        }

                        // NOTE flush is a no-op on coherent memory.  Whole-buffer flush costs some
                        // coherence traffic but never changes unmodified slots, and the written
                        // ranges are annoying to compute per channel across a wrap.
                        view.flush_range(0, total_bytes as u64)?;

                        // Publish new write head
                        write_head += to_write as u64;
                        writer_control
                            .write_head
                            .store(write_head, Ordering::Release);

                        // Pick up new timing data, which is written by now.
                        match rx.timing() {
                            Ok(new_time) => {
                                timing = Some(new_time);
                            }
                            Err(e) => {
                                // Realistically poisoning was the only error type.  Suddenly closed
                                // upstream still has a phase estimate, just one that will never
                                // arrive again.
                                return Err(e);
                            }
                        };
                        // Ring update is flushed & coherent.  Host view of state is updated.  Call
                        // the callback with the updated view, including the latest timing data.
                        let mut device_ring_view = ring_layout.clone();
                        device_ring_view.timing = timing;
                        device_ring_view.write_head = write_head;
                        device_ring_view.read_head = read_head;
                        let retired = match import_sink.process(&device_ring_view) {
                            Ok(retired) => retired as u64,

                            Err(e) => {
                                println!("Audio import process callback failed: {:?}", e);
                                return Err(e);
                            }
                        };
                        // A sink that retires more than it was shown is a contract violation.
                        // Clamping keeps the invariant rather than trusting the return blindly.
                        debug_assert!(retired <= write_head - read_head, "sink over-retired");
                        read_head = (read_head + retired).min(write_head);
                        writer_control.read_head.store(read_head, Ordering::Release);
                    }
                    Err(MutateError::Timeout(_)) => {
                        // DEBT tracing and environment toggles for scoped logging 💀
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
        Ok(AudioImport {
            buffer,
            read_thread_handle,
            ring_layout,
            control,
        })
    }

    /// Will set a flag for upstream and returns when that thread joins.  This method blocks.
    pub fn destroy(&mut self, device: &Device) -> Result<(), MutateError> {
        // tombstone, join the reader thread, and destroy the allocation.
        self.control.closed.store(true, Ordering::Relaxed);
        let join_result = if let Some(handle) = self.read_thread_handle.take() {
            handle.join()
        } else {
            Ok(Ok(()))
        };

        // MAYBE  still give the buffer back and let the caller queue the deletion.
        // DEBT no sub-allocator, so memory also needs deferred deletion.
        device.deletion_queue.push(self.buffer.buffer);
        device.deletion_queue.push(self.buffer.memory);

        join_result.map_err(|_| MutateError::AudioTerminate)?
    }
}
