// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Audio
//!
//! Select a device.  Set up stream from server to device.  Run a callback on each audio tick.  The
//! callback pumps an audio graph.  The outputs of the audio graph are published best-effort
//! (consumers can choose to wait on upcoming dispatches or not) via `AudioOutputs`.  On the host
//! side we are publishing addresses and sizes so that the downstream dispatches can point at the
//! right memory.
//!
//! Audio and video will basically never tick on the same clock or at the same rate, and VRR
//! displays and other frontends will just further expose the independence.  The tracking & slew and
//! re-sampling are all unavoidable and consumers **must** be built these dynamics in mind.
//!
//! Eventually the audio side will become a runtime component that can drive an audio graph and
//! manage pushing values to reactive dependents (pointer swaps, read/write heads etc).

// DEBT Reactive updates.  Keep modularizing audio pipelines for downstreams.  Dispatching a bunch
// of IIRs and other audio processing will parallelize easily and the output addresses can just be
// exposed to video pipelines without dynamic resolution for now.
// NEXT Extend vk::Device for things that don't require the wrapped Device.  Then &Device grows
// those methods via Deref.
// MAYBE get rid of fences on most submissions?
// NOTE resource reactivity for audio pipelines is interesting.  The signal to resize the screen for
// example might trigger resource recreation.  The resource change notifications then need to be
// sent to the thread and finally pulled by.. the caller during the callback.
// NEXT automatically promote u32 -> UInt32 and newtypes thereof
// XXX DeviceAddress and vk::DeviceAddress are too redundant.
// NOTE Like the visualizer, support for a dynamic set of audio pipelines is necessary.  Lacking
// that, the RMS pipeline was fully removed to add the DFT.  See blame.  We would like support for
// mixtures of pipelines, and the resource runtime will need to orchestrate this.

pub mod dft;
pub mod downsample;
pub mod plan;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use ash::vk;
use mutate_lib::{self as utate, audio, prelude::*};

pub struct CallbackResources {
    /// How far have we read into the data so far?
    // NOTE unbuffered, same clock
    consume_head: u64,
    pool_ring: PoolRing<Graphics, 4>,
    dft: dft::Dft,
    downsample: downsample::Downsample,
    outputs: Arc<Mutex<Option<AudioOutputs>>>,
    last_consumed: u32,
    dead: AtomicBool, // Hacky tombstone to stop dispatches faster.
}

/// Until some kind of reactive parameter design is done, cram outputs onto this struct so that each
/// visualization has a common interface.  The resource spec system's upcoming role is to inject
/// parameters onto an input structure for the command recorder to pull.
#[derive(Clone, Debug)]
pub struct AudioOutputs {
    pub dft: dft::DftOutput,
    pub downsample: downsample::DownsampleOutput,
    // Dynamic range gain factor location (on device)
    // Timing data for downstream buffered tracking
    // XXX isn't the timing daata available for clone?
}

pub struct Audio {
    context: AudioContext,
    resources: *mut CallbackResources,
    pub audio_import: AudioImport<2>,
    // NOTE suppose we already have the resources runtime ready.  The Option goes away because this
    // is a dependency for the downstream renderer.  They don't want to care if the upstream is
    // running.  No input, don't call them.  The Arc goes away because the runtime owns the values.
    // The Mutex goes away because we only see one epoch per frontend call.
    pub outputs: Arc<Mutex<Option<AudioOutputs>>>,
}

const EMPTY_SPAN: DeviceSpan = DeviceSpan { base: 0, len: 0 };

impl Audio {
    // NOTE all of the audio processing can happen in compute queues, but the resources for hand-off
    // to graphics for presentation will need concurrent access since the ring buffers cannot be
    // QFOT and we otherwise have to do awkward window copying.
    pub fn new(device: &Device, queue: &QueueRef<Graphics>) -> Result<Self, utate::MutateError> {
        // NEXT Handle audio choice via deafult + config so that user input is only necessary where
        // explicitly requested or updated at runtime.
        let context = audio::AudioContext::new()?;
        println!("Choose the audio source:");
        let mut first_choices = Vec::new();
        let check = |choices: &[audio::AudioChoice]| {
            first_choices.extend_from_slice(choices);
        };
        context.with_choices_blocking(check).unwrap();
        let max_name_width = first_choices
            .iter()
            .map(|c| c.name().len())
            .max()
            .unwrap_or(0);
        first_choices.iter().enumerate().for_each(|(i, c)| {
            println!("[{}] {:<max_name_width$}  [{}]", i, c.name(), c.kind());
        });
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        // FIXME handle invalid input choices.
        let choice_idx = input.trim().parse().unwrap();
        let choice = first_choices.remove(choice_idx);
        let rx = context.connect(&choice, "µTate")?;

        // Allocate the ring so that callback and later the DeviceImport
        let (buffer, ring_layout) = AudioImport::<2>::allocate(device, 4096)?;

        // MAYBE IIRC Untorn is right type here.  Consumer only wants the latest value and ticks
        // independently, so best-effort on-time delivery is fine.  In any case, it's a short lock
        // to read for now.  Option is only because the RingLayout is currently only known after the
        // import begins.  Memory addresses needed by both upstream and downstream need to be
        // decided / resolvable.
        let outputs = Arc::new(Mutex::new(None));
        let callback_queue = queue.clone();
        let callback_device = device.as_raw().clone();
        let callback_outputs = outputs.clone();
        let pool_ring = PoolRing::new(device, &callback_queue)?;

        let downsample = downsample::Downsample::new(device, ring_layout)?;
        let dft = dft::Dft::new(device, downsample.level_layout(2))?;
        let resources = Box::into_raw(Box::new(CallbackResources {
            consume_head: 0,
            pool_ring,
            dft,
            downsample,
            outputs: outputs.clone(),
            last_consumed: 0,
            dead: false.into(),
        }));

        // Pass resources address into the callback.  Ownership and cleanup remain with us.
        let addr = resources as usize;
        let on_flush = move |state: &utate::audio::import::DeviceRingView<2>| {
            // Drive the audio pipeline (◕‿◕)♡
            let outputs = &callback_outputs;
            let device = &callback_device;
            let timing = state.timing;
            let res = unsafe { &mut *(addr as *mut CallbackResources) };

            if res.dead.load(Ordering::Acquire) {
                // Stop advancing, stop dispatching, tell pipewire upstream it's okay to reclaim
                // everything.
                return Ok(state.occupied_len());
            }

            let regions = state.regions_since(res.consume_head);
            if regions[0].occupied_len() == 0 {
                return Ok(0);
            }
            let layout = state.ring_layout;

            let (pool, intent) = match res.pool_ring.acquire(device, 16_000_000_000) {
                Ok(acquired) => acquired,
                Err(e) => {
                    println!("Pool acquisition: {:?}", e);
                    // XXX this error path has not been scrutanized.  Or encountered.
                    return Ok(0);
                }
            };
            let cb = pool.primary(device)?;

            let downsample::DownsampleDispatch {
                output: downsample_out,
                ready: downsample_ready,
                consumed: downsample_consumed,
            } = res.downsample.dispatch(device, &cb, state)?;
            let downsample_previous = downsample_ready.predecessor();

            let dft_out_wait = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);
            let barriers = [dft_out_wait];
            let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
            unsafe { device.cmd_pipeline_barrier2(**&cb, &dep) };

            // TODO get downsample outputs for the DFT

            let dft::DftDispatch {
                consumed: dft_consumed,
                ready: dft_ready,
                output: dft_out,
            } = res.dft.dispatch(device, &cb, &downsample_out)?;
            let dft_previous = dft_ready.predecessor();

            // DFT dependents need this barrier.  The need is graph-detected later.
            let dft_out_wait = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);
            let barriers = [dft_out_wait];
            let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
            unsafe { device.cmd_pipeline_barrier2(**&cb, &dep) };

            // NEXT use shared on outputs and wait on the correct semaphores
            let done = cb.end(device)?;
            callback_queue
                .submission()
                .wait(downsample_previous, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .wait(dft_previous, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .execute(done)
                // XXX Rip out the individual semaphores.  Most consumers will prefer to see one
                // consistent audio graph.
                .signal(downsample_ready, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .signal(dft_ready, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .signal(intent, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .submit(&callback_device, vk::Fence::null())?;

            // All data has been sent down the pipe.
            res.consume_head = state.write_head;

            *outputs.lock().unwrap() = Some(AudioOutputs {
                dft: dft_out,
                downsample: downsample_out,
            });

            // XXX Super hack here, but consistent.  We should catch up a bit differently to try and
            // ensure good recovery after allowing the producer to stall.  The real fix is to skip
            // half of the input and allow the producer to reclaim it.  The remaining half should
            // then be read a bit faster to get back up to the write head.  We want maximum ring
            // slack because the consumer slews in **output**.  This callback is reading **input**.
            // if state.occupied_len() > 1024 {
            //     res.last_consumed = 1024;
            //     Ok(consumed - 1024)
            // } else {
            //     let last = res.last_consumed;
            //     res.last_consumed = consumed;
            //     Ok(last)
            // }

            Ok(state.occupied_len())
        };

        // Create the import, which fires the callback.
        let audio_import =
            AudioImport::with_allocation(device, rx, (buffer, ring_layout), on_flush)?;

        Ok(Self {
            context,
            audio_import,
            resources,
            outputs,
        })
    }

    pub fn destroy(mut self, device: &Device) -> Result<(), MutateError> {
        let Audio {
            context,
            resources,
            audio_import: mut consumer,
            outputs,
        } = self;
        // Notify the callback that it should stop sending anything downstream
        let resources = unsafe { Box::from_raw(resources) };
        resources.dead.store(true, Ordering::Release);

        // Destroy the audio stream first so it will stop calling our callback.
        consumer.destroy(device)?;

        // If you're getting validation issues, check that sinks (video) are being killed first.
        resources.pool_ring.drain(device, 1_000_000_000)?;
        resources.pool_ring.destroy(device);
        resources.downsample.destroy(device);
        resources.dft.destroy(device);
        // context has no vulkan resources and may just drop.
        Ok(())
    }
}
