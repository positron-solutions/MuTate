// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Audio
//!
//! Select a device.  Set up stream from server to device.  Run a callback on each audio tick.

use ash::vk;
use mutate_lib::{
    self as utate,
    audio::{
        self,
        import::{ChannelRegion, Consumer, DeviceSpan},
        AudioContext,
    },
    prelude::*,
};

// NEXT Slang texel types
// NEXT Extend vk::Device for things that don't require Device.  Then &Device grows those methods
// via Deref.
// MAYBE get rid of fences on most submissions?
// NOTE resource reactivity for audio pipelines is interesting.  The signal to resize the screen for
// example might trigger resource recreation.  The resource change notifications then need to be
// sent to the thread and finally pulled by.. the caller during the callback.
// NEXT automatically promote u32 -> UInt32 and newtypes thereof
// XXX DeviceAddress and vk::DeviceAddress are too duplicated.

#[compute_pipeline(
    compute = stage!("audio/rms", Compute, c"main"),
    push = push!(RmsConstants {
        pub left_head: DeviceAddress,
        pub right_head: DeviceAddress,
        pub count_head: UInt,
        pub left_tail: DeviceAddress,
        pub right_tail: DeviceAddress,
        pub count_tail: UInt,
        pub output: DeviceAddress,
    }),
)]
struct RmsComputePipeline;

pub struct CallbackResources {
    /// How far have we read into the data so far?
    consume_head: u64,
    pool_ring: PoolRing<Graphics, 2>,
    pipeline: ComputePipeline<RmsComputePipeline>,
}

pub struct Audio {
    context: AudioContext,
    resources: *mut CallbackResources,
    pub consumer: Consumer<2>,
    pub output: DeviceBuffer,
    pub output_address: vk::DeviceAddress,
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

        let callback_queue = queue.clone();
        let callback_device = device.as_raw().clone();

        // Four bytes to tell the world!  Note, this refreshes so fast that downstreams will not be
        // able to track it, leading to aliasing.
        let output = utate::vulkan::resource::buffer::DeviceBuffer::new(device, 4)?;
        let output_address = output.device_address(device)?;
        let resources = Box::into_raw(Box::new(CallbackResources {
            consume_head: 0,
            pool_ring: PoolRing::new(device, &callback_queue)?,
            pipeline: ComputePipeline::<RmsComputePipeline>::new(device)?,
        }));
        let addr = resources as usize;
        // XXX head_offset and head_count?  Ah counting heads!!!!
        let mut constants = RmsConstants {
            left_head: DeviceAddress::NULL,
            right_head: DeviceAddress::NULL,
            count_head: 0.into(),
            left_tail: DeviceAddress::NULL,
            right_tail: DeviceAddress::NULL,
            count_tail: 0.into(),
            output: output_address.clone().into(),
        };
        let on_flush = move |state: &utate::audio::import::DeviceRingView<2>| {
            let device = &callback_device;
            let timing = state.timing;
            let res = unsafe { &mut *(addr as *mut CallbackResources) };

            let (pool, intent) = res.pool_ring.acquire(device, 0)?;
            let cb = pool.primary(device)?;

            // XXX we're not interpreting timing data here at all.
            let regions = state.regions_since(res.consume_head);
            let layout = state.ring_layout;

            let [left, right] = state.regions_since(res.consume_head);

            // Zero-length spans are safe: the shader loops `k < count`.
            let mut seed = |l: Option<DeviceSpan>, r: Option<DeviceSpan>| {
                let (l, r) = (l.unwrap_or(EMPTY_SPAN), r.unwrap_or(EMPTY_SPAN));
                debug_assert_eq!(l.len, r.len, "channels desynced");
                (l.base.into(), r.base.into(), l.len.into())
            };

            let mut l_spans = left.spans();
            let mut r_spans = right.spans();

            let (lh, rh, nh) = seed(l_spans.next(), r_spans.next());
            constants.left_head = lh;
            constants.right_head = rh;
            constants.count_head = nh;

            let (lt, rt, nt) = seed(l_spans.next(), r_spans.next());
            constants.left_tail = lt;
            constants.right_tail = rt;
            constants.count_tail = nt;

            // println!("regions since: {:?}", &regions);

            res.pipeline.push(device, *cb, &constants);
            // ROLL manual dispatch due to problem in the dispatch thread-safety.
            unsafe {
                device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::COMPUTE, *res.pipeline);
                device.cmd_dispatch(*cb, 1, 1, 1);
            }

            let done = cb.end(device)?;
            callback_queue
                .submission()
                .execute(done)
                .signal(intent, vk::PipelineStageFlags2::COMPUTE_SHADER)
                .submit(&callback_device, vk::Fence::null())?;
            res.consume_head = state.write_head;

            // Returns all of the data to allow it all to be reclaimed
            Ok(state.occupied_len())
        };
        let consumer = context.import_to_device(device, &choice, 6400, "µTate", on_flush)?;
        Ok(Self {
            context,
            consumer,
            resources,
            output,
            output_address,
        })
    }

    pub fn destroy(mut self, device: &Device) -> Result<(), MutateError> {
        let Audio {
            context,
            output,
            resources,
            mut consumer,
            output_address,
        } = self;
        consumer.destroy(device)?;
        let resources = unsafe { Box::from_raw(resources) };
        // If you're getting validation issues, check that sinks (video) are being killed first.
        resources.pool_ring.drain(device, 1_000_000_000)?;
        resources.pool_ring.destroy(device);
        resources.pipeline.destroy(device);
        output.destroy(device);
        // context has no vulkan resources and may just drop.
        Ok(())
    }
}
