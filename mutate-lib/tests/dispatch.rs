// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use std::slice;

use ash::vk;

use mutate_macros::*;
use mutate_vulkan as vulkan;
use mutate_vulkan::prelude::*;

use mutate_vulkan::dispatch::cb::{ExecutableBuffer, RecordingBuffer};
use mutate_vulkan::dispatch::pool::CommandPool;
use mutate_vulkan::resource::buffer;

#[test]
fn dispatch_increment_read_back() {
    #[compute_pipeline(
        compute = stage!("test/increment", Compute, c"main"),
        push = push!(IncrementConstants {
            output_buffer_idx: SsboIdx,
        }),
    )]
    pub struct IncrementPipeline;

    vulkan::with_context!(|device_ctx, _vulkan_ctx| {
        let pipeline = ComputePipeline::<IncrementPipeline>::new(&device_ctx).unwrap();
        let mut output_buffer = buffer::MappedAllocation::<u32>::new(1, &device_ctx).unwrap();
        output_buffer.as_mut_slice()[0] = 41;
        output_buffer.flush(&device_ctx).unwrap();
        let output_idx: SsboIdx = output_buffer.bound(&mut device_ctx);

        let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
        let pool = CommandPool::<Compute, OneTime>::transient(&device_ctx, &queue).unwrap();
        let cb = pool.primary(&device_ctx).unwrap();
        let device = device_ctx.device(); // XXX use device context in more places

        let push_constants = IncrementConstants {
            output_buffer_idx: output_idx,
        };
        pipeline.push(&device_ctx, *cb, &push_constants);
        pipeline.dispatch(&device_ctx, *cb, 1, 1, 1);

        // XXX this barrier is pretty ceremonious.
        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        let dependency_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));
        unsafe {
            device.cmd_pipeline_barrier2(*cb, &dependency_info);
        }

        let done = cb.end(&device_ctx).unwrap();
        // XXX submit info can be gotten off of the done buffer
        let cb_info = vk::CommandBufferSubmitInfo::default().command_buffer(*done);
        let raw = done.kill(&device_ctx).unwrap(); // XXX dropwire bullshit

        // Synchronize and submit
        // XXX synchronization on the pool semaphore?
        let wait = device_ctx.make_timeline_semaphore(0).unwrap();
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(wait.as_raw())
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
        let submit = vk::SubmitInfo2::default()
            .signal_semaphore_infos(slice::from_ref(&signal_info))
            .command_buffer_infos(slice::from_ref(&cb_info));

        // XXX the submission is not interesting.. we can do safe... use our Queue and DeviceCtx
        unsafe {
            device
                .queue_submit2(queue.raw(), &[submit], vk::Fence::null())
                .unwrap();
        }

        // after waiting on the dispatch, read back the value from the buffer.
        // XXX submissions should contain enough data to be waited on.
        let wait_raw = wait.as_raw();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&wait_raw))
            .values(slice::from_ref(&1u64));
        unsafe {
            device.wait_semaphores(&wait_info, 1_000_000u64).unwrap();
        }

        // The read back
        output_buffer.invalidate(&device_ctx);
        let observed = output_buffer.as_mut_slice()[0];

        assert_eq!(observed, 42);
        println!("observed output: {observed}");

        // Clean up
        unsafe { device.destroy_semaphore(wait.into_raw(), None) };
        pool.destroy(&device_ctx);
        output_buffer.destroy(&device_ctx);
        pipeline.destroy(&device_ctx);
    })
}
