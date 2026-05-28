// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use std::slice;

use ash::vk;

use mutate_vulkan::{prelude::*, resource::buffer};

#[test]
fn dispatch_increment_read_back() {
    #[compute_pipeline(
        compute = stage!("test/increment", Compute, c"main"),
        push = push!(IncrementConstants {
            output_buffer_idx: SsboIdx,
        }),
    )]
    pub struct IncrementPipeline;

    with_context!(|device_ctx, _vulkan_ctx| {
        // NEXT any test with dispatch can re-use this kind of initialization
        let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
        let mut pool = CommandPool::<Compute, OneTime>::transient(&device_ctx, &queue).unwrap();
        let cb = pool.primary(&device_ctx).unwrap();

        let mut output_buffer = buffer::MappedAllocation::<u32>::new(1, &device_ctx).unwrap();
        output_buffer.as_mut_slice()[0] = 41;
        output_buffer.flush(&device_ctx).unwrap();
        let output_idx: SsboIdx = output_buffer.bound(&mut device_ctx);

        let pipeline = ComputePipeline::<IncrementPipeline>::new(&device_ctx).unwrap();
        pipeline.push(
            &device_ctx,
            *cb, // XXX accept non-raw CB
            &IncrementConstants {
                output_buffer_idx: output_idx,
            },
        );
        pipeline.dispatch(&device_ctx, *cb, 1, 1, 1);

        // In future runtime support, barriers will come from runtime dependency, a declarative API
        // that is unrelated to testing.  This barrier is for CPU readback. We might be able to
        // create some read / write intent API for this case, which is not that related to any
        // future runtime API.
        let device = device_ctx.device();
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

        // Synchronize and submit
        let done = cb.end(&device_ctx).unwrap();
        let mut semaphore = device_ctx.make_timeline_semaphore().unwrap();
        let intent = semaphore.next_signal();
        let wait_value = intent.wait_value();
        // XXX use device context
        queue
            .submission()
            .execute(done)
            .signal(intent, vk::PipelineStageFlags2::COMPUTE_SHADER)
            .submit(device, vk::Fence::null())
            .unwrap();

        // Wait host-side, invalidate and read back.
        wait_value.wait(&device_ctx, 100_000_000).unwrap();
        output_buffer.invalidate(&device_ctx);
        let observed = output_buffer.as_mut_slice()[0];
        assert_eq!(observed, 42);

        // DEBT RAII and manual destruction is still conservative
        semaphore.destroy(&device_ctx);
        pool.destroy(&device_ctx);
        output_buffer.destroy(&device_ctx);
        pipeline.destroy(&device_ctx);
    })
}
