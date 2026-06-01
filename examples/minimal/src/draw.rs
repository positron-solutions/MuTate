/// # Draw
///
/// The [`HelloRenderer`] just implements the interface that `ComputePresent` needs, accepting a
/// command buffer and an `AcquiredImage` from the swapchain.  It builds one pipeline.  For
/// rendering, it just pushes some simple push constants, various sine waves into the color
/// channels.  This color is produced at every pixel in the image by dispatching the shader once for
/// each screen pixel.  Part of the dimensions are encoded in the Slang shader and usually match the
/// warp size on the device, usually 32 lanes.
use ash::vk;
use mutate_lib::{self as utate, prelude::*};
use utate::vulkan::resource::{buffer, image};

#[compute_pipeline(
    compute = stage!("hello/compute", Compute, c"main"),
    push = push!(HelloConstants {
        pub r: Float,
        pub g: Float,
        pub b: Float,
        pub a: Float,
        pub window_width: Float,
        pub window_height: Float,
        pub output_idx: SsboIdx,
    }),
)]
pub struct HelloPipeline;

pub struct HelloDraw {
    pipeline: ComputePipeline<HelloPipeline>,
    color: [f32; 4],

    output_buffer: Option<buffer::MappedAllocation<rgb::Rgba<u8>>>,
    output_idx: SsboIdx,
}

impl HelloDraw {
    pub fn new(context: &DeviceContext, color: [f32; 4]) -> Self {
        Self {
            pipeline: ComputePipeline::<HelloPipeline>::new(context).unwrap(),
            color,
            output_buffer: None,
            output_idx: SsboIdx::INVALID,
        }
    }

    pub fn provision(
        &mut self,
        context: &mut DeviceContext,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        let output_buffer =
            buffer::MappedAllocation::new((size.width * size.height) as usize, context)?;

        self.output_idx = output_buffer.bound(context);
        self.output_buffer = Some(output_buffer);

        Ok(())
    }

    pub fn draw(
        &self,
        context: &DeviceContext,
        cb: &RecordingBuffer<Graphics, OneTime>,
        acquired_image: &AcquiredImage,
    ) {
        let device = context.device();
        let extent = acquired_image.extent;

        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_pre(&cb, context);

        let push = HelloConstants {
            r: self.color[0].into(),
            g: self.color[1].into(),
            b: self.color[2].into(),
            a: self.color[3].into(),
            window_width: (extent.width as f32).into(),
            window_height: (extent.height as f32).into(),
            output_idx: self.output_idx,
        };
        // XXX allow pushing to wrapped buffers
        self.pipeline.push(context, **cb, &push);

        // This dispatch math needs to respect the compute stage's declared dimensions.  We can make
        // that adjustable with specialization constants during the pipeline compilation.  This math
        // can be abstracted with some reflection data as the general pattern is that we need the
        // dispatch geometry to exceed the invocation area necessary for the output.  The slang code
        // checks bounds and masks off lanes that are unnecessary.
        let wg_x = 8;
        let wg_y = 4;
        let dispatch_x = (extent.width + wg_x - 1) / wg_x;
        let dispatch_y = (extent.height + wg_y - 1) / wg_y;
        self.pipeline
            .dispatch(context, **cb, dispatch_x, dispatch_y, 1);

        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_post(&cb, context);

        let region = buffer::buffer_image_copy_full(extent);
        unsafe {
            device.cmd_copy_buffer_to_image(
                **cb,
                self.output_buffer.as_ref().unwrap().buffer,
                acquired_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    pub fn destroy(self, context: &mut DeviceContext) -> Result<(), utate::MutateError> {
        unsafe {
            self.pipeline.destroy(context);
            if let Some(allocated) = self.output_buffer {
                allocated.destroy(&context)?;
                context.descriptors.unbind_ssbo(self.output_idx);
            }
        }
        Ok(())
    }
}
