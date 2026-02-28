#![cfg(feature = "vulkan")]

use ash::vk;
use mutate_lib::vulkan;

#[test]
fn test_context_lifecycle() {
    let context = vulkan::context::VkContext::new();
    context.destroy();
}

#[test]
fn test_image_lifecycle() {
    use vulkan::image;
    let context = vulkan::context::VkContext::new();

    let extent = vk::Extent2D {
        width: 1,
        height: 1,
    };
    let format = vk::Format::R8G8B8A8_SRGB;
    let flags = vk::ImageUsageFlags::INPUT_ATTACHMENT;
    let image = image::Image::new(&context, extent, format, flags).unwrap();
    image.destroy(&context);
    context.destroy();
}

#[test]
fn test_buffer_lifecycle() {
    use vulkan::buffer;
    let context = vulkan::context::VkContext::new();

    let buffer = buffer::MappedAllocation::<u8>::new(1, &context).unwrap();
    buffer.destroy(&context);

    context.destroy();
}
