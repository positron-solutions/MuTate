// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use ash::vk;
use mutate_lib::vulkan;

#[test]
fn image_lifecycle() {
    use vulkan::resource::image;
    vulkan::with_context!(|device| {
        let extent = vk::Extent2D {
            width: 1,
            height: 1,
        };
        let format = vk::Format::R8G8B8A8_SRGB;
        let flags = vk::ImageUsageFlags::INPUT_ATTACHMENT;
        let image = image::Image::new(&device, extent, format, flags).unwrap();
        image.destroy(&device).unwrap();
    })
}

#[test]
fn buffer_lifecycle() {
    use vulkan::resource::buffer;
    vulkan::with_context!(|device| {
        let buffer = buffer::MappedAllocation::<u8>::new(1, &device).unwrap();
        buffer.destroy(&device).unwrap();
    })
}

#[test]
fn buffer_bind() {
    use vulkan::resource::buffer;
    vulkan::with_context!(|device| {
        let buffer = buffer::MappedAllocation::<u8>::new(1, &device).unwrap();
        let index = buffer.bound(&mut device);
        println!("buffer bound to descriptor slot: {:?}", index);
        buffer.destroy(&device).unwrap();
    })
}

#[test]
fn buffer_device_address() {
    use vulkan::resource::buffer;
    vulkan::with_context!(|device| {
        let buffer = buffer::MappedAllocation::<u8>::new(1, &device).unwrap();
        let device_address = buffer.device_address(&device);
        buffer.destroy(&device).unwrap();
    })
}

// XXX incomplete without dispatch
#[test]
fn buffer_readback() {
    use vulkan::resource::buffer;
    vulkan::with_context!(|device| {
        let mut buffer = buffer::MappedAllocation::<u8>::new(1, &device).unwrap();
        buffer.as_mut_slice()[0] = 255;
        buffer.flush(&device).unwrap();

        // Dispatch something

        device.wait_idle().unwrap();
        buffer.destroy(&device).unwrap();
    })
}

#[test]
fn shader_load() {
    vulkan::with_context!(|device| {
        let shader = vulkan::resource::shader::ShaderModule::load(&device, "test/compute");
    })
}
