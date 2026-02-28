#![cfg(feature = "vulkan")]

use mutate_lib::vulkan;

#[test]
fn context_creation_test() {
    let context = vulkan::context::VkContext::new();
    context.destroy();
}
