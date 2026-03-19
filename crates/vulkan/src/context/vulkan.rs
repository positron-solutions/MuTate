// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # VkContext
//!
//! Talk to the platform's Vulkan implementation.  Inspect physical devices for required Vulkan
//! extension support.  Prioritize devices.  Set up Vulkan for the platform's surface
//! implementation.
//!
//! For headless rendering to a TUI etc, we likely don't need any platform extensions, but for
//! rendering to a window, check `with_extensions` and its use of `ash_window` to cooperate with a
//! `winit::event_loop::ActiveEventLoop` usually, although these specific dependencies are not
//! strictly required.

use std::{os::raw::c_char, ffi::{c_void, CStr}};

use ash::vk;

use mutate_assets as assets;

use crate::present::surface::VkSurface;

/// The entry and instance represent a connection to the Vulkan implementation.
pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
}

// Waiting for this to hit ash?
pub const KHR_PRESENT_TIMING_NAME: &CStr = c"VK_EXT_present_timing";
// DEBT currently the requirements and support checks are all hardcoded.  There is duplicate
// information in side of VkContext that needs to be runtime decided and then passed into
// this function to avoid the hardcode.
pub(crate) const DEVICE_EXTENSIONS: [&CStr; 8] = [
    vk::KHR_SWAPCHAIN_NAME,
    // MAYBE this is Windows only?  Evidently only old windows?
    // vk::EXT_FULL_SCREEN_EXCLUSIVE_NAME,
    vk::EXT_EXTENDED_DYNAMIC_STATE_NAME,
    vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME,
    vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME,
    // XXX Remove / redundant
    // vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
    // NEXT better debug gating (see validation layer activation above).
    // Enables some debug functionality in shaders.
    vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME,
    vk::EXT_TOOLING_INFO_NAME,
    // MAYBE I might just need to install something, but this fails at runtime on my machine.
    // vk::EXT_DEBUG_UTILS_NAME,

    // MAYBE If we start running into lots of pipeline creation costs for slight variants,
    // we are advised to look at this extension.
    // vk::EXT_GRAPHICS_PIPELINE_LIBRARY_NAME,
    // ROLL holding off on this until other hardware vendors have supporting drivers.  This
    // is another path to reducing the cost of pipeline combinatorics.
    // vk::EXT_SHADER_OBJECT_NAME,

    // "gives an implementation the opportunity to reduce the number of indirections an
    // implementation takes to access uniform values, when only a few values are used"
    // XXX redundant
    // vk::EXT_INLINE_UNIFORM_BLOCK_NAME,
    // MAYBE So we can proactively change our memory behavior, downsampling etc.
    // vk::EXT_MEMORY_BUDGET_NAME,
    // vk::EXT_MEMORY_PRIORITY_NAME,
    // XXX redundant
    // vk::KHR_TIMELINE_SEMAPHORE_NAME,
    // ROLL VK_EXT_present_timing is still too new.  I have no supported devices / drivers yet.
    // KHR_PRESENT_TIMING_NAME,
    vk::KHR_PRESENT_WAIT_NAME,
    vk::KHR_PRESENT_ID_NAME,
];

// DEBT Errors instead of panics, but that might require a lot of test re-writing that should be
// done with macros to ease future pain.
impl VkContext {
    /// Basic context for testing.  Does not have platform extensions for windows etc.  Still useful
    /// for some workloads like compute.
    pub fn new () -> Self {
        Self::with_extensions(&[])
    }

    /// Context with `required_exts` for the display platform enabled.  You cannot create a context
    /// that will support any window without enabling some extensions.  Usually use `ash_window`.
    ///
    /// ```ignore
    /// // `event_loop` from winit::event_loop::ActiveEventLoop
    /// let required = ash_window::enumerate_required_extensions(event_loop);
    /// let vk_context = VkContext::with_extensions(required)
    /// ```
    pub fn with_extensions (required_exts: &'static [*const c_char]) -> Self {
        let entry = unsafe { ash::Entry::load().expect("failed to load Vulkan library") };
        assert_loader_version(&entry); // Checks 1.3+
        let available_entry_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };
        let available_names: Vec<&CStr> = available_entry_extensions
            .iter()
            .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
            .collect();
        for req in required_exts.iter() {
            let req_cstr = unsafe { CStr::from_ptr(*req) };
            assert!(
                available_names.contains(&req_cstr),
                "Required Vulkan extension {:?} not found",
                req_cstr
            );
        }

        let app_info =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let validation_layers = [
            // #[cfg(debug_assertions)]
            // NOTE Leaving this on all the time because there are still issues in `--release`
            // builds and we need to default to leaving it on via the dev shells or something.
            c"VK_LAYER_KHRONOS_validation".as_ptr()
        ];
        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_exts)
            .enabled_layer_names(&validation_layers);
        let instance = unsafe { entry.create_instance(&instance_ci, None).unwrap() };
        Self {
            entry,
            instance,
        }
    }

    pub fn surface_loader(&self) -> ash::khr::surface::Instance {
       ash::khr::surface::Instance::new(&self.entry, &self.instance)
    }

    /// Returns a list of physical devices that meet requirements, sorted in order of preference for
    /// discrete, integrated, and virtual, with memory heap sizes as the secondary sort key.
    ///
    /// Later filters can check queue families for presentation support, such as when rendering to a
    /// surface.  This can also be used to create prompts for the user.
    pub fn supported_devices(&self) -> Vec<SupportedDevice<NoPresentation>> {
        let physical_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .expect("No physical devices with Vulkan support found")
        };
        let mut physical_devices: Vec<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let props = unsafe { self.instance.get_physical_device_properties(physical_device) };
                let meets_version = {
                    let major = vk::api_version_major(props.api_version);
                    let minor = vk::api_version_minor(props.api_version);
                    major > 1 || (major == 1 && minor >= 3)
                };
                (meets_version && self.device_meets_features(physical_device) && self.device_meets_extensions(physical_device)).then_some((physical_device, props))
            })
            .collect();
        physical_devices.sort_by_key(|(physical_device, props)| {
            let device_type_rank = |t: vk::PhysicalDeviceType| match t {
                vk::PhysicalDeviceType::DISCRETE_GPU  => 0u8,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            };

            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(*physical_device) };
            let max_memory = mem_props.memory_heaps[..mem_props.memory_heap_count as usize]
                .iter()
                .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size)
                .max()
                .unwrap_or(0);
            (device_type_rank(props.device_type), std::cmp::Reverse(max_memory))
        });
        physical_devices
            .into_iter()
            .map(|(physical_device, _)| SupportedDevice::<NoPresentation>::new(physical_device, self))
            .collect()
    }

    fn device_meets_features(&self, physical_device: vk::PhysicalDevice) -> bool {

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default();

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default())
            .push_next(&mut features_1_3)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_1);

        unsafe {
            self.instance.get_physical_device_features2(physical_device, &mut features2);
        }

        let checks: &[(&'static str, bool)] = &[
            ("shader_int16",                                            features2.features.shader_int16 == vk::TRUE),
            ("1.1 storage_buffer16_bit_access",
            features_1_1.storage_buffer16_bit_access == vk::TRUE),
            // XXX Axe this feature
            // ("1.1 storage_input_output16",                              features_1_1.storage_input_output16 == vk::TRUE),
            ("1.1 shader_draw_parameters",                              features_1_1.shader_draw_parameters == vk::TRUE),
            ("1.1 storage_push_constant16",                             features_1_1.storage_push_constant16 == vk::TRUE),
            ("1.1 uniform_and_storage_buffer16_bit_access",             features_1_1.uniform_and_storage_buffer16_bit_access == vk::TRUE),
            ("1.2 buffer_device_address",                               features_1_2.buffer_device_address == vk::TRUE),
            ("1.2 descriptor_binding_partially_bound",                  features_1_2.descriptor_binding_partially_bound == vk::TRUE),
            ("1.2 descriptor_binding_sampled_image_update_after_bind",  features_1_2.descriptor_binding_sampled_image_update_after_bind == vk::TRUE),
            ("1.2 descriptor_binding_storage_buffer_update_after_bind", features_1_2.descriptor_binding_storage_buffer_update_after_bind == vk::TRUE),
            ("1.2 descriptor_binding_storage_image_update_after_bind",  features_1_2.descriptor_binding_storage_image_update_after_bind == vk::TRUE),
            ("1.2 descriptor_binding_variable_descriptor_count",        features_1_2.descriptor_binding_variable_descriptor_count == vk::TRUE),
            ("1.2 descriptor_indexing",                                 features_1_2.descriptor_indexing == vk::TRUE),
            ("1.2 draw_indirect_count",                                 features_1_2.draw_indirect_count == vk::TRUE),
            ("1.2 runtime_descriptor_array",                            features_1_2.runtime_descriptor_array == vk::TRUE),
            ("1.2 scalar_block_layout",                                 features_1_2.scalar_block_layout == vk::TRUE),
            ("1.2 shader_float16",                                      features_1_2.shader_float16 == vk::TRUE),
            ("1.2 shader_int8",                                         features_1_2.shader_int8 == vk::TRUE),
            ("1.2 shader_sampled_image_array_non_uniform_indexing",     features_1_2.shader_sampled_image_array_non_uniform_indexing == vk::TRUE),
            ("1.2 shader_storage_buffer_array_non_uniform_indexing",    features_1_2.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE),
            ("1.2 shader_storage_image_array_non_uniform_indexing",     features_1_2.shader_storage_image_array_non_uniform_indexing == vk::TRUE),
            ("1.2 shader_uniform_buffer_array_non_uniform_indexing",    features_1_2.shader_uniform_buffer_array_non_uniform_indexing == vk::TRUE),
            ("1.2 storage_buffer8_bit_access",                          features_1_2.storage_buffer8_bit_access == vk::TRUE),
            ("1.2 storage_push_constant8",                              features_1_2.storage_push_constant8 == vk::TRUE),
            ("1.2 uniform_and_storage_buffer8_bit_access",              features_1_2.uniform_and_storage_buffer8_bit_access == vk::TRUE),
            ("1.3 compute_full_subgroups",                              features_1_3.compute_full_subgroups == vk::TRUE),
            ("1.3 dynamic_rendering",                                   features_1_3.dynamic_rendering == vk::TRUE),
            ("1.3 maintenance4",                                        features_1_3.maintenance4 == vk::TRUE),
            ("1.3 shader_demote_to_helper_invocation",                  features_1_3.shader_demote_to_helper_invocation == vk::TRUE),
            ("1.3 synchronization2",                                    features_1_3.synchronization2 == vk::TRUE),
        ];

        let missing: Vec<&'static str> = checks
            .iter()
            .filter_map(|(name, present)| (!present).then_some(*name))
            .collect();

        if missing.is_empty() {
            true
        } else {
            // DEBT logging.  We could return an error but it's not an error for a device to be
            // missing functionality, only for all devices to be missing some functionality.
            #[cfg(debug_assertions)]
            {
                let props = unsafe {
                    self.instance.get_physical_device_properties(physical_device)
                };
                let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
                println!("Physical device: {}", name.to_string_lossy());
                for m in missing {
                    println!("missing feature: {}", m)
                }
            }
            false
        }
    }

    fn device_meets_version(&self, physical_device: vk::PhysicalDevice) -> bool {
        let props = unsafe { self.instance.get_physical_device_properties(physical_device) };
        let api_version = props.api_version;

        let major = vk::api_version_major(api_version);
        let minor = vk::api_version_minor(api_version);

        major > 1 || (major == 1 && minor >= 3)
    }

    fn device_meets_extensions (&self, physical_device: vk::PhysicalDevice) -> bool {
        let available_device_extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to enumerate device extensions")
        };

        let missing: Vec<&CStr> = DEVICE_EXTENSIONS
            .iter()
            .filter_map(|req| {
                let found = available_device_extensions.iter().any(|ext| unsafe {
                    let ext_cstr = CStr::from_ptr(ext.extension_name.as_ptr());
                    let req_cstr = CStr::from_ptr(req.as_ptr());
                    ext_cstr == req_cstr
                });
                if found {
                    None
                } else {
                    Some(*req)
                }
            })
            .collect();

        if missing.is_empty() {
            true
        } else {
            // DEBT logging.  We could return an error but it's not an error for a device to be
            // missing functionality, only for all devices to be missing some functionality.
            #[cfg(debug_assertions)]
            {
                let props = unsafe {
                    self.instance.get_physical_device_properties(physical_device)
                };
                let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) };
                println!("Physical device: {}", name.to_string_lossy());
                for m in missing {
                    println!("missing extension: {}", m.to_string_lossy())
                }
            }
            false
        }
    }

    pub fn destroy(&self) {
        unsafe {self.instance.destroy_instance(None)}
    }
}

/// Runs a block with an initialized Vulkan context, handling creation and destruction automatically.
///
/// Two forms are available depending on whether you need access to the underlying [`VkContext`]:
///
/// # Usage
///
/// ```rust
/// // With context only:
/// with_context!(|context| {
///     // `context: DeviceContext` is available here
/// });
///
/// // With both context and vk_context:
/// with_context!(|context, vk_context| {
///     // `context: DeviceContext` is available here
///     // `vk_context: VkContext` is available here
/// });
/// ```
#[macro_export]
macro_rules! with_context {
    (|$context:ident| $($body:tt)*) => {{
        let mut vk_context = $crate::context::VkContext::new();
        let mut $context = $crate::context::DeviceContext::new(&vk_context);
        let __result = (|| { $($body)* })();
        $context.destroy();
        vk_context.destroy();
        __result
    }};
    (|$context:ident, mut $vk_context:ident| $($body:tt)*) => {{
        let mut $vk_context = $crate::context::VkContext::new();
        let mut $context = $crate::context::DeviceContext::new(&$vk_context);
        let __result = (|| { $($body)* })();
        $context.destroy();
        $vk_context.destroy();
        __result
    }};
}

fn assert_loader_version (entry: &ash::Entry) {
    let loader_version = unsafe {
        entry.try_enumerate_instance_version()
            .unwrap_or(Some(vk::make_api_version(0, 1, 0, 0)))
            .unwrap()
    };

    if loader_version < vk::make_api_version(0, 1, 3, 0) {
        panic!("Loader does not support Vulkan 1.3");
    }
}

/// Marker type for `SupportedDevice`s that have been inspected for queue family support for a
/// surface.
#[derive(Debug, Clone)]
pub struct HasPresentation {queue_families: Vec<u32>}
/// Marker type for `SupportedDevice`s without any checked surface support.
#[derive(Debug, Clone)]
pub struct NoPresentation {}

#[derive(Debug, Clone)]
pub struct SupportedDevice<P = NoPresentation> {
    physical_device: vk::PhysicalDevice,
    name: String,
    presentation: P
}

impl SupportedDevice<NoPresentation> {
    fn new(physical_device: vk::PhysicalDevice, vk_context: &VkContext) -> SupportedDevice<NoPresentation>{
        let name = unsafe {
            let props =
                vk_context.instance.get_physical_device_properties(physical_device);
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        Self {
            physical_device,
            name,
            presentation: NoPresentation {},
        }
    }

    /// Returns either a device with presentation support or `None`.  If `Some`, then the
    /// presentation queue family index is our preferred index for presentation.
    ///
    /// Usually this is the first index.  In odd (usually old, so we might not ever care) cases, the
    /// only queue family that can present might not have graphics capability.  A transfer is
    /// required and the swapchain must use [`ash::vk::SharingMode::CONCURRENT`].  Support for this
    /// has been filed as DEBT and we just assume the graphics queue supports presentation for now.
    pub fn with_surface_support(
        self,
        surface: vk::SurfaceKHR,
        vk_context: &VkContext,
    ) -> Option<SupportedDevice<HasPresentation>> {
        let surface_loader = vk_context.surface_loader();
        let VkContext { entry, instance } = vk_context;
        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(
                self.physical_device,
            )
        }
            .iter()
            .enumerate()
            .filter_map(|(index, _family)| {
                let index = index as u32;
                let supports_present = unsafe {
                    surface_loader
                        .get_physical_device_surface_support(
                            self.physical_device,
                            index,
                            surface,
                        )
                        .unwrap_or(false)
                };
                supports_present.then_some(index)
            })
            .collect();
        Some(SupportedDevice {
            physical_device: self.physical_device,
            name: self.name,
            presentation: HasPresentation {
                queue_families,
            },
        })
    }
}

impl<P> SupportedDevice<P> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device(&self) -> vk::PhysicalDevice {
        self.physical_device.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn supported_devices () {
        let vk_context = VkContext::new();
        let supported = vk_context.supported_devices();
        println!("Supported devices:");
        for device in supported.iter() {
            println!("  {}", device.name());
        }
    }

    // NEXT Headless tests.  Fake windows.  Something.  Want to check on surface support!
}
