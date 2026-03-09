// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile and assert: DeviceAddress has correct layout constants,
// NULL sentinel, and satisfies GpuPod. device_address_newtype! derivatives
// must inherit all of these properties.

use mutate_vulkan::slang::prelude::*;

device_address_newtype!(MeshletBufferPtr, "MeshletBufferPtr");
device_address_newtype!(TransformBufferPtr, "TransformBufferPtr");

fn require_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}
fn require_device_address<T: IsDeviceAddress>() {}

fn main() {
    require_gpu_pod::<ScalarLayout, DeviceAddress>();
    require_gpu_pod::<ScalarLayout, MeshletBufferPtr>();
    require_gpu_pod::<ScalarLayout, TransformBufferPtr>();

    require_device_address::<DeviceAddress>();
    require_device_address::<MeshletBufferPtr>();
    require_device_address::<TransformBufferPtr>();

    // 8 bytes, 8-byte aligned — matches VkDeviceAddress
    assert_eq!(<DeviceAddress as GpuType<ScalarLayout>>::SIZE, 8);
    assert_eq!(<DeviceAddress as GpuType<ScalarLayout>>::ALIGN, 8);
    assert_eq!(<MeshletBufferPtr as GpuType<ScalarLayout>>::SIZE, 8);

    // PRIMITIVE is UInt64 — the raw wire type
    assert_eq!(
        <DeviceAddress as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt64
    );
    assert_eq!(
        <MeshletBufferPtr as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt64
    );

    // SLANG_NAME is the newtype's name, not "uint64_t"
    assert_eq!(
        <MeshletBufferPtr as GpuType<ScalarLayout>>::SLANG_NAME,
        "MeshletBufferPtr"
    );
    assert_eq!(
        <TransformBufferPtr as GpuType<ScalarLayout>>::SLANG_NAME,
        "TransformBufferPtr"
    );

    // NULL sentinel round-trips to raw 0
    assert_eq!(DeviceAddress::NULL.raw(), 0u64);
    assert_eq!(MeshletBufferPtr::NULL.raw(), 0u64);

    // From<u64> and From<UInt64> both exist
    let addr = DeviceAddress::from(0xDEAD_BEEF_u64);
    assert_eq!(addr.raw(), 0xDEAD_BEEF_u64);

    let wrapped = MeshletBufferPtr::from(DeviceAddress::from(42u64));
    assert_eq!(wrapped.raw(), 42u64);
}
