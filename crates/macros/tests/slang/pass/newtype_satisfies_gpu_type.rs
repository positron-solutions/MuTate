// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile: slang_newtype! forwards PRIMITIVE, SIZE, ALIGN from its inner type.

use mutate_vulkan::slang::prelude::*;

slang_newtype!(Hotness, Float32, "Hotness");
slang_newtype!(JointIndex, UInt16, "JointIndex");

fn require_gpu_type<L: LayoutRule, T: GpuType<L>>() {}

fn main() {
    require_gpu_type::<ScalarLayout, Hotness>();
    require_gpu_type::<ScalarLayout, JointIndex>();

    // PRIMITIVE forwards from inner
    assert_eq!(
        <Hotness as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::Float32,
    );
    assert_eq!(
        <JointIndex as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt16,
    );

    // SIZE / ALIGN match the inner scalar
    assert_eq!(<Hotness as GpuType<ScalarLayout>>::SIZE, 4);
    assert_eq!(<Hotness as GpuType<ScalarLayout>>::ALIGN, 4);
    assert_eq!(<JointIndex as GpuType<ScalarLayout>>::SIZE, 2);
    assert_eq!(<JointIndex as GpuType<ScalarLayout>>::ALIGN, 2);

    // SLANG_NAME is the *wrapper* name, not the inner one
    assert_eq!(<Hotness as GpuType<ScalarLayout>>::SLANG_NAME, "Hotness");
    assert_eq!(
        <JointIndex as GpuType<ScalarLayout>>::SLANG_NAME,
        "JointIndex"
    );
}
