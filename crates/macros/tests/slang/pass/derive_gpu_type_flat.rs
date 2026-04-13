// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_macros::GpuType;
use mutate_vulkan::prelude::*;

#[derive(GpuType)]
#[repr(C)]
struct SlangTypes {
    foo: UInt,
    bar: Float,
}

fn main() {
    type D = Scalar;

    // total size: 8 bytes
    assert_eq!(<SlangTypes as GpuType<D>>::SIZE, 8, "total size");

    // SLANG_NAME round-trips ────────────────────────────────────────────────
    assert_eq!(<SlangTypes as GpuType<D>>::SLANG_NAME, "SlangTypes");

    // Pack byte-exactness ───────────────────────────────────────────────────
    let foo_val: u32 = 0xDEAD_C0DE_u32;
    let bar_val: u32 = (-118.625_f32).to_bits(); // 0xC2ED_4000 🤖

    let v = SlangTypes {
        foo: UInt::from(foo_val),
        bar: Float::from(f32::from_bits(bar_val)),
    };

    let mut buf = [0u8; <SlangTypes as GpuType<D>>::SIZE];
    <SlangTypes as Pack<D>>::pack_into(&v, &mut buf);

    let mut expected = [0u8; 8];
    expected[0..4].copy_from_slice(&foo_val.to_le_bytes());
    expected[4..8].copy_from_slice(&bar_val.to_le_bytes());

    assert_eq!(&buf[0..4], &expected[0..4], "foo region");
    assert_eq!(&buf[4..8], &expected[4..8], "bar region");
}
