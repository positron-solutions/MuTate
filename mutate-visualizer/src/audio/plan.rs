// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plan
//!
//! Currently just some tools for getting the control data sizes and offsets figured out.

// NEXT we will need a whole lot more control data writing, and making it easier is a huge time
// saver.

/// Bump cursor for computing `u32` offsets from a base address.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cursor(u32);

impl Cursor {
    /// Reserve `count` contiguous `T`, returning the offset of element zero.
    pub fn push<T>(&mut self, count: u32) -> u32 {
        self.push_bytes(size_of::<T>() as u32 * count, align_of::<T>() as u32)
    }

    /// Reserve raw bytes.  `align` must be a power of two.
    pub fn push_bytes(&mut self, size: u32, align: u32) -> u32 {
        debug_assert!(align.is_power_of_two());
        let offset = self.0.next_multiple_of(align);
        self.0 = offset
            .checked_add(size)
            .expect("layout exceeds u32 offsets");
        offset
    }

    /// Advance alignment only
    pub fn align_to(&mut self, align: u32) -> u32 {
        self.push_bytes(0, align)
    }

    pub fn len(self) -> u32 {
        self.0
    }
}

/// Write `T`
pub fn put<T>(bytes: &mut [u8], offset: u32, value: T) {
    let offset = offset as usize;
    assert!(offset + size_of::<T>() <= bytes.len());
    debug_assert_eq!(offset % align_of::<T>(), 0);
    unsafe { bytes.as_mut_ptr().add(offset).cast::<T>().write(value) };
}

/// Write `&[T]`
pub fn put_slice<T: Copy>(bytes: &mut [u8], offset: u32, values: &[T]) {
    let offset = offset as usize;
    assert!(offset + size_of_val(values) <= bytes.len());
    debug_assert_eq!(offset % align_of::<T>(), 0);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            bytes.as_mut_ptr().add(offset).cast::<T>(),
            values.len(),
        )
    };
}

/// Greatest common divisor. `gcd(0, 0) == 0`.
pub const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
