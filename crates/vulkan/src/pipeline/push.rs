// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Push Constants
//!
//! Input data directly embedded into the command buffers.  This data controls the reading of all
//! other data.
//!
//! ## Trading Size for Indirection
//!
//! Push constants have a limited size (only 128 bytes guaranteed).  If we lack sufficient space
//! within that 128 bytes, we must move some data into SSBOs or UBOs and point to that data.  This
//! is done by indexing into descriptor sets with 4 bytes.  If there are a lot of instances of this
//! control data, it might be more convenient to allocate an inner array and then use 4 bytes of
//! descriptor index alongside 4 bytes to index into the array.
//!
//! **In short, we use either 4 bytes or 8 bytes of data per control data structure**, meaning every
//! stage of a pipeline can use a separate control data structure while still affording us enough
//! room for indexes.  Even if every stage of a pipeline requires a separate control data
//! pointer,there is plenty of room.  Such structs can be likely re-used between stages and will
//! lead to natural cache read efficiency. 😌
//!
//! ## Scalar Layouts
//!
//! Push constants have just a bit of alignment and size requirements.  We enabled 8 bit and 16 bit
//! scalars and scalar the block storage feature, giving us quite a bit more flexibility.  The fact
//! remains though that push constants might need to be written to a layout that doesn't quite match
//! the naive struct layout.  We handle the generation using a proc macro.
//!
//! ## Indirect Type Agreement
//!
//! LIES below may still be prospective, but it is planned.
//!
//! All of our SSBO, Image, and UBO types have runtime typed handles. That's 32bit pointer with a
//! descriptor type and an item layout.  The build time check only ensures that pipelines and
//! shaders are in agreement, meaning we are writing enough handles and the contents of those
//! handles align with what the shader expects at a type level.  When mixing visuals, runtime
//! decisions are inevitable.  Buffers may be compatible by contained type and semantic category
//! rather than a specific names.  This allows us to let machine learning participate in the
//! modulation.

//! # Push Constants
//!
//! Every pipeline that implements PushConstants will have an array of bytes for PushConstants.
//! These are typed to assist reduction in silly errors.
//!
//! All PushConstants can be accessed raw or as ranges.

// pub struct PushConstantBuffer([u8; 128]);

// impl<const B: usize> PushConstantBuffer<B> {
//     pub fn range<const O: usize, const N: usize>(&self) -> PushConstantRange<O, N> {
//         // Compile-time check
//         const _: () = assert!(
//             O + N <= B,
//             "Requested PushConstantRange<{O},{N}> overflows max push constant bytes: {B}"
//         );
//         PushConstantRange {
//             data: &self.data[O..O + N],
//         }
//     }

//     pub fn range_mut<const O: usize, const N: usize>(&mut self) -> PushConstantRangeMut<O, N> {
//         const _: () = assert!(
//             O + N <= B,
//             "Requested PushConstantRange<{O},{N}> overflows max push constant bytes: {B}"
//         );
//         PushConstantRangeMut {
//             data: &mut self.data[O..O + N],
//         }
//     }
// }

// /// A view into a PushConstantBuffer at a given offset and length.
// /// - `O`: offset into the buffer
// /// - `N`: number of bytes in the range
// pub struct PushConstantRange<'a, const O: usize, const N: usize> {
//     data: &'a [u8],
// }

// pub struct PushConstantRangeMut<'a, const O: usize, const N: usize> {
//     data: &'a mut [u8],
// }

// impl<'a, const O: usize, const N: usize> PushConstantRange<'a, O, N> {
//     /// Returns a fixed-size slice for this range
//     pub fn as_slice(&self) -> &[u8; N] {
//         self.data.try_into().expect("slice length mismatch")
//     }
// }

// impl<'a, const O: usize, const N: usize> PushConstantRangeMut<'a, O, N> {
//     pub fn as_slice(&self) -> &[u8; N] {
//         self.data.try_into().expect("slice length mismatch")
//     }

//     pub fn as_mut_slice(&mut self) -> &mut [u8; N] {
//         self.data.try_into().expect("slice length mismatch")
//     }
// }
