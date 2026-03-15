// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Resource
//!

// This module began to grow into a fully fledged async resource creation system.  That work has
// been put off to allow more concrete code to drive the development.  What was learned is that we
// really, really want late binding.  That will make every streaming, shared ownership, compaction
// problem so much easier.  Until then, we will focus on making the highly manual bits less manual.

pub mod buffer;
pub mod image;
pub mod shader;
pub mod ubo;

#[cfg(test)]
mod test {}
