// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Present
//!
//! Presentation tends to live slightly outside of command recording and has a bit specific
//! synchronization requirements.  This module encapsulates the pre and post-render integration with
//! surrounding optional bits such as windows.

pub mod surface;
pub mod swapchain;
