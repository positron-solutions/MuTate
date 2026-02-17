// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spectrogram
//!
//! *Well, that's... why I'm here! - Larry Kenobi*
//!
//! A spectrogram is a moving spectrograph.  This module covers the description of a filter bank so
//! that it may be implemented in GPU logic.

/// Width of a 4k monitor
pub const RESOLUTION_4K_WIDTH: usize = 3840;
/// Height of a 4k monitor
pub const RESOLUTION_4K_HEIGHT: usize = 2160;
