// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Audio Nodes
//!
//! Audio first must be acquired, using the `crate::node::audio::raw::RawAudio` node.  As the graph
//! interfaces evolve, so must these nodes and their types.

// pub mod cqt;
pub mod colors;
// pub mod iso226;
pub mod kweight;
pub mod raw;
pub mod rms;
