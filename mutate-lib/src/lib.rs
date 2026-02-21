// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # MuTate
//!
//! Core µTate audio and video recognition & transformation capabilities. Alternative frontends
//! and applications may be interested in obtaining raw inputs to drive behaviors besides
//! visualization.  This crate is kept separate so that µTate behaviors can be embedded directly
//! into 3rd party applications without the need to run a separate daemon.
//!
//! ## Workbench
//!
//! This crate also contains the engineering support to design and hardcode filter banks, which is
//! behind the **workbench** feature.  See the workbench binary and most of its functionality,
//! within the dsp module.
// XXX Re-deNY
#![allow(dead_code)]
#![allow(unused)]

// You need to break up the audio module into per-platform modules and implement AudioContext.  Only
// Linux via pipewire is supported right now.  You will support it.  Welcome to open source
// development.
#[cfg(target_os = "linux")]
pub mod audio;

pub mod context;
#[cfg(feature = "dsp")]
pub mod dsp;
#[cfg(target_os = "linux")]
use pipewire as pw;

use mutate_assets as assets;

pub mod prelude {
    pub use crate::MutateError;
    // NEXT feature flag the Vulkan stuff in one crate
    pub use crate::context::VkContext;
}

// NEXT Audio will be its own kind of error that must fit into the MutateError hierarchy.
#[derive(thiserror::Error, Debug)]
pub enum MutateError {
    #[cfg(target_os = "linux")]
    #[error("Pipewire: {0}")]
    Pipewire(#[from] pw::Error),
    #[error("thread poisoned")]
    Poison,

    #[error("audio source error: {0}")]
    AudioSource(String),
    #[error("cannot use dropped audio connection")]
    Dropped,

    #[error("audio connection error: {0}")]
    AudioConnect(&'static str),
    #[error("audio thread termination error")]
    AudioTerminate,

    #[error("Timeout: {0}")]
    Timeout(&'static str),

    #[error("Vulkan: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("AssetError: {0}")]
    AssetError(#[from] assets::AssetError),
}

impl<T> From<std::sync::PoisonError<T>> for MutateError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        MutateError::Poison
    }
}
