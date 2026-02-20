// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Assets
//!
//! Many assets need pre-processing steps.  Put them here.  We will load them via the assets
//! interfaces, sometimes assisted with macros to enable checks based on asset metadata that allows
//! the compiler to fail early if we are not using those assets correctly.
//!
//! The build support and runtime asset loading are both feature gated just to help keep compile
//! time down.  The build feature enables writing short build scrips.  The runtime feature enables
//! loading the assets.  Use the runtime feature in dependencies but use the runtime feature in your
//! normal dependencies.

#[cfg(feature = "runtime")]
pub mod assets;
#[cfg(feature = "build")]
pub mod build;

#[cfg(feature = "runtime")]
pub use assets::*;
