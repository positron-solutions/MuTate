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

use std::ffi::OsStr;

// NEXT error normalization across the modules.  Build currently does not provide much helpful
// output.
mod prelude {
    pub use super::AssetError;
    pub use super::AssetKind;
}

#[derive(Debug)]
pub enum AssetKind {
    Shader,
    Hash,
}

impl AssetKind {
    fn ext(&self) -> &'static OsStr {
        match self {
            AssetKind::Shader => OsStr::new("spv"),
            AssetKind::Hash => OsStr::new("xx3h"),
        }
    }

    fn subdir(&self) -> &'static OsStr {
        match self {
            AssetKind::Shader => OsStr::new("shaders"),
            // LIES most hash lookups will use find_hash, which accepts an `AssetKind` parameter,
            // meaning this path is basically never expected to be used unless we produce some kind
            // of hash not associated with a specific asset.
            AssetKind::Hash => OsStr::new("hashes"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("read failed: {:?}", .0)]
    ReadError(#[from] std::io::Error),
    #[error("file not found: {:?}", .0)]
    NotFound(String),
    #[error("load spirv failed: {:?}", .0)]
    InvalidShader(String),
}
