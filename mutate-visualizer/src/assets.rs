// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The assets module encapsulates how lookups can vary across platforms and between usage and
//! development.  `AssetDirs` is a set of realized directories where lookups may search.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

pub enum AssetKind {
    Shader,
}

impl AssetKind {
    fn ext(&self) -> &'static OsStr {
        match self {
            AssetKind::Shader => OsStr::new("spv"),
        }
    }

    fn subdir(&self) -> &'static OsStr {
        match self {
            AssetKind::Shader => OsStr::new("shaders"),
        }
    }
}

/// Pre-calculated and checked parent paths for reuse in asset look-ups.  Hold onto this object for
/// the duration that such paths are valid, such as when listing assets or looking up several assets.
pub struct AssetDirs {
    env: Option<PathBuf>,
    #[allow(dead_code)]
    local: Option<PathBuf>,
    #[allow(dead_code)]
    default: Option<PathBuf>,
}

const DEFAULT_ASSET_DIR: Option<&str> = option_env!("MUTATE_ASSETS_DIR");

impl AssetDirs {
    /// Checks asset search directories once on construction.
    pub fn new() -> Self {
        let env = std::env::var("MUTATE_ASSETS_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|path| path.exists());

        let local = dirs::data_local_dir()
            .map(|p| p.join("mutate"))
            .filter(|p| p.exists());

        let system = dirs::data_dir().map(|p| p.join("mutate"));
        let hardcode = DEFAULT_ASSET_DIR.map(|d| PathBuf::from(d));
        let system = [hardcode, system]
            .into_iter()
            .flatten()
            .find(|p| p.exists());

        AssetDirs {
            env,
            local,
            default: system,
        }
    }

    #[cfg(debug_assertions)]
    /// Checks asset paths for `name`.  Debug builds only look for build tree assets.  Use
    /// environment variables to override.
    pub fn find(&self, name: &str, kind: AssetKind) -> Option<PathBuf> {
        let mut file = PathBuf::from(kind.subdir()).join(name);
        file.set_extension(kind.ext());
        let crate_assets = Some(Path::new(std::env!("CARGO_MANIFEST_DIR")).join("assets"));
        [self.env.as_deref(), crate_assets.as_deref()]
            .into_iter()
            .flatten()
            .map(|root| root.join(&file))
            .find(|candidate| candidate.exists())
    }

    #[cfg(not(debug_assertions))]
    pub fn find(&self, name: &str, kind: AssetKind) -> Option<PathBuf> {
        let mut file = PathBuf::from(kind.subdir()).join(name);
        file.set_extension(kind.ext());

        [&self.env, &self.local, &self.default]
            .into_iter()
            .flatten()
            .map(|root| root.join(&file))
            .find(|candidate| candidate.exists())
    }

    pub fn find_bytes(&self, name: &str, kind: AssetKind) -> Result<Vec<u8>, AssetError> {
        self.find(name, kind)
            .ok_or(AssetError::NotFound(name.to_owned()))
            .and_then(|found| std::fs::read(found).map_err(|e| e.into()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("read failed: {:?}", .0)]
    ReadError(#[from] std::io::Error),
    #[error("file not found: {:?}", .0)]
    NotFound(String),
}
