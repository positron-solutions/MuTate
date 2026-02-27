// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Assets
//!
//! The assets module encapsulates how lookups can vary across platforms and between usage and
//! development.  `AssetDirs` is a set of realized directories where lookups may search.  Hold onto
//! it for doing many lookups at computer speed but re-initialize it for human-speed queries.
//!
//! ## Precedence Rules
//!
//! - On **debug builds**, things are simple.  We use:
//!
//!   1. `MUTATE_ASSETS_DIR` enabling overrides for any purpose.
//!   2. The source tree's assets folder, below the build time `CARGO_MANIFEST_DIR`.
//!
//! - On **release builds**, life is more complex.  We use:
//!
//!   1. `MUTATE_ASSETS_DIR`
//!   2. The user's home directory
//!   3. A preferred installation direction controlled by `MUTATE_BUILD_ASSETS_DIR` or
//!      `package.metadata.mutate.asset_dir` in the Cargo.toml.
//!   4. The expected system directory as a backup.
//!
//! When set, `MUTATE_ASSETS_DIR` and `MUTATE_BUILD_ASSETS_DIR` should point directly to an assets
//! root i.e. a folder containing a shaders directory.

use std::ffi::OsStr;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use crate::prelude::*;

/// Pre-calculated and checked parent paths for reuse in asset look-ups.  Hold onto this object for
/// the duration that such paths are valid, such as when listing assets or looking up several
/// assets.
pub struct AssetDirs {
    search_paths: Vec<PathBuf>,
}

/// The build script will extract this from the toml and propagate through this variable, or you can
/// set it by hand.  This sets the "default" directory, which is only used for release builds and is
/// intended for packaging.
const DEFAULT_ASSET_DIR: Option<&str> = option_env!("MUTATE_BUILD_ASSETS_DIR");

impl AssetDirs {
    /// Checks asset search directories once on construction.
    pub fn new() -> Self {
        let mut search_paths = Vec::with_capacity(4);

        let as_assets_root = |p: PathBuf| -> Option<PathBuf> {
            p.canonicalize().ok().filter(|p| p.exists() && p.is_dir())
        };

        // Treat the given path as a parent containing an `assets/` subdir.
        let with_assets_subdir = |p: PathBuf| {
            let mut p = p;
            p.push("assets");
            as_assets_root(p)
        };

        // Always highest priority: explicit override.
        if let Ok(raw) = std::env::var("MUTATE_ASSETS_DIR") {
            match as_assets_root(PathBuf::from(&raw)) {
                Some(path) => search_paths.push(path),
                None => eprintln!(
                    "warning: invalid MUTATE_ASSETS_DIR (path not found): {}",
                    raw
                ),
            }
        }

        // NOTE the macro-time feature is equivalent to debug behavior, reading the environment to
        // allow situational overriding but otherwise only using source directory assets even when
        // building for release.
        if cfg!(any(debug_assertions, feature = "macro-time")) {
            std::env::var("CARGO_MANIFEST_DIR")
                .ok()
                .map(PathBuf::from)
                .and_then(with_assets_subdir)
                .into_iter()
                .for_each(|p| search_paths.push(p));
        } else {
            // Release: user home/local data dir.
            dirs::data_local_dir()
                .and_then(with_assets_subdir)
                .into_iter()
                .for_each(|p| search_paths.push(p));

            // Preferred installation directory (build-time).
            DEFAULT_ASSET_DIR
                .map(PathBuf::from)
                .and_then(as_assets_root)
                .into_iter()
                .for_each(|p| search_paths.push(p));

            // Fallback system directory.
            dirs::data_dir()
                .and_then(with_assets_subdir)
                .into_iter()
                .for_each(|p| search_paths.push(p));
        }

        AssetDirs { search_paths }
    }

    #[cfg(debug_assertions)]
    /// Checks asset paths for `name`.  Debug builds only look for build tree assets.  Use
    /// environment variables to override.
    pub fn find(&self, name: &str, kind: AssetKind) -> Option<PathBuf> {
        let mut file = PathBuf::from(kind.subdir()).join(name);
        file.set_extension(kind.ext());

        let checked: Vec<PathBuf> = self
            .search_paths
            .iter()
            .map(|root| root.join(&file))
            .collect();

        if let Some(found) = checked.iter().find(|candidate| candidate.exists()) {
            return Some(found.clone());
        } else {
            // DEBT tracing
            eprintln!("warning: {kind:?} {name:} not found.");
            checked.iter().for_each(|pb| {
                eprintln!("  checked: {pb:?}");
            });
            None
        }
    }

    #[cfg(not(debug_assertions))]
    /// Checks asset paths for `name`.  Debug builds only look for build tree assets.  Use
    /// environment variables to override.
    pub fn find(&self, name: &str, kind: AssetKind) -> Option<PathBuf> {
        let mut file = PathBuf::from(kind.subdir()).join(name);
        file.set_extension(kind.ext());

        self.search_paths
            .iter()
            .map(|root| root.join(&file))
            .find(|candidate| candidate.exists())
    }

    pub fn find_bytes(&self, name: &str, kind: AssetKind) -> Result<Vec<u8>, AssetError> {
        self.find(name, kind)
            .ok_or(AssetError::NotFound(name.to_owned()))
            .and_then(|found| std::fs::read(found).map_err(|e| e.into()))
    }

    pub fn find_shader(&self, name: &str) -> Result<Vec<u32>, AssetError> {
        let path = self
            .find(name, AssetKind::Shader)
            .ok_or_else(|| AssetError::NotFound(name.to_owned()))?;

        let mut file = std::fs::File::open(&path)?;

        let byte_len = file.metadata()?.len() as usize;

        if byte_len % size_of::<u32>() != 0 {
            return Err(AssetError::InvalidShader(format!(
                "SPIR-V length not multiple of 4: {} bytes",
                byte_len
            )));
        }

        let word_len = byte_len / size_of::<u32>();
        let mut words = Vec::<u32>::with_capacity(word_len);

        // Treating things as just things, as things were meant to be!  🤗
        unsafe {
            words.set_len(word_len);
            let byte_slice =
                std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, byte_len);
            file.read_exact(byte_slice)?;
        }
        Ok(words)
    }

    pub fn find_hash(&self, name: &str, kind: AssetKind) -> Option<PathBuf> {
        let mut path = self.find(name, kind)?;

        path.set_extension(AssetKind::Hash.ext());

        if path.exists() {
            Some(path)
        } else {
            // DEBT tracing
            if cfg!(debug_assertions) {
                eprintln!("warning: {:?} {name} not found.", AssetKind::Hash);
                eprintln!("  checked: {path:?}");
            }
            None
        }
    }
}
