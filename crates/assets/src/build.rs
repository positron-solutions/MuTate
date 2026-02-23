// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Build Support
//!
//! This module contains the build time functionality.

use std::{ffi, fs, path::Path, process};

/// Use slangc to recursively compile shaders from shaders to assets/shaders.
// NEXT emit metadata from slangc
pub fn build_shaders() {
    let manifest_dir = &std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_root = Path::new(manifest_dir);

    let src_root = crate_root.join("shaders");
    let dest_root = crate_root.join("assets/shaders");

    println!("cargo:rerun-if-changed=shaders");

    if process::Command::new("slangc").arg("-v").output().is_err() {
        panic!("no slangc found");
    }

    fn compile_dir(dir: &Path, src_root: &Path, dest_root: &Path, ext: &ffi::OsStr) {
        let mut out_ensured = false;
        let out_dir = dest_root.join(dir.strip_prefix(src_root).unwrap());

        // NEXT for Apple to use pre-compiled Metal libs instead of runtime translated (once) Spirv,
        // emit the MSL targets and call the Apple tooling.  During introspection, proc macros can
        // determine the layout differences from introspection and convert declarations to the
        // appropriate code for each pipeline.
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                compile_dir(&path, &src_root, &dest_root, &ext);
            } else if path.extension() == Some(ext) {
                if !out_ensured {
                    fs::create_dir_all(&out_dir).unwrap();
                    out_ensured = true;
                }
                let stem = path.strip_prefix(src_root).unwrap();
                let out = dest_root.join(stem).with_extension("spv");
                let mut out_reflect = dest_root.join(stem).with_extension("reflection.json");

                // Run slangc:
                // `slangc <input> -o <out> -reflection-json -o <out-reflect>`
                let status = process::Command::new("slangc")
                    .arg(path.as_os_str())
                    .arg("-o")
                    .arg(out)
                    .arg("-reflection-json")
                    .arg(out_reflect)
                    .status()
                    .unwrap();

                if !status.success() {
                    panic!("slangc failed for {:?}", path);
                }
            }
        }
    }

    if src_root.exists() && fs::read_dir(&src_root).unwrap().next().is_some() {
        fs::create_dir_all(&dest_root).unwrap();
    }

    let slang_ext = ffi::OsStr::new("slang");
    if src_root.exists() {
        compile_dir(&src_root, &src_root, &dest_root, &slang_ext);
    }
}

///  Sets the path for hard coding into the binary for use at runtime by the assets module.
// Packagers, see the Cargo.toml for the visualizer.
pub fn set_asset_default_dir() {
    let manifest_dir = &std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_root = Path::new(manifest_dir);

    let manifest = crate_root.join("Cargo.toml");
    dbg!(&manifest);
    let cargo = fs::read_to_string(&manifest).unwrap();
    let parsed: toml::Value = toml::from_str(&cargo).unwrap();

    let mutate_build_assets_dir = parsed["package"]["metadata"]["mutate"]["asset_dir"]
        .as_str()
        .unwrap_or("assets");
    println!("cargo:rustc-env=MUTATE_BUILD_ASSETS_DIR={mutate_build_assets_dir}");
}
