// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi, fs, path::Path, process};

fn main() {
    set_asset_default_dir();
    build_shaders();
}

fn build_shaders() {
    let src_root = Path::new("shaders");
    let dest_root = Path::new("assets/shaders");

    println!("cargo:rerun-if-changed=shaders");

    if process::Command::new("slangc").arg("-v").output().is_err() {
        panic!("no slangc found");
    }

    fn compile_dir(dir: &Path, src_root: &Path, dest_root: &Path, ext: &ffi::OsStr) {
        let mut out_ensured = false;
        let out_dir = dest_root.join(dir.strip_prefix(src_root).unwrap());

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
                let out = dest_root
                    .join(path.strip_prefix(src_root).unwrap())
                    .with_extension("spv");

                // Run slangc: `slangc <input> -o <output>`
                let status = process::Command::new("slangc")
                    .arg(path.as_os_str())
                    .arg("-o")
                    .arg(out.as_os_str())
                    .status()
                    .unwrap();

                if !status.success() {
                    panic!("slangc failed for {:?}", path);
                }
            }
        }
    }

    let slang_ext = ffi::OsStr::new("slang");
    compile_dir(&src_root, &src_root, &dest_root, &slang_ext);
}

/// Packagers, see the Cargo.toml.  Sets the path for hard coding into the binary for use at
/// runtime.  Does not affect build time lookups.
fn set_asset_default_dir() {
    let cargo = fs::read_to_string("Cargo.toml").unwrap();
    let parsed: toml::Value = toml::from_str(&cargo).unwrap();

    let mutate_build_assets_dir = parsed["package"]["metadata"]["mutate"]["asset_dir"]
        .as_str()
        .unwrap_or("assets");
    println!("cargo:rustc-env=MUTATE_BUILD_ASSETS_DIR={mutate_build_assets_dir}");
}
