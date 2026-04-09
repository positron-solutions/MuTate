// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_assets as assets;

fn main() {
    // We need to override the environment for try-build tests, configuring the MUTATE_ASSETS_DIR to
    // always point to the source directory assets.  This will be inherited by the try-build
    // sub-processes, allowing those synthetic test crates to look back at the assets defined in
    // this crate.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-env=MUTATE_ASSETS_DIR={}/assets", manifest);
    println!("cargo:rerun-if-env-changed=CARGO_MANIFEST_DIR");

    assets::build::set_asset_default_dir();
    assets::build::build_shaders();
}
