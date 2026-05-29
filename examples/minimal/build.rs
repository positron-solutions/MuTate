// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_assets as assets;

fn main() {
    assets::build::set_asset_default_dir();
    assets::build::build_shaders();
}
