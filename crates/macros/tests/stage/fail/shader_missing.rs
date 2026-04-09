// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_macros::shader;

#[shader("test/does_not_exist", COMPUTE, c"main")]
struct BadStage {}

fn main() {}
