// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_macros::shader;

#[shader("test/hello_compute", COMPUTE, c"main")]
struct GoodStage {}

fn main() {}
