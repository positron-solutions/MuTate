// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_lib::prelude::*;

#[stage("test/hello_compute", Compute, c"main")]
struct GoodStage {}

fn main() {}
