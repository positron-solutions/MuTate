// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused, dead_code)]
use mutate_dropwire::DropWire;

struct Bomb {
    wire: DropWire<Bomb>,
}

impl Bomb {
    fn new() -> Self {
        Self {
            wire: DropWire::ARMED,
        }
    }
}

fn main() {
    let _b = Bomb::new();
}
//@error-in-other-file: evaluation panicked
