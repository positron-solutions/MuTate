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
            wire: DropWire::armed(),
        }
    }
}

struct Bomber {
    bomb: Bomb,
}

fn main() {
    let _bomber = Bomber { bomb: Bomb::new() };
}
//@error-in-other-file: evaluation panicked
