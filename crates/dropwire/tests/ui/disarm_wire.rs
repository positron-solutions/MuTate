// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_dropwire::DropWire;

struct Missile {
    wire: DropWire<Missile>,
}

impl Missile {
    fn new() -> Self {
        Self {
            wire: DropWire::armed(),
        }
    }

    fn fire(self) {
        let Self { wire } = self;
        wire.disarm();
    }
}

fn main() {
    let missile = Missile::new();
    missile.fire();
}

//@check-pass
