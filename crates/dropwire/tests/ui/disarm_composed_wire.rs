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

struct Missileer {
    missile: Missile,
}

impl Missileer {
    fn fire_ze_missiles(self) {
        let Missileer { missile } = self;
        missile.fire(); // !
    }
}

fn main() {
    let missileer = Missileer {
        missile: Missile::new(),
    };
    missileer.fire_ze_missiles();
}

//@check-pass
