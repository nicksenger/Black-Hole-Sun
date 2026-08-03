//! Animal definitions for black-hole-flux.

use jungle_sdk::prelude::*;
use jungle_zoo::Noop;

use crate::cell::CellState;
use crate::{Nucleus, Primordium};

/// The Progenitor: the first and simplest cell — a bare quark-inference loop
/// with no input/output processing and no metadata.
pub struct Progenitor;

#[jungle::animal(id = 0, generation = 0)]
impl Animal for Progenitor {
    type State = CellState;
    type Seed = black_hole_spec::ObjectId;
    type Flow = Primordium;
}
