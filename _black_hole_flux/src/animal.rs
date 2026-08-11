//! Animal definitions for black-hole-flux.

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;

use crate::cell::CellState;
use crate::Primordium;

/// The Progenitor: the first and simplest cell — a bare quark-inference loop
/// with no input/output processing and no metadata.
#[derive(Clone)]
pub struct Progenitor;

#[jungle::animal(id = 0, generation = 0)]
impl Animal for Progenitor {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = Primordium;
}
