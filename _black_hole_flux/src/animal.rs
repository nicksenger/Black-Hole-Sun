//! Animal definitions for black-hole-flux.

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;

use crate::cell::CellState;
use crate::Primordium;
use crate::Ray;

/// The Progenitor: the first and simplest cell — a bare quark-inference loop
/// with no input/output processing and no metadata.
#[derive(Clone)]
pub struct Progenitor;

#[jungle::animal(observe, id = 0, generation = 0)]
impl Animal for Progenitor {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = Primordium;
}

impl Observe for Progenitor {
    type Appearance = Ray;

    fn observe(state: &Self::State) -> Self::Appearance {
        Ray {
            frozen: state.is_frozen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progenitor_observe_reports_cell_frozen_state() {
        let mut state = CellState::default();
        state.is_frozen = true;
        assert_eq!(Progenitor::observe(&state), Ray { frozen: true });
    }
}
