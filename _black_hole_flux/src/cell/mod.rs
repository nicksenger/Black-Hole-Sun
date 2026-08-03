//! Cell module - higher-order flows for cell training loops.

pub mod action;
pub mod effect;

pub use action::CellState;

use action::{
    Optimize as Optimize_, PerturbDown as PerturbDown_, PerturbUp as PerturbUp_,
    Transmit as Transmit_, WaitForPotentiationAction as WaitForPotentiationAction_,
    WaitForPropagationAction as WaitForPropagationAction_,
};
use black_hole_spec::EmissionId;
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use jungle_zoo::Noop;

use crate::Nucleus;

/// A Cell wraps a nucleus flow in an infinite QuZO training loop driven by
/// [`Transmission`](black_hole_spec::Transmission) messages from void.
#[derive(Flow)]
pub struct Cell<N>(While<Always<CellState, ()>, Cytoplasm<N>>);

/// The body of one iteration of a [`Cell`] loop.
#[derive(Flow)]
pub struct Cytoplasm<N>(
    Step<PerturbUp_>,
    Step<WaitForPropagationAction_>,
    N,
    Step<Transmit_>,
    Step<PerturbDown_>,
    Step<WaitForPropagationAction_>,
    N,
    Step<Transmit_>,
    Step<WaitForPotentiationAction_>,
    Step<Optimize_>,
);

/// A eukaryotic cell: a [`Cell`] with an arbitrarily complex nucleus.
pub type Eukaryote<In, Out, M> = Cell<Nucleus<In, Out, M>>;

/// A prokaryotic cell: a [`Cell`] whose nucleus has no input/output processing.
pub type Prokaryote<M> =
    Cell<Nucleus<Step<Noop<CellState, EmissionId>>, Step<Noop<CellState, EmissionId>>, M>>;

/// A primordial cell: the simplest possible [`Cell`] with no input/output
/// processing and no metadata.
pub type Primordium =
    Cell<Nucleus<Step<Noop<CellState, EmissionId>>, Step<Noop<CellState, EmissionId>>, ()>>;
