//! Cell module - higher-order flows for cell training loops.

pub mod action;
pub mod effect;

pub use action::CellState;

use action::{
    GenerateModelId as GenerateModelId_, InitRecvId as InitRecvId_, Optimize as Optimize_,
    PerturbDown as PerturbDown_, PerturbUp as PerturbUp_, PrepareAtomInput as PrepareAtomInput_,
    StartModel as StartModel_, Transmit as Transmit_,
    WaitForPotentiationAction as WaitForPotentiationAction_,
    WaitForPropagationAction as WaitForPropagationAction_,
};
use black_hole_spec::EmissionId;
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use jungle_zoo::Noop;
use uuid::Uuid;

use crate::Atom;

/// A Cell wraps a atom flow in an infinite QuZO training loop driven by
/// [`Transmission`](black_hole_spec::Transmission) messages from void.
#[derive(Flow)]
pub struct CellWithState<N, S>(
    Step<InitRecvId_<S>>,
    Step<GenerateModelId_<S>>,
    Step<StartModel_<S>>,
    While<Always<CellState<S>, ()>, CytoplasmWithState<N, S>>,
);

pub type Cell<N, S = ()> = CellWithState<N, S>;

/// The body of one iteration of a [`Cell`] loop.
#[derive(Flow)]
pub struct CytoplasmWithState<N, S>(
    Step<PerturbUp_<S>>,
    Step<WaitForPropagationAction_<S>>,
    Step<PrepareAtomInput_<S>>,
    N,
    Step<Transmit_<S>>,
    Step<PerturbDown_<S>>,
    Step<WaitForPropagationAction_<S>>,
    Step<PrepareAtomInput_<S>>,
    N,
    Step<Transmit_<S>>,
    Step<WaitForPotentiationAction_<S>>,
    Step<Optimize_<S>>,
);

pub type Cytoplasm<N, S = ()> = CytoplasmWithState<N, S>;

/// A eukaryotic cell: a [`Cell`] with an arbitrarily complex atom.
pub type Eukaryote<In, Out, M, S = ()> = Cell<Atom<In, Out, M, S>, S>;

/// A prokaryotic cell: a [`Cell`] whose atom has no input/output processing.
pub type Prokaryote<M, S = ()> = Cell<
    Atom<Step<Noop<CellState<S>, (Uuid, EmissionId)>>, Step<Noop<CellState<S>, EmissionId>>, M, S>,
    S,
>;

/// A primordial cell: the simplest possible [`Cell`] with no input/output
/// processing and no metadata.
pub type Primordium<S = ()> = Cell<
    Atom<Step<Noop<CellState<S>, (Uuid, EmissionId)>>, Step<Noop<CellState<S>, EmissionId>>, (), S>,
    S,
>;
