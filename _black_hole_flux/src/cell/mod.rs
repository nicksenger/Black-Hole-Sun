//! Cell module - higher-order flows for cell training loops.

pub mod action;
pub mod effect;

pub use action::CellState;

use action::{
    AdvanceGradientStep as AdvanceGradientStep_,
    BeginGradientAccumulation as BeginGradientAccumulation_, GenerateModelId as GenerateModelId_,
    InitRecvId as InitRecvId_, Optimize as Optimize_, PerturbDown as PerturbDown_,
    PerturbUp as PerturbUp_, PrepareAtomInput as PrepareAtomInput_, Potentiation,
    StartModel as StartModel_, Transmit as Transmit_,
    WaitForPotentiationAction as WaitForPotentiationAction_,
    WaitForPropagationAction as WaitForPropagationAction_,
};
use black_hole_spec::EmissionId;
use jungle_sdk::prelude::*;
use jungle_zoo::backoff::Backoff;
use jungle_zoo::predicate::Always;
use jungle_zoo::Noop;
use uuid::Uuid;

use crate::model_config::{DefaultConfig, ModelConfig};
use crate::Atom;

/// Predicate that keeps running cell microsteps until `grad_steps` is reached.
pub struct HasPendingGradientStep<S>(std::marker::PhantomData<fn() -> S>);

impl<S> Predicate<(&CellState<S>, &())> for HasPendingGradientStep<S> {
    fn eval((state, _): &(&CellState<S>, &())) -> bool {
        state.grad_step < state.grad_steps.max(1)
    }
}

const CELL_STEP_BACKOFF_INITIAL_DELAY_MS: u64 = 100;
const CELL_STEP_BACKOFF_MAX_DELAY_MS: u64 = 10_000;
const CELL_STEP_BACKOFF_MULTIPLIER: u8 = 2;

/// Runs one model-start step with automatic retry backoff.
#[derive(Flow)]
pub struct StartModelWithBackoff<S, H: ModelConfig>(
    Backoff<
        CellState<S>,
        Uuid,
        (),
        Step<StartModel_<S, H>>,
        CELL_STEP_BACKOFF_INITIAL_DELAY_MS,
        CELL_STEP_BACKOFF_MAX_DELAY_MS,
        CELL_STEP_BACKOFF_MULTIPLIER,
    >,
);

/// Runs one perturb-up step with automatic retry backoff.
#[derive(Flow)]
pub struct PerturbUpWithBackoff<S>(
    Backoff<
        CellState<S>,
        (),
        (),
        Step<PerturbUp_<S>>,
        CELL_STEP_BACKOFF_INITIAL_DELAY_MS,
        CELL_STEP_BACKOFF_MAX_DELAY_MS,
        CELL_STEP_BACKOFF_MULTIPLIER,
    >,
);

/// Runs one perturb-down step with automatic retry backoff.
#[derive(Flow)]
pub struct PerturbDownWithBackoff<S>(
    Backoff<
        CellState<S>,
        (),
        (),
        Step<PerturbDown_<S>>,
        CELL_STEP_BACKOFF_INITIAL_DELAY_MS,
        CELL_STEP_BACKOFF_MAX_DELAY_MS,
        CELL_STEP_BACKOFF_MULTIPLIER,
    >,
);

/// Runs one optimize step with automatic retry backoff.
#[derive(Flow)]
pub struct OptimizeWithBackoff<S>(
    Backoff<
        CellState<S>,
        Potentiation,
        (),
        Step<Optimize_<S>>,
        CELL_STEP_BACKOFF_INITIAL_DELAY_MS,
        CELL_STEP_BACKOFF_MAX_DELAY_MS,
        CELL_STEP_BACKOFF_MULTIPLIER,
    >,
);

/// A Cell wraps a atom flow in an infinite QuZO training loop driven by
/// [`Transmission`](black_hole_spec::Transmission) messages from void.
#[derive(Flow)]
pub struct CellWithState<N, S, H: ModelConfig>(
    Step<InitRecvId_<S>>,
    Step<GenerateModelId_<S>>,
    StartModelWithBackoff<S, H>,
    While<Always<CellState<S>, ()>, CytoplasmWithState<N, S>>,
);

pub type Cell<N, S = (), H = DefaultConfig> = CellWithState<N, S, H>;

/// The body of one iteration of a [`Cell`] loop.
#[derive(Flow)]
pub struct CytoplasmWithState<N, S>(
    Step<BeginGradientAccumulation_<S>>,
    PerturbUpWithBackoff<S>,
    While<HasPendingGradientStep<S>, CytoplasmPropagationMicrostepWithState<N, S>>,
    Step<BeginGradientAccumulation_<S>>,
    PerturbDownWithBackoff<S>,
    While<HasPendingGradientStep<S>, CytoplasmPropagationMicrostepWithState<N, S>>,
    Step<WaitForPotentiationAction_<S>>,
    OptimizeWithBackoff<S>,
);

/// One propagation/infer/transmit microstep.
#[derive(Flow)]
pub struct CytoplasmPropagationMicrostepWithState<N, S>(
    Step<WaitForPropagationAction_<S>>,
    Step<PrepareAtomInput_<S>>,
    N,
    Step<Transmit_<S>>,
    Step<AdvanceGradientStep_<S>>,
);

pub type Cytoplasm<N, S = ()> = CytoplasmWithState<N, S>;

/// A eukaryotic cell: a [`Cell`] with an arbitrarily complex atom.
pub type Eukaryote<In, Out, M, S = (), H = DefaultConfig> = Cell<Atom<In, Out, M, S, H>, S, H>;

/// A prokaryotic cell: a [`Cell`] whose atom has no input/output processing.
pub type Prokaryote<M, S = (), H = DefaultConfig> = Cell<
    Atom<
        Step<Noop<CellState<S>, (Uuid, EmissionId)>>,
        Step<Noop<CellState<S>, EmissionId>>,
        M,
        S,
        H,
    >,
    S,
    H,
>;

/// A primordial cell: the simplest possible [`Cell`] with no input/output
/// processing and no metadata.
pub type Primordium<S = (), H = DefaultConfig> = Cell<
    Atom<
        Step<Noop<CellState<S>, (Uuid, EmissionId)>>,
        Step<Noop<CellState<S>, EmissionId>>,
        (),
        S,
        H,
    >,
    S,
    H,
>;
