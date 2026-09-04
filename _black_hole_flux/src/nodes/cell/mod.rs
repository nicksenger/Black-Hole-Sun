//! Cell module - higher-order flows for cell training loops.

pub mod action;
pub mod effect;

pub use action::CellState;

use action::{
    AdvanceGradientStep as AdvanceGradientStep_,
    BeginGradientAccumulation as BeginGradientAccumulation_, GenerateModelId as GenerateModelId_,
    InitRecvId as InitRecvId_, Optimize as Optimize_, OptimizeOperation as OptimizeOperation_,
    PerturbDown as PerturbDown_, PerturbOperationDown as PerturbOperationDown_,
    PerturbOperationUp as PerturbOperationUp_, PerturbUp as PerturbUp_, Potentiation,
    PrepareAtomInput as PrepareAtomInput_, PrepareOperationInput as PrepareOperationInput_,
    StartModel as StartModel_, StartOperation as StartOperation_, Transmit as Transmit_,
    TransmitArtifact as TransmitArtifact_, WaitForArtifact as WaitForArtifact_,
    WaitForOperationalControl as WaitForOperationalControl_,
    WaitForPotentiation as WaitForPotentiation_, WaitForPropagation as WaitForPropagation_,
};
use black_hole_spec::TensorContract;
use black_hole_type::EmissionId;
use jungle_sdk::prelude::*;
use jungle_zoo::backoff::Backoff;
use jungle_zoo::predicate::Always;
use jungle_zoo::Noop;
use uuid::Uuid;

use crate::mass::{DefaultConfig, ModelConfig};
use crate::{Atom, OperationAtom};

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
/// [`Transmission`](black_hole_type::Transmission) messages from void.
#[derive(Flow)]
pub struct CellWithState<N, S, H: ModelConfig>(
    Step<InitRecvId_<S>>,
    Step<GenerateModelId_<S>>,
    StartModelWithBackoff<S, H>,
    While<Always<CellState<S>, ()>, InnerWithState<N, S>>,
);

pub type Cell<N, S = (), H = DefaultConfig> = CellWithState<N, S, H>;

/// QuZO-compatible Cell whose data plane is typed by an operation contract.
///
/// This is the executable generic counterpart to [`Cell`]. It uses
/// [`ArtifactDelivery`](black_hole_type::ArtifactDelivery) for inference data
/// and [`OperationalControl`](black_hole_type::OperationalControl) for the
/// strategy-selected optimization command.
#[derive(Flow)]
pub struct OperationCellWithState<
    N,
    Op: TensorContract<Input: Send, Output: Send> + Send + Sync + 'static,
    S,
>(
    Step<InitRecvId_<S>>,
    Step<GenerateModelId_<S>>,
    Step<StartOperation_<Op, S>>,
    While<Always<CellState<S>, ()>, OperationInnerWithState<N, Op, S>>,
);

pub type OperationCell<N, Op, S = ()> = OperationCellWithState<N, Op, S>;

/// Forward-only operation host.
///
/// Unlike [`OperationCell`], this flow starts the operation and repeatedly
/// processes typed artifact deliveries without perturbation, optimization, or
/// any other QuZO lifecycle capability.
#[derive(Flow)]
pub struct ForwardOperationCellWithState<
    N,
    Op: TensorContract<Input: Send, Output: Send> + Send + Sync + 'static,
    S,
>(
    Step<InitRecvId_<S>>,
    Step<GenerateModelId_<S>>,
    Step<StartOperation_<Op, S>>,
    While<Always<CellState<S>, ()>, OperationPropagationMicrostepWithState<N, Op, S>>,
);

pub type ForwardOperationCell<N, Op, S = ()> = ForwardOperationCellWithState<N, Op, S>;

/// The body of one iteration of a [`Cell`] loop.
#[derive(Flow)]
pub struct InnerWithState<N, S>(
    Step<BeginGradientAccumulation_<S>>,
    PerturbUpWithBackoff<S>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<BeginGradientAccumulation_<S>>,
    PerturbDownWithBackoff<S>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<WaitForPotentiation_<S>>,
    //Step<Optimize_<S>>,
    OptimizeWithBackoff<S>,
);

/// One propagation/infer/transmit microstep.
#[derive(Flow)]
pub struct InnerPropagationMicrostepWithState<N, S>(
    Step<WaitForPropagation_<S>>,
    Step<PrepareAtomInput_<S>>,
    N,
    Step<Transmit_<S>>,
    Step<AdvanceGradientStep_<S>>,
);

pub type Inner<N, S = ()> = InnerWithState<N, S>;

/// One two-sided optimization iteration over operation-typed artifacts.
#[derive(Flow)]
pub struct OperationInnerWithState<
    N,
    Op: TensorContract<Input: Send, Output: Send> + Send + Sync + 'static,
    S,
>(
    Step<BeginGradientAccumulation_<S>>,
    Step<PerturbOperationUp_<Op, S>>,
    While<HasPendingGradientStep<S>, OperationPropagationMicrostepWithState<N, Op, S>>,
    Step<BeginGradientAccumulation_<S>>,
    Step<PerturbOperationDown_<Op, S>>,
    While<HasPendingGradientStep<S>, OperationPropagationMicrostepWithState<N, Op, S>>,
    Step<WaitForOperationalControl_<Potentiation, S>>,
    Step<OptimizeOperation_<Op, S>>,
);

/// One typed delivery → operation → delivery microstep.
#[derive(Flow)]
pub struct OperationPropagationMicrostepWithState<
    N,
    Op: TensorContract<Input: Send, Output: Send> + Send + Sync + 'static,
    S,
>(
    Step<WaitForArtifact_<Op::Input, S>>,
    Step<PrepareOperationInput_<Op::Input, S>>,
    N,
    Step<TransmitArtifact_<Op::Output, S>>,
    Step<AdvanceGradientStep_<S>>,
);

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

/// Bare operation-typed Cell with no input or output transforms.
pub type OperationPrimordium<Op, S = ()> = OperationCell<
    OperationAtom<
        Step<Noop<CellState<S>, (Uuid, EmissionId<<Op as TensorContract>::Input>)>>,
        Step<Noop<CellState<S>, EmissionId<<Op as TensorContract>::Output>>>,
        (),
        Op,
        S,
    >,
    Op,
    S,
>;

/// Bare forward-only operation node with no input or output transforms.
pub type ForwardOperationPrimordium<Op, S = ()> = ForwardOperationCell<
    OperationAtom<
        Step<Noop<CellState<S>, (Uuid, EmissionId<<Op as TensorContract>::Input>)>>,
        Step<Noop<CellState<S>, EmissionId<<Op as TensorContract>::Output>>>,
        (),
        Op,
        S,
    >,
    Op,
    S,
>;
