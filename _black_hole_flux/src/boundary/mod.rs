//! Boundary module - higher-order flows for warp boundaries.

pub mod action;
pub mod effect;

use action::{
    AdvanceGradientStep as AdvanceGradientStep_,
    BeginGradientAccumulation as BeginGradientAccumulation_, CellState, InitRecvId as InitRecvId_,
    Transmit as Transmit_, WaitForPotentiation as WaitForPotentiation_,
    WaitForPropagation as WaitForPropagation_,
};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;

/// Predicate that keeps running boundary microsteps until `grad_steps` is reached.
pub struct HasPendingGradientStep<S>(std::marker::PhantomData<fn() -> S>);

impl<S> Predicate<(&CellState<S>, &())> for HasPendingGradientStep<S> {
    fn eval((state, _): &(&CellState<S>, &())) -> bool {
        state.grad_step < state.grad_steps.max(1)
    }
}

/// A model-free boundary loop that behaves like a unary cell around `N`.
#[derive(Flow)]
pub struct NoModelBoundaryWithState<N, S>(
    Step<InitRecvId_<S>>,
    While<Always<CellState<S>, ()>, InnerWithState<N, S>>,
);

pub type NoModelBoundary<N, S = ()> = NoModelBoundaryWithState<N, S>;
pub type Boundary<N, S = ()> = NoModelBoundary<N, S>;

/// The body of one boundary loop iteration.
#[derive(Flow)]
pub struct InnerWithState<N, S>(
    Step<BeginGradientAccumulation_<S>>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<BeginGradientAccumulation_<S>>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<WaitForPotentiation_<S>>,
);

/// One boundary propagation/transmit microstep around `N`.
#[derive(Flow)]
pub struct InnerPropagationMicrostepWithState<N, S>(
    Step<WaitForPropagation_<S>>,
    N,
    Step<Transmit_<S>>,
    Step<AdvanceGradientStep_<S>>,
);

pub type Inner<N, S = ()> = InnerWithState<N, S>;
pub type InnerPropagationMicrostep<N, S = ()> = InnerPropagationMicrostepWithState<N, S>;
