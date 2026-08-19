//! Boundary actions for model-free warp boundary loops.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;

pub use crate::cell::action::{
    AdvanceGradientStep, BeginGradientAccumulation, CellState, EmissionId, Potentiation,
    Propagation, Transmit, WaitForPropagation,
};
pub use crate::sun::BoundaryInit;

use super::effect::WaitForBoundaryPotentiationEffect;

/// Initializes boundary state from [`BoundaryInit`].
///
/// The `warp_journey_id` is stored in `CellState::model_id` so downstream
/// boundary actions can address the associated warp journey without introducing
/// a separate state shape.
pub struct InitRecvId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = BoundaryInit)]
impl<S> Action for InitRecvId<S> {
    type Effect = NoEffect;
    type Input = BoundaryInit;
    type Output = ();

    fn emit(_state: &CellState<S>, input: Self::Input) -> ((), BoundaryInit) {
        ((), input)
    }

    fn absorb(
        state: &mut CellState<S>,
        _output: EffectCompletion<Self::Effect>,
        carry: BoundaryInit,
    ) -> Result<Self::Output, Failure> {
        state.recv_id = carry.recv_id;
        state.grad_steps = carry.grad_steps.max(1);
        state.grad_step = 0;
        state.model_id = carry.warp_journey_id;
        Ok(())
    }
}

/// Waits for a boundary potentiation envelope and advances the receive mailbox.
pub struct WaitForPotentiation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPotentiation<S> {
    type Effect = WaitForBoundaryPotentiationEffect;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (potentiation, recv_id) = output.map_err(|error| {
            Failure::Message(format!("wait for boundary potentiation failed: {error}"))
        })?;
        state.recv_id = recv_id;
        state.perturbation_seed = potentiation.seed;
        Ok(())
    }
}

/// Waits for boundary potentiation and emits the payload to downstream steps.
pub struct WaitForPotentiationForInput<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPotentiationForInput<S> {
    type Effect = WaitForBoundaryPotentiationEffect;
    type Input = ();
    type Output = Potentiation;

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (potentiation, recv_id) = output.map_err(|error| {
            Failure::Message(format!("wait for boundary potentiation failed: {error}"))
        })?;
        state.recv_id = recv_id;
        state.perturbation_seed = potentiation.seed;
        Ok(potentiation)
    }
}
