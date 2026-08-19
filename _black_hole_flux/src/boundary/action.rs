//! Boundary actions for model-free warp boundary loops.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

pub use crate::cell::action::{
    AdvanceGradientStep, BeginGradientAccumulation, EmissionId, Potentiation, Propagation,
    Transmit, WaitForPotentiation, WaitForPropagation,
};
pub use crate::sun::BoundaryInit;

use super::effect::{ObserveWarpEffect, PerturbWarpEffect};
use super::BoundaryState;

/// Initializes boundary state from [`BoundaryInit`].
///
/// The `warp_journey_id` is stored in `BoundaryState::model_id` so downstream
/// boundary actions can address the associated warp journey without introducing
/// a separate state shape.
pub struct InitRecvId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = BoundaryInit)]
impl<S> Action for InitRecvId<S> {
    type Effect = NoEffect;
    type Input = BoundaryInit;
    type Output = ();

    fn emit(_state: &BoundaryState<S>, input: Self::Input) -> ((), BoundaryInit) {
        ((), input)
    }

    fn absorb(
        state: &mut BoundaryState<S>,
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

// ---------------------------------------------------------------------------
// ObserveWarp — query the warp journey appearance into state
// ---------------------------------------------------------------------------

/// Queries the associated warp journey's appearance and stores it in
/// [`BoundaryState::inner`], passing the incoming emission id through.
pub struct ObserveWarp<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = EmissionId)]
impl<S> Action for ObserveWarp<S>
where
    S: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = ObserveWarpEffect<S>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(state: &BoundaryState<S>, emission_id: Self::Input) -> (Uuid, EmissionId) {
        (state.warp_journey_id, emission_id)
    }

    fn absorb(
        state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
        emission_id: EmissionId,
    ) -> Result<Self::Output, Failure> {
        let appearance =
            output.map_err(|error| Failure::Message(format!("observe warp failed: {error}")))?;
        state.inner = appearance;
        Ok(emission_id)
    }
}

// ---------------------------------------------------------------------------
// PerturbWarp — forward a potentiation to the warp journey
// ---------------------------------------------------------------------------

/// Forwards a [`Potentiation`] to the associated warp journey via perturb.
pub struct PerturbWarp<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PerturbWarp<S> {
    type Effect = PerturbWarpEffect;
    type Input = Potentiation;
    type Output = ();

    fn emit(state: &BoundaryState<S>, input: Self::Input) -> (Uuid, Potentiation) {
        (state.warp_journey_id, input)
    }

    fn absorb(
        _state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("perturb warp failed: {error}")))
    }
}
