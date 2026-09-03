//! Boundary actions for model-free warp boundary loops.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

pub use crate::nodes::cell::action::{EmissionId, Potentiation, Propagation};
pub use crate::sun::BoundaryInit;

use super::effect::{
    ObserveWarpEffect, PerturbWarpEffect, TransmitEffect,
    WaitForBoundaryPotentiationEffect as WaitForPotentiationEffect, WaitForPropagationEffect,
};
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
        state.warp_journey_id = carry.warp_journey_id;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Gradient accumulation cursor — boundary-state mirrors of the cell actions
// ---------------------------------------------------------------------------

/// Resets the microstep cursor before a propagation microstep phase begins.
pub struct BeginGradientAccumulation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for BeginGradientAccumulation<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &BoundaryState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("begin accumulation failed".to_string()))?;
        state.grad_step = 0;
        if state.grad_steps == 0 {
            state.grad_steps = 1;
        }
        Ok(())
    }
}

/// Advances the microstep cursor after one propagation/transmit microstep.
pub struct AdvanceGradientStep<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for AdvanceGradientStep<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &BoundaryState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("advance accumulation failed".to_string()))?;
        state.grad_step = state.grad_step.saturating_add(1);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WaitForPropagation — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Action that waits for a propagation transmission using the recv_id from
/// [`BoundaryState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` and `send_id` in state, and emits the emission ID.
pub struct WaitForPropagation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPropagation<S> {
    type Effect = WaitForPropagationEffect;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &BoundaryState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let propagation =
            output.map_err(|e| Failure::Message(format!("wait for propagation failed: {e}")))?;
        state.recv_id = propagation.recv_id;
        state.send_id = propagation.send_id;
        Ok(propagation.emission_id)
    }
}

// ---------------------------------------------------------------------------
// WaitForPotentiation — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Action that waits for a potentiation transmission using the recv_id from
/// [`BoundaryState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` and perturbation seed in state, and emits the [`Potentiation`] payload.
pub struct WaitForPotentiation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPotentiation<S> {
    type Effect = WaitForPotentiationEffect;
    type Input = ();
    type Output = Potentiation;
    type Carry = ();

    fn emit(state: &BoundaryState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (potentiation, recv_id) =
            output.map_err(|e| Failure::Message(format!("wait for potentiation failed: {e}")))?;
        state.recv_id = recv_id;
        state.perturbation_seed = potentiation.seed;
        Ok(potentiation)
    }
}

// ---------------------------------------------------------------------------
// Transmit — propagates an emission to the next cell
// ---------------------------------------------------------------------------

/// Propagates an [`EmissionId`] to the next cell using the `send_id` from
/// [`BoundaryState`].
pub struct Transmit<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for Transmit<S> {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(
        state: &BoundaryState<S>,
        input: Self::Input,
    ) -> (EmissionId, black_hole_spec::ObjectId) {
        (input, state.send_id)
    }

    fn absorb(
        _state: &mut BoundaryState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("transmit failed: {e}")))
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
