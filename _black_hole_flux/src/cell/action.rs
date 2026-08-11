//! Cell actions for perturbation, optimization, transmission, and waiting.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CellState — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Cell`](crate::Cell) journey.
///
/// Animals that use [`Cell`](crate::Cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID. The generic `S` payload is available to user flows via
/// [`CellState::inner`] and defaults to `()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CellState<S = ()> {
    /// Stable ID of the quark model instance owned by this cell.
    pub model_id: Uuid,
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to download.
    pub recv_id: black_hole_spec::ObjectId,
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to upload.
    pub send_id: black_hole_spec::ObjectId,
    /// Random seed passed to the perturb-up step each iteration.
    pub perturbation_seed: u64,
    /// User-provided state threaded through all cell actions.
    #[serde(default)]
    pub inner: S,
}

pub use black_hole_spec::EmissionId;

use super::effect::{
    GenerateModelIdEffect, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, QuarkShutdown,
    QuarkStart, Transmit as TransmitEffect, WaitForPotentiation, WaitForPropagation,
};

// ---------------------------------------------------------------------------
// InitRecvId — set recv_id from the seed ObjectId (first step of Cell flow)
// ---------------------------------------------------------------------------

/// Action that initializes `recv_id` in [`CellState`] from the seed
/// [`ObjectId`](black_hole_spec::ObjectId).
///
/// This is the very first step of a [`Cell`](crate::Cell) flow, converting the
/// animal seed into the initial receive ID for the training loop.
pub struct InitRecvId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = black_hole_spec::ObjectId)]
impl<S> Action for InitRecvId<S> {
    type Effect = NoEffect;
    type Input = black_hole_spec::ObjectId;
    type Output = ();

    fn emit(_state: &CellState<S>, input: Self::Input) -> ((), black_hole_spec::ObjectId) {
        ((), input)
    }

    fn absorb(
        state: &mut CellState<S>,
        _output: EffectCompletion<Self::Effect>,
        carry: black_hole_spec::ObjectId,
    ) -> Result<Self::Output, Failure> {
        state.recv_id = carry;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Model instance lifecycle
// ---------------------------------------------------------------------------

/// Generates the stable model instance ID owned by this cell.
pub struct GenerateModelId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for GenerateModelId<S> {
    type Effect = GenerateModelIdEffect;
    type Input = ();
    type Output = Uuid;

    fn emit(_state: &CellState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("generate model ID failed: {error}")))
    }
}

/// Starts the generated quark model instance and stores its ID in cell state.
pub struct StartModel<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = Uuid)]
impl<S> Action for StartModel<S> {
    type Effect = QuarkStart;
    type Input = Uuid;
    type Output = ();

    fn emit(_state: &CellState<S>, model_id: Self::Input) -> (Uuid, Uuid) {
        (model_id, model_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        model_id: Uuid,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("start model failed: {error}")))?;
        state.model_id = model_id;
        Ok(())
    }
}

/// Shuts down the quark model instance owned by this cell.
pub struct ShutdownModel<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ShutdownModel<S> {
    type Effect = QuarkShutdown;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> Uuid {
        state.model_id
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("shutdown model failed: {error}")))
    }
}

/// Adds this cell's model ID to the input passed into its Atom.
pub struct PrepareAtomInput<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = EmissionId)]
impl<S> Action for PrepareAtomInput<S> {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = (Uuid, EmissionId);

    fn emit(_state: &CellState<S>, emission_id: Self::Input) -> ((), EmissionId) {
        ((), emission_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        emission_id: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare Atom input failed".to_string()))?;
        Ok((state.model_id, emission_id))
    }
}

// ---------------------------------------------------------------------------
// Potentiation — payload from a Transmission::Potentiation
// ---------------------------------------------------------------------------

/// Payload carried by a [`Transmission::Potentiation`](black_hole_spec::Transmission).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Potentiation {
    pub loss_up: f32,
    pub loss_down: f32,
    pub recv_id: black_hole_spec::ObjectId,
}

// ---------------------------------------------------------------------------
// Propagation — payload from a Transmission::Propagation
// ---------------------------------------------------------------------------

/// Payload carried by a [`Transmission::Propagation`](black_hole_spec::Transmission).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Propagation {
    pub emission_id: EmissionId,
    pub recv_id: black_hole_spec::ObjectId,
    pub send_id: black_hole_spec::ObjectId,
}

// ---------------------------------------------------------------------------
// QuarkInferStep — action wrapper for use inside Atom flows
// ---------------------------------------------------------------------------

/// Action that performs quark inference for one model instance.
pub struct QuarkInferStep<M, S = ()>(PhantomData<fn() -> (M, S)>);

#[jungle::action]
impl<M, S> Action for QuarkInferStep<M, S>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = super::super::atom::effect::QuarkInfer<M>;
    type Input = (Uuid, EmissionId);
    type Output = EmissionId;

    fn emit(_state: &CellState<S>, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("quark inference failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbUp — perturb quark weights upward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbUp<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PerturbUp<S> {
    type Effect = QuarkPerturbUp;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> (Uuid, u64) {
        (state.model_id, state.perturbation_seed)
    }

    fn absorb(
        _: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb up failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbDown — perturb quark weights downward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbDown<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PerturbDown<S> {
    type Effect = QuarkPerturbDown;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> Uuid {
        state.model_id
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb down failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Optimize — apply QuZO optimization (no carry)
// ---------------------------------------------------------------------------

pub struct Optimize<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for Optimize<S> {
    type Effect = QuarkOptimize;
    type Input = Potentiation;
    type Output = ();

    fn emit(state: &CellState<S>, input: Self::Input) -> (Uuid, Potentiation) {
        (state.model_id, input)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("optimize failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// WaitForPropagationAction — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Action that waits for a propagation transmission using the recv_id from
/// [`CellState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` and `send_id` in state, and emits the [`Propagation`] payload.
pub struct WaitForPropagationAction<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPropagationAction<S> {
    type Effect = WaitForPropagation;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
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
// WaitForPotentiationAction — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Action that waits for a potentiation transmission using the recv_id from
/// [`CellState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` in state, and emits the [`Potentiation`] payload.
pub struct WaitForPotentiationAction<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPotentiationAction<S> {
    type Effect = WaitForPotentiation;
    type Input = ();
    type Output = Potentiation;
    type Carry = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (potentiation, recv_id) =
            output.map_err(|e| Failure::Message(format!("wait for potentiation failed: {e}")))?;
        state.recv_id = recv_id;
        Ok(potentiation)
    }
}

// ---------------------------------------------------------------------------
// Transmit — propagates an emission to the next cell
// ---------------------------------------------------------------------------

/// Propagates an [`EmissionId`] to the next cell.
///
/// Replaces the previous `DiscardEmission` action by actually transmitting
/// the emission output rather than discarding it.
pub struct Transmit<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for Transmit<S> {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(state: &CellState<S>, input: Self::Input) -> (EmissionId, black_hole_spec::ObjectId) {
        (input, state.send_id)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("transmit failed: {e}")))
    }
}
