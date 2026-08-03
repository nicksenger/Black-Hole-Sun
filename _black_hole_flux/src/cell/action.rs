//! Cell actions for perturbation, optimization, transmission, and waiting.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CellState — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Cell`](crate::Cell) journey.
///
/// Animals that use [`Cell`](crate::Cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CellState {
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to download.
    pub recv_id: black_hole_spec::ObjectId,
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to upload.
    pub send_id: black_hole_spec::ObjectId,
    /// Random seed passed to the perturb-up step each iteration.
    pub perturbation_seed: u64,
}

pub use black_hole_spec::EmissionId;

use super::effect::{
    QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, Transmit as TransmitEffect,
    WaitForPotentiation, WaitForPropagation,
};

// ---------------------------------------------------------------------------
// InitRecvId — set recv_id from the seed ObjectId (first step of Cell flow)
// ---------------------------------------------------------------------------

/// Action that initializes `recv_id` in [`CellState`] from the seed
/// [`ObjectId`](black_hole_spec::ObjectId).
///
/// This is the very first step of a [`Cell`](crate::Cell) flow, converting the
/// animal seed into the initial receive ID for the training loop.
pub struct InitRecvId;

#[jungle::action(carry = black_hole_spec::ObjectId)]
impl Action for InitRecvId {
    type Effect = NoEffect;
    type Input = black_hole_spec::ObjectId;
    type Output = ();

    fn emit(_state: &CellState, input: Self::Input) -> ((), black_hole_spec::ObjectId) {
        ((), input)
    }

    fn absorb(
        state: &mut CellState,
        _output: EffectCompletion<Self::Effect>,
        carry: black_hole_spec::ObjectId,
    ) -> Result<Self::Output, Failure> {
        state.recv_id = carry;
        Ok(())
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
// QuarkInferStep — action wrapper for use inside Nucleus flows
// ---------------------------------------------------------------------------

/// Action that performs quark inference on an [`EmissionId`].
pub struct QuarkInferStep<M>(PhantomData<fn() -> M>);

#[jungle::action(carry = EmissionId)]
impl<M> Action for QuarkInferStep<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = super::super::nucleus::effect::QuarkInfer<M>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, input: Self::Input) -> (EmissionId, EmissionId) {
        (input.clone(), input)
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
        _carry: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("quark inference failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbUp — perturb quark weights upward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbUp;

#[jungle::action]
impl Action for PerturbUp {
    type Effect = QuarkPerturbUp;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState, _input: Self::Input) -> u64 {
        state.perturbation_seed
    }

    fn absorb(
        _: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb up failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbDown — perturb quark weights downward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbDown;

#[jungle::action]
impl Action for PerturbDown {
    type Effect = QuarkPerturbDown;
    type Input = ();
    type Output = ();

    fn emit(_state: &CellState, input: Self::Input) -> () {
        input
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb down failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Optimize — apply QuZO optimization (no carry)
// ---------------------------------------------------------------------------

pub struct Optimize;

#[jungle::action]
impl Action for Optimize {
    type Effect = QuarkOptimize;
    type Input = Potentiation;
    type Output = ();

    fn emit(_state: &CellState, input: Self::Input) -> Potentiation {
        input
    }

    fn absorb(
        _state: &mut CellState,
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
pub struct WaitForPropagationAction;

#[jungle::action]
impl Action for WaitForPropagationAction {
    type Effect = WaitForPropagation;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState,
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
pub struct WaitForPotentiationAction;

#[jungle::action]
impl Action for WaitForPotentiationAction {
    type Effect = WaitForPotentiation;
    type Input = ();
    type Output = Potentiation;
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> black_hole_spec::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState,
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
pub struct Transmit;

#[jungle::action]
impl Action for Transmit {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(state: &CellState, input: Self::Input) -> (EmissionId, black_hole_spec::ObjectId) {
        (input, state.send_id)
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("transmit failed: {e}")))
    }
}
