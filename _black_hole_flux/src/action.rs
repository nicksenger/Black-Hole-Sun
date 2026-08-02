//! Actions for quark inference, perturbation, optimization, and transmission waiting.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub use black_hole_spec::{EmissionId, ObjectId};

use crate::effect::{
    QuarkInfer, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, WaitForInitiation,
    WaitForPotentiation, WaitForPropagation,
};

// ---------------------------------------------------------------------------
// CellState — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Cell`](crate::cell) journey.
///
/// Animals that use [`Cell`](crate::cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CellState {
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to download.
    pub recv_id: ObjectId,
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to upload.
    pub send_id: ObjectId,
}

// ---------------------------------------------------------------------------
// Potentiation — payload from a Transmission::Potentiation
// ---------------------------------------------------------------------------

/// Payload carried by a [`Transmission::Potentiation`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Potentiation {
    pub loss_up: f32,
    pub loss_down: f32,
    pub recv_id: ObjectId,
}

// ---------------------------------------------------------------------------
// Propagation — payload from a Transmission::Propagation
// ---------------------------------------------------------------------------

/// Payload carried by a [`Transmission::Propagation`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Propagation {
    pub emission_id: EmissionId,
    pub recv_id: ObjectId,
    pub send_id: ObjectId,
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
    type Effect = QuarkInfer<M>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &(), input: Self::Input) -> (EmissionId, EmissionId) {
        (input.clone(), input)
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
        _carry: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("quark inference failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbUp — perturb quark weights upward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbUp<const SEED: u64>;

#[jungle::action]
impl<const SEED: u64> Action for PerturbUp<SEED> {
    type Effect = QuarkPerturbUp;
    type Input = ();
    type Output = ();

    fn emit(_state: &(), _input: Self::Input) -> u64 {
        SEED
    }

    fn absorb(
        _state: &mut (),
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

    fn emit(_state: &(), input: Self::Input) -> () {
        input
    }

    fn absorb(
        _state: &mut (),
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

    fn emit(_state: &(), input: Self::Input) -> Potentiation {
        input
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("optimize failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// WaitForInitiationAction — await a Transmission::Initiation from void
// ---------------------------------------------------------------------------

/// Action that waits for an initiation transmission using the recv_id from
/// [`CellState`].
///
/// Reads `recv_id` from state, downloads the transmission, and stores the
/// new `recv_id` back into state. Returns unit — there is no data payload
/// to thread downstream; the emission ID is embedded in the initiation.
pub struct WaitForInitiationAction;

#[jungle::action]
impl Action for WaitForInitiationAction {
    type Effect = WaitForInitiation;
    type Input = ();
    type Output = ();
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let ((), recv_id) =
            output.map_err(|e| Failure::Message(format!("wait for initiation failed: {e}")))?;
        state.recv_id = recv_id;
        Ok(())
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

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
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

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
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
    type Effect = crate::effect::Transmit;
    type Input = EmissionId;
    type Output = ();

    fn emit(_state: &(), input: Self::Input) -> EmissionId {
        input
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("transmit failed: {e}")))
    }
}
