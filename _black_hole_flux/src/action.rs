//! Actions for quark inference, perturbation, optimization, and transmission waiting.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub use black_hole_spec::{EmissionId, ObjectId};

use crate::effect::{
    QuarkInfer, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp,
    WaitForInitiation, WaitForPropagation, WaitForPotentiation,
};

// ---------------------------------------------------------------------------
// CellState — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Cell`](crate::Cell) journey.
///
/// Animals that use [`Cell`](crate::Cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID.
#[derive(Debug, Clone, Copy)]
pub struct CellState {
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to download.
    pub next_id: ObjectId,
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
    type Input = (f32, f32);
    type Output = ();

    fn emit(_state: &(), input: Self::Input) -> (f32, f32) {
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

/// Action that waits for an initiation transmission using the next_id from
/// [`CellState`].
///
/// Reads `next_id` from state, downloads the transmission, extracts the
/// emission ID to process, and stores the new `next_id` back into state.
pub struct WaitForInitiationAction;

#[jungle::action]
impl Action for WaitForInitiationAction {
    type Effect = WaitForInitiation;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
        state.next_id
    }

    fn absorb(
        state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (emission_id, next_id) = output.map_err(|e| {
            Failure::Message(format!("wait for initiation failed: {e}"))
        })?;
        state.next_id = next_id;
        Ok(emission_id)
    }
}

// ---------------------------------------------------------------------------
// WaitForPropagationAction — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Action that waits for a propagation transmission using the next_id from
/// [`CellState`].
pub struct WaitForPropagationAction;

#[jungle::action]
impl Action for WaitForPropagationAction {
    type Effect = WaitForPropagation;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
        state.next_id
    }

    fn absorb(
        state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (emission_id, next_id) = output.map_err(|e| {
            Failure::Message(format!("wait for propagation failed: {e}"))
        })?;
        state.next_id = next_id;
        Ok(emission_id)
    }
}

// ---------------------------------------------------------------------------
// WaitForPotentiationAction — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Action that waits for a potentiation transmission using the next_id from
/// [`CellState`].
pub struct WaitForPotentiationAction;

#[jungle::action]
impl Action for WaitForPotentiationAction {
    type Effect = WaitForPotentiation;
    type Input = ();
    type Output = (f32, f32);
    type Carry = ();

    fn emit(state: &CellState, _input: Self::Input) -> ObjectId {
        state.next_id
    }

    fn absorb(
        state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (loss, next_id) = output.map_err(|e| {
            Failure::Message(format!("wait for potentiation failed: {e}"))
        })?;
        state.next_id = next_id;
        Ok(loss)
    }
}

// ---------------------------------------------------------------------------
// DiscardEmission — converts EmissionId to () (no-op bridge action)
// ---------------------------------------------------------------------------

/// Discards an [`EmissionId`] and produces `()`.
///
/// Used to bridge between EmissionId-producing stages and unit-input stages
/// in the Cell loop.
pub struct DiscardEmission;

#[jungle::action]
impl Action for DiscardEmission {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(_state: &(), _input: Self::Input) -> () {}

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("discard failed: {e:?}")))?;
        Ok(())
    }
}
