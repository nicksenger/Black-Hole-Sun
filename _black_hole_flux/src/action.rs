//! Actions for quark inference, perturbation, optimization, and perturbation claiming.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub use black_hole_spec::EmissionId;

use crate::effect::{ClaimLoss, ClaimPerturbation, QuarkInfer, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp};

// ---------------------------------------------------------------------------
// QuarkInferStep — action wrapper for use inside Nucleus flows
// ---------------------------------------------------------------------------

/// Action that performs quark inference on an [`EmissionId`].
///
/// Stateless — bound to any animal state via the [`Identity`] aspect.
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

/// Action that perturbs the associated quark's weights in the positive direction.
///
/// Takes `()` and returns `()`.  Uses a compile-time seed constant `SEED` for
/// the perturbation.
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

/// Action that perturbs the associated quark's weights in the negative direction.
///
/// Takes `()` and returns `()`.
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

/// Action that applies the QuZO optimization step with up/down loss values.
///
/// Takes `(loss_up: f32, loss_down: f32)` and returns `()`.
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
// ClaimPerturbationAction — await an external perturbation (EmissionId)
// ---------------------------------------------------------------------------

/// Action that awaits an external jungle perturbation containing an [`EmissionId`].
///
/// Takes `()` and returns the deserialized [`EmissionId`].
pub struct ClaimPerturbationAction;

#[jungle::action]
impl Action for ClaimPerturbationAction {
    type Effect = ClaimPerturbation;
    type Input = ();
    type Output = EmissionId;

    fn emit(_state: &(), input: Self::Input) -> () {
        input
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("claim perturbation failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// ClaimLossAction — await an external perturbation with loss values
// ---------------------------------------------------------------------------

/// Action that awaits an external jungle perturbation containing
/// `(loss_up: f32, loss_down: f32)`.
///
/// Similar to [`ClaimPerturbationAction`] but deserializes the payload as
/// a loss tuple instead of an [`EmissionId`].
pub struct ClaimLossAction;

#[jungle::action]
impl Action for ClaimLossAction {
    type Effect = ClaimLoss;
    type Input = ();
    type Output = (f32, f32);

    fn emit(_state: &(), input: Self::Input) -> () {
        input
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("claim loss perturbation failed: {e}")))
    }
}
