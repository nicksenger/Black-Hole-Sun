//! Actions and state for the model-free two-input fusion protocol.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use black_hole_spec::{EmissionId, ObjectId};

use super::effect::{
    GenerateTransformIdEffect, WaitForFusionPotentiation, WaitForFusionPropagation,
};
use crate::cell::action::Potentiation;
use crate::cell::effect::{
    QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, QuarkStart, Transmit as TransmitEffect,
};
use crate::model_config::{DefaultConfig, ModelConfig};

/// Initial receive mailboxes for a binary vertex, in declared `P1`, `P2` order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FusionSeed {
    pub p1_recv_id: ObjectId,
    pub p2_recv_id: ObjectId,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
}

/// Runtime identity and mailbox state for a [`Fusion`](super::Fusion) journey.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FusionState {
    /// Stable ID passed to the transform on every propagation pass.
    pub transform_id: Uuid,
    p1_recv_id: ObjectId,
    p2_recv_id: ObjectId,
    send_id: ObjectId,
    /// Random seed passed to perturb-up before each propagation pass.
    pub perturbation_seed: u64,
    /// Number of propagation microsteps to run per propagation phase.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    /// Number of completed microsteps in the current propagation phase.
    #[serde(default)]
    pub grad_step: usize,
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

/// Initializes both independent input-port mailbox chains from [`FusionSeed`].
pub struct InitFusion;

#[jungle::action(carry = FusionSeed)]
impl Action for InitFusion {
    type Effect = NoEffect;
    type Input = FusionSeed;
    type Output = ();

    fn emit(_state: &FusionState, seed: Self::Input) -> ((), FusionSeed) {
        ((), seed)
    }

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
        seed: FusionSeed,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("initialize fusion failed".to_string()))?;
        state.p1_recv_id = seed.p1_recv_id;
        state.p2_recv_id = seed.p2_recv_id;
        state.grad_steps = seed.grad_steps.max(1);
        state.grad_step = 0;
        Ok(())
    }
}

/// Resets the fusion microstep cursor before one propagation microstep phase.
pub struct BeginFusionGradientAccumulation;

#[jungle::action]
impl Action for BeginFusionGradientAccumulation {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &FusionState, _input: Self::Input) {}

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("begin fusion accumulation failed".to_string()))?;
        state.grad_step = 0;
        if state.grad_steps == 0 {
            state.grad_steps = 1;
        }
        Ok(())
    }
}

/// Advances the fusion microstep cursor after one propagation/infer microstep.
pub struct AdvanceFusionGradientStep;

#[jungle::action]
impl Action for AdvanceFusionGradientStep {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &FusionState, _input: Self::Input) {}

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("advance fusion accumulation failed".to_string()))?;
        state.grad_step = state.grad_step.saturating_add(1);
        Ok(())
    }
}

/// Generates and stores the stable ID for this fusion journey's transform.
pub struct GenerateTransformId;

#[jungle::action]
impl Action for GenerateTransformId {
    type Effect = GenerateTransformIdEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &FusionState, _input: Self::Input) {}

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        state.transform_id = output
            .map_err(|error| Failure::Message(format!("generate transform ID failed: {error}")))?;
        Ok(())
    }
}

/// Starts the quark model instance keyed by this fusion journey's ID.
pub struct FusionStartModel<H = DefaultConfig>(PhantomData<fn() -> H>);

#[jungle::action]
impl<H> Action for FusionStartModel<H>
where
    H: ModelConfig,
{
    type Effect = QuarkStart<H>;
    type Input = ();
    type Output = ();

    fn emit(state: &FusionState, _input: Self::Input) -> Uuid {
        state.transform_id
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output
            .map(|_| ())
            .map_err(|error| Failure::Message(format!("start fusion model failed: {error}")))
    }
}

/// Perturbs the quark model instance upward before one propagation pass.
pub struct FusionPerturbUp;

#[jungle::action]
impl Action for FusionPerturbUp {
    type Effect = QuarkPerturbUp;
    type Input = ();
    type Output = ();

    fn emit(state: &FusionState, _input: Self::Input) -> (Uuid, u64) {
        (state.transform_id, state.perturbation_seed)
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion perturb up failed: {error}")))
    }
}

/// Perturbs the quark model instance downward between propagation passes.
pub struct FusionPerturbDown;

#[jungle::action]
impl Action for FusionPerturbDown {
    type Effect = QuarkPerturbDown;
    type Input = ();
    type Output = ();

    fn emit(state: &FusionState, _input: Self::Input) -> Uuid {
        state.transform_id
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion perturb down failed: {error}")))
    }
}

/// Receives both propagation envelopes and emits their IDs in `P1`, `P2` order.
pub struct WaitForFusionPropagationAction;

#[jungle::action]
impl Action for WaitForFusionPropagationAction {
    type Effect = WaitForFusionPropagation;
    type Input = ();
    type Output = (EmissionId, EmissionId);

    fn emit(state: &FusionState, _input: Self::Input) -> (ObjectId, ObjectId) {
        (state.p1_recv_id, state.p2_recv_id)
    }

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (p1, p2) = output.map_err(|error| {
            Failure::Message(format!("wait for fusion propagation failed: {error}"))
        })?;

        if p1.send_id != p2.send_id {
            return Err(Failure::Message(format!(
                "fusion propagation send addresses disagree: P1={}, P2={}",
                p1.send_id, p2.send_id
            )));
        }

        state.p1_recv_id = p1.recv_id;
        state.p2_recv_id = p2.recv_id;
        state.send_id = p1.send_id;

        Ok((p1.emission_id, p2.emission_id))
    }
}

/// Adds this fusion journey's stable ID to the pair passed into its transform.
pub struct PrepareTransformInput;

#[jungle::action(carry = (EmissionId, EmissionId))]
impl Action for PrepareTransformInput {
    type Effect = NoEffect;
    type Input = (EmissionId, EmissionId);
    type Output = (Uuid, (EmissionId, EmissionId));

    fn emit(_state: &FusionState, emissions: Self::Input) -> ((), Self::Input) {
        ((), emissions)
    }

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
        emissions: Self::Input,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare transform input failed".to_string()))?;
        Ok((state.transform_id, emissions))
    }
}

/// Publishes one transformed emission to the binary vertex's shared output.
pub struct FusionTransmit;

#[jungle::action]
impl Action for FusionTransmit {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(state: &FusionState, emission_id: Self::Input) -> (EmissionId, ObjectId) {
        (emission_id, state.send_id)
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion transmit failed: {error}")))
    }
}

/// Runs quark inference for one transformed fusion emission.
pub struct FusionQuarkInferStep<M>(PhantomData<fn() -> M>);

#[jungle::action]
impl<M> Action for FusionQuarkInferStep<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = crate::atom::effect::QuarkInfer<M>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(state: &FusionState, emission_id: Self::Input) -> (Uuid, EmissionId) {
        (state.transform_id, emission_id)
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion quark inference failed: {error}")))
    }
}

/// Receives matching potentiation envelopes and advances both port chains.
pub struct WaitForFusionPotentiationAction;

#[jungle::action]
impl Action for WaitForFusionPotentiationAction {
    type Effect = WaitForFusionPotentiation;
    type Input = ();
    type Output = ();

    fn emit(state: &FusionState, _input: Self::Input) -> (ObjectId, ObjectId) {
        (state.p1_recv_id, state.p2_recv_id)
    }

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (p1, p2) = output.map_err(|error| {
            Failure::Message(format!("wait for fusion potentiation failed: {error}"))
        })?;

        if p1.loss_up.to_bits() != p2.loss_up.to_bits()
            || p1.loss_down.to_bits() != p2.loss_down.to_bits()
        {
            return Err(Failure::Message(format!(
                "fusion potentiation losses disagree: P1=({}, {}), P2=({}, {})",
                p1.loss_up, p1.loss_down, p2.loss_up, p2.loss_down
            )));
        }

        state.p1_recv_id = p1.recv_id;
        state.p2_recv_id = p2.recv_id;
        Ok(())
    }
}

/// Receives matching potentiation envelopes and emits losses for optimization.
pub struct WaitForFusionPotentiationForOptimize;

#[jungle::action]
impl Action for WaitForFusionPotentiationForOptimize {
    type Effect = WaitForFusionPotentiation;
    type Input = ();
    type Output = Potentiation;

    fn emit(state: &FusionState, _input: Self::Input) -> (ObjectId, ObjectId) {
        (state.p1_recv_id, state.p2_recv_id)
    }

    fn absorb(
        state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (p1, p2) = output.map_err(|error| {
            Failure::Message(format!("wait for fusion potentiation failed: {error}"))
        })?;

        if p1.loss_up.to_bits() != p2.loss_up.to_bits()
            || p1.loss_down.to_bits() != p2.loss_down.to_bits()
        {
            return Err(Failure::Message(format!(
                "fusion potentiation losses disagree: P1=({}, {}), P2=({}, {})",
                p1.loss_up, p1.loss_down, p2.loss_up, p2.loss_down
            )));
        }

        state.p1_recv_id = p1.recv_id;
        state.p2_recv_id = p2.recv_id;

        Ok(p1)
    }
}

/// Applies quark optimization using the synchronized fusion losses.
pub struct FusionOptimize;

#[jungle::action]
impl Action for FusionOptimize {
    type Effect = QuarkOptimize;
    type Input = Potentiation;
    type Output = ();

    fn emit(state: &FusionState, potentiation: Self::Input) -> (Uuid, Potentiation) {
        (state.transform_id, potentiation)
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output
            .map(|_| ())
            .map_err(|error| Failure::Message(format!("fusion optimize failed: {error}")))
    }
}
