//! Actions and state for the model-free two-input fusion protocol.

use jungle_sdk::prelude::*;
use serde::{Deserialize, Serialize};

use black_hole_spec::{EmissionId, ObjectId};

use super::effect::{WaitForFusionPotentiation, WaitForFusionPropagation};
use crate::cell::effect::Transmit as TransmitEffect;

/// Initial receive mailboxes for a binary vertex, in declared `P1`, `P2` order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FusionSeed {
    pub p1_recv_id: ObjectId,
    pub p2_recv_id: ObjectId,
}

/// Runtime mailbox state for a [`Fusion`](super::Fusion) journey.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FusionState {
    p1_recv_id: ObjectId,
    p2_recv_id: ObjectId,
    send_id: ObjectId,
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
        Ok(())
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
