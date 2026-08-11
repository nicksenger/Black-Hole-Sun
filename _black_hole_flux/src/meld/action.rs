//! Actions and state for the model-free two-input meld protocol.

use jungle_sdk::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use black_hole_spec::{EmissionId, ObjectId};

use super::effect::{GenerateTransformIdEffect, WaitForMeldPotentiation, WaitForMeldPropagation};
use crate::cell::effect::Transmit as TransmitEffect;

/// Initial receive mailboxes for a binary vertex, in declared `P1`, `P2` order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MeldSeed {
    pub p1_recv_id: ObjectId,
    pub p2_recv_id: ObjectId,
}

/// Runtime identity and mailbox state for a [`Meld`](super::Meld) journey.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MeldState {
    /// Stable ID passed to the transform on every propagation pass.
    pub transform_id: Uuid,
    p1_recv_id: ObjectId,
    p2_recv_id: ObjectId,
    send_id: ObjectId,
}

/// Initializes both independent input-port mailbox chains from [`MeldSeed`].
pub struct InitMeld;

#[jungle::action(carry = MeldSeed)]
impl Action for InitMeld {
    type Effect = NoEffect;
    type Input = MeldSeed;
    type Output = ();

    fn emit(_state: &MeldState, seed: Self::Input) -> ((), MeldSeed) {
        ((), seed)
    }

    fn absorb(
        state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
        seed: MeldSeed,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("initialize meld failed".to_string()))?;
        state.p1_recv_id = seed.p1_recv_id;
        state.p2_recv_id = seed.p2_recv_id;
        Ok(())
    }
}

/// Generates and stores the stable ID for this meld journey's transform.
pub struct GenerateTransformId;

#[jungle::action]
impl Action for GenerateTransformId {
    type Effect = GenerateTransformIdEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &MeldState, _input: Self::Input) {}

    fn absorb(
        state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        state.transform_id = output
            .map_err(|error| Failure::Message(format!("generate transform ID failed: {error}")))?;
        Ok(())
    }
}

/// Receives both propagation envelopes and emits their IDs in `P1`, `P2` order.
pub struct WaitForMeldPropagationAction;

#[jungle::action]
impl Action for WaitForMeldPropagationAction {
    type Effect = WaitForMeldPropagation;
    type Input = ();
    type Output = (EmissionId, EmissionId);

    fn emit(state: &MeldState, _input: Self::Input) -> (ObjectId, ObjectId) {
        (state.p1_recv_id, state.p2_recv_id)
    }

    fn absorb(
        state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (p1, p2) =
            output.map_err(|error| Failure::Message(format!("wait for meld propagation failed: {error}")))?;

        if p1.send_id != p2.send_id {
            return Err(Failure::Message(format!(
                "meld propagation send addresses disagree: P1={}, P2={}",
                p1.send_id, p2.send_id
            )));
        }

        state.p1_recv_id = p1.recv_id;
        state.p2_recv_id = p2.recv_id;
        state.send_id = p1.send_id;

        Ok((p1.emission_id, p2.emission_id))
    }
}

/// Adds this meld journey's stable ID to the pair passed into its transform.
pub struct PrepareTransformInput;

#[jungle::action(carry = (EmissionId, EmissionId))]
impl Action for PrepareTransformInput {
    type Effect = NoEffect;
    type Input = (EmissionId, EmissionId);
    type Output = (Uuid, (EmissionId, EmissionId));

    fn emit(_state: &MeldState, emissions: Self::Input) -> ((), Self::Input) {
        ((), emissions)
    }

    fn absorb(
        state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
        emissions: Self::Input,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare transform input failed".to_string()))?;
        Ok((state.transform_id, emissions))
    }
}

/// Publishes one transformed emission to the binary vertex's shared output.
pub struct MeldTransmit;

#[jungle::action]
impl Action for MeldTransmit {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(state: &MeldState, emission_id: Self::Input) -> (EmissionId, ObjectId) {
        (emission_id, state.send_id)
    }

    fn absorb(
        _state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("meld transmit failed: {error}")))
    }
}

/// Receives matching potentiation envelopes and advances both port chains.
pub struct WaitForMeldPotentiationAction;

#[jungle::action]
impl Action for WaitForMeldPotentiationAction {
    type Effect = WaitForMeldPotentiation;
    type Input = ();
    type Output = ();

    fn emit(state: &MeldState, _input: Self::Input) -> (ObjectId, ObjectId) {
        (state.p1_recv_id, state.p2_recv_id)
    }

    fn absorb(
        state: &mut MeldState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (p1, p2) =
            output.map_err(|error| Failure::Message(format!("wait for meld potentiation failed: {error}")))?;

        if p1.loss_up.to_bits() != p2.loss_up.to_bits()
            || p1.loss_down.to_bits() != p2.loss_down.to_bits()
        {
            return Err(Failure::Message(format!(
                "meld potentiation losses disagree: P1=({}, {}), P2=({}, {})",
                p1.loss_up, p1.loss_down, p2.loss_up, p2.loss_down
            )));
        }

        state.p1_recv_id = p1.recv_id;
        state.p2_recv_id = p2.recv_id;
        Ok(())
    }
}
