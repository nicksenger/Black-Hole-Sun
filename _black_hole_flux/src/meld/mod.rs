//! Model-free two-input meld protocol.
//!
//! A [`Meld`] journey owns two independent mailbox chains. For each of the
//! two propagation passes it receives both envelopes in declared `P1`, `P2`
//! order, verifies that they name one shared output mailbox, passes its stable
//! UUID and the pair of emission IDs through `Transform` as
//! `(transform_id, (p1_emission_id, p2_emission_id))`, and transmits once. It
//! then consumes matching potentiation envelopes and rotates both chains
//! without applying model optimization.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;

pub use action::{
    GenerateTransformId, InitMeld, MeldSeed, MeldState, MeldTransmit, PrepareTransformInput,
    WaitForMeldPotentiationAction, WaitForMeldPropagationAction,
};

pub use effect::{GenerateTransformIdEffect, WaitForMeldPotentiation, WaitForMeldPropagation};

/// One complete two-pass meld epoch.
#[derive(Flow)]
pub struct MeldEpoch<Transform>(
    Step<WaitForMeldPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<MeldTransmit>,
    Step<WaitForMeldPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<MeldTransmit>,
    Step<WaitForMeldPotentiationAction>,
);

/// Infinite model-free meld loop driven by two input-port mailbox chains.
#[derive(Flow)]
pub struct Meld<Transform>(
    Step<InitMeld>,
    Step<GenerateTransformId>,
    While<Always<MeldState, ()>, MeldEpoch<Transform>>,
);

/// Marker implemented only by model-free [`Meld`] flow templates.
pub trait MeldFlow: sealed::Sealed {}

impl<Transform> MeldFlow for Meld<Transform> {}

mod sealed {
    pub trait Sealed {}

    impl<Transform> Sealed for super::Meld<Transform> {}
}
