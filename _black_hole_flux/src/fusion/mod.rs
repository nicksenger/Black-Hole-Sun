//! Model-free two-input fusion protocol.
//!
//! A [`Fusion`] journey owns two independent mailbox chains. For each of the
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
    FusionSeed, FusionState, FusionTransmit, GenerateTransformId, InitFusion,
    PrepareTransformInput, WaitForFusionPotentiationAction, WaitForFusionPropagationAction,
};

pub use effect::{GenerateTransformIdEffect, WaitForFusionPotentiation, WaitForFusionPropagation};

/// One complete two-pass fusion epoch.
#[derive(Flow)]
pub struct FusionEpoch<Transform>(
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionTransmit>,
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionTransmit>,
    Step<WaitForFusionPotentiationAction>,
);

/// Infinite model-free fusion loop driven by two input-port mailbox chains.
#[derive(Flow)]
pub struct Fusion<Transform>(
    Step<InitFusion>,
    Step<GenerateTransformId>,
    While<Always<FusionState, ()>, FusionEpoch<Transform>>,
);

/// Marker implemented only by model-free [`Fusion`] flow templates.
pub trait FusionFlow: sealed::Sealed {}

impl<Transform> FusionFlow for Fusion<Transform> {}

mod sealed {
    pub trait Sealed {}

    impl<Transform> Sealed for super::Fusion<Transform> {}
}
