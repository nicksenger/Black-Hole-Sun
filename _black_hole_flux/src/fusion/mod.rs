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
    FusionOptimize, FusionPerturbDown, FusionPerturbUp, FusionQuarkInferStep, FusionSeed,
    FusionStartModel, FusionState, FusionTransmit, GenerateTransformId, InitFusion,
    PrepareTransformInput, WaitForFusionPotentiationAction, WaitForFusionPotentiationForOptimize,
    WaitForFusionPropagationAction,
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

/// One complete model-aware two-pass fusion epoch.
#[derive(Flow)]
pub struct QuzoFusionEpoch<
    Transform,
    M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
>(
    Step<FusionPerturbUp>,
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionQuarkInferStep<M>>,
    Step<FusionTransmit>,
    Step<FusionPerturbDown>,
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionQuarkInferStep<M>>,
    Step<FusionTransmit>,
    Step<WaitForFusionPotentiationForOptimize>,
    Step<FusionOptimize>,
);

/// Infinite two-input QuZO loop for model-aware twin transforms.
#[derive(Flow)]
pub struct QuzoFusion<Transform, M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static>(
    Step<InitFusion>,
    Step<GenerateTransformId>,
    Step<FusionStartModel>,
    While<Always<FusionState, ()>, QuzoFusionEpoch<Transform, M>>,
);

/// Marker implemented only by model-free [`Fusion`] flow templates.
pub trait FusionFlow: sealed::Sealed {}

impl<Transform> FusionFlow for Fusion<Transform> {}
impl<Transform, M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> FusionFlow
    for QuzoFusion<Transform, M>
{
}

mod sealed {
    pub trait Sealed {}

    impl<Transform> Sealed for super::Fusion<Transform> {}
    impl<Transform, M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static> Sealed
        for super::QuzoFusion<Transform, M>
    {
    }
}
