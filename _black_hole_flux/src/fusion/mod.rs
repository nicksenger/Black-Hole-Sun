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

use crate::model_config::{DefaultConfig, ModelConfig};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;

pub use action::{
    AdvanceFusionGradientStep, BeginFusionGradientAccumulation, FusionOptimize, FusionPerturbDown,
    FusionPerturbUp, FusionQuarkInferStep, FusionSeed, FusionStartModel, FusionState,
    FusionTransmit, GenerateTransformId, InitFusion, PrepareTransformInput,
    WaitForFusionPotentiationAction, WaitForFusionPotentiationForOptimize,
    WaitForFusionPropagationAction,
};

pub use effect::{GenerateTransformIdEffect, WaitForFusionPotentiation, WaitForFusionPropagation};

/// Predicate that keeps running fusion microsteps until `grad_steps` is reached.
pub struct HasPendingFusionGradientStep;

impl Predicate<(&FusionState, &())> for HasPendingFusionGradientStep {
    fn eval((state, _): &(&FusionState, &())) -> bool {
        state.grad_step < state.grad_steps.max(1)
    }
}

/// One single-pass model-free fusion microstep.
#[derive(Flow)]
pub struct FusionPropagationMicrostep<Transform>(
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionTransmit>,
    Step<AdvanceFusionGradientStep>,
);

/// One complete model-free accumulation epoch.
#[derive(Flow)]
pub struct FusionEpoch<Transform>(
    Step<BeginFusionGradientAccumulation>,
    While<HasPendingFusionGradientStep, FusionPropagationMicrostep<Transform>>,
    Step<BeginFusionGradientAccumulation>,
    While<HasPendingFusionGradientStep, FusionPropagationMicrostep<Transform>>,
    Step<WaitForFusionPotentiationAction>,
);

/// Infinite model-free fusion loop driven by two staged mailbox chains.
#[derive(Flow)]
pub struct Fusion<Transform>(
    Step<InitFusion>,
    Step<GenerateTransformId>,
    While<Always<FusionState, ()>, FusionEpoch<Transform>>,
);

/// One single-pass model-aware fusion microstep.
#[derive(Flow)]
pub struct QuzoFusionPropagationMicrostep<
    Transform,
    M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
>(
    Step<WaitForFusionPropagationAction>,
    Step<PrepareTransformInput>,
    Transform,
    Step<FusionQuarkInferStep<M>>,
    Step<FusionTransmit>,
    Step<AdvanceFusionGradientStep>,
);

/// One complete model-aware accumulation epoch.
#[derive(Flow)]
pub struct QuzoFusionEpoch<
    Transform,
    M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
>(
    Step<BeginFusionGradientAccumulation>,
    Step<FusionPerturbUp>,
    While<HasPendingFusionGradientStep, QuzoFusionPropagationMicrostep<Transform, M>>,
    Step<BeginFusionGradientAccumulation>,
    Step<FusionPerturbDown>,
    While<HasPendingFusionGradientStep, QuzoFusionPropagationMicrostep<Transform, M>>,
    Step<WaitForFusionPotentiationForOptimize>,
    Step<FusionOptimize>,
);

/// Infinite two-input QuZO loop for model-aware twin transforms.
#[derive(Flow)]
pub struct QuzoFusionWithModelConfig<
    Transform,
    M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    H: ModelConfig,
>(
    Step<InitFusion>,
    Step<GenerateTransformId>,
    Step<FusionStartModel<H>>,
    While<Always<FusionState, ()>, QuzoFusionEpoch<Transform, M>>,
);

pub type QuzoFusion<Transform, M, H = DefaultConfig> = QuzoFusionWithModelConfig<Transform, M, H>;

/// Marker implemented only by model-free [`Fusion`] flow templates.
pub trait FusionFlow: sealed::Sealed {}

impl<Transform> FusionFlow for Fusion<Transform> {}
impl<
        Transform,
        M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
        H: ModelConfig,
    > FusionFlow for QuzoFusionWithModelConfig<Transform, M, H>
{
}

mod sealed {
    use crate::model_config::ModelConfig;

    pub trait Sealed {}

    impl<Transform> Sealed for super::Fusion<Transform> {}
    impl<
            Transform,
            M: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
            H: ModelConfig,
        > Sealed for super::QuzoFusionWithModelConfig<Transform, M, H>
    {
    }
}
