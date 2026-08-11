//! Twin module - two-input quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use action::QuarkInferStep;

pub use action::{LeftStack, RightStack};

/// A Twin composes three sequential stages.
///
/// Its input flow is expected to transform a two-input fusion payload from
/// `(Uuid, (EmissionId, EmissionId))` into `(Uuid, EmissionId)` so one
/// quark-inference step can run before handing off to the output flow.
///
/// Use [`LeftStack`] or [`RightStack`] for default "stack and infer" behavior.
#[derive(Flow)]
pub struct TwinWithState<In, Out, M: Serialize + DeserializeOwned + Send + 'static, S>(
    In,
    Step<QuarkInferStep<M, S>>,
    Out,
);

pub type Twin<In, Out, M, S = ()> = TwinWithState<In, Out, M, S>;
