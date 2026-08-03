//! Nucleus module - quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use action::QuarkInferStep;

/// A Nucleus composes three sequential stages.
#[derive(Flow)]
pub struct Nucleus<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    In,
    Step<QuarkInferStep<M>>,
    Out,
);

#[derive(Flow)]
pub struct Nucleoli<In, Out>(
    In,
    Out,
);
