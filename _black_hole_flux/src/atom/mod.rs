//! Atom module - quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use action::QuarkInferStep;

/// A Atom composes three sequential stages.
#[derive(Flow)]
pub struct Atom<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    In,
    Step<QuarkInferStep<M>>,
    Out,
);
