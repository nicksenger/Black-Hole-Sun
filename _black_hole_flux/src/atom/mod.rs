//! Atom module - quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use action::QuarkInferStep;

/// An Atom composes three sequential stages.
///
/// Its input flow receives `(model_id, emission_id)` so inference is routed to
/// the model instance owned by the surrounding Cell.
#[derive(Flow)]
pub struct AtomWithState<In, Out, M: Serialize + DeserializeOwned + Send + 'static, S>(
    In,
    Step<QuarkInferStep<M, S>>,
    Out,
);

pub type Atom<In, Out, M, S = ()> = AtomWithState<In, Out, M, S>;
