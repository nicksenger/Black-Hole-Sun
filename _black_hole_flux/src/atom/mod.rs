//! Atom module - quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::model_config::{DefaultConfig, ModelConfig};
use action::QuarkInferStep;

/// An Atom composes three sequential stages.
///
/// Its input flow receives `(model_id, emission_id)` so inference is routed to
/// the model instance owned by the surrounding Cell.
///
/// `H` is the compile-time model configuration type used by the surrounding
/// [`Cell`](crate::Cell) to configure per-instance quark startup defaults.
#[derive(Flow)]
pub struct AtomWithState<
    In,
    Out,
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(In, Step<QuarkInferStep<M, S, H>>, Out);

pub type Atom<In, Out, M, S = (), H = DefaultConfig> = AtomWithState<In, Out, M, S, H>;
