//! Atom module - quark-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use jungle_zoo::backoff::Backoff;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use crate::cell::action::CellState;
use crate::model_config::{DefaultConfig, ModelConfig};
use crate::EmissionId;
use action::QuarkInferStep;

const QUARK_INFER_BACKOFF_INITIAL_DELAY_MS: u64 = 100;
const QUARK_INFER_BACKOFF_MAX_DELAY_MS: u64 = 10_000;
const QUARK_INFER_BACKOFF_MULTIPLIER: u8 = 2;

/// Runs one Atom inference step with automatic retry backoff.
#[derive(Flow)]
pub struct QuarkInferWithBackoff<
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(
    Backoff<
        CellState<S>,
        (Uuid, EmissionId),
        EmissionId,
        Step<QuarkInferStep<M, S, H>>,
        QUARK_INFER_BACKOFF_INITIAL_DELAY_MS,
        QUARK_INFER_BACKOFF_MAX_DELAY_MS,
        QUARK_INFER_BACKOFF_MULTIPLIER,
    >,
);

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
>(In, QuarkInferWithBackoff<M, S, H>, Out);

pub type Atom<In, Out, M, S = (), H = DefaultConfig> = AtomWithState<In, Out, M, S, H>;

#[derive(Flow)]
pub struct NoBackoffAtom<
    In,
    Out,
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(In, QuarkInferWithoutBackoff<M, S, H>, Out);

#[derive(Flow)]
pub struct QuarkInferWithoutBackoff<
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(Step<QuarkInferStep<M, S, H>>);
