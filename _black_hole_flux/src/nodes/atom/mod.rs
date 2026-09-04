//! Atom module - mass-inference pipeline components.

pub mod action;
pub mod effect;

use jungle_sdk::prelude::*;
use jungle_zoo::backoff::Backoff;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use crate::nodes::cell::action::CellState;
use crate::mass::{DefaultConfig, ModelConfig};
use crate::EmissionId;
use action::MassInferStep;
use action::OperationMassInferStep;
use black_hole_spec::TensorContract;

const MASS_INFER_BACKOFF_INITIAL_DELAY_MS: u64 = 100;
const MASS_INFER_BACKOFF_MAX_DELAY_MS: u64 = 10_000;
const MASS_INFER_BACKOFF_MULTIPLIER: u8 = 2;

/// Runs one Atom inference step with automatic retry backoff.
#[derive(Flow)]
pub struct MassInferWithBackoff<M: Serialize + DeserializeOwned + Send + 'static, S, H: ModelConfig>(
    Backoff<
        CellState<S>,
        (Uuid, EmissionId),
        EmissionId,
        Step<MassInferStep<M, S, H>>,
        MASS_INFER_BACKOFF_INITIAL_DELAY_MS,
        MASS_INFER_BACKOFF_MAX_DELAY_MS,
        MASS_INFER_BACKOFF_MULTIPLIER,
    >,
);

/// An Atom composes three sequential stages.
///
/// Its input flow receives `(model_id, emission_id)` so inference is routed to
/// the model instance owned by the surrounding Cell.
///
/// `H` is the compile-time model configuration type used by the surrounding
/// [`Cell`](crate::Cell) to configure per-instance mass startup defaults.
#[derive(Flow)]
pub struct AtomWithState<
    In,
    Out,
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(In, MassInferWithBackoff<M, S, H>, Out);

pub type Atom<In, Out, M, S = (), H = DefaultConfig> = AtomWithState<In, Out, M, S, H>;

/// Generic operation Atom. Its surrounding flows must produce and consume
/// emission IDs matching the operation contract's input/output bundles.
#[derive(Flow)]
pub struct OperationAtomWithState<
    In,
    Out,
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: TensorContract<Input: Send, Output: Send> + Send + Sync + 'static,
    S,
>(In, Step<OperationMassInferStep<M, Op, S>>, Out);

pub type OperationAtom<In, Out, M, Op, S = ()> = OperationAtomWithState<In, Out, M, Op, S>;

#[derive(Flow)]
pub struct NoBackoffAtom<
    In,
    Out,
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(In, MassInferWithoutBackoff<M, S, H>, Out);

#[derive(Flow)]
pub struct MassInferWithoutBackoff<
    M: Serialize + DeserializeOwned + Send + 'static,
    S,
    H: ModelConfig,
>(Step<MassInferStep<M, S, H>>);
