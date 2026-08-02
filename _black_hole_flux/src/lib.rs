//! Higher-order Jungle flows for quark-inference cells.
//!
//! A **Cell** composes an input flow, a single quark-inference step, and an
//! output flow into one sequential pipeline.  Given an [`EmissionId`] pointing
//! to an [`Emission<M>`] stored in void, the Cell:
//!
//! 1. Runs the **In** flow to produce a (possibly transformed) `EmissionId`.
//! 2. Downloads that emission from void, performs quark inference on the
//!    contained sequences, uploads the result, and yields the new `EmissionId`.
//! 3. Passes the resulting `EmissionId` through the **Out** flow.
//!
//! # Trait requirement
//!
//! The Jungle instance supplied at runtime must implement [`VoidInferOps`],
//! which guarantees access to void (upload / download) and quark inference.


use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Submodules
// ---------------------------------------------------------------------------

pub mod action;
pub mod effect;
pub mod ops;

// ---------------------------------------------------------------------------
// Re-exports — keep common spec types handy at the crate root
// ---------------------------------------------------------------------------

pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceRequest, LogitEntry,
    ObjectId, PredictedToken, QuarkIn, QuarkOut, SequenceOutput,
};

// Re-export key items from submodules so they're reachable from the crate root.
pub use action::QuarkInferStep;
pub use effect::QuarkInfer;
pub use ops::VoidInferOps;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during a quark-inference cell step.
#[derive(Debug, Error)]
pub enum CellError {
    #[error("void download failed: {0}")]
    Download(String),

    #[error("quark inference failed: {0}")]
    Inference(String),

    #[error("void upload failed: {0}")]
    Upload(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),
}

// ---------------------------------------------------------------------------
// Cell higher-order flow
// ---------------------------------------------------------------------------

use action::QuarkInferStep as QuarkInferStep_;

/// A Cell composes three sequential stages:
///
/// 1. **In** flow — pre-processes the input [`EmissionId`] (e.g., transforms
///    or validates emission data).  Takes `EmissionId`, produces `EmissionId`.
/// 2. **Quark inference** step — downloads the emission from void, runs quark
///    inference on its sequences, uploads the result, yields a new `EmissionId`.
/// 3. **Out** flow — post-processes the output [`EmissionId`] (e.g., stores
///    references or triggers downstream work).  Takes `EmissionId`, produces
///    `EmissionId`.
///
/// # Type parameters
///
/// * `In` — the input flow (must accept `EmissionId` and produce `EmissionId`).
/// * `Out` — the output flow (must accept `EmissionId` and produce `EmissionId`).
/// * `M` — the metadata type stored inside each [`Emission<M>`].
#[derive(Flow)]
pub struct Cell<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    In,
    Step<QuarkInferStep_<M>>,
    Out,
);
