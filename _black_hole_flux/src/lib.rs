//! Higher-order Jungle flows for mass-inference nuclei and cells.
//!
//! A **Atom** composes an input flow, a single mass-inference step, and an
//! output flow into one sequential pipeline.  Given an [`EmissionId`] pointing
//! to an [`Emission<M>`] stored in void, the Atom:
//!
//! 1. Runs the **In** flow to produce a (possibly transformed) `EmissionId`.
//! 2. Downloads that emission from void, performs mass inference on the
//!    contained output ID, uploads the result emission, and yields the new `EmissionId`.
//! 3. Passes the resulting `EmissionId` through the **Out** flow.
//!
//! A **Twin** is the fusion parallel of Atom. It follows the same three-stage
//! shape, but its **In** flow starts from a pair and converts
//! `(Uuid, (EmissionId, EmissionId))` into `(Uuid, EmissionId)` before
//! inference.
//!
//! A **Cell** wraps a atom flow in an infinite QuZO training loop driven by
//! [`Transmission`] messages from void:
//!
//! Each Cell first generates a stable model UUID and starts that model instance
//! on mass. Every Atom and QuZO request is routed to that UUID.
//!
//! 1. **PerturbUp** - perturbs the associated mass model's weights upward.
//! 2. **N × (WaitForPropagation -> Atom -> Transmit)** - runs the first
//!    propagation phase `N` times, resetting model runtime state after each
//!    inference.
//! 3. **PerturbDown** - perturbs the mass's weights downward.
//! 4. **N × (WaitForPropagation -> Atom -> Transmit)** - runs the second
//!    propagation phase `N` times.
//! 5. **WaitForPotentiation** - reads `recv_id` from state, downloads a
//!    `Transmission::Potentiation`, stores the new `recv_id`, and emits a
//!    [`Potentiation`] payload.
//! 6. **Optimize** - applies the QuZO optimization update.
//!
//! # State
//!
//! The [`CellState`] type holds the Cell's stable `model_id`, `recv_id` (void
//! key of the next [`Transmission`] to download), and `send_id` (void key of
//! the last received send transmission). `CellState` is generic and includes
//! an `inner` payload for user-provided state (`CellState<T>` defaults to
//! `CellState<()>`). Animals that use [`Cell`] as their Journey should use
//! [`CellState`] (or a superset) as their state type.
//!
//! # Flow pattern
//!
//! The `recv_id` is threaded through [`CellState`] (state-mediated), while
//! [`EmissionId`] and [`Potentiation`] are passed through action Output types.
//! The Cell while loop operates over [`CellState`] so that wait-for actions
//! can access and mutate `recv_id`.
//!
//! # Trait requirement
//!
//! The Jungle instance supplied at runtime must implement [`VoidInferOps`],
//! which guarantees access to void (upload / download), mass inference, and
//! mass perturbation / optimization.
#![allow(clippy::manual_async_fn)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod animal;
pub mod atom;
pub mod cell;
pub mod fusion;
pub mod model_config;
pub mod ops;
pub mod ray;
pub mod sun;
pub mod twin;

pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, MassErrorFeedbackConfig, MassErrorFeedbackMode, MassIn,
    MassModelConfig, MassModelParams, MassOut, ObjectId, SequenceOutput, Transmission,
};

pub use animal::Progenitor;
pub use atom::effect::MassInfer;
pub use cell::action::{
    AdvanceGradientStep, BeginGradientAccumulation, CellState, GenerateModelId, Init as CellInit,
    InitRecvId, MassInferStep, Optimize, PerturbDown, PerturbUp, Potentiation, PrepareAtomInput,
    Propagation, ShutdownModel, StartModel, Transmit, WaitForPotentiation, WaitForPropagation,
};
pub use cell::effect::{
    GenerateModelIdEffect, MassOptimize, MassPerturbDown, MassPerturbUp, MassShutdown, MassStart,
    Transmit as TransmitEffect, WaitForPotentiationEffect, WaitForPropagationEffect,
};
pub use fusion::action::{FusionSeed, FusionState};
pub use fusion::{
    Fusion, FusionEpoch, FusionFlow, QuzoFusion, QuzoFusionEpoch, QuzoFusionWithModelConfig,
};
pub use model_config::{
    DefaultConfig, ErrorFeedbackPolicy, ModelConfig, NoErrorFeedback, NoOscillation,
    OscillationSchedule,
};
pub use ops::VoidInferOps;
pub use ray::Ray;

pub use atom::Atom;
pub use atom::NoBackoffAtom;
pub use cell::{Cell, Primordium};
pub use twin::action::{LeftStack, RandStack, RightStack};
pub use twin::Twin;

pub use sun::{
    action::{
        BroadcastPotentiation, BroadcastPotentiationInput, InitializePropagation, NodeIdsFromList,
        ProcessNextNode, PropagationState, SendRootPropagation, Spawn,
    },
    effect::{
        BroadcastPotentiationEffect, BroadcastPotentiationResult, NodeTransmission,
        SendRootPropagationEffect, SendRootPropagationInput, WaitForNodeTransmissionEffect,
        WaitForNodeTransmissionInput,
    },
    Binary, BlackHole, Epoch, Manifest, PendingNotEmpty, PropA, PropAFlow, PropB, PropBFlow,
    PropagationFlows, PropagationLoop, SpawnAnimal, StatelessManifest, SunAppearance,
    SunEdgeAppearance, SunInner, SunNodeAppearance, SunNodeState, SunState, Unary,
};

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AtomError {
    #[error("mass model start failed: {0}")]
    ModelStart(String),

    #[error("mass model shutdown failed: {0}")]
    ModelShutdown(String),

    #[error("void download failed: {0}")]
    Download(String),

    #[error("mass inference failed: {0}")]
    Inference(String),

    #[error("mass model reset failed: {0}")]
    ModelReset(String),

    #[error("void upload failed: {0}")]
    Upload(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("mass perturb up failed: {0}")]
    PerturbUp(String),

    #[error("mass perturb down failed: {0}")]
    PerturbDown(String),

    #[error("mass optimize failed: {0}")]
    Optimize(String),

    #[error("transmission error: {0}")]
    Transmission(String),

    #[error("spawn failed: {0}")]
    Spawn(String),
}
