//! Higher-order Jungle flows for quark-inference nuclei and cells.
//!
//! A **Atom** composes an input flow, a single quark-inference step, and an
//! output flow into one sequential pipeline.  Given an [`EmissionId`] pointing
//! to an [`Emission<M>`] stored in void, the Atom:
//!
//! 1. Runs the **In** flow to produce a (possibly transformed) `EmissionId`.
//! 2. Downloads that emission from void, performs quark inference on the
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
//! on quark. Every Atom and QuZO request is routed to that UUID.
//!
//! 1. **PerturbUp** - perturbs the associated quark model's weights upward.
//! 2. **WaitForPropagation** - reads `recv_id` from state, downloads a
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 3. **Atom** - runs the atom pipeline.
//! 4. **Transmit** - propagates the emission output to the next cell.
//! 5. **PerturbDown** - perturbs the quark's weights downward.
//! 6. **WaitForPropagation** - reads `recv_id` from state, downloads a
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 7. **Atom** - runs the atom pipeline again.
//! 8. **Transmit** - propagates the emission output to the next cell.
//! 9. **WaitForPotentiation** - reads `recv_id` from state, downloads a
//!    `Transmission::Potentiation`, stores the new `recv_id`, emits loss values.
//! 10. **Optimize** - applies the QuZO optimization update.
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
//! which guarantees access to void (upload / download), quark inference, and
//! quark perturbation / optimization.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod animal;
pub mod atom;
pub mod cell;
pub mod fusion;
pub mod ops;
pub mod sun;
pub mod twin;

pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, ObjectId, QuarkIn, QuarkOut, SequenceOutput, Transmission,
};

pub use animal::Progenitor;
pub use atom::effect::QuarkInfer;
pub use cell::action::{
    CellState, GenerateModelId, Optimize, PerturbDown, PerturbUp, Potentiation, PrepareAtomInput,
    Propagation, QuarkInferStep, ShutdownModel, StartModel, Transmit, WaitForPotentiationAction,
    WaitForPropagationAction,
};
pub use cell::effect::{
    GenerateModelIdEffect, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, QuarkShutdown,
    QuarkStart, Transmit as TransmitEffect, WaitForPotentiation, WaitForPropagation,
};
pub use fusion::action::{FusionSeed, FusionState};
pub use fusion::{Fusion, FusionEpoch, FusionFlow};
pub use ops::VoidInferOps;

pub use atom::Atom;
pub use cell::{Cell, Cytoplasm, Eukaryote, Primordium, Prokaryote};
pub use twin::action::{LeftStack, RightStack};
pub use twin::Twin;

pub use sun::{
    action::{
        BroadcastPotentiation, BroadcastPotentiationInput, InitializePropagation, NodeIdsFromList,
        ProcessNextNode, PropagationState, SendRootPropagation, Spawn,
    },
    effect::{
        BroadcastPotentiationEffect, BroadcastPotentiationResult, NodeTransmission,
        SendRootPropagationEffect, SendRootPropagationInput, WaitForNodeTransmission,
        WaitForNodeTransmissionInput,
    },
    Binary, BlackHole, Epoch, PendingNotEmpty, PropA, PropAFlow, PropB, PropBFlow,
    PropagationFlows, PropagationLoop, SpawnAnimal, SunAppearance, SunEdgeAppearance, SunInner,
    SunNodeAppearance, SunNodeState, SunState, Unary,
};

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AtomError {
    #[error("quark model start failed: {0}")]
    ModelStart(String),

    #[error("quark model shutdown failed: {0}")]
    ModelShutdown(String),

    #[error("void download failed: {0}")]
    Download(String),

    #[error("quark inference failed: {0}")]
    Inference(String),

    #[error("void upload failed: {0}")]
    Upload(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("quark perturb up failed: {0}")]
    PerturbUp(String),

    #[error("quark perturb down failed: {0}")]
    PerturbDown(String),

    #[error("quark optimize failed: {0}")]
    Optimize(String),

    #[error("transmission error: {0}")]
    Transmission(String),

    #[error("spawn failed: {0}")]
    Spawn(String),
}
