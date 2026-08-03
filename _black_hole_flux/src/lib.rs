//! Higher-order Jungle flows for quark-inference nuclei and cells.
//!
//! A **Nucleus** composes an input flow, a single quark-inference step, and an
//! output flow into one sequential pipeline.  Given an [`EmissionId`] pointing
//! to an [`Emission<M>`] stored in void, the Nucleus:
//!
//! 1. Runs the **In** flow to produce a (possibly transformed) `EmissionId`.
//! 2. Downloads that emission from void, performs quark inference on the
//!    contained output ID, uploads the result emission, and yields the new `EmissionId`.
//! 3. Passes the resulting `EmissionId` through the **Out** flow.
//!
//! A **Cell** wraps a nucleus flow in an infinite QuZO training loop driven by
//! [`Transmission`] messages from void:
//!
//! 1. **WaitForInitiation** - reads `recv_id` from [`CellState`], downloads a
//!    `Transmission::Initiation`, stores the new `recv_id` in state.
//! 2. **PerturbUp** - perturbs the associated quark's weights upward.
//! 3. **WaitForPropagation** - reads `recv_id` from state, downloads a
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 4. **Nucleus** - runs the nucleus pipeline.
//! 5. **Transmit** - propagates the emission output to the next cell.
//! 6. **PerturbDown** - perturbs the quark's weights downward.
//! 7. **WaitForPropagation** - reads `recv_id` from state, downloads another
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 8. **Nucleus** - runs the nucleus pipeline again.
//! 9. **Transmit** - propagates the emission output to the next cell.
//! 10. **WaitForPotentiation** - reads `recv_id` from state, downloads a
//!     `Transmission::Potentiation`, stores the new `recv_id`, emits loss values.
//! 11. **Optimize** - applies the QuZO optimization update.
//!
//! # State
//!
//! The [`CellState`] type holds `recv_id` (void key of the next
//! [`Transmission`] to download) and `send_id` (void key of the last received
//! send transmission).  Animals that use [`Cell`] as their Journey
//! should use [`CellState`] (or a superset) as their state type.
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
pub mod cell;
pub mod nucleus;
pub mod ops;
pub mod sun;

pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, ObjectId, QuarkIn, QuarkOut, SequenceOutput, Transmission,
};

pub use animal::Progenitor;
pub use cell::action::{
    CellState, Optimize, PerturbDown, PerturbUp, Potentiation, Propagation, QuarkInferStep,
    Transmit, WaitForInitiationAction, WaitForPotentiationAction, WaitForPropagationAction,
};
pub use cell::effect::{
    QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, Transmit as TransmitEffect, WaitForInitiation,
    WaitForPotentiation, WaitForPropagation,
};
pub use nucleus::effect::QuarkInfer;
pub use ops::VoidInferOps;

pub use cell::{Cell, Cytoplasm, Eukaryote, Primordium, Prokaryote};
pub use nucleus::Nucleus;

pub use sun::{EdgeIdsFromList, Spawn, SpawnAnimal, SunState, Tag};

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum NucleusError {
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
