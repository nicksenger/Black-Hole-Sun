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
//! A **Cell** wraps a Nucleus in an infinite QuZO training loop driven by
//! [`Transmission`] messages from void:
//!
//! 1. **WaitForInitiation** — reads `recv_id` from [`CellState`], downloads a
//!    `Transmission::Initiation`, stores the new `recv_id` in state.
//! 2. **PerturbUp** — perturbs the associated quark's weights upward.
//! 3. **WaitForPropagation** — reads `recv_id` from state, downloads a
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 4. **In → Nucleus → Out** — runs the full Nucleus pipeline.
//! 5. **Transmit** — propagates the emission output to the next cell.
//! 6. **PerturbDown** — perturbs the quark's weights downward.
//! 7. **WaitForPropagation** — reads `recv_id` from state, downloads another
//!    `Transmission::Propagation`, stores the new `recv_id` and `send_id`, emits the emission ID.
//! 8. **In → Nucleus → Out** — runs the Nucleus pipeline again.
//! 9. **Transmit** — propagates the emission output to the next cell.
//! 10. **WaitForPotentiation** — reads `recv_id` from state, downloads a
//!     `Transmission::Potentiation`, stores the new `recv_id`, emits loss values.
//! 11. **Optimize** — applies the QuZO optimization update.
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

use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
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
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, ObjectId, QuarkIn, QuarkOut, SequenceOutput, Transmission,
};

// Re-export key items from submodules so they're reachable from the crate root.
pub use action::{
    CellState, Optimize, PerturbDown, PerturbUp, Potentiation, Propagation, QuarkInferStep,
    Transmit, WaitForInitiationAction, WaitForPotentiationAction, WaitForPropagationAction,
};
pub use effect::{
    QuarkInfer, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp, Transmit as TransmitEffect,
    WaitForInitiation, WaitForPotentiation, WaitForPropagation,
};
pub use ops::VoidInferOps;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during a quark-inference nucleus step or cell loop.
#[derive(Debug, Error)]
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
}

// ---------------------------------------------------------------------------
// Nucleus higher-order flow
// ---------------------------------------------------------------------------

use action::QuarkInferStep as QuarkInferStep_;

/// A Nucleus composes three sequential stages:
///
/// 1. **In** flow — pre-processes the input [`EmissionId`].
/// 2. **Quark inference** step — downloads, infers, uploads, yields new `EmissionId`.
/// 3. **Out** flow — post-processes the output [`EmissionId`].
#[derive(Flow)]
pub struct Nucleus<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    In,
    Step<QuarkInferStep_<M>>,
    Out,
);

// ---------------------------------------------------------------------------
// Cell higher-order flow
// ---------------------------------------------------------------------------

use action::{
    Optimize as Optimize_, PerturbDown as PerturbDown_, PerturbUp as PerturbUp_,
    Transmit as Transmit_, WaitForInitiationAction as WaitForInitiationAction_,
    WaitForPotentiationAction as WaitForPotentiationAction_,
    WaitForPropagationAction as WaitForPropagationAction_,
};

/// Seed used for the perturb-up step in each Cell iteration.
const CELL_PERTURB_UP_SEED: u64 = 42;

/// A Cell wraps a [`Nucleus`] in an infinite QuZO training loop driven by
/// [`Transmission`] messages from void.
///
/// See module-level documentation for the full iteration sequence.
#[derive(Flow)]
pub struct Cell<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    Step<WaitForInitiationAction_>,
    While<Always<CellState, ()>, CellBody<In, Out, M>>,
);

/// The body of one iteration of a [`Cell`] loop.
///
/// Sequential stages (each iteration):
///
/// PerturbUp → WaitForPropagation → In → QuarkInfer → Out → Transmit →
/// PerturbDown → WaitForPropagation → In → QuarkInfer → Out → Transmit →
/// WaitForPotentiation → Optimize
///
/// The [`CellState`](action::CellState) holds `recv_id` which is read by each
/// wait-for action to know which void key to download, and updated with the
/// next transmission ID after each download completes.  Data payloads
/// (EmissionId, Potentiation) flow through action Output types.
#[derive(Flow)]
pub struct CellBody<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    // Perturb up, then wait for propagation to get first emission
    Step<PerturbUp_<CELL_PERTURB_UP_SEED>>,
    Step<WaitForPropagationAction_>,
    // Run nucleus on the emission from propagation
    Nucleus<In, Out, M>,
    // Transmit emission output, perturb down
    Step<Transmit_>,
    Step<PerturbDown_>,
    // Wait for propagation to get second emission
    Step<WaitForPropagationAction_>,
    // Run nucleus on the emission from propagation
    Nucleus<In, Out, M>,
    // Transmit emission output, wait for potentiation to get losses
    Step<Transmit_>,
    Step<WaitForPotentiationAction_>,
    // Optimize with the loss values (returns unit)
    Step<Optimize_>,
);
