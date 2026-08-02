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
//! A **Cell** wraps a Nucleus in an infinite QuZO training loop: perturb up →
//! infer → perturb down → infer → optimize, repeating indefinitely.  Between
//! each inference step the Cell awaits an external jungle perturbation carrying
//! the next [`EmissionId`], and after both inferences it awaits a perturbation
//! carrying the `(loss_up, loss_down)` pair for optimization.
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
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutputId, InferenceOutput,
    InferenceRequest, LogitEntry, ObjectId, QuarkIn, QuarkOut, SequenceOutput,
};

// Re-export key items from submodules so they're reachable from the crate root.
pub use action::{
    ClaimLossAction, ClaimPerturbationAction, Optimize, PerturbDown, PerturbUp, QuarkInferStep,
};
pub use effect::{
    ClaimLoss, ClaimPerturbation, QuarkInfer, QuarkOptimize, QuarkPerturbDown, QuarkPerturbUp,
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

    #[error("failed to claim or deserialize perturbation: {0}")]
    Claim(String),
}

// ---------------------------------------------------------------------------
// Nucleus higher-order flow
// ---------------------------------------------------------------------------

use action::QuarkInferStep as QuarkInferStep_;

/// A Nucleus composes three sequential stages:
///
/// 1. **In** flow — pre-processes the input [`EmissionId`] (e.g., transforms
///    or validates emission data).  Takes `EmissionId`, produces `EmissionId`.
/// 2. **Quark inference** step — downloads the emission from void, runs quark
///    inference on its output ID, uploads the result emission, and yields a new `EmissionId`.
/// 3. **Out** flow — post-processes the output [`EmissionId`] (e.g., stores
///    references or triggers downstream work).  Takes `EmissionId`, produces `EmissionId`.
///
/// # Type parameters
///
/// * `In` — the input flow (must accept `EmissionId` and produce `EmissionId`).
/// * `Out` — the output flow (must accept `EmissionId` and produce `EmissionId`).
/// * `M` — the metadata type stored inside each [`Emission<M>`].
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
    ClaimLossAction as ClaimLossAction_,
    ClaimPerturbationAction as ClaimPerturbationAction_,
    Optimize as Optimize_,
    PerturbDown as PerturbDown_,
    PerturbUp as PerturbUp_,
};

/// Seed used for the perturb-up step in each Cell iteration.
const CELL_PERTURB_UP_SEED: u64 = 42;

/// A Cell wraps a [`Nucleus`] in an infinite QuZO training loop.
///
/// Each iteration performs the following sequence:
///
/// 1. **PerturbUp** — perturbs the associated quark's weights upward.
/// 2. **Claim** — awaits a jungle perturbation containing an [`EmissionId`].
/// 3. **In → Nucleus → Out** — runs the full Nucleus pipeline with the received `EmissionId`.
///    (Uses the same In/Out/M as provided to the outer Cell.)
/// 4. **PerturbDown** — perturbs the quark's weights downward.
/// 5. **Claim** — awaits another jungle perturbation containing an [`EmissionId`].
/// 6. **In → Nucleus → Out** — runs the Nucleus pipeline again.
/// 7. **ClaimLoss** — awaits a jungle perturbation containing `(loss_up, loss_down)`.
/// 8. **Optimize** — applies the QuZO optimization update with the received losses.
///
/// The loop then repeats from step 1 indefinitely.
///
/// # Type parameters
///
/// * `In` — the input flow shared with [`Nucleus`] (must accept/produce `EmissionId`).
/// * `Out` — the output flow shared with [`Nucleus`] (must accept/produce `EmissionId`).
/// * `M` — the metadata type stored inside each [`Emission<M>`].
#[derive(Flow)]
pub struct Cell<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    While<Always<(), ()>, CellBody<In, Out, M>>,
);

/// The body of one iteration of a [`Cell`] loop.
///
/// Each iteration is self-contained: it takes `()` and produces `()`.
/// The [`EmissionId`] values are produced by internal Claim steps that await
/// external perturbations.
///
/// Sequential stages (each iteration):
///
/// PerturbUp → ClaimEmissionId → In → QuarkInfer → Out → Discard →
/// PerturbDown → ClaimEmissionId → In → QuarkInfer → Out → Discard →
/// ClaimLoss → Optimize
#[derive(Flow)]
pub struct CellBody<In, Out, M: Serialize + DeserializeOwned + Send + 'static>(
    // --- Up phase ---
    // Step 1: PerturbUp — () → ()
    Step<PerturbUp_<CELL_PERTURB_UP_SEED>>,
    // Step 2: Claim EmissionId — () → EmissionId
    Step<ClaimPerturbationAction_>,
    // Step 3: In flow — EmissionId → EmissionId
    In,
    // Step 4: Quark inference (up) — EmissionId → EmissionId
    Step<QuarkInferStep_<M>>,
    // Step 5: Out flow — EmissionId → EmissionId
    Out,
    // Step 6: Discard EmissionId, produce () for PerturbDown
    Step<DiscardEmission>,
    // --- Down phase ---
    // Step 7: PerturbDown — () → ()
    Step<PerturbDown_>,
    // Step 8: Claim EmissionId — () → EmissionId
    Step<ClaimPerturbationAction_>,
    // Step 9: In flow — EmissionId → EmissionId
    In,
    // Step 10: Quark inference (down) — EmissionId → EmissionId
    Step<QuarkInferStep_<M>>,
    // Step 11: Out flow — EmissionId → EmissionId
    Out,
    // Step 12: Discard EmissionId, produce () for ClaimLoss
    Step<DiscardEmission>,
    // --- Optimize phase ---
    // Step 13: Claim loss tuple — () → (f32, f32)
    Step<ClaimLossAction_>,
    // Step 14: Optimize — (f32, f32) → ()
    Step<Optimize_>,
);

/// Discards an [`EmissionId`] and produces `()`.
///
/// Used to bridge between EmissionId-producing stages (In/Out flows) and
/// unit-input stages (PerturbDown, ClaimLoss) in the Cell loop.
pub struct DiscardEmission;

#[jungle::action]
impl Action for DiscardEmission {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(_state: &(), _input: Self::Input) -> () {}

    fn absorb(
        _state: &mut (),
        _output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        Ok(())
    }
}
