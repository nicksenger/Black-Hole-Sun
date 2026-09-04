//! Boundary module - higher-order flows for warp boundaries.

pub mod action;
pub mod effect;

use action::{
    AdvanceGradientStep as AdvanceGradientStep_,
    BeginGradientAccumulation as BeginGradientAccumulation_, InitRecvId as InitRecvId_,
    ObserveWarp, PerturbWarp, Transmit as Transmit_, WaitForPotentiation as WaitForPotentiation_,
    WaitForPropagation as WaitForPropagation_,
};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Boundary — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Boundary`](crate::Boundary) journey.
///
/// Animals that use [`Boundary`](crate::Boundary) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID. The generic `S` payload is available to user flows via
/// [`BoundaryState::inner`] and defaults to `()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoundaryState<S = ()> {
    /// Stable ID of the mass model instance owned by this cell.
    pub model_id: Uuid,
    /// Stable ID of the warp journey associated with this boundary.
    pub warp_journey_id: Uuid,
    /// Void key of the next [`Transmission`](black_hole_type::Transmission) to download.
    pub recv_id: black_hole_type::ObjectId,
    /// Void key of the next [`Transmission`](black_hole_type::Transmission) to upload.
    pub send_id: black_hole_type::ObjectId,
    /// Random seed passed to the perturb-up step each iteration.
    pub perturbation_seed: u64,
    /// Last known frozen status for this model instance.
    #[serde(default)]
    pub is_frozen: bool,
    /// Number of infer/transmit microsteps per perturbation phase.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    /// Number of completed microsteps in the current propagation phase.
    #[serde(default)]
    pub grad_step: usize,
    /// User-provided state threaded through all cell actions.
    #[serde(default)]
    pub inner: S,
}
fn default_gradient_accumulation_steps() -> usize {
    1
}

/// Predicate that keeps running boundary microsteps until `grad_steps` is reached.
pub struct HasPendingGradientStep<S>(std::marker::PhantomData<fn() -> S>);

impl<S> Predicate<(&BoundaryState<S>, &())> for HasPendingGradientStep<S> {
    fn eval((state, _): &(&BoundaryState<S>, &())) -> bool {
        state.grad_step < state.grad_steps.max(1)
    }
}

/// A model-free boundary loop that behaves like a unary cell around `N`.
#[derive(Flow)]
pub struct NoModelBoundaryWithState<N, S: Serialize + DeserializeOwned + Send + 'static>(
    Step<InitRecvId_<S>>,
    While<Always<BoundaryState<S>, ()>, InnerWithState<N, S>>,
);

pub type NoModelBoundary<N, S = ()> = NoModelBoundaryWithState<N, S>;
pub type Boundary<N, S = ()> = NoModelBoundary<N, S>;

/// The body of one boundary loop iteration.
#[derive(Flow)]
pub struct InnerWithState<N, S: Serialize + DeserializeOwned + Send + 'static>(
    Step<BeginGradientAccumulation_<S>>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<BeginGradientAccumulation_<S>>,
    While<HasPendingGradientStep<S>, InnerPropagationMicrostepWithState<N, S>>,
    Step<WaitForPotentiation_<S>>,
    Step<PerturbWarp<S>>, // Forwards the Potentiation to the WarpAnimal (via perturb)
);

/// One boundary propagation/transmit microstep around `N`.
#[derive(Flow)]
pub struct InnerPropagationMicrostepWithState<N, S: Serialize + DeserializeOwned + Send + 'static>(
    Step<WaitForPropagation_<S>>,
    Step<ObserveWarp<S>>, // Queries the WarpAnimal Appearance and updates the state
    N,
    Step<Transmit_<S>>,
    Step<AdvanceGradientStep_<S>>,
);

pub type Inner<N, S = ()> = InnerWithState<N, S>;
pub type InnerPropagationMicrostep<N, S = ()> = InnerPropagationMicrostepWithState<N, S>;
