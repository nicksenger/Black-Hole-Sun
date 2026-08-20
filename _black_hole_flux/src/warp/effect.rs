//! Boundary effects for warp observation and perturbation.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;
use uuid::Uuid;

pub use crate::cell::effect::{
    Transmit as TransmitEffect, WaitForPotentiationEffect as WaitForBoundaryPotentiationEffect,
    WaitForPropagationEffect,
};

use crate::ops::SunOps;
use crate::{AtomError, Potentiation};

// ---------------------------------------------------------------------------
// ObserveWarpEffect — observe the appearance of a warp journey
// ---------------------------------------------------------------------------

/// Effect that observes the appearance of a warp journey and decodes it as `Ap`.
pub struct ObserveWarpEffect<Ap>(PhantomData<fn() -> Ap>);

#[jungle::effect(id = 74)]
impl<Ap: Serialize + DeserializeOwned + Send + 'static, J: SunOps> Effect<J>
    for ObserveWarpEffect<Ap>
{
    type In = Uuid;
    type Out = Ap;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        journey_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%journey_id, "observing warp journey appearance");
            let appearance = jungle
                .observe_animal::<Ap>(journey_id)
                .await
                .map_err(AtomError::ObserveWarp)?;
            Ok(appearance)
        }
    }
}

// ---------------------------------------------------------------------------
// PerturbWarpEffect — perturb a warp journey with a potentiation stimulus
// ---------------------------------------------------------------------------

/// Effect that perturbs a warp journey by forwarding a [`Potentiation`] payload.
pub struct PerturbWarpEffect;

#[jungle::effect(id = 75)]
impl<J: SunOps> Effect<J> for PerturbWarpEffect {
    type In = (Uuid, Potentiation);
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (journey_id, potentiation): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                %journey_id,
                loss_up = potentiation.loss_up,
                loss_down = potentiation.loss_down,
                seed = potentiation.seed,
                "perturbing warp journey"
            );
            jungle
                .perturb_animal(journey_id, &potentiation)
                .await
                .map_err(AtomError::PerturbWarp)?;
            Ok(())
        }
    }
}
