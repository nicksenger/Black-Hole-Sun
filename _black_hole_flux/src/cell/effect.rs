//! Cell effects for perturbation, optimization, transmission, and waiting.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use black_hole_spec::{ObjectId, Transmission};

use super::action::{Potentiation, Propagation};
use crate::model_config::{DefaultConfig, ModelConfig};
use crate::ops::VoidInferOps;
use crate::AtomError;

// ---------------------------------------------------------------------------
// Model instance lifecycle
// ---------------------------------------------------------------------------

pub struct GenerateModelIdEffect;

#[jungle::effect(id = 59)]
impl<J> Effect<J> for GenerateModelIdEffect {
    type In = ();
    type Out = Uuid;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async { Ok(Uuid::new_v4()) }
    }
}

pub struct MassStart<H = DefaultConfig>(PhantomData<fn() -> H>);

#[jungle::effect(id = 60)]
impl<H: ModelConfig, J: VoidInferOps> Effect<J> for MassStart<H> {
    type In = Uuid;
    type Out = bool;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "starting mass model instance");
            jungle
                .start_model(model_id, H::mass_model_config())
                .await
                .map_err(AtomError::ModelStart)?;
            let params = jungle
                .query_model_params(model_id)
                .await
                .map_err(AtomError::ModelStart)?;
            Ok(params.is_frozen)
        }
    }
}

pub struct MassShutdown;

#[jungle::effect(id = 61)]
impl<J: VoidInferOps> Effect<J> for MassShutdown {
    type In = Uuid;
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "shutting down mass model instance");
            jungle
                .shutdown_model(model_id)
                .await
                .map_err(AtomError::ModelShutdown)
        }
    }
}

// ---------------------------------------------------------------------------
// MassPerturbUp — perturb mass weights in the positive direction
// ---------------------------------------------------------------------------

pub struct MassPerturbUp;

#[jungle::effect(id = 62)]
impl<J: VoidInferOps> Effect<J> for MassPerturbUp {
    type In = (Uuid, u64);
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (model_id, seed): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, seed, "perturbing mass weights up");
            jungle
                .perturb_up(model_id, seed)
                .await
                .map_err(AtomError::PerturbUp)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// MassPerturbDown — perturb mass weights in the negative direction
// ---------------------------------------------------------------------------

pub struct MassPerturbDown;

#[jungle::effect(id = 63)]
impl<J: VoidInferOps> Effect<J> for MassPerturbDown {
    type In = Uuid;
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "perturbing mass weights down");
            jungle
                .perturb_down(model_id)
                .await
                .map_err(AtomError::PerturbDown)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// MassOptimize — apply QuZO optimization update
// ---------------------------------------------------------------------------

pub struct MassOptimize;

#[jungle::effect(id = 64)]
impl<J: VoidInferOps> Effect<J> for MassOptimize {
    type In = (Uuid, Potentiation);
    type Out = bool;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (model_id, potentiation): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                %model_id,
                loss_up = potentiation.loss_up,
                loss_down = potentiation.loss_down,
                seed = potentiation.seed,
                "applying mass optimization"
            );
            jungle
                .optimize(model_id, potentiation.loss_up, potentiation.loss_down)
                .await
                .map_err(AtomError::Optimize)?;
            let params = jungle
                .query_model_params(model_id)
                .await
                .map_err(AtomError::Optimize)?;
            Ok(params.is_frozen)
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForPropagationEffect — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Effect that waits for a [`Transmission::Propagation`] at the given [`ObjectId`].
///
/// Downloads the transmission from void, extracts the emission ID to process,
/// the next receive transmission ID for state threading, and the send ID.
pub struct WaitForPropagationEffect;

#[jungle::effect(id = 65)]
impl<J: VoidInferOps> Effect<J> for WaitForPropagationEffect {
    type In = ObjectId;
    type Out = Propagation;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%id, "awaiting propagation transmission");
            let transmission = jungle
                .wait_for_transmission(id)
                .await
                .map_err(AtomError::Transmission)?;
            match transmission {
                Transmission::Propagation {
                    emission_id,
                    recv,
                    send,
                } => {
                    debug!(emission_id = %emission_id.0, recv = %recv, send = %send, "propagation received");
                    Ok(Propagation {
                        emission_id,
                        recv_id: recv,
                        send_id: send,
                    })
                }
                other => {
                    let msg = format!("expected Propagation, got {:?}", other);
                    debug!("propagation failed: {msg}");
                    Err(AtomError::Transmission(msg))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForPotentiationEffect — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Effect that waits for a [`Transmission::Potentiation`] at the given [`ObjectId`].
///
/// Downloads the transmission from void, constructs a [`Potentiation`] payload
/// and returns it alongside the next transmission ID for state threading.
pub struct WaitForPotentiationEffect;

#[jungle::effect(id = 66)]
impl<J: VoidInferOps> Effect<J> for WaitForPotentiationEffect {
    type In = ObjectId;
    type Out = (Potentiation, ObjectId);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%id, "awaiting potentiation transmission");
            let transmission = jungle
                .wait_for_transmission(id)
                .await
                .map_err(AtomError::Transmission)?;
            match transmission {
                Transmission::Potentiation { potentiation, recv } => {
                    debug!(
                        loss_up = potentiation.loss_up,
                        loss_down = potentiation.loss_down,
                        seed = potentiation.seed,
                        %recv,
                        "potentiation received"
                    );
                    Ok((potentiation, recv))
                }
                other => {
                    let msg = format!("expected Potentiation, got {:?}", other);
                    debug!("potentiation failed: {msg}");
                    Err(AtomError::Transmission(msg))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transmit — propagate an emission to the next cell
// ---------------------------------------------------------------------------

/// Effect that propagates an [`EmissionId`](black_hole_spec::EmissionId) to the next cell.
pub struct Transmit;

#[jungle::effect(id = 67)]
impl<J: VoidInferOps> Effect<J> for Transmit {
    type In = (black_hole_spec::EmissionId, ObjectId);
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let (emission_id, send_id) = input;
            debug!(emission_id = %emission_id.0, %send_id, "transmitting emission to next cell");
            jungle
                .transmit(emission_id, send_id)
                .await
                .map_err(AtomError::Transmission)?;
            Ok(())
        }
    }
}
