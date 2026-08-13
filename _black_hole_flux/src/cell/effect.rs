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

impl<J> EffectSchema<J> for GenerateModelIdEffect {
    type Id = u64;
    type In = ();
    type Out = Uuid;
    type Err = AtomError;
}

impl<J> Effect<J> for GenerateModelIdEffect {
    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async { Ok(Uuid::new_v4()) }
    }
}

pub struct QuarkStart<H = DefaultConfig>(PhantomData<fn() -> H>);

impl<H, J> EffectSchema<J> for QuarkStart<H>
where
    H: ModelConfig,
{
    type Id = u64;
    type In = Uuid;
    type Out = ();
    type Err = AtomError;
}

impl<H, J> Effect<J> for QuarkStart<H>
where
    H: ModelConfig,
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "starting quark model instance");
            jungle
                .start_model(model_id, H::quark_model_config())
                .await
                .map_err(AtomError::ModelStart)
        }
    }
}

pub struct QuarkShutdown;

impl<J> EffectSchema<J> for QuarkShutdown {
    type Id = u64;
    type In = Uuid;
    type Out = ();
    type Err = AtomError;
}

impl<J> Effect<J> for QuarkShutdown
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "shutting down quark model instance");
            jungle
                .shutdown_model(model_id)
                .await
                .map_err(AtomError::ModelShutdown)
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkPerturbUp — perturb quark weights in the positive direction
// ---------------------------------------------------------------------------

pub struct QuarkPerturbUp;

impl<J> EffectSchema<J> for QuarkPerturbUp {
    type Id = u64;
    type In = (Uuid, u64);
    type Out = ();
    type Err = AtomError;
}

impl<J> Effect<J> for QuarkPerturbUp
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (model_id, seed): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, seed, "perturbing quark weights up");
            jungle
                .perturb_up(model_id, seed)
                .await
                .map_err(AtomError::PerturbUp)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkPerturbDown — perturb quark weights in the negative direction
// ---------------------------------------------------------------------------

pub struct QuarkPerturbDown;

impl<J> EffectSchema<J> for QuarkPerturbDown {
    type Id = u64;
    type In = Uuid;
    type Out = ();
    type Err = AtomError;
}

impl<J> Effect<J> for QuarkPerturbDown
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        model_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%model_id, "perturbing quark weights down");
            jungle
                .perturb_down(model_id)
                .await
                .map_err(AtomError::PerturbDown)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkOptimize — apply QuZO optimization update
// ---------------------------------------------------------------------------

pub struct QuarkOptimize;

impl<J> EffectSchema<J> for QuarkOptimize {
    type Id = u64;
    type In = (Uuid, Potentiation);
    type Out = bool;
    type Err = AtomError;
}

impl<J> Effect<J> for QuarkOptimize
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (model_id, potentiation): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                %model_id,
                loss_up = potentiation.loss_up,
                loss_down = potentiation.loss_down,
                "applying quark optimization"
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
// WaitForPropagation — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Effect that waits for a [`Transmission::Propagation`] at the given [`ObjectId`].
///
/// Downloads the transmission from void, extracts the emission ID to process,
/// the next receive transmission ID for state threading, and the send ID.
pub struct WaitForPropagation;

impl<J> EffectSchema<J> for WaitForPropagation {
    type Id = u64;
    type In = ObjectId;
    type Out = Propagation;
    type Err = AtomError;
}

impl<J> Effect<J> for WaitForPropagation
where
    J: VoidInferOps,
{
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
// WaitForPotentiation — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Effect that waits for a [`Transmission::Potentiation`] at the given [`ObjectId`].
///
/// Downloads the transmission from void, constructs a [`Potentiation`] payload
/// and returns it alongside the next transmission ID for state threading.
pub struct WaitForPotentiation;

impl<J> EffectSchema<J> for WaitForPotentiation {
    type Id = u64;
    type In = ObjectId;
    type Out = (Potentiation, ObjectId);
    type Err = AtomError;
}

impl<J> Effect<J> for WaitForPotentiation
where
    J: VoidInferOps,
{
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
                Transmission::Potentiation {
                    loss_up,
                    loss_down,
                    recv,
                } => {
                    let potentiation = Potentiation {
                        loss_up,
                        loss_down,
                        recv_id: recv,
                    };
                    debug!(loss_up, loss_down, %recv, "potentiation received");
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

impl<J> EffectSchema<J> for Transmit {
    type Id = u64;
    type In = (black_hole_spec::EmissionId, ObjectId);
    type Out = ();
    type Err = AtomError;
}

impl<J> Effect<J> for Transmit
where
    J: VoidInferOps,
{
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
