//! Cell effects for perturbation, optimization, transmission, and waiting.

use std::future::Future;

use jungle_sdk::prelude::*;
use tracing::debug;

use black_hole_spec::{ObjectId, Transmission};

use super::action::{Potentiation, Propagation};
use crate::ops::VoidInferOps;
use crate::NucleusError;

// ---------------------------------------------------------------------------
// QuarkPerturbUp — perturb quark weights in the positive direction
// ---------------------------------------------------------------------------

pub struct QuarkPerturbUp;

impl<J> EffectSchema<J> for QuarkPerturbUp {
    type Id = u64;
    type In = u64;
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkPerturbUp
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        seed: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(seed, "perturbing quark weights up");
            jungle
                .perturb_up(seed)
                .await
                .map_err(NucleusError::PerturbUp)?;
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
    type In = ();
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkPerturbDown
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("perturbing quark weights down");
            jungle
                .perturb_down()
                .await
                .map_err(NucleusError::PerturbDown)?;
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
    type In = Potentiation;
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkOptimize
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        potentiation: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                loss_up = potentiation.loss_up,
                loss_down = potentiation.loss_down,
                "applying quark optimization"
            );
            jungle
                .optimize(potentiation.loss_up, potentiation.loss_down)
                .await
                .map_err(NucleusError::Optimize)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForInitiation — await a Transmission::Initiation from void
// ---------------------------------------------------------------------------

/// Effect that waits for a [`Transmission::Initiation`] at the given [`ObjectId`].
///
/// Downloads the transmission from void and returns the next transmission ID
/// for state threading. Returns unit as the data payload since initiation
/// carries no emission to process downstream.
pub struct WaitForInitiation;

impl<J> EffectSchema<J> for WaitForInitiation {
    type Id = num::U55;
    type In = ObjectId;
    type Out = ((), ObjectId);
    type Err = NucleusError;
}

impl<J> Effect<J> for WaitForInitiation
where
    J: VoidInferOps,
{
    fn effect(jungle: &J, id: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            debug!(%id, "awaiting initiation transmission");
            let transmission = jungle
                .wait_for_transmission(id)
                .await
                .map_err(NucleusError::Transmission)?;
            match transmission {
                Transmission::Initiation { recv } => {
                    debug!(%recv, "initiation received");
                    Ok(((), recv))
                }
                other => {
                    let msg = format!("expected Initiation, got {:?}", other);
                    debug!("initiation failed: {msg}");
                    Err(NucleusError::Transmission(msg))
                }
            }
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
    type Err = NucleusError;
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
                .map_err(NucleusError::Transmission)?;
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
                    Err(NucleusError::Transmission(msg))
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
    type Err = NucleusError;
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
                .map_err(NucleusError::Transmission)?;
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
                    Err(NucleusError::Transmission(msg))
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
    type Err = NucleusError;
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
                .map_err(NucleusError::Transmission)?;
            Ok(())
        }
    }
}
