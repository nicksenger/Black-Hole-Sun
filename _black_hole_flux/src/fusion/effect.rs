//! Effects used by the model-free two-input fusion protocol.

use std::future::Future;

use black_hole_spec::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::cell::action::{Potentiation, Propagation};
use crate::ops::VoidInferOps;
use crate::AtomError;

/// Generates the stable ID passed to one fusion journey's transform.
pub struct GenerateTransformIdEffect;

#[jungle::effect(id = 68)]
impl<J> Effect<J> for GenerateTransformIdEffect {
    type In = ();
    type Out = Uuid;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        std::future::ready(Ok(Uuid::new_v4()))
    }
}

async fn wait_for_pair<J>(
    jungle: &J,
    p1_recv_id: ObjectId,
    p2_recv_id: ObjectId,
) -> Result<(Transmission, Transmission), AtomError>
where
    J: VoidInferOps,
{
    let p1 = async {
        jungle
            .wait_for_transmission(p1_recv_id)
            .await
            .map_err(|error| AtomError::Transmission(format!("wait for fusion port P1: {error}")))
    };
    let p2 = async {
        jungle
            .wait_for_transmission(p2_recv_id)
            .await
            .map_err(|error| AtomError::Transmission(format!("wait for fusion port P2: {error}")))
    };

    futures::try_join!(p1, p2)
}

/// Waits for one propagation envelope on each fusion input port.
pub struct WaitForFusionPropagation;

#[jungle::effect(id = 69)]
impl<J: VoidInferOps> Effect<J> for WaitForFusionPropagation {
    type In = (ObjectId, ObjectId);
    type Out = (Propagation, Propagation);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (p1_recv_id, p2_recv_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%p1_recv_id, %p2_recv_id, "awaiting fusion propagation pair");
            let (p1, p2) = wait_for_pair(jungle, p1_recv_id, p2_recv_id).await?;

            let p1 = match p1 {
                Transmission::Propagation {
                    emission_id,
                    recv,
                    send,
                } => Propagation {
                    emission_id,
                    recv_id: recv,
                    send_id: send,
                },
                other => {
                    return Err(AtomError::Transmission(format!(
                        "expected Propagation on fusion port P1, got {other:?}"
                    )));
                }
            };
            let p2 = match p2 {
                Transmission::Propagation {
                    emission_id,
                    recv,
                    send,
                } => Propagation {
                    emission_id,
                    recv_id: recv,
                    send_id: send,
                },
                other => {
                    return Err(AtomError::Transmission(format!(
                        "expected Propagation on fusion port P2, got {other:?}"
                    )));
                }
            };

            Ok((p1, p2))
        }
    }
}

/// Waits for one potentiation envelope on each fusion input port.
pub struct WaitForFusionPotentiation;

#[jungle::effect(id = 70)]
impl<J: VoidInferOps> Effect<J> for WaitForFusionPotentiation {
    type In = (ObjectId, ObjectId);
    type Out = (Potentiation, Potentiation);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (p1_recv_id, p2_recv_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%p1_recv_id, %p2_recv_id, "awaiting fusion potentiation pair");
            let (p1, p2) = wait_for_pair(jungle, p1_recv_id, p2_recv_id).await?;

            let p1 = match p1 {
                Transmission::Potentiation {
                    loss_up,
                    loss_down,
                    recv,
                } => Potentiation {
                    loss_up,
                    loss_down,
                    recv_id: recv,
                },
                other => {
                    return Err(AtomError::Transmission(format!(
                        "expected Potentiation on fusion port P1, got {other:?}"
                    )));
                }
            };
            let p2 = match p2 {
                Transmission::Potentiation {
                    loss_up,
                    loss_down,
                    recv,
                } => Potentiation {
                    loss_up,
                    loss_down,
                    recv_id: recv,
                },
                other => {
                    return Err(AtomError::Transmission(format!(
                        "expected Potentiation on fusion port P2, got {other:?}"
                    )));
                }
            };

            Ok((p1, p2))
        }
    }
}
