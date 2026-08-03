//! Sun effects — spawning, transmission waiting, kick-off, loss computation, and potentiation.

use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::ops::{SunOps, VoidInferOps};
use crate::NucleusError;

// ---------------------------------------------------------------------------
// SpawnAnimal — spawn an animal and return its journey ID
// ---------------------------------------------------------------------------

/// Effect that spawns an animal of type `A` into the jungle.
pub struct SpawnAnimal<A>(PhantomData<fn() -> A>);

impl<A, J> EffectSchema<J> for SpawnAnimal<A>
where
    A: Animal,
    A::Id: AnimalIdValue,
    A::Generation: typosaurus::num::Unsigned,
    A::Seed: Sync + Send + 'static,
{
    type Id = u64;
    type In = A::Seed;
    type Out = Uuid;
    type Err = NucleusError;
}

impl<A, J> Effect<J> for SpawnAnimal<A>
where
    A: Animal,
    A::Id: AnimalIdValue,
    A::Generation: typosaurus::num::Unsigned,
    A::Seed: Sync + Send + 'static,
    J: SunOps,
{
    fn effect(
        jungle: &J,
        seed: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(animal_id = <A::Id as AnimalIdValue>::U32, "spawning animal");
            let journey_id = jungle
                .spawn_animal::<A>(&seed)
                .await
                .map_err(NucleusError::Spawn)?;
            debug!(?journey_id, "animal spawned");
            Ok(journey_id)
        }
    }
}

// ---------------------------------------------------------------------------
// WaitForLayerTransmission — wait for any transmission from current layer nodes
// ---------------------------------------------------------------------------

/// Result of waiting for a transmission from the current layer.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LayerTransmission {
    /// The node id (u32) that received this transmission.
    pub node_id: u32,
    /// The transmission received.
    pub transmission: black_hole_spec::Transmission,
}

/// Effect that waits for the first available transmission from any of the
/// rx ObjectIds associated with the current layer of nodes.
pub struct WaitForLayerTransmission;

impl<J> EffectSchema<J> for WaitForLayerTransmission {
    type Id = u64;
    type In = Vec<(u32, ObjectId)>;
    type Out = LayerTransmission;
    type Err = NucleusError;
}

impl<J> Effect<J> for WaitForLayerTransmission
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        endpoints: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            if endpoints.is_empty() {
                return Err(NucleusError::Transmission(
                    "no endpoints to wait for".to_string(),
                ));
            }

            debug!(count = endpoints.len(), "waiting for layer transmission");

            let futures: Vec<_> = endpoints
                .into_iter()
                .map(|(node_id, id)| {
                    let jungle_ref = jungle;
                    Box::pin(async move {
                        debug!(%id, node_id, "polling endpoint");
                        let transmission = jungle_ref
                            .wait_for_transmission(id)
                            .await
                            .map_err(NucleusError::Transmission)?;
                        Ok::<_, NucleusError>(LayerTransmission {
                            node_id,
                            transmission,
                        })
                    })
                })
                .collect();

            let (result, _index, _rest) = futures::future::select_all(futures).await;

            match result {
                Ok(transmission) => {
                    debug!(node_id = transmission.node_id, "layer transmission received");
                    Ok(transmission)
                }
                Err(e) => Err(e),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// KickOffEffect — generate initial TransmissionId and send to root nodes
// ---------------------------------------------------------------------------

/// Output from [`KickOffEffect`]: the transmission id and rx map for root nodes.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct KickOffResult {
    /// The transmission id used to kick off propagation.
    pub transmission_id: ObjectId,
    /// Map of root node ids to their rx ObjectIds.
    pub rx_map: Vec<(u32, ObjectId)>,
}

/// Effect that generates a new TransmissionId and uploads Propagation
/// transmissions to the rx endpoints of all root nodes (those with no
/// incoming edges). This kicks off the propagation through the graph.
pub struct KickOffEffect;

impl<J> EffectSchema<J> for KickOffEffect {
    type Id = u64;
    type In = Vec<u32>;
    type Out = KickOffResult;
    type Err = NucleusError;
}

impl<J> Effect<J> for KickOffEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        root_nodes: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(count = root_nodes.len(), "kicking off propagation for root nodes");

            let transmission_id = Uuid::new_v4();
            let mut rx_map = Vec::new();

            for &node_id in &root_nodes {
                let rx_id = Uuid::new_v4();

                let propagation = black_hole_spec::Transmission::Propagation {
                    emission_id: black_hole_spec::EmissionId(ObjectId::nil()),
                    recv: rx_id,
                    send: ObjectId::nil(),
                };

                let data = postcard::to_allocvec(&propagation)
                    .map_err(|e| NucleusError::Transmission(format!("serialize: {e}")))?;

                jungle
                    .upload_to_void(data)
                    .await
                    .map_err(|e| {
                        NucleusError::Transmission(format!(
                            "upload kick-off to rx for node {}: {e}",
                            node_id
                        ))
                    })?;

                rx_map.push((node_id, rx_id));
                debug!(node_id, %rx_id, "uploaded kick-off transmission");
            }

            Ok(KickOffResult {
                transmission_id,
                rx_map,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeLossEffect — compute (loss_up, loss_down) from a TransmissionId
// ---------------------------------------------------------------------------

/// Effect that takes a TransmissionId, downloads the transmission, and computes
/// the loss values (loss_up, loss_down) for potentiation.
pub struct ComputeLossEffect;

impl<J> EffectSchema<J> for ComputeLossEffect {
    type Id = u64;
    type In = ObjectId;
    type Out = (f32, f32);
    type Err = NucleusError;
}

impl<J> Effect<J> for ComputeLossEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        transmission_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(%transmission_id, "computing loss from transmission");

            let (loss_up, loss_down) = jungle
                .compute_loss(transmission_id)
                .await
                .map_err(NucleusError::Transmission)?;

            debug!(loss_up, loss_down, "loss computation complete");
            Ok((loss_up, loss_down))
        }
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiationEffect — broadcast losses to all nodes
// ---------------------------------------------------------------------------

/// Output from [`BroadcastPotentiationEffect`]: the new rx map for next epoch.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationResult {
    /// Map of node ids to their new recv ObjectIds for the next epoch.
    pub new_rx_map: Vec<(u32, ObjectId)>,
}

/// Effect that broadcasts `Transmission::Potentiation` with the given loss
/// values to all nodes' potentiation endpoints (po_tx). Generates a new recv
/// ObjectId for each node and uploads the transmission. Does not wait for
/// any response.
pub struct BroadcastPotentiationEffect;

impl<J> EffectSchema<J> for BroadcastPotentiationEffect {
    type Id = u64;
    type In = super::action::BroadcastPotentiationInput;
    type Out = BroadcastPotentiationResult;
    type Err = NucleusError;
}

impl<J> Effect<J> for BroadcastPotentiationEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                loss_up = input.loss_up,
                loss_down = input.loss_down,
                node_count = input.node_ids.len(),
                "broadcasting potentiation to all nodes"
            );

            let mut new_rx_map = Vec::new();

            for &node_id in &input.node_ids {
                let new_rx = Uuid::new_v4();

                let potentiation = black_hole_spec::Transmission::Potentiation {
                    loss_up: input.loss_up,
                    loss_down: input.loss_down,
                    recv: new_rx,
                };

                let data = postcard::to_allocvec(&potentiation)
                    .map_err(|e| NucleusError::Transmission(format!("serialize: {e}")))?;

                jungle
                    .upload_to_void(data)
                    .await
                    .map_err(|e| {
                        NucleusError::Transmission(format!(
                            "upload potentiation to po_tx for node {}: {e}",
                            node_id
                        ))
                    })?;

                new_rx_map.push((node_id, new_rx));
                debug!(node_id, %new_rx, "uploaded potentiation transmission");
            }

            Ok(BroadcastPotentiationResult { new_rx_map })
        }
    }
}
