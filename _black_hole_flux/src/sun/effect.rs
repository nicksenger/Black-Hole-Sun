//! Sun effects — spawning, transmission waiting, and node advancement.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use black_hole_spec::ObjectId;

use crate::ops::{SunOps, VoidInferOps};
use crate::NucleusError;

// ---------------------------------------------------------------------------
// SpawnAnimal — spawn an animal and return its journey ID
// ---------------------------------------------------------------------------

/// Effect that spawns an animal of type `A` into the jungle.
///
/// Takes the animal's seed as input, calls [`JungleClient::spawn`](jungle_sdk::JungleClient::spawn),
/// and returns the journey UUID.
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
/// Contains the node id that received the transmission and the transmission data.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LayerTransmission {
    /// The node id (u32) that received this transmission.
    pub node_id: u32,
    /// The transmission received.
    pub transmission: black_hole_spec::Transmission,
}

/// Effect that waits for the first available transmission from any of the
/// rx ObjectIds associated with the current layer of nodes.
///
/// Takes a vector of (node_id, rx_object_id) pairs and polls them concurrently,
/// returning the first one that succeeds.
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

            // Spawn a task for each endpoint and race them
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

            // Use futures::future::select_all to get the first completion
            let (result, _index, _rest) =
                futures::future::select_all(futures).await;

            match result {
                Ok(transmission) => {
                    debug!(
                        node_id = transmission.node_id,
                        "layer transmission received"
                    );
                    Ok(transmission)
                }
                Err(e) => Err(e),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AdvanceNode — process a received transmission and forward to outgoing nodes
// ---------------------------------------------------------------------------

/// Input for the [`AdvanceNode`] effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct AdvanceInput {
    /// The node id that received the transmission.
    pub node_id: u32,
    /// The rx object id of this node (to be replaced).
    pub current_rx: ObjectId,
    /// The list of outgoing edge targets (node ids).
    pub outgoing_nodes: Vec<u32>,
    /// The received transmission to forward.
    pub transmission: black_hole_spec::Transmission,
}

/// Output from [`AdvanceNode`]: the new rx id for the processed node.
pub type AdvanceOutput = ObjectId;

/// Effect that processes a received transmission for a node:
/// 1. Generates a new rx Uuid for the processed node
/// 2. For each outgoing node, generates a new tx Uuid
/// 3. Creates a Propagation transmission with the new tx as recv field
/// 4. Uploads the transmission to void at the outgoing node's tx endpoint
pub struct AdvanceNode;

impl<J> EffectSchema<J> for AdvanceNode {
    type Id = u64;
    type In = AdvanceInput;
    type Out = AdvanceOutput;
    type Err = NucleusError;
}

impl<J> Effect<J> for AdvanceNode
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let AdvanceInput {
                node_id,
                current_rx: _current_rx,
                outgoing_nodes,
                transmission,
            } = input;

            // Generate new rx id for this node (for next iteration)
            let new_rx = Uuid::new_v4();

            debug!(
                node_id,
                ?new_rx,
                outgoing_count = outgoing_nodes.len(),
                "advancing node"
            );

            // For each outgoing node, generate a new tx id and upload the transmission
            for target_id in &outgoing_nodes {
                let tx_id = Uuid::new_v4();

                // Create a Propagation transmission for the outgoing node
                let propagation = black_hole_spec::Transmission::Propagation {
                    emission_id: match &transmission {
                        black_hole_spec::Transmission::Propagation { emission_id, .. } => {
                            emission_id.clone()
                        }
                        other => {
                            return Err(NucleusError::Transmission(format!(
                                "expected Propagation for forwarding, got {:?}",
                                other
                            )));
                        }
                    },
                    recv: tx_id,
                    send: ObjectId::nil(),
                };

                // Serialize and upload the propagation to void at the tx endpoint
                let data = postcard::to_allocvec(&propagation)
                    .map_err(|e| NucleusError::Transmission(format!("serialize: {e}")))?;

                jungle
                    .upload_to_void(data)
                    .await
                    .map_err(|e| {
                        NucleusError::Transmission(format!(
                            "upload to tx for node {}: {e}",
                            target_id
                        ))
                    })?;

                debug!(
                    target_id,
                    ?tx_id,
                    "uploaded propagation to outgoing node"
                );
            }

            Ok(new_rx)
        }
    }
}
