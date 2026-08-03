//! Sun effects — spawning, transmission waiting, kick-off, loss computation, and potentiation.

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::{ObjectId, Transmission};
use futures::future::join_all;
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

/// Input for [`WaitForLayerTransmission`]: rx endpoints to wait on plus
/// downstream forwarding targets keyed by source node id.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WaitForLayerTransmissionInput {
    /// (node_id, rx_object_id) pairs for the current layer nodes.
    pub rx_endpoints: Vec<(u32, ObjectId)>,
    /// Map from source node id to its downstream (node_id, rx_object_id) targets.
    pub downstream: HashMap<u32, Vec<(u32, ObjectId)>>,
}

/// Effect that waits for the first available transmission from any of the
/// rx ObjectIds associated with the current layer of nodes, then forwards
/// the received transmission to the rx endpoints of the downstream nodes
/// for the specific node that received it, so propagation continues through
/// the graph.
pub struct WaitForLayerTransmission;
impl<J> EffectSchema<J> for WaitForLayerTransmission {
    type Id = u64;
    type In = WaitForLayerTransmissionInput;
    type Out = LayerTransmission;
    type Err = NucleusError;
}

impl<J> Effect<J> for WaitForLayerTransmission
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let WaitForLayerTransmissionInput {
                rx_endpoints,
                downstream,
            } = input;

            if rx_endpoints.is_empty() {
                return Err(NucleusError::Transmission(
                    "no endpoints to wait for".to_string(),
                ));
            }

            debug!(count = rx_endpoints.len(), "waiting for layer transmission");

            let futures: Vec<_> = rx_endpoints
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
                    // Only forward to the downstream nodes of the specific
                    // node that received this transmission.
                    let forward_targets = downstream
                        .get(&transmission.node_id)
                        .cloned()
                        .unwrap_or_default();

                    debug!(
                        node_id = transmission.node_id,
                        forward_count = forward_targets.len(),
                        "layer transmission received, forwarding to downstream nodes"
                    );

                    // Serialize the transmission for forwarding.
                    let data = postcard::to_allocvec(&transmission.transmission).map_err(|e| {
                        NucleusError::Transmission(format!("serialize for forward: {e}"))
                    })?;

                    // Forward to all downstream nodes of this source in parallel.
                    let forward_futures: Vec<_> = forward_targets
                        .into_iter()
                        .map(|(target_id, _rx_id)| {
                            let jungle_ref = jungle;
                            let data = data.clone();
                            Box::pin(async move {
                                jungle_ref.upload_to_void(data).await.map_err(|e| {
                                    NucleusError::Transmission(format!(
                                        "forward to downstream node {}: {e}",
                                        target_id
                                    ))
                                })
                            })
                        })
                        .collect();

                    let results = join_all(forward_futures).await;
                    for result in results {
                        result?;
                    }

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

/// Effect that generates a new TransmissionId and uploads Propagation
/// transmissions to the rx endpoints of all root nodes (those with no
/// incoming edges). This kicks off the propagation through the graph.
pub struct KickOffEffect;

impl<J> EffectSchema<J> for KickOffEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
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
            debug!("kicking off propagation for root nodes");

            let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
            let tokenizer = get_tokenizer();
            let dark_tokens = text_to_dark_tokens(input_text, &tokenizer);
            let inference_output = InferenceOutput {
                results: vec![SequenceOutput(dark_tokens)],
            };
            let inference_output_bytes =
                to_allocvec(&inference_output).expect("serialize inference output");
            let inference_output_id = void_upload(
                &make_client_endpoint().await,
                void_addr,
                inference_output_bytes,
            )
            .await;
            let emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(inference_output_id),
            };
            let emission_bytes = to_allocvec(&emission).expect("serialize emission");
            let emission_void_id =
                void_upload(&make_client_endpoint().await, void_addr, emission_bytes).await;
            let propagation = Transmission::Propagation {
                emission_id: EmissionId(emission_void_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            };

            Ok((propagation.clone(), propagation))
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeLossEffect — compute (loss_up, loss_down) from a TransmissionId
// ---------------------------------------------------------------------------

/// Effect that takes a TransmissionId, downloads the transmission, and computes
/// the loss values (loss_up, loss_down) for potentiation.
pub struct ComputeLossEffect;
#[jungle::effect]
impl<J> Effect<J> for ComputeLossEffect {
    type Id = u64;
    type In = (Transmission, Transmission);
    type Out = (f32, f32);
    type Err = NucleusError;
    fn effect(
        _jungle: &J,
        _transmission_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("faking loss from transmission");
            Ok((0.1, 0.1))
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

                jungle.upload_to_void(data).await.map_err(|e| {
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
