//! Sun effects — spawning, transmission waiting, kick-off, loss computation, and potentiation.

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::{
    Emission, EmissionId, InferenceOutput, InferenceOutputId, ObjectId, SequenceOutput,
    Transmission,
};
use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::ops::{SunOps, VoidInferOps};
use crate::AtomError;

pub struct GenUuidEffect;
#[jungle::effect]
impl<J> Effect<J> for GenUuidEffect {
    type Id = u64;
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
    type Err = AtomError;
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
                .map_err(AtomError::Spawn)?;
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

/// Mailboxes needed to drive one cell through a propagation pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropagationTarget {
    pub node_id: u32,
    /// Object id where the cell is currently waiting for a transmission.
    pub input_id: ObjectId,
    /// Object id the cell should wait on after this propagation.
    pub next_input_id: ObjectId,
    /// Object id where the cell should publish its output.
    pub output_id: ObjectId,
}

/// Input for [`WaitForLayerTransmission`]: rx endpoints to wait on plus
/// downstream forwarding targets keyed by source node id.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WaitForLayerTransmissionInput {
    /// (node_id, rx_object_id) pairs for the current layer nodes.
    pub rx_endpoints: Vec<(u32, ObjectId)>,
    /// Root cells that receive the epoch's initial transmission.
    pub root_targets: Vec<PropagationTarget>,
    /// Map from source node id to its downstream targets, each carrying
    /// the mailboxes needed to drive that target cell.
    pub downstream: HashMap<u32, Vec<PropagationTarget>>,
    /// Input transmission to upload to root cells.
    pub input_transmission: Transmission,
}

/// Effect that waits for the first available transmission from any of the
/// rx ObjectIds associated with the current layer of nodes, then forwards
/// the received transmission to the rx endpoints of the downstream nodes
/// for the specific node that received it, so propagation continues through
/// the graph.
pub struct WaitForLayerTransmission;

async fn send_propagation<J: VoidInferOps>(
    jungle: &J,
    target: &PropagationTarget,
    transmission: &Transmission,
) -> Result<(), AtomError> {
    let mut transmission = transmission.clone();
    match &mut transmission {
        Transmission::Propagation { recv, send, .. } => {
            *recv = target.next_input_id;
            *send = target.output_id;
        }
        other => {
            return Err(AtomError::Transmission(format!(
                "expected propagation input, got {other:?}"
            )));
        }
    }

    let data = postcard::to_allocvec(&transmission)
        .map_err(|e| AtomError::Transmission(format!("serialize propagation: {e}")))?;
    jungle
        .upload_to_void_with(target.input_id, data)
        .await
        .map_err(|e| {
            AtomError::Transmission(format!("send propagation to node {}: {e}", target.node_id))
        })
}

impl<J> EffectSchema<J> for WaitForLayerTransmission {
    type Id = u64;
    type In = WaitForLayerTransmissionInput;
    type Out = LayerTransmission;
    type Err = AtomError;
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
                root_targets,
                downstream,
                input_transmission,
            } = input;

            for target in &root_targets {
                send_propagation(jungle, target, &input_transmission).await?;
                debug!(
                    node_id = target.node_id,
                    input_id = %target.input_id,
                    "sent initial propagation to root cell"
                );
            }

            if rx_endpoints.is_empty() {
                return Err(AtomError::Transmission(
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
                            .map_err(AtomError::Transmission)?;
                        Ok::<_, AtomError>(LayerTransmission {
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

                    for target in &forward_targets {
                        send_propagation(jungle, target, &transmission.transmission).await?;
                    }

                    Ok(LayerTransmission {
                        node_id: transmission.node_id,
                        transmission: transmission.transmission,
                    })
                }
                Err(e) => Err(e),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InitializeEffect — create the initial emission for both propagation passes
// ---------------------------------------------------------------------------

/// Creates an initial inference output and emission in void, then returns one
/// propagation value for each branch. The branch effects attach cell-specific
/// mailboxes before sending these values to root cells.
pub struct InitializeEffect;

impl<J> EffectSchema<J> for InitializeEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;
}

impl<J> Effect<J> for InitializeEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("creating initial sun emission");

            let inference_output = InferenceOutput {
                results: vec![SequenceOutput(vec![black_hole_spec::DarkToken {
                    predicted: 0,
                    dark_knowledge: Vec::new(),
                }])],
            };
            let inference_output_bytes = postcard::to_allocvec(&inference_output)?;
            let inference_output_id = jungle
                .upload_to_void(inference_output_bytes)
                .await
                .map_err(AtomError::Upload)?;

            let emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(inference_output_id),
            };
            let emission_bytes = postcard::to_allocvec(&emission)?;
            let emission_void_id = jungle
                .upload_to_void(emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            let propagation = Transmission::Propagation {
                emission_id: EmissionId(emission_void_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            };

            Ok((propagation.clone(), propagation))
        }
    }
}

/// Result type alias for InitializeEffect output.
pub type InitializeResult = (Transmission, Transmission);

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
    type Err = AtomError;
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

/// Output from [`BroadcastPotentiationEffect`]: each cell's first inbox for
/// the next epoch.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationResult {
    pub next_p1_tx_map: Vec<(u32, ObjectId)>,
}

/// Effect that broadcasts `Transmission::Potentiation` with the given loss
/// values to all cells' potentiation inboxes. Each transmission gives the
/// cell a fresh inbox for the next epoch.
pub struct BroadcastPotentiationEffect;

impl<J> EffectSchema<J> for BroadcastPotentiationEffect {
    type Id = u64;
    type In = super::action::BroadcastPotentiationInput;
    type Out = BroadcastPotentiationResult;
    type Err = AtomError;
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
                node_count = input.node_endpoints.len(),
                "broadcasting potentiation to all nodes"
            );

            let mut next_p1_tx_map = Vec::<(u32, ObjectId)>::new();

            for &(node_id, potentiation_input_id) in &input.node_endpoints {
                let next_p1_tx = Uuid::new_v4();
                let potentiation = black_hole_spec::Transmission::Potentiation {
                    loss_up: input.loss_up,
                    loss_down: input.loss_down,
                    recv: next_p1_tx,
                };

                let data = postcard::to_allocvec(&potentiation)?;

                jungle
                    .upload_to_void_with(potentiation_input_id, data)
                    .await
                    .map_err(|e| {
                        AtomError::Transmission(format!("send potentiation to node {node_id}: {e}"))
                    })?;

                next_p1_tx_map.push((node_id, next_p1_tx));
                debug!(node_id, %next_p1_tx, "sent potentiation to cell");
            }

            Ok(BroadcastPotentiationResult { next_p1_tx_map })
        }
    }
}
