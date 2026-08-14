//! Sun effects — spawning, transmission waiting, and potentiation.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::ops::{SunOps, VoidInferOps};
use crate::{AtomError, FusionSeed};

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

pub struct GenFusionSeedEffect;
#[jungle::effect]
impl<J> Effect<J> for GenFusionSeedEffect {
    type Id = u64;
    type In = ();
    type Out = FusionSeed;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            Ok(FusionSeed {
                p1_recv_id: Uuid::new_v4(),
                p2_recv_id: Uuid::new_v4(),
                grad_steps: 1,
            })
        }
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
// WaitForNodeTransmission — wait for any currently-ready node
// ---------------------------------------------------------------------------

/// Result of waiting for a transmission from the ready frontier.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct NodeTransmission {
    /// The node id (u32) that received this transmission.
    pub node_id: u32,
    /// The transmission received.
    pub transmission: black_hole_spec::Transmission,
    /// Downstream nodes that were sent this transmission.
    pub sent_node_ids: Vec<u32>,
}

/// Mailboxes needed to drive one cell through a propagation pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropagationTarget {
    /// Internal vertex that owns this destination port.
    pub node_id: u32,
    /// Public destination port whose independent mailbox receives the envelope.
    pub port_id: u32,
    /// Object id where the cell is currently waiting for a transmission.
    pub input_id: ObjectId,
    /// Object id the cell should wait on after this propagation.
    pub next_input_id: ObjectId,
    /// Object id where the cell should publish its output.
    pub output_id: ObjectId,
}

/// First-pass or second-pass propagation sent to every root port.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SendRootPropagationInput {
    pub targets: Vec<PropagationTarget>,
    pub transmission: Transmission,
}

/// One root-propagation send operation with an explicit transmission payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RootPropagationSend {
    pub target: PropagationTarget,
    pub transmission: Transmission,
}

/// Sends one propagation pass to all root ports before output processing begins.
pub struct SendRootPropagationEffect;
/// Sends many root propagations where each target can use a different payload.
pub struct SendRootTaskPropagationsEffect;

/// Input for [`WaitForNodeTransmission`]: ready rx endpoints plus
/// downstream forwarding targets keyed by source node id.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WaitForNodeTransmissionInput {
    /// (node_id, rx_object_id) pairs for nodes in the ready frontier.
    pub rx_endpoints: Vec<(u32, ObjectId)>,
    /// Map from source node id to its downstream targets, each carrying
    /// the mailboxes needed to drive that target cell.
    pub downstream: HashMap<u32, Vec<PropagationTarget>>,
}

/// Effect that waits for the first available transmission from any of the
/// rx ObjectIds associated with the current ready frontier, then forwards
/// the received transmission to the rx endpoints of the downstream nodes
/// for the specific node that received it, so propagation continues through
/// the graph.
pub struct WaitForNodeTransmission;

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
            AtomError::Transmission(format!(
                "send propagation to vertex {} port {}: {e}",
                target.node_id, target.port_id
            ))
        })
}

impl<J> EffectSchema<J> for SendRootPropagationEffect {
    type Id = u64;
    type In = SendRootPropagationInput;
    type Out = Vec<u32>;
    type Err = AtomError;
}

impl<J> Effect<J> for SendRootPropagationEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let mut sent_node_ids = BTreeSet::new();
            for target in &input.targets {
                send_propagation(jungle, target, &input.transmission).await?;
                sent_node_ids.insert(target.node_id);
                debug!(
                    node_id = target.node_id,
                    port_id = target.port_id,
                    input_id = %target.input_id,
                    "sent propagation to root vertex port"
                );
            }
            Ok(sent_node_ids.into_iter().collect())
        }
    }
}

impl<J> EffectSchema<J> for SendRootTaskPropagationsEffect {
    type Id = u64;
    type In = Vec<RootPropagationSend>;
    type Out = Vec<u32>;
    type Err = AtomError;
}

impl<J> Effect<J> for SendRootTaskPropagationsEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        sends: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let mut sent_node_ids = BTreeSet::new();
            for send in &sends {
                send_propagation(jungle, &send.target, &send.transmission).await?;
                sent_node_ids.insert(send.target.node_id);
                debug!(
                    node_id = send.target.node_id,
                    port_id = send.target.port_id,
                    input_id = %send.target.input_id,
                    "sent root task propagation"
                );
            }
            Ok(sent_node_ids.into_iter().collect())
        }
    }
}

impl<J> EffectSchema<J> for WaitForNodeTransmission {
    type Id = u64;
    type In = WaitForNodeTransmissionInput;
    type Out = NodeTransmission;
    type Err = AtomError;
}

impl<J> Effect<J> for WaitForNodeTransmission
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let WaitForNodeTransmissionInput {
                rx_endpoints,
                downstream,
            } = input;

            if rx_endpoints.is_empty() {
                return Err(AtomError::Transmission(
                    "no endpoints to wait for".to_string(),
                ));
            }

            debug!(count = rx_endpoints.len(), "waiting for ready node");

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
                        Ok::<_, AtomError>(NodeTransmission {
                            node_id,
                            transmission,
                            sent_node_ids: Vec::new(),
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
                        "node transmission received, forwarding to downstream nodes"
                    );

                    let mut sent_node_ids = BTreeSet::new();
                    for target in &forward_targets {
                        send_propagation(jungle, target, &transmission.transmission).await?;
                        sent_node_ids.insert(target.node_id);
                    }

                    Ok(NodeTransmission {
                        node_id: transmission.node_id,
                        transmission: transmission.transmission,
                        sent_node_ids: sent_node_ids.into_iter().collect(),
                    })
                }
                Err(e) => Err(e),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiationEffect — broadcast losses to all nodes
// ---------------------------------------------------------------------------

/// Output from [`BroadcastPotentiationEffect`]: each port's first inbox for
/// the next epoch.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationResult {
    pub next_p1_tx_map: Vec<(u32, ObjectId)>,
}

/// Effect that broadcasts `Transmission::Potentiation` with the given loss
/// values to all input ports. Each transmission gives that port a fresh inbox
/// for the next epoch.
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
                port_count = input.port_endpoints.len(),
                "broadcasting potentiation to all input ports"
            );

            let mut next_p1_tx_map = Vec::<(u32, ObjectId)>::new();

            for &(port_id, potentiation_input_id) in &input.port_endpoints {
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
                        AtomError::Transmission(format!("send potentiation to port {port_id}: {e}"))
                    })?;

                next_p1_tx_map.push((port_id, next_p1_tx));
                debug!(port_id, %next_p1_tx, "sent potentiation to input port");
            }

            Ok(BroadcastPotentiationResult { next_p1_tx_map })
        }
    }
}
