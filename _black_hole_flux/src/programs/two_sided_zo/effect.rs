//! Two-sided zeroth-order effects — transmission waiting/forwarding and
//! potentiation broadcast over void.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;

use black_hole_spec::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::ops::VoidInferOps;
use crate::topology::PropagationTarget;
use crate::AtomError;

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


/// Input for [`WaitForNodeTransmissionEffect`]: ready rx endpoints plus
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
pub struct WaitForNodeTransmissionEffect;


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


#[jungle::effect(id = 54)]
impl<J: VoidInferOps> Effect<J> for SendRootPropagationEffect {
    type In = SendRootPropagationInput;
    type Out = Vec<u32>;
    type Err = AtomError;

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


#[jungle::effect(id = 55)]
impl<J: VoidInferOps> Effect<J> for SendRootTaskPropagationsEffect {
    type In = Vec<RootPropagationSend>;
    type Out = Vec<u32>;
    type Err = AtomError;

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


#[jungle::effect(id = 56)]
impl<J: VoidInferOps> Effect<J> for WaitForNodeTransmissionEffect {
    type In = WaitForNodeTransmissionInput;
    type Out = NodeTransmission;
    type Err = AtomError;

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
// BroadcastPotentiationEffect — broadcast potentiation payloads to all nodes
// ---------------------------------------------------------------------------

/// Output from [`BroadcastPotentiationEffect`]: each port's first inbox for
/// the next epoch.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationResult {
    pub next_p1_tx_map: Vec<(u32, ObjectId)>,
}

/// Effect that broadcasts `Transmission::Potentiation` payloads to all input
/// ports. Each transmission gives that port a fresh inbox for the next epoch.
pub struct BroadcastPotentiationEffect;

#[jungle::effect(id = 57)]
impl<J: VoidInferOps> Effect<J> for BroadcastPotentiationEffect {
    type In = super::action::BroadcastPotentiationInput;
    type Out = BroadcastPotentiationResult;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(
                loss_up = input.potentiation.loss_up,
                loss_down = input.potentiation.loss_down,
                seed = input.potentiation.seed,
                port_count = input.port_endpoints.len(),
                "broadcasting potentiation to all input ports"
            );

            let mut next_p1_tx_map = Vec::<(u32, ObjectId)>::new();

            for &(port_id, potentiation_input_id) in &input.port_endpoints {
                let next_p1_tx = Uuid::new_v4();
                let potentiation = black_hole_spec::Transmission::Potentiation {
                    potentiation: input.potentiation.clone(),
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
