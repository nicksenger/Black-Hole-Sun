//! Forward-pass effects — typed artifact delivery to and from node inboxes.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;

use crate::ops::VoidOps;
use crate::topology::PropagationTarget;
use crate::AtomError;

/// Operation-typed scheduler completion for the neutral data plane.
///
/// The current two-sided ZO driver continues to use [`NodeTransmission`] as a
/// compatibility program payload. Generic schedulers use this type so an
/// output bundle cannot be relayed as a different artifact type.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct SchedulerDelivery<T> {
    pub node_id: u32,
    pub delivery: black_hole_spec::ArtifactDelivery<T>,
    pub sent_node_ids: Vec<u32>,
}


/// Typed root delivery for one neutral forward pass.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct SendRootArtifactDeliveryInput<T> {
    pub targets: Vec<PropagationTarget>,
    pub delivery: black_hole_spec::ArtifactDelivery<T>,
}

/// Sends an operation-typed input to every root port.
pub struct SendRootArtifactDeliveryEffect<T>(PhantomData<fn() -> T>);


/// Typed counterpart to [`WaitForNodeTransmissionInput`].
///
/// A scheduler instance handles one source artifact type at a time. The
/// runtime graph finalizer has already checked that every listed downstream
/// target accepts that source type.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct WaitForNodeArtifactDeliveryInput<T> {
    pub rx_endpoints: Vec<(u32, ObjectId)>,
    pub downstream: HashMap<u32, Vec<PropagationTarget>>,
    marker: PhantomData<fn() -> T>,
}

impl<T> WaitForNodeArtifactDeliveryInput<T> {
    pub fn new(
        rx_endpoints: Vec<(u32, ObjectId)>,
        downstream: HashMap<u32, Vec<PropagationTarget>>,
    ) -> Self {
        Self {
            rx_endpoints,
            downstream,
            marker: PhantomData,
        }
    }
}

/// Waits for and forwards a typed artifact delivery without interpreting any
/// training-program control message.
pub struct WaitForNodeArtifactDeliveryEffect<T>(PhantomData<fn() -> T>);


async fn send_artifact_delivery<J: VoidOps, T>(
    jungle: &J,
    target: &PropagationTarget,
    delivery: black_hole_spec::ArtifactDelivery<T>,
) -> Result<(), AtomError> {
    let delivery = black_hole_spec::ArtifactDelivery {
        emission_id: delivery.emission_id,
        recv: target.next_input_id,
        send: target.output_id,
    };
    let data = postcard::to_allocvec(&delivery).map_err(|error| {
        AtomError::Transmission(format!("serialize artifact delivery: {error}"))
    })?;
    VoidOps::upload_to_void_with(jungle, target.input_id, data)
        .await
        .map_err(|error| {
            AtomError::Transmission(format!(
                "send artifact to vertex {} port {}: {error}",
                target.node_id, target.port_id
            ))
        })
}


#[jungle::effect(id = 84)]
impl<T, J> Effect<J> for SendRootArtifactDeliveryEffect<T>
where
    T: Send + 'static,
    J: VoidOps,
{
    type In = SendRootArtifactDeliveryInput<T>;
    type Out = Vec<u32>;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let mut sent_node_ids = BTreeSet::new();
            for target in input.targets {
                send_artifact_delivery(jungle, &target, input.delivery).await?;
                sent_node_ids.insert(target.node_id);
            }
            Ok(sent_node_ids.into_iter().collect())
        }
    }
}


#[jungle::effect(id = 83)]
impl<T, J> Effect<J> for WaitForNodeArtifactDeliveryEffect<T>
where
    T: Send + 'static,
    J: VoidOps,
{
    type In = WaitForNodeArtifactDeliveryInput<T>;
    type Out = SchedulerDelivery<T>;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            if input.rx_endpoints.is_empty() {
                return Err(AtomError::Transmission(
                    "no typed artifact endpoints to wait for".to_string(),
                ));
            }

            let futures: Vec<_> = input
                .rx_endpoints
                .into_iter()
                .map(|(node_id, id)| {
                    let jungle_ref = jungle;
                    Box::pin(async move {
                        let delivery = VoidOps::wait_for_artifact_delivery::<T>(jungle_ref, id)
                            .await
                            .map_err(AtomError::Transmission)?;
                        Ok::<_, AtomError>((node_id, delivery))
                    })
                })
                .collect();
            let (result, _index, _rest) = futures::future::select_all(futures).await;
            let (node_id, delivery) = result?;

            let mut sent_node_ids = BTreeSet::new();
            for target in input.downstream.get(&node_id).cloned().unwrap_or_default() {
                send_artifact_delivery(jungle, &target, delivery).await?;
                sent_node_ids.insert(target.node_id);
            }

            Ok(SchedulerDelivery {
                node_id,
                delivery,
                sent_node_ids: sent_node_ids.into_iter().collect(),
            })
        }
    }
}
