//! Forward-pass actions — preparing, seeding, processing, and completing one
//! dependency-aware typed graph execution.

use std::collections::HashMap;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use uuid::Uuid;

use super::effect::{
    SendRootArtifactDeliveryEffect, SendRootArtifactDeliveryInput,
    WaitForNodeArtifactDeliveryEffect, WaitForNodeArtifactDeliveryInput,
};
use crate::topology::{
    advance_frontier, initial_ready_nodes, pending_dependency_counts, port_ids, sorted_node_ids,
    vertex_ids, PropagationTarget,
};

// ---------------------------------------------------------------------------
// ForwardPass — phase-neutral typed graph execution
// ---------------------------------------------------------------------------

/// Initializes one dependency-aware forward pass using a program-neutral
/// frontier and endpoint set. The typed boundary is erased only while the
/// already contract-validated graph is scheduled.
pub struct PrepareForwardPass<S, Input>(PhantomData<fn() -> (S, Input)>);

#[jungle::action(carry = black_hole_spec::ArtifactDelivery<Input>)]
impl<S, Input> Action for PrepareForwardPass<S, Input>
where
    Input: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_spec::ArtifactDelivery<Input>;
    type Output = black_hole_spec::ArtifactDelivery<()>;

    fn emit(
        _state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> ((), black_hole_spec::ArtifactDelivery<Input>) {
        ((), input)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_spec::ArtifactDelivery<Input>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare forward pass failed".to_string()))?;

        let mut topology = state.topology.lock().unwrap();
        let pending = pending_dependency_counts(&topology);
        let ready = initial_ready_nodes(&pending);
        let ports = port_ids(&topology);
        let nodes = vertex_ids(&topology);

        state.runtime.next_inputs = ports
            .into_iter()
            .map(|port_id| (port_id, Uuid::new_v4()))
            .collect();
        state.runtime.outputs = nodes
            .iter()
            .copied()
            .map(|node_id| (node_id, Uuid::new_v4()))
            .collect();
        for node_id in nodes {
            topology
                .node_operational_states
                .insert(node_id, crate::topology::SunOperationalState::Queued);
            topology.node_phase_annotations.remove(&node_id);
        }
        state.runtime.pending = pending;
        state.runtime.ready = ready;
        Ok(black_hole_spec::ArtifactDelivery {
            emission_id: black_hole_spec::ObjectRef::new(carry.emission_id.id()),
            recv: carry.recv,
            send: carry.send,
        })
    }
}

/// Sends the typed input artifact to every root of a forward pass.
pub struct SendForwardRoots<S>(PhantomData<fn() -> S>);

#[jungle::action(carry = black_hole_spec::ArtifactDelivery<()>)]
impl<S> Action for SendForwardRoots<S> {
    type Effect = SendRootArtifactDeliveryEffect<()>;
    type Input = black_hole_spec::ArtifactDelivery<()>;
    type Output = black_hole_spec::ArtifactDelivery<()>;

    fn emit(
        state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> (
        SendRootArtifactDeliveryInput<()>,
        black_hole_spec::ArtifactDelivery<()>,
    ) {
        let topology = state.topology.lock().unwrap();
        let mut targets = Vec::new();
        for (&node_id, ports) in &topology.vertex_ports {
            if !topology
                .incoming
                .get(&node_id)
                .is_none_or(|sources| sources.is_empty())
            {
                continue;
            }
            for &port_id in ports {
                let (Some(&input_id), Some(&next_input_id), Some(&output_id)) = (
                    state.runtime.inputs.get(&port_id),
                    state.runtime.next_inputs.get(&port_id),
                    state.runtime.outputs.get(&node_id),
                ) else {
                    continue;
                };
                targets.push(PropagationTarget {
                    node_id,
                    port_id,
                    input_id,
                    next_input_id,
                    output_id,
                });
            }
        }
        targets.sort_by_key(|target| (target.node_id, target.port_id));

        (
            SendRootArtifactDeliveryInput {
                targets,
                delivery: input,
            },
            input,
        )
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_spec::ArtifactDelivery<()>,
    ) -> Result<Self::Output, Failure> {
        let sent = output
            .map_err(|error| Failure::Message(format!("send forward roots failed: {error}")))?;
        state.topology.lock().unwrap().record_forward_started(sent);
        Ok(carry)
    }
}

/// Waits for one ready typed node, forwards its output, and advances the
/// dependency frontier.
pub struct ProcessForwardNode<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessForwardNode<S> {
    type Effect = WaitForNodeArtifactDeliveryEffect<()>;
    type Input = black_hole_spec::ArtifactDelivery<()>;
    type Output = black_hole_spec::ArtifactDelivery<()>;

    fn emit(
        state: &super::ForwardSunState<S>,
        _input: Self::Input,
    ) -> WaitForNodeArtifactDeliveryInput<()> {
        let ready = sorted_node_ids(&state.runtime.ready);
        let topology = state.topology.lock().unwrap();
        let rx_endpoints = ready
            .iter()
            .filter_map(|node_id| state.runtime.outputs.get(node_id).map(|id| (*node_id, *id)))
            .collect();
        let mut downstream = HashMap::new();

        for node_id in ready {
            let targets = topology
                .outgoing
                .get(&node_id)
                .into_iter()
                .flatten()
                .filter_map(|target| {
                    Some(PropagationTarget {
                        node_id: target.vertex_id,
                        port_id: target.port_id,
                        input_id: *state.runtime.inputs.get(&target.port_id)?,
                        next_input_id: *state.runtime.next_inputs.get(&target.port_id)?,
                        output_id: *state.runtime.outputs.get(&target.vertex_id)?,
                    })
                })
                .collect();
            downstream.insert(node_id, targets);
        }

        WaitForNodeArtifactDeliveryInput::new(rx_endpoints, downstream)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let completion = output
            .map_err(|error| Failure::Message(format!("process forward node failed: {error}")))?;
        let outgoing = state
            .topology
            .lock()
            .unwrap()
            .outgoing
            .get(&completion.node_id)
            .cloned()
            .unwrap_or_default();
        advance_frontier(
            &mut state.runtime.pending,
            &mut state.runtime.ready,
            completion.node_id,
            &outgoing,
        )?;
        let mut topology = state.topology.lock().unwrap();
        topology.record_forward_completed(completion.node_id);
        topology.record_forward_started(completion.sent_node_ids);
        Ok(completion.delivery)
    }
}

/// Rotates each node to the inbox provisioned for the next serving request.
pub struct CompleteForwardPass<S, Output>(PhantomData<fn() -> (S, Output)>);

#[jungle::action(carry = black_hole_spec::ArtifactDelivery<()>)]
impl<S, Output> Action for CompleteForwardPass<S, Output>
where
    Output: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_spec::ArtifactDelivery<()>;
    type Output = black_hole_spec::ArtifactDelivery<Output>;

    fn emit(
        _state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> ((), black_hole_spec::ArtifactDelivery<()>) {
        ((), input)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_spec::ArtifactDelivery<()>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("complete forward pass failed".to_string()))?;
        state.runtime.inputs = std::mem::take(&mut state.runtime.next_inputs);
        state.runtime.outputs.clear();
        Ok(black_hole_spec::ArtifactDelivery {
            emission_id: black_hole_spec::ObjectRef::new(carry.emission_id.id()),
            recv: carry.recv,
            send: carry.send,
        })
    }
}

/// Default serving sink used when the caller only needs the durable artifact
/// emitted by the final node.
pub struct DiscardForwardOutput<S, T>(PhantomData<fn() -> (S, T)>);

#[jungle::action]
impl<S, T> Action for DiscardForwardOutput<S, T>
where
    T: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_spec::ArtifactDelivery<T>;
    type Output = ();

    fn emit(_state: &super::ForwardSunState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("discard forward output failed".to_string()))
    }
}