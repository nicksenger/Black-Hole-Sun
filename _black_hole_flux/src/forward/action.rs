//! Forward-pass actions — preparing, seeding, processing, and completing one
//! dependency-aware typed graph execution.

use std::collections::HashMap;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use uuid::Uuid;

use super::effect::{
    RootArtifactDeliverySend, SendReadyRootArtifactDeliveriesEffect,
    SendRootArtifactDeliveryEffect, SendRootArtifactDeliveryInput,
    WaitForNodeArtifactDeliveryEffect, WaitForNodeArtifactDeliveryInput,
};
use crate::topology::{
    advance_frontier, initial_ready_nodes, pending_dependency_counts, port_ids, root_vertex_ids,
    sorted_node_ids, vertex_ids, PropagationTarget,
};

// ---------------------------------------------------------------------------
// ForwardPass — phase-neutral typed graph execution
// ---------------------------------------------------------------------------

fn pipeline_task_for_node<S>(state: &super::ForwardSunState<S>, node_id: u32) -> Option<usize> {
    let completed = state
        .runtime
        .node_completed
        .get(&node_id)
        .copied()
        .unwrap_or_default();
    (completed < state.runtime.pipeline_inputs.len()).then_some(completed)
}

fn pipeline_ready_nodes<S>(
    state: &super::ForwardSunState<S>,
    topology: &crate::topology::SunTopology,
) -> Vec<u32> {
    let mut ready = Vec::new();
    for node_id in vertex_ids(topology) {
        let Some(input_index) = pipeline_task_for_node(state, node_id) else {
            continue;
        };
        let is_root = topology
            .incoming
            .get(&node_id)
            .is_none_or(|sources| sources.is_empty());
        let dependencies_ready = if is_root {
            state
                .runtime
                .root_sent
                .get(&node_id)
                .copied()
                .unwrap_or_default()
                > input_index
        } else {
            topology
                .incoming
                .get(&node_id)
                .into_iter()
                .flatten()
                .all(|predecessor| {
                    state
                        .runtime
                        .node_completed
                        .get(predecessor)
                        .copied()
                        .unwrap_or_default()
                        > input_index
                })
        };
        if dependencies_ready {
            ready.push(node_id);
        }
    }
    ready.sort_unstable();
    ready
}

fn pipeline_target<S>(
    state: &super::ForwardSunState<S>,
    topology: &crate::topology::SunTopology,
    input_index: usize,
    port_id: u32,
) -> PropagationTarget {
    let node_id = *topology
        .port_vertices
        .get(&port_id)
        .unwrap_or_else(|| panic!("missing node for port {port_id}"));
    let input_id = state.pipeline_input_id(input_index, port_id);
    let next_input_id = state
        .runtime
        .pipeline_input_ids
        .get(input_index + 1)
        .and_then(|inputs| inputs.get(&port_id).copied())
        .or_else(|| state.runtime.next_inputs.get(&port_id).copied())
        .unwrap_or_else(|| {
            panic!("missing next pipeline mailbox for port {port_id} at input {input_index}")
        });
    let output_id = state
        .runtime
        .pipeline_output_ids
        .get(input_index)
        .and_then(|outputs| outputs.get(&node_id).copied())
        .unwrap_or_else(|| {
            panic!("missing pipeline output for node {node_id} at input {input_index}")
        });
    PropagationTarget {
        node_id,
        port_id,
        input_id,
        next_input_id,
        output_id,
    }
}

impl<S> super::ForwardSunState<S> {
    fn pipeline_input_id(&self, input_index: usize, port_id: u32) -> black_hole_type::ObjectId {
        self.runtime
            .pipeline_input_ids
            .get(input_index)
            .and_then(|inputs| inputs.get(&port_id).copied())
            .unwrap_or_else(|| {
                panic!("missing pipeline input for port {port_id} at input {input_index}")
            })
    }
}

/// Starts a bounded pipeline window using the limit selected by the forward
/// strategy. The source is not invoked again until every input in this window
/// has reached its sink, providing backpressure without coupling source flows
/// to their consumers.
pub struct BeginForwardPipeline<S, const MAX_IN_FLIGHT: usize>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S, const MAX_IN_FLIGHT: usize> Action for BeginForwardPipeline<S, MAX_IN_FLIGHT> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::ForwardSunState<S>, _input: ()) {}

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("begin forward pipeline failed".to_string()))?;
        state.runtime.pipeline_inputs.clear();
        state.runtime.completed_outputs.clear();
        state.runtime.pipeline_window = MAX_IN_FLIGHT.max(1);
        Ok(())
    }
}

/// Stores one generated input in the current pipeline window.
pub struct StoreForwardPipelineInput<S, Input>(PhantomData<fn() -> (S, Input)>);

#[jungle::action(carry = black_hole_type::ArtifactDelivery<Input>)]
impl<S, Input> Action for StoreForwardPipelineInput<S, Input>
where
    Input: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_type::ArtifactDelivery<Input>;
    type Output = ();

    fn emit(
        _state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> ((), black_hole_type::ArtifactDelivery<Input>) {
        ((), input)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_type::ArtifactDelivery<Input>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("store forward pipeline input failed".to_string()))?;
        state
            .runtime
            .pipeline_inputs
            .push(black_hole_type::ArtifactDelivery {
                emission_id: black_hole_type::ObjectRef::new(carry.emission_id.id()),
                recv: carry.recv,
                send: carry.send,
            });
        Ok(())
    }
}

/// Allocates request-specific mailboxes and resets per-node pipeline progress.
pub struct PrepareForwardPipeline<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PrepareForwardPipeline<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::ForwardSunState<S>, _input: ()) {}

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("prepare forward pipeline failed".to_string()))?;
        let input_count = state.runtime.pipeline_inputs.len();
        if input_count == 0 {
            return Err(Failure::Message(
                "forward pipeline requires at least one input".to_string(),
            ));
        }

        let (ports, nodes, roots, sink_id) = {
            let topology = state.topology.lock().unwrap();
            let mut ports = port_ids(&topology);
            ports.sort_unstable();
            let mut nodes = vertex_ids(&topology);
            nodes.sort_unstable();
            let roots = root_vertex_ids(&topology);
            let sink_id = state.runtime.sink_id.ok_or_else(|| {
                Failure::Message("forward graph has no sink after finalization".to_string())
            })?;
            (ports, nodes, roots, sink_id)
        };

        state.runtime.pipeline_input_ids = vec![HashMap::new(); input_count];
        state.runtime.pipeline_output_ids = vec![HashMap::new(); input_count];
        state.runtime.next_inputs.clear();
        for port_id in ports {
            let initial = state.runtime.inputs.get(&port_id).copied().ok_or_else(|| {
                Failure::Message(format!(
                    "missing initial forward mailbox for port {port_id}"
                ))
            })?;
            state.runtime.pipeline_input_ids[0].insert(port_id, initial);
            for input_index in 1..input_count {
                state.runtime.pipeline_input_ids[input_index].insert(port_id, Uuid::new_v4());
            }
            state.runtime.next_inputs.insert(port_id, Uuid::new_v4());
        }
        for node_id in &nodes {
            for input_index in 0..input_count {
                state.runtime.pipeline_output_ids[input_index].insert(*node_id, Uuid::new_v4());
            }
        }

        state.runtime.node_completed = nodes.iter().copied().map(|node| (node, 0)).collect();
        state.runtime.root_sent = roots.into_iter().map(|node| (node, 0)).collect();
        state.runtime.pipeline_completions = 0;
        state.runtime.pipeline_target_completions = nodes.len() * input_count;
        state.runtime.sink_id = Some(sink_id);

        let mut topology = state.topology.lock().unwrap();
        for node_id in nodes {
            topology
                .node_operational_states
                .insert(node_id, crate::topology::SunOperationalState::Queued);
            topology.node_phase_annotations.remove(&node_id);
        }
        Ok(())
    }
}

/// Seeds the next input for every root whose previous input has completed.
pub struct SendReadyForwardRoots<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for SendReadyForwardRoots<S> {
    type Effect = SendReadyRootArtifactDeliveriesEffect<()>;
    type Input = ();
    type Output = ();

    fn emit(state: &super::ForwardSunState<S>, _input: ()) -> Vec<RootArtifactDeliverySend<()>> {
        let topology = state.topology.lock().unwrap();
        let mut sends = Vec::new();
        for root_id in root_vertex_ids(&topology) {
            let Some(input_index) = pipeline_task_for_node(state, root_id) else {
                continue;
            };
            if state
                .runtime
                .root_sent
                .get(&root_id)
                .copied()
                .unwrap_or_default()
                > input_index
            {
                continue;
            }
            let delivery = state.runtime.pipeline_inputs[input_index];
            for port_id in topology.vertex_ports.get(&root_id).into_iter().flatten() {
                sends.push(RootArtifactDeliverySend {
                    target: pipeline_target(state, &topology, input_index, *port_id),
                    delivery,
                });
            }
        }
        sends
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        let sent = output.map_err(|error| {
            Failure::Message(format!("send ready forward roots failed: {error}"))
        })?;
        for node_id in &sent {
            let next = state
                .runtime
                .root_sent
                .get(node_id)
                .copied()
                .unwrap_or_default()
                .saturating_add(1);
            state.runtime.root_sent.insert(*node_id, next);
        }
        state.topology.lock().unwrap().record_forward_started(sent);
        Ok(())
    }
}

/// Waits for the next ready node/input completion, forwards it within that
/// input's DAG, and releases the same node for the following input.
pub struct ProcessReadyForwardPipelineNode<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessReadyForwardPipelineNode<S> {
    type Effect = WaitForNodeArtifactDeliveryEffect<()>;
    type Input = ();
    type Output = ();

    fn emit(state: &super::ForwardSunState<S>, _input: ()) -> WaitForNodeArtifactDeliveryInput<()> {
        let topology = state.topology.lock().unwrap();
        let ready = pipeline_ready_nodes(state, &topology);
        let mut rx_endpoints = Vec::new();
        let mut downstream = HashMap::new();
        for node_id in ready {
            let input_index = pipeline_task_for_node(state, node_id)
                .expect("ready pipeline node must have an active input");
            if let Some(output_id) = state
                .runtime
                .pipeline_output_ids
                .get(input_index)
                .and_then(|outputs| outputs.get(&node_id).copied())
            {
                rx_endpoints.push((node_id, output_id));
            }
            let targets = topology
                .outgoing
                .get(&node_id)
                .into_iter()
                .flatten()
                .map(|target| pipeline_target(state, &topology, input_index, target.port_id))
                .collect();
            downstream.insert(node_id, targets);
        }
        WaitForNodeArtifactDeliveryInput::new(rx_endpoints, downstream)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        let completion = output.map_err(|error| {
            Failure::Message(format!(
                "process ready forward pipeline node failed: {error}"
            ))
        })?;
        let node_id = completion.node_id;
        let input_index = pipeline_task_for_node(state, node_id).ok_or_else(|| {
            Failure::Message(format!(
                "completed node {node_id} has no active forward pipeline input"
            ))
        })?;
        let completed = state
            .runtime
            .node_completed
            .get_mut(&node_id)
            .ok_or_else(|| {
                Failure::Message(format!(
                    "missing forward pipeline counter for node {node_id}"
                ))
            })?;
        *completed = completed.saturating_add(1);
        state.runtime.pipeline_completions = state.runtime.pipeline_completions.saturating_add(1);
        if Some(node_id) == state.runtime.sink_id {
            state
                .runtime
                .completed_outputs
                .push_back(completion.delivery);
        }
        if state.runtime.pipeline_completions == state.runtime.pipeline_target_completions {
            state.runtime.inputs = std::mem::take(&mut state.runtime.next_inputs);
        }

        let mut topology = state.topology.lock().unwrap();
        topology.record_forward_completed(node_id);
        topology.record_forward_started(completion.sent_node_ids);
        topology
            .node_phase_annotations
            .insert(node_id, format!("pipeline input {}", input_index + 1));
        Ok(())
    }
}

/// Returns the oldest completed sink output with its public artifact type.
pub struct TakeForwardPipelineOutput<S, Output>(PhantomData<fn() -> (S, Output)>);

#[jungle::action(carry = black_hole_type::ArtifactDelivery<()>)]
impl<S, Output> Action for TakeForwardPipelineOutput<S, Output>
where
    Output: Send + 'static,
{
    type Effect = NoEffect;
    type Input = ();
    type Output = black_hole_type::ArtifactDelivery<Output>;

    fn emit(
        state: &super::ForwardSunState<S>,
        _input: (),
    ) -> ((), black_hole_type::ArtifactDelivery<()>) {
        let delivery = *state
            .runtime
            .completed_outputs
            .front()
            .expect("output predicate guarantees a completed forward output");
        ((), delivery)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_type::ArtifactDelivery<()>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("take forward pipeline output failed".to_string()))?;
        state
            .runtime
            .completed_outputs
            .pop_front()
            .ok_or_else(|| Failure::Message("completed forward output disappeared".to_string()))?;
        Ok(black_hole_type::ArtifactDelivery {
            emission_id: black_hole_type::ObjectRef::new(carry.emission_id.id()),
            recv: carry.recv,
            send: carry.send,
        })
    }
}

/// Initializes one dependency-aware forward pass using a program-neutral
/// frontier and endpoint set. The typed boundary is erased only while the
/// already contract-validated graph is scheduled.
pub struct PrepareForwardPass<S, Input>(PhantomData<fn() -> (S, Input)>);

#[jungle::action(carry = black_hole_type::ArtifactDelivery<Input>)]
impl<S, Input> Action for PrepareForwardPass<S, Input>
where
    Input: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_type::ArtifactDelivery<Input>;
    type Output = black_hole_type::ArtifactDelivery<()>;

    fn emit(
        _state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> ((), black_hole_type::ArtifactDelivery<Input>) {
        ((), input)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_type::ArtifactDelivery<Input>,
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
        Ok(black_hole_type::ArtifactDelivery {
            emission_id: black_hole_type::ObjectRef::new(carry.emission_id.id()),
            recv: carry.recv,
            send: carry.send,
        })
    }
}

/// Sends the typed input artifact to every root of a forward pass.
pub struct SendForwardRoots<S>(PhantomData<fn() -> S>);

#[jungle::action(carry = black_hole_type::ArtifactDelivery<()>)]
impl<S> Action for SendForwardRoots<S> {
    type Effect = SendRootArtifactDeliveryEffect<()>;
    type Input = black_hole_type::ArtifactDelivery<()>;
    type Output = black_hole_type::ArtifactDelivery<()>;

    fn emit(
        state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> (
        SendRootArtifactDeliveryInput<()>,
        black_hole_type::ArtifactDelivery<()>,
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
        carry: black_hole_type::ArtifactDelivery<()>,
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
    type Input = black_hole_type::ArtifactDelivery<()>;
    type Output = black_hole_type::ArtifactDelivery<()>;

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

#[jungle::action(carry = black_hole_type::ArtifactDelivery<()>)]
impl<S, Output> Action for CompleteForwardPass<S, Output>
where
    Output: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_type::ArtifactDelivery<()>;
    type Output = black_hole_type::ArtifactDelivery<Output>;

    fn emit(
        _state: &super::ForwardSunState<S>,
        input: Self::Input,
    ) -> ((), black_hole_type::ArtifactDelivery<()>) {
        ((), input)
    }

    fn absorb(
        state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: black_hole_type::ArtifactDelivery<()>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("complete forward pass failed".to_string()))?;
        state.runtime.inputs = std::mem::take(&mut state.runtime.next_inputs);
        state.runtime.outputs.clear();
        Ok(black_hole_type::ArtifactDelivery {
            emission_id: black_hole_type::ObjectRef::new(carry.emission_id.id()),
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
    type Input = black_hole_type::ArtifactDelivery<T>;
    type Output = ();

    fn emit(_state: &super::ForwardSunState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut super::ForwardSunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("discard forward output failed".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::PendingForwardPipelineInputs;
    use crate::topology::PortTarget;

    struct TestForwardAnimal;

    impl Animal for TestForwardAnimal {
        type Id = ();
        type Generation = ();
        type State = super::super::ForwardSunState;
        type Seed = ();
        type Flow = ();
    }

    fn begin_pipeline<const MAX_IN_FLIGHT: usize>(state: &mut super::super::ForwardSunState) {
        type Begin<const N: usize> =
            <BeginForwardPipeline<(), N> as Action>::Bind<TestForwardAnimal>;
        <Begin<MAX_IN_FLIGHT> as BoundAction<TestForwardAnimal>>::absorb(state, Ok(())).unwrap();
    }

    fn delivery(seed: u128) -> black_hole_type::ArtifactDelivery<()> {
        black_hole_type::ArtifactDelivery {
            emission_id: black_hole_type::ObjectRef::new(Uuid::from_u128(seed)),
            recv: Uuid::nil(),
            send: Uuid::nil(),
        }
    }

    #[test]
    fn strategy_limit_bounds_generated_pipeline_inputs() {
        let mut state = super::super::ForwardSunState::default();
        state.runtime.pipeline_inputs = vec![delivery(1), delivery(2), delivery(3)];
        state.runtime.completed_outputs.push_back(delivery(4));

        begin_pipeline::<2>(&mut state);

        assert_eq!(state.runtime.pipeline_window, 2);
        assert!(state.runtime.pipeline_inputs.is_empty());
        assert!(state.runtime.completed_outputs.is_empty());
        assert!(PendingForwardPipelineInputs::<()>::eval(&(&state, &())));

        state.runtime.pipeline_inputs = vec![delivery(5), delivery(6)];
        assert!(!PendingForwardPipelineInputs::<()>::eval(&(&state, &())));
    }

    #[test]
    fn zero_strategy_limit_still_allows_one_input() {
        let mut state = super::super::ForwardSunState::default();
        begin_pipeline::<0>(&mut state);
        assert_eq!(state.runtime.pipeline_window, 1);
    }

    #[test]
    fn next_input_enters_root_while_previous_input_advances_downstream() {
        let mut state = super::super::ForwardSunState::<()>::default();
        {
            let mut topology = state.topology.lock().unwrap();
            topology.journey_ids.insert(0, Uuid::new_v4());
            topology.journey_ids.insert(1, Uuid::new_v4());
            topology.incoming.insert(0, vec![]);
            topology.incoming.insert(1, vec![0]);
            topology.outgoing.insert(
                0,
                vec![PortTarget {
                    port_id: 1,
                    vertex_id: 1,
                }],
            );
            topology.outgoing.insert(1, vec![]);
        }
        state.runtime.pipeline_inputs = vec![delivery(1), delivery(2)];
        state.runtime.node_completed = HashMap::from([(0, 1), (1, 0)]);
        state.runtime.root_sent = HashMap::from([(0, 2)]);

        let topology = state.topology.lock().unwrap();
        assert_eq!(pipeline_ready_nodes(&state, &topology), vec![0, 1]);
    }

    #[test]
    fn root_waits_for_its_previous_input_before_accepting_another() {
        let mut state = super::super::ForwardSunState::<()>::default();
        {
            let mut topology = state.topology.lock().unwrap();
            topology.journey_ids.insert(0, Uuid::new_v4());
            topology.incoming.insert(0, vec![]);
            topology.outgoing.insert(0, vec![]);
        }
        state.runtime.pipeline_inputs = vec![delivery(1), delivery(2)];
        state.runtime.node_completed = HashMap::from([(0, 0)]);
        state.runtime.root_sent = HashMap::from([(0, 1)]);

        let topology = state.topology.lock().unwrap();
        assert_eq!(pipeline_ready_nodes(&state, &topology), vec![0]);
        drop(topology);

        state.runtime.node_completed.insert(0, 1);
        let topology = state.topology.lock().unwrap();
        assert!(pipeline_ready_nodes(&state, &topology).is_empty());
    }
}
