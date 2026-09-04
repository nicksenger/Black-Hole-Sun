//! Neutral dependency-aware forward execution primitive.
//!
//! [`ForwardPassWithState`] schedules a contract-validated graph once per
//! typed artifact: seed the roots, process ready nodes as they complete, and
//! rotate inboxes for the next request. No propagation or potentiation
//! semantics are involved.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_type::ObjectId;
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;

use crate::topology::{
    SunAppearance, SunEdgeAppearance, SunNodeAppearance, SunNodeState, SunStateView, SunTopology,
    SunTopologyState,
};

/// Neutral scheduler state used by [`ForwardPass`].
#[derive(Clone, Default, Debug)]
pub struct ForwardRuntime<S> {
    pub inner: S,
    pub pending: HashMap<u32, usize>,
    pub ready: HashSet<u32>,
    pub inputs: HashMap<u32, ObjectId>,
    pub next_inputs: HashMap<u32, ObjectId>,
    pub outputs: HashMap<u32, ObjectId>,
    pub sink_id: Option<u32>,
    /// Inputs collected for the current pipeline window.
    pub pipeline_inputs: Vec<black_hole_type::ArtifactDelivery<()>>,
    /// Per-input mailboxes used to drive each input port.
    pub pipeline_input_ids: Vec<HashMap<u32, ObjectId>>,
    /// Per-input mailboxes used to collect each node output.
    pub pipeline_output_ids: Vec<HashMap<u32, ObjectId>>,
    /// Number of inputs completed by each node in the current window.
    pub node_completed: HashMap<u32, usize>,
    /// Number of inputs sent to each root in the current window.
    pub root_sent: HashMap<u32, usize>,
    /// Sink outputs waiting to be consumed by the serving policy.
    pub completed_outputs: VecDeque<black_hole_type::ArtifactDelivery<()>>,
    /// Number of node/input tasks completed in the current window.
    pub pipeline_completions: usize,
    /// Total node/input tasks in the current window.
    pub pipeline_target_completions: usize,
    /// Number of inputs collected for each pipeline window.
    pub pipeline_window: usize,
}

/// State for forward programs; it contains no two-sided strategy fields.
#[derive(Clone, Debug)]
pub struct ForwardSunState<S = ()> {
    pub topology: Arc<Mutex<SunTopology>>,
    pub runtime: ForwardRuntime<S>,
}

impl<S: Default> Default for ForwardSunState<S> {
    fn default() -> Self {
        Self {
            topology: Arc::new(Mutex::new(SunTopology::default())),
            runtime: ForwardRuntime::default(),
        }
    }
}

impl<S> SunTopologyState for ForwardSunState<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

impl<S> ForwardSunState<S> {
    pub fn appearance(&self) -> SunAppearance {
        neutral_appearance(&self.topology.lock().unwrap())
    }
}

impl<S> SunStateView for ForwardSunState<S> {
    fn sun_appearance(&self) -> SunAppearance {
        self.appearance()
    }
}

/// State for programs that only need a compiled topology and private payload.
#[derive(Clone, Debug)]
pub struct NeutralSunState<S = ()> {
    pub topology: Arc<Mutex<SunTopology>>,
    pub inner: S,
}

impl<S: Default> Default for NeutralSunState<S> {
    fn default() -> Self {
        Self {
            topology: Arc::new(Mutex::new(SunTopology::default())),
            inner: S::default(),
        }
    }
}

impl<S> SunTopologyState for NeutralSunState<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

impl<S> NeutralSunState<S> {
    pub fn appearance(&self) -> SunAppearance {
        neutral_appearance(&self.topology.lock().unwrap())
    }
}

impl<S> SunStateView for NeutralSunState<S> {
    fn sun_appearance(&self) -> SunAppearance {
        self.appearance()
    }
}

fn neutral_appearance(topology: &SunTopology) -> SunAppearance {
    let mut nodes = topology
        .journey_ids
        .keys()
        .copied()
        .map(|id| SunNodeAppearance {
            id,
            journey_id: topology.journey_ids[&id],
            warp_journey_id: topology
                .warp_journey_ids
                .get(&id)
                .copied()
                .unwrap_or_default(),
            label: topology
                .node_labels
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("cell {id}")),
            input_ports: topology.vertex_ports.get(&id).cloned().unwrap_or_default(),
            state: SunNodeState::Idle,
            state_sequence: topology
                .node_state_sequences
                .get(&id)
                .copied()
                .unwrap_or_default(),
            grad_step: 1,
            operational_state: topology
                .node_operational_states
                .get(&id)
                .copied()
                .unwrap_or_default(),
            phase_annotation: topology.node_phase_annotations.get(&id).cloned(),
        })
        .collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.id);
    let mut edges = topology
        .outgoing
        .iter()
        .flat_map(|(&source, targets)| {
            targets.iter().map(move |target| SunEdgeAppearance {
                source,
                target: target.vertex_id,
                target_port: target.port_id,
            })
        })
        .collect::<Vec<_>>();
    edges.sort_by_key(|edge| (edge.source, edge.target, edge.target_port));
    edges.dedup();
    SunAppearance {
        finalized: topology.finalized,
        grad_steps: 1,
        nodes,
        edges,
    }
}

/// Predicate for a neutral dependency-aware forward pass.
pub struct PendingForwardWork<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&ForwardSunState<S>, &black_hole_type::ArtifactDelivery<()>)>
    for PendingForwardWork<S>
{
    fn eval((state, _): &(&ForwardSunState<S>, &black_hole_type::ArtifactDelivery<()>)) -> bool {
        !state.runtime.pending.is_empty()
    }
}

/// One completion step in a neutral typed forward pass.
#[derive(Flow)]
pub struct ForwardPassLoop<S>(Step<action::ProcessForwardNode<S>>);

/// Dependency-aware, operation-typed graph execution primitive.
///
/// This primitive has no perturbation, up/down, policy, or potentiation
/// semantics. Programs provide one typed root artifact; the scheduler runs
/// every ready node once and routes the sink completion.
#[derive(Flow)]
pub struct ForwardPassWithState<Input: Send + 'static, Output: Send + 'static, S>(
    Step<action::PrepareForwardPass<S, Input>>,
    Step<action::SendForwardRoots<S>>,
    While<PendingForwardWork<S>, ForwardPassLoop<S>>,
    Step<action::CompleteForwardPass<S, Output>>,
);

pub type ForwardPass<Input, Output = Input> = ForwardPassWithState<Input, Output, ()>;

/// One forward-only serving request.
#[derive(Flow)]
pub struct ServeRequest<Source, Input: Send + 'static, Output: Send + 'static, S>(
    Source,
    ForwardPassWithState<Input, Output, S>,
    Step<action::DiscardForwardOutput<S, Output>>,
);

/// Predicate that fills one bounded forward pipeline window.
pub struct PendingForwardPipelineInputs<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&ForwardSunState<S>, &())> for PendingForwardPipelineInputs<S> {
    fn eval((state, _): &(&ForwardSunState<S>, &())) -> bool {
        state.runtime.pipeline_inputs.len() < state.runtime.pipeline_window
    }
}

/// Predicate that advances the scheduler until every node/input task completes.
pub struct PendingForwardPipelineWork<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&ForwardSunState<S>, &())> for PendingForwardPipelineWork<S> {
    fn eval((state, _): &(&ForwardSunState<S>, &())) -> bool {
        state.runtime.pipeline_completions < state.runtime.pipeline_target_completions
    }
}

/// Predicate that drains completed sink outputs through a policy or sink.
pub struct PendingForwardOutputs<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&ForwardSunState<S>, &())> for PendingForwardOutputs<S> {
    fn eval((state, _): &(&ForwardSunState<S>, &())) -> bool {
        !state.runtime.completed_outputs.is_empty()
    }
}

/// Generates and stores one input for the current pipeline window.
#[derive(Flow)]
pub struct CollectForwardPipelineInput<Source, Input: Send + 'static, S>(
    Source,
    Step<action::StoreForwardPipelineInput<S, Input>>,
);

/// Sends newly-ready roots and consumes the next ready node completion.
#[derive(Flow)]
pub struct ForwardPipelineProgress<S>(
    Step<action::SendReadyForwardRoots<S>>,
    Step<action::ProcessReadyForwardPipelineNode<S>>,
);

/// Applies the serving policy to one completed sink output.
#[derive(Flow)]
pub struct ApplyForwardPolicy<Output: Send + 'static, S, Policy>(
    Step<action::TakeForwardPipelineOutput<S, Output>>,
    Policy,
);

/// Discards one completed sink output.
#[derive(Flow)]
pub struct DiscardForwardPipelineOutput<Output: Send + 'static, S>(
    Step<action::TakeForwardPipelineOutput<S, Output>>,
    Step<action::DiscardForwardOutput<S, Output>>,
);

/// One pipelined serving window without a policy.
#[derive(Flow)]
pub struct ServePipelineWindow<Source, Input: Send + 'static, Output: Send + 'static, S>(
    Step<action::BeginForwardPipeline<S>>,
    While<PendingForwardPipelineInputs<S>, CollectForwardPipelineInput<Source, Input, S>>,
    Step<action::PrepareForwardPipeline<S>>,
    While<PendingForwardPipelineWork<S>, ForwardPipelineProgress<S>>,
    While<PendingForwardOutputs<S>, DiscardForwardPipelineOutput<Output, S>>,
);

/// One pipelined serving window with a policy applied as sink outputs arrive.
#[derive(Flow)]
pub struct ServePipelineWindowWithPolicy<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    S,
    Policy,
>(
    Step<action::BeginForwardPipeline<S>>,
    While<PendingForwardPipelineInputs<S>, CollectForwardPipelineInput<Source, Input, S>>,
    Step<action::PrepareForwardPipeline<S>>,
    While<PendingForwardPipelineWork<S>, ForwardPipelineProgressWithPolicy<Output, S, Policy>>,
);

/// One scheduler tick followed by any policy work made ready by that tick.
#[derive(Flow)]
pub struct ForwardPipelineProgressWithPolicy<Output: Send + 'static, S, Policy>(
    ForwardPipelineProgress<S>,
    While<PendingForwardOutputs<S>, ApplyForwardPolicy<Output, S, Policy>>,
);

/// Serving driver: finalize once, then continuously execute bounded pipeline
/// windows. The window is sized from the compiled graph, allowing successive
/// inputs to occupy different nodes concurrently.
#[derive(Flow)]
pub struct ServeFlow<Source, Input: Send + 'static, Output: Send + 'static, S>(
    Step<crate::compile::action::FinalizeForwardGraph<S>>,
    While<Always<ForwardSunState<S>, ()>, ServePipelineWindow<Source, Input, Output, S>>,
);

/// Serving driver variant that applies a policy to each completed sink.
#[derive(Flow)]
pub struct ServeFlowWithPolicy<Source, Input: Send + 'static, Output: Send + 'static, S, Policy>(
    Step<crate::compile::action::FinalizeForwardGraph<S>>,
    While<
        Always<ForwardSunState<S>, ()>,
        ServePipelineWindowWithPolicy<Source, Input, Output, S, Policy>,
    >,
);

#[derive(Flow)]
pub struct ServeRequestWithPolicy<Source, Input: Send + 'static, Output: Send + 'static, S, Policy>(
    Source,
    ForwardPassWithState<Input, Output, S>,
    Policy,
);
