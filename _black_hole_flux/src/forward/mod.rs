//! Neutral dependency-aware forward execution primitive.
//!
//! [`ForwardPassWithState`] schedules a contract-validated graph once per
//! typed artifact: seed the roots, process ready nodes as they complete, and
//! rotate inboxes for the next request. No propagation or potentiation
//! semantics are involved.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet};
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

/// Minimal serving driver: finalize once, then execute one neutral forward
/// pass for every artifact emitted by `Source`.
#[derive(Flow)]
pub struct ServeFlow<Source, Input: Send + 'static, Output: Send + 'static, S>(
    Step<crate::compile::action::FinalizeForwardGraph<S>>,
    While<Always<ForwardSunState<S>, ()>, ServeRequest<Source, Input, Output, S>>,
);

/// Serving driver variant that applies a policy to each completed sink.
#[derive(Flow)]
pub struct ServeFlowWithPolicy<Source, Input: Send + 'static, Output: Send + 'static, S, Policy>(
    Step<crate::compile::action::FinalizeForwardGraph<S>>,
    While<Always<ForwardSunState<S>, ()>, ServeRequestWithPolicy<Source, Input, Output, S, Policy>>,
);

#[derive(Flow)]
pub struct ServeRequestWithPolicy<Source, Input: Send + 'static, Output: Send + 'static, S, Policy>(
    Source,
    ForwardPassWithState<Input, Output, S>,
    Policy,
);
