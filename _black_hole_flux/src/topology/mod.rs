//! Type-level graph descriptors and the neutral runtime topology model.
//!
//! This module is strategy-neutral: it describes what a compiled Sun graph
//! looks like (vertices, typed edges, contracts, ports) and how progress is
//! observed. Strategy-specific state lives in [`crate::programs`].

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_spec::{BackwardContract, QwenDarkInference, TensorContract};
use black_hole_type::{ContractDescriptor, ObjectId};
use jungle_sdk::prelude::*;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Descriptors — type-level vertices and their input ports
// ---------------------------------------------------------------------------

/// Declares which operation contract a spawned node executes.
///
/// Legacy animals are Qwen nodes by default. Generic operation animals opt in
/// explicitly, preventing a topology descriptor from advertising an
/// unrelated contract for the spawned implementation.
pub trait OperationNode<Op: TensorContract>: Animal {}

impl<A> OperationNode<QwenDarkInference> for A where A: Animal {}

/// Type-level unary vertex with one input port and a list of output ports.
///
/// `P` is both the public input port and the deterministic internal vertex key.
pub struct Unary<
    P: Unsigned,
    A: Animal + OperationNode<Op>,
    E: NodeIdsFromList + DeclaredEdges<Op>,
    Op: TensorContract = QwenDarkInference,
>(
    PhantomData<P>,
    PhantomData<A>,
    PhantomData<E>,
    PhantomData<Op>,
);

/// Type-level binary vertex whose two input ports share one spawned animal and
/// one output mailbox per propagation pass.
///
/// `P1` is the deterministic internal vertex key; both `P1` and `P2` resolve
/// to it during graph finalization.
pub struct Binary<
    P1: Unsigned,
    P2: Unsigned,
    A: Animal + OperationNode<Op>,
    E: NodeIdsFromList + DeclaredEdges<Op>,
    Op: TensorContract = QwenDarkInference,
>(
    PhantomData<P1>,
    PhantomData<P2>,
    PhantomData<A>,
    PhantomData<E>,
    PhantomData<Op>,
);

/// A destination port paired with the operation contract that owns it.
///
/// Use this inside [`TypedEdges`]. The destination operation's input bundle
/// is required to equal the source operation's output bundle when the graph
/// is compiled through [`BlackHole`].
pub struct Edge<P: Unsigned, Destination: TensorContract>(PhantomData<(P, Destination)>);

/// Explicitly typed output-edge list for a node descriptor.
///
/// Legacy `list![U1, U2]` output lists remain accepted and mean Qwen-to-Qwen
/// edges. Generic graphs use `TypedEdges<list![Edge<U1, NextOp>]>`.
pub struct TypedEdges<E>(PhantomData<E>);

/// Typed edges that additionally prove the counter-propagating gradient
/// bundle matches (`Destination::InputGrad = Source::OutputGrad`).
pub struct BackwardTypedEdges<E>(PhantomData<E>);

/// Type-level warp vertex that composes a nested Sun animal behind a boundary
/// cell that handles ingress/egress behavior in the parent graph.
///
/// `P` is both the public input port and deterministic internal vertex key.
/// The warp animal is spawned first, then the boundary animal is spawned with a
/// [`BoundaryInit`] that includes the warp journey id.
pub struct Warp<
    P: Unsigned,
    WarpAnimal: Animal + Observe,
    BoundaryAnimal: Animal + OperationNode<Op>,
    E: NodeIdsFromList + DeclaredEdges<Op>,
    Op: TensorContract = QwenDarkInference,
>(
    PhantomData<P>,
    PhantomData<WarpAnimal>,
    PhantomData<BoundaryAnimal>,
    PhantomData<E>,
    PhantomData<Op>,
);

/// Initialization payload for one spawned warp boundary cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BoundaryInit {
    /// First propagation mailbox that this boundary should await.
    pub recv_id: ObjectId,
    /// Number of propagation microsteps to run per perturbation phase.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    /// Journey id of the spawned nested warp animal associated with this boundary.
    pub warp_journey_id: Uuid,
}

impl Default for BoundaryInit {
    fn default() -> Self {
        Self {
            recv_id: ObjectId::nil(),
            grad_steps: default_gradient_accumulation_steps(),
            warp_journey_id: Uuid::nil(),
        }
    }
}

/// The latest orchestration phase reached by a Sun node.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum SunNodeState {
    /// The node has been spawned but has not been sent propagation.
    #[default]
    Idle,
    /// The first propagation pass has been sent to the node.
    Propagation1,
    /// The first pass has emitted and the second pass has been sent to the node.
    Propagation2,
    /// The step's loss has been sent to the node.
    Optimization,
}

/// Program-independent execution status for an observable topology node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SunOperationalState {
    #[default]
    Queued,
    Running,
    Succeeded,
    Failed,
}

/// One node in the observable Sun topology.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SunNodeAppearance {
    pub id: u32,
    /// Journey ID of the spawned child workflow represented by this node.
    pub journey_id: Uuid,
    /// Journey ID of the nested warp animal when this node is a warp vertex;
    /// nil for ordinary vertices.
    #[serde(default)]
    pub warp_journey_id: Uuid,
    pub label: String,
    pub input_ports: Vec<u32>,
    /// Legacy two-sided phase retained during the compatibility migration.
    pub state: SunNodeState,
    /// Monotonic logical phase position, including phases crossed between snapshots.
    pub state_sequence: u64,
    /// 1-based gradient accumulation step currently associated with this node.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_step: usize,
    /// Neutral execution state shared by training and serving programs.
    #[serde(default)]
    pub operational_state: SunOperationalState,
    /// Optional strategy-selected phase label (for example `propagation 1`,
    /// `potentiation`, or `forward`).
    #[serde(default)]
    pub phase_annotation: Option<String>,
}

/// One port-aware directed edge in the observable Sun topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SunEdgeAppearance {
    pub source: u32,
    pub target: u32,
    pub target_port: u32,
}

/// Serializable projection of a running Black Hole Sun.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SunAppearance {
    /// True after the runtime graph has been resolved and validated.
    pub finalized: bool,
    /// Number of gradient accumulation microsteps per optimization step.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    pub nodes: Vec<SunNodeAppearance>,
    pub edges: Vec<SunEdgeAppearance>,
}

pub(crate) fn default_gradient_accumulation_steps() -> usize {
    1
}

/// Immutable/resolved topology and program-neutral observation data.
///
/// Program mailboxes, phase counters, and strategy state deliberately do not
/// live here. They belong to the runtime selected by [`SunProgram`].
#[derive(Optic, Clone, Default, Debug)]
pub struct SunTopology {
    pub finalized: bool,
    pub journey_ids: HashMap<u32, Uuid>,
    pub warp_journey_ids: HashMap<u32, Uuid>,
    pub node_labels: HashMap<u32, String>,
    pub node_operational_states: HashMap<u32, SunOperationalState>,
    pub node_phase_annotations: HashMap<u32, String>,
    pub node_state_sequences: HashMap<u32, u64>,
    pub vertex_ports: HashMap<u32, Vec<u32>>,
    pub port_vertices: HashMap<u32, u32>,
    pub declared_outputs: HashMap<u32, Vec<u32>>,
    pub node_contracts: HashMap<u32, ContractDescriptor>,
    pub declared_edges: HashMap<u32, Vec<DeclaredEdge>>,
    pub duplicate_ports: HashSet<u32>,
    pub incoming: HashMap<u32, Vec<u32>>,
    pub outgoing: HashMap<u32, Vec<PortTarget>>,
}

impl SunTopology {
    pub(crate) fn record_forward_started(&mut self, node_ids: impl IntoIterator<Item = u32>) {
        for node_id in node_ids {
            self.node_operational_states
                .insert(node_id, SunOperationalState::Running);
            self.node_phase_annotations
                .insert(node_id, "forward".to_string());
            *self.node_state_sequences.entry(node_id).or_default() += 1;
        }
    }

    pub(crate) fn record_forward_completed(&mut self, node_id: u32) {
        self.node_operational_states
            .insert(node_id, SunOperationalState::Succeeded);
    }

    /// Records that a pipeline step has begun processing its stages.
    pub(crate) fn record_pipeline_started(&mut self, node_ids: impl IntoIterator<Item = u32>) {
        for node_id in node_ids {
            let was_running = self.node_operational_states.get(&node_id)
                == Some(&SunOperationalState::Running);
            self.node_operational_states
                .insert(node_id, SunOperationalState::Running);
            self.node_phase_annotations
                .insert(node_id, "pipeline".to_string());
            if !was_running {
                *self.node_state_sequences.entry(node_id).or_default() += 1;
            }
        }
    }

    /// Records that a pipeline step completed processing a stage.
    pub(crate) fn record_pipeline_completed(&mut self, node_id: u32) {
        self.node_operational_states
            .insert(node_id, SunOperationalState::Succeeded);
    }

    /// Build a deterministic, serializable view of the resolved graph.
    ///
    /// Node phase and gradient-step detail is strategy-specific and lives in
    /// each program's state; this view carries only what every Sun shares: the
    /// resolved nodes, their labels and ports, whatever operational state has
    /// been recorded on the topology, and the edges.
    pub fn appearance(&self) -> SunAppearance {
        let mut nodes = self
            .journey_ids
            .keys()
            .copied()
            .map(|id| SunNodeAppearance {
                id,
                journey_id: self.journey_ids[&id],
                warp_journey_id: self
                    .warp_journey_ids
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                label: self
                    .node_labels
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("cell {id}")),
                input_ports: self.vertex_ports.get(&id).cloned().unwrap_or_default(),
                state: SunNodeState::Idle,
                state_sequence: self
                    .node_state_sequences
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                grad_step: 1,
                operational_state: self
                    .node_operational_states
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                phase_annotation: self.node_phase_annotations.get(&id).cloned(),
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.id);
        let mut edges = self
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
            finalized: self.finalized,
            grad_steps: 1,
            nodes,
            edges,
        }
    }
}

/// Access to the neutral topology shared by every Sun program state.
pub trait SunTopologyState {
    fn topology(&self) -> &Arc<Mutex<SunTopology>>;
}

pub trait SunStateView: SunTopologyState {
    fn sun_appearance(&self) -> SunAppearance;
}

/// Trait that converts a type-level list of typenum integers into a runtime
/// vector of node IDs (u32 values).
pub trait NodeIdsFromList {
    fn node_ids() -> Vec<u32>;
}

impl NodeIdsFromList for Empty {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<H, T> NodeIdsFromList for List<(H, T)>
where
    H: Unsigned,
    T: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![<H as Unsigned>::U32];
        ids.extend(T::node_ids());
        ids
    }
}

impl NodeIdsFromList for TypedEdges<Empty> {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<P, Destination, T> NodeIdsFromList for TypedEdges<List<(Edge<P, Destination>, T)>>
where
    P: Unsigned,
    Destination: TensorContract,
    TypedEdges<T>: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![P::U32];
        ids.extend(<TypedEdges<T> as NodeIdsFromList>::node_ids());
        ids
    }
}

impl NodeIdsFromList for BackwardTypedEdges<Empty> {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<P, Destination, T> NodeIdsFromList for BackwardTypedEdges<List<(Edge<P, Destination>, T)>>
where
    P: Unsigned,
    Destination: BackwardContract,
    BackwardTypedEdges<T>: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![P::U32];
        ids.extend(<BackwardTypedEdges<T> as NodeIdsFromList>::node_ids());
        ids
    }
}

/// Produces runtime edge descriptors while enforcing compile-time bundle
/// equality between every source output and destination input.
pub trait DeclaredEdges<Source: TensorContract>: NodeIdsFromList {
    fn declared_edges() -> Vec<DeclaredEdge>;
}

impl<Source: TensorContract> DeclaredEdges<Source> for Empty {
    fn declared_edges() -> Vec<DeclaredEdge> {
        Vec::new()
    }
}

// Compatibility for the original numeric-only topology syntax. It is
// intentionally limited to the Qwen artifact bundle; generic graphs must name
// destination contracts with `TypedEdges`.
impl<Source, P, T> DeclaredEdges<Source> for List<(P, T)>
where
    Source: TensorContract<Output = <QwenDarkInference as TensorContract>::Input>,
    P: Unsigned,
    T: DeclaredEdges<Source>,
{
    fn declared_edges() -> Vec<DeclaredEdge> {
        let mut edges = vec![DeclaredEdge {
            port_id: P::U32,
            source_contract: Source::descriptor(),
            destination_contract: QwenDarkInference::descriptor(),
        }];
        edges.extend(T::declared_edges());
        edges
    }
}

impl<Source: TensorContract> DeclaredEdges<Source> for TypedEdges<Empty> {
    fn declared_edges() -> Vec<DeclaredEdge> {
        Vec::new()
    }
}

impl<Source, P, Destination, T> DeclaredEdges<Source>
    for TypedEdges<List<(Edge<P, Destination>, T)>>
where
    Source: TensorContract,
    Destination: TensorContract<Input = Source::Output>,
    P: Unsigned,
    TypedEdges<T>: DeclaredEdges<Source>,
{
    fn declared_edges() -> Vec<DeclaredEdge> {
        let mut edges = vec![DeclaredEdge {
            port_id: P::U32,
            source_contract: Source::descriptor(),
            destination_contract: Destination::descriptor(),
        }];
        edges.extend(<TypedEdges<T> as DeclaredEdges<Source>>::declared_edges());
        edges
    }
}

impl<Source: BackwardContract> DeclaredEdges<Source> for BackwardTypedEdges<Empty> {
    fn declared_edges() -> Vec<DeclaredEdge> {
        Vec::new()
    }
}

impl<Source, P, Destination, T> DeclaredEdges<Source>
    for BackwardTypedEdges<List<(Edge<P, Destination>, T)>>
where
    Source: BackwardContract,
    Destination: BackwardContract<Input = Source::Output, InputGrad = Source::OutputGrad>,
    P: Unsigned,
    BackwardTypedEdges<T>: DeclaredEdges<Source>,
{
    fn declared_edges() -> Vec<DeclaredEdge> {
        let mut edges = vec![DeclaredEdge {
            port_id: P::U32,
            source_contract: Source::descriptor(),
            destination_contract: Destination::descriptor(),
        }];
        edges.extend(<BackwardTypedEdges<T> as DeclaredEdges<Source>>::declared_edges());
        edges
    }
}

/// Erased form of a compile-time checked edge. Separate binaries and rolling
/// deployments must still agree on these descriptors at runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredEdge {
    pub port_id: u32,
    pub source_contract: ContractDescriptor,
    pub destination_contract: ContractDescriptor,
}

/// A resolved edge target. `port_id` identifies the destination mailbox while
/// `vertex_id` identifies the single animal/output shared by all of its ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortTarget {
    pub port_id: u32,
    pub vertex_id: u32,
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

pub(crate) fn pending_dependency_counts(topology: &SunTopology) -> HashMap<u32, usize> {
    topology
        .journey_ids
        .keys()
        .map(|&node_id| {
            let unresolved = topology.incoming.get(&node_id).map_or(0, Vec::len);
            (node_id, unresolved)
        })
        .collect()
}

pub(crate) fn initial_ready_nodes(pending: &HashMap<u32, usize>) -> HashSet<u32> {
    pending
        .iter()
        .filter_map(|(&node_id, &unresolved)| (unresolved == 0).then_some(node_id))
        .collect()
}

pub(crate) fn sorted_node_ids(nodes: &HashSet<u32>) -> Vec<u32> {
    let mut ready: Vec<_> = nodes.iter().copied().collect();
    ready.sort_unstable();
    ready
}

pub(crate) fn advance_frontier(
    pending: &mut HashMap<u32, usize>,
    ready: &mut HashSet<u32>,
    node_id: u32,
    outgoing: &[PortTarget],
) -> Result<(), Failure> {
    let unresolved = pending
        .get(&node_id)
        .copied()
        .ok_or_else(|| Failure::Message(format!("completed node {node_id} is not pending")))?;
    if unresolved != 0 {
        return Err(Failure::Message(format!(
            "completed node {node_id} still has {unresolved} unresolved predecessors"
        )));
    }
    if !ready.contains(&node_id) {
        return Err(Failure::Message(format!(
            "completed node {node_id} is not in the ready frontier"
        )));
    }

    let mut decrements = HashMap::<u32, usize>::new();
    for target in outgoing {
        *decrements.entry(target.vertex_id).or_default() += 1;
    }
    for (&target_id, &decrement) in &decrements {
        let target_unresolved = pending.get(&target_id).ok_or_else(|| {
            Failure::Message(format!("downstream node {target_id} is not pending"))
        })?;
        if *target_unresolved < decrement {
            return Err(Failure::Message(format!(
                "downstream node {target_id} has {target_unresolved} unresolved predecessors, \
                 cannot resolve {decrement}"
            )));
        }
    }

    pending.remove(&node_id);
    ready.remove(&node_id);
    for (target_id, decrement) in decrements {
        let target_unresolved = pending
            .get_mut(&target_id)
            .expect("downstream node was validated above");
        *target_unresolved -= decrement;
        if *target_unresolved == 0 {
            ready.insert(target_id);
        }
    }
    Ok(())
}

pub(crate) fn port_ids(topology: &SunTopology) -> Vec<u32> {
    topology.port_vertices.keys().copied().collect()
}

pub(crate) fn vertex_ids(topology: &SunTopology) -> Vec<u32> {
    topology.journey_ids.keys().copied().collect()
}

pub(crate) fn root_vertex_ids(topology: &SunTopology) -> Vec<u32> {
    let mut roots: Vec<_> = topology
        .incoming
        .iter()
        .filter_map(|(&node_id, sources)| sources.is_empty().then_some(node_id))
        .collect();
    roots.sort_unstable();
    roots
}

#[cfg(test)]
mod tests {
    use super::{SunNodeState, SunOperationalState, SunTopology};
    use uuid::Uuid;

    #[test]
    fn pipeline_status_is_published_in_appearance() {
        let mut topology = SunTopology::default();
        topology.journey_ids.insert(4, Uuid::new_v4());

        topology.record_pipeline_started([4]);
        let appearance = topology.appearance();
        assert_eq!(appearance.nodes[0].state, SunNodeState::Idle);
        assert_eq!(
            appearance.nodes[0].operational_state,
            SunOperationalState::Running
        );
        assert_eq!(
            appearance.nodes[0].phase_annotation.as_deref(),
            Some("pipeline")
        );
        assert_eq!(appearance.nodes[0].state_sequence, 1);

        topology.record_pipeline_completed(4);
        assert_eq!(
            topology.appearance().nodes[0].operational_state,
            SunOperationalState::Succeeded
        );
    }
}
