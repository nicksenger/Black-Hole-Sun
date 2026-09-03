//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use action::GenUuid;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_contract::{QwenDarkInference, TensorContract};
use black_hole_spec::{ContractDescriptor, ObjectId, Transmission};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use typosaurus::traits::semigroup::Mappend;
use uuid::Uuid;

use crate::fusion::action::FusionSeed;

pub use action::{
    InitializePropagation, NodeIdsFromList, ProcessNextNode, PropagationState, SendRootPropagation,
    Spawn, SpawnWarpAnimal, SpawnWarpBoundary,
};
pub use effect::{
    SendRootPropagationEffect, SendRootPropagationInput, SpawnAnimal,
    WaitForNodeTransmissionEffect, WaitForNodeTransmissionInput,
};

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
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
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
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
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
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
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

// ---------------------------------------------------------------------------
// SunState — runtime state for sun orchestration
// ---------------------------------------------------------------------------

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
    /// The epoch's loss has been sent to the node.
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
    /// Number of gradient accumulation microsteps per optimization epoch.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    pub nodes: Vec<SunNodeAppearance>,
    pub edges: Vec<SunEdgeAppearance>,
}

fn default_gradient_accumulation_steps() -> usize {
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
}

/// Access to the neutral topology shared by every Sun program state.
pub trait SunTopologyState {
    fn topology(&self) -> &Arc<Mutex<SunTopology>>;
}

pub trait SunStateView: SunTopologyState {
    fn sun_appearance(&self) -> SunAppearance;
}

/// State for two-sided propagation branch A.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropA {
    pub topology: Arc<Mutex<SunTopology>>,
    /// Shared two-sided bookkeeping (Arc so both branches share it).
    pub shared: Arc<Mutex<TwoSidedZoInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// State for propagation branch B.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropB {
    pub topology: Arc<Mutex<SunTopology>>,
    /// Shared two-sided bookkeeping (Arc so both branches share it).
    pub shared: Arc<Mutex<TwoSidedZoInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// Runtime state that tracks the topology and transmission endpoints
/// for a sun of spawned animals.
#[derive(Optic, Clone, Debug)]
pub struct SunStateWithInner<S> {
    /// Program-neutral graph and observation data.
    pub topology: Arc<Mutex<SunTopology>>,
    /// State for propagation branch A — uses p1_tx / p1_rx maps.
    #[jungle(focus = a)]
    pub a: PropA,
    /// State for propagation branch B — uses p2_tx / p2_rx maps.
    #[jungle(focus = b)]
    pub b: PropB,
    /// User-provided state threaded through Sun actions and flows.
    pub inner: S,
    /// Generated second-pass root transmissions waiting for phase B.
    pub propagation_down_inputs: VecDeque<Transmission>,
    /// Generated first-pass root transmissions, one per accumulation step.
    pub propagation_up_inputs: Vec<Transmission>,
    /// First-pass sink outputs captured during phase A accumulation.
    pub propagation_up_outputs: Vec<Transmission>,
    /// Propagation outputs collected across accumulation microsteps.
    pub propagation_pairs: Vec<(Transmission, Transmission)>,
    /// Step-indexed first-pass input mailboxes per input port.
    pub p1_step_tx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed second-pass input mailboxes per input port.
    pub p2_step_tx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed first-pass output mailboxes per node.
    pub p1_step_rx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed second-pass output mailboxes per node.
    pub p2_step_rx: Vec<HashMap<u32, ObjectId>>,
    /// Completed first-pass microsteps per node.
    pub node_p1_completed: HashMap<u32, usize>,
    /// Completed second-pass microsteps per node.
    pub node_p2_completed: HashMap<u32, usize>,
    /// First-pass root microsteps already seeded per root node.
    pub root_p1_sent: HashMap<u32, usize>,
    /// Second-pass root microsteps already seeded per root node.
    pub root_p2_sent: HashMap<u32, usize>,
    /// Number of node phase-microsteps completed in the current epoch.
    pub pipeline_completions: usize,
    /// Total node phase-microsteps expected in the current epoch.
    pub pipeline_target_completions: usize,
    /// Cached sink node id for the currently finalized graph.
    pub sink_id: Option<u32>,
}

/// State available to Sun actions and flows.
///
/// The generic `S` payload is available to user flows via
/// [`SunState::inner`] and defaults to `()`.
pub type TwoSidedZoState<S = ()> = SunStateWithInner<S>;
/// Compatibility name for the state owned by the [`TwoSidedZo`] program.
pub type SunState<S = ()> = TwoSidedZoState<S>;

impl<S> Default for SunStateWithInner<S>
where
    S: Default,
{
    fn default() -> Self {
        let topology = Arc::new(Mutex::new(SunTopology::default()));
        let shared = Arc::new(Mutex::new(TwoSidedZoInner::default()));
        Self {
            a: PropA {
                topology: Arc::clone(&topology),
                shared: Arc::clone(&shared),
                ..PropA::default()
            },
            b: PropB {
                topology: Arc::clone(&topology),
                shared,
                ..PropB::default()
            },
            topology,
            inner: S::default(),
            propagation_down_inputs: VecDeque::new(),
            propagation_up_inputs: Vec::new(),
            propagation_up_outputs: Vec::new(),
            propagation_pairs: Vec::new(),
            p1_step_tx: Vec::new(),
            p2_step_tx: Vec::new(),
            p1_step_rx: Vec::new(),
            p2_step_rx: Vec::new(),
            node_p1_completed: HashMap::new(),
            node_p2_completed: HashMap::new(),
            root_p1_sent: HashMap::new(),
            root_p2_sent: HashMap::new(),
            pipeline_completions: 0,
            pipeline_target_completions: 0,
            sink_id: None,
        }
    }
}

impl<S> SunStateWithInner<S> {
    /// Build a deterministic, serializable view of the runtime graph.
    pub fn appearance(&self) -> SunAppearance {
        let topology = self.topology.lock().unwrap();
        let strategy = self.a.shared.lock().unwrap();
        let grad_steps = strategy.grad_steps.max(1);
        let mut nodes = topology
            .journey_ids
            .keys()
            .copied()
            .map(|id| SunNodeAppearance {
                id,
                journey_id: topology
                    .journey_ids
                    .get(&id)
                    .copied()
                    .unwrap_or_else(Uuid::nil),
                warp_journey_id: topology
                    .warp_journey_ids
                    .get(&id)
                    .copied()
                    .unwrap_or_else(Uuid::nil),
                label: topology
                    .node_labels
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("cell {id}")),
                input_ports: topology.vertex_ports.get(&id).cloned().unwrap_or_default(),
                operational_state: topology
                    .node_operational_states
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                phase_annotation: topology.node_phase_annotations.get(&id).cloned(),
                state: strategy.node_states.get(&id).copied().unwrap_or_default(),
                state_sequence: topology
                    .node_state_sequences
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                grad_step: strategy
                    .node_grad_steps
                    .get(&id)
                    .copied()
                    .unwrap_or_else(|| strategy.current_grad_step()),
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
            grad_steps,
            nodes,
            edges,
        }
    }
}

impl<S> SunTopologyState for SunStateWithInner<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

impl<S> SunStateView for SunStateWithInner<S> {
    fn sun_appearance(&self) -> SunAppearance {
        self.appearance()
    }
}

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

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
#[derive(Optic, Clone, Default, Debug)]
pub struct TwoSidedZoInner {
    /// Latest observable orchestration phase reached by each internal vertex.
    pub node_states: HashMap<u32, SunNodeState>,
    /// 1-based gradient accumulation step currently associated with each vertex.
    pub node_grad_steps: HashMap<u32, usize>,
    /// Nodes whose first-pass output has been received in the current epoch.
    pub p1_completed: HashSet<u32>,
    /// Nodes whose second pass has been sent in the current epoch.
    pub p2_sent: HashSet<u32>,
    /// Number of propagation microsteps per optimization epoch.
    pub grad_steps: usize,
    /// Current microstep index (0-based) inside the accumulation epoch.
    pub active_micro_step: usize,
    /// First-pass input endpoints keyed by port id.
    pub p1_tx: HashMap<u32, ObjectId>,
    /// Next first-pass endpoints used between microsteps when accumulation > 1.
    pub next_p1_tx: HashMap<u32, ObjectId>,
    /// First-pass output endpoints keyed by vertex id.
    pub p1_rx: HashMap<u32, ObjectId>,
    /// Second-pass input endpoints keyed by port id.
    pub p2_tx: HashMap<u32, ObjectId>,
    /// Next second-pass endpoints used between microsteps when accumulation > 1.
    pub next_p2_tx: HashMap<u32, ObjectId>,
    /// Second-pass output endpoints keyed by vertex id.
    pub p2_rx: HashMap<u32, ObjectId>,
    /// Potentiation input endpoints keyed by port id.
    pub po_tx: HashMap<u32, ObjectId>,
}

/// Erased form of a compile-time checked edge. Separate binaries and rolling
/// deployments must still agree on these descriptors at runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredEdge {
    pub port_id: u32,
    pub source_contract: ContractDescriptor,
    pub destination_contract: ContractDescriptor,
}

impl TwoSidedZoInner {
    fn current_grad_step(&self) -> usize {
        let grad_steps = self.grad_steps.max(1);
        self.active_micro_step
            .saturating_add(1)
            .clamp(1, grad_steps)
    }

    fn grad_step_for_phase(&self, phase: SunNodeState) -> usize {
        match phase {
            SunNodeState::Optimization => self.grad_steps.max(1),
            _ => self.current_grad_step(),
        }
    }

    fn record_state_sent(&mut self, topology: &mut SunTopology, node_id: u32, phase: SunNodeState) {
        let grad_step = self.grad_step_for_phase(phase);
        let current = self.node_states.get(&node_id).copied().unwrap_or_default();
        topology
            .node_operational_states
            .insert(node_id, SunOperationalState::Running);
        topology.node_phase_annotations.insert(
            node_id,
            match phase {
                SunNodeState::Idle => "idle",
                SunNodeState::Propagation1 => "propagation 1",
                SunNodeState::Propagation2 => "propagation 2",
                SunNodeState::Optimization => "potentiation",
            }
            .to_string(),
        );
        if current == phase {
            self.node_grad_steps.insert(node_id, grad_step);
            return;
        }

        let mut next = current;
        let mut distance = 0;
        while next != phase {
            next = match next {
                SunNodeState::Idle | SunNodeState::Optimization => SunNodeState::Propagation1,
                SunNodeState::Propagation1 => SunNodeState::Propagation2,
                SunNodeState::Propagation2 => SunNodeState::Optimization,
            };
            distance += 1;
        }

        self.node_states.insert(node_id, phase);
        *topology.node_state_sequences.entry(node_id).or_default() += distance;
        self.node_grad_steps.insert(node_id, grad_step);
    }

    pub(crate) fn record_propagation_sent(
        &mut self,
        topology: &mut SunTopology,
        node_ids: impl IntoIterator<Item = u32>,
        phase: SunNodeState,
    ) {
        debug_assert!(matches!(
            phase,
            SunNodeState::Propagation1 | SunNodeState::Propagation2
        ));
        for node_id in node_ids {
            match phase {
                SunNodeState::Propagation1 => {
                    if self.node_states.get(&node_id) != Some(&SunNodeState::Propagation2) {
                        self.record_state_sent(topology, node_id, phase);
                    }
                }
                SunNodeState::Propagation2 => {
                    self.p2_sent.insert(node_id);
                    if self.p1_completed.contains(&node_id) {
                        self.record_state_sent(topology, node_id, phase);
                    }
                }
                _ => unreachable!("propagation phase was validated above"),
            }
        }
    }

    pub(crate) fn record_propagation_completed(
        &mut self,
        topology: &mut SunTopology,
        node_id: u32,
        phase: SunNodeState,
    ) {
        topology
            .node_operational_states
            .insert(node_id, SunOperationalState::Succeeded);
        if phase == SunNodeState::Propagation1 {
            self.p1_completed.insert(node_id);
            if self.p2_sent.contains(&node_id) {
                self.record_state_sent(topology, node_id, SunNodeState::Propagation2);
            }
        }
    }

    pub(crate) fn record_optimization_sent(
        &mut self,
        topology: &mut SunTopology,
        node_ids: impl IntoIterator<Item = u32>,
    ) {
        for node_id in node_ids {
            self.record_state_sent(topology, node_id, SunNodeState::Optimization);
            self.p1_completed.remove(&node_id);
            self.p2_sent.remove(&node_id);
        }
    }
}

/// Compatibility name for the two-sided strategy state.
pub type SunInner = TwoSidedZoInner;

/// A resolved edge target. `port_id` identifies the destination mailbox while
/// `vertex_id` identifies the single animal/output shared by all of its ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortTarget {
    pub port_id: u32,
    pub vertex_id: u32,
}

/// Generate a program-selected unary seed, then spawn and register its animal.
#[derive(Flow)]
pub struct UnarySunStepWithProgram<
    Program: SunProgram,
    P: Unsigned,
    AnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed> + OperationNode<Op>,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<GenUuid<Program>>,
    Step<action::SpawnUnary<P, AnimalT, E, Program, Op>>,
);

/// Compatibility program used by the descriptor-step aliases.
pub struct DeploymentProgram<S, const ACCUM_STEPS: usize>(PhantomData<fn() -> S>);

impl<S, const ACCUM_STEPS: usize> SunProgram for DeploymentProgram<S, ACCUM_STEPS> {
    type State = SunState<S>;
    type Driver = ();
    type UnarySeed = crate::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::cell::action::Init {
            recv_id: inbox,
            grad_steps: ACCUM_STEPS.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: ACCUM_STEPS.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: ACCUM_STEPS.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

pub type UnarySunStepWithState<P, AnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    UnarySunStepWithProgram<DeploymentProgram<S, ACCUM_STEPS>, P, AnimalT, E, Op>;

pub type UnarySunStep<
    P,
    AnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = UnarySunStepWithState<P, AnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// Generate a two-port seed, then spawn and register one binary animal.
#[derive(Flow)]
pub struct BinarySunStepWithProgram<
    Program: SunProgram,
    P1: Unsigned,
    P2: Unsigned,
    AnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed> + OperationNode<Op>,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<action::GenFusionSeed<Program>>,
    Step<action::SpawnBinary<P1, P2, AnimalT, E, Program, Op>>,
);

pub type BinarySunStepWithState<P1, P2, AnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    BinarySunStepWithProgram<DeploymentProgram<S, ACCUM_STEPS>, P1, P2, AnimalT, E, Op>;

pub type BinarySunStep<
    P1,
    P2,
    AnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = BinarySunStepWithState<P1, P2, AnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// Generate boundary mailboxes, spawn the nested warp animal, then spawn and
/// register the boundary animal in the parent topology.
#[derive(Flow)]
pub struct WarpSunStepWithProgram<
    Program: SunProgram,
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed> + OperationNode<Op>,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<GenUuid<Program>>,
    Step<action::SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>>,
    Step<action::SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>>,
);

pub type WarpSunStepWithState<P, WarpAnimalT, BoundaryAnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    WarpSunStepWithProgram<
        DeploymentProgram<S, ACCUM_STEPS>,
        P,
        WarpAnimalT,
        BoundaryAnimalT,
        E,
        Op,
    >;

pub type WarpSunStep<
    P,
    WarpAnimalT,
    BoundaryAnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = WarpSunStepWithState<P, WarpAnimalT, BoundaryAnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// One descriptor-specific spawn flow followed by the remaining descriptors.
#[derive(Flow)]
pub struct SunNode<S, U>(S, U);

/// Compiles a type-level topology into the executable driver selected by `P`.
///
/// `<Topology as BlackHole>::Sun<Program>` is the canonical application
/// point. The recursive fold still emits the topology-specific deployment
/// steps; the terminal case now attaches `Program::Driver` instead of a fixed
/// QuZO epoch.
pub trait BlackHole {
    type Sun<P: SunProgram>
    where
        Self: CompileSun<P>;
}

impl<T> BlackHole for T {
    type Sun<P: SunProgram>
        = <T as CompileSun<P>>::Flow
    where
        T: CompileSun<P>;
}

/// Program-specific compilation proof for a topology. This is where node
/// seed requirements are selected; the recursive [`BlackHole`] facade no
/// longer fixes them globally.
pub trait CompileSun<Program: SunProgram> {
    type Flow;
}

impl<Program: SunProgram, U> CompileSun<Program> for List<(Empty, U)>
where
    U: CompileSun<Program>,
{
    type Flow = U::Flow;
}

impl<Program: SunProgram, T1, T2, U> CompileSun<Program> for List<(List<(T1, T2)>, U)>
where
    (List<(T1, T2)>, U): Mappend,
    <(List<(T1, T2)>, U) as Mappend>::Out: CompileSun<Program>,
{
    type Flow = <<(List<(T1, T2)>, U) as Mappend>::Out as CompileSun<Program>>::Flow;
}

impl<Program, P, A, E, Op, U> CompileSun<Program> for List<(Unary<P, A, E, Op>, U)>
where
    Program: SunProgram,
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow = SunNode<UnarySunStepWithProgram<Program, P, A, E, Op>, U::Flow>;
}

impl<Program, P1, P2, A, E, Op, U> CompileSun<Program> for List<(Binary<P1, P2, A, E, Op>, U)>
where
    Program: SunProgram,
    P1: Unsigned,
    P2: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow = SunNode<BinarySunStepWithProgram<Program, P1, P2, A, E, Op>, U::Flow>;
}

impl<Program, P, WarpAnimalT, BoundaryAnimalT, E, Op, U> CompileSun<Program>
    for List<(Warp<P, WarpAnimalT, BoundaryAnimalT, E, Op>, U)>
where
    Program: SunProgram,
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + action::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow =
        SunNode<WarpSunStepWithProgram<Program, P, WarpAnimalT, BoundaryAnimalT, E, Op>, U::Flow>;
}

impl<Program: SunProgram> CompileSun<Program> for Empty {
    type Flow = Program::Driver;
}

/// Selects the state, deployment settings, and executable driver for a Sun.
pub trait SunProgram {
    type State: SunTopologyState;
    type Driver;
    type UnarySeed: Clone + Send + Sync + 'static;
    type BinarySeed: Clone + Send + Sync + 'static;
    type WarpSeed: Clone + Send + Sync + 'static;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed;
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId;
    fn binary_seed(inboxes: [ObjectId; 2]) -> Self::BinarySeed;
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2];
    fn warp_seed(inbox: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed;
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]);
}

/// Legacy generator/policy/state bundle accepted by [`TwoSidedZoManifest`].
pub trait Manifest {
    type Generator;
    type Policy;
    type State;
}

/// The existing two-sided zeroth-order training schedule as a Sun program.
pub struct TwoSidedZoWithState<Generator, Policy, S, const ACCUM_STEPS: usize = 1>(
    PhantomData<Generator>,
    PhantomData<Policy>,
    PhantomData<fn() -> S>,
);

impl<G, P, S, const A: usize> SunProgram for TwoSidedZoWithState<G, P, S, A> {
    type State = SunState<S>;
    type Driver = Sun<G, P, S, A>;
    type UnarySeed = crate::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::cell::action::Init {
            recv_id: inbox,
            grad_steps: A.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: A.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: A.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

pub type TwoSidedZo<Generator, Policy, const ACCUM_STEPS: usize = 1> =
    TwoSidedZoWithState<Generator, Policy, (), ACCUM_STEPS>;

/// Adapts the former manifest shape to the new program-based entrypoint.
pub struct TwoSidedZoManifest<M: Manifest, const ACCUM_STEPS: usize = 1>(PhantomData<M>);

impl<M: Manifest, const A: usize> SunProgram for TwoSidedZoManifest<M, A> {
    type State = SunState<M::State>;
    type Driver = Sun<M::Generator, M::Policy, M::State, A>;
    type UnarySeed = crate::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::cell::action::Init {
            recv_id: inbox,
            grad_steps: A.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: A.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: A.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

/// Legacy stateless generator/policy bundle.
pub struct StatelessManifest<Generator, Policy, const ACCUM_STEPS: usize = 1>(
    PhantomData<Generator>,
    PhantomData<Policy>,
);

impl<G, P, const A: usize> Manifest for StatelessManifest<G, P, A> {
    type Generator = G;
    type Policy = P;
    type State = ();
}

/// Compatibility alias for code that still defines a legacy [`Manifest`].
pub type LegacySun<T, M, const ACCUM_STEPS: usize> =
    <T as BlackHole>::Sun<TwoSidedZoManifest<M, ACCUM_STEPS>>;

// ---------------------------------------------------------------------------
// Predicates — loop continuation conditions
// ---------------------------------------------------------------------------

/// Predicate that checks whether a propagation branch has unfinished nodes.
pub struct PendingNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &Transmission)> for PendingNotEmpty<S>
where
    S: PropagationState,
{
    fn eval((state, _): &(&S, &Transmission)) -> bool {
        !state.pending().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Flow definitions — ready-node processing and orchestration
// ---------------------------------------------------------------------------
//
/// Processes the next ready node in propagation branch B.
#[derive(Flow)]
pub struct PropBLoop(Step<action::ProcessNextNode<PropB>>);

/// Processes the next ready node in propagation branch A.
#[derive(Flow)]
pub struct PropALoop(Step<action::ProcessNextNode<PropA>>);

#[derive(Flow)]
#[jungle(focus = PropB)]
pub struct PropBFlow(
    Step<action::InitializePropagation<PropB>>,
    Step<action::SendRootPropagation<PropB>>,
    While<FocusedLoopCondition<PendingNotEmpty<PropB>, PropB>, PropBLoop>,
);

#[derive(Flow)]
#[jungle(focus = PropA)]
pub struct PropAFlow(
    Step<action::InitializePropagation<PropA>>,
    Step<action::SendRootPropagation<PropA>>,
    While<FocusedLoopCondition<PendingNotEmpty<PropA>, PropA>, PropALoop>,
);

/// The two propagation branches (A and B) running in parallel via focused join.
pub type PropagationFlows = Join<PropAFlow, PropBFlow>;

/// Alias for one iteration of the propagation loop (branch A).
pub type PropagationLoop = PropALoop;

/// Predicate that keeps collecting generator outputs until `N` pairs exist.
pub struct PendingGeneratedPropagationInputs<const N: usize, S>(PhantomData<fn() -> S>);

impl<const N: usize, S> Predicate<(&SunState<S>, &())> for PendingGeneratedPropagationInputs<N, S> {
    fn eval((state, _): &(&SunState<S>, &())) -> bool {
        state.propagation_up_inputs.len() < N
    }
}

/// Predicate that keeps advancing the per-node propagation scheduler.
pub struct PendingPipelineWork<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&SunState<S>, &())> for PendingPipelineWork<S> {
    fn eval((state, _): &(&SunState<S>, &())) -> bool {
        state.pipeline_completions < state.pipeline_target_completions
    }
}

/// Predicate for a neutral dependency-aware forward pass.
pub struct PendingForwardWork<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&ForwardSunState<S>, &black_hole_spec::ArtifactDelivery<()>)>
    for PendingForwardWork<S>
{
    fn eval((state, _): &(&ForwardSunState<S>, &black_hole_spec::ArtifactDelivery<()>)) -> bool {
        !state.runtime.pending.is_empty()
    }
}

/// One generator emission pair capture step.
#[derive(Flow)]
pub struct CollectPropagationInputsStep<Generator, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Generator,
    Step<action::StorePropagationInputPair<S, GRADIENT_ACCUMULATION_STEPS>>,
);

/// One scheduler tick: seed ready roots, then process one ready node output.
#[derive(Flow)]
pub struct PipelineProgressStep<S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::SendReadyRootTasks<S, GRADIENT_ACCUMULATION_STEPS>>,
    Step<action::ProcessReadyPipelineNode<S, GRADIENT_ACCUMULATION_STEPS>>,
);

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
    Step<action::FinalizeForwardGraph<S>>,
    While<Always<ForwardSunState<S>, ()>, ServeRequest<Source, Input, Output, S>>,
);

/// Forward-only Sun program for a homogeneous operation topology.
///
/// Nodes used with this program can run [`crate::ForwardOperationCell`] and
/// therefore require only `MassOps<Op>`; no perturb or optimize capability is
/// part of the driver.
pub struct ForwardOnly<Source, InputOp: TensorContract, OutputOp: TensorContract = InputOp, S = ()>(
    PhantomData<Source>,
    PhantomData<InputOp>,
    PhantomData<OutputOp>,
    PhantomData<fn() -> S>,
);

impl<Source, InputOp, OutputOp, S> SunProgram for ForwardOnly<Source, InputOp, OutputOp, S>
where
    InputOp: TensorContract,
    OutputOp: TensorContract,
    InputOp::Input: Send + 'static,
    OutputOp::Output: Send + 'static,
{
    type State = ForwardSunState<S>;
    type Driver = ServeFlow<Source, InputOp::Input, OutputOp::Output, S>;
    type UnarySeed = crate::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::cell::action::Init {
            recv_id: inbox,
            grad_steps: 1,
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: 1,
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: 1,
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state.runtime.inputs.extend(ports.iter().copied());
    }
}

/// A small non-forward, non-QuZO schedule proving that a program can compile
/// a topology and then run arbitrary checkpoint and evaluation flows without
/// inheriting propagation state.
#[derive(Flow)]
pub struct CheckpointEvaluateFlow<Checkpoint, Evaluation, S>(
    Step<action::FinalizeNeutralGraph<S>>,
    Checkpoint,
    Evaluation,
);

pub struct CheckpointEvaluate<Checkpoint, Evaluation, S = ()>(
    PhantomData<Checkpoint>,
    PhantomData<Evaluation>,
    PhantomData<fn() -> S>,
);

impl<Checkpoint, Evaluation, S> SunProgram for CheckpointEvaluate<Checkpoint, Evaluation, S> {
    type State = NeutralSunState<S>;
    type Driver = CheckpointEvaluateFlow<Checkpoint, Evaluation, S>;
    type UnarySeed = crate::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::cell::action::Init {
            recv_id: inbox,
            grad_steps: 1,
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: 1,
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: 1,
            warp_journey_id,
        }
    }
    fn register_inboxes(_state: &mut Self::State, _ports: &[(u32, ObjectId)]) {}
}

// ---------------------------------------------------------------------------
// BlackHole — the top-level orchestration flow
// ---------------------------------------------------------------------------

/// One complete training epoch: generate → propagate → apply policy → broadcast potentiation.
#[derive(Flow)]
pub struct EpochWithState<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::BeginGradientAccumulation<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<
        PendingGeneratedPropagationInputs<GRADIENT_ACCUMULATION_STEPS, S>,
        CollectPropagationInputsStep<Generator, S, GRADIENT_ACCUMULATION_STEPS>,
    >,
    Step<action::PreparePropagationPipeline<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<PendingPipelineWork<S>, PipelineProgressStep<S, GRADIENT_ACCUMULATION_STEPS>>,
    Step<action::CollectedPropagationPairs<S, GRADIENT_ACCUMULATION_STEPS>>,
    Policy,
    Step<action::BroadcastPotentiation<S>>,
);

pub type Epoch<Generator, Policy, S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1> =
    EpochWithState<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>;

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
#[derive(Flow)]
pub struct SunFlow<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::BuildAddrs<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<
        Always<SunState<S>, ()>,
        EpochWithState<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>,
    >,
);

pub type Sun<Generator, Policy, S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1> =
    SunFlow<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>;
