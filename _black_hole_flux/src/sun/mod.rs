//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use action::GenUuid;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_spec::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use uuid::Uuid;

use crate::fusion::action::{FusionSeed, FusionState};
use crate::fusion::FusionFlow;

pub use action::{
    InitializePropagation, NodeIdsFromList, ProcessNextNode, PropagationState, SendRootPropagation,
    Spawn,
};
pub use effect::{
    SendRootPropagationEffect, SendRootPropagationInput, SpawnAnimal, WaitForNodeTransmission,
    WaitForNodeTransmissionInput,
};

// ---------------------------------------------------------------------------
// Descriptors — type-level vertices and their input ports
// ---------------------------------------------------------------------------

/// Type-level unary vertex with one input port and a list of output ports.
///
/// `P` is both the public input port and the deterministic internal vertex key.
pub struct Unary<P: Unsigned, A: Animal, E: NodeIdsFromList>(
    PhantomData<P>,
    PhantomData<A>,
    PhantomData<E>,
);

/// Type-level binary vertex whose two input ports share one spawned animal and
/// one output mailbox per propagation pass.
///
/// `P1` is the deterministic internal vertex key; both `P1` and `P2` resolve
/// to it during graph finalization.
pub struct Binary<P1: Unsigned, P2: Unsigned, A: Animal, E: NodeIdsFromList>(
    PhantomData<P1>,
    PhantomData<P2>,
    PhantomData<A>,
    PhantomData<E>,
);

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

/// One node in the observable Sun topology.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SunNodeAppearance {
    pub id: u32,
    pub label: String,
    pub input_ports: Vec<u32>,
    pub state: SunNodeState,
    /// Monotonic logical phase position, including phases crossed between snapshots.
    pub state_sequence: u64,
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
    pub nodes: Vec<SunNodeAppearance>,
    pub edges: Vec<SunEdgeAppearance>,
}

/// State for propagation branch A.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropA {
    /// Shared bookkeeping (Arc so both branches share topology data).
    pub shared: Arc<Mutex<SunInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// State for propagation branch B.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropB {
    /// Shared bookkeeping (Arc so both branches share topology data).
    pub shared: Arc<Mutex<SunInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// Runtime state that tracks the topology and transmission endpoints
/// for a sun of spawned animals.
#[derive(Optic, Clone, Debug)]
pub struct SunState {
    /// State for propagation branch A — uses p1_tx / p1_rx maps.
    #[jungle(focus = a)]
    pub a: PropA,
    /// State for propagation branch B — uses p2_tx / p2_rx maps.
    #[jungle(focus = b)]
    pub b: PropB,
}

impl Default for SunState {
    fn default() -> Self {
        let shared = Arc::new(Mutex::new(SunInner::default()));
        Self {
            a: PropA {
                shared: Arc::clone(&shared),
                ..PropA::default()
            },
            b: PropB {
                shared,
                ..PropB::default()
            },
        }
    }
}

impl SunState {
    /// Build a deterministic, serializable view of the runtime graph.
    pub fn appearance(&self) -> SunAppearance {
        let inner = self.a.shared.lock().unwrap();
        let mut nodes = inner
            .journey_ids
            .keys()
            .copied()
            .map(|id| SunNodeAppearance {
                id,
                label: inner
                    .node_labels
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("cell {id}")),
                input_ports: inner.vertex_ports.get(&id).cloned().unwrap_or_default(),
                state: inner.node_states.get(&id).copied().unwrap_or_default(),
                state_sequence: inner
                    .node_state_sequences
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.id);

        let mut edges = inner
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
            finalized: inner.finalized,
            nodes,
            edges,
        }
    }
}

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
#[derive(Optic, Clone, Default, Debug)]
pub struct SunInner {
    /// Whether the graph has passed runtime topology validation.
    pub finalized: bool,
    /// Maps an internal vertex key to its associated journey ID.
    pub journey_ids: HashMap<u32, Uuid>,
    /// Human-readable animal type names keyed by internal vertex.
    pub node_labels: HashMap<u32, String>,
    /// Latest observable orchestration phase reached by each internal vertex.
    pub node_states: HashMap<u32, SunNodeState>,
    /// Logical phase position for each vertex, used to recover skipped observations.
    pub node_state_sequences: HashMap<u32, u64>,
    /// Nodes whose first-pass output has been received in the current epoch.
    pub p1_completed: HashSet<u32>,
    /// Nodes whose second pass has been sent in the current epoch.
    pub p2_sent: HashSet<u32>,
    /// Input ports owned by each vertex, in descriptor order.
    pub vertex_ports: HashMap<u32, Vec<u32>>,
    /// Resolves every public input port to its internal vertex key.
    pub port_vertices: HashMap<u32, u32>,
    /// Ports declared as outputs by each vertex, before graph finalization.
    pub declared_outputs: HashMap<u32, Vec<u32>>,
    /// Ports claimed by more than one descriptor.
    pub duplicate_ports: HashSet<u32>,
    /// Maps each vertex to the vertices of its incoming edges.
    pub incoming: HashMap<u32, Vec<u32>>,
    /// Resolved outgoing destination ports for each vertex.
    pub outgoing: HashMap<u32, Vec<PortTarget>>,
    /// First-pass input endpoints keyed by port id.
    pub p1_tx: HashMap<u32, ObjectId>,
    /// First-pass output endpoints keyed by vertex id.
    pub p1_rx: HashMap<u32, ObjectId>,
    /// Second-pass input endpoints keyed by port id.
    pub p2_tx: HashMap<u32, ObjectId>,
    /// Second-pass output endpoints keyed by vertex id.
    pub p2_rx: HashMap<u32, ObjectId>,
    /// Potentiation input endpoints keyed by port id.
    pub po_tx: HashMap<u32, ObjectId>,
}

impl SunInner {
    fn record_state_sent(&mut self, node_id: u32, phase: SunNodeState) {
        let current = self.node_states.get(&node_id).copied().unwrap_or_default();
        if current == phase {
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
        *self.node_state_sequences.entry(node_id).or_default() += distance;
    }

    pub(crate) fn record_propagation_sent(
        &mut self,
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
                        self.record_state_sent(node_id, phase);
                    }
                }
                SunNodeState::Propagation2 => {
                    self.p2_sent.insert(node_id);
                    if self.p1_completed.contains(&node_id) {
                        self.record_state_sent(node_id, phase);
                    }
                }
                _ => unreachable!("propagation phase was validated above"),
            }
        }
    }

    pub(crate) fn record_propagation_completed(&mut self, node_id: u32, phase: SunNodeState) {
        if phase == SunNodeState::Propagation1 {
            self.p1_completed.insert(node_id);
            if self.p2_sent.contains(&node_id) {
                self.record_state_sent(node_id, SunNodeState::Propagation2);
            }
        }
    }

    pub(crate) fn record_optimization_sent(&mut self, node_ids: impl IntoIterator<Item = u32>) {
        for node_id in node_ids {
            self.record_state_sent(node_id, SunNodeState::Optimization);
            self.p1_completed.remove(&node_id);
            self.p2_sent.remove(&node_id);
        }
    }
}

/// A resolved edge target. `port_id` identifies the destination mailbox while
/// `vertex_id` identifies the single animal/output shared by all of its ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortTarget {
    pub port_id: u32,
    pub vertex_id: u32,
}

/// Generate a unary seed, then spawn and register its animal.
#[derive(Flow)]
pub struct UnarySunStep<
    P: Unsigned,
    AnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId>,
    E: NodeIdsFromList,
>(Step<GenUuid>, Step<action::SpawnUnary<P, AnimalT, E>>);

/// Generate a two-port seed, then spawn and register one binary animal.
#[derive(Flow)]
pub struct BinarySunStep<
    P1: Unsigned,
    P2: Unsigned,
    AnimalT: Animal<
        Id: AnimalIdValue,
        Generation: Unsigned,
        Seed = FusionSeed,
        State = FusionState,
        Flow: FusionFlow,
    >,
    E: NodeIdsFromList,
>(
    Step<action::GenFusionSeed>,
    Step<action::SpawnBinary<P1, P2, AnimalT, E>>,
);

/// One descriptor-specific spawn flow followed by the remaining descriptors.
#[derive(Flow)]
pub struct SunNode<S, U>(S, U);

/// Maps a type-level graph to its orchestration flow.
///
/// `Generator` is a Jungle flow from `()` to `(Transmission, Transmission)`;
/// `Policy` is a Jungle flow from `(Transmission, Transmission)` to
/// `(f32, f32)`. Keeping both as flow parameters lets callers compose arbitrary
/// generation and policy pipelines around the fixed graph propagation
/// machinery.
pub trait BlackHole {
    type Sun<Generator, Policy>;
}
impl<P, A, E, U> BlackHole for List<(Unary<P, A, E>, U)>
where
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId>,
    E: NodeIdsFromList,
    U: BlackHole,
{
    type Sun<Generator, Policy> =
        SunNode<UnarySunStep<P, A, E>, <U as BlackHole>::Sun<Generator, Policy>>;
}
impl<P1, P2, A, E, U> BlackHole for List<(Binary<P1, P2, A, E>, U)>
where
    P1: Unsigned,
    P2: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>,
    A::Flow: FusionFlow,
    E: NodeIdsFromList,
    U: BlackHole,
{
    type Sun<Generator, Policy> =
        SunNode<BinarySunStep<P1, P2, A, E>, <U as BlackHole>::Sun<Generator, Policy>>;
}
impl BlackHole for Empty {
    type Sun<Generator, Policy> = Sun<Generator, Policy>;
}

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

// ---------------------------------------------------------------------------
// BlackHole — the top-level orchestration flow
// ---------------------------------------------------------------------------

/// One complete training epoch: generate → propagate → apply policy → broadcast potentiation.
#[derive(Flow)]
pub struct Epoch<Generator, Policy>(
    Generator,
    PropagationFlows,
    Policy,
    Step<action::BroadcastPotentiation>,
);

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
#[derive(Flow)]
pub struct Sun<Generator, Policy>(
    Step<action::BuildAddrs>,
    While<Always<SunState, ()>, Epoch<Generator, Policy>>,
);
