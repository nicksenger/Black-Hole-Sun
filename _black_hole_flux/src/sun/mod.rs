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
    BuildTopologicalSort, NodeIdsFromList, PopLayer, ProcessNode, Spawn, TopologyState,
};
pub use effect::SpawnAnimal;

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

/// State for propagation branch A.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropA {
    /// Shared bookkeeping (Arc so both branches share topology data).
    pub shared: Arc<Mutex<SunInner>>,
    /// Topological layers of node IDs (outer-to-inner).
    pub topo: Vec<HashSet<u32>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<u32>,
}

/// State for propagation branch B.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropB {
    /// Shared bookkeeping (Arc so both branches share topology data).
    pub shared: Arc<Mutex<SunInner>>,
    /// Topological layers of node IDs (outer-to-inner).
    pub topo: Vec<HashSet<u32>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<u32>,
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

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
#[derive(Optic, Clone, Default, Debug)]
pub struct SunInner {
    /// Maps an internal vertex key to its associated journey ID.
    pub journey_ids: HashMap<u32, Uuid>,
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

/// Predicate that checks if the topological layer queue is non-empty.
pub struct TopoNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &Transmission)> for TopoNotEmpty<S>
where
    S: TopologyState,
{
    fn eval((state, _): &(&S, &Transmission)) -> bool {
        !state.get_topo().is_empty()
    }
}

/// Predicate that checks if the current layer has unprocessed nodes.
pub struct CurrentNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &Transmission)> for CurrentNotEmpty<S>
where
    S: TopologyState,
{
    fn eval((state, _): &(&S, &Transmission)) -> bool {
        !state.get_current().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Flow definitions — layer processing and orchestration
// ---------------------------------------------------------------------------
//
/// Body of the inner loop: process nodes in the current layer until empty.
#[derive(Flow)]
pub struct InnerBLoop(Step<action::ProcessNode<PropB>>);

/// Body of the inner loop: process nodes in the current layer until empty.
#[derive(Flow)]
pub struct InnerALoop(Step<action::ProcessNode<PropA>>);

/// Body of the outer loop: pop a layer, then process all its nodes.
#[derive(Flow)]
pub struct BranchBBody(
    Step<action::PopLayer<PropB>>,
    While<FocusedLoopCondition<CurrentNotEmpty<PropB>, PropB>, InnerBLoop>,
);

/// Body of the outer loop: pop a layer, then process all its nodes.
#[derive(Flow)]
pub struct BranchABody(
    Step<action::PopLayer<PropA>>,
    While<FocusedLoopCondition<CurrentNotEmpty<PropA>, PropA>, InnerALoop>,
);

#[derive(Flow)]
#[jungle(focus = PropB)]
pub struct PropBFlow(
    Step<action::BuildTopologicalSort<PropB>>,
    While<FocusedLoopCondition<TopoNotEmpty<PropB>, PropB>, BranchBBody>,
);

#[derive(Flow)]
#[jungle(focus = PropA)]
pub struct PropAFlow(
    Step<action::BuildTopologicalSort<PropA>>,
    While<FocusedLoopCondition<TopoNotEmpty<PropA>, PropA>, BranchABody>,
);

/// The two propagation branches (A and B) running in parallel via focused join.
pub type PropagationFlows = Join<PropAFlow, PropBFlow>;

/// Alias for the inner loop flow (inner A loop).
pub type InnerLoop = InnerALoop;

/// Alias for the branch body flow (branch A body).
pub type BranchBody = BranchABody;

/// Alias for the full propagation layer flow.
pub type LayerFlow = PropagationFlows;

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
