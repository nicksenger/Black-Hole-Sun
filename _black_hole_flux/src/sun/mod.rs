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

pub use action::{
    BuildTopologicalSort, NodeIdsFromList, PopLayer, ProcessNode, Spawn, TopologyState,
};
pub use effect::SpawnAnimal;

//impl<T> From<T> for () {
//    fn from(_value: T) -> Self {}
//}
//impl<T> From<T> for (()) {
//    fn from(value: T) -> Self {
//        ()
//    }
//}

// ---------------------------------------------------------------------------
// Tag — type-level descriptor for a single node in the sun graph
// ---------------------------------------------------------------------------

/// Type-level tag that describes a single node in the sun graph.
pub struct Tag<N: Unsigned, A: Animal, E: NodeIdsFromList>(
    PhantomData<N>,
    PhantomData<A>,
    PhantomData<E>,
);
pub trait Tagged {
    type N: Unsigned;
    type A: Animal;
    type E: NodeIdsFromList;
}

impl<N: Unsigned, A: Animal, E: NodeIdsFromList> Tagged for Tag<N, A, E> {
    type N = N;
    type A = A;
    type E = E;
}

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
#[derive(Optic, Clone, Default, Debug)]
pub struct SunState {
    /// State for propagation branch A — uses p1_tx / p1_rx maps.
    #[jungle(focus = a)]
    pub a: PropA,
    /// State for propagation branch B — uses p2_tx / p2_rx maps.
    #[jungle(focus = b)]
    pub b: PropB,
}

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
#[derive(Optic, Clone, Default, Debug)]
pub struct SunInner {
    /// Maps the node u32 id to its associated journey ID.
    pub journey_ids: HashMap<u32, Uuid>,
    /// Maps each node to the nodes of its incoming edges.
    pub incoming: HashMap<u32, Vec<u32>>,
    /// Maps each node to the nodes of its outgoing edges.
    pub outgoing: HashMap<u32, Vec<u32>>,
    /// Propagation A send endpoints keyed by node id.
    pub p1_tx: HashMap<u32, ObjectId>,
    /// Propagation A receive endpoints keyed by node id.
    pub p1_rx: HashMap<u32, ObjectId>,
    /// Propagation B send endpoints keyed by node id.
    pub p2_tx: HashMap<u32, ObjectId>,
    /// Propagation B receive endpoints keyed by node id.
    pub p2_rx: HashMap<u32, ObjectId>,
    /// Potentiation send endpoints keyed by node id.
    pub po_tx: HashMap<u32, ObjectId>,
}

/// Single-node spawn step: generate UUID then spawn the animal for one tag.
#[derive(Flow)]
pub struct SunStep<
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
>(Step<GenUuid>, Step<Spawn<T>>);

/// Wrapper that gives a Join-tree of flows a proper FlowScope implementation.
/// Used as the top-level flow type returned by EventHorizon.
#[derive(Flow)]
pub struct SunFlow<F>(F);

pub trait EventHorizon {
    type Flow;
}
impl<T, U> EventHorizon for List<(T, U)>
where
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
    U: EventHorizon,
{
    // Build a Join-tree: SunStep<T> composed with the rest of the list.
    // Wrapped in SunFlow so the top-level type implements FlowScope.
    type Flow = SunFlow<Join<SunStep<T>, <U as EventHorizon>::Flow>>;
}
impl EventHorizon for Empty {
    // Base case: wrap BlackHole in SunFlow for consistent wrapping.
    type Flow = SunFlow<BlackHole>;
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
    While<CurrentNotEmpty<PropB>, InnerBLoop>,
);

/// Body of the outer loop: pop a layer, then process all its nodes.
#[derive(Flow)]
pub struct BranchABody(
    Step<action::PopLayer<PropA>>,
    While<CurrentNotEmpty<PropA>, InnerALoop>,
);

#[derive(Flow)]
#[jungle(focus = PropB)]
pub struct PropBFlow(
    Step<action::BuildTopologicalSort<PropB>>,
    While<TopoNotEmpty<PropB>, BranchBBody>,
);

#[derive(Flow)]
#[jungle(focus = PropA)]
pub struct PropAFlow(
    Step<action::BuildTopologicalSort<PropA>>,
    While<TopoNotEmpty<PropA>, BranchABody>,
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

/// One complete training epoch: kick-off → propagation → compute-loss → broadcast-potentiation.
#[derive(Flow)]
pub struct Epoch(
    Step<action::Initialize>,
    PropagationFlows,
    Step<action::ComputeLoss>,
    Step<action::BroadcastPotentiation>,
);

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
#[derive(Flow)]
pub struct BlackHole(Step<action::BuildAddrs>, While<Always<SunState, ()>, Epoch>);
