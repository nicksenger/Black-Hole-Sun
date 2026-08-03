//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use uuid::Uuid;

pub use action::{
    BuildTopologicalSort, NodeIdsFromList, PopLayer, ProcessNode, Spawn, TopologyState,
};
pub use effect::SpawnAnimal;

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

// ---------------------------------------------------------------------------
// SunState — runtime state for sun orchestration
// ---------------------------------------------------------------------------

/// State for propagation branch A.
pub struct PropA {
    /// Shared bookkeeping (Arc so both branches share topology data).
    pub shared: Arc<Mutex<SunInner>>,
    /// Topological layers of node IDs (outer-to-inner).
    pub topo: Vec<HashSet<u32>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<u32>,
}

/// State for propagation branch B.
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
#[derive(Optic)]
pub struct SunState {
    /// State for propagation branch A — uses p1_tx / p1_rx maps.
    #[jungle(focus = a)]
    pub a: PropA,
    /// State for propagation branch B — uses p2_tx / p2_rx maps.
    #[jungle(focus = b)]
    pub b: PropB,
}

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
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
    /// The current transmission id (set by KickOff, used by ComputeLoss).
    pub transmission_id: ObjectId,
}

#[derive(Flow)]
pub struct Sun<
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
    U,
>(Step<Spawn<T>>, U);

pub trait EventHorizon {
    type Flow;
}
impl<T, U> EventHorizon for List<(T, U)>
where
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
    U: EventHorizon,
{
    type Flow = Sun<T, <U as EventHorizon>::Flow>;
}
impl EventHorizon for Empty {
    type Flow = BlackHole;
}

// ---------------------------------------------------------------------------
// Predicates — loop continuation conditions
// ---------------------------------------------------------------------------

/// Predicate that checks if the topological layer queue is non-empty.
pub struct TopoNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &())> for TopoNotEmpty<S>
where
    S: TopologyState,
{
    fn eval((state, _): &(&S, &())) -> bool {
        !state.get_topo().is_empty()
    }
}

/// Predicate that checks if the current layer has unprocessed nodes.
pub struct CurrentNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &())> for CurrentNotEmpty<S>
where
    S: TopologyState,
{
    fn eval((state, _): &(&S, &())) -> bool {
        !state.get_current().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Flow definitions — layer processing and orchestration
// ---------------------------------------------------------------------------

/// Body of the inner loop: process nodes in the current layer until empty.
#[derive(Flow)]
pub struct InnerLoop<S: TopologyState>(Step<action::ProcessNode<S>>);

/// Body of the outer loop: pop a layer, then process all its nodes.
#[derive(Flow)]
pub struct BranchBody<S: TopologyState>(
    Step<action::PopLayer<S>>,
    While<CurrentNotEmpty<S>, InnerLoop<S>>,
);

#[derive(Flow)]
#[jungle(focus = FocusState)]
pub struct PropBFlow(
    Step<action::BuildTopologicalSort<S>>,
    Step<action::BuildAddrs<S>>,
    While<TopoNotEmpty<S>, BranchBody<S>>,
);

#[derive(Flow)]
#[jungle(focus = FocusState)]
pub struct PropAFlow(
    Step<action::BuildTopologicalSort<S>>,
    Step<action::BuildAddrs<S>>,
    While<TopoNotEmpty<S>, BranchBody<S>>,
);

/// The two propagation branches (A and B) running in parallel via focused join.
pub type PropagationFlows = Join<PropAFlow, PropBFlow>;

// ---------------------------------------------------------------------------
// BlackHole — the top-level orchestration flow
// ---------------------------------------------------------------------------

/// One complete training epoch: kick-off → propagation → compute-loss → broadcast-potentiation.
#[derive(Flow)]
pub struct Epoch(
    Step<action::KickOff>,
    PropagationFlows,
    Step<action::ComputeLoss>,
    Step<action::BroadcastPotentiation>,
);

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
///
/// Runs a continuous outer loop containing one complete training epoch per
/// iteration:
///
/// 1. **KickOff** — takes unit input, finds root nodes (no incoming edges),
///    generates a TransmissionId stored in shared state, and sends Propagation
///    transmissions to each root node's rx endpoint. This kicks off propagation.
/// 2. **PropagationFlows** — two focused branches (A and B) run in parallel,
///    each processing nodes topologically. Branch A uses p1_tx/p1_rx maps,
///    branch B uses p2_tx/p2_rx maps.
/// 3. **ComputeLoss** — retrieves the TransmissionId from shared state,
///    downloads the transmission, and computes (loss_up, loss_down).
/// 4. **BroadcastPotentiation** — broadcasts `Transmission::Potentiation` with
///    the loss values to all nodes' po_tx endpoints, including a new recv
///    ObjectId that replaces the used tx. Exits without waiting for responses.
#[derive(Flow)]
pub struct BlackHole(While<Always<SunState, ()>, Epoch>);
