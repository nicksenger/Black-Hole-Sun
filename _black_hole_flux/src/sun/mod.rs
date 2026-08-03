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
// Tag — type-level descriptor for a sun node
// ---------------------------------------------------------------------------

/// Type-level tag that describes a single node in the sun graph.
///
/// - `N`: the node's ID as a typenum integer
/// - `T`: the [`Animal`] type to be spawned at this node
/// - `E`: a type-level heterogeneous list of typenum integers representing
///   the IDs of this node's outgoing edges
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

pub struct A {
    /// Shared bookkeeping
    pub shared: Arc<Mutex<SunInner>>,
    /// Topological layers of node IDs (outer-to-inner).
    pub topo: Vec<HashSet<u32>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<u32>,
}
pub struct B {
    /// Shared bookkeeping
    pub shared: Arc<Mutex<SunInner>>,
    /// Topological layers of node IDs (outer-to-inner).
    pub topo: Vec<HashSet<u32>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<u32>,
}
pub struct C {
    /// Shared bookkeeping
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
    /// State for the propagation branches
    #[jungle(focus)]
    pub propagation: SunPropagation,
    /// State for potentiation branch
    #[jungle(focus)]
    pub c: C,
}

#[derive(Optic)]
pub struct SunPropagation {
    /// State for propagation A branch
    #[jungle(focus)]
    pub a: A,
    /// State for propagation B branch
    #[jungle(focus)]
    pub b: B,
}

pub struct SunInner {
    /// Maps the node u32 id to its associated journey ID
    pub journey_ids: HashMap<u32, Uuid>,
    /// Maps each node to the nodes of its incoming edges
    pub incoming: HashMap<u32, Vec<u32>>,
    /// Maps each node to the nodes of its outgoing edges
    pub outgoing: HashMap<u32, Vec<u32>>,
    /// Transmission send endpoints keyed by node id.
    pub tx: HashMap<u32, ObjectId>,
    /// Transmission receive endpoints keyed by node id.
    pub rx: HashMap<u32, ObjectId>,
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
// Flow definitions — branch layer processing
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

/// One complete pass through the topology for a single branch:
/// build topological sort, then process all layers.
#[derive(Flow)]
pub struct LayerFlow<S: TopologyState>(
    Step<action::BuildTopologicalSort<S>>,
    While<TopoNotEmpty<S>, BranchBody<S>>,
);

/// The two propagation branches (A and B) running in parallel via focused join.
pub type PropagationFlows = Join<LayerFlow<A>, LayerFlow<B>>;

/// The potentiation branch (C) running as a single layer flow.
pub type PotentiationFlow = LayerFlow<C>;

// ---------------------------------------------------------------------------
// BlackHole — the top-level orchestration flow
// ---------------------------------------------------------------------------

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
///
/// Runs a continuous outer loop containing a focused-join over 3 branches:
/// - **A** (Propagation): processes nodes in the A branch topologically
/// - **B** (Propagation): processes nodes in the B branch topologically
/// - **C** (Potentiation): processes nodes in the C branch topologically
///
/// Each branch independently:
/// 1. Builds topological layers via Kahn's algorithm
/// 2. Pops the outermost layer into current
/// 3. Waits for transmissions from rx ObjectIds of current-layer nodes
/// 4. On receiving a transmission, updates the node's rx id, generates new
///    tx Uuids for outgoing nodes, and forwards the transmission
#[derive(Flow)]
pub struct BlackHole(While<Always<SunState, ()>, Join<PropagationFlows, PotentiationFlow>>);
