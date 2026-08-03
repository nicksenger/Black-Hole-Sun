//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
use jungle_zoo::Noop;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use uuid::Uuid;

pub use action::{NodeIdsFromList, Spawn};
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
    /// Maps the node u32 id the its associated journey ID
    pub journey_ids: HashMap<u32, Uuid>,
    /// Maps each node to the nodes of its incoming edges
    pub incoming: HashMap<u32, Vec<u32>>,
    /// Maps each node to the nodes of its outgoing edges
    pub outgoing: HashMap<u32, Vec<u32>>,
    /// Transmission send endpoints keyed by edge id.
    pub tx: HashMap<u32, ObjectId>,
    /// Transmission receive endpoints keyed by edge id.
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

//// LOOP FOREVER
///     FOCUSED-JOIN OVER 3 STATES (propagate1, propagate2, potentiate), for each branch:
//// // 2. Build topological ordering
//// // // WHILE TOPO NOT EMPTY
//// // // 3. Pop topo vec into current
//// // // // WHILE CURRENT NOT EMPTY
//// // // // 4. wait for whichever rx from the set becomes available on void first
//// // // // 5. remove from current and update rx for node
//// // // // 6. construct transmission and send to outgoing-tx with rx ObjectIds as send
//// // // // 7. update tx for outgoing nodes
//#[derive(Flow)]
//pub struct BlackHole(Step<TopologicalSort>);
#[derive(Flow)]
pub struct BlackHole();
