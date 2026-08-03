//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
use jungle_zoo::Noop;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use uuid::Uuid;

pub use action::{EdgeIdsFromList, Spawn};
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
pub struct Tag<N: Unsigned, A: Animal, E: EdgeIdsFromList>(
    PhantomData<N>,
    PhantomData<A>,
    PhantomData<E>,
);
pub trait Tagged {
    type N: Unsigned;
    type A: Animal;
    type E: EdgeIdsFromList;
}

// ---------------------------------------------------------------------------
// SunState — runtime state for sun orchestration
// ---------------------------------------------------------------------------

/// Runtime state that tracks the topology and transmission endpoints
/// for a sun of spawned animals.
pub struct SunState {
    pub a: SunInner,
    pub b: SunInner,
    pub c: SunInner,
    /// Topological layers of node journey IDs (outer-to-inner).
    pub topo: Vec<HashSet<Uuid>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<Uuid>,
}

pub struct SunInner {
    /// Maps an edge UUID to the list of node journey IDs that receive on that edge.
    pub incoming: HashMap<Uuid, Vec<Uuid>>,
    /// Maps a node journey ID to the list of outgoing edge UUIDs.
    pub outgoing: HashMap<Uuid, Vec<Uuid>>,
    /// Transmission send endpoints keyed by edge UUID.
    pub tx: HashMap<Uuid, ObjectId>,
    /// Transmission receive endpoints keyed by edge UUID.
    pub rx: HashMap<Uuid, ObjectId>,
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

//// 1. Spawn all children getting uuids
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
