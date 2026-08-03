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
    /// Maps an edge UUID to the list of node journey IDs that receive on that edge.
    pub incoming: HashMap<Uuid, Vec<Uuid>>,
    /// Maps a node journey ID to the list of outgoing edge UUIDs.
    pub outgoing: HashMap<Uuid, Vec<Uuid>>,
    /// Transmission send endpoints keyed by edge UUID.
    pub tx: HashMap<Uuid, ObjectId>,
    /// Transmission receive endpoints keyed by edge UUID.
    pub rx: HashMap<Uuid, ObjectId>,
    /// Topological layers of node journey IDs (outer-to-inner).
    pub topo: Vec<HashSet<Uuid>>,
    /// Current layer being processed (popped from topo).
    pub current: HashSet<Uuid>,
}

#[derive(Flow)]
pub struct Sun<
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
    U,
>(Step<Spawn<T>>, U);

pub trait BlackHole {
    type Flow;
}
impl<T, U> BlackHole for List<(T, U)>
where
    T: Tagged<A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed: Send + Sync + 'static>>,
    U: BlackHole,
{
    type Flow = Sun<T, <U as BlackHole>::Flow>;
}
impl BlackHole for Empty {
    type Flow = Step<Noop<SunState, ()>>;
}
