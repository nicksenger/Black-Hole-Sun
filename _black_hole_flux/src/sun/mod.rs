//! Sun module — spawning and orchestrating animal journeys.

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use black_hole_spec::ObjectId;
use jungle_sdk::prelude::*;
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
pub struct Tag<N, T, E>(PhantomData<N>, PhantomData<T>, PhantomData<E>);

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

// ---------------------------------------------------------------------------
// Sun — higher-order flow for sun orchestration
// ---------------------------------------------------------------------------

/// A Sun orchestrates a graph of spawned animals.
///
/// `Cells` is a type-level list of [`Tag`] descriptors, one per node.
pub struct Sun<Tags>(PhantomData<Tags>);

// ---------------------------------------------------------------------------
// Ray — single-node spawn flow
// ---------------------------------------------------------------------------

/// A Ray spawns a single animal tagged by a [`Tag<N, Animal, Edges>`]
/// and chains it with a downstream flow `U`.
///
/// Instantiate with a concrete Tag type so the [`Spawn`] action bounds are met.
pub struct Ray<Tag, U>(Step<Spawn<Tag>>, U)
where
    Spawn<Tag>: Action;
