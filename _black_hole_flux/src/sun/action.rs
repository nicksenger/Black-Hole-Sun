//! Sun actions — spawning animals and populating sun state.

use std::marker::PhantomData;

use super::Tagged;
use jungle_sdk::prelude::*;
use typosaurus::collections::list::{Empty, List};
use typosaurus::num::Unsigned;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// NodeIdsFromList — extract runtime node IDs from a type-level integer list
// ---------------------------------------------------------------------------

/// Trait that converts a type-level list of typenum integers into a runtime
/// vector of node IDs (u32 values).
pub trait NodeIdsFromList {
    fn node_ids() -> Vec<u32>;
}

impl NodeIdsFromList for Empty {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<H, T> NodeIdsFromList for List<(H, T)>
where
    H: Unsigned,
    T: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![<H as Unsigned>::U32];
        ids.extend(T::node_ids());
        ids
    }
}

// ---------------------------------------------------------------------------
// Spawn — spawn an animal and populate SunState with outgoing edges
// ---------------------------------------------------------------------------

/// Action that spawns an animal `T` tagged by [`Tag`](super::Tag) into the jungle.
///
/// Takes the animal's seed as input, spawns it via [`SpawnAnimal`](super::effect::SpawnAnimal)
/// effect, receives the journey UUID, then populates the [`SunState`](super::SunState)
/// outgoing map with directed edges from this node to each outgoing node ID
/// derived from the type-level list `E`.
///
/// Returns the journey UUID for downstream use.
pub struct Spawn<Tag>(PhantomData<fn() -> Tag>);

#[jungle::action]
impl<T> Action for Spawn<T>
where
    T: Tagged,
    <T as Tagged>::N: Unsigned,
    <<T as Tagged>::A as Animal>::Id: AnimalIdValue,
    <<T as Tagged>::A as Animal>::Generation: Unsigned,
    <<T as Tagged>::A as Animal>::Seed: Sync + Send + 'static,
{
    type Effect = super::effect::SpawnAnimal<<T as Tagged>::A>;
    type Input = <<T as Tagged>::A as Animal>::Seed;
    type Output = Uuid;
    type Carry = ();

    fn emit(_state: &super::SunState, input: Self::Input) -> <<T as Tagged>::A as Animal>::Seed {
        input
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let node_id = <<T as Tagged>::N as Unsigned>::U32;

        let outgoing_node_ids = <<T as Tagged>::E as NodeIdsFromList>::node_ids();

        // Lock the inner struct and register this node + its outgoing edges
        let mut inner = state.a.shared.lock().unwrap();

        // Store the journey ID for this node
        inner.journey_ids.insert(node_id, journey_id);

        // Register outgoing edges: this node -> each outgoing node
        inner.outgoing.insert(node_id, outgoing_node_ids.clone());

        // Register each outgoing node with this node as an incoming edge
        for target in outgoing_node_ids {
            inner.incoming.entry(target).or_default().push(node_id);
        }

        Ok(journey_id)
    }
}
