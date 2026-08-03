//! Sun actions — spawning animals and populating sun state.

use std::marker::PhantomData;

use super::Tagged;
use jungle_sdk::prelude::*;
use typosaurus::collections::list::{Empty, List};
use typosaurus::num::Unsigned;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// EdgeIdsFromList — extract runtime edge UUIDs from a type-level integer list
// ---------------------------------------------------------------------------

/// Trait that converts a type-level list of typenum integers into a runtime
/// vector of edge UUIDs.
///
/// Each typenum integer `N` in the list is converted to a UUID by placing
/// `N::U32` in the last four bytes of an otherwise-zero UUID.
pub trait EdgeIdsFromList {
    fn edge_ids() -> Vec<Uuid>;

    #[inline]
    fn uuid_from<N: Unsigned>() -> Uuid {
        let mut bytes = [0u8; 16];
        bytes[12..16].copy_from_slice(&<N as Unsigned>::U32.to_be_bytes());
        Uuid::from_bytes(bytes)
    }
}

impl EdgeIdsFromList for Empty {
    fn edge_ids() -> Vec<Uuid> {
        Vec::new()
    }
}

impl<H, T> EdgeIdsFromList for List<(H, T)>
where
    H: Unsigned,
    T: EdgeIdsFromList,
{
    fn edge_ids() -> Vec<Uuid> {
        let mut ids = vec![Self::uuid_from::<H>()];
        ids.extend(T::edge_ids());
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
/// outgoing map with deterministic edge UUIDs derived from the type-level edge list `E`.
///
/// Returns the journey UUID for downstream use.
pub struct Spawn<Tag>(PhantomData<fn() -> Tag>);

#[jungle::action]
impl<T> Action for Spawn<T>
where
    T: Tagged,
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

        let edge_ids = <<T as Tagged>::E as EdgeIdsFromList>::edge_ids();

        // Register outgoing edges: this node -> edge UUIDs
        state.outgoing.insert(journey_id, edge_ids.clone());

        // Register each edge as pointing back to this node (for incoming lookup)
        for edge_id in edge_ids {
            state.incoming.entry(edge_id).or_default().push(journey_id);
        }

        Ok(journey_id)
    }
}
