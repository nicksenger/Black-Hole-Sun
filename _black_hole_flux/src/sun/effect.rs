//! Sun effects — spawning animals into the jungle.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::ops::{SunOps, VoidInferOps};
use crate::NucleusError;

// ---------------------------------------------------------------------------
// SpawnAnimal — spawn an animal and return its journey ID
// ---------------------------------------------------------------------------

/// Effect that spawns an animal of type `A` into the jungle.
///
/// Takes the animal's seed as input, calls [`JungleClient::spawn`](jungle_sdk::JungleClient::spawn),
/// and returns the journey UUID.
pub struct SpawnAnimal<A>(PhantomData<fn() -> A>);

impl<A, J> EffectSchema<J> for SpawnAnimal<A>
where
    A: Animal,
    A::Id: AnimalIdValue,
    A::Generation: typosaurus::num::Unsigned,
    A::Seed: Sync + Send + 'static,
{
    type Id = u64;
    type In = A::Seed;
    type Out = Uuid;
    type Err = NucleusError;
}

impl<A, J> Effect<J> for SpawnAnimal<A>
where
    A: Animal,
    A::Id: AnimalIdValue,
    A::Generation: typosaurus::num::Unsigned,
    A::Seed: Sync + Send + 'static,
    J: SunOps,
{
    fn effect(
        jungle: &J,
        seed: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(animal_id = <A::Id as AnimalIdValue>::U32, "spawning animal");
            let journey_id = jungle
                .spawn_animal::<A>(&seed)
                .await
                .map_err(NucleusError::Spawn)?;
            debug!(?journey_id, "animal spawned");
            Ok(journey_id)
        }
    }
}
