//! Deployment effects — UUID/fusion-seed generation and animal spawning.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use tracing::debug;
use uuid::Uuid;

use crate::nodes::fusion::action::FusionSeed;
use crate::ops::SunOps;
use crate::AtomError;

pub struct GenUuidEffect;
#[jungle::effect(id = 51)]
impl<J> Effect<J> for GenUuidEffect {
    type In = ();
    type Out = Uuid;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async { Ok(Uuid::new_v4()) }
    }
}

pub struct GenFusionSeedEffect;
#[jungle::effect(id = 52)]
impl<J> Effect<J> for GenFusionSeedEffect {
    type In = ();
    type Out = FusionSeed;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            Ok(FusionSeed {
                p1_recv_id: Uuid::new_v4(),
                p2_recv_id: Uuid::new_v4(),
                grad_steps: 1,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// SpawnAnimal — spawn an animal and return its journey ID
// ---------------------------------------------------------------------------

/// Effect that spawns an animal of type `A` into the jungle.
pub struct SpawnAnimal<A>(PhantomData<fn() -> A>);
#[jungle::effect(id = 53)]
impl<
        A: Animal<
            Id: AnimalIdValue,
            Generation: typosaurus::num::Unsigned,
            Seed: Sync + Send + 'static,
        >,
        J: SunOps,
    > Effect<J> for SpawnAnimal<A>
{
    type In = A::Seed;
    type Out = Uuid;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        seed: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(animal_id = <A::Id as AnimalIdValue>::U32, "spawning animal");
            let journey_id = jungle
                .spawn_animal::<A>(&seed)
                .await
                .map_err(AtomError::Spawn)?;
            debug!(?journey_id, "animal spawned");
            Ok(journey_id)
        }
    }
}