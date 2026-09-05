//! Jungle capabilities for the two-replica corgi pipeline.

use async_trait::async_trait;
use black_hole_sun::ops::{BackwardOps, CheckpointOps, MassOps, StepOps, SunOps, VoidOps};
use black_hole_sun::{ArtifactRef, ObjectId};
use jungle_sdk::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

use corgi_fwd::spec::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};

use crate::spec::{
    CorgiParallel, HeadFusion, Stage1Fusion, Stage2Fusion, Stage3Fusion, Stage4Fusion, StemFusion,
    MICRO_BATCHES,
};

#[derive(Clone)]
pub struct CorgiParallelJungle {
    pub inner: corgi_bwd::flow::CorgiJungle,
}

#[derive(Animals)]
pub struct CorgiParallelAnimals(
    StemFusion,
    Stage1Fusion,
    Stage2Fusion,
    Stage3Fusion,
    Stage4Fusion,
    HeadFusion,
    CorgiParallel<MICRO_BATCHES>,
);

impl Ecosystem for CorgiParallelJungle {
    const NAME: &'static str = "corgi-par";
    type Animals = CorgiParallelAnimals;
}

#[async_trait]
impl VoidOps for CorgiParallelJungle {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        self.inner.download_raw(id).await
    }
    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        self.inner.download_raw_wait(id, timeout_ms).await
    }
    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        self.inner.upload_to_void(data).await
    }
    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        self.inner.upload_to_void_with(id, data).await
    }
}

macro_rules! delegate_operation {
    ($op:ty) => {
        #[async_trait]
        impl MassOps<$op> for CorgiParallelJungle {
            async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
                MassOps::<$op>::start_operation(&self.inner, id).await
            }
            async fn forward(
                &self,
                id: ObjectId,
                input: ArtifactRef<<$op as black_hole_sun::TensorContract>::Input>,
            ) -> Result<ArtifactRef<<$op as black_hole_sun::TensorContract>::Output>, String> {
                MassOps::<$op>::forward(&self.inner, id, input).await
            }
            async fn shutdown_operation(&self, id: ObjectId) -> Result<(), String> {
                MassOps::<$op>::shutdown_operation(&self.inner, id).await
            }
        }

        #[async_trait]
        impl BackwardOps<$op> for CorgiParallelJungle {
            async fn backward(
                &self,
                id: ObjectId,
                gradient: ArtifactRef<
                    <$op as black_hole_sun::black_hole_spec::BackwardContract>::OutputGrad,
                >,
            ) -> Result<
                ArtifactRef<<$op as black_hole_sun::black_hole_spec::BackwardContract>::InputGrad>,
                String,
            > {
                BackwardOps::<$op>::backward(&self.inner, id, gradient).await
            }
        }

        #[async_trait]
        impl StepOps<$op> for CorgiParallelJungle {
            async fn step(&self, id: ObjectId) -> Result<(), String> {
                StepOps::<$op>::step(&self.inner, id).await
            }
        }

        #[async_trait]
        impl CheckpointOps<$op> for CorgiParallelJungle {
            async fn checkpoint_operation(&self, id: ObjectId) -> Result<ObjectId, String> {
                CheckpointOps::<$op>::checkpoint_operation(&self.inner, id).await
            }
        }
    };
}

delegate_operation!(StemOp);
delegate_operation!(Stage1Op);
delegate_operation!(Stage2Op);
delegate_operation!(Stage3Op);
delegate_operation!(Stage4Op);
delegate_operation!(HeadOp);

#[async_trait]
impl SunOps for CorgiParallelJungle {
    async fn spawn_animal<A>(&self, seed: &A::Seed) -> Result<uuid::Uuid, String>
    where
        A: Animal,
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Sync + Send,
    {
        self.inner.spawn_animal::<A>(seed).await
    }

    async fn observe_animal<Appearance>(&self, journey_id: uuid::Uuid) -> Result<Appearance, String>
    where
        Appearance: DeserializeOwned + Send,
    {
        self.inner.observe_animal(journey_id).await
    }

    async fn perturb_animal<S>(&self, journey_id: uuid::Uuid, stimulus: &S) -> Result<(), String>
    where
        S: Serialize + Sync + Send,
    {
        self.inner.perturb_animal(journey_id, stimulus).await
    }
}
