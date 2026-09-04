//! Client-side jungle glue: the ecosystem and its capability impls.

use async_trait::async_trait;
use black_hole_sun::ops::{MassOps, ResetOps, SunOps, VoidOps};
use black_hole_sun::{
    decode_output, encode_input, ArtifactRef, MassClient, ObjectId, OperationCapabilities,
    TensorContract, VoidClient,
};
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::{Serialize, de::DeserializeOwned};
use toy_common::dataset::SampleMetadata;

use crate::contracts::{
    CorgiForward, HeadCell, HeadOp, Stage1Cell, Stage1Op, Stage2Cell, Stage2Op, Stage3Cell,
    Stage3Op, Stage4Cell, Stage4Op, StemCell, StemOp,
};

#[derive(Clone)]
pub struct CorgiJungle {
    pub client: FusedClient,
    pub void: VoidClient,
    pub stem: MassClient<StemOp>,
    pub stage1: MassClient<Stage1Op>,
    pub stage2: MassClient<Stage2Op>,
    pub stage3: MassClient<Stage3Op>,
    pub stage4: MassClient<Stage4Op>,
    pub head: MassClient<HeadOp>,
}

#[derive(Animals)]
pub struct CorgiAnimals(
    StemCell,
    Stage1Cell,
    Stage2Cell,
    Stage3Cell,
    Stage4Cell,
    HeadCell,
    CorgiForward,
);

impl Ecosystem for CorgiJungle {
    const NAME: &'static str = "corgi-fwd";
    type Animals = CorgiAnimals;
}

#[async_trait]
impl VoidOps for CorgiJungle {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        self.void.download(id).await
    }
    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        self.void.download_wait(id, timeout_ms).await
    }
    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        self.void.upload(data).await
    }
    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        self.void.upload_with(id, data).await.map(|_| ())
    }
}

/// Re-tag the previous operation's output for the next operation's input
/// contract while preserving the typed tensor bundle. The wire envelope is
/// tagged with the producing operation's descriptor, so it must be re-encoded
/// before crossing the Mass boundary.
async fn retag<From, To>(
    void: &VoidClient,
    input: &ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String>
where
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
{
    let bytes = void.receive_artifact(input).await?;
    let decoded = decode_output::<From>(&bytes)?;
    let tagged = encode_input::<To>(&decoded.tensors, &decoded.metadata)?;
    Ok(ArtifactRef::from_object_id(void.upload(tagged).await?))
}

macro_rules! mass_ops {
    ($contract:ty, $field:ident, $from:ty, $to:ty) => {
        #[async_trait]
        impl MassOps<$contract> for CorgiJungle {
            async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
                self.$field.start_operation(instance_id).await
            }
            async fn forward(
                &self,
                instance_id: ObjectId,
                input: ArtifactRef<<$contract as TensorContract>::Input>,
            ) -> Result<ArtifactRef<<$contract as TensorContract>::Output>, String> {
                let input = retag::<$from, $to>(&self.void, &input).await?;
                self.$field.forward(instance_id, input).await
            }
            async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
                self.$field.shutdown_operation(instance_id).await
            }
        }
        #[async_trait]
        impl ResetOps<$contract> for CorgiJungle {
            async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
                self.$field.reset_operation(instance_id).await
            }
        }
    };
}

#[async_trait]
impl MassOps<StemOp> for CorgiJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.stem.start_operation(instance_id).await
    }
    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<StemOp as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<StemOp as TensorContract>::Output>, String> {
        tracing::info!("sending image to stem operation");
        let output = self.stem.forward(instance_id, input).await;
        tracing::info!(success = output.is_ok(), "stem operation request completed");
        output
    }
    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.stem.shutdown_operation(instance_id).await
    }
}
#[async_trait]
impl ResetOps<StemOp> for CorgiJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.stem.reset_operation(instance_id).await
    }
}
mass_ops!(Stage1Op, stage1, StemOp, Stage1Op);
mass_ops!(Stage2Op, stage2, Stage1Op, Stage2Op);
mass_ops!(Stage3Op, stage3, Stage2Op, Stage3Op);
mass_ops!(Stage4Op, stage4, Stage3Op, Stage4Op);
mass_ops!(HeadOp, head, Stage4Op, HeadOp);

#[async_trait]
impl SunOps for CorgiJungle {
    async fn spawn_animal<A>(&self, seed: &A::Seed) -> Result<uuid::Uuid, String>
    where
        A: jungle_sdk::Animal,
        A::Id: jungle_sdk::AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Sync + Send,
    {
        self.client
            .spawn::<A>(seed)
            .await
            .map(|handle| handle.journey_id)
            .map_err(|e| e.to_string())
    }

    async fn observe_animal<Appearance>(&self, journey_id: uuid::Uuid) -> Result<Appearance, String>
    where
        Appearance: DeserializeOwned + Send,
    {
        let bytes = self
            .client
            .animal_appearance(journey_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("appearance unavailable for {journey_id}"))?;
        postcard::from_bytes(&bytes).map_err(|e| e.to_string())
    }

    async fn perturb_animal<S>(&self, _journey_id: uuid::Uuid, _stimulus: &S) -> Result<(), String>
    where
        S: Serialize + Sync + Send,
    {
        Err("perturbation is not used by corgi-fwd".to_owned())
    }
}

pub fn capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        ..OperationCapabilities::default()
    }
}
