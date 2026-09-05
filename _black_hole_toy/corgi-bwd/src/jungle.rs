//! Client-side jungle glue: the ecosystem and its capability impls.

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::BackwardContract;
use black_hole_sun::ops::{BackwardOps, CheckpointOps, MassOps, StepOps, SunOps, VoidOps};
use black_hole_sun::{
    decode_input_gradient, decode_output, encode_input, encode_output_gradient, ArtifactRef,
    MassClient, ObjectId, OperationConfig, TensorContract, VoidClient,
};
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::{de::DeserializeOwned, Serialize};
use toy_common::dataset::SampleMetadata;

use corgi_fwd::contracts::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};

use crate::contracts::{
    CorgiBackward, HeadCell, Stage1Cell, Stage2Cell, Stage3Cell, Stage4Cell, StemCell,
    MICRO_BATCHES,
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
    pub optimizer_config: OperationConfig,
}

#[derive(Animals)]
pub struct CorgiAnimals(
    StemCell,
    Stage1Cell,
    Stage2Cell,
    Stage3Cell,
    Stage4Cell,
    HeadCell,
    CorgiBackward<MICRO_BATCHES>,
);

impl Ecosystem for CorgiJungle {
    const NAME: &'static str = "corgi-bwd";
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
async fn retag_forward<From, To>(
    void: &VoidClient,
    input: ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String>
where
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
{
    let bytes = void.receive_artifact(&input).await?;
    let decoded = decode_output::<From>(&bytes)?;
    let bytes = encode_input::<To>(&decoded.tensors, &decoded.metadata)?;
    Ok(ArtifactRef::from_object_id(void.upload(bytes).await?))
}

async fn retag_backward<From, To>(
    void: &VoidClient,
    input: ArtifactRef<To::OutputGrad>,
) -> Result<ArtifactRef<To::OutputGrad>, String>
where
    From: BackwardContract<Metadata = SampleMetadata>,
    To: BackwardContract<Metadata = SampleMetadata>,
{
    let bytes = void.receive_artifact(&input).await?;
    let decoded = decode_input_gradient::<From>(&bytes)?;
    let bytes = encode_output_gradient::<To>(&decoded.tensors, &decoded.metadata)?;
    Ok(ArtifactRef::from_object_id(void.upload(bytes).await?))
}

macro_rules! jungle_ops {
    ($contract:ty, $field:ident, $forward_from:ty, $backward_from:ty) => {
        #[async_trait]
        impl MassOps<$contract> for CorgiJungle {
            async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field
                    .start_backward_operation(id, Some(self.optimizer_config.clone()))
                    .await
            }
            async fn forward(
                &self,
                id: ObjectId,
                input: ArtifactRef<<$contract as TensorContract>::Input>,
            ) -> Result<ArtifactRef<<$contract as TensorContract>::Output>, String> {
                let input = retag_forward::<$forward_from, $contract>(&self.void, input).await?;
                self.$field.forward(id, input).await
            }
            async fn shutdown_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field.shutdown_operation(id).await
            }
        }
        #[async_trait]
        impl BackwardOps<$contract> for CorgiJungle {
            async fn backward(
                &self,
                id: ObjectId,
                gradient: ArtifactRef<<$contract as BackwardContract>::OutputGrad>,
            ) -> Result<ArtifactRef<<$contract as BackwardContract>::InputGrad>, String> {
                let gradient =
                    retag_backward::<$backward_from, $contract>(&self.void, gradient).await?;
                self.$field.backward(id, gradient).await
            }
        }
        #[async_trait]
        impl StepOps<$contract> for CorgiJungle {
            async fn step(&self, id: ObjectId) -> Result<(), String> {
                self.$field.step(id).await
            }
        }
        #[async_trait]
        impl CheckpointOps<$contract> for CorgiJungle {
            async fn checkpoint_operation(&self, id: ObjectId) -> Result<ObjectId, String> {
                self.$field.checkpoint_operation(id).await
            }
        }
    };
}

#[async_trait]
impl MassOps<StemOp> for CorgiJungle {
    async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
        self.stem
            .start_backward_operation(id, Some(self.optimizer_config.clone()))
            .await
    }
    async fn forward(
        &self,
        id: ObjectId,
        input: ArtifactRef<<StemOp as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<StemOp as TensorContract>::Output>, String> {
        self.stem.forward(id, input).await
    }
    async fn shutdown_operation(&self, id: ObjectId) -> Result<(), String> {
        self.stem.shutdown_operation(id).await
    }
}
#[async_trait]
impl BackwardOps<StemOp> for CorgiJungle {
    async fn backward(
        &self,
        id: ObjectId,
        gradient: ArtifactRef<<StemOp as BackwardContract>::OutputGrad>,
    ) -> Result<ArtifactRef<<StemOp as BackwardContract>::InputGrad>, String> {
        self.stem
            .backward(
                id,
                retag_backward::<Stage1Op, StemOp>(&self.void, gradient).await?,
            )
            .await
    }
}
#[async_trait]
impl StepOps<StemOp> for CorgiJungle {
    async fn step(&self, id: ObjectId) -> Result<(), String> {
        self.stem.step(id).await
    }
}
#[async_trait]
impl CheckpointOps<StemOp> for CorgiJungle {
    async fn checkpoint_operation(&self, id: ObjectId) -> Result<ObjectId, String> {
        self.stem.checkpoint_operation(id).await
    }
}

jungle_ops!(Stage1Op, stage1, StemOp, Stage2Op);
jungle_ops!(Stage2Op, stage2, Stage1Op, Stage3Op);
jungle_ops!(Stage3Op, stage3, Stage2Op, Stage4Op);
jungle_ops!(Stage4Op, stage4, Stage3Op, HeadOp);

// The head consumes Stage4 activations. Its gradient seed is reframed by the
// node and therefore already has HeadOp's reverse descriptor.
#[async_trait]
impl MassOps<HeadOp> for CorgiJungle {
    async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
        self.head
            .start_backward_operation(id, Some(self.optimizer_config.clone()))
            .await
    }
    async fn forward(
        &self,
        id: ObjectId,
        input: ArtifactRef<<HeadOp as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<HeadOp as TensorContract>::Output>, String> {
        self.head
            .forward(
                id,
                retag_forward::<Stage4Op, HeadOp>(&self.void, input).await?,
            )
            .await
    }
    async fn shutdown_operation(&self, id: ObjectId) -> Result<(), String> {
        self.head.shutdown_operation(id).await
    }
}
#[async_trait]
impl BackwardOps<HeadOp> for CorgiJungle {
    async fn backward(
        &self,
        id: ObjectId,
        gradient: ArtifactRef<<HeadOp as BackwardContract>::OutputGrad>,
    ) -> Result<ArtifactRef<<HeadOp as BackwardContract>::InputGrad>, String> {
        self.head.backward(id, gradient).await
    }
}
#[async_trait]
impl StepOps<HeadOp> for CorgiJungle {
    async fn step(&self, id: ObjectId) -> Result<(), String> {
        self.head.step(id).await
    }
}
#[async_trait]
impl CheckpointOps<HeadOp> for CorgiJungle {
    async fn checkpoint_operation(&self, id: ObjectId) -> Result<ObjectId, String> {
        self.head.checkpoint_operation(id).await
    }
}

#[async_trait]
impl SunOps for CorgiJungle {
    async fn spawn_animal<A>(&self, seed: &A::Seed) -> Result<uuid::Uuid, String>
    where
        A: Animal,
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Sync + Send,
    {
        self.client
            .spawn::<A>(seed)
            .await
            .map(|h| h.journey_id)
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
        Err("perturbation is not used by corgi-bwd".into())
    }
}

pub fn required_capabilities() -> black_hole_sun::OperationCapabilities {
    black_hole_sun::OperationCapabilities {
        forward: true,
        backward: true,
        step: true,
        checkpoint: true,
        ..Default::default()
    }
}
