//! Client-side jungle glue: the ecosystem and its capability impls.

use async_trait::async_trait;
use black_hole_sun::ops::{MassOps, OptimizeOps, PerturbOps, ResetOps, SunOps, VoidInferOps};
use black_hole_sun::{
    decode_output, encode_input, ArtifactRef, DarkToken, EmissionId, InferenceRequest,
    MassClient, MassModelConfig, MassModelParams, ObjectId, OperationCapabilities,
    TensorContract, Transmission, VoidClient,
};
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::{Serialize, de::DeserializeOwned};
use toys_common::dataset::SampleMetadata;

use crate::contracts::{CorgiZo, HeadCell, Stage1Cell, Stage2Cell, Stage3Cell, Stage4Cell, StemCell};
use corgi_fwd::contracts::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};

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
    CorgiZo,
);

impl Ecosystem for CorgiJungle {
    const NAME: &'static str = "corgi-zo";
    type Animals = CorgiAnimals;
}

#[async_trait]
impl VoidInferOps for CorgiJungle {
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
    async fn start_model(
        &self,
        _model_id: uuid::Uuid,
        _config: Option<MassModelConfig>,
    ) -> Result<(), String> {
        Err("Qwen model lifecycle is not used by corgi-zo".into())
    }
    async fn infer(
        &self,
        _model_id: uuid::Uuid,
        _request: InferenceRequest,
    ) -> Result<ObjectId, String> {
        Err("Qwen inference is not used by corgi-zo".into())
    }
    async fn reset_model(&self, _model_id: uuid::Uuid) -> Result<(), String> {
        Err("use typed operation reset".into())
    }
    async fn checkpoint_model(&self, _model_id: uuid::Uuid) -> Result<ObjectId, String> {
        Err("checkpoint not used".into())
    }
    async fn fuse_weights(
        &self,
        _model_id: uuid::Uuid,
        _checkpoint_id: ObjectId,
        _contribution: f32,
    ) -> Result<ObjectId, String> {
        Err("fusion not used".into())
    }
    fn darken(&self, _prompt: &str) -> Result<Vec<DarkToken>, String> {
        Err("darkening not used".into())
    }
    fn decode(&self, _tokens: &[DarkToken]) -> String {
        String::new()
    }
    async fn perturb_up(&self, _model_id: uuid::Uuid, _seed: u64) -> Result<(), String> {
        Err("use typed operation perturbation".into())
    }
    async fn perturb_down(&self, _model_id: uuid::Uuid) -> Result<(), String> {
        Err("use typed operation perturbation".into())
    }
    async fn optimize(
        &self,
        _model_id: uuid::Uuid,
        _loss_up: f32,
        _loss_down: f32,
    ) -> Result<(), String> {
        Err("use typed operation optimization".into())
    }
    async fn query_model_params(&self, _model_id: uuid::Uuid) -> Result<MassModelParams, String> {
        Err("query not used".into())
    }
    async fn shutdown_model(&self, _model_id: uuid::Uuid) -> Result<(), String> {
        Err("use typed operation shutdown".into())
    }
    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String> {
        let transmission = Transmission::Propagation {
            emission_id,
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        };
        self.void
            .upload_with(
                send_id,
                postcard::to_allocvec(&transmission).map_err(|e| e.to_string())?,
            )
            .await
            .map(|_| ())
    }
}

/// Re-tag the previous operation's output for the next operation's input
/// contract while preserving the typed tensor bundle. The wire envelope is
/// tagged with the producing operation's descriptor, so it must be re-encoded
/// before crossing the Mass boundary.
async fn retag<
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
>(
    void: &VoidClient,
    input: &ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String> {
    let bytes = void.receive_artifact(input).await?;
    let decoded = decode_output::<From>(&bytes)?;
    let tagged = encode_input::<To>(&decoded.tensors, &decoded.metadata)?;
    Ok(ArtifactRef::from_object_id(void.upload(tagged).await?))
}

macro_rules! mass_ops {
    ($contract:ty, $field:ident, $from:ty, $to:ty) => {
        #[async_trait]
        impl MassOps<$contract> for CorgiJungle {
            async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field.start_operation(id).await
            }
            async fn forward(
                &self,
                id: ObjectId,
                input: ArtifactRef<<$contract as TensorContract>::Input>,
            ) -> Result<ArtifactRef<<$contract as TensorContract>::Output>, String> {
                self.$field
                    .forward(id, retag::<$from, $to>(&self.void, &input).await?)
                    .await
            }
            async fn shutdown_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field.shutdown_operation(id).await
            }
        }
        #[async_trait]
        impl ResetOps<$contract> for CorgiJungle {
            async fn reset_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field.reset_operation(id).await
            }
        }
        #[async_trait]
        impl PerturbOps<$contract> for CorgiJungle {
            async fn perturb_up_operation(&self, id: ObjectId, seed: u64) -> Result<(), String> {
                self.$field.perturb_up_operation(id, seed).await
            }
            async fn perturb_down_operation(&self, id: ObjectId) -> Result<(), String> {
                self.$field.perturb_down_operation(id).await
            }
        }
        #[async_trait]
        impl OptimizeOps<$contract> for CorgiJungle {
            async fn optimize_operation(
                &self,
                id: ObjectId,
                up: f32,
                down: f32,
            ) -> Result<(), String> {
                self.$field.optimize_operation(id, up, down).await
            }
        }
    };
}

#[async_trait]
impl MassOps<StemOp> for CorgiJungle {
    async fn start_operation(&self, id: ObjectId) -> Result<(), String> {
        self.stem.start_operation(id).await
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
impl ResetOps<StemOp> for CorgiJungle {
    async fn reset_operation(&self, id: ObjectId) -> Result<(), String> {
        self.stem.reset_operation(id).await
    }
}
#[async_trait]
impl PerturbOps<StemOp> for CorgiJungle {
    async fn perturb_up_operation(&self, id: ObjectId, seed: u64) -> Result<(), String> {
        self.stem.perturb_up_operation(id, seed).await
    }
    async fn perturb_down_operation(&self, id: ObjectId) -> Result<(), String> {
        self.stem.perturb_down_operation(id).await
    }
}
#[async_trait]
impl OptimizeOps<StemOp> for CorgiJungle {
    async fn optimize_operation(&self, id: ObjectId, up: f32, down: f32) -> Result<(), String> {
        self.stem.optimize_operation(id, up, down).await
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
        A: Animal,
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Sync + Send,
    {
        self.client
            .spawn::<A>(seed)
            .await
            .map(|handle| handle.journey_id)
            .map_err(|e| e.to_string())
    }
    async fn observe_animal<Appearance: DeserializeOwned + Send>(
        &self,
        journey_id: uuid::Uuid,
    ) -> Result<Appearance, String> {
        let bytes = self
            .client
            .animal_appearance(journey_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("appearance unavailable for {journey_id}"))?;
        postcard::from_bytes(&bytes).map_err(|e| e.to_string())
    }
    async fn perturb_animal<S: Serialize + Sync + Send>(
        &self,
        _journey_id: uuid::Uuid,
        _stimulus: &S,
    ) -> Result<(), String> {
        Err("external perturbation is not used".into())
    }
}

pub fn capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        perturb: true,
        optimize: true,
        ..OperationCapabilities::default()
    }
}
