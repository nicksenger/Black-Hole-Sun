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

use crate::contracts::{Matmul, MatmulCell, MatmulForward, Relu, ReluCell, Scale, ScaleCell};

#[derive(Clone)]
pub struct TensorJungle {
    pub client: FusedClient,
    pub void: VoidClient,
    pub matmul: MassClient<Matmul>,
    pub scale: MassClient<Scale>,
    pub relu: MassClient<Relu>,
}

#[derive(Animals)]
pub struct TensorAnimals(MatmulCell, ScaleCell, ReluCell, MatmulForward);

impl Ecosystem for TensorJungle {
    const NAME: &'static str = "tensor-fwd";
    type Animals = TensorAnimals;
}

#[async_trait]
impl VoidOps for TensorJungle {
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
async fn retag<M, From, To>(
    void: &VoidClient,
    input: &ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String>
where
    From: TensorContract<Metadata = M>,
    To: TensorContract<Metadata = M>,
    M: Serialize + Clone + Send + Sync + DeserializeOwned,
{
    let bytes = void.receive_artifact(input).await?;
    let decoded = decode_output::<From>(&bytes).map_err(|error| error.to_string())?;
    let tagged = encode_input::<To>(&decoded.tensors, &decoded.metadata)
        .map_err(|error| error.to_string())?;
    Ok(ArtifactRef::from_object_id(void.upload(tagged).await?))
}

#[async_trait]
impl MassOps<Matmul> for TensorJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.matmul.start_operation(instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<Matmul as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<Matmul as TensorContract>::Output>, String> {
        self.matmul.forward(instance_id, input).await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.matmul.shutdown_operation(instance_id).await
    }
}

#[async_trait]
impl MassOps<Scale> for TensorJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.start_operation(instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<Scale as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<Scale as TensorContract>::Output>, String> {
        let input = retag::<(), Matmul, Scale>(&self.void, &input).await?;
        self.scale.forward(instance_id, input).await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.shutdown_operation(instance_id).await
    }
}

#[async_trait]
impl MassOps<Relu> for TensorJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.start_operation(instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<Relu as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<Relu as TensorContract>::Output>, String> {
        let input = retag::<(), Scale, Relu>(&self.void, &input).await?;
        self.relu.forward(instance_id, input).await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.shutdown_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Matmul> for TensorJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.matmul.reset_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Scale> for TensorJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.reset_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Relu> for TensorJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.reset_operation(instance_id).await
    }
}

#[async_trait]
impl SunOps for TensorJungle {
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
            .map_err(|error| error.to_string())
    }

    async fn observe_animal<Appearance>(&self, journey_id: uuid::Uuid) -> Result<Appearance, String>
    where
        Appearance: DeserializeOwned + Send,
    {
        let bytes = self
            .client
            .animal_appearance(journey_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("appearance unavailable for {journey_id}"))?;
        postcard::from_bytes(&bytes).map_err(|error| error.to_string())
    }

    async fn perturb_animal<S>(&self, _journey_id: uuid::Uuid, _stimulus: &S) -> Result<(), String>
    where
        S: Serialize + Sync + Send,
    {
        Err("perturbation is not used by tensor-fwd".to_owned())
    }
}

pub fn forward_reset_capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        ..OperationCapabilities::default()
    }
}
