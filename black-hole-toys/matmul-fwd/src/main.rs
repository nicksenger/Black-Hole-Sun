use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::ops::{MassOps, ResetOps, SunOps, VoidOps};
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::{
    decode_input, decode_output, encode_output, ArtifactRef, MassClient, MassServerBuilder,
    ObjectId, OperationCapabilities, OperationCapability, OperationConfig, OperationImplementation,
    RawTensor, TensorContract, TensorDtype, VoidClient, VoidServerBuilder,
};
use black_hole_sun::{object_store, persist};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use matmul_fwd::{
    Matmul, MatmulCell, MatmulForward, RawArtifactOps, Relu, ReluCell, Scale, ScaleCell,
    LOGGED_OUTPUTS,
};
use serde::Serialize;

const TARGET_PASSES: usize = 4;

struct MatmulOperation;

#[async_trait]
impl OperationImplementation for MatmulOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Matmul>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = decode_input::<Matmul>(&input).map_err(|error| error.to_string())?;
        let values = input.tensors[0]
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let weights = [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let mut output = Vec::with_capacity(8);
        for row in 0..2 {
            for column in 0..4 {
                output.push(
                    (0..3)
                        .map(|k| values[row * 3 + k] * weights[k][column])
                        .sum(),
                );
            }
        }
        encode_output::<Matmul>(&[tensor("product_matrix", [2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}

struct ScaleOperation;

#[async_trait]
impl OperationImplementation for ScaleOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Scale>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = decode_input::<Scale>(&input).map_err(|error| error.to_string())?;
        let output = input.tensors[0]
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()) * 0.5)
            .collect::<Vec<_>>();
        encode_output::<Scale>(&[tensor("product_matrix", [2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}

struct ReluOperation;

#[async_trait]
impl OperationImplementation for ReluOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<Relu>();
        capability.operations.reset = true;
        capability
    }

    async fn start(
        &self,
        _instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn forward(&self, _instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = decode_input::<Relu>(&input).map_err(|error| error.to_string())?;
        let output = input.tensors[0]
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).max(0.0))
            .collect::<Vec<_>>();
        encode_output::<Relu>(&[tensor("product_matrix", [2, 4], output)], &())
            .map_err(|error| error.to_string())
    }

    async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }

    async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
        Ok(())
    }
}

fn tensor(name: &str, shape: [usize; 2], values: Vec<f32>) -> RawTensor {
    RawTensor {
        name: name.to_owned(),
        dtype: TensorDtype::F32,
        shape: shape.to_vec(),
        data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
    }
}

#[derive(Clone)]
struct MatmulJungle {
    client: FusedClient,
    void: VoidClient,
    matmul: MassClient<Matmul>,
    scale: MassClient<Scale>,
    relu: MassClient<Relu>,
}

#[derive(Animals)]
struct MatmulAnimals(MatmulCell, ScaleCell, ReluCell, MatmulForward);

impl Ecosystem for MatmulJungle {
    const NAME: &'static str = "matmul-fwd";
    type Animals = MatmulAnimals;
}

#[async_trait]
impl VoidOps for MatmulJungle {
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

#[async_trait]
impl RawArtifactOps for MatmulJungle {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String> {
        self.void.receive_artifact(reference).await
    }
}

#[async_trait]
impl MassOps<Matmul> for MatmulJungle {
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
impl MassOps<Scale> for MatmulJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.start_operation(instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<Scale as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<Scale as TensorContract>::Output>, String> {
        // The tensor bundle is shared by the edge, but the wire envelope is
        // tagged with the producing operation's descriptor. Re-encode it for
        // the consuming operation before crossing the Mass boundary.
        let bytes = self.void.receive_artifact(&input).await?;
        let decoded = decode_output::<Matmul>(&bytes).map_err(|error| error.to_string())?;
        let input = black_hole_sun::encode_input::<Scale>(&decoded.tensors, &decoded.metadata)
            .map_err(|error| error.to_string())?;
        let input_id = self.void.upload(input).await?;
        self.scale
            .forward(instance_id, ArtifactRef::from_object_id(input_id))
            .await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.shutdown_operation(instance_id).await
    }
}

#[async_trait]
impl MassOps<Relu> for MatmulJungle {
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.start_operation(instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<Relu as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<Relu as TensorContract>::Output>, String> {
        // Re-tag the previous operation's output for this operation's input
        // contract while preserving the typed tensor shape.
        let bytes = self.void.receive_artifact(&input).await?;
        let decoded = decode_output::<Scale>(&bytes).map_err(|error| error.to_string())?;
        let input = black_hole_sun::encode_input::<Relu>(&decoded.tensors, &decoded.metadata)
            .map_err(|error| error.to_string())?;
        let input_id = self.void.upload(input).await?;
        self.relu
            .forward(instance_id, ArtifactRef::from_object_id(input_id))
            .await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.shutdown_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Matmul> for MatmulJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.matmul.reset_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Scale> for MatmulJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.scale.reset_operation(instance_id).await
    }
}

#[async_trait]
impl ResetOps<Relu> for MatmulJungle {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.relu.reset_operation(instance_id).await
    }
}

#[async_trait]
impl SunOps for MatmulJungle {
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
        Appearance: serde::de::DeserializeOwned + Send,
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
        Err("perturbation is not used by matmul-fwd".to_owned())
    }
}

fn server_address() -> SocketAddr {
    "127.0.0.1:0".parse().expect("valid loopback address")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;

    let (void_addr, void_task) = VoidServerBuilder::new(
        Box::new(object_store::InMemoryObjectStore::new()),
        Box::new(persist::InMemoryStore::new()),
    )
    .tcp()
    .listen(server_address())
    .serve()
    .await?;

    let (matmul_addr, matmul_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(MatmulOperation)
        .serve()
        .await?;
    let (scale_addr, scale_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(ScaleOperation)
        .serve()
        .await?;
    let (relu_addr, relu_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(ReluOperation)
        .serve()
        .await?;

    let client = FusedClient::builder().build().await?;
    let jungle = MatmulJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(void_addr),
        matmul: MassClient::new_tcp_typed(matmul_addr).requiring(forward_reset_capabilities()),
        scale: MassClient::new_tcp_typed(scale_addr).requiring(forward_reset_capabilities()),
        relu: MassClient::new_tcp_typed(relu_addr).requiring(forward_reset_capabilities()),
    };

    let _parent = client.spawn::<MatmulForward>(&()).await?;
    let workers = (0..4)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                if let Err(error) = worker.spawn().await {
                    eprintln!("matmul-fwd worker stopped: {error}");
                }
            })
        })
        .collect::<Vec<_>>();

    let result = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if LOGGED_OUTPUTS.load(Ordering::Acquire) >= TARGET_PASSES {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;

    for worker in workers {
        worker.abort();
        let _ = worker.await;
    }
    void_task.abort();
    matmul_task.abort();
    scale_task.abort();
    relu_task.abort();

    result.map_err(|_| "timed out waiting for matmul-fwd output")?;
    println!(
        "matmul-fwd completed {} output pass(es)",
        LOGGED_OUTPUTS.load(Ordering::Acquire)
    );
    Ok(())
}

fn forward_reset_capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        ..OperationCapabilities::default()
    }
}
