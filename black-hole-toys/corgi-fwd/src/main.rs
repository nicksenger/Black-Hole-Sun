use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::{env, fs};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::ops::{MassOps, ResetOps, SunOps, VoidOps};
use black_hole_sun::{
    decode_input, decode_output, encode_output, ArtifactRef, MassClient, MassServerBuilder,
    ObjectId, OperationCapabilities, OperationCapability, OperationConfig, OperationImplementation,
    RawTensor, TensorContract, TensorDtype, TensorSpec, VoidClient, VoidServerBuilder,
};
use black_hole_sun::{object_store, persist};
use candle::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::Parser;
use corgi_fwd::{
    build_head, build_stage1, build_stage2, build_stage3, build_stage4, build_stem, pool_stage4,
    CorgiForward, HeadOp, SampleMetadata, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp,
    DATASET_SAMPLES, LOGGED_OUTPUTS,
};
use hf_hub::HFClientSync;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Run ResNet-18 over Stanford Dogs images")]
struct Args {
    /// Number of dataset images to process before exiting.
    #[arg(long, default_value_t = 10)]
    n_samples: usize,

    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Writable Hugging Face cache directory for model and dataset files.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct ModelOperation<C, M> {
    model: M,
    device: Device,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> ModelOperation<C, M> {
    fn new(model: M, device: Device) -> Self {
        Self {
            model,
            device,
            _contract: std::marker::PhantomData,
        }
    }
}

fn tensor_from_input<C: TensorContract<Metadata = SampleMetadata>>(
    input: &[u8],
    device: &Device,
) -> Result<(Tensor, SampleMetadata), String> {
    let input = decode_input::<C>(input).map_err(|error| error.to_string())?;
    let raw = input
        .tensors
        .first()
        .ok_or_else(|| "operation input has no tensor".to_owned())?;
    if raw.dtype != TensorDtype::F32 {
        return Err(format!("expected f32 input, got {:?}", raw.dtype));
    }
    let values = raw
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, raw.shape.clone(), device)
        .map(|tensor| (tensor, input.metadata))
        .map_err(|error| error.to_string())
}

fn tensor_output<C: TensorContract<Metadata = SampleMetadata>>(
    tensor: Tensor,
    metadata: &SampleMetadata,
) -> Result<Vec<u8>, String> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .flatten_all()
        .map_err(|error| error.to_string())?
        .to_vec1::<f32>()
        .map_err(|error| error.to_string())?;
    encode_output::<C>(
        &[RawTensor {
            name: C::Output::descriptor()[0].name.clone(),
            dtype: TensorDtype::F32,
            shape,
            data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
        }],
        metadata,
    )
    .map_err(|error| error.to_string())
}

macro_rules! operation_impl {
    ($operation:ident, $contract:ty, $model:ty, $forward:expr) => {
        struct $operation(ModelOperation<$contract, $model>);

        #[async_trait]
        impl OperationImplementation for $operation {
            fn capability(&self) -> OperationCapability {
                let mut capability = operation_capability::<$contract>();
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

            async fn forward(
                &self,
                _instance_id: uuid::Uuid,
                input: Vec<u8>,
            ) -> Result<Vec<u8>, String> {
                tracing::info!(operation = stringify!($operation), "starting operation");
                let (tensor, metadata) = tensor_from_input::<$contract>(&input, &self.0.device)?;
                let output =
                    ($forward)(&self.0.model, &tensor).map_err(|error| error.to_string())?;
                tracing::info!(operation = stringify!($operation), "finished operation");
                tensor_output::<$contract>(output, &metadata)
            }

            async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
                Ok(())
            }
            async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
                Ok(())
            }
        }
    };
}

operation_impl!(
    StemOperation,
    StemOp,
    candle_nn::Func<'static>,
    |model: &candle_nn::Func<'static>, xs: &Tensor| model.forward(xs)
);
operation_impl!(
    Stage1Operation,
    Stage1Op,
    candle_nn::Func<'static>,
    |model: &candle_nn::Func<'static>, xs: &Tensor| model.forward(xs)
);
operation_impl!(
    Stage2Operation,
    Stage2Op,
    candle_nn::Func<'static>,
    |model: &candle_nn::Func<'static>, xs: &Tensor| model.forward(xs)
);
operation_impl!(
    Stage3Operation,
    Stage3Op,
    candle_nn::Func<'static>,
    |model: &candle_nn::Func<'static>, xs: &Tensor| model.forward(xs)
);
operation_impl!(
    Stage4Operation,
    Stage4Op,
    candle_nn::Func<'static>,
    |model: &candle_nn::Func<'static>, xs: &Tensor| model.forward(xs)
);
operation_impl!(
    HeadOperation,
    HeadOp,
    candle_nn::Linear,
    |model: &candle_nn::Linear, xs: &Tensor| model.forward(&pool_stage4(xs)?)
);

#[derive(Clone)]
struct CorgiJungle {
    client: FusedClient,
    void: VoidClient,
    stem: MassClient<StemOp>,
    stage1: MassClient<Stage1Op>,
    stage2: MassClient<Stage2Op>,
    stage3: MassClient<Stage3Op>,
    stage4: MassClient<Stage4Op>,
    head: MassClient<HeadOp>,
}

#[derive(Animals)]
struct CorgiAnimals(
    corgi_fwd::StemCell,
    corgi_fwd::Stage1Cell,
    corgi_fwd::Stage2Cell,
    corgi_fwd::Stage3Cell,
    corgi_fwd::Stage4Cell,
    corgi_fwd::HeadCell,
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

#[async_trait]
impl corgi_fwd::RawArtifactOps for CorgiJungle {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String> {
        self.void.receive_artifact(reference).await
    }
}

async fn retag<From, To>(
    void: &VoidClient,
    input: &ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String>
where
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
{
    let bytes = void.receive_artifact(input).await?;
    let decoded = decode_output::<From>(&bytes).map_err(|error| error.to_string())?;
    let tagged = black_hole_sun::encode_input::<To>(&decoded.tensors, &decoded.metadata)
        .map_err(|error| error.to_string())?;
    let id = void.upload(tagged).await?;
    Ok(ArtifactRef::from_object_id(id))
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
        Appearance: serde::de::DeserializeOwned + Send,
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

fn server_address() -> SocketAddr {
    "127.0.0.1:0".parse().expect("valid loopback address")
}

fn model_path(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = argument {
        return Ok(path);
    }
    let api = HFClientSync::new()?;
    let repo = api.model("lmz".to_owned(), "candle-resnet".to_owned());
    Ok(repo
        .download_file()
        .filename("resnet18.safetensors")
        .send()?)
}

fn writable(path: &PathBuf) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    let probe = path.join(".corgi-fwd-write-test");
    fs::write(&probe, [])?;
    fs::remove_file(probe)
}

fn configure_hf_cache(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let local = env::current_dir()?.join("target/corgi-fwd/huggingface");
    let explicit = argument.is_some();
    let requested = argument.or_else(|| env::var_os("HF_HUB_CACHE").map(PathBuf::from));
    if let Some(path) = requested {
        if writable(&path).is_ok() {
            env::set_var("HF_HUB_CACHE", &path);
            return Ok(path);
        }
        if explicit {
            return Err(format!("Hugging Face cache is not writable: {}", path.display()).into());
        }
        eprintln!(
            "ignoring non-writable HF_HUB_CACHE {}; using {}",
            path.display(),
            local.display()
        );
    }
    writable(&local)?;
    env::set_var("HF_HUB_CACHE", &local);
    Ok(local)
}

fn builder(
    path: &PathBuf,
    device: &Device,
) -> Result<VarBuilder<'static>, Box<dyn std::error::Error>> {
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? })
}

fn capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        ..OperationCapabilities::default()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if args.n_samples == 0 {
        return Ok(());
    }
    let cache_dir = configure_hf_cache(args.cache_dir)?;
    eprintln!("using Hugging Face cache {}", cache_dir.display());
    let path = model_path(args.model)?;
    let device = Device::Cpu;

    let (void_addr, void_task) = VoidServerBuilder::new(
        Box::new(object_store::InMemoryObjectStore::new()),
        Box::new(persist::InMemoryStore::new()),
    )
    .tcp()
    .listen(server_address())
    .serve()
    .await?;
    let (stem_addr, stem_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(StemOperation(ModelOperation::new(
            build_stem(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;
    let (stage1_addr, stage1_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage1Operation(ModelOperation::new(
            build_stage1(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;
    let (stage2_addr, stage2_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage2Operation(ModelOperation::new(
            build_stage2(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;
    let (stage3_addr, stage3_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage3Operation(ModelOperation::new(
            build_stage3(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;
    let (stage4_addr, stage4_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage4Operation(ModelOperation::new(
            build_stage4(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;
    let (head_addr, head_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(HeadOperation(ModelOperation::new(
            build_head(builder(&path, &device)?)?,
            device.clone(),
        )))
        .serve()
        .await?;

    let client = FusedClient::builder().build().await?;
    let jungle = CorgiJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(void_addr),
        stem: MassClient::new_tcp_typed(stem_addr).requiring(capabilities()),
        stage1: MassClient::new_tcp_typed(stage1_addr).requiring(capabilities()),
        stage2: MassClient::new_tcp_typed(stage2_addr).requiring(capabilities()),
        stage3: MassClient::new_tcp_typed(stage3_addr).requiring(capabilities()),
        stage4: MassClient::new_tcp_typed(stage4_addr).requiring(capabilities()),
        head: MassClient::new_tcp_typed(head_addr).requiring(capabilities()),
    };
    let _parent = client.spawn::<CorgiForward>(&()).await?;
    let worker_error = Arc::new(Mutex::new(None::<String>));
    let workers = (0..4)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            let worker_error = Arc::clone(&worker_error);
            tokio::spawn(async move {
                if let Err(error) = worker.spawn().await {
                    eprintln!("corgi-fwd worker stopped: {error}");
                    if let Ok(mut slot) = worker_error.lock() {
                        slot.get_or_insert_with(|| error.to_string());
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    let result = loop {
        if LOGGED_OUTPUTS.load(Ordering::Acquire) >= args.n_samples {
            break Ok(());
        }
        if let Some(error) = worker_error.lock().ok().and_then(|error| error.clone()) {
            break Err(error);
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };

    for worker in workers {
        worker.abort();
        let _ = worker.await;
    }
    void_task.abort();
    stem_task.abort();
    stage1_task.abort();
    stage2_task.abort();
    stage3_task.abort();
    stage4_task.abort();
    head_task.abort();
    println!(
        "corgi-fwd processed {} sample(s) (dataset contains {DATASET_SAMPLES})",
        args.n_samples
    );
    result.map_err(Into::into)
}
