use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{env, fs};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::ops::{MassOps, OptimizeOps, PerturbOps, ResetOps, SunOps, VoidInferOps};
use black_hole_sun::{
    decode_input, encode_output, ArtifactRef, EmissionId, MassClient, MassModelParams,
    MassServerBuilder, ObjectId, OperationCapabilities, OperationCapability, OperationConfig,
    OperationImplementation, RawTensor, TensorContract, TensorDtype, TensorSpec, Transmission,
    VoidClient, VoidServerBuilder,
};
use black_hole_sun::{object_store, persist};
use candle::{DType, Device, Tensor, Var};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use clap::Parser;
use corgi_fwd::{HeadOp, SampleMetadata, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use corgi_qzo::{CorgiZo, OPTIMIZED_EPOCHS};
use hf_hub::HFClientSync;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::Serialize;

const DATASET_SAMPLES: usize = corgi_fwd::DATASET_SAMPLES;

#[derive(Debug, Parser)]
#[command(about = "Run two-sided zeroth-order optimization on ResNet-18")]
struct Args {
    /// Number of ZO epochs to run before exiting.
    #[arg(long, default_value_t = 10)]
    epochs: usize,

    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Writable Hugging Face cache directory for model and dataset files.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

struct ZoModel<M> {
    model: M,
    vars: Vec<Var>,
    direction: Option<Vec<Tensor>>,
}

struct ModelOperation<C, M> {
    state: Arc<Mutex<ZoModel<M>>>,
    device: Device,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> ModelOperation<C, M> {
    fn new(model: M, varmap: &VarMap, device: Device) -> Self {
        Self {
            state: Arc::new(Mutex::new(ZoModel {
                model,
                vars: varmap.all_vars(),
                direction: None,
            })),
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

fn next_random(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    if (*state >> 63) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn perturb_up<M>(state: &mut ZoModel<M>, device: &Device, seed: u64) -> Result<(), String> {
    if state.direction.is_some() {
        return Err("perturb up called before completing the previous ZO step".into());
    }
    let mut rng = seed;
    let mut directions = Vec::with_capacity(state.vars.len());
    for var in &state.vars {
        let direction = Tensor::from_vec(
            (0..var.shape().elem_count())
                .map(|_| next_random(&mut rng))
                .collect::<Vec<_>>(),
            var.shape().clone(),
            device,
        )
        .map_err(|error| error.to_string())?;
        let updated = (var.as_tensor()
            + &(&direction * 1e-3).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        var.set(&updated).map_err(|error| error.to_string())?;
        directions.push(direction);
    }
    state.direction = Some(directions);
    Ok(())
}

fn perturb_down<M>(state: &mut ZoModel<M>) -> Result<(), String> {
    let directions = state
        .direction
        .as_ref()
        .ok_or_else(|| "perturb down called without perturb up".to_owned())?;
    for (var, direction) in state.vars.iter().zip(directions) {
        let delta = (direction * 2e-3).map_err(|error| error.to_string())?;
        let updated = (var.as_tensor() - &delta).map_err(|error| error.to_string())?;
        var.set(&updated).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn optimize<M>(state: &mut ZoModel<M>, loss_up: f32, loss_down: f32) -> Result<(), String> {
    let directions = state
        .direction
        .take()
        .ok_or_else(|| "optimize called without perturbation state".to_owned())?;
    // Restore -epsilon to the base point and apply the central-difference
    // update. The clip keeps a single difficult image from dominating a step.
    let derivative = ((loss_up - loss_down) / 2e-3).clamp(-10.0, 10.0);
    for (var, direction) in state.vars.iter().zip(directions) {
        let restore = (direction.clone() * 1e-3).map_err(|error| error.to_string())?;
        let restored = (var.as_tensor() + &restore).map_err(|error| error.to_string())?;
        let update =
            (&direction * (1e-4 * derivative as f64)).map_err(|error| error.to_string())?;
        let updated = (restored - &update).map_err(|error| error.to_string())?;
        var.set(&updated).map_err(|error| error.to_string())?;
    }
    Ok(())
}

macro_rules! operation_impl {
    ($operation:ident, $contract:ty, $model:ty) => {
        struct $operation(ModelOperation<$contract, $model>);

        #[async_trait]
        impl OperationImplementation for $operation {
            fn capability(&self) -> OperationCapability {
                let mut capability = operation_capability::<$contract>();
                capability.operations = OperationCapabilities {
                    forward: true,
                    reset: true,
                    perturb: true,
                    optimize: true,
                    ..OperationCapabilities::default()
                };
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
                let (tensor, metadata) = tensor_from_input::<$contract>(&input, &self.0.device)?;
                let state = self.0.state.lock().map_err(|_| "model lock poisoned")?;
                let output = state
                    .model
                    .forward(&tensor)
                    .map_err(|error| error.to_string())?;
                tensor_output::<$contract>(output, &metadata)
            }

            async fn reset(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
                Ok(())
            }

            async fn shutdown(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
                Ok(())
            }

            async fn perturb_up(&self, _instance_id: uuid::Uuid, seed: u64) -> Result<(), String> {
                let mut state = self.0.state.lock().map_err(|_| "model lock poisoned")?;
                perturb_up(&mut *state, &self.0.device, seed)
            }

            async fn perturb_down(&self, _instance_id: uuid::Uuid) -> Result<(), String> {
                let mut state = self.0.state.lock().map_err(|_| "model lock poisoned")?;
                perturb_down(&mut *state)
            }

            async fn optimize(
                &self,
                _instance_id: uuid::Uuid,
                loss_up: f32,
                loss_down: f32,
            ) -> Result<(), String> {
                let mut state = self.0.state.lock().map_err(|_| "model lock poisoned")?;
                optimize(&mut *state, loss_up, loss_down)
            }
        }
    };
}

operation_impl!(StemOperation, StemOp, candle_nn::Func<'static>);
operation_impl!(Stage1Operation, Stage1Op, candle_nn::Func<'static>);
operation_impl!(Stage2Operation, Stage2Op, candle_nn::Func<'static>);
operation_impl!(Stage3Operation, Stage3Op, candle_nn::Func<'static>);
operation_impl!(Stage4Operation, Stage4Op, candle_nn::Func<'static>);

struct HeadModel(Linear);
impl Module for HeadModel {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        self.0.forward(&corgi_qzo::pool_stage4(xs)?)
    }
}
operation_impl!(HeadOperation, HeadOp, HeadModel);

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
    corgi_qzo::StemCell,
    corgi_qzo::Stage1Cell,
    corgi_qzo::Stage2Cell,
    corgi_qzo::Stage3Cell,
    corgi_qzo::Stage4Cell,
    corgi_qzo::HeadCell,
    CorgiZo,
);

impl Ecosystem for CorgiJungle {
    const NAME: &'static str = "corgi-qzo";
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
        _config: Option<black_hole_sun::MassModelConfig>,
    ) -> Result<(), String> {
        Err("Qwen model lifecycle is not used by corgi-qzo".into())
    }
    async fn infer(
        &self,
        _model_id: uuid::Uuid,
        _request: black_hole_sun::InferenceRequest,
    ) -> Result<ObjectId, String> {
        Err("Qwen inference is not used by corgi-qzo".into())
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
    fn darken(&self, _prompt: &str) -> Result<Vec<black_hole_sun::DarkToken>, String> {
        Err("darkening not used".into())
    }
    fn decode(&self, _tokens: &[black_hole_sun::DarkToken]) -> String {
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

async fn retag<
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
>(
    void: &VoidClient,
    input: &ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String> {
    let bytes = void.receive_artifact(input).await?;
    let decoded =
        black_hole_sun::decode_output::<From>(&bytes).map_err(|error| error.to_string())?;
    let tagged = black_hole_sun::encode_input::<To>(&decoded.tensors, &decoded.metadata)
        .map_err(|error| error.to_string())?;
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
    async fn observe_animal<Appearance: serde::de::DeserializeOwned + Send>(
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

fn server_address() -> SocketAddr {
    "127.0.0.1:0".parse().expect("valid loopback address")
}

fn model_path(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = argument {
        return Ok(path);
    }
    let api = HFClientSync::new()?;
    Ok(api
        .model("lmz", "candle-resnet")
        .download_file()
        .filename("resnet18.safetensors")
        .send()?)
}

fn configure_hf_cache(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let local = env::current_dir()?.join("target/corgi-qzo/huggingface");
    let explicit = argument.is_some();
    let requested = argument.or_else(|| env::var_os("HF_HUB_CACHE").map(PathBuf::from));
    if let Some(path) = requested {
        if fs::create_dir_all(&path).is_ok() && fs::write(path.join(".write-test"), []).is_ok() {
            let _ = fs::remove_file(path.join(".write-test"));
            env::set_var("HF_HUB_CACHE", &path);
            return Ok(path);
        }
        if explicit {
            return Err(format!("Hugging Face cache is not writable: {}", path.display()).into());
        }
    }
    fs::create_dir_all(&local)?;
    env::set_var("HF_HUB_CACHE", &local);
    Ok(local)
}

fn mutable_stage<C, F>(
    path: &PathBuf,
    device: &Device,
    build: F,
) -> Result<ModelOperation<C, candle_nn::Func<'static>>, Box<dyn std::error::Error>>
where
    C: TensorContract<Metadata = SampleMetadata>,
    F: FnOnce(VarBuilder<'_>) -> candle::Result<candle_nn::Func<'static>>,
{
    let mut varmap = VarMap::new();
    let model = build(VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    varmap.load(path)?;
    Ok(ModelOperation::new(model, &varmap, device.clone()))
}

fn mutable_head(
    path: &PathBuf,
    device: &Device,
) -> Result<ModelOperation<HeadOp, HeadModel>, Box<dyn std::error::Error>> {
    let source = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    let original = corgi_qzo::build_head(source)?;
    let mut varmap = VarMap::new();
    let model = candle_nn::linear(512, 2, VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    varmap.set_one("weight", original.weight())?;
    varmap.set_one("bias", original.bias().expect("corgi head has a bias"))?;
    Ok(ModelOperation::new(
        HeadModel(model),
        &varmap,
        device.clone(),
    ))
}

fn capabilities() -> OperationCapabilities {
    OperationCapabilities {
        forward: true,
        reset: true,
        perturb: true,
        optimize: true,
        ..OperationCapabilities::default()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if args.epochs == 0 {
        return Ok(());
    }
    let cache = configure_hf_cache(args.cache_dir)?;
    eprintln!("using Hugging Face cache {}", cache.display());
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
        .operation(StemOperation(mutable_stage(
            &path,
            &device,
            corgi_qzo::build_stem,
        )?))
        .serve()
        .await?;
    let (stage1_addr, stage1_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage1Operation(mutable_stage(
            &path,
            &device,
            corgi_qzo::build_stage1,
        )?))
        .serve()
        .await?;
    let (stage2_addr, stage2_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage2Operation(mutable_stage(
            &path,
            &device,
            corgi_qzo::build_stage2,
        )?))
        .serve()
        .await?;
    let (stage3_addr, stage3_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage3Operation(mutable_stage(
            &path,
            &device,
            corgi_qzo::build_stage3,
        )?))
        .serve()
        .await?;
    let (stage4_addr, stage4_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(Stage4Operation(mutable_stage(
            &path,
            &device,
            corgi_qzo::build_stage4,
        )?))
        .serve()
        .await?;
    let (head_addr, head_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(HeadOperation(mutable_head(&path, &device)?))
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
    client.spawn::<CorgiZo>(&()).await?;
    let worker_error = Arc::new(Mutex::new(None::<String>));
    let workers = (0..8)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            let worker_error = Arc::clone(&worker_error);
            tokio::spawn(async move {
                if let Err(error) = worker.spawn().await {
                    if let Ok(mut slot) = worker_error.lock() {
                        slot.get_or_insert(error.to_string());
                    }
                }
            })
        })
        .collect::<Vec<_>>();

    let result = loop {
        if OPTIMIZED_EPOCHS.load(std::sync::atomic::Ordering::Acquire) >= args.epochs {
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
        "corgi-qzo completed {} epoch(s) (dataset contains {DATASET_SAMPLES})",
        args.epochs
    );
    result.map_err(Into::into)
}
