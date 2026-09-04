#![allow(clippy::manual_async_fn)]

use std::collections::VecDeque;
use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::{backward_operation_capability, BackwardContract};
use black_hole_sun::ops::{BackwardOps, MassOps, StepOps, SunOps, VoidOps};
use black_hole_sun::{
    decode_input, decode_input_gradient, decode_output_gradient, encode_input_gradient,
    encode_output, ArtifactRef, MassClient, MassServerBuilder, ObjectId, OperationCapabilities,
    OperationCapability, OperationConfig, OperationImplementation, RawTensor, TensorContract,
    TensorDtype, TensorSpec, VoidClient, VoidServerBuilder,
};
use black_hole_sun::{object_store, persist};
use candle::backprop::GradStore;
use candle::{DType, Device, Tensor, Var, D};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap, SGD};
use clap::Parser;
use corgi_bwd::{
    build_stage1, build_stage2, build_stage3, build_stage4, build_stem, pool_stage4, CorgiBackward,
    HeadOp, SampleMetadata, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp, COMPLETED_EPOCHS,
    DATASET_SAMPLES,
};
use hf_hub::HFClientSync;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::{FusedClient, JungleClient};
use serde::{Deserialize, Serialize};

const MICRO_BATCHES: usize = 8;

#[derive(Debug, Parser)]
#[command(about = "Train a pipeline-parallel ResNet-18 corgi identifier")]
struct Args {
    /// Number of optimizer steps (each consumes eight image micro-batches).
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    /// Learning rate used independently by every stage.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    /// Optional local Candle ResNet-18 safetensors checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Writable Hugging Face cache directory.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerConfig {
    learning_rate: f64,
}

struct CachedForward {
    input: Var,
    output: Tensor,
    metadata: SampleMetadata,
}

struct TrainOperation<C, M> {
    model: M,
    device: Device,
    pending: Mutex<VecDeque<CachedForward>>,
    gradients: Mutex<GradStore>,
    optimizer: Mutex<SGD>,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> TrainOperation<C, M> {
    fn new(
        model: M,
        device: Device,
        variables: Vec<Var>,
        learning_rate: f64,
    ) -> candle::Result<Self> {
        Ok(Self {
            model,
            device,
            pending: Mutex::new(VecDeque::new()),
            gradients: Mutex::new(GradStore::default()),
            optimizer: Mutex::new(SGD::new(variables, learning_rate)?),
            _contract: std::marker::PhantomData,
        })
    }
}

fn raw_tensor(raw: &RawTensor, device: &Device) -> Result<Tensor, String> {
    if raw.dtype != TensorDtype::F32 {
        return Err(format!("expected f32 tensor, got {:?}", raw.dtype));
    }
    let values = raw
        .data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, raw.shape.clone(), device).map_err(|e| e.to_string())
}

fn raw_f32(name: String, tensor: &Tensor) -> Result<RawTensor, String> {
    let values = tensor
        .flatten_all()
        .map_err(|e| e.to_string())?
        .to_vec1::<f32>()
        .map_err(|e| e.to_string())?;
    Ok(RawTensor {
        name,
        dtype: TensorDtype::F32,
        shape: tensor.dims().to_vec(),
        data: values.into_iter().flat_map(f32::to_le_bytes).collect(),
    })
}

fn capability<C: BackwardContract>() -> OperationCapability {
    let mut capability = backward_operation_capability::<C>();
    capability.operations = OperationCapabilities {
        forward: true,
        backward: true,
        step: true,
        ..OperationCapabilities::default()
    };
    capability
}

fn start_config(config: Option<&OperationConfig>) -> Result<Option<OptimizerConfig>, String> {
    config
        .map(|config| postcard::from_bytes(&config.data).map_err(|e| e.to_string()))
        .transpose()
}

fn forward<C, M>(operation: &TrainOperation<C, M>, input: Vec<u8>) -> Result<Vec<u8>, String>
where
    C: BackwardContract<Metadata = SampleMetadata>,
    M: Module,
{
    let decoded = decode_input::<C>(&input).map_err(|e| e.to_string())?;
    let input = Var::from_tensor(&raw_tensor(&decoded.tensors[0], &operation.device)?)
        .map_err(|e| e.to_string())?;
    let output = operation
        .model
        .forward(input.as_tensor())
        .map_err(|e| e.to_string())?;
    let bytes = encode_output::<C>(
        &[raw_f32(C::Output::descriptor()[0].name.clone(), &output)?],
        &decoded.metadata,
    )
    .map_err(|e| e.to_string())?;
    operation
        .pending
        .lock()
        .map_err(|_| "forward cache poisoned".to_owned())?
        .push_back(CachedForward {
            input,
            output,
            metadata: decoded.metadata,
        });
    Ok(bytes)
}

fn backward<C, M>(
    operation: &TrainOperation<C, M>,
    gradient: Vec<u8>,
    is_head: bool,
) -> Result<Vec<u8>, String>
where
    C: BackwardContract<Metadata = SampleMetadata>,
    M: Module,
{
    let decoded = decode_output_gradient::<C>(&gradient).map_err(|e| e.to_string())?;
    let cached = operation
        .pending
        .lock()
        .map_err(|_| "forward cache poisoned".to_owned())?
        .pop_front()
        .ok_or_else(|| "backward arrived without a cached forward".to_owned())?;
    let objective = if is_head {
        let target = usize::from(!matches!(cached.metadata.dataset_label, 111 | 112));
        let log_probs =
            candle_nn::ops::log_softmax(&cached.output, D::Minus1).map_err(|e| e.to_string())?;
        log_probs
            .narrow(1, target, 1)
            .map_err(|e| e.to_string())?
            .neg()
            .map_err(|e| e.to_string())?
            .sum_all()
            .map_err(|e| e.to_string())?
    } else {
        let grad = raw_tensor(&decoded.tensors[0], &operation.device)?;
        cached
            .output
            .mul(&grad)
            .map_err(|e| e.to_string())?
            .sum_all()
            .map_err(|e| e.to_string())?
    };
    let grads = objective.backward().map_err(|e| e.to_string())?;
    let input_grad = grads
        .get(cached.input.as_tensor())
        .cloned()
        .ok_or_else(|| "backward did not produce an input gradient".to_owned())?;
    operation
        .gradients
        .lock()
        .map_err(|_| "gradient accumulator poisoned".to_owned())?
        .extend(grads)
        .map_err(|e| e.to_string())?;
    encode_input_gradient::<C>(
        &[raw_f32(
            C::InputGrad::descriptor()[0].name.clone(),
            &input_grad,
        )?],
        &cached.metadata,
    )
    .map_err(|e| e.to_string())
}

fn step<C, M>(operation: &TrainOperation<C, M>) -> Result<(), String> {
    if !operation
        .pending
        .lock()
        .map_err(|_| "forward cache poisoned".to_owned())?
        .is_empty()
    {
        return Err("step arrived while forward graphs were still cached".into());
    }
    let mut gradients = operation
        .gradients
        .lock()
        .map_err(|_| "gradient accumulator poisoned".to_owned())?;
    operation
        .optimizer
        .lock()
        .map_err(|_| "optimizer poisoned".to_owned())?
        .step(&gradients)
        .map_err(|e| e.to_string())?;
    *gradients = GradStore::default();
    Ok(())
}

macro_rules! operation_impl {
    ($name:ident, $contract:ty, $model:ty, $head:expr) => {
        struct $name(TrainOperation<$contract, $model>);
        #[async_trait]
        impl OperationImplementation for $name {
            fn capability(&self) -> OperationCapability {
                capability::<$contract>()
            }
            async fn start(
                &self,
                _id: uuid::Uuid,
                config: Option<&OperationConfig>,
            ) -> Result<(), String> {
                if let Some(config) = start_config(config)? {
                    self.0
                        .optimizer
                        .lock()
                        .map_err(|_| "optimizer poisoned".to_owned())?
                        .set_learning_rate(config.learning_rate);
                }
                Ok(())
            }
            async fn forward(&self, _id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
                forward::<$contract, _>(&self.0, input)
            }
            async fn backward(&self, _id: uuid::Uuid, grad: Vec<u8>) -> Result<Vec<u8>, String> {
                backward::<$contract, _>(&self.0, grad, $head)
            }
            async fn step(&self, _id: uuid::Uuid) -> Result<(), String> {
                step(&self.0)
            }
            async fn shutdown(&self, _id: uuid::Uuid) -> Result<(), String> {
                Ok(())
            }
        }
    };
}

operation_impl!(StemOperation, StemOp, candle_nn::Func<'static>, false);
operation_impl!(Stage1Operation, Stage1Op, candle_nn::Func<'static>, false);
operation_impl!(Stage2Operation, Stage2Op, candle_nn::Func<'static>, false);
operation_impl!(Stage3Operation, Stage3Op, candle_nn::Func<'static>, false);
operation_impl!(Stage4Operation, Stage4Op, candle_nn::Func<'static>, false);
operation_impl!(HeadOperation, HeadOp, candle_nn::Func<'static>, true);

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
    optimizer_config: OperationConfig,
}

#[derive(Animals)]
struct CorgiAnimals(
    corgi_bwd::StemCell,
    corgi_bwd::Stage1Cell,
    corgi_bwd::Stage2Cell,
    corgi_bwd::Stage3Cell,
    corgi_bwd::Stage4Cell,
    corgi_bwd::HeadCell,
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

#[async_trait]
impl corgi_bwd::RawArtifactOps for CorgiJungle {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String> {
        self.void.receive_artifact(reference).await
    }
}

async fn retag_forward<From, To>(
    void: &VoidClient,
    input: ArtifactRef<To::Input>,
) -> Result<ArtifactRef<To::Input>, String>
where
    From: TensorContract<Metadata = SampleMetadata>,
    To: TensorContract<Metadata = SampleMetadata>,
{
    let bytes = void.receive_artifact(&input).await?;
    let decoded = black_hole_sun::decode_output::<From>(&bytes).map_err(|e| e.to_string())?;
    let bytes = black_hole_sun::encode_input::<To>(&decoded.tensors, &decoded.metadata)
        .map_err(|e| e.to_string())?;
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
    let decoded = decode_input_gradient::<From>(&bytes).map_err(|e| e.to_string())?;
    let bytes = black_hole_sun::encode_output_gradient::<To>(&decoded.tensors, &decoded.metadata)
        .map_err(|e| e.to_string())?;
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
        Err("perturbation is not used by corgi-bwd".into())
    }
}

fn server_address() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn model_path(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = argument {
        return Ok(path);
    }
    let api = HFClientSync::new()?;
    Ok(api
        .model("lmz".to_owned(), "candle-resnet".to_owned())
        .download_file()
        .filename("resnet18.safetensors")
        .send()?)
}

fn configure_hf_cache(argument: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = argument
        .or_else(|| env::var_os("HF_HUB_CACHE").map(PathBuf::from))
        .unwrap_or(env::current_dir()?.join("target/corgi-bwd/huggingface"));
    std::fs::create_dir_all(&path)?;
    env::set_var("HF_HUB_CACHE", &path);
    Ok(path)
}

fn build_trainable<M>(
    path: &Path,
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, Vec<Var>)> {
    let mut variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    variables.load(path)?;
    Ok((model, variables.all_vars()))
}

fn build_random<M>(
    device: &Device,
    build: impl FnOnce(VarBuilder<'_>) -> candle::Result<M>,
) -> candle::Result<(M, Vec<Var>)> {
    let variables = VarMap::new();
    let model = build(VarBuilder::from_varmap(&variables, DType::F32, device))?;
    Ok((model, variables.all_vars()))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    black_hole_sun::init_tracing()?;
    let args = Args::parse();
    if args.epochs == 0 {
        return Ok(());
    }
    configure_hf_cache(args.cache_dir)?;
    let path = model_path(args.model)?;
    let device = Device::Cpu;
    let optimizer_config = OperationConfig {
        encoding: black_hole_sun::EncodingId::POSTCARD_V1,
        data: postcard::to_allocvec(&OptimizerConfig {
            learning_rate: args.learning_rate,
        })?,
    };

    let (void_addr, void_task) = VoidServerBuilder::new(
        Box::new(object_store::InMemoryObjectStore::new()),
        Box::new(persist::InMemoryStore::new()),
    )
    .tcp()
    .listen(server_address())
    .serve()
    .await?;
    macro_rules! serve_stage {
        ($build:expr, $wrapper:ident) => {{
            let (model, vars) = build_trainable(&path, &device, $build)?;
            MassServerBuilder::new("unused")
                .tcp()
                .listen(server_address())
                .void_addr(void_addr)
                .operation($wrapper(TrainOperation::new(
                    model,
                    device.clone(),
                    vars,
                    args.learning_rate,
                )?))
                .serve()
                .await?
        }};
    }
    let (stem_addr, stem_task) = serve_stage!(build_stem, StemOperation);
    let (stage1_addr, stage1_task) = serve_stage!(build_stage1, Stage1Operation);
    let (stage2_addr, stage2_task) = serve_stage!(build_stage2, Stage2Operation);
    let (stage3_addr, stage3_task) = serve_stage!(build_stage3, Stage3Operation);
    let (stage4_addr, stage4_task) = serve_stage!(build_stage4, Stage4Operation);
    let (head_model, head_vars) = build_random(&device, |vb| {
        let linear = candle_nn::linear(512, 2, vb.pp("binary_head"))?;
        Ok(candle_nn::Func::new(move |xs| {
            linear.forward(&pool_stage4(xs)?)
        }))
    })?;
    let (head_addr, head_task) = MassServerBuilder::new("unused")
        .tcp()
        .listen(server_address())
        .void_addr(void_addr)
        .operation(HeadOperation(TrainOperation::new(
            head_model,
            device.clone(),
            head_vars,
            args.learning_rate,
        )?))
        .serve()
        .await?;

    let required = OperationCapabilities {
        forward: true,
        backward: true,
        step: true,
        ..Default::default()
    };
    let client = FusedClient::builder().build().await?;
    let jungle = CorgiJungle {
        client: client.clone(),
        void: VoidClient::new_tcp(void_addr),
        stem: MassClient::new_tcp_typed(stem_addr).requiring(required),
        stage1: MassClient::new_tcp_typed(stage1_addr).requiring(required),
        stage2: MassClient::new_tcp_typed(stage2_addr).requiring(required),
        stage3: MassClient::new_tcp_typed(stage3_addr).requiring(required),
        stage4: MassClient::new_tcp_typed(stage4_addr).requiring(required),
        head: MassClient::new_tcp_typed(head_addr).requiring(required),
        optimizer_config,
    };
    let _parent = client.spawn::<CorgiBackward<MICRO_BATCHES>>(&()).await?;
    let worker_error = Arc::new(Mutex::new(None::<String>));
    let workers = (0..8)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            let error = Arc::clone(&worker_error);
            tokio::spawn(async move {
                if let Err(e) = worker.spawn().await {
                    if let Ok(mut slot) = error.lock() {
                        slot.get_or_insert_with(|| e.to_string());
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let result = loop {
        if COMPLETED_EPOCHS.load(Ordering::Acquire) >= args.epochs {
            break Ok(());
        }
        if let Some(error) = worker_error.lock().ok().and_then(|e| e.clone()) {
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
    println!("corgi-bwd completed {} optimizer step(s), {} micro-batches each (dataset contains {DATASET_SAMPLES})", args.epochs, MICRO_BATCHES);
    result.map_err(Into::into)
}
