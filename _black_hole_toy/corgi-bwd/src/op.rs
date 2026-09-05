//! Server-side training operations: cached forward, backward, and step.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::{backward_operation_capability, BackwardContract};
use black_hole_sun::{
    decode_input, decode_output_gradient, encode_input_gradient, encode_output,
    OperationCapabilities, OperationCapability, OperationConfig, OperationImplementation,
    RawTensor,
};
use candle::backprop::GradStore;
use candle::{Device, Tensor, Var};
use candle_nn::{Module, Optimizer, SGD};
use serde::{Deserialize, Serialize};
use toy_common::dataset::{SampleMetadata, BATCH_SIZE};

use corgi_fwd::spec::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
}

struct CachedForward {
    input: Var,
    output: Tensor,
    metadata: SampleMetadata,
}

pub struct TrainOperation<C, M> {
    pub model: M,
    pub device: Device,
    varmap: candle_nn::VarMap,
    pending: Mutex<VecDeque<CachedForward>>,
    gradients: Mutex<GradStore>,
    optimizer: Mutex<SGD>,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> TrainOperation<C, M> {
    pub fn new(
        model: M,
        device: Device,
        varmap: candle_nn::VarMap,
        learning_rate: f64,
    ) -> candle::Result<Self> {
        let variables = varmap.all_vars();
        let optimizer = SGD::new(variables.clone(), learning_rate)?;
        Ok(Self {
            model,
            device,
            varmap,
            pending: Mutex::new(VecDeque::new()),
            gradients: Mutex::new(GradStore::default()),
            optimizer: Mutex::new(optimizer),
            _contract: std::marker::PhantomData,
        })
    }
}

fn tensor_from_raw(raw: &RawTensor, device: &Device) -> Result<Tensor, String> {
    let values = raw.to_f32()?;
    Tensor::from_vec(values, raw.shape.clone(), device).map_err(|e| e.to_string())
}

fn capability<C: BackwardContract>() -> OperationCapability {
    let mut capability = backward_operation_capability::<C>();
    capability.operations = OperationCapabilities {
        forward: true,
        backward: true,
        step: true,
        checkpoint: true,
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
    let decoded = decode_input::<C>(&input)?;
    let input = Var::from_tensor(&tensor_from_raw(
        decoded.first_tensor()?,
        &operation.device,
    )?)
    .map_err(|e| e.to_string())?;
    let output = operation
        .model
        .forward(input.as_tensor())
        .map_err(|e| e.to_string())?;
    let shape = output.dims().to_vec();
    let values = output
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let bytes = encode_output::<C>(&[C::output_f32(&shape, values)], &decoded.metadata)?;
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
    let decoded = decode_output_gradient::<C>(&gradient)?;
    let cached = operation
        .pending
        .lock()
        .map_err(|_| "forward cache poisoned".to_owned())?
        .pop_front()
        .ok_or_else(|| "backward arrived without a cached forward".to_owned())?;
    let objective = if is_head {
        let log_probs = candle_nn::ops::log_softmax(&cached.output, candle::D::Minus1)
            .map_err(|e| e.to_string())?;
        let mut objective: Option<Tensor> = None;
        for (index, label) in cached.metadata.dataset_labels.iter().enumerate() {
            let target = usize::from(!matches!(
                *label,
                corgi_fwd::PEMBROKE_LABEL | corgi_fwd::CARDIGAN_LABEL
            ));
            let loss = log_probs
                .narrow(0, index, 1)
                .map_err(|e| e.to_string())?
                .narrow(1, target, 1)
                .map_err(|e| e.to_string())?
                .neg()
                .map_err(|e| e.to_string())?
                .sum_all()
                .map_err(|e| e.to_string())?;
            objective = Some(match objective {
                Some(total) => (total + loss).map_err(|e| e.to_string())?,
                None => loss,
            });
        }
        objective
            .ok_or_else(|| "batch has no labels".to_owned())?
            .affine(1.0 / BATCH_SIZE as f64, 0.0)
            .map_err(|e| e.to_string())?
    } else {
        let grad = tensor_from_raw(decoded.first_tensor()?, &operation.device)?;
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
    let shape = input_grad.dims().to_vec();
    let values = input_grad
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    Ok(encode_input_gradient::<C>(
        &[C::input_grad_f32(&shape, values)],
        &cached.metadata,
    )?)
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

fn checkpoint<C, M>(operation: &TrainOperation<C, M>, id: uuid::Uuid) -> Result<Vec<u8>, String> {
    let _gradients = operation
        .gradients
        .lock()
        .map_err(|_| "gradient accumulator poisoned")?;
    let path = std::env::temp_dir().join(format!("corgi-bwd-checkpoint-{id}.safetensors"));
    operation
        .varmap
        .save(&path)
        .map_err(|error| format!("save checkpoint: {error}"))?;
    let bytes = std::fs::read(&path).map_err(|error| format!("read checkpoint: {error}"));
    let _ = std::fs::remove_file(path);
    bytes
}

/// Reconstruct one unified model checkpoint from the trained stage shards.
pub fn unify_checkpoint_shards(directory: &Path, step: usize) -> Result<bool, String> {
    let prefix = format!("step-{step}-");
    let mut paths = std::fs::read_dir(directory)
        .map_err(|error| format!("read checkpoint directory: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".checkpoint"))
        })
        .collect::<Vec<_>>();
    if paths.len() < 6 {
        return Ok(false);
    }
    paths.sort();

    let mut tensors = std::collections::HashMap::new();
    for path in &paths {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read checkpoint {}: {error}", path.display()))?;
        let shard = candle::safetensors::load_buffer(&bytes, &Device::Cpu)
            .map_err(|error| format!("decode checkpoint {}: {error}", path.display()))?;
        for (name, tensor) in shard {
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(format!("duplicate tensor {name} in checkpoint shards"));
            }
        }
    }

    let output = directory.join(format!("step-{step}.checkpoint"));
    let temporary = directory.join(format!("step-{step}.checkpoint.tmp"));
    candle::safetensors::save(&tensors, &temporary)
        .map_err(|error| format!("write unified checkpoint: {error}"))?;
    std::fs::rename(&temporary, &output)
        .map_err(|error| format!("publish unified checkpoint: {error}"))?;
    for path in paths {
        std::fs::remove_file(&path)
            .map_err(|error| format!("remove checkpoint {}: {error}", path.display()))?;
    }
    Ok(true)
}

macro_rules! operation_impl {
    ($name:ident, $contract:ty, $model:ty, $head:expr) => {
        pub struct $name(pub TrainOperation<$contract, $model>);
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
            async fn checkpoint(&self, id: uuid::Uuid) -> Result<Vec<u8>, String> {
                checkpoint(&self.0, id)
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
