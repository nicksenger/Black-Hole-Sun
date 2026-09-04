//! Server-side ZO operations: forward plus perturb/optimize on shared vars.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::{
    decode_input, encode_output, OperationCapabilities, OperationCapability, OperationConfig,
    OperationImplementation, TensorContract,
};
use candle::{Device, Tensor, Var};
use candle_nn::{Linear, Module, VarMap};
use corgi_fwd::model::pool_stage4;
use toys_common::dataset::SampleMetadata;

use corgi_fwd::contracts::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};

pub struct ZoModel<M> {
    pub model: M,
    pub vars: Vec<Var>,
    pub direction: Option<Vec<Tensor>>,
}

pub struct ModelOperation<C, M> {
    pub state: Arc<Mutex<ZoModel<M>>>,
    pub device: Device,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> ModelOperation<C, M> {
    pub fn new(model: M, varmap: &VarMap, device: Device) -> Self {
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
    let decoded = decode_input::<C>(input)?;
    let raw = decoded.first_tensor()?;
    let values = raw.to_f32()?;
    Tensor::from_vec(values, raw.shape.clone(), device)
        .map(|tensor| (tensor, decoded.metadata))
        .map_err(|error| error.to_string())
}

fn tensor_output<C: TensorContract<Metadata = SampleMetadata>>(
    tensor: Tensor,
    metadata: &SampleMetadata,
) -> Result<Vec<u8>, String> {
    let shape = tensor.dims().to_vec();
    let values = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|error| error.to_string())?;
    Ok(encode_output::<C>(&[C::output_f32(&shape, values)], metadata)?)
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
        pub struct $operation(pub ModelOperation<$contract, $model>);

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

pub struct HeadModel(pub Linear);
impl Module for HeadModel {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        self.0.forward(&pool_stage4(xs)?)
    }
}
operation_impl!(HeadOperation, HeadOp, HeadModel);
