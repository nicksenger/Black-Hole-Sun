//! Server-side tensor operations wrapping the ResNet-18 stages.

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::operation_capability;
use black_hole_sun::{
    decode_input, encode_output, OperationCapability, OperationConfig, OperationImplementation,
    TensorContract,
};
use candle::{Device, Tensor};
use candle_nn::Module;
use toy_common::dataset::SampleMetadata;

use crate::spec::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use crate::model::pool_stage4;

#[derive(Clone)]
pub struct ModelOperation<C, M> {
    pub model: M,
    pub device: Device,
    _contract: std::marker::PhantomData<C>,
}

impl<C, M> ModelOperation<C, M> {
    pub fn new(model: M, device: Device) -> Self {
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

macro_rules! operation_impl {
    ($operation:ident, $contract:ty, $model:ty, $forward:expr) => {
        pub struct $operation(pub ModelOperation<$contract, $model>);

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
