//! Atom effects — mass inference.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;
use uuid::Uuid;

pub use black_hole_spec::{Emission, EmissionId, InferenceOutputId, InferenceRequest};

use crate::mass::DefaultConfig;
use crate::ops::VoidInferOps;
use crate::ops::{MassOps, ResetOps, VoidOps};
use crate::AtomError;
use black_hole_contract::TensorContract;

// ---------------------------------------------------------------------------
// MassInfer — download -> infer -> upload in a single effect
// ---------------------------------------------------------------------------

/// Effect that performs one mass-inference step.
pub struct MassInfer<M, H = DefaultConfig>(PhantomData<fn() -> (M, H)>);

/// Operation-typed inference effect used by generic tensor nodes.
///
/// The legacy [`MassInfer`] above remains the Qwen adapter. This effect keeps
/// the operation's input and output bundle types on the emission IDs all the
/// way through persistence and forwarding.
pub struct OperationMassInfer<M, Op>(PhantomData<fn() -> (M, Op)>);

#[jungle::effect(id = 73)]
impl<M, Op, J> Effect<J> for OperationMassInfer<M, Op>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: TensorContract + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
    J: VoidOps + MassOps<Op> + ResetOps<Op>,
{
    type In = (Uuid, black_hole_spec::EmissionId<Op::Input>);
    type Out = black_hole_spec::EmissionId<Op::Output>;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (instance_id, input_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let bytes = VoidOps::download_raw(jungle, input_id.id())
                .await
                .map_err(AtomError::Download)?;
            let emission: black_hole_spec::Emission<M, Op::Input> = postcard::from_bytes(&bytes)?;
            let output = MassOps::<Op>::forward(jungle, instance_id, emission.output_id)
                .await
                .map_err(AtomError::Inference)?;
            ResetOps::<Op>::reset_operation(jungle, instance_id)
                .await
                .map_err(AtomError::ModelReset)?;

            let output_emission = black_hole_spec::Emission::<M, Op::Output> {
                metadata: emission.metadata,
                output_id: output,
            };
            let bytes = postcard::to_allocvec(&output_emission)?;
            let id = VoidOps::upload_to_void(jungle, bytes)
                .await
                .map_err(AtomError::Upload)?;
            Ok(black_hole_spec::EmissionId::new(id))
        }
    }
}

#[jungle::effect(id = 58)]
impl<M: Serialize + DeserializeOwned + Send + 'static, H, J: VoidInferOps> Effect<J>
    for MassInfer<M, H>
{
    type In = (Uuid, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (model_id, input_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let obj_id = input_id.id();

            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(AtomError::Download)?;
            let input_output_id = emission.output_id.object_id();
            debug!(emission_id = %obj_id, "downloaded emission for inference");

            let request = InferenceRequest::VoidId {
                id: InferenceOutputId::new(input_output_id),
                limit: None,
            };
            let output_id = jungle
                .infer(model_id, request)
                .await
                .map_err(AtomError::Inference)?;
            debug!(%model_id, output_id = %output_id, "mass inference complete");

            jungle
                .reset_model(model_id)
                .await
                .map_err(AtomError::ModelReset)?;
            debug!(%model_id, "mass model reset complete");

            let output_emission = Emission {
                metadata: emission.metadata,
                output_id: InferenceOutputId::new(output_id).into(),
            };
            let result_bytes = postcard::to_allocvec(&output_emission)?;
            let result_id = jungle
                .upload_to_void(result_bytes)
                .await
                .map_err(AtomError::Upload)?;
            debug!(result_id = %result_id, "uploaded inference result emission");

            Ok(EmissionId::new(result_id))
        }
    }
}
