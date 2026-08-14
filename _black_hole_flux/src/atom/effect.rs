//! Atom effects — quark inference.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;
use uuid::Uuid;

pub use black_hole_spec::{Emission, EmissionId, InferenceOutputId, InferenceRequest};

use crate::model_config::DefaultConfig;
use crate::ops::VoidInferOps;
use crate::AtomError;

// ---------------------------------------------------------------------------
// QuarkInfer — download -> infer -> upload in a single effect
// ---------------------------------------------------------------------------

/// Effect that performs one quark-inference step.
pub struct QuarkInfer<M, H = DefaultConfig>(PhantomData<fn() -> (M, H)>);

#[jungle::effect(id = 8)]
impl<M: Serialize + DeserializeOwned + Send + 'static, H, J: VoidInferOps> Effect<J>
    for QuarkInfer<M, H>
{
    type In = (Uuid, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (model_id, input_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let obj_id = input_id.0;

            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(AtomError::Download)?;
            let input_output_id = emission.output_id.0;
            debug!(emission_id = %obj_id, "downloaded emission for inference");

            let request = InferenceRequest::VoidId {
                id: InferenceOutputId(input_output_id),
                limit: None,
            };
            let output_id = jungle
                .infer(model_id, request)
                .await
                .map_err(AtomError::Inference)?;
            debug!(%model_id, output_id = %output_id, "quark inference complete");

            jungle
                .reset_model(model_id)
                .await
                .map_err(AtomError::ModelReset)?;
            debug!(%model_id, "quark model reset complete");

            let output_emission = Emission {
                metadata: emission.metadata,
                output_id: InferenceOutputId(output_id),
            };
            let result_bytes = postcard::to_allocvec(&output_emission)?;
            let result_id = jungle
                .upload_to_void(result_bytes)
                .await
                .map_err(AtomError::Upload)?;
            debug!(result_id = %result_id, "uploaded inference result emission");

            Ok(EmissionId(result_id))
        }
    }
}
