//! Nucleus effects — quark inference.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;

pub use black_hole_spec::{Emission, EmissionId, InferenceOutputId, InferenceRequest};

use crate::ops::VoidInferOps;
use crate::NucleusError;

// ---------------------------------------------------------------------------
// QuarkInfer — download -> infer -> upload in a single effect
// ---------------------------------------------------------------------------

/// Effect that performs one quark-inference step.
pub struct QuarkInfer<M>(PhantomData<fn() -> M>);

impl<M, J> EffectSchema<J> for QuarkInfer<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Id = u64;
    type In = EmissionId;
    type Out = EmissionId;
    type Err = NucleusError;
}

impl<M, J> Effect<J> for QuarkInfer<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        input_id: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let obj_id = input_id.0;

            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(NucleusError::Download)?;
            let input_output_id = emission.output_id.0;
            debug!(emission_id = %obj_id, "downloaded emission for inference");

            let request = InferenceRequest::VoidId {
                id: InferenceOutputId(input_output_id),
                limit: 256,
            };
            let output_id = jungle
                .infer(request)
                .await
                .map_err(NucleusError::Inference)?;
            debug!(output_id = %output_id, "quark inference complete");

            let output_emission = Emission {
                metadata: emission.metadata,
                output_id: InferenceOutputId(output_id),
            };
            let result_bytes = postcard::to_allocvec(&output_emission)?;
            let result_id = jungle
                .upload_to_void(result_bytes)
                .await
                .map_err(NucleusError::Upload)?;
            debug!(result_id = %result_id, "uploaded inference result emission");

            Ok(EmissionId(result_id))
        }
    }
}
