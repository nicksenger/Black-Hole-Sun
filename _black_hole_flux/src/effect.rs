//! QuarkInfer effect — download → infer → upload in a single effect.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;

use crate::ops::VoidInferOps;

pub use black_hole_spec::{
    Emission, EmissionId, InferenceOutputId, ObjectId,
};

use crate::CellError;

/// Effect that performs one quark-inference step.
///
/// Takes an [`EmissionId`] pointing to an `Emission<M>` in void, downloads it to
/// obtain the `InferenceOutputId`, passes that ID directly to quark inference,
/// wraps the returned output ID into a new `Emission<M>`, uploads it, and returns
/// the new [`EmissionId`].
///
/// The Jungle instance must implement [`VoidInferOps`].
pub struct QuarkInfer<M>(PhantomData<fn() -> M>);

impl<M, J> EffectSchema<J> for QuarkInfer<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Id = u64;
    type In = EmissionId;
    type Out = EmissionId;
    type Err = CellError;
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

            // 1. Download the emission to get the output_id and metadata.
            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(CellError::Download)?;
            let input_output_id = emission.output_id.0;
            debug!(emission_id = %obj_id, "downloaded emission for inference");

            // 2. Run quark inference, passing the InferenceOutputId directly.
            let output_id = jungle
                .infer(input_output_id)
                .await
                .map_err(CellError::Inference)?;
            debug!(output_id = %output_id, "quark inference complete");

            // 3. Wrap the output ID into a new Emission<M> (preserving metadata) and upload.
            let output_emission = Emission {
                metadata: emission.metadata,
                output_id: InferenceOutputId(output_id),
            };
            let result_bytes = postcard::to_allocvec(&output_emission)?;
            let result_id = jungle
                .upload_to_void(result_bytes)
                .await
                .map_err(CellError::Upload)?;
            debug!(result_id = %result_id, "uploaded inference result emission");

            Ok(EmissionId(result_id))
        }
    }
}
