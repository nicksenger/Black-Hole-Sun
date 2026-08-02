//! QuarkInfer effect — download → infer → upload in a single effect.

use std::future::Future;
use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;

use crate::ops::VoidInferOps;

pub use black_hole_spec::{
    Emission, EmissionId, InferenceInput, InferenceOutput, InferenceRequest, ObjectId,
};

use crate::CellError;

/// Effect that performs one quark-inference step.
///
/// Takes an [`EmissionId`] pointing to an `Emission<M>` in void, downloads it,
/// runs inference on the contained sequences via quark, uploads the resulting
/// `InferenceOutput` (wrapped as a new `Emission<M>`), and returns the new
/// [`EmissionId`].
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

            // 1. Download the emission from void.
            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(CellError::Download)?;
            debug!(
                emission_id = %obj_id,
                seq_count = emission.sequences.len(),
                "downloaded emission for inference"
            );

            // 2. Build an InferenceRequest from the emission sequences and upload it.
            let request = InferenceRequest {
                sequences: emission.sequences,
                limit: 256,
            };
            let request_bytes = postcard::to_allocvec(&request)?;
            let request_id = jungle
                .upload_to_void(request_bytes)
                .await
                .map_err(CellError::Upload)?;
            debug!(request_id = %request_id, "uploaded inference request");

            // 3. Run quark inference.
            let output_id = jungle
                .infer(request_id)
                .await
                .map_err(CellError::Inference)?;
            debug!(output_id = %output_id, "quark inference complete");

            // 4. Download the InferenceOutput from void.
            let output_bytes = jungle
                .download_raw(output_id)
                .await
                .map_err(CellError::Download)?;
            let inference_output: InferenceOutput = postcard::from_bytes(&output_bytes)?;

            // 5. Wrap as a new Emission<M> (preserving metadata) and upload.
            let output_emission = Emission {
                metadata: emission.metadata,
                sequences: inference_output
                    .results
                    .into_iter()
                    .map(|seq| {
                        seq.predictions
                            .into_iter()
                            .map(|tok| InferenceInput::Tokens(vec![tok.token_id]))
                            .collect()
                    })
                    .collect(),
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
