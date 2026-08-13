//! Twin effects — quark inference plus fusion stack transforms.

use std::future::Future;
use std::marker::PhantomData;

use black_hole_spec::InferenceOutput;
use jungle_sdk::prelude::*;
use rand::random;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::debug;

pub use crate::atom::effect::*;
use crate::ops::VoidInferOps;
use crate::AtomError;

async fn download_output<J>(
    jungle: &J,
    output_id: InferenceOutputId,
) -> Result<InferenceOutput, AtomError>
where
    J: VoidInferOps,
{
    let output_bytes = jungle
        .download_raw(output_id.0)
        .await
        .map_err(AtomError::Download)?;
    postcard::from_bytes(&output_bytes).map_err(AtomError::from)
}

async fn upload_stacked_emission<J, M>(
    jungle: &J,
    metadata: M,
    output: InferenceOutput,
) -> Result<EmissionId, AtomError>
where
    J: VoidInferOps,
    M: Serialize + DeserializeOwned + Send + 'static,
{
    let output_bytes = postcard::to_allocvec(&output)?;
    let output_id = jungle
        .upload_to_void(output_bytes)
        .await
        .map_err(AtomError::Upload)?;

    let emission = Emission {
        metadata,
        output_id: InferenceOutputId(output_id),
    };
    let emission_bytes = postcard::to_allocvec(&emission)?;
    let emission_id = jungle
        .upload_to_void(emission_bytes)
        .await
        .map_err(AtomError::Upload)?;

    Ok(EmissionId(emission_id))
}

fn stack_dark_tokens(
    base_side: &str,
    stacked_side: &str,
    mut base_output: InferenceOutput,
    stacked_output: InferenceOutput,
) -> Result<InferenceOutput, AtomError> {
    let base_sequences = base_output.results.len();
    let stacked_sequences = stacked_output.results.len();
    if base_sequences != stacked_sequences {
        return Err(AtomError::Inference(format!(
            "twin {base_side} stack requires matching sequence counts: \
             {base_side}={base_sequences}, {stacked_side}={stacked_sequences}"
        )));
    }

    for (base_sequence, stacked_sequence) in
        base_output.results.iter_mut().zip(stacked_output.results)
    {
        base_sequence.0.extend(stacked_sequence.0);
    }

    Ok(base_output)
}

fn total_dark_tokens(output: &InferenceOutput) -> usize {
    output.results.iter().map(|sequence| sequence.0.len()).sum()
}

/// Merge two emissions by appending right-hand dark tokens into each
/// corresponding left-hand sequence.
pub struct LeftStackEffect<M>(PhantomData<fn() -> M>);

impl<M, J> EffectSchema<J> for LeftStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<M, J> Effect<J> for LeftStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.0)
                .await
                .map_err(AtomError::Download)?;

            let left_output = download_output(jungle, left_emission.output_id).await?;
            let right_output = download_output(jungle, right_emission.output_id).await?;
            let left_output = stack_dark_tokens("left", "right", left_output, right_output)?;

            debug!(
                left = %left_id.0,
                right = %right_id.0,
                sequence_count = left_output.results.len(),
                combined_dark_tokens = total_dark_tokens(&left_output),
                "stacked twin emissions with left metadata"
            );

            upload_stacked_emission(jungle, left_emission.metadata, left_output).await
        }
    }
}

/// Merge two emissions by appending left-hand dark tokens into each
/// corresponding right-hand sequence.
pub struct RightStackEffect<M>(PhantomData<fn() -> M>);

impl<M, J> EffectSchema<J> for RightStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<M, J> Effect<J> for RightStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.0)
                .await
                .map_err(AtomError::Download)?;

            let left_output = download_output(jungle, left_emission.output_id).await?;
            let right_output = download_output(jungle, right_emission.output_id).await?;
            let right_output = stack_dark_tokens("right", "left", right_output, left_output)?;

            debug!(
                left = %left_id.0,
                right = %right_id.0,
                sequence_count = right_output.results.len(),
                combined_dark_tokens = total_dark_tokens(&right_output),
                "stacked twin emissions with right metadata"
            );

            upload_stacked_emission(jungle, right_emission.metadata, right_output).await
        }
    }
}

/// Merge two emissions by randomly choosing left- or right-based stacking of
/// per-sequence dark tokens.
pub struct RandStackEffect<M>(PhantomData<fn() -> M>);

impl<M, J> EffectSchema<J> for RandStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<M, J> Effect<J> for RandStackEffect<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.0)
                .await
                .map_err(AtomError::Download)?;

            let choose_left = random::<bool>();
            let mut left_output = download_output(jungle, left_emission.output_id).await?;
            let mut right_output = download_output(jungle, right_emission.output_id).await?;

            if choose_left {
                left_output = stack_dark_tokens("left", "right", left_output, right_output)?;
                debug!(
                    left = %left_id.0,
                    right = %right_id.0,
                    picked = "left",
                    sequence_count = left_output.results.len(),
                    combined_dark_tokens = total_dark_tokens(&left_output),
                    "stacked twin emissions with random-side metadata"
                );
                upload_stacked_emission(jungle, left_emission.metadata, left_output).await
            } else {
                right_output = stack_dark_tokens("right", "left", right_output, left_output)?;
                debug!(
                    left = %left_id.0,
                    right = %right_id.0,
                    picked = "right",
                    sequence_count = right_output.results.len(),
                    combined_dark_tokens = total_dark_tokens(&right_output),
                    "stacked twin emissions with random-side metadata"
                );
                upload_stacked_emission(jungle, right_emission.metadata, right_output).await
            }
        }
    }
}
