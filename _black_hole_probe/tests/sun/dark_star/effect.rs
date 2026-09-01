use std::future::Future;
use std::marker::PhantomData;

use black_hole_sun::ops::{InferenceOutputOps, TransmissionOps, VoidInferOps};
use black_hole_sun::{
    ArtifactRef, AtomError, Emission, InferenceOutput, InferenceOutputId, SequenceOutput,
};
use postcard::to_allocvec;
use rand::random;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tracing::{debug, info};

use super::*;

const BLACK_DWARF_BATCH_SIZE: usize = 8;

#[jungle::effect(id = 76)]
impl<J: VoidInferOps> Effect<J> for GenerateDarkStarPromptEffect {
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let dark_tokens = jungle
                .darken(SPACE_PROBE_DISTANCE_PROMPT)
                .map_err(AtomError::Inference)?;
            let output = InferenceOutput {
                results: vec![SequenceOutput(dark_tokens)],
            };
            let propagation =
                Transmission::propagation_from_inference_output(jungle, &output).await?;
            Ok((propagation.clone(), propagation))
        }
    }
}

#[jungle::effect(id = 77)]
impl<J: VoidInferOps> Effect<J> for GenerateBlackDwarfPromptEffect {
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let dark_tokens = jungle
                .darken(SPACE_PROBE_DISTANCE_PROMPT)
                .map_err(AtomError::Inference)?;
            let output = InferenceOutput {
                results: (0..BLACK_DWARF_BATCH_SIZE)
                    .map(|_| SequenceOutput(dark_tokens.clone()))
                    .collect(),
            };
            let propagation =
                Transmission::propagation_from_inference_output(jungle, &output).await?;
            Ok((propagation.clone(), propagation))
        }
    }
}

#[jungle::effect(id = 78)]
impl<J> Effect<J> for DarkStarLossPolicyEffect {
    type In = [(Transmission, Transmission); 1];
    type Out = Potentiation;
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            Ok(Potentiation {
                loss_up: 0.4,
                loss_down: 0.8,
                seed: 29,
            })
        }
    }
}

#[jungle::effect(id = 79)]
impl<J: VoidInferOps> Effect<J> for BlackDwarfLossPolicyEffect {
    type In = [(Transmission, Transmission); 1];
    type Out = Potentiation;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let (up_tx, down_tx) = &input[0];
            let output_up = InferenceOutput::from_transmission(jungle, up_tx).await?;
            let output_down = InferenceOutput::from_transmission(jungle, down_tx).await?;
            let up_batch_size = output_up.results.len();
            let down_batch_size = output_down.results.len();

            info!(
                up_batch_size,
                down_batch_size, "black_dwarf reward received batch outputs"
            );

            if up_batch_size != BLACK_DWARF_BATCH_SIZE || down_batch_size != BLACK_DWARF_BATCH_SIZE
            {
                return Err(AtomError::Inference(format!(
                    "expected batch size {BLACK_DWARF_BATCH_SIZE} in black_dwarf reward fn, got up={up_batch_size}, down={down_batch_size}"
                )));
            }

            Ok(Potentiation {
                loss_up: 0.4,
                loss_down: 0.8,
                seed: 31,
            })
        }
    }
}

#[jungle::effect(id = 80)]
impl<J: VoidInferOps + FusionConcatOps> Effect<J> for ConcatFusionOutputsEffect {
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (_transform_id, (left_id, right_id)): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let mut merged_output = InferenceOutput::from_emission(jungle, left_id).await?;
            let right_output = InferenceOutput::from_emission(jungle, right_id).await?;
            merged_output.results.extend(right_output.results);

            let merged_output_bytes = to_allocvec(&merged_output)?;
            let merged_output_id = jungle
                .upload_to_void(merged_output_bytes)
                .await
                .map_err(AtomError::Upload)?;
            let merged_emission = Emission {
                metadata: (),
                output_id: InferenceOutputId::new(merged_output_id).into(),
            };
            let merged_emission_bytes = to_allocvec(&merged_emission)?;
            let merged_emission_id = jungle
                .upload_to_void(merged_emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            jungle.record_fusion_concat();
            Ok(EmissionId::new(merged_emission_id))
        }
    }
}

// ---------------------------------------------------------------------------
// Twin stack effects — moved here from black-hole-flux's `twin` module, which
// only the dark_star tests used.
// ---------------------------------------------------------------------------

async fn download_output<J>(
    jungle: &J,
    output_id: ArtifactRef<InferenceOutput>,
) -> Result<InferenceOutput, AtomError>
where
    J: VoidInferOps,
{
    let output_bytes = jungle
        .download_raw(output_id.object_id())
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
        output_id: InferenceOutputId::new(output_id).into(),
    };
    let emission_bytes = postcard::to_allocvec(&emission)?;
    let emission_id = jungle
        .upload_to_void(emission_bytes)
        .await
        .map_err(AtomError::Upload)?;

    Ok(EmissionId::new(emission_id))
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
struct LeftStackEffect<M>(PhantomData<fn() -> M>);

#[jungle::effect(id = 71)]
impl<M: Serialize + DeserializeOwned + Send + 'static, J: VoidInferOps> Effect<J>
    for LeftStackEffect<M>
{
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.id())
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.id())
                .await
                .map_err(AtomError::Download)?;

            let left_output = download_output(jungle, left_emission.output_id).await?;
            let right_output = download_output(jungle, right_emission.output_id).await?;
            let left_output = stack_dark_tokens("left", "right", left_output, right_output)?;

            debug!(
                left = %left_id,
                right = %right_id,
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
struct RightStackEffect<M>(PhantomData<fn() -> M>);

#[jungle::effect(id = 72)]
impl<M: Serialize + DeserializeOwned + Send + 'static, J: VoidInferOps> Effect<J>
    for RightStackEffect<M>
{
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.id())
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.id())
                .await
                .map_err(AtomError::Download)?;

            let left_output = download_output(jungle, left_emission.output_id).await?;
            let right_output = download_output(jungle, right_emission.output_id).await?;
            let right_output = stack_dark_tokens("right", "left", right_output, left_output)?;

            debug!(
                left = %left_id,
                right = %right_id,
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
struct RandStackEffect<M>(PhantomData<fn() -> M>);

#[jungle::effect(id = 73)]
impl<M: Serialize + DeserializeOwned + Send + 'static, J: VoidInferOps> Effect<J>
    for RandStackEffect<M>
{
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<M> = jungle
                .download_emission(left_id.id())
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<M> = jungle
                .download_emission(right_id.id())
                .await
                .map_err(AtomError::Download)?;

            let choose_left = random::<bool>();
            let mut left_output = download_output(jungle, left_emission.output_id).await?;
            let mut right_output = download_output(jungle, right_emission.output_id).await?;

            if choose_left {
                left_output = stack_dark_tokens("left", "right", left_output, right_output)?;
                debug!(
                    left = %left_id,
                    right = %right_id,
                    picked = "left",
                    sequence_count = left_output.results.len(),
                    combined_dark_tokens = total_dark_tokens(&left_output),
                    "stacked twin emissions with random-side metadata"
                );
                upload_stacked_emission(jungle, left_emission.metadata, left_output).await
            } else {
                right_output = stack_dark_tokens("right", "left", right_output, left_output)?;
                debug!(
                    left = %left_id,
                    right = %right_id,
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

#[jungle::effect(id = 81)]
impl<J: VoidInferOps + FusionConcatOps> Effect<J> for LeftStackTwinOutputsEffect {
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let merged_id = LeftStackEffect::<()>::effect(jungle, (left_id, right_id)).await?;
            jungle.record_fusion_concat();
            Ok(merged_id)
        }
    }
}

#[jungle::effect(id = 82)]
impl<J: VoidInferOps + FusionConcatOps> Effect<J> for RightStackTwinOutputsEffect {
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let merged_id = RightStackEffect::<()>::effect(jungle, (left_id, right_id)).await?;
            jungle.record_fusion_concat();
            Ok(merged_id)
        }
    }
}

#[jungle::effect(id = 83)]
impl<J: VoidInferOps + FusionConcatOps> Effect<J> for RandStackTwinOutputsEffect {
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (left_id, right_id): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let merged_id = RandStackEffect::<()>::effect(jungle, (left_id, right_id)).await?;
            jungle.record_fusion_concat();
            Ok(merged_id)
        }
    }
}
