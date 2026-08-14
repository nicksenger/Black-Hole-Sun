use std::future::Future;

use black_hole_sun::ops::{InferenceOutputOps, TransmissionOps, VoidInferOps};
use black_hole_sun::twin::effect::{LeftStackEffect, RandStackEffect, RightStackEffect};
use black_hole_sun::{AtomError, Emission, InferenceOutput, InferenceOutputId, SequenceOutput};
use postcard::to_allocvec;
use tracing::info;

use super::*;

const BLACK_DWARF_BATCH_SIZE: usize = 8;

impl<J> EffectSchema<J> for GenerateDarkStarPromptEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;
}

impl<J> Effect<J> for GenerateDarkStarPromptEffect
where
    J: VoidInferOps,
{
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

impl<J> EffectSchema<J> for GenerateBlackDwarfPromptEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;
}

impl<J> Effect<J> for GenerateBlackDwarfPromptEffect
where
    J: VoidInferOps,
{
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

#[jungle::effect]
impl<J> Effect<J> for DarkStarLossPolicyEffect {
    type Id = u64;
    type In = [(Transmission, Transmission); 1];
    type Out = (f32, f32);
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move { Ok((0.4, 0.8)) }
    }
}

impl<J> EffectSchema<J> for BlackDwarfLossPolicyEffect {
    type Id = u64;
    type In = [(Transmission, Transmission); 1];
    type Out = (f32, f32);
    type Err = AtomError;
}

impl<J> Effect<J> for BlackDwarfLossPolicyEffect
where
    J: VoidInferOps,
{
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

            Ok((0.4, 0.8))
        }
    }
}

impl<J> EffectSchema<J> for ConcatFusionOutputsEffect {
    type Id = u64;
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for ConcatFusionOutputsEffect
where
    J: VoidInferOps + FusionConcatOps,
{
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
                output_id: InferenceOutputId(merged_output_id),
            };
            let merged_emission_bytes = to_allocvec(&merged_emission)?;
            let merged_emission_id = jungle
                .upload_to_void(merged_emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            jungle.record_fusion_concat();
            Ok(EmissionId(merged_emission_id))
        }
    }
}

impl<J> EffectSchema<J> for LeftStackTwinOutputsEffect {
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for LeftStackTwinOutputsEffect
where
    J: VoidInferOps + FusionConcatOps,
{
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

impl<J> EffectSchema<J> for RightStackTwinOutputsEffect {
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for RightStackTwinOutputsEffect
where
    J: VoidInferOps + FusionConcatOps,
{
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

impl<J> EffectSchema<J> for RandStackTwinOutputsEffect {
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for RandStackTwinOutputsEffect
where
    J: VoidInferOps + FusionConcatOps,
{
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
