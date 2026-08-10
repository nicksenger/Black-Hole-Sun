use super::*;
use black_hole_flux::ops::VoidInferOps;
use black_hole_flux::{AtomError, Emission};
use black_hole_sun::{InferenceOutput, InferenceOutputId, SequenceOutput};
use postcard::from_bytes;
use std::future::Future;

impl<J> EffectSchema<J> for DelayedLeftEffect {
    type Id = u64;
    type In = ();
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for DelayedLeftEffect {
    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(EmissionId(Uuid::from_u128(LEFT_EMISSION)))
        }
    }
}

impl<J> EffectSchema<J> for RecordFusionInputsEffect {
    type Id = u64;
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for RecordFusionInputsEffect
where
    J: FusionProbeOps,
{
    fn effect(
        jungle: &J,
        (transform_id, (p1, p2)): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        jungle.record_fusion_inputs(transform_id, p1.0, p2.0);
        std::future::ready(Ok(EmissionId(Uuid::from_u128(FUSED_EMISSION))))
    }
}

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
            let tokenizer = dark_star_tokenizer().map_err(AtomError::Inference)?;
            let dark_tokens = prompt_to_dark_tokens(SPACE_PROBE_DISTANCE_PROMPT, tokenizer)
                .map_err(AtomError::Inference)?;
            let output = InferenceOutput {
                results: vec![SequenceOutput(dark_tokens)],
            };
            let output_bytes = to_allocvec(&output)?;
            let output_id = jungle
                .upload_to_void(output_bytes)
                .await
                .map_err(AtomError::Upload)?;
            let emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(output_id),
            };
            let emission_bytes = to_allocvec(&emission)?;
            let emission_id = jungle
                .upload_to_void(emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            let propagation = Transmission::Propagation {
                emission_id: EmissionId(emission_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            };
            Ok((propagation.clone(), propagation))
        }
    }
}

#[jungle::effect]
impl<J> Effect<J> for DarkStarLossPolicyEffect {
    type Id = u64;
    type In = (Transmission, Transmission);
    type Out = (f32, f32);
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move { Ok((0.4, 0.8)) }
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
            let left_emission: Emission<()> = jungle
                .download_emission(left_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<()> = jungle
                .download_emission(right_id.0)
                .await
                .map_err(AtomError::Download)?;

            let left_bytes = jungle
                .download_raw(left_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_bytes = jungle
                .download_raw(right_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;

            let mut merged_output: InferenceOutput = from_bytes(&left_bytes)?;
            let right_output: InferenceOutput = from_bytes(&right_bytes)?;
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
