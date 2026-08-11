use std::future::Future;

use black_hole_sun::ops::VoidInferOps;
use black_hole_sun::{AtomError, Emission, InferenceOutput, InferenceOutputId, SequenceOutput};
use postcard::{from_bytes, to_allocvec};
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

fn propagation_emission_id(transmission: &Transmission) -> Result<EmissionId, AtomError> {
    match transmission {
        Transmission::Propagation { emission_id, .. } => Ok(emission_id.clone()),
        Transmission::Potentiation { .. } => Err(AtomError::Inference(
            "expected propagation transmission in black dwarf reward".to_string(),
        )),
    }
}

impl<J> EffectSchema<J> for BlackDwarfLossPolicyEffect {
    type Id = u64;
    type In = (Transmission, Transmission);
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
            let up_emission_id = propagation_emission_id(&input.0)?;
            let down_emission_id = propagation_emission_id(&input.1)?;

            let up_emission: Emission<()> = jungle
                .download_emission(up_emission_id.0)
                .await
                .map_err(AtomError::Download)?;
            let down_emission: Emission<()> = jungle
                .download_emission(down_emission_id.0)
                .await
                .map_err(AtomError::Download)?;

            let up_bytes = jungle
                .download_raw(up_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;
            let down_bytes = jungle
                .download_raw(down_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;

            let output_up: InferenceOutput = from_bytes(&up_bytes)?;
            let output_down: InferenceOutput = from_bytes(&down_bytes)?;
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
