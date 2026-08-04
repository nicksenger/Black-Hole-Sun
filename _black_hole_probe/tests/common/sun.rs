//! Test generator and policy flows for Sun orchestration.

use std::future::Future;

use black_hole_sun::black_hole_flux::ops::VoidInferOps;
use black_hole_sun::black_hole_flux::{AtomError, SunState};
use black_hole_sun::{
    DarkToken, Emission, EmissionId, InferenceOutput, InferenceOutputId, ObjectId, SequenceOutput,
    Transmission,
};
use jungle_sdk::prelude::*;
use tracing::debug;

/// Test flow that creates both initial propagation values.
#[derive(Flow)]
pub struct Generator(Step<Initialize>);

/// Test flow that converts completed propagation values into fixed losses.
#[derive(Flow)]
pub struct Policy(Step<ComputeLoss>);

/// Creates one initial transmission for each propagation branch.
pub struct Initialize;

#[jungle::action]
impl Action for Initialize {
    type Effect = InitializeEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("initialize failed".to_string()))
    }
}

/// Creates a minimal initial emission for the test Sun.
pub struct InitializeEffect;

impl<J> EffectSchema<J> for InitializeEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;
}

impl<J> Effect<J> for InitializeEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("creating initial test sun emission");

            let inference_output = InferenceOutput {
                results: vec![SequenceOutput(vec![DarkToken {
                    predicted: 0,
                    dark_knowledge: Vec::new(),
                }])],
            };
            let inference_output_bytes = postcard::to_allocvec(&inference_output)?;
            let inference_output_id = jungle
                .upload_to_void(inference_output_bytes)
                .await
                .map_err(AtomError::Upload)?;

            let emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(inference_output_id),
            };
            let emission_bytes = postcard::to_allocvec(&emission)?;
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

/// Computes fixed losses for test Suns.
pub struct ComputeLoss;

#[jungle::action]
impl Action for ComputeLoss {
    type Effect = ComputeLossEffect;
    type Input = (Transmission, Transmission);
    type Output = (f32, f32);
    type Carry = ();

    fn emit(_state: &SunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("compute loss failed: {error}")))
    }
}

/// Returns deterministic losses suitable for orchestration tests.
pub struct ComputeLossEffect;

#[jungle::effect]
impl<J> Effect<J> for ComputeLossEffect {
    type Id = u64;
    type In = (Transmission, Transmission);
    type Out = (f32, f32);
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("using fixed test loss");
            Ok((0.1, 0.1))
        }
    }
}
