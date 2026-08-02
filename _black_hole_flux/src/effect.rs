//! Effects for quark inference, perturbation, optimization, and perturbation claiming.

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

use crate::NucleusError;

// ---------------------------------------------------------------------------
// QuarkInfer — download → infer → upload in a single effect
// ---------------------------------------------------------------------------

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

            // 1. Download the emission to get the output_id and metadata.
            let emission: Emission<M> = jungle
                .download_emission(obj_id)
                .await
                .map_err(NucleusError::Download)?;
            let input_output_id = emission.output_id.0;
            debug!(emission_id = %obj_id, "downloaded emission for inference");

            // 2. Run quark inference, passing the InferenceOutputId directly.
            let output_id = jungle
                .infer(input_output_id)
                .await
                .map_err(NucleusError::Inference)?;
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
                .map_err(NucleusError::Upload)?;
            debug!(result_id = %result_id, "uploaded inference result emission");

            Ok(EmissionId(result_id))
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkPerturbUp — perturb quark weights in the positive direction
// ---------------------------------------------------------------------------

/// Effect that perturbs the associated quark's weights upward.
///
/// Takes a random `seed` for reproducibility and returns `()` on success.
pub struct QuarkPerturbUp;

impl<J> EffectSchema<J> for QuarkPerturbUp {
    type Id = u64;
    type In = u64;
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkPerturbUp
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        seed: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(seed, "perturbing quark weights up");
            jungle
                .perturb_up(seed)
                .await
                .map_err(NucleusError::PerturbUp)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkPerturbDown — perturb quark weights in the negative direction
// ---------------------------------------------------------------------------

/// Effect that perturbs the associated quark's weights downward.
pub struct QuarkPerturbDown;

impl<J> EffectSchema<J> for QuarkPerturbDown {
    type Id = u64;
    type In = ();
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkPerturbDown
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("perturbing quark weights down");
            jungle
                .perturb_down()
                .await
                .map_err(NucleusError::PerturbDown)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// QuarkOptimize — apply QuZO optimization update
// ---------------------------------------------------------------------------

/// Effect that applies the QuZO optimization step using up/down loss values.
///
/// Takes `(loss_up, loss_down)` and returns `()` on success.
pub struct QuarkOptimize;

impl<J> EffectSchema<J> for QuarkOptimize {
    type Id = u64;
    type In = (f32, f32);
    type Out = ();
    type Err = NucleusError;
}

impl<J> Effect<J> for QuarkOptimize
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        (loss_up, loss_down): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!(loss_up, loss_down, "applying quark optimization");
            jungle
                .optimize(loss_up, loss_down)
                .await
                .map_err(NucleusError::Optimize)?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// ClaimPerturbation — await an external jungle perturbation
// ---------------------------------------------------------------------------

/// Effect that awaits an external jungle perturbation containing an [`EmissionId`].
///
/// Delegates to [`VoidInferOps::claim_perturbation`] which should poll the
/// Jungle runtime for a claimed perturbation with backoff until one arrives.
pub struct ClaimPerturbation;

impl<J> EffectSchema<J> for ClaimPerturbation {
    type Id = u64;
    type In = ();
    type Out = EmissionId;
    type Err = NucleusError;
}

impl<J> Effect<J> for ClaimPerturbation
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("awaiting external perturbation (EmissionId)");
            let emission_id = jungle
                .claim_perturbation()
                .await
                .map_err(NucleusError::Claim)?;
            debug!(emission_id = %emission_id.0, "perturbation claimed");
            Ok(emission_id)
        }
    }
}

// ---------------------------------------------------------------------------
// ClaimLoss — await an external jungle perturbation with loss values
// ---------------------------------------------------------------------------

/// Effect that awaits an external jungle perturbation containing
/// `(loss_up: f32, loss_down: f32)`.
///
/// Delegates to [`VoidInferOps::claim_loss_perturbation`] which should poll
/// the Jungle runtime for a claimed perturbation with backoff, deserialize
/// it as a loss tuple, and acknowledge it.
pub struct ClaimLoss;

impl<J> EffectSchema<J> for ClaimLoss {
    type Id = u64;
    type In = ();
    type Out = (f32, f32);
    type Err = NucleusError;
}

impl<J> Effect<J> for ClaimLoss
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            debug!("awaiting external perturbation (loss tuple)");
            let loss = jungle
                .claim_loss_perturbation()
                .await
                .map_err(NucleusError::Claim)?;
            debug!(?loss, "loss perturbation claimed");
            Ok(loss)
        }
    }
}
