//! Twin actions - default `In` transforms plus quark inference step.

use std::marker::PhantomData;

use black_hole_spec::EmissionId;
use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

use super::effect::{LeftStackEffect, RandStackEffect, RightStackEffect};

pub use crate::cell::action::QuarkInferStep;

/// Default `Twin` input transform that keeps left metadata and appends right
/// dark tokens onto each corresponding left sequence.
pub struct LeftStack<S = (), M = ()>(PhantomData<fn() -> (S, M)>);

#[jungle::action(carry = Uuid)]
impl<S, M> Action for LeftStack<S, M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = LeftStackEffect<M>;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = (Uuid, EmissionId);

    fn emit(_state: &S, (model_id, emissions): Self::Input) -> ((EmissionId, EmissionId), Uuid) {
        (emissions, model_id)
    }

    fn absorb(
        _state: &mut S,
        output: EffectCompletion<Self::Effect>,
        model_id: Uuid,
    ) -> Result<Self::Output, Failure> {
        let emission_id =
            output.map_err(|error| Failure::Message(format!("left stack failed: {error}")))?;
        Ok((model_id, emission_id))
    }
}

/// Default `Twin` input transform that keeps right metadata and appends left
/// dark tokens onto each corresponding right sequence.
pub struct RightStack<S = (), M = ()>(PhantomData<fn() -> (S, M)>);

#[jungle::action(carry = Uuid)]
impl<S, M> Action for RightStack<S, M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = RightStackEffect<M>;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = (Uuid, EmissionId);

    fn emit(_state: &S, (model_id, emissions): Self::Input) -> ((EmissionId, EmissionId), Uuid) {
        (emissions, model_id)
    }

    fn absorb(
        _state: &mut S,
        output: EffectCompletion<Self::Effect>,
        model_id: Uuid,
    ) -> Result<Self::Output, Failure> {
        let emission_id =
            output.map_err(|error| Failure::Message(format!("right stack failed: {error}")))?;
        Ok((model_id, emission_id))
    }
}

/// Default `Twin` input transform that randomly chooses left- or right-based
/// stacking behavior each invocation.
pub struct RandStack<S = (), M = ()>(PhantomData<fn() -> (S, M)>);

#[jungle::action(carry = Uuid)]
impl<S, M> Action for RandStack<S, M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = RandStackEffect<M>;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = (Uuid, EmissionId);

    fn emit(_state: &S, (model_id, emissions): Self::Input) -> ((EmissionId, EmissionId), Uuid) {
        (emissions, model_id)
    }

    fn absorb(
        _state: &mut S,
        output: EffectCompletion<Self::Effect>,
        model_id: Uuid,
    ) -> Result<Self::Output, Failure> {
        let emission_id =
            output.map_err(|error| Failure::Message(format!("rand stack failed: {error}")))?;
        Ok((model_id, emission_id))
    }
}
