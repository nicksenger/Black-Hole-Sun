//! Atom actions — re-exports for the Atom flow.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use uuid::Uuid;

use crate::EmissionId;

pub use crate::cell::action::QuarkInferStep;

/// Unwraps the terminal result from the generic backoff flow payload.
pub struct ExtractQuarkInferBackoffResult<S>(PhantomData<fn() -> S>);

#[jungle::action(carry = (u32, ((Uuid, EmissionId), Result<EmissionId, Failure>)))]
impl<S> Action for ExtractQuarkInferBackoffResult<S> {
    type Effect = NoEffect;
    type Input = (u32, ((Uuid, EmissionId), Result<EmissionId, Failure>));
    type Output = EmissionId;

    fn emit(
        _state: &crate::cell::action::CellState<S>,
        input: Self::Input,
    ) -> (<Self::Effect as EffectSchema>::In, Self::Carry) {
        ((), input)
    }

    fn absorb(
        _state: &mut crate::cell::action::CellState<S>,
        _output: EffectCompletion<Self::Effect>,
        carry: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        carry.1.1
    }
}
