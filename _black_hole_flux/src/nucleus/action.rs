//! Nucleus actions — quark inference step.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::cell::CellState;
pub use black_hole_spec::EmissionId;

use super::effect::QuarkInfer;

// ---------------------------------------------------------------------------
// QuarkInferStep — action wrapper for use inside Nucleus flows
// ---------------------------------------------------------------------------

/// Action that performs quark inference on an [`EmissionId`].
pub struct QuarkInferStep<M>(PhantomData<fn() -> M>);

#[jungle::action(carry = EmissionId)]
impl<M> Action for QuarkInferStep<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = QuarkInfer<M>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, input: Self::Input) -> (EmissionId, EmissionId) {
        (input.clone(), input)
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
        _carry: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("quark inference failed: {e}")))
    }
}
