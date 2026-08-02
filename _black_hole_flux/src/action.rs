//! QuarkInferStep — action wrapper for use inside Cell flows.

use std::marker::PhantomData;

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub use black_hole_spec::EmissionId;

use crate::effect::QuarkInfer;

/// Action that performs quark inference on an [`EmissionId`].
///
/// Stateless — bound to any animal state via the [`Identity`] aspect.
pub struct QuarkInferStep<M>(PhantomData<fn() -> M>);

#[jungle::action(carry = EmissionId)]
impl<M> Action for QuarkInferStep<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = QuarkInfer<M>;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &(), input: Self::Input) -> (EmissionId, EmissionId) {
        (input.clone(), input)
    }

    fn absorb(
        _state: &mut (),
        output: EffectCompletion<Self::Effect>,
        _carry: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("quark inference failed: {e}")))
    }
}
