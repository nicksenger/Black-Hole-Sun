//! Atom actions — re-exports for the Atom flow.
pub use crate::nodes::cell::action::MassInferStep;

use std::marker::PhantomData;

use black_hole_contract::TensorContract;
use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use uuid::Uuid;

/// Action wrapper for an operation-typed Atom inference step.
pub struct OperationMassInferStep<M, Op, S = ()>(PhantomData<fn() -> (M, Op, S)>);

#[jungle::action]
impl<M, Op, S> Action for OperationMassInferStep<M, Op, S>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: TensorContract + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
{
    type Effect = super::effect::OperationMassInfer<M, Op>;
    type Input = (Uuid, black_hole_spec::EmissionId<Op::Input>);
    type Output = black_hole_spec::EmissionId<Op::Output>;

    fn emit(_state: &crate::nodes::cell::action::CellState<S>, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut crate::nodes::cell::action::CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("operation inference failed: {error}")))
    }
}
