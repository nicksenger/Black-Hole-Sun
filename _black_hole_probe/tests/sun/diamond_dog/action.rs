use super::*;

#[jungle::action]
impl Action for FinishStep {
    type Effect = NoEffect;
    type Input = Potentiation;
    type Output = ();

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("finish step failed".to_string()))
    }
}

#[jungle::action(carry = EmissionId)]
impl Action for PassEmission {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, input: Self::Input) -> ((), EmissionId) {
        ((), input)
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
        emission_id: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("pass emission failed".to_string()))?;
        Ok(emission_id)
    }
}

#[jungle::action]
impl Action for MarkLeft {
    type Effect = DelayedLeftEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("mark left emission failed: {error}")))
    }
}

#[jungle::action]
impl Action for MarkRight {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("mark right emission failed".to_string()))?;
        Ok(EmissionId::new(Uuid::from_u128(RIGHT_EMISSION)))
    }
}

#[jungle::action]
impl Action for RecordFusionInputs {
    type Effect = RecordFusionInputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(_state: &FusionState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("record fusion inputs failed: {error}")))
    }
}
