use super::*;

#[jungle::action]
impl Action for GenerateDarkStarPrompt {
    type Effect = GenerateDarkStarPromptEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("dark star generator failed: {error}")))
    }
}

#[jungle::action]
impl Action for GenerateBlackDwarfPrompt {
    type Effect = GenerateBlackDwarfPromptEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("black dwarf generator failed: {error}")))
    }
}

#[jungle::action]
impl Action for DarkStarLossPolicy {
    type Effect = DarkStarLossPolicyEffect;
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
        output.map_err(|error| Failure::Message(format!("dark star policy failed: {error}")))
    }
}

#[jungle::action]
impl Action for BlackDwarfLossPolicy {
    type Effect = BlackDwarfLossPolicyEffect;
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
        output.map_err(|error| Failure::Message(format!("black dwarf policy failed: {error}")))
    }
}

#[jungle::action]
impl Action for ConcatFusionOutputs {
    type Effect = ConcatFusionOutputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(_state: &FusionState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion concatenation failed: {error}")))
    }
}

#[jungle::action]
impl Action for LeftStackTwinOutputs {
    type Effect = LeftStackTwinOutputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(
        _state: &FusionState,
        (_transform_id, emissions): Self::Input,
    ) -> (EmissionId, EmissionId) {
        emissions
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("left stack twin failed: {error}")))
    }
}

#[jungle::action]
impl Action for RightStackTwinOutputs {
    type Effect = RightStackTwinOutputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(
        _state: &FusionState,
        (_transform_id, emissions): Self::Input,
    ) -> (EmissionId, EmissionId) {
        emissions
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("right stack twin failed: {error}")))
    }
}

#[jungle::action]
impl Action for RandStackTwinOutputs {
    type Effect = RandStackTwinOutputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(
        _state: &FusionState,
        (_transform_id, emissions): Self::Input,
    ) -> (EmissionId, EmissionId) {
        emissions
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("rand stack twin failed: {error}")))
    }
}
