//! Cell actions for perturbation, optimization, transmission, and waiting.

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

use jungle_sdk::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::mass::{DefaultConfig, ModelConfig};
use black_hole_spec::TensorContract;
use black_hole_type::ObjectId;

// ---------------------------------------------------------------------------
// CellState — holds the next transmission ID threaded across Cell iterations
// ---------------------------------------------------------------------------

/// State carried by a [`Cell`](crate::Cell) journey.
///
/// Animals that use [`Cell`](crate::Cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID. The generic `S` payload is available to user flows via
/// [`CellState::inner`] and defaults to `()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CellState<S = ()> {
    /// Stable ID of the mass model instance owned by this cell.
    pub model_id: Uuid,
    /// Void key of the next [`Transmission`](black_hole_type::Transmission) to download.
    pub recv_id: black_hole_type::ObjectId,
    /// Void key of the next [`Transmission`](black_hole_type::Transmission) to upload.
    pub send_id: black_hole_type::ObjectId,
    /// Random seed passed to the perturb-up step each iteration.
    pub perturbation_seed: u64,
    /// Last known frozen status for this model instance.
    #[serde(default)]
    pub is_frozen: bool,
    /// Number of infer/transmit microsteps per perturbation phase.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
    /// Number of completed microsteps in the current propagation phase.
    #[serde(default)]
    pub grad_step: usize,
    /// Number of completed optimization steps for this model instance.
    #[serde(default)]
    pub optimization_step: usize,
    /// Save a checkpoint every this many optimization steps; zero disables it.
    #[serde(default)]
    pub checkpoint_steps: usize,
    /// Local directory where checkpoints are written when enabled.
    #[serde(default)]
    pub checkpoint_dir: Option<PathBuf>,
    /// User-provided state threaded through all cell actions.
    #[serde(default)]
    pub inner: S,
}

#[derive(Debug, Clone, Default)]
struct CheckpointSettings {
    steps: usize,
    directory: Option<PathBuf>,
}

static CHECKPOINT_SETTINGS: OnceLock<RwLock<CheckpointSettings>> = OnceLock::new();

/// Configure checkpointing for subsequently spawned cells in this process.
///
/// The local toy examples use this to pass their command-line setting through
/// the type-level deployment flow into every operation cell.
pub fn configure_checkpointing(steps: usize, directory: Option<PathBuf>) {
    let settings = CHECKPOINT_SETTINGS.get_or_init(|| RwLock::new(CheckpointSettings::default()));
    *settings.write().expect("checkpoint settings lock poisoned") = CheckpointSettings {
        steps,
        directory: (steps > 0).then_some(directory).flatten(),
    };
}

fn checkpoint_settings() -> CheckpointSettings {
    CHECKPOINT_SETTINGS
        .get_or_init(|| RwLock::new(CheckpointSettings::default()))
        .read()
        .expect("checkpoint settings lock poisoned")
        .clone()
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

pub use black_hole_type::EmissionId;
pub use black_hole_type::Potentiation;

use super::effect::{
    GenerateModelIdEffect, MassOptimize, MassPerturbDown, MassPerturbUp, MassShutdown, MassStart,
    OperationMassOptimize, OperationMassPerturbDown, OperationMassPerturbUp, OperationMassStart,
    Transmit as TransmitEffect, TransmitArtifactEffect, WaitForArtifactDeliveryEffect,
    WaitForOperationalControlEffect, WaitForPotentiationEffect, WaitForPropagationEffect,
};

// ---------------------------------------------------------------------------
// InitRecvId — set recv_id from the seed ObjectId (first step of Cell flow)
// ---------------------------------------------------------------------------

/// Initialization payload for one spawned [`Cell`](crate::Cell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Init {
    /// First propagation mailbox that this cell should await.
    pub recv_id: black_hole_type::ObjectId,
    /// Number of propagation microsteps to run per perturbation phase.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub grad_steps: usize,
}

impl Default for Init {
    fn default() -> Self {
        Self {
            recv_id: black_hole_type::ObjectId::nil(),
            grad_steps: default_gradient_accumulation_steps(),
        }
    }
}

/// Action that initializes `recv_id` in [`CellState`] from the seed
/// [`Init`] payload.
///
/// This is the very first step of a [`Cell`](crate::Cell) flow, converting the
/// animal seed into the initial receive ID and accumulation settings for the
/// training loop.
pub struct InitRecvId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = Init)]
impl<S> Action for InitRecvId<S> {
    type Effect = NoEffect;
    type Input = Init;
    type Output = ();

    fn emit(_state: &CellState<S>, input: Self::Input) -> ((), Init) {
        ((), input)
    }

    fn absorb(
        state: &mut CellState<S>,
        _output: EffectCompletion<Self::Effect>,
        carry: Init,
    ) -> Result<Self::Output, Failure> {
        state.recv_id = carry.recv_id;
        state.grad_steps = carry.grad_steps.max(1);
        state.grad_step = 0;
        let settings = checkpoint_settings();
        state.checkpoint_steps = settings.steps;
        state.checkpoint_dir = settings.directory;
        Ok(())
    }
}

/// Resets the microstep cursor before a propagation microstep phase begins.
pub struct BeginGradientAccumulation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for BeginGradientAccumulation<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &CellState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("begin accumulation failed".to_string()))?;
        state.grad_step = 0;
        if state.grad_steps == 0 {
            state.grad_steps = 1;
        }
        Ok(())
    }
}

/// Advances the microstep cursor after one propagation/infer microstep.
pub struct AdvanceGradientStep<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for AdvanceGradientStep<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &CellState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("advance accumulation failed".to_string()))?;
        state.grad_step = state.grad_step.saturating_add(1);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Model instance lifecycle
// ---------------------------------------------------------------------------

/// Generates the stable model instance ID owned by this cell.
pub struct GenerateModelId<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for GenerateModelId<S> {
    type Effect = GenerateModelIdEffect;
    type Input = ();
    type Output = Uuid;

    fn emit(_state: &CellState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("generate model ID failed: {error}")))
    }
}

/// Starts the generated mass model instance and stores its ID in cell state.
pub struct StartModel<S = (), H = DefaultConfig>(PhantomData<fn() -> (S, H)>);

#[jungle::action(carry = Uuid)]
impl<S, H> Action for StartModel<S, H>
where
    H: ModelConfig,
{
    type Effect = MassStart<H>;
    type Input = Uuid;
    type Output = ();

    fn emit(_state: &CellState<S>, model_id: Self::Input) -> (Uuid, Uuid) {
        (model_id, model_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        model_id: Uuid,
    ) -> Result<Self::Output, Failure> {
        state.is_frozen =
            output.map_err(|error| Failure::Message(format!("start model failed: {error}")))?;
        state.model_id = model_id;
        Ok(())
    }
}

/// Shuts down the mass model instance owned by this cell.
pub struct ShutdownModel<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ShutdownModel<S> {
    type Effect = MassShutdown;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> Uuid {
        state.model_id
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("shutdown model failed: {error}")))
    }
}

/// Adds this cell's model ID to the input passed into its Atom.
pub struct PrepareAtomInput<S = ()>(PhantomData<fn() -> S>);

#[jungle::action(carry = EmissionId)]
impl<S> Action for PrepareAtomInput<S> {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = (Uuid, EmissionId);

    fn emit(_state: &CellState<S>, emission_id: Self::Input) -> ((), EmissionId) {
        ((), emission_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        emission_id: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare Atom input failed".to_string()))?;
        Ok((state.model_id, emission_id))
    }
}

/// Starts an instance of a generic tensor operation and stores its ID.
pub struct StartOperation<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action(carry = ObjectId)]
impl<Op, S> Action for StartOperation<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = OperationMassStart<Op>;
    type Input = ObjectId;
    type Output = ();

    fn emit(_state: &CellState<S>, instance_id: Self::Input) -> (ObjectId, ObjectId) {
        (instance_id, instance_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        instance_id: ObjectId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("start operation failed: {error}")))?;
        state.model_id = instance_id;
        Ok(())
    }
}

/// Adds the cell's operation instance ID to a typed input emission.
pub struct PrepareOperationInput<T, S = ()>(PhantomData<fn() -> (T, S)>);

#[jungle::action(carry = black_hole_type::EmissionId<T>)]
impl<T, S> Action for PrepareOperationInput<T, S>
where
    T: Send + 'static,
{
    type Effect = NoEffect;
    type Input = black_hole_type::EmissionId<T>;
    type Output = (ObjectId, black_hole_type::EmissionId<T>);

    fn emit(
        _state: &CellState<S>,
        emission_id: Self::Input,
    ) -> ((), black_hole_type::EmissionId<T>) {
        ((), emission_id)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
        emission_id: black_hole_type::EmissionId<T>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare operation input failed".to_string()))?;
        Ok((state.model_id, emission_id))
    }
}

// ---------------------------------------------------------------------------
// Potentiation — payload from a Transmission::Potentiation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Propagation — payload from a Transmission::Propagation
// ---------------------------------------------------------------------------

/// Payload carried by a [`Transmission::Propagation`](black_hole_type::Transmission).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Propagation {
    pub emission_id: EmissionId,
    pub recv_id: black_hole_type::ObjectId,
    pub send_id: black_hole_type::ObjectId,
}

// ---------------------------------------------------------------------------
// MassInferStep — action wrapper for use inside Atom flows
// ---------------------------------------------------------------------------

/// Action that performs mass inference for one model instance.
pub struct MassInferStep<M, S = (), H = DefaultConfig>(PhantomData<fn() -> (M, S, H)>);

#[jungle::action]
impl<M, S, H> Action for MassInferStep<M, S, H>
where
    M: Serialize + DeserializeOwned + Send + 'static,
    H: ModelConfig,
{
    type Effect = super::super::atom::effect::MassInfer<M, H>;
    type Input = (Uuid, EmissionId);
    type Output = EmissionId;

    fn emit(_state: &CellState<S>, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("mass inference failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbUp — perturb mass weights upward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbUp<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PerturbUp<S> {
    type Effect = MassPerturbUp;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> (Uuid, u64) {
        (state.model_id, state.perturbation_seed)
    }

    fn absorb(
        _: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb up failed: {e}")))
    }
}

pub struct PerturbOperationUp<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for PerturbOperationUp<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = OperationMassPerturbUp<Op>;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> (ObjectId, u64) {
        (state.model_id, state.perturbation_seed)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("perturb operation up failed: {error}")))
    }
}

// ---------------------------------------------------------------------------
// PerturbDown — perturb mass weights downward (no carry)
// ---------------------------------------------------------------------------

pub struct PerturbDown<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PerturbDown<S> {
    type Effect = MassPerturbDown;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> Uuid {
        state.model_id
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("perturb down failed: {e}")))
    }
}

pub struct PerturbOperationDown<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for PerturbOperationDown<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = OperationMassPerturbDown<Op>;
    type Input = ();
    type Output = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> ObjectId {
        state.model_id
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("perturb operation down failed: {error}")))
    }
}

// ---------------------------------------------------------------------------
// Optimize — apply QuZO optimization (no carry)
// ---------------------------------------------------------------------------

pub struct Optimize<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for Optimize<S> {
    type Effect = MassOptimize;
    type Input = Potentiation;
    type Output = ();

    fn emit(state: &CellState<S>, input: Self::Input) -> (Uuid, Potentiation) {
        (state.model_id, input)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        state.is_frozen = output.map_err(|e| Failure::Message(format!("optimize failed: {e}")))?;
        state.optimization_step = state.optimization_step.saturating_add(1);
        Ok(())
    }
}

pub struct OptimizeOperation<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for OptimizeOperation<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = OperationMassOptimize<Op>;
    type Input = Potentiation;
    type Output = ();

    fn emit(state: &CellState<S>, input: Self::Input) -> (ObjectId, Potentiation) {
        (state.model_id, input)
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("optimize operation failed: {error}")))?;
        state.optimization_step = state.optimization_step.saturating_add(1);
        Ok(())
    }
}

/// Saves an operation checkpoint when the cell's configured interval is due.
pub struct CheckpointOperation<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for CheckpointOperation<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = super::effect::OperationMassCheckpoint<Op>;
    type Input = ();
    type Output = ();

    fn emit(
        state: &CellState<S>,
        _input: Self::Input,
    ) -> (ObjectId, usize, usize, Option<PathBuf>) {
        (
            state.model_id,
            state.optimization_step,
            state.checkpoint_steps,
            state.checkpoint_dir.clone(),
        )
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("checkpoint operation failed: {error}")))
    }
}

// ---------------------------------------------------------------------------
// WaitForPropagation — await a Transmission::Propagation from void
// ---------------------------------------------------------------------------

/// Action that waits for a propagation transmission using the recv_id from
/// [`CellState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` and `send_id` in state, and emits the [`Propagation`] payload.
pub struct WaitForPropagation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPropagation<S> {
    type Effect = WaitForPropagationEffect;
    type Input = ();
    type Output = EmissionId;
    type Carry = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_type::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let propagation =
            output.map_err(|e| Failure::Message(format!("wait for propagation failed: {e}")))?;
        state.recv_id = propagation.recv_id;
        state.send_id = propagation.send_id;
        Ok(propagation.emission_id)
    }
}

/// Operation-typed variant of [`WaitForPropagation`].
///
/// The two-sided ZO scheduler transports its graph messages as
/// `Transmission::Propagation`. Convert the untyped emission reference at
/// this boundary so the operation atom can still retain its typed contract.
pub struct WaitForOperationPropagation<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for WaitForOperationPropagation<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
    Op::Input: Send,
{
    type Effect = WaitForPropagationEffect;
    type Input = ();
    type Output = EmissionId<Op::Input>;

    fn emit(state: &CellState<S>, _input: Self::Input) -> ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let propagation = output.map_err(|error| {
            Failure::Message(format!("wait for operation propagation failed: {error}"))
        })?;
        state.recv_id = propagation.recv_id;
        state.send_id = propagation.send_id;
        Ok(EmissionId::new(propagation.emission_id.id()))
    }
}

/// Waits for a typed artifact delivery and advances the cell mailboxes.
pub struct WaitForArtifact<T, S = ()>(PhantomData<fn() -> (T, S)>);

#[jungle::action]
impl<T, S> Action for WaitForArtifact<T, S>
where
    T: Send + 'static,
{
    type Effect = WaitForArtifactDeliveryEffect<T>;
    type Input = ();
    type Output = black_hole_type::EmissionId<T>;

    fn emit(state: &CellState<S>, _input: Self::Input) -> ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let delivery = output
            .map_err(|error| Failure::Message(format!("wait for artifact failed: {error}")))?;
        state.recv_id = delivery.recv;
        state.send_id = delivery.send;
        Ok(delivery.emission_id)
    }
}

// ---------------------------------------------------------------------------
// WaitForPotentiation — await a Transmission::Potentiation from void
// ---------------------------------------------------------------------------

/// Action that waits for a potentiation transmission using the recv_id from
/// [`CellState`].
///
/// Reads `recv_id` from state, downloads the transmission, stores the new
/// `recv_id` in state, and emits the [`Potentiation`] payload.
pub struct WaitForPotentiation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for WaitForPotentiation<S> {
    type Effect = WaitForPotentiationEffect;
    type Input = ();
    type Output = Potentiation;
    type Carry = ();

    fn emit(state: &CellState<S>, _input: Self::Input) -> black_hole_type::ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (potentiation, recv_id) =
            output.map_err(|e| Failure::Message(format!("wait for potentiation failed: {e}")))?;
        state.recv_id = recv_id;
        state.perturbation_seed = potentiation.seed;
        Ok(potentiation)
    }
}

/// Waits for strategy-selected control without coupling it to the data plane.
pub struct WaitForOperationalControl<C, S = ()>(PhantomData<fn() -> (C, S)>);

#[jungle::action]
impl<C, S> Action for WaitForOperationalControl<C, S>
where
    C: Serialize + DeserializeOwned + Send + 'static,
{
    type Effect = WaitForOperationalControlEffect<C>;
    type Input = ();
    type Output = C;

    fn emit(state: &CellState<S>, _input: Self::Input) -> ObjectId {
        state.recv_id
    }

    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let control = output
            .map_err(|error| Failure::Message(format!("wait for control failed: {error}")))?;
        state.recv_id = control.recv;
        Ok(control.control)
    }
}

// ---------------------------------------------------------------------------
// Transmit — propagates an emission to the next cell
// ---------------------------------------------------------------------------

/// Propagates an [`EmissionId`] to the next cell.
///
/// Replaces the previous `DiscardEmission` action by actually transmitting
/// the emission output rather than discarding it.
pub struct Transmit<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for Transmit<S> {
    type Effect = TransmitEffect;
    type Input = EmissionId;
    type Output = ();

    fn emit(state: &CellState<S>, input: Self::Input) -> (EmissionId, black_hole_type::ObjectId) {
        (input, state.send_id)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("transmit failed: {e}")))
    }
}

/// Operation-typed variant of [`Transmit`] for the two-sided ZO wire format.
pub struct TransmitOperation<Op, S = ()>(PhantomData<fn() -> (Op, S)>);

#[jungle::action]
impl<Op, S> Action for TransmitOperation<Op, S>
where
    Op: TensorContract + Send + Sync + 'static,
    Op::Output: Send,
{
    type Effect = TransmitEffect;
    type Input = EmissionId<Op::Output>;
    type Output = ();

    fn emit(state: &CellState<S>, emission_id: Self::Input) -> (EmissionId, ObjectId) {
        (EmissionId::new(emission_id.id()), state.send_id)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("transmit operation failed: {error}")))
    }
}

/// Publishes a typed output emission for a parent scheduler.
pub struct TransmitArtifact<T, S = ()>(PhantomData<fn() -> (T, S)>);

#[jungle::action]
impl<T, S> Action for TransmitArtifact<T, S>
where
    T: Send + 'static,
{
    type Effect = TransmitArtifactEffect<T>;
    type Input = black_hole_type::EmissionId<T>;
    type Output = ();

    fn emit(
        state: &CellState<S>,
        emission_id: Self::Input,
    ) -> (black_hole_type::EmissionId<T>, ObjectId) {
        (emission_id, state.send_id)
    }

    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("transmit artifact failed: {error}")))
    }
}
