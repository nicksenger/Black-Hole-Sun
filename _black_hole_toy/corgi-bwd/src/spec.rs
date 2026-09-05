//! Cells, the backward Sun flow, and the training policy.

use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{OnceLock, RwLock};

use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::topology::{BackwardTypedEdges, Edge, Unary};
use black_hole_sun::{
    ArtifactDelivery, BackwardOperationPrimordium, OperationNode, PipelineBackward,
    PipelineBackwardState, PipelineStepResult, VoidOps,
};
use corgi_fwd::spec::{HeadOp, Image, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use toy_common::dataset::BATCH_SIZE;
use typenum::consts::{U0, U1, U2, U3, U4, U5, U6};

/// Micro-batches per pipeline step.
pub const MICRO_BATCHES: usize = 8;

#[derive(Clone, Default)]
struct UnifiedCheckpointSettings {
    every: usize,
    directory: Option<PathBuf>,
}

static UNIFIED_CHECKPOINT_SETTINGS: OnceLock<RwLock<UnifiedCheckpointSettings>> = OnceLock::new();

pub fn configure_unified_checkpointing(every: usize, directory: Option<PathBuf>) {
    let settings = UNIFIED_CHECKPOINT_SETTINGS
        .get_or_init(|| RwLock::new(UnifiedCheckpointSettings::default()));
    *settings
        .write()
        .expect("unified checkpoint settings lock poisoned") =
        UnifiedCheckpointSettings { every, directory };
}

fn unified_checkpoint_settings() -> UnifiedCheckpointSettings {
    UNIFIED_CHECKPOINT_SETTINGS
        .get_or_init(|| RwLock::new(UnifiedCheckpointSettings::default()))
        .read()
        .expect("unified checkpoint settings lock poisoned")
        .clone()
}

macro_rules! operation_cell {
    ($cell:ident, $id:ty, $op:ty) => {
        pub struct $cell;
        impl Animal for $cell {
            type Id = Id<$id>;
            type Generation = U0;
            type State = CellState;
            type Seed = black_hole_sun::CellInit;
            type Flow = BackwardOperationPrimordium<$op>;
        }
        impl Observable for $cell {
            type Observation = NoopObservation;
        }
        impl Perturbable for $cell {
            type Perturbation = NoopPerturbation;
        }
        impl OperationNode<$op> for $cell {}
    };
}

operation_cell!(StemCell, U0, StemOp);
operation_cell!(Stage1Cell, U1, Stage1Op);
operation_cell!(Stage2Cell, U2, Stage2Op);
operation_cell!(Stage3Cell, U3, Stage3Op);
operation_cell!(Stage4Cell, U4, Stage4Op);
operation_cell!(HeadCell, U5, HeadOp);

pub type CorgiGraph = list![
    Unary<U0, StemCell, BackwardTypedEdges<list![Edge<U1, Stage1Op>]>, StemOp>,
    Unary<U1, Stage1Cell, BackwardTypedEdges<list![Edge<U2, Stage2Op>]>, Stage1Op>,
    Unary<U2, Stage2Cell, BackwardTypedEdges<list![Edge<U3, Stage3Op>]>, Stage2Op>,
    Unary<U3, Stage3Cell, BackwardTypedEdges<list![Edge<U4, Stage4Op>]>, Stage3Op>,
    Unary<U4, Stage4Cell, BackwardTypedEdges<list![Edge<U5, HeadOp>]>, Stage4Op>,
    Unary<U5, HeadCell, BackwardTypedEdges<list![]>, HeadOp>
];

#[derive(Flow)]
pub struct Generator(Step<GenerateImage>, Step<AugmentImage>);

pub struct GenerateImage;
#[jungle::action]
impl Action for GenerateImage {
    type Effect = GenerateImageEffect;
    type Input = ();
    type Output = ArtifactDelivery<Image>;
    fn emit(_state: &PipelineBackwardState, _input: ()) {}
    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("image generator failed: {e}")))
    }
}

pub struct GenerateImageEffect;
#[jungle::effect(id = 211)]
impl<J: VoidOps> Effect<J> for GenerateImageEffect {
    type In = ();
    type Out = ArtifactDelivery<Image>;
    type Err = String;
    fn effect(jungle: &J, _input: ()) -> impl Future<Output = Result<Self::Out, String>> + Send {
        toy_common::dataset::generate_training_image::<J, StemOp>(jungle)
    }
}

pub struct AugmentImage;
#[jungle::action]
impl Action for AugmentImage {
    type Effect = AugmentImageEffect;
    type Input = ArtifactDelivery<Image>;
    type Output = ArtifactDelivery<Image>;

    fn emit(_state: &PipelineBackwardState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("image augmentation failed: {error}")))
    }
}

pub struct AugmentImageEffect;
#[jungle::effect(id = 213)]
impl<J: VoidOps> Effect<J> for AugmentImageEffect {
    type In = ArtifactDelivery<Image>;
    type Out = ArtifactDelivery<Image>;
    type Err = String;

    fn effect(
        jungle: &J,
        delivery: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        toy_common::dataset::augment_image::<J, StemOp>(jungle, delivery)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepMetrics {
    pub step: usize,
    pub mean_loss: f32,
    pub accuracy: f32,
}

#[derive(Flow)]
pub struct LogPolicy(Step<LogStep>, Step<UnifiedCheckpoint>);

pub struct LogStep;
#[jungle::action]
impl Action for LogStep {
    type Effect = LogStepEffect;
    type Input = PipelineStepResult;
    type Output = StepMetrics;
    fn emit(_state: &PipelineBackwardState, input: Self::Input) -> Self::Input {
        input
    }
    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<StepMetrics, Failure> {
        let metrics =
            output.map_err(|e| Failure::Message(format!("training policy failed: {e}")))?;
        COMPLETED_STEPS.fetch_add(1, Ordering::Release);
        Ok(metrics)
    }
}

/// Reconstructs a unified checkpoint from all six stage checkpoints.
pub struct UnifiedCheckpoint;
#[jungle::action]
impl Action for UnifiedCheckpoint {
    type Effect = UnifiedCheckpointEffect;
    type Input = StepMetrics;
    type Output = StepMetrics;

    fn emit(_state: &PipelineBackwardState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("unified checkpoint failed: {error}")))
    }
}

pub struct UnifiedCheckpointEffect;
#[jungle::effect(id = 214)]
impl<J> Effect<J> for UnifiedCheckpointEffect {
    type In = StepMetrics;
    type Out = StepMetrics;
    type Err = String;

    fn effect(
        _jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let settings = unified_checkpoint_settings();
            if settings.every == 0 || input.step == 0 || input.step % settings.every != 0 {
                return Ok(input);
            }
            let directory = settings
                .directory
                .ok_or_else(|| "unified checkpoint directory is not configured".to_owned())?;
            for _ in 0..1_000 {
                let attempt_directory = directory.clone();
                let unified = tokio::task::spawn_blocking(move || {
                    crate::op::unify_checkpoint_shards(&attempt_directory, input.step)
                })
                .await
                .map_err(|error| format!("unified checkpoint task failed: {error}"))??;
                if unified {
                    return Ok(input);
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            Err(format!(
                "timed out waiting for operation shards for step {}",
                input.step
            ))
        }
    }
}

pub struct LogStepEffect;
pub static COMPLETED_STEPS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 212)]
impl<J: VoidOps> Effect<J> for LogStepEffect {
    type In = PipelineStepResult;
    type Out = StepMetrics;
    type Err = String;
    fn effect(
        jungle: &J,
        step: Self::In,
    ) -> impl Future<Output = Result<Self::Out, String>> + Send {
        async move {
            let mut loss = 0f32;
            let mut correct = 0usize;
            for delivery in &step.outputs {
                let decoded = jungle
                    .receive_emission::<HeadOp, ()>(delivery.emission_id.clone())
                    .await?;
                let logits = decoded.first_tensor()?.to_f32()?;
                if logits.len() != BATCH_SIZE * 2 {
                    return Err(
                        "classifier output does not contain one pair of logits per image".into(),
                    );
                }
                for (index, label) in decoded.metadata.dataset_labels.iter().enumerate() {
                    let offset = index * 2;
                    let target = usize::from(!matches!(
                        *label,
                        corgi_fwd::PEMBROKE_LABEL | corgi_fwd::CARDIGAN_LABEL
                    ));
                    let max = logits[offset..offset + 2]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let normalizer = logits[offset..offset + 2]
                        .iter()
                        .map(|v| (*v - max).exp())
                        .sum::<f32>();
                    loss += normalizer.ln() + max - logits[offset + target];
                    correct +=
                        usize::from(usize::from(logits[offset + 1] > logits[offset]) == target);
                }
            }
            let n = (step.outputs.len() * BATCH_SIZE).max(1) as f32;
            let metrics = StepMetrics {
                step: step.step,
                mean_loss: loss / n,
                accuracy: correct as f32 / n,
            };
            tracing::warn!(
                step = metrics.step,
                mean_loss = metrics.mean_loss,
                accuracy = metrics.accuracy,
                "corgi-bwd training step"
            );
            Ok(metrics)
        }
    }
}

pub type CorgiSun<const M: usize> =
    <CorgiGraph as BlackHole>::Sun<PipelineBackward<Generator, StemOp, HeadOp, LogPolicy, (), M>>;

pub struct CorgiBackward<const M: usize>;
impl<const M: usize> Animal for CorgiBackward<M> {
    type Id = Id<U6>;
    type Generation = U0;
    type State = PipelineBackwardState;
    type Seed = ();
    type Flow = CorgiSun<M>;
}
impl<const M: usize> Observable for CorgiBackward<M> {
    type Observation = NoopObservation;
}
impl<const M: usize> Perturbable for CorgiBackward<M> {
    type Perturbation = NoopPerturbation;
}
