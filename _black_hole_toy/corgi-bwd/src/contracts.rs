//! Cells, the backward Sun flow, and the training policy.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::topology::{BackwardTypedEdges, Edge, Unary};
use black_hole_sun::{
    ArtifactDelivery, BackwardOperationPrimordium, OperationNode, PipelineBackward,
    PipelineBackwardState, PipelineEpochResult, VoidOps,
};
use corgi_fwd::contracts::{HeadOp, Image, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use toy_common::dataset::BATCH_SIZE;
use typenum::consts::{U0, U1, U2, U3, U4, U5, U6};

/// Micro-batches per pipeline epoch.
pub const MICRO_BATCHES: usize = 8;

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
pub struct Generator(Step<GenerateImage>);

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub mean_loss: f32,
    pub accuracy: f32,
}

#[derive(Flow)]
pub struct LogPolicy(Step<LogEpoch>);

pub struct LogEpoch;
#[jungle::action]
impl Action for LogEpoch {
    type Effect = LogEpochEffect;
    type Input = PipelineEpochResult;
    type Output = ();
    fn emit(_state: &PipelineBackwardState, input: Self::Input) -> Self::Input {
        input
    }
    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|e| Failure::Message(format!("training policy failed: {e}")))?;
        COMPLETED_EPOCHS.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

pub struct LogEpochEffect;
pub static COMPLETED_EPOCHS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 212)]
impl<J: VoidOps> Effect<J> for LogEpochEffect {
    type In = PipelineEpochResult;
    type Out = EpochMetrics;
    type Err = String;
    fn effect(
        jungle: &J,
        epoch: Self::In,
    ) -> impl Future<Output = Result<Self::Out, String>> + Send {
        async move {
            let mut loss = 0f32;
            let mut correct = 0usize;
            for delivery in &epoch.outputs {
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
            let n = (epoch.outputs.len() * BATCH_SIZE).max(1) as f32;
            let metrics = EpochMetrics {
                epoch: epoch.epoch,
                mean_loss: loss / n,
                accuracy: correct as f32 / n,
            };
            tracing::warn!(
                epoch = metrics.epoch,
                mean_loss = metrics.mean_loss,
                accuracy = metrics.accuracy,
                "corgi-bwd training epoch"
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
