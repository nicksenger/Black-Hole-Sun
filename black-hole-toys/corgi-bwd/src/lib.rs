//! Pipeline-parallel ResNet-18 training for a binary corgi identifier.
#![allow(clippy::manual_async_fn)]

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::topology::{BackwardTypedEdges, Edge, Unary};
use black_hole_sun::{
    ArtifactDelivery, ArtifactRef, BackwardOperationPrimordium, Emission, OperationNode,
    PipelineBackward, PipelineBackwardState, PipelineEpochResult, VoidOps,
};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use typenum::consts::{U0, U1, U2, U3, U4, U5, U6};

pub use corgi_fwd::{
    build_stage1, build_stage2, build_stage3, build_stage4, build_trainable_stem, generate_image,
    pool_stage4, HeadOp, Image, Logits, SampleMetadata, Stage1Op, Stage2Op, Stage3Op, Stage4Op,
    StemOp, DATASET_SAMPLES,
};

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
        generate_image(jungle)
    }
}

#[async_trait]
pub trait RawArtifactOps: VoidOps {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String>;
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
impl<J: RawArtifactOps> Effect<J> for LogEpochEffect {
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
                let bytes = jungle.download_raw(delivery.emission_id.id()).await?;
                let emission: Emission<SampleMetadata, Logits> =
                    postcard::from_bytes(&bytes).map_err(|e| e.to_string())?;
                let output = jungle.receive_raw_artifact(&emission.output_id).await?;
                let decoded =
                    black_hole_sun::decode_output::<HeadOp>(&output).map_err(|e| e.to_string())?;
                let logits = decoded.tensors[0]
                    .data
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
                    .collect::<Vec<_>>();
                let target = usize::from(!matches!(emission.metadata.dataset_label, 111 | 112));
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let normalizer = logits.iter().map(|v| (*v - max).exp()).sum::<f32>();
                loss += normalizer.ln() + max - logits[target];
                correct += usize::from(usize::from(logits[1] > logits[0]) == target);
            }
            let n = epoch.outputs.len().max(1) as f32;
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
