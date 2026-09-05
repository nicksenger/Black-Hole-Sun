//! Fusion stages, hybrid topology, source, and training policy.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use black_hole_sun::compile::BlackHole;
use black_hole_sun::topology::{
    BackwardTypedEdges, Binary, Edge, SunAppearance, SunEdgeAppearance,
};
use black_hole_sun::{
    ArtifactDelivery, DataParallelBackwardOperationPrimordium, DataParallelOperationState,
    DataParallelPipelineBackward, FusionSeed, OperationNode, PipelineBackwardState,
    PipelineStepResult, VoidOps,
};
use corgi_fwd::spec::{HeadOp, Image, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use toy_common::dataset::BATCH_SIZE;
use typenum::consts::{U0, U1, U10, U11, U12, U2, U3, U4, U5, U6, U7, U8, U9};

/// Local micro-batches processed by each pipeline replica per optimizer step.
pub const MICRO_BATCHES: usize = 8;
pub const DATA_PARALLEL_REPLICAS: usize = 2;

macro_rules! fusion_operation {
    ($cell:ident, $id:ty, $op:ty) => {
        pub struct $cell;
        impl Animal for $cell {
            type Id = Id<$id>;
            type Generation = U0;
            type State = DataParallelOperationState;
            type Seed = FusionSeed;
            type Flow = DataParallelBackwardOperationPrimordium<$op>;
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

fusion_operation!(StemFusion, U0, StemOp);
fusion_operation!(Stage1Fusion, U2, Stage1Op);
fusion_operation!(Stage2Fusion, U4, Stage2Op);
fusion_operation!(Stage3Fusion, U6, Stage3Op);
fusion_operation!(Stage4Fusion, U8, Stage4Op);
fusion_operation!(HeadFusion, U10, HeadOp);

/// Two side-by-side six-stage pipelines. Each `Binary` is the vertical
/// data-parallel group for one model depth; its two ports preserve lane
/// identity while its Fusion-style cell joins the optimizer step. The Beam
/// appearance below expands those two lanes into separate display nodes.
pub type CorgiParallelGraph = list![
    Binary<
        U0,
        U1,
        StemFusion,
        BackwardTypedEdges<list![Edge<U2, Stage1Op>, Edge<U3, Stage1Op>]>,
        StemOp
    >,
    Binary<
        U2,
        U3,
        Stage1Fusion,
        BackwardTypedEdges<list![Edge<U4, Stage2Op>, Edge<U5, Stage2Op>]>,
        Stage1Op
    >,
    Binary<
        U4,
        U5,
        Stage2Fusion,
        BackwardTypedEdges<list![Edge<U6, Stage3Op>, Edge<U7, Stage3Op>]>,
        Stage2Op
    >,
    Binary<
        U6,
        U7,
        Stage3Fusion,
        BackwardTypedEdges<list![Edge<U8, Stage4Op>, Edge<U9, Stage4Op>]>,
        Stage3Op
    >,
    Binary<
        U8,
        U9,
        Stage4Fusion,
        BackwardTypedEdges<list![Edge<U10, HeadOp>, Edge<U11, HeadOp>]>,
        Stage4Op
    >,
    Binary<U10, U11, HeadFusion, BackwardTypedEdges<list![]>, HeadOp>
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
        output.map_err(|error| Failure::Message(format!("image generator failed: {error}")))
    }
}

pub struct GenerateImageEffect;
#[jungle::effect(id = 221)]
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
#[jungle::effect(id = 222)]
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

pub struct LogStep;
#[jungle::action]
impl Action for LogStep {
    type Effect = LogStepEffect;
    type Input = PipelineStepResult;
    type Output = ();
    fn emit(_state: &PipelineBackwardState, input: Self::Input) -> Self::Input {
        input
    }
    fn absorb(
        _state: &mut PipelineBackwardState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("training policy failed: {error}")))?;
        COMPLETED_STEPS.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

pub struct LogStepEffect;
pub static COMPLETED_STEPS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 223)]
impl<J: VoidOps> Effect<J> for LogStepEffect {
    type In = PipelineStepResult;
    type Out = ();
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
                    .receive_emission::<HeadOp, ()>(delivery.emission_id)
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
                        .map(|value| (*value - max).exp())
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
                replicas = DATA_PARALLEL_REPLICAS,
                "corgi-par training step"
            );
            Ok(())
        }
    }
}

pub type CorgiParallelSun<const M: usize> = <CorgiParallelGraph as BlackHole>::Sun<
    DataParallelPipelineBackward<Generator, StemOp, HeadOp, Step<LogStep>, (), M>,
>;

pub struct CorgiParallel<const M: usize>;
impl<const M: usize> Animal for CorgiParallel<M> {
    type Id = Id<U12>;
    type Generation = U0;
    type State = PipelineBackwardState;
    type Seed = ();
    type Flow = CorgiParallelSun<M>;
}
impl<const M: usize> Observable for CorgiParallel<M> {
    type Observation = ObserveObservation;
}
impl<const M: usize> Perturbable for CorgiParallel<M> {
    type Perturbation = NoopPerturbation;
}

/// Live Black Hole Beam view of the training Sun.
impl<const M: usize> Observe for CorgiParallel<M> {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        expand_replica_appearance(state.appearance())
    }
}

/// The runtime uses one Fusion animal per stage to coordinate its two model
/// replicas. Expose each lane as a node in Beam so the parallel topology is
/// visible without changing the execution topology or journey identities.
fn expand_replica_appearance(mut appearance: SunAppearance) -> SunAppearance {
    let original_nodes = appearance.nodes.clone();
    let replica_ids = original_nodes
        .iter()
        .filter_map(|node| {
            (node.input_ports.len() == 2).then_some((node.id, [node.id, node.input_ports[1]]))
        })
        .collect::<HashMap<_, _>>();

    let mut nodes = Vec::with_capacity(original_nodes.len() + replica_ids.len());
    for node in &original_nodes {
        if let Some(ids) = replica_ids.get(&node.id) {
            let mut first = node.clone();
            first.input_ports = vec![node.input_ports[0]];
            first.id = ids[0];
            nodes.push(first);

            let mut second = node.clone();
            second.input_ports = vec![node.input_ports[1]];
            second.id = ids[1];
            nodes.push(second);
        } else {
            nodes.push(node.clone());
        }
    }

    let original_by_id = original_nodes
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    let edges = appearance
        .edges
        .iter()
        .map(|edge| {
            let lane = original_by_id
                .get(&edge.target)
                .and_then(|node| {
                    node.input_ports
                        .iter()
                        .position(|port| *port == edge.target_port)
                })
                .unwrap_or(0);
            let source = replica_ids
                .get(&edge.source)
                .map(|ids| ids[lane.min(1)])
                .unwrap_or(edge.source);
            let target = replica_ids
                .get(&edge.target)
                .map(|ids| ids[lane.min(1)])
                .unwrap_or(edge.target);
            SunEdgeAppearance {
                source,
                target,
                target_port: edge.target_port,
            }
        })
        .collect();

    appearance.nodes = nodes;
    appearance.edges = edges;
    appearance
}

#[cfg(test)]
mod tests {
    use super::expand_replica_appearance;
    use black_hole_sun::topology::{
        SunAppearance, SunEdgeAppearance, SunNodeAppearance, SunNodeState,
    };
    use uuid::Uuid;

    #[test]
    fn beam_appearance_exposes_both_data_parallel_lanes() {
        let appearance = SunAppearance {
            finalized: true,
            nodes: vec![node(0, vec![0, 1]), node(2, vec![2, 3])],
            edges: vec![
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 2,
                },
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 3,
                },
            ],
            ..SunAppearance::default()
        };

        let expanded = expand_replica_appearance(appearance);

        assert_eq!(
            expanded
                .nodes
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            expanded
                .edges
                .iter()
                .map(|edge| (edge.source, edge.target))
                .collect::<Vec<_>>(),
            vec![(0, 2), (1, 3)]
        );
    }

    fn node(id: u32, input_ports: Vec<u32>) -> SunNodeAppearance {
        SunNodeAppearance {
            id,
            journey_id: Uuid::new_v4(),
            warp_journey_id: Uuid::nil(),
            label: format!("Stage{id}"),
            input_ports,
            state: SunNodeState::Idle,
            state_sequence: 0,
            grad_step: 1,
            operational_state: Default::default(),
            phase_annotation: None,
        }
    }
}
