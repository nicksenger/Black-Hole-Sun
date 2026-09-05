//! Ports, contracts, cells, and the compiled Sun flow for the pipeline.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use black_hole_sun::black_hole_spec::{
    glowstick::{Shape2, Shape4},
    BackwardContract, SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::forward::ForwardSunState;
use black_hole_sun::topology::{Edge, TypedEdges, Unary};
use black_hole_sun::{
    ArtifactDelivery, CellInit, ContractId, DtypeConstraint, ForwardOnlyWithPolicy,
    ForwardOperationPrimordium, OperationNode, TensorDtype, VoidOps,
};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use toy_common::dataset::{SampleMetadata, BATCH_SIZE};
use tracing::warn;
use typenum::consts::{U0, U1, U128, U14, U2, U224, U256, U28, U3, U4, U5, U512, U56, U6, U64, U7};

use crate::model::{CARDIGAN_LABEL, PEMBROKE_LABEL};

pub struct ImagePort;
impl TensorPortSpec for ImagePort {
    type Shape = Shape4<U4, U3, U224, U224>;
    const NAME: &'static str = "image";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct StemPort;
impl TensorPortSpec for StemPort {
    type Shape = Shape4<U4, U64, U56, U56>;
    const NAME: &'static str = "stem";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage1Port;
impl TensorPortSpec for Stage1Port {
    type Shape = Shape4<U4, U64, U56, U56>;
    const NAME: &'static str = "stage1";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage2Port;
impl TensorPortSpec for Stage2Port {
    type Shape = Shape4<U4, U128, U28, U28>;
    const NAME: &'static str = "stage2";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage3Port;
impl TensorPortSpec for Stage3Port {
    type Shape = Shape4<U4, U256, U14, U14>;
    const NAME: &'static str = "stage3";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct Stage4Port;
impl TensorPortSpec for Stage4Port {
    type Shape = Shape4<U4, U512, U7, U7>;
    const NAME: &'static str = "stage4";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub struct LogitsPort;
impl TensorPortSpec for LogitsPort {
    type Shape = Shape2<U4, U2>;
    const NAME: &'static str = "logits";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub type Image = SingleTensorSpec<ImagePort>;
pub type Stem = SingleTensorSpec<StemPort>;
pub type Stage1 = SingleTensorSpec<Stage1Port>;
pub type Stage2 = SingleTensorSpec<Stage2Port>;
pub type Stage3 = SingleTensorSpec<Stage3Port>;
pub type Stage4 = SingleTensorSpec<Stage4Port>;
pub type Logits = SingleTensorSpec<LogitsPort>;

macro_rules! contract {
    ($name:ident, $input:ty, $output:ty, $id:expr) => {
        pub struct $name;
        impl TensorContract for $name {
            type Input = $input;
            type Output = $output;
            type Metadata = SampleMetadata;
            const ID: ContractId = ContractId::from_u128($id);
            const VERSION: u32 = 1;
        }
        impl BackwardContract for $name {
            type OutputGrad = $output;
            type InputGrad = $input;
        }
    };
}

contract!(StemOp, Image, Stem, 0x636f7267692d7374656d2d3030303031);
contract!(Stage1Op, Stem, Stage1, 0x636f7267692d73746167653130303031);
contract!(Stage2Op, Stage1, Stage2, 0x636f7267692d73746167653230303031);
contract!(Stage3Op, Stage2, Stage3, 0x636f7267692d73746167653330303031);
contract!(Stage4Op, Stage3, Stage4, 0x636f7267692d73746167653430303031);
contract!(HeadOp, Stage4, Logits, 0x636f7267692d686561642d3030303031);

pub struct StemCell;
impl Animal for StemCell {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<StemOp>;
}
impl Observable for StemCell {
    type Observation = NoopObservation;
}
impl Perturbable for StemCell {
    type Perturbation = NoopPerturbation;
}
impl OperationNode<StemOp> for StemCell {}

macro_rules! operation_cell {
    ($cell:ident, $id:ty, $op:ty) => {
        pub struct $cell;
        impl Animal for $cell {
            type Id = Id<$id>;
            type Generation = U0;
            type State = CellState;
            type Seed = CellInit;
            type Flow = ForwardOperationPrimordium<$op>;
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
operation_cell!(Stage1Cell, U1, Stage1Op);
operation_cell!(Stage2Cell, U2, Stage2Op);
operation_cell!(Stage3Cell, U3, Stage3Op);
operation_cell!(Stage4Cell, U4, Stage4Op);
operation_cell!(HeadCell, U5, HeadOp);

pub type CorgiGraph = list![
    Unary<U0, StemCell, TypedEdges<list![Edge<U1, Stage1Op>]>, StemOp>,
    Unary<U1, Stage1Cell, TypedEdges<list![Edge<U2, Stage2Op>]>, Stage1Op>,
    Unary<U2, Stage2Cell, TypedEdges<list![Edge<U3, Stage3Op>]>, Stage2Op>,
    Unary<U3, Stage3Cell, TypedEdges<list![Edge<U4, Stage4Op>]>, Stage3Op>,
    Unary<U4, Stage4Cell, TypedEdges<list![Edge<U5, HeadOp>]>, Stage4Op>,
    Unary<U5, HeadCell, TypedEdges<list![]>, HeadOp>
];

#[derive(Flow)]
pub struct Generator(Step<GenerateImage>);

pub struct GenerateImage;
#[jungle::action]
impl Action for GenerateImage {
    type Effect = GenerateImageEffect;
    type Input = ();
    type Output = ArtifactDelivery<Image>;

    fn emit(_state: &ForwardSunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("image generator failed: {error}")))
    }
}

pub struct GenerateImageEffect;

#[jungle::effect(id = 201)]
impl<J: VoidOps> Effect<J> for GenerateImageEffect {
    type In = ();
    type Out = ArtifactDelivery<Image>;
    type Err = String;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        toy_common::dataset::generate_image::<J, StemOp>(jungle)
    }
}

#[derive(Flow)]
pub struct LogPolicy(Step<LogPrediction>);

pub struct LogPrediction;

#[jungle::action]
impl Action for LogPrediction {
    type Effect = LogPredictionEffect;
    type Input = ArtifactDelivery<Logits>;
    type Output = ();

    fn emit(_state: &ForwardSunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("classification policy failed: {error}")))
    }
}

pub struct LogPredictionEffect;

/// Number of completed policy invocations observed by the runnable example.
pub static LOGGED_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 202)]
impl<J: VoidOps> Effect<J> for LogPredictionEffect {
    type In = ArtifactDelivery<Logits>;
    type Out = ();
    type Err = String;

    fn effect(
        jungle: &J,
        delivery: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let output = jungle.receive::<HeadOp>(&delivery).await?;
            let logits = output.first_tensor()?.to_f32()?;
            if logits.len() != BATCH_SIZE * 2 {
                return Err(format!(
                    "classifier output has {} values, expected {}",
                    logits.len(),
                    BATCH_SIZE * 2
                ));
            }
            for (index, label) in output.metadata.dataset_labels.iter().enumerate() {
                let offset = index * 2;
                let prediction = if logits[offset] >= logits[offset + 1] {
                    "corgi"
                } else {
                    "not a corgi"
                };
                let expected = matches!(*label, PEMBROKE_LABEL | CARDIGAN_LABEL);
                warn!(prediction, expected, "corgi-fwd classification");
            }
            LOGGED_OUTPUTS.fetch_add(BATCH_SIZE, Ordering::Release);
            Ok(())
        }
    }
}

/// Keep only one image batch resident while the CPU pipeline consumes it.
pub type CorgiSun =
    <CorgiGraph as BlackHole>::Sun<ForwardOnlyWithPolicy<Generator, StemOp, HeadOp, LogPolicy>>;

pub struct CorgiForward;
impl Animal for CorgiForward {
    type Id = Id<U6>;
    type Generation = U0;
    type State = ForwardSunState;
    type Seed = ();
    type Flow = CorgiSun;
}
impl Observable for CorgiForward {
    type Observation = NoopObservation;
}
impl Perturbable for CorgiForward {
    type Perturbation = NoopPerturbation;
}
