//! Cells, the TwoSidedZo Sun flow, and the loss policy.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::ops::{VoidInferOps, VoidOps};
use black_hole_sun::programs::two_sided_zo::{SunState, TwoSidedZo};
use black_hole_sun::topology::{Edge, TypedEdges, Unary};
use black_hole_sun::{AtomError, InferenceOutput, OperationNode, Potentiation, Transmission};
use corgi_fwd::contracts::{HeadOp, Stage1Op, Stage2Op, Stage3Op, Stage4Op, StemOp};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use tracing::info;
use typenum::consts::{U0, U1, U2, U3, U4, U5, U6};

pub static OPTIMIZED_EPOCHS: AtomicUsize = AtomicUsize::new(0);

macro_rules! operation_cell {
    ($cell:ident, $id:ty, $op:ty) => {
        pub struct $cell;
        impl Animal for $cell {
            type Id = Id<$id>;
            type Generation = U0;
            type State = CellState;
            type Seed = black_hole_sun::CellInit;
            type Flow = black_hole_sun::OperationPrimordium<$op>;
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

pub struct StemCell;
impl Animal for StemCell {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = black_hole_sun::CellInit;
    type Flow = black_hole_sun::OperationPrimordium<StemOp>;
}
impl Observable for StemCell {
    type Observation = NoopObservation;
}
impl Perturbable for StemCell {
    type Perturbation = NoopPerturbation;
}
impl OperationNode<StemOp> for StemCell {}

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
    type Output = (Transmission, Transmission);

    fn emit(_state: &SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("image generator failed: {error}")))
    }
}

pub struct GenerateImageEffect;
#[jungle::effect(id = 203)]
impl<J: VoidInferOps> Effect<J> for GenerateImageEffect {
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;

    #[allow(clippy::manual_async_fn)]
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let delivery = toy_common::dataset::generate_image::<J, StemOp>(jungle)
                .await
                .map_err(AtomError::Upload)?;
            let propagation = Transmission::Propagation {
                emission_id: black_hole_sun::ObjectRef::new(delivery.emission_id.id()),
                recv: black_hole_sun::ObjectId::nil(),
                send: black_hole_sun::ObjectId::nil(),
            };
            Ok((propagation.clone(), propagation))
        }
    }
}

#[derive(Flow)]
pub struct Policy(Step<ComputeLoss>);

pub struct ComputeLoss;
#[jungle::action]
impl Action for ComputeLoss {
    type Effect = ComputeLossEffect;
    type Input = [(Transmission, Transmission); 1];
    type Output = Potentiation;

    fn emit(_state: &SunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("corgi-zo loss policy failed: {error}")))
    }
}

pub struct ComputeLossEffect;
#[jungle::effect(id = 204)]
impl<J: VoidInferOps> Effect<J> for ComputeLossEffect {
    type In = [(Transmission, Transmission); 1];
    type Out = Potentiation;
    type Err = AtomError;

    #[allow(clippy::manual_async_fn)]
    fn effect(
        jungle: &J,
        [(up, down)]: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let loss_up = classification_loss(jungle, &up).await?;
            let loss_down = classification_loss(jungle, &down).await?;
            let epoch = OPTIMIZED_EPOCHS.fetch_add(1, Ordering::AcqRel) + 1;
            info!(epoch, loss_up, loss_down, "corgi-zo optimization losses");
            Ok(Potentiation {
                loss_up,
                loss_down,
                seed: epoch as u64,
            })
        }
    }
}

async fn classification_loss<J: VoidInferOps>(
    jungle: &J,
    transmission: &Transmission,
) -> Result<f32, AtomError> {
    let emission_id = match transmission {
        Transmission::Propagation { emission_id, .. } => *emission_id,
        Transmission::Potentiation { .. } => {
            return Err(AtomError::Inference(
                "expected propagation at classifier sink".into(),
            ))
        }
    };
    let decoded = jungle
        .receive_emission::<HeadOp, InferenceOutput>(emission_id)
        .await
        .map_err(AtomError::Download)?;
    let logits = decoded
        .first_tensor()
        .and_then(|raw| raw.to_f32())
        .map_err(|error| AtomError::Inference(error.to_string()))?;
    if logits.len() != 2 {
        return Err(AtomError::Inference(
            "classifier output is not two f32 logits".into(),
        ));
    }
    let target = if matches!(
        decoded.metadata.dataset_label,
        corgi_fwd::PEMBROKE_LABEL | corgi_fwd::CARDIGAN_LABEL
    ) {
        0
    } else {
        1
    };
    let max_logit = logits[0].max(logits[1]);
    let log_sum_exp =
        max_logit + ((logits[0] - max_logit).exp() + (logits[1] - max_logit).exp()).ln();
    Ok(log_sum_exp - logits[target])
}

pub type CorgiSun = <CorgiGraph as BlackHole>::Sun<TwoSidedZo<Generator, Policy, 1>>;

pub struct CorgiZo;
impl Animal for CorgiZo {
    type Id = Id<U6>;
    type Generation = U0;
    type State = SunState;
    type Seed = ();
    type Flow = CorgiSun;
}
impl Observable for CorgiZo {
    type Observation = NoopObservation;
}
impl Perturbable for CorgiZo {
    type Perturbation = NoopPerturbation;
}
