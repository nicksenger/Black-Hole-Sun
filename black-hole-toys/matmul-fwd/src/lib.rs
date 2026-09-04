//! A small forward-only tensor pipeline.
//!
//! The example deliberately keeps the tensor and operations simple so the
//! topology and shape transition are easy to inspect:
//!
//! ```text
//! Generator (2x3) -> Matmul (2x4) -> Scale -> ReLU -> LogPolicy
//! ```
//!
//! The three operation cells are a statically typed DAG.  In particular, the
//! `TypedEdges` declarations make each downstream input contract part of the
//! graph definition, while the enclosing flow is assembled through the
//! canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use black_hole_sun::black_hole_spec::{
    glowstick::Shape2, SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::forward::ForwardSunState;
use black_hole_sun::topology::{Edge, TypedEdges, Unary};
use black_hole_sun::{
    ArtifactDelivery, ArtifactRef, CellInit, ContractId, DtypeConstraint, Emission,
    ForwardOnlyWithPolicy, ForwardOperationPrimordium, ObjectId, ObjectRef, OperationNode,
    RawTensor, TensorDtype, VoidOps,
};
use jungle_sdk::list;
use jungle_sdk::prelude::*;
use tracing::info;
use typenum::{U0, U1, U2, U3, U4};

/// The generator's 2x3 input matrix.
pub struct InputMatrixPort;

impl TensorPortSpec for InputMatrixPort {
    type Shape = Shape2<U2, U3>;

    const NAME: &'static str = "input_matrix";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

/// The 2x4 matrix produced by the matmul and carried by downstream cells.
pub struct ProductMatrixPort;

impl TensorPortSpec for ProductMatrixPort {
    type Shape = Shape2<U2, U4>;

    const NAME: &'static str = "product_matrix";
    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

pub type InputMatrix = SingleTensorSpec<InputMatrixPort>;
pub type ProductMatrix = SingleTensorSpec<ProductMatrixPort>;

/// Contract for the first cell, which represents a matrix multiplication.
pub struct Matmul;

impl TensorContract for Matmul {
    type Input = InputMatrix;
    type Output = ProductMatrix;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x6d61_746d_756c_2d66_7764_2d303031);
    const VERSION: u32 = 1;
}

/// Contract for the second cell, which scales the matrix.
pub struct Scale;

impl TensorContract for Scale {
    type Input = ProductMatrix;
    type Output = ProductMatrix;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x7363_616c_652d_6677_642d_3030_3100);
    const VERSION: u32 = 1;
}

/// Contract for the final cell, which applies a ReLU.
pub struct Relu;

impl TensorContract for Relu {
    type Input = ProductMatrix;
    type Output = ProductMatrix;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x7265_6c75_2d66_7764_2d30_3031_0000);
    const VERSION: u32 = 1;
}

pub struct MatmulCell;

impl Animal for MatmulCell {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Matmul>;
}

impl Observable for MatmulCell {
    type Observation = NoopObservation;
}

impl Perturbable for MatmulCell {
    type Perturbation = NoopPerturbation;
}

impl OperationNode<Matmul> for MatmulCell {}

pub struct ScaleCell;

impl Animal for ScaleCell {
    type Id = Id<U1>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Scale>;
}

impl Observable for ScaleCell {
    type Observation = NoopObservation;
}

impl Perturbable for ScaleCell {
    type Perturbation = NoopPerturbation;
}

impl OperationNode<Scale> for ScaleCell {}

pub struct ReluCell;

impl Animal for ReluCell {
    type Id = Id<U2>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Relu>;
}

impl Observable for ReluCell {
    type Observation = NoopObservation;
}

impl Perturbable for ReluCell {
    type Perturbation = NoopPerturbation;
}

impl OperationNode<Relu> for ReluCell {}

/// Three-cell matrix pipeline, with compile-time checked operation edges.
pub type MatmulGraph = list![
    Unary<U0, MatmulCell, TypedEdges<list![Edge<U1, Scale>]>, Matmul>,
    Unary<U1, ScaleCell, TypedEdges<list![Edge<U2, Relu>]>, Scale>,
    Unary<U2, ReluCell, TypedEdges<list![]>, Relu>
];

/// Repeatedly emits the same 2x3 matrix as a typed input artifact.
#[derive(Flow)]
pub struct Generator(Step<GenerateTensor>);

pub struct GenerateTensor;

#[jungle::action]
impl Action for GenerateTensor {
    type Effect = GenerateTensorEffect;
    type Input = ();
    type Output = ArtifactDelivery<InputMatrix>;

    fn emit(_state: &ForwardSunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("tensor generator failed: {error}")))
    }
}

pub struct GenerateTensorEffect;

#[jungle::effect(id = 101)]
impl<J: VoidOps> Effect<J> for GenerateTensorEffect {
    type In = ();
    type Out = ArtifactDelivery<InputMatrix>;
    type Err = String;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let input = black_hole_sun::encode_input::<Matmul>(
                &[RawTensor {
                    name: "input_matrix".to_string(),
                    dtype: TensorDtype::F32,
                    shape: vec![2, 3],
                    data: [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                }],
                &(),
            )
            .map_err(|error| error.to_string())?;
            let tensor_id = jungle.upload_to_void(input).await?;

            let emission = Emission::<(), InputMatrix> {
                metadata: (),
                output_id: ArtifactRef::committed(ObjectRef::new(tensor_id)),
            };
            let emission_id = jungle
                .upload_to_void(
                    postcard::to_allocvec(&emission).map_err(|error| error.to_string())?,
                )
                .await?;

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;

            Ok(ArtifactDelivery {
                emission_id: ObjectRef::new(emission_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            })
        }
    }
}

/// Logs the final tensor artifact after each forward pass.
#[derive(Flow)]
pub struct LogPolicy(Step<LogTensor>);

pub struct LogTensor;

/// Raw artifact resolution used by the policy so committed and streamed
/// tensor references are handled through the same Void path.
#[async_trait]
pub trait RawArtifactOps: VoidOps {
    async fn receive_raw_artifact<T: Send>(
        &self,
        reference: &ArtifactRef<T>,
    ) -> Result<Vec<u8>, String>;
}

#[jungle::action]
impl Action for LogTensor {
    type Effect = LogTensorEffect;
    type Input = ArtifactDelivery<ProductMatrix>;
    type Output = ();

    fn emit(_state: &ForwardSunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut ForwardSunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("tensor policy failed: {error}")))
    }
}

pub struct LogTensorEffect;

/// Number of completed policy invocations observed by the runnable example.
pub static LOGGED_OUTPUTS: AtomicUsize = AtomicUsize::new(0);

#[jungle::effect(id = 102)]
impl<J: RawArtifactOps> Effect<J> for LogTensorEffect {
    type In = ArtifactDelivery<ProductMatrix>;
    type Out = ();
    type Err = String;

    fn effect(
        jungle: &J,
        delivery: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let emission_bytes = jungle.download_raw(delivery.emission_id.id()).await?;
            let emission: Emission<(), ProductMatrix> =
                postcard::from_bytes(&emission_bytes).map_err(|error| error.to_string())?;
            let tensor_bytes = jungle.receive_raw_artifact(&emission.output_id).await?;
            let tensor = black_hole_sun::decode_output::<Relu>(&tensor_bytes)
                .map_err(|error| error.to_string())?;
            info!(tensor = ?tensor.tensors, "matmul-fwd output");
            LOGGED_OUTPUTS.fetch_add(1, Ordering::Release);
            Ok(())
        }
    }
}

/// The complete flow produced by the redesigned `BlackHole::Sun` compiler.
pub type MatmulSun =
    <MatmulGraph as BlackHole>::Sun<ForwardOnlyWithPolicy<Generator, Matmul, Relu, LogPolicy>>;

/// Top-level Jungle animal for the example Sun.
pub struct MatmulForward;

impl Animal for MatmulForward {
    type Id = Id<U3>;
    type Generation = U0;
    type State = ForwardSunState;
    type Seed = ();
    type Flow = MatmulSun;
}

impl Observable for MatmulForward {
    type Observation = NoopObservation;
}

impl Perturbable for MatmulForward {
    type Perturbation = NoopPerturbation;
}
