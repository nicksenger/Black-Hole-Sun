//! A small forward-only tensor pipeline.
//!
//! The example deliberately keeps the tensor and operations simple so the
//! topology is easy to inspect:
//!
//! ```text
//! Generator -> Matmul -> Scale -> ReLU -> LogPolicy
//! ```
//!
//! The three operation cells are a statically typed DAG.  In particular, the
//! `TypedEdges` declarations make each downstream input contract part of the
//! graph definition, while the enclosing flow is assembled through the
//! canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.

use std::future::Future;

use black_hole_sun::black_hole_spec::{
    glowstick::Shape2, SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_sun::cell::CellState;
use black_hole_sun::compile::BlackHole;
use black_hole_sun::forward::ForwardSunState;
use black_hole_sun::topology::{Edge, TypedEdges, Unary};
use black_hole_sun::{
    ArtifactDelivery, ArtifactRef, CellInit, ContractId, DimensionDescriptor, DtypeConstraint,
    Emission, ForwardOnlyWithPolicy, ForwardOperationPrimordium, ObjectId, ObjectRef,
    OperationNode, RawTensor, TensorDtype, VoidOps,
};
use jungle_sdk::prelude::*;
use tracing::info;
use typenum::{U0, U1, U2, U3};
use typosaurus::collections::list::{Empty, List};
use typosaurus::list;

/// The one tensor carried by this toy pipeline: a 2x2 matrix of f32 values.
pub struct MatrixPort;

impl TensorPortSpec for MatrixPort {
    type Shape = Shape2<U2, U2>;

    const NAME: &'static str = "matrix";

    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![
            DimensionDescriptor::Static(2),
            DimensionDescriptor::Static(2),
        ]
    }

    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

pub type Matrix = SingleTensorSpec<MatrixPort>;

/// Contract for the first cell, which represents a matrix multiplication.
pub struct Matmul;

impl TensorContract for Matmul {
    type Input = Matrix;
    type Output = Matrix;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x6d61_746d_756c_2d66_7764_2d303031);
    const VERSION: u32 = 1;
}

/// Contract for the second cell, which scales the matrix.
pub struct Scale;

impl TensorContract for Scale {
    type Input = Matrix;
    type Output = Matrix;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x7363_616c_652d_6677_642d_3030_3100);
    const VERSION: u32 = 1;
}

/// Contract for the final cell, which applies a ReLU.
pub struct Relu;

impl TensorContract for Relu {
    type Input = Matrix;
    type Output = Matrix;
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

impl OperationNode<Matmul> for MatmulCell {}

pub struct ScaleCell;

impl Animal for ScaleCell {
    type Id = Id<U1>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Scale>;
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

impl OperationNode<Relu> for ReluCell {}

type ReluNode = List<(Unary<U2, ReluCell, TypedEdges<Empty>, Relu>, Empty)>;
type ScaleNode = List<(
    Unary<U1, ScaleCell, TypedEdges<list![Edge<U2, Relu>]>, Scale>,
    ReluNode,
)>;

/// Three-cell matrix pipeline, with compile-time checked operation edges.
pub type MatmulGraph = List<(
    Unary<U0, MatmulCell, TypedEdges<list![Edge<U1, Scale>]>, Matmul>,
    ScaleNode,
)>;

/// Repeatedly emits the same 2x2 matrix as a typed input artifact.
#[derive(Flow)]
pub struct Generator(Step<GenerateTensor>);

pub struct GenerateTensor;

#[jungle::action]
impl Action for GenerateTensor {
    type Effect = GenerateTensorEffect;
    type Input = ();
    type Output = ArtifactDelivery<Matrix>;

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
    type Out = ArtifactDelivery<Matrix>;
    type Err = String;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let input = black_hole_sun::encode_input::<Matmul>(
                &[RawTensor {
                    name: "matrix".to_string(),
                    dtype: TensorDtype::F32,
                    shape: vec![2, 2],
                    data: [1.0_f32, 2.0, 3.0, 4.0]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                }],
                &(),
            )
            .map_err(|error| error.to_string())?;
            let tensor_id = jungle.upload_to_void(input).await?;

            let emission = Emission::<(), Matrix> {
                metadata: (),
                output_id: ArtifactRef::committed(ObjectRef::new(tensor_id)),
            };
            let emission_id = jungle
                .upload_to_void(
                    postcard::to_allocvec(&emission).map_err(|error| error.to_string())?,
                )
                .await?;

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

#[jungle::action]
impl Action for LogTensor {
    type Effect = LogTensorEffect;
    type Input = ArtifactDelivery<Matrix>;
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

#[jungle::effect(id = 102)]
impl<J: VoidOps> Effect<J> for LogTensorEffect {
    type In = ArtifactDelivery<Matrix>;
    type Out = ();
    type Err = String;

    fn effect(
        jungle: &J,
        delivery: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let emission_bytes = jungle.download_raw(delivery.emission_id.id()).await?;
            let emission: Emission<(), Matrix> =
                postcard::from_bytes(&emission_bytes).map_err(|error| error.to_string())?;
            let tensor_bytes = jungle.download_raw(emission.output_id.object_id()).await?;
            let tensor = black_hole_sun::decode_output::<Relu>(&tensor_bytes)
                .map_err(|error| error.to_string())?;
            info!(tensor = ?tensor.tensors, "matmul-fwd output");
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
