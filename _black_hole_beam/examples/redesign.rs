//! Visual acceptance example for the generic tensor-operation redesign.
//!
//! Run it from the workspace root:
//!
//! ```text
//! cargo run -p black-hole-beam --example redesign
//! ```
//!
//! The Beam window displays a forward-only, non-Qwen tensor pipeline compiled
//! through the canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.

use black_hole_beam::BeamBuilder;
use black_hole_contract::{
    glowstick::{Dyn, Shape2},
    SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_flux::{
    BlackHole, CellInit, CellState, Edge, ForwardOnly, ForwardOperationPrimordium, OperationNode,
    Primordium, TypedEdges, Unary,
};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use jungle_sdk::typosaurus::collections::list::{Empty, List};
use jungle_sdk::{Animal, Id};
use typenum::{U0, U1, U2, U3, U8};

struct Batch;

struct FeaturePort;

impl TensorPortSpec for FeaturePort {
    type Shape = Shape2<Dyn<Batch>, U8>;

    const NAME: &'static str = "features";

    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![
            DimensionDescriptor::Symbolic("batch".into()),
            DimensionDescriptor::Static(8),
        ]
    }

    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

type FeatureBatch = SingleTensorSpec<FeaturePort>;

struct Normalize;

impl TensorContract for Normalize {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x4e4f_524d_414c_495a_45);
    const VERSION: u32 = 1;
}

struct Encode;

impl TensorContract for Encode {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x454e_434f_4445);
    const VERSION: u32 = 1;
}

struct Classify;

impl TensorContract for Classify {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x434c_4153_5349_4659);
    const VERSION: u32 = 1;
}

struct NormalizeFeatures;

impl Animal for NormalizeFeatures {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Normalize>;
}

impl OperationNode<Normalize> for NormalizeFeatures {}

struct EncodeFeatures;

impl Animal for EncodeFeatures {
    type Id = Id<U1>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Encode>;
}

impl OperationNode<Encode> for EncodeFeatures {}

struct ClassifyFeatures;

impl Animal for ClassifyFeatures {
    type Id = Id<U2>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Classify>;
}

impl OperationNode<Classify> for ClassifyFeatures {}

type ClassifyNode = List<(Unary<U2, ClassifyFeatures, Empty, Classify>, Empty)>;
type EncodeNode = List<(
    Unary<U1, EncodeFeatures, TypedEdges<List<(Edge<U2, Classify>, Empty)>>, Encode>,
    ClassifyNode,
)>;
type Topology = List<(
    Unary<U0, NormalizeFeatures, TypedEdges<List<(Edge<U1, Encode>, Empty)>>, Normalize>,
    EncodeNode,
)>;

type RedesignSun = <Topology as BlackHole>::Sun<ForwardOnly<Primordium, Normalize>>;

struct RedesignDemo;

impl Animal for RedesignDemo {
    type Id = Id<U3>;
    type Generation = U0;
    type State = ();
    type Seed = ();
    type Flow = RedesignSun;
}

fn main() -> iced::Result {
    println!(
        "Opening the typed forward-only pipeline: NormalizeFeatures -> EncodeFeatures -> ClassifyFeatures"
    );
    BeamBuilder::new()
        .title("REDESIGN VERIFIED: typed ForwardOnly tensor pipeline")
        .window_size(1100.0, 700.0)
        .microdot_layout()
        .view::<RedesignDemo>()
}
