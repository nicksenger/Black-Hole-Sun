//! Visual acceptance example for the generic tensor-operation redesign.
//!
//! Run it from the workspace root:
//!
//! ```text
//! cargo run -p black-hole-beam --example redesign
//! ```
//!
//! The Beam window animates a forward-only, non-Qwen tensor pipeline compiled
//! through the canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.
//! Click any node to inspect its forward-only child flow.

use std::time::{Duration, Instant};

use black_hole_beam::BeamBuilder;
use black_hole_contract::{
    glowstick::{Dyn, Shape2},
    SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_flux::sun::{
    SunAppearance, SunEdgeAppearance, SunNodeAppearance, SunNodeState, SunOperationalState,
    SunState,
};
use black_hole_flux::{
    BlackHole, CellInit, CellState, Edge, ForwardOnly, ForwardOperationPrimordium, OperationNode,
    Primordium, TypedEdges, Unary,
};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use jungle_client::MockClient;
use jungle_sdk::typosaurus::collections::list::{Empty, List};
use jungle_sdk::{Animal, Id, Observe};
use typenum::{U0, U1, U2, U3, U8};
use uuid::Uuid;

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
    type State = SunState;
    type Seed = ();
    type Flow = RedesignSun;
}

impl Observe for RedesignDemo {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

fn demo_appearance(elapsed: Duration) -> SunAppearance {
    let cycle_stage = (elapsed.as_millis() / 1_200 % 5) as usize;
    let cycle = elapsed.as_millis() / (1_200 * 5);
    let node = |id: u32, label: &str, input_ports: Vec<u32>| {
        let position = id as usize + 1;
        let operational_state = if cycle_stage == 0 || position > cycle_stage {
            SunOperationalState::Queued
        } else if position == cycle_stage {
            SunOperationalState::Running
        } else {
            SunOperationalState::Succeeded
        };
        SunNodeAppearance {
            id,
            journey_id: Uuid::from_u128(0x100 + u128::from(id)),
            warp_journey_id: Uuid::nil(),
            label: label.to_string(),
            input_ports,
            state: SunNodeState::Idle,
            state_sequence: (cycle * 5 + cycle_stage as u128) as u64,
            grad_step: 1,
            operational_state,
            phase_annotation: Some("forward".to_string()),
        }
    };

    SunAppearance {
        finalized: true,
        grad_steps: 1,
        nodes: vec![
            node(0, "NormalizeFeatures", vec![0]),
            node(1, "EncodeFeatures", vec![1]),
            node(2, "ClassifyFeatures", vec![2]),
        ],
        edges: vec![
            SunEdgeAppearance {
                source: 0,
                target: 1,
                target_port: 1,
            },
            SunEdgeAppearance {
                source: 1,
                target: 2,
                target_port: 2,
            },
        ],
    }
}

fn main() -> iced::Result {
    let journey_id = Uuid::from_u128(0x5245_4445_5349_474e);
    let started = Instant::now();
    let client = MockClient::builder()
        .on_flow_appearance(move |requested_id| {
            let appearance = (requested_id == journey_id)
                .then(|| postcard::to_allocvec(&demo_appearance(started.elapsed())).unwrap());
            async move { Ok(appearance) }
        })
        .build();

    println!("Opening an animated typed forward pass. Click any node to inspect its child flow.");
    BeamBuilder::new()
        .title("REDESIGN VERIFIED: typed ForwardOnly tensor pipeline")
        .window_size(1100.0, 700.0)
        .microdot_layout()
        .register_static_subpanel_animal::<NormalizeFeatures>()
        .register_static_subpanel_animal::<EncodeFeatures>()
        .register_static_subpanel_animal::<ClassifyFeatures>()
        .view_live::<RedesignDemo>(client, journey_id)
}
