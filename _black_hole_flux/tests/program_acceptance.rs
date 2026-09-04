use black_hole_spec::{glowstick::Shape1, TensorBundleSpec, TensorContract, TensorPortSpec};
use black_hole_flux::{
    BlackHole, CellInit, CellState, CheckpointEvaluate, CompileSun, Edge, ForwardOnly,
    ForwardOperationPrimordium, ForwardSunState, NeutralSunState, OperationNode, TypedEdges, Unary,
};
use black_hole_type::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use jungle_sdk::{Animal, Id, Step};
use jungle_zoo::Noop;
use typenum::{U0, U1, U2, U3};
use typosaurus::{
    collections::list::{Empty, List},
    list,
};

macro_rules! port {
    ($name:ident, $label:literal) => {
        struct $name;
        impl TensorPortSpec for $name {
            type Shape = Shape1<U3>;
            const NAME: &'static str = $label;
            fn dimensions() -> Vec<DimensionDescriptor> {
                vec![DimensionDescriptor::Static(3)]
            }
            fn dtype() -> DtypeConstraint {
                DtypeConstraint::Exact(TensorDtype::F32)
            }
        }
    };
}

port!(RawPort, "raw");
port!(FeaturePort, "features");
port!(ScorePort, "scores");

type Raw = TensorBundleSpec<(RawPort,)>;
type Features = TensorBundleSpec<(FeaturePort,)>;
type Scores = TensorBundleSpec<(ScorePort,)>;

struct Featurize;
impl TensorContract for Featurize {
    type Input = Raw;
    type Output = Features;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(0x10);
    const VERSION: u32 = 1;
}

struct Score;
impl TensorContract for Score {
    type Input = Features;
    type Output = Scores;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(0x20);
    const VERSION: u32 = 1;
}

struct FeatureNode;
impl Animal for FeatureNode {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Featurize>;
}
impl OperationNode<Featurize> for FeatureNode {}

struct ScoreNode;
impl Animal for ScoreNode {
    type Id = Id<U1>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Score>;
}
impl OperationNode<Score> for ScoreNode {}

type Graph = List<(
    Unary<U0, FeatureNode, TypedEdges<list![Edge<U1, Score>]>, Featurize>,
    List<(Unary<U1, ScoreNode, Empty, Score>, Empty)>,
)>;

type HeterogeneousForward = ForwardOnly<(), Featurize, Score>;
#[derive(Default)]
struct EvaluationState;
type CheckpointStage = Step<Noop<NeutralSunState<EvaluationState>>>;
type EvaluationStage = Step<Noop<NeutralSunState<EvaluationState>>>;
type CheckpointThenEvaluate = CheckpointEvaluate<CheckpointStage, EvaluationStage, EvaluationState>;
type CheckpointEvaluateSun = <Graph as BlackHole>::Sun<CheckpointThenEvaluate>;

struct CheckpointEvaluateAnimal;
impl Animal for CheckpointEvaluateAnimal {
    type Id = Id<U2>;
    type Generation = U0;
    type State = NeutralSunState<EvaluationState>;
    type Seed = ();
    type Flow = CheckpointEvaluateSun;
}

fn assert_program<P: black_hole_flux::SunProgram, G: CompileSun<P>>() {
    let _: std::marker::PhantomData<<G as BlackHole>::Sun<P>> = std::marker::PhantomData;
}

#[test]
fn heterogeneous_forward_graph_compiles_without_two_sided_state() {
    assert_program::<HeterogeneousForward, Graph>();
    let state = ForwardSunState::<()>::default();
    assert!(state.runtime.inputs.is_empty());
    assert!(state.runtime.next_inputs.is_empty());
    assert!(state.runtime.outputs.is_empty());
}

#[test]
fn checkpoint_evaluate_program_uses_neutral_state() {
    assert_program::<CheckpointThenEvaluate, Graph>();
    let state = NeutralSunState::<EvaluationState>::default();
    assert!(!state.appearance().finalized);
    let _: std::marker::PhantomData<CheckpointEvaluateAnimal> = std::marker::PhantomData;
}
