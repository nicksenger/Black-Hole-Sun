//! Topology compilation — turning a type-level graph into a program driver.
//!
//! [`BlackHole`] / [`CompileSun`] perform the recursive fold over a topology
//! descriptor; [`SunProgram`] selects the state, node seeds, and executable
//! driver for the compiled result. Deployment actions live in
//! [`action`](self::action).

pub mod action;
pub mod effect;

use black_hole_spec::{QwenDarkInference, TensorContract};
use black_hole_type::ObjectId;
use jungle_sdk::prelude::*;
use typenum::Unsigned;
use typosaurus::collections::list::{Empty, List};
use typosaurus::traits::semigroup::Mappend;
use uuid::Uuid;

use crate::programs::two_sided_zo::DeploymentProgram;
use crate::topology::{Binary, NodeIdsFromList, OperationNode, SunTopologyState, Unary, Warp};
use action::GenUuid;

/// Generate a program-selected unary seed, then spawn and register its animal.
#[derive(Flow)]
pub struct UnarySunStepWithProgram<
    Program: SunProgram,
    P: Unsigned,
    AnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed> + OperationNode<Op>,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<GenUuid<Program>>,
    Step<action::SpawnUnary<P, AnimalT, E, Program, Op>>,
);

pub type UnarySunStepWithState<P, AnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    UnarySunStepWithProgram<DeploymentProgram<S, ACCUM_STEPS>, P, AnimalT, E, Op>;

pub type UnarySunStep<
    P,
    AnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = UnarySunStepWithState<P, AnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// Generate a two-port seed, then spawn and register one binary animal.
#[derive(Flow)]
pub struct BinarySunStepWithProgram<
    Program: SunProgram,
    P1: Unsigned,
    P2: Unsigned,
    AnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed> + OperationNode<Op>,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<action::GenFusionSeed<Program>>,
    Step<action::SpawnBinary<P1, P2, AnimalT, E, Program, Op>>,
);

pub type BinarySunStepWithState<P1, P2, AnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    BinarySunStepWithProgram<DeploymentProgram<S, ACCUM_STEPS>, P1, P2, AnimalT, E, Op>;

pub type BinarySunStep<
    P1,
    P2,
    AnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = BinarySunStepWithState<P1, P2, AnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// Generate boundary mailboxes, spawn the nested warp animal, then spawn and
/// register the boundary animal in the parent topology.
#[derive(Flow)]
pub struct WarpSunStepWithProgram<
    Program: SunProgram,
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed> + OperationNode<Op>,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    Op: TensorContract,
>(
    Step<GenUuid<Program>>,
    Step<action::SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>>,
    Step<action::SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, Program, Op>>,
);

pub type WarpSunStepWithState<P, WarpAnimalT, BoundaryAnimalT, E, Op, S, const ACCUM_STEPS: usize> =
    WarpSunStepWithProgram<
        DeploymentProgram<S, ACCUM_STEPS>,
        P,
        WarpAnimalT,
        BoundaryAnimalT,
        E,
        Op,
    >;

pub type WarpSunStep<
    P,
    WarpAnimalT,
    BoundaryAnimalT,
    E,
    S = (),
    const GRADIENT_ACCUMULATION_STEPS: usize = 1,
    Op = QwenDarkInference,
> = WarpSunStepWithState<P, WarpAnimalT, BoundaryAnimalT, E, Op, S, GRADIENT_ACCUMULATION_STEPS>;

/// One descriptor-specific spawn flow followed by the remaining descriptors.
#[derive(Flow)]
pub struct SunNode<S, U>(S, U);

/// Compiles a type-level topology into the executable driver selected by `P`.
///
/// `<Topology as BlackHole>::Sun<Program>` is the canonical application
/// point. The recursive fold still emits the topology-specific deployment
/// steps; the terminal case now attaches `Program::Driver` instead of a fixed
/// QuZO step.
pub trait BlackHole {
    type Sun<P: SunProgram>
    where
        Self: CompileSun<P>;
}

impl<T> BlackHole for T {
    type Sun<P: SunProgram>
        = <T as CompileSun<P>>::Flow
    where
        T: CompileSun<P>;
}

/// Program-specific compilation proof for a topology. This is where node
/// seed requirements are selected; the recursive [`BlackHole`] facade no
/// longer fixes them globally.
pub trait CompileSun<Program: SunProgram> {
    type Flow;
}

impl<Program: SunProgram, U> CompileSun<Program> for List<(Empty, U)>
where
    U: CompileSun<Program>,
{
    type Flow = U::Flow;
}

impl<Program: SunProgram, T1, T2, U> CompileSun<Program> for List<(List<(T1, T2)>, U)>
where
    (List<(T1, T2)>, U): Mappend,
    <(List<(T1, T2)>, U) as Mappend>::Out: CompileSun<Program>,
{
    type Flow = <<(List<(T1, T2)>, U) as Mappend>::Out as CompileSun<Program>>::Flow;
}

impl<Program, P, A, E, Op, U> CompileSun<Program> for List<(Unary<P, A, E, Op>, U)>
where
    Program: SunProgram,
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow = SunNode<UnarySunStepWithProgram<Program, P, A, E, Op>, U::Flow>;
}

impl<Program, P1, P2, A, E, Op, U> CompileSun<Program> for List<(Binary<P1, P2, A, E, Op>, U)>
where
    Program: SunProgram,
    P1: Unsigned,
    P2: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow = SunNode<BinarySunStepWithProgram<Program, P1, P2, A, E, Op>, U::Flow>;
}

impl<Program, P, WarpAnimalT, BoundaryAnimalT, E, Op, U> CompileSun<Program>
    for List<(Warp<P, WarpAnimalT, BoundaryAnimalT, E, Op>, U)>
where
    Program: SunProgram,
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::WarpSeed>
        + OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + crate::topology::DeclaredEdges<Op>,
    U: CompileSun<Program>,
{
    type Flow =
        SunNode<WarpSunStepWithProgram<Program, P, WarpAnimalT, BoundaryAnimalT, E, Op>, U::Flow>;
}

impl<Program: SunProgram> CompileSun<Program> for Empty {
    type Flow = Program::Driver;
}

/// Selects the state, deployment settings, and executable driver for a Sun.
pub trait SunProgram {
    type State: SunTopologyState;
    type Driver;
    type UnarySeed: Clone + Send + Sync + 'static;
    type BinarySeed: Clone + Send + Sync + 'static;
    type WarpSeed: Clone + Send + Sync + 'static;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed;
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId;
    fn binary_seed(inboxes: [ObjectId; 2]) -> Self::BinarySeed;
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2];
    fn warp_seed(inbox: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed;
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]);
}
