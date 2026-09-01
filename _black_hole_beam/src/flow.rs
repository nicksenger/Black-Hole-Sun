//! Marker traits describing which Jungle animals and structural flows can be
//! rendered as Black Hole Sun views.

use black_hole_flux::sun::{
    BinarySunStepWithProgram, NodeIdsFromList, OperationNode, ServeFlow, Sun, SunAppearance,
    SunNode, SunProgram, SunState, UnarySunStepWithProgram,
};
use black_hole_flux::{DeclaredEdges, TensorContract};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState};
use jungle_sdk::{Animal, AnimalIdValue, Observe};
use typenum::Unsigned;

use crate::model::CellDefinition;

/// Marker for Sun animals whose runtime state is `SunState<S>`.
pub trait BlackHoleSunAnimal: Animal + Observe<Appearance = SunAppearance> {}

impl<A, S> BlackHoleSunAnimal for A where
    A: Animal<State = SunState<S>> + Observe<Appearance = SunAppearance>
{
}

pub(crate) mod private {
    pub(crate) trait DescribeSun {
        fn append_cells(cells: &mut Vec<crate::model::CellDefinition>);
    }
}

/// Marker for the structural flow produced by
/// `<Graph as BlackHole>::Sun<Program>`.
///
/// The trait is sealed and is only implemented for the `SunNode<…>` chain
/// emitted by [`BlackHole`](black_hole_flux::sun::BlackHole).
#[allow(private_bounds)]
pub trait BlackHoleSunFlow: private::DescribeSun {}

impl<T> BlackHoleSunFlow for T where T: private::DescribeSun {}

impl<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for Sun<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>
{
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Source, T: Send + 'static, S> private::DescribeSun for ServeFlow<Source, T, S> {
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Program, Port, A, Edges, Tail, Op> private::DescribeSun
    for SunNode<UnarySunStepWithProgram<Program, Port, A, Edges, Op>, Tail>
where
    Program: SunProgram,
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = black_hole_flux::CellInit> + 'static,
    A: OperationNode<Op>,
    Op: TensorContract,
    Edges: NodeIdsFromList + DeclaredEdges<Op>,
    Tail: private::DescribeSun,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            Port::U32,
            vec![Port::U32],
            Edges::node_ids(),
        ));
        Tail::append_cells(cells);
    }
}

impl<Program, PortA, PortB, A, Edges, Tail, Op> private::DescribeSun
    for SunNode<BinarySunStepWithProgram<Program, PortA, PortB, A, Edges, Op>, Tail>
where
    Program: SunProgram,
    PortA: Unsigned,
    PortB: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>
        + 'static,
    A: OperationNode<Op>,
    A::Flow: FusionFlow,
    Op: TensorContract,
    Edges: NodeIdsFromList + DeclaredEdges<Op>,
    Tail: private::DescribeSun,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            PortA::U32,
            vec![PortA::U32, PortB::U32],
            Edges::node_ids(),
        ));
        Tail::append_cells(cells);
    }
}
