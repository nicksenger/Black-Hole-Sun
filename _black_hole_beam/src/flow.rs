//! Marker traits describing which Jungle animals and structural flows can be
//! rendered as Black Hole Sun views.

use black_hole_flux::compile::{
    BinarySunStepWithProgram, SunNode, SunProgram, UnarySunStepWithProgram,
};
use black_hole_flux::forward::ServeFlow;
use black_hole_flux::programs::two_sided_zo::Sun;
use black_hole_flux::topology::{NodeIdsFromList, OperationNode, SunAppearance, SunStateView};
use black_hole_flux::{DeclaredEdges, TensorContract};
use jungle_sdk::{Animal, AnimalIdValue, Observe};
use typenum::Unsigned;

use crate::model::CellDefinition;

/// Marker for Sun animals whose runtime state is `SunState<S>`.
pub trait BlackHoleSunAnimal: Animal + Observe<Appearance = SunAppearance> {}

impl<A> BlackHoleSunAnimal for A
where
    A: Animal + Observe<Appearance = SunAppearance>,
    A::State: SunStateView,
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
/// emitted by [`BlackHole`](black_hole_flux::compile::BlackHole).
#[allow(private_bounds)]
pub trait BlackHoleSunFlow: private::DescribeSun {}

impl<T> BlackHoleSunFlow for T where T: private::DescribeSun {}

impl<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for Sun<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>
{
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Source, Input: Send + 'static, Output: Send + 'static, S> private::DescribeSun
    for ServeFlow<Source, Input, Output, S>
{
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Program, Port, A, Edges, Tail, Op> private::DescribeSun
    for SunNode<UnarySunStepWithProgram<Program, Port, A, Edges, Op>, Tail>
where
    Program: SunProgram,
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::UnarySeed> + 'static,
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
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = Program::BinarySeed> + 'static,
    A: OperationNode<Op>,
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
