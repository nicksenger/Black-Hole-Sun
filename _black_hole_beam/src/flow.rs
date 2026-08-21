//! Marker traits describing which Jungle animals and structural flows can be
//! rendered as Black Hole Sun views.

use black_hole_flux::sun::{
    BinarySunStep, NodeIdsFromList, Sun, SunAppearance, SunNode, SunState, UnarySunStep,
};
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
/// `<Graph as BlackHole>::Sun<M, N>`, where `M` is a
/// [`Manifest`](black_hole_flux::sun::Manifest) bundling generator, policy,
/// and state.
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

impl<Port, A, Edges, Tail, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for SunNode<UnarySunStep<Port, A, Edges, S, GRADIENT_ACCUMULATION_STEPS>, Tail>
where
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = black_hole_flux::CellInit> + 'static,
    Edges: NodeIdsFromList,
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

impl<PortA, PortB, A, Edges, Tail, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for SunNode<BinarySunStep<PortA, PortB, A, Edges, S, GRADIENT_ACCUMULATION_STEPS>, Tail>
where
    PortA: Unsigned,
    PortB: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>
        + 'static,
    A::Flow: FusionFlow,
    Edges: NodeIdsFromList,
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
