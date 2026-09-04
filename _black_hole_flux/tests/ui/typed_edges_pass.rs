use black_hole_spec::{
    glowstick::{Dyn, Shape1},
    TensorBundleSpec, TensorContract, TensorPortSpec,
};
use black_hole_flux::{DeclaredEdges, Edge, TypedEdges};
use black_hole_type::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use typenum::{U1, U3};
use typosaurus::list;

struct StaticPort;
impl TensorPortSpec for StaticPort {
    type Shape = Shape1<U3>;
    const NAME: &'static str = "static";
    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![DimensionDescriptor::Static(3)]
    }
    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

struct Batch;
struct SymbolicPort;
impl TensorPortSpec for SymbolicPort {
    type Shape = Shape1<Dyn<Batch>>;
    const NAME: &'static str = "symbolic";
    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![DimensionDescriptor::Symbolic("batch".into())]
    }
    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

type StaticBundle = TensorBundleSpec<(StaticPort,)>;
type SymbolicBundle = TensorBundleSpec<(SymbolicPort,)>;

struct StaticSource;
impl TensorContract for StaticSource {
    type Input = StaticBundle;
    type Output = StaticBundle;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(1);
    const VERSION: u32 = 1;
}
struct StaticDestination;
impl TensorContract for StaticDestination {
    type Input = StaticBundle;
    type Output = StaticBundle;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(2);
    const VERSION: u32 = 1;
}
struct SymbolicSource;
impl TensorContract for SymbolicSource {
    type Input = SymbolicBundle;
    type Output = SymbolicBundle;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(3);
    const VERSION: u32 = 1;
}
struct SymbolicDestination;
impl TensorContract for SymbolicDestination {
    type Input = SymbolicBundle;
    type Output = SymbolicBundle;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(4);
    const VERSION: u32 = 1;
}

fn assert_edges<S: TensorContract, E: DeclaredEdges<S>>() {}

fn main() {
    assert_edges::<StaticSource, TypedEdges<list![Edge<U1, StaticDestination>]>>();
    assert_edges::<SymbolicSource, TypedEdges<list![Edge<U1, SymbolicDestination>]>>();
}
