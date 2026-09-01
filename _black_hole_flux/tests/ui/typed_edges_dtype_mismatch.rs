use black_hole_contract::{glowstick::Shape1, TensorBundleSpec, TensorContract, TensorPortSpec};
use black_hole_flux::{DeclaredEdges, Edge, TypedEdges};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use typenum::{U1, U3};
use typosaurus::list;

macro_rules! port {
    ($name:ident, $dtype:expr) => {
        struct $name;
        impl TensorPortSpec for $name {
            type Shape = Shape1<U3>;
            const NAME: &'static str = "value";
            fn dimensions() -> Vec<DimensionDescriptor> {
                vec![DimensionDescriptor::Static(3)]
            }
            fn dtype() -> DtypeConstraint {
                DtypeConstraint::Exact($dtype)
            }
        }
    };
}
port!(FloatValue, TensorDtype::F32);
port!(IntegerValue, TensorDtype::U32);
struct Source;
impl TensorContract for Source {
    type Input = TensorBundleSpec<(FloatValue,)>;
    type Output = TensorBundleSpec<(FloatValue,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(1);
    const VERSION: u32 = 1;
}
struct Destination;
impl TensorContract for Destination {
    type Input = TensorBundleSpec<(IntegerValue,)>;
    type Output = TensorBundleSpec<(IntegerValue,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(2);
    const VERSION: u32 = 1;
}
fn assert_edges<S: TensorContract, E: DeclaredEdges<S>>() {}
fn main() {
    assert_edges::<Source, TypedEdges<list![Edge<U1, Destination>]>>();
}
