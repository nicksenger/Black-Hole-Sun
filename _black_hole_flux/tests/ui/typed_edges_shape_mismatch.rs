use black_hole_contract::{glowstick::Shape1, TensorBundleSpec, TensorContract, TensorPortSpec};
use black_hole_flux::{DeclaredEdges, Edge, TypedEdges};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use typenum::{U1, U3, U4};
use typosaurus::list;

macro_rules! port {
    ($name:ident, $dim:ty, $size:literal) => {
        struct $name;
        impl TensorPortSpec for $name {
            type Shape = Shape1<$dim>;
            const NAME: &'static str = "value";
            fn dimensions() -> Vec<DimensionDescriptor> {
                vec![DimensionDescriptor::Static($size)]
            }
            fn dtype() -> DtypeConstraint {
                DtypeConstraint::Exact(TensorDtype::F32)
            }
        }
    };
}
port!(Three, U3, 3);
port!(Four, U4, 4);
struct Source;
impl TensorContract for Source {
    type Input = TensorBundleSpec<(Three,)>;
    type Output = TensorBundleSpec<(Three,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(1);
    const VERSION: u32 = 1;
}
struct Destination;
impl TensorContract for Destination {
    type Input = TensorBundleSpec<(Four,)>;
    type Output = TensorBundleSpec<(Four,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(2);
    const VERSION: u32 = 1;
}
fn assert_edges<S: TensorContract, E: DeclaredEdges<S>>() {}
fn main() {
    assert_edges::<Source, TypedEdges<list![Edge<U1, Destination>]>>();
}
