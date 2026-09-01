use black_hole_contract::{glowstick::Shape1, TensorBundleSpec, TensorContract, TensorPortSpec};
use black_hole_flux::{DeclaredEdges, Edge, TypedEdges};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use typenum::{U1, U3};
use typosaurus::list;

// These ports deliberately have identical runtime shape/dtype descriptors,
// but distinct Rust identities represent different artifact semantics.
macro_rules! semantic_port {
    ($name:ident) => {
        struct $name;
        impl TensorPortSpec for $name {
            type Shape = Shape1<U3>;
            const NAME: &'static str = "embedding";
            fn dimensions() -> Vec<DimensionDescriptor> {
                vec![DimensionDescriptor::Static(3)]
            }
            fn dtype() -> DtypeConstraint {
                DtypeConstraint::Exact(TensorDtype::F32)
            }
        }
    };
}
semantic_port!(ImageEmbedding);
semantic_port!(TextEmbedding);
struct Source;
impl TensorContract for Source {
    type Input = TensorBundleSpec<(ImageEmbedding,)>;
    type Output = TensorBundleSpec<(ImageEmbedding,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(1);
    const VERSION: u32 = 1;
}
struct Destination;
impl TensorContract for Destination {
    type Input = TensorBundleSpec<(TextEmbedding,)>;
    type Output = TensorBundleSpec<(TextEmbedding,)>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(2);
    const VERSION: u32 = 1;
}
fn assert_edges<S: TensorContract, E: DeclaredEdges<S>>() {}
fn main() {
    assert_edges::<Source, TypedEdges<list![Edge<U1, Destination>]>>();
}
