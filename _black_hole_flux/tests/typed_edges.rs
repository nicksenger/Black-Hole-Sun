#[test]
fn operation_typed_edges_are_checked_at_compile_time() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/typed_edges_pass.rs");
    cases.compile_fail("tests/ui/typed_edges_shape_mismatch.rs");
    cases.compile_fail("tests/ui/typed_edges_dtype_mismatch.rs");
    cases.compile_fail("tests/ui/typed_edges_contract_mismatch.rs");
}
