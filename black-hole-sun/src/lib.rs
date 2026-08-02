//! Re-exports for black-hole workspace crates.
//!
//! Use this crate as the single dependency point for black-hole-probe.

pub use black_hole_spec;
pub use black_hole_void;
pub use black_hole_quark;

// Convenience re-exports — spec types
pub use black_hole_spec::{
    ObjectId, QuarkIn, QuarkInferenceInput, QuarkInferenceOutput, QuarkInferenceRequest, QuarkOut, SequenceOutput, PredictedToken,
};

// Convenience re-exports — void types
pub use black_hole_void::{
    init_tracing, object_store, persist, ServerBuilder as VoidServerBuilder,
};

// Convenience re-exports — quark types
pub use black_hole_quark::ServerBuilder as QuarkServerBuilder;
