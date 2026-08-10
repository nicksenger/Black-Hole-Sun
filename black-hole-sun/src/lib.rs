//! Re-exports for black-hole workspace crates.
//!
//! Use this crate as the single dependency point for black-hole-probe.

pub use black_hole_quark;
pub use black_hole_spec;
pub use black_hole_void;

// Convenience re-exports — flux modules and core sun types
pub use black_hole_flux::{atom, cell, ops, sun, AtomError, Fusion, FusionSeed, FusionState, Progenitor};

// Convenience re-exports — spec types
pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, ObjectId, QuarkIn, QuarkOut, SequenceOutput, Transmission,
};

// Convenience re-exports — void types
pub use black_hole_void::{
    init_tracing, object_store, persist, ServerBuilder as VoidServerBuilder,
};

// Convenience re-exports — quark types
pub use black_hole_quark::ServerBuilder as QuarkServerBuilder;
