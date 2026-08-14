//! Re-exports for black-hole workspace crates.
//!
//! Use this crate as the single dependency point for black-hole-probe.

mod accumulated_transmissions;
mod quark_client;
#[cfg(feature = "test")]
mod test_utils;
mod tokenizer;
mod void_client;

pub use accumulated_transmissions::{AccumulatedTransmissions, Monoid};
pub use black_hole_quark;
pub use black_hole_spec;
pub use black_hole_void;
pub use quark_client::QuarkClient;
#[cfg(feature = "test")]
pub use test_utils::{
    make_client_endpoint, NoCertVerifier, RunningTestQuarkServer, RunningTestVoidServer,
    TestQuarkServer, TestVoidServer,
};
pub use tokenizer::{Tokenizer, TokenizerBuilder};
pub use void_client::VoidClient;

// Convenience re-exports — flux modules and core sun types
pub use black_hole_flux::{
    atom, cell, fusion, ops, sun, twin, AtomError, CellInit, DefaultConfig, ErrorFeedbackPolicy,
    Fusion, FusionSeed, FusionState, LeftStack, ModelConfig, NoErrorFeedback, NoOscillation,
    OscillationSchedule, Progenitor, QuzoFusion, QuzoFusionWithModelConfig, RandStack, Ray,
    RightStack, Twin,
};

// Convenience re-exports — spec types
pub use black_hole_spec::{
    DarkToken, Emission, EmissionId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LogitEntry, ObjectId, QuarkErrorFeedbackConfig, QuarkErrorFeedbackMode,
    QuarkIn, QuarkModelCapacity, QuarkModelConfig, QuarkModelParams, QuarkOut, SequenceOutput,
    Transmission,
};

// Convenience re-exports — void types
pub use black_hole_void::{
    init_tracing, object_store, persist, ServerBuilder as VoidServerBuilder, VoidIn, VoidOut,
};

// Convenience re-exports — quark types
pub use black_hole_quark::ServerBuilder as QuarkServerBuilder;
