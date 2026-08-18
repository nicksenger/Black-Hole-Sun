//! Re-exports for black-hole workspace crates.
//!
//! Use this crate as the single dependency point for black-hole-probe.

mod accumulated_transmissions;
mod mass_client;
mod prompt_ops;
#[cfg(feature = "test")]
mod test_utils;
mod tokenizer;
mod void_client;

pub use accumulated_transmissions::{AccumulatedTransmissions, Monoid};
pub use black_hole_mass;
pub use black_hole_spec;
pub use black_hole_void;
pub use mass_client::MassClient;
pub use prompt_ops::{InferPromptOps, SeqPromptOps, TokenOps};
#[cfg(feature = "test")]
pub use test_utils::{
    make_client_endpoint, NoCertVerifier, RunningTestMassServer, RunningTestVoidServer,
    TestMassServer, TestVoidServer,
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
    InferenceRequest, LogitEntry, ObjectId, Potentiation, MassErrorFeedbackConfig,
    MassErrorFeedbackMode, MassIn, MassModelCapacity, MassModelConfig, MassModelParams,
    MassOut, SequenceOutput, Transmission,
};

// Convenience re-exports — void types
pub use black_hole_void::{
    init_tracing, object_store, persist, ServerBuilder as VoidServerBuilder, VoidIn, VoidOut,
};

// Convenience re-exports — mass types
pub use black_hole_mass::ServerBuilder as MassServerBuilder;
