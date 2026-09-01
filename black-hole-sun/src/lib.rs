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
pub use black_hole_contract;
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
    atom, cell, fusion, ops, sun, warp, AtomError, Boundary, BoundaryInit, BoundaryInner,
    BoundaryMicrostep, BoundaryState, CellInit, CheckpointOps, DefaultConfig, ErrorFeedbackPolicy,
    FuseOps, Fusion, FusionSeed, FusionState, InitBoundaryRecvId, MassOps, ModelConfig,
    NoErrorFeedback, NoModelBoundary, NoOscillation, OptimizeOps, OscillationSchedule, PerturbOps,
    QuzoFusion, QuzoFusionWithModelConfig, QwenAdapterOps, Ray, ResetOps, VoidInferOps, VoidOps,
    Warp,
};

// Convenience re-exports — spec types
pub use black_hole_spec::{
    ArtifactRef, ContractDescriptor, ContractHash, ContractId, ContractSide, DarkToken,
    DimensionDescriptor, DtypeConstraint, Emission, EmissionId, EncodingId, InferenceInput,
    InferenceOutput, InferenceOutputId, InferenceRequest, LayoutConstraint, LogitEntry,
    MassErrorFeedbackConfig, MassErrorFeedbackMode, MassIn, MassModelCapacity, MassModelConfig,
    MassModelParams, MassOut, MassPerturbationMode, ObjectId, ObjectRef, OperationArtifactRef,
    OperationCapability, Potentiation, SequenceOutput, TensorDtype, TensorEnvelope,
    TensorPortDescriptor, Transmission,
};

// Convenience re-exports — typed operation contracts and tensor codec
pub use black_hole_contract::{
    decode_input, decode_output, descriptor_hash, encode_input, encode_output,
    operation_capability, validate_artifact, CodecError, DecodedTensorBundle, PortList,
    QwenDarkInference, RawTensor, SingleTensorSpec, TensorBundleSpec, TensorContract,
    TensorPortSpec, TensorSpec, ValidatedArtifact,
};

// Convenience re-exports — void types
pub use black_hole_void::{
    init_tracing, object_store, persist, ServerBuilder as VoidServerBuilder, VoidIn, VoidOut,
};

// Convenience re-exports — mass types
pub use black_hole_mass::{OperationImplementation, ServerBuilder as MassServerBuilder};

// The beam crate; its `piano` feature (score text format, piano event
// types) is enabled by this crate's `piano` feature.
pub use black_hole_beam;
