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
pub use black_hole_type;
pub use black_hole_void;
pub use mass_client::MassClient;
pub use prompt_ops::{InferPromptOps, SeqPromptOps, TokenOps};
#[cfg(feature = "test")]
pub use test_utils::{
    make_client_endpoint, DeterministicFakeContract, DeterministicFakeOperation, NoCertVerifier,
    QuadraticContract, QuadraticOperation, RunningTestMassServer, RunningTestVoidServer,
    TensorSlicerContract, TensorSlicerOperation, TestMassServer, TestVoidServer,
};
pub use tokenizer::{Tokenizer, TokenizerBuilder};
pub use void_client::{PreparedTransfer, VoidClient};

// Convenience re-exports — flux modules and core sun types
pub use black_hole_flux::nodes::{atom, cell, fusion, warp};
pub use black_hole_flux::ForwardOnlyWithPolicy;
pub use black_hole_flux::{compile, forward, programs, topology};
pub use black_hole_flux::{
    ops, AtomError, BackwardOps, BackwardTypedEdges, Boundary, BoundaryInit, BoundaryInner,
    BoundaryMicrostep, BoundaryState, CellInit, CheckpointOps, DefaultConfig, ErrorFeedbackPolicy,
    ForwardOperationCell, ForwardOperationPrimordium, FuseOps, Fusion, FusionSeed, FusionState,
    InitBoundaryRecvId, MassOps, ModelConfig, NoErrorFeedback, NoModelBoundary, NoOscillation,
    OperationAtom, OperationCell, OperationNode, OperationPrimordium, OptimizeOps,
    OscillationSchedule, PerturbOps, QuzoFusion, QuzoFusionWithModelConfig, QwenAdapterOps, Ray,
    ResetOps, StepOps, VoidInferOps, VoidOps, Warp,
};
pub use black_hole_flux::configure_checkpointing;
pub use black_hole_flux::{
    BackwardOperationCell, BackwardOperationPrimordium, PipelineBackward, PipelineBackwardState,
    PipelineStepResult,
};

// Convenience re-exports — spec types
pub use black_hole_type::{
    ArtifactDelivery, ArtifactRef, BackwardCapability, ContractDescriptor, ContractHash,
    ContractId, ContractSide, DarkToken, DimensionDescriptor, DtypeConstraint, DurabilityPolicy,
    Emission, EmissionId, EncodingId, InferenceInput, InferenceOutput, InferenceOutputId,
    InferenceRequest, LayoutConstraint, LogitEntry, MassErrorFeedbackConfig, MassErrorFeedbackMode,
    MassIn, MassModelCapacity, MassModelConfig, MassModelParams, MassOut, MassPerturbationMode,
    ObjectId, ObjectRef, OperationArtifactRef, OperationCapabilities, OperationCapability,
    OperationConfig, OperationalControl, Potentiation, SequenceOutput, StreamRef,
    StreamingChunkOrder, StreamingFinalization, TensorDtype, TensorEnvelope, TensorPortDescriptor,
    TransferAbort, TransferBegin, TransferChunk, TransferHash, TransferManifest, TransferRecord,
    TransferRef, TransferStreamFrame, TransferTicket, Transmission, TRANSFER_PROTOCOL_VERSION,
};

// Convenience re-exports — typed operation contracts and tensor codec
pub use black_hole_spec::{
    backward_operation_capability, decode_input, decode_input_gradient, decode_output,
    decode_output_gradient, decode_qwen_output, descriptor_hash, encode_input,
    encode_input_gradient, encode_output, encode_output_gradient, encode_qwen_request,
    operation_capability, tensor_stream_header, validate_artifact, BackwardContract, CodecError,
    DecodedTensorBundle, PortList, QwenDarkInference, QwenInferenceMetadata, RawTensor,
    SingleTensorSpec, StreamingTensorOp, TensorBundleSpec, TensorContract, TensorPortSpec,
    TensorSpec, ValidatedArtifact,
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
