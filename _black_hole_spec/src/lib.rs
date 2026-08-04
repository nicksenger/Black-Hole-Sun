//! Shared types for the black-hole workspace.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque identifier for objects stored in void.
pub type ObjectId = Uuid;

// ---------------------------------------------------------------------------
// Quark wire protocol (black-hole-quark <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the quark QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuarkIn {
    /// Start a new model instance with the provided stable ID.
    Start { model_id: Uuid },
    /// Perturb model weights in the positive direction.
    PerturbUp { model_id: Uuid, seed: u64 },
    /// Run inference on the input object stored in void.
    /// Returns QuarkOut::Inferred(output_id).
    Infer { model_id: Uuid, input_id: ObjectId },
    /// Perturb model weights in the negative direction.
    PerturbDown { model_id: Uuid },
    /// Apply the QuZO optimization update with both loss values.
    Optimize {
        model_id: Uuid,
        loss_up: f32,
        loss_down: f32,
    },
    /// Shut down the model instance with the provided ID.
    Shutdown { model_id: Uuid },
}

/// Response sent by the quark server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuarkOut {
    /// Acknowledges a lifecycle, perturb, or optimize step.
    Ack,
    /// Inference complete; contains the void object ID of the output.
    Inferred { output_id: ObjectId },
    /// Error from any operation.
    Error { message: String },
}

// ---------------------------------------------------------------------------
// Inference input format (stored in void objects)
// ---------------------------------------------------------------------------

/// A single logit entry (token ID + log probability) for dark prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitEntry {
    pub token_id: u32,
    pub log_prob: f32,
}

/// A dark token position for dark-knowledge transfer between model forward passes.
/// Carries the predicted (committed) token ID and a top-K distribution from a teacher model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkToken {
    /// The predicted (committed) token ID for this position.
    pub predicted: u32,
    /// Top-K logit entries representing the teacher model's distribution at this position.
    pub dark_knowledge: Vec<LogitEntry>,
}

/// Serializable inference input, mirroring paramecia-engine's ModelInput.
/// Stored inside void objects and converted to ModelInput by the quark service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInput {
    /// Text context (tokenized by the model host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
    /// Dark prompt: a sequence of dark tokens carrying predicted token IDs and
    /// dark-knowledge distributions.
    Dark(Vec<DarkToken>),
}

/// Serializable inference request stored in void objects.
/// Either contains inline sequences or points to an existing InferenceOutput
/// in void that should be converted to dark input for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceRequest {
    /// Inline sequences with explicit inputs.
    Sequences {
        /// Each element is one sequence (a list of inputs concatenated in order).
        sequences: Vec<Vec<InferenceInput>>,
        limit: u32,
    },
    /// Reference to an existing InferenceOutput in void.
    /// Quark downloads it, converts the results to dark input, and proceeds.
    VoidId {
        /// Void object ID of the InferenceOutput to use as input.
        id: InferenceOutputId,
        limit: u32,
    },
}

// ---------------------------------------------------------------------------
// Inference output format (stored in void objects)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOutput(pub Vec<DarkToken>);

/// Serializable inference output stored in void objects.
/// Contains per-sequence results from a batched forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub results: Vec<SequenceOutput>,
}

/// Void ID for an InferenceOutput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutputId(pub ObjectId);

/// Input / Output from a Atom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Emission<M> {
    pub metadata: M,
    pub output_id: InferenceOutputId,
}

/// Void ID for an Emission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionId(pub ObjectId);

/// Input / Output from a Cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transmission {
    Propagation {
        emission_id: EmissionId,
        recv: ObjectId,
        send: ObjectId,
    },
    Potentiation {
        loss_up: f32,
        loss_down: f32,
        recv: ObjectId,
    },
}
