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
    /// Perturb model weights in the positive direction.
    PerturbUp { seed: u64 },
    /// Run inference on the input object stored in void.
    /// Returns QuarkOut::Inferred(output_id).
    Infer { input_id: ObjectId },
    /// Perturb model weights in the negative direction.
    PerturbDown,
    /// Apply the QuZO optimization update with both loss values.
    Optimize { loss_up: f32, loss_down: f32 },
}

/// Response sent by the quark server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuarkOut {
    /// Acknowledges a perturb or optimize step.
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
pub enum QuarkInferenceInput {
    /// Text context (tokenized by the model host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
    /// Darkness prompt: a sequence of dark tokens carrying predicted token IDs and
    /// dark-knowledge distributions.
    Darkness(Vec<DarkToken>),
}

/// Serializable batch of inference sequences for a single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarkInferenceRequest {
    /// Each element is one sequence (a list of inputs concatenated in order).
    pub sequences: Vec<Vec<QuarkInferenceInput>>,
    pub limit: u32,
}

// ---------------------------------------------------------------------------
// Inference output format (stored in void objects)
// ---------------------------------------------------------------------------

/// A single predicted token with its top-K distribution from a model forward pass.
/// Stored as a dark token so downstream models can use it for dark-knowledge transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedToken {
    /// The predicted (committed) token ID.
    pub token_id: u32,
    /// Decoded text for this token, if available.
    pub text: Option<String>,
    /// Top-K logit entries representing the model's distribution at this position.
    pub top_k: Vec<LogitEntry>,
}

/// Predictions for a single sequence within a batched inference result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOutput {
    pub predictions: Vec<PredictedToken>,
}

/// Serializable inference output stored in void objects.
/// Contains per-sequence results from a batched forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarkInferenceOutput {
    pub results: Vec<SequenceOutput>,
}
