//! Shared types for the black-hole workspace.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque identifier for objects stored in void.
pub type ObjectId = Uuid;

// ---------------------------------------------------------------------------
// QuZO wire protocol (black-hole-quark <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the quark QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuzoIn {
    /// Perturb model weights in the positive direction.
    PerturbUp { seed: u64 },
    /// Run inference on the input object stored in void.
    /// Returns QuzoOut::Inferred(output_id).
    Infer { input_id: ObjectId },
    /// Perturb model weights in the negative direction.
    PerturbDown,
    /// Apply the QuZO optimization update with both loss values.
    Optimize { loss_up: f32, loss_down: f32 },
}

/// Response sent by the quark server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuzoOut {
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

/// A single logit entry (token ID + log probability) for soft prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitEntry {
    pub token_id: u32,
    pub log_prob: f32,
}

/// A soft token position for dark-knowledge transfer between model forward passes.
/// Carries the predicted (committed) token ID and a top-K distribution from a teacher model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftToken {
    /// The predicted (committed) token ID for this position.
    pub predicted: u32,
    /// Top-K logit entries representing the teacher model's distribution at this position.
    pub dark_knowledge: Vec<LogitEntry>,
}

/// Serializable inference input, mirroring paramecia-engine's ModelInput.
/// Stored inside void objects and converted to ModelInput by the quark service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuzoInferInput {
    /// Text context (tokenized by the model host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
    /// Soft prompt: a sequence of soft tokens carrying predicted token IDs and
    /// dark-knowledge distributions.
    Soft(Vec<SoftToken>),
}

/// Serializable list of inference inputs for a single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuzoInferRequest {
    pub inputs: Vec<QuzoInferInput>,
}
