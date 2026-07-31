//! Shared types for the black-hole workspace.

use serde::{Deserialize, Serialize};

/// A UUID as a hyphenated string (e.g. "550e8400-e29b-41d4-a716-446655440000").
pub type ObjectId = String;

// ---------------------------------------------------------------------------
// QuZO wire protocol (black-hole-quark <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the quark QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuzoIn {
    /// Perturb model weights in the positive direction.
    PerturbUp,
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

/// Serializable inference input, mirroring paramecia-engine's ModelInput.
/// Stored inside void objects and converted to ModelInput by the quark service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuzoInferInput {
    /// Text context (tokenized by the model host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
}

/// Serializable list of inference inputs for a single forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuzoInferRequest {
    pub inputs: Vec<QuzoInferInput>,
}
