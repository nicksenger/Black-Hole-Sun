//! Shared types for the black-hole workspace.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const IM_START: u32 = 248045;
pub const IM_END: u32 = 248046;
pub const PAD: u32 = 248044;
pub const THINK_OPEN: u32 = 248068;
pub const THINK_CLOSE: u32 = 248069;

/// Opaque identifier for objects stored in void.
pub type ObjectId = Uuid;

// ---------------------------------------------------------------------------
// Mass wire protocol (black-hole-mass <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the mass QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum MassIn {
    /// Start a new model instance with the provided stable ID.
    ///
    /// When `model_config` is `None`, mass uses server defaults. When present,
    /// the provided values override defaults for this specific model instance.
    Start {
        model_id: Uuid,
        model_config: Option<MassModelConfig>,
    },
    /// Perturb model weights in the positive direction.
    PerturbUp { model_id: Uuid, seed: u64 },
    /// Run inference on the input object stored in void.
    /// Returns MassOut::Inferred(output_id).
    Infer { model_id: Uuid, input_id: ObjectId },
    /// Reset model runtime state (for example KV cache) for the instance.
    Reset { model_id: Uuid },
    /// Perturb model weights in the negative direction.
    PerturbDown { model_id: Uuid },
    /// Upload current model weights to void and return their object ID.
    Checkpoint { model_id: Uuid },
    /// Apply the QuZO optimization update with both loss values.
    Optimize {
        model_id: Uuid,
        loss_up: f32,
        loss_down: f32,
    },
    /// Shut down the model instance with the provided ID.
    Shutdown { model_id: Uuid },
    /// Query the current runtime parameters for a model instance.
    QueryModelParams { model_id: Uuid },
    /// Query recursive model instance capacity for this mass subtree.
    QueryModelCapacity,
    /// Register a one-hop tunnel worker with a root mass.
    RegisterTunnel {
        /// Stable worker identity used by parent masss to match reconnects.
        worker_id: Uuid,
        /// Optional total model capacity advertised by this worker subtree (defaults to 1).
        max_instances: Option<usize>,
    },
    /// Update the advertised tunnel capacity for an already-registered worker token.
    UpdateTunnelCapacity {
        /// Root/parent-issued token for the registered worker.
        token: Uuid,
        /// Optional total model capacity for this worker subtree (defaults to 1).
        max_instances: Option<usize>,
    },
    /// Forward a model operation through a registered tunnel worker.
    TunnelForward {
        /// Root-issued token proving this request was authorized for the worker.
        token: Uuid,
        /// Forwarded model operation.
        request: TunnelRequest,
    },
}

/// Response sent by the mass server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum MassOut {
    /// Acknowledges a lifecycle, perturb, or optimize step.
    Ack,
    /// Inference complete; contains the void object ID of the output.
    Inferred { output_id: ObjectId },
    /// Checkpoint upload complete; contains the void object ID of model weights.
    Checkpointed { checkpoint_id: ObjectId },
    /// Runtime model parameters for a running instance.
    ModelParams { params: MassModelParams },
    /// Recursive model instance capacity for this mass subtree.
    ModelCapacity { capacity: MassModelCapacity },
    /// Tunnel worker registration complete; contains root-issued auth token.
    TunnelRegistered { token: Uuid },
    /// Error from any operation.
    Error { message: String },
}

/// Runtime model parameters resolved for a running mass model instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassModelParams {
    pub inference_limit: u32,
    pub top_k: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub training_lr: f64,
    pub training_epsilon: f64,
    pub training_z_loss: f64,
    pub training_lb_loss: f64,
    pub training_clip_threshold: f64,
    pub training_error_feedback: MassErrorFeedbackConfig,
    pub is_frozen: bool,
    pub optimize_steps: u32,
    pub oscillation_period_steps: Option<u32>,
    pub oscillation_train_steps: Option<u32>,
    pub oscillation_phase_steps: Option<u32>,
    pub oscillation_warmup_steps: Option<u32>,
}

/// Recursive model-capacity snapshot for a mass server subtree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MassModelCapacity {
    /// Total model-instance capacity (local + descendants). None means unbounded.
    pub total: Option<usize>,
    /// Available model-instance capacity (total minus occupied). None means unbounded.
    pub available: Option<usize>,
    /// Occupied model-instance slots currently routed in this subtree.
    pub occupied: usize,
}

/// Error-feedback mode selector for QuZO optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassErrorFeedbackMode {
    Off,
    Persistent,
    Replay,
}

/// Per-model QuZO error-feedback configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MassErrorFeedbackConfig {
    Off,
    Persistent { decay: f64, gain: f64 },
    Replay { steps: u32, decay: f64, gain: f64 },
}

/// Forwardable model operation used for root->worker tunnel requests.
#[derive(Debug, Serialize, Deserialize)]
pub enum TunnelRequest {
    Start {
        model_id: Uuid,
        model_config: Option<MassModelConfig>,
    },
    PerturbUp {
        model_id: Uuid,
        seed: u64,
    },
    Infer {
        model_id: Uuid,
        input_id: ObjectId,
    },
    Reset {
        model_id: Uuid,
    },
    PerturbDown {
        model_id: Uuid,
    },
    Checkpoint {
        model_id: Uuid,
    },
    Optimize {
        model_id: Uuid,
        loss_up: f32,
        loss_down: f32,
    },
    Shutdown {
        model_id: Uuid,
    },
    QueryModelParams {
        model_id: Uuid,
    },
}

/// Per-model-instance mass configuration overrides.
///
/// Each field is optional; omitted values fall back to server defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MassModelConfig {
    pub top_k: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub inference_limit: Option<u32>,
    pub training_lr: Option<f64>,
    pub training_epsilon: Option<f64>,
    pub training_z_loss: Option<f64>,
    pub training_lb_loss: Option<f64>,
    pub training_clip_threshold: Option<f64>,
    pub training_error_feedback: Option<MassErrorFeedbackConfig>,
    pub frozen: Option<bool>,
    /// Optional optimize-step period for train/freeze oscillation scheduling.
    ///
    /// When set (with `oscillation_train_steps`), mass applies a deterministic
    /// train window each cycle after warmup instead of flipping prior state.
    pub oscillation_period_steps: Option<u32>,
    /// Optional count of trainable optimize steps in each oscillation cycle.
    ///
    /// Must be less than or equal to `oscillation_period_steps`.
    pub oscillation_train_steps: Option<u32>,
    /// Optional per-instance phase shift (in optimize steps), modulo period.
    pub oscillation_phase_steps: Option<u32>,
    /// Optional number of optimize steps to wait before schedule activation.
    ///
    /// Ignored when `oscillation_period_steps` is `None`.
    pub oscillation_warmup_steps: Option<u32>,
    /// Optional checkpoint object to load model weights from for this instance.
    ///
    /// When `None`, mass loads weights from its configured server model path.
    pub checkpoint_id: Option<ObjectId>,
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
/// Stored inside void objects and converted to ModelInput by the mass service.
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
        /// Optional generation cap. If `None`, mass applies its server default.
        limit: Option<u32>,
    },
    /// Reference to an existing InferenceOutput in void.
    /// Mass downloads it, converts the results to dark input, and proceeds.
    VoidId {
        /// Void object ID of the InferenceOutput to use as input.
        id: InferenceOutputId,
        /// Optional generation cap. If `None`, mass applies its server default.
        limit: Option<u32>,
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
        potentiation: Potentiation,
        recv: ObjectId,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Potentiation {
    pub loss_up: f32,
    pub loss_down: f32,
    pub seed: u64,
}
