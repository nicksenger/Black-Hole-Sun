//! Shared types for the black-hole workspace.

use std::net::SocketAddr;

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
// Quark wire protocol (black-hole-quark <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the quark QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuarkIn {
    /// Start a new model instance with the provided stable ID.
    ///
    /// When `model_config` is `None`, quark uses server defaults. When present,
    /// the provided values override defaults for this specific model instance.
    Start {
        model_id: Uuid,
        model_config: Option<QuarkModelConfig>,
    },
    /// Perturb model weights in the positive direction.
    PerturbUp { model_id: Uuid, seed: u64 },
    /// Run inference on the input object stored in void.
    /// Returns QuarkOut::Inferred(output_id).
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
    /// Register a one-hop tunnel worker with a root quark.
    RegisterTunnel {
        /// Address where the worker quark accepts forwarded tunnel requests.
        worker_addr: SocketAddr,
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

/// Response sent by the quark server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum QuarkOut {
    /// Acknowledges a lifecycle, perturb, or optimize step.
    Ack,
    /// Inference complete; contains the void object ID of the output.
    Inferred { output_id: ObjectId },
    /// Checkpoint upload complete; contains the void object ID of model weights.
    Checkpointed { checkpoint_id: ObjectId },
    /// Runtime model parameters for a running instance.
    ModelParams { params: QuarkModelParams },
    /// Tunnel worker registration complete; contains root-issued auth token.
    TunnelRegistered { token: Uuid },
    /// Error from any operation.
    Error { message: String },
}

/// Runtime model parameters resolved for a running quark model instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuarkModelParams {
    pub inference_limit: u32,
    pub top_k: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub training_lr: f64,
    pub training_epsilon: f64,
    pub is_frozen: bool,
    pub optimize_steps: u32,
    pub oscillation_period_steps: Option<u32>,
    pub oscillation_train_steps: Option<u32>,
    pub oscillation_phase_steps: Option<u32>,
    pub oscillation_warmup_steps: Option<u32>,
}

/// Forwardable model operation used for root->worker tunnel requests.
#[derive(Debug, Serialize, Deserialize)]
pub enum TunnelRequest {
    Start {
        model_id: Uuid,
        model_config: Option<QuarkModelConfig>,
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

/// Per-model-instance quark configuration overrides.
///
/// Each field is optional; omitted values fall back to server defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuarkModelConfig {
    pub top_k: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub inference_limit: Option<u32>,
    pub training_lr: Option<f64>,
    pub training_epsilon: Option<f64>,
    pub frozen: Option<bool>,
    /// Optional optimize-step period for train/freeze oscillation scheduling.
    ///
    /// When set (with `oscillation_train_steps`), quark applies a deterministic
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
    /// When `None`, quark loads weights from its configured server model path.
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

impl DarkToken {
    pub fn one_hot(token_id: u32) -> Self {
        Self {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
            }],
        }
    }
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
        /// Optional generation cap. If `None`, quark applies its server default.
        limit: Option<u32>,
    },
    /// Reference to an existing InferenceOutput in void.
    /// Quark downloads it, converts the results to dark input, and proceeds.
    VoidId {
        /// Void object ID of the InferenceOutput to use as input.
        id: InferenceOutputId,
        /// Optional generation cap. If `None`, quark applies its server default.
        limit: Option<u32>,
    },
}

// ---------------------------------------------------------------------------
// Inference output format (stored in void objects)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOutput(pub Vec<DarkToken>);
impl SequenceOutput {
    /// Removes trailing pad tokens (248044, 248046) from the sequence.
    pub fn trim_padding(&mut self) {
        if let Some(last_non_padding) = self
            .0
            .iter()
            .rposition(|dt| dt.predicted != 248044 && dt.predicted != 248046)
        {
            self.0.truncate(last_non_padding + 1);
        } else {
            self.0.clear();
        }
    }

    /// Pads the sequence from the start to the specified length
    pub fn pad_start_to(&mut self, len: usize) {
        let mut pad = vec![
            DarkToken {
                predicted: 248044,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248044,
                    log_prob: 0.0
                }]
            };
            len.saturating_sub(self.0.len())
        ];
        pad.append(&mut self.0);
        self.0 = pad;
    }
}

/// Serializable inference output stored in void objects.
/// Contains per-sequence results from a batched forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub results: Vec<SequenceOutput>,
}

impl InferenceOutput {
    pub fn pad_start(&mut self) {
        let max = self
            .results
            .iter()
            .map(|seq| seq.0.len())
            .max()
            .unwrap_or_default();
        for seq in &mut self.results {
            seq.pad_start_to(max);
        }
    }

    pub fn trim_padding(&mut self) {
        for seq in &mut self.results {
            seq.trim_padding();
        }
    }

    /// Trims padding from sequences, then frames them with the provided tokens
    /// and pads from start to the max length
    pub fn frame(&mut self, before: Vec<DarkToken>, after: Vec<DarkToken>) {
        self.trim_padding();
        for seq in &mut self.results {
            let mut new = before.clone();
            new.append(&mut seq.0);
            new.append(&mut after.clone());
            seq.0 = new;
        }
        self.pad_start();
    }

    /// Frames each sequence with the corresponding before/after sequence from the provided iterator
    pub fn frame_with<T: Iterator<Item = (Vec<DarkToken>, Vec<DarkToken>)>>(&mut self, frames: T) {
        self.trim_padding();
        for (seq, (before, mut after)) in &mut self.results.iter_mut().zip(frames) {
            let mut new = before;
            new.append(&mut seq.0);
            new.append(&mut after);
            seq.0 = new;
        }
        self.pad_start();
    }
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

#[cfg(test)]
mod test {
    use super::*;

    fn tok(token_id: u32) -> DarkToken {
        DarkToken {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
            }],
        }
    }

    #[test]
    fn test_trim() {
        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248046, 248044, 248044, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248046, 248044, 248046, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248044, 248044, 248044, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput([1, 2, 3, 4, 5].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput([].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![]
        );

        let mut seq = SequenceOutput([248044, 248044, 248044].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![]
        );

        let mut seq = SequenceOutput([248044, 248046].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![]
        );

        let mut seq = SequenceOutput(vec![
            DarkToken {
                predicted: 248044,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248044,
                    log_prob: 0.0,
                }],
            },
            DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.4,
                }],
            },
            DarkToken {
                predicted: 248044,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248044,
                    log_prob: 0.0,
                }],
            },
            DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.8,
                }],
            },
        ]);
        seq.trim_padding();
        assert_eq!(
            seq.0
                .into_iter()
                .map(|dt| dt.dark_knowledge[0].log_prob)
                .collect::<Vec<_>>(),
            vec![]
        );
    }

    #[test]
    fn test_frame() {
        let mut out = InferenceOutput {
            results: vec![
                SequenceOutput(
                    [1, 2, 3, 4, 5, 248044, 248044, 248044, 248044]
                        .into_iter()
                        .map(tok)
                        .collect(),
                ),
                SequenceOutput(
                    [1, 2, 3, 248046, 248044, 248044, 248044]
                        .into_iter()
                        .map(tok)
                        .collect(),
                ),
                SequenceOutput([1, 248044, 248044].into_iter().map(tok).collect()),
            ],
        };
        out.frame(
            vec![DarkToken {
                predicted: 248045,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248045,
                    log_prob: 0.0,
                }],
            }],
            vec![DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.0,
                }],
            }],
        );

        assert_eq!(
            out.results[0]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248045, 1, 2, 3, 4, 5, 248046]
        );
        assert_eq!(
            out.results[1]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248044, 248044, 248045, 1, 2, 3, 248046]
        );
        assert_eq!(
            out.results[2]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248044, 248044, 248044, 248044, 248045, 1, 248046]
        );
    }
}
