//! Model configuration traits for per-instance quark start settings.

use black_hole_spec::{
    ObjectId, QuarkErrorFeedbackConfig, QuarkErrorFeedbackMode, QuarkModelConfig,
};

/// Compile-time oscillation schedule for train/freeze windows.
///
/// Implementors expose optional schedule constants. Leaving all values as
/// `None` disables oscillation and keeps the model's frozen state static.
pub trait OscillationSchedule {
    const PERIOD_STEPS: Option<u32> = None;
    const TRAIN_STEPS: Option<u32> = None;
    const PHASE_STEPS: Option<u32> = None;
    const WARMUP_STEPS: Option<u32> = None;
}

/// Default oscillation schedule that applies no oscillation overrides.
#[derive(Clone)]
pub struct NoOscillation;

impl OscillationSchedule for NoOscillation {}

/// Compile-time QuZO error-feedback override policy.
///
/// Implementors can leave `MODE` as `None` to emit no per-instance override.
/// When set to a mode variant, decay/gain and replay steps are encoded into
/// `QuarkModelConfig::training_error_feedback`.
pub trait ErrorFeedbackPolicy {
    const MODE: Option<QuarkErrorFeedbackMode> = None;
    const DECAY: f64 = 0.9;
    const GAIN: f64 = 1.0;
    const REPLAY_STEPS: u32 = 50;

    fn quark_error_feedback() -> Option<QuarkErrorFeedbackConfig> {
        match Self::MODE {
            None => None,
            Some(QuarkErrorFeedbackMode::Off) => Some(QuarkErrorFeedbackConfig::Off),
            Some(QuarkErrorFeedbackMode::Persistent) => {
                Some(QuarkErrorFeedbackConfig::Persistent {
                    decay: Self::DECAY,
                    gain: Self::GAIN,
                })
            }
            Some(QuarkErrorFeedbackMode::Replay) => Some(QuarkErrorFeedbackConfig::Replay {
                steps: Self::REPLAY_STEPS,
                decay: Self::DECAY,
                gain: Self::GAIN,
            }),
        }
    }
}

/// Default error-feedback policy that emits no overrides.
#[derive(Clone)]
pub struct NoErrorFeedback;

impl ErrorFeedbackPolicy for NoErrorFeedback {}

/// Compile-time model configuration for a cell-owned quark instance.
///
/// Implementors expose optional overrides as associated constants. Any constant
/// left as `None` falls back to the quark server default for that field.
pub trait ModelConfig {
    type Oscillation: OscillationSchedule;
    type ErrorFeedback: ErrorFeedbackPolicy;

    const TOP_K: Option<usize> = None;
    const TEMPERATURE: Option<f64> = None;
    const TOP_P: Option<f64> = None;
    const REPEAT_PENALTY: Option<f32> = None;
    const PRESENCE_PENALTY: Option<f32> = None;
    const INFERENCE_LIMIT: Option<u32> = None;
    const TRAINING_LR: Option<f64> = None;
    const TRAINING_EPSILON: Option<f64> = None;
    const TRAINING_Z_LOSS: Option<f64> = None;
    const TRAINING_LB_LOSS: Option<f64> = None;
    const TRAINING_CLIP_THRESHOLD: Option<f64> = None;
    const FROZEN: Option<bool> = None;
    const CHECKPOINT: Option<u128> = None;

    fn quark_model_config() -> Option<QuarkModelConfig> {
        let config = QuarkModelConfig {
            top_k: Self::TOP_K,
            temperature: Self::TEMPERATURE,
            top_p: Self::TOP_P,
            repeat_penalty: Self::REPEAT_PENALTY,
            presence_penalty: Self::PRESENCE_PENALTY,
            inference_limit: Self::INFERENCE_LIMIT,
            training_lr: Self::TRAINING_LR,
            training_epsilon: Self::TRAINING_EPSILON,
            training_z_loss: Self::TRAINING_Z_LOSS,
            training_lb_loss: Self::TRAINING_LB_LOSS,
            training_clip_threshold: Self::TRAINING_CLIP_THRESHOLD,
            training_error_feedback:
                <Self::ErrorFeedback as ErrorFeedbackPolicy>::quark_error_feedback(),
            frozen: Self::FROZEN,
            oscillation_period_steps: <Self::Oscillation as OscillationSchedule>::PERIOD_STEPS,
            oscillation_train_steps: <Self::Oscillation as OscillationSchedule>::TRAIN_STEPS,
            oscillation_phase_steps: <Self::Oscillation as OscillationSchedule>::PHASE_STEPS,
            oscillation_warmup_steps: <Self::Oscillation as OscillationSchedule>::WARMUP_STEPS,
            checkpoint_id: Self::CHECKPOINT.map(ObjectId::from_u128),
        };

        let has_any_override = config.top_k.is_some()
            || config.temperature.is_some()
            || config.top_p.is_some()
            || config.repeat_penalty.is_some()
            || config.presence_penalty.is_some()
            || config.inference_limit.is_some()
            || config.training_lr.is_some()
            || config.training_epsilon.is_some()
            || config.training_z_loss.is_some()
            || config.training_lb_loss.is_some()
            || config.training_clip_threshold.is_some()
            || config.training_error_feedback.is_some()
            || config.frozen.is_some()
            || config.oscillation_period_steps.is_some()
            || config.oscillation_train_steps.is_some()
            || config.oscillation_phase_steps.is_some()
            || config.oscillation_warmup_steps.is_some()
            || config.checkpoint_id.is_some();
        if has_any_override {
            Some(config)
        } else {
            None
        }
    }
}

/// Default model configuration that passes through all quark server defaults.
pub struct DefaultConfig;

impl ModelConfig for DefaultConfig {
    type Oscillation = NoOscillation;
    type ErrorFeedback = NoErrorFeedback;
}

#[cfg(test)]
mod tests {
    use super::{ErrorFeedbackPolicy, ModelConfig, OscillationSchedule};
    use black_hole_spec::{QuarkErrorFeedbackConfig, QuarkErrorFeedbackMode};

    struct FrozenConfig;
    impl ModelConfig for FrozenConfig {
        type Oscillation = super::NoOscillation;
        type ErrorFeedback = super::NoErrorFeedback;
        const FROZEN: Option<bool> = Some(true);
    }

    struct CheckpointConfig;
    impl ModelConfig for CheckpointConfig {
        type Oscillation = super::NoOscillation;
        type ErrorFeedback = super::NoErrorFeedback;
        const CHECKPOINT: Option<u128> = Some(42);
    }

    struct TrainingOverridesConfig;
    impl ModelConfig for TrainingOverridesConfig {
        type Oscillation = super::NoOscillation;
        type ErrorFeedback = super::NoErrorFeedback;
        const TRAINING_Z_LOSS: Option<f64> = Some(0.02);
        const TRAINING_LB_LOSS: Option<f64> = Some(0.03);
        const TRAINING_CLIP_THRESHOLD: Option<f64> = Some(1.5);
    }

    struct WindowedOscillation;
    impl OscillationSchedule for WindowedOscillation {
        const PERIOD_STEPS: Option<u32> = Some(10);
        const TRAIN_STEPS: Option<u32> = Some(3);
        const PHASE_STEPS: Option<u32> = Some(2);
        const WARMUP_STEPS: Option<u32> = Some(20);
    }

    struct OscillationConfig;
    impl ModelConfig for OscillationConfig {
        type Oscillation = WindowedOscillation;
        type ErrorFeedback = super::NoErrorFeedback;
    }

    struct ReplayErrorFeedback;
    impl ErrorFeedbackPolicy for ReplayErrorFeedback {
        const MODE: Option<QuarkErrorFeedbackMode> = Some(QuarkErrorFeedbackMode::Replay);
        const DECAY: f64 = 0.85;
        const GAIN: f64 = 0.7;
        const REPLAY_STEPS: u32 = 64;
    }

    struct ErrorFeedbackConfig;
    impl ModelConfig for ErrorFeedbackConfig {
        type Oscillation = super::NoOscillation;
        type ErrorFeedback = ReplayErrorFeedback;
    }

    #[test]
    fn default_config_emits_no_overrides() {
        assert!(super::DefaultConfig::quark_model_config().is_none());
    }

    #[test]
    fn frozen_override_emits_model_config() {
        let config = FrozenConfig::quark_model_config().expect("expected override config");
        assert_eq!(config.frozen, Some(true));
    }

    #[test]
    fn checkpoint_override_emits_model_config() {
        let config = CheckpointConfig::quark_model_config().expect("expected override config");
        assert_eq!(
            config.checkpoint_id,
            Some(black_hole_spec::ObjectId::from_u128(42))
        );
    }

    #[test]
    fn training_overrides_emit_model_config() {
        let config =
            TrainingOverridesConfig::quark_model_config().expect("expected override config");
        assert_eq!(config.training_z_loss, Some(0.02));
        assert_eq!(config.training_lb_loss, Some(0.03));
        assert_eq!(config.training_clip_threshold, Some(1.5));
    }

    #[test]
    fn oscillation_overrides_emit_model_config() {
        let config = OscillationConfig::quark_model_config().expect("expected override config");
        assert_eq!(config.oscillation_period_steps, Some(10));
        assert_eq!(config.oscillation_train_steps, Some(3));
        assert_eq!(config.oscillation_phase_steps, Some(2));
        assert_eq!(config.oscillation_warmup_steps, Some(20));
    }

    #[test]
    fn error_feedback_overrides_emit_model_config() {
        let config = ErrorFeedbackConfig::quark_model_config().expect("expected override config");
        assert_eq!(
            config.training_error_feedback,
            Some(QuarkErrorFeedbackConfig::Replay {
                steps: 64,
                decay: 0.85,
                gain: 0.7,
            })
        );
    }
}
