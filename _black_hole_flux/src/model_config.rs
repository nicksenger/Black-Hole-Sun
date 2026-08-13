//! Model configuration traits for per-instance quark start settings.

use black_hole_spec::{ObjectId, QuarkModelConfig};

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
pub struct NoOscillation;

impl OscillationSchedule for NoOscillation {}

/// Compile-time model configuration for a cell-owned quark instance.
///
/// Implementors expose optional overrides as associated constants. Any constant
/// left as `None` falls back to the quark server default for that field.
pub trait ModelConfig {
    type Oscillation: OscillationSchedule;

    const TOP_K: Option<usize> = None;
    const TEMPERATURE: Option<f64> = None;
    const TOP_P: Option<f64> = None;
    const REPEAT_PENALTY: Option<f32> = None;
    const PRESENCE_PENALTY: Option<f32> = None;
    const INFERENCE_LIMIT: Option<u32> = None;
    const TRAINING_LR: Option<f64> = None;
    const TRAINING_EPSILON: Option<f64> = None;
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
}

#[cfg(test)]
mod tests {
    use super::{ModelConfig, OscillationSchedule};

    struct FrozenConfig;
    impl ModelConfig for FrozenConfig {
        type Oscillation = super::NoOscillation;
        const FROZEN: Option<bool> = Some(true);
    }

    struct CheckpointConfig;
    impl ModelConfig for CheckpointConfig {
        type Oscillation = super::NoOscillation;
        const CHECKPOINT: Option<u128> = Some(42);
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
    fn oscillation_overrides_emit_model_config() {
        let config = OscillationConfig::quark_model_config().expect("expected override config");
        assert_eq!(config.oscillation_period_steps, Some(10));
        assert_eq!(config.oscillation_train_steps, Some(3));
        assert_eq!(config.oscillation_phase_steps, Some(2));
        assert_eq!(config.oscillation_warmup_steps, Some(20));
    }
}
