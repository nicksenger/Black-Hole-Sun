//! Model configuration traits for per-instance quark start settings.

use black_hole_spec::QuarkModelConfig;

/// Compile-time model configuration for a cell-owned quark instance.
///
/// Implementors expose optional overrides as associated constants. Any constant
/// left as `None` falls back to the quark server default for that field.
pub trait ModelConfig {
    const TOP_K: Option<usize> = None;
    const TEMPERATURE: Option<f64> = None;
    const TOP_P: Option<f64> = None;
    const REPEAT_PENALTY: Option<f32> = None;
    const PRESENCE_PENALTY: Option<f32> = None;
    const INFERENCE_LIMIT: Option<u32> = None;
    const TRAINING_LR: Option<f64> = None;
    const TRAINING_EPSILON: Option<f64> = None;
    const FROZEN: Option<bool> = None;

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
        };

        let has_any_override = config.top_k.is_some()
            || config.temperature.is_some()
            || config.top_p.is_some()
            || config.repeat_penalty.is_some()
            || config.presence_penalty.is_some()
            || config.inference_limit.is_some()
            || config.training_lr.is_some()
            || config.training_epsilon.is_some()
            || config.frozen.is_some();
        if has_any_override {
            Some(config)
        } else {
            None
        }
    }
}

/// Default model configuration that passes through all quark server defaults.
pub struct DefaultConfig;

impl ModelConfig for DefaultConfig {}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    struct FrozenConfig;
    impl ModelConfig for FrozenConfig {
        const FROZEN: Option<bool> = Some(true);
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
}
