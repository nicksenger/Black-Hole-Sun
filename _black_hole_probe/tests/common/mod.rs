#![allow(dead_code)]

mod sun;

pub use sun::{Generator, Policy};

use std::sync::Once;
use tracing::warn;

pub use black_hole_sun::make_client_endpoint;

// ─── Token decoding ───────────────────────────────────────────────────────────

use black_hole_sun::Tokenizer;

pub fn print_inference_output(
    label: &str,
    output: &black_hole_sun::InferenceOutput,
    seq_idx: usize,
    tokenizer: &Tokenizer,
) {
    let output_text = tokenizer.decode(&output.results[seq_idx].0);
    println!("{}: {}", label, output_text);
}

// ─── Model path guard ─────────────────────────────────────────────────────────

/// Returns the model path if set, otherwise logs a warning and returns None.
pub fn require_model_path(test_name: &str) -> Option<String> {
    match std::env::var("BLACK_HOLE_PROBE_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) => {
            warn!(
                test = %test_name,
                "Skipping test: BLACK_HOLE_PROBE_MODEL_PATH is not set"
            );
            None
        }
    }
}

/// Initialize tracing for the test process (delegates to black_hole_sun).
pub fn init_tracing() {
    static RUSTLS_PROVIDER: Once = Once::new();
    RUSTLS_PROVIDER.call_once(|| {
        // rustls is built with default-features = false + ring, so tests must
        // install the process-wide crypto provider before TLS client setup.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
    let _ = black_hole_sun::init_tracing();
}
