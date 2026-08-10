#![allow(dead_code)]

mod sun;

pub use sun::{Generator, Policy};

use std::net::SocketAddr;
use std::sync::{Arc, Once};
use std::time::Duration;

use quinn::crypto::rustls::QuicClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};
use tracing::warn;

// ─── No-op cert verifier (self-signed certs in local dev) ──────────────────────

#[derive(Debug)]
pub struct NoCertVerifier;

impl rustls::client::danger::ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

// ─── QUIC client helpers ──────────────────────────────────────────────────────

pub async fn make_client_endpoint() -> quinn::Endpoint {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
        .with_no_client_auth();

    let quic_crypto = QuicClientConfig::try_from(Arc::new(crypto)).unwrap();
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));
    client_config.transport_config(Arc::new(transport));

    let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let mut endpoint = quinn::Endpoint::client(local_addr).unwrap();
    endpoint.set_default_client_config(client_config);
    endpoint
}

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
