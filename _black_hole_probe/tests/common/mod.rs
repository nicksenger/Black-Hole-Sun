#![allow(dead_code)]
use std::net::SocketAddr;
use std::sync::Arc;

use black_hole_sun::{ObjectId, QuarkIn, QuarkOut};
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tracing::warn;

// ─── Wire protocol for void (mirrors black-hole-void) ─────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub enum VoidIn {
    Upload { data: Vec<u8> },
    UploadWith { id: ObjectId, data: Vec<u8> },
    Download { id: ObjectId },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum VoidOut {
    Uploaded { id: ObjectId },
    Downloaded { data: Vec<u8> },
    Error { message: String },
}

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
    let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

    let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let mut endpoint = quinn::Endpoint::client(local_addr).unwrap();
    endpoint.set_default_client_config(client_config);
    endpoint
}

pub async fn void_upload(endpoint: &quinn::Endpoint, addr: SocketAddr, data: Vec<u8>) -> ObjectId {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &VoidIn::Upload { data }).await;
    let resp: VoidOut = read_frame(&mut recv).await;
    match resp {
        VoidOut::Uploaded { id } => id,
        VoidOut::Error { message } => panic!("void upload error: {message}"),
        _ => panic!("unexpected void response for upload"),
    }
}

pub async fn void_upload_with(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    id: ObjectId,
    data: Vec<u8>,
) -> ObjectId {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &VoidIn::UploadWith { id, data }).await;
    let resp: VoidOut = read_frame(&mut recv).await;
    match resp {
        VoidOut::Uploaded { id } => id,
        VoidOut::Error { message } => panic!("void upload error: {message}"),
        _ => panic!("unexpected void response for upload with"),
    }
}

pub async fn void_download(endpoint: &quinn::Endpoint, addr: SocketAddr, id: ObjectId) -> Vec<u8> {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &VoidIn::Download { id }).await;
    let resp: VoidOut = read_frame(&mut recv).await;
    match resp {
        VoidOut::Downloaded { data } => data,
        VoidOut::Error { message } => panic!("void download error: {message}"),
        _ => panic!("unexpected void response for download"),
    }
}

pub async fn void_download_result(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    id: ObjectId,
) -> Result<Vec<u8>, String> {
    let server_name = "localhost";
    let conn = match endpoint.connect(addr, &server_name).unwrap().await {
        Ok(c) => c,
        Err(e) => return Err(format!("connect failed: {e}")),
    };
    let (mut send, mut recv) = match conn.open_bi().await {
        Ok(s) => s,
        Err(e) => return Err(format!("open_bi failed: {e}")),
    };
    send_frame(&mut send, &VoidIn::Download { id }).await;
    let resp: VoidOut = read_frame(&mut recv).await;
    match resp {
        VoidOut::Downloaded { data } => Ok(data),
        VoidOut::Error { message } => Err(message),
        _ => Err("unexpected void response for download".into()),
    }
}

pub async fn quark_infer_result(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    input_id: ObjectId,
) -> Result<ObjectId, String> {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::Infer { input_id }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Inferred { output_id } => Ok(output_id),
        QuarkOut::Error { message } => Err(message),
        _ => Err("unexpected quark response for infer".to_string()),
    }
}

pub async fn quark_infer(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    input_id: ObjectId,
) -> ObjectId {
    quark_infer_result(endpoint, addr, input_id)
        .await
        .unwrap_or_else(|message| panic!("quark infer error: {message}"))
}

pub async fn quark_perturb_up_result(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    seed: u64,
) -> Result<(), String> {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::PerturbUp { seed }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => Ok(()),
        QuarkOut::Error { message } => Err(message),
        _ => Err("unexpected quark response for perturb_up".to_string()),
    }
}

pub async fn quark_perturb_up(endpoint: &quinn::Endpoint, addr: SocketAddr, seed: u64) {
    quark_perturb_up_result(endpoint, addr, seed)
        .await
        .unwrap_or_else(|message| panic!("quark perturb_up error: {message}"));
}

pub async fn quark_perturb_down_result(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
) -> Result<(), String> {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::PerturbDown).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => Ok(()),
        QuarkOut::Error { message } => Err(message),
        _ => Err("unexpected quark response for perturb_down".to_string()),
    }
}

pub async fn quark_perturb_down(endpoint: &quinn::Endpoint, addr: SocketAddr) {
    quark_perturb_down_result(endpoint, addr)
        .await
        .unwrap_or_else(|message| panic!("quark perturb_down error: {message}"));
}

pub async fn quark_optimize_result(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    loss_up: f32,
    loss_down: f32,
) -> Result<(), String> {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::Optimize { loss_up, loss_down }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => Ok(()),
        QuarkOut::Error { message } => Err(message),
        _ => Err("unexpected quark response for optimize".to_string()),
    }
}

pub async fn quark_optimize(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    loss_up: f32,
    loss_down: f32,
) {
    quark_optimize_result(endpoint, addr, loss_up, loss_down)
        .await
        .unwrap_or_else(|message| panic!("quark optimize error: {message}"));
}

async fn send_frame(send: &mut quinn::SendStream, msg: &impl Serialize) {
    let payload = to_allocvec(msg).expect("failed to encode frame");
    let len = u32::try_from(payload.len()).expect("frame too large");
    send.write_all(&len.to_be_bytes())
        .await
        .expect("failed to write frame len");
    send.write_all(&payload)
        .await
        .expect("failed to write frame payload");
}

async fn read_frame<T: for<'de> Deserialize<'de>>(recv: &mut quinn::RecvStream) -> T {
    let len = recv.read_u32().await.expect("failed to read frame len") as usize;
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .expect("failed to read frame payload");
    from_bytes(&payload).expect("failed to decode frame")
}

// ─── Token decoding ───────────────────────────────────────────────────────────

use black_hole_sun::DarkToken;

/// Decode a sequence of DarkToken predicted IDs into text using a tokenizer.
pub fn decode_dark_tokens(tokenizer: &tokenizers::Tokenizer, tokens: &[DarkToken]) -> String {
    let ids: Vec<u32> = tokens.iter().map(|t| t.predicted).collect();
    tokenizer
        .decode(&ids, true)
        .unwrap_or_else(|_| ids.iter().map(|id| id.to_string()).collect())
}

pub fn print_inference_output(
    label: &str,
    output: &black_hole_sun::InferenceOutput,
    seq_idx: usize,
    tokenizer: &tokenizers::Tokenizer,
) {
    let output_text = decode_dark_tokens(tokenizer, &output.results[seq_idx].0);
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
    let _ = black_hole_sun::init_tracing();
}
