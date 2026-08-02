use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::QuarkServerBuilder;
use black_hole_sun::VoidServerBuilder;
use black_hole_sun::{ObjectId, QuarkIn, InferenceInput, InferenceRequest, QuarkOut, DarkToken, LogitEntry};
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

// ─── Wire protocol for void (mirrors black-hole-void) ─────────────────────────

#[derive(Debug, Serialize, Deserialize)]
enum VoidIn {
    Upload { data: Vec<u8> },
    Download { id: ObjectId },
}

#[derive(Debug, Serialize, Deserialize)]
enum VoidOut {
    Uploaded { id: ObjectId },
    Downloaded { data: Vec<u8> },
    Error { message: String },
}

// ─── No-op cert verifier (self-signed certs in local dev) ──────────────────────

#[derive(Debug)]
struct NoCertVerifier;

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

async fn make_client_endpoint() -> quinn::Endpoint {
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

async fn void_upload(endpoint: &quinn::Endpoint, addr: SocketAddr, data: Vec<u8>) -> ObjectId {
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

async fn void_download(endpoint: &quinn::Endpoint, addr: SocketAddr, id: ObjectId) -> Vec<u8> {
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

async fn quark_infer(endpoint: &quinn::Endpoint, addr: SocketAddr, input_id: ObjectId) -> ObjectId {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::Infer { input_id }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Inferred { output_id } => output_id,
        QuarkOut::Error { message } => panic!("quark infer error: {message}"),
        _ => panic!("unexpected quark response for infer"),
    }
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

// ─── Integration test ─────────────────────────────────────────────────────────

/// Decode a sequence of DarkToken predicted IDs into text using a tokenizer.
fn decode_dark_tokens(tokenizer: &tokenizers::Tokenizer, tokens: &[DarkToken]) -> String {
    let ids: Vec<u32> = tokens.iter().map(|t| t.predicted).collect();
    tokenizer
        .decode(&ids, true)
        .unwrap_or_else(|_| ids.iter().map(|id| id.to_string()).collect())
}

#[ignore = "Sanity check — performs real model inference"]
#[tokio::test]
async fn inference() {
    // Install rustls default crypto provider before any TLS config.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    black_hole_sun::init_tracing().ok();

    // 1. Start void server on a random port with in-memory store.
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");

    // 2. Start quark server on a random port, pointing at void.
    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to a compatible GGUF file");
    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(&model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");

    // Give servers a moment to be ready.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // 3. Create client endpoints for void and quark.
    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    // Download tokenizer from HuggingFace for decoding output tokens.
    let tokenizer_repo = "Qwen/Qwen3.5-0.8B".to_string();
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub api");
    let repo = api.repo(hf_hub::Repo::with_revision(
        tokenizer_repo.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_file = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json from HuggingFace");
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
        .expect("failed to load tokenizer");

    // 4. Upload inference input to void (batch size 2).
    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    println!("Input text: {input_text}");
    let request = InferenceRequest::Sequences {
        sequences: vec![
            vec![InferenceInput::Text(input_text.into())],
            vec![InferenceInput::Text(input_text.into())],
        ],
        limit: 100,
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_upload(&void_client, void_local, request_bytes).await;

    // 5. Send Infer request to quark.
    let output_id = quark_infer(&quark_client, quark_local, input_id).await;

    // 6. Download inference output from void and assert we got predictions.
    let output_bytes = void_download(&void_client, void_local, output_id).await;
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 2, "expected 2 batch results");

    for (i, seq_result) in output.results.iter().enumerate() {
        let label = i + 1;
        assert!(
            !seq_result.0.is_empty(),
            "output {label} has zero predictions"
        );
        let output_text = decode_dark_tokens(&tokenizer, &seq_result.0);

        println!("Output {label}: {output_text}");
        assert!(
            !output_text.is_empty(),
            "output {label} has no decoded text"
        );
    }

    // Cleanup: drop endpoints to close QUIC connections, then abort server tasks.
    drop(void_client);
    drop(quark_client);
    void_handle.abort();
    quark_handle.abort();
}


// ─── Dark inference integration test ──────────────────────────────────────

#[ignore = "Sanity check — performs dark-prompt model inference"]
#[tokio::test]
async fn dark_inference() {
    // Install rustls default crypto provider before any TLS config.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    black_hole_sun::init_tracing().ok();

    // 1. Start void server on a random port with in-memory store.
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");

    // 2. Start quark server on a random port, pointing at void.
    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to a compatible GGUF file");
    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(&model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");

    // Give servers a moment to be ready.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // 3. Create client endpoints for void and quark.
    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    // 4. Tokenize input text and convert to dark tokens.
    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    println!("Input text: {input_text}");

    // Download tokenizer from HuggingFace.
    let tokenizer_repo = "Qwen/Qwen3.5-0.8B".to_string();
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub api");
    let repo = api.repo(hf_hub::Repo::with_revision(
        tokenizer_repo.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_file = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json from HuggingFace");
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
        .expect("failed to load tokenizer");

    let tokens: Vec<u32> = tokenizer
        .encode(input_text, false)
        .expect("failed to tokenize input")
        .get_ids()
        .iter()
        .map(|&id| id as u32)
        .collect();

    // Build dark tokens: each token has probability 1.0 (log_prob = 0.0).
    let dark_tokens: Vec<DarkToken> = tokens
        .iter()
        .map(|&token_id| DarkToken {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0, // ln(1.0) = 0.0 → full probability on this token
            }],
        })
        .collect();

    let request = InferenceRequest::Sequences {
        sequences: vec![
            vec![InferenceInput::Dark(dark_tokens.clone())],
            vec![InferenceInput::Dark(dark_tokens)],
        ],
        limit: 100,
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_upload(&void_client, void_local, request_bytes).await;

    // 5. Send Infer request to quark.
    let output_id = quark_infer(&quark_client, quark_local, input_id).await;

    // 6. Download inference output from void and assert we got predictions.
    let output_bytes = void_download(&void_client, void_local, output_id).await;
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 2, "expected 2 batch results");

    for (i, seq_result) in output.results.iter().enumerate() {
        let label = i + 1;
        assert!(
            !seq_result.0.is_empty(),
            "output {label} has zero predictions"
        );
        let output_text = decode_dark_tokens(&tokenizer, &seq_result.0);

        println!("Output {label}: {output_text}");
        assert!(
            !output_text.is_empty(),
            "output {label} has no decoded text"
        );
    }

    // Cleanup: drop endpoints to close QUIC connections, then abort server tasks.
    drop(void_client);
    drop(quark_client);
    void_handle.abort();
    quark_handle.abort();
}

async fn quark_perturb_up(endpoint: &quinn::Endpoint, addr: SocketAddr, seed: u64) {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::PerturbUp { seed }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => {}
        QuarkOut::Error { message } => panic!("quark perturb_up error: {message}"),
        _ => panic!("unexpected quark response for perturb_up"),
    }
}

async fn quark_perturb_down(endpoint: &quinn::Endpoint, addr: SocketAddr) {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::PerturbDown).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => {}
        QuarkOut::Error { message } => panic!("quark perturb_down error: {message}"),
        _ => panic!("unexpected quark response for perturb_down"),
    }
}

async fn quark_optimize(
    endpoint: &quinn::Endpoint,
    addr: SocketAddr,
    loss_up: f32,
    loss_down: f32,
) {
    let server_name = "localhost";
    let conn = endpoint.connect(addr, &server_name).unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    send_frame(&mut send, &QuarkIn::Optimize { loss_up, loss_down }).await;
    let resp: QuarkOut = read_frame(&mut recv).await;
    match resp {
        QuarkOut::Ack => {}
        QuarkOut::Error { message } => panic!("quark optimize error: {message}"),
        _ => panic!("unexpected quark response for optimize"),
    }
}

fn print_inference_output(label: &str, output: &black_hole_sun::InferenceOutput, seq_idx: usize, tokenizer: &tokenizers::Tokenizer) {
    let output_text = decode_dark_tokens(tokenizer, &output.results[seq_idx].0);
    println!("{}: {}", label, output_text);
}

// ─── QuZO end-to-end sanity check ────────────────────────────────────────

#[ignore = "Sanity check — performs full QuZO optimization flow"]
#[tokio::test]
async fn optimization() {
    // Install rustls default crypto provider before any TLS config.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    black_hole_sun::init_tracing().ok();

    // 1. Start void server on a random port with in-memory store.
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");

    // 2. Start quark server on a random port, pointing at void.
    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to a compatible GGUF file");
    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(&model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");

    // Give servers a moment to be ready.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // 3. Create client endpoints for void and quark.
    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    // Download tokenizer from HuggingFace for decoding output tokens.
    let tokenizer_repo = "Qwen/Qwen3.5-0.8B".to_string();
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub api");
    let repo = api.repo(hf_hub::Repo::with_revision(
        tokenizer_repo.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_file = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json from HuggingFace");
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
        .expect("failed to load tokenizer");

    // 4. Upload inference input to void (same input as the idle inference test).
    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let input_text_2 =
        "A starship traveling at constant velocity measures a distance of 1,200 light-years to a distant galaxy. After covering half the distance, it detects an anomaly and must divert, adding 300 light-years to its route. How many total light-years will the journey be?";
    println!("Input text: {input_text}");
    let request = InferenceRequest::Sequences {
        sequences: vec![
            vec![InferenceInput::Text(input_text.into())],
            vec![InferenceInput::Text(input_text_2.into())],
        ],
        limit: 100,
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_upload(&void_client, void_local, request_bytes).await;

    // ─── QuZO flow: PerturbUp -> Infer -> PerturbDown -> Infer -> Optimize -> Infer ───

    // Step 1: PerturbUp
    println!("\n--- Step 1: PerturbUp (seed=42) ---");
    quark_perturb_up(&quark_client, quark_local, 42).await;

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_up = void_download(&void_client, void_local, output_id_up).await;
    let output_up: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_up).expect("failed to decode inference output (up)");
    print_inference_output("PerturbUp Inference 1", &output_up, 0, &tokenizer);
    print_inference_output("PerturbUp Inference 2", &output_up, 1, &tokenizer);
    assert!(
        !output_up.results[0].0.is_empty(),
        "up inference returned zero predictions"
    );
    assert!(
        !output_up.results[1].0.is_empty(),
        "up inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_up.results.len(), 2, "up inference should have 2 results for batch size 2");

    // Step 3: PerturbDown
    println!("\n--- Step 3: PerturbDown ---");
    quark_perturb_down(&quark_client, quark_local).await;

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_down = void_download(&void_client, void_local, output_id_down).await;
    let output_down: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_down).expect("failed to decode inference output (down)");
    print_inference_output("PerturbDown Inference 1", &output_down, 0, &tokenizer);
    print_inference_output("PerturbDown Inference 2", &output_down, 1, &tokenizer);
    assert!(
        !output_down.results[0].0.is_empty(),
        "down inference returned zero predictions"
    );
    assert!(
        !output_down.results[1].0.is_empty(),
        "down inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_down.results.len(), 2, "down inference should have 2 results for batch size 2");

    // Step 5: Optimize with fake loss values
    let fake_loss_up = 0.5f32;
    let fake_loss_down = 1.0f32;
    println!(
        "\n--- Step 5: Optimize (loss_up={}, loss_down={}) ---",
        fake_loss_up, fake_loss_down
    );
    quark_optimize(&quark_client, quark_local, fake_loss_up, fake_loss_down).await;

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_final = void_download(&void_client, void_local, output_id_final).await;
    let output_final: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_final).expect("failed to decode inference output (final)");
    print_inference_output("Post-Optimize Inference 1", &output_final, 0, &tokenizer);
    print_inference_output("Post-Optimize Inference 2", &output_final, 1, &tokenizer);
    assert!(
        !output_final.results[0].0.is_empty(),
        "final inference returned zero predictions"
    );
    assert!(
        !output_final.results[1].0.is_empty(),
        "final inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_final.results.len(), 2, "final inference should have 2 results for batch size 2");

    // Verify the output contains plausible text.
    let final_text = decode_dark_tokens(&tokenizer, &output_final.results[0].0);
    let final_text_2 = decode_dark_tokens(&tokenizer, &output_final.results[1].0);

    println!("\n--- Summary ---");
    println!("All QuZO steps completed successfully.");
    assert!(
        !final_text.is_empty(),
        "post-optimize predicted tokens had no decoded text"
    );
    assert!(
        !final_text_2.is_empty(),
        "post-optimize sequence 1 predicted tokens had no decoded text"
    );

    // Cleanup: drop endpoints to close QUIC connections, then abort server tasks.
    drop(void_client);
    drop(quark_client);
    void_handle.abort();
    quark_handle.abort();
}

// ─── Dark QuZO end-to-end sanity check ──────────────────────────────────────

#[ignore = "Sanity check — performs full QuZO optimization flow with dark inputs"]
#[tokio::test]
async fn dark_optimization() {
    // Install rustls default crypto provider before any TLS config.
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    black_hole_sun::init_tracing().ok();

    // 1. Start void server on a random port with in-memory store.
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");

    // 2. Start quark server on a random port, pointing at void.
    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to a compatible GGUF file");
    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(&model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");

    // Give servers a moment to be ready.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // 3. Create client endpoints for void and quark.
    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    // 4. Tokenize input texts and convert to dark tokens.
    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let input_text_2 =
        "A starship traveling at constant velocity measures a distance of 1,200 light-years to a distant galaxy. After covering half the distance, it detects an anomaly and must divert, adding 300 light-years to its route. How many total light-years will the journey be?";
    println!("Input text: {input_text}");

    // Download tokenizer from HuggingFace.
    let tokenizer_repo = "Qwen/Qwen3.5-0.8B".to_string();
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub api");
    let repo = api.repo(hf_hub::Repo::with_revision(
        tokenizer_repo.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_file = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json from HuggingFace");
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
        .expect("failed to load tokenizer");

    let fn_to_dark_tokens = |text: &str, tokenizer: &tokenizers::Tokenizer| -> Vec<DarkToken> {
        let tokens: Vec<u32> = tokenizer
            .encode(text, false)
            .expect("failed to tokenize input")
            .get_ids()
            .iter()
            .map(|&id| id as u32)
            .collect();
        tokens
            .iter()
            .map(|&token_id| DarkToken {
                predicted: token_id,
                dark_knowledge: vec![LogitEntry {
                    token_id,
                    log_prob: 0.0,
                }],
            })
            .collect()
    };

    let dark_tokens_1 = fn_to_dark_tokens(input_text, &tokenizer);
    let dark_tokens_2 = fn_to_dark_tokens(input_text_2, &tokenizer);

    let request = InferenceRequest::Sequences {
        sequences: vec![
            vec![InferenceInput::Dark(dark_tokens_1)],
            vec![InferenceInput::Dark(dark_tokens_2)],
        ],
        limit: 100,
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_upload(&void_client, void_local, request_bytes).await;

    // ─── QuZO flow: PerturbUp -> Infer -> PerturbDown -> Infer -> Optimize -> Infer ───

    // Step 1: PerturbUp
    println!("\n--- Step 1: PerturbUp (seed=42) ---");
    quark_perturb_up(&quark_client, quark_local, 42).await;

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_up = void_download(&void_client, void_local, output_id_up).await;
    let output_up: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_up).expect("failed to decode inference output (up)");
    print_inference_output("PerturbUp Inference 1", &output_up, 0, &tokenizer);
    print_inference_output("PerturbUp Inference 2", &output_up, 1, &tokenizer);
    assert!(
        !output_up.results[0].0.is_empty(),
        "up inference returned zero predictions"
    );
    assert!(
        !output_up.results[1].0.is_empty(),
        "up inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_up.results.len(), 2, "up inference should have 2 results for batch size 2");

    // Step 3: PerturbDown
    println!("\n--- Step 3: PerturbDown ---");
    quark_perturb_down(&quark_client, quark_local).await;

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_down = void_download(&void_client, void_local, output_id_down).await;
    let output_down: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_down).expect("failed to decode inference output (down)");
    print_inference_output("PerturbDown Inference 1", &output_down, 0, &tokenizer);
    print_inference_output("PerturbDown Inference 2", &output_down, 1, &tokenizer);
    assert!(
        !output_down.results[0].0.is_empty(),
        "down inference returned zero predictions"
    );
    assert!(
        !output_down.results[1].0.is_empty(),
        "down inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_down.results.len(), 2, "down inference should have 2 results for batch size 2");

    // Step 5: Optimize with fake loss values
    let fake_loss_up = 0.5f32;
    let fake_loss_down = 1.0f32;
    println!(
        "\n--- Step 5: Optimize (loss_up={}, loss_down={}) ---",
        fake_loss_up, fake_loss_down
    );
    quark_optimize(&quark_client, quark_local, fake_loss_up, fake_loss_down).await;

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_infer(&quark_client, quark_local, input_id).await;
    let output_bytes_final = void_download(&void_client, void_local, output_id_final).await;
    let output_final: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes_final).expect("failed to decode inference output (final)");
    print_inference_output("Post-Optimize Inference 1", &output_final, 0, &tokenizer);
    print_inference_output("Post-Optimize Inference 2", &output_final, 1, &tokenizer);
    assert!(
        !output_final.results[0].0.is_empty(),
        "final inference returned zero predictions"
    );
    assert!(
        !output_final.results[1].0.is_empty(),
        "final inference sequence 1 returned zero predictions"
    );
    assert_eq!(output_final.results.len(), 2, "final inference should have 2 results for batch size 2");

    // Verify the output contains plausible text.
    let final_text = decode_dark_tokens(&tokenizer, &output_final.results[0].0);
    let final_text_2 = decode_dark_tokens(&tokenizer, &output_final.results[1].0);

    println!("\n--- Summary ---");
    println!("All QuZO steps completed successfully.");
    assert!(
        !final_text.is_empty(),
        "post-optimize predicted tokens had no decoded text"
    );
    assert!(
        !final_text_2.is_empty(),
        "post-optimize sequence 1 predicted tokens had no decoded text"
    );

    // Cleanup: drop endpoints to close QUIC connections, then abort server tasks.
    drop(void_client);
    drop(quark_client);
    void_handle.abort();
    quark_handle.abort();
}
