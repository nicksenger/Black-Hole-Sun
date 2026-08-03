mod common;

use std::path::PathBuf;

use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::QuarkServerBuilder;
use black_hole_sun::VoidServerBuilder;
use black_hole_sun::{InferenceInput, InferenceRequest, DarkToken, LogitEntry};
use postcard::{from_bytes, to_allocvec};

use common::*;

/// Download the Qwen tokenizer from HuggingFace.
fn get_tokenizer() -> tokenizers::Tokenizer {
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
    tokenizers::Tokenizer::from_file(tokenizer_file)
        .expect("failed to load tokenizer")
}

/// Start void and quark servers, returning their local addresses and abort handles.
async fn start_servers(
    model_path: &str,
) -> (std::net::SocketAddr, tokio::task::AbortHandle, std::net::SocketAddr, tokio::task::AbortHandle) {
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");
    let void_abort = void_handle.abort_handle();

    let quark_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");
    let quark_abort = quark_handle.abort_handle();

    // Drop the join handles so tasks run independently (abort via handles below).
    drop(void_handle);
    drop(quark_handle);

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    (void_local, void_abort, quark_local, quark_abort)
}

#[tokio::test]
async fn inference() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("inference") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    let tokenizer = get_tokenizer();

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

    let output_id = quark_infer(&quark_client, quark_local, input_id).await;

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

    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}

#[tokio::test]
async fn dark_inference() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("dark_inference") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    println!("Input text: {input_text}");

    let tokenizer = get_tokenizer();

    let tokens: Vec<u32> = tokenizer
        .encode(input_text, false)
        .expect("failed to tokenize input")
        .get_ids()
        .iter()
        .map(|&id| id as u32)
        .collect();

    let dark_tokens: Vec<DarkToken> = tokens
        .iter()
        .map(|&token_id| DarkToken {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
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

    let output_id = quark_infer(&quark_client, quark_local, input_id).await;

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

    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}
