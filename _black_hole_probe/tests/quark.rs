mod common;

use std::path::PathBuf;

use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::QuarkServerBuilder;
use black_hole_sun::VoidServerBuilder;
use black_hole_sun::{DarkToken, InferenceInput, InferenceRequest, LogitEntry};
use postcard::{from_bytes, to_allocvec};
use uuid::Uuid;

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
    tokenizers::Tokenizer::from_file(tokenizer_file).expect("failed to load tokenizer")
}

/// Start void and quark servers, returning their local addresses and abort handles.
async fn start_servers(
    model_path: &str,
) -> (
    std::net::SocketAddr,
    tokio::task::AbortHandle,
    std::net::SocketAddr,
    tokio::task::AbortHandle,
) {
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
async fn rejects_requests_for_unknown_model_instance() {
    init_tracing();

    let (quark_local, quark_handle) =
        QuarkServerBuilder::new(PathBuf::from("model-is-not-loaded-for-this-test"))
            .listen("127.0.0.1:0".parse().unwrap())
            .serve()
            .await
            .expect("failed to start quark server");
    let quark_abort = quark_handle.abort_handle();
    drop(quark_handle);

    let client = make_client_endpoint().await;
    let model_id = Uuid::new_v4();

    for _ in 0..2 {
        let error = quark_start_result(&client, quark_local, model_id)
            .await
            .expect_err("invalid model path should fail to start");
        assert!(
            error.contains("Model path does not exist"),
            "unexpected start error: {error}"
        );
    }

    for result in [
        quark_infer_result(&client, quark_local, model_id, Uuid::nil())
            .await
            .map(|_| ()),
        quark_perturb_up_result(&client, quark_local, model_id, 42).await,
        quark_perturb_down_result(&client, quark_local, model_id).await,
        quark_optimize_result(&client, quark_local, model_id, 0.0, 0.0).await,
        quark_shutdown_result(&client, quark_local, model_id).await,
    ] {
        let error = result.expect_err("unknown model request should fail");
        assert!(
            error.contains("is not running"),
            "unexpected quark error: {error}"
        );
    }

    quark_abort.abort();
}

#[tokio::test]
async fn inference() {
    init_tracing();

    let model_path = match require_model_path("inference") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;
    let model_id = Uuid::new_v4();
    quark_start(&quark_client, quark_local, model_id).await;

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

    let output_id = quark_infer(&quark_client, quark_local, model_id, input_id).await;

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

    quark_shutdown(&quark_client, quark_local, model_id).await;
    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}

#[tokio::test]
async fn dark_inference() {
    init_tracing();

    let model_path = match require_model_path("dark_inference") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;
    let model_id = Uuid::new_v4();
    quark_start(&quark_client, quark_local, model_id).await;

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

    let output_id = quark_infer(&quark_client, quark_local, model_id, input_id).await;

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

    quark_shutdown(&quark_client, quark_local, model_id).await;
    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}

#[tokio::test]
async fn optimization() {
    init_tracing();

    let model_path = match require_model_path("optimization") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;
    let model_id = Uuid::new_v4();
    quark_start(&quark_client, quark_local, model_id).await;

    let tokenizer = get_tokenizer();

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
    quark_perturb_up(&quark_client, quark_local, model_id, 42).await;

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_up.results.len(),
        2,
        "up inference should have 2 results for batch size 2"
    );

    // Step 3: PerturbDown
    println!("\n--- Step 3: PerturbDown ---");
    quark_perturb_down(&quark_client, quark_local, model_id).await;

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_down.results.len(),
        2,
        "down inference should have 2 results for batch size 2"
    );

    // Step 5: Optimize with fake loss values
    let fake_loss_up = 0.5f32;
    let fake_loss_down = 1.0f32;
    println!(
        "\n--- Step 5: Optimize (loss_up={}, loss_down={}) ---",
        fake_loss_up, fake_loss_down
    );
    quark_optimize(
        &quark_client,
        quark_local,
        model_id,
        fake_loss_up,
        fake_loss_down,
    )
    .await;

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_final.results.len(),
        2,
        "final inference should have 2 results for batch size 2"
    );

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

    quark_shutdown(&quark_client, quark_local, model_id).await;
    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}

#[tokio::test]
async fn dark_optimization() {
    init_tracing();

    let model_path = match require_model_path("dark_optimization") {
        Some(path) => path,
        None => return,
    };

    let (void_local, void_abort, quark_local, quark_abort) = start_servers(&model_path).await;

    let void_client = make_client_endpoint().await;
    let quark_client = make_client_endpoint().await;
    let model_id = Uuid::new_v4();
    quark_start(&quark_client, quark_local, model_id).await;

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let input_text_2 =
        "A starship traveling at constant velocity measures a distance of 1,200 light-years to a distant galaxy. After covering half the distance, it detects an anomaly and must divert, adding 300 light-years to its route. How many total light-years will the journey be?";
    println!("Input text: {input_text}");

    let tokenizer = get_tokenizer();

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
    quark_perturb_up(&quark_client, quark_local, model_id, 42).await;

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_up.results.len(),
        2,
        "up inference should have 2 results for batch size 2"
    );

    // Step 3: PerturbDown
    println!("\n--- Step 3: PerturbDown ---");
    quark_perturb_down(&quark_client, quark_local, model_id).await;

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_down.results.len(),
        2,
        "down inference should have 2 results for batch size 2"
    );

    // Step 5: Optimize with fake loss values
    let fake_loss_up = 0.5f32;
    let fake_loss_down = 1.0f32;
    println!(
        "\n--- Step 5: Optimize (loss_up={}, loss_down={}) ---",
        fake_loss_up, fake_loss_down
    );
    quark_optimize(
        &quark_client,
        quark_local,
        model_id,
        fake_loss_up,
        fake_loss_down,
    )
    .await;

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_infer(&quark_client, quark_local, model_id, input_id).await;
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
    assert_eq!(
        output_final.results.len(),
        2,
        "final inference should have 2 results for batch size 2"
    );

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

    quark_shutdown(&quark_client, quark_local, model_id).await;
    drop(void_client);
    drop(quark_client);
    void_abort.abort();
    quark_abort.abort();
}
