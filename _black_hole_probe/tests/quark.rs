mod common;

use black_hole_sun::{
    DarkToken, InferenceInput, InferenceRequest, LogitEntry, QuarkClient, TestQuarkServer,
    TestVoidServer, Tokenizer, VoidClient,
};
use postcard::{from_bytes, to_allocvec};
use uuid::Uuid;

use common::*;

#[tokio::test]
async fn rejects_requests_for_unknown_model_instance() {
    init_tracing();

    let quark_server = TestQuarkServer::new("model-is-not-loaded-for-this-test")
        .serve()
        .await
        .expect("failed to start quark server");
    let quark_local = quark_server.local_addr();

    let client = make_client_endpoint().await;
    let quark_client = QuarkClient::new(&client, quark_local, "localhost");
    let model_id = Uuid::new_v4();

    for _ in 0..2 {
        let error = quark_client
            .start(model_id)
            .await
            .expect_err("invalid model path should fail to start");
        assert!(
            error.contains("Model path does not exist"),
            "unexpected start error: {error}"
        );
    }

    for result in [
        quark_client.infer(model_id, Uuid::nil()).await.map(|_| ()),
        quark_client.perturb_up(model_id, 42).await,
        quark_client.perturb_down(model_id).await,
        quark_client.optimize(model_id, 0.0, 0.0).await,
        quark_client.shutdown(model_id).await,
    ] {
        let error = result.expect_err("unknown model request should fail");
        assert!(
            error.contains("is not running"),
            "unexpected quark error: {error}"
        );
    }

    quark_server.abort();
}

#[tokio::test]
async fn inference() {
    init_tracing();

    let model_path = match require_model_path("inference") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");
    let void_local = void_server.local_addr();
    let quark_local = quark_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let quark_endpoint = make_client_endpoint().await;
    let quark_client = QuarkClient::new(&quark_endpoint, quark_local, "localhost");
    let model_id = Uuid::new_v4();
    quark_client.start(model_id).await.unwrap();

    let tokenizer = Tokenizer::init();

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    println!("Input text: {input_text}");

    let request = InferenceRequest::Sequences {
        sequences: vec![
            vec![InferenceInput::Text(input_text.into())],
            vec![InferenceInput::Text(input_text.into())],
        ],
        limit: Some(100),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    let output_id = quark_client.infer(model_id, input_id).await.unwrap();

    let output_bytes = void_client.download(output_id).await.unwrap();
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 2, "expected 2 batch results");

    for (i, seq_result) in output.results.iter().enumerate() {
        let label = i + 1;
        assert!(
            !seq_result.0.is_empty(),
            "output {label} has zero predictions"
        );
        let output_text = tokenizer.decode(&seq_result.0);

        println!("Output {label}: {output_text}");
        assert!(
            !output_text.is_empty(),
            "output {label} has no decoded text"
        );
    }

    quark_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(quark_endpoint);
    void_server.abort();
    quark_server.abort();
}

#[tokio::test]
async fn dark_inference() {
    init_tracing();

    let model_path = match require_model_path("dark_inference") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");
    let void_local = void_server.local_addr();
    let quark_local = quark_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let quark_endpoint = make_client_endpoint().await;
    let quark_client = QuarkClient::new(&quark_endpoint, quark_local, "localhost");
    let model_id = Uuid::new_v4();
    quark_client.start(model_id).await.unwrap();

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    println!("Input text: {input_text}");

    let tokenizer = Tokenizer::init();

    let tokens = tokenizer
        .encode_ids(input_text)
        .expect("failed to tokenize input");

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
        limit: Some(100),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    let output_id = quark_client.infer(model_id, input_id).await.unwrap();

    let output_bytes = void_client.download(output_id).await.unwrap();
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 2, "expected 2 batch results");

    for (i, seq_result) in output.results.iter().enumerate() {
        let label = i + 1;
        assert!(
            !seq_result.0.is_empty(),
            "output {label} has zero predictions"
        );
        let output_text = tokenizer.decode(&seq_result.0);

        println!("Output {label}: {output_text}");
        assert!(
            !output_text.is_empty(),
            "output {label} has no decoded text"
        );
    }

    quark_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(quark_endpoint);
    void_server.abort();
    quark_server.abort();
}

#[tokio::test]
async fn optimization() {
    init_tracing();

    let model_path = match require_model_path("optimization") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");
    let void_local = void_server.local_addr();
    let quark_local = quark_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let quark_endpoint = make_client_endpoint().await;
    let quark_client = QuarkClient::new(&quark_endpoint, quark_local, "localhost");
    let model_id = Uuid::new_v4();
    quark_client.start(model_id).await.unwrap();

    let tokenizer = Tokenizer::init();

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
        limit: Some(100),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    // ─── QuZO flow: PerturbUp -> Infer -> PerturbDown -> Infer -> Optimize -> Infer ───

    // Step 1: PerturbUp
    println!("\n--- Step 1: PerturbUp (seed=42) ---");
    quark_client.perturb_up(model_id, 42).await.unwrap();

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_up = void_client.download(output_id_up).await.unwrap();
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
    quark_client.perturb_down(model_id).await.unwrap();

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_down = void_client.download(output_id_down).await.unwrap();
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
    quark_client
        .optimize(model_id, fake_loss_up, fake_loss_down)
        .await
        .unwrap();

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_final = void_client.download(output_id_final).await.unwrap();
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
    let final_text = tokenizer.decode(&output_final.results[0].0);
    let final_text_2 = tokenizer.decode(&output_final.results[1].0);

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

    quark_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(quark_endpoint);
    void_server.abort();
    quark_server.abort();
}

#[tokio::test]
async fn dark_optimization() {
    init_tracing();

    let model_path = match require_model_path("dark_optimization") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");
    let void_local = void_server.local_addr();
    let quark_local = quark_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let quark_endpoint = make_client_endpoint().await;
    let quark_client = QuarkClient::new(&quark_endpoint, quark_local, "localhost");
    let model_id = Uuid::new_v4();
    quark_client.start(model_id).await.unwrap();

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let input_text_2 =
        "A starship traveling at constant velocity measures a distance of 1,200 light-years to a distant galaxy. After covering half the distance, it detects an anomaly and must divert, adding 300 light-years to its route. How many total light-years will the journey be?";
    println!("Input text: {input_text}");

    let tokenizer = Tokenizer::init();

    let fn_to_dark_tokens = |text: &str, tokenizer: &Tokenizer| -> Vec<DarkToken> {
        let tokens = tokenizer
            .encode_ids(text)
            .expect("failed to tokenize input");
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
        limit: Some(100),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    // ─── QuZO flow: PerturbUp -> Infer -> PerturbDown -> Infer -> Optimize -> Infer ───

    // Step 1: PerturbUp
    println!("\n--- Step 1: PerturbUp (seed=42) ---");
    quark_client.perturb_up(model_id, 42).await.unwrap();

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_up = void_client.download(output_id_up).await.unwrap();
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
    quark_client.perturb_down(model_id).await.unwrap();

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_down = void_client.download(output_id_down).await.unwrap();
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
    quark_client
        .optimize(model_id, fake_loss_up, fake_loss_down)
        .await
        .unwrap();

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = quark_client.infer(model_id, input_id).await.unwrap();
    let output_bytes_final = void_client.download(output_id_final).await.unwrap();
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
    let final_text = tokenizer.decode(&output_final.results[0].0);
    let final_text_2 = tokenizer.decode(&output_final.results[1].0);

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

    quark_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(quark_endpoint);
    void_server.abort();
    quark_server.abort();
}
