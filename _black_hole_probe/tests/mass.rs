#![allow(dead_code, unused_imports, clippy::manual_async_fn)]

mod common;

use black_hole_sun::{
    DarkToken, InferenceInput, InferenceRequest, LogitEntry, MassClient, MassModelCapacity,
    MassModelConfig, TestMassServer, TestVoidServer, Tokenizer, VoidClient,
};
use postcard::{from_bytes, to_allocvec};
use std::time::Duration;
use uuid::Uuid;

use common::*;

#[tokio::test]
async fn rejects_requests_for_unknown_model_instance() {
    init_tracing();

    let mass_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .serve()
        .await
        .expect("failed to start mass server");
    let mass_local = mass_server.local_addr();

    let client = make_client_endpoint().await;
    let mass_client = MassClient::new(&client, mass_local, "localhost");
    let model_id = Uuid::new_v4();

    for _ in 0..2 {
        let error = mass_client
            .start(model_id, None)
            .await
            .expect_err("invalid model path should fail to start");
        assert!(
            error.contains("Model path does not exist"),
            "unexpected start error: {error}"
        );
    }

    for result in [
        mass_client.infer(model_id, Uuid::nil()).await.map(|_| ()),
        mass_client.reset(model_id).await,
        mass_client.perturb_up(model_id, 42).await,
        mass_client.perturb_down(model_id).await,
        mass_client.checkpoint(model_id).await.map(|_| ()),
        mass_client.optimize(model_id, 0.0, 0.0).await,
        mass_client.shutdown(model_id).await,
        mass_client.query_model_params(model_id).await.map(|_| ()),
    ] {
        let error = result.expect_err("unknown model request should fail");
        assert!(
            error.contains("is not running"),
            "unexpected mass error: {error}"
        );
    }

    mass_server.abort();
}

#[tokio::test]
async fn tunnel_worker_rejects_direct_model_requests() {
    init_tracing();

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .serve()
        .await
        .expect("failed to start root mass server");
    let worker_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tunnel(root_server.local_addr())
        .max_instances(1)
        .serve()
        .await
        .expect("failed to start worker mass server");

    let client = make_client_endpoint().await;
    let worker_client = MassClient::new(&client, worker_server.local_addr(), "localhost");
    let model_id = Uuid::new_v4();
    let error = worker_client
        .start(model_id, None)
        .await
        .expect_err("direct requests to tunnel worker should fail");
    assert!(
        error.contains("tunnel worker rejects direct model requests"),
        "unexpected worker error: {error}"
    );

    worker_server.abort();
    root_server.abort();
}

#[tokio::test]
async fn tunnel_root_forwards_start_to_registered_worker() {
    init_tracing();

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .max_instances(0)
        .serve()
        .await
        .expect("failed to start root mass server");
    let worker_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tunnel(root_server.local_addr())
        .max_instances(1)
        .serve()
        .await
        .expect("failed to start worker mass server");

    let client = make_client_endpoint().await;
    let root_client = MassClient::new(&client, root_server.local_addr(), "localhost");
    let model_id = Uuid::new_v4();
    let error = root_client
        .start(model_id, None)
        .await
        .expect_err("start should be forwarded to worker and fail on invalid model path");
    assert!(
        error.contains("Model path does not exist"),
        "unexpected forwarded start error: {error}"
    );

    worker_server.abort();
    root_server.abort();
}

#[tokio::test]
async fn tcp_tunnel_root_forwards_start_to_registered_worker() {
    init_tracing();

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tcp()
        .max_instances(0)
        .serve()
        .await
        .expect("failed to start root mass server");
    let worker_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tcp()
        .tunnel(root_server.local_addr())
        .max_instances(1)
        .serve()
        .await
        .expect("failed to start worker mass server");

    let root_client = MassClient::new_tcp(root_server.local_addr());
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(capacity) = root_client.query_model_capacity().await {
                if capacity.total == Some(1) && capacity.available == Some(1) {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("worker capacity should propagate over tcp tunnel");

    let model_id = Uuid::new_v4();
    let error = root_client
        .start(model_id, None)
        .await
        .expect_err("start should be forwarded to worker and fail on invalid model path");
    assert!(
        error.contains("Model path does not exist"),
        "unexpected forwarded start error: {error}"
    );

    worker_server.abort();
    root_server.abort();
}

#[tokio::test]
async fn tcp_tunnel_root_forwards_model_load_and_inference_to_registered_worker() {
    init_tracing();

    let model_path = match require_model_path(
        "tcp_tunnel_root_forwards_model_load_and_inference_to_registered_worker",
    ) {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .tcp()
        .serve()
        .await
        .expect("failed to start tcp void server");
    let root_server = TestMassServer::new(&model_path)
        .tcp()
        .void_addr(void_server.local_addr())
        .max_instances(0)
        .serve()
        .await
        .expect("failed to start tcp root mass server");
    let worker_server = TestMassServer::new(&model_path)
        .tcp()
        .void_addr(void_server.local_addr())
        .tunnel(root_server.local_addr())
        .max_instances(1)
        .serve()
        .await
        .expect("failed to start tcp worker mass server");

    let void_client = VoidClient::new_tcp(void_server.local_addr());
    let root_client = MassClient::new_tcp(root_server.local_addr());
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(capacity) = root_client.query_model_capacity().await {
                if capacity.total == Some(1) && capacity.available == Some(1) {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("worker capacity should propagate over tcp tunnel before model load");

    let model_id = Uuid::new_v4();
    root_client
        .start(
            model_id,
            Some(MassModelConfig {
                inference_limit: Some(1),
                ..MassModelConfig::default()
            }),
        )
        .await
        .expect("root should forward model initialization to tcp tunnel worker");

    let request = InferenceRequest::Sequences {
        sequences: vec![vec![InferenceInput::Text(
            "Return one token to prove tcp tunnel inference routing.".into(),
        )]],
        limit: Some(1),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    let output_id = root_client
        .infer(model_id, input_id)
        .await
        .expect("root should forward inference to tcp tunnel worker");
    let output_bytes = void_client
        .download(output_id)
        .await
        .expect("tcp void should return tunneled inference output");
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 1, "expected one batch result");
    assert!(
        output.results[0].0.len() <= 1,
        "inference limit should cap output to at most one token"
    );

    root_client
        .shutdown(model_id)
        .await
        .expect("shutdown should succeed through tcp tunnel root");

    worker_server.abort();
    root_server.abort();
    void_server.abort();
}

#[tokio::test]
async fn tcp_void_upload_download_round_trip() {
    init_tracing();

    let void_server = TestVoidServer::new()
        .tcp()
        .serve()
        .await
        .expect("failed to start tcp void server");
    let void_client = VoidClient::new_tcp(void_server.local_addr());

    let payload = b"tcp transport keeps length-prefixed postcard framing".to_vec();
    let object_id = void_client
        .upload(payload.clone())
        .await
        .expect("tcp upload should succeed");
    let downloaded = void_client
        .download(object_id)
        .await
        .expect("tcp download should succeed");
    assert_eq!(downloaded, payload);

    void_server.abort();
}

#[tokio::test]
async fn tunnel_worker_retries_parent_registration_until_root_starts() {
    init_tracing();

    let reserved = std::net::UdpSocket::bind("127.0.0.1:0").expect("failed to reserve root port");
    let root_addr = reserved
        .local_addr()
        .expect("failed to read reserved root port");
    drop(reserved);

    let worker_task = tokio::spawn(async move {
        TestMassServer::new("model-is-not-loaded-for-this-test")
            .tunnel(root_addr)
            .max_instances(1)
            .tunnel_connect_deadline(Duration::from_secs(3))
            .serve()
            .await
    });

    tokio::time::sleep(Duration::from_millis(300)).await;

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .listen(root_addr)
        .max_instances(0)
        .serve()
        .await
        .expect("failed to start root mass server");

    let worker_server = tokio::time::timeout(Duration::from_secs(4), worker_task)
        .await
        .expect("worker should register before timeout")
        .expect("worker task should not panic")
        .expect("worker should keep retrying until root is available");

    let client = make_client_endpoint().await;
    let root_client = MassClient::new(&client, root_server.local_addr(), "localhost");
    let capacity = root_client
        .query_model_capacity()
        .await
        .expect("capacity query should succeed");

    // Feature-gated builds compile in an architecture, so even a server that
    // has never loaded a model advertises per-architecture capacity. The root
    // allows 0 instances here; the worker's limit of 1 dominates.
    let mut expected_per_architecture = Vec::new();
    if let Some(architecture) = black_hole_sun::black_hole_mass::COMPILED_ARCHITECTURE {
        expected_per_architecture.push((
            architecture,
            MassModelCapacity {
                total: Some(1),
                available: Some(1),
                occupied: 0,
                per_architecture: Vec::new(),
            },
        ));
    }

    assert_eq!(
        capacity,
        MassModelCapacity {
            total: Some(1),
            available: Some(1),
            occupied: 0,
            per_architecture: expected_per_architecture,
        }
    );

    worker_server.abort();
    root_server.abort();
}

#[tokio::test]
async fn tunnel_worker_re_registers_after_root_restart() {
    init_tracing();

    let reserved = std::net::UdpSocket::bind("127.0.0.1:0").expect("failed to reserve root port");
    let root_addr = reserved
        .local_addr()
        .expect("failed to read reserved root port");
    drop(reserved);

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .listen(root_addr)
        .max_instances(0)
        .serve()
        .await
        .expect("failed to start root mass server");
    let worker_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tunnel(root_addr)
        .max_instances(1)
        .serve()
        .await
        .expect("failed to start worker mass server");

    let client = make_client_endpoint().await;
    let root_client = MassClient::new(&client, root_addr, "localhost");

    let initial_capacity = root_client
        .query_model_capacity()
        .await
        .expect("initial capacity query should succeed");
    assert_eq!(initial_capacity.total, Some(1));
    assert_eq!(initial_capacity.available, Some(1));
    assert_eq!(initial_capacity.occupied, 0);

    root_server.abort();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let restarted_root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .listen(root_addr)
        .max_instances(0)
        .serve()
        .await
        .expect("failed to restart root mass server");

    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            if let Ok(capacity) = root_client.query_model_capacity().await {
                if capacity.total == Some(1)
                    && capacity.available == Some(1)
                    && capacity.occupied == 0
                {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .expect("worker should re-register with restarted root");

    worker_server.abort();
    restarted_root_server.abort();
}

#[tokio::test]
async fn recursive_capacity_query_reports_total_available_and_occupied() {
    init_tracing();

    let root_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .max_instances(2)
        .serve()
        .await
        .expect("failed to start root mass server");
    let worker_server = TestMassServer::new("model-is-not-loaded-for-this-test")
        .tunnel(root_server.local_addr())
        .max_instances(3)
        .serve()
        .await
        .expect("failed to start worker mass server");

    let client = make_client_endpoint().await;
    let root_client = MassClient::new(&client, root_server.local_addr(), "localhost");
    let capacity = root_client
        .query_model_capacity()
        .await
        .expect("capacity query should succeed");
    assert_eq!(capacity.total, Some(5));
    assert_eq!(capacity.available, Some(5));
    assert_eq!(capacity.occupied, 0);

    worker_server.abort();
    root_server.abort();
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
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");
    let model_id = Uuid::new_v4();
    mass_client.start(model_id, None).await.unwrap();

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

    let output_id = mass_client.infer(model_id, input_id).await.unwrap();

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

    let checkpoint_id = mass_client.checkpoint(model_id).await.unwrap();
    let checkpoint_bytes = void_client.download(checkpoint_id).await.unwrap();
    assert!(
        !checkpoint_bytes.is_empty(),
        "checkpoint output should upload non-empty model bytes"
    );

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
}

#[ignore]
#[tokio::test]
async fn qwen3_8_27b_q3_k_s_initialize_and_single_token_inference() {
    init_tracing();

    let Some(model_path) = std::env::var("BLACK_HOLE_PROBE_27B_MODEL_PATH").ok() else {
        tracing::warn!(
            test = "qwen3_8_27b_q3_k_s_initialize_and_single_token_inference",
            "Skipping test: BLACK_HOLE_PROBE_27B_MODEL_PATH not set"
        );
        return;
    };
    if !std::path::Path::new(&model_path).exists() {
        tracing::warn!(
            test = "qwen3_8_27b_q3_k_s_initialize_and_single_token_inference",
            "Skipping test: model file does not exist"
        );
        return;
    }

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let mass_server = TestMassServer::new(model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");

    let model_id = Uuid::new_v4();
    mass_client
        .start(
            model_id,
            Some(MassModelConfig {
                inference_limit: Some(1),
                ..MassModelConfig::default()
            }),
        )
        .await
        .expect("qwen3.8-27b-q3_k_s model should initialize");

    let tokenizer = Tokenizer::init();

    let request = InferenceRequest::Sequences {
        sequences: vec![vec![InferenceInput::Text(
            "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?".into(),
        )]; 2],
        limit: Some(100),
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    mass_client
        .perturb_up(model_id, 42)
        .await
        .expect("perturb-up should succeed before inference");

    let output_id = mass_client
        .infer(model_id, input_id)
        .await
        .expect("inference with perturbed-up weights should succeed");
    let output_bytes = void_client.download(output_id).await.unwrap();
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(
        output.results.len(),
        2,
        "expected one result for batch size 2"
    );
    assert!(
        !output.results[0].0.is_empty(),
        "inference should produce at least one predicted token"
    );
    let output_text = tokenizer.decode(&output.results[0].0);
    println!("Output: {output_text}");

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
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
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");
    let model_id = Uuid::new_v4();
    mass_client.start(model_id, None).await.unwrap();

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

    let output_id = mass_client.infer(model_id, input_id).await.unwrap();

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

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
}

#[tokio::test]
async fn start_model_applies_instance_default_inference_limit_override() {
    init_tracing();

    let model_path =
        match require_model_path("start_model_applies_instance_default_inference_limit_override") {
            Some(path) => path,
            None => return,
        };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");

    let model_id = Uuid::new_v4();
    mass_client
        .start(
            model_id,
            Some(MassModelConfig {
                inference_limit: Some(0),
                ..MassModelConfig::default()
            }),
        )
        .await
        .expect("model should start with override config");

    let request = InferenceRequest::Sequences {
        sequences: vec![vec![InferenceInput::Text(
            "This request omits limit and should use the per-instance default.".into(),
        )]],
        limit: None,
    };
    let request_bytes = to_allocvec(&request).expect("failed to serialize inference request");
    let input_id = void_client.upload(request_bytes).await.unwrap();

    let output_id = mass_client.infer(model_id, input_id).await.unwrap();
    let output_bytes = void_client.download(output_id).await.unwrap();
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");

    assert_eq!(output.results.len(), 1, "expected one batch result");
    assert!(
        output.results[0].0.is_empty(),
        "instance default inference limit override should produce zero decoded tokens"
    );

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
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
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");
    let model_id = Uuid::new_v4();
    mass_client.start(model_id, None).await.unwrap();

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
    mass_client.perturb_up(model_id, 42).await.unwrap();

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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
    mass_client.perturb_down(model_id).await.unwrap();

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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
    mass_client
        .optimize(model_id, fake_loss_up, fake_loss_down)
        .await
        .unwrap();

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
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
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");
    let void_local = void_server.local_addr();
    let mass_local = mass_server.local_addr();

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_local, "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_local, "localhost");
    let model_id = Uuid::new_v4();
    mass_client.start(model_id, None).await.unwrap();

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
    mass_client.perturb_up(model_id, 42).await.unwrap();

    // Step 2: Inference with perturbed-up weights
    println!("--- Step 2: Infer (up) ---");
    let output_id_up = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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
    mass_client.perturb_down(model_id).await.unwrap();

    // Step 4: Inference with perturbed-down weights
    println!("--- Step 4: Infer (down) ---");
    let output_id_down = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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
    mass_client
        .optimize(model_id, fake_loss_up, fake_loss_down)
        .await
        .unwrap();

    // Step 6: Final inference after optimization (back to Idle state)
    println!("--- Step 6: Infer (post-optimize) ---");
    let output_id_final = mass_client.infer(model_id, input_id).await.unwrap();
    mass_client.reset(model_id).await.unwrap();
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

    mass_client.shutdown(model_id).await.unwrap();
    drop(void_endpoint);
    drop(mass_endpoint);
    void_server.abort();
    mass_server.abort();
}

// ---------------------------------------------------------------------------
// Void chunked transfer (multipart upload + ranged download)
// ---------------------------------------------------------------------------

/// Size that forces the chunked path: above mass's 64 MB single-frame cap.
const MULTIPART_TEST_SIZE: usize = 70 * 1024 * 1024; // 70 MB

fn patterned_bytes(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 251) as u8).collect()
}

async fn roundtrip_file_through_void(
    void_client: &VoidClient,
    source_path: &std::path::Path,
) -> std::io::Result<Vec<u8>> {
    let id = void_client.upload_file(source_path).await.unwrap();
    let downloaded_path = std::env::temp_dir().join(format!("bhs-void-roundtrip-{id}.bin"));
    let written = void_client
        .download_to_file(id, &downloaded_path)
        .await
        .unwrap();
    let bytes = std::fs::read(&downloaded_path).unwrap();
    let _ = std::fs::remove_file(&downloaded_path);
    assert_eq!(written as usize, bytes.len(), "download byte count mismatch");
    Ok(bytes)
}

#[tokio::test]
async fn void_multipart_roundtrip_in_memory_store() {
    init_tracing();

    let void_server = TestVoidServer::new()
        .tcp()
        .serve()
        .await
        .expect("failed to start void server");

    let void_client = VoidClient::new_tcp(void_server.local_addr());

    let source_path = std::env::temp_dir().join(format!(
        "bhs-void-multipart-src-{}.bin",
        Uuid::new_v4()
    ));
    std::fs::write(&source_path, patterned_bytes(MULTIPART_TEST_SIZE)).unwrap();

    let downloaded = roundtrip_file_through_void(&void_client, &source_path)
        .await
        .expect("multipart roundtrip should succeed");
    let _ = std::fs::remove_file(&source_path);

    assert_eq!(
        downloaded.len(),
        MULTIPART_TEST_SIZE,
        "roundtripped size mismatch"
    );
    assert_eq!(
        downloaded,
        patterned_bytes(MULTIPART_TEST_SIZE),
        "multipart roundtrip corrupted data"
    );

    void_server.abort();
}

#[tokio::test]
async fn void_multipart_roundtrip_filesystem_store() {
    init_tracing();

    let store_root = std::env::temp_dir().join(format!("bhs-void-fs-store-{}", Uuid::new_v4()));
    let store = black_hole_sun::object_store::FilesystemObjectStore::new(&store_root)
        .expect("failed to create filesystem object store");
    let void_server = TestVoidServer::new()
        .tcp()
        .object_store(Box::new(store))
        .serve()
        .await
        .expect("failed to start void server");

    let void_client = VoidClient::new_tcp(void_server.local_addr());

    let source_path = std::env::temp_dir().join(format!(
        "bhs-void-multipart-src-{}.bin",
        Uuid::new_v4()
    ));
    std::fs::write(&source_path, patterned_bytes(MULTIPART_TEST_SIZE)).unwrap();

    let downloaded = roundtrip_file_through_void(&void_client, &source_path)
        .await
        .expect("multipart roundtrip should succeed");
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_dir_all(&store_root);

    assert_eq!(
        downloaded.len(),
        MULTIPART_TEST_SIZE,
        "roundtripped size mismatch"
    );
    assert_eq!(
        downloaded,
        patterned_bytes(MULTIPART_TEST_SIZE),
        "multipart roundtrip corrupted data"
    );

    void_server.abort();
}

#[tokio::test]
async fn mass_void_client_multipart_roundtrip() {
    init_tracing();

    let void_server = TestVoidServer::new()
        .tcp()
        .serve()
        .await
        .expect("failed to start void server");

    // The mass-side client is the one used by FuseWeights/Checkpoint flows.
    let void_client = black_hole_sun::black_hole_mass::VoidClient::connect(
        void_server.local_addr(),
        black_hole_sun::black_hole_mass::TransportMode::Tcp,
    )
    .await
    .expect("failed to connect mass void client");

    let source_path = std::env::temp_dir().join(format!(
        "bhs-mass-void-src-{}.bin",
        Uuid::new_v4()
    ));
    std::fs::write(&source_path, patterned_bytes(MULTIPART_TEST_SIZE)).unwrap();

    let id = void_client
        .upload_file(&source_path)
        .await
        .expect("mass multipart upload should succeed");
    let downloaded_path = std::env::temp_dir().join(format!("bhs-mass-void-dl-{id}.bin"));
    let written = void_client
        .download_to_file(id, &downloaded_path)
        .await
        .expect("mass ranged download should succeed");
    let bytes = std::fs::read(&downloaded_path).unwrap();
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&downloaded_path);

    assert_eq!(written as usize, MULTIPART_TEST_SIZE);
    assert_eq!(
        bytes,
        patterned_bytes(MULTIPART_TEST_SIZE),
        "mass multipart roundtrip corrupted data"
    );

    void_server.abort();
}

// ---------------------------------------------------------------------------
// FuseWeights
// ---------------------------------------------------------------------------

// Current-thread flavor (like every other model test): paramecia's model
// actor must stay pinned to one OS thread or CUDA loses its device context.
#[tokio::test]
async fn fuse_weights_with_checkpoint() {
    init_tracing();

    let model_path = match require_model_path("fuse_weights_with_checkpoint") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let mass_server = TestMassServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start mass server");

    let void_endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&void_endpoint, void_server.local_addr(), "localhost");
    let mass_endpoint = make_client_endpoint().await;
    let mass_client = MassClient::new(&mass_endpoint, mass_server.local_addr(), "localhost");

    let model_id = Uuid::new_v4();
    mass_client.start(model_id, None).await.unwrap();

    // Checkpoint the fresh weights into void.
    let checkpoint_id = mass_client.checkpoint(model_id).await.unwrap();
    let checkpoint_path = std::env::temp_dir().join(format!("bhs-ckpt-{checkpoint_id}.gguf"));
    let checkpoint_len = void_client
        .download_to_file(checkpoint_id, &checkpoint_path)
        .await
        .unwrap_or_else(|e| panic!("checkpoint should download: {e}"));
    assert!(checkpoint_len >= 4, "checkpoint object is empty");

    // Fuse live weights with the checkpoint at 50/50.
    let fused_id = mass_client
        .fuse_weights(model_id, checkpoint_id, 0.5)
        .await
        .unwrap_or_else(|e| panic!("fuse_weights should succeed: {e}"));

    // The fused object must be a valid, non-trivial GGUF of comparable size.
    let fused_path = std::env::temp_dir().join(format!("bhs-fused-{fused_id}.gguf"));
    let written = void_client
        .download_to_file(fused_id, &fused_path)
        .await
        .unwrap_or_else(|e| panic!("fused object should download: {e}"));
    let fused_bytes = std::fs::read(&fused_path).unwrap();
    let _ = std::fs::remove_file(&fused_path);
    let _ = std::fs::remove_file(&checkpoint_path);

    assert_eq!(written as usize, fused_bytes.len());
    assert!(fused_bytes.len() >= 4, "fused object is too small to be a GGUF");
    assert_eq!(
        &fused_bytes[0..4],
        b"GGUF",
        "fused object is missing the GGUF magic header"
    );
    assert!(
        (fused_bytes.len() as u64) >= checkpoint_len / 2
            && (fused_bytes.len() as u64) <= checkpoint_len * 2,
        "fused size {} not comparable to checkpoint size {checkpoint_len}",
        fused_bytes.len()
    );

    // Fusing an unknown model must fail cleanly.
    let error = mass_client
        .fuse_weights(Uuid::new_v4(), checkpoint_id, 0.5)
        .await
        .expect_err("fusing an unknown model should fail");
    assert!(
        error.contains("not running"),
        "unexpected unknown-model fuse error: {error}"
    );

    // Fusing against a missing void object must fail cleanly.
    let error = mass_client
        .fuse_weights(model_id, Uuid::new_v4(), 0.5)
        .await
        .expect_err("fusing a missing checkpoint should fail");
    assert!(
        !error.is_empty(),
        "missing-checkpoint fuse error should not be empty"
    );

    // The running instance must still work after fusion (fusion is
    // non-destructive: it only produced a new void object).
    let tokenizer = Tokenizer::init();
    let output_id = mass_client
        .infer(
            model_id,
            void_client
                .upload(
                    to_allocvec(&InferenceRequest::Sequences {
                        sequences: vec![vec![InferenceInput::Text("Hello".into())]],
                        limit: Some(8),
                    })
                    .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await
        .unwrap();
    let output_bytes = void_client.download(output_id).await.unwrap();
    let output: black_hole_sun::InferenceOutput =
        from_bytes(&output_bytes).expect("failed to decode inference output");
    assert!(
        !output.results[0].0.is_empty(),
        "post-fusion inference returned zero predictions"
    );
    let _ = tokenizer.decode(&output.results[0].0);

    mass_client.shutdown(model_id).await.unwrap();
    void_server.abort();
    mass_server.abort();
}
