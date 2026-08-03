mod common;

use std::net::{Ipv6Addr, SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::effect::{QuarkInfer, QuarkPerturbDown, QuarkPerturbUp, WaitForPropagation};
use black_hole_flux::ops::VoidInferOps;
use black_hole_flux::{CellState, Progenitor};
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{
    DarkToken, Emission, EmissionId, InferenceOutput, InferenceOutputId, InferenceRequest,
    LogitEntry, ObjectId, QuarkServerBuilder, SequenceOutput, Transmission, VoidServerBuilder,
};
use futures::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::server::ServerBuilder;
use postcard::to_allocvec;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use uuid::Uuid;

use common::*;

// ─── Ecosystem ───────────────────────────────────────────────────────────────

#[derive(Animals)]
pub struct SpaceAnimals(Progenitor);

/// A Jungle implementation backed by void + quark servers over QUIC.
pub struct SpaceJungle {
    void_addr: SocketAddr,
    quark_addr: SocketAddr,
}

impl SpaceJungle {
    pub fn new(void_addr: SocketAddr, quark_addr: SocketAddr) -> Self {
        Self {
            void_addr,
            quark_addr,
        }
    }
}

impl Ecosystem for SpaceJungle {
    const NAME: &'static str = "space-jungle";
    type Animals = SpaceAnimals;
}

#[async_trait]
impl VoidInferOps for SpaceJungle {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        let endpoint = make_client_endpoint().await;
        Ok(void_download(&endpoint, self.void_addr, id).await)
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        let endpoint = make_client_endpoint().await;
        Ok(void_upload(&endpoint, self.void_addr, data).await)
    }

    async fn infer(&self, request: InferenceRequest) -> Result<ObjectId, String> {
        let request_bytes = to_allocvec(&request).map_err(|e| format!("serialize: {e}"))?;
        let endpoint = make_client_endpoint().await;
        let request_id = void_upload(&endpoint, self.void_addr, request_bytes).await;
        Ok(quark_infer(&endpoint, self.quark_addr, request_id).await)
    }

    async fn perturb_up(&self, seed: u64) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_perturb_up(&endpoint, self.quark_addr, seed).await;
        Ok(())
    }

    async fn perturb_down(&self) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_perturb_down(&endpoint, self.quark_addr).await;
        Ok(())
    }

    async fn optimize(&self, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_optimize(&endpoint, self.quark_addr, loss_up, loss_down).await;
        Ok(())
    }

    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String> {
        let propagation = Transmission::Propagation {
            emission_id,
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        };
        let data = to_allocvec(&propagation).map_err(|e| format!("serialize: {e}"))?;
        let endpoint = make_client_endpoint().await;
        void_upload_with(&endpoint, self.void_addr, send_id, data).await;
        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn upload_transmission(addr: SocketAddr, transmission: &Transmission) -> ObjectId {
    let endpoint = make_client_endpoint().await;
    let data = to_allocvec(transmission).expect("failed to serialize transmission");
    void_upload(&endpoint, addr, data).await
}

/// Poll void until data appears at `id`, deserializing as `Transmission`.
async fn wait_for_void_transmission(addr: SocketAddr, id: ObjectId) -> Transmission {
    use tokio::time::{sleep, Duration};
    loop {
        let endpoint = make_client_endpoint().await;
        match void_download_result(&endpoint, addr, id).await {
            Ok(data) => {
                return postcard::from_bytes(&data)
                    .expect("failed to deserialize Transmission from void");
            }
            Err(e) => {
                tracing::debug!(%id, error = %e, "download failed, retrying in 1s");
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

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

/// Tokenize text into DarkTokens suitable for InferenceOutput.
fn text_to_dark_tokens(text: &str, tokenizer: &tokenizers::Tokenizer) -> Vec<DarkToken> {
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
}

async fn start_servers(
    model_path: &str,
) -> (
    SocketAddr,
    tokio::task::AbortHandle,
    SocketAddr,
    tokio::task::AbortHandle,
) {
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");
    let void_abort = void_handle.abort_handle();

    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");
    let quark_abort = quark_handle.abort_handle();

    drop(void_handle);
    drop(quark_handle);

    tokio::time::sleep(Duration::from_millis(200)).await;
    (void_local, void_abort, quark_local, quark_abort)
}

fn reserve_local_addr() -> SocketAddr {
    let socket = UdpSocket::bind((Ipv6Addr::LOCALHOST, 0))
        .expect("should bind temporary udp socket for test port reservation");
    socket
        .local_addr()
        .expect("temporary udp socket should expose local address")
}

async fn connect_client_with_retry(remote: SocketAddr) -> jungle_sdk::Client {
    for attempt in 0..40 {
        match jungle_sdk::client::Client::builder()
            .remote(remote)
            .server_name("localhost")
            .build()
            .await
        {
            Ok(client) => return client,
            Err(err) if attempt < 39 => {
                std::thread::sleep(Duration::from_millis(25));
                let _ = err;
            }
            Err(err) => panic!("failed to connect to test server: {err}"),
        }
    }
    unreachable!("retry loop always returns or panics")
}

// ─── Integration test ────────────────────────────────────────────────────────

/// End-to-end test of black-hole-flux constructs through a Jungle Worker.
///
/// This test exercises the void + quark higher-order flows from black-hole-flux:
/// perturbation (PerturbUp/Down), transmission waiting (WaitForPropagation),
/// and quark inference (QuarkInfer).
///
/// Flow:
/// 1. Start void + quark servers on random ports.
/// 2. Create a SpaceJungle with VoidInferOps backed by those servers.
/// 3. Spawn a jungle-server (in-memory backend) and connect a client.
/// 4. Upload a Propagation transmission to void, then spawn a Progenitor.
/// 5. Subscribe to step updates for the journey.
/// 6. Start a JungleWorker so effects execute after we're subscribed.
/// 7. Assert that effects execute successfully within a 30s timeout.
#[tokio::test]
async fn progenitor_flux_flow() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let listen_1 = Uuid::new_v4();
    let listen_2 = Uuid::new_v4();
    let listen_3 = Uuid::new_v4();
    let listen_4 = Uuid::new_v4();

    let model_path = match require_model_path("progenitor_flux_flow") {
        Some(path) => path,
        None => return,
    };

    // 1. Start void and quark servers on random ports.
    let (void_addr, void_abort, quark_addr, quark_abort) = start_servers(&model_path).await;

    // 2. Build the SpaceJungle with void/quark capabilities.
    let jungle = SpaceJungle::new(void_addr, quark_addr);

    // 3. Spawn a jungle-server with in-memory backend and connect a client.
    let listen_addr = reserve_local_addr();
    let server_handle = tokio::spawn(async move {
        ServerBuilder::new()
            .listen(listen_addr)
            .memory()
            .run()
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = connect_client_with_retry(listen_addr).await;

    // 4. Prepare emission data and Propagation transmission chain in void.
    //    The Progenitor (Primordium) cell optimization loop:
    //    PerturbUp -> WaitForPropagation -> Nucleus -> Transmit ->
    //    PerturbDown -> WaitForPropagation -> Nucleus -> Transmit ->
    //    WaitForPotentiation -> Optimize -> (loop back to WaitForPropagation ...)
    //
    //    Chain: Initiation -> Propagation(1) -> Propagation(2) -> Potentiation -> Propagation(3) -> Propagation(4)

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let tokenizer = get_tokenizer();

    // ── Fourth propagation (end of chain) ──
    let dark_tokens_4 = text_to_dark_tokens(input_text, &tokenizer);
    let inference_output_4 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_4)],
    };
    let inference_output_bytes_4 =
        to_allocvec(&inference_output_4).expect("serialize inference output 4");
    let inference_output_id_4 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        inference_output_bytes_4,
    )
    .await;
    let emission_4 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_4),
    };
    let emission_bytes_4 = to_allocvec(&emission_4).expect("serialize emission 4");
    let emission_void_id_4 =
        void_upload(&make_client_endpoint().await, void_addr, emission_bytes_4).await;
    let propagation_4 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_4),
        recv: ObjectId::nil(),
        send: listen_4,
    };
    let propagation_bytes_4 = to_allocvec(&propagation_4).expect("serialize propagation 4");
    let propagation_void_id_4 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        propagation_bytes_4,
    )
    .await;

    // ── Third propagation (points to fourth) ──
    let dark_tokens_3 = text_to_dark_tokens(input_text, &tokenizer);
    let inference_output_3 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_3)],
    };
    let inference_output_bytes_3 =
        to_allocvec(&inference_output_3).expect("serialize inference output 3");
    let inference_output_id_3 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        inference_output_bytes_3,
    )
    .await;
    let emission_3 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_3),
    };
    let emission_bytes_3 = to_allocvec(&emission_3).expect("serialize emission 3");
    let emission_void_id_3 =
        void_upload(&make_client_endpoint().await, void_addr, emission_bytes_3).await;
    let propagation_3 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_3),
        recv: propagation_void_id_4,
        send: listen_3,
    };
    let propagation_bytes_3 = to_allocvec(&propagation_3).expect("serialize propagation 3");
    let propagation_void_id_3 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        propagation_bytes_3,
    )
    .await;

    // ── Potentiation (links second propagation to third) ──
    let potentiation = Transmission::Potentiation {
        loss_up: 0.5,
        loss_down: 0.3,
        recv: propagation_void_id_3,
    };
    let potentiation_bytes = to_allocvec(&potentiation).expect("serialize potentiation");
    let potentiation_void_id = void_upload(
        &make_client_endpoint().await,
        void_addr,
        potentiation_bytes,
    )
    .await;

    // ── Second propagation (points to potentiation) ──
    let dark_tokens_2 = text_to_dark_tokens(input_text, &tokenizer);
    let inference_output_2 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_2)],
    };
    let inference_output_bytes_2 =
        to_allocvec(&inference_output_2).expect("serialize inference output 2");
    let inference_output_id_2 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        inference_output_bytes_2,
    )
    .await;
    let emission_2 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_2),
    };
    let emission_bytes_2 = to_allocvec(&emission_2).expect("serialize emission 2");
    let emission_void_id_2 =
        void_upload(&make_client_endpoint().await, void_addr, emission_bytes_2).await;
    let propagation_2 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_2),
        recv: potentiation_void_id,
        send: listen_2,
    };
    let propagation_bytes_2 = to_allocvec(&propagation_2).expect("serialize propagation 2");
    let propagation_void_id_2 = void_upload(
        &make_client_endpoint().await,
        void_addr,
        propagation_bytes_2,
    )
    .await;

    // ── First propagation (points to second) ──
    let dark_tokens = text_to_dark_tokens(input_text, &tokenizer);
    let inference_output = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens)],
    };
    let inference_output_bytes =
        to_allocvec(&inference_output).expect("serialize inference output");
    let inference_output_id = void_upload(
        &make_client_endpoint().await,
        void_addr,
        inference_output_bytes,
    )
    .await;
    let emission = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id),
    };
    let emission_bytes = to_allocvec(&emission).expect("serialize emission");
    let emission_void_id =
        void_upload(&make_client_endpoint().await, void_addr, emission_bytes).await;
    let propagation = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id),
        recv: propagation_void_id_2,
        send: listen_1,
    };
    let propagation_bytes = to_allocvec(&propagation).expect("serialize propagation");
    let propagation_void_id =
        void_upload(&make_client_endpoint().await, void_addr, propagation_bytes).await;

    let initiation = Transmission::Initiation {
        recv: propagation_void_id,
    };
    let init_void_id = upload_transmission(void_addr, &initiation).await;

    // 5. Spawn the Progenitor with state pointing to the Initiation.
    let spawn_result = client.spawn::<Progenitor>(&init_void_id).await;
    assert!(
        spawn_result.is_ok(),
        "spawn should succeed: {:?}",
        spawn_result
    );
    let journey_id = spawn_result.unwrap().journey_id;
    println!("Spawned Progenitor journey: {journey_id}");

    // 6. Start a JungleWorker with Progenitor support.
    //    Spawned after subscribing so effects execute only after we're listening.
    let worker = JungleWorker::new(jungle, client.clone());
    let worker_handle = tokio::spawn(async move {
        let _ = worker.spawn().await;
    });

    let result = tokio::time::timeout(
        Duration::from_secs(60),
        async {
            let (t1, t2, t3, t4) = tokio::join!(
                wait_for_void_transmission(void_addr, listen_1),
                wait_for_void_transmission(void_addr, listen_2),
                wait_for_void_transmission(void_addr, listen_3),
                wait_for_void_transmission(void_addr, listen_4),
            );
            (t1, t2, t3, t4)
        },
    )
    .await;

    match result {
        Ok((
            Transmission::Propagation { emission_id: e1, .. },
            Transmission::Propagation { emission_id: e2, .. },
            Transmission::Propagation { emission_id: e3, .. },
            Transmission::Propagation { emission_id: e4, .. },
        )) => {
            println!("Flux flow completed through full cell optimization loop:");
            println!("  propagation 1 emitted {}", e1.0);
            println!("  propagation 2 emitted {}", e2.0);
            println!("  propagation 3 emitted {}", e3.0);
            println!("  propagation 4 emitted {}", e4.0);
        }
        Ok((t1, t2, t3, t4)) => {
            panic!(
                "expected Propagation transmissions, got {:?}, {:?}, {:?}, {:?}",
                t1, t2, t3, t4
            );
        }
        Err(e) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!("timeout waiting for flux flow outputs (60s): {e}, status: {status:?}");
        }
    }
    // Cleanup.
    worker_handle.abort();
    let _ = worker_handle.await;
    server_handle.abort();
    let _ = server_handle.await;
    drop(client);
    void_abort.abort();
    quark_abort.abort();
}
