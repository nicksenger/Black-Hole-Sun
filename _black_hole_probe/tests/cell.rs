mod common;

use std::net::{Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::atom::effect::QuarkInfer;
use black_hole_sun::cell::action::CellState;
use black_hole_sun::cell::effect::{QuarkPerturbDown, QuarkPerturbUp, WaitForPropagation};
use black_hole_sun::ops::VoidInferOps;
use black_hole_sun::{
    Emission, EmissionId, InferenceOutput, InferenceOutputId, InferenceRequest, ObjectId,
    Progenitor, QuarkClient, SequenceOutput, TestQuarkServer, TestVoidServer, Tokenizer,
    Transmission, VoidClient,
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
    void_client: VoidClient,
    quark_client: QuarkClient,
    tokenizer: Arc<OnceLock<Result<Tokenizer, String>>>,
}

impl SpaceJungle {
    pub fn new(void_client: VoidClient, quark_client: QuarkClient) -> Self {
        Self {
            void_client,
            quark_client,
            tokenizer: Arc::new(OnceLock::new()),
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
        Ok(self.void_client.download(id).await.unwrap())
    }

    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        self.void_client.download_wait(id, timeout_ms).await
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        Ok(self.void_client.upload(data).await.unwrap())
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        self.void_client.upload_with(id, data).await.unwrap();
        Ok(())
    }

    fn darken(&self, prompt: &str) -> Result<Vec<black_hole_sun::DarkToken>, String> {
        let tokenizer_result = self.tokenizer.get_or_init(Tokenizer::try_init);
        let tokenizer = tokenizer_result
            .as_ref()
            .map_err(|error| format!("failed to initialize tokenizer: {error}"))?;
        tokenizer
            .darken(prompt)
            .map_err(|error| format!("failed to darken prompt: {error}"))
    }

    fn decode(&self, tokens: &[black_hole_sun::DarkToken]) -> String {
        let tokenizer_result = self.tokenizer.get_or_init(Tokenizer::try_init);
        match tokenizer_result.as_ref() {
            Ok(tokenizer) => tokenizer.decode(tokens),
            Err(_) => tokens
                .iter()
                .map(|token| token.predicted.to_string())
                .collect(),
        }
    }

    async fn start_model(&self, model_id: Uuid) -> Result<(), String> {
        self.quark_client.start(model_id).await
    }

    async fn infer(&self, model_id: Uuid, request: InferenceRequest) -> Result<ObjectId, String> {
        let request_bytes = to_allocvec(&request).map_err(|e| format!("serialize: {e}"))?;
        let request_id = self.void_client.upload(request_bytes).await.unwrap();
        self.quark_client.infer(model_id, request_id).await
    }

    async fn perturb_up(&self, model_id: Uuid, seed: u64) -> Result<(), String> {
        self.quark_client.perturb_up(model_id, seed).await
    }

    async fn perturb_down(&self, model_id: Uuid) -> Result<(), String> {
        self.quark_client.perturb_down(model_id).await
    }

    async fn optimize(&self, model_id: Uuid, loss_up: f32, loss_down: f32) -> Result<(), String> {
        self.quark_client
            .optimize(model_id, loss_up, loss_down)
            .await
    }

    async fn shutdown_model(&self, model_id: Uuid) -> Result<(), String> {
        self.quark_client.shutdown(model_id).await
    }

    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String> {
        let propagation = Transmission::Propagation {
            emission_id,
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        };
        let data = to_allocvec(&propagation).map_err(|e| format!("serialize: {e}"))?;
        self.void_client.upload_with(send_id, data).await.unwrap();
        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn upload_transmission(void_client: &VoidClient, transmission: &Transmission) -> ObjectId {
    let data = to_allocvec(transmission).expect("failed to serialize transmission");
    void_client.upload(data).await.unwrap()
}

/// Poll void until data appears at `id`, deserializing as `Transmission`.
async fn wait_for_void_transmission(void_client: VoidClient, id: ObjectId) -> Transmission {
    loop {
        match void_client.download_wait(id, 30_000).await {
            Ok(Some(data)) => {
                return postcard::from_bytes(&data)
                    .expect("failed to deserialize Transmission from void");
            }
            Ok(None) => {
                tracing::debug!(%id, "download_wait timed out, retrying");
            }
            Err(e) => {
                tracing::debug!(%id, error = %e, "download_wait failed, retrying");
            }
        }
    }
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
async fn cell() {
    init_tracing();

    let listen_1 = Uuid::new_v4();
    let listen_2 = Uuid::new_v4();
    let listen_3 = Uuid::new_v4();
    let listen_4 = Uuid::new_v4();

    let model_path = match require_model_path("cell") {
        Some(path) => path,
        None => return,
    };

    // 1. Start void and quark servers on random ports.
    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");
    let void_addr = void_server.local_addr();
    let quark_addr = quark_server.local_addr();

    // 2. Build the SpaceJungle with void/quark capabilities.
    let endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
    let quark_client = QuarkClient::new(&endpoint, quark_addr, "localhost");
    let jungle = SpaceJungle::new(void_client.clone(), quark_client);

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
    //    PerturbUp -> WaitForPropagation -> Atom -> Transmit ->
    //    PerturbDown -> WaitForPropagation -> Atom -> Transmit ->
    //    WaitForPotentiation -> Optimize -> (loop back to WaitForPropagation ...)
    //
    //    Chain: Propagation(1) -> Propagation(2) -> Potentiation -> Propagation(3) -> Propagation(4)

    let input_text =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
    let tokenizer = Tokenizer::init();

    // ── Fourth propagation (end of chain) ──
    let dark_tokens_4 = tokenizer
        .darken(input_text)
        .expect("failed to darken input for inference output 4");
    let inference_output_4 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_4)],
    };
    let inference_output_bytes_4 =
        to_allocvec(&inference_output_4).expect("serialize inference output 4");
    let inference_output_id_4 = void_client.upload(inference_output_bytes_4).await.unwrap();
    let emission_4 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_4),
    };
    let emission_bytes_4 = to_allocvec(&emission_4).expect("serialize emission 4");
    let emission_void_id_4 = void_client.upload(emission_bytes_4).await.unwrap();
    let propagation_4 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_4),
        recv: ObjectId::nil(),
        send: listen_4,
    };
    let propagation_bytes_4 = to_allocvec(&propagation_4).expect("serialize propagation 4");
    let propagation_void_id_4 = void_client.upload(propagation_bytes_4).await.unwrap();

    // ── Third propagation (points to fourth) ──
    let dark_tokens_3 = tokenizer
        .darken(input_text)
        .expect("failed to darken input for inference output 3");
    let inference_output_3 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_3)],
    };
    let inference_output_bytes_3 =
        to_allocvec(&inference_output_3).expect("serialize inference output 3");
    let inference_output_id_3 = void_client.upload(inference_output_bytes_3).await.unwrap();
    let emission_3 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_3),
    };
    let emission_bytes_3 = to_allocvec(&emission_3).expect("serialize emission 3");
    let emission_void_id_3 = void_client.upload(emission_bytes_3).await.unwrap();
    let propagation_3 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_3),
        recv: propagation_void_id_4,
        send: listen_3,
    };
    let propagation_bytes_3 = to_allocvec(&propagation_3).expect("serialize propagation 3");
    let propagation_void_id_3 = void_client.upload(propagation_bytes_3).await.unwrap();

    // ── Potentiation (links second propagation to third) ──
    let potentiation = Transmission::Potentiation {
        loss_up: 0.5,
        loss_down: 0.3,
        recv: propagation_void_id_3,
    };
    let potentiation_bytes = to_allocvec(&potentiation).expect("serialize potentiation");
    let potentiation_void_id = void_client.upload(potentiation_bytes).await.unwrap();

    // ── Second propagation (points to potentiation) ──
    let dark_tokens_2 = tokenizer
        .darken(input_text)
        .expect("failed to darken input for inference output 2");
    let inference_output_2 = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens_2)],
    };
    let inference_output_bytes_2 =
        to_allocvec(&inference_output_2).expect("serialize inference output 2");
    let inference_output_id_2 = void_client.upload(inference_output_bytes_2).await.unwrap();
    let emission_2 = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id_2),
    };
    let emission_bytes_2 = to_allocvec(&emission_2).expect("serialize emission 2");
    let emission_void_id_2 = void_client.upload(emission_bytes_2).await.unwrap();
    let propagation_2 = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id_2),
        recv: potentiation_void_id,
        send: listen_2,
    };
    let propagation_bytes_2 = to_allocvec(&propagation_2).expect("serialize propagation 2");
    let propagation_void_id_2 = void_client.upload(propagation_bytes_2).await.unwrap();

    // ── First propagation (points to second) ──
    let dark_tokens = tokenizer
        .darken(input_text)
        .expect("failed to darken input for inference output 1");
    let inference_output = InferenceOutput {
        results: vec![SequenceOutput(dark_tokens)],
    };
    let inference_output_bytes =
        to_allocvec(&inference_output).expect("serialize inference output");
    let inference_output_id = void_client.upload(inference_output_bytes).await.unwrap();
    let emission = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id),
    };
    let emission_bytes = to_allocvec(&emission).expect("serialize emission");
    let emission_void_id = void_client.upload(emission_bytes).await.unwrap();
    let propagation = Transmission::Propagation {
        emission_id: EmissionId(emission_void_id),
        recv: propagation_void_id_2,
        send: listen_1,
    };
    let propagation_bytes = to_allocvec(&propagation).expect("serialize propagation");
    let propagation_void_id = void_client.upload(propagation_bytes).await.unwrap();

    // 5. Spawn the Progenitor with state pointing to the first propagation.
    let spawn_result = client.spawn::<Progenitor>(&propagation_void_id).await;
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

    let result = tokio::time::timeout(Duration::from_secs(60), async {
        let wait_client_1 = void_client.clone();
        let wait_client_2 = void_client.clone();
        let wait_client_3 = void_client.clone();
        let wait_client_4 = void_client.clone();
        let (t1, t2, t3, t4) = tokio::join!(
            wait_for_void_transmission(wait_client_1, listen_1),
            wait_for_void_transmission(wait_client_2, listen_2),
            wait_for_void_transmission(wait_client_3, listen_3),
            wait_for_void_transmission(wait_client_4, listen_4),
        );
        (t1, t2, t3, t4)
    })
    .await;

    match result {
        Ok((
            Transmission::Propagation {
                emission_id: e1, ..
            },
            Transmission::Propagation {
                emission_id: e2, ..
            },
            Transmission::Propagation {
                emission_id: e3, ..
            },
            Transmission::Propagation {
                emission_id: e4, ..
            },
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
    void_server.abort();
    quark_server.abort();
}
