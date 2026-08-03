mod common;

use std::net::SocketAddr;
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
    Emission, EmissionId, InferenceOutputId, ObjectId, QuarkServerBuilder, Transmission,
    VoidServerBuilder,
};
use futures::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::server::ServerBuilder;
use postcard::to_allocvec;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

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

    async fn infer(&self, input_id: ObjectId) -> Result<ObjectId, String> {
        let endpoint = make_client_endpoint().await;
        Ok(quark_infer(&endpoint, self.quark_addr, input_id).await)
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

    async fn optimize(&self, _loss_up: f32, _loss_down: f32) -> Result<(), String> {
        Ok(())
    }

    async fn transmit(&self, _emission_id: EmissionId) -> Result<(), String> {
        Ok(())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async fn upload_transmission(addr: SocketAddr, transmission: &Transmission) -> ObjectId {
    let endpoint = make_client_endpoint().await;
    let data = to_allocvec(transmission).expect("failed to serialize transmission");
    void_upload(&endpoint, addr, data).await
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
/// 4. Start a JungleWorker supporting the Progenitor.
/// 5. Upload a Propagation transmission to void, then spawn a Progenitor.
/// 6. Assert that effects execute successfully within a 30s timeout.
#[tokio::test]
async fn progenitor_flux_flow() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("progenitor_flux_flow") {
        Some(path) => path,
        None => return,
    };

    // 1. Start void and quark servers on random ports.
    let (void_addr, void_abort, quark_addr, quark_abort) = start_servers(&model_path).await;

    // 2. Build the SpaceJungle with void/quark capabilities.
    let jungle = SpaceJungle::new(void_addr, quark_addr);

    // 3. Spawn a jungle-server with in-memory backend and connect a client.
    let listen_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let server_handle = tokio::spawn(async move {
        ServerBuilder::new()
            .listen(listen_addr)
            .memory()
            .run()
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = connect_client_with_retry(listen_addr).await;

    // 4. Start a JungleWorker with Progenitor support.
    let worker = JungleWorker::new(jungle, client.clone());
    let worker_handle = tokio::spawn(async move {
        let _ = worker.spawn().await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // 5. Prepare emission data and Propagation transmission in void.
    //    The Progenitor flow: PerturbUp -> WaitForPropagation -> QuarkInfer -> PerturbDown
    //    WaitForPropagation reads recv_id from state and downloads a Transmission::Propagation.

    let inference_output_id = ObjectId::nil();
    let emission = Emission {
        metadata: (),
        output_id: InferenceOutputId(inference_output_id),
    };
    let emission_bytes = to_allocvec(&emission).expect("serialize emission");
    let emission_obj_id =
        void_upload(&make_client_endpoint().await, void_addr, emission_bytes).await;

    let propagation = Transmission::Propagation {
        emission_id: EmissionId(emission_obj_id),
        recv: ObjectId::nil(),
        send: ObjectId::nil(),
    };
    let propagation_id = upload_transmission(void_addr, &propagation).await;

    // 6. Spawn the Progenitor with state pointing to the Propagation.
    let spawn_result = client.spawn::<Progenitor>(&()).await;
    assert!(
        spawn_result.is_ok(),
        "spawn should succeed: {:?}",
        spawn_result
    );
    let journey_id = spawn_result.unwrap().journey_id;
    println!("Spawned Progenitor journey: {journey_id}");

    // 7. Subscribe to step updates and wait for effect completion (30s timeout).
    let mut subscription = client
        .subscribe_step_updates(journey_id, None)
        .await
        .expect("subscribe should succeed");

    let data_received = tokio::time::timeout(Duration::from_secs(30), async {
        use jungle_sdk::RunnerUpdateOut;
        let mut effects_started = 0u32;
        let mut effects_succeeded = 0u32;

        while let Some(update_result) = subscription.next().await {
            let update = update_result.expect("stream item should be ok");
            match update.event {
                RunnerUpdateOut::EffectInput { .. } => {
                    effects_started += 1;
                }
                RunnerUpdateOut::EffectSuccessOutput { .. } => {
                    effects_succeeded += 1;
                    if effects_succeeded >= 2 {
                        return Some((effects_started, effects_succeeded));
                    }
                }
                RunnerUpdateOut::EffectFailureOutput { uuid, .. } => {
                    println!("Effect failed for journey {uuid}");
                    return None;
                }
                RunnerUpdateOut::NodeLifecycle { .. }
                | RunnerUpdateOut::SleepScheduled { .. }
                | RunnerUpdateOut::SleepFired { .. }
                | RunnerUpdateOut::PerturbationApplied { .. } => {}
            }
        }
        Some((effects_started, effects_succeeded))
    })
    .await;

    match data_received {
        Ok(Some((started, succeeded))) => {
            println!("Flux flow completed: {started} effects started, {succeeded} succeeded");
            assert!(started > 0, "expected at least one effect to start");
            assert!(succeeded > 0, "expected at least one effect to succeed");
        }
        Ok(None) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!("stream ended without effect success, status: {status:?}");
        }
        Err(e) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!("timeout waiting for flux flow data (30s): {e}, status: {status:?}");
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
