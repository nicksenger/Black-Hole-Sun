mod common;

use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::{BlackHole, SunState, Tag};
use black_hole_flux::Progenitor;
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{
    EmissionId, InferenceRequest, ObjectId, QuarkServerBuilder, Transmission, VoidServerBuilder,
};
use futures::stream::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::to_allocvec;
use tokio::sync::Barrier;
use typosaurus::num::consts::*;
use uuid::Uuid;

use common::*;

const NODE_COUNT: usize = 3;

type Tag0 = Tag<U0, Progenitor, list![U1]>;
type Tag1 = Tag<U1, Progenitor, list![U2]>;
type Tag2 = Tag<U2, Progenitor, list![]>;
type ThreeProgenitorSun = list![Tag0, Tag1, Tag2];

pub struct ProgenitorBlackHole;

#[jungle::animal(id = 1, generation = 0)]
impl Animal for ProgenitorBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun;
}

#[derive(Animals)]
pub struct SpaceAnimals(Progenitor, ProgenitorBlackHole);

/// Coordinates a model mutation once all cells reach the same training phase.
///
/// The three Progenitor journeys share one quark, so perturbation and
/// optimization apply once per Sun epoch while each node still runs inference.
#[derive(Clone)]
struct ModelPhase {
    enter: Arc<Barrier>,
    operation_complete: Arc<Barrier>,
    result_read: Arc<Barrier>,
    result: Arc<Mutex<Option<Result<(), String>>>>,
}

impl ModelPhase {
    fn new(participants: usize) -> Self {
        Self {
            enter: Arc::new(Barrier::new(participants)),
            operation_complete: Arc::new(Barrier::new(participants)),
            result_read: Arc::new(Barrier::new(participants)),
            result: Arc::new(Mutex::new(None)),
        }
    }

    async fn run<F, Fut>(&self, operation: F) -> Result<(), String>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<(), String>> + Send,
    {
        let wait = self.enter.wait().await;
        if wait.is_leader() {
            *self.result.lock().unwrap() = Some(operation().await);
        }

        self.operation_complete.wait().await;
        let result = self
            .result
            .lock()
            .unwrap()
            .clone()
            .expect("phase leader should publish a result");
        self.result_read.wait().await;
        result
    }
}

#[derive(Clone)]
pub struct SpaceJungle {
    void_addr: SocketAddr,
    quark_addr: SocketAddr,
    client: Option<FusedClient>,
    perturb_up_phase: ModelPhase,
    perturb_down_phase: ModelPhase,
    optimize_phase: ModelPhase,
    potentiation_writes: Arc<AtomicUsize>,
    inference_calls: Arc<AtomicUsize>,
    optimized_cells: Arc<AtomicUsize>,
    model_error: Arc<Mutex<Option<String>>>,
}

impl SpaceJungle {
    fn new(void_addr: SocketAddr, quark_addr: SocketAddr) -> Self {
        Self {
            void_addr,
            quark_addr,
            client: None,
            perturb_up_phase: ModelPhase::new(NODE_COUNT),
            perturb_down_phase: ModelPhase::new(NODE_COUNT),
            optimize_phase: ModelPhase::new(NODE_COUNT),
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            inference_calls: Arc::new(AtomicUsize::new(0)),
            optimized_cells: Arc::new(AtomicUsize::new(0)),
            model_error: Arc::new(Mutex::new(None)),
        }
    }

    fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
    }

    fn record_model_error<T>(&self, operation: &str, result: &Result<T, String>) {
        if let Err(error) = result {
            let mut first_error = self.model_error.lock().unwrap();
            if first_error.is_none() {
                *first_error = Some(format!("{operation}: {error}"));
            }
        }
    }
}

impl Ecosystem for SpaceJungle {
    const NAME: &'static str = "progenitor-sun-jungle";
    type Animals = SpaceAnimals;
}

#[async_trait]
impl VoidInferOps for SpaceJungle {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        let endpoint = make_client_endpoint().await;
        void_download_result(&endpoint, self.void_addr, id).await
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        let endpoint = make_client_endpoint().await;
        Ok(void_upload(&endpoint, self.void_addr, data).await)
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        let is_potentiation = matches!(
            postcard::from_bytes(&data),
            Ok(Transmission::Potentiation { .. })
        );
        let endpoint = make_client_endpoint().await;
        void_upload_with(&endpoint, self.void_addr, id, data).await;
        if is_potentiation {
            self.potentiation_writes.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn infer(&self, request: InferenceRequest) -> Result<ObjectId, String> {
        // One generated token is enough to prove each Progenitor atom reached
        // the real model while keeping this integration test bounded.
        let request = match request {
            InferenceRequest::Sequences { sequences, .. } => InferenceRequest::Sequences {
                sequences,
                limit: 1,
            },
            InferenceRequest::VoidId { id, .. } => InferenceRequest::VoidId { id, limit: 1 },
        };
        let request_bytes = to_allocvec(&request).map_err(|error| error.to_string())?;
        let endpoint = make_client_endpoint().await;
        let request_id = void_upload(&endpoint, self.void_addr, request_bytes).await;
        let result = quark_infer_result(&endpoint, self.quark_addr, request_id).await;
        self.record_model_error("infer", &result);
        if result.is_ok() {
            self.inference_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn perturb_up(&self, seed: u64) -> Result<(), String> {
        let quark_addr = self.quark_addr;
        let result = self
            .perturb_up_phase
            .run(move || async move {
                let endpoint = make_client_endpoint().await;
                quark_perturb_up_result(&endpoint, quark_addr, seed).await
            })
            .await;
        self.record_model_error("perturb up", &result);
        result
    }

    async fn perturb_down(&self) -> Result<(), String> {
        let quark_addr = self.quark_addr;
        let result = self
            .perturb_down_phase
            .run(move || async move {
                let endpoint = make_client_endpoint().await;
                quark_perturb_down_result(&endpoint, quark_addr).await
            })
            .await;
        self.record_model_error("perturb down", &result);
        result
    }

    async fn optimize(&self, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let quark_addr = self.quark_addr;
        let result = self
            .optimize_phase
            .run(move || async move {
                let endpoint = make_client_endpoint().await;
                quark_optimize_result(&endpoint, quark_addr, loss_up, loss_down).await
            })
            .await;
        self.record_model_error("optimize", &result);
        if result.is_ok() {
            self.optimized_cells.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String> {
        let propagation = Transmission::Propagation {
            emission_id,
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        };
        let data = to_allocvec(&propagation).map_err(|error| error.to_string())?;
        let endpoint = make_client_endpoint().await;
        void_upload_with(&endpoint, self.void_addr, send_id, data).await;
        Ok(())
    }
}

#[async_trait]
impl SunOps for SpaceJungle {
    async fn spawn_animal<A: Animal>(&self, seed: &A::Seed) -> Result<Uuid, String>
    where
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Send + Sync + Send,
    {
        let client = self.client.clone().expect("client not set");
        let handle = client
            .spawn::<A>(seed)
            .await
            .map_err(|error| error.to_string())?;
        Ok(handle.journey_id)
    }
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
    let (void_addr, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen("127.0.0.1:0".parse().unwrap())
        .serve()
        .await
        .expect("failed to start void server");
    let void_abort = void_handle.abort_handle();

    let (quark_addr, quark_handle) = QuarkServerBuilder::new(PathBuf::from(model_path))
        .listen("127.0.0.1:0".parse().unwrap())
        .void_addr(void_addr)
        .serve()
        .await
        .expect("failed to start quark server");
    let quark_abort = quark_handle.abort_handle();

    drop(void_handle);
    drop(quark_handle);
    tokio::time::sleep(Duration::from_millis(200)).await;

    (void_addr, void_abort, quark_addr, quark_abort)
}

/// Runs the same U0 -> U1 -> U2 Sun topology as `sun`, with real Progenitor
/// cells backed by a quark model.
#[tokio::test]
async fn sun_progenitor() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("sun_progenitor") {
        Some(path) => path,
        None => return,
    };
    let (void_addr, void_abort, quark_addr, quark_abort) = start_servers(&model_path).await;

    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");
    let mut jungle = SpaceJungle::new(void_addr, quark_addr);
    jungle.set_client(client.clone());

    let potentiation_writes = Arc::clone(&jungle.potentiation_writes);
    let inference_calls = Arc::clone(&jungle.inference_calls);
    let optimized_cells = Arc::clone(&jungle.optimized_cells);
    let model_error = Arc::clone(&jungle.model_error);

    let parent = client
        .spawn::<ProgenitorBlackHole>(&())
        .await
        .expect("Progenitor Sun should spawn");
    let mut subscription = client
        .subscribe_step_updates(parent.journey_id, None)
        .await
        .expect("parent subscription should succeed");

    let worker_handles: Vec<_> = (0..NODE_COUNT + 1)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    let result = tokio::time::timeout(Duration::from_secs(180), async {
        loop {
            if let Some(error) = model_error.lock().unwrap().clone() {
                return Err(error);
            }

            if potentiation_writes.load(Ordering::SeqCst) >= NODE_COUNT
                && inference_calls.load(Ordering::SeqCst) >= NODE_COUNT * 2
                && optimized_cells.load(Ordering::SeqCst) >= NODE_COUNT
            {
                return Ok::<(), String>(());
            }

            tokio::select! {
                update = subscription.next() => {
                    match update {
                        Some(Ok(update)) => match update.event {
                            RunnerUpdateOut::EffectFailureOutput { node_id, .. } => {
                                return Err(format!("parent effect {node_id} failed"));
                            }
                            RunnerUpdateOut::NodeLifecycle(node)
                                if node.phase == jungle_sdk::types::NodeLifecyclePhase::Failed =>
                            {
                                return Err(format!("parent node {} failed", node.node_id));
                            }
                            _ => {}
                        },
                        Some(Err(error)) => {
                            return Err(format!("step update stream failed: {error}"));
                        }
                        None => {
                            return Err("step update stream ended before one epoch".to_string());
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
    })
    .await;

    if let Err(error) = &result {
        let status = client
            .journey_details(parent.journey_id)
            .await
            .expect("parent journey details should be available");
        panic!(
            "timeout waiting for Progenitor Sun epoch ({error}); \
             inferences={}, potentiations={}, optimized_cells={}, status={status:?}",
            inference_calls.load(Ordering::SeqCst),
            potentiation_writes.load(Ordering::SeqCst),
            optimized_cells.load(Ordering::SeqCst),
        );
    }
    if let Ok(Err(error)) = result {
        panic!("Progenitor Sun failed: {error}");
    }

    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_abort.abort();
    quark_abort.abort();
}
