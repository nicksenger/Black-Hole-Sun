mod common;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::cell::action::InitRecvId;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::BlackHole;
use black_hole_flux::sun::{SunState, Unary};
use black_hole_flux::{
    CellState, Potentiation, Transmit, WaitForPotentiationAction, WaitForPropagationAction,
};
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{EmissionId, InferenceRequest, ObjectId, Transmission, VoidServerBuilder};
use futures::stream::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::to_allocvec;
use typosaurus::num::consts::*;
use uuid::Uuid;

use common::*;

// ─── 3-unary Sun graph type (U0 -> U1 -> U2) ────────────────────────────────

/// Unary nodes: each node carries a typenum index, the animal type,
/// and the list of outgoing edge targets.
type Unary0 = Unary<U0, TestCell, list![U1]>;
type Unary1 = Unary<U1, TestCell, list![U2]>;
type Unary2 = Unary<U2, TestCell, list![]>;

/// The complete three-node sun: a type-level list of all unary nodes.
type ThreeUnarySunType = list![Unary0, Unary1, Unary2];

// ─── TestCell ────────────────────────────────────────────────────────────────

/// Completes one test-cell epoch after consuming its potentiation.
pub struct FinishEpoch;

#[jungle::action]
impl Action for FinishEpoch {
    type Effect = NoEffect;
    type Input = Potentiation;
    type Output = ();

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("finish epoch failed".to_string()))
    }
}

#[derive(Flow)]
pub struct TestCellEpoch(
    Step<WaitForPropagationAction>,
    Step<Transmit>,
    Step<WaitForPropagationAction>,
    Step<Transmit>,
    Step<WaitForPotentiationAction>,
    Step<FinishEpoch>,
);

pub struct AlwaysEpoch;

impl Predicate<(&CellState, &())> for AlwaysEpoch {
    fn eval(_input: &(&CellState, &())) -> bool {
        true
    }
}

#[derive(Flow)]
pub struct TestCellFlow(Step<InitRecvId>, While<AlwaysEpoch, TestCellEpoch>);

/// A lightweight cell protocol used to isolate Sun orchestration from model
/// inference, which is covered by the separate cell integration test.
pub struct TestCell;

#[jungle::animal(id = 0, generation = 0)]
impl Animal for TestCell {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow;
}

// ─── BlackHoleAnimal ─────────────────────────────────────────────────────────

/// An animal that runs the full BlackHole orchestration flow over a Sun graph.
pub struct BlackHoleAnimal;

#[jungle::animal(id = 1, generation = 0)]
impl Animal for BlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeUnarySunType as BlackHole>::Sun;
}

// ─── Ecosystem ───────────────────────────────────────────────────────────────

#[derive(Animals)]
pub struct SpaceAnimals(TestCell, BlackHoleAnimal);

/// A Jungle implementation backed by void over QUIC.
#[derive(Clone)]
pub struct SpaceJungle {
    void_addr: SocketAddr,
    client: Option<FusedClient>,
    potentiation_writes: Arc<AtomicUsize>,
}

impl SpaceJungle {
    pub fn new(void_addr: SocketAddr) -> Self {
        Self {
            void_addr,
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
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

    async fn infer(&self, _request: InferenceRequest) -> Result<ObjectId, String> {
        Err("inference is not used by TestCell".to_string())
    }

    async fn perturb_up(&self, _seed: u64) -> Result<(), String> {
        Err("perturbation is not used by TestCell".to_string())
    }

    async fn perturb_down(&self) -> Result<(), String> {
        Err("perturbation is not used by TestCell".to_string())
    }

    async fn optimize(&self, _loss_up: f32, _loss_down: f32) -> Result<(), String> {
        Err("optimization is not used by TestCell".to_string())
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

#[async_trait]
impl SunOps for SpaceJungle {
    async fn spawn_animal<A: Animal>(&self, seed: &A::Seed) -> Result<Uuid, String>
    where
        A::Id: AnimalIdValue,
        <A as Animal>::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Send + Sync + Send,
    {
        let client = self.client.clone().expect("client not set");
        let handle = client.spawn::<A>(seed).await.map_err(|e| e.to_string())?;
        Ok(handle.journey_id)
    }
}

// ─── Server helper ───────────────────────────────────────────────────────────

async fn start_server() -> (SocketAddr, tokio::task::AbortHandle) {
    let object_store = Box::new(InMemoryObjectStore::new());
    let store = Box::new(InMemoryStore::new());
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");
    let void_abort = void_handle.abort_handle();

    drop(void_handle);

    (void_local, void_abort)
}

// ─── Test ────────────────────────────────────────────────────────────────────

/// Exercises the full BlackHole flow over a three-node Sun graph (U0 -> U1 -> U2).
///
/// This test follows the same pattern as the cell.rs test but uses the BlackHole
/// Flow to automatically handle all input generation, propagation, loss computation,
/// and potentiation broadcasting. No manual upload of inputs or void interactions
/// is needed — the Flow orchestrates everything.
///
/// Three potentiation writes per epoch prove that every tagged cell completed
/// both propagation passes and that the parent reached the epoch boundary.
#[tokio::test]
async fn sun() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    // 1. Start void on a random port.
    let (void_addr, void_abort) = start_server().await;

    // 2. Build the SpaceJungle with void capabilities.
    let mut jungle = SpaceJungle::new(void_addr);
    let potentiation_writes = Arc::clone(&jungle.potentiation_writes);

    // 3. Build a FusedClient with in-memory backend.
    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");

    // Store the client inside SpaceJungle so SunOps can spawn child animals.
    jungle.set_client(client.clone());

    // 4. Spawn the BlackHoleAnimal with unit seed — no manual input needed.
    let spawn_result = client.spawn::<BlackHoleAnimal>(&()).await;
    assert!(
        spawn_result.is_ok(),
        "spawn should succeed: {:?}",
        spawn_result
    );
    let journey_id = spawn_result.unwrap().journey_id;
    println!("Spawned BlackHoleAnimal journey: {journey_id}");

    // 5. Subscribe to step updates for the journey.
    let mut subscription = client
        .subscribe_step_updates(journey_id, None)
        .await
        .expect("subscribe_step_updates should succeed");

    // 6. Run one worker per journey so the parent can wait while its three
    // child journeys execute.
    let worker_handles: Vec<_> = (0..4)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    // 7. Wait for three complete epochs. Each parent broadcast writes one
    // potentiation transmission for each of the three tagged cells.
    const EXPECTED_POTENTIATION_WRITES: usize = 3 * 3;
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let writes = potentiation_writes.load(Ordering::SeqCst);
            if writes >= EXPECTED_POTENTIATION_WRITES {
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
                            return Err("step update stream ended before three epochs".to_string());
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
    })
    .await;

    match result {
        Ok(Ok(())) => {
            println!("BlackHole flow completed 3 epochs");
        }
        Ok(Err(e)) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!("flow assertion failed: {}, status: {:?}", e, status);
        }
        Err(e) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!(
                "timeout waiting for 3 epochs (30s): {}, potentiation writes: {}, status: {:?}",
                e,
                potentiation_writes.load(Ordering::SeqCst),
                status
            );
        }
    }

    // Cleanup.
    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_abort.abort();
}
