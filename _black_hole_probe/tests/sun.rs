#[path = "common/mod.rs"]
mod common;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::cell::action::InitRecvId;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::{Binary, BlackHole, SunState, Unary};
use black_hole_flux::{
    AtomError, CellState, Fusion, FusionSeed, FusionState, Potentiation, Transmit,
    WaitForPotentiationAction, WaitForPropagationAction,
};
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{EmissionId, InferenceRequest, ObjectId, Transmission, VoidServerBuilder};
#[cfg(test)]
use futures::stream::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::to_allocvec;
use typosaurus::num::consts::*;
use uuid::Uuid;

use common::*;

const LEFT_EMISSION: u128 = 1;
const RIGHT_EMISSION: u128 = 2;
const FUSED_EMISSION: u128 = 3;

// ─── Multi-epoch diamond graph ───────────────────────────────────────────────

type Root = Unary<U0, RootAnimal, list![U1, U2]>;
type Left = Unary<U1, LeftAnimal, list![U3]>;
type Right = Unary<U2, RightAnimal, list![U4]>;
type Merge = Binary<U3, U4, FusionAnimal, list![U5]>;
type Sink = Unary<U5, SinkAnimal, list![]>;
type DiamondSun = list![Root, Left, Right, Merge, Sink];

// ─── Lightweight unary animals ───────────────────────────────────────────────

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
pub struct TestCellEpoch<Transform>(
    Step<WaitForPropagationAction>,
    Transform,
    Step<Transmit>,
    Step<WaitForPropagationAction>,
    Transform,
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
pub struct TestCellFlow<Transform>(
    Step<InitRecvId>,
    While<AlwaysEpoch, TestCellEpoch<Transform>>,
);

pub struct PassEmission;

#[jungle::action(carry = EmissionId)]
impl Action for PassEmission {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, input: Self::Input) -> ((), EmissionId) {
        ((), input)
    }

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
        emission_id: EmissionId,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("pass emission failed".to_string()))?;
        Ok(emission_id)
    }
}

pub struct MarkLeft;

pub struct DelayedLeftEffect;

impl<J> EffectSchema<J> for DelayedLeftEffect {
    type Id = u64;
    type In = ();
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for DelayedLeftEffect {
    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(EmissionId(Uuid::from_u128(LEFT_EMISSION)))
        }
    }
}

#[jungle::action]
impl Action for MarkLeft {
    type Effect = DelayedLeftEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("mark left emission failed: {error}")))
    }
}

pub struct MarkRight;

#[jungle::action]
impl Action for MarkRight {
    type Effect = NoEffect;
    type Input = EmissionId;
    type Output = EmissionId;

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("mark right emission failed".to_string()))?;
        Ok(EmissionId(Uuid::from_u128(RIGHT_EMISSION)))
    }
}

pub struct RootAnimal;

#[jungle::animal(id = 0, generation = 0)]
impl Animal for RootAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<PassEmission>>;
}

pub struct LeftAnimal;

#[jungle::animal(id = 2, generation = 0)]
impl Animal for LeftAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<MarkLeft>>;
}

pub struct RightAnimal;

#[jungle::animal(id = 3, generation = 0)]
impl Animal for RightAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<MarkRight>>;
}

pub struct SinkAnimal;

#[jungle::animal(id = 5, generation = 0)]
impl Animal for SinkAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<PassEmission>>;
}

// ─── Explicit fusion transform animal ────────────────────────────────────────

trait FusionProbeOps: Send + Sync {
    fn record_fusion_inputs(&self, p1: ObjectId, p2: ObjectId);
}

pub struct RecordFusionInputsEffect;

impl<J> EffectSchema<J> for RecordFusionInputsEffect {
    type Id = u64;
    type In = (EmissionId, EmissionId);
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for RecordFusionInputsEffect
where
    J: FusionProbeOps,
{
    fn effect(
        jungle: &J,
        (p1, p2): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        jungle.record_fusion_inputs(p1.0, p2.0);
        std::future::ready(Ok(EmissionId(Uuid::from_u128(FUSED_EMISSION))))
    }
}

pub struct RecordFusionInputs;

#[jungle::action]
impl Action for RecordFusionInputs {
    type Effect = RecordFusionInputsEffect;
    type Input = (EmissionId, EmissionId);
    type Output = EmissionId;

    fn emit(_state: &FusionState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("record fusion inputs failed: {error}")))
    }
}

#[derive(Flow)]
pub struct FusionTransform(Step<RecordFusionInputs>);

pub struct FusionAnimal;

#[jungle::animal(id = 4, generation = 0)]
impl Animal for FusionAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = Fusion<FusionTransform>;
}

// ─── BlackHoleAnimal ─────────────────────────────────────────────────────────

/// An animal that runs the full BlackHole orchestration flow over a Sun graph.
pub struct BlackHoleAnimal;

#[jungle::animal(id = 1, generation = 0)]
impl Animal for BlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = <DiamondSun as BlackHole>::Sun<Generator, Policy>;
}

// ─── Ecosystem ───────────────────────────────────────────────────────────────

#[derive(Animals)]
pub struct SpaceAnimals(
    RootAnimal,
    LeftAnimal,
    RightAnimal,
    FusionAnimal,
    SinkAnimal,
    BlackHoleAnimal,
);

/// A Jungle implementation backed by void over QUIC.
#[derive(Clone)]
pub struct SpaceJungle {
    void_addr: SocketAddr,
    client: Option<FusedClient>,
    potentiation_writes: Arc<AtomicUsize>,
    fusion_inputs: Arc<Mutex<Vec<(ObjectId, ObjectId)>>>,
}

impl SpaceJungle {
    pub fn new(void_addr: SocketAddr) -> Self {
        Self {
            void_addr,
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            fusion_inputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
    }
}

impl FusionProbeOps for SpaceJungle {
    fn record_fusion_inputs(&self, p1: ObjectId, p2: ObjectId) {
        self.fusion_inputs.lock().unwrap().push((p1, p2));
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

/// Exercises a multi-epoch unary diamond feeding one binary fusion vertex.
///
/// The left and right unary branches stamp distinct emission IDs, with the P1
/// branch deliberately delayed so P2 arrives first. The explicit fusion
/// transform records each pair, proving that declared `P1`, `P2` order remains
/// stable on both propagation passes in every completed epoch.
#[cfg(test)]
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
    let fusion_inputs = Arc::clone(&jungle.fusion_inputs);

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

    // 6. Run one worker per journey: five graph vertices plus the parent.
    let worker_handles: Vec<_> = (0..6)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    // 7. Wait for three complete epochs. There are six input ports: four
    // unary ports plus both independently chained binary ports.
    const EPOCHS: usize = 3;
    const PORT_COUNT: usize = 6;
    const EXPECTED_POTENTIATION_WRITES: usize = EPOCHS * PORT_COUNT;
    let result = tokio::time::timeout(Duration::from_secs(60), async {
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
                "timeout waiting for 3 epochs (60s): {}, potentiation writes: {}, status: {:?}",
                e,
                potentiation_writes.load(Ordering::SeqCst),
                status
            );
        }
    }

    let observed = fusion_inputs.lock().unwrap();
    assert!(
        observed.len() >= EPOCHS * 2,
        "expected two fusion transforms per epoch, observed {observed:?}"
    );
    for epoch in 0..EPOCHS {
        for pass in 0..2 {
            let (p1, p2) = observed[epoch * 2 + pass];
            assert_eq!(
                p1,
                Uuid::from_u128(LEFT_EMISSION),
                "epoch {epoch} propagation pass {pass} did not preserve P1"
            );
            assert_eq!(
                p2,
                Uuid::from_u128(RIGHT_EMISSION),
                "epoch {epoch} propagation pass {pass} did not preserve P2"
            );
        }
    }
    drop(observed);

    // Cleanup.
    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_abort.abort();
}

/// Runs the diamond Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    let (client, journey_id) = runtime.block_on(async {
        let (void_addr, _void_abort) = start_server().await;

        let mut jungle = SpaceJungle::new(void_addr);
        let client = FusedClient::builder()
            .build()
            .await
            .expect("fused client should build");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<BlackHoleAnimal>(&())
            .await
            .expect("BlackHoleAnimal should spawn")
            .journey_id;
        println!("Spawned BlackHoleAnimal journey: {journey_id}");

        let _worker_handles: Vec<_> = (0..6)
            .map(|_| {
                let worker = JungleWorker::new(jungle.clone(), client.clone());
                tokio::spawn(async move {
                    let _ = worker.spawn().await;
                })
            })
            .collect();

        (client, journey_id)
    });

    black_hole_beam::BeamBuilder::new()
        .title("Diamond Sun")
        .view_live::<BlackHoleAnimal>(client, journey_id)
        .expect("Black Hole Beam should run");
}

/// Launches the Beam example in a process whose UI runs on its main thread.
#[cfg(test)]
#[test]
#[ignore]
fn beam() {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = std::process::Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["run", "--quiet", "--example", "beam"])
        .status()
        .expect("Beam example should launch");

    assert!(status.success(), "Beam example exited with {status}");
}
