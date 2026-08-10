mod action;
#[path = "../common/mod.rs"]
mod common;
mod effect;

#[cfg(test)]
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::cell::action::InitRecvId;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::{Binary, BlackHole, SunAppearance, SunNodeState, SunState, Unary};
use black_hole_flux::{
    CellState, Fusion, FusionSeed, FusionState, Potentiation, Progenitor, Transmit,
    WaitForPotentiationAction, WaitForPropagationAction,
};
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{
    DarkToken, EmissionId, InferenceRequest, LogitEntry, ObjectId, QuarkServerBuilder,
    Transmission, VoidServerBuilder,
};
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

type FusionObservation = (Uuid, ObjectId, ObjectId);

// ─── Multi-epoch diamond graph ───────────────────────────────────────────────

type Root = Unary<U0, RootAnimal, list![U1, U2]>;
type Left = Unary<U1, LeftAnimal, list![U3]>;
type Right = Unary<U2, RightAnimal, list![U4]>;
type Merge = Binary<U3, U4, FusionAnimal, list![U5]>;
type Sink = Unary<U5, SinkAnimal, list![]>;
type DiamondSun = list![Root, Left, Right, Merge, Sink];

// ─── Expanded diamond graph ──────────────────────────────────────────────────

type ExpandedInput = Unary<U0, RootAnimal, list![U1, U2]>;
type ExpandedL0 = Unary<U1, RootAnimal, list![U3, U4]>;
type ExpandedR0 = Unary<U2, RootAnimal, list![U5, U6]>;
type ExpandedL1 = Unary<U3, LeftAnimal, list![U7]>;
type ExpandedR1 = Unary<U4, RightAnimal, list![U8]>;
type ExpandedL2 = Unary<U5, LeftAnimal, list![U9]>;
type ExpandedR2 = Unary<U6, RightAnimal, list![U10]>;
type ExpandedF0 = Binary<U7, U8, FusionAnimal, list![U11]>;
type ExpandedF1 = Binary<U9, U10, FusionAnimal, list![U12]>;
type ExpandedF2 = Binary<U11, U12, FusionAnimal, list![]>;
type ExpandedDiamondSun = list![
    ExpandedInput,
    ExpandedL0,
    ExpandedR0,
    ExpandedL1,
    ExpandedR1,
    ExpandedL2,
    ExpandedR2,
    ExpandedF0,
    ExpandedF1,
    ExpandedF2
];

// ─── Lightweight unary animals ───────────────────────────────────────────────

/// Completes one test-cell epoch after consuming its potentiation.
pub struct FinishEpoch;

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

pub struct MarkLeft;

pub struct DelayedLeftEffect;

pub struct MarkRight;

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
    fn record_fusion_inputs(&self, transform_id: Uuid, p1: ObjectId, p2: ObjectId);
}

pub struct RecordFusionInputsEffect;

pub struct RecordFusionInputs;

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

#[jungle::animal(observe, id = 1, generation = 0)]
impl Animal for BlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = <DiamondSun as BlackHole>::Sun<Generator, Policy>;
}

impl Observe for BlackHoleAnimal {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

/// Runs the expanded, three-fusion diamond topology.
pub struct ExpandedBlackHoleAnimal;

#[jungle::animal(observe, id = 6, generation = 0)]
impl Animal for ExpandedBlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = <ExpandedDiamondSun as BlackHole>::Sun<Generator, Policy>;
}

impl Observe for ExpandedBlackHoleAnimal {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

// ─── Ecosystem ───────────────────────────────────────────────────────────────

#[derive(Animals)]
pub struct ProbeSpaceAnimals(
    RootAnimal,
    LeftAnimal,
    RightAnimal,
    FusionAnimal,
    SinkAnimal,
    BlackHoleAnimal,
    ExpandedBlackHoleAnimal,
);

/// A Jungle implementation backed by void over QUIC.
#[derive(Clone)]
pub struct ProbeSpaceJungle {
    void_addr: SocketAddr,
    client: Option<FusedClient>,
    potentiation_writes: Arc<AtomicUsize>,
    fusion_inputs: Arc<Mutex<Vec<FusionObservation>>>,
}

impl ProbeSpaceJungle {
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

impl FusionProbeOps for ProbeSpaceJungle {
    fn record_fusion_inputs(&self, transform_id: Uuid, p1: ObjectId, p2: ObjectId) {
        self.fusion_inputs
            .lock()
            .unwrap()
            .push((transform_id, p1, p2));
    }
}

impl Ecosystem for ProbeSpaceJungle {
    const NAME: &'static str = "space-jungle";
    type Animals = ProbeSpaceAnimals;
}

#[async_trait]
impl VoidInferOps for ProbeSpaceJungle {
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

    async fn start_model(&self, _model_id: Uuid) -> Result<(), String> {
        Err("model lifecycle is not used by TestCell".to_string())
    }

    async fn infer(&self, _model_id: Uuid, _request: InferenceRequest) -> Result<ObjectId, String> {
        Err("inference is not used by TestCell".to_string())
    }

    async fn perturb_up(&self, _model_id: Uuid, _seed: u64) -> Result<(), String> {
        Err("perturbation is not used by TestCell".to_string())
    }

    async fn perturb_down(&self, _model_id: Uuid) -> Result<(), String> {
        Err("perturbation is not used by TestCell".to_string())
    }

    async fn optimize(
        &self,
        _model_id: Uuid,
        _loss_up: f32,
        _loss_down: f32,
    ) -> Result<(), String> {
        Err("optimization is not used by TestCell".to_string())
    }

    async fn shutdown_model(&self, _model_id: Uuid) -> Result<(), String> {
        Err("model lifecycle is not used by TestCell".to_string())
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
impl SunOps for ProbeSpaceJungle {
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

#[cfg(test)]
async fn exercise_diamond_dog<A>(
    name: &str,
    vertex_count: usize,
    port_count: usize,
    epochs: usize,
) -> Vec<FusionObservation>
where
    A: Animal<Seed = (), State = SunState> + Observe<Appearance = SunAppearance>,
    A::Id: AnimalIdValue,
    A::Generation: jungle_sdk::typosaurus::num::Unsigned,
{
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let (void_addr, void_abort) = start_server().await;
    let mut jungle = ProbeSpaceJungle::new(void_addr);
    let potentiation_writes = Arc::clone(&jungle.potentiation_writes);
    let fusion_inputs = Arc::clone(&jungle.fusion_inputs);

    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");
    jungle.set_client(client.clone());

    let journey_id = client
        .spawn::<A>(&())
        .await
        .unwrap_or_else(|error| panic!("{name} should spawn: {error}"))
        .journey_id;
    println!("Spawned {name} journey: {journey_id}");

    let mut subscription = client
        .subscribe_step_updates(journey_id, None)
        .await
        .expect("subscribe_step_updates should succeed");

    let worker_handles: Vec<_> = (0..vertex_count + 1)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    let expected_potentiation_writes = epochs * port_count;
    let result = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            if potentiation_writes.load(Ordering::SeqCst) >= expected_potentiation_writes {
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
                            return Err(format!(
                                "step update stream ended before {epochs} epochs"
                            ));
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
        }
    })
    .await;

    match result {
        Ok(Ok(())) => println!("{name} completed {epochs} epochs"),
        Ok(Err(error)) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!("{name} flow assertion failed: {error}, status: {status:?}");
        }
        Err(error) => {
            let status = client
                .journey_details(journey_id)
                .await
                .expect("journey_details should succeed");
            panic!(
                "timeout waiting for {name} to complete {epochs} epochs (60s): {error}, \
                 potentiation writes: {}, status: {status:?}",
                potentiation_writes.load(Ordering::SeqCst),
            );
        }
    }

    let appearance = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(bytes) = client
                .animal_appearance(journey_id)
                .await
                .expect("animal_appearance should succeed")
            {
                let appearance = postcard::from_bytes::<SunAppearance>(&bytes)
                    .expect("Sun appearance should deserialize");
                if appearance.finalized && appearance.nodes.len() == vertex_count {
                    break appearance;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("finalized Sun appearance should become available");
    assert_eq!(appearance.nodes.len(), vertex_count);
    assert!(appearance.nodes.iter().all(|node| !node.label.is_empty()));
    assert!(
        appearance
            .nodes
            .iter()
            .all(|node| node.state != SunNodeState::Idle),
        "every node should expose an orchestration phase after completed epochs"
    );
    assert!(
        !appearance.edges.is_empty(),
        "the exercised Sun should expose its runtime topology"
    );

    let observed = fusion_inputs.lock().unwrap().clone();
    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_abort.abort();

    observed
}

/// Exercises a multi-epoch unary diamond feeding one binary fusion vertex.
///
/// The left and right unary branches stamp distinct emission IDs, with the P1
/// branch deliberately delayed so P2 arrives first. The explicit fusion
/// transform records its stable ID and each pair, proving that its identity and
/// declared `P1`, `P2` order remain stable on both propagation passes.
#[cfg(test)]
#[tokio::test]
async fn diamond_dog() {
    const EPOCHS: usize = 3;
    let observed = exercise_diamond_dog::<BlackHoleAnimal>("diamond_dog", 5, 6, EPOCHS).await;

    assert!(
        observed.len() >= EPOCHS * 2,
        "expected two fusion transforms per epoch, observed {observed:?}"
    );
    let expected_transform_id = observed[0].0;
    assert_ne!(
        expected_transform_id,
        Uuid::nil(),
        "fusion transform ID should be generated"
    );
    for epoch in 0..EPOCHS {
        for pass in 0..2 {
            let (transform_id, p1, p2) = observed[epoch * 2 + pass];
            assert_eq!(
                transform_id, expected_transform_id,
                "fusion transform ID changed in epoch {epoch} propagation pass {pass}"
            );
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
}

/// Exercises an extra diamond layer ending in a third binary fusion:
///
/// `Input -> [L0, R0]`, `L0 -> [L1, R1]`, `R0 -> [L2, R2]`,
/// `[L1, R1] -> F0`, `[L2, R2] -> F1`, and `[F0, F1] -> F2`.
#[cfg(test)]
#[tokio::test]
async fn sun_dog() {
    const EPOCHS: usize = 3;
    const PROPAGATION_PASSES: usize = 2;
    const FIRST_LAYER_FUSIONS: usize = 2;
    const FINAL_LAYER_FUSIONS: usize = 1;
    const FUSION_TRANSFORMS: usize =
        EPOCHS * PROPAGATION_PASSES * (FIRST_LAYER_FUSIONS + FINAL_LAYER_FUSIONS);

    // Ten vertices own thirteen input ports: seven unary and six binary.
    let observed = exercise_diamond_dog::<ExpandedBlackHoleAnimal>(
        "sun_dog",
        10,
        13,
        EPOCHS,
    )
    .await;
    assert!(
        observed.len() >= FUSION_TRANSFORMS,
        "expected {FUSION_TRANSFORMS} fusion transforms, observed {observed:?}"
    );

    let completed_epochs = &observed[..FUSION_TRANSFORMS];
    let first_layer_pair = (
        Uuid::from_u128(LEFT_EMISSION),
        Uuid::from_u128(RIGHT_EMISSION),
    );
    let final_layer_pair = (
        Uuid::from_u128(FUSED_EMISSION),
        Uuid::from_u128(FUSED_EMISSION),
    );
    assert!(
        completed_epochs
            .iter()
            .all(|(_, p1, p2)| (*p1, *p2) == first_layer_pair || (*p1, *p2) == final_layer_pair),
        "unexpected fusion inputs in completed epochs: {completed_epochs:?}"
    );
    assert_eq!(
        completed_epochs
            .iter()
            .filter(|(_, p1, p2)| (*p1, *p2) == first_layer_pair)
            .count(),
        EPOCHS * PROPAGATION_PASSES * FIRST_LAYER_FUSIONS,
        "both first-layer fusions should run on every pass"
    );
    assert_eq!(
        completed_epochs
            .iter()
            .filter(|(_, p1, p2)| (*p1, *p2) == final_layer_pair)
            .count(),
        EPOCHS * PROPAGATION_PASSES * FINAL_LAYER_FUSIONS,
        "the final fusion should run on every pass"
    );

    let mut inputs_by_transform = HashMap::new();
    for &(transform_id, p1, p2) in &observed {
        assert_ne!(
            transform_id,
            Uuid::nil(),
            "fusion transform ID should be generated"
        );
        if let Some(previous_inputs) = inputs_by_transform.insert(transform_id, (p1, p2)) {
            assert_eq!(
                previous_inputs,
                (p1, p2),
                "fusion transform {transform_id} did not retain a stable identity"
            );
        }
    }
    assert_eq!(
        inputs_by_transform.len(),
        FIRST_LAYER_FUSIONS + FINAL_LAYER_FUSIONS,
        "each fusion journey should have a distinct transform ID"
    );
    assert_eq!(
        inputs_by_transform
            .values()
            .filter(|inputs| **inputs == first_layer_pair)
            .count(),
        FIRST_LAYER_FUSIONS,
        "both first-layer transforms should have distinct stable IDs"
    );
    assert_eq!(
        inputs_by_transform
            .values()
            .filter(|inputs| **inputs == final_layer_pair)
            .count(),
        FINAL_LAYER_FUSIONS,
        "the final-layer transform should have its own stable ID"
    );
}

/// Runs the expanded diamond Sun indefinitely with a live Black Hole Beam viewer.
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

        let mut jungle = ProbeSpaceJungle::new(void_addr);
        let client = FusedClient::builder()
            .build()
            .await
            .expect("fused client should build");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<ExpandedBlackHoleAnimal>(&())
            .await
            .expect("ExpandedBlackHoleAnimal should spawn")
            .journey_id;
        println!("Spawned ExpandedBlackHoleAnimal journey: {journey_id}");

        // One worker per journey: ten graph vertices plus the parent.
        let _worker_handles: Vec<_> = (0..11)
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
        .dot_layout()
        .view_live::<ExpandedBlackHoleAnimal>(client, journey_id)
        .expect("Black Hole Beam should run");
}

/// Launches the Beam example in a process whose UI runs on its main thread.
#[cfg(test)]
fn run_beam_example(example: &str) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = std::process::Command::new(cargo);
    command.current_dir(env!("CARGO_MANIFEST_DIR")).args([
        "run",
        "--quiet",
        "--no-default-features",
    ]);

    // This test launches a second Cargo process so that the UI can own its main
    // thread. Cargo does not propagate the parent invocation's profile or
    // features to that process, so mirror the active build explicitly.
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }

    let features = [
        (cfg!(feature = "cuda"), "cuda"),
        (cfg!(feature = "metal"), "metal"),
        (cfg!(feature = "qwen35_0p8b"), "qwen35_0p8b"),
        (cfg!(feature = "qwen35_2b"), "qwen35_2b"),
        (cfg!(feature = "qwen35_4b"), "qwen35_4b"),
        (cfg!(feature = "qwen35_9b"), "qwen35_9b"),
        (cfg!(feature = "qwen35_27b"), "qwen35_27b"),
    ]
    .into_iter()
    .filter_map(|(enabled, feature)| enabled.then_some(feature))
    .collect::<Vec<_>>()
    .join(",");
    if !features.is_empty() {
        command.args(["--features", &features]);
    }

    let status = command
        .args(["--example", example])
        .status()
        .unwrap_or_else(|error| panic!("{example} example should launch: {error}"));

    assert!(status.success(), "{example} example exited with {status}");
}

#[cfg(test)]
#[test]
#[ignore]
fn beam_test() {
    run_beam_example("beam");
}

const PROGENITOR_NODE_COUNT: usize = 3;
pub(crate) const SPACE_PROBE_DISTANCE_PROMPT: &str =
        "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
const DARK_STAR_MODEL_CELL_COUNT: usize = 7;
const DARK_STAR_VERTEX_COUNT: usize = 10;
const DARK_STAR_PORT_COUNT: usize = 13;
const DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH: usize = 6;
const QWEN_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-0.8B";
static DARK_STAR_TOKENIZER: OnceLock<Result<tokenizers::Tokenizer, String>> = OnceLock::new();

type Unary0 = Unary<U0, Progenitor, list![U1]>;
type Unary1 = Unary<U1, Progenitor, list![U2]>;
type Unary2 = Unary<U2, Progenitor, list![]>;
type ThreeProgenitorSun = list![Unary0, Unary1, Unary2];

type DarkStarInput = Unary<U0, Progenitor, list![U1, U2]>;
type DarkStarL0 = Unary<U1, Progenitor, list![U3, U4]>;
type DarkStarR0 = Unary<U2, Progenitor, list![U5, U6]>;
type DarkStarL1 = Unary<U3, Progenitor, list![U7]>;
type DarkStarR1 = Unary<U4, Progenitor, list![U8]>;
type DarkStarL2 = Unary<U5, Progenitor, list![U9]>;
type DarkStarR2 = Unary<U6, Progenitor, list![U10]>;
type DarkStarF0 = Binary<U7, U8, ConcatFusionAnimal, list![U11]>;
type DarkStarF1 = Binary<U9, U10, ConcatFusionAnimal, list![U12]>;
type DarkStarF2 = Binary<U11, U12, ConcatFusionAnimal, list![]>;
type DarkStarSun = list![
    DarkStarInput,
    DarkStarL0,
    DarkStarR0,
    DarkStarL1,
    DarkStarR1,
    DarkStarL2,
    DarkStarR2,
    DarkStarF0,
    DarkStarF1,
    DarkStarF2
];

pub(crate) fn dark_star_tokenizer() -> Result<&'static tokenizers::Tokenizer, String> {
    let tokenizer_result = DARK_STAR_TOKENIZER.get_or_init(|| {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|error| format!("failed to create hf hub api: {error}"))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            QWEN_TOKENIZER_REPO.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_file = repo.get("tokenizer.json").map_err(|error| {
            format!("failed to download tokenizer.json from HuggingFace: {error}")
        })?;
        tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|error| format!("failed to load tokenizer.json: {error}"))
    });
    match tokenizer_result {
        Ok(tokenizer) => Ok(tokenizer),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn prompt_to_dark_tokens(
    prompt: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<DarkToken>, String> {
    let tokens = tokenizer
        .encode(prompt, false)
        .map_err(|error| format!("failed to tokenize prompt: {error}"))?;

    Ok(tokens
        .get_ids()
        .iter()
        .map(|&id| {
            let token_id = id as u32;
            DarkToken {
                predicted: token_id,
                dark_knowledge: vec![LogitEntry {
                    token_id,
                    log_prob: 0.0,
                }],
            }
        })
        .collect())
}

#[derive(Flow)]
pub struct DarkStarGenerator(Step<GenerateDarkStarPrompt>);

pub struct GenerateDarkStarPrompt;

pub struct GenerateDarkStarPromptEffect;

#[derive(Flow)]
pub struct DarkStarPolicy(Step<DarkStarLossPolicy>);

pub struct DarkStarLossPolicy;

pub struct DarkStarLossPolicyEffect;

pub(crate) trait FusionConcatOps: Send + Sync {
    fn record_fusion_concat(&self);
}

pub struct ConcatFusionOutputs;

pub struct ConcatFusionOutputsEffect;

#[derive(Flow)]
pub struct ConcatFusionTransform(Step<ConcatFusionOutputs>);

pub struct ConcatFusionAnimal;

#[jungle::animal(id = 2, generation = 0)]
impl Animal for ConcatFusionAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = Fusion<ConcatFusionTransform>;
}

pub struct ProgenitorBlackHole;

#[jungle::animal(observe, id = 1, generation = 0)]
impl Animal for ProgenitorBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<Generator, Policy>;
}

impl Observe for ProgenitorBlackHole {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

pub struct DarkStarBlackHole;

#[jungle::animal(observe, id = 3, generation = 0)]
impl Animal for DarkStarBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <DarkStarSun as BlackHole>::Sun<DarkStarGenerator, DarkStarPolicy>;
}

impl Observe for DarkStarBlackHole {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

pub struct BlackDwarfBlackHole;

#[jungle::animal(observe, id = 4, generation = 0)]
impl Animal for BlackDwarfBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<DarkStarGenerator, DarkStarPolicy>;
}

impl Observe for BlackDwarfBlackHole {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

#[derive(Animals)]
pub struct SpaceAnimals(
    Progenitor,
    ProgenitorBlackHole,
    ConcatFusionAnimal,
    DarkStarBlackHole,
    BlackDwarfBlackHole,
);

#[derive(Clone)]
pub struct SpaceJungle {
    void_addr: SocketAddr,
    quark_addr: SocketAddr,
    client: Option<FusedClient>,
    potentiation_writes: Arc<AtomicUsize>,
    inference_calls: Arc<AtomicUsize>,
    optimized_cells: Arc<AtomicUsize>,
    fusion_concat_calls: Arc<AtomicUsize>,
    model_error: Arc<Mutex<Option<String>>>,
}

impl SpaceJungle {
    fn new(void_addr: SocketAddr, quark_addr: SocketAddr, _model_cell_count: usize) -> Self {
        Self {
            void_addr,
            quark_addr,
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            inference_calls: Arc::new(AtomicUsize::new(0)),
            optimized_cells: Arc::new(AtomicUsize::new(0)),
            fusion_concat_calls: Arc::new(AtomicUsize::new(0)),
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

impl FusionConcatOps for SpaceJungle {
    fn record_fusion_concat(&self) {
        self.fusion_concat_calls.fetch_add(1, Ordering::SeqCst);
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

    async fn start_model(&self, model_id: Uuid) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        let result = quark_start_result(&endpoint, self.quark_addr, model_id).await;
        self.record_model_error("start model", &result);
        result
    }

    async fn infer(&self, model_id: Uuid, request: InferenceRequest) -> Result<ObjectId, String> {
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
        let result = quark_infer_result(&endpoint, self.quark_addr, model_id, request_id).await;
        self.record_model_error("infer", &result);
        if result.is_ok() {
            self.inference_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn perturb_up(&self, model_id: Uuid, seed: u64) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        let result = quark_perturb_up_result(&endpoint, self.quark_addr, model_id, seed).await;
        self.record_model_error("perturb up", &result);
        result
    }

    async fn perturb_down(&self, model_id: Uuid) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        let result = quark_perturb_down_result(&endpoint, self.quark_addr, model_id).await;
        self.record_model_error("perturb down", &result);
        result
    }

    async fn optimize(&self, model_id: Uuid, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        let result =
            quark_optimize_result(&endpoint, self.quark_addr, model_id, loss_up, loss_down).await;
        self.record_model_error("optimize", &result);
        if result.is_ok() {
            self.optimized_cells.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn shutdown_model(&self, model_id: Uuid) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        let result = quark_shutdown_result(&endpoint, self.quark_addr, model_id).await;
        self.record_model_error("shutdown model", &result);
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

async fn exercise_diamond_dog_epoch<A>(
    test_name: &str,
    model_path: &str,
    model_cell_count: usize,
    vertex_count: usize,
    expected_potentiation_writes: usize,
    expected_fusion_concats: usize,
) where
    A: Animal<Seed = ()>,
    A::Id: AnimalIdValue,
    A::Generation: jungle_sdk::typosaurus::num::Unsigned,
{
    let (void_addr, void_abort, quark_addr, quark_abort) = start_servers(model_path).await;

    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");
    let mut jungle = SpaceJungle::new(void_addr, quark_addr, model_cell_count);
    jungle.set_client(client.clone());

    let potentiation_writes = Arc::clone(&jungle.potentiation_writes);
    let inference_calls = Arc::clone(&jungle.inference_calls);
    let optimized_cells = Arc::clone(&jungle.optimized_cells);
    let fusion_concat_calls = Arc::clone(&jungle.fusion_concat_calls);
    let model_error = Arc::clone(&jungle.model_error);

    let parent = client
        .spawn::<A>(&())
        .await
        .unwrap_or_else(|error| panic!("{test_name} should spawn: {error}"));
    let mut subscription = client
        .subscribe_step_updates(parent.journey_id, None)
        .await
        .expect("parent subscription should succeed");

    let worker_handles: Vec<_> = (0..vertex_count + 1)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    let result = tokio::time::timeout(Duration::from_secs(240), async {
            loop {
                if let Some(error) = model_error.lock().unwrap().clone() {
                    return Err(error);
                }

                if potentiation_writes.load(Ordering::SeqCst) >= expected_potentiation_writes
                    && inference_calls.load(Ordering::SeqCst) >= model_cell_count * 2
                    && optimized_cells.load(Ordering::SeqCst) >= model_cell_count
                    && fusion_concat_calls.load(Ordering::SeqCst) >= expected_fusion_concats
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
                                return Err(format!("step update stream ended before {test_name} completed an epoch"));
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                }
            }
        })
        .await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let status = client
                .journey_details(parent.journey_id)
                .await
                .expect("parent journey details should be available");
            panic!(
                    "{test_name} failed: {error}; inferences={}, potentiations={}, optimized_cells={}, fusion_concats={}, status={status:?}",
                    inference_calls.load(Ordering::SeqCst),
                    potentiation_writes.load(Ordering::SeqCst),
                    optimized_cells.load(Ordering::SeqCst),
                    fusion_concat_calls.load(Ordering::SeqCst),
                );
        }
        Err(error) => {
            let status = client
                .journey_details(parent.journey_id)
                .await
                .expect("parent journey details should be available");
            panic!(
                    "timeout waiting for {test_name} epoch (240s): {error}; inferences={}, potentiations={}, optimized_cells={}, fusion_concats={}, status={status:?}",
                    inference_calls.load(Ordering::SeqCst),
                    potentiation_writes.load(Ordering::SeqCst),
                    optimized_cells.load(Ordering::SeqCst),
                    fusion_concat_calls.load(Ordering::SeqCst),
                );
        }
    }

    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_abort.abort();
    quark_abort.abort();
}

/// Runs the same U0 -> U1 -> U2 Sun topology as `diamond_dog`, with real Progenitor
/// cells backed by a quark model.
#[ignore]
#[tokio::test]
async fn primordia() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("primordia") {
        Some(path) => path,
        None => return,
    };
    exercise_diamond_dog_epoch::<ProgenitorBlackHole>(
        "Progenitor Sun",
        &model_path,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        0,
    )
    .await;
}

/// Runs an expanded diamond with Fusion nodes that concatenate outputs.
#[ignore]
#[tokio::test]
async fn dark_star() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("dark_star") {
        Some(path) => path,
        None => return,
    };

    exercise_diamond_dog_epoch::<DarkStarBlackHole>(
        "dark_star Sun",
        &model_path,
        DARK_STAR_MODEL_CELL_COUNT,
        DARK_STAR_VERTEX_COUNT,
        DARK_STAR_PORT_COUNT,
        DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH,
    )
    .await;
}

/// Runs the same topology as `primordia`, but with dark_star's generator/policy.
#[ignore]
#[tokio::test]
async fn black_dwarf() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("black_dwarf") {
        Some(path) => path,
        None => return,
    };

    exercise_diamond_dog_epoch::<BlackDwarfBlackHole>(
        "black_dwarf Sun",
        &model_path,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        0,
    )
    .await;
}

/// Runs the dark_star Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam_dark_star() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to run beam_dark_star");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    let (client, journey_id) = runtime.block_on(async {
        let (void_addr, _void_abort, quark_addr, _quark_abort) = start_servers(&model_path).await;

        let mut jungle = SpaceJungle::new(void_addr, quark_addr, DARK_STAR_MODEL_CELL_COUNT);
        let client = FusedClient::builder()
            .build()
            .await
            .expect("fused client should build");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<DarkStarBlackHole>(&())
            .await
            .expect("DarkStarBlackHole should spawn")
            .journey_id;
        println!("Spawned DarkStarBlackHole journey: {journey_id}");

        // One worker per journey: dark_star graph vertices plus the parent.
        let _worker_handles: Vec<_> = (0..(DARK_STAR_VERTEX_COUNT + 1))
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
        .view_live::<DarkStarBlackHole>(client, journey_id)
        .expect("Black Hole Beam should run");
}

/// Runs the black_dwarf Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam_black_dwarf() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to run beam_black_dwarf");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    let (client, journey_id) = runtime.block_on(async {
        let (void_addr, _void_abort, quark_addr, _quark_abort) = start_servers(&model_path).await;

        let mut jungle = SpaceJungle::new(void_addr, quark_addr, PROGENITOR_NODE_COUNT);
        let client = FusedClient::builder()
            .build()
            .await
            .expect("fused client should build");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<BlackDwarfBlackHole>(&())
            .await
            .expect("BlackDwarfBlackHole should spawn")
            .journey_id;
        println!("Spawned BlackDwarfBlackHole journey: {journey_id}");

        // One worker per journey: black_dwarf graph vertices plus the parent.
        let _worker_handles: Vec<_> = (0..(PROGENITOR_NODE_COUNT + 1))
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
        .dot_layout()
        .view_live::<BlackDwarfBlackHole>(client, journey_id)
        .expect("Black Hole Beam should run");
}

/// Launches the Dark Star Beam example in a process whose UI runs on its main thread.
#[test]
#[ignore]
fn beam_dark_star() {
    run_beam_example("beam_dark_star");
}

/// Launches the Black Dwarf Beam example in a process whose UI runs on its main thread.
#[test]
#[ignore]
fn beam_black_dwarf() {
    run_beam_example("beam_black_dwarf");
}
