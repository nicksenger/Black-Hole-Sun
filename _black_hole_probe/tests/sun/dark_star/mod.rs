mod action;
mod effect;

#[cfg(test)]
use futures::stream::StreamExt;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::ops::{SunOps, VoidInferOps};
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::sun::{Binary, BlackHole, SunAppearance, SunState, Unary};
use black_hole_sun::{
    DarkToken, EmissionId, InferenceRequest, LogitEntry, ObjectId, QuarkClient, QuarkServerBuilder,
    Tokenizer, Transmission, VoidClient, VoidServerBuilder,
};
use black_hole_sun::{Fusion, FusionSeed, FusionState, Progenitor};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::*;

pub(super) const PROGENITOR_NODE_COUNT: usize = 3;
const DARK_STAR_MODEL_CELL_COUNT: usize = 7;
const DARK_STAR_VERTEX_COUNT: usize = 10;
const DARK_STAR_PORT_COUNT: usize = 13;
const DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH: usize = 6;
static DARK_STAR_TOKENIZER: OnceLock<Result<Tokenizer, String>> = OnceLock::new();

pub(super) const SPACE_PROBE_DISTANCE_PROMPT: &str = "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";

pub(super) type ThreeUnary0 = Unary<U0, Progenitor, list![U1]>;
pub(super) type ThreeUnary1 = Unary<U1, Progenitor, list![U2]>;
pub(super) type ThreeUnary2 = Unary<U2, Progenitor, list![]>;
pub(super) type ThreeProgenitorSun = list![ThreeUnary0, ThreeUnary1, ThreeUnary2];

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

pub(super) fn dark_star_tokenizer() -> Result<&'static Tokenizer, String> {
    let tokenizer_result = DARK_STAR_TOKENIZER.get_or_init(Tokenizer::try_init);
    match tokenizer_result {
        Ok(tokenizer) => Ok(tokenizer),
        Err(error) => Err(error.clone()),
    }
}

pub(super) fn prompt_to_dark_tokens(
    prompt: &str,
    tokenizer: &Tokenizer,
) -> Result<Vec<DarkToken>, String> {
    let tokens = tokenizer
        .encode_ids(prompt)
        .map_err(|error| format!("failed to tokenize prompt: {error}"))?;

    Ok(tokens
        .iter()
        .map(|&token_id| DarkToken {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
            }],
        })
        .collect())
}

#[derive(Flow)]
pub(super) struct DarkStarGenerator(Step<GenerateDarkStarPrompt>);

pub(super) struct GenerateDarkStarPrompt;
pub struct GenerateDarkStarPromptEffect;

#[derive(Flow)]
pub(super) struct DarkStarPolicy(Step<DarkStarLossPolicy>);

pub(super) struct DarkStarLossPolicy;
pub struct DarkStarLossPolicyEffect;

pub(super) trait FusionConcatOps: Send + Sync {
    fn record_fusion_concat(&self);
}

pub(super) struct ConcatFusionOutputs;
pub struct ConcatFusionOutputsEffect;

#[derive(Flow)]
pub(super) struct ConcatFusionTransform(Step<ConcatFusionOutputs>);

pub(super) struct ConcatFusionAnimal;

#[jungle::animal(id = 2, generation = 0)]
impl Animal for ConcatFusionAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = Fusion<ConcatFusionTransform>;
}

pub(super) struct ProgenitorBlackHole;

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

pub(super) struct DarkStarBlackHole;

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

pub(super) struct BlackDwarfBlackHole;

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
pub(super) struct SpaceAnimals(
    Progenitor,
    ProgenitorBlackHole,
    ConcatFusionAnimal,
    DarkStarBlackHole,
    BlackDwarfBlackHole,
);

#[derive(Clone)]
pub(super) struct SpaceJungle {
    void_client: VoidClient,
    quark_client: QuarkClient,
    client: Option<FusedClient>,
    pub(super) potentiation_writes: Arc<AtomicUsize>,
    pub(super) inference_calls: Arc<AtomicUsize>,
    pub(super) optimized_cells: Arc<AtomicUsize>,
    pub(super) fusion_concat_calls: Arc<AtomicUsize>,
    pub(super) model_error: Arc<Mutex<Option<String>>>,
}

impl SpaceJungle {
    pub(super) fn new(
        void_client: VoidClient,
        quark_client: QuarkClient,
        _model_cell_count: usize,
    ) -> Self {
        Self {
            void_client,
            quark_client,
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            inference_calls: Arc::new(AtomicUsize::new(0)),
            optimized_cells: Arc::new(AtomicUsize::new(0)),
            fusion_concat_calls: Arc::new(AtomicUsize::new(0)),
            model_error: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) fn set_client(&mut self, client: FusedClient) {
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
        self.void_client.download(id).await
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        Ok(self.void_client.upload(data).await.unwrap())
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        let is_potentiation = matches!(
            postcard::from_bytes(&data),
            Ok(Transmission::Potentiation { .. })
        );
        self.void_client.upload_with(id, data).await.unwrap();
        if is_potentiation {
            self.potentiation_writes.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    async fn start_model(&self, model_id: Uuid) -> Result<(), String> {
        let result = self.quark_client.start(model_id).await;
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
        let request_bytes = postcard::to_allocvec(&request).map_err(|error| error.to_string())?;
        let request_id = self.void_client.upload(request_bytes).await.unwrap();
        let result = self.quark_client.infer(model_id, request_id).await;
        self.record_model_error("infer", &result);
        if result.is_ok() {
            self.inference_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn perturb_up(&self, model_id: Uuid, seed: u64) -> Result<(), String> {
        let result = self.quark_client.perturb_up(model_id, seed).await;
        self.record_model_error("perturb up", &result);
        result
    }

    async fn perturb_down(&self, model_id: Uuid) -> Result<(), String> {
        let result = self.quark_client.perturb_down(model_id).await;
        self.record_model_error("perturb down", &result);
        result
    }

    async fn optimize(&self, model_id: Uuid, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let result = self.quark_client.optimize(model_id, loss_up, loss_down).await;
        self.record_model_error("optimize", &result);
        if result.is_ok() {
            self.optimized_cells.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn shutdown_model(&self, model_id: Uuid) -> Result<(), String> {
        let result = self.quark_client.shutdown(model_id).await;
        self.record_model_error("shutdown model", &result);
        result
    }

    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String> {
        let propagation = Transmission::Propagation {
            emission_id,
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        };
        let data = postcard::to_allocvec(&propagation).map_err(|error| error.to_string())?;
        self.void_client.upload_with(send_id, data).await.unwrap();
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

pub(super) async fn start_servers(
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

#[cfg(test)]
pub(super) async fn exercise_epoch<A>(
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
    let endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
    let quark_client = QuarkClient::new(&endpoint, quark_addr, "localhost");
    let mut jungle = SpaceJungle::new(void_client, quark_client, model_cell_count);
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
                            return Err(format!(
                                "step update stream ended before {test_name} completed an epoch"
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

/// Runs an expanded diamond with Fusion nodes that concatenate outputs.
#[cfg(test)]
#[ignore]
#[tokio::test]
async fn dark_star() {
    init_tracing();

    let model_path = match require_model_path("dark_star") {
        Some(path) => path,
        None => return,
    };

    exercise_epoch::<DarkStarBlackHole>(
        "dark_star Sun",
        &model_path,
        DARK_STAR_MODEL_CELL_COUNT,
        DARK_STAR_VERTEX_COUNT,
        DARK_STAR_PORT_COUNT,
        DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH,
    )
    .await;
}

/// Runs the dark_star Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam_dark_star() {
    init_tracing();

    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to run beam_dark_star");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    let (client, journey_id) = runtime.block_on(async {
        let (void_addr, _void_abort, quark_addr, _quark_abort) = start_servers(&model_path).await;

        let endpoint = make_client_endpoint().await;
        let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
        let quark_client = QuarkClient::new(&endpoint, quark_addr, "localhost");
        let mut jungle = SpaceJungle::new(void_client, quark_client, DARK_STAR_MODEL_CELL_COUNT);
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

#[cfg(test)]
#[test]
#[ignore]
fn beam_dark_star() {
    super::run_beam_example("beam_dark_star");
}
