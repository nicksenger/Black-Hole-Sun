mod action;
mod effect;

#[cfg(test)]
use futures::stream::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::cell::action::{
    CellState, InitRecvId, Potentiation, Transmit, WaitForPotentiationAction,
    WaitForPropagationAction,
};
use black_hole_sun::ops::{SunOps, VoidInferOps};
use black_hole_sun::sun::{Binary, BlackHole, SunAppearance, SunState, Unary};
use black_hole_sun::{
    EmissionId, InferenceRequest, ObjectId, QuarkClient, TestQuarkServer, TestVoidServer,
    Tokenizer, Transmission, VoidClient,
};
use black_hole_sun::{FusionSeed, FusionState, Progenitor, QuzoFusion};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::*;

pub(super) const PROGENITOR_NODE_COUNT: usize = 3;
const DARK_STAR_MODEL_NODE_COUNT: usize = 3;
const DARK_STAR_VERTEX_COUNT: usize = 10;
const DARK_STAR_PORT_COUNT: usize = 13;
const DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH: usize = 6;

pub(super) const SPACE_PROBE_DISTANCE_PROMPT: &str = "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";

pub(super) type ThreeUnary0 = Unary<U0, Progenitor, list![U1]>;
pub(super) type ThreeUnary1 = Unary<U1, Progenitor, list![U2]>;
pub(super) type ThreeUnary2 = Unary<U2, Progenitor, list![]>;
pub(super) type ThreeProgenitorSun = list![ThreeUnary0, ThreeUnary1, ThreeUnary2];

pub(super) struct FinishDarkStarTestCellEpoch;

#[derive(Flow)]
pub(super) struct DarkStarTestCellEpoch<Transform>(
    Step<WaitForPropagationAction>,
    Transform,
    Step<Transmit>,
    Step<WaitForPropagationAction>,
    Transform,
    Step<Transmit>,
    Step<WaitForPotentiationAction>,
    Step<FinishDarkStarTestCellEpoch>,
);

pub(super) struct AlwaysDarkStarEpoch;

impl Predicate<(&CellState, &())> for AlwaysDarkStarEpoch {
    fn eval(_input: &(&CellState, &())) -> bool {
        true
    }
}

#[derive(Flow)]
pub(super) struct DarkStarTestCellFlow<Transform>(
    Step<InitRecvId>,
    While<AlwaysDarkStarEpoch, DarkStarTestCellEpoch<Transform>>,
);

pub(super) struct PassDarkStarEmission;

#[jungle::action]
impl Action for FinishDarkStarTestCellEpoch {
    type Effect = NoEffect;
    type Input = Potentiation;
    type Output = ();

    fn emit(_state: &CellState, _input: Self::Input) {}

    fn absorb(
        _state: &mut CellState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("finish dark_star test-cell epoch failed".to_string()))
    }
}

#[jungle::action(carry = EmissionId)]
impl Action for PassDarkStarEmission {
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
        output.map_err(|_| Failure::Message("pass dark_star emission failed".to_string()))?;
        Ok(emission_id)
    }
}

pub(super) struct DarkStarTestCell;

#[jungle::animal(id = 7, generation = 0)]
impl Animal for DarkStarTestCell {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = DarkStarTestCellFlow<Step<PassDarkStarEmission>>;
}

type DarkStarInput = Unary<U0, DarkStarTestCell, list![U1, U2]>;
type DarkStarL0 = Unary<U1, DarkStarTestCell, list![U3, U4]>;
type DarkStarR0 = Unary<U2, DarkStarTestCell, list![U5, U6]>;
type DarkStarL1 = Unary<U3, DarkStarTestCell, list![U7]>;
type DarkStarR1 = Unary<U4, DarkStarTestCell, list![U8]>;
type DarkStarL2 = Unary<U5, DarkStarTestCell, list![U9]>;
type DarkStarR2 = Unary<U6, DarkStarTestCell, list![U10]>;
type DarkStarF0 = Binary<U7, U8, LeftStackTwinAnimal, list![U11]>;
type DarkStarF1 = Binary<U9, U10, RightStackTwinAnimal, list![U12]>;
type DarkStarF2 = Binary<U11, U12, RandStackTwinAnimal, list![]>;
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

#[derive(Flow)]
pub(super) struct DarkStarGenerator(Step<GenerateDarkStarPrompt>);

pub(super) struct GenerateDarkStarPrompt;
pub struct GenerateDarkStarPromptEffect;

#[derive(Flow)]
pub(super) struct BlackDwarfGenerator(Step<GenerateBlackDwarfPrompt>);

pub(super) struct GenerateBlackDwarfPrompt;
pub struct GenerateBlackDwarfPromptEffect;

#[derive(Flow)]
pub(super) struct DarkStarPolicy(Step<DarkStarLossPolicy>);

pub(super) struct DarkStarLossPolicy;
pub struct DarkStarLossPolicyEffect;

#[derive(Flow)]
pub(super) struct BlackDwarfPolicy(Step<BlackDwarfLossPolicy>);

pub(super) struct BlackDwarfLossPolicy;
pub struct BlackDwarfLossPolicyEffect;

pub(super) trait FusionConcatOps: Send + Sync {
    fn record_fusion_concat(&self);
}

pub(super) struct ConcatFusionOutputs;
pub struct ConcatFusionOutputsEffect;

#[derive(Flow)]
pub(super) struct ConcatFusionTransform(Step<ConcatFusionOutputs>);

pub(super) struct LeftStackTwinOutputs;
pub struct LeftStackTwinOutputsEffect;

#[jungle::animal(id = 2, generation = 0)]
impl Animal for LeftStackTwinAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = QuzoFusion<LeftStackTwinTransform, ()>;
}

#[derive(Flow)]
pub(super) struct LeftStackTwinTransform(Step<LeftStackTwinOutputs>);

pub(super) struct LeftStackTwinAnimal;

pub(super) struct RightStackTwinOutputs;
pub struct RightStackTwinOutputsEffect;

#[derive(Flow)]
pub(super) struct RightStackTwinTransform(Step<RightStackTwinOutputs>);

pub(super) struct RightStackTwinAnimal;

#[jungle::animal(id = 5, generation = 0)]
impl Animal for RightStackTwinAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = QuzoFusion<RightStackTwinTransform, ()>;
}

pub(super) struct RandStackTwinOutputs;
pub struct RandStackTwinOutputsEffect;

#[derive(Flow)]
pub(super) struct RandStackTwinTransform(Step<RandStackTwinOutputs>);

pub(super) struct RandStackTwinAnimal;

#[jungle::animal(id = 6, generation = 0)]
impl Animal for RandStackTwinAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = QuzoFusion<RandStackTwinTransform, ()>;
}

pub(super) struct ProgenitorBlackHole;

#[jungle::animal(observe, id = 1, generation = 0)]
impl Animal for ProgenitorBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<Generator, Policy, ()>;
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
    type Flow = <DarkStarSun as BlackHole>::Sun<DarkStarGenerator, DarkStarPolicy, ()>;
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
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<BlackDwarfGenerator, BlackDwarfPolicy, ()>;
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
    DarkStarTestCell,
    ProgenitorBlackHole,
    LeftStackTwinAnimal,
    RightStackTwinAnimal,
    RandStackTwinAnimal,
    DarkStarBlackHole,
    BlackDwarfBlackHole,
);

#[derive(Clone)]
pub(super) struct SpaceJungle {
    void_client: VoidClient,
    quark_client: QuarkClient,
    tokenizer: Arc<OnceLock<Result<Tokenizer, String>>>,
    client: Option<FusedClient>,
    pub(super) potentiation_writes: Arc<AtomicUsize>,
    pub(super) inference_calls: Arc<AtomicUsize>,
    pub(super) perturb_up_calls: Arc<AtomicUsize>,
    pub(super) perturb_down_calls: Arc<AtomicUsize>,
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
            tokenizer: Arc::new(OnceLock::new()),
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            inference_calls: Arc::new(AtomicUsize::new(0)),
            perturb_up_calls: Arc::new(AtomicUsize::new(0)),
            perturb_down_calls: Arc::new(AtomicUsize::new(0)),
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
        let request_bytes = postcard::to_allocvec(&request).map_err(|error| error.to_string())?;
        let request_id = self.void_client.upload(request_bytes).await.unwrap();
        let result = self.quark_client.infer(model_id, request_id).await;
        self.record_model_error("infer", &result);
        if result.is_ok() {
            self.inference_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
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

    async fn perturb_up(&self, model_id: Uuid, seed: u64) -> Result<(), String> {
        let result = self.quark_client.perturb_up(model_id, seed).await;
        self.record_model_error("perturb up", &result);
        if result.is_ok() {
            self.perturb_up_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn perturb_down(&self, model_id: Uuid) -> Result<(), String> {
        let result = self.quark_client.perturb_down(model_id).await;
        self.record_model_error("perturb down", &result);
        if result.is_ok() {
            self.perturb_down_calls.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    async fn optimize(&self, model_id: Uuid, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let result = self
            .quark_client
            .optimize(model_id, loss_up, loss_down)
            .await;
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

#[cfg(test)]
pub(super) async fn exercise_epoch<A>(
    test_name: &str,
    model_path: &str,
    model_cell_count: usize,
    vertex_count: usize,
    expected_potentiation_writes: usize,
    expected_fusion_concats: usize,
    quark_default_inference_limit: Option<u32>,
) where
    A: Animal<Seed = ()>,
    A::Id: AnimalIdValue,
    A::Generation: jungle_sdk::typosaurus::num::Unsigned,
{
    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let mut quark_builder = TestQuarkServer::new(model_path).void_addr(void_server.local_addr());
    if let Some(limit) = quark_default_inference_limit {
        quark_builder = quark_builder.default_inference_limit(limit);
    }
    let quark_server = quark_builder
        .serve()
        .await
        .expect("failed to start quark server");
    let void_addr = void_server.local_addr();
    let quark_addr = quark_server.local_addr();

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
    let perturb_up_calls = Arc::clone(&jungle.perturb_up_calls);
    let perturb_down_calls = Arc::clone(&jungle.perturb_down_calls);
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
                && perturb_up_calls.load(Ordering::SeqCst) >= model_cell_count
                && perturb_down_calls.load(Ordering::SeqCst) >= model_cell_count
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
                "{test_name} failed: {error}; inferences={}, perturb_up={}, perturb_down={}, potentiations={}, optimized_cells={}, fusion_concats={}, status={status:?}",
                inference_calls.load(Ordering::SeqCst),
                perturb_up_calls.load(Ordering::SeqCst),
                perturb_down_calls.load(Ordering::SeqCst),
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
                "timeout waiting for {test_name} epoch (240s): {error}; inferences={}, perturb_up={}, perturb_down={}, potentiations={}, optimized_cells={}, fusion_concats={}, status={status:?}",
                inference_calls.load(Ordering::SeqCst),
                perturb_up_calls.load(Ordering::SeqCst),
                perturb_down_calls.load(Ordering::SeqCst),
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
    void_server.abort();
    quark_server.abort();
}

/// Runs an expanded diamond with Fusion nodes using Twin stack transforms.
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
        DARK_STAR_MODEL_NODE_COUNT,
        DARK_STAR_VERTEX_COUNT,
        DARK_STAR_PORT_COUNT,
        DARK_STAR_FUSION_TRANSFORMS_PER_EPOCH,
        None,
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
    let (client, journey_id, _void_server, _quark_server) = runtime.block_on(async {
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

        let endpoint = make_client_endpoint().await;
        let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
        let quark_client = QuarkClient::new(&endpoint, quark_addr, "localhost");
        let mut jungle = SpaceJungle::new(void_client, quark_client, DARK_STAR_MODEL_NODE_COUNT);
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

        (client, journey_id, void_server, quark_server)
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
