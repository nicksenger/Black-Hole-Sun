mod common;

use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::{Binary, BlackHole, SunState, Unary};
use black_hole_flux::{AtomError, Emission, Fusion, FusionSeed, FusionState, Progenitor};
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{
    DarkToken, EmissionId, InferenceOutput, InferenceOutputId, LogitEntry, ObjectId,
    QuarkServerBuilder, SequenceOutput, Transmission, VoidServerBuilder,
    InferenceRequest,
};
#[cfg(test)]
use futures::stream::StreamExt;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::{from_bytes, to_allocvec};
use typosaurus::num::consts::*;
use uuid::Uuid;

use common::*;

const PROGENITOR_NODE_COUNT: usize = 3;
const SPACE_PROBE_DISTANCE_PROMPT: &str =
    "A space probe in a decaying orbit measures its distance to the event horizon of a black hole. At point A, it is 3,600 kilometers away. Strong gravitational attraction pulls the probe inward, closing 2/3 of its initial distance. Orbital decay then pulls the probe another 450 kilometers closer to the event horizon. How many kilometers is the probe from the event horizon now?";
const DARK_STAR_MODEL_CELL_COUNT: usize = 7;
const DARK_STAR_VERTEX_COUNT: usize = 10;
#[cfg(test)]
const DARK_STAR_PORT_COUNT: usize = 13;
#[cfg(test)]
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

fn dark_star_tokenizer() -> Result<&'static tokenizers::Tokenizer, String> {
    let tokenizer_result = DARK_STAR_TOKENIZER.get_or_init(|| {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|error| format!("failed to create hf hub api: {error}"))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            QWEN_TOKENIZER_REPO.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_file = repo
            .get("tokenizer.json")
            .map_err(|error| format!("failed to download tokenizer.json from HuggingFace: {error}"))?;
        tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(|error| format!("failed to load tokenizer.json: {error}"))
    });
    match tokenizer_result {
        Ok(tokenizer) => Ok(tokenizer),
        Err(error) => Err(error.clone()),
    }
}

fn prompt_to_dark_tokens(
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

#[jungle::action]
impl Action for GenerateDarkStarPrompt {
    type Effect = GenerateDarkStarPromptEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("dark star generator failed: {error}")))
    }
}

pub struct GenerateDarkStarPromptEffect;

impl<J> EffectSchema<J> for GenerateDarkStarPromptEffect {
    type Id = u64;
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;
}

impl<J> Effect<J> for GenerateDarkStarPromptEffect
where
    J: VoidInferOps,
{
    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let tokenizer = dark_star_tokenizer().map_err(AtomError::Inference)?;
            let dark_tokens = prompt_to_dark_tokens(SPACE_PROBE_DISTANCE_PROMPT, tokenizer)
                .map_err(AtomError::Inference)?;
            let output = InferenceOutput {
                results: vec![SequenceOutput(dark_tokens)],
            };
            let output_bytes = to_allocvec(&output)?;
            let output_id = jungle
                .upload_to_void(output_bytes)
                .await
                .map_err(AtomError::Upload)?;
            let emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(output_id),
            };
            let emission_bytes = to_allocvec(&emission)?;
            let emission_id = jungle
                .upload_to_void(emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            let propagation = Transmission::Propagation {
                emission_id: EmissionId(emission_id),
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            };
            Ok((propagation.clone(), propagation))
        }
    }
}

#[derive(Flow)]
pub struct DarkStarPolicy(Step<DarkStarLossPolicy>);

pub struct DarkStarLossPolicy;

#[jungle::action]
impl Action for DarkStarLossPolicy {
    type Effect = DarkStarLossPolicyEffect;
    type Input = (Transmission, Transmission);
    type Output = (f32, f32);
    type Carry = ();

    fn emit(_state: &SunState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("dark star policy failed: {error}")))
    }
}

pub struct DarkStarLossPolicyEffect;

#[jungle::effect]
impl<J> Effect<J> for DarkStarLossPolicyEffect {
    type Id = u64;
    type In = (Transmission, Transmission);
    type Out = (f32, f32);
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move { Ok((0.4, 0.8)) }
    }
}

trait FusionConcatOps: Send + Sync {
    fn record_fusion_concat(&self);
}

pub struct ConcatFusionOutputs;

#[jungle::action]
impl Action for ConcatFusionOutputs {
    type Effect = ConcatFusionOutputsEffect;
    type Input = (Uuid, (EmissionId, EmissionId));
    type Output = EmissionId;

    fn emit(_state: &FusionState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut FusionState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("fusion concatenation failed: {error}")))
    }
}

pub struct ConcatFusionOutputsEffect;

impl<J> EffectSchema<J> for ConcatFusionOutputsEffect {
    type Id = u64;
    type In = (Uuid, (EmissionId, EmissionId));
    type Out = EmissionId;
    type Err = AtomError;
}

impl<J> Effect<J> for ConcatFusionOutputsEffect
where
    J: VoidInferOps + FusionConcatOps,
{
    fn effect(
        jungle: &J,
        (_transform_id, (left_id, right_id)): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left_emission: Emission<()> = jungle
                .download_emission(left_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_emission: Emission<()> = jungle
                .download_emission(right_id.0)
                .await
                .map_err(AtomError::Download)?;

            let left_bytes = jungle
                .download_raw(left_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;
            let right_bytes = jungle
                .download_raw(right_emission.output_id.0)
                .await
                .map_err(AtomError::Download)?;

            let mut merged_output: InferenceOutput = from_bytes(&left_bytes)?;
            let right_output: InferenceOutput = from_bytes(&right_bytes)?;
            merged_output.results.extend(right_output.results);

            let merged_output_bytes = to_allocvec(&merged_output)?;
            let merged_output_id = jungle
                .upload_to_void(merged_output_bytes)
                .await
                .map_err(AtomError::Upload)?;
            let merged_emission = Emission {
                metadata: (),
                output_id: InferenceOutputId(merged_output_id),
            };
            let merged_emission_bytes = to_allocvec(&merged_emission)?;
            let merged_emission_id = jungle
                .upload_to_void(merged_emission_bytes)
                .await
                .map_err(AtomError::Upload)?;

            jungle.record_fusion_concat();
            Ok(EmissionId(merged_emission_id))
        }
    }
}

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

#[jungle::animal(id = 1, generation = 0)]
impl Animal for ProgenitorBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<Generator, Policy>;
}

pub struct DarkStarBlackHole;

#[jungle::animal(id = 3, generation = 0)]
impl Animal for DarkStarBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <DarkStarSun as BlackHole>::Sun<DarkStarGenerator, DarkStarPolicy>;
}

pub struct BlackDwarfBlackHole;

#[jungle::animal(id = 4, generation = 0)]
impl Animal for BlackDwarfBlackHole {
    type State = SunState;
    type Seed = ();
    type Flow = <ThreeProgenitorSun as BlackHole>::Sun<DarkStarGenerator, DarkStarPolicy>;
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

#[cfg(test)]
async fn exercise_sun_epoch<A>(
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

/// Runs the same U0 -> U1 -> U2 Sun topology as `sun`, with real Progenitor
/// cells backed by a quark model.
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
    exercise_sun_epoch::<ProgenitorBlackHole>(
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

    exercise_sun_epoch::<DarkStarBlackHole>(
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

    exercise_sun_epoch::<BlackDwarfBlackHole>(
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
pub(crate) fn run_beam() {
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
        .view_live::<BlackDwarfBlackHole>(client, journey_id)
        .expect("Black Hole Beam should run");
}

/// Launches the Dark Star Beam example in a process whose UI runs on its main thread.
#[cfg(test)]
#[test]
#[ignore]
fn beam_dark_star() {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = std::process::Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["run", "--quiet", "--example", "beam_dark_star"])
        .status()
        .expect("beam_dark_star example should launch");

    assert!(
        status.success(),
        "beam_dark_star example exited with {status}"
    );
}

/// Launches the Black Dwarf Beam example in a process whose UI runs on its main thread.
#[cfg(test)]
#[test]
#[ignore]
fn beam_black_dwarf() {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = std::process::Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["run", "--quiet", "--example", "beam_black_dwarf"])
        .status()
        .expect("beam_black_dwarf example should launch");

    assert!(
        status.success(),
        "beam_black_dwarf example exited with {status}"
    );
}
