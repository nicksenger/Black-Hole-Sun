#[cfg(test)]
use futures::stream::StreamExt;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::cell::{CellState, Primordium};
use black_hole_sun::ops::{InferenceOutputOps, SunOps, TransmissionOps, VoidInferOps};
use black_hole_sun::sun::{BlackHole, SunAppearance, SunNodeState, SunState, Unary};
use black_hole_sun::{
    AtomError, CellInit, DarkToken, EmissionId, InferenceOutput, InferenceRequest, ModelConfig,
    NoErrorFeedback, NoOscillation, ObjectId, QuarkClient, QuarkModelConfig, QuarkModelParams,
    Ray, SequenceOutput, TestQuarkServer, TestVoidServer, Tokenizer, Transmission, VoidClient,
};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use jungle_sdk::JungleClient;
use tracing::info;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::{init_tracing, make_client_endpoint, require_model_path};
use super::dark_star::SPACE_PROBE_DISTANCE_PROMPT;

const WHITE_DWARF_NODE_COUNT: usize = 2;
const WHITE_DWARF_BATCH_SIZE: usize = 8;
const WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS: usize = 2;
const WHITE_DWARF_DEFAULT_INFERENCE_LIMIT: u32 = 512;
const WHITE_DWARF_CELL_INFERENCE_LIMIT: u32 = 10;

#[derive(Default)]
struct WhiteDwarfStateInner {
    _seen_epoch_count: usize,
}

type WhiteDwarfState = SunState<WhiteDwarfStateInner>;

struct WhiteDwarfModelConfig;

impl ModelConfig for WhiteDwarfModelConfig {
    type Oscillation = NoOscillation;
    type ErrorFeedback = NoErrorFeedback;
    const INFERENCE_LIMIT: Option<u32> = Some(WHITE_DWARF_CELL_INFERENCE_LIMIT);
}

struct WhiteDwarfCellAnimal;

#[jungle::animal(observe, id = 45, generation = 0)]
impl Animal for WhiteDwarfCellAnimal {
    type State = CellState;
    type Seed = CellInit;
    type Flow = Primordium<(), WhiteDwarfModelConfig>;
}

impl Observe for WhiteDwarfCellAnimal {
    type Appearance = Ray;

    fn observe(state: &Self::State) -> Self::Appearance {
        Ray {
            frozen: state.is_frozen,
        }
    }
}

type WhiteDwarfCell0 = Unary<U0, WhiteDwarfCellAnimal, list![U1]>;
type WhiteDwarfCell1 = Unary<U1, WhiteDwarfCellAnimal, list![]>;
type WhiteDwarfSun = list![WhiteDwarfCell0, WhiteDwarfCell1];

#[derive(Flow)]
struct WhiteDwarfGenerator(Step<GenerateWhiteDwarfPrompt>);

struct GenerateWhiteDwarfPrompt;
pub struct GenerateWhiteDwarfPromptEffect;

#[jungle::action]
impl Action for GenerateWhiteDwarfPrompt {
    type Effect = GenerateWhiteDwarfPromptEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &WhiteDwarfState, _input: Self::Input) {}

    fn absorb(
        _state: &mut WhiteDwarfState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("white dwarf generator failed: {error}")))
    }
}

#[jungle::effect(id = 37)]
impl<J: VoidInferOps> Effect<J> for GenerateWhiteDwarfPromptEffect {
    type In = ();
    type Out = (Transmission, Transmission);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        _input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let dark_tokens = jungle
                .darken(SPACE_PROBE_DISTANCE_PROMPT)
                .map_err(AtomError::Inference)?;
            let output = InferenceOutput {
                results: (0..WHITE_DWARF_BATCH_SIZE)
                    .map(|_| SequenceOutput(dark_tokens.clone()))
                    .collect(),
            };
            let propagation =
                Transmission::propagation_from_inference_output(jungle, &output).await?;
            Ok((propagation.clone(), propagation))
        }
    }
}

#[derive(Flow)]
struct WhiteDwarfPolicy(Step<WhiteDwarfLossPolicy>);

struct WhiteDwarfLossPolicy;
pub struct WhiteDwarfLossPolicyEffect;

#[jungle::action]
impl Action for WhiteDwarfLossPolicy {
    type Effect = WhiteDwarfLossPolicyEffect;
    type Input = [(Transmission, Transmission); WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS];
    type Output = (f32, f32);
    type Carry = ();

    fn emit(_state: &WhiteDwarfState, input: Self::Input) -> Self::Input {
        input
    }

    fn absorb(
        _state: &mut WhiteDwarfState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|error| Failure::Message(format!("white dwarf policy failed: {error}")))
    }
}

#[jungle::effect(id = 38)]
impl<J: VoidInferOps> Effect<J> for WhiteDwarfLossPolicyEffect {
    type In = [(Transmission, Transmission); WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS];
    type Out = (f32, f32);
    type Err = AtomError;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            for (index, (up_tx, down_tx)) in input.iter().enumerate() {
                let output_up = InferenceOutput::from_transmission(jungle, up_tx).await?;
                let output_down = InferenceOutput::from_transmission(jungle, down_tx).await?;
                let up_batch_size = output_up.results.len();
                let down_batch_size = output_down.results.len();
                let accumulation_step = index + 1;

                info!(
                    accumulation_step,
                    up_batch_size, down_batch_size, "white_dwarf reward received batch outputs"
                );

                if up_batch_size != WHITE_DWARF_BATCH_SIZE
                    || down_batch_size != WHITE_DWARF_BATCH_SIZE
                {
                    return Err(AtomError::Inference(format!(
                        "expected batch size {WHITE_DWARF_BATCH_SIZE} in white_dwarf reward fn \
                         at accumulation step {accumulation_step}, got up={up_batch_size}, \
                         down={down_batch_size}"
                    )));
                }
            }

            Ok((0.4, 0.8))
        }
    }
}

struct WhiteDwarfBlackHole;

#[jungle::animal(observe, id = 44, generation = 0)]
impl Animal for WhiteDwarfBlackHole {
    type State = WhiteDwarfState;
    type Seed = ();
    type Flow = <WhiteDwarfSun as BlackHole>::Sun<
        WhiteDwarfGenerator,
        WhiteDwarfPolicy,
        WhiteDwarfStateInner,
        WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS,
    >;
}

impl Observe for WhiteDwarfBlackHole {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

#[derive(Animals)]
struct WhiteDwarfAnimals(WhiteDwarfCellAnimal, WhiteDwarfBlackHole);

#[derive(Clone)]
struct WhiteDwarfJungle {
    void_client: VoidClient,
    quark_client: QuarkClient,
    tokenizer: Arc<OnceLock<Result<Tokenizer, String>>>,
    client: Option<FusedClient>,
}

impl WhiteDwarfJungle {
    fn new(void_client: VoidClient, quark_client: QuarkClient) -> Self {
        Self {
            void_client,
            quark_client,
            tokenizer: Arc::new(OnceLock::new()),
            client: None,
        }
    }

    fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
    }
}

impl Ecosystem for WhiteDwarfJungle {
    const NAME: &'static str = "white-dwarf-jungle";
    type Animals = WhiteDwarfAnimals;
}

#[async_trait]
impl VoidInferOps for WhiteDwarfJungle {
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
        self.void_client.upload(data).await
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        self.void_client.upload_with(id, data).await.map(|_| ())
    }

    async fn start_model(
        &self,
        model_id: Uuid,
        model_config: Option<QuarkModelConfig>,
    ) -> Result<(), String> {
        self.quark_client.start(model_id, model_config).await
    }

    async fn infer(&self, model_id: Uuid, request: InferenceRequest) -> Result<ObjectId, String> {
        let request_bytes = postcard::to_allocvec(&request).map_err(|error| error.to_string())?;
        let request_id = self.void_client.upload(request_bytes).await?;
        self.quark_client.infer(model_id, request_id).await
    }

    async fn reset_model(&self, model_id: Uuid) -> Result<(), String> {
        self.quark_client.reset(model_id).await
    }

    async fn checkpoint_model(&self, model_id: Uuid) -> Result<ObjectId, String> {
        self.quark_client.checkpoint(model_id).await
    }

    fn darken(&self, prompt: &str) -> Result<Vec<DarkToken>, String> {
        let tokenizer_result = self.tokenizer.get_or_init(Tokenizer::try_init);
        let tokenizer = tokenizer_result
            .as_ref()
            .map_err(|error| format!("failed to initialize tokenizer: {error}"))?;
        tokenizer
            .darken(prompt)
            .map_err(|error| format!("failed to darken prompt: {error}"))
    }

    fn decode(&self, tokens: &[DarkToken]) -> String {
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

    async fn query_model_params(&self, model_id: Uuid) -> Result<QuarkModelParams, String> {
        self.quark_client.query_model_params(model_id).await
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
        let data = postcard::to_allocvec(&propagation).map_err(|error| error.to_string())?;
        self.void_client.upload_with(send_id, data).await?;
        Ok(())
    }
}

#[async_trait]
impl SunOps for WhiteDwarfJungle {
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

#[tokio::test]
async fn white_dwarf() {
    init_tracing();

    let model_path = match require_model_path("white_dwarf") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .default_inference_limit(WHITE_DWARF_DEFAULT_INFERENCE_LIMIT)
        .serve()
        .await
        .expect("failed to start quark server");
    let void_addr = void_server.local_addr();
    let quark_addr = quark_server.local_addr();

    let endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
    let quark_client = QuarkClient::new(&endpoint, quark_addr, "localhost");
    let mut jungle = WhiteDwarfJungle::new(void_client, quark_client);

    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");
    jungle.set_client(client.clone());

    let parent_journey_id = client
        .spawn::<WhiteDwarfBlackHole>(&())
        .await
        .expect("WhiteDwarfBlackHole should spawn")
        .journey_id;
    let mut subscription = client
        .subscribe_step_updates(parent_journey_id, None)
        .await
        .expect("subscribe_step_updates should succeed");

    let worker_handles: Vec<_> = (0..(WHITE_DWARF_NODE_COUNT + 1))
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    let mut seen_grad_steps: HashMap<u32, HashSet<usize>> = HashMap::new();
    let mut seen_state_sequences: HashMap<u32, HashSet<u64>> = HashMap::new();
    let mut latest_appearance: Option<SunAppearance> = None;
    let timeout_result = tokio::time::timeout(Duration::from_secs(240), async {
        loop {
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
                            return Err("step update stream ended unexpectedly".to_string());
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }

            let Some(parent_bytes) = client
                .animal_appearance(parent_journey_id)
                .await
                .map_err(|error| format!("parent animal_appearance failed: {error}"))?
            else {
                continue;
            };
            let appearance = postcard::from_bytes::<SunAppearance>(&parent_bytes)
                .map_err(|error| format!("Sun appearance should deserialize: {error}"))?;
            if !appearance.finalized || appearance.nodes.len() != WHITE_DWARF_NODE_COUNT {
                continue;
            }

            if appearance.grad_steps != WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS {
                return Err(format!(
                    "expected {} grad steps, got {}",
                    WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS, appearance.grad_steps
                ));
            }
            latest_appearance = Some(appearance.clone());

            for node in appearance
                .nodes
                .iter()
                .filter(|node| node.state != SunNodeState::Idle)
            {
                seen_grad_steps
                    .entry(node.id)
                    .or_default()
                    .insert(node.grad_step);
                seen_state_sequences
                    .entry(node.id)
                    .or_default()
                    .insert(node.state_sequence);
            }

            let each_node_saw_both_grad_steps = (0..WHITE_DWARF_NODE_COUNT as u32).all(|node_id| {
                seen_grad_steps.get(&node_id).is_some_and(|steps| {
                    (1..=WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS)
                        .all(|expected_step| steps.contains(&expected_step))
                })
            });
            if each_node_saw_both_grad_steps {
                return Ok::<(), String>(());
            }
        }
    })
    .await;

    match timeout_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let status = client
                .journey_details(parent_journey_id)
                .await
                .expect("journey_details should succeed");
            panic!(
                "white_dwarf failed: {error}, seen_grad_steps={seen_grad_steps:?}, \
                 seen_state_sequences={seen_state_sequences:?}, latest_appearance={latest_appearance:?}, status: {status:?}"
            );
        }
        Err(error) => {
            let status = client
                .journey_details(parent_journey_id)
                .await
                .expect("journey_details should succeed");
            panic!(
                "timeout waiting for white_dwarf progression (240s): {error}; \
                 seen_grad_steps={seen_grad_steps:?}, seen_state_sequences={seen_state_sequences:?}, \
                 latest_appearance={latest_appearance:?}, status: {status:?}"
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
