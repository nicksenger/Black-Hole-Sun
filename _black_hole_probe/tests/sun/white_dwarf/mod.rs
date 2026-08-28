#[cfg(test)]
use futures::stream::StreamExt;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::net::{Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::cell::action::Potentiation;
use black_hole_sun::cell::{CellState, Primordium};
use black_hole_sun::ops::{InferenceOutputOps, SunOps, TransmissionOps, VoidInferOps};
use black_hole_sun::sun::{BlackHole, Manifest, SunAppearance, SunNodeState, SunState, Unary};
use black_hole_sun::{
    AtomError, CellInit, DarkToken, EmissionId, ErrorFeedbackPolicy, InferenceOutput,
    InferenceRequest, MassClient, MassErrorFeedbackMode, MassModelConfig, MassModelParams,
    ModelConfig, ObjectId, OscillationSchedule, Ray, RunningTestMassServer, RunningTestVoidServer,
    SequenceOutput, TestMassServer, TestVoidServer, Tokenizer, Transmission, VoidClient,
};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::server::ServerBuilder;
use jungle_sdk::JungleClient;
use tokio::task::JoinHandle;
use tracing::info;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::{init_tracing, require_model_path};
use super::dark_star::SPACE_PROBE_DISTANCE_PROMPT;

const WHITE_DWARF_NODE_COUNT: usize = 2;
const WHITE_DWARF_BATCH_SIZE: usize = 8;
const WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS: usize = 2;
const WHITE_DWARF_DEFAULT_INFERENCE_LIMIT: u32 = 512;
const WHITE_DWARF_CELL_INFERENCE_LIMIT: u32 = 10;

/// Namespace shared by every jungle client of a launched journey.
const JUNGLE_NAMESPACE: &str = "default";

/// TLS server name of the local jungle QUIC server (self-signed cert).
const JUNGLE_SERVER_NAME: &str = "localhost";

/// Connection attempts while the jungle server is still starting up.
const JUNGLE_CONNECT_ATTEMPTS: usize = 40;

/// Delay between jungle client connection attempts.
const JUNGLE_CONNECT_RETRY_DELAY: Duration = Duration::from_millis(25);

#[derive(Default)]
pub struct WhiteDwarfStateInner {
    _seen_epoch_count: usize,
}

type WhiteDwarfState = SunState<WhiteDwarfStateInner>;

#[derive(Clone)]
struct WhiteDwarfOscillationA;

impl OscillationSchedule for WhiteDwarfOscillationA {
    const PERIOD_STEPS: Option<u32> = Some(2);
    const TRAIN_STEPS: Option<u32> = Some(1);
    const PHASE_STEPS: Option<u32> = Some(0);
}

#[derive(Clone)]
struct WhiteDwarfOscillationB;

impl OscillationSchedule for WhiteDwarfOscillationB {
    const PERIOD_STEPS: Option<u32> = Some(2);
    const TRAIN_STEPS: Option<u32> = Some(1);
    const PHASE_STEPS: Option<u32> = Some(1);
}

struct WhiteDwarfPersistentResiduals;

impl ErrorFeedbackPolicy for WhiteDwarfPersistentResiduals {
    const MODE: Option<MassErrorFeedbackMode> = Some(MassErrorFeedbackMode::Persistent);
}

struct WhiteDwarfModelConfigA;

impl ModelConfig for WhiteDwarfModelConfigA {
    type Oscillation = WhiteDwarfOscillationA;
    type ErrorFeedback = WhiteDwarfPersistentResiduals;
    const FROZEN: Option<bool> = Some(false);
    const INFERENCE_LIMIT: Option<u32> = Some(WHITE_DWARF_CELL_INFERENCE_LIMIT);
}

struct WhiteDwarfModelConfigB;

impl ModelConfig for WhiteDwarfModelConfigB {
    type Oscillation = WhiteDwarfOscillationB;
    type ErrorFeedback = WhiteDwarfPersistentResiduals;
    const FROZEN: Option<bool> = Some(false);
    const INFERENCE_LIMIT: Option<u32> = Some(WHITE_DWARF_CELL_INFERENCE_LIMIT);
}

// Primordium expands to Cell<Atom<...>> and keeps inference on Atom's
// MassInferWithBackoff retry path.
type WhiteDwarfBackoffPrimordium<H> = Primordium<(), H>;

struct WhiteDwarfCell0Animal;

#[jungle::animal(observe, id = 45, generation = 0)]
impl Animal for WhiteDwarfCell0Animal {
    type State = CellState;
    type Seed = CellInit;
    type Flow = WhiteDwarfBackoffPrimordium<WhiteDwarfModelConfigA>;
}

impl Observe for WhiteDwarfCell0Animal {
    type Appearance = Ray;

    fn observe(state: &Self::State) -> Self::Appearance {
        Ray {
            frozen: state.is_frozen,
        }
    }
}

struct WhiteDwarfCell1Animal;

#[jungle::animal(observe, id = 46, generation = 0)]
impl Animal for WhiteDwarfCell1Animal {
    type State = CellState;
    type Seed = CellInit;
    type Flow = WhiteDwarfBackoffPrimordium<WhiteDwarfModelConfigB>;
}

impl Observe for WhiteDwarfCell1Animal {
    type Appearance = Ray;

    fn observe(state: &Self::State) -> Self::Appearance {
        Ray {
            frozen: state.is_frozen,
        }
    }
}

type WhiteDwarfCell0 = Unary<U0, WhiteDwarfCell0Animal, list![U1]>;
type WhiteDwarfCell1 = Unary<U1, WhiteDwarfCell1Animal, list![]>;
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

#[jungle::effect(id = 87)]
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
    type Output = Potentiation;
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

#[jungle::effect(id = 88)]
impl<J: VoidInferOps> Effect<J> for WhiteDwarfLossPolicyEffect {
    type In = [(Transmission, Transmission); WHITE_DWARF_GRADIENT_ACCUMULATION_STEPS];
    type Out = Potentiation;
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

            Ok(Potentiation {
                loss_up: 0.4,
                loss_down: 0.8,
                seed: 11,
            })
        }
    }
}

/// Manifest for the white dwarf sun — bundles generator, policy, and state.
struct WhiteDwarfManifest;

impl Manifest for WhiteDwarfManifest {
    type Generator = WhiteDwarfGenerator;
    type Policy = WhiteDwarfPolicy;
    type State = WhiteDwarfStateInner;
}

struct WhiteDwarfBlackHole;

#[jungle::animal(observe, id = 44, generation = 0)]
impl Animal for WhiteDwarfBlackHole {
    type State = WhiteDwarfState;
    type Seed = ();
    type Flow = <WhiteDwarfSun as BlackHole>::Sun<
        WhiteDwarfManifest,
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
struct WhiteDwarfAnimals(
    WhiteDwarfCell0Animal,
    WhiteDwarfCell1Animal,
    WhiteDwarfBlackHole,
);

#[derive(Clone)]
struct WhiteDwarfJungle {
    void_client: VoidClient,
    mass_client: MassClient,
    tokenizer: Arc<OnceLock<Result<Tokenizer, String>>>,
    client: Option<jungle_sdk::Client>,
}

impl WhiteDwarfJungle {
    fn new(void_client: VoidClient, mass_client: MassClient) -> Self {
        Self {
            void_client,
            mass_client,
            tokenizer: Arc::new(OnceLock::new()),
            client: None,
        }
    }

    fn set_client(&mut self, client: jungle_sdk::Client) {
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
        model_config: Option<MassModelConfig>,
    ) -> Result<(), String> {
        self.mass_client.start(model_id, model_config).await
    }

    async fn infer(&self, model_id: Uuid, request: InferenceRequest) -> Result<ObjectId, String> {
        let request_bytes = postcard::to_allocvec(&request).map_err(|error| error.to_string())?;
        let request_id = self.void_client.upload(request_bytes).await?;
        self.mass_client.infer(model_id, request_id).await
    }

    async fn reset_model(&self, model_id: Uuid) -> Result<(), String> {
        self.mass_client.reset(model_id).await
    }

    async fn checkpoint_model(&self, model_id: Uuid) -> Result<ObjectId, String> {
        self.mass_client.checkpoint(model_id).await
    }

    async fn fuse_weights(
        &self,
        model_id: Uuid,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        self.mass_client.fuse_weights(model_id, checkpoint_id, contribution).await
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
        self.mass_client.perturb_up(model_id, seed).await
    }

    async fn perturb_down(&self, model_id: Uuid) -> Result<(), String> {
        self.mass_client.perturb_down(model_id).await
    }

    async fn optimize(&self, model_id: Uuid, loss_up: f32, loss_down: f32) -> Result<(), String> {
        self.mass_client
            .optimize(model_id, loss_up, loss_down)
            .await
    }

    async fn query_model_params(&self, model_id: Uuid) -> Result<MassModelParams, String> {
        self.mass_client.query_model_params(model_id).await
    }

    async fn shutdown_model(&self, model_id: Uuid) -> Result<(), String> {
        self.mass_client.shutdown(model_id).await
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

    async fn observe_animal<Ap>(&self, journey_id: Uuid) -> Result<Ap, String>
    where
        Ap: serde::de::DeserializeOwned + Send,
    {
        let client = self.client.clone().expect("client not set");
        let appearance_bytes = client
            .animal_appearance(journey_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("appearance unavailable for journey {journey_id}"))?;
        postcard::from_bytes::<Ap>(&appearance_bytes)
            .map_err(|error| format!("deserialize appearance failed: {error}"))
    }

    async fn perturb_animal<S>(&self, journey_id: Uuid, stimulus: &S) -> Result<(), String>
    where
        S: serde::Serialize + Sync + Send,
    {
        let payload = postcard::to_allocvec(stimulus)
            .map_err(|error| format!("serialize perturb stimulus failed: {error}"))?;
        let client = self.client.clone().expect("client not set");
        client
            .perturb_animal(journey_id, payload)
            .await
            .map_err(|error| error.to_string())
    }
}

/// Reserve an ephemeral local UDP port for the jungle QUIC server (the
/// reserved socket is dropped immediately, freeing the port for the server).
fn reserve_local_addr() -> SocketAddr {
    let socket = UdpSocket::bind((Ipv6Addr::LOCALHOST, 0))
        .expect("should bind temporary udp socket for jungle port reservation");
    socket
        .local_addr()
        .expect("temporary udp socket should expose local address")
}

/// Build a QUIC client connected to the jungle server at `addr`, retrying
/// while the server is still binding and generating its self-signed cert.
async fn connect_jungle_client(
    addr: SocketAddr,
    namespace: &str,
) -> Result<jungle_sdk::Client, String> {
    for attempt in 0..JUNGLE_CONNECT_ATTEMPTS {
        match jungle_sdk::Client::builder()
            .namespace(namespace)
            .remote(addr)
            .server_name(JUNGLE_SERVER_NAME)
            .build()
            .await
        {
            Ok(client) => return Ok(client),
            Err(error) if attempt + 1 < JUNGLE_CONNECT_ATTEMPTS => {
                tokio::time::sleep(JUNGLE_CONNECT_RETRY_DELAY).await;
                let _ = error;
            }
            Err(error) => {
                return Err(format!(
                    "failed to connect jungle client to {addr}: {error}"
                ))
            }
        }
    }

    unreachable!("jungle client retry loop always returns")
}

/// Connection details for the local jungle QUIC server started by the test,
/// enough for a caller to build their own client.
///
/// The spawn phase keeps its own client for the journey and worker pool and
/// does not hand one out: a QUIC client is bound to the runtime it was built
/// on, so callers that need their own (e.g. to observe the journey from
/// another runtime) should build one with [`Self::connect`] on the runtime
/// they will use it from.
#[derive(Debug, Clone)]
struct JungleConnection {
    /// Address the jungle QUIC server is listening on.
    addr: SocketAddr,
    /// Namespace all clients use (journeys are namespaced).
    namespace: String,
}

impl JungleConnection {
    /// Build a QUIC client connected to this server on the **current**
    /// runtime, retrying while the server is still starting up.
    async fn connect(&self) -> Result<jungle_sdk::Client, String> {
        connect_jungle_client(self.addr, &self.namespace).await
    }
}

/// Handle for the local jungle QUIC server. Dropping it stops the server.
struct RunningJungleServer {
    abort_handle: tokio::task::AbortHandle,
}

impl RunningJungleServer {
    fn new(task: JoinHandle<()>) -> Self {
        Self {
            abort_handle: task.abort_handle(),
        }
    }
}

impl Drop for RunningJungleServer {
    fn drop(&mut self) {
        self.abort_handle.abort();
    }
}

/// Everything a launched journey needs to stay alive: the servers, the jungle
/// connection details, and the worker pool. Dropping this tears down the
/// servers.
struct Launched {
    journey_id: Uuid,
    /// Details for building a client connected to the local jungle server.
    jungle: JungleConnection,
    /// The local jungle QUIC server; stopped when dropped with `Launched`.
    jungle_server: RunningJungleServer,
    void_server: RunningTestVoidServer,
    mass_server: RunningTestMassServer,
    workers: Vec<JoinHandle<()>>,
}

#[test]
fn white_dwarf() {
    init_tracing();

    let model_path = match require_model_path("white_dwarf") {
        Some(path) => path,
        None => return,
    };

    // Follow the model-eval bin's runtime strategy: build a multi-threaded
    // runtime and block on it to launch the journey.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Tokio runtime should build");

    let launched = runtime.block_on(async {
        let void_server = TestVoidServer::new()
            .tcp()
            .listen_on_all_interfaces()
            .listen_port(8889)
            .serve()
            .await
            .expect("failed to start void server");
        let mass_server = TestMassServer::new(&model_path)
            .tcp()
            .listen_on_all_interfaces()
            .listen_port(8888)
            .void_addr(void_server.local_addr())
            .default_inference_limit(WHITE_DWARF_DEFAULT_INFERENCE_LIMIT)
            .serve()
            .await
            .expect("failed to start mass server");
        let void_addr = void_server.local_addr();
        let mass_addr = mass_server.local_addr();

        let void_client = VoidClient::new_tcp(void_addr);
        let mass_client = MassClient::new_tcp(mass_addr);
        let mut jungle = WhiteDwarfJungle::new(void_client, mass_client);

        // Start the local jungle QUIC server on an ephemeral port. The spawn
        // phase keeps its own client for the journey and worker pool; callers
        // that need a client of their own get the connection details in
        // `Launched` and build one on their own runtime.
        let jungle_addr = reserve_local_addr();
        let jungle_server = RunningJungleServer::new(tokio::spawn(async move {
            if let Err(error) = ServerBuilder::new()
                .listen(jungle_addr)
                .memory()
                .run()
                .await
            {
                tracing::error!("jungle server exited with error: {error}");
            }
        }));

        let client = connect_jungle_client(jungle_addr, JUNGLE_NAMESPACE)
            .await
            .expect("jungle client should connect");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<WhiteDwarfBlackHole>(&())
            .await
            .expect("WhiteDwarfBlackHole should spawn")
            .journey_id;

        let workers: Vec<_> = (0..(WHITE_DWARF_NODE_COUNT + 1))
            .map(|_| {
                let worker = JungleWorker::new(jungle.clone(), client.clone());
                tokio::spawn(async move {
                    let _ = worker.spawn().await;
                })
            })
            .collect();

        Launched {
            journey_id,
            jungle: JungleConnection {
                addr: jungle_addr,
                namespace: JUNGLE_NAMESPACE.to_string(),
            },
            jungle_server,
            void_server,
            mass_server,
            workers,
        }
    });

    // Run the assertions on a separate runtime, driving the journey through
    // the client from `launched` — mirroring how model-eval keeps its launched
    // journey alive while external code observes it.
    let assert_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Tokio runtime should build");

    let mut seen_grad_steps: HashMap<u32, HashSet<usize>> = HashMap::new();
    let mut seen_state_sequences: HashMap<u32, HashSet<u64>> = HashMap::new();
    let mut latest_appearance: Option<SunAppearance> = None;
    let timeout_result = assert_runtime.block_on(async {
        // Build the assertion client on this runtime rather than reusing one
        // created elsewhere: a QUIC client is bound to the runtime it was
        // built on.
        let client = launched
            .jungle
            .connect()
            .await
            .expect("jungle client should connect");

        let mut subscription = client
            .subscribe_step_updates(launched.journey_id, None)
            .await
            .expect("subscribe_step_updates should succeed");

        tokio::time::timeout(Duration::from_secs(240), async {
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
                    .animal_appearance(launched.journey_id)
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
        .await
    });

    match timeout_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            let status = assert_runtime.block_on(async {
                let client = launched
                    .jungle
                    .connect()
                    .await
                    .expect("jungle client should connect");
                client
                    .journey_details(launched.journey_id)
                    .await
                    .expect("journey_details should succeed")
            });
            panic!(
                "white_dwarf failed: {error}, seen_grad_steps={seen_grad_steps:?}, \
                 seen_state_sequences={seen_state_sequences:?}, latest_appearance={latest_appearance:?}, status: {status:?}"
            );
        }
        Err(error) => {
            let status = assert_runtime.block_on(async {
                let client = launched
                    .jungle
                    .connect()
                    .await
                    .expect("jungle client should connect");
                client
                    .journey_details(launched.journey_id)
                    .await
                    .expect("journey_details should succeed")
            });
            panic!(
                "timeout waiting for white_dwarf progression (240s): {error}; \
                 seen_grad_steps={seen_grad_steps:?}, seen_state_sequences={seen_state_sequences:?}, \
                 latest_appearance={latest_appearance:?}, status: {status:?}"
            );
        }
    }

    for worker_handle in &launched.workers {
        worker_handle.abort();
    }
    runtime.block_on(async {
        for worker_handle in launched.workers {
            let _ = worker_handle.await;
        }
    });
    drop(launched.jungle_server);
    launched.void_server.abort();
    launched.mass_server.abort();
}
