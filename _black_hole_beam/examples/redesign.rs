//! Visual acceptance example for the generic tensor-operation redesign.
//!
//! Run it from the workspace root:
//!
//! ```text
//! cargo run -p black-hole-beam --example redesign
//! ```
//!
//! The Beam window animates a forward-only, non-Qwen tensor pipeline compiled
//! through the canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.
//! Click any node to inspect its forward-only child flow.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use black_hole_beam::BeamBuilder;
use black_hole_contract::{
    glowstick::{Dyn, Shape2},
    SingleTensorSpec, TensorContract, TensorPortSpec,
};
use black_hole_flux::sun::{
    SunAppearance, SunEdgeAppearance, SunNodeAppearance, SunNodeState, SunOperationalState,
    SunState,
};
use black_hole_flux::{
    BlackHole, CellInit, CellState, Edge, ForwardOnly, ForwardOperationPrimordium, OperationNode,
    Primordium, TypedEdges, Unary,
};
use black_hole_spec::{ContractId, DimensionDescriptor, DtypeConstraint, TensorDtype};
use iced::futures::stream;
use jungle_client::MockClient;
use jungle_sdk::typosaurus::collections::list::{Empty, List};
use jungle_sdk::{
    Animal, ClaimedPerturbable, ExecutorError, Id, JourneyAst, JourneyHandle, JourneyRecord,
    JourneyReplayPage, JourneyStatus, JourneyUpdateEvent, JungleClient, NodeLifecycle,
    NodeLifecyclePhase, Observe, OwnerWake, RunnerOut, RunnerUpdateOut, SpawnableAnimal,
    SupportedAnimal, Work,
};
use typenum::{U0, U1, U2, U3, U8};
use uuid::Uuid;

struct Batch;

struct FeaturePort;

impl TensorPortSpec for FeaturePort {
    type Shape = Shape2<Dyn<Batch>, U8>;

    const NAME: &'static str = "features";

    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![
            DimensionDescriptor::Symbolic("batch".into()),
            DimensionDescriptor::Static(8),
        ]
    }

    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

type FeatureBatch = SingleTensorSpec<FeaturePort>;

struct Normalize;

impl TensorContract for Normalize {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x4e4f_524d_414c_495a_45);
    const VERSION: u32 = 1;
}

struct Encode;

impl TensorContract for Encode {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x454e_434f_4445);
    const VERSION: u32 = 1;
}

struct Classify;

impl TensorContract for Classify {
    type Input = FeatureBatch;
    type Output = FeatureBatch;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x434c_4153_5349_4659);
    const VERSION: u32 = 1;
}

struct NormalizeFeatures;

impl Animal for NormalizeFeatures {
    type Id = Id<U0>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Normalize>;
}

impl OperationNode<Normalize> for NormalizeFeatures {}

struct EncodeFeatures;

impl Animal for EncodeFeatures {
    type Id = Id<U1>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Encode>;
}

impl OperationNode<Encode> for EncodeFeatures {}

struct ClassifyFeatures;

impl Animal for ClassifyFeatures {
    type Id = Id<U2>;
    type Generation = U0;
    type State = CellState;
    type Seed = CellInit;
    type Flow = ForwardOperationPrimordium<Classify>;
}

impl OperationNode<Classify> for ClassifyFeatures {}

type ClassifyNode = List<(Unary<U2, ClassifyFeatures, Empty, Classify>, Empty)>;
type EncodeNode = List<(
    Unary<U1, EncodeFeatures, TypedEdges<List<(Edge<U2, Classify>, Empty)>>, Encode>,
    ClassifyNode,
)>;
type Topology = List<(
    Unary<U0, NormalizeFeatures, TypedEdges<List<(Edge<U1, Encode>, Empty)>>, Normalize>,
    EncodeNode,
)>;

type RedesignSun = <Topology as BlackHole>::Sun<ForwardOnly<Primordium, Normalize>>;

struct RedesignDemo;

impl Animal for RedesignDemo {
    type Id = Id<U3>;
    type Generation = U0;
    type State = SunState;
    type Seed = ();
    type Flow = RedesignSun;
}

impl Observe for RedesignDemo {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

fn demo_appearance(elapsed: Duration) -> SunAppearance {
    let cycle_stage = (elapsed.as_millis() / 1_200 % 5) as usize;
    let cycle = elapsed.as_millis() / (1_200 * 5);
    let node = |id: u32, label: &str, input_ports: Vec<u32>| {
        let position = id as usize + 1;
        let operational_state = if cycle_stage == 0 || position > cycle_stage {
            SunOperationalState::Queued
        } else if position == cycle_stage {
            SunOperationalState::Running
        } else {
            SunOperationalState::Succeeded
        };
        SunNodeAppearance {
            id,
            journey_id: Uuid::from_u128(0x100 + u128::from(id)),
            warp_journey_id: Uuid::nil(),
            label: label.to_string(),
            input_ports,
            state: SunNodeState::Idle,
            state_sequence: (cycle * 5 + cycle_stage as u128) as u64,
            grad_step: 1,
            operational_state,
            phase_annotation: Some("forward".to_string()),
        }
    };

    SunAppearance {
        finalized: true,
        grad_steps: 1,
        nodes: vec![
            node(0, "NormalizeFeatures", vec![0]),
            node(1, "EncodeFeatures", vec![1]),
            node(2, "ClassifyFeatures", vec![2]),
        ],
        edges: vec![
            SunEdgeAppearance {
                source: 0,
                target: 1,
                target_port: 1,
            },
            SunEdgeAppearance {
                source: 1,
                target: 2,
                target_port: 2,
            },
        ],
    }
}

#[derive(Clone)]
struct AnimatedDemoClient {
    inner: MockClient,
    child_runtime_nodes: u32,
}

impl AnimatedDemoClient {
    fn new(inner: MockClient) -> Self {
        let ast =
            <ForwardOperationPrimordium<Normalize> as jungle_sdk::JourneyAstSource>::journey_ast();
        Self {
            inner,
            child_runtime_nodes: runtime_node_count(&ast),
        }
    }
}

fn runtime_node_count(ast: &JourneyAst) -> u32 {
    match ast {
        JourneyAst::Empty => 0,
        JourneyAst::Sequence(nodes) => nodes.iter().map(runtime_node_count).sum(),
        JourneyAst::Step { .. } => 1,
        JourneyAst::Conditional { left, right, .. }
        | JourneyAst::Select { left, right, .. }
        | JourneyAst::Join { left, right, .. } => {
            1 + runtime_node_count(left) + runtime_node_count(right)
        }
        JourneyAst::While { body, .. }
        | JourneyAst::Transparent { body, .. }
        | JourneyAst::Attempt { body, .. } => 1 + runtime_node_count(body),
    }
}

fn unix_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[async_trait]
impl JungleClient for AnimatedDemoClient {
    async fn spawn<A>(&self, _seed: &A::Seed) -> Result<JourneyHandle, ExecutorError>
    where
        Self: Sized,
        A: SpawnableAnimal,
        A::Seed: Sync,
    {
        Err(ExecutorError::ClientTransport(
            "the visual acceptance fixture does not spawn journeys".to_string(),
        ))
    }

    async fn journey_history(&self, id: Uuid) -> Result<Vec<RunnerOut>, ExecutorError> {
        self.inner.journey_history(id).await
    }

    async fn journey_replay_page(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
        snapshot_end_sequence_id: Option<u64>,
        limit: u32,
    ) -> Result<JourneyReplayPage, ExecutorError> {
        self.inner
            .journey_replay_page(
                journey_id,
                after_sequence_id,
                snapshot_end_sequence_id,
                limit,
            )
            .await
    }

    async fn list_journeys(&self, namespace: String) -> Result<Vec<JourneyRecord>, ExecutorError> {
        self.inner.list_journeys(namespace).await
    }

    async fn subscribe_step_updates(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
    ) -> Result<jungle_sdk::client::JourneyUpdateSubscription, ExecutorError> {
        let child_index = journey_id
            .as_u128()
            .checked_sub(0x100)
            .filter(|index| *index < 3)
            .ok_or_else(|| {
                ExecutorError::ClientTransport(format!(
                    "no animated child flow registered for journey {journey_id}"
                ))
            })?;
        let runtime_nodes = u64::from(self.child_runtime_nodes.max(1));
        let start_sequence = after_sequence_id.unwrap_or(0);
        let updates = stream::unfold(start_sequence, move |sequence| async move {
            if sequence > start_sequence {
                tokio::time::sleep(Duration::from_millis(350)).await;
            }
            let phase_index = sequence % (runtime_nodes * 2);
            let node_id = (phase_index / 2) as u32;
            let phase = if phase_index % 2 == 0 {
                NodeLifecyclePhase::Entered
            } else {
                NodeLifecyclePhase::Succeeded
            };
            let cycle = sequence / (runtime_nodes * 2);
            let update = JourneyUpdateEvent {
                sequence_id: sequence + 1,
                event_unix_ms: unix_time_ms(),
                event: RunnerUpdateOut::NodeLifecycle(NodeLifecycle {
                    node_id,
                    activation_path: vec![child_index as u64, cycle],
                    phase,
                    uuid: journey_id,
                }),
            };
            Some((Ok(update), sequence + 1))
        });
        Ok(jungle_sdk::client::JourneyUpdateSubscription::from_stream(
            updates,
        ))
    }

    async fn journey_details(&self, id: Uuid) -> Result<JourneyStatus, ExecutorError> {
        self.inner.journey_details(id).await
    }

    async fn animal_appearance(&self, id: Uuid) -> Result<Option<Vec<u8>>, ExecutorError> {
        self.inner.animal_appearance(id).await
    }

    async fn animal_appearance_update(&self, id: Uuid, data: Vec<u8>) -> Result<(), ExecutorError> {
        self.inner.animal_appearance_update(id, data).await
    }

    async fn perturb_animal(&self, id: Uuid, payload: Vec<u8>) -> Result<(), ExecutorError> {
        self.inner.perturb_animal(id, payload).await
    }

    async fn claim_animal_perturbation(
        &self,
        id: Uuid,
    ) -> Result<Option<ClaimedPerturbable>, ExecutorError> {
        self.inner.claim_animal_perturbation(id).await
    }

    async fn ack_animal_perturbation(
        &self,
        id: Uuid,
        perturbation_id: u64,
    ) -> Result<(), ExecutorError> {
        self.inner
            .ack_animal_perturbation(id, perturbation_id)
            .await
    }

    async fn heartbeat_journey_lease(
        &self,
        journey_id: Uuid,
        owner_id: Uuid,
        lease_ttl_ms: i64,
    ) -> Result<(), ExecutorError> {
        self.inner
            .heartbeat_journey_lease(journey_id, owner_id, lease_ttl_ms)
            .await
    }

    async fn poll_owner_wake(&self, owner_id: Uuid) -> Result<Option<OwnerWake>, ExecutorError> {
        self.inner.poll_owner_wake(owner_id).await
    }

    async fn schedule_sleep_timer(
        &self,
        journey_id: Uuid,
        timer_id: Uuid,
        wake_at_unix_ms: i64,
    ) -> Result<(), ExecutorError> {
        self.inner
            .schedule_sleep_timer(journey_id, timer_id, wake_at_unix_ms)
            .await
    }

    async fn complete_journey(&self, id: Uuid) -> Result<(), ExecutorError> {
        self.inner.complete_journey(id).await
    }

    async fn dead_journey(&self, id: Uuid) -> Result<(), ExecutorError> {
        self.inner.dead_journey(id).await
    }

    async fn poll_timers(&self) -> Result<Option<()>, ExecutorError> {
        self.inner.poll_timers().await
    }

    async fn poll_work(
        &self,
        supported_animals: Vec<SupportedAnimal>,
    ) -> Result<Option<Work>, ExecutorError> {
        self.inner.poll_work(supported_animals).await
    }

    async fn wait_for_worker_wake(
        &self,
        owner_id: Uuid,
        supported_animals: Vec<SupportedAnimal>,
        timeout: Duration,
    ) -> Result<(), ExecutorError> {
        self.inner
            .wait_for_worker_wake(owner_id, supported_animals, timeout)
            .await
    }

    async fn effect_input(
        &self,
        id: Uuid,
        node_id: u32,
        input: Vec<u8>,
    ) -> Result<(), ExecutorError> {
        self.inner.effect_input(id, node_id, input).await
    }

    async fn effect_success_output(
        &self,
        id: Uuid,
        node_id: u32,
        output: Vec<u8>,
    ) -> Result<(), ExecutorError> {
        self.inner.effect_success_output(id, node_id, output).await
    }

    async fn effect_failure_output(
        &self,
        id: Uuid,
        node_id: u32,
        error: Vec<u8>,
    ) -> Result<(), ExecutorError> {
        self.inner.effect_failure_output(id, node_id, error).await
    }

    async fn submit_history_event(&self, event: RunnerOut) -> Result<(), ExecutorError> {
        self.inner.submit_history_event(event).await
    }
}

fn main() -> iced::Result {
    let journey_id = Uuid::from_u128(0x5245_4445_5349_474e);
    let started = Instant::now();
    let client = AnimatedDemoClient::new(
        MockClient::builder()
            .on_flow_appearance(move |requested_id| {
                let appearance = (requested_id == journey_id)
                    .then(|| postcard::to_allocvec(&demo_appearance(started.elapsed())).unwrap());
                async move { Ok(appearance) }
            })
            .build(),
    );

    println!(
        "Opening an animated typed forward pass. Click any node to watch its live child flow."
    );
    BeamBuilder::new()
        .title("REDESIGN VERIFIED: typed ForwardOnly tensor pipeline")
        .window_size(1100.0, 700.0)
        .microdot_layout()
        .register_subpanel_animal::<NormalizeFeatures>()
        .register_subpanel_animal::<EncodeFeatures>()
        .register_subpanel_animal::<ClassifyFeatures>()
        .view_live::<RedesignDemo>(client, journey_id)
}
