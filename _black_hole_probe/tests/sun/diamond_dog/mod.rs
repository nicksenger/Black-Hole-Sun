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
use black_hole_sun::sun::{Binary, BlackHole, SunAppearance, SunNodeState, SunState, Unary};
use black_hole_sun::{
    EmissionId, InferenceRequest, ObjectId, TestVoidServer, Tokenizer, Transmission, VoidClient,
};
use black_hole_sun::{Fusion, FusionSeed, FusionState};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::to_allocvec;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::*;

pub(super) const LEFT_EMISSION: u128 = 1;
pub(super) const RIGHT_EMISSION: u128 = 2;
pub(super) const FUSED_EMISSION: u128 = 3;

pub(super) type FusionObservation = (Uuid, ObjectId, ObjectId);

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
pub(super) struct FinishEpoch;

#[derive(Flow)]
pub(super) struct TestCellEpoch<Transform>(
    Step<WaitForPropagationAction>,
    Transform,
    Step<Transmit>,
    Step<WaitForPropagationAction>,
    Transform,
    Step<Transmit>,
    Step<WaitForPotentiationAction>,
    Step<FinishEpoch>,
);

pub(super) struct AlwaysEpoch;

impl Predicate<(&CellState, &())> for AlwaysEpoch {
    fn eval(_input: &(&CellState, &())) -> bool {
        true
    }
}

#[derive(Flow)]
pub(super) struct TestCellFlow<Transform>(
    Step<InitRecvId>,
    While<AlwaysEpoch, TestCellEpoch<Transform>>,
);

pub(super) struct PassEmission;
pub(super) struct MarkLeft;
pub struct DelayedLeftEffect;
pub(super) struct MarkRight;

pub(super) struct RootAnimal;

#[jungle::animal(id = 0, generation = 0)]
impl Animal for RootAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<PassEmission>>;
}

pub(super) struct LeftAnimal;

#[jungle::animal(id = 2, generation = 0)]
impl Animal for LeftAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<MarkLeft>>;
}

pub(super) struct RightAnimal;

#[jungle::animal(id = 3, generation = 0)]
impl Animal for RightAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<MarkRight>>;
}

pub(super) struct SinkAnimal;

#[jungle::animal(id = 5, generation = 0)]
impl Animal for SinkAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = TestCellFlow<Step<PassEmission>>;
}

// ─── Explicit fusion transform animal ────────────────────────────────────────

pub(super) trait FusionProbeOps: Send + Sync {
    fn record_fusion_inputs(&self, transform_id: Uuid, p1: ObjectId, p2: ObjectId);
}

pub struct RecordFusionInputsEffect;
pub(super) struct RecordFusionInputs;

#[derive(Flow)]
pub(super) struct FusionTransform(Step<RecordFusionInputs>);

pub(super) struct FusionAnimal;

#[jungle::animal(id = 4, generation = 0)]
impl Animal for FusionAnimal {
    type State = FusionState;
    type Seed = FusionSeed;
    type Flow = Fusion<FusionTransform>;
}

// ─── BlackHoleAnimal ─────────────────────────────────────────────────────────

/// An animal that runs the full BlackHole orchestration flow over a Sun graph.
pub(super) struct BlackHoleAnimal;

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
pub(super) struct ExpandedBlackHoleAnimal;

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
pub(super) struct ProbeSpaceAnimals(
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
pub(super) struct ProbeSpaceJungle {
    void_client: VoidClient,
    tokenizer: Arc<OnceLock<Result<Tokenizer, String>>>,
    client: Option<FusedClient>,
    pub(super) potentiation_writes: Arc<AtomicUsize>,
    pub(super) fusion_inputs: Arc<Mutex<Vec<FusionObservation>>>,
}

impl ProbeSpaceJungle {
    pub(super) fn new(void_client: VoidClient) -> Self {
        Self {
            void_client,
            tokenizer: Arc::new(OnceLock::new()),
            client: None,
            potentiation_writes: Arc::new(AtomicUsize::new(0)),
            fusion_inputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub(super) fn set_client(&mut self, client: FusedClient) {
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
        self.void_client.upload_with(send_id, data).await.unwrap();
        Ok(())
    }
}

#[async_trait]
impl SunOps for ProbeSpaceJungle {
    async fn spawn_animal<A: Animal>(&self, seed: &A::Seed) -> Result<Uuid, String>
    where
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Send + Sync + Send,
    {
        let client = self.client.clone().expect("client not set");
        let handle = client.spawn::<A>(seed).await.map_err(|e| e.to_string())?;
        Ok(handle.journey_id)
    }
}

// ─── Harness ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) async fn exercise_diamond_dog<A>(
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

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let void_addr = void_server.local_addr();
    let endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&endpoint, void_addr, "localhost");
    let mut jungle = ProbeSpaceJungle::new(void_client);
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
    void_server.abort();

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
