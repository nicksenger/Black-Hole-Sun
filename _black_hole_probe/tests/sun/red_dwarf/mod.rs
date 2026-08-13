#[cfg(test)]
use futures::stream::StreamExt;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_sun::cell::{CellState, Primordium};
use black_hole_sun::ops::{SunOps, VoidInferOps};
use black_hole_sun::sun::{BlackHole, SunAppearance, SunNodeState, SunState, Unary};
use black_hole_sun::{
    EmissionId, InferenceRequest, ObjectId, QuarkClient, QuarkModelConfig, QuarkModelParams, Ray,
    TestQuarkServer, TestVoidServer, Transmission, VoidClient,
};
use black_hole_sun::{ModelConfig, OscillationSchedule};
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use jungle_sdk::JungleClient;
use postcard::to_allocvec;
use typosaurus::num::consts::*;
use uuid::Uuid;

use super::common::{init_tracing, make_client_endpoint, require_model_path, Generator, Policy};

#[derive(Clone)]
struct AlternatingEveryStep;

impl OscillationSchedule for AlternatingEveryStep {
    const PERIOD_STEPS: Option<u32> = Some(2);
    const TRAIN_STEPS: Option<u32> = Some(1);
    const PHASE_STEPS: Option<u32> = Some(0);
}

struct AlternatingModelConfig;

impl ModelConfig for AlternatingModelConfig {
    type Oscillation = AlternatingEveryStep;
    const FROZEN: Option<bool> = Some(false);
    const INFERENCE_LIMIT: Option<u32> = Some(0);
}

struct OscillatingCellAnimal;

#[jungle::animal(observe, id = 42, generation = 0)]
impl Animal for OscillatingCellAnimal {
    type State = CellState;
    type Seed = ObjectId;
    type Flow = Primordium<(), AlternatingModelConfig>;
}

impl Observe for OscillatingCellAnimal {
    type Appearance = Ray;

    fn observe(state: &Self::State) -> Self::Appearance {
        Ray {
            frozen: state.is_frozen,
        }
    }
}

type SingleOscillatingCell = Unary<U0, OscillatingCellAnimal, list![]>;
type RedDwarfSun = list![SingleOscillatingCell];

struct RedDwarfBlackHoleAnimal;

#[jungle::animal(observe, id = 43, generation = 0)]
impl Animal for RedDwarfBlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = <RedDwarfSun as BlackHole>::Sun<Generator, Policy, ()>;
}

impl Observe for RedDwarfBlackHoleAnimal {
    type Appearance = SunAppearance;

    fn observe(state: &Self::State) -> Self::Appearance {
        state.appearance()
    }
}

#[derive(Animals)]
struct RedDwarfAnimals(OscillatingCellAnimal, RedDwarfBlackHoleAnimal);

#[derive(Clone)]
struct RedDwarfJungle {
    void_client: VoidClient,
    quark_client: QuarkClient,
    client: Option<FusedClient>,
}

impl RedDwarfJungle {
    fn new(void_client: VoidClient, quark_client: QuarkClient) -> Self {
        Self {
            void_client,
            quark_client,
            client: None,
        }
    }

    fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
    }
}

impl Ecosystem for RedDwarfJungle {
    const NAME: &'static str = "red-dwarf-jungle";
    type Animals = RedDwarfAnimals;
}

#[async_trait]
impl VoidInferOps for RedDwarfJungle {
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
        let request_bytes = to_allocvec(&request).map_err(|error| format!("serialize: {error}"))?;
        let request_id = self.void_client.upload(request_bytes).await?;
        self.quark_client.infer(model_id, request_id).await
    }

    async fn reset_model(&self, model_id: Uuid) -> Result<(), String> {
        self.quark_client.reset(model_id).await
    }

    async fn checkpoint_model(&self, model_id: Uuid) -> Result<ObjectId, String> {
        self.quark_client.checkpoint(model_id).await
    }

    fn darken(&self, _prompt: &str) -> Result<Vec<black_hole_sun::DarkToken>, String> {
        Err("darken is not used by red_dwarf".to_string())
    }

    fn decode(&self, tokens: &[black_hole_sun::DarkToken]) -> String {
        tokens
            .iter()
            .map(|token| token.predicted.to_string())
            .collect::<Vec<_>>()
            .join(" ")
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
        let data = to_allocvec(&propagation).map_err(|error| format!("serialize: {error}"))?;
        self.void_client.upload_with(send_id, data).await?;
        Ok(())
    }
}

#[async_trait]
impl SunOps for RedDwarfJungle {
    async fn spawn_animal<A: Animal>(&self, seed: &A::Seed) -> Result<Uuid, String>
    where
        A::Id: AnimalIdValue,
        A::Generation: jungle_sdk::typosaurus::num::Unsigned,
        A::Seed: Send + Sync + Send,
    {
        let client = self.client.clone().expect("client not set");
        let handle = client.spawn::<A>(seed).await.map_err(|error| error.to_string())?;
        Ok(handle.journey_id)
    }
}

#[cfg(test)]
#[tokio::test]
async fn red_dwarf_child_ray_frozen_state_oscillates() {
    init_tracing();

    let model_path = match require_model_path("red_dwarf_child_ray_frozen_state_oscillates") {
        Some(path) => path,
        None => return,
    };

    let void_server = TestVoidServer::new()
        .serve()
        .await
        .expect("failed to start void server");
    let quark_server = TestQuarkServer::new(&model_path)
        .void_addr(void_server.local_addr())
        .serve()
        .await
        .expect("failed to start quark server");

    let endpoint = make_client_endpoint().await;
    let void_client = VoidClient::new(&endpoint, void_server.local_addr(), "localhost");
    let quark_client = QuarkClient::new(&endpoint, quark_server.local_addr(), "localhost");
    let mut jungle = RedDwarfJungle::new(void_client, quark_client);

    let client = FusedClient::builder()
        .build()
        .await
        .expect("fused client should build");
    jungle.set_client(client.clone());

    let parent_journey_id = client
        .spawn::<RedDwarfBlackHoleAnimal>(&())
        .await
        .expect("RedDwarfBlackHoleAnimal should spawn")
        .journey_id;
    let mut subscription = client
        .subscribe_step_updates(parent_journey_id, None)
        .await
        .expect("subscribe_step_updates should succeed");

    let worker_handles: Vec<_> = (0..2)
        .map(|_| {
            let worker = JungleWorker::new(jungle.clone(), client.clone());
            tokio::spawn(async move {
                let _ = worker.spawn().await;
            })
        })
        .collect();

    let child_journey_id = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if let Some(bytes) = client
                .animal_appearance(parent_journey_id)
                .await
                .expect("parent animal_appearance should succeed")
            {
                let appearance = postcard::from_bytes::<SunAppearance>(&bytes)
                    .expect("Sun appearance should deserialize");
                if let Some(node) = appearance
                    .nodes
                    .iter()
                    .find(|node| node.id == 0 && node.journey_id != Uuid::nil())
                {
                    break node.journey_id;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("child journey id should become available");

    let mut seen_propagation2_sequence = None;
    let mut skipped_first_propagation2 = false;
    let mut frozen_samples = Vec::new();
    let sample_result = tokio::time::timeout(Duration::from_secs(180), async {
        loop {
            if frozen_samples.len() >= 2 {
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
            let Some(node) = appearance.nodes.iter().find(|node| node.id == 0) else {
                continue;
            };
            if node.state != SunNodeState::Propagation2 {
                continue;
            }
            if seen_propagation2_sequence == Some(node.state_sequence) {
                continue;
            }
            seen_propagation2_sequence = Some(node.state_sequence);

            if !skipped_first_propagation2 {
                skipped_first_propagation2 = true;
                continue;
            }

            let child_ray = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let Some(bytes) = client
                        .animal_appearance(child_journey_id)
                        .await
                        .map_err(|error| format!("child animal_appearance failed: {error}"))?
                    {
                        let ray = postcard::from_bytes::<Ray>(&bytes)
                            .map_err(|error| format!("Ray should deserialize: {error}"))?;
                        break Ok::<Ray, String>(ray);
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .map_err(|error| format!("timed out waiting for child Ray appearance: {error}"))??;
            frozen_samples.push(child_ray.frozen);
        }
    })
    .await
    .expect("frozen-state sampling should complete");

    if let Err(error) = sample_result {
        let status = client
            .journey_details(parent_journey_id)
            .await
            .expect("journey_details should succeed");
        panic!("red_dwarf failed: {error}, status: {status:?}");
    }

    assert_eq!(
        frozen_samples,
        vec![false, true],
        "expected child Ray frozen state to flip after successive optimize steps"
    );

    for worker_handle in worker_handles {
        worker_handle.abort();
        let _ = worker_handle.await;
    }
    drop(client);
    void_server.abort();
    quark_server.abort();
}
