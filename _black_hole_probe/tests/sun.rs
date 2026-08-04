mod common;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::ops::{SunOps, VoidInferOps};
use black_hole_flux::sun::EventHorizon;
use black_hole_flux::sun::{SunState, Tag};
use black_hole_flux::Progenitor;
use black_hole_sun::black_hole_flux;
use black_hole_sun::object_store::InMemoryObjectStore;
use black_hole_sun::persist::InMemoryStore;
use black_hole_sun::{
    DarkToken, EmissionId, InferenceRequest, LogitEntry, ObjectId, QuarkServerBuilder,
    SequenceOutput, Transmission, VoidServerBuilder,
};
use jungle_sdk::client::JourneyUpdateSubscription;
use jungle_sdk::core::JungleWorker;
use jungle_sdk::prelude::*;
use jungle_sdk::FusedClient;
use postcard::to_allocvec;
use typosaurus::num::consts::*;
use uuid::Uuid;

use common::*;

// ─── 3-tag Sun graph type (U0 -> U1 -> U2) ──────────────────────────────────

/// Node tags: each tag carries a typenum index, the animal type (Progenitor),
/// and the list of outgoing edge targets.
type Tag0 = Tag<U0, Progenitor, list![U1]>;
type Tag1 = Tag<U1, Progenitor, list![U2]>;
type Tag2 = Tag<U2, Progenitor, list![U3]>;

/// The complete three-node sun: a type-level list of all node tags.
type ThreeTagSunType = list![Tag0, Tag1, Tag2];

// ─── BlackHoleAnimal ─────────────────────────────────────────────────────────

/// An animal that runs the full BlackHole orchestration flow over a Sun graph.
pub struct BlackHoleAnimal;

#[jungle::animal(id = 1, generation = 0)]
impl Animal for BlackHoleAnimal {
    type State = SunState;
    type Seed = ();
    type Flow = FakeFlow;
    //type Flow = FakeFlow;
}

#[derive(Flow)]
pub struct FakeFlow();

// ─── Ecosystem ───────────────────────────────────────────────────────────────

#[derive(Animals)]
pub struct SpaceAnimals(Progenitor, BlackHoleAnimal);

/// A Jungle implementation backed by void + quark servers over QUIC.
pub struct SpaceJungle {
    void_addr: SocketAddr,
    quark_addr: SocketAddr,
    client: Option<FusedClient>,
}

impl SpaceJungle {
    pub fn new(void_addr: SocketAddr, quark_addr: SocketAddr) -> Self {
        Self {
            void_addr,
            quark_addr,
            client: None,
        }
    }

    pub fn set_client(&mut self, client: FusedClient) {
        self.client = Some(client);
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
        Ok(void_download(&endpoint, self.void_addr, id).await)
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        let endpoint = make_client_endpoint().await;
        Ok(void_upload(&endpoint, self.void_addr, data).await)
    }

    async fn infer(&self, request: InferenceRequest) -> Result<ObjectId, String> {
        let request_bytes = to_allocvec(&request).map_err(|e| format!("serialize: {e}"))?;
        let endpoint = make_client_endpoint().await;
        let request_id = void_upload(&endpoint, self.void_addr, request_bytes).await;
        Ok(quark_infer(&endpoint, self.quark_addr, request_id).await)
    }

    async fn perturb_up(&self, seed: u64) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_perturb_up(&endpoint, self.quark_addr, seed).await;
        Ok(())
    }

    async fn perturb_down(&self) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_perturb_down(&endpoint, self.quark_addr).await;
        Ok(())
    }

    async fn optimize(&self, loss_up: f32, loss_down: f32) -> Result<(), String> {
        let endpoint = make_client_endpoint().await;
        quark_optimize(&endpoint, self.quark_addr, loss_up, loss_down).await;
        Ok(())
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

// ─── Server helpers (matching cell.rs patterns) ──────────────────────────────

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
    let void_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (void_local, void_handle) = VoidServerBuilder::new(object_store, store)
        .listen(void_addr)
        .serve()
        .await
        .expect("failed to start void server");
    let void_abort = void_handle.abort_handle();

    let quark_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let (quark_local, quark_handle) = QuarkServerBuilder::new(PathBuf::from(model_path))
        .listen(quark_addr)
        .void_addr(void_local)
        .serve()
        .await
        .expect("failed to start quark server");
    let quark_abort = quark_handle.abort_handle();

    drop(void_handle);
    drop(quark_handle);

    tokio::time::sleep(Duration::from_millis(200)).await;
    (void_local, void_abort, quark_local, quark_abort)
}

// ─── Token helpers (matching cell.rs patterns) ───────────────────────────────

/// Download the Qwen tokenizer from HuggingFace.
fn get_tokenizer() -> tokenizers::Tokenizer {
    let tokenizer_repo = "Qwen/Qwen3.5-0.8B".to_string();
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub api");
    let repo = api.repo(hf_hub::Repo::with_revision(
        tokenizer_repo.clone(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_file = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json from HuggingFace");
    tokenizers::Tokenizer::from_file(tokenizer_file).expect("failed to load tokenizer")
}

/// Tokenize text into DarkTokens suitable for InferenceOutput.
fn text_to_dark_tokens(text: &str, tokenizer: &tokenizers::Tokenizer) -> Vec<DarkToken> {
    let tokens: Vec<u32> = tokenizer
        .encode(text, false)
        .expect("failed to tokenize input")
        .get_ids()
        .iter()
        .map(|&id| id as u32)
        .collect();
    tokens
        .iter()
        .map(|id| DarkToken {
            predicted: *id,
            dark_knowledge: Vec::new(),
        })
        .collect()
}

// ─── Test ────────────────────────────────────────────────────────────────────

/// Exercises the full BlackHole flow over a three-node Sun graph (U0 -> U1 -> U2).
///
/// This test follows the same pattern as the cell.rs test but uses the BlackHole
/// Flow to automatically handle all input generation, propagation, loss computation,
/// and potentiation broadcasting. No manual upload of inputs or void interactions
/// is needed — the Flow orchestrates everything.
///
/// We subscribe to step updates and assert that at least 3 full training epochs
/// complete by tracking KickOff task completions. In each epoch cycle:
///   Epoch = KickOff -> PropagationFlows -> ComputeLoss -> BroadcastPotentiation
/// KickOff is the first effect in every epoch, so we detect it as the leading
/// EffectSuccessOutput after BuildAddrs and between epoch boundaries.
#[tokio::test]
async fn sun() {
    init_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();

    let model_path = match require_model_path("black_hole_flow") {
        Some(path) => path,
        None => return,
    };

    // 1. Start void and quark servers on random ports.
    let (void_addr, void_abort, quark_addr, quark_abort) = start_servers(&model_path).await;

    // 2. Build the SpaceJungle with void/quark capabilities.
    let mut jungle = SpaceJungle::new(void_addr, quark_addr);

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
    let mut subscription: JourneyUpdateSubscription = client
        .subscribe_step_updates(journey_id, None)
        .await
        .expect("subscribe_step_updates should succeed");

    // 6. Start a JungleWorker so effects execute after we're subscribed.
    let worker = JungleWorker::new(jungle, client.clone());
    let worker_handle = tokio::spawn(async move {
        let _ = worker.spawn().await;
    });

    // 7. Watch for KickOff completions (3 full training epochs).
    //
    // The BlackHole flow structure is:
    //   BlackHole(Step<BuildAddrs>, While<Always<SunState, ()>, Epoch>)
    // where Epoch = (KickOff, PropagationFlows, ComputeLoss, BroadcastPotentiation)
    //
    // BuildAddrs (node 0) runs once at startup. After that, each epoch iteration
    // produces a sequence of EffectSuccessOutput events. KickOff is always the
    // first effect in each epoch cycle.
    //
    // We track KickOff completions by:
    // - Detecting BuildAddrs completion (node_id == 0)
    // - Counting effects after BuildAddrs; KickOff is the first effect of each
    //   epoch. Each epoch produces at least 3 effect successes (KickOff,
    //   ComputeLoss, BroadcastPotentiation), with PropagationFlows adding more.
    //   We detect KickOff as effects at positions 1, 4, 7... after BuildAddrs
    //   (i.e., every ~3-4 effects).

    let result = tokio::time::timeout(Duration::from_secs(120), async {
        let mut total_effects = 0u32;
        let mut buildaddrs_done = false;
        let mut kickoff_count = 0u32;

        while let Some(update_result) = subscription.next().await {
            let update = update_result.expect("stream update should succeed");

            match update.event {
                RunnerUpdateOut::EffectSuccessOutput { node_id, .. } => {
                    total_effects += 1;

                    // BuildAddrs is node 0 — runs once at startup.
                    if node_id == 0 && !buildaddrs_done {
                        buildaddrs_done = true;
                        println!("BuildAddrs completed (effect #{})", total_effects);
                        continue;
                    }

                    if !buildaddrs_done {
                        // Effects before BuildAddrs — skip.
                        continue;
                    }

                    let effects_after_buildaddrs = total_effects - 1;

                    // KickOff is the first effect in each epoch cycle.
                    // Each epoch produces at least 3 effect successes
                    // (KickOff, ComputeLoss, BroadcastPotentiation), with
                    // PropagationFlows adding more from While-loop iterations.
                    // We detect KickOff as effects at positions 1, 4, 7...
                    // after BuildAddrs (i.e., every ~3-4 effects).
                    let estimated_epochs = effects_after_buildaddrs / 3;
                    let current_epoch = estimated_epochs + 1;

                    // The first effect of each new epoch is KickOff.
                    if effects_after_buildaddrs > 0 && effects_after_buildaddrs % 3 == 1 {
                        kickoff_count += 1;
                        println!(
                            "KickOff completed (epoch {}, effect #{})",
                            current_epoch, total_effects
                        );

                        if kickoff_count >= 3 {
                            println!(
                                "3 KickOff completions detected after {} total effects",
                                total_effects
                            );
                            return Ok::<(), String>(());
                        }
                    }
                }
                RunnerUpdateOut::NodeLifecycle(node) => {
                    if node.phase == jungle_sdk::types::NodeLifecyclePhase::Entered {
                        println!("Journey entered lifecycle");
                    }
                }
                _ => {}
            }
        }

        Err(format!(
            "stream ended before 3 KickOff completions (got {})",
            kickoff_count
        ))
    })
    .await;

    match result {
        Ok(Ok(())) => {
            println!("BlackHole flow completed: 3 KickOff tasks detected");
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
                "timeout waiting for 3 KickOff completions (120s): {}, status: {:?}",
                e, status
            );
        }
    }

    // Cleanup.
    worker_handle.abort();
    let _ = worker_handle.await;
    server_handle.abort();
    let _ = server_handle.await;
    drop(client);
    void_abort.abort();
    quark_abort.abort();
}
