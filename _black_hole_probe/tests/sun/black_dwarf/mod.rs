mod action;
mod effect;

use jungle_sdk::core::JungleWorker;
use jungle_sdk::FusedClient;
use jungle_sdk::JungleClient;

#[cfg(test)]
use super::common::require_model_path;
use super::common::{init_tracing, make_client_endpoint};
#[cfg(test)]
use super::dark_star::{exercise_epoch, ProgenitorBlackHole};
use super::dark_star::{BlackDwarfBlackHole, SpaceJungle, PROGENITOR_NODE_COUNT};
use black_hole_sun::{TestQuarkServer, TestVoidServer};

/// Runs the same U0 -> U1 -> U2 Sun topology as `diamond_dog`, with real Progenitor
/// cells backed by a quark model.
#[cfg(test)]
#[ignore]
#[tokio::test]
async fn primordia() {
    init_tracing();

    let model_path = match require_model_path("primordia") {
        Some(path) => path,
        None => return,
    };
    exercise_epoch::<ProgenitorBlackHole>(
        "Progenitor Sun",
        &model_path,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        0,
    )
    .await;
}

/// Runs the same topology as `primordia`, but with dark_star's generator/policy.
#[cfg(test)]
#[ignore]
#[tokio::test]
async fn test_black_dwarf() {
    init_tracing();

    let model_path = match require_model_path("black_dwarf") {
        Some(path) => path,
        None => return,
    };

    exercise_epoch::<BlackDwarfBlackHole>(
        "black_dwarf Sun",
        &model_path,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        PROGENITOR_NODE_COUNT,
        0,
    )
    .await;
}

/// Runs the black_dwarf Sun continuously without the Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_continuous_black_dwarf() {
    init_tracing();

    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to run continuous_black_dwarf");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    runtime.block_on(async {
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
        let void_client = black_hole_sun::VoidClient::new(&endpoint, void_addr, "localhost");
        let quark_client = black_hole_sun::QuarkClient::new(&endpoint, quark_addr, "localhost");
        let mut jungle = SpaceJungle::new(void_client, quark_client, PROGENITOR_NODE_COUNT);
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

        println!("Running black_dwarf continuously. Press Ctrl-C to stop.");
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl-C signal listener should install");
    });
}

/// Runs the black_dwarf Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam_black_dwarf() {
    init_tracing();

    let model_path = std::env::var("BLACK_HOLE_PROBE_MODEL_PATH")
        .expect("BLACK_HOLE_PROBE_MODEL_PATH must be set to run beam_black_dwarf");

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
        let void_client = black_hole_sun::VoidClient::new(&endpoint, void_addr, "localhost");
        let quark_client = black_hole_sun::QuarkClient::new(&endpoint, quark_addr, "localhost");
        let mut jungle = SpaceJungle::new(void_client, quark_client, PROGENITOR_NODE_COUNT);
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

        (client, journey_id, void_server, quark_server)
    });

    black_hole_beam::BeamBuilder::new()
        .dot_layout()
        .view_live::<BlackDwarfBlackHole>(client, journey_id)
        .expect("Black Hole Beam should run");
}

#[cfg(test)]
#[test]
#[ignore]
fn continuous_black_dwarf() {
    super::run_beam_example("continuous_black_dwarf");
}

#[cfg(test)]
#[test]
#[ignore]
fn beam_black_dwarf() {
    super::run_beam_example("beam_black_dwarf");
}
