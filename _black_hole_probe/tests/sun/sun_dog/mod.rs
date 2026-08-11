mod action;
mod effect;

#[cfg(test)]
use std::collections::HashMap;

use jungle_sdk::core::JungleWorker;
use jungle_sdk::FusedClient;
use jungle_sdk::JungleClient;
#[cfg(test)]
use uuid::Uuid;

use super::common::{init_tracing, make_client_endpoint};
#[cfg(test)]
use super::diamond_dog::{exercise_diamond_dog, FUSED_EMISSION, LEFT_EMISSION, RIGHT_EMISSION};
use super::diamond_dog::{ExpandedBlackHoleAnimal, ProbeSpaceJungle};
use black_hole_sun::TestVoidServer;

/// Exercises an extra diamond layer ending in a third binary fusion:
///
/// `Input -> [L0, R0]`, `L0 -> [L1, R1]`, `R0 -> [L2, R2]`,
/// `[L1, R1] -> F0`, `[L2, R2] -> F1`, and `[F0, F1] -> F2`.
#[cfg(test)]
#[tokio::test]
async fn sun_dog() {
    const EPOCHS: usize = 3;
    const PROPAGATION_PASSES: usize = 2;
    const FIRST_LAYER_FUSIONS: usize = 2;
    const FINAL_LAYER_FUSIONS: usize = 1;
    const FUSION_TRANSFORMS: usize =
        EPOCHS * PROPAGATION_PASSES * (FIRST_LAYER_FUSIONS + FINAL_LAYER_FUSIONS);

    // Ten vertices own thirteen input ports: seven unary and six binary.
    let observed = exercise_diamond_dog::<ExpandedBlackHoleAnimal>("sun_dog", 10, 13, EPOCHS).await;
    assert!(
        observed.len() >= FUSION_TRANSFORMS,
        "expected {FUSION_TRANSFORMS} fusion transforms, observed {observed:?}"
    );

    let completed_epochs = &observed[..FUSION_TRANSFORMS];
    let first_layer_pair = (
        Uuid::from_u128(LEFT_EMISSION),
        Uuid::from_u128(RIGHT_EMISSION),
    );
    let final_layer_pair = (
        Uuid::from_u128(FUSED_EMISSION),
        Uuid::from_u128(FUSED_EMISSION),
    );
    assert!(
        completed_epochs
            .iter()
            .all(|(_, p1, p2)| (*p1, *p2) == first_layer_pair || (*p1, *p2) == final_layer_pair),
        "unexpected fusion inputs in completed epochs: {completed_epochs:?}"
    );
    assert_eq!(
        completed_epochs
            .iter()
            .filter(|(_, p1, p2)| (*p1, *p2) == first_layer_pair)
            .count(),
        EPOCHS * PROPAGATION_PASSES * FIRST_LAYER_FUSIONS,
        "both first-layer fusions should run on every pass"
    );
    assert_eq!(
        completed_epochs
            .iter()
            .filter(|(_, p1, p2)| (*p1, *p2) == final_layer_pair)
            .count(),
        EPOCHS * PROPAGATION_PASSES * FINAL_LAYER_FUSIONS,
        "the final fusion should run on every pass"
    );

    let mut inputs_by_transform = HashMap::new();
    for &(transform_id, p1, p2) in &observed {
        assert_ne!(
            transform_id,
            Uuid::nil(),
            "fusion transform ID should be generated"
        );
        if let Some(previous_inputs) = inputs_by_transform.insert(transform_id, (p1, p2)) {
            assert_eq!(
                previous_inputs,
                (p1, p2),
                "fusion transform {transform_id} did not retain a stable identity"
            );
        }
    }
    assert_eq!(
        inputs_by_transform.len(),
        FIRST_LAYER_FUSIONS + FINAL_LAYER_FUSIONS,
        "each fusion journey should have a distinct transform ID"
    );
    assert_eq!(
        inputs_by_transform
            .values()
            .filter(|inputs| **inputs == first_layer_pair)
            .count(),
        FIRST_LAYER_FUSIONS,
        "both first-layer transforms should have distinct stable IDs"
    );
    assert_eq!(
        inputs_by_transform
            .values()
            .filter(|inputs| **inputs == final_layer_pair)
            .count(),
        FINAL_LAYER_FUSIONS,
        "the final-layer transform should have its own stable ID"
    );
}

/// Runs the expanded diamond Sun indefinitely with a live Black Hole Beam viewer.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn run_beam() {
    init_tracing();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Tokio runtime should build");
    let (client, journey_id, _void_server) = runtime.block_on(async {
        let void_server = TestVoidServer::new()
            .serve()
            .await
            .expect("failed to start void server");
        let void_addr = void_server.local_addr();

        let endpoint = make_client_endpoint().await;
        let void_client = black_hole_sun::VoidClient::new(&endpoint, void_addr, "localhost");
        let mut jungle = ProbeSpaceJungle::new(void_client);
        let client = FusedClient::builder()
            .build()
            .await
            .expect("fused client should build");
        jungle.set_client(client.clone());

        let journey_id = client
            .spawn::<ExpandedBlackHoleAnimal>(&())
            .await
            .expect("ExpandedBlackHoleAnimal should spawn")
            .journey_id;
        println!("Spawned ExpandedBlackHoleAnimal journey: {journey_id}");

        // One worker per journey: ten graph vertices plus the parent.
        let _worker_handles: Vec<_> = (0..11)
            .map(|_| {
                let worker = JungleWorker::new(jungle.clone(), client.clone());
                tokio::spawn(async move {
                    let _ = worker.spawn().await;
                })
            })
            .collect();

        (client, journey_id, void_server)
    });

    black_hole_beam::BeamBuilder::new()
        .dot_layout()
        .view_live::<ExpandedBlackHoleAnimal>(client, journey_id)
        .expect("Black Hole Beam should run");
}

#[cfg(test)]
#[test]
#[ignore]
fn beam_test() {
    super::run_beam_example("beam");
}
