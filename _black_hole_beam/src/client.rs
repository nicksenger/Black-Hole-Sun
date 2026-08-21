//! A [`JungleClient`] wrapper shared between the main view and subpanels.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use jungle_sdk::JungleClient;
use uuid::Uuid;

#[derive(Clone)]
pub(crate) struct SharedJungleClient {
    inner: Arc<dyn JungleClient>,
}

impl SharedJungleClient {
    pub(crate) fn new(inner: Arc<dyn JungleClient>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl JungleClient for SharedJungleClient {
    async fn spawn<A>(
        &self,
        _seed: &A::Seed,
    ) -> Result<jungle_sdk::JourneyHandle, jungle_sdk::ExecutorError>
    where
        Self: Sized,
        A: jungle_sdk::SpawnableAnimal,
        A::Seed: Sync,
    {
        Err(jungle_sdk::ExecutorError::ClientTransport(
            "shared beam client does not support spawn".to_string(),
        ))
    }

    async fn journey_history(
        &self,
        id: Uuid,
    ) -> Result<Vec<jungle_sdk::RunnerOut>, jungle_sdk::ExecutorError> {
        self.inner.journey_history(id).await
    }

    async fn journey_replay_page(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
        snapshot_end_sequence_id: Option<u64>,
        limit: u32,
    ) -> Result<jungle_sdk::JourneyReplayPage, jungle_sdk::ExecutorError> {
        self.inner
            .journey_replay_page(
                journey_id,
                after_sequence_id,
                snapshot_end_sequence_id,
                limit,
            )
            .await
    }

    async fn list_journeys(
        &self,
        namespace: String,
    ) -> Result<Vec<jungle_sdk::JourneyRecord>, jungle_sdk::ExecutorError> {
        self.inner.list_journeys(namespace).await
    }

    async fn subscribe_step_updates(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
    ) -> Result<jungle_sdk::client::JourneyUpdateSubscription, jungle_sdk::ExecutorError> {
        self.inner
            .subscribe_step_updates(journey_id, after_sequence_id)
            .await
    }

    async fn journey_details(
        &self,
        id: Uuid,
    ) -> Result<jungle_sdk::JourneyStatus, jungle_sdk::ExecutorError> {
        self.inner.journey_details(id).await
    }

    async fn animal_appearance(
        &self,
        id: Uuid,
    ) -> Result<Option<Vec<u8>>, jungle_sdk::ExecutorError> {
        self.inner.animal_appearance(id).await
    }

    async fn animal_appearance_update(
        &self,
        id: Uuid,
        data: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.animal_appearance_update(id, data).await
    }

    async fn perturb_animal(
        &self,
        id: Uuid,
        payload: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.perturb_animal(id, payload).await
    }

    async fn claim_animal_perturbation(
        &self,
        id: Uuid,
    ) -> Result<Option<jungle_sdk::ClaimedPerturbable>, jungle_sdk::ExecutorError> {
        self.inner.claim_animal_perturbation(id).await
    }

    async fn ack_animal_perturbation(
        &self,
        id: Uuid,
        perturbation_id: u64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .ack_animal_perturbation(id, perturbation_id)
            .await
    }

    async fn heartbeat_journey_lease(
        &self,
        journey_id: Uuid,
        owner_id: Uuid,
        lease_ttl_ms: i64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .heartbeat_journey_lease(journey_id, owner_id, lease_ttl_ms)
            .await
    }

    async fn poll_owner_wake(
        &self,
        owner_id: Uuid,
    ) -> Result<Option<jungle_sdk::OwnerWake>, jungle_sdk::ExecutorError> {
        self.inner.poll_owner_wake(owner_id).await
    }

    async fn schedule_sleep_timer(
        &self,
        journey_id: Uuid,
        timer_id: Uuid,
        wake_at_unix_ms: i64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .schedule_sleep_timer(journey_id, timer_id, wake_at_unix_ms)
            .await
    }

    async fn complete_journey(&self, id: Uuid) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.complete_journey(id).await
    }

    async fn dead_journey(&self, id: Uuid) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.dead_journey(id).await
    }

    async fn poll_timers(&self) -> Result<Option<()>, jungle_sdk::ExecutorError> {
        self.inner.poll_timers().await
    }

    async fn poll_work(
        &self,
        supported_animals: Vec<jungle_sdk::SupportedAnimal>,
    ) -> Result<Option<jungle_sdk::Work>, jungle_sdk::ExecutorError> {
        self.inner.poll_work(supported_animals).await
    }

    async fn wait_for_worker_wake(
        &self,
        owner_id: Uuid,
        supported_animals: Vec<jungle_sdk::SupportedAnimal>,
        timeout: Duration,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .wait_for_worker_wake(owner_id, supported_animals, timeout)
            .await
    }

    async fn effect_input(
        &self,
        id: Uuid,
        node_id: u32,
        input: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_input(id, node_id, input).await
    }

    async fn effect_success_output(
        &self,
        id: Uuid,
        node_id: u32,
        output: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_success_output(id, node_id, output).await
    }

    async fn effect_failure_output(
        &self,
        id: Uuid,
        node_id: u32,
        err: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_failure_output(id, node_id, err).await
    }

    async fn submit_history_event(
        &self,
        event: jungle_sdk::RunnerOut,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.submit_history_event(event).await
    }
}
