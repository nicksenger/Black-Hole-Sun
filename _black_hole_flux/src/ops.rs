//! Void + inference operations trait.

use serde::de::DeserializeOwned;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Re-exports — keep common spec types handy alongside the trait
// ---------------------------------------------------------------------------

pub use black_hole_spec::{Emission, EmissionId, InferenceRequest, ObjectId, Transmission};

/// Capability trait that guarantees a Jungle can talk to void and quark.
///
/// Implement this for your Jungle wrapper so the [`QuarkInfer`] effect can
/// download emissions, run inference, and upload results.
#[async_trait::async_trait]
pub trait VoidInferOps: Send + Sync {
    /// Download raw bytes from void by object id.
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String>;

    /// Download a deserialized `Emission<M>` from void by its object id.
    ///
    /// Default implementation delegates to [`download_raw`][Self::download_raw]
    /// and deserializes with postcard.
    async fn download_emission<M: Serialize + DeserializeOwned + Send>(
        &self,
        id: ObjectId,
    ) -> Result<Emission<M>, String> {
        let data = self.download_raw(id).await?;
        postcard::from_bytes(&data).map_err(|e| format!("postcard deserialize: {e}"))
    }

    /// Upload data to void and return the assigned object id.
    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String>;

    /// Upload data to void at a specific object id.
    ///
    /// Sun orchestration uses stable object ids as mailboxes shared with
    /// spawned cells, so these writes must preserve the requested id.
    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String>;

    /// Run quark inference on an emission stored at `input_id` in void.
    /// Returns the void id of the resulting `InferenceOutput`.
    async fn infer(&self, request: InferenceRequest) -> Result<ObjectId, String>;

    /// Perturb the associated quark's weights in the positive direction.
    ///
    /// The `seed` parameter controls the random perturbation for reproducibility.
    async fn perturb_up(&self, seed: u64) -> Result<(), String>;

    /// Perturb the associated quark's weights in the negative direction.
    async fn perturb_down(&self) -> Result<(), String>;

    /// Apply the QuZO optimization update using the up and down loss values.
    ///
    /// The quark uses the difference between `loss_up` and `loss_down` to
    /// estimate a gradient and update its weights.
    async fn optimize(&self, loss_up: f32, loss_down: f32) -> Result<(), String>;

    /// Wait for a [`Transmission`] from the void by object id.
    ///
    /// Downloads raw bytes and deserializes as `Transmission`.
    async fn wait_for_transmission(&self, id: ObjectId) -> Result<Transmission, String> {
        use tokio::time::{sleep, Duration};
        loop {
            match self.download_raw(id).await {
                Ok(data) => {
                    return postcard::from_bytes(&data)
                        .map_err(|e| format!("postcard deserialize: {e}"));
                }
                Err(e) => {
                    tracing::debug!(?id, error = %e, "download failed, retrying in 1s");
                }
            }
            sleep(Duration::from_secs(1)).await;
        }
    }

    /// Propagates the emission to the next cell - the implementor of
    /// VoidInferOps is responsible for converting the EmissionId into a
    /// Transmission::Propagation by supplying the recv and send void Ids
    /// for the next node.
    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String>;
}

#[async_trait::async_trait]
pub trait SunOps: Send + Sync {
    /// Spawn an animal of type `A` with the given seed and return the journey ID.
    ///
    /// The implementor is responsible for forwarding this to a
    /// [`JungleClient`](jungle_sdk::JungleClient).
    async fn spawn_animal<A>(&self, seed: &A::Seed) -> Result<uuid::Uuid, String>
    where
        A: jungle_sdk::Animal,
        A::Id: jungle_sdk::AnimalIdValue,
        A::Generation: typosaurus::num::Unsigned,
        A::Seed: Sync + Send;
}
