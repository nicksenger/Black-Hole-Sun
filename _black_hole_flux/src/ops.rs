//! Void + inference operations trait.

use serde::de::DeserializeOwned;
use serde::Serialize;

const DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS: u64 = 30_000;

// ---------------------------------------------------------------------------
// Re-exports — keep common spec types handy alongside the trait
// ---------------------------------------------------------------------------

pub use black_hole_spec::{
    ArtifactRef, DarkToken, Emission, EmissionId, InferenceOutput, InferenceOutputId,
    InferenceRequest, MassModelConfig, MassModelParams, ObjectId, ObjectRef, Transmission,
};

use black_hole_contract::{QwenDarkInference, TensorContract};

use crate::AtomError;

// ---------------------------------------------------------------------------
// Generic Stage 2 capabilities
// ---------------------------------------------------------------------------

/// Persistence and resolution of raw or typed committed artifacts.
#[async_trait::async_trait]
pub trait VoidOps: Send + Sync {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String>;

    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        use tokio::time::{sleep, Duration, Instant};

        if timeout_ms == 0 {
            return Ok(None);
        }

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            match self.download_raw(id).await {
                Ok(data) => return Ok(Some(data)),
                Err(error) => {
                    let now = Instant::now();
                    if now >= deadline {
                        tracing::debug!(?id, error = %error, "download wait timed out");
                        return Ok(None);
                    }
                    sleep(
                        deadline
                            .saturating_duration_since(now)
                            .min(Duration::from_secs(1)),
                    )
                    .await;
                }
            }
        }
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String>;

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String>;

    async fn persist<T>(&self, value: &T) -> Result<ObjectRef<T>, String>
    where
        T: Serialize + Sync,
    {
        let bytes = postcard::to_allocvec(value).map_err(|error| error.to_string())?;
        self.upload_to_void(bytes).await.map(ObjectRef::new)
    }

    async fn resolve<T>(&self, reference: ObjectRef<T>) -> Result<T, String>
    where
        T: DeserializeOwned + Send,
    {
        let bytes = self.download_raw(reference.id()).await?;
        postcard::from_bytes(&bytes).map_err(|error| error.to_string())
    }

    async fn persist_artifact<T>(&self, value: &T) -> Result<ArtifactRef<T>, String>
    where
        T: Serialize + Sync,
    {
        self.persist(value).await.map(ArtifactRef::committed)
    }

    async fn resolve_artifact<T>(&self, reference: &ArtifactRef<T>) -> Result<T, String>
    where
        T: DeserializeOwned + Send,
    {
        self.resolve(reference.object_ref()).await
    }

    /// Wait for one operation-typed artifact delivery.
    ///
    /// The payload marker is compile-time only, so the wire representation
    /// remains the three object IDs used by the scheduler and destination
    /// cell.
    async fn wait_for_artifact_delivery<T>(
        &self,
        id: ObjectId,
    ) -> Result<black_hole_spec::ArtifactDelivery<T>, String>
    where
        T: Send,
    {
        loop {
            match self
                .download_raw_wait(id, DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS)
                .await
            {
                Ok(Some(data)) => {
                    return postcard::from_bytes(&data)
                        .map_err(|error| format!("postcard deserialize: {error}"));
                }
                Ok(None) => tracing::debug!(
                    ?id,
                    timeout_ms = DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS,
                    "artifact delivery wait timed out, retrying"
                ),
                Err(error) => {
                    tracing::debug!(?id, %error, "artifact delivery wait failed, retrying")
                }
            }
        }
    }

    /// Wait for a program-selected control payload.
    async fn wait_for_operational_control<C>(
        &self,
        id: ObjectId,
    ) -> Result<black_hole_spec::OperationalControl<C>, String>
    where
        C: DeserializeOwned + Send,
    {
        loop {
            match self
                .download_raw_wait(id, DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS)
                .await
            {
                Ok(Some(data)) => {
                    return postcard::from_bytes(&data)
                        .map_err(|error| format!("postcard deserialize: {error}"));
                }
                Ok(None) => tracing::debug!(
                    ?id,
                    timeout_ms = DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS,
                    "operational control wait timed out, retrying"
                ),
                Err(error) => {
                    tracing::debug!(?id, %error, "operational control wait failed, retrying")
                }
            }
        }
    }

    async fn download_emission<M, T>(&self, id: EmissionId<T>) -> Result<Emission<M, T>, String>
    where
        M: Serialize + DeserializeOwned + Send,
        T: Send,
    {
        let bytes = self.download_raw(id.id()).await?;
        postcard::from_bytes(&bytes).map_err(|error| format!("postcard deserialize: {error}"))
    }
}

/// Start, forward, and shutdown capabilities for one tensor operation
/// contract. Input and output artifact types are derived from `Op`.
#[async_trait::async_trait]
pub trait MassOps<Op>: Send + Sync
where
    Op: TensorContract,
{
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String>;

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<Op::Input>,
    ) -> Result<ArtifactRef<Op::Output>, String>;

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String>;
}

#[async_trait::async_trait]
pub trait ResetOps<Op: TensorContract>: Send + Sync {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String>;
}

#[async_trait::async_trait]
pub trait PerturbOps<Op: TensorContract>: Send + Sync {
    async fn perturb_up_operation(&self, instance_id: ObjectId, seed: u64) -> Result<(), String>;
    async fn perturb_down_operation(&self, instance_id: ObjectId) -> Result<(), String>;
}

#[async_trait::async_trait]
pub trait OptimizeOps<Op: TensorContract>: Send + Sync {
    async fn optimize_operation(
        &self,
        instance_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String>;
}

#[async_trait::async_trait]
pub trait CheckpointOps<Op: TensorContract>: Send + Sync {
    async fn checkpoint_operation(&self, instance_id: ObjectId) -> Result<ObjectId, String>;
}

#[async_trait::async_trait]
pub trait FuseOps<Op: TensorContract>: Send + Sync {
    async fn fuse_operation(
        &self,
        instance_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String>;
}

/// Qwen/LLM-specific compatibility surface. Tokenization and postcard
/// conversion deliberately live outside the generic tensor capabilities.
#[async_trait::async_trait]
pub trait QwenAdapterOps: Send + Sync {
    fn darken(&self, prompt: &str) -> Result<Vec<DarkToken>, String>;
    fn decode(&self, tokens: &[DarkToken]) -> String;

    async fn qwen_forward(
        &self,
        instance_id: ObjectId,
        request: InferenceRequest,
    ) -> Result<ObjectRef<InferenceOutput>, String>;
}

/// Capability trait that guarantees a Jungle can talk to void and mass.
///
/// Implement this for your Jungle wrapper so the [`MassInfer`] effect can
/// download emissions, run inference, and upload results.
#[async_trait::async_trait]
pub trait VoidInferOps: Send + Sync {
    /// Download raw bytes from void by object id.
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String>;

    /// Download raw bytes from void, waiting up to `timeout_ms` for the object to appear.
    ///
    /// Default behavior preserves legacy 1-second polling against `download_raw` for
    /// implementors that don't support server-side long-polling.
    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        use tokio::time::{sleep, Duration, Instant};

        if timeout_ms == 0 {
            return Ok(None);
        }

        let timeout = Duration::from_millis(timeout_ms);
        let deadline = Instant::now() + timeout;

        loop {
            match self.download_raw(id).await {
                Ok(data) => return Ok(Some(data)),
                Err(error) => {
                    let now = Instant::now();
                    if now >= deadline {
                        tracing::debug!(?id, error = %error, "download wait timed out");
                        return Ok(None);
                    }
                    let remaining = deadline.saturating_duration_since(now);
                    sleep(remaining.min(Duration::from_secs(1))).await;
                }
            }
        }
    }

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

    /// Start a mass model instance with a stable ID and optional per-instance config overrides.
    async fn start_model(
        &self,
        model_id: uuid::Uuid,
        model_config: Option<MassModelConfig>,
    ) -> Result<(), String>;

    /// Run mass inference on an emission stored at `input_id` in void.
    /// Returns the void id of the resulting `InferenceOutput`.
    async fn infer(
        &self,
        model_id: uuid::Uuid,
        request: InferenceRequest,
    ) -> Result<ObjectId, String>;

    /// Reset the model runtime state (for example KV cache) after an inference step.
    async fn reset_model(&self, model_id: uuid::Uuid) -> Result<(), String>;

    /// Upload current model weights to void and return the checkpoint object ID.
    async fn checkpoint_model(&self, model_id: uuid::Uuid) -> Result<ObjectId, String>;

    /// Fuse the current weights of a running model instance with a checkpoint
    /// stored in void using task arithmetic. Returns the void object ID of the
    /// fused weights.
    ///
    /// The instance must be idle (no perturbation in flight); the running
    /// instance itself is left unmodified.
    async fn fuse_weights(
        &self,
        model_id: uuid::Uuid,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String>;

    /// Convert text into dark-tokenized inputs for dark inference flows.
    fn darken(&self, prompt: &str) -> Result<Vec<DarkToken>, String>;

    /// Decode `DarkToken` predictions into text.
    ///
    /// Implementors that manage a tokenizer should delegate to tokenizer decode.
    fn decode(&self, tokens: &[DarkToken]) -> String;

    /// Perturb the associated mass's weights in the positive direction.
    ///
    /// The `seed` parameter controls the random perturbation for reproducibility.
    async fn perturb_up(&self, model_id: uuid::Uuid, seed: u64) -> Result<(), String>;

    /// Perturb the associated mass's weights in the negative direction.
    async fn perturb_down(&self, model_id: uuid::Uuid) -> Result<(), String>;

    /// Apply the QuZO optimization update using the up and down loss values.
    ///
    /// The mass uses the difference between `loss_up` and `loss_down` to
    /// estimate a gradient and update its weights.
    async fn optimize(
        &self,
        model_id: uuid::Uuid,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String>;

    /// Query the current runtime parameters for a running mass model instance.
    async fn query_model_params(&self, model_id: uuid::Uuid) -> Result<MassModelParams, String>;

    /// Shut down a mass model instance.
    async fn shutdown_model(&self, model_id: uuid::Uuid) -> Result<(), String>;

    /// Wait for a [`Transmission`] from the void by object id.
    ///
    /// Downloads raw bytes and deserializes as `Transmission`, using server-side long-polling
    /// when available.
    async fn wait_for_transmission(&self, id: ObjectId) -> Result<Transmission, String> {
        loop {
            match self
                .download_raw_wait(id, DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS)
                .await
            {
                Ok(Some(data)) => {
                    return postcard::from_bytes(&data)
                        .map_err(|e| format!("postcard deserialize: {e}"));
                }
                Ok(None) => {
                    tracing::debug!(
                        ?id,
                        timeout_ms = DEFAULT_TRANSMISSION_LONG_POLL_TIMEOUT_MS,
                        "download wait timed out, retrying"
                    );
                }
                Err(e) => tracing::debug!(?id, error = %e, "download wait failed, retrying"),
            }
        }
    }

    /// Propagates the emission to the next cell - the implementor of
    /// VoidInferOps is responsible for converting the EmissionId into a
    /// Transmission::Propagation by supplying the recv and send void Ids
    /// for the next node.
    async fn transmit(&self, emission_id: EmissionId, send_id: ObjectId) -> Result<(), String>;
}

// Existing Jungle implementations remain valid during the staged migration.
// Their monolithic implementation is projected into the new, narrow
// capability traits; new generic code can depend only on what it uses.
#[async_trait::async_trait]
impl<T> VoidOps for T
where
    T: VoidInferOps,
{
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        <T as VoidInferOps>::download_raw(self, id).await
    }

    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        <T as VoidInferOps>::download_raw_wait(self, id, timeout_ms).await
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        <T as VoidInferOps>::upload_to_void(self, data).await
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        <T as VoidInferOps>::upload_to_void_with(self, id, data).await
    }
}

#[async_trait::async_trait]
impl<T> MassOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        <T as VoidInferOps>::start_model(self, instance_id, None).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<<QwenDarkInference as TensorContract>::Input>,
    ) -> Result<ArtifactRef<<QwenDarkInference as TensorContract>::Output>, String> {
        let bytes = <T as VoidInferOps>::download_raw(self, input.object_id()).await?;
        let request = postcard::from_bytes(&bytes)
            .map_err(|error| format!("postcard deserialize: {error}"))?;
        let output_id = <T as VoidInferOps>::infer(self, instance_id, request).await?;
        Ok(ArtifactRef::from_object_id(output_id))
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        <T as VoidInferOps>::shutdown_model(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<T> ResetOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        <T as VoidInferOps>::reset_model(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<T> PerturbOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn perturb_up_operation(&self, instance_id: ObjectId, seed: u64) -> Result<(), String> {
        <T as VoidInferOps>::perturb_up(self, instance_id, seed).await
    }

    async fn perturb_down_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        <T as VoidInferOps>::perturb_down(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<T> OptimizeOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn optimize_operation(
        &self,
        instance_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        <T as VoidInferOps>::optimize(self, instance_id, loss_up, loss_down).await
    }
}

#[async_trait::async_trait]
impl<T> CheckpointOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn checkpoint_operation(&self, instance_id: ObjectId) -> Result<ObjectId, String> {
        <T as VoidInferOps>::checkpoint_model(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<T> FuseOps<QwenDarkInference> for T
where
    T: VoidInferOps,
{
    async fn fuse_operation(
        &self,
        instance_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        <T as VoidInferOps>::fuse_weights(self, instance_id, checkpoint_id, contribution).await
    }
}

#[async_trait::async_trait]
impl<T> QwenAdapterOps for T
where
    T: VoidInferOps,
{
    fn darken(&self, prompt: &str) -> Result<Vec<DarkToken>, String> {
        <T as VoidInferOps>::darken(self, prompt)
    }

    fn decode(&self, tokens: &[DarkToken]) -> String {
        <T as VoidInferOps>::decode(self, tokens)
    }

    async fn qwen_forward(
        &self,
        instance_id: ObjectId,
        request: InferenceRequest,
    ) -> Result<ObjectRef<InferenceOutput>, String> {
        <T as VoidInferOps>::infer(self, instance_id, request)
            .await
            .map(ObjectRef::new)
    }
}

#[async_trait::async_trait]
pub trait InferenceOutputOps: Sized {
    /// Download an [`Emission`] and then its referenced [`InferenceOutput`],
    /// preserving the emission metadata.
    async fn from_emission_with_metadata<J, M>(
        jungle: &J,
        emission_id: EmissionId,
    ) -> Result<(Self, M), AtomError>
    where
        J: VoidInferOps,
        M: Serialize + DeserializeOwned + Send;

    /// Download an [`Emission`] and then its referenced [`InferenceOutput`],
    /// assuming unit metadata.
    async fn from_emission<J>(jungle: &J, emission_id: EmissionId) -> Result<Self, AtomError>
    where
        J: VoidInferOps,
    {
        let (output, _metadata) =
            Self::from_emission_with_metadata::<J, ()>(jungle, emission_id).await?;
        Ok(output)
    }

    /// Resolve an [`InferenceOutput`] and metadata from a propagation [`Transmission`].
    async fn from_transmission_with_metadata<J, M>(
        jungle: &J,
        transmission: &Transmission,
    ) -> Result<(Self, M), AtomError>
    where
        J: VoidInferOps,
        M: Serialize + DeserializeOwned + Send;

    /// Resolve an [`InferenceOutput`] from a propagation [`Transmission`],
    /// assuming unit metadata.
    async fn from_transmission<J>(
        jungle: &J,
        transmission: &Transmission,
    ) -> Result<Self, AtomError>
    where
        J: VoidInferOps,
    {
        let (output, _metadata) =
            Self::from_transmission_with_metadata::<J, ()>(jungle, transmission).await?;
        Ok(output)
    }
}

#[async_trait::async_trait]
pub trait TransmissionOps: Sized {
    /// Build a propagation [`Transmission`] by uploading an [`InferenceOutput`]
    /// and its metadata payload.
    async fn propagation_from_inference_output_with_metadata<J, M>(
        jungle: &J,
        output: &InferenceOutput,
        metadata: M,
    ) -> Result<Self, AtomError>
    where
        J: VoidInferOps,
        M: Serialize + Send;

    /// Build a propagation [`Transmission`] by uploading an [`InferenceOutput`]
    /// with unit metadata.
    async fn propagation_from_inference_output<J>(
        jungle: &J,
        output: &InferenceOutput,
    ) -> Result<Self, AtomError>
    where
        J: VoidInferOps,
    {
        Self::propagation_from_inference_output_with_metadata(jungle, output, ()).await
    }

    /// Extract an [`EmissionId`] from a propagation [`Transmission`].
    fn propagation_emission_id(&self) -> Result<EmissionId, AtomError>;
}

#[async_trait::async_trait]
impl InferenceOutputOps for InferenceOutput {
    async fn from_emission_with_metadata<J, M>(
        jungle: &J,
        emission_id: EmissionId,
    ) -> Result<(Self, M), AtomError>
    where
        J: VoidInferOps,
        M: Serialize + DeserializeOwned + Send,
    {
        let emission: Emission<M> = jungle
            .download_emission(emission_id.id())
            .await
            .map_err(AtomError::Download)?;

        let output_bytes = jungle
            .download_raw(emission.output_id.object_id())
            .await
            .map_err(AtomError::Download)?;
        let output = postcard::from_bytes(&output_bytes).map_err(AtomError::from)?;
        Ok((output, emission.metadata))
    }

    async fn from_transmission_with_metadata<J, M>(
        jungle: &J,
        transmission: &Transmission,
    ) -> Result<(Self, M), AtomError>
    where
        J: VoidInferOps,
        M: Serialize + DeserializeOwned + Send,
    {
        let emission_id = transmission.propagation_emission_id()?;
        Self::from_emission_with_metadata::<J, M>(jungle, emission_id).await
    }
}

#[async_trait::async_trait]
impl TransmissionOps for Transmission {
    async fn propagation_from_inference_output_with_metadata<J, M>(
        jungle: &J,
        output: &InferenceOutput,
        metadata: M,
    ) -> Result<Self, AtomError>
    where
        J: VoidInferOps,
        M: Serialize + Send,
    {
        let output_bytes = postcard::to_allocvec(output)?;
        let output_id = jungle
            .upload_to_void(output_bytes)
            .await
            .map_err(AtomError::Upload)?;

        let emission = Emission {
            metadata,
            output_id: InferenceOutputId::new(output_id).into(),
        };
        let emission_bytes = postcard::to_allocvec(&emission)?;
        let emission_id = jungle
            .upload_to_void(emission_bytes)
            .await
            .map_err(AtomError::Upload)?;

        Ok(Transmission::Propagation {
            emission_id: EmissionId::new(emission_id),
            recv: ObjectId::nil(),
            send: ObjectId::nil(),
        })
    }

    fn propagation_emission_id(&self) -> Result<EmissionId, AtomError> {
        match self {
            Transmission::Propagation { emission_id, .. } => Ok(emission_id.clone()),
            Transmission::Potentiation { .. } => Err(AtomError::Transmission(
                "expected propagation transmission".to_string(),
            )),
        }
    }
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

    /// Observe one spawned animal by journey id and decode its typed appearance.
    async fn observe_animal<Ap>(&self, journey_id: uuid::Uuid) -> Result<Ap, String>
    where
        Ap: DeserializeOwned + Send;

    /// Perturb one spawned animal by journey id with a typed stimulus payload.
    async fn perturb_animal<S>(&self, journey_id: uuid::Uuid, stimulus: &S) -> Result<(), String>
    where
        S: Serialize + Sync + Send;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use black_hole_spec::{LogitEntry, SequenceOutput};
    use futures::executor::block_on;

    use super::*;

    #[derive(Clone, Default)]
    struct TestJungle {
        objects: Arc<Mutex<HashMap<ObjectId, Vec<u8>>>>,
    }

    #[async_trait::async_trait]
    impl VoidInferOps for TestJungle {
        async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
            self.objects
                .lock()
                .map_err(|error| format!("mutex lock failed: {error}"))?
                .get(&id)
                .cloned()
                .ok_or_else(|| format!("object not found: {id}"))
        }

        async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
            let id = ObjectId::new_v4();
            self.objects
                .lock()
                .map_err(|error| format!("mutex lock failed: {error}"))?
                .insert(id, data);
            Ok(id)
        }

        async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
            self.objects
                .lock()
                .map_err(|error| format!("mutex lock failed: {error}"))?
                .insert(id, data);
            Ok(())
        }

        async fn start_model(
            &self,
            _model_id: uuid::Uuid,
            _model_config: Option<MassModelConfig>,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn infer(
            &self,
            _model_id: uuid::Uuid,
            _request: InferenceRequest,
        ) -> Result<ObjectId, String> {
            let id = ObjectId::new_v4();
            let bytes =
                postcard::to_allocvec(&sample_output(99)).map_err(|error| error.to_string())?;
            self.objects
                .lock()
                .map_err(|error| format!("mutex lock failed: {error}"))?
                .insert(id, bytes);
            Ok(id)
        }

        async fn reset_model(&self, _model_id: uuid::Uuid) -> Result<(), String> {
            Ok(())
        }

        async fn checkpoint_model(&self, _model_id: uuid::Uuid) -> Result<ObjectId, String> {
            Err("unsupported in tests".to_string())
        }

        async fn fuse_weights(
            &self,
            _model_id: uuid::Uuid,
            _checkpoint_id: ObjectId,
            _contribution: f32,
        ) -> Result<ObjectId, String> {
            Err("unsupported in tests".to_string())
        }

        fn darken(&self, _prompt: &str) -> Result<Vec<DarkToken>, String> {
            Err("unsupported in tests".to_string())
        }

        fn decode(&self, tokens: &[DarkToken]) -> String {
            tokens
                .iter()
                .map(|token| token.predicted.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }

        async fn perturb_up(&self, _model_id: uuid::Uuid, _seed: u64) -> Result<(), String> {
            Ok(())
        }

        async fn perturb_down(&self, _model_id: uuid::Uuid) -> Result<(), String> {
            Ok(())
        }

        async fn optimize(
            &self,
            _model_id: uuid::Uuid,
            _loss_up: f32,
            _loss_down: f32,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn query_model_params(
            &self,
            _model_id: uuid::Uuid,
        ) -> Result<MassModelParams, String> {
            Err("unsupported in tests".to_string())
        }

        async fn shutdown_model(&self, _model_id: uuid::Uuid) -> Result<(), String> {
            Ok(())
        }

        async fn transmit(
            &self,
            _emission_id: EmissionId,
            _send_id: ObjectId,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    fn sample_output(token_id: u32) -> InferenceOutput {
        InferenceOutput {
            results: vec![SequenceOutput(vec![DarkToken {
                predicted: token_id,
                dark_knowledge: vec![LogitEntry {
                    token_id,
                    log_prob: 0.0,
                }],
            }])],
        }
    }

    #[test]
    fn from_emission_with_metadata_preserves_metadata() {
        block_on(async {
            let jungle = TestJungle::default();
            let output = sample_output(7);
            let output_bytes = postcard::to_allocvec(&output).unwrap();
            let output_id = VoidInferOps::upload_to_void(&jungle, output_bytes)
                .await
                .unwrap();
            let emission = Emission {
                metadata: "cell-a".to_string(),
                output_id: InferenceOutputId::new(output_id).into(),
            };
            let emission_bytes = postcard::to_allocvec(&emission).unwrap();
            let emission_id = EmissionId::new(
                VoidInferOps::upload_to_void(&jungle, emission_bytes)
                    .await
                    .unwrap(),
            );

            let (resolved_output, metadata) =
                InferenceOutput::from_emission_with_metadata::<_, String>(&jungle, emission_id)
                    .await
                    .unwrap();

            assert_eq!(metadata, "cell-a");
            assert_eq!(resolved_output.results.len(), 1);
            assert_eq!(resolved_output.results[0].0[0].predicted, 7);
        });
    }

    #[test]
    fn from_transmission_with_metadata_preserves_metadata() {
        block_on(async {
            let jungle = TestJungle::default();
            let output = sample_output(11);
            let output_bytes = postcard::to_allocvec(&output).unwrap();
            let output_id = VoidInferOps::upload_to_void(&jungle, output_bytes)
                .await
                .unwrap();
            let emission = Emission {
                metadata: vec!["left".to_string(), "right".to_string()],
                output_id: InferenceOutputId::new(output_id).into(),
            };
            let emission_bytes = postcard::to_allocvec(&emission).unwrap();
            let emission_id = EmissionId::new(
                VoidInferOps::upload_to_void(&jungle, emission_bytes)
                    .await
                    .unwrap(),
            );
            let transmission = Transmission::Propagation {
                emission_id,
                recv: ObjectId::nil(),
                send: ObjectId::nil(),
            };

            let (resolved_output, metadata) = InferenceOutput::from_transmission_with_metadata::<
                _,
                Vec<String>,
            >(&jungle, &transmission)
            .await
            .unwrap();

            assert_eq!(metadata, vec!["left".to_string(), "right".to_string()]);
            assert_eq!(resolved_output.results.len(), 1);
            assert_eq!(resolved_output.results[0].0[0].predicted, 11);
        });
    }

    #[test]
    fn propagation_from_inference_output_with_metadata_uploads_metadata() {
        block_on(async {
            let jungle = TestJungle::default();
            let output = sample_output(21);

            let transmission = Transmission::propagation_from_inference_output_with_metadata(
                &jungle,
                &output,
                ("black-hole".to_string(), 2_u32),
            )
            .await
            .unwrap();

            let emission_id = transmission.propagation_emission_id().unwrap();
            let uploaded: Emission<(String, u32)> =
                VoidInferOps::download_emission(&jungle, emission_id.id())
                    .await
                    .unwrap();

            assert_eq!(uploaded.metadata, ("black-hole".to_string(), 2_u32));
        });
    }

    #[test]
    fn legacy_implementation_projects_into_typed_void_ops() {
        block_on(async {
            let jungle = TestJungle::default();
            let reference = VoidOps::persist(&jungle, &vec![1_u32, 2, 3]).await.unwrap();
            let resolved: Vec<u32> = VoidOps::resolve(&jungle, reference).await.unwrap();
            assert_eq!(resolved, vec![1, 2, 3]);
        });
    }

    #[test]
    fn qwen_adapter_projects_legacy_postcard_forward_into_mass_ops() {
        block_on(async {
            fn assert_capabilities<T>()
            where
                T: MassOps<QwenDarkInference>
                    + ResetOps<QwenDarkInference>
                    + PerturbOps<QwenDarkInference>
                    + OptimizeOps<QwenDarkInference>
                    + CheckpointOps<QwenDarkInference>
                    + FuseOps<QwenDarkInference>
                    + QwenAdapterOps,
            {
            }
            assert_capabilities::<TestJungle>();

            let jungle = TestJungle::default();
            let request = InferenceRequest::Sequences {
                sequences: Vec::new(),
                limit: Some(1),
            };
            let request_bytes = postcard::to_allocvec(&request).unwrap();
            let request_id = VoidInferOps::upload_to_void(&jungle, request_bytes)
                .await
                .unwrap();
            let input = ArtifactRef::<<QwenDarkInference as TensorContract>::Input>::from_object_id(
                request_id,
            );

            let output = MassOps::<QwenDarkInference>::forward(&jungle, ObjectId::new_v4(), input)
                .await
                .unwrap();
            let bytes = VoidInferOps::download_raw(&jungle, output.object_id())
                .await
                .unwrap();
            let decoded: InferenceOutput = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.results[0].0[0].predicted, 99);
        });
    }
}
