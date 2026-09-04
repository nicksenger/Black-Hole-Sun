use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use quinn::crypto::rustls::QuicClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};

use crate::{
    decode_input, encode_output, object_store, operation_capability, persist, ContractId,
    DimensionDescriptor, DtypeConstraint, EncodingId, MassServerBuilder, OperationCapabilities,
    OperationCapability, OperationConfig, OperationImplementation, RawTensor, SingleTensorSpec,
    TensorContract, TensorDtype, TensorPortSpec, VoidServerBuilder,
};

pub struct FakeValues;
pub struct FakeLength;
pub struct DeterministicFakeContract;
pub struct SliceValues;
pub struct TensorSlicerContract;

impl TensorPortSpec for FakeValues {
    type Shape = crate::black_hole_spec::glowstick::Shape1<
        crate::black_hole_spec::glowstick::Dyn<FakeLength>,
    >;

    const NAME: &'static str = "values";

    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![DimensionDescriptor::Symbolic("length".into())]
    }

    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::U32)
    }
}

impl TensorContract for DeterministicFakeContract {
    type Input = SingleTensorSpec<FakeValues>;
    type Output = SingleTensorSpec<FakeValues>;
    type Metadata = ();

    const ID: ContractId = ContractId::from_u128(0x6465_7465_726d_696e_6973_7469_632d_6f70);
    const VERSION: u32 = 1;
}

impl TensorPortSpec for SliceValues {
    type Shape = crate::black_hole_spec::glowstick::Shape1<
        crate::black_hole_spec::glowstick::Dyn<FakeLength>,
    >;
    const NAME: &'static str = "slice";
    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![DimensionDescriptor::Symbolic("length".into())]
    }
    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::U32)
    }
}

impl TensorContract for TensorSlicerContract {
    type Input = SingleTensorSpec<FakeValues>;
    type Output = SingleTensorSpec<SliceValues>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(0x7465_6e73_6f72_2d73_6c69_6365_7200_0001);
    const VERSION: u32 = 1;
}

#[derive(Default)]
pub struct TensorSlicerOperation {
    instances: Mutex<HashSet<uuid::Uuid>>,
}

#[async_trait::async_trait]
impl OperationImplementation for TensorSlicerOperation {
    fn capability(&self) -> OperationCapability {
        operation_capability::<TensorSlicerContract>()
    }
    async fn start(
        &self,
        instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        self.instances.lock().unwrap().insert(instance_id);
        Ok(())
    }
    async fn forward(&self, instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        if !self.instances.lock().unwrap().contains(&instance_id) {
            return Err("slicer instance is not running".into());
        }
        let decoded =
            decode_input::<TensorSlicerContract>(&input).map_err(|error| error.to_string())?;
        let input = &decoded.tensors[0];
        let output_len = input.shape[0] / 2;
        encode_output::<TensorSlicerContract>(
            &[RawTensor {
                name: "slice".into(),
                dtype: TensorDtype::U32,
                shape: vec![output_len],
                data: input.data[..output_len * 4].to_vec(),
            }],
            &(),
        )
        .map_err(|error| error.to_string())
    }
    async fn shutdown(&self, instance_id: uuid::Uuid) -> Result<(), String> {
        self.instances.lock().unwrap().remove(&instance_id);
        Ok(())
    }
}

#[derive(Default)]
pub struct DeterministicFakeOperation {
    instances: Mutex<HashSet<uuid::Uuid>>,
    calls: std::sync::atomic::AtomicUsize,
}

impl DeterministicFakeOperation {
    pub fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl OperationImplementation for DeterministicFakeOperation {
    fn capability(&self) -> OperationCapability {
        operation_capability::<DeterministicFakeContract>()
    }

    async fn start(
        &self,
        instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        if !self.instances.lock().unwrap().insert(instance_id) {
            return Err("fake instance already started".into());
        }
        Ok(())
    }

    async fn forward(&self, instance_id: uuid::Uuid, input: Vec<u8>) -> Result<Vec<u8>, String> {
        if !self.instances.lock().unwrap().contains(&instance_id) {
            return Err("fake instance is not running".into());
        }
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let decoded =
            decode_input::<DeterministicFakeContract>(&input).map_err(|error| error.to_string())?;
        let mut tensor = decoded.tensors.into_iter().next().unwrap();
        for bytes in tensor.data.chunks_exact_mut(4) {
            let value = u32::from_le_bytes(bytes.try_into().unwrap()).wrapping_add(1);
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        encode_output::<DeterministicFakeContract>(&[tensor], &())
            .map_err(|error| error.to_string())
    }

    async fn shutdown(&self, instance_id: uuid::Uuid) -> Result<(), String> {
        if !self.instances.lock().unwrap().remove(&instance_id) {
            return Err("fake instance is not running".into());
        }
        Ok(())
    }
}

pub struct QuadraticValue;
pub struct QuadraticContract;

impl TensorPortSpec for QuadraticValue {
    type Shape = crate::black_hole_spec::glowstick::Shape1<
        crate::black_hole_spec::glowstick::num::U1,
    >;
    const NAME: &'static str = "parameter";
    fn dimensions() -> Vec<DimensionDescriptor> {
        vec![DimensionDescriptor::Static(1)]
    }
    fn dtype() -> DtypeConstraint {
        DtypeConstraint::Exact(TensorDtype::F32)
    }
}

impl TensorContract for QuadraticContract {
    type Input = SingleTensorSpec<QuadraticValue>;
    type Output = SingleTensorSpec<QuadraticValue>;
    type Metadata = ();
    const ID: ContractId = ContractId::from_u128(0x7175_6164_7261_7469_632d_6661_6b65_0001);
    const VERSION: u32 = 1;
}

#[derive(Debug, Clone, Copy)]
struct QuadraticState {
    base: f32,
    value: f32,
    epsilon: f32,
}

pub struct QuadraticOperation {
    states: Mutex<HashMap<uuid::Uuid, QuadraticState>>,
    learning_rate: f32,
}

impl Default for QuadraticOperation {
    fn default() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            learning_rate: 0.2,
        }
    }
}

impl QuadraticOperation {
    pub fn parameter(&self, instance_id: uuid::Uuid) -> Option<f32> {
        self.states
            .lock()
            .unwrap()
            .get(&instance_id)
            .map(|state| state.base)
    }
}

#[async_trait::async_trait]
impl OperationImplementation for QuadraticOperation {
    fn capability(&self) -> OperationCapability {
        let mut capability = operation_capability::<QuadraticContract>();
        capability.operations = OperationCapabilities::OPTIMIZING;
        capability
    }

    async fn start(
        &self,
        instance_id: uuid::Uuid,
        _config: Option<&OperationConfig>,
    ) -> Result<(), String> {
        let old = self.states.lock().unwrap().insert(
            instance_id,
            QuadraticState {
                base: 4.0,
                value: 4.0,
                epsilon: 0.1,
            },
        );
        if old.is_some() {
            Err("quadratic instance already started".into())
        } else {
            Ok(())
        }
    }

    async fn forward(&self, instance_id: uuid::Uuid, _input: Vec<u8>) -> Result<Vec<u8>, String> {
        let value = self
            .states
            .lock()
            .unwrap()
            .get(&instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?
            .value;
        encode_output::<QuadraticContract>(
            &[RawTensor {
                name: "parameter".into(),
                dtype: TensorDtype::F32,
                shape: vec![1],
                data: value.to_le_bytes().to_vec(),
            }],
            &(),
        )
        .map_err(|error| error.to_string())
    }

    async fn reset(&self, instance_id: uuid::Uuid) -> Result<(), String> {
        self.states
            .lock()
            .unwrap()
            .contains_key(&instance_id)
            .then_some(())
            .ok_or_else(|| "quadratic instance is not running".into())
    }

    async fn perturb_up(&self, instance_id: uuid::Uuid, _seed: u64) -> Result<(), String> {
        let mut states = self.states.lock().unwrap();
        let state = states
            .get_mut(&instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?;
        state.value = state.base + state.epsilon;
        Ok(())
    }

    async fn perturb_down(&self, instance_id: uuid::Uuid) -> Result<(), String> {
        let mut states = self.states.lock().unwrap();
        let state = states
            .get_mut(&instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?;
        state.value = state.base - state.epsilon;
        Ok(())
    }

    async fn optimize(
        &self,
        instance_id: uuid::Uuid,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        let mut states = self.states.lock().unwrap();
        let state = states
            .get_mut(&instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?;
        let gradient = (loss_up - loss_down) / (2.0 * state.epsilon);
        state.base -= self.learning_rate * gradient;
        state.value = state.base;
        Ok(())
    }

    async fn checkpoint(&self, instance_id: uuid::Uuid) -> Result<Vec<u8>, String> {
        self.parameter(instance_id)
            .map(|value| value.to_le_bytes().to_vec())
            .ok_or_else(|| "quadratic instance is not running".into())
    }

    async fn fuse(
        &self,
        instance_id: uuid::Uuid,
        checkpoint: Vec<u8>,
        contribution: f32,
    ) -> Result<Vec<u8>, String> {
        let checkpoint = f32::from_le_bytes(
            checkpoint
                .as_slice()
                .try_into()
                .map_err(|_| "invalid quadratic checkpoint")?,
        );
        let current = self
            .parameter(instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?;
        Ok((current * (1.0 - contribution) + checkpoint * contribution)
            .to_le_bytes()
            .to_vec())
    }

    async fn query(&self, instance_id: uuid::Uuid) -> Result<OperationConfig, String> {
        let value = self
            .parameter(instance_id)
            .ok_or_else(|| "quadratic instance is not running".to_string())?;
        Ok(OperationConfig {
            encoding: EncodingId::POSTCARD_V1,
            data: value.to_le_bytes().to_vec(),
        })
    }

    async fn shutdown(&self, instance_id: uuid::Uuid) -> Result<(), String> {
        self.states
            .lock()
            .unwrap()
            .remove(&instance_id)
            .map(|_| ())
            .ok_or_else(|| "quadratic instance is not running".into())
    }
}

/// Certificate verifier for local self-signed test certificates.
#[derive(Debug)]
pub struct NoCertVerifier;

impl rustls::client::danger::ServerCertVerifier for NoCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

/// Build a QUIC client endpoint configured for local self-signed certs.
pub async fn make_client_endpoint() -> quinn::Endpoint {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(NoCertVerifier))
        .with_no_client_auth();

    let quic_crypto = QuicClientConfig::try_from(Arc::new(crypto)).unwrap();
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    let mut client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));
    client_config.transport_config(Arc::new(transport));

    let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let mut endpoint = quinn::Endpoint::client(local_addr).unwrap();
    endpoint.set_default_client_config(client_config);
    endpoint
}

/// Running test void server handle.
pub struct RunningTestVoidServer {
    local_addr: SocketAddr,
    abort_handle: tokio::task::AbortHandle,
}

impl RunningTestVoidServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn abort_handle(&self) -> tokio::task::AbortHandle {
        self.abort_handle.clone()
    }

    pub fn abort(&self) {
        self.abort_handle.abort();
    }
}

impl Drop for RunningTestVoidServer {
    fn drop(&mut self) {
        self.abort_handle.abort();
    }
}

/// Builder for a local test void server.
pub struct TestVoidServer {
    listen_addr: SocketAddr,
    transport_mode: black_hole_void::TransportMode,
    object_store: Box<dyn object_store::ObjectStore>,
    store: Box<dyn persist::VoidStore>,
}

impl TestVoidServer {
    pub fn new() -> Self {
        Self {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            transport_mode: black_hole_void::TransportMode::Quic,
            object_store: Box::new(object_store::InMemoryObjectStore::new()),
            store: Box::new(persist::InMemoryStore::new()),
        }
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }

    pub fn listen_port(mut self, port: u16) -> Self {
        self.listen_addr.set_port(port);
        self
    }

    pub fn listen_on_all_interfaces(mut self) -> Self {
        let port = self.listen_addr.port();
        self.listen_addr = SocketAddr::from(([0, 0, 0, 0], port));
        self
    }

    pub fn tcp(mut self) -> Self {
        self.transport_mode = black_hole_void::TransportMode::Tcp;
        self
    }

    pub fn object_store(mut self, object_store: Box<dyn object_store::ObjectStore>) -> Self {
        self.object_store = object_store;
        self
    }

    pub fn store(mut self, store: Box<dyn persist::VoidStore>) -> Self {
        self.store = store;
        self
    }

    pub async fn serve(self) -> Result<RunningTestVoidServer, black_hole_void::ServerError> {
        let (local_addr, handle) = VoidServerBuilder::new(self.object_store, self.store)
            .transport_mode(self.transport_mode)
            .listen(self.listen_addr)
            .serve()
            .await?;
        let abort_handle = handle.abort_handle();
        drop(handle);
        Ok(RunningTestVoidServer {
            local_addr,
            abort_handle,
        })
    }
}

impl Default for TestVoidServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Running test mass server handle.
pub struct RunningTestMassServer {
    local_addr: SocketAddr,
    abort_handle: tokio::task::AbortHandle,
}

impl RunningTestMassServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn abort_handle(&self) -> tokio::task::AbortHandle {
        self.abort_handle.clone()
    }

    pub fn abort(&self) {
        self.abort_handle.abort();
    }
}

impl Drop for RunningTestMassServer {
    fn drop(&mut self) {
        self.abort_handle.abort();
    }
}

/// Builder for a local test mass server.
pub struct TestMassServer {
    model_path: PathBuf,
    listen_addr: SocketAddr,
    transport_mode: black_hole_mass::TransportMode,
    void_addr: Option<SocketAddr>,
    tunnel: Option<SocketAddr>,
    tunnel_connect_deadline: Option<Duration>,
    max_instances: Option<usize>,
    top_k: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    kv_cache_quant: black_hole_mass::KvCacheQuantization,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    default_inference_limit: Option<u32>,
    training_lr: Option<f64>,
    training_epsilon: Option<f64>,
    frozen: bool,
    operation: Option<Arc<dyn black_hole_mass::OperationImplementation>>,
}

impl TestMassServer {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            transport_mode: black_hole_mass::TransportMode::Quic,
            void_addr: None,
            tunnel: None,
            tunnel_connect_deadline: None,
            max_instances: None,
            top_k: None,
            temperature: None,
            top_p: None,
            kv_cache_quant: black_hole_mass::KvCacheQuantization::Q8_0,
            repeat_penalty: None,
            presence_penalty: None,
            default_inference_limit: None,
            training_lr: None,
            training_epsilon: None,
            frozen: false,
            operation: None,
        }
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }

    pub fn listen_port(mut self, port: u16) -> Self {
        self.listen_addr.set_port(port);
        self
    }

    pub fn listen_on_all_interfaces(mut self) -> Self {
        let port = self.listen_addr.port();
        self.listen_addr = SocketAddr::from(([0, 0, 0, 0], port));
        self
    }

    pub fn tcp(mut self) -> Self {
        self.transport_mode = black_hole_mass::TransportMode::Tcp;
        self
    }

    pub fn void_addr(mut self, addr: SocketAddr) -> Self {
        self.void_addr = Some(addr);
        self
    }

    pub fn tunnel(mut self, addr: SocketAddr) -> Self {
        self.tunnel = Some(addr);
        self
    }

    pub fn tunnel_connect_deadline(mut self, deadline: Duration) -> Self {
        self.tunnel_connect_deadline = Some(deadline);
        self
    }

    pub fn max_instances(mut self, limit: usize) -> Self {
        self.max_instances = Some(limit);
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn kv_cache_quant(mut self, quant: black_hole_mass::KvCacheQuantization) -> Self {
        self.kv_cache_quant = quant;
        self
    }

    pub fn disable_kv_cache_quantization(mut self) -> Self {
        self.kv_cache_quant = black_hole_mass::KvCacheQuantization::F16;
        self
    }

    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    pub fn default_inference_limit(mut self, limit: u32) -> Self {
        self.default_inference_limit = Some(limit);
        self
    }

    pub fn training_lr(mut self, lr: f64) -> Self {
        self.training_lr = Some(lr);
        self
    }

    pub fn training_epsilon(mut self, epsilon: f64) -> Self {
        self.training_epsilon = Some(epsilon);
        self
    }

    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    pub fn operation(
        mut self,
        operation: Arc<dyn black_hole_mass::OperationImplementation>,
    ) -> Self {
        self.operation = Some(operation);
        self
    }

    pub async fn serve(self) -> Result<RunningTestMassServer, black_hole_mass::ServerError> {
        let mut builder = MassServerBuilder::new(self.model_path)
            .transport_mode(self.transport_mode)
            .listen(self.listen_addr)
            .kv_cache_quant(self.kv_cache_quant);
        if self.frozen {
            builder = builder.frozen();
        }
        if let Some(operation) = self.operation {
            builder = builder.operation_shared(operation);
        }
        if let Some(void_addr) = self.void_addr {
            builder = builder.void_addr(void_addr);
        }
        if let Some(tunnel) = self.tunnel {
            builder = builder.tunnel(tunnel);
        }
        if let Some(deadline) = self.tunnel_connect_deadline {
            builder = builder.tunnel_connect_deadline(deadline);
        }
        if let Some(max_instances) = self.max_instances {
            builder = builder.max_instances(max_instances);
        }
        if let Some(top_k) = self.top_k {
            builder = builder.top_k(top_k);
        }
        if let Some(temperature) = self.temperature {
            builder = builder.temperature(temperature);
        }
        if let Some(top_p) = self.top_p {
            builder = builder.top_p(Some(top_p));
        }
        if let Some(repeat_penalty) = self.repeat_penalty {
            builder = builder.repeat_penalty(repeat_penalty);
        }
        if let Some(presence_penalty) = self.presence_penalty {
            builder = builder.presence_penalty(presence_penalty);
        }
        if let Some(limit) = self.default_inference_limit {
            builder = builder.default_inference_limit(limit);
        }
        if let Some(lr) = self.training_lr {
            builder = builder.training_lr(lr);
        }
        if let Some(epsilon) = self.training_epsilon {
            builder = builder.training_epsilon(epsilon);
        }

        let (local_addr, handle) = builder.serve().await?;
        let abort_handle = handle.abort_handle();
        drop(handle);
        Ok(RunningTestMassServer {
            local_addr,
            abort_handle,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{TestMassServer, TestVoidServer};
    use std::net::SocketAddr;

    #[test]
    fn void_server_listen_on_all_interfaces_preserves_configured_port() {
        let server = TestVoidServer::new()
            .listen_port(4545)
            .listen_on_all_interfaces();
        assert_eq!(server.listen_addr, SocketAddr::from(([0, 0, 0, 0], 4545)));
    }

    #[test]
    fn mass_server_listen_port_works_after_switching_to_all_interfaces() {
        let server = TestMassServer::new("model-does-not-need-to-exist")
            .listen_on_all_interfaces()
            .listen_port(5656);
        assert_eq!(server.listen_addr, SocketAddr::from(([0, 0, 0, 0], 5656)));
    }
}
