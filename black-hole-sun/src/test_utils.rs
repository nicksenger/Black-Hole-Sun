use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use quinn::crypto::rustls::QuicClientConfig;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};

use crate::{object_store, persist, QuarkServerBuilder, VoidServerBuilder};

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
    object_store: Box<dyn object_store::ObjectStore>,
    store: Box<dyn persist::VoidStore>,
}

impl TestVoidServer {
    pub fn new() -> Self {
        Self {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            object_store: Box::new(object_store::InMemoryObjectStore::new()),
            store: Box::new(persist::InMemoryStore::new()),
        }
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
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

/// Running test quark server handle.
pub struct RunningTestQuarkServer {
    local_addr: SocketAddr,
    abort_handle: tokio::task::AbortHandle,
}

impl RunningTestQuarkServer {
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

impl Drop for RunningTestQuarkServer {
    fn drop(&mut self) {
        self.abort_handle.abort();
    }
}

/// Builder for a local test quark server.
pub struct TestQuarkServer {
    model_path: PathBuf,
    listen_addr: SocketAddr,
    void_addr: Option<SocketAddr>,
    top_k: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    kv_cache_quant: black_hole_quark::KvCacheQuantization,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    default_inference_limit: Option<u32>,
    training_lr: Option<f64>,
    training_epsilon: Option<f64>,
    frozen: bool,
}

impl TestQuarkServer {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            void_addr: None,
            top_k: None,
            temperature: None,
            top_p: None,
            kv_cache_quant: black_hole_quark::KvCacheQuantization::Q8_0,
            repeat_penalty: None,
            presence_penalty: None,
            default_inference_limit: None,
            training_lr: None,
            training_epsilon: None,
            frozen: false,
        }
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }

    pub fn void_addr(mut self, addr: SocketAddr) -> Self {
        self.void_addr = Some(addr);
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

    pub fn kv_cache_quant(mut self, quant: black_hole_quark::KvCacheQuantization) -> Self {
        self.kv_cache_quant = quant;
        self
    }

    pub fn disable_kv_cache_quantization(mut self) -> Self {
        self.kv_cache_quant = black_hole_quark::KvCacheQuantization::F16;
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

    pub async fn serve(self) -> Result<RunningTestQuarkServer, black_hole_quark::ServerError> {
        let mut builder = QuarkServerBuilder::new(self.model_path)
            .listen(self.listen_addr)
            .kv_cache_quant(self.kv_cache_quant);
        if self.frozen {
            builder = builder.frozen();
        }
        if let Some(void_addr) = self.void_addr {
            builder = builder.void_addr(void_addr);
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
        Ok(RunningTestQuarkServer {
            local_addr,
            abort_handle,
        })
    }
}
