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
    default_inference_limit: Option<u32>,
    frozen: bool,
}

impl TestQuarkServer {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            void_addr: None,
            default_inference_limit: None,
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

    pub fn default_inference_limit(mut self, limit: u32) -> Self {
        self.default_inference_limit = Some(limit);
        self
    }

    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    pub async fn serve(self) -> Result<RunningTestQuarkServer, black_hole_quark::ServerError> {
        let mut builder = QuarkServerBuilder::new(self.model_path).listen(self.listen_addr);
        if self.frozen {
            builder = builder.frozen();
        }
        if let Some(void_addr) = self.void_addr {
            builder = builder.void_addr(void_addr);
        }
        if let Some(limit) = self.default_inference_limit {
            builder = builder.default_inference_limit(limit);
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
