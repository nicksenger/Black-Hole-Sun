use std::{collections::HashMap, fs, io, net::SocketAddr, path::PathBuf, sync::Arc};

use paramecia_engine::{ModelEngine, TrainingConfig};
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use black_hole_spec::{
    DarkToken, InferenceInput, InferenceOutput, InferenceRequest, LogitEntry, ObjectId, QuarkIn,
    QuarkModelConfig, QuarkOut, SequenceOutput, TunnelRequest,
};
pub use paramecia_engine::KvCacheQuantization;

const DEFAULT_LISTEN_ADDR: &str = "[::1]:4433";
const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MB
const DEFAULT_ENGINE_TOP_K: usize = 256;
const DEFAULT_ENGINE_TEMPERATURE: f64 = 0.7;
const DEFAULT_ENGINE_REPEAT_PENALTY: f32 = 1.0;
const DEFAULT_ENGINE_PRESENCE_PENALTY: f32 = 0.0;
const DEFAULT_INFERENCE_LIMIT: u32 = 256;
const DEFAULT_MAX_INSTANCES: usize = 1;
const DEFAULT_CHECKPOINT_TOKENIZER_FILE: &str = "tokenizer.json";
const DEFAULT_CHECKPOINT_TOKENIZER_DIR: &str = ".black-hole-sun/tokenizers";
const CHECKPOINT_CACHE_DIR: &str = "black-hole-quark/checkpoints";

// ---------------------------------------------------------------------------
// Void client — connects to black-hole-void over QUIC, sends/receives frames
// ---------------------------------------------------------------------------

/// Wire request sent to the void service.
#[derive(Debug, Serialize, Deserialize)]
enum VoidIn {
    Upload { data: Vec<u8> },
    UploadWith { id: ObjectId, data: Vec<u8> },
    Download { id: ObjectId },
}

/// Wire response from the void service.
#[derive(Debug, Serialize, Deserialize)]
enum VoidOut {
    Uploaded { id: ObjectId },
    Downloaded { data: Vec<u8> },
    Error { message: String },
}

/// No-op certificate verifier for connecting to void (self-signed certs in local dev).
#[derive(Debug)]
struct PermissiveCertVerifier;

impl rustls::client::danger::ServerCertVerifier for PermissiveCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
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

/// A QUIC client connection to the void object store.
pub struct VoidClient {
    endpoint: quinn::Endpoint,
    remote_addr: SocketAddr,
}

impl VoidClient {
    /// Connect to the void service at `addr`.
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        // Void uses self-signed certs by default in local dev — skip verification.
        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PermissiveCertVerifier))
            .with_no_client_auth();

        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
            .map_err(|e| ServerError::VoidCrypto(e.to_string()))?;

        let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

        // Bind to any local address/port.
        let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let mut endpoint =
            quinn::Endpoint::client(local_addr).map_err(ServerError::BindVoidClient)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            endpoint,
            remote_addr: addr,
        })
    }

    /// Open a bidirectional stream, send a void request, and read the response.
    async fn call(&self, req: VoidIn) -> Result<VoidOut> {
        let server_name = self.remote_addr.ip().to_string();
        let conn = self
            .endpoint
            .connect(self.remote_addr, &server_name)
            .map_err(|e| ServerError::VoidConnect(e.to_string()))?
            .await
            .map_err(|e| ServerError::VoidConnect(e.to_string()))?;

        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| ServerError::VoidStream(e.to_string()))?;

        write_frame(&mut send, &req).await?;
        let resp = read_frame(&mut recv).await?;
        Ok(resp)
    }

    /// Download an object from void by its ID. Returns the raw bytes.
    pub async fn download(&self, id: ObjectId) -> Result<Vec<u8>> {
        let resp = self.call(VoidIn::Download { id }).await?;
        match resp {
            VoidOut::Downloaded { data } => Ok(data),
            VoidOut::Error { message } => Err(ServerError::VoidError(message)),
            _ => Err(ServerError::VoidError(
                "unexpected void response for download".into(),
            )),
        }
    }

    /// Upload data to void. Returns the assigned object ID.
    pub async fn upload(&self, data: Vec<u8>) -> Result<ObjectId> {
        let resp = self.call(VoidIn::Upload { data }).await?;
        match resp {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(ServerError::VoidError(message)),
            _ => Err(ServerError::VoidError(
                "unexpected void response for upload".into(),
            )),
        }
    }
}

/// A QUIC client connection to another quark server.
struct QuarkRpcClient {
    endpoint: quinn::Endpoint,
    remote_addr: SocketAddr,
}

impl QuarkRpcClient {
    async fn connect(addr: SocketAddr) -> Result<Self> {
        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PermissiveCertVerifier))
            .with_no_client_auth();
        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
            .map_err(|e| ServerError::TunnelCrypto(e.to_string()))?;
        let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

        let local_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let mut endpoint =
            quinn::Endpoint::client(local_addr).map_err(ServerError::BindTunnelClient)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            endpoint,
            remote_addr: addr,
        })
    }

    async fn request(&self, req: QuarkIn) -> Result<QuarkOut> {
        let server_name = self.remote_addr.ip().to_string();
        let conn = self
            .endpoint
            .connect(self.remote_addr, &server_name)
            .map_err(|e| ServerError::TunnelConnect(e.to_string()))?
            .await
            .map_err(|e| ServerError::TunnelConnect(e.to_string()))?;
        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| ServerError::TunnelStream(e.to_string()))?;
        write_frame(&mut send, &req).await?;
        read_frame(&mut recv).await
    }
}

/// Write a length-prefixed postcard frame to a QUIC send stream.
async fn write_frame(send: &mut quinn::SendStream, msg: &impl Serialize) -> Result<()> {
    let payload = to_allocvec(msg).map_err(ServerError::EncodeFrame)?;
    let len =
        u32::try_from(payload.len()).map_err(|_| ServerError::FrameTooLarge(payload.len()))?;

    send.write_all(&len.to_be_bytes())
        .await
        .map_err(ServerError::WriteFrame)?;
    send.write_all(&payload)
        .await
        .map_err(ServerError::WriteFrame)?;
    Ok(())
}

/// Read a length-prefixed postcard frame from a QUIC recv stream.
async fn read_frame<T: for<'de> Deserialize<'de>>(recv: &mut quinn::RecvStream) -> Result<T> {
    let len = match recv.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(ServerError::UnexpectedEof);
        }
        Err(e) => return Err(ServerError::ReadFrameLength(e)),
    };

    if len > MAX_FRAME_SIZE {
        return Err(ServerError::FrameTooLarge(len));
    }

    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(ServerError::ReadFramePayload)?;

    from_bytes(&payload).map_err(ServerError::DecodeFrame)
}

// ---------------------------------------------------------------------------
// QuZO state — tracks perturb/up-loss between client requests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuarkState {
    /// Initial or post-update: next expected step is PerturbUp.
    Idle,
    /// After PerturbUp: waiting for the first up-inference.
    PostPerturbUp,
    /// After one or more up-inferences: waiting for PerturbDown.
    AwaitingPerturbDown,
    /// After PerturbDown: waiting for the first down-inference.
    PostPerturbDown,
    /// After one or more down-inferences: waiting for Optimize.
    AwaitingOptimize,
}

struct QuarkSession {
    state: QuarkState,
    running: bool,
    frozen: bool,
    optimize_steps: usize,
}

impl QuarkSession {
    fn new(frozen: bool) -> Self {
        Self {
            state: QuarkState::Idle,
            running: true,
            frozen,
            optimize_steps: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrozenOscillation {
    steps: usize,
    offset: usize,
}

// ---------------------------------------------------------------------------
// Server context — shared across connections
// ---------------------------------------------------------------------------

struct QuarkInstance {
    engine: ModelEngine,
    inference_limit: u32,
    oscillation: Option<FrozenOscillation>,
    checkpoint_path: Option<PathBuf>,
    session: tokio::sync::Mutex<QuarkSession>,
}

struct ResolvedModelSource {
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    checkpoint_path: Option<PathBuf>,
}

enum ModelSlot {
    Starting,
    Running(Arc<QuarkInstance>),
    ShuttingDown,
}

#[derive(Debug, Clone, Copy)]
enum QuarkMode {
    Root,
    Worker {
        parent_addr: SocketAddr,
        tunnel_token: Uuid,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RouteTarget {
    Local,
    Worker(Uuid),
}

#[derive(Debug, Clone, Copy)]
struct TunnelWorker {
    token: Uuid,
    addr: SocketAddr,
    max_instances: Option<usize>,
}

struct QuarkContext {
    #[allow(dead_code)]
    local_addr: SocketAddr,
    model_path: PathBuf,
    void_client: Option<Arc<VoidClient>>,
    defaults: QuarkServerDefaults,
    frozen: bool,
    max_instances: Option<usize>,
    mode: QuarkMode,
    routes: tokio::sync::RwLock<HashMap<Uuid, RouteTarget>>,
    workers: tokio::sync::RwLock<HashMap<Uuid, TunnelWorker>>,
    instances: tokio::sync::RwLock<HashMap<Uuid, ModelSlot>>,
}

#[derive(Clone)]
struct QuarkServerDefaults {
    top_k: usize,
    temperature: f64,
    top_p: Option<f64>,
    kv_cache_quant: KvCacheQuantization,
    repeat_penalty: f32,
    presence_penalty: f32,
    inference_limit: u32,
    training_config: TrainingConfig,
}

impl Default for QuarkServerDefaults {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_ENGINE_TOP_K,
            temperature: DEFAULT_ENGINE_TEMPERATURE,
            top_p: None,
            kv_cache_quant: KvCacheQuantization::Q8_0,
            repeat_penalty: DEFAULT_ENGINE_REPEAT_PENALTY,
            presence_penalty: DEFAULT_ENGINE_PRESENCE_PENALTY,
            inference_limit: DEFAULT_INFERENCE_LIMIT,
            training_config: TrainingConfig::default(),
        }
    }
}

impl QuarkServerDefaults {
    fn with_overrides(&self, model_config: Option<&QuarkModelConfig>) -> Self {
        let mut resolved = self.clone();
        if let Some(model_config) = model_config {
            if let Some(top_k) = model_config.top_k {
                resolved.top_k = top_k;
            }
            if let Some(temperature) = model_config.temperature {
                resolved.temperature = temperature;
            }
            if let Some(top_p) = model_config.top_p {
                resolved.top_p = Some(top_p);
            }
            if let Some(repeat_penalty) = model_config.repeat_penalty {
                resolved.repeat_penalty = repeat_penalty;
            }
            if let Some(presence_penalty) = model_config.presence_penalty {
                resolved.presence_penalty = presence_penalty;
            }
            if let Some(inference_limit) = model_config.inference_limit {
                resolved.inference_limit = inference_limit;
            }
            if let Some(training_lr) = model_config.training_lr {
                resolved.training_config.lr = training_lr;
            }
            if let Some(training_epsilon) = model_config.training_epsilon {
                resolved.training_config.epsilon = training_epsilon;
            }
        }
        resolved
    }
}

fn resolve_max_instances(limit: Option<usize>) -> Option<usize> {
    Some(limit.unwrap_or(DEFAULT_MAX_INSTANCES))
}

// ---------------------------------------------------------------------------
// Server builder
// ---------------------------------------------------------------------------

pub struct ServerBuilder {
    keylog: bool,
    key: Option<PathBuf>,
    cert: Option<PathBuf>,
    stateless_retry: bool,
    frozen: bool,
    listen: SocketAddr,
    model_path: PathBuf,
    void_addr: Option<SocketAddr>,
    tunnel: Option<SocketAddr>,
    max_instances: Option<usize>,
    defaults: QuarkServerDefaults,
}

impl ServerBuilder {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            keylog: false,
            key: None,
            cert: None,
            stateless_retry: false,
            frozen: false,
            listen: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default listen address must be valid"),
            model_path: model_path.into(),
            void_addr: None,
            tunnel: None,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            defaults: QuarkServerDefaults::default(),
        }
    }

    pub fn keylog(mut self, v: bool) -> Self {
        self.keylog = v;
        self
    }

    pub fn key(mut self, p: PathBuf) -> Self {
        self.key = Some(p);
        self
    }

    pub fn cert(mut self, p: PathBuf) -> Self {
        self.cert = Some(p);
        self
    }

    pub fn stateless_retry(mut self, v: bool) -> Self {
        self.stateless_retry = v;
        self
    }

    /// Freeze model weights by disabling perturb/update mutations.
    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen = addr;
        self
    }

    pub fn void_addr(mut self, addr: SocketAddr) -> Self {
        self.void_addr = Some(addr);
        self
    }

    /// Register this quark as a tunnel worker of the parent at `addr`.
    pub fn tunnel(mut self, addr: SocketAddr) -> Self {
        self.tunnel = Some(addr);
        self
    }

    /// Limit concurrent model instances handled by this quark.
    pub fn max_instances(mut self, limit: usize) -> Self {
        self.max_instances = Some(limit);
        self
    }

    /// Configure the default top-k used by model instances spawned by this server.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.defaults.top_k = top_k;
        self
    }

    /// Configure the default sampling temperature used by model instances.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.defaults.temperature = temperature;
        self
    }

    /// Configure the optional top-p sampler parameter for model instances.
    pub fn top_p(mut self, top_p: Option<f64>) -> Self {
        self.defaults.top_p = top_p;
        self
    }

    /// Configure the KV-cache quantization used by model instances.
    pub fn kv_cache_quant(mut self, quant: KvCacheQuantization) -> Self {
        self.defaults.kv_cache_quant = quant;
        self
    }

    /// Disable KV-cache quantization (uses f16 cache tensors).
    pub fn disable_kv_cache_quantization(mut self) -> Self {
        self.defaults.kv_cache_quant = KvCacheQuantization::F16;
        self
    }

    /// Configure the default repeat-penalty used by model instances.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.defaults.repeat_penalty = penalty;
        self
    }

    /// Configure the default presence-penalty used by model instances.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.defaults.presence_penalty = penalty;
        self
    }

    /// Configure the default max tokens generated when requests omit `limit`.
    pub fn default_inference_limit(mut self, limit: u32) -> Self {
        self.defaults.inference_limit = limit;
        self
    }

    /// Configure the default QuZO learning rate.
    pub fn training_lr(mut self, lr: f64) -> Self {
        self.defaults.training_config.lr = lr;
        self
    }

    /// Configure the default QuZO epsilon.
    pub fn training_epsilon(mut self, epsilon: f64) -> Self {
        self.defaults.training_config.epsilon = epsilon;
        self
    }

    /// Configure the full default training config used for new model instances.
    pub fn training_config(mut self, config: TrainingConfig) -> Self {
        self.defaults.training_config = config;
        self
    }

    /// Build the void client, endpoint and shared server context.
    async fn setup(self) -> Result<(quinn::Endpoint, SocketAddr, Arc<QuarkContext>)> {
        let model_path_str = self.model_path.to_string_lossy().to_string();
        info!(model_path = %model_path_str, "configured model");

        // Optionally connect to void.
        let void_client = if let Some(addr) = self.void_addr {
            info!(%addr, "connecting to void");
            let client = VoidClient::connect(addr).await?;
            Some(Arc::new(client))
        } else {
            warn!("no void address configured — inference will fail without object store");
            None
        };

        let (cert_chain, key) = match (self.key.as_ref(), self.cert.as_ref()) {
            (Some(key_path), Some(cert_path)) => {
                load_user_cert_chain_and_key(key_path, cert_path)?
            }
            _ => load_or_generate_self_signed_cert()?,
        };

        let mut server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(cert_chain, key)
            .map_err(ServerError::RustlsCertConfig)?;

        if self.keylog {
            server_config.key_log = Arc::new(rustls::KeyLogFile::new());
        }

        let crypto =
            QuicServerConfig::try_from(server_config).map_err(ServerError::QuicServerConfig)?;

        let endpoint_cfg = quinn::ServerConfig::with_crypto(Arc::new(crypto));

        let listener = std::net::UdpSocket::bind(self.listen).map_err(ServerError::BindEndpoint)?;
        let runtime = quinn::TokioRuntime;
        let endpoint = quinn::Endpoint::new(
            Default::default(),
            Some(endpoint_cfg),
            listener,
            Arc::new(runtime),
        )
        .map_err(ServerError::BindEndpoint)?;

        let local_addr = endpoint.local_addr().map_err(ServerError::LocalAddr)?;
        info!(%local_addr, "listening");

        let mode = if let Some(parent_addr) = self.tunnel {
            info!(%parent_addr, %local_addr, "registering tunnel worker");
            let tunnel_token = register_tunnel_worker(
                parent_addr,
                local_addr,
                resolve_max_instances(self.max_instances),
            )
            .await?;
            info!(%parent_addr, %local_addr, token = %tunnel_token, "tunnel worker registered");
            QuarkMode::Worker {
                parent_addr,
                tunnel_token,
            }
        } else {
            QuarkMode::Root
        };

        let context = Arc::new(QuarkContext {
            local_addr,
            model_path: self.model_path,
            void_client,
            defaults: self.defaults,
            frozen: self.frozen,
            max_instances: resolve_max_instances(self.max_instances),
            mode,
            routes: tokio::sync::RwLock::new(HashMap::new()),
            workers: tokio::sync::RwLock::new(HashMap::new()),
            instances: tokio::sync::RwLock::new(HashMap::new()),
        });

        Ok((endpoint, local_addr, context))
    }

    /// Start the server in a background task. Returns the bound address and
    /// a handle that can be used to await or abort the server.
    pub async fn serve(self) -> Result<(SocketAddr, tokio::task::JoinHandle<Result<()>>)> {
        let stateless_retry = self.stateless_retry;
        let (endpoint, local_addr, context) = self.setup().await?;
        let handle = tokio::spawn(Self::accept_loop(endpoint, context, stateless_retry));
        Ok((local_addr, handle))
    }

    /// Run the server, blocking until the endpoint is closed.
    pub async fn run(self) -> Result<()> {
        let stateless_retry = self.stateless_retry;
        let (endpoint, _local_addr, context) = self.setup().await?;
        Self::accept_loop(endpoint, context, stateless_retry).await
    }

    /// Accept-loop shared by both `run()` and `serve()`.
    async fn accept_loop(
        endpoint: quinn::Endpoint,
        context: Arc<QuarkContext>,
        stateless_retry: bool,
    ) -> Result<()> {
        loop {
            let conn = tokio::select! {
                incoming = endpoint.accept() => match incoming {
                    Some(c) => c,
                    None => break,
                },
            };

            if stateless_retry && !conn.remote_address_validated() {
                info!("requiring connection to validate its address");
                let _ = conn.retry();
                continue;
            }

            info!(remote = %conn.remote_address(), "accepting connection");
            let ctx = Arc::clone(&context);
            tokio::spawn(handle_connection(conn, ctx));
        }

        Ok(())
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        panic!("use ServerBuilder::new(model_path) instead");
    }
}

// ---------------------------------------------------------------------------
// Connection / stream handlers
// ---------------------------------------------------------------------------

async fn handle_connection(incoming: quinn::Incoming, context: Arc<QuarkContext>) {
    let connection = match incoming.await {
        Ok(c) => c,
        Err(e) => {
            error!("connection failed: {e}");
            return;
        }
    };

    info!("established");

    loop {
        let stream = match connection.accept_bi().await {
            Err(quinn::ConnectionError::ApplicationClosed { .. }) => {
                info!("connection closed");
                return;
            }
            Err(e) => {
                error!("stream error: {e}");
                return;
            }
            Ok(s) => s,
        };

        let ctx = Arc::clone(&context);
        tokio::spawn(handle_stream(stream, ctx));
    }
}

async fn handle_stream(
    (mut send, mut recv): (quinn::SendStream, quinn::RecvStream),
    context: Arc<QuarkContext>,
) {
    let req: QuarkIn = match read_frame(&mut recv).await {
        Ok(r) => r,
        Err(e) => {
            let _ = write_frame(
                &mut send,
                &QuarkOut::Error {
                    message: e.to_string(),
                },
            )
            .await;
            return;
        }
    };

    debug!(?req, "handling request");

    let out = match handle_request(req, &context).await {
        Ok(o) => o,
        Err(e) => QuarkOut::Error {
            message: e.to_string(),
        },
    };

    if write_frame(&mut send, &out).await.is_err() {
        warn!("failed to write response");
    }
}

async fn handle_request(req: QuarkIn, ctx: &QuarkContext) -> Result<QuarkOut> {
    match req {
        QuarkIn::RegisterTunnel {
            worker_addr,
            max_instances,
        } => handle_register_tunnel(worker_addr, max_instances, ctx).await,
        QuarkIn::UpdateTunnelCapacity {
            token,
            max_instances,
        } => handle_update_tunnel_capacity(token, max_instances, ctx).await,
        QuarkIn::TunnelForward { token, request } => {
            handle_tunnel_forward(token, request, ctx).await
        }
        QuarkIn::Start {
            model_id,
            model_config,
        } => handle_start_routed(model_id, model_config, ctx).await,
        QuarkIn::PerturbUp { model_id, seed } => {
            handle_perturb_up_routed(model_id, seed, ctx).await
        }
        QuarkIn::Infer { model_id, input_id } => handle_infer_routed(model_id, input_id, ctx).await,
        QuarkIn::Reset { model_id } => handle_reset_routed(model_id, ctx).await,
        QuarkIn::PerturbDown { model_id } => handle_perturb_down_routed(model_id, ctx).await,
        QuarkIn::Checkpoint { model_id } => handle_checkpoint_routed(model_id, ctx).await,
        QuarkIn::Optimize {
            model_id,
            loss_up,
            loss_down,
        } => handle_optimize_routed(model_id, loss_up, loss_down, ctx).await,
        QuarkIn::Shutdown { model_id } => handle_shutdown_routed(model_id, ctx).await,
    }
}

fn ensure_root_mode(ctx: &QuarkContext) -> Result<()> {
    match ctx.mode {
        QuarkMode::Root => Ok(()),
        QuarkMode::Worker { .. } => Err(ServerError::TunnelWorkerRejectsModelRequests),
    }
}

async fn register_tunnel_worker(
    parent_addr: SocketAddr,
    worker_addr: SocketAddr,
    max_instances: Option<usize>,
) -> Result<Uuid> {
    let client = QuarkRpcClient::connect(parent_addr).await?;
    let out = client
        .request(QuarkIn::RegisterTunnel {
            worker_addr,
            max_instances,
        })
        .await?;
    match out {
        QuarkOut::TunnelRegistered { token } => Ok(token),
        QuarkOut::Error { message } => Err(ServerError::TunnelRegistrationRejected(message)),
        _ => Err(ServerError::UnexpectedTunnelResponse(
            "register tunnel response",
        )),
    }
}

async fn update_tunnel_capacity(
    parent_addr: SocketAddr,
    tunnel_token: Uuid,
    max_instances: Option<usize>,
) -> Result<()> {
    let client = QuarkRpcClient::connect(parent_addr).await?;
    let out = client
        .request(QuarkIn::UpdateTunnelCapacity {
            token: tunnel_token,
            max_instances,
        })
        .await?;
    match out {
        QuarkOut::Ack => Ok(()),
        QuarkOut::Error { message } => Err(ServerError::TunnelCapacityUpdateRejected(message)),
        _ => Err(ServerError::UnexpectedTunnelResponse(
            "update tunnel capacity response",
        )),
    }
}

fn sum_capacity(lhs: Option<usize>, rhs: Option<usize>) -> Option<usize> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.saturating_add(rhs)),
        _ => None,
    }
}

async fn advertised_capacity(ctx: &QuarkContext) -> Option<usize> {
    let mut total = ctx.max_instances;
    for worker in ctx.workers.read().await.values() {
        total = sum_capacity(total, worker.max_instances);
    }
    total
}

async fn propagate_capacity_to_parent(ctx: &QuarkContext) -> Result<()> {
    match ctx.mode {
        QuarkMode::Root => Ok(()),
        QuarkMode::Worker {
            parent_addr,
            tunnel_token,
        } => {
            let max_instances = advertised_capacity(ctx).await;
            update_tunnel_capacity(parent_addr, tunnel_token, max_instances).await
        }
    }
}

async fn handle_register_tunnel(
    worker_addr: SocketAddr,
    max_instances: Option<usize>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    let max_instances = resolve_max_instances(max_instances);

    let token = {
        let mut workers = ctx.workers.write().await;
        if let Some((token, worker)) = workers
            .iter_mut()
            .find(|(_, worker)| worker.addr == worker_addr)
        {
            worker.max_instances = max_instances;
            *token
        } else {
            let token = Uuid::new_v4();
            workers.insert(
                token,
                TunnelWorker {
                    token,
                    addr: worker_addr,
                    max_instances,
                },
            );
            token
        }
    };

    propagate_capacity_to_parent(ctx).await?;

    info!(%worker_addr, ?max_instances, token = %token, "registered tunnel worker");
    Ok(QuarkOut::TunnelRegistered { token })
}

async fn handle_update_tunnel_capacity(
    token: Uuid,
    max_instances: Option<usize>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    let max_instances = resolve_max_instances(max_instances);
    let mut workers = ctx.workers.write().await;
    let worker = workers
        .get_mut(&token)
        .ok_or(ServerError::TunnelWorkerUnavailable(token))?;
    worker.max_instances = max_instances;
    drop(workers);

    propagate_capacity_to_parent(ctx).await?;

    info!(token = %token, ?max_instances, "updated tunnel worker capacity");
    Ok(QuarkOut::Ack)
}

async fn handle_tunnel_forward(
    token: Uuid,
    request: TunnelRequest,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match ctx.mode {
        QuarkMode::Worker { tunnel_token, .. } if tunnel_token == token => {
            handle_tunnel_request_local(request, ctx).await
        }
        QuarkMode::Worker { .. } => Err(ServerError::TunnelUnauthorizedForward),
        QuarkMode::Root => Err(ServerError::TunnelForwardUnsupportedOnRoot),
    }
}

async fn handle_tunnel_request_local(
    request: TunnelRequest,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match request {
        TunnelRequest::Start {
            model_id,
            model_config,
        } => handle_start_distributed(model_id, model_config, ctx).await,
        TunnelRequest::PerturbUp { model_id, seed } => {
            handle_perturb_up_distributed(model_id, seed, ctx).await
        }
        TunnelRequest::Infer { model_id, input_id } => {
            handle_infer_distributed(model_id, input_id, ctx).await
        }
        TunnelRequest::Reset { model_id } => handle_reset_distributed(model_id, ctx).await,
        TunnelRequest::PerturbDown { model_id } => {
            handle_perturb_down_distributed(model_id, ctx).await
        }
        TunnelRequest::Checkpoint { model_id } => {
            handle_checkpoint_distributed(model_id, ctx).await
        }
        TunnelRequest::Optimize {
            model_id,
            loss_up,
            loss_down,
        } => handle_optimize_distributed(model_id, loss_up, loss_down, ctx).await,
        TunnelRequest::Shutdown { model_id } => handle_shutdown_distributed(model_id, ctx).await,
    }
}

async fn route_for_model(model_id: Uuid, ctx: &QuarkContext) -> Result<RouteTarget> {
    ctx.routes
        .read()
        .await
        .get(&model_id)
        .copied()
        .ok_or(ServerError::ModelInstanceNotRunning(model_id))
}

fn has_capacity(limit: Option<usize>, current: usize) -> bool {
    limit.is_none_or(|max| current < max)
}

async fn select_start_target(ctx: &QuarkContext) -> Result<RouteTarget> {
    let routes = ctx.routes.read().await;
    let mut local_count = 0usize;
    let mut worker_counts: HashMap<Uuid, usize> = HashMap::new();
    for target in routes.values() {
        match target {
            RouteTarget::Local => local_count += 1,
            RouteTarget::Worker(token) => {
                *worker_counts.entry(*token).or_insert(0) += 1;
            }
        }
    }
    drop(routes);

    let mut best: Option<(RouteTarget, usize)> = None;
    if has_capacity(ctx.max_instances, local_count) {
        best = Some((RouteTarget::Local, local_count));
    }

    let mut workers: Vec<TunnelWorker> = ctx.workers.read().await.values().copied().collect();
    workers.sort_by_key(|worker| worker.token);
    for worker in workers {
        let current = worker_counts
            .get(&worker.token)
            .copied()
            .unwrap_or_default();
        if !has_capacity(worker.max_instances, current) {
            continue;
        }
        match best {
            Some((_, best_count)) if best_count <= current => {}
            _ => best = Some((RouteTarget::Worker(worker.token), current)),
        }
    }

    best.map(|(target, _)| target)
        .ok_or(ServerError::NoTunnelCapacity)
}

async fn get_worker(token: Uuid, ctx: &QuarkContext) -> Result<TunnelWorker> {
    ctx.workers
        .read()
        .await
        .get(&token)
        .copied()
        .ok_or(ServerError::TunnelWorkerUnavailable(token))
}

async fn forward_tunnel_request(worker: TunnelWorker, request: TunnelRequest) -> Result<QuarkOut> {
    let client = QuarkRpcClient::connect(worker.addr).await?;
    let out = client
        .request(QuarkIn::TunnelForward {
            token: worker.token,
            request,
        })
        .await?;
    match out {
        QuarkOut::Error { message } => Err(ServerError::TunnelWorkerError(message)),
        _ => Ok(out),
    }
}

async fn handle_start_routed(
    model_id: Uuid,
    model_config: Option<QuarkModelConfig>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_start_distributed(model_id, model_config, ctx).await
}

async fn handle_start_distributed(
    model_id: Uuid,
    model_config: Option<QuarkModelConfig>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    if ctx.routes.read().await.contains_key(&model_id)
        || ctx.instances.read().await.contains_key(&model_id)
    {
        return Err(ServerError::ModelInstanceAlreadyRunning(model_id));
    }

    let target = select_start_target(ctx).await?;
    let out = match target {
        RouteTarget::Local => handle_start(model_id, model_config, ctx).await?,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(
                worker,
                TunnelRequest::Start {
                    model_id,
                    model_config,
                },
            )
            .await?
        }
    };

    if !matches!(out, QuarkOut::Ack) {
        return Err(ServerError::UnexpectedTunnelResponse("start response"));
    }

    ctx.routes.write().await.insert(model_id, target);
    Ok(QuarkOut::Ack)
}

async fn handle_perturb_up_routed(
    model_id: Uuid,
    seed: u64,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_perturb_up_distributed(model_id, seed, ctx).await
}

async fn handle_perturb_up_distributed(
    model_id: Uuid,
    seed: u64,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_perturb_up(model_id, seed, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::PerturbUp { model_id, seed }).await
        }
    }
}

async fn handle_infer_routed(
    model_id: Uuid,
    input_id: ObjectId,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_infer_distributed(model_id, input_id, ctx).await
}

async fn handle_infer_distributed(
    model_id: Uuid,
    input_id: ObjectId,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_infer(model_id, input_id, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::Infer { model_id, input_id }).await
        }
    }
}

async fn handle_reset_routed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_reset_distributed(model_id, ctx).await
}

async fn handle_reset_distributed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_reset(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::Reset { model_id }).await
        }
    }
}

async fn handle_perturb_down_routed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_perturb_down_distributed(model_id, ctx).await
}

async fn handle_perturb_down_distributed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_perturb_down(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::PerturbDown { model_id }).await
        }
    }
}

async fn handle_checkpoint_routed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_checkpoint_distributed(model_id, ctx).await
}

async fn handle_checkpoint_distributed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_checkpoint(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::Checkpoint { model_id }).await
        }
    }
}

async fn handle_optimize_routed(
    model_id: Uuid,
    loss_up: f32,
    loss_down: f32,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_optimize_distributed(model_id, loss_up, loss_down, ctx).await
}

async fn handle_optimize_distributed(
    model_id: Uuid,
    loss_up: f32,
    loss_down: f32,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_optimize(model_id, loss_up, loss_down, ctx).await,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(
                worker,
                TunnelRequest::Optimize {
                    model_id,
                    loss_up,
                    loss_down,
                },
            )
            .await
        }
    }
}

async fn handle_shutdown_routed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_shutdown_distributed(model_id, ctx).await
}

async fn handle_shutdown_distributed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    let target = route_for_model(model_id, ctx).await?;
    let out = match target {
        RouteTarget::Local => handle_shutdown(model_id, ctx).await?,
        RouteTarget::Worker(token) => {
            let worker = get_worker(token, ctx).await?;
            forward_tunnel_request(worker, TunnelRequest::Shutdown { model_id }).await?
        }
    };
    if matches!(out, QuarkOut::Ack) {
        ctx.routes.write().await.remove(&model_id);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Model instance lifecycle
// ---------------------------------------------------------------------------

async fn handle_start(
    model_id: Uuid,
    model_config: Option<QuarkModelConfig>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    {
        let mut instances = ctx.instances.write().await;
        if let Some(limit) = ctx.max_instances {
            if instances.len() >= limit {
                return Err(ServerError::NoLocalCapacity(limit));
            }
        }
        match instances.entry(model_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(ModelSlot::Starting);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(ServerError::ModelInstanceAlreadyRunning(model_id));
            }
        }
    }

    info!(%model_id, "starting model instance");
    let resolved_source = match resolve_model_source(model_id, model_config.as_ref(), ctx).await {
        Ok(source) => source,
        Err(error) => {
            ctx.instances.write().await.remove(&model_id);
            return Err(error);
        }
    };
    let model_path = resolved_source.model_path.clone();
    let tokenizer_path = resolved_source.tokenizer_path.clone();
    let checkpoint_path = resolved_source.checkpoint_path.clone();
    let defaults = ctx.defaults.with_overrides(model_config.as_ref());
    let inference_limit = defaults.inference_limit;
    let frozen = resolve_model_frozen(ctx.frozen, model_config.as_ref());
    let oscillation = match resolve_model_oscillation(model_config.as_ref()) {
        Ok(oscillation) => oscillation,
        Err(error) => {
            ctx.instances.write().await.remove(&model_id);
            if let Some(path) = checkpoint_path.as_ref() {
                cleanup_checkpoint_file(path);
            }
            return Err(error);
        }
    };
    let engine_result = match tokio::task::spawn_blocking(move || {
        let mut builder = paramecia_engine::ModelEngineBuilder::new(model_path)
            .top_k(defaults.top_k)
            .temperature(defaults.temperature)
            .kv_cache_quant(defaults.kv_cache_quant)
            .repeat_penalty(defaults.repeat_penalty)
            .presence_penalty(defaults.presence_penalty)
            .training_config(defaults.training_config);
        if let Some(top_p) = defaults.top_p {
            builder = builder.top_p(top_p);
        }
        if let Some(tokenizer_path) = tokenizer_path {
            builder = builder.tokenizer_path(tokenizer_path);
        }
        builder
            .build()
            .map_err(|error| ServerError::ModelError(error.to_string()))
    })
    .await
    {
        Ok(result) => result,
        Err(error) => {
            ctx.instances.write().await.remove(&model_id);
            if let Some(path) = checkpoint_path.as_ref() {
                cleanup_checkpoint_file(path);
            }
            return Err(ServerError::ModelError(format!(
                "model load task failed: {error}"
            )));
        }
    };

    let engine = match engine_result {
        Ok(engine) => engine,
        Err(error) => {
            ctx.instances.write().await.remove(&model_id);
            if let Some(path) = checkpoint_path.as_ref() {
                cleanup_checkpoint_file(path);
            }
            return Err(error);
        }
    };

    let instance = Arc::new(QuarkInstance {
        engine,
        inference_limit,
        oscillation,
        checkpoint_path,
        session: tokio::sync::Mutex::new(QuarkSession::new(frozen)),
    });
    ctx.instances
        .write()
        .await
        .insert(model_id, ModelSlot::Running(instance));

    info!(%model_id, "model instance started");
    Ok(QuarkOut::Ack)
}

async fn handle_shutdown(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    let instance = {
        let mut instances = ctx.instances.write().await;
        let slot = instances
            .get_mut(&model_id)
            .ok_or(ServerError::ModelInstanceNotRunning(model_id))?;
        match slot {
            ModelSlot::Running(instance) => {
                let instance = Arc::clone(instance);
                *slot = ModelSlot::ShuttingDown;
                instance
            }
            ModelSlot::Starting | ModelSlot::ShuttingDown => {
                return Err(ServerError::ModelInstanceNotRunning(model_id));
            }
        }
    };

    let mut session = instance.session.lock().await;
    session.running = false;
    let unload_result = instance
        .engine
        .unload_model()
        .await
        .map_err(|error| ServerError::ModelError(error.to_string()));
    if let Some(path) = instance.checkpoint_path.as_ref() {
        cleanup_checkpoint_file(path);
    }
    ctx.instances.write().await.remove(&model_id);
    unload_result?;

    info!(%model_id, "model instance shut down");
    Ok(QuarkOut::Ack)
}

async fn get_instance(model_id: Uuid, ctx: &QuarkContext) -> Result<Arc<QuarkInstance>> {
    match ctx.instances.read().await.get(&model_id) {
        Some(ModelSlot::Running(instance)) => Ok(Arc::clone(instance)),
        Some(ModelSlot::Starting | ModelSlot::ShuttingDown) | None => {
            Err(ServerError::ModelInstanceNotRunning(model_id))
        }
    }
}

fn ensure_running(session: &QuarkSession, model_id: Uuid) -> Result<()> {
    if session.running {
        Ok(())
    } else {
        Err(ServerError::ModelInstanceNotRunning(model_id))
    }
}

fn resolve_model_frozen(server_frozen: bool, model_config: Option<&QuarkModelConfig>) -> bool {
    model_config
        .and_then(|model_config| model_config.frozen)
        .unwrap_or(server_frozen)
}

fn resolve_model_oscillation(
    model_config: Option<&QuarkModelConfig>,
) -> Result<Option<FrozenOscillation>> {
    let Some(config) = model_config else {
        return Ok(None);
    };
    let Some(steps) = config.oscillation_steps else {
        return Ok(None);
    };
    if steps == 0 {
        return Err(ServerError::InvalidOscillationSteps(steps));
    }
    Ok(Some(FrozenOscillation {
        steps,
        offset: config.oscillation_offset.unwrap_or_default(),
    }))
}

fn apply_frozen_oscillation(
    model_id: Uuid,
    session: &mut QuarkSession,
    oscillation: Option<FrozenOscillation>,
) {
    let Some(oscillation) = oscillation else {
        return;
    };
    session.optimize_steps = session.optimize_steps.saturating_add(1);
    if session.optimize_steps > oscillation.offset
        && (session.optimize_steps - oscillation.offset) % oscillation.steps == 0
    {
        session.frozen = !session.frozen;
        info!(
            %model_id,
            frozen = session.frozen,
            optimize_steps = session.optimize_steps,
            oscillation_steps = oscillation.steps,
            oscillation_offset = oscillation.offset,
            "flipped model frozen state by oscillation config"
        );
    }
}

fn resolve_checkpoint_tokenizer_path() -> Result<PathBuf> {
    let base_dirs =
        directories_next::BaseDirs::new().ok_or(ServerError::HomeDirectoryUnavailable)?;
    let home_dir = base_dirs.home_dir();
    let tokenizer_path = home_dir
        .join(DEFAULT_CHECKPOINT_TOKENIZER_DIR)
        .join(DEFAULT_CHECKPOINT_TOKENIZER_FILE);
    if tokenizer_path.exists() {
        Ok(tokenizer_path)
    } else {
        Err(ServerError::CheckpointTokenizerMissing(tokenizer_path))
    }
}

fn write_checkpoint_file(
    model_id: Uuid,
    checkpoint_id: ObjectId,
    checkpoint_bytes: &[u8],
) -> Result<PathBuf> {
    let checkpoint_dir = std::env::temp_dir().join(CHECKPOINT_CACHE_DIR);
    fs::create_dir_all(&checkpoint_dir).map_err(|source| ServerError::CreateCheckpointDir {
        path: checkpoint_dir.clone(),
        source,
    })?;
    let checkpoint_path = checkpoint_dir.join(format!("{model_id}-{checkpoint_id}.gguf"));
    fs::write(&checkpoint_path, checkpoint_bytes).map_err(|source| {
        ServerError::WriteCheckpoint {
            path: checkpoint_path.clone(),
            source,
        }
    })?;
    Ok(checkpoint_path)
}

fn cleanup_checkpoint_file(path: &PathBuf) {
    if let Err(error) = fs::remove_file(path) {
        warn!(
            path = %path.display(),
            error = %error,
            "failed to remove temporary checkpoint file"
        );
    }
}

async fn resolve_model_source(
    model_id: Uuid,
    model_config: Option<&QuarkModelConfig>,
    ctx: &QuarkContext,
) -> Result<ResolvedModelSource> {
    let Some(checkpoint_id) = model_config.and_then(|config| config.checkpoint_id) else {
        return Ok(ResolvedModelSource {
            model_path: ctx.model_path.clone(),
            tokenizer_path: None,
            checkpoint_path: None,
        });
    };

    let tokenizer_path = resolve_checkpoint_tokenizer_path()?;
    let void = ctx
        .void_client
        .as_ref()
        .ok_or_else(|| ServerError::VoidNotConfigured)?;
    let checkpoint_bytes = void.download(checkpoint_id).await?;
    if checkpoint_bytes.is_empty() {
        return Err(ServerError::CheckpointEmpty(checkpoint_id));
    }
    let checkpoint_path = write_checkpoint_file(model_id, checkpoint_id, &checkpoint_bytes)?;
    Ok(ResolvedModelSource {
        model_path: checkpoint_path.clone(),
        tokenizer_path: Some(tokenizer_path),
        checkpoint_path: Some(checkpoint_path),
    })
}

async fn reset_model(engine: &ModelEngine) -> Result<()> {
    engine
        .reset_state()
        .await
        .map_err(|error| ServerError::ModelError(error.to_string()))
}

async fn checkpoint_model(engine: &ModelEngine) -> Result<Vec<u8>> {
    let checkpoint_path = engine
        .save_checkpoint()
        .await
        .map_err(|error| ServerError::ModelError(error.to_string()))?;
    let checkpoint_bytes = fs::read(&checkpoint_path).map_err(|error| {
        ServerError::ModelError(format!(
            "failed to read checkpoint {}: {error}",
            checkpoint_path.display()
        ))
    })?;
    if let Err(error) = fs::remove_file(&checkpoint_path) {
        warn!(
            path = %checkpoint_path.display(),
            error = %error,
            "failed to remove temporary checkpoint file"
        );
    }
    Ok(checkpoint_bytes)
}

// ---------------------------------------------------------------------------
// QuZO step handlers
// ---------------------------------------------------------------------------

async fn handle_perturb_up(model_id: Uuid, seed: u64, ctx: &QuarkContext) -> Result<QuarkOut> {
    info!(%model_id, "received perturb up request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != QuarkState::Idle {
        warn!("expected Idle, got {:?}", session.state);
        return Err(ServerError::InvalidQuarkState(format!(
            "expected Idle, got {:?}",
            session.state
        )));
    }

    if session.frozen {
        info!(%model_id, "skipping perturb up because model instance is frozen");
    } else {
        instance
            .engine
            .perturb_up(Some(seed))
            .await
            .map_err(|e| ServerError::ModelError(e.to_string()))?;
    }

    session.state = QuarkState::PostPerturbUp;
    Ok(QuarkOut::Ack)
}

async fn handle_reset(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    info!(%model_id, "received reset request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    reset_model(&instance.engine).await?;
    Ok(QuarkOut::Ack)
}

async fn handle_infer(model_id: Uuid, input_id: ObjectId, ctx: &QuarkContext) -> Result<QuarkOut> {
    info!(%model_id, "received inference request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if !matches!(
        session.state,
        QuarkState::Idle
            | QuarkState::PostPerturbUp
            | QuarkState::AwaitingPerturbDown
            | QuarkState::PostPerturbDown
            | QuarkState::AwaitingOptimize
    ) {
        warn!(
            "inference requires Idle or an active perturbation phase, got {:?}",
            session.state
        );
        return Err(ServerError::InvalidQuarkState(format!(
            "inference requires Idle or an active perturbation phase, got {:?}",
            session.state
        )));
    }
    let state = session.state;

    // Resolve void client.
    let void = ctx
        .void_client
        .as_ref()
        .ok_or_else(|| ServerError::VoidNotConfigured)?;

    // Download input object from void and decode the inference request.
    let input_bytes = void.download(input_id).await?;
    let infer_req: InferenceRequest = from_bytes(&input_bytes).map_err(ServerError::DecodeFrame)?;

    // Resolve sequences and limit from the request variant.
    let (sequences, limit) = match infer_req {
        InferenceRequest::Sequences {
            sequences: raw_seqs,
            limit,
        } => {
            let seqs: Vec<Vec<paramecia_engine::ModelInput>> = raw_seqs
                .into_iter()
                .map(|seq_inputs| {
                    seq_inputs
                        .into_iter()
                        .map(|inp| match inp {
                            InferenceInput::Text(t) => paramecia_engine::ModelInput::Text(t),
                            InferenceInput::Tokens(ids) => {
                                paramecia_engine::ModelInput::Tokens(ids)
                            }
                            InferenceInput::Dark(tokens) => paramecia_engine::ModelInput::Soft(
                                tokens
                                    .into_iter()
                                    .map(|t| paramecia_engine::SoftToken {
                                        predicted: t.predicted,
                                        dark_knowledge: t
                                            .dark_knowledge
                                            .into_iter()
                                            .map(|e| paramecia_engine::LogitEntry {
                                                token_id: e.token_id,
                                                log_prob: e.log_prob,
                                            })
                                            .collect(),
                                    })
                                    .collect(),
                            ),
                        })
                        .collect()
                })
                .collect();
            (seqs, limit)
        }
        InferenceRequest::VoidId { id, limit } => {
            // Download the InferenceOutput and convert to dark input sequences.
            let output_bytes = void.download(id.0).await?;
            let inference_output: InferenceOutput =
                from_bytes(&output_bytes).map_err(ServerError::DecodeFrame)?;

            let seqs: Vec<Vec<paramecia_engine::ModelInput>> = inference_output
                .results
                .into_iter()
                .map(|seq_out| {
                    vec![paramecia_engine::ModelInput::Soft(
                        seq_out
                            .0
                            .into_iter()
                            .map(|tok| paramecia_engine::SoftToken {
                                predicted: tok.predicted,
                                dark_knowledge: tok
                                    .dark_knowledge
                                    .into_iter()
                                    .map(|e| paramecia_engine::LogitEntry {
                                        token_id: e.token_id,
                                        log_prob: e.log_prob,
                                    })
                                    .collect(),
                            })
                            .collect(),
                    )]
                })
                .collect();
            (seqs, limit)
        }
    };
    let limit = limit.unwrap_or(instance.inference_limit);

    // Run batched inference.
    let seq_results = run_batched_inference(&instance.engine, &sequences, limit).await?;

    // Convert per-sequence predictions to serializable output.
    let output = InferenceOutput {
        results: seq_results
            .into_iter()
            .map(|predictions| {
                SequenceOutput(
                    predictions
                        .into_iter()
                        .map(|p| DarkToken {
                            predicted: p.token_id,
                            dark_knowledge: p
                                .top_k
                                .into_iter()
                                .map(|e| LogitEntry {
                                    token_id: e.token_id,
                                    log_prob: e.log_prob,
                                })
                                .collect(),
                        })
                        .collect(),
                )
            })
            .collect(),
    };

    // Upload output to void.
    let output_bytes = to_allocvec(&output).map_err(ServerError::EncodeFrame)?;
    let output_id = void.upload(output_bytes).await?;

    // Advance state.
    session.state = match state {
        QuarkState::PostPerturbUp | QuarkState::AwaitingPerturbDown => {
            QuarkState::AwaitingPerturbDown
        }
        QuarkState::Idle | QuarkState::PostPerturbDown | QuarkState::AwaitingOptimize => {
            QuarkState::AwaitingOptimize
        }
    };

    info!(%model_id, "finished processing inference request");
    Ok(QuarkOut::Inferred { output_id })
}

async fn handle_perturb_down(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    info!(%model_id, "received perturb down request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != QuarkState::AwaitingPerturbDown {
        warn!("expected AwaitingPerturbDown, got {:?}", session.state);
        return Err(ServerError::InvalidQuarkState(format!(
            "expected AwaitingPerturbDown, got {:?}",
            session.state
        )));
    }

    if session.frozen {
        info!(%model_id, "skipping perturb down because model instance is frozen");
    } else {
        instance
            .engine
            .perturb_down()
            .await
            .map_err(|e| ServerError::ModelError(e.to_string()))?;
    }

    session.state = QuarkState::PostPerturbDown;
    Ok(QuarkOut::Ack)
}

async fn handle_checkpoint(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    info!(%model_id, "received checkpoint request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;

    let void = ctx
        .void_client
        .as_ref()
        .ok_or_else(|| ServerError::VoidNotConfigured)?;
    let checkpoint_bytes = checkpoint_model(&instance.engine).await?;
    let checkpoint_id = void.upload(checkpoint_bytes).await?;

    Ok(QuarkOut::Checkpointed { checkpoint_id })
}

async fn handle_optimize(
    model_id: Uuid,
    loss_up: f32,
    loss_down: f32,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    info!(%model_id, "received optimization request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != QuarkState::AwaitingOptimize {
        warn!("expected AwaitingOptimize, got {:?}", session.state);
        return Err(ServerError::InvalidQuarkState(format!(
            "expected AwaitingOptimize, got {:?}",
            session.state
        )));
    }

    reset_model(&instance.engine).await?;
    if session.frozen {
        info!(%model_id, "skipping optimization because model instance is frozen");
    } else {
        instance
            .engine
            .update(loss_up, loss_down)
            .await
            .map_err(|e| {
                warn!("optimization failed");
                ServerError::ModelError(e.to_string())
            })?;
    }

    session.state = QuarkState::Idle;
    apply_frozen_oscillation(model_id, &mut session, instance.oscillation);
    info!(%model_id, "finished optimization update");
    Ok(QuarkOut::Ack)
}

// ---------------------------------------------------------------------------
// Inference helper
// ---------------------------------------------------------------------------

async fn run_batched_inference(
    engine: &ModelEngine,
    sequences: &[Vec<paramecia_engine::ModelInput>],
    limit: u32,
) -> Result<Vec<Vec<paramecia_engine::Predicted>>> {
    if limit == 0 {
        return Ok(vec![Vec::new(); sequences.len()]);
    }

    // Start batched streaming completion — returns (result_rx, cancel_tx).
    let (mut result_rx, _cancel_tx) = engine
        .predict_completions_batched(sequences)
        .await
        .map_err(|e| ServerError::ModelError(e.to_string()))?;

    let n_seqs = sequences.len();
    // Accumulate per-sequence predictions: seq_results[i] holds all Predicted for sequence i.
    let mut seq_results: Vec<Vec<paramecia_engine::Predicted>> = vec![Vec::new(); n_seqs];
    let mut done = false;

    while !done {
        let Some(result) = result_rx.recv().await else {
            break;
        };
        match result {
            Ok(step_predictions) => {
                // step_predictions has one Predicted per sequence for this decode step.
                for (i, pred) in step_predictions.into_iter().enumerate() {
                    if i < n_seqs && seq_results[i].len() < limit as usize {
                        seq_results[i].push(pred);
                    }
                }
                // Stop when all sequences have reached the limit.
                done = seq_results.iter().all(|s| s.len() >= limit as usize);
            }
            Err(e) => {
                // Non-fatal errors (e.g. max length) are fine.
                warn!(error = %e, "batched prediction ended with error");
                break;
            }
        }
    }

    Ok(seq_results)
}

// Certificate helpers
// ---------------------------------------------------------------------------

fn load_user_cert_chain_and_key(
    key_path: &PathBuf,
    cert_path: &PathBuf,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let key = if key_path.extension().is_some_and(|x| x == "der") {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            fs::read(key_path).map_err(ServerError::ReadPrivateKeyFile)?,
        ))
    } else {
        PrivateKeyDer::from_pem_file(key_path).map_err(ServerError::ReadPrivateKeyPem)?
    };

    let cert_chain = if cert_path.extension().is_some_and(|x| x == "der") {
        vec![CertificateDer::from(
            fs::read(cert_path).map_err(ServerError::ReadCertChainFile)?,
        )]
    } else {
        CertificateDer::pem_file_iter(cert_path)
            .map_err(ServerError::ReadCertChainPem)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(ServerError::InvalidPemCert)?
    };

    Ok((cert_chain, key))
}

fn load_or_generate_self_signed_cert(
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let dirs = directories_next::ProjectDirs::from("org", "blackhole", "quark").unwrap();
    let path = dirs.data_local_dir();
    let cert_path = path.join("cert.der");
    let key_path = path.join("key.der");

    let (cert, key) = match fs::read(&cert_path).and_then(|x| Ok((x, fs::read(&key_path)?))) {
        Ok((cert, key)) => (
            CertificateDer::from(cert),
            PrivateKeyDer::try_from(key).map_err(|e| ServerError::ParseDerKey(e.to_owned()))?,
        ),
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
            info!("generating self-signed certificate");
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
            let key = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
            let cert = cert.cert.into();
            fs::create_dir_all(path).map_err(ServerError::CreateCertDir)?;
            fs::write(&cert_path, &cert).map_err(ServerError::WriteCert)?;
            fs::write(&key_path, key.secret_pkcs8_der()).map_err(ServerError::WritePrivateKey)?;
            (cert, key.into())
        }
        Err(e) => {
            return Err(ServerError::ReadCertificate(e));
        }
    };

    Ok((vec![cert], key))
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("failed to read private key file: {0}")]
    ReadPrivateKeyFile(#[source] io::Error),
    #[error("failed to read PEM from private key file: {0}")]
    ReadPrivateKeyPem(#[source] rustls::pki_types::pem::Error),
    #[error("failed to read certificate chain file: {0}")]
    ReadCertChainFile(#[source] io::Error),
    #[error("failed to read PEM from certificate chain file: {0}")]
    ReadCertChainPem(#[source] rustls::pki_types::pem::Error),
    #[error("invalid PEM-encoded certificate: {0}")]
    InvalidPemCert(#[source] rustls::pki_types::pem::Error),
    #[error("failed to parse DER private key: {0}")]
    ParseDerKey(String),
    #[error("failed to create certificate directory: {0}")]
    CreateCertDir(#[source] io::Error),
    #[error("failed to write certificate: {0}")]
    WriteCert(#[source] io::Error),
    #[error("failed to write private key: {0}")]
    WritePrivateKey(#[source] io::Error),
    #[error("failed to read certificate: {0}")]
    ReadCertificate(io::Error),
    #[error("failed to configure rustls certificate: {0}")]
    RustlsCertConfig(#[source] rustls::Error),
    #[error("failed to build QUIC rustls config: {0}")]
    QuicServerConfig(#[source] quinn::crypto::rustls::NoInitialCipherSuite),
    #[error("failed to bind QUIC endpoint: {0}")]
    BindEndpoint(#[source] io::Error),
    #[error("failed to fetch local listen address: {0}")]
    LocalAddr(#[source] io::Error),
    #[error("model error: {0}")]
    ModelError(String),
    #[error("model instance {0} is already running")]
    ModelInstanceAlreadyRunning(Uuid),
    #[error("model instance {0} is not running")]
    ModelInstanceNotRunning(Uuid),
    #[error("invalid Quark state machine transition: {0}")]
    InvalidQuarkState(String),
    #[error("oscillation_steps must be greater than zero, got {0}")]
    InvalidOscillationSteps(usize),
    #[error("void service not configured")]
    VoidNotConfigured,
    #[error("failed to resolve home directory for checkpoint tokenizer")]
    HomeDirectoryUnavailable,
    #[error("checkpoint start requires tokenizer file at {0}")]
    CheckpointTokenizerMissing(PathBuf),
    #[error("checkpoint {0} downloaded from void is empty")]
    CheckpointEmpty(ObjectId),
    #[error("failed to create checkpoint cache directory {path}: {source}")]
    CreateCheckpointDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write checkpoint cache file {path}: {source}")]
    WriteCheckpoint {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to bind void client endpoint: {0}")]
    BindVoidClient(#[source] io::Error),
    #[error("failed to connect to void: {0}")]
    VoidConnect(String),
    #[error("failed to open void stream: {0}")]
    VoidStream(String),
    #[error("void crypto config error: {0}")]
    VoidCrypto(String),
    #[error("void error: {0}")]
    VoidError(String),
    #[error("failed to bind tunnel client endpoint: {0}")]
    BindTunnelClient(#[source] io::Error),
    #[error("failed to connect to tunnel peer: {0}")]
    TunnelConnect(String),
    #[error("failed to open tunnel stream: {0}")]
    TunnelStream(String),
    #[error("tunnel crypto config error: {0}")]
    TunnelCrypto(String),
    #[error("tunnel registration rejected: {0}")]
    TunnelRegistrationRejected(String),
    #[error("tunnel capacity update rejected: {0}")]
    TunnelCapacityUpdateRejected(String),
    #[error("tunnel worker rejects direct model requests")]
    TunnelWorkerRejectsModelRequests,
    #[error("unauthorized tunnel forward request")]
    TunnelUnauthorizedForward,
    #[error("tunnel forward request is only valid on worker quarks")]
    TunnelForwardUnsupportedOnRoot,
    #[error("no quark capacity available across local and registered workers")]
    NoTunnelCapacity,
    #[error("local quark reached max_instances capacity ({0})")]
    NoLocalCapacity(usize),
    #[error("tunnel worker {0} is unavailable")]
    TunnelWorkerUnavailable(Uuid),
    #[error("tunnel worker returned error: {0}")]
    TunnelWorkerError(String),
    #[error("unexpected tunnel protocol response: {0}")]
    UnexpectedTunnelResponse(&'static str),
    #[error("unexpected EOF while reading frame length")]
    UnexpectedEof,
    #[error("failed to read frame length: {0}")]
    ReadFrameLength(#[source] io::Error),
    #[error("frame payload too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("failed to read frame payload: {0}")]
    ReadFramePayload(#[source] quinn::ReadExactError),
    #[error("failed to decode frame: {0}")]
    DecodeFrame(postcard::Error),
    #[error("failed to encode frame: {0}")]
    EncodeFrame(postcard::Error),
    #[error("failed to write frame: {0}")]
    WriteFrame(quinn::WriteError),
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{
        apply_frozen_oscillation, handle_register_tunnel, resolve_max_instances,
        resolve_model_frozen, resolve_model_oscillation, FrozenOscillation, QuarkContext,
        QuarkMode, QuarkServerDefaults, QuarkSession, QuarkState, ServerBuilder,
        DEFAULT_INFERENCE_LIMIT, DEFAULT_MAX_INSTANCES,
    };
    use black_hole_spec::QuarkModelConfig;
    use std::{collections::HashMap, net::SocketAddr, path::PathBuf};
    use tokio::sync::RwLock;

    #[test]
    fn model_config_none_passes_through_server_defaults() {
        let defaults = QuarkServerDefaults::default();
        let resolved = defaults.with_overrides(None);

        assert_eq!(resolved.top_k, defaults.top_k);
        assert_eq!(resolved.temperature, defaults.temperature);
        assert_eq!(resolved.top_p, defaults.top_p);
        assert_eq!(resolved.repeat_penalty, defaults.repeat_penalty);
        assert_eq!(resolved.presence_penalty, defaults.presence_penalty);
        assert_eq!(resolved.inference_limit, defaults.inference_limit);
        assert_eq!(resolved.training_config.lr, defaults.training_config.lr);
        assert_eq!(
            resolved.training_config.epsilon,
            defaults.training_config.epsilon
        );
    }

    #[test]
    fn model_config_overrides_selected_fields() {
        let defaults = QuarkServerDefaults::default();
        let resolved = defaults.with_overrides(Some(&QuarkModelConfig {
            top_k: Some(64),
            temperature: Some(0.2),
            top_p: Some(0.9),
            repeat_penalty: Some(1.3),
            presence_penalty: None,
            inference_limit: Some(12),
            training_lr: Some(0.0002),
            training_epsilon: Some(0.001),
            frozen: None,
            oscillation_steps: None,
            oscillation_offset: None,
            checkpoint_id: None,
        }));

        assert_eq!(resolved.top_k, 64);
        assert_eq!(resolved.temperature, 0.2);
        assert_eq!(resolved.top_p, Some(0.9));
        assert_eq!(resolved.repeat_penalty, 1.3);
        assert_eq!(resolved.presence_penalty, defaults.presence_penalty);
        assert_eq!(resolved.inference_limit, 12);
        assert_eq!(resolved.training_config.lr, 0.0002);
        assert_eq!(resolved.training_config.epsilon, 0.001);
        assert_ne!(resolved.inference_limit, DEFAULT_INFERENCE_LIMIT);
    }

    #[test]
    fn model_config_frozen_override_falls_back_to_server_default() {
        assert!(resolve_model_frozen(true, None));
        assert!(!resolve_model_frozen(false, None));
    }

    #[test]
    fn model_config_frozen_override_can_freeze_or_unfreeze_instance() {
        assert!(resolve_model_frozen(
            false,
            Some(&QuarkModelConfig {
                frozen: Some(true),
                ..QuarkModelConfig::default()
            })
        ));
        assert!(!resolve_model_frozen(
            true,
            Some(&QuarkModelConfig {
                frozen: Some(false),
                ..QuarkModelConfig::default()
            })
        ));
    }

    #[test]
    fn model_config_oscillation_defaults_to_none() {
        assert_eq!(resolve_model_oscillation(None).unwrap(), None);
        assert_eq!(
            resolve_model_oscillation(Some(&QuarkModelConfig {
                oscillation_offset: Some(7),
                ..QuarkModelConfig::default()
            }))
            .unwrap(),
            None
        );
    }

    #[test]
    fn model_config_oscillation_resolves_steps_and_offset() {
        let resolved = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_steps: Some(10),
            oscillation_offset: Some(20),
            ..QuarkModelConfig::default()
        }))
        .expect("oscillation config should resolve")
        .expect("oscillation config should be present");
        assert_eq!(
            resolved,
            FrozenOscillation {
                steps: 10,
                offset: 20
            }
        );
    }

    #[test]
    fn model_config_oscillation_rejects_zero_steps() {
        let err = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_steps: Some(0),
            ..QuarkModelConfig::default()
        }))
        .expect_err("zero oscillation steps should be rejected");
        assert!(matches!(
            err,
            super::ServerError::InvalidOscillationSteps(0)
        ));
    }

    #[test]
    fn oscillation_flips_frozen_after_offset_and_every_cadence() {
        let mut session = QuarkSession {
            state: QuarkState::AwaitingOptimize,
            running: true,
            frozen: true,
            optimize_steps: 0,
        };
        let oscillation = Some(FrozenOscillation {
            steps: 10,
            offset: 20,
        });
        let model_id = uuid::Uuid::new_v4();
        let mut flips = Vec::new();
        for step in 1..=50 {
            let before = session.frozen;
            apply_frozen_oscillation(model_id, &mut session, oscillation);
            if session.frozen != before {
                flips.push(step);
            }
        }
        assert_eq!(flips, vec![30, 40, 50]);
        assert!(!session.frozen);
    }

    #[test]
    fn max_instances_defaults_to_one_when_omitted() {
        assert_eq!(resolve_max_instances(None), Some(DEFAULT_MAX_INSTANCES));
        assert_eq!(resolve_max_instances(Some(7)), Some(7));
    }

    #[test]
    fn server_builder_defaults_max_instances_to_one() {
        let builder = ServerBuilder::new("model-is-not-loaded-for-this-test");
        assert_eq!(builder.max_instances, Some(DEFAULT_MAX_INSTANCES));
    }

    #[tokio::test]
    async fn tunnel_registration_defaults_capacity_to_one_when_omitted() {
        let ctx = QuarkContext {
            local_addr: "127.0.0.1:61001".parse().expect("valid socket address"),
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: QuarkMode::Root,
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };
        let worker_addr: SocketAddr = "127.0.0.1:54321".parse().expect("valid socket address");

        let out = handle_register_tunnel(worker_addr, None, &ctx)
            .await
            .expect("registration should succeed");
        let token = match out {
            black_hole_spec::QuarkOut::TunnelRegistered { token } => token,
            other => panic!("unexpected registration response: {other:?}"),
        };
        let worker = ctx
            .workers
            .read()
            .await
            .get(&token)
            .copied()
            .expect("worker should be tracked");
        assert_eq!(worker.max_instances, Some(DEFAULT_MAX_INSTANCES));
    }

    #[tokio::test]
    async fn tunnel_registration_preserves_explicit_capacity() {
        let ctx = QuarkContext {
            local_addr: "127.0.0.1:61002".parse().expect("valid socket address"),
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: QuarkMode::Root,
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };
        let worker_addr: SocketAddr = "127.0.0.1:54322".parse().expect("valid socket address");
        let requested = Some(3usize);

        let out = handle_register_tunnel(worker_addr, requested, &ctx)
            .await
            .expect("registration should succeed");
        let token = match out {
            black_hole_spec::QuarkOut::TunnelRegistered { token } => token,
            other => panic!("unexpected registration response: {other:?}"),
        };
        let worker = ctx
            .workers
            .read()
            .await
            .get(&token)
            .copied()
            .expect("worker should be tracked");
        assert_eq!(worker.max_instances, requested);
    }
}

pub type Result<T> = std::result::Result<T, ServerError>;

pub fn init_tracing() -> std::result::Result<(), tracing::subscriber::SetGlobalDefaultError> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish(),
    )
}
