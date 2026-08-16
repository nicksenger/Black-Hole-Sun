use std::{
    collections::HashMap,
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use paramecia_engine::{
    ErrorFeedbackMode, ErrorFeedbackParams, HyperParameterUpdate, ModelEngine, ReplayParams,
    TrainingConfig,
};
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, Mutex},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use black_hole_spec::{
    DarkToken, InferenceInput, InferenceOutput, InferenceRequest, LogitEntry, ObjectId,
    QuarkErrorFeedbackConfig, QuarkIn, QuarkModelCapacity, QuarkModelConfig, QuarkModelParams,
    QuarkOut, SequenceOutput, TunnelRequest,
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
const DEFAULT_TUNNEL_CONNECT_RETRY_MS: u64 = 200;
const MAX_TUNNEL_CONNECT_RETRY_MS: u64 = 25_600;
const TUNNEL_REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(5);
const RESIDUAL_UPDATE_UNSUPPORTED_FRAGMENT: &str =
    "restore_and_update_with_residual not supported for ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    Quic,
    Tcp,
}

impl Default for TransportMode {
    fn default() -> Self {
        Self::Quic
    }
}

fn client_bind_addr_for(remote_addr: SocketAddr) -> SocketAddr {
    if remote_addr.is_ipv4() {
        "0.0.0.0:0".parse().expect("valid ipv4 any-address")
    } else {
        "[::]:0".parse().expect("valid ipv6 any-address")
    }
}

fn path_root(path: &Path) -> Option<&Path> {
    path.ancestors().last()
}

fn repair_duplicated_absolute_model_path(model_path: &Path, cwd: &Path) -> Option<PathBuf> {
    if !model_path.is_absolute() || model_path.exists() {
        return None;
    }

    let duplicated_suffix = model_path.strip_prefix(cwd).ok()?;
    if duplicated_suffix.as_os_str().is_empty() {
        return None;
    }

    let root = path_root(cwd)?;
    let repaired = root.join(duplicated_suffix);
    if repaired == model_path || !repaired.exists() {
        return None;
    }
    Some(repaired)
}

fn resolve_configured_model_path(model_path: &Path) -> PathBuf {
    let Ok(cwd) = std::env::current_dir() else {
        return model_path.to_path_buf();
    };
    if let Some(repaired) = repair_duplicated_absolute_model_path(model_path, &cwd) {
        warn!(
            configured_model_path = %model_path.display(),
            repaired_model_path = %repaired.display(),
            cwd = %cwd.display(),
            "corrected duplicated absolute model path"
        );
        repaired
    } else {
        model_path.to_path_buf()
    }
}

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

enum VoidClientTransport {
    Quic {
        endpoint: quinn::Endpoint,
        remote_addr: SocketAddr,
    },
    Tcp {
        remote_addr: SocketAddr,
    },
}

/// A client connection to the void object store.
pub struct VoidClient {
    transport: VoidClientTransport,
}

impl VoidClient {
    /// Connect to the void service at `addr`.
    pub async fn connect(addr: SocketAddr, transport_mode: TransportMode) -> Result<Self> {
        match transport_mode {
            TransportMode::Quic => Self::connect_quic(addr).await,
            TransportMode::Tcp => Ok(Self {
                transport: VoidClientTransport::Tcp { remote_addr: addr },
            }),
        }
    }

    async fn connect_quic(addr: SocketAddr) -> Result<Self> {
        // Void uses self-signed certs by default in local dev — skip verification.
        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PermissiveCertVerifier))
            .with_no_client_auth();

        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
            .map_err(|e| ServerError::VoidCrypto(e.to_string()))?;

        let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

        // Bind with a compatible local address family for the remote endpoint.
        let local_addr = client_bind_addr_for(addr);
        let mut endpoint =
            quinn::Endpoint::client(local_addr).map_err(ServerError::BindVoidClient)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            transport: VoidClientTransport::Quic {
                endpoint,
                remote_addr: addr,
            },
        })
    }

    /// Open a transport channel, send a void request, and read the response.
    async fn call(&self, req: VoidIn) -> Result<VoidOut> {
        match &self.transport {
            VoidClientTransport::Quic {
                endpoint,
                remote_addr,
            } => {
                let server_name = remote_addr.ip().to_string();
                let conn = endpoint
                    .connect(*remote_addr, &server_name)
                    .map_err(|e| ServerError::VoidConnect(e.to_string()))?
                    .await
                    .map_err(|e| ServerError::VoidConnect(e.to_string()))?;

                let (mut send, mut recv) = conn
                    .open_bi()
                    .await
                    .map_err(|e| ServerError::VoidStream(e.to_string()))?;

                write_frame_quic(&mut send, &req).await?;
                let resp = read_frame_quic(&mut recv).await?;
                Ok(resp)
            }
            VoidClientTransport::Tcp { remote_addr } => {
                let mut stream = TcpStream::connect(*remote_addr)
                    .await
                    .map_err(|e| ServerError::VoidTcpConnect(e.to_string()))?;
                write_frame_io(&mut stream, &req).await?;
                read_frame_io(&mut stream).await
            }
        }
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

enum QuarkRpcClientInner {
    Quic {
        endpoint: quinn::Endpoint,
        remote_addr: SocketAddr,
    },
    Tcp {
        remote_addr: SocketAddr,
    },
}

/// A client connection to another quark server.
struct QuarkRpcClient {
    inner: QuarkRpcClientInner,
}

#[derive(Clone)]
enum TunnelConnectionHandle {
    Quic(quinn::Connection),
    Tcp(Arc<TcpTunnelSession>),
}

impl TunnelConnectionHandle {
    async fn close(&self, reason: &'static [u8]) {
        match self {
            Self::Quic(connection) => connection.close(0u32.into(), reason),
            Self::Tcp(session) => session.close("tunnel tcp session replaced").await,
        }
    }
}

struct ParentTunnelSession {
    // Holds the client endpoint open for the lifetime of the parent tunnel.
    _client: QuarkRpcClient,
    connection: TunnelConnectionHandle,
}

enum RpcConnection {
    Quic(quinn::Connection),
    Tcp(TcpStream),
}

#[derive(Debug, Serialize, Deserialize)]
enum TunnelTcpEnvelope {
    Request { request_id: u64, request: QuarkIn },
    Response { request_id: u64, response: QuarkOut },
}

#[derive(Debug)]
struct TunnelTcpRequest {
    request_id: u64,
    request: QuarkIn,
}

struct TcpTunnelSession {
    outbound: mpsc::Sender<TunnelTcpEnvelope>,
    inbound: Mutex<mpsc::Receiver<TunnelTcpRequest>>,
    pending: Mutex<HashMap<u64, oneshot::Sender<QuarkOut>>>,
    next_request_id: AtomicU64,
    closed_tx: tokio::sync::watch::Sender<bool>,
    reader_abort: std::sync::Mutex<Option<tokio::task::AbortHandle>>,
    writer_abort: std::sync::Mutex<Option<tokio::task::AbortHandle>>,
}

impl TcpTunnelSession {
    fn new(stream: TcpStream) -> Arc<Self> {
        let (read_half, write_half) = tokio::io::split(stream);
        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let (inbound_tx, inbound_rx) = mpsc::channel(256);
        let (closed_tx, _closed_rx) = tokio::sync::watch::channel(false);
        let session = Arc::new(Self {
            outbound: outbound_tx,
            inbound: Mutex::new(inbound_rx),
            pending: Mutex::new(HashMap::new()),
            next_request_id: AtomicU64::new(1),
            closed_tx,
            reader_abort: std::sync::Mutex::new(None),
            writer_abort: std::sync::Mutex::new(None),
        });

        let reader_session = Arc::clone(&session);
        let reader_handle = tokio::spawn(async move {
            reader_session.reader_loop(read_half, inbound_tx).await;
        });
        if let Ok(mut slot) = session.reader_abort.lock() {
            *slot = Some(reader_handle.abort_handle());
        }

        let writer_session = Arc::clone(&session);
        let writer_handle = tokio::spawn(async move {
            writer_session.writer_loop(write_half, outbound_rx).await;
        });
        if let Ok(mut slot) = session.writer_abort.lock() {
            *slot = Some(writer_handle.abort_handle());
        }

        session
    }

    async fn reader_loop<R>(
        self: Arc<Self>,
        mut read_half: R,
        inbound_tx: mpsc::Sender<TunnelTcpRequest>,
    ) where
        R: AsyncRead + Unpin,
    {
        loop {
            let frame: TunnelTcpEnvelope = match read_frame_io(&mut read_half).await {
                Ok(frame) => frame,
                Err(ServerError::UnexpectedEof) => {
                    self.mark_closed("tunnel tcp session reached EOF").await;
                    return;
                }
                Err(error) => {
                    warn!(error = %error, "failed to read tunnel tcp frame");
                    self.mark_closed(&format!("failed to read tunnel tcp frame: {error}"))
                        .await;
                    return;
                }
            };

            match frame {
                TunnelTcpEnvelope::Request {
                    request_id,
                    request,
                } => {
                    if inbound_tx
                        .send(TunnelTcpRequest {
                            request_id,
                            request,
                        })
                        .await
                        .is_err()
                    {
                        self.mark_closed("tunnel tcp request channel closed").await;
                        return;
                    }
                }
                TunnelTcpEnvelope::Response {
                    request_id,
                    response,
                } => {
                    if let Some(waiter) = self.pending.lock().await.remove(&request_id) {
                        let _ = waiter.send(response);
                    } else {
                        debug!(
                            request_id,
                            "dropping tunnel tcp response for unknown request"
                        );
                    }
                }
            }
        }
    }

    async fn writer_loop<W>(
        self: Arc<Self>,
        mut write_half: W,
        mut outbound_rx: mpsc::Receiver<TunnelTcpEnvelope>,
    ) where
        W: AsyncWrite + Unpin,
    {
        while let Some(frame) = outbound_rx.recv().await {
            if let Err(error) = write_frame_io(&mut write_half, &frame).await {
                warn!(error = %error, "failed to write tunnel tcp frame");
                self.mark_closed(&format!("failed to write tunnel tcp frame: {error}"))
                    .await;
                return;
            }
        }

        self.mark_closed("tunnel tcp writer channel closed").await;
    }

    async fn mark_closed(&self, reason: &str) {
        let already_closed = *self.closed_tx.borrow();
        if !already_closed {
            let _ = self.closed_tx.send(true);
        }

        let mut pending = self.pending.lock().await;
        for (_, waiter) in pending.drain() {
            let _ = waiter.send(QuarkOut::Error {
                message: reason.to_string(),
            });
        }
    }

    async fn close(&self, reason: &str) {
        if let Ok(mut slot) = self.reader_abort.lock() {
            if let Some(abort) = slot.take() {
                abort.abort();
            }
        }
        if let Ok(mut slot) = self.writer_abort.lock() {
            if let Some(abort) = slot.take() {
                abort.abort();
            }
        }
        self.mark_closed(reason).await;
    }

    async fn call(&self, request: QuarkIn) -> Result<QuarkOut> {
        if *self.closed_tx.borrow() {
            return Err(ServerError::TunnelTcpSessionClosed);
        }
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(request_id, tx);
        if let Err(_send_error) = self
            .outbound
            .send(TunnelTcpEnvelope::Request {
                request_id,
                request,
            })
            .await
        {
            self.pending.lock().await.remove(&request_id);
            return Err(ServerError::TunnelTcpSessionClosed);
        }
        match rx.await {
            Ok(response) => Ok(response),
            Err(_) => {
                self.pending.lock().await.remove(&request_id);
                Err(ServerError::TunnelTcpSessionClosed)
            }
        }
    }

    async fn send_response(&self, request_id: u64, response: QuarkOut) -> Result<()> {
        self.outbound
            .send(TunnelTcpEnvelope::Response {
                request_id,
                response,
            })
            .await
            .map_err(|_| ServerError::TunnelTcpSessionClosed)
    }

    async fn recv_request(&self) -> Option<TunnelTcpRequest> {
        self.inbound.lock().await.recv().await
    }
}

impl QuarkRpcClient {
    async fn connect(addr: SocketAddr, transport_mode: TransportMode) -> Result<Self> {
        let inner = match transport_mode {
            TransportMode::Quic => {
                let crypto = rustls::ClientConfig::builder()
                    .dangerous()
                    .with_custom_certificate_verifier(Arc::new(PermissiveCertVerifier))
                    .with_no_client_auth();
                let quic_crypto =
                    quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
                        .map_err(|e| ServerError::TunnelCrypto(e.to_string()))?;
                let client_config = quinn::ClientConfig::new(Arc::new(quic_crypto));

                let local_addr = client_bind_addr_for(addr);
                let mut endpoint =
                    quinn::Endpoint::client(local_addr).map_err(ServerError::BindTunnelClient)?;
                endpoint.set_default_client_config(client_config);
                QuarkRpcClientInner::Quic {
                    endpoint,
                    remote_addr: addr,
                }
            }
            TransportMode::Tcp => QuarkRpcClientInner::Tcp { remote_addr: addr },
        };
        Ok(Self { inner })
    }

    async fn establish_connection(&self) -> Result<RpcConnection> {
        match &self.inner {
            QuarkRpcClientInner::Quic {
                endpoint,
                remote_addr,
            } => {
                let server_name = remote_addr.ip().to_string();
                let connection = endpoint
                    .connect(*remote_addr, &server_name)
                    .map_err(|e| ServerError::TunnelConnect(e.to_string()))?
                    .await
                    .map_err(|e| ServerError::TunnelConnect(e.to_string()))?;
                Ok(RpcConnection::Quic(connection))
            }
            QuarkRpcClientInner::Tcp { remote_addr } => {
                let stream = TcpStream::connect(*remote_addr)
                    .await
                    .map_err(|e| ServerError::TunnelTcpConnect(e.to_string()))?;
                Ok(RpcConnection::Tcp(stream))
            }
        }
    }
}

async fn request_over_connection(
    connection: &TunnelConnectionHandle,
    req: QuarkIn,
) -> Result<QuarkOut> {
    match connection {
        TunnelConnectionHandle::Quic(connection) => {
            let (mut send, mut recv) = connection
                .open_bi()
                .await
                .map_err(|e| ServerError::TunnelStream(e.to_string()))?;
            write_frame_quic(&mut send, &req).await?;
            read_frame_quic(&mut recv).await
        }
        TunnelConnectionHandle::Tcp(session) => session.call(req).await,
    }
}

/// Write a length-prefixed postcard frame to a QUIC send stream.
async fn write_frame_quic(send: &mut quinn::SendStream, msg: &impl Serialize) -> Result<()> {
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
async fn read_frame_quic<T: for<'de> Deserialize<'de>>(recv: &mut quinn::RecvStream) -> Result<T> {
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

async fn read_frame_io<R, T>(recv: &mut R) -> Result<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
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
        .map_err(ServerError::ReadFramePayloadIo)?;
    from_bytes(&payload).map_err(ServerError::DecodeFrame)
}

async fn write_frame_io<W>(send: &mut W, msg: &impl Serialize) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let payload = to_allocvec(msg).map_err(ServerError::EncodeFrame)?;
    let len =
        u32::try_from(payload.len()).map_err(|_| ServerError::FrameTooLarge(payload.len()))?;
    send.write_all(&len.to_be_bytes())
        .await
        .map_err(ServerError::WriteFrameIo)?;
    send.write_all(&payload)
        .await
        .map_err(ServerError::WriteFrameIo)?;
    Ok(())
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
    optimize_steps: u32,
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
    period_steps: u32,
    train_steps: u32,
    phase_steps: u32,
    warmup_steps: u32,
}

// ---------------------------------------------------------------------------
// Server context — shared across connections
// ---------------------------------------------------------------------------

struct QuarkInstance {
    engine: ModelEngine,
    runtime_config: ModelRuntimeConfig,
    oscillation: Option<FrozenOscillation>,
    checkpoint_path: Option<PathBuf>,
    session: tokio::sync::Mutex<QuarkSession>,
}

#[derive(Debug, Clone, Copy)]
struct ModelRuntimeConfig {
    inference_limit: u32,
    top_k: usize,
    temperature: f64,
    top_p: Option<f64>,
    repeat_penalty: f32,
    presence_penalty: f32,
    training_lr: f64,
    training_epsilon: f64,
    training_z_loss: f64,
    training_lb_loss: f64,
    training_clip_threshold: f64,
    training_error_feedback: QuarkErrorFeedbackConfig,
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

enum QuarkMode {
    Root,
    Worker(Arc<WorkerModeState>),
}

struct WorkerModeState {
    parent_addr: SocketAddr,
    worker_id: Uuid,
    transport_mode: TransportMode,
    tunnel_token: tokio::sync::RwLock<Uuid>,
    parent_session: tokio::sync::RwLock<Option<ParentTunnelSession>>,
    tunnel_connect_deadline: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RouteTarget {
    Local,
    Worker(Uuid),
}

#[derive(Debug, Clone, Copy)]
struct TunnelWorker {
    token: Uuid,
    worker_id: Uuid,
    max_instances: Option<usize>,
}

struct QuarkContext {
    model_path: PathBuf,
    transport_mode: TransportMode,
    void_client: Option<Arc<VoidClient>>,
    defaults: QuarkServerDefaults,
    frozen: bool,
    max_instances: Option<usize>,
    mode: QuarkMode,
    start_dispatch: tokio::sync::Mutex<()>,
    routes: tokio::sync::RwLock<HashMap<Uuid, RouteTarget>>,
    workers: tokio::sync::RwLock<HashMap<Uuid, TunnelWorker>>,
    worker_connections: tokio::sync::RwLock<HashMap<Uuid, TunnelConnectionHandle>>,
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
    training_error_feedback: QuarkErrorFeedbackConfig,
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
            training_error_feedback: QuarkErrorFeedbackConfig::Off,
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
            if let Some(training_z_loss) = model_config.training_z_loss {
                resolved.training_config.z_loss = training_z_loss;
            }
            if let Some(training_lb_loss) = model_config.training_lb_loss {
                resolved.training_config.lb_loss = training_lb_loss;
            }
            if let Some(training_clip_threshold) = model_config.training_clip_threshold {
                resolved.training_config.clip_threshold = training_clip_threshold;
            }
            if let Some(training_error_feedback) = model_config.training_error_feedback {
                resolved.training_error_feedback = training_error_feedback;
            }
        }
        resolved
    }
}

fn to_engine_error_feedback(config: QuarkErrorFeedbackConfig) -> ErrorFeedbackMode {
    match config {
        QuarkErrorFeedbackConfig::Off => ErrorFeedbackMode::None,
        QuarkErrorFeedbackConfig::Persistent { decay, gain } => {
            ErrorFeedbackMode::Persistent(ErrorFeedbackParams { decay, gain })
        }
        QuarkErrorFeedbackConfig::Replay { steps, decay, gain } => {
            ErrorFeedbackMode::Replay(ReplayParams { steps, decay, gain })
        }
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
    transport_mode: TransportMode,
    frozen: bool,
    listen: SocketAddr,
    model_path: PathBuf,
    void_addr: Option<SocketAddr>,
    tunnel: Option<SocketAddr>,
    tunnel_connect_deadline: Option<Duration>,
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
            transport_mode: TransportMode::Quic,
            frozen: false,
            listen: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default listen address must be valid"),
            model_path: model_path.into(),
            void_addr: None,
            tunnel: None,
            tunnel_connect_deadline: None,
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

    pub fn transport_mode(mut self, mode: TransportMode) -> Self {
        self.transport_mode = mode;
        self
    }

    pub fn tcp(mut self) -> Self {
        self.transport_mode = TransportMode::Tcp;
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

    /// Configure how long tunnel workers retry parent registration before failing.
    pub fn tunnel_connect_deadline(mut self, deadline: Duration) -> Self {
        self.tunnel_connect_deadline = Some(deadline);
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
    async fn setup(self) -> Result<(QuarkListener, SocketAddr, Arc<QuarkContext>)> {
        let configured_model_path = resolve_configured_model_path(&self.model_path);
        let model_path_str = configured_model_path.to_string_lossy().to_string();
        info!(model_path = %model_path_str, "configured model");

        // Optionally connect to void.
        let void_client = if let Some(addr) = self.void_addr {
            info!(%addr, "connecting to void");
            let client = VoidClient::connect(addr, self.transport_mode).await?;
            Some(Arc::new(client))
        } else {
            warn!("no void address configured — inference will fail without object store");
            None
        };

        let listener = match self.transport_mode {
            TransportMode::Quic => {
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

                let crypto = QuicServerConfig::try_from(server_config)
                    .map_err(ServerError::QuicServerConfig)?;

                let endpoint_cfg = quinn::ServerConfig::with_crypto(Arc::new(crypto));

                let udp_listener =
                    std::net::UdpSocket::bind(self.listen).map_err(ServerError::BindEndpoint)?;
                let runtime = quinn::TokioRuntime;
                let endpoint = quinn::Endpoint::new(
                    Default::default(),
                    Some(endpoint_cfg),
                    udp_listener,
                    Arc::new(runtime),
                )
                .map_err(ServerError::BindEndpoint)?;
                QuarkListener::Quic(endpoint)
            }
            TransportMode::Tcp => {
                if self.keylog || self.key.is_some() || self.cert.is_some() {
                    warn!(
                        "ignoring TLS options (--keylog/--key/--cert) because TCP transport is enabled"
                    );
                }
                if self.stateless_retry {
                    warn!("ignoring --stateless-retry because TCP transport is enabled");
                }
                let listener = TcpListener::bind(self.listen)
                    .await
                    .map_err(ServerError::BindTcpListener)?;
                QuarkListener::Tcp(listener)
            }
        };

        let local_addr = listener.local_addr().map_err(ServerError::LocalAddr)?;
        info!(%local_addr, "listening");

        let mode = if let Some(parent_addr) = self.tunnel {
            if let Some(deadline) = self.tunnel_connect_deadline {
                info!(
                    %parent_addr,
                    %local_addr,
                    deadline_ms = deadline.as_millis() as u64,
                    "registering tunnel worker"
                );
            } else {
                info!(
                    %parent_addr,
                    %local_addr,
                    deadline = "none",
                    "registering tunnel worker"
                );
            }
            let worker_id = Uuid::new_v4();
            let (tunnel_token, parent_session) = register_tunnel_worker_with_retry(
                parent_addr,
                worker_id,
                resolve_max_instances(self.max_instances),
                self.tunnel_connect_deadline,
                self.transport_mode,
            )
            .await?;
            info!(
                %parent_addr,
                %local_addr,
                %worker_id,
                token = %tunnel_token,
                "tunnel worker registered"
            );
            QuarkMode::Worker(Arc::new(WorkerModeState {
                parent_addr,
                worker_id,
                transport_mode: self.transport_mode,
                tunnel_token: tokio::sync::RwLock::new(tunnel_token),
                parent_session: tokio::sync::RwLock::new(Some(parent_session)),
                tunnel_connect_deadline: self.tunnel_connect_deadline,
            }))
        } else {
            QuarkMode::Root
        };

        let context = Arc::new(QuarkContext {
            model_path: configured_model_path,
            transport_mode: self.transport_mode,
            void_client,
            defaults: self.defaults,
            frozen: self.frozen,
            max_instances: resolve_max_instances(self.max_instances),
            mode,
            start_dispatch: tokio::sync::Mutex::new(()),
            routes: tokio::sync::RwLock::new(HashMap::new()),
            workers: tokio::sync::RwLock::new(HashMap::new()),
            worker_connections: tokio::sync::RwLock::new(HashMap::new()),
            instances: tokio::sync::RwLock::new(HashMap::new()),
        });

        Ok((listener, local_addr, context))
    }

    /// Start the server in a background task. Returns the bound address and
    /// a handle that can be used to await or abort the server.
    pub async fn serve(self) -> Result<(SocketAddr, tokio::task::JoinHandle<Result<()>>)> {
        let stateless_retry = self.stateless_retry;
        let (listener, local_addr, context) = self.setup().await?;
        let handle = tokio::spawn(Self::run_server_loops(listener, context, stateless_retry));
        Ok((local_addr, handle))
    }

    /// Run the server, blocking until the endpoint is closed.
    pub async fn run(self) -> Result<()> {
        let stateless_retry = self.stateless_retry;
        let (listener, _local_addr, context) = self.setup().await?;
        Self::run_server_loops(listener, context, stateless_retry).await
    }

    async fn run_server_loops(
        listener: QuarkListener,
        context: Arc<QuarkContext>,
        stateless_retry: bool,
    ) -> Result<()> {
        let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        if matches!(&context.mode, QuarkMode::Worker(_)) {
            let transport_mode = context.transport_mode;
            match transport_mode {
                TransportMode::Quic => {
                    tokio::select! {
                        accept_result = Self::accept_loop(listener, Arc::clone(&context), stateless_retry, shutdown_rx.clone()) => accept_result,
                        _ = maintain_parent_registration_loop(Arc::clone(&context)) => Ok(()),
                        _ = parent_tunnel_stream_loop(context) => Ok(()),
                    }
                }
                TransportMode::Tcp => {
                    tokio::select! {
                        accept_result = Self::accept_loop(listener, Arc::clone(&context), stateless_retry, shutdown_rx.clone()) => accept_result,
                        _ = maintain_parent_registration_loop(Arc::clone(&context)) => Ok(()),
                        _ = parent_tunnel_tcp_session_loop(context) => Ok(()),
                    }
                }
            }
        } else {
            Self::accept_loop(listener, context, stateless_retry, shutdown_rx).await
        }
    }

    /// Accept-loop shared by both `run()` and `serve()`.
    async fn accept_loop(
        listener: QuarkListener,
        context: Arc<QuarkContext>,
        stateless_retry: bool,
        mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
        match listener {
            QuarkListener::Quic(endpoint) => loop {
                let conn = tokio::select! {
                    incoming = endpoint.accept() => match incoming {
                        Some(c) => c,
                        None => break,
                    },
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        if *shutdown_rx.borrow() {
                            break;
                        }
                        continue;
                    }
                };

                if stateless_retry && !conn.remote_address_validated() {
                    info!("requiring connection to validate its address");
                    let _ = conn.retry();
                    continue;
                }

                info!(remote = %conn.remote_address(), "accepting connection");
                let ctx = Arc::clone(&context);
                tokio::spawn(handle_connection(conn, ctx, shutdown_rx.clone()));
            },
            QuarkListener::Tcp(listener) => loop {
                let accepted = tokio::select! {
                    accepted = listener.accept() => accepted.map_err(ServerError::AcceptTcpConnection)?,
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                        continue;
                    }
                };
                let (stream, remote_addr) = accepted;
                debug!(remote = %remote_addr, "accepting tcp connection");
                let ctx = Arc::clone(&context);
                tokio::spawn(handle_tcp_connection(stream, ctx));
            },
        }

        Ok(())
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        panic!("use ServerBuilder::new(model_path) instead");
    }
}

enum QuarkListener {
    Quic(quinn::Endpoint),
    Tcp(TcpListener),
}

impl QuarkListener {
    fn local_addr(&self) -> io::Result<SocketAddr> {
        match self {
            Self::Quic(endpoint) => endpoint.local_addr(),
            Self::Tcp(listener) => listener.local_addr(),
        }
    }
}

// ---------------------------------------------------------------------------
// Connection / stream handlers
// ---------------------------------------------------------------------------

async fn handle_connection(
    incoming: quinn::Incoming,
    context: Arc<QuarkContext>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    let connection = match incoming.await {
        Ok(c) => c,
        Err(e) => {
            error!("connection failed: {e}");
            return;
        }
    };

    info!("established");

    loop {
        let stream = match tokio::select! {
            stream = connection.accept_bi() => stream,
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    connection.close(0u32.into(), b"server shutdown");
                    return;
                }
                continue;
            }
        } {
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
        tokio::spawn(handle_stream(
            stream,
            ctx,
            Some(TunnelConnectionHandle::Quic(connection.clone())),
        ));
    }
}

async fn parent_tunnel_stream_loop(context: Arc<QuarkContext>) {
    let worker_mode = match &context.mode {
        QuarkMode::Worker(worker_mode) => Arc::clone(worker_mode),
        QuarkMode::Root => return,
    };
    let mut last_connection_id: Option<usize> = None;
    let mut logged_closed_for_connection = false;
    loop {
        let connection = {
            let parent_session = worker_mode.parent_session.read().await;
            parent_session
                .as_ref()
                .map(|session| session.connection.clone())
        };
        let Some(TunnelConnectionHandle::Quic(connection)) = connection else {
            last_connection_id = None;
            logged_closed_for_connection = false;
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        };
        let connection_id = connection.stable_id();
        if last_connection_id != Some(connection_id) {
            last_connection_id = Some(connection_id);
            logged_closed_for_connection = false;
        }

        let stream = match connection.accept_bi().await {
            Ok(stream) => stream,
            Err(quinn::ConnectionError::ApplicationClosed { .. }) => {
                if !logged_closed_for_connection {
                    info!(
                        parent_addr = %worker_mode.parent_addr,
                        worker_id = %worker_mode.worker_id,
                        "parent tunnel connection closed"
                    );
                    logged_closed_for_connection = true;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            Err(error) => {
                if !logged_closed_for_connection {
                    warn!(
                        parent_addr = %worker_mode.parent_addr,
                        worker_id = %worker_mode.worker_id,
                        error = %error,
                        "parent tunnel stream accept failed"
                    );
                    logged_closed_for_connection = true;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        logged_closed_for_connection = false;

        let ctx = Arc::clone(&context);
        tokio::spawn(handle_stream(
            stream,
            ctx,
            Some(TunnelConnectionHandle::Quic(connection.clone())),
        ));
    }
}

async fn parent_tunnel_tcp_session_loop(context: Arc<QuarkContext>) {
    let worker_mode = match &context.mode {
        QuarkMode::Worker(worker_mode) => Arc::clone(worker_mode),
        QuarkMode::Root => return,
    };

    loop {
        let session = {
            let parent_session = worker_mode.parent_session.read().await;
            parent_session
                .as_ref()
                .and_then(|session| match &session.connection {
                    TunnelConnectionHandle::Tcp(session) => Some(Arc::clone(session)),
                    TunnelConnectionHandle::Quic(_) => None,
                })
        };
        let Some(session) = session else {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        };

        let Some(request) = session.recv_request().await else {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        };

        let ctx = Arc::clone(&context);
        tokio::spawn(async move {
            handle_tunnel_tcp_request(request, session, ctx).await;
        });
    }
}

async fn handle_tunnel_tcp_request(
    request: TunnelTcpRequest,
    session: Arc<TcpTunnelSession>,
    context: Arc<QuarkContext>,
) {
    let out = match handle_request(request.request, &context, None).await {
        Ok(out) => out,
        Err(error) => {
            warn!(error = %error, "tunnel tcp request failed");
            QuarkOut::Error {
                message: error.to_string(),
            }
        }
    };
    if let Err(error) = session.send_response(request.request_id, out).await {
        warn!(error = %error, "failed to send tunnel tcp response");
    }
}

async fn set_worker_connection(
    token: Uuid,
    connection: TunnelConnectionHandle,
    ctx: &QuarkContext,
) {
    let old_connection = ctx
        .worker_connections
        .write()
        .await
        .insert(token, connection);
    if let Some(old_connection) = old_connection {
        old_connection
            .close(b"replaced worker tunnel connection")
            .await;
    }
}

async fn drive_worker_tcp_session(
    token: Uuid,
    session: Arc<TcpTunnelSession>,
    context: Arc<QuarkContext>,
) {
    loop {
        let Some(request) = session.recv_request().await else {
            break;
        };
        let session = Arc::clone(&session);
        let ctx = Arc::clone(&context);
        tokio::spawn(async move {
            handle_tunnel_tcp_request(request, session, ctx).await;
        });
    }

    let mut worker_connections = context.worker_connections.write().await;
    let should_remove = worker_connections
        .get(&token)
        .is_some_and(|connection| match connection {
            TunnelConnectionHandle::Tcp(existing) => Arc::ptr_eq(existing, &session),
            TunnelConnectionHandle::Quic(_) => false,
        });
    if should_remove {
        worker_connections.remove(&token);
    }
}

async fn handle_tcp_connection(mut stream: TcpStream, context: Arc<QuarkContext>) {
    loop {
        let req: QuarkIn = match read_frame_io(&mut stream).await {
            Ok(req) => req,
            Err(ServerError::UnexpectedEof) => return,
            Err(error) => {
                warn!(error = %error, "failed to read tcp request frame");
                return;
            }
        };

        match req {
            QuarkIn::RegisterTunnel {
                worker_id,
                max_instances,
            } => {
                let out =
                    match handle_register_tunnel(worker_id, max_instances, None, &context).await {
                        Ok(out) => out,
                        Err(error) => QuarkOut::Error {
                            message: error.to_string(),
                        },
                    };
                if let Err(error) = write_frame_io(&mut stream, &out).await {
                    warn!(error = %error, "failed to write tcp tunnel registration response");
                    return;
                }
                let QuarkOut::TunnelRegistered { token } = out else {
                    return;
                };
                let session = TcpTunnelSession::new(stream);
                set_worker_connection(
                    token,
                    TunnelConnectionHandle::Tcp(Arc::clone(&session)),
                    &context,
                )
                .await;
                tokio::spawn(drive_worker_tcp_session(token, session, context));
                return;
            }
            req => {
                let out = match handle_request(req, &context, None).await {
                    Ok(out) => out,
                    Err(error) => QuarkOut::Error {
                        message: error.to_string(),
                    },
                };
                if let Err(error) = write_frame_io(&mut stream, &out).await {
                    if matches!(
                        error,
                        ServerError::WriteFrameIo(ref io_error)
                            if matches!(
                                io_error.kind(),
                                io::ErrorKind::BrokenPipe
                                    | io::ErrorKind::ConnectionAborted
                                    | io::ErrorKind::ConnectionReset
                            )
                    ) {
                        debug!("tcp client disconnected before response write completed");
                    } else {
                        warn!(error = %error, "failed to write tcp response frame");
                    }
                    return;
                }
            }
        }
    }
}

async fn handle_stream(
    (mut send, mut recv): (quinn::SendStream, quinn::RecvStream),
    context: Arc<QuarkContext>,
    connection: Option<TunnelConnectionHandle>,
) {
    let req: QuarkIn = match read_frame_quic(&mut recv).await {
        Ok(r) => r,
        Err(e) => {
            let _ = write_frame_quic(
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

    let out = match handle_request(req, &context, connection).await {
        Ok(o) => o,
        Err(error) => {
            warn!(error = %error, "request failed");
            QuarkOut::Error {
                message: error.to_string(),
            }
        }
    };

    if write_frame_quic(&mut send, &out).await.is_err() {
        warn!("failed to write response");
    }
}

async fn handle_request(
    req: QuarkIn,
    ctx: &QuarkContext,
    connection: Option<TunnelConnectionHandle>,
) -> Result<QuarkOut> {
    match req {
        QuarkIn::RegisterTunnel {
            worker_id,
            max_instances,
        } => handle_register_tunnel(worker_id, max_instances, connection, ctx).await,
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
        QuarkIn::QueryModelParams { model_id } => {
            handle_query_model_params_routed(model_id, ctx).await
        }
        QuarkIn::QueryModelCapacity => handle_query_model_capacity(ctx).await,
    }
}

fn ensure_root_mode(ctx: &QuarkContext) -> Result<()> {
    match &ctx.mode {
        QuarkMode::Root => Ok(()),
        QuarkMode::Worker(_) => Err(ServerError::TunnelWorkerRejectsModelRequests),
    }
}

async fn register_tunnel_worker(
    parent_addr: SocketAddr,
    worker_id: Uuid,
    max_instances: Option<usize>,
    transport_mode: TransportMode,
) -> Result<(Uuid, ParentTunnelSession)> {
    let client = QuarkRpcClient::connect(parent_addr, transport_mode).await?;
    let connection = client.establish_connection().await?;
    match connection {
        RpcConnection::Quic(connection) => {
            let out = request_over_connection(
                &TunnelConnectionHandle::Quic(connection.clone()),
                QuarkIn::RegisterTunnel {
                    worker_id,
                    max_instances,
                },
            )
            .await?;
            match out {
                QuarkOut::TunnelRegistered { token } => Ok((
                    token,
                    ParentTunnelSession {
                        _client: client,
                        connection: TunnelConnectionHandle::Quic(connection),
                    },
                )),
                QuarkOut::Error { message } => {
                    Err(ServerError::TunnelRegistrationRejected(message))
                }
                _ => Err(ServerError::UnexpectedTunnelResponse(
                    "register tunnel response",
                )),
            }
        }
        RpcConnection::Tcp(mut stream) => {
            write_frame_io(
                &mut stream,
                &QuarkIn::RegisterTunnel {
                    worker_id,
                    max_instances,
                },
            )
            .await?;
            let out: QuarkOut = read_frame_io(&mut stream).await?;
            match out {
                QuarkOut::TunnelRegistered { token } => Ok((
                    token,
                    ParentTunnelSession {
                        _client: client,
                        connection: TunnelConnectionHandle::Tcp(TcpTunnelSession::new(stream)),
                    },
                )),
                QuarkOut::Error { message } => {
                    Err(ServerError::TunnelRegistrationRejected(message))
                }
                _ => Err(ServerError::UnexpectedTunnelResponse(
                    "register tunnel response",
                )),
            }
        }
    }
}

async fn register_tunnel_worker_with_retry(
    parent_addr: SocketAddr,
    worker_id: Uuid,
    max_instances: Option<usize>,
    deadline: Option<Duration>,
    transport_mode: TransportMode,
) -> Result<(Uuid, ParentTunnelSession)> {
    let start = Instant::now();
    let deadline_at = deadline.map(|deadline| start + deadline);
    let mut retry_delay = Duration::from_millis(DEFAULT_TUNNEL_CONNECT_RETRY_MS);
    let max_retry_delay = Duration::from_millis(MAX_TUNNEL_CONNECT_RETRY_MS);
    let mut attempts = 0u32;
    loop {
        attempts = attempts.saturating_add(1);
        match register_tunnel_worker(parent_addr, worker_id, max_instances, transport_mode).await {
            Ok(registered) => return Ok(registered),
            Err(error) => {
                let now = Instant::now();
                if let Some(deadline_at) = deadline_at {
                    if now >= deadline_at {
                        return Err(ServerError::TunnelRegistrationDeadlineExceeded {
                            parent_addr,
                            deadline: deadline.expect("deadline_at implies deadline is set"),
                            attempts,
                            last_error: error.to_string(),
                        });
                    }
                }
                let sleep_for = if let Some(deadline_at) = deadline_at {
                    let remaining = deadline_at.saturating_duration_since(now);
                    let sleep_for = std::cmp::min(retry_delay, remaining);
                    warn!(
                        %parent_addr,
                        attempt = attempts,
                        retry_ms = sleep_for.as_millis() as u64,
                        remaining_ms = remaining.as_millis() as u64,
                        error = %error,
                        "tunnel registration failed; retrying"
                    );
                    sleep_for
                } else {
                    warn!(
                        %parent_addr,
                        attempt = attempts,
                        retry_ms = retry_delay.as_millis() as u64,
                        error = %error,
                        "tunnel registration failed; retrying"
                    );
                    retry_delay
                };
                tokio::time::sleep(sleep_for).await;
                retry_delay = retry_delay.saturating_mul(2).min(max_retry_delay);
            }
        }
    }
}

async fn update_tunnel_capacity(
    parent_connection: &TunnelConnectionHandle,
    tunnel_token: Uuid,
    max_instances: Option<usize>,
) -> Result<()> {
    let out = request_over_connection(
        parent_connection,
        QuarkIn::UpdateTunnelCapacity {
            token: tunnel_token,
            max_instances,
        },
    )
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
    match &ctx.mode {
        QuarkMode::Root => Ok(()),
        QuarkMode::Worker(worker_mode) => {
            let max_instances = advertised_capacity(ctx).await;
            let tunnel_token = *worker_mode.tunnel_token.read().await;
            let parent_connection = {
                let parent_session = worker_mode.parent_session.read().await;
                parent_session
                    .as_ref()
                    .map(|session| session.connection.clone())
                    .ok_or_else(|| {
                        ServerError::TunnelConnect("tunnel parent session unavailable".into())
                    })?
            };
            match update_tunnel_capacity(&parent_connection, tunnel_token, max_instances).await {
                Ok(()) => Ok(()),
                Err(error) => {
                    warn!(
                        parent_addr = %worker_mode.parent_addr,
                        worker_id = %worker_mode.worker_id,
                        token = %tunnel_token,
                        error = %error,
                        "tunnel capacity update failed; attempting worker re-registration"
                    );
                    let (new_token, new_session) = register_tunnel_worker_with_retry(
                        worker_mode.parent_addr,
                        worker_mode.worker_id,
                        max_instances,
                        worker_mode.tunnel_connect_deadline,
                        worker_mode.transport_mode,
                    )
                    .await?;
                    {
                        let mut parent_session = worker_mode.parent_session.write().await;
                        if let Some(old_session) = parent_session.replace(new_session) {
                            old_session
                                .connection
                                .close(b"replaced parent tunnel session")
                                .await;
                        }
                    }
                    {
                        let mut tunnel_token = worker_mode.tunnel_token.write().await;
                        *tunnel_token = new_token;
                    }
                    info!(
                        parent_addr = %worker_mode.parent_addr,
                        worker_id = %worker_mode.worker_id,
                        token = %new_token,
                        "tunnel worker re-registered after parent reconnect"
                    );
                    Ok(())
                }
            }
        }
    }
}

async fn maintain_parent_registration_loop(context: Arc<QuarkContext>) {
    loop {
        tokio::time::sleep(TUNNEL_REGISTRATION_REFRESH_INTERVAL).await;
        if let Err(error) = propagate_capacity_to_parent(context.as_ref()).await {
            warn!(
                error = %error,
                "failed to refresh tunnel worker registration with parent"
            );
        }
    }
}

async fn handle_register_tunnel(
    worker_id: Uuid,
    max_instances: Option<usize>,
    connection: Option<TunnelConnectionHandle>,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    let max_instances = resolve_max_instances(max_instances);

    let token = {
        let mut workers = ctx.workers.write().await;
        if let Some((token, worker)) = workers
            .iter_mut()
            .find(|(_, worker)| worker.worker_id == worker_id)
        {
            worker.max_instances = max_instances;
            *token
        } else {
            let token = Uuid::new_v4();
            workers.insert(
                token,
                TunnelWorker {
                    token,
                    worker_id,
                    max_instances,
                },
            );
            token
        }
    };

    if let Some(connection) = connection {
        set_worker_connection(token, connection, ctx).await;
    }

    propagate_capacity_to_parent(ctx).await?;

    info!(%worker_id, ?max_instances, token = %token, "registered tunnel worker");
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

    debug!(token = %token, ?max_instances, "updated tunnel worker capacity");
    Ok(QuarkOut::Ack)
}

async fn handle_tunnel_forward(
    token: Uuid,
    request: TunnelRequest,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match &ctx.mode {
        QuarkMode::Worker(worker_mode) => {
            let tunnel_token = *worker_mode.tunnel_token.read().await;
            if tunnel_token == token {
                handle_tunnel_request_local(request, ctx).await
            } else {
                Err(ServerError::TunnelUnauthorizedForward)
            }
        }
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
        TunnelRequest::QueryModelParams { model_id } => {
            handle_query_model_params_distributed(model_id, ctx).await
        }
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
    let local_count = ctx.instances.read().await.len();
    let mut worker_counts: HashMap<Uuid, usize> = HashMap::new();
    for target in routes.values() {
        match target {
            RouteTarget::Local => {}
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

async fn get_worker_connection(token: Uuid, ctx: &QuarkContext) -> Result<TunnelConnectionHandle> {
    ctx.worker_connections
        .read()
        .await
        .get(&token)
        .cloned()
        .ok_or(ServerError::TunnelWorkerUnavailable(token))
}

async fn forward_tunnel_request(
    worker_token: Uuid,
    request: TunnelRequest,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    let _worker = get_worker(worker_token, ctx).await?;
    let connection = get_worker_connection(worker_token, ctx).await?;
    let out = request_over_connection(
        &connection,
        QuarkIn::TunnelForward {
            token: worker_token,
            request,
        },
    )
    .await
    .map_err(|error| ServerError::TunnelWorkerError(error.to_string()))?;
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
    // Keep distributed starts serialized so routed and local model initialization
    // cannot overlap on the same quark context.
    let _start_dispatch_guard = ctx.start_dispatch.lock().await;

    if ctx.routes.read().await.contains_key(&model_id)
        || ctx.instances.read().await.contains_key(&model_id)
    {
        return Err(ServerError::ModelInstanceAlreadyRunning(model_id));
    }

    let target = select_start_target(ctx).await?;
    let out = match target {
        RouteTarget::Local => handle_start(model_id, model_config, ctx).await?,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(
                token,
                TunnelRequest::Start {
                    model_id,
                    model_config,
                },
                ctx,
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
            forward_tunnel_request(token, TunnelRequest::PerturbUp { model_id, seed }, ctx).await
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
            forward_tunnel_request(token, TunnelRequest::Infer { model_id, input_id }, ctx).await
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
            forward_tunnel_request(token, TunnelRequest::Reset { model_id }, ctx).await
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
            forward_tunnel_request(token, TunnelRequest::PerturbDown { model_id }, ctx).await
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
            forward_tunnel_request(token, TunnelRequest::Checkpoint { model_id }, ctx).await
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
            forward_tunnel_request(
                token,
                TunnelRequest::Optimize {
                    model_id,
                    loss_up,
                    loss_down,
                },
                ctx,
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
            forward_tunnel_request(token, TunnelRequest::Shutdown { model_id }, ctx).await?
        }
    };
    if matches!(out, QuarkOut::Ack) {
        ctx.routes.write().await.remove(&model_id);
    }
    Ok(out)
}

async fn handle_query_model_params_routed(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    ensure_root_mode(ctx)?;
    handle_query_model_params_distributed(model_id, ctx).await
}

async fn handle_query_model_params_distributed(
    model_id: Uuid,
    ctx: &QuarkContext,
) -> Result<QuarkOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_query_model_params(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::QueryModelParams { model_id }, ctx).await
        }
    }
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
    let runtime_config = ModelRuntimeConfig {
        inference_limit: defaults.inference_limit,
        top_k: defaults.top_k,
        temperature: defaults.temperature,
        top_p: defaults.top_p,
        repeat_penalty: defaults.repeat_penalty,
        presence_penalty: defaults.presence_penalty,
        training_lr: defaults.training_config.lr,
        training_epsilon: defaults.training_config.epsilon,
        training_z_loss: defaults.training_config.z_loss,
        training_lb_loss: defaults.training_config.lb_loss,
        training_clip_threshold: defaults.training_config.clip_threshold,
        training_error_feedback: defaults.training_error_feedback,
    };
    let error_feedback = defaults.training_error_feedback;
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
    info!(
        %model_id,
        inference_limit = runtime_config.inference_limit,
        top_k = runtime_config.top_k,
        temperature = runtime_config.temperature,
        top_p = ?runtime_config.top_p,
        repeat_penalty = runtime_config.repeat_penalty,
        presence_penalty = runtime_config.presence_penalty,
        training_lr = runtime_config.training_lr,
        training_epsilon = runtime_config.training_epsilon,
        training_z_loss = runtime_config.training_z_loss,
        training_lb_loss = runtime_config.training_lb_loss,
        training_clip_threshold = runtime_config.training_clip_threshold,
        training_error_feedback = ?runtime_config.training_error_feedback,
        frozen,
        has_tokenizer_override = tokenizer_path.is_some(),
        has_checkpoint = checkpoint_path.is_some(),
        oscillation_enabled = oscillation.is_some(),
        "initializing model instance"
    );
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

    if let Err(error) = engine
        .set_hyper_parameters(HyperParameterUpdate {
            error_feedback: Some(to_engine_error_feedback(error_feedback)),
            ..HyperParameterUpdate::default()
        })
        .await
    {
        warn!(
            %model_id,
            training_error_feedback = ?runtime_config.training_error_feedback,
            error = %error,
            "failed to set initial hyper parameters"
        );
        ctx.instances.write().await.remove(&model_id);
        if let Some(path) = checkpoint_path.as_ref() {
            cleanup_checkpoint_file(path);
        }
        return Err(ServerError::ModelError(format!(
            "failed to set initial hyper parameters: {error}"
        )));
    }

    let mut session = QuarkSession::new(frozen);
    apply_initial_frozen_oscillation(model_id, &mut session, oscillation);
    let instance = Arc::new(QuarkInstance {
        engine,
        runtime_config,
        oscillation,
        checkpoint_path,
        session: tokio::sync::Mutex::new(session),
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

fn build_model_params(
    runtime_config: ModelRuntimeConfig,
    oscillation: Option<FrozenOscillation>,
    session: &QuarkSession,
) -> QuarkModelParams {
    QuarkModelParams {
        inference_limit: runtime_config.inference_limit,
        top_k: runtime_config.top_k,
        temperature: runtime_config.temperature,
        top_p: runtime_config.top_p,
        repeat_penalty: runtime_config.repeat_penalty,
        presence_penalty: runtime_config.presence_penalty,
        training_lr: runtime_config.training_lr,
        training_epsilon: runtime_config.training_epsilon,
        training_z_loss: runtime_config.training_z_loss,
        training_lb_loss: runtime_config.training_lb_loss,
        training_clip_threshold: runtime_config.training_clip_threshold,
        training_error_feedback: runtime_config.training_error_feedback,
        is_frozen: session.frozen,
        optimize_steps: session.optimize_steps,
        oscillation_period_steps: oscillation.map(|osc| osc.period_steps),
        oscillation_train_steps: oscillation.map(|osc| osc.train_steps),
        oscillation_phase_steps: oscillation.map(|osc| osc.phase_steps),
        oscillation_warmup_steps: oscillation.map(|osc| osc.warmup_steps),
    }
}

async fn handle_query_model_params(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    Ok(QuarkOut::ModelParams {
        params: build_model_params(instance.runtime_config, instance.oscillation, &session),
    })
}

async fn handle_query_model_capacity(ctx: &QuarkContext) -> Result<QuarkOut> {
    let occupied = ctx.routes.read().await.len();
    let total = advertised_capacity(ctx).await;
    let available = total.map(|total| total.saturating_sub(occupied));
    Ok(QuarkOut::ModelCapacity {
        capacity: QuarkModelCapacity {
            total,
            available,
            occupied,
        },
    })
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
    let Some(period_steps) = config.oscillation_period_steps else {
        return Ok(None);
    };
    if period_steps == 0 {
        return Err(ServerError::InvalidOscillationPeriodSteps(period_steps));
    }
    let Some(train_steps) = config.oscillation_train_steps else {
        return Err(ServerError::MissingOscillationTrainSteps);
    };
    if train_steps > period_steps {
        return Err(ServerError::InvalidOscillationTrainSteps {
            train_steps,
            period_steps,
        });
    }
    Ok(Some(FrozenOscillation {
        period_steps,
        train_steps,
        phase_steps: config.oscillation_phase_steps.unwrap_or_default(),
        warmup_steps: config.oscillation_warmup_steps.unwrap_or_default(),
    }))
}

fn frozen_state_for_optimize_step(
    optimize_steps: u32,
    oscillation: FrozenOscillation,
) -> Option<bool> {
    if optimize_steps <= oscillation.warmup_steps {
        return None;
    }
    let relative_step = optimize_steps - oscillation.warmup_steps - 1;
    let cycle_position = (relative_step + oscillation.phase_steps % oscillation.period_steps)
        % oscillation.period_steps;
    let should_train = cycle_position < oscillation.train_steps;
    Some(!should_train)
}

fn apply_initial_frozen_oscillation(
    model_id: Uuid,
    session: &mut QuarkSession,
    oscillation: Option<FrozenOscillation>,
) {
    let Some(oscillation) = oscillation else {
        return;
    };
    let Some(frozen) = frozen_state_for_optimize_step(1, oscillation) else {
        return;
    };
    session.frozen = frozen;
    info!(
        %model_id,
        frozen = session.frozen,
        optimize_steps = session.optimize_steps,
        oscillation_period_steps = oscillation.period_steps,
        oscillation_train_steps = oscillation.train_steps,
        oscillation_phase_steps = oscillation.phase_steps,
        oscillation_warmup_steps = oscillation.warmup_steps,
        "initialized model frozen state from oscillation schedule"
    );
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
    let Some(frozen) = frozen_state_for_optimize_step(session.optimize_steps, oscillation) else {
        return;
    };
    session.frozen = frozen;
    debug!(
        %model_id,
        frozen = session.frozen,
        optimize_steps = session.optimize_steps,
        oscillation_period_steps = oscillation.period_steps,
        oscillation_train_steps = oscillation.train_steps,
        oscillation_phase_steps = oscillation.phase_steps,
        oscillation_warmup_steps = oscillation.warmup_steps,
        "applied model frozen state from oscillation schedule"
    );
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
    let void = require_void_client(ctx, "checkpoint restore")?;
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

fn require_void_client<'a>(
    ctx: &'a QuarkContext,
    operation: &'static str,
) -> Result<&'a Arc<VoidClient>> {
    ctx.void_client
        .as_ref()
        .ok_or_else(|| void_not_configured_error(ctx, operation))
}

fn void_not_configured_error(ctx: &QuarkContext, operation: &'static str) -> ServerError {
    match &ctx.mode {
        QuarkMode::Root => ServerError::VoidNotConfigured,
        QuarkMode::Worker(_) => ServerError::TunnelWorkerVoidNotConfigured(operation),
    }
}

async fn reset_model(engine: &ModelEngine) -> Result<()> {
    engine
        .reset_state()
        .await
        .map_err(|error| ServerError::ModelError(error.to_string()))
}

fn unsupported_residual_update_dtype(error_message: &str) -> Option<String> {
    let (_, suffix) = error_message.split_once(RESIDUAL_UPDATE_UNSUPPORTED_FRAGMENT)?;
    let token = suffix.split_whitespace().next()?;
    let dtype = token.trim_end_matches(['.', ',', ';', ':']);
    if dtype.is_empty() {
        None
    } else {
        Some(dtype.to_string())
    }
}

fn error_feedback_mode_name(config: QuarkErrorFeedbackConfig) -> Option<&'static str> {
    match config {
        QuarkErrorFeedbackConfig::Off => None,
        QuarkErrorFeedbackConfig::Persistent { .. } => Some("persistent"),
        QuarkErrorFeedbackConfig::Replay { .. } => Some("replay"),
    }
}

fn error_feedback_support_hint(
    training_error_feedback: QuarkErrorFeedbackConfig,
    unsupported_dtype: Option<&str>,
) -> Option<String> {
    let mode = error_feedback_mode_name(training_error_feedback)?;
    let dtype = unsupported_dtype.unwrap_or("this");
    Some(format!(
        "paramecia does not support {mode} error-feedback updates for {dtype} weights. Set training_error_feedback=Off or use a quantized checkpoint."
    ))
}

fn optimization_model_error(error_message: &str, hint: Option<&str>) -> ServerError {
    let mut message = format!("optimization failed: {error_message}");
    if let Some(hint) = hint {
        message.push_str(". ");
        message.push_str(hint);
    }
    ServerError::ModelError(message)
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
    debug!(%model_id, "received perturb up request");
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
        debug!(%model_id, "skipping perturb up because model instance is frozen");
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
    debug!(%model_id, "received reset request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    reset_model(&instance.engine).await?;
    Ok(QuarkOut::Ack)
}

async fn handle_infer(model_id: Uuid, input_id: ObjectId, ctx: &QuarkContext) -> Result<QuarkOut> {
    debug!(%model_id, "received inference request");
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
    let void = require_void_client(ctx, "inference")?;

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
    let limit = limit.unwrap_or(instance.runtime_config.inference_limit);

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

    debug!(%model_id, "finished processing inference request");
    Ok(QuarkOut::Inferred { output_id })
}

async fn handle_perturb_down(model_id: Uuid, ctx: &QuarkContext) -> Result<QuarkOut> {
    debug!(%model_id, "received perturb down request");
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
        debug!(%model_id, "skipping perturb down because model instance is frozen");
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
    debug!(%model_id, "received checkpoint request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;

    let void = require_void_client(ctx, "checkpoint upload")?;
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
    debug!(%model_id, "received optimization request");
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
        debug!(%model_id, "skipping optimization because model instance is frozen");
    } else {
        let runtime_config = instance.runtime_config;
        instance
            .engine
            .update(loss_up, loss_down)
            .await
            .map_err(|error| {
                let error_message = error.to_string();
                let unsupported_dtype = unsupported_residual_update_dtype(&error_message);
                let error_feedback_hint = error_feedback_support_hint(
                    runtime_config.training_error_feedback,
                    unsupported_dtype.as_deref(),
                );
                warn!(
                    %model_id,
                    loss_up,
                    loss_down,
                    training_lr = runtime_config.training_lr,
                    training_epsilon = runtime_config.training_epsilon,
                    training_z_loss = runtime_config.training_z_loss,
                    training_lb_loss = runtime_config.training_lb_loss,
                    training_clip_threshold = runtime_config.training_clip_threshold,
                    training_error_feedback = ?runtime_config.training_error_feedback,
                    unsupported_error_feedback_dtype = ?unsupported_dtype,
                    error_feedback_hint = ?error_feedback_hint,
                    error = %error_message,
                    "optimization failed"
                );
                optimization_model_error(&error_message, error_feedback_hint.as_deref())
            })?;
    }

    session.state = QuarkState::Idle;
    apply_frozen_oscillation(model_id, &mut session, instance.oscillation);
    debug!(%model_id, "finished optimization update");
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
    #[error("failed to bind TCP listener: {0}")]
    BindTcpListener(#[source] io::Error),
    #[error("failed to accept TCP connection: {0}")]
    AcceptTcpConnection(#[source] io::Error),
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
    #[error("oscillation_period_steps must be greater than zero, got {0}")]
    InvalidOscillationPeriodSteps(u32),
    #[error("oscillation_train_steps must be provided when oscillation_period_steps is set")]
    MissingOscillationTrainSteps,
    #[error(
        "oscillation_train_steps must be <= oscillation_period_steps, got {train_steps} > {period_steps}"
    )]
    InvalidOscillationTrainSteps { train_steps: u32, period_steps: u32 },
    #[error("void service not configured")]
    VoidNotConfigured,
    #[error(
        "void service not configured on tunnel worker (required for {0}); set --void-addr to the same void service as the root quark"
    )]
    TunnelWorkerVoidNotConfigured(&'static str),
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
    #[error("failed to connect to void over tcp: {0}")]
    VoidTcpConnect(String),
    #[error("void crypto config error: {0}")]
    VoidCrypto(String),
    #[error("void error: {0}")]
    VoidError(String),
    #[error("failed to bind tunnel client endpoint: {0}")]
    BindTunnelClient(#[source] io::Error),
    #[error("failed to connect to tunnel peer: {0}")]
    TunnelConnect(String),
    #[error("failed to connect to tunnel peer over tcp: {0}")]
    TunnelTcpConnect(String),
    #[error("failed to open tunnel stream: {0}")]
    TunnelStream(String),
    #[error("tunnel crypto config error: {0}")]
    TunnelCrypto(String),
    #[error("tunnel registration rejected: {0}")]
    TunnelRegistrationRejected(String),
    #[error(
        "failed to register tunnel worker with parent {parent_addr} within {:?} after {attempts} attempts: {last_error}",
        .deadline
    )]
    TunnelRegistrationDeadlineExceeded {
        parent_addr: SocketAddr,
        deadline: Duration,
        attempts: u32,
        last_error: String,
    },
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
    #[error("tunnel tcp session closed")]
    TunnelTcpSessionClosed,
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
    #[error("failed to read frame payload: {0}")]
    ReadFramePayloadIo(#[source] io::Error),
    #[error("failed to decode frame: {0}")]
    DecodeFrame(postcard::Error),
    #[error("failed to encode frame: {0}")]
    EncodeFrame(postcard::Error),
    #[error("failed to write frame: {0}")]
    WriteFrame(quinn::WriteError),
    #[error("failed to write frame: {0}")]
    WriteFrameIo(io::Error),
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{
        apply_frozen_oscillation, apply_initial_frozen_oscillation, build_model_params,
        client_bind_addr_for, handle_query_model_capacity, handle_register_tunnel,
        repair_duplicated_absolute_model_path, resolve_max_instances, resolve_model_frozen,
        resolve_model_oscillation, select_start_target, to_engine_error_feedback,
        FrozenOscillation, ModelRuntimeConfig, ModelSlot, QuarkContext, QuarkMode,
        QuarkServerDefaults, QuarkSession, QuarkState, RouteTarget, ServerBuilder, TransportMode,
        TunnelWorker, DEFAULT_INFERENCE_LIMIT, DEFAULT_MAX_INSTANCES,
    };
    use black_hole_spec::{QuarkErrorFeedbackConfig, QuarkModelCapacity, QuarkModelConfig};
    use std::{collections::HashMap, fs, net::SocketAddr, path::PathBuf};
    use tokio::sync::{Mutex, RwLock};

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
        assert_eq!(
            resolved.training_error_feedback,
            defaults.training_error_feedback
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
            training_z_loss: Some(0.12),
            training_lb_loss: Some(0.24),
            training_clip_threshold: Some(0.5),
            training_error_feedback: Some(QuarkErrorFeedbackConfig::Persistent {
                decay: 0.8,
                gain: 0.6,
            }),
            frozen: None,
            oscillation_period_steps: None,
            oscillation_train_steps: None,
            oscillation_phase_steps: None,
            oscillation_warmup_steps: None,
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
        assert_eq!(resolved.training_config.z_loss, 0.12);
        assert_eq!(resolved.training_config.lb_loss, 0.24);
        assert_eq!(resolved.training_config.clip_threshold, 0.5);
        assert_eq!(
            resolved.training_error_feedback,
            QuarkErrorFeedbackConfig::Persistent {
                decay: 0.8,
                gain: 0.6,
            }
        );
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
                oscillation_warmup_steps: Some(7),
                ..QuarkModelConfig::default()
            }))
            .unwrap(),
            None
        );
    }

    #[test]
    fn model_config_oscillation_resolves_period_train_phase_and_warmup() {
        let resolved = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_period_steps: Some(10),
            oscillation_train_steps: Some(3),
            oscillation_phase_steps: Some(4),
            oscillation_warmup_steps: Some(20),
            ..QuarkModelConfig::default()
        }))
        .expect("oscillation config should resolve")
        .expect("oscillation config should be present");
        assert_eq!(
            resolved,
            FrozenOscillation {
                period_steps: 10,
                train_steps: 3,
                phase_steps: 4,
                warmup_steps: 20
            }
        );
    }

    #[test]
    fn model_config_oscillation_rejects_zero_period() {
        let err = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_period_steps: Some(0),
            oscillation_train_steps: Some(1),
            ..QuarkModelConfig::default()
        }))
        .expect_err("zero oscillation period should be rejected");
        assert!(matches!(
            err,
            super::ServerError::InvalidOscillationPeriodSteps(0)
        ));
    }

    #[test]
    fn model_config_oscillation_requires_train_steps_when_period_is_set() {
        let err = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_period_steps: Some(8),
            ..QuarkModelConfig::default()
        }))
        .expect_err("missing train steps should be rejected");
        assert!(matches!(
            err,
            super::ServerError::MissingOscillationTrainSteps
        ));
    }

    #[test]
    fn model_config_oscillation_rejects_train_steps_greater_than_period() {
        let err = resolve_model_oscillation(Some(&QuarkModelConfig {
            oscillation_period_steps: Some(8),
            oscillation_train_steps: Some(9),
            ..QuarkModelConfig::default()
        }))
        .expect_err("train steps above period should be rejected");
        assert!(matches!(
            err,
            super::ServerError::InvalidOscillationTrainSteps {
                train_steps: 9,
                period_steps: 8
            }
        ));
    }

    #[test]
    fn oscillation_applies_windowed_schedule_after_warmup() {
        let mut session = QuarkSession {
            state: QuarkState::AwaitingOptimize,
            running: true,
            frozen: true,
            optimize_steps: 0,
        };
        let oscillation = Some(FrozenOscillation {
            period_steps: 5,
            train_steps: 2,
            phase_steps: 3,
            warmup_steps: 2,
        });
        let model_id = uuid::Uuid::new_v4();
        let mut trainable_steps = Vec::new();
        for step in 1..=12 {
            apply_frozen_oscillation(model_id, &mut session, oscillation);
            if !session.frozen {
                trainable_steps.push(step);
            }
        }
        assert_eq!(trainable_steps, vec![5, 6, 10, 11]);
        assert!(session.frozen);
    }

    #[test]
    fn model_params_report_effective_frozen_state_after_oscillation() {
        let runtime_config = ModelRuntimeConfig {
            inference_limit: 32,
            top_k: 123,
            temperature: 0.4,
            top_p: Some(0.95),
            repeat_penalty: 1.1,
            presence_penalty: 0.2,
            training_lr: 0.0007,
            training_epsilon: 0.00001,
            training_z_loss: 0.005,
            training_lb_loss: 0.015,
            training_clip_threshold: 1.25,
            training_error_feedback: QuarkErrorFeedbackConfig::Replay {
                steps: 64,
                decay: 0.88,
                gain: 0.42,
            },
        };
        let oscillation = Some(FrozenOscillation {
            period_steps: 4,
            train_steps: 2,
            phase_steps: 0,
            warmup_steps: 0,
        });
        let mut session = QuarkSession {
            state: QuarkState::AwaitingOptimize,
            running: true,
            frozen: true,
            optimize_steps: 0,
        };
        let model_id = uuid::Uuid::new_v4();

        apply_frozen_oscillation(model_id, &mut session, oscillation);
        let first = build_model_params(runtime_config, oscillation, &session);
        assert_eq!(first.optimize_steps, 1);
        assert_eq!(first.is_frozen, false);
        assert_eq!(first.oscillation_period_steps, Some(4));
        assert_eq!(first.oscillation_train_steps, Some(2));
        assert_eq!(first.training_lr, 0.0007);
        assert_eq!(first.training_epsilon, 0.00001);
        assert_eq!(first.training_z_loss, 0.005);
        assert_eq!(first.training_lb_loss, 0.015);
        assert_eq!(first.training_clip_threshold, 1.25);
        assert_eq!(
            first.training_error_feedback,
            QuarkErrorFeedbackConfig::Replay {
                steps: 64,
                decay: 0.88,
                gain: 0.42,
            }
        );
        assert_eq!(first.top_k, 123);
        assert_eq!(first.inference_limit, 32);

        apply_frozen_oscillation(model_id, &mut session, oscillation);
        apply_frozen_oscillation(model_id, &mut session, oscillation);
        let third = build_model_params(runtime_config, oscillation, &session);
        assert_eq!(third.optimize_steps, 3);
        assert_eq!(third.is_frozen, true);
    }

    #[test]
    fn quark_error_feedback_maps_to_engine_modes() {
        assert!(matches!(
            to_engine_error_feedback(QuarkErrorFeedbackConfig::Off),
            paramecia_engine::ErrorFeedbackMode::None
        ));
        assert!(matches!(
            to_engine_error_feedback(QuarkErrorFeedbackConfig::Persistent {
                decay: 0.9,
                gain: 1.0
            }),
            paramecia_engine::ErrorFeedbackMode::Persistent(_)
        ));
        assert!(matches!(
            to_engine_error_feedback(QuarkErrorFeedbackConfig::Replay {
                steps: 8,
                decay: 0.7,
                gain: 0.5
            }),
            paramecia_engine::ErrorFeedbackMode::Replay(_)
        ));
    }

    #[test]
    fn oscillation_phase_can_initialize_frozen_state_before_first_optimize() {
        let mut up = QuarkSession {
            state: QuarkState::Idle,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let mut down = QuarkSession {
            state: QuarkState::Idle,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let up_oscillation = Some(FrozenOscillation {
            period_steps: 2,
            train_steps: 1,
            phase_steps: 0,
            warmup_steps: 0,
        });
        let down_oscillation = Some(FrozenOscillation {
            period_steps: 2,
            train_steps: 1,
            phase_steps: 1,
            warmup_steps: 0,
        });
        let model_id = uuid::Uuid::new_v4();

        apply_initial_frozen_oscillation(model_id, &mut up, up_oscillation);
        apply_initial_frozen_oscillation(model_id, &mut down, down_oscillation);

        assert!(!up.frozen);
        assert!(down.frozen);
        assert_eq!(up.optimize_steps, 0);
        assert_eq!(down.optimize_steps, 0);
    }

    #[test]
    fn half_up_and_half_down_oscillations_report_opposite_runtime_frozen_states() {
        let mut half_up = QuarkSession {
            state: QuarkState::AwaitingOptimize,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let mut half_down = QuarkSession {
            state: QuarkState::AwaitingOptimize,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let half_up_oscillation = Some(FrozenOscillation {
            period_steps: 2,
            train_steps: 1,
            phase_steps: 0,
            warmup_steps: 0,
        });
        let half_down_oscillation = Some(FrozenOscillation {
            period_steps: 2,
            train_steps: 1,
            phase_steps: 1,
            warmup_steps: 0,
        });
        let model_id = uuid::Uuid::new_v4();

        apply_initial_frozen_oscillation(model_id, &mut half_up, half_up_oscillation);
        apply_initial_frozen_oscillation(model_id, &mut half_down, half_down_oscillation);

        let mut half_up_states = vec![half_up.frozen];
        let mut half_down_states = vec![half_down.frozen];
        for _ in 0..4 {
            apply_frozen_oscillation(model_id, &mut half_up, half_up_oscillation);
            apply_frozen_oscillation(model_id, &mut half_down, half_down_oscillation);
            half_up_states.push(half_up.frozen);
            half_down_states.push(half_down.frozen);
        }

        assert_eq!(half_up_states, vec![false, false, true, false, true]);
        assert_eq!(half_down_states, vec![true, true, false, true, false]);
    }

    #[test]
    fn optimization_error_includes_engine_message() {
        let err = super::optimization_model_error("engine says clip threshold is invalid", None);
        let super::ServerError::ModelError(message) = err else {
            panic!("expected model error");
        };
        assert_eq!(
            message,
            "optimization failed: engine says clip threshold is invalid"
        );
    }

    #[test]
    fn optimization_error_adds_actionable_error_feedback_hint() {
        let engine_error =
            "Train error: update failed: restore_and_update_with_residual not supported for F32";
        let unsupported_dtype = super::unsupported_residual_update_dtype(engine_error);
        assert_eq!(unsupported_dtype.as_deref(), Some("F32"));

        let hint = super::error_feedback_support_hint(
            QuarkErrorFeedbackConfig::Persistent {
                decay: 0.9,
                gain: 1.0,
            },
            unsupported_dtype.as_deref(),
        )
        .expect("persistent mode should emit support hint");
        assert!(hint.contains("persistent"));
        assert!(hint.contains("F32"));

        let err = super::optimization_model_error(engine_error, Some(&hint));
        let super::ServerError::ModelError(message) = err else {
            panic!("expected model error");
        };
        assert!(message.contains(engine_error));
        assert!(message.contains("training_error_feedback=Off"));
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

    #[test]
    fn server_builder_defaults_tunnel_connect_deadline_to_none() {
        let builder = ServerBuilder::new("model-is-not-loaded-for-this-test");
        assert_eq!(builder.tunnel_connect_deadline, None);
    }

    #[test]
    fn repair_duplicated_absolute_model_path_recovers_existing_original_path() {
        let root = std::env::temp_dir().join(format!(
            "black-hole-quark-path-repair-{}",
            uuid::Uuid::new_v4()
        ));
        let cwd = root.join("sandbox/current");
        fs::create_dir_all(&cwd).expect("failed to create cwd directory");
        let original = root.join("weights/model.gguf");
        fs::create_dir_all(
            original
                .parent()
                .expect("model path should include a parent directory"),
        )
        .expect("failed to create model parent directory");
        fs::write(&original, b"gguf").expect("failed to create model file");

        let cwd_root = cwd
            .ancestors()
            .last()
            .expect("cwd should have a filesystem root");
        let absolute_suffix = original
            .strip_prefix(cwd_root)
            .expect("original path should share root with cwd");
        let duplicated = cwd.join(absolute_suffix);

        let repaired = repair_duplicated_absolute_model_path(&duplicated, &cwd)
            .expect("duplicated absolute path should be repaired");
        assert_eq!(repaired, original);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn repair_duplicated_absolute_model_path_returns_none_when_repaired_path_missing() {
        let root = std::env::temp_dir().join(format!(
            "black-hole-quark-path-repair-missing-{}",
            uuid::Uuid::new_v4()
        ));
        let cwd = root.join("sandbox/current");
        fs::create_dir_all(&cwd).expect("failed to create cwd directory");
        let expected = root.join("weights/missing-model.gguf");

        let cwd_root = cwd
            .ancestors()
            .last()
            .expect("cwd should have a filesystem root");
        let absolute_suffix = expected
            .strip_prefix(cwd_root)
            .expect("expected path should share root with cwd");
        let duplicated = cwd.join(absolute_suffix);

        let repaired = repair_duplicated_absolute_model_path(&duplicated, &cwd);
        assert!(
            repaired.is_none(),
            "repair should not rewrite to a non-existent model path"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn client_bind_addr_matches_ipv4_remote_family() {
        let remote: SocketAddr = "127.0.0.1:4433".parse().expect("valid socket address");
        let local = client_bind_addr_for(remote);
        assert!(local.is_ipv4());
    }

    #[test]
    fn client_bind_addr_matches_ipv6_remote_family() {
        let remote: SocketAddr = "[::1]:4433".parse().expect("valid socket address");
        let local = client_bind_addr_for(remote);
        assert!(local.is_ipv6());
    }

    #[tokio::test]
    async fn query_model_capacity_reports_recursive_totals_and_occupancy() {
        let worker_a = uuid::Uuid::new_v4();
        let worker_b = uuid::Uuid::new_v4();
        let model_a = uuid::Uuid::new_v4();
        let model_b = uuid::Uuid::new_v4();
        let model_c = uuid::Uuid::new_v4();
        let model_d = uuid::Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            worker_a,
            TunnelWorker {
                token: worker_a,
                worker_id: uuid::Uuid::new_v4(),
                max_instances: Some(3),
            },
        );
        workers.insert(
            worker_b,
            TunnelWorker {
                token: worker_b,
                worker_id: uuid::Uuid::new_v4(),
                max_instances: Some(2),
            },
        );
        let mut routes = HashMap::new();
        routes.insert(model_a, RouteTarget::Local);
        routes.insert(model_b, RouteTarget::Local);
        routes.insert(model_c, RouteTarget::Worker(worker_a));
        routes.insert(model_d, RouteTarget::Worker(worker_b));
        let ctx = QuarkContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(2),
            mode: QuarkMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };

        let out = handle_query_model_capacity(&ctx)
            .await
            .expect("capacity query should succeed");
        let black_hole_spec::QuarkOut::ModelCapacity { capacity } = out else {
            panic!("unexpected query response");
        };
        assert_eq!(
            capacity,
            QuarkModelCapacity {
                total: Some(7),
                available: Some(3),
                occupied: 4,
            }
        );
    }

    #[tokio::test]
    async fn query_model_capacity_saturates_available_at_zero() {
        let mut routes = HashMap::new();
        routes.insert(uuid::Uuid::new_v4(), RouteTarget::Local);
        routes.insert(uuid::Uuid::new_v4(), RouteTarget::Local);
        let ctx = QuarkContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: QuarkMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };

        let out = handle_query_model_capacity(&ctx)
            .await
            .expect("capacity query should succeed");
        let black_hole_spec::QuarkOut::ModelCapacity { capacity } = out else {
            panic!("unexpected query response");
        };
        assert_eq!(
            capacity,
            QuarkModelCapacity {
                total: Some(1),
                available: Some(0),
                occupied: 2,
            }
        );
    }

    #[tokio::test]
    async fn tunnel_registration_defaults_capacity_to_one_when_omitted() {
        let ctx = QuarkContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: QuarkMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };
        let worker_id = uuid::Uuid::new_v4();

        let out = handle_register_tunnel(worker_id, None, None, &ctx)
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
        assert_eq!(worker.worker_id, worker_id);
        assert_eq!(worker.max_instances, Some(DEFAULT_MAX_INSTANCES));
    }

    #[tokio::test]
    async fn tunnel_registration_preserves_explicit_capacity() {
        let ctx = QuarkContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: QuarkMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
        };
        let worker_id = uuid::Uuid::new_v4();
        let requested = Some(3usize);

        let out = handle_register_tunnel(worker_id, requested, None, &ctx)
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
        assert_eq!(worker.worker_id, worker_id);
        assert_eq!(worker.max_instances, requested);
    }

    #[tokio::test]
    async fn select_start_target_uses_starting_local_instances_for_capacity() {
        let worker_token = uuid::Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            worker_token,
            TunnelWorker {
                token: worker_token,
                worker_id: uuid::Uuid::new_v4(),
                max_instances: Some(1),
            },
        );
        let mut instances = HashMap::new();
        instances.insert(uuid::Uuid::new_v4(), ModelSlot::Starting);
        let ctx = QuarkContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Tcp,
            void_client: None,
            defaults: QuarkServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: QuarkMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(instances),
        };

        let selected = select_start_target(&ctx)
            .await
            .expect("worker should be selected when local start is in progress");
        assert_eq!(selected, RouteTarget::Worker(worker_token));
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
