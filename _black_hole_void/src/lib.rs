use std::{collections::HashMap, fs, io, net::SocketAddr, path::PathBuf, sync::Arc};

use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    time::{timeout, Duration},
};
use tracing::{debug, error, info, warn};

#[cfg(feature = "postgres")]
pub mod migrate;
pub mod object_store;
pub mod persist;

const DEFAULT_LISTEN_ADDR: &str = "[::1]:4434";
const S3_MAX_FRAME_SIZE: usize = 1024 * 1024 * 1024; // 1 GB
const MAX_DOWNLOAD_WAIT_TIMEOUT_MS: u64 = 30_000;

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

/// Wire request sent by the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum VoidIn {
    /// Upload data to object storage. Server responds with VoidOut::Uploaded(id).
    Upload { data: Vec<u8> },
    /// Upload data with a caller-supplied ID, overwriting any existing object for that ID.
    /// Server responds with VoidOut::Uploaded(id).
    UploadWith { id: uuid::Uuid, data: Vec<u8> },
    /// Download an object by its opaque ID. Server responds with VoidOut::Downloaded(data).
    Download { id: uuid::Uuid },
    /// Wait up to `timeout_ms` for an object to exist, then download it.
    DownloadWait { id: uuid::Uuid, timeout_ms: u64 },
}

/// Wire response sent by the server.
#[derive(Debug, Serialize, Deserialize)]
pub enum VoidOut {
    /// Confirms upload; contains the opaque ID.
    Uploaded { id: uuid::Uuid },
    /// Returns downloaded data.
    Downloaded { data: Vec<u8> },
    /// Request timed out while waiting for an upload.
    TimedOut { id: uuid::Uuid },
    /// Error message for any failure.
    Error { message: String },
}

pub struct ServerBuilder {
    keylog: bool,
    key: Option<PathBuf>,
    cert: Option<PathBuf>,
    stateless_retry: bool,
    transport_mode: TransportMode,
    listen: SocketAddr,
    object_namespace: String,
    object_store: Box<dyn object_store::ObjectStore>,
    store: Box<dyn persist::VoidStore>,
}

impl ServerBuilder {
    pub fn new(
        object_store: Box<dyn object_store::ObjectStore>,
        store: Box<dyn persist::VoidStore>,
    ) -> Self {
        Self {
            keylog: false,
            key: None,
            cert: None,
            stateless_retry: false,
            transport_mode: TransportMode::Quic,
            listen: DEFAULT_LISTEN_ADDR
                .parse()
                .expect("default listen address must be valid"),
            object_namespace: "memory".to_string(),
            object_store,
            store,
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

    pub fn listen(mut self, addr: SocketAddr) -> Self {
        self.listen = addr;
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

    pub fn object_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.object_namespace = namespace.into();
        self
    }

    /// Build the server endpoint and context, returning them for reuse.
    async fn setup(self) -> Result<(VoidListener, SocketAddr, Arc<VoidContext>)> {
        // Run migrations before accepting connections.
        self.store.migrate().await.map_err(|e| {
            ServerError::Store(persist::PersistenceError::Message(format!(
                "migration failed: {e}"
            )))
        })?;

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
                VoidListener::Quic(endpoint)
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
                VoidListener::Tcp(listener)
            }
        };

        let local_addr = listener.local_addr().map_err(ServerError::LocalAddr)?;
        info!(%local_addr, "listening");

        let context = Arc::new(VoidContext {
            object_namespace: self.object_namespace,
            object_store: self.object_store,
            store: self.store,
            wait_registry: WaitRegistry::default(),
        });

        Ok((listener, local_addr, context))
    }

    /// Start the server in a background task. Returns the bound address and
    /// a handle that can be used to await or abort the server.
    pub async fn serve(self) -> Result<(SocketAddr, tokio::task::JoinHandle<Result<()>>)> {
        let stateless_retry = self.stateless_retry;
        let (listener, local_addr, context) = self.setup().await?;
        let handle = tokio::spawn(Self::accept_loop(listener, context, stateless_retry));
        Ok((local_addr, handle))
    }

    /// Run the server, blocking until the endpoint is closed.
    pub async fn run(self) -> Result<()> {
        let stateless_retry = self.stateless_retry;
        let (listener, _local_addr, context) = self.setup().await?;
        Self::accept_loop(listener, context, stateless_retry).await
    }

    /// Accept-loop shared by both `run()` and `serve()`.
    async fn accept_loop(
        listener: VoidListener,
        context: Arc<VoidContext>,
        stateless_retry: bool,
    ) -> Result<()> {
        match listener {
            VoidListener::Quic(endpoint) => loop {
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
            },
            VoidListener::Tcp(listener) => loop {
                let (stream, remote_addr) = listener
                    .accept()
                    .await
                    .map_err(ServerError::AcceptTcpConnection)?;
                debug!(remote = %remote_addr, "accepting tcp connection");
                let ctx = Arc::clone(&context);
                tokio::spawn(handle_tcp_connection(stream, ctx));
            },
        }
        Ok(())
    }
}

enum VoidListener {
    Quic(quinn::Endpoint),
    Tcp(TcpListener),
}

impl VoidListener {
    fn local_addr(&self) -> io::Result<SocketAddr> {
        match self {
            Self::Quic(endpoint) => endpoint.local_addr(),
            Self::Tcp(listener) => listener.local_addr(),
        }
    }
}

struct VoidContext {
    object_namespace: String,
    object_store: Box<dyn object_store::ObjectStore>,
    store: Box<dyn persist::VoidStore>,
    wait_registry: WaitRegistry,
}

#[derive(Default)]
struct WaitRegistry {
    waiters: Mutex<HashMap<uuid::Uuid, Arc<tokio::sync::Notify>>>,
}

impl WaitRegistry {
    async fn waiter_for(&self, id: uuid::Uuid) -> Arc<tokio::sync::Notify> {
        let mut waiters = self.waiters.lock().await;
        waiters
            .entry(id)
            .or_insert_with(|| Arc::new(tokio::sync::Notify::new()))
            .clone()
    }

    async fn notify_upload(&self, id: uuid::Uuid) {
        let waiter = {
            let mut waiters = self.waiters.lock().await;
            waiters.remove(&id)
        };
        if let Some(waiter) = waiter {
            waiter.notify_waiters();
        }
    }

    async fn release_if_idle(&self, id: uuid::Uuid, waiter: &Arc<tokio::sync::Notify>) {
        let mut waiters = self.waiters.lock().await;
        if let Some(current) = waiters.get(&id) {
            if Arc::ptr_eq(current, waiter) && Arc::strong_count(current) == 2 {
                waiters.remove(&id);
            }
        }
    }
}

async fn handle_connection(incoming: quinn::Incoming, context: Arc<VoidContext>) {
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
    (mut send, recv): (quinn::SendStream, quinn::RecvStream),
    context: Arc<VoidContext>,
) {
    let mut recv = recv;
    let request = match read_frame_quic(&mut recv).await {
        Ok(r) => r,
        Err(e) => {
            error!("failed to read request frame: {e}");
            return;
        }
    };

    let response = match request {
        VoidIn::Upload { data } => handle_upload(&context, None, data).await,
        VoidIn::UploadWith { id, data } => handle_upload(&context, Some(id), data).await,
        VoidIn::Download { id } => handle_download(&context, id).await,
        VoidIn::DownloadWait { id, timeout_ms } => {
            handle_download_wait(&context, id, timeout_ms).await
        }
    };

    if let Err(e) = write_frame_quic(&mut send, &response).await {
        match &e {
            // High-throughput flows can cancel in-flight requests while this server
            // is preparing a response. Treat those disconnects as expected churn.
            ServerError::WriteFrame(quinn::WriteError::ConnectionLost(_)) => {
                debug!("client disconnected before response frame write completed")
            }
            _ => error!("failed to write response frame: {e}"),
        }
    }
}

async fn handle_tcp_connection(mut stream: TcpStream, context: Arc<VoidContext>) {
    loop {
        let request: VoidIn = match read_frame_io(&mut stream).await {
            Ok(request) => request,
            Err(ServerError::UnexpectedEof) => return,
            Err(error) => {
                error!("failed to read tcp request frame: {error}");
                return;
            }
        };

        let response = match request {
            VoidIn::Upload { data } => handle_upload(&context, None, data).await,
            VoidIn::UploadWith { id, data } => handle_upload(&context, Some(id), data).await,
            VoidIn::Download { id } => handle_download(&context, id).await,
            VoidIn::DownloadWait { id, timeout_ms } => {
                handle_download_wait(&context, id, timeout_ms).await
            }
        };

        if let Err(error) = write_frame_io(&mut stream, &response).await {
            if matches!(error, ServerError::WriteFrameIo(ref err) if matches!(
                err.kind(),
                io::ErrorKind::BrokenPipe
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::ConnectionReset
            )) {
                debug!("tcp client disconnected before response frame write completed");
            } else {
                error!("failed to write tcp response frame: {error}");
            }
            return;
        }
    }
}

async fn handle_upload(context: &VoidContext, id: Option<uuid::Uuid>, data: Vec<u8>) -> VoidOut {
    if data.len() > S3_MAX_FRAME_SIZE {
        return VoidOut::Error {
            message: format!(
                "upload size {} exceeds maximum {}",
                data.len(),
                S3_MAX_FRAME_SIZE
            ),
        };
    }

    let id = id.unwrap_or_else(uuid::Uuid::new_v4);
    let key = id.to_string();
    let size_bytes = i64::try_from(data.len()).unwrap_or(i64::MAX);

    match context.object_store.put(key.clone(), data).await {
        Ok(_) => {
            // Persist the object metadata.
            if let Err(e) = context
                .store
                .insert_object(
                    id,
                    context.object_namespace.clone(),
                    key.clone(),
                    size_bytes,
                )
                .await
            {
                warn!(error = %e, "failed to persist object metadata");
            }
            if let Err(e) = context.store.publish_upload_notification(id).await {
                warn!(error = %e, "failed to publish upload notification");
            }

            info!(%id, "uploaded");
            context.wait_registry.notify_upload(id).await;
            VoidOut::Uploaded { id }
        }
        Err(e) => {
            error!(error = %e, "put_object failed");
            VoidOut::Error {
                message: format!("upload failed: {e}"),
            }
        }
    }
}

async fn handle_download(context: &VoidContext, id: uuid::Uuid) -> VoidOut {
    match try_download(context, id).await {
        DownloadAttempt::Found(data) => {
            info!(%id, bytes = data.len(), "downloaded");
            VoidOut::Downloaded { data }
        }
        DownloadAttempt::Missing => VoidOut::Error {
            message: format!("object not found: {id}"),
        },
        DownloadAttempt::Failed(message) => {
            warn!(%id, error = %message, "download failed");
            VoidOut::Error { message }
        }
    }
}

async fn handle_download_wait(context: &VoidContext, id: uuid::Uuid, timeout_ms: u64) -> VoidOut {
    if timeout_ms == 0 || timeout_ms > MAX_DOWNLOAD_WAIT_TIMEOUT_MS {
        return VoidOut::Error {
            message: format!(
                "timeout_ms must be in 1..={MAX_DOWNLOAD_WAIT_TIMEOUT_MS}, got {timeout_ms}"
            ),
        };
    }

    match try_download(context, id).await {
        DownloadAttempt::Found(data) => {
            info!(%id, bytes = data.len(), "downloaded immediately");
            return VoidOut::Downloaded { data };
        }
        DownloadAttempt::Failed(message) => return VoidOut::Error { message },
        DownloadAttempt::Missing => {}
    }

    let waiter = context.wait_registry.waiter_for(id).await;
    match try_download(context, id).await {
        DownloadAttempt::Found(data) => {
            info!(%id, bytes = data.len(), "downloaded after waiter registration");
            return VoidOut::Downloaded { data };
        }
        DownloadAttempt::Failed(message) => return VoidOut::Error { message },
        DownloadAttempt::Missing => {}
    }

    let wait_duration = Duration::from_millis(timeout_ms);
    let deadline = tokio::time::Instant::now() + wait_duration;
    let mut use_backend_wait = context.store.supports_wait_notifications();

    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            context.wait_registry.release_if_idle(id, &waiter).await;
            return VoidOut::TimedOut { id };
        }
        let remaining = deadline.saturating_duration_since(now);

        if use_backend_wait {
            let remaining_ms = u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX);
            tokio::select! {
                _ = waiter.notified() => break,
                backend_wait = context.store.wait_for_upload_notification(id, remaining_ms) => {
                    match backend_wait {
                        Ok(true) => break,
                        Ok(false) => {
                            context.wait_registry.release_if_idle(id, &waiter).await;
                            return VoidOut::TimedOut { id };
                        }
                        Err(e) => {
                            warn!(%id, error = %e, "backend wait notification failed");
                            use_backend_wait = false;
                            continue;
                        }
                    }
                }
                _ = tokio::time::sleep(remaining) => {
                    context.wait_registry.release_if_idle(id, &waiter).await;
                    return VoidOut::TimedOut { id };
                }
            }
        } else if timeout(remaining, waiter.notified()).await.is_err() {
            context.wait_registry.release_if_idle(id, &waiter).await;
            return VoidOut::TimedOut { id };
        } else {
            break;
        }
    }

    context.wait_registry.release_if_idle(id, &waiter).await;
    match try_download(context, id).await {
        DownloadAttempt::Found(data) => {
            info!(%id, bytes = data.len(), "downloaded after upload notification");
            VoidOut::Downloaded { data }
        }
        DownloadAttempt::Failed(message) => VoidOut::Error { message },
        DownloadAttempt::Missing => {
            warn!(%id, "waiter notified but object is still missing");
            VoidOut::TimedOut { id }
        }
    }
}

enum DownloadAttempt {
    Found(Vec<u8>),
    Missing,
    Failed(String),
}

async fn try_download(context: &VoidContext, id: uuid::Uuid) -> DownloadAttempt {
    let record = match context.store.get_object(id).await {
        Ok(Some(record)) => record,
        Ok(None) => return DownloadAttempt::Missing,
        Err(e) => {
            error!(%id, error = %e, "failed to look up object");
            return DownloadAttempt::Failed(format!("lookup failed: {e}"));
        }
    };

    match context.object_store.get(&record.key).await {
        Ok(data) => DownloadAttempt::Found(data),
        Err(e) => DownloadAttempt::Failed(format!("not found: {e}")),
    }
}

/// Read a length-prefixed postcard frame from the stream.
async fn read_frame_quic<T: for<'de> Deserialize<'de>>(
    recv: &mut quinn::RecvStream,
) -> std::result::Result<T, ServerError> {
    let len = match recv.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(ServerError::UnexpectedEof);
        }
        Err(e) => {
            return Err(ServerError::ReadFrameLength(e));
        }
    };

    if len > S3_MAX_FRAME_SIZE {
        return Err(ServerError::FrameTooLarge(len));
    }

    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(ServerError::ReadFramePayload)?;

    from_bytes(&payload).map_err(ServerError::DecodeRequest)
}

/// Write a length-prefixed postcard frame to the stream.
async fn write_frame_quic(
    send: &mut quinn::SendStream,
    out: &impl Serialize,
) -> std::result::Result<(), ServerError> {
    let payload = to_allocvec(out).map_err(ServerError::EncodeResponse)?;
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

async fn read_frame_io<R, T>(recv: &mut R) -> std::result::Result<T, ServerError>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let len = match recv.read_u32().await {
        Ok(len) => len as usize,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            return Err(ServerError::UnexpectedEof);
        }
        Err(e) => {
            return Err(ServerError::ReadFrameLength(e));
        }
    };

    if len > S3_MAX_FRAME_SIZE {
        return Err(ServerError::FrameTooLarge(len));
    }

    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(ServerError::ReadFramePayloadIo)?;

    from_bytes(&payload).map_err(ServerError::DecodeRequest)
}

async fn write_frame_io<W>(
    send: &mut W,
    out: &impl Serialize,
) -> std::result::Result<(), ServerError>
where
    W: AsyncWrite + Unpin,
{
    let payload = to_allocvec(out).map_err(ServerError::EncodeResponse)?;
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
    let dirs = directories_next::ProjectDirs::from("org", "blackhole", "void").unwrap();
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
    #[error("unexpected EOF while reading frame length")]
    UnexpectedEof,
    #[error("failed to read frame length: {0}")]
    ReadFrameLength(io::Error),
    #[error("frame payload too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("failed to read frame payload: {0}")]
    ReadFramePayload(#[source] quinn::ReadExactError),
    #[error("failed to read frame payload: {0}")]
    ReadFramePayloadIo(#[source] io::Error),
    #[error("failed to decode request: {0}")]
    DecodeRequest(postcard::Error),
    #[error("failed to encode response: {0}")]
    EncodeResponse(postcard::Error),
    #[error("failed to write frame: {0}")]
    WriteFrame(quinn::WriteError),
    #[error("failed to write frame: {0}")]
    WriteFrameIo(io::Error),
    #[error("persistence error: {0}")]
    Store(#[source] persist::PersistenceError),
}

pub type Result<T> = std::result::Result<T, ServerError>;

pub fn init_tracing() -> std::result::Result<(), tracing::subscriber::SetGlobalDefaultError> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .finish(),
    )
}
