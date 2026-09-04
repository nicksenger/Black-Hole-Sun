use std::{
    collections::{HashMap, HashSet},
    fs, io,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use black_hole_type::{
    TransferAbort, TransferBegin, TransferChunk, TransferHash, TransferManifest, TransferRecord,
    TransferStreamFrame, TRANSFER_PROTOCOL_VERSION,
};
use postcard::{from_bytes, to_allocvec};
use quinn::crypto::rustls::QuicServerConfig;
use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
/// Maximum number of parts in one multipart upload (matches the S3 limit).
const MAX_MULTIPART_PARTS: u32 = 10_000;
const TRANSFER_CLEANUP_INTERVAL: Duration = Duration::from_secs(1);
const IMMUTABLE_TRANSFER_CHUNK_PREFIX: &str = "transfer-chunk-";
const TRANSFER_RECORD_PREFIX: &str = "transfer-record-";

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
    /// Begin a multipart upload for objects too large for one frame. When
    /// `id` is None the server assigns one. Server responds with VoidOut::Uploaded(id).
    UploadBegin {
        id: Option<uuid::Uuid>,
        total_size: u64,
    },
    /// Upload one part (1-indexed) of an in-flight multipart upload.
    UploadPart {
        id: uuid::Uuid,
        part_number: u32,
        data: Vec<u8>,
    },
    /// Complete an in-flight multipart upload, making the object visible to
    /// downloads. `part_count` must match the number of parts uploaded.
    UploadFinish { id: uuid::Uuid, part_count: u32 },
    /// Download up to `length` bytes of an object starting at `offset`.
    /// Returns fewer bytes (possibly zero) at end of object.
    DownloadRange {
        id: uuid::Uuid,
        offset: u64,
        length: u64,
    },
    /// Begin a progressive transfer. The transfer record is durable but its
    /// artifact is not authoritative until `TransferCommit`.
    TransferBegin { begin: TransferBegin },
    /// Persist one immutable, independently readable transfer chunk.
    TransferChunk {
        transfer_id: uuid::Uuid,
        index: u32,
        data: Vec<u8>,
        hash: TransferHash,
    },
    /// Read current transfer state so a receiver can stage available chunks.
    TransferInspect { transfer_id: uuid::Uuid },
    /// Atomically publish a complete transfer manifest after validating every
    /// chunk and the aggregate hash.
    TransferCommit {
        transfer_id: uuid::Uuid,
        aggregate_hash: TransferHash,
    },
    /// Abort a transfer and remove all of its chunks.
    TransferAbort {
        transfer_id: uuid::Uuid,
        reason: String,
    },
    /// Switch this QUIC/TCP channel into producer streaming mode. Bytes are
    /// persisted as immutable chunks while they are received.
    TransferStreamUpload {
        begin: TransferBegin,
        authorization: [u8; 32],
    },
    /// Switch this channel into receiver streaming mode. Already-persisted
    /// and newly arriving chunks are sent in order until commit or abort.
    TransferStreamDownload {
        transfer_id: uuid::Uuid,
        authorization: [u8; 32],
    },
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
    /// Generic acknowledgment (part uploads, etc).
    Ack,
    /// Error message for any failure.
    Error { message: String },
    /// Current state of a progressive transfer.
    Transfer { record: TransferRecord },
    /// One chunk was persisted and is now independently readable.
    TransferChunkStored { chunk: TransferChunk },
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
            multipart_uploads: tokio::sync::Mutex::new(HashMap::new()),
            transfer_mutation: tokio::sync::Mutex::new(()),
            active_transfers: tokio::sync::Mutex::new(HashSet::new()),
        });
        spawn_transfer_cleanup(&context);

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

/// One in-flight multipart upload, tracked between UploadBegin and
/// UploadFinish. `store_session_id` is the backend-specific session handle;
/// `parts` maps part number -> part token (ETag for S3).
#[derive(Debug)]
struct MultipartSession {
    store_session_id: String,
    parts: std::collections::BTreeMap<u32, String>,
    total_size: u64,
}

struct VoidContext {
    object_namespace: String,
    object_store: Box<dyn object_store::ObjectStore>,
    store: Box<dyn persist::VoidStore>,
    wait_registry: WaitRegistry,
    multipart_uploads: tokio::sync::Mutex<HashMap<uuid::Uuid, MultipartSession>>,
    /// Serializes transfer record read-modify-write updates. Chunks remain
    /// independently downloadable while a different transfer is updated.
    transfer_mutation: tokio::sync::Mutex<()>,
    active_transfers: tokio::sync::Mutex<HashSet<uuid::Uuid>>,
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

    match request {
        VoidIn::TransferStreamUpload {
            begin,
            authorization,
        } => {
            if let Err(error) = handle_transfer_stream_upload_quic(
                &mut send,
                &mut recv,
                &context,
                begin,
                authorization,
            )
            .await
            {
                debug!(%error, "transfer upload stream ended");
            }
            return;
        }
        VoidIn::TransferStreamDownload {
            transfer_id,
            authorization,
        } => {
            if let Err(error) = handle_transfer_stream_download_quic(
                &mut send,
                &context,
                transfer_id,
                authorization,
            )
            .await
            {
                debug!(%error, "transfer download stream ended");
            }
            return;
        }
        request => {
            let response = handle_request(request, &context).await;
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
            return;
        }
    }
}

async fn handle_request(request: VoidIn, context: &VoidContext) -> VoidOut {
    let response = match request {
        VoidIn::Upload { data } => handle_upload(&context, None, data).await,
        VoidIn::UploadWith { id, data } => handle_upload(&context, Some(id), data).await,
        VoidIn::Download { id } => handle_download(&context, id).await,
        VoidIn::DownloadWait { id, timeout_ms } => {
            handle_download_wait(&context, id, timeout_ms).await
        }
        VoidIn::UploadBegin { id, total_size } => {
            handle_upload_begin(&context, id, total_size).await
        }
        VoidIn::UploadPart {
            id,
            part_number,
            data,
        } => handle_upload_part(&context, id, part_number, data).await,
        VoidIn::UploadFinish { id, part_count } => {
            handle_upload_finish(&context, id, part_count).await
        }
        VoidIn::DownloadRange { id, offset, length } => {
            handle_download_range(&context, id, offset, length).await
        }
        VoidIn::TransferBegin { begin } => handle_transfer_begin(&context, begin).await,
        VoidIn::TransferChunk {
            transfer_id,
            index,
            data,
            hash,
        } => handle_transfer_chunk(&context, transfer_id, index, data, hash).await,
        VoidIn::TransferInspect { transfer_id } => {
            handle_transfer_inspect(&context, transfer_id).await
        }
        VoidIn::TransferCommit {
            transfer_id,
            aggregate_hash,
        } => handle_transfer_commit(&context, transfer_id, aggregate_hash).await,
        VoidIn::TransferAbort {
            transfer_id,
            reason,
        } => handle_transfer_abort(&context, transfer_id, reason).await,
        VoidIn::TransferStreamUpload { .. } | VoidIn::TransferStreamDownload { .. } => {
            VoidOut::Error {
                message: "stream transfer request requires a dedicated channel".to_string(),
            }
        }
    };
    response
}

async fn handle_tcp_connection(mut stream: TcpStream, context: Arc<VoidContext>) {
    loop {
        let request: VoidIn = match read_frame_io(&mut stream).await {
            Ok(request) => request,
            Err(ServerError::UnexpectedEof) => return,
            Err(error) => {
                warn!("failed to read tcp request frame: {error}");
                return;
            }
        };

        match request {
            VoidIn::TransferStreamUpload {
                begin,
                authorization,
            } => {
                if let Err(error) =
                    handle_transfer_stream_upload_io(&mut stream, &context, begin, authorization)
                        .await
                {
                    debug!(%error, "TCP transfer upload stream ended");
                }
                return;
            }
            VoidIn::TransferStreamDownload {
                transfer_id,
                authorization,
            } => {
                if let Err(error) = handle_transfer_stream_download_io(
                    &mut stream,
                    &context,
                    transfer_id,
                    authorization,
                )
                .await
                {
                    debug!(%error, "TCP transfer download stream ended");
                }
                return;
            }
            request => {
                let response = handle_request(request, &context).await;
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
        };
    }
}

fn valid_transfer_authorization(begin: &TransferBegin, authorization: &[u8; 32]) -> bool {
    TransferHash(Sha256::digest(authorization).into()) == begin.authorization_hash
}

async fn handle_stream_upload_frame(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    expected_index: &mut u32,
    frame: TransferStreamFrame,
) -> VoidOut {
    match frame {
        TransferStreamFrame::Chunk { index, data, hash } => {
            if index != *expected_index {
                return VoidOut::Error {
                    message: format!(
                        "stream chunk index {index} is out of order; expected {}",
                        *expected_index
                    ),
                };
            }
            let response = handle_transfer_chunk(context, transfer_id, index, data, hash).await;
            if matches!(response, VoidOut::TransferChunkStored { .. }) {
                *expected_index = expected_index.saturating_add(1);
            }
            response
        }
        TransferStreamFrame::Commit { aggregate_hash } => {
            handle_transfer_commit(context, transfer_id, aggregate_hash).await
        }
        TransferStreamFrame::Abort { reason } => {
            handle_transfer_abort(context, transfer_id, reason).await
        }
        TransferStreamFrame::Begin(_) => VoidOut::Error {
            message: "duplicate begin frame on transfer upload stream".to_string(),
        },
    }
}

async fn handle_transfer_stream_upload_quic(
    send: &mut quinn::SendStream,
    recv: &mut quinn::RecvStream,
    context: &VoidContext,
    begin: TransferBegin,
    authorization: [u8; 32],
) -> Result<()> {
    if !valid_transfer_authorization(&begin, &authorization) {
        return write_frame_quic(
            send,
            &VoidOut::Error {
                message: "invalid transfer authorization".to_string(),
            },
        )
        .await;
    }
    let transfer_id = begin.transfer_id;
    let response = handle_transfer_begin(context, begin).await;
    let accepted = matches!(
        &response,
        VoidOut::Transfer {
            record: TransferRecord::InProgress { .. }
        }
    );
    write_frame_quic(send, &response).await?;
    if !accepted {
        return Ok(());
    }

    let mut expected_index = 0;
    loop {
        let frame: TransferStreamFrame = match read_frame_quic(recv).await {
            Ok(frame) => frame,
            Err(ServerError::UnexpectedEof) => return Ok(()),
            Err(error) => return Err(error),
        };
        let terminal = matches!(
            &frame,
            TransferStreamFrame::Commit { .. } | TransferStreamFrame::Abort { .. }
        );
        let response =
            handle_stream_upload_frame(context, transfer_id, &mut expected_index, frame).await;
        let failed = matches!(&response, VoidOut::Error { .. });
        if terminal || failed {
            write_frame_quic(send, &response).await?;
            return Ok(());
        }
    }
}

async fn handle_transfer_stream_upload_io<S>(
    stream: &mut S,
    context: &VoidContext,
    begin: TransferBegin,
    authorization: [u8; 32],
) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    if !valid_transfer_authorization(&begin, &authorization) {
        return write_frame_io(
            stream,
            &VoidOut::Error {
                message: "invalid transfer authorization".to_string(),
            },
        )
        .await;
    }
    let transfer_id = begin.transfer_id;
    let response = handle_transfer_begin(context, begin).await;
    let accepted = matches!(
        &response,
        VoidOut::Transfer {
            record: TransferRecord::InProgress { .. }
        }
    );
    write_frame_io(stream, &response).await?;
    if !accepted {
        return Ok(());
    }

    let mut expected_index = 0;
    loop {
        let frame: TransferStreamFrame = match read_frame_io(stream).await {
            Ok(frame) => frame,
            Err(ServerError::UnexpectedEof) => return Ok(()),
            Err(error) => return Err(error),
        };
        let terminal = matches!(
            &frame,
            TransferStreamFrame::Commit { .. } | TransferStreamFrame::Abort { .. }
        );
        let response =
            handle_stream_upload_frame(context, transfer_id, &mut expected_index, frame).await;
        let failed = matches!(&response, VoidOut::Error { .. });
        if terminal || failed {
            write_frame_io(stream, &response).await?;
            return Ok(());
        }
    }
}

async fn transfer_stream_snapshot(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
) -> std::result::Result<TransferRecord, String> {
    match handle_transfer_inspect(context, transfer_id).await {
        VoidOut::Transfer { record } => Ok(record),
        VoidOut::Error { message } => Err(message),
        _ => Err("unexpected transfer inspection response".to_string()),
    }
}

async fn read_transfer_chunk_frame(
    context: &VoidContext,
    chunk: &TransferChunk,
) -> std::result::Result<TransferStreamFrame, String> {
    let data = match try_download(context, chunk.object_id).await {
        DownloadAttempt::Found(data) => data,
        DownloadAttempt::Missing => {
            return Err(format!("transfer chunk {} is missing", chunk.object_id))
        }
        DownloadAttempt::Failed(message) => return Err(message),
    };
    if data.len() as u64 != chunk.len || transfer_hash(&data) != chunk.hash {
        return Err(format!(
            "transfer chunk {} failed validation",
            chunk.object_id
        ));
    }
    Ok(TransferStreamFrame::Chunk {
        index: chunk.index,
        data,
        hash: chunk.hash,
    })
}

async fn handle_transfer_stream_download_quic(
    send: &mut quinn::SendStream,
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    authorization: [u8; 32],
) -> Result<()> {
    let initial = transfer_stream_snapshot(context, transfer_id)
        .await
        .map_err(|message| ServerError::Transfer(message))?;
    if !valid_transfer_authorization(initial.begin(), &authorization) {
        return Err(ServerError::Transfer(
            "invalid transfer authorization".to_string(),
        ));
    }
    write_frame_quic(send, &TransferStreamFrame::Begin(initial.begin().clone())).await?;
    let mut next_index = 0usize;
    loop {
        let record = transfer_stream_snapshot(context, transfer_id)
            .await
            .map_err(ServerError::Transfer)?;
        let (chunks, terminal) = match &record {
            TransferRecord::InProgress { chunks, .. } => (chunks.as_slice(), None),
            TransferRecord::Committed(manifest) => (
                manifest.chunks.as_slice(),
                Some(TransferStreamFrame::Commit {
                    aggregate_hash: manifest.begin.expected_hash,
                }),
            ),
            TransferRecord::Aborted(abort) => (
                &[][..],
                Some(TransferStreamFrame::Abort {
                    reason: abort.reason.clone(),
                }),
            ),
        };
        while let Some(chunk) = chunks.get(next_index) {
            let frame = read_transfer_chunk_frame(context, chunk)
                .await
                .map_err(ServerError::Transfer)?;
            write_frame_quic(send, &frame).await?;
            next_index += 1;
        }
        if let Some(terminal) = terminal {
            write_frame_quic(send, &terminal).await?;
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn handle_transfer_stream_download_io<S>(
    stream: &mut S,
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    authorization: [u8; 32],
) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let initial = transfer_stream_snapshot(context, transfer_id)
        .await
        .map_err(ServerError::Transfer)?;
    if !valid_transfer_authorization(initial.begin(), &authorization) {
        return Err(ServerError::Transfer(
            "invalid transfer authorization".to_string(),
        ));
    }
    write_frame_io(stream, &TransferStreamFrame::Begin(initial.begin().clone())).await?;
    let mut next_index = 0usize;
    loop {
        let record = transfer_stream_snapshot(context, transfer_id)
            .await
            .map_err(ServerError::Transfer)?;
        let (chunks, terminal) = match &record {
            TransferRecord::InProgress { chunks, .. } => (chunks.as_slice(), None),
            TransferRecord::Committed(manifest) => (
                manifest.chunks.as_slice(),
                Some(TransferStreamFrame::Commit {
                    aggregate_hash: manifest.begin.expected_hash,
                }),
            ),
            TransferRecord::Aborted(abort) => (
                &[][..],
                Some(TransferStreamFrame::Abort {
                    reason: abort.reason.clone(),
                }),
            ),
        };
        while let Some(chunk) = chunks.get(next_index) {
            let frame = read_transfer_chunk_frame(context, chunk)
                .await
                .map_err(ServerError::Transfer)?;
            write_frame_io(stream, &frame).await?;
            next_index += 1;
        }
        if let Some(terminal) = terminal {
            write_frame_io(stream, &terminal).await?;
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
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
    match context.store.get_object(id).await {
        Ok(Some(existing))
            if existing.key.starts_with(IMMUTABLE_TRANSFER_CHUNK_PREFIX)
                || existing.key.starts_with(TRANSFER_RECORD_PREFIX) =>
        {
            return VoidOut::Error {
                message: format!("object {id} is managed by the transfer protocol"),
            };
        }
        Ok(_) => {}
        Err(error) => {
            return VoidOut::Error {
                message: format!("failed to check existing object {id}: {error}"),
            }
        }
    }
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

async fn handle_transfer_chunk_upload(
    context: &VoidContext,
    id: uuid::Uuid,
    data: Vec<u8>,
) -> VoidOut {
    if data.len() > S3_MAX_FRAME_SIZE {
        return VoidOut::Error {
            message: format!(
                "upload size {} exceeds maximum {}",
                data.len(),
                S3_MAX_FRAME_SIZE
            ),
        };
    }
    match context.store.get_object(id).await {
        Ok(Some(_)) => {
            return VoidOut::Error {
                message: format!("object already exists: {id}"),
            }
        }
        Ok(None) => {}
        Err(error) => {
            return VoidOut::Error {
                message: format!("failed to check transfer chunk object {id}: {error}"),
            }
        }
    }
    let key = format!("{IMMUTABLE_TRANSFER_CHUNK_PREFIX}{id}");
    let size_bytes = i64::try_from(data.len()).unwrap_or(i64::MAX);
    if let Err(error) = context.object_store.put(key.clone(), data).await {
        return VoidOut::Error {
            message: format!("transfer chunk upload failed: {error}"),
        };
    }
    if let Err(error) = context
        .store
        .insert_object(
            id,
            context.object_namespace.clone(),
            key.clone(),
            size_bytes,
        )
        .await
    {
        let _ = context.object_store.delete(&key).await;
        return VoidOut::Error {
            message: format!("failed to persist transfer chunk metadata: {error}"),
        };
    }
    if let Err(error) = context.store.publish_upload_notification(id).await {
        warn!(%id, %error, "failed to publish transfer chunk notification");
    }
    context.wait_registry.notify_upload(id).await;
    VoidOut::Uploaded { id }
}

async fn handle_upload_begin(
    context: &VoidContext,
    id: Option<uuid::Uuid>,
    total_size: u64,
) -> VoidOut {
    let id = id.unwrap_or_else(uuid::Uuid::new_v4);

    {
        let uploads = context.multipart_uploads.lock().await;
        if uploads.contains_key(&id) {
            return VoidOut::Error {
                message: format!("multipart upload already in flight for {id}"),
            };
        }
    }

    let key = id.to_string();
    match context.object_store.begin_multipart(&key).await {
        Ok(store_session_id) => {
            context.multipart_uploads.lock().await.insert(
                id,
                MultipartSession {
                    store_session_id,
                    parts: std::collections::BTreeMap::new(),
                    total_size: 0,
                },
            );
            info!(%id, total_size, "multipart upload begun");
            VoidOut::Uploaded { id }
        }
        Err(e) => {
            error!(error = %e, "begin_multipart failed");
            VoidOut::Error {
                message: format!("failed to begin multipart upload: {e}"),
            }
        }
    }
}

async fn handle_upload_part(
    context: &VoidContext,
    id: uuid::Uuid,
    part_number: u32,
    data: Vec<u8>,
) -> VoidOut {
    if part_number == 0 || part_number > MAX_MULTIPART_PARTS {
        return VoidOut::Error {
            message: format!("part_number must be in 1..={MAX_MULTIPART_PARTS}, got {part_number}"),
        };
    }

    let key = id.to_string();
    let data_len = data.len() as u64;
    let store_session_id = {
        let uploads = context.multipart_uploads.lock().await;
        match uploads.get(&id) {
            Some(session) => session.store_session_id.clone(),
            None => {
                return VoidOut::Error {
                    message: format!("no multipart upload in flight for {id}"),
                }
            }
        }
    };

    match context
        .object_store
        .upload_part(&key, &store_session_id, part_number, data)
        .await
    {
        Ok(token) => {
            let mut uploads = context.multipart_uploads.lock().await;
            if let Some(session) = uploads.get_mut(&id) {
                session.parts.insert(part_number, token);
                session.total_size = session.total_size.saturating_add(data_len);
            }
            VoidOut::Ack
        }
        Err(e) => {
            error!(error = %e, part_number, "upload_part failed");
            // Drop the session so a retry can begin a fresh upload.
            let mut uploads = context.multipart_uploads.lock().await;
            if let Some(session) = uploads.remove(&id) {
                let _ = context
                    .object_store
                    .abort_multipart(&key, &session.store_session_id)
                    .await;
            }
            VoidOut::Error {
                message: format!("failed to upload part {part_number}: {e}"),
            }
        }
    }
}

async fn handle_upload_finish(context: &VoidContext, id: uuid::Uuid, part_count: u32) -> VoidOut {
    if part_count == 0 || part_count > MAX_MULTIPART_PARTS {
        return VoidOut::Error {
            message: format!("part_count must be in 1..={MAX_MULTIPART_PARTS}, got {part_count}"),
        };
    }

    let key = id.to_string();
    let Some(session) = context.multipart_uploads.lock().await.remove(&id) else {
        return VoidOut::Error {
            message: format!("no multipart upload in flight for {id}"),
        };
    };

    // Parts must be exactly 1..=part_count with no gaps or duplicates.
    let expected: std::collections::BTreeSet<u32> = (1..=part_count).collect();
    if session
        .parts
        .keys()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        != expected
    {
        let _ = context
            .object_store
            .abort_multipart(&key, &session.store_session_id)
            .await;
        return VoidOut::Error {
            message: format!(
                "multipart upload for {id} has {} parts, expected exactly 1..={part_count}",
                session.parts.len()
            ),
        };
    }

    let parts: Vec<(u32, String)> = session.parts.into_iter().collect();
    let total_size = session.total_size;

    if let Err(e) = context
        .object_store
        .finish_multipart(&key, &session.store_session_id, &parts)
        .await
    {
        error!(error = %e, "finish_multipart failed");
        return VoidOut::Error {
            message: format!("failed to finish multipart upload: {e}"),
        };
    }

    // Persist the object metadata (same as single-shot uploads).
    if let Err(e) = context
        .store
        .insert_object(
            id,
            context.object_namespace.clone(),
            key.clone(),
            total_size as i64,
        )
        .await
    {
        warn!(error = %e, "failed to persist object metadata");
    }
    if let Err(e) = context.store.publish_upload_notification(id).await {
        warn!(error = %e, "failed to publish upload notification");
    }

    info!(%id, bytes = total_size, parts = part_count, "multipart upload finished");
    context.wait_registry.notify_upload(id).await;
    VoidOut::Uploaded { id }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn transfer_hash(data: &[u8]) -> TransferHash {
    TransferHash(Sha256::digest(data).into())
}

async fn persist_transfer_record(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    record: &TransferRecord,
) -> std::result::Result<(), String> {
    let bytes = to_allocvec(record).map_err(|error| format!("encode transfer record: {error}"))?;
    let key = format!("{TRANSFER_RECORD_PREFIX}{transfer_id}");
    let size_bytes = i64::try_from(bytes.len()).unwrap_or(i64::MAX);
    let existed = context
        .store
        .get_object(transfer_id)
        .await
        .map_err(|error| format!("failed to inspect transfer record metadata: {error}"))?
        .is_some();
    context
        .object_store
        .put(key.clone(), bytes)
        .await
        .map_err(|error| format!("failed to persist transfer record: {error}"))?;
    if let Err(error) = context
        .store
        .insert_object(
            transfer_id,
            context.object_namespace.clone(),
            key.clone(),
            size_bytes,
        )
        .await
    {
        if !existed {
            let _ = context.object_store.delete(&key).await;
            return Err(format!(
                "failed to persist transfer record metadata: {error}"
            ));
        }
        warn!(
            %transfer_id,
            %error,
            "failed to refresh existing transfer record metadata"
        );
    }
    if let Err(error) = context.store.publish_upload_notification(transfer_id).await {
        warn!(%transfer_id, %error, "failed to publish transfer update");
    }
    context.wait_registry.notify_upload(transfer_id).await;
    Ok(())
}

async fn load_transfer_record(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
) -> std::result::Result<TransferRecord, String> {
    match try_download(context, transfer_id).await {
        DownloadAttempt::Found(bytes) => from_bytes(&bytes)
            .map_err(|error| format!("invalid transfer record {transfer_id}: {error}")),
        DownloadAttempt::Missing => Err(format!("transfer not found: {transfer_id}")),
        DownloadAttempt::Failed(message) => Err(message),
    }
}

async fn delete_object(context: &VoidContext, id: uuid::Uuid) -> std::result::Result<(), String> {
    let record = context
        .store
        .get_object(id)
        .await
        .map_err(|error| format!("failed to look up object {id} for deletion: {error}"))?;
    if let Some(record) = record {
        context
            .object_store
            .delete(&record.key)
            .await
            .map_err(|error| format!("failed to delete object {id}: {error}"))?;
        context
            .store
            .delete_object(id)
            .await
            .map_err(|error| format!("failed to delete object metadata {id}: {error}"))?;
    }
    Ok(())
}

async fn abort_transfer_locked(
    context: &VoidContext,
    begin: TransferBegin,
    chunks: Vec<TransferChunk>,
    reason: String,
) -> std::result::Result<TransferRecord, String> {
    for chunk in chunks {
        if let Err(error) = delete_object(context, chunk.object_id).await {
            warn!(
                transfer_id = %begin.transfer_id,
                chunk_id = %chunk.object_id,
                %error,
                "failed to clean up transfer chunk"
            );
        }
    }
    let transfer_id = begin.transfer_id;
    let record = TransferRecord::Aborted(TransferAbort {
        begin,
        reason,
        aborted_unix_ms: unix_time_ms(),
    });
    persist_transfer_record(context, transfer_id, &record).await?;
    context.active_transfers.lock().await.remove(&transfer_id);
    Ok(record)
}

async fn expire_transfer_locked(
    context: &VoidContext,
    record: TransferRecord,
) -> std::result::Result<TransferRecord, String> {
    match record {
        TransferRecord::InProgress {
            begin,
            chunks,
            revision: _,
        } if unix_time_ms() >= begin.deadline_unix_ms => {
            abort_transfer_locked(context, begin, chunks, "transfer lease expired".to_string())
                .await
        }
        record => Ok(record),
    }
}

fn spawn_transfer_cleanup(context: &Arc<VoidContext>) {
    let context = Arc::downgrade(context);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(TRANSFER_CLEANUP_INTERVAL).await;
            let Some(context) = context.upgrade() else {
                break;
            };
            let transfer_ids: Vec<_> = context
                .active_transfers
                .lock()
                .await
                .iter()
                .copied()
                .collect();
            for transfer_id in transfer_ids {
                let _mutation = context.transfer_mutation.lock().await;
                let Ok(record) = load_transfer_record(&context, transfer_id).await else {
                    context.active_transfers.lock().await.remove(&transfer_id);
                    continue;
                };
                if let Err(error) = expire_transfer_locked(&context, record).await {
                    warn!(%transfer_id, %error, "failed to expire transfer");
                }
            }
        }
    });
}

async fn handle_transfer_begin(context: &VoidContext, begin: TransferBegin) -> VoidOut {
    let _mutation = context.transfer_mutation.lock().await;
    if begin.protocol_version != TRANSFER_PROTOCOL_VERSION {
        return VoidOut::Error {
            message: format!(
                "unsupported transfer protocol version {}, expected {TRANSFER_PROTOCOL_VERSION}",
                begin.protocol_version
            ),
        };
    }
    if begin.transfer_id.is_nil() {
        return VoidOut::Error {
            message: "transfer ID cannot be nil".to_string(),
        };
    }
    if begin.expected_chunks > MAX_MULTIPART_PARTS {
        return VoidOut::Error {
            message: format!(
                "expected_chunks must be <= {MAX_MULTIPART_PARTS}, got {}",
                begin.expected_chunks
            ),
        };
    }
    if (begin.expected_chunks == 0) != (begin.expected_len == 0) {
        return VoidOut::Error {
            message: "zero-length transfers must declare zero chunks, and vice versa".to_string(),
        };
    }
    if begin.deadline_unix_ms <= unix_time_ms() {
        return VoidOut::Error {
            message: "transfer deadline has already expired".to_string(),
        };
    }
    match context.store.get_object(begin.transfer_id).await {
        Ok(Some(_)) => {
            return VoidOut::Error {
                message: format!("transfer already exists: {}", begin.transfer_id),
            }
        }
        Ok(None) => {}
        Err(error) => {
            return VoidOut::Error {
                message: format!("failed to check transfer ID: {error}"),
            }
        }
    }

    let transfer_id = begin.transfer_id;
    let record = TransferRecord::InProgress {
        begin,
        chunks: Vec::new(),
        revision: 0,
    };
    if let Err(message) = persist_transfer_record(context, transfer_id, &record).await {
        return VoidOut::Error { message };
    }
    context.active_transfers.lock().await.insert(transfer_id);
    VoidOut::Transfer { record }
}

async fn handle_transfer_chunk(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    index: u32,
    data: Vec<u8>,
    hash: TransferHash,
) -> VoidOut {
    let _mutation = context.transfer_mutation.lock().await;
    let record = match load_transfer_record(context, transfer_id).await {
        Ok(record) => match expire_transfer_locked(context, record).await {
            Ok(record) => record,
            Err(message) => return VoidOut::Error { message },
        },
        Err(message) => return VoidOut::Error { message },
    };
    let TransferRecord::InProgress {
        begin,
        mut chunks,
        revision,
    } = record
    else {
        return VoidOut::Error {
            message: format!("transfer {transfer_id} is not in progress"),
        };
    };
    if index >= begin.expected_chunks {
        return VoidOut::Error {
            message: format!(
                "chunk index {index} is outside expected range 0..{}",
                begin.expected_chunks
            ),
        };
    }
    if chunks.iter().any(|chunk| chunk.index == index) {
        return VoidOut::Error {
            message: format!("duplicate chunk index {index} for transfer {transfer_id}"),
        };
    }
    let actual_hash = transfer_hash(&data);
    if actual_hash != hash {
        return VoidOut::Error {
            message: format!("chunk {index} hash mismatch for transfer {transfer_id}"),
        };
    }
    let prospective_len = chunks
        .iter()
        .map(|chunk| chunk.len)
        .sum::<u64>()
        .saturating_add(data.len() as u64);
    if prospective_len > begin.expected_len {
        return VoidOut::Error {
            message: format!(
                "transfer {transfer_id} exceeds declared length {}",
                begin.expected_len
            ),
        };
    }

    let object_id = uuid::Uuid::new_v4();
    match handle_transfer_chunk_upload(context, object_id, data).await {
        VoidOut::Uploaded { .. } => {}
        VoidOut::Error { message } => return VoidOut::Error { message },
        _ => {
            return VoidOut::Error {
                message: "unexpected response while storing transfer chunk".to_string(),
            }
        }
    }
    let chunk = TransferChunk {
        index,
        object_id,
        len: prospective_len - chunks.iter().map(|chunk| chunk.len).sum::<u64>(),
        hash,
    };
    chunks.push(chunk.clone());
    chunks.sort_by_key(|chunk| chunk.index);
    let updated = TransferRecord::InProgress {
        begin,
        chunks,
        revision: revision.saturating_add(1),
    };
    if let Err(message) = persist_transfer_record(context, transfer_id, &updated).await {
        let _ = delete_object(context, object_id).await;
        return VoidOut::Error { message };
    }
    VoidOut::TransferChunkStored { chunk }
}

async fn handle_transfer_inspect(context: &VoidContext, transfer_id: uuid::Uuid) -> VoidOut {
    let _mutation = context.transfer_mutation.lock().await;
    let record = match load_transfer_record(context, transfer_id).await {
        Ok(record) => match expire_transfer_locked(context, record).await {
            Ok(record) => record,
            Err(message) => return VoidOut::Error { message },
        },
        Err(message) => return VoidOut::Error { message },
    };
    VoidOut::Transfer { record }
}

async fn handle_transfer_commit(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    aggregate_hash: TransferHash,
) -> VoidOut {
    let _mutation = context.transfer_mutation.lock().await;
    let record = match load_transfer_record(context, transfer_id).await {
        Ok(record) => match expire_transfer_locked(context, record).await {
            Ok(record) => record,
            Err(message) => return VoidOut::Error { message },
        },
        Err(message) => return VoidOut::Error { message },
    };
    let TransferRecord::InProgress { begin, chunks, .. } = record else {
        return VoidOut::Error {
            message: format!("transfer {transfer_id} is not in progress"),
        };
    };
    if aggregate_hash != begin.expected_hash {
        return VoidOut::Error {
            message: format!("commit hash does not match declaration for transfer {transfer_id}"),
        };
    }
    if chunks.len() != begin.expected_chunks as usize
        || chunks
            .iter()
            .enumerate()
            .any(|(index, chunk)| chunk.index as usize != index)
    {
        return VoidOut::Error {
            message: format!(
                "transfer {transfer_id} has {} chunks, expected exactly 0..{}",
                chunks.len(),
                begin.expected_chunks
            ),
        };
    }
    let total_len = chunks.iter().map(|chunk| chunk.len).sum::<u64>();
    if total_len != begin.expected_len {
        return VoidOut::Error {
            message: format!(
                "transfer {transfer_id} length {total_len} does not match expected {}",
                begin.expected_len
            ),
        };
    }

    let mut aggregate = Sha256::new();
    for chunk in &chunks {
        let data = match try_download(context, chunk.object_id).await {
            DownloadAttempt::Found(data) => data,
            DownloadAttempt::Missing => {
                return VoidOut::Error {
                    message: format!("transfer chunk {} is missing", chunk.object_id),
                }
            }
            DownloadAttempt::Failed(message) => return VoidOut::Error { message },
        };
        if data.len() as u64 != chunk.len || transfer_hash(&data) != chunk.hash {
            return VoidOut::Error {
                message: format!(
                    "transfer chunk {} failed length or hash validation",
                    chunk.object_id
                ),
            };
        }
        aggregate.update(&data);
    }
    if TransferHash(aggregate.finalize().into()) != aggregate_hash {
        return VoidOut::Error {
            message: format!("aggregate hash mismatch for transfer {transfer_id}"),
        };
    }

    let manifest = TransferManifest {
        begin,
        chunks,
        committed_unix_ms: unix_time_ms(),
    };
    let committed = TransferRecord::Committed(manifest);
    if let Err(message) = persist_transfer_record(context, transfer_id, &committed).await {
        return VoidOut::Error { message };
    }
    context.active_transfers.lock().await.remove(&transfer_id);
    VoidOut::Transfer { record: committed }
}

async fn handle_transfer_abort(
    context: &VoidContext,
    transfer_id: uuid::Uuid,
    reason: String,
) -> VoidOut {
    let _mutation = context.transfer_mutation.lock().await;
    let record = match load_transfer_record(context, transfer_id).await {
        Ok(record) => record,
        Err(message) => return VoidOut::Error { message },
    };
    let TransferRecord::InProgress { begin, chunks, .. } = record else {
        return VoidOut::Error {
            message: format!("transfer {transfer_id} is not in progress"),
        };
    };
    match abort_transfer_locked(context, begin, chunks, reason).await {
        Ok(record) => VoidOut::Transfer { record },
        Err(message) => VoidOut::Error { message },
    }
}

async fn handle_download_range(
    context: &VoidContext,
    id: uuid::Uuid,
    offset: u64,
    length: u64,
) -> VoidOut {
    if length == 0 {
        return VoidOut::Downloaded { data: Vec::new() };
    }

    let record = match context.store.get_object(id).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            return VoidOut::Error {
                message: format!("object not found: {id}"),
            }
        }
        Err(e) => {
            error!(%id, error = %e, "failed to look up object");
            return VoidOut::Error {
                message: format!("lookup failed: {e}"),
            };
        }
    };

    match context
        .object_store
        .get_range(&record.key, offset, length)
        .await
    {
        Ok(data) => {
            debug!(%id, offset, bytes = data.len(), "downloaded range");
            VoidOut::Downloaded { data }
        }
        Err(e) => {
            warn!(%id, error = %e, "range download failed");
            VoidOut::Error {
                message: format!("not found: {e}"),
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
    #[error("transfer error: {0}")]
    Transfer(String),
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
