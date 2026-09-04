use std::{
    collections::{HashMap, HashSet},
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use paramecia_engine::{
    fuse_models, ErrorFeedbackMode, ErrorFeedbackParams, HyperParameterUpdate, ModelEngine,
    PerturbationMode, QuantConflictStrategy, ReplayParams, TrainingConfig,
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
    sync::{mpsc, oneshot, Mutex},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use black_hole_type::{
    ContractDescriptor, ContractId, ContractSide, DarkToken, DurabilityPolicy, InferenceInput,
    InferenceOutput, InferenceRequest, LogitEntry, MassArchitecture, MassErrorFeedbackConfig,
    MassIn, MassModelCapacity, MassModelConfig, MassModelParams, MassOut, MassPerturbationMode,
    ObjectId, OperationArtifactRef, OperationCapability, SequenceOutput, TransferBegin,
    TransferChunk, TransferHash, TransferRecord, TransferStreamFrame, TransferTicket,
    TunnelRequest, WorkerCapabilities, MASS_OPERATION_PROTOCOL_VERSION, TRANSFER_PROTOCOL_VERSION,
};
pub use paramecia_engine::KvCacheQuantization;

const DEFAULT_LISTEN_ADDR: &str = "[::1]:4433";
const MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MB
/// Chunk size for streaming void transfers (must fit within one frame).
const VOID_CHUNK_SIZE: usize = 16 * 1024 * 1024; // 16 MB
const STREAM_TRANSFER_LEASE: Duration = Duration::from_secs(5 * 60);
const DEFAULT_ENGINE_TOP_K: usize = 256;
const DEFAULT_ENGINE_TEMPERATURE: f64 = 0.7;
const DEFAULT_ENGINE_REPEAT_PENALTY: f32 = 1.0;
const DEFAULT_ENGINE_PRESENCE_PENALTY: f32 = 0.0;
const DEFAULT_INFERENCE_LIMIT: u32 = 256;
const DEFAULT_MAX_INSTANCES: usize = 1;
const DEFAULT_CHECKPOINT_TOKENIZER_FILE: &str = "tokenizer.json";

/// Architecture this mass binary's engine was compiled for.
///
/// paramecia selects model shapes at compile time via cargo features, so a
/// given build serves exactly one architecture (or none). Tunnel workers
/// advertise this to roots so model instances are only placed on compatible
/// engines, and it is used to reject starts this engine cannot serve.
pub const COMPILED_ARCHITECTURE: Option<MassArchitecture> = if cfg!(feature = "qwen35_0p8b") {
    Some(MassArchitecture::Qwen35_0p8b)
} else if cfg!(feature = "qwen35_2b") {
    Some(MassArchitecture::Qwen35_2b)
} else if cfg!(feature = "qwen35_4b") {
    Some(MassArchitecture::Qwen35_4b)
} else if cfg!(feature = "qwen35_9b") {
    Some(MassArchitecture::Qwen35_9b)
} else if cfg!(feature = "qwen35_27b") {
    Some(MassArchitecture::Qwen35_27b)
} else if cfg!(feature = "qwen38_27b") {
    Some(MassArchitecture::Qwen38_27b)
} else {
    None
};

fn local_worker_capabilities(
    operation: Option<&Arc<dyn OperationImplementation>>,
) -> WorkerCapabilities {
    let mut operations = vec![black_hole_spec::operation_capability::<
        black_hole_spec::QwenDarkInference,
    >()];
    if let Some(operation) = operation {
        let capability = operation.capability();
        if !operations.contains(&capability) {
            operations.push(capability);
        }
    }
    WorkerCapabilities {
        architectures: COMPILED_ARCHITECTURE.into_iter().collect(),
        operations,
    }
}
const DEFAULT_CHECKPOINT_TOKENIZER_DIR: &str = ".black-hole-sun/tokenizers";
const CHECKPOINT_CACHE_DIR: &str = "black-hole-mass/checkpoints";
const DEFAULT_TUNNEL_CONNECT_RETRY_MS: u64 = 200;
const MAX_TUNNEL_CONNECT_RETRY_MS: u64 = 25_600;
const TUNNEL_REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(5);
const RESIDUAL_UPDATE_UNSUPPORTED_FRAGMENT: &str =
    "restore_and_update_with_residual not supported for ";

/// Injected implementation of one tensor-operation contract.
///
/// Mass owns distributed identity, routing, persistence, and instance
/// lifecycle. Implementations own operation-specific state and the actual
/// forward computation. Inputs have already been validated against
/// `capability()` before `forward` is called; outputs are validated again by
/// Mass before publication.
#[async_trait::async_trait]
pub trait OperationImplementation: Send + Sync + 'static {
    fn capability(&self) -> OperationCapability;

    async fn start(&self, instance_id: Uuid) -> std::result::Result<(), String>;

    async fn forward(
        &self,
        instance_id: Uuid,
        input: Vec<u8>,
    ) -> std::result::Result<Vec<u8>, String>;

    async fn shutdown(&self, instance_id: Uuid) -> std::result::Result<(), String>;
}

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
///
/// Variant indices must stay aligned with black_hole_void::VoidIn — postcard
/// encodes enums by index, so variants this client never sends (DownloadWait)
/// are kept as placeholders.
#[derive(Debug, Serialize, Deserialize)]
enum VoidIn {
    Upload {
        data: Vec<u8>,
    },
    UploadWith {
        id: ObjectId,
        data: Vec<u8>,
    },
    Download {
        id: ObjectId,
    },
    /// Placeholder for wire compatibility with void (unused by this client).
    DownloadWait {
        id: ObjectId,
        timeout_ms: u64,
    },
    UploadBegin {
        id: Option<ObjectId>,
        total_size: u64,
    },
    UploadPart {
        id: ObjectId,
        part_number: u32,
        data: Vec<u8>,
    },
    UploadFinish {
        id: ObjectId,
        part_count: u32,
    },
    DownloadRange {
        id: ObjectId,
        offset: u64,
        length: u64,
    },
    TransferBegin {
        begin: TransferBegin,
    },
    TransferChunk {
        transfer_id: ObjectId,
        index: u32,
        data: Vec<u8>,
        hash: TransferHash,
    },
    TransferInspect {
        transfer_id: ObjectId,
    },
    TransferCommit {
        transfer_id: ObjectId,
        aggregate_hash: TransferHash,
    },
    TransferAbort {
        transfer_id: ObjectId,
        reason: String,
    },
    TransferStreamUpload {
        begin: TransferBegin,
        authorization: [u8; 32],
    },
    TransferStreamDownload {
        transfer_id: ObjectId,
        authorization: [u8; 32],
    },
}

/// Wire response from the void service.
///
/// Variant indices must stay aligned with black_hole_void::VoidOut.
#[derive(Debug, Serialize, Deserialize)]
enum VoidOut {
    Uploaded {
        id: ObjectId,
    },
    Downloaded {
        data: Vec<u8>,
    },
    /// Placeholder for wire compatibility with void (unused by this client).
    TimedOut {
        id: ObjectId,
    },
    Ack,
    Error {
        message: String,
    },
    Transfer {
        record: TransferRecord,
    },
    TransferChunkStored {
        chunk: TransferChunk,
    },
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

enum PendingTransferUpload {
    Quic {
        send: quinn::SendStream,
        recv: quinn::RecvStream,
    },
    Tcp(TcpStream),
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

    /// Resolve an operation input. Stream references use their live ticket
    /// first and fall back to the committed Void manifest if the live channel
    /// is interrupted. A full-tensor operation receives bytes only after the
    /// stream's commit frame, so its output cannot precede durable input.
    pub async fn download_artifact(&self, reference: OperationArtifactRef) -> Result<Vec<u8>> {
        match reference {
            OperationArtifactRef::Committed(id) => self.download(id).await,
            OperationArtifactRef::Transfer(id) => self.download_committed_transfer(id).await,
            OperationArtifactRef::Stream {
                ticket_id,
                fallback_transfer_id,
            } => {
                let ticket = self.download_stream_ticket(ticket_id).await;
                let durability = ticket.as_ref().ok().map(|ticket| ticket.durability);
                let live = match ticket {
                    Ok(ticket) => {
                        if ticket.eventual_void_id != fallback_transfer_id
                            || ticket.transfer_id != fallback_transfer_id
                        {
                            Err(ServerError::VoidError(
                                "stream ticket does not match its durable fallback".into(),
                            ))
                        } else {
                            self.receive_stream(&ticket).await
                        }
                    }
                    Err(error) => Err(error),
                };
                match live {
                    Ok(bytes) => Ok(bytes),
                    Err(error) if durability == Some(DurabilityPolicy::Ephemeral) => Err(error),
                    Err(live_error) => {
                        debug!(
                            %ticket_id,
                            %fallback_transfer_id,
                            error = %live_error,
                            "live artifact stream failed; resolving committed Void fallback"
                        );
                        self.download_committed_transfer(fallback_transfer_id)
                            .await
                            .map_err(|fallback_error| {
                                ServerError::VoidError(format!(
                                    "live stream failed ({live_error}); durable fallback failed ({fallback_error})"
                                ))
                            })
                    }
                }
            }
        }
    }

    async fn download_stream_ticket(&self, ticket_id: ObjectId) -> Result<TransferTicket> {
        let bytes = self.download(ticket_id).await?;
        let ticket: TransferTicket = from_bytes(&bytes).map_err(ServerError::DecodeFrame)?;
        validate_transfer_ticket(&ticket)?;
        Ok(ticket)
    }

    async fn receive_stream(&self, ticket: &TransferTicket) -> Result<Vec<u8>> {
        let source: SocketAddr = ticket.source.parse().map_err(|error| {
            ServerError::VoidError(format!("invalid transfer source in ticket: {error}"))
        })?;
        let bytes = match &self.transport {
            VoidClientTransport::Quic { endpoint, .. } => {
                let connection = endpoint
                    .connect(source, &source.ip().to_string())
                    .map_err(|error| ServerError::VoidConnect(error.to_string()))?
                    .await
                    .map_err(|error| ServerError::VoidConnect(error.to_string()))?;
                let (mut send, mut recv) = connection
                    .open_bi()
                    .await
                    .map_err(|error| ServerError::VoidStream(error.to_string()))?;
                write_frame_quic(
                    &mut send,
                    &VoidIn::TransferStreamDownload {
                        transfer_id: ticket.transfer_id,
                        authorization: ticket.authorization,
                    },
                )
                .await?;
                receive_transfer_frames_quic(&mut recv, ticket).await?
            }
            VoidClientTransport::Tcp { .. } => {
                let mut stream = TcpStream::connect(source)
                    .await
                    .map_err(|error| ServerError::VoidTcpConnect(error.to_string()))?;
                write_frame_io(
                    &mut stream,
                    &VoidIn::TransferStreamDownload {
                        transfer_id: ticket.transfer_id,
                        authorization: ticket.authorization,
                    },
                )
                .await?;
                receive_transfer_frames_io(&mut stream, ticket).await?
            }
        };
        validate_received_transfer(ticket, bytes)
    }

    /// Publish a validated operation result as a live stream and tee every
    /// frame into Void. The transfer begin and ticket are durable before this
    /// returns; chunk upload and commit continue in the background.
    pub async fn publish_artifact(
        &self,
        descriptor: ContractDescriptor,
        side: ContractSide,
        data: Vec<u8>,
    ) -> Result<OperationArtifactRef> {
        let validated = black_hole_spec::validate_artifact(&descriptor, side, &data)
            .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
        let tensor_header = black_hole_spec::tensor_stream_header(&data)
            .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
        let expected_len = data.len() as u64;
        let expected_hash = transfer_hash(&data);
        let expected_chunks = if data.is_empty() {
            0
        } else {
            u32::try_from(data.len().div_ceil(VOID_CHUNK_SIZE)).map_err(|_| {
                ServerError::VoidError("artifact requires too many transfer chunks".into())
            })?
        };
        let transfer_id = ObjectId::new_v4();
        let authorization = random_transfer_authorization();
        let deadline_unix_ms = unix_time_ms().saturating_add(
            STREAM_TRANSFER_LEASE
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        let begin = TransferBegin {
            protocol_version: TRANSFER_PROTOCOL_VERSION,
            transfer_id,
            envelope: validated.envelope.clone(),
            tensor_header: tensor_header.clone(),
            expected_chunks,
            expected_len,
            expected_hash,
            deadline_unix_ms,
            authorization_hash: transfer_hash(&authorization),
        };
        let ticket = TransferTicket {
            descriptor,
            envelope: validated.envelope,
            tensor_header,
            transfer_id,
            source: self.source_authority(),
            authorization,
            expected_len,
            expected_hash,
            deadline_unix_ms,
            durability: DurabilityPolicy::ReplayRequired,
            eventual_void_id: transfer_id,
        };

        let mut upload = self.open_transfer_upload(begin, authorization).await?;
        let ticket_bytes = to_allocvec(&ticket).map_err(ServerError::EncodeFrame)?;
        let ticket_id = match self.upload(ticket_bytes).await {
            Ok(ticket_id) => ticket_id,
            Err(error) => {
                let _ =
                    abort_pending_upload(&mut upload, "failed to persist transfer ticket").await;
                return Err(error);
            }
        };
        tokio::spawn(async move {
            if let Err(error) = finish_transfer_upload(upload, data, expected_hash).await {
                warn!(%transfer_id, %error, "background artifact stream failed");
            }
        });
        Ok(OperationArtifactRef::Stream {
            ticket_id,
            fallback_transfer_id: transfer_id,
        })
    }

    fn source_authority(&self) -> String {
        match &self.transport {
            VoidClientTransport::Quic { remote_addr, .. }
            | VoidClientTransport::Tcp { remote_addr } => remote_addr.to_string(),
        }
    }

    async fn open_transfer_upload(
        &self,
        begin: TransferBegin,
        authorization: [u8; 32],
    ) -> Result<PendingTransferUpload> {
        let request = VoidIn::TransferStreamUpload {
            begin,
            authorization,
        };
        match &self.transport {
            VoidClientTransport::Quic {
                endpoint,
                remote_addr,
            } => {
                let connection = endpoint
                    .connect(*remote_addr, &remote_addr.ip().to_string())
                    .map_err(|error| ServerError::VoidConnect(error.to_string()))?
                    .await
                    .map_err(|error| ServerError::VoidConnect(error.to_string()))?;
                let (mut send, mut recv) = connection
                    .open_bi()
                    .await
                    .map_err(|error| ServerError::VoidStream(error.to_string()))?;
                write_frame_quic(&mut send, &request).await?;
                expect_transfer_started(read_frame_quic(&mut recv).await?)?;
                Ok(PendingTransferUpload::Quic { send, recv })
            }
            VoidClientTransport::Tcp { remote_addr } => {
                let mut stream = TcpStream::connect(*remote_addr)
                    .await
                    .map_err(|error| ServerError::VoidTcpConnect(error.to_string()))?;
                write_frame_io(&mut stream, &request).await?;
                expect_transfer_started(read_frame_io(&mut stream).await?)?;
                Ok(PendingTransferUpload::Tcp(stream))
            }
        }
    }

    async fn download_committed_transfer(&self, transfer_id: ObjectId) -> Result<Vec<u8>> {
        let mut staged = HashMap::<u32, Vec<u8>>::new();
        loop {
            let record = match self.call(VoidIn::TransferInspect { transfer_id }).await? {
                VoidOut::Transfer { record } => record,
                VoidOut::Error { message } => return Err(ServerError::VoidError(message)),
                _ => {
                    return Err(ServerError::VoidError(
                        "unexpected void response for transfer inspect".into(),
                    ))
                }
            };
            let (begin, chunks, committed) = match record {
                TransferRecord::InProgress { begin, chunks, .. } => (begin, chunks, false),
                TransferRecord::Committed(manifest) => (manifest.begin, manifest.chunks, true),
                TransferRecord::Aborted(abort) => {
                    return Err(ServerError::VoidError(format!(
                        "transfer {transfer_id} aborted: {}",
                        abort.reason
                    )))
                }
            };
            for chunk in &chunks {
                if staged.contains_key(&chunk.index) {
                    continue;
                }
                let data = self.download(chunk.object_id).await?;
                if data.len() as u64 != chunk.len
                    || TransferHash(Sha256::digest(&data).into()) != chunk.hash
                {
                    return Err(ServerError::VoidError(format!(
                        "transfer {transfer_id} chunk {} failed validation",
                        chunk.index
                    )));
                }
                staged.insert(chunk.index, data);
            }
            if committed {
                if chunks.len() != begin.expected_chunks as usize {
                    return Err(ServerError::VoidError(format!(
                        "transfer {transfer_id} has an invalid manifest"
                    )));
                }
                let mut output = Vec::new();
                let mut aggregate = Sha256::new();
                for index in 0..begin.expected_chunks {
                    let data = staged.remove(&index).ok_or_else(|| {
                        ServerError::VoidError(format!(
                            "transfer {transfer_id} is missing chunk {index}"
                        ))
                    })?;
                    aggregate.update(&data);
                    output.extend_from_slice(&data);
                }
                if output.len() as u64 != begin.expected_len
                    || TransferHash(aggregate.finalize().into()) != begin.expected_hash
                {
                    return Err(ServerError::VoidError(format!(
                        "transfer {transfer_id} failed aggregate validation"
                    )));
                }
                return Ok(output);
            }
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            if now_ms >= begin.deadline_unix_ms {
                return Err(ServerError::VoidError(format!(
                    "transfer {transfer_id} lease expired before commit"
                )));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
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

    /// Upload a local file to void. Files that fit in one frame use the
    /// single-shot upload; larger files are streamed as a chunked multipart
    /// upload. Returns the assigned object ID.
    pub async fn upload_file(&self, path: &Path) -> Result<ObjectId> {
        let size = fs::metadata(path)
            .map_err(|source| ServerError::FileMetadata {
                path: path.to_path_buf(),
                source,
            })?
            .len();

        if size <= MAX_FRAME_SIZE as u64 {
            let data = fs::read(path).map_err(|source| ServerError::ReadFile {
                path: path.to_path_buf(),
                source,
            })?;
            return self.upload(data).await;
        }

        let id = match self
            .call(VoidIn::UploadBegin {
                id: None,
                total_size: size,
            })
            .await?
        {
            VoidOut::Uploaded { id } => id,
            VoidOut::Error { message } => return Err(ServerError::VoidError(message)),
            _ => {
                return Err(ServerError::VoidError(
                    "unexpected void response for upload begin".into(),
                ))
            }
        };

        let mut file =
            tokio::fs::File::open(path)
                .await
                .map_err(|source| ServerError::OpenFile {
                    path: path.to_path_buf(),
                    source,
                })?;
        let mut part_number: u32 = 1;
        loop {
            let mut buffer = vec![0u8; VOID_CHUNK_SIZE];
            let read = file
                .read(&mut buffer)
                .await
                .map_err(|source| ServerError::ReadFile {
                    path: path.to_path_buf(),
                    source,
                })?;
            if read == 0 {
                break;
            }
            buffer.truncate(read);
            match self
                .call(VoidIn::UploadPart {
                    id,
                    part_number,
                    data: buffer,
                })
                .await?
            {
                VoidOut::Ack => {}
                VoidOut::Error { message } => return Err(ServerError::VoidError(message)),
                _ => {
                    return Err(ServerError::VoidError(
                        "unexpected void response for upload part".into(),
                    ))
                }
            }
            part_number += 1;
        }

        let part_count = part_number - 1;
        match self.call(VoidIn::UploadFinish { id, part_count }).await? {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(ServerError::VoidError(message)),
            _ => Err(ServerError::VoidError(
                "unexpected void response for upload finish".into(),
            )),
        }
    }

    /// Download an object from void directly to a local file using ranged
    /// reads, so arbitrarily large objects never need to fit in one frame.
    /// Returns the number of bytes written.
    pub async fn download_to_file(&self, id: ObjectId, path: &Path) -> Result<u64> {
        let mut file =
            tokio::fs::File::create(path)
                .await
                .map_err(|source| ServerError::OpenFile {
                    path: path.to_path_buf(),
                    source,
                })?;
        let mut offset: u64 = 0;
        loop {
            let data = match self
                .call(VoidIn::DownloadRange {
                    id,
                    offset,
                    length: VOID_CHUNK_SIZE as u64,
                })
                .await?
            {
                VoidOut::Downloaded { data } => data,
                VoidOut::Error { message } => return Err(ServerError::VoidError(message)),
                _ => {
                    return Err(ServerError::VoidError(
                        "unexpected void response for download range".into(),
                    ))
                }
            };
            if data.is_empty() {
                break;
            }
            file.write_all(&data)
                .await
                .map_err(|source| ServerError::WriteFile {
                    path: path.to_path_buf(),
                    source,
                })?;
            offset += data.len() as u64;
            if data.len() < VOID_CHUNK_SIZE {
                break;
            }
        }
        Ok(offset)
    }
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

fn random_transfer_authorization() -> [u8; 32] {
    let left = ObjectId::new_v4();
    let right = ObjectId::new_v4();
    let mut authorization = [0; 32];
    authorization[..16].copy_from_slice(left.as_bytes());
    authorization[16..].copy_from_slice(right.as_bytes());
    authorization
}

fn expect_transfer_started(response: VoidOut) -> Result<()> {
    match response {
        VoidOut::Transfer {
            record: TransferRecord::InProgress { .. },
        } => Ok(()),
        VoidOut::Error { message } => Err(ServerError::VoidError(message)),
        _ => Err(ServerError::VoidError(
            "unexpected response while opening transfer stream".into(),
        )),
    }
}

async fn write_pending_upload(
    upload: &mut PendingTransferUpload,
    frame: &TransferStreamFrame,
) -> Result<()> {
    match upload {
        PendingTransferUpload::Quic { send, .. } => write_frame_quic(send, frame).await,
        PendingTransferUpload::Tcp(stream) => write_frame_io(stream, frame).await,
    }
}

async fn read_pending_upload(upload: &mut PendingTransferUpload) -> Result<VoidOut> {
    match upload {
        PendingTransferUpload::Quic { recv, .. } => read_frame_quic(recv).await,
        PendingTransferUpload::Tcp(stream) => read_frame_io(stream).await,
    }
}

async fn abort_pending_upload(
    upload: &mut PendingTransferUpload,
    reason: impl Into<String>,
) -> Result<()> {
    write_pending_upload(
        upload,
        &TransferStreamFrame::Abort {
            reason: reason.into(),
        },
    )
    .await?;
    match read_pending_upload(upload).await? {
        VoidOut::Transfer {
            record: TransferRecord::Aborted(_),
        } => Ok(()),
        VoidOut::Error { message } => Err(ServerError::VoidError(message)),
        _ => Err(ServerError::VoidError(
            "unexpected response while aborting transfer stream".into(),
        )),
    }
}

async fn finish_transfer_upload(
    mut upload: PendingTransferUpload,
    data: Vec<u8>,
    aggregate_hash: TransferHash,
) -> Result<()> {
    for (index, chunk) in data.chunks(VOID_CHUNK_SIZE).enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| ServerError::VoidError("too many stream chunks".into()))?;
        let data = chunk.to_vec();
        let hash = transfer_hash(&data);
        write_pending_upload(
            &mut upload,
            &TransferStreamFrame::Chunk { index, data, hash },
        )
        .await?;
    }
    write_pending_upload(&mut upload, &TransferStreamFrame::Commit { aggregate_hash }).await?;
    match read_pending_upload(&mut upload).await? {
        VoidOut::Transfer {
            record: TransferRecord::Committed(_),
        } => Ok(()),
        VoidOut::Error { message } => Err(ServerError::VoidError(message)),
        _ => Err(ServerError::VoidError(
            "unexpected response while committing transfer stream".into(),
        )),
    }
}

fn validate_transfer_ticket(ticket: &TransferTicket) -> Result<()> {
    if ticket.descriptor.id != ticket.envelope.contract_id
        || ticket.descriptor.version != ticket.envelope.contract_version
        || black_hole_spec::descriptor_hash(&ticket.descriptor) != ticket.envelope.contract_hash
    {
        return Err(ServerError::VoidError(
            "transfer ticket descriptor does not match its tensor envelope".into(),
        ));
    }
    if ticket.eventual_void_id != ticket.transfer_id {
        return Err(ServerError::VoidError(
            "transfer ticket durable object does not match its transfer ID".into(),
        ));
    }
    let declared_len = black_hole_spec::validate_tensor_stream_header(
        &ticket.descriptor,
        ticket.envelope.side,
        &ticket.tensor_header,
    )
    .map_err(|error| ServerError::VoidError(format!("invalid tensor stream header: {error}")))?;
    if declared_len != ticket.expected_len {
        return Err(ServerError::VoidError(format!(
            "tensor stream header declares {declared_len} bytes, but ticket declares {}",
            ticket.expected_len
        )));
    }
    Ok(())
}

fn validate_stream_begin(begin: &TransferBegin, ticket: &TransferTicket) -> Result<()> {
    if begin.transfer_id != ticket.transfer_id
        || begin.envelope != ticket.envelope
        || begin.tensor_header != ticket.tensor_header
        || begin.expected_len != ticket.expected_len
        || begin.expected_hash != ticket.expected_hash
        || begin.deadline_unix_ms != ticket.deadline_unix_ms
        || begin.authorization_hash != transfer_hash(&ticket.authorization)
    {
        return Err(ServerError::VoidError(
            "live stream begin frame does not match its ticket".into(),
        ));
    }
    Ok(())
}

fn apply_transfer_frame(
    frame: TransferStreamFrame,
    ticket: &TransferTicket,
    begin: &mut Option<TransferBegin>,
    next_index: &mut u32,
    aggregate: &mut Sha256,
    output: &mut Vec<u8>,
) -> Result<bool> {
    match frame {
        TransferStreamFrame::Begin(received) => {
            if begin.is_some() {
                return Err(ServerError::VoidError(
                    "duplicate begin frame on transfer stream".into(),
                ));
            }
            validate_stream_begin(&received, ticket)?;
            *begin = Some(received);
            Ok(false)
        }
        TransferStreamFrame::Chunk { index, data, hash } => {
            if begin.is_none() {
                return Err(ServerError::VoidError(
                    "transfer chunk arrived before begin frame".into(),
                ));
            }
            if index != *next_index {
                return Err(ServerError::VoidError(format!(
                    "transfer stream chunk {index} arrived out of order; expected {}",
                    *next_index
                )));
            }
            if transfer_hash(&data) != hash {
                return Err(ServerError::VoidError(format!(
                    "transfer stream chunk {index} failed validation"
                )));
            }
            let prefix_start = output.len().min(ticket.tensor_header.len());
            let prefix_end = output
                .len()
                .saturating_add(data.len())
                .min(ticket.tensor_header.len());
            if prefix_start < prefix_end
                && data[..prefix_end - prefix_start]
                    != ticket.tensor_header[prefix_start..prefix_end]
            {
                return Err(ServerError::VoidError(format!(
                    "transfer stream chunk {index} does not match the authenticated tensor header"
                )));
            }
            *next_index = next_index.saturating_add(1);
            aggregate.update(&data);
            output.extend_from_slice(&data);
            Ok(false)
        }
        TransferStreamFrame::Commit { aggregate_hash } => {
            let begin = begin.as_ref().ok_or_else(|| {
                ServerError::VoidError("transfer committed before begin frame".into())
            })?;
            let actual_hash = TransferHash(aggregate.clone().finalize().into());
            if *next_index != begin.expected_chunks
                || output.len() as u64 != begin.expected_len
                || aggregate_hash != begin.expected_hash
                || actual_hash != aggregate_hash
            {
                return Err(ServerError::VoidError(
                    "committed transfer stream failed aggregate validation".into(),
                ));
            }
            Ok(true)
        }
        TransferStreamFrame::Abort { reason } => Err(ServerError::VoidError(format!(
            "transfer aborted: {reason}"
        ))),
    }
}

async fn receive_transfer_frames_quic(
    recv: &mut quinn::RecvStream,
    ticket: &TransferTicket,
) -> Result<Vec<u8>> {
    let mut begin = None;
    let mut next_index = 0;
    let mut aggregate = Sha256::new();
    let mut output = Vec::new();
    loop {
        let frame = read_frame_quic(recv).await?;
        if apply_transfer_frame(
            frame,
            ticket,
            &mut begin,
            &mut next_index,
            &mut aggregate,
            &mut output,
        )? {
            return Ok(output);
        }
    }
}

async fn receive_transfer_frames_io<S>(stream: &mut S, ticket: &TransferTicket) -> Result<Vec<u8>>
where
    S: AsyncRead + Unpin,
{
    let mut begin = None;
    let mut next_index = 0;
    let mut aggregate = Sha256::new();
    let mut output = Vec::new();
    loop {
        let frame = read_frame_io(stream).await?;
        if apply_transfer_frame(
            frame,
            ticket,
            &mut begin,
            &mut next_index,
            &mut aggregate,
            &mut output,
        )? {
            return Ok(output);
        }
    }
}

fn validate_received_transfer(ticket: &TransferTicket, bytes: Vec<u8>) -> Result<Vec<u8>> {
    if bytes.len() as u64 != ticket.expected_len
        || transfer_hash(&bytes) != ticket.expected_hash
        || !bytes.starts_with(&ticket.tensor_header)
    {
        return Err(ServerError::VoidError(
            "received tensor does not match its transfer ticket".into(),
        ));
    }
    black_hole_spec::validate_artifact(&ticket.descriptor, ticket.envelope.side, &bytes)
        .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
    Ok(bytes)
}

enum MassRpcClientInner {
    Quic {
        endpoint: quinn::Endpoint,
        remote_addr: SocketAddr,
    },
    Tcp {
        remote_addr: SocketAddr,
    },
}

/// A client connection to another mass server.
struct MassRpcClient {
    inner: MassRpcClientInner,
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
    _client: MassRpcClient,
    connection: TunnelConnectionHandle,
}

enum RpcConnection {
    Quic(quinn::Connection),
    Tcp(TcpStream),
}

#[derive(Debug, Serialize, Deserialize)]
enum TunnelTcpEnvelope {
    Request { request_id: u64, request: MassIn },
    Response { request_id: u64, response: MassOut },
}

#[derive(Debug)]
struct TunnelTcpRequest {
    request_id: u64,
    request: MassIn,
}

struct TcpTunnelSession {
    outbound: mpsc::Sender<TunnelTcpEnvelope>,
    inbound: Mutex<mpsc::Receiver<TunnelTcpRequest>>,
    pending: Mutex<HashMap<u64, oneshot::Sender<MassOut>>>,
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
            let _ = waiter.send(MassOut::Error {
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

    async fn call(&self, request: MassIn) -> Result<MassOut> {
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

    async fn send_response(&self, request_id: u64, response: MassOut) -> Result<()> {
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

impl MassRpcClient {
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
                MassRpcClientInner::Quic {
                    endpoint,
                    remote_addr: addr,
                }
            }
            TransportMode::Tcp => MassRpcClientInner::Tcp { remote_addr: addr },
        };
        Ok(Self { inner })
    }

    async fn establish_connection(&self) -> Result<RpcConnection> {
        match &self.inner {
            MassRpcClientInner::Quic {
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
            MassRpcClientInner::Tcp { remote_addr } => {
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
    req: MassIn,
) -> Result<MassOut> {
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

/// Compatibility adapter for the legacy Qwen/QuZO protocol.
///
/// The generic host never sees paramecia inputs or dark tokens. This adapter
/// is the boundary that owns those conversions while the old protocol remains
/// available during migration.
struct QwenOperationAdapter;

impl QwenOperationAdapter {
    fn dark_input(tokens: Vec<DarkToken>) -> paramecia_engine::ModelInput {
        paramecia_engine::ModelInput::Soft(
            tokens
                .into_iter()
                .map(|token| paramecia_engine::SoftToken {
                    predicted: token.predicted,
                    dark_knowledge: token
                        .dark_knowledge
                        .into_iter()
                        .map(|entry| paramecia_engine::LogitEntry {
                            token_id: entry.token_id,
                            log_prob: entry.log_prob,
                        })
                        .collect(),
                })
                .collect(),
        )
    }

    fn model_input(input: InferenceInput) -> paramecia_engine::ModelInput {
        match input {
            InferenceInput::Text(text) => paramecia_engine::ModelInput::Text(text),
            InferenceInput::Tokens(tokens) => paramecia_engine::ModelInput::Tokens(tokens),
            InferenceInput::Dark(tokens) => Self::dark_input(tokens),
        }
    }

    fn output(results: Vec<Vec<paramecia_engine::Predicted>>) -> InferenceOutput {
        InferenceOutput {
            results: results
                .into_iter()
                .map(|predictions| {
                    SequenceOutput(
                        predictions
                            .into_iter()
                            .map(|prediction| DarkToken {
                                predicted: prediction.token_id,
                                dark_knowledge: prediction
                                    .top_k
                                    .into_iter()
                                    .map(|entry| LogitEntry {
                                        token_id: entry.token_id,
                                        log_prob: entry.log_prob,
                                    })
                                    .collect(),
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MassState {
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

struct MassSession {
    state: MassState,
    running: bool,
    frozen: bool,
    optimize_steps: u32,
}

impl MassSession {
    fn new(frozen: bool) -> Self {
        Self {
            state: MassState::Idle,
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

struct MassInstance {
    engine: ModelEngine,
    runtime_config: ModelRuntimeConfig,
    oscillation: Option<FrozenOscillation>,
    checkpoint_path: Option<PathBuf>,
    session: tokio::sync::Mutex<MassSession>,
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
    training_perturbation_mode: MassPerturbationMode,
    training_error_feedback: MassErrorFeedbackConfig,
}

struct ResolvedModelSource {
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    checkpoint_path: Option<PathBuf>,
}

enum ModelSlot {
    Starting,
    Running(Arc<MassInstance>),
    ShuttingDown,
}

enum MassMode {
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
    capabilities: WorkerCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RouteTarget {
    Local,
    Worker(Uuid),
}

#[derive(Debug, Clone)]
struct TunnelWorker {
    token: Uuid,
    worker_id: Uuid,
    max_instances: Option<usize>,
    /// Capabilities advertised by the worker at registration. Empty on
    /// legacy workers that predate capability advertising.
    capabilities: WorkerCapabilities,
}

struct MassContext {
    model_path: PathBuf,
    transport_mode: TransportMode,
    void_client: Option<Arc<VoidClient>>,
    defaults: MassServerDefaults,
    frozen: bool,
    max_instances: Option<usize>,
    mode: MassMode,
    start_dispatch: tokio::sync::Mutex<()>,
    routes: tokio::sync::RwLock<HashMap<Uuid, RouteTarget>>,
    workers: tokio::sync::RwLock<HashMap<Uuid, TunnelWorker>>,
    worker_connections: tokio::sync::RwLock<HashMap<Uuid, TunnelConnectionHandle>>,
    instances: tokio::sync::RwLock<HashMap<Uuid, ModelSlot>>,
    operation: Option<Arc<dyn OperationImplementation>>,
    operation_instances: tokio::sync::RwLock<HashSet<Uuid>>,
    /// Architecture requirement of each routed model instance (None when the
    /// start carried no requirement), tracked for per-architecture capacity.
    instance_requirements: tokio::sync::RwLock<HashMap<Uuid, Option<MassArchitecture>>>,
}

#[derive(Clone)]
struct MassServerDefaults {
    top_k: usize,
    temperature: f64,
    top_p: Option<f64>,
    kv_cache_quant: KvCacheQuantization,
    repeat_penalty: f32,
    presence_penalty: f32,
    inference_limit: u32,
    training_config: TrainingConfig,
    training_error_feedback: MassErrorFeedbackConfig,
}

impl Default for MassServerDefaults {
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
            training_error_feedback: MassErrorFeedbackConfig::Off,
        }
    }
}

impl MassServerDefaults {
    fn with_overrides(&self, model_config: Option<&MassModelConfig>) -> Self {
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
            if let Some(training_perturbation_mode) = model_config.training_perturbation_mode {
                resolved.training_config.perturbation_mode =
                    to_engine_perturbation_mode(training_perturbation_mode);
            }
            if let Some(training_error_feedback) = model_config.training_error_feedback {
                resolved.training_error_feedback = training_error_feedback;
            }
        }
        resolved
    }
}

fn to_engine_error_feedback(config: MassErrorFeedbackConfig) -> ErrorFeedbackMode {
    match config {
        MassErrorFeedbackConfig::Off => ErrorFeedbackMode::None,
        MassErrorFeedbackConfig::Persistent { decay, gain } => {
            ErrorFeedbackMode::Persistent(ErrorFeedbackParams { decay, gain })
        }
        MassErrorFeedbackConfig::Replay { steps, decay, gain } => {
            ErrorFeedbackMode::Replay(ReplayParams { steps, decay, gain })
        }
    }
}

fn to_engine_perturbation_mode(mode: MassPerturbationMode) -> PerturbationMode {
    match mode {
        MassPerturbationMode::Weight => PerturbationMode::Weight,
        MassPerturbationMode::LowRank(rank) => PerturbationMode::LowRank(rank),
    }
}

fn to_mass_perturbation_mode(mode: PerturbationMode) -> MassPerturbationMode {
    match mode {
        PerturbationMode::Weight => MassPerturbationMode::Weight,
        PerturbationMode::LowRank(rank) => MassPerturbationMode::LowRank(rank),
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
    defaults: MassServerDefaults,
    operation: Option<Arc<dyn OperationImplementation>>,
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
            defaults: MassServerDefaults::default(),
            operation: None,
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

    /// Register this mass as a tunnel worker of the parent at `addr`.
    pub fn tunnel(mut self, addr: SocketAddr) -> Self {
        self.tunnel = Some(addr);
        self
    }

    /// Configure how long tunnel workers retry parent registration before failing.
    pub fn tunnel_connect_deadline(mut self, deadline: Duration) -> Self {
        self.tunnel_connect_deadline = Some(deadline);
        self
    }

    /// Limit concurrent model instances handled by this mass.
    pub fn max_instances(mut self, limit: usize) -> Self {
        self.max_instances = Some(limit);
        self
    }

    /// Inject a generic tensor operation hosted alongside the legacy Qwen
    /// compatibility path.
    pub fn operation(mut self, operation: impl OperationImplementation) -> Self {
        self.operation = Some(Arc::new(operation));
        self
    }

    /// Inject a shared operation implementation. Useful for test harnesses
    /// and applications that also retain a handle to operation state.
    pub fn operation_shared(mut self, operation: Arc<dyn OperationImplementation>) -> Self {
        self.operation = Some(operation);
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

    /// Configure the default QuZO perturbation direction mode.
    pub fn training_perturbation_mode(mut self, mode: MassPerturbationMode) -> Self {
        self.defaults.training_config.perturbation_mode = to_engine_perturbation_mode(mode);
        self
    }

    /// Configure the full default training config used for new model instances.
    pub fn training_config(mut self, config: TrainingConfig) -> Self {
        self.defaults.training_config = config;
        self
    }

    /// Build the void client, endpoint and shared server context.
    async fn setup(self) -> Result<(MassListener, SocketAddr, Arc<MassContext>)> {
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
                MassListener::Quic(endpoint)
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
                MassListener::Tcp(listener)
            }
        };

        let local_addr = listener.local_addr().map_err(ServerError::LocalAddr)?;
        info!(%local_addr, "listening");

        let local_capabilities = local_worker_capabilities(self.operation.as_ref());
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
                local_capabilities.clone(),
            )
            .await?;
            info!(
                %parent_addr,
                %local_addr,
                %worker_id,
                token = %tunnel_token,
                "tunnel worker registered"
            );
            MassMode::Worker(Arc::new(WorkerModeState {
                parent_addr,
                worker_id,
                transport_mode: self.transport_mode,
                tunnel_token: tokio::sync::RwLock::new(tunnel_token),
                parent_session: tokio::sync::RwLock::new(Some(parent_session)),
                tunnel_connect_deadline: self.tunnel_connect_deadline,
                capabilities: local_capabilities,
            }))
        } else {
            MassMode::Root
        };

        let context = Arc::new(MassContext {
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
            operation: self.operation,
            operation_instances: tokio::sync::RwLock::new(HashSet::new()),
            instance_requirements: tokio::sync::RwLock::new(HashMap::new()),
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
        listener: MassListener,
        context: Arc<MassContext>,
        stateless_retry: bool,
    ) -> Result<()> {
        let (_shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        if matches!(&context.mode, MassMode::Worker(_)) {
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
        listener: MassListener,
        context: Arc<MassContext>,
        stateless_retry: bool,
        mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
        match listener {
            MassListener::Quic(endpoint) => loop {
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
            MassListener::Tcp(listener) => loop {
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

enum MassListener {
    Quic(quinn::Endpoint),
    Tcp(TcpListener),
}

impl MassListener {
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
    context: Arc<MassContext>,
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

async fn parent_tunnel_stream_loop(context: Arc<MassContext>) {
    let worker_mode = match &context.mode {
        MassMode::Worker(worker_mode) => Arc::clone(worker_mode),
        MassMode::Root => return,
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

async fn parent_tunnel_tcp_session_loop(context: Arc<MassContext>) {
    let worker_mode = match &context.mode {
        MassMode::Worker(worker_mode) => Arc::clone(worker_mode),
        MassMode::Root => return,
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
    context: Arc<MassContext>,
) {
    let out = match handle_request(request.request, &context, None).await {
        Ok(out) => out,
        Err(error) => {
            warn!(error = %error, "tunnel tcp request failed");
            MassOut::Error {
                message: error.to_string(),
            }
        }
    };
    if let Err(error) = session.send_response(request.request_id, out).await {
        warn!(error = %error, "failed to send tunnel tcp response");
    }
}

async fn set_worker_connection(token: Uuid, connection: TunnelConnectionHandle, ctx: &MassContext) {
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
    context: Arc<MassContext>,
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

async fn handle_tcp_connection(mut stream: TcpStream, context: Arc<MassContext>) {
    loop {
        let req: MassIn = match read_frame_io(&mut stream).await {
            Ok(req) => req,
            Err(ServerError::UnexpectedEof) => return,
            Err(error) => {
                warn!(error = %error, "failed to read tcp request frame");
                return;
            }
        };

        match req {
            MassIn::RegisterTunnel {
                worker_id,
                max_instances,
                capabilities,
            } => {
                let out = match handle_register_tunnel(
                    worker_id,
                    max_instances,
                    capabilities,
                    None,
                    &context,
                )
                .await
                {
                    Ok(out) => out,
                    Err(error) => MassOut::Error {
                        message: error.to_string(),
                    },
                };
                if let Err(error) = write_frame_io(&mut stream, &out).await {
                    warn!(error = %error, "failed to write tcp tunnel registration response");
                    return;
                }
                let MassOut::TunnelRegistered { token } = out else {
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
                    Err(error) => MassOut::Error {
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
    context: Arc<MassContext>,
    connection: Option<TunnelConnectionHandle>,
) {
    let req: MassIn = match read_frame_quic(&mut recv).await {
        Ok(r) => r,
        Err(e) => {
            let _ = write_frame_quic(
                &mut send,
                &MassOut::Error {
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
            MassOut::Error {
                message: error.to_string(),
            }
        }
    };

    if write_frame_quic(&mut send, &out).await.is_err() {
        warn!("failed to write response");
    }
}

async fn handle_request(
    req: MassIn,
    ctx: &MassContext,
    connection: Option<TunnelConnectionHandle>,
) -> Result<MassOut> {
    match req {
        MassIn::RegisterTunnel {
            worker_id,
            max_instances,
            capabilities,
        } => handle_register_tunnel(worker_id, max_instances, capabilities, connection, ctx).await,
        MassIn::UpdateTunnelCapacity {
            token,
            max_instances,
        } => handle_update_tunnel_capacity(token, max_instances, ctx).await,
        MassIn::TunnelForward { token, request } => {
            handle_tunnel_forward(token, request, ctx).await
        }
        MassIn::Start {
            model_id,
            model_config,
        } => handle_start_routed(model_id, model_config, ctx).await,
        MassIn::PerturbUp { model_id, seed } => handle_perturb_up_routed(model_id, seed, ctx).await,
        MassIn::Infer { model_id, input_id } => handle_infer_routed(model_id, input_id, ctx).await,
        MassIn::Reset { model_id } => handle_reset_routed(model_id, ctx).await,
        MassIn::PerturbDown { model_id } => handle_perturb_down_routed(model_id, ctx).await,
        MassIn::Checkpoint { model_id } => handle_checkpoint_routed(model_id, ctx).await,
        MassIn::Optimize {
            model_id,
            loss_up,
            loss_down,
        } => handle_optimize_routed(model_id, loss_up, loss_down, ctx).await,
        MassIn::Shutdown { model_id } => handle_shutdown_routed(model_id, ctx).await,
        MassIn::QueryModelParams { model_id } => {
            handle_query_model_params_routed(model_id, ctx).await
        }
        MassIn::QueryModelCapacity => handle_query_model_capacity(ctx).await,
        MassIn::FuseWeights {
            model_id,
            checkpoint_id,
            contribution,
        } => handle_fuse_weights_routed(model_id, checkpoint_id, contribution, ctx).await,
        MassIn::StartOperation {
            protocol_version,
            instance_id,
            capability,
        } => {
            ensure_operation_protocol_version(protocol_version)?;
            handle_operation_start_routed(instance_id, capability, ctx).await
        }
        MassIn::ForwardOperation {
            protocol_version,
            instance_id,
            input,
        } => {
            ensure_operation_protocol_version(protocol_version)?;
            handle_operation_forward_routed(instance_id, input, ctx).await
        }
        MassIn::ShutdownOperation { instance_id } => {
            handle_operation_shutdown_routed(instance_id, ctx).await
        }
    }
}

fn ensure_root_mode(ctx: &MassContext) -> Result<()> {
    match &ctx.mode {
        MassMode::Root => Ok(()),
        MassMode::Worker(_) => Err(ServerError::TunnelWorkerRejectsModelRequests),
    }
}

async fn register_tunnel_worker(
    parent_addr: SocketAddr,
    worker_id: Uuid,
    max_instances: Option<usize>,
    transport_mode: TransportMode,
    capabilities: WorkerCapabilities,
) -> Result<(Uuid, ParentTunnelSession)> {
    let client = MassRpcClient::connect(parent_addr, transport_mode).await?;
    let connection = client.establish_connection().await?;
    match connection {
        RpcConnection::Quic(connection) => {
            let out = request_over_connection(
                &TunnelConnectionHandle::Quic(connection.clone()),
                MassIn::RegisterTunnel {
                    worker_id,
                    max_instances,
                    capabilities: Some(capabilities),
                },
            )
            .await?;
            match out {
                MassOut::TunnelRegistered { token } => Ok((
                    token,
                    ParentTunnelSession {
                        _client: client,
                        connection: TunnelConnectionHandle::Quic(connection),
                    },
                )),
                MassOut::Error { message } => Err(ServerError::TunnelRegistrationRejected(message)),
                _ => Err(ServerError::UnexpectedTunnelResponse(
                    "register tunnel response",
                )),
            }
        }
        RpcConnection::Tcp(mut stream) => {
            write_frame_io(
                &mut stream,
                &MassIn::RegisterTunnel {
                    worker_id,
                    max_instances,
                    capabilities: Some(capabilities),
                },
            )
            .await?;
            let out: MassOut = read_frame_io(&mut stream).await?;
            match out {
                MassOut::TunnelRegistered { token } => Ok((
                    token,
                    ParentTunnelSession {
                        _client: client,
                        connection: TunnelConnectionHandle::Tcp(TcpTunnelSession::new(stream)),
                    },
                )),
                MassOut::Error { message } => Err(ServerError::TunnelRegistrationRejected(message)),
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
    capabilities: WorkerCapabilities,
) -> Result<(Uuid, ParentTunnelSession)> {
    let start = Instant::now();
    let deadline_at = deadline.map(|deadline| start + deadline);
    let mut retry_delay = Duration::from_millis(DEFAULT_TUNNEL_CONNECT_RETRY_MS);
    let max_retry_delay = Duration::from_millis(MAX_TUNNEL_CONNECT_RETRY_MS);
    let mut attempts = 0u32;
    loop {
        attempts = attempts.saturating_add(1);
        match register_tunnel_worker(
            parent_addr,
            worker_id,
            max_instances,
            transport_mode,
            capabilities.clone(),
        )
        .await
        {
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
        MassIn::UpdateTunnelCapacity {
            token: tunnel_token,
            max_instances,
        },
    )
    .await?;
    match out {
        MassOut::Ack => Ok(()),
        MassOut::Error { message } => Err(ServerError::TunnelCapacityUpdateRejected(message)),
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

async fn advertised_capacity(ctx: &MassContext) -> Option<usize> {
    let mut total = ctx.max_instances;
    for worker in ctx.workers.read().await.values() {
        total = sum_capacity(total, worker.max_instances);
    }
    total
}

async fn propagate_capacity_to_parent(ctx: &MassContext) -> Result<()> {
    match &ctx.mode {
        MassMode::Root => Ok(()),
        MassMode::Worker(worker_mode) => {
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
                        worker_mode.capabilities.clone(),
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

async fn maintain_parent_registration_loop(context: Arc<MassContext>) {
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
    capabilities: Option<WorkerCapabilities>,
    connection: Option<TunnelConnectionHandle>,
    ctx: &MassContext,
) -> Result<MassOut> {
    let max_instances = resolve_max_instances(max_instances);

    let token = {
        let mut workers = ctx.workers.write().await;
        if let Some((token, worker)) = workers
            .iter_mut()
            .find(|(_, worker)| worker.worker_id == worker_id)
        {
            worker.max_instances = max_instances;
            // Capabilities are compile-time fixed; refresh them when the
            // (re)registering worker advertises them.
            if let Some(capabilities) = capabilities {
                worker.capabilities = capabilities;
            }
            *token
        } else {
            let token = Uuid::new_v4();
            workers.insert(
                token,
                TunnelWorker {
                    token,
                    worker_id,
                    max_instances,
                    capabilities: capabilities.unwrap_or_default(),
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
    Ok(MassOut::TunnelRegistered { token })
}

async fn handle_update_tunnel_capacity(
    token: Uuid,
    max_instances: Option<usize>,
    ctx: &MassContext,
) -> Result<MassOut> {
    let max_instances = resolve_max_instances(max_instances);
    let mut workers = ctx.workers.write().await;
    let worker = workers
        .get_mut(&token)
        .ok_or(ServerError::TunnelWorkerUnavailable(token))?;
    worker.max_instances = max_instances;
    drop(workers);

    propagate_capacity_to_parent(ctx).await?;

    debug!(token = %token, ?max_instances, "updated tunnel worker capacity");
    Ok(MassOut::Ack)
}

async fn handle_tunnel_forward(
    token: Uuid,
    request: TunnelRequest,
    ctx: &MassContext,
) -> Result<MassOut> {
    match &ctx.mode {
        MassMode::Worker(worker_mode) => {
            let tunnel_token = *worker_mode.tunnel_token.read().await;
            if tunnel_token == token {
                handle_tunnel_request_local(request, ctx).await
            } else {
                Err(ServerError::TunnelUnauthorizedForward)
            }
        }
        MassMode::Root => Err(ServerError::TunnelForwardUnsupportedOnRoot),
    }
}

async fn handle_tunnel_request_local(request: TunnelRequest, ctx: &MassContext) -> Result<MassOut> {
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
        TunnelRequest::FuseWeights {
            model_id,
            checkpoint_id,
            contribution,
        } => handle_fuse_weights_distributed(model_id, checkpoint_id, contribution, ctx).await,
        TunnelRequest::StartOperation {
            protocol_version,
            instance_id,
            capability,
        } => {
            ensure_operation_protocol_version(protocol_version)?;
            handle_operation_start_distributed(instance_id, capability, ctx).await
        }
        TunnelRequest::ForwardOperation {
            protocol_version,
            instance_id,
            input,
        } => {
            ensure_operation_protocol_version(protocol_version)?;
            handle_operation_forward_distributed(instance_id, input, ctx).await
        }
        TunnelRequest::ShutdownOperation { instance_id } => {
            handle_operation_shutdown_distributed(instance_id, ctx).await
        }
    }
}

async fn route_for_model(model_id: Uuid, ctx: &MassContext) -> Result<RouteTarget> {
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

/// Whether a worker's advertised capabilities satisfy an architecture
/// requirement. A requirement-less start matches any engine; a required
/// architecture only matches engines compiled for it.
fn architecture_satisfies(
    capabilities: &WorkerCapabilities,
    required_architecture: Option<MassArchitecture>,
) -> bool {
    match required_architecture {
        None => true,
        Some(required) => capabilities.architectures.contains(&required),
    }
}

fn validate_capability(capability: &OperationCapability) -> Result<()> {
    let actual_hash = black_hole_spec::descriptor_hash(&capability.descriptor);
    if capability.descriptor_hash != actual_hash {
        return Err(ServerError::OperationContractHashMismatch);
    }
    if capability.tensor_encodings.is_empty() || capability.metadata_encodings.is_empty() {
        return Err(ServerError::OperationCodecSetEmpty);
    }
    Ok(())
}

fn ensure_operation_protocol_version(version: u16) -> Result<()> {
    if version == MASS_OPERATION_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ServerError::UnsupportedOperationProtocolVersion(version))
    }
}

/// Full contract identity and codec match. Identical shapes are deliberately
/// insufficient: semantic ID, version, descriptor hash, and codecs all have
/// to agree.
fn operation_satisfies(advertised: &OperationCapability, requested: &OperationCapability) -> bool {
    advertised.descriptor == requested.descriptor
        && advertised.descriptor_hash == requested.descriptor_hash
        && requested
            .tensor_encodings
            .iter()
            .all(|encoding| advertised.tensor_encodings.contains(encoding))
        && requested
            .metadata_encodings
            .iter()
            .all(|encoding| advertised.metadata_encodings.contains(encoding))
}

async fn select_operation_target(
    ctx: &MassContext,
    requested: &OperationCapability,
) -> Result<RouteTarget> {
    validate_capability(requested)?;
    let routes = ctx.routes.read().await;
    let local_count = routes
        .values()
        .filter(|target| matches!(target, RouteTarget::Local))
        .count();
    let mut worker_counts: HashMap<Uuid, usize> = HashMap::new();
    for target in routes.values() {
        if let RouteTarget::Worker(token) = target {
            *worker_counts.entry(*token).or_insert(0) += 1;
        }
    }
    drop(routes);

    let local_eligible = local_worker_capabilities(ctx.operation.as_ref())
        .operations
        .iter()
        .any(|advertised| operation_satisfies(advertised, requested));
    let mut best = local_eligible
        .then_some((RouteTarget::Local, local_count))
        .filter(|_| has_capacity(ctx.max_instances, local_count));
    let mut eligible_exists = local_eligible;

    let mut workers: Vec<TunnelWorker> = ctx.workers.read().await.values().cloned().collect();
    workers.sort_by_key(|worker| worker.token);
    for worker in workers {
        if !worker
            .capabilities
            .operations
            .iter()
            .any(|advertised| operation_satisfies(advertised, requested))
        {
            continue;
        }
        eligible_exists = true;
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

    match (best, eligible_exists) {
        (Some((target, _)), _) => Ok(target),
        (None, true) => Err(ServerError::NoTunnelCapacity),
        (None, false) => Err(ServerError::NoCompatibleOperation {
            id: requested.descriptor.id,
            version: requested.descriptor.version,
        }),
    }
}

async fn select_start_target(
    ctx: &MassContext,
    required_architecture: Option<MassArchitecture>,
) -> Result<RouteTarget> {
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

    let local_eligible = architecture_satisfies(
        &WorkerCapabilities {
            architectures: std::iter::once(COMPILED_ARCHITECTURE)
                .filter_map(|architecture| architecture)
                .collect(),
            operations: Vec::new(),
        },
        required_architecture,
    );

    let mut best: Option<(RouteTarget, usize)> = None;
    let mut eligible_exists = local_eligible;
    if local_eligible && has_capacity(ctx.max_instances, local_count) {
        best = Some((RouteTarget::Local, local_count));
    }

    let mut workers: Vec<TunnelWorker> = ctx.workers.read().await.values().cloned().collect();
    workers.sort_by_key(|worker| worker.token);
    for worker in workers {
        if !architecture_satisfies(&worker.capabilities, required_architecture) {
            continue;
        }
        eligible_exists = true;
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

    match (best, eligible_exists) {
        (Some((target, _)), _) => Ok(target),
        // Eligible engines exist but are all at capacity.
        (None, true) => Err(ServerError::NoTunnelCapacity),
        // No engine matches the required architecture at all.
        (None, false) if required_architecture.is_some() => Err(
            ServerError::NoCompatibleTunnelCapacity(required_architecture.expect("checked above")),
        ),
        (None, false) => Err(ServerError::NoTunnelCapacity),
    }
}

async fn get_worker(token: Uuid, ctx: &MassContext) -> Result<TunnelWorker> {
    ctx.workers
        .read()
        .await
        .get(&token)
        .cloned()
        .ok_or(ServerError::TunnelWorkerUnavailable(token))
}

async fn get_worker_connection(token: Uuid, ctx: &MassContext) -> Result<TunnelConnectionHandle> {
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
    ctx: &MassContext,
) -> Result<MassOut> {
    let _worker = get_worker(worker_token, ctx).await?;
    let connection = get_worker_connection(worker_token, ctx).await?;
    let out = request_over_connection(
        &connection,
        MassIn::TunnelForward {
            token: worker_token,
            request,
        },
    )
    .await
    .map_err(|error| ServerError::TunnelWorkerError(error.to_string()))?;
    match out {
        MassOut::Error { message } => Err(ServerError::TunnelWorkerError(message)),
        _ => Ok(out),
    }
}

async fn handle_start_routed(
    model_id: Uuid,
    model_config: Option<MassModelConfig>,
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_start_distributed(model_id, model_config, ctx).await
}

async fn handle_start_distributed(
    model_id: Uuid,
    model_config: Option<MassModelConfig>,
    ctx: &MassContext,
) -> Result<MassOut> {
    // Keep distributed starts serialized so routed and local model initialization
    // cannot overlap on the same mass context.
    let _start_dispatch_guard = ctx.start_dispatch.lock().await;

    if ctx.routes.read().await.contains_key(&model_id)
        || ctx.instances.read().await.contains_key(&model_id)
    {
        return Err(ServerError::ModelInstanceAlreadyRunning(model_id));
    }

    let required_architecture = model_config
        .as_ref()
        .and_then(|config| config.required_architecture);
    let target = select_start_target(ctx, required_architecture).await?;
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

    if !matches!(out, MassOut::Ack) {
        return Err(ServerError::UnexpectedTunnelResponse("start response"));
    }

    ctx.routes.write().await.insert(model_id, target);
    ctx.instance_requirements
        .write()
        .await
        .insert(model_id, required_architecture);
    Ok(MassOut::Ack)
}

async fn handle_perturb_up_routed(model_id: Uuid, seed: u64, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_perturb_up_distributed(model_id, seed, ctx).await
}

async fn handle_perturb_up_distributed(
    model_id: Uuid,
    seed: u64,
    ctx: &MassContext,
) -> Result<MassOut> {
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
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_infer_distributed(model_id, input_id, ctx).await
}

async fn handle_infer_distributed(
    model_id: Uuid,
    input_id: ObjectId,
    ctx: &MassContext,
) -> Result<MassOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_infer(model_id, input_id, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::Infer { model_id, input_id }, ctx).await
        }
    }
}

async fn handle_reset_routed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_reset_distributed(model_id, ctx).await
}

async fn handle_reset_distributed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_reset(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::Reset { model_id }, ctx).await
        }
    }
}

async fn handle_perturb_down_routed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_perturb_down_distributed(model_id, ctx).await
}

async fn handle_perturb_down_distributed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_perturb_down(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::PerturbDown { model_id }, ctx).await
        }
    }
}

async fn handle_checkpoint_routed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_checkpoint_distributed(model_id, ctx).await
}

async fn handle_checkpoint_distributed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
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
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_optimize_distributed(model_id, loss_up, loss_down, ctx).await
}

async fn handle_optimize_distributed(
    model_id: Uuid,
    loss_up: f32,
    loss_down: f32,
    ctx: &MassContext,
) -> Result<MassOut> {
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

async fn handle_shutdown_routed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_shutdown_distributed(model_id, ctx).await
}

async fn handle_shutdown_distributed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    let target = route_for_model(model_id, ctx).await?;
    let out = match target {
        RouteTarget::Local => handle_shutdown(model_id, ctx).await?,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::Shutdown { model_id }, ctx).await?
        }
    };
    if matches!(out, MassOut::Ack) {
        let mut routes = ctx.routes.write().await;
        routes.remove(&model_id);
        drop(routes);
        ctx.instance_requirements.write().await.remove(&model_id);
    }
    Ok(out)
}

async fn handle_query_model_params_routed(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_query_model_params_distributed(model_id, ctx).await
}

async fn handle_query_model_params_distributed(
    model_id: Uuid,
    ctx: &MassContext,
) -> Result<MassOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_query_model_params(model_id, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::QueryModelParams { model_id }, ctx).await
        }
    }
}

// ---------------------------------------------------------------------------
// Generic operation lifecycle
// ---------------------------------------------------------------------------

async fn handle_operation_start_routed(
    instance_id: Uuid,
    capability: OperationCapability,
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_operation_start_distributed(instance_id, capability, ctx).await
}

async fn handle_operation_start_distributed(
    instance_id: Uuid,
    capability: OperationCapability,
    ctx: &MassContext,
) -> Result<MassOut> {
    let _start_dispatch_guard = ctx.start_dispatch.lock().await;
    if ctx.routes.read().await.contains_key(&instance_id)
        || ctx.instances.read().await.contains_key(&instance_id)
        || ctx.operation_instances.read().await.contains(&instance_id)
    {
        return Err(ServerError::ModelInstanceAlreadyRunning(instance_id));
    }

    let target = select_operation_target(ctx, &capability).await?;
    let out = match target {
        RouteTarget::Local => handle_operation_start_local(instance_id, &capability, ctx).await?,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(
                token,
                TunnelRequest::StartOperation {
                    protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                    instance_id,
                    capability: capability.clone(),
                },
                ctx,
            )
            .await?
        }
    };
    if !matches!(out, MassOut::Ack) {
        return Err(ServerError::UnexpectedTunnelResponse(
            "generic operation start response",
        ));
    }
    ctx.routes.write().await.insert(instance_id, target);
    Ok(MassOut::Ack)
}

async fn handle_operation_start_local(
    instance_id: Uuid,
    requested: &OperationCapability,
    ctx: &MassContext,
) -> Result<MassOut> {
    let qwen =
        black_hole_spec::operation_capability::<black_hole_spec::QwenDarkInference>();
    if operation_satisfies(&qwen, requested) {
        let out = handle_start(instance_id, None, ctx).await?;
        ctx.operation_instances.write().await.insert(instance_id);
        return Ok(out);
    }
    let operation = ctx
        .operation
        .as_ref()
        .ok_or(ServerError::OperationNotConfigured)?;
    let advertised = operation.capability();
    if !operation_satisfies(&advertised, requested) {
        return Err(ServerError::OperationContractMismatch);
    }
    {
        let mut instances = ctx.operation_instances.write().await;
        if let Some(limit) = ctx.max_instances {
            let legacy = ctx.instances.read().await.len();
            if instances.len().saturating_add(legacy) >= limit {
                return Err(ServerError::NoLocalCapacity(limit));
            }
        }
        if !instances.insert(instance_id) {
            return Err(ServerError::ModelInstanceAlreadyRunning(instance_id));
        }
    }
    if let Err(message) = operation.start(instance_id).await {
        ctx.operation_instances.write().await.remove(&instance_id);
        return Err(ServerError::OperationError(message));
    }
    Ok(MassOut::Ack)
}

async fn handle_operation_forward_routed(
    instance_id: Uuid,
    input: OperationArtifactRef,
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_operation_forward_distributed(instance_id, input, ctx).await
}

async fn handle_operation_forward_distributed(
    instance_id: Uuid,
    input: OperationArtifactRef,
    ctx: &MassContext,
) -> Result<MassOut> {
    match route_for_model(instance_id, ctx).await? {
        RouteTarget::Local => handle_operation_forward_local(instance_id, input, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(
                token,
                TunnelRequest::ForwardOperation {
                    protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                    instance_id,
                    input,
                },
                ctx,
            )
            .await
        }
    }
}

async fn handle_operation_forward_local(
    instance_id: Uuid,
    input: OperationArtifactRef,
    ctx: &MassContext,
) -> Result<MassOut> {
    if !ctx.operation_instances.read().await.contains(&instance_id) {
        return Err(ServerError::ModelInstanceNotRunning(instance_id));
    }
    let operation = ctx.operation.as_ref();
    if ctx.instances.read().await.contains_key(&instance_id) {
        return handle_qwen_operation_forward(instance_id, input, ctx).await;
    }
    let operation = operation.ok_or(ServerError::OperationNotConfigured)?;
    let capability = operation.capability();
    let void = require_void_client(ctx, "generic operation forward")?;
    let input_bytes = void.download_artifact(input).await?;
    black_hole_spec::validate_artifact(
        &capability.descriptor,
        ContractSide::Input,
        &input_bytes,
    )
    .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;

    let output_bytes = operation
        .forward(instance_id, input_bytes)
        .await
        .map_err(ServerError::OperationError)?;
    black_hole_spec::validate_artifact(
        &capability.descriptor,
        ContractSide::Output,
        &output_bytes,
    )
    .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
    let output = void
        .publish_artifact(capability.descriptor, ContractSide::Output, output_bytes)
        .await?;
    Ok(MassOut::Forwarded { output })
}

async fn handle_qwen_operation_forward(
    instance_id: Uuid,
    input: OperationArtifactRef,
    ctx: &MassContext,
) -> Result<MassOut> {
    use black_hole_spec::{decode_input, encode_output, QwenDarkInference, RawTensor};

    let void = require_void_client(ctx, "generic Qwen forward")?;
    let input_bytes = void.download_artifact(input).await?;
    let decoded = decode_input::<QwenDarkInference>(&input_bytes)
        .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
    let predictions = &decoded.tensors[0];
    let token_ids = &decoded.tensors[1];
    let log_probs = &decoded.tensors[2];
    let [batch, sequence]: [usize; 2] = predictions
        .shape
        .as_slice()
        .try_into()
        .expect("Qwen contract validation guarantees rank 2");
    let top_k = token_ids.shape[2];

    let predictions: Vec<u32> = predictions
        .data
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let token_ids: Vec<u32> = token_ids
        .data
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let log_probs: Vec<f32> = log_probs
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let mut sequences = Vec::with_capacity(batch);
    for batch_index in 0..batch {
        let mut dark = Vec::with_capacity(sequence);
        for sequence_index in 0..sequence {
            let position = batch_index * sequence + sequence_index;
            let distribution_start = position * top_k;
            let dark_knowledge = (0..top_k)
                .map(|offset| LogitEntry {
                    token_id: token_ids[distribution_start + offset],
                    log_prob: log_probs[distribution_start + offset],
                })
                .collect();
            dark.push(DarkToken {
                predicted: predictions[position],
                dark_knowledge,
            });
        }
        sequences.push(vec![InferenceInput::Dark(dark)]);
    }

    let legacy_request = InferenceRequest::Sequences {
        sequences,
        limit: None,
    };
    let legacy_request_id = void
        .upload(to_allocvec(&legacy_request).map_err(ServerError::EncodeFrame)?)
        .await?;
    let MassOut::Inferred { output_id } = handle_infer(instance_id, legacy_request_id, ctx).await?
    else {
        return Err(ServerError::UnexpectedTunnelResponse(
            "Qwen compatibility forward response",
        ));
    };
    let legacy_output: InferenceOutput =
        from_bytes(&void.download(output_id).await?).map_err(ServerError::DecodeFrame)?;
    let output_batch = legacy_output.results.len();
    let output_sequence = legacy_output
        .results
        .first()
        .map_or(0, |sequence| sequence.0.len());
    let output_top_k = legacy_output
        .results
        .first()
        .and_then(|sequence| sequence.0.first())
        .map_or(0, |token| token.dark_knowledge.len());
    if legacy_output.results.iter().any(|result| {
        result.0.len() != output_sequence
            || result
                .0
                .iter()
                .any(|token| token.dark_knowledge.len() != output_top_k)
    }) {
        return Err(ServerError::OperationError(
            "Qwen compatibility output is ragged and cannot be encoded as a dense bundle".into(),
        ));
    }

    let mut output_predictions = Vec::new();
    let mut output_token_ids = Vec::new();
    let mut output_log_probs = Vec::new();
    for result in legacy_output.results {
        for token in result.0 {
            output_predictions.extend_from_slice(&token.predicted.to_le_bytes());
            for entry in token.dark_knowledge {
                output_token_ids.extend_from_slice(&entry.token_id.to_le_bytes());
                output_log_probs.extend_from_slice(&entry.log_prob.to_le_bytes());
            }
        }
    }
    let output_tensors = [
        RawTensor {
            name: "predictions".into(),
            dtype: black_hole_type::TensorDtype::U32,
            shape: vec![output_batch, output_sequence],
            data: output_predictions,
        },
        RawTensor {
            name: "dark_token_ids".into(),
            dtype: black_hole_type::TensorDtype::U32,
            shape: vec![output_batch, output_sequence, output_top_k],
            data: output_token_ids,
        },
        RawTensor {
            name: "dark_log_probs".into(),
            dtype: black_hole_type::TensorDtype::F32,
            shape: vec![output_batch, output_sequence, output_top_k],
            data: output_log_probs,
        },
    ];
    let output_bytes = encode_output::<QwenDarkInference>(&output_tensors, &())
        .map_err(|error| ServerError::OperationPayloadInvalid(error.to_string()))?;
    let output = void
        .publish_artifact(
            <black_hole_spec::QwenDarkInference as black_hole_spec::TensorContract>::descriptor(),
            ContractSide::Output,
            output_bytes,
        )
        .await?;
    Ok(MassOut::Forwarded { output })
}

async fn handle_operation_shutdown_routed(instance_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_operation_shutdown_distributed(instance_id, ctx).await
}

async fn handle_operation_shutdown_distributed(
    instance_id: Uuid,
    ctx: &MassContext,
) -> Result<MassOut> {
    let target = route_for_model(instance_id, ctx).await?;
    let out = match target {
        RouteTarget::Local => handle_operation_shutdown_local(instance_id, ctx).await?,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(token, TunnelRequest::ShutdownOperation { instance_id }, ctx)
                .await?
        }
    };
    if matches!(out, MassOut::Ack) {
        ctx.routes.write().await.remove(&instance_id);
    }
    Ok(out)
}

async fn handle_operation_shutdown_local(instance_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    if !ctx.operation_instances.read().await.contains(&instance_id) {
        return Err(ServerError::ModelInstanceNotRunning(instance_id));
    }
    let operation = ctx.operation.as_ref();
    if ctx.instances.read().await.contains_key(&instance_id) {
        let out = handle_shutdown(instance_id, ctx).await?;
        ctx.operation_instances.write().await.remove(&instance_id);
        return Ok(out);
    }
    let operation = operation.ok_or(ServerError::OperationNotConfigured)?;
    operation
        .shutdown(instance_id)
        .await
        .map_err(ServerError::OperationError)?;
    ctx.operation_instances.write().await.remove(&instance_id);
    Ok(MassOut::Ack)
}

// ---------------------------------------------------------------------------
// Model instance lifecycle
// ---------------------------------------------------------------------------

async fn handle_start(
    model_id: Uuid,
    model_config: Option<MassModelConfig>,
    ctx: &MassContext,
) -> Result<MassOut> {
    // Reject starts this compiled engine cannot serve. The root already
    // filters placement by capability; this guards direct and forwarded
    // starts so a mismatch fails fast instead of at weight-load time.
    if let Some(required) = model_config
        .as_ref()
        .and_then(|config| config.required_architecture)
    {
        if COMPILED_ARCHITECTURE != Some(required) {
            return Err(ServerError::ArchitectureMismatch {
                required,
                compiled: COMPILED_ARCHITECTURE,
            });
        }
    }

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
        training_perturbation_mode: to_mass_perturbation_mode(
            defaults.training_config.perturbation_mode,
        ),
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
        training_perturbation_mode = ?runtime_config.training_perturbation_mode,
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

    let mut session = MassSession::new(frozen);
    apply_initial_frozen_oscillation(model_id, &mut session, oscillation);
    let instance = Arc::new(MassInstance {
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
    Ok(MassOut::Ack)
}

async fn handle_shutdown(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
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
    Ok(MassOut::Ack)
}

async fn get_instance(model_id: Uuid, ctx: &MassContext) -> Result<Arc<MassInstance>> {
    match ctx.instances.read().await.get(&model_id) {
        Some(ModelSlot::Running(instance)) => Ok(Arc::clone(instance)),
        Some(ModelSlot::Starting | ModelSlot::ShuttingDown) | None => {
            Err(ServerError::ModelInstanceNotRunning(model_id))
        }
    }
}

fn ensure_running(session: &MassSession, model_id: Uuid) -> Result<()> {
    if session.running {
        Ok(())
    } else {
        Err(ServerError::ModelInstanceNotRunning(model_id))
    }
}

fn build_model_params(
    runtime_config: ModelRuntimeConfig,
    oscillation: Option<FrozenOscillation>,
    session: &MassSession,
) -> MassModelParams {
    MassModelParams {
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
        training_perturbation_mode: runtime_config.training_perturbation_mode,
        training_error_feedback: runtime_config.training_error_feedback,
        is_frozen: session.frozen,
        optimize_steps: session.optimize_steps,
        oscillation_period_steps: oscillation.map(|osc| osc.period_steps),
        oscillation_train_steps: oscillation.map(|osc| osc.train_steps),
        oscillation_phase_steps: oscillation.map(|osc| osc.phase_steps),
        oscillation_warmup_steps: oscillation.map(|osc| osc.warmup_steps),
    }
}

async fn handle_query_model_params(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    Ok(MassOut::ModelParams {
        params: build_model_params(instance.runtime_config, instance.oscillation, &session),
    })
}

/// Architectures a route target's engine can serve (local engine or worker).
fn target_architectures(
    target: &RouteTarget,
    workers: &HashMap<Uuid, TunnelWorker>,
) -> Vec<MassArchitecture> {
    match target {
        RouteTarget::Local => std::iter::once(COMPILED_ARCHITECTURE)
            .filter_map(|architecture| architecture)
            .collect(),
        RouteTarget::Worker(token) => workers
            .get(token)
            .map(|worker| worker.capabilities.architectures.clone())
            .unwrap_or_default(),
    }
}

async fn handle_query_model_capacity(ctx: &MassContext) -> Result<MassOut> {
    let (occupied, total, per_architecture) = {
        let routes = ctx.routes.read().await;
        let workers = ctx.workers.read().await;
        let requirements = ctx.instance_requirements.read().await;

        // Total capacity across local + workers (inlined from advertised_capacity
        // because the worker lock is already held).
        let mut total = ctx.max_instances;
        for worker in workers.values() {
            total = sum_capacity(total, worker.max_instances);
        }

        // Per-architecture view: one entry per architecture any engine in the
        // subtree can serve. An instance occupies the architectures of its
        // serving engine; instances started with an explicit requirement only
        // count against that architecture.
        let mut architectures: Vec<MassArchitecture> = Vec::new();
        if let Some(architecture) = COMPILED_ARCHITECTURE {
            architectures.push(architecture);
        }
        for worker in workers.values() {
            for architecture in &worker.capabilities.architectures {
                if !architectures.contains(architecture) {
                    architectures.push(*architecture);
                }
            }
        }
        let per_architecture = architectures
            .into_iter()
            .map(|architecture| {
                // Start from the local engine's limit when it serves this
                // architecture (None means unbounded), else zero; sum_capacity
                // treats None as unbounded, so an ineligible local engine must
                // not contribute.
                let mut arch_total = match COMPILED_ARCHITECTURE {
                    Some(local) if local == architecture => ctx.max_instances,
                    _ => Some(0),
                };
                for worker in workers.values() {
                    if worker.capabilities.architectures.contains(&architecture) {
                        arch_total = sum_capacity(arch_total, worker.max_instances);
                    }
                }
                let mut arch_occupied = 0usize;
                for (model_id, target) in routes.iter() {
                    if !target_architectures(target, &workers).contains(&architecture) {
                        continue;
                    }
                    let counts_against_arch = match requirements.get(model_id).copied().flatten() {
                        Some(required) => required == architecture,
                        None => true,
                    };
                    if counts_against_arch {
                        arch_occupied += 1;
                    }
                }
                let available = arch_total.map(|total| total.saturating_sub(arch_occupied));
                (
                    architecture,
                    MassModelCapacity {
                        total: arch_total,
                        available,
                        occupied: arch_occupied,
                        per_architecture: Vec::new(),
                    },
                )
            })
            .collect();

        (routes.len(), total, per_architecture)
    };

    let available = total.map(|total| total.saturating_sub(occupied));
    Ok(MassOut::ModelCapacity {
        capacity: MassModelCapacity {
            total,
            available,
            occupied,
            per_architecture,
        },
    })
}

fn resolve_model_frozen(server_frozen: bool, model_config: Option<&MassModelConfig>) -> bool {
    model_config
        .and_then(|model_config| model_config.frozen)
        .unwrap_or(server_frozen)
}

fn resolve_model_oscillation(
    model_config: Option<&MassModelConfig>,
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
    session: &mut MassSession,
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
    session: &mut MassSession,
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

fn checkpoint_cache_dir() -> Result<PathBuf> {
    let checkpoint_dir = std::env::temp_dir().join(CHECKPOINT_CACHE_DIR);
    fs::create_dir_all(&checkpoint_dir).map_err(|source| ServerError::CreateCheckpointDir {
        path: checkpoint_dir.clone(),
        source,
    })?;
    Ok(checkpoint_dir)
}

fn checkpoint_file_path(model_id: Uuid, checkpoint_id: ObjectId) -> Result<PathBuf> {
    let checkpoint_dir = checkpoint_cache_dir()?;
    Ok(checkpoint_dir.join(format!("{model_id}-{checkpoint_id}.gguf")))
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
    model_config: Option<&MassModelConfig>,
    ctx: &MassContext,
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
    let checkpoint_path = checkpoint_file_path(model_id, checkpoint_id)?;
    let downloaded = void
        .download_to_file(checkpoint_id, &checkpoint_path)
        .await?;
    if downloaded == 0 {
        cleanup_checkpoint_file(&checkpoint_path);
        return Err(ServerError::CheckpointEmpty(checkpoint_id));
    }
    Ok(ResolvedModelSource {
        model_path: checkpoint_path.clone(),
        tokenizer_path: Some(tokenizer_path),
        checkpoint_path: Some(checkpoint_path),
    })
}

fn require_void_client<'a>(
    ctx: &'a MassContext,
    operation: &'static str,
) -> Result<&'a Arc<VoidClient>> {
    ctx.void_client
        .as_ref()
        .ok_or_else(|| void_not_configured_error(ctx, operation))
}

fn void_not_configured_error(ctx: &MassContext, operation: &'static str) -> ServerError {
    match &ctx.mode {
        MassMode::Root => ServerError::VoidNotConfigured,
        MassMode::Worker(_) => ServerError::TunnelWorkerVoidNotConfigured(operation),
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

fn error_feedback_mode_name(config: MassErrorFeedbackConfig) -> Option<&'static str> {
    match config {
        MassErrorFeedbackConfig::Off => None,
        MassErrorFeedbackConfig::Persistent { .. } => Some("persistent"),
        MassErrorFeedbackConfig::Replay { .. } => Some("replay"),
    }
}

fn error_feedback_support_hint(
    training_error_feedback: MassErrorFeedbackConfig,
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

/// Save the engine's current weights to a GGUF file on disk.
async fn save_model_checkpoint(engine: &ModelEngine) -> Result<PathBuf> {
    engine
        .save_checkpoint()
        .await
        .map_err(|error| ServerError::ModelError(error.to_string()))
}

// ---------------------------------------------------------------------------
// QuZO step handlers
// ---------------------------------------------------------------------------

async fn handle_perturb_up(model_id: Uuid, seed: u64, ctx: &MassContext) -> Result<MassOut> {
    debug!(%model_id, "received perturb up request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != MassState::Idle {
        warn!("expected Idle, got {:?}", session.state);
        return Err(ServerError::InvalidMassState(format!(
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

    session.state = MassState::PostPerturbUp;
    Ok(MassOut::Ack)
}

async fn handle_reset(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    debug!(%model_id, "received reset request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    reset_model(&instance.engine).await?;
    Ok(MassOut::Ack)
}

async fn handle_infer(model_id: Uuid, input_id: ObjectId, ctx: &MassContext) -> Result<MassOut> {
    debug!(%model_id, "received inference request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if !matches!(
        session.state,
        MassState::Idle
            | MassState::PostPerturbUp
            | MassState::AwaitingPerturbDown
            | MassState::PostPerturbDown
            | MassState::AwaitingOptimize
    ) {
        warn!(
            "inference requires Idle or an active perturbation phase, got {:?}",
            session.state
        );
        return Err(ServerError::InvalidMassState(format!(
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
                .map(|inputs| {
                    inputs
                        .into_iter()
                        .map(QwenOperationAdapter::model_input)
                        .collect()
                })
                .collect();
            (seqs, limit)
        }
        InferenceRequest::VoidId { id, limit } => {
            // Download the InferenceOutput and convert to dark input sequences.
            let output_bytes = void.download(id.id()).await?;
            let inference_output: InferenceOutput =
                from_bytes(&output_bytes).map_err(ServerError::DecodeFrame)?;

            let seqs: Vec<Vec<paramecia_engine::ModelInput>> = inference_output
                .results
                .into_iter()
                .map(|sequence| vec![QwenOperationAdapter::dark_input(sequence.0)])
                .collect();
            (seqs, limit)
        }
    };
    let limit = limit.unwrap_or(instance.runtime_config.inference_limit);

    // Run batched inference.
    let seq_results = run_batched_inference(&instance.engine, &sequences, limit).await?;

    // Convert per-sequence predictions to serializable output.
    let output = QwenOperationAdapter::output(seq_results);

    // Upload output to void.
    let output_bytes = to_allocvec(&output).map_err(ServerError::EncodeFrame)?;
    let output_id = void.upload(output_bytes).await?;

    // Advance state.
    session.state = match state {
        MassState::PostPerturbUp | MassState::AwaitingPerturbDown => MassState::AwaitingPerturbDown,
        MassState::Idle | MassState::PostPerturbDown | MassState::AwaitingOptimize => {
            MassState::AwaitingOptimize
        }
    };

    debug!(%model_id, "finished processing inference request");
    Ok(MassOut::Inferred { output_id })
}

async fn handle_perturb_down(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    debug!(%model_id, "received perturb down request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != MassState::AwaitingPerturbDown {
        warn!("expected AwaitingPerturbDown, got {:?}", session.state);
        return Err(ServerError::InvalidMassState(format!(
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

    session.state = MassState::PostPerturbDown;
    Ok(MassOut::Ack)
}

async fn handle_checkpoint(model_id: Uuid, ctx: &MassContext) -> Result<MassOut> {
    debug!(%model_id, "received checkpoint request");
    let instance = get_instance(model_id, ctx).await?;
    let session = instance.session.lock().await;
    ensure_running(&session, model_id)?;

    let void = require_void_client(ctx, "checkpoint upload")?;
    let checkpoint_path = save_model_checkpoint(&instance.engine).await?;
    let checkpoint_id = match void.upload_file(&checkpoint_path).await {
        Ok(id) => id,
        Err(error) => {
            cleanup_checkpoint_file(&checkpoint_path);
            return Err(error);
        }
    };
    cleanup_checkpoint_file(&checkpoint_path);

    Ok(MassOut::Checkpointed { checkpoint_id })
}

async fn handle_fuse_weights_routed(
    model_id: Uuid,
    checkpoint_id: ObjectId,
    contribution: f32,
    ctx: &MassContext,
) -> Result<MassOut> {
    ensure_root_mode(ctx)?;
    handle_fuse_weights_distributed(model_id, checkpoint_id, contribution, ctx).await
}

async fn handle_fuse_weights_distributed(
    model_id: Uuid,
    checkpoint_id: ObjectId,
    contribution: f32,
    ctx: &MassContext,
) -> Result<MassOut> {
    match route_for_model(model_id, ctx).await? {
        RouteTarget::Local => handle_fuse_weights(model_id, checkpoint_id, contribution, ctx).await,
        RouteTarget::Worker(token) => {
            forward_tunnel_request(
                token,
                TunnelRequest::FuseWeights {
                    model_id,
                    checkpoint_id,
                    contribution,
                },
                ctx,
            )
            .await
        }
    }
}

/// Fuse the instance's current weights with a void-stored checkpoint using
/// task arithmetic (base = live weights, member = checkpoint at `contribution`),
/// and upload the fused GGUF back to void.
async fn handle_fuse_weights(
    model_id: Uuid,
    checkpoint_id: ObjectId,
    contribution: f32,
    ctx: &MassContext,
) -> Result<MassOut> {
    debug!(%model_id, %checkpoint_id, contribution, "received fuse weights request");
    let instance = get_instance(model_id, ctx).await?;

    // Capture the live weights while holding the session lock so a concurrent
    // PerturbUp cannot interleave and leave us fusing perturbed weights.
    let base_path = {
        let session = instance.session.lock().await;
        ensure_running(&session, model_id)?;
        if session.state != MassState::Idle {
            return Err(ServerError::InvalidMassState(format!(
                "FuseWeights requires Idle state, got {:?}",
                session.state
            )));
        }
        instance
            .engine
            .save_checkpoint()
            .await
            .map_err(|error| ServerError::ModelError(error.to_string()))?
    };

    let void = require_void_client(ctx, "weight fusion")?;

    // Stream the checkpoint down from void.
    let checkpoint_dir = checkpoint_cache_dir()?;
    let checkpoint_path = checkpoint_dir.join(format!("{model_id}-{checkpoint_id}.gguf"));
    let downloaded = void
        .download_to_file(checkpoint_id, &checkpoint_path)
        .await?;
    if downloaded == 0 {
        cleanup_checkpoint_file(&checkpoint_path);
        return Err(ServerError::CheckpointEmpty(checkpoint_id));
    }

    // Run the (CPU-bound, file-level) fusion off the async runtime.
    let fused_path = checkpoint_dir.join(format!("{model_id}-fused-{checkpoint_id}.gguf"));
    let fuse_error = {
        let base_path = base_path.clone();
        let checkpoint_path = checkpoint_path.clone();
        let fused_path = fused_path.clone();
        tokio::task::spawn_blocking(move || {
            fuse_models(
                &base_path,
                &[(checkpoint_path, contribution)],
                &fused_path,
                QuantConflictStrategy::Reject,
            )
        })
        .await
    };
    match fuse_error {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            cleanup_checkpoint_file(&base_path);
            cleanup_checkpoint_file(&checkpoint_path);
            return Err(ServerError::ModelError(format!(
                "model fusion failed: {error}"
            )));
        }
        Err(join_error) => {
            cleanup_checkpoint_file(&base_path);
            cleanup_checkpoint_file(&checkpoint_path);
            return Err(ServerError::ModelError(format!(
                "fusion task failed: {join_error}"
            )));
        }
    }

    // Stream the fused weights back up to void.
    let fused_id = match void.upload_file(&fused_path).await {
        Ok(id) => id,
        Err(error) => {
            cleanup_checkpoint_file(&base_path);
            cleanup_checkpoint_file(&checkpoint_path);
            cleanup_checkpoint_file(&fused_path);
            return Err(error);
        }
    };

    // Best-effort cleanup of the temporary GGUFs.
    cleanup_checkpoint_file(&base_path);
    cleanup_checkpoint_file(&checkpoint_path);
    cleanup_checkpoint_file(&fused_path);

    info!(%model_id, %fused_id, "fused weights uploaded to void");
    Ok(MassOut::FusedWeights { fused_id })
}

async fn handle_optimize(
    model_id: Uuid,
    loss_up: f32,
    loss_down: f32,
    ctx: &MassContext,
) -> Result<MassOut> {
    debug!(%model_id, "received optimization request");
    let instance = get_instance(model_id, ctx).await?;
    let mut session = instance.session.lock().await;
    ensure_running(&session, model_id)?;
    if session.state != MassState::AwaitingOptimize {
        warn!("expected AwaitingOptimize, got {:?}", session.state);
        return Err(ServerError::InvalidMassState(format!(
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

    session.state = MassState::Idle;
    apply_frozen_oscillation(model_id, &mut session, instance.oscillation);
    debug!(%model_id, "finished optimization update");
    Ok(MassOut::Ack)
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
    let dirs = directories_next::ProjectDirs::from("org", "blackhole", "mass").unwrap();
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
    #[error("generic operation is not configured on this mass")]
    OperationNotConfigured,
    #[error("unsupported generic operation protocol version {0}")]
    UnsupportedOperationProtocolVersion(u16),
    #[error("generic operation contract does not match the injected implementation")]
    OperationContractMismatch,
    #[error("generic operation contract descriptor hash mismatch")]
    OperationContractHashMismatch,
    #[error("generic operation must declare at least one tensor and metadata codec")]
    OperationCodecSetEmpty,
    #[error("generic operation payload validation failed: {0}")]
    OperationPayloadInvalid(String),
    #[error("generic operation failed: {0}")]
    OperationError(String),
    #[error("no mass advertises operation contract {id:?} version {version}")]
    NoCompatibleOperation { id: ContractId, version: u32 },
    #[error("model instance {0} is already running")]
    ModelInstanceAlreadyRunning(Uuid),
    #[error("model instance {0} is not running")]
    ModelInstanceNotRunning(Uuid),
    #[error("invalid Mass state machine transition: {0}")]
    InvalidMassState(String),
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
        "void service not configured on tunnel worker (required for {0}); set --void-addr to the same void service as the root mass"
    )]
    TunnelWorkerVoidNotConfigured(&'static str),
    #[error("failed to resolve home directory for checkpoint tokenizer")]
    HomeDirectoryUnavailable,
    #[error("checkpoint start requires tokenizer file at {0}")]
    CheckpointTokenizerMissing(PathBuf),
    #[error("checkpoint {0} downloaded from void is empty")]
    CheckpointEmpty(ObjectId),
    #[error("failed to read file metadata for {path}: {source}")]
    FileMetadata {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to open file {path}: {source}")]
    OpenFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to read file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to write file {path}: {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to create checkpoint cache directory {path}: {source}")]
    CreateCheckpointDir {
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
    #[error("tunnel forward request is only valid on worker masss")]
    TunnelForwardUnsupportedOnRoot,
    #[error("no mass capacity available across local and registered workers")]
    NoTunnelCapacity,
    #[error(
        "no engine compiled for architecture {0:?} is available (local or tunnel worker); \
         start a mass built with that model feature and point it at this root via --tunnel"
    )]
    NoCompatibleTunnelCapacity(MassArchitecture),
    #[error("local mass reached max_instances capacity ({0})")]
    NoLocalCapacity(usize),
    #[error(
        "architecture mismatch: instance requires {required:?} but this mass engine was \
         compiled for {compiled:?}"
    )]
    ArchitectureMismatch {
        required: MassArchitecture,
        compiled: Option<MassArchitecture>,
    },
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
        client_bind_addr_for, ensure_operation_protocol_version, handle_query_model_capacity,
        handle_register_tunnel, handle_start, repair_duplicated_absolute_model_path,
        resolve_max_instances, resolve_model_frozen, resolve_model_oscillation, route_for_model,
        select_operation_target, select_start_target, to_engine_error_feedback,
        to_engine_perturbation_mode, to_mass_perturbation_mode, FrozenOscillation, MassContext,
        MassMode, MassServerDefaults, MassSession, MassState, ModelRuntimeConfig, ModelSlot,
        RouteTarget, ServerBuilder, ServerError, TransportMode, TunnelWorker,
        DEFAULT_INFERENCE_LIMIT, DEFAULT_MAX_INSTANCES,
    };
    use black_hole_spec::{
        encode_output,
        glowstick::{Dyn, Shape1},
        operation_capability, QwenDarkInference, RawTensor, SingleTensorSpec, TensorContract,
        TensorPortSpec,
    };
    use black_hole_type::{
        ContractId, DimensionDescriptor, DtypeConstraint, EncodingId, MassArchitecture,
        MassErrorFeedbackConfig, MassModelCapacity, MassModelConfig, MassPerturbationMode,
        OperationArtifactRef, TensorDtype, WorkerCapabilities,
    };
    use std::{collections::HashMap, fs, net::SocketAddr, path::PathBuf};
    use tokio::sync::{Mutex, RwLock};
    use uuid::Uuid;

    struct FakeOperation;
    struct SameShapeOtherOperation;
    struct FakeOperationV2;
    struct StreamAxis;
    struct StreamPort;
    struct StreamOperation;

    impl TensorPortSpec for StreamPort {
        type Shape = Shape1<Dyn<StreamAxis>>;

        const NAME: &'static str = "bytes";

        fn dimensions() -> Vec<DimensionDescriptor> {
            vec![DimensionDescriptor::Dynamic]
        }

        fn dtype() -> DtypeConstraint {
            DtypeConstraint::Exact(TensorDtype::U8)
        }
    }

    impl TensorContract for StreamOperation {
        type Input = SingleTensorSpec<StreamPort>;
        type Output = SingleTensorSpec<StreamPort>;
        type Metadata = ();

        const ID: ContractId = ContractId::from_u128(0x7374_7265_616d_2d6f_7065_7261_7469_6f6e);
        const VERSION: u32 = 1;
    }

    macro_rules! qwen_shaped_contract {
        ($operation:ty, $id:expr, $version:expr) => {
            impl TensorContract for $operation {
                type Input = <QwenDarkInference as TensorContract>::Input;
                type Output = <QwenDarkInference as TensorContract>::Output;
                type Metadata = ();

                const ID: ContractId = ContractId::from_u128($id);
                const VERSION: u32 = $version;
            }
        };
    }

    qwen_shaped_contract!(FakeOperation, 0x6661_6b65_2d6f_7065_7261_7469_6f6e_0001, 1);
    qwen_shaped_contract!(
        SameShapeOtherOperation,
        0x7361_6d65_2d73_6861_7065_2d6f_7468_6572,
        1
    );

    #[tokio::test]
    async fn operation_artifacts_are_published_as_live_replayable_streams() {
        let (void_addr, void_handle) = black_hole_void::ServerBuilder::new(
            Box::new(black_hole_void::object_store::InMemoryObjectStore::new()),
            Box::new(black_hole_void::persist::InMemoryStore::new()),
        )
        .tcp()
        .listen("127.0.0.1:0".parse().unwrap())
        .serve()
        .await
        .unwrap();
        let publisher = super::VoidClient::connect(void_addr, TransportMode::Tcp)
            .await
            .unwrap();
        let consumer = super::VoidClient::connect(void_addr, TransportMode::Tcp)
            .await
            .unwrap();
        let payload = b"progressive-output".to_vec();
        let frame = encode_output::<StreamOperation>(
            &[RawTensor {
                name: "bytes".into(),
                dtype: TensorDtype::U8,
                shape: vec![payload.len()],
                data: payload,
            }],
            &(),
        )
        .unwrap();

        let reference = publisher
            .publish_artifact(
                StreamOperation::descriptor(),
                super::ContractSide::Output,
                frame.clone(),
            )
            .await
            .unwrap();
        assert!(matches!(reference, OperationArtifactRef::Stream { .. }));
        assert_eq!(consumer.download_artifact(reference).await.unwrap(), frame);

        let OperationArtifactRef::Stream {
            ticket_id,
            fallback_transfer_id,
        } = reference
        else {
            unreachable!()
        };
        assert_eq!(
            publisher
                .download_committed_transfer(fallback_transfer_id)
                .await
                .unwrap(),
            frame
        );

        let ticket_bytes = publisher.download(ticket_id).await.unwrap();
        let mut interrupted_ticket: super::TransferTicket =
            postcard::from_bytes(&ticket_bytes).unwrap();
        interrupted_ticket.source = "127.0.0.1:9".into();
        let interrupted_ticket_id = publisher
            .upload(postcard::to_allocvec(&interrupted_ticket).unwrap())
            .await
            .unwrap();
        assert_eq!(
            consumer
                .download_artifact(OperationArtifactRef::Stream {
                    ticket_id: interrupted_ticket_id,
                    fallback_transfer_id,
                })
                .await
                .unwrap(),
            frame
        );
        void_handle.abort();
    }

    #[test]
    fn generic_protocol_versions_fail_closed() {
        assert!(ensure_operation_protocol_version(
            black_hole_type::MASS_OPERATION_PROTOCOL_VERSION
        )
        .is_ok());
        assert!(matches!(
            ensure_operation_protocol_version(black_hole_type::MASS_OPERATION_PROTOCOL_VERSION + 1),
            Err(ServerError::UnsupportedOperationProtocolVersion(_))
        ));
    }
    qwen_shaped_contract!(
        FakeOperationV2,
        0x6661_6b65_2d6f_7065_7261_7469_6f6e_0001,
        2
    );

    #[test]
    fn model_config_none_passes_through_server_defaults() {
        let defaults = MassServerDefaults::default();
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
            resolved.training_config.perturbation_mode,
            defaults.training_config.perturbation_mode
        );
        assert_eq!(
            resolved.training_error_feedback,
            defaults.training_error_feedback
        );
    }

    #[test]
    fn model_config_overrides_selected_fields() {
        let defaults = MassServerDefaults::default();
        let resolved = defaults.with_overrides(Some(&MassModelConfig {
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
            training_perturbation_mode: Some(MassPerturbationMode::LowRank(4)),
            training_error_feedback: Some(MassErrorFeedbackConfig::Persistent {
                decay: 0.8,
                gain: 0.6,
            }),
            frozen: None,
            oscillation_period_steps: None,
            oscillation_train_steps: None,
            oscillation_phase_steps: None,
            oscillation_warmup_steps: None,
            checkpoint_id: None,
            required_architecture: None,
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
            resolved.training_config.perturbation_mode,
            paramecia_engine::PerturbationMode::LowRank(4)
        );
        assert_eq!(
            resolved.training_error_feedback,
            MassErrorFeedbackConfig::Persistent {
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
            Some(&MassModelConfig {
                frozen: Some(true),
                ..MassModelConfig::default()
            })
        ));
        assert!(!resolve_model_frozen(
            true,
            Some(&MassModelConfig {
                frozen: Some(false),
                ..MassModelConfig::default()
            })
        ));
    }

    #[test]
    fn model_config_oscillation_defaults_to_none() {
        assert_eq!(resolve_model_oscillation(None).unwrap(), None);
        assert_eq!(
            resolve_model_oscillation(Some(&MassModelConfig {
                oscillation_warmup_steps: Some(7),
                ..MassModelConfig::default()
            }))
            .unwrap(),
            None
        );
    }

    #[test]
    fn model_config_oscillation_resolves_period_train_phase_and_warmup() {
        let resolved = resolve_model_oscillation(Some(&MassModelConfig {
            oscillation_period_steps: Some(10),
            oscillation_train_steps: Some(3),
            oscillation_phase_steps: Some(4),
            oscillation_warmup_steps: Some(20),
            ..MassModelConfig::default()
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
        let err = resolve_model_oscillation(Some(&MassModelConfig {
            oscillation_period_steps: Some(0),
            oscillation_train_steps: Some(1),
            ..MassModelConfig::default()
        }))
        .expect_err("zero oscillation period should be rejected");
        assert!(matches!(
            err,
            super::ServerError::InvalidOscillationPeriodSteps(0)
        ));
    }

    #[test]
    fn model_config_oscillation_requires_train_steps_when_period_is_set() {
        let err = resolve_model_oscillation(Some(&MassModelConfig {
            oscillation_period_steps: Some(8),
            ..MassModelConfig::default()
        }))
        .expect_err("missing train steps should be rejected");
        assert!(matches!(
            err,
            super::ServerError::MissingOscillationTrainSteps
        ));
    }

    #[test]
    fn model_config_oscillation_rejects_train_steps_greater_than_period() {
        let err = resolve_model_oscillation(Some(&MassModelConfig {
            oscillation_period_steps: Some(8),
            oscillation_train_steps: Some(9),
            ..MassModelConfig::default()
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
        let mut session = MassSession {
            state: MassState::AwaitingOptimize,
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
            training_perturbation_mode: MassPerturbationMode::LowRank(2),
            training_error_feedback: MassErrorFeedbackConfig::Replay {
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
        let mut session = MassSession {
            state: MassState::AwaitingOptimize,
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
            first.training_perturbation_mode,
            MassPerturbationMode::LowRank(2)
        );
        assert_eq!(
            first.training_error_feedback,
            MassErrorFeedbackConfig::Replay {
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
    fn mass_error_feedback_maps_to_engine_modes() {
        assert!(matches!(
            to_engine_error_feedback(MassErrorFeedbackConfig::Off),
            paramecia_engine::ErrorFeedbackMode::None
        ));
        assert!(matches!(
            to_engine_error_feedback(MassErrorFeedbackConfig::Persistent {
                decay: 0.9,
                gain: 1.0
            }),
            paramecia_engine::ErrorFeedbackMode::Persistent(_)
        ));
        assert!(matches!(
            to_engine_error_feedback(MassErrorFeedbackConfig::Replay {
                steps: 8,
                decay: 0.7,
                gain: 0.5
            }),
            paramecia_engine::ErrorFeedbackMode::Replay(_)
        ));
    }

    #[test]
    fn mass_perturbation_mode_maps_to_engine_modes() {
        assert_eq!(
            to_engine_perturbation_mode(MassPerturbationMode::Weight),
            paramecia_engine::PerturbationMode::Weight
        );
        assert_eq!(
            to_engine_perturbation_mode(MassPerturbationMode::LowRank(1)),
            paramecia_engine::PerturbationMode::LowRank(1)
        );
        assert_eq!(
            to_mass_perturbation_mode(paramecia_engine::PerturbationMode::Weight),
            MassPerturbationMode::Weight
        );
        assert_eq!(
            to_mass_perturbation_mode(paramecia_engine::PerturbationMode::LowRank(4)),
            MassPerturbationMode::LowRank(4)
        );
    }

    #[test]
    fn oscillation_phase_can_initialize_frozen_state_before_first_optimize() {
        let mut up = MassSession {
            state: MassState::Idle,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let mut down = MassSession {
            state: MassState::Idle,
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
        let mut half_up = MassSession {
            state: MassState::AwaitingOptimize,
            running: true,
            frozen: false,
            optimize_steps: 0,
        };
        let mut half_down = MassSession {
            state: MassState::AwaitingOptimize,
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
            MassErrorFeedbackConfig::Persistent {
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
            "black-hole-mass-path-repair-{}",
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
            "black-hole-mass-path-repair-missing-{}",
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
                capabilities: WorkerCapabilities::default(),
            },
        );
        workers.insert(
            worker_b,
            TunnelWorker {
                token: worker_b,
                worker_id: uuid::Uuid::new_v4(),
                max_instances: Some(2),
                capabilities: WorkerCapabilities::default(),
            },
        );
        let mut routes = HashMap::new();
        routes.insert(model_a, RouteTarget::Local);
        routes.insert(model_b, RouteTarget::Local);
        routes.insert(model_c, RouteTarget::Worker(worker_a));
        routes.insert(model_d, RouteTarget::Worker(worker_b));
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(2),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };

        let out = handle_query_model_capacity(&ctx)
            .await
            .expect("capacity query should succeed");
        let black_hole_type::MassOut::ModelCapacity { capacity } = out else {
            panic!("unexpected query response");
        };
        assert_eq!(
            capacity,
            MassModelCapacity {
                total: Some(7),
                available: Some(3),
                occupied: 4,
                per_architecture: super::COMPILED_ARCHITECTURE
                    .map(|architecture| {
                        vec![(
                            architecture,
                            MassModelCapacity {
                                total: Some(2),
                                available: Some(0),
                                occupied: 2,
                                per_architecture: vec![],
                            },
                        )]
                    })
                    .unwrap_or_default(),
            }
        );
    }

    #[tokio::test]
    async fn query_model_capacity_saturates_available_at_zero() {
        let mut routes = HashMap::new();
        routes.insert(uuid::Uuid::new_v4(), RouteTarget::Local);
        routes.insert(uuid::Uuid::new_v4(), RouteTarget::Local);
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };

        let out = handle_query_model_capacity(&ctx)
            .await
            .expect("capacity query should succeed");
        let black_hole_type::MassOut::ModelCapacity { capacity } = out else {
            panic!("unexpected query response");
        };
        assert_eq!(
            capacity,
            MassModelCapacity {
                total: Some(1),
                available: Some(0),
                occupied: 2,
                per_architecture: super::COMPILED_ARCHITECTURE
                    .map(|architecture| {
                        vec![(
                            architecture,
                            MassModelCapacity {
                                total: Some(1),
                                available: Some(0),
                                occupied: 2,
                                per_architecture: vec![],
                            },
                        )]
                    })
                    .unwrap_or_default(),
            }
        );
    }

    #[tokio::test]
    async fn tunnel_registration_defaults_capacity_to_one_when_omitted() {
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };
        let worker_id = uuid::Uuid::new_v4();

        let out = handle_register_tunnel(worker_id, None, None, None, &ctx)
            .await
            .expect("registration should succeed");
        let token = match out {
            black_hole_type::MassOut::TunnelRegistered { token } => token,
            other => panic!("unexpected registration response: {other:?}"),
        };
        let worker = ctx
            .workers
            .read()
            .await
            .get(&token)
            .cloned()
            .expect("worker should be tracked");
        assert_eq!(worker.worker_id, worker_id);
        assert_eq!(worker.max_instances, Some(DEFAULT_MAX_INSTANCES));
    }

    #[tokio::test]
    async fn tunnel_registration_preserves_explicit_capacity() {
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };
        let worker_id = uuid::Uuid::new_v4();
        let requested = Some(3usize);

        let out = handle_register_tunnel(worker_id, requested, None, None, &ctx)
            .await
            .expect("registration should succeed");
        let token = match out {
            black_hole_type::MassOut::TunnelRegistered { token } => token,
            other => panic!("unexpected registration response: {other:?}"),
        };
        let worker = ctx
            .workers
            .read()
            .await
            .get(&token)
            .cloned()
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
                capabilities: WorkerCapabilities::default(),
            },
        );
        let mut instances = HashMap::new();
        instances.insert(uuid::Uuid::new_v4(), ModelSlot::Starting);
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Tcp,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(instances),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };

        let selected = select_start_target(&ctx, None)
            .await
            .expect("worker should be selected when local start is in progress");
        assert_eq!(selected, RouteTarget::Worker(worker_token));
    }

    fn ctx_with_workers(
        workers: HashMap<Uuid, TunnelWorker>,
        routes: HashMap<Uuid, RouteTarget>,
    ) -> MassContext {
        MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        }
    }

    fn operation_worker(
        token: Uuid,
        operations: Vec<black_hole_type::OperationCapability>,
    ) -> TunnelWorker {
        TunnelWorker {
            token,
            worker_id: Uuid::new_v4(),
            max_instances: Some(1),
            capabilities: WorkerCapabilities {
                architectures: Vec::new(),
                operations,
            },
        }
    }

    #[tokio::test]
    async fn generic_routing_rejects_same_shape_different_contract() {
        let token = Uuid::new_v4();
        let workers = HashMap::from([(
            token,
            operation_worker(token, vec![operation_capability::<FakeOperation>()]),
        )]);
        let ctx = ctx_with_workers(workers, HashMap::new());

        let error =
            select_operation_target(&ctx, &operation_capability::<SameShapeOtherOperation>())
                .await
                .expect_err("shape equality must not imply contract compatibility");
        assert!(matches!(error, ServerError::NoCompatibleOperation { .. }));
    }

    #[tokio::test]
    async fn generic_routing_rejects_contract_version_mismatch() {
        let token = Uuid::new_v4();
        let workers = HashMap::from([(
            token,
            operation_worker(token, vec![operation_capability::<FakeOperation>()]),
        )]);
        let ctx = ctx_with_workers(workers, HashMap::new());

        let error = select_operation_target(&ctx, &operation_capability::<FakeOperationV2>())
            .await
            .expect_err("contract versions must match");
        assert!(matches!(error, ServerError::NoCompatibleOperation { .. }));
    }

    #[tokio::test]
    async fn generic_routing_rejects_unsupported_codec() {
        let token = Uuid::new_v4();
        let workers = HashMap::from([(
            token,
            operation_worker(token, vec![operation_capability::<FakeOperation>()]),
        )]);
        let ctx = ctx_with_workers(workers, HashMap::new());
        let mut requested = operation_capability::<FakeOperation>();
        requested.tensor_encodings.push(EncodingId(99));

        let error = select_operation_target(&ctx, &requested)
            .await
            .expect_err("unsupported codecs must fail closed");
        assert!(matches!(error, ServerError::NoCompatibleOperation { .. }));
    }

    #[tokio::test]
    async fn generic_route_remains_pinned_after_worker_capabilities_change() {
        let token = Uuid::new_v4();
        let workers = HashMap::from([(
            token,
            operation_worker(token, vec![operation_capability::<FakeOperation>()]),
        )]);
        let instance_id = Uuid::new_v4();
        let ctx = ctx_with_workers(workers, HashMap::new());
        let target = select_operation_target(&ctx, &operation_capability::<FakeOperation>())
            .await
            .expect("matching worker should be selected");
        ctx.routes.write().await.insert(instance_id, target);
        ctx.workers
            .write()
            .await
            .get_mut(&token)
            .unwrap()
            .capabilities
            .operations
            .clear();

        assert_eq!(
            route_for_model(instance_id, &ctx).await.unwrap(),
            RouteTarget::Worker(token)
        );
    }

    #[tokio::test]
    async fn tunnel_registration_stores_advertised_capabilities() {
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(DEFAULT_MAX_INSTANCES),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };
        let worker_id = uuid::Uuid::new_v4();
        let capabilities = WorkerCapabilities {
            architectures: vec![MassArchitecture::Qwen38_27b],
            operations: Vec::new(),
        };

        let out = handle_register_tunnel(worker_id, None, Some(capabilities.clone()), None, &ctx)
            .await
            .expect("registration should succeed");
        let token = match out {
            black_hole_type::MassOut::TunnelRegistered { token } => token,
            other => panic!("unexpected registration response: {other:?}"),
        };
        let worker = ctx
            .workers
            .read()
            .await
            .get(&token)
            .cloned()
            .expect("worker should be tracked");
        assert_eq!(worker.capabilities, capabilities);
    }

    #[tokio::test]
    async fn select_start_target_routes_required_architecture_to_matching_worker() {
        // Test builds have COMPILED_ARCHITECTURE == None, so the local engine
        // never satisfies a requirement; only matching workers are eligible.
        let teacher_token = Uuid::new_v4();
        let student_token = Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            teacher_token,
            TunnelWorker {
                token: teacher_token,
                worker_id: Uuid::new_v4(),
                max_instances: Some(1),
                capabilities: WorkerCapabilities {
                    architectures: vec![MassArchitecture::Qwen38_27b],
                    operations: Vec::new(),
                },
            },
        );
        workers.insert(
            student_token,
            TunnelWorker {
                token: student_token,
                worker_id: Uuid::new_v4(),
                max_instances: Some(1),
                capabilities: WorkerCapabilities {
                    architectures: vec![MassArchitecture::Qwen35_0p8b],
                    operations: Vec::new(),
                },
            },
        );
        let ctx = ctx_with_workers(workers, HashMap::new());

        let selected = select_start_target(&ctx, Some(MassArchitecture::Qwen38_27b))
            .await
            .expect("compatible worker should be selected");
        assert_eq!(selected, RouteTarget::Worker(teacher_token));

        let selected = select_start_target(&ctx, Some(MassArchitecture::Qwen35_0p8b))
            .await
            .expect("compatible worker should be selected");
        assert_eq!(
            selected,
            if super::COMPILED_ARCHITECTURE == Some(MassArchitecture::Qwen35_0p8b) {
                RouteTarget::Local
            } else {
                RouteTarget::Worker(student_token)
            }
        );
    }

    #[tokio::test]
    async fn select_start_target_rejects_unknown_architecture() {
        let teacher_token = Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            teacher_token,
            TunnelWorker {
                token: teacher_token,
                worker_id: Uuid::new_v4(),
                max_instances: Some(1),
                capabilities: WorkerCapabilities {
                    architectures: vec![MassArchitecture::Qwen38_27b],
                    operations: Vec::new(),
                },
            },
        );
        let ctx = ctx_with_workers(workers, HashMap::new());

        let error = select_start_target(&ctx, Some(MassArchitecture::Qwen35_2b))
            .await
            .expect_err("no engine serves the required architecture");
        assert!(matches!(
            error,
            ServerError::NoCompatibleTunnelCapacity(MassArchitecture::Qwen35_2b)
        ));
    }

    #[tokio::test]
    async fn select_start_target_prefers_matching_worker_over_local() {
        // Local engine (None in test builds) is ineligible for a required
        // architecture even when it has free capacity.
        let worker_token = Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            worker_token,
            TunnelWorker {
                token: worker_token,
                worker_id: Uuid::new_v4(),
                max_instances: Some(1),
                capabilities: WorkerCapabilities {
                    architectures: vec![MassArchitecture::Qwen35_0p8b],
                    operations: Vec::new(),
                },
            },
        );
        let ctx = ctx_with_workers(workers, HashMap::new());

        let selected = select_start_target(&ctx, Some(MassArchitecture::Qwen35_0p8b))
            .await
            .expect("matching worker should be selected");
        assert_eq!(
            selected,
            if super::COMPILED_ARCHITECTURE == Some(MassArchitecture::Qwen35_0p8b) {
                RouteTarget::Local
            } else {
                RouteTarget::Worker(worker_token)
            }
        );
    }

    #[tokio::test]
    async fn select_start_target_without_requirement_matches_any_engine() {
        // Legacy behavior: a requirement-less start matches engines with no
        // advertised capabilities (local in test builds).
        let worker_token = Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            worker_token,
            TunnelWorker {
                token: worker_token,
                worker_id: uuid::Uuid::new_v4(),
                max_instances: Some(1),
                capabilities: WorkerCapabilities::default(),
            },
        );
        let ctx = ctx_with_workers(workers, HashMap::new());

        // Local is free and eligible, so it wins the tie.
        let selected = select_start_target(&ctx, None)
            .await
            .expect("any engine should be eligible");
        assert_eq!(selected, RouteTarget::Local);
    }

    #[tokio::test]
    async fn handle_start_rejects_architecture_mismatch() {
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(HashMap::new()),
        };
        let model_config = MassModelConfig {
            required_architecture: Some(MassArchitecture::Qwen38_27b),
            ..Default::default()
        };

        let error = handle_start(Uuid::new_v4(), Some(model_config), &ctx)
            .await
            .expect_err("test engine is not compiled for the required architecture");
        assert!(matches!(error, ServerError::ArchitectureMismatch { .. }));
    }

    #[tokio::test]
    async fn query_model_capacity_reports_per_architecture() {
        let teacher_token = Uuid::new_v4();
        let mut workers = HashMap::new();
        workers.insert(
            teacher_token,
            TunnelWorker {
                token: teacher_token,
                worker_id: Uuid::new_v4(),
                max_instances: Some(2),
                capabilities: WorkerCapabilities {
                    architectures: vec![MassArchitecture::Qwen38_27b],
                    operations: Vec::new(),
                },
            },
        );
        let model_id = Uuid::new_v4();
        let mut routes = HashMap::new();
        routes.insert(model_id, RouteTarget::Worker(teacher_token));
        let mut requirements = HashMap::new();
        requirements.insert(model_id, Some(MassArchitecture::Qwen38_27b));
        let ctx = MassContext {
            model_path: PathBuf::from("model-is-not-loaded-for-this-test"),
            transport_mode: TransportMode::Quic,
            void_client: None,
            defaults: MassServerDefaults::default(),
            frozen: false,
            max_instances: Some(1),
            mode: MassMode::Root,
            start_dispatch: Mutex::new(()),
            routes: RwLock::new(routes),
            workers: RwLock::new(workers),
            worker_connections: RwLock::new(HashMap::new()),
            instances: RwLock::new(HashMap::new()),
            operation: None,
            operation_instances: RwLock::new(Default::default()),
            instance_requirements: RwLock::new(requirements),
        };

        let out = handle_query_model_capacity(&ctx)
            .await
            .expect("capacity query should succeed");
        let black_hole_type::MassOut::ModelCapacity { capacity } = out else {
            panic!("unexpected query response");
        };
        let teacher_architecture = MassArchitecture::Qwen38_27b;
        let mut expected = Vec::new();
        if let Some(local_architecture) = super::COMPILED_ARCHITECTURE {
            expected.push((
                local_architecture,
                MassModelCapacity {
                    total: Some(if local_architecture == teacher_architecture {
                        3
                    } else {
                        1
                    }),
                    available: Some(if local_architecture == teacher_architecture {
                        2
                    } else {
                        1
                    }),
                    occupied: usize::from(local_architecture == teacher_architecture),
                    per_architecture: vec![],
                },
            ));
        }
        if super::COMPILED_ARCHITECTURE != Some(teacher_architecture) {
            expected.push((
                teacher_architecture,
                MassModelCapacity {
                    total: Some(2),
                    available: Some(1),
                    occupied: 1,
                    per_architecture: vec![],
                },
            ));
        }
        assert_eq!(capacity.per_architecture, expected);
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
