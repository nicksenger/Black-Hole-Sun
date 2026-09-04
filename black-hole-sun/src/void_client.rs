use std::{
    fs,
    net::SocketAddr,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use black_hole_flux::ops::VoidOps;
use black_hole_type::{
    ArtifactRef, ContractDescriptor, DurabilityPolicy, ObjectId, StreamRef, TensorEnvelope,
    TransferBegin, TransferChunk, TransferHash, TransferRecord, TransferRef, TransferStreamFrame,
    TransferTicket, TRANSFER_PROTOCOL_VERSION,
};
use black_hole_void::{VoidIn, VoidOut};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};

#[derive(Clone, Debug)]
enum VoidTransport {
    Quic {
        endpoint: quinn::Endpoint,
        addr: SocketAddr,
        server_name: String,
    },
    Tcp {
        addr: SocketAddr,
    },
}

/// Chunk size for streaming void transfers (must fit within one frame on
/// both ends; mass caps frames at 64 MB).
const VOID_CHUNK_SIZE: usize = 16 * 1024 * 1024; // 16 MB

/// Client for interacting with the Void service.
#[derive(Clone, Debug)]
pub struct VoidClient {
    transport: VoidTransport,
}

/// A live stream location that can be shared with consumers before the
/// producer starts sending bytes.
pub struct PreparedTransfer<T> {
    pub artifact: ArtifactRef<T>,
    pub ticket: TransferTicket,
    begin: TransferBegin,
}

impl VoidClient {
    pub fn new(
        endpoint: &quinn::Endpoint,
        addr: SocketAddr,
        server_name: impl Into<String>,
    ) -> Self {
        Self {
            transport: VoidTransport::Quic {
                endpoint: endpoint.clone(),
                addr,
                server_name: server_name.into(),
            },
        }
    }

    pub fn new_tcp(addr: SocketAddr) -> Self {
        Self {
            transport: VoidTransport::Tcp { addr },
        }
    }

    pub async fn upload(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        let resp = self.request(&VoidIn::Upload { data }).await?;
        match resp {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for upload".to_string()),
        }
    }

    pub async fn upload_with(&self, id: ObjectId, data: Vec<u8>) -> Result<ObjectId, String> {
        let resp = self.request(&VoidIn::UploadWith { id, data }).await?;
        match resp {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for upload_with".to_string()),
        }
    }

    pub async fn download(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        let resp = self.request(&VoidIn::Download { id }).await?;
        match resp {
            VoidOut::Downloaded { data } => Ok(data),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for download".to_string()),
        }
    }

    pub async fn download_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        let resp = self
            .request(&VoidIn::DownloadWait { id, timeout_ms })
            .await?;
        match resp {
            VoidOut::Downloaded { data } => Ok(Some(data)),
            VoidOut::TimedOut { .. } => Ok(None),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for download_wait".to_string()),
        }
    }

    /// Begin a replayable progressive transfer.
    pub async fn begin_transfer<T>(&self, begin: TransferBegin) -> Result<TransferRef<T>, String> {
        let transfer_id = begin.transfer_id;
        match self.request(&VoidIn::TransferBegin { begin }).await? {
            VoidOut::Transfer {
                record: TransferRecord::InProgress { .. },
            } => Ok(TransferRef::new(transfer_id)),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for transfer begin".to_string()),
        }
    }

    /// Persist one independently readable transfer chunk.
    pub async fn upload_transfer_chunk(
        &self,
        transfer_id: ObjectId,
        index: u32,
        data: Vec<u8>,
        hash: TransferHash,
    ) -> Result<TransferChunk, String> {
        match self
            .request(&VoidIn::TransferChunk {
                transfer_id,
                index,
                data,
                hash,
            })
            .await?
        {
            VoidOut::TransferChunkStored { chunk } => Ok(chunk),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for transfer chunk".to_string()),
        }
    }

    /// Inspect current transfer state. Receivers can download every listed
    /// chunk immediately, before commit.
    pub async fn inspect_transfer(&self, transfer_id: ObjectId) -> Result<TransferRecord, String> {
        match self
            .request(&VoidIn::TransferInspect { transfer_id })
            .await?
        {
            VoidOut::Transfer { record } => Ok(record),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for transfer inspect".to_string()),
        }
    }

    pub async fn commit_transfer<T>(
        &self,
        reference: TransferRef<T>,
        aggregate_hash: TransferHash,
    ) -> Result<ArtifactRef<T>, String> {
        match self
            .request(&VoidIn::TransferCommit {
                transfer_id: reference.id(),
                aggregate_hash,
            })
            .await?
        {
            VoidOut::Transfer {
                record: TransferRecord::Committed(_),
            } => Ok(ArtifactRef::transfer(reference)),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for transfer commit".to_string()),
        }
    }

    pub async fn abort_transfer(
        &self,
        transfer_id: ObjectId,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        match self
            .request(&VoidIn::TransferAbort {
                transfer_id,
                reason: reason.into(),
            })
            .await?
        {
            VoidOut::Transfer {
                record: TransferRecord::Aborted(_),
            } => Ok(()),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for transfer abort".to_string()),
        }
    }

    /// Create and persist a live-stream ticket. The returned artifact can be
    /// sent through a Flow immediately; its durable fallback becomes
    /// authoritative only after `stream_transfer` commits.
    pub async fn prepare_stream_transfer<T>(
        &self,
        descriptor: ContractDescriptor,
        envelope: TensorEnvelope,
        tensor_header: Vec<u8>,
        expected_len: u64,
        expected_hash: TransferHash,
        expected_chunks: u32,
        lease: Duration,
        durability: DurabilityPolicy,
    ) -> Result<PreparedTransfer<T>, String> {
        if descriptor.id != envelope.contract_id
            || descriptor.version != envelope.contract_version
            || black_hole_spec::descriptor_hash(&descriptor) != envelope.contract_hash
        {
            return Err("transfer descriptor does not match the tensor envelope".to_string());
        }
        let declared_len = black_hole_spec::validate_tensor_stream_header(
            &descriptor,
            envelope.side,
            &tensor_header,
        )
        .map_err(|error| format!("invalid tensor stream header: {error}"))?;
        if declared_len != expected_len {
            return Err(format!(
                "tensor stream header declares {declared_len} bytes, but transfer declares {expected_len}"
            ));
        }
        if lease.is_zero() {
            return Err("transfer lease must be non-zero".to_string());
        }
        let transfer_id = ObjectId::new_v4();
        let authorization = random_authorization();
        let deadline_unix_ms =
            unix_time_ms().saturating_add(lease.as_millis().try_into().unwrap_or(u64::MAX));
        let begin = TransferBegin {
            protocol_version: TRANSFER_PROTOCOL_VERSION,
            transfer_id,
            envelope: envelope.clone(),
            tensor_header: tensor_header.clone(),
            expected_chunks,
            expected_len,
            expected_hash,
            deadline_unix_ms,
            authorization_hash: hash_bytes(&authorization),
        };
        let ticket = TransferTicket {
            descriptor,
            envelope,
            tensor_header,
            transfer_id,
            source: self.source_authority(),
            authorization,
            expected_len,
            expected_hash,
            deadline_unix_ms,
            durability,
            eventual_void_id: transfer_id,
        };
        let ticket_id = self
            .upload(postcard::to_allocvec(&ticket).map_err(|error| error.to_string())?)
            .await?;
        Ok(PreparedTransfer {
            artifact: ArtifactRef::stream(StreamRef::new(ticket_id, transfer_id)),
            ticket,
            begin,
        })
    }

    /// Send a tensor artifact over one backpressured QUIC/TCP channel while
    /// Void persists each frame as an immutable replay chunk.
    pub async fn stream_transfer<T>(
        &self,
        prepared: &PreparedTransfer<T>,
        chunks: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<ArtifactRef<T>, String> {
        match &self.transport {
            VoidTransport::Quic {
                endpoint,
                addr,
                server_name,
            } => {
                let conn = endpoint
                    .connect(*addr, server_name)
                    .map_err(|error| format!("connect init failed: {error}"))?
                    .await
                    .map_err(|error| format!("connect failed: {error}"))?;
                let (mut send, mut recv) = conn
                    .open_bi()
                    .await
                    .map_err(|error| format!("open_bi failed: {error}"))?;
                send_frame_quic(
                    &mut send,
                    &VoidIn::TransferStreamUpload {
                        begin: prepared.begin.clone(),
                        authorization: prepared.ticket.authorization,
                    },
                )
                .await?;
                expect_transfer_started(read_frame_quic(&mut recv).await?)?;
                send_stream_chunks_quic(&mut send, &mut recv, prepared, chunks.into_iter()).await
            }
            VoidTransport::Tcp { addr } => {
                let mut stream = TcpStream::connect(*addr)
                    .await
                    .map_err(|error| format!("tcp connect failed: {error}"))?;
                send_frame_io(
                    &mut stream,
                    &VoidIn::TransferStreamUpload {
                        begin: prepared.begin.clone(),
                        authorization: prepared.ticket.authorization,
                    },
                )
                .await?;
                expect_transfer_started(read_frame_io(&mut stream).await?)?;
                send_stream_chunks_io(&mut stream, prepared, chunks.into_iter()).await
            }
        }
    }

    /// Receive progressively over the live channel. If that channel is
    /// interrupted, resolution waits for and validates the committed Void
    /// manifest instead.
    pub async fn receive_stream(&self, ticket: &TransferTicket) -> Result<Vec<u8>, String> {
        validate_ticket(ticket)?;
        let live = match &self.transport {
            VoidTransport::Quic {
                endpoint,
                addr: _,
                server_name: _,
            } => {
                let result = async {
                    let source_addr: SocketAddr = ticket
                        .source
                        .parse()
                        .map_err(|error| format!("invalid transfer source: {error}"))?;
                    let conn = endpoint
                        .connect(source_addr, &source_addr.ip().to_string())
                        .map_err(|error| format!("connect init failed: {error}"))?
                        .await
                        .map_err(|error| format!("connect failed: {error}"))?;
                    let (mut send, mut recv) = conn
                        .open_bi()
                        .await
                        .map_err(|error| format!("open_bi failed: {error}"))?;
                    send_frame_quic(
                        &mut send,
                        &VoidIn::TransferStreamDownload {
                            transfer_id: ticket.transfer_id,
                            authorization: ticket.authorization,
                        },
                    )
                    .await?;
                    receive_stream_frames_quic(&mut recv, ticket).await
                }
                .await;
                result
            }
            VoidTransport::Tcp { addr: _ } => {
                let result = async {
                    let source_addr: SocketAddr = ticket
                        .source
                        .parse()
                        .map_err(|error| format!("invalid transfer source: {error}"))?;
                    let mut stream = TcpStream::connect(source_addr)
                        .await
                        .map_err(|error| format!("tcp connect failed: {error}"))?;
                    send_frame_io(
                        &mut stream,
                        &VoidIn::TransferStreamDownload {
                            transfer_id: ticket.transfer_id,
                            authorization: ticket.authorization,
                        },
                    )
                    .await?;
                    receive_stream_frames_io(&mut stream, ticket).await
                }
                .await;
                result
            }
        };
        match live {
            Ok(bytes) => validate_received_tensor(ticket, bytes),
            Err(live_error) if ticket.durability == DurabilityPolicy::ReplayRequired => self
                .wait_for_committed_transfer(ticket)
                .await
                .and_then(|bytes| validate_received_tensor(ticket, bytes))
                .map_err(|fallback_error| {
                    format!(
                        "live stream failed ({live_error}); durable fallback failed ({fallback_error})"
                    )
                }),
            Err(error) => Err(error),
        }
    }

    /// Resolve raw bytes from any typed artifact location. Live references
    /// fetch and authenticate their ticket, consume the progressive stream,
    /// and retain the committed transfer as their replay fallback.
    pub async fn receive_artifact<T>(&self, reference: &ArtifactRef<T>) -> Result<Vec<u8>, String> {
        match reference {
            ArtifactRef::Committed(reference) => self.download(reference.id()).await,
            ArtifactRef::Transfer(reference) => {
                VoidOps::resolve_transfer_raw(self, reference.id()).await
            }
            ArtifactRef::Stream(reference) => {
                let ticket_bytes = self.download(reference.ticket_id).await?;
                let ticket: TransferTicket = postcard::from_bytes(&ticket_bytes)
                    .map_err(|error| format!("invalid transfer ticket: {error}"))?;
                if ticket.transfer_id != reference.fallback_transfer_id
                    || ticket.eventual_void_id != reference.fallback_transfer_id
                {
                    return Err(
                        "transfer ticket does not match the artifact's durable fallback".into(),
                    );
                }
                self.receive_stream(&ticket).await
            }
        }
    }

    fn source_authority(&self) -> String {
        match &self.transport {
            VoidTransport::Quic { addr, .. } | VoidTransport::Tcp { addr } => addr.to_string(),
        }
    }

    async fn wait_for_committed_transfer(
        &self,
        ticket: &TransferTicket,
    ) -> Result<Vec<u8>, String> {
        loop {
            match VoidOps::resolve_transfer_raw(self, ticket.transfer_id).await {
                Ok(bytes) => return Ok(bytes),
                Err(error) if unix_time_ms() < ticket.deadline_unix_ms => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    let _ = error;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Upload a local file to void. Files that fit in one frame use the
    /// single-shot upload; larger files are streamed as a chunked multipart
    /// upload. Returns the assigned object ID.
    pub async fn upload_file(&self, path: &Path) -> Result<ObjectId, String> {
        let size = fs::metadata(path)
            .map_err(|e| format!("failed to read file metadata for {}: {e}", path.display()))?;

        if size.len() <= 64 * 1024 * 1024 {
            let data = fs::read(path)
                .map_err(|e| format!("failed to read file {}: {e}", path.display()))?;
            return self.upload(data).await;
        }

        let id = match self
            .request(&VoidIn::UploadBegin {
                id: None,
                total_size: size.len() as u64,
            })
            .await?
        {
            VoidOut::Uploaded { id } => id,
            VoidOut::Error { message } => return Err(message),
            _ => return Err("unexpected void response for upload_begin".to_string()),
        };

        let mut file = tokio::fs::File::open(path)
            .await
            .map_err(|e| format!("failed to open file {}: {e}", path.display()))?;
        let mut part_number: u32 = 1;
        loop {
            let mut buffer = vec![0u8; VOID_CHUNK_SIZE];
            let read = file
                .read(&mut buffer)
                .await
                .map_err(|e| format!("failed to read file {}: {e}", path.display()))?;
            if read == 0 {
                break;
            }
            buffer.truncate(read);
            match self
                .request(&VoidIn::UploadPart {
                    id,
                    part_number,
                    data: buffer,
                })
                .await?
            {
                VoidOut::Ack => {}
                VoidOut::Error { message } => return Err(message),
                _ => return Err("unexpected void response for upload_part".to_string()),
            }
            part_number += 1;
        }

        let part_count = part_number - 1;
        match self
            .request(&VoidIn::UploadFinish { id, part_count })
            .await?
        {
            VoidOut::Uploaded { id } => Ok(id),
            VoidOut::Error { message } => Err(message),
            _ => Err("unexpected void response for upload_finish".to_string()),
        }
    }

    /// Download an object from void directly to a local file using ranged
    /// reads, so arbitrarily large objects never need to fit in one frame.
    /// Returns the number of bytes written.
    pub async fn download_to_file(&self, id: ObjectId, path: &Path) -> Result<u64, String> {
        let mut file = tokio::fs::File::create(path)
            .await
            .map_err(|e| format!("failed to create file {}: {e}", path.display()))?;
        let mut offset: u64 = 0;
        loop {
            let data = match self
                .request(&VoidIn::DownloadRange {
                    id,
                    offset,
                    length: VOID_CHUNK_SIZE as u64,
                })
                .await?
            {
                VoidOut::Downloaded { data } => data,
                VoidOut::Error { message } => return Err(message),
                _ => return Err("unexpected void response for download_range".to_string()),
            };
            if data.is_empty() {
                break;
            }
            file.write_all(&data)
                .await
                .map_err(|e| format!("failed to write file {}: {e}", path.display()))?;
            offset += data.len() as u64;
            if data.len() < VOID_CHUNK_SIZE {
                break;
            }
        }
        Ok(offset)
    }

    async fn request(&self, request: &VoidIn) -> Result<VoidOut, String> {
        match &self.transport {
            VoidTransport::Quic {
                endpoint,
                addr,
                server_name,
            } => {
                let connecting = endpoint
                    .connect(*addr, server_name)
                    .map_err(|e| format!("connect init failed: {e}"))?;
                let conn = connecting
                    .await
                    .map_err(|e| format!("connect failed: {e}"))?;
                let (mut send, mut recv) = conn
                    .open_bi()
                    .await
                    .map_err(|e| format!("open_bi failed: {e}"))?;

                send_frame_quic(&mut send, request).await?;
                read_frame_quic(&mut recv).await
            }
            VoidTransport::Tcp { addr } => {
                let mut stream = TcpStream::connect(*addr)
                    .await
                    .map_err(|e| format!("tcp connect failed: {e}"))?;
                send_frame_io(&mut stream, request).await?;
                read_frame_io(&mut stream).await
            }
        }
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

fn hash_bytes(data: &[u8]) -> TransferHash {
    TransferHash(Sha256::digest(data).into())
}

fn random_authorization() -> [u8; 32] {
    let left = ObjectId::new_v4();
    let right = ObjectId::new_v4();
    let mut authorization = [0; 32];
    authorization[..16].copy_from_slice(left.as_bytes());
    authorization[16..].copy_from_slice(right.as_bytes());
    authorization
}

fn expect_transfer_started(response: VoidOut) -> Result<(), String> {
    match response {
        VoidOut::Transfer {
            record: TransferRecord::InProgress { .. },
        } => Ok(()),
        VoidOut::Error { message } => Err(message),
        _ => Err("unexpected response while opening transfer stream".to_string()),
    }
}

async fn send_stream_chunks_quic<T, I>(
    send: &mut quinn::SendStream,
    recv: &mut quinn::RecvStream,
    prepared: &PreparedTransfer<T>,
    chunks: I,
) -> Result<ArtifactRef<T>, String>
where
    I: Iterator<Item = Vec<u8>>,
{
    let mut aggregate = Sha256::new();
    let mut actual_len = 0u64;
    let mut actual_chunks = 0u32;
    for data in chunks {
        let index = actual_chunks;
        actual_chunks = actual_chunks
            .checked_add(1)
            .ok_or_else(|| "transfer has too many chunks".to_string())?;
        actual_len = actual_len.saturating_add(data.len() as u64);
        aggregate.update(&data);
        let hash = hash_bytes(&data);
        send_frame_quic(send, &TransferStreamFrame::Chunk { index, data, hash }).await?;
    }
    let actual_hash = TransferHash(aggregate.finalize().into());
    if actual_chunks != prepared.begin.expected_chunks
        || actual_len != prepared.begin.expected_len
        || actual_hash != prepared.begin.expected_hash
    {
        send_frame_quic(
            send,
            &TransferStreamFrame::Abort {
                reason: "stream chunks did not match the transfer declaration".to_string(),
            },
        )
        .await?;
        let _ = read_frame_quic::<VoidOut>(recv).await;
        return Err("stream chunks do not match the prepared transfer declaration".to_string());
    }
    send_frame_quic(
        send,
        &TransferStreamFrame::Commit {
            aggregate_hash: prepared.begin.expected_hash,
        },
    )
    .await?;
    match read_frame_quic(recv).await? {
        VoidOut::Transfer {
            record: TransferRecord::Committed(_),
        } => Ok(prepared.artifact),
        VoidOut::Error { message } => Err(message),
        _ => Err("unexpected response while committing transfer stream".to_string()),
    }
}

async fn send_stream_chunks_io<T, I, S>(
    stream: &mut S,
    prepared: &PreparedTransfer<T>,
    chunks: I,
) -> Result<ArtifactRef<T>, String>
where
    I: Iterator<Item = Vec<u8>>,
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut aggregate = Sha256::new();
    let mut actual_len = 0u64;
    let mut actual_chunks = 0u32;
    for data in chunks {
        let index = actual_chunks;
        actual_chunks = actual_chunks
            .checked_add(1)
            .ok_or_else(|| "transfer has too many chunks".to_string())?;
        actual_len = actual_len.saturating_add(data.len() as u64);
        aggregate.update(&data);
        let hash = hash_bytes(&data);
        send_frame_io(stream, &TransferStreamFrame::Chunk { index, data, hash }).await?;
    }
    let actual_hash = TransferHash(aggregate.finalize().into());
    if actual_chunks != prepared.begin.expected_chunks
        || actual_len != prepared.begin.expected_len
        || actual_hash != prepared.begin.expected_hash
    {
        send_frame_io(
            stream,
            &TransferStreamFrame::Abort {
                reason: "stream chunks did not match the transfer declaration".to_string(),
            },
        )
        .await?;
        let _ = read_frame_io::<_, VoidOut>(stream).await;
        return Err("stream chunks do not match the prepared transfer declaration".to_string());
    }
    send_frame_io(
        stream,
        &TransferStreamFrame::Commit {
            aggregate_hash: prepared.begin.expected_hash,
        },
    )
    .await?;
    match read_frame_io(stream).await? {
        VoidOut::Transfer {
            record: TransferRecord::Committed(_),
        } => Ok(prepared.artifact),
        VoidOut::Error { message } => Err(message),
        _ => Err("unexpected response while committing transfer stream".to_string()),
    }
}

fn validate_stream_begin(begin: &TransferBegin, ticket: &TransferTicket) -> Result<(), String> {
    validate_ticket(ticket)?;
    if begin.transfer_id != ticket.transfer_id
        || begin.envelope != ticket.envelope
        || begin.tensor_header != ticket.tensor_header
        || begin.expected_len != ticket.expected_len
        || begin.expected_hash != ticket.expected_hash
        || begin.deadline_unix_ms != ticket.deadline_unix_ms
        || begin.authorization_hash != hash_bytes(&ticket.authorization)
    {
        return Err("live stream begin frame does not match its ticket".to_string());
    }
    Ok(())
}

fn validate_ticket(ticket: &TransferTicket) -> Result<(), String> {
    if ticket.descriptor.id != ticket.envelope.contract_id
        || ticket.descriptor.version != ticket.envelope.contract_version
        || black_hole_spec::descriptor_hash(&ticket.descriptor) != ticket.envelope.contract_hash
    {
        return Err("transfer ticket descriptor does not match its tensor envelope".to_string());
    }
    if ticket.eventual_void_id != ticket.transfer_id {
        return Err("transfer ticket durable object does not match its transfer ID".to_string());
    }
    let declared_len = black_hole_spec::validate_tensor_stream_header(
        &ticket.descriptor,
        ticket.envelope.side,
        &ticket.tensor_header,
    )
    .map_err(|error| format!("invalid tensor stream header: {error}"))?;
    if declared_len != ticket.expected_len {
        return Err(format!(
            "tensor stream header declares {declared_len} bytes, but ticket declares {}",
            ticket.expected_len
        ));
    }
    Ok(())
}

fn validate_received_tensor(ticket: &TransferTicket, bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if bytes.len() as u64 != ticket.expected_len
        || hash_bytes(&bytes) != ticket.expected_hash
        || !bytes.starts_with(&ticket.tensor_header)
    {
        return Err("received tensor does not match its transfer ticket".to_string());
    }
    black_hole_spec::validate_artifact(&ticket.descriptor, ticket.envelope.side, &bytes)
        .map_err(|error| format!("received tensor payload is invalid: {error}"))?;
    Ok(bytes)
}

fn apply_incoming_frame(
    frame: TransferStreamFrame,
    ticket: &TransferTicket,
    begin: &mut Option<TransferBegin>,
    next_index: &mut u32,
    aggregate: &mut Sha256,
    output: &mut Vec<u8>,
) -> Result<bool, String> {
    match frame {
        TransferStreamFrame::Begin(received) => {
            if begin.is_some() {
                return Err("duplicate begin frame on transfer stream".to_string());
            }
            validate_stream_begin(&received, ticket)?;
            *begin = Some(received);
            Ok(false)
        }
        TransferStreamFrame::Chunk { index, data, hash } => {
            if begin.is_none() {
                return Err("transfer chunk arrived before begin frame".to_string());
            }
            if index != *next_index {
                return Err(format!(
                    "transfer stream chunk {index} arrived out of order; expected {}",
                    *next_index
                ));
            }
            if hash_bytes(&data) != hash {
                return Err(format!("transfer stream chunk {index} failed validation"));
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
                return Err(format!(
                    "transfer stream chunk {index} does not match the authenticated tensor header"
                ));
            }
            *next_index = next_index.saturating_add(1);
            aggregate.update(&data);
            output.extend_from_slice(&data);
            Ok(false)
        }
        TransferStreamFrame::Commit { aggregate_hash } => {
            let begin = begin
                .as_ref()
                .ok_or_else(|| "transfer committed before begin frame".to_string())?;
            let actual_hash = TransferHash(aggregate.clone().finalize().into());
            if *next_index != begin.expected_chunks
                || output.len() as u64 != begin.expected_len
                || aggregate_hash != begin.expected_hash
                || actual_hash != aggregate_hash
            {
                return Err("committed transfer stream failed aggregate validation".to_string());
            }
            Ok(true)
        }
        TransferStreamFrame::Abort { reason } => Err(format!("transfer aborted: {reason}")),
    }
}

async fn receive_stream_frames_quic(
    recv: &mut quinn::RecvStream,
    ticket: &TransferTicket,
) -> Result<Vec<u8>, String> {
    let mut begin = None;
    let mut next_index = 0;
    let mut aggregate = Sha256::new();
    let mut output = Vec::new();
    loop {
        let frame = read_frame_quic(recv).await?;
        if apply_incoming_frame(
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

async fn receive_stream_frames_io<S>(
    stream: &mut S,
    ticket: &TransferTicket,
) -> Result<Vec<u8>, String>
where
    S: AsyncRead + Unpin,
{
    let mut begin = None;
    let mut next_index = 0;
    let mut aggregate = Sha256::new();
    let mut output = Vec::new();
    loop {
        let frame = read_frame_io(stream).await?;
        if apply_incoming_frame(
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

#[async_trait::async_trait]
impl VoidOps for VoidClient {
    async fn download_raw(&self, id: ObjectId) -> Result<Vec<u8>, String> {
        self.download(id).await
    }

    async fn download_raw_wait(
        &self,
        id: ObjectId,
        timeout_ms: u64,
    ) -> Result<Option<Vec<u8>>, String> {
        self.download_wait(id, timeout_ms).await
    }

    async fn upload_to_void(&self, data: Vec<u8>) -> Result<ObjectId, String> {
        self.upload(data).await
    }

    async fn upload_to_void_with(&self, id: ObjectId, data: Vec<u8>) -> Result<(), String> {
        self.upload_with(id, data).await.map(|_| ())
    }
}

async fn send_frame_quic(send: &mut quinn::SendStream, msg: &impl Serialize) -> Result<(), String> {
    let payload = to_allocvec(msg).map_err(|e| format!("failed to encode frame: {e}"))?;
    let len = u32::try_from(payload.len()).map_err(|_| "frame too large".to_string())?;
    send.write_all(&len.to_be_bytes())
        .await
        .map_err(|e| format!("failed to write frame len: {e}"))?;
    send.write_all(&payload)
        .await
        .map_err(|e| format!("failed to write frame payload: {e}"))?;
    Ok(())
}

async fn read_frame_quic<T: for<'de> Deserialize<'de>>(
    recv: &mut quinn::RecvStream,
) -> Result<T, String> {
    let len = recv
        .read_u32()
        .await
        .map_err(|e| format!("failed to read frame len: {e}"))? as usize;
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| format!("failed to read frame payload: {e}"))?;
    from_bytes(&payload).map_err(|e| format!("failed to decode frame: {e}"))
}

async fn send_frame_io<W>(send: &mut W, msg: &impl Serialize) -> Result<(), String>
where
    W: AsyncWrite + Unpin,
{
    let payload = to_allocvec(msg).map_err(|e| format!("failed to encode frame: {e}"))?;
    let len = u32::try_from(payload.len()).map_err(|_| "frame too large".to_string())?;
    send.write_all(&len.to_be_bytes())
        .await
        .map_err(|e| format!("failed to write frame len: {e}"))?;
    send.write_all(&payload)
        .await
        .map_err(|e| format!("failed to write frame payload: {e}"))?;
    Ok(())
}

async fn read_frame_io<R, T>(recv: &mut R) -> Result<T, String>
where
    R: AsyncRead + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let len = recv
        .read_u32()
        .await
        .map_err(|e| format!("failed to read frame len: {e}"))? as usize;
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| format!("failed to read frame payload: {e}"))?;
    from_bytes(&payload).map_err(|e| format!("failed to decode frame: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use black_hole_spec::{
        decode_input, encode_input,
        glowstick::{Dyn, Shape1},
        tensor_stream_header, RawTensor, SingleTensorSpec, TensorContract, TensorPortSpec,
    };
    use black_hole_type::{
        ContractId, ContractSide, DimensionDescriptor, DtypeConstraint, EncodingId, TensorDtype,
        TransferRecord,
    };

    struct ByteAxis;
    struct BytePort;

    impl TensorPortSpec for BytePort {
        type Shape = Shape1<Dyn<ByteAxis>>;

        const NAME: &'static str = "bytes";

        fn dimensions() -> Vec<DimensionDescriptor> {
            vec![DimensionDescriptor::Dynamic]
        }

        fn dtype() -> DtypeConstraint {
            DtypeConstraint::Exact(TensorDtype::U8)
        }
    }

    struct StreamContract;

    impl TensorContract for StreamContract {
        type Input = SingleTensorSpec<BytePort>;
        type Output = SingleTensorSpec<BytePort>;
        type Metadata = ();

        const ID: ContractId = ContractId::from_u128(8);
        const VERSION: u32 = 1;
    }

    struct RunningVoid {
        client: VoidClient,
        handle: tokio::task::JoinHandle<black_hole_void::Result<()>>,
    }

    impl Drop for RunningVoid {
        fn drop(&mut self) {
            self.handle.abort();
        }
    }

    async fn running_void() -> RunningVoid {
        let (addr, handle) = black_hole_void::ServerBuilder::new(
            Box::new(black_hole_void::object_store::InMemoryObjectStore::new()),
            Box::new(black_hole_void::persist::InMemoryStore::new()),
        )
        .tcp()
        .listen("127.0.0.1:0".parse().unwrap())
        .serve()
        .await
        .unwrap();
        RunningVoid {
            client: VoidClient::new_tcp(addr),
            handle,
        }
    }

    fn envelope(tensor_len: u64) -> TensorEnvelope {
        let descriptor = descriptor();
        TensorEnvelope {
            envelope_version: TensorEnvelope::VERSION,
            contract_id: descriptor.id,
            contract_version: descriptor.version,
            contract_hash: black_hole_spec::descriptor_hash(&descriptor),
            side: ContractSide::Input,
            tensor_encoding: EncodingId::SAFETENSORS_V1,
            metadata_encoding: EncodingId::POSTCARD_V1,
            metadata_len: 0,
            tensor_len,
        }
    }

    fn descriptor() -> ContractDescriptor {
        ContractDescriptor {
            id: ContractId::from_u128(7),
            version: 1,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn begin(data: &[u8], expected_chunks: u32, lease_ms: u64) -> TransferBegin {
        let authorization = [4; 32];
        TransferBegin {
            protocol_version: TRANSFER_PROTOCOL_VERSION,
            transfer_id: ObjectId::new_v4(),
            envelope: envelope(data.len() as u64),
            tensor_header: b"safetensors-header".to_vec(),
            expected_chunks,
            expected_len: data.len() as u64,
            expected_hash: hash_bytes(data),
            deadline_unix_ms: unix_time_ms().saturating_add(lease_ms),
            authorization_hash: hash_bytes(&authorization),
        }
    }

    fn tensor_frame(payload: &[u8]) -> Vec<u8> {
        encode_input::<StreamContract>(
            &[RawTensor {
                name: "bytes".to_string(),
                dtype: TensorDtype::U8,
                shape: vec![payload.len()],
                data: payload.to_vec(),
            }],
            &(),
        )
        .unwrap()
    }

    fn stream_declaration(frame: &[u8], expected_chunks: u32, lease_ms: u64) -> TransferBegin {
        let authorization = [4; 32];
        TransferBegin {
            protocol_version: TRANSFER_PROTOCOL_VERSION,
            transfer_id: ObjectId::new_v4(),
            envelope: decode_input::<StreamContract>(frame).unwrap().envelope,
            tensor_header: tensor_stream_header(frame).unwrap(),
            expected_chunks,
            expected_len: frame.len() as u64,
            expected_hash: hash_bytes(frame),
            deadline_unix_ms: unix_time_ms().saturating_add(lease_ms),
            authorization_hash: hash_bytes(&authorization),
        }
    }

    fn ticket_for(client: &VoidClient, declaration: &TransferBegin) -> TransferTicket {
        TransferTicket {
            descriptor: StreamContract::descriptor(),
            envelope: declaration.envelope.clone(),
            tensor_header: declaration.tensor_header.clone(),
            transfer_id: declaration.transfer_id,
            source: client.source_authority(),
            authorization: [4; 32],
            expected_len: declaration.expected_len,
            expected_hash: declaration.expected_hash,
            deadline_unix_ms: declaration.deadline_unix_ms,
            durability: DurabilityPolicy::ReplayRequired,
            eventual_void_id: declaration.transfer_id,
        }
    }

    #[tokio::test]
    async fn chunks_are_stageable_but_not_authoritative_before_commit() {
        let running = running_void().await;
        let client = &running.client;
        let all = b"header-payload";
        let declaration = begin(all, 2, 5_000);
        let transfer = client
            .begin_transfer::<Vec<u8>>(declaration.clone())
            .await
            .unwrap();

        let first = all[..6].to_vec();
        let first_descriptor = client
            .upload_transfer_chunk(transfer.id(), 0, first.clone(), hash_bytes(&first))
            .await
            .unwrap();
        assert_eq!(
            client.download(first_descriptor.object_id).await.unwrap(),
            first
        );
        assert!(client
            .upload_with(first_descriptor.object_id, b"overwrite".to_vec())
            .await
            .unwrap_err()
            .contains("transfer"));
        let record = client.inspect_transfer(transfer.id()).await.unwrap();
        assert!(matches!(
            record,
            TransferRecord::InProgress {
                ref chunks,
                revision: 1,
                ..
            } if chunks.len() == 1
        ));
        assert!(VoidOps::resolve_transfer_raw(client, transfer.id())
            .await
            .unwrap_err()
            .contains("not committed"));

        let duplicate = client
            .upload_transfer_chunk(transfer.id(), 0, first.clone(), hash_bytes(&first))
            .await
            .unwrap_err();
        assert!(duplicate.contains("duplicate"));

        let second = all[6..].to_vec();
        let corrupt = client
            .upload_transfer_chunk(transfer.id(), 1, second.clone(), TransferHash([0; 32]))
            .await
            .unwrap_err();
        assert!(corrupt.contains("hash mismatch"));
        client
            .upload_transfer_chunk(transfer.id(), 1, second.clone(), hash_bytes(&second))
            .await
            .unwrap();

        let artifact = client
            .commit_transfer(transfer, declaration.expected_hash)
            .await
            .unwrap();
        assert_eq!(
            VoidOps::resolve_transfer_raw(client, artifact.durable_id())
                .await
                .unwrap(),
            all
        );
        assert!(matches!(
            client
                .inspect_transfer(declaration.transfer_id)
                .await
                .unwrap(),
            TransferRecord::Committed(_)
        ));
        assert!(client
            .upload_with(declaration.transfer_id, b"rewrite-manifest".to_vec())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn missing_chunks_abort_and_expiry_never_replay() {
        let running = running_void().await;
        let client = &running.client;
        let mut wrong_version = begin(b"x", 1, 5_000);
        wrong_version.protocol_version += 1;
        assert!(client
            .begin_transfer::<Vec<u8>>(wrong_version)
            .await
            .unwrap_err()
            .contains("protocol version"));

        let data = b"two-parts";
        let declaration = begin(data, 2, 5_000);
        let transfer = client
            .begin_transfer::<Vec<u8>>(declaration.clone())
            .await
            .unwrap();
        let first = data[..3].to_vec();
        let first_descriptor = client
            .upload_transfer_chunk(transfer.id(), 0, first.clone(), hash_bytes(&first))
            .await
            .unwrap();
        let missing = client
            .commit_transfer(transfer, declaration.expected_hash)
            .await
            .unwrap_err();
        assert!(missing.contains("expected exactly"));

        client
            .abort_transfer(declaration.transfer_id, "producer cancelled")
            .await
            .unwrap();
        assert!(client.download(first_descriptor.object_id).await.is_err());
        assert!(matches!(
            client
                .inspect_transfer(declaration.transfer_id)
                .await
                .unwrap(),
            TransferRecord::Aborted(_)
        ));

        let expiring = begin(b"x", 1, 10);
        let expiring_id = expiring.transfer_id;
        let transfer = client.begin_transfer::<Vec<u8>>(expiring).await.unwrap();
        let chunk = client
            .upload_transfer_chunk(transfer.id(), 0, vec![b'x'], hash_bytes(b"x"))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let record = client.inspect_transfer(expiring_id).await.unwrap();
        assert!(
            matches!(record, TransferRecord::Aborted(ref abort) if abort.reason.contains("expired"))
        );
        assert!(client.download(chunk.object_id).await.is_err());
        assert!(VoidOps::resolve_transfer_raw(client, expiring_id)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn live_stream_commits_and_interruption_falls_back_to_void() {
        let running = running_void().await;
        let client = &running.client;
        let all = tensor_frame(b"safe-tensor");
        let split = all.len() / 2;
        let chunks = vec![all[..split].to_vec(), all[split..].to_vec()];
        let decoded = decode_input::<StreamContract>(&all).unwrap();
        let prepared = client
            .prepare_stream_transfer::<Vec<u8>>(
                StreamContract::descriptor(),
                decoded.envelope,
                tensor_stream_header(&all).unwrap(),
                all.len() as u64,
                hash_bytes(&all),
                chunks.len() as u32,
                Duration::from_secs(5),
                DurabilityPolicy::ReplayRequired,
            )
            .await
            .unwrap();
        client.stream_transfer(&prepared, chunks).await.unwrap();

        let mut interrupted_ticket = prepared.ticket.clone();
        interrupted_ticket.authorization = [0; 32];
        let replayed = client.receive_stream(&interrupted_ticket).await.unwrap();
        assert_eq!(replayed, all);
    }

    #[tokio::test]
    async fn stream_preparation_rejects_an_invalid_tensor_header() {
        let running = running_void().await;
        let client = &running.client;
        let frame = tensor_frame(b"payload");
        let decoded = decode_input::<StreamContract>(&frame).unwrap();
        let mut header = tensor_stream_header(&frame).unwrap();
        let dtype = header
            .windows(2)
            .position(|window| window == b"U8")
            .expect("header contains the declared dtype");
        header[dtype..dtype + 2].copy_from_slice(b"I8");

        let error = client
            .prepare_stream_transfer::<Vec<u8>>(
                StreamContract::descriptor(),
                decoded.envelope,
                header,
                frame.len() as u64,
                hash_bytes(&frame),
                1,
                Duration::from_secs(5),
                DurabilityPolicy::ReplayRequired,
            )
            .await
            .err()
            .expect("invalid tensor header must be rejected");
        assert!(error.contains("dtype mismatch"));
    }

    #[tokio::test]
    async fn streamed_result_stays_uncommitted_until_delayed_commit() {
        let running = running_void().await;
        let client = &running.client;
        let all = tensor_frame(b"header-body");
        let split = all.len() / 2;
        let chunks = [all[..split].to_vec(), all[split..].to_vec()];
        let declaration = stream_declaration(&all, chunks.len() as u32, 5_000);
        let ticket = ticket_for(client, &declaration);

        let receiver_client = client.clone();
        let receiver_ticket = ticket.clone();
        let receiver =
            tokio::spawn(async move { receiver_client.receive_stream(&receiver_ticket).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            !receiver.is_finished(),
            "receiver failed before the producer began its transfer"
        );
        let transfer = client
            .begin_transfer::<Vec<u8>>(declaration.clone())
            .await
            .unwrap();
        client
            .upload_transfer_chunk(transfer.id(), 0, chunks[0].clone(), hash_bytes(&chunks[0]))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !receiver.is_finished(),
            "receiver exposed an output before transfer commit"
        );
        client
            .upload_transfer_chunk(transfer.id(), 1, chunks[1].clone(), hash_bytes(&chunks[1]))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !receiver.is_finished(),
            "receiver exposed an output before delayed commit"
        );
        client
            .commit_transfer(transfer, declaration.expected_hash)
            .await
            .unwrap();
        assert_eq!(receiver.await.unwrap().unwrap(), all);
    }

    #[tokio::test]
    async fn receiver_cancellation_does_not_abort_durable_transfer() {
        let running = running_void().await;
        let client = &running.client;
        let data = tensor_frame(b"durable");
        let declaration = stream_declaration(&data, 1, 5_000);
        let transfer = client
            .begin_transfer::<Vec<u8>>(declaration.clone())
            .await
            .unwrap();
        let ticket = ticket_for(client, &declaration);
        let receiver_client = client.clone();
        let receiver = tokio::spawn(async move { receiver_client.receive_stream(&ticket).await });
        tokio::time::sleep(Duration::from_millis(10)).await;
        receiver.abort();

        client
            .upload_transfer_chunk(transfer.id(), 0, data.clone(), hash_bytes(&data))
            .await
            .unwrap();
        client
            .commit_transfer(transfer, declaration.expected_hash)
            .await
            .unwrap();
        assert_eq!(
            VoidOps::resolve_transfer_raw(client, declaration.transfer_id)
                .await
                .unwrap(),
            data
        );
    }
}
