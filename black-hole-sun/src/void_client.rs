use std::{fs, net::SocketAddr, path::Path};

use black_hole_flux::ops::VoidOps;
use black_hole_spec::ObjectId;
use black_hole_void::{VoidIn, VoidOut};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
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

    /// Upload a local file to void. Files that fit in one frame use the
    /// single-shot upload; larger files are streamed as a chunked multipart
    /// upload. Returns the assigned object ID.
    pub async fn upload_file(&self, path: &Path) -> Result<ObjectId, String> {
        let size = fs::metadata(path)
            .map_err(|e| format!("failed to read file metadata for {}: {e}", path.display()))?;

        if size.len() <= 64 * 1024 * 1024 {
            let data =
                fs::read(path).map_err(|e| format!("failed to read file {}: {e}", path.display()))?;
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
