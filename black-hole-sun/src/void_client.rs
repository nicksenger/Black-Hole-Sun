use std::net::SocketAddr;

use black_hole_spec::ObjectId;
use black_hole_void::{VoidIn, VoidOut};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

/// QUIC client for interacting with the Void service.
#[derive(Clone, Debug)]
pub struct VoidClient {
    endpoint: quinn::Endpoint,
    addr: SocketAddr,
    server_name: String,
}

impl VoidClient {
    pub fn new(
        endpoint: &quinn::Endpoint,
        addr: SocketAddr,
        server_name: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.clone(),
            addr,
            server_name: server_name.into(),
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

    async fn request(&self, request: &VoidIn) -> Result<VoidOut, String> {
        let connecting = self
            .endpoint
            .connect(self.addr, &self.server_name)
            .map_err(|e| format!("connect init failed: {e}"))?;
        let conn = connecting
            .await
            .map_err(|e| format!("connect failed: {e}"))?;
        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| format!("open_bi failed: {e}"))?;

        send_frame(&mut send, request).await?;
        read_frame(&mut recv).await
    }
}

async fn send_frame(send: &mut quinn::SendStream, msg: &impl Serialize) -> Result<(), String> {
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

async fn read_frame<T: for<'de> Deserialize<'de>>(
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
