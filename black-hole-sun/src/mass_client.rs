use std::net::SocketAddr;

use black_hole_spec::{
    MassIn, MassModelCapacity, MassModelConfig, MassModelParams, MassOut, ObjectId,
};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};

#[derive(Clone, Debug)]
enum MassTransport {
    Quic {
        endpoint: quinn::Endpoint,
        addr: SocketAddr,
        server_name: String,
    },
    Tcp {
        addr: SocketAddr,
    },
}

/// Client for interacting with the Mass service.
#[derive(Clone, Debug)]
pub struct MassClient {
    transport: MassTransport,
}

impl MassClient {
    pub fn new(
        endpoint: &quinn::Endpoint,
        addr: SocketAddr,
        server_name: impl Into<String>,
    ) -> Self {
        Self {
            transport: MassTransport::Quic {
                endpoint: endpoint.clone(),
                addr,
                server_name: server_name.into(),
            },
        }
    }

    pub fn new_tcp(addr: SocketAddr) -> Self {
        Self {
            transport: MassTransport::Tcp { addr },
        }
    }

    pub async fn start(
        &self,
        model_id: ObjectId,
        model_config: Option<MassModelConfig>,
    ) -> Result<(), String> {
        let resp = self
            .request(&MassIn::Start {
                model_id,
                model_config,
            })
            .await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for start".to_string()),
        }
    }

    pub async fn infer(&self, model_id: ObjectId, input_id: ObjectId) -> Result<ObjectId, String> {
        let resp = self.request(&MassIn::Infer { model_id, input_id }).await?;
        match resp {
            MassOut::Inferred { output_id } => Ok(output_id),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for infer".to_string()),
        }
    }

    pub async fn reset(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&MassIn::Reset { model_id }).await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for reset".to_string()),
        }
    }

    pub async fn perturb_up(&self, model_id: ObjectId, seed: u64) -> Result<(), String> {
        let resp = self.request(&MassIn::PerturbUp { model_id, seed }).await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for perturb_up".to_string()),
        }
    }

    pub async fn perturb_down(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&MassIn::PerturbDown { model_id }).await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for perturb_down".to_string()),
        }
    }

    pub async fn checkpoint(&self, model_id: ObjectId) -> Result<ObjectId, String> {
        let resp = self.request(&MassIn::Checkpoint { model_id }).await?;
        match resp {
            MassOut::Checkpointed { checkpoint_id } => Ok(checkpoint_id),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for checkpoint".to_string()),
        }
    }

    /// Fuse the current weights of a running model instance with a
    /// void-stored checkpoint using task arithmetic. Returns the void object
    /// ID of the fused weights.
    pub async fn fuse_weights(
        &self,
        model_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        let resp = self
            .request(&MassIn::FuseWeights {
                model_id,
                checkpoint_id,
                contribution,
            })
            .await?;
        match resp {
            MassOut::FusedWeights { fused_id } => Ok(fused_id),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for fuse_weights".to_string()),
        }
    }

    pub async fn optimize(
        &self,
        model_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        let resp = self
            .request(&MassIn::Optimize {
                model_id,
                loss_up,
                loss_down,
            })
            .await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for optimize".to_string()),
        }
    }

    pub async fn shutdown(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&MassIn::Shutdown { model_id }).await?;
        match resp {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for shutdown".to_string()),
        }
    }

    pub async fn query_model_params(&self, model_id: ObjectId) -> Result<MassModelParams, String> {
        let resp = self.request(&MassIn::QueryModelParams { model_id }).await?;
        match resp {
            MassOut::ModelParams { params } => Ok(params),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for query_model_params".to_string()),
        }
    }

    pub async fn query_model_capacity(&self) -> Result<MassModelCapacity, String> {
        let resp = self.request(&MassIn::QueryModelCapacity).await?;
        match resp {
            MassOut::ModelCapacity { capacity } => Ok(capacity),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for query_model_capacity".to_string()),
        }
    }

    async fn request(&self, request: &MassIn) -> Result<MassOut, String> {
        match &self.transport {
            MassTransport::Quic {
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
            MassTransport::Tcp { addr } => {
                let mut stream = TcpStream::connect(*addr)
                    .await
                    .map_err(|e| format!("tcp connect failed: {e}"))?;
                send_frame_io(&mut stream, request).await?;
                read_frame_io(&mut stream).await
            }
        }
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
