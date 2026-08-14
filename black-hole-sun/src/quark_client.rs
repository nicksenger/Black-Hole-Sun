use std::net::SocketAddr;

use black_hole_spec::{
    ObjectId, QuarkIn, QuarkModelCapacity, QuarkModelConfig, QuarkModelParams, QuarkOut,
};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

/// QUIC client for interacting with the Quark service.
#[derive(Clone, Debug)]
pub struct QuarkClient {
    endpoint: quinn::Endpoint,
    addr: SocketAddr,
    server_name: String,
}

impl QuarkClient {
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

    pub async fn start(
        &self,
        model_id: ObjectId,
        model_config: Option<QuarkModelConfig>,
    ) -> Result<(), String> {
        let resp = self
            .request(&QuarkIn::Start {
                model_id,
                model_config,
            })
            .await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for start".to_string()),
        }
    }

    pub async fn infer(&self, model_id: ObjectId, input_id: ObjectId) -> Result<ObjectId, String> {
        let resp = self.request(&QuarkIn::Infer { model_id, input_id }).await?;
        match resp {
            QuarkOut::Inferred { output_id } => Ok(output_id),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for infer".to_string()),
        }
    }

    pub async fn reset(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&QuarkIn::Reset { model_id }).await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for reset".to_string()),
        }
    }

    pub async fn perturb_up(&self, model_id: ObjectId, seed: u64) -> Result<(), String> {
        let resp = self.request(&QuarkIn::PerturbUp { model_id, seed }).await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for perturb_up".to_string()),
        }
    }

    pub async fn perturb_down(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&QuarkIn::PerturbDown { model_id }).await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for perturb_down".to_string()),
        }
    }

    pub async fn checkpoint(&self, model_id: ObjectId) -> Result<ObjectId, String> {
        let resp = self.request(&QuarkIn::Checkpoint { model_id }).await?;
        match resp {
            QuarkOut::Checkpointed { checkpoint_id } => Ok(checkpoint_id),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for checkpoint".to_string()),
        }
    }

    pub async fn optimize(
        &self,
        model_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        let resp = self
            .request(&QuarkIn::Optimize {
                model_id,
                loss_up,
                loss_down,
            })
            .await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for optimize".to_string()),
        }
    }

    pub async fn shutdown(&self, model_id: ObjectId) -> Result<(), String> {
        let resp = self.request(&QuarkIn::Shutdown { model_id }).await?;
        match resp {
            QuarkOut::Ack => Ok(()),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for shutdown".to_string()),
        }
    }

    pub async fn query_model_params(&self, model_id: ObjectId) -> Result<QuarkModelParams, String> {
        let resp = self
            .request(&QuarkIn::QueryModelParams { model_id })
            .await?;
        match resp {
            QuarkOut::ModelParams { params } => Ok(params),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for query_model_params".to_string()),
        }
    }

    pub async fn query_model_capacity(&self) -> Result<QuarkModelCapacity, String> {
        let resp = self.request(&QuarkIn::QueryModelCapacity).await?;
        match resp {
            QuarkOut::ModelCapacity { capacity } => Ok(capacity),
            QuarkOut::Error { message } => Err(message),
            _ => Err("unexpected quark response for query_model_capacity".to_string()),
        }
    }

    async fn request(&self, request: &QuarkIn) -> Result<QuarkOut, String> {
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
