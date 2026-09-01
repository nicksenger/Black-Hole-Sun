use std::{marker::PhantomData, net::SocketAddr};

use black_hole_contract::{operation_capability, QwenDarkInference, TensorContract};
use black_hole_flux::ops::{CheckpointOps, FuseOps, MassOps, OptimizeOps, PerturbOps, ResetOps};
use black_hole_spec::{
    ArtifactRef, MassIn, MassModelCapacity, MassModelConfig, MassModelParams, MassOut, ObjectId,
    MASS_OPERATION_PROTOCOL_VERSION,
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
pub struct MassClient<Op = QwenDarkInference> {
    transport: MassTransport,
    operation: PhantomData<fn() -> Op>,
}

impl<Op> Clone for MassClient<Op> {
    fn clone(&self) -> Self {
        Self {
            transport: self.transport.clone(),
            operation: PhantomData,
        }
    }
}

impl<Op> std::fmt::Debug for MassClient<Op> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MassClient")
            .field("transport", &self.transport)
            .field("operation", &std::any::type_name::<Op>())
            .finish()
    }
}

impl MassClient<QwenDarkInference> {
    pub fn new(
        endpoint: &quinn::Endpoint,
        addr: SocketAddr,
        server_name: impl Into<String>,
    ) -> Self {
        Self::new_typed(endpoint, addr, server_name)
    }

    pub fn new_tcp(addr: SocketAddr) -> Self {
        Self::new_tcp_typed(addr)
    }
}

impl<Op> MassClient<Op> {
    pub fn new_typed(
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
            operation: PhantomData,
        }
    }

    pub fn new_tcp_typed(addr: SocketAddr) -> Self {
        Self {
            transport: MassTransport::Tcp { addr },
            operation: PhantomData,
        }
    }

    /// Re-tag this transport for another operation contract.
    pub fn with_operation<Next>(self) -> MassClient<Next> {
        MassClient {
            transport: self.transport,
            operation: PhantomData,
        }
    }
}

impl MassClient<QwenDarkInference> {
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
}

impl<Op> MassClient<Op> {
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

impl<Op> MassClient<Op>
where
    Op: TensorContract,
{
    /// Start a generic operation instance with its full runtime contract and
    /// supported v1 codecs.
    pub async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        match self
            .request(&MassIn::StartOperation {
                protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                instance_id,
                capability: operation_capability::<Op>(),
            })
            .await?
        {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for operation start".to_string()),
        }
    }

    /// Forward a typed artifact through Mass. The host validates the concrete
    /// tensor envelope against the start contract before execution.
    pub async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<Op::Input>,
    ) -> Result<ArtifactRef<Op::Output>, String> {
        match self
            .request(&MassIn::ForwardOperation {
                protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                instance_id,
                input: input.into(),
            })
            .await?
        {
            MassOut::Forwarded { output } => Ok(output.into_typed()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for operation forward".to_string()),
        }
    }

    pub async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        match self
            .request(&MassIn::ShutdownOperation { instance_id })
            .await?
        {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for operation shutdown".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl<Op> MassOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        MassClient::start_operation(self, instance_id).await
    }

    async fn forward(
        &self,
        instance_id: ObjectId,
        input: ArtifactRef<Op::Input>,
    ) -> Result<ArtifactRef<Op::Output>, String> {
        MassClient::forward(self, instance_id, input).await
    }

    async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        MassClient::shutdown_operation(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl ResetOps<QwenDarkInference> for MassClient<QwenDarkInference> {
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.reset(instance_id).await
    }
}

#[async_trait::async_trait]
impl PerturbOps<QwenDarkInference> for MassClient<QwenDarkInference> {
    async fn perturb_up_operation(&self, instance_id: ObjectId, seed: u64) -> Result<(), String> {
        self.perturb_up(instance_id, seed).await
    }

    async fn perturb_down_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.perturb_down(instance_id).await
    }
}

#[async_trait::async_trait]
impl OptimizeOps<QwenDarkInference> for MassClient<QwenDarkInference> {
    async fn optimize_operation(
        &self,
        instance_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        self.optimize(instance_id, loss_up, loss_down).await
    }
}

#[async_trait::async_trait]
impl CheckpointOps<QwenDarkInference> for MassClient<QwenDarkInference> {
    async fn checkpoint_operation(&self, instance_id: ObjectId) -> Result<ObjectId, String> {
        self.checkpoint(instance_id).await
    }
}

#[async_trait::async_trait]
impl FuseOps<QwenDarkInference> for MassClient<QwenDarkInference> {
    async fn fuse_operation(
        &self,
        instance_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        self.fuse_weights(instance_id, checkpoint_id, contribution)
            .await
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
    use black_hole_spec::ContractId;

    use super::*;

    struct FakeOperation;

    impl TensorContract for FakeOperation {
        type Input = <QwenDarkInference as TensorContract>::Input;
        type Output = <QwenDarkInference as TensorContract>::Output;
        type Metadata = ();

        const ID: ContractId = ContractId::from_u128(0x6661_6b65_2d6f_7065_7261_7469_6f6e_0001);
        const VERSION: u32 = 1;
    }

    #[test]
    fn client_is_contract_typed_and_qwen_keeps_optional_capabilities() {
        fn assert_forward<Op: TensorContract, T: MassOps<Op>>() {}
        fn assert_qwen_capabilities<T>()
        where
            T: ResetOps<QwenDarkInference>
                + PerturbOps<QwenDarkInference>
                + OptimizeOps<QwenDarkInference>
                + CheckpointOps<QwenDarkInference>
                + FuseOps<QwenDarkInference>,
        {
        }

        assert_forward::<FakeOperation, MassClient<FakeOperation>>();
        assert_qwen_capabilities::<MassClient<QwenDarkInference>>();
    }
}
