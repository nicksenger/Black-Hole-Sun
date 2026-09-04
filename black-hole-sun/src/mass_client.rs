use std::{marker::PhantomData, net::SocketAddr};

use black_hole_spec::{operation_capability, QwenDarkInference, TensorContract};
use black_hole_flux::ops::{CheckpointOps, FuseOps, MassOps, OptimizeOps, PerturbOps, ResetOps};
use black_hole_type::{
    ArtifactRef, EncodingId, MassIn, MassModelCapacity, MassModelConfig, MassModelParams, MassOut,
    ObjectId, OperationCapabilities, OperationConfig, MASS_OPERATION_PROTOCOL_VERSION,
};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};

use crate::VoidClient;

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
    required_capabilities: OperationCapabilities,
    operation: PhantomData<fn() -> Op>,
}

impl<Op> Clone for MassClient<Op> {
    fn clone(&self) -> Self {
        Self {
            transport: self.transport.clone(),
            required_capabilities: self.required_capabilities,
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
        Self::new_typed(endpoint, addr, server_name).requiring(OperationCapabilities::OPTIMIZING)
    }

    pub fn new_tcp(addr: SocketAddr) -> Self {
        Self::new_tcp_typed(addr).requiring(OperationCapabilities::OPTIMIZING)
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
            required_capabilities: OperationCapabilities::FORWARD_ONLY,
            operation: PhantomData,
        }
    }

    pub fn new_tcp_typed(addr: SocketAddr) -> Self {
        Self {
            transport: MassTransport::Tcp { addr },
            required_capabilities: OperationCapabilities::FORWARD_ONLY,
            operation: PhantomData,
        }
    }

    /// Re-tag this transport for another operation contract.
    pub fn with_operation<Next>(self) -> MassClient<Next> {
        MassClient {
            transport: self.transport,
            required_capabilities: OperationCapabilities::FORWARD_ONLY,
            operation: PhantomData,
        }
    }

    /// Require this lifecycle surface when the instance is placed. The host
    /// rejects the start unless one implementation advertises every bit.
    pub fn requiring(mut self, capabilities: OperationCapabilities) -> Self {
        self.required_capabilities = capabilities;
        self
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
    /// Start an operation instance with its full contract and required
    /// lifecycle surface.
    pub async fn start_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        self.start_operation_with(instance_id, None).await
    }

    pub async fn start_operation_with(
        &self,
        instance_id: ObjectId,
        config: Option<OperationConfig>,
    ) -> Result<(), String> {
        let mut capability = operation_capability::<Op>();
        capability.operations = self.required_capabilities;
        match self
            .request(&MassIn::StartInstance {
                protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                instance_id,
                capability,
                config,
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
            .request(&MassIn::Invoke {
                protocol_version: MASS_OPERATION_PROTOCOL_VERSION,
                instance_id,
                input: input.into(),
            })
            .await?
        {
            MassOut::Invoked { output } => Ok(output.into_typed()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for operation forward".to_string()),
        }
    }

    pub async fn shutdown_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        match self
            .request(&MassIn::ShutdownInstance { instance_id })
            .await?
        {
            MassOut::Ack => Ok(()),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for operation shutdown".to_string()),
        }
    }

    pub async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        expect_ack(
            self.request(&MassIn::ResetInstance { instance_id }).await?,
            "reset",
        )
    }

    pub async fn perturb_up_operation(
        &self,
        instance_id: ObjectId,
        seed: u64,
    ) -> Result<(), String> {
        expect_ack(
            self.request(&MassIn::PerturbUpInstance { instance_id, seed })
                .await?,
            "perturb up",
        )
    }

    pub async fn perturb_down_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        expect_ack(
            self.request(&MassIn::PerturbDownInstance { instance_id })
                .await?,
            "perturb down",
        )
    }

    pub async fn optimize_operation(
        &self,
        instance_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        expect_ack(
            self.request(&MassIn::OptimizeInstance {
                instance_id,
                loss_up,
                loss_down,
            })
            .await?,
            "optimize",
        )
    }

    pub async fn checkpoint_operation(&self, instance_id: ObjectId) -> Result<ObjectId, String> {
        match self
            .request(&MassIn::CheckpointInstance { instance_id })
            .await?
        {
            MassOut::Checkpointed { checkpoint_id } => Ok(checkpoint_id),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for checkpoint".into()),
        }
    }

    pub async fn fuse_operation(
        &self,
        instance_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        match self
            .request(&MassIn::FuseInstance {
                instance_id,
                checkpoint_id,
                contribution,
            })
            .await?
        {
            MassOut::Fused { fused_id } => Ok(fused_id),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for fuse".into()),
        }
    }

    pub async fn query_operation(&self, instance_id: ObjectId) -> Result<OperationConfig, String> {
        match self.request(&MassIn::QueryInstance { instance_id }).await? {
            MassOut::Instance { params } => Ok(params),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for query".into()),
        }
    }

    pub async fn query_instance_capacity(&self) -> Result<MassModelCapacity, String> {
        match self.request(&MassIn::QueryInstanceCapacity).await? {
            MassOut::InstanceCapacity { capacity } => Ok(capacity),
            MassOut::Error { message } => Err(message),
            _ => Err("unexpected mass response for capacity query".into()),
        }
    }
}

fn expect_ack(response: MassOut, operation: &str) -> Result<(), String> {
    match response {
        MassOut::Ack => Ok(()),
        MassOut::Error { message } => Err(message),
        _ => Err(format!("unexpected mass response for {operation}")),
    }
}

impl MassClient<QwenDarkInference> {
    pub async fn start_qwen(
        &self,
        instance_id: ObjectId,
        config: Option<MassModelConfig>,
    ) -> Result<(), String> {
        let config = config
            .map(|config| {
                to_allocvec(&config)
                    .map(|data| OperationConfig {
                        encoding: EncodingId::POSTCARD_V1,
                        data,
                    })
                    .map_err(|error| format!("failed to encode Qwen config: {error}"))
            })
            .transpose()?;
        self.start_operation_with(instance_id, config).await
    }

    pub async fn query_qwen(&self, instance_id: ObjectId) -> Result<MassModelParams, String> {
        let config = self.query_operation(instance_id).await?;
        if config.encoding != EncodingId::POSTCARD_V1 {
            return Err("Qwen query returned an unsupported config encoding".into());
        }
        from_bytes(&config.data).map_err(|error| format!("failed to decode Qwen params: {error}"))
    }

    /// Qwen application adapter over the unified typed invocation protocol.
    pub async fn invoke_qwen_request(
        &self,
        void: &VoidClient,
        instance_id: ObjectId,
        request: black_hole_type::InferenceRequest,
    ) -> Result<ObjectId, String> {
        let input = black_hole_spec::encode_qwen_request(request)
            .map_err(|error| format!("failed to encode Qwen request: {error}"))?;
        let input_id = void
            .upload(input)
            .await
            .map_err(|error| error.to_string())?;
        let output = self
            .forward(instance_id, ArtifactRef::from_object_id(input_id))
            .await?;
        let output_bytes = void
            .receive_artifact(&output)
            .await
            .map_err(|error| error.to_string())?;
        let output = black_hole_spec::decode_qwen_output(&output_bytes)
            .map_err(|error| format!("failed to decode Qwen output: {error}"))?;
        let bytes = to_allocvec(&output)
            .map_err(|error| format!("failed to encode Qwen application output: {error}"))?;
        void.upload(bytes).await.map_err(|error| error.to_string())
    }

    pub async fn invoke_qwen_object(
        &self,
        void: &VoidClient,
        instance_id: ObjectId,
        request_id: ObjectId,
    ) -> Result<ObjectId, String> {
        let bytes = void
            .download(request_id)
            .await
            .map_err(|error| error.to_string())?;
        let request = from_bytes(&bytes)
            .map_err(|error| format!("failed to decode Qwen request: {error}"))?;
        self.invoke_qwen_request(void, instance_id, request).await
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
impl<Op> ResetOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn reset_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        MassClient::reset_operation(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<Op> PerturbOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn perturb_up_operation(&self, instance_id: ObjectId, seed: u64) -> Result<(), String> {
        MassClient::perturb_up_operation(self, instance_id, seed).await
    }

    async fn perturb_down_operation(&self, instance_id: ObjectId) -> Result<(), String> {
        MassClient::perturb_down_operation(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<Op> OptimizeOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn optimize_operation(
        &self,
        instance_id: ObjectId,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<(), String> {
        MassClient::optimize_operation(self, instance_id, loss_up, loss_down).await
    }
}

#[async_trait::async_trait]
impl<Op> CheckpointOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn checkpoint_operation(&self, instance_id: ObjectId) -> Result<ObjectId, String> {
        MassClient::checkpoint_operation(self, instance_id).await
    }
}

#[async_trait::async_trait]
impl<Op> FuseOps<Op> for MassClient<Op>
where
    Op: TensorContract + Send + Sync,
{
    async fn fuse_operation(
        &self,
        instance_id: ObjectId,
        checkpoint_id: ObjectId,
        contribution: f32,
    ) -> Result<ObjectId, String> {
        MassClient::fuse_operation(self, instance_id, checkpoint_id, contribution).await
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
    use black_hole_type::ContractId;

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
