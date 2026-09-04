//! Compile-time tensor contracts and the backend-neutral v1 wire codec.
//!
//! The contract side depends on glowstick for shape identity, but neither the
//! wire schema nor the codec depends on Candle (or any other tensor backend).

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    marker::PhantomData,
};

use black_hole_type::{
    BackwardCapability, ContractDescriptor, ContractHash, ContractId, ContractSide, DarkToken,
    DimensionDescriptor, DtypeConstraint, EncodingId, InferenceOutput, InferenceRequest,
    LayoutConstraint, LogitEntry, OperationCapability, SequenceOutput, StreamingChunkOrder,
    StreamingFinalization, TensorDtype, TensorEnvelope, TensorPortDescriptor,
};
use postcard::{from_bytes, to_allocvec};
use safetensors::{
    tensor::{Dtype, Metadata as SafeTensorMetadata, SafeTensors, TensorInfo, TensorView},
    SafeTensorError,
};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use glowstick;

const FRAME_MAGIC: &[u8; 8] = b"BHSTEN01";
const FRAME_PREFIX_LEN: usize = FRAME_MAGIC.len() + size_of::<u32>();

/// One named, glowstick-shaped port in a tensor bundle.
///
/// Dimensions are derived from Glowstick's shape. A zero type-level dimension
/// denotes a dynamic dimension.
pub trait TensorPortSpec {
    type Shape: glowstick::Shape + ShapeDimensions;

    const NAME: &'static str;
    const LAYOUT: LayoutConstraint = LayoutConstraint::Contiguous;
    const DTYPE: DtypeConstraint;

    fn descriptor() -> TensorPortDescriptor {
        TensorPortDescriptor {
            name: Self::NAME.to_owned(),
            dimensions: <Self::Shape as ShapeDimensions>::dimensions(),
            dtype: Self::DTYPE.clone(),
            layout: Self::LAYOUT,
        }
    }
}

/// Converts a Glowstick shape's type-level dimensions into runtime descriptors.
pub trait ShapeDimensions {
    fn dimensions() -> Vec<DimensionDescriptor>;
}

trait ShapeFragmentDimensions {
    fn dimensions() -> Vec<DimensionDescriptor>;
}

fn dimension_descriptor(dimension: usize) -> DimensionDescriptor {
    match dimension {
        0 => DimensionDescriptor::Dynamic,
        dimension => DimensionDescriptor::Static(dimension as u64),
    }
}

impl ShapeFragmentDimensions for glowstick::Empty {
    fn dimensions() -> Vec<DimensionDescriptor> {
        Vec::new()
    }
}

impl<Dimension, Next> ShapeFragmentDimensions for glowstick::Shp<(Dimension, Next)>
where
    Dimension: glowstick::Dimension,
    Next: ShapeFragmentDimensions,
{
    fn dimensions() -> Vec<DimensionDescriptor> {
        let mut dimensions = vec![dimension_descriptor(Dimension::USIZE)];
        dimensions.extend(Next::dimensions());
        dimensions
    }
}

impl<Fragment> ShapeDimensions for glowstick::TensorShape<Fragment>
where
    Fragment: glowstick::ShapeFragment + ShapeFragmentDimensions,
{
    fn dimensions() -> Vec<DimensionDescriptor> {
        Fragment::dimensions()
    }
}

/// Type-level list of named tensor ports.
pub trait PortList {
    fn descriptors() -> Vec<TensorPortDescriptor>;
}

macro_rules! port_list {
    ($($name:ident),+) => {
        impl<$($name: TensorPortSpec),+> PortList for ($($name,)+) {
            fn descriptors() -> Vec<TensorPortDescriptor> {
                vec![$($name::descriptor()),+]
            }
        }
    };
}

port_list!(A);
port_list!(A, B);
port_list!(A, B, C);
port_list!(A, B, C, D);
port_list!(A, B, C, D, E);
port_list!(A, B, C, D, E, F);
port_list!(A, B, C, D, E, F, G);
port_list!(A, B, C, D, E, F, G, H);

/// A named bundle of one or more tensor ports.
pub struct TensorBundleSpec<Ports>(PhantomData<Ports>);

/// Compile-time tensor bundle used as a contract input or output.
pub trait TensorSpec {
    fn descriptor() -> Vec<TensorPortDescriptor>;
}

impl<Ports: PortList> TensorSpec for TensorBundleSpec<Ports> {
    fn descriptor() -> Vec<TensorPortDescriptor> {
        Ports::descriptors()
    }
}

/// Convenience alias for the common one-tensor bundle.
pub type SingleTensorSpec<Port> = TensorBundleSpec<(Port,)>;

/// Compile-time half of a distributed tensor operation contract.
pub trait TensorContract {
    type Input: TensorSpec;
    type Output: TensorSpec;
    type Metadata;

    const ID: ContractId;
    const VERSION: u32;

    fn descriptor() -> ContractDescriptor {
        ContractDescriptor {
            id: Self::ID,
            version: Self::VERSION,
            inputs: Self::Input::descriptor(),
            outputs: Self::Output::descriptor(),
        }
    }
}

/// Reverse-mode tensor types for an operation that retains its forward graph.
///
/// `OutputGrad` is received from the downstream stage (the derivative with
/// respect to this operation's output); `InputGrad` is returned upstream.
pub trait BackwardContract: TensorContract {
    type OutputGrad: TensorSpec;
    type InputGrad: TensorSpec;

    fn backward_descriptor() -> ContractDescriptor {
        ContractDescriptor {
            id: Self::ID,
            version: Self::VERSION,
            inputs: Self::OutputGrad::descriptor(),
            outputs: Self::InputGrad::descriptor(),
        }
    }
}

/// Explicit opt-in contract for operators that can execute before a complete
/// tensor artifact has arrived.
///
/// Ordinary [`TensorContract`] implementations remain full-artifact
/// operations. Implementing this trait declares the chunk axis, required
/// ordering, and finalization boundary that a streaming runtime must enforce.
pub trait StreamingTensorOp: TensorContract {
    const CHUNK_AXIS: usize;
    const CHUNK_ORDER: StreamingChunkOrder;
    const FINALIZATION: StreamingFinalization;
}

// ---------------------------------------------------------------------------
// Legacy Qwen compatibility contract
// ---------------------------------------------------------------------------

/// Stable operation contract for the existing dark-token Qwen path.
///
/// The compatibility adapter continues to encode `InferenceRequest` and
/// `InferenceOutput` with postcard. These port specs define the tensor bundle
/// that the adapter will expose when the generic Mass protocol lands in Stage
/// 3: predicted tokens plus their top-k dark-knowledge distribution. Input
/// and output deliberately use the same bundle because the legacy path feeds
/// an `InferenceOutput` back as the next operation's dark input.
pub struct QwenDarkInference;

pub struct QwenPredictions;
pub struct QwenDarkTokenIds;
pub struct QwenDarkLogProbs;

pub struct QwenBatch;
pub struct QwenSequence;
pub struct QwenTopK;

/// Qwen-specific invocation metadata. Text/token requests stay behind the
/// Qwen implementation boundary; the Mass protocol carries only a normal
/// typed artifact.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum QwenInferenceMetadata {
    /// Interpret the three tensors as a dense dark-token batch.
    Dark { limit: Option<u32> },
    /// Let the Qwen backend resolve text, token, dark, or Void-referenced
    /// inputs without adding a backend-specific Mass message.
    Request(InferenceRequest),
}

impl TensorPortSpec for QwenPredictions {
    type Shape = glowstick::Shape2<glowstick::Dyn<QwenBatch>, glowstick::Dyn<QwenSequence>>;

    const NAME: &'static str = "predictions";

    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::U32);
}

impl TensorPortSpec for QwenDarkTokenIds {
    type Shape = glowstick::Shape3<
        glowstick::Dyn<QwenBatch>,
        glowstick::Dyn<QwenSequence>,
        glowstick::Dyn<QwenTopK>,
    >;

    const NAME: &'static str = "dark_token_ids";

    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::U32);
}

impl TensorPortSpec for QwenDarkLogProbs {
    type Shape = glowstick::Shape3<
        glowstick::Dyn<QwenBatch>,
        glowstick::Dyn<QwenSequence>,
        glowstick::Dyn<QwenTopK>,
    >;

    const NAME: &'static str = "dark_log_probs";

    const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
}

impl TensorContract for QwenDarkInference {
    type Input = TensorBundleSpec<(QwenPredictions, QwenDarkTokenIds, QwenDarkLogProbs)>;
    type Output = TensorBundleSpec<(QwenPredictions, QwenDarkTokenIds, QwenDarkLogProbs)>;
    type Metadata = QwenInferenceMetadata;

    // Application-assigned, stable ID; never derived from the Rust type name.
    const ID: ContractId = ContractId::from_u128(0x7177_656e_2d64_6172_6b2d_696e_6665_7231);
    const VERSION: u32 = 1;
}

/// Encode a Qwen request as an ordinary typed input artifact. Empty dense
/// tensors satisfy the contract while the backend-owned request lives in
/// metadata; dark tensor pipelines use [`QwenInferenceMetadata::Dark`]
/// instead.
pub fn encode_qwen_request(request: InferenceRequest) -> Result<Vec<u8>, CodecError> {
    let tensors = [
        RawTensor {
            name: "predictions".into(),
            dtype: TensorDtype::U32,
            shape: vec![0, 0],
            data: Vec::new(),
        },
        RawTensor {
            name: "dark_token_ids".into(),
            dtype: TensorDtype::U32,
            shape: vec![0, 0, 0],
            data: Vec::new(),
        },
        RawTensor {
            name: "dark_log_probs".into(),
            dtype: TensorDtype::F32,
            shape: vec![0, 0, 0],
            data: Vec::new(),
        },
    ];
    encode_input::<QwenDarkInference>(&tensors, &QwenInferenceMetadata::Request(request))
}

/// Decode the dense Qwen output bundle into the application-level dark-token
/// value used by prompt and loss policies.
pub fn decode_qwen_output(bytes: &[u8]) -> Result<InferenceOutput, CodecError> {
    let decoded = decode_output::<QwenDarkInference>(bytes)?;
    let predictions = &decoded.tensors[0];
    let token_ids = &decoded.tensors[1];
    let log_probs = &decoded.tensors[2];
    let batch = predictions.shape[0];
    let sequence = predictions.shape[1];
    let top_k = token_ids.shape[2];
    let predictions: Vec<u32> = predictions
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let token_ids: Vec<u32> = token_ids
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let log_probs: Vec<f32> = log_probs
        .data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let mut results = Vec::with_capacity(batch);
    for batch_index in 0..batch {
        let mut tokens = Vec::with_capacity(sequence);
        for sequence_index in 0..sequence {
            let position = batch_index * sequence + sequence_index;
            let distribution_start = position * top_k;
            tokens.push(DarkToken {
                predicted: predictions[position],
                dark_knowledge: (0..top_k)
                    .map(|offset| LogitEntry {
                        token_id: token_ids[distribution_start + offset],
                        log_prob: log_probs[distribution_start + offset],
                    })
                    .collect(),
            });
        }
        results.push(SequenceOutput(tokens));
    }
    Ok(InferenceOutput { results })
}

/// Owned backend-neutral dense tensor value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    /// Little-endian, contiguous row-major element bytes.
    pub data: Vec<u8>,
}

/// Validated decoded tensor artifact, tagged with its compile-time spec.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedTensorBundle<S, M> {
    pub envelope: TensorEnvelope,
    pub metadata: M,
    pub tensors: Vec<RawTensor>,
    marker: PhantomData<S>,
}

/// Runtime-validated artifact used by type-erased operation hosts.
///
/// Metadata stays encoded because an injected operation, rather than the Mass
/// router, owns its concrete type.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedArtifact {
    pub envelope: TensorEnvelope,
    pub metadata: Vec<u8>,
    pub tensors: Vec<RawTensor>,
}

/// Fail-closed validation or encoding error.
#[derive(Debug, Error)]
pub enum CodecError {
    #[error("tensor frame is truncated")]
    Truncated,
    #[error("invalid tensor frame magic")]
    InvalidMagic,
    #[error("unsupported envelope version {0}")]
    UnsupportedEnvelopeVersion(u16),
    #[error("unsupported tensor encoding {0:?}")]
    UnsupportedTensorEncoding(EncodingId),
    #[error("unsupported metadata encoding {0:?}")]
    UnsupportedMetadataEncoding(EncodingId),
    #[error("contract id mismatch")]
    ContractIdMismatch,
    #[error("contract version mismatch: expected {expected}, got {actual}")]
    ContractVersionMismatch { expected: u32, actual: u32 },
    #[error("contract descriptor hash mismatch")]
    ContractHashMismatch,
    #[error("artifact is for the {actual:?} side, not {expected:?}")]
    ContractSideMismatch {
        expected: ContractSide,
        actual: ContractSide,
    },
    #[error("frame length does not match its envelope")]
    LengthMismatch,
    #[error("contract contains duplicate tensor port {0:?}")]
    DuplicatePort(String),
    #[error("tensor bundle contains duplicate tensor {0:?}")]
    DuplicateTensor(String),
    #[error("missing tensor {0:?}")]
    MissingTensor(String),
    #[error("unexpected tensor {0:?}")]
    UnexpectedTensor(String),
    #[error("dtype mismatch for {name:?}: expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        name: String,
        expected: DtypeConstraint,
        actual: TensorDtype,
    },
    #[error("rank mismatch for {name:?}: expected {expected}, got {actual}")]
    RankMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("dimension mismatch for {name:?} axis {axis}: expected {expected}, got {actual}")]
    DimensionMismatch {
        name: String,
        axis: usize,
        expected: usize,
        actual: usize,
    },
    #[error("symbolic dimension {label:?} was bound to both {first} and {second}")]
    SymbolicDimensionMismatch {
        label: String,
        first: usize,
        second: usize,
    },
    #[error("safetensors dtype {0:?} is not supported by the v1 contract schema")]
    UnsupportedDtype(Dtype),
    #[error("integer length does not fit on this platform")]
    LengthOverflow,
    #[error("safetensors: {0}")]
    Safetensors(#[from] SafeTensorError),
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("safetensors header JSON: {0}")]
    HeaderJson(#[from] serde_json::Error),
}

/// Stable hash of a contract's canonical postcard representation.
pub fn descriptor_hash(descriptor: &ContractDescriptor) -> ContractHash {
    let bytes = to_allocvec(descriptor).expect("ContractDescriptor serialization is infallible");
    ContractHash(Sha256::digest(bytes).into())
}

/// V1 runtime capability declaration for a compile-time contract.
pub fn operation_capability<C: TensorContract>() -> OperationCapability {
    let descriptor = C::descriptor();
    OperationCapability {
        descriptor_hash: descriptor_hash(&descriptor),
        descriptor,
        tensor_encodings: vec![EncodingId::SAFETENSORS_V1],
        metadata_encodings: vec![EncodingId::POSTCARD_V1],
        operations: black_hole_type::OperationCapabilities::FORWARD_ONLY,
        backward: None,
    }
}

/// V1 runtime declaration for a forward-and-backward contract.
pub fn backward_operation_capability<C: BackwardContract>() -> OperationCapability {
    let mut capability = operation_capability::<C>();
    let descriptor = C::backward_descriptor();
    capability.backward = Some(Box::new(BackwardCapability {
        descriptor_hash: descriptor_hash(&descriptor),
        descriptor,
    }));
    capability
}

/// Encode a contract input bundle with postcard metadata and safetensors data.
pub fn encode_input<C>(tensors: &[RawTensor], metadata: &C::Metadata) -> Result<Vec<u8>, CodecError>
where
    C: TensorContract,
    C::Metadata: Serialize,
{
    encode::<C>(ContractSide::Input, tensors, metadata)
}

/// Encode a contract output bundle with postcard metadata and safetensors data.
pub fn encode_output<C>(
    tensors: &[RawTensor],
    metadata: &C::Metadata,
) -> Result<Vec<u8>, CodecError>
where
    C: TensorContract,
    C::Metadata: Serialize,
{
    encode::<C>(ContractSide::Output, tensors, metadata)
}

/// Decode and validate a contract input bundle.
pub fn decode_input<C>(
    frame: &[u8],
) -> Result<DecodedTensorBundle<C::Input, C::Metadata>, CodecError>
where
    C: TensorContract,
    C::Metadata: DeserializeOwned,
{
    decode::<C, C::Input>(ContractSide::Input, frame)
}

/// Decode and validate a contract output bundle.
pub fn decode_output<C>(
    frame: &[u8],
) -> Result<DecodedTensorBundle<C::Output, C::Metadata>, CodecError>
where
    C: TensorContract,
    C::Metadata: DeserializeOwned,
{
    decode::<C, C::Output>(ContractSide::Output, frame)
}

/// Encode the downstream gradient consumed by `C::backward`.
pub fn encode_output_gradient<C>(
    tensors: &[RawTensor],
    metadata: &C::Metadata,
) -> Result<Vec<u8>, CodecError>
where
    C: BackwardContract,
    C::Metadata: Serialize,
{
    encode_descriptor::<C::Metadata>(
        C::backward_descriptor(),
        ContractSide::Input,
        tensors,
        metadata,
    )
}

/// Decode the downstream gradient consumed by `C::backward`.
pub fn decode_output_gradient<C>(
    frame: &[u8],
) -> Result<DecodedTensorBundle<C::OutputGrad, C::Metadata>, CodecError>
where
    C: BackwardContract,
    C::Metadata: DeserializeOwned,
{
    decode_descriptor::<C::OutputGrad, C::Metadata>(
        &C::backward_descriptor(),
        ContractSide::Input,
        frame,
    )
}

/// Encode the upstream gradient produced by `C::backward`.
pub fn encode_input_gradient<C>(
    tensors: &[RawTensor],
    metadata: &C::Metadata,
) -> Result<Vec<u8>, CodecError>
where
    C: BackwardContract,
    C::Metadata: Serialize,
{
    encode_descriptor::<C::Metadata>(
        C::backward_descriptor(),
        ContractSide::Output,
        tensors,
        metadata,
    )
}

/// Decode the upstream gradient produced by `C::backward`.
pub fn decode_input_gradient<C>(
    frame: &[u8],
) -> Result<DecodedTensorBundle<C::InputGrad, C::Metadata>, CodecError>
where
    C: BackwardContract,
    C::Metadata: DeserializeOwned,
{
    decode_descriptor::<C::InputGrad, C::Metadata>(
        &C::backward_descriptor(),
        ContractSide::Output,
        frame,
    )
}

/// Decode and validate a tensor artifact against a runtime contract.
///
/// This is the distributed counterpart to `decode_input::<C>`: Mass can
/// validate an injected operation's actual payload after Rust types have been
/// erased at the process boundary.
pub fn validate_artifact(
    descriptor: &ContractDescriptor,
    expected_side: ContractSide,
    frame: &[u8],
) -> Result<ValidatedArtifact, CodecError> {
    if frame.len() < FRAME_PREFIX_LEN {
        return Err(CodecError::Truncated);
    }
    if &frame[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let envelope_len = u32::from_le_bytes(
        frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
            .try_into()
            .expect("fixed-size prefix"),
    ) as usize;
    let envelope_end = FRAME_PREFIX_LEN
        .checked_add(envelope_len)
        .ok_or(CodecError::LengthOverflow)?;
    let envelope_bytes = frame
        .get(FRAME_PREFIX_LEN..envelope_end)
        .ok_or(CodecError::Truncated)?;
    let envelope: TensorEnvelope = from_bytes(envelope_bytes)?;
    validate_runtime_envelope(&envelope, descriptor, expected_side)?;

    let metadata_len: usize = envelope
        .metadata_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let tensor_len: usize = envelope
        .tensor_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let metadata_end = envelope_end
        .checked_add(metadata_len)
        .ok_or(CodecError::LengthOverflow)?;
    let tensor_end = metadata_end
        .checked_add(tensor_len)
        .ok_or(CodecError::LengthOverflow)?;
    if tensor_end != frame.len() {
        return Err(CodecError::LengthMismatch);
    }
    let metadata = frame
        .get(envelope_end..metadata_end)
        .ok_or(CodecError::Truncated)?
        .to_vec();
    let tensor_bytes = frame
        .get(metadata_end..tensor_end)
        .ok_or(CodecError::Truncated)?;
    let safetensors = SafeTensors::deserialize(tensor_bytes)?;
    let mut tensors = Vec::with_capacity(safetensors.len());
    for (name, view) in safetensors.tensors() {
        tensors.push(RawTensor {
            name,
            dtype: from_safetensors_dtype(view.dtype())?,
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        });
    }

    let ports = ports_for(descriptor, expected_side);
    validate_schema(ports)?;
    validate_tensors(ports, &tensors)?;
    tensors.sort_by_key(|tensor| {
        ports
            .iter()
            .position(|port| port.name == tensor.name)
            .expect("validated tensor has a matching port")
    });
    Ok(ValidatedArtifact {
        envelope,
        metadata,
        tensors,
    })
}

/// Return the Black Hole envelope, metadata, and safetensors header prefix
/// that must lead a live stream. A receiver can validate the concrete tensor
/// names, dtypes, shapes, and offsets from this prefix before payload bytes
/// arrive.
pub fn tensor_stream_header(frame: &[u8]) -> Result<Vec<u8>, CodecError> {
    if frame.len() < FRAME_PREFIX_LEN {
        return Err(CodecError::Truncated);
    }
    if &frame[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let envelope_len = u32::from_le_bytes(
        frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
            .try_into()
            .expect("fixed-size prefix"),
    ) as usize;
    let envelope_end = FRAME_PREFIX_LEN
        .checked_add(envelope_len)
        .ok_or(CodecError::LengthOverflow)?;
    let envelope: TensorEnvelope = from_bytes(
        frame
            .get(FRAME_PREFIX_LEN..envelope_end)
            .ok_or(CodecError::Truncated)?,
    )?;
    let metadata_len: usize = envelope
        .metadata_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let tensor_start = envelope_end
        .checked_add(metadata_len)
        .ok_or(CodecError::LengthOverflow)?;
    let safetensors_len_end = tensor_start
        .checked_add(size_of::<u64>())
        .ok_or(CodecError::LengthOverflow)?;
    let safetensors_header_len = u64::from_le_bytes(
        frame
            .get(tensor_start..safetensors_len_end)
            .ok_or(CodecError::Truncated)?
            .try_into()
            .expect("fixed-size safetensors header length"),
    );
    let safetensors_header_len: usize = safetensors_header_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let header_end = safetensors_len_end
        .checked_add(safetensors_header_len)
        .ok_or(CodecError::LengthOverflow)?;
    Ok(frame
        .get(..header_end)
        .ok_or(CodecError::Truncated)?
        .to_vec())
}

/// Validate a streamed tensor's complete header before any payload bytes are
/// accepted or a destination tensor is allocated.
///
/// `header` contains the Black Hole envelope, metadata, and the safetensors
/// header, but no tensor data. The returned length is the exact full frame
/// length declared by those headers.
pub fn validate_tensor_stream_header(
    descriptor: &ContractDescriptor,
    expected_side: ContractSide,
    header: &[u8],
) -> Result<u64, CodecError> {
    if header.len() < FRAME_PREFIX_LEN {
        return Err(CodecError::Truncated);
    }
    if &header[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let envelope_len = u32::from_le_bytes(
        header[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
            .try_into()
            .expect("fixed-size prefix"),
    ) as usize;
    let envelope_end = FRAME_PREFIX_LEN
        .checked_add(envelope_len)
        .ok_or(CodecError::LengthOverflow)?;
    let envelope: TensorEnvelope = from_bytes(
        header
            .get(FRAME_PREFIX_LEN..envelope_end)
            .ok_or(CodecError::Truncated)?,
    )?;
    validate_runtime_envelope(&envelope, descriptor, expected_side)?;

    let metadata_len: usize = envelope
        .metadata_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let tensor_start = envelope_end
        .checked_add(metadata_len)
        .ok_or(CodecError::LengthOverflow)?;
    let safetensors_len_end = tensor_start
        .checked_add(size_of::<u64>())
        .ok_or(CodecError::LengthOverflow)?;
    let safetensors_header_len = u64::from_le_bytes(
        header
            .get(tensor_start..safetensors_len_end)
            .ok_or(CodecError::Truncated)?
            .try_into()
            .expect("fixed-size safetensors header length"),
    );
    let safetensors_header_len: usize = safetensors_header_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let header_end = safetensors_len_end
        .checked_add(safetensors_header_len)
        .ok_or(CodecError::LengthOverflow)?;
    if header_end != header.len() {
        return Err(CodecError::LengthMismatch);
    }

    let mut entries: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header[safetensors_len_end..header_end])?;
    let metadata = entries
        .remove("__metadata__")
        .map(serde_json::from_value)
        .transpose()?;
    let mut infos = entries
        .into_iter()
        .map(|(name, value)| serde_json::from_value::<TensorInfo>(value).map(|info| (name, info)))
        .collect::<Result<Vec<_>, _>>()?;
    infos.sort_by_key(|(_, info)| info.data_offsets.0);
    let metadata = SafeTensorMetadata::new(metadata, infos)?;

    let safetensors_len = size_of::<u64>()
        .checked_add(safetensors_header_len)
        .and_then(|len| len.checked_add(metadata.data_len()))
        .ok_or(CodecError::LengthOverflow)?;
    if u64::try_from(safetensors_len).map_err(|_| CodecError::LengthOverflow)?
        != envelope.tensor_len
    {
        return Err(CodecError::LengthMismatch);
    }

    let tensors = metadata
        .offset_keys()
        .into_iter()
        .map(|name| {
            let info = metadata
                .info(&name)
                .expect("offset key always identifies tensor metadata");
            Ok(RawTensor {
                name,
                dtype: from_safetensors_dtype(info.dtype)?,
                shape: info.shape.clone(),
                data: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, CodecError>>()?;
    let ports = ports_for(descriptor, expected_side);
    validate_schema(ports)?;
    validate_tensors(ports, &tensors)?;

    let full_frame_len = tensor_start
        .checked_add(safetensors_len)
        .ok_or(CodecError::LengthOverflow)?;
    u64::try_from(full_frame_len).map_err(|_| CodecError::LengthOverflow)
}

fn encode<C>(
    side: ContractSide,
    tensors: &[RawTensor],
    metadata: &C::Metadata,
) -> Result<Vec<u8>, CodecError>
where
    C: TensorContract,
    C::Metadata: Serialize,
{
    encode_descriptor::<C::Metadata>(C::descriptor(), side, tensors, metadata)
}

fn encode_descriptor<M: Serialize>(
    descriptor: ContractDescriptor,
    side: ContractSide,
    tensors: &[RawTensor],
    metadata: &M,
) -> Result<Vec<u8>, CodecError> {
    let ports = ports_for(&descriptor, side);
    validate_schema(ports)?;
    validate_tensors(ports, tensors)?;

    let mut views = BTreeMap::new();
    for tensor in tensors {
        let view = TensorView::new(
            to_safetensors_dtype(tensor.dtype),
            tensor.shape.clone(),
            &tensor.data,
        )?;
        if views.insert(tensor.name.as_str(), view).is_some() {
            return Err(CodecError::DuplicateTensor(tensor.name.clone()));
        }
    }
    let tensor_bytes = safetensors::tensor::serialize(views, None)?;
    let metadata_bytes = to_allocvec(metadata)?;
    let envelope = TensorEnvelope {
        envelope_version: TensorEnvelope::VERSION,
        contract_id: descriptor.id,
        contract_version: descriptor.version,
        contract_hash: descriptor_hash(&descriptor),
        side,
        tensor_encoding: EncodingId::SAFETENSORS_V1,
        metadata_encoding: EncodingId::POSTCARD_V1,
        metadata_len: metadata_bytes
            .len()
            .try_into()
            .map_err(|_| CodecError::LengthOverflow)?,
        tensor_len: tensor_bytes
            .len()
            .try_into()
            .map_err(|_| CodecError::LengthOverflow)?,
    };
    let envelope_bytes = to_allocvec(&envelope)?;
    let envelope_len: u32 = envelope_bytes
        .len()
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;

    let mut frame = Vec::with_capacity(
        FRAME_PREFIX_LEN + envelope_bytes.len() + metadata_bytes.len() + tensor_bytes.len(),
    );
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&envelope_len.to_le_bytes());
    frame.extend_from_slice(&envelope_bytes);
    frame.extend_from_slice(&metadata_bytes);
    frame.extend_from_slice(&tensor_bytes);
    Ok(frame)
}

fn decode<C, S>(
    expected_side: ContractSide,
    frame: &[u8],
) -> Result<DecodedTensorBundle<S, C::Metadata>, CodecError>
where
    C: TensorContract,
    C::Metadata: DeserializeOwned,
{
    decode_descriptor::<S, C::Metadata>(&C::descriptor(), expected_side, frame)
}

fn decode_descriptor<S, M>(
    descriptor: &ContractDescriptor,
    expected_side: ContractSide,
    frame: &[u8],
) -> Result<DecodedTensorBundle<S, M>, CodecError>
where
    M: DeserializeOwned,
{
    if frame.len() < FRAME_PREFIX_LEN {
        return Err(CodecError::Truncated);
    }
    if &frame[..FRAME_MAGIC.len()] != FRAME_MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let envelope_len = u32::from_le_bytes(
        frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
            .try_into()
            .expect("fixed-size prefix"),
    ) as usize;
    let envelope_end = FRAME_PREFIX_LEN
        .checked_add(envelope_len)
        .ok_or(CodecError::LengthOverflow)?;
    let envelope_bytes = frame
        .get(FRAME_PREFIX_LEN..envelope_end)
        .ok_or(CodecError::Truncated)?;
    let envelope: TensorEnvelope = from_bytes(envelope_bytes)?;

    validate_runtime_envelope(&envelope, descriptor, expected_side)?;

    let metadata_len: usize = envelope
        .metadata_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let tensor_len: usize = envelope
        .tensor_len
        .try_into()
        .map_err(|_| CodecError::LengthOverflow)?;
    let metadata_end = envelope_end
        .checked_add(metadata_len)
        .ok_or(CodecError::LengthOverflow)?;
    let tensor_end = metadata_end
        .checked_add(tensor_len)
        .ok_or(CodecError::LengthOverflow)?;
    if tensor_end != frame.len() {
        return Err(CodecError::LengthMismatch);
    }
    let metadata_bytes = frame
        .get(envelope_end..metadata_end)
        .ok_or(CodecError::Truncated)?;
    let tensor_bytes = frame
        .get(metadata_end..tensor_end)
        .ok_or(CodecError::Truncated)?;
    let metadata = from_bytes(metadata_bytes)?;
    let safetensors = SafeTensors::deserialize(tensor_bytes)?;
    let mut tensors = Vec::with_capacity(safetensors.len());
    for (name, view) in safetensors.tensors() {
        tensors.push(RawTensor {
            name,
            dtype: from_safetensors_dtype(view.dtype())?,
            shape: view.shape().to_vec(),
            data: view.data().to_vec(),
        });
    }

    let ports = ports_for(descriptor, expected_side);
    validate_schema(ports)?;
    validate_tensors(ports, &tensors)?;
    tensors.sort_by_key(|tensor| {
        ports
            .iter()
            .position(|port| port.name == tensor.name)
            .expect("validated tensor has a matching port")
    });
    Ok(DecodedTensorBundle {
        envelope,
        metadata,
        tensors,
        marker: PhantomData,
    })
}

fn validate_runtime_envelope(
    envelope: &TensorEnvelope,
    descriptor: &ContractDescriptor,
    expected_side: ContractSide,
) -> Result<(), CodecError> {
    if envelope.envelope_version != TensorEnvelope::VERSION {
        return Err(CodecError::UnsupportedEnvelopeVersion(
            envelope.envelope_version,
        ));
    }
    if envelope.tensor_encoding != EncodingId::SAFETENSORS_V1 {
        return Err(CodecError::UnsupportedTensorEncoding(
            envelope.tensor_encoding,
        ));
    }
    if envelope.metadata_encoding != EncodingId::POSTCARD_V1 {
        return Err(CodecError::UnsupportedMetadataEncoding(
            envelope.metadata_encoding,
        ));
    }
    if envelope.contract_id != descriptor.id {
        return Err(CodecError::ContractIdMismatch);
    }
    if envelope.contract_version != descriptor.version {
        return Err(CodecError::ContractVersionMismatch {
            expected: descriptor.version,
            actual: envelope.contract_version,
        });
    }
    if envelope.contract_hash != descriptor_hash(descriptor) {
        return Err(CodecError::ContractHashMismatch);
    }
    if envelope.side != expected_side {
        return Err(CodecError::ContractSideMismatch {
            expected: expected_side,
            actual: envelope.side,
        });
    }
    Ok(())
}

fn ports_for(descriptor: &ContractDescriptor, side: ContractSide) -> &[TensorPortDescriptor] {
    match side {
        ContractSide::Input => &descriptor.inputs,
        ContractSide::Output => &descriptor.outputs,
    }
}

fn validate_schema(ports: &[TensorPortDescriptor]) -> Result<(), CodecError> {
    let mut names = HashSet::with_capacity(ports.len());
    for port in ports {
        if !names.insert(port.name.as_str()) {
            return Err(CodecError::DuplicatePort(port.name.clone()));
        }
    }
    Ok(())
}

fn validate_tensors(
    ports: &[TensorPortDescriptor],
    tensors: &[RawTensor],
) -> Result<(), CodecError> {
    let mut actual = HashMap::with_capacity(tensors.len());
    for tensor in tensors {
        if actual.insert(tensor.name.as_str(), tensor).is_some() {
            return Err(CodecError::DuplicateTensor(tensor.name.clone()));
        }
    }
    for tensor in tensors {
        if !ports.iter().any(|port| port.name == tensor.name) {
            return Err(CodecError::UnexpectedTensor(tensor.name.clone()));
        }
    }

    let mut bindings = HashMap::<&str, usize>::new();
    for port in ports {
        let tensor = actual
            .get(port.name.as_str())
            .ok_or_else(|| CodecError::MissingTensor(port.name.clone()))?;
        if !port.dtype.accepts(tensor.dtype) {
            return Err(CodecError::DtypeMismatch {
                name: port.name.clone(),
                expected: port.dtype.clone(),
                actual: tensor.dtype,
            });
        }
        if port.layout != LayoutConstraint::Any && port.layout != LayoutConstraint::Contiguous {
            unreachable!("all current layout variants are handled")
        }
        if port.dimensions.len() != tensor.shape.len() {
            return Err(CodecError::RankMismatch {
                name: port.name.clone(),
                expected: port.dimensions.len(),
                actual: tensor.shape.len(),
            });
        }
        for (axis, (rule, &actual)) in port.dimensions.iter().zip(&tensor.shape).enumerate() {
            match rule {
                DimensionDescriptor::Static(expected) => {
                    let expected: usize = (*expected)
                        .try_into()
                        .map_err(|_| CodecError::LengthOverflow)?;
                    if expected != actual {
                        return Err(CodecError::DimensionMismatch {
                            name: port.name.clone(),
                            axis,
                            expected,
                            actual,
                        });
                    }
                }
                DimensionDescriptor::Symbolic(label) => {
                    if let Some(&first) = bindings.get(label.as_str()) {
                        if first != actual {
                            return Err(CodecError::SymbolicDimensionMismatch {
                                label: label.clone(),
                                first,
                                second: actual,
                            });
                        }
                    } else {
                        bindings.insert(label, actual);
                    }
                }
                DimensionDescriptor::Dynamic => {}
            }
        }
    }
    Ok(())
}

fn to_safetensors_dtype(dtype: TensorDtype) -> Dtype {
    match dtype {
        TensorDtype::Bool => Dtype::BOOL,
        TensorDtype::U8 => Dtype::U8,
        TensorDtype::U16 => Dtype::U16,
        TensorDtype::U32 => Dtype::U32,
        TensorDtype::U64 => Dtype::U64,
        TensorDtype::I8 => Dtype::I8,
        TensorDtype::I16 => Dtype::I16,
        TensorDtype::I32 => Dtype::I32,
        TensorDtype::I64 => Dtype::I64,
        TensorDtype::F8E4M3 => Dtype::F8_E4M3,
        TensorDtype::F8E5M2 => Dtype::F8_E5M2,
        TensorDtype::F16 => Dtype::F16,
        TensorDtype::BF16 => Dtype::BF16,
        TensorDtype::F32 => Dtype::F32,
        TensorDtype::F64 => Dtype::F64,
        TensorDtype::F4 => Dtype::F4,
        TensorDtype::F6E2M3 => Dtype::F6_E2M3,
        TensorDtype::F6E3M2 => Dtype::F6_E3M2,
        TensorDtype::F8E8M0 => Dtype::F8_E8M0,
        TensorDtype::C64 => Dtype::C64,
    }
}

fn from_safetensors_dtype(dtype: Dtype) -> Result<TensorDtype, CodecError> {
    match dtype {
        Dtype::BOOL => Ok(TensorDtype::Bool),
        Dtype::U8 => Ok(TensorDtype::U8),
        Dtype::U16 => Ok(TensorDtype::U16),
        Dtype::U32 => Ok(TensorDtype::U32),
        Dtype::U64 => Ok(TensorDtype::U64),
        Dtype::I8 => Ok(TensorDtype::I8),
        Dtype::I16 => Ok(TensorDtype::I16),
        Dtype::I32 => Ok(TensorDtype::I32),
        Dtype::I64 => Ok(TensorDtype::I64),
        Dtype::F8_E4M3 => Ok(TensorDtype::F8E4M3),
        Dtype::F8_E5M2 => Ok(TensorDtype::F8E5M2),
        Dtype::F16 => Ok(TensorDtype::F16),
        Dtype::BF16 => Ok(TensorDtype::BF16),
        Dtype::F32 => Ok(TensorDtype::F32),
        Dtype::F64 => Ok(TensorDtype::F64),
        Dtype::F4 => Ok(TensorDtype::F4),
        Dtype::F6_E2M3 => Ok(TensorDtype::F6E2M3),
        Dtype::F6_E3M2 => Ok(TensorDtype::F6E3M2),
        Dtype::F8_E8M0 => Ok(TensorDtype::F8E8M0),
        Dtype::C64 => Ok(TensorDtype::C64),
        other => Err(CodecError::UnsupportedDtype(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glowstick::{num::U3, Dyn, Shape1, Shape2};
    use serde::{Deserialize, Serialize};

    struct Batch;
    struct Width;

    struct Image;
    impl TensorPortSpec for Image {
        type Shape = Shape2<Dyn<Batch>, U3>;
        const NAME: &'static str = "image";
        const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
    }

    struct Mask;
    impl TensorPortSpec for Mask {
        type Shape = Shape1<Dyn<Batch>>;
        const NAME: &'static str = "mask";
        const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::U8);
    }

    struct Scores;
    impl TensorPortSpec for Scores {
        type Shape = Shape2<Dyn<Batch>, Dyn<Width>>;
        const NAME: &'static str = "scores";
        const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
    }

    struct ScoresGradient;
    impl TensorPortSpec for ScoresGradient {
        type Shape = Shape2<Dyn<Batch>, Dyn<Width>>;
        const NAME: &'static str = "scores_gradient";
        const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
    }

    struct ImageGradient;
    impl TensorPortSpec for ImageGradient {
        type Shape = Shape2<Dyn<Batch>, U3>;
        const NAME: &'static str = "image_gradient";
        const DTYPE: DtypeConstraint = DtypeConstraint::Exact(TensorDtype::F32);
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct Metadata {
        request: u32,
    }

    struct ExampleContract;
    impl TensorContract for ExampleContract {
        type Input = TensorBundleSpec<(Image, Mask)>;
        type Output = SingleTensorSpec<Scores>;
        type Metadata = Metadata;
        const ID: ContractId = ContractId::from_u128(0x112233445566778899aabbccddeeff00);
        const VERSION: u32 = 7;
    }

    impl BackwardContract for ExampleContract {
        type OutputGrad = SingleTensorSpec<ScoresGradient>;
        type InputGrad = SingleTensorSpec<ImageGradient>;
    }

    #[test]
    fn backward_contract_round_trips_both_gradient_directions() {
        let metadata = Metadata { request: 42 };
        let output_gradient = RawTensor {
            name: "scores_gradient".into(),
            dtype: TensorDtype::F32,
            shape: vec![2, 4],
            data: vec![0; 2 * 4 * 4],
        };
        let encoded = encode_output_gradient::<ExampleContract>(
            std::slice::from_ref(&output_gradient),
            &metadata,
        )
        .unwrap();
        let decoded = decode_output_gradient::<ExampleContract>(&encoded).unwrap();
        assert_eq!(decoded.metadata, metadata);
        assert_eq!(decoded.tensors, vec![output_gradient]);

        let input_gradient = RawTensor {
            name: "image_gradient".into(),
            dtype: TensorDtype::F32,
            shape: vec![2, 3],
            data: vec![0; 2 * 3 * 4],
        };
        let encoded = encode_input_gradient::<ExampleContract>(
            std::slice::from_ref(&input_gradient),
            &metadata,
        )
        .unwrap();
        let decoded = decode_input_gradient::<ExampleContract>(&encoded).unwrap();
        assert_eq!(decoded.tensors, vec![input_gradient]);

        let capability = backward_operation_capability::<ExampleContract>();
        let backward = capability.backward.expect("backward descriptor");
        assert_eq!(backward.descriptor.inputs[0].name, "scores_gradient");
        assert_eq!(backward.descriptor.outputs[0].name, "image_gradient");
        assert_eq!(
            backward.descriptor_hash,
            descriptor_hash(&backward.descriptor)
        );
    }

    struct OtherContract;
    impl TensorContract for OtherContract {
        type Input = TensorBundleSpec<(Image, Mask)>;
        type Output = SingleTensorSpec<Scores>;
        type Metadata = Metadata;
        const ID: ContractId = ContractId::from_u128(0xff2233445566778899aabbccddeeff00);
        const VERSION: u32 = 7;
    }

    #[test]
    fn qwen_compatibility_contract_has_stable_dark_token_ports() {
        let descriptor = QwenDarkInference::descriptor();
        assert_eq!(descriptor.id, QwenDarkInference::ID);
        assert_eq!(descriptor.version, 1);
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|port| port.name.as_str())
                .collect::<Vec<_>>(),
            vec!["predictions", "dark_token_ids", "dark_log_probs"]
        );
        assert_eq!(
            descriptor
                .outputs
                .iter()
                .map(|port| port.name.as_str())
                .collect::<Vec<_>>(),
            vec!["predictions", "dark_token_ids", "dark_log_probs"]
        );
    }

    fn input_tensors() -> Vec<RawTensor> {
        vec![
            RawTensor {
                name: "image".into(),
                dtype: TensorDtype::F32,
                shape: vec![2, 3],
                data: vec![0; 2 * 3 * 4],
            },
            RawTensor {
                name: "mask".into(),
                dtype: TensorDtype::U8,
                shape: vec![2],
                data: vec![1, 0],
            },
        ]
    }

    fn rewrite_envelope(mut frame: Vec<u8>, update: impl FnOnce(&mut TensorEnvelope)) -> Vec<u8> {
        let envelope_len = u32::from_le_bytes(
            frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
                .try_into()
                .unwrap(),
        ) as usize;
        let end = FRAME_PREFIX_LEN + envelope_len;
        let mut envelope: TensorEnvelope = from_bytes(&frame[FRAME_PREFIX_LEN..end]).unwrap();
        let payload = frame[end..].to_vec();
        update(&mut envelope);
        let encoded = to_allocvec(&envelope).unwrap();
        let encoded_len = u32::try_from(encoded.len()).unwrap();
        frame.truncate(FRAME_MAGIC.len());
        frame.extend_from_slice(&encoded_len.to_le_bytes());
        frame.extend_from_slice(&encoded);
        frame.extend_from_slice(&payload);
        frame
    }

    fn replace_safetensors(frame: Vec<u8>, tensor_bytes: &[u8]) -> Vec<u8> {
        let envelope_len = u32::from_le_bytes(
            frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
                .try_into()
                .unwrap(),
        ) as usize;
        let envelope_end = FRAME_PREFIX_LEN + envelope_len;
        let mut envelope: TensorEnvelope =
            from_bytes(&frame[FRAME_PREFIX_LEN..envelope_end]).unwrap();
        let metadata_end = envelope_end + envelope.metadata_len as usize;
        envelope.tensor_len = tensor_bytes.len() as u64;
        let encoded = to_allocvec(&envelope).unwrap();
        let encoded_len = u32::try_from(encoded.len()).unwrap();
        let mut replaced = Vec::new();
        replaced.extend_from_slice(FRAME_MAGIC);
        replaced.extend_from_slice(&encoded_len.to_le_bytes());
        replaced.extend_from_slice(&encoded);
        replaced.extend_from_slice(&frame[envelope_end..metadata_end]);
        replaced.extend_from_slice(tensor_bytes);
        replaced
    }

    fn unchecked_safetensors(tensors: &[RawTensor]) -> Vec<u8> {
        let views = tensors.iter().map(|tensor| {
            (
                tensor.name.as_str(),
                TensorView::new(
                    to_safetensors_dtype(tensor.dtype),
                    tensor.shape.clone(),
                    &tensor.data,
                )
                .unwrap(),
            )
        });
        safetensors::tensor::serialize(views, None).unwrap()
    }

    fn corrupt_first_data_offset(mut frame: Vec<u8>) -> Vec<u8> {
        let envelope_len = u32::from_le_bytes(
            frame[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
                .try_into()
                .unwrap(),
        ) as usize;
        let envelope_end = FRAME_PREFIX_LEN + envelope_len;
        let envelope: TensorEnvelope = from_bytes(&frame[FRAME_PREFIX_LEN..envelope_end]).unwrap();
        let tensor_start = envelope_end + envelope.metadata_len as usize;
        let needle = b"data_offsets\":[0,";
        let relative = frame[tensor_start..]
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("safetensors header contains data offsets");
        let value_start = tensor_start + relative + needle.len();
        let closing = frame[value_start..]
            .iter()
            .position(|byte| *byte == b']')
            .map(|index| value_start + index)
            .unwrap();
        frame[closing - 1] = if frame[closing - 1] == b'9' {
            b'8'
        } else {
            b'9'
        };
        frame
    }

    #[test]
    fn round_trip_named_bundle_and_metadata() {
        let tensors = input_tensors();
        let frame = encode_input::<ExampleContract>(&tensors, &Metadata { request: 42 }).unwrap();
        let decoded = decode_input::<ExampleContract>(&frame).unwrap();
        assert_eq!(decoded.metadata, Metadata { request: 42 });
        assert_eq!(decoded.tensors, tensors);
        assert_eq!(decoded.envelope.tensor_encoding, EncodingId::SAFETENSORS_V1);
    }

    #[test]
    fn streaming_prefix_contains_the_complete_safetensors_header() {
        let frame =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 42 }).unwrap();
        let header = tensor_stream_header(&frame).unwrap();
        assert!(frame.starts_with(&header));
        assert!(header.len() < frame.len());

        let envelope_len = u32::from_le_bytes(
            header[FRAME_MAGIC.len()..FRAME_PREFIX_LEN]
                .try_into()
                .unwrap(),
        ) as usize;
        let envelope_end = FRAME_PREFIX_LEN + envelope_len;
        let envelope: TensorEnvelope = from_bytes(&header[FRAME_PREFIX_LEN..envelope_end]).unwrap();
        let tensor_start = envelope_end + envelope.metadata_len as usize;
        let safe_header_len = u64::from_le_bytes(
            header[tensor_start..tensor_start + size_of::<u64>()]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(
            header.len(),
            tensor_start + size_of::<u64>() + safe_header_len
        );
        assert_eq!(
            validate_tensor_stream_header(
                &ExampleContract::descriptor(),
                ContractSide::Input,
                &header,
            )
            .unwrap(),
            frame.len() as u64,
        );
    }

    #[test]
    fn streaming_prefix_rejects_invalid_schema_before_tensor_data_arrives() {
        let frame =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 42 }).unwrap();
        let mut header = tensor_stream_header(&frame).unwrap();
        let dtype = header
            .windows(b"F32".len())
            .position(|window| window == b"F32")
            .expect("header contains the image dtype");
        header[dtype..dtype + 3].copy_from_slice(b"I32");
        assert!(matches!(
            validate_tensor_stream_header(
                &ExampleContract::descriptor(),
                ContractSide::Input,
                &header,
            ),
            Err(CodecError::DtypeMismatch { .. })
        ));

        let malformed = corrupt_first_data_offset(frame);
        let header = tensor_stream_header(&malformed).unwrap();
        assert!(matches!(
            validate_tensor_stream_header(
                &ExampleContract::descriptor(),
                ContractSide::Input,
                &header,
            ),
            Err(CodecError::Safetensors(_))
        ));
    }

    #[test]
    fn contract_descriptor_serialization_and_hash_are_golden() {
        let descriptor = ExampleContract::descriptor();
        assert_eq!(
            to_allocvec(&descriptor).unwrap(),
            vec![
                17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 0, 7, 2, 5,
                105, 109, 97, 103, 101, 2, 2, 0, 3, 1, 13, 1, 4, 109, 97, 115, 107, 1, 2, 1, 1, 1,
                1, 6, 115, 99, 111, 114, 101, 115, 2, 2, 2, 1, 13, 1,
            ]
        );
        assert_eq!(
            descriptor_hash(&descriptor),
            ContractHash([
                19, 157, 146, 194, 212, 212, 107, 125, 99, 208, 244, 113, 161, 241, 145, 172, 41,
                101, 112, 78, 41, 108, 179, 127, 141, 127, 18, 77, 172, 154, 222, 231,
            ])
        );
    }

    #[test]
    fn rejects_wrong_contract_and_version() {
        let frame =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 1 }).unwrap();
        assert!(matches!(
            decode_input::<OtherContract>(&frame),
            Err(CodecError::ContractIdMismatch)
        ));

        let frame = rewrite_envelope(frame, |envelope| envelope.contract_version += 1);
        assert!(matches!(
            decode_input::<ExampleContract>(&frame),
            Err(CodecError::ContractVersionMismatch { .. })
        ));
    }

    #[test]
    fn rejects_wrong_dtype_rank_and_static_dimension() {
        let base =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 1 }).unwrap();

        let mut wrong_dtype = input_tensors();
        wrong_dtype[0].dtype = TensorDtype::I32;
        let frame = replace_safetensors(base.clone(), &unchecked_safetensors(&wrong_dtype));
        assert!(matches!(
            decode_input::<ExampleContract>(&frame),
            Err(CodecError::DtypeMismatch { .. })
        ));

        let mut wrong_rank = input_tensors();
        wrong_rank[0].shape = vec![6];
        let frame = replace_safetensors(base.clone(), &unchecked_safetensors(&wrong_rank));
        assert!(matches!(
            decode_input::<ExampleContract>(&frame),
            Err(CodecError::RankMismatch { .. })
        ));

        let mut wrong_dimension = input_tensors();
        wrong_dimension[0].shape = vec![3, 2];
        let frame = replace_safetensors(base, &unchecked_safetensors(&wrong_dimension));
        assert!(matches!(
            decode_input::<ExampleContract>(&frame),
            Err(CodecError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn accepts_independent_dynamic_dimensions() {
        let base =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 1 }).unwrap();
        let mut tensors = input_tensors();
        tensors[1].shape = vec![3];
        tensors[1].data = vec![1, 0, 1];
        let frame = replace_safetensors(base, &unchecked_safetensors(&tensors));
        assert!(decode_input::<ExampleContract>(&frame).is_ok());
    }

    #[test]
    fn rejects_malformed_offsets_and_unknown_encodings() {
        let frame =
            encode_input::<ExampleContract>(&input_tensors(), &Metadata { request: 1 }).unwrap();
        let malformed = corrupt_first_data_offset(frame.clone());
        assert!(matches!(
            decode_input::<ExampleContract>(&malformed),
            Err(CodecError::Safetensors(_))
        ));

        let unknown = rewrite_envelope(frame.clone(), |envelope| {
            envelope.tensor_encoding = EncodingId(999)
        });
        assert!(matches!(
            decode_input::<ExampleContract>(&unknown),
            Err(CodecError::UnsupportedTensorEncoding(EncodingId(999)))
        ));

        let unknown = rewrite_envelope(frame.clone(), |envelope| {
            envelope.metadata_encoding = EncodingId(998)
        });
        assert!(matches!(
            decode_input::<ExampleContract>(&unknown),
            Err(CodecError::UnsupportedMetadataEncoding(EncodingId(998)))
        ));

        let unknown = rewrite_envelope(frame, |envelope| envelope.envelope_version = 2);
        assert!(matches!(
            decode_input::<ExampleContract>(&unknown),
            Err(CodecError::UnsupportedEnvelopeVersion(2))
        ));
    }
}
