//! Shared types for the black-hole workspace.

use std::{fmt, marker::PhantomData};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

pub const IM_START: u32 = 248045;
pub const IM_END: u32 = 248046;
pub const PAD: u32 = 248044;
pub const THINK_OPEN: u32 = 248068;
pub const THINK_CLOSE: u32 = 248069;
/// Current generic Mass operation protocol version.
pub const MASS_OPERATION_PROTOCOL_VERSION: u16 = 2;
/// Current progressive artifact-transfer protocol version.
pub const TRANSFER_PROTOCOL_VERSION: u16 = 1;

/// Opaque identifier for objects stored in void.
pub type ObjectId = Uuid;

/// A zero-cost, typed reference to an object stored in void.
///
/// Only the UUID is serialized. The payload type exists solely to prevent an
/// object containing one kind of value from being used where another kind is
/// expected.
#[repr(transparent)]
pub struct ObjectRef<T> {
    id: ObjectId,
    marker: PhantomData<fn() -> T>,
}

impl<T> ObjectRef<T> {
    pub const fn new(id: ObjectId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    pub const fn id(&self) -> ObjectId {
        self.id
    }

    pub const fn into_id(self) -> ObjectId {
        self.id
    }
}

impl<T> Clone for ObjectRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ObjectRef<T> {}

impl<T> PartialEq for ObjectRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for ObjectRef<T> {}

impl<T> std::hash::Hash for ObjectRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> fmt::Debug for ObjectRef<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("ObjectRef").field(&self.id).finish()
    }
}

impl<T> fmt::Display for ObjectRef<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(formatter)
    }
}

impl<T> From<ObjectId> for ObjectRef<T> {
    fn from(id: ObjectId) -> Self {
        Self::new(id)
    }
}

impl<T> From<ObjectRef<T>> for ObjectId {
    fn from(reference: ObjectRef<T>) -> Self {
        reference.id
    }
}

impl<T> Serialize for ObjectRef<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.id.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for ObjectRef<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        ObjectId::deserialize(deserializer).map(Self::new)
    }
}

/// Typed handle for an artifact assembled from immutable Void chunks.
#[repr(transparent)]
pub struct TransferRef<T> {
    id: ObjectId,
    marker: PhantomData<fn() -> T>,
}

impl<T> TransferRef<T> {
    pub const fn new(id: ObjectId) -> Self {
        Self {
            id,
            marker: PhantomData,
        }
    }

    pub const fn id(&self) -> ObjectId {
        self.id
    }
}

impl<T> Clone for TransferRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for TransferRef<T> {}

impl<T> PartialEq for TransferRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for TransferRef<T> {}

impl<T> std::hash::Hash for TransferRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> fmt::Debug for TransferRef<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("TransferRef")
            .field(&self.id)
            .finish()
    }
}

impl<T> Serialize for TransferRef<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.id.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for TransferRef<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        ObjectId::deserialize(deserializer).map(Self::new)
    }
}

/// Typed live-stream location with a durable chunk-transfer fallback.
///
/// The ticket is persisted separately so the location remains compact enough
/// to pass through schedulers while still carrying everything needed to
/// reconnect to the source.
pub struct StreamRef<T> {
    pub ticket_id: ObjectId,
    pub fallback_transfer_id: ObjectId,
    marker: PhantomData<fn() -> T>,
}

impl<T> StreamRef<T> {
    pub const fn new(ticket_id: ObjectId, fallback_transfer_id: ObjectId) -> Self {
        Self {
            ticket_id,
            fallback_transfer_id,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for StreamRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for StreamRef<T> {}

impl<T> PartialEq for StreamRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ticket_id == other.ticket_id && self.fallback_transfer_id == other.fallback_transfer_id
    }
}

impl<T> Eq for StreamRef<T> {}

impl<T> std::hash::Hash for StreamRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ticket_id.hash(state);
        self.fallback_transfer_id.hash(state);
    }
}

impl<T> fmt::Debug for StreamRef<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamRef")
            .field("ticket_id", &self.ticket_id)
            .field("fallback_transfer_id", &self.fallback_transfer_id)
            .finish()
    }
}

impl<T> Serialize for StreamRef<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (self.ticket_id, self.fallback_transfer_id).serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for StreamRef<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (ticket_id, fallback_transfer_id) = <(ObjectId, ObjectId)>::deserialize(deserializer)?;
        Ok(Self::new(ticket_id, fallback_transfer_id))
    }
}

/// Location-independent reference carried between typed flows.
///
/// A normal resolver accepts only committed data. `Transfer` resolves a
/// committed transfer manifest and its immutable chunks; `Stream` may use the
/// live source but always retains the transfer fallback for replay.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub enum ArtifactRef<T> {
    Committed(ObjectRef<T>),
    Transfer(TransferRef<T>),
    Stream(StreamRef<T>),
}

impl<T> ArtifactRef<T> {
    pub const fn committed(reference: ObjectRef<T>) -> Self {
        Self::Committed(reference)
    }

    pub const fn from_object_id(id: ObjectId) -> Self {
        Self::Committed(ObjectRef::new(id))
    }

    pub const fn transfer(reference: TransferRef<T>) -> Self {
        Self::Transfer(reference)
    }

    pub const fn stream(reference: StreamRef<T>) -> Self {
        Self::Stream(reference)
    }

    pub const fn committed_object_ref(&self) -> Option<ObjectRef<T>> {
        match self {
            Self::Committed(reference) => Some(*reference),
            Self::Transfer(_) | Self::Stream(_) => None,
        }
    }

    /// Durable object or transfer ID used to resolve this artifact.
    pub const fn durable_id(&self) -> ObjectId {
        match self {
            Self::Committed(reference) => reference.id(),
            Self::Transfer(reference) => reference.id(),
            Self::Stream(reference) => reference.fallback_transfer_id,
        }
    }

    /// Compatibility accessor for callers that can consume only committed
    /// objects. New generic code should branch on the location or use
    /// `durable_id`.
    pub const fn object_id(&self) -> ObjectId {
        self.durable_id()
    }
}

impl<T> Clone for ArtifactRef<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Committed(reference) => Self::Committed(*reference),
            Self::Transfer(reference) => Self::Transfer(*reference),
            Self::Stream(reference) => Self::Stream(*reference),
        }
    }
}

impl<T> Copy for ArtifactRef<T> {}

impl<T> PartialEq for ArtifactRef<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Committed(left), Self::Committed(right)) => left == right,
            (Self::Transfer(left), Self::Transfer(right)) => left == right,
            (Self::Stream(left), Self::Stream(right)) => left == right,
            _ => false,
        }
    }
}

impl<T> Eq for ArtifactRef<T> {}

impl<T> std::hash::Hash for ArtifactRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Committed(reference) => reference.hash(state),
            Self::Transfer(reference) => reference.hash(state),
            Self::Stream(reference) => reference.hash(state),
        }
    }
}

impl<T> fmt::Debug for ArtifactRef<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Committed(reference) => formatter
                .debug_tuple("ArtifactRef::Committed")
                .field(reference)
                .finish(),
            Self::Transfer(reference) => formatter
                .debug_tuple("ArtifactRef::Transfer")
                .field(reference)
                .finish(),
            Self::Stream(reference) => formatter
                .debug_tuple("ArtifactRef::Stream")
                .field(reference)
                .finish(),
        }
    }
}

impl<T> From<ObjectRef<T>> for ArtifactRef<T> {
    fn from(reference: ObjectRef<T>) -> Self {
        Self::committed(reference)
    }
}

// ---------------------------------------------------------------------------
// Tensor operation contract and artifact wire format
// ---------------------------------------------------------------------------

/// Stable, application-assigned identity for a tensor operation contract.
///
/// Contract IDs are deliberately explicit rather than derived from Rust type
/// names, which are not stable across compilers or builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContractId(pub [u8; 16]);

impl ContractId {
    pub const fn from_u128(value: u128) -> Self {
        Self(value.to_be_bytes())
    }
}

/// SHA-256 digest of the canonical postcard representation of a contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContractHash(pub [u8; 32]);

/// Concrete data types understood by the v1 dense tensor codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDtype {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F8E4M3,
    F8E5M2,
    F16,
    BF16,
    F32,
    F64,
    F4,
    F6E2M3,
    F6E3M2,
    F8E8M0,
    C64,
}

/// Dtype constraint for a named tensor port.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DtypeConstraint {
    Any,
    Exact(TensorDtype),
    OneOf(Vec<TensorDtype>),
}

impl DtypeConstraint {
    pub fn accepts(&self, dtype: TensorDtype) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(expected) => *expected == dtype,
            Self::OneOf(expected) => expected.contains(&dtype),
        }
    }
}

/// Runtime dimension rule for a tensor port.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimensionDescriptor {
    /// A dimension whose concrete size is fixed by the contract.
    Static(u64),
    /// A runtime dimension shared by every occurrence of this label.
    Symbolic(String),
    /// An unconstrained runtime dimension.
    Dynamic,
}

/// Memory-layout constraint for a tensor port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutConstraint {
    Any,
    /// Dense row-major storage. This is the only layout emitted by the v1
    /// safetensors codec.
    Contiguous,
}

/// Descriptor for one named tensor in an input or output bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorPortDescriptor {
    pub name: String,
    pub dimensions: Vec<DimensionDescriptor>,
    pub dtype: DtypeConstraint,
    pub layout: LayoutConstraint,
}

/// Complete distributed identity and schema for an operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractDescriptor {
    pub id: ContractId,
    pub version: u32,
    pub inputs: Vec<TensorPortDescriptor>,
    pub outputs: Vec<TensorPortDescriptor>,
}

/// Extensible identifier for an on-wire encoding.
///
/// Unknown values deserialize successfully so protocol implementations can
/// reject them explicitly instead of coupling wire compatibility to an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodingId(pub u16);

impl EncodingId {
    pub const SAFETENSORS_V1: Self = Self(1);
    pub const POSTCARD_V1: Self = Self(2);
}

/// Identifies which half of a contract a tensor bundle inhabits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractSide {
    Input,
    Output,
}

/// Authoritative Black Hole Sun header surrounding an encoded tensor bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEnvelope {
    pub envelope_version: u16,
    pub contract_id: ContractId,
    pub contract_version: u32,
    pub contract_hash: ContractHash,
    pub side: ContractSide,
    pub tensor_encoding: EncodingId,
    pub metadata_encoding: EncodingId,
    pub metadata_len: u64,
    pub tensor_len: u64,
}

/// Runtime contract and codec declaration used by Mass discovery and start.
///
/// The hash covers `descriptor`'s canonical postcard representation. Keeping
/// both values on the wire lets a receiver reject descriptors that were
/// corrupted or paired with the wrong distributed identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCapability {
    pub descriptor: ContractDescriptor,
    pub descriptor_hash: ContractHash,
    pub tensor_encodings: Vec<EncodingId>,
    pub metadata_encodings: Vec<EncodingId>,
    /// Lifecycle verbs implemented by this operation host.
    pub operations: OperationCapabilities,
    /// Reverse-mode tensor contract. Present only for backward-capable hosts.
    #[serde(default)]
    pub backward: Option<Box<BackwardCapability>>,
}

/// Runtime descriptor for the reverse half of an operation contract.
///
/// `descriptor.inputs` describes the gradient received from downstream and
/// `descriptor.outputs` describes the gradient returned upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackwardCapability {
    pub descriptor: ContractDescriptor,
    pub descriptor_hash: ContractHash,
}

/// Runtime capability set for a hosted operation.
///
/// The tensor contract describes the values crossing the operation boundary;
/// this set describes what the selected implementation can do with its
/// internal state. Keeping the two independent allows the same contract to be
/// hosted by forward-only and optimizing backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCapabilities {
    pub forward: bool,
    pub backward: bool,
    pub step: bool,
    pub reset: bool,
    pub perturb: bool,
    pub optimize: bool,
    pub checkpoint: bool,
    pub fuse: bool,
    pub query: bool,
}

impl OperationCapabilities {
    pub const FORWARD_ONLY: Self = Self {
        forward: true,
        backward: false,
        step: false,
        reset: false,
        perturb: false,
        optimize: false,
        checkpoint: false,
        fuse: false,
        query: false,
    };

    pub const OPTIMIZING: Self = Self {
        forward: true,
        backward: false,
        step: false,
        reset: true,
        perturb: true,
        optimize: true,
        checkpoint: true,
        fuse: true,
        query: true,
    };

    pub const fn satisfies(self, required: Self) -> bool {
        (!required.forward || self.forward)
            && (!required.backward || self.backward)
            && (!required.step || self.step)
            && (!required.reset || self.reset)
            && (!required.perturb || self.perturb)
            && (!required.optimize || self.optimize)
            && (!required.checkpoint || self.checkpoint)
            && (!required.fuse || self.fuse)
            && (!required.query || self.query)
    }
}

/// Opaque, implementation-owned configuration carried by unified instance
/// start and query messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationConfig {
    pub encoding: EncodingId,
    pub data: Vec<u8>,
}

/// Location-erased artifact reference used by the generic Mass wire protocol.
///
/// Operation-typed clients convert this to and from `ArtifactRef<Op::Input>`
/// and `ArtifactRef<Op::Output>`. Future transfer and stream locations can be
/// added without changing the generic start/forward variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationArtifactRef {
    Committed(ObjectId),
    Transfer(ObjectId),
    Stream {
        ticket_id: ObjectId,
        fallback_transfer_id: ObjectId,
    },
}

impl OperationArtifactRef {
    pub const fn committed(id: ObjectId) -> Self {
        Self::Committed(id)
    }

    pub const fn object_id(self) -> ObjectId {
        match self {
            Self::Committed(id) => id,
            Self::Transfer(id) => id,
            Self::Stream {
                fallback_transfer_id,
                ..
            } => fallback_transfer_id,
        }
    }

    pub const fn durable_id(self) -> ObjectId {
        self.object_id()
    }

    pub const fn into_typed<T>(self) -> ArtifactRef<T> {
        match self {
            Self::Committed(id) => ArtifactRef::Committed(ObjectRef::new(id)),
            Self::Transfer(id) => ArtifactRef::Transfer(TransferRef::new(id)),
            Self::Stream {
                ticket_id,
                fallback_transfer_id,
            } => ArtifactRef::Stream(StreamRef::new(ticket_id, fallback_transfer_id)),
        }
    }
}

impl<T> From<ArtifactRef<T>> for OperationArtifactRef {
    fn from(reference: ArtifactRef<T>) -> Self {
        match reference {
            ArtifactRef::Committed(reference) => Self::Committed(reference.id()),
            ArtifactRef::Transfer(reference) => Self::Transfer(reference.id()),
            ArtifactRef::Stream(reference) => Self::Stream {
                ticket_id: reference.ticket_id,
                fallback_transfer_id: reference.fallback_transfer_id,
            },
        }
    }
}

impl TensorEnvelope {
    pub const VERSION: u16 = 1;
}

// ---------------------------------------------------------------------------
// Progressive artifact transfer protocol
// ---------------------------------------------------------------------------

/// SHA-256 digest used for individual chunks and complete transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransferHash(pub [u8; 32]);

/// Immutable declaration written before any transfer chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferBegin {
    pub protocol_version: u16,
    pub transfer_id: ObjectId,
    pub envelope: TensorEnvelope,
    /// Safetensors header bytes sent before payload frames so receivers can
    /// validate concrete names, dtypes, and dimensions before allocation.
    pub tensor_header: Vec<u8>,
    pub expected_chunks: u32,
    pub expected_len: u64,
    pub expected_hash: TransferHash,
    /// Unix timestamp in milliseconds. An uncommitted transfer is aborted
    /// after this lease expires.
    pub deadline_unix_ms: u64,
    /// SHA-256 digest of the bearer token in a [`TransferTicket`].
    pub authorization_hash: TransferHash,
}

/// Descriptor for one immutable Void object in a progressive transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferChunk {
    pub index: u32,
    pub object_id: ObjectId,
    pub len: u64,
    pub hash: TransferHash,
}

/// Durable manifest that makes all chunks authoritative and replayable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferManifest {
    pub begin: TransferBegin,
    pub chunks: Vec<TransferChunk>,
    pub committed_unix_ms: u64,
}

/// Terminal record for an explicitly aborted or expired transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferAbort {
    pub begin: TransferBegin,
    pub reason: String,
    pub aborted_unix_ms: u64,
}

/// Durable transfer state stored under `TransferBegin::transfer_id`.
///
/// Replay resolvers accept only `Committed`; begin/chunk state is observable
/// for progressive staging but never authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferRecord {
    InProgress {
        begin: TransferBegin,
        chunks: Vec<TransferChunk>,
        revision: u64,
    },
    Committed(TransferManifest),
    Aborted(TransferAbort),
}

impl TransferRecord {
    pub fn begin(&self) -> &TransferBegin {
        match self {
            Self::InProgress { begin, .. } => begin,
            Self::Committed(manifest) => &manifest.begin,
            Self::Aborted(abort) => &abort.begin,
        }
    }
}

/// Durability required before a streamed operation may publish its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DurabilityPolicy {
    /// The stream is a latency optimization; the transfer must commit to Void
    /// before dependent externally visible output can commit.
    ReplayRequired,
    /// The stream may be consumed without a durable replay artifact.
    Ephemeral,
}

/// Ordering guarantee an operation requires for progressive tensor chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingChunkOrder {
    Sequential,
    Unordered,
}

/// Point at which a streaming operation may finalize its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingFinalization {
    WholeArtifact,
    EndOfAxis,
}

/// Authorization and routing information for a live QUIC tensor stream.
///
/// `source` is an authority string (`ip:port`) rather than a socket type so
/// tickets remain stable across IPv4 and IPv6 deployments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferTicket {
    pub descriptor: ContractDescriptor,
    pub envelope: TensorEnvelope,
    pub tensor_header: Vec<u8>,
    pub transfer_id: ObjectId,
    pub source: String,
    pub authorization: [u8; 32],
    pub expected_len: u64,
    pub expected_hash: TransferHash,
    pub deadline_unix_ms: u64,
    pub durability: DurabilityPolicy,
    pub eventual_void_id: ObjectId,
}

/// Frames carried on a live transfer stream. The begin descriptor is always
/// sent first, so a receiver can validate and allocate before tensor bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferStreamFrame {
    Begin(TransferBegin),
    Chunk {
        index: u32,
        data: Vec<u8>,
        hash: TransferHash,
    },
    Commit {
        aggregate_hash: TransferHash,
    },
    Abort {
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Mass wire protocol (black-hole-mass <-> client)
// ---------------------------------------------------------------------------

/// Request sent by a client to the mass QUIC server.
#[derive(Debug, Serialize, Deserialize)]
pub enum MassIn {
    /// Select an implementation and start one black-box operation instance.
    StartInstance {
        protocol_version: u16,
        instance_id: Uuid,
        /// Complete contract, codec, and required-lifecycle declaration.
        capability: OperationCapability,
        /// Backend-owned configuration (for example Qwen sampling options).
        config: Option<OperationConfig>,
    },
    /// Invoke the instance's forward operation on a typed artifact.
    Invoke {
        protocol_version: u16,
        instance_id: Uuid,
        input: OperationArtifactRef,
    },
    /// Run reverse-mode differentiation for one cached forward micro-batch.
    BackwardInstance {
        protocol_version: u16,
        instance_id: Uuid,
        grad_input: OperationArtifactRef,
    },
    /// Apply accumulated parameter gradients and clear step-local state.
    StepInstance {
        instance_id: Uuid,
    },
    ResetInstance {
        instance_id: Uuid,
    },
    PerturbUpInstance {
        instance_id: Uuid,
        seed: u64,
    },
    PerturbDownInstance {
        instance_id: Uuid,
    },
    CheckpointInstance {
        instance_id: Uuid,
    },
    OptimizeInstance {
        instance_id: Uuid,
        loss_up: f32,
        loss_down: f32,
    },
    FuseInstance {
        instance_id: Uuid,
        checkpoint_id: ObjectId,
        contribution: f32,
    },
    ShutdownInstance {
        instance_id: Uuid,
    },
    /// Query opaque implementation-owned runtime parameters.
    QueryInstance {
        instance_id: Uuid,
    },
    /// Query recursive hosted-instance capacity for this Mass subtree.
    QueryInstanceCapacity,
    /// Register a one-hop tunnel worker with a root mass.
    RegisterTunnel {
        /// Stable worker identity used by parent masss to match reconnects.
        worker_id: Uuid,
        /// Optional total model capacity advertised by this worker subtree (defaults to 1).
        max_instances: Option<usize>,
        /// Complete operation capabilities advertised by this worker.
        capabilities: WorkerCapabilities,
    },
    /// Update the advertised tunnel capacity for an already-registered worker token.
    UpdateTunnelCapacity {
        /// Root/parent-issued token for the registered worker.
        token: Uuid,
        /// Optional total model capacity for this worker subtree (defaults to 1).
        max_instances: Option<usize>,
    },
    /// Forward a hosted-instance operation through a registered tunnel worker.
    TunnelForward {
        /// Root-issued token proving this request was authorized for the worker.
        token: Uuid,
        /// Forwarded hosted-instance operation.
        request: TunnelRequest,
    },
}

/// Response sent by the mass server to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum MassOut {
    /// Acknowledges a lifecycle, perturb, or optimize step.
    Ack,
    /// Forward invocation complete; contains the typed output artifact.
    Invoked { output: OperationArtifactRef },
    /// Checkpoint upload complete; contains the void object ID of model weights.
    Checkpointed { checkpoint_id: ObjectId },
    /// Opaque implementation-owned runtime parameters for an instance.
    Instance { params: OperationConfig },
    /// Recursive hosted-instance capacity for this mass subtree.
    InstanceCapacity { capacity: MassModelCapacity },
    /// Tunnel worker registration complete; contains root-issued auth token.
    TunnelRegistered { token: Uuid },
    /// Weight fusion complete; contains the void object ID of the fused weights.
    Fused { fused_id: ObjectId },
    /// Error from any operation.
    Error { message: String },
}

/// Runtime model parameters resolved for a running mass model instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MassModelParams {
    pub inference_limit: u32,
    pub top_k: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub training_lr: f64,
    pub training_epsilon: f64,
    pub training_z_loss: f64,
    pub training_lb_loss: f64,
    pub training_clip_threshold: f64,
    pub training_perturbation_mode: MassPerturbationMode,
    pub training_error_feedback: MassErrorFeedbackConfig,
    pub is_frozen: bool,
    pub optimize_steps: u32,
    pub oscillation_period_steps: Option<u32>,
    pub oscillation_train_steps: Option<u32>,
    pub oscillation_phase_steps: Option<u32>,
    pub oscillation_warmup_steps: Option<u32>,
}

/// Recursive model-capacity snapshot for a mass server subtree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MassModelCapacity {
    /// Total model-instance capacity (local + descendants). None means unbounded.
    pub total: Option<usize>,
    /// Available model-instance capacity (total minus occupied). None means unbounded.
    pub available: Option<usize>,
    /// Occupied model-instance slots currently routed in this subtree.
    pub occupied: usize,
    /// Per-architecture capacity view across the subtree.
    ///
    /// One entry per architecture any engine in the subtree can serve; empty
    /// when no engine advertises an architecture (legacy builds). For workers
    /// advertising multiple architectures, shared slots count against each.
    pub per_architecture: Vec<(MassArchitecture, MassModelCapacity)>,
}

/// Error-feedback mode selector for QuZO optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassErrorFeedbackMode {
    Off,
    Persistent,
    Replay,
}

/// Per-model QuZO error-feedback configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MassErrorFeedbackConfig {
    Off,
    Persistent { decay: f64, gain: f64 },
    Replay { steps: u32, decay: f64, gain: f64 },
}

/// QuZO perturbation direction mode selector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MassPerturbationMode {
    /// Sample perturbations in the full weight space (default).
    #[default]
    Weight,
    /// Sample factored low-rank activation-space directions for linear weights
    /// using `rank` factors (`LowRank(1)` is the narrowest factored direction).
    LowRank(usize),
}

/// Model architectures a mass engine binary can be compiled for.
///
/// paramecia selects model shapes at compile time via cargo features, so a
/// given `black-hole-mass` build serves exactly one of these (or none).
/// Tunnel workers advertise which one they are so roots can place model
/// instances only on compatible engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MassArchitecture {
    Qwen35_0p8b,
    Qwen35_2b,
    Qwen35_4b,
    Qwen35_9b,
    Qwen35_27b,
    Qwen38_27b,
}

/// Capabilities advertised by a tunnel worker at registration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Architectures this worker's compiled engine can load.
    ///
    /// Empty means the worker predates capability advertising (or was built
    /// without an architecture feature); such workers only receive starts
    /// that carry no architecture requirement.
    pub architectures: Vec<MassArchitecture>,
    /// Complete operation contracts and codecs this worker can host.
    #[serde(default)]
    pub operations: Vec<OperationCapability>,
}

/// Forwardable hosted-instance operation used for root->worker tunnel requests.
#[derive(Debug, Serialize, Deserialize)]
pub enum TunnelRequest {
    StartInstance {
        protocol_version: u16,
        instance_id: Uuid,
        capability: OperationCapability,
        config: Option<OperationConfig>,
    },
    Invoke {
        protocol_version: u16,
        instance_id: Uuid,
        input: OperationArtifactRef,
    },
    BackwardInstance {
        protocol_version: u16,
        instance_id: Uuid,
        grad_input: OperationArtifactRef,
    },
    StepInstance {
        instance_id: Uuid,
    },
    ResetInstance {
        instance_id: Uuid,
    },
    PerturbUpInstance {
        instance_id: Uuid,
        seed: u64,
    },
    PerturbDownInstance {
        instance_id: Uuid,
    },
    CheckpointInstance {
        instance_id: Uuid,
    },
    OptimizeInstance {
        instance_id: Uuid,
        loss_up: f32,
        loss_down: f32,
    },
    FuseInstance {
        instance_id: Uuid,
        checkpoint_id: ObjectId,
        contribution: f32,
    },
    ShutdownInstance {
        instance_id: Uuid,
    },
    QueryInstance {
        instance_id: Uuid,
    },
}

/// Per-model-instance mass configuration overrides.
///
/// Each field is optional; omitted values fall back to server defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MassModelConfig {
    pub top_k: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub inference_limit: Option<u32>,
    pub training_lr: Option<f64>,
    pub training_epsilon: Option<f64>,
    pub training_z_loss: Option<f64>,
    pub training_lb_loss: Option<f64>,
    pub training_clip_threshold: Option<f64>,
    /// Optional QuZO perturbation direction mode for this instance.
    ///
    /// When `None`, mass uses its configured server default.
    pub training_perturbation_mode: Option<MassPerturbationMode>,
    pub training_error_feedback: Option<MassErrorFeedbackConfig>,
    pub frozen: Option<bool>,
    /// Optional optimize-step period for train/freeze oscillation scheduling.
    ///
    /// When set (with `oscillation_train_steps`), mass applies a deterministic
    /// train window each cycle after warmup instead of flipping prior state.
    pub oscillation_period_steps: Option<u32>,
    /// Optional count of trainable optimize steps in each oscillation cycle.
    ///
    /// Must be less than or equal to `oscillation_period_steps`.
    pub oscillation_train_steps: Option<u32>,
    /// Optional per-instance phase shift (in optimize steps), modulo period.
    pub oscillation_phase_steps: Option<u32>,
    /// Optional number of optimize steps to wait before schedule activation.
    ///
    /// Ignored when `oscillation_period_steps` is `None`.
    pub oscillation_warmup_steps: Option<u32>,
    /// Optional checkpoint object to load model weights from for this instance.
    ///
    /// When `None`, mass loads weights from its configured server model path.
    pub checkpoint_id: Option<ObjectId>,
    /// Architecture the serving mass engine must be compiled for.
    ///
    /// When set, a root only routes this instance to local/worker engines
    /// whose advertised capabilities include this architecture, and workers
    /// reject starts their compiled engine cannot serve. When `None`, any
    /// engine may serve the instance.
    pub required_architecture: Option<MassArchitecture>,
}

// ---------------------------------------------------------------------------
// Inference input format (stored in void objects)
// ---------------------------------------------------------------------------

/// A single logit entry (token ID + log probability) for dark prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitEntry {
    pub token_id: u32,
    pub log_prob: f32,
}

/// A dark token position for dark-knowledge transfer between model forward passes.
/// Carries the predicted (committed) token ID and a top-K distribution from a teacher model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkToken {
    /// The predicted (committed) token ID for this position.
    pub predicted: u32,
    /// Top-K logit entries representing the teacher model's distribution at this position.
    pub dark_knowledge: Vec<LogitEntry>,
}

/// Serializable inference input, mirroring paramecia-engine's ModelInput.
/// Stored inside void objects and converted to ModelInput by the mass service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceInput {
    /// Text context (tokenized by the model host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
    /// Dark prompt: a sequence of dark tokens carrying predicted token IDs and
    /// dark-knowledge distributions.
    Dark(Vec<DarkToken>),
}

/// Serializable inference request stored in void objects.
/// Either contains inline sequences or points to an existing InferenceOutput
/// in void that should be converted to dark input for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceRequest {
    /// Inline sequences with explicit inputs.
    Sequences {
        /// Each element is one sequence (a list of inputs concatenated in order).
        sequences: Vec<Vec<InferenceInput>>,
        /// Optional generation cap. If `None`, mass applies its server default.
        limit: Option<u32>,
    },
    /// Reference to an existing InferenceOutput in void.
    /// Mass downloads it, converts the results to dark input, and proceeds.
    VoidId {
        /// Void object ID of the InferenceOutput to use as input.
        id: InferenceOutputId,
        /// Optional generation cap. If `None`, mass applies its server default.
        limit: Option<u32>,
    },
}

// ---------------------------------------------------------------------------
// Inference output format (stored in void objects)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceOutput(pub Vec<DarkToken>);

/// Serializable inference output stored in void objects.
/// Contains per-sequence results from a batched forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub results: Vec<SequenceOutput>,
}

/// Typed void reference for a Qwen inference output.
pub type InferenceOutputId = ObjectRef<InferenceOutput>;

/// Input / Output from an Atom.
///
/// The payload type is independent from metadata. Qwen is the default marker
/// for the application-facing adapter; generic paths use
/// `Emission<M, Op::Output>`.
#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "M: Serialize", deserialize = "M: Deserialize<'de>"))]
pub struct Emission<M, T = InferenceOutput> {
    pub metadata: M,
    pub output_id: ArtifactRef<T>,
}

impl<M: Clone, T> Clone for Emission<M, T> {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            output_id: self.output_id,
        }
    }
}

impl<M: fmt::Debug, T> fmt::Debug for Emission<M, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Emission")
            .field("metadata", &self.metadata)
            .field("output_id", &self.output_id)
            .finish()
    }
}

/// Type marker for a persisted emission whose output is `T`.
pub enum EmissionArtifact<T> {
    #[doc(hidden)]
    __Marker(std::convert::Infallible, PhantomData<fn() -> T>),
}

/// Typed void ID for an emission.
///
/// Metadata remains independently generic and is intentionally not part of
/// this reference yet; Stage 4 carries the full operation type through Flux.
pub type EmissionId<T = InferenceOutput> = ObjectRef<EmissionArtifact<T>>;

/// Neutral data-plane message used by operation-typed schedulers.
///
/// Unlike [`Transmission`], this type contains no training-program vocabulary.
/// `T` is the artifact bundle produced by the source operation, so forwarding
/// it to a destination retains the compile-time payload identity.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ArtifactDelivery<T> {
    pub emission_id: EmissionId<T>,
    pub recv: ObjectId,
    pub send: ObjectId,
}

impl<T> Clone for ArtifactDelivery<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ArtifactDelivery<T> {}

impl<T> fmt::Debug for ArtifactDelivery<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactDelivery")
            .field("emission_id", &self.emission_id)
            .field("recv", &self.recv)
            .field("send", &self.send)
            .finish()
    }
}

/// Program-selected control-plane message. Generic graph execution carries
/// artifact deliveries; strategies such as two-sided ZO choose their own
/// control payload instead of adding variants to the shared data plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalControl<C> {
    pub control: C,
    pub recv: ObjectId,
}

/// Input / Output from a Cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transmission {
    Propagation {
        emission_id: EmissionId,
        recv: ObjectId,
        send: ObjectId,
    },
    Potentiation {
        potentiation: Potentiation,
        recv: ObjectId,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Potentiation {
    pub loss_up: f32,
    pub loss_down: f32,
    pub seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Input;
    struct Output;

    #[test]
    fn object_refs_are_zero_cost_and_serialize_as_the_uuid() {
        assert_eq!(size_of::<ObjectRef<Input>>(), size_of::<ObjectId>());
        assert_eq!(size_of::<TransferRef<Input>>(), size_of::<ObjectId>());

        let id = ObjectId::from_u128(7);
        let reference = ObjectRef::<Input>::new(id);
        assert_eq!(
            postcard::to_allocvec(&reference).unwrap(),
            postcard::to_allocvec(&id).unwrap()
        );
        let decoded: ObjectRef<Input> =
            postcard::from_bytes(&postcard::to_allocvec(&reference).unwrap()).unwrap();
        assert_eq!(decoded.id(), id);
    }

    #[test]
    fn artifact_and_emission_references_retain_payload_types() {
        fn accepts_input(_: ArtifactRef<Input>) {}
        fn accepts_output_emission(_: EmissionId<Output>) {}

        let input = ArtifactRef::from_object_id(ObjectId::from_u128(11));
        let output = EmissionId::<Output>::new(ObjectId::from_u128(12));
        accepts_input(input);
        accepts_output_emission(output);
    }

    #[test]
    fn artifact_locations_round_trip_through_erased_mass_references() {
        let transfer = ArtifactRef::<Input>::transfer(TransferRef::new(ObjectId::from_u128(21)));
        let erased: OperationArtifactRef = transfer.into();
        assert!(matches!(
            erased,
            OperationArtifactRef::Transfer(id) if id == ObjectId::from_u128(21)
        ));
        assert!(matches!(
            erased.into_typed::<Input>(),
            ArtifactRef::Transfer(reference) if reference.id() == ObjectId::from_u128(21)
        ));

        let stream = ArtifactRef::<Input>::stream(StreamRef::new(
            ObjectId::from_u128(22),
            ObjectId::from_u128(23),
        ));
        let encoded = postcard::to_allocvec(&stream).unwrap();
        let decoded: ArtifactRef<Input> = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded, stream);
    }

    #[test]
    fn qwen_inference_id_uses_the_typed_reference_wire_format() {
        let id = ObjectId::from_u128(13);
        let request = InferenceRequest::VoidId {
            id: InferenceOutputId::new(id),
            limit: Some(4),
        };
        let decoded: InferenceRequest =
            postcard::from_bytes(&postcard::to_allocvec(&request).unwrap()).unwrap();
        match decoded {
            InferenceRequest::VoidId { id: decoded, limit } => {
                assert_eq!(decoded.id(), id);
                assert_eq!(limit, Some(4));
            }
            InferenceRequest::Sequences { .. } => panic!("wrong request variant"),
        }
    }
}
