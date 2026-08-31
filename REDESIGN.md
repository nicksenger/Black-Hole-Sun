nick:
mitb, I want your design/architecture opinion on something. I'm thinking of taking this project in a totally different direction, almost a complete 180 but reusing much of the existing code. You see, as written the project is totally coupled to paramecia (and qwen3.* gguf), but this is not really the cool part of the project. The cool part is that it provides a type-level DSL for orchestrating inference & RL-training loops across a distributed system. So I'm thinking of leaning away from paramecia and into the type-level DSL.
Allow me to explain some of my ideas:

if you take a look at ../glowstick, you'll see that I've already built a crate for expressing tensor shapes as Rust types, so I'm thinking: what if instead of being coupled so tightly to qwen inference, the black-hole-mass servers became "generic" over tensors of some Input and Output shape?
We could carry the types through to the VoidInferOps (maybe split it into VoidOps and MassOps<InShape, OutShape>), and then wire up the Flow in such a way that the compiler forbids shape mismatches across the distributed system
The Mass client would be generic over a particular input/output shape, and the tunneling functionality could be shape-dependent/aware (rejecting clients which try to connect with an unsupported Input/Output shape pair)
The Mass server would have a narrower, less paramecia-specific API, and accept some trait implementer for the given Input/Output shapes that must provide the forward operation Tensor<InputShape, Metadata> -> Tensor<OutputShape, Metadata> 

The InferenceOutput no longer deals in DarkTokens (which are fairly qwen/llm specific), and instead becomes generic over a shape and contains a tensor of that shape

The net result/intent would be that Black-Hole-Sun becomes much more general-purpose as a way of orchestrating a distributed system of generic tensor operations in a type-safe manner, rather than just qwen inference.
What do you think of this design direction? No need to change the code yet, I'm just curious to get your opinion.

mitb:
[This is an automated reply from the Man in the Box]
I think this is the right direction, and it matches where the strongest abstraction already is. The reusable part here is the typed orchestration/training topology; paramecia/Qwen should be one implementation of it rather than the definition of it.
My main qualification is that I would make the system generic over an operation contract, not just an (InShape, OutShape) pair. Shape compatibility is necessary, but it is not sufficient: two operations can have identical shapes and incompatible dtype, layout, encoding, semantics, or metadata requirements. Something along these lines feels like the useful center:

trait TensorOp {
type Input: TensorSpec;
type Output: TensorSpec;
type Config;

const CONTRACT_ID: &'static str;
const VERSION: u32;

async fn forward(
&mut self,
input: Tensor<Self::Input>,
) -> Result<Tensor<Self::Output>, Self::Error>;
}


Then MassClient<Op>, MassOps<Op>, and MassServer<Op, Impl> all derive their input/output types from the same contract. That avoids permitting arbitrary shape pairs which happen to fit but do not represent an operation a server actually implements.
I do like splitting VoidInferOps. Right now it combines at least four concerns: object storage, generic inference transport, model lifecycle/training, and LLM tokenization (darken/decode). I would expect the core traits to look more like:


VoidOps for typed object storage;

MassOps<Op> (or ForwardOps<Op>) for start/forward/shutdown;
optional capability traits for perturb/optimize/checkpoint/fuse/reset;
a Qwen/LLM adapter trait for text, tokenization, and dark knowledge.

That separation matters because a generic tensor operation may be forward-only, stateless, frozen, or trainable by a mechanism other than the current QuZO lifecycle. The existing Cell can continue to require the zeroth-order training capabilities, while a more general forward DAG only requires MassOps<Op>.
I would also make the void references typed. At present, EmissionId and InferenceOutputId erase the payload type before it enters most of the Flow, so the compiler cannot distinguish one node's output from another's input. A zero-cost wrapper such as ObjectRef<T>(ObjectId, PhantomData<T>), plus Emission<T, M>, would let a Flow output ObjectRef<Op::Output> and require the next node to accept exactly that type. Metadata can remain independently generic rather than becoming part of the tensor shape itself.
There is an important boundary between the compile-time and distributed guarantees:

Within one compiled Flow, glowstick types can reject incompatible edges.
Across void serialization, separate binaries, rolling deployments, and tunnels, Rust types no longer provide proof. The wire payload still needs a canonical runtime descriptor and validation.

I would put rank/dimensions (including symbolic/dynamic dimensions), dtype, layout/encoding, and a schema/contract version into that descriptor, and derive a stable contract ID/hash from it. MassClient<Op> can be statically typed while sending this erased descriptor on the wire. A server should reject a mismatch at start/handshake and validate the actual tensor again when decoding it. This is especially important for glowstick's dynamic dimensions: sharing the same symbolic type proves the relationship at compile time, but the concrete dimension bindings still have to be checked at runtime.
The tunnel capability mechanism already provides a very natural migration point. WorkerCapabilities.architectures could become (or be supplemented by) advertised operation contracts. Routing should match the full contract, not only a shape pair, and continue pinning subsequent requests for a model/operation instance to the selected worker as it does now.
I would probably migrate incrementally:

Split VoidInferOps and introduce typed object references plus a wire-level contract descriptor, while keeping today's Qwen types working through aliases/adapters.
Carry the payload/operation type through Emission, Transmission, Atom/Cell, and the Jungle Flow edges; add compile-fail tests demonstrating that mismatched shapes cannot compose.
Make Mass accept an injected TensorOp implementation and route tunnel starts by contract.
Move DarkToken, tokenizer behavior, Qwen configuration, and paramecia-specific lifecycle code behind a Qwen adapter crate/module.

That should preserve almost all of the orchestration and routing work while making Qwen a strong first backend/example. So yes: I am strongly in favor of the 180. I think the durable identity of Black-Hole-Sun is “a type-safe distributed tensor-operation and training DSL,” with the current LLM system as one specialization.

nick:
mitb, thanks. I think in general I agree with all of your points here. What are you leaning towards regarding the tensor encoding and descriptor? Just lean into something like safetensors directly or try to make it a bit more flexible?

mitb:
[This is an automated reply from the Man in the Box]
I am leaning toward safetensors as the required v1 payload encoding, while keeping the protocol one layer more flexible than safetensors itself.
I would not invent a new dense-tensor byte format. Safetensors already gives us the important physical facts—concrete shape, dtype, named tensors, offsets, and contiguous data—in a format Candle and other ecosystems understand. It also fits void particularly well: void already stores opaque bytes and supports multipart upload and range reads, so a consumer can read the safetensors header and then fetch only the tensor ranges it needs.
I would keep two concepts separate, though:

The operation contract describes what is allowed: contract/version, symbolic input/output shapes, dtype constraints, named axes/ports if useful, and semantic identity.
The encoded tensor instance describes what arrived: concrete dimensions and dtype, with safetensors as the initial physical encoding.

For example, a contract might expect [Batch, Sequence, U4096]; a particular safetensors object contains [8, 128, 4096]. Decoding binds Batch = 8 and Sequence = 128, then validates every repeated symbolic dimension and constraint. Safetensors cannot express that symbolic relationship or the operation semantics, so I would not try to make its metadata the authoritative descriptor.
Concretely, I picture a small versioned envelope/header roughly containing:

struct TensorEnvelope {
envelope_version: u16,
contract_id: ContractId,
contract_version: u32,
encoding: EncodingId,       // initially safetensors/v1
metadata_encoding: EncodingId,
// payload follows
}


The payload would be safetensors bytes. Typed flow metadata would remain separately encoded (postcard is perfectly reasonable initially), rather than being stuffed into safetensors' string-only __metadata__. The decoder validates the safetensors header against the contract before constructing Tensor<S>.
I would make EncodingId extensible, but not make arbitrary encoding negotiation a goal for the first implementation. V1 can require safetensors for dense tensors everywhere. Later, a contract/server capability could explicitly opt into another codec for a demonstrated need—e.g. sparse tensors, ragged data, a quantized activation format, compression, or device-local transport. Unknown encodings should fail closed, and the selected encoding should be part of the advertised worker contract/capability so a client cannot silently choose something the operation did not declare.
One other reason to preserve that boundary is that safetensors is excellent for dense tensor values but should not become the universal artifact format. Existing GGUF checkpoints can remain GGUF; structured/non-tensor control data can remain postcard; only forward inputs/outputs need the tensor codec abstraction.
I would also consider allowing a contract to describe a named tensor bundle, even if the first API presents a single tensor. Masks, positions, multimodal inputs, and multi-head outputs otherwise force us to redesign the envelope quickly, while safetensors already naturally supports multiple named tensors.
So my preferred balance is:

standardize aggressively on safetensors now;
own a small canonical contract/envelope format above it;
keep the codec field versioned/extensible;
add another tensor encoding only when an actual workload cannot be represented efficiently by safetensors.

That gives us interoperability and avoids format design work now, without making safetensors' current feature set the permanent boundary of Black-Hole-Sun.

nick:
mitb, you reminded me of 2 other thoughts related to this that I wanted to run by you:

The current Void-based transport is convenient for event-replay purposes, but ultimately quite inefficient. The receiver needs to wait for full upload/flush of the data into an object store before it can start receiving bytes. Are there patterns we could follow to improve this? The reason I ask is that for a generic ML/tensor-op system, some of these tensors will get big, which means we may be leaving a lot of time on the table with the current approach. Even a protocol which just splits across several void objects and allows the recipient to start reading as soon as the first is written (and forces it to fail if any of the writes fail) seems like it'd potentially be a big improvement, performance-wise.
The current BlackHole::Sun implies a 2-phase blackbox optimization process with gradient steps in each phase. If the framework becomes more generic/flexible, this doesn't really map naturally to the majority of workloads anymore. What are your thoughts about the overall orchestration pattern?


mitb:
[This is an automated reply from the Man in the Box]
Both thoughts reinforce the same architectural split for me: the durable event/control plane should not also be the only data plane, and the typed graph topology should not also define the training algorithm.
1. Transport: keep Void as the durable plane, add a streaming data plane
The current multipart implementation has an all-or-nothing publication model: parts are uploaded, then UploadFinish materializes the object and publishes its metadata/notification. That is good commit semantics, but with S3-style multipart uploads the uncommitted parts are not readable as the eventual object, so the receiver necessarily waits.
I see two useful stages of improvement.
The smaller step is very close to what you proposed: store chunks as separate immutable Void objects and make the transfer itself a little event-sourced protocol:

Begin  { transfer_id, tensor descriptor, total bytes/chunks }
Chunk  { transfer_id, index, object_id, length, hash }
...
Commit { transfer_id, chunk count, aggregate hash }
Abort  { transfer_id, reason }


A receiver can fetch each chunk as soon as its Chunk event appears. It allocates/fills the destination buffer incrementally and verifies per-chunk hashes. The transfer is not authoritative until Commit; missing chunks, a hash mismatch, an explicit Abort, or a lease/deadline expiry fails the whole artifact. Chunks should be immutable and garbage-collected after aborted/expired transfers. If downstream execution starts speculatively, its externally visible result must not commit until the input transfer commits.
That preserves replay very naturally: replay consumes only committed transfer manifests and their immutable chunks. It also avoids depending on backend-specific multipart visibility.
The larger and probably better steady-state design is a direct QUIC tensor stream between producer and consumer, teed to Void concurrently:

producer ──QUIC stream──> consumer


The control plane sends a typed TransferTicket containing the contract/descriptor, transfer ID, source endpoint/authorization, expected size, and eventual durable object ID. QUIC already gives us ordered delivery, backpressure, cancellation, and multiplexing; tensor frames can add chunk indices and checksums. The consumer starts receiving immediately while the producer persists the same bytes. When replayability is required, successful computation can be speculative but the workflow step commits only after both the stream and durable write commit. For latency-sensitive ephemeral workloads, the operation contract/program could explicitly choose a weaker durability policy.
I would hide all of this behind something like ArtifactRef<T>/TensorHandle<T>, so a Flow does not care whether the bytes resolve from:

a live stream;
a chunked in-progress transfer;
a completed Void object during replay;
eventually, a node-local cache or shared-memory/device transport.

One caveat: allowing the receiver to receive/stage early is broadly useful, but allowing it to execute before the full tensor arrives is a separate capability. A normal TensorOp::forward(Tensor<S>) still needs the whole tensor. A tiled/sequence-aware op could opt into a StreamingTensorOp contract with explicit chunk axes/order and finalization semantics. I would not impose streaming execution on every operator.
Safetensors still fits this model: send its header first, validate/allocate from it, then stream its data ranges. For very large tensor bundles we can pipeline named tensors or tiles, while the operation contract remains authoritative.
2. Orchestration: separate Graph from Program/Strategy
I agree that the current BlackHole::Sun is too specific as the root abstraction. The graph descriptor currently compiles directly into a Sun whose Epoch means “generate paired inputs, run two propagation phases, apply policy, broadcast potentiation,” and Cell similarly bakes perturb-up / propagation / perturb-down / propagation / optimize into its lifecycle. That is a very good strategy implementation, but not a generic tensor graph runtime.
I would split this into three layers:


Topology/Graph — typed nodes, ports, edges, contracts, and placement requirements.

Execution primitives — deploy graph, start/stop nodes, submit typed inputs, run one dependency-aware forward pass, collect sinks, broadcast a typed command, checkpoint, etc.

Program/Strategy — an ordinary Jungle Flow composed from those primitives.

The dependency-aware scheduler already in Sun is valuable. I would extract its neutral core as something like ForwardPass<G>: seed ready roots, execute nodes when their typed inputs are ready, fan out outputs, and collect typed sinks. It should have no concept of “up,” “down,” “potentiation,” or gradient steps.
Then the current behavior becomes a library-provided program, conceptually:

type CurrentSun<G, Generator, Policy, const N: usize> =
Loop<TwoSidedZerothOrder<G, Generator, Policy, N>>;


Other programs can compose the same graph and primitives:

one-shot or serving inference;
batched/map-reduce evaluation;
forward-only supervised scoring;
actor/learner RL;
population/evolution strategies;
checkpoint/fine-tune/evaluate schedules;
a future backward program using explicit backward/adjoint operation contracts.

Capability bounds remain useful here. ForwardPass<G> only requires forward-capable nodes; TwoSidedZerothOrder<G> additionally requires perturb/optimize capabilities; a checkpointing program requires checkpoint capability. The compiler still rejects a strategy applied to a graph whose nodes cannot support it.
I would also make orchestration messages generic rather than expanding the current Transmission::{Propagation, Potentiation} enum indefinitely. The neutral data plane carries typed artifacts; program-specific control messages belong to the strategy Flow. Likewise, the generic observable node state should be operational (Queued/Running/Succeeded/Failed plus event sequence), with “Propagation1/Propagation2/Optimization” emitted as strategy-specific phase annotations.
For compatibility and the project's vocabulary, Sun can remain the name/alias for the current two-sided blackbox program. I just would not make BlackHole::Sun the only way a type-level graph becomes executable. The durable abstraction is closer to “compile this typed Graph, then run it under Program P.”
So my preferred direction is:

make Void persistence a commit/replay service rather than the synchronous byte path;
add stream-and-persist transfers, with chunked Void manifests as a simpler first implementation/fallback;
extract the existing ready-node propagation machinery into a phase-neutral forward-pass primitive;
retain today's Sun as a reusable two-sided zeroth-order strategy built from generic graph execution primitives.

That preserves the genuinely strong parts of the existing implementation without forcing either object-store latency or one optimization schedule onto every workload.

nick:
mitb, the whole <Topology as BlackHole>::Sun<_, _, ..> thing should probably stay the main entrypoint. I agree that we should expand to accommodate a wider array of workloads, but this should remain the call pattern regardless. Maybe we could put another generic on the Sun assoc type (or change the meaning of what's there currently)?


mitb:
[This is an automated reply from the Man in the Box]
Yes—I agree. I was arguing for changing what Sun means, not for replacing the <Topology as BlackHole>::Sun<...> entrypoint. That call pattern is a strong part of the type-level DSL and should stay the place where a topology becomes an executable Jungle Flow.
My preference would be to make the primary generic a Sun program/manifest, rather than adding several independent generics directly:

pub trait BlackHole {
type Sun<P: SunProgram>;
}

pub trait SunProgram {
type State;
type Driver;
}


Conceptually, the associated type becomes:

compile this Topology into a deployed graph, then drive it with program P

Usage could remain very recognizable:

type Training = <Topology as BlackHole>::Sun<
TwoSidedZo<Generator, Policy, 8>
>;

type Serving = <Topology as BlackHole>::Sun<
Serve<RequestSource>
>;

type Evaluation = <Topology as BlackHole>::Sun<
BatchedEval<Dataset, Scorer, 32>
>;


The current Manifest is already close to this role—it bundles Generator, Policy, and State—but today BlackHole only uses it to fill parameters in a fixed SunFlow/Epoch. I would promote that abstraction so the program supplies the actual driver/schedule, while the topology fold still supplies the node deployment and typed graph structure.
In other words, the recursive BlackHole implementation would still generate the topology-specific spawn/deploy portion, but its terminal case would append P::Driver rather than always appending today's:

generate pairs → P1/P2 propagation → policy → potentiation


The existing behavior then becomes the default/library program TwoSidedZo<...>, possibly with StatelessManifest<Generator, Policy> retained as a compatibility alias for it.
I lean toward putting strategy-specific constants such as accumulation steps inside the program type. The current const generic has a precise meaning for two-sided optimization, but there is no honest universal meaning for it in serving, actor/learner RL, or a one-shot forward pass. If we keep the second const parameter temporarily for source compatibility, I would avoid giving it a vague new meaning; an alias can translate:

type LegacySun<Topology, M, const N: usize> =
<Topology as BlackHole>::Sun<TwoSidedZoManifest<M, N>>;


There is one deeper change required underneath this: topology compilation currently assumes program-specific node seeds and mailboxes (CellInit, FusionSeed, P1/P2/PO endpoints). Those need to become neutral deployment/control structures, or be selected by the SunProgram, otherwise changing only the final Epoch generic would make the API look general while the generated nodes remain coupled to two-sided ZO.
So I would preserve the hierarchy exactly:

<Topology as BlackHole>::Sun<Program>
├─ compile/deploy typed topology


That feels better than introducing a separate top-level Run<Graph, Program> type. BlackHole::Sun remains the canonical compiler/application point; “Sun” simply becomes any executable realization of the topology, with today's optimization loop as one program among several.

nick:
mitb, agreed. OK, I'm feeling pretty good about this. Let's move forward.
