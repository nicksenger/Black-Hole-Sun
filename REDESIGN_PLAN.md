# Black Hole Sun Redesign Plan

This redesign will be delivered as a sequence of independently testable
migrations.

## Non-negotiable invariants

A Mass server is a black box identified by two things: the tensor contract it
hosts (including input/output shapes and encodings) and the capabilities it
advertises. There is no protocol-level distinction between a "model" and an
"operation". A Qwen forward pass, a tensor slice, and a chain of tensor
operations are different implementations of the same hosted-instance
abstraction.

Capabilities belong to the implementation. Forward, reset, perturb,
optimize, checkpoint, and fuse are advertised per hosted implementation and
validated fail-closed at start and invocation. Any backend may implement the
full zeroth-order lifecycle, and the backend owns all state required to do so.
Mass owns identity, routing, validation, and durable artifact publication.

This project has no compatibility requirement for the pre-redesign API or
wire protocol. Migration is complete only when the unified path expresses all
required behavior and the parallel legacy path has been deleted. Temporary
aliases, duplicate registries, and duplicate message dialects are not an
acceptable end state.

## 1. Establish the operation contract and tensor wire format

- Add wire-only types to `black-hole-type`: `ContractId`,
  `ContractDescriptor`, input and output port descriptors,
  static/symbolic/dynamic dimensions, dtype and layout constraints,
  `EncodingId`, and a versioned `TensorEnvelope`.
- Add a small shared typed-contract crate instead of making the wire crate
  depend on Candle or another tensor backend. It will depend on glowstick and
  define the compile-time side of the contract, along these lines:

  ```rust
  trait TensorContract {
      type Input: TensorSpec;
      type Output: TensorSpec;
      type Metadata;

      const ID: ContractId;
      const VERSION: u32;

      fn descriptor() -> ContractDescriptor;
  }
  ```

  The explicit ID, version, and descriptor are important: distributed
  identity must not be derived from Rust `type_name` or other
  compiler-specific details.
- Represent inputs and outputs as named tensor bundles from the beginning,
  even where a convenience API exposes one tensor. This prevents masks,
  positions, and multi-output operations from forcing a second envelope
  redesign.
- Standardize v1 dense payloads on safetensors. The Black Hole Sun envelope
  remains authoritative for the contract, metadata encoding, and codec
  version; the safetensors header describes the concrete tensor instance.
  Metadata remains separately encoded, using postcard initially.
- On decode, first validate the envelope and contract, then the safetensors
  names, dtype, rank, and concrete dimensions. Bind glowstick symbolic
  dimensions and verify every repeated binding before constructing a typed
  tensor. Unknown envelope versions and codecs fail closed.
- Add golden serialization and hash tests, plus negative tests for the wrong
  contract/version, dtype, rank, dimension, inconsistent symbolic bindings,
  malformed offsets, and unknown encodings. If glowstick needs a runtime
  descriptor or binding trait for `Dyn<Label>`, add that explicitly instead
  of treating its current numeric iterator as a complete runtime schema.

## 2. Introduce typed artifact references and split the capability traits

- Replace the erased `InferenceOutputId`/`EmissionId` path in
  `_black_hole_type/src/lib.rs` with zero-cost typed references such as
  `ObjectRef<T>` and then `ArtifactRef<T>`. `ArtifactRef` is the abstraction
  carried by a Flow. Initially it can resolve only to a committed Void object;
  later it can also resolve to an in-progress transfer or live stream.
- Generalize `Emission<M>` so it carries a typed output reference independently
  from metadata, for example `Emission<T, M>`. Metadata does not become part
  of the tensor shape.
- Split `VoidInferOps` in `_black_hole_flux/src/ops.rs` into:

  - `VoidOps` for raw and typed artifact persistence and waiting;
  - `MassOps<Op>` for start, forward, and shutdown;
  - optional `ResetOps`, `PerturbOps`, `OptimizeOps`, `CheckpointOps`, and
    `FuseOps` capabilities;
  - a Qwen/LLM adapter trait for darkening, decoding, and dark-token
    conversions.

- Make `MassClient<Op>` typed, deriving its request and response artifact
  types from `Op`. Update all callers in the workspace in the same migration;
  do not retain aliases or adapters for the replaced Mass protocol.
- Define a first `QwenDarkInference` contract and adapter that maps the current
  DarkToken/postcard behavior onto the generic API. This is the compatibility
  backend and proves that the abstraction preserves existing behavior.

## 3. Unify Mass around hosted operations

- Refactor all local execution in `_black_hole_mass/src/lib.rs` behind an
  injected operation implementation. The server owns instance identity,
  routing, contract validation, capability gating, and durable publication.
  Implementations own operation state and execution. The Qwen adapter owns
  `paramecia_engine::ModelInput`, DarkToken conversion, tokenizer/model
  configuration, the paramecia engine, and QuZO-specific state. This is an
  engine boundary, not merely a data-conversion adapter.
- Replace the two instance registries and protocol dialects with one hosted
  instance registry and one versioned protocol: start, invoke, optional
  lifecycle verbs, and shutdown. Start carries the complete contract
  descriptor/hash, codec set, required capabilities, and opaque
  implementation configuration. Every subsequent verb routes by the same
  instance ID. Checkpoint and fuse inputs/results are durable Void objects
  addressed by `ObjectId`.
- Replace architecture-only placement with advertised operation contracts,
  encodings, and lifecycle capabilities. Backend-specific configuration may
  still contain an architecture requirement, but selection must first match
  the complete hosted-operation capability. Routes pin every later operation
  for an instance to its selected worker.
- Preserve lifecycle operations as separate capability bounds. A forward-only
  server is not required to implement perturb, optimize, or checkpoint, while
  the Qwen QuZO adapter implements the full set.
- Port routing tests to use a small deterministic fake tensor operation first,
  then retain the Qwen regression tests. Add tunnel tests for:

  - rejecting a same-shape, different-contract worker;
  - contract-version mismatch;
  - unsupported codecs;
  - worker pinning; and
  - validation of a malformed actual payload.
- Add a fake backend implementing perturb/optimize over a tiny parameter
  buffer. An end-to-end `TwoSidedZo` run must measurably reduce a quadratic
  loss in ordinary CI. Add tensor-slicer and two-operation-chain tests so the
  abstraction is exercised beyond a single identity-shaped fake.
- Make `paramecia-engine` optional behind the Qwen backend feature. A Mass
  binary hosting only an injected tensor operation must not link Candle or
  paramecia.

## 4. Carry operation types through Flux and make topology edges checkable

- Parameterize Atom, Cell/node descriptors, emissions, transmissions, and
  scheduler payloads by the operation or artifact type. The current
  `Unary<P, A, E>` and `Binary<P1, P2, A, E>` descriptors retain only numeric
  downstream ports, so the descriptor and edge representation must also
  retain the destination input contract.
- Add a type-level compatibility bound for every edge: the source contract's
  output bundle must equal the destination contract's input bundle, including
  dtype, layout, and semantic contract identity rather than only rank and
  dimensions. The runtime graph finalizer must retain and revalidate the
  erased descriptors for separate binaries and rolling deployments.
- Replace generic data-plane use of
  `Transmission::{Propagation, Potentiation}` with typed artifact delivery and
  neutral operational control. Keep potentiation in the `TwoSidedZo` program
  instead of growing the shared transport enum with every future strategy.
- Add compile-pass and compile-fail coverage with `trybuild` for:

  - a matching static edge;
  - matching shared symbolic dimensions;
  - a shape mismatch;
  - a dtype mismatch; and
  - a same-shape, different-contract mismatch.

  Existing red-dwarf, white-dwarf, dark-star, and diamond-dog topology tests
  continue to prove fan-out, binary ports, and warp boundaries.

## 5. Generalize Sun while preserving the canonical entrypoint

`<Topology as BlackHole>::Sun<Program>` remains the canonical application
point.

- First extract the dependency-aware portion of the current `EpochWithState`
  (`PreparePropagationPipeline`, `SendReadyRootTasks`, and
  `ProcessReadyPipelineNode`) into a neutral `ForwardPass` execution primitive
  with no up/down/potentiation vocabulary.
- Promote the current `Manifest` into a real `SunProgram`. It supplies state,
  program-selected deployment/control requirements, and the driver Flow. The
  recursive `BlackHole` fold still compiles and spawns the topology; its
  terminal case attaches the program driver instead of always attaching
  `SunFlow<Generator, Policy, ...>`.
- Rebuild today's behavior as
  `TwoSidedZo<Generator, Policy, const N: usize>`. Accumulation steps belong
  inside this program because they have no universal meaning. Provide a
  source-compatibility alias while downstream code migrates:

  ```rust
  type LegacySun<T, M, const N: usize> =
      <T as BlackHole>::Sun<TwoSidedZoManifest<M, N>>;
  ```

- Add a minimal forward-only/serving program as the second implementation.
  This is an important acceptance test: `ForwardPass` requires only forward
  capability, while `TwoSidedZo` must fail to compile unless every
  participating node supplies perturb and optimize capabilities.
- Neutralize generic node state to `Queued`, `Running`, `Succeeded`, or
  `Failed`, plus a sequence number and optional program phase annotations.
  Update Beam to render the annotations so the existing propagation and
  optimization visualization remains available without baking those phases
  into the runtime core.
- Verify behavioral equivalence of the old training schedule, including
  accumulation, fusion, warp, retries, and observation. Add a serving test
  that runs the same compiled topology without starting the QuZO lifecycle.

## 6. Add progressive streaming without weakening replay semantics

- Build streaming on `ArtifactRef<T>` and the tensor envelope rather than
  directly into Flow types.
- First add the backend-independent Void transfer protocol discussed in
  `REDESIGN.md`: `Begin`, immutable `Chunk` objects/events, `Commit`, and
  `Abort`, with per-chunk hashes, aggregate hash, expected count and length,
  deadline/lease, and cleanup for aborted or expired transfers. A receiver can
  stage chunks immediately, but the artifact becomes authoritative only after
  `Commit`.
- Make replay resolve committed manifests and chunks only. If a consumer
  executes speculatively, prevent its externally visible output from
  committing until its input transfer commits.
- Then add direct QUIC tensor streams, teed to Void concurrently. A typed
  `TransferTicket` identifies the contract, descriptor, transfer ID,
  source/authorization, expected size and hash, durability policy, and
  eventual Void ID. Send the safetensors header first so the receiver can
  validate and allocate before data frames arrive.
- Keep full-tensor `TensorOp::forward` as the default. Introduce a separate
  `StreamingTensorOp` only for contracts that explicitly define chunk axis,
  order, and finalization semantics.
- Test missing, corrupt, and duplicate chunks; abort and lease expiry;
  receiver cancellation and backpressure; stream interruption with Void
  fallback; delayed commit; replay; and the invariant that a result cannot
  commit before its durable input.

## Delivery order and acceptance gates

Deliver the phases above as separate merge requests in order, with the Qwen
compatibility contract introduced in the first or second merge request. Every
merge request must leave `cargo test --workspace` green and keep the current
public facade in `black-hole-sun/src/lib.rs` usable.

Delete the legacy protocol and types as part of this redesign. The deletion
gate is that all of the following pass through the unified protocol:

- the generic fake-operation path;
- the Qwen adapter;
- generic tunnel routing;
- `TwoSidedZo` behavioral parity, including convergence against a non-Qwen
  optimizing backend; and
- a forward-only Sun program.

Direct QUIC streaming is deliberately last. The contract/envelope, typed
references, generic Mass path, and neutral scheduler give it stable interfaces,
while the chunked-Void implementation serves as both the first latency
improvement and the fallback.

The first implementation slice is therefore:

1. contract and envelope types;
2. the safetensors codec;
3. typed `ObjectRef` and `Emission`; and
4. the Qwen compatibility adapter.

This establishes the foundation required by every later slice without
changing orchestration behavior, and keeps the initial review scope
manageable.
