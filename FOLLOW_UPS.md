# Follow-ups: strategy/backend neutrality and the black-box mass server

Recorded after reviewing the `black-hole-probe` test suite against the generic
tensor-operation refactor (REDESIGN_PLAN stages 1–3). The review surfaced that
the design doc under-expressed the project's core intent, and that the current
implementation has concrete gaps against it — including one structural flaw in
the mass server itself.

## Overarching intent (restated)

**Every Sun strategy must operate over mass servers hosting whatever input/
output tensor shapes are contracted — including tiny deterministic fakes.**
That is the whole point of the project.

- `TwoSidedZo` (`_black_hole_flux/src/programs/two_sided_zo/mod.rs`),
  `ForwardOnly` (`_black_hole_flux/src/programs/forward_only/mod.rs`), and
  `CheckpointEvaluate`
  (`_black_hole_flux/src/programs/checkpoint_evaluate/mod.rs`) are *strategies*,
  not backends. Any one of them must compile and run against a mass server
  hosting an operation for any `TensorContract`, from a 4×4 f32 matrix to a
  27B GGUF.
- QuZO is a **weight-space strategy**, not a Qwen feature: perturb weights ±,
  evaluate twice, apply an update from two caller-supplied scalar losses.
  Nothing in that loop requires an LLM, a tokenizer, or even differentiability —
  it is zeroth-order optimization over a parameter buffer. The tests already
  pass literal fake losses (`0.5` / `1.0`) to the real path; the strategy is
  loss-agnostic by construction.
- The capability bounds in the design (`PerturbOps<Op>`, `OptimizeOps<Op>`,
  `CheckpointOps<Op>`, `FuseOps<Op>` in `_black_hole_flux/src/ops.rs`) are
  **backend-neutral by intent**: any operation may supply them, and a strategy
  that requires them must fail to compile against nodes that don't —
  regardless of which backend the node hosts.

### The mass server is a black box

The mass server is a goddamn black box. It is defined by exactly two things:

1. **Capabilities** — what it can do: `_FORWARD_`, `_QUZO_PERTURB_` (up/down),
   `_OPTIMIZE_`, `_RESET_`, `_CHECKPOINT_`, `_FUSE_`.
2. **Shapes** — what shape the input tensor is, and what shape the output
   tensor is.

That is the entire identity of a mass server. There is no third axis along
which "a model" differs from "an arbitrary tensor operation", or from "a string
of a few arbitrary tensor operations". A mass server that slices a tensor and a
mass server that performs an entire Qwen forward pass are still just Mass
servers. The framework has **no concept of a model instance distinct from an
operation instance**; any code, protocol message, registry, or client API that
treats them as different species is a design error, not a migration detail.

### There are no delicate users

There exists no precious and delicate user of Black-Hole-Sun for which the API
must be preserved. This migration **will** break things, and that is expected —
breaking changes are not a side effect to be minimized, they are the point. We
should not clutter the project with legacy relics on behalf of some gentle user
which doesn't exist: no "temporary aliases", no "alongside during migration"
dialects, no compatibility shims kept alive out of politeness.

The bar for the replacement is capability, not continuity: the new protocol
must be general enough to cover everything the old one did. If making it that
general pushes work onto users of the framework — fine. Users get updated
accordingly; the project does not accrete fossils for them.

### Where the original design went wrong

REDESIGN_PLAN.md stage 3 says:

> Preserve lifecycle operations as separate capability bounds. A forward-only
> server is not required to implement perturb, optimize, or checkpoint, while
> the Qwen QuZO adapter implements the full set.

and stage 5 says:

> `ForwardPass` requires only forward capability, while `TwoSidedZo` must fail
> to compile unless every participating node supplies perturb and optimize
> capabilities.

Both sentences were **misinterpreted** as "QuZO lifecycle stays Qwen-owned; the
generic path is forward-only, full stop." That reading makes the capability
bounds vacuous for non-Qwen backends and contradicts the intent above: the
bounds exist precisely so that *any* operation can opt in to QuZO.

The plan's own framing compounded the error. Stage 3 is titled "Make Mass an
injected operation host" and speaks of "the Qwen adapter" as though the server
were a generic substrate *plus* a Qwen path — two species coexisting — rather
than one black-box abstraction parameterized by capabilities and shapes. The
plan never names the unification; it only ever describes adding a second path
alongside the first. And its acceptance gate "the Qwen adapter" was read
narrowly (a data conversion), when the stage-3 text — "The Qwen adapter owns
... QuZO-specific state" — describes an engine boundary that has not been
drawn.

Finally, the plan's incremental-migration courtesy is exactly the relic instinct
the no-delicate-users policy rejects. Stage 2 says: "Keep temporary aliases or
adapters for `InferenceRequest`, `InferenceOutput`, and `VoidInferOps` so the
existing probes can migrate incrementally instead of requiring a flag-day
patch." That language is superseded by the decisions below.

## Shortcomings of the current approach

1. **Two species of instance.** The server keeps two disjoint registries:
   `instances: HashMap<Uuid, ModelSlot>` holding paramecia engines
   (`_black_hole_mass/src/lib.rs:1735`) and
   `operation_instances: HashSet<Uuid>` plus one shared
   `Arc<dyn OperationImplementation>` (lib.rs:1737). Separate start paths
   (`handle_start` loads a GGUF; `handle_operation_start_local` validates a
   contract), separate routing — `route_for_model` (lib.rs:3071) only consults
   routes populated for model instances, so every QuZO verb sent for an
   operation instance id fails with "not running". Under the black-box
   principle this split does not exist: one registry of instances, each
   declared at start time by (contract descriptor, capability set), routed by
   instance id regardless of kind.

2. **Two protocol dialects.** The wire carries two languages for what should be
   one protocol: legacy `MassIn::{Start, Infer, PerturbUp, PerturbDown, Reset,
   Checkpoint, Optimize, FuseWeights}` (postcard `InferenceRequest` /
   `InferenceOutput`, addressed by model id) and generic
   `MassIn::{StartOperation, ForwardOperation, ShutdownOperation}` (tensor
   envelope, capability-declared). The unified protocol is: *start* (declare
   contract + capabilities) → *invoke* (forward a tensor artifact) →
   capability-gated lifecycle verbs → *shutdown*. The legacy dialect persists
   because the Qwen path predates the abstraction and was kept "alongside" it
   instead of being folded in as one black-box implementation.

3. **Capabilities are not first-class per-instance declarations.**
   `OperationCapability` already has the right shape (contract descriptor +
   encodings) but only the generic start path declares it, and it carries no
   lifecycle flags. A Qwen instance gets the full QuZO set *by construction*
   (because it is a paramecia engine); an operation instance gets none. The
   server should accept a capability declaration at start time for every
   instance and gate every verb on it, fail-closed — which is exactly what the
   black-box principle requires.

4. **No lifecycle surface on injected operations.**
   `OperationImplementation` (lib.rs:109) has only `start` / `forward` /
   `shutdown`. Consequence of #1/#3: a generic operation cannot participate in
   `TwoSidedZo` at all, and flux's `OperationPrimordium<Op>`
   (`_black_hole_flux/src/nodes/cell/mod.rs`) compiles against any contract but
   can never *run* against a non-Qwen backend — the perturb/optimize effects it
   emits have no dispatch target.

5. **No engine boundary; paramecia is the instance type.**
   `MassInstance.engine` is paramecia's concrete `ModelEngine`
   (lib.rs:1655-1661), constructed inline via
   `paramecia_engine::ModelEngineBuilder::new(model_path)` (lib.rs:3951). The
   existing `QwenOperationAdapter` (lib.rs:1549) is a *data* adapter only — it
   converts `DarkToken` / `InferenceInput` / `InferenceOutput` to paramecia
   types. It does not own the engine or the QuZO state, contrary to stage 3's
   wording. "The generic server owns instance concerns" is currently true for
   operation instances (bare Uuids + a shared trait object) and false for model
   instances, which are paramecia engines by construction.

6. **The QuZO update rule lives in paramecia.**
   `engine.perturb_up(seed)` / `engine.update(loss_up, loss_down)`
   (lib.rs:4462, 4761) delegate to `paramecia-opt/src/qzo/mod.rs`, where the
   zeroth-order state (`DecomposedZOState`), residual/error-feedback handling,
   and clipping operate on GGUF/Candle weights. bhs contains no backend-neutral
   implementation of the update rule; it forwards. (The *scheduling* half — the
   `MassState` machine, frozen flags, oscillation, tunnel routing — is already
   neutral and lives in bhs; only the weight mutation is delegated.)

7. **The client API leaks the split.** `MassClient<Op = QwenDarkInference>`
   carries both legacy methods (`start` / `infer` / `perturb_up` / ...) and
   typed operation methods (`start_operation` / `forward` /
   `shutdown_operation`) on the same struct, and `ResetOps<Op>` /
   `PerturbOps<Op>` / `OptimizeOps<Op>` / `CheckpointOps<Op>` / `FuseOps<Op>`
   are implemented **only for `QwenDarkInference`**
   (`black-hole-sun/src/mass_client.rs:343-393`), forwarding to the legacy
   dialect. One client, one message set, capability-checked server-side is what
   the black-box principle implies.

8. **Hard paramecia dependency.** `paramecia-engine` is a non-optional git
   dependency of `_black_hole_mass` (Cargo.toml:23). Every mass server —
   including a forward-only server hosting only injected fakes — links
   paramecia and candle. The `qwen35_*` features select which architecture
   compiles in; none make the engine itself optional.

9. **Test-suite consequences (black-hole-probe).** Roughly ten tests
   early-return unless `BLACK_HOLE_PROBE_MODEL_PATH` points at a real GGUF, and
   they skip *silently* (a warning log) in no-model CI: `cell` (tests/cell.rs),
   `inference`, `dark_inference`,
   `start_model_applies_instance_default_inference_limit_override`,
   `optimization`, `dark_optimization`, `fuse_weights_with_checkpoint`, and
   `tcp_tunnel_root_forwards_model_load_and_inference_to_registered_worker`
   (tests/mass.rs), plus the red_dwarf/white_dwarf sun tests. QuZO
   state-machine and scheduling coverage can therefore only run against a real
   model, and **convergence is untestable**: with a fake tensor operation we
   could assert that QuZO measurably decreases a quadratic loss over epochs —
   something impossible in CI against a 0.8B model.

10. **Documentation gap.** REDESIGN_PLAN.md never states the black-box
    principle, never names the unification of the two instance species or the
    two protocol dialects, never says paramecia should become optional, and its
    stage-5 capability language is ambiguous about whether non-Qwen
    implementations must exist. The misreading in the previous section is a
    direct consequence; the plan needs an explicit correction (see below).

## Remediation

Ordered roughly by dependency; each item is independently testable.

1. **Unify the instance model.** One registry of instances, one start message
   carrying (contract descriptor, declared capability set), one invoke, and
   lifecycle verbs gated on declared capabilities — fail-closed when a verb is
   not advertised, like contract mismatches today. Collapse
   `operation_instances` into the same routing table as model instances so an
   instance id resolves regardless of kind. Qwen becomes *one black-box
   implementation* inside this model, not a parallel species.

2. **Make capabilities first-class.** Extend `OperationCapability` (or its
   successor) with lifecycle flags — forward, perturb_up/down, optimize, reset,
   checkpoint, fuse — so every instance, Qwen or fake, declares the same way at
   start time. Worker capability advertising (`WorkerCapabilities`) and tunnel
   routing should key off these flags instead of architecture lists where the
   contract already determines compatibility.

3. **Lifecycle methods on the operation trait.** Add `perturb_up(instance_id,
   seed)`, `perturb_down(instance_id)`, `optimize(instance_id, loss_up,
   loss_down)`, `reset(instance_id)`, and checkpoint/fuse as dedicated void
   objects with replay semantics (per Decisions §4): `checkpoint(instance_id)`
   produces a first-class void object and returns its `ObjectId`;
   `fuse(instance_id, checkpoint_id, contribution)` consumes the referenced
   checkpoint and returns the `ObjectId` of the fused result. Protocol messages
   carry IDs, not inline byte blobs. Per Decisions §2, zeroth-order delta state
   is the implementor's problem: the
   backend that advertises zeroth-order support owns whatever state its update
   rule needs, keyed however it likes (e.g. per `instance_id` behind a mutex,
   as `DeterministicFakeOperation` already does for its instance set). The Mass
   server only requires that the trait is implemented in order to accept the
   advertisement.

4. **One client message set.** Implement `PerturbOps<Op>` / `OptimizeOps<Op>` /
   `CheckpointOps<Op>` / `FuseOps<Op>` / `ResetOps<Op>` on `MassClient<Op>` for
   all contracts against the unified protocol, and delete the legacy method
   surface on `MassClient` — no aliases. Update every caller in the workspace.

5. **Engine boundary; optional paramecia.** Introduce a model-engine trait
   (`load / predict / perturb_up / perturb_down / update / reset / checkpoint /
   fuse`) with a `QwenEngine` implementation wrapping paramecia, so an instance
   holds the trait rather than the concrete engine. Then make `paramecia-engine`
   optional behind a feature so forward-only deployments do not link it. This
   completes the "Qwen adapter" that stage 3 actually describes — and makes the
   Qwen server what it has always been supposed to be: a black box that
   happens to contain an LLM.

6. **Delete the legacy protocol.** Once the unified start/invoke/lifecycle
   protocol is in place, remove `MassIn::{Start, Infer, PerturbUp, PerturbDown,
   Reset, Checkpoint, Optimize, FuseWeights}` and their `MassOut` /
   `TunnelRequest` counterparts — no sunset period, no compatibility aliases.
   The precondition is coverage, per the no-delicate-users policy: the unified
   protocol must first be proven general enough to express everything the old
   one did (text/token/dark-token inputs, per-instance sampling configuration
   such as inference limits, and the full QuZO lifecycle). Where expressing
   that pushes work onto framework users, that is acceptable and expected. The
   Qwen backend keeps working because it is now one black-box implementation
   behind the unified protocol; its token-level types move inside that
   implementation (or become a contract's metadata), not into the wire.

7. **Test migration (black-hole-probe).** Promote `DeterministicFakeContract` /
   `DeterministicFakeOperation` from tests/mass.rs into shared test support
   (with a call counter for execution-locality assertions). Then: rewrite the
   tunnel-inference and `cell` tests over injected fakes; fold `inference`'s
   batch case into the generic path; keep the genuinely GGUF-semantic tests
   (`fuse_weights_with_checkpoint`, dark-token smoke tests) model-gated but
   *loudly* (doc line + `#[ignore]` where appropriate). Add a fake "tensor
   slicer" operation and a two-operation chain as first-class test citizens —
   proof that slicing and chaining are just Mass servers, per the principle.
   Finally add the convergence test: a small fake operation with a quadratic
   loss whose QuZO loop must decrease loss over epochs.

8. **Correct REDESIGN_PLAN.md.** State the black-box principle explicitly (a
   mass server is defined by capabilities and shapes; there is no model/
   operation distinction), state the no-delicate-users policy, name the
   unification of instance registries and protocol dialects as an explicit
   stage with deletion (not coexistence) as the outcome, amend stage 3 to name
   the engine boundary as part of the Qwen adapter, and update the acceptance
   gates so "TwoSidedZo behavioral parity" includes a run against a non-Qwen
   operation backend.

## Decisions

1. **Legacy protocol: delete, don't sunset.** No legacy protocol is preserved.
   The unified protocol must be general enough to cover everything the old one
   did; where that pushes work onto users of the framework, that is fine. All
   users get updated accordingly. This supersedes REDESIGN_PLAN's "alongside
   during migration" and "temporary aliases" language throughout.

2. **Zeroth-order state: the implementor's problem.** Where zeroth-order delta
   state lives is the responsibility of whoever implements a Mass-server
   backend that advertises zeroth-order optimization support. The Mass server
   itself doesn't give a rat's ass — it just requires that something implements
   the trait in order to accept the capability advertisement. No server-side
   delta bookkeeping, no matched-pair validation by default.

3. **Optional capabilities: the implementor's choice.** How optional
   capabilities are expressed (trait shape, default methods, supertraits) is up
   to the implementor, as long as it lines up with the core goals of the
   framework: one black box, declared capabilities, fail-closed gating.

4. **Checkpoint/fuse payloads: dedicated void objects with replay semantics.**
   Checkpoints and fuse contributions/results are first-class void-store
   objects addressed by `ObjectId`, not opaque inline byte buffers. Messages
   reference IDs; the payload lives in void. That gives replay semantics — a
   checkpoint can be re-fetched, routed to another instance for fusion,
   logged, and verified independently of the call that produced it — and aligns
   the generic path with what the legacy Qwen client API already does
   (`checkpoint()` returns the void ID of current weights; `fuse` returns the
   void ID of the fused result, `_black_hole_flux/src/ops.rs:375-380`).
