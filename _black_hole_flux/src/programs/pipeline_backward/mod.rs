//! First-in-first-out pipeline-parallel reverse-mode training.
//!
//! Each node owns its forward graphs and optimizer state. The Sun transports
//! only activations and boundary gradients, warms each stage by its pipeline
//! depth, then gives ready backward work priority to produce a 1F1B schedule.

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_spec::{BackwardContract, TensorContract};
use black_hole_type::{ArtifactDelivery, Emission, ObjectId, ObjectRef, OperationalControl};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use uuid::Uuid;

use crate::compile::SunProgram;
use crate::nodes::cell::action::{CellState, Init};
use crate::nodes::fusion::action::FusionSeed;
use crate::ops::{BackwardOps, CheckpointOps, MassOps, StepOps, VoidOps};
use crate::topology::{BoundaryInit, SunAppearance, SunStateView, SunTopology, SunTopologyState};
use crate::AtomError;

/// Strategy-owned command envelope. Micro-batch identity stays on the
/// control plane instead of changing tensor payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineCommand {
    Forward {
        micro_batch: usize,
        emission_id: ObjectId,
        reply: ObjectId,
    },
    Backward {
        micro_batch: usize,
        gradient_emission_id: ObjectId,
        seed_from_forward: bool,
        reply: ObjectId,
    },
    Step {
        reply: ObjectId,
        step: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineResponse {
    Artifact {
        micro_batch: usize,
        emission_id: ObjectId,
    },
    Stepped,
    Failed {
        micro_batch: Option<usize>,
        message: String,
    },
}

#[derive(Clone, Debug)]
pub struct PipelineBackwardState<S = ()> {
    pub topology: Arc<Mutex<SunTopology>>,
    pub inner: S,
    /// Current command inbox for each operation input port.
    pub inboxes: HashMap<u32, ObjectId>,
    pub micro_batches: Vec<ArtifactDelivery<()>>,
    pub step: usize,
}

impl<S: Default> Default for PipelineBackwardState<S> {
    fn default() -> Self {
        Self {
            topology: Arc::new(Mutex::new(SunTopology::default())),
            inner: S::default(),
            inboxes: HashMap::new(),
            micro_batches: Vec::new(),
            step: 0,
        }
    }
}

impl<S> SunTopologyState for PipelineBackwardState<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

impl<S> PipelineBackwardState<S> {
    /// Build a deterministic, serializable view of the resolved graph.
    pub fn appearance(&self) -> SunAppearance {
        self.topology.lock().unwrap().appearance()
    }
}

impl<S> SunStateView for PipelineBackwardState<S> {
    fn sun_appearance(&self) -> SunAppearance {
        self.appearance()
    }
}

pub struct NeedsMicroBatches<S, const M: usize>(PhantomData<fn() -> S>);
impl<S, const M: usize> Predicate<(&PipelineBackwardState<S>, &())> for NeedsMicroBatches<S, M> {
    fn eval((state, _): &(&PipelineBackwardState<S>, &())) -> bool {
        state.micro_batches.len() < M.max(1)
    }
}

/// Collects one local set of micro-batches for each of two pipeline replicas.
pub struct NeedsDataParallelMicroBatches<S, const M: usize>(PhantomData<fn() -> S>);
impl<S, const M: usize> Predicate<(&PipelineBackwardState<S>, &())>
    for NeedsDataParallelMicroBatches<S, M>
{
    fn eval((state, _): &(&PipelineBackwardState<S>, &())) -> bool {
        state.micro_batches.len() < 2 * M.max(1)
    }
}

pub struct BeginStep<S>(PhantomData<fn() -> S>);
#[jungle::action]
impl<S> Action for BeginStep<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();
    fn emit(_state: &PipelineBackwardState<S>, _input: ()) {}
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("begin pipeline step failed".into()))?;
        state.micro_batches.clear();
        Ok(())
    }
}

/// Carrying variant used by the driver to retain the source delivery.
pub struct StoreMicroBatch<Input, S>(PhantomData<fn() -> (Input, S)>);
#[jungle::action(carry = ArtifactDelivery<Input>)]
impl<Input: Send + 'static, S> Action for StoreMicroBatch<Input, S> {
    type Effect = NoEffect;
    type Input = ArtifactDelivery<Input>;
    type Output = ();
    fn emit(_state: &PipelineBackwardState<S>, input: Self::Input) -> ((), Self::Input) {
        ((), input)
    }
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
        delivery: Self::Input,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("store pipeline micro-batch failed".into()))?;
        state.micro_batches.push(ArtifactDelivery {
            emission_id: ObjectRef::new(delivery.emission_id.id()),
            recv: delivery.recv,
            send: delivery.send,
        });
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStepResult {
    pub step: usize,
    pub outputs: Vec<ArtifactDelivery<()>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RunPipelineInput {
    nodes: Vec<u32>,
    /// Input ports for each pipeline stage, ordered by data-parallel replica.
    ports: Vec<Vec<u32>>,
    replicas: usize,
    linear: bool,
    inboxes: HashMap<u32, ObjectId>,
    micro_batches: Vec<ArtifactDelivery<()>>,
    step: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RunPipelineOutput {
    inboxes: HashMap<u32, ObjectId>,
    outputs: Vec<ObjectId>,
    step: usize,
}

pub struct RunPipelineEffect;

#[derive(Clone, Copy)]
enum WorkKind {
    Forward,
    Backward,
}

#[derive(Clone, Copy)]
struct PendingReply {
    kind: WorkKind,
    micro_batch: usize,
    reply: ObjectId,
}

async fn send_command<J: VoidOps>(
    jungle: &J,
    inboxes: &mut HashMap<u32, ObjectId>,
    port: u32,
    command: impl FnOnce(ObjectId) -> PipelineCommand,
) -> Result<ObjectId, String> {
    let inbox = *inboxes
        .get(&port)
        .ok_or_else(|| format!("missing pipeline inbox for port {port}"))?;
    let next = Uuid::new_v4();
    let reply = Uuid::new_v4();
    let message = OperationalControl {
        control: command(reply),
        recv: next,
    };
    jungle
        .upload_to_void_with(
            inbox,
            postcard::to_allocvec(&message).map_err(|e| e.to_string())?,
        )
        .await?;
    inboxes.insert(port, next);
    Ok(reply)
}

async fn wait_response<J: VoidOps>(
    jungle: &J,
    id: ObjectId,
) -> Result<Option<PipelineResponse>, String> {
    jungle
        .download_raw_wait(id, 1)
        .await?
        .map(|bytes| postcard::from_bytes(&bytes).map_err(|e| e.to_string()))
        .transpose()
}

#[jungle::effect(id = 85)]
impl<J: VoidOps> Effect<J> for RunPipelineEffect {
    type In = RunPipelineInput;
    type Out = RunPipelineOutput;
    type Err = String;

    fn effect(
        jungle: &J,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let nodes = input.nodes;
            if nodes.is_empty() {
                return Err("pipeline graph has no stages".into());
            }
            if !input.linear {
                return Err("PipelineBackward currently requires one linear unary chain".into());
            }
            let ports = input.ports;
            let replicas = input.replicas.max(1);
            if input.micro_batches.len() % replicas != 0 {
                return Err(format!(
                    "{} micro-batches cannot be split across {replicas} pipeline replicas",
                    input.micro_batches.len()
                ));
            }
            let m = input.micro_batches.len() / replicas;
            let p = nodes.len();
            let mut inboxes = input.inboxes;
            let mut forward = vec![vec![vec![None::<ObjectId>; p]; m]; replicas];
            let mut backward = vec![vec![vec![None::<ObjectId>; p]; m]; replicas];
            let mut next_fwd = vec![vec![0usize; p]; replicas];
            let mut next_bwd = vec![vec![0usize; p]; replicas];
            let mut busy = vec![vec![None::<PendingReply>; p]; replicas];

            while next_bwd.iter().flatten().any(|&n| n < m)
                || busy.iter().flatten().any(Option::is_some)
            {
                for s in 0..p {
                    for replica in 0..replicas {
                        if busy[replica][s].is_some() {
                            continue;
                        }
                        let b = next_bwd[replica][s];
                        let warmup = (p - s).min(m);
                        let backward_ready = b < m
                            && next_fwd[replica][s] >= warmup
                            && forward[replica][b][s].is_some()
                            && (s + 1 == p || backward[replica][b][s + 1].is_some());
                        if backward_ready {
                            let gradient = if s + 1 == p {
                                forward[replica][b][s].unwrap()
                            } else {
                                backward[replica][b][s + 1].unwrap()
                            };
                            let reply =
                                send_command(jungle, &mut inboxes, ports[s][replica], |reply| {
                                    PipelineCommand::Backward {
                                        micro_batch: b,
                                        gradient_emission_id: gradient,
                                        seed_from_forward: s + 1 == p,
                                        reply,
                                    }
                                })
                                .await?;
                            busy[replica][s] = Some(PendingReply {
                                kind: WorkKind::Backward,
                                micro_batch: b,
                                reply,
                            });
                            next_bwd[replica][s] += 1;
                            continue;
                        }
                        let f = next_fwd[replica][s];
                        let forward_ready =
                            f < m && (s == 0 || forward[replica][f][s - 1].is_some());
                        if forward_ready {
                            let emission = if s == 0 {
                                input.micro_batches[replica * m + f].emission_id.id()
                            } else {
                                forward[replica][f][s - 1].unwrap()
                            };
                            let reply =
                                send_command(jungle, &mut inboxes, ports[s][replica], |reply| {
                                    PipelineCommand::Forward {
                                        micro_batch: f,
                                        emission_id: emission,
                                        reply,
                                    }
                                })
                                .await?;
                            busy[replica][s] = Some(PendingReply {
                                kind: WorkKind::Forward,
                                micro_batch: f,
                                reply,
                            });
                            next_fwd[replica][s] += 1;
                        }
                    }
                }

                let mut progressed = false;
                for s in 0..p {
                    for replica in 0..replicas {
                        let Some(pending) = busy[replica][s] else {
                            continue;
                        };
                        let Some(response) = wait_response(jungle, pending.reply).await? else {
                            continue;
                        };
                        let (micro_batch, emission_id) = match response {
                            PipelineResponse::Artifact {
                                micro_batch,
                                emission_id,
                            } => (micro_batch, emission_id),
                            PipelineResponse::Failed {
                                micro_batch,
                                message,
                            } => {
                                return Err(format!(
                                    "pipeline replica {replica}, stage {s} failed for micro-batch {micro_batch:?}: {message}"
                                ));
                            }
                            PipelineResponse::Stepped => {
                                return Err(
                                    "stage returned step acknowledgement for tensor work".into()
                                );
                            }
                        };
                        if micro_batch != pending.micro_batch {
                            return Err("stage returned the wrong micro-batch".into());
                        }
                        match pending.kind {
                            WorkKind::Forward => {
                                forward[replica][micro_batch][s] = Some(emission_id)
                            }
                            WorkKind::Backward => {
                                backward[replica][micro_batch][s] = Some(emission_id)
                            }
                        }
                        busy[replica][s] = None;
                        progressed = true;
                    }
                }
                if !progressed {
                    tokio::task::yield_now().await;
                }
            }

            let mut steps = Vec::with_capacity(p * replicas);
            for stage_ports in ports.iter().take(p) {
                for &port in stage_ports {
                    let reply =
                        send_command(jungle, &mut inboxes, port, |reply| PipelineCommand::Step {
                            reply,
                            step: input.step + 1,
                        })
                        .await?;
                    steps.push(reply);
                }
            }
            for reply in steps {
                loop {
                    if let Some(response) = wait_response(jungle, reply).await? {
                        match response {
                            PipelineResponse::Stepped => {}
                            PipelineResponse::Failed { message, .. } => {
                                return Err(format!("pipeline step failed: {message}"));
                            }
                            PipelineResponse::Artifact { .. } => {
                                return Err("stage returned artifact for step".into());
                            }
                        }
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            }
            Ok(RunPipelineOutput {
                inboxes,
                outputs: (0..replicas)
                    .flat_map(|replica| {
                        let forward = &forward;
                        (0..m).map(move |mb| {
                            forward[replica][mb][p - 1].expect("all forwards completed")
                        })
                    })
                    .collect(),
                step: input.step,
            })
        }
    }
}

pub struct RunPipeline<Output, S>(PhantomData<fn() -> (Output, S)>);
#[jungle::action]
impl<Output: Send + 'static, S> Action for RunPipeline<Output, S> {
    type Effect = RunPipelineEffect;
    type Input = ();
    type Output = PipelineStepResult;
    fn emit(state: &PipelineBackwardState<S>, _input: ()) -> RunPipelineInput {
        let topology = state.topology.lock().unwrap();
        let mut nodes = topology.journey_ids.keys().copied().collect::<Vec<_>>();
        nodes.sort_unstable();
        let ports = nodes
            .iter()
            .filter_map(|node| {
                topology
                    .vertex_ports
                    .get(node)
                    .and_then(|p| p.first())
                    .copied()
            })
            .map(|port| vec![port])
            .collect::<Vec<_>>();
        let linear = ports.len() == nodes.len()
            && nodes.windows(2).all(|pair| {
                let outgoing = topology.outgoing.get(&pair[0]).cloned().unwrap_or_default();
                outgoing.len() == 1 && outgoing[0].vertex_id == pair[1]
            });
        RunPipelineInput {
            nodes,
            ports,
            replicas: 1,
            linear,
            inboxes: state.inboxes.clone(),
            micro_batches: state.micro_batches.clone(),
            step: state.step,
        }
    }
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let output = output.map_err(|e| Failure::Message(format!("pipeline step failed: {e}")))?;
        state.inboxes = output.inboxes;
        state.step += 1;
        Ok(PipelineStepResult {
            step: output.step,
            outputs: output
                .outputs
                .into_iter()
                .map(|id| ArtifactDelivery {
                    emission_id: ObjectRef::new(id),
                    recv: ObjectId::nil(),
                    send: ObjectId::nil(),
                })
                .collect(),
        })
    }
}

/// Runs two 1F1B lanes through a chain of two-port Fusion-style stages.
pub struct RunDataParallelPipeline<Output, S>(PhantomData<fn() -> (Output, S)>);
#[jungle::action]
impl<Output: Send + 'static, S> Action for RunDataParallelPipeline<Output, S> {
    type Effect = RunPipelineEffect;
    type Input = ();
    type Output = PipelineStepResult;

    fn emit(state: &PipelineBackwardState<S>, _input: ()) -> RunPipelineInput {
        let topology = state.topology.lock().unwrap();
        let mut nodes = topology.journey_ids.keys().copied().collect::<Vec<_>>();
        nodes.sort_unstable();
        let ports = nodes
            .iter()
            .filter_map(|node| topology.vertex_ports.get(node).cloned())
            .collect::<Vec<_>>();
        let linear = ports.len() == nodes.len()
            && ports.iter().all(|ports| ports.len() == 2)
            && nodes.windows(2).all(|pair| {
                let mut outgoing = topology.outgoing.get(&pair[0]).cloned().unwrap_or_default();
                outgoing.sort_by_key(|target| target.port_id);
                outgoing.len() == 2
                    && outgoing.iter().all(|target| target.vertex_id == pair[1])
                    && outgoing
                        .iter()
                        .map(|target| target.port_id)
                        .eq(topology.vertex_ports[&pair[1]].iter().copied())
            });
        RunPipelineInput {
            nodes,
            ports,
            replicas: 2,
            linear,
            inboxes: state.inboxes.clone(),
            micro_batches: state.micro_batches.clone(),
            step: state.step,
        }
    }

    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let output = output.map_err(|error| {
            Failure::Message(format!("data-parallel pipeline step failed: {error}"))
        })?;
        state.inboxes = output.inboxes;
        state.step += 1;
        Ok(PipelineStepResult {
            step: output.step,
            outputs: output
                .outputs
                .into_iter()
                .map(|id| ArtifactDelivery {
                    emission_id: ObjectRef::new(id),
                    recv: ObjectId::nil(),
                    send: ObjectId::nil(),
                })
                .collect(),
        })
    }
}

#[derive(Flow)]
pub struct CollectOne<Source, Input: Send + 'static, S>(Source, Step<StoreMicroBatch<Input, S>>);

#[derive(Flow)]
pub struct PipelineStep<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    Policy,
    S,
    const M: usize,
>(
    Step<BeginStep<S>>,
    While<NeedsMicroBatches<S, M>, CollectOne<Source, Input, S>>,
    Step<RunPipeline<Output, S>>,
    Policy,
);

#[derive(Flow)]
pub struct DataParallelPipelineStep<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    Policy,
    S,
    const M: usize,
>(
    Step<BeginStep<S>>,
    While<NeedsDataParallelMicroBatches<S, M>, CollectOne<Source, Input, S>>,
    Step<RunDataParallelPipeline<Output, S>>,
    Policy,
);

#[derive(Flow)]
pub struct PipelineBackwardFlow<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    Policy,
    S,
    const M: usize,
>(
    Step<FinalizePipelineGraph<S>>,
    While<Always<PipelineBackwardState<S>, ()>, PipelineStep<Source, Input, Output, Policy, S, M>>,
);

#[derive(Flow)]
pub struct DataParallelPipelineBackwardFlow<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    Policy,
    S,
    const M: usize,
>(
    Step<FinalizePipelineGraph<S>>,
    While<
        Always<PipelineBackwardState<S>, ()>,
        DataParallelPipelineStep<Source, Input, Output, Policy, S, M>,
    >,
);

pub struct FinalizePipelineGraph<S>(PhantomData<fn() -> S>);
#[jungle::action]
impl<S> Action for FinalizePipelineGraph<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();
    fn emit(_state: &PipelineBackwardState<S>, _input: ()) {}
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("finalize pipeline graph failed".into()))?;
        crate::compile::action::resolve_neutral_topology(&mut state.topology.lock().unwrap())?;
        Ok(())
    }
}

/// Pipeline-parallel reverse-mode Sun program.
pub struct PipelineBackward<
    Source,
    InputOp,
    OutputOp = InputOp,
    Policy = (),
    S = (),
    const M: usize = 1,
>(
    PhantomData<(Source, InputOp, OutputOp, Policy)>,
    PhantomData<fn() -> S>,
);

impl<Source, InputOp, OutputOp, Policy, S, const M: usize> SunProgram
    for PipelineBackward<Source, InputOp, OutputOp, Policy, S, M>
where
    InputOp: BackwardContract,
    OutputOp: BackwardContract,
    InputOp::Input: Send + 'static,
    OutputOp::Output: Send + 'static,
{
    type State = PipelineBackwardState<S>;
    type Driver = PipelineBackwardFlow<Source, InputOp::Input, OutputOp::Output, Policy, S, M>;
    type UnarySeed = Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;
    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        Init {
            recv_id: inbox,
            grad_steps: M.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: M.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: M.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state.inboxes.extend(ports.iter().copied());
    }
}

/// Backward program for two data-parallel replicas of one pipeline.
///
/// Every topology vertex must be [`Binary`](crate::topology::Binary): its P1
/// and P2 ports are the corresponding stage in replicas 0 and 1. The node
/// joins matching commands and steps its shared stage only after both replicas
/// have completed all local micro-batches.
pub struct DataParallelPipelineBackward<
    Source,
    InputOp,
    OutputOp = InputOp,
    Policy = (),
    S = (),
    const M: usize = 1,
>(
    PhantomData<(Source, InputOp, OutputOp, Policy)>,
    PhantomData<fn() -> S>,
);

impl<Source, InputOp, OutputOp, Policy, S, const M: usize> SunProgram
    for DataParallelPipelineBackward<Source, InputOp, OutputOp, Policy, S, M>
where
    InputOp: BackwardContract,
    OutputOp: BackwardContract,
    InputOp::Input: Send + 'static,
    OutputOp::Output: Send + 'static,
{
    type State = PipelineBackwardState<S>;
    type Driver =
        DataParallelPipelineBackwardFlow<Source, InputOp::Input, OutputOp::Output, Policy, S, M>;
    type UnarySeed = Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        Init {
            recv_id: inbox,
            grad_steps: M.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: M.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: M.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state.inboxes.extend(ports.iter().copied());
    }
}

pub struct WaitForPipelineCommand<S>(PhantomData<fn() -> S>);
#[jungle::action]
impl<S> Action for WaitForPipelineCommand<S> {
    type Effect = crate::nodes::cell::effect::WaitForOperationalControlEffect<PipelineCommand>;
    type Input = ();
    type Output = PipelineCommand;
    fn emit(state: &CellState<S>, _input: ()) -> ObjectId {
        state.recv_id
    }
    fn absorb(
        state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let message = output
            .map_err(|e| Failure::Message(format!("wait for pipeline command failed: {e}")))?;
        state.recv_id = message.recv;
        Ok(message.control)
    }
}

pub struct ExecutePipelineCommand<M, Op>(PhantomData<fn() -> (M, Op)>);
#[jungle::effect(id = 86)]
impl<M, Op, J> Effect<J> for ExecutePipelineCommand<M, Op>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: BackwardContract<Metadata = M> + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
    Op::OutputGrad: Send,
    Op::InputGrad: Send,
    J: VoidOps + MassOps<Op> + BackwardOps<Op> + StepOps<Op> + CheckpointOps<Op>,
{
    type In = (ObjectId, PipelineCommand, usize, Option<std::path::PathBuf>);
    type Out = ();
    type Err = AtomError;
    fn effect(
        jungle: &J,
        (instance_id, command, checkpoint_steps, checkpoint_dir): Self::In,
    ) -> impl Future<Output = Result<(), AtomError>> + Send {
        async move {
            let (reply, micro_batch) = match &command {
                PipelineCommand::Forward {
                    micro_batch, reply, ..
                }
                | PipelineCommand::Backward {
                    micro_batch, reply, ..
                } => (*reply, Some(*micro_batch)),
                PipelineCommand::Step { reply, .. } => (*reply, None),
            };
            let response = async {
                Ok::<PipelineResponse, AtomError>(match command {
                    PipelineCommand::Forward {
                        micro_batch,
                        emission_id,
                        ..
                    } => {
                        let bytes = jungle
                            .download_raw(emission_id)
                            .await
                            .map_err(AtomError::Download)?;
                        let emission: Emission<M, Op::Input> = postcard::from_bytes(&bytes)?;
                        let output =
                            MassOps::<Op>::forward(jungle, instance_id, emission.output_id)
                                .await
                                .map_err(AtomError::Inference)?;
                        let emission = Emission::<M, Op::Output> {
                            metadata: emission.metadata,
                            output_id: output,
                        };
                        let id = jungle
                            .upload_to_void(postcard::to_allocvec(&emission)?)
                            .await
                            .map_err(AtomError::Upload)?;
                        PipelineResponse::Artifact {
                            micro_batch,
                            emission_id: id,
                        }
                    }
                    PipelineCommand::Backward {
                        micro_batch,
                        gradient_emission_id,
                        seed_from_forward,
                        ..
                    } => {
                        let bytes = jungle
                            .download_raw(gradient_emission_id)
                            .await
                            .map_err(AtomError::Download)?;
                        let emission: Emission<M, Op::OutputGrad> = postcard::from_bytes(&bytes)?;
                        let gradient = if seed_from_forward {
                            // The sink seeds reverse mode from its forward output. Reframe
                            // that tensor under the reverse descriptor before Mass validates it.
                            let bytes = jungle
                                .receive_artifact_raw(&emission.output_id)
                                .await
                                .map_err(AtomError::Download)?;
                            let output =
                                black_hole_spec::decode_output::<Op>(&bytes).map_err(|e| {
                                    AtomError::Inference(format!("invalid sink gradient seed: {e}"))
                                })?;
                            let bytes = black_hole_spec::encode_output_gradient::<Op>(
                                &output.tensors,
                                &output.metadata,
                            )
                            .map_err(|e| {
                                AtomError::Inference(format!("encode sink gradient seed: {e}"))
                            })?;
                            let id = jungle
                                .upload_to_void(bytes)
                                .await
                                .map_err(AtomError::Upload)?;
                            black_hole_type::ArtifactRef::from_object_id(id)
                        } else {
                            emission.output_id
                        };
                        let output = BackwardOps::<Op>::backward(jungle, instance_id, gradient)
                            .await
                            .map_err(AtomError::Inference)?;
                        let emission = Emission::<M, Op::InputGrad> {
                            metadata: emission.metadata,
                            output_id: output,
                        };
                        let id = jungle
                            .upload_to_void(postcard::to_allocvec(&emission)?)
                            .await
                            .map_err(AtomError::Upload)?;
                        PipelineResponse::Artifact {
                            micro_batch,
                            emission_id: id,
                        }
                    }
                    PipelineCommand::Step { step, .. } => {
                        StepOps::<Op>::step(jungle, instance_id)
                            .await
                            .map_err(AtomError::Optimize)?;
                        crate::nodes::cell::effect::save_operation_checkpoint::<Op, J>(
                            jungle,
                            instance_id,
                            step,
                            checkpoint_steps,
                            checkpoint_dir.as_deref(),
                        )
                        .await
                        .map_err(AtomError::Optimize)?;
                        PipelineResponse::Stepped
                    }
                })
            }
            .await
            .unwrap_or_else(|error| {
                tracing::error!(
                    %instance_id,
                    ?micro_batch,
                    %error,
                    "pipeline stage command failed"
                );
                PipelineResponse::Failed {
                    micro_batch,
                    message: error.to_string(),
                }
            });
            jungle
                .upload_to_void_with(reply, postcard::to_allocvec(&response)?)
                .await
                .map_err(AtomError::Transmission)
        }
    }
}

pub struct ExecutePipeline<M, Op, S>(PhantomData<fn() -> (M, Op, S)>);
#[jungle::action]
impl<M, Op, S> Action for ExecutePipeline<M, Op, S>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: BackwardContract<Metadata = M> + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
    Op::OutputGrad: Send,
    Op::InputGrad: Send,
{
    type Effect = ExecutePipelineCommand<M, Op>;
    type Input = PipelineCommand;
    type Output = ();
    fn emit(
        state: &CellState<S>,
        command: Self::Input,
    ) -> (ObjectId, PipelineCommand, usize, Option<std::path::PathBuf>) {
        (
            state.model_id,
            command,
            state.checkpoint_steps,
            state.checkpoint_dir.clone(),
        )
    }
    fn absorb(
        _state: &mut CellState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|e| Failure::Message(format!("pipeline operation failed: {e}")))
    }
}

#[derive(Flow)]
pub struct BackwardOperationCellWithState<
    Op: BackwardContract<
            Input: Send,
            Output: Send,
            OutputGrad: Send,
            InputGrad: Send,
            Metadata: Serialize + DeserializeOwned + Send + Sync + 'static,
        > + Send
        + Sync
        + 'static,
    S,
>(
    Step<crate::nodes::cell::action::InitRecvId<S>>,
    Step<crate::nodes::cell::action::GenerateModelId<S>>,
    Step<crate::nodes::cell::action::StartOperation<Op, S>>,
    While<Always<CellState<S>, ()>, BackwardOperationMicrostep<Op, S>>,
);

#[derive(Flow)]
pub struct BackwardOperationMicrostep<
    Op: BackwardContract<
            Input: Send,
            Output: Send,
            OutputGrad: Send,
            InputGrad: Send,
            Metadata: Serialize + DeserializeOwned + Send + Sync + 'static,
        > + Send
        + Sync
        + 'static,
    S,
>(
    Step<WaitForPipelineCommand<S>>,
    Step<ExecutePipeline<<Op as TensorContract>::Metadata, Op, S>>,
);

pub type BackwardOperationCell<Op, S = ()> = BackwardOperationCellWithState<Op, S>;
pub type BackwardOperationPrimordium<Op, S = ()> = BackwardOperationCellWithState<Op, S>;

/// State for a Fusion-style stage shared by two data-parallel pipeline lanes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DataParallelOperationState {
    pub model_ids: [ObjectId; 2],
    pub recv_ids: [ObjectId; 2],
}

pub struct InitDataParallelOperation;
#[jungle::action(carry = FusionSeed)]
impl Action for InitDataParallelOperation {
    type Effect = NoEffect;
    type Input = FusionSeed;
    type Output = ();

    fn emit(_state: &DataParallelOperationState, seed: Self::Input) -> ((), FusionSeed) {
        ((), seed)
    }

    fn absorb(
        state: &mut DataParallelOperationState,
        output: EffectCompletion<Self::Effect>,
        seed: FusionSeed,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("initialize data-parallel stage failed".into()))?;
        state.recv_ids = [seed.p1_recv_id, seed.p2_recv_id];
        Ok(())
    }
}

pub struct GenerateDataParallelModelIdsEffect;
#[jungle::effect(id = 90)]
impl<J> Effect<J> for GenerateDataParallelModelIdsEffect {
    type In = ();
    type Out = [ObjectId; 2];
    type Err = AtomError;

    fn effect(
        _jungle: &J,
        _input: (),
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async { Ok([Uuid::new_v4(), Uuid::new_v4()]) }
    }
}

pub struct GenerateDataParallelModelIds;
#[jungle::action]
impl Action for GenerateDataParallelModelIds {
    type Effect = GenerateDataParallelModelIdsEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &DataParallelOperationState, _input: ()) {}

    fn absorb(
        state: &mut DataParallelOperationState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        state.model_ids = output.map_err(|error| {
            Failure::Message(format!("generate data-parallel model IDs failed: {error}"))
        })?;
        Ok(())
    }
}

pub struct StartDataParallelOperationEffect<Op>(PhantomData<fn() -> Op>);
#[jungle::effect(id = 91)]
impl<Op, J> Effect<J> for StartDataParallelOperationEffect<Op>
where
    Op: TensorContract + Send + Sync + 'static,
    J: MassOps<Op>,
{
    type In = [ObjectId; 2];
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        ids: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left = MassOps::<Op>::start_operation(jungle, ids[0]);
            let right = MassOps::<Op>::start_operation(jungle, ids[1]);
            futures::future::try_join(left, right)
                .await
                .map(|_| ())
                .map_err(AtomError::ModelStart)
        }
    }
}

pub struct StartDataParallelOperation<Op>(PhantomData<fn() -> Op>);
#[jungle::action]
impl<Op> Action for StartDataParallelOperation<Op>
where
    Op: TensorContract + Send + Sync + 'static,
{
    type Effect = StartDataParallelOperationEffect<Op>;
    type Input = ();
    type Output = ();

    fn emit(state: &DataParallelOperationState, _input: ()) -> [ObjectId; 2] {
        state.model_ids
    }

    fn absorb(
        _state: &mut DataParallelOperationState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|error| {
            Failure::Message(format!("start data-parallel operation failed: {error}"))
        })
    }
}

pub struct WaitForDataParallelPipelineCommandsEffect;
#[jungle::effect(id = 92)]
impl<J: VoidOps> Effect<J> for WaitForDataParallelPipelineCommandsEffect {
    type In = [ObjectId; 2];
    type Out = [OperationalControl<PipelineCommand>; 2];
    type Err = AtomError;

    fn effect(
        jungle: &J,
        ids: Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            let left = jungle.wait_for_operational_control(ids[0]);
            let right = jungle.wait_for_operational_control(ids[1]);
            let (left, right) = futures::future::try_join(left, right)
                .await
                .map_err(AtomError::Transmission)?;
            Ok([left, right])
        }
    }
}

pub struct WaitForDataParallelPipelineCommands;
#[jungle::action]
impl Action for WaitForDataParallelPipelineCommands {
    type Effect = WaitForDataParallelPipelineCommandsEffect;
    type Input = ();
    type Output = [PipelineCommand; 2];

    fn emit(state: &DataParallelOperationState, _input: ()) -> [ObjectId; 2] {
        state.recv_ids
    }

    fn absorb(
        state: &mut DataParallelOperationState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let [left, right] = output.map_err(|error| {
            Failure::Message(format!("wait for data-parallel commands failed: {error}"))
        })?;
        state.recv_ids = [left.recv, right.recv];
        Ok([left.control, right.control])
    }
}

fn matching_pipeline_commands(commands: &[PipelineCommand; 2]) -> bool {
    match (&commands[0], &commands[1]) {
        (
            PipelineCommand::Forward { micro_batch: a, .. },
            PipelineCommand::Forward { micro_batch: b, .. },
        )
        | (
            PipelineCommand::Backward { micro_batch: a, .. },
            PipelineCommand::Backward { micro_batch: b, .. },
        ) => a == b,
        (PipelineCommand::Step { step: a, .. }, PipelineCommand::Step { step: b, .. }) => a == b,
        _ => false,
    }
}

pub struct ExecuteDataParallelPipelineCommands<M, Op>(PhantomData<fn() -> (M, Op)>);
#[jungle::effect(id = 93)]
impl<M, Op, J> Effect<J> for ExecuteDataParallelPipelineCommands<M, Op>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: BackwardContract<Metadata = M> + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
    Op::OutputGrad: Send,
    Op::InputGrad: Send,
    J: VoidOps + MassOps<Op> + BackwardOps<Op> + StepOps<Op> + CheckpointOps<Op>,
{
    type In = ([ObjectId; 2], [PipelineCommand; 2]);
    type Out = ();
    type Err = AtomError;

    fn effect(
        jungle: &J,
        (ids, commands): Self::In,
    ) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            if !matching_pipeline_commands(&commands) {
                return Err(AtomError::Inference(
                    "data-parallel stage received mismatched lane commands".into(),
                ));
            }
            if let [PipelineCommand::Step { reply: left, .. }, PipelineCommand::Step { reply: right, .. }] =
                &commands
            {
                let response = match StepOps::<Op>::step(jungle, ids[0]).await {
                    Ok(()) => PipelineResponse::Stepped,
                    Err(message) => PipelineResponse::Failed {
                        micro_batch: None,
                        message,
                    },
                };
                let bytes = postcard::to_allocvec(&response)?;
                let left = jungle.upload_to_void_with(*left, bytes.clone());
                let right = jungle.upload_to_void_with(*right, bytes);
                futures::future::try_join(left, right)
                    .await
                    .map(|_| ())
                    .map_err(AtomError::Transmission)
            } else {
                let [left_command, right_command] = commands;
                let left = <ExecutePipelineCommand<M, Op> as Effect<J>>::effect(
                    jungle,
                    (ids[0], left_command, 0, None),
                );
                let right = <ExecutePipelineCommand<M, Op> as Effect<J>>::effect(
                    jungle,
                    (ids[1], right_command, 0, None),
                );
                futures::future::try_join(left, right).await.map(|_| ())
            }
        }
    }
}

pub struct ExecuteDataParallelPipeline<M, Op>(PhantomData<fn() -> (M, Op)>);
#[jungle::action]
impl<M, Op> Action for ExecuteDataParallelPipeline<M, Op>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
    Op: BackwardContract<Metadata = M> + Send + Sync + 'static,
    Op::Input: Send,
    Op::Output: Send,
    Op::OutputGrad: Send,
    Op::InputGrad: Send,
{
    type Effect = ExecuteDataParallelPipelineCommands<M, Op>;
    type Input = [PipelineCommand; 2];
    type Output = ();

    fn emit(
        state: &DataParallelOperationState,
        commands: Self::Input,
    ) -> ([ObjectId; 2], [PipelineCommand; 2]) {
        (state.model_ids, commands)
    }

    fn absorb(
        _state: &mut DataParallelOperationState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|error| {
            Failure::Message(format!("data-parallel pipeline operation failed: {error}"))
        })
    }
}

#[derive(Flow)]
pub struct DataParallelBackwardOperationCell<
    Op: BackwardContract<
            Input: Send,
            Output: Send,
            OutputGrad: Send,
            InputGrad: Send,
            Metadata: Serialize + DeserializeOwned + Send + Sync + 'static,
        > + Send
        + Sync
        + 'static,
>(
    Step<InitDataParallelOperation>,
    Step<GenerateDataParallelModelIds>,
    Step<StartDataParallelOperation<Op>>,
    While<Always<DataParallelOperationState, ()>, DataParallelBackwardOperationMicrostep<Op>>,
);

#[derive(Flow)]
pub struct DataParallelBackwardOperationMicrostep<
    Op: BackwardContract<
            Input: Send,
            Output: Send,
            OutputGrad: Send,
            InputGrad: Send,
            Metadata: Serialize + DeserializeOwned + Send + Sync + 'static,
        > + Send
        + Sync
        + 'static,
>(
    Step<WaitForDataParallelPipelineCommands>,
    Step<ExecuteDataParallelPipeline<<Op as TensorContract>::Metadata, Op>>,
);

pub type DataParallelBackwardOperationPrimordium<Op> = DataParallelBackwardOperationCell<Op>;
