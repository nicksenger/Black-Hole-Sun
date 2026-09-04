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
use crate::ops::{BackwardOps, MassOps, StepOps, VoidOps};
use crate::topology::{BoundaryInit, SunTopology, SunTopologyState};
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
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineResponse {
    Artifact {
        micro_batch: usize,
        emission_id: ObjectId,
    },
    Stepped,
}

#[derive(Clone, Debug)]
pub struct PipelineBackwardState<S = ()> {
    pub topology: Arc<Mutex<SunTopology>>,
    pub inner: S,
    /// Current command inbox for each operation input port.
    pub inboxes: HashMap<u32, ObjectId>,
    pub micro_batches: Vec<ArtifactDelivery<()>>,
    pub epoch: usize,
}

impl<S: Default> Default for PipelineBackwardState<S> {
    fn default() -> Self {
        Self {
            topology: Arc::new(Mutex::new(SunTopology::default())),
            inner: S::default(),
            inboxes: HashMap::new(),
            micro_batches: Vec::new(),
            epoch: 0,
        }
    }
}

impl<S> SunTopologyState for PipelineBackwardState<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

pub struct NeedsMicroBatches<S, const M: usize>(PhantomData<fn() -> S>);
impl<S, const M: usize> Predicate<(&PipelineBackwardState<S>, &())> for NeedsMicroBatches<S, M> {
    fn eval((state, _): &(&PipelineBackwardState<S>, &())) -> bool {
        state.micro_batches.len() < M.max(1)
    }
}

pub struct BeginEpoch<S>(PhantomData<fn() -> S>);
#[jungle::action]
impl<S> Action for BeginEpoch<S> {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();
    fn emit(_state: &PipelineBackwardState<S>, _input: ()) {}
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<(), Failure> {
        output.map_err(|_| Failure::Message("begin pipeline epoch failed".into()))?;
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
pub struct PipelineEpochResult {
    pub epoch: usize,
    pub outputs: Vec<ArtifactDelivery<()>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RunPipelineInput {
    nodes: Vec<u32>,
    ports: Vec<u32>,
    linear: bool,
    inboxes: HashMap<u32, ObjectId>,
    micro_batches: Vec<ArtifactDelivery<()>>,
    epoch: usize,
}

#[derive(Serialize, Deserialize)]
pub struct RunPipelineOutput {
    inboxes: HashMap<u32, ObjectId>,
    outputs: Vec<ObjectId>,
    epoch: usize,
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
            let m = input.micro_batches.len();
            let p = nodes.len();
            let mut inboxes = input.inboxes;
            let mut forward = vec![vec![None::<ObjectId>; p]; m];
            let mut backward = vec![vec![None::<ObjectId>; p]; m];
            let mut next_fwd = vec![0usize; p];
            let mut next_bwd = vec![0usize; p];
            let mut busy = vec![None::<PendingReply>; p];

            while next_bwd.iter().any(|&n| n < m) || busy.iter().any(Option::is_some) {
                for s in 0..p {
                    if busy[s].is_some() {
                        continue;
                    }
                    let b = next_bwd[s];
                    let warmup = (p - s).min(m);
                    let backward_ready = b < m
                        && next_fwd[s] >= warmup
                        && forward[b][s].is_some()
                        && (s + 1 == p || backward[b][s + 1].is_some());
                    if backward_ready {
                        let gradient = if s + 1 == p {
                            forward[b][s].unwrap()
                        } else {
                            backward[b][s + 1].unwrap()
                        };
                        let reply = send_command(jungle, &mut inboxes, ports[s], |reply| {
                            PipelineCommand::Backward {
                                micro_batch: b,
                                gradient_emission_id: gradient,
                                seed_from_forward: s + 1 == p,
                                reply,
                            }
                        })
                        .await?;
                        busy[s] = Some(PendingReply {
                            kind: WorkKind::Backward,
                            micro_batch: b,
                            reply,
                        });
                        next_bwd[s] += 1;
                        continue;
                    }
                    let f = next_fwd[s];
                    let forward_ready = f < m && (s == 0 || forward[f][s - 1].is_some());
                    if forward_ready {
                        let emission = if s == 0 {
                            input.micro_batches[f].emission_id.id()
                        } else {
                            forward[f][s - 1].unwrap()
                        };
                        let reply = send_command(jungle, &mut inboxes, ports[s], |reply| {
                            PipelineCommand::Forward {
                                micro_batch: f,
                                emission_id: emission,
                                reply,
                            }
                        })
                        .await?;
                        busy[s] = Some(PendingReply {
                            kind: WorkKind::Forward,
                            micro_batch: f,
                            reply,
                        });
                        next_fwd[s] += 1;
                    }
                }

                let mut progressed = false;
                for s in 0..p {
                    let Some(pending) = busy[s] else { continue };
                    let Some(response) = wait_response(jungle, pending.reply).await? else {
                        continue;
                    };
                    let PipelineResponse::Artifact {
                        micro_batch,
                        emission_id,
                    } = response
                    else {
                        return Err("stage returned step acknowledgement for tensor work".into());
                    };
                    if micro_batch != pending.micro_batch {
                        return Err("stage returned the wrong micro-batch".into());
                    }
                    match pending.kind {
                        WorkKind::Forward => forward[micro_batch][s] = Some(emission_id),
                        WorkKind::Backward => backward[micro_batch][s] = Some(emission_id),
                    }
                    busy[s] = None;
                    progressed = true;
                }
                if !progressed {
                    tokio::task::yield_now().await;
                }
            }

            let mut steps = Vec::with_capacity(p);
            for &port in ports.iter().take(p) {
                let reply = send_command(jungle, &mut inboxes, port, |reply| {
                    PipelineCommand::Step { reply }
                })
                .await?;
                steps.push(reply);
            }
            for reply in steps {
                loop {
                    if let Some(response) = wait_response(jungle, reply).await? {
                        if !matches!(response, PipelineResponse::Stepped) {
                            return Err("stage returned artifact for step".into());
                        }
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            }
            Ok(RunPipelineOutput {
                inboxes,
                outputs: (0..m)
                    .map(|mb| forward[mb][p - 1].expect("all forwards completed"))
                    .collect(),
                epoch: input.epoch,
            })
        }
    }
}

pub struct RunPipeline<Output, S>(PhantomData<fn() -> (Output, S)>);
#[jungle::action]
impl<Output: Send + 'static, S> Action for RunPipeline<Output, S> {
    type Effect = RunPipelineEffect;
    type Input = ();
    type Output = PipelineEpochResult;
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
            .collect::<Vec<_>>();
        let linear = ports.len() == nodes.len()
            && nodes.windows(2).all(|pair| {
                let outgoing = topology.outgoing.get(&pair[0]).cloned().unwrap_or_default();
                outgoing.len() == 1 && outgoing[0].vertex_id == pair[1]
            });
        RunPipelineInput {
            nodes,
            ports,
            linear,
            inboxes: state.inboxes.clone(),
            micro_batches: state.micro_batches.clone(),
            epoch: state.epoch,
        }
    }
    fn absorb(
        state: &mut PipelineBackwardState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let output = output.map_err(|e| Failure::Message(format!("pipeline epoch failed: {e}")))?;
        state.inboxes = output.inboxes;
        state.epoch += 1;
        Ok(PipelineEpochResult {
            epoch: output.epoch,
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
pub struct PipelineEpoch<
    Source,
    Input: Send + 'static,
    Output: Send + 'static,
    Policy,
    S,
    const M: usize,
>(
    Step<BeginEpoch<S>>,
    While<NeedsMicroBatches<S, M>, CollectOne<Source, Input, S>>,
    Step<RunPipeline<Output, S>>,
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
    While<Always<PipelineBackwardState<S>, ()>, PipelineEpoch<Source, Input, Output, Policy, S, M>>,
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
    J: VoidOps + MassOps<Op> + BackwardOps<Op> + StepOps<Op>,
{
    type In = (ObjectId, PipelineCommand);
    type Out = ();
    type Err = AtomError;
    fn effect(
        jungle: &J,
        (instance_id, command): Self::In,
    ) -> impl Future<Output = Result<(), AtomError>> + Send {
        async move {
            let (reply, response) = match command {
                PipelineCommand::Forward {
                    micro_batch,
                    emission_id,
                    reply,
                } => {
                    let bytes = jungle
                        .download_raw(emission_id)
                        .await
                        .map_err(AtomError::Download)?;
                    let emission: Emission<M, Op::Input> = postcard::from_bytes(&bytes)?;
                    let output = MassOps::<Op>::forward(jungle, instance_id, emission.output_id)
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
                    (
                        reply,
                        PipelineResponse::Artifact {
                            micro_batch,
                            emission_id: id,
                        },
                    )
                }
                PipelineCommand::Backward {
                    micro_batch,
                    gradient_emission_id,
                    seed_from_forward,
                    reply,
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
                        let output = black_hole_spec::decode_output::<Op>(&bytes).map_err(|e| {
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
                    (
                        reply,
                        PipelineResponse::Artifact {
                            micro_batch,
                            emission_id: id,
                        },
                    )
                }
                PipelineCommand::Step { reply } => {
                    StepOps::<Op>::step(jungle, instance_id)
                        .await
                        .map_err(AtomError::Optimize)?;
                    (reply, PipelineResponse::Stepped)
                }
            };
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
    fn emit(state: &CellState<S>, command: Self::Input) -> (ObjectId, PipelineCommand) {
        (state.model_id, command)
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
