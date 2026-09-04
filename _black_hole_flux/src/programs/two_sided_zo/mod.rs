//! The two-sided zeroth-order training strategy ("Sun").
//!
//! Owns the epoch driver ([`SunFlow`]), the propagation branch states
//! ([`PropA`] / [`PropB`]), the shared bookkeeping ([`TwoSidedZoInner`]), and
//! the program entrypoints (`TwoSidedZo*`). Pipeline actions and effects live
//! in [`action`](self::action) / [`effect`](self::effect).

pub mod action;
pub mod effect;

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use black_hole_type::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use jungle_zoo::predicate::Always;
use uuid::Uuid;

use crate::compile::{BlackHole, SunProgram};
use crate::nodes::fusion::action::FusionSeed;
use crate::topology::{
    BoundaryInit, SunAppearance, SunEdgeAppearance, SunNodeAppearance, SunNodeState,
    SunOperationalState, SunStateView, SunTopology, SunTopologyState,
};
use action::PropagationState;

/// State for two-sided propagation branch A.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropA {
    pub topology: Arc<Mutex<SunTopology>>,
    /// Shared two-sided bookkeeping (Arc so both branches share it).
    pub shared: Arc<Mutex<TwoSidedZoInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// State for propagation branch B.
#[derive(Optic, Clone, Default, Debug)]
pub struct PropB {
    pub topology: Arc<Mutex<SunTopology>>,
    /// Shared two-sided bookkeeping (Arc so both branches share it).
    pub shared: Arc<Mutex<TwoSidedZoInner>>,
    /// Unfinished nodes and their unresolved incoming-edge counts.
    pub pending: HashMap<u32, usize>,
    /// Unfinished nodes whose incoming edges have all completed.
    pub ready: HashSet<u32>,
}

/// Runtime state that tracks the topology and transmission endpoints
/// for a sun of spawned animals.
#[derive(Optic, Clone, Debug)]
pub struct SunStateWithInner<S> {
    /// Program-neutral graph and observation data.
    pub topology: Arc<Mutex<SunTopology>>,
    /// State for propagation branch A — uses p1_tx / p1_rx maps.
    #[jungle(focus = a)]
    pub a: PropA,
    /// State for propagation branch B — uses p2_tx / p2_rx maps.
    #[jungle(focus = b)]
    pub b: PropB,
    /// User-provided state threaded through Sun actions and flows.
    pub inner: S,
    /// Generated second-pass root transmissions waiting for phase B.
    pub propagation_down_inputs: VecDeque<Transmission>,
    /// Generated first-pass root transmissions, one per accumulation step.
    pub propagation_up_inputs: Vec<Transmission>,
    /// First-pass sink outputs captured during phase A accumulation.
    pub propagation_up_outputs: Vec<Transmission>,
    /// Propagation outputs collected across accumulation microsteps.
    pub propagation_pairs: Vec<(Transmission, Transmission)>,
    /// Step-indexed first-pass input mailboxes per input port.
    pub p1_step_tx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed second-pass input mailboxes per input port.
    pub p2_step_tx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed first-pass output mailboxes per node.
    pub p1_step_rx: Vec<HashMap<u32, ObjectId>>,
    /// Step-indexed second-pass output mailboxes per node.
    pub p2_step_rx: Vec<HashMap<u32, ObjectId>>,
    /// Completed first-pass microsteps per node.
    pub node_p1_completed: HashMap<u32, usize>,
    /// Completed second-pass microsteps per node.
    pub node_p2_completed: HashMap<u32, usize>,
    /// First-pass root microsteps already seeded per root node.
    pub root_p1_sent: HashMap<u32, usize>,
    /// Second-pass root microsteps already seeded per root node.
    pub root_p2_sent: HashMap<u32, usize>,
    /// Number of node phase-microsteps completed in the current epoch.
    pub pipeline_completions: usize,
    /// Total node phase-microsteps expected in the current epoch.
    pub pipeline_target_completions: usize,
    /// Cached sink node id for the currently finalized graph.
    pub sink_id: Option<u32>,
}

/// State available to Sun actions and flows.
///
/// The generic `S` payload is available to user flows via
/// [`SunState::inner`] and defaults to `()`.
pub type TwoSidedZoState<S = ()> = SunStateWithInner<S>;
/// Compatibility name for the state owned by the [`TwoSidedZo`] program.
pub type SunState<S = ()> = TwoSidedZoState<S>;

impl<S> Default for SunStateWithInner<S>
where
    S: Default,
{
    fn default() -> Self {
        let topology = Arc::new(Mutex::new(SunTopology::default()));
        let shared = Arc::new(Mutex::new(TwoSidedZoInner::default()));
        Self {
            a: PropA {
                topology: Arc::clone(&topology),
                shared: Arc::clone(&shared),
                ..PropA::default()
            },
            b: PropB {
                topology: Arc::clone(&topology),
                shared,
                ..PropB::default()
            },
            topology,
            inner: S::default(),
            propagation_down_inputs: VecDeque::new(),
            propagation_up_inputs: Vec::new(),
            propagation_up_outputs: Vec::new(),
            propagation_pairs: Vec::new(),
            p1_step_tx: Vec::new(),
            p2_step_tx: Vec::new(),
            p1_step_rx: Vec::new(),
            p2_step_rx: Vec::new(),
            node_p1_completed: HashMap::new(),
            node_p2_completed: HashMap::new(),
            root_p1_sent: HashMap::new(),
            root_p2_sent: HashMap::new(),
            pipeline_completions: 0,
            pipeline_target_completions: 0,
            sink_id: None,
        }
    }
}

impl<S> SunStateWithInner<S> {
    /// Build a deterministic, serializable view of the runtime graph.
    pub fn appearance(&self) -> SunAppearance {
        let topology = self.topology.lock().unwrap();
        let strategy = self.a.shared.lock().unwrap();
        let grad_steps = strategy.grad_steps.max(1);
        let mut nodes = topology
            .journey_ids
            .keys()
            .copied()
            .map(|id| SunNodeAppearance {
                id,
                journey_id: topology
                    .journey_ids
                    .get(&id)
                    .copied()
                    .unwrap_or_else(Uuid::nil),
                warp_journey_id: topology
                    .warp_journey_ids
                    .get(&id)
                    .copied()
                    .unwrap_or_else(Uuid::nil),
                label: topology
                    .node_labels
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("cell {id}")),
                input_ports: topology.vertex_ports.get(&id).cloned().unwrap_or_default(),
                operational_state: topology
                    .node_operational_states
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                phase_annotation: topology.node_phase_annotations.get(&id).cloned(),
                state: strategy.node_states.get(&id).copied().unwrap_or_default(),
                state_sequence: topology
                    .node_state_sequences
                    .get(&id)
                    .copied()
                    .unwrap_or_default(),
                grad_step: strategy
                    .node_grad_steps
                    .get(&id)
                    .copied()
                    .unwrap_or_else(|| strategy.current_grad_step()),
            })
            .collect::<Vec<_>>();
        nodes.sort_by_key(|node| node.id);

        let mut edges = topology
            .outgoing
            .iter()
            .flat_map(|(&source, targets)| {
                targets.iter().map(move |target| SunEdgeAppearance {
                    source,
                    target: target.vertex_id,
                    target_port: target.port_id,
                })
            })
            .collect::<Vec<_>>();
        edges.sort_by_key(|edge| (edge.source, edge.target, edge.target_port));
        edges.dedup();

        SunAppearance {
            finalized: topology.finalized,
            grad_steps,
            nodes,
            edges,
        }
    }
}

impl<S> SunTopologyState for SunStateWithInner<S> {
    fn topology(&self) -> &Arc<Mutex<SunTopology>> {
        &self.topology
    }
}

impl<S> SunStateView for SunStateWithInner<S> {
    fn sun_appearance(&self) -> SunAppearance {
        self.appearance()
    }
}

/// Shared inner state accessible by both propagation branches via Arc<Mutex>.
#[derive(Optic, Clone, Default, Debug)]
pub struct TwoSidedZoInner {
    /// Latest observable orchestration phase reached by each internal vertex.
    pub node_states: HashMap<u32, SunNodeState>,
    /// 1-based gradient accumulation step currently associated with each vertex.
    pub node_grad_steps: HashMap<u32, usize>,
    /// Nodes whose first-pass output has been received in the current epoch.
    pub p1_completed: HashSet<u32>,
    /// Nodes whose second pass has been sent in the current epoch.
    pub p2_sent: HashSet<u32>,
    /// Number of propagation microsteps per optimization epoch.
    pub grad_steps: usize,
    /// Current microstep index (0-based) inside the accumulation epoch.
    pub active_micro_step: usize,
    /// First-pass input endpoints keyed by port id.
    pub p1_tx: HashMap<u32, ObjectId>,
    /// Next first-pass endpoints used between microsteps when accumulation > 1.
    pub next_p1_tx: HashMap<u32, ObjectId>,
    /// First-pass output endpoints keyed by vertex id.
    pub p1_rx: HashMap<u32, ObjectId>,
    /// Second-pass input endpoints keyed by port id.
    pub p2_tx: HashMap<u32, ObjectId>,
    /// Next second-pass endpoints used between microsteps when accumulation > 1.
    pub next_p2_tx: HashMap<u32, ObjectId>,
    /// Second-pass output endpoints keyed by vertex id.
    pub p2_rx: HashMap<u32, ObjectId>,
    /// Potentiation input endpoints keyed by port id.
    pub po_tx: HashMap<u32, ObjectId>,
}

impl TwoSidedZoInner {
    fn current_grad_step(&self) -> usize {
        let grad_steps = self.grad_steps.max(1);
        self.active_micro_step
            .saturating_add(1)
            .clamp(1, grad_steps)
    }

    fn grad_step_for_phase(&self, phase: SunNodeState) -> usize {
        match phase {
            SunNodeState::Optimization => self.grad_steps.max(1),
            _ => self.current_grad_step(),
        }
    }

    fn record_state_sent(&mut self, topology: &mut SunTopology, node_id: u32, phase: SunNodeState) {
        let grad_step = self.grad_step_for_phase(phase);
        let current = self.node_states.get(&node_id).copied().unwrap_or_default();
        topology
            .node_operational_states
            .insert(node_id, SunOperationalState::Running);
        topology.node_phase_annotations.insert(
            node_id,
            match phase {
                SunNodeState::Idle => "idle",
                SunNodeState::Propagation1 => "propagation 1",
                SunNodeState::Propagation2 => "propagation 2",
                SunNodeState::Optimization => "potentiation",
            }
            .to_string(),
        );
        if current == phase {
            self.node_grad_steps.insert(node_id, grad_step);
            return;
        }

        let mut next = current;
        let mut distance = 0;
        while next != phase {
            next = match next {
                SunNodeState::Idle | SunNodeState::Optimization => SunNodeState::Propagation1,
                SunNodeState::Propagation1 => SunNodeState::Propagation2,
                SunNodeState::Propagation2 => SunNodeState::Optimization,
            };
            distance += 1;
        }

        self.node_states.insert(node_id, phase);
        *topology.node_state_sequences.entry(node_id).or_default() += distance;
        self.node_grad_steps.insert(node_id, grad_step);
    }

    pub(crate) fn record_propagation_sent(
        &mut self,
        topology: &mut SunTopology,
        node_ids: impl IntoIterator<Item = u32>,
        phase: SunNodeState,
    ) {
        debug_assert!(matches!(
            phase,
            SunNodeState::Propagation1 | SunNodeState::Propagation2
        ));
        for node_id in node_ids {
            match phase {
                SunNodeState::Propagation1 => {
                    if self.node_states.get(&node_id) != Some(&SunNodeState::Propagation2) {
                        self.record_state_sent(topology, node_id, phase);
                    }
                }
                SunNodeState::Propagation2 => {
                    self.p2_sent.insert(node_id);
                    if self.p1_completed.contains(&node_id) {
                        self.record_state_sent(topology, node_id, phase);
                    }
                }
                _ => unreachable!("propagation phase was validated above"),
            }
        }
    }

    pub(crate) fn record_propagation_completed(
        &mut self,
        topology: &mut SunTopology,
        node_id: u32,
        phase: SunNodeState,
    ) {
        topology
            .node_operational_states
            .insert(node_id, SunOperationalState::Succeeded);
        if phase == SunNodeState::Propagation1 {
            self.p1_completed.insert(node_id);
            if self.p2_sent.contains(&node_id) {
                self.record_state_sent(topology, node_id, SunNodeState::Propagation2);
            }
        }
    }

    pub(crate) fn record_optimization_sent(
        &mut self,
        topology: &mut SunTopology,
        node_ids: impl IntoIterator<Item = u32>,
    ) {
        for node_id in node_ids {
            self.record_state_sent(topology, node_id, SunNodeState::Optimization);
            self.p1_completed.remove(&node_id);
            self.p2_sent.remove(&node_id);
        }
    }
}

/// Compatibility name for the two-sided strategy state.
pub type SunInner = TwoSidedZoInner;

/// Compatibility program used by the descriptor-step aliases.
pub struct DeploymentProgram<S, const ACCUM_STEPS: usize>(PhantomData<fn() -> S>);

impl<S, const ACCUM_STEPS: usize> SunProgram for DeploymentProgram<S, ACCUM_STEPS> {
    type State = SunState<S>;
    type Driver = ();
    type UnarySeed = crate::nodes::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::nodes::cell::action::Init {
            recv_id: inbox,
            grad_steps: ACCUM_STEPS.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: ACCUM_STEPS.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: ACCUM_STEPS.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

/// Legacy generator/policy/state bundle accepted by [`TwoSidedZoManifest`].
pub trait Manifest {
    type Generator;
    type Policy;
    type State;
}

/// The existing two-sided zeroth-order training schedule as a Sun program.
pub struct TwoSidedZoWithState<Generator, Policy, S, const ACCUM_STEPS: usize = 1>(
    PhantomData<Generator>,
    PhantomData<Policy>,
    PhantomData<fn() -> S>,
);

impl<G, P, S, const A: usize> SunProgram for TwoSidedZoWithState<G, P, S, A> {
    type State = SunState<S>;
    type Driver = Sun<G, P, S, A>;
    type UnarySeed = crate::nodes::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::nodes::cell::action::Init {
            recv_id: inbox,
            grad_steps: A.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: A.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: A.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

pub type TwoSidedZo<Generator, Policy, const ACCUM_STEPS: usize = 1> =
    TwoSidedZoWithState<Generator, Policy, (), ACCUM_STEPS>;

/// Adapts the former manifest shape to the new program-based entrypoint.
pub struct TwoSidedZoManifest<M: Manifest, const ACCUM_STEPS: usize = 1>(PhantomData<M>);

impl<M: Manifest, const A: usize> SunProgram for TwoSidedZoManifest<M, A> {
    type State = SunState<M::State>;
    type Driver = Sun<M::Generator, M::Policy, M::State, A>;
    type UnarySeed = crate::nodes::cell::action::Init;
    type BinarySeed = FusionSeed;
    type WarpSeed = BoundaryInit;

    fn unary_seed(inbox: ObjectId) -> Self::UnarySeed {
        crate::nodes::cell::action::Init {
            recv_id: inbox,
            grad_steps: A.max(1),
        }
    }
    fn unary_inbox(seed: &Self::UnarySeed) -> ObjectId {
        seed.recv_id
    }
    fn binary_seed([p1_recv_id, p2_recv_id]: [ObjectId; 2]) -> Self::BinarySeed {
        FusionSeed {
            p1_recv_id,
            p2_recv_id,
            grad_steps: A.max(1),
        }
    }
    fn binary_inboxes(seed: &Self::BinarySeed) -> [ObjectId; 2] {
        [seed.p1_recv_id, seed.p2_recv_id]
    }
    fn warp_seed(recv_id: ObjectId, warp_journey_id: Uuid) -> Self::WarpSeed {
        BoundaryInit {
            recv_id,
            grad_steps: A.max(1),
            warp_journey_id,
        }
    }
    fn register_inboxes(state: &mut Self::State, ports: &[(u32, ObjectId)]) {
        state
            .a
            .shared
            .lock()
            .unwrap()
            .p1_tx
            .extend(ports.iter().copied());
    }
}

/// Legacy stateless generator/policy bundle.
pub struct StatelessManifest<Generator, Policy, const ACCUM_STEPS: usize = 1>(
    PhantomData<Generator>,
    PhantomData<Policy>,
);

impl<G, P, const A: usize> Manifest for StatelessManifest<G, P, A> {
    type Generator = G;
    type Policy = P;
    type State = ();
}

/// Compatibility alias for code that still defines a legacy [`Manifest`].
pub type LegacySun<T, M, const ACCUM_STEPS: usize> =
    <T as BlackHole>::Sun<TwoSidedZoManifest<M, ACCUM_STEPS>>;

// ---------------------------------------------------------------------------
// Predicates — loop continuation conditions
// ---------------------------------------------------------------------------

/// Predicate that checks whether a propagation branch has unfinished nodes.
pub struct PendingNotEmpty<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&S, &Transmission)> for PendingNotEmpty<S>
where
    S: PropagationState,
{
    fn eval((state, _): &(&S, &Transmission)) -> bool {
        !state.pending().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Flow definitions — ready-node processing and orchestration
// ---------------------------------------------------------------------------
//
/// Processes the next ready node in propagation branch B.
#[derive(Flow)]
pub struct PropBLoop(Step<action::ProcessNextNode<PropB>>);

/// Processes the next ready node in propagation branch A.
#[derive(Flow)]
pub struct PropALoop(Step<action::ProcessNextNode<PropA>>);

#[derive(Flow)]
#[jungle(focus = PropB)]
pub struct PropBFlow(
    Step<action::InitializePropagation<PropB>>,
    Step<action::SendRootPropagation<PropB>>,
    While<FocusedLoopCondition<PendingNotEmpty<PropB>, PropB>, PropBLoop>,
);

#[derive(Flow)]
#[jungle(focus = PropA)]
pub struct PropAFlow(
    Step<action::InitializePropagation<PropA>>,
    Step<action::SendRootPropagation<PropA>>,
    While<FocusedLoopCondition<PendingNotEmpty<PropA>, PropA>, PropALoop>,
);

/// The two propagation branches (A and B) running in parallel via focused join.
pub type PropagationFlows = Join<PropAFlow, PropBFlow>;

/// Alias for one iteration of the propagation loop (branch A).
pub type PropagationLoop = PropALoop;

/// Predicate that keeps collecting generator outputs until `N` pairs exist.
pub struct PendingGeneratedPropagationInputs<const N: usize, S>(PhantomData<fn() -> S>);

impl<const N: usize, S> Predicate<(&SunState<S>, &())> for PendingGeneratedPropagationInputs<N, S> {
    fn eval((state, _): &(&SunState<S>, &())) -> bool {
        state.propagation_up_inputs.len() < N
    }
}

/// Predicate that keeps advancing the per-node propagation scheduler.
pub struct PendingPipelineWork<S>(PhantomData<fn() -> S>);

impl<S> Predicate<(&SunState<S>, &())> for PendingPipelineWork<S> {
    fn eval((state, _): &(&SunState<S>, &())) -> bool {
        state.pipeline_completions < state.pipeline_target_completions
    }
}

/// One generator emission pair capture step.
#[derive(Flow)]
pub struct CollectPropagationInputsStep<Generator, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Generator,
    Step<action::StorePropagationInputPair<S, GRADIENT_ACCUMULATION_STEPS>>,
);

/// One scheduler tick: seed ready roots, then process one ready node output.
#[derive(Flow)]
pub struct PipelineProgressStep<S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::SendReadyRootTasks<S, GRADIENT_ACCUMULATION_STEPS>>,
    Step<action::ProcessReadyPipelineNode<S, GRADIENT_ACCUMULATION_STEPS>>,
);

// ---------------------------------------------------------------------------
// BlackHole — the top-level orchestration flow
// ---------------------------------------------------------------------------

/// One complete training epoch: generate → propagate → apply policy → broadcast potentiation.
#[derive(Flow)]
pub struct EpochWithState<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::BeginGradientAccumulation<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<
        PendingGeneratedPropagationInputs<GRADIENT_ACCUMULATION_STEPS, S>,
        CollectPropagationInputsStep<Generator, S, GRADIENT_ACCUMULATION_STEPS>,
    >,
    Step<action::PreparePropagationPipeline<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<PendingPipelineWork<S>, PipelineProgressStep<S, GRADIENT_ACCUMULATION_STEPS>>,
    Step<action::CollectedPropagationPairs<S, GRADIENT_ACCUMULATION_STEPS>>,
    Policy,
    Step<action::BroadcastPotentiation<S>>,
);

pub type Epoch<Generator, Policy, S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1> =
    EpochWithState<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>;

/// Top-level orchestration flow that drives all underlying Cell flows
/// associated with the BlackHoleSun graph.
#[derive(Flow)]
pub struct SunFlow<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize>(
    Step<action::BuildAddrs<S, GRADIENT_ACCUMULATION_STEPS>>,
    While<
        Always<SunState<S>, ()>,
        EpochWithState<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>,
    >,
);

pub type Sun<Generator, Policy, S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1> =
    SunFlow<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>;
