//! Sun actions — spawning animals, propagation, and potentiation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use crate::sun::effect::{GenFusionSeedEffect, GenUuidEffect};
use crate::{FusionSeed, FusionState};

use super::effect::{
    BroadcastPotentiationEffect, PropagationTarget, RootPropagationSend, SendRootPropagationEffect,
    SendRootPropagationInput, SendRootTaskPropagationsEffect, WaitForNodeTransmissionEffect,
    WaitForNodeTransmissionInput,
};
use black_hole_contract::{QwenDarkInference, TensorContract};
use black_hole_spec::{ObjectId, Transmission};
use jungle_sdk::prelude::*;
use typosaurus::collections::list::{Empty, List};
use typosaurus::num::Unsigned;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// NodeIdsFromList — extract runtime node IDs from a type-level integer list
// ---------------------------------------------------------------------------

/// Trait that converts a type-level list of typenum integers into a runtime
/// vector of node IDs (u32 values).
pub trait NodeIdsFromList {
    fn node_ids() -> Vec<u32>;
}

impl NodeIdsFromList for Empty {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<H, T> NodeIdsFromList for List<(H, T)>
where
    H: Unsigned,
    T: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![<H as Unsigned>::U32];
        ids.extend(T::node_ids());
        ids
    }
}

impl NodeIdsFromList for super::TypedEdges<Empty> {
    fn node_ids() -> Vec<u32> {
        Vec::new()
    }
}

impl<P, Destination, T> NodeIdsFromList
    for super::TypedEdges<List<(super::Edge<P, Destination>, T)>>
where
    P: Unsigned,
    Destination: TensorContract,
    super::TypedEdges<T>: NodeIdsFromList,
{
    fn node_ids() -> Vec<u32> {
        let mut ids = vec![P::U32];
        ids.extend(<super::TypedEdges<T> as NodeIdsFromList>::node_ids());
        ids
    }
}

/// Produces runtime edge descriptors while enforcing compile-time bundle
/// equality between every source output and destination input.
pub trait DeclaredEdges<Source: TensorContract>: NodeIdsFromList {
    fn declared_edges() -> Vec<super::DeclaredEdge>;
}

impl<Source: TensorContract> DeclaredEdges<Source> for Empty {
    fn declared_edges() -> Vec<super::DeclaredEdge> {
        Vec::new()
    }
}

// Compatibility for the original numeric-only topology syntax. It is
// intentionally limited to the Qwen artifact bundle; generic graphs must name
// destination contracts with `TypedEdges`.
impl<Source, P, T> DeclaredEdges<Source> for List<(P, T)>
where
    Source: TensorContract<Output = <QwenDarkInference as TensorContract>::Input>,
    P: Unsigned,
    T: DeclaredEdges<Source>,
{
    fn declared_edges() -> Vec<super::DeclaredEdge> {
        let mut edges = vec![super::DeclaredEdge {
            port_id: P::U32,
            source_contract: Source::descriptor(),
            destination_contract: QwenDarkInference::descriptor(),
        }];
        edges.extend(T::declared_edges());
        edges
    }
}

impl<Source: TensorContract> DeclaredEdges<Source> for super::TypedEdges<Empty> {
    fn declared_edges() -> Vec<super::DeclaredEdge> {
        Vec::new()
    }
}

impl<Source, P, Destination, T> DeclaredEdges<Source>
    for super::TypedEdges<List<(super::Edge<P, Destination>, T)>>
where
    Source: TensorContract,
    Destination: TensorContract<Input = Source::Output>,
    P: Unsigned,
    super::TypedEdges<T>: DeclaredEdges<Source>,
{
    fn declared_edges() -> Vec<super::DeclaredEdge> {
        let mut edges = vec![super::DeclaredEdge {
            port_id: P::U32,
            source_contract: Source::descriptor(),
            destination_contract: Destination::descriptor(),
        }];
        edges.extend(<super::TypedEdges<T> as DeclaredEdges<Source>>::declared_edges());
        edges
    }
}

// ---------------------------------------------------------------------------
// Spawn — descriptor-specific animal spawning and graph registration
// ---------------------------------------------------------------------------

fn register_vertex<S>(
    state: &mut super::SunState<S>,
    vertex_id: u32,
    node_label: String,
    ports: &[(u32, ObjectId)],
    contract: black_hole_spec::ContractDescriptor,
    declared_edges: Vec<super::DeclaredEdge>,
    journey_id: Uuid,
    warp_journey_id: Option<Uuid>,
) {
    let mut inner = state.a.shared.lock().unwrap();

    inner.journey_ids.entry(vertex_id).or_insert(journey_id);
    if let Some(warp_journey_id) = warp_journey_id {
        inner.warp_journey_ids.insert(vertex_id, warp_journey_id);
    }
    inner.node_labels.entry(vertex_id).or_insert(node_label);
    inner.node_states.entry(vertex_id).or_default();
    inner.node_state_sequences.entry(vertex_id).or_default();
    inner
        .node_grad_steps
        .entry(vertex_id)
        .or_insert_with(super::default_gradient_accumulation_steps);
    inner
        .vertex_ports
        .entry(vertex_id)
        .or_insert_with(|| ports.iter().map(|(port_id, _)| *port_id).collect());
    inner.node_contracts.entry(vertex_id).or_insert(contract);
    inner
        .declared_outputs
        .entry(vertex_id)
        .or_insert_with(|| declared_edges.iter().map(|edge| edge.port_id).collect());
    inner
        .declared_edges
        .entry(vertex_id)
        .or_insert(declared_edges);

    for &(port_id, initial_recv_id) in ports {
        if inner.port_vertices.contains_key(&port_id) {
            inner.duplicate_ports.insert(port_id);
            continue;
        }
        inner.port_vertices.insert(port_id, vertex_id);
        inner.p1_tx.insert(port_id, initial_recv_id);
    }
}

fn short_type_name<T: ?Sized>() -> String {
    short_type_label(core::any::type_name::<T>())
}

fn short_type_label(label: &str) -> String {
    let mut shortened = String::with_capacity(label.len());
    let mut token = String::new();

    for ch in label.chars() {
        let token_char = ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '\'');
        if token_char {
            token.push(ch);
            continue;
        }

        push_shortened_type_token(&mut shortened, &mut token);
        shortened.push(ch);
    }

    push_shortened_type_token(&mut shortened, &mut token);
    shortened.trim().to_string()
}

fn push_shortened_type_token(shortened: &mut String, token: &mut String) {
    if token.is_empty() {
        return;
    }

    let short = token.rsplit("::").next().unwrap_or(token.as_str());
    shortened.push_str(short);
    token.clear();
}

/// Spawns and registers a [`Unary`](super::Unary) descriptor.
pub struct SpawnUnary<P, A, E, S = (), Op = QwenDarkInference>(
    PhantomData<fn() -> (P, A, E, S, Op)>,
);

#[jungle::action]
impl<P, A, E, S, Op> Action for SpawnUnary<P, A, E, S, Op>
where
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = crate::cell::action::Init>
        + super::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = crate::cell::action::Init;
    type Output = ();
    type Carry = crate::cell::action::Init;

    fn emit(
        _state: &super::SunState<S>,
        input: Self::Input,
    ) -> (crate::cell::action::Init, crate::cell::action::Init) {
        (input, input)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        initial_recv_id: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let port_id = P::U32;
        register_vertex(
            state,
            port_id,
            short_type_name::<A>(),
            &[(port_id, initial_recv_id.recv_id)],
            Op::descriptor(),
            E::declared_edges(),
            journey_id,
            None,
        );

        Ok(())
    }
}

/// Backwards-compatible name for the unary spawn action.
pub type Spawn<P, A, E, S = (), Op = QwenDarkInference> = SpawnUnary<P, A, E, S, Op>;

/// Spawns and registers a [`Binary`](super::Binary) descriptor.
pub struct SpawnBinary<P1, P2, A, E, S = (), Op = QwenDarkInference>(
    PhantomData<fn() -> (P1, P2, A, E, S, Op)>,
);

#[jungle::action]
impl<P1, P2, A, E, S, Op> Action for SpawnBinary<P1, P2, A, E, S, Op>
where
    P1: Unsigned,
    P2: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>
        + super::OperationNode<Op>,
    A::Flow: crate::fusion::FusionFlow,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = FusionSeed;
    type Output = ();
    type Carry = FusionSeed;

    fn emit(_state: &super::SunState<S>, seed: Self::Input) -> (FusionSeed, FusionSeed) {
        (seed, seed)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        seed: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let p1 = P1::U32;
        let p2 = P2::U32;
        register_vertex(
            state,
            p1,
            short_type_name::<A>(),
            &[(p1, seed.p1_recv_id), (p2, seed.p2_recv_id)],
            Op::descriptor(),
            E::declared_edges(),
            journey_id,
            None,
        );

        Ok(())
    }
}

/// Spawns and registers a [`Warp`](super::Warp) descriptor's boundary node.
///
/// This runs in two steps:
/// 1. Spawn the nested warp animal and keep its journey id.
/// 2. Spawn the boundary animal with [`super::BoundaryInit`], then register
///    the boundary journey as the parent graph vertex for scheduling.
pub struct SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, S = (), Op = QwenDarkInference>(
    PhantomData<fn() -> (P, WarpAnimalT, BoundaryAnimalT, E, S, Op)>,
);

#[jungle::action]
impl<P, WarpAnimalT, BoundaryAnimalT, E, S, Op> Action
    for SpawnWarpAnimal<P, WarpAnimalT, BoundaryAnimalT, E, S, Op>
where
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<
            Id: AnimalIdValue,
            Generation: Unsigned,
            Seed = super::BoundaryInit,
            State = crate::BoundaryState<<WarpAnimalT as Observe>::Appearance>,
        > + super::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<WarpAnimalT>;
    type Input = crate::cell::action::Init;
    type Output = (crate::cell::action::Init, Uuid);
    type Carry = crate::cell::action::Init;

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> ((), crate::cell::action::Init) {
        ((), input)
    }

    fn absorb(
        _state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: crate::cell::action::Init,
    ) -> Result<Self::Output, Failure> {
        let warp_journey_id =
            output.map_err(|e| Failure::Message(format!("warp spawn failed: {e}")))?;
        Ok((carry, warp_journey_id))
    }
}

pub struct SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, S = (), Op = QwenDarkInference>(
    PhantomData<fn() -> (P, WarpAnimalT, BoundaryAnimalT, E, S, Op)>,
);

#[jungle::action]
impl<P, WarpAnimalT, BoundaryAnimalT, E, S, Op> Action
    for SpawnWarpBoundary<P, WarpAnimalT, BoundaryAnimalT, E, S, Op>
where
    P: Unsigned,
    WarpAnimalT: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ()> + Observe,
    BoundaryAnimalT: Animal<
            Id: AnimalIdValue,
            Generation: Unsigned,
            Seed = super::BoundaryInit,
            State = crate::BoundaryState<<WarpAnimalT as Observe>::Appearance>,
        > + super::OperationNode<Op>,
    Op: TensorContract,
    E: NodeIdsFromList + DeclaredEdges<Op>,
{
    type Effect = super::effect::SpawnAnimal<BoundaryAnimalT>;
    type Input = (crate::cell::action::Init, Uuid);
    type Output = ();
    type Carry = (crate::cell::action::Init, Uuid);

    fn emit(
        _state: &super::SunState<S>,
        input: Self::Input,
    ) -> (super::BoundaryInit, (crate::cell::action::Init, Uuid)) {
        let (init, warp_journey_id) = input;
        (
            super::BoundaryInit {
                recv_id: init.recv_id,
                grad_steps: init.grad_steps,
                warp_journey_id,
            },
            (init, warp_journey_id),
        )
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: (crate::cell::action::Init, Uuid),
    ) -> Result<Self::Output, Failure> {
        let boundary_journey_id =
            output.map_err(|e| Failure::Message(format!("boundary spawn failed: {e}")))?;
        let (init, warp_journey_id) = carry;
        let port_id = P::U32;
        register_vertex(
            state,
            port_id,
            format!(
                "Warp<{}, {}>",
                short_type_name::<WarpAnimalT>(),
                short_type_name::<BoundaryAnimalT>()
            ),
            &[(port_id, init.recv_id)],
            Op::descriptor(),
            E::declared_edges(),
            boundary_journey_id,
            Some(warp_journey_id),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InitializePropagation — initialize the dynamic Kahn frontier
// ---------------------------------------------------------------------------

fn pending_dependency_counts(inner: &super::SunInner) -> HashMap<u32, usize> {
    inner
        .journey_ids
        .keys()
        .map(|&node_id| {
            let unresolved = inner.incoming.get(&node_id).map_or(0, Vec::len);
            (node_id, unresolved)
        })
        .collect()
}

fn initial_ready_nodes(pending: &HashMap<u32, usize>) -> HashSet<u32> {
    pending
        .iter()
        .filter_map(|(&node_id, &unresolved)| (unresolved == 0).then_some(node_id))
        .collect()
}

fn sorted_node_ids(nodes: &HashSet<u32>) -> Vec<u32> {
    let mut ready: Vec<_> = nodes.iter().copied().collect();
    ready.sort_unstable();
    ready
}

fn advance_frontier(
    pending: &mut HashMap<u32, usize>,
    ready: &mut HashSet<u32>,
    node_id: u32,
    outgoing: &[super::PortTarget],
) -> Result<(), Failure> {
    let unresolved = pending
        .get(&node_id)
        .copied()
        .ok_or_else(|| Failure::Message(format!("completed node {node_id} is not pending")))?;
    if unresolved != 0 {
        return Err(Failure::Message(format!(
            "completed node {node_id} still has {unresolved} unresolved predecessors"
        )));
    }
    if !ready.contains(&node_id) {
        return Err(Failure::Message(format!(
            "completed node {node_id} is not in the ready frontier"
        )));
    }

    let mut decrements = HashMap::<u32, usize>::new();
    for target in outgoing {
        *decrements.entry(target.vertex_id).or_default() += 1;
    }
    for (&target_id, &decrement) in &decrements {
        let target_unresolved = pending.get(&target_id).ok_or_else(|| {
            Failure::Message(format!("downstream node {target_id} is not pending"))
        })?;
        if *target_unresolved < decrement {
            return Err(Failure::Message(format!(
                "downstream node {target_id} has {target_unresolved} unresolved predecessors, \
                 cannot resolve {decrement}"
            )));
        }
    }

    pending.remove(&node_id);
    ready.remove(&node_id);
    for (target_id, decrement) in decrements {
        let target_unresolved = pending
            .get_mut(&target_id)
            .expect("downstream node was validated above");
        *target_unresolved -= decrement;
        if *target_unresolved == 0 {
            ready.insert(target_id);
        }
    }
    Ok(())
}

/// Initializes one propagation pass with every node's unresolved predecessor
/// count. Nodes whose count is zero form the initial ready frontier.
pub struct InitializePropagation<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for InitializePropagation<S>
where
    S: PropagationState,
{
    type Effect = NoEffect;
    type Input = Transmission;
    type Output = Transmission;
    type Carry = Transmission;

    fn emit(_state: &S, input: Self::Input) -> ((), Transmission) {
        ((), input)
    }

    fn absorb(
        state: &mut S,
        _output: EffectCompletion<Self::Effect>,
        carry: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let pending = {
            let inner = state.get_shared().lock().unwrap();
            pending_dependency_counts(&inner)
        };
        let ready = initial_ready_nodes(&pending);

        let (state_pending, state_ready) = state.scheduler_mut();
        *state_pending = pending;
        *state_ready = ready;

        Ok(carry)
    }
}

// ---------------------------------------------------------------------------
// FinalizeGraph — resolve ports, validate the DAG, and allocate phase mailboxes
// ---------------------------------------------------------------------------

fn port_ids(inner: &super::SunInner) -> Vec<u32> {
    inner.port_vertices.keys().copied().collect()
}

fn vertex_ids(inner: &super::SunInner) -> Vec<u32> {
    inner.journey_ids.keys().copied().collect()
}

fn root_vertex_ids(inner: &super::SunInner) -> Vec<u32> {
    let mut roots: Vec<_> = inner
        .incoming
        .iter()
        .filter_map(|(&node_id, sources)| sources.is_empty().then_some(node_id))
        .collect();
    roots.sort_unstable();
    roots
}

fn task_for_node<S>(
    state: &super::SunState<S>,
    node_id: u32,
    grad_steps: usize,
) -> Option<(super::SunNodeState, usize)> {
    let p1_completed = state
        .node_p1_completed
        .get(&node_id)
        .copied()
        .unwrap_or_default();
    if p1_completed < grad_steps {
        return Some((super::SunNodeState::Propagation1, p1_completed + 1));
    }

    let p2_completed = state
        .node_p2_completed
        .get(&node_id)
        .copied()
        .unwrap_or_default();
    if p2_completed < grad_steps {
        return Some((super::SunNodeState::Propagation2, p2_completed + 1));
    }

    None
}

fn task_deps_satisfied<S>(
    state: &super::SunState<S>,
    inner: &super::SunInner,
    node_id: u32,
    phase: super::SunNodeState,
    step: usize,
) -> bool {
    let is_root = inner
        .incoming
        .get(&node_id)
        .is_none_or(|sources| sources.is_empty());
    if is_root {
        return match phase {
            super::SunNodeState::Propagation1 => {
                state
                    .root_p1_sent
                    .get(&node_id)
                    .copied()
                    .unwrap_or_default()
                    >= step
            }
            super::SunNodeState::Propagation2 => {
                state
                    .root_p2_sent
                    .get(&node_id)
                    .copied()
                    .unwrap_or_default()
                    >= step
            }
            _ => false,
        };
    }

    let predecessors = inner.incoming.get(&node_id).cloned().unwrap_or_default();
    predecessors.into_iter().all(|pred_id| match phase {
        super::SunNodeState::Propagation1 => {
            state
                .node_p1_completed
                .get(&pred_id)
                .copied()
                .unwrap_or_default()
                >= step
        }
        super::SunNodeState::Propagation2 => {
            state
                .node_p2_completed
                .get(&pred_id)
                .copied()
                .unwrap_or_default()
                >= step
        }
        _ => false,
    })
}

fn ready_nodes<S>(state: &super::SunState<S>, inner: &super::SunInner) -> Vec<u32> {
    let grad_steps = inner.grad_steps.max(1);
    let mut ready = Vec::new();
    for node_id in vertex_ids(inner) {
        let Some((phase, step)) = task_for_node(state, node_id, grad_steps) else {
            continue;
        };
        if task_deps_satisfied(state, inner, node_id, phase, step) {
            ready.push(node_id);
        }
    }
    ready.sort_unstable();
    ready
}

fn step_target<S>(
    state: &super::SunState<S>,
    inner: &super::SunInner,
    phase: super::SunNodeState,
    step: usize,
    port_id: u32,
) -> PropagationTarget {
    let node_id = *inner
        .port_vertices
        .get(&port_id)
        .unwrap_or_else(|| panic!("missing node for port {port_id}"));
    let idx = step.checked_sub(1).expect("step index must be at least 1");
    let grad_steps = inner.grad_steps.max(1);
    let input_id = match phase {
        super::SunNodeState::Propagation1 => state
            .p1_step_tx
            .get(idx)
            .and_then(|map| map.get(&port_id).copied()),
        super::SunNodeState::Propagation2 => state
            .p2_step_tx
            .get(idx)
            .and_then(|map| map.get(&port_id).copied()),
        _ => None,
    }
    .unwrap_or_else(|| panic!("missing input mailbox for port {port_id} at {phase:?} step {step}"));

    let next_input_id = match phase {
        super::SunNodeState::Propagation1 if step < grad_steps => state
            .p1_step_tx
            .get(idx + 1)
            .and_then(|map| map.get(&port_id).copied()),
        super::SunNodeState::Propagation1 => state
            .p2_step_tx
            .first()
            .and_then(|map| map.get(&port_id).copied()),
        super::SunNodeState::Propagation2 if step < grad_steps => state
            .p2_step_tx
            .get(idx + 1)
            .and_then(|map| map.get(&port_id).copied()),
        super::SunNodeState::Propagation2 => inner.po_tx.get(&port_id).copied(),
        _ => None,
    }
    .unwrap_or_else(|| {
        panic!("missing next input mailbox for port {port_id} at {phase:?} step {step}")
    });

    let output_id = match phase {
        super::SunNodeState::Propagation1 => state
            .p1_step_rx
            .get(idx)
            .and_then(|map| map.get(&node_id).copied()),
        super::SunNodeState::Propagation2 => state
            .p2_step_rx
            .get(idx)
            .and_then(|map| map.get(&node_id).copied()),
        _ => None,
    }
    .unwrap_or_else(|| {
        panic!("missing output mailbox for node {node_id} at {phase:?} step {step}")
    });

    PropagationTarget {
        node_id,
        port_id,
        input_id,
        next_input_id,
        output_id,
    }
}

fn task_output_id<S>(
    state: &super::SunState<S>,
    node_id: u32,
    phase: super::SunNodeState,
    step: usize,
) -> Option<ObjectId> {
    let idx = step.checked_sub(1)?;
    match phase {
        super::SunNodeState::Propagation1 => state.p1_step_rx.get(idx)?.get(&node_id).copied(),
        super::SunNodeState::Propagation2 => state.p2_step_rx.get(idx)?.get(&node_id).copied(),
        _ => None,
    }
}

fn reset_epoch_mailboxes(inner: &mut super::SunInner) {
    let port_ids = port_ids(inner);
    let vertex_ids = vertex_ids(inner);

    inner.p2_tx.clear();
    inner.p1_rx.clear();
    inner.p2_rx.clear();
    inner.next_p1_tx.clear();
    inner.next_p2_tx.clear();

    for port_id in &port_ids {
        inner.p2_tx.insert(*port_id, Uuid::new_v4());
    }
    for vertex_id in &vertex_ids {
        inner.p1_rx.insert(*vertex_id, Uuid::new_v4());
        inner.p2_rx.insert(*vertex_id, Uuid::new_v4());
    }
    if inner.grad_steps > 1 {
        for port_id in &port_ids {
            inner.next_p1_tx.insert(*port_id, Uuid::new_v4());
            inner.next_p2_tx.insert(*port_id, Uuid::new_v4());
        }
    }
}

fn prepare_next_up_microstep(inner: &mut super::SunInner, completed_steps: usize) {
    inner.p1_tx = inner.next_p1_tx.clone();
    inner.active_micro_step = completed_steps;

    let port_ids = port_ids(inner);
    let vertex_ids = vertex_ids(inner);

    inner.p1_rx.clear();
    inner.next_p1_tx.clear();

    for vertex_id in &vertex_ids {
        inner.p1_rx.insert(*vertex_id, Uuid::new_v4());
    }
    if completed_steps + 1 < inner.grad_steps {
        for port_id in port_ids {
            inner.next_p1_tx.insert(port_id, Uuid::new_v4());
        }
    }
}

fn prepare_next_down_microstep(inner: &mut super::SunInner, completed_steps: usize) {
    inner.p2_tx = inner.next_p2_tx.clone();
    inner.active_micro_step = completed_steps;

    let port_ids = port_ids(inner);
    let vertex_ids = vertex_ids(inner);

    inner.p2_rx.clear();
    inner.next_p2_tx.clear();

    for vertex_id in &vertex_ids {
        inner.p2_rx.insert(*vertex_id, Uuid::new_v4());
    }
    if completed_steps + 1 < inner.grad_steps {
        for port_id in port_ids {
            inner.next_p2_tx.insert(port_id, Uuid::new_v4());
        }
    }
}

pub struct FinalizeGraph<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for FinalizeGraph<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("finalize graph failed".to_string()))?;

        let mut inner = state.a.shared.lock().unwrap();
        inner.finalized = false;
        state.propagation_down_inputs.clear();
        state.propagation_up_inputs.clear();
        state.propagation_up_outputs.clear();
        state.propagation_pairs.clear();
        state.p1_step_tx.clear();
        state.p2_step_tx.clear();
        state.p1_step_rx.clear();
        state.p2_step_rx.clear();
        state.node_p1_completed.clear();
        state.node_p2_completed.clear();
        state.root_p1_sent.clear();
        state.root_p2_sent.clear();
        state.pipeline_completions = 0;
        state.pipeline_target_completions = 0;
        state.sink_id = None;

        if GRADIENT_ACCUMULATION_STEPS == 0 {
            return Err(Failure::Message(
                "gradient accumulation steps must be at least 1".to_string(),
            ));
        }

        if !inner.duplicate_ports.is_empty() {
            let mut ports: Vec<_> = inner.duplicate_ports.iter().copied().collect();
            ports.sort_unstable();
            return Err(Failure::Message(format!(
                "duplicate input port ownership: {ports:?}"
            )));
        }

        let vertices: HashSet<u32> = inner.journey_ids.keys().copied().collect();
        if vertices.is_empty() {
            return Err(Failure::Message(
                "sun graph must contain at least one vertex".to_string(),
            ));
        }

        let mut producer_counts: HashMap<u32, usize> = inner
            .port_vertices
            .keys()
            .copied()
            .map(|port_id| (port_id, 0))
            .collect();
        let mut outgoing: HashMap<u32, Vec<super::PortTarget>> = vertices
            .iter()
            .copied()
            .map(|vertex_id| (vertex_id, Vec::new()))
            .collect();
        let mut incoming: HashMap<u32, Vec<u32>> = vertices
            .iter()
            .copied()
            .map(|vertex_id| (vertex_id, Vec::new()))
            .collect();

        for (&source_vertex, output_ports) in &inner.declared_outputs {
            let source_contract = inner.node_contracts.get(&source_vertex).ok_or_else(|| {
                Failure::Message(format!(
                    "vertex {source_vertex} did not register an operation contract"
                ))
            })?;
            let declared_edges = inner
                .declared_edges
                .get(&source_vertex)
                .cloned()
                .unwrap_or_default();

            for &port_id in output_ports {
                let Some(&target_vertex) = inner.port_vertices.get(&port_id) else {
                    return Err(Failure::Message(format!(
                        "output from vertex {source_vertex} targets missing port {port_id}"
                    )));
                };

                let edge = declared_edges
                    .iter()
                    .find(|edge| edge.port_id == port_id)
                    .ok_or_else(|| {
                        Failure::Message(format!(
                            "output from vertex {source_vertex} to port {port_id} has no contract descriptor"
                        ))
                    })?;
                let destination_contract =
                    inner.node_contracts.get(&target_vertex).ok_or_else(|| {
                        Failure::Message(format!(
                        "destination vertex {target_vertex} did not register an operation contract"
                    ))
                    })?;
                if &edge.source_contract != source_contract {
                    return Err(Failure::Message(format!(
                        "source contract mismatch for edge {source_vertex} -> port {port_id}"
                    )));
                }
                if &edge.destination_contract != destination_contract {
                    return Err(Failure::Message(format!(
                        "destination contract mismatch for edge {source_vertex} -> port {port_id}"
                    )));
                }
                if source_contract.outputs != destination_contract.inputs {
                    return Err(Failure::Message(format!(
                        "artifact bundle mismatch for edge {source_vertex} -> port {port_id}"
                    )));
                }

                let producer_count = producer_counts
                    .get_mut(&port_id)
                    .expect("resolved port should have a producer counter");
                *producer_count += 1;

                outgoing
                    .entry(source_vertex)
                    .or_default()
                    .push(super::PortTarget {
                        port_id,
                        vertex_id: target_vertex,
                    });
                incoming
                    .entry(target_vertex)
                    .or_default()
                    .push(source_vertex);
            }
        }

        for (&port_id, &producer_count) in &producer_counts {
            if producer_count > 1 {
                return Err(Failure::Message(format!(
                    "input port {port_id} has {producer_count} producers; expected at most one"
                )));
            }
        }

        for (&vertex_id, ports) in &inner.vertex_ports {
            let counts: Vec<_> = ports
                .iter()
                .map(|port_id| producer_counts.get(port_id).copied().unwrap_or(0))
                .collect();
            let is_root = counts.iter().all(|count| *count == 0);
            let is_fully_connected = counts.iter().all(|count| *count == 1);
            if !is_root && !is_fully_connected {
                return Err(Failure::Message(format!(
                    "vertex {vertex_id} has incorrect producer counts for ports {ports:?}: {counts:?}"
                )));
            }
        }

        let mut in_degree: HashMap<u32, usize> = incoming
            .iter()
            .map(|(&vertex_id, sources)| (vertex_id, sources.len()))
            .collect();
        let mut roots: Vec<_> = in_degree
            .iter()
            .filter_map(|(&vertex_id, &degree)| (degree == 0).then_some(vertex_id))
            .collect();
        roots.sort_unstable();
        let mut queue: VecDeque<_> = roots.into();
        let mut visited = 0;

        while let Some(vertex_id) = queue.pop_front() {
            visited += 1;
            if let Some(targets) = outgoing.get(&vertex_id) {
                for target in targets {
                    let degree = in_degree
                        .get_mut(&target.vertex_id)
                        .expect("resolved target should have an in-degree");
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(target.vertex_id);
                    }
                }
            }
        }

        if visited != vertices.len() {
            return Err(Failure::Message("sun graph contains a cycle".to_string()));
        }

        let mut sinks: Vec<_> = vertices
            .iter()
            .copied()
            .filter(|vertex_id| outgoing.get(vertex_id).is_none_or(Vec::is_empty))
            .collect();
        sinks.sort_unstable();
        if sinks.len() != 1 {
            return Err(Failure::Message(format!(
                "sun graph must contain exactly one sink; found {sinks:?}"
            )));
        }
        state.sink_id = sinks.first().copied();

        inner.incoming = incoming;
        inner.outgoing = outgoing;
        inner.grad_steps = GRADIENT_ACCUMULATION_STEPS;
        inner.active_micro_step = 0;
        for node_id in vertex_ids(&inner) {
            inner
                .node_grad_steps
                .insert(node_id, super::default_gradient_accumulation_steps());
        }
        inner.po_tx.clear();
        for port_id in port_ids(&inner) {
            inner.po_tx.insert(port_id, Uuid::new_v4());
        }
        reset_epoch_mailboxes(&mut inner);
        inner.finalized = true;

        Ok(())
    }
}

/// Compatibility alias for the former mailbox-only graph setup action.
pub type BuildAddrs<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1> =
    FinalizeGraph<S, GRADIENT_ACCUMULATION_STEPS>;

// ---------------------------------------------------------------------------
// SendRootPropagation — seed every root before waiting for ready output
// ---------------------------------------------------------------------------

pub struct SendRootPropagation<S>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for SendRootPropagation<S>
where
    S: PropagationState,
{
    type Effect = SendRootPropagationEffect;
    type Input = Transmission;
    type Output = Transmission;
    type Carry = Transmission;

    fn emit(state: &S, input: Self::Input) -> (SendRootPropagationInput, Transmission) {
        let carry = input.clone();
        let inner = state.get_shared().lock().unwrap();
        let (input_map, next_input_map, output_map) = S::transmission_maps(&inner);
        let target = |port_id| {
            let node_id = *inner.port_vertices.get(&port_id)?;
            Some(PropagationTarget {
                node_id,
                port_id,
                input_id: *input_map.get(&port_id)?,
                next_input_id: *next_input_map.get(&port_id)?,
                output_id: *output_map.get(&node_id)?,
            })
        };

        let mut targets = inner
            .vertex_ports
            .iter()
            .filter(|(node_id, _)| {
                inner
                    .incoming
                    .get(node_id)
                    .is_none_or(|sources| sources.is_empty())
            })
            .flat_map(|(_, ports)| ports.iter().copied())
            .filter_map(target)
            .collect::<Vec<_>>();
        targets.sort_by_key(|target| (target.node_id, target.port_id));

        (
            SendRootPropagationInput {
                targets,
                transmission: input,
            },
            carry,
        )
    }

    fn absorb(
        state: &mut S,
        output: EffectCompletion<Self::Effect>,
        carry: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let sent_node_ids =
            output.map_err(|e| Failure::Message(format!("send root propagation failed: {e}")))?;
        state
            .get_shared()
            .lock()
            .unwrap()
            .record_propagation_sent(sent_node_ids, S::PROPAGATION_STATE);
        Ok(carry)
    }
}

// ---------------------------------------------------------------------------
// ProcessNextNode — wait for a ready node, then advance the frontier
// ---------------------------------------------------------------------------

/// Action that processes whichever ready node completes first.
///
/// Waits for a [`Transmission::Propagation`] on any of the rx endpoints for
/// nodes with no unresolved predecessors (using the branch-specific rx map).
/// The [`WaitForNodeTransmissionEffect`] effect forwards the received transmission
/// to that node's downstream ports. The completed node is then removed and
/// its successors' unresolved predecessor counts are decremented, making each
/// successor eligible immediately when its count reaches zero.
pub struct ProcessNextNode<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessNextNode<S>
where
    S: PropagationState,
{
    type Effect = WaitForNodeTransmissionEffect;
    type Input = Transmission;
    type Output = Transmission;

    fn emit(state: &S, _input: Self::Input) -> WaitForNodeTransmissionInput {
        let ready = sorted_node_ids(state.ready());
        let inner = state.get_shared().lock().unwrap();
        let outgoing = &inner.outgoing;

        // Each branch writes to the cell's current inbox, tells the cell which
        // inbox to use next, and waits at a dedicated output mailbox.
        let (input_map, next_input_map, output_map) = S::transmission_maps(&inner);

        let target = |port_id| {
            let node_id = *inner.port_vertices.get(&port_id)?;
            Some(PropagationTarget {
                node_id,
                port_id,
                input_id: *input_map.get(&port_id)?,
                next_input_id: *next_input_map.get(&port_id)?,
                output_id: *output_map.get(&node_id)?,
            })
        };

        // Parent-side mailboxes where vertices publish their completed emissions.
        let rx_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = ready
            .iter()
            .filter_map(|&node_id| output_map.get(&node_id).map(|rx| (node_id, *rx)))
            .collect();

        // A completed vertex emission is forwarded to every declared
        // destination port with that port's next mailbox and its vertex's
        // shared output mailbox attached.
        let mut downstream: HashMap<u32, Vec<PropagationTarget>> = HashMap::new();
        for &node_id in &ready {
            if let Some(targets) = outgoing.get(&node_id) {
                let targets_with_endpoints: Vec<_> = targets
                    .iter()
                    .filter_map(|target_port| target(target_port.port_id))
                    .collect();
                downstream.insert(node_id, targets_with_endpoints);
            }
        }

        WaitForNodeTransmissionInput {
            rx_endpoints,
            downstream,
        }
    }

    fn absorb(
        state: &mut S,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let node_tx = output
            .map_err(|e| Failure::Message(format!("wait for node transmission failed: {e}")))?;

        let node_id = node_tx.node_id;
        let outgoing = state
            .get_shared()
            .lock()
            .unwrap()
            .outgoing
            .get(&node_id)
            .cloned()
            .unwrap_or_default();

        let (pending, ready) = state.scheduler_mut();
        advance_frontier(pending, ready, node_id, &outgoing)?;

        {
            let mut inner = state.get_shared().lock().unwrap();
            inner.record_propagation_completed(node_id, S::PROPAGATION_STATE);
            inner.record_propagation_sent(node_tx.sent_node_ids, S::PROPAGATION_STATE);
        }

        Ok(node_tx.transmission)
    }
}

// ---------------------------------------------------------------------------
// GenUuid
// ---------------------------------------------------------------------------

/// Generates the initial inbox used to seed one spawned cell journey.
pub struct GenUuid<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for GenUuid<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = GenUuidEffect;
    type Input = ();
    type Output = crate::cell::action::Init;

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}
    fn absorb(
        _state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let recv_id =
            output.map_err(|_e| Failure::Message("failed to generate a uuid...".to_string()))?;
        Ok(crate::cell::action::Init {
            recv_id,
            grad_steps: GRADIENT_ACCUMULATION_STEPS.max(1),
        })
    }
}

/// Generates the two independent initial inboxes for a binary vertex.
pub struct GenFusionSeed<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for GenFusionSeed<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = GenFusionSeedEffect;
    type Input = ();
    type Output = FusionSeed;

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let mut seed =
            output.map_err(|_| Failure::Message("failed to generate fusion seed".to_string()))?;
        seed.grad_steps = GRADIENT_ACCUMULATION_STEPS.max(1);
        Ok(seed)
    }
}

/// Clears collected propagation outputs and resets accumulation counters.
pub struct BeginGradientAccumulation<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for BeginGradientAccumulation<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("begin sun accumulation failed".to_string()))?;
        state.propagation_down_inputs.clear();
        state.propagation_up_inputs.clear();
        state.propagation_up_outputs.clear();
        state.propagation_pairs.clear();
        state.p1_step_tx.clear();
        state.p2_step_tx.clear();
        state.p1_step_rx.clear();
        state.p2_step_rx.clear();
        state.node_p1_completed.clear();
        state.node_p2_completed.clear();
        state.root_p1_sent.clear();
        state.root_p2_sent.clear();
        state.pipeline_completions = 0;
        state.pipeline_target_completions = 0;
        state.sink_id = None;
        let mut inner = state.a.shared.lock().unwrap();
        inner.active_micro_step = 0;
        Ok(())
    }
}

/// Stores one generated `(up, down)` root-propagation pair.
pub struct StorePropagationInputPair<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action(carry = (Transmission, Transmission))]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for StorePropagationInputPair<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = (Transmission, Transmission);
    type Output = ();

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> ((), (Transmission, Transmission)) {
        ((), input)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: (Transmission, Transmission),
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("store propagation input pair failed".to_string()))?;
        let (up, down) = carry;
        state.propagation_up_inputs.push(up);
        state.propagation_down_inputs.push_back(down);
        Ok(())
    }
}

/// Allocates per-step Sun mailboxes and resets per-node progress tracking.
pub struct PreparePropagationPipeline<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for PreparePropagationPipeline<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("prepare propagation pipeline failed".to_string()))?;
        let expected_steps = GRADIENT_ACCUMULATION_STEPS.max(1);
        if state.propagation_up_inputs.len() != expected_steps {
            return Err(Failure::Message(format!(
                "expected {expected_steps} generated up propagations, got {}",
                state.propagation_up_inputs.len()
            )));
        }
        if state.propagation_down_inputs.len() != expected_steps {
            return Err(Failure::Message(format!(
                "expected {expected_steps} generated down propagations, got {}",
                state.propagation_down_inputs.len()
            )));
        }
        state.propagation_up_outputs.clear();
        state.propagation_pairs.clear();

        let (grad_steps, ports, nodes, roots, sink_id, initial_p1_tx) = {
            let inner = state.a.shared.lock().unwrap();
            let grad_steps = inner.grad_steps.max(1);
            let mut ports = port_ids(&inner);
            ports.sort_unstable();
            let mut nodes = vertex_ids(&inner);
            nodes.sort_unstable();
            let roots = root_vertex_ids(&inner);
            let mut sinks: Vec<_> = nodes
                .iter()
                .copied()
                .filter(|node_id| inner.outgoing.get(node_id).is_none_or(Vec::is_empty))
                .collect();
            sinks.sort_unstable();
            let sink_id = sinks.first().copied().ok_or_else(|| {
                Failure::Message("sun graph has no sink after finalization".to_string())
            })?;
            Ok::<_, Failure>((
                grad_steps,
                ports,
                nodes,
                roots,
                sink_id,
                inner.p1_tx.clone(),
            ))
        }?;

        if grad_steps != expected_steps {
            return Err(Failure::Message(format!(
                "Sun configured for {grad_steps} grad steps but flow expects {expected_steps}"
            )));
        }

        state.p1_step_tx = vec![HashMap::new(); grad_steps];
        state.p2_step_tx = vec![HashMap::new(); grad_steps];
        state.p1_step_rx = vec![HashMap::new(); grad_steps];
        state.p2_step_rx = vec![HashMap::new(); grad_steps];

        for port_id in ports {
            let initial = initial_p1_tx.get(&port_id).copied().ok_or_else(|| {
                Failure::Message(format!("missing initial p1 mailbox for port {port_id}"))
            })?;
            state.p1_step_tx[0].insert(port_id, initial);
            for step in 1..grad_steps {
                state.p1_step_tx[step].insert(port_id, Uuid::new_v4());
            }
            for step in 0..grad_steps {
                state.p2_step_tx[step].insert(port_id, Uuid::new_v4());
            }
        }
        for node_id in &nodes {
            for step in 0..grad_steps {
                state.p1_step_rx[step].insert(*node_id, Uuid::new_v4());
                state.p2_step_rx[step].insert(*node_id, Uuid::new_v4());
            }
        }

        state.node_p1_completed = nodes.iter().copied().map(|node_id| (node_id, 0)).collect();
        state.node_p2_completed = nodes.iter().copied().map(|node_id| (node_id, 0)).collect();
        state.root_p1_sent = roots.iter().copied().map(|node_id| (node_id, 0)).collect();
        state.root_p2_sent = roots.iter().copied().map(|node_id| (node_id, 0)).collect();
        state.pipeline_completions = 0;
        state.pipeline_target_completions = nodes.len() * grad_steps * 2;
        state.sink_id = Some(sink_id);

        let mut inner = state.a.shared.lock().unwrap();
        inner.active_micro_step = 0;
        Ok(())
    }
}

/// Seeds any ready root tasks that have not been sent yet.
pub struct SendReadyRootTasks<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for SendReadyRootTasks<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = SendRootTaskPropagationsEffect;
    type Input = ();
    type Output = ();

    fn emit(state: &super::SunState<S>, _input: Self::Input) -> Vec<RootPropagationSend> {
        let inner = state.a.shared.lock().unwrap();
        let grad_steps = inner.grad_steps.max(1);
        let roots = root_vertex_ids(&inner);
        let mut sends = Vec::new();

        for root_id in roots {
            let Some((phase, step)) = task_for_node(state, root_id, grad_steps) else {
                continue;
            };
            let already_sent = match phase {
                super::SunNodeState::Propagation1 => state
                    .root_p1_sent
                    .get(&root_id)
                    .copied()
                    .unwrap_or_default(),
                super::SunNodeState::Propagation2 => state
                    .root_p2_sent
                    .get(&root_id)
                    .copied()
                    .unwrap_or_default(),
                _ => 0,
            };
            if already_sent >= step {
                continue;
            }

            let transmission = match phase {
                super::SunNodeState::Propagation1 => state
                    .propagation_up_inputs
                    .get(step - 1)
                    .cloned()
                    .expect("missing generated up propagation for root step"),
                super::SunNodeState::Propagation2 => state
                    .propagation_down_inputs
                    .get(step - 1)
                    .cloned()
                    .expect("missing generated down propagation for root step"),
                _ => continue,
            };

            let ports = inner
                .vertex_ports
                .get(&root_id)
                .cloned()
                .unwrap_or_default();
            for port_id in ports {
                sends.push(RootPropagationSend {
                    target: step_target(state, &inner, phase, step, port_id),
                    transmission: transmission.clone(),
                });
            }
        }

        sends
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let sent_node_ids =
            output.map_err(|e| Failure::Message(format!("send ready roots failed: {e}")))?;
        if sent_node_ids.is_empty() {
            return Ok(());
        }

        let grad_steps = {
            let inner = state.a.shared.lock().unwrap();
            inner.grad_steps.max(1)
        };

        let mut updates = Vec::new();
        for node_id in sent_node_ids {
            let Some((phase, step)) = task_for_node(state, node_id, grad_steps) else {
                continue;
            };
            match phase {
                super::SunNodeState::Propagation1 => {
                    state.root_p1_sent.insert(node_id, step);
                }
                super::SunNodeState::Propagation2 => {
                    state.root_p2_sent.insert(node_id, step);
                }
                _ => {}
            }
            updates.push((node_id, phase, step));
        }

        let mut inner = state.a.shared.lock().unwrap();
        for (node_id, phase, step) in updates {
            inner.active_micro_step = step.saturating_sub(1);
            inner.record_propagation_sent([node_id], phase);
        }
        Ok(())
    }
}

/// Waits for the next ready node task completion and advances pipeline progress.
pub struct ProcessReadyPipelineNode<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for ProcessReadyPipelineNode<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = WaitForNodeTransmissionEffect;
    type Input = ();
    type Output = ();

    fn emit(state: &super::SunState<S>, _input: Self::Input) -> WaitForNodeTransmissionInput {
        let inner = state.a.shared.lock().unwrap();
        let grad_steps = inner.grad_steps.max(1);
        let ready = ready_nodes(state, &inner);
        let mut rx_endpoints = Vec::new();
        let mut downstream = HashMap::<u32, Vec<PropagationTarget>>::new();

        for node_id in ready {
            let Some((phase, step)) = task_for_node(state, node_id, grad_steps) else {
                continue;
            };
            if let Some(output_id) = task_output_id(state, node_id, phase, step) {
                rx_endpoints.push((node_id, output_id));
            }
            let targets = inner.outgoing.get(&node_id).cloned().unwrap_or_default();
            if targets.is_empty() {
                continue;
            }
            let mapped: Vec<_> = targets
                .into_iter()
                .map(|target| step_target(state, &inner, phase, step, target.port_id))
                .collect();
            downstream.insert(node_id, mapped);
        }

        WaitForNodeTransmissionInput {
            rx_endpoints,
            downstream,
        }
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let node_tx = output
            .map_err(|e| Failure::Message(format!("wait for ready pipeline node failed: {e}")))?;
        let node_id = node_tx.node_id;

        let (phase, step) = {
            let inner = state.a.shared.lock().unwrap();
            let grad_steps = inner.grad_steps.max(1);
            task_for_node(state, node_id, grad_steps).ok_or_else(|| {
                Failure::Message(format!(
                    "completed node {node_id} has no active pipeline task"
                ))
            })?
        };

        match phase {
            super::SunNodeState::Propagation1 => {
                let completed = state.node_p1_completed.get_mut(&node_id).ok_or_else(|| {
                    Failure::Message(format!("missing p1 counter for node {node_id}"))
                })?;
                *completed = completed.saturating_add(1);
            }
            super::SunNodeState::Propagation2 => {
                let completed = state.node_p2_completed.get_mut(&node_id).ok_or_else(|| {
                    Failure::Message(format!("missing p2 counter for node {node_id}"))
                })?;
                *completed = completed.saturating_add(1);
            }
            _ => {
                return Err(Failure::Message(format!(
                    "invalid phase for completed node {node_id}: {phase:?}"
                )));
            }
        }
        state.pipeline_completions = state.pipeline_completions.saturating_add(1);

        if Some(node_id) == state.sink_id {
            match phase {
                super::SunNodeState::Propagation1 => {
                    state
                        .propagation_up_outputs
                        .push(node_tx.transmission.clone());
                }
                super::SunNodeState::Propagation2 => {
                    let up = state.propagation_up_outputs.get(step - 1).cloned().ok_or_else(|| {
                        Failure::Message(format!(
                            "missing first-pass sink output for step {step} before second-pass output"
                        ))
                    })?;
                    state
                        .propagation_pairs
                        .push((up, node_tx.transmission.clone()));
                }
                _ => {}
            }
        }

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.active_micro_step = step.saturating_sub(1);
            inner.record_propagation_completed(node_id, phase);
            inner.record_propagation_sent(node_tx.sent_node_ids, phase);
        }

        Ok(())
    }
}

/// Stores one generated second-pass root transmission and returns first-pass input.
pub struct StoreDownPropagationInput<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action(carry = (Transmission, Transmission))]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for StoreDownPropagationInput<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = (Transmission, Transmission);
    type Output = Transmission;

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> ((), (Transmission, Transmission)) {
        ((), input)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: (Transmission, Transmission),
    ) -> Result<Transmission, Failure> {
        output.map_err(|_| Failure::Message("store down propagation input failed".to_string()))?;
        let (up, down) = carry;
        state.propagation_down_inputs.push_back(down);
        Ok(up)
    }
}

/// Records one first-pass propagation output and rotates first-pass mailboxes.
pub struct RecordUpPropagationOutput<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for RecordUpPropagationOutput<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = Transmission;
    type Output = ();
    type Carry = Transmission;

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> ((), Transmission) {
        ((), input)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: Transmission,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("record up propagation output failed".to_string()))?;
        state.propagation_up_outputs.push(carry);
        let completed_steps = state.propagation_up_outputs.len();
        let mut inner = state.a.shared.lock().unwrap();
        let grad_steps = inner.grad_steps.max(1);
        if completed_steps < grad_steps {
            prepare_next_up_microstep(&mut inner, completed_steps);
        }
        Ok(())
    }
}

/// Switches Sun propagation bookkeeping from first-pass to second-pass phase.
pub struct BeginDownPropagationPhase<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for BeginDownPropagationPhase<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("begin down propagation phase failed".to_string()))?;
        let expected_steps = GRADIENT_ACCUMULATION_STEPS.max(1);
        if state.propagation_down_inputs.len() != expected_steps {
            return Err(Failure::Message(format!(
                "expected {expected_steps} queued down inputs, got {}",
                state.propagation_down_inputs.len()
            )));
        }
        if state.propagation_up_outputs.len() != expected_steps {
            return Err(Failure::Message(format!(
                "expected {expected_steps} first-pass outputs, got {}",
                state.propagation_up_outputs.len()
            )));
        }
        let mut inner = state.a.shared.lock().unwrap();
        inner.active_micro_step = 0;
        Ok(())
    }
}

/// Pops the next generated second-pass root transmission.
pub struct NextDownPropagationInput<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for NextDownPropagationInput<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = Transmission;

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("next down propagation input failed".to_string()))?;
        state
            .propagation_down_inputs
            .pop_front()
            .ok_or_else(|| Failure::Message("missing queued down propagation input".to_string()))
    }
}

/// Records one second-pass output and builds one policy input pair.
pub struct RecordDownPropagationOutput<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for RecordDownPropagationOutput<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = Transmission;
    type Output = ();
    type Carry = Transmission;

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> ((), Transmission) {
        ((), input)
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
        carry: Transmission,
    ) -> Result<Self::Output, Failure> {
        output
            .map_err(|_| Failure::Message("record down propagation output failed".to_string()))?;
        let pair_index = state.propagation_pairs.len();
        let up = state
            .propagation_up_outputs
            .get(pair_index)
            .cloned()
            .ok_or_else(|| {
                Failure::Message(format!(
                    "missing first-pass output for pair index {pair_index}"
                ))
            })?;
        state.propagation_pairs.push((up, carry));

        let completed_steps = state.propagation_pairs.len();
        let mut inner = state.a.shared.lock().unwrap();
        let grad_steps = inner.grad_steps.max(1);
        if completed_steps < grad_steps {
            prepare_next_down_microstep(&mut inner, completed_steps);
        }
        Ok(())
    }
}

/// Converts collected propagation pairs into the policy input array.
pub struct CollectedPropagationPairs<S = (), const GRADIENT_ACCUMULATION_STEPS: usize = 1>(
    PhantomData<fn() -> S>,
);

#[jungle::action]
impl<S, const GRADIENT_ACCUMULATION_STEPS: usize> Action
    for CollectedPropagationPairs<S, GRADIENT_ACCUMULATION_STEPS>
{
    type Effect = NoEffect;
    type Input = ();
    type Output = [(Transmission, Transmission); GRADIENT_ACCUMULATION_STEPS];

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("collect propagation pairs failed".to_string()))?;
        state.propagation_pairs.clone().try_into().map_err(
            |pairs: Vec<(Transmission, Transmission)>| {
                Failure::Message(format!(
                    "expected {GRADIENT_ACCUMULATION_STEPS} propagation pairs, got {}",
                    pairs.len()
                ))
            },
        )
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiation — broadcast potentiation payload to all nodes
// ---------------------------------------------------------------------------

/// Broadcasts matching potentiation envelopes to every input port.
///
/// Unary vertices receive one envelope and binary vertices receive one per
/// independent port. Each envelope assigns that port a fresh first-pass inbox.
pub struct BroadcastPotentiation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for BroadcastPotentiation<S> {
    type Effect = BroadcastPotentiationEffect;
    type Input = black_hole_spec::Potentiation;
    type Output = ();
    type Carry = ();

    fn emit(state: &super::SunState<S>, input: Self::Input) -> BroadcastPotentiationInput {
        let inner = state.a.shared.lock().unwrap();
        let mut port_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = inner
            .port_vertices
            .keys()
            .filter_map(|&port_id| inner.po_tx.get(&port_id).map(|tx| (port_id, *tx)))
            .collect();
        port_endpoints.sort_by_key(|(port_id, _)| *port_id);
        drop(inner);

        BroadcastPotentiationInput {
            potentiation: input,
            port_endpoints,
        }
    }

    fn absorb(
        state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let result =
            output.map_err(|e| Failure::Message(format!("broadcast potentiation failed: {e}")))?;

        let mut inner = state.a.shared.lock().unwrap();
        let optimized_node_ids = result
            .next_p1_tx_map
            .iter()
            .filter_map(|(port_id, _)| inner.port_vertices.get(port_id).copied())
            .collect::<HashSet<_>>();
        inner.p1_tx.clear();
        for (port_id, next_p1_tx) in &result.next_p1_tx_map {
            inner.p1_tx.insert(*port_id, *next_p1_tx);
        }
        inner.po_tx.clear();
        for port_id in port_ids(&inner) {
            inner.po_tx.insert(port_id, Uuid::new_v4());
        }
        inner.active_micro_step = 0;
        reset_epoch_mailboxes(&mut inner);
        inner.record_optimization_sent(optimized_node_ids);
        drop(inner);
        state.propagation_down_inputs.clear();
        state.propagation_up_inputs.clear();
        state.propagation_up_outputs.clear();
        state.propagation_pairs.clear();
        state.p1_step_tx.clear();
        state.p2_step_tx.clear();
        state.p1_step_rx.clear();
        state.p2_step_rx.clear();
        state.node_p1_completed.clear();
        state.node_p2_completed.clear();
        state.root_p1_sent.clear();
        state.root_p2_sent.clear();
        state.pipeline_completions = 0;
        state.pipeline_target_completions = 0;
        state.sink_id = None;

        Ok(())
    }
}

/// Input for the [`BroadcastPotentiation`] effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationInput {
    pub potentiation: black_hole_spec::Potentiation,
    /// (port_id, potentiation inbox) pairs.
    pub port_endpoints: Vec<(u32, ObjectId)>,
}

// ---------------------------------------------------------------------------
// PropagationState — trait for branch state types (PropA, PropB)
// ---------------------------------------------------------------------------

/// Trait that provides accessors for the propagation-related fields
/// shared by [`PropA`](super::PropA) and [`PropB`](super::PropB).
pub trait PropagationState {
    /// Appearance phase corresponding to this propagation branch.
    const PROPAGATION_STATE: super::SunNodeState;

    /// Access the shared inner state.
    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>>;

    /// Select this branch's current input, next input, and output mailboxes.
    fn transmission_maps(
        inner: &super::SunInner,
    ) -> (
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
    );

    /// Unfinished nodes and the number of incoming edges whose source has not
    /// completed yet.
    fn pending(&self) -> &HashMap<u32, usize>;

    /// Nodes that are eligible to be processed now.
    fn ready(&self) -> &HashSet<u32>;

    /// Mutable access to both scheduler collections.
    fn scheduler_mut(&mut self) -> (&mut HashMap<u32, usize>, &mut HashSet<u32>);
}

impl PropagationState for super::PropA {
    const PROPAGATION_STATE: super::SunNodeState = super::SunNodeState::Propagation1;

    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>> {
        &self.shared
    }
    fn transmission_maps(
        inner: &super::SunInner,
    ) -> (
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
    ) {
        let next_map = if inner.active_micro_step + 1 < inner.grad_steps.max(1) {
            &inner.next_p1_tx
        } else {
            &inner.p2_tx
        };
        (&inner.p1_tx, next_map, &inner.p1_rx)
    }
    fn pending(&self) -> &HashMap<u32, usize> {
        &self.pending
    }
    fn ready(&self) -> &HashSet<u32> {
        &self.ready
    }
    fn scheduler_mut(&mut self) -> (&mut HashMap<u32, usize>, &mut HashSet<u32>) {
        (&mut self.pending, &mut self.ready)
    }
}

impl PropagationState for super::PropB {
    const PROPAGATION_STATE: super::SunNodeState = super::SunNodeState::Propagation2;

    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>> {
        &self.shared
    }
    fn transmission_maps(
        inner: &super::SunInner,
    ) -> (
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
        &HashMap<u32, ObjectId>,
    ) {
        let next_map = if inner.active_micro_step + 1 < inner.grad_steps.max(1) {
            &inner.next_p2_tx
        } else {
            &inner.po_tx
        };
        (&inner.p2_tx, next_map, &inner.p2_rx)
    }
    fn pending(&self) -> &HashMap<u32, usize> {
        &self.pending
    }
    fn ready(&self) -> &HashSet<u32> {
        &self.ready
    }
    fn scheduler_mut(&mut self) -> (&mut HashMap<u32, usize>, &mut HashSet<u32>) {
        (&mut self.pending, &mut self.ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jungle_sdk::Id;
    use typenum::{U0, U1, U2};

    struct GenericType<T>(std::marker::PhantomData<T>);

    struct TestSunAnimal;

    impl Animal for TestSunAnimal {
        type Id = ();
        type Generation = ();
        type State = super::super::SunState;
        type Seed = ();
        type Flow = ();
    }

    struct TestSunAnimalWithPayload;

    impl Animal for TestSunAnimalWithPayload {
        type Id = Id<U0>;
        type Generation = U0;
        type State = super::super::SunState<(String, String)>;
        type Seed = ();
        type Flow = ();
    }

    struct TestUnaryChildAnimal;

    impl Animal for TestUnaryChildAnimal {
        type Id = Id<U1>;
        type Generation = U0;
        type State = crate::CellState;
        type Seed = crate::cell::action::Init;
        type Flow = crate::Primordium;
    }

    struct TestFusionChildAnimal;

    impl Animal for TestFusionChildAnimal {
        type Id = Id<U2>;
        type Generation = U0;
        type State = crate::FusionState;
        type Seed = crate::FusionSeed;
        type Flow = crate::Fusion<crate::Primordium>;
    }

    struct TestWarpChildAnimal;

    impl Animal for TestWarpChildAnimal {
        type Id = Id<U1>;
        type Generation = U0;
        type State = super::super::SunState;
        type Seed = ();
        type Flow = ();
    }

    impl Observe for TestWarpChildAnimal {
        type Appearance = crate::Ray;

        fn observe(_state: &Self::State) -> Self::Appearance {
            crate::Ray { frozen: false }
        }
    }

    struct TestWarpBoundaryAnimal;

    impl Animal for TestWarpBoundaryAnimal {
        type Id = Id<U2>;
        type Generation = U0;
        type State = crate::BoundaryState<<TestWarpChildAnimal as Observe>::Appearance>;
        type Seed = super::super::BoundaryInit;
        type Flow = ();
    }

    fn add_vertex(
        state: &mut super::super::SunState,
        vertex_id: u32,
        port_ids: &[u32],
        outputs: &[u32],
    ) {
        let ports: Vec<_> = port_ids
            .iter()
            .map(|&port_id| (port_id, Uuid::new_v4()))
            .collect();
        register_vertex(
            state,
            vertex_id,
            format!("Node{vertex_id}"),
            &ports,
            QwenDarkInference::descriptor(),
            outputs
                .iter()
                .map(|&port_id| super::super::DeclaredEdge {
                    port_id,
                    source_contract: QwenDarkInference::descriptor(),
                    destination_contract: QwenDarkInference::descriptor(),
                })
                .collect(),
            Uuid::new_v4(),
            None,
        );
    }

    fn finalize(state: &mut super::super::SunState) -> Result<(), Failure> {
        type Bound = <FinalizeGraph<(), 1> as Action>::Bind<TestSunAnimal>;
        <Bound as BoundAction<TestSunAnimal>>::absorb(state, Ok(()))
    }

    fn finalize_with_steps<const STEPS: usize>(
        state: &mut super::super::SunState,
    ) -> Result<(), Failure> {
        type Bound<const N: usize> = <FinalizeGraph<(), N> as Action>::Bind<TestSunAnimal>;
        <Bound<STEPS> as BoundAction<TestSunAnimal>>::absorb(state, Ok(()))
    }

    fn propagation(seed: u128) -> Transmission {
        Transmission::Propagation {
            emission_id: black_hole_spec::EmissionId::new(Uuid::from_u128(seed)),
            recv: Uuid::new_v4(),
            send: Uuid::new_v4(),
        }
    }

    fn prepare_pipeline_with_steps<const STEPS: usize>(
        state: &mut super::super::SunState,
    ) -> Result<(), Failure> {
        state.propagation_up_inputs = (0..STEPS)
            .map(|step| propagation(100 + step as u128))
            .collect();
        state.propagation_down_inputs = (0..STEPS)
            .map(|step| propagation(200 + step as u128))
            .collect();
        type Bound<const N: usize> =
            <PreparePropagationPipeline<(), N> as Action>::Bind<TestSunAnimal>;
        <Bound<STEPS> as BoundAction<TestSunAnimal>>::absorb(state, Ok(()))
    }

    #[test]
    fn sun_state_supports_custom_inner_payload() {
        let mut state = super::super::SunState::<(String, String)>::default();
        state.inner.0.push_str("left");
        state.inner.1.push_str("right");
        assert_eq!(state.inner, ("left".to_string(), "right".to_string()));
    }

    #[test]
    fn sun_actions_bind_with_custom_state_payload() {
        type Payload = (String, String);

        let state = super::super::SunState::<Payload>::default();

        type GenUuidBound = <GenUuid<Payload, 1> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenUuidBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type GenFusionSeedBound =
            <GenFusionSeed<Payload, 1> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenFusionSeedBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type FinalizeBound = <FinalizeGraph<Payload, 1> as Action>::Bind<TestSunAnimalWithPayload>;
        <FinalizeBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type BroadcastBound =
            <BroadcastPotentiation<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <BroadcastBound as BoundAction<TestSunAnimalWithPayload>>::emit(
            &state,
            black_hole_spec::Potentiation {
                loss_up: 0.1,
                loss_down: 0.2,
                seed: 7,
            },
        );

        type SpawnUnaryBound =
            <SpawnUnary<U1, TestUnaryChildAnimal, Empty, Payload> as Action>::Bind<
                TestSunAnimalWithPayload,
            >;
        let seed = crate::cell::action::Init {
            recv_id: Uuid::new_v4(),
            grad_steps: 1,
        };
        let effect_seed =
            <SpawnUnaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, seed);
        assert_eq!(effect_seed, seed);

        type SpawnBinaryBound =
            <SpawnBinary<U1, U2, TestFusionChildAnimal, Empty, Payload> as Action>::Bind<
                TestSunAnimalWithPayload,
            >;
        let seed = crate::FusionSeed {
            p1_recv_id: Uuid::new_v4(),
            p2_recv_id: Uuid::new_v4(),
            grad_steps: 1,
        };
        let effect_seed =
            <SpawnBinaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, seed);
        assert_eq!(effect_seed.p1_recv_id, seed.p1_recv_id);
        assert_eq!(effect_seed.p2_recv_id, seed.p2_recv_id);
    }

    #[test]
    fn warp_actions_seed_boundary_with_spawned_warp_journey() {
        type Payload = (String, String);
        let mut state = super::super::SunState::<Payload>::default();
        let init = crate::cell::action::Init {
            recv_id: Uuid::new_v4(),
            grad_steps: 3,
        };

        type SpawnWarpAnimalBound = <SpawnWarpAnimal<
            U1,
            TestWarpChildAnimal,
            TestWarpBoundaryAnimal,
            Empty,
            Payload,
        > as Action>::Bind<TestSunAnimalWithPayload>;
        let spawn_input =
            <SpawnWarpAnimalBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, init);
        assert_eq!(spawn_input, ());

        let warp_journey_id = Uuid::new_v4();
        let (boundary_init, boundary_carry) = <SpawnWarpAnimalBound as BoundAction<
            TestSunAnimalWithPayload,
        >>::absorb_with_carry(
            &mut state, Ok(warp_journey_id), init
        )
        .unwrap();
        assert_eq!(boundary_init, init);
        assert_eq!(boundary_carry, warp_journey_id);

        type SpawnWarpBoundaryBound = <SpawnWarpBoundary<
            U1,
            TestWarpChildAnimal,
            TestWarpBoundaryAnimal,
            Empty,
            Payload,
        > as Action>::Bind<TestSunAnimalWithPayload>;
        let boundary_seed = <SpawnWarpBoundaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(
            &state,
            (boundary_init, boundary_carry),
        );
        assert_eq!(boundary_seed.recv_id, init.recv_id);
        assert_eq!(boundary_seed.grad_steps, init.grad_steps);
        assert_eq!(boundary_seed.warp_journey_id, warp_journey_id);

        let boundary_journey_id = Uuid::new_v4();
        <SpawnWarpBoundaryBound as BoundAction<TestSunAnimalWithPayload>>::absorb_with_carry(
            &mut state,
            Ok(boundary_journey_id),
            (boundary_init, boundary_carry),
        )
        .unwrap();

        let inner = state.a.shared.lock().unwrap();
        assert_eq!(inner.journey_ids.get(&U1::U32), Some(&boundary_journey_id));
        assert_eq!(inner.warp_journey_ids.get(&U1::U32), Some(&warp_journey_id));
        assert_eq!(inner.port_vertices.get(&U1::U32), Some(&U1::U32));
    }

    #[test]
    fn short_type_name_preserves_generic_arguments() {
        type Nested = GenericType<Result<String, Vec<u8>>>;
        assert_eq!(
            short_type_name::<Nested>(),
            "GenericType<Result<String, Vec<u8>>>"
        );
        assert_eq!(
            short_type_label("my::module::Animal<crate::Inner<leaf::Type>, alloc::vec::Vec<u8>>"),
            "Animal<Inner<Type>, Vec<u8>>"
        );
    }

    #[test]
    fn finalization_rejects_duplicate_port_ownership() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[]);
        add_vertex(&mut state, 1, &[0], &[]);

        let error = finalize(&mut state).unwrap_err();
        assert!(error.to_string().contains("duplicate input port ownership"));
    }

    #[test]
    fn finalization_rejects_missing_output_ports() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[9]);

        let error = finalize(&mut state).unwrap_err();
        assert!(error.to_string().contains("targets missing port 9"));
    }

    #[test]
    fn finalization_rejects_a_destination_contract_changed_after_compilation() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[1]);
        add_vertex(&mut state, 1, &[1], &[]);

        let mut wrong = QwenDarkInference::descriptor();
        wrong.version += 1;
        state
            .a
            .shared
            .lock()
            .unwrap()
            .node_contracts
            .insert(1, wrong);

        let error = finalize(&mut state).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("destination contract mismatch for edge 0 -> port 1"),
            "{error}"
        );
    }

    #[test]
    fn finalization_rejects_partially_connected_binary_vertices() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[1]);
        add_vertex(&mut state, 1, &[1, 2], &[]);

        let error = finalize(&mut state).unwrap_err();
        assert!(error.to_string().contains("incorrect producer counts"));
    }

    #[test]
    fn finalization_rejects_cycles() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[1]);
        add_vertex(&mut state, 1, &[1], &[0]);

        let error = finalize(&mut state).unwrap_err();
        assert!(error.to_string().contains("contains a cycle"));
    }

    #[test]
    fn finalization_rejects_multiple_sinks() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[]);
        add_vertex(&mut state, 1, &[1], &[]);

        let error = finalize(&mut state).unwrap_err();
        assert!(error.to_string().contains("exactly one sink"));
    }

    #[test]
    fn completed_node_immediately_unblocks_deeper_successor() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[1, 2]);
        add_vertex(&mut state, 1, &[1], &[3]);
        add_vertex(&mut state, 2, &[2], &[4]);
        add_vertex(&mut state, 3, &[3], &[5]);
        add_vertex(&mut state, 4, &[4, 5], &[]);
        finalize(&mut state).unwrap();

        let inner = state.a.shared.lock().unwrap();
        let mut pending = pending_dependency_counts(&inner);
        let mut ready = initial_ready_nodes(&pending);
        assert_eq!(sorted_node_ids(&ready), vec![0]);

        advance_frontier(&mut pending, &mut ready, 0, &inner.outgoing[&0]).unwrap();
        assert_eq!(sorted_node_ids(&ready), vec![1, 2]);

        // Node 1 finishes while its same-layer sibling, node 2, is still
        // pending. Its child becomes ready immediately instead of waiting for
        // all of the old topological layer to complete.
        advance_frontier(&mut pending, &mut ready, 1, &inner.outgoing[&1]).unwrap();
        assert_eq!(sorted_node_ids(&ready), vec![2, 3]);
    }

    #[test]
    fn pipeline_scheduler_allows_branch_progress_without_global_step_barrier() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[2]);
        add_vertex(&mut state, 1, &[1], &[3]);
        add_vertex(&mut state, 2, &[2], &[4]);
        add_vertex(&mut state, 3, &[3], &[5]);
        add_vertex(&mut state, 4, &[4, 5], &[]);
        finalize_with_steps::<2>(&mut state).unwrap();
        prepare_pipeline_with_steps::<2>(&mut state).unwrap();

        // Root 0 and its child 2 complete step 1, then root 0 completes step 2
        // while the sibling branch rooted at 1 has not finished step 1 yet.
        state.root_p1_sent.insert(0, 2);
        state.root_p1_sent.insert(1, 1);
        state.node_p1_completed.insert(0, 2);
        state.node_p1_completed.insert(1, 0);
        state.node_p1_completed.insert(2, 1);
        state.node_p1_completed.insert(3, 0);
        state.node_p1_completed.insert(4, 0);

        let inner = state.a.shared.lock().unwrap();
        let ready = ready_nodes(&state, &inner);
        assert!(
            ready.contains(&2),
            "node 2 should be ready for p1 step 2 without waiting for branch rooted at node 1"
        );
    }

    #[test]
    fn appearance_contains_deterministic_port_aware_topology() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 1, &[1], &[3]);
        add_vertex(&mut state, 0, &[0], &[2]);
        add_vertex(&mut state, 2, &[2, 3], &[]);
        finalize(&mut state).unwrap();

        let appearance = state.appearance();
        assert!(appearance.finalized);
        assert_eq!(appearance.grad_steps, 1);
        assert_eq!(
            appearance
                .nodes
                .iter()
                .map(|node| (
                    node.id,
                    node.label.as_str(),
                    node.input_ports.clone(),
                    node.state,
                    node.grad_step,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, "Node0", vec![0], super::super::SunNodeState::Idle, 1),
                (1, "Node1", vec![1], super::super::SunNodeState::Idle, 1),
                (2, "Node2", vec![2, 3], super::super::SunNodeState::Idle, 1),
            ]
        );
        assert!(
            appearance
                .nodes
                .iter()
                .all(|node| node.journey_id != Uuid::nil()),
            "each node should expose its spawned child workflow journey id"
        );
        assert_eq!(
            appearance.edges,
            vec![
                super::super::SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 2,
                },
                super::super::SunEdgeAppearance {
                    source: 1,
                    target: 2,
                    target_port: 3,
                },
            ]
        );
    }

    #[test]
    fn propagation_two_waits_for_propagation_one_completion() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[]);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation2);
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation1);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation1,
            "sending propagation 2 must not hide in-flight propagation 1"
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 1);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_completed(0, super::super::SunNodeState::Propagation1);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation2,
            "propagation 2 becomes visible after propagation 1 emits"
        );
        assert_eq!(
            state.appearance().nodes[0].state_sequence,
            2,
            "each visible propagation phase advances the sequence"
        );

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_optimization_sent([0]);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Optimization
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 3);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation1);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation1
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 4);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation2);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation1,
            "the prior epoch's completion must not expose propagation 2 early"
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 4);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_completed(0, super::super::SunNodeState::Propagation1);
        }
        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation2
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 5);
    }

    #[test]
    fn propagation_two_is_exposed_when_sent_after_propagation_one_completes() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[]);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation1);
            inner.record_propagation_completed(0, super::super::SunNodeState::Propagation1);
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation2);
        }

        assert_eq!(
            state.appearance().nodes[0].state,
            super::super::SunNodeState::Propagation2
        );
        assert_eq!(state.appearance().nodes[0].state_sequence, 2);
    }

    #[test]
    fn appearance_tracks_per_node_gradient_step() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 0, &[0], &[1]);
        add_vertex(&mut state, 1, &[1], &[]);
        finalize_with_steps::<4>(&mut state).unwrap();

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation1);
        }
        let appearance = state.appearance();
        assert_eq!(appearance.grad_steps, 4);
        assert_eq!(
            appearance
                .nodes
                .iter()
                .find(|node| node.id == 0)
                .unwrap()
                .grad_step,
            1
        );
        assert_eq!(
            appearance
                .nodes
                .iter()
                .find(|node| node.id == 1)
                .unwrap()
                .grad_step,
            1
        );

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.active_micro_step = 2;
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation1);
            inner.record_propagation_completed(0, super::super::SunNodeState::Propagation1);
            inner.record_propagation_sent([0], super::super::SunNodeState::Propagation2);
        }
        let appearance = state.appearance();
        let node0 = appearance.nodes.iter().find(|node| node.id == 0).unwrap();
        assert_eq!(node0.state, super::super::SunNodeState::Propagation2);
        assert_eq!(node0.grad_step, 3);

        {
            let mut inner = state.a.shared.lock().unwrap();
            inner.record_optimization_sent([0]);
        }
        let appearance = state.appearance();
        let node0 = appearance.nodes.iter().find(|node| node.id == 0).unwrap();
        assert_eq!(node0.state, super::super::SunNodeState::Optimization);
        assert_eq!(node0.grad_step, 4);
    }
}
