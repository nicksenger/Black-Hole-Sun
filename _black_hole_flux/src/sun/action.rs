//! Sun actions — spawning animals, propagation, and potentiation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use crate::sun::effect::{GenFusionSeedEffect, GenUuidEffect};
use crate::{FusionSeed, FusionState};

use super::effect::{
    BroadcastPotentiationEffect, PropagationTarget, SendRootPropagationEffect,
    SendRootPropagationInput, WaitForNodeTransmission, WaitForNodeTransmissionInput,
};
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

// ---------------------------------------------------------------------------
// Spawn — descriptor-specific animal spawning and graph registration
// ---------------------------------------------------------------------------

fn register_vertex<S>(
    state: &mut super::SunState<S>,
    vertex_id: u32,
    node_label: String,
    ports: &[(u32, ObjectId)],
    declared_outputs: Vec<u32>,
    journey_id: Uuid,
) {
    let mut inner = state.a.shared.lock().unwrap();

    inner.journey_ids.entry(vertex_id).or_insert(journey_id);
    inner.node_labels.entry(vertex_id).or_insert(node_label);
    inner.node_states.entry(vertex_id).or_default();
    inner.node_state_sequences.entry(vertex_id).or_default();
    inner
        .vertex_ports
        .entry(vertex_id)
        .or_insert_with(|| ports.iter().map(|(port_id, _)| *port_id).collect());
    inner
        .declared_outputs
        .entry(vertex_id)
        .or_insert(declared_outputs);

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
    let full = core::any::type_name::<T>();
    let cleaned = strip_type_generics(full);
    cleaned
        .rsplit("::")
        .next()
        .unwrap_or(cleaned.as_str())
        .to_string()
}

fn strip_type_generics(name: &str) -> String {
    let mut depth = 0usize;
    let mut cleaned = String::with_capacity(name.len());

    for ch in name.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => cleaned.push(ch),
            _ => {}
        }
    }

    cleaned.trim().to_string()
}

/// Spawns and registers a [`Unary`](super::Unary) descriptor.
pub struct SpawnUnary<P, A, E, S = ()>(PhantomData<fn() -> (P, A, E, S)>);

#[jungle::action]
impl<P, A, E, S> Action for SpawnUnary<P, A, E, S>
where
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId>,
    E: NodeIdsFromList,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = ObjectId;
    type Output = ();
    type Carry = ObjectId;

    fn emit(_state: &super::SunState<S>, input: Self::Input) -> (ObjectId, ObjectId) {
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
            &[(port_id, initial_recv_id)],
            E::node_ids(),
            journey_id,
        );

        Ok(())
    }
}

/// Backwards-compatible name for the unary spawn action.
pub type Spawn<P, A, E, S = ()> = SpawnUnary<P, A, E, S>;

/// Spawns and registers a [`Binary`](super::Binary) descriptor.
pub struct SpawnBinary<P1, P2, A, E, S = ()>(PhantomData<fn() -> (P1, P2, A, E, S)>);

#[jungle::action]
impl<P1, P2, A, E, S> Action for SpawnBinary<P1, P2, A, E, S>
where
    P1: Unsigned,
    P2: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>,
    A::Flow: crate::fusion::FusionFlow,
    E: NodeIdsFromList,
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
            E::node_ids(),
            journey_id,
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

pub struct FinalizeGraph<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for FinalizeGraph<S> {
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
            for &port_id in output_ports {
                let Some(&target_vertex) = inner.port_vertices.get(&port_id) else {
                    return Err(Failure::Message(format!(
                        "output from vertex {source_vertex} targets missing port {port_id}"
                    )));
                };

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

        inner.incoming = incoming;
        inner.outgoing = outgoing;

        for port_id in inner.port_vertices.keys().copied().collect::<Vec<_>>() {
            inner.p2_tx.insert(port_id, Uuid::new_v4());
            inner.po_tx.insert(port_id, Uuid::new_v4());
        }
        for vertex_id in vertices {
            inner.p1_rx.insert(vertex_id, Uuid::new_v4());
            inner.p2_rx.insert(vertex_id, Uuid::new_v4());
        }
        inner.finalized = true;

        Ok(())
    }
}

/// Compatibility alias for the former mailbox-only graph setup action.
pub type BuildAddrs<S = ()> = FinalizeGraph<S>;

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
/// The [`WaitForNodeTransmission`] effect forwards the received transmission
/// to that node's downstream ports. The completed node is then removed and
/// its successors' unresolved predecessor counts are decremented, making each
/// successor eligible immediately when its count reaches zero.
pub struct ProcessNextNode<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessNextNode<S>
where
    S: PropagationState,
{
    type Effect = WaitForNodeTransmission;
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
pub struct GenUuid<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for GenUuid<S> {
    type Effect = GenUuidEffect;
    type Input = ();
    type Output = Uuid;

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}
    fn absorb(
        _state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_e| Failure::Message("failed to generate a uuid...".to_string()))
    }
}

/// Generates the two independent initial inboxes for a binary vertex.
pub struct GenFusionSeed<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for GenFusionSeed<S> {
    type Effect = GenFusionSeedEffect;
    type Input = ();
    type Output = FusionSeed;

    fn emit(_state: &super::SunState<S>, _input: Self::Input) {}

    fn absorb(
        _state: &mut super::SunState<S>,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("failed to generate fusion seed".to_string()))
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiation — broadcast losses to all nodes
// ---------------------------------------------------------------------------

/// Broadcasts matching potentiation envelopes to every input port.
///
/// Unary vertices receive one envelope and binary vertices receive one per
/// independent port. Each envelope assigns that port a fresh first-pass inbox.
pub struct BroadcastPotentiation<S = ()>(PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for BroadcastPotentiation<S> {
    type Effect = BroadcastPotentiationEffect;
    type Input = (f32, f32);
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
            loss_up: input.0,
            loss_down: input.1,
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
        for (port_id, next_p1_tx) in &result.next_p1_tx_map {
            inner.p1_tx.insert(*port_id, *next_p1_tx);
            inner.p2_tx.insert(*port_id, Uuid::new_v4());
            inner.po_tx.insert(*port_id, Uuid::new_v4());
        }
        for vertex_id in inner.journey_ids.keys().copied().collect::<Vec<_>>() {
            inner.p1_rx.insert(vertex_id, Uuid::new_v4());
            inner.p2_rx.insert(vertex_id, Uuid::new_v4());
        }
        inner.record_optimization_sent(optimized_node_ids);
        drop(inner);

        Ok(())
    }
}

/// Input for the [`BroadcastPotentiation`] effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastPotentiationInput {
    pub loss_up: f32,
    pub loss_down: f32,
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
        (&inner.p1_tx, &inner.p2_tx, &inner.p1_rx)
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
        (&inner.p2_tx, &inner.po_tx, &inner.p2_rx)
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
        type Seed = ObjectId;
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
            outputs.to_vec(),
            Uuid::new_v4(),
        );
    }

    fn finalize(state: &mut super::super::SunState) -> Result<(), Failure> {
        type Bound = <FinalizeGraph as Action>::Bind<TestSunAnimal>;
        <Bound as BoundAction<TestSunAnimal>>::absorb(state, Ok(()))
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

        type GenUuidBound = <GenUuid<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenUuidBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type GenFusionSeedBound =
            <GenFusionSeed<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <GenFusionSeedBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type FinalizeBound = <FinalizeGraph<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <FinalizeBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, ());

        type BroadcastBound =
            <BroadcastPotentiation<Payload> as Action>::Bind<TestSunAnimalWithPayload>;
        <BroadcastBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, (0.1, 0.2));

        type SpawnUnaryBound =
            <SpawnUnary<U1, TestUnaryChildAnimal, Empty, Payload> as Action>::Bind<
                TestSunAnimalWithPayload,
            >;
        let seed = Uuid::new_v4();
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
        };
        let effect_seed =
            <SpawnBinaryBound as BoundAction<TestSunAnimalWithPayload>>::emit(&state, seed);
        assert_eq!(effect_seed.p1_recv_id, seed.p1_recv_id);
        assert_eq!(effect_seed.p2_recv_id, seed.p2_recv_id);
    }

    #[test]
    fn short_type_name_omits_generic_arguments() {
        type Nested = GenericType<Result<String, Vec<u8>>>;
        assert_eq!(short_type_name::<Nested>(), "GenericType");
        assert_eq!(
            strip_type_generics("crate::Animal<module::Inner<leaf::Type>>"),
            "crate::Animal"
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
    fn appearance_contains_deterministic_port_aware_topology() {
        let mut state = super::super::SunState::default();
        add_vertex(&mut state, 1, &[1], &[3]);
        add_vertex(&mut state, 0, &[0], &[2]);
        add_vertex(&mut state, 2, &[2, 3], &[]);
        finalize(&mut state).unwrap();

        let appearance = state.appearance();
        assert!(appearance.finalized);
        assert_eq!(
            appearance
                .nodes
                .iter()
                .map(|node| (
                    node.id,
                    node.label.as_str(),
                    node.input_ports.clone(),
                    node.state
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, "Node0", vec![0], super::super::SunNodeState::Idle),
                (1, "Node1", vec![1], super::super::SunNodeState::Idle),
                (2, "Node2", vec![2, 3], super::super::SunNodeState::Idle),
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
}
