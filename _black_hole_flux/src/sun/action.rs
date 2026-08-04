//! Sun actions — spawning animals, propagation, and potentiation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

use crate::sun::effect::{GenFusionSeedEffect, GenUuidEffect};
use crate::{FusionSeed, FusionState};

use super::effect::{
    BroadcastPotentiationEffect, PropagationTarget, WaitForLayerTransmission,
    WaitForLayerTransmissionInput,
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

fn register_vertex(
    state: &mut super::SunState,
    vertex_id: u32,
    ports: &[(u32, ObjectId)],
    declared_outputs: Vec<u32>,
    journey_id: Uuid,
) {
    let mut inner = state.a.shared.lock().unwrap();

    inner.journey_ids.entry(vertex_id).or_insert(journey_id);
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

/// Spawns and registers a [`Unary`](super::Unary) descriptor.
pub struct SpawnUnary<P, A, E>(PhantomData<fn() -> (P, A, E)>);

#[jungle::action]
impl<P, A, E> Action for SpawnUnary<P, A, E>
where
    P: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId>,
    E: NodeIdsFromList,
{
    type Effect = super::effect::SpawnAnimal<A>;
    type Input = ObjectId;
    type Output = ();
    type Carry = ObjectId;

    fn emit(_state: &super::SunState, input: Self::Input) -> (ObjectId, ObjectId) {
        (input, input)
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
        initial_recv_id: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let port_id = P::U32;
        register_vertex(
            state,
            port_id,
            &[(port_id, initial_recv_id)],
            E::node_ids(),
            journey_id,
        );

        Ok(())
    }
}

/// Backwards-compatible name for the unary spawn action.
pub type Spawn<P, A, E> = SpawnUnary<P, A, E>;

/// Spawns and registers a [`Binary`](super::Binary) descriptor.
pub struct SpawnBinary<P1, P2, A, E>(PhantomData<fn() -> (P1, P2, A, E)>);

#[jungle::action]
impl<P1, P2, A, E> Action for SpawnBinary<P1, P2, A, E>
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

    fn emit(_state: &super::SunState, seed: Self::Input) -> (FusionSeed, FusionSeed) {
        (seed, seed)
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
        seed: Self::Carry,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let p1 = P1::U32;
        let p2 = P2::U32;
        register_vertex(
            state,
            p1,
            &[(p1, seed.p1_recv_id), (p2, seed.p2_recv_id)],
            E::node_ids(),
            journey_id,
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BuildTopologicalSort — build topological layers using Kahn's algorithm
// ---------------------------------------------------------------------------

pub struct BuildTopologicalSort<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for BuildTopologicalSort<S>
where
    S: TopologyState,
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
        let (all_nodes, outgoing) = {
            let inner = state.get_shared().lock().unwrap();
            let all_nodes: std::collections::HashSet<u32> =
                inner.journey_ids.keys().cloned().collect();
            let outgoing = inner.outgoing.clone();
            (all_nodes, outgoing)
        };

        let mut in_degree: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &node in &all_nodes {
            in_degree.entry(node).or_insert(0);
        }
        for targets in outgoing.values() {
            for target in targets {
                if all_nodes.contains(&target.vertex_id) {
                    *in_degree.entry(target.vertex_id).or_insert(0) += 1;
                }
            }
        }

        let mut topo: Vec<std::collections::HashSet<u32>> = Vec::new();
        let mut remaining = in_degree.clone();

        loop {
            let layer: std::collections::HashSet<u32> = remaining
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&node, _)| node)
                .collect();

            if layer.is_empty() {
                break;
            }

            topo.push(layer.clone());

            for node in &layer {
                remaining.remove(node);
                if let Some(targets) = outgoing.get(node) {
                    for target in targets {
                        if let Some(deg) = remaining.get_mut(&target.vertex_id) {
                            *deg -= 1;
                        }
                    }
                }
            }
        }

        state.set_topo(topo);
        state.set_current(std::collections::HashSet::new());

        Ok(carry)
    }
}

// ---------------------------------------------------------------------------
// FinalizeGraph — resolve ports, validate the DAG, and allocate phase mailboxes
// ---------------------------------------------------------------------------

pub struct FinalizeGraph;

#[jungle::action]
impl Action for FinalizeGraph {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(_state: &super::SunState, _input: Self::Input) {}

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_| Failure::Message("finalize graph failed".to_string()))?;

        let mut inner = state.a.shared.lock().unwrap();

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

        Ok(())
    }
}

/// Compatibility alias for the former mailbox-only graph setup action.
pub type BuildAddrs = FinalizeGraph;

// ---------------------------------------------------------------------------
// PopLayer — pop the next topological layer into current
// ---------------------------------------------------------------------------

pub struct PopLayer<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for PopLayer<S>
where
    S: TopologyState,
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
        carry: Transmission,
    ) -> Result<Self::Output, Failure> {
        let topo = state.get_topo().clone();
        let layer = topo
            .first()
            .cloned()
            .ok_or_else(|| Failure::Message("no topological layers remaining".to_string()))?;

        let mut topo = state.get_topo().clone();
        topo.drain(..1);
        state.set_topo(topo);
        state.set_current(layer);

        Ok(carry)
    }
}

// ---------------------------------------------------------------------------
// ProcessNode — wait for transmission, forward to outgoing nodes
// ---------------------------------------------------------------------------

/// Action that processes a single node in the current topological layer.
///
/// Waits for a [`Transmission::Propagation`] on any of the rx endpoints for
/// nodes in the current layer (using the branch-specific rx map). The
/// [`WaitForLayerTransmission`] effect handles forwarding the received
/// transmission to the rx endpoints of downstream nodes, so propagation
/// continues through the graph. After receiving, removes the processed node
/// from the current layer.
pub struct ProcessNode<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessNode<S>
where
    S: TopologyState + 'static,
{
    type Effect = WaitForLayerTransmission;
    type Input = Transmission;
    type Output = Transmission;

    fn emit(state: &S, input: Self::Input) -> WaitForLayerTransmissionInput {
        let current = state.get_current().clone();
        let inner = state.get_shared().lock().unwrap();
        let outgoing = &inner.outgoing;

        // Each branch writes to the cell's current inbox, tells the cell which
        // inbox to use next, and waits at a dedicated output mailbox.
        let (input_map, next_input_map, output_map): (
            &HashMap<u32, black_hole_spec::ObjectId>,
            &HashMap<u32, black_hole_spec::ObjectId>,
            &HashMap<u32, black_hole_spec::ObjectId>,
        ) = {
            if std::any::TypeId::of::<S>() == std::any::TypeId::of::<super::PropA>() {
                (&inner.p1_tx, &inner.p2_tx, &inner.p1_rx)
            } else {
                (&inner.p2_tx, &inner.po_tx, &inner.p2_rx)
            }
        };

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
        let rx_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = current
            .iter()
            .filter_map(|&node_id| output_map.get(&node_id).map(|rx| (node_id, *rx)))
            .collect();

        // Every input port on a root vertex receives the initial transmission.
        let root_targets = current
            .iter()
            .copied()
            .filter(|node_id| {
                inner
                    .incoming
                    .get(node_id)
                    .is_none_or(|sources| sources.is_empty())
            })
            .flat_map(|node_id| {
                inner
                    .vertex_ports
                    .get(&node_id)
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .filter_map(&target)
            .collect();

        // A completed vertex emission is forwarded to every declared
        // destination port with that port's next mailbox and its vertex's
        // shared output mailbox attached.
        let mut downstream: HashMap<u32, Vec<PropagationTarget>> = HashMap::new();
        for &node_id in &current {
            if let Some(targets) = outgoing.get(&node_id) {
                let targets_with_endpoints: Vec<_> = targets
                    .iter()
                    .filter_map(|target_port| target(target_port.port_id))
                    .collect();
                downstream.insert(node_id, targets_with_endpoints);
            }
        }

        WaitForLayerTransmissionInput {
            rx_endpoints,
            root_targets,
            downstream,
            input_transmission: input,
        }
    }

    fn absorb(
        state: &mut S,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let layer_tx = output
            .map_err(|e| Failure::Message(format!("wait for layer transmission failed: {e}")))?;

        let node_id = layer_tx.node_id;

        // Remove the processed node from the current layer.
        let mut current = state.get_current().clone();
        current.remove(&node_id);
        state.set_current(current);

        Ok(layer_tx.transmission)
    }
}

// ---------------------------------------------------------------------------
// GenUuid
// ---------------------------------------------------------------------------

/// Generates the initial inbox used to seed one spawned cell journey.
pub struct GenUuid;

#[jungle::action]
impl Action for GenUuid {
    type Effect = GenUuidEffect;
    type Input = ();
    type Output = Uuid;

    fn emit(_state: &super::SunState, _input: Self::Input) {}
    fn absorb(
        _state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_e| Failure::Message("failed to generate a uuid...".to_string()))
    }
}

/// Generates the two independent initial inboxes for a binary vertex.
pub struct GenFusionSeed;

#[jungle::action]
impl Action for GenFusionSeed {
    type Effect = GenFusionSeedEffect;
    type Input = ();
    type Output = FusionSeed;

    fn emit(_state: &super::SunState, _input: Self::Input) {}

    fn absorb(
        _state: &mut super::SunState,
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
pub struct BroadcastPotentiation;

#[jungle::action]
impl Action for BroadcastPotentiation {
    type Effect = BroadcastPotentiationEffect;
    type Input = (f32, f32);
    type Output = ();
    type Carry = ();

    fn emit(state: &super::SunState, input: Self::Input) -> BroadcastPotentiationInput {
        let inner = state.a.shared.lock().unwrap();
        let port_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = inner
            .port_vertices
            .keys()
            .filter_map(|&port_id| inner.po_tx.get(&port_id).map(|tx| (port_id, *tx)))
            .collect();
        drop(inner);

        BroadcastPotentiationInput {
            loss_up: input.0,
            loss_down: input.1,
            port_endpoints,
        }
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let result =
            output.map_err(|e| Failure::Message(format!("broadcast potentiation failed: {e}")))?;

        let mut inner = state.a.shared.lock().unwrap();
        for (port_id, next_p1_tx) in &result.next_p1_tx_map {
            inner.p1_tx.insert(*port_id, *next_p1_tx);
            inner.p2_tx.insert(*port_id, Uuid::new_v4());
            inner.po_tx.insert(*port_id, Uuid::new_v4());
        }
        for vertex_id in inner.journey_ids.keys().copied().collect::<Vec<_>>() {
            inner.p1_rx.insert(vertex_id, Uuid::new_v4());
            inner.p2_rx.insert(vertex_id, Uuid::new_v4());
        }
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
// TopologyState — trait for branch state types (PropA, PropB)
// ---------------------------------------------------------------------------

/// Trait that provides accessors for the topology-related fields
/// shared by [`PropA`](super::PropA) and [`PropB`](super::PropB).
pub trait TopologyState {
    /// Access the shared inner state.
    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>>;

    /// Get a reference to the topological layers.
    fn get_topo(&self) -> &Vec<std::collections::HashSet<u32>>;

    /// Set the topological layers.
    fn set_topo(&mut self, topo: Vec<std::collections::HashSet<u32>>);

    /// Get a reference to the current layer being processed.
    fn get_current(&self) -> &std::collections::HashSet<u32>;

    /// Set the current layer being processed.
    fn set_current(&mut self, current: std::collections::HashSet<u32>);
}

impl TopologyState for super::PropA {
    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>> {
        &self.shared
    }
    fn get_topo(&self) -> &Vec<std::collections::HashSet<u32>> {
        &self.topo
    }
    fn set_topo(&mut self, topo: Vec<std::collections::HashSet<u32>>) {
        self.topo = topo;
    }
    fn get_current(&self) -> &std::collections::HashSet<u32> {
        &self.current
    }
    fn set_current(&mut self, current: std::collections::HashSet<u32>) {
        self.current = current;
    }
}

impl TopologyState for super::PropB {
    fn get_shared(&self) -> &std::sync::Arc<std::sync::Mutex<super::SunInner>> {
        &self.shared
    }
    fn get_topo(&self) -> &Vec<std::collections::HashSet<u32>> {
        &self.topo
    }
    fn set_topo(&mut self, topo: Vec<std::collections::HashSet<u32>>) {
        self.topo = topo;
    }
    fn get_current(&self) -> &std::collections::HashSet<u32> {
        &self.current
    }
    fn set_current(&mut self, current: std::collections::HashSet<u32>) {
        self.current = current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSunAnimal;

    impl Animal for TestSunAnimal {
        type Id = ();
        type Generation = ();
        type State = super::super::SunState;
        type Seed = ();
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
        register_vertex(state, vertex_id, &ports, outputs.to_vec(), Uuid::new_v4());
    }

    fn finalize(state: &mut super::super::SunState) -> Result<(), Failure> {
        type Bound = <FinalizeGraph as Action>::Bind<TestSunAnimal>;
        <Bound as BoundAction<TestSunAnimal>>::absorb(state, Ok(()))
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
}
