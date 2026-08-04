//! Sun actions — spawning animals, propagation, loss computation, and potentiation.

use std::marker::PhantomData;

use std::collections::HashMap;

use crate::sun::effect::GenUuidEffect;

use super::effect::{
    BroadcastPotentiationEffect, ComputeLossEffect, InitializeEffect, PropagationTarget,
    WaitForLayerTransmission, WaitForLayerTransmissionInput,
};
use super::Tagged;
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
// Spawn — spawn an animal and populate SunState with outgoing edges
// ---------------------------------------------------------------------------

/// Action that spawns the animal described by a [`Unary`](super::Unary) node.
pub struct Spawn<T>(PhantomData<fn() -> T>);

#[jungle::action]
impl<T> Action for Spawn<T>
where
    T: Tagged<A: Animal<Seed = ObjectId>>,
    <T as Tagged>::N: Unsigned,
    <<T as Tagged>::A as Animal>::Id: AnimalIdValue,
    <<T as Tagged>::A as Animal>::Generation: Unsigned,
{
    type Effect = super::effect::SpawnAnimal<<T as Tagged>::A>;
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

        let node_id = <<T as Tagged>::N as Unsigned>::U32;
        let outgoing_node_ids = <<T as Tagged>::E as NodeIdsFromList>::node_ids();

        let mut inner = state.a.shared.lock().unwrap();
        inner.journey_ids.insert(node_id, journey_id);
        inner.p1_tx.insert(node_id, initial_recv_id);
        inner.outgoing.insert(node_id, outgoing_node_ids.clone());
        for target in outgoing_node_ids {
            inner.incoming.entry(target).or_default().push(node_id);
        }

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
        for (_src, targets) in &outgoing {
            for &target in targets {
                if all_nodes.contains(&target) {
                    *in_degree.entry(target).or_insert(0) += 1;
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
                    for &target in targets {
                        if let Some(deg) = remaining.get_mut(&target) {
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
// BuildAddrs — build full set of recv/send addrs for all nodes
// ---------------------------------------------------------------------------

pub struct BuildAddrs;
#[jungle::action]
impl Action for BuildAddrs {
    type Effect = NoEffect;
    type Input = ();
    type Output = ();

    fn emit(state: &super::SunState, _input: Self::Input) {
        let mut inner = state.a.shared.lock().unwrap();
        for node in inner.journey_ids.keys().copied().collect::<Vec<_>>() {
            inner.p1_rx.insert(node, Uuid::new_v4());
            inner.p2_tx.insert(node, Uuid::new_v4());
            inner.p2_rx.insert(node, Uuid::new_v4());
            inner.po_tx.insert(node, Uuid::new_v4());
        }
    }

    fn absorb(
        _state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_e| Failure::Message(format!("Build addrs failed")))
    }
}

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

        let target = |node_id| {
            Some(PropagationTarget {
                node_id,
                input_id: *input_map.get(&node_id)?,
                next_input_id: *next_input_map.get(&node_id)?,
                output_id: *output_map.get(&node_id)?,
            })
        };

        // Parent-side mailboxes where cells publish their completed emissions.
        let rx_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = current
            .iter()
            .filter_map(|&node_id| output_map.get(&node_id).map(|rx| (node_id, *rx)))
            .collect();

        // Root cells receive the epoch's initial transmission directly.
        let root_targets = current
            .iter()
            .copied()
            .filter(|node_id| {
                inner
                    .incoming
                    .get(node_id)
                    .is_none_or(|sources| sources.is_empty())
            })
            .filter_map(target)
            .collect();

        // A completed cell emission is forwarded to each downstream cell's
        // current inbox with that cell's next/output mailboxes attached.
        let mut downstream: HashMap<u32, Vec<PropagationTarget>> = HashMap::new();
        for &node_id in &current {
            if let Some(targets) = outgoing.get(&node_id) {
                let targets_with_endpoints: Vec<_> =
                    targets.iter().copied().filter_map(target).collect();
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

// ---------------------------------------------------------------------------
// Initialize — create the initial transmissions for both propagation passes
// ---------------------------------------------------------------------------

/// Creates one initial transmission for each propagation branch. Root delivery
/// is handled by [`ProcessNode`] once branch-specific mailboxes are available.
pub struct Initialize;

#[jungle::action]
impl Action for Initialize {
    type Effect = InitializeEffect;
    type Input = ();
    type Output = (Transmission, Transmission);

    fn emit(_state: &super::SunState, _input: Self::Input) {}
    fn absorb(
        _state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|_e| Failure::Message("initialize failed".to_string()))
    }
}

// ---------------------------------------------------------------------------
// ComputeLoss — compute (loss_up, loss_down) from branch outputs
// ---------------------------------------------------------------------------

/// Computes the loss values from the completed outputs of both propagation
/// branches.
pub struct ComputeLoss;

#[jungle::action]
impl Action for ComputeLoss {
    type Effect = ComputeLossEffect;
    type Input = (Transmission, Transmission);
    type Output = (f32, f32);
    type Carry = ();

    fn emit(_state: &super::SunState, input: Self::Input) -> (Transmission, Transmission) {
        input
    }

    fn absorb(
        _state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| Failure::Message(format!("compute loss failed: {e}")))
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiation — broadcast losses to all nodes
// ---------------------------------------------------------------------------

/// Action that broadcasts `Transmission::Potentiation` with the computed
/// loss values to all nodes' potentiation endpoints (po_tx).
///
/// Sends losses to each cell's potentiation inbox, assigns each cell a fresh
/// first-pass inbox, and rotates all intermediate mailboxes for the next epoch.
pub struct BroadcastPotentiation;

#[jungle::action]
impl Action for BroadcastPotentiation {
    type Effect = BroadcastPotentiationEffect;
    type Input = (f32, f32);
    type Output = ();
    type Carry = ();

    fn emit(state: &super::SunState, input: Self::Input) -> BroadcastPotentiationInput {
        let inner = state.a.shared.lock().unwrap();
        let node_endpoints: Vec<(u32, black_hole_spec::ObjectId)> = inner
            .journey_ids
            .keys()
            .filter_map(|&node_id| inner.po_tx.get(&node_id).map(|tx| (node_id, *tx)))
            .collect();
        drop(inner);

        BroadcastPotentiationInput {
            loss_up: input.0,
            loss_down: input.1,
            node_endpoints,
        }
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let result =
            output.map_err(|e| Failure::Message(format!("broadcast potentiation failed: {e}")))?;

        let mut inner = state.a.shared.lock().unwrap();
        for (node_id, next_p1_tx) in &result.next_p1_tx_map {
            inner.p1_tx.insert(*node_id, *next_p1_tx);
            inner.p1_rx.insert(*node_id, Uuid::new_v4());
            inner.p2_tx.insert(*node_id, Uuid::new_v4());
            inner.p2_rx.insert(*node_id, Uuid::new_v4());
            inner.po_tx.insert(*node_id, Uuid::new_v4());
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
    /// (node_id, potentiation inbox) pairs.
    pub node_endpoints: Vec<(u32, ObjectId)>,
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

    /// Get a reference to the branch-specific tx map from locked inner state.
    fn get_tx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId>;

    /// Get a mutable reference to the branch-specific tx map from locked inner state.
    fn get_tx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId>;

    /// Get a reference to the branch-specific rx map from locked inner state.
    fn get_rx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId>;

    /// Get a mutable reference to the branch-specific rx map from locked inner state.
    fn get_rx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId>;
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
    fn get_tx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p1_tx
    }
    fn get_tx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p1_tx
    }
    fn get_rx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p1_rx
    }
    fn get_rx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p1_rx
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
    fn get_tx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p2_tx
    }
    fn get_tx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p2_tx
    }
    fn get_rx<'a>(
        &self,
        inner: &'a super::SunInner,
    ) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p2_rx
    }
    fn get_rx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p2_rx
    }
}
