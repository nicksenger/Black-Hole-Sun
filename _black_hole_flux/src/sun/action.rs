//! Sun actions — spawning animals, propagation, loss computation, and potentiation.

use std::marker::PhantomData;

use super::Tagged;
use super::effect::{BroadcastPotentiationEffect, ComputeLossEffect, KickOffEffect, WaitForLayerTransmission};
use black_hole_spec::ObjectId;
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

/// Action that spawns an animal `T` tagged by [`Tag`](super::Tag) into the jungle.
pub struct Spawn<Tag>(PhantomData<fn() -> Tag>);

#[jungle::action]
impl<T> Action for Spawn<T>
where
    T: Tagged,
    <T as Tagged>::N: Unsigned,
    <<T as Tagged>::A as Animal>::Id: AnimalIdValue,
    <<T as Tagged>::A as Animal>::Generation: Unsigned,
    <<T as Tagged>::A as Animal>::Seed: Sync + Send + 'static,
{
    type Effect = super::effect::SpawnAnimal<<T as Tagged>::A>;
    type Input = <<T as Tagged>::A as Animal>::Seed;
    type Output = Uuid;
    type Carry = ();

    fn emit(_state: &super::SunState, input: Self::Input) -> <<T as Tagged>::A as Animal>::Seed {
        input
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let journey_id = output.map_err(|e| Failure::Message(format!("spawn failed: {e}")))?;

        let node_id = <<T as Tagged>::N as Unsigned>::U32;
        let outgoing_node_ids = <<T as Tagged>::E as NodeIdsFromList>::node_ids();

        let mut inner = state.a.shared.lock().unwrap();
        inner.journey_ids.insert(node_id, journey_id);
        inner.outgoing.insert(node_id, outgoing_node_ids.clone());
        for target in outgoing_node_ids {
            inner.incoming.entry(target).or_default().push(node_id);
        }

        Ok(journey_id)
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
    type Input = ();
    type Output = ();

    fn emit(_state: &S, _input: Self::Input) -> () {
        ()
    }

    fn absorb(
        state: &mut S,
        _output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let (all_nodes, outgoing) = {
            let inner = state.get_shared().lock().unwrap();
            let all_nodes: std::collections::HashSet<u32> =
                inner.journey_ids.keys().cloned().collect();
            let outgoing = inner.outgoing.clone();
            (all_nodes, outgoing)
        };

        let mut in_degree: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
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

        Ok(())
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
    type Input = ();
    type Output = ();

    fn emit(_state: &S, _input: Self::Input) -> () {
        ()
    }

    fn absorb(
        state: &mut S,
        _output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let topo = state.get_topo().clone();
        let layer = topo.first().cloned().ok_or_else(|| {
            Failure::Message("no topological layers remaining".to_string())
        })?;

        let mut topo = state.get_topo().clone();
        topo.drain(..1);
        state.set_topo(topo);
        state.set_current(layer);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProcessNode — wait for transmission, forward to outgoing nodes
// ---------------------------------------------------------------------------

/// Action that processes a single node in the current topological layer.
///
/// Waits for a [`Transmission::Propagation`] on any of the rx endpoints for
/// nodes in the current layer (using the branch-specific rx map). On receiving
/// a transmission, generates new tx ids for each outgoing edge and stores them
/// in the branch-specific tx map.
pub struct ProcessNode<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action]
impl<S> Action for ProcessNode<S>
where
    S: TopologyState,
{
    type Effect = WaitForLayerTransmission;
    type Input = ();
    type Output = ();
    type Carry = super::effect::LayerTransmission;

    fn emit(
        state: &S,
        _input: Self::Input,
    ) -> (Vec<(u32, black_hole_spec::ObjectId)>, super::effect::LayerTransmission) {
        let current = state.get_current().clone();
        let inner = state.get_shared().lock().unwrap();

        let endpoints: Vec<(u32, black_hole_spec::ObjectId)> = current
            .iter()
            .filter_map(|&node_id| {
                state.get_rx(&inner).get(&node_id).map(|rx| (node_id, *rx))
            })
            .collect();

        let dummy = super::effect::LayerTransmission {
            node_id: 0,
            transmission: black_hole_spec::Transmission::Propagation {
                emission_id: black_hole_spec::EmissionId(black_hole_spec::ObjectId::nil()),
                recv: black_hole_spec::ObjectId::nil(),
                send: black_hole_spec::ObjectId::nil(),
            },
        };

        (endpoints, dummy)
    }

    fn absorb(
        state: &mut S,
        output: EffectCompletion<Self::Effect>,
        _carry: super::effect::LayerTransmission,
    ) -> Result<Self::Output, Failure> {
        let layer_tx = output.map_err(|e| {
            Failure::Message(format!("wait for layer transmission failed: {e}"))
        })?;

        let node_id = layer_tx.node_id;
        let transmission = layer_tx.transmission;

        let outgoing_nodes = {
            let inner = state.get_shared().lock().unwrap();
            inner.outgoing.get(&node_id).cloned().unwrap_or_default()
        };

        let new_rx = uuid::Uuid::new_v4();

        {
            let mut inner = state.get_shared().lock().unwrap();
            state.get_rx_mut(&mut inner).insert(node_id, new_rx);

            for target_id in &outgoing_nodes {
                let tx_id = uuid::Uuid::new_v4();

                match &transmission {
                    black_hole_spec::Transmission::Propagation { .. } => {}
                    other => {
                        return Err(Failure::Message(format!(
                            "expected Propagation for forwarding from node {}, got {:?}",
                            node_id, other
                        )));
                    }
                }

                state.get_tx_mut(&mut inner).insert(*target_id, tx_id);

                tracing::debug!(
                    node_id, target_id, ?tx_id,
                    "forwarded transmission to outgoing node"
                );
            }
        }

        let mut current = state.get_current().clone();
        current.remove(&node_id);
        state.set_current(current);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KickOff — generate initial TransmissionId and send to first-layer nodes
// ---------------------------------------------------------------------------

/// Action that kicks off propagation by generating a TransmissionId and
/// sending it to the rx endpoints of all nodes in the first topological layer
/// (those with no incoming edges / dependencies).
///
/// Takes unit input, finds root nodes from shared state, generates a new
/// TransmissionId stored in shared state, and uploads Propagation transmissions
/// to each root node's rx endpoint. Outputs unit — the transmission id is stored
/// in shared state for ComputeLoss to retrieve later.
pub struct KickOff;

#[jungle::action]
impl Action for KickOff {
    type Effect = KickOffEffect;
    type Input = ();
    type Output = ();
    type Carry = ();

    fn emit(state: &super::SunState, _input: Self::Input) -> Vec<u32> {
        let inner = state.a.shared.lock().unwrap();
        let all_nodes: std::collections::HashSet<u32> =
            inner.journey_ids.keys().cloned().collect();
        let incoming = inner.incoming.clone();
        drop(inner);

        let roots: Vec<u32> = all_nodes
            .into_iter()
            .filter(|&node| incoming.get(&node).map_or(true, |v| v.is_empty()))
            .collect();

        roots
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let result = output.map_err(|e| {
            Failure::Message(format!("kick-off failed: {e}"))
        })?;

        // Store the initial rx ids and transmission id in shared state
        let mut inner = state.a.shared.lock().unwrap();
        inner.transmission_id = result.transmission_id;
        for (node_id, rx_id) in &result.rx_map {
            inner.p1_rx.insert(*node_id, *rx_id);
            inner.p2_rx.insert(*node_id, *rx_id);
        }
        drop(inner);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ComputeLoss — compute (loss_up, loss_down) from shared TransmissionId
// ---------------------------------------------------------------------------

/// Action that retrieves the TransmissionId stored by KickOff in shared state,
/// and computes the loss values (loss_up, loss_down) for potentiation.
pub struct ComputeLoss;

#[jungle::action]
impl Action for ComputeLoss {
    type Effect = ComputeLossEffect;
    type Input = ();
    type Output = (f32, f32);
    type Carry = ();

    fn emit(state: &super::SunState, _input: Self::Input) -> ObjectId {
        let inner = state.a.shared.lock().unwrap();
        inner.transmission_id
    }

    fn absorb(
        _state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        output.map_err(|e| {
            Failure::Message(format!("compute loss failed: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// BroadcastPotentiation — broadcast losses to all nodes
// ---------------------------------------------------------------------------

/// Action that broadcasts `Transmission::Potentiation` with the computed
/// loss values to all nodes' potentiation endpoints (po_tx).
///
/// Takes (loss_up, loss_down) as input, generates a new recv ObjectId for
/// each node, uploads the Potentiation transmission to each po_tx endpoint,
/// and updates the rx maps so the next epoch can begin. Exits without waiting
/// for any response transmission.
pub struct BroadcastPotentiation;

#[jungle::action]
impl Action for BroadcastPotentiation {
    type Effect = BroadcastPotentiationEffect;
    type Input = (f32, f32);
    type Output = ();
    type Carry = ();

    fn emit(state: &super::SunState, input: Self::Input) -> BroadcastPotentiationInput {
        let inner = state.a.shared.lock().unwrap();
        let node_ids: Vec<u32> = inner.journey_ids.keys().cloned().collect();
        drop(inner);

        BroadcastPotentiationInput {
            loss_up: input.0,
            loss_down: input.1,
            node_ids,
        }
    }

    fn absorb(
        state: &mut super::SunState,
        output: EffectCompletion<Self::Effect>,
    ) -> Result<Self::Output, Failure> {
        let result = output.map_err(|e| {
            Failure::Message(format!("broadcast potentiation failed: {e}"))
        })?;

        let mut inner = state.a.shared.lock().unwrap();
        for (node_id, new_rx) in &result.new_rx_map {
            inner.p1_rx.insert(*node_id, *new_rx);
            inner.p2_rx.insert(*node_id, *new_rx);
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
    pub node_ids: Vec<u32>,
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
    fn get_tx<'a>(&self, inner: &'a super::SunInner) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p1_tx
    }
    fn get_tx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p1_tx
    }
    fn get_rx<'a>(&self, inner: &'a super::SunInner) -> &'a std::collections::HashMap<u32, ObjectId> {
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
    fn get_tx<'a>(&self, inner: &'a super::SunInner) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p2_tx
    }
    fn get_tx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p2_tx
    }
    fn get_rx<'a>(&self, inner: &'a super::SunInner) -> &'a std::collections::HashMap<u32, ObjectId> {
        &inner.p2_rx
    }
    fn get_rx_mut<'a>(
        &self,
        inner: &'a mut super::SunInner,
    ) -> &'a mut std::collections::HashMap<u32, ObjectId> {
        &mut inner.p2_rx
    }
}
