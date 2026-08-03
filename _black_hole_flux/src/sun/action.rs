//! Sun actions — spawning animals and populating sun state.

use std::marker::PhantomData;

use super::Tagged;
use super::effect::WaitForLayerTransmission;
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
///
/// Takes the animal's seed as input, spawns it via [`SpawnAnimal`](super::effect::SpawnAnimal)
/// effect, receives the journey UUID, then populates the [`SunState`](super::SunState)
/// outgoing map with directed edges from this node to each outgoing node ID
/// derived from the type-level list `E`.
///
/// Returns the journey UUID for downstream use.
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

        // Lock the inner struct and register this node + its outgoing edges
        let mut inner = state.propagation.a.shared.lock().unwrap();

        // Store the journey ID for this node
        inner.journey_ids.insert(node_id, journey_id);

        // Register outgoing edges: this node -> each outgoing node
        inner.outgoing.insert(node_id, outgoing_node_ids.clone());

        // Register each outgoing node with this node as an incoming edge
        for target in outgoing_node_ids {
            inner.incoming.entry(target).or_default().push(node_id);
        }

        Ok(journey_id)
    }
}

// ---------------------------------------------------------------------------
// BuildTopologicalSort — build topological layers using Kahn's algorithm
// ---------------------------------------------------------------------------

/// Action that builds the topological layer ordering for a branch.
///
/// Uses Kahn's algorithm: starts with nodes having no incoming edges (within
/// the branch), assigns them to the first layer, then removes those nodes'
/// outgoing edges and repeats until all nodes are layered.
///
/// The layers are stored outermost-first in `state.topo`, so popping from the
/// back of the vec gives the innermost layer last (outer-to-inner processing).
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
        // Clone data from shared state while holding the lock
        let (all_nodes, outgoing) = {
            let inner = state.get_shared().lock().unwrap();
            let all_nodes: std::collections::HashSet<u32> =
                inner.journey_ids.keys().cloned().collect();
            let outgoing = inner.outgoing.clone();
            (all_nodes, outgoing)
        };
        // Lock is now dropped

        // Build in-degree map (only count edges within the known nodes)
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

        // Kahn's algorithm — build layers outermost to innermost
        let mut topo: Vec<std::collections::HashSet<u32>> = Vec::new();
        let mut remaining = in_degree.clone();

        loop {
            // Find all nodes with in-degree 0
            let layer: std::collections::HashSet<u32> = remaining
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&node, _)| node)
                .collect();

            if layer.is_empty() {
                // If there are still nodes remaining, there's a cycle
                if !remaining.is_empty() {
                    return Err(Failure::Message(
                        "cycle detected in graph topology".to_string(),
                    ));
                }
                break;
            }

            topo.push(layer.clone());

            // Remove this layer and decrease in-degree of their targets
            for &node in &layer {
                remaining.remove(&node);
                if let Some(targets) = outgoing.get(&node) {
                    for &target in targets {
                        if let Some(deg) = remaining.get_mut(&target) {
                            *deg = deg.saturating_sub(1);
                        }
                    }
                }
            }
        }

        // Store the layers (outermost first, so pop from back for innermost last)
        state.set_topo(topo);
        state.set_current(std::collections::HashSet::new());

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PopLayer — pop the outermost layer into current
// ---------------------------------------------------------------------------

/// Action that pops the next layer from `state.topo` into `state.current`.
///
/// Pops from the front (index 0) to process outermost layers first.
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
        let topo = state.get_topo();
        if topo.is_empty() {
            return Err(Failure::Message(
                "no layers to pop".to_string(),
            ));
        }

        // Pop the first layer (outermost) and shift remaining
        let mut topo = topo.clone();
        let current = topo.remove(0);

        state.set_topo(topo);
        state.set_current(current);

        Ok(())
    }
}


// ---------------------------------------------------------------------------
// ProcessNode — wait for transmission, advance node, remove from current
// ---------------------------------------------------------------------------

/// Action that processes one node from the current layer:
/// 1. Collects rx endpoints for all nodes in the current layer
/// 2. Waits for the first available transmission (via WaitForLayerTransmission)
/// 3. Advances the received node — updates rx, forwards tx to outgoing nodes
/// 4. Removes the processed node from current
pub struct ProcessNode<S>(std::marker::PhantomData<fn() -> S>);

#[jungle::action(carry = super::effect::LayerTransmission)]
impl<S> Action for ProcessNode<S>
where
    S: TopologyState,
{
    type Effect = WaitForLayerTransmission;
    type Input = ();
    type Output = ();

    fn emit(
        state: &S,
        _input: Self::Input,
    ) -> (Vec<(u32, black_hole_spec::ObjectId)>, super::effect::LayerTransmission) {
        // Collect rx endpoints for all nodes in the current layer
        let current = state.get_current();
        let inner = state.get_shared().lock().unwrap();

        let endpoints: Vec<(u32, black_hole_spec::ObjectId)> = current
            .iter()
            .filter_map(|&node_id| {
                inner.rx.get(&node_id).map(|rx| (node_id, *rx))
            })
            .collect();

        // Dummy carry — the actual value comes from effect output via absorb
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

        // Get outgoing nodes for this node
        let inner = state.get_shared().lock().unwrap();
        let outgoing_nodes = inner.outgoing.get(&node_id).cloned().unwrap_or_default();
        drop(inner);

        // Generate new rx id for this node
        let new_rx = uuid::Uuid::new_v4();

        // Update the rx entry for this node in shared state
        {
            let mut inner = state.get_shared().lock().unwrap();
            inner.rx.insert(node_id, new_rx);
        }

        // For each outgoing node, generate a new tx id and upload the transmission
        for target_id in &outgoing_nodes {
            let tx_id = uuid::Uuid::new_v4();

            // Create a Propagation transmission for the outgoing node
            let _propagation = match &transmission {
                black_hole_spec::Transmission::Propagation { emission_id, .. } => {
                    black_hole_spec::Transmission::Propagation {
                        emission_id: emission_id.clone(),
                        recv: tx_id,
                        send: black_hole_spec::ObjectId::nil(),
                    }
                }
                other => {
                    return Err(Failure::Message(format!(
                        "expected Propagation for forwarding from node {}, got {:?}",
                        node_id, other
                    )));
                }
            };

            // Store the tx id in shared state for this outgoing edge
            {
                let mut inner = state.get_shared().lock().unwrap();
                inner.tx.insert(*target_id, tx_id);
            }

            tracing::debug!(
                node_id,
                target_id,
                ?tx_id,
                "forwarded transmission to outgoing node"
            );
        }

        // Remove the processed node from current
        let mut current = state.get_current().clone();
        current.remove(&node_id);
        state.set_current(current);

        Ok(())
    }
}
// ---------------------------------------------------------------------------
// TopologyState — trait for branch state types (A, B, C)
// ---------------------------------------------------------------------------

/// Trait that provides accessors for the topology-related fields
/// shared by [`A`](super::A), [`B`](super::B), and [`C`](super::C).
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

impl TopologyState for super::A {
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

impl TopologyState for super::B {
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

impl TopologyState for super::C {
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
