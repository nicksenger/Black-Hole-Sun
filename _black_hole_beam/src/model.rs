//! The Black Hole Sun cell model: cells, edges, and graph construction from
//! static flows or live appearances.

use std::collections::{HashMap, HashSet};

use black_hole_flux::sun::{SunAppearance, SunNodeState, SunOperationalState};
use black_hole_flux::Ray;
use iced_sugiyama::Graph;
use jungle_sdk::Animal;
use uuid::Uuid;

use crate::flow::BlackHoleSunFlow;
use crate::labels::{animal_label_key, short_type_name};

#[derive(Clone)]
pub(crate) struct CellDefinition {
    pub(crate) id: u32,
    pub(crate) journey_id: Uuid,
    /// Journey of the nested warp animal for warp cells; nil otherwise.
    pub(crate) warp_journey_id: Uuid,
    pub(crate) ports: Vec<u32>,
    pub(crate) outgoing_ports: Vec<u32>,
    pub(crate) animal_name: String,
    pub(crate) operational_state: SunOperationalState,
    pub(crate) phase_annotation: Option<String>,
    pub(crate) state: SunNodeState,
    pub(crate) state_sequence: u64,
    pub(crate) grad_step: usize,
    pub(crate) grad_steps: usize,
    pub(crate) frozen: Option<bool>,
}

impl CellDefinition {
    pub(crate) fn new<A>(id: u32, ports: Vec<u32>, outgoing_ports: Vec<u32>) -> Self
    where
        A: Animal + 'static,
    {
        Self {
            id,
            journey_id: Uuid::nil(),
            warp_journey_id: Uuid::nil(),
            ports,
            outgoing_ports,
            animal_name: short_type_name::<A>(),
            operational_state: SunOperationalState::Queued,
            phase_annotation: None,
            state: SunNodeState::Idle,
            state_sequence: 0,
            grad_step: 1,
            grad_steps: 1,
            frozen: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct BeamModel {
    pub(crate) cells: Vec<CellDefinition>,
    pub(crate) graph: Graph,
    pub(crate) grad_steps: usize,
    pub(crate) errors: Vec<String>,
    /// Main-graph id -> path of local cell ids (top level first) for every
    /// warp cell, e.g. `[7]` for a top-level warp and `[7, 3]` for warp cell
    /// 3 inside cell 7's nested sun. Empty for statically built models.
    pub(crate) warp_paths: HashMap<u32, Vec<u32>>,
}

impl BeamModel {
    pub(crate) fn empty() -> Self {
        Self {
            cells: Vec::new(),
            graph: Graph::new(Vec::new(), Vec::new()),
            grad_steps: 1,
            errors: Vec::new(),
            warp_paths: HashMap::new(),
        }
    }

    pub(crate) fn build<F>() -> Self
    where
        F: BlackHoleSunFlow,
    {
        let mut cells = Vec::new();
        <F as crate::flow::private::DescribeSun>::append_cells(&mut cells);

        let mut errors = Vec::new();
        let mut port_owner = HashMap::<u32, u32>::new();
        let mut cell_index_by_id = HashMap::<u32, usize>::new();

        for (index, cell) in cells.iter().enumerate() {
            if cell_index_by_id.insert(cell.id, index).is_some() {
                errors.push(format!("duplicate cell id {}", cell.id));
            }
            for port in &cell.ports {
                if let Some(owner) = port_owner.insert(*port, cell.id) {
                    errors.push(format!(
                        "input port {port} belongs to both cell {owner} and cell {}",
                        cell.id
                    ));
                }
            }
        }

        let mut edges = Vec::new();
        let mut seen_edges = HashSet::new();
        for cell in &cells {
            for port in &cell.outgoing_ports {
                match port_owner.get(port).copied() {
                    Some(target) if target != cell.id => {
                        if seen_edges.insert((cell.id, target)) {
                            edges.push((cell.id, target));
                        }
                    }
                    Some(_) => {
                        errors.push(format!("cell {} has a self edge on port {port}", cell.id))
                    }
                    None => errors.push(format!(
                        "cell {} targets unknown input port {port}",
                        cell.id
                    )),
                }
            }
        }

        let nodes = cells.iter().map(|cell| cell.id).collect::<Vec<_>>();
        edges.sort_unstable();
        let graph = Graph::new(nodes, edges);

        if cells.is_empty() {
            errors.push("the Black Hole Sun contains no cells".to_string());
        }

        Self {
            cells,
            graph,
            grad_steps: 1,
            errors,
            warp_paths: HashMap::new(),
        }
    }

    /// Builds the main graph from a live appearance, merging every finalized
    /// nested sun listed in `warp_appearances`.
    ///
    /// `warp_appearances` is keyed by the path of cell ids that locates the
    /// warp cell relative to `appearance`: `[7]` is a warp cell of this
    /// appearance, while `[7, 3]` is warp cell 3 inside cell 7's nested sun.
    /// Merging is recursive: when a nested sun joins the main graph, its own
    /// listed sub-suns join with it.
    pub(crate) fn from_appearance(
        appearance: SunAppearance,
        child_rays: &HashMap<Uuid, Ray>,
        warp_appearances: &HashMap<Vec<u32>, SunAppearance>,
    ) -> Result<Self, String> {
        if !appearance.finalized {
            return Err("the Black Hole Sun topology is not finalized".to_string());
        }
        let grad_steps = appearance.grad_steps.max(1);

        let mut errors = Vec::new();
        let (cells, pending_edges, warp_paths) =
            Self::merge_appearance(&appearance, child_rays, warp_appearances, &mut errors);
        let edges = Self::validate(&cells, &pending_edges, &mut errors);

        if cells.is_empty() {
            errors.push("the Black Hole Sun contains no cells".to_string());
        }
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }

        let nodes = cells.iter().map(|cell| cell.id).collect();
        Ok(Self {
            cells,
            graph: Graph::new(nodes, edges),
            grad_steps,
            errors: Vec::new(),
            warp_paths,
        })
    }

    /// Combines `appearance`'s cells with every listed nested sun (recursively),
    /// remapping nested node and port ids to fresh values so they cannot
    /// collide with the outer topology, and adding an edge from each boundary
    /// cell to its subgraph's terminal node.
    ///
    /// Returns the combined cells, the unvalidated edges (source, target,
    /// target port), and the warp paths of every merged warp cell. Structural
    /// validation happens in [`Self::validate`]; violations are reported in
    /// `errors`.
    fn merge_appearance(
        appearance: &SunAppearance,
        child_rays: &HashMap<Uuid, Ray>,
        warp_appearances: &HashMap<Vec<u32>, SunAppearance>,
        errors: &mut Vec<String>,
    ) -> (
        Vec<CellDefinition>,
        Vec<(u32, u32, u32)>,
        HashMap<u32, Vec<u32>>,
    ) {
        let grad_steps = appearance.grad_steps.max(1);

        let mut cells: Vec<CellDefinition> = appearance
            .nodes
            .iter()
            .map(|node| CellDefinition {
                id: node.id,
                journey_id: node.journey_id,
                warp_journey_id: node.warp_journey_id,
                ports: node.input_ports.clone(),
                outgoing_ports: Vec::new(),
                animal_name: animal_label_key(&node.label),
                operational_state: node.operational_state,
                phase_annotation: node.phase_annotation.clone(),
                state: node.state,
                state_sequence: node.state_sequence,
                grad_step: node.grad_step.clamp(1, grad_steps),
                grad_steps,
                frozen: child_rays.get(&node.journey_id).map(|ray| ray.frozen),
            })
            .collect();
        cells.sort_by_key(|cell| cell.id);

        // Outer edges plus every merged warp subgraph edge, validated
        // together by the caller.
        let mut pending_edges: Vec<(u32, u32, u32)> = appearance
            .edges
            .iter()
            .map(|edge| (edge.source, edge.target, edge.target_port))
            .collect();

        // Merge each warp cell's nested sun into the main graph. Nested node
        // and port ids are remapped to fresh values so they cannot collide
        // with the outer topology, and an edge connects the boundary cell to
        // the nested sink (the subgraph's terminal node).
        let mut next_id = cells.iter().map(|cell| cell.id).max().unwrap_or(0) + 1;
        let mut next_port = cells
            .iter()
            .flat_map(|cell| cell.ports.iter().copied())
            .max()
            .unwrap_or(0)
            + 1;
        // Every warp cell of this appearance is locatable by its own id.
        let mut warp_paths: HashMap<u32, Vec<u32>> = cells
            .iter()
            .filter(|cell| !cell.warp_journey_id.is_nil())
            .map(|cell| (cell.id, vec![cell.id]))
            .collect();
        // Warp cells of this appearance are the length-1 paths; longer paths
        // belong to deeper levels and travel with their parent's sub-map.
        let mut warp_cell_ids: Vec<u32> = warp_appearances
            .keys()
            .filter(|path| path.len() == 1)
            .map(|path| path[0])
            .collect();
        warp_cell_ids.sort_unstable();
        for parent_id in warp_cell_ids {
            let warp_appearance = &warp_appearances[&vec![parent_id]];
            if !warp_appearance.finalized {
                // The subgraph joins the main graph once the nested sun
                // finalizes.
                continue;
            }
            if !cells.iter().any(|cell| cell.id == parent_id) {
                errors.push(format!("warp appearance for unknown cell {parent_id}"));
                continue;
            }
            // Deeper expansions are re-keyed relative to the nested sun and
            // merged recursively.
            let sub_expansions: HashMap<Vec<u32>, SunAppearance> = warp_appearances
                .iter()
                .filter(|(path, _)| path.len() > 1 && path.first() == Some(&parent_id))
                .map(|(path, appearance)| (path[1..].to_vec(), appearance.clone()))
                .collect();
            // Validate the nested sun with the same rules as the outer one;
            // a malformed subgraph is skipped rather than failing the whole
            // model.
            let mut nested_errors = Vec::new();
            let (nested_cells, nested_edges, nested_paths) = Self::merge_appearance(
                warp_appearance,
                child_rays,
                &sub_expansions,
                &mut nested_errors,
            );
            let _ = Self::validate(&nested_cells, &nested_edges, &mut nested_errors);
            if !nested_errors.is_empty() {
                continue;
            }

            // A finalized sun has exactly one sink: the node with no
            // outgoing edges in its own appearance. Deeper merges connect
            // through their own boundaries, so this stays the terminal of
            // this subgraph.
            let sources = warp_appearance
                .edges
                .iter()
                .map(|edge| edge.source)
                .collect::<HashSet<_>>();
            let sinks: Vec<u32> = warp_appearance
                .nodes
                .iter()
                .filter(|node| !sources.contains(&node.id))
                .map(|node| node.id)
                .collect();
            if sinks.len() != 1 {
                continue;
            }

            let mut id_map = HashMap::new();
            for cell in &nested_cells {
                id_map.insert(cell.id, next_id);
                next_id += 1;
            }
            let mut nested_ports: Vec<u32> = nested_cells
                .iter()
                .flat_map(|cell| cell.ports.iter().copied())
                .collect();
            nested_ports.sort_unstable();
            nested_ports.dedup();
            let mut port_map = HashMap::new();
            for port in nested_ports {
                port_map.insert(port, next_port);
                next_port += 1;
            }

            // Carry the nested sun's warp paths over, prefixed with this
            // boundary cell so merged warp cells stay locatable.
            for (local_id, local_path) in &nested_paths {
                if let Some(&merged_id) = id_map.get(local_id) {
                    let mut path = vec![parent_id];
                    path.extend_from_slice(local_path);
                    warp_paths.insert(merged_id, path);
                }
            }

            for cell in nested_cells {
                let id = id_map[&cell.id];
                cells.push(CellDefinition {
                    id,
                    ports: cell.ports.iter().map(|port| port_map[port]).collect(),
                    ..cell
                });
            }
            for (source, target, target_port) in nested_edges {
                pending_edges.push((id_map[&source], id_map[&target], port_map[&target_port]));
            }

            // Connect the boundary cell to the nested sink through a
            // dedicated input port.
            let sink_id = id_map[&sinks[0]];
            let connector_port = next_port;
            next_port += 1;
            if let Some(sink_cell) = cells.iter_mut().find(|cell| cell.id == sink_id) {
                sink_cell.ports.push(connector_port);
            }
            pending_edges.push((parent_id, sink_id, connector_port));
        }

        (cells, pending_edges, warp_paths)
    }

    /// Checks merged cells and edges for duplicates and dangling references,
    /// returning the deduplicated `(source, target)` edge list.
    fn validate(
        cells: &[CellDefinition],
        pending_edges: &[(u32, u32, u32)],
        errors: &mut Vec<String>,
    ) -> Vec<(u32, u32)> {
        let mut node_ids = HashSet::new();
        let mut port_owner = HashMap::new();
        for cell in cells {
            if !node_ids.insert(cell.id) {
                errors.push(format!("duplicate cell id {}", cell.id));
            }
            for &port in &cell.ports {
                if let Some(owner) = port_owner.insert(port, cell.id) {
                    errors.push(format!(
                        "input port {port} belongs to both cell {owner} and cell {}",
                        cell.id
                    ));
                }
            }
        }

        let mut edges = Vec::new();
        let mut seen_edges = HashSet::new();
        for (source, target, target_port) in pending_edges {
            if !node_ids.contains(source) {
                errors.push(format!("edge starts at unknown cell {source}"));
                continue;
            }
            if !node_ids.contains(target) {
                errors.push(format!("edge targets unknown cell {target}"));
                continue;
            }
            if source == target {
                errors.push(format!(
                    "cell {source} has a self edge on port {target_port}"
                ));
                continue;
            }
            if port_owner.get(target_port) != Some(target) {
                errors.push(format!(
                    "edge to cell {target} references unowned input port {target_port}"
                ));
                continue;
            }
            if seen_edges.insert((*source, *target)) {
                edges.push((*source, *target));
            }
        }
        edges.sort_unstable();
        edges
    }
}

pub(crate) fn model_display_changed(current: &BeamModel, next: &BeamModel) -> bool {
    current.graph.nodes != next.graph.nodes
        || current.graph.edges != next.graph.edges
        || current.grad_steps != next.grad_steps
        || current.cells.len() != next.cells.len()
        || current
            .cells
            .iter()
            .zip(next.cells.iter())
            .any(|(current, next)| {
                current.id != next.id
                    || current.journey_id != next.journey_id
                    || current.animal_name != next.animal_name
                    || current.operational_state != next.operational_state
                    || current.phase_annotation != next.phase_annotation
                    || current.grad_step != next.grad_step
                    || current.grad_steps != next.grad_steps
                    || current.frozen != next.frozen
            })
}
