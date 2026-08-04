//! Visualize Black Hole Sun cell graphs and their Jungle child flows.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole) with the circular `circo`
//! layout. Selecting a cell opens the Jungle flow for the animal hosted by that
//! cell. Live views discover the child journey IDs from the parent Sun journey
//! and apply each child's update stream to its flow graph.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use black_hole_flux::sun::action::{SpawnBinary, SpawnUnary};
use black_hole_flux::sun::{BinarySunStep, NodeIdsFromList, Sun, SunNode, UnarySunStep};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState, ObjectId};
use iced::futures::{self, Stream, StreamExt};
use iced::widget::{button, column, container, row, text, Space};
use iced::{Background, Color, Element, Font, Length, Shadow, Subscription, Task, Theme, Vector};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{circo_layout, AutoFit, Cluster, Graph, LayoutInput, Sugiyama};
use jungle_sdk::client::JourneyUpdateSubscription;
use jungle_sdk::core::dag::{Dag, DagSnapshot, LiveDagState, Phase, RuntimeState};
use jungle_sdk::{
    Action, Animal, AnimalIdValue, JourneyAst, JourneyAstSource, JourneyUpdateEvent, JungleClient,
    RunnerOut,
};
use typenum::Unsigned;
use uuid::Uuid;

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;
const CELL_NODE_WIDTH: f64 = 210.0;
const CELL_NODE_HEIGHT: f64 = 78.0;
const FLOW_NODE_WIDTH: f64 = 230.0;
const FLOW_NODE_HEIGHT: f64 = 76.0;
const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
const FLOW_GRAPH_ID: &str = "black-hole-beam-child-flow";
const DISCOVERY_INTERVAL: Duration = Duration::from_millis(750);

/// Builder for Black Hole Sun graph viewers.
///
/// A static view can be launched directly:
///
/// ```ignore
/// BeamBuilder::new().view::<MyBlackHoleAnimal>()
/// ```
///
/// A live view accepts the Jungle client and parent Sun journey:
///
/// ```ignore
/// BeamBuilder::new()
///     .view_live::<MyBlackHoleAnimal>(client, journey_id)
/// ```
#[derive(Clone)]
pub struct BeamBuilder {
    title: String,
    width: f32,
    height: f32,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
}

impl Default for BeamBuilder {
    fn default() -> Self {
        Self {
            title: "Black Hole Beam".to_string(),
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            animation_duration: None,
            animation_easing: None,
        }
    }
}

impl BeamBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render a static Black Hole Sun and its child flows.
    pub fn view<A>(self) -> iced::Result
    where
        A: Animal + 'static,
        A::Flow: BlackHoleSunFlow,
    {
        run_beam::<A>(self.into_config(), None)
    }

    /// Render a live Black Hole Sun and each spawned child journey.
    pub fn view_live<A>(self, client: impl JungleClient + 'static, journey_id: Uuid) -> iced::Result
    where
        A: Animal + 'static,
        A::Flow: BlackHoleSunFlow,
    {
        let live = LiveConfig {
            client: Arc::new(client),
            journey_id,
        };
        run_beam::<A>(self.into_config(), Some(live))
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn window_size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn animation_duration(mut self, duration: Duration) -> Self {
        self.animation_duration = Some(duration);
        self
    }

    pub fn animation_easing(mut self, easing: &'static Easing) -> Self {
        self.animation_easing = Some(easing);
        self
    }

    fn into_config(self) -> BeamConfig {
        BeamConfig {
            title: self.title,
            width: self.width,
            height: self.height,
            animation_duration: self.animation_duration,
            animation_easing: self.animation_easing,
        }
    }
}

/// Render a static Black Hole Sun with default viewer settings.
pub fn view<A>() -> iced::Result
where
    A: Animal + 'static,
    A::Flow: BlackHoleSunFlow,
{
    BeamBuilder::new().view::<A>()
}

/// Render a live Black Hole Sun with default viewer settings.
pub fn view_live<A>(client: impl JungleClient + 'static, journey_id: Uuid) -> iced::Result
where
    A: Animal + 'static,
    A::Flow: BlackHoleSunFlow,
{
    BeamBuilder::new().view_live::<A>(client, journey_id)
}

mod private {
    pub(crate) trait DescribeSun {
        fn append_cells(cells: &mut Vec<super::CellDefinition>);
    }
}

/// Marker for the structural flow produced by
/// `<Graph as BlackHole>::Sun<Generator, Policy>`.
///
/// The trait is sealed and is only implemented for the `SunNode<…>` chain
/// emitted by [`BlackHole`](black_hole_flux::sun::BlackHole).
#[allow(private_bounds)]
pub trait BlackHoleSunFlow: JourneyAstSource + private::DescribeSun {}

impl<T> BlackHoleSunFlow for T where T: JourneyAstSource + private::DescribeSun {}

impl<Generator, Policy> private::DescribeSun for Sun<Generator, Policy>
where
    Sun<Generator, Policy>: JourneyAstSource,
{
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Port, A, Edges, Tail> private::DescribeSun for SunNode<UnarySunStep<Port, A, Edges>, Tail>
where
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId> + 'static,
    A::Flow: JourneyAstSource,
    Edges: NodeIdsFromList,
    Tail: private::DescribeSun,
    SunNode<UnarySunStep<Port, A, Edges>, Tail>: JourneyAstSource,
    SpawnUnary<Port, A, Edges>: Action,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            Port::U32,
            vec![Port::U32],
            Edges::node_ids(),
            <SpawnUnary<Port, A, Edges> as Action>::NAME,
        ));
        Tail::append_cells(cells);
    }
}

impl<PortA, PortB, A, Edges, Tail> private::DescribeSun
    for SunNode<BinarySunStep<PortA, PortB, A, Edges>, Tail>
where
    PortA: Unsigned,
    PortB: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = FusionSeed, State = FusionState>
        + 'static,
    A::Flow: FusionFlow + JourneyAstSource,
    Edges: NodeIdsFromList,
    Tail: private::DescribeSun,
    SunNode<BinarySunStep<PortA, PortB, A, Edges>, Tail>: JourneyAstSource,
    SpawnBinary<PortA, PortB, A, Edges>: Action,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            PortA::U32,
            vec![PortA::U32, PortB::U32],
            Edges::node_ids(),
            <SpawnBinary<PortA, PortB, A, Edges> as Action>::NAME,
        ));
        Tail::append_cells(cells);
    }
}

#[derive(Clone)]
struct BeamConfig {
    title: String,
    width: f32,
    height: f32,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
}

#[derive(Clone)]
struct LiveConfig {
    client: Arc<dyn JungleClient>,
    journey_id: Uuid,
}

#[derive(Clone)]
struct CellDefinition {
    id: u32,
    ports: Vec<u32>,
    outgoing_ports: Vec<u32>,
    animal_name: String,
    flow_name: String,
    dag: Dag,
    graph: Graph,
    static_colors: HashMap<u32, Color>,
    spawn_action: &'static str,
}

impl CellDefinition {
    fn new<A>(
        id: u32,
        ports: Vec<u32>,
        outgoing_ports: Vec<u32>,
        spawn_action: &'static str,
    ) -> Self
    where
        A: Animal + 'static,
        A::Flow: JourneyAstSource,
    {
        let dag = Dag::from_ast(<A::Flow as JourneyAstSource>::journey_ast());
        let nodes = dag.nodes.iter().map(|node| node.id).collect::<Vec<_>>();
        let graph = Graph::new(nodes.clone(), dag.edges.clone());
        let static_colors = graph_colors(&nodes, &dag.edges);

        Self {
            id,
            ports,
            outgoing_ports,
            animal_name: short_type_name::<A>(),
            flow_name: short_type_name::<A::Flow>(),
            dag,
            graph,
            static_colors,
            spawn_action,
        }
    }
}

struct BeamModel {
    cells: Vec<CellDefinition>,
    graph: Graph,
    static_colors: HashMap<u32, Color>,
    spawn_runtime_to_cell: HashMap<u32, usize>,
    errors: Vec<String>,
}

impl BeamModel {
    fn build<F>() -> Self
    where
        F: BlackHoleSunFlow,
    {
        let mut cells = Vec::new();
        <F as private::DescribeSun>::append_cells(&mut cells);

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
        let static_colors = graph_colors(&nodes, &edges);
        let graph = Graph::new(nodes, edges);

        let ast = <F as JourneyAstSource>::journey_ast();
        let mut next_runtime_id = 0;
        let mut runtime_steps = Vec::new();
        collect_runtime_steps(&ast, &mut next_runtime_id, &mut runtime_steps);
        let spawn_steps = runtime_steps
            .into_iter()
            .filter(|(_, label)| cells.iter().any(|cell| cell.spawn_action == *label))
            .collect::<Vec<_>>();
        let mut spawn_runtime_to_cell = HashMap::new();

        if spawn_steps.len() != cells.len() {
            errors.push(format!(
                "found {} spawn steps for {} cells",
                spawn_steps.len(),
                cells.len()
            ));
        }
        for (index, ((runtime_id, label), cell)) in
            spawn_steps.into_iter().zip(cells.iter()).enumerate()
        {
            if label != cell.spawn_action {
                errors.push(format!(
                    "spawn step {label} did not match {} for cell {}",
                    cell.spawn_action, cell.id
                ));
                continue;
            }
            spawn_runtime_to_cell.insert(runtime_id, index);
        }

        if cells.is_empty() {
            errors.push("the Black Hole Sun contains no cells".to_string());
        }

        Self {
            cells,
            graph,
            static_colors,
            spawn_runtime_to_cell,
            errors,
        }
    }
}

fn collect_runtime_steps<'a>(
    ast: &'a JourneyAst,
    next_runtime_id: &mut u32,
    steps: &mut Vec<(u32, &'a str)>,
) {
    match ast {
        JourneyAst::Empty => {}
        JourneyAst::Sequence(items) => {
            for item in items {
                collect_runtime_steps(item, next_runtime_id, steps);
            }
        }
        JourneyAst::Step { label } => {
            let runtime_id = *next_runtime_id;
            *next_runtime_id = next_runtime_id.saturating_add(1);
            steps.push((runtime_id, label));
        }
        JourneyAst::Conditional { left, right, .. }
        | JourneyAst::Select { left, right, .. }
        | JourneyAst::Join { left, right, .. } => {
            *next_runtime_id = next_runtime_id.saturating_add(1);
            collect_runtime_steps(left, next_runtime_id, steps);
            collect_runtime_steps(right, next_runtime_id, steps);
        }
        JourneyAst::While { body, .. }
        | JourneyAst::Transparent { body, .. }
        | JourneyAst::Attempt { body, .. } => {
            *next_runtime_id = next_runtime_id.saturating_add(1);
            collect_runtime_steps(body, next_runtime_id, steps);
        }
    }
}

fn run_beam<A>(config: BeamConfig, live: Option<LiveConfig>) -> iced::Result
where
    A: Animal + 'static,
    A::Flow: BlackHoleSunFlow,
{
    let title = config.title.clone();
    let width = config.width;
    let height = config.height;
    iced::application(
        move || BeamApp::new::<A>(config.clone(), live.clone()),
        BeamApp::update,
        BeamApp::view,
    )
    .title(move |_app: &BeamApp| title.clone())
    .subscription(BeamApp::subscription)
    .theme(beam_theme)
    .default_font(Font::with_name("Times"))
    .window_size((width, height))
    .antialiasing(true)
    .run()
}

#[derive(Debug, Clone)]
enum Message {
    SelectCell(usize),
    DiscoveryTick,
    ChildrenDiscovered(Result<Vec<(usize, Uuid)>, String>),
    ChildUpdate {
        cell_index: usize,
        update: Result<JourneyUpdateEvent, String>,
    },
}

struct CellRuntime {
    journey_id: Option<Uuid>,
    live: LiveDagState,
    stream_error: Option<String>,
}

impl CellRuntime {
    fn new(cell: &CellDefinition) -> Self {
        let mut live = LiveDagState::default();
        live.bind_model(&cell.dag);
        Self {
            journey_id: None,
            live,
            stream_error: None,
        }
    }
}

struct BeamApp {
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
    cell_runtime: Vec<CellRuntime>,
    selected_cell: Option<usize>,
    discovering: bool,
    discovery_error: Option<String>,
}

impl BeamApp {
    fn new<A>(config: BeamConfig, live: Option<LiveConfig>) -> (Self, Task<Message>)
    where
        A: Animal + 'static,
        A::Flow: BlackHoleSunFlow,
    {
        let model = BeamModel::build::<A::Flow>();
        let cell_runtime = model.cells.iter().map(CellRuntime::new).collect();
        let discovering = live.is_some();
        let task = live
            .as_ref()
            .map(|live| discovery_task(live.clone(), model.spawn_runtime_to_cell.clone()))
            .unwrap_or_else(Task::none);

        (
            Self {
                config,
                model,
                live,
                cell_runtime,
                selected_cell: None,
                discovering,
                discovery_error: None,
            },
            task,
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SelectCell(index) => {
                if index < self.model.cells.len() {
                    self.selected_cell = Some(index);
                    return iced_sugiyama::force_review(iced_sugiyama::Id::new(FLOW_GRAPH_ID));
                }
            }
            Message::DiscoveryTick => {
                if !self.discovering
                    && self.live.is_some()
                    && self
                        .cell_runtime
                        .iter()
                        .any(|runtime| runtime.journey_id.is_none())
                {
                    self.discovering = true;
                    if let Some(live) = self.live.clone() {
                        return discovery_task(live, self.model.spawn_runtime_to_cell.clone());
                    }
                }
            }
            Message::ChildrenDiscovered(result) => {
                self.discovering = false;
                match result {
                    Ok(children) => {
                        self.discovery_error = None;
                        for (index, journey_id) in children {
                            let Some(runtime) = self.cell_runtime.get_mut(index) else {
                                continue;
                            };
                            if runtime.journey_id != Some(journey_id) {
                                runtime.journey_id = Some(journey_id);
                                runtime.live = LiveDagState::default();
                                runtime.live.bind_model(&self.model.cells[index].dag);
                                runtime.stream_error = None;
                            }
                        }
                        return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
                    }
                    Err(error) => self.discovery_error = Some(error),
                }
            }
            Message::ChildUpdate { cell_index, update } => {
                let Some(runtime) = self.cell_runtime.get_mut(cell_index) else {
                    return Task::none();
                };
                match update {
                    Ok(update) => {
                        runtime.stream_error = None;
                        runtime.live.apply_update(update);
                    }
                    Err(error) => runtime.stream_error = Some(error),
                }

                let mut tasks = vec![iced_sugiyama::force_review(iced_sugiyama::Id::new(
                    CELL_GRAPH_ID,
                ))];
                if self.selected_cell == Some(cell_index) {
                    tasks.push(iced_sugiyama::force_review(iced_sugiyama::Id::new(
                        FLOW_GRAPH_ID,
                    )));
                }
                return Task::batch(tasks);
            }
        }

        Task::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        let Some(live) = &self.live else {
            return Subscription::none();
        };

        let mut subscriptions = Vec::new();
        if self
            .cell_runtime
            .iter()
            .any(|runtime| runtime.journey_id.is_none())
        {
            subscriptions
                .push(iced::time::every(DISCOVERY_INTERVAL).map(|_| Message::DiscoveryTick));
        }

        for (cell_index, runtime) in self.cell_runtime.iter().enumerate() {
            let Some(journey_id) = runtime.journey_id else {
                continue;
            };
            subscriptions.push(Subscription::run_with(
                ChildSubscription {
                    client: live.client.clone(),
                    journey_id,
                    cell_index,
                },
                child_updates_stream,
            ));
        }

        Subscription::batch(subscriptions)
    }

    fn view(&self) -> Element<'_, Message> {
        let mode_label = match &self.live {
            Some(live) => format!("live · parent {}", short_uuid(live.journey_id)),
            None => "static".to_string(),
        };
        let header = row![
            column![
                text(&self.config.title)
                    .size(28)
                    .color(inferno_text_bright()),
                text(mode_label).size(13).color(inferno_text_muted()),
            ]
            .spacing(2),
            Space::new().width(Length::Fill),
            text(format!("{} cells", self.model.cells.len()))
                .size(14)
                .color(inferno_text_muted()),
        ]
        .align_y(iced::Alignment::Center)
        .padding([14, 20]);

        let body = row![self.cell_graph_panel(), self.child_flow_panel()]
            .spacing(12)
            .padding(12)
            .height(Length::Fill);

        container(column![header, body].height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(app_background_style)
            .into()
    }

    fn cell_graph_panel(&self) -> Element<'_, Message> {
        let labels = self
            .model
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                (
                    cell.id,
                    (
                        index,
                        cell.animal_name.clone(),
                        self.cell_status_label(index),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let colors = self.cell_colors();
        let selected = self.selected_cell;
        let colors_for_nodes = colors.clone();
        let colors_for_edges = colors.clone();

        let mut graph =
            Sugiyama::<Message, Theme, iced::Renderer>::new(&self.model.graph, move |node_id| {
                let (index, animal_name, status) = labels.get(&node_id).cloned().unwrap_or((
                    0,
                    format!("cell {node_id}"),
                    "unknown".to_string(),
                ));
                let color = colors_for_nodes
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                let is_selected = selected == Some(index);
                button(
                    column![
                        text(animal_name).size(16).color(contrasting_text(color)),
                        text(format!("cell {node_id} · {status}"))
                            .size(12)
                            .color(contrasting_text(color).scale_alpha(0.82)),
                    ]
                    .spacing(3),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding([10, 12])
                .on_press(Message::SelectCell(index))
                .style(move |_theme, status| node_button_style(color, is_selected, status))
                .into()
            })
            .id(iced_sugiyama::Id::new(CELL_GRAPH_ID))
            .layout_fn(|input| {
                // circo's ported implementation addresses edge coordinates by
                // node index. Remap public cell IDs so sparse port numbers keep
                // their edges attached to the correct nodes.
                let original_nodes = input.nodes.clone();
                let node_index = original_nodes
                    .iter()
                    .enumerate()
                    .map(|(index, node)| (*node, index as u32))
                    .collect::<HashMap<_, _>>();
                let remapped_nodes =
                    Arc::from((0..original_nodes.len() as u32).collect::<Vec<_>>());
                let remapped_edges = Arc::from(
                    input
                        .edges
                        .iter()
                        .map(|(from, to)| (node_index[from], node_index[to]))
                        .collect::<Vec<_>>(),
                );
                let original_node_size = input.node_size.clone();
                let node_ids_for_size = original_nodes.clone();
                let original_edge_label = input.edge_label.clone();
                let original_edges = input.edges.clone();

                #[allow(clippy::arc_with_non_send_sync)]
                let remapped_input = LayoutInput {
                    nodes: remapped_nodes,
                    edges: remapped_edges,
                    config: input.config,
                    render_config: input.render_config,
                    clusters: Arc::from(Vec::<Cluster>::new()),
                    node_size: Arc::new(move |index| {
                        original_node_size(node_ids_for_size[index as usize])
                    }),
                    edge_label: Arc::new(move |index, _| {
                        original_edge_label(index, original_edges[index])
                    }),
                };
                circo_layout(&remapped_input)
            })
            .node_size(|_| (CELL_NODE_WIDTH, CELL_NODE_HEIGHT))
            .edge_color(move |ctx| {
                let start = colors_for_edges
                    .get(&ctx.edge.0)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                let end = colors_for_edges
                    .get(&ctx.edge.1)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                (lighten(start, 0.18), end)
            })
            .stroke_width(1.4)
            .edge_corner_radius(16.0)
            .padding(28)
            .auto_fit(AutoFit::Ongoing)
            .keep_centered(true);
        if let Some(duration) = self.config.animation_duration {
            graph = graph.animation_duration(duration);
        }
        if let Some(easing) = self.config.animation_easing {
            graph = graph.animation_easing(easing);
        }

        let mut content = column![
            text("Sun cells").size(18).color(inferno_text_bright()),
            text("Select a cell to inspect its animal flow")
                .size(12)
                .color(inferno_text_muted()),
        ]
        .spacing(3);
        if !self.model.errors.is_empty() {
            content = content.push(
                text(self.model.errors.join(" · "))
                    .size(12)
                    .color(Color::from_rgb8(255, 120, 105)),
            );
        }
        if let Some(error) = &self.discovery_error {
            content = content.push(
                text(format!("child discovery: {error}"))
                    .size(12)
                    .color(Color::from_rgb8(255, 120, 105)),
            );
        }
        content = content.push(
            container(graph)
                .width(Length::Fill)
                .height(Length::Fill)
                .clip(true),
        );

        container(content.padding(16).height(Length::Fill))
            .width(Length::FillPortion(5))
            .height(Length::Fill)
            .style(panel_style)
            .into()
    }

    fn child_flow_panel(&self) -> Element<'_, Message> {
        let Some(index) = self.selected_cell else {
            return container(
                column![
                    text("Child flow").size(18).color(inferno_text_bright()),
                    Space::new().height(Length::Fill),
                    text("Select a Sun cell to view its Jungle flow")
                        .size(16)
                        .color(inferno_text_muted()),
                    Space::new().height(Length::Fill),
                ]
                .align_x(iced::Alignment::Center)
                .padding(16),
            )
            .width(Length::FillPortion(7))
            .height(Length::Fill)
            .style(panel_style)
            .into();
        };

        let cell = &self.model.cells[index];
        let runtime = &self.cell_runtime[index];
        let live_data = self.live.as_ref().map(|_| &runtime.live);
        let snapshot = DagSnapshot::new(&cell.dag, live_data);
        let labels = cell
            .dag
            .nodes
            .iter()
            .map(|node| (node.id, node.label.clone()))
            .collect::<HashMap<_, _>>();
        let colors = cell
            .dag
            .nodes
            .iter()
            .map(|node| {
                let color = match snapshot.node_phase(node.id) {
                    Phase::Static => cell
                        .static_colors
                        .get(&node.id)
                        .copied()
                        .unwrap_or_else(inferno_node_base),
                    Phase::Live(state) => runtime_color(state),
                };
                (node.id, color)
            })
            .collect::<HashMap<_, _>>();
        let colors_for_nodes = colors.clone();
        let colors_for_edges = colors.clone();
        let clusters = cell
            .dag
            .clusters
            .iter()
            .map(|cluster| {
                let mut value = Cluster::new(cluster.nodes.clone());
                if let Some(padding) = cluster.padding {
                    value = value.padding(padding.into());
                }
                if let Some(parent) = cluster.parent {
                    value = value.parent(parent);
                }
                value
            })
            .collect::<Vec<_>>();

        let mut graph =
            Sugiyama::<Message, Theme, iced::Renderer>::new(&cell.graph, move |node_id| {
                let label = labels
                    .get(&node_id)
                    .map(|label| truncate_label(label, 38))
                    .unwrap_or_else(|| format!("step {node_id}"));
                let color = colors_for_nodes
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                container(text(label).size(14).color(contrasting_text(color)))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .padding([8, 10])
                    .style(move |_theme| flow_node_style(color))
                    .into()
            })
            .id(iced_sugiyama::Id::new(FLOW_GRAPH_ID))
            .node_size(|_| (FLOW_NODE_WIDTH, FLOW_NODE_HEIGHT))
            .edge_color(move |ctx| {
                let start = colors_for_edges
                    .get(&ctx.edge.0)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                let end = colors_for_edges
                    .get(&ctx.edge.1)
                    .copied()
                    .unwrap_or_else(inferno_node_base);
                (lighten(start, 0.2), end)
            })
            .stroke_width(1.2)
            .edge_corner_radius(18.0)
            .clusters(clusters)
            .cluster_color(cluster_fill_color)
            .padding(28)
            .auto_fit(AutoFit::Ongoing)
            .keep_centered(true);
        if let Some(duration) = self.config.animation_duration {
            graph = graph.animation_duration(duration);
        }
        if let Some(easing) = self.config.animation_easing {
            graph = graph.animation_easing(easing);
        }

        let ports = cell
            .ports
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let journey = runtime
            .journey_id
            .map(|id| format!(" · journey {}", short_uuid(id)))
            .unwrap_or_default();
        let mut content = column![
            text(format!("{} child flow", cell.animal_name))
                .size(18)
                .color(inferno_text_bright()),
            text(format!("{} · ports {ports}{journey}", cell.flow_name))
                .size(12)
                .color(inferno_text_muted()),
        ]
        .spacing(3);
        if let Some(error) = &runtime.stream_error {
            content = content.push(
                text(format!("live stream: {error}"))
                    .size(12)
                    .color(Color::from_rgb8(255, 120, 105)),
            );
        }
        content = content.push(
            container(graph)
                .width(Length::Fill)
                .height(Length::Fill)
                .clip(true),
        );

        container(content.padding(16).height(Length::Fill))
            .width(Length::FillPortion(7))
            .height(Length::Fill)
            .style(panel_style)
            .into()
    }

    fn cell_status_label(&self, index: usize) -> String {
        if self.live.is_none() {
            return "static".to_string();
        }
        let runtime = &self.cell_runtime[index];
        if runtime.journey_id.is_none() {
            return "discovering".to_string();
        }
        if !runtime.live.failed_runtime_ids.is_empty() {
            "failed".to_string()
        } else if !runtime.live.active_runtime_ids.is_empty() {
            "running".to_string()
        } else if runtime.live.latest_event_count > 0 {
            "idle".to_string()
        } else {
            "pending".to_string()
        }
    }

    fn cell_colors(&self) -> HashMap<u32, Color> {
        self.model
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let runtime = &self.cell_runtime[index];
                let color = if self.live.is_none() {
                    self.model
                        .static_colors
                        .get(&cell.id)
                        .copied()
                        .unwrap_or_else(inferno_node_base)
                } else if runtime.journey_id.is_none() {
                    inferno_gradient(0.04)
                } else if !runtime.live.failed_runtime_ids.is_empty() {
                    Color::from_rgb8(126, 38, 80)
                } else if !runtime.live.active_runtime_ids.is_empty() {
                    inferno_gradient(0.96)
                } else if runtime.live.latest_event_count > 0 {
                    inferno_gradient(0.62)
                } else {
                    inferno_gradient(0.16)
                };
                (cell.id, color)
            })
            .collect()
    }
}

#[derive(Clone)]
struct ChildSubscription {
    client: Arc<dyn JungleClient>,
    journey_id: Uuid,
    cell_index: usize,
}

impl Hash for ChildSubscription {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.journey_id.hash(state);
        self.cell_index.hash(state);
    }
}

fn child_updates_stream(config: &ChildSubscription) -> impl Stream<Item = Message> {
    let client = config.client.clone();
    let journey_id = config.journey_id;
    let cell_index = config.cell_index;
    futures::stream::once(async move {
        let stream: Pin<Box<dyn Stream<Item = Message> + Send>> =
            match client.subscribe_step_updates(journey_id, None).await {
                Ok(subscription) => Box::pin(map_child_updates(subscription, cell_index)),
                Err(error) => Box::pin(futures::stream::once(async move {
                    Message::ChildUpdate {
                        cell_index,
                        update: Err(error.to_string()),
                    }
                })),
            };
        stream
    })
    .flatten()
}

fn map_child_updates(
    subscription: JourneyUpdateSubscription,
    cell_index: usize,
) -> impl Stream<Item = Message> {
    subscription.map(move |update| Message::ChildUpdate {
        cell_index,
        update: update.map_err(|error| error.to_string()),
    })
}

fn discovery_task(live: LiveConfig, spawn_runtime_to_cell: HashMap<u32, usize>) -> Task<Message> {
    Task::perform(
        discover_children(live, spawn_runtime_to_cell),
        Message::ChildrenDiscovered,
    )
}

async fn discover_children(
    live: LiveConfig,
    spawn_runtime_to_cell: HashMap<u32, usize>,
) -> Result<Vec<(usize, Uuid)>, String> {
    let history = live
        .client
        .journey_history(live.journey_id)
        .await
        .map_err(|error| error.to_string())?;
    decode_child_journeys(history, &spawn_runtime_to_cell)
}

fn decode_child_journeys(
    history: impl IntoIterator<Item = RunnerOut>,
    spawn_runtime_to_cell: &HashMap<u32, usize>,
) -> Result<Vec<(usize, Uuid)>, String> {
    let mut children = HashMap::<usize, Uuid>::new();

    for event in history {
        let RunnerOut::EffectSuccessOutput { node_id, data, .. } = event else {
            continue;
        };
        let Some(cell_index) = spawn_runtime_to_cell.get(&node_id).copied() else {
            continue;
        };
        let journey_id = postcard::from_bytes::<Uuid>(&data).map_err(|error| {
            format!("could not decode child journey for spawn node {node_id}: {error}")
        })?;
        children.insert(cell_index, journey_id);
    }

    let mut children = children.into_iter().collect::<Vec<_>>();
    children.sort_by_key(|(index, _)| *index);
    Ok(children)
}

fn graph_colors(nodes: &[u32], edges: &[(u32, u32)]) -> HashMap<u32, Color> {
    if nodes.is_empty() {
        return HashMap::new();
    }

    let mut indegree = nodes
        .iter()
        .map(|node| (*node, 0_usize))
        .collect::<HashMap<_, _>>();
    let mut outgoing = HashMap::<u32, Vec<u32>>::new();
    let mut degree = nodes
        .iter()
        .map(|node| (*node, 0_usize))
        .collect::<HashMap<_, _>>();
    for (from, to) in edges {
        *indegree.entry(*to).or_default() += 1;
        outgoing.entry(*from).or_default().push(*to);
        *degree.entry(*from).or_default() += 1;
        *degree.entry(*to).or_default() += 1;
    }

    let mut depth = HashMap::<u32, usize>::new();
    let mut queue = VecDeque::new();
    for node in nodes {
        if indegree.get(node).copied().unwrap_or(0) == 0 {
            depth.insert(*node, 0);
            queue.push_back(*node);
        }
    }
    if queue.is_empty() {
        depth.insert(nodes[0], 0);
        queue.push_back(nodes[0]);
    }
    while let Some(node) = queue.pop_front() {
        let next_depth = depth.get(&node).copied().unwrap_or(0).saturating_add(1);
        for target in outgoing.get(&node).into_iter().flatten() {
            let should_update = depth
                .get(target)
                .is_none_or(|current| next_depth < *current);
            if should_update {
                depth.insert(*target, next_depth);
                queue.push_back(*target);
            }
        }
    }

    let max_depth = depth.values().copied().max().unwrap_or(0).max(1) as f32;
    let max_degree = degree.values().copied().max().unwrap_or(0).max(1) as f32;
    nodes
        .iter()
        .map(|node| {
            let layer = depth.get(node).copied().unwrap_or(0) as f32 / max_depth;
            let connected = degree.get(node).copied().unwrap_or(0) as f32 / max_degree;
            let heat = (0.14 + 0.76 * layer + 0.10 * connected).clamp(0.0, 1.0);
            (*node, inferno_gradient(heat))
        })
        .collect()
}

fn runtime_color(state: RuntimeState) -> Color {
    match state {
        RuntimeState::Pending => inferno_gradient(0.08),
        RuntimeState::Running => inferno_gradient(0.97),
        RuntimeState::Completed => inferno_gradient(0.62),
        RuntimeState::Failed => Color::from_rgb8(126, 38, 80),
    }
}

fn inferno_gradient(heat: f32) -> Color {
    let cool = Color::from_rgb8(46, 6, 10);
    let ember = Color::from_rgb8(124, 20, 16);
    let flame = Color::from_rgb8(216, 74, 18);
    let gold = Color::from_rgb8(250, 184, 54);
    let t = heat.clamp(0.0, 1.0);
    if t < 0.33 {
        lerp_color(cool, ember, t / 0.33)
    } else if t < 0.72 {
        lerp_color(ember, flame, (t - 0.33) / 0.39)
    } else {
        lerp_color(flame, gold, (t - 0.72) / 0.28)
    }
}

fn inferno_node_base() -> Color {
    inferno_gradient(0.5)
}

fn inferno_text_bright() -> Color {
    Color::from_rgb8(252, 226, 184)
}

fn inferno_text_muted() -> Color {
    Color::from_rgb8(224, 170, 130)
}

fn beam_theme(_app: &BeamApp) -> Theme {
    Theme::Dark
}

fn app_background_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgb8(16, 7, 9))),
        text_color: Some(inferno_text_bright()),
        ..Default::default()
    }
}

fn panel_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgb8(22, 9, 11))),
        text_color: Some(inferno_text_bright()),
        border: iced::border::rounded(12)
            .color(Color::from_rgb8(73, 30, 28))
            .width(1),
        ..Default::default()
    }
}

fn node_button_style(
    color: Color,
    selected: bool,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let color = match status {
        iced::widget::button::Status::Hovered => lighten(color, 0.10),
        iced::widget::button::Status::Pressed => lighten(color, 0.17),
        _ => color,
    };
    let border_color = if selected {
        Color::from_rgb8(255, 228, 112)
    } else {
        lighten(color, 0.28)
    };
    iced::widget::button::Style {
        background: Some(Background::Color(color)),
        text_color: contrasting_text(color),
        border: iced::border::rounded(9)
            .color(border_color)
            .width(if selected { 2 } else { 1 }),
        shadow: Shadow {
            color: Color::from_rgba(color.r, color.g, color.b, 0.28),
            offset: Vector::new(0.0, 2.0),
            blur_radius: 10.0,
        },
        ..Default::default()
    }
}

fn flow_node_style(color: Color) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(color)),
        text_color: Some(contrasting_text(color)),
        border: iced::border::rounded(9).color(lighten(color, 0.3)).width(1),
        shadow: Shadow {
            color: Color::from_rgba(color.r, color.g, color.b, 0.22),
            offset: Vector::new(0.0, 1.0),
            blur_radius: 8.0,
        },
        ..Default::default()
    }
}

fn cluster_fill_color(_index: usize) -> Color {
    Color::from_rgba8(124, 20, 16, 0.12)
}

fn contrasting_text(background: Color) -> Color {
    let luminance = 0.2126 * background.r + 0.7152 * background.g + 0.0722 * background.b;
    if luminance > 0.58 {
        Color::from_rgb8(26, 14, 9)
    } else {
        inferno_text_bright()
    }
}

fn lerp_color(a: Color, b: Color, amount: f32) -> Color {
    let amount = amount.clamp(0.0, 1.0);
    Color::from_rgba(
        a.r + (b.r - a.r) * amount,
        a.g + (b.g - a.g) * amount,
        a.b + (b.b - a.b) * amount,
        a.a + (b.a - a.a) * amount,
    )
}

fn lighten(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::WHITE, amount)
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    if label.chars().count() <= max_chars {
        return label.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    format!("{}...", label.chars().take(keep).collect::<String>())
}

fn short_type_name<T: ?Sized>() -> String {
    let full = core::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full).to_string()
}

fn short_uuid(id: Uuid) -> String {
    id.to_string().chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use black_hole_flux::sun::{Binary, BlackHole, Unary};
    use black_hole_flux::{CellState, Fusion, Primordium};
    use jungle_sdk::typosaurus::collections::list::{Empty, List};
    use jungle_sdk::Id;
    use typenum::{U0, U1, U2};

    struct TestCell;

    impl Animal for TestCell {
        type Id = Id<U1>;
        type Generation = U0;
        type State = CellState;
        type Seed = ObjectId;
        type Flow = Primordium;
    }

    struct TestFusion;

    impl Animal for TestFusion {
        type Id = Id<U2>;
        type Generation = U0;
        type State = FusionState;
        type Seed = FusionSeed;
        type Flow = Fusion<Primordium>;
    }

    type PortOne = List<(U1, Empty)>;
    type Tail = List<(Unary<U1, TestCell, Empty>, Empty)>;
    type TestGraph = List<(Unary<U0, TestCell, PortOne>, Tail)>;
    type TestSun = <TestGraph as BlackHole>::Sun<Primordium, Primordium>;
    type TestBinaryGraph = List<(Binary<U0, U1, TestFusion, Empty>, Empty)>;
    type TestBinarySun = <TestBinaryGraph as BlackHole>::Sun<Primordium, Primordium>;

    #[test]
    fn extracts_cells_edges_and_spawn_nodes_from_black_hole_sun() {
        let model = BeamModel::build::<TestSun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0, 1]);
        assert_eq!(model.graph.edges, vec![(0, 1)]);
        assert_eq!(model.spawn_runtime_to_cell.get(&1), Some(&0));
        assert_eq!(model.spawn_runtime_to_cell.get(&3), Some(&1));
    }

    #[test]
    fn extracts_binary_cell_ports() {
        let model = BeamModel::build::<TestBinarySun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0]);
        assert_eq!(model.cells[0].ports, vec![0, 1]);
        assert_eq!(model.spawn_runtime_to_cell.get(&1), Some(&0));
    }

    #[test]
    fn decodes_spawn_outputs_as_child_journeys() {
        let first = Uuid::from_u128(1);
        let second = Uuid::from_u128(2);
        let history = vec![
            RunnerOut::EffectSuccessOutput {
                node_id: 99,
                data: vec![0xff],
                uuid: Uuid::nil(),
            },
            RunnerOut::EffectSuccessOutput {
                node_id: 3,
                data: postcard::to_allocvec(&second).unwrap(),
                uuid: Uuid::nil(),
            },
            RunnerOut::EffectSuccessOutput {
                node_id: 1,
                data: postcard::to_allocvec(&first).unwrap(),
                uuid: Uuid::nil(),
            },
        ];
        let spawn_nodes = HashMap::from([(1, 0), (3, 1)]);

        let children = decode_child_journeys(history, &spawn_nodes).unwrap();

        assert_eq!(children, vec![(0, first), (1, second)]);
    }
}
