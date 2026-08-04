//! Visualize Black Hole Sun cell graphs.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole) with the circular `circo`
//! layout. Live views discover the child journey IDs from the parent Sun
//! journey and use each child's update stream to color its cell by activity.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use black_hole_flux::sun::action::{SpawnBinary, SpawnUnary};
use black_hole_flux::sun::{BinarySunStep, NodeIdsFromList, Sun, SunNode, UnarySunStep};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState, ObjectId};
use iced::futures::{self, Stream, StreamExt};
use iced::time::Instant;
use iced::widget::{column, container, text};
use iced::{Background, Color, Element, Font, Length, Shadow, Subscription, Task, Theme, Vector};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{circo_layout, AutoFit, Cluster, Graph, LayoutInput, Sugiyama};
use jungle_sdk::client::JourneyUpdateSubscription;
use jungle_sdk::core::dag::{Dag, LiveDagState};
use jungle_sdk::{
    Action, Animal, AnimalIdValue, JourneyAst, JourneyAstSource, JourneyUpdateEvent, JungleClient,
    RunnerOut,
};
use typenum::Unsigned;
use uuid::Uuid;

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;
const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
const DISCOVERY_INTERVAL: Duration = Duration::from_millis(750);
const COLOR_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);
const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);

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
            title: "Black Hole Sun".to_string(),
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

    /// Render a static Black Hole Sun.
    pub fn view<A>(self) -> iced::Result
    where
        A: Animal + 'static,
        A::Flow: BlackHoleSunFlow,
    {
        run_beam::<A>(self.into_config(), None)
    }

    /// Render a live Black Hole Sun colored by each spawned child journey.
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
    dag: Dag,
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

        Self {
            id,
            ports,
            outgoing_ports,
            animal_name: short_type_name::<A>(),
            dag,
            spawn_action,
        }
    }
}

struct BeamModel {
    cells: Vec<CellDefinition>,
    graph: Graph,
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
    DiscoveryTick,
    ColorTick(Instant),
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
    visual: CellVisualState,
}

impl CellRuntime {
    fn new(cell: &CellDefinition) -> Self {
        let mut live = LiveDagState::default();
        live.bind_model(&cell.dag);
        Self {
            journey_id: None,
            live,
            stream_error: None,
            visual: CellVisualState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CellActivity {
    Idle,
    Processing,
    Propagating,
    Optimizing,
    Failed,
}

impl CellActivity {
    fn from_step_label(label: &str) -> Self {
        if label.starts_with("WaitFor") || matches!(label, "InitRecvId" | "InitFusion") {
            Self::Idle
        } else {
            let label = label.to_ascii_lowercase();
            if label.contains("optimiz") || label.contains("perturb") {
                Self::Optimizing
            } else if label.contains("transmit") || label.contains("propagat") {
                Self::Propagating
            } else {
                Self::Processing
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Processing => "processing",
            Self::Propagating => "propagating",
            Self::Optimizing => "optimizing",
            Self::Failed => "failed",
        }
    }

    fn palette(self) -> (Color, Color) {
        match self {
            Self::Idle => (
                Color::from_rgb8(220, 76, 24),
                Color::from_rgb8(246, 164, 46),
            ),
            Self::Processing => (
                Color::from_rgb8(123, 58, 202),
                Color::from_rgb8(164, 87, 232),
            ),
            Self::Propagating => (
                Color::from_rgb8(238, 161, 35),
                Color::from_rgb8(250, 215, 72),
            ),
            Self::Optimizing => (Color::from_rgb8(202, 42, 67), Color::from_rgb8(238, 72, 57)),
            Self::Failed => (Color::from_rgb8(151, 24, 40), Color::from_rgb8(194, 40, 49)),
        }
    }

    fn color(self, cell_id: u32) -> Color {
        let (low, high) = self.palette();
        lerp_color(low, high, cell_palette_position(cell_id))
    }
}

#[derive(Debug, Clone)]
struct CellVisualState {
    previous: CellActivity,
    current: CellActivity,
    transition_started_at: Option<Instant>,
    pending: Option<CellActivity>,
}

impl Default for CellVisualState {
    fn default() -> Self {
        Self {
            previous: CellActivity::Idle,
            current: CellActivity::Idle,
            transition_started_at: None,
            pending: None,
        }
    }
}

impl CellVisualState {
    fn observe(&mut self, activity: CellActivity, now: Instant) -> bool {
        if activity == self.current {
            self.pending = None;
            return false;
        }

        if self.can_transition(now) {
            self.begin_transition(activity, now);
            true
        } else {
            self.pending = Some(activity);
            false
        }
    }

    fn advance(&mut self, now: Instant) -> bool {
        if !self.can_transition(now) {
            return false;
        }

        let Some(activity) = self.pending.take() else {
            return false;
        };
        if activity == self.current {
            return false;
        }

        self.begin_transition(activity, now);
        true
    }

    fn color(&self, cell_id: u32, now: Instant) -> Color {
        let progress = self
            .transition_started_at
            .map(|started_at| {
                now.saturating_duration_since(started_at).as_secs_f32()
                    / COLOR_FADE_DURATION.as_secs_f32()
            })
            .unwrap_or(1.0);
        lerp_color(
            self.previous.color(cell_id),
            self.current.color(cell_id),
            progress,
        )
    }

    fn is_fading(&self, now: Instant) -> bool {
        self.transition_started_at.is_some_and(|started_at| {
            now.saturating_duration_since(started_at) < COLOR_FADE_DURATION
        })
    }

    fn needs_tick(&self, now: Instant) -> bool {
        self.pending.is_some() || self.is_fading(now)
    }

    fn can_transition(&self, now: Instant) -> bool {
        self.transition_started_at.is_none_or(|started_at| {
            now.saturating_duration_since(started_at) >= MIN_COLOR_STATE_DURATION
        })
    }

    fn begin_transition(&mut self, activity: CellActivity, now: Instant) {
        self.previous = self.current;
        self.current = activity;
        self.transition_started_at = Some(now);
        self.pending = None;
    }
}

struct BeamApp {
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
    cell_runtime: Vec<CellRuntime>,
    discovering: bool,
    color_now: Instant,
}

impl BeamApp {
    fn new<A>(config: BeamConfig, live: Option<LiveConfig>) -> (Self, Task<Message>)
    where
        A: Animal + 'static,
        A::Flow: BlackHoleSunFlow,
    {
        let model = BeamModel::build::<A::Flow>();
        debug_assert!(
            model.errors.is_empty(),
            "invalid Black Hole Sun: {:?}",
            &model.errors
        );
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
                discovering,
                color_now: Instant::now(),
            },
            task,
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
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
            Message::ColorTick(now) => {
                let was_fading = self
                    .cell_runtime
                    .iter()
                    .any(|runtime| runtime.visual.is_fading(self.color_now));
                self.color_now = now;
                let mut transitioned = false;
                for runtime in &mut self.cell_runtime {
                    transitioned |= runtime.visual.advance(now);
                }
                let is_fading = self
                    .cell_runtime
                    .iter()
                    .any(|runtime| runtime.visual.is_fading(now));

                if was_fading || transitioned || is_fading {
                    return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
                }
            }
            Message::ChildrenDiscovered(result) => {
                self.discovering = false;
                if let Ok(children) = result {
                    let now = Instant::now();
                    self.color_now = now;
                    for (index, journey_id) in children {
                        let Some(runtime) = self.cell_runtime.get_mut(index) else {
                            continue;
                        };
                        if runtime.journey_id != Some(journey_id) {
                            runtime.journey_id = Some(journey_id);
                            runtime.live = LiveDagState::default();
                            runtime.live.bind_model(&self.model.cells[index].dag);
                            runtime.stream_error = None;
                            runtime.visual.observe(CellActivity::Idle, now);
                        }
                    }
                    return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
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
                let now = Instant::now();
                self.color_now = now;
                let activity = cell_activity(&self.model.cells[cell_index], runtime);
                runtime.visual.observe(activity, now);

                return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
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
        if self
            .cell_runtime
            .iter()
            .any(|runtime| runtime.visual.needs_tick(self.color_now))
        {
            subscriptions.push(iced::time::every(COLOR_FRAME_INTERVAL).map(Message::ColorTick));
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
        container(self.cell_graph())
            .width(Length::Fill)
            .height(Length::Fill)
            .style(app_background_style)
            .into()
    }

    fn cell_graph(&self) -> Element<'_, Message> {
        let labels = self
            .model
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let activity = self.cell_runtime[index].visual.current;
                (cell.id, (cell.animal_name.clone(), activity))
            })
            .collect::<HashMap<_, _>>();
        let colors = self.cell_colors();
        let colors_for_nodes = colors.clone();
        let colors_for_edges = colors.clone();

        let mut graph =
            Sugiyama::<Message, Theme, iced::Renderer>::new(&self.model.graph, move |node_id| {
                let (animal_name, activity) = labels
                    .get(&node_id)
                    .cloned()
                    .unwrap_or((format!("cell {node_id}"), CellActivity::Idle));
                let color = colors_for_nodes
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| CellActivity::Idle.color(node_id));
                container(
                    column![
                        text(animal_name).size(16).color(contrasting_text(color)),
                        text(format!("cell {node_id} · {}", activity.label()))
                            .size(12)
                            .color(contrasting_text(color).scale_alpha(0.82)),
                    ]
                    .spacing(3),
                )
                .padding([10, 12])
                .style(move |_theme| cell_node_style(color))
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
            .edge_color(move |ctx| {
                let start = colors_for_edges
                    .get(&ctx.edge.0)
                    .copied()
                    .unwrap_or_else(|| CellActivity::Idle.color(ctx.edge.0));
                let end = colors_for_edges
                    .get(&ctx.edge.1)
                    .copied()
                    .unwrap_or_else(|| CellActivity::Idle.color(ctx.edge.1));
                (lighten(start, 0.18), end)
            })
            .stroke_width(1.4)
            .edge_corner_radius(16.0)
            .padding(24)
            .auto_fit(AutoFit::Initial)
            .keep_centered(false);
        if let Some(duration) = self.config.animation_duration {
            graph = graph.animation_duration(duration);
        }
        if let Some(easing) = self.config.animation_easing {
            graph = graph.animation_easing(easing);
        }

        container(graph)
            .width(Length::Fill)
            .height(Length::Fill)
            .clip(true)
            .into()
    }

    fn cell_colors(&self) -> HashMap<u32, Color> {
        self.model
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let runtime = &self.cell_runtime[index];
                (cell.id, runtime.visual.color(cell.id, self.color_now))
            })
            .collect()
    }
}

fn cell_activity(cell: &CellDefinition, runtime: &CellRuntime) -> CellActivity {
    if runtime.stream_error.is_some() || !runtime.live.failed_runtime_ids.is_empty() {
        return CellActivity::Failed;
    }

    cell.dag
        .nodes
        .iter()
        .filter(|node| {
            node.runtime_node_id
                .is_some_and(|id| runtime.live.active_runtime_ids.contains(&id))
                || node
                    .proxy_runtime_ids
                    .iter()
                    .any(|id| runtime.live.active_runtime_ids.contains(id))
        })
        .map(|node| CellActivity::from_step_label(&node.label))
        .max()
        .unwrap_or(CellActivity::Idle)
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

fn black_hole_text() -> Color {
    Color::from_rgb8(252, 226, 184)
}

fn beam_theme(_app: &BeamApp) -> Theme {
    Theme::Dark
}

fn app_background_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::BLACK)),
        text_color: Some(black_hole_text()),
        ..Default::default()
    }
}

fn cell_node_style(color: Color) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(color)),
        text_color: Some(contrasting_text(color)),
        border: iced::border::rounded(9),
        shadow: Shadow {
            color: Color::from_rgba(color.r, color.g, color.b, 0.32),
            offset: Vector::new(0.0, 2.0),
            blur_radius: 10.0,
        },
        ..Default::default()
    }
}

fn contrasting_text(background: Color) -> Color {
    let luminance = 0.2126 * background.r + 0.7152 * background.g + 0.0722 * background.b;
    if luminance > 0.58 {
        Color::from_rgb8(26, 14, 9)
    } else {
        black_hole_text()
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

fn cell_palette_position(cell_id: u32) -> f32 {
    const POSITIONS: [f32; 8] = [0.08, 0.76, 0.31, 0.91, 0.52, 0.18, 0.67, 0.42];
    POSITIONS[cell_id as usize % POSITIONS.len()]
}

fn lighten(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::WHITE, amount)
}

fn short_type_name<T: ?Sized>() -> String {
    let full = core::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full).to_string()
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
    fn uses_black_hole_sun_title_and_activity_palette() {
        assert_eq!(BeamBuilder::default().title, "Black Hole Sun");
        assert_eq!(
            CellActivity::Idle.palette(),
            (
                Color::from_rgb8(220, 76, 24),
                Color::from_rgb8(246, 164, 46)
            )
        );
        assert_eq!(
            CellActivity::Processing.palette(),
            (
                Color::from_rgb8(123, 58, 202),
                Color::from_rgb8(164, 87, 232)
            )
        );
        assert_eq!(
            CellActivity::Propagating.palette(),
            (
                Color::from_rgb8(238, 161, 35),
                Color::from_rgb8(250, 215, 72)
            )
        );
        assert_eq!(
            CellActivity::Optimizing.palette(),
            (Color::from_rgb8(202, 42, 67), Color::from_rgb8(238, 72, 57))
        );
        assert_eq!(
            CellActivity::Failed.palette(),
            (Color::from_rgb8(151, 24, 40), Color::from_rgb8(194, 40, 49))
        );
        assert_ne!(CellActivity::Idle.color(0), CellActivity::Idle.color(1));
        assert!(CellActivity::Idle.color(1).g > CellActivity::Idle.color(0).g);
    }

    #[test]
    fn fades_and_holds_ui_activity_colors() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert_eq!(COLOR_FADE_DURATION, Duration::from_millis(400));
        assert_eq!(MIN_COLOR_STATE_DURATION, Duration::from_secs(1));
        assert!(visual.observe(CellActivity::Processing, start));
        assert_eq!(
            visual.color(0, start),
            CellActivity::Idle.color(0),
            "the fade starts from the previous activity color"
        );
        assert_eq!(
            visual.color(0, start + COLOR_FADE_DURATION / 2),
            lerp_color(
                CellActivity::Idle.color(0),
                CellActivity::Processing.color(0),
                0.5
            ),
            "the color is blended halfway through the fade"
        );
        assert_eq!(
            visual.color(0, start + COLOR_FADE_DURATION),
            CellActivity::Processing.color(0),
            "the fade reaches the new color after 400ms"
        );

        assert!(!visual.observe(CellActivity::Propagating, start + COLOR_FADE_DURATION));
        assert!(!visual.advance(start + MIN_COLOR_STATE_DURATION - Duration::from_millis(1)));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current, CellActivity::Propagating);
    }

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
    fn derives_cell_activity_from_the_active_child_step() {
        let model = BeamModel::build::<TestSun>();
        let cell = &model.cells[0];
        let mut runtime = CellRuntime::new(cell);

        for (label, expected) in [
            ("WaitForPropagationAction", CellActivity::Idle),
            ("QuarkInferStep", CellActivity::Processing),
            ("Transmit", CellActivity::Propagating),
            ("PerturbUp", CellActivity::Optimizing),
            ("Optimize", CellActivity::Optimizing),
        ] {
            let runtime_id = cell
                .dag
                .nodes
                .iter()
                .find(|node| node.label == label)
                .and_then(|node| node.runtime_node_id)
                .unwrap_or_else(|| panic!("missing runtime node for {label}"));
            runtime.live.active_runtime_ids.clear();
            runtime.live.active_runtime_ids.insert(runtime_id);

            assert_eq!(cell_activity(cell, &runtime), expected, "{label}");
        }

        runtime.stream_error = Some("subscription closed".to_string());
        assert_eq!(cell_activity(cell, &runtime), CellActivity::Failed);
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
