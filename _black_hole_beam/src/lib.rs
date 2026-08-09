//! Visualize Black Hole Sun cell graphs.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole), using the circular `circo`
//! layout by default. Live views use the parent Sun animal's Jungle
//! [`Observe`](jungle_sdk::Observe) appearance as the source of graph topology
//! and node phase.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use black_hole_flux::sun::{
    BinarySunStep, NodeIdsFromList, Sun, SunAppearance, SunNode, SunNodeState, SunState,
    UnarySunStep,
};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState, ObjectId};
use iced::time::Instant;
use iced::widget::{column, container, text};
use iced::{Background, Color, Element, Font, Length, Shadow, Subscription, Task, Theme, Vector};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{circo_layout, AutoFit, Cluster, Graph, LayoutInput, Sugiyama};
use jungle_sdk::{Animal, AnimalIdValue, JungleClient, Observe};
use typenum::Unsigned;
use uuid::Uuid;

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;
const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
const APPEARANCE_INTERVAL: Duration = Duration::from_millis(100);
const COLOR_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);
const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);
const MAX_PENDING_PHASES: usize = 4;

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
    layout: BeamLayout,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
}

#[derive(Clone, Copy)]
enum BeamLayout {
    Circo,
    Dot,
}

impl Default for BeamBuilder {
    fn default() -> Self {
        Self {
            title: "Black Hole Sun".to_string(),
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            layout: BeamLayout::Circo,
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
        run_beam(self.into_config(), BeamModel::build::<A::Flow>(), None)
    }

    /// Render a live Black Hole Sun from its Jungle appearance.
    pub fn view_live<A>(self, client: impl JungleClient + 'static, journey_id: Uuid) -> iced::Result
    where
        A: Animal<State = SunState> + Observe<Appearance = SunAppearance> + 'static,
    {
        let live = LiveConfig {
            client: Arc::new(client),
            journey_id,
        };
        run_beam(self.into_config(), BeamModel::empty(), Some(live))
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

    /// Use iced-sugiyama's default layout (`dot`) for node placement.
    pub fn dot_layout(mut self) -> Self {
        self.layout = BeamLayout::Dot;
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
            layout: self.layout,
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
    A: Animal<State = SunState> + Observe<Appearance = SunAppearance> + 'static,
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
pub trait BlackHoleSunFlow: private::DescribeSun {}

impl<T> BlackHoleSunFlow for T where T: private::DescribeSun {}

impl<Generator, Policy> private::DescribeSun for Sun<Generator, Policy> {
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Port, A, Edges, Tail> private::DescribeSun for SunNode<UnarySunStep<Port, A, Edges>, Tail>
where
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = ObjectId> + 'static,
    Edges: NodeIdsFromList,
    Tail: private::DescribeSun,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            Port::U32,
            vec![Port::U32],
            Edges::node_ids(),
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
    A::Flow: FusionFlow,
    Edges: NodeIdsFromList,
    Tail: private::DescribeSun,
{
    fn append_cells(cells: &mut Vec<CellDefinition>) {
        cells.push(CellDefinition::new::<A>(
            PortA::U32,
            vec![PortA::U32, PortB::U32],
            Edges::node_ids(),
        ));
        Tail::append_cells(cells);
    }
}

#[derive(Clone)]
struct BeamConfig {
    title: String,
    width: f32,
    height: f32,
    layout: BeamLayout,
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
    state: SunNodeState,
    state_sequence: u64,
}

impl CellDefinition {
    fn new<A>(id: u32, ports: Vec<u32>, outgoing_ports: Vec<u32>) -> Self
    where
        A: Animal + 'static,
    {
        Self {
            id,
            ports,
            outgoing_ports,
            animal_name: short_type_name::<A>(),
            state: SunNodeState::Idle,
            state_sequence: 0,
        }
    }
}

#[derive(Clone)]
struct BeamModel {
    cells: Vec<CellDefinition>,
    graph: Graph,
    errors: Vec<String>,
}

impl BeamModel {
    fn empty() -> Self {
        Self {
            cells: Vec::new(),
            graph: Graph::new(Vec::new(), Vec::new()),
            errors: Vec::new(),
        }
    }

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
        edges.sort_unstable();
        let graph = Graph::new(nodes, edges);

        if cells.is_empty() {
            errors.push("the Black Hole Sun contains no cells".to_string());
        }

        Self {
            cells,
            graph,
            errors,
        }
    }

    fn from_appearance(appearance: SunAppearance) -> Result<Self, String> {
        if !appearance.finalized {
            return Err("the Black Hole Sun topology is not finalized".to_string());
        }

        let mut errors = Vec::new();
        let mut cells = appearance
            .nodes
            .into_iter()
            .map(|node| CellDefinition {
                id: node.id,
                ports: node.input_ports,
                outgoing_ports: Vec::new(),
                animal_name: node.label,
                state: node.state,
                state_sequence: node.state_sequence,
            })
            .collect::<Vec<_>>();
        cells.sort_by_key(|cell| cell.id);

        let mut node_ids = HashSet::new();
        let mut port_owner = HashMap::new();
        for cell in &cells {
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
        for edge in appearance.edges {
            if !node_ids.contains(&edge.source) {
                errors.push(format!("edge starts at unknown cell {}", edge.source));
                continue;
            }
            if !node_ids.contains(&edge.target) {
                errors.push(format!("edge targets unknown cell {}", edge.target));
                continue;
            }
            if edge.source == edge.target {
                errors.push(format!(
                    "cell {} has a self edge on port {}",
                    edge.source, edge.target_port
                ));
                continue;
            }
            if port_owner.get(&edge.target_port) != Some(&edge.target) {
                errors.push(format!(
                    "edge to cell {} references unowned input port {}",
                    edge.target, edge.target_port
                ));
                continue;
            }
            if seen_edges.insert((edge.source, edge.target)) {
                edges.push((edge.source, edge.target));
            }
        }
        edges.sort_unstable();

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
            errors,
        })
    }
}

fn run_beam(config: BeamConfig, model: BeamModel, live: Option<LiveConfig>) -> iced::Result {
    let title = config.title.clone();
    let width = config.width;
    let height = config.height;
    iced::application(
        move || BeamApp::new(config.clone(), model.clone(), live.clone()),
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
    AppearanceTick,
    AppearanceLoaded(Result<Option<SunAppearance>, String>),
    ColorTick(Instant),
}

trait NodeStateVisual {
    fn label(self) -> &'static str;
    fn palette(self) -> (Color, Color);
    fn color(self, cell_id: u32) -> Color;
}

impl NodeStateVisual for SunNodeState {
    fn label(self) -> &'static str {
        match self {
            SunNodeState::Idle => "idle",
            SunNodeState::Propagation1 => "propagation 1",
            SunNodeState::Propagation2 => "propagation 2",
            SunNodeState::Optimization => "optimization",
        }
    }

    fn palette(self) -> (Color, Color) {
        match self {
            SunNodeState::Idle => (
                Color::from_rgb8(220, 76, 24),
                Color::from_rgb8(246, 164, 46),
            ),
            SunNodeState::Propagation1 => (
                Color::from_rgb8(238, 161, 35),
                Color::from_rgb8(250, 215, 72),
            ),
            SunNodeState::Propagation2 => {
                (Color::from_rgb8(202, 42, 67), Color::from_rgb8(238, 72, 57))
            }
            SunNodeState::Optimization => (
                Color::from_rgb8(123, 58, 202),
                Color::from_rgb8(164, 87, 232),
            ),
        }
    }

    fn color(self, cell_id: u32) -> Color {
        let (low, high) = self.palette();
        lerp_color(low, high, cell_palette_position(cell_id))
    }
}

fn next_phase(phase: SunNodeState) -> SunNodeState {
    match phase {
        SunNodeState::Idle | SunNodeState::Optimization => SunNodeState::Propagation1,
        SunNodeState::Propagation1 => SunNodeState::Propagation2,
        SunNodeState::Propagation2 => SunNodeState::Optimization,
    }
}

fn phase_path(from: SunNodeState, to: SunNodeState) -> Vec<SunNodeState> {
    if from == to {
        return Vec::new();
    }
    if to == SunNodeState::Idle {
        return vec![SunNodeState::Idle];
    }

    let mut path = Vec::new();
    let mut phase = from;
    for _ in 0..4 {
        phase = next_phase(phase);
        path.push(phase);
        if phase == to {
            return path;
        }
    }
    vec![to]
}

fn recent_phase_steps(mut phase: SunNodeState, count: u64) -> Vec<SunNodeState> {
    let retained = count.min(MAX_PENDING_PHASES as u64);
    let mut skipped = count - retained;

    if skipped > 0 && phase == SunNodeState::Idle {
        phase = next_phase(phase);
        skipped -= 1;
    }
    for _ in 0..skipped % 3 {
        phase = next_phase(phase);
    }

    (0..retained)
        .map(|_| {
            phase = next_phase(phase);
            phase
        })
        .collect()
}

#[derive(Debug, Clone)]
struct CellVisualState {
    previous: SunNodeState,
    current: SunNodeState,
    transition_started_at: Option<Instant>,
    pending: VecDeque<SunNodeState>,
    observed_sequence: u64,
}

impl Default for CellVisualState {
    fn default() -> Self {
        Self {
            previous: SunNodeState::Idle,
            current: SunNodeState::Idle,
            transition_started_at: None,
            pending: VecDeque::new(),
            observed_sequence: 0,
        }
    }
}

impl CellVisualState {
    fn observe(&mut self, activity: SunNodeState, sequence: u64, now: Instant) -> bool {
        let latest = self.pending.back().copied().unwrap_or(self.current);
        if sequence < self.observed_sequence {
            return false;
        }

        let path = if sequence > self.observed_sequence {
            let path = recent_phase_steps(latest, sequence - self.observed_sequence);
            self.observed_sequence = sequence;
            if path.last().copied() == Some(activity) {
                path
            } else {
                phase_path(latest, activity)
            }
        } else {
            phase_path(latest, activity)
        };
        if path.is_empty() {
            return false;
        }

        self.pending.extend(path);
        while self.pending.len() > MAX_PENDING_PHASES {
            self.pending.pop_front();
        }
        if self.can_transition(now) {
            return self.begin_next_transition(now);
        }
        false
    }

    fn advance(&mut self, now: Instant) -> bool {
        if !self.can_transition(now) {
            return false;
        }
        self.begin_next_transition(now)
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
        !self.pending.is_empty() || self.is_fading(now)
    }

    fn can_transition(&self, now: Instant) -> bool {
        self.transition_started_at.is_none_or(|started_at| {
            now.saturating_duration_since(started_at) >= MIN_COLOR_STATE_DURATION
        })
    }

    fn begin_next_transition(&mut self, now: Instant) -> bool {
        let Some(activity) = self.pending.pop_front() else {
            return false;
        };
        if activity == self.current {
            return self.begin_next_transition(now);
        }
        self.previous = self.current;
        self.current = activity;
        self.transition_started_at = Some(now);
        true
    }
}

struct BeamApp {
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
    visuals: HashMap<u32, CellVisualState>,
    appearance_loading: bool,
    appearance_error: Option<String>,
    color_now: Instant,
}

impl BeamApp {
    fn new(
        config: BeamConfig,
        model: BeamModel,
        live: Option<LiveConfig>,
    ) -> (Self, Task<Message>) {
        debug_assert!(
            model.errors.is_empty(),
            "invalid Black Hole Sun: {:?}",
            &model.errors
        );
        let now = Instant::now();
        let mut visuals = HashMap::new();
        for cell in &model.cells {
            let mut visual = CellVisualState::default();
            visual.observe(cell.state, cell.state_sequence, now);
            visuals.insert(cell.id, visual);
        }
        let appearance_loading = live.is_some();
        let task = live
            .as_ref()
            .map(|live| appearance_task(live.clone()))
            .unwrap_or_else(Task::none);

        (
            Self {
                config,
                model,
                live,
                visuals,
                appearance_loading,
                appearance_error: None,
                color_now: now,
            },
            task,
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::AppearanceTick => {
                if !self.appearance_loading {
                    self.appearance_loading = true;
                    if let Some(live) = self.live.clone() {
                        return appearance_task(live);
                    }
                }
            }
            Message::AppearanceLoaded(result) => {
                self.appearance_loading = false;
                match result {
                    Ok(Some(appearance)) if appearance.finalized => {
                        match BeamModel::from_appearance(appearance) {
                            Ok(model) => {
                                let now = Instant::now();
                                self.color_now = now;
                                let node_ids = model
                                    .cells
                                    .iter()
                                    .map(|cell| cell.id)
                                    .collect::<HashSet<_>>();
                                self.visuals.retain(|node_id, _| node_ids.contains(node_id));
                                for cell in &model.cells {
                                    self.visuals.entry(cell.id).or_default().observe(
                                        cell.state,
                                        cell.state_sequence,
                                        now,
                                    );
                                }
                                self.model = model;
                                self.appearance_error = None;
                                return iced_sugiyama::force_review(iced_sugiyama::Id::new(
                                    CELL_GRAPH_ID,
                                ));
                            }
                            Err(error) => self.appearance_error = Some(error),
                        }
                    }
                    Ok(Some(_)) | Ok(None) => {}
                    Err(error) => self.appearance_error = Some(error),
                }
            }
            Message::ColorTick(now) => {
                let was_fading = self
                    .visuals
                    .values()
                    .any(|visual| visual.is_fading(self.color_now));
                self.color_now = now;
                let mut transitioned = false;
                for visual in self.visuals.values_mut() {
                    transitioned |= visual.advance(now);
                }
                let is_fading = self.visuals.values().any(|visual| visual.is_fading(now));

                if was_fading || transitioned || is_fading {
                    return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
                }
            }
        }

        Task::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        let mut subscriptions = Vec::new();
        if self.live.is_some() && !self.appearance_loading {
            subscriptions
                .push(iced::time::every(APPEARANCE_INTERVAL).map(|_| Message::AppearanceTick));
        }
        if self
            .visuals
            .values()
            .any(|visual| visual.needs_tick(self.color_now))
        {
            subscriptions.push(iced::time::every(COLOR_FRAME_INTERVAL).map(Message::ColorTick));
        }

        Subscription::batch(subscriptions)
    }

    fn view(&self) -> Element<'_, Message> {
        let content: Element<'_, Message> = if self.model.cells.is_empty() {
            text(
                self.appearance_error
                    .as_deref()
                    .unwrap_or("Waiting for Black Hole Sun appearance…"),
            )
            .size(16)
            .color(black_hole_text())
            .into()
        } else {
            self.cell_graph()
        };
        container(content)
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
            .map(|cell| {
                let activity = self
                    .visuals
                    .get(&cell.id)
                    .map(|visual| visual.current)
                    .unwrap_or(cell.state);
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
                    .unwrap_or((format!("cell {node_id}"), SunNodeState::Idle));
                let color = colors_for_nodes
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| SunNodeState::Idle.color(node_id));
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
            .id(iced_sugiyama::Id::new(CELL_GRAPH_ID));

        if matches!(self.config.layout, BeamLayout::Circo) {
            graph = graph.layout_fn(|input| {
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
            });
        }

        graph = graph
            .edge_color(move |ctx| {
                let start = colors_for_edges
                    .get(&ctx.edge.0)
                    .copied()
                    .unwrap_or_else(|| SunNodeState::Idle.color(ctx.edge.0));
                let end = colors_for_edges
                    .get(&ctx.edge.1)
                    .copied()
                    .unwrap_or_else(|| SunNodeState::Idle.color(ctx.edge.1));
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
            .map(|cell| {
                let color = self
                    .visuals
                    .get(&cell.id)
                    .map(|visual| visual.color(cell.id, self.color_now))
                    .unwrap_or_else(|| cell.state.color(cell.id));
                (cell.id, color)
            })
            .collect()
    }
}

fn appearance_task(live: LiveConfig) -> Task<Message> {
    Task::perform(fetch_appearance(live), Message::AppearanceLoaded)
}

async fn fetch_appearance(live: LiveConfig) -> Result<Option<SunAppearance>, String> {
    let bytes = live
        .client
        .animal_appearance(live.journey_id)
        .await
        .map_err(|error| error.to_string())?;
    bytes
        .map(|bytes| {
            postcard::from_bytes::<SunAppearance>(&bytes)
                .map_err(|error| format!("could not decode Sun appearance: {error}"))
        })
        .transpose()
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
    use black_hole_flux::sun::{Binary, BlackHole, SunEdgeAppearance, SunNodeAppearance, Unary};
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
        assert!(matches!(BeamBuilder::default().layout, BeamLayout::Circo));
        assert!(matches!(
            BeamBuilder::new().dot_layout().layout,
            BeamLayout::Dot
        ));
        assert_eq!(
            SunNodeState::Idle.palette(),
            (
                Color::from_rgb8(220, 76, 24),
                Color::from_rgb8(246, 164, 46)
            )
        );
        assert_eq!(
            SunNodeState::Propagation1.palette(),
            (
                Color::from_rgb8(238, 161, 35),
                Color::from_rgb8(250, 215, 72)
            )
        );
        assert_eq!(
            SunNodeState::Propagation2.palette(),
            (Color::from_rgb8(202, 42, 67), Color::from_rgb8(238, 72, 57))
        );
        assert_eq!(
            SunNodeState::Optimization.palette(),
            (
                Color::from_rgb8(123, 58, 202),
                Color::from_rgb8(164, 87, 232)
            )
        );
        assert_ne!(SunNodeState::Idle.color(0), SunNodeState::Idle.color(1));
        assert!(SunNodeState::Idle.color(1).g > SunNodeState::Idle.color(0).g);
    }

    #[test]
    fn fades_and_holds_ui_activity_colors() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert_eq!(COLOR_FADE_DURATION, Duration::from_millis(400));
        assert_eq!(MIN_COLOR_STATE_DURATION, Duration::from_secs(1));
        assert!(visual.observe(SunNodeState::Propagation1, 1, start));
        assert_eq!(
            visual.color(0, start),
            SunNodeState::Idle.color(0),
            "the fade starts from the previous activity color"
        );
        assert_eq!(
            visual.color(0, start + COLOR_FADE_DURATION / 2),
            lerp_color(
                SunNodeState::Idle.color(0),
                SunNodeState::Propagation1.color(0),
                0.5
            ),
            "the color is blended halfway through the fade"
        );
        assert_eq!(
            visual.color(0, start + COLOR_FADE_DURATION),
            SunNodeState::Propagation1.color(0),
            "the fade reaches the new color after 400ms"
        );

        assert!(!visual.observe(SunNodeState::Propagation2, 2, start + COLOR_FADE_DURATION));
        assert!(!visual.advance(start + MIN_COLOR_STATE_DURATION - Duration::from_millis(1)));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current, SunNodeState::Propagation2);
    }

    #[test]
    fn queues_intermediate_phases_when_an_appearance_jumps() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 2, start));
        assert_eq!(visual.current, SunNodeState::Propagation1);
        assert_eq!(visual.pending, VecDeque::from([SunNodeState::Propagation2]));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current, SunNodeState::Propagation2);
    }

    #[test]
    fn replays_an_epoch_when_snapshots_repeat_the_same_phase() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 2, start));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current, SunNodeState::Propagation2);
        assert!(visual.pending.is_empty());

        assert!(visual.observe(
            SunNodeState::Propagation2,
            5,
            start + MIN_COLOR_STATE_DURATION * 2
        ));
        assert_eq!(visual.current, SunNodeState::Optimization);
        assert_eq!(
            visual.pending,
            VecDeque::from([SunNodeState::Propagation1, SunNodeState::Propagation2])
        );
    }

    #[test]
    fn bounds_pending_phases_when_epochs_outpace_the_animation() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();
        for (sequence, phase) in [
            (2, SunNodeState::Propagation2),
            (3, SunNodeState::Optimization),
            (4, SunNodeState::Propagation1),
            (5, SunNodeState::Propagation2),
            (6, SunNodeState::Optimization),
            (7, SunNodeState::Propagation1),
        ] {
            visual.observe(phase, sequence, start);
        }

        assert!(visual.pending.len() <= MAX_PENDING_PHASES);
    }

    #[test]
    fn extracts_static_cells_and_edges_from_black_hole_sun() {
        let model = BeamModel::build::<TestSun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0, 1]);
        assert_eq!(model.graph.edges, vec![(0, 1)]);
    }

    #[test]
    fn extracts_binary_cell_ports() {
        let model = BeamModel::build::<TestBinarySun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0]);
        assert_eq!(model.cells[0].ports, vec![0, 1]);
    }

    #[test]
    fn builds_live_model_from_serialized_sun_appearance() {
        let appearance = SunAppearance {
            finalized: true,
            nodes: vec![
                SunNodeAppearance {
                    id: 2,
                    label: "Fusion".to_string(),
                    input_ports: vec![2, 3],
                    state: SunNodeState::Optimization,
                    state_sequence: 3,
                },
                SunNodeAppearance {
                    id: 0,
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Propagation1,
                    state_sequence: 1,
                },
            ],
            edges: vec![
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 3,
                },
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 2,
                },
            ],
        };
        let bytes = postcard::to_allocvec(&appearance).unwrap();
        let decoded = postcard::from_bytes::<SunAppearance>(&bytes).unwrap();
        let model = BeamModel::from_appearance(decoded).unwrap();

        assert_eq!(model.graph.nodes, vec![0, 2]);
        assert_eq!(
            model.graph.edges,
            vec![(0, 2)],
            "parallel destination-port edges collapse only for rendering"
        );
        assert_eq!(model.cells[0].animal_name, "Root");
        assert_eq!(model.cells[0].state, SunNodeState::Propagation1);
        assert_eq!(model.cells[0].state_sequence, 1);
        assert_eq!(model.cells[1].state, SunNodeState::Optimization);
        assert_eq!(model.cells[1].state_sequence, 3);
    }

    #[test]
    fn rejects_appearance_edges_with_the_wrong_destination_port() {
        let appearance = SunAppearance {
            finalized: true,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                },
                SunNodeAppearance {
                    id: 1,
                    label: "Sink".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 1,
                target_port: 9,
            }],
        };

        let error = BeamModel::from_appearance(appearance)
            .err()
            .expect("appearance should be rejected");
        assert!(error.contains("unowned input port 9"));
    }
}
