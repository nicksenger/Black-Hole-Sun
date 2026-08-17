//! Visualize Black Hole Sun cell graphs.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole), using the circular `circo`
//! layout by default. Live views use the parent Sun animal's Jungle
//! [`Observe`](jungle_sdk::Observe) appearance as the source of graph topology
//! and node phase.
//!
//! With the `av` feature enabled, an AV1/Opus Matroska video at
//! `~/.black-hole-sun/black-hole-sun.mkv` (if present) plays as a
//! semi-transparent overlay on top of the Black Hole Sun graph.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "av")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::sun::{
    BinarySunStep, NodeIdsFromList, Sun, SunAppearance, SunNode, SunNodeState, SunState,
    UnarySunStep,
};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState, Ray};
#[cfg(feature = "av")]
use directories_next::BaseDirs;
use iced::mouse;
use iced::time::Instant;
use iced::widget::canvas::{self, Path};
use iced::widget::{button, column, container, mouse_area, opaque, row, stack, text};
use iced::{
    Background, Color, Element, Font, Length, Point, Rectangle, Shadow, Subscription, Task, Theme,
    Vector,
};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{
    circo_layout, AutoFit, Cluster, EdgeEndpointKind, Graph, LayoutInput, Sugiyama,
};
use jungle_sdk::{Animal, AnimalIdValue, JourneyAstSource, JungleClient, Observe};
use jungle_vision::{
    AnyAnimal, DefaultTheme, EjectedViewer, EjectedViewerMessage, JungleViewerBuilder,
};
#[cfg(feature = "av")]
use tracing::{info, warn};
use typenum::Unsigned;
use uuid::Uuid;

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;
const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
const DOT_VERTEX_SPACING: f64 = 128.0;
const EDGE_STROKE_WIDTH: f32 = 2.4;
const APPEARANCE_INTERVAL: Duration = Duration::from_millis(200);
const COLOR_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_TRANSITION_POLL_INTERVAL: Duration = Duration::from_millis(100);
const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);
const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);
const MAX_PENDING_PHASES: usize = 4;
#[cfg(feature = "av")]
const AV_OVERLAY_OPACITY: f32 = 0.25;
type JungleSubpanelViewer = EjectedViewer<DefaultTheme, AnyAnimal>;

#[derive(Clone)]
struct SubpanelConfig {
    animal_label: String,
    title: String,
    build_viewer: fn(SharedJungleClient, Uuid) -> JungleSubpanelViewer,
}

fn build_subpanel_viewer<A>(client: SharedJungleClient, journey_id: Uuid) -> JungleSubpanelViewer
where
    A: Animal + 'static,
    A::Flow: JourneyAstSource,
{
    JungleViewerBuilder::new().eject_live_animal::<A, SharedJungleClient>(client, journey_id)
}

#[derive(Debug, Clone, Copy)]
enum EdgeEndpointGlyphKind {
    NormalArrow,
}

#[derive(Debug, Clone, Copy)]
struct EdgeEndpointGlyph {
    kind: EdgeEndpointGlyphKind,
    color: Color,
    angle_radians: f32,
}

impl EdgeEndpointGlyph {
    fn size(self) -> f32 {
        match self.kind {
            EdgeEndpointGlyphKind::NormalArrow => 20.0,
        }
    }
}

impl<Message, Theme, Renderer> canvas::Program<Message, Theme, Renderer> for EdgeEndpointGlyph
where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let anchor = frame.center();

        match self.kind {
            EdgeEndpointGlyphKind::NormalArrow => {
                let arrow = Path::new(|path| {
                    path.move_to(Point::new(0.0, 0.0));
                    path.line_to(Point::new(-10.0, 4.0));
                    path.line_to(Point::new(-7.25, 0.0));
                    path.line_to(Point::new(-10.0, -4.0));
                    path.close();
                });

                frame.with_save(|frame| {
                    frame.translate(Vector::new(anchor.x, anchor.y));
                    frame.rotate(self.angle_radians);
                    frame.fill(&arrow, self.color);
                });
            }
        }

        vec![frame.into_geometry()]
    }
}

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
///     .register_subpanel_animal::<MyChildAnimal>()
///     .view_live::<MyBlackHoleAnimal>(client, journey_id)
/// ```
#[derive(Clone)]
pub struct BeamBuilder {
    title: String,
    width: f32,
    height: f32,
    layout: BeamLayout,
    subpanel_animals: Vec<SubpanelConfig>,
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
            subpanel_animals: Vec::new(),
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
        A: BlackHoleSunAnimal + 'static,
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

    /// Register an animal type that can be shown in a node-click subpanel.
    ///
    /// In live mode, clicking a node whose animal matches one of these
    /// registrations opens that node's journey in the subpanel overlay,
    /// replacing any child flow that is currently open. Only one child
    /// flow is shown at a time.
    pub fn register_subpanel_animal<A>(mut self) -> Self
    where
        A: Animal + 'static,
        A::Flow: JourneyAstSource,
    {
        let animal_label = short_type_name::<A>();
        if self
            .subpanel_animals
            .iter()
            .all(|registered| registered.animal_label != animal_label)
        {
            self.subpanel_animals.push(SubpanelConfig {
                animal_label: animal_label.clone(),
                title: animal_label,
                build_viewer: build_subpanel_viewer::<A>,
            });
        }
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
            subpanel_animals: self.subpanel_animals,
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
    A: BlackHoleSunAnimal + 'static,
{
    BeamBuilder::new().view_live::<A>(client, journey_id)
}

/// Marker for Sun animals whose runtime state is `SunState<S>`.
pub trait BlackHoleSunAnimal: Animal + Observe<Appearance = SunAppearance> {}

impl<A, S> BlackHoleSunAnimal for A where
    A: Animal<State = SunState<S>> + Observe<Appearance = SunAppearance>
{
}

mod private {
    pub(crate) trait DescribeSun {
        fn append_cells(cells: &mut Vec<super::CellDefinition>);
    }
}

/// Marker for the structural flow produced by
/// `<Graph as BlackHole>::Sun<Generator, Policy, S, N>`.
///
/// The trait is sealed and is only implemented for the `SunNode<…>` chain
/// emitted by [`BlackHole`](black_hole_flux::sun::BlackHole).
#[allow(private_bounds)]
pub trait BlackHoleSunFlow: private::DescribeSun {}

impl<T> BlackHoleSunFlow for T where T: private::DescribeSun {}

impl<Generator, Policy, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for Sun<Generator, Policy, S, GRADIENT_ACCUMULATION_STEPS>
{
    fn append_cells(_cells: &mut Vec<CellDefinition>) {}
}

impl<Port, A, Edges, Tail, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for SunNode<UnarySunStep<Port, A, Edges, S, GRADIENT_ACCUMULATION_STEPS>, Tail>
where
    Port: Unsigned,
    A: Animal<Id: AnimalIdValue, Generation: Unsigned, Seed = black_hole_flux::CellInit> + 'static,
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

impl<PortA, PortB, A, Edges, Tail, S, const GRADIENT_ACCUMULATION_STEPS: usize> private::DescribeSun
    for SunNode<BinarySunStep<PortA, PortB, A, Edges, S, GRADIENT_ACCUMULATION_STEPS>, Tail>
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
    subpanel_animals: Vec<SubpanelConfig>,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
}

#[derive(Clone)]
struct LiveConfig {
    client: Arc<dyn JungleClient>,
    journey_id: Uuid,
}

#[derive(Clone)]
struct SharedJungleClient {
    inner: Arc<dyn JungleClient>,
}

impl SharedJungleClient {
    fn new(inner: Arc<dyn JungleClient>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl JungleClient for SharedJungleClient {
    async fn spawn<A>(
        &self,
        _seed: &A::Seed,
    ) -> Result<jungle_sdk::JourneyHandle, jungle_sdk::ExecutorError>
    where
        Self: Sized,
        A: jungle_sdk::SpawnableAnimal,
        A::Seed: Sync,
    {
        Err(jungle_sdk::ExecutorError::ClientTransport(
            "shared beam client does not support spawn".to_string(),
        ))
    }

    async fn journey_history(
        &self,
        id: Uuid,
    ) -> Result<Vec<jungle_sdk::RunnerOut>, jungle_sdk::ExecutorError> {
        self.inner.journey_history(id).await
    }

    async fn journey_replay_page(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
        snapshot_end_sequence_id: Option<u64>,
        limit: u32,
    ) -> Result<jungle_sdk::JourneyReplayPage, jungle_sdk::ExecutorError> {
        self.inner
            .journey_replay_page(
                journey_id,
                after_sequence_id,
                snapshot_end_sequence_id,
                limit,
            )
            .await
    }

    async fn list_journeys(
        &self,
        namespace: String,
    ) -> Result<Vec<jungle_sdk::JourneyRecord>, jungle_sdk::ExecutorError> {
        self.inner.list_journeys(namespace).await
    }

    async fn subscribe_step_updates(
        &self,
        journey_id: Uuid,
        after_sequence_id: Option<u64>,
    ) -> Result<jungle_sdk::client::JourneyUpdateSubscription, jungle_sdk::ExecutorError> {
        self.inner
            .subscribe_step_updates(journey_id, after_sequence_id)
            .await
    }

    async fn journey_details(
        &self,
        id: Uuid,
    ) -> Result<jungle_sdk::JourneyStatus, jungle_sdk::ExecutorError> {
        self.inner.journey_details(id).await
    }

    async fn animal_appearance(
        &self,
        id: Uuid,
    ) -> Result<Option<Vec<u8>>, jungle_sdk::ExecutorError> {
        self.inner.animal_appearance(id).await
    }

    async fn animal_appearance_update(
        &self,
        id: Uuid,
        data: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.animal_appearance_update(id, data).await
    }

    async fn perturb_animal(
        &self,
        id: Uuid,
        payload: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.perturb_animal(id, payload).await
    }

    async fn claim_animal_perturbation(
        &self,
        id: Uuid,
    ) -> Result<Option<jungle_sdk::ClaimedPerturbable>, jungle_sdk::ExecutorError> {
        self.inner.claim_animal_perturbation(id).await
    }

    async fn ack_animal_perturbation(
        &self,
        id: Uuid,
        perturbation_id: u64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .ack_animal_perturbation(id, perturbation_id)
            .await
    }

    async fn heartbeat_journey_lease(
        &self,
        journey_id: Uuid,
        owner_id: Uuid,
        lease_ttl_ms: i64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .heartbeat_journey_lease(journey_id, owner_id, lease_ttl_ms)
            .await
    }

    async fn poll_owner_wake(
        &self,
        owner_id: Uuid,
    ) -> Result<Option<jungle_sdk::OwnerWake>, jungle_sdk::ExecutorError> {
        self.inner.poll_owner_wake(owner_id).await
    }

    async fn schedule_sleep_timer(
        &self,
        journey_id: Uuid,
        timer_id: Uuid,
        wake_at_unix_ms: i64,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .schedule_sleep_timer(journey_id, timer_id, wake_at_unix_ms)
            .await
    }

    async fn complete_journey(&self, id: Uuid) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.complete_journey(id).await
    }

    async fn dead_journey(&self, id: Uuid) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.dead_journey(id).await
    }

    async fn poll_timers(&self) -> Result<Option<()>, jungle_sdk::ExecutorError> {
        self.inner.poll_timers().await
    }

    async fn poll_work(
        &self,
        supported_animals: Vec<jungle_sdk::SupportedAnimal>,
    ) -> Result<Option<jungle_sdk::Work>, jungle_sdk::ExecutorError> {
        self.inner.poll_work(supported_animals).await
    }

    async fn wait_for_worker_wake(
        &self,
        owner_id: Uuid,
        supported_animals: Vec<jungle_sdk::SupportedAnimal>,
        timeout: Duration,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner
            .wait_for_worker_wake(owner_id, supported_animals, timeout)
            .await
    }

    async fn effect_input(
        &self,
        id: Uuid,
        node_id: u32,
        input: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_input(id, node_id, input).await
    }

    async fn effect_success_output(
        &self,
        id: Uuid,
        node_id: u32,
        output: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_success_output(id, node_id, output).await
    }

    async fn effect_failure_output(
        &self,
        id: Uuid,
        node_id: u32,
        err: Vec<u8>,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.effect_failure_output(id, node_id, err).await
    }

    async fn submit_history_event(
        &self,
        event: jungle_sdk::RunnerOut,
    ) -> Result<(), jungle_sdk::ExecutorError> {
        self.inner.submit_history_event(event).await
    }
}

struct SubpanelState {
    node_id: u32,
    title: String,
    journey_id: Uuid,
    viewer: JungleSubpanelViewer,
}

#[derive(Clone)]
struct CellDefinition {
    id: u32,
    journey_id: Uuid,
    ports: Vec<u32>,
    outgoing_ports: Vec<u32>,
    animal_name: String,
    state: SunNodeState,
    state_sequence: u64,
    grad_step: usize,
    grad_steps: usize,
    frozen: Option<bool>,
}

impl CellDefinition {
    fn new<A>(id: u32, ports: Vec<u32>, outgoing_ports: Vec<u32>) -> Self
    where
        A: Animal + 'static,
    {
        Self {
            id,
            journey_id: Uuid::nil(),
            ports,
            outgoing_ports,
            animal_name: short_type_name::<A>(),
            state: SunNodeState::Idle,
            state_sequence: 0,
            grad_step: 1,
            grad_steps: 1,
            frozen: None,
        }
    }
}

#[derive(Clone)]
struct BeamModel {
    cells: Vec<CellDefinition>,
    graph: Graph,
    grad_steps: usize,
    errors: Vec<String>,
}

impl BeamModel {
    fn empty() -> Self {
        Self {
            cells: Vec::new(),
            graph: Graph::new(Vec::new(), Vec::new()),
            grad_steps: 1,
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
            grad_steps: 1,
            errors,
        }
    }

    fn from_appearance(
        appearance: SunAppearance,
        child_rays: &HashMap<Uuid, Ray>,
    ) -> Result<Self, String> {
        if !appearance.finalized {
            return Err("the Black Hole Sun topology is not finalized".to_string());
        }
        let grad_steps = appearance.grad_steps.max(1);

        let mut errors = Vec::new();
        let mut cells = appearance
            .nodes
            .into_iter()
            .map(|node| CellDefinition {
                id: node.id,
                journey_id: node.journey_id,
                ports: node.input_ports,
                outgoing_ports: Vec::new(),
                animal_name: animal_label_key(&node.label),
                state: node.state,
                state_sequence: node.state_sequence,
                grad_step: node.grad_step.clamp(1, grad_steps),
                grad_steps,
                frozen: child_rays.get(&node.journey_id).map(|ray| ray.frozen),
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
            grad_steps,
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
    AppearanceLoaded(Result<Option<LiveAppearanceSnapshot>, String>),
    ColorTick(Instant),
    NodeSelected(u32),
    CloseSubpanel,
    Subpanel(EjectedViewerMessage),
    SubpanelOverlayPointerEvent,
    #[cfg(feature = "av")]
    Av(iced_av1::widget::Message),
}

#[derive(Debug, Clone)]
struct LiveAppearanceSnapshot {
    appearance: SunAppearance,
    child_rays: HashMap<Uuid, Ray>,
}

trait NodeStateVisual {
    fn label(self) -> &'static str;
}

impl NodeStateVisual for SunNodeState {
    fn label(self) -> &'static str {
        match self {
            SunNodeState::Idle => "idle",
            SunNodeState::Propagation1 => "propagation 1",
            SunNodeState::Propagation2 => "propagation 2",
            SunNodeState::Optimization => "potentiation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodeProgress {
    state: SunNodeState,
    grad_step: usize,
}

impl NodeProgress {
    const fn idle() -> Self {
        Self {
            state: SunNodeState::Idle,
            grad_step: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NodeStyleColors {
    body: Color,
    border: Color,
    text: Color,
}

fn default_grad_step_for_state(state: SunNodeState, grad_steps: usize) -> usize {
    match state {
        SunNodeState::Optimization => grad_steps.max(1),
        _ => 1,
    }
}

fn node_phase_progress(grad_step: usize, grad_steps: usize) -> f32 {
    let grad_steps = grad_steps.max(1);
    let step = grad_step.clamp(1, grad_steps);
    step as f32 / grad_steps as f32
}

fn node_style_colors(
    state: SunNodeState,
    grad_step: usize,
    grad_steps: usize,
    frozen: Option<bool>,
) -> NodeStyleColors {
    let idle_orange = Color::from_rgb8(228, 108, 30);
    let bright_yellow = Color::from_rgb8(255, 233, 68);
    let deep_crimson = Color::from_rgb8(195, 24, 41);

    if state == SunNodeState::Optimization {
        if frozen == Some(true) {
            let body = Color::from_rgb8(94, 122, 214);
            let border = Color::from_rgb8(154, 92, 232);
            return NodeStyleColors {
                body,
                border,
                text: contrasting_text(body),
            };
        }
        return NodeStyleColors {
            body: Color::from_rgb8(255, 255, 255),
            border: Color::from_rgb8(255, 120, 18),
            text: Color::from_rgb8(18, 12, 8),
        };
    }

    let body = match state {
        SunNodeState::Idle => idle_orange,
        SunNodeState::Propagation1 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            lerp_color(idle_orange, bright_yellow, progress)
        }
        SunNodeState::Propagation2 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            lerp_color(idle_orange, deep_crimson, progress)
        }
        SunNodeState::Optimization => unreachable!("optimization is handled above"),
    };
    NodeStyleColors {
        body,
        border: lighten(body, 0.18),
        text: contrasting_text(body),
    }
}

fn displayed_grad_step(state: SunNodeState, observed: NodeProgress, grad_steps: usize) -> usize {
    if state == observed.state {
        return observed.grad_step.clamp(1, grad_steps.max(1));
    }
    default_grad_step_for_state(state, grad_steps.max(1))
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
    previous: NodeProgress,
    current: NodeProgress,
    transition_started_at: Option<Instant>,
    pending: VecDeque<NodeProgress>,
    observed_sequence: u64,
    latest_frozen: Option<bool>,
    optimization_frozen: Option<bool>,
}

impl Default for CellVisualState {
    fn default() -> Self {
        Self {
            previous: NodeProgress::idle(),
            current: NodeProgress::idle(),
            transition_started_at: None,
            pending: VecDeque::new(),
            observed_sequence: 0,
            latest_frozen: None,
            optimization_frozen: None,
        }
    }
}

impl CellVisualState {
    fn observe(
        &mut self,
        activity: SunNodeState,
        grad_step: usize,
        grad_steps: usize,
        sequence: u64,
        frozen: Option<bool>,
        now: Instant,
    ) -> bool {
        let grad_steps = grad_steps.max(1);
        let latest = self.pending.back().copied().unwrap_or(self.current);
        let observed = NodeProgress {
            state: activity,
            grad_step: grad_step.clamp(1, grad_steps),
        };
        if sequence < self.observed_sequence {
            return false;
        }
        self.latest_frozen = frozen;

        let path = if sequence > self.observed_sequence {
            let path = recent_phase_steps(latest.state, sequence - self.observed_sequence);
            self.observed_sequence = sequence;
            if path.last().copied() == Some(activity) {
                path
            } else {
                phase_path(latest.state, activity)
            }
        } else {
            phase_path(latest.state, activity)
        };
        if path.is_empty() {
            if latest != observed {
                self.pending.push_back(observed);
            } else {
                return false;
            }
        } else {
            let path_len = path.len();
            for (index, state) in path.into_iter().enumerate() {
                let progress = if index + 1 == path_len {
                    observed
                } else {
                    NodeProgress {
                        state,
                        grad_step: default_grad_step_for_state(state, grad_steps),
                    }
                };
                self.pending.push_back(progress);
            }
        }
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

    fn style(&self, grad_steps: usize, frozen: Option<bool>, now: Instant) -> NodeStyleColors {
        let progress = self
            .transition_started_at
            .map(|started_at| {
                now.saturating_duration_since(started_at).as_secs_f32()
                    / COLOR_FADE_DURATION.as_secs_f32()
            })
            .unwrap_or(1.0);
        let previous_frozen = self.frozen_for_state(self.previous.state, frozen);
        let current_frozen = self.frozen_for_state(self.current.state, frozen);
        let previous = node_style_colors(
            self.previous.state,
            self.previous.grad_step,
            grad_steps,
            previous_frozen,
        );
        let current = node_style_colors(
            self.current.state,
            self.current.grad_step,
            grad_steps,
            current_frozen,
        );
        NodeStyleColors {
            body: lerp_color(previous.body, current.body, progress),
            border: lerp_color(previous.border, current.border, progress),
            text: lerp_color(previous.text, current.text, progress),
        }
    }

    fn frozen_for_state(&self, state: SunNodeState, fallback: Option<bool>) -> Option<bool> {
        if state == SunNodeState::Optimization {
            return self.optimization_frozen.or(self.latest_frozen).or(fallback);
        }
        fallback
    }

    fn is_fading(&self, now: Instant) -> bool {
        self.transition_started_at.is_some_and(|started_at| {
            now.saturating_duration_since(started_at) < COLOR_FADE_DURATION
        })
    }

    fn needs_color_frame(&self, now: Instant) -> bool {
        self.is_fading(now)
    }

    fn needs_transition_poll(&self, now: Instant) -> bool {
        !self.pending.is_empty() && !self.is_fading(now)
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
        if self.current.state == SunNodeState::Propagation1
            && activity.state == SunNodeState::Propagation2
        {
            self.optimization_frozen = self.latest_frozen;
        }
        self.previous = self.current;
        self.current = activity;
        self.transition_started_at = Some(now);
        true
    }
}

fn model_display_changed(current: &BeamModel, next: &BeamModel) -> bool {
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
                    || current.grad_step != next.grad_step
                    || current.grad_steps != next.grad_steps
                    || current.frozen != next.frozen
            })
}

struct BeamApp {
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
    subpanel: Option<SubpanelState>,
    visuals: HashMap<u32, CellVisualState>,
    appearance_loading: bool,
    appearance_error: Option<String>,
    subpanel_notice: Option<String>,
    color_now: Instant,
    #[cfg(feature = "av")]
    video_overlay: Option<iced_av1::widget::State>,
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
            visual.observe(
                cell.state,
                cell.grad_step,
                cell.grad_steps,
                cell.state_sequence,
                cell.frozen,
                now,
            );
            visuals.insert(cell.id, visual);
        }
        let appearance_loading = live.is_some();
        let mut tasks = vec![live
            .as_ref()
            .map(|live| appearance_task(live.clone()))
            .unwrap_or_else(Task::none)];
        if !model.cells.is_empty() {
            tasks.push(iced_sugiyama::fit_to_view(iced_sugiyama::Id::new(
                CELL_GRAPH_ID,
            )));
        }
        let task = Task::batch(tasks);
        let subpanel = None;

        (
            Self {
                config,
                model,
                live,
                subpanel,
                visuals,
                appearance_loading,
                appearance_error: None,
                subpanel_notice: None,
                color_now: now,
                #[cfg(feature = "av")]
                video_overlay: load_video_overlay(),
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
                    Ok(Some(snapshot)) if snapshot.appearance.finalized => {
                        match BeamModel::from_appearance(snapshot.appearance, &snapshot.child_rays)
                        {
                            Ok(model) => {
                                let had_cells = !self.model.cells.is_empty();
                                let now = Instant::now();
                                let mut transitioned = false;
                                let node_ids = model
                                    .cells
                                    .iter()
                                    .map(|cell| cell.id)
                                    .collect::<HashSet<_>>();
                                self.visuals.retain(|node_id, _| node_ids.contains(node_id));
                                for cell in &model.cells {
                                    transitioned |=
                                        self.visuals.entry(cell.id).or_default().observe(
                                            cell.state,
                                            cell.grad_step,
                                            cell.grad_steps,
                                            cell.state_sequence,
                                            cell.frozen,
                                            now,
                                        );
                                }
                                let display_changed = model_display_changed(&self.model, &model);
                                let had_error = self.appearance_error.is_some();
                                self.model = model;
                                self.appearance_error = None;
                                let mut tasks = Vec::new();
                                if !had_cells && !self.model.cells.is_empty() {
                                    tasks.push(iced_sugiyama::fit_to_view(iced_sugiyama::Id::new(
                                        CELL_GRAPH_ID,
                                    )));
                                }
                                if display_changed || transitioned || had_error {
                                    self.color_now = now;
                                    tasks.push(iced_sugiyama::force_review(
                                        iced_sugiyama::Id::new(CELL_GRAPH_ID),
                                    ));
                                }
                                if !tasks.is_empty() {
                                    return Task::batch(tasks);
                                }
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
                let mut transitioned = false;
                for visual in self.visuals.values_mut() {
                    transitioned |= visual.advance(now);
                }
                let is_fading = self.visuals.values().any(|visual| visual.is_fading(now));

                if was_fading || transitioned || is_fading {
                    self.color_now = now;
                    return iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID));
                }
            }
            Message::NodeSelected(node_id) => {
                self.open_subpanel_for_node(node_id);
            }
            Message::CloseSubpanel => {
                self.subpanel = None;
            }
            Message::Subpanel(message) => {
                if let Some(subpanel) = self.subpanel.as_mut() {
                    return subpanel.viewer.update(message).map(Message::Subpanel);
                }
            }
            Message::SubpanelOverlayPointerEvent => {}
            #[cfg(feature = "av")]
            Message::Av(event) => {
                if let Some(video_overlay) = self.video_overlay.as_mut() {
                    video_overlay.update(event);
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
        let needs_color_frame = self
            .visuals
            .values()
            .any(|visual| visual.needs_color_frame(self.color_now));
        let needs_transition_poll = self
            .visuals
            .values()
            .any(|visual| visual.needs_transition_poll(self.color_now));
        if needs_color_frame {
            subscriptions.push(iced::time::every(COLOR_FRAME_INTERVAL).map(Message::ColorTick));
        } else if needs_transition_poll {
            subscriptions
                .push(iced::time::every(COLOR_TRANSITION_POLL_INTERVAL).map(Message::ColorTick));
        }
        if let Some(subpanel) = self.subpanel.as_ref() {
            subscriptions.push(subpanel.viewer.subscription().map(Message::Subpanel));
        }
        #[cfg(feature = "av")]
        if let Some(video_overlay) = self.video_overlay.as_ref() {
            subscriptions.push(video_overlay.subscription(map_av_message));
        }

        Subscription::batch(subscriptions)
    }

    fn view(&self) -> Element<'_, Message> {
        let main_panel: Element<'_, Message> = if self.model.cells.is_empty() {
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
        let show_subpanel_overlay = self.live.is_some()
            && !self.config.subpanel_animals.is_empty()
            && self.subpanel.is_some();
        let main_layer = container(main_panel)
            .width(Length::Fill)
            .height(Length::Fill);

        let overlay_layer: Element<'_, Message> = if show_subpanel_overlay {
            let subpanel_styles = self.cell_styles();
            let Some(subpanel) = &self.subpanel else {
                unreachable!("the subpanel overlay requires an open subpanel");
            };
            let subpanel_colors = subpanel_styles
                .get(&subpanel.node_id)
                .copied()
                .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None));
            let phase = self
                .subpanel_phase(subpanel.node_id)
                .unwrap_or_else(|| "unknown".to_string());
            let title = format!("Cell {} · {} ({phase})", subpanel.node_id, subpanel.title);
            let header = row![
                text(title)
                    .size(14)
                    .color(black_hole_text().scale_alpha(0.86))
                    .width(Length::Fill),
                button(text("X").size(13))
                    .padding([1, 6])
                    .style(subpanel_close_button_style)
                    .on_press(Message::CloseSubpanel),
            ];

            let subpanel_panel = container(
                column![
                    header,
                    container(subpanel.viewer.view().map(Message::Subpanel))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .style(subpanel_child_canvas_style),
                ]
                .spacing(8)
                .height(Length::Fill),
            )
            .padding([10, 12])
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_theme| subpanel_style(subpanel_colors));

            let subpanel_stack = mouse_area(opaque(
                container(subpanel_panel)
                    .width(Length::FillPortion(1))
                    .height(Length::Fill)
                    .padding(8)
                    .style(subpanel_overlay_style),
            ))
            .on_press(Message::SubpanelOverlayPointerEvent)
            .on_right_press(Message::SubpanelOverlayPointerEvent)
            .on_middle_press(Message::SubpanelOverlayPointerEvent)
            .on_scroll(|_| Message::SubpanelOverlayPointerEvent);

            let right_overlay = row![
                container(column![]).width(Length::FillPortion(2)),
                subpanel_stack,
            ]
            .width(Length::Fill)
            .height(Length::Fill);

            right_overlay.into()
        } else {
            // Keep a stable stack root even when there is no overlay so
            // opening/closing subpanels does not remount the graph viewport.
            container(column![])
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()
        };
        #[cfg(feature = "av")]
        let video_layer = self
            .video_overlay
            .as_ref()
            .and_then(|video| video.overlay_view(map_av_message));
        #[cfg(not(feature = "av"))]
        let video_layer: Option<Element<'_, Message>> = None;

        let mut layers = vec![main_layer.into(), overlay_layer];
        if let Some(video_layer) = video_layer {
            layers.insert(0, video_layer);
        }
        let content = stack(layers).width(Length::Fill).height(Length::Fill);
        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(app_background_style)
            .into()
    }

    fn cell_graph(&self) -> Element<'_, Message> {
        let mut layout_graph = self.model.graph.clone();
        //if matches!(self.config.layout, BeamLayout::Dot) {
        // Match the spacing used by iced-sugiyama's "moar" example.
        layout_graph.config.vertex_spacing = DOT_VERTEX_SPACING;
        //}

        let labels = self
            .model
            .cells
            .iter()
            .map(|cell| {
                let visual = self.visuals.get(&cell.id);
                let activity = visual
                    .map(|state| state.current.state)
                    .unwrap_or(cell.state);
                let grad_steps = cell.grad_steps.max(1);
                let grad_step = visual
                    .map(|state| displayed_grad_step(activity, state.current, grad_steps))
                    .unwrap_or_else(|| cell.grad_step.clamp(1, grad_steps));
                let phase_label = if matches!(activity, SunNodeState::Idle) {
                    activity.label().to_string()
                } else {
                    format!("{} · step {grad_step}/{grad_steps}", activity.label())
                };
                (cell.id, (cell.animal_name.clone(), phase_label))
            })
            .collect::<HashMap<_, _>>();
        let styles = self.cell_styles();
        let styles_for_nodes = styles.clone();
        let styles_for_edges = styles.clone();
        let styles_for_endpoints = styles;

        let mut graph = Sugiyama::<Message, Theme, iced::Renderer>::new(
            Cow::Owned(layout_graph),
            move |node_id| {
                let (animal_name, phase_label) = labels.get(&node_id).cloned().unwrap_or((
                    format!("cell {node_id}"),
                    SunNodeState::Idle.label().to_string(),
                ));
                let style = styles_for_nodes
                    .get(&node_id)
                    .copied()
                    .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None));
                button(
                    container(
                        column![
                            text(animal_name).size(16).color(style.text),
                            text(format!("cell {node_id} · {phase_label}"))
                                .size(12)
                                .color(style.text.scale_alpha(0.82)),
                        ]
                        .spacing(3),
                    )
                    .padding([10, 12])
                    .style(move |_theme| cell_node_style(style)),
                )
                .padding(0)
                .style(graph_node_button_style)
                .on_press(Message::NodeSelected(node_id))
                .into()
            },
        )
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
                let start = styles_for_edges
                    .get(&ctx.edge.0)
                    .map(|style| style.body)
                    .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
                let end = styles_for_edges
                    .get(&ctx.edge.1)
                    .map(|style| style.body)
                    .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
                (start, end)
            })
            .edge_endpoint(move |_, edge, kind, endpoint| {
                if matches!(kind, EdgeEndpointKind::Source) {
                    return None;
                }
                let node_id = edge.1;
                let color = styles_for_endpoints
                    .get(&node_id)
                    .map(|style| style.body)
                    .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
                let glyph = EdgeEndpointGlyph {
                    kind: EdgeEndpointGlyphKind::NormalArrow,
                    color,
                    angle_radians: endpoint.angle_radians(),
                };
                Some(
                    canvas::Canvas::new(glyph)
                        .width(glyph.size())
                        .height(glyph.size())
                        .into(),
                )
            })
            .stroke_width(EDGE_STROKE_WIDTH)
            .edge_corner_radius(16.0)
            .padding(24)
            .auto_fit(AutoFit::Off)
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

    fn cell_styles(&self) -> HashMap<u32, NodeStyleColors> {
        self.model
            .cells
            .iter()
            .map(|cell| {
                let style = self
                    .visuals
                    .get(&cell.id)
                    .map(|visual| visual.style(cell.grad_steps, cell.frozen, self.color_now))
                    .unwrap_or_else(|| {
                        node_style_colors(cell.state, cell.grad_step, cell.grad_steps, cell.frozen)
                    });
                (cell.id, style)
            })
            .collect()
    }

    fn open_subpanel_for_node(&mut self, node_id: u32) {
        let Some(client) = self.live.as_ref().map(|live| live.client.clone()) else {
            return;
        };

        let Some(cell) = self.model.cells.iter().find(|cell| cell.id == node_id) else {
            return;
        };
        let journey_id = cell.journey_id;
        let animal_name = cell.animal_name.clone();

        if journey_id.is_nil() {
            self.subpanel_notice = Some(format!(
                "Cell {node_id} does not have an active journey yet."
            ));
            return;
        }

        let Some(subpanel_config) = self.resolve_subpanel_config(&animal_name) else {
            self.subpanel_notice = Some(format!(
                "No registered subpanel animal for {}.",
                animal_name
            ));
            return;
        };

        self.subpanel_notice = None;
        if self
            .subpanel
            .as_ref()
            .is_some_and(|subpanel| subpanel.journey_id == journey_id)
        {
            // The requested child flow is already on display.
            return;
        }

        self.subpanel = Some(SubpanelState {
            node_id,
            title: subpanel_config.title,
            journey_id,
            viewer: (subpanel_config.build_viewer)(SharedJungleClient::new(client), journey_id),
        });
    }

    fn resolve_subpanel_config(&self, animal_label: &str) -> Option<SubpanelConfig> {
        let lookup_key = animal_label_key(animal_label);
        self.config
            .subpanel_animals
            .iter()
            .find(|config| config.animal_label == lookup_key)
            .cloned()
    }

    fn subpanel_phase(&self, node_id: u32) -> Option<String> {
        self.model
            .cells
            .iter()
            .find(|cell| cell.id == node_id)
            .map(|cell| {
                self.visuals
                    .get(&node_id)
                    .map(|visual| visual.current.state)
                    .unwrap_or(cell.state)
                    .label()
                    .to_string()
            })
    }
}

fn appearance_task(live: LiveConfig) -> Task<Message> {
    Task::perform(fetch_appearance(live), Message::AppearanceLoaded)
}

async fn fetch_appearance(live: LiveConfig) -> Result<Option<LiveAppearanceSnapshot>, String> {
    let bytes = live
        .client
        .animal_appearance(live.journey_id)
        .await
        .map_err(|error| error.to_string())?;
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let appearance = postcard::from_bytes::<SunAppearance>(&bytes)
        .map_err(|error| format!("could not decode Sun appearance: {error}"))?;
    let child_rays = fetch_child_rays(&live, &appearance).await;
    Ok(Some(LiveAppearanceSnapshot {
        appearance,
        child_rays,
    }))
}

async fn fetch_child_rays(live: &LiveConfig, appearance: &SunAppearance) -> HashMap<Uuid, Ray> {
    let mut rays = HashMap::new();
    for node in &appearance.nodes {
        let maybe_bytes = live
            .client
            .animal_appearance(node.journey_id)
            .await
            .ok()
            .flatten();
        let Some(bytes) = maybe_bytes else {
            continue;
        };
        let Ok(ray) = postcard::from_bytes::<Ray>(&bytes) else {
            continue;
        };
        rays.insert(node.journey_id, ray);
    }
    rays
}

#[cfg(feature = "av")]
fn load_video_overlay() -> Option<iced_av1::widget::State> {
    let path = video_overlay_path()?;
    let opacity = iced_av1::OpacityOptions {
        opacity: AV_OVERLAY_OPACITY,
        ..Default::default()
    };
    match iced_av1::widget::State::new_with_media_source_and_opacity_options(
        iced_av1::MediaSource::File(path.clone()),
        iced_av1::PlaybackOptions::default(),
        opacity,
    ) {
        Ok(video) => {
            info!(path = %path.display(), opacity = AV_OVERLAY_OPACITY, "loaded black hole sun video overlay");
            Some(video)
        }
        Err(error) => {
            warn!(path = %path.display(), %error, "failed to load black hole sun video overlay");
            None
        }
    }
}

#[cfg(feature = "av")]
fn video_overlay_path() -> Option<PathBuf> {
    let path = BaseDirs::new()?
        .home_dir()
        .join(".black-hole-sun")
        .join("black-hole-sun.mkv");
    path.is_file().then_some(path)
}

#[cfg(feature = "av")]
fn map_av_message(message: iced_av1::widget::Message) -> Message {
    Message::Av(message)
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

fn subpanel_style(colors: NodeStyleColors) -> iced::widget::container::Style {
    iced::widget::container::Style {
        // Mirror each node's phase tint while keeping subpanel content readable.
        background: Some(Background::Color(Color::from_rgba(
            colors.body.r,
            colors.body.g,
            colors.body.b,
            0.2,
        ))),
        border: iced::Border {
            color: Color::from_rgba(colors.border.r, colors.border.g, colors.border.b, 0.58),
            width: 1.0,
            ..iced::border::rounded(8)
        },
        text_color: Some(colors.text),
        ..Default::default()
    }
}

fn subpanel_overlay_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(3, 3, 3, 0.7))),
        border: iced::Border {
            color: Color::from_rgba8(120, 120, 120, 0.25),
            width: 1.0,
            ..iced::border::rounded(10)
        },
        ..Default::default()
    }
}

fn subpanel_child_canvas_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        // Child graph canvases should feel translucent without reducing text legibility.
        background: Some(Background::Color(Color::from_rgba8(7, 17, 11, 0.62))),
        border: iced::Border {
            color: Color::from_rgba8(104, 140, 121, 0.24),
            width: 1.0,
            ..iced::border::rounded(6)
        },
        ..Default::default()
    }
}

fn subpanel_close_button_style(
    _theme: &Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let text_color = match status {
        iced::widget::button::Status::Hovered => Color::from_rgb8(255, 205, 156),
        _ => black_hole_text().scale_alpha(0.88),
    };
    iced::widget::button::Style {
        background: None,
        text_color,
        border: iced::Border {
            color: Color::from_rgba8(176, 176, 176, 0.35),
            width: 1.0,
            ..iced::border::rounded(6)
        },
        shadow: Shadow::default(),
        snap: false,
    }
}

fn graph_node_button_style(
    _theme: &Theme,
    _status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    iced::widget::button::Style {
        background: None,
        text_color: black_hole_text(),
        border: iced::Border::default(),
        shadow: Shadow::default(),
        snap: false,
    }
}

fn cell_node_style(colors: NodeStyleColors) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(colors.body)),
        text_color: Some(colors.text),
        border: iced::Border {
            color: colors.border,
            width: 2.2,
            ..iced::border::rounded(9)
        },
        shadow: Shadow {
            color: Color::from_rgba(colors.body.r, colors.body.g, colors.body.b, 0.32),
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

fn lighten(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::WHITE, amount)
}

fn short_type_name<T: ?Sized>() -> String {
    animal_label_key(core::any::type_name::<T>())
}

fn animal_label_key(label: &str) -> String {
    short_type_label(label)
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
        type Seed = black_hole_flux::CellInit;
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

    struct GenericCell<T>(std::marker::PhantomData<T>);

    impl<T> Animal for GenericCell<T> {
        type Id = Id<U1>;
        type Generation = U0;
        type State = CellState;
        type Seed = black_hole_flux::CellInit;
        type Flow = Primordium;
    }

    type PortOne = List<(U1, Empty)>;
    type Tail = List<(Unary<U1, TestCell, Empty>, Empty)>;
    type TestGraph = List<(Unary<U0, TestCell, PortOne>, Tail)>;
    type TestSun = <TestGraph as BlackHole>::Sun<Primordium, Primordium, (), 1>;
    type TestSunWithCustomState =
        <TestGraph as BlackHole>::Sun<Primordium, Primordium, (String, String), 1>;
    type TestBinaryGraph = List<(Binary<U0, U1, TestFusion, Empty>, Empty)>;
    type TestBinarySun = <TestBinaryGraph as BlackHole>::Sun<Primordium, Primordium, (), 1>;

    #[test]
    fn uses_black_hole_sun_title_and_grad_step_palette() {
        assert_eq!(BeamBuilder::default().title, "Black Hole Sun");
        assert!(matches!(BeamBuilder::default().layout, BeamLayout::Circo));
        assert!(matches!(
            BeamBuilder::new().dot_layout().layout,
            BeamLayout::Dot
        ));
        let p1_step1 = node_style_colors(SunNodeState::Propagation1, 1, 4, None).body;
        let p1_step4 = node_style_colors(SunNodeState::Propagation1, 4, 4, None).body;
        assert!(p1_step4.g > p1_step1.g);

        let p2_step1 = node_style_colors(SunNodeState::Propagation2, 1, 4, None).body;
        let p2_step4 = node_style_colors(SunNodeState::Propagation2, 4, 4, None).body;
        assert!(p2_step4.g < p2_step1.g);
        assert!(p2_step4.r - p2_step4.g > p2_step1.r - p2_step1.g);

        let optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(false));
        assert_eq!(optimize.body, Color::from_rgb8(255, 255, 255));
        assert_eq!(optimize.text, Color::from_rgb8(18, 12, 8));
        assert_eq!(optimize.border, Color::from_rgb8(255, 120, 18));

        let frozen_optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        assert_eq!(frozen_optimize.body, Color::from_rgb8(94, 122, 214));
        assert_eq!(frozen_optimize.border, Color::from_rgb8(154, 92, 232));
    }

    #[test]
    fn registers_unique_subpanel_animals() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .register_subpanel_animal::<GenericCell<String>>()
            .register_subpanel_animal::<GenericCell<u8>>()
            .register_subpanel_animal::<TestCell>()
            .into_config();

        assert_eq!(config.subpanel_animals.len(), 3);
        assert_eq!(config.subpanel_animals[0].animal_label, "TestCell");
        assert_eq!(config.subpanel_animals[0].title, "TestCell");
        assert_eq!(
            config.subpanel_animals[1].animal_label,
            "GenericCell<String>"
        );
        assert_eq!(config.subpanel_animals[1].title, "GenericCell<String>");
        assert_eq!(config.subpanel_animals[2].animal_label, "GenericCell<u8>");
        assert_eq!(config.subpanel_animals[2].title, "GenericCell<u8>");
    }

    #[test]
    fn fades_and_holds_ui_activity_colors() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert_eq!(APPEARANCE_INTERVAL, Duration::from_millis(200));
        assert_eq!(COLOR_FADE_DURATION, Duration::from_millis(400));
        assert_eq!(MIN_COLOR_STATE_DURATION, Duration::from_secs(1));
        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, None, start));
        let idle = node_style_colors(SunNodeState::Idle, 1, 4, None).body;
        let p1_step1 = node_style_colors(SunNodeState::Propagation1, 1, 4, None).body;
        assert_eq!(
            visual.style(4, None, start).body,
            idle,
            "the fade starts from the previous activity color"
        );
        assert_eq!(
            visual.style(4, None, start + COLOR_FADE_DURATION / 2).body,
            lerp_color(idle, p1_step1, 0.5),
            "the color is blended halfway through the fade"
        );
        assert_eq!(
            visual.style(4, None, start + COLOR_FADE_DURATION).body,
            p1_step1,
            "the fade reaches the new color after 400ms"
        );

        assert!(!visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            None,
            start + COLOR_FADE_DURATION
        ));
        assert!(!visual.advance(start + MIN_COLOR_STATE_DURATION - Duration::from_millis(1)));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
    }

    #[test]
    fn optimization_color_uses_frozen_state_captured_at_propagation_transition() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, Some(false), start));
        assert!(visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            Some(true),
            start + MIN_COLOR_STATE_DURATION
        ));
        assert_eq!(visual.optimization_frozen, Some(true));

        assert!(visual.observe(
            SunNodeState::Optimization,
            4,
            4,
            3,
            Some(false),
            start + MIN_COLOR_STATE_DURATION * 2
        ));

        let style = visual.style(
            4,
            Some(false),
            start + MIN_COLOR_STATE_DURATION * 2 + COLOR_FADE_DURATION,
        );
        assert_eq!(
            style.body,
            Color::from_rgb8(94, 122, 214),
            "optimization style keeps the frozen color captured at propagation1 -> propagation2"
        );
    }

    #[test]
    fn queues_intermediate_phases_when_an_appearance_jumps() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert_eq!(visual.current.state, SunNodeState::Propagation1);
        assert_eq!(
            visual.pending,
            VecDeque::from([NodeProgress {
                state: SunNodeState::Propagation2,
                grad_step: 1,
            }])
        );
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
    }

    #[test]
    fn replays_an_epoch_when_snapshots_repeat_the_same_phase() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
        assert!(visual.pending.is_empty());

        assert!(visual.observe(
            SunNodeState::Propagation2,
            3,
            4,
            5,
            None,
            start + MIN_COLOR_STATE_DURATION * 2
        ));
        assert_eq!(visual.current.state, SunNodeState::Optimization);
        assert_eq!(
            visual.pending,
            VecDeque::from([
                NodeProgress {
                    state: SunNodeState::Propagation1,
                    grad_step: 1,
                },
                NodeProgress {
                    state: SunNodeState::Propagation2,
                    grad_step: 3,
                },
            ])
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
            visual.observe(phase, 1, 4, sequence, None, start);
        }

        assert!(visual.pending.len() <= MAX_PENDING_PHASES);
    }

    #[test]
    fn throttles_color_ticks_while_waiting_for_next_transition() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();
        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert_eq!(visual.current.state, SunNodeState::Propagation1);

        let fade_end = start + COLOR_FADE_DURATION;
        assert!(!visual.needs_color_frame(fade_end));
        assert!(visual.needs_transition_poll(fade_end));
    }

    #[test]
    fn extracts_static_cells_and_edges_from_black_hole_sun() {
        let model = BeamModel::build::<TestSun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0, 1]);
        assert_eq!(model.graph.edges, vec![(0, 1)]);
    }

    #[test]
    fn extracts_static_cells_and_edges_from_stateful_black_hole_sun() {
        let model = BeamModel::build::<TestSunWithCustomState>();

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
            grad_steps: 4,
            nodes: vec![
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    label: "Fusion".to_string(),
                    input_ports: vec![2, 3],
                    state: SunNodeState::Optimization,
                    state_sequence: 3,
                    grad_step: 4,
                },
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Propagation1,
                    state_sequence: 1,
                    grad_step: 1,
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
        let model = BeamModel::from_appearance(decoded, &HashMap::new()).unwrap();

        assert_eq!(model.grad_steps, 4);
        assert_eq!(model.graph.nodes, vec![0, 2]);
        assert_eq!(
            model.graph.edges,
            vec![(0, 2)],
            "parallel destination-port edges collapse only for rendering"
        );
        assert_eq!(model.cells[0].animal_name, "Root");
        assert_eq!(model.cells[0].state, SunNodeState::Propagation1);
        assert_eq!(model.cells[0].state_sequence, 1);
        assert_eq!(model.cells[0].grad_step, 1);
        assert_eq!(model.cells[0].grad_steps, 4);
        assert_eq!(model.cells[1].state, SunNodeState::Optimization);
        assert_eq!(model.cells[1].state_sequence, 3);
        assert_eq!(model.cells[1].grad_step, 4);
    }

    #[test]
    fn maps_child_ray_frozen_state_into_live_cells() {
        let frozen_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: frozen_journey,
                label: "Root".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Optimization,
                state_sequence: 4,
                grad_step: 1,
            }],
            edges: vec![],
        };
        let rays = HashMap::from([(frozen_journey, Ray { frozen: true })]);
        let model = BeamModel::from_appearance(appearance, &rays).unwrap();
        assert_eq!(model.cells[0].frozen, Some(true));
    }

    #[test]
    fn rejects_appearance_edges_with_the_wrong_destination_port() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 1,
                    journey_id: Uuid::new_v4(),
                    label: "Sink".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 1,
                target_port: 9,
            }],
        };

        let error = BeamModel::from_appearance(appearance, &HashMap::new())
            .err()
            .expect("appearance should be rejected");
        assert!(error.contains("unowned input port 9"));
    }

    #[test]
    fn keeps_generics_in_type_labels() {
        assert_eq!(
            short_type_name::<GenericCell<String>>(),
            "GenericCell<String>"
        );
        assert_eq!(
            animal_label_key("my::module::RootAnimal<crate::leaf::Type, alloc::vec::Vec<u8>>"),
            "RootAnimal<Type, Vec<u8>>"
        );
    }

    #[test]
    fn keeps_generics_in_live_appearance_labels() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: Uuid::new_v4(),
                label: "RootAnimal<Result<String, Vec<u8>>>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };

        let model = BeamModel::from_appearance(appearance, &HashMap::new()).unwrap();
        assert_eq!(
            model.cells[0].animal_name,
            "RootAnimal<Result<String, Vec<u8>>>"
        );
    }

    #[test]
    fn labels_optimization_phase_as_potentiation() {
        assert_eq!(SunNodeState::Optimization.label(), "potentiation");
    }
}
