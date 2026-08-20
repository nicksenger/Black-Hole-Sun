//! Visualize Black Hole Sun cell graphs.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole), using the circular `circo`
//! layout by default. Live views use the parent Sun animal's Jungle
//! [`Observe`](jungle_sdk::Observe) appearance as the source of graph topology
//! and node phase.
//!
//! The `piano` feature adds a NanoMoog-styled 88-key piano to the bottom of
//! the viewer. Use `BeamBuilder::on_piano_event` to capture its expressive,
//! performance-timed attack and release events.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "piano")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use black_hole_flux::sun::{
    BinarySunStep, NodeIdsFromList, Sun, SunAppearance, SunNode, SunNodeState, SunState,
    UnarySunStep,
};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState, Ray};
#[cfg(feature = "piano")]
use iced::keyboard;
use iced::mouse;
use iced::time::Instant;
use iced::widget::canvas::{self, Path};
use iced::widget::{button, column, container, mouse_area, opaque, row, rule, space, stack, text};
use iced::{
    Background, Color, Element, Font, Length, Point, Rectangle, Shadow, Subscription, Task, Theme,
    Vector,
};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{
    circo_layout, microdot_layout, AutoFit, Cluster, EdgeEndpointKind, Graph, LayoutInput, Sugiyama,
};
use jungle_sdk::{Animal, AnimalIdValue, JourneyAstSource, JungleClient, Observe};
use jungle_vision::{
    AnyAnimal, ClusterExpansionConfig, ClusterExpansionMode, DefaultTheme, EjectedViewer,
    EjectedViewerMessage, JungleViewerBuilder,
};
use typenum::Unsigned;
use uuid::Uuid;

#[cfg(feature = "piano")]
mod piano;
#[cfg(feature = "piano")]
mod piano_audio;
#[cfg(feature = "piano")]
mod piano_score;
#[cfg(feature = "piano")]
pub mod score_text;

#[cfg(feature = "piano")]
pub use piano::{PianoAction, PianoEvent, PianoInputSource, PianoNote};
#[cfg(feature = "piano")]
use piano::{PianoKeyAppearance, PianoKeyboard, PianoMessage, PianoPointerSource, PIANO_HEIGHT};
#[cfg(feature = "piano")]
use piano_audio::PianoAudioEngine;
#[cfg(feature = "piano")]
pub use piano_audio::{render_piano_score_to_wav, PianoRenderReport};
#[cfg(feature = "piano")]
use piano_score::{PianoScorePlayback, SCORE_TICK_INTERVAL};
#[cfg(feature = "piano")]
use score_text::BhsScore;

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
    let theme = DefaultTheme::default().with_cluster_expansion_config(ClusterExpansionConfig {
        while_clusters: ClusterExpansionMode::AlwaysExpanded,
        transparent_clusters: ClusterExpansionMode::AlwaysExpanded,
    });
    JungleViewerBuilder::new().eject_live_animal_with_theme::<A, SharedJungleClient, _, AnyAnimal>(
        client, journey_id, theme,
    )
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
    #[cfg(feature = "piano")]
    piano_event_handler: Option<Arc<dyn Fn(PianoEvent) + Send + Sync>>,
    #[cfg(feature = "piano")]
    piano_score_path: Option<PathBuf>,
    #[cfg(feature = "piano")]
    piano_score_data: Option<Vec<u8>>,
    #[cfg(feature = "piano")]
    piano_score: Option<BhsScore>,
}

#[derive(Clone, Copy)]
enum BeamLayout {
    Circo,
    Microdot,
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
            #[cfg(feature = "piano")]
            piano_event_handler: None,
            #[cfg(feature = "piano")]
            piano_score_path: None,
            #[cfg(feature = "piano")]
            piano_score_data: None,
            #[cfg(feature = "piano")]
            piano_score: None,
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

    /// Use iced-sugiyama's microdot layout for node placement.
    pub fn microdot_layout(mut self) -> Self {
        self.layout = BeamLayout::Microdot;
        self
    }

    /// Register an animal type that can be shown in a node-click subpanel.
    ///
    /// In live mode, clicking a node whose animal matches one of these
    /// registrations opens that node's journey in the subpanel overlay,
    /// replacing any child flow that is currently open. Only one child
    /// flow is shown at a time.
    ///
    /// Warp nodes are labeled `Warp<WarpAnimal, BoundaryAnimal>` and run
    /// the boundary animal's journey, so registering a warp node's boundary
    /// animal opens its subpanel on that boundary's live journey.
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

    /// Receive expressive attack and release events from the 88-key piano.
    ///
    /// Every event contains performance-relative timing, stable ordering, a
    /// voice ID, note/frequency, velocity, and input source. Computer-keyboard
    /// input uses [`PianoEvent::BINARY_VELOCITY`]; the event schema retains
    /// continuous values for velocity-sensitive inputs.
    #[cfg(feature = "piano")]
    pub fn on_piano_event(mut self, handler: impl Fn(PianoEvent) + Send + Sync + 'static) -> Self {
        self.piano_event_handler = Some(Arc::new(handler));
        self
    }

    /// Continuously loop a recorded piano performance from a file.
    ///
    /// The file is a `bhs-score-v1` text score (conventionally with a `.bhs`
    /// extension) described in [`score_text`]. Events are ordered by
    /// timestamp and sequence and are routed through the same audio,
    /// visualization, and callback paths as live key presses. If both a path
    /// and data are set via [`Self::score_data`], the path wins.
    #[cfg(feature = "piano")]
    pub fn score_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.piano_score_path = Some(path.into());
        self
    }

    /// Continuously loop a recorded piano performance from in-memory bytes.
    ///
    /// `data` is a `bhs-score-v1` text score described in [`score_text`],
    /// encoded as UTF-8. Events are routed through the same audio,
    /// visualization, and callback paths as live key presses.
    #[cfg(feature = "piano")]
    pub fn score_data(mut self, data: &[u8]) -> Self {
        self.piano_score_data = Some(data.to_vec());
        self
    }

    /// Continuously loop a recorded piano performance from an owned score.
    ///
    /// `score` is a parsed [`BhsScore`] (see [`score_text`]), ready to play
    /// as-is — for example one loaded and mutated in memory. Events are
    /// routed through the same audio, visualization, and callback paths as
    /// live key presses. This source is used only when neither a path nor
    /// data is set via [`Self::score_data`].
    #[cfg(feature = "piano")]
    pub fn score(mut self, score: BhsScore) -> Self {
        self.piano_score = Some(score);
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
            #[cfg(feature = "piano")]
            piano_event_handler: self.piano_event_handler,
            #[cfg(feature = "piano")]
            piano_score_path: self.piano_score_path,
            #[cfg(feature = "piano")]
            piano_score_data: self.piano_score_data,
            #[cfg(feature = "piano")]
            piano_score: self.piano_score,
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
/// `<Graph as BlackHole>::Sun<M, N>`, where `M` is a
/// [`Manifest`](black_hole_flux::sun::Manifest) bundling generator, policy,
/// and state.
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
    #[cfg(feature = "piano")]
    piano_event_handler: Option<Arc<dyn Fn(PianoEvent) + Send + Sync>>,
    #[cfg(feature = "piano")]
    piano_score_path: Option<PathBuf>,
    #[cfg(feature = "piano")]
    piano_score_data: Option<Vec<u8>>,
    #[cfg(feature = "piano")]
    piano_score: Option<BhsScore>,
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
    /// Journey of the nested warp animal for warp cells; nil otherwise.
    warp_journey_id: Uuid,
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
            warp_journey_id: Uuid::nil(),
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
    /// Main-graph id -> path of local cell ids (top level first) for every
    /// warp cell, e.g. `[7]` for a top-level warp and `[7, 3]` for warp cell
    /// 3 inside cell 7's nested sun. Empty for statically built models.
    warp_paths: HashMap<u32, Vec<u32>>,
}

impl BeamModel {
    fn empty() -> Self {
        Self {
            cells: Vec::new(),
            graph: Graph::new(Vec::new(), Vec::new()),
            grad_steps: 1,
            errors: Vec::new(),
            warp_paths: HashMap::new(),
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
    fn from_appearance(
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
    ) -> (Vec<CellDefinition>, Vec<(u32, u32, u32)>, HashMap<u32, Vec<u32>>) {
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
                .filter(|(path, _)| {
                    path.len() > 1 && path.first() == Some(&parent_id)
                })
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
                errors.push(format!("cell {source} has a self edge on port {target_port}"));
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

#[cfg(feature = "piano")]
fn piano_computer_key(key: &keyboard::Key, physical_key: keyboard::key::Physical) -> Option<char> {
    let key = key.to_latin(physical_key)?.to_ascii_lowercase();
    Some(match key {
        '<' => ',',
        '>' => '.',
        '?' => '/',
        '{' => '[',
        '}' => ']',
        key => key,
    })
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
    #[cfg(feature = "piano")]
    Piano(PianoMessage),
    #[cfg(feature = "piano")]
    PianoKeyboard(keyboard::Event),
    #[cfg(feature = "piano")]
    PianoScoreTick(Instant),
    #[cfg(feature = "piano")]
    PianoVisualTick(Instant),
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PianoInputId {
    ComputerKeyboard(char),
    Pointer(PianoPointerSource),
    Score { cycle: u64, voice_id: u64 },
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy)]
struct ActivePianoNote {
    note: PianoNote,
    voice_id: u64,
    started_at: Instant,
    source: PianoInputSource,
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy)]
struct PianoStrikeVisual {
    midi_note: u8,
    velocity: f32,
    pressure: Option<f32>,
    attacked_at: Instant,
    released: Option<(Instant, f32)>,
}

#[cfg(feature = "piano")]
impl PianoStrikeVisual {
    fn appearance(self, now: Instant) -> PianoKeyAppearance {
        let velocity = self.velocity.clamp(0.0, 1.0);
        let pressure = self.pressure.unwrap_or(velocity * 0.72).clamp(0.0, 1.0);
        let attack_duration = Duration::from_secs_f32(0.035 - velocity * 0.030);
        let attack_progress = now
            .saturating_duration_since(self.attacked_at)
            .as_secs_f32()
            / attack_duration.as_secs_f32();
        let attack_progress = attack_progress.clamp(0.0, 1.0);
        let attack_curve = attack_progress * attack_progress * (3.0 - 2.0 * attack_progress);
        let strike = 0.22 + velocity * 0.66 + pressure * 0.12;
        let sustain = 0.18 + velocity * 0.34 + pressure * 0.28;
        let settle_progress = (now
            .saturating_duration_since(self.attacked_at)
            .as_secs_f32()
            / 0.16)
            .clamp(0.0, 1.0);
        let held_intensity = (strike + (sustain - strike) * settle_progress) * attack_curve;

        let intensity = if let Some((released_at, release_velocity)) = self.released {
            let release_duration = 0.34 - release_velocity.clamp(0.0, 1.0) * 0.23;
            let release_progress = (now.saturating_duration_since(released_at).as_secs_f32()
                / release_duration)
                .clamp(0.0, 1.0);
            held_intensity * (1.0 - release_progress).powi(2)
        } else {
            held_intensity
        };
        PianoKeyAppearance {
            intensity: intensity.clamp(0.0, 1.0),
        }
    }

    fn needs_frame(self, now: Instant) -> bool {
        self.released.is_some()
            || now.saturating_duration_since(self.attacked_at) < Duration::from_millis(160)
    }

    fn finished(self, now: Instant) -> bool {
        self.released.is_some() && self.appearance(now).intensity <= 0.001
    }
}

#[derive(Debug, Clone)]
struct LiveAppearanceSnapshot {
    appearance: SunAppearance,
    child_rays: HashMap<Uuid, Ray>,
    /// Nested Sun appearances for warp cells, keyed by the path of cell ids
    /// that locates the warp cell: `[7]` is a top-level warp, `[7, 3]` is
    /// warp cell 3 inside cell 7's nested sun.
    warp_appearances: HashMap<Vec<u32>, SunAppearance>,
    /// Why a warp cell's nested appearance could not be used yet, keyed by
    /// the same paths. Present whenever no usable model was produced.
    warp_diagnostics: HashMap<Vec<u32>, String>,
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
    let fire_red = Color::from_rgb8(240, 96, 24);
    let potentiation_blue = Color::from_rgb8(65, 105, 225);
    let frozen_potentiation_violet = Color::from_rgb8(124, 77, 255);

    if state == SunNodeState::Optimization {
        if frozen == Some(true) {
            // Black body so the frozen node recedes, but a violet outline keeps
            // it identifiable during potentiation.
            return NodeStyleColors {
                body: Color::BLACK,
                border: frozen_potentiation_violet,
                text: Color::BLACK,
            };
        }
        return NodeStyleColors {
            body: Color::from_rgb8(255, 255, 255),
            border: potentiation_blue,
            text: Color::from_rgb8(18, 12, 8),
        };
    }

    let (body, border) = match state {
        SunNodeState::Idle => (idle_orange, lighten(idle_orange, 0.18)),
        SunNodeState::Propagation1 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            let body = lerp_color(idle_orange, deep_crimson, progress);
            (body, lighten(body, 0.18))
        }
        SunNodeState::Propagation2 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            let body = lerp_color(idle_orange, bright_yellow, progress);
            // Keep a fixed fire-red outline so the yellow body stays distinct.
            (body, fire_red)
        }
        SunNodeState::Optimization => unreachable!("optimization is handled above"),
    };
    NodeStyleColors {
        body,
        border,
        text: contrasting_text(body),
    }
}

/// Unexpanded warp nodes render with a black body and white text while
/// keeping the phase-driven border color. Once their subgraph is expanded
/// into the main graph they are colored like regular nodes instead.
fn warp_node_style_colors(
    state: SunNodeState,
    grad_step: usize,
    grad_steps: usize,
    frozen: Option<bool>,
) -> NodeStyleColors {
    let mut colors = node_style_colors(state, grad_step, grad_steps, frozen);
    colors.body = Color::BLACK;
    colors.text = Color::WHITE;
    colors
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

    fn style(
        &self,
        grad_steps: usize,
        frozen: Option<bool>,
        warp: bool,
        now: Instant,
    ) -> NodeStyleColors {
        let progress = self
            .transition_started_at
            .map(|started_at| {
                now.saturating_duration_since(started_at).as_secs_f32()
                    / COLOR_FADE_DURATION.as_secs_f32()
            })
            .unwrap_or(1.0);
        let previous_frozen = self.frozen_for_state(self.previous.state, frozen);
        let current_frozen = self.frozen_for_state(self.current.state, frozen);
        let style_fn = if warp {
            warp_node_style_colors
        } else {
            node_style_colors
        };
        let previous = style_fn(
            self.previous.state,
            self.previous.grad_step,
            grad_steps,
            previous_frozen,
        );
        let current = style_fn(
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
    /// Warp cells whose nested sun is merged into the main graph, as paths of
    /// local cell ids from the top level (e.g. `[7]` or `[7, 3]`). Toggled by
    /// clicking the boundary cell; collapsing a path also collapses every
    /// expanded sub-path beneath it.
    expanded_warp_cells: HashSet<Vec<u32>>,
    /// Latest polled snapshot; the source for rebuilding the main graph when
    /// warp subgraphs expand or collapse.
    last_snapshot: Option<LiveAppearanceSnapshot>,
    visuals: HashMap<u32, CellVisualState>,
    appearance_loading: bool,
    appearance_error: Option<String>,
    subpanel_notice: Option<String>,
    color_now: Instant,
    #[cfg(feature = "piano")]
    piano_started_at: Instant,
    #[cfg(feature = "piano")]
    piano_event_sequence: u64,
    #[cfg(feature = "piano")]
    piano_voice_sequence: u64,
    #[cfg(feature = "piano")]
    active_piano_notes: HashMap<PianoInputId, ActivePianoNote>,
    #[cfg(feature = "piano")]
    piano_strike_visuals: HashMap<u64, PianoStrikeVisual>,
    #[cfg(feature = "piano")]
    piano_visual_now: Instant,
    #[cfg(feature = "piano")]
    piano_audio: Option<PianoAudioEngine>,
    #[cfg(feature = "piano")]
    piano_audio_error: Option<String>,
    #[cfg(feature = "piano")]
    piano_score: Option<PianoScorePlayback>,
    #[cfg(feature = "piano")]
    piano_score_error: Option<String>,
    #[cfg(feature = "piano")]
    piano_score_cycle: u64,
}

impl BeamApp {
    fn new(
        #[allow(unused_mut)] mut config: BeamConfig,
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
        #[cfg(feature = "piano")]
        let (piano_audio, piano_audio_error) = if cfg!(test) {
            (None, None)
        } else {
            match PianoAudioEngine::new() {
                Ok(audio) => (Some(audio), None),
                Err(error) => (None, Some(error)),
            }
        };
        #[cfg(feature = "piano")]
        let (piano_score, piano_score_error) =
            if let Some(path) = config.piano_score_path.as_deref() {
                match PianoScorePlayback::load(path, Instant::now()) {
                    Ok(score) => (Some(score), None),
                    Err(error) => (None, Some(error)),
                }
            } else if let Some(data) = config.piano_score_data.as_deref() {
                match PianoScorePlayback::from_bytes(data, Instant::now()) {
                    Ok(score) => (Some(score), None),
                    Err(error) => (None, Some(error)),
                }
            } else if let Some(score) = config.piano_score.take() {
                match PianoScorePlayback::from_score(score, Instant::now()) {
                    Ok(score) => (Some(score), None),
                    Err(error) => (None, Some(error)),
                }
            } else {
                (None, None)
            };

        (
            Self {
                config,
                model,
                live,
                subpanel,
                expanded_warp_cells: HashSet::new(),
                last_snapshot: None,
                visuals,
                appearance_loading,
                appearance_error: None,
                subpanel_notice: None,
                color_now: now,
                #[cfg(feature = "piano")]
                piano_started_at: now,
                #[cfg(feature = "piano")]
                piano_event_sequence: 0,
                #[cfg(feature = "piano")]
                piano_voice_sequence: 0,
                #[cfg(feature = "piano")]
                active_piano_notes: HashMap::new(),
                #[cfg(feature = "piano")]
                piano_strike_visuals: HashMap::new(),
                #[cfg(feature = "piano")]
                piano_visual_now: now,
                #[cfg(feature = "piano")]
                piano_audio,
                #[cfg(feature = "piano")]
                piano_audio_error,
                #[cfg(feature = "piano")]
                piano_score,
                #[cfg(feature = "piano")]
                piano_score_error,
                #[cfg(feature = "piano")]
                piano_score_cycle: 0,
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
                        // Only warp subgraphs whose full path is expanded
                        // join the main graph.
                        let warp_appearances = self.expanded_warp_appearances(&snapshot);
                        match BeamModel::from_appearance(
                            snapshot.appearance.clone(),
                            &snapshot.child_rays,
                            &warp_appearances,
                        ) {
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
                                self.last_snapshot = Some(snapshot);
                                self.model = model;
                                // Keep the notice current while a subpanel
                                // stays open on an expanded warp cell whose
                                // subgraph is not in the main graph yet.
                                if let Some(subpanel) = self.subpanel.as_ref() {
                                    let is_warp = self.model.cells.iter().any(|cell| {
                                        cell.id == subpanel.node_id
                                            && !cell.warp_journey_id.is_nil()
                                    });
                                    if is_warp {
                                        let merged = self.is_warp_merged(subpanel.node_id)
                                            && self.warp_appearance_available(subpanel.node_id);
                                        self.subpanel_notice = if merged {
                                            None
                                        } else {
                                            Some(match self.warp_diagnostic(subpanel.node_id) {
                                                Some(diagnostic) => format!(
                                                    "Cell {}: {diagnostic}",
                                                    subpanel.node_id
                                                ),
                                                None => format!(
                                                    "Cell {}'s warp journey has not exposed its Black Hole Sun appearance yet; its subgraph will join the main graph once it does.",
                                                    subpanel.node_id
                                                ),
                                            })
                                        };
                                    }
                                }
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
            Message::NodeSelected(node_id) => return self.open_subpanel_for_node(node_id),
            Message::CloseSubpanel => {
                // Closing the subpanel also collapses any warp subgraph it
                // expanded.
                let closing_warp_node = self
                    .subpanel
                    .as_ref()
                    .and_then(|subpanel| {
                        self.model
                            .cells
                            .iter()
                            .find(|cell| cell.id == subpanel.node_id)
                    })
                    .is_some_and(|cell| !cell.warp_journey_id.is_nil());
                if closing_warp_node {
                    let node_id = self.subpanel.as_ref().unwrap().node_id;
                    let path = self.model.warp_paths.get(&node_id).cloned();
                    if let Some(path) = path {
                        self.collapse_warp_path(&path);
                    }
                }
                self.subpanel = None;
                if closing_warp_node {
                    return self.rebuild_model();
                }
            }
            Message::Subpanel(message) => {
                if let Some(subpanel) = self.subpanel.as_mut() {
                    return subpanel.viewer.update(message).map(Message::Subpanel);
                }
            }
            Message::SubpanelOverlayPointerEvent => {}
            #[cfg(feature = "piano")]
            Message::Piano(message) => self.update_piano(message),
            #[cfg(feature = "piano")]
            Message::PianoKeyboard(event) => self.update_piano_keyboard(event),
            #[cfg(feature = "piano")]
            Message::PianoScoreTick(now) => self.update_piano_score(now),
            #[cfg(feature = "piano")]
            Message::PianoVisualTick(now) => self.update_piano_visuals(now),
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
        #[cfg(feature = "piano")]
        subscriptions.push(keyboard::listen().map(Message::PianoKeyboard));
        #[cfg(feature = "piano")]
        if self.piano_score.is_some() {
            subscriptions.push(iced::time::every(SCORE_TICK_INTERVAL).map(Message::PianoScoreTick));
        } else if self
            .piano_strike_visuals
            .values()
            .any(|visual| visual.needs_frame(self.piano_visual_now))
        {
            subscriptions
                .push(iced::time::every(COLOR_FRAME_INTERVAL).map(Message::PianoVisualTick));
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
        let main_layer: Element<'_, Message> = if let Some(notice) = &self.subpanel_notice {
            column![
                container(text(notice).size(13))
                    .width(Length::Fill)
                    .padding(iced::Padding {
                        top: 8.0,
                        left: 12.0,
                        right: 12.0,
                        bottom: 0.0
                    })
                    .style(move |theme| subpanel_notice_style(theme)),
                container(main_panel)
                    .width(Length::Fill)
                    .height(Length::Fill),
            ]
            .spacing(0)
            .into()
        } else {
            container(main_panel)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        };

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
            let header_space = || space::Space::new().width(Length::Fixed(8.0));
            let header = row![
                header_space(),
                text(title)
                    .size(14)
                    .color(black_hole_text().scale_alpha(0.86))
                    .width(Length::Fill),
                button(text("X").size(13))
                    .padding([1, 6])
                    .style(subpanel_close_button_style)
                    .on_press(Message::CloseSubpanel),
                header_space(),
            ];

            // Iced borders are uniform on all sides, so the panel's left edge
            // is drawn as a vertical rule.
            let subpanel_panel = container(
                row![
                    rule::vertical(1)
                        .style(move |_theme| subpanel_left_edge_style(subpanel_colors)),
                    container(
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
                    // Keep top padding for the header gap but let the child
                    // canvas sit flush with the panel's right and bottom edges.
                    .padding(iced::Padding {
                        top: 10.0,
                        right: 0.0,
                        bottom: 0.0,
                        left: 0.0,
                    })
                    .width(Length::Fill)
                    .height(Length::Fill),
                ]
                .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(move |_theme| subpanel_style(subpanel_colors));

            let subpanel_stack = mouse_area(opaque(
                container(subpanel_panel)
                    .width(Length::FillPortion(1))
                    .height(Length::Fill)
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
        let content = stack([main_layer.into(), overlay_layer])
            .width(Length::Fill)
            .height(Length::Fill);
        #[cfg(feature = "piano")]
        let content = column![content, self.piano_keyboard()]
            .width(Length::Fill)
            .height(Length::Fill)
            .spacing(0);
        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(app_background_style)
            .into()
    }

    #[cfg(feature = "piano")]
    fn piano_keyboard(&self) -> Element<'_, Message> {
        let mut appearances = HashMap::<u8, PianoKeyAppearance>::new();
        for visual in self.piano_strike_visuals.values() {
            let appearance = visual.appearance(self.piano_visual_now);
            appearances
                .entry(visual.midi_note)
                .and_modify(|current| {
                    current.intensity = current.intensity.max(appearance.intensity)
                })
                .or_insert(appearance);
        }
        let keyboard: Element<'_, PianoMessage> =
            canvas::Canvas::new(PianoKeyboard::new(appearances))
                .width(Length::Fill)
                .height(Length::Fixed(PIANO_HEIGHT))
                .into();
        let keyboard: Element<'_, Message> = keyboard.map(Message::Piano);
        let audio_error = self
            .piano_audio_error
            .clone()
            .or_else(|| self.piano_audio.as_ref().and_then(PianoAudioEngine::error));
        let status_error = [audio_error, self.piano_score_error.clone()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" · ");
        let keyboard = if !status_error.is_empty() {
            stack![
                keyboard,
                container(
                    text(status_error)
                        .size(12)
                        .color(Color::from_rgb8(255, 215, 180))
                )
                .padding([4, 8])
                .style(|_theme| container::Style {
                    background: Some(Background::Color(Color::from_rgba8(90, 12, 8, 0.88))),
                    ..container::Style::default()
                }),
            ]
        } else {
            stack![keyboard]
        };
        container(keyboard)
            .width(Length::Fill)
            .height(Length::Fixed(PIANO_HEIGHT))
            .style(|_theme| container::Style {
                background: Some(Background::Color(Color::BLACK)),
                ..container::Style::default()
            })
            .into()
    }

    #[cfg(feature = "piano")]
    fn update_piano(&mut self, message: PianoMessage) {
        match message {
            PianoMessage::Press {
                midi_note,
                velocity,
                source,
            } => self.attack_piano_note(
                PianoInputId::Pointer(source),
                source.public(),
                midi_note,
                velocity,
                None,
            ),
            PianoMessage::Release { source, velocity } => {
                self.release_piano_note(PianoInputId::Pointer(source), velocity)
            }
        }
    }

    #[cfg(feature = "piano")]
    fn update_piano_keyboard(&mut self, event: keyboard::Event) {
        match event {
            keyboard::Event::KeyPressed {
                key,
                physical_key,
                repeat,
                ..
            } if !repeat => {
                let Some(key) = piano_computer_key(&key, physical_key) else {
                    return;
                };
                let Some(midi_note) = piano::computer_key_note(key) else {
                    return;
                };
                self.attack_piano_note(
                    PianoInputId::ComputerKeyboard(key),
                    PianoInputSource::ComputerKeyboard { key },
                    midi_note,
                    PianoEvent::BINARY_VELOCITY,
                    None,
                );
            }
            keyboard::Event::KeyReleased {
                key, physical_key, ..
            } => {
                let Some(key) = piano_computer_key(&key, physical_key) else {
                    return;
                };
                self.release_piano_note(PianoInputId::ComputerKeyboard(key), 0.0);
            }
            _ => {}
        }
    }

    #[cfg(feature = "piano")]
    fn update_piano_score(&mut self, now: Instant) {
        self.update_piano_visuals(now);
        let due = self
            .piano_score
            .as_mut()
            .map(|score| score.take_due(now))
            .unwrap_or_default();
        for due_event in due {
            if due_event.cycle > self.piano_score_cycle {
                let stale = self
                    .active_piano_notes
                    .keys()
                    .filter(|input| {
                        matches!(
                            input,
                            PianoInputId::Score { cycle, .. } if *cycle < due_event.cycle
                        )
                    })
                    .copied()
                    .collect::<Vec<_>>();
                for input in stale {
                    self.release_piano_note(input, 1.0);
                }
                self.piano_score_cycle = due_event.cycle;
            }

            let input = PianoInputId::Score {
                cycle: due_event.cycle,
                voice_id: due_event.event.voice_id,
            };
            match due_event.event.action {
                PianoAction::Attack { velocity, pressure } => self.attack_piano_note(
                    input,
                    PianoInputSource::Score,
                    due_event.event.note.midi_note,
                    velocity,
                    pressure,
                ),
                PianoAction::Release { velocity, .. } => self.release_piano_note(input, velocity),
            }
        }
    }

    #[cfg(feature = "piano")]
    fn update_piano_visuals(&mut self, now: Instant) {
        self.piano_visual_now = now;
        self.piano_strike_visuals
            .retain(|_, visual| !visual.finished(now));
    }

    #[cfg(feature = "piano")]
    fn attack_piano_note(
        &mut self,
        input: PianoInputId,
        source: PianoInputSource,
        midi_note: u8,
        velocity: f32,
        pressure: Option<f32>,
    ) {
        if self.active_piano_notes.contains_key(&input) {
            return;
        }
        let now = Instant::now();
        self.piano_voice_sequence = self.piano_voice_sequence.wrapping_add(1);
        let active = ActivePianoNote {
            note: PianoNote::from_midi(midi_note),
            voice_id: self.piano_voice_sequence,
            started_at: now,
            source,
        };
        self.active_piano_notes.insert(input, active);
        self.piano_strike_visuals.insert(
            active.voice_id,
            PianoStrikeVisual {
                midi_note,
                velocity,
                pressure,
                attacked_at: now,
                released: None,
            },
        );
        self.piano_visual_now = now;
        self.emit_piano_event(PianoEvent {
            sequence: 0,
            timestamp: now.saturating_duration_since(self.piano_started_at),
            voice_id: active.voice_id,
            note: active.note,
            action: PianoAction::Attack {
                velocity: velocity.clamp(0.0, 1.0),
                pressure: pressure.map(|pressure| pressure.clamp(0.0, 1.0)),
            },
            source,
        });
    }

    #[cfg(feature = "piano")]
    fn release_piano_note(&mut self, input: PianoInputId, velocity: f32) {
        let Some(active) = self.active_piano_notes.remove(&input) else {
            return;
        };
        let now = Instant::now();
        if let Some(visual) = self.piano_strike_visuals.get_mut(&active.voice_id) {
            visual.released = Some((now, velocity.clamp(0.0, 1.0)));
        }
        self.piano_visual_now = now;
        self.emit_piano_event(PianoEvent {
            sequence: 0,
            timestamp: now.saturating_duration_since(self.piano_started_at),
            voice_id: active.voice_id,
            note: active.note,
            action: PianoAction::Release {
                velocity: velocity.clamp(0.0, 1.0),
                held_for: now.saturating_duration_since(active.started_at),
            },
            source: active.source,
        });
    }

    #[cfg(feature = "piano")]
    fn emit_piano_event(&mut self, mut event: PianoEvent) {
        self.piano_event_sequence = self.piano_event_sequence.wrapping_add(1);
        event.sequence = self.piano_event_sequence;
        if let Some(audio) = &self.piano_audio {
            audio.perform(event);
        }
        if let Some(handler) = &self.config.piano_event_handler {
            handler(event);
        }
    }

    fn cell_graph(&self) -> Element<'_, Message> {
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

        let graph = build_sun_graph(
            self.model.graph.clone(),
            labels,
            styles,
            self.config.layout,
            self.config.animation_duration,
            self.config.animation_easing,
            |node_id: u32, animal_name: String, phase_label: String, style| {
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
        .id(iced_sugiyama::Id::new(CELL_GRAPH_ID))
        .padding(24)
        .auto_fit(AutoFit::Off)
        .keep_centered(false);

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
                // A warp cell whose subgraph is merged into the main graph is
                // colored like a regular node rather than as a warp boundary.
                let warp = !cell.warp_journey_id.is_nil() && !self.is_warp_merged(cell.id);
                let style = self
                    .visuals
                    .get(&cell.id)
                    .map(|visual| visual.style(cell.grad_steps, cell.frozen, warp, self.color_now))
                    .unwrap_or_else(|| {
                        if warp {
                            warp_node_style_colors(
                                cell.state,
                                cell.grad_step,
                                cell.grad_steps,
                                cell.frozen,
                            )
                        } else {
                            node_style_colors(
                                cell.state,
                                cell.grad_step,
                                cell.grad_steps,
                                cell.frozen,
                            )
                        }
                    });
                (cell.id, style)
            })
            .collect()
    }

    /// Opens the child-flow subpanel for a node. Clicking a warp cell also
    /// toggles its nested sun in and out of the main graph.
    fn open_subpanel_for_node(&mut self, node_id: u32) -> Task<Message> {
        let Some(client) = self.live.as_ref().map(|live| live.client.clone()) else {
            return Task::none();
        };

        let Some(cell) = self.model.cells.iter().find(|cell| cell.id == node_id) else {
            return Task::none();
        };
        let journey_id = cell.journey_id;
        let animal_name = cell.animal_name.clone();
        let is_warp = !cell.warp_journey_id.is_nil();

        self.subpanel_notice = None;
        if journey_id.is_nil() {
            self.subpanel_notice = Some(format!(
                "Cell {node_id} does not have an active journey yet."
            ));
            return Task::none();
        }

        let Some(subpanel_config) = self.resolve_subpanel_config(&animal_name) else {
            self.subpanel_notice = Some(format!(
                "No registered subpanel animal for {}.",
                animal_name
            ));
            return Task::none();
        };

        // The requested child flow is already on display; clicking the same
        // node again closes it and collapses any warp subgraph it expanded.
        let closing = self
            .subpanel
            .as_ref()
            .is_some_and(|subpanel| subpanel.journey_id == journey_id);
        if is_warp {
            let path = self.model.warp_paths.get(&node_id).cloned();
            if let Some(path) = path {
                if closing {
                    self.collapse_warp_path(&path);
                } else {
                    self.expanded_warp_cells.insert(path);
                }
            }
        }

        if closing {
            self.subpanel = None;
        } else {
            // A warp cell's nested sun joins the main graph once its warp
            // journey exposes a decodable `SunAppearance`; until then,
            // explain why the subgraph is missing.
            if is_warp && !self.warp_appearance_available(node_id) {
                self.subpanel_notice = Some(match self.warp_diagnostic(node_id) {
                    Some(diagnostic) => format!("Cell {node_id}: {diagnostic}"),
                    None => format!(
                        "Cell {node_id}'s warp journey has not exposed its Black Hole Sun appearance yet; its subgraph will join the main graph once it does."
                    ),
                });
            }

            self.subpanel = Some(SubpanelState {
                node_id,
                title: subpanel_config.title,
                journey_id,
                viewer: (subpanel_config.build_viewer)(SharedJungleClient::new(client), journey_id),
            });
        }

        if is_warp {
            // The main graph topology changed with the expansion.
            return self.rebuild_model();
        }
        Task::none()
    }

    /// Whether a warp cell's nested sun is currently merged into the main
    /// graph.
    fn is_warp_merged(&self, node_id: u32) -> bool {
        self.model
            .warp_paths
            .get(&node_id)
            .is_some_and(|path| self.is_path_expanded(path))
    }

    /// Whether every level of a warp path is expanded, i.e. the subgraph at
    /// that path joins the main graph.
    fn is_path_expanded(&self, path: &[u32]) -> bool {
        (1..=path.len())
            .all(|len| self.expanded_warp_cells.contains(&path[..len].to_vec()))
    }

    /// Removes a warp path and every expanded sub-path beneath it, so
    /// collapsing a subgraph also collapses its expanded child subgraphs.
    fn collapse_warp_path(&mut self, path: &[u32]) {
        let collapsed: Vec<Vec<u32>> = self
            .expanded_warp_cells
            .iter()
            .filter(|expanded| expanded.starts_with(path))
            .cloned()
            .collect();
        for expanded in collapsed {
            self.expanded_warp_cells.remove(&expanded);
        }
    }

    /// The warp appearances from the latest snapshot whose full path is
    /// expanded; these are the subgraphs that join the main graph.
    fn expanded_warp_appearances<'a>(
        &'a self,
        snapshot: &'a LiveAppearanceSnapshot,
    ) -> HashMap<Vec<u32>, SunAppearance> {
        snapshot
            .warp_appearances
            .iter()
            .filter(|(path, _)| self.is_path_expanded(path))
            .map(|(path, appearance)| (path.clone(), appearance.clone()))
            .collect()
    }

    /// Whether the latest snapshot carries a usable (finalized) nested sun
    /// for a warp cell.
    fn warp_appearance_available(&self, node_id: u32) -> bool {
        let Some(path) = self.model.warp_paths.get(&node_id) else {
            return false;
        };
        self.last_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.warp_appearances.get(path))
        .is_some_and(|appearance| appearance.finalized)
    }

    /// The latest diagnostic explaining why a warp cell's nested sun could
    /// not be used, if any.
    fn warp_diagnostic(&self, node_id: u32) -> Option<&String> {
        let path = self.model.warp_paths.get(&node_id)?;
        self.last_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.warp_diagnostics.get(path))
    }

    /// Rebuilds the main graph from the latest snapshot, merging only the
    /// warp subgraphs whose full path is expanded.
    fn rebuild_model(&mut self) -> Task<Message> {
        let Some(snapshot) = &self.last_snapshot else {
            return Task::none();
        };
        let warp_appearances = self.expanded_warp_appearances(snapshot);
        let Ok(model) = BeamModel::from_appearance(
            snapshot.appearance.clone(),
            &snapshot.child_rays,
            &warp_appearances,
        ) else {
            return Task::none();
        };

        let now = Instant::now();
        let node_ids = model
            .cells
            .iter()
            .map(|cell| cell.id)
            .collect::<HashSet<_>>();
        self.visuals.retain(|node_id, _| node_ids.contains(node_id));
        for cell in &model.cells {
            self.visuals.entry(cell.id).or_default().observe(
                cell.state,
                cell.grad_step,
                cell.grad_steps,
                cell.state_sequence,
                cell.frozen,
                now,
            );
        }
        self.model = model;
        self.color_now = now;
        iced_sugiyama::force_review(iced_sugiyama::Id::new(CELL_GRAPH_ID))
    }

    fn resolve_subpanel_config(&self, animal_label: &str) -> Option<SubpanelConfig> {
        let lookup_key = animal_label_key(animal_label);
        self.config
            .subpanel_animals
            .iter()
            .find(|config| config.animal_label == lookup_key)
            .or_else(|| {
                // Warp nodes are labeled `Warp<WarpAnimal, BoundaryAnimal>`
                // and run the boundary animal's journey, so a registered
                // boundary animal stands in for the whole warp node.
                warp_boundary_label(&lookup_key).and_then(|boundary_key| {
                    self.config
                        .subpanel_animals
                        .iter()
                        .find(|config| config.animal_label == boundary_key)
                })
            })
            .cloned()
    }

    fn subpanel_phase(&self, node_id: u32) -> Option<String> {
        let cell = self.model.cells.iter().find(|cell| cell.id == node_id)?;
        let (state, grad_step) = self
            .visuals
            .get(&node_id)
            .map(|visual| (visual.current.state, visual.current.grad_step))
            .unwrap_or((cell.state, cell.grad_step));
        match state {
            // Only propagation phases advance through gradient steps.
            SunNodeState::Propagation1 | SunNodeState::Propagation2 => {
                Some(format!("{}, step {grad_step}", state.label()))
            }
            SunNodeState::Optimization => {
                let frozen = self
                    .visuals
                    .get(&node_id)
                    .map(|visual| visual.frozen_for_state(state, cell.frozen))
                    .unwrap_or(cell.frozen);
                let status = if frozen == Some(true) {
                    "frozen"
                } else {
                    "optimizing"
                };
                Some(format!("{} [{status}]", state.label()))
            }
            _ => Some(state.label().to_string()),
        }
    }
}

/// Builds a Sugiyama graph widget for one Black Hole Sun model.
fn build_sun_graph(
    graph: Graph,
    labels: HashMap<u32, (String, String)>,
    styles: HashMap<u32, NodeStyleColors>,
    layout: BeamLayout,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
    view_node: impl Fn(u32, String, String, NodeStyleColors) -> Element<'static, Message> + 'static,
) -> Sugiyama<'static, Message, Theme, iced::Renderer> {
    let mut layout_graph = graph;
    // Match the spacing used by iced-sugiyama's "moar" example.
    layout_graph.config.vertex_spacing = DOT_VERTEX_SPACING;

    let node_labels = labels.clone();
    let styles_for_nodes = styles.clone();
    let view_node = move |node_id: u32| {
        let (animal_name, phase_label) = node_labels.get(&node_id).cloned().unwrap_or((
            format!("cell {node_id}"),
            SunNodeState::Idle.label().to_string(),
        ));
        let style = styles_for_nodes
            .get(&node_id)
            .copied()
            .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None));
        view_node(node_id, animal_name, phase_label, style)
    };

    let mut graph =
        Sugiyama::<Message, Theme, iced::Renderer>::new(Cow::Owned(layout_graph), view_node);

    if matches!(layout, BeamLayout::Circo) {
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
            let remapped_nodes = Arc::from((0..original_nodes.len() as u32).collect::<Vec<_>>());
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
    } else if matches!(layout, BeamLayout::Microdot) {
        graph = graph.layout_fn(microdot_layout);
    }

    let styles_for_edges = styles.clone();
    let styles_for_endpoints = styles;
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
            // iced-sugiyama paints the first tuple element at the edge's
            // head and the second at its tail, so pass (end, start) to
            // gradient from the source color into the target color.
            (end, start)
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
        .edge_corner_radius(16.0);

    if let Some(duration) = animation_duration {
        graph = graph.animation_duration(duration);
    }
    if let Some(easing) = animation_easing {
        graph = graph.animation_easing(easing);
    }
    graph
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
    let (warp_appearances, warp_diagnostics) = fetch_warp_appearances(&live, &appearance).await;
    Ok(Some(LiveAppearanceSnapshot {
        appearance,
        child_rays,
        warp_appearances,
        warp_diagnostics,
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

/// How many levels of nested warps to fetch. Guards against cyclic warp
/// journeys, which would otherwise be fetched forever.
const MAX_WARP_DEPTH: usize = 16;

/// Fetches the nested Sun appearance behind every warp cell, recursing into
/// each nested sun so that warps within warps are discovered as well.
///
/// Returns the decodable appearances together with a per-cell diagnostic for
/// every warp cell that did not yield one, both keyed by the path of cell
/// ids from the top level, so the UI can explain why a warp node has not
/// expanded.
async fn fetch_warp_appearances(
    live: &LiveConfig,
    appearance: &SunAppearance,
) -> (HashMap<Vec<u32>, SunAppearance>, HashMap<Vec<u32>, String>) {
    let mut warp_appearances = HashMap::new();
    let mut warp_diagnostics = HashMap::new();
    // Depth-first worklist of (sun to scan, path prefix). Iterative so that
    // arbitrarily deep warp chains do not recurse.
    let mut stack: Vec<(SunAppearance, Vec<u32>)> = vec![(appearance.clone(), Vec::new())];
    while let Some((sun, prefix)) = stack.pop() {
        for node in &sun.nodes {
            if node.warp_journey_id.is_nil() {
                continue;
            }
            let mut path = prefix.clone();
            path.push(node.id);
            if path.len() > MAX_WARP_DEPTH {
                warp_diagnostics.insert(
                    path,
                    format!(
                        "warp journey {} is nested more than {MAX_WARP_DEPTH} levels deep; deeper subgraphs are not fetched",
                        node.warp_journey_id
                    ),
                );
                continue;
            }
            match fetch_sun_appearance(live, node.warp_journey_id).await {
                Ok(warp_appearance) => {
                    warp_appearances.insert(path.clone(), warp_appearance.clone());
                    stack.push((warp_appearance, path));
                }
                Err(diagnostic) => {
                    warp_diagnostics.insert(path, diagnostic);
                }
            }
        }
    }
    (warp_appearances, warp_diagnostics)
}

/// Fetches and decodes one journey's Sun appearance, with a diagnostic
/// message for every way it can fail.
async fn fetch_sun_appearance(
    live: &LiveConfig,
    journey_id: Uuid,
) -> Result<SunAppearance, String> {
    match live.client.animal_appearance(journey_id).await {
        Ok(Some(bytes)) => postcard::from_bytes::<SunAppearance>(&bytes).map_err(|error| {
            format!(
                "warp journey {journey_id} published an appearance that is not a decodable Black Hole Sun (SunAppearance): {error}"
            )
        }),
        Ok(None) => Err(format!(
            "warp journey {journey_id} has not published an appearance yet"
        )),
        Err(error) => Err(format!("fetching warp journey {journey_id} failed: {error}")),
    }
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

fn subpanel_notice_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgb8(195, 24, 41))),
        text_color: Some(Color::WHITE),
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
        // The panel is outlined only on its left edge, which is rendered as a
        // vertical rule because iced borders apply to all sides.
        text_color: Some(colors.text),
        ..Default::default()
    }
}

fn subpanel_left_edge_style(colors: NodeStyleColors) -> iced::widget::rule::Style {
    iced::widget::rule::Style {
        color: Color::from_rgba(colors.border.r, colors.border.g, colors.border.b, 0.58),
        radius: 0.0.into(),
        fill_mode: iced::widget::rule::FillMode::Full,
        snap: true,
    }
}

fn subpanel_overlay_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(3, 3, 3, 0.7))),
        border: iced::Border {
            color: Color::from_rgba8(120, 120, 120, 0.25),
            width: 1.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn subpanel_child_canvas_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        // Child graph canvases should feel translucent without reducing text legibility.
        // Near-black with a faint blue tint to match the beam jungle theme.
        background: Some(Background::Color(Color::from_rgba8(5, 7, 14, 0.7))),
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
        shadow: Shadow::default(),
        snap: false,
        ..Default::default()
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

/// Extracts the boundary animal label from a warp node label of the form
/// `Warp<WarpAnimal, BoundaryAnimal>`.
///
/// Animal labels may carry generic arguments with nested angle brackets and
/// commas, so the split happens at the first top-level comma.
fn warp_boundary_label(label: &str) -> Option<String> {
    let inner = label.strip_prefix("Warp<")?.strip_suffix('>')?;
    let mut depth = 0i32;
    for (index, char) in inner.char_indices() {
        match char {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(animal_label_key(&inner[index + 1..])),
            _ => {}
        }
    }
    None
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
    use black_hole_flux::sun::{
        Binary, BlackHole, Manifest, StatelessManifest, SunEdgeAppearance, SunNodeAppearance, Unary,
    };
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
    struct CustomStateManifest;

    impl Manifest for CustomStateManifest {
        type Generator = Primordium;
        type Policy = Primordium;
        type State = (String, String);
    }

    type TestSun = <TestGraph as BlackHole>::Sun<StatelessManifest<Primordium, Primordium>, 1>;
    type TestSunWithCustomState = <TestGraph as BlackHole>::Sun<CustomStateManifest, 1>;
    type TestBinaryGraph = List<(Binary<U0, U1, TestFusion, Empty>, Empty)>;
    type TestBinarySun =
        <TestBinaryGraph as BlackHole>::Sun<StatelessManifest<Primordium, Primordium>, 1>;

    #[test]
    fn uses_black_hole_sun_title_and_grad_step_palette() {
        assert_eq!(BeamBuilder::default().title, "Black Hole Sun");
        assert!(matches!(BeamBuilder::default().layout, BeamLayout::Circo));
        assert!(matches!(
            BeamBuilder::new().microdot_layout().layout,
            BeamLayout::Microdot
        ));
        let p1_step1 = node_style_colors(SunNodeState::Propagation1, 1, 4, None).body;
        let p1_step4 = node_style_colors(SunNodeState::Propagation1, 4, 4, None).body;
        assert!(p1_step4.g < p1_step1.g);
        assert!(p1_step4.r - p1_step4.g > p1_step1.r - p1_step1.g);

        let p2_step1 = node_style_colors(SunNodeState::Propagation2, 1, 4, None).body;
        let p2_step4 = node_style_colors(SunNodeState::Propagation2, 4, 4, None).body;
        assert!(p2_step4.g > p2_step1.g);

        let optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(false));
        assert_eq!(optimize.body, Color::from_rgb8(255, 255, 255));
        assert_eq!(optimize.text, Color::from_rgb8(18, 12, 8));
        assert_eq!(optimize.border, Color::from_rgb8(65, 105, 225));

        let frozen_optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        assert_eq!(frozen_optimize.body, Color::BLACK);
        assert_eq!(frozen_optimize.border, Color::from_rgb8(124, 77, 255));
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
    fn extracts_warp_boundary_labels() {
        assert_eq!(
            warp_boundary_label("Warp<MyWarp, MyBoundary>").as_deref(),
            Some("MyBoundary")
        );
        assert_eq!(
            warp_boundary_label("Warp<Outer<A, B>, Inner<C>>").as_deref(),
            Some("Inner<C>"),
            "the split happens at the first top-level comma"
        );
        assert_eq!(warp_boundary_label("MyWarp<A, B>"), None);
        assert_eq!(warp_boundary_label("Warp<Solo>"), None);
        assert_eq!(warp_boundary_label("Warp<A,"), None);
    }

    #[test]
    fn warp_node_subpanel_resolves_registered_boundary_animal() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let (app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        assert_eq!(
            app.resolve_subpanel_config("Warp<OtherAnimal, TestCell>")
                .map(|config| config.animal_label),
            Some("TestCell".to_string()),
            "the registered boundary animal stands in for the warp node"
        );
        assert_eq!(
            app.resolve_subpanel_config("Warp<TestCell, OtherAnimal>")
                .map(|config| config.animal_label),
            None,
            "registering the warp animal does not open the warp node"
        );
        assert_eq!(
            app.resolve_subpanel_config("TestCell")
                .map(|config| config.animal_label),
            Some("TestCell".to_string()),
            "direct registrations still match plain nodes"
        );
    }

    #[test]
    fn warp_node_click_opens_the_boundary_journey_subpanel() {
        let boundary_journey = Uuid::new_v4();
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let mut cell = CellDefinition::new::<TestCell>(7, vec![0], vec![]);
        cell.animal_name = "Warp<OtherAnimal, TestCell>".to_string();
        cell.journey_id = boundary_journey;
        app.model.cells.push(cell);

        app.open_subpanel_for_node(7);

        let subpanel = app
            .subpanel
            .as_ref()
            .expect("the warp node should open its subpanel");
        assert_eq!(subpanel.node_id, 7);
        assert_eq!(subpanel.title, "TestCell");
        assert_eq!(
            subpanel.journey_id, boundary_journey,
            "the subpanel shows the boundary's live journey"
        );
    }

    fn nested_sun_appearance(finalized: bool) -> SunAppearance {
        SunAppearance {
            finalized,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::nil(),
                label: "NestedRoot".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        }
    }

    #[test]
    fn warp_subgraph_joins_main_graph_when_nested_appearance_arrives() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let warp_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: warp_journey,
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };

        // First poll: the nested appearance has not been exposed yet.
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(
            LiveAppearanceSnapshot {
                appearance: appearance.clone(),
                child_rays: HashMap::new(),
                warp_appearances: HashMap::new(),
                warp_diagnostics: HashMap::from([(
                    vec![7],
                    "warp journey 0f3c has not published an appearance yet".to_string(),
                )]),
            },
        ))));

        // Clicking the warp node opens the subpanel and explains why its
        // subgraph is missing...
        let _task = app.update(Message::NodeSelected(7));
        assert!(app.subpanel.is_some());
        let notice = app
            .subpanel_notice
            .as_deref()
            .expect("a missing nested appearance should be surfaced to the user");
        assert!(
            notice.starts_with("Cell 7:") && notice.contains("warp journey"),
            "the notice should carry the per-cell diagnostic, got: {notice}"
        );

        // ...until the nested appearance arrives with the subpanel open.
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(
            LiveAppearanceSnapshot {
                appearance,
                child_rays: HashMap::new(),
                warp_appearances: HashMap::from([(vec![7], nested_sun_appearance(true))]),
                warp_diagnostics: HashMap::new(),
            },
        ))));

        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8],
            "the nested node joins the main graph under a fresh id"
        );
        assert_eq!(
            app.model.graph.edges,
            vec![(7, 8)],
            "the boundary cell connects to the nested sink"
        );
        assert!(
            app.expanded_warp_cells.contains(&vec![7]),
            "the expanded warp cell is tracked for rebuilds"
        );
        assert_eq!(
            app.subpanel_notice, None,
            "the notice clears once the subgraph has joined the main graph"
        );

        // Clicking the same node again closes the subpanel and collapses
        // the merged subgraph out of the main graph.
        let _task = app.update(Message::NodeSelected(7));
        assert!(app.subpanel.is_none());
        assert!(!app.expanded_warp_cells.contains(&vec![7]));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "collapsing removes the nested nodes from the main graph"
        );
        assert!(
            app.model.graph.edges.is_empty(),
            "the connector edge goes away with the subgraph"
        );
    }

    #[test]
    fn plain_node_click_opens_the_subpanel() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let mut cell = CellDefinition::new::<TestCell>(3, vec![0], vec![]);
        cell.journey_id = Uuid::new_v4();
        app.model.cells.push(cell);

        app.open_subpanel_for_node(3);
        assert!(app.subpanel.is_some());
        assert_eq!(app.subpanel_notice, None);
    }

    #[test]
    fn from_appearance_merges_finalized_warp_subgraphs() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 3,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<A, B>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 5,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![2],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![],
        };
        let warp_appearances = HashMap::from([
            (vec![3], nested_sun_appearance(true)),
            (vec![5], nested_sun_appearance(false)),
        ]);

        let model = BeamModel::from_appearance(appearance, &HashMap::new(), &warp_appearances)
            .unwrap();

        // The finalized subgraph is merged under fresh ids; the unfinalized
        // one is skipped.
        assert_eq!(
            model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![0, 3, 5, 6]
        );
        assert_eq!(
            model.graph.edges,
            vec![(3, 6)],
            "the boundary cell connects to the nested sink"
        );
    }

    #[test]
    fn from_appearance_merges_nested_warps_recursively() {
        let outer = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<A, B>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        // Cell 2 is itself a warp cell inside the nested sun; the edge makes
        // it the unique sink of the middle sun.
        let middle = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "NestedRoot".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 2,
                target_port: 1,
            }],
        };
        let inner = nested_sun_appearance(true);

        let warp_appearances = HashMap::from([
            (vec![7], middle.clone()),
            (vec![7, 2], inner),
        ]);
        let model = BeamModel::from_appearance(outer, &HashMap::new(), &warp_appearances)
            .unwrap();

        assert_eq!(
            model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9, 10],
            "every level of the warp chain joins the main graph"
        );
        assert_eq!(
            model.graph.edges,
            vec![(7, 9), (8, 9), (9, 10)],
            "the boundary connects to the middle sink, which in turn connects to the inner sink"
        );
        assert_eq!(
            model.warp_paths.get(&7),
            Some(&vec![7]),
            "the top-level warp is locatable by its own id"
        );
        assert_eq!(
            model.warp_paths.get(&9),
            Some(&vec![7, 2]),
            "the merged nested warp keeps its path from the top level"
        );
    }

    #[test]
    fn nested_warp_subgraphs_expand_and_collapse_recursively() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let boundary_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: boundary_journey,
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        // The nested sun contains its own warp cell (id 2), whose sub-sun is
        // exposed as well.
        let middle = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "NestedRoot".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<InnerAnimal, TestCell>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 2,
                target_port: 1,
            }],
        };

        let snapshot = LiveAppearanceSnapshot {
            appearance,
            child_rays: HashMap::new(),
            warp_appearances: HashMap::from([
                (vec![7], middle.clone()),
                (vec![7, 2], nested_sun_appearance(true)),
            ]),
            warp_diagnostics: HashMap::new(),
        };
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(snapshot))));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "nothing merges until a warp cell is clicked"
        );

        // Expand the top-level warp: its nested sun joins the main graph.
        let _task = app.update(Message::NodeSelected(7));
        assert_eq!(app.expanded_warp_cells, HashSet::from([vec![7]]));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9]
        );

        // Expand the merged nested warp: its sub-sun joins as well.
        let _task = app.update(Message::NodeSelected(9));
        assert_eq!(
            app.expanded_warp_cells,
            HashSet::from([vec![7], vec![7, 2]]),
            "both levels of the warp chain are tracked"
        );
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9, 10],
            "the deeper sub-sun joins the main graph"
        );
        assert_eq!(app.model.graph.edges, vec![(7, 9), (8, 9), (9, 10)]);

        // Clicking the top-level warp again while its child subgraph is
        // still expanded only switches the subpanel back to it...
        let _task = app.update(Message::NodeSelected(7));
        assert_eq!(
            app.expanded_warp_cells,
            HashSet::from([vec![7], vec![7, 2]]),
            "the child subgraph stays expanded while its parent is re-clicked"
        );

        // ...and clicking it once more closes the subpanel and collapses the
        // whole chain, including the still-expanded child subgraph.
        let _task = app.update(Message::NodeSelected(7));
        assert!(
            app.expanded_warp_cells.is_empty(),
            "collapsing the parent also collapses its expanded child subgraphs"
        );
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "the whole merged chain leaves the main graph"
        );
        assert!(app.model.graph.edges.is_empty());
    }

    #[test]
    fn expanded_warp_cells_are_colored_like_regular_nodes() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        let snapshot = LiveAppearanceSnapshot {
            appearance,
            child_rays: HashMap::new(),
            warp_appearances: HashMap::from([(vec![7], nested_sun_appearance(true))]),
            warp_diagnostics: HashMap::new(),
        };
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(snapshot))));

        let styles = app.cell_styles();
        assert_eq!(
            styles[&7].body,
            Color::BLACK,
            "an unexpanded warp cell keeps the black body"
        );

        // Expanding the subgraph recolors the boundary node like a regular
        // one.
        let _task = app.update(Message::NodeSelected(7));
        let styles = app.cell_styles();
        assert_eq!(
            styles[&7],
            node_style_colors(SunNodeState::Idle, 1, 1, None),
            "an expanded warp cell is colored like a regular node"
        );

        // Collapsing it restores the warp styling.
        let _task = app.update(Message::NodeSelected(7));
        let styles = app.cell_styles();
        assert_eq!(
            styles[&7].body,
            Color::BLACK,
            "collapsing the subgraph restores the warp styling"
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_events_preserve_order_timing_and_overlapping_voices() {
        use std::sync::Mutex;

        let events = Arc::new(Mutex::new(Vec::<PianoEvent>::new()));
        let captured = Arc::clone(&events);
        let config = BeamBuilder::new()
            .on_piano_event(move |event| captured.lock().unwrap().push(event))
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        app.attack_piano_note(
            PianoInputId::ComputerKeyboard('z'),
            PianoInputSource::ComputerKeyboard { key: 'z' },
            60,
            PianoEvent::BINARY_VELOCITY,
            None,
        );
        app.attack_piano_note(
            PianoInputId::Pointer(PianoPointerSource::Mouse),
            PianoInputSource::Mouse,
            60,
            0.42,
            Some(0.7),
        );
        app.release_piano_note(PianoInputId::ComputerKeyboard('z'), 0.0);
        assert_eq!(app.active_piano_notes.len(), 1);
        app.release_piano_note(PianoInputId::Pointer(PianoPointerSource::Mouse), 0.31);

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(events[0].note.midi_note, 60);
        assert_eq!(events[0].voice_id, events[2].voice_id);
        assert_eq!(events[1].voice_id, events[3].voice_id);
        assert_ne!(events[0].voice_id, events[1].voice_id);
        assert!(events
            .windows(2)
            .all(|events| events[0].timestamp <= events[1].timestamp));
        assert!(matches!(
            events[1].action,
            PianoAction::Attack {
                velocity,
                pressure: Some(pressure),
            } if velocity == 0.42 && pressure == 0.7
        ));
        assert!(matches!(
            events[3].action,
            PianoAction::Release { velocity, .. } if velocity == 0.31
        ));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_strike_opacity_tracks_velocity_pressure_and_release_speed() {
        let start = Instant::now();
        let visual = |velocity, pressure| PianoStrikeVisual {
            midi_note: 60,
            velocity,
            pressure,
            attacked_at: start,
            released: None,
        };
        let settled = start + Duration::from_millis(80);
        assert!(
            visual(0.9, Some(0.8)).appearance(settled).intensity
                > visual(0.3, Some(0.2)).appearance(settled).intensity
        );
        assert!(
            visual(0.6, Some(1.0)).appearance(settled).intensity
                > visual(0.6, Some(0.0)).appearance(settled).intensity
        );

        let released_at = start + Duration::from_millis(100);
        let slow_release = PianoStrikeVisual {
            released: Some((released_at, 0.0)),
            ..visual(0.8, Some(0.7))
        };
        let fast_release = PianoStrikeVisual {
            released: Some((released_at, 1.0)),
            ..visual(0.8, Some(0.7))
        };
        let fading = released_at + Duration::from_millis(80);
        assert!(
            slow_release.appearance(fading).intensity > fast_release.appearance(fading).intensity
        );
        assert!(fast_release.needs_frame(fading));
        assert!(fast_release.finished(released_at + Duration::from_millis(400)));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_a_score_path() {
        let config = BeamBuilder::new().score_path("moonlight.bhs").into_config();
        assert_eq!(
            config.piano_score_path,
            Some(PathBuf::from("moonlight.bhs"))
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_score_data() {
        let data = b"format bhs-score-v1";
        let config = BeamBuilder::new().score_data(data).into_config();
        assert_eq!(config.piano_score_data.as_deref(), Some(data.as_slice()));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_an_owned_score() {
        let score = BhsScore::parse("format bhs-score-v1\nticks_per_second 960\n0 960 C4 80\n")
            .expect("the fixture should parse");
        let config = BeamBuilder::new().score(score).into_config();
        assert_eq!(
            config
                .piano_score
                .as_ref()
                .map(|score| score.ticks_per_second),
            Some(960)
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn score_ticks_loop_through_audio_event_and_visual_paths() {
        use std::sync::Mutex;

        let start = Instant::now();
        // A 0.5s C4 at 2000 ticks/second, looping every 0.5s: the release
        // lands exactly when the next cycle's attack is due.
        let score_text = "\
format bhs-score-v1
ticks_per_second 2000
loop_ticks 1000
0 1000 C4 98 36
";
        let score = PianoScorePlayback::from_text(score_text, start).unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let callback_events = Arc::clone(&captured);
        let config = BeamBuilder::new()
            .on_piano_event(move |event| callback_events.lock().unwrap().push(event))
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        app.piano_score = Some(score);

        app.update_piano_score(start);
        assert_eq!(app.active_piano_notes.len(), 1);
        assert_eq!(app.piano_strike_visuals.len(), 1);

        app.update_piano_score(start + Duration::from_millis(500));
        assert_eq!(
            app.active_piano_notes.len(),
            1,
            "cycle 1 attacks immediately"
        );
        assert_eq!(app.piano_strike_visuals.len(), 2);
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 3);
        let attack_velocity = f32::from(98u8) / 127.0;
        let release_velocity = f32::from(36u8) / 127.0;
        assert!(matches!(
            captured[0].action,
            PianoAction::Attack {
                velocity,
                pressure: None
            } if (velocity - attack_velocity).abs() < 1e-6
        ));
        assert!(matches!(
            captured[1].action,
            PianoAction::Release { velocity, .. }
                if (velocity - release_velocity).abs() < 1e-6
        ));
        assert_eq!(captured[0].source, PianoInputSource::Score);
        assert_eq!(captured[2].source, PianoInputSource::Score);
        assert_ne!(captured[0].voice_id, captured[2].voice_id);
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
            visual.style(4, None, false, start).body,
            idle,
            "the fade starts from the previous activity color"
        );
        assert_eq!(
            visual
                .style(4, None, false, start + COLOR_FADE_DURATION / 2)
                .body,
            lerp_color(idle, p1_step1, 0.5),
            "the color is blended halfway through the fade"
        );
        assert_eq!(
            visual
                .style(4, None, false, start + COLOR_FADE_DURATION)
                .body,
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
            false,
            start + MIN_COLOR_STATE_DURATION * 2 + COLOR_FADE_DURATION,
        );
        assert_eq!(
            style.body,
            Color::BLACK,
            "optimization style keeps the frozen color captured at propagation1 -> propagation2"
        );
    }

    #[test]
    fn warp_nodes_use_black_body_and_white_text_with_phase_border() {
        let base = node_style_colors(SunNodeState::Propagation1, 2, 4, None);
        let warp = warp_node_style_colors(SunNodeState::Propagation1, 2, 4, None);
        assert_eq!(warp.body, Color::BLACK);
        assert_eq!(warp.text, Color::WHITE);
        assert_eq!(
            warp.border, base.border,
            "the phase border color is preserved on warp nodes"
        );

        let frozen_base = node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        let frozen_warp = warp_node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        assert_eq!(frozen_warp.body, Color::BLACK);
        assert_eq!(frozen_warp.text, Color::WHITE);
        assert_eq!(frozen_warp.border, frozen_base.border);
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
                    warp_journey_id: Uuid::nil(),
                    label: "Fusion".to_string(),
                    input_ports: vec![2, 3],
                    state: SunNodeState::Optimization,
                    state_sequence: 3,
                    grad_step: 4,
                },
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
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
        let model =
            BeamModel::from_appearance(decoded, &HashMap::new(), &HashMap::new()).unwrap();

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
                warp_journey_id: Uuid::nil(),
                label: "Root".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Optimization,
                state_sequence: 4,
                grad_step: 1,
            }],
            edges: vec![],
        };
        let rays = HashMap::from([(frozen_journey, Ray { frozen: true })]);
        let model = BeamModel::from_appearance(appearance, &rays, &HashMap::new()).unwrap();
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
                    warp_journey_id: Uuid::nil(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 1,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
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

        let error = BeamModel::from_appearance(appearance, &HashMap::new(), &HashMap::new())
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
                warp_journey_id: Uuid::nil(),
                label: "RootAnimal<Result<String, Vec<u8>>>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };

        let model = BeamModel::from_appearance(appearance, &HashMap::new(), &HashMap::new())
            .unwrap();
        assert_eq!(
            model.cells[0].animal_name,
            "RootAnimal<Result<String, Vec<u8>>>"
        );
    }

    #[test]
    fn labels_optimization_phase_as_potentiation() {
        assert_eq!(SunNodeState::Optimization.label(), "potentiation");
    }

    #[test]
    fn subpanel_phase_reports_potentiation_frozen_status() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let mut frozen_cell = CellDefinition::new::<TestCell>(1, vec![0], vec![]);
        frozen_cell.state = SunNodeState::Optimization;
        frozen_cell.frozen = Some(true);
        app.model.cells.push(frozen_cell);

        let mut open_cell = CellDefinition::new::<TestCell>(2, vec![0], vec![]);
        open_cell.state = SunNodeState::Optimization;
        open_cell.frozen = Some(false);
        app.model.cells.push(open_cell);

        assert_eq!(
            app.subpanel_phase(1).as_deref(),
            Some("potentiation [frozen]")
        );
        assert_eq!(
            app.subpanel_phase(2).as_deref(),
            Some("potentiation [optimizing]")
        );
    }

    #[test]
    fn subpanel_phase_uses_frozen_state_captured_at_propagation_transition() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let mut cell = CellDefinition::new::<TestCell>(1, vec![0], vec![]);
        cell.state = SunNodeState::Optimization;
        cell.frozen = Some(false);
        app.model.cells.push(cell);

        let start = Instant::now();
        let mut visual = CellVisualState::default();
        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, Some(true), start));
        assert!(visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            Some(true),
            start + MIN_COLOR_STATE_DURATION
        ));
        assert!(visual.observe(
            SunNodeState::Optimization,
            4,
            4,
            3,
            Some(false),
            start + MIN_COLOR_STATE_DURATION * 2
        ));
        app.visuals.insert(1, visual);

        assert_eq!(
            app.subpanel_phase(1).as_deref(),
            Some("potentiation [frozen]"),
            "the subpanel matches the violet frozen style captured at propagation1 -> propagation2"
        );
    }
}

#[cfg(test)]
mod warp_fetch_diagnostics {
    use super::*;
    use black_hole_flux::sun::SunNodeAppearance;

    fn warp_appearance_with(journey_id: Uuid) -> SunAppearance {
        SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id,
                warp_journey_id: Uuid::nil(),
                label: "NestedRoot".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        }
    }

    #[test]
    fn reports_why_each_warp_cell_has_no_nested_model() {
        let missing = Uuid::new_v4();
        let foreign = Uuid::new_v4();
        let valid = Uuid::new_v4();
        // Bytes that postcard will not decode as a SunAppearance.
        let foreign_bytes = postcard::to_allocvec(&1u8).unwrap();

        let client = Arc::new(
            jungle_client::MockClient::builder()
                .on_flow_appearance(move |id| {
                    let foreign_bytes = foreign_bytes.clone();
                    async move {
                        Ok(match id {
                            i if i == missing => None,
                            i if i == foreign => Some(foreign_bytes.clone()),
                            i if i == valid => {
                                Some(postcard::to_allocvec(&warp_appearance_with(i)).unwrap())
                            }
                            _ => None,
                        })
                    }
                })
                .build(),
        );
        let live = LiveConfig {
            client,
            journey_id: Uuid::new_v4(),
        };

        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 1,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: missing,
                    label: "Warp<A, B>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: foreign,
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 3,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: valid,
                    label: "Warp<E, F>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![],
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (models, diagnostics) = rt.block_on(fetch_warp_appearances(&live, &appearance));

        assert!(
            models.contains_key(&vec![3]),
            "a decodable nested sun becomes a model"
        );
        assert!(!models.contains_key(&vec![1]));
        assert!(!models.contains_key(&vec![2]));
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics[&vec![1]].contains("has not published an appearance yet"),
            "missing journey: {}",
            diagnostics[&vec![1]]
        );
        assert!(
            diagnostics[&vec![2]].contains("not a decodable Black Hole Sun"),
            "foreign appearance: {}",
            diagnostics[&vec![2]]
        );
    }
}
