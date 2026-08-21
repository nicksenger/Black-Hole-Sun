#!/usr/bin/env python3
"""Make internal items pub(crate) so the crate-root test modules can use them."""
import sys

def patch(path, pairs):
    text = open(path).read()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            print(f"FAIL {path}: found {count}x: {old[:70]!r}")
            sys.exit(1)
        text = text.replace(old, new)
    open(path, "w").write(text)
    print(f"patched {path} ({len(pairs)} edits)")

# --- builder.rs ---
patch("src/builder.rs", [
    ("""pub struct BeamBuilder {
    title: String,
    width: f32,
    height: f32,
    layout: BeamLayout,""",
     """pub struct BeamBuilder {
    pub(crate) title: String,
    width: f32,
    height: f32,
    pub(crate) layout: BeamLayout,"""),
    ("enum BeamLayout {", "pub(crate) enum BeamLayout {"),
    ("    fn into_config(self) -> BeamConfig {",
     "    pub(crate) fn into_config(self) -> BeamConfig {"),
    ("""#[derive(Clone)]
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
    #[cfg(feature = "piano")]
    piano_log: Option<PianoLog>,
    #[cfg(feature = "piano")]
    piano_labels: bool,
}""",
     """#[derive(Clone)]
pub(crate) struct BeamConfig {
    pub(crate) title: String,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) layout: BeamLayout,
    pub(crate) subpanel_animals: Vec<SubpanelConfig>,
    pub(crate) animation_duration: Option<Duration>,
    pub(crate) animation_easing: Option<&'static Easing>,
    #[cfg(feature = "piano")]
    pub(crate) piano_event_handler: Option<Arc<dyn Fn(PianoEvent) + Send + Sync>>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score_path: Option<PathBuf>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score_data: Option<Vec<u8>>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score: Option<BhsScore>,
    #[cfg(feature = "piano")]
    pub(crate) piano_log: Option<PianoLog>,
    #[cfg(feature = "piano")]
    pub(crate) piano_labels: bool,
}"""),
])

# --- model.rs ---
patch("src/model.rs", [
    ("""#[derive(Clone)]
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
}""",
     """#[derive(Clone)]
pub(crate) struct CellDefinition {
    pub(crate) id: u32,
    pub(crate) journey_id: Uuid,
    /// Journey of the nested warp animal for warp cells; nil otherwise.
    pub(crate) warp_journey_id: Uuid,
    pub(crate) ports: Vec<u32>,
    pub(crate) outgoing_ports: Vec<u32>,
    pub(crate) animal_name: String,
    pub(crate) state: SunNodeState,
    pub(crate) state_sequence: u64,
    pub(crate) grad_step: usize,
    pub(crate) grad_steps: usize,
    pub(crate) frozen: Option<bool>,
}"""),
    ("    fn new<A>(id: u32, ports: Vec<u32>, outgoing_ports: Vec<u32>) -> Self",
     "    pub(crate) fn new<A>(id: u32, ports: Vec<u32>, outgoing_ports: Vec<u32>) -> Self"),
    ("""#[derive(Clone)]
struct BeamModel {
    cells: Vec<CellDefinition>,
    graph: Graph,
    grad_steps: usize,
    errors: Vec<String>,""",
     """#[derive(Clone)]
pub(crate) struct BeamModel {
    pub(crate) cells: Vec<CellDefinition>,
    pub(crate) graph: Graph,
    pub(crate) grad_steps: usize,
    pub(crate) errors: Vec<String>,"""),
    ("    warp_paths: HashMap<u32, Vec<u32>>,",
     "    pub(crate) warp_paths: HashMap<u32, Vec<u32>>,"),
    ("    fn empty() -> Self {", "    pub(crate) fn empty() -> Self {"),
    ("    fn build<F>() -> Self", "    pub(crate) fn build<F>() -> Self"),
])

# --- app/mod.rs ---
text = open("src/app/mod.rs").read()
old_struct = """struct BeamApp {
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
    subpanel: Option<SubpanelState>,"""
new_struct = """pub(crate) struct BeamApp {
    pub(crate) config: BeamConfig,
    pub(crate) model: BeamModel,
    pub(crate) live: Option<LiveConfig>,
    pub(crate) subpanel: Option<SubpanelState>,"""
assert text.count(old_struct) == 1
text = text.replace(old_struct, new_struct)

# Remaining plain fields (non-cfg-gated) in the struct.
for field in [
    "    expanded_warp_cells: HashSet<Vec<u32>>,",
    "    last_snapshot: Option<LiveAppearanceSnapshot>,",
    "    visuals: HashMap<u32, CellVisualState>,",
    "    appearance_loading: bool,",
    "    appearance_error: Option<String>,",
    "    subpanel_notice: Option<String>,",
    "    color_now: Instant,",
]:
    assert text.count(field) == 1, field
    text = text.replace(field, field.replace(": ", ": pub(crate) ", 1).replace("pub(crate):", "pub(crate) :"))
# fix the awkward replacement above
text = text.replace(": pub(crate) ", ": pub(crate) ")

# cfg-gated piano fields.
for field in [
    "    piano_started_at: Instant,",
    "    piano_event_sequence: u64,",
    "    piano_voice_sequence: u64,",
    "    active_piano_notes: HashMap<PianoInputId, ActivePianoNote>,",
    "    piano_strike_visuals: HashMap<u64, PianoStrikeVisual>,",
    "    piano_visual_now: Instant,",
    "    piano_audio: Option<PianoAudioEngine>,",
    "    piano_audio_error: Option<String>,",
    "    piano_score: Option<PianoScorePlayback>,",
    "    piano_score_error: Option<String>,",
    "    piano_score_cycle: u64,",
    "    piano_octave: i8,",
    "    piano_shift_left: bool,",
    "    piano_shift_right: bool,",
    "    piano_log_attacks: HashMap<u64, (Duration, u8)>,",
]:
    assert text.count(field) == 1, field
    name = field.strip().rstrip(",")
    split = name.split(": ", 1)
    text = text.replace(
        f"{split[0]}: {split[1]}",
        f"{split[0]}: pub(crate) {split[1]}",
        1,
    )

for old, new in [
    ("enum Message {", "pub(crate) enum Message {"),
    ("const APPEARANCE_INTERVAL: Duration = Duration::from_millis(200);",
     "pub(crate) const APPEARANCE_INTERVAL: Duration = Duration::from_millis(200);"),
    ("    fn new(\n        #[allow(unused_mut)] mut config: BeamConfig,",
     "    pub(crate) fn new(\n        #[allow(unused_mut)] mut config: BeamConfig,"),
    ("    fn update(&mut self, message: Message) -> Task<Message> {",
     "    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {"),
    ("    fn cell_styles(&self) -> HashMap<u32, NodeStyleColors> {",
     "    pub(crate) fn cell_styles(&self) -> HashMap<u32, NodeStyleColors> {"),
    ("    fn open_subpanel_for_node(&mut self, node_id: u32) -> Task<Message> {",
     "    pub(crate) fn open_subpanel_for_node(&mut self, node_id: u32) -> Task<Message> {"),
    ("    fn resolve_subpanel_config(&self, animal_label: &str) -> Option<SubpanelConfig> {",
     "    pub(crate) fn resolve_subpanel_config(&self, animal_label: &str) -> Option<SubpanelConfig> {"),
    ("    fn subpanel_phase(&self, node_id: u32) -> Option<String> {",
     "    pub(crate) fn subpanel_phase(&self, node_id: u32) -> Option<String> {"),
]:
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)
open("src/app/mod.rs", "w").write(text)
print("patched src/app/mod.rs")

# --- app/piano.rs ---
patch("src/app/piano.rs", [
    ("enum PianoInputId {", "pub(crate) enum PianoInputId {"),
    ("""#[derive(Debug, Clone, Copy)]
struct PianoStrikeVisual {
    midi_note: u8,
    velocity: f32,
    pressure: Option<f32>,
    attacked_at: Instant,
    released: Option<(Instant, f32)>,
}""",
     """#[derive(Debug, Clone, Copy)]
pub(crate) struct PianoStrikeVisual {
    pub(crate) midi_note: u8,
    pub(crate) velocity: f32,
    pub(crate) pressure: Option<f32>,
    pub(crate) attacked_at: Instant,
    pub(crate) released: Option<(Instant, f32)>,
}"""),
    ("fn piano_log_ticks(duration: Duration) -> u64 {",
     "pub(crate) fn piano_log_ticks(duration: Duration) -> u64 {"),
    ("    fn attack_piano_note(\n        &mut self,",
     "    pub(crate) fn attack_piano_note(\n        &mut self,"),
    ("    fn release_piano_note(&mut self, input: PianoInputId, velocity: f32) {",
     "    pub(crate) fn release_piano_note(&mut self, input: PianoInputId, velocity: f32) {"),
    ("    fn piano_log_line(&mut self, event: &PianoEvent) -> Option<String> {",
     "    pub(crate) fn piano_log_line(&mut self, event: &PianoEvent) -> Option<String> {"),
    ("    fn piano_label_octave(&self) -> Option<i8> {",
     "    pub(crate) fn piano_label_octave(&self) -> Option<i8> {"),
    ("    fn update_piano_keyboard(&mut self, event: keyboard::Event) {",
     "    pub(crate) fn update_piano_keyboard(&mut self, event: keyboard::Event) {"),
    ("    fn update_piano_score(&mut self, now: Instant) {",
     "    pub(crate) fn update_piano_score(&mut self, now: Instant) {"),
])

# --- visual.rs ---
patch("src/visual.rs", [
    ("const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);",
     "pub(crate) const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);"),
    ("const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);",
     "pub(crate) const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);"),
    ("const MAX_PENDING_PHASES: usize = 4;",
     "pub(crate) const MAX_PENDING_PHASES: usize = 4;"),
    ("trait NodeStateVisual {", "pub(crate) trait NodeStateVisual {"),
    ("""#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodeProgress {
    state: SunNodeState,
    grad_step: usize,
}""",
     """#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeProgress {
    pub(crate) state: SunNodeState,
    pub(crate) grad_step: usize,
}"""),
    ("""#[derive(Debug, Clone, Copy, PartialEq)]
struct NodeStyleColors {
    body: Color,
    border: Color,
    text: Color,
}""",
     """#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NodeStyleColors {
    pub(crate) body: Color,
    pub(crate) border: Color,
    pub(crate) text: Color,
}"""),
    ("fn node_style_colors(\n    state: SunNodeState,",
     "pub(crate) fn node_style_colors(\n    state: SunNodeState,"),
    ("fn warp_node_style_colors(\n    state: SunNodeState,",
     "pub(crate) fn warp_node_style_colors(\n    state: SunNodeState,"),
    ("fn lerp_color(a: Color, b: Color, amount: f32) -> Color {",
     "pub(crate) fn lerp_color(a: Color, b: Color, amount: f32) -> Color {"),
    ("""#[derive(Debug, Clone)]
struct CellVisualState {
    previous: NodeProgress,
    current: NodeProgress,
    transition_started_at: Option<Instant>,
    pending: VecDeque<NodeProgress>,
    observed_sequence: u64,
    latest_frozen: Option<bool>,
    optimization_frozen: Option<bool>,
}""",
     """#[derive(Debug, Clone)]
pub(crate) struct CellVisualState {
    previous: NodeProgress,
    pub(crate) current: NodeProgress,
    transition_started_at: Option<Instant>,
    pub(crate) pending: VecDeque<NodeProgress>,
    observed_sequence: u64,
    latest_frozen: Option<bool>,
    pub(crate) optimization_frozen: Option<bool>,
}"""),
    ("        now: Instant,\n    ) -> bool {\n        let grad_steps = grad_steps.max(1);",
     "        now: Instant,\n    ) -> bool {\n        let grad_steps = grad_steps.max(1);"),  # no-op guard
    ("    fn observe(\n        &mut self,", "    pub(crate) fn observe(\n        &mut self,"),
    ("    fn advance(&mut self, now: Instant) -> bool {",
     "    pub(crate) fn advance(&mut self, now: Instant) -> bool {"),
    ("    fn style(\n        &self,\n        grad_steps: usize,",
     "    pub(crate) fn style(\n        &self,\n        grad_steps: usize,"),
    ("    fn needs_color_frame(&self, now: Instant) -> bool {",
     "    pub(crate) fn needs_color_frame(&self, now: Instant) -> bool {"),
    ("    fn needs_transition_poll(&self, now: Instant) -> bool {",
     "    pub(crate) fn needs_transition_poll(&self, now: Instant) -> bool {"),
])

# --- labels.rs ---
patch("src/labels.rs", [
    ("fn short_type_name<T: ?Sized>() -> String {",
     "pub(crate) fn short_type_name<T: ?Sized>() -> String {"),
    ("fn animal_label_key(label: &str) -> String {",
     "pub(crate) fn animal_label_key(label: &str) -> String {"),
    ("fn warp_boundary_label(label: &str) -> Option<String> {",
     "pub(crate) fn warp_boundary_label(label: &str) -> Option<String> {"),
])

# --- live.rs ---
patch("src/live.rs", [
    ("""#[derive(Clone)]
struct LiveConfig {
    client: Arc<dyn JungleClient>,
    journey_id: Uuid,
}""",
     """#[derive(Clone)]
pub(crate) struct LiveConfig {
    pub(crate) client: Arc<dyn JungleClient>,
    pub(crate) journey_id: Uuid,
}"""),
    ("""#[derive(Debug, Clone)]
struct LiveAppearanceSnapshot {
    appearance: SunAppearance,
    child_rays: HashMap<Uuid, Ray>,""",
     """#[derive(Debug, Clone)]
pub(crate) struct LiveAppearanceSnapshot {
    pub(crate) appearance: SunAppearance,
    pub(crate) child_rays: HashMap<Uuid, Ray>,"""),
    ("    warp_appearances: HashMap<Vec<u32>, SunAppearance>,",
     "    pub(crate) warp_appearances: HashMap<Vec<u32>, SunAppearance>,"),
    ("    warp_diagnostics: HashMap<Vec<u32>, String>,",
     "    pub(crate) warp_diagnostics: HashMap<Vec<u32>, String>,"),
    ("async fn fetch_warp_appearances(\n    live: &LiveConfig,",
     "pub(crate) async fn fetch_warp_appearances(\n    live: &LiveConfig,"),
])
