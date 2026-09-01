//! The public [`BeamBuilder`] API for static and live Black Hole Sun views.

#[cfg(feature = "piano")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use iced_sugiyama::motion::easing::Easing;
use jungle_sdk::{Animal, JourneyAstSource, JungleClient};
use uuid::Uuid;

use crate::app::run_beam;
use crate::flow::{BlackHoleSunAnimal, BlackHoleSunFlow};
use crate::labels::short_type_name;
use crate::live::LiveConfig;
use crate::model::BeamModel;
#[cfg(feature = "piano")]
use crate::piano::score_text::BhsScore;
#[cfg(feature = "piano")]
use crate::piano::PianoEvent;
use crate::subpanel::{build_static_subpanel_viewer, build_subpanel_viewer, SubpanelConfig};

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;

/// What [`BeamBuilder::piano_log`] prints to stdout while notes are played.
#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PianoLog {
    /// Only the notes the user inputs (e.g. through the computer keyboard or
    /// a pointer); notes played from a configured score are not logged.
    Input,
    /// Every note, including those played from a configured score.
    All,
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
    pub(crate) title: String,
    width: f32,
    height: f32,
    pub(crate) layout: BeamLayout,
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
    piano_score_skip_seconds: Option<u64>,
    #[cfg(feature = "piano")]
    piano_log: Option<PianoLog>,
    #[cfg(feature = "piano")]
    piano_labels: bool,
    #[cfg(feature = "piano")]
    piano_metronome_bpm: Option<u32>,
}

#[derive(Clone, Copy)]
pub(crate) enum BeamLayout {
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
            #[cfg(feature = "piano")]
            piano_score_skip_seconds: None,
            #[cfg(feature = "piano")]
            piano_log: None,
            #[cfg(feature = "piano")]
            piano_labels: false,
            #[cfg(feature = "piano")]
            piano_metronome_bpm: None,
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
    pub fn register_subpanel_animal<A>(self) -> Self
    where
        A: Animal + 'static,
        A::Flow: JourneyAstSource,
    {
        self.register_subpanel::<A>(false)
    }

    /// Register an animal whose node-click subpanel always shows its static
    /// flow structure.
    ///
    /// This is useful for self-contained visual examples: the main Beam can
    /// display a simulated or remote Sun appearance while each child panel
    /// remains inspectable without a live Jungle journey.
    pub fn register_static_subpanel_animal<A>(self) -> Self
    where
        A: Animal + 'static,
        A::Flow: JourneyAstSource,
    {
        self.register_subpanel::<A>(true)
    }

    fn register_subpanel<A>(mut self, prefer_static: bool) -> Self
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
                prefer_static,
                build_static_viewer: build_static_subpanel_viewer::<A>,
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
    ///
    /// Calling this more than once merges the scores so they play
    /// simultaneously: each added score's pairs are rescaled onto the first
    /// score's ticks-per-second grid, and the merged score loops over the
    /// longer of the two loop lengths (see [`BhsScore::merge_with`]).
    #[cfg(feature = "piano")]
    pub fn score(mut self, score: BhsScore) -> Self {
        let merged = match self.piano_score.take() {
            Some(previous) => previous.merge_with(score),
            None => score,
        };
        self.piano_score = Some(merged);
        self
    }

    /// Skip the first `seconds` seconds of any configured score before
    /// playback: pairs that start before the skip point are dropped and the
    /// remaining pairs shift back so playback begins at the skip point, as
    /// if the score had jumped straight there (see
    /// [`BhsScore::skip_seconds`]). Applies to a score set via
    /// [`Self::score_path`], [`Self::score_data`], or [`Self::score`]; a
    /// no-op when `seconds` is zero or no score is configured.
    ///
    /// Notes played from the score sound at app time with the skipped intro
    /// removed — a note at 10s in a score skipped by 5s sounds at 5s — but
    /// [`Self::piano_log`] reports their position in the original score, so
    /// that same note logs at 10s. Notes the user inputs are padded by the
    /// same amount, so every logged line sits on the original score's
    /// timeline. Skipping past every note of the score is an error, surfaced
    /// in the viewer's status line.
    #[cfg(feature = "piano")]
    pub fn score_skip(mut self, seconds: u64) -> Self {
        self.piano_score_skip_seconds = Some(seconds);
        self
    }

    /// Log played notes to stdout as `bhs-score-v1` note pairs.
    ///
    /// Each released note prints one score-pair line —
    /// `start_tick duration_ticks note velocity release_velocity` (see
    /// [`score_text`]) — on a 1920 ticks-per-second grid where the
    /// application start is tick 0, for example `86267 602 B3 55 55`. When a
    /// score skip is configured ([`Self::score_skip`]), every logged time is
    /// padded by the skipped amount, so tick 0 is the beginning of the
    /// original score rather than the application start. With
    /// [`PianoLog::Input`] only notes the user inputs (e.g. through the
    /// computer keyboard) are logged; [`PianoLog::All`] also logs notes
    /// played from a configured score. In either mode, pressing the spacebar
    /// prints a blank line.
    #[cfg(feature = "piano")]
    pub fn piano_log(mut self, log: PianoLog) -> Self {
        self.piano_log = Some(log);
        self
    }

    /// Show the computer-keyboard bindings above the piano keys.
    ///
    /// Each key in the currently active octave — the one selected by number
    /// keys `0`-`7`, transposed one octave down or up while left or right
    /// Shift is held — gets its binding character drawn in white directly
    /// above it, and keys outside the active octave are unlabeled. Enabling
    /// labels adds a label row above the keys, increasing the piano's
    /// height.
    #[cfg(feature = "piano")]
    pub fn piano_labels(mut self) -> Self {
        self.piano_labels = true;
        self
    }

    /// Play a metronome click on every beat at `bpm` beats per minute.
    ///
    /// The clicks are scheduled on the audio thread's sample clock, so they
    /// stay in time regardless of the window's frame rate, and sound
    /// alongside whatever is played on the piano. They are not
    /// [`PianoEvent`]s: neither [`Self::on_piano_event`] nor
    /// [`Self::piano_log`] reports them. A zero `bpm` leaves the metronome
    /// silent.
    #[cfg(feature = "piano")]
    pub fn metronome(mut self, bpm: u32) -> Self {
        self.piano_metronome_bpm = Some(bpm);
        self
    }

    pub(crate) fn into_config(self) -> BeamConfig {
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
            #[cfg(feature = "piano")]
            piano_score_skip_seconds: self.piano_score_skip_seconds,
            #[cfg(feature = "piano")]
            piano_log: self.piano_log,
            #[cfg(feature = "piano")]
            piano_labels: self.piano_labels,
            #[cfg(feature = "piano")]
            piano_metronome_bpm: self.piano_metronome_bpm,
        }
    }
}

#[derive(Clone)]
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
    pub(crate) piano_score_skip_seconds: Option<u64>,
    #[cfg(feature = "piano")]
    pub(crate) piano_log: Option<PianoLog>,
    #[cfg(feature = "piano")]
    pub(crate) piano_labels: bool,
    #[cfg(feature = "piano")]
    pub(crate) piano_metronome_bpm: Option<u32>,
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
