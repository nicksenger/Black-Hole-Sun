//! The iced application: window, messages, and the main view.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use black_hole_flux::sun::{SunAppearance, SunNodeState};
use iced::keyboard;
use iced::time::Instant;
use iced::widget::{button, column, container, mouse_area, opaque, row, rule, space, stack, text};
use iced::{Element, Font, Length, Subscription, Task};
use iced_sugiyama::AutoFit;
use jungle_vision::EjectedViewerMessage;

#[cfg(feature = "piano")]
use self::piano::{ActivePianoNote, PianoInputId, PianoStrikeVisual};
use crate::builder::BeamConfig;
use crate::client::SharedJungleClient;
use crate::graph::build_sun_graph;
use crate::labels::{animal_label_key, warp_boundary_label};
use crate::live::{appearance_task, LiveAppearanceSnapshot, LiveConfig};
use crate::model::{model_display_changed, BeamModel};
#[cfg(feature = "piano")]
use crate::piano::piano_audio::PianoAudioEngine;
#[cfg(feature = "piano")]
use crate::piano::piano_score::{load_score_document, PianoScorePlayback, SCORE_TICK_INTERVAL};
#[cfg(feature = "piano")]
use crate::piano::score_text::BhsScore;
#[cfg(feature = "piano")]
use crate::piano::PianoMessage;
use crate::style::{
    app_background_style, beam_theme, black_hole_text, cell_node_style, graph_node_button_style,
    subpanel_child_canvas_style, subpanel_close_button_style, subpanel_left_edge_style,
    subpanel_notice_style, subpanel_overlay_style, subpanel_style,
};
use crate::subpanel::{SubpanelConfig, SubpanelState};
use crate::visual::{
    displayed_grad_step, node_style_colors, warp_node_style_colors, CellVisualState,
    NodeStateVisual, NodeStyleColors,
};

#[cfg(feature = "piano")]
pub(crate) mod piano;

const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
pub(crate) const APPEARANCE_INTERVAL: Duration = Duration::from_millis(200);
const COLOR_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_TRANSITION_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone)]
pub(crate) enum Message {
    AppearanceTick,
    AppearanceLoaded(Result<Option<LiveAppearanceSnapshot>, String>),
    ColorTick(Instant),
    NodeSelected(u32),
    CloseSubpanel,
    Escape,
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

pub(crate) struct BeamApp {
    pub(crate) config: BeamConfig,
    pub(crate) model: BeamModel,
    pub(crate) live: Option<LiveConfig>,
    pub(crate) subpanel: Option<SubpanelState>,
    /// Warp cells whose nested sun is merged into the main graph, as paths of
    /// local cell ids from the top level (e.g. `[7]` or `[7, 3]`). Toggled by
    /// clicking the boundary cell; collapsing a path also collapses every
    /// expanded sub-path beneath it.
    pub(crate) expanded_warp_cells: HashSet<Vec<u32>>,
    /// Latest polled snapshot; the source for rebuilding the main graph when
    /// warp subgraphs expand or collapse.
    pub(crate) last_snapshot: Option<LiveAppearanceSnapshot>,
    pub(crate) visuals: HashMap<u32, CellVisualState>,
    pub(crate) appearance_loading: bool,
    pub(crate) appearance_error: Option<String>,
    pub(crate) subpanel_notice: Option<String>,
    pub(crate) color_now: Instant,
    #[cfg(feature = "piano")]
    pub(crate) piano_started_at: Instant,
    #[cfg(feature = "piano")]
    pub(crate) piano_event_sequence: u64,
    #[cfg(feature = "piano")]
    pub(crate) piano_voice_sequence: u64,
    #[cfg(feature = "piano")]
    pub(crate) active_piano_notes: HashMap<PianoInputId, ActivePianoNote>,
    #[cfg(feature = "piano")]
    pub(crate) piano_strike_visuals: HashMap<u64, PianoStrikeVisual>,
    #[cfg(feature = "piano")]
    pub(crate) piano_visual_now: Instant,
    #[cfg(feature = "piano")]
    pub(crate) piano_audio: Option<PianoAudioEngine>,
    #[cfg(feature = "piano")]
    pub(crate) piano_audio_error: Option<String>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score: Option<PianoScorePlayback>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score_error: Option<String>,
    #[cfg(feature = "piano")]
    pub(crate) piano_score_cycle: u64,
    /// The skipped intro of a configured score ([`BeamBuilder::score_skip`]),
    /// padded into every time reported by [`Self::piano_log_line`] so the
    /// log's timeline starts at the beginning of the original score.
    #[cfg(feature = "piano")]
    pub(crate) piano_score_skip: Duration,
    /// The scientific-pitch octave selected by number keys `0`-`7`; the home
    /// row sounds the white keys from the A in this octave (through E of the
    /// next on Enter), and the top row plays the black keys between them.
    #[cfg(feature = "piano")]
    pub(crate) piano_octave: i8,
    /// Whether the left Shift key is currently held for piano input; it
    /// strikes one octave below the mapped note.
    #[cfg(feature = "piano")]
    pub(crate) piano_shift_left: bool,
    /// Whether the right Shift key is currently held for piano input; it
    /// strikes one octave above the mapped note.
    #[cfg(feature = "piano")]
    pub(crate) piano_shift_right: bool,
    /// Attacks awaiting release for [`Self::piano_log_line`], keyed by voice
    /// id: the attack timestamp and quantized attack velocity.
    #[cfg(feature = "piano")]
    pub(crate) piano_log_attacks: HashMap<u64, (Duration, u8)>,
}

pub(crate) fn run_beam(
    config: BeamConfig,
    model: BeamModel,
    live: Option<LiveConfig>,
) -> iced::Result {
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

impl BeamApp {
    pub(crate) fn new(
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
                Ok(audio) => {
                    // A zero bpm leaves the metronome silent.
                    if let Some(bpm) = config.piano_metronome_bpm.filter(|bpm| *bpm > 0) {
                        audio.enable_metronome(bpm);
                    }
                    (Some(audio), None)
                }
                Err(error) => (None, Some(error)),
            }
        };
        #[cfg(feature = "piano")]
        let (piano_score, piano_score_error) = match Self::configured_score(&mut config) {
            Ok(Some(score)) => match PianoScorePlayback::from_score(score, Instant::now()) {
                Ok(score) => (Some(score), None),
                Err(error) => (None, Some(error)),
            },
            Ok(None) => (None, None),
            Err(error) => (None, Some(error)),
        };
        #[cfg(feature = "piano")]
        let piano_score_skip = config
            .piano_score_skip_seconds
            .map_or(Duration::ZERO, Duration::from_secs);

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
                #[cfg(feature = "piano")]
                piano_score_skip,
                #[cfg(feature = "piano")]
                piano_octave: 4,
                #[cfg(feature = "piano")]
                piano_shift_left: false,
                #[cfg(feature = "piano")]
                piano_shift_right: false,
                #[cfg(feature = "piano")]
                piano_log_attacks: HashMap::new(),
            },
            task,
        )
    }

    /// The score document configured on `config`, if any: a path or data
    /// source is parsed here and an owned score is taken as-is. A configured
    /// skip ([`BeamBuilder::score_skip`]) is applied to whatever source is
    /// set before the document is returned.
    #[cfg(feature = "piano")]
    fn configured_score(config: &mut BeamConfig) -> Result<Option<BhsScore>, String> {
        let document = if let Some(path) = config.piano_score_path.as_deref() {
            Some(load_score_document(path)?)
        } else if let Some(data) = config.piano_score_data.as_deref() {
            let text = std::str::from_utf8(data)
                .map_err(|error| format!("piano score is not valid UTF-8: {error}"))?;
            Some(BhsScore::parse(text)?)
        } else {
            config.piano_score.take()
        };
        if let Some(mut document) = document {
            if let Some(seconds) = config.piano_score_skip_seconds {
                document.skip_seconds(seconds)?;
            }
            Ok(Some(document))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn update(&mut self, message: Message) -> Task<Message> {
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
                if self.close_subpanel() {
                    return self.rebuild_model();
                }
            }
            Message::Escape => {
                // Escape closes the open subpanel and collapses every
                // expanded warp subgraph.
                let mut topology_changed = self.close_subpanel();
                if !self.expanded_warp_cells.is_empty() {
                    self.expanded_warp_cells.clear();
                    topology_changed = true;
                }
                if topology_changed {
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
        // Escape closes any open subpanel and collapses every expanded warp
        // subgraph, with or without the piano feature.
        subscriptions.push(
            keyboard::listen()
                .filter_map(|event| {
                    matches!(
                        event,
                        keyboard::Event::KeyPressed {
                            key: keyboard::Key::Named(keyboard::key::Named::Escape),
                            repeat: false,
                            ..
                        }
                    )
                    .then_some(())
                })
                .map(|_| Message::Escape),
        );
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

    pub(crate) fn cell_styles(&self) -> HashMap<u32, NodeStyleColors> {
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
    pub(crate) fn open_subpanel_for_node(&mut self, node_id: u32) -> Task<Message> {
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

    /// Closes the open subpanel, collapsing any warp subgraph it expanded.
    /// Returns whether the main graph topology changed and needs rebuilding.
    fn close_subpanel(&mut self) -> bool {
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
            if let Some(path) = self.model.warp_paths.get(&node_id).cloned() {
                self.collapse_warp_path(&path);
            }
        }
        self.subpanel = None;
        closing_warp_node
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
        (1..=path.len()).all(|len| self.expanded_warp_cells.contains(&path[..len].to_vec()))
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

    pub(crate) fn resolve_subpanel_config(&self, animal_label: &str) -> Option<SubpanelConfig> {
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

    pub(crate) fn subpanel_phase(&self, node_id: u32) -> Option<String> {
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
