//! Piano input, events, logging, and strike visuals for the beam app.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use iced::keyboard;
use iced::widget::canvas;
use iced::widget::{container, stack, text};
use iced::{Background, Color, Element, Length};

use crate::builder::PianoLog;
use crate::piano::piano_audio::PianoAudioEngine;
use crate::piano::score_text;
use crate::piano::{
    piano_height, PianoAction, PianoEvent, PianoInputSource, PianoKeyAppearance, PianoKeyboard,
    PianoMessage, PianoNote, PianoPointerSource,
};

use super::{BeamApp, Message};

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


/// The tick grid [`BeamBuilder::piano_log`] prints on; the application start
/// is tick 0.
#[cfg(feature = "piano")]
const PIANO_LOG_TICKS_PER_SECOND: u64 = 1920;

/// Convert a duration to integer ticks on the piano log's tick grid.
#[cfg(feature = "piano")]
pub(crate) fn piano_log_ticks(duration: Duration) -> u64 {
    let ticks_per_second = u128::from(PIANO_LOG_TICKS_PER_SECOND);
    let total = u128::from(duration.as_secs()) * ticks_per_second
        + u128::from(duration.subsec_nanos()) * ticks_per_second / 1_000_000_000;
    total as u64
}

/// Quantize a normalized `0.0..=1.0` velocity onto the score's `0..=127`
/// grid.
#[cfg(feature = "piano")]
fn piano_log_velocity(velocity: f32) -> u8 {
    (velocity.clamp(0.0, 1.0) * f32::from(score_text::MAX_VELOCITY)).round() as u8
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PianoInputId {
    ComputerKeyboard(char),
    Pointer(PianoPointerSource),
    Score { cycle: u64, voice_id: u64 },
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ActivePianoNote {
    pub(crate) note: PianoNote,
    pub(crate) voice_id: u64,
    pub(crate) started_at: Instant,
    pub(crate) source: PianoInputSource,
}

#[cfg(feature = "piano")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PianoStrikeVisual {
    pub(crate) midi_note: u8,
    pub(crate) velocity: f32,
    pub(crate) pressure: Option<f32>,
    pub(crate) attacked_at: Instant,
    pub(crate) released: Option<(Instant, f32)>,
}

#[cfg(feature = "piano")]
impl PianoStrikeVisual {
    pub(crate) fn appearance(self, now: Instant) -> PianoKeyAppearance {
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

    pub(crate) fn needs_frame(self, now: Instant) -> bool {
        self.released.is_some()
            || now.saturating_duration_since(self.attacked_at) < Duration::from_millis(160)
    }

    pub(crate) fn finished(self, now: Instant) -> bool {
        self.released.is_some() && self.appearance(now).intensity <= 0.001
    }
}

impl BeamApp {
    #[cfg(feature = "piano")]
    pub(crate) fn piano_keyboard(&self) -> Element<'_, Message> {
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
        let label_octave = self.piano_label_octave();
        let piano_height = piano_height(label_octave.is_some());
        let keyboard: Element<'_, PianoMessage> =
            canvas::Canvas::new(PianoKeyboard::new(appearances, label_octave))
                .width(Length::Fill)
                .height(Length::Fixed(piano_height))
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
            .height(Length::Fixed(piano_height))
            .style(|_theme| container::Style {
                background: Some(Background::Color(Color::BLACK)),
                ..container::Style::default()
            })
            .into()
    }

    #[cfg(feature = "piano")]
    pub(crate) fn update_piano(&mut self, message: PianoMessage) {
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
    pub(crate) fn update_piano_keyboard(&mut self, event: keyboard::Event) {
        match event {
            keyboard::Event::KeyPressed {
                key,
                physical_key,
                repeat,
                ..
            } if !repeat => {
                // Track the held Shift keys so mapped notes can be shifted
                // by an octave; the Shift keys themselves never sound.
                match physical_key {
                    keyboard::key::Physical::Code(keyboard::key::Code::ShiftLeft) => {
                        self.piano_shift_left = true;
                        return;
                    }
                    keyboard::key::Physical::Code(keyboard::key::Code::ShiftRight) => {
                        self.piano_shift_right = true;
                        return;
                    }
                    _ => {}
                }
                // The Enter key sounds the top white key of the home row.
                if physical_key == keyboard::key::Physical::Code(keyboard::key::Code::Enter) {
                    let midi_note = crate::piano::computer_key_note(
                        '\r',
                        self.piano_octave,
                        self.piano_shift_offset(),
                    );
                    if let Some(midi_note) = midi_note {
                        self.attack_piano_note(
                            PianoInputId::ComputerKeyboard('\r'),
                            PianoInputSource::ComputerKeyboard { key: '\r' },
                            midi_note,
                            PianoEvent::BINARY_VELOCITY,
                            None,
                        );
                    }
                    return;
                }
                let Some(key) = piano_computer_key(&key, physical_key) else {
                    return;
                };
                if key == ' ' {
                    // The spacebar is not a note; in log mode it prints a
                    // blank line.
                    if self.config.piano_log.is_some() {
                        println!();
                    }
                    return;
                }
                // Number keys select the octave rather than sounding a note.
                if let Some(octave) = crate::piano::computer_octave_key(key) {
                    self.piano_octave = octave;
                    return;
                }
                let Some(midi_note) = crate::piano::computer_key_note(
                    key,
                    self.piano_octave,
                    self.piano_shift_offset(),
                )
                else {
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
                match physical_key {
                    keyboard::key::Physical::Code(keyboard::key::Code::ShiftLeft) => {
                        self.piano_shift_left = false;
                        return;
                    }
                    keyboard::key::Physical::Code(keyboard::key::Code::ShiftRight) => {
                        self.piano_shift_right = false;
                        return;
                    }
                    _ => {}
                }
                // The Enter key sounds the top white key of the home row.
                if physical_key == keyboard::key::Physical::Code(keyboard::key::Code::Enter) {
                    self.release_piano_note(PianoInputId::ComputerKeyboard('\r'), 0.0);
                    return;
                }
                let Some(key) = piano_computer_key(&key, physical_key) else {
                    return;
                };
                self.release_piano_note(PianoInputId::ComputerKeyboard(key), 0.0);
            }
            _ => {}
        }
    }

    /// The octave whose computer-keyboard bindings are labeled above the
    /// piano keys, or `None` when labels are disabled; held Shift keys
    /// transpose it like they transpose struck notes.
    #[cfg(feature = "piano")]
    pub(crate) fn piano_label_octave(&self) -> Option<i8> {
        self.config.piano_labels.then(|| self.piano_octave + self.piano_shift_offset())
    }

    /// The octave transposition currently held for piano input: left Shift
    /// strikes one octave down, right Shift one octave up, and with both
    /// held the mapped note sounds natural.
    #[cfg(feature = "piano")]
    fn piano_shift_offset(&self) -> i8 {
        match (self.piano_shift_left, self.piano_shift_right) {
            (true, false) => -1,
            (false, true) => 1,
            _ => 0,
        }
    }

    #[cfg(feature = "piano")]
    pub(crate) fn update_piano_score(&mut self, now: Instant) {
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
    pub(crate) fn update_piano_visuals(&mut self, now: Instant) {
        self.piano_visual_now = now;
        self.piano_strike_visuals
            .retain(|_, visual| !visual.finished(now));
    }

    #[cfg(feature = "piano")]
    pub(crate) fn attack_piano_note(
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
    pub(crate) fn release_piano_note(&mut self, input: PianoInputId, velocity: f32) {
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
        if let Some(line) = self.piano_log_line(&event) {
            println!("{line}");
        }
        if let Some(handler) = &self.config.piano_event_handler {
            handler(event);
        }
    }

    /// The line [`Self::emit_piano_event`] should print for `event`, if any.
    ///
    /// Attacks remember their start so the release can print the whole score
    /// pair; returns `None` while logging is off, while an attack awaits its
    /// release, when a release has no logged attack, or when the mode
    /// excludes the event's source.
    #[cfg(feature = "piano")]
    pub(crate) fn piano_log_line(&mut self, event: &PianoEvent) -> Option<String> {
        let mode = self.config.piano_log?;
        if matches!(mode, PianoLog::Input) && event.source == PianoInputSource::Score {
            return None;
        }
        match event.action {
            PianoAction::Attack { velocity, .. } => {
                self.piano_log_attacks.insert(
                    event.voice_id,
                    (event.timestamp, piano_log_velocity(velocity)),
                );
                None
            }
            PianoAction::Release { velocity, .. } => {
                let (start, attack) = self.piano_log_attacks.remove(&event.voice_id)?;
                let duration = event.timestamp.saturating_sub(start);
                // The log's timeline starts at the beginning of the original
                // score, so pad every logged time by the skipped intro.
                let start = start + self.piano_score_skip;
                Some(format!(
                    "{} {} {} {} {}",
                    piano_log_ticks(start),
                    piano_log_ticks(duration).max(1),
                    score_text::note_name(event.note.midi_note),
                    attack,
                    piano_log_velocity(velocity)
                ))
            }
        }
    }
}
