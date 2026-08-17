//! Expressive 88-key piano input for the beam UI.

use std::collections::HashMap;
use std::time::Duration;

use iced::mouse;
use iced::touch;
use iced::widget::canvas::{self, Path};
use iced::{Color, Point, Rectangle, Size, Theme};
use serde::{Deserialize, Serialize};

pub(crate) const PIANO_HEIGHT: f32 = 185.0;
const FIRST_MIDI_NOTE: u8 = 21; // A0
const LAST_MIDI_NOTE: u8 = 108; // C8
const WHITE_KEY_COUNT: f32 = 52.0;
const BLACK_KEY_HEIGHT: f32 = PIANO_HEIGHT * 0.63;

/// A piano note in equal temperament with A4 tuned to 440 Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PianoNote {
    /// The MIDI note number (`21..=108` for the on-screen keyboard).
    pub midi_note: u8,
    /// The note's fundamental frequency in hertz.
    pub frequency_hz: f32,
}

impl PianoNote {
    pub(crate) fn from_midi(midi_note: u8) -> Self {
        Self {
            midi_note,
            frequency_hz: 440.0 * 2.0_f32.powf((f32::from(midi_note) - 69.0) / 12.0),
        }
    }

    /// Scientific-pitch octave number (`4` for middle C).
    pub fn octave(self) -> i8 {
        (self.midi_note / 12) as i8 - 1
    }

    /// Pitch-class name, such as `C`, `F#`, or `Bb`.
    pub fn name(self) -> &'static str {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
        ];
        NAMES[usize::from(self.midi_note % 12)]
    }
}

/// The device surface that originated a piano event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PianoInputSource {
    /// A mapped key on a conventional computer keyboard.
    ComputerKeyboard { key: char },
    /// The primary mouse button on an on-screen key.
    Mouse,
    /// A touch contact on an on-screen key.
    Touch { finger: u64 },
    /// An event loaded from a looping JSON score.
    Score,
}

/// The expressive phase of a performed note.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PianoAction {
    /// A key began sounding. Velocity and pressure are normalized to `0.0..=1.0`.
    Attack {
        velocity: f32,
        pressure: Option<f32>,
    },
    /// A key stopped being held. Velocity is normalized to `0.0..=1.0`.
    Release {
        velocity: f32,
        #[serde(with = "duration_seconds")]
        held_for: Duration,
    },
}

/// A lossless-in-time description of an on-screen piano performance event.
///
/// `voice_id` pairs an attack with its release even when the same note is
/// played concurrently by multiple inputs. `sequence` provides stable event
/// ordering when timestamps are equal. JSON serialization represents
/// `timestamp` and release `held_for` durations as numeric seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PianoEvent {
    pub sequence: u64,
    #[serde(with = "duration_seconds")]
    pub timestamp: Duration,
    pub voice_id: u64,
    pub note: PianoNote,
    pub action: PianoAction,
    pub source: PianoInputSource,
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let seconds = f64::deserialize(deserializer)?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(serde::de::Error::custom(
                "duration must be a finite, non-negative number of seconds",
            ));
        }
        Ok(Duration::from_secs_f64(seconds))
    }
}

impl PianoEvent {
    /// Default velocity for devices, such as computer keyboards, that only
    /// report binary pressed/released state.
    pub const BINARY_VELOCITY: f32 = 1.0;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PianoMessage {
    Press {
        midi_note: u8,
        velocity: f32,
        source: PianoPointerSource,
    },
    Release {
        source: PianoPointerSource,
        velocity: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PianoPointerSource {
    Mouse,
    Touch(u64),
}

impl PianoPointerSource {
    pub(crate) fn public(self) -> PianoInputSource {
        match self {
            Self::Mouse => PianoInputSource::Mouse,
            Self::Touch(finger) => PianoInputSource::Touch { finger },
        }
    }
}

#[derive(Default)]
pub(crate) struct PianoCanvasState {
    mouse_note: Option<u8>,
    touch_notes: HashMap<touch::Finger, u8>,
}

pub(crate) struct PianoKeyboard {
    appearances: HashMap<u8, PianoKeyAppearance>,
}

impl PianoKeyboard {
    pub(crate) fn new(appearances: HashMap<u8, PianoKeyAppearance>) -> Self {
        Self { appearances }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct PianoKeyAppearance {
    pub intensity: f32,
}

impl canvas::Program<PianoMessage> for PianoKeyboard {
    type State = PianoCanvasState;

    fn update(
        &self,
        state: &mut Self::State,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<PianoMessage>> {
        match event {
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let position = cursor.position_in(bounds)?;
                let midi_note = note_at(position, bounds.size())?;
                if state.mouse_note.replace(midi_note).is_some() {
                    return None;
                }
                Some(
                    canvas::Action::publish(PianoMessage::Press {
                        midi_note,
                        velocity: strike_velocity(position, bounds.size(), midi_note),
                        source: PianoPointerSource::Mouse,
                    })
                    .and_capture(),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.mouse_note.take().map(|_| {
                    canvas::Action::publish(PianoMessage::Release {
                        source: PianoPointerSource::Mouse,
                        velocity: 0.5,
                    })
                    .and_capture()
                })
            }
            canvas::Event::Touch(touch::Event::FingerPressed { id, position }) => {
                let position = *position - iced::Vector::new(bounds.x, bounds.y);
                let midi_note = note_at(position, bounds.size())?;
                state.touch_notes.insert(*id, midi_note);
                Some(
                    canvas::Action::publish(PianoMessage::Press {
                        midi_note,
                        velocity: strike_velocity(position, bounds.size(), midi_note),
                        source: PianoPointerSource::Touch(id.0),
                    })
                    .and_capture(),
                )
            }
            canvas::Event::Touch(touch::Event::FingerLifted { id, .. })
            | canvas::Event::Touch(touch::Event::FingerLost { id, .. }) => {
                state.touch_notes.remove(id).map(|_| {
                    canvas::Action::publish(PianoMessage::Release {
                        source: PianoPointerSource::Touch(id.0),
                        velocity: 0.5,
                    })
                    .and_capture()
                })
            }
            _ => None,
        }
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let key_height = bounds.height;
        let white_width = bounds.width / WHITE_KEY_COUNT;

        frame.fill_rectangle(Point::ORIGIN, bounds.size(), Color::BLACK);

        let mut white_index = 0_u8;
        for midi_note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
            if is_black(midi_note) {
                continue;
            }
            let x = f32::from(white_index) * white_width;
            draw_white_key(
                &mut frame,
                Rectangle::new(Point::new(x, 0.0), Size::new(white_width, key_height)),
                self.appearances
                    .get(&midi_note)
                    .copied()
                    .unwrap_or_default(),
                midi_note < LAST_MIDI_NOTE && is_black(midi_note + 1),
            );
            white_index += 1;
        }

        let black_width = white_width * 0.68;
        let black_height = key_height * 0.63;
        let mut whites_before = 0_u8;
        for midi_note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
            if is_black(midi_note) {
                let x = f32::from(whites_before) * white_width - black_width / 2.0;
                draw_black_key(
                    &mut frame,
                    Rectangle::new(Point::new(x, 0.0), Size::new(black_width, black_height)),
                    self.appearances
                        .get(&midi_note)
                        .copied()
                        .unwrap_or_default(),
                );
            } else {
                whites_before += 1;
            }
        }

        vec![frame.into_geometry()]
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if cursor.is_over(bounds) {
            mouse::Interaction::Pointer
        } else {
            mouse::Interaction::None
        }
    }
}

fn draw_white_key(
    frame: &mut canvas::Frame,
    bounds: Rectangle,
    appearance: PianoKeyAppearance,
    has_black_key_after: bool,
) {
    let intensity = appearance.intensity.clamp(0.0, 1.0);
    let face = Path::rectangle(bounds.position(), bounds.size());
    let gradient = || {
        canvas::gradient::Linear::new(
            Point::new(bounds.x, bounds.y),
            Point::new(bounds.x, bounds.y + bounds.height),
        )
    };
    let fill = gradient()
        .add_stop(
            0.0,
            mix_color(Color::WHITE, Color::from_rgb8(255, 220, 142), intensity),
        )
        .add_stop(
            0.72,
            mix_color(
                Color::from_rgb8(248, 246, 239),
                Color::from_rgb8(236, 165, 82),
                intensity,
            ),
        )
        .add_stop(
            1.0,
            mix_color(
                Color::from_rgb8(205, 202, 194),
                Color::from_rgb8(214, 124, 45),
                intensity,
            ),
        );
    frame.fill(&face, fill);
    frame.stroke(
        &face,
        canvas::Stroke::default()
            .with_color(Color::from_rgb8(89, 89, 92))
            .with_width(0.8),
    );

    if has_black_key_after {
        // NanoMoog's white keys carry a dark inner shoulder below black keys.
        let shoulder = Path::new(|path| {
            path.move_to(Point::new(bounds.x + bounds.width * 0.78, 1.0));
            path.line_to(Point::new(
                bounds.x + bounds.width * 0.78,
                bounds.height * 0.63,
            ));
            path.line_to(Point::new(
                bounds.x + bounds.width * 0.96,
                bounds.height * 0.55,
            ));
            path.line_to(Point::new(bounds.x + bounds.width * 0.94, 1.0));
            path.close();
        });
        frame.fill(
            &shoulder,
            Color::from_rgba8(54, 54, 54, 0.34 - intensity * 0.14),
        );
    }
}

fn draw_black_key(frame: &mut canvas::Frame, bounds: Rectangle, appearance: PianoKeyAppearance) {
    let intensity = appearance.intensity.clamp(0.0, 1.0);
    let face = Path::new(|path| {
        path.move_to(bounds.position());
        path.line_to(Point::new(bounds.x + bounds.width, bounds.y));
        path.line_to(Point::new(bounds.x + bounds.width * 0.94, bounds.height));
        path.line_to(Point::new(bounds.x + bounds.width * 0.06, bounds.height));
        path.close();
    });
    let gradient = || {
        canvas::gradient::Linear::new(
            Point::new(bounds.x, bounds.y),
            Point::new(bounds.x, bounds.y + bounds.height),
        )
    };
    let fill = gradient()
        .add_stop(
            0.0,
            mix_color(
                Color::from_rgb8(10, 10, 10),
                Color::from_rgb8(105, 50, 16),
                intensity,
            ),
        )
        .add_stop(
            0.72,
            mix_color(
                Color::from_rgb8(34, 34, 34),
                Color::from_rgb8(169, 78, 21),
                intensity,
            ),
        )
        .add_stop(
            1.0,
            mix_color(Color::BLACK, Color::from_rgb8(226, 108, 30), intensity),
        );
    frame.fill(&face, fill);
    frame.stroke(
        &face,
        canvas::Stroke::default()
            .with_color(Color::from_rgb8(73, 73, 73))
            .with_width(0.8),
    );

    let left_highlight = Path::line(
        Point::new(bounds.x + bounds.width * 0.22, 2.0),
        Point::new(bounds.x + bounds.width * 0.22, bounds.height * 0.86),
    );
    frame.stroke(
        &left_highlight,
        canvas::Stroke::default()
            .with_color(Color::from_rgba8(160, 160, 160, 0.7))
            .with_width(0.8),
    );

    // A beveled lip recreates the distinctive shaded black-key front.
    let lip = Path::new(|path| {
        path.move_to(Point::new(
            bounds.x + bounds.width * 0.22,
            bounds.height * 0.86,
        ));
        path.line_to(Point::new(
            bounds.x + bounds.width * 0.78,
            bounds.height * 0.86,
        ));
        path.line_to(Point::new(bounds.x + bounds.width * 0.94, bounds.height));
        path.line_to(Point::new(bounds.x + bounds.width * 0.06, bounds.height));
        path.close();
    });
    frame.fill(
        &lip,
        mix_color(
            Color::from_rgb8(91, 91, 91),
            Color::from_rgb8(120, 56, 18),
            intensity,
        ),
    );
}

fn mix_color(from: Color, to: Color, amount: f32) -> Color {
    Color::from_rgba(
        from.r + (to.r - from.r) * amount,
        from.g + (to.g - from.g) * amount,
        from.b + (to.b - from.b) * amount,
        from.a + (to.a - from.a) * amount,
    )
}

fn note_at(position: Point, size: Size) -> Option<u8> {
    if position.x < 0.0 || position.y < 0.0 || position.x > size.width || position.y > size.height {
        return None;
    }
    let white_width = size.width / WHITE_KEY_COUNT;
    let black_width = white_width * 0.68;
    if position.y <= size.height * 0.63 {
        let mut whites_before = 0_u8;
        for midi_note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
            if is_black(midi_note) {
                let left = f32::from(whites_before) * white_width - black_width / 2.0;
                if (left..=left + black_width).contains(&position.x) {
                    return Some(midi_note);
                }
            } else {
                whites_before += 1;
            }
        }
    }

    let white_index = (position.x / white_width).floor() as u8;
    let mut current_white = 0_u8;
    (FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).find(|midi_note| {
        if is_black(*midi_note) {
            false
        } else {
            let found = current_white == white_index;
            current_white += 1;
            found
        }
    })
}

fn strike_velocity(position: Point, size: Size, midi_note: u8) -> f32 {
    let height = if is_black(midi_note) {
        size.height * (BLACK_KEY_HEIGHT / PIANO_HEIGHT)
    } else {
        size.height
    };
    (0.35 + 0.65 * (position.y / height)).clamp(0.0, 1.0)
}

const fn is_black(midi_note: u8) -> bool {
    matches!(midi_note % 12, 1 | 3 | 6 | 8 | 10)
}

pub(crate) fn computer_key_note(key: char) -> Option<u8> {
    Some(match key.to_ascii_lowercase() {
        'z' => 60,
        's' => 61,
        'x' => 62,
        'd' => 63,
        'c' => 64,
        'v' => 65,
        'g' => 66,
        'b' => 67,
        'h' => 68,
        'n' => 69,
        'j' => 70,
        'm' => 71,
        ',' | 'q' => 72,
        '2' => 73,
        '.' | 'w' => 74,
        '3' => 75,
        '/' | 'e' => 76,
        'r' => 77,
        '5' => 78,
        't' => 79,
        '6' => 80,
        'y' => 81,
        '7' => 82,
        'u' => 83,
        'i' => 84,
        '9' => 85,
        'o' => 86,
        '0' => 87,
        'p' => 88,
        '[' => 89,
        ']' => 90,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_contains_exactly_standard_88_key_range() {
        assert_eq!(FIRST_MIDI_NOTE, 21);
        assert_eq!(LAST_MIDI_NOTE, 108);
        assert_eq!(
            (FIRST_MIDI_NOTE..=LAST_MIDI_NOTE)
                .filter(|note| !is_black(*note))
                .count(),
            WHITE_KEY_COUNT as usize
        );
    }

    #[test]
    fn equal_temperament_notes_are_named_and_tuned() {
        let middle_c = PianoNote::from_midi(60);
        assert_eq!(middle_c.name(), "C");
        assert_eq!(middle_c.octave(), 4);
        assert!((middle_c.frequency_hz - 261.625_55).abs() < 0.001);
        assert_eq!(PianoNote::from_midi(69).frequency_hz, 440.0);
    }

    #[test]
    fn hit_testing_prefers_raised_black_keys() {
        let size = Size::new(1040.0, PIANO_HEIGHT);
        // A#0 is centered on the boundary between the first two white keys.
        assert_eq!(note_at(Point::new(20.0, 20.0), size), Some(22));
        assert_eq!(note_at(Point::new(20.0, 180.0), size), Some(23));
    }

    #[test]
    fn computer_keyboard_mapping_is_chromatic() {
        assert_eq!(computer_key_note('z'), Some(60));
        assert_eq!(computer_key_note('s'), Some(61));
        assert_eq!(computer_key_note('x'), Some(62));
        assert_eq!(computer_key_note(']'), Some(90));
        assert_eq!(computer_key_note('!'), None);
    }

    #[test]
    fn piano_events_round_trip_as_human_readable_json_seconds() {
        let event = PianoEvent {
            sequence: 12,
            timestamp: Duration::from_millis(1_250),
            voice_id: 7,
            note: PianoNote::from_midi(60),
            action: PianoAction::Release {
                velocity: 0.4,
                held_for: Duration::from_millis(750),
            },
            source: PianoInputSource::Score,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"timestamp\":1.25"));
        assert!(json.contains("\"held_for\":0.75"));
        assert_eq!(serde_json::from_str::<PianoEvent>(&json).unwrap(), event);
    }
}
