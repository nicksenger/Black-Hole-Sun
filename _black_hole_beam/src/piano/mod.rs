use std::collections::HashMap;
use std::time::Duration;

use iced::alignment::Vertical;
use iced::mouse;
use iced::touch;
use iced::widget::canvas::{self, Path};
use iced::widget::text::Alignment;
use iced::{Color, Pixels, Point, Rectangle, Size, Theme};

pub mod piano_audio;
pub mod piano_score;
pub mod score_text;

pub(crate) const PIANO_HEIGHT: f32 = 185.0;
/// The height of the keybind label row drawn above the keys when labels are
/// enabled.
pub(crate) const PIANO_LABEL_ROW_HEIGHT: f32 = 20.0;
/// The font size of keybind labels in the label row.
const PIANO_LABEL_FONT_SIZE: f32 = 13.0;
const FIRST_MIDI_NOTE: u8 = 21; // A0
const LAST_MIDI_NOTE: u8 = 108; // C8
const WHITE_KEY_COUNT: f32 = 52.0;
const BLACK_KEY_HEIGHT: f32 = PIANO_HEIGHT * 0.63;

/// The piano canvas height for a given label state: the key row plus the
/// keybind label row when labels are enabled.
pub(crate) fn piano_height(labels_enabled: bool) -> f32 {
    PIANO_HEIGHT + if labels_enabled { PIANO_LABEL_ROW_HEIGHT } else { 0.0 }
}

/// A piano note in equal temperament with A4 tuned to 440 Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PianoNote {
    /// The MIDI note number (`21..=108` for the on-screen keyboard).
    pub midi_note: u8,
    /// The note's fundamental frequency in hertz.
    pub frequency_hz: f32,
}

impl PianoNote {
    /// Build a note from its MIDI number, computing the equal-temperament
    /// frequency with A4 tuned to 440 Hz.
    pub fn from_midi(midi_note: u8) -> Self {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PianoInputSource {
    /// A mapped key on a conventional computer keyboard.
    ComputerKeyboard { key: char },
    /// The primary mouse button on an on-screen key.
    Mouse,
    /// A touch contact on an on-screen key.
    Touch { finger: u64 },
    /// An event loaded from a looping score.
    Score,
}

/// The expressive phase of a performed note.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PianoAction {
    /// A key began sounding. Velocity and pressure are normalized to `0.0..=1.0`.
    Attack {
        velocity: f32,
        pressure: Option<f32>,
    },
    /// A key stopped being held. Velocity is normalized to `0.0..=1.0`.
    Release { velocity: f32, held_for: Duration },
}

/// A lossless-in-time description of an on-screen piano performance event.
///
/// `voice_id` pairs an attack with its release even when the same note is
/// played concurrently by multiple inputs. `sequence` provides stable event
/// ordering when timestamps are equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PianoEvent {
    pub sequence: u64,
    pub timestamp: Duration,
    pub voice_id: u64,
    pub note: PianoNote,
    pub action: PianoAction,
    pub source: PianoInputSource,
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
    /// The octave whose computer-keyboard bindings are labeled above the
    /// keys; it already includes any held-Shift transposition. `None` hides
    /// the label row.
    label_octave: Option<i8>,
}

impl PianoKeyboard {
    pub(crate) fn new(
        appearances: HashMap<u8, PianoKeyAppearance>,
        label_octave: Option<i8>,
    ) -> Self {
        Self {
            appearances,
            label_octave,
        }
    }

    /// The top edge of the key area in canvas coordinates; the label row
    /// occupies the space above it when labels are enabled.
    fn key_top(&self) -> f32 {
        if self.label_octave.is_some() {
            PIANO_LABEL_ROW_HEIGHT
        } else {
            0.0
        }
    }

    /// The bounds of the key area within the canvas, excluding the label
    /// row.
    fn key_bounds(&self, bounds: Rectangle) -> Rectangle {
        let key_top = self.key_top();
        Rectangle::new(
            Point::new(bounds.x, bounds.y + key_top),
            Size::new(bounds.width, bounds.height - key_top),
        )
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
                let key_bounds = self.key_bounds(bounds);
                let position = cursor.position_in(key_bounds)?;
                let midi_note = note_at(position, key_bounds.size())?;
                if state.mouse_note.replace(midi_note).is_some() {
                    return None;
                }
                Some(
                    canvas::Action::publish(PianoMessage::Press {
                        midi_note,
                        velocity: strike_velocity(position, key_bounds.size(), midi_note),
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
                let key_bounds = self.key_bounds(bounds);
                let position = *position - iced::Vector::new(key_bounds.x, key_bounds.y);
                let midi_note = note_at(position, key_bounds.size())?;
                state.touch_notes.insert(*id, midi_note);
                Some(
                    canvas::Action::publish(PianoMessage::Press {
                        midi_note,
                        velocity: strike_velocity(position, key_bounds.size(), midi_note),
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
        let key_top = self.key_top();
        let key_height = bounds.height - key_top;
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
                Rectangle::new(Point::new(x, key_top), Size::new(white_width, key_height)),
                self.appearances
                    .get(&midi_note)
                    .copied()
                    .unwrap_or_default(),
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
                    Rectangle::new(Point::new(x, key_top), Size::new(black_width, black_height)),
                    self.appearances
                        .get(&midi_note)
                        .copied()
                        .unwrap_or_default(),
                );
            } else {
                whites_before += 1;
            }
        }

        // Keybind labels sit in the row above the keys, each centered over
        // its key. Only the bindings of the active octave are labeled.
        if let Some(octave) = self.label_octave {
            for (key, label) in MAPPED_KEY_LABELS {
                let Some(midi_note) = computer_key_note(key, octave, 0) else {
                    continue;
                };
                let mut text = canvas::Text::from(label);
                text.position = Point::new(key_center_x(midi_note, bounds.width), key_top / 2.0);
                text.color = Color::WHITE;
                text.size = Pixels(PIANO_LABEL_FONT_SIZE);
                text.align_x = Alignment::Center;
                text.align_y = Vertical::Center;
                frame.fill_text(text);
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

fn draw_white_key(frame: &mut canvas::Frame, bounds: Rectangle, appearance: PianoKeyAppearance) {
    let intensity = appearance.intensity.clamp(0.0, 1.0);
    let face = Path::rectangle(bounds.position(), bounds.size());
    frame.fill(&face, mix_color(Color::WHITE, Color::from_rgb8(255, 220, 142), intensity));
    frame.stroke(
        &face,
        canvas::Stroke::default()
            .with_color(mix_color(
                Color::from_rgb8(89, 89, 92),
                Color::from_rgb8(150, 100, 40),
                intensity,
            ))
            .with_width(0.8),
    );
}

/// A point inside a black key's bounding box, in the normalized coordinates
/// of NanoMoog's original SVG art: `u` runs across the key and `v` down its
/// height.
fn key_point(bounds: Rectangle, u: f32, v: f32) -> Point {
    Point::new(bounds.x + bounds.width * u, bounds.y + bounds.height * v)
}

/// Draws a black key in NanoMoog's style: a black body with a faint top
/// sheen, two highlight lines, a gray bullet band up top, a diagonal sheen
/// band below it, and a beveled front lip shaded white-to-black.
fn draw_black_key(frame: &mut canvas::Frame, bounds: Rectangle, appearance: PianoKeyAppearance) {
    let intensity = appearance.intensity.clamp(0.0, 1.0);
    let p = |u: f32, v: f32| key_point(bounds, u, v);

    // The body is black with a faint sheen fading in from the top edge.
    let face = Path::rectangle(bounds.position(), bounds.size());
    let fill = canvas::gradient::Linear::new(
        Point::new(bounds.x, bounds.y),
        Point::new(bounds.x, bounds.y + bounds.height),
    )
    .add_stop(
        0.0,
        mix_color(Color::from_rgb8(42, 42, 42), Color::from_rgb8(105, 50, 16), intensity),
    )
    .add_stop(0.10, mix_color(Color::BLACK, Color::from_rgb8(90, 42, 12), intensity))
    .add_stop(1.0, mix_color(Color::BLACK, Color::from_rgb8(90, 42, 12), intensity));
    frame.fill(&face, fill);

    // Thin highlight lines run down the key face between its bands.
    let line_color = mix_color(
        Color::from_rgb8(123, 123, 123),
        Color::from_rgb8(196, 142, 60),
        intensity,
    );
    for u in [0.2117_f32, 0.7748] {
        let line = Path::line(p(u, 0.0), p(u, 0.8695));
        frame.stroke(
            &line,
            canvas::Stroke::default()
                .with_color(line_color)
                .with_width(0.8),
        );
    }

    // The beveled front lip, shaded white-to-black from top to bottom.
    let lip = Path::new(|path| {
        path.move_to(p(0.2117, 0.8747));
        path.line_to(p(0.7748, 0.8747));
        path.line_to(p(0.9685, 0.9940));
        path.line_to(p(0.0270, 0.9940));
        path.close();
    });
    let lip_fill = canvas::gradient::Linear::new(p(0.0, 0.5588), p(0.0, 0.9902))
        .add_stop(
            0.0,
            mix_color(Color::WHITE, Color::from_rgb8(255, 220, 142), intensity),
        )
        .add_stop(1.0, mix_color(Color::BLACK, Color::from_rgb8(80, 38, 10), intensity));
    frame.fill(&lip, lip_fill);
    frame.stroke(
        &lip,
        canvas::Stroke::default()
            .with_color(mix_color(
                Color::from_rgb8(73, 73, 73),
                Color::from_rgb8(120, 66, 20),
                intensity,
            ))
            .with_width(0.8),
    );

    // The upper band: a gray bullet with a curved lower end.
    let bullet = Path::new(|path| {
        path.move_to(p(0.2568, 0.3511));
        path.line_to(p(0.2568, 0.0137));
        path.line_to(p(0.7387, 0.0137));
        path.line_to(p(0.7387, 0.2215));
        path.bezier_curve_to(p(0.7342, 0.2215), p(0.2613, 0.2300), p(0.2568, 0.3511));
        path.close();
    });
    frame.fill(
        &bullet,
        mix_color(Color::from_rgb8(99, 99, 99), Color::from_rgb8(150, 84, 26), intensity),
    );

    // The lower band: a diagonal white-to-black sheen.
    let sheen = Path::new(|path| {
        path.move_to(p(0.7342, 0.8695));
        path.line_to(p(0.2568, 0.8695));
        path.line_to(p(0.2568, 0.5270));
        path.bezier_curve_to(p(0.2568, 0.5270), p(0.7297, 0.5717), p(0.7387, 0.6215));
        path.bezier_curve_to(p(0.7432, 0.6721), p(0.7342, 0.8695), p(0.7342, 0.8695));
        path.close();
    });
    let sheen_fill = canvas::gradient::Linear::new(p(-0.9640, 0.9441), p(0.8234, 0.6431))
        .add_stop(
            0.0,
            mix_color(Color::WHITE, Color::from_rgb8(255, 220, 142), intensity),
        )
        .add_stop(1.0, mix_color(Color::BLACK, Color::from_rgb8(80, 38, 10), intensity));
    frame.fill(&sheen, sheen_fill);
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

/// The scientific-pitch octave selected by a number key. `0` selects the
/// partial bottom octave (only A0 and B0 exist on the keyboard) and `8`
/// through `9` select nothing because the keyboard ends at C8, which the
/// home row reaches in octave 7.
pub(crate) fn computer_octave_key(key: char) -> Option<i8> {
    let digit = key.to_digit(10)?;
    (digit <= 7).then_some(digit as i8)
}

/// The semitone offset of a home-row white-key letter from the A that opens
/// the row: `a` is A, `s` is B, `d` is C, and so on through `'`, which is D
/// of the next octave. Enter (`\r`) closes the row at E of the next
/// octave. Letters outside the row map to `None`.
const fn white_key_semitones(key: char) -> Option<i32> {
    match key {
        'a' => Some(0),
        's' => Some(2),
        'd' => Some(3),
        'f' => Some(5),
        'g' => Some(7),
        'h' => Some(8),
        'j' => Some(10),
        'k' => Some(12),
        'l' => Some(14),
        ';' => Some(15),
        '\'' => Some(17),
        '\r' => Some(19),
        _ => None,
    }
}

/// The semitone offset of a top-row black-key letter from the A that opens
/// the home row: `q` is Ab/G# just below that A, `w` is A#, and so on through
/// `\`, which is F# of the next octave. Letters outside the row map to
/// `None`.
const fn black_key_semitones(key: char) -> Option<i32> {
    match key {
        'q' => Some(-1),
        'w' => Some(1),
        'r' => Some(4),
        't' => Some(6),
        'u' => Some(9),
        'i' => Some(11),
        'o' => Some(13),
        'p' => Some(16),
        '[' => Some(18),
        '\\' => Some(21),
        _ => None,
    }
}

/// Every computer-keyboard character that can strike a note, paired with
/// the string drawn as its label; Enter is shown as `↵`.
const MAPPED_KEY_LABELS: [(char, &str); 22] = [
    ('a', "a"),
    ('s', "s"),
    ('d', "d"),
    ('f', "f"),
    ('g', "g"),
    ('h', "h"),
    ('j', "j"),
    ('k', "k"),
    ('l', "l"),
    (';', ";"),
    ('\'' , "'"),
    ('\r', "↵"),
    ('q', "q"),
    ('w', "w"),
    ('r', "r"),
    ('t', "t"),
    ('u', "u"),
    ('i', "i"),
    ('o', "o"),
    ('p', "p"),
    ('[', "["),
    ('\\', "\\"),
];

/// The horizontal center of a key within a keyboard of the given width,
/// matching the layout drawn by [`PianoKeyboard`]: white keys are centered
/// in their slot and black keys on the boundary between two white keys.
fn key_center_x(midi_note: u8, width: f32) -> f32 {
    let white_width = width / WHITE_KEY_COUNT;
    let whites_before =
        (FIRST_MIDI_NOTE..midi_note).filter(|note| !is_black(*note)).count() as u8;
    if is_black(midi_note) {
        f32::from(whites_before) * white_width
    } else {
        (f32::from(whites_before) + 0.5) * white_width
    }
}

/// The MIDI note struck by a computer keyboard key in the selected
/// scientific-pitch octave, transposed one octave down or up when `shift`
/// is negative or positive, or `None` if the key is unmapped or the note
/// falls outside the 88-key range. The home row plays the white keys from A
/// of the octave through E of the next (Enter); the top row plays the black
/// keys between them.
pub(crate) fn computer_key_note(key: char, octave: i8, shift: i8) -> Option<u8> {
    let key = key.to_ascii_lowercase();
    let semitones = white_key_semitones(key).or_else(|| black_key_semitones(key))?;
    let midi_note = 12 * (i32::from(octave) + 1) + 9 + semitones + 12 * i32::from(shift);
    u8::try_from(midi_note)
        .ok()
        .filter(|note| (FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(note))
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
    fn home_row_walks_white_keys_from_a() {
        // The row walks the white keys from A of the selected octave: a is
        // A4, s is B4, d wraps to C5, and so on.
        let expected = [69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86];
        for (key, midi_note) in "asd fg hjkl;'"
            .chars()
            .filter(|c| *c != ' ')
            .zip(expected)
        {
            assert_eq!(computer_key_note(key, 4, 0), Some(midi_note));
        }
        // The row is case-insensitive and the mapping reaches both ends of
        // the 88-key range.
        assert_eq!(computer_key_note('A', 4, 0), Some(69));
        assert_eq!(computer_key_note('a', 0, 0), Some(21));
        assert_eq!(computer_key_note('s', 0, 0), Some(23));
        assert_eq!(computer_key_note('d', 7, 0), Some(108));
        // Notes outside the keyboard are not struck.
        assert_eq!(computer_key_note('a', -1, 0), None);
        assert_eq!(computer_key_note('\'' , 8, 0), None);
        // Unmapped letters are ignored.
        assert_eq!(computer_key_note('z', 4, 0), None);
        assert_eq!(computer_key_note('e', 4, 0), None);
        assert_eq!(computer_key_note('!', 4, 0), None);
    }

    #[test]
    fn top_row_walks_the_black_keys() {
        // q is Ab4/G#4, w is A#4, and so on through \, which is F#6.
        let expected = [68, 70, 73, 75, 78, 80, 82, 85, 87, 90];
        for (key, midi_note) in "qwrtuiop[\\".chars().zip(expected) {
            assert_eq!(computer_key_note(key, 4, 0), Some(midi_note));
        }
        // The row is case-insensitive and range-checked.
        assert_eq!(computer_key_note('W', 4, 0), Some(70));
        assert_eq!(computer_key_note('\\', 9, 0), None);
    }

    #[test]
    fn shift_octaves_transpose_every_mapped_key() {
        // Negative shifts drop an octave; positive shifts raise one.
        assert_eq!(computer_key_note('a', 4, -1), Some(57));
        assert_eq!(computer_key_note('a', 4, 1), Some(81));
        assert_eq!(computer_key_note('w', 4, -1), Some(58));
        assert_eq!(computer_key_note('\\', 4, 1), Some(102));
        // Shifted notes are range-checked like natural ones.
        assert_eq!(computer_key_note('a', 0, -1), None);
        assert_eq!(computer_key_note('a', 8, 1), None);
        assert_eq!(computer_key_note('\'' , 8, 1), None);
    }

    #[test]
    fn number_keys_select_octaves() {
        for digit in 0..=7 {
            let key = char::from(b'0' + digit);
            assert_eq!(computer_octave_key(key), Some(digit as i8));
        }
        assert_eq!(computer_octave_key('8'), None);
        assert_eq!(computer_octave_key('9'), None);
        assert_eq!(computer_octave_key('a'), None);
    }

    #[test]
    fn computer_keys_reach_every_key_on_the_keyboard() {
        // The home row plus the top row cover all 88 keys from A0 to C8.
        let letters = "asdfghjkl;'qwrtuiop[\\\r";
        for midi_note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
            let reachable = letters.chars().any(|key| {
                (0i8..=7)
                    .any(|octave| computer_key_note(key, octave, 0) == Some(midi_note))
            });
            assert!(reachable, "MIDI note {midi_note} is unreachable");
        }
    }

    #[test]
    fn the_label_row_extends_the_piano_height() {
        assert_eq!(piano_height(false), PIANO_HEIGHT);
        assert_eq!(piano_height(true), PIANO_HEIGHT + PIANO_LABEL_ROW_HEIGHT);
    }

    #[test]
    fn key_centers_match_the_drawn_layout() {
        let width = 1040.0;
        let white_width = width / WHITE_KEY_COUNT;
        // D4 is the 25th white key (index 24): A0 and B0, then the seven
        // white keys of each of octaves 1 through 3, then C4. C#4 sits on
        // the boundary after it, between C4 and D4.
        assert_eq!(key_center_x(62, width), 24.5 * white_width);
        assert_eq!(key_center_x(61, width), 24.0 * white_width);
    }

    #[test]
    fn labels_cover_exactly_the_keys_of_the_active_octave() {
        // At octave 4 every mapped key lands on the keyboard, so all 22
        // bindings are labeled from G#4 (q) through F#6 (\\).
        let labeled: Vec<u8> = MAPPED_KEY_LABELS
            .iter()
            .filter_map(|(key, _)| computer_key_note(*key, 4, 0))
            .collect();
        assert_eq!(labeled.len(), 22);
        assert_eq!(labeled.iter().min(), Some(&68));
        assert_eq!(labeled.iter().max(), Some(&90));

        // At octave 7 only the bindings that still land on the keyboard are
        // labeled: a, s, d and q, w.
        let labeled: Vec<u8> = MAPPED_KEY_LABELS
            .iter()
            .filter_map(|(key, _)| computer_key_note(*key, 7, 0))
            .collect();
        assert_eq!(labeled, vec![105, 107, 108, 104, 106]);
    }
}
