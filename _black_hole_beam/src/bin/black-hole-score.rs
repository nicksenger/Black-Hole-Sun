//! Convert Standard MIDI files into JSON piano scores for `black-hole-beam`.
//!
//! The output is accepted by [`BeamBuilder::score`](black_hole_beam::BeamBuilder::score):
//! a JSON array of `PianoEvent` values, or — when `--loop-duration` is given — an
//! object with `events` and a numeric `loop_duration` in seconds.
//!
//! MIDI note-on/note-off pairs become `Attack`/`Release` events sharing a
//! `voice_id`. Velocities are normalized to `0.0..=1.0`; a note-off with
//! velocity zero releases at the note's attack velocity. Notes still held at
//! the end of the file are released there so the score loops cleanly.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use black_hole_beam::{PianoAction, PianoEvent, PianoInputSource, PianoNote};
use clap::Parser;
use midly::{Format, MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use serde::Serialize;

const FIRST_MIDI_NOTE: u8 = 21; // A0
const LAST_MIDI_NOTE: u8 = 108; // C8
/// The MIDI default tempo when a file carries no tempo meta events: 120 BPM.
const DEFAULT_MICROS_PER_QUARTER: u32 = 500_000;

#[derive(Parser)]
#[command(
    name = "black-hole-score",
    version,
    about = "Convert a Standard MIDI file into a JSON piano score for black-hole-beam"
)]
struct Args {
    /// MIDI file to convert (`-` reads from standard input)
    input: String,

    /// Write the JSON score to this file instead of standard output
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Explicit loop length in seconds; emits the wrapped score form with `loop_duration`
    #[arg(long, value_name = "SECONDS")]
    loop_duration: Option<f64>,

    /// Only include notes from this MIDI channel (0-15)
    #[arg(long, value_name = "CHANNEL")]
    channel: Option<u8>,
}

#[derive(Clone, Copy)]
struct RawNote {
    channel: u8,
    key: u8,
    velocity: u8,
    tick: u64,
    on: bool,
}

/// A note-on/note-off pair in file ticks. `off_velocity` of zero means the
/// release carries no velocity information (a note-off with velocity zero or
/// a synthesized end-of-file release).
#[derive(Clone, Copy)]
struct NotePair {
    key: u8,
    on_tick: u64,
    off_tick: u64,
    on_velocity: u8,
    off_velocity: u8,
}

#[derive(Serialize)]
struct ScoreDocument<'a> {
    events: &'a [PianoEvent],
    loop_duration: f64,
}

/// A piecewise tempo map converting file ticks to seconds.
struct TempoMap {
    starts: Vec<u64>,
    seconds: Vec<f64>,
    micros_per_quarter: Vec<u32>,
    division: u32,
}

impl TempoMap {
    fn new(tempos: Vec<(u64, u32)>, division: u32) -> Self {
        let mut tempos = tempos;
        tempos.sort_by_key(|tempo| tempo.0);

        let mut starts = Vec::new();
        let mut seconds = Vec::new();
        let mut micros_per_quarter = Vec::new();
        let mut tick: u64 = 0;
        let mut elapsed: f64 = 0.0;
        let mut micros = DEFAULT_MICROS_PER_QUARTER;
        for &(tempo_tick, tempo_micros) in &tempos {
            if tempo_tick > tick {
                starts.push(tick);
                seconds.push(elapsed);
                micros_per_quarter.push(micros);
                elapsed += (tempo_tick - tick) as f64 * f64::from(micros)
                    / 1_000_000.0
                    / f64::from(division);
                tick = tempo_tick;
            }
            micros = tempo_micros;
        }
        starts.push(tick);
        seconds.push(elapsed);
        micros_per_quarter.push(micros);

        Self {
            starts,
            seconds,
            micros_per_quarter,
            division,
        }
    }

    fn at(&self, tick: u64) -> f64 {
        let segment = match self.starts.binary_search(&tick) {
            Ok(segment) => segment,
            Err(segment) => segment - 1,
        };
        self.seconds[segment]
            + (tick - self.starts[segment]) as f64 * f64::from(self.micros_per_quarter[segment])
                / 1_000_000.0
                / f64::from(self.division)
    }
}

/// The note and tempo events of a parsed MIDI file, in file ticks.
struct ParsedMidi {
    division: u32,
    notes: Vec<RawNote>,
    tempos: Vec<(u64, u32)>,
    end_tick: u64,
}

fn parse_midi(bytes: &[u8], wanted_channel: Option<u8>) -> Result<ParsedMidi, String> {
    let smf = Smf::parse(bytes).map_err(|error| format!("could not parse MIDI file: {error}"))?;

    let division = match smf.header.timing {
        Timing::Metrical(ticks_per_beat) => u32::from(ticks_per_beat.as_int()),
        Timing::Timecode(..) => return Err("SMPTE-timed MIDI files are not supported".to_string()),
    };
    if division == 0 {
        return Err("the MIDI file has zero ticks per beat".to_string());
    }

    let tracks: Vec<&Vec<midly::TrackEvent>> = match smf.header.format {
        Format::SingleTrack | Format::Parallel => smf.tracks.iter().collect(),
        Format::Sequential => {
            eprintln!("warning: sequential MIDI file; converting only the first track");
            smf.tracks.iter().take(1).collect()
        }
    };

    let mut notes = Vec::new();
    let mut tempos = Vec::new();
    let mut end_tick: u64 = 0;
    for track in &tracks {
        let mut tick: u64 = 0;
        for event in track.iter() {
            tick += u64::from(event.delta.as_int());
            match &event.kind {
                TrackEventKind::Midi { channel, message } => {
                    let note_channel = u8::from(*channel);
                    if wanted_channel.is_some_and(|wanted| wanted != note_channel) {
                        continue;
                    }
                    match message {
                        // By MIDI convention a note-on with velocity zero is a note-off.
                        MidiMessage::NoteOn { key, vel } => notes.push(RawNote {
                            channel: note_channel,
                            key: u8::from(*key),
                            velocity: u8::from(*vel),
                            tick,
                            on: u8::from(*vel) != 0,
                        }),
                        MidiMessage::NoteOff { key, vel } => notes.push(RawNote {
                            channel: note_channel,
                            key: u8::from(*key),
                            velocity: u8::from(*vel),
                            tick,
                            on: false,
                        }),
                        _ => {}
                    }
                }
                TrackEventKind::Meta(MetaMessage::Tempo(micros_per_quarter)) => {
                    tempos.push((tick, u32::from(*micros_per_quarter)));
                }
                _ => {}
            }
        }
        end_tick = end_tick.max(tick);
    }

    Ok(ParsedMidi {
        division,
        notes,
        tempos,
        end_tick,
    })
}

/// Pair note-ons with their note-offs in global time order. A retriggered
/// note (a new note-on before the previous note-off) closes the earlier voice
/// at the retrigger instant; notes still held at `end_tick` are released
/// there so the score loops cleanly instead of re-attacking over its own
/// sustain.
fn pair_notes(notes: &[RawNote], end_tick: u64) -> Vec<NotePair> {
    let mut ordered = notes.to_vec();
    ordered.sort_by_key(|note| note.tick);

    let mut held: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
    let mut pairs = Vec::new();
    for note in &ordered {
        match held.entry((note.channel, note.key)) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let (on_tick, on_velocity) = *occupied.get();
                if note.on {
                    pairs.push(NotePair {
                        key: note.key,
                        on_tick,
                        off_tick: note.tick,
                        on_velocity,
                        off_velocity: 0,
                    });
                    occupied.insert((note.tick, note.velocity));
                } else {
                    pairs.push(NotePair {
                        key: note.key,
                        on_tick,
                        off_tick: note.tick,
                        on_velocity,
                        off_velocity: note.velocity,
                    });
                    occupied.remove();
                }
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                if note.on {
                    vacant.insert((note.tick, note.velocity));
                }
                // Stray releases without an open note are ignored.
            }
        }
    }
    for ((_, key), (on_tick, on_velocity)) in &held {
        pairs.push(NotePair {
            key: *key,
            on_tick: *on_tick,
            off_tick: end_tick,
            on_velocity: *on_velocity,
            off_velocity: 0,
        });
    }
    pairs
}

/// Build score events from note pairs. Returns the events (unsorted, with
/// `sequence` unset) and the number of notes skipped for falling outside the
/// 88-key range.
fn build_events(pairs: &[NotePair], tempo_map: &TempoMap) -> (Vec<PianoEvent>, usize) {
    let mut events = Vec::new();
    let mut skipped_notes = 0;
    let mut voice_id = 0u64;
    for pair in pairs {
        if !(FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&pair.key) {
            skipped_notes += 1;
            continue;
        }
        let note = PianoNote::from_midi(pair.key);
        let attack_velocity = f32::from(pair.on_velocity) / 127.0;
        // Note-offs without velocity information release at the attack
        // velocity, preserving the phrase's dynamics.
        let release_velocity = if pair.off_velocity == 0 {
            attack_velocity
        } else {
            f32::from(pair.off_velocity) / 127.0
        };
        let on_seconds = tempo_map.at(pair.on_tick);
        let off_seconds = tempo_map.at(pair.off_tick);
        voice_id += 1;
        events.push(PianoEvent {
            sequence: 0,
            timestamp: Duration::from_secs_f64(on_seconds),
            voice_id,
            note,
            action: PianoAction::Attack {
                velocity: attack_velocity,
                pressure: None,
            },
            source: PianoInputSource::Score,
        });
        events.push(PianoEvent {
            sequence: 0,
            timestamp: Duration::from_secs_f64(off_seconds),
            voice_id,
            note,
            action: PianoAction::Release {
                velocity: release_velocity,
                held_for: Duration::from_secs_f64((off_seconds - on_seconds).max(0.0)),
            },
            source: PianoInputSource::Score,
        });
    }
    (events, skipped_notes)
}

/// Order events by performance time and assign stable sequence numbers.
fn finalize_events(events: &mut [PianoEvent]) {
    events.sort_by(|a, b| {
        a.timestamp
            .as_secs_f64()
            .partial_cmp(&b.timestamp.as_secs_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| action_rank(&a.action).cmp(&action_rank(&b.action)))
            .then_with(|| a.note.midi_note.cmp(&b.note.midi_note))
    });
    for (index, event) in events.iter_mut().enumerate() {
        event.sequence = index as u64 + 1;
    }
}

fn action_rank(action: &PianoAction) -> u8 {
    match action {
        PianoAction::Attack { .. } => 0,
        PianoAction::Release { .. } => 1,
    }
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if let Some(channel) = args.channel {
        if channel > 15 {
            return Err(format!("MIDI channel {channel} is out of range (0-15)"));
        }
    }
    if let Some(loop_duration) = args.loop_duration {
        if !loop_duration.is_finite() || loop_duration <= 0.0 {
            return Err(
                "loop duration must be a finite number of seconds greater than zero".to_string(),
            );
        }
    }

    let bytes = read_input(&args.input)?;
    let midi = parse_midi(&bytes, args.channel)?;
    let tempo_map = TempoMap::new(midi.tempos, midi.division);
    let pairs = pair_notes(&midi.notes, midi.end_tick);
    let (mut events, skipped_notes) = build_events(&pairs, &tempo_map);
    if skipped_notes > 0 {
        eprintln!(
            "warning: skipped {skipped_notes} notes outside the 88-key range \
             {FIRST_MIDI_NOTE}..={LAST_MIDI_NOTE}"
        );
    }
    if events.is_empty() {
        return Err(format!(
            "the MIDI file contains no notes in the 88-key range {FIRST_MIDI_NOTE}..={LAST_MIDI_NOTE}"
        ));
    }
    finalize_events(&mut events);

    let last_timestamp = events.last().expect("events is not empty").timestamp;
    let json = if let Some(loop_duration) = args.loop_duration {
        if Duration::from_secs_f64(loop_duration) < last_timestamp {
            return Err(format!(
                "loop duration ({loop_duration}s) precedes the last event ({}s)",
                last_timestamp.as_secs_f64()
            ));
        }
        serde_json::to_string_pretty(&ScoreDocument {
            events: &events,
            loop_duration,
        })
        .map_err(|error| format!("could not serialize the score: {error}"))?
    } else {
        serde_json::to_string_pretty(&events)
            .map_err(|error| format!("could not serialize the score: {error}"))?
    };

    match &args.output {
        Some(path) => std::fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("could not write {}: {error}", path.display()))?,
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(handle, "{json}")
                .map_err(|error| format!("could not write to stdout: {error}"))?;
        }
    }

    Ok(())
}

fn read_input(input: &str) -> Result<Vec<u8>, String> {
    if input == "-" {
        let mut bytes = Vec::new();
        io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|error| format!("could not read MIDI from standard input: {error}"))?;
        return Ok(bytes);
    }
    std::fs::read(input).map_err(|error| format!("could not read {input}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-measure C-major file at 120 BPM (division 480): a whole-note C4
    /// followed by two quarter notes, built as raw SMF bytes.
    fn c_major_smf() -> Vec<u8> {
        let mut track = Vec::new();
        // Tempo: 500_000 us per quarter note (120 BPM).
        track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        // C4 on at tick 0.
        track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x50]);
        // C4 off at tick 480 (delta 480).
        track.extend_from_slice(&[0x83, 0x60, 0x80, 0x3C, 0x00]);
        // E4 on at tick 480 (delta 0), off at tick 720 (delta 240).
        track.extend_from_slice(&[0x00, 0x90, 0x40, 0x50]);
        track.extend_from_slice(&[0x81, 0x70, 0x80, 0x40, 0x00]);
        // G4 on at tick 720 (delta 0), off at tick 960 (delta 240).
        track.extend_from_slice(&[0x00, 0x90, 0x43, 0x50]);
        track.extend_from_slice(&[0x81, 0x70, 0x80, 0x43, 0x00]);
        // End of track.
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6u32.to_be_bytes()); // MThd payload is 6 bytes
        file.extend_from_slice(&0u16.to_be_bytes()); // format 0
        file.extend_from_slice(&1u16.to_be_bytes()); // one track
        file.extend_from_slice(&480u16.to_be_bytes()); // ticks per quarter
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(track.len() as u32).to_be_bytes());
        file.extend_from_slice(&track);
        file
    }

    fn convert(bytes: &[u8]) -> Vec<PianoEvent> {
        let midi = parse_midi(bytes, None).expect("hand-built SMF should parse");
        let tempo_map = TempoMap::new(midi.tempos, midi.division);
        let pairs = pair_notes(&midi.notes, midi.end_tick);
        let (mut events, skipped) = build_events(&pairs, &tempo_map);
        assert_eq!(skipped, 0);
        finalize_events(&mut events);
        events
    }

    #[test]
    fn tempo_map_applies_default_and_explicit_tempos() {
        // 480 ticks per quarter at the default 120 BPM: 480 ticks is 0.5s.
        let map = TempoMap::new(Vec::new(), 480);
        assert!((map.at(480) - 0.5).abs() < 1e-9);

        // A tempo change at tick 480 to 250_000 us (240 BPM) halves the rate.
        let map = TempoMap::new(vec![(0, 500_000), (480, 250_000)], 480);
        assert!((map.at(480) - 0.5).abs() < 1e-9);
        assert!((map.at(960) - 0.75).abs() < 1e-9);
    }

    #[test]
    fn converts_note_pairs_into_score_events() {
        let events = convert(&c_major_smf());
        assert_eq!(events.len(), 6);

        // C4: 0.0s..0.5s, E4: 0.5s..0.75s, G4: 0.75s..1.0s at 120 BPM.
        // At equal timestamps attacks order before releases, so the layout
        // is: C4 on, E4 on, C4 off, G4 on, E4 off, G4 off.
        let c4_attack = &events[0];
        assert_eq!(c4_attack.note.midi_note, 60);
        assert_eq!(c4_attack.timestamp, Duration::ZERO);
        match &c4_attack.action {
            PianoAction::Attack { velocity, .. } => {
                assert!((velocity - f32::from(80u8) / 127.0).abs() < 1e-6);
            }
            other => panic!("expected an attack, got {other:?}"),
        }

        let e4_attack = &events[1];
        assert_eq!(e4_attack.note.midi_note, 64);
        assert!((e4_attack.timestamp.as_secs_f64() - 0.5).abs() < 1e-9);

        let c4_release = &events[2];
        assert_eq!(c4_release.voice_id, c4_attack.voice_id);
        assert_eq!(
            &c4_release.action,
            &PianoAction::Release {
                velocity: f32::from(80u8) / 127.0,
                held_for: Duration::from_secs_f64(0.5),
            }
        );

        let g4_release = &events[5];
        assert_eq!(g4_release.note.midi_note, 67);
        assert!((g4_release.timestamp.as_secs_f64() - 1.0).abs() < 1e-9);

        // Sequence numbers are dense and follow performance time.
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            (1..=6).collect::<Vec<_>>()
        );
        // Every event is source-tagged for the score playback path.
        assert!(events
            .iter()
            .all(|event| event.source == PianoInputSource::Score));

        // The serialized form round-trips through the score loader's schema.
        let json = serde_json::to_string(&events).unwrap();
        assert!(json.contains(r#""source":"Score""#));
        assert_eq!(
            serde_json::from_str::<Vec<PianoEvent>>(&json).unwrap(),
            events
        );
    }

    #[test]
    fn retriggered_notes_close_the_earlier_voice() {
        let mut track = Vec::new();
        // C4 on at tick 0, C4 on again at tick 240 (retrigger), off at 480.
        track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x50]);
        track.extend_from_slice(&[0x81, 0x70, 0x90, 0x3C, 0x50]);
        track.extend_from_slice(&[0x81, 0x70, 0x80, 0x3C, 0x40]);
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6u32.to_be_bytes()); // MThd payload is 6 bytes
        file.extend_from_slice(&0u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&480u16.to_be_bytes());
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(track.len() as u32).to_be_bytes());
        file.extend_from_slice(&track);

        let events = convert(&file);
        assert_eq!(events.len(), 4);
        // Layout: voice 1 on @0.0, voice 2 on @0.25 (attack before release
        // at equal timestamps), voice 1 off @0.25, voice 2 off @0.5.
        let first_voice = events[0].voice_id;
        assert_ne!(events[1].voice_id, first_voice);
        let first_release = &events[2];
        assert_eq!(first_release.voice_id, first_voice);
        match &first_release.action {
            PianoAction::Release { held_for, .. } => {
                assert!((held_for.as_secs_f64() - 0.25).abs() < 1e-9);
            }
            other => panic!("expected a release, got {other:?}"),
        }
        assert!((first_release.timestamp.as_secs_f64() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn notes_held_at_end_of_file_are_released_there() {
        let mut track = Vec::new();
        // C4 on at tick 0, never released.
        track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x50]);
        // E4 on at tick 480 (delta 480), off at tick 960 (delta 480) gives the
        // file a nonzero end.
        track.extend_from_slice(&[0x83, 0x60, 0x90, 0x40, 0x50]);
        track.extend_from_slice(&[0x83, 0x60, 0x80, 0x40, 0x00]);
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6u32.to_be_bytes()); // MThd payload is 6 bytes
        file.extend_from_slice(&0u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&480u16.to_be_bytes());
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(track.len() as u32).to_be_bytes());
        file.extend_from_slice(&track);

        let events = convert(&file);
        assert_eq!(events.len(), 4);
        // C4's synthesized release lands at the file end (tick 960, 1.0s).
        let c4_release = events
            .iter()
            .find(|event| {
                matches!(
                    event.action,
                    PianoAction::Release { .. } if event.note.midi_note == 60
                )
            })
            .expect("C4 should be released at the file end");
        assert!((c4_release.timestamp.as_secs_f64() - 1.0).abs() < 1e-9);
        match &c4_release.action {
            PianoAction::Release { held_for, .. } => {
                assert!((held_for.as_secs_f64() - 1.0).abs() < 1e-9);
            }
            other => panic!("expected a release, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_notes_are_skipped_with_a_count() {
        let mut track = Vec::new();
        // B0 (key 20, below the range) and D#8 (key 109, above it).
        track.extend_from_slice(&[0x00, 0x90, 0x14, 0x50]);
        track.extend_from_slice(&[0x78, 0x80, 0x14, 0x00]);
        track.extend_from_slice(&[0x78, 0x90, 0x6D, 0x50]);
        track.extend_from_slice(&[0x78, 0x80, 0x6D, 0x00]);
        // C4 in range.
        track.extend_from_slice(&[0x78, 0x90, 0x3C, 0x50]);
        track.extend_from_slice(&[0x78, 0x80, 0x3C, 0x00]);
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut file = Vec::new();
        file.extend_from_slice(b"MThd");
        file.extend_from_slice(&6u32.to_be_bytes()); // MThd payload is 6 bytes
        file.extend_from_slice(&0u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&480u16.to_be_bytes());
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(track.len() as u32).to_be_bytes());
        file.extend_from_slice(&track);

        let midi = parse_midi(&file, None).unwrap();
        let tempo_map = TempoMap::new(midi.tempos, midi.division);
        let pairs = pair_notes(&midi.notes, midi.end_tick);
        let (events, skipped) = build_events(&pairs, &tempo_map);
        assert_eq!(skipped, 2);
        assert_eq!(events.len(), 2);
    }
}
