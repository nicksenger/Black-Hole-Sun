//! Loading and clocking looping JSON piano scores.

use std::fs;
use std::path::Path;
use std::time::Duration;

use iced::time::Instant;
use serde::Deserialize;

use crate::{PianoAction, PianoEvent, PianoNote};

pub(crate) const SCORE_TICK_INTERVAL: Duration = Duration::from_millis(5);
const MAX_EVENTS_PER_TICK: usize = 4_096;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ScoreDocument {
    Events(Vec<PianoEvent>),
    Wrapped {
        events: Vec<PianoEvent>,
        #[serde(default)]
        loop_duration: Option<f64>,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DueScoreEvent {
    pub cycle: u64,
    pub event: PianoEvent,
}

pub(crate) struct PianoScorePlayback {
    events: Vec<PianoEvent>,
    loop_duration: Duration,
    started_at: Instant,
    event_index: usize,
    cycle: u64,
}

impl PianoScorePlayback {
    pub(crate) fn load(path: &Path, started_at: Instant) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        Self::from_json(&json, started_at)
            .map_err(|error| format!("invalid piano score {}: {error}", path.display()))
    }

    pub(crate) fn from_json(json: &str, started_at: Instant) -> Result<Self, String> {
        let document: ScoreDocument =
            serde_json::from_str(json).map_err(|error| error.to_string())?;
        let (mut events, explicit_loop_duration) = match document {
            ScoreDocument::Events(events) => (events, None),
            ScoreDocument::Wrapped {
                events,
                loop_duration,
            } => (events, loop_duration),
        };
        if events.is_empty() {
            return Err("the event list is empty".to_string());
        }

        for event in &mut events {
            if !(21..=108).contains(&event.note.midi_note) {
                return Err(format!(
                    "MIDI note {} is outside the 88-key range 21..=108",
                    event.note.midi_note
                ));
            }
            event.note = PianoNote::from_midi(event.note.midi_note);
            let (velocity, pressure) = match event.action {
                PianoAction::Attack { velocity, pressure } => (velocity, pressure),
                PianoAction::Release { velocity, .. } => (velocity, None),
            };
            if !velocity.is_finite() || !(0.0..=1.0).contains(&velocity) {
                return Err(format!(
                    "event {} has velocity outside 0.0..=1.0",
                    event.sequence
                ));
            }
            if pressure
                .is_some_and(|pressure| !pressure.is_finite() || !(0.0..=1.0).contains(&pressure))
            {
                return Err(format!(
                    "event {} has pressure outside 0.0..=1.0",
                    event.sequence
                ));
            }
        }
        events.sort_by_key(|event| (event.timestamp, event.sequence));

        let last_timestamp = events.last().expect("events is not empty").timestamp;
        let loop_duration = if let Some(seconds) = explicit_loop_duration {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err("loop_duration must be a finite number greater than zero".to_string());
            }
            Duration::from_secs_f64(seconds)
        } else {
            last_timestamp
        };
        if loop_duration.is_zero() {
            return Err(
                "the inferred loop duration is zero; add a later event or loop_duration"
                    .to_string(),
            );
        }
        if loop_duration < last_timestamp {
            return Err(format!(
                "loop_duration ({loop_duration:?}) precedes the last event ({last_timestamp:?})"
            ));
        }

        Ok(Self {
            events,
            loop_duration,
            started_at,
            event_index: 0,
            cycle: 0,
        })
    }

    pub(crate) fn take_due(&mut self, now: Instant) -> Vec<DueScoreEvent> {
        let elapsed = now.saturating_duration_since(self.started_at);
        let mut due = Vec::new();
        while due.len() < MAX_EVENTS_PER_TICK {
            let event = self.events[self.event_index];
            let scheduled = self
                .loop_duration
                .saturating_mul(self.cycle as u32)
                .saturating_add(event.timestamp);
            if scheduled > elapsed {
                break;
            }
            due.push(DueScoreEvent {
                cycle: self.cycle,
                event,
            });
            self.event_index += 1;
            if self.event_index == self.events.len() {
                self.event_index = 0;
                self.cycle = self.cycle.wrapping_add(1);
            }
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCORE: &str = r#"[
      {"sequence":2,"timestamp":0.5,"voice_id":9,
       "note":{"midi_note":60,"frequency_hz":1.0},
       "action":{"Release":{"velocity":0.3,"held_for":0.5}},"source":"Score"},
      {"sequence":1,"timestamp":0.0,"voice_id":9,
       "note":{"midi_note":60,"frequency_hz":1.0},
       "action":{"Attack":{"velocity":0.8,"pressure":0.6}},"source":"Score"}
    ]"#;

    #[test]
    fn parses_sorts_normalizes_and_loops_score_events() {
        let start = Instant::now();
        let mut score = PianoScorePlayback::from_json(SCORE, start).unwrap();
        let first = score.take_due(start);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].event.sequence, 1);
        assert!((first[0].event.note.frequency_hz - 261.625_55).abs() < 0.001);

        let boundary = score.take_due(start + Duration::from_millis(500));
        assert_eq!(boundary.len(), 2);
        assert_eq!(boundary[0].event.sequence, 2);
        assert_eq!(boundary[0].cycle, 0);
        assert_eq!(boundary[1].event.sequence, 1);
        assert_eq!(boundary[1].cycle, 1);
    }

    #[test]
    fn wrapper_can_add_silence_to_the_end_of_each_loop() {
        let wrapped = format!(r#"{{"events":{SCORE},"loop_duration":1.25}}"#);
        let start = Instant::now();
        let mut score = PianoScorePlayback::from_json(&wrapped, start).unwrap();
        assert_eq!(score.take_due(start).len(), 1);
        assert_eq!(score.take_due(start + Duration::from_secs(1)).len(), 1);
        assert_eq!(
            score.take_due(start + Duration::from_millis(1_250)).len(),
            1
        );
    }
}
