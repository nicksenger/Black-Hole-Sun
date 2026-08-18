//! Loading and clocking looping piano scores.
//!
//! Scores are the compact hand-editable `bhs-score-v1` text format (see
//! [`crate::score_text`]).

use std::fs;
use std::path::Path;
use std::time::Duration;

use iced::time::Instant;

use crate::score_text::{self, BhsScore};

pub(crate) const SCORE_TICK_INTERVAL: Duration = Duration::from_millis(5);
const MAX_EVENTS_PER_TICK: usize = 4_096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DueScoreEvent {
    pub cycle: u64,
    pub event: crate::PianoEvent,
}

pub(crate) struct PianoScorePlayback {
    events: Vec<crate::PianoEvent>,
    loop_duration: Duration,
    started_at: Instant,
    event_index: usize,
    cycle: u64,
}

pub(crate) struct LoadedPianoScore {
    pub events: Vec<crate::PianoEvent>,
    pub loop_duration: Duration,
}

pub(crate) fn load_score(path: &Path) -> Result<LoadedPianoScore, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    parse_score_text(&text)
        .map_err(|error| format!("invalid piano score {}: {error}", path.display()))
}

fn parse_score_text(text: &str) -> Result<LoadedPianoScore, String> {
    let document = BhsScore::parse(text)?;
    let events = document.to_events()?;
    let loop_duration =
        score_text::ticks_to_duration(document.effective_loop_ticks(), document.ticks_per_second);
    if loop_duration.is_zero() {
        return Err("the inferred loop duration is zero; add a later event or loop_duration"
            .to_string());
    }
    Ok(LoadedPianoScore { events, loop_duration })
}

impl PianoScorePlayback {
    pub(crate) fn load(path: &Path, started_at: Instant) -> Result<Self, String> {
        Ok(Self::from_loaded(load_score(path)?, started_at))
    }

    #[cfg(test)]
    pub(crate) fn from_text(text: &str, started_at: Instant) -> Result<Self, String> {
        Ok(Self::from_loaded(parse_score_text(text)?, started_at))
    }

    fn from_loaded(score: LoadedPianoScore, started_at: Instant) -> Self {
        Self {
            events: score.events,
            loop_duration: score.loop_duration,
            started_at,
            event_index: 0,
            cycle: 0,
        }
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
    use crate::PianoAction;

    const TEXT_SCORE: &str = "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 1200
0 600 C4 80
";

    #[test]
    fn loads_and_loops_text_scores() {
        let start = Instant::now();
        let mut score = PianoScorePlayback::from_text(TEXT_SCORE, start).unwrap();

        // The attack is due immediately; the release lands at tick 600,
        // 0.625s into a 960 ticks/second grid.
        let first = score.take_due(start);
        assert_eq!(first.len(), 1);
        match &first[0].event.action {
            PianoAction::Attack { velocity, .. } => {
                assert!((velocity - f32::from(80u8) / 127.0).abs() < 1e-6);
            }
            other => panic!("expected an attack, got {other:?}"),
        }
        assert_eq!(score.take_due(start + Duration::from_millis(500)).len(), 0);
        let release = score.take_due(start + Duration::from_millis(625));
        assert_eq!(release.len(), 1);
        assert!(matches!(release[0].event.action, PianoAction::Release { .. }));

        // The loop_ticks header (1200 ticks = 1.25s) closes the cycle with
        // tail silence before the next attack.
        let wrapped = score.take_due(start + Duration::from_millis(1_250));
        assert_eq!(wrapped.len(), 1);
        assert_eq!(wrapped[0].cycle, 1);
    }

    #[test]
    fn pairs_expand_in_performance_order_regardless_of_file_order() {
        // The longer pair is listed first in the file; expansion must still
        // order events by performance time and loop cleanly.
        let score = "\
format bhs-score-v1
ticks_per_second 960
0 960 E4 64
0 480 C4 80
";
        let start = Instant::now();
        let mut playback = PianoScorePlayback::from_text(score, start).unwrap();

        // Both attacks are due immediately; at equal timestamps the lower
        // note orders first.
        let first = playback.take_due(start);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].event.note.midi_note, 60);
        assert!(matches!(
            first[0].event.action,
            PianoAction::Attack { .. }
        ));
        assert_eq!(first[1].event.note.midi_note, 64);

        // At tick 480 (0.5s) only the shorter pair's release is due.
        let mid = playback.take_due(start + Duration::from_millis(500));
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].event.note.midi_note, 60);
        assert!(matches!(
            mid[0].event.action,
            PianoAction::Release { .. }
        ));

        // At tick 960 (1.0s) the E4 release closes cycle 0 and both attacks
        // of cycle 1 are due at the same instant; the cyclic walk keeps the
        // load-time order, so the release leads.
        let boundary = playback.take_due(start + Duration::from_secs(1));
        assert_eq!(boundary.len(), 3);
        assert_eq!(boundary[0].event.note.midi_note, 64);
        assert_eq!(boundary[0].cycle, 0);
        assert!(matches!(
            boundary[0].event.action,
            PianoAction::Release { .. }
        ));
        assert_eq!(boundary[1].event.note.midi_note, 60);
        assert_eq!(boundary[1].cycle, 1);
        assert!(matches!(
            boundary[1].event.action,
            PianoAction::Attack { .. }
        ));
        assert_eq!(boundary[2].event.note.midi_note, 64);
        assert_eq!(boundary[2].cycle, 1);
        assert!(matches!(
            boundary[2].event.action,
            PianoAction::Attack { .. }
        ));
    }

    #[test]
    fn text_scores_reject_bad_documents() {
        let start = Instant::now();
        let error = PianoScorePlayback::from_text(
            "format bhs-score-v1\nticks_per_second 960\n0 960 5 80\n",
            start,
        )
        .err()
        .expect("the out-of-range note should be rejected");
        assert!(error.contains("outside the 88-key range"), "{error}");
    }

    #[test]
    fn note_names_match_the_piano_note_table() {
        // A sanity check that the score's note spelling agrees with the
        // piano's own naming for every in-range note.
        for midi_note in 21..=108u8 {
            let name = crate::PianoNote::from_midi(midi_note);
            assert_eq!(
                crate::score_text::note_name(midi_note),
                format!("{}{}", name.name(), name.octave()),
                "note {midi_note}"
            );
        }
    }
}
