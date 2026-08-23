//! The `bhs-score-v1` text format for piano scores.
//!
//! A score is a table of note pairs in integer tick space, one pair per line:
//!
//! ```text
//! ; black-hole-beam piano score
//! ; loop 279.8s | 3037 notes | range Eb1..Ab6
//! format bhs-score-v1
//! ticks_per_second 1920
//! measure_ticks 7680
//! loop_ticks 537150
//! ; start duration note velocity [release_velocity]
//! ; --- measure 1  (tick 0, t=0.00s) ---
//! 2073 580 Eb4 28
//! 2073 602 Ab2 92
//! ```
//!
//! Notes are named with scientific pitch notation (`A2`, `F#3`) or bare MIDI
//! numbers; both spellings of a pitch class (`C#`/`Db`) parse. Velocities are
//! integers in `0..=127`. A missing release velocity means the note releases
//! at its attack velocity. Lines may carry trailing comments, and full-line
//! comments are free-form; measure anchor lines such as
//! `; --- measure 12 (tick 84480, t=44.00s) ---` are generated from
//! `measure_ticks` when present. Re-emitting a score ([`BhsScore::format`])
//! is canonical: pairs sort by start tick, then note, then velocity, and user
//! comments are preserved attached to the pair that follows them.
//!
//! Files conventionally use the `.bhs` extension; loaders recognize scores by
//! content (the `format` line), not by extension.

use std::time::Duration;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::{PianoAction, PianoEvent, PianoInputSource, PianoNote};

/// The format name expected on the first structural line of a score file.
pub const FORMAT_NAME: &str = "bhs-score-v1";

/// The lowest MIDI note a score may contain (A0).
pub const FIRST_MIDI_NOTE: u8 = 21;
/// The highest MIDI note a score may contain (C8).
pub const LAST_MIDI_NOTE: u8 = 108;
/// The highest velocity a score may contain.
pub const MAX_VELOCITY: u8 = 127;
/// The decay scale, in semitones, for mutated note selection. A key's
/// selection probability falls off exponentially with its distance from the
/// keyboard center, so middle keys are favored and the outermost keys (about
/// 18 times rarer than the center) are possible but unlikely.
pub const MUTATED_NOTE_DECAY_SEMITONES: f64 = 15.0;
/// The longest a mutated or inserted note may be held, in seconds.
pub const MAX_MUTATED_DURATION_SECS: u64 = 5;
/// The shortest a mutated or inserted note may be held, in seconds. Mutated
/// durations are drawn on a natural-log scale between this floor and
/// [`MAX_MUTATED_DURATION_SECS`], so shorter notes are more likely than
/// longer ones.
pub const MIN_MUTATED_DURATION_SECS: f64 = 0.1;

const TITLE_COMMENT: &str = "; black-hole-beam piano score";
const LEGEND_COMMENT: &str = "; start duration note velocity [release_velocity]";
const ANCHOR_PREFIX: &str = "; --- measure ";

/// One sounded note: an attack at `start_tick` held for `duration_ticks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreNotePair {
    /// The tick at which the note is attacked.
    pub start_tick: u64,
    /// How long the note is held, in ticks. At least one.
    pub duration_ticks: u64,
    /// The MIDI note number (`21..=108`).
    pub midi_note: u8,
    /// The attack velocity, `0..=127`.
    pub velocity: u8,
    /// The release velocity; `None` releases at the attack velocity.
    pub release_velocity: Option<u8>,
}

impl ScoreNotePair {
    /// The tick at which the note is released.
    pub fn end_tick(&self) -> u64 {
        self.start_tick + self.duration_ticks
    }
}

/// A parsed `bhs-score-v1` score document.
#[derive(Debug, Clone)]
pub struct BhsScore {
    /// The time grid: ticks per second.
    pub ticks_per_second: u64,
    /// Presentational measure length used to emit anchor comments.
    pub measure_ticks: Option<u64>,
    /// An explicit loop length in ticks; `None` loops at the last release.
    pub loop_ticks: Option<u64>,
    pub pairs: Vec<AnnotatedPair>,
    leading_comments: Vec<String>,
    trailing_comments: Vec<String>,
    anchors: Vec<(usize, u64)>,
}

/// A pair plus the user comments that precede it in the source file.
#[derive(Debug, Clone)]
pub struct AnnotatedPair {
    pair: ScoreNotePair,
    comments: Vec<String>,
}

/// The severity of a [`BhsScore::diagnostics`] finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    /// The score will not load.
    Error,
    /// The score loads but differs from canonical form or is suspicious.
    Warning,
}

/// A finding about a document; see [`DiagnosticLevel`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    /// The 1-based source line the finding refers to, if any.
    pub line: Option<usize>,
    pub message: String,
}

impl BhsScore {
    /// Build a document from scratch for writers. No anchors or comments are
    /// remembered; [`BhsScore::format`] regenerates them.
    pub fn new(
        ticks_per_second: u64,
        measure_ticks: Option<u64>,
        loop_ticks: Option<u64>,
        pairs: Vec<ScoreNotePair>,
    ) -> Self {
        Self {
            ticks_per_second,
            measure_ticks,
            loop_ticks,
            pairs: pairs
                .into_iter()
                .map(|pair| AnnotatedPair {
                    pair,
                    comments: Vec::new(),
                })
                .collect(),
            leading_comments: Vec::new(),
            trailing_comments: Vec::new(),
            anchors: Vec::new(),
        }
    }

    /// Parse a `bhs-score-v1` document from text.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut ticks_per_second = None;
        let mut measure_ticks = None;
        let mut loop_ticks = None;
        let mut pairs: Vec<AnnotatedPair> = Vec::new();
        let mut leading_comments: Vec<String> = Vec::new();
        // User comments wait for the pair that follows them.
        let mut pending_comments: Vec<String> = Vec::new();
        let mut anchors: Vec<(usize, u64)> = Vec::new();
        let mut saw_structural_line = false;

        for (index, raw_line) in text.lines().enumerate() {
            let line = index + 1;
            let (content, trailing_comment) = split_comment(raw_line);
            let content = content.trim();
            if content.is_empty() {
                // A full-line comment (or a blank line).
                let raw_trimmed = raw_line.trim();
                if !raw_trimmed.is_empty() {
                    match classify_comment(raw_trimmed) {
                        CommentKind::Structural => {}
                        CommentKind::Anchor(measure, tick) => anchors.push((measure, tick)),
                        CommentKind::User => {
                            if pairs.is_empty() {
                                leading_comments.push(raw_trimmed.to_string());
                            } else {
                                pending_comments.push(raw_trimmed.to_string());
                            }
                        }
                    }
                }
                continue;
            }

            let mut fields = content.split_whitespace();
            let keyword = fields.next().expect("content is non-empty");
            match keyword {
                "format" => {
                    if saw_structural_line {
                        return Err(format!(
                            "line {line}: the format line must be the first structural line"
                        ));
                    }
                    let name = fields.next().ok_or_else(|| {
                        format!("line {line}: the format line needs a format name")
                    })?;
                    if name != FORMAT_NAME {
                        return Err(format!(
                            "line {line}: unknown format {name:?}; expected {FORMAT_NAME}"
                        ));
                    }
                    reject_extra_fields(fields, line, "the format line")?;
                }
                "ticks_per_second" | "measure_ticks" | "loop_ticks" => {
                    if !pairs.is_empty() {
                        return Err(format!(
                            "line {line}: the {keyword} header must precede the note pairs"
                        ));
                    }
                    let value = parse_header_value(fields, line, keyword)?;
                    match keyword {
                        "ticks_per_second" => ticks_per_second = Some(value),
                        "measure_ticks" => measure_ticks = Some(value),
                        _ => loop_ticks = Some(value),
                    }
                }
                first => {
                    if ticks_per_second.is_none() {
                        return Err(format!(
                            "line {line}: note pair before the ticks_per_second header"
                        ));
                    }
                    let pair = parse_pair_fields(first, fields, line)?;
                    let mut comments = std::mem::take(&mut pending_comments);
                    if let Some(comment) = trailing_comment {
                        comments.push(comment.to_string());
                    }
                    pairs.push(AnnotatedPair { pair, comments });
                }
            }
            saw_structural_line = true;
        }

        let ticks_per_second = ticks_per_second
            .filter(|&tps| tps > 0)
            .ok_or_else(|| "the score is missing a positive ticks_per_second header".to_string())?;
        if pairs.is_empty() {
            return Err("the score has no notes".to_string());
        }

        Ok(Self {
            ticks_per_second,
            measure_ticks,
            loop_ticks,
            pairs,
            leading_comments,
            trailing_comments: pending_comments,
            anchors,
        })
    }

    /// The note pairs in file order.
    pub fn pairs(&self) -> impl Iterator<Item = &ScoreNotePair> {
        self.pairs.iter().map(|annotated| &annotated.pair)
    }

    /// Mutable access to the note pair at `index`, in file order.
    pub fn pair_mut_at(&mut self, index: usize) -> &mut ScoreNotePair {
        &mut self.pairs[index].pair
    }

    /// Remove the note pair at `index`, in file order.
    pub fn remove_pair_at(&mut self, index: usize) {
        self.pairs.remove(index);
    }

    /// Insert a new pair immediately after the pair at `index`, in file order.
    /// The new pair carries no comments; [`BhsScore::format`] re-sorts pairs
    /// canonically on re-emit.
    pub fn insert_pair_after(&mut self, index: usize, pair: ScoreNotePair) {
        self.pairs.insert(
            index + 1,
            AnnotatedPair {
                pair,
                comments: Vec::new(),
            },
        );
    }

    /// Consume the score and wrap it with a fresh entropy-seeded RNG for
    /// random mutation; see [`MutantScore`].
    pub fn into_mutant(self) -> MutantScore {
        MutantScore {
            inner: self,
            rng: StdRng::seed_from_u64(rand::random()),
        }
    }

    /// The note pairs in canonical order: start tick, then note, then velocity.
    pub fn sorted_pairs(&self) -> Vec<&ScoreNotePair> {
        let mut sorted: Vec<&ScoreNotePair> = self.pairs.iter().map(|a| &a.pair).collect();
        sorted.sort_by(|a, b| canonical_pair_order(a, b));
        sorted
    }

    /// The tick at which the last note is released.
    pub fn last_release_tick(&self) -> u64 {
        self.pairs
            .iter()
            .map(|annotated| annotated.pair.end_tick())
            .max()
            .expect("at least one pair")
    }

    /// The loop length in ticks: the explicit header value, or the last
    /// release tick when the header is absent.
    pub fn effective_loop_ticks(&self) -> u64 {
        self.loop_ticks.unwrap_or_else(|| self.last_release_tick())
    }

    /// Transpose the score by `semitones`, moving it into a new key: every
    /// note shifts by that many semitones (positive up, negative down).
    /// Notes that would leave the 88-key range are clamped to
    /// [`FIRST_MIDI_NOTE`] or [`LAST_MIDI_NOTE`]; timing, durations, and
    /// velocities are untouched.
    pub fn transpose(&mut self, semitones: i32) {
        if semitones == 0 {
            return;
        }
        for annotated in &mut self.pairs {
            let shifted = i32::from(annotated.pair.midi_note).saturating_add(semitones);
            annotated.pair.midi_note =
                shifted.clamp(i32::from(FIRST_MIDI_NOTE), i32::from(LAST_MIDI_NOTE)) as u8;
        }
    }

    /// Consuming counterpart of [`BhsScore::transpose`]: shift every note by
    /// `semitones` and return the result, consuming the original score.
    pub fn transposed(self, semitones: i32) -> Self {
        let mut score = self;
        score.transpose(semitones);
        score
    }

    /// Skip the first `seconds` seconds of the score: drop every pair that
    /// starts before the skip point, then shift the remaining pairs (and any
    /// explicit loop length) back by the same amount so playback begins at
    /// the skip point, as if the score had jumped straight there. An implicit
    /// loop (no `loop_ticks` header) follows the last release and needs no
    /// change.
    ///
    /// Zero seconds is a no-op. Returns an error when every pair starts
    /// before the skip point, leaving no notes to play.
    pub fn skip_seconds(&mut self, seconds: u64) -> Result<(), String> {
        let skip_ticks = seconds.saturating_mul(self.ticks_per_second);
        if skip_ticks == 0 {
            return Ok(());
        }

        // Drop pairs that start before the skip point. Walk in reverse so the
        // removals do not shift the indices still to be examined.
        let mut index = self.pairs.len();
        while index > 0 {
            index -= 1;
            if self.pairs[index].pair.start_tick < skip_ticks {
                self.pairs.remove(index);
            }
        }
        if self.pairs.is_empty() {
            return Err(format!("skipping {seconds}s leaves no notes in the score"));
        }

        // Shift the survivors so the skip point becomes tick 0. Kept pairs
        // all start at or after the skip point, so this cannot underflow.
        for annotated in &mut self.pairs {
            annotated.pair.start_tick -= skip_ticks;
        }

        // The score now covers the tail of one original cycle, so shorten an
        // explicit loop length by the same amount to keep the period aligned.
        if let Some(loop_ticks) = self.loop_ticks {
            self.loop_ticks = Some(loop_ticks.saturating_sub(skip_ticks));
        }

        Ok(())
    }

    /// Rescale the score so its loop length becomes `new_seconds`: every note
    /// (and the measure grid, when present) moves to the same relative
    /// position in the new duration, as if the whole performance were
    /// stretched or compressed. The old duration is the effective loop
    /// length ([`BhsScore::effective_loop_ticks`]); the rescaled score carries
    /// an explicit `loop_ticks` header of exactly `new_seconds`, so a score
    /// with an implicit loop gains one.
    ///
    /// Tick counts are rounded to the nearest tick and every note keeps at
    /// least one tick of duration. A note whose tail rounds past the new loop
    /// end is pulled back into it so the score stays playable. Returns an
    /// error when `new_seconds` is not a positive finite number or is shorter
    /// than one tick on this grid.
    pub fn rescale_to_duration(&mut self, new_seconds: f64) -> Result<(), String> {
        if !new_seconds.is_finite() || new_seconds <= 0.0 {
            return Err(
                "the new duration must be a finite number of seconds greater than zero".to_string(),
            );
        }
        let old_loop = self.effective_loop_ticks();
        let new_loop = (new_seconds * self.ticks_per_second as f64).round() as u64;
        if new_loop < 1 {
            return Err(format!(
                "a duration of {new_seconds}s is shorter than one tick at {} ticks/second",
                self.ticks_per_second
            ));
        }

        let rescale = |ticks: u64| rescale_ticks(ticks, old_loop, new_loop);
        for annotated in &mut self.pairs {
            let pair = &mut annotated.pair;
            let start = rescale(pair.start_tick);
            let duration = rescale(pair.duration_ticks).max(1);
            // Rounding can push a note's tail a tick past the new loop end;
            // pull the whole note back so the score stays playable.
            if start.saturating_add(duration) > new_loop {
                pair.start_tick = new_loop - 1;
                pair.duration_ticks = 1;
            } else {
                pair.start_tick = start;
                pair.duration_ticks = duration;
            }
        }
        if let Some(measure_ticks) = self.measure_ticks {
            self.measure_ticks = Some(rescale(measure_ticks).max(1));
        }
        self.loop_ticks = Some(new_loop);
        Ok(())
    }

    /// Shift every note later or earlier by `delta_ticks` ticks (positive
    /// later, negative earlier), dropping any note that would start before
    /// tick 0. Durations and velocities are untouched. An explicit loop
    /// length moves with the notes so the period stays aligned; an implicit
    /// loop follows the last release and needs no change.
    ///
    /// Zero is a no-op. Returns an error when every note would start before
    /// tick 0, leaving no notes to play.
    pub fn shift_ticks(&mut self, delta_ticks: i64) -> Result<(), String> {
        if delta_ticks == 0 {
            return Ok(());
        }

        // Drop pairs that would start before tick 0 (negative shifts only).
        // Walk in reverse so the removals do not shift the indices still to
        // be examined.
        if delta_ticks < 0 {
            let drop_before = (-delta_ticks) as u64;
            let mut index = self.pairs.len();
            while index > 0 {
                index -= 1;
                if self.pairs[index].pair.start_tick < drop_before {
                    self.pairs.remove(index);
                }
            }
            if self.pairs.is_empty() {
                return Err(format!(
                    "shifting by {delta_ticks} ticks leaves no notes in the score"
                ));
            }
        }

        for annotated in &mut self.pairs {
            let pair = &mut annotated.pair;
            if delta_ticks > 0 {
                pair.start_tick += delta_ticks as u64;
            } else {
                // Surviving pairs all start at or after the shift point, so
                // this cannot underflow.
                pair.start_tick -= (-delta_ticks) as u64;
            }
        }

        // Move an explicit loop length with the notes to keep the period
        // aligned. Notes survive only when the content (and therefore the
        // loop) extends past the shift point, so this cannot underflow.
        if let Some(loop_ticks) = self.loop_ticks {
            self.loop_ticks = Some(if delta_ticks > 0 {
                loop_ticks.saturating_add(delta_ticks as u64)
            } else {
                loop_ticks - (-delta_ticks) as u64
            });
        }

        Ok(())
    }

    /// Shift every note later or earlier by `seconds` (positive later,
    /// negative earlier), dropping any note that would start before tick 0.
    /// The shift is rounded to the nearest tick on this grid; see
    /// [`BhsScore::shift_ticks`] for the details. Returns an error when
    /// `seconds` is not finite or does not fit in a tick count.
    pub fn shift_seconds(&mut self, seconds: f64) -> Result<(), String> {
        if !seconds.is_finite() {
            return Err("the shift must be a finite number of seconds".to_string());
        }
        let delta = (seconds * self.ticks_per_second as f64).round();
        // Beyond this magnitude the shift does not fit in an i64 tick count.
        if !(-9_200_000_000_000_000_000.0..=9_200_000_000_000_000_000.0).contains(&delta) {
            return Err(format!("a shift of {seconds}s overflows the tick count"));
        }
        self.shift_ticks(delta as i64)
    }

    /// Merge `other` into this score so both play simultaneously.
    ///
    /// The result keeps this score's grid ([`BhsScore::ticks_per_second`]);
    /// `other`'s pairs are rescaled onto it, each tick count rounded to the
    /// nearest tick and every note kept at least one tick long. Comments
    /// attached to `other`'s pairs carry over; its leading comments do not.
    /// The merged loop length is the longer of the two effective loop
    /// lengths (after rescaling), so scores that already share a loop length
    /// keep playing in lockstep; when they differ, the shorter score's notes
    /// sound once per merged cycle rather than repeating within it.
    pub fn merge_with(mut self, other: BhsScore) -> Self {
        let this_tps = self.ticks_per_second;
        let other_tps = other.ticks_per_second;
        let rescale = |ticks: u64| rescale_ticks(ticks, other_tps, this_tps);

        let this_loop = self.effective_loop_ticks();
        let other_loop = rescale(other.effective_loop_ticks());

        for annotated in other.pairs {
            self.pairs.push(AnnotatedPair {
                pair: ScoreNotePair {
                    start_tick: rescale(annotated.pair.start_tick),
                    duration_ticks: rescale(annotated.pair.duration_ticks).max(1),
                    midi_note: annotated.pair.midi_note,
                    velocity: annotated.pair.velocity,
                    release_velocity: annotated.pair.release_velocity,
                },
                comments: annotated.comments,
            });
        }
        self.loop_ticks = Some(this_loop.max(other_loop));
        self
    }

    /// Expand the pairs into score events, ready for playback: attacks and
    /// releases in performance order with dense sequence numbers and voice ids
    /// assigned in canonical pair order.
    pub fn to_events(&self) -> Result<Vec<PianoEvent>, String> {
        let ticks_per_second = self.ticks_per_second;
        if let Some(loop_ticks) = self.loop_ticks {
            let last_release = self.last_release_tick();
            if loop_ticks < last_release {
                return Err(format!(
                    "loop_ticks ({loop_ticks}) precedes the last release (tick {last_release})"
                ));
            }
        }

        let mut events = Vec::with_capacity(self.pairs.len() * 2);
        for (index, pair) in self.sorted_pairs().into_iter().enumerate() {
            // Voice ids follow the canonical pair order.
            let voice_id = index as u64 + 1;
            let note = PianoNote::from_midi(pair.midi_note);
            let attack_velocity = f32::from(pair.velocity) / f32::from(MAX_VELOCITY);
            let release_velocity =
                f32::from(pair.release_velocity.unwrap_or(pair.velocity)) / f32::from(MAX_VELOCITY);
            events.push(PianoEvent {
                sequence: 0,
                timestamp: ticks_to_duration(pair.start_tick, ticks_per_second),
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
                timestamp: ticks_to_duration(pair.end_tick(), ticks_per_second),
                voice_id,
                note,
                action: PianoAction::Release {
                    velocity: release_velocity,
                    held_for: ticks_to_duration(pair.duration_ticks, ticks_per_second),
                },
                source: PianoInputSource::Score,
            });
        }

        events.sort_by(|a, b| {
            a.timestamp
                .cmp(&b.timestamp)
                .then_with(|| action_rank(&a.action).cmp(&action_rank(&b.action)))
                .then_with(|| a.note.midi_note.cmp(&b.note.midi_note))
        });
        for (index, event) in events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
        }

        Ok(events)
    }

    /// Re-emit the document in canonical form, preserving user comments.
    pub fn format(&self) -> String {
        let mut out = String::new();
        for comment in &self.leading_comments {
            out.push_str(comment);
            out.push('\n');
        }
        out.push_str(TITLE_COMMENT);
        out.push('\n');

        let annotated_pairs = self.sorted_pairs_with_comments();
        let loop_ticks = self.effective_loop_ticks();
        let loop_seconds = loop_ticks as f64 / self.ticks_per_second as f64;
        let lowest = annotated_pairs
            .iter()
            .map(|(pair, _)| pair.midi_note)
            .min()
            .expect("at least one pair");
        let highest = annotated_pairs
            .iter()
            .map(|(pair, _)| pair.midi_note)
            .max()
            .expect("at least one pair");
        out.push_str(&format!(
            "; loop {:.1}s | {} notes | range {}..{}\n",
            loop_seconds,
            annotated_pairs.len(),
            note_name(lowest),
            note_name(highest)
        ));

        out.push_str(&format!("format {FORMAT_NAME}\n"));
        out.push_str(&format!("ticks_per_second {}\n", self.ticks_per_second));
        if let Some(measure_ticks) = self.measure_ticks {
            out.push_str(&format!("measure_ticks {measure_ticks}\n"));
        }
        out.push_str(&format!("loop_ticks {loop_ticks}\n"));
        out.push_str(LEGEND_COMMENT);
        out.push('\n');

        let mut current_measure = 0usize;
        for (pair, comments) in &annotated_pairs {
            if let Some(measure_ticks) = self.measure_ticks {
                let measure = (pair.start_tick / measure_ticks) as usize + 1;
                if measure != current_measure {
                    current_measure = measure;
                    let start_tick = (measure - 1) as u64 * measure_ticks;
                    out.push_str(&format!(
                        "{ANCHOR_PREFIX}{measure}  (tick {start_tick}, t={:.2}s) ---\n",
                        start_tick as f64 / self.ticks_per_second as f64
                    ));
                }
            }
            for comment in comments.iter() {
                out.push_str(comment);
                out.push('\n');
            }
            let mut line = format!(
                "{} {} {} {}",
                pair.start_tick,
                pair.duration_ticks,
                note_name(pair.midi_note),
                pair.velocity
            );
            if let Some(release_velocity) = pair.release_velocity {
                line.push_str(&format!(" {release_velocity}"));
            }
            out.push_str(&line);
            out.push('\n');
        }
        for comment in &self.trailing_comments {
            out.push_str(comment);
            out.push('\n');
        }

        out
    }

    /// Findings about the document: duplicate-pair and canonical-order
    /// warnings, anchor drift when `measure_ticks` is set, and loop-length
    /// errors. An empty result means the document is already canonical.
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        let mut findings = Vec::new();

        let sorted = self.sorted_pairs();
        for window in sorted.windows(2) {
            if window[0] == window[1] {
                findings.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    line: None,
                    message: format!(
                        "duplicate note pair: {} at tick {} for {} ticks",
                        note_name(window[0].midi_note),
                        window[0].start_tick,
                        window[0].duration_ticks
                    ),
                });
            }
        }

        let in_canonical_order = self
            .pairs()
            .zip(sorted.iter())
            .all(|(file_order, sorted_order)| file_order == *sorted_order);
        if !in_canonical_order {
            findings.push(Diagnostic {
                level: DiagnosticLevel::Warning,
                line: None,
                message: "note pairs are out of canonical order (start tick, note, velocity); \
                          re-emit to sort"
                    .to_string(),
            });
        }

        if let Some(measure_ticks) = self.measure_ticks {
            let mut expected: Vec<(usize, u64)> = Vec::new();
            let mut current_measure = 0usize;
            for pair in &sorted {
                let measure = (pair.start_tick / measure_ticks) as usize + 1;
                if measure != current_measure {
                    current_measure = measure;
                    expected.push((measure, (measure - 1) as u64 * measure_ticks));
                }
            }
            if self.anchors != expected {
                findings.push(Diagnostic {
                    level: DiagnosticLevel::Warning,
                    line: None,
                    message: format!(
                        "measure anchors are stale ({} in file, {} expected); re-emit to refresh",
                        self.anchors.len(),
                        expected.len()
                    ),
                });
            }
        }

        if let Some(loop_ticks) = self.loop_ticks {
            let last_release = self.last_release_tick();
            if loop_ticks < last_release {
                findings.push(Diagnostic {
                    level: DiagnosticLevel::Error,
                    line: None,
                    message: format!(
                        "loop_ticks ({loop_ticks}) precedes the last release (tick {last_release})"
                    ),
                });
            }
        }

        findings
    }

    /// Sorted pairs carrying their attached comments, in canonical order.
    fn sorted_pairs_with_comments(&self) -> Vec<(&ScoreNotePair, &Vec<String>)> {
        let mut annotated: Vec<(&ScoreNotePair, &Vec<String>)> =
            self.pairs.iter().map(|a| (&a.pair, &a.comments)).collect();
        annotated.sort_by(|(a, _), (b, _)| canonical_pair_order(a, b));
        annotated
    }
}

/// A score paired with the RNG that draws its mutations.
///
/// Obtain one by consuming a [`BhsScore`] via [`BhsScore::into_mutant`], or
/// construct it directly with a seeded generator (any [`RngExt`]) for
/// deterministic mutations. Pairs may no longer be in canonical order after
/// mutation; re-emitting via [`BhsScore::format`] normalizes them.
pub struct MutantScore<R = StdRng> {
    /// The score being mutated.
    pub inner: BhsScore,
    /// The generator drawing the mutations.
    pub rng: R,
}

impl<R: RngExt> MutantScore<R> {
    /// Keep at most the first `len` pairs in file order, dropping the rest,
    /// and clear any explicit loop length so the score loops at the last
    /// release of what remains.
    pub fn truncate_pairs(&mut self, len: usize) {
        self.inner.pairs.truncate(len);
        self.inner.loop_ticks = None;
    }

    /// Mutate the score in place by an amount proportional to `noise`, a
    /// normalized scalar in `0.0..=1.0`.
    ///
    /// Applies `M = (pairs.len() as f32 * noise) as usize` mutations, each a
    /// uniform draw from [`MutantScore::delete_random_pair`],
    /// [`MutantScore::substitute_random_pair`], and
    /// [`MutantScore::insert_after_random_pair`]. Mutation stops early once
    /// the score is empty; returns the number of mutations applied.
    pub fn mutate(&mut self, noise: f32) -> usize {
        let mutations = (self.inner.pairs.len() as f32 * noise.clamp(0.0, 1.0)) as usize;
        let mut applied = 0;
        for _ in 0..mutations {
            if self.inner.pairs.is_empty() {
                // Nothing left to mutate (an empty score is not a valid
                // bhs-score-v1 document anyway).
                break;
            }
            match self.rng.random_range(0..3) {
                0 => {
                    self.delete_random_pair();
                }
                1 => {
                    self.substitute_random_pair();
                }
                _ => {
                    self.insert_after_random_pair();
                }
            };
            applied += 1;
        }
        applied
    }

    /// Remove a random pair from the score and return it. Returns `None`
    /// without mutating if the score has no pairs.
    pub fn delete_random_pair(&mut self) -> Option<ScoreNotePair> {
        if self.inner.pairs.is_empty() {
            return None;
        }
        let index = self.rng.random_range(0..self.inner.pairs.len());
        Some(self.inner.pairs.remove(index).pair)
    }

    /// Give a random pair a new center-biased random 88-key note (see
    /// [`MUTATED_NOTE_DECAY_SEMITONES`]), a log-scaled random
    /// duration between [`MIN_MUTATED_DURATION_SECS`] and
    /// [`MAX_MUTATED_DURATION_SECS`] seconds (shorter is more likely), and a
    /// random velocity, keeping its start tick and release velocity. Returns
    /// the previous values; `None` without mutating if the score has no pairs.
    pub fn substitute_random_pair(&mut self) -> Option<ScoreNotePair> {
        if self.inner.pairs.is_empty() {
            return None;
        }
        let index = self.rng.random_range(0..self.inner.pairs.len());
        let previous = self.inner.pairs[index].pair;
        let pair = &mut self.inner.pairs[index].pair;
        pair.midi_note = random_note(&mut self.rng);
        pair.duration_ticks = random_duration_ticks(&mut self.rng, self.inner.ticks_per_second);
        pair.velocity = random_velocity(&mut self.rng);
        Some(previous)
    }

    /// Insert a new pair immediately after a random existing pair and return
    /// the new pair's index in file order. The new pair starts at a random
    /// tick between the picked pair's start tick and the following pair's
    /// start tick (or the end of the loop for the last pair), with a
    /// center-biased random note (see [`MUTATED_NOTE_DECAY_SEMITONES`]), a
    /// log-scaled random duration between
    /// [`MIN_MUTATED_DURATION_SECS`] and [`MAX_MUTATED_DURATION_SECS`] seconds
    /// (shorter is more likely), and a random velocity. It carries no
    /// comments; returns `None` without mutating if the score has no pairs.
    pub fn insert_after_random_pair(&mut self) -> Option<usize> {
        if self.inner.pairs.is_empty() {
            return None;
        }
        let index = self.rng.random_range(0..self.inner.pairs.len());
        // The following pair in file order bounds the window; for a score
        // that is still in canonical (start-tick) order that is the picked
        // pair's temporal successor. The last pair's window runs to the end
        // of the loop.
        let chosen = self.inner.pairs[index].pair;
        let window_end = self
            .inner
            .pairs
            .get(index + 1)
            .map(|next| next.pair.start_tick)
            .unwrap_or_else(|| self.inner.effective_loop_ticks())
            .max(chosen.start_tick);
        let pair = ScoreNotePair {
            start_tick: self.rng.random_range(chosen.start_tick..=window_end),
            duration_ticks: random_duration_ticks(&mut self.rng, self.inner.ticks_per_second),
            midi_note: random_note(&mut self.rng),
            velocity: random_velocity(&mut self.rng),
            release_velocity: None,
        };
        let inserted_at = index + 1;
        self.inner.pairs.insert(
            inserted_at,
            AnnotatedPair {
                pair,
                comments: Vec::new(),
            },
        );
        Some(inserted_at)
    }
}

/// A random 88-key MIDI note, weighted so selection probability decays
/// exponentially with distance from the keyboard center (see
/// [`MUTATED_NOTE_DECAY_SEMITONES`]): middle keys are most likely, and the
/// first and last keys are possible but unlikely.
fn random_note(rng: &mut impl RngExt) -> u8 {
    let center = (FIRST_MIDI_NOTE as f64 + LAST_MIDI_NOTE as f64) / 2.0;
    let weight = |note: u8| {
        (-((note as f64 - center).abs() / MUTATED_NOTE_DECAY_SEMITONES)).exp()
    };
    // Walk the cumulative weights; the range is fixed and tiny, so a linear
    // scan is simpler (and faster) than precomputing a table.
    let total: f64 = (FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).map(weight).sum();
    let mut remaining = rng.random::<f64>() * total;
    for note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
        remaining -= weight(note);
        if remaining <= 0.0 {
            return note;
        }
    }
    LAST_MIDI_NOTE
}

/// A random velocity in `0..=MAX_VELOCITY`.
fn random_velocity(rng: &mut impl RngExt) -> u8 {
    rng.random_range(0..=MAX_VELOCITY)
}

/// A random duration between [`MIN_MUTATED_DURATION_SECS`] and
/// [`MAX_MUTATED_DURATION_SECS`] seconds, drawn uniformly on a natural-log
/// scale so shorter durations are more likely and longer ones far less
/// likely (every doubling of length is equally probable). The result is at
/// least one tick so the pair stays a valid note.
fn random_duration_ticks(rng: &mut impl RngExt, ticks_per_second: u64) -> u64 {
    let min_secs = MIN_MUTATED_DURATION_SECS;
    let max_secs = MAX_MUTATED_DURATION_SECS as f64;
    // Uniform in ln(seconds), exponentiated back to seconds.
    let seconds = min_secs * (rng.random::<f64>() * (max_secs / min_secs).ln()).exp();
    ((seconds * ticks_per_second as f64).round() as u64).max(1)
}

/// Split a line at its first `;` into content and trailing comment.
fn split_comment(line: &str) -> (&str, Option<&str>) {
    match line.find(';') {
        Some(position) => (&line[..position], Some(&line[position..])),
        None => (line, None),
    }
}

enum CommentKind {
    /// A generated header comment (title, stats, legend); regenerated on emit.
    Structural,
    /// A generated measure anchor, remembered for drift checks.
    Anchor(usize, u64),
    /// Anything else; preserved on re-emit.
    User,
}

/// Classify a full-line comment so re-emission can regenerate the structural
/// ones and preserve the rest.
fn classify_comment(line: &str) -> CommentKind {
    if line == TITLE_COMMENT || line.starts_with("; loop ") || line == LEGEND_COMMENT {
        return CommentKind::Structural;
    }
    if let Some(rest) = line.strip_prefix(ANCHOR_PREFIX) {
        let mut parts = rest.split_whitespace();
        if let Some(measure_text) = parts.next() {
            if let Ok(measure) = measure_text.parse::<usize>() {
                for token in parts {
                    if let Some(tick_text) = token.strip_suffix(',') {
                        if let Ok(tick) = tick_text.parse::<u64>() {
                            return CommentKind::Anchor(measure, tick);
                        }
                    }
                }
            }
        }
    }
    CommentKind::User
}

/// Parse the single numeric field of a header keyword line.
fn parse_header_value<'a>(
    mut fields: impl Iterator<Item = &'a str>,
    line: usize,
    keyword: &str,
) -> Result<u64, String> {
    let text = fields
        .next()
        .ok_or_else(|| format!("line {line}: the {keyword} header needs a value"))?;
    let value: u64 = text.parse().map_err(|_| {
        format!("line {line}: the {keyword} value {text:?} is not a non-negative integer")
    })?;
    reject_extra_fields(fields, line, &format!("the {keyword} header"))?;
    Ok(value)
}

fn reject_extra_fields<'a>(
    mut fields: impl Iterator<Item = &'a str>,
    line: usize,
    what: &str,
) -> Result<(), String> {
    if fields.next().is_some() {
        return Err(format!("line {line}: {what} has extra fields"));
    }
    Ok(())
}

/// Parse `start duration note velocity [release_velocity]` fields.
fn parse_pair_fields<'a>(
    first: &str,
    mut fields: impl Iterator<Item = &'a str>,
    line: usize,
) -> Result<ScoreNotePair, String> {
    let start_tick = parse_u64(first, line, "start tick")?;
    let duration_ticks = fields
        .next()
        .map(|text| parse_u64(text, line, "duration"))
        .transpose()?
        .ok_or_else(|| format!("line {line}: the note pair needs a duration"))?;
    if duration_ticks == 0 {
        return Err(format!(
            "line {line}: the duration must be at least one tick"
        ));
    }
    let note_text = fields
        .next()
        .ok_or_else(|| format!("line {line}: the note pair needs a note name"))?;
    let midi_note = parse_note(note_text).map_err(|message| format!("line {line}: {message}"))?;
    let velocity_text = fields
        .next()
        .ok_or_else(|| format!("line {line}: the note pair needs a velocity"))?;
    let velocity = parse_velocity(velocity_text, line, "velocity")?;
    let release_velocity = match fields.next() {
        Some(text) => Some(parse_velocity(text, line, "release velocity")?),
        None => None,
    };
    reject_extra_fields(fields, line, "the note pair")?;

    Ok(ScoreNotePair {
        start_tick,
        duration_ticks,
        midi_note,
        velocity,
        release_velocity,
    })
}

fn parse_u64(text: &str, line: usize, what: &str) -> Result<u64, String> {
    text.parse::<u64>()
        .map_err(|_| format!("line {line}: the {what} {text:?} is not a non-negative integer"))
}

fn parse_velocity(text: &str, line: usize, what: &str) -> Result<u8, String> {
    let value: u32 = text
        .parse()
        .map_err(|_| format!("line {line}: the {what} {text:?} is not an integer"))?;
    if value > u32::from(MAX_VELOCITY) {
        return Err(format!(
            "line {line}: the {what} {value} is outside 0..={MAX_VELOCITY}"
        ));
    }
    Ok(value as u8)
}

/// Parse a note as a MIDI number or scientific pitch notation.
fn parse_note(text: &str) -> Result<u8, String> {
    if let Ok(number) = text.parse::<u8>() {
        return validate_midi_note(number);
    }
    let upper = text.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    if bytes.len() < 2 || bytes.len() > 3 {
        return Err(format!(
            "note {text:?} is not a MIDI number or a note name like C4 or F#3"
        ));
    }
    let spelling = &upper[..bytes.len() - 1];
    let octave: u8 = upper[bytes.len() - 1..]
        .parse()
        .map_err(|_| format!("note {text:?} has no octave digit (try C4 or F#3)"))?;
    if octave > 8 {
        return Err(format!(
            "note {text:?} is outside the 88-key range {FIRST_MIDI_NOTE}..={LAST_MIDI_NOTE}"
        ));
    }
    let pitch_class = PITCH_CLASS_SPELLINGS
        .iter()
        .find(|(_pc, spellings)| {
            spellings
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(spelling))
        })
        .map(|(pitch_class, _)| *pitch_class)
        .ok_or_else(|| {
            format!(
                "note {text:?} is not a note name like C4 or F#3 (try A-G with an optional # or b)"
            )
        })?;
    validate_midi_note(octave * 12 + pitch_class + 12)
}

fn validate_midi_note(midi_note: u8) -> Result<u8, String> {
    if !(FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&midi_note) {
        return Err(format!(
            "MIDI note {midi_note} is outside the 88-key range {FIRST_MIDI_NOTE}..={LAST_MIDI_NOTE}"
        ));
    }
    Ok(midi_note)
}

/// Accepted spellings per pitch class; the first spelling is canonical and
/// matches [`PianoNote::name`]. Parsing compares case-insensitively.
const PITCH_CLASS_SPELLINGS: [(u8, &[&str]); 12] = [
    (0, &["C"]),
    (1, &["C#", "Db"]),
    (2, &["D"]),
    (3, &["Eb", "D#"]),
    (4, &["E"]),
    (5, &["F"]),
    (6, &["F#", "Gb"]),
    (7, &["G"]),
    (8, &["Ab"]),
    (9, &["A"]),
    (10, &["Bb", "A#"]),
    (11, &["B"]),
];

/// The canonical name of a MIDI note, such as `C4` or `F#3`.
pub fn note_name(midi_note: u8) -> String {
    format!(
        "{}{}",
        PITCH_CLASS_SPELLINGS[usize::from(midi_note % 12)].1[0],
        i32::from(midi_note / 12) - 1
    )
}

/// Convert `ticks` from a grid of `from_tps` ticks per second onto a grid of
/// `to_tps`, rounding to the nearest tick on the target grid.
fn rescale_ticks(ticks: u64, from_tps: u64, to_tps: u64) -> u64 {
    let numerator = u128::from(ticks) * u128::from(to_tps);
    let denominator = u128::from(from_tps);
    let rounded = (numerator + denominator / 2) / denominator;
    u64::try_from(rounded).unwrap_or(u64::MAX)
}

/// Convert integer ticks on a ticks-per-second grid to an exact duration.
pub fn ticks_to_duration(ticks: u64, ticks_per_second: u64) -> Duration {
    let seconds = ticks / ticks_per_second;
    let remainder_nanos =
        (u128::from(ticks % ticks_per_second) * 1_000_000_000) / u128::from(ticks_per_second);
    Duration::new(seconds, remainder_nanos as u32)
}

fn canonical_pair_order(a: &ScoreNotePair, b: &ScoreNotePair) -> std::cmp::Ordering {
    canonical_pair_order_key(a).cmp(&canonical_pair_order_key(b))
}

fn canonical_pair_order_key(pair: &ScoreNotePair) -> (u64, u8, u8) {
    (pair.start_tick, pair.midi_note, pair.velocity)
}

fn action_rank(action: &PianoAction) -> u8 {
    match action {
        PianoAction::Attack { .. } => 0,
        PianoAction::Release { .. } => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const SCORE: &str = "\
; written by hand
format bhs-score-v1
ticks_per_second 960
measure_ticks 3840
loop_ticks 7680
; start duration note velocity [release_velocity]
; --- measure 1  (tick 0, t=0.00s) ---
0 960 C4 80
0 960 A2 127
960 480 E4 64 ; an annotation on a data line
1440 960 F#4 50 30
; --- measure 2  (tick 3840, t=4.00s) ---
3840 1920 Gb4 70
";

    #[test]
    fn parses_names_numbers_and_comments() {
        let score = BhsScore::parse(SCORE).expect("the fixture should parse");
        assert_eq!(score.ticks_per_second, 960);
        assert_eq!(score.measure_ticks, Some(3840));
        assert_eq!(score.loop_ticks, Some(7680));

        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(pairs.len(), 5);
        assert_eq!(
            pairs[0],
            ScoreNotePair {
                start_tick: 0,
                duration_ticks: 960,
                midi_note: 60,
                velocity: 80,
                release_velocity: None,
            }
        );
        // Enharmonic spellings parse to the same pitch.
        assert_eq!(pairs[4].midi_note, 66);
        assert_eq!(pairs[3].release_velocity, Some(30));
        assert_eq!(
            score.leading_comments,
            vec!["; written by hand".to_string()]
        );
    }

    #[test]
    fn rejects_malformed_documents() {
        let cases = [
            ("garbage", "note pair before the ticks_per_second header"),
            (
                "format bhs-score-v2\nticks_per_second 960\n0 960 C4 80\n",
                "unknown format",
            ),
            (
                "format bhs-score-v1\n0 960 C4 80\n",
                "note pair before the ticks_per_second header",
            ),
            (
                "format bhs-score-v1\nticks_per_second 960\n",
                "the score has no notes",
            ),
            (
                "format bhs-score-v1\nticks_per_second 960\n0 0 C4 80\n",
                "at least one tick",
            ),
            (
                "format bhs-score-v1\nticks_per_second 960\n0 960 5 80\n",
                "outside the 88-key range",
            ),
            (
                "format bhs-score-v1\nticks_per_second 960\n0 960 C4 128\n",
                "outside 0..=127",
            ),
            (
                "format bhs-score-v1\nticks_per_second 960\n0 960 C4 80 10 7\n",
                "the note pair has extra fields",
            ),
        ];
        for (text, expected) in cases {
            let error = BhsScore::parse(text).expect_err("should reject");
            assert!(
                error.contains(expected),
                "for {text:?}: {error} does not mention {expected:?}"
            );
        }
    }

    #[test]
    fn expands_pairs_into_ordered_events() {
        let score = BhsScore::parse(SCORE).expect("the fixture should parse");
        let events = score.to_events().expect("valid score");
        assert_eq!(events.len(), 10);

        // Voice ids follow canonical pair order (start, note, velocity): the
        // tick-0 A2 attack is voice 1 and the tick-0 C4 attack is voice 2.
        let c4_attack = events
            .iter()
            .find(|event| {
                event.note.midi_note == 60 && matches!(event.action, PianoAction::Attack { .. })
            })
            .expect("the C4 attack should exist");
        assert_eq!(c4_attack.voice_id, 2);
        match &c4_attack.action {
            PianoAction::Attack { velocity, .. } => {
                assert!((velocity - f32::from(80u8) / 127.0).abs() < 1e-6);
            }
            other => panic!("expected an attack, got {other:?}"),
        }

        // The C4 release lands at tick 960 (one second on a 960 ticks/second
        // grid) with the attack's velocity.
        let c4_release = events
            .iter()
            .find(|event| {
                event.note.midi_note == 60 && matches!(event.action, PianoAction::Release { .. })
            })
            .expect("the C4 release should exist");
        assert_eq!(c4_release.voice_id, 2);
        assert_eq!(c4_release.timestamp, Duration::from_secs(1));
        match &c4_release.action {
            PianoAction::Release { velocity, held_for } => {
                assert!((velocity - f32::from(80u8) / 127.0).abs() < 1e-6);
                assert_eq!(*held_for, Duration::from_secs(1));
            }
            other => panic!("expected a release, got {other:?}"),
        }

        // At equal timestamps attacks precede releases (the E4 attack and the
        // two tick-960 releases share a timestamp).
        let e4_attack = events
            .iter()
            .position(|event| {
                event.note.midi_note == 64 && matches!(event.action, PianoAction::Attack { .. })
            })
            .expect("the E4 attack should exist");
        let a2_release = events
            .iter()
            .position(|event| {
                event.note.midi_note == 45 && matches!(event.action, PianoAction::Release { .. })
            })
            .expect("the A2 release should exist");
        assert!(e4_attack < a2_release);

        // Sequence numbers are dense and timestamps non-decreasing.
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            (1..=10).collect::<Vec<_>>()
        );
        assert!(events
            .windows(2)
            .all(|window| window[0].timestamp <= window[1].timestamp));
    }

    #[test]
    fn format_is_canonical_and_idempotent() {
        let score = BhsScore::parse(SCORE).expect("the fixture should parse");
        let once = score.format();
        let twice = BhsScore::parse(&once)
            .expect("canonical form should re-parse")
            .format();
        assert_eq!(once, twice);

        // User comments survive; structural anchors regenerate.
        assert!(once.contains("; written by hand"));
        assert!(once.contains("; an annotation on a data line"));
        assert!(once.contains("; --- measure 2  (tick 3840, t=4.00s) ---"));

        // The explicit release velocity round-trips.
        assert!(once.contains("1440 960 F#4 50 30"));
    }

    #[test]
    fn unsorted_input_normalizes_on_reemit() {
        let shuffled = "\
format bhs-score-v1
ticks_per_second 960
960 480 E4 64
0 960 C4 80
";
        let score = BhsScore::parse(shuffled).expect("order is not required to parse");
        let emitted = score.format();
        let lines: Vec<&str> = emitted.lines().collect();
        let c4 = lines
            .iter()
            .position(|line| line.starts_with("0 960 C4"))
            .unwrap();
        let e4 = lines
            .iter()
            .position(|line| line.starts_with("960 480 E4"))
            .unwrap();
        assert!(c4 < e4, "canonical order sorts by start tick:\n{emitted}");

        let findings = score.diagnostics();
        assert!(findings
            .iter()
            .any(|finding| finding.message.contains("canonical order")));
    }

    #[test]
    fn diagnostics_flag_duplicates_and_stale_anchors() {
        let duplicated = "\
format bhs-score-v1
ticks_per_second 960
measure_ticks 3840
0 960 C4 80
0 960 C4 80
";
        let score = BhsScore::parse(duplicated).expect("duplicates parse");
        let findings = score.diagnostics();
        assert!(findings
            .iter()
            .any(|finding| finding.message.contains("duplicate note pair")));
        // The file has no anchors while measure_ticks expects one.
        assert!(findings
            .iter()
            .any(|finding| finding.message.contains("anchors are stale")));

        let truncated_loop = "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 500
0 960 C4 80
";
        let score = BhsScore::parse(truncated_loop).expect("parses; loading should fail");
        assert!(score.to_events().is_err());
        assert!(score
            .diagnostics()
            .iter()
            .any(|finding| finding.level == DiagnosticLevel::Error));
    }

    #[test]
    fn note_names_match_the_piano_note_table() {
        for midi_note in FIRST_MIDI_NOTE..=LAST_MIDI_NOTE {
            let expected = PianoNote::from_midi(midi_note);
            assert_eq!(
                note_name(midi_note),
                format!("{}{}", expected.name(), expected.octave())
            );
        }
    }

    fn fixture_mutant(seed: u64) -> MutantScore {
        let inner = BhsScore::parse(SCORE).expect("the fixture should parse");
        MutantScore {
            inner,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    #[test]
    fn truncating_pairs_clears_the_loop_length() {
        let mut mutant = fixture_mutant(0);
        assert_eq!(mutant.inner.loop_ticks, Some(7680));
        mutant.truncate_pairs(2);
        assert_eq!(mutant.inner.pairs().count(), 2);
        assert_eq!(mutant.inner.loop_ticks, None);
        // The loop now falls back to the last release of the remaining pairs.
        assert_eq!(mutant.inner.effective_loop_ticks(), 960);

        // Truncating past the end keeps every pair but still clears the loop.
        let mut mutant = fixture_mutant(0);
        mutant.truncate_pairs(100);
        assert_eq!(mutant.inner.pairs().count(), 5);
        assert_eq!(mutant.inner.loop_ticks, None);
    }

    #[test]
    fn transpose_shifts_notes_and_clamps_to_the_88_key_range() {
        let mut score = BhsScore::parse(SCORE).expect("the fixture should parse");
        // The fixture's notes, in file order: C4 A2 E4 F#4 Gb4.
        let before: Vec<ScoreNotePair> = score.pairs().copied().collect();
        score.transpose(2);
        let after: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            after.iter().map(|pair| pair.midi_note).collect::<Vec<_>>(),
            vec![62, 47, 66, 68, 68]
        );
        // Only the notes move; timing and velocities are untouched.
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(new.start_tick, old.start_tick);
            assert_eq!(new.duration_ticks, old.duration_ticks);
            assert_eq!(new.velocity, old.velocity);
            assert_eq!(new.release_velocity, old.release_velocity);
        }

        // Zero is a no-op.
        let mut score = BhsScore::parse(SCORE).expect("the fixture should parse");
        let before: Vec<ScoreNotePair> = score.pairs().copied().collect();
        score.transpose(0);
        assert_eq!(score.pairs().copied().collect::<Vec<_>>(), before);

        // Large shifts clamp to the 88-key range instead of overflowing.
        let mut score = BhsScore::parse(SCORE).expect("the fixture should parse");
        score.transpose(-1000);
        assert!(score.pairs().all(|pair| pair.midi_note == FIRST_MIDI_NOTE));
        let mut score = BhsScore::parse(SCORE).expect("the fixture should parse");
        score.transpose(1000);
        assert!(score.pairs().all(|pair| pair.midi_note == LAST_MIDI_NOTE));

        // The consuming variant shifts the same way.
        let score = BhsScore::parse(SCORE).expect("the fixture should parse");
        let shifted = score.transposed(2);
        assert_eq!(
            shifted
                .pairs()
                .map(|pair| pair.midi_note)
                .collect::<Vec<_>>(),
            vec![62, 47, 66, 68, 68]
        );
    }

    #[test]
    fn skipping_drops_early_pairs_and_shifts_the_rest_back() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 960 C4 80
3840 960 E4 64
5760 960 G4 50
",
        )
        .expect("the fixture should parse");

        // Skip four seconds (3840 ticks): the tick-0 C4 is dropped, the pair
        // that starts exactly at the skip point survives at tick 0, and the
        // rest shift back by the same amount.
        score.skip_seconds(4).expect("the skip should succeed");
        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 0,
                    duration_ticks: 960,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 1920,
                    duration_ticks: 960,
                    midi_note: 67,
                    velocity: 50,
                    release_velocity: None,
                },
            ]
        );
        // The explicit loop shortens by the skip so the period stays aligned.
        assert_eq!(score.loop_ticks, Some(3840));

        // The shifted score plays from tick 0: the first attack is due
        // immediately and the last release closes the shortened loop.
        let events = score.to_events().expect("the skipped score should play");
        assert_eq!(events[0].timestamp, Duration::ZERO);
        assert_eq!(score.last_release_tick(), 2880);
    }

    #[test]
    fn skipping_an_implicit_loop_needs_no_loop_change() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 960 C4 80
3840 960 E4 64
",
        )
        .expect("the fixture should parse");
        assert_eq!(score.loop_ticks, None);

        score.skip_seconds(4).expect("the skip should succeed");
        assert_eq!(score.loop_ticks, None, "an implicit loop is untouched");
        // The loop now follows the last release of what remains.
        assert_eq!(score.effective_loop_ticks(), 960);
    }

    #[test]
    fn skipping_past_every_note_is_an_error() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 960 C4 80
",
        )
        .expect("the fixture should parse");
        let error = score.skip_seconds(1).expect_err("no notes remain");
        assert!(error.contains("leaves no notes"), "{error}");

        // Zero is a no-op, even for an empty-after-skip score.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 960 C4 80
",
        )
        .expect("the fixture should parse");
        let before: Vec<ScoreNotePair> = score.pairs().copied().collect();
        score.skip_seconds(0).expect("zero is a no-op");
        assert_eq!(score.pairs().copied().collect::<Vec<_>>(), before);
    }

    #[test]
    fn skipping_beyond_the_content_is_an_error() {
        // The skip walks the static pair table, not the infinite loop: a
        // skip past the last pair's start leaves no pairs and is rejected.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 1920
480 480 C4 80
",
        )
        .expect("the fixture should parse");
        let error = score.skip_seconds(3).expect_err("no pairs start after the skip point");
        assert!(error.contains("leaves no notes"), "{error}");
    }

    fn rescale_fixture() -> BhsScore {
        BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
measure_ticks 3840
loop_ticks 7680
0 960 C4 80
3840 960 E4 64
5760 960 G4 50
",
        )
        .expect("the fixture should parse")
    }

    #[test]
    fn rescaling_stretches_notes_to_the_same_relative_positions() {
        let mut score = rescale_fixture();
        // The loop is eight seconds; double it to sixteen.
        score.rescale_to_duration(16.0).expect("the rescale should succeed");

        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 0,
                    duration_ticks: 1920,
                    midi_note: 60,
                    velocity: 80,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 7680,
                    duration_ticks: 1920,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 11520,
                    duration_ticks: 1920,
                    midi_note: 67,
                    velocity: 50,
                    release_velocity: None,
                },
            ]
        );
        assert_eq!(score.loop_ticks, Some(15360));
        // The measure grid stretches with the loop.
        assert_eq!(score.measure_ticks, Some(7680));

        // Every note keeps its relative position in the loop: 0, 0.5, 0.75.
        for (pair, expected) in pairs.iter().zip([0.0, 0.5, 0.75]) {
            let relative = pair.start_tick as f64 / 15360.0;
            assert!((relative - expected).abs() < 1e-9);
        }
        assert!(score.to_events().is_ok());
    }

    #[test]
    fn rescaling_compresses_notes_and_keeps_them_in_the_loop() {
        let mut score = rescale_fixture();
        // Halve the eight-second loop to four.
        score.rescale_to_duration(4.0).expect("the rescale should succeed");

        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 0,
                    duration_ticks: 480,
                    midi_note: 60,
                    velocity: 80,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 1920,
                    duration_ticks: 480,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 2880,
                    duration_ticks: 480,
                    midi_note: 67,
                    velocity: 50,
                    release_velocity: None,
                },
            ]
        );
        assert_eq!(score.loop_ticks, Some(3840));
        assert_eq!(score.measure_ticks, Some(1920));

        // No note escapes the shortened loop, and the score still plays.
        for pair in &pairs {
            assert!(pair.end_tick() <= 3840);
        }
        let events = score.to_events().expect("the rescaled score should play");
        assert_eq!(events.len(), 6);
    }

    #[test]
    fn rescaling_an_implicit_loop_makes_it_explicit() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        assert_eq!(score.loop_ticks, None);
        // The implicit loop is the last release: tick 480, half a second.
        score.rescale_to_duration(2.0).expect("the rescale should succeed");

        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![ScoreNotePair {
                start_tick: 0,
                duration_ticks: 1920,
                midi_note: 60,
                velocity: 80,
                release_velocity: None,
            }]
        );
        assert_eq!(score.loop_ticks, Some(1920));
    }

    #[test]
    fn rescaling_by_a_fractional_factor_keeps_relative_positions() {
        let mut score = rescale_fixture();
        // Grow the eight-second loop to twelve (a 1.5x stretch).
        score.rescale_to_duration(12.0).expect("the rescale should succeed");

        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs.iter().map(|pair| pair.start_tick).collect::<Vec<_>>(),
            vec![0, 5760, 8640]
        );
        assert!(pairs.iter().all(|pair| pair.duration_ticks == 1440));
        assert_eq!(score.loop_ticks, Some(11520));
        assert_eq!(score.measure_ticks, Some(5760));
    }

    #[test]
    fn rescaling_rejects_bad_durations() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut score = rescale_fixture();
            assert!(
                score.rescale_to_duration(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }

        // A positive duration shorter than one tick on a 960-tick grid is
        // still rejected.
        let mut score = rescale_fixture();
        assert!(score.rescale_to_duration(1e-9).is_err());
    }

    #[test]
    fn shifting_later_moves_every_note_and_the_loop() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 480 C4 80
3840 480 E4 64
",
        )
        .expect("the fixture should parse");

        score.shift_ticks(1920).expect("the shift should succeed");
        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 1920,
                    duration_ticks: 480,
                    midi_note: 60,
                    velocity: 80,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 5760,
                    duration_ticks: 480,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
            ]
        );
        // The explicit loop moves with the notes so the content still fits.
        assert_eq!(score.loop_ticks, Some(9600));
        assert!(score.to_events().is_ok());

        // Zero is a no-op.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let before: Vec<ScoreNotePair> = score.pairs().copied().collect();
        score.shift_ticks(0).expect("zero is a no-op");
        assert_eq!(score.pairs().copied().collect::<Vec<_>>(), before);
        assert_eq!(score.loop_ticks, Some(7680));
    }

    #[test]
    fn shifting_earlier_drops_notes_that_would_go_negative() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 480 C4 80
1920 480 E4 64
5760 480 G4 50
",
        )
        .expect("the fixture should parse");

        // Shift two seconds earlier: the tick-0 note is dropped, a note that
        // starts exactly at the shift point survives at tick 0, and the rest
        // move up.
        score.shift_ticks(-1920).expect("the shift should succeed");
        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 0,
                    duration_ticks: 480,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 3840,
                    duration_ticks: 480,
                    midi_note: 67,
                    velocity: 50,
                    release_velocity: None,
                },
            ]
        );
        // The explicit loop shortens with the notes.
        assert_eq!(score.loop_ticks, Some(5760));
        let events = score.to_events().expect("the shifted score should play");
        assert_eq!(events.len(), 4);

        // Shifting past every note leaves nothing to play.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let error = score.shift_ticks(-480).expect_err("no notes remain");
        assert!(error.contains("leaves no notes"), "{error}");
    }

    #[test]
    fn shifting_an_implicit_loop_needs_no_loop_change() {
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
480 480 C4 80
",
        )
        .expect("the fixture should parse");
        assert_eq!(score.loop_ticks, None);

        score.shift_ticks(960).expect("the shift should succeed");
        assert_eq!(score.loop_ticks, None, "an implicit loop is untouched");
        // The loop now follows the shifted last release (tick 1920).
        assert_eq!(score.effective_loop_ticks(), 1920);

        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
480 480 C4 80
",
        )
        .expect("the fixture should parse");
        score.shift_ticks(-480).expect("the shift should succeed");
        assert_eq!(score.loop_ticks, None);
        // The note now starts at tick 0 and the loop follows its release.
        let pairs: Vec<ScoreNotePair> = score.pairs().copied().collect();
        assert_eq!(pairs[0].start_tick, 0);
        assert_eq!(score.effective_loop_ticks(), 480);
    }

    #[test]
    fn shifting_by_seconds_rounds_to_the_nearest_tick() {
        // One quarter second on a 960-tick grid is exactly 240 ticks.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        score.shift_seconds(0.25).expect("the shift should succeed");
        assert_eq!(score.pairs().next().unwrap().start_tick, 240);
        assert_eq!(score.loop_ticks, Some(7920));

        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 7680
960 480 C4 80
",
        )
        .expect("the fixture should parse");
        score.shift_seconds(-0.25).expect("the shift should succeed");
        assert_eq!(score.pairs().next().unwrap().start_tick, 720);
        assert_eq!(score.loop_ticks, Some(7440));

        // Non-finite shifts are rejected without mutating the score.
        let mut score = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY] {
            assert!(score.shift_seconds(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn merging_keeps_the_first_grid_and_rescales_the_other() {
        let first = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 1920
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let second = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 1920
loop_ticks 3840
480 480 E4 64 ; a comment on the pair
",
        )
        .expect("the fixture should parse");

        let merged = first.clone().merge_with(second);

        // The first score's grid wins; the second's ticks rescale onto it.
        assert_eq!(merged.ticks_per_second, 960);
        let pairs: Vec<ScoreNotePair> = merged.pairs().copied().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(
            pairs[0],
            ScoreNotePair {
                start_tick: 0,
                duration_ticks: 480,
                midi_note: 60,
                velocity: 80,
                release_velocity: None,
            }
        );
        // 480 ticks at 1920/second is a quarter second: 240 ticks at
        // 960/second, both for the start and the duration.
        assert_eq!(
            pairs[1],
            ScoreNotePair {
                start_tick: 240,
                duration_ticks: 240,
                midi_note: 64,
                velocity: 64,
                release_velocity: None,
            }
        );

        // Both scores loop at two seconds, so the merged loop is 1920 ticks
        // on the first grid and every note still fits inside it.
        assert_eq!(merged.loop_ticks, Some(1920));
        let events = merged.to_events().expect("the merge should be playable");
        assert_eq!(events.len(), 4);

        // The second score's pair comment survives the merge.
        assert!(merged.format().contains("; a comment on the pair"));
    }

    #[test]
    fn merging_unequal_loops_keeps_the_longer_one() {
        let first = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 960
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let second = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 1920
0 480 E4 64
",
        )
        .expect("the fixture should parse");

        let merged = first.merge_with(second);
        // The merged loop is the longer of the two (two seconds); the
        // shorter score's notes sound once per merged cycle.
        assert_eq!(merged.loop_ticks, Some(1920));
        assert_eq!(merged.pairs().count(), 2);
        assert!(merged.to_events().is_ok());
    }

    #[test]
    fn merging_scores_on_the_same_grid_is_exact() {
        let first = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let second = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
240 480 E4 64
",
        )
        .expect("the fixture should parse");

        let merged = first.merge_with(second);
        assert_eq!(merged.ticks_per_second, 960);
        // No explicit loop header: the merge infers one from the last
        // release (tick 720).
        assert_eq!(merged.loop_ticks, Some(720));
        let pairs: Vec<ScoreNotePair> = merged.pairs().copied().collect();
        assert_eq!(
            pairs,
            vec![
                ScoreNotePair {
                    start_tick: 0,
                    duration_ticks: 480,
                    midi_note: 60,
                    velocity: 80,
                    release_velocity: None,
                },
                ScoreNotePair {
                    start_tick: 240,
                    duration_ticks: 480,
                    midi_note: 64,
                    velocity: 64,
                    release_velocity: None,
                },
            ]
        );
    }

    #[test]
    fn rescaling_rounds_to_the_nearest_tick_and_keeps_notes_sounded() {
        // One tick on a 1920 grid is half a tick on a 960 grid: it rounds
        // to the nearest tick, and a note shrinks to at least one tick.
        assert_eq!(rescale_ticks(480, 1920, 960), 240);
        assert_eq!(rescale_ticks(3840, 1920, 960), 1920);
        assert_eq!(rescale_ticks(0, 1920, 960), 0);
        // The identity conversion is exact.
        assert_eq!(rescale_ticks(12345, 960, 960), 12345);

        let first = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 960
0 480 C4 80
",
        )
        .expect("the fixture should parse");
        let second = BhsScore::parse(
            "\
format bhs-score-v1
ticks_per_second 1920
1 1 E4 64
",
        )
        .expect("the fixture should parse");
        let merged = first.merge_with(second);
        let pairs: Vec<ScoreNotePair> = merged.pairs().copied().collect();
        // One tick on the 1920 grid is half a tick on the 960 grid; the
        // tie rounds up, and the note keeps at least one tick of duration.
        assert_eq!(pairs[1].start_tick, 1);
        assert_eq!(pairs[1].duration_ticks, 1);
    }

    #[test]
    fn zero_noise_leaves_score_untouched() {
        let mut mutant = BhsScore::parse(SCORE)
            .expect("the fixture should parse")
            .into_mutant();
        let before: Vec<ScoreNotePair> = mutant.inner.pairs().copied().collect();
        assert_eq!(mutant.mutate(0.0), 0);
        let after: Vec<ScoreNotePair> = mutant.inner.pairs().copied().collect();
        assert_eq!(after, before);
    }

    #[test]
    fn mutation_primitives_keep_pairs_valid() {
        // Deletion removes exactly one pair.
        let mut mutant = fixture_mutant(2);
        let before = mutant.inner.pairs().count();
        let removed = mutant
            .delete_random_pair()
            .expect("a pair should be removed");
        assert_eq!(mutant.inner.pairs().count(), before - 1);
        assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&removed.midi_note));

        // Substitution keeps the pair count, start ticks, and release
        // velocities while rewriting note, duration, and velocity.
        let mut mutant = fixture_mutant(2);
        let before: Vec<ScoreNotePair> = mutant.inner.pairs().copied().collect();
        mutant
            .substitute_random_pair()
            .expect("a pair should be substituted");
        let after: Vec<ScoreNotePair> = mutant.inner.pairs().copied().collect();
        assert_eq!(after.len(), before.len());
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(new.start_tick, old.start_tick);
            assert_eq!(new.release_velocity, old.release_velocity);
            assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&new.midi_note));
            assert!(new.duration_ticks >= 1);
            assert!(new.velocity <= MAX_VELOCITY);
        }

        // Insertion adds exactly one valid pair at the reported index.
        let mut mutant = fixture_mutant(2);
        let before = mutant.inner.pairs().count();
        let inserted_at = mutant
            .insert_after_random_pair()
            .expect("a pair should be inserted");
        assert_eq!(mutant.inner.pairs().count(), before + 1);
        let inserted = *mutant
            .inner
            .pairs()
            .nth(inserted_at)
            .expect("the inserted pair should exist");
        assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&inserted.midi_note));
        assert!(inserted.duration_ticks >= 1);
        assert!(inserted.velocity <= MAX_VELOCITY);
        assert_eq!(inserted.release_velocity, None);
    }

    #[test]
    fn mutated_notes_stay_in_range_and_bias_toward_center() {
        let mut rng = StdRng::seed_from_u64(12);
        let samples = 10_000;
        let mut center_hits = 0;
        for _ in 0..samples {
            let note = random_note(&mut rng);
            assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&note));
            // The middle third of the keyboard (E3..=G#5).
            if (48..=79).contains(&note) {
                center_hits += 1;
            }
        }
        // Exponential decay from the center puts ~69% of the draws in the
        // middle third; a uniform draw would give ~36%.
        assert!(center_hits * 5 > samples * 3);
    }

    #[test]
    fn mutated_durations_stay_in_bounds_and_bias_short() {
        let ticks_per_second = 960;
        let max_ticks = MAX_MUTATED_DURATION_SECS * ticks_per_second;
        let mut rng = StdRng::seed_from_u64(11);
        let samples = 10_000;
        let mut under_two_seconds = 0;
        for _ in 0..samples {
            let ticks = random_duration_ticks(&mut rng, ticks_per_second);
            assert!((1..=max_ticks).contains(&ticks));
            if (ticks as f64 / ticks_per_second as f64) < 2.0 {
                under_two_seconds += 1;
            }
        }
        // The log scale puts most draws in the lower part of the range: ~77%
        // under two seconds, versus ~40% for a uniform draw over 0..5s.
        assert!(under_two_seconds * 5 > samples * 3);
    }

    #[test]
    fn heavy_noise_keeps_score_valid() {
        let mut mutant = fixture_mutant(3);
        mutant.mutate(1.0);
        for pair in mutant.inner.pairs() {
            assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&pair.midi_note));
            assert!(pair.duration_ticks >= 1);
            assert!(pair.velocity <= MAX_VELOCITY);
        }
        // The mutated document still round-trips through the text format.
        let reemitted =
            BhsScore::parse(&mutant.inner.format()).expect("mutated score should re-parse");
        assert_eq!(reemitted.pairs().count(), mutant.inner.pairs().count());
    }

    #[test]
    fn mutations_on_an_empty_score_are_no_ops() {
        let mut mutant = MutantScore {
            inner: BhsScore::new(960, None, None, Vec::new()),
            rng: StdRng::seed_from_u64(4),
        };
        assert!(mutant.delete_random_pair().is_none());
        assert!(mutant.substitute_random_pair().is_none());
        assert!(mutant.insert_after_random_pair().is_none());
        assert_eq!(mutant.mutate(1.0), 0);
    }

    #[test]
    fn ticks_convert_exactly() {
        assert_eq!(ticks_to_duration(0, 1920), Duration::ZERO);
        assert_eq!(ticks_to_duration(1920, 1920), Duration::from_secs(1));
        // 2073 ticks at 1920/second is exactly 1.0796875s.
        assert_eq!(ticks_to_duration(2073, 1920), Duration::new(1, 79_687_500));
    }
}
