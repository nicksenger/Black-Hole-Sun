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
/// The longest a mutated or inserted note may be held, in seconds.
pub const MAX_MUTATED_DURATION_SECS: u64 = 5;

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

    /// Give a random pair a new random 88-key note, a random duration between
    /// 0 and [`MAX_MUTATED_DURATION_SECS`] seconds, and a random velocity,
    /// keeping its start tick and release velocity. Returns the previous
    /// values; `None` without mutating if the score has no pairs.
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
    /// start tick (or the end of the loop for the last pair), with a random
    /// note, a random duration between 0 and [`MAX_MUTATED_DURATION_SECS`]
    /// seconds, and a random velocity. It carries no comments; returns `None`
    /// without mutating if the score has no pairs.
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

/// A random 88-key MIDI note.
fn random_note(rng: &mut impl RngExt) -> u8 {
    rng.random_range(FIRST_MIDI_NOTE..=LAST_MIDI_NOTE)
}

/// A random velocity in `0..=MAX_VELOCITY`.
fn random_velocity(rng: &mut impl RngExt) -> u8 {
    rng.random_range(0..=MAX_VELOCITY)
}

/// A random duration between 0 and [`MAX_MUTATED_DURATION_SECS`] seconds,
/// kept at least one tick so the pair stays a valid note.
fn random_duration_ticks(rng: &mut impl RngExt, ticks_per_second: u64) -> u64 {
    let max_ticks = MAX_MUTATED_DURATION_SECS.saturating_mul(ticks_per_second);
    rng.random_range(1..=max_ticks.max(1))
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
            shifted.pairs().map(|pair| pair.midi_note).collect::<Vec<_>>(),
            vec![62, 47, 66, 68, 68]
        );
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
        let removed = mutant.delete_random_pair().expect("a pair should be removed");
        assert_eq!(mutant.inner.pairs().count(), before - 1);
        assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&removed.midi_note));

        // Substitution keeps the pair count, start ticks, and release
        // velocities while rewriting note, duration, and velocity.
        let mut mutant = fixture_mutant(2);
        let before: Vec<ScoreNotePair> = mutant.inner.pairs().copied().collect();
        mutant.substitute_random_pair().expect("a pair should be substituted");
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
        let inserted_at =
            mutant.insert_after_random_pair().expect("a pair should be inserted");
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
    fn heavy_noise_keeps_score_valid() {
        let mut mutant = fixture_mutant(3);
        mutant.mutate(1.0);
        for pair in mutant.inner.pairs() {
            assert!((FIRST_MIDI_NOTE..=LAST_MIDI_NOTE).contains(&pair.midi_note));
            assert!(pair.duration_ticks >= 1);
            assert!(pair.velocity <= MAX_VELOCITY);
        }
        // The mutated document still round-trips through the text format.
        let reemitted = BhsScore::parse(&mutant.inner.format())
            .expect("mutated score should re-parse");
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
