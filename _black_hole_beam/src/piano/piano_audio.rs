//! Low-latency physically inspired piano synthesis.

use std::f64::consts::TAU;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use tracing::warn;

use crate::piano::piano_score::load_score;
use crate::{PianoAction, PianoEvent};

const PARTIAL_COUNT: usize = 48;
const MAX_POLYPHONY: usize = 128;
const MASTER_GAIN: f32 = 0.19;
/// Hammer noise is kept well below the string body; the reference piano's
/// attack transient is mostly hammer knock, not broadband hiss.
const NOISE_SCALE: f32 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PianoRenderReport {
    pub frames: u64,
    pub sample_rate: u32,
    pub event_count: usize,
}

/// Render one pass of a JSON piano score plus a decay tail to stereo PCM s16le WAV.
pub fn render_piano_score_to_wav(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    sample_rate: u32,
    tail: Duration,
) -> Result<PianoRenderReport, String> {
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(format!(
            "sample rate {sample_rate} is outside the supported range 8000..=192000"
        ));
    }
    let score = load_score(input.as_ref())?;
    let event_count = score.events.len();
    let scheduled = score
        .events
        .iter()
        .copied()
        .map(|event| {
            let frame = (event.timestamp.as_secs_f64() * f64::from(sample_rate)).round() as u64;
            (frame, command_from_event(event))
        })
        .collect::<Vec<_>>();
    let frames =
        ((score.loop_duration + tail).as_secs_f64() * f64::from(sample_rate)).ceil() as u64;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output.as_ref(), spec)
        .map_err(|error| format!("could not create {}: {error}", output.as_ref().display()))?;
    let mut synth = PianoSynth::new(sample_rate as f32);
    let mut event_index = 0;
    for frame in 0..frames {
        while event_index < scheduled.len() && scheduled[event_index].0 <= frame {
            synth.command(scheduled[event_index].1);
            event_index += 1;
        }
        let (left, right) = synth.next_sample();
        writer
            .write_sample(float_to_pcm_s16(left))
            .and_then(|_| writer.write_sample(float_to_pcm_s16(right)))
            .map_err(|error| format!("could not write {}: {error}", output.as_ref().display()))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("could not finalize {}: {error}", output.as_ref().display()))?;
    Ok(PianoRenderReport {
        frames,
        sample_rate,
        event_count,
    })
}

fn float_to_pcm_s16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
}

#[derive(Debug, Clone, Copy)]
enum Command {
    Attack {
        voice_id: u64,
        midi_note: u8,
        frequency_hz: f32,
        velocity: f32,
        pressure: Option<f32>,
    },
    Release {
        voice_id: u64,
        velocity: f32,
    },
    /// Start (or restart) the metronome at `bpm` beats per minute; a
    /// non-positive or non-finite tempo leaves it silent.
    Metronome {
        bpm: f32,
    },
}

/// Owns the cpal stream. Dropping this value stops playback.
pub(crate) struct PianoAudioEngine {
    command_tx: Sender<Command>,
    runtime_error: Arc<Mutex<Option<String>>>,
    _stream: Stream,
}

impl PianoAudioEngine {
    pub(crate) fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default audio output device was found".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|error| format!("could not read the default audio format: {error}"))?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let (command_tx, command_rx) = mpsc::channel();
        let runtime_error = Arc::new(Mutex::new(None));

        let stream = match sample_format {
            SampleFormat::F32 => {
                build_stream::<f32>(&device, &config, command_rx, Arc::clone(&runtime_error))
            }
            SampleFormat::I16 => {
                build_stream::<i16>(&device, &config, command_rx, Arc::clone(&runtime_error))
            }
            SampleFormat::U16 => {
                build_stream::<u16>(&device, &config, command_rx, Arc::clone(&runtime_error))
            }
            format => return Err(format!("unsupported audio sample format: {format:?}")),
        }
        .map_err(|error| format!("could not create the audio output stream: {error}"))?;
        stream
            .play()
            .map_err(|error| format!("could not start the audio output stream: {error}"))?;

        Ok(Self {
            command_tx,
            runtime_error,
            _stream: stream,
        })
    }

    pub(crate) fn perform(&self, event: PianoEvent) {
        let command = command_from_event(event);
        if self.command_tx.send(command).is_err() {
            warn!("piano audio command discarded because the output stream has stopped");
        }
    }

    /// Start a metronome click on every beat at `bpm` beats per minute.
    ///
    /// The clock runs on the audio thread's sample counter, so beats stay
    /// sample-accurate regardless of the window's frame rate. A zero `bpm`
    /// leaves the metronome silent.
    pub(crate) fn enable_metronome(&self, bpm: u32) {
        let command = Command::Metronome { bpm: bpm as f32 };
        if self.command_tx.send(command).is_err() {
            warn!("metronome command discarded because the output stream has stopped");
        }
    }

    pub(crate) fn error(&self) -> Option<String> {
        self.runtime_error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }
}

fn command_from_event(event: PianoEvent) -> Command {
    match event.action {
        PianoAction::Attack { velocity, pressure } => Command::Attack {
            voice_id: event.voice_id,
            midi_note: event.note.midi_note,
            frequency_hz: event.note.frequency_hz,
            velocity,
            pressure,
        },
        PianoAction::Release { velocity, .. } => Command::Release {
            voice_id: event.voice_id,
            velocity,
        },
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    command_rx: Receiver<Command>,
    runtime_error: Arc<Mutex<Option<String>>>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = usize::from(config.channels);
    let mut synth = PianoSynth::new(config.sample_rate as f32);
    device.build_output_stream(
        config,
        move |output: &mut [T], _info| {
            while let Ok(command) = command_rx.try_recv() {
                synth.command(command);
            }
            render(output, channels, &mut synth);
        },
        move |error| handle_stream_error(error, &runtime_error),
        None,
    )
}

fn handle_stream_error(error: cpal::StreamError, runtime_error: &Mutex<Option<String>>) {
    match error {
        cpal::StreamError::BufferUnderrun => {
            warn!("piano audio buffer underrun/overrun; playback recovered");
        }
        error => {
            warn!(%error, "piano audio stream failure");
            if let Ok(mut current) = runtime_error.lock() {
                *current = Some(format!("audio stream stopped: {error}"));
            }
        }
    }
}

fn render<T>(output: &mut [T], channels: usize, synth: &mut PianoSynth)
where
    T: Sample + FromSample<f32>,
{
    if channels == 0 {
        return;
    }
    for frame in output.chunks_mut(channels) {
        let (left, right) = synth.next_sample();
        if channels == 1 {
            frame[0] = T::from_sample((left + right) * 0.5);
        } else {
            frame[0] = T::from_sample(left);
            frame[1] = T::from_sample(right);
            let center = T::from_sample((left + right) * 0.5);
            for sample in &mut frame[2..] {
                *sample = center;
            }
        }
    }
}

struct PianoSynth {
    sample_rate: f32,
    voices: Vec<PianoVoice>,
    soundboard: StereoSoundboard,
    eq: StereoEq,
    /// Total samples rendered so far; the metronome's clock runs on it.
    frame: u64,
    /// The active metronome, if one was enabled via [`Command::Metronome`].
    metronome: Option<MetronomeClock>,
    /// Metronome clicks currently sounding; each decays away on its own.
    clicks: Vec<MetronomeClick>,
}

impl PianoSynth {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            voices: Vec::with_capacity(MAX_POLYPHONY),
            soundboard: StereoSoundboard::new(sample_rate),
            eq: StereoEq::new(sample_rate),
            frame: 0,
            metronome: None,
            clicks: Vec::new(),
        }
    }

    fn command(&mut self, command: Command) {
        match command {
            Command::Attack {
                voice_id,
                midi_note,
                frequency_hz,
                velocity,
                pressure,
            } => {
                if self.voices.len() >= MAX_POLYPHONY {
                    let steal = self
                        .voices
                        .iter()
                        .position(|voice| voice.released)
                        .unwrap_or(0);
                    self.voices.remove(steal);
                }
                self.voices.push(PianoVoice::new(
                    voice_id,
                    midi_note,
                    frequency_hz,
                    velocity,
                    pressure,
                    self.sample_rate,
                ));
            }
            Command::Release { voice_id, velocity } => {
                if let Some(voice) = self
                    .voices
                    .iter_mut()
                    .find(|voice| voice.voice_id == voice_id)
                {
                    voice.release(velocity);
                }
            }
            Command::Metronome { bpm } => {
                if bpm.is_finite() && bpm > 0.0 {
                    // The interval is clamped to a whole sample so the
                    // scheduler below can never spin on sub-sample beats.
                    let beat_interval = (self.sample_rate as f64 * 60.0 / f64::from(bpm)).max(1.0);
                    self.metronome = Some(MetronomeClock {
                        beat_interval_samples: beat_interval,
                        // Anchor the first click to the next rendered
                        // sample so the metronome starts immediately.
                        next_beat_at: self.frame as f64,
                    });
                } else {
                    self.metronome = None;
                }
            }
        }
    }

    fn next_sample(&mut self) -> (f32, f32) {
        let mut left = 0.0;
        let mut right = 0.0;
        self.voices.retain_mut(|voice| {
            let (voice_left, voice_right) = voice.next_sample();
            left += voice_left;
            right += voice_right;
            !voice.finished()
        });

        // Metronome clicks are scheduled on the sample-accurate frame clock
        // so beats do not drift with the UI's message loop. They are kept
        // dry — outside the soundboard's room — so each tick stays a crisp
        // transient instead of ringing out between beats.
        let mut click = 0.0;
        if let Some(metronome) = &mut self.metronome {
            while (self.frame as f64) >= metronome.next_beat_at {
                self.clicks.push(MetronomeClick::new(self.sample_rate));
                metronome.next_beat_at += metronome.beat_interval_samples;
            }
        }
        for click_voice in &mut self.clicks {
            click += click_voice.next_sample();
        }
        self.clicks.retain(|click_voice| !click_voice.finished());

        let (body_left, body_right) = self.soundboard.process(left, right);
        let left = soft_limit((left + body_left + click) * MASTER_GAIN);
        let right = soft_limit((right + body_right + click) * MASTER_GAIN);
        let (left, right) = self.eq.process(left, right);
        // The midrange lift can push rare peaks just past unity; keep the
        // output contract of bounded samples.
        let (left, right) = (left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0));
        self.frame += 1;
        (left, right)
    }
}

/// The metronome's beat clock, in samples relative to [`PianoSynth::frame`].
struct MetronomeClock {
    /// Samples between beats; at least one sample per beat.
    beat_interval_samples: f64,
    /// The frame at which the next click sounds.
    next_beat_at: f64,
}

/// A single metronome click: a short woodblock-like tick built from two
/// inharmonic partials with a fast exponential decay.
struct MetronomeClick {
    fundamental_sine: f64,
    fundamental_cosine: f64,
    fundamental_step_sine: f64,
    fundamental_step_cosine: f64,
    overtone_sine: f64,
    overtone_cosine: f64,
    overtone_step_sine: f64,
    overtone_step_cosine: f64,
    envelope: f32,
    decay: f32,
}

impl MetronomeClick {
    const FUNDAMENTAL_HZ: f32 = 1_000.0;
    /// The overtone's inharmonic ratio and level relative to the fundamental.
    const OVERTONE_RATIO: f32 = 2.7;
    const OVERTONE_GAIN: f64 = 0.35;
    /// Peak click amplitude before the master gain; a full piano attack sums
    /// several times higher, so the tick stays subordinate to the notes.
    const PEAK_ENVELOPE: f32 = 0.75;
    /// The click's decay time constant in seconds; ≈ 80 ms (5.6 time
    /// constants) until the click is dropped as finished.
    const DECAY_SECONDS: f32 = 0.014;

    fn new(sample_rate: f32) -> Self {
        let step = |frequency_hz: f32| {
            let phase_step = TAU * f64::from(frequency_hz / sample_rate);
            (phase_step.sin(), phase_step.cos())
        };
        let (fundamental_step_sine, fundamental_step_cosine) = step(Self::FUNDAMENTAL_HZ);
        let (overtone_step_sine, overtone_step_cosine) =
            step(Self::FUNDAMENTAL_HZ * Self::OVERTONE_RATIO);
        Self {
            // Starting at zero phase fades the tick in from silence, so the
            // trigger itself cannot click.
            fundamental_sine: 0.0,
            fundamental_cosine: 1.0,
            fundamental_step_sine,
            fundamental_step_cosine,
            overtone_sine: 0.0,
            overtone_cosine: 1.0,
            overtone_step_sine,
            overtone_step_cosine,
            envelope: Self::PEAK_ENVELOPE,
            decay: (-1.0 / (Self::DECAY_SECONDS * sample_rate)).exp(),
        }
    }

    fn next_sample(&mut self) -> f32 {
        let sample = (self.fundamental_sine + Self::OVERTONE_GAIN * self.overtone_sine) as f32
            * self.envelope;
        rotate(
            &mut self.fundamental_sine,
            &mut self.fundamental_cosine,
            self.fundamental_step_sine,
            self.fundamental_step_cosine,
        );
        rotate(
            &mut self.overtone_sine,
            &mut self.overtone_cosine,
            self.overtone_step_sine,
            self.overtone_step_cosine,
        );
        self.envelope *= self.decay;
        sample
    }

    fn finished(&self) -> bool {
        self.envelope < 0.002
    }
}

struct PianoVoice {
    voice_id: u64,
    midi_note: u8,
    sample_rate: f32,
    partial_sine: [f64; PARTIAL_COUNT],
    partial_cosine: [f64; PARTIAL_COUNT],
    partial_step_sine: [f64; PARTIAL_COUNT],
    partial_step_cosine: [f64; PARTIAL_COUNT],
    beat_sine: [f64; PARTIAL_COUNT],
    beat_cosine: [f64; PARTIAL_COUNT],
    beat_step_sine: [f64; PARTIAL_COUNT],
    beat_step_cosine: [f64; PARTIAL_COUNT],
    amplitude: [f32; PARTIAL_COUNT],
    envelope: [f32; PARTIAL_COUNT],
    natural_decay: [f32; PARTIAL_COUNT],
    age_samples: u64,
    attack_samples: f32,
    released: bool,
    release_gain: f32,
    release_decay: f32,
    hammer_envelope: f32,
    hammer_decay: f32,
    hammer_sine: f64,
    hammer_cosine: f64,
    hammer_step_sine: f64,
    hammer_step_cosine: f64,
    previous_noise: f32,
    noise_state: u32,
    pan_left: f32,
    pan_right: f32,
}

impl PianoVoice {
    fn new(
        voice_id: u64,
        midi_note: u8,
        frequency_hz: f32,
        velocity: f32,
        pressure: Option<f32>,
        sample_rate: f32,
    ) -> Self {
        let velocity = velocity.clamp(0.01, 1.0);
        let pressure = pressure.unwrap_or(velocity * 0.72).clamp(0.0, 1.0);
        let key_position = (f32::from(midi_note) - 21.0) / 87.0;
        let pan = key_position * 1.5 - 0.75;
        let pan_left = ((1.0 - pan) * 0.5).sqrt();
        let pan_right = ((1.0 + pan) * 0.5).sqrt();
        let inharmonicity = 0.000_3 * 2.0_f32.powf(key_position * 6.4);
        let hammer_position = 0.115 + 0.035 * key_position;
        let spectral_rolloff = 0.35 + 0.58 * (1.0 - velocity) + 0.28 * (1.0 - pressure);
        let base_decay_seconds = 3.4 - 2.7 * key_position.powf(0.72);
        let velocity_gain = velocity.powf(0.72) * (0.88 + pressure * 0.12);

        let mut partial_sine = [0.0; PARTIAL_COUNT];
        let mut partial_cosine = [0.0; PARTIAL_COUNT];
        let mut partial_step_sine = [0.0; PARTIAL_COUNT];
        let mut partial_step_cosine = [0.0; PARTIAL_COUNT];
        let mut beat_sine = [0.0; PARTIAL_COUNT];
        let mut beat_cosine = [0.0; PARTIAL_COUNT];
        let mut beat_step_sine = [0.0; PARTIAL_COUNT];
        let mut beat_step_cosine = [0.0; PARTIAL_COUNT];
        let mut amplitude = [0.0; PARTIAL_COUNT];
        let envelope = [1.0; PARTIAL_COUNT];
        let mut natural_decay = [0.0; PARTIAL_COUNT];

        for partial in 0..PARTIAL_COUNT {
            let harmonic = (partial + 1) as f32;
            let partial_frequency =
                frequency_hz * harmonic * (1.0 + inharmonicity * harmonic * harmonic).sqrt();
            let phase = pseudo_phase(voice_id, partial);
            partial_sine[partial] = phase.sin();
            partial_cosine[partial] = phase.cos();
            let phase_step = f64::from(partial_frequency / sample_rate) * TAU;
            partial_step_sine[partial] = phase_step.sin();
            partial_step_cosine[partial] = phase_step.cos();
            let beat_phase = pseudo_phase(voice_id.wrapping_add(17), partial);
            beat_sine[partial] = beat_phase.sin();
            beat_cosine[partial] = beat_phase.cos();
            let unison_hz = 0.18 + 0.018 * harmonic + 0.35 * key_position;
            let beat_step = f64::from(unison_hz / sample_rate) * TAU;
            beat_step_sine[partial] = beat_step.sin();
            beat_step_cosine[partial] = beat_step.cos();

            let hammer_node = (std::f32::consts::PI * harmonic * hammer_position)
                .sin()
                .abs()
                .powf(0.2);
            amplitude[partial] = velocity_gain * hammer_node / harmonic.powf(spectral_rolloff);
            if partial == 0 {
                // Damp the lowest fundamentals so the soundboard body, rather
                // than raw bass strings, carries the low end.
                let ramp = ((key_position - 0.126) / (0.15 - 0.126)).clamp(0.4, 1.0);
                amplitude[partial] *= ramp;
            }
            if partial_frequency > sample_rate * 0.47 {
                amplitude[partial] = 0.0;
            }
            let decay_seconds = base_decay_seconds / (1.0 + 0.13 * (harmonic - 1.0).powf(1.16));
            natural_decay[partial] = (-1.0 / (decay_seconds * sample_rate)).exp();
        }

        Self {
            voice_id,
            midi_note,
            sample_rate,
            partial_sine,
            partial_cosine,
            partial_step_sine,
            partial_step_cosine,
            beat_sine,
            beat_cosine,
            beat_step_sine,
            beat_step_cosine,
            amplitude,
            envelope,
            natural_decay,
            age_samples: 0,
            attack_samples: sample_rate * (0.0012 + 0.0028 * (1.0 - velocity)),
            released: false,
            release_gain: 1.0,
            release_decay: 1.0,
            hammer_envelope: 0.06 + velocity * 0.16,
            hammer_decay: (-1.0 / (sample_rate * (0.006 + 0.006 * velocity))).exp(),
            hammer_sine: 0.0,
            hammer_cosine: 1.0,
            hammer_step_sine: (TAU
                * f64::from((1_150.0 + 14.0 * f32::from(midi_note)) / sample_rate))
            .sin(),
            hammer_step_cosine: (TAU
                * f64::from((1_150.0 + 14.0 * f32::from(midi_note)) / sample_rate))
            .cos(),
            previous_noise: 0.0,
            noise_state: (voice_id as u32)
                .wrapping_mul(747_796_405)
                .wrapping_add(u32::from(midi_note).wrapping_mul(2_891_336_453)),
            pan_left,
            pan_right,
        }
    }

    fn release(&mut self, release_velocity: f32) {
        self.released = true;
        // Fast releases damp harder; bass strings still take longer to settle.
        let key_position = (f32::from(self.midi_note) - 21.0) / 87.0;
        let release_seconds =
            (0.55 - 0.2 * key_position) * (1.0 - 0.4 * release_velocity.clamp(0.0, 1.0));
        self.release_decay = (-1.0 / (release_seconds * self.sample_rate)).exp();
    }

    fn next_sample(&mut self) -> (f32, f32) {
        let attack = (self.age_samples as f32 / self.attack_samples).clamp(0.0, 1.0);
        let attack = attack * attack * (3.0 - 2.0 * attack);
        let mut string = 0.0;

        for partial in 0..PARTIAL_COUNT {
            let unison = 1.0 + 0.055 * self.beat_sine[partial] as f32;
            string += self.partial_sine[partial] as f32
                * self.amplitude[partial]
                * self.envelope[partial]
                * unison;
            rotate(
                &mut self.partial_sine[partial],
                &mut self.partial_cosine[partial],
                self.partial_step_sine[partial],
                self.partial_step_cosine[partial],
            );
            rotate(
                &mut self.beat_sine[partial],
                &mut self.beat_cosine[partial],
                self.beat_step_sine[partial],
                self.beat_step_cosine[partial],
            );
            self.envelope[partial] *= self.natural_decay[partial];
        }

        let noise = self.next_noise();
        let hammer_noise = (noise - self.previous_noise) * self.hammer_envelope * NOISE_SCALE;
        self.previous_noise = noise;
        let hammer_knock = self.hammer_sine as f32 * self.hammer_envelope * 0.55;
        rotate(
            &mut self.hammer_sine,
            &mut self.hammer_cosine,
            self.hammer_step_sine,
            self.hammer_step_cosine,
        );
        self.hammer_envelope *= self.hammer_decay;
        if self.released {
            self.release_gain *= self.release_decay;
        }
        self.age_samples += 1;

        // Fade the whole strike in from silence. The hammer's differentiated
        // noise can otherwise begin at an arbitrary non-zero value, producing
        // a click that becomes especially noticeable across dense chords.
        let sample = (string + hammer_noise + hammer_knock) * attack * self.release_gain;
        (sample * self.pan_left, sample * self.pan_right)
    }

    fn next_noise(&mut self) -> f32 {
        self.noise_state ^= self.noise_state << 13;
        self.noise_state ^= self.noise_state >> 17;
        self.noise_state ^= self.noise_state << 5;
        (self.noise_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn finished(&self) -> bool {
        self.release_gain < 0.000_08
            || (self.age_samples > 2_880_000
                && self.envelope.iter().copied().fold(0.0, f32::max) < 0.000_08)
    }
}

fn pseudo_phase(seed: u64, partial: usize) -> f64 {
    let mixed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((partial as u64 + 1).wrapping_mul(1_442_695_040_888_963_407));
    (mixed as f64 / u64::MAX as f64) * TAU
}

fn rotate(sine: &mut f64, cosine: &mut f64, step_sine: f64, step_cosine: f64) {
    let next_sine = *sine * step_cosine + *cosine * step_sine;
    let next_cosine = *cosine * step_cosine - *sine * step_sine;
    *sine = next_sine;
    *cosine = next_cosine;
}

struct StereoSoundboard {
    left: [DampedComb; 4],
    right: [DampedComb; 4],
}

impl StereoSoundboard {
    fn new(sample_rate: f32) -> Self {
        let scale = sample_rate / 44_100.0;
        let delay = |samples: usize| ((samples as f32 * scale).round() as usize).max(1);
        Self {
            left: [
                DampedComb::new(delay(1_117), 0.79, 0.28),
                DampedComb::new(delay(1_351), 0.77, 0.31),
                DampedComb::new(delay(1_481), 0.75, 0.34),
                DampedComb::new(delay(1_607), 0.73, 0.37),
            ],
            right: [
                DampedComb::new(delay(1_139), 0.79, 0.28),
                DampedComb::new(delay(1_373), 0.77, 0.31),
                DampedComb::new(delay(1_523), 0.75, 0.34),
                DampedComb::new(delay(1_631), 0.73, 0.37),
            ],
        }
    }

    fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let center = (left + right) * 0.5;
        let mut wet_left = 0.0;
        let mut wet_right = 0.0;
        for comb in &mut self.left {
            wet_left += comb.process(center + left * 0.12);
        }
        for comb in &mut self.right {
            wet_right += comb.process(center + right * 0.12);
        }
        (wet_left * 0.075, wet_right * 0.075)
    }
}

struct DampedComb {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damping: f32,
    filtered: f32,
}

impl DampedComb {
    fn new(length: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; length],
            index: 0,
            feedback,
            damping,
            filtered: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.index];
        self.filtered = delayed * (1.0 - self.damping) + self.filtered * self.damping;
        self.buffer[self.index] = input + self.filtered * self.feedback;
        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
        }
        delayed
    }
}

/// RBJ (Audio EQ Cookbook) biquad with running state.
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn unity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn low_shelf(cutoff_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w = std::f32::consts::TAU * cutoff_hz / sample_rate;
        let cw = w.cos();
        let sw = w.sin();
        let s_a = a.sqrt();
        let alpha = sw / (2.0 * s_a);
        let b0 = a * ((a + 1.0) - (a - 1.0) * cw + 2.0 * s_a * alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cw - 2.0 * s_a * alpha);
        let a0 = (a + 1.0) + (a - 1.0) * cw + 2.0 * s_a * alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw);
        let a2 = (a + 1.0) + (a - 1.0) * cw - 2.0 * s_a * alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn high_shelf(cutoff_hz: f32, gain_db: f32, sample_rate: f32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w = std::f32::consts::TAU * cutoff_hz / sample_rate;
        let cw = w.cos();
        let sw = w.sin();
        let s_a = a.sqrt();
        let alpha = sw / (2.0 * s_a);
        let b0 = a * ((a + 1.0) + (a - 1.0) * cw + 2.0 * s_a * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cw - 2.0 * s_a * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cw + 2.0 * s_a * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw);
        let a2 = (a + 1.0) - (a - 1.0) * cw - 2.0 * s_a * alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn peaking(cutoff_hz: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w = std::f32::consts::TAU * cutoff_hz / sample_rate;
        let cw = w.cos();
        let sw = w.sin();
        let alpha = sw / (2.0 * q);
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cw;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cw;
        let a2 = 1.0 - alpha / a;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let w = x - self.a1 * self.z1 - self.a2 * self.z2;
        let y = self.b0 * w + self.b1 * self.z1 + self.b2 * self.z2;
        self.z2 = self.z1;
        self.z1 = w;
        y
    }
}

/// Tone-matching chain applied to the final stereo output: tames the low end,
/// lifts the midrange, and rolls off the top so the band balance follows the
/// reference piano rather than a raw partial stack.
struct StereoEq {
    left: [Biquad; 5],
    right: [Biquad; 5],
}

impl StereoEq {
    fn new(sample_rate: f32) -> Self {
        // Filters whose corner sits above ~0.48 * sample rate would be
        // meaningless at low render rates, so they collapse to unity there.
        let make = |cutoff_hz: f32, build: fn(f32, f32) -> Biquad| -> Biquad {
            if cutoff_hz < sample_rate * 0.48 {
                build(cutoff_hz, sample_rate)
            } else {
                Biquad::unity()
            }
        };
        let specs: [fn(f32, f32) -> Biquad; 5] = [
            |fc, sr| Biquad::low_shelf(fc, -12.0, sr),
            |fc, sr| Biquad::peaking(fc, 4.0, 1.1, sr),
            |fc, sr| Biquad::peaking(fc, 5.0, 1.2, sr),
            |fc, sr| Biquad::peaking(fc, -16.0, 0.6, sr),
            |fc, sr| Biquad::high_shelf(fc, -14.0, sr),
        ];
        let cutoffs = [55.0_f32, 700.0, 1800.0, 13_000.0, 6000.0];
        Self {
            left: std::array::from_fn(|index| make(cutoffs[index], specs[index])),
            right: std::array::from_fn(|index| make(cutoffs[index], specs[index])),
        }
    }

    fn process(&mut self, left: f32, right: f32) -> (f32, f32) {
        let mut out_left = left;
        for biquad in &mut self.left {
            out_left = biquad.process(out_left);
        }
        let mut out_right = right;
        for biquad in &mut self.right {
            out_right = biquad.process(out_right);
        }
        (out_left, out_right)
    }
}

fn soft_limit(sample: f32) -> f32 {
    sample / (1.0 + sample.abs() * 0.72)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PianoInputSource, PianoNote};
    use std::fs;
    use std::time::Duration;

    #[test]
    fn attack_is_stereo_finite_and_audible() {
        let mut synth = PianoSynth::new(48_000.0);
        synth.command(Command::Attack {
            voice_id: 1,
            midi_note: 60,
            frequency_hz: 261.625_55,
            velocity: 0.8,
            pressure: None,
        });
        let mut energy = 0.0;
        for _ in 0..4_800 {
            let (left, right) = synth.next_sample();
            assert!(left.is_finite() && right.is_finite());
            assert!(left.abs() <= 1.0 && right.abs() <= 1.0);
            energy += left * left + right * right;
        }
        assert!(energy > 0.01);
        assert_eq!(synth.voices.len(), 1);
    }

    #[test]
    fn attack_fades_the_hammer_transient_in_from_silence() {
        let mut voice = PianoVoice::new(1, 60, 261.625_55, 1.0, None, 48_000.0);

        let (left, right) = voice.next_sample();

        assert_eq!((left, right), (0.0, 0.0));
        assert_ne!(voice.previous_noise, 0.0, "the hammer noise path ran");
    }

    #[test]
    fn release_damps_the_matching_voice() {
        let mut synth = PianoSynth::new(48_000.0);
        synth.command(Command::Attack {
            voice_id: 7,
            midi_note: 69,
            frequency_hz: 440.0,
            velocity: 0.7,
            pressure: None,
        });
        synth.command(Command::Release {
            voice_id: 7,
            velocity: 0.5,
        });
        assert!(synth.voices[0].released);
        for _ in 0..500_000 {
            synth.next_sample();
            if synth.voices.is_empty() {
                break;
            }
        }
        assert!(synth.voices.is_empty());
    }

    #[test]
    fn score_sized_chords_are_not_cut_off_at_the_old_voice_limit() {
        let mut synth = PianoSynth::new(48_000.0);
        for voice_id in 1..=96 {
            let midi_note = 21 + (voice_id % 88) as u8;
            synth.command(Command::Attack {
                voice_id,
                midi_note,
                frequency_hz: PianoNote::from_midi(midi_note).frequency_hz,
                velocity: 0.6,
                pressure: None,
            });
        }
        assert_eq!(synth.voices.len(), 96);
    }

    #[test]
    fn metronome_clicks_land_on_the_beat() {
        let sample_rate = 48_000.0;
        let mut synth = PianoSynth::new(sample_rate);
        synth.command(Command::Metronome { bpm: 120.0 });

        // 120 bpm at 48 kHz is one beat every 24 000 samples; render two
        // beats and split the energy on and off the beat. The ≈ 80 ms click
        // tail stays inside its own quarter of the render.
        let mut on_beat = 0.0f32;
        let mut off_beat = 0.0f32;
        for frame in 0..48_000 {
            let (left, right) = synth.next_sample();
            let energy = left * left + right * right;
            if frame % 24_000 < 12_000 {
                on_beat += energy;
            } else {
                off_beat += energy;
            }
        }
        assert!(
            on_beat > 1e-4,
            "the clicks should be audible, got {on_beat}"
        );
        assert!(
            off_beat < 1e-5,
            "beats should stay silent between clicks, got {off_beat}"
        );
        assert!(synth.clicks.is_empty(), "clicks decay away");
    }

    #[test]
    fn metronome_keeps_the_beat_across_many_cycles() {
        let sample_rate = 48_000.0;
        let mut synth = PianoSynth::new(sample_rate);
        synth.command(Command::Metronome { bpm: 60.0 });

        // 60 bpm is one beat per second; over five beats the clicks must not
        // drift, so each lands in its own one-second window.
        let mut windows = [0.0f32; 5];
        for frame in 0..240_000 {
            let (left, right) = synth.next_sample();
            windows[(frame / 48_000).min(4)] += left * left + right * right;
        }
        for (index, window) in windows.iter().enumerate() {
            assert!(*window > 1e-4, "beat {index} should click, got {window}");
        }
    }

    #[test]
    fn metronome_rejects_invalid_tempi() {
        let mut synth = PianoSynth::new(48_000.0);
        synth.command(Command::Metronome { bpm: 0.0 });
        assert!(synth.metronome.is_none(), "zero bpm stays silent");
        synth.command(Command::Metronome { bpm: f32::NAN });
        assert!(synth.metronome.is_none(), "a non-finite bpm stays silent");
    }

    #[test]
    fn eq_chain_boosts_the_midrange_and_tames_the_top() {
        let sample_rate = 48_000.0;

        let drive = |frequency_hz: f32| -> f32 {
            let mut eq = StereoEq::new(sample_rate);
            let settle_samples = (sample_rate / 100.0) as usize;
            let mut peak = 0.0f32;
            for sample in 0..(sample_rate * 2.0) as usize {
                let x = (std::f32::consts::TAU * frequency_hz * sample as f32 / sample_rate).sin()
                    * 0.5;
                let (out, _) = eq.process(x, 0.0);
                // Ignore the first 10 ms while the biquad states settle.
                if sample >= settle_samples {
                    peak = peak.max(out.abs());
                }
            }
            peak
        };

        let midrange = drive(700.0);
        let top = drive(13_000.0);

        assert!(
            midrange > 0.5 * 1.4,
            "700 Hz should be boosted, got {midrange}"
        );
        assert!(
            top < 0.5 * 0.12,
            "13 kHz should be heavily attenuated, got {top}"
        );
    }

    #[test]
    fn recovered_buffer_xruns_do_not_become_ui_errors() {
        let runtime_error = Mutex::new(None);
        handle_stream_error(cpal::StreamError::BufferUnderrun, &runtime_error);
        assert_eq!(*runtime_error.lock().unwrap(), None);

        handle_stream_error(cpal::StreamError::StreamInvalidated, &runtime_error);
        assert!(runtime_error
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|error| error.contains("audio stream stopped")));
    }

    #[test]
    fn offline_renderer_writes_stereo_pcm_s16le_and_schedules_every_event() {
        let nonce = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let input = std::env::temp_dir().join(format!("black-hole-play-{nonce}.bhs"));
        let output = std::env::temp_dir().join(format!("black-hole-play-{nonce}.wav"));
        // A C4 struck at t=0, released 10 ms later, looping every 20 ms.
        let score = "\
format bhs-score-v1
ticks_per_second 1000
loop_ticks 20
0 10 C4 102 51
";
        fs::write(&input, score).unwrap();

        let report =
            render_piano_score_to_wav(&input, &output, 8_000, Duration::from_millis(10)).unwrap();
        assert_eq!(report.event_count, 2);
        assert_eq!(report.frames, 240);

        let reader = hound::WavReader::open(&output).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 2);
        assert_eq!(spec.sample_rate, 8_000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        assert_eq!(reader.duration(), 240);

        fs::remove_file(input).unwrap();
        fs::remove_file(output).unwrap();
    }

    #[test]
    #[ignore = "requires an audible default output device"]
    fn plays_c_major_through_the_default_cpal_device() {
        let engine = PianoAudioEngine::new().expect("default cpal output should open");
        for (voice_id, midi_note, velocity) in [(1, 60, 0.72), (2, 64, 0.64), (3, 67, 0.78)] {
            engine.perform(PianoEvent {
                sequence: voice_id,
                timestamp: Duration::ZERO,
                voice_id,
                note: PianoNote::from_midi(midi_note),
                action: PianoAction::Attack {
                    velocity,
                    pressure: None,
                },
                source: PianoInputSource::Mouse,
            });
        }
        std::thread::sleep(Duration::from_millis(900));
        for (voice_id, midi_note) in [(1, 60), (2, 64), (3, 67)] {
            engine.perform(PianoEvent {
                sequence: voice_id + 3,
                timestamp: Duration::from_millis(900),
                voice_id,
                note: PianoNote::from_midi(midi_note),
                action: PianoAction::Release {
                    velocity: 0.35,
                    held_for: Duration::from_millis(900),
                },
                source: PianoInputSource::Mouse,
            });
        }
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(engine.error(), None);
    }

    /// Pure-math full-score verification (no audio output): renders the whole
    /// `score.bhs` through [`PianoSynth`] at 48 kHz and checks that the band
    /// balance matches the reference-piano target calibrated in the offline
    /// simulator. The expected values are the tuned pipeline's aggregates on
    /// the 0.499 s / 23952-sample window convention; this test uses a
    /// 16384-sample hop (power of two for the radix-2 FFT) and shifts each
    /// window's band level by `10*log10(23952/16384)` to stay on that scale.
    #[test]
    #[ignore = "runs the full score through the synth; needs /home/chip/Desktop/score.bhs"]
    fn full_score_sim_band_balance() {
        use crate::piano::piano_score::load_score;

        const SR: f32 = 48_000.0;
        const N: usize = 13_532_880; // same length as the reference render
        const HOP: usize = 16_384;
        let scale_shift = (23_952.0_f64 / 16_384.0_f64).log10() * 10.0;

        let score = load_score(std::path::Path::new("/home/chip/Desktop/score.bhs"))
            .expect("score.bhs must exist for this test");

        // Pre-schedule every command at its render frame.
        let mut pending: Vec<(usize, Command)> = score
            .events
            .iter()
            .map(|event| {
                let frame = (event.timestamp.as_secs_f64() * SR as f64).round() as usize;
                (frame, command_from_event(*event))
            })
            .collect();
        pending.sort_by_key(|(frame, _)| *frame);

        fn fft(x: &mut [f64], y: &mut [f64]) {
            let n = x.len();
            debug_assert_eq!(n & (n - 1), 0);
            let mut j = 0usize;
            for i in 1..n {
                let mut bit = n >> 1;
                while j & bit != 0 {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                if i < j {
                    x.swap(i, j);
                    y.swap(i, j);
                }
            }
            let mut len = 2;
            while len <= n {
                let ang = -2.0 * std::f64::consts::PI / len as f64;
                let (wr, wi) = (ang.cos(), ang.sin());
                let mut i = 0;
                while i < n {
                    let (mut cr, mut ci) = (1.0, 0.0);
                    for k in 0..len / 2 {
                        let (ur, ui) = (x[i + k], y[i + k]);
                        let vr = x[i + k + len / 2] * cr - y[i + k + len / 2] * ci;
                        let vi = x[i + k + len / 2] * ci + y[i + k + len / 2] * cr;
                        x[i + k] = ur + vr;
                        y[i + k] = ui + vi;
                        x[i + k + len / 2] = ur - vr;
                        y[i + k + len / 2] = ui - vi;
                        let nr = cr * wr - ci * wi;
                        ci = cr * wi + ci * wr;
                        cr = nr;
                    }
                    i += len;
                }
                len <<= 1;
            }
        }

        // Symmetric Hann, matching numpy's `np.hanning`.
        let win: Vec<f64> = (0..HOP)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (HOP as f64 - 1.0)).cos())
            .collect();
        let bands: [(f64, f64); 7] = [
            (20.0, 54.0),
            (54.0, 148.0),
            (148.0, 403.0),
            (403.0, 1100.0),
            (1100.0, 3000.0),
            (3000.0, 8100.0),
            (8100.0, 22_000.0),
        ];

        let mut synth = PianoSynth::new(SR);
        let mut ev_idx = 0usize;
        let mut band_sum = [0.0f64; 7];
        let mut band_cnt = [0u32; 7];
        let mut cent_sum = 0.0f64;
        let mut cent_cnt = 0u32;

        for window in 0..N / HOP {
            let mut xl = vec![0.0f64; HOP];
            let mut xr = vec![0.0f64; HOP];
            for k in 0..HOP {
                let frame = window * HOP + k;
                while ev_idx < pending.len() && pending[ev_idx].0 <= frame {
                    synth.command(pending[ev_idx].1);
                    ev_idx += 1;
                }
                let (left, right) = synth.next_sample();
                xl[k] = ((left.clamp(-1.0, 1.0) * 32_767.0).round() as i16) as f64 * win[k];
                xr[k] = ((right.clamp(-1.0, 1.0) * 32_767.0).round() as i16) as f64 * win[k];
            }
            let mut yl = vec![0.0f64; HOP];
            fft(&mut xl, &mut yl);
            let mut yr = vec![0.0f64; HOP];
            fft(&mut xr, &mut yr);

            let df = SR as f64 / HOP as f64;
            let mut total = 0.0f64;
            let mut centroid_num = 0.0f64;
            for k in 0..=HOP / 2 {
                let p = (xl[k] * xl[k] + yl[k] * yl[k] + xr[k] * xr[k] + yr[k] * yr[k]) * 0.25;
                total += p;
                centroid_num += p * (k as f64 * df);
            }
            for (band, &(lo, hi)) in bands.iter().enumerate() {
                let k0 = (lo / df).ceil() as usize;
                let k1 = (((hi - 1e-9) / df).floor() as usize + 1).min(HOP / 2 + 1);
                let mut power = 0.0f64;
                for k in k0..k1 {
                    power += xl[k] * xl[k] + yl[k] * yl[k] + xr[k] * xr[k] + yr[k] * yr[k];
                }
                power *= 0.25;
                let db = (if power > 0.0 {
                    10.0 * power.log10()
                } else {
                    -120.0
                }) + scale_shift;
                if db > -50.0 {
                    band_sum[band] += db;
                    band_cnt[band] += 1;
                }
            }
            if total > 0.0 {
                cent_sum += centroid_num / total;
                cent_cnt += 1;
            }
        }

        let expected: [f64; 7] = [107.0, 139.7, 144.7, 145.7, 140.9, 125.8, 101.6];
        let centroid = cent_sum / cent_cnt as f64;
        eprintln!("full-score sim: centroid {centroid:.1} Hz");
        for band in 0..7 {
            let got = band_sum[band] / band_cnt[band] as f64;
            eprintln!(
                "  band{}: {got:.1} dB (expected ~{} +/- 5)",
                band + 1,
                expected[band]
            );
        }
        for band in 0..7 {
            let got = band_sum[band] / band_cnt[band] as f64;
            assert!(
                (got - expected[band]).abs() < 5.0,
                "band {} drifted: got {got:.1} dB, expected ~{} dB",
                band + 1,
                expected[band]
            );
        }
        assert!(
            (centroid - 599.6).abs() < 80.0,
            "centroid drifted: got {centroid:.1} Hz, expected ~600 Hz"
        );
    }
}
