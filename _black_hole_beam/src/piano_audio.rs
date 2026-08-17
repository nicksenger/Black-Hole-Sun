//! Low-latency physically inspired piano synthesis.

use std::f64::consts::TAU;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};

use crate::{PianoAction, PianoEvent};

const PARTIAL_COUNT: usize = 16;
const MAX_POLYPHONY: usize = 64;
const MASTER_GAIN: f32 = 0.19;

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
}

/// Owns the cpal stream. Dropping this value stops playback.
pub(crate) struct PianoAudioEngine {
    command_tx: SyncSender<Command>,
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
        let (command_tx, command_rx) = mpsc::sync_channel(512);
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
        let command = match event.action {
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
        };
        let _ = self.command_tx.try_send(command);
    }

    pub(crate) fn error(&self) -> Option<String> {
        self.runtime_error
            .lock()
            .ok()
            .and_then(|error| error.clone())
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
        move |error| {
            if let Ok(mut current) = runtime_error.lock() {
                *current = Some(format!("audio stream stopped: {error}"));
            }
        },
        None,
    )
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
}

impl PianoSynth {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            voices: Vec::with_capacity(MAX_POLYPHONY),
            soundboard: StereoSoundboard::new(sample_rate),
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

        let (body_left, body_right) = self.soundboard.process(left, right);
        let left = soft_limit((left + body_left) * MASTER_GAIN);
        let right = soft_limit((right + body_right) * MASTER_GAIN);
        (left, right)
    }
}

struct PianoVoice {
    voice_id: u64,
    midi_note: u8,
    sample_rate: f32,
    phase: [f64; PARTIAL_COUNT],
    phase_increment: [f64; PARTIAL_COUNT],
    beat_phase: [f64; PARTIAL_COUNT],
    beat_increment: [f64; PARTIAL_COUNT],
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
    hammer_phase: f32,
    hammer_phase_increment: f32,
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
        let inharmonicity = 0.000_04 * 2.0_f32.powf(key_position * 6.4);
        let hammer_position = 0.115 + 0.035 * key_position;
        let spectral_rolloff = 0.68 + 1.25 * (1.0 - velocity) + 0.28 * (1.0 - pressure);
        let base_decay_seconds = 21.0 - 16.5 * key_position.powf(0.72);
        let velocity_gain = velocity.powf(0.72) * (0.88 + pressure * 0.12);

        let mut phase = [0.0; PARTIAL_COUNT];
        let mut phase_increment = [0.0; PARTIAL_COUNT];
        let mut beat_phase = [0.0; PARTIAL_COUNT];
        let mut beat_increment = [0.0; PARTIAL_COUNT];
        let mut amplitude = [0.0; PARTIAL_COUNT];
        let envelope = [1.0; PARTIAL_COUNT];
        let mut natural_decay = [0.0; PARTIAL_COUNT];

        for partial in 0..PARTIAL_COUNT {
            let harmonic = (partial + 1) as f32;
            let partial_frequency =
                frequency_hz * harmonic * (1.0 + inharmonicity * harmonic * harmonic).sqrt();
            phase[partial] = pseudo_phase(voice_id, partial);
            phase_increment[partial] = f64::from(partial_frequency / sample_rate) * TAU;
            beat_phase[partial] = pseudo_phase(voice_id.wrapping_add(17), partial);
            let unison_hz = 0.18 + 0.018 * harmonic + 0.35 * key_position;
            beat_increment[partial] = f64::from(unison_hz / sample_rate) * TAU;

            let hammer_node = (std::f32::consts::PI * harmonic * hammer_position)
                .sin()
                .abs()
                .powf(0.45);
            amplitude[partial] = velocity_gain * hammer_node / harmonic.powf(spectral_rolloff);
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
            phase,
            phase_increment,
            beat_phase,
            beat_increment,
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
            hammer_phase: 0.0,
            hammer_phase_increment: TAU as f32 * (1_150.0 + 14.0 * f32::from(midi_note))
                / sample_rate,
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
            (1.25 - 0.82 * key_position) * (1.12 - 0.52 * release_velocity.clamp(0.0, 1.0));
        self.release_decay = (-1.0 / (release_seconds * self.sample_rate)).exp();
    }

    fn next_sample(&mut self) -> (f32, f32) {
        let attack = (self.age_samples as f32 / self.attack_samples).clamp(0.0, 1.0);
        let attack = attack * attack * (3.0 - 2.0 * attack);
        let mut string = 0.0;

        for partial in 0..PARTIAL_COUNT {
            let unison = 1.0 + 0.055 * self.beat_phase[partial].sin() as f32;
            string += self.phase[partial].sin() as f32
                * self.amplitude[partial]
                * self.envelope[partial]
                * unison;
            self.phase[partial] = wrap_phase(self.phase[partial] + self.phase_increment[partial]);
            self.beat_phase[partial] =
                wrap_phase(self.beat_phase[partial] + self.beat_increment[partial]);
            self.envelope[partial] *= self.natural_decay[partial];
        }

        let noise = self.next_noise();
        let hammer_noise = (noise - self.previous_noise) * self.hammer_envelope;
        self.previous_noise = noise;
        let hammer_knock = self.hammer_phase.sin() * self.hammer_envelope * 0.55;
        self.hammer_phase = (self.hammer_phase + self.hammer_phase_increment) % TAU as f32;
        self.hammer_envelope *= self.hammer_decay;
        if self.released {
            self.release_gain *= self.release_decay;
        }
        self.age_samples += 1;

        let sample = (string * attack + hammer_noise + hammer_knock) * self.release_gain;
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

fn wrap_phase(phase: f64) -> f64 {
    if phase >= TAU {
        phase % TAU
    } else {
        phase
    }
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

fn soft_limit(sample: f32) -> f32 {
    sample / (1.0 + sample.abs() * 0.72)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PianoInputSource, PianoNote};
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
}
