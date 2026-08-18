//! Render a black-hole-beam `bhs-score-v1` piano score to a 16-bit PCM WAV
//! file.

use std::path::PathBuf;
use std::time::Duration;

use black_hole_beam::render_piano_score_to_wav;
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "black-hole-play",
    version,
    about = "Render a black-hole-beam piano score to stereo pcm_s16le WAV"
)]
struct Args {
    /// Piano score to render (bhs-score-v1 text, conventionally .bhs)
    input: PathBuf,

    /// WAV file to create
    output: PathBuf,

    /// Output sample rate in hertz
    #[arg(long, default_value_t = 48_000)]
    sample_rate: u32,

    /// Soundboard and string decay rendered after the end of the score
    #[arg(long, default_value_t = 4.0)]
    tail_seconds: f64,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if !args.tail_seconds.is_finite() || args.tail_seconds < 0.0 {
        return Err("tail-seconds must be finite and non-negative".to_string());
    }
    let report = render_piano_score_to_wav(
        &args.input,
        &args.output,
        args.sample_rate,
        Duration::from_secs_f64(args.tail_seconds),
    )?;
    eprintln!(
        "rendered {} events, {:.3}s at {} Hz to {}",
        report.event_count,
        report.frames as f64 / f64::from(report.sample_rate),
        report.sample_rate,
        args.output.display()
    );
    Ok(())
}
