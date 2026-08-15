#![allow(dead_code, unused_imports, clippy::manual_async_fn)]

#[path = "../common/mod.rs"]
mod common;

mod black_dwarf;
mod dark_star;
mod diamond_dog;
mod red_dwarf;
mod sun_dog;
mod white_dwarf;

#[cfg(not(test))]
pub(crate) use black_dwarf::run_beam_black_dwarf;
#[cfg(not(test))]
pub(crate) use black_dwarf::run_continuous_black_dwarf;
#[cfg(not(test))]
pub(crate) use dark_star::run_beam_dark_star;
#[cfg(not(test))]
pub(crate) use sun_dog::run_beam;

#[cfg(test)]
fn run_beam_example(example: &str) {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = std::process::Command::new(cargo);
    command.current_dir(env!("CARGO_MANIFEST_DIR")).args([
        "run",
        "--quiet",
        "--no-default-features",
    ]);

    // This test launches a second Cargo process so that the UI can own its main
    // thread. Cargo does not propagate the parent invocation's profile or
    // features to that process, so mirror the active build explicitly.
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }

    let features = [
        (cfg!(feature = "cuda"), "cuda"),
        (cfg!(feature = "metal"), "metal"),
        (cfg!(feature = "qwen35_0p8b"), "qwen35_0p8b"),
        (cfg!(feature = "qwen35_2b"), "qwen35_2b"),
        (cfg!(feature = "qwen35_4b"), "qwen35_4b"),
        (cfg!(feature = "qwen35_9b"), "qwen35_9b"),
        (cfg!(feature = "qwen35_27b"), "qwen35_27b"),
        (cfg!(feature = "qwen38_27b"), "qwen38_27b"),
    ]
    .into_iter()
    .filter_map(|(enabled, feature)| enabled.then_some(feature))
    .collect::<Vec<_>>()
    .join(",");
    if !features.is_empty() {
        command.args(["--features", &features]);
    }

    let status = command
        .args(["--example", example])
        .status()
        .unwrap_or_else(|error| panic!("{example} example should launch: {error}"));

    assert!(status.success(), "{example} example exited with {status}");
}
