#!/usr/bin/env python3
"""Move the piano impl-BeamApp methods from app/mod.rs into app/piano.rs."""

mod_lines = open("src/app/mod.rs").read().split("\n")

# 1-indexed: block is lines 602..962 inclusive (piano_keyboard .. piano_log_line)
assert mod_lines[601].startswith('    #[cfg(feature = "piano")]')
assert mod_lines[602] == "    fn piano_keyboard(&self) -> Element<'_, Message> {"
assert mod_lines[961] == "    }"
assert mod_lines[962] == ""
assert mod_lines[963].startswith("    fn cell_graph")

block = mod_lines[601:962]  # 0-indexed slice of lines 602..962
# Remove block plus the trailing blank line (index 962).
del mod_lines[601:963]
open("src/app/mod.rs", "w").write("\n".join(mod_lines))

piano = open("src/app/piano.rs").read()

# Fix app/piano.rs header imports.
old_header = """use crate::piano::piano_audio::PianoAudioEngine;
use crate::piano::score_text::{self, BhsScore};
"""
new_header = """use crate::piano::score_text;
"""
assert piano.count(old_header) == 1
piano = piano.replace(old_header, new_header)

# Append the moved methods in their own impl block.
piano = piano.rstrip("\n") + "\n\nimpl BeamApp {\n" + "\n".join(block) + "\n}\n"
open("src/app/piano.rs", "w").write(piano)

# Fix app/mod.rs header imports: add PianoMessage and the app::piano items.
mod_text = open("src/app/mod.rs").read()
old_mod_header = """#[cfg(feature = "piano")]
use crate::piano::piano_audio::PianoAudioEngine;
#[cfg(feature = "piano")]
use crate::piano::piano_score::{PianoScorePlayback, SCORE_TICK_INTERVAL};
"""
new_mod_header = """#[cfg(feature = "piano")]
use self::piano::{ActivePianoNote, PianoInputId, PianoStrikeVisual};
#[cfg(feature = "piano")]
use crate::piano::piano_audio::PianoAudioEngine;
#[cfg(feature = "piano")]
use crate::piano::piano_score::{PianoScorePlayback, SCORE_TICK_INTERVAL};
#[cfg(feature = "piano")]
use crate::piano::PianoMessage;
"""
assert mod_text.count(old_mod_header) == 1
mod_text = mod_text.replace(old_mod_header, new_mod_header)
open("src/app/mod.rs", "w").write(mod_text)

print("moved piano methods; updated headers")
