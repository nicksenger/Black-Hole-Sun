#!/usr/bin/env python3
"""Write the new modularized src/lib.rs, preserving test modules verbatim."""

lines = open("src/lib.rs").read().split("\n")

def seg(a, b):
    return "\n".join(lines[a - 1 : b])

docs = seg(1, 12)
tests = seg(3374, 5061)          # #[cfg(test)] mod tests { ... } (no trailing brace line)
warp_tests = seg(5063, 5178)     # #[cfg(test)] mod warp_fetch_diagnostics { ... }

header = """mod app;
mod builder;
mod client;
mod flow;
mod graph;
mod labels;
mod live;
#[cfg(feature = "piano")]
mod piano;
mod model;
mod style;
mod subpanel;
mod visual;

pub use builder::{view, view_live, BeamBuilder};
#[cfg(feature = "piano")]
pub use builder::PianoLog;
pub use flow::{BlackHoleSunAnimal, BlackHoleSunFlow};

#[cfg(feature = "piano")]
pub use piano::score_text;
#[cfg(feature = "piano")]
pub use piano::{PianoAction, PianoEvent, PianoInputSource, PianoNote};
#[cfg(feature = "piano")]
pub use piano::piano_audio::{render_piano_score_to_wav, PianoRenderReport};

"""

tests_imports = """
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use black_hole_flux::sun::{SunAppearance, SunNodeState};
    use black_hole_flux::{FusionSeed, FusionState, Ray};
    use iced::keyboard;
    use iced::time::Instant;
    use iced::Color;
    use jungle_sdk::Animal;
    use uuid::Uuid;

    use crate::app::piano::{piano_log_ticks, PianoInputId, PianoStrikeVisual};
    use crate::app::{BeamApp, Message, APPEARANCE_INTERVAL};
    use crate::builder::{BeamConfig, BeamLayout};
    use crate::labels::{animal_label_key, short_type_name, warp_boundary_label};
    use crate::live::{fetch_warp_appearances, LiveAppearanceSnapshot, LiveConfig};
    use crate::model::{BeamModel, CellDefinition};
    use crate::piano::piano_score::PianoScorePlayback;
    use crate::piano::score_text::BhsScore;
    use crate::piano::PianoPointerSource;
    use crate::visual::{
        lerp_color, node_style_colors, warp_node_style_colors, CellVisualState, NodeProgress,
        NodeStateVisual, COLOR_FADE_DURATION, MAX_PENDING_PHASES, MIN_COLOR_STATE_DURATION,
    };
"""

warp_tests_imports = """
    use std::sync::Arc;

    use black_hole_flux::sun::{SunAppearance, SunNodeState};
    use uuid::Uuid;

    use crate::live::{fetch_warp_appearances, LiveConfig};
"""

# Insert imports right after `use super::*;` in each test module.
def insert_after(body, marker, extra):
    parts = body.split(marker, 1)
    assert len(parts) == 2, f"marker not found: {marker!r}"
    return parts[0] + marker + extra + parts[1]

tests = insert_after(tests, "    use super::*;", tests_imports)
warp_tests = insert_after(warp_tests, "    use super::*;", warp_tests_imports)

# Merge the now-duplicated SunNodeAppearance import in warp_fetch_diagnostics.
warp_tests = warp_tests.replace(
    "    use black_hole_flux::sun::{SunAppearance, SunNodeState};\n    use uuid::Uuid;\n\n    use crate::live::{fetch_warp_appearances, LiveConfig};\n    use black_hole_flux::sun::SunNodeAppearance;",
    "    use black_hole_flux::sun::{SunAppearance, SunNodeAppearance, SunNodeState};\n    use uuid::Uuid;\n\n    use crate::live::{fetch_warp_appearances, LiveConfig};",
)

out = docs + "\n\n" + header + tests + "\n\n" + warp_tests + "\n"
open("src/lib.rs", "w").write(out)
print(f"wrote src/lib.rs ({len(out.splitlines())} lines)")
