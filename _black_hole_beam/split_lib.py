#!/usr/bin/env python3
"""Split src/lib.rs of black-hole-beam into logical modules."""
import io, os

SRC = "src/lib.rs"
lines = open(SRC).read().split("\n")
# lines is 0-indexed; helper to grab inclusive 1-indexed ranges
def seg(a, b):
    return "\n".join(lines[a - 1 : b])

SEGS = {
    "docs": (1, 12),
    "subpanel": (79, 98),
    "glyph": (100, 158),
    "pianolog": (159, 168),
    "builder_struct": (170, 206),
    "beamlayout": (208, 212),
    "builder_default": (214, 238),
    "builder_impl": (240, 434),
    "view_fn": (436, 443),
    "view_live_fn": (445, 451),
    "flow": (453, 522),
    "beamconfig": (524, 545),
    "liveconfig": (547, 551),
    "shared_client_struct": (553, 562),
    "shared_client_impl": (564, 760),
    "subpanel_state": (762, 767),
    "celldef": (769, 804),
    "beammodel": (806, 1159),
    "run_beam": (1161, 1177),
    "piano_computer_key": (1179, 1191),
    "piano_log_const": (1192, 1195),
    "piano_log_ticks": (1197, 1204),
    "piano_log_velocity": (1206, 1211),
    "message": (1213, 1231),
    "piano_input_id": (1233, 1239),
    "active_piano_note": (1241, 1248),
    "strike_visual": (1250, 1303),
    "live_snapshot": (1305, 1316),
    "node_state_visual": (1318, 1331),
    "node_progress": (1333, 1346),
    "node_style_colors_struct": (1348, 1353),
    "visual_fns": (1355, 1488),
    "cell_visual_state": (1490, 1667),
    "model_display_changed": (1669, 1686),
    "beamapp_struct": (1688, 1745),
    "beamapp_impl": (1747, 2923),
    "build_sun_graph": (2925, 3048),
    "appearance_task": (3050, 3052),
    "fetch_appearance": (3054, 3077),
    "fetch_child_rays": (3079, 3105),
    "max_warp_depth": (3107, 3109),
    "fetch_warp_appearances": (3111, 3156),
    "fetch_sun_appearance": (3158, 3175),
    "style_fns": (3177, 3293),
    "color_utils": (3295, 3316),
    "labels": (3318, 3372),
    "tests": (3374, 5061),
    "warp_tests": (5063, 5178),
}

def get(name):
    a, b = SEGS[name]
    return seg(a, b)

FILES = {}

FILES["src/builder.rs"] = """//! The public [`BeamBuilder`] API for static and live Black Hole Sun views.

#[cfg(feature = "piano")]
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use iced_sugiyama::motion::easing::Easing;
use jungle_sdk::{Animal, JungleClient};
use uuid::Uuid;

#[cfg(feature = "piano")]
use crate::piano::score_text::BhsScore;
#[cfg(feature = "piano")]
use crate::piano::PianoEvent;
use crate::app::run_beam;
use crate::client::SharedJungleClient;
use crate::flow::{BlackHoleSunAnimal, BlackHoleSunFlow};
use crate::live::LiveConfig;
use crate::model::BeamModel;
use crate::subpanel::{build_subpanel_viewer, SubpanelConfig};

const DEFAULT_WINDOW_WIDTH: f32 = 1440.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 900.0;

""" + get("pianolog") + "\n\n" + get("builder_struct") + "\n\n" + get("beamlayout") + "\n\n" + get("builder_default") + "\n\n" + get("builder_impl") + "\n\n" + get("view_fn") + "\n\n" + get("view_live_fn") + "\n"

FILES["src/flow.rs"] = """//! Marker traits describing which Jungle animals and structural flows can be
//! rendered as Black Hole Sun views.

use black_hole_flux::sun::{
    BinarySunStep, NodeIdsFromList, Sun, SunAppearance, SunNode, UnarySunStep,
};
use black_hole_flux::{FusionFlow, FusionSeed, FusionState};
use jungle_sdk::{Animal, AnimalIdValue, JourneyAstSource, Observe};
use typenum::Unsigned;

use crate::model::CellDefinition;

""" + get("flow") + "\n"

FILES["src/client.rs"] = """//! A [`JungleClient`] wrapper shared between the main view and subpanels.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use jungle_sdk::JungleClient;
use uuid::Uuid;

""" + get("shared_client_struct") + "\n\n" + get("shared_client_impl") + "\n"

FILES["src/live.rs"] = """//! Polling and decoding of live Jungle appearances for the main Sun and its
//! nested warp Suns.

use std::collections::HashMap;
use std::sync::Arc;

use black_hole_flux::sun::SunAppearance;
use black_hole_flux::Ray;
use iced::Task;
use jungle_sdk::JungleClient;
use uuid::Uuid;

use crate::app::Message;

""" + get("liveconfig") + "\n\n" + get("live_snapshot") + "\n\n" + get("appearance_task") + "\n\n" + get("fetch_appearance") + "\n\n" + get("fetch_child_rays") + "\n\n" + get("max_warp_depth") + "\n\n" + get("fetch_warp_appearances") + "\n\n" + get("fetch_sun_appearance") + "\n"

FILES["src/model.rs"] = """//! The Black Hole Sun cell model: cells, edges, and graph construction from
//! static flows or live appearances.

use std::collections::{HashMap, HashSet};

use black_hole_flux::sun::{SunAppearance, SunNodeState};
use black_hole_flux::Ray;
use iced_sugiyama::Graph;
use uuid::Uuid;

use crate::flow::BlackHoleSunFlow;
use crate::labels::{animal_label_key, short_type_name};

""" + get("celldef") + "\n\n" + get("beammodel") + "\n\n" + get("model_display_changed") + "\n"

FILES["src/visual.rs"] = """//! Node phase and color state: how each cell's Sun phase is displayed and
//! animated over time.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use black_hole_flux::sun::SunNodeState;
use iced::Color;

const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);
const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);
const MAX_PENDING_PHASES: usize = 4;

""" + get("node_state_visual") + "\n\n" + get("node_progress") + "\n\n" + get("node_style_colors_struct") + "\n\n" + get("visual_fns") + "\n\n" + get("cell_visual_state") + "\n\n" + get("color_utils") + "\n"

FILES["src/graph.rs"] = """//! The Sugiyama graph widget for the main Black Hole Sun model.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use black_hole_flux::sun::SunNodeState;
use iced::mouse;
use iced::widget::canvas::{self, Path};
use iced::widget::Element;
use iced::{Color, Point, Rectangle, Theme, Vector};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{
    circo_layout, microdot_layout, AutoFit, Cluster, EdgeEndpointKind, Graph, LayoutInput, Sugiyama,
};

use crate::app::Message;
use crate::builder::BeamLayout;
use crate::visual::{node_style_colors, NodeStyleColors};

const DOT_VERTEX_SPACING: f64 = 128.0;
const EDGE_STROKE_WIDTH: f32 = 2.4;

""" + get("glyph") + "\n\n" + get("build_sun_graph") + "\n"

FILES["src/style.rs"] = """//! Theme and widget styles for the beam viewer.

use iced::{Background, Border, Color, Shadow, Theme, Vector};

use crate::visual::NodeStyleColors;

""" + get("style_fns") + "\n"

FILES["src/labels.rs"] = """//! Animal label formatting: shortening type names and extracting warp
//! boundary labels.

""" + get("labels") + "\n"

FILES["src/subpanel.rs"] = """//! The node-click subpanel overlay that shows a child flow's live journey.

use jungle_sdk::{Animal, JourneyAstSource};
use jungle_vision::{
    AnyAnimal, ClusterExpansionConfig, ClusterExpansionMode, DefaultTheme, EjectedViewer,
    JungleViewerBuilder,
};
use uuid::Uuid;

use crate::client::SharedJungleClient;

type JungleSubpanelViewer = EjectedViewer<DefaultTheme, AnyAnimal>;

""" + get("subpanel") + "\n\n" + get("subpanel_state") + "\n"

FILES["src/app/mod.rs"] = """//! The iced application: window, messages, and the main view.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use black_hole_flux::sun::SunNodeState;
use iced::keyboard;
use iced::time::Instant;
use iced::widget::{button, column, container, mouse_area, opaque, row, rule, space, stack, text};
use iced::{Element, Font, Length, Subscription, Task, Theme};

#[cfg(feature = "piano")]
use crate::piano::piano_audio::PianoAudioEngine;
#[cfg(feature = "piano")]
use crate::piano::piano_score::{PianoScorePlayback, SCORE_TICK_INTERVAL};
use crate::builder::BeamConfig;
use crate::client::SharedJungleClient;
use crate::graph::build_sun_graph;
use crate::labels::{animal_label_key, warp_boundary_label};
use crate::live::{appearance_task, LiveAppearanceSnapshot, LiveConfig};
use crate::model::{model_display_changed, BeamModel};
use crate::style::{
    app_background_style, beam_theme, black_hole_text, cell_node_style,
    graph_node_button_style, subpanel_child_canvas_style, subpanel_close_button_style,
    subpanel_left_edge_style, subpanel_notice_style, subpanel_overlay_style, subpanel_style,
};
use crate::subpanel::{SubpanelConfig, SubpanelState};
use crate::visual::{
    displayed_grad_step, node_style_colors, warp_node_style_colors, CellVisualState,
    NodeStyleColors,
};

mod piano;

const CELL_GRAPH_ID: &str = "black-hole-beam-cells";
const APPEARANCE_INTERVAL: Duration = Duration::from_millis(200);
const COLOR_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const COLOR_TRANSITION_POLL_INTERVAL: Duration = Duration::from_millis(100);

""" + get("message") + "\n\n" + get("beamapp_struct") + "\n\n" + get("run_beam") + "\n\n" + get("beamapp_impl") + "\n"

FILES["src/app/piano.rs"] = """//! Piano input, events, logging, and strike visuals for the beam app.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use iced::keyboard;
use iced::widget::Element;
use iced::Color;

use crate::piano::piano_audio::PianoAudioEngine;
use crate::piano::score_text::{self, BhsScore};
use crate::piano::{
    piano_height, PianoAction, PianoEvent, PianoInputSource, PianoKeyAppearance, PianoKeyboard,
    PianoMessage, PianoNote, PianoPointerSource,
};

use super::{BeamApp, Message};

""" + get("piano_computer_key") + "\n\n" + get("piano_log_const") + "\n\n" + get("piano_log_ticks") + "\n\n" + get("piano_log_velocity") + "\n\n" + get("piano_input_id") + "\n\n" + get("active_piano_note") + "\n\n" + get("strike_visual") + "\n"

for path, content in FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"wrote {path} ({len(content.splitlines())} lines)")

# Sanity check: every original non-blank line must appear in exactly one output file or the new lib.rs.
import collections
out_lines = []
for path, content in FILES.items():
    out_lines += [l for l in content.split("\n")]
orig_content_lines = [l for l in lines if l.strip()]
out_content_lines = [l for l in out_lines if l.strip()]
missing = collections.Counter(orig_content_lines) - collections.Counter(out_content_lines)
extra = collections.Counter(out_content_lines) - collections.Counter(orig_content_lines)
print("missing lines:", sum(missing.values()))
for l, c in list(missing.items())[:20]:
    print("  MISSING", c, repr(l))
print("extra lines:", sum(extra.values()))
for l, c in list(extra.items())[:20]:
    print("  EXTRA", c, repr(l))
