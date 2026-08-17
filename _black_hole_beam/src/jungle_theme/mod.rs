//! Beam-local jungle-vision theme for subpanel child graphs.
//!
//! This mirrors `jungle_vision::DefaultTheme` (as of the locked Jungle
//! revision) so behavior stays in lockstep with upstream, but every
//! green in the palette is hue-rotated to blue: completed nodes and edges
//! render blue instead of green, and cluster fills use a dark blue base.
//! Running (yellow), failed (red), and pending (gray) are unchanged.

mod animated_cluster;
mod animated_step;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// Like upstream jungle-vision, the theme state uses a tokio mutex.
use tokio::sync::Mutex;

use iced::Color;
use iced::{Element, Task};
use jungle_sdk::{NodeLifecyclePhase, RunnerUpdateOut};
use jungle_vision::{
    AnyAnimal, ClusterExpansionConfig, ClusterExpansionMode, ClusterKind, ClusterLive,
    ClusterView, ClusterViewCtx, EdgeStyle, EdgeStyleCtx, JunglePanelTheme, Phase, RuntimeState,
    StepKind, StepViewCtx, ViewerEvent,
};

use animated_cluster::AnimatedClusterView;
use animated_step::AnimatedStepNode;

const NODE_ANIMATION_DURATION: Duration = Duration::from_millis(320);
const CLUSTER_BORDER_ANIMATION_DURATION: Duration = Duration::from_millis(320);
const CLUSTER_RECOLLAPSE_DELAY: Duration = Duration::from_secs(2);

/// Jungle-vision's runtime palette with the completed green (55, 144, 81)
/// hue-rotated to a bright outer-space azure (92, 128, 240). The other
/// states are unchanged.
fn runtime_color(state: RuntimeState) -> Color {
    match state {
        RuntimeState::Pending => Color::from_rgb8(120, 120, 120),
        RuntimeState::Running => Color::from_rgb8(212, 190, 68),
        RuntimeState::Completed => Color::from_rgb8(92, 128, 240),
        RuntimeState::Failed => Color::from_rgb8(165, 61, 61),
    }
}

fn cluster_border_color_gray() -> Color {
    runtime_color(RuntimeState::Pending)
}

/// Port of jungle-vision's `cluster_panel::target_color` with the dark green
/// fill base (20, 46, 30) hue-rotated to a brighter blue (33, 43, 77), and
/// the fill alpha lowered from 0.10 to 0.08 for a more translucent panel.
fn cluster_fill_target_color(kind: ClusterKind, phase: Phase<ClusterLive>) -> Color {
    let alpha = match phase {
        Phase::Static => kind_pending_alpha(kind),
        Phase::Live(live) => {
            if live.has_failed {
                kind_failed_alpha(kind)
            } else if live.has_running {
                kind_running_alpha(kind)
            } else if live.has_completed {
                kind_completed_alpha(kind)
            } else {
                kind_pending_alpha(kind)
            }
        }
    };
    Color::from_rgba8(33, 43, 77, alpha.clamp(0.0, 1.0))
}

fn kind_pending_alpha(kind: ClusterKind) -> f32 {
    match kind {
        ClusterKind::While => 0.08,
        ClusterKind::Join => 0.08,
        ClusterKind::Transparent => 0.08,
        ClusterKind::Attempt => 0.08,
    }
}

fn kind_running_alpha(kind: ClusterKind) -> f32 {
    match kind {
        ClusterKind::While => 0.08,
        ClusterKind::Join => 0.08,
        ClusterKind::Transparent => 0.08,
        ClusterKind::Attempt => 0.08,
    }
}

fn kind_completed_alpha(kind: ClusterKind) -> f32 {
    match kind {
        ClusterKind::While => 0.08,
        ClusterKind::Join => 0.08,
        ClusterKind::Transparent => 0.08,
        ClusterKind::Attempt => 0.08,
    }
}

fn kind_failed_alpha(kind: ClusterKind) -> f32 {
    match kind {
        ClusterKind::While => 0.08,
        ClusterKind::Join => 0.08,
        ClusterKind::Transparent => 0.08,
        ClusterKind::Attempt => 0.08,
    }
}

/// Port of the private `ClusterExpansionConfig::mode_for` helper.
fn expansion_mode_for(
    config: ClusterExpansionConfig,
    kind: ClusterKind,
) -> ClusterExpansionMode {
    match kind {
        ClusterKind::While => config.while_clusters,
        ClusterKind::Join => config.transparent_clusters,
        ClusterKind::Transparent => config.transparent_clusters,
        ClusterKind::Attempt => config.transparent_clusters,
    }
}

fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    Color {
        r: from.r + (to.r - from.r) * t,
        g: from.g + (to.g - from.g) * t,
        b: from.b + (to.b - from.b) * t,
        a: from.a + (to.a - from.a) * t,
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

#[derive(Debug, Clone, Copy)]
struct NodeVisual {
    state: RuntimeState,
}

#[derive(Debug, Clone)]
struct ClusterRuntimeIndex {
    kind: ClusterKind,
    entry_runtime_ids: HashSet<u32>,
    member_runtime_ids: HashSet<u32>,
    successor_runtime_ids: HashSet<u32>,
}

#[derive(Debug, Clone, Copy)]
struct ClusterVisual {
    expanded: bool,
    border_state: RuntimeState,
    completed_at: Option<Instant>,
}

#[derive(Debug)]
pub struct BeamJungleThemeState {
    node_visuals: HashMap<u32, NodeVisual>,
    cluster_index: HashMap<u32, ClusterRuntimeIndex>,
    cluster_visuals: HashMap<u32, ClusterVisual>,
    force_pending_runtime_ids: HashSet<u32>,
    cluster_expansion: ClusterExpansionConfig,
}

impl BeamJungleThemeState {
    fn new(cluster_expansion: ClusterExpansionConfig) -> Self {
        Self {
            node_visuals: HashMap::new(),
            cluster_index: HashMap::new(),
            cluster_visuals: HashMap::new(),
            force_pending_runtime_ids: HashSet::new(),
            cluster_expansion,
        }
    }

    fn register_cluster(&mut self, cx: &ClusterViewCtx<'_>) {
        let now = Instant::now();
        let expansion_mode = expansion_mode_for(self.cluster_expansion, cx.kind);
        let expanded = matches!(expansion_mode, ClusterExpansionMode::AlwaysExpanded)
            || matches!(cx.phase, Phase::Live(live) if live.has_running || live.has_failed);
        let border_state = match cx.phase {
            Phase::Live(live) if live.has_failed => RuntimeState::Failed,
            Phase::Live(live) if live.has_running => RuntimeState::Running,
            Phase::Live(live) if live.has_completed => RuntimeState::Completed,
            _ => RuntimeState::Pending,
        };
        self.cluster_index
            .entry(cx.cluster_id)
            .or_insert_with(|| ClusterRuntimeIndex {
                kind: cx.kind,
                entry_runtime_ids: cx.entry_runtime_ids.iter().copied().collect(),
                member_runtime_ids: cx.member_runtime_ids.iter().copied().collect(),
                successor_runtime_ids: cx.successor_runtime_ids.iter().copied().collect(),
            });
        let visual = self
            .cluster_visuals
            .entry(cx.cluster_id)
            .or_insert(ClusterVisual {
                expanded,
                border_state,
                completed_at: None,
            });
        if matches!(expansion_mode, ClusterExpansionMode::AlwaysExpanded)
            || matches!(border_state, RuntimeState::Running | RuntimeState::Failed)
        {
            visual.expanded = true;
        }
        if visual.border_state != border_state {
            visual.completed_at = if matches!(border_state, RuntimeState::Completed) {
                Some(now)
            } else {
                None
            };
            visual.border_state = border_state;
        }
    }

    fn cluster_is_expanded(&self, cluster_id: u32) -> bool {
        self.cluster_visuals
            .get(&cluster_id)
            .map(|visual| visual.expanded)
            .unwrap_or(false)
    }

    fn update_node_state(&mut self, runtime_id: u32, to: RuntimeState) -> bool {
        if !matches!(to, RuntimeState::Pending) {
            self.force_pending_runtime_ids.remove(&runtime_id);
        }
        let entry = self.node_visuals.entry(runtime_id).or_insert(NodeVisual {
            state: RuntimeState::Pending,
        });

        if entry.state == to {
            return false;
        }

        entry.state = to;
        true
    }

    fn reset_cluster_members_to_pending(
        &mut self,
        cluster_id: u32,
        except_runtime_id: u32,
    ) -> bool {
        let Some(index) = self.cluster_index.get(&cluster_id) else {
            return false;
        };
        let members = index.member_runtime_ids.iter().copied().collect::<Vec<_>>();
        let mut changed = false;
        for member_id in members {
            if member_id == except_runtime_id {
                continue;
            }
            self.force_pending_runtime_ids.insert(member_id);
            changed |= self.update_node_state(member_id, RuntimeState::Pending);
        }
        changed
    }

    fn update_clusters_for_effect_input(&mut self, runtime_id: u32, now: Instant) -> bool {
        let mut changed = false;
        let cluster_ids = self.cluster_index.keys().copied().collect::<Vec<_>>();
        for cluster_id in cluster_ids {
            let Some(index) = self.cluster_index.get(&cluster_id) else {
                continue;
            };
            let contains_member = index.member_runtime_ids.contains(&runtime_id);
            let contains_entry = index.entry_runtime_ids.contains(&runtime_id);
            let contains_successor = index.successor_runtime_ids.contains(&runtime_id);
            let is_while_cluster = matches!(index.kind, ClusterKind::While);
            let expansion_mode = expansion_mode_for(self.cluster_expansion, index.kind);

            let mut activated_iteration = false;
            if let Some(visual) = self.cluster_visuals.get_mut(&cluster_id) {
                if matches!(expansion_mode, ClusterExpansionMode::AlwaysExpanded)
                    && !visual.expanded
                {
                    visual.expanded = true;
                    changed = true;
                }
                let while_reentered_via_non_entry_member = is_while_cluster
                    && contains_member
                    && !contains_entry
                    && !contains_successor
                    && !matches!(visual.border_state, RuntimeState::Running);
                let member_activation = contains_member
                    && match expansion_mode {
                        ClusterExpansionMode::Automatic => !visual.expanded,
                        ClusterExpansionMode::AlwaysExpanded => {
                            !matches!(visual.border_state, RuntimeState::Running)
                        }
                    };
                if (is_while_cluster && contains_entry)
                    || member_activation
                    || while_reentered_via_non_entry_member
                {
                    let expansion_changed = !visual.expanded;
                    visual.expanded = true;
                    visual.completed_at = None;
                    let border_changed = visual.border_state != RuntimeState::Running;
                    visual.border_state = RuntimeState::Running;
                    changed |= border_changed || expansion_changed;
                    activated_iteration = true;
                } else if visual.expanded && contains_successor {
                    let border_changed = visual.border_state != RuntimeState::Completed;
                    visual.border_state = RuntimeState::Completed;
                    changed |= border_changed;
                    visual.completed_at.get_or_insert(now);
                }
            }

            if activated_iteration || (is_while_cluster && contains_entry) {
                changed |= self.reset_cluster_members_to_pending(cluster_id, runtime_id);
            }
        }
        changed
    }

    fn apply_force_pending_override(
        &mut self,
        runtime_id: u32,
        phase_target: RuntimeState,
    ) -> RuntimeState {
        if !self.force_pending_runtime_ids.contains(&runtime_id) {
            return phase_target;
        }

        match phase_target {
            RuntimeState::Pending => RuntimeState::Pending,
            RuntimeState::Running => RuntimeState::Running,
            RuntimeState::Completed | RuntimeState::Failed => {
                self.force_pending_runtime_ids.remove(&runtime_id);
                phase_target
            }
        }
    }

    fn maybe_collapse_completed_cluster_for_pending_successor(
        &mut self,
        cx: &ClusterViewCtx<'_>,
        now: Instant,
    ) -> bool {
        if !matches!(
            cx.kind,
            ClusterKind::While
                | ClusterKind::Join
                | ClusterKind::Transparent
                | ClusterKind::Attempt
        ) {
            return false;
        }
        if matches!(
            expansion_mode_for(self.cluster_expansion, cx.kind),
            ClusterExpansionMode::AlwaysExpanded
        ) {
            return false;
        }

        let Phase::Live(live) = cx.phase else {
            return false;
        };
        if live.has_running {
            return false;
        }

        let should_collapse = self
            .cluster_visuals
            .get(&cx.cluster_id)
            .map(|visual| {
                visual.expanded
                    && matches!(visual.border_state, RuntimeState::Completed)
                    && visual
                        .completed_at
                        .map(|completed_at| {
                            now.saturating_duration_since(completed_at) >= CLUSTER_RECOLLAPSE_DELAY
                        })
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        if !should_collapse {
            return false;
        }

        let Some(visual) = self.cluster_visuals.get_mut(&cx.cluster_id) else {
            return false;
        };
        visual.expanded = false;
        visual.completed_at = None;
        let border_changed = visual.border_state != RuntimeState::Pending;
        visual.border_state = RuntimeState::Pending;
        border_changed
    }

    fn cluster_border_color(&self, cluster_id: u32) -> Color {
        self.cluster_visuals
            .get(&cluster_id)
            .map(|visual| runtime_color(visual.border_state))
            .unwrap_or_else(cluster_border_color_gray)
    }
}

/// Jungle-vision theme that mirrors `DefaultTheme` with a blue palette.
#[derive(Clone, Copy, Debug, Default)]
pub struct BeamJungleTheme {
    cluster_expansion: ClusterExpansionConfig,
}

impl BeamJungleTheme {
    pub fn with_cluster_expansion_config(
        mut self,
        cluster_expansion: ClusterExpansionConfig,
    ) -> Self {
        self.cluster_expansion = cluster_expansion;
        self
    }
}

impl JunglePanelTheme<AnyAnimal> for BeamJungleTheme {
    type State = Mutex<BeamJungleThemeState>;
    type Message = ();

    fn init(&self) -> Self::State {
        Mutex::new(BeamJungleThemeState::new(self.cluster_expansion))
    }

    fn update(
        &self,
        state: &mut Self::State,
        event: ViewerEvent<Self::Message>,
    ) -> Task<ViewerEvent<Self::Message>> {
        match event {
            ViewerEvent::JourneyUpdate(update) => {
                let guard = state.get_mut();
                let now = Instant::now();
                match update.event {
                    RunnerUpdateOut::EffectInput { node_id, .. } => {
                        let _ = guard.update_node_state(node_id, RuntimeState::Running);
                        let _ = guard.update_clusters_for_effect_input(node_id, now);
                    }
                    RunnerUpdateOut::EffectSuccessOutput { node_id, .. } => {
                        let _ = guard.update_node_state(node_id, RuntimeState::Completed);
                    }
                    RunnerUpdateOut::EffectFailureOutput { node_id, .. } => {
                        let _ = guard.update_node_state(node_id, RuntimeState::Failed);
                    }
                    RunnerUpdateOut::NodeLifecycle(node) => match node.phase {
                        NodeLifecyclePhase::Entered => {
                            let _ = guard.update_node_state(node.node_id, RuntimeState::Running);
                            let _ = guard.update_clusters_for_effect_input(node.node_id, now);
                        }
                        NodeLifecyclePhase::Succeeded => {
                            let _ = guard.update_node_state(node.node_id, RuntimeState::Completed);
                        }
                        NodeLifecyclePhase::Failed => {
                            let _ = guard.update_node_state(node.node_id, RuntimeState::Failed);
                        }
                    },
                    RunnerUpdateOut::SleepScheduled { .. }
                    | RunnerUpdateOut::SleepFired { .. }
                    | RunnerUpdateOut::PerturbationApplied { .. } => {}
                }
            }
            ViewerEvent::Message(()) => {}
        }

        Task::none()
    }

    fn view_step(
        &self,
        state: &Self::State,
        cx: &StepViewCtx<'_>,
    ) -> (Element<'static, ViewerEvent<Self::Message>>, (f64, f64)) {
        let role = match cx.kind {
            StepKind::Conditional => "condition",
            StepKind::Select => "select",
            StepKind::Join => "join",
            StepKind::Step => "step",
        };

        let fill = if let Some(runtime_id) = cx.runtime_id {
            let phase_target = match cx.phase {
                Phase::Live(target) => target,
                Phase::Static => RuntimeState::Pending,
            };
            let phase_target = if let Ok(mut guard) = state.try_lock() {
                guard.apply_force_pending_override(runtime_id, phase_target)
            } else {
                phase_target
            };
            runtime_color(phase_target)
        } else {
            let phase_target = match cx.phase {
                Phase::Live(target) => target,
                Phase::Static => RuntimeState::Pending,
            };
            runtime_color(phase_target)
        };
        (
            AnimatedStepNode::<ViewerEvent<Self::Message>>::new(
                state as *const Self::State as usize as u64,
                cx.display_id,
                cx.runtime_id,
                role,
                cx.label.to_string(),
                cx.metadata.map(str::to_string),
                fill,
                NODE_ANIMATION_DURATION,
            )
            .into(),
            (240.0, 80.0),
        )
    }

    fn view_cluster(
        &self,
        state: &Self::State,
        cx: &ClusterViewCtx<'_>,
    ) -> ClusterView<Self::Message> {
        let now = Instant::now();
        let (expanded, border_color) = if let Ok(mut guard) = state.try_lock() {
            guard.register_cluster(cx);
            guard.maybe_collapse_completed_cluster_for_pending_successor(cx, now);
            (
                guard.cluster_is_expanded(cx.cluster_id),
                guard.cluster_border_color(cx.cluster_id),
            )
        } else {
            (false, cluster_border_color_gray())
        };
        let fill = cluster_fill_target_color(cx.kind, cx.phase);
        let overlay = AnimatedClusterView::<ViewerEvent<Self::Message>>::overlay(
            cx.cluster_id,
            cx.label.to_string(),
            border_color,
            fill,
            CLUSTER_BORDER_ANIMATION_DURATION,
        )
        .into();

        if expanded {
            ClusterView::Expanded {
                overlay: Some(overlay),
                fill,
            }
        } else {
            ClusterView::Collapsed {
                element: AnimatedClusterView::<ViewerEvent<Self::Message>>::chip(
                    cx.cluster_id,
                    cx.label.to_string(),
                    border_color,
                    CLUSTER_BORDER_ANIMATION_DURATION,
                )
                .into(),
                size: (240.0, 46.0),
            }
        }
    }

    fn edge_style(&self, state: &Self::State, cx: EdgeStyleCtx) -> Option<EdgeStyle> {
        let source_phase = match cx.source_phase {
            Phase::Live(target) => target,
            Phase::Static => RuntimeState::Pending,
        };
        let source_phase = if let Some(runtime_id) = cx.source_runtime_id {
            if let Ok(mut guard) = state.try_lock() {
                guard.apply_force_pending_override(runtime_id, source_phase)
            } else {
                source_phase
            }
        } else {
            source_phase
        };
        let target_phase = match cx.target_phase {
            Phase::Live(target) => target,
            Phase::Static => RuntimeState::Pending,
        };
        let target_phase = if let Some(runtime_id) = cx.target_runtime_id {
            if let Ok(mut guard) = state.try_lock() {
                guard.apply_force_pending_override(runtime_id, target_phase)
            } else {
                target_phase
            }
        } else {
            target_phase
        };
        let phase_target = match target_phase {
            RuntimeState::Pending => match source_phase {
                RuntimeState::Running | RuntimeState::Failed => source_phase,
                RuntimeState::Completed | RuntimeState::Pending => RuntimeState::Pending,
            },
            RuntimeState::Running | RuntimeState::Completed | RuntimeState::Failed => target_phase,
        };
        let (from_color, to_color) = {
            let color = runtime_color(phase_target);
            (color, color)
        };

        let progress = cx.extent.clamp(0.0, 1.0);
        let source_t = ease_out_cubic((progress / 0.55).clamp(0.0, 1.0));
        let target_t = ease_out_cubic(((progress - 0.25) / 0.75).clamp(0.0, 1.0));
        let start = lerp_color(from_color, to_color, source_t);
        let end = lerp_color(from_color, to_color, target_t);

        Some(EdgeStyle {
            width: 1.6,
            start,
            end,
        })
    }
}
