//! Node phase and color state: how each cell's Sun phase is displayed and
//! animated over time.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use black_hole_flux::topology::{SunNodeState, SunOperationalState};
use iced::Color;

use crate::style::black_hole_text;

pub(crate) const COLOR_FADE_DURATION: Duration = Duration::from_millis(400);
pub(crate) const MIN_COLOR_STATE_DURATION: Duration = Duration::from_secs(1);
pub(crate) const MAX_PENDING_PHASES: usize = 4;

pub(crate) trait NodeStateVisual {
    fn label(self) -> &'static str;
}

impl NodeStateVisual for SunNodeState {
    fn label(self) -> &'static str {
        match self {
            SunNodeState::Idle => "idle",
            SunNodeState::Propagation1 => "propagation 1",
            SunNodeState::Propagation2 => "propagation 2",
            SunNodeState::Optimization => "potentiation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeProgress {
    pub(crate) state: SunNodeState,
    pub(crate) grad_step: usize,
}

impl NodeProgress {
    const fn idle() -> Self {
        Self {
            state: SunNodeState::Idle,
            grad_step: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NodeStyleColors {
    pub(crate) body: Color,
    pub(crate) border: Color,
    pub(crate) text: Color,
}

fn default_grad_step_for_state(state: SunNodeState, grad_steps: usize) -> usize {
    match state {
        SunNodeState::Optimization => grad_steps.max(1),
        _ => 1,
    }
}

fn node_phase_progress(grad_step: usize, grad_steps: usize) -> f32 {
    let grad_steps = grad_steps.max(1);
    let step = grad_step.clamp(1, grad_steps);
    step as f32 / grad_steps as f32
}

pub(crate) fn node_style_colors(
    state: SunNodeState,
    grad_step: usize,
    grad_steps: usize,
    frozen: Option<bool>,
) -> NodeStyleColors {
    let idle_orange = Color::from_rgb8(228, 108, 30);
    let bright_yellow = Color::from_rgb8(255, 233, 68);
    let deep_crimson = Color::from_rgb8(195, 24, 41);
    let fire_red = Color::from_rgb8(240, 96, 24);
    let potentiation_blue = Color::from_rgb8(65, 105, 225);
    let frozen_potentiation_violet = Color::from_rgb8(124, 77, 255);

    if state == SunNodeState::Optimization {
        if frozen == Some(true) {
            // Black body so the frozen node recedes, but a violet outline keeps
            // it identifiable during potentiation.
            return NodeStyleColors {
                body: Color::BLACK,
                border: frozen_potentiation_violet,
                text: Color::BLACK,
            };
        }
        return NodeStyleColors {
            body: Color::from_rgb8(255, 255, 255),
            border: potentiation_blue,
            text: Color::from_rgb8(18, 12, 8),
        };
    }

    let (body, border) = match state {
        SunNodeState::Idle => (idle_orange, lighten(idle_orange, 0.18)),
        SunNodeState::Propagation1 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            let body = lerp_color(idle_orange, deep_crimson, progress);
            (body, lighten(body, 0.18))
        }
        SunNodeState::Propagation2 => {
            let progress = node_phase_progress(grad_step, grad_steps);
            let body = lerp_color(idle_orange, bright_yellow, progress);
            // Keep a fixed fire-red outline so the yellow body stays distinct.
            (body, fire_red)
        }
        SunNodeState::Optimization => unreachable!("optimization is handled above"),
    };
    NodeStyleColors {
        body,
        border,
        text: contrasting_text(body),
    }
}

/// Program-neutral status colors used when a Sun program publishes a phase
/// annotation (for example, a forward-only pass) instead of a legacy
/// two-sided optimization phase.
pub(crate) fn operational_node_style_colors(state: SunOperationalState) -> NodeStyleColors {
    let (body, border) = match state {
        SunOperationalState::Queued => {
            let body = Color::from_rgb8(66, 78, 96);
            (body, lighten(body, 0.2))
        }
        SunOperationalState::Running => {
            let body = Color::from_rgb8(228, 108, 30);
            (body, Color::from_rgb8(255, 190, 80))
        }
        SunOperationalState::Succeeded => {
            let body = Color::from_rgb8(36, 156, 92);
            (body, Color::from_rgb8(101, 232, 160))
        }
        SunOperationalState::Failed => {
            let body = Color::from_rgb8(195, 24, 41);
            (body, Color::from_rgb8(255, 104, 116))
        }
    };
    NodeStyleColors {
        body,
        border,
        text: contrasting_text(body),
    }
}

/// Unexpanded warp nodes render with a black body and white text while
/// keeping the phase-driven border color. Once their subgraph is expanded
/// into the main graph they are colored like regular nodes instead.
pub(crate) fn warp_node_style_colors(
    state: SunNodeState,
    grad_step: usize,
    grad_steps: usize,
    frozen: Option<bool>,
) -> NodeStyleColors {
    let mut colors = node_style_colors(state, grad_step, grad_steps, frozen);
    colors.body = Color::BLACK;
    colors.text = Color::WHITE;
    colors
}

pub(crate) fn displayed_grad_step(
    state: SunNodeState,
    observed: NodeProgress,
    grad_steps: usize,
) -> usize {
    if state == observed.state {
        return observed.grad_step.clamp(1, grad_steps.max(1));
    }
    default_grad_step_for_state(state, grad_steps.max(1))
}

fn next_phase(phase: SunNodeState) -> SunNodeState {
    match phase {
        SunNodeState::Idle | SunNodeState::Optimization => SunNodeState::Propagation1,
        SunNodeState::Propagation1 => SunNodeState::Propagation2,
        SunNodeState::Propagation2 => SunNodeState::Optimization,
    }
}

fn phase_path(from: SunNodeState, to: SunNodeState) -> Vec<SunNodeState> {
    if from == to {
        return Vec::new();
    }
    if to == SunNodeState::Idle {
        return vec![SunNodeState::Idle];
    }

    let mut path = Vec::new();
    let mut phase = from;
    for _ in 0..4 {
        phase = next_phase(phase);
        path.push(phase);
        if phase == to {
            return path;
        }
    }
    vec![to]
}

fn recent_phase_steps(mut phase: SunNodeState, count: u64) -> Vec<SunNodeState> {
    let retained = count.min(MAX_PENDING_PHASES as u64);
    let mut skipped = count - retained;

    if skipped > 0 && phase == SunNodeState::Idle {
        phase = next_phase(phase);
        skipped -= 1;
    }
    for _ in 0..skipped % 3 {
        phase = next_phase(phase);
    }

    (0..retained)
        .map(|_| {
            phase = next_phase(phase);
            phase
        })
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) struct CellVisualState {
    previous: NodeProgress,
    pub(crate) current: NodeProgress,
    transition_started_at: Option<Instant>,
    pub(crate) pending: VecDeque<NodeProgress>,
    observed_sequence: u64,
    latest_frozen: Option<bool>,
    pub(crate) optimization_frozen: Option<bool>,
}

impl Default for CellVisualState {
    fn default() -> Self {
        Self {
            previous: NodeProgress::idle(),
            current: NodeProgress::idle(),
            transition_started_at: None,
            pending: VecDeque::new(),
            observed_sequence: 0,
            latest_frozen: None,
            optimization_frozen: None,
        }
    }
}

impl CellVisualState {
    pub(crate) fn observe(
        &mut self,
        activity: SunNodeState,
        grad_step: usize,
        grad_steps: usize,
        sequence: u64,
        frozen: Option<bool>,
        now: Instant,
    ) -> bool {
        let grad_steps = grad_steps.max(1);
        let latest = self.pending.back().copied().unwrap_or(self.current);
        let observed = NodeProgress {
            state: activity,
            grad_step: grad_step.clamp(1, grad_steps),
        };
        if sequence < self.observed_sequence {
            return false;
        }
        self.latest_frozen = frozen;

        let path = if sequence > self.observed_sequence {
            let path = recent_phase_steps(latest.state, sequence - self.observed_sequence);
            self.observed_sequence = sequence;
            if path.last().copied() == Some(activity) {
                path
            } else {
                phase_path(latest.state, activity)
            }
        } else {
            phase_path(latest.state, activity)
        };
        if path.is_empty() {
            if latest != observed {
                self.pending.push_back(observed);
            } else {
                return false;
            }
        } else {
            let path_len = path.len();
            for (index, state) in path.into_iter().enumerate() {
                let progress = if index + 1 == path_len {
                    observed
                } else {
                    NodeProgress {
                        state,
                        grad_step: default_grad_step_for_state(state, grad_steps),
                    }
                };
                self.pending.push_back(progress);
            }
        }
        while self.pending.len() > MAX_PENDING_PHASES {
            self.pending.pop_front();
        }
        if self.can_transition(now) {
            return self.begin_next_transition(now);
        }
        false
    }

    pub(crate) fn advance(&mut self, now: Instant) -> bool {
        if !self.can_transition(now) {
            return false;
        }
        self.begin_next_transition(now)
    }

    pub(crate) fn style(
        &self,
        grad_steps: usize,
        frozen: Option<bool>,
        warp: bool,
        now: Instant,
    ) -> NodeStyleColors {
        let progress = self
            .transition_started_at
            .map(|started_at| {
                now.saturating_duration_since(started_at).as_secs_f32()
                    / COLOR_FADE_DURATION.as_secs_f32()
            })
            .unwrap_or(1.0);
        let previous_frozen = self.frozen_for_state(self.previous.state, frozen);
        let current_frozen = self.frozen_for_state(self.current.state, frozen);
        let style_fn = if warp {
            warp_node_style_colors
        } else {
            node_style_colors
        };
        let previous = style_fn(
            self.previous.state,
            self.previous.grad_step,
            grad_steps,
            previous_frozen,
        );
        let current = style_fn(
            self.current.state,
            self.current.grad_step,
            grad_steps,
            current_frozen,
        );
        NodeStyleColors {
            body: lerp_color(previous.body, current.body, progress),
            border: lerp_color(previous.border, current.border, progress),
            text: lerp_color(previous.text, current.text, progress),
        }
    }

    pub(crate) fn frozen_for_state(
        &self,
        state: SunNodeState,
        fallback: Option<bool>,
    ) -> Option<bool> {
        if state == SunNodeState::Optimization {
            return self.optimization_frozen.or(self.latest_frozen).or(fallback);
        }
        fallback
    }

    pub(crate) fn is_fading(&self, now: Instant) -> bool {
        self.transition_started_at.is_some_and(|started_at| {
            now.saturating_duration_since(started_at) < COLOR_FADE_DURATION
        })
    }

    pub(crate) fn needs_color_frame(&self, now: Instant) -> bool {
        self.is_fading(now)
    }

    pub(crate) fn needs_transition_poll(&self, now: Instant) -> bool {
        !self.pending.is_empty() && !self.is_fading(now)
    }

    fn can_transition(&self, now: Instant) -> bool {
        self.transition_started_at.is_none_or(|started_at| {
            now.saturating_duration_since(started_at) >= MIN_COLOR_STATE_DURATION
        })
    }

    fn begin_next_transition(&mut self, now: Instant) -> bool {
        let Some(activity) = self.pending.pop_front() else {
            return false;
        };
        if activity == self.current {
            return self.begin_next_transition(now);
        }
        if self.current.state == SunNodeState::Propagation1
            && activity.state == SunNodeState::Propagation2
        {
            self.optimization_frozen = self.latest_frozen;
        }
        self.previous = self.current;
        self.current = activity;
        self.transition_started_at = Some(now);
        true
    }
}

fn contrasting_text(background: Color) -> Color {
    let luminance = 0.2126 * background.r + 0.7152 * background.g + 0.0722 * background.b;
    if luminance > 0.58 {
        Color::from_rgb8(26, 14, 9)
    } else {
        black_hole_text()
    }
}

pub(crate) fn lerp_color(a: Color, b: Color, amount: f32) -> Color {
    let amount = amount.clamp(0.0, 1.0);
    Color::from_rgba(
        a.r + (b.r - a.r) * amount,
        a.g + (b.g - a.g) * amount,
        a.b + (b.b - a.b) * amount,
        a.a + (b.a - a.a) * amount,
    )
}

fn lighten(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::WHITE, amount)
}
