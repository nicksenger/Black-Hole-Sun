//! Theme and widget styles for the beam viewer.

use iced::{Background, Color, Shadow, Theme, Vector};

use crate::app::BeamApp;
use crate::visual::NodeStyleColors;

pub(crate) fn black_hole_text() -> Color {
    Color::from_rgb8(252, 226, 184)
}

pub(crate) fn beam_theme(_app: &BeamApp) -> Theme {
    Theme::Dark
}

pub(crate) fn app_background_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::BLACK)),
        text_color: Some(black_hole_text()),
        ..Default::default()
    }
}

pub(crate) fn subpanel_notice_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgb8(195, 24, 41))),
        text_color: Some(Color::WHITE),
        ..Default::default()
    }
}

pub(crate) fn subpanel_style(colors: NodeStyleColors) -> iced::widget::container::Style {
    iced::widget::container::Style {
        // Mirror each node's phase tint while keeping subpanel content readable.
        background: Some(Background::Color(Color::from_rgba(
            colors.body.r,
            colors.body.g,
            colors.body.b,
            0.2,
        ))),
        // The panel is outlined only on its left edge, which is rendered as a
        // vertical rule because iced borders apply to all sides.
        text_color: Some(colors.text),
        ..Default::default()
    }
}

pub(crate) fn subpanel_left_edge_style(colors: NodeStyleColors) -> iced::widget::rule::Style {
    iced::widget::rule::Style {
        color: Color::from_rgba(colors.border.r, colors.border.g, colors.border.b, 0.58),
        radius: 0.0.into(),
        fill_mode: iced::widget::rule::FillMode::Full,
        snap: true,
    }
}

pub(crate) fn subpanel_overlay_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(3, 3, 3, 0.7))),
        border: iced::Border {
            color: Color::from_rgba8(120, 120, 120, 0.25),
            width: 1.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

pub(crate) fn subpanel_child_canvas_style(_theme: &Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        // Child graph canvases should feel translucent without reducing text legibility.
        // Near-black with a faint blue tint to match the beam jungle theme.
        background: Some(Background::Color(Color::from_rgba8(5, 7, 14, 0.7))),
        ..Default::default()
    }
}

pub(crate) fn subpanel_close_button_style(
    _theme: &Theme,
    status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    let text_color = match status {
        iced::widget::button::Status::Hovered => Color::from_rgb8(255, 205, 156),
        _ => black_hole_text().scale_alpha(0.88),
    };
    iced::widget::button::Style {
        background: None,
        text_color,
        shadow: Shadow::default(),
        snap: false,
        ..Default::default()
    }
}

pub(crate) fn graph_node_button_style(
    _theme: &Theme,
    _status: iced::widget::button::Status,
) -> iced::widget::button::Style {
    iced::widget::button::Style {
        background: None,
        text_color: black_hole_text(),
        border: iced::Border::default(),
        shadow: Shadow::default(),
        snap: false,
    }
}

pub(crate) fn cell_node_style(colors: NodeStyleColors) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(colors.body)),
        text_color: Some(colors.text),
        border: iced::Border {
            color: colors.border,
            width: 2.2,
            ..iced::border::rounded(9)
        },
        shadow: Shadow {
            color: Color::from_rgba(colors.body.r, colors.body.g, colors.body.b, 0.32),
            offset: Vector::new(0.0, 2.0),
            blur_radius: 10.0,
        },
        ..Default::default()
    }
}
