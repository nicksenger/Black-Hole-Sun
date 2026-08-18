//! A container that casts a shadow behind its content.
//!
//! Ported from ice_nine's `ShadowBox`, which splits the box vertically by
//! height so each part can cast a different shadow. This version splits
//! horizontally with [`Cutoff`] instead: the piano uses it to shade the
//! subpanel area (the right third of the view) without shading the main
//! graph.
//!
//! Unlike ice_nine's widget, this one does not reserve layout space for the
//! shadow; like a styled [`Container`], the shadow is allowed to spill outside
//! the box's bounds (e.g. upward over whatever sits above it).

use iced::advanced::layout;
use iced::advanced::mouse;
use iced::advanced::renderer::{self, Quad};
use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::widget::Operation;
use iced::advanced::{Clipboard, Layout, Shell, Widget};
use iced::widget::container::{self, Container};
use iced::{Background, Border, Color, Element, Event, Length, Rectangle, Shadow, Size, Vector};

#[allow(missing_debug_implementations)]
pub struct ShadowBox<'a, Message, Theme, Renderer>
where
    Theme: container::Catalog,
    Renderer: iced::advanced::Renderer,
{
    content: Container<'a, Message, Theme, Renderer>,
    shadow: Shadow,
    cutoff: Option<Cutoff>,
}

/// Splits a [`ShadowBox`] into two horizontal regions.
#[derive(Debug, Clone, Copy)]
pub struct Cutoff {
    /// Position of the split, as a fraction of the box's width (`0.0..=1.0`).
    pub x: f32,
    /// Shadow cast by the region to the left of the split. The region to the
    /// right uses the [`ShadowBox`] shadow.
    pub shadow: Shadow,
}

impl<'a, Message, Theme, Renderer> ShadowBox<'a, Message, Theme, Renderer>
where
    Message: 'a,
    Theme: 'a + container::Catalog,
    Renderer: iced::advanced::Renderer + 'a,
{
    /// Creates a new [`ShadowBox`] with the given content.
    pub fn new(content: Container<'a, Message, Theme, Renderer>) -> Self {
        ShadowBox {
            content,
            shadow: Shadow {
                color: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.5,
                },
                offset: Vector::new(0.0, 2.0),
                blur_radius: 4.0,
            },
            cutoff: None,
        }
    }

    /// Sets the shadow cast by the box (the region to the right of any
    /// [`Cutoff`]).
    pub fn shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = shadow;
        self
    }

    /// Splits the box horizontally so the left region casts `cutoff.shadow`
    /// instead of the [`ShadowBox`] shadow.
    pub fn cutoff(mut self, cutoff: Cutoff) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Draws `shadow` behind `region`, clipping it in a layer so the shadow
    /// cannot spill past the box's outer edges or across a cutoff into the
    /// other region.
    fn draw_region(
        &self,
        renderer: &mut Renderer,
        region: Rectangle,
        shadow: Shadow,
        extends_left: bool,
        extends_right: bool,
    ) {
        if region.width <= 0.0 || shadow.color.a == 0.0 || shadow.blur_radius <= 0.0 {
            return;
        }

        let left = if extends_left {
            shadow.blur_radius - shadow.offset.x.min(0.0)
        } else {
            0.0
        };
        let right = if extends_right {
            shadow.blur_radius + shadow.offset.x.max(0.0)
        } else {
            0.0
        };
        let top = shadow.blur_radius - shadow.offset.y.min(0.0);
        let bottom = shadow.blur_radius + shadow.offset.y.max(0.0);

        renderer.with_layer(
            Rectangle {
                x: region.x - left,
                y: region.y - top,
                width: region.width + left + right,
                height: region.height + top + bottom,
            },
            |renderer| {
                renderer.fill_quad(
                    Quad {
                        bounds: region,
                        border: Border {
                            radius: 0.0.into(),
                            width: 0.0,
                            color: Color::TRANSPARENT,
                        },
                        shadow,
                        snap: false,
                    },
                    Background::Color(Color::TRANSPARENT),
                );
            },
        );
    }
}

impl<'a, Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for ShadowBox<'a, Message, Theme, Renderer>
where
    Message: 'a + Clone,
    Theme: container::Catalog + 'a,
    Renderer: 'a + iced::advanced::Renderer,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::new())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content as &dyn Widget<Message, Theme, _>)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(&[&self.content as &dyn Widget<Message, Theme, _>]);
    }

    fn size(&self) -> Size<Length> {
        Widget::size(&self.content)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        // No padding is reserved for the shadow: it may spill outside the box
        // (upward over whatever sits above it), like a styled container's.
        //
        // The returned node *is* the content container's node. `draw`,
        // `update`, `operate`, `overlay`, and `mouse_interaction` must
        // therefore forward `layout` to the content as-is, never descending
        // into it (the content descends one level itself).
        self.content.layout(&mut tree.children[0], renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        // `layout` is the content container's own node (see `layout`), so it
        // is forwarded as-is rather than descended into.
        self.content.operate(
            &mut tree.children[0],
            layout,
            renderer,
            operation,
        );
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor_position: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content.update(
            &mut tree.children[0],
            event,
            layout,
            cursor_position,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor_position: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        // `layout` is the content container's own node (see `layout`), so its
        // bounds are the box's bounds and it is forwarded as-is.
        let bounds = layout.bounds();

        match self.cutoff {
            Some(Cutoff { x, shadow }) if (0.0..1.0).contains(&x) => {
                let split_x = bounds.x + (x * bounds.width).round();
                self.draw_region(
                    renderer,
                    Rectangle {
                        x: bounds.x,
                        y: bounds.y,
                        width: split_x - bounds.x,
                        height: bounds.height,
                    },
                    shadow,
                    true,
                    false,
                );
                self.draw_region(
                    renderer,
                    Rectangle {
                        x: split_x,
                        y: bounds.y,
                        width: bounds.x + bounds.width - split_x,
                        height: bounds.height,
                    },
                    self.shadow,
                    false,
                    true,
                );
            }
            _ => self.draw_region(renderer, bounds, self.shadow, true, true),
        }

        renderer.with_layer(bounds, |renderer| {
            self.content.draw(
                &tree.children[0],
                renderer,
                theme,
                style,
                layout,
                cursor_position,
                viewport,
            );
        });
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor_position: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.mouse_interaction(
            &tree.children[0],
            layout,
            cursor_position,
            viewport,
            renderer,
        )
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<iced::advanced::overlay::Element<'b, Message, Theme, Renderer>> {
        self.content.overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Message, Theme, Renderer> From<ShadowBox<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: Clone + 'a,
    Theme: container::Catalog + 'a,
    Renderer: iced::advanced::Renderer + 'a,
{
    fn from(shadow_box: ShadowBox<'a, Message, Theme, Renderer>) -> Self {
        Self::new(shadow_box)
    }
}

/// The local state of a [`ShadowBox`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct State;

impl State {
    /// Creates a new [`State`].
    pub const fn new() -> State {
        State
    }
}
