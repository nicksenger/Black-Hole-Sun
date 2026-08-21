//! The Sugiyama graph widget for the main Black Hole Sun model.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use black_hole_flux::sun::SunNodeState;
use iced::mouse;
use iced::widget::canvas::{self, Path};
use iced::{Color, Element, Point, Rectangle, Theme, Vector};
use iced_sugiyama::motion::easing::Easing;
use iced_sugiyama::{
    circo_layout, microdot_layout, Cluster, EdgeEndpointKind, Graph, LayoutInput, Sugiyama,
};

use crate::app::Message;
use crate::builder::BeamLayout;
use crate::visual::{node_style_colors, NodeStateVisual, NodeStyleColors};

const DOT_VERTEX_SPACING: f64 = 128.0;
const EDGE_STROKE_WIDTH: f32 = 2.4;

#[derive(Debug, Clone, Copy)]
enum EdgeEndpointGlyphKind {
    NormalArrow,
}

#[derive(Debug, Clone, Copy)]
struct EdgeEndpointGlyph {
    kind: EdgeEndpointGlyphKind,
    color: Color,
    angle_radians: f32,
}

impl EdgeEndpointGlyph {
    fn size(self) -> f32 {
        match self.kind {
            EdgeEndpointGlyphKind::NormalArrow => 20.0,
        }
    }
}

impl<Message, Theme, Renderer> canvas::Program<Message, Theme, Renderer> for EdgeEndpointGlyph
where
    Renderer: iced::advanced::graphics::geometry::Renderer,
{
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let anchor = frame.center();

        match self.kind {
            EdgeEndpointGlyphKind::NormalArrow => {
                let arrow = Path::new(|path| {
                    path.move_to(Point::new(0.0, 0.0));
                    path.line_to(Point::new(-10.0, 4.0));
                    path.line_to(Point::new(-7.25, 0.0));
                    path.line_to(Point::new(-10.0, -4.0));
                    path.close();
                });

                frame.with_save(|frame| {
                    frame.translate(Vector::new(anchor.x, anchor.y));
                    frame.rotate(self.angle_radians);
                    frame.fill(&arrow, self.color);
                });
            }
        }

        vec![frame.into_geometry()]
    }
}


/// Builds a Sugiyama graph widget for one Black Hole Sun model.
pub(crate) fn build_sun_graph(
    graph: Graph,
    labels: HashMap<u32, (String, String)>,
    styles: HashMap<u32, NodeStyleColors>,
    layout: BeamLayout,
    animation_duration: Option<Duration>,
    animation_easing: Option<&'static Easing>,
    view_node: impl Fn(u32, String, String, NodeStyleColors) -> Element<'static, Message> + 'static,
) -> Sugiyama<'static, Message, Theme, iced::Renderer> {
    let mut layout_graph = graph;
    // Match the spacing used by iced-sugiyama's "moar" example.
    layout_graph.config.vertex_spacing = DOT_VERTEX_SPACING;

    let node_labels = labels.clone();
    let styles_for_nodes = styles.clone();
    let view_node = move |node_id: u32| {
        let (animal_name, phase_label) = node_labels.get(&node_id).cloned().unwrap_or((
            format!("cell {node_id}"),
            SunNodeState::Idle.label().to_string(),
        ));
        let style = styles_for_nodes
            .get(&node_id)
            .copied()
            .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None));
        view_node(node_id, animal_name, phase_label, style)
    };

    let mut graph =
        Sugiyama::<Message, Theme, iced::Renderer>::new(Cow::Owned(layout_graph), view_node);

    if matches!(layout, BeamLayout::Circo) {
        graph = graph.layout_fn(|input| {
            // circo's ported implementation addresses edge coordinates by
            // node index. Remap public cell IDs so sparse port numbers keep
            // their edges attached to the correct nodes.
            let original_nodes = input.nodes.clone();
            let node_index = original_nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (*node, index as u32))
                .collect::<HashMap<_, _>>();
            let remapped_nodes = Arc::from((0..original_nodes.len() as u32).collect::<Vec<_>>());
            let remapped_edges = Arc::from(
                input
                    .edges
                    .iter()
                    .map(|(from, to)| (node_index[from], node_index[to]))
                    .collect::<Vec<_>>(),
            );
            let original_node_size = input.node_size.clone();
            let node_ids_for_size = original_nodes.clone();
            let original_edge_label = input.edge_label.clone();
            let original_edges = input.edges.clone();

            #[allow(clippy::arc_with_non_send_sync)]
            let remapped_input = LayoutInput {
                nodes: remapped_nodes,
                edges: remapped_edges,
                config: input.config,
                render_config: input.render_config,
                clusters: Arc::from(Vec::<Cluster>::new()),
                node_size: Arc::new(move |index| {
                    original_node_size(node_ids_for_size[index as usize])
                }),
                edge_label: Arc::new(move |index, _| {
                    original_edge_label(index, original_edges[index])
                }),
            };
            circo_layout(&remapped_input)
        });
    } else if matches!(layout, BeamLayout::Microdot) {
        graph = graph.layout_fn(microdot_layout);
    }

    let styles_for_edges = styles.clone();
    let styles_for_endpoints = styles;
    graph = graph
        .edge_color(move |ctx| {
            let start = styles_for_edges
                .get(&ctx.edge.0)
                .map(|style| style.body)
                .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
            let end = styles_for_edges
                .get(&ctx.edge.1)
                .map(|style| style.body)
                .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
            // iced-sugiyama paints the first tuple element at the edge's
            // head and the second at its tail, so pass (end, start) to
            // gradient from the source color into the target color.
            (end, start)
        })
        .edge_endpoint(move |_, edge, kind, endpoint| {
            if matches!(kind, EdgeEndpointKind::Source) {
                return None;
            }
            let node_id = edge.1;
            let color = styles_for_endpoints
                .get(&node_id)
                .map(|style| style.body)
                .unwrap_or_else(|| node_style_colors(SunNodeState::Idle, 1, 1, None).body);
            let glyph = EdgeEndpointGlyph {
                kind: EdgeEndpointGlyphKind::NormalArrow,
                color,
                angle_radians: endpoint.angle_radians(),
            };
            Some(
                canvas::Canvas::new(glyph)
                    .width(glyph.size())
                    .height(glyph.size())
                    .into(),
            )
        })
        .stroke_width(EDGE_STROKE_WIDTH)
        .edge_corner_radius(16.0);

    if let Some(duration) = animation_duration {
        graph = graph.animation_duration(duration);
    }
    if let Some(easing) = animation_easing {
        graph = graph.animation_easing(easing);
    }
    graph
}
