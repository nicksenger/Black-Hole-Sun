//! Visualize Black Hole Sun cell graphs.
//!
//! [`BeamBuilder`] renders the type-level cell topology of a
//! [`BlackHole`](black_hole_flux::sun::BlackHole), using the circular `circo`
//! layout by default. Live views use the parent Sun animal's Jungle
//! [`Observe`](jungle_sdk::Observe) appearance as the source of graph topology
//! and node phase.
//!
//! The `piano` feature adds a NanoMoog-styled 88-key piano to the bottom of
//! the viewer. Use `BeamBuilder::on_piano_event` to capture its expressive,
//! performance-timed attack and release events, or `BeamBuilder::piano_log`
//! to log played notes to stdout as `bhs-score-v1` note pairs.

mod app;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet, VecDeque};
    #[cfg(feature = "piano")]
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use black_hole_flux::sun::{SunAppearance, SunNodeState};
    use black_hole_flux::{FusionSeed, FusionState, Ray};
    #[cfg(feature = "piano")]
    use iced::keyboard;
    use iced::time::Instant;
    use iced::Color;
    use jungle_sdk::Animal;
    use uuid::Uuid;

    #[cfg(feature = "piano")]
    use crate::app::piano::{piano_log_ticks, PianoInputId, PianoStrikeVisual};
    use crate::app::{BeamApp, Message, APPEARANCE_INTERVAL};
    use crate::builder::BeamLayout;
    use crate::labels::{animal_label_key, short_type_name, warp_boundary_label};
    use crate::live::{LiveAppearanceSnapshot, LiveConfig};
    use crate::model::{BeamModel, CellDefinition};
    #[cfg(feature = "piano")]
    use crate::piano::piano_score::PianoScorePlayback;
    #[cfg(feature = "piano")]
    use crate::piano::score_text::BhsScore;
    #[cfg(feature = "piano")]
    use crate::piano::PianoPointerSource;
    use crate::visual::{
        lerp_color, node_style_colors, warp_node_style_colors, CellVisualState, NodeProgress,
        NodeStateVisual, COLOR_FADE_DURATION, MAX_PENDING_PHASES, MIN_COLOR_STATE_DURATION,
    };

    use black_hole_flux::sun::{
        Binary, BlackHole, Manifest, StatelessManifest, SunEdgeAppearance, SunNodeAppearance, Unary,
    };
    use black_hole_flux::{CellState, Fusion, Primordium};
    use jungle_sdk::typosaurus::collections::list::{Empty, List};
    use jungle_sdk::Id;
    use typenum::{U0, U1, U2};

    struct TestCell;

    impl Animal for TestCell {
        type Id = Id<U1>;
        type Generation = U0;
        type State = CellState;
        type Seed = black_hole_flux::CellInit;
        type Flow = Primordium;
    }

    struct TestFusion;

    impl Animal for TestFusion {
        type Id = Id<U2>;
        type Generation = U0;
        type State = FusionState;
        type Seed = FusionSeed;
        type Flow = Fusion<Primordium>;
    }

    struct GenericCell<T>(std::marker::PhantomData<T>);

    impl<T> Animal for GenericCell<T> {
        type Id = Id<U1>;
        type Generation = U0;
        type State = CellState;
        type Seed = black_hole_flux::CellInit;
        type Flow = Primordium;
    }

    type PortOne = List<(U1, Empty)>;
    type Tail = List<(Unary<U1, TestCell, Empty>, Empty)>;
    type TestGraph = List<(Unary<U0, TestCell, PortOne>, Tail)>;
    struct CustomStateManifest;

    impl Manifest for CustomStateManifest {
        type Generator = Primordium;
        type Policy = Primordium;
        type State = (String, String);
    }

    type TestSun = <TestGraph as BlackHole>::Sun<StatelessManifest<Primordium, Primordium>, 1>;
    type TestSunWithCustomState = <TestGraph as BlackHole>::Sun<CustomStateManifest, 1>;
    type TestBinaryGraph = List<(Binary<U0, U1, TestFusion, Empty>, Empty)>;
    type TestBinarySun =
        <TestBinaryGraph as BlackHole>::Sun<StatelessManifest<Primordium, Primordium>, 1>;

    #[test]
    fn uses_black_hole_sun_title_and_grad_step_palette() {
        assert_eq!(BeamBuilder::default().title, "Black Hole Sun");
        assert!(matches!(BeamBuilder::default().layout, BeamLayout::Circo));
        assert!(matches!(
            BeamBuilder::new().microdot_layout().layout,
            BeamLayout::Microdot
        ));
        let p1_step1 = node_style_colors(SunNodeState::Propagation1, 1, 4, None).body;
        let p1_step4 = node_style_colors(SunNodeState::Propagation1, 4, 4, None).body;
        assert!(p1_step4.g < p1_step1.g);
        assert!(p1_step4.r - p1_step4.g > p1_step1.r - p1_step1.g);

        let p2_step1 = node_style_colors(SunNodeState::Propagation2, 1, 4, None).body;
        let p2_step4 = node_style_colors(SunNodeState::Propagation2, 4, 4, None).body;
        assert!(p2_step4.g > p2_step1.g);

        let optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(false));
        assert_eq!(optimize.body, Color::from_rgb8(255, 255, 255));
        assert_eq!(optimize.text, Color::from_rgb8(18, 12, 8));
        assert_eq!(optimize.border, Color::from_rgb8(65, 105, 225));

        let frozen_optimize = node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        assert_eq!(frozen_optimize.body, Color::BLACK);
        assert_eq!(frozen_optimize.border, Color::from_rgb8(124, 77, 255));
    }

    #[test]
    fn registers_unique_subpanel_animals() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .register_subpanel_animal::<GenericCell<String>>()
            .register_subpanel_animal::<GenericCell<u8>>()
            .register_subpanel_animal::<TestCell>()
            .into_config();

        assert_eq!(config.subpanel_animals.len(), 3);
        assert_eq!(config.subpanel_animals[0].animal_label, "TestCell");
        assert_eq!(config.subpanel_animals[0].title, "TestCell");
        assert_eq!(
            config.subpanel_animals[1].animal_label,
            "GenericCell<String>"
        );
        assert_eq!(config.subpanel_animals[1].title, "GenericCell<String>");
        assert_eq!(config.subpanel_animals[2].animal_label, "GenericCell<u8>");
        assert_eq!(config.subpanel_animals[2].title, "GenericCell<u8>");
    }

    #[test]
    fn extracts_warp_boundary_labels() {
        assert_eq!(
            warp_boundary_label("Warp<MyWarp, MyBoundary>").as_deref(),
            Some("MyBoundary")
        );
        assert_eq!(
            warp_boundary_label("Warp<Outer<A, B>, Inner<C>>").as_deref(),
            Some("Inner<C>"),
            "the split happens at the first top-level comma"
        );
        assert_eq!(warp_boundary_label("MyWarp<A, B>"), None);
        assert_eq!(warp_boundary_label("Warp<Solo>"), None);
        assert_eq!(warp_boundary_label("Warp<A,"), None);
    }

    #[test]
    fn warp_node_subpanel_resolves_registered_boundary_animal() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let (app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        assert_eq!(
            app.resolve_subpanel_config("Warp<OtherAnimal, TestCell>")
                .map(|config| config.animal_label),
            Some("TestCell".to_string()),
            "the registered boundary animal stands in for the warp node"
        );
        assert_eq!(
            app.resolve_subpanel_config("Warp<TestCell, OtherAnimal>")
                .map(|config| config.animal_label),
            None,
            "registering the warp animal does not open the warp node"
        );
        assert_eq!(
            app.resolve_subpanel_config("TestCell")
                .map(|config| config.animal_label),
            Some("TestCell".to_string()),
            "direct registrations still match plain nodes"
        );
    }

    #[test]
    fn warp_node_click_opens_the_boundary_journey_subpanel() {
        let boundary_journey = Uuid::new_v4();
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let mut cell = CellDefinition::new::<TestCell>(7, vec![0], vec![]);
        cell.animal_name = "Warp<OtherAnimal, TestCell>".to_string();
        cell.journey_id = boundary_journey;
        app.model.cells.push(cell);

        let _task = app.open_subpanel_for_node(7);

        let subpanel = app
            .subpanel
            .as_ref()
            .expect("the warp node should open its subpanel");
        assert_eq!(subpanel.node_id, 7);
        assert_eq!(subpanel.title, "TestCell");
        assert_eq!(
            subpanel.journey_id, boundary_journey,
            "the subpanel shows the boundary's live journey"
        );
    }

    fn nested_sun_appearance(finalized: bool) -> SunAppearance {
        SunAppearance {
            finalized,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::nil(),
                label: "NestedRoot".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        }
    }

    #[test]
    fn warp_subgraph_joins_main_graph_when_nested_appearance_arrives() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let warp_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: warp_journey,
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };

        // First poll: the nested appearance has not been exposed yet.
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(
            LiveAppearanceSnapshot {
                appearance: appearance.clone(),
                child_rays: HashMap::new(),
                warp_appearances: HashMap::new(),
                warp_diagnostics: HashMap::from([(
                    vec![7],
                    "warp journey 0f3c has not published an appearance yet".to_string(),
                )]),
            },
        ))));

        // Clicking the warp node opens the subpanel and explains why its
        // subgraph is missing...
        let _task = app.update(Message::NodeSelected(7));
        assert!(app.subpanel.is_some());
        let notice = app
            .subpanel_notice
            .as_deref()
            .expect("a missing nested appearance should be surfaced to the user");
        assert!(
            notice.starts_with("Cell 7:") && notice.contains("warp journey"),
            "the notice should carry the per-cell diagnostic, got: {notice}"
        );

        // ...until the nested appearance arrives with the subpanel open.
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(
            LiveAppearanceSnapshot {
                appearance,
                child_rays: HashMap::new(),
                warp_appearances: HashMap::from([(vec![7], nested_sun_appearance(true))]),
                warp_diagnostics: HashMap::new(),
            },
        ))));

        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8],
            "the nested node joins the main graph under a fresh id"
        );
        assert_eq!(
            app.model.graph.edges,
            vec![(7, 8)],
            "the boundary cell connects to the nested sink"
        );
        assert!(
            app.expanded_warp_cells.contains(&vec![7]),
            "the expanded warp cell is tracked for rebuilds"
        );
        assert_eq!(
            app.subpanel_notice, None,
            "the notice clears once the subgraph has joined the main graph"
        );

        // Clicking the same node again closes the subpanel and collapses
        // the merged subgraph out of the main graph.
        let _task = app.update(Message::NodeSelected(7));
        assert!(app.subpanel.is_none());
        assert!(!app.expanded_warp_cells.contains(&vec![7]));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "collapsing removes the nested nodes from the main graph"
        );
        assert!(
            app.model.graph.edges.is_empty(),
            "the connector edge goes away with the subgraph"
        );
    }

    #[test]
    fn plain_node_click_opens_the_subpanel() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let mut cell = CellDefinition::new::<TestCell>(3, vec![0], vec![]);
        cell.journey_id = Uuid::new_v4();
        app.model.cells.push(cell);

        let _task = app.open_subpanel_for_node(3);
        assert!(app.subpanel.is_some());
        assert_eq!(app.subpanel_notice, None);
    }

    #[test]
    fn from_appearance_merges_finalized_warp_subgraphs() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 3,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<A, B>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 5,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![2],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![],
        };
        let warp_appearances = HashMap::from([
            (vec![3], nested_sun_appearance(true)),
            (vec![5], nested_sun_appearance(false)),
        ]);

        let model = BeamModel::from_appearance(appearance, &HashMap::new(), &warp_appearances)
            .unwrap();

        // The finalized subgraph is merged under fresh ids; the unfinalized
        // one is skipped.
        assert_eq!(
            model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![0, 3, 5, 6]
        );
        assert_eq!(
            model.graph.edges,
            vec![(3, 6)],
            "the boundary cell connects to the nested sink"
        );
    }

    #[test]
    fn from_appearance_merges_nested_warps_recursively() {
        let outer = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<A, B>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        // Cell 2 is itself a warp cell inside the nested sun; the edge makes
        // it the unique sink of the middle sun.
        let middle = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "NestedRoot".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 2,
                target_port: 1,
            }],
        };
        let inner = nested_sun_appearance(true);

        let warp_appearances = HashMap::from([
            (vec![7], middle.clone()),
            (vec![7, 2], inner),
        ]);
        let model = BeamModel::from_appearance(outer, &HashMap::new(), &warp_appearances)
            .unwrap();

        assert_eq!(
            model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9, 10],
            "every level of the warp chain joins the main graph"
        );
        assert_eq!(
            model.graph.edges,
            vec![(7, 9), (8, 9), (9, 10)],
            "the boundary connects to the middle sink, which in turn connects to the inner sink"
        );
        assert_eq!(
            model.warp_paths.get(&7),
            Some(&vec![7]),
            "the top-level warp is locatable by its own id"
        );
        assert_eq!(
            model.warp_paths.get(&9),
            Some(&vec![7, 2]),
            "the merged nested warp keeps its path from the top level"
        );
    }

    #[test]
    fn nested_warp_subgraphs_expand_and_collapse_recursively() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let boundary_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: boundary_journey,
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        // The nested sun contains its own warp cell (id 2), whose sub-sun is
        // exposed as well.
        let middle = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "NestedRoot".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::new_v4(),
                    label: "Warp<InnerAnimal, TestCell>".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 2,
                target_port: 1,
            }],
        };

        let snapshot = LiveAppearanceSnapshot {
            appearance,
            child_rays: HashMap::new(),
            warp_appearances: HashMap::from([
                (vec![7], middle.clone()),
                (vec![7, 2], nested_sun_appearance(true)),
            ]),
            warp_diagnostics: HashMap::new(),
        };
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(snapshot))));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "nothing merges until a warp cell is clicked"
        );

        // Expand the top-level warp: its nested sun joins the main graph.
        let _task = app.update(Message::NodeSelected(7));
        assert_eq!(app.expanded_warp_cells, HashSet::from([vec![7]]));
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9]
        );

        // Expand the merged nested warp: its sub-sun joins as well.
        let _task = app.update(Message::NodeSelected(9));
        assert_eq!(
            app.expanded_warp_cells,
            HashSet::from([vec![7], vec![7, 2]]),
            "both levels of the warp chain are tracked"
        );
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7, 8, 9, 10],
            "the deeper sub-sun joins the main graph"
        );
        assert_eq!(app.model.graph.edges, vec![(7, 9), (8, 9), (9, 10)]);

        // Clicking the top-level warp again while its child subgraph is
        // still expanded only switches the subpanel back to it...
        let _task = app.update(Message::NodeSelected(7));
        assert_eq!(
            app.expanded_warp_cells,
            HashSet::from([vec![7], vec![7, 2]]),
            "the child subgraph stays expanded while its parent is re-clicked"
        );

        // ...and clicking it once more closes the subpanel and collapses the
        // whole chain, including the still-expanded child subgraph.
        let _task = app.update(Message::NodeSelected(7));
        assert!(
            app.expanded_warp_cells.is_empty(),
            "collapsing the parent also collapses its expanded child subgraphs"
        );
        assert_eq!(
            app.model.cells.iter().map(|cell| cell.id).collect::<Vec<_>>(),
            vec![7],
            "the whole merged chain leaves the main graph"
        );
        assert!(app.model.graph.edges.is_empty());
    }

    #[test]
    fn expanded_warp_cells_are_colored_like_regular_nodes() {
        let config = BeamBuilder::new()
            .register_subpanel_animal::<TestCell>()
            .into_config();
        let live = LiveConfig {
            client: Arc::new(jungle_client::MockClient::default()),
            journey_id: Uuid::new_v4(),
        };
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), Some(live));

        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 7,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::new_v4(),
                label: "Warp<OtherAnimal, TestCell>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };
        let snapshot = LiveAppearanceSnapshot {
            appearance,
            child_rays: HashMap::new(),
            warp_appearances: HashMap::from([(vec![7], nested_sun_appearance(true))]),
            warp_diagnostics: HashMap::new(),
        };
        let _task = app.update(Message::AppearanceLoaded(Ok(Some(snapshot))));

        let styles = app.cell_styles();
        assert_eq!(
            styles[&7].body,
            Color::BLACK,
            "an unexpanded warp cell keeps the black body"
        );

        // Expanding the subgraph recolors the boundary node like a regular
        // one.
        let _task = app.update(Message::NodeSelected(7));
        let styles = app.cell_styles();
        assert_eq!(
            styles[&7],
            node_style_colors(SunNodeState::Idle, 1, 1, None),
            "an expanded warp cell is colored like a regular node"
        );

        // Collapsing it restores the warp styling.
        let _task = app.update(Message::NodeSelected(7));
        let styles = app.cell_styles();
        assert_eq!(
            styles[&7].body,
            Color::BLACK,
            "collapsing the subgraph restores the warp styling"
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_events_preserve_order_timing_and_overlapping_voices() {
        use std::sync::Mutex;

        let events = Arc::new(Mutex::new(Vec::<PianoEvent>::new()));
        let captured = Arc::clone(&events);
        let config = BeamBuilder::new()
            .on_piano_event(move |event| captured.lock().unwrap().push(event))
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        app.attack_piano_note(
            PianoInputId::ComputerKeyboard('c'),
            PianoInputSource::ComputerKeyboard { key: 'c' },
            60,
            PianoEvent::BINARY_VELOCITY,
            None,
        );
        app.attack_piano_note(
            PianoInputId::Pointer(PianoPointerSource::Mouse),
            PianoInputSource::Mouse,
            60,
            0.42,
            Some(0.7),
        );
        app.release_piano_note(PianoInputId::ComputerKeyboard('c'), 0.0);
        assert_eq!(app.active_piano_notes.len(), 1);
        app.release_piano_note(PianoInputId::Pointer(PianoPointerSource::Mouse), 0.31);

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(events[0].note.midi_note, 60);
        assert_eq!(events[0].voice_id, events[2].voice_id);
        assert_eq!(events[1].voice_id, events[3].voice_id);
        assert_ne!(events[0].voice_id, events[1].voice_id);
        assert!(events
            .windows(2)
            .all(|events| events[0].timestamp <= events[1].timestamp));
        assert!(matches!(
            events[1].action,
            PianoAction::Attack {
                velocity,
                pressure: Some(pressure),
            } if velocity == 0.42 && pressure == 0.7
        ));
        assert!(matches!(
            events[3].action,
            PianoAction::Release { velocity, .. } if velocity == 0.31
        ));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_strike_opacity_tracks_velocity_pressure_and_release_speed() {
        let start = Instant::now();
        let visual = |velocity, pressure| PianoStrikeVisual {
            midi_note: 60,
            velocity,
            pressure,
            attacked_at: start,
            released: None,
        };
        let settled = start + Duration::from_millis(80);
        assert!(
            visual(0.9, Some(0.8)).appearance(settled).intensity
                > visual(0.3, Some(0.2)).appearance(settled).intensity
        );
        assert!(
            visual(0.6, Some(1.0)).appearance(settled).intensity
                > visual(0.6, Some(0.0)).appearance(settled).intensity
        );

        let released_at = start + Duration::from_millis(100);
        let slow_release = PianoStrikeVisual {
            released: Some((released_at, 0.0)),
            ..visual(0.8, Some(0.7))
        };
        let fast_release = PianoStrikeVisual {
            released: Some((released_at, 1.0)),
            ..visual(0.8, Some(0.7))
        };
        let fading = released_at + Duration::from_millis(80);
        assert!(
            slow_release.appearance(fading).intensity > fast_release.appearance(fading).intensity
        );
        assert!(fast_release.needs_frame(fading));
        assert!(fast_release.finished(released_at + Duration::from_millis(400)));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_a_score_path() {
        let config = BeamBuilder::new().score_path("moonlight.bhs").into_config();
        assert_eq!(
            config.piano_score_path,
            Some(PathBuf::from("moonlight.bhs"))
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_score_data() {
        let data = b"format bhs-score-v1";
        let config = BeamBuilder::new().score_data(data).into_config();
        assert_eq!(config.piano_score_data.as_deref(), Some(data.as_slice()));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_an_owned_score() {
        let score = BhsScore::parse("format bhs-score-v1\nticks_per_second 960\n0 960 C4 80\n")
            .expect("the fixture should parse");
        let config = BeamBuilder::new().score(score).into_config();
        assert_eq!(
            config
                .piano_score
                .as_ref()
                .map(|score| score.ticks_per_second),
            Some(960)
        );
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_merges_repeated_owned_scores() {
        let first = BhsScore::parse(
            "format bhs-score-v1\nticks_per_second 960\nloop_ticks 960\n0 480 C4 80\n",
        )
        .expect("the fixture should parse");
        let second = BhsScore::parse(
            "format bhs-score-v1\nticks_per_second 1920\nloop_ticks 1920\n480 480 E4 64\n",
        )
        .expect("the fixture should parse");
        let config = BeamBuilder::new().score(first).score(second).into_config();

        // The second call merges into the first instead of replacing it: the
        // first score's grid wins and both scores' notes are present.
        let merged = config.piano_score.as_ref().expect("the scores should merge");
        assert_eq!(merged.ticks_per_second, 960);
        assert_eq!(merged.pairs().count(), 2);
        // The second score's pair rescales from 1920 to 960 ticks/second.
        let pairs: Vec<_> = merged.pairs().collect();
        assert_eq!(pairs[1].start_tick, 240);
        assert_eq!(pairs[1].duration_ticks, 240);
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_score_skip() {
        assert_eq!(BeamBuilder::new().into_config().piano_score_skip_seconds, None);
        let config = BeamBuilder::new().score_path("intro.bhs").score_skip(5).into_config();
        assert_eq!(config.piano_score_skip_seconds, Some(5));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_piano_log() {
        let config = BeamBuilder::new().piano_log(PianoLog::Input).into_config();
        assert_eq!(config.piano_log, Some(PianoLog::Input));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn builder_records_piano_labels() {
        assert!(!BeamBuilder::new().into_config().piano_labels);
        assert!(BeamBuilder::new().piano_labels().into_config().piano_labels);
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_labels_follow_the_active_octave_and_shifts() {
        let config = BeamBuilder::new().piano_labels().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        // Labels are off by default.
        let plain_config = BeamBuilder::new().into_config();
        let (plain_app, _) = BeamApp::new(plain_config, BeamModel::empty(), None);
        assert_eq!(plain_app.piano_label_octave(), None);

        // The labeled octave starts at the selected octave...
        assert_eq!(app.piano_label_octave(), Some(4));

        let press_digit = |app: &mut BeamApp, key: char, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyPressed {
                key: keyboard::Key::Character(key.to_string().into()),
                modified_key: keyboard::Key::Character(key.to_string().into()),
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
                text: Some(key.to_string().into()),
                repeat: false,
            })
        };

        // ...and follows number-key octave selection.
        press_digit(&mut app, '5', keyboard::key::Code::Digit5);
        assert_eq!(app.piano_label_octave(), Some(5));

        let shift = |code: keyboard::key::Code| keyboard::Event::KeyPressed {
            key: keyboard::Key::Named(keyboard::key::Named::Shift),
            modified_key: keyboard::Key::Named(keyboard::key::Named::Shift),
            physical_key: keyboard::key::Physical::Code(code),
            location: keyboard::Location::Standard,
            modifiers: Default::default(),
            text: None,
            repeat: false,
        };

        // ...and is transposed by held Shift keys.
        app.update_piano_keyboard(shift(keyboard::key::Code::ShiftLeft));
        assert_eq!(app.piano_label_octave(), Some(4), "left shift lowers");
        app.update_piano_keyboard(shift(keyboard::key::Code::ShiftRight));
        assert_eq!(app.piano_label_octave(), Some(5), "both shifts cancel");
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_log_ticks_use_a_1920_tick_grid() {
        assert_eq!(piano_log_ticks(Duration::ZERO), 0);
        assert_eq!(piano_log_ticks(Duration::from_millis(500)), 960);
        // 812.5ms is exactly 1560 ticks; sub-tick remainders truncate.
        assert_eq!(piano_log_ticks(Duration::new(0, 812_500_000)), 1560);
        assert_eq!(piano_log_ticks(Duration::from_millis(1)), 1);
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_log_formats_released_notes_as_score_pairs() {
        let config = BeamBuilder::new().piano_log(PianoLog::Input).into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let attack = PianoEvent {
            sequence: 1,
            timestamp: Duration::from_millis(500),
            voice_id: 7,
            note: PianoNote::from_midi(59), // B3
            action: PianoAction::Attack {
                velocity: PianoEvent::BINARY_VELOCITY,
                pressure: None,
            },
            source: PianoInputSource::ComputerKeyboard { key: 'z' },
        };
        assert!(
            app.piano_log_line(&attack).is_none(),
            "the attack waits for its release"
        );

        let release = PianoEvent {
            sequence: 2,
            timestamp: Duration::new(0, 812_500_000),
            note: attack.note,
            action: PianoAction::Release {
                velocity: 0.0,
                held_for: Duration::new(0, 312_500_000),
            },
            ..attack
        };
        assert_eq!(
            app.piano_log_line(&release).as_deref(),
            Some("960 600 B3 127 0"),
            "start tick, duration ticks, note, attack velocity, release velocity"
        );

        // A release with no logged attack prints nothing.
        let stray = PianoEvent {
            sequence: 3,
            timestamp: Duration::from_millis(1),
            voice_id: 8,
            ..release
        };
        assert!(app.piano_log_line(&stray).is_none());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn piano_log_modes_choose_which_sources_print() {
        let events = |source| {
            [
                PianoEvent {
                    sequence: 1,
                    timestamp: Duration::from_millis(500),
                    voice_id: 7,
                    note: PianoNote::from_midi(60), // C4
                    action: PianoAction::Attack {
                        velocity: 0.5,
                        pressure: None,
                    },
                    source,
                },
                PianoEvent {
                    sequence: 2,
                    timestamp: Duration::from_millis(1_000),
                    voice_id: 7,
                    note: PianoNote::from_midi(60),
                    action: PianoAction::Release {
                        velocity: 0.5,
                        held_for: Duration::from_millis(500),
                    },
                    source,
                },
            ]
        };

        // Input mode skips notes played from a configured score.
        let config = BeamBuilder::new().piano_log(PianoLog::Input).into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        for event in events(PianoInputSource::Score) {
            assert!(app.piano_log_line(&event).is_none());
        }

        // All mode logs them.
        let config = BeamBuilder::new().piano_log(PianoLog::All).into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        let score_events = events(PianoInputSource::Score);
        let mut logged = score_events
            .iter()
            .filter_map(|event| app.piano_log_line(event));
        assert_eq!(logged.next().as_deref(), Some("960 960 C4 64 64"));
        assert!(logged.next().is_none());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn spacebar_is_not_a_piano_note() {
        let config = BeamBuilder::new().piano_log(PianoLog::All).into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        let event = keyboard::Event::KeyPressed {
            key: keyboard::Key::Character(" ".into()),
            modified_key: keyboard::Key::Character(" ".into()),
            physical_key: keyboard::key::Physical::Code(keyboard::key::Code::Space),
            location: keyboard::Location::Standard,
            modifiers: Default::default(),
            text: Some(" ".into()),
            repeat: false,
        };
        app.update_piano_keyboard(event);
        assert!(app.active_piano_notes.is_empty());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn number_keys_select_the_octave_for_home_row_notes() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        assert_eq!(app.piano_octave, 4);

        let press = |app: &mut BeamApp, key: &str, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyPressed {
                key: keyboard::Key::Character(key.into()),
                modified_key: keyboard::Key::Character(key.into()),
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
                text: Some(key.into()),
                repeat: false,
            })
        };

        let release = |app: &mut BeamApp, key: char| {
            app.release_piano_note(PianoInputId::ComputerKeyboard(key), 0.0)
        };
        let struck = |app: &BeamApp, key: char| {
            app.active_piano_notes
                .get(&PianoInputId::ComputerKeyboard(key))
                .map(|active| active.note.midi_note)
        };

        // The default octave is 4, so 'a' strikes A4 and 's' strikes B4.
        press(&mut app, "a", keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), Some(69));
        release(&mut app, 'a');
        press(&mut app, "s", keyboard::key::Code::KeyS);
        assert_eq!(struck(&app, 's'), Some(71));
        release(&mut app, 's');

        // '5' selects octave 5 without sounding a note; the next 'a' is A5.
        press(&mut app, "5", keyboard::key::Code::Digit5);
        assert_eq!(app.piano_octave, 5);
        assert!(app.active_piano_notes.is_empty());
        press(&mut app, "a", keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), Some(81));
        release(&mut app, 'a');

        // Octave 7 reaches the keyboard's top note on the row's third key.
        press(&mut app, "7", keyboard::key::Code::Digit7);
        press(&mut app, "d", keyboard::key::Code::KeyD);
        assert_eq!(struck(&app, 'd'), Some(108));
        release(&mut app, 'd');

        // Octave 8 selects nothing because the keyboard ends at C8; the
        // selection stays at 7.
        press(&mut app, "8", keyboard::key::Code::Digit8);
        assert_eq!(app.piano_octave, 7, "'8' is ignored");

        // Octave 0 reaches the keyboard's bottom two keys.
        press(&mut app, "0", keyboard::key::Code::Digit0);
        press(&mut app, "a", keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), Some(21));
        release(&mut app, 'a');
        press(&mut app, "s", keyboard::key::Code::KeyS);
        assert_eq!(struck(&app, 's'), Some(23));
    }

    #[cfg(feature = "piano")]
    #[test]
    fn shift_keys_shift_mapped_notes_by_one_octave() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let press = |app: &mut BeamApp, key: keyboard::Key, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyPressed {
                key: key.clone(),
                modified_key: key,
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
                text: None,
                repeat: false,
            })
        };
        let release = |app: &mut BeamApp, key: keyboard::Key, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyReleased {
                key: key.clone(),
                modified_key: key,
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
            })
        };
        let letter = |key: char| keyboard::Key::Character(key.to_string().into());
        let shift = || keyboard::Key::Named(keyboard::key::Named::Shift);
        let struck = |app: &BeamApp, key: char| {
            app.active_piano_notes
                .get(&PianoInputId::ComputerKeyboard(key))
                .map(|active| active.note.midi_note)
        };

        // Left shift strikes an octave below the mapped note; releasing the
        // Shift key does not cut off the held note.
        press(&mut app, shift(), keyboard::key::Code::ShiftLeft);
        press(&mut app, letter('a'), keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), Some(57), "left-shift a is A3");
        release(&mut app, shift(), keyboard::key::Code::ShiftLeft);
        assert_eq!(
            struck(&app, 'a'),
            Some(57),
            "releasing shift keeps the note sounding"
        );
        release(&mut app, letter('a'), keyboard::key::Code::KeyA);

        // Right shift strikes an octave above the mapped note.
        press(&mut app, shift(), keyboard::key::Code::ShiftRight);
        press(&mut app, letter('a'), keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), Some(81), "right-shift a is A5");
        release(&mut app, letter('a'), keyboard::key::Code::KeyA);
        release(&mut app, shift(), keyboard::key::Code::ShiftRight);

        // Shifted notes outside the keyboard are not struck.
        press(
            &mut app,
            keyboard::Key::Character("0".into()),
            keyboard::key::Code::Digit0,
        );
        press(&mut app, shift(), keyboard::key::Code::ShiftLeft);
        press(&mut app, letter('a'), keyboard::key::Code::KeyA);
        assert_eq!(struck(&app, 'a'), None, "A-1 is below the keyboard");
        release(&mut app, shift(), keyboard::key::Code::ShiftLeft);

        // Restore the default octave before checking both shifts held.
        press(
            &mut app,
            keyboard::Key::Character("4".into()),
            keyboard::key::Code::Digit4,
        );

        // With both Shift keys held the mapped note sounds natural.
        press(&mut app, shift(), keyboard::key::Code::ShiftLeft);
        press(&mut app, shift(), keyboard::key::Code::ShiftRight);
        press(&mut app, letter('a'), keyboard::key::Code::KeyA);
        assert_eq!(
            struck(&app, 'a'),
            Some(69),
            "both shifts sound the natural note"
        );
        release(&mut app, letter('a'), keyboard::key::Code::KeyA);
        release(&mut app, shift(), keyboard::key::Code::ShiftLeft);
        release(&mut app, shift(), keyboard::key::Code::ShiftRight);
        assert!(app.active_piano_notes.is_empty());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn enter_key_sounds_the_top_white_key_of_the_home_row() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let press = |app: &mut BeamApp, key: keyboard::Key, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyPressed {
                key: key.clone(),
                modified_key: key,
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
                text: None,
                repeat: false,
            })
        };
        let release = |app: &mut BeamApp, key: keyboard::Key, code: keyboard::key::Code| {
            app.update_piano_keyboard(keyboard::Event::KeyReleased {
                key: key.clone(),
                modified_key: key,
                physical_key: keyboard::key::Physical::Code(code),
                location: keyboard::Location::Standard,
                modifiers: Default::default(),
            })
        };
        let enter = || keyboard::Key::Named(keyboard::key::Named::Enter);

        // In the default octave 4, Enter strikes E of the next octave.
        press(&mut app, enter(), keyboard::key::Code::Enter);
        assert_eq!(
            app.active_piano_notes
                .get(&PianoInputId::ComputerKeyboard('\r'))
                .map(|active| active.note.midi_note),
            Some(88)
        );
        release(&mut app, enter(), keyboard::key::Code::Enter);
        assert!(app.active_piano_notes.is_empty());

        // The shift octaves apply to Enter as well.
        press(
            &mut app,
            keyboard::Key::Named(keyboard::key::Named::Shift),
            keyboard::key::Code::ShiftLeft,
        );
        press(&mut app, enter(), keyboard::key::Code::Enter);
        assert_eq!(
            app.active_piano_notes
                .get(&PianoInputId::ComputerKeyboard('\r'))
                .map(|active| active.note.midi_note),
            Some(76)
        );
        release(&mut app, enter(), keyboard::key::Code::Enter);
        release(
            &mut app,
            keyboard::Key::Named(keyboard::key::Named::Shift),
            keyboard::key::Code::ShiftLeft,
        );
        assert!(app.active_piano_notes.is_empty());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn score_ticks_loop_through_audio_event_and_visual_paths() {
        use std::sync::Mutex;

        let start = Instant::now();
        // A 0.5s C4 at 2000 ticks/second, looping every 0.5s: the release
        // lands exactly when the next cycle's attack is due.
        let score_text = "\
format bhs-score-v1
ticks_per_second 2000
loop_ticks 1000
0 1000 C4 98 36
";
        let score = PianoScorePlayback::from_text(score_text, start).unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let callback_events = Arc::clone(&captured);
        let config = BeamBuilder::new()
            .on_piano_event(move |event| callback_events.lock().unwrap().push(event))
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        app.piano_score = Some(score);

        app.update_piano_score(start);
        assert_eq!(app.active_piano_notes.len(), 1);
        assert_eq!(app.piano_strike_visuals.len(), 1);

        app.update_piano_score(start + Duration::from_millis(500));
        assert_eq!(
            app.active_piano_notes.len(),
            1,
            "cycle 1 attacks immediately"
        );
        assert_eq!(app.piano_strike_visuals.len(), 2);
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 3);
        let attack_velocity = f32::from(98u8) / 127.0;
        let release_velocity = f32::from(36u8) / 127.0;
        assert!(matches!(
            captured[0].action,
            PianoAction::Attack {
                velocity,
                pressure: None
            } if (velocity - attack_velocity).abs() < 1e-6
        ));
        assert!(matches!(
            captured[1].action,
            PianoAction::Release { velocity, .. }
                if (velocity - release_velocity).abs() < 1e-6
        ));
        assert_eq!(captured[0].source, PianoInputSource::Score);
        assert_eq!(captured[2].source, PianoInputSource::Score);
        assert_ne!(captured[0].voice_id, captured[2].voice_id);
    }

    #[cfg(feature = "piano")]
    #[test]
    fn score_skip_shifts_playback() {
        // A 20-second loop with a note at t=0 and one at t=10s: skipping 5s
        // drops the first note and makes the 10s note sound at app time 5s.
        let score_text = "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 19200
0 960 C4 80
9600 960 E4 80
";
        let config = BeamBuilder::new()
            .score_data(score_text.as_bytes())
            .score_skip(5)
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        assert!(app.piano_score_error.is_none(), "{:?}", app.piano_score_error);
        let start = Instant::now();

        // Nothing sounds at the start: the tick-0 note was skipped and the
        // survivor does not sound until 5s in.
        app.update_piano_score(start);
        assert!(app.active_piano_notes.is_empty());

        // The 10s note sounds at app time 5s and releases one second later.
        app.update_piano_score(start + Duration::from_secs(5));
        assert_eq!(app.active_piano_notes.len(), 1);
        app.update_piano_score(start + Duration::from_secs(6));
        assert!(app.active_piano_notes.is_empty());
    }

    #[cfg(feature = "piano")]
    #[test]
    fn score_skip_subtracts_skipped_time_from_logged_times() {
        use std::sync::Mutex;

        // The same score, skipped by 5s and played on a clock that is already
        // 5s into the app's run: the note that sat at 10s sounds now, so its
        // log line must report ~5s — not its original 10s.
        let score_text = "\
format bhs-score-v1
ticks_per_second 960
loop_ticks 19200
0 960 C4 80
9600 960 E4 80
";
        let mut score = BhsScore::parse(score_text).expect("the fixture should parse");
        score.skip_seconds(5).expect("the skip should succeed");

        let captured = Arc::new(Mutex::new(Vec::new()));
        let callback_events = Arc::clone(&captured);
        let config = BeamBuilder::new()
            .piano_log(PianoLog::All)
            .on_piano_event(move |event| callback_events.lock().unwrap().push(event))
            .into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);
        let start = Instant::now();
        app.piano_score = Some(
            PianoScorePlayback::from_score(score, start - Duration::from_secs(5))
                .expect("the skipped score should play"),
        );
        app.piano_started_at = start - Duration::from_secs(5);

        // The attack is due now (it was scheduled 5s after the shifted start)
        // and stamps itself at ~5s of app time; the release is not due yet.
        app.update_piano_score(start);
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 1, "only the attack is due yet");
        let attack = captured[0];
        assert_eq!(attack.note.midi_note, 64);
        drop(captured);

        // Log the pair through the same path the app uses: 5s on the log's
        // 1920 ticks/second grid is tick 9600 — the note's original 10s
        // (tick 19200) minus the skipped 5s.
        let mut release = attack;
        release.action = PianoAction::Release {
            velocity: f32::from(80u8) / 127.0,
            held_for: Duration::from_secs(1),
        };
        release.timestamp = attack.timestamp + Duration::from_secs(1);
        assert!(app.piano_log_line(&attack).is_none(), "the attack waits for its release");
        let line = app
            .piano_log_line(&release)
            .expect("the release should log a pair");
        let mut fields = line.split_whitespace();
        let start_tick: u64 = fields.next().expect("a start tick").parse().unwrap();
        let duration_ticks: u64 = fields.next().expect("a duration").parse().unwrap();
        assert_eq!(
            (fields.next(), fields.next(), fields.next()),
            (Some("E4"), Some("80"), Some("80"))
        );
        assert!((start_tick as i64 - 9600).abs() <= 192, "{line}");
        assert_eq!(duration_ticks, 1920);
    }

    #[test]
    fn fades_and_holds_ui_activity_colors() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert_eq!(APPEARANCE_INTERVAL, Duration::from_millis(200));
        assert_eq!(COLOR_FADE_DURATION, Duration::from_millis(400));
        assert_eq!(MIN_COLOR_STATE_DURATION, Duration::from_secs(1));
        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, None, start));
        let idle = node_style_colors(SunNodeState::Idle, 1, 4, None).body;
        let p1_step1 = node_style_colors(SunNodeState::Propagation1, 1, 4, None).body;
        assert_eq!(
            visual.style(4, None, false, start).body,
            idle,
            "the fade starts from the previous activity color"
        );
        assert_eq!(
            visual
                .style(4, None, false, start + COLOR_FADE_DURATION / 2)
                .body,
            lerp_color(idle, p1_step1, 0.5),
            "the color is blended halfway through the fade"
        );
        assert_eq!(
            visual
                .style(4, None, false, start + COLOR_FADE_DURATION)
                .body,
            p1_step1,
            "the fade reaches the new color after 400ms"
        );

        assert!(!visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            None,
            start + COLOR_FADE_DURATION
        ));
        assert!(!visual.advance(start + MIN_COLOR_STATE_DURATION - Duration::from_millis(1)));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
    }

    #[test]
    fn optimization_color_uses_frozen_state_captured_at_propagation_transition() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, Some(false), start));
        assert!(visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            Some(true),
            start + MIN_COLOR_STATE_DURATION
        ));
        assert_eq!(visual.optimization_frozen, Some(true));

        assert!(visual.observe(
            SunNodeState::Optimization,
            4,
            4,
            3,
            Some(false),
            start + MIN_COLOR_STATE_DURATION * 2
        ));

        let style = visual.style(
            4,
            Some(false),
            false,
            start + MIN_COLOR_STATE_DURATION * 2 + COLOR_FADE_DURATION,
        );
        assert_eq!(
            style.body,
            Color::BLACK,
            "optimization style keeps the frozen color captured at propagation1 -> propagation2"
        );
    }

    #[test]
    fn warp_nodes_use_black_body_and_white_text_with_phase_border() {
        let base = node_style_colors(SunNodeState::Propagation1, 2, 4, None);
        let warp = warp_node_style_colors(SunNodeState::Propagation1, 2, 4, None);
        assert_eq!(warp.body, Color::BLACK);
        assert_eq!(warp.text, Color::WHITE);
        assert_eq!(
            warp.border, base.border,
            "the phase border color is preserved on warp nodes"
        );

        let frozen_base = node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        let frozen_warp = warp_node_style_colors(SunNodeState::Optimization, 4, 4, Some(true));
        assert_eq!(frozen_warp.body, Color::BLACK);
        assert_eq!(frozen_warp.text, Color::WHITE);
        assert_eq!(frozen_warp.border, frozen_base.border);
    }

    #[test]
    fn queues_intermediate_phases_when_an_appearance_jumps() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert_eq!(visual.current.state, SunNodeState::Propagation1);
        assert_eq!(
            visual.pending,
            VecDeque::from([NodeProgress {
                state: SunNodeState::Propagation2,
                grad_step: 1,
            }])
        );
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
    }

    #[test]
    fn replays_an_epoch_when_snapshots_repeat_the_same_phase() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();

        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert!(visual.advance(start + MIN_COLOR_STATE_DURATION));
        assert_eq!(visual.current.state, SunNodeState::Propagation2);
        assert!(visual.pending.is_empty());

        assert!(visual.observe(
            SunNodeState::Propagation2,
            3,
            4,
            5,
            None,
            start + MIN_COLOR_STATE_DURATION * 2
        ));
        assert_eq!(visual.current.state, SunNodeState::Optimization);
        assert_eq!(
            visual.pending,
            VecDeque::from([
                NodeProgress {
                    state: SunNodeState::Propagation1,
                    grad_step: 1,
                },
                NodeProgress {
                    state: SunNodeState::Propagation2,
                    grad_step: 3,
                },
            ])
        );
    }

    #[test]
    fn bounds_pending_phases_when_epochs_outpace_the_animation() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();
        for (sequence, phase) in [
            (2, SunNodeState::Propagation2),
            (3, SunNodeState::Optimization),
            (4, SunNodeState::Propagation1),
            (5, SunNodeState::Propagation2),
            (6, SunNodeState::Optimization),
            (7, SunNodeState::Propagation1),
        ] {
            visual.observe(phase, 1, 4, sequence, None, start);
        }

        assert!(visual.pending.len() <= MAX_PENDING_PHASES);
    }

    #[test]
    fn throttles_color_ticks_while_waiting_for_next_transition() {
        let start = Instant::now();
        let mut visual = CellVisualState::default();
        assert!(visual.observe(SunNodeState::Propagation2, 1, 4, 2, None, start));
        assert_eq!(visual.current.state, SunNodeState::Propagation1);

        let fade_end = start + COLOR_FADE_DURATION;
        assert!(!visual.needs_color_frame(fade_end));
        assert!(visual.needs_transition_poll(fade_end));
    }

    #[test]
    fn extracts_static_cells_and_edges_from_black_hole_sun() {
        let model = BeamModel::build::<TestSun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0, 1]);
        assert_eq!(model.graph.edges, vec![(0, 1)]);
    }

    #[test]
    fn extracts_static_cells_and_edges_from_stateful_black_hole_sun() {
        let model = BeamModel::build::<TestSunWithCustomState>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0, 1]);
        assert_eq!(model.graph.edges, vec![(0, 1)]);
    }

    #[test]
    fn extracts_binary_cell_ports() {
        let model = BeamModel::build::<TestBinarySun>();

        assert!(model.errors.is_empty(), "{:?}", model.errors);
        assert_eq!(model.graph.nodes, vec![0]);
        assert_eq!(model.cells[0].ports, vec![0, 1]);
    }

    #[test]
    fn builds_live_model_from_serialized_sun_appearance() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 4,
            nodes: vec![
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Fusion".to_string(),
                    input_ports: vec![2, 3],
                    state: SunNodeState::Optimization,
                    state_sequence: 3,
                    grad_step: 4,
                },
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Propagation1,
                    state_sequence: 1,
                    grad_step: 1,
                },
            ],
            edges: vec![
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 3,
                },
                SunEdgeAppearance {
                    source: 0,
                    target: 2,
                    target_port: 2,
                },
            ],
        };
        let bytes = postcard::to_allocvec(&appearance).unwrap();
        let decoded = postcard::from_bytes::<SunAppearance>(&bytes).unwrap();
        let model =
            BeamModel::from_appearance(decoded, &HashMap::new(), &HashMap::new()).unwrap();

        assert_eq!(model.grad_steps, 4);
        assert_eq!(model.graph.nodes, vec![0, 2]);
        assert_eq!(
            model.graph.edges,
            vec![(0, 2)],
            "parallel destination-port edges collapse only for rendering"
        );
        assert_eq!(model.cells[0].animal_name, "Root");
        assert_eq!(model.cells[0].state, SunNodeState::Propagation1);
        assert_eq!(model.cells[0].state_sequence, 1);
        assert_eq!(model.cells[0].grad_step, 1);
        assert_eq!(model.cells[0].grad_steps, 4);
        assert_eq!(model.cells[1].state, SunNodeState::Optimization);
        assert_eq!(model.cells[1].state_sequence, 3);
        assert_eq!(model.cells[1].grad_step, 4);
    }

    #[test]
    fn maps_child_ray_frozen_state_into_live_cells() {
        let frozen_journey = Uuid::new_v4();
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: frozen_journey,
                warp_journey_id: Uuid::nil(),
                label: "Root".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Optimization,
                state_sequence: 4,
                grad_step: 1,
            }],
            edges: vec![],
        };
        let rays = HashMap::from([(frozen_journey, Ray { frozen: true })]);
        let model = BeamModel::from_appearance(appearance, &rays, &HashMap::new()).unwrap();
        assert_eq!(model.cells[0].frozen, Some(true));
    }

    #[test]
    fn rejects_appearance_edges_with_the_wrong_destination_port() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 0,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Root".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 1,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: Uuid::nil(),
                    label: "Sink".to_string(),
                    input_ports: vec![1],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![SunEdgeAppearance {
                source: 0,
                target: 1,
                target_port: 9,
            }],
        };

        let error = BeamModel::from_appearance(appearance, &HashMap::new(), &HashMap::new())
            .err()
            .expect("appearance should be rejected");
        assert!(error.contains("unowned input port 9"));
    }

    #[test]
    fn keeps_generics_in_type_labels() {
        assert_eq!(
            short_type_name::<GenericCell<String>>(),
            "GenericCell<String>"
        );
        assert_eq!(
            animal_label_key("my::module::RootAnimal<crate::leaf::Type, alloc::vec::Vec<u8>>"),
            "RootAnimal<Type, Vec<u8>>"
        );
    }

    #[test]
    fn keeps_generics_in_live_appearance_labels() {
        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id: Uuid::new_v4(),
                warp_journey_id: Uuid::nil(),
                label: "RootAnimal<Result<String, Vec<u8>>>".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        };

        let model = BeamModel::from_appearance(appearance, &HashMap::new(), &HashMap::new())
            .unwrap();
        assert_eq!(
            model.cells[0].animal_name,
            "RootAnimal<Result<String, Vec<u8>>>"
        );
    }

    #[test]
    fn labels_optimization_phase_as_potentiation() {
        assert_eq!(SunNodeState::Optimization.label(), "potentiation");
    }

    #[test]
    fn subpanel_phase_reports_potentiation_frozen_status() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let mut frozen_cell = CellDefinition::new::<TestCell>(1, vec![0], vec![]);
        frozen_cell.state = SunNodeState::Optimization;
        frozen_cell.frozen = Some(true);
        app.model.cells.push(frozen_cell);

        let mut open_cell = CellDefinition::new::<TestCell>(2, vec![0], vec![]);
        open_cell.state = SunNodeState::Optimization;
        open_cell.frozen = Some(false);
        app.model.cells.push(open_cell);

        assert_eq!(
            app.subpanel_phase(1).as_deref(),
            Some("potentiation [frozen]")
        );
        assert_eq!(
            app.subpanel_phase(2).as_deref(),
            Some("potentiation [optimizing]")
        );
    }

    #[test]
    fn subpanel_phase_uses_frozen_state_captured_at_propagation_transition() {
        let config = BeamBuilder::new().into_config();
        let (mut app, _task) = BeamApp::new(config, BeamModel::empty(), None);

        let mut cell = CellDefinition::new::<TestCell>(1, vec![0], vec![]);
        cell.state = SunNodeState::Optimization;
        cell.frozen = Some(false);
        app.model.cells.push(cell);

        let start = Instant::now();
        let mut visual = CellVisualState::default();
        assert!(visual.observe(SunNodeState::Propagation1, 1, 4, 1, Some(true), start));
        assert!(visual.observe(
            SunNodeState::Propagation2,
            1,
            4,
            2,
            Some(true),
            start + MIN_COLOR_STATE_DURATION
        ));
        assert!(visual.observe(
            SunNodeState::Optimization,
            4,
            4,
            3,
            Some(false),
            start + MIN_COLOR_STATE_DURATION * 2
        ));
        app.visuals.insert(1, visual);

        assert_eq!(
            app.subpanel_phase(1).as_deref(),
            Some("potentiation [frozen]"),
            "the subpanel matches the violet frozen style captured at propagation1 -> propagation2"
        );
    }
}

#[cfg(test)]
mod warp_fetch_diagnostics {
    use std::sync::Arc;

    use black_hole_flux::sun::{SunAppearance, SunNodeState};
    use uuid::Uuid;

    use crate::live::{fetch_warp_appearances, LiveConfig};

    use black_hole_flux::sun::SunNodeAppearance;

    fn warp_appearance_with(journey_id: Uuid) -> SunAppearance {
        SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![SunNodeAppearance {
                id: 0,
                journey_id,
                warp_journey_id: Uuid::nil(),
                label: "NestedRoot".to_string(),
                input_ports: vec![0],
                state: SunNodeState::Idle,
                state_sequence: 0,
                grad_step: 1,
            }],
            edges: vec![],
        }
    }

    #[test]
    fn reports_why_each_warp_cell_has_no_nested_model() {
        let missing = Uuid::new_v4();
        let foreign = Uuid::new_v4();
        let valid = Uuid::new_v4();
        // Bytes that postcard will not decode as a SunAppearance.
        let foreign_bytes = postcard::to_allocvec(&1u8).unwrap();

        let client = Arc::new(
            jungle_client::MockClient::builder()
                .on_flow_appearance(move |id| {
                    let foreign_bytes = foreign_bytes.clone();
                    async move {
                        Ok(match id {
                            i if i == missing => None,
                            i if i == foreign => Some(foreign_bytes.clone()),
                            i if i == valid => {
                                Some(postcard::to_allocvec(&warp_appearance_with(i)).unwrap())
                            }
                            _ => None,
                        })
                    }
                })
                .build(),
        );
        let live = LiveConfig {
            client,
            journey_id: Uuid::new_v4(),
        };

        let appearance = SunAppearance {
            finalized: true,
            grad_steps: 1,
            nodes: vec![
                SunNodeAppearance {
                    id: 1,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: missing,
                    label: "Warp<A, B>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 2,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: foreign,
                    label: "Warp<C, D>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
                SunNodeAppearance {
                    id: 3,
                    journey_id: Uuid::new_v4(),
                    warp_journey_id: valid,
                    label: "Warp<E, F>".to_string(),
                    input_ports: vec![0],
                    state: SunNodeState::Idle,
                    state_sequence: 0,
                    grad_step: 1,
                },
            ],
            edges: vec![],
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (models, diagnostics) = rt.block_on(fetch_warp_appearances(&live, &appearance));

        assert!(
            models.contains_key(&vec![3]),
            "a decodable nested sun becomes a model"
        );
        assert!(!models.contains_key(&vec![1]));
        assert!(!models.contains_key(&vec![2]));
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics[&vec![1]].contains("has not published an appearance yet"),
            "missing journey: {}",
            diagnostics[&vec![1]]
        );
        assert!(
            diagnostics[&vec![2]].contains("not a decodable Black Hole Sun"),
            "foreign appearance: {}",
            diagnostics[&vec![2]]
        );
    }
}

