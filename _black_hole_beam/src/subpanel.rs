//! The node-click subpanel overlay that shows a child flow's live journey.

use jungle_sdk::{Animal, JourneyAstSource};
use jungle_vision::{
    AnyAnimal, ClusterExpansionConfig, ClusterExpansionMode, DefaultTheme, EjectedViewer,
    JungleViewerBuilder,
};
use uuid::Uuid;

use crate::client::SharedJungleClient;

type JungleSubpanelViewer = EjectedViewer<DefaultTheme, AnyAnimal>;

#[derive(Clone)]
pub(crate) struct SubpanelConfig {
    pub(crate) animal_label: String,
    pub(crate) title: String,
    pub(crate) prefer_static: bool,
    pub(crate) build_static_viewer: fn() -> JungleSubpanelViewer,
    pub(crate) build_viewer: fn(SharedJungleClient, Uuid) -> JungleSubpanelViewer,
}

pub(crate) fn build_static_subpanel_viewer<A>() -> JungleSubpanelViewer
where
    A: Animal + 'static,
    A::Flow: JourneyAstSource,
{
    JungleViewerBuilder::new().eject_animal_with_theme::<A, _, AnyAnimal>(subpanel_theme())
}

pub(crate) fn build_subpanel_viewer<A>(
    client: SharedJungleClient,
    journey_id: Uuid,
) -> JungleSubpanelViewer
where
    A: Animal + 'static,
    A::Flow: JourneyAstSource,
{
    JungleViewerBuilder::new().eject_live_animal_with_theme::<A, SharedJungleClient, _, AnyAnimal>(
        client,
        journey_id,
        subpanel_theme(),
    )
}

fn subpanel_theme() -> DefaultTheme {
    DefaultTheme::default().with_cluster_expansion_config(ClusterExpansionConfig {
        while_clusters: ClusterExpansionMode::AlwaysExpanded,
        transparent_clusters: ClusterExpansionMode::AlwaysExpanded,
    })
}

pub(crate) struct SubpanelState {
    pub(crate) node_id: u32,
    pub(crate) title: String,
    pub(crate) journey_id: Uuid,
    pub(crate) viewer: JungleSubpanelViewer,
}
