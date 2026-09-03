//! Polling and decoding of live Jungle appearances for the main Sun and its
//! nested warp Suns.

use std::collections::HashMap;
use std::sync::Arc;

use black_hole_flux::topology::SunAppearance;
use black_hole_flux::Ray;
use iced::Task;
use jungle_sdk::JungleClient;
use uuid::Uuid;

use crate::app::Message;

#[derive(Clone)]
pub(crate) struct LiveConfig {
    pub(crate) client: Arc<dyn JungleClient>,
    pub(crate) journey_id: Uuid,
}

#[derive(Debug, Clone)]
pub(crate) struct LiveAppearanceSnapshot {
    pub(crate) appearance: SunAppearance,
    pub(crate) child_rays: HashMap<Uuid, Ray>,
    /// Nested Sun appearances for warp cells, keyed by the path of cell ids
    /// that locates the warp cell: `[7]` is a top-level warp, `[7, 3]` is
    /// warp cell 3 inside cell 7's nested sun.
    pub(crate) warp_appearances: HashMap<Vec<u32>, SunAppearance>,
    /// Why a warp cell's nested appearance could not be used yet, keyed by
    /// the same paths. Present whenever no usable model was produced.
    pub(crate) warp_diagnostics: HashMap<Vec<u32>, String>,
}

pub(crate) fn appearance_task(live: LiveConfig) -> Task<Message> {
    Task::perform(fetch_appearance(live), Message::AppearanceLoaded)
}

async fn fetch_appearance(live: LiveConfig) -> Result<Option<LiveAppearanceSnapshot>, String> {
    let bytes = live
        .client
        .animal_appearance(live.journey_id)
        .await
        .map_err(|error| error.to_string())?;
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let appearance = postcard::from_bytes::<SunAppearance>(&bytes)
        .map_err(|error| format!("could not decode Sun appearance: {error}"))?;
    let (warp_appearances, warp_diagnostics) = fetch_warp_appearances(&live, &appearance).await;
    // Rays are fetched for every node of the top-level sun and of each nested
    // warp sun so that merged subgraph cells keep their own frozen state.
    let mut appearances: Vec<&SunAppearance> = vec![&appearance];
    appearances.extend(warp_appearances.values());
    let child_rays = fetch_child_rays(&live, &appearances).await;
    Ok(Some(LiveAppearanceSnapshot {
        appearance,
        child_rays,
        warp_appearances,
        warp_diagnostics,
    }))
}

async fn fetch_child_rays(live: &LiveConfig, appearances: &[&SunAppearance]) -> HashMap<Uuid, Ray> {
    let mut rays = HashMap::new();
    for appearance in appearances {
        for node in &appearance.nodes {
            if rays.contains_key(&node.journey_id) {
                continue;
            }
            let maybe_bytes = live
                .client
                .animal_appearance(node.journey_id)
                .await
                .ok()
                .flatten();
            let Some(bytes) = maybe_bytes else {
                continue;
            };
            let Ok(ray) = postcard::from_bytes::<Ray>(&bytes) else {
                continue;
            };
            rays.insert(node.journey_id, ray);
        }
    }
    rays
}

/// How many levels of nested warps to fetch. Guards against cyclic warp
/// journeys, which would otherwise be fetched forever.
const MAX_WARP_DEPTH: usize = 16;

/// Fetches the nested Sun appearance behind every warp cell, recursing into
/// each nested sun so that warps within warps are discovered as well.
///
/// Returns the decodable appearances together with a per-cell diagnostic for
/// every warp cell that did not yield one, both keyed by the path of cell
/// ids from the top level, so the UI can explain why a warp node has not
/// expanded.
pub(crate) async fn fetch_warp_appearances(
    live: &LiveConfig,
    appearance: &SunAppearance,
) -> (HashMap<Vec<u32>, SunAppearance>, HashMap<Vec<u32>, String>) {
    let mut warp_appearances = HashMap::new();
    let mut warp_diagnostics = HashMap::new();
    // Depth-first worklist of (sun to scan, path prefix). Iterative so that
    // arbitrarily deep warp chains do not recurse.
    let mut stack: Vec<(SunAppearance, Vec<u32>)> = vec![(appearance.clone(), Vec::new())];
    while let Some((sun, prefix)) = stack.pop() {
        for node in &sun.nodes {
            if node.warp_journey_id.is_nil() {
                continue;
            }
            let mut path = prefix.clone();
            path.push(node.id);
            if path.len() > MAX_WARP_DEPTH {
                warp_diagnostics.insert(
                    path,
                    format!(
                        "warp journey {} is nested more than {MAX_WARP_DEPTH} levels deep; deeper subgraphs are not fetched",
                        node.warp_journey_id
                    ),
                );
                continue;
            }
            match fetch_sun_appearance(live, node.warp_journey_id).await {
                Ok(warp_appearance) => {
                    warp_appearances.insert(path.clone(), warp_appearance.clone());
                    stack.push((warp_appearance, path));
                }
                Err(diagnostic) => {
                    warp_diagnostics.insert(path, diagnostic);
                }
            }
        }
    }
    (warp_appearances, warp_diagnostics)
}

/// Fetches and decodes one journey's Sun appearance, with a diagnostic
/// message for every way it can fail.
async fn fetch_sun_appearance(
    live: &LiveConfig,
    journey_id: Uuid,
) -> Result<SunAppearance, String> {
    match live.client.animal_appearance(journey_id).await {
        Ok(Some(bytes)) => postcard::from_bytes::<SunAppearance>(&bytes).map_err(|error| {
            format!(
                "warp journey {journey_id} published an appearance that is not a decodable Black Hole Sun (SunAppearance): {error}"
            )
        }),
        Ok(None) => Err(format!(
            "warp journey {journey_id} has not published an appearance yet"
        )),
        Err(error) => Err(format!("fetching warp journey {journey_id} failed: {error}")),
    }
}
