//! Cell state shared across cell and nucleus actions.

pub mod action;
pub mod effect;

use serde::{Deserialize, Serialize};

pub use black_hole_spec::ObjectId;

/// State carried by a [`Cell`](crate::Cell) journey.
///
/// Animals that use [`Cell`](crate::Cell) as their Journey should use this as
/// their state type so the wait-for actions can read and write the next
/// transmission ID.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CellState {
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to download.
    pub recv_id: ObjectId,
    /// Void key of the next [`Transmission`](black_hole_spec::Transmission) to upload.
    pub send_id: ObjectId,
    /// Random seed passed to the perturb-up step each iteration.
    pub perturbation_seed: u64,
}
