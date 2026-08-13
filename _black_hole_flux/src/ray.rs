//! Lightweight child-workflow appearance for frozen-state visualization.

use serde::{Deserialize, Serialize};

/// Optional per-child appearance consumed by beam for optimize-state coloring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ray {
    pub frozen: bool,
}
